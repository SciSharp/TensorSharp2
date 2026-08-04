// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DeepSeek V4 (Flash) — driver for the native whole-model executor.
//
// The architecture (4-stream hyper-connections, shared single-KV-head
// attention with sinks and inverse-RoPE outputs, per-layer raw/CSA/HCA
// compressed attention with a lightning indexer, sqrt-softplus MoE with hash
// routing) is implemented natively in
// TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp as one ggml graph per
// ubatch, scheduled across every visible GPU (layer split) so models larger
// than a single GPU's VRAM are hosted across all of them.
//
// This class parses metadata/tokenizer from the (first) GGUF shard, feeds
// token batches to the native executor, and returns last-token logits. KV
// truncation is unsupported (compressed caches ratchet forward), so
// multi-turn prompts re-prefill; the engine handles that automatically.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : ModelBase, IBatchedPagedModel
    {
        private IntPtr _handle;
        private DeepSeek4CpuExecutor _cpuExec;
        private DeepSeek4CudaExecutor _cudaExec;
        private readonly object _sync = new object();

        public DeepSeek4Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            string draftModelPath = null)
            : base(ggufPath, NormalizeBackend(backend), 1, null)
        {
            string arch = _gguf.GetString("general.architecture") ?? "deepseek4";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            ParseTokenizer();

            int maxContext = ResolveConfiguredContextLength();
            // The GGUF advertises 1M context; cache rows scale with n_ctx, so keep a
            // practical default unless the operator asks for more via MAX_CONTEXT.
            string ctxEnv = Environment.GetEnvironmentVariable("MAX_CONTEXT");
            if (string.IsNullOrWhiteSpace(ctxEnv))
                maxContext = Math.Min(maxContext, 65536);
            _maxContextLength = maxContext;

            // GPU prefill chunks amortize the MoE expert-GEMM tile padding (256
            // experts x top-6 leaves ~nt/42 rows per expert, so per-chunk MoE
            // cost is nearly flat in nt): 1024 measured ~11-21% faster overall
            // prefill than 512 and also halves what a non-multiple tail chunk
            // costs relative to the whole prompt. The CPU executor stays at 512
            // (activation memory bound, no tile padding to amortize).
            int nUbatch = ParseEnvInt("TS_DSV4_UBATCH", backend == BackendType.Cpu ? 512 : 1024);

            if (backend == BackendType.Cuda)
            {
                // Direct-CUDA whole-model executor: quantized weights resident in
                // device memory, layer-split across the visible GPUs, driver-API
                // kernels only (no ggml).
                int nGpu = ParseEnvInt("TS_DSV4_NGPU", tpDegree > 1 ? tpDegree : 0); // 0 = all visible GPUs
                string dspark = ResolveDsparkPath(draftModelPath);
                Console.WriteLine($"Model: {arch} (direct-CUDA whole-model executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}" +
                    (dspark != null ? ", DSpark drafter" : string.Empty));
                _cudaExec = new DeepSeek4CudaExecutor(ggufPath, maxContext, nUbatch, nGpu, dspark, ResolveCpuMoeLayers());
            }
            else if (_backend == BackendType.Cpu)
            {
                // Pure C# whole-model executor: quantized weights served straight
                // from the memory-mapped GGUF shards, managed SIMD kernels only.
                WarnDsparkUnavailable(draftModelPath, backend);
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Environment.ProcessorCount);
                Console.WriteLine($"Model: {arch} (pure C# CPU executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}");
                _cpuExec = new DeepSeek4CpuExecutor(ggufPath, maxContext, nUbatch, nThreads, _allocator);
            }
            else
            {
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Math.Min(Environment.ProcessorCount, 32));
                int nGpu = ParseEnvInt("TS_DSV4_NGPU", tpDegree > 1 ? tpDegree : 0); // 0 = all visible GPUs
                string dspark = _backend == BackendType.GgmlCuda ? ResolveDsparkPath(draftModelPath) : null;
                if (dspark == null)
                    WarnDsparkUnavailable(draftModelPath, backend);

                Console.WriteLine($"Model: {arch} (native whole-model executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}" +
                    (dspark != null ? ", DSpark drafter" : string.Empty));

                int nCpuMoe = ResolveCpuMoeLayers();
                // `backend`, not `_backend`: the base ctor coerces everything to
                // GgmlCpu so no second GPU context is created, but the native
                // executor still has to pick its devices from the backend the
                // operator actually asked for.
                string backendName = BackendRegistryName(backend);
                _handle = dspark != null
                    ? GgmlDeepSeek4Native.LoadModelWithDspark(ggufPath, nGpu, maxContext, nUbatch, nThreads, dspark, nCpuMoe, backendName)
                    : GgmlDeepSeek4Native.LoadModel(ggufPath, nGpu, maxContext, nUbatch, nThreads, nCpuMoe, backendName);
                _nativeUBatch = nUbatch;
                _nativeDsparkBlock = _handle != IntPtr.Zero && dspark != null
                    ? GgmlDeepSeek4Native.DsparkBlockSize(_handle) : 0;
                if (_handle == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to load DeepSeek V4 model from {ggufPath} (see stderr for details).");
            }
        }

        /// <summary>
        /// Translate the process-wide <see cref="MoeCpuOffloadConfig"/> into the
        /// native loader's routed-expert offload policy.
        ///
        /// <para>Offload is OFF unless the operator asks for it, exactly as it is
        /// for every other architecture. DeepSeek V4 used to default to an
        /// automatic spill because a Q8_K_XL checkpoint (151 GiB) outweighs most
        /// hosts, but choosing that silently is the wrong trade on a host that
        /// *does* have the VRAM: it moves ~29 GiB of experts to the CPU and costs
        /// most of the decode throughput for no reason. A host that genuinely
        /// cannot fit the model now gets a load error naming the fewest layers
        /// that would (see the "[dsv4] does not fit" message), which is a better
        /// answer than a silent slowdown or an out-of-memory abort.</para>
        /// </summary>
        private static int ResolveCpuMoeLayers()
        {
            if (!MoeCpuOffloadConfig.IsExplicitlySet)
                return 0;
            return MoeCpuOffloadConfig.AllLayers ? int.MaxValue : MoeCpuOffloadConfig.CpuMoeLayers;
        }

        /// <summary>
        /// ggml backend registry name whose GPU devices the native executor
        /// should run on. GgmlOps links every backend it was built with and the
        /// loader enumerates devices in registration order, so without this a
        /// CUDA-capable box runs <c>--backend ggml_vulkan</c> on its CUDA
        /// devices and the Vulkan path is never exercised. Null = any GPU.
        /// </summary>
        private static string BackendRegistryName(BackendType backend) => backend switch
        {
            // GGML_CUDA_NAME is "CUDA" / "ROCm" / "MUSA" depending on how
            // ggml-cuda was built; the native side treats them as one family.
            BackendType.GgmlCuda => "CUDA",
            BackendType.GgmlVulkan => "Vulkan",
            BackendType.GgmlMetal => "Metal",
            _ => null,
        };

        /// <summary>Draft block size of the native (ggml) executor's drafter,
        /// 0 when none is loaded.</summary>
        private int _nativeDsparkBlock;

        /// <summary>Prefill micro-batch of the native executor.</summary>
        private int _nativeUBatch = 512;

        /// <summary>
        /// The DSpark drafter is implemented inside the direct-CUDA engine
        /// (Dsv4CudaEngine.Dspark.cs), on top of that engine's own kernels and
        /// cache rings. Every other executor serves plain decode, so say so
        /// instead of silently ignoring the drafter the operator asked for.
        /// </summary>
        private static void WarnDsparkUnavailable(string draftModelPath, BackendType backend)
        {
            if (ResolveDsparkPath(draftModelPath) == null)
                return;
            Console.Error.WriteLine(
                $"[dsv4] a DSpark drafter was configured but the {backend} backend has no speculative " +
                "path for DeepSeek V4 (it is implemented in the direct-CUDA and ggml_cuda engines); " +
                "decoding without it.");
        }

        private static BackendType NormalizeBackend(BackendType backend)
        {
            // BackendType.Cpu runs the pure C# executor. BackendType.Cuda runs
            // the direct-CUDA executor, which owns its own per-device contexts —
            // the base class only needs a lightweight CPU allocator, so it is
            // coerced to Cpu (the ctor dispatches on the ORIGINAL backend). The
            // other backends drive the native ggml executor; the managed side
            // only needs the GGML library loaded, so everything else is coerced
            // to GgmlCpu so no second GPU context is created.
            return backend switch
            {
                BackendType.Cpu => BackendType.Cpu,
                BackendType.Cuda => BackendType.Cpu,
                BackendType.GgmlCuda => BackendType.GgmlCuda,
                BackendType.GgmlVulkan => BackendType.GgmlVulkan,
                _ => BackendType.GgmlCpu,
            };
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        public override bool SupportsKVCacheTruncation => false;

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (_sync)
            {
                var logits = new float[Config.VocabSize];
                if (_cudaExec != null)
                {
                    _cudaExec.Forward(tokens, logits);
                }
                else if (_cpuExec != null)
                {
                    _cpuExec.Forward(tokens, logits);
                }
                else if (!GgmlDeepSeek4Native.Forward(_handle, tokens, logits))
                {
                    throw new InvalidOperationException("DeepSeek V4 native forward failed (see stderr).");
                }
                return logits;
            }
        }

        protected override void ResetKVCacheCore()
        {
            lock (_sync)
            {
                if (_cudaExec != null)
                    _cudaExec.Reset();
                else if (_cpuExec != null)
                    _cpuExec.Reset();
                else if (_handle != IntPtr.Zero)
                    GgmlDeepSeek4Native.Reset(_handle);
            }
        }

        public override void WarmUpKernels()
        {
            // One tiny forward allocates the scheduler's compute buffers so the
            // first user request does not pay the allocation cost.
            try
            {
                int bos = Tokenizer?.BosTokenId ?? 0;
                ForwardCore(new[] { bos });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[dsv4] warmup forward failed: {ex.Message}");
            }
            finally
            {
                ResetKVCacheCore();
            }
        }

        public override void Dispose()
        {
            lock (_sync)
            {
                if (_cudaExec != null)
                {
                    _cudaExec.Dispose();
                    _cudaExec = null;
                }
                if (_cpuExec != null)
                {
                    _cpuExec.Dispose();
                    _cpuExec = null;
                }
                if (_handle != IntPtr.Zero)
                {
                    GgmlDeepSeek4Native.Free(_handle);
                    _handle = IntPtr.Zero;
                }
            }
            base.Dispose();
        }
    }
}
