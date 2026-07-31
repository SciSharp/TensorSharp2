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

namespace TensorSharp.Models
{
    public class DeepSeek4Model : ModelBase
    {
        private IntPtr _handle;
        private DeepSeek4CpuExecutor _cpuExec;
        private readonly object _sync = new object();

        public DeepSeek4Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null)
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
            int nUbatch = ParseEnvInt("TS_DSV4_UBATCH", _backend == BackendType.Cpu ? 512 : 1024);

            if (_backend == BackendType.Cpu)
            {
                // Pure C# whole-model executor: quantized weights served straight
                // from the memory-mapped GGUF shards, managed SIMD kernels only.
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Environment.ProcessorCount);
                Console.WriteLine($"Model: {arch} (pure C# CPU executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}");
                _cpuExec = new DeepSeek4CpuExecutor(ggufPath, maxContext, nUbatch, nThreads);
            }
            else
            {
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Math.Min(Environment.ProcessorCount, 32));
                int nGpu = ParseEnvInt("TS_DSV4_NGPU", tpDegree > 1 ? tpDegree : 0); // 0 = all visible GPUs

                Console.WriteLine($"Model: {arch} (native whole-model executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}");

                _handle = GgmlDeepSeek4Native.LoadModel(ggufPath, nGpu, maxContext, nUbatch, nThreads);
                if (_handle == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to load DeepSeek V4 model from {ggufPath} (see stderr for details).");
            }
        }

        private static BackendType NormalizeBackend(BackendType backend)
        {
            // BackendType.Cpu runs the pure C# executor. The other backends drive
            // the native ggml executor; the managed side only needs the GGML
            // library loaded, so coerce everything else to GgmlCpu so no second
            // GPU context is created.
            return backend switch
            {
                BackendType.Cpu => BackendType.Cpu,
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
                if (_cpuExec != null)
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
                if (_cpuExec != null)
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
