// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) whole-model executor for the pure C# CPU backend.
//
// This is a from-scratch managed port of the native ggml executor in
// TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp (itself a port of
// llama.cpp's src/models/deepseek4.cpp), restricted to a single sequence:
//
//   * 4-stream hyper-connections (pre/post mixing + Sinkhorn-normalized 4x4
//     combination matrix per token).
//   * LoRA-factored Q, single shared 512-dim K(=V) head, per-head Q RMS norm,
//     attention sinks, RoPE -> attention -> inverse RoPE on the rope slice,
//     grouped LoRA output projection.
//   * Per-layer compress ratio 0 (raw SWA-128 attention only), 4 (CSA:
//     overlap-compressed rows + lightning-indexer top-k selection) or 128
//     (HCA: block-compressed rows, all visible).
//   * MoE: sqrt(softplus) routing in F32, +bias for selection, first
//     `hash_layer_count` layers route by token id through a tid2eid LUT,
//     swiglu clamp, weight normalization, x1.5 routed scale + shared expert.
//
// Weights stay quantized in the memory-mapped GGUF shards; matmuls run
// through ManagedQuantizedOps (integer dot kernels for Q8_0/Q6_K/...,
// dequant+float dot for IQ3_S/MXFP4/BF16). Because the executor is
// imperative, the graph-shape tricks the ggml port needs (masked scratch
// rows, persist plans, zero/-inf ring rows) reduce to plain loops over
// block boundaries.
//
// KV caches are stored as F32 but every cache commit is rounded through F16
// so results track the native executor / llama.cpp (both use F16 caches).
// The Hadamard rotation is skipped for the same reason as the native port:
// it only matters for quantized caches.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    internal sealed unsafe partial class DeepSeek4CpuExecutor : IDisposable
    {
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;
        private const int HC = 4;                       // hyper-connection streams
        private const int HcMixDim = (2 + HC) * HC;     // 24

        private struct WeightRef
        {
            public byte* Ptr;
            public GgmlTensorType Type;
            public int Ne0;      // row length (input dim)
            public int Ne1;      // rows (output dim)
            public int Ne2;      // experts (3D tensors), else 1
            public long RowBytes;
            public bool IsValid => Ptr != null;

            public byte* Row(long row) => Ptr + row * RowBytes;
            public byte* Expert(long e) => Ptr + e * Ne1 * RowBytes;
        }

        private sealed class Layer
        {
            public int Ratio;

            public float[] AttnNorm, QANorm, KvNorm, Sinks;
            public WeightRef WqA, WqB, Wkv, WoA, WoB;

            public WeightRef HcAttnFn, HcFfnFn;
            public float[] HcAttnScale, HcAttnBase, HcFfnScale, HcFfnBase;

            public WeightRef CompWkv, CompWgate;
            public float[] CompApe, CompNorm;

            public WeightRef IdxProj, IdxQB, IdxCompWkv, IdxCompWgate;
            public float[] IdxCompApe, IdxCompNorm;

            public float[] GateInp;          // router, dequantized to F32 [nExpert x nEmbd]
            public float[] ExpProbsBias;     // [nExpert] (null for hash layers)
            public int[] Tid2Eid;            // [nVocab x nExpertUsed] (hash layers only)
            public float[] FfnNorm;
            public WeightRef GateExps, DownExps, UpExps;
            public WeightRef GateShexp, DownShexp, UpShexp;

            // caches (native memory, F32 values rounded through F16 on commit)
            public float* RawK;              // [ringRaw x headDim]
            public float* CompK;             // CSA or HCA compressed K rows [rows x headDim]
            public float* LidK;              // lightning-indexer K rows [rows x idxHeadSize]
            public float* HistKv;            // compressor state ring [stateSize x coff*headDim]
            public float* HistScore;
            public float* LidHistKv;         // [2*CsaRatio x 2*idxHeadSize]
            public float* LidHistScore;
        }

        // hparams
        private int _nLayer, _nEmbd, _nHead, _nVocab, _headDim, _nRot, _qLoraRank, _oGroups, _oLoraRank, _nSwa;
        private float _rmsEps;
        private int _nExpert, _nExpertUsed, _nFfExp, _hashLayerCount;
        private float _expertWeightsScale;
        private bool _expertWeightsNorm;
        private float[] _swigluClampExp = Array.Empty<float>();
        private float[] _swigluClampShexp = Array.Empty<float>();
        private int _idxNHead, _idxHeadSize, _idxTopK;
        private int[] _compressRatios = Array.Empty<int>();
        private float _compressRopeBase = 10000f, _ropeFreqBase = 10000f;
        private float _yarnFreqScale = 1f, _yarnExtFactor;
        private float _yarnBetaFast = 32f, _yarnBetaSlow = 1f;
        private int _nCtxOrig;
        private int _hcSinkhornIters = 20;
        private float _hcEps = 1e-6f;
        private float _compCorr0, _compCorr1;    // YaRN correction dims (compress rope)

        // model-level weights
        private WeightRef _tokEmbd, _output, _hcHeadFn;
        private float[] _outputNorm, _hcHeadScale, _hcHeadBase;

        private Layer[] _layers;

        // geometry / state
        private int _nCtx, _nUbatch;
        private int _ringRaw, _compRowsCsa, _compRowsHca;
        private int _nPast;

        private readonly List<GgufFile> _shards = new List<GgufFile>();
        private readonly List<string> _shardPaths = new List<string>();
        private readonly Dictionary<string, (GgufFile File, GgufTensorInfo Info)> _tensorMap
            = new Dictionary<string, (GgufFile, GgufTensorInfo)>(StringComparer.Ordinal);
        private readonly List<IntPtr> _ownedBuffers = new List<IntPtr>();

        private readonly ParallelOptions _po;
        private readonly Dsv4SpinPool _pool;
        private readonly int _perf;
        private bool _useMmap = true;
        private Dictionary<string, IntPtr> _bufferedTensors;

        /// <summary>
        /// Hot-path parallel-for. Routes through the persistent spin pool (region
        /// latency in the microseconds) unless TS_DSV4_SPINPOOL=0 falls back to
        /// TPL Parallel.For.
        /// </summary>
        private void PFor(int n, Action<int> body)
        {
            if (_pool != null)
                _pool.For(n, body);
            else
                Parallel.For(0, n, _po, body);
        }

        // stage timing (ticks), printed when TS_DSV4_PERF >= 2
        private readonly long[] _stageTicks = new long[12];
        private static readonly string[] StageNames =
        {
            "embed", "hc", "attnproj", "comp", "idx", "attncore", "outproj", "router", "experts", "shexp", "lmhead", "rope"
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Tick(int stage, long t0) => _stageTicks[stage] += Stopwatch.GetTimestamp() - t0;

        public int VocabSize => _nVocab;
        public int ContextSize => _nCtx;
        public int NPast => _nPast;

        public DeepSeek4CpuExecutor(string ggufPath, int maxContext, int nUbatch, int nThreads)
        {
            var sw = Stopwatch.StartNew();
            int dop = nThreads > 0 ? nThreads : Environment.ProcessorCount;
            _po = new ParallelOptions { MaxDegreeOfParallelism = dop };
            _perf = ParseEnvInt("TS_DSV4_PERF", 0);

            // The forward pass issues thousands of short Parallel.For regions per
            // token; without a raised floor the thread pool ramps workers up too
            // slowly for any of them to reach full width.
            ThreadPool.GetMinThreads(out int minWorker, out int minIo);
            if (minWorker < dop)
                ThreadPool.SetMinThreads(dop, minIo);

            _useMmap = ParseEnvInt("TS_DSV4_MMAP", 1) != 0;
            // Persistent spinning workers only pay off when the host gives this
            // process dedicated cores; under a shared CPU quota the spinners eat
            // the budget the compute needs. Off by default, opt in via env.
            if (ParseEnvInt("TS_DSV4_SPINPOOL", 0) != 0)
                _pool = new Dsv4SpinPool(dop);

            OpenShards(ggufPath);
            ParseHparams();

            var shardSelected = new bool[_shards.Count];
            bool anySelected = false;
            if (!_useMmap)
            {
                for (int i = 0; i < shardSelected.Length; i++) shardSelected[i] = true;
                anySelected = true;
            }
            else
            {
                string bufShards = Environment.GetEnvironmentVariable("TS_DSV4_BUFFER_SHARDS");
                if (!string.IsNullOrWhiteSpace(bufShards))
                {
                    foreach (string part in bufShards.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
                    {
                        if (int.TryParse(part, out int idx) && idx >= 1 && idx <= shardSelected.Length)
                        {
                            shardSelected[idx - 1] = true;
                            anySelected = true;
                        }
                    }
                }
            }
            if (anySelected)
                MaterializeTensors(shardSelected);

            _nCtx = maxContext > 0 ? maxContext : 16384;
            _nUbatch = nUbatch > 0 ? nUbatch : 512;
            _ringRaw = Pad(_nSwa + _nUbatch, 256);
            _compRowsCsa = _nCtx / CsaRatio + 1;
            _compRowsHca = _nCtx / HcaRatio + 1;

            LoadWeights();
            AllocateCaches();
            AllocateScratch();

            // Pin the mapped shards in RAM (best effort). Without this, weights
            // served from a FUSE/network filesystem are re-fetched on every
            // forward pass, which caps decode at the network's throughput.
            if (ParseEnvInt("TS_DSV4_MLOCK", 1) != 0)
            {
                var lockSw = Stopwatch.StartNew();
                int locked = 0;
                Parallel.ForEach(_shards, shard => { if (shard.TryLockMappedRegion()) Interlocked.Increment(ref locked); });
                if (locked > 0 || lockSw.Elapsed.TotalSeconds > 1)
                    Console.Error.WriteLine($"[dsv4-cpu] mlock: pinned {locked}/{_shards.Count} shard(s) in {lockSw.Elapsed.TotalSeconds:F1}s");
            }

            double gib = 0;
            foreach (var s in _shards)
                foreach (var kv in s.Tensors)
                    gib += s.GetTensorByteCount(kv.Value);
            Console.Error.WriteLine($"[dsv4-cpu] mapped {gib / (1024.0 * 1024.0 * 1024.0):F1} GiB across {_shards.Count} shard(s) in {sw.Elapsed.TotalSeconds:F1}s " +
                $"(n_ctx={_nCtx}, ubatch={_nUbatch}, threads={_po.MaxDegreeOfParallelism})");
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        private static int Pad(int v, int p) => (v + p - 1) / p * p;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float F16(float v) => (float)(System.Half)v;

        // -------------------------------------------------------------------
        // Loading
        // -------------------------------------------------------------------

        private void OpenShards(string firstPath)
        {
            var first = new GgufFile(firstPath);
            _shards.Add(first);
            _shardPaths.Add(firstPath);

            int splitCount = (int)first.GetUint32("split.count", 1);
            if (splitCount > 1)
            {
                const string marker = "-00001-of-";
                int pos = firstPath.IndexOf(marker, StringComparison.Ordinal);
                if (pos >= 0)
                {
                    for (int i = 2; i <= splitCount; i++)
                    {
                        string path = firstPath.Substring(0, pos) + $"-{i:D5}-of-" + firstPath.Substring(pos + marker.Length);
                        _shards.Add(new GgufFile(path));
                        _shardPaths.Add(path);
                    }
                }
            }

            foreach (var shard in _shards)
                foreach (var kv in shard.Tensors)
                    _tensorMap[kv.Key] = (shard, kv.Value);
        }

        private void ParseHparams()
        {
            GgufFile g = _shards[0];
            const string a = "deepseek4";
            _nLayer = (int)g.GetUint32($"{a}.block_count");
            _nEmbd = (int)g.GetUint32($"{a}.embedding_length");
            _nHead = (int)g.GetUint32($"{a}.attention.head_count");
            _headDim = (int)g.GetUint32($"{a}.attention.key_length");
            _nRot = (int)g.GetUint32($"{a}.rope.dimension_count");
            _qLoraRank = (int)g.GetUint32($"{a}.attention.q_lora_rank");
            _oGroups = (int)g.GetUint32($"{a}.attention.output_group_count");
            _oLoraRank = (int)g.GetUint32($"{a}.attention.output_lora_rank");
            _nSwa = (int)g.GetUint32($"{a}.attention.sliding_window");
            _rmsEps = g.GetFloat32($"{a}.attention.layer_norm_rms_epsilon", 1e-6f);
            _nExpert = (int)g.GetUint32($"{a}.expert_count");
            _nExpertUsed = (int)g.GetUint32($"{a}.expert_used_count");
            _nFfExp = (int)g.GetUint32($"{a}.expert_feed_forward_length");
            _expertWeightsScale = g.GetFloat32($"{a}.expert_weights_scale", 1.0f);
            _expertWeightsNorm = g.GetBool($"{a}.expert_weights_norm");
            _hashLayerCount = (int)g.GetUint32($"{a}.hash_layer_count");
            _idxNHead = (int)g.GetUint32($"{a}.attention.indexer.head_count");
            _idxHeadSize = (int)g.GetUint32($"{a}.attention.indexer.key_length");
            _idxTopK = (int)g.GetUint32($"{a}.attention.indexer.top_k");
            _compressRatios = g.GetInt32Array($"{a}.attention.compress_ratios") ?? Array.Empty<int>();
            _compressRopeBase = g.GetFloat32($"{a}.attention.compress_rope_freq_base", 10000f);
            _hcSinkhornIters = (int)g.GetUint32($"{a}.hyper_connection.sinkhorn_iterations", 20);
            _hcEps = g.GetFloat32($"{a}.hyper_connection.epsilon", 1e-6f);
            _swigluClampExp = g.GetFloatArray($"{a}.swiglu_clamp_exp") ?? Array.Empty<float>();
            _swigluClampShexp = g.GetFloatArray($"{a}.swiglu_clamp_shexp") ?? _swigluClampExp;
            _ropeFreqBase = g.GetFloat32($"{a}.rope.freq_base", 10000f);

            float yarnFactor = g.GetFloat32($"{a}.rope.scaling.factor", 0f);
            if (yarnFactor > 0f)
            {
                _yarnFreqScale = 1.0f / yarnFactor;
                _yarnExtFactor = 1.0f;
            }
            _nCtxOrig = (int)g.GetUint32($"{a}.rope.scaling.original_context_length");
            _yarnBetaFast = g.GetFloat32($"{a}.rope.scaling.yarn_beta_fast", 32f);
            _yarnBetaSlow = g.GetFloat32($"{a}.rope.scaling.yarn_beta_slow", 1f);

            int hc = (int)g.GetUint32($"{a}.hyper_connection.count", HC);
            if (hc != HC)
                throw new NotSupportedException($"DeepSeek4 CPU executor supports hyper_connection.count == {HC}, got {hc}.");
            if (_nLayer <= 0 || _compressRatios.Length < _nLayer)
                throw new InvalidOperationException("Missing or invalid deepseek4 GGUF metadata.");

            // YaRN correction dims for the compress-rope parameter set.
            _compCorr0 = MathF.Max(0f, MathF.Floor(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaFast, _compressRopeBase)));
            _compCorr1 = MathF.Min(_nRot - 1, MathF.Ceiling(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaSlow, _compressRopeBase)));
        }

        private static float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(freqBase));
        }

        /// <summary>
        /// Copies the selected shards' tensors into anonymous RAM up front using
        /// parallel positional reads (thread-safe RandomAccess on one handle per
        /// shard, which keeps a network filesystem's readahead streaming).
        /// TS_DSV4_MMAP=0 buffers every shard; TS_DSV4_BUFFER_SHARDS=3,4 buffers
        /// only those (1-based) shards. Makes steady-state inference independent
        /// of the filesystem's page-cache behaviour.
        /// </summary>
        private void MaterializeTensors(bool[] shardSelected)
        {
            var sw = Stopwatch.StartNew();
            _bufferedTensors = new Dictionary<string, IntPtr>(StringComparer.Ordinal);

            var handles = new Microsoft.Win32.SafeHandles.SafeFileHandle[_shardPaths.Count];
            var shardIndexOf = new Dictionary<GgufFile, int>();
            for (int s = 0; s < _shards.Count; s++)
            {
                shardIndexOf[_shards[s]] = s;
                if (shardSelected[s])
                    handles[s] = File.OpenHandle(_shardPaths[s], FileMode.Open, FileAccess.Read, FileShare.Read);
            }

            var names = new List<string>();
            foreach (var kv in _tensorMap)
            {
                if (shardSelected[shardIndexOf[kv.Value.File]])
                    names.Add(kv.Key);
            }

            var slots = new IntPtr[names.Count];
            long totalBytes = 0;
            Parallel.For(0, names.Count, _po, i =>
            {
                var entry = _tensorMap[names[i]];
                int s = shardIndexOf[entry.File];
                long bytes = entry.File.GetTensorByteCount(entry.Info);
                long fileOff = entry.File.DataOffset + (long)entry.Info.Offset;
                IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
                byte* dst = (byte*)buf;
                long done = 0;
                while (done < bytes)
                {
                    int want = (int)Math.Min(32 << 20, bytes - done);
                    int got = RandomAccess.Read(handles[s], new Span<byte>(dst + done, want), fileOff + done);
                    if (got <= 0)
                        throw new IOException($"[dsv4-cpu] short read materializing {names[i]}");
                    done += got;
                }
                slots[i] = buf;
                Interlocked.Add(ref totalBytes, bytes);
            });

            foreach (var h in handles)
                h?.Dispose();

            for (int i = 0; i < names.Count; i++)
            {
                _bufferedTensors[names[i]] = slots[i];
                _ownedBuffers.Add(slots[i]);
            }
            Console.Error.WriteLine($"[dsv4-cpu] buffered {totalBytes / (1024.0 * 1024 * 1024):F1} GiB of weights into RAM in {sw.Elapsed.TotalSeconds:F1}s " +
                $"({totalBytes / (1024.0 * 1024 * 1024) / Math.Max(0.001, sw.Elapsed.TotalSeconds):F2} GiB/s)");
        }

        private WeightRef GetW(string name, bool required = true)
        {
            if (!_tensorMap.TryGetValue(name, out var entry))
            {
                if (required)
                    throw new InvalidOperationException($"[dsv4-cpu] missing tensor: {name}");
                return default;
            }

            GgufTensorInfo info = entry.Info;
            byte* ptr;
            if (_bufferedTensors != null && _bufferedTensors.TryGetValue(name, out IntPtr buffered))
            {
                ptr = (byte*)buffered;
            }
            else if (entry.File.TryGetTensorDataPointer(info, out IntPtr mapped))
            {
                ptr = (byte*)mapped;
            }
            else
            {
                long bytes = entry.File.GetTensorByteCount(info);
                IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
                _ownedBuffers.Add(buf);
                entry.File.ReadTensorDataToNative(info, buf, bytes);
                ptr = (byte*)buf;
            }

            var w = new WeightRef
            {
                Ptr = ptr,
                Type = info.Type,
                Ne0 = (int)info.Shape[0],
                Ne1 = info.Shape.Length > 1 ? (int)info.Shape[1] : 1,
                Ne2 = info.Shape.Length > 2 ? (int)info.Shape[2] : 1,
            };
            w.RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, w.Ne0);
            return w;
        }

        /// <summary>Loads any-typed small tensor fully dequantized into a managed F32 array.</summary>
        private float[] GetF32(string name, bool required = true)
        {
            var w = GetW(name, required);
            if (!w.IsValid)
                return null;
            long n = (long)w.Ne0 * w.Ne1 * w.Ne2;
            var arr = new float[n];
            ManagedQuantizedOps.DequantizeToFloat32((int)w.Type, (IntPtr)w.Ptr, arr, 0, n);
            return arr;
        }

        private int[] GetI32(string name)
        {
            var w = GetW(name);
            if (w.Type != GgmlTensorType.I32)
                throw new InvalidOperationException($"[dsv4-cpu] {name}: expected I32, got {w.Type}");
            long n = (long)w.Ne0 * w.Ne1;
            var arr = new int[n];
            Marshal.Copy((IntPtr)w.Ptr, arr, 0, (int)n);
            return arr;
        }

        private void LoadWeights()
        {
            _tokEmbd = GetW("token_embd.weight");
            _nVocab = _tokEmbd.Ne1;
            _output = GetW("output.weight");
            _outputNorm = GetF32("output_norm.weight");
            _hcHeadFn = GetW("output_hc_fn.weight");
            _hcHeadScale = GetF32("output_hc_scale.weight");
            _hcHeadBase = GetF32("output_hc_base.weight");

            _layers = new Layer[_nLayer];
            for (int il = 0; il < _nLayer; il++)
            {
                var L = new Layer { Ratio = _compressRatios[il] };
                _layers[il] = L;
                string p = $"blk.{il}.";

                L.AttnNorm = GetF32(p + "attn_norm.weight");
                L.Sinks = GetF32(p + "attn_sinks.weight");
                L.WqA = GetW(p + "attn_q_a.weight");
                L.QANorm = GetF32(p + "attn_q_a_norm.weight");
                L.WqB = GetW(p + "attn_q_b.weight");
                L.Wkv = GetW(p + "attn_kv.weight");
                L.KvNorm = GetF32(p + "attn_kv_a_norm.weight");
                L.WoA = GetW(p + "attn_output_a.weight");
                L.WoB = GetW(p + "attn_output_b.weight");

                L.HcAttnFn = GetW(p + "hc_attn_fn.weight");
                L.HcAttnScale = GetF32(p + "hc_attn_scale.weight");
                L.HcAttnBase = GetF32(p + "hc_attn_base.weight");
                L.HcFfnFn = GetW(p + "hc_ffn_fn.weight");
                L.HcFfnScale = GetF32(p + "hc_ffn_scale.weight");
                L.HcFfnBase = GetF32(p + "hc_ffn_base.weight");

                if (L.Ratio != 0)
                {
                    L.CompWkv = GetW(p + "attn_compressor_kv.weight");
                    L.CompWgate = GetW(p + "attn_compressor_gate.weight");
                    L.CompApe = GetF32(p + "attn_compressor_ape.weight");
                    L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                    if (L.Ratio == CsaRatio)
                    {
                        L.IdxProj = GetW(p + "indexer.proj.weight");
                        L.IdxQB = GetW(p + "indexer.attn_q_b.weight");
                        L.IdxCompWkv = GetW(p + "indexer_compressor_kv.weight");
                        L.IdxCompWgate = GetW(p + "indexer_compressor_gate.weight");
                        L.IdxCompApe = GetF32(p + "indexer_compressor_ape.weight");
                        L.IdxCompNorm = GetF32(p + "indexer_compressor_norm.weight");
                    }
                }

                L.GateInp = GetF32(p + "ffn_gate_inp.weight");
                if (il < _hashLayerCount)
                    L.Tid2Eid = GetI32(p + "ffn_gate_tid2eid.weight");
                else
                    L.ExpProbsBias = GetF32(p + "exp_probs_b.bias");
                L.FfnNorm = GetF32(p + "ffn_norm.weight");
                L.GateExps = GetW(p + "ffn_gate_exps.weight");
                L.DownExps = GetW(p + "ffn_down_exps.weight");
                L.UpExps = GetW(p + "ffn_up_exps.weight");
                L.GateShexp = GetW(p + "ffn_gate_shexp.weight");
                L.DownShexp = GetW(p + "ffn_down_shexp.weight");
                L.UpShexp = GetW(p + "ffn_up_shexp.weight");
            }
        }

        private float* AllocF32(long count)
        {
            float* p = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
            _nativeAllocs.Add((IntPtr)p);
            NativeMemory.Clear(p, (nuint)(count * sizeof(float)));
            return p;
        }

        private readonly List<IntPtr> _nativeAllocs = new List<IntPtr>();

        private void AllocateCaches()
        {
            foreach (var L in _layers)
            {
                L.RawK = AllocF32((long)_ringRaw * _headDim);
                if (L.Ratio == CsaRatio)
                {
                    L.CompK = AllocF32((long)_compRowsCsa * _headDim);
                    L.LidK = AllocF32((long)_compRowsCsa * _idxHeadSize);
                    L.HistKv = AllocF32(2L * CsaRatio * 2 * _headDim);
                    L.HistScore = AllocF32(2L * CsaRatio * 2 * _headDim);
                    L.LidHistKv = AllocF32(2L * CsaRatio * 2 * _idxHeadSize);
                    L.LidHistScore = AllocF32(2L * CsaRatio * 2 * _idxHeadSize);
                }
                else if (L.Ratio == HcaRatio)
                {
                    L.CompK = AllocF32((long)_compRowsHca * _headDim);
                    L.HistKv = AllocF32((long)HcaRatio * _headDim);
                    L.HistScore = AllocF32((long)HcaRatio * _headDim);
                }
            }
        }

        // ubatch scratch buffers
        private float* _xs;          // [nt x HC x nEmbd] hidden streams
        private float* _cur;         // [nt x nEmbd]
        private float* _pre;         // [nt x HC]
        private float* _post;        // [nt x HC]
        private float* _comb;        // [nt x HC x HC]
        private float* _mixes;       // [nt x HcMixDim]
        private float* _qr;          // [nt x qLoraRank]
        private float* _q;           // [nt x nHead x headDim]
        private float* _kvRow;       // [nt x headDim]
        private float* _stKv;        // [nt x 2*headDim]
        private float* _stScore;     // [nt x 2*headDim]
        private float* _lidStKv;     // [nt x 2*idxHeadSize]
        private float* _lidStScore;  // [nt x 2*idxHeadSize]
        private float* _iq;          // [nt x idxNHead x idxHeadSize]
        private float* _iw;          // [nt x idxNHead]
        private float* _idxScores;   // [nt x compRowsCsa]
        private int* _topK;          // [nt x idxTopK]
        private int* _topKCount;     // [nt]
        private float* _attnO;       // [nt x nHead x headDim]
        private float* _oG;          // [nt x oGroups*oLoraRank]
        private float* _attnOut;     // [nt x nEmbd]
        private float* _ffnOut;      // [nt x nEmbd]
        private float* _routerLogits;// [nt x nExpert]
        private float* _routerProbs; // [nt x nExpert]
        private int* _selExperts;    // [nt x nExpertUsed]
        private float* _selWeights;  // [nt x nExpertUsed]
        private float* _ropeRawCache;  // [nt x nRot] interleaved cos/sin (raw params)
        private float* _ropeCompCache; // [nt x nRot] interleaved cos/sin (compress params)
        private byte* _actQuant;     // quantized activations scratch [nt x maxActRowBytes]
        private float* _expertPack;  // gather buffer for MoE prefill [nt x nEmbd]
        private float* _expertGate;  // [nt x nFfExp]
        private float* _expertUp;    // [nt x nFfExp]
        private float* _expertDown;  // [nt x nEmbd]
        private float* _shGate;      // [nt x shexp ff]
        private float* _shUp;
        private float* _shDown;      // [nt x nEmbd]
        private int _maxActRowBytes;
        private int* _expCount;      // [nExpert] fused-prefill MoE grouping
        private int* _expOffset;     // [nExpert]
        private int* _expCursor;     // [nExpert]
        private int* _rowOfSlot;     // [nt x nExpertUsed]
        private int* _slotToken;     // [nt x nExpertUsed]
        private int* _slotGroupSize; // [nt x nExpertUsed]

        // Expert groups at least this large use dequantize-row-once + float dots
        // in the fused prefill MoE; smaller groups use the integer dot path.
        private const int DequantGroupMin = 4;

        private void AllocateScratch()
        {
            int nt = _nUbatch;
            _xs = AllocF32((long)nt * HC * _nEmbd);
            _cur = AllocF32((long)nt * _nEmbd);
            _pre = AllocF32((long)nt * HC);
            _post = AllocF32((long)nt * HC);
            _comb = AllocF32((long)nt * HC * HC);
            _mixes = AllocF32((long)nt * HcMixDim);
            _qr = AllocF32((long)nt * _qLoraRank);
            _q = AllocF32((long)nt * _nHead * _headDim);
            _kvRow = AllocF32((long)nt * _headDim);
            _stKv = AllocF32((long)nt * 2 * _headDim);
            _stScore = AllocF32((long)nt * 2 * _headDim);
            _lidStKv = AllocF32((long)nt * 2 * _idxHeadSize);
            _lidStScore = AllocF32((long)nt * 2 * _idxHeadSize);
            _iq = AllocF32((long)nt * _idxNHead * _idxHeadSize);
            _iw = AllocF32((long)nt * _idxNHead);
            _idxScores = AllocF32((long)nt * _compRowsCsa);
            _topK = (int*)AllocF32((long)nt * _idxTopK);
            _topKCount = (int*)AllocF32(nt);
            _attnO = AllocF32((long)nt * _nHead * _headDim);
            _oG = AllocF32((long)nt * _oGroups * _oLoraRank);
            _attnOut = AllocF32((long)nt * _nEmbd);
            _ffnOut = AllocF32((long)nt * _nEmbd);
            _routerLogits = AllocF32((long)nt * _nExpert);
            _routerProbs = AllocF32((long)nt * _nExpert);
            _selExperts = (int*)AllocF32((long)nt * _nExpertUsed);
            _selWeights = AllocF32((long)nt * _nExpertUsed);
            _ropeRawCache = AllocF32((long)nt * _nRot);
            _ropeCompCache = AllocF32((long)nt * _nRot);
            _hcPostTmp = AllocF32((long)nt * HC * _nEmbd);
            // sized nt * nExpertUsed: a token may select the same expert more than
            // once via the hash-router LUT, so one expert's group can exceed nt rows
            _expertPack = AllocF32((long)nt * _nExpertUsed * _nEmbd);
            _expertGate = AllocF32((long)nt * _nExpertUsed * _nFfExp);
            _expertUp = AllocF32((long)nt * _nExpertUsed * _nFfExp);
            _expertDown = AllocF32((long)nt * _nExpertUsed * _nEmbd);
            int shFf = _layers[0].UpShexp.IsValid ? _layers[0].UpShexp.Ne1 : _nFfExp;
            _shGate = AllocF32((long)nt * shFf);
            _shUp = AllocF32((long)nt * shFf);
            _shDown = AllocF32((long)nt * _nEmbd);
            _expCount = (int*)AllocF32(_nExpert);
            _expOffset = (int*)AllocF32(_nExpert);
            _expCursor = (int*)AllocF32(_nExpert);
            _rowOfSlot = (int*)AllocF32((long)nt * _nExpertUsed);
            _slotToken = (int*)AllocF32((long)nt * _nExpertUsed);
            _slotGroupSize = (int*)AllocF32((long)nt * _nExpertUsed);

            // activation quantization scratch: largest quant-weight row length is
            // oGroups*oLoraRank (wo_b input); round up generously. Capacity is
            // nt * max(8, oGroups) rows so the fused grouped-out-projection and
            // fused decode MoE can hold per-group / per-expert activations.
            int maxNe0 = Math.Max(Math.Max(_nEmbd, _oGroups * _oLoraRank), Math.Max(_qLoraRank, _nFfExp));
            _maxActRowBytes = (maxNe0 / 256 + 1) * 300;
            _actQuant = (byte*)NativeMemory.AlignedAlloc((nuint)((long)nt * Math.Max(8, _oGroups) * _maxActRowBytes), 64);
            _nativeAllocs.Add((IntPtr)_actQuant);
        }

        public void Reset()
        {
            _nPast = 0;
            foreach (var L in _layers)
            {
                NativeMemory.Clear(L.RawK, (nuint)((long)_ringRaw * _headDim * sizeof(float)));
                if (L.CompK != null)
                {
                    long rows = L.Ratio == CsaRatio ? _compRowsCsa : _compRowsHca;
                    NativeMemory.Clear(L.CompK, (nuint)(rows * _headDim * sizeof(float)));
                }
                if (L.LidK != null)
                    NativeMemory.Clear(L.LidK, (nuint)((long)_compRowsCsa * _idxHeadSize * sizeof(float)));
                if (L.HistKv != null)
                {
                    long n = L.Ratio == CsaRatio ? 2L * CsaRatio * 2 * _headDim : (long)HcaRatio * _headDim;
                    NativeMemory.Clear(L.HistKv, (nuint)(n * sizeof(float)));
                    NativeMemory.Clear(L.HistScore, (nuint)(n * sizeof(float)));
                }
                if (L.LidHistKv != null)
                {
                    long n = 2L * CsaRatio * 2 * _idxHeadSize;
                    NativeMemory.Clear(L.LidHistKv, (nuint)(n * sizeof(float)));
                    NativeMemory.Clear(L.LidHistScore, (nuint)(n * sizeof(float)));
                }
            }
        }

        // -------------------------------------------------------------------
        // Forward
        // -------------------------------------------------------------------

        public void Forward(int[] tokens, float[] logitsOut)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            if (_nPast + tokens.Length > _nCtx)
                throw new InvalidOperationException($"[dsv4-cpu] context overflow: n_past={_nPast} + {tokens.Length} > n_ctx={_nCtx}");

            var sw = Stopwatch.StartNew();
            int done = 0;
            while (done < tokens.Length)
            {
                int nt = Math.Min(_nUbatch, tokens.Length - done);
                bool last = done + nt == tokens.Length;
                ForwardUbatch(tokens, done, nt, _nPast, last ? logitsOut : null);
                _nPast += nt;
                done += nt;
            }
            if (_perf > 0)
            {
                double s = sw.Elapsed.TotalSeconds;
                Console.Error.WriteLine($"[dsv4-cpu] forward {tokens.Length} tokens in {s:F3}s ({tokens.Length / s:F1} tok/s)");
                if (_perf >= 2)
                {
                    var sb = new System.Text.StringBuilder("[dsv4-cpu]   stages:");
                    for (int i = 0; i < _stageTicks.Length; i++)
                    {
                        if (_stageTicks[i] == 0) continue;
                        sb.Append($" {StageNames[i]}={_stageTicks[i] * 1000.0 / Stopwatch.Frequency:F0}ms");
                        _stageTicks[i] = 0;
                    }
                    Console.Error.WriteLine(sb.ToString());
                }
            }
        }

        private void ForwardUbatch(int[] tokens, int tokOff, int nt, int p0, float[] logitsOut)
        {
            int E = _nEmbd;
            long t0 = Stopwatch.GetTimestamp();

            // token embedding, replicated over the HC streams
            PFor(nt, t =>
            {
                float* dst = _xs + (long)t * HC * E;
                ManagedQuantizedOps.DequantizeRowToFloat32(
                    (int)_tokEmbd.Type, (IntPtr)_tokEmbd.Row(tokens[tokOff + t]), dst, E);
                for (int s = 1; s < HC; s++)
                    Buffer.MemoryCopy(dst, dst + (long)s * E, E * sizeof(float), E * sizeof(float));
            });
            Tick(0, t0);

            // per-token rope caches for this ubatch (raw + compress parameter sets)
            t0 = Stopwatch.GetTimestamp();
            PFor(nt, t =>
            {
                RopeCacheInit(p0 + t, comp: false, _ropeRawCache + (long)t * _nRot);
                RopeCacheInit(p0 + t, comp: true, _ropeCompCache + (long)t * _nRot);
            });
            Tick(11, t0);

            for (int il = 0; il < _nLayer; il++)
            {
                Layer L = _layers[il];

                // ---- attention super-block ----
                t0 = Stopwatch.GetTimestamp();
                HcPre(L.HcAttnFn, L.HcAttnScale, L.HcAttnBase, nt, computeComb: true);
                RmsNormRows(_cur, L.AttnNorm, nt, E);
                Tick(1, t0);
                Attention(il, nt, p0);
                t0 = Stopwatch.GetTimestamp();
                HcPost(_attnOut, nt);

                // ---- FFN super-block ----
                HcPre(L.HcFfnFn, L.HcFfnScale, L.HcFfnBase, nt, computeComb: true);
                RmsNormRows(_cur, L.FfnNorm, nt, E);
                Tick(1, t0);
                MoeFfn(il, nt, tokens, tokOff);
                t0 = Stopwatch.GetTimestamp();
                HcPost(_ffnOut, nt);
                Tick(1, t0);
            }

            if (logitsOut != null)
            {
                t0 = Stopwatch.GetTimestamp();
                ComputeLogits(nt - 1, logitsOut);
                Tick(10, t0);
            }
        }

        // -------------------------------------------------------------------
        // Hyper connections
        // -------------------------------------------------------------------

        /// <summary>
        /// Computes mixes = hc_fn(rms(flat(x))) per token, derives the pre/post
        /// sigmoid gates and the Sinkhorn-normalized 4x4 comb matrix, and
        /// collapses the HC streams into _cur (weighted by pre).
        /// </summary>
        private void HcPre(in WeightRef fn, float[] scale, float[] baseW, int nt, bool computeComb)
        {
            int E = _nEmbd;
            int flatDim = HC * E;

            // mixes = fn x rms(flat). RMS scaling is folded into the dot result.
            var fnRef = fn;
            PFor(nt, t =>
            {
                float* flat = _xs + (long)t * flatDim;
                double ss = 0;
                for (int i = 0; i < flatDim; i++) ss += (double)flat[i] * flat[i];
                float inv = 1.0f / MathF.Sqrt((float)(ss / flatDim) + _rmsEps);

                float* mixes = _mixes + (long)t * HcMixDim;
                DotRowsF32(fnRef, flat, mixes, HcMixDim);
                for (int r = 0; r < HcMixDim; r++) mixes[r] *= inv;

                float* pre = _pre + (long)t * HC;
                float* post = _post + (long)t * HC;
                for (int s = 0; s < HC; s++)
                {
                    pre[s] = Sigmoid(mixes[s] * scale[0] + baseW[s]) + _hcEps;
                    post[s] = 2.0f * Sigmoid(mixes[HC + s] * scale[1] + baseW[HC + s]);
                }

                if (computeComb)
                {
                    float* comb = _comb + (long)t * HC * HC;
                    float scaleComb = scale[2];
                    for (int isrc = 0; isrc < HC; isrc++)
                    {
                        float max = float.NegativeInfinity;
                        for (int idst = 0; idst < HC; idst++)
                        {
                            int idx = idst + HC * isrc;
                            float v = mixes[2 * HC + idx] * scaleComb + baseW[2 * HC + idx];
                            comb[idx] = v;
                            if (v > max) max = v;
                        }
                        float sum = 0f;
                        for (int idst = 0; idst < HC; idst++)
                        {
                            int idx = idst + HC * isrc;
                            float v = MathF.Exp(comb[idx] - max);
                            comb[idx] = v;
                            sum += v;
                        }
                        float invSum = 1.0f / sum;
                        for (int idst = 0; idst < HC; idst++)
                        {
                            int idx = idst + HC * isrc;
                            comb[idx] = comb[idx] * invSum + _hcEps;
                        }
                    }
                    CombNormCols(comb, _hcEps);
                    for (int i = 1; i < _hcSinkhornIters; i++)
                    {
                        CombNormRows(comb, _hcEps);
                        CombNormCols(comb, _hcEps);
                    }
                }

                // collapse streams: cur[e] = sum_s x[s][e] * pre[s]
                float* cur = _cur + (long)t * E;
                var curSpan = new Span<float>(cur, E);
                new ReadOnlySpan<float>(flat, E).CopyTo(curSpan);
                TensorPrimitives.Multiply(curSpan, pre[0], curSpan);
                for (int s = 1; s < HC; s++)
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(flat + (long)s * E, E), pre[s], curSpan, curSpan);
            });
        }

        private static void CombNormCols(float* comb, float eps)
        {
            for (int idst = 0; idst < HC; idst++)
            {
                float sum = eps;
                for (int isrc = 0; isrc < HC; isrc++) sum += comb[idst + HC * isrc];
                float inv = 1.0f / sum;
                for (int isrc = 0; isrc < HC; isrc++) comb[idst + HC * isrc] *= inv;
            }
        }

        private static void CombNormRows(float* comb, float eps)
        {
            for (int isrc = 0; isrc < HC; isrc++)
            {
                float sum = eps;
                for (int idst = 0; idst < HC; idst++) sum += comb[idst + HC * isrc];
                float inv = 1.0f / sum;
                for (int idst = 0; idst < HC; idst++) comb[idst + HC * isrc] *= inv;
            }
        }

        /// <summary>x[e, idst] = blockOut[e]*post[idst] + sum_isrc residual[e, isrc]*comb[idst, isrc].</summary>
        private void HcPost(float* blockOut, int nt)
        {
            int E = _nEmbd;
            PFor(nt, t =>
            {
                float* x = _xs + (long)t * HC * E;
                float* o = blockOut + (long)t * E;
                float* post = _post + (long)t * HC;
                float* comb = _comb + (long)t * HC * HC;

                // The original residual streams are needed for every idst while x
                // is being overwritten, so mix into a temp buffer and copy back.
                float* outBuf = _hcPostTmp + (long)t * HC * E;
                for (int idst = 0; idst < HC; idst++)
                {
                    float* dst = outBuf + (long)idst * E;
                    var dstSpan = new Span<float>(dst, E);
                    new ReadOnlySpan<float>(o, E).CopyTo(dstSpan);
                    TensorPrimitives.Multiply(dstSpan, post[idst], dstSpan);
                    for (int isrc = 0; isrc < HC; isrc++)
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(x + (long)isrc * E, E), comb[idst + HC * isrc], dstSpan, dstSpan);
                }
                Buffer.MemoryCopy(outBuf, x, (long)HC * E * sizeof(float), (long)HC * E * sizeof(float));
            });
        }

        private float* _hcPostTmp;

        // -------------------------------------------------------------------
        // MoE FFN
        // -------------------------------------------------------------------

        private void MoeFfn(int il, int nt, int[] tokens, int tokOff)
        {
            Layer L = _layers[il];
            int E = _nEmbd;
            int nEx = _nExpert;
            int nUsed = _nExpertUsed;
            float clampExp = il < _swigluClampExp.Length ? _swigluClampExp[il] : 0f;
            float clampSh = il < _swigluClampShexp.Length ? _swigluClampShexp[il] : 0f;

            long t0 = Stopwatch.GetTimestamp();

            // router: logits -> probs = sqrt(softplus), selection, weights.
            // Parallelize over (token, expert-chunk) so decode (nt == 1) still
            // spreads the 256 x 4096 dot products across the pool.
            const int ExpertChunk = 16;
            int nExChunks = (nEx + ExpertChunk - 1) / ExpertChunk;
            PFor(nt * nExChunks, work =>
            {
                int t = work / nExChunks;
                int e0 = work % nExChunks * ExpertChunk;
                int e1 = Math.Min(e0 + ExpertChunk, nEx);
                float* cur = _cur + (long)t * E;
                float* logits = _routerLogits + (long)t * nEx;
                float* probs = _routerProbs + (long)t * nEx;
                var curSpan = new ReadOnlySpan<float>(cur, E);
                for (int e = e0; e < e1; e++)
                {
                    float v = TensorPrimitives.Dot(new ReadOnlySpan<float>(L.GateInp, e * E, E), curSpan);
                    logits[e] = v;
                    float sp = v > 20f ? v : MathF.Log(1f + MathF.Exp(v));
                    probs[e] = MathF.Sqrt(sp);
                }
            });

            PFor(nt, t =>
            {
                float* probs = _routerProbs + (long)t * nEx;
                int* sel = _selExperts + (long)t * nUsed;
                float* w = _selWeights + (long)t * nUsed;
                if (L.Tid2Eid != null)
                {
                    int tid = tokens[tokOff + t];
                    for (int j = 0; j < nUsed; j++)
                        sel[j] = L.Tid2Eid[(long)tid * nUsed + j];
                }
                else
                {
                    SelectTopK(probs, L.ExpProbsBias, nEx, nUsed, sel);
                }

                float sum = 0f;
                for (int j = 0; j < nUsed; j++)
                {
                    w[j] = probs[sel[j]];
                    sum += w[j];
                }
                if (_expertWeightsNorm)
                {
                    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
                    float inv = 1.0f / sum;
                    for (int j = 0; j < nUsed; j++) w[j] *= inv;
                }
                if (_expertWeightsScale != 0f && _expertWeightsScale != 1f)
                    for (int j = 0; j < nUsed; j++) w[j] *= _expertWeightsScale;
            });

            Tick(7, t0);
            t0 = Stopwatch.GetTimestamp();

            NativeMemory.Clear(_ffnOut, (nuint)((long)nt * E * sizeof(float)));

            if (nt == 1)
            {
                if (!TryRunExpertsDecodeFused(L, clampExp))
                {
                    // fallback: run the selected experts one by one
                    for (int j = 0; j < nUsed; j++)
                    {
                        int e = _selExperts[j];
                        float w = _selWeights[j];
                        ExpertMatmuls(L, e, clampExp, _cur, 1, _expertDown);
                        var dst = new Span<float>(_ffnOut, E);
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_expertDown, E), w, dst, dst);
                    }
                }
            }
            else if (!TryRunExpertsPrefillFused(L, nt, clampExp))
            {
                // fallback: group tokens by expert to amortize weight dequantization
                var byExpert = new Dictionary<int, List<(int Token, float Weight)>>();
                for (int t = 0; t < nt; t++)
                {
                    for (int j = 0; j < nUsed; j++)
                    {
                        int e = _selExperts[(long)t * nUsed + j];
                        if (!byExpert.TryGetValue(e, out var list))
                            byExpert[e] = list = new List<(int, float)>();
                        list.Add((t, _selWeights[(long)t * nUsed + j]));
                    }
                }

                // deterministic accumulation order (float sums depend on it)
                var expertIds = new List<int>(byExpert.Keys);
                expertIds.Sort();
                foreach (int e in expertIds)
                {
                    var list = byExpert[e];
                    int m = list.Count;
                    for (int i = 0; i < m; i++)
                    {
                        float* src = _cur + (long)list[i].Token * E;
                        Buffer.MemoryCopy(src, _expertPack + (long)i * E, E * sizeof(float), E * sizeof(float));
                    }

                    ExpertMatmuls(L, e, clampExp, _expertPack, m, _expertDown);

                    for (int i = 0; i < m; i++)
                    {
                        float w = list[i].Weight;
                        float* dst = _ffnOut + (long)list[i].Token * E;
                        var dstSpan = new Span<float>(dst, E);
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_expertDown + (long)i * E, E), w, dstSpan, dstSpan);
                    }
                }
            }

            Tick(8, t0);
            t0 = Stopwatch.GetTimestamp();

            // shared expert (dense) — add into _ffnOut
            {
                int shFf = L.UpShexp.Ne1;
                MatMul(L.UpShexp, 0, shFf, _cur, E, nt, _shUp, shFf);
                MatMul(L.GateShexp, 0, shFf, _cur, E, nt, _shGate, shFf);
                SwigluClamp(_shGate, _shUp, (long)nt * shFf, clampSh);
                MatMul(L.DownShexp, 0, E, _shGate, shFf, nt, _shDown, E);
                var all = new Span<float>(_ffnOut, nt * E);
                TensorPrimitives.Add(new ReadOnlySpan<float>(_shDown, nt * E), all, all);
            }
            Tick(9, t0);
        }

        /// <summary>
        /// Decode-path MoE: all selected experts' gate+up matvecs in one parallel
        /// region sharing a single quantized activation, one swiglu sweep, then all
        /// down matvecs in a second region. Cuts the per-layer region count from
        /// ~20 to 2 and skips redundant activation quantization — that matters
        /// under a CPU-quota cgroup where every fork/join risks a throttle stall.
        /// Returns false when the expert tensors lack a direct integer-dot plan.
        /// </summary>
        private bool TryRunExpertsDecodeFused(Layer L, float clamp)
        {
            int E = _nEmbd, ff = _nFfExp, nUsed = _nExpertUsed;
            WeightRef gateW = L.GateExps, upW = L.UpExps, downW = L.DownExps;

            if (upW.Type != gateW.Type || upW.Ne0 != gateW.Ne0)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(gateW.Type, E, out int actBytesGU) || actBytesGU > _maxActRowBytes)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(downW.Type, ff, out int actBytesDown) || actBytesDown > _maxActRowBytes)
                return false;

            // slot 0: gate/up activation; slots 1..nUsed: per-expert down activations
            byte* actGU = _actQuant;
            ManagedQuantizedOps.QuantizeActivationRow(gateW.Type, _cur, actGU, E);

            const int RowsPerTask = 128;
            int tasksPerMat = (ff + RowsPerTask - 1) / RowsPerTask;
            int totalA = nUsed * 2 * tasksPerMat;
            PFor(totalA, w =>
            {
                int slot = w / (2 * tasksPerMat);
                int rem = w % (2 * tasksPerMat);
                bool isGate = rem < tasksPerMat;
                int r0 = (isGate ? rem : rem - tasksPerMat) * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, ff);
                int e = _selExperts[slot];
                WeightRef wRef = isGate ? L.GateExps : L.UpExps;
                byte* basePtr = wRef.Expert(e);
                float* dst = (isGate ? _expertGate : _expertUp) + (long)slot * ff;
                for (int r = r0; r < r1; r++)
                    dst[r] = ManagedQuantizedOps.DotQuantizedRow(wRef.Type, basePtr + r * wRef.RowBytes, actGU, E);
            });

            SwigluClamp(_expertGate, _expertUp, (long)nUsed * ff, clamp);

            for (int slot = 0; slot < nUsed; slot++)
                ManagedQuantizedOps.QuantizeActivationRow(downW.Type, _expertGate + (long)slot * ff, _actQuant + (long)(slot + 1) * _maxActRowBytes, ff);

            int tasksPerDown = (E + RowsPerTask - 1) / RowsPerTask;
            PFor(nUsed * tasksPerDown, w =>
            {
                int slot = w / tasksPerDown;
                int r0 = w % tasksPerDown * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, E);
                int e = _selExperts[slot];
                byte* basePtr = L.DownExps.Expert(e);
                byte* act = _actQuant + (long)(slot + 1) * _maxActRowBytes;
                float* dst = _expertDown + (long)slot * E;
                for (int r = r0; r < r1; r++)
                    dst[r] = ManagedQuantizedOps.DotQuantizedRow(L.DownExps.Type, basePtr + r * L.DownExps.RowBytes, act, ff);
            });

            var outSpan = new Span<float>(_ffnOut, E);
            for (int slot = 0; slot < nUsed; slot++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_expertDown + (long)slot * E, E), _selWeights[slot], outSpan, outSpan);

            return true;
        }

        /// <summary>
        /// Prefill-path MoE: tokens are packed into per-expert groups, then the
        /// whole layer runs as six wide parallel regions (quantize, gate+up dots,
        /// swiglu, re-quantize, down dots, weighted scatter) instead of three
        /// regions per active expert. Weight rows are read once per expert and
        /// dotted against every member token.
        /// </summary>
        private bool TryRunExpertsPrefillFused(Layer L, int nt, float clamp)
        {
            int E = _nEmbd, ff = _nFfExp, nUsed = _nExpertUsed, nEx = _nExpert;
            WeightRef gateW = L.GateExps, upW = L.UpExps, downW = L.DownExps;

            if (upW.Type != gateW.Type || upW.Ne0 != gateW.Ne0)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(gateW.Type, E, out int actBytesGU) || actBytesGU > _maxActRowBytes)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(downW.Type, ff, out int actBytesDown) || actBytesDown > _maxActRowBytes)
                return false;

            // group (token, slot) pairs by expert; packed row s is the position
            // in expert-major order
            int S = nt * nUsed;
            for (int e = 0; e < nEx; e++) _expCount[e] = 0;
            for (int i = 0; i < S; i++) _expCount[_selExperts[i]]++;
            int off = 0;
            for (int e = 0; e < nEx; e++) { _expOffset[e] = off; off += _expCount[e]; _expCursor[e] = _expOffset[e]; }
            for (int t = 0; t < nt; t++)
            {
                for (int j = 0; j < nUsed; j++)
                {
                    int i = t * nUsed + j;
                    int e = _selExperts[i];
                    int row = _expCursor[e]++;
                    _rowOfSlot[i] = row;
                    _slotToken[row] = t;
                    _slotGroupSize[row] = _expCount[e];
                }
            }

            // region 1: quantize all token activations for gate/up
            byte* actGU = _actQuant;                          // rows [0, nt)
            byte* actDown = _actQuant + (long)nt * _maxActRowBytes; // rows [0, S)
            PFor(nt, t =>
                ManagedQuantizedOps.QuantizeActivationRow(gateW.Type, _cur + (long)t * E, actGU + (long)t * actBytesGU, E));

            // region 2: gate+up dots, task = (expert, mat, ff-row-block).
            // For groups with enough member tokens it is cheaper to dequantize
            // each weight row once and run float dots (this is where a batched
            // prefill amortizes the IQ3_S decode cost); small groups keep the
            // integer path.
            const int RowsPerTask = 128;
            int blocksFF = (ff + RowsPerTask - 1) / RowsPerTask;
            PFor(nEx * 2 * blocksFF, w =>
            {
                int e = w / (2 * blocksFF);
                int m = _expCount[e];
                if (m == 0) return;
                int rem = w % (2 * blocksFF);
                bool isGate = rem < blocksFF;
                int r0 = (isGate ? rem : rem - blocksFF) * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, ff);
                WeightRef wRef = isGate ? L.GateExps : L.UpExps;
                byte* basePtr = wRef.Expert(e);
                float* dstBase = (isGate ? _expertGate : _expertUp);
                int groupOff = _expOffset[e];
                if (m >= DequantGroupMin)
                {
                    float* wrow = stackalloc float[E];
                    for (int r = r0; r < r1; r++)
                    {
                        ManagedQuantizedOps.DequantizeRowToFloat32((int)wRef.Type, (IntPtr)(basePtr + r * wRef.RowBytes), wrow, E);
                        var wSpan = new ReadOnlySpan<float>(wrow, E);
                        for (int i = 0; i < m; i++)
                        {
                            int tok = _slotToken[groupOff + i];
                            dstBase[(long)(groupOff + i) * ff + r] =
                                TensorPrimitives.Dot(wSpan, new ReadOnlySpan<float>(_cur + (long)tok * E, E));
                        }
                    }
                }
                else
                {
                    for (int r = r0; r < r1; r++)
                    {
                        byte* wr = basePtr + r * wRef.RowBytes;
                        for (int i = 0; i < m; i++)
                        {
                            int tok = _slotToken[groupOff + i];
                            dstBase[(long)(groupOff + i) * ff + r] =
                                ManagedQuantizedOps.DotQuantizedRow(wRef.Type, wr, actGU + (long)tok * actBytesGU, E);
                        }
                    }
                }
            });

            // region 3: swiglu over all packed rows
            SwigluClamp(_expertGate, _expertUp, (long)S * ff, clamp);

            // region 4: quantize the packed hidden rows for the down matmul —
            // only for slots whose expert group takes the integer path
            PFor(S, s =>
            {
                if (_slotGroupSize[s] >= DequantGroupMin)
                    return;
                ManagedQuantizedOps.QuantizeActivationRow(downW.Type, _expertGate + (long)s * ff, actDown + (long)s * actBytesDown, ff);
            });

            // region 5: down dots, task = (expert, E-row-block); same hybrid
            int blocksE = (E + RowsPerTask - 1) / RowsPerTask;
            PFor(nEx * blocksE, w =>
            {
                int e = w / blocksE;
                int m = _expCount[e];
                if (m == 0) return;
                int r0 = w % blocksE * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, E);
                byte* basePtr = L.DownExps.Expert(e);
                int groupOff = _expOffset[e];
                if (m >= DequantGroupMin)
                {
                    float* wrow = stackalloc float[ff];
                    for (int r = r0; r < r1; r++)
                    {
                        ManagedQuantizedOps.DequantizeRowToFloat32((int)L.DownExps.Type, (IntPtr)(basePtr + r * L.DownExps.RowBytes), wrow, ff);
                        var wSpan = new ReadOnlySpan<float>(wrow, ff);
                        for (int i = 0; i < m; i++)
                            _expertDown[(long)(groupOff + i) * E + r] =
                                TensorPrimitives.Dot(wSpan, new ReadOnlySpan<float>(_expertGate + (long)(groupOff + i) * ff, ff));
                    }
                }
                else
                {
                    for (int r = r0; r < r1; r++)
                    {
                        byte* wr = basePtr + r * L.DownExps.RowBytes;
                        for (int i = 0; i < m; i++)
                            _expertDown[(long)(groupOff + i) * E + r] =
                                ManagedQuantizedOps.DotQuantizedRow(L.DownExps.Type, wr, actDown + (long)(groupOff + i) * actBytesDown, ff);
                    }
                }
            });

            // region 6: weighted scatter back to tokens (deterministic per token)
            PFor(nt, t =>
            {
                var dst = new Span<float>(_ffnOut + (long)t * E, E);
                for (int j = 0; j < nUsed; j++)
                {
                    int i = t * nUsed + j;
                    TensorPrimitives.MultiplyAdd(
                        new ReadOnlySpan<float>(_expertDown + (long)_rowOfSlot[i] * E, E),
                        _selWeights[i], dst, dst);
                }
            });

            return true;
        }

        /// <summary>gate/up/swiglu/down for one expert over m packed rows; result in downOut [m x E].</summary>
        private void ExpertMatmuls(Layer L, int e, float clamp, float* packedIn, int m, float* downOut)
        {
            int E = _nEmbd;
            int ff = _nFfExp;
            MatMulExpert(L.UpExps, e, packedIn, E, m, _expertUp, ff);
            MatMulExpert(L.GateExps, e, packedIn, E, m, _expertGate, ff);
            SwigluClamp(_expertGate, _expertUp, (long)m * ff, clamp);
            MatMulExpert(L.DownExps, e, _expertGate, ff, m, downOut, E);
        }

        /// <summary>up = clamp(up, ±limit); gate = min(gate, limit); gate = silu(gate) * up (written into gate).</summary>
        private void SwigluClamp(float* gate, float* up, long n, float clamp)
        {
            bool doClamp = clamp > 1e-6f;
            int chunk = 1 << 16;
            long nChunks = (n + chunk - 1) / chunk;
            PFor((int)nChunks, ci =>
            {
                long start = (long)ci * chunk;
                long end = Math.Min(start + chunk, n);
                for (long i = start; i < end; i++)
                {
                    float g = gate[i];
                    float u = up[i];
                    if (doClamp)
                    {
                        if (u > clamp) u = clamp; else if (u < -clamp) u = -clamp;
                        if (g > clamp) g = clamp;
                    }
                    gate[i] = g / (1.0f + MathF.Exp(-g)) * u;
                }
            });
        }

        private static void SelectTopK(float* probs, float[] bias, int n, int k, int* outIdx)
        {
            // simple partial selection (k is small: 6)
            float* best = stackalloc float[k];
            for (int j = 0; j < k; j++) { best[j] = float.NegativeInfinity; outIdx[j] = 0; }
            for (int e = 0; e < n; e++)
            {
                float v = probs[e] + bias[e];
                if (v <= best[k - 1]) continue;
                int j = k - 1;
                while (j > 0 && best[j - 1] < v)
                {
                    best[j] = best[j - 1];
                    outIdx[j] = outIdx[j - 1];
                    j--;
                }
                best[j] = v;
                outIdx[j] = e;
            }
        }

        // -------------------------------------------------------------------
        // Output head
        // -------------------------------------------------------------------

        private void ComputeLogits(int t, float[] logitsOut)
        {
            int E = _nEmbd;
            int flatDim = HC * E;
            float* x = _xs + (long)t * flatDim;

            // hc_head: mixes = output_hc_fn(rms(flat)); pre = sigmoid(m*scale+base)+eps; collapse
            float* mixes = stackalloc float[HC];
            double ss = 0;
            for (int i = 0; i < flatDim; i++) ss += (double)x[i] * x[i];
            float inv = 1.0f / MathF.Sqrt((float)(ss / flatDim) + _rmsEps);
            DotRowsF32(_hcHeadFn, x, mixes, HC);

            float* headBuf = stackalloc float[HC];
            for (int s = 0; s < HC; s++)
            {
                float scale = _hcHeadScale.Length >= HC ? _hcHeadScale[s] : _hcHeadScale[0];
                float b = _hcHeadBase.Length >= HC ? _hcHeadBase[s] : _hcHeadBase[0];
                headBuf[s] = Sigmoid(mixes[s] * inv * scale + b) + _hcEps;
            }

            float* cur = _cur; // reuse token-0 slot
            var curSpan = new Span<float>(cur, E);
            new ReadOnlySpan<float>(x, E).CopyTo(curSpan);
            TensorPrimitives.Multiply(curSpan, headBuf[0], curSpan);
            for (int s = 1; s < HC; s++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(x + (long)s * E, E), headBuf[s], curSpan, curSpan);

            RmsNormRows(cur, _outputNorm, 1, E);

            fixed (float* lo = logitsOut)
            {
                MatMul(_output, 0, _nVocab, cur, E, 1, lo, _nVocab);
            }
        }

        // -------------------------------------------------------------------
        // Primitives
        // -------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));

        /// <summary>RMS-norm each of nt rows of len n in place, optional weight.</summary>
        private void RmsNormRows(float* data, float[] weight, int nt, int n)
        {
            PFor(nt, t =>
            {
                float* row = data + (long)t * n;
                double ss = 0;
                for (int i = 0; i < n; i++) ss += (double)row[i] * row[i];
                float inv = 1.0f / MathF.Sqrt((float)(ss / n) + _rmsEps);
                if (weight != null)
                {
                    for (int i = 0; i < n; i++) row[i] = row[i] * inv * weight[i];
                }
                else
                {
                    for (int i = 0; i < n; i++) row[i] *= inv;
                }
            });
        }

        /// <summary>Dot input (len = w.Ne0) against the first nRows rows of an F32/any-typed small weight, serially.</summary>
        private static void DotRowsF32(in WeightRef w, float* input, float* output, int nRows)
        {
            if (w.Type == GgmlTensorType.F32)
            {
                var inSpan = new ReadOnlySpan<float>(input, w.Ne0);
                for (int r = 0; r < nRows; r++)
                    output[r] = TensorPrimitives.Dot(new ReadOnlySpan<float>((float*)w.Row(r), w.Ne0), inSpan);
            }
            else
            {
                ManagedQuantizedOps.DotRowBatchToFloat32((int)w.Type, (IntPtr)w.Ptr, input, w.Ne0, 1, w.Ne0, output);
                for (int r = 1; r < nRows; r++)
                    ManagedQuantizedOps.DotRowBatchToFloat32((int)w.Type, (IntPtr)w.Row(r), input, w.Ne0, 1, w.Ne0, output + r);
            }
        }

        /// <summary>
        /// output[t, r] = dot(weight row (rowBase + r), input row t) for r in [0, nRows), t in [0, nt).
        /// Parallelizes over blocked output rows; uses integer dot kernels when the
        /// weight type supports them (activations quantized once per call).
        /// </summary>
        private void MatMul(in WeightRef w, long rowBase, int nRows, float* input, int inStride, int nt, float* output, int outStride)
        {
            GgmlTensorType type = w.Type;
            int ne0 = w.Ne0;
            byte* basePtr = w.Row(rowBase);
            long rowBytes = w.RowBytes;

            if (ManagedQuantizedOps.TryGetActivationPlan(type, ne0, out int actRowBytes) && actRowBytes <= _maxActRowBytes)
            {
                byte* act = _actQuant;
                if (nt == 1)
                {
                    ManagedQuantizedOps.QuantizeActivationRow(type, input, act, ne0);
                }
                else
                {
                    var localType = type;
                    var localIn = input;
                    var localAct = act;
                    int localNe0 = ne0, localStride = inStride, localActBytes = actRowBytes;
                    PFor(nt, t =>
                        ManagedQuantizedOps.QuantizeActivationRow(localType, localIn + (long)t * localStride, localAct + (long)t * localActBytes, localNe0));
                }

                int blockRows = Math.Max(8, nRows / (_po.MaxDegreeOfParallelism * 8));
                int nBlocks = (nRows + blockRows - 1) / blockRows;
                var bType = type;
                var bBase = basePtr;
                var bAct = act;
                var bOut = output;
                int bNe0 = ne0, bNt = nt, bOutStride = outStride, bActBytes = actRowBytes;
                long bRowBytes = rowBytes;
                PFor(nBlocks, bi =>
                {
                    int r0 = bi * blockRows;
                    int r1 = Math.Min(r0 + blockRows, nRows);
                    for (int r = r0; r < r1; r++)
                    {
                        byte* wr = bBase + (long)r * bRowBytes;
                        for (int t = 0; t < bNt; t++)
                            bOut[(long)t * bOutStride + r] = ManagedQuantizedOps.DotQuantizedRow(bType, wr, bAct + (long)t * bActBytes, bNe0);
                    }
                });
            }
            else
            {
                // dequant + float dot fallback (F32/F16/BF16/IQ3_S/MXFP4/...)
                int blockRows = Math.Max(4, nRows / (_po.MaxDegreeOfParallelism * 8));
                int nBlocks = (nRows + blockRows - 1) / blockRows;
                var bType = type;
                var bBase = basePtr;
                var bIn = input;
                var bOut = output;
                int bNe0 = ne0, bNt = nt, bInStride = inStride, bOutStride = outStride;
                long bRowBytes = rowBytes;
                PFor(nBlocks, bi =>
                {
                    int r0 = bi * blockRows;
                    int r1 = Math.Min(r0 + blockRows, nRows);
                    float* rowOut = stackalloc float[512];
                    for (int r = r0; r < r1; r++)
                    {
                        byte* wr = bBase + (long)r * bRowBytes;
                        int tDone = 0;
                        while (tDone < bNt)
                        {
                            int tn = Math.Min(512, bNt - tDone);
                            ManagedQuantizedOps.DotRowBatchToFloat32((int)bType, (IntPtr)wr, bIn + (long)tDone * bInStride, bInStride, tn, bNe0, rowOut);
                            for (int t = 0; t < tn; t++)
                                bOut[(long)(tDone + t) * bOutStride + r] = rowOut[t];
                            tDone += tn;
                        }
                    }
                });
            }
        }

        /// <summary>MatMul against expert e of a stacked 3D expert tensor.</summary>
        private void MatMulExpert(in WeightRef w, int e, float* input, int inStride, int nt, float* output, int outStride)
        {
            var view = w;
            view.Ptr = w.Expert(e);
            MatMul(view, 0, w.Ne1, input, inStride, nt, output, outStride);
        }

        public void Dispose()
        {
            foreach (var p in _nativeAllocs)
                NativeMemory.AlignedFree((void*)p);
            _nativeAllocs.Clear();
            foreach (var p in _ownedBuffers)
                Marshal.FreeHGlobal(p);
            _ownedBuffers.Clear();
            foreach (var s in _shards)
                s.Dispose();
            _shards.Clear();
        }
    }
}
