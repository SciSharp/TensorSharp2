// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) executor for the direct-CUDA backend.
//
// The model-side half of the DSV4 direct-CUDA path: opens the split-GGUF
// shards (memory-mapped), parses the deepseek4 hyper-parameters, dequantizes
// the small tensors (norms, gates, sinks, APE tables, the BF16 router) to
// F32 host arrays, precomputes the raw/compress RoPE cos-sin tables (same
// YaRN math as DeepSeek4CpuExecutor.RopeCacheInit), and hands everything to
// TensorSharp.Cuda.Dsv4CudaEngine, which uploads the quantized weights to
// device memory (layer-split across all visible GPUs) and runs the forward
// pass with driver-API kernels — fully independent of ggml.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    internal sealed unsafe class DeepSeek4CudaExecutor : IDisposable
    {
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;

        private readonly List<GgufFile> _shards = new List<GgufFile>();
        private readonly List<string> _shardPaths = new List<string>();
        private readonly Dictionary<string, (GgufFile File, GgufTensorInfo Info)> _tensorMap
            = new Dictionary<string, (GgufFile, GgufTensorInfo)>(StringComparer.Ordinal);
        private readonly List<IntPtr> _ownedBuffers = new List<IntPtr>();
        private Dictionary<string, IntPtr> _bufferedTensors;

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
        private float _compCorr0, _compCorr1;

        private Dsv4CudaEngine _engine;

        public int VocabSize => _nVocab;
        public int NPast => _engine.NPast;

        public DeepSeek4CudaExecutor(string ggufPath, int maxContext, int nUbatch, int nGpu)
        {
            var sw = Stopwatch.StartNew();
            OpenShards(ggufPath);
            ParseHparams();

            // Weights are read from the host exactly once (the upload to VRAM),
            // so materializing with wide parallel positional reads beats paying
            // serial mmap page faults inside cuMemcpyHtoD — on a network
            // filesystem by 2-3x. TS_DSV4_MMAP=1 opts back into mmap.
            bool materialize = ParseEnvInt("TS_DSV4_MMAP", 0) == 0;
            if (materialize)
                MaterializeTensors();

            int nCtx = maxContext > 0 ? maxContext : 16384;
            int ubatch = nUbatch > 0 ? nUbatch : 1024;

            var desc = BuildModelDesc(nCtx, ubatch);
            _engine = new Dsv4CudaEngine(desc, nGpu);

            if (materialize)
            {
                // Everything lives in VRAM now; the staged host copies (~the
                // whole model) are dead weight.
                _bufferedTensors = null;
                foreach (var p in _ownedBuffers)
                    Marshal.FreeHGlobal(p);
                _ownedBuffers.Clear();
            }

            Console.Error.WriteLine($"[dsv4-cuda] model ready in {sw.Elapsed.TotalSeconds:F1}s");
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        public void Forward(int[] tokens, float[] logitsOut) => _engine.Forward(tokens, logitsOut);

        public void Reset() => _engine.Reset();

        // -------------------------------------------------------------------
        // Loading (mirrors DeepSeek4CpuExecutor's split-shard resolver)
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

            int hc = (int)g.GetUint32($"{a}.hyper_connection.count", 4);
            if (hc != 4)
                throw new NotSupportedException($"DeepSeek4 CUDA executor supports hyper_connection.count == 4, got {hc}.");
            if (_nLayer <= 0 || _compressRatios.Length < _nLayer)
                throw new InvalidOperationException("Missing or invalid deepseek4 GGUF metadata.");

            _compCorr0 = MathF.Max(0f, MathF.Floor(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaFast, _compressRopeBase)));
            _compCorr1 = MathF.Min(_nRot - 1, MathF.Ceiling(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaSlow, _compressRopeBase)));
        }

        private static float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(freqBase));
        }

        /// <summary>
        /// TS_DSV4_MMAP=0: copy every tensor into anonymous RAM with parallel
        /// positional reads (for network filesystems whose mmap page faults are
        /// slow). The upload to VRAM reads each page exactly once, so the
        /// default mmap path is usually fine.
        /// </summary>
        private void MaterializeTensors()
        {
            var sw = Stopwatch.StartNew();
            _bufferedTensors = new Dictionary<string, IntPtr>(StringComparer.Ordinal);

            var handles = new Microsoft.Win32.SafeHandles.SafeFileHandle[_shardPaths.Count];
            var shardIndexOf = new Dictionary<GgufFile, int>();
            for (int s = 0; s < _shards.Count; s++)
            {
                shardIndexOf[_shards[s]] = s;
                handles[s] = File.OpenHandle(_shardPaths[s], FileMode.Open, FileAccess.Read, FileShare.Read);
            }

            var names = new List<string>(_tensorMap.Keys);
            var slots = new IntPtr[names.Count];
            long totalBytes = 0;
            Parallel.For(0, names.Count, i =>
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
                        throw new IOException($"[dsv4-cuda] short read materializing {names[i]}");
                    done += got;
                }
                slots[i] = buf;
                System.Threading.Interlocked.Add(ref totalBytes, bytes);
            });

            foreach (var h in handles)
                h?.Dispose();

            for (int i = 0; i < names.Count; i++)
            {
                _bufferedTensors[names[i]] = slots[i];
                _ownedBuffers.Add(slots[i]);
            }
            Console.Error.WriteLine($"[dsv4-cuda] buffered {totalBytes / (1024.0 * 1024 * 1024):F1} GiB of weights into RAM in {sw.Elapsed.TotalSeconds:F1}s");
        }

        private (IntPtr Ptr, GgufTensorInfo Info) GetRaw(string name, bool required = true)
        {
            if (!_tensorMap.TryGetValue(name, out var entry))
            {
                if (required)
                    throw new InvalidOperationException($"[dsv4-cuda] missing tensor: {name}");
                return (IntPtr.Zero, null);
            }

            GgufTensorInfo info = entry.Info;
            if (_bufferedTensors != null && _bufferedTensors.TryGetValue(name, out IntPtr buffered))
                return (buffered, info);
            if (entry.File.TryGetTensorDataPointer(info, out IntPtr mapped))
                return (mapped, info);

            long bytes = entry.File.GetTensorByteCount(info);
            IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
            _ownedBuffers.Add(buf);
            entry.File.ReadTensorDataToNative(info, buf, bytes);
            return (buf, info);
        }

        private Dsv4CudaEngine.QuantWeightDesc GetQW(string name, bool required = true)
        {
            var (ptr, info) = GetRaw(name, required);
            if (ptr == IntPtr.Zero)
                return default;
            var desc = new Dsv4CudaEngine.QuantWeightDesc
            {
                HostPtr = ptr,
                GgmlType = (int)info.Type,
                Ne0 = (int)info.Shape[0],
                Ne1 = info.Shape.Length > 1 ? (int)info.Shape[1] : 1,
                Ne2 = info.Shape.Length > 2 ? (int)info.Shape[2] : 1,
                RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]),
                Name = name,
            };
            return desc;
        }

        private float[] GetF32(string name, bool required = true)
        {
            var (ptr, info) = GetRaw(name, required);
            if (ptr == IntPtr.Zero)
                return null;
            long n = info.NumElements;
            var arr = new float[n];
            ManagedQuantizedOps.DequantizeToFloat32((int)info.Type, ptr, arr, 0, n);
            return arr;
        }

        private int[] GetI32(string name)
        {
            var (ptr, info) = GetRaw(name);
            if (info.Type != GgmlTensorType.I32)
                throw new InvalidOperationException($"[dsv4-cuda] {name}: expected I32, got {info.Type}");
            long n = info.NumElements;
            var arr = new int[n];
            Marshal.Copy(ptr, arr, 0, (int)n);
            return arr;
        }

        // -------------------------------------------------------------------
        // Descriptor construction
        // -------------------------------------------------------------------

        private Dsv4CudaEngine.ModelDesc BuildModelDesc(int nCtx, int nUbatch)
        {
            var tokEmbd = GetQW("token_embd.weight");
            _nVocab = tokEmbd.Ne1;
            int tokType = tokEmbd.GgmlType;
            if (tokType != 8 && tokType != 1 && tokType != 0)
                throw new NotSupportedException($"[dsv4-cuda] token_embd type {tokType} unsupported (Q8_0/F16/F32).");

            var m = new Dsv4CudaEngine.ModelDesc
            {
                NLayer = _nLayer,
                NEmbd = _nEmbd,
                NHead = _nHead,
                HeadDim = _headDim,
                NRot = _nRot,
                QLoraRank = _qLoraRank,
                OGroups = _oGroups,
                OLoraRank = _oLoraRank,
                NSwa = _nSwa,
                NVocab = _nVocab,
                NExpert = _nExpert,
                NExpertUsed = _nExpertUsed,
                NFfExp = _nFfExp,
                HashLayerCount = _hashLayerCount,
                IdxNHead = _idxNHead,
                IdxHeadSize = _idxHeadSize,
                IdxTopK = _idxTopK,
                HcSinkhornIters = _hcSinkhornIters,
                RmsEps = _rmsEps,
                HcEps = _hcEps,
                ExpertWeightsScale = _expertWeightsScale,
                ExpertWeightsNorm = _expertWeightsNorm,
                NCtx = nCtx,
                NUbatch = nUbatch,
                TokEmbd = tokEmbd,
                Output = GetQW("output.weight"),
                OutputNorm = GetF32("output_norm.weight"),
                HcHeadFn = GetF32("output_hc_fn.weight"),
                HcHeadScale = GetF32("output_hc_scale.weight"),
                HcHeadBase = GetF32("output_hc_base.weight"),
                RopeRawTable = BuildRopeTable(nCtx, comp: false),
                RopeCompTable = BuildRopeTable(nCtx, comp: true),
                Layers = new Dsv4CudaEngine.LayerDesc[_nLayer],
            };

            for (int il = 0; il < _nLayer; il++)
            {
                string p = $"blk.{il}.";
                var L = new Dsv4CudaEngine.LayerDesc
                {
                    Ratio = _compressRatios[il],
                    ClampExp = il < _swigluClampExp.Length ? _swigluClampExp[il] : 0f,
                    ClampShexp = il < _swigluClampShexp.Length ? _swigluClampShexp[il] : 0f,
                    AttnNorm = GetF32(p + "attn_norm.weight"),
                    Sinks = GetF32(p + "attn_sinks.weight"),
                    WqA = GetQW(p + "attn_q_a.weight"),
                    QANorm = GetF32(p + "attn_q_a_norm.weight"),
                    WqB = GetQW(p + "attn_q_b.weight"),
                    Wkv = GetQW(p + "attn_kv.weight"),
                    KvNorm = GetF32(p + "attn_kv_a_norm.weight"),
                    WoA = GetQW(p + "attn_output_a.weight"),
                    WoB = GetQW(p + "attn_output_b.weight"),
                    HcAttnFn = GetF32(p + "hc_attn_fn.weight"),
                    HcAttnScale = GetF32(p + "hc_attn_scale.weight"),
                    HcAttnBase = GetF32(p + "hc_attn_base.weight"),
                    HcFfnFn = GetF32(p + "hc_ffn_fn.weight"),
                    HcFfnScale = GetF32(p + "hc_ffn_scale.weight"),
                    HcFfnBase = GetF32(p + "hc_ffn_base.weight"),
                    GateInp = GetF32(p + "ffn_gate_inp.weight"),
                    FfnNorm = GetF32(p + "ffn_norm.weight"),
                    GateExps = GetQW(p + "ffn_gate_exps.weight"),
                    DownExps = GetQW(p + "ffn_down_exps.weight"),
                    UpExps = GetQW(p + "ffn_up_exps.weight"),
                    GateShexp = GetQW(p + "ffn_gate_shexp.weight"),
                    DownShexp = GetQW(p + "ffn_down_shexp.weight"),
                    UpShexp = GetQW(p + "ffn_up_shexp.weight"),
                };

                if (L.Ratio != 0)
                {
                    L.CompWkv = GetQW(p + "attn_compressor_kv.weight");
                    L.CompWgate = GetQW(p + "attn_compressor_gate.weight");
                    L.CompApe = GetF32(p + "attn_compressor_ape.weight");
                    L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                    if (L.Ratio == CsaRatio)
                    {
                        L.IdxProj = GetQW(p + "indexer.proj.weight");
                        L.IdxQB = GetQW(p + "indexer.attn_q_b.weight");
                        L.IdxCompWkv = GetQW(p + "indexer_compressor_kv.weight");
                        L.IdxCompWgate = GetQW(p + "indexer_compressor_gate.weight");
                        L.IdxCompApe = GetF32(p + "indexer_compressor_ape.weight");
                        L.IdxCompNorm = GetF32(p + "indexer_compressor_norm.weight");
                    }
                }

                if (il < _hashLayerCount)
                    L.Tid2Eid = GetI32(p + "ffn_gate_tid2eid.weight");
                else
                    L.ExpProbsBias = GetF32(p + "exp_probs_b.bias");

                m.Layers[il] = L;
            }

            return m;
        }

        /// <summary>
        /// Interleaved (cos, sin) per position with the YaRN attention factor
        /// folded in — the table-driven form of DeepSeek4CpuExecutor.RopeCacheInit.
        /// </summary>
        private float[] BuildRopeTable(int nCtx, bool comp)
        {
            int nDims = _nRot;
            float freqBase = comp ? _compressRopeBase : _ropeFreqBase;
            float freqScale = comp ? _yarnFreqScale : 1f;
            float extFactor = comp ? _yarnExtFactor : 0f;
            float attnFactor = extFactor == 0f ? 1f : 1f / (1f + 0.1f * MathF.Log(1f / freqScale));
            float thetaScale = MathF.Pow(freqBase, -2f / nDims);

            var table = new float[(long)nCtx * nDims];
            Parallel.For(0, nCtx, pos =>
            {
                float theta = pos;
                long baseIdx = (long)pos * nDims;
                for (int i0 = 0; i0 < nDims; i0 += 2)
                {
                    float thetaInterp = freqScale * theta;
                    float th = thetaInterp;
                    float mscale = attnFactor;
                    if (extFactor != 0f)
                    {
                        float y = (i0 / 2 - _compCorr0) / MathF.Max(0.001f, _compCorr1 - _compCorr0);
                        float ramp = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                        th = thetaInterp * (1f - ramp) + theta * ramp;
                        mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);
                    }
                    table[baseIdx + i0 + 0] = MathF.Cos(th) * mscale;
                    table[baseIdx + i0 + 1] = MathF.Sin(th) * mscale;
                    theta *= thetaScale;
                }
            });
            return table;
        }

        public void Dispose()
        {
            _engine?.Dispose();
            _engine = null;
            foreach (var p in _ownedBuffers)
                Marshal.FreeHGlobal(p);
            _ownedBuffers.Clear();
            foreach (var s in _shards)
                s.Dispose();
            _shards.Clear();
        }
    }
}
