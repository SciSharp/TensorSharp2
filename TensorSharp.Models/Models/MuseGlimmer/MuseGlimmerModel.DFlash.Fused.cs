// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Routes the DFlash drafter's two passes through one fused GGML graph each.
    ///
    /// The per-op drafter issues ~150 dispatches per speculative step for ~2.5 GB of
    /// weight reads -- about a quarter of one target forward, but it was costing more
    /// wall time than the forward it was meant to save, which made speculation a net
    /// loss. llama.cpp does not solve this with graph reuse (its draft context rebuilds
    /// every step, because ENCODER -&gt; DECODER(embd) -&gt; DECODER(token) can never hit
    /// its single-slot graph cache); it just keeps the rebuild cheap. Here the graph is
    /// built once and replayed, so ggml-cuda can capture it.
    ///
    /// One thing this does that llama.cpp does not: the draft block's argmax and its
    /// winning probability are reduced ON DEVICE. llama.cpp pulls the whole
    /// [202048, 16] probability block back to the host every step -- 12.9 MB over PCIe
    /// plus a 3.2 M-element scan -- and that readback is a large share of its per-step
    /// cost. Two 16-element tensors carry everything the executor needs.
    /// </summary>
    public partial class MuseGlimmerModel
    {
        private sealed class DFlashArrays
        {
            public IntPtr[] AttnNorm, QNorm, KNorm, FfnNorm;
            public IntPtr[] Q, K, V, O, Gate, Up, Down;
            public int[] QType, KType, VType, OType, GateType, UpType, DownType;
            public long[] QNe0, QNe1, QBytes;
            public long[] KNe0, KNe1, KBytes;
            public long[] VNe0, VNe1, VBytes;
            public long[] ONe0, ONe1, OBytes;
            public long[] GateNe0, GateNe1, GateBytes;
            public long[] UpNe0, UpNe1, UpBytes;
            public long[] DownNe0, DownNe1, DownBytes;
            public IntPtr[] RingK, RingV;

            /// <summary>Encoder: the [featureSize -&gt; hidden] projection and its norm.</summary>
            public IntPtr Fc;
            public int FcType;
            public long FcNe0, FcNe1, FcBytes;
            public IntPtr EncNorm;

            /// <summary>Drafter's final norm, plus the TARGET's embedding table and LM head.</summary>
            public IntPtr OutNorm;
            public IntPtr TokEmbd;
            public int TokEmbdType;
            public long TokEmbdNe0, TokEmbdNe1, TokEmbdBytes;
            public IntPtr LmHead;
            public int LmHeadType;
            public long LmHeadNe0, LmHeadNe1, LmHeadBytes;

            public int RingRows;
        }

        private DFlashArrays _dflashArrays;
        private bool _dflashFusedProbed;

        /// <summary>Highest ring position written + 1. The ring slot for position p is
        /// p % ringRows, so this is all that is needed to reconstruct which slot holds
        /// which position (and which slots were never written).</summary>
        private int _dflashRingFilled;

        /// <summary>Set when the fused kernels have written the ring on the device, so
        /// the host mirror is behind.</summary>
        private bool _dflashRingHostStale;

        /// <summary>
        /// Pull device-side ring writes back to the host copy. Only the per-op drafter
        /// walks the ring with host pointers, so this is a no-op on the fused path.
        /// </summary>
        private void EnsureDFlashRingHostSynchronized()
        {
            if (!_dflashRingHostStale || _dflashRingK == null)
                return;
            for (int l = 0; l < _dflash.NumLayers; l++)
            {
                SyncTensorHostCache(_dflashRingK[l]);
                SyncTensorHostCache(_dflashRingV[l]);
            }
            _dflashRingHostStale = false;
        }

        private int[] _dflashRingSlotPos;
        private int[] _dflashDraftIds;
        private float[] _dflashDraftConf;
        private long[] _dflashInjectIdx;
        private int[] _dflashInjectPos;

        /// <summary>GGML_TYPE_F32. The ring is allocated as Float32 (see the ring
        /// construction in the loader), and the kernel needs the ggml enum value.</summary>
        private const int GgmlTypeF32 = 0;

        private static readonly bool DFlashFusedEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_DFLASH_FUSED"), "0", StringComparison.Ordinal);

        /// <summary>
        /// Same backend restriction as the trunk kernel: the fused graphs go through
        /// GGML's gallocr and persistent-buffer paths that are exercised on CUDA and
        /// Vulkan in this repo. CPU keeps the per-op drafter, which is correct.
        /// </summary>
        private bool CanUseFusedDFlash =>
            DFlashFusedEnabled && _hasDFlash && !IsTensorParallel &&
            (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan);

        private unsafe void BuildDFlashArrays()
        {
            _dflashFusedProbed = true;
            _dflashArrays = null;
            if (!CanUseFusedDFlash)
                return;

            var cfg = _dflash;
            int n = cfg.NumLayers;
            var a = new DFlashArrays
            {
                AttnNorm = new IntPtr[n], QNorm = new IntPtr[n], KNorm = new IntPtr[n], FfnNorm = new IntPtr[n],
                Q = new IntPtr[n], K = new IntPtr[n], V = new IntPtr[n], O = new IntPtr[n],
                Gate = new IntPtr[n], Up = new IntPtr[n], Down = new IntPtr[n],
                QType = new int[n], KType = new int[n], VType = new int[n], OType = new int[n],
                GateType = new int[n], UpType = new int[n], DownType = new int[n],
                QNe0 = new long[n], QNe1 = new long[n], QBytes = new long[n],
                KNe0 = new long[n], KNe1 = new long[n], KBytes = new long[n],
                VNe0 = new long[n], VNe1 = new long[n], VBytes = new long[n],
                ONe0 = new long[n], ONe1 = new long[n], OBytes = new long[n],
                GateNe0 = new long[n], GateNe1 = new long[n], GateBytes = new long[n],
                UpNe0 = new long[n], UpNe1 = new long[n], UpBytes = new long[n],
                DownNe0 = new long[n], DownNe1 = new long[n], DownBytes = new long[n],
                RingK = new IntPtr[n], RingV = new IntPtr[n],
                RingRows = _dflashRingRows,
            };

            for (int l = 0; l < n; l++)
            {
                string[] wn = _dflashLayerNames[l];
                if (!TryQuant(wn[DfAttnQ], out var q) || !TryQuant(wn[DfAttnK], out var k)
                    || !TryQuant(wn[DfAttnV], out var v) || !TryQuant(wn[DfAttnOutput], out var o)
                    || !TryQuant(wn[DfFfnGate], out var gate) || !TryQuant(wn[DfFfnUp], out var up)
                    || !TryQuant(wn[DfFfnDown], out var down))
                {
                    Console.WriteLine($"  DFlash fused drafter disabled: layer {l} has a non-quantized projection.");
                    return;
                }
                if (!_weights.TryGetValue(wn[DfAttnNorm], out var an) || !_weights.TryGetValue(wn[DfAttnQNorm], out var qn)
                    || !_weights.TryGetValue(wn[DfAttnKNorm], out var kn) || !_weights.TryGetValue(wn[DfFfnNorm], out var fn))
                {
                    Console.WriteLine($"  DFlash fused drafter disabled: layer {l} is missing a norm weight.");
                    return;
                }

                a.AttnNorm[l] = (IntPtr)GetFloatPtr(an);
                a.QNorm[l] = (IntPtr)GetFloatPtr(qn);
                a.KNorm[l] = (IntPtr)GetFloatPtr(kn);
                a.FfnNorm[l] = (IntPtr)GetFloatPtr(fn);

                a.Q[l] = q.CacheKey; a.QType[l] = q.GgmlType; a.QNe0[l] = q.Ne0; a.QNe1[l] = q.Ne1; a.QBytes[l] = q.RawBytes;
                a.K[l] = k.CacheKey; a.KType[l] = k.GgmlType; a.KNe0[l] = k.Ne0; a.KNe1[l] = k.Ne1; a.KBytes[l] = k.RawBytes;
                a.V[l] = v.CacheKey; a.VType[l] = v.GgmlType; a.VNe0[l] = v.Ne0; a.VNe1[l] = v.Ne1; a.VBytes[l] = v.RawBytes;
                a.O[l] = o.CacheKey; a.OType[l] = o.GgmlType; a.ONe0[l] = o.Ne0; a.ONe1[l] = o.Ne1; a.OBytes[l] = o.RawBytes;
                a.Gate[l] = gate.CacheKey; a.GateType[l] = gate.GgmlType; a.GateNe0[l] = gate.Ne0; a.GateNe1[l] = gate.Ne1; a.GateBytes[l] = gate.RawBytes;
                a.Up[l] = up.CacheKey; a.UpType[l] = up.GgmlType; a.UpNe0[l] = up.Ne0; a.UpNe1[l] = up.Ne1; a.UpBytes[l] = up.RawBytes;
                a.Down[l] = down.CacheKey; a.DownType[l] = down.GgmlType; a.DownNe0[l] = down.Ne0; a.DownNe1[l] = down.Ne1; a.DownBytes[l] = down.RawBytes;

                a.RingK[l] = TensorComputePrimitives.GetStoragePointer(_dflashRingK[l]);
                a.RingV[l] = TensorComputePrimitives.GetStoragePointer(_dflashRingV[l]);
            }

            string fcName = DFlashConfig.WeightPrefix + "fc.weight";
            if (!TryQuant(fcName, out var fc))
            {
                Console.WriteLine("  DFlash fused drafter disabled: the encoder projection is not quantized.");
                return;
            }
            a.Fc = fc.CacheKey; a.FcType = fc.GgmlType; a.FcNe0 = fc.Ne0; a.FcNe1 = fc.Ne1; a.FcBytes = fc.RawBytes;

            if (!_weights.TryGetValue(DFlashConfig.WeightPrefix + "enc.output_norm.weight", out var encNorm)
                || !_weights.TryGetValue(DFlashConfig.WeightPrefix + "output_norm.weight", out var outNorm))
            {
                Console.WriteLine("  DFlash fused drafter disabled: a drafter norm weight is missing.");
                return;
            }
            a.EncNorm = (IntPtr)GetFloatPtr(encNorm);
            a.OutNorm = (IntPtr)GetFloatPtr(outNorm);

            // Both borrowed from the target, exactly as llama.cpp's dflash graph does
            // through cparams.ctx_other -- the drafter owns neither.
            if (!TryQuant("token_embd.weight", out var tok) || !TryQuant(TargetOutputWeightName, out var head))
            {
                Console.WriteLine("  DFlash fused drafter disabled: the target embedding/LM head is not quantized.");
                return;
            }
            a.TokEmbd = tok.CacheKey; a.TokEmbdType = tok.GgmlType; a.TokEmbdNe0 = tok.Ne0; a.TokEmbdNe1 = tok.Ne1; a.TokEmbdBytes = tok.RawBytes;
            a.LmHead = head.CacheKey; a.LmHeadType = head.GgmlType; a.LmHeadNe0 = head.Ne0; a.LmHeadNe1 = head.Ne1; a.LmHeadBytes = head.RawBytes;

            _dflashArrays = a;
            Console.WriteLine($"  DFlash fused drafter armed ({n} draft layers, ring {a.RingRows} rows, on-device top-1).");
        }

        private DFlashArrays GetDFlashArrays()
        {
            if (!_dflashFusedProbed)
                BuildDFlashArrays();
            if (_dflashArrays != null && _dflashArrays.RingRows != _dflashRingRows)
            {
                GgmlBasicOps.DFlashResetCaches();
                BuildDFlashArrays();
            }
            return _dflashArrays;
        }

        /// <summary>Drops the persistent DFlash graphs (ring reallocation or KV reset).</summary>
        private void ResetFusedDFlashCaches()
        {
            if (_dflashArrays != null)
                GgmlBasicOps.DFlashResetCaches();
            _dflashRingFilled = 0;
        }

        /// <summary>
        /// Which absolute position each ring slot currently holds, or -1 for a slot
        /// that was never written. The kernel reads the WHOLE ring every step so its
        /// graph shape stays constant; this map is what lets its mask express liveness,
        /// causality and the sliding window without changing the shape.
        /// </summary>
        private int[] BuildRingSlotPositions()
        {
            int rows = _dflashRingRows;
            if (_dflashRingSlotPos == null || _dflashRingSlotPos.Length != rows)
                _dflashRingSlotPos = new int[rows];
            var map = _dflashRingSlotPos;
            for (int i = 0; i < rows; i++)
                map[i] = -1;
            int lo = Math.Max(0, _dflashRingFilled - rows);
            for (int p = lo; p < _dflashRingFilled; p++)
                map[p % rows] = p;
            return map;
        }

        /// <summary>
        /// PASS A+B in one graph: fc -&gt; RMSNorm -&gt; per layer {k/v projection, k head
        /// norm, NeoX RoPE, ring scatter}. Returns false when the fused path is not
        /// available, in which case the caller runs the per-op version.
        /// </summary>
        private bool TryFusedDFlashInject(float[] hRows, int rowOffset, int n, int startPos)
        {
            var a = GetDFlashArrays();
            if (a == null)
                return false;

            var cfg = _dflash;
            int feat = cfg.FeatureSize;

            // The ring drops rows older than its capacity, so only the last min(n, rows)
            // survive; writing the earlier ones would just be overwritten in place.
            int keep = Math.Min(n, _dflashRingRows);
            int skip = n - keep;

            float[] rows = hRows;
            if (rowOffset + skip != 0)
            {
                rows = new float[(long)keep * feat];
                Array.Copy(hRows, (long)(rowOffset + skip) * feat, rows, 0, (long)keep * feat);
            }

            if (_dflashInjectIdx == null || _dflashInjectIdx.Length < keep)
            {
                _dflashInjectIdx = new long[keep];
                _dflashInjectPos = new int[keep];
            }
            for (int i = 0; i < keep; i++)
            {
                int p = startPos + skip + i;
                _dflashInjectIdx[i] = p % _dflashRingRows;
                _dflashInjectPos[i] = p;
            }

            bool ok = GgmlBasicOps.DFlashInject(
                rows, feat, keep, _dflashInjectIdx, _dflashInjectPos,
                cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumKVHeads, _dflashRingRows,
                cfg.Eps, cfg.RopeBase, 1f,
                a.Fc, a.FcType, a.FcNe0, a.FcNe1, a.FcBytes, a.EncNorm,
                a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                a.KNorm, a.RingK, a.RingV, GgmlTypeF32);
            if (!ok)
                return false;

            _dflashRingFilled = Math.Max(_dflashRingFilled, startPos + n);
            _dflashRingHostStale = true;
            return true;
        }

        /// <summary>
        /// PASS C in one graph: [anchor, MASK x (b-1)] through the draft blocks, then
        /// the target's LM head, softmax, and an on-device argmax. Returns the number
        /// of drafted tokens, or -1 when the fused path is unavailable.
        /// </summary>
        private int TryFusedDFlashDraftBlock(int anchorToken, int position, int b, int[] draftOut, float[] confOut)
        {
            var a = GetDFlashArrays();
            if (a == null)
                return -1;

            var cfg = _dflash;
            if (_dflashDraftIds == null || _dflashDraftIds.Length < b)
            {
                _dflashDraftIds = new int[b];
                _dflashDraftConf = new float[b];
            }
            int[] ids = new int[b];
            int[] positions = new int[b];
            for (int i = 0; i < b; i++)
            {
                ids[i] = i == 0 ? anchorToken : cfg.MaskTokenId;
                positions[i] = position + i;
            }

            bool ok = GgmlBasicOps.DFlashDraftBlock(
                ids, b, positions,
                cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads, _dflashRingRows,
                cfg.Eps, cfg.RopeBase, 1f, 1f / MathF.Sqrt(cfg.HeadDim),
                BuildRingSlotPositions(), cfg.SlidingWindow,
                a.AttnNorm,
                a.Q, a.QType, a.QNe0, a.QNe1, a.QBytes,
                a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                a.QNorm, a.KNorm,
                a.O, a.OType, a.ONe0, a.ONe1, a.OBytes,
                a.FfnNorm,
                a.Gate, a.GateType, a.GateNe0, a.GateNe1, a.GateBytes,
                a.Up, a.UpType, a.UpNe0, a.UpNe1, a.UpBytes,
                a.Down, a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                a.RingK, a.RingV, GgmlTypeF32,
                a.OutNorm,
                a.TokEmbd, a.TokEmbdType, a.TokEmbdNe0, a.TokEmbdNe1, a.TokEmbdBytes,
                a.LmHead, a.LmHeadType, a.LmHeadNe0, a.LmHeadNe1, a.LmHeadBytes,
                Config.VocabSize, _dflashDraftIds, _dflashDraftConf);
            if (!ok)
                return -1;

            // Row 0 is the anchor's own prediction; plain DFlash discards it.
            int drafted = b - 1;
            for (int i = 0; i < drafted; i++)
            {
                draftOut[i] = _dflashDraftIds[i + 1];
                if (confOut != null && i < confOut.Length)
                    confOut[i] = _dflashDraftConf[i + 1];
            }
            return drafted;
        }
    }
}
