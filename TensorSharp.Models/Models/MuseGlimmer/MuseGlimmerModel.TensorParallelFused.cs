// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ---------------------------------------------------------------------------
// Fused tensor-parallel forward for Muse-Glimmer on the GGML GPU backends.
//
// The per-op TP forward in MuseGlimmerModel.TensorParallel.cs runs one native
// op at a time, and every GGML op submits a graph, synchronizes and copies its
// result back to host memory. A 52-layer Muse-Glimmer step is ~600 of those,
// multiplied by the rank count - which is why the single-GPU path was given a
// whole-model kernel in the first place (per-op decode measured 4.3 tok/s
// against 22 for llama.cpp on the same card).
//
// This path builds the SAME whole-model graph the single-GPU forward uses,
// once per rank over that rank's shards, and cuts it at the two places per
// layer where a row-parallel projection leaves a partial sum: attn_output and
// ffn_down. The native driver then submits segment k on every GPU
// asynchronously, AllReduces the partials in VRAM and moves on - llama.cpp's
// --split-mode tensor schedule. Activations never cross the PCIe bus and the
// host issues 2*L graph launches per step instead of ~600 op dispatches.
//
// Everything outside those two projections is replicated work: every rank runs
// the norms, the QK norms, RoPE and the attention gate on identical values and
// gets identical answers. Only rank 0 folds the final norm and the LM head -
// on the 30B that head is ~1.1 GB, the largest tensor left once the layers are
// sharded, and replicating it would hand back much of what TP just saved.
// TS_MUSE_GLIMMER_TP_FUSED=0 forces the per-op TP path for A/B.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class MuseGlimmerModel : ModelBase
    {
        /// <summary>One pointer/shape table per rank, holding that rank's weight
        /// shards and KV cache pointers. Null when the fused TP path is off.</summary>
        private FusedArrays[] _tpFusedArrays;
        private IntPtr[] _tpFusedPlans;
        private bool _tpFusedProbed;
        private bool _tpFusedLogged;
        private int _tpFusedFailures;
        private float[] _tpFusedLogits;

        /// <summary>
        /// Set once a fused TP step has written KV rows straight into the
        /// device-resident cache copies.
        ///
        /// It is deliberately NOT paired with
        /// <see cref="ModelBase.InvalidateTensorDeviceCache"/> - see trap 2 in
        /// TENSOR_PARALLELISM_PLAN.md. Invalidating drops the device copy on every
        /// rank, so the next layer re-uploads the whole cache AND re-uploads the
        /// STALE host copy, which the fused kernel never wrote (measured
        /// 15.6 -> 11.4 tok/s). The device copy stays authoritative and is pulled
        /// back only when something needs to read the cache through host pointers.
        /// </summary>
        private bool _tpKvDeviceDirty;

        private static readonly bool TpFusedForwardEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_TP_FUSED"), "0", StringComparison.Ordinal);

        /// <summary>
        /// True when the fused whole-model graph can be built per rank.
        ///
        /// Metal is excluded (unlike <see cref="CanUseFusedForward"/>): a rank is
        /// one ggml backend on one GPU, and the device collective the plan driver
        /// reduces with exists on ggml-cuda / ggml-vulkan only. The multi-node
        /// layout is excluded too - the driver submits every rank from one
        /// thread, so it is a single-process facility.
        ///
        /// This also gates the sliding-window RING in ResolveSwaRows: the ring
        /// layout is readable by the fused kernels only, so arming it under TP is
        /// exactly as safe (and as unsafe) as arming it on one GPU.
        /// </summary>
        internal bool CanUseTpFusedForward =>
            FusedForwardEnabled && TpFusedForwardEnabled && IsTensorParallel && IsGgmlBackend &&
            (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan) &&
            GlobalTpDegree == TpDegree &&
            GgmlBasicOps.TensorParallelFusedAvailable(TpDegree);

        /// <summary>
        /// Build (or rebuild) the per-rank pointer tables. Called after weight
        /// sharding, the per-rank device preload and
        /// <see cref="InitTpKVCache"/>, so every shard already has a device cache
        /// key and every cache tensor its final address.
        /// </summary>
        private unsafe void BuildMuseGlimmerTpDecodeArrays()
        {
            _tpFusedProbed = true;
            _tpFusedArrays = null;
            _tpFusedPlans = null;
            if (!CanUseTpFusedForward || _tpKvCacheK == null)
                return;

            int n = Config.NumLayers;
            int tp = TpDegree;
            var perRank = new FusedArrays[tp];

            for (int r = 0; r < tp; r++)
            {
                var a = NewTpFusedArrays(n);
                // Stamped so GetTpFusedArrays can spot a cache grow; a stale
                // stamp (0) would rebuild the tables on every single step and
                // throw away the parked graphs with them.
                a.CacheCapacity = _kvCacheCapacity;

                for (int l = 0; l < n; l++)
                {
                    string[] wn = _layerWeightNames[l];

                    // Read the projections from the SHARDED table. Sharding moves
                    // these entries OUT of _quantWeights, which is why the
                    // single-GPU BuildFusedArrays reports "layer 0 has a
                    // non-quantized projection" and disarms itself under TP.
                    if (!TryTpQuant(wn[1], r, out var q) || !TryTpQuant(wn[2], r, out var k)
                        || !TryTpQuant(wn[3], r, out var v) || !TryTpQuant(wn[4], r, out var gate)
                        || !TryTpQuant(wn[7], r, out var o) || !TryTpQuant(wn[10], r, out var gu)
                        || !TryTpQuant(wn[11], r, out var down))
                    {
                        Console.WriteLine($"  Muse-Glimmer fused TP forward disabled: layer {l} has a non-quantized shard on rank {r}.");
                        return;
                    }

                    // The six norms are REPLICATED, so this is one host pointer
                    // for every rank; the native per-rank buffer cache is keyed
                    // by (rank, pointer) and gives each GPU its own device copy.
                    if (!_weights.TryGetValue(wn[0], out var attnNorm) || !_weights.TryGetValue(wn[5], out var qNorm)
                        || !_weights.TryGetValue(wn[6], out var kNorm) || !_weights.TryGetValue(wn[8], out var postAttn)
                        || !_weights.TryGetValue(wn[9], out var ffnNorm) || !_weights.TryGetValue(wn[12], out var postFfn))
                    {
                        Console.WriteLine($"  Muse-Glimmer fused TP forward disabled: layer {l} is missing a norm weight.");
                        return;
                    }

                    a.AttnNorm[l] = (IntPtr)GetFloatPtr(attnNorm);
                    a.QNorm[l] = (IntPtr)GetFloatPtr(qNorm);
                    a.KNorm[l] = (IntPtr)GetFloatPtr(kNorm);
                    a.PostAttnNorm[l] = (IntPtr)GetFloatPtr(postAttn);
                    a.FfnNorm[l] = (IntPtr)GetFloatPtr(ffnNorm);
                    a.PostFfnNorm[l] = (IntPtr)GetFloatPtr(postFfn);

                    // CacheKey, not Data: after the per-rank preload the device
                    // copy is addressed by an opaque handle and Data would miss it.
                    a.Q[l] = q.CacheKey; a.QType[l] = q.GgmlType; a.QNe0[l] = q.Ne0; a.QNe1[l] = q.Ne1; a.QBytes[l] = q.RawBytes;
                    a.K[l] = k.CacheKey; a.KType[l] = k.GgmlType; a.KNe0[l] = k.Ne0; a.KNe1[l] = k.Ne1; a.KBytes[l] = k.RawBytes;
                    a.V[l] = v.CacheKey; a.VType[l] = v.GgmlType; a.VNe0[l] = v.Ne0; a.VNe1[l] = v.Ne1; a.VBytes[l] = v.RawBytes;
                    a.Gate[l] = gate.CacheKey; a.GateType[l] = gate.GgmlType; a.GateNe0[l] = gate.Ne0; a.GateNe1[l] = gate.Ne1; a.GateBytes[l] = gate.RawBytes;
                    a.O[l] = o.CacheKey; a.OType[l] = o.GgmlType; a.ONe0[l] = o.Ne0; a.ONe1[l] = o.Ne1; a.OBytes[l] = o.RawBytes;
                    a.Gu[l] = gu.CacheKey; a.GuType[l] = gu.GgmlType; a.GuNe0[l] = gu.Ne0; a.GuNe1[l] = gu.Ne1; a.GuBytes[l] = gu.RawBytes;
                    a.Down[l] = down.CacheKey; a.DownType[l] = down.GgmlType; a.DownNe0[l] = down.Ne0; a.DownNe1[l] = down.Ne1; a.DownBytes[l] = down.RawBytes;

                    a.KCache[l] = TensorComputePrimitives.GetStoragePointer(_tpKvCacheK[l][r]);
                    a.VCache[l] = TensorComputePrimitives.GetStoragePointer(_tpKvCacheV[l][r]);
                    a.IsSwa[l] = _isSwaLayer[l] ? 1 : 0;
                }

                // Only rank 0 folds the output head. Every other rank stops at
                // the trunk, so its graph never binds the 1.1 GB LM head.
                if (r == 0)
                {
                    string headName = _hasTiedOutput ? "token_embd.weight" : "output.weight";
                    if (!_quantWeights.TryGetValue(headName, out var head) || head.CacheKey == IntPtr.Zero
                        || !_weights.TryGetValue("output_norm.weight", out var fnorm))
                    {
                        Console.WriteLine("  Muse-Glimmer fused TP forward disabled: no quantized LM head / final norm to fold.");
                        return;
                    }
                    a.LmHead = head.CacheKey;
                    a.LmHeadType = head.GgmlType;
                    a.LmHeadNe0 = head.Ne0;
                    a.LmHeadNe1 = head.Ne1;
                    a.LmHeadBytes = head.RawBytes;
                    a.FinalNorm = (IntPtr)GetFloatPtr(fnorm);
                }

                // TokEmbd stays IntPtr.Zero on EVERY rank, including rank 0. The
                // in-graph embedding gather would pin the whole 202K-row table on
                // each GPU, which is precisely the memory tensor parallelism was
                // bought to save; the host-side Embedding() gather costs two
                // dispatches per step instead.
                perRank[r] = a;
            }

            _tpFusedArrays = perRank;
            _tpFusedPlans = new IntPtr[tp];
            Console.WriteLine($"  Muse-Glimmer fused tensor-parallel forward armed ({tp} GPUs, {n} layers, " +
                $"{2 * n} AllReduce points/step, LM head folded on rank 0).");
        }

        private static FusedArrays NewTpFusedArrays(int n) => new FusedArrays
        {
            AttnNorm = new IntPtr[n], Q = new IntPtr[n], K = new IntPtr[n], V = new IntPtr[n],
            Gate = new IntPtr[n], QNorm = new IntPtr[n], KNorm = new IntPtr[n], O = new IntPtr[n],
            PostAttnNorm = new IntPtr[n], FfnNorm = new IntPtr[n], Gu = new IntPtr[n],
            Down = new IntPtr[n], PostFfnNorm = new IntPtr[n],
            KCache = new IntPtr[n], VCache = new IntPtr[n],
            IsSwa = new int[n],
            QType = new int[n], KType = new int[n], VType = new int[n], GateType = new int[n],
            OType = new int[n], GuType = new int[n], DownType = new int[n],
            QNe0 = new long[n], QNe1 = new long[n], QBytes = new long[n],
            KNe0 = new long[n], KNe1 = new long[n], KBytes = new long[n],
            VNe0 = new long[n], VNe1 = new long[n], VBytes = new long[n],
            GateNe0 = new long[n], GateNe1 = new long[n], GateBytes = new long[n],
            ONe0 = new long[n], ONe1 = new long[n], OBytes = new long[n],
            GuNe0 = new long[n], GuNe1 = new long[n], GuBytes = new long[n],
            DownNe0 = new long[n], DownNe1 = new long[n], DownBytes = new long[n],
        };

        private bool TryTpQuant(string name, int rank, out QuantizedWeight qw)
        {
            qw = null;
            if (!_tpQuantWeights.TryGetValue(name, out var shards) || shards == null || rank >= shards.Length)
                return false;
            qw = shards[rank];
            return qw != null && qw.CacheKey != IntPtr.Zero;
        }

        /// <summary>
        /// The per-rank tables, rebuilt when the cache capacity moved. Mirrors
        /// <see cref="GetFusedArrays"/>: a grow reallocates every KV tensor, so
        /// both the captured pointers and the kernel's baked cache extent are
        /// stale, and replaying a parked graph against freed memory hangs.
        /// </summary>
        private FusedArrays[] GetTpFusedArrays()
        {
            if (!_tpFusedProbed)
                BuildMuseGlimmerTpDecodeArrays();
            if (_tpFusedArrays != null && _tpFusedArrays[0].CacheCapacity != _kvCacheCapacity)
            {
                ResetMuseGlimmerTpGraphs();
                BuildMuseGlimmerTpDecodeArrays();
            }
            return _tpFusedArrays;
        }

        /// <summary>
        /// Run the whole model as one graph per rank. <paramref name="hidden"/>
        /// is [nTokens, hidden] and already embedded + input-normed on rank 0;
        /// the native side gives each rank its own device copy keyed by that host
        /// pointer (trap 4: the replicated activation is one buffer PER RANK, so
        /// a rank must never read another's).
        ///
        /// Returns the last row's logits, or null when any rank declines - in
        /// which case NOTHING was executed (the plans are built, not run) and the
        /// caller can safely re-run the step on the per-op TP path.
        /// </summary>
        private unsafe float[] TryFusedForwardTP(Tensor hidden, int seqLen, int startPos)
        {
            var arrays = GetTpFusedArrays();
            if (arrays == null)
                return null;

            int tp = TpDegree;
            long needed = Config.VocabSize;
            if (_tpFusedLogits == null || _tpFusedLogits.LongLength < needed)
                _tpFusedLogits = new float[needed];
            float[] target = _tpFusedLogits;

            IntPtr hiddenPtr = (IntPtr)GetFloatPtr(hidden);
            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            bool ok = true;

            long t0 = Stopwatch.GetTimestamp();
            try
            {
                fixed (float* logitsPtr = target)
                {
                    for (int r = 0; r < tp; r++)
                    {
                        var a = arrays[r];
                        GgmlBasicOps.SetActiveRank(r);
                        planSlot[0] = IntPtr.Zero;

                        // tpDegree > 1 with a non-null tpPlanOut means "build the
                        // graph for the CURRENT active rank and park it", not
                        // "execute": the ranks only meet in the driver below.
                        bool built = GgmlBasicOps.MuseGlimmerModelForward(
                            hiddenPtr, Config.HiddenSize, seqLen, Config.NumLayers,
                            a.AttnNorm, a.Q, a.K, a.V, a.Gate, a.QNorm, a.KNorm, a.O,
                            a.PostAttnNorm, a.FfnNorm, a.Gu, a.Down, a.PostFfnNorm,
                            a.KCache, a.VCache, a.IsSwa,
                            a.QType, a.QNe0, a.QNe1, a.QBytes,
                            a.KType, a.KNe0, a.KNe1, a.KBytes,
                            a.VType, a.VNe0, a.VNe1, a.VBytes,
                            a.GateType, a.GateNe0, a.GateNe1, a.GateBytes,
                            a.OType, a.ONe0, a.ONe1, a.OBytes,
                            a.GuType, a.GuNe0, a.GuNe1, a.GuBytes,
                            a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                            // Per-rank head counts: the column-parallel shards
                            // produce this rank's heads only, and its KV cache
                            // holds just those.
                            Config.NumHeads / tp, Config.NumKVHeads / tp, _headDim,
                            _kvCacheCapacity, _kvSwaRows, startPos, _slidingWindow,
                            Config.Eps, PostNormEps, Config.RopeBase, 1.0f / Config.RopeScale,
                            1f / MathF.Sqrt(_headDim), (int)_kvCacheDtype.GgmlType(),
                            r == 0 ? (IntPtr)logitsPtr : IntPtr.Zero, r == 0 ? Config.VocabSize : 0,
                            a.LmHead, a.LmHeadType, a.LmHeadNe0, a.LmHeadNe1, a.LmHeadBytes,
                            a.FinalNorm, _finalLogitScale, _finalLogitSoftcap,
                            IntPtr.Zero, null, 0,
                            // No in-graph embedding under TP (see BuildMuseGlimmerTpDecodeArrays).
                            IntPtr.Zero, 0, 0, 0, 0, null,
                            0,
                            tpDegree: tp, tpPlanOut: planSlot);

                        if (!built || planSlot[0] == IntPtr.Zero)
                        {
                            ok = false;
                            break;
                        }
                        _tpFusedPlans[r] = planSlot[0];
                    }

                    if (ok)
                        GgmlBasicOps.TensorParallelExecutePlans(_tpFusedPlans);
                }
            }
            catch (InvalidOperationException)
            {
                // The native side declined this shape (an unusually wide prefill
                // chunk, a window it cannot express). Nothing ran.
                ok = false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }
            _linearTicks += Stopwatch.GetTimestamp() - t0;

            if (!ok)
            {
                // Ranks that DID build left a parked transient graph behind; drop
                // them now rather than leaking one per declined step.
                ResetMuseGlimmerTpGraphs();

                // The decline is not sticky: a shape-specific refusal must not
                // cost every later decode token the fast path. Only a run of
                // consecutive failures disarms.
                if (++_tpFusedFailures <= 3)
                {
                    Console.WriteLine($"  Muse-Glimmer fused TP forward declined (seqLen={seqLen} startPos={startPos} " +
                        $"cap={_kvCacheCapacity}): {GgmlBasicOps.LastNativeError()}; falling back to the per-op TP path.");
                }
                if (_tpFusedFailures >= 16)
                {
                    Console.WriteLine("  Muse-Glimmer fused TP forward disabled after repeated failures.");
                    _tpFusedArrays = null;
                }
                return null;
            }
            _tpFusedFailures = 0;

            // The kernels wrote K/V into the DEVICE copies. Do not invalidate
            // them here - see the _tpKvDeviceDirty remarks above.
            _tpKvDeviceDirty = true;

            if (!_tpFusedLogged)
            {
                _tpFusedLogged = true;
                Console.WriteLine($"  Muse-Glimmer fused tensor-parallel forward active ({tp} GPUs, " +
                    $"{2 * Config.NumLayers} AllReduce points/step).");
            }
            return target;
        }

        /// <summary>
        /// Pull the device-resident KV caches back to host memory. The fused TP
        /// path writes KV rows straight into the device copies; the per-op TP
        /// forward reads the same tensors through host pointers, so without this
        /// a fallback step (or a KV snapshot) would attend over a cache missing
        /// every token the fused path produced.
        /// </summary>
        private void SyncMuseGlimmerTpKvCacheToHost()
        {
            if (!_tpKvDeviceDirty || _tpKvCacheK == null)
                return;
            _tpKvDeviceDirty = false;

            // Sync each BUFFER once: a rank's K and V are distinct tensors, but
            // dedup by storage pointer keeps this honest if two layers ever come
            // to share one (and makes the cost O(buffers), not O(layers*ranks)).
            var seen = new HashSet<IntPtr>();
            for (int l = 0; l < _tpKvCacheK.Length; l++)
            {
                if (_tpKvCacheK[l] == null) continue;
                for (int r = 0; r < _tpKvCacheK[l].Length; r++)
                {
                    SyncOneTpKvTensor(_tpKvCacheK[l][r], seen);
                    SyncOneTpKvTensor(_tpKvCacheV[l][r], seen);
                }
            }
        }

        private void SyncOneTpKvTensor(Tensor t, HashSet<IntPtr> seen)
        {
            if (t == null)
                return;
            IntPtr ptr = TensorComputePrimitives.GetStoragePointer(t);
            if (ptr == IntPtr.Zero || !seen.Add(ptr))
                return;
            SyncTensorHostCache(t);
        }

        /// <summary>
        /// Free the parked per-rank transient graphs. They bake the KV-cache
        /// device addresses and the scratch-pool layout, both of which a cache
        /// reset, grow or truncation moves; replaying a stale one hangs the GPU.
        /// </summary>
        private void ResetMuseGlimmerTpGraphs()
        {
            if (IsGgmlBackend && IsTensorParallel)
                GgmlBasicOps.MuseGlimmerReleaseTpGraphs();
        }
    }
}
