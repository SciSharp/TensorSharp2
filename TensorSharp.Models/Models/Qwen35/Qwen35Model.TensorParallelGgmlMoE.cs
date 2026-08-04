// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ============================================================================
// Qwen35Model.TensorParallelGgmlMoE.cs
//
// Expert-parallel MoE for the Qwen3.5/3.6 tensor-parallel path on the GGML
// backends.
//
// The direct-CUDA TP path slices INSIDE each expert (gate/up column-parallel,
// down row-parallel) and drives the result with its own fused on-device MoE
// kernels. That split cannot work on GGML: every rank ends up holding a piece
// of every expert, so the layer no longer fits a single ggml_mul_mat_id
// dispatch and degenerates into a per-(token, expert) loop. With 256 experts
// and 8 routed per token over 40 MoE layers that is hundreds of thousands of
// tiny matmuls per prefill.
//
// Whole experts partition cleanly instead. The stacked expert tensor has the
// expert index as its outer dimension, so rank r's share is a contiguous byte
// range — a zero-copy view — and each rank runs the same batched kernel the
// single-GPU path uses. Each rank sums only the experts it owns, so the
// AllReduce the block already performs is exactly the right recombination.
//
// The shared expert stays Megatron-split (column/row parallel) because it is
// dense: every token uses it, so slicing it inside is both correct and the only
// way it contributes to the memory split.
// ============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // Per-rank slices of the stacked expert weights, [layer][rank].
        private StackedExpertWeights[][] _tpStackedGate;
        private StackedExpertWeights[][] _tpStackedUp;
        private StackedExpertWeights[][] _tpStackedDown;
        private int _tpExpertsPerRank;

        /// <summary>
        /// True when the routed experts are partitioned by whole expert rather
        /// than sliced inside each expert.
        /// </summary>
        private bool UsesExpertParallelMoE => _tpStackedGate != null;

        /// <summary>
        /// Whether this model/backend combination will take the expert-parallel
        /// path. Evaluated before sharding (from <c>ValidateTpConstraints</c>) as
        /// well as during it, so the two must agree — hence one predicate.
        /// </summary>
        private bool CanUseGgmlExpertParallelMoE()
        {
            int tp = GlobalTpDegree;
            if (!IsGgmlBackend || _numExperts <= 0 || tp <= 1 || (_numExperts % tp) != 0)
                return false;
            // The kernel neutralises a foreign route by pointing it at an unused
            // LOCAL expert id, which requires there to be one.
            if (_numExperts / tp < _numExpertsUsed)
                return false;
            if (_layerStackedGate == null || _layerStackedUp == null || _layerStackedDown == null)
                return false;

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isMoeLayer == null || l >= _isMoeLayer.Length || !_isMoeLayer[l])
                    continue;
                if (_layerStackedGate[l] == null || _layerStackedUp[l] == null || _layerStackedDown[l] == null)
                    return false;
                if (_layerStackedGate[l].NumExperts != _numExperts)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Build the per-rank whole-expert slices. Returns false when the model
        /// did not expose usable stacked expert weights, leaving the caller on
        /// the per-expert sharding path.
        /// </summary>
        private bool BuildQwen35ExpertParallelShards()
        {
            if (!CanUseGgmlExpertParallelMoE())
                return false;

            int globalTp = GlobalTpDegree;
            int localTp = TpDegree;
            int rankOffset = TpRankOffset;
            int perRank = _numExperts / globalTp;
            int n = Config.NumLayers;

            var gate = new StackedExpertWeights[n][];
            var up = new StackedExpertWeights[n][];
            var down = new StackedExpertWeights[n][];

            for (int layer = 0; layer < n; layer++)
            {
                if (_isMoeLayer == null || !_isMoeLayer[layer])
                    continue;

                gate[layer] = new StackedExpertWeights[localTp];
                up[layer] = new StackedExpertWeights[localTp];
                down[layer] = new StackedExpertWeights[localTp];

                for (int lr = 0; lr < localTp; lr++)
                {
                    int firstExpert = (rankOffset + lr) * perRank;
                    gate[layer][lr] = SliceStackedExperts(_layerStackedGate[layer], firstExpert, perRank);
                    up[layer][lr] = SliceStackedExperts(_layerStackedUp[layer], firstExpert, perRank);
                    down[layer][lr] = SliceStackedExperts(_layerStackedDown[layer], firstExpert, perRank);
                }
            }

            _tpStackedGate = gate;
            _tpStackedUp = up;
            _tpStackedDown = down;
            _tpExpertsPerRank = perRank;

            Console.WriteLine(
                $"  Qwen3.5 MoE: expert-parallel across {globalTp} GPU(s), {perRank} of {_numExperts} experts per GPU " +
                "(one batched ggml_mul_mat_id dispatch per projection per layer).");
            return true;
        }

        /// <summary>
        /// Zero-copy view of <paramref name="count"/> consecutive experts. The
        /// expert index is the stacked tensor's outer dimension, so this is a
        /// byte offset — no copy, and each rank's device cache holds only its own
        /// slice.
        /// </summary>
        private static StackedExpertWeights SliceStackedExperts(StackedExpertWeights src, int firstExpert, int count)
        {
            long perExpertBytes = src.PerExpertRawBytes;
            return new StackedExpertWeights(
                new IntPtr(src.Data.ToInt64() + firstExpert * perExpertBytes),
                src.GgmlType,
                src.PerExpertNe0,
                src.PerExpertNe1,
                count,
                perExpertBytes * count,
                isExternalView: true,
                ownerToken: src,
                ownedBuffer: IntPtr.Zero);
        }

        /// <summary>
        /// Make each rank's expert slice device-resident at load time. Without
        /// this the first forward pays the whole upload — on a 35B that is
        /// ~8 GB per rank and looks like a hang.
        /// </summary>
        protected override unsafe void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
        {
            // Runs inside the per-rank preload fan-out: the calling thread is
            // pinned to this rank's GPU, and every rank uploads concurrently.
            if (rank == 0)
            {
                // The MoE routers are F32 and replicated, so they are not in
                // _tpQuantWeights and the generic loop skips them. Preloading them
                // here (rather than letting the first forward create the device copy)
                // keeps every weight's residency decided at load time — a weight that
                // first becomes resident mid-forward competes with the activation
                // allocations for whatever VRAM is left, and on this model that cost
                // the LM head its own residency.
                for (int layer = 0; layer < Config.NumLayers; layer++)
                {
                    if (_isMoeLayer == null || layer >= _isMoeLayer.Length || !_isMoeLayer[layer])
                        continue;
                    if (!_weights.TryGetValue(_ffnGateInpKey[layer], out var router) || router == null
                        || router.DimensionCount != 2 || !router.IsContiguous())
                        continue;

                    IntPtr data = (IntPtr)GetFloatPtr(router);
                    long bytes = router.ElementCount() * sizeof(float);
                    if (GgmlBasicOps.PreloadQuantizedWeight(data, data, 0, router.Sizes[1], router.Sizes[0], bytes))
                    {
                        bytesPerRank[0] += bytes;
                        countPerRank[0]++;
                    }
                }
            }

            if (!UsesExpertParallelMoE)
                return;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isMoeLayer == null || !_isMoeLayer[layer] || _tpStackedGate[layer] == null)
                    continue;
                // --n-cpu-moe: this layer's experts are multiplied on the host
                // out of the GGUF mmap. Not uploading them IS the VRAM saving.
                if (MoeCpuOffloadConfig.IsLayerOnCpu(layer))
                    continue;
                PreloadStackedShard(_tpStackedGate[layer][rank], bytesPerRank, countPerRank, rank);
                PreloadStackedShard(_tpStackedUp[layer][rank], bytesPerRank, countPerRank, rank);
                PreloadStackedShard(_tpStackedDown[layer][rank], bytesPerRank, countPerRank, rank);
            }
        }

        private static void PreloadStackedShard(StackedExpertWeights w, long[] bytesPerRank, int[] countPerRank, int rank)
        {
            if (w == null) return;
            // The MoE kernel looks the buffer up by the data pointer, so the
            // cache key must be that same pointer.
            if (GgmlBasicOps.PreloadQuantizedWeight(
                    w.Data, w.Data, w.GgmlType, w.PerExpertNe0, w.PerExpertNe1 * w.NumExperts, w.TotalRawBytes))
            {
                bytesPerRank[rank] += w.TotalRawBytes;
                countPerRank[rank]++;
            }
        }

        /// <summary>
        /// Run one MoE layer expert-parallel: each rank dispatches the batched
        /// <c>ggml_mul_mat_id</c> kernel over the experts it owns, then adds its
        /// slice of the (Megatron-split) shared expert. Every rank's result is a
        /// PARTIAL that the caller's AllReduce completes.
        ///
        /// Returns false if the shapes don't suit the kernel, leaving the caller
        /// on the per-expert path.
        /// </summary>
        private unsafe bool TryQwen35MoEExpertParallel(
            Tensor[] moeInput,
            Tensor[] results,
            Tensor routerData,
            bool routeRowsAreLogits,
            int layer,
            int seqLen)
        {
            var gateShards = _tpStackedGate?[layer];
            var upShards = _tpStackedUp?[layer];
            var downShards = _tpStackedDown?[layer];
            if (gateShards == null || upShards == null || downShards == null)
                return false;

            int tp = TpDegree;
            int rankOffset = TpRankOffset;
            int hiddenSize = Config.HiddenSize;
            int nUsed = _numExpertsUsed;
            int perRank = _tpExpertsPerRank;
            int intermediate = (int)gateShards[0].PerExpertNe1;

            if (gateShards[0].PerExpertNe0 != hiddenSize || upShards[0].PerExpertNe0 != hiddenSize
                || upShards[0].PerExpertNe1 != intermediate
                || downShards[0].PerExpertNe0 != intermediate || downShards[0].PerExpertNe1 != hiddenSize)
                return false;

            // Host top-K routing once, from the (identical on every rank) router
            // logits, then a per-rank dense route table. Built on the calling
            // thread: the rank workers must not race on shared scratch.
            int totalRoutes = seqLen * nUsed;
            var globalExperts = new int[totalRoutes];
            var globalWeights = new float[totalRoutes];
            var tokTop = new int[nUsed];
            var tokW = new float[nUsed];
            float* routePtr = GetFloatPtr(routerData);
            for (int s = 0; s < seqLen; s++)
            {
                SelectTopKRouteWeights(routePtr + (long)s * _numExperts, routeRowsAreLogits, tokTop, tokW);
                for (int k = 0; k < nUsed; k++)
                {
                    globalExperts[s * nUsed + k] = tokTop[k];
                    globalWeights[s * nUsed + k] = tokW[k];
                }
            }

            var localExperts = new int[tp][];
            var localWeights = new float[tp][];
            for (int r = 0; r < tp; r++)
            {
                int first = (rankOffset + r) * perRank;
                int last = first + perRank;
                var ids = new int[totalRoutes];
                var wts = new float[totalRoutes];
                var taken = new bool[perRank];

                for (int s = 0; s < seqLen; s++)
                {
                    int baseIdx = s * nUsed;
                    Array.Clear(taken, 0, perRank);

                    // Own routes first, so the filler pass below can see which
                    // local experts this token already uses.
                    for (int k = 0; k < nUsed; k++)
                    {
                        int i = baseIdx + k;
                        int e = globalExperts[i];
                        if (e >= first && e < last)
                        {
                            int local = e - first;
                            ids[i] = local;
                            taken[local] = true;
                            wts[i] = globalWeights[i];
                        }
                        else
                        {
                            ids[i] = -1;
                            wts[i] = 0f;
                        }
                    }

                    // A route owned by another rank still occupies a slot in the
                    // dense route table; it is neutralised with weight 0. It must
                    // get a DISTINCT local expert id: real top-k routing never
                    // repeats an expert within a token and the batched kernel's
                    // per-expert gather/scatter relies on that — duplicates make
                    // two destination slots claim one source row and the layer
                    // faults inside the mul_mat_id launch. perRank >= nUsed
                    // guarantees a free id exists.
                    int probe = 0;
                    for (int k = 0; k < nUsed; k++)
                    {
                        int i = baseIdx + k;
                        if (ids[i] >= 0) continue;
                        while (probe < perRank && taken[probe]) probe++;
                        if (probe < perRank) { taken[probe] = true; ids[i] = probe; }
                        else ids[i] = 0;
                    }
                }

                localExperts[r] = ids;
                localWeights[r] = wts;
            }

            // The shared-expert sigmoid gate is a scalar per token derived from
            // the replicated gate vector and the (replicated) MoE input, so it is
            // identical on every rank and distributes over the partial sum.
            float[] sharedGateScale = null;
            if (_hasSharedExperts != null && _hasSharedExperts[layer]
                && _hasSharedExpertGate != null && _hasSharedExpertGate[layer]
                && _ffnGateInpShexpVec?[layer] != null)
            {
                var gateVec = _ffnGateInpShexpVec[layer];
                int dim = Math.Min((int)gateVec.ElementCount(), hiddenSize);
                float* gatePtr = GetFloatPtr(gateVec);
                float* inputPtr = GetFloatPtr(moeInput[0]);
                sharedGateScale = new float[seqLen];
                for (int s = 0; s < seqLen; s++)
                    sharedGateScale[s] = SigmoidScalar(VecDot(inputPtr + (long)s * hiddenSize, gatePtr, dim));
            }

            bool hasShared = _hasSharedExperts != null && _hasSharedExperts[layer];
            bool ok = true;

            long t0 = Stopwatch.GetTimestamp();
            _tpGroup.RunPerRank(r =>
            {
                var alloc = _tpGroup.GetAllocator(r);
                var output = new Tensor(alloc, DType.Float32, seqLen, hiddenSize);
                var g = gateShards[r];
                var u = upShards[r];
                var d = downShards[r];

                try
                {
                    GgmlBasicOps.MoEFFNPrefill(
                        moeInput[r], output,
                        seqLen, hiddenSize, intermediate, perRank, nUsed,
                        localExperts[r], localWeights[r],
                        g.Data, g.GgmlType, g.PerExpertNe0, g.PerExpertNe1, g.TotalRawBytes,
                        u.Data, u.GgmlType, u.PerExpertNe0, u.PerExpertNe1, u.TotalRawBytes,
                        d.Data, d.GgmlType, d.PerExpertNe0, d.PerExpertNe1, d.TotalRawBytes,
                        gateBias: null, upBias: null, downBias: null,
                        activation: GgmlBasicOps.MoEActivation.SwiGLUSplit,
                        runOnCpu: MoeCpuOffloadConfig.IsLayerOnCpu(layer));
                    InvalidateTensorDeviceCache(output);
                }
                catch (Exception)
                {
                    output.Dispose();
                    ok = false;
                    return;
                }

                long tShared = Stopwatch.GetTimestamp();
                if (hasShared && !AddTpSharedExpert(output, moeInput[r], sharedGateScale, layer, r, seqLen, hiddenSize))
                {
                    output.Dispose();
                    ok = false;
                    return;
                }
                if (r == 0) _tpSharedExpertTicks += Stopwatch.GetTimestamp() - tShared;

                results[r] = output;
            });
            long moeElapsed = Stopwatch.GetTimestamp() - t0;
            _linearTicks += moeElapsed;
            _tpMoeDispatchTicks += moeElapsed;

            if (!ok)
            {
                for (int r = 0; r < tp; r++)
                {
                    results[r]?.Dispose();
                    results[r] = null;
                }
            }
            return ok;
        }

        /// <summary>
        /// Add this rank's slice of the shared expert into <paramref name="output"/>.
        /// gate/up are column-parallel and down is row-parallel, so the result is
        /// itself a partial that the block's AllReduce completes.
        /// </summary>
        private unsafe bool AddTpSharedExpert(
            Tensor output, Tensor input, float[] gateScale, int layer, int rank, int seqLen, int hiddenSize)
        {
            string prefix = $"blk.{layer}.";
            Tensor sharedGate = TryTpExpertLinearLocal(input, prefix + "ffn_gate_shexp.weight", rank, seqLen);
            Tensor sharedUp = TryTpExpertLinearLocal(input, prefix + "ffn_up_shexp.weight", rank, seqLen);
            Tensor sharedDown = null;
            if (sharedGate != null && sharedUp != null)
            {
                Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                sharedDown = TryTpExpertLinearLocal(sharedGate, prefix + "ffn_down_shexp.weight", rank, seqLen);
            }
            sharedUp?.Dispose();
            sharedGate?.Dispose();
            if (sharedDown == null)
                return false;

            if (gateScale == null)
            {
                Ops.Add(output, output, sharedDown);
            }
            else
            {
                float* outPtr = GetFloatPtr(output);
                float* sharedPtr = GetFloatPtr(sharedDown);
                for (int s = 0; s < seqLen; s++)
                    VecScaleAdd(outPtr + (long)s * hiddenSize,
                                sharedPtr + (long)s * hiddenSize, gateScale[s], hiddenSize);
                InvalidateTensorDeviceCache(output);
            }
            sharedDown.Dispose();
            return true;
        }
    }
}
