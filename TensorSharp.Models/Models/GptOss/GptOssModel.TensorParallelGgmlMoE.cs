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
// ============================================================================
// GptOssModel.TensorParallelGgmlMoE.cs
//
// Expert-parallel MoE for the GPT-OSS tensor-parallel path on the GGML backends.
//
// What it replaces. The original TP MoE sliced INSIDE each expert (gate_up
// column-parallel, down row-parallel) and drove it from a host-side loop: for
// every layer, for every rank, for each of the 32 experts, gather that expert's
// tokens into a staging buffer, run three small matmuls, then scatter the rows
// back — all through host memory. That is ~64 gather/matmul/scatter rounds per
// layer per token, and it measured 1.9 tok/s decode against 64.3 on a single
// GPU: tensor parallelism made the model 34x SLOWER than not using it.
//
// Whole experts partition cleanly instead, exactly as they do for Qwen3.5. The
// stacked expert tensor has the expert index as its outer dimension, so rank r's
// share is a contiguous byte range — a zero-copy view — and each rank runs the
// same batched ggml_mul_mat_id dispatch the single-GPU path uses, over the
// experts it owns. Each rank sums only its own experts, so the AllReduce the
// block already performs is exactly the right recombination.
//
// The router stays replicated and the top-k stays global; a route belonging to
// another rank is neutralised with weight 0 (see the filler-id comment below).
// ============================================================================
using System;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class GptOssModel
    {
        // Per-rank slices of the stacked expert weights, [layer][rank].
        private StackedExpertWeights[][] _tpStackedGate;
        private StackedExpertWeights[][] _tpStackedUp;
        private StackedExpertWeights[][] _tpStackedDown;
        // Per-rank slices of the stacked per-expert biases, [layer][rank].
        private float[][][] _tpGateBias;
        private float[][][] _tpUpBias;
        private float[][][] _tpDownBias;
        private int _tpExpertsPerRank;
        private bool _tpEpLogged;

        /// <summary>Disable with TS_GPTOSS_TP_EXPERT_PARALLEL=0 (A/B against the
        /// per-expert slicing path).</summary>
        private static readonly bool _tpExpertParallelEnabled =
            Environment.GetEnvironmentVariable("TS_GPTOSS_TP_EXPERT_PARALLEL") != "0";

        /// <summary>
        /// True when the routed experts are partitioned by whole expert rather
        /// than sliced inside each expert.
        /// </summary>
        private bool UsesExpertParallelMoE => _tpStackedGate != null;

        /// <summary>
        /// Whether this model/backend combination will take the expert-parallel
        /// path. Evaluated before sharding (from ValidateTpConstraints) as well
        /// as during it, so the two must agree — hence one predicate.
        /// </summary>
        private bool CanUseGgmlExpertParallelMoE()
        {
            int tp = GlobalTpDegree;
            if (!_tpExpertParallelEnabled || !IsGgmlBackend || _numExperts <= 0 || tp <= 1 || (_numExperts % tp) != 0)
                return false;
            // A foreign route is neutralised by pointing it at an unused LOCAL
            // expert id, which requires there to be one.
            if (_numExperts / tp < _numExpertsUsed)
                return false;
            if (_layerStackedReady == 0
                || _layerStackedGate == null || _layerStackedUp == null || _layerStackedDown == null)
                return false;
            for (int l = 0; l < Config.NumLayers; l++)
            {
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
        private bool BuildGptOssExpertParallelShards()
        {
            if (!CanUseGgmlExpertParallelMoE())
                return false;

            int globalTp = GlobalTpDegree;
            int localTp = TpDegree;
            int rankOffset = TpRankOffset;
            int perRank = _numExperts / globalTp;
            int n = Config.NumLayers;
            int nFf = _expertFfnLength;
            int hidden = Config.HiddenSize;

            var gate = new StackedExpertWeights[n][];
            var up = new StackedExpertWeights[n][];
            var down = new StackedExpertWeights[n][];
            var gateB = new float[n][][];
            var upB = new float[n][][];
            var downB = new float[n][][];

            for (int layer = 0; layer < n; layer++)
            {
                gate[layer] = new StackedExpertWeights[localTp];
                up[layer] = new StackedExpertWeights[localTp];
                down[layer] = new StackedExpertWeights[localTp];
                gateB[layer] = new float[localTp][];
                upB[layer] = new float[localTp][];
                downB[layer] = new float[localTp][];

                // The gate_up bias is stored fused ([gate|up] per expert); the
                // kernel's separate gate/up weight path wants them apart, so
                // split and slice in one pass.
                float[] fusedBias = _layerGateUpBiasStacked?[layer];
                float[] downStack = _layerDownBiasStacked?[layer];

                for (int lr = 0; lr < localTp; lr++)
                {
                    int firstExpert = (rankOffset + lr) * perRank;
                    gate[layer][lr] = SliceStackedExperts(_layerStackedGate[layer], firstExpert, perRank);
                    up[layer][lr] = SliceStackedExperts(_layerStackedUp[layer], firstExpert, perRank);
                    down[layer][lr] = SliceStackedExperts(_layerStackedDown[layer], firstExpert, perRank);

                    if (fusedBias != null)
                    {
                        var g = new float[(long)nFf * perRank];
                        var u = new float[(long)nFf * perRank];
                        for (int e = 0; e < perRank; e++)
                        {
                            int src = (firstExpert + e) * 2 * nFf;
                            Array.Copy(fusedBias, src, g, e * nFf, nFf);
                            Array.Copy(fusedBias, src + nFf, u, e * nFf, nFf);
                        }
                        gateB[layer][lr] = g;
                        upB[layer][lr] = u;
                    }
                    if (downStack != null)
                    {
                        var d = new float[(long)hidden * perRank];
                        Array.Copy(downStack, (long)firstExpert * hidden, d, 0, (long)perRank * hidden);
                        downB[layer][lr] = d;
                    }
                }
            }

            _tpStackedGate = gate;
            _tpStackedUp = up;
            _tpStackedDown = down;
            _tpGateBias = gateB;
            _tpUpBias = upB;
            _tpDownBias = downB;
            _tpExpertsPerRank = perRank;

            Console.WriteLine(
                $"  GPT-OSS MoE: expert-parallel across {globalTp} GPU(s), {perRank} of {_numExperts} experts per GPU " +
                "(one batched ggml_mul_mat_id dispatch per projection per layer).");
            return true;
        }

        /// <summary>
        /// The per-expert <c>_exps.E.weight</c> split views are never dispatched
        /// once the STACKED expert blocks are available: the whole-model decode
        /// and prefill graphs, the fused MoE kernel and (under TP) the batched
        /// expert-parallel kernel all read the stacked tensor, which gets its own
        /// device copy on first use. Preloading the split views as well puts a
        /// SECOND full copy of every expert byte in VRAM.
        ///
        /// That was worth 9.7 GB on gpt-oss-20b: 24.8 GB resident against
        /// llama.cpp's 12.0 GB for the same 11.5 GB of weights. Under TP it had
        /// already been caught in one direction (rank 0 held 16019 MB against
        /// rank 1's 5163) and fixed with the <c>UsesExpertParallelMoE</c> term
        /// below — but single-GPU took the same double copy and nothing was
        /// excluding it there. The stacked-weights term covers both.
        ///
        /// The host views stay mapped either way, so the per-expert fallback loop
        /// can still stream the bytes if a graph ever refuses at runtime.
        /// </summary>
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !((UsesExpertParallelMoE || _layerStackedReady != 0) && IsPerExpertView(weightName))
               && base.ShouldPreloadCudaQuantWeightToDevice(weightName);

        /// <summary>
        /// True for a single expert's slice of a routed-expert tensor —
        /// <c>blk.3.ffn_gate_up_exps.17.weight</c>. Both the GGUF's own split
        /// views (which <see cref="_stackedExpertMemberNames"/> tracks) and the
        /// fused [gate|up] views FuseExpertGateUpWeights builds (which it does
        /// not) match, and under expert parallelism neither is ever dispatched.
        /// The stacked tensors themselves carry no per-expert index, so they are
        /// deliberately NOT matched: those are what each rank's slice views.
        /// </summary>
        private bool IsPerExpertView(string weightName)
        {
            if (_stackedExpertMemberNames.Contains(weightName))
                return true;
            if (string.IsNullOrEmpty(weightName))
                return false;
            int marker = weightName.IndexOf("_exps.", StringComparison.Ordinal);
            if (marker < 0)
                return false;
            int i = marker + "_exps.".Length;
            int start = i;
            while (i < weightName.Length && weightName[i] >= '0' && weightName[i] <= '9') i++;
            return i > start && i < weightName.Length && weightName[i] == '.';
        }

        /// <summary>
        /// The expert-parallel path is one batched dispatch per projection per
        /// layer, so it does not need the lightweight startup warmup the
        /// per-(token, expert) loop does.
        /// </summary>
        protected override bool MoEUnderTpIsSlow => IsTensorParallel && !UsesExpertParallelMoE;

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
        /// this the first forward pays the whole upload, which looks like a hang.
        /// Layers offloaded by --n-cpu-moe are skipped: their experts are
        /// multiplied on the host out of the GGUF mmap, and not uploading them is
        /// the VRAM saving that flag exists for.
        /// </summary>
        protected override void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
        {
            if (!UsesExpertParallelMoE)
                return;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_tpStackedGate[layer] == null || MoeCpuOffloadConfig.IsLayerOnCpu(layer))
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
        /// <c>ggml_mul_mat_id</c> kernel over the experts it owns. Every rank's
        /// result is a PARTIAL that the caller's AllReduce completes.
        ///
        /// Returns false if the shapes don't suit the kernel, leaving the caller
        /// on the per-expert path.
        /// </summary>
        private bool TryGptOssMoEExpertParallel(
            Tensor[] moeInput,
            Tensor[] results,
            int[] selectedExperts,
            float[] routeWeights,
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
            int nFf = (int)gateShards[0].PerExpertNe1;

            if (gateShards[0].PerExpertNe0 != hiddenSize || upShards[0].PerExpertNe0 != hiddenSize
                || upShards[0].PerExpertNe1 != nFf
                || downShards[0].PerExpertNe0 != nFf || downShards[0].PerExpertNe1 != hiddenSize)
                return false;

            // Per-rank dense route tables, built once on the calling thread: the
            // rank workers must not race on shared scratch.
            int totalRoutes = seqLen * nUsed;
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
                        int e = selectedExperts[i];
                        if (e >= first && e < last)
                        {
                            int local = e - first;
                            ids[i] = local;
                            taken[local] = true;
                            wts[i] = routeWeights[i];
                        }
                        else
                        {
                            ids[i] = -1;
                            wts[i] = 0f;
                        }
                    }

                    // A route owned by another rank still occupies a slot in the
                    // dense route table; it is neutralised with weight 0 (which
                    // also cancels that expert's down bias, since the kernel adds
                    // the bias before weighting). It must get a DISTINCT local
                    // expert id: real top-k routing never repeats an expert
                    // within a token and the batched kernel's per-expert
                    // gather/scatter relies on that — duplicates make two
                    // destination slots claim one source row and the layer faults
                    // inside the mul_mat_id launch. perRank >= nUsed guarantees a
                    // free id exists.
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

            bool ok = true;
            _tpGroup.RunPerRank(r =>
            {
                var alloc = _tpGroup.GetAllocator(r);
                var output = new Tensor(alloc, DType.Float32, seqLen, hiddenSize);
                var g = gateShards[r];
                var u = upShards[r];
                var d = downShards[r];

                try
                {
                    GgmlBasicOps.MoEFFNPrefillSwiGLU(
                        moeInput[r], output,
                        seqLen, hiddenSize, nFf, perRank, nUsed,
                        localExperts[r], localWeights[r],
                        g.Data, g.GgmlType, g.PerExpertNe0, g.PerExpertNe1, g.TotalRawBytes,
                        u.Data, u.GgmlType, u.PerExpertNe0, u.PerExpertNe1, u.TotalRawBytes,
                        d.Data, d.GgmlType, d.PerExpertNe0, d.PerExpertNe1, d.TotalRawBytes,
                        _tpGateBias?[layer]?[r], _tpUpBias?[layer]?[r], _tpDownBias?[layer]?[r],
                        useSwiGLUOAI: true,
                        oaiAlpha: SiluAlpha,
                        oaiLimit: SiluLimit,
                        runOnCpu: MoeCpuOffloadConfig.IsLayerOnCpu(layer));
                    InvalidateTensorDeviceCache(output);
                    results[r] = output;
                }
                catch (Exception)
                {
                    output.Dispose();
                    ok = false;
                }
            });

            if (!ok)
            {
                for (int r = 0; r < tp; r++)
                {
                    results[r]?.Dispose();
                    results[r] = null;
                }
                return false;
            }

            if (!_tpEpLogged)
            {
                _tpEpLogged = true;
                Console.WriteLine($"  GPT-OSS expert-parallel MoE active ({TpDegree} GPUs, " +
                    "one batched dispatch per projection per layer).");
            }
            return true;
        }
    }
}
