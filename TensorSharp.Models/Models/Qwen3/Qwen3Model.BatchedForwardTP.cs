/// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Tensor-parallel batched paged-attention forward path for Qwen3. Combines
// the Megatron-LM column/row-parallel pattern (from Qwen3Model.TensorParallel.cs)
// with the continuous-batching paged-attention path (from Qwen3Model.BatchedForward.cs).
//
// Each GPU holds 1/tp of the QKV and FFN weights, and 1/tp of the KV heads
// in its own paged buffer. Linear/norm/FFN ops use the TP primitives
// (TpColumnParallelLinear, TpRowParallelLinear, TpRMSNorm). Attention runs
// per-rank via ManagedPagedAttention with numHeads/tp Q heads and
// numKVHeads/tp KV heads.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Qwen3Model
    {
        // Per-rank paged K/V buffers: [rank][layer] -> float[].
        // Each buffer has shape [numBlocks, blockSize, numKVHeads/tp, headDim].
        private float[][][] _tpPagedK;
        private float[][][] _tpPagedV;
        private int _tpPagedNumBlocks;
        private int _tpPagedBlockSize;

        private IReadOnlyList<float[]> ForwardBatchTP(BatchedForwardContext ctx)
        {
            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            int tp = TpDegree;
            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;
            int numKvHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int numHeadsPerGpu = numHeads / tp;
            int numKvHeadsPerGpu = numKvHeads / tp;
            int qDimPerGpu = numHeadsPerGpu * headDim;
            int kDimPerGpu = numKvHeadsPerGpu * headDim;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Determine block-pool dimensions.
            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            int requiredBlocks = maxBlockId + 1;
            EnsureTpPagedBuffersAllocated(requiredBlocks, blockSize, numKvHeadsPerGpu, headDim, tp);

            // Flatten metadata.
            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];
            int[] positions = ctx.Positions.ToArray();
            int[] queryStartLoc = ctx.QueryStartLoc.ToArray();
            int[] slotMapping = ctx.SlotMapping.ToArray();
            int[] seqLens = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
                seqLens[s] = ctx.Sequences[s].NumComputedTokens + ctx.NumScheduledTokens[s];

            // Build concatenated input-tokens array.
            int[] flatTokens = new int[numTokens];
            int writeCursor = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var seq = ctx.Sequences[s];
                int startTok = seq.NumComputedTokens;
                int take = ctx.NumScheduledTokens[s];
                for (int i = 0; i < take; i++)
                    flatTokens[writeCursor++] = seq.TokenAt(startTok + i);
            }

            // Per-rank RoPE positions tensors (reduced head counts).
            using var positionsTensorQ = BuildRoPEPositionsTensor(positions, numHeadsPerGpu);
            using var positionsTensorK = BuildRoPEPositionsTensor(positions, numKvHeadsPerGpu);

            // ----- embed + broadcast -----
            Tensor hidden0 = Embedding(flatTokens);
            Tensor[] hiddenStates = BroadcastTensorToAllRanks(hidden0);

            // ----- per-layer transformer -----
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                string[] wn = _layerWeightNames[layer];

                // 1. Attention norm (replicated — each GPU normalizes its own copy).
                Tensor[] normed = TpRMSNorm(hiddenStates, wn[0]);

                // 2. Column-parallel QKV projection.
                // Each GPU produces [numTokens, (qDim + 2*kDim) / tp].
                Tensor[] qkvFused = TpColumnParallelLinear(normed[0], wn[1]);
                for (int r = 0; r < tp; r++)
                    normed[r].Dispose();

                // 3. Per-rank attention with paged KV.
                Tensor[] attnOut = BatchedAttentionTP(
                    qkvFused, layer, wn, numTokens, numHeadsPerGpu, numKvHeadsPerGpu,
                    headDim, qDimPerGpu, kDimPerGpu, scale,
                    positionsTensorQ, positionsTensorK,
                    positions, queryStartLoc, seqLens, slotMapping, ctx.BlockTables, numSeqs);

                // 4. Row-parallel output projection + AllReduce.
                Tensor reducedAttn = TpRowParallelLinear(attnOut, wn[4]);
                for (int r = 0; r < tp; r++)
                    attnOut[r].Dispose();

                // 5. Residual add.
                Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
                TpResidualAdd(hiddenStates, attnReplicated);
                for (int r = 1; r < tp; r++)
                    attnReplicated[r].Dispose();
                reducedAttn.Dispose();

                // 6. FFN norm (replicated).
                Tensor[] normed2 = TpRMSNorm(hiddenStates, wn[5]);

                // 7. Column-parallel gate/up projection.
                Tensor[] gateUp = TpColumnParallelLinear(normed2[0], wn[6]);
                for (int r = 0; r < tp; r++)
                    normed2[r].Dispose();

                // 8. Per-GPU SiLU·mul.
                int halfDim = (int)(gateUp[0].Sizes[1] / 2);
                Tensor[] gateResults = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    using var gView = gateUp[r].Narrow(1, 0, halfDim);
                    Tensor gate = Ops.NewContiguous(gView);
                    using var uView = gateUp[r].Narrow(1, halfDim, halfDim);
                    Tensor up = Ops.NewContiguous(uView);
                    gateUp[r].Dispose();

                    Ops.SiLUMul(gate, gate, up);
                    up.Dispose();
                    gateResults[r] = gate;
                }

                // 9. Row-parallel down projection + AllReduce.
                Tensor ffnOut = TpRowParallelLinear(gateResults, wn[7]);
                for (int r = 0; r < tp; r++)
                    gateResults[r].Dispose();

                // 10. Residual add.
                Tensor[] ffnReplicated = BroadcastTensorToAllRanks(ffnOut);
                TpResidualAdd(hiddenStates, ffnReplicated);
                for (int r = 1; r < tp; r++)
                    ffnReplicated[r].Dispose();
                ffnOut.Dispose();
            }

            // ----- final norm + LM head on rank 0 -----
            Tensor finalNormed = RMSNormOp(hiddenStates[0], "output_norm.weight");
            for (int r = 0; r < tp; r++)
                hiddenStates[r].Dispose();

            // Gather the last token of each sequence.
            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastTokensPacked = PagedKvBatchOps.GatherLastTokenPerSeq(
                finalFlat, hidden, queryStartLoc, numSeqs);
            Tensor lastHidden = CreateFloatTensor(lastTokensPacked, numSeqs, hidden);

            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            lastHidden.Dispose();

            float[] allLogits = logitsTensor.GetElementsAsFloat(numSeqs * Config.VocabSize);
            logitsTensor.Dispose();

            // Split per-sequence.
            var perSeq = new float[numSeqs][];
            for (int s = 0; s < numSeqs; s++)
            {
                var slice = new float[Config.VocabSize];
                Buffer.BlockCopy(allLogits, s * Config.VocabSize * sizeof(float),
                                 slice, 0, Config.VocabSize * sizeof(float));
                perSeq[s] = slice;
            }
            return perSeq;
        }

        /// <summary>
        /// Per-rank batched paged attention. Each rank splits Q/K/V from its
        /// column-parallel QKV shard, applies QK norm + RoPE, scatters K/V
        /// into its own paged buffer (numKVHeads/tp heads), and runs
        /// ManagedPagedAttention with numHeads/tp Q heads.
        /// </summary>
        private Tensor[] BatchedAttentionTP(
            Tensor[] qkvFused, int layer, string[] wn,
            int numTokens, int numHeadsPerGpu, int numKvHeadsPerGpu,
            int headDim, int qDimPerGpu, int kDimPerGpu, float scale,
            Tensor positionsTensorQ, Tensor positionsTensorK,
            int[] positions, int[] queryStartLoc, int[] seqLens,
            int[] slotMapping, int[][] blockTables, int numSeqs)
        {
            int tp = TpDegree;
            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                // Split Q, K, V from the fused QKV output.
                using var qView = qkvFused[r].Narrow(1, 0, qDimPerGpu);
                Tensor q = Ops.NewContiguous(qView);
                using var kView = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu);
                Tensor k = Ops.NewContiguous(kView);
                using var vView = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, kDimPerGpu);
                Tensor v = Ops.NewContiguous(vView);
                qkvFused[r].Dispose();

                // QK norm (per-head RMSNorm, replicated weights).
                q = ApplyQKNormBatchedTP(q, wn[2], numTokens, numHeadsPerGpu, headDim, r);
                k = ApplyQKNormBatchedTP(k, wn[3], numTokens, numKvHeadsPerGpu, headDim, r);

                // Per-token RoPE with arbitrary positions.
                q = ApplyBatchedRoPETP(q, positionsTensorQ, numTokens, numHeadsPerGpu, headDim, r);
                k = ApplyBatchedRoPETP(k, positionsTensorK, numTokens, numKvHeadsPerGpu, headDim, r);

                // Scatter K/V into this rank's paged buffer.
                float[] kFlat = k.GetElementsAsFloat(numTokens * kDimPerGpu);
                float[] vFlat = v.GetElementsAsFloat(numTokens * kDimPerGpu);
                k.Dispose();
                v.Dispose();

                PagedKvBatchOps.ScatterKv(
                    kFlat, vFlat, _tpPagedK[r][layer], _tpPagedV[r][layer],
                    slotMapping, numTokens, numKvHeadsPerGpu, headDim, _tpPagedBlockSize);

                // Per-sequence paged attention on this rank.
                float[] qFlat = q.GetElementsAsFloat(numTokens * qDimPerGpu);
                q.Dispose();
                float[] attnFlat = new float[numTokens * qDimPerGpu];
                ManagedPagedAttention.Forward(
                    qFlat, _tpPagedK[r][layer], _tpPagedV[r][layer], attnFlat,
                    numTokens, numHeadsPerGpu, numKvHeadsPerGpu, headDim,
                    _tpPagedBlockSize,
                    queryStartLoc, seqLens, positions, blockTables, numSeqs,
                    scale, causal: true);

                results[r] = new Tensor(alloc, DType.Float32, numTokens, qDimPerGpu);
                results[r].SetElementsAsFloat(attnFlat);
            }

            return results;
        }

        /// <summary>Per-head RMSNorm over a batched [numTokens, dim] buffer,
        /// with the norm weight replicated to the target rank.</summary>
        private Tensor ApplyQKNormBatchedTP(Tensor data, string weightName, int numTokens, int numHeads, int headDim, int rank)
        {
            var alpha = _weights[weightName];
            Tensor alphaLocal = ReplicateTensorToRank(alpha, rank);

            using var reshaped = data.View(numTokens * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alphaLocal, null, Config.Eps);
            data.Dispose();
            if (!ReferenceEquals(alphaLocal, alpha)) alphaLocal.Dispose();

            Tensor result = normed.View(numTokens, numHeads * headDim);
            normed.Dispose();
            return result;
        }

        /// <summary>Apply RoPE with arbitrary per-token positions on a
        /// specific rank's GPU.</summary>
        private Tensor ApplyBatchedRoPETP(Tensor data, Tensor positionsTensor, int numTokens, int numHeads, int headDim, int rank)
        {
            Tensor posLocal = ReplicateTensorToRank(positionsTensor, rank);

            using var reshaped = data.View(1, numTokens, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posLocal, headDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);
            data.Dispose();
            if (!ReferenceEquals(posLocal, positionsTensor)) posLocal.Dispose();

            Tensor flat = result.View(numTokens, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private void EnsureTpPagedBuffersAllocated(int numBlocks, int blockSize, int numKvHeadsPerGpu, int headDim, int tp)
        {
            if (_tpPagedK != null && _tpPagedNumBlocks >= numBlocks && _tpPagedBlockSize == blockSize)
                return;

            int targetBlocks = Math.Max(numBlocks, _tpPagedNumBlocks * 2);
            int numLayers = Config.NumLayers;

            float[][][] oldK = _tpPagedK;
            float[][][] oldV = _tpPagedV;
            int oldNumBlocks = _tpPagedNumBlocks;
            int oldBlockSize = _tpPagedBlockSize;

            _tpPagedK = new float[tp][][];
            _tpPagedV = new float[tp][][];
            for (int r = 0; r < tp; r++)
            {
                _tpPagedK[r] = new float[numLayers][];
                _tpPagedV[r] = new float[numLayers][];
                for (int l = 0; l < numLayers; l++)
                {
                    _tpPagedK[r][l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeadsPerGpu, headDim);
                    _tpPagedV[r][l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeadsPerGpu, headDim);
                    if (oldK != null && oldK[r][l] != null && oldBlockSize == blockSize)
                    {
                        long copyLen = Math.Min(
                            (long)oldNumBlocks * blockSize * numKvHeadsPerGpu * headDim,
                            oldK[r][l].LongLength);
                        Array.Copy(oldK[r][l], _tpPagedK[r][l], copyLen);
                        Array.Copy(oldV[r][l], _tpPagedV[r][l], copyLen);
                    }
                }
            }
            _tpPagedNumBlocks = targetBlocks;
            _tpPagedBlockSize = blockSize;
        }
    }
}
