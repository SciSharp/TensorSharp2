// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Gemma 3's vLLM-style batched paged forward: one pass over a batch of
// sequences whose K/V lives in a shared block pool, addressed through per-
// sequence block tables, so the engine can run continuous batching instead of
// serialising requests through the per-sequence linear cache.
//
// Gemma 3 differs from the Qwen 3 reference implementation in five ways, all of
// which show up below:
//   * separate attn_q / attn_k / attn_v weights instead of one fused QKV,
//   * a per-head RMSNorm on Q and on K before RoPE,
//   * NeoX RoPE whose base and frequency scale depend on whether the layer is
//     GLOBAL or SLIDING-WINDOW (every 6th layer is global),
//   * Q is pre-scaled by 1/sqrt(key_len), so the attention kernel runs at
//     scale 1.0 - matching the per-sequence path exactly,
//   * GeGLU (not SwiGLU) feed-forward, and Gemma's second pair of norms
//     (post_attention_norm, post_ffw_norm) around each residual add.
using System;
using System.Collections.Generic;

using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Gemma3Model : IBatchedPagedModel
    {
        // Per-layer paged K/V, each a flat float[] shaped
        // [numBlocks, blockSize, numKvHeads, headDim]. Allocated on the first
        // ForwardBatch once the engine's block-pool geometry is known.
        private float[][] _pagedK;
        private float[][] _pagedV;
        private int _pagedNumBlocks;
        private int _pagedBlockSize;

        /// <summary>
        /// The batched path is unavailable under tensor parallelism: the paged
        /// buffers here are single-allocator host arrays, while a TP run shards K/V
        /// per rank. The per-sequence TP path serves that case.
        /// </summary>
        public bool BatchedForwardAvailable => !IsTensorParallel;

        /// <summary>
        /// Vision spans are injected into the embedding row range by the
        /// per-sequence path; the batched path has no per-sequence embedding
        /// splice yet, so the executor peels multimodal sequences off.
        /// </summary>
        public bool SupportsBatchedMultimodal => false;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            ArgumentNullException.ThrowIfNull(ctx);
            if (IsTensorParallel)
                throw new NotSupportedException("Gemma 3's batched paged path does not run under tensor parallelism.");

            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0)
                return Array.Empty<float[]>();

            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;
            int numKvHeads = Config.NumKVHeads;
            int keyLen = _attnKeyLen;
            int valLen = _attnValLen;
            if (keyLen != valLen)
            {
                // The block pool stores one head_dim for both K and V; a model with
                // asymmetric key/value lengths would need two pools.
                throw new NotSupportedException(
                    $"Gemma 3's batched paged path needs key_length == value_length (got {keyLen} / {valLen}).");
            }

            int qDim = numHeads * keyLen;
            int kDim = numKvHeads * keyLen;

            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                int[] bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            EnsurePagedBuffersAllocated(maxBlockId + 1, blockSize, numKvHeads, keyLen);

            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];

            int[] positions = ctx.Positions.ToArray();
            int[] queryStartLoc = ctx.QueryStartLoc.ToArray();
            int[] slotMapping = ctx.SlotMapping.ToArray();
            int[] seqLens = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
                seqLens[s] = ctx.Sequences[s].NumComputedTokens + ctx.NumScheduledTokens[s];

            int[] flatTokens = new int[numTokens];
            int cursor = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var seq = ctx.Sequences[s];
                int start = seq.NumComputedTokens;
                int take = ctx.NumScheduledTokens[s];
                for (int i = 0; i < take; i++)
                    flatTokens[cursor++] = seq.TokenAt(start + i);
            }

            // Gemma's RoPE splits into two frequency families, so the position
            // tables are built once per family and reused across layers.
            using var posQ = BuildBatchedRoPEPositions(positions, numHeads);
            using var posK = BuildBatchedRoPEPositions(positions, numKvHeads);

            Tensor hiddenStates = Embedding(flatTokens);
            ScaleEmbedding(hiddenStates);

            float qScale = 1f / MathF.Sqrt(keyLen);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                string prefix = $"blk.{layer}";
                bool isGlobal = IsGlobalLayer(layer);

                Tensor residual = Ops.NewContiguous(hiddenStates);

                Tensor normed = RMSNormOp(hiddenStates, $"{prefix}.attn_norm.weight");
                hiddenStates.Dispose();

                Tensor q = LinearForward(normed, $"{prefix}.attn_q.weight");
                Tensor k = LinearForward(normed, $"{prefix}.attn_k.weight");
                Tensor v = LinearForward(normed, $"{prefix}.attn_v.weight");
                normed.Dispose();

                q = ApplyBatchRMSNorm(q, $"{prefix}.attn_q_norm.weight", numHeads, numTokens, keyLen);
                k = ApplyBatchRMSNorm(k, $"{prefix}.attn_k_norm.weight", numKvHeads, numTokens, keyLen);

                float ropeBase = isGlobal ? _ropeGlobalBase : _ropeLocalBase;
                float freqScale = isGlobal ? (1.0f / _ropeScale) : 1.0f;
                q = ApplyBatchedNeoXRoPE(q, posQ, numTokens, numHeads, keyLen, ropeBase, freqScale);
                k = ApplyBatchedNeoXRoPE(k, posK, numTokens, numKvHeads, keyLen, ropeBase, freqScale);

                // Pre-scale Q exactly as the per-sequence path does, so the paged
                // kernel runs at scale 1.0 and the two paths agree numerically.
                ScaleTensor(q, qScale);

                float[] kFlat = k.GetElementsAsFloat(numTokens * kDim);
                float[] vFlat = v.GetElementsAsFloat(numTokens * kDim);
                k.Dispose();
                v.Dispose();
                PagedKvBatchOps.ScatterKv(
                    kFlat, vFlat, _pagedK[layer], _pagedV[layer],
                    slotMapping, numTokens, numKvHeads, keyLen, _pagedBlockSize);

                float[] qFlat = q.GetElementsAsFloat(numTokens * qDim);
                q.Dispose();
                float[] attnFlat = new float[numTokens * qDim];
                ManagedPagedAttention.Forward(
                    qFlat, _pagedK[layer], _pagedV[layer], attnFlat,
                    numTokens, numHeads, numKvHeads, keyLen,
                    _pagedBlockSize,
                    queryStartLoc, seqLens, positions, ctx.BlockTables, numSeqs,
                    scale: 1.0f, causal: true,
                    slidingWindow: isGlobal ? 0 : _slidingWindow);

                Tensor attnOut = CreateFloatTensor(attnFlat, numTokens, qDim);
                Tensor attnProj = LinearForward(attnOut, $"{prefix}.attn_output.weight");
                attnOut.Dispose();

                // Gemma normalises the SUBLAYER OUTPUT before the residual add,
                // which is the pair of norms plain Llama-style blocks do not have.
                Tensor postAttn = RMSNormOp(attnProj, $"{prefix}.post_attention_norm.weight");
                attnProj.Dispose();
                Ops.Add(residual, residual, postAttn);
                postAttn.Dispose();

                Tensor ffnNormed = RMSNormOp(residual, $"{prefix}.ffn_norm.weight");
                Tensor ffnOut = FFNGelu(ffnNormed, $"{prefix}.ffn_gate_up.weight",
                    $"{prefix}.ffn_down.weight", numTokens);
                ffnNormed.Dispose();

                Tensor postFfn = RMSNormOp(ffnOut, $"{prefix}.post_ffw_norm.weight");
                ffnOut.Dispose();
                Ops.Add(residual, residual, postFfn);
                postFfn.Dispose();

                hiddenStates = residual;
            }

            Tensor finalNormed = RMSNormOp(hiddenStates, "output_norm.weight");
            hiddenStates.Dispose();

            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastPacked = PagedKvBatchOps.GatherLastTokenPerSeq(
                finalFlat, hidden, queryStartLoc, numSeqs);
            Tensor lastHidden = CreateFloatTensor(lastPacked, numSeqs, hidden);

            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            lastHidden.Dispose();

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            int vocab = Config.VocabSize;
            float[] allLogits = logitsTensor.GetElementsAsFloat(numSeqs * vocab);
            logitsTensor.Dispose();

            var perSeq = new float[numSeqs][];
            for (int s = 0; s < numSeqs; s++)
            {
                var slice = new float[vocab];
                Buffer.BlockCopy(allLogits, s * vocab * sizeof(float), slice, 0, vocab * sizeof(float));
                perSeq[s] = slice;
            }
            return perSeq;
        }

        /// <summary>NeoX RoPE over a batched [numTokens, numHeads*headDim] buffer with
        /// arbitrary per-token positions (the batch is not one contiguous span).</summary>
        private Tensor ApplyBatchedNeoXRoPE(Tensor data, Tensor positionsTensor,
            int numTokens, int numHeads, int headDim, float ropeBase, float freqScale)
        {
            using var reshaped = data.View(1, numTokens, numHeads, headDim);
            Tensor rotated = Ops.RoPEEx(
                null, reshaped, positionsTensor, headDim, 2, 0,
                ropeBase, freqScale, 0.0f, 1.0f, 0.0f, 0.0f);
            data.Dispose();
            Tensor flat = rotated.View(numTokens, numHeads * headDim);
            rotated.Dispose();
            return flat;
        }

        /// <summary>One position per (token, head) row, as <see cref="Ops.RoPEEx"/> wants.</summary>
        private Tensor BuildBatchedRoPEPositions(int[] tokenPositions, int numHeads)
        {
            int total = tokenPositions.Length * numHeads;
            int[] expanded = new int[total];
            for (int t = 0; t < tokenPositions.Length; t++)
                for (int h = 0; h < numHeads; h++)
                    expanded[t * numHeads + h] = tokenPositions[t];
            return CreateIntTensor(expanded, total);
        }

        private void EnsurePagedBuffersAllocated(int numBlocks, int blockSize, int numKvHeads, int headDim)
        {
            if (_pagedK != null && _pagedNumBlocks >= numBlocks && _pagedBlockSize == blockSize)
                return;

            int targetBlocks = Math.Max(numBlocks, _pagedNumBlocks * 2);

            float[][] oldK = _pagedK;
            float[][] oldV = _pagedV;
            int oldNumBlocks = _pagedNumBlocks;
            int oldBlockSize = _pagedBlockSize;

            int numLayers = Config.NumLayers;
            _pagedK = new float[numLayers][];
            _pagedV = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                _pagedK[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);
                _pagedV[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);

                // Growing must PRESERVE what is already written: the engine submits
                // sequences over many scheduler steps, and a fresh allocation would
                // zero the K/V of every sequence already in flight.
                if (oldK != null && oldK[l] != null && oldBlockSize == blockSize)
                {
                    long copyLen = Math.Min(
                        (long)oldNumBlocks * blockSize * numKvHeads * headDim,
                        oldK[l].LongLength);
                    Array.Copy(oldK[l], _pagedK[l], copyLen);
                    Array.Copy(oldV[l], _pagedV[l], copyLen);
                }
            }
            _pagedNumBlocks = targetBlocks;
            _pagedBlockSize = blockSize;
        }
    }
}
