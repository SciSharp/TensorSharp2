// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Diagnostics;
using TensorSharp;

namespace TensorSharp.Models
{
    /// <summary>
    /// Tensor-parallel forward pass for Gemma 3. Splits attention heads and FFN
    /// intermediate dimensions across GPUs using the Megatron-LM pattern:
    ///   column-parallel (Q, K, V, gate_up) → per-GPU attention/activation →
    ///   row-parallel (output, down) + AllReduce.
    ///
    /// Gemma 3 specifics handled here:
    ///   - Separate Q, K, V projections (not fused)
    ///   - 4 RMSNorms per layer (attn_norm, post_attention_norm, ffn_norm, post_ffw_norm)
    ///   - GELU activation (GeGLU) instead of SiLU
    ///   - QK-norm (per-head RMSNorm on Q and K)
    ///   - NeoX-style RoPE with local/global frequency bases
    ///   - Q scaling by 1/sqrt(keyLen)
    ///   - Sliding window attention for local layers
    ///   - Embedding scaling by sqrt(hidden_size)
    ///   - Final logit softcapping
    ///
    /// Each GPU stores its own subset of KV heads, so the KV cache memory is
    /// also split across devices.
    /// </summary>
    public partial class Gemma3Model
    {
        // Per-GPU KV caches: [layer][rank]. Only populated when TP is active.
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        private void InitTpKVCache(int initialSeqLen, int maxSeqLen)
        {
            int tp = TpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            DType kvDtype = _kvCacheDtype.ToDType();

            _maxContextLength = maxSeqLen;
            _tpKvCacheCapacity = initialSeqLen;
            _tpKvCacheK = new Tensor[Config.NumLayers][];
            _tpKvCacheV = new Tensor[Config.NumLayers][];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                _tpKvCacheK[l] = new Tensor[tp];
                _tpKvCacheV[l] = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, _attnKeyLen);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, _attnValLen);
                    InitializeCacheTensor(_tpKvCacheK[l][r]);
                    InitializeCacheTensor(_tpKvCacheV[l][r]);
                }
            }
        }

        private void EnsureTpCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _tpKvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_tpKvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            int tp = TpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            DType kvDtype = _kvCacheDtype.ToDType();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    var newK = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, _attnKeyLen);
                    var newV = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, _attnValLen);
                    InitializeCacheTensor(newK);
                    InitializeCacheTensor(newV);

                    if (_cacheSeqLen > 0)
                    {
                        using var srcK = _tpKvCacheK[l][r].Narrow(1, 0, _cacheSeqLen);
                        using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
                        Ops.Copy(dstK, srcK);

                        using var srcV = _tpKvCacheV[l][r].Narrow(1, 0, _cacheSeqLen);
                        using var dstV = newV.Narrow(1, 0, _cacheSeqLen);
                        Ops.Copy(dstV, srcV);
                    }

                    _tpKvCacheK[l][r].Dispose();
                    _tpKvCacheV[l][r].Dispose();
                    _tpKvCacheK[l][r] = newK;
                    _tpKvCacheV[l][r] = newV;
                }
            }

            _tpKvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded Gemma3 TP attention cache to {newCapacity} tokens ({tp} GPUs).");
        }

        private float[] ForwardTP(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int tp = TpDegree;
            EnsureTpCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden0 = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            // Gemma3 scales embeddings by sqrt(hidden_size).
            ScaleEmbedding(hidden0);

            // Inject pending vision embeddings on rank 0 before broadcasting
            // (mirrors the non-TP ForwardCore). Also clears the queue so the
            // embeddings don't leak or get re-injected on a later forward.
            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hidden0, embeddings, position, startPos);
                    embeddings.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            // Broadcast embedding to all GPUs.
            Tensor[] hidden = BroadcastTensorToAllRanks(hidden0);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                hidden = TransformerBlockTP(hidden, layer, seqLen, startPos);
            }

            // Final norm + LM head on GPU 0 only (hidden is replicated after AllReduce).
            Tensor normed = RMSNormOp(hidden[0], "output_norm.weight");
            for (int r = 0; r < tp; r++)
                hidden[r].Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            long t3 = Stopwatch.GetTimestamp();
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];

            unsafe
            {
                float* ptr = GetFloatPtr(logitsTensor);
                fixed (float* dst = _logitsBuffer)
                    Buffer.MemoryCopy(ptr, dst, Config.VocabSize * 4, Config.VocabSize * 4);
            }
            logitsTensor.Dispose();
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private Tensor[] TransformerBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            string prefix = $"blk.{layer}";
            int tp = TpDegree;

            // 1. Pre-attention norm (replicated — each GPU normalizes its own copy).
            Tensor[] normed = TpRMSNorm(hidden, $"{prefix}.attn_norm.weight");

            // 2. Column-parallel Q, K, V projections (separate in Gemma3).
            Tensor[] q = TpColumnParallelLinear(normed[0], $"{prefix}.attn_q.weight");
            Tensor[] k = TpColumnParallelLinear(normed[0], $"{prefix}.attn_k.weight");
            Tensor[] v = TpColumnParallelLinear(normed[0], $"{prefix}.attn_v.weight");
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU attention (QK norm, RoPE, Q scaling, sliding window, KV cache).
            Tensor[] attnOut = AttentionTP(q, k, v, layer, seqLen, startPos);

            // 4. Row-parallel output projection + AllReduce.
            Tensor reducedAttn = TpRowParallelLinear(attnOut, $"{prefix}.attn_output.weight");
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            // 5. Broadcast + post-attention norm.
            Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
            Tensor[] postAttnNormed = TpRMSNorm(attnReplicated, $"{prefix}.post_attention_norm.weight");
            for (int r = 1; r < tp; r++)
                attnReplicated[r].Dispose();
            reducedAttn.Dispose();

            // 6. Residual: postAttnNormed += hidden.
            TpResidualAdd(postAttnNormed, hidden);
            for (int r = 0; r < tp; r++)
                hidden[r].Dispose();

            // 7. Pre-FFN norm (replicated).
            Tensor[] ffnNormed = TpRMSNorm(postAttnNormed, $"{prefix}.ffn_norm.weight");

            // 8. Column-parallel gate/up projection.
            Tensor[] gateUp = TpColumnParallelLinear(ffnNormed[0], $"{prefix}.ffn_gate_up.weight");
            for (int r = 0; r < tp; r++)
                ffnNormed[r].Dispose();

            // 9. Per-GPU GELU·mul (GeGLU activation).
            int halfDim = (int)(gateUp[0].Sizes[1] / 2);

            Tensor[] gateResults = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                Tensor gate, up;
                if (seqLen == 1)
                {
                    gate = gateUp[r].Narrow(1, 0, halfDim);
                    up = gateUp[r].Narrow(1, halfDim, halfDim);
                }
                else
                {
                    using var gView = gateUp[r].Narrow(1, 0, halfDim);
                    gate = Ops.NewContiguous(gView);
                    using var uView = gateUp[r].Narrow(1, halfDim, halfDim);
                    up = Ops.NewContiguous(uView);
                }
                gateUp[r].Dispose();

                Ops.GELUMul(gate, gate, up);
                up.Dispose();
                gateResults[r] = gate;
            }

            // 10. Row-parallel down projection + AllReduce.
            Tensor ffnOut = TpRowParallelLinear(gateResults, $"{prefix}.ffn_down.weight");
            for (int r = 0; r < tp; r++)
                gateResults[r].Dispose();

            // 11. Broadcast + post-FFN norm.
            Tensor[] ffnReplicated = BroadcastTensorToAllRanks(ffnOut);
            Tensor[] postFfnNormed = TpRMSNorm(ffnReplicated, $"{prefix}.post_ffw_norm.weight");
            for (int r = 1; r < tp; r++)
                ffnReplicated[r].Dispose();
            ffnOut.Dispose();

            // 12. Residual: postAttnNormed += postFfnNormed.
            TpResidualAdd(postAttnNormed, postFfnNormed);
            for (int r = 0; r < tp; r++)
                postFfnNormed[r].Dispose();

            return postAttnNormed;
        }

        private Tensor[] AttentionTP(Tensor[] q, Tensor[] k, Tensor[] v, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int totalSeqLen = startPos + seqLen;
            bool isGlobal = IsGlobalLayer(layer);
            string prefix = $"blk.{layer}";

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                // QK norm (per-GPU, replicated weights).
                q[r] = ApplyQKNormTP(q[r], $"{prefix}.attn_q_norm.weight", numHeadsPerGpu, _attnKeyLen, seqLen, r);
                k[r] = ApplyQKNormTP(k[r], $"{prefix}.attn_k_norm.weight", numKVHeadsPerGpu, _attnKeyLen, seqLen, r);

                // NeoX-style RoPE with local/global frequency bases.
                if (seqLen == 1)
                {
                    float[] freqs = isGlobal ? _ropeFreqsGlobal : _ropeFreqsLocal;
                    ApplyNeoXRoPEDecode(q[r], numHeadsPerGpu, _attnKeyLen, startPos, freqs);
                    ApplyNeoXRoPEDecode(k[r], numKVHeadsPerGpu, _attnKeyLen, startPos, freqs);
                }
                else
                {
                    float ropeBase = isGlobal ? _ropeGlobalBase : _ropeLocalBase;
                    float freqScale = isGlobal ? (1.0f / _ropeScale) : 1.0f;
                    q[r] = ApplyRoPEPrefill(q[r], numHeadsPerGpu, _attnKeyLen, seqLen, startPos, ropeBase, freqScale);
                    k[r] = ApplyRoPEPrefill(k[r], numKVHeadsPerGpu, _attnKeyLen, seqLen, startPos, ropeBase, freqScale);
                }

                // Q scaling: multiply by 1/sqrt(keyLen).
                float qScale = 1f / MathF.Sqrt(_attnKeyLen);
                ScaleTensor(q[r], qScale);

                if (seqLen == 1)
                {
                    // Decode path: copy K/V to per-GPU cache, run attention with sliding window.
                    CopyToCacheDecode(_tpKvCacheK[layer][r], k[r], _tpKvCacheV[layer][r], v[r],
                        numKVHeadsPerGpu, _attnKeyLen, startPos);
                    k[r].Dispose();
                    v[r].Dispose();

                    int attendLen = isGlobal ? totalSeqLen : Math.Min(totalSeqLen, _slidingWindow);
                    int attendStart = totalSeqLen - attendLen;

                    var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * _attnValLen);
                    AttentionDecodeWithWindow(q[r], _tpKvCacheK[layer][r], _tpKvCacheV[layer][r], attnResult,
                        numHeadsPerGpu, numKVHeadsPerGpu, _attnKeyLen, _attnValLen,
                        attendStart, totalSeqLen, 1f);
                    q[r].Dispose();

                    results[r] = attnResult;
                }
                else
                {
                    // Prefill path.
                    Tensor qHeads = ReshapeToHeads(q[r], numHeadsPerGpu, seqLen, _attnKeyLen);
                    q[r].Dispose();
                    Tensor kHeads = ReshapeToHeads(k[r], numKVHeadsPerGpu, seqLen, _attnKeyLen);
                    k[r].Dispose();
                    Tensor vHeads = ReshapeToHeads(v[r], numKVHeadsPerGpu, seqLen, _attnValLen);
                    v[r].Dispose();

                    CopyToCache(_tpKvCacheK[layer][r], kHeads, startPos, seqLen);
                    CopyToCache(_tpKvCacheV[layer][r], vHeads, startPos, seqLen);
                    kHeads.Dispose();
                    vHeads.Dispose();

                    int groupSize = numHeadsPerGpu / numKVHeadsPerGpu;
                    Tensor kExpanded = ExpandKVHeads(_tpKvCacheK[layer][r], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_tpKvCacheV[layer][r], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, totalSeqLen);
                    Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    int windowSize = isGlobal ? 0 : _slidingWindow;
                    ApplyCausalMask(scores, seqLen, totalSeqLen, windowSize);
                    Ops.Softmax(scores, scores);

                    var attnOutTensor = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, _attnValLen);
                    Ops.AddmmBatch(attnOutTensor, 0, attnOutTensor, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    Tensor flatOutput = ReshapeFromHeads(attnOutTensor, numHeadsPerGpu, seqLen, _attnValLen);
                    attnOutTensor.Dispose();

                    results[r] = flatOutput;
                }
            }

            return results;
        }

        /// <summary>
        /// Apply per-head RMSNorm to Q or K in the tensor-parallel path.
        /// The norm weight is replicated across GPUs.
        /// </summary>
        private Tensor ApplyQKNormTP(Tensor data, string weightName, int numHeads, int headDim, int seqLen, int rank)
        {
            var alpha = _weights[weightName];
            Tensor alphaLocal = ReplicateTensorToRank(alpha, rank);

            if (seqLen == 1)
            {
                RMSNormInPlace(data, alphaLocal, numHeads, headDim, Config.Eps);
                if (!ReferenceEquals(alphaLocal, alpha)) alphaLocal.Dispose();
                return data;
            }

            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alphaLocal, null, Config.Eps);
            data.Dispose();
            if (!ReferenceEquals(alphaLocal, alpha)) alphaLocal.Dispose();

            Tensor result = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return result;
        }

        /// <summary>
        /// Shard Gemma 3 weights for tensor parallelism. Called from the
        /// constructor after weight loading and fusion.
        /// </summary>
        private void ShardGemma3WeightsForTP()
        {
            // attn_q/k/v are separate (single column segment each), so the
            // generic contiguous split is correct for them. The fused
            // ffn_gate_up ([gate|up]) is not: a contiguous split would give
            // rank 0 all of gate and rank 1 all of up. Shard it segment-aware.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: new[] { "attn_q.weight", "attn_k.weight", "attn_v.weight" },
                rowParallelPatterns: new[] { "attn_output.weight", "ffn_down.weight" });

            for (int layer = 0; layer < Config.NumLayers; layer++)
                ShardFusedGateUpColumnParallel($"blk.{layer}.ffn_gate_up.weight");
        }
    }
}
