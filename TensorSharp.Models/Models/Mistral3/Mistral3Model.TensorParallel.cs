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
    /// Tensor-parallel forward pass for Mistral3. Splits attention heads and FFN
    /// intermediate dimensions across GPUs using the Megatron-LM pattern:
    ///   column-parallel (QKV, gate_up) → per-GPU attention/activation →
    ///   row-parallel (output, down) + AllReduce.
    ///
    /// Each GPU stores its own subset of KV heads, so the KV cache memory is
    /// also split across devices.
    ///
    /// Handles both fused QKV (attn_qkv.weight) and separate Q/K/V
    /// (attn_q.weight, attn_k.weight, attn_v.weight) per-layer.
    /// </summary>
    public partial class Mistral3Model
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
            Console.WriteLine($"Expanded Mistral3 TP attention cache to {newCapacity} tokens ({tp} GPUs).");
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

            // Inject pending vision embeddings on rank 0 before broadcasting
            // (mirrors the non-TP ForwardCore and the batched TP path). Without
            // this, multimodal prompts silently drop the image under TP.
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
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private Tensor[] TransformerBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            string[] wn = _layerWeightNames[layer];
            int tp = TpDegree;
            bool fused = _layerQkvFused[layer];

            // Weight indices differ for fused vs separate QKV.
            int outputIdx = fused ? 2 : 4;
            int ffnNormIdx = fused ? 3 : 5;
            int gateUpIdx = fused ? 4 : 6;
            int downIdx = fused ? 5 : 7;

            // 1. Attention norm (replicated — each GPU normalizes its own copy).
            Tensor[] normed = TpRMSNorm(hidden, wn[0]);

            // 2. Column-parallel QKV projection + per-GPU attention.
            Tensor[] attnOut;
            if (fused)
            {
                // Fused QKV: single column-parallel linear, then split Q/K/V per GPU.
                Tensor[] qkvFused = TpColumnParallelLinear(normed[0], wn[1]);
                for (int r = 0; r < tp; r++)
                    normed[r].Dispose();

                attnOut = AttentionTPFused(qkvFused, layer, seqLen, startPos);
            }
            else
            {
                // Separate Q, K, V: three column-parallel linears.
                Tensor[] qProj = TpColumnParallelLinear(normed[0], wn[1]);
                Tensor[] kProj = TpColumnParallelLinear(normed[0], wn[2]);
                Tensor[] vProj = TpColumnParallelLinear(normed[0], wn[3]);
                for (int r = 0; r < tp; r++)
                    normed[r].Dispose();

                attnOut = AttentionTPSeparate(qProj, kProj, vProj, layer, seqLen, startPos);
            }

            // 3. Row-parallel output projection + AllReduce.
            Tensor reducedAttn = TpRowParallelLinear(attnOut, wn[outputIdx]);
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            // 4. Residual add (replicated after AllReduce).
            Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
            TpResidualAdd(hidden, attnReplicated);
            for (int r = 1; r < tp; r++)
                attnReplicated[r].Dispose();
            reducedAttn.Dispose();

            // 5. FFN norm (replicated).
            Tensor[] normed2 = TpRMSNorm(hidden, wn[ffnNormIdx]);

            // 6. Column-parallel gate/up projection.
            Tensor[] gateUp = TpColumnParallelLinear(normed2[0], wn[gateUpIdx]);
            for (int r = 0; r < tp; r++)
                normed2[r].Dispose();

            // 7. Per-GPU SiLU·mul.
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

                Ops.SiLUMul(gate, gate, up);
                up.Dispose();
                gateResults[r] = gate;
            }

            // 8. Row-parallel down projection + AllReduce.
            Tensor ffnOut = TpRowParallelLinear(gateResults, wn[downIdx]);
            for (int r = 0; r < tp; r++)
                gateResults[r].Dispose();

            // 9. Residual add.
            Tensor[] ffnReplicated = BroadcastTensorToAllRanks(ffnOut);
            TpResidualAdd(hidden, ffnReplicated);
            for (int r = 1; r < tp; r++)
                ffnReplicated[r].Dispose();
            ffnOut.Dispose();

            return hidden;
        }

        /// <summary>
        /// Per-GPU attention for the fused QKV case. Each GPU's qkvFused[r]
        /// contains [seqLen, qDimPerGpu + kDimPerGpu + vDimPerGpu].
        /// </summary>
        private Tensor[] AttentionTPFused(Tensor[] qkvFused, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = _attnKeyLen;
            int valDim = _attnValLen;
            int qDimPerGpu = numHeadsPerGpu * headDim;
            int kDimPerGpu = numKVHeadsPerGpu * headDim;
            int vDimPerGpu = numKVHeadsPerGpu * valDim;

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                // Split Q, K, V from the fused QKV output.
                Tensor qTensor, kTensor, vTensor;
                if (seqLen == 1)
                {
                    qTensor = qkvFused[r].Narrow(1, 0, qDimPerGpu);
                    kTensor = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu);
                    vTensor = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, vDimPerGpu);
                    qkvFused[r].Dispose();
                }
                else
                {
                    using (var qView = qkvFused[r].Narrow(1, 0, qDimPerGpu))
                        qTensor = Ops.NewContiguous(qView);
                    using (var kView = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, vDimPerGpu))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused[r].Dispose();
                }

                results[r] = RunAttentionPerGpu(qTensor, kTensor, vTensor, layer, r,
                    numHeadsPerGpu, numKVHeadsPerGpu, headDim, valDim, seqLen, startPos);
            }

            return results;
        }

        /// <summary>
        /// Per-GPU attention for the separate Q/K/V case. Each GPU already has
        /// its own slice from the column-parallel projections.
        /// </summary>
        private Tensor[] AttentionTPSeparate(Tensor[] qProj, Tensor[] kProj, Tensor[] vProj,
            int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = _attnKeyLen;
            int valDim = _attnValLen;

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                results[r] = RunAttentionPerGpu(qProj[r], kProj[r], vProj[r], layer, r,
                    numHeadsPerGpu, numKVHeadsPerGpu, headDim, valDim, seqLen, startPos);
            }

            return results;
        }

        /// <summary>
        /// Core per-GPU attention computation: RoPE → KV cache → attention.
        /// Shared by both fused and separate QKV paths.
        /// </summary>
        private Tensor RunAttentionPerGpu(Tensor qTensor, Tensor kTensor, Tensor vTensor,
            int layer, int rank, int numHeadsPerGpu, int numKVHeadsPerGpu,
            int headDim, int valDim, int seqLen, int startPos)
        {
            var alloc = _tpGroup.GetAllocator(rank);
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // RoPE (GPT-J style).
            if (seqLen == 1)
            {
                ApplyRoPEDecode(qTensor, numHeadsPerGpu, headDim, startPos);
                ApplyRoPEDecode(kTensor, numKVHeadsPerGpu, headDim, startPos);

                // Position-dependent Q scaling for YaRN.
                if (_ropeOrigCtx > 0)
                    ApplyPositionScale(qTensor, numHeadsPerGpu * headDim, startPos);
            }
            else
            {
                qTensor = ApplyRoPEPrefill(qTensor, numHeadsPerGpu, headDim, seqLen, startPos);
                kTensor = ApplyRoPEPrefill(kTensor, numKVHeadsPerGpu, headDim, seqLen, startPos);

                // Position-dependent Q scaling for YaRN.
                if (_ropeOrigCtx > 0)
                    ApplyPositionScalePrefill(qTensor, numHeadsPerGpu, headDim, seqLen, startPos);
            }

            if (seqLen == 1)
            {
                // Decode path: copy K/V to per-GPU cache, run attention.
                CopyToCacheDecode(_tpKvCacheK[layer][rank], kTensor, _tpKvCacheV[layer][rank], vTensor,
                    numKVHeadsPerGpu, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * headDim);
                AttentionDecodePureCS(qTensor, _tpKvCacheK[layer][rank], _tpKvCacheV[layer][rank],
                    attnResult, numHeadsPerGpu, numKVHeadsPerGpu, headDim, totalSeqLen, scale);
                qTensor.Dispose();

                return attnResult;
            }
            else
            {
                // Prefill path.
                Tensor qHeads = ReshapeToHeads(qTensor, numHeadsPerGpu, seqLen, headDim);
                qTensor.Dispose();
                Tensor kHeads = ReshapeToHeads(kTensor, numKVHeadsPerGpu, seqLen, headDim);
                kTensor.Dispose();
                Tensor vHeads = ReshapeToHeads(vTensor, numKVHeadsPerGpu, seqLen, valDim);
                vTensor.Dispose();

                CopyToCache(_tpKvCacheK[layer][rank], kHeads, startPos, seqLen);
                CopyToCache(_tpKvCacheV[layer][rank], vHeads, startPos, seqLen);
                kHeads.Dispose();
                vHeads.Dispose();

                int groupSize = numHeadsPerGpu / numKVHeadsPerGpu;
                Tensor kExpanded = ExpandKVHeads(_tpKvCacheK[layer][rank], groupSize, totalSeqLen);
                Tensor vExpanded = ExpandKVHeads(_tpKvCacheV[layer][rank], groupSize, totalSeqLen);

                using var kT = kExpanded.Transpose(1, 2);
                var scores = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, totalSeqLen);
                Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
                qHeads.Dispose();
                kExpanded.Dispose();

                Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                Ops.Softmax(scores, scores);

                var attnOut = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, valDim);
                Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                scores.Dispose();
                vExpanded.Dispose();

                Tensor flatOutput = ReshapeFromHeads(attnOut, numHeadsPerGpu, seqLen, valDim);
                attnOut.Dispose();

                return flatOutput;
            }
        }

        /// <summary>
        /// Shard Mistral3 weights for tensor parallelism. Called from the
        /// constructor after weight loading and fusion.
        /// </summary>
        private void ShardMistral3WeightsForTP()
        {
            // Separate attn_q/k/v are single column segments (contiguous split
            // is correct). The fused attn_qkv ([Q|K|V]) and ffn_gate_up
            // ([gate|up]) need segment-aware sharding: a contiguous split would
            // hand each rank whole segments rather than its [Q_r|K_r|V_r] /
            // [gate_r|up_r] slice, corrupting the forward re-split.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: new[] { "attn_q.weight", "attn_k.weight", "attn_v.weight" },
                rowParallelPatterns: new[] { "attn_output.weight", "ffn_down.weight" });

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                // attn_qkv exists only on fused-QKV layers; no-op otherwise.
                ShardConcatenatedColumnParallel($"blk.{layer}.attn_qkv.weight",
                    Config.NumHeads * _attnKeyLen,     // Q
                    Config.NumKVHeads * _attnKeyLen,   // K
                    Config.NumKVHeads * _attnValLen);  // V
                ShardFusedGateUpColumnParallel($"blk.{layer}.ffn_gate_up.weight");
            }
        }
    }
}
