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
    /// Tensor-parallel forward pass for Qwen3. Splits attention heads and FFN
    /// intermediate dimensions across GPUs using the Megatron-LM pattern:
    ///   column-parallel (QKV, gate_up) → per-GPU attention/activation →
    ///   row-parallel (output, down) + AllReduce.
    ///
    /// Each GPU stores its own subset of KV heads, so the KV cache memory is
    /// also split across devices.
    /// </summary>
    public partial class Qwen3Model
    {
        // Per-GPU KV caches: [layer][rank]. Only populated when TP is active.
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        private void InitTpKVCache(int initialSeqLen, int maxSeqLen)
        {
            int tp = TpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();

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
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, headDim);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, headDim);
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
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    var newK = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, headDim);
                    var newV = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, headDim);
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
            Console.WriteLine($"Expanded Qwen3 TP attention cache to {newCapacity} tokens ({tp} GPUs).");
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

            // 1. Attention norm (replicated — each GPU normalizes its own copy).
            Tensor[] normed = TpRMSNorm(hidden, wn[0]);

            // 2. Column-parallel QKV projection.
            // Each GPU produces [seqLen, (qDim + 2*kDim) / tp].
            Tensor[] qkvFused = TpColumnParallelLinear(normed[0], wn[1]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU attention (each GPU handles numHeads/tp Q heads).
            Tensor[] attnOut = AttentionTP(qkvFused, layer, wn, seqLen, startPos);

            // 4. Row-parallel output projection + AllReduce.
            Tensor reducedAttn = TpRowParallelLinear(attnOut, wn[4]);
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            // 5. Residual add (replicated after AllReduce).
            // Broadcast reducedAttn (rank 0) to all GPUs, then add.
            Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
            TpResidualAdd(hidden, attnReplicated);
            for (int r = 1; r < tp; r++)
                attnReplicated[r].Dispose();
            reducedAttn.Dispose();

            // 6. FFN norm (replicated).
            Tensor[] normed2 = TpRMSNorm(hidden, wn[5]);

            // 7. Column-parallel gate/up projection.
            Tensor[] gateUp = TpColumnParallelLinear(normed2[0], wn[6]);
            for (int r = 0; r < tp; r++)
                normed2[r].Dispose();

            // 8. Per-GPU SiLU·mul.
            int intermSize = Config.IntermediateSize;
            int halfDimPerGpu = (intermSize > 0 ? intermSize : (int)(gateUp[0].Sizes[1] / 2));
            // gateUp[r] has shape [seqLen, 2 * halfDimPerGpu]
            // Actually: gateUp output dim = 2*intermediateSize/tp, so half = intermediateSize/tp
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

            // 9. Row-parallel down projection + AllReduce.
            Tensor ffnOut = TpRowParallelLinear(gateResults, wn[7]);
            for (int r = 0; r < tp; r++)
                gateResults[r].Dispose();

            // 10. Residual add.
            Tensor[] ffnReplicated = BroadcastTensorToAllRanks(ffnOut);
            TpResidualAdd(hidden, ffnReplicated);
            for (int r = 1; r < tp; r++)
                ffnReplicated[r].Dispose();
            ffnOut.Dispose();

            return hidden;
        }

        private Tensor[] AttentionTP(Tensor[] qkvFused, int layer, string[] wn, int seqLen, int startPos)
        {
            int tp = TpDegree;
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            int qDimPerGpu = numHeadsPerGpu * headDim;
            int kDimPerGpu = numKVHeadsPerGpu * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                // Split Q, K, V from the fused QKV output.
                Tensor qTensor, kTensor, vTensor;
                if (seqLen == 1)
                {
                    qTensor = qkvFused[r].Narrow(1, 0, qDimPerGpu);
                    kTensor = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu);
                    vTensor = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, kDimPerGpu);
                    qkvFused[r].Dispose();
                }
                else
                {
                    using (var qView = qkvFused[r].Narrow(1, 0, qDimPerGpu))
                        qTensor = Ops.NewContiguous(qView);
                    using (var kView = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, kDimPerGpu))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused[r].Dispose();
                }

                // QK norm (per-GPU, replicated weights).
                qTensor = ApplyQKNormInPlaceTP(qTensor, wn[2], numHeadsPerGpu, seqLen, r);
                kTensor = ApplyQKNormInPlaceTP(kTensor, wn[3], numKVHeadsPerGpu, seqLen, r);

                // RoPE.
                if (seqLen == 1)
                {
                    ApplyRoPEDecodeInPlace(qTensor, numHeadsPerGpu, headDim, startPos);
                    ApplyRoPEDecodeInPlace(kTensor, numKVHeadsPerGpu, headDim, startPos);
                }
                else
                {
                    qTensor = ApplyRoPEInPlace(qTensor, numHeadsPerGpu, headDim, seqLen, startPos);
                    kTensor = ApplyRoPEInPlace(kTensor, numKVHeadsPerGpu, headDim, seqLen, startPos);
                }

                if (seqLen == 1)
                {
                    // Decode path: copy K/V to per-GPU cache, run attention.
                    CopyToCacheDecode(_tpKvCacheK[layer][r], kTensor, _tpKvCacheV[layer][r], vTensor,
                        numKVHeadsPerGpu, headDim, startPos);
                    kTensor.Dispose();
                    vTensor.Dispose();

                    var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * headDim);
                    AttentionDecodePureCS(qTensor, _tpKvCacheK[layer][r], _tpKvCacheV[layer][r],
                        attnResult, numHeadsPerGpu, numKVHeadsPerGpu, headDim, totalSeqLen, scale);
                    qTensor.Dispose();

                    results[r] = attnResult;
                }
                else
                {
                    // Prefill path.
                    Tensor qHeads = ReshapeToHeads(qTensor, numHeadsPerGpu, seqLen, headDim);
                    qTensor.Dispose();
                    Tensor kHeads = ReshapeToHeads(kTensor, numKVHeadsPerGpu, seqLen, headDim);
                    kTensor.Dispose();
                    Tensor vHeads = ReshapeToHeads(vTensor, numKVHeadsPerGpu, seqLen, headDim);
                    vTensor.Dispose();

                    CopyToCache(_tpKvCacheK[layer][r], kHeads, startPos, seqLen);
                    CopyToCache(_tpKvCacheV[layer][r], vHeads, startPos, seqLen);
                    kHeads.Dispose();
                    vHeads.Dispose();

                    int groupSize = numHeadsPerGpu / numKVHeadsPerGpu;
                    Tensor kExpanded = ExpandKVHeads(_tpKvCacheK[layer][r], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_tpKvCacheV[layer][r], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, totalSeqLen);
                    Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                    Ops.Softmax(scores, scores);

                    var attnOut = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, headDim);
                    Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    Tensor flatOutput = ReshapeFromHeads(attnOut, numHeadsPerGpu, seqLen, headDim);
                    attnOut.Dispose();

                    results[r] = flatOutput;
                }
            }

            return results;
        }

        private Tensor ApplyQKNormInPlaceTP(Tensor data, string weightName, int numHeads, int seqLen, int rank)
        {
            int headDim = Config.HeadDim;
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
        /// Shard Qwen3 weights for tensor parallelism. Called from the
        /// constructor after weight loading and fusion.
        /// </summary>
        private void ShardQwen3WeightsForTP()
        {
            // attn_output / ffn_down are row-parallel. The fused attn_qkv
            // ([Q|K|V]) and ffn_gate_up ([gate|up]) are column-parallel but must
            // be sharded segment-aware: a plain contiguous split would give
            // rank 0 whole segments (all Q / all gate) and corrupt the per-rank
            // [Q_r|K_r|V_r] / [gate_r|up_r] layout the forward pass re-splits.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: Array.Empty<string>(),
                rowParallelPatterns: new[] { "attn_output.weight", "ffn_down.weight" });

            int hd = Config.HeadDim;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                ShardConcatenatedColumnParallel($"blk.{layer}.attn_qkv.weight",
                    Config.NumHeads * hd,     // Q
                    Config.NumKVHeads * hd,   // K
                    Config.NumKVHeads * hd);  // V
                ShardFusedGateUpColumnParallel($"blk.{layer}.ffn_gate_up.weight");
            }
        }
    }
}
