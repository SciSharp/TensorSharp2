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
// Tensor-parallel forward for Muse-Glimmer (Megatron-LM column/row split).
//
//   column-parallel (Q, K, V, attn_gate, ffn_gate_up)
//     -> per-rank attention / SwiGLU on that rank's heads and FFN columns
//     -> row-parallel (attn_output, ffn_down) + AllReduce
//
// Two AllReduce points per layer, both landing on the RAW matmul output and
// never after the post-norm: RMSNorm is non-linear, so norming a partial sum
// and reducing afterwards produces plausible-looking but wrong activations.
//
// Muse-Glimmer specifics handled here:
//   - 2 KV heads on the 30B, so --tp is limited to 1 or 2 (see the validator).
//   - Per-head RMSNorm on Q and K with 1-D [headDim] weights. Those are
//     REPLICATED, not sharded: they are per-head, and the Q weight additionally
//     carries the folded qk_scale_factor.
//   - NORM-flavour (interleaved-pair) RoPE on the sliding-window layers only;
//     the full-attention layers are NoPE.
//   - The attention output gate, applied on each rank's own head slice before
//     the row-parallel output projection.
//   - Four norms per layer: the pre-norms at Config.Eps, the post-norms at the
//     hardcoded 1e-8 (PostNormEps).
//   - SwiGLU (SiLU), not GeGLU. Gemma 3's TP file uses Ops.GELUMul; copying
//     that here would silently change the activation.
//   - The sliding-window RING KV cache. Its row count is fixed at construction
//     and is rank-independent, so KvRow/KvRows are reused unchanged and only
//     the head dimension is split per rank.
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
        // Per-GPU KV caches: [layer][rank]. Only populated when TP is active; the
        // single-GPU _kvCacheK/_kvCacheV stay null so a stray host-pointer read
        // faults loudly instead of attending over an empty cache.
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;

        // ====================================================================
        // TP constraint validation
        // ====================================================================

        /// <summary>
        /// Reject a TP degree the architecture cannot express, listing every
        /// failure in one throw so a mis-configured run is fixed in one pass
        /// rather than one error at a time.
        /// </summary>
        private void ValidateMuseGlimmerTpConstraints()
        {
            int tp = GlobalTpDegree;
            var errors = new List<string>();

            if (Config.NumHeads % tp != 0)
                errors.Add($"Attention heads ({Config.NumHeads}) not divisible by global TP degree ({tp})");

            // The 30B ships 2 KV heads. GQA heads cannot be split below one head
            // per rank without duplicating K/V (which would also duplicate the
            // cache), so this is a hard cap: --tp 1 or --tp 2 only.
            if (Config.NumKVHeads % tp != 0)
            {
                errors.Add($"KV heads ({Config.NumKVHeads}) not divisible by global TP degree ({tp}). " +
                    $"Muse-Glimmer-30B has {Config.NumKVHeads} KV heads, so only --tp 1 or --tp 2 are possible.");
            }

            if (Config.IntermediateSize % tp != 0)
                errors.Add($"FFN intermediate size ({Config.IntermediateSize}) not divisible by global TP degree ({tp})");

            // Row-parallel weights are split along ne0 (the INPUT dim) and a
            // quantized row cannot be cut inside a block, so the block count has
            // to divide evenly too. attn_output and ffn_down have different ne0
            // (n_embd vs n_ff), so both are checked.
            ValidateRowParallelBlocks("blk.0.attn_output.weight", tp, errors);
            ValidateRowParallelBlocks("blk.0.ffn_down.weight", tp, errors);

            // Column-parallel FFN needs the FUSED [gate|up] tensor: ModelBase's
            // FuseGateUpWeights skips a layer whose gate/up carry different quant
            // types, and the per-rank forward has no separate-projection path.
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string gu = $"blk.{l}.ffn_gate_up.weight";
                if (!_quantWeights.ContainsKey(gu) && !_weights.ContainsKey(gu))
                {
                    errors.Add($"Layer {l} has no fused '{gu}' (gate/up quant types differ, so FuseGateUpWeights " +
                        "skipped it); the tensor-parallel FFN needs the fused tensor");
                    break;
                }
            }

            // The per-rank fan-out needs one ggml backend per GPU. Metal is a
            // single-device backend and direct CUDA has no Muse-Glimmer kernels,
            // so neither can serve a rank.
            if (_backend is not (BackendType.GgmlCuda or BackendType.GgmlVulkan))
                errors.Add($"Muse-Glimmer TP requires ggml-cuda or ggml-vulkan, got {_backend}");

            if (errors.Count > 0)
                throw new InvalidOperationException("Muse-Glimmer TP validation failed:\n  " + string.Join("\n  ", errors));

            Console.WriteLine($"  TP constraints validated: globalTp={tp}, localTp={TpDegree}, " +
                $"Heads={Config.NumHeads}/{Config.NumHeads / tp} per rank, " +
                $"KVHeads={Config.NumKVHeads}/{Config.NumKVHeads / tp} per rank, " +
                $"FFN={Config.IntermediateSize}/{Config.IntermediateSize / tp} per rank");
        }

        private void ValidateRowParallelBlocks(string weightName, int tp, List<string> errors)
        {
            if (!_quantWeights.TryGetValue(weightName, out var qw))
                return;
            long blockSize = GgufFile.GetBlockSize((GgmlTensorType)qw.GgmlType);
            if (blockSize <= 0)
                return;
            long blocksPerRow = qw.Ne0 / blockSize;
            if (blocksPerRow % tp != 0)
            {
                errors.Add($"Row-parallel '{weightName}' has {blocksPerRow} quant blocks per row " +
                    $"(ne0={qw.Ne0}, block={blockSize}), not divisible by global TP degree ({tp})");
            }
        }

        // ====================================================================
        // Weight sharding
        // ====================================================================

        /// <summary>
        /// Shard the transformer weights across ranks.
        ///
        /// Every weight is addressed by its EXACT per-layer name rather than by a
        /// substring pattern. <see cref="ModelBase.ShardWeightsForTensorParallelism"/>
        /// matches with <c>name.Contains(pattern)</c>, and the DFlash drafter's
        /// tensors live in the SAME dictionaries under a "dflash." prefix with
        /// identical leaf names ("dflash.blk.0.attn_q.weight"), so any pattern -
        /// even the full leaf name - also matches the drafter and would shard a
        /// model that is never run tensor-parallel. The post-condition below is
        /// the belt to that braces.
        /// </summary>
        private void ShardMuseGlimmerWeightsForTP()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string p = $"blk.{l}.";

                // Column-parallel: each rank gets its own heads.
                ShardColumnParallelExact(p + "attn_q.weight");
                ShardColumnParallelExact(p + "attn_k.weight");
                ShardColumnParallelExact(p + "attn_v.weight");
                ShardColumnParallelExact(p + "attn_gate.weight");

                // The fused [gate|up] is TWO logical segments. A contiguous split
                // would hand rank 0 all of gate and rank 1 all of up - shapes
                // still line up, SiLUMul still runs, and the output is silently
                // wrong. Each half is split independently instead.
                ShardFusedGateUpColumnParallel(p + "ffn_gate_up.weight");

                // Row-parallel: each rank consumes its own slice of the input dim
                // and produces a PARTIAL sum, reduced by the AllReduce in
                // TransformerBlockTP.
                ShardRowParallelExact(p + "attn_output.weight");
                ShardRowParallelExact(p + "ffn_down.weight");

                // Replicated, deliberately left in _weights:
                //   attn_norm, post_attention_norm, ffn_norm, post_ffw_norm
                //     - full-width [hidden] norms, every rank needs all of it;
                //   attn_q_norm, attn_k_norm
                //     - 1-D [headDim] PER-HEAD norms. Splitting them along their
                //       only dimension would give each rank a fragment of every
                //       head's scale and corrupt all of them; the Q weight also
                //       carries the folded qk_scale_factor.
            }

            // output_norm.weight, token_embd.weight and output.weight are also
            // replicated: the tail runs on rank 0 (see ForwardTP).

            foreach (var name in _tpQuantWeights.Keys)
            {
                if (name.StartsWith(DFlashConfig.WeightPrefix, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        $"Muse-Glimmer TP sharding touched the DFlash drafter weight '{name}'. The drafter runs " +
                        "whole on rank 0 and must never be sharded; this is a sharding-key bug.");
                }
            }

            Console.WriteLine($"  Muse-Glimmer TP weight sharding complete ({TpDegree} GPU(s), " +
                $"{_tpQuantWeights.Count} quantized + {_tpWeights.Count} F32 sharded tensors).");
        }

        /// <summary>
        /// Column-parallel split of ONE named weight along its output dimension.
        /// Expressed as a single-segment
        /// <see cref="ModelBase.ShardConcatenatedColumnParallel"/> so the split is
        /// keyed by the exact name instead of a substring pattern.
        /// </summary>
        private void ShardColumnParallelExact(string weightName)
        {
            int outDim = GetFusedOutputDim(weightName);
            if (outDim <= 0)
                return;
            ShardConcatenatedColumnParallel(weightName, outDim);
        }

        /// <summary>
        /// Row-parallel split of ONE named weight along its input dimension
        /// (ne0 / Sizes[1]). Quantized rows are cut on a whole-block boundary -
        /// the block count per row is validated up front - so each rank holds a
        /// self-contained quantized sub-row.
        /// </summary>
        private void ShardRowParallelExact(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                var type = (GgmlTensorType)qw.GgmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                long typeSize = GgufFile.GetTypeSize(type);
                long blocksPerRow = qw.Ne0 / blockSize;
                long blocksPerShard = blocksPerRow / globalTp;
                long ne0PerShard = blocksPerShard * blockSize;
                long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                long dstRowBytes = blocksPerShard * typeSize;
                long totalBytesPerShard = qw.Ne1 * dstRowBytes;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        long srcBlockOffset = (rankOffset + r) * blocksPerShard * typeSize;
                        for (long row = 0; row < qw.Ne1; row++)
                        {
                            Buffer.MemoryCopy(
                                src + row * srcRowBytes + srcBlockOffset,
                                dst + row * dstRowBytes,
                                dstRowBytes, dstRowBytes);
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                        qw.GgmlType, ne0PerShard, qw.Ne1);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                // The shards own independent buffers, so the source is free to go.
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                long shardSize = w.Sizes[1] / globalTp;
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    using var view = w.Narrow(1, (rankOffset + r) * shardSize, shardSize);
                    shards[r] = Ops.NewContiguous(view);
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        // ====================================================================
        // Per-rank KV cache
        // ====================================================================

        /// <summary>
        /// Allocate the per-rank KV caches: [layer][rank], each
        /// [numKVHeads / tp, rows, headDim] on that rank's allocator, so the
        /// cache is split across GPUs exactly like the heads that fill it.
        ///
        /// The sliding-window RING is preserved unchanged. Its row count is a
        /// function of the MAX context and the prefill chunk only - both
        /// rank-independent - so <see cref="KvRow"/> and <see cref="KvRows"/>
        /// keep working as written, and a ring layer is allocated at its final
        /// size here and never revisited by the grow path.
        ///
        /// <see cref="ZeroCacheTensor"/>, not
        /// <see cref="ModelBase.InitializeCacheTensor"/>: the fused kernel reads
        /// a window padded to a fixed stride and masks the padding with -inf, and
        /// -inf added to an uninitialized NaN is still NaN.
        /// </summary>
        private void InitTpKVCache(int initialSeqLen, int maxSeqLen)
        {
            int tp = TpDegree;
            int kvHeadsPerRank = Config.NumKVHeads / GlobalTpDegree;

            _maxContextLength = maxSeqLen;
            // The TP caches share _kvCacheCapacity with the single-GPU path on
            // purpose: KvRows() reads it, and the fused TP arrays compare against
            // it to detect a grow.
            _kvCacheCapacity = initialSeqLen;

            // The KV dtype default follows the model's DOMINANT quant type, and
            // sharding already moved every layer projection out of _quantWeights -
            // what is left there is the embedding table and the LM head. Deciding
            // from those alone would pick a different cache dtype under --tp than
            // on one GPU, so rank 0's shards are merged back in for the vote.
            var quantForKvDefault = new Dictionary<string, QuantizedWeight>(_quantWeights);
            foreach (var kv in _tpQuantWeights)
                quantForKvDefault[kv.Key] = kv.Value[0];
            ApplyModelAlignedKvCacheDefault(quantForKvDefault);
            // Read AFTER the vote above: it is what sets _kvCacheDtype.
            DType kvDtype = _kvCacheDtype.ToDType();

            _kvSwaRows = ResolveSwaRows(maxSeqLen);

            _tpKvCacheK = new Tensor[Config.NumLayers][];
            _tpKvCacheV = new Tensor[Config.NumLayers][];

            int swaLayers = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                bool ring = _kvSwaRows > 0 && _isSwaLayer[l];
                int rows = ring ? _kvSwaRows : initialSeqLen;
                if (ring) swaLayers++;

                _tpKvCacheK[l] = new Tensor[tp];
                _tpKvCacheV[l] = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerRank, rows, _headDim);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerRank, rows, _headDim);
                    ZeroCacheTensor(_tpKvCacheK[l][r]);
                    ZeroCacheTensor(_tpKvCacheV[l][r]);
                }
            }

            long sized = (long)(Config.NumLayers - swaLayers) * initialSeqLen + (long)swaLayers * _kvSwaRows;
            long bytes = sized * kvHeadsPerRank * _headDim * 2 * kvDtype.Size();
            Console.WriteLine($"  Muse-Glimmer TP KV cache: {tp} rank(s), {kvHeadsPerRank} KV head(s) each, " +
                (_kvSwaRows > 0
                    ? $"{swaLayers} sliding-window layers ring at {_kvSwaRows} rows, " +
                      $"{Config.NumLayers - swaLayers} full layers at {initialSeqLen} ({bytes / (1024 * 1024)} MB per rank)."
                    : $"{Config.NumLayers} layers at {initialSeqLen} rows."));

            _cacheSeqLen = 0;
        }

        /// <summary>
        /// Grow the FULL-attention layers of every rank's cache. Ring layers are
        /// skipped: they were allocated at their final row count in
        /// <see cref="InitTpKVCache"/> and their size is published through
        /// MaxReusablePrefixTokens / KVStateFingerprint, which the scheduler
        /// samples once at engine construction.
        /// </summary>
        private void EnsureTpCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            int tp = TpDegree;
            int kvHeadsPerRank = Config.NumKVHeads / GlobalTpDegree;
            DType kvDtype = _kvCacheDtype.ToDType();

            // The device copies are about to be freed, so anything the fused TP
            // path wrote on-device has to come back to the host first or the
            // copy below would carry a stale mirror into the new buffers.
            SyncMuseGlimmerTpKvCacheToHost();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvSwaRows > 0 && _isSwaLayer[l])
                    continue;   // ring layers are already at their final size

                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    var newK = new Tensor(alloc, kvDtype, kvHeadsPerRank, newCapacity, _headDim);
                    var newV = new Tensor(alloc, kvDtype, kvHeadsPerRank, newCapacity, _headDim);
                    ZeroCacheTensor(newK);
                    ZeroCacheTensor(newV);

                    if (_cacheSeqLen > 0)
                    {
                        int keep = Math.Min(_cacheSeqLen, _kvCacheCapacity);
                        using (var srcK = _tpKvCacheK[l][r].Narrow(1, 0, keep))
                        using (var dstK = newK.Narrow(1, 0, keep))
                            Ops.Copy(dstK, srcK);
                        using (var srcV = _tpKvCacheV[l][r].Narrow(1, 0, keep))
                        using (var dstV = newV.Narrow(1, 0, keep))
                            Ops.Copy(dstV, srcV);
                    }

                    _tpKvCacheK[l][r].Dispose();
                    _tpKvCacheV[l][r].Dispose();
                    _tpKvCacheK[l][r] = newK;
                    _tpKvCacheV[l][r] = newV;
                }
            }

            _kvCacheCapacity = newCapacity;
            // Every parked per-rank graph baked the old KV addresses and the old
            // cache extent; replaying one against the reallocated buffers hangs.
            ResetMuseGlimmerTpGraphs();
            BuildMuseGlimmerTpDecodeArrays();
            Console.WriteLine($"Expanded Muse-Glimmer TP attention cache to {newCapacity} tokens ({tp} GPU(s)).");
        }

        // ====================================================================
        // TP forward pass
        // ====================================================================

        private float[] ForwardTP(int[] tokens)
        {
            // Chunking and the ring-width guard are rank-independent, so they run
            // before anything is allocated - same reasoning (and same limits) as
            // ForwardCore.
            if (PrefillChunkTokens > 0 && tokens.Length > PrefillChunkTokens)
                return ForwardChunked(tokens, PrefillChunkTokens);

            if (_kvSwaRows > 0 && tokens.Length > _kvSwaRows - _slidingWindow)
            {
                throw new InvalidOperationException(
                    $"Muse-Glimmer forward of {tokens.Length} tokens exceeds what the sliding-window " +
                    $"ring can hold in one pass ({_kvSwaRows - _slidingWindow} rows: ring {_kvSwaRows} " +
                    $"minus window {_slidingWindow}). Raise TS_MUSE_GLIMMER_PREFILL_CHUNK (the ring is " +
                    "sized from it) or set TS_MUSE_GLIMMER_SWA_RING=0.");
            }

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int tp = TpDegree;
            EnsureTpCacheCapacity(startPos + seqLen);

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden0 = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            // Multimodal rows replace token embeddings BEFORE the input norm, and
            // - critically - before the broadcast: injecting after it would leave
            // ranks > 0 running on plain token embeddings and the image would
            // simply vanish from the answer.
            DrainPendingVisionEmbeddings(hidden0, startPos);
            ApplyInputEmbeddingNorm(hidden0, seqLen);

            // Whole-model fused graph, one per rank, cut at the two row-parallel
            // projections per layer.
            float[] fused = TryFusedForwardTP(hidden0, seqLen, startPos);
            if (fused != null)
            {
                hidden0.Dispose();
                _logitsBuffer = fused;
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return _logitsBuffer;
            }

            // The per-op blocks below read the KV caches through host pointers; a
            // fused step leaves its writes on the devices only.
            SyncMuseGlimmerTpKvCacheToHost();

            Tensor[] hidden = BroadcastTensorToAllRanks(hidden0);
            hidden0.Dispose();

            for (int layer = 0; layer < Config.NumLayers; layer++)
                hidden = TransformerBlockTP(hidden, layer, seqLen, startPos);

            // Tail on rank 0. The LM head is the largest tensor left once the
            // layers are sharded, so v1 keeps it (and the final norm) replicated
            // there rather than giving back the memory TP just saved.
            Tensor normed = RMSNormOp(hidden[0], "output_norm.weight");
            for (int r = 0; r < tp; r++)
                hidden[r].Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var lastRow = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(lastRow);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t1 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, _hasTiedOutput ? "token_embd.weight" : "output.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t1;
            lastHidden.Dispose();

            ApplyOutputTransform(logitsTensor);

            long t2 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t2;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        /// <summary>
        /// One transformer block across all ranks. The op order matches
        /// <see cref="TransformerBlock"/> exactly; the only structural difference
        /// is where the two AllReduce points sit.
        ///
        /// Both reductions land on the RAW row-parallel matmul output, BEFORE the
        /// post-norm. Reducing after the norm normalises a partial sum, which is
        /// not the norm of the whole - the output still looks coherent, which is
        /// what makes that mistake expensive to find.
        /// </summary>
        private Tensor[] TransformerBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            string[] names = _layerWeightNames[layer];

            // 1. Pre-attention norm (replicated weight, per-rank activation).
            Tensor[] normed = TpRMSNorm(hidden, names[0]);

            // 2. Column-parallel Q / K / V / attention gate. Muse-Glimmer keeps
            // them separate in the GGUF, and each is a single output segment, so
            // the contiguous per-rank split is the right one.
            Tensor[] q = TpColumnParallelLinear(normed[0], names[1]);
            Tensor[] k = TpColumnParallelLinear(normed[0], names[2]);
            Tensor[] v = TpColumnParallelLinear(normed[0], names[3]);
            Tensor[] gate = TpColumnParallelLinear(normed[0], names[4]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-rank attention over that rank's heads, ending with the
            // output gate applied inside the per-rank region (it is elementwise
            // on the head slice, so it must NOT wait for the AllReduce).
            Tensor[] attnOut = AttentionTP(q, k, v, gate, layer, names, seqLen, startPos);

            // 4. Row-parallel output projection.        <-- AllReduce #1
            Tensor[] o = TpRowParallelLinearAllRanks(attnOut, names[7]);
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            // 5. post_attention_norm at the hardcoded 1e-8, on the reduced sum.
            Tensor[] postAttn = TpRMSNormWithEps(o, names[8], PostNormEps);
            for (int r = 0; r < tp; r++)
                o[r].Dispose();

            // 6. Residual.
            TpResidualAdd(postAttn, hidden);
            for (int r = 0; r < tp; r++)
                hidden[r].Dispose();

            // 7. Pre-FFN norm.
            Tensor[] ffnNormed = TpRMSNorm(postAttn, names[9]);

            // 8. Column-parallel fused [gate|up] + SwiGLU. Each rank holds its
            // own 1/tp slice of BOTH halves (segment-aware sharding), so the
            // split point is the local half.
            Tensor[] gateUp = TpColumnParallelLinear(ffnNormed[0], names[10]);
            for (int r = 0; r < tp; r++)
                ffnNormed[r].Dispose();

            int halfDim = (int)(gateUp[0].Sizes[1] / 2);
            var activated = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                Tensor g, up;
                if (seqLen == 1)
                {
                    g = gateUp[r].Narrow(1, 0, halfDim);
                    up = gateUp[r].Narrow(1, halfDim, halfDim);
                }
                else
                {
                    using (var gView = gateUp[r].Narrow(1, 0, halfDim))
                        g = Ops.NewContiguous(gView);
                    using (var uView = gateUp[r].Narrow(1, halfDim, halfDim))
                        up = Ops.NewContiguous(uView);
                }
                gateUp[r].Dispose();

                // SwiGLU. Muse-Glimmer is SiLU-gated; Gemma 3's TP file uses
                // Ops.GELUMul here and copying it would change the model.
                Ops.SiLUMul(g, g, up);
                up.Dispose();
                activated[r] = g;
            }

            // 9. Row-parallel down projection.          <-- AllReduce #2
            Tensor[] d = TpRowParallelLinearAllRanks(activated, names[11]);
            for (int r = 0; r < tp; r++)
                activated[r].Dispose();

            // 10. post_ffw_norm at 1e-8, on the reduced sum.
            Tensor[] postFfn = TpRMSNormWithEps(d, names[12], PostNormEps);
            for (int r = 0; r < tp; r++)
                d[r].Dispose();

            // 11. Residual.
            TpResidualAdd(postAttn, postFfn);
            for (int r = 0; r < tp; r++)
                postFfn[r].Dispose();

            return postAttn;
        }

        /// <summary>
        /// Per-rank attention: QK per-head RMSNorm, RoPE on sliding-window layers
        /// only, KV write into that rank's cache, windowed attention, output gate.
        ///
        /// Written as a SEQUENTIAL rank loop rather than a
        /// <c>_tpGroup.RunPerRank</c> body on purpose:
        /// <see cref="_cachedSWAMaskWidths"/>, <see cref="_cachedSWAMaskQueryLen"/>
        /// and <see cref="_cachedSWAMaskStartPos"/> are shared instance fields that
        /// two rank workers would race on (one rank reallocating the array while
        /// the other reads it). The heavy work - the projections and the two
        /// reductions - already gets its concurrency from
        /// <see cref="ModelBase.TpColumnParallelLinear"/> and
        /// <see cref="ModelBase.TpRowParallelLinearAllRanks"/>.
        /// </summary>
        private Tensor[] AttentionTP(Tensor[] q, Tensor[] k, Tensor[] v, Tensor[] gate,
            int layer, string[] names, int seqLen, int startPos)
        {
            long t0 = Stopwatch.GetTimestamp();
            int tp = TpDegree;
            int numHeadsPerRank = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerRank = Config.NumKVHeads / GlobalTpDegree;
            int totalSeqLen = startPos + seqLen;
            bool useRope = _isSwaLayer[layer];
            float qScale = 1f / MathF.Sqrt(_headDim);

            // Materialize the replicated QK-norm replicas up front. The replica
            // dictionary is not thread-safe, and doing it here also keeps the
            // per-rank loop free of a cross-GPU copy on its first iteration.
            var qNormAlpha = new Tensor[tp];
            var kNormAlpha = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                qNormAlpha[r] = TpReplicatedWeight(_weights[names[5]], r);
                kNormAlpha[r] = TpReplicatedWeight(_weights[names[6]], r);
            }

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                // Per-head QK RMSNorm. The weights are [headDim] and REPLICATED,
                // so each rank norms its own heads with the full scale vector.
                q[r] = ApplyQKNormTP(q[r], qNormAlpha[r], numHeadsPerRank, seqLen);
                k[r] = ApplyQKNormTP(k[r], kNormAlpha[r], numKVHeadsPerRank, seqLen);

                // NORM-flavour (interleaved-pair) RoPE, sliding-window layers
                // only; the full-attention layers are NoPE.
                if (useRope)
                {
                    if (seqLen == 1)
                    {
                        ApplyNormRoPEDecode(q[r], numHeadsPerRank, _headDim, startPos, _ropeFreqs);
                        ApplyNormRoPEDecode(k[r], numKVHeadsPerRank, _headDim, startPos, _ropeFreqs);
                    }
                    else
                    {
                        q[r] = ApplyRoPEPrefill(q[r], numHeadsPerRank, _headDim, seqLen, startPos);
                        k[r] = ApplyRoPEPrefill(k[r], numKVHeadsPerRank, _headDim, seqLen, startPos);
                    }
                }

                Ops.Mul(q[r], q[r], qScale);

                Tensor attn;
                if (seqLen == 1)
                {
                    // The per-op path addresses a cache row as an absolute
                    // position, which a ring row is not. The ring is only ever
                    // armed when a fused kernel is available, so reaching here
                    // means the fused TP forward declined at run time.
                    RequireUniformCache(layer);
                    CopyToCacheDecode(_tpKvCacheK[layer][r], k[r], _tpKvCacheV[layer][r], v[r],
                        numKVHeadsPerRank, _headDim, startPos);
                    k[r].Dispose();
                    v[r].Dispose();

                    int attendLen = _isSwaLayer[layer] ? Math.Min(totalSeqLen, _slidingWindow) : totalSeqLen;
                    int attendStart = totalSeqLen - attendLen;

                    attn = new Tensor(alloc, DType.Float32, 1, numHeadsPerRank * _headDim);
                    AttentionDecodeWithWindow(q[r], _tpKvCacheK[layer][r], _tpKvCacheV[layer][r], attn,
                        numHeadsPerRank, numKVHeadsPerRank, _headDim, _headDim,
                        attendStart, totalSeqLen);
                    q[r].Dispose();
                }
                else
                {
                    Tensor qHeads = ReshapeToHeads(q[r], numHeadsPerRank, seqLen, _headDim);
                    q[r].Dispose();
                    Tensor kHeads = ReshapeToHeads(k[r], numKVHeadsPerRank, seqLen, _headDim);
                    k[r].Dispose();
                    Tensor vHeads = ReshapeToHeads(v[r], numKVHeadsPerRank, seqLen, _headDim);
                    v[r].Dispose();

                    RequireUniformCache(layer);
                    CopyToCache(_tpKvCacheK[layer][r], kHeads, startPos, seqLen);
                    CopyToCache(_tpKvCacheV[layer][r], vHeads, startPos, seqLen);
                    kHeads.Dispose();
                    vHeads.Dispose();

                    int groupSize = numHeadsPerRank / numKVHeadsPerRank;
                    Tensor kExpanded = ExpandKVHeads(_tpKvCacheK[layer][r], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_tpKvCacheV[layer][r], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(alloc, DType.Float32, numHeadsPerRank, seqLen, totalSeqLen);
                    Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    ApplyCausalMask(scores, seqLen, totalSeqLen, _isSwaLayer[layer] ? _slidingWindow : 0);
                    Ops.Softmax(scores, scores);

                    var attnHeads = new Tensor(alloc, DType.Float32, numHeadsPerRank, seqLen, _headDim);
                    Ops.AddmmBatch(attnHeads, 0, attnHeads, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    attn = ReshapeFromHeads(attnHeads, numHeadsPerRank, seqLen, _headDim);
                    attnHeads.Dispose();
                }

                // Attention output gate: attn * sigmoid(gate), before o_proj and
                // therefore inside the per-rank region - the gate columns were
                // sharded alongside the heads they scale.
                Ops.SigmoidMul(attn, attn, gate[r]);
                gate[r].Dispose();

                results[r] = attn;
            }

            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return results;
        }

        /// <summary>
        /// Per-head RMSNorm of Q or K on one rank, with the REPLICATED
        /// [headDim] weight already resident on that rank's GPU.
        /// </summary>
        private Tensor ApplyQKNormTP(Tensor data, Tensor alphaLocal, int numHeads, int seqLen)
        {
            if (seqLen == 1)
            {
                RMSNormInPlace(data, alphaLocal, numHeads, _headDim, Config.Eps);
                return data;
            }

            using var reshaped = data.View(seqLen * numHeads, _headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alphaLocal, null, Config.Eps);
            data.Dispose();
            Tensor flat = normed.View(seqLen, numHeads * _headDim);
            normed.Dispose();
            return flat;
        }

        /// <summary>
        /// <see cref="ModelBase.TpRMSNorm"/> with an explicit epsilon: the two
        /// post-norms use the hardcoded 1e-8 rather than Config.Eps (llama.cpp
        /// src/models/muse-glimmer.cpp).
        /// </summary>
        private Tensor[] TpRMSNormWithEps(Tensor[] inputs, string weightName, float eps)
        {
            long t0 = Stopwatch.GetTimestamp();
            int tp = TpDegree;
            var alpha = _weights[weightName];

            // Materialize every rank's replica up front: TpReplicatedWeight fills
            // a shared dictionary, which the concurrent rank workers below must
            // not touch.
            var alphaLocal = new Tensor[tp];
            for (int r = 0; r < tp; r++)
                alphaLocal[r] = TpReplicatedWeight(alpha, r);

            var results = new Tensor[tp];
            _tpGroup.RunPerRank(r =>
            {
                int rows = (int)inputs[r].Sizes[0];
                int dim = (int)(inputs[r].ElementCount() / rows);
                Tensor input2d = inputs[r].Sizes.Length != 2 ? inputs[r].View(rows, dim) : null;
                Tensor src = input2d ?? inputs[r];
                results[r] = Ops.RMSNorm(null, src, alphaLocal[r], null, eps);
                input2d?.Dispose();
            });

            _normTicks += Stopwatch.GetTimestamp() - t0;
            return results;
        }

        // ====================================================================
        // TP-aware teardown
        // ====================================================================

        private void DisposeMuseGlimmerTpState()
        {
            ResetMuseGlimmerTpGraphs();

            if (_tpKvCacheK != null)
            {
                for (int l = 0; l < _tpKvCacheK.Length; l++)
                {
                    if (_tpKvCacheK[l] == null) continue;
                    for (int r = 0; r < _tpKvCacheK[l].Length; r++)
                    {
                        _tpKvCacheK[l][r]?.Dispose();
                        _tpKvCacheV[l][r]?.Dispose();
                    }
                }
                _tpKvCacheK = null;
                _tpKvCacheV = null;
            }

            DisposeTpWeightReplicaCache();
        }

        /// <summary>
        /// Zero every rank's cache in place, for
        /// <see cref="MuseGlimmerModel.ResetKVCacheCore"/>. Also drops the parked
        /// per-rank graphs, which bake the start position the reset just moved.
        /// </summary>
        private void ResetTpKVCache()
        {
            // Nothing on-device is worth keeping past a reset, so the pending
            // device->host sync is simply dropped.
            _tpKvDeviceDirty = false;
            ResetMuseGlimmerTpGraphs();
            if (_tpKvCacheK == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                for (int r = 0; r < _tpKvCacheK[l].Length; r++)
                {
                    ResetCacheTensor(_tpKvCacheK[l][r]);
                    ResetCacheTensor(_tpKvCacheV[l][r]);
                }
            }
        }
    }
}
