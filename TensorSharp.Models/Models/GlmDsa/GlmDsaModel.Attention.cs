// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// glm-dsa attention: MLA with weight absorption + the DSA "lightning indexer".
//
// MLA stores ONE compressed row per token — [kv_lora_rank | rope] — and folds
// the per-head K/V decompression into the query and the output instead:
//
//   q_absorbed[h] = wk_b[h]^T @ q_nope[h]            (n_embd_head_qk_nope -> kv_lora_rank)
//   scores[h]     = ([q_absorbed[h] | q_pe[h]] . [kv_cmpr | k_pe]) * kq_scale
//   ctx[h]        = softmax(scores[h]) @ kv_cmpr
//   out[h]        = wv_b[h]^T @ ctx[h]               (kv_lora_rank -> n_embd_head_v_mla)
//
// so the 64 query heads attend to a single 576-wide key head (MQA), which is
// exactly what makes a 1M-token context affordable: 576 floats per token per
// layer instead of 64 x (256 + 256).
//
// The indexer scores every cached token with a cheap 32-head/128-dim product,
// and only its top-k (2048) survive into the real attention mask. Layers with
// indexer_full = false skip the indexer and reuse the previous full layer's
// selection.
using System;
using System.Diagnostics;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        private Tensor _cachedPosQ;   // [seqLen * n_head]
        private Tensor _cachedPosK;   // [seqLen]
        private Tensor _cachedPosIdx; // [seqLen * indexer_heads]
        private int _cachedPosSeqLen = -1;
        private int _cachedPosStartPos = -1;

        /// <summary>
        /// Cached [seqLen * heads] int32 position tensor, laid out so that row
        /// (s * heads + h) holds startPos + s — the layout <see cref="Ops.RoPEEx"/>
        /// wants for a [1, seqLen, heads, headDim] view.
        /// </summary>
        private Tensor EnsurePositions(int seqLen, int startPos, int heads, ref Tensor slot)
        {
            if (_cachedPosSeqLen != seqLen || _cachedPosStartPos != startPos)
            {
                _cachedPosQ?.Dispose(); _cachedPosQ = null;
                _cachedPosK?.Dispose(); _cachedPosK = null;
                _cachedPosIdx?.Dispose(); _cachedPosIdx = null;
                _cachedPosSeqLen = seqLen;
                _cachedPosStartPos = startPos;
                slot = null;
            }
            if (slot != null)
                return slot;

            int rows = seqLen * heads;
            var positions = new int[rows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < heads; h++)
                    positions[s * heads + h] = startPos + s;
            slot = CreateIntTensor(positions, rows);
            return slot;
        }

        /// <param name="dense">
        /// Skip the DSA lightning indexer and attend over the whole causal window.
        /// The NextN/MTP draft block runs this way: llama.cpp's <c>graph_mtp</c>
        /// builds it with the plain MLA attention input and never touches the
        /// indexer, even though the block ships indexer weights. Passing the
        /// trunk's selection in instead would be a DIFFERENT model, and the block
        /// has no indexer key cache to score against anyway.
        /// </param>
        private Tensor Attention(Tensor input, int layer, string[] wn, int seqLen, int startPos, bool dense = false)
        {
            int nHead = Config.NumHeads;
            int hidden = Config.HiddenSize;
            int kvLen = startPos + seqLen;

            // ---- shared query LoRA ------------------------------------------
            Tensor qr = LinearForward(input, wn[WN_Q_A]);                  // [T, q_lora_rank]
            Tensor qrNorm = Ops.RMSNorm(null, qr, _weights[wn[WN_Q_A_NORM]], null, Config.Eps);
            qr.Dispose();
            Trace("qr", layer, qrNorm);

            // ---- DSA lightning indexer --------------------------------------
            int[] topK = null;
            int topKPerToken = 0;
            if (!dense && SparseAttentionActive(kvLen))
            {
                // Only "full" layers score the cache; the rest inherit the most
                // recent full layer's selection, which is what makes GLM-5.2's
                // indexer affordable (21 of 78 layers pay for it).
                if (_indexerFull[layer])
                    RunLightningIndexer(input, qrNorm, layer, wn, seqLen, startPos);
                if (_sharedTopKCount > 0)
                {
                    topK = _sharedTopK;
                    topKPerToken = _sharedTopKCount;
                }
            }
            else if (!dense && _indexerFull[layer])
            {
                // Still append this token's indexer key so a later, longer step can
                // score it. Cheap: one [T, 128] projection per full layer.
                AppendIndexerKeys(input, layer, wn, seqLen, startPos);
            }

            // ---- queries ------------------------------------------------------
            Tensor q = LinearForward(qrNorm, wn[WN_Q_B]);                  // [T, n_head * head_k]
            qrNorm.Dispose();
            Trace("q", layer, q);

            // glm5next is NoPE MLA (rope.dimension_count = 0): there is no rope
            // half anywhere, _headDimNope == _headDimK, and the compressed latent
            // IS the whole cache row. Splitting anyway narrows a zero-width slice.
            // Mirrors the `hp.n_rot == 0` branches in ggml_ops_glm_dsa.cpp.
            bool nopeOnly = _ropeDim == 0;

            Tensor qNopeH, qPeH = null;
            using (var q3 = q.View(seqLen, nHead, _headDimK))
            {
                using (var nopeView = q3.Narrow(2, 0, _headDimNope))
                using (var t = nopeView.Transpose(0, 1))
                    qNopeH = Ops.NewContiguous(t);                          // [H, T, nope]
                if (!nopeOnly)
                    using (var peView = q3.Narrow(2, _headDimNope, _ropeDim))
                        qPeH = Ops.NewContiguous(peView);                   // [T, H, rope]
            }
            q.Dispose();

            if (!nopeOnly)
            {
                var posQ = EnsurePositions(seqLen, startPos, nHead, ref _cachedPosQ);
                using (var qPeRope = qPeH.View(1, seqLen, nHead, _ropeDim))
                    Ops.RoPEEx(qPeRope, qPeRope, posQ, _ropeDim, 0, Config.OriginalContextLength,
                        Config.RopeBase, 1.0f / Config.RopeScale, 0.0f, 1.0f, 0.0f, 0.0f);
            }

            // [T, H, rope] -> [H, T, rope] so it shares the query-head-major layout
            // of q_absorbed and both can drive one 2D GEMM against the cache.
            Tensor qPeHeadMajor = null;
            if (!nopeOnly)
            {
                using (var t = qPeH.Transpose(0, 1))
                    qPeHeadMajor = Ops.NewContiguous(t);
                qPeH.Dispose();
            }

            // q_absorbed[h] = q_nope[h] @ wk_b[h]^T : [H, T, nope] x [H, nope, kv_lora]
            var qAbs = new Tensor(_allocator, DType.Float32, nHead, seqLen, _kvLoraRank);
            using (var wkT = _wkB[layer].Transpose(1, 2))
                Ops.AddmmBatch(qAbs, 0, qAbs, 1.0f, qNopeH, wkT);
            qNopeH.Dispose();
            Trace("q_nope_absorbed", layer, qAbs);
            Trace("q_pe", layer, qPeHeadMajor);

            // ---- compressed KV for the new tokens -----------------------------
            Tensor kvPe = LinearForward(input, wn[WN_KV_A_MQA]);            // [T, kv_lora + rope]
            Tensor kvCmpr, kPe;
            using (var cmprView = kvPe.Narrow(1, 0, _kvLoraRank))
            using (var cmpr = Ops.NewContiguous(cmprView))
                kvCmpr = Ops.RMSNorm(null, cmpr, _weights[wn[WN_KV_A_NORM]], null, Config.Eps);
            kPe = null;
            if (!nopeOnly)
            {
                using (var peView = kvPe.Narrow(1, _kvLoraRank, _ropeDim))
                    kPe = Ops.NewContiguous(peView);
            }
            kvPe.Dispose();

            if (!nopeOnly)
            {
                var posK = EnsurePositions(seqLen, startPos, 1, ref _cachedPosK);
                using (var kPeRope = kPe.View(1, seqLen, 1, _ropeDim))
                    Ops.RoPEEx(kPeRope, kPeRope, posK, _ropeDim, 0, Config.OriginalContextLength,
                        Config.RopeBase, 1.0f / Config.RopeScale, 0.0f, 1.0f, 0.0f, 0.0f);
            }

            using (var dstCmpr = _kvCache[layer].Narrow(0, startPos, seqLen))
                Ops.Copy(dstCmpr, kvCmpr);
            if (!nopeOnly)
            {
                using (var dstPe = _kPeCache[layer].Narrow(0, startPos, seqLen))
                    Ops.Copy(dstPe, kPe);
            }
            Trace("kv_cmpr", layer, kvCmpr);
            if (kPe != null) Trace("k_pe", layer, kPe);
            kvCmpr.Dispose();
            kPe?.Dispose();

            // ---- attention ----------------------------------------------------
            long t0 = Stopwatch.GetTimestamp();

            int rowsQ = nHead * seqLen;
            var scores = new Tensor(_allocator, DType.Float32, rowsQ, kvLen);
            using (var kvActive = _kvCache[layer].Narrow(0, 0, kvLen))
            using (var kvActiveT = kvActive.Transpose())
            using (var qAbs2d = qAbs.View(rowsQ, _kvLoraRank))
                Ops.Addmm(scores, 0, scores, _kqScale, qAbs2d, kvActiveT);
            if (!nopeOnly)
            {
                using (var peActive = _kPeCache[layer].Narrow(0, 0, kvLen))
                using (var peActiveT = peActive.Transpose())
                using (var qPe2d = qPeHeadMajor.View(rowsQ, _ropeDim))
                    Ops.Addmm(scores, 1.0f, scores, _kqScale, qPe2d, peActiveT);
            }
            qAbs.Dispose();
            qPeHeadMajor?.Dispose();

            using (var scores3 = scores.View(nHead, seqLen, kvLen))
                ApplyAttentionMaskAndSoftmax(scores3, seqLen, startPos, kvLen, topK, topKPerToken);

            var ctx = new Tensor(_allocator, DType.Float32, rowsQ, _kvLoraRank);
            using (var kvActive = _kvCache[layer].Narrow(0, 0, kvLen))
                Ops.Addmm(ctx, 0, ctx, 1.0f, scores, kvActive);
            scores.Dispose();

            // out[h] = ctx[h] @ wv_b[h]^T : [H, T, kv_lora] x [H, kv_lora, head_v]
            var headOut = new Tensor(_allocator, DType.Float32, nHead, seqLen, _headDimV);
            using (var ctx3 = ctx.View(nHead, seqLen, _kvLoraRank))
            using (var wvT = _wvB[layer].Transpose(1, 2))
                Ops.AddmmBatch(headOut, 0, headOut, 1.0f, ctx3, wvT);
            ctx.Dispose();

            Tensor flat;
            using (var t = headOut.Transpose(0, 1))
            using (var contiguous = Ops.NewContiguous(t))
                flat = contiguous.View(seqLen, nHead * _headDimV);
            headOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;
            Trace("kqv_out", layer, flat);

            Tensor output = LinearForward(flat, wn[WN_ATTN_OUT]);
            flat.Dispose();
            return output;
        }

        /// <summary>
        /// Whether the indexer can actually remove anything at this context length.
        ///
        /// <para>llama.cpp always builds the indexer, takes <c>top_k =
        /// min(n_kv, indexer_top_k)</c>, and unmasks those cells. When the cache
        /// holds no more cells than top-k, every cell is unmasked and the mask
        /// collapses to the plain causal one — the indexer cannot change a single
        /// score. Skipping it there is not an approximation, it is the same
        /// function computed cheaply, and it keeps short prompts (the common case)
        /// off a path that costs an extra [tokens x 32 x n_kv] product per full
        /// layer.</para>
        /// </summary>
        private bool SparseAttentionActive(int kvLen) => kvLen > _indexerTopK;

        /// <summary>
        /// Project and cache this chunk's indexer keys without scoring anything.
        /// Used while the context is still shorter than top-k: the keys must be in
        /// the cache before a later, longer step scores them.
        /// </summary>
        private void AppendIndexerKeys(Tensor input, int layer, string[] wn, int seqLen, int startPos)
        {
            Tensor k = ComputeIndexerKeys(input, wn, seqLen, startPos);
            using (var dst = _indexerCache[layer].Narrow(0, startPos, seqLen))
                Ops.Copy(dst, k);
            k.Dispose();
        }

        private Tensor ComputeIndexerKeys(Tensor input, string[] wn, int seqLen, int startPos)
        {
            Tensor k = LinearForward(input, wn[WN_IDX_K]);                       // [T, 128]
            Tensor kNorm = Ops.LayerNorm(null, k, _weights[wn[WN_IDX_K_NORM_W]],
                                         _weights[wn[WN_IDX_K_NORM_B]], IndexerNormEps);
            k.Dispose();

            // The native indexer ropes unconditionally with hp.n_rot, which is a
            // no-op at n_rot == 0 (glm5next); skipping it outright is the same
            // thing without depending on RoPEEx's zero-width behaviour.
            if (_ropeDim > 0)
            {
                var posK = EnsurePositions(seqLen, startPos, 1, ref _cachedPosK);
                using (var rope = kNorm.View(1, seqLen, 1, _indexerHeadDim))
                    Ops.RoPEEx(rope, rope, posK, _ropeDim, 0, Config.OriginalContextLength,
                        Config.RopeBase, 1.0f / Config.RopeScale, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            return kNorm;
        }

        /// <summary>
        /// glm-dsa carries no <c>attention.layer_norm_epsilon</c>, so llama.cpp's
        /// indexer LayerNorm runs with hparams.f_norm_eps left at its zero-initialised
        /// default. Matching that exactly matters: the indexer keys are 128-wide and
        /// a 1e-5 epsilon visibly shifts their norm.
        /// </summary>
        private const float IndexerNormEps = 0.0f;

        /// <summary>
        /// Score every cached token for every query token and keep the top-k.
        ///
        /// <para>score[t, j] = sum_h relu(q[t,h] . k[j]) * w[t,h], masked causally.
        /// The Hadamard rotation llama.cpp applies to q and k is deliberately not
        /// reproduced: it is an orthogonal, symmetric involution applied to BOTH
        /// sides of that dot product, so it cancels exactly. Its purpose upstream is
        /// to spread outliers before a QUANTIZED indexer-key cache rounds them; this
        /// cache is F32, so rotating and un-rotating would only add float error.</para>
        /// </summary>
        private void RunLightningIndexer(Tensor input, Tensor qrNorm, int layer, string[] wn,
            int seqLen, int startPos)
        {
            int H = _indexerHeads;
            int D = _indexerHeadDim;
            int kvLen = startPos + seqLen;

            Tensor k = ComputeIndexerKeys(input, wn, seqLen, startPos);
            using (var dst = _indexerCache[layer].Narrow(0, startPos, seqLen))
                Ops.Copy(dst, k);
            k.Dispose();

            Tensor q = LinearForward(qrNorm, wn[WN_IDX_Q_B]);                     // [T, H*D]
            if (_ropeDim > 0)
            {
                var posIdx = EnsurePositions(seqLen, startPos, H, ref _cachedPosIdx);
                using (var rope = q.View(1, seqLen, H, D))
                    Ops.RoPEEx(rope, rope, posIdx, _ropeDim, 0, Config.OriginalContextLength,
                        Config.RopeBase, 1.0f / Config.RopeScale, 0.0f, 1.0f, 0.0f, 0.0f);
            }

            Tensor w = LinearForward(input, wn[WN_IDX_PROJ]);                     // [T, H]
            // Pre-scaled exactly as upstream, so the scale never touches the big
            // [tokens x n_kv] score tensor.
            float wScale = 1.0f / MathF.Sqrt((float)(D * H));

            Trace("indexer_w", layer, w);
            float[] wHost = TensorToFloatArray(w);
            w.Dispose();
            for (int i = 0; i < wHost.Length; i++)
                wHost[i] *= wScale;

            int topKCount = Math.Min(kvLen, _indexerTopK);
            int needed = seqLen * topKCount;
            if (_sharedTopK == null || _sharedTopK.Length < needed)
                _sharedTopK = new int[needed];
            int[] topK = _sharedTopK;
            _sharedTopKCount = topKCount;

            // Chunk over query tokens: the raw per-head score block is
            // [chunk * H, kvLen] floats and would otherwise be gigabytes on a long
            // prompt (512 tokens x 32 heads x 32K cells = 2 GiB).
            int chunk = Math.Max(1, IndexerChunkTokens(kvLen, H));
            var block = new Tensor(_allocator, DType.Float32, chunk * H, kvLen);
            try
            {
                for (int t0 = 0; t0 < seqLen; t0 += chunk)
                {
                    int tn = Math.Min(chunk, seqLen - t0);
                    using (var qChunk = q.Narrow(0, t0, tn))
                    using (var qChunk2d = qChunk.View(tn * H, D))
                    using (var blockRows = block.Narrow(0, 0, tn * H))
                    using (var kActive = _indexerCache[layer].Narrow(0, 0, kvLen))
                    using (var kActiveT = kActive.Transpose())
                    {
                        Ops.Addmm(blockRows, 0, blockRows, 1.0f, qChunk2d, kActiveT);
                        if (t0 == 0)
                        {
                            Trace("indexer_q", layer, qChunk2d);
                            Trace("indexer_k", layer, kActive);
                            Trace("indexer_kq", layer, blockRows);
                        }
                    }

                    ReduceAndSelectTopK(block, wHost, t0, tn, H, kvLen, startPos, topK, topKCount);
                }
            }
            finally
            {
                block.Dispose();
                q.Dispose();
            }
        }

        private static int IndexerChunkTokens(int kvLen, int heads)
        {
            // ~64 MiB of scratch per chunk.
            long budgetFloats = 16L * 1024 * 1024;
            long perToken = (long)heads * kvLen;
            long n = budgetFloats / Math.Max(1, perToken);
            return (int)Math.Clamp(n, 1, 256);
        }

        /// <summary>
        /// Collapse the per-head indexer products into one score per (token, cell),
        /// mask the future, and take the top-k cells. Runs on the host: the reduction
        /// is a relu-weighted sum that no generic tensor op expresses, and the top-k
        /// result is consumed as a host-side mask anyway.
        /// </summary>
        private unsafe void ReduceAndSelectTopK(Tensor block, float[] wHost, int tokenOffset, int tokenCount,
            int heads, int kvLen, int startPos, int[] topK, int topKCount)
        {
            InvalidateTensorDeviceCache(block);
            float* blockPtr = GetFloatPtr(block);

            Parallel.For(0, tokenCount, t =>
            {
                int globalToken = tokenOffset + t;
                int lastVisible = startPos + globalToken;          // inclusive
                int visible = lastVisible + 1;
                int outBase = globalToken * topKCount;

                if (visible <= topKCount)
                {
                    // Every visible cell survives; the causal mask alone is exact.
                    for (int i = 0; i < visible; i++)
                        topK[outBase + i] = i;
                    for (int i = visible; i < topKCount; i++)
                        topK[outBase + i] = -1;
                    return;
                }

                var scores = new float[visible];
                for (int h = 0; h < heads; h++)
                {
                    float weight = wHost[globalToken * heads + h];
                    float* row = blockPtr + (long)(t * heads + h) * kvLen;
                    for (int j = 0; j < visible; j++)
                    {
                        float v = row[j];
                        if (v > 0.0f)
                            scores[j] += v * weight;
                    }
                }

                SelectTopKIndices(scores, visible, topKCount, topK, outBase);
            });
        }

        /// <summary>
        /// Indices of the <paramref name="k"/> largest of <paramref name="n"/> scores,
        /// in no particular order (which is all the DSA mask needs).
        ///
        /// <para>Quickselect on a value copy, then one pass to collect. The obvious
        /// alternative — the shared <c>SelectTopKInPlace</c> min-heap — is O(n*k),
        /// and DSA's k is 2048 against an n of tens of thousands, so it would cost
        /// ~10^8 comparisons per token per layer.</para>
        /// </summary>
        private static void SelectTopKIndices(float[] scores, int n, int k, int[] dst, int dstOffset)
        {
            if (k >= n)
            {
                for (int i = 0; i < n; i++) dst[dstOffset + i] = i;
                for (int i = n; i < k; i++) dst[dstOffset + i] = -1;
                return;
            }

            var values = new float[n];
            Array.Copy(scores, values, n);
            float threshold = NthLargest(values, n, k);

            int written = 0;
            // Strictly-greater first, so ties never overfill the output.
            for (int i = 0; i < n && written < k; i++)
            {
                if (scores[i] > threshold)
                    dst[dstOffset + written++] = i;
            }
            for (int i = 0; i < n && written < k; i++)
            {
                if (scores[i] == threshold)
                    dst[dstOffset + written++] = i;
            }
            while (written < k)
                dst[dstOffset + written++] = -1;
        }

        /// <summary>Value of the k-th largest element (1-based k), reordering <paramref name="v"/>.</summary>
        private static float NthLargest(float[] v, int n, int k)
        {
            int lo = 0, hi = n - 1, target = k - 1;
            var rng = new FastRng(unchecked((uint)n * 2654435761u + (uint)k));
            while (lo < hi)
            {
                int pivotIdx = lo + (int)(rng.Next() % (uint)(hi - lo + 1));
                float pivot = v[pivotIdx];
                (v[pivotIdx], v[hi]) = (v[hi], v[pivotIdx]);

                int store = lo;
                for (int i = lo; i < hi; i++)
                {
                    if (v[i] > pivot)
                    {
                        (v[i], v[store]) = (v[store], v[i]);
                        store++;
                    }
                }
                (v[store], v[hi]) = (v[hi], v[store]);

                if (store == target) return v[store];
                if (store < target) lo = store + 1;
                else hi = store - 1;
            }
            return v[lo];
        }

        /// <summary>xorshift32 — a deterministic pivot chooser, so a run is reproducible.</summary>
        private struct FastRng
        {
            private uint _state;
            public FastRng(uint seed) { _state = seed == 0 ? 0x9E3779B9u : seed; }
            public uint Next()
            {
                _state ^= _state << 13;
                _state ^= _state >> 17;
                _state ^= _state << 5;
                return _state;
            }
        }

        /// <summary>
        /// Causal mask (+ the DSA top-k mask when the indexer is active) followed by
        /// the row softmax.
        /// </summary>
        private unsafe void ApplyAttentionMaskAndSoftmax(Tensor scores3, int seqLen, int startPos, int kvLen,
            int[] topK, int topKPerToken)
        {
            int nHead = (int)scores3.Sizes[0];

            if (topK == null)
            {
                if (IsGgmlBackend)
                {
                    GgmlBasicOps.AttentionSoftmaxWithSinks(
                        scores3, sinks: null,
                        numHeads: nHead, seqLen: seqLen, kvLen: kvLen,
                        maskStartPos: startPos, slidingWindow: 0, scale: 1.0f);
                }
                else
                {
                    Ops.AddCausalMask(scores3, seqLen, startPos, float.NegativeInfinity);
                    Ops.Softmax(scores3, scores3);
                }
                return;
            }

            // Sparse: build the [seqLen, kvLen] additive mask once and broadcast it
            // over the heads, then softmax.
            var mask = new Tensor(_allocator, DType.Float32, seqLen, kvLen);
            float* maskPtr = GetFloatPtr(mask);
            Parallel.For(0, seqLen, t =>
            {
                float* row = maskPtr + (long)t * kvLen;
                for (int j = 0; j < kvLen; j++)
                    row[j] = float.NegativeInfinity;
                int outBase = t * topKPerToken;
                int lastVisible = startPos + t;
                for (int i = 0; i < topKPerToken; i++)
                {
                    int j = topK[outBase + i];
                    if (j >= 0 && j <= lastVisible)
                        row[j] = 0.0f;
                }
            });

            // The mask was just written on the host; the device copy keyed by that
            // buffer has to be dropped BEFORE it is used as an operand, or a
            // pooled buffer's stale upload is what the kernel reads.
            InvalidateTensorDeviceCache(mask);

            // Added head by head against a plain [seqLen, kvLen] operand rather
            // than through one stride-0 broadcast: the broadcast form is not
            // reproduced identically by every backend's elementwise kernel, and
            // materialising the [n_head, seqLen, kvLen] copy instead would cost
            // 64x the mask's memory.
            for (int h = 0; h < nHead; h++)
            {
                using var head = scores3.Select(0, h);
                Ops.Add(head, head, mask);
            }
            mask.Dispose();

            Ops.Softmax(scores3, scores3);
        }
    }
}
