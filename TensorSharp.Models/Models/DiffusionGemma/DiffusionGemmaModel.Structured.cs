// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    public sealed partial class DiffusionGemmaModel
    {
        private int[] _structuredPromptTokens;
        private DiffusionSeqState _structuredPrompt;
        private bool _structuredPromptUsesFusedAttention;
        // The retained image spans are part of what a prompt prefill bakes into the cached K/V, but they
        // are NOT part of the token array, so the token comparison below cannot see them change.
        private int _structuredPromptVisionSpanVersion = -1;
        private int[] _structuredLabelTokens;
        private QuantizedWeight _structuredHead;
        private Tensor _structuredFloatHead;

        /// <summary>
        /// Read a seeded Jev canvas once, without sampling, committing tokens, or self-conditioning.
        /// Each result row is the temperature-one conditional distribution over its requested label
        /// token IDs, in the caller's order, after the model's final logit softcap. Positions are
        /// zero-based within the canvas. The canvas may be shorter than <see cref="CanvasLength"/>.
        /// Only requested hidden positions and label weight rows are projected; no full vocabulary
        /// logits are materialized. The caller's arrays are never modified and returned arrays are owned.
        /// </summary>
        /// <remarks>
        /// Like generation, this API requires exclusive model access. It retains at most one prompt's
        /// K/V and one label head for repeated reads; <see cref="ClearStructuredCache"/> releases them.
        /// Cancellation is observed between managed layers and before/after a fused native dispatch.
        /// </remarks>
        public float[][] ReadStructured(int[] promptTokens, int[] seedCanvas, int[] positions,
            int[][] tokenIds, CancellationToken cancellationToken = default)
            => ReadStructuredCore(promptTokens, seedCanvas, positions, tokenIds, false, cancellationToken);

        /// <summary>
        /// Full-vocabulary reference for validating and benchmarking <see cref="ReadStructured"/>.
        /// Runs the identical zero-self-conditioning forward but projects and downloads the full canvas
        /// vocabulary before restricting and normalizing the labels. This intentionally uses more memory.
        /// </summary>
        public float[][] ReadStructuredReference(int[] promptTokens, int[] seedCanvas, int[] positions,
            int[][] tokenIds, CancellationToken cancellationToken = default)
            => ReadStructuredCore(promptTokens, seedCanvas, positions, tokenIds, true, cancellationToken);

        private float[][] ReadStructuredCore(int[] promptTokens, int[] seedCanvas, int[] positions,
            int[][] tokenIds, bool fullVocabulary, CancellationToken cancellationToken)
        {
            ValidateStructuredRequest(promptTokens, seedCanvas, positions, tokenIds,
                VocabSize, CanvasLength, MaxContextLength);
            cancellationToken.ThrowIfCancellationRequested();
            // The selected-label read looks at a few canvas rows only; a backend that can skip the other
            // rows' last-layer work returns just these (compact), in ascending order.
            int[] keep = fullVocabulary ? null : positions.Distinct().Order().ToArray();
            using Tensor hidden = StructuredCanvasHidden(promptTokens, seedCanvas, cancellationToken, keep, out bool compact);
            cancellationToken.ThrowIfCancellationRequested();

            float[][] rows;
            if (fullVocabulary)
            {
                bool unitScale = !_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight fullHead)
                    || fullHead.Scale == 1f;
                float[] logits = IsGgmlBackend && !_fusedLmHeadTailDisabled && unitScale
                    ? TryFusedLmHead(hidden, seedCanvas.Length) : null;
                if (logits == null)
                {
                    using Tensor normalized = RMSNormOp(hidden, "output_norm.weight");
                    using Tensor projected = LinearForward(normalized, "token_embd.weight");
                    logits = ReadbackFresh(projected, checked(seedCanvas.Length * VocabSize));
                    ApplyStructuredSoftcap(logits);
                }
                rows = new float[positions.Length][];
                for (int r = 0; r < rows.Length; r++)
                {
                    rows[r] = new float[tokenIds[r].Length];
                    for (int j = 0; j < rows[r].Length; j++)
                        rows[r][j] = logits[checked(positions[r] * VocabSize + tokenIds[r][j])];
                }
            }
            else
            {
                var labels = new SortedSet<int>();
                foreach (int[] row in tokenIds)
                    foreach (int id in row) labels.Add(id);
                var union = new int[labels.Count];
                labels.CopyTo(union);
                EnsureStructuredHead(union);

                // Output normalization is row independent, so gather positions first as well.
                using var selected = new Tensor(_allocator, DType.Float32, positions.Length, Config.HiddenSize);
                for (int r = 0; r < positions.Length; r++)
                {
                    using Tensor source = hidden.Narrow(0, compact ? Array.BinarySearch(keep, positions[r]) : positions[r], 1);
                    using Tensor target = selected.Narrow(0, r, 1);
                    Ops.Copy(target, source);
                }
                using Tensor normalized = RMSNormOp(selected, "output_norm.weight");
                using var projected = new Tensor(_allocator, DType.Float32, positions.Length, union.Length);
                if (_structuredHead != null)
                {
                    AddmmQuantManaged(projected, normalized, _structuredHead);
                    if (_structuredHead.Scale != 1f)
                        Ops.Mul(projected, projected, _structuredHead.Scale);
                }
                else
                {
                    using Tensor transpose = _structuredFloatHead.Transpose();
                    Ops.Addmm(projected, 0, projected, 1, normalized, transpose);
                }
                float[] logits = ReadbackFresh(projected, checked(positions.Length * union.Length));
                ApplyStructuredSoftcap(logits);
                rows = new float[positions.Length][];
                for (int r = 0; r < rows.Length; r++)
                {
                    rows[r] = new float[tokenIds[r].Length];
                    for (int j = 0; j < rows[r].Length; j++)
                        rows[r][j] = logits[r * union.Length + Array.BinarySearch(union, tokenIds[r][j])];
                }
            }
            cancellationToken.ThrowIfCancellationRequested();
            foreach (float[] row in rows) NormalizeStructuredLabels(row);
            return rows;
        }

        private Tensor StructuredCanvasHidden(int[] promptTokens, int[] canvas, CancellationToken cancellationToken,
            int[] keepRows, out bool compact)
        {
            compact = false;
            // Raw CUDA/MLX's inherited cached global-RoPE path does not include startPos.
            // Their unified forward supplies full absolute positions and is the correct fallback.
            // The pure-C# CPU host prompt-KV path rotates canvas rows at absolute positions P+i, so it
            // reuses the prompt's K/V across repeated reads (Jev auto sampling re-reads one prompt).
            if (!SupportsPromptKvCache || (!IsGgmlBackend && !UsesHostPromptKv))
            {
                var tokens = new int[checked(promptTokens.Length + canvas.Length)];
                promptTokens.CopyTo(tokens, 0);
                canvas.CopyTo(tokens, promptTokens.Length);
                return ForwardCanvasHidden(tokens, promptTokens.Length, cancellationToken: cancellationToken);
            }

            if (_structuredPromptTokens == null || !_structuredPromptTokens.AsSpan().SequenceEqual(promptTokens)
                || _structuredPromptUsesFusedAttention != UseFusedPromptAttention
                || _structuredPromptVisionSpanVersion != _visionSpanVersion)
            {
                DisposeSeqState(_structuredPrompt);
                _structuredPromptTokens = null;
                _structuredPrompt = CreateSeqState();
                try
                {
                    _structuredPrompt.PromptLen = PrefillPromptInto(promptTokens,
                        _structuredPrompt.PromptK, _structuredPrompt.PromptV, cancellationToken);
                    _structuredPromptTokens = (int[])promptTokens.Clone();
                    _structuredPromptUsesFusedAttention = UseFusedPromptAttention;
                    _structuredPromptVisionSpanVersion = _visionSpanVersion;
                }
                catch
                {
                    DisposeSeqState(_structuredPrompt);
                    _structuredPrompt = null;
                    throw;
                }
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (UsesHostPromptKv && keepRows != null && keepRows.Length > 0 && keepRows.Length < canvas.Length)
            {
                compact = true;
                return CpuDecodeHidden(_structuredPrompt.PromptK, _structuredPrompt.PromptV,
                    _structuredPrompt.PromptLen, canvas, null, 0f, 1f, cancellationToken, keepRows);
            }
            return DecodeCanvasHidden(_structuredPrompt.PromptK, _structuredPrompt.PromptV,
                _structuredPrompt.PromptLen, canvas, null, 0f, 1f, cancellationToken);
        }

        private unsafe void EnsureStructuredHead(int[] labels)
        {
            if (_structuredLabelTokens != null && _structuredLabelTokens.AsSpan().SequenceEqual(labels)) return;
            ReleaseStructuredHead();
            if (_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight source) && source.HasHostData)
            {
                // Copy original packed rows; do not dequantize/requantize. This keeps the same
                // quantization and scale as the full tied lm-head, with only O(labels * hidden) bytes.
                long rowBytes = NativeDequant.RowSize(source.GgmlType, source.Ne0);
                long size = checked(rowBytes * labels.Length);
                IntPtr buffer = QuantizedWeight.AllocateBuffer(size);
                try
                {
                    for (int i = 0; i < labels.Length; i++)
                        Buffer.MemoryCopy((byte*)source.Data + labels[i] * rowBytes,
                            (byte*)buffer + i * rowBytes, rowBytes, rowBytes);
                    _structuredHead = new QuantizedWeight(buffer, size, source.GgmlType, source.Ne0, labels.Length)
                    { Scale = source.Scale };
                }
                catch
                {
                    QuantizedWeight.FreeBuffer(buffer);
                    throw;
                }
            }
            else
            {
                _structuredFloatHead = Embedding(labels);
                if (source != null && source.Scale != 1f)
                    Ops.Mul(_structuredFloatHead, _structuredFloatHead, source.Scale);
            }
            _structuredLabelTokens = (int[])labels.Clone();
        }

        private void ApplyStructuredSoftcap(float[] values)
        {
            if (_finalLogitSoftcap <= 0f) return;
            float inverse = 1f / _finalLogitSoftcap;
            for (int i = 0; i < values.Length; i++)
                values[i] = MathF.Tanh(values[i] * inverse) * _finalLogitSoftcap;
        }

        internal static void NormalizeStructuredLabels(float[] logits)
        {
            float max = float.NegativeInfinity;
            foreach (float value in logits)
            {
                if (!float.IsFinite(value))
                    throw new InvalidOperationException("The model produced non-finite structured label logits.");
                max = Math.Max(max, value);
            }
            double denominator = 0;
            foreach (float value in logits) denominator += Math.Exp((double)value - max);
            for (int i = 0; i < logits.Length; i++)
                logits[i] = (float)(Math.Exp((double)logits[i] - max) / denominator);
        }

        internal static void ValidateStructuredRequest(int[] promptTokens, int[] seedCanvas, int[] positions,
            int[][] tokenIds, int vocabSize, int canvasLength, int contextLength)
        {
            ArgumentNullException.ThrowIfNull(promptTokens);
            ArgumentNullException.ThrowIfNull(seedCanvas);
            ArgumentNullException.ThrowIfNull(positions);
            ArgumentNullException.ThrowIfNull(tokenIds);
            if (promptTokens.Length == 0) throw new ArgumentException("A nonempty prompt is required.", nameof(promptTokens));
            if (seedCanvas.Length == 0 || seedCanvas.Length > canvasLength)
                throw new ArgumentOutOfRangeException(nameof(seedCanvas), $"Canvas length must be between 1 and {canvasLength}.");
            if (contextLength > 0 && (long)promptTokens.Length + seedCanvas.Length > contextLength)
                throw new ArgumentException($"Prompt plus canvas exceeds the {contextLength}-token context window.", nameof(promptTokens));
            if (positions.Length == 0 || positions.Length != tokenIds.Length)
                throw new ArgumentException("Positions and label rows must have the same nonzero length.", nameof(positions));
            foreach (int id in promptTokens)
                if ((uint)id >= (uint)vocabSize) throw new ArgumentOutOfRangeException(nameof(promptTokens), "Prompt token outside vocabulary.");
            foreach (int id in seedCanvas)
                if ((uint)id >= (uint)vocabSize) throw new ArgumentOutOfRangeException(nameof(seedCanvas), "Canvas token outside vocabulary.");
            for (int r = 0; r < positions.Length; r++)
            {
                if ((uint)positions[r] >= (uint)seedCanvas.Length)
                    throw new ArgumentOutOfRangeException(nameof(positions), "Position outside canvas.");
                if (tokenIds[r] == null || tokenIds[r].Length == 0)
                    throw new ArgumentException("Each label row must contain at least one token.", nameof(tokenIds));
                var seen = new HashSet<int>();
                foreach (int id in tokenIds[r])
                {
                    if ((uint)id >= (uint)vocabSize)
                        throw new ArgumentOutOfRangeException(nameof(tokenIds), "Label token outside vocabulary.");
                    if (!seen.Add(id)) throw new ArgumentException("Label tokens must be distinct within each row.", nameof(tokenIds));
                }
            }
        }

        private void ReleaseStructuredHead()
        {
            if (_structuredHead != null)
            {
                if (IsGgmlBackend) GgmlBasicOps.InvalidateHostBuffer(_structuredHead.CacheKey);
                if (_allocator is CudaAllocator cuda) CudaQuantizedOps.ReleaseQuantizedWeight(cuda, _structuredHead.CacheKey);
                if (_allocator is MlxAllocator mlx) MlxQuantizedOps.ReleaseQuantizedWeight(mlx, _structuredHead.CacheKey);
                _structuredHead.Dispose();
                _structuredHead = null;
            }
            _structuredFloatHead?.Dispose();
            _structuredFloatHead = null;
            _structuredLabelTokens = null;
        }

        /// <summary>Release the bounded prompt and selected-label caches retained by structured reads.</summary>
        public void ClearStructuredCache()
        {
            DisposeSeqState(_structuredPrompt);
            _structuredPrompt = null;
            _structuredPromptTokens = null;
            ReleaseStructuredHead();
        }
    }
}
