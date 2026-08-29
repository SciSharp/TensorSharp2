// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 vision tower: the Qwen3-VL ViT of
// TSGgml_MiniMaxH3VisionEncode expressed against the shared direct primitives,
// so BackendType.Cpu (the 100% pure-C# backend) can do reference conditioning
// with no ggml.
//
// Five things differ from the two sibling Direct* stages in this folder and are
// easy to get wrong by copying them:
//   - every norm here is a MEAN-CENTRED LayerNorm with weight AND bias
//     (ggml_norm), not the RMSNorm the text encoder and the video VAE use;
//   - the QKV projection is a CONTIGUOUS q|k|v column split, not the per-head
//     interleaved layout MiniMaxH3DirectVideoVae.SplitInterleaved unpacks;
//   - the block MLP uses the tanh-approximated GELU while the four mergers use
//     the erf GELU, which TensorSharp has no op for;
//   - the final merger normalizes at width dim BEFORE the 2x2 fold while the
//     DeepStack mergers normalize at width dim*4 AFTER it, from identically
//     named tensors;
//   - the patch embedding is a 4-D GGUF tensor that only becomes a linear once
//     it is forced to [patchDim, dim], which neither generic DirectLinear
//     factory derives from the shape.
//
// The flash-attention V pre-scale in the native h3_attend is deliberately NOT
// ported: it exists only to keep ggml's FP16 softmax numerator finite, and this
// path is F32 end to end, so replicating it would be an exact no-op.
using System;
using System.Collections.Generic;
using System.Numerics;

using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed Qwen3-VL vision tower for MiniMax-H3 (no ggml).</summary>
    internal sealed class MiniMaxH3DirectVisionEncoder : IDisposable
    {
        private sealed class BlockW
        {
            public Tensor Norm1W, Norm1B, Norm2W, Norm2B;
            public DirectLinear Qkv, Proj, Fc1, Fc2;
        }

        private sealed class MergerW
        {
            public Tensor NormW, NormB;
            public DirectLinear Fc1, Fc2;
            /// <summary>True for the final merger (normalize at dim, before the fold),
            /// false for the DeepStack mergers (normalize at dim*4, after it).</summary>
            public bool NormBeforeMerge;
        }

        private readonly DirectContext _ctx;
        private readonly DirectLinear _patchEmbed;
        private readonly BlockW[] _blocks;
        private readonly MergerW[] _mergers;          // [0] final, [1..] DeepStack
        private readonly int[] _deepstackLayers;
        private readonly List<DirectLinear> _owned = new();
        private readonly int _dim, _heads, _headDim, _outDim, _patchDim, _mergeSize, _mergeDim;
        private readonly float _eps;
        private bool _disposed;

        /// <param name="gguf">Text-encoder checkpoint carrying the "visual.*" tower.</param>
        /// <param name="dim">Tower width (1152 here); must be heads * headDim.</param>
        /// <param name="heads">Attention heads; never stored in the checkpoint.</param>
        /// <param name="outDim">Merger output width, i.e. the language model's hidden size.</param>
        /// <param name="patchDim">Flattened patch length, 3 * temporal * patch * patch.</param>
        /// <param name="mergeSize">Spatial merge factor; each output token folds mergeSize^2 patches.</param>
        /// <param name="deepstackLayers">Block indices whose output feeds a DeepStack merger.</param>
        /// <param name="eps">LayerNorm epsilon, applied inside the sqrt.</param>
        /// <param name="allocator">Direct-backend allocator; CUDA is rejected.</param>
        public MiniMaxH3DirectVisionEncoder(GgufFile gguf, int dim, int heads, int outDim,
                                            int patchDim, int mergeSize, int[] deepstackLayers,
                                            float eps, IAllocator allocator)
        {
            if (gguf == null) throw new ArgumentNullException(nameof(gguf));
            if (deepstackLayers == null) throw new ArgumentNullException(nameof(deepstackLayers));
            _ctx = new DirectContext(allocator);
            if (_ctx.IsCuda)
                throw new NotSupportedException(
                    "MiniMaxH3DirectVisionEncoder is the managed CPU path; MiniMax-H3 on " +
                    "BackendType.Cuda is not implemented (use a GGML backend, or BackendType.Cpu).");

            if (heads <= 0 || dim <= 0 || dim % heads != 0)
                throw new ArgumentException($"bad vision head geometry: dim {dim}, heads {heads}.");
            if (mergeSize <= 0)
                throw new ArgumentException($"bad spatial merge factor {mergeSize}.");

            _dim = dim; _heads = heads; _headDim = dim / heads; _outDim = outDim;
            _patchDim = patchDim; _mergeSize = mergeSize; _mergeDim = dim * mergeSize * mergeSize;
            _eps = eps;
            _deepstackLayers = (int[])deepstackLayers.Clone();

            DirectLinear Lin(string w, string b)
            {
                var l = DirectLinear.FromGguf(_ctx, gguf, w, gguf.Tensors.ContainsKey(b ?? "") ? b : null);
                _owned.Add(l);
                return l;
            }
            Tensor Vec(string name, long expected)
            {
                long n = gguf.Tensors[name].NumElements;
                // The final and the DeepStack mergers have identically named norms of
                // DIFFERENT widths (dim vs dim*4), so a swap would otherwise be silent.
                if (n != expected)
                    throw new System.IO.InvalidDataException(
                        $"'{name}' has {n} elements, expected {expected}.");
                return _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(gguf, name), n));
            }
            MergerW Merger(string prefix, bool normBeforeMerge)
            {
                var m = new MergerW
                {
                    NormW = Vec(prefix + "norm.weight", normBeforeMerge ? dim : _mergeDim),
                    NormB = Vec(prefix + "norm.bias", normBeforeMerge ? dim : _mergeDim),
                    Fc1 = Lin(prefix + "linear_fc1.weight", prefix + "linear_fc1.bias"),
                    Fc2 = Lin(prefix + "linear_fc2.weight", prefix + "linear_fc2.bias"),
                    NormBeforeMerge = normBeforeMerge,
                };
                if (m.Fc1.InDim != _mergeDim || m.Fc1.OutDim != _mergeDim ||
                    m.Fc2.InDim != _mergeDim || m.Fc2.OutDim != outDim)
                    throw new System.IO.InvalidDataException(
                        $"'{prefix}' merger geometry is not {_mergeDim}->{_mergeDim}->{outDim}.");
                return m;
            }

            // The Conv3d kernel ships as ne [16, 16, 2, 3456]; read as a per-patch
            // linear it is the same memory laid out [patchDim, dim], which neither
            // DirectLinear factory computes from the shape (FromGguf gives
            // 16 x 110592, FromGgufPatch 512 x 3456). Dequantizing it costs only
            // ~7 MB of F32, so the explicit shape goes in through FromFloats rather
            // than warranting a whole new factory.
            if (gguf.Tensors["visual.patch_embed.proj.weight"].NumElements != (long)patchDim * dim)
                throw new System.IO.InvalidDataException(
                    "'visual.patch_embed.proj.weight' does not hold patchDim * dim elements; " +
                    "the 4-D to 2-D reinterpretation would silently scramble it.");
            _patchEmbed = DirectLinear.FromFloats(
                _ctx,
                DirectOps.DequantTensor(gguf, "visual.patch_embed.proj.weight"),
                patchDim, dim,
                gguf.Tensors.ContainsKey("visual.patch_embed.proj.bias")
                    ? DirectOps.DequantTensor(gguf, "visual.patch_embed.proj.bias")
                    : null);
            _owned.Add(_patchEmbed);

            // The block count is not passed in; probe it the way the driver does.
            int nl = 0;
            while (gguf.Tensors.ContainsKey($"visual.blocks.{nl}.norm1.weight")) nl++;
            if (nl == 0)
                throw new InvalidOperationException("this checkpoint has no vision tower blocks.");
            _blocks = new BlockW[nl];
            for (int i = 0; i < nl; i++)
            {
                string p = $"visual.blocks.{i}.";
                var bw = new BlockW
                {
                    Norm1W = Vec(p + "norm1.weight", dim),
                    Norm1B = Vec(p + "norm1.bias", dim),
                    Norm2W = Vec(p + "norm2.weight", dim),
                    Norm2B = Vec(p + "norm2.bias", dim),
                    Qkv = Lin(p + "attn.qkv.weight", p + "attn.qkv.bias"),
                    Proj = Lin(p + "attn.proj.weight", p + "attn.proj.bias"),
                    Fc1 = Lin(p + "mlp.linear_fc1.weight", p + "mlp.linear_fc1.bias"),
                    Fc2 = Lin(p + "mlp.linear_fc2.weight", p + "mlp.linear_fc2.bias"),
                };
                if (bw.Qkv.InDim != dim || bw.Qkv.OutDim != 3L * dim)
                    throw new System.IO.InvalidDataException(
                        $"'{p}attn.qkv.weight' is {bw.Qkv.InDim}x{bw.Qkv.OutDim}, expected {dim}x{3 * dim}.");
                if (bw.Fc1.OutDim != bw.Fc2.InDim)
                    throw new System.IO.InvalidDataException($"'{p}mlp' fc1/fc2 widths disagree.");
                _blocks[i] = bw;
            }

            foreach (int l in _deepstackLayers)
                if (l < 0 || l >= nl)
                    throw new ArgumentException(
                        $"DeepStack tap layer {l} is outside the {nl}-block tower.");

            _mergers = new MergerW[1 + _deepstackLayers.Length];
            _mergers[0] = Merger("visual.merger.", normBeforeMerge: true);
            for (int i = 0; i < _deepstackLayers.Length; i++)
                _mergers[1 + i] = Merger($"visual.deepstack_merger_list.{i}.", normBeforeMerge: false);
        }

        /// <summary>
        /// Run the tower on one preprocessed image or video block and return the
        /// packed [(1 + deepstack) * merged, outDim] row-major buffer the driver
        /// slices: row blocks [final | tap0 | tap1 | tap2], merged = tokens / mergeSize^2.
        /// </summary>
        /// <param name="patches">[tokens, patchDim] flattened patches in MERGE-BLOCK order.</param>
        /// <param name="pos">[tokens, dim] resampled position embeddings, same order.</param>
        /// <param name="cos">[tokens, headDim] rotate-half cosine table.</param>
        /// <param name="sin">[tokens, headDim] rotate-half sine table.</param>
        /// <param name="tokens">Pre-merge patch count, gridHeight * gridWidth.</param>
        public float[] Encode(float[] patches, float[] pos, float[] cos, float[] sin, int tokens)
        {
            if (patches == null) throw new ArgumentNullException(nameof(patches));
            if (pos == null) throw new ArgumentNullException(nameof(pos));
            if (cos == null) throw new ArgumentNullException(nameof(cos));
            if (sin == null) throw new ArgumentNullException(nameof(sin));

            int seq = tokens, hd = _headDim, blockTokens = _mergeSize * _mergeSize;
            if (seq <= 0 || seq % blockTokens != 0)
                throw new ArgumentException(
                    $"token count {seq} is not a whole number of {blockTokens}-patch merge blocks.");
            int merged = seq / blockTokens;
            // EXACT, not a minimum: these two go to DirectContext.FromFloats, and
            // CpuStorage.SetElementsAsFloat copies value.Length elements with no
            // check against the destination, so an over-long array would write past
            // the tensor buffer instead of being ignored. cos/sin below are read
            // through explicit indexing, so a minimum is the right check for them.
            if (patches.LongLength != (long)seq * _patchDim)
                throw new ArgumentException(
                    $"patches holds {patches.LongLength} floats; {seq} x {_patchDim} is required.",
                    nameof(patches));
            if (pos.LongLength != (long)seq * _dim)
                throw new ArgumentException(
                    $"pos holds {pos.LongLength} floats; {seq} x {_dim} is required.", nameof(pos));
            if (cos.LongLength < (long)seq * hd || sin.LongLength < (long)seq * hd)
                throw new ArgumentException($"cos/sin are shorter than {seq} x {hd}.");

            int visStop = VisStopStage();
            float scale = 1.0f / MathF.Sqrt(hd);
            int nds = _deepstackLayers.Length;

            // Mirrors the native's std::vector: slot 0 is reserved for the final
            // merger and the taps are appended in FIRING order, so an unsorted tap
            // list lands in the same output order as ggml's ggml_concat chain.
            var outputs = new List<float[]>(1 + nds) { null };

            Tensor h;
            using (Tensor px = _ctx.FromFloats(patches, seq, _patchDim))
                h = _patchEmbed.Forward(px);                       // [seq, dim]
            try
            {
                // Elementwise add of two [seq, dim] tensors, not a per-column broadcast.
                using (Tensor posT = _ctx.FromFloats(pos, seq, _dim))
                    Ops.Add(h, h, posT);
                if (visStop == 0) ReportProbe(0, h);

                for (int l = 0; l < _blocks.Length; l++)
                {
                    BlockW bw = _blocks[l];

                    using (Tensor n1 = LayerNormRows(h, bw.Norm1W, bw.Norm1B))
                    using (Tensor qkv = bw.Qkv.Forward(n1))        // [seq, 3*dim], contiguous q|k|v
                    {
                        Tensor q = Cols(qkv, 0, _dim);
                        Tensor k = Cols(qkv, _dim, _dim);
                        using Tensor v = Cols(qkv, 2 * _dim, _dim);
                        RopeHalf(q, cos, sin, seq, _heads, hd);
                        RopeHalf(k, cos, sin, seq, _heads, hd);    // V is not rotated
                        // Qwen3-VL runs FULL bidirectional attention on every layer;
                        // the windowed variant in the Qwen2.5-VL config is dead here.
                        using (Tensor attn = DirectOps.Attention(_ctx, q, k, v, _heads, hd, scale))
                        using (Tensor proj = bw.Proj.Forward(attn))
                            Ops.Add(h, h, proj);
                        q.Dispose();
                        k.Dispose();
                    }

                    using (Tensor n2 = LayerNormRows(h, bw.Norm2W, bw.Norm2B))
                    using (Tensor hid = bw.Fc1.Forward(n2))
                    // The block MLP takes the tanh-approximated GELU; the mergers take
                    // the erf one, and swapping them is a silent ~1% error.
                    using (Tensor act = DirectOps.Gelu(_ctx, hid))
                    using (Tensor down = bw.Fc2.Forward(act))
                        Ops.Add(h, h, down);

                    // The tap reads the block's FINAL h and must leave it untouched.
                    for (int t = 0; t < nds; t++)
                        if (_deepstackLayers[t] == l)
                            outputs.Add(RunMerger(_mergers[1 + t], h, merged));

                    if (visStop == l + 1) ReportProbe(l + 1, h);
                }

                // There is no final trunk norm: the merger's own norm is the only one.
                outputs[0] = RunMerger(_mergers[0], h, merged);
            }
            finally
            {
                h.Dispose();
            }

            // The only thing that catches a tap layer that never matched a block.
            if (outputs.Count != 1 + nds)
                throw new InvalidOperationException(
                    "MiniMax-H3 vision encode: DeepStack tap layers did not all fire.");

            long span = (long)merged * _outDim;
            var flat = new float[span * outputs.Count];
            for (int i = 0; i < outputs.Count; i++)
                Array.Copy(outputs[i], 0, flat, i * span, span);
            return flat;
        }

        // ---- merger ---------------------------------------------------------

        /// <summary>
        /// One merger: optionally normalize, fold each mergeSize^2 spatial block into
        /// a single dim*4 token, optionally normalize at that width, then two linears
        /// with an erf-GELU between them. Never mutates <paramref name="x"/>.
        /// </summary>
        private float[] RunMerger(MergerW m, Tensor x, int merged)
        {
            // Both branches allocate, so the fold below always reshapes a private
            // copy rather than the live trunk state a DeepStack tap is reading.
            Tensor src = m.NormBeforeMerge
                ? LayerNormRows(x, m.NormW, m.NormB)
                : Ops.NewContiguous(x);
            try
            {
                // Tokens arrive in merge-block order (block-row, block-col, inner-row,
                // inner-col), so folding four of them is a pure reshape of contiguous
                // memory -- not a transpose and not a gather.
                using Tensor folded = src.View(merged, _mergeDim);
                Tensor afterNorm = m.NormBeforeMerge ? null : LayerNormRows(folded, m.NormW, m.NormB);
                try
                {
                    using Tensor a = m.Fc1.Forward(afterNorm ?? folded);
                    GeluErf(a);
                    using Tensor o = m.Fc2.Forward(a);
                    return DirectOps.ToArray(o);
                }
                finally { afterNorm?.Dispose(); }
            }
            finally { src.Dispose(); }
        }

        // ---- primitives -----------------------------------------------------

        /// <summary>Pull a contiguous column range out of the fused QKV projection.</summary>
        private static Tensor Cols(Tensor src, int from, int count)
        {
            using var view = src.Narrow(1, from, count);
            return Ops.NewContiguous(view);
        }

        /// <summary>
        /// Mean-centred LayerNorm (ggml_norm) with weight and bias, parallel over rows.
        ///
        /// Ops.LayerNorm computes the same value, but TensorApplyCPU walks its rows
        /// SERIALLY with SIMD only inside a row; this tower runs two norms per block
        /// plus one per merger, so at a few thousand tokens that is a single-core
        /// pass wedged between fully parallel GEMMs. The accumulation order below is
        /// deliberately identical to TensorApplyCPU.LayerNorm's, tail split included,
        /// so the two agree bit for bit.
        /// </summary>
        private unsafe Tensor LayerNormRows(Tensor x, Tensor gamma, Tensor beta)
        {
            long rows = x.Sizes[0], cols = x.Sizes[1];
            float eps = _eps;
            Tensor outT = _ctx.NewF32(rows, cols);
            float* pin = (float*)CpuNativeHelpers.GetBufferStart(x);
            float* pout = (float*)CpuNativeHelpers.GetBufferStart(outT);
            float* pg = (float*)CpuNativeHelpers.GetBufferStart(gamma);
            float* pb = beta != null ? (float*)CpuNativeHelpers.GetBufferStart(beta) : null;
            int vw = Vector<float>.Count;
            int n = checked((int)cols);
            DirectOps.RowsParallel(rows, r =>
            {
                float* sp = pin + r * cols;
                float* so = pout + r * cols;
                var span = new ReadOnlySpan<float>(sp, n);

                var acc = Vector<float>.Zero;
                int i = 0;
                for (; i < n - vw; i += vw) acc += new Vector<float>(span.Slice(i));
                float sum = Vector.Dot(acc, Vector<float>.One);
                for (; i < n; i++) sum += sp[i];
                float mean = sum / n;

                var vMean = new Vector<float>(mean);
                float sqSum = 0f;
                for (i = 0; i < n - vw; i += vw)
                {
                    var d = new Vector<float>(span.Slice(i)) - vMean;
                    sqSum += Vector.Dot(d, d);
                }
                for (; i < n; i++) { float d = sp[i] - mean; sqSum += d * d; }

                // eps is INSIDE the sqrt, which is what ggml_norm does too.
                float sigma = MathF.Sqrt(eps + sqSum / n);
                var vSigma = new Vector<float>(sigma);
                for (i = 0; i < n - vw; i += vw)
                {
                    var t = new Vector<float>(new ReadOnlySpan<float>(pg + i, vw))
                          * ((new Vector<float>(span.Slice(i)) - vMean) / vSigma);
                    if (pb != null) t += new Vector<float>(new ReadOnlySpan<float>(pb + i, vw));
                    t.CopyTo(new Span<float>(so + i, vw));
                }
                for (; i < n; i++)
                {
                    float t = pg[i] * ((sp[i] - mean) / sigma);
                    so[i] = pb != null ? t + pb[i] : t;
                }
            });
            return outT;
        }

        /// <summary>
        /// In-place rotate-half RoPE over x [seq, heads*hd] with per-token tables
        /// [seq, hd]. The whole head rotates (rot == hd), so the native's tail-concat
        /// branch is dead. BuildRope duplicates the halves (cos[j] == cos[j + hd/2]),
        /// but the +half index is written out so this stays correct if it ever stops.
        /// </summary>
        private static unsafe void RopeHalf(Tensor x, float[] cos, float[] sin,
                                            int seq, int heads, int hd)
        {
            int half = hd / 2;
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            fixed (float* pcos = cos, psin = sin)
            {
                float* pc = pcos, ps = psin;
                CpuWorkerPool.Shared.For(seq, t =>
                {
                    float* ct = pc + (long)t * hd;
                    float* st = ps + (long)t * hd;
                    float* row = px + (long)t * heads * hd;
                    for (int h = 0; h < heads; h++)
                    {
                        float* v = row + (long)h * hd;
                        for (int i = 0; i < half; i++)
                        {
                            float a = v[i], b = v[i + half];
                            v[i] = a * ct[i] - b * st[i];
                            v[i + half] = b * ct[i + half] + a * st[i + half];
                        }
                    }
                });
            }
        }

        /// <summary>
        /// ggml_gelu_erf, in place: 0.5 * x * (1 + erf(x / sqrt(2))). This is NOT
        /// Ops.GELU, which is the tanh approximation the block MLP wants.
        /// </summary>
        private static unsafe void GeluErf(Tensor t)
        {
            const float InvSqrt2 = 0.70710678118654752440f;
            long rows = t.Sizes[0], cols = t.Sizes[1];
            float* p = (float*)CpuNativeHelpers.GetBufferStart(t);
            DirectOps.RowsParallel(rows, r =>
            {
                float* row = p + r * cols;
                for (long i = 0; i < cols; i++)
                    row[i] = 0.5f * row[i] * (1.0f + Erf(row[i] * InvSqrt2));
            });
        }

        /// <summary>
        /// Abramowitz and Stegun 7.1.26 in double precision: maximum absolute error
        /// 1.5e-7, i.e. at the resolution of the float32 result it feeds. Used
        /// because .NET has no Math.Erf and the exact series and continued-fraction
        /// forms are far too slow for the evaluations a full-size reference costs.
        /// </summary>
        private static float Erf(float value)
        {
            double x = value;
            double sign = x < 0.0 ? -1.0 : 1.0;
            x = Math.Abs(x);
            if (x > 6.0)
                return (float)sign;   // |erf| is 1 to well within float precision

            double t = 1.0 / (1.0 + 0.3275911 * x);
            double poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                            - 0.284496736) * t + 0.254829592) * t;
            double y = 1.0 - poly * Math.Exp(-x * x);
            return (float)(sign * y);
        }

        // ---- parity probe ----------------------------------------------------

        /// <summary>TS_H3_VIS_STOP=N truncates the tower after stage N (0 = the patch
        /// embedding, k = after block k-1) so a NaN localizes to one stage instead of
        /// "the tower is wrong"; anything else, the default included, runs it all.</summary>
        private static int VisStopStage()
            => int.TryParse(Environment.GetEnvironmentVariable("TS_H3_VIS_STOP"), out int v) ? v : -1;

        /// <summary>
        /// Write the truncated stage's shape/NaN/RMS line in the native probe's
        /// format so a per-stage bisect can diff the two backends, then ALWAYS throw.
        /// The native returns success here without writing its output, which hands
        /// the caller a silent all-zero embedding; refusing to return at all is the
        /// managed equivalent that cannot be mistaken for a parity failure.
        /// </summary>
        private static void ReportProbe(int stage, Tensor h)
        {
            float[] data = DirectOps.ToArray(h);
            long nans = 0;
            double acc = 0;
            foreach (float v in data)
            {
                if (float.IsNaN(v) || float.IsInfinity(v)) nans++;
                else acc += (double)v * v;
            }
            // ne order, so the tower width prints first -- as it does in the native.
            string line = $"[h3-vis-probe] stage {stage}: {h.Sizes[1]} x {h.Sizes[0]}, " +
                          $"nan/inf={nans} of {data.LongLength} " +
                          $"rms={Math.Sqrt(acc / Math.Max(1L, data.LongLength - nans)):F6}";
            Console.Error.WriteLine(line);
            string path = Environment.GetEnvironmentVariable("TS_H3_VIS_DUMP");
            if (!string.IsNullOrEmpty(path))
                System.IO.File.AppendAllText(path, line + Environment.NewLine);
            throw new InvalidOperationException(
                $"TS_H3_VIS_STOP={stage} truncated the vision tower; no embedding was produced.");
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var l in _owned) l.Dispose();
            _owned.Clear();
            _ctx.Dispose();
        }
    }
}
