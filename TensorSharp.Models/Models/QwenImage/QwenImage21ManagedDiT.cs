// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage;

/// <summary>
/// The Qwen-Image-2.1 transformer in pure C# (the <c>cpu</c> backend): the graph of
/// ggml_ops_qwen_image21.cpp computed op by op with the same formulas, in the same order,
/// from the same file-mapped weights and descriptors (<see cref="QwenImage21ForwardArgs"/>,
/// <see cref="QwenImage21Block"/>, <see cref="QwenImage21Adapter"/>). Nothing here calls
/// native code: everything runs in <see cref="QwenImage21CpuKernels"/> (or, with
/// TS_QWEN21_CPU_MATMUL=q8, the projections in <see cref="ManagedQuantizedOps"/>).
/// </summary>
/// <remarks>
/// <para><b>Layout.</b> Activations are token-major [rows, dim] with head h at columns
/// h * headDim, i.e. ggml's [dim, rows] tensors. For attention K is transposed per head and
/// V copied per head, so every tile the kernels stream is contiguous.</para>
/// <para><b>Numerics.</b> The projections keep the activations in F32 against dequantized
/// weights, where ggml-cpu quantizes them to Q8_K per projection; see <see cref="F32Matmul"/>
/// for why that is the default here, and why no second implementation can match ggml-cpu's
/// quantized output closer than ggml-cpu matches itself under a 1e-6 input change.</para>
/// <para><b>Prefix cache.</b> Text and reference tokens are modulated at t=0 and never attend
/// to the target, so their per-layer K/V are step independent. An "extract" forward stores
/// them in a <see cref="QwenImage21ManagedPrefix"/>; a "cached" forward then computes only
/// the target rows and loads the prefix rows of K/V back. Every per-row operation (and the
/// attention of a target row, whose key tiles start at the same positions) is independent of
/// how many other rows run, so an F32 cache reproduces the uncached prediction exactly.</para>
/// </remarks>
internal sealed unsafe class QwenImage21ManagedDiT : IDisposable
{
    private enum Slot { Joint, Hidden, Query, Key, Value, Output, KeysT, GateUp, ShrinkGroup, ShrinkOwn,
        TimeA, TimeB, Modulation, NormOut, Ada, TextA, TextB, TextC, Head, Rounded, ValuesH, Count }

    private readonly QwenImage21ForwardArgs _w;
    private readonly QwenImage21Block[] _blocks;
    private readonly int _dim, _heads, _headDim, _channels, _textDim, _layers, _ff;
    private readonly float _eps;
    private readonly object _gate = new();
    private readonly IntPtr[] _scratch = new IntPtr[(int)Slot.Count];
    private readonly long[] _scratchBytes = new long[(int)Slot.Count];
    private AdapterF32 _adapter;

    // ggml-cpu's GELU reads an F16 table; the default is the tanh formula in F32 (as CUDA
    // and Metal compute it). TS_QWEN21_CPU_GELU_FP16=1 reproduces ggml-cpu for A/B parity.
    private static readonly bool GeluFp16 = Environment.GetEnvironmentVariable("TS_QWEN21_CPU_GELU_FP16") == "1";
    // The MLP runs in row chunks so its [rows, 2 * ff] activation stays bounded (400 MB at
    // 4096 rows unchunked); a chunk still gives the matmul hundreds of rows to amortize.
    private static readonly int MlpRows = EnvInt("TS_QWEN21_CPU_MLP_ROWS", 1024);
    // ggml-cpu multiplies an F16/BF16 weight by activations converted to that type (its
    // vec_dot_type), so img_in/txt_in see BF16-rounded inputs there. The default keeps them
    // in F32; TS_QWEN21_CPU_ROUND_ACTIVATIONS=1 reproduces the rounding for A/B parity.
    private static readonly bool RoundActivations = Environment.GetEnvironmentVariable("TS_QWEN21_CPU_ROUND_ACTIVATIONS") == "1";
    /// <summary>
    /// Projections multiply F32 activations by dequantized weight tiles
    /// (<see cref="QwenImage21CpuKernels.GemmDequantNT"/>) instead of quantizing the activations
    /// to Q8_K/Q8_0 for an integer dot as ggml and <see cref="ManagedQuantizedOps"/> do.
    /// Measured on this transformer (i7-11800H, 279 rows): 300-500 GFLOPS against 66-106 for
    /// the managed integer dot, and numerically stable where the integer dot is not: a 1e-6
    /// relative change of the input latents moved the Q8_K pipeline's velocity by 2.5e-2
    /// relative L2 (every activation re-quantization can flip a rounding, 224 times per
    /// forward) but the F32 one by 1e-5. TS_QWEN21_CPU_MATMUL=q8 selects the integer dot.
    /// </summary>
    internal static bool F32Matmul { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_QWEN21_CPU_MATMUL")?.Trim(), "q8", StringComparison.OrdinalIgnoreCase);
    // Attention reads each head's values from a head-major copy (GatherValues);
    // TS_QWEN21_CPU_GATHER_V=0 reads the token-major V in place.
    internal static bool GatherV { get; set; } = Environment.GetEnvironmentVariable("TS_QWEN21_CPU_GATHER_V") != "0";
    // TS_QWEN21_CPU_PROFILE=1 prints one line per forward with the time of each stage.
    private static readonly bool Profile = Environment.GetEnvironmentVariable("TS_QWEN21_CPU_PROFILE") == "1";

    private static int EnvInt(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;

    internal QwenImage21ManagedDiT(in QwenImage21ForwardArgs weights, QwenImage21Block[] blocks)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        _w = weights;
        _blocks = blocks;
        _dim = weights.Dim; _heads = weights.Heads; _headDim = weights.HeadDim;
        _channels = weights.Channels; _textDim = weights.TextDim; _layers = weights.NumLayers; _eps = weights.Eps;
        if (_dim <= 0 || _headDim <= 0 || _headDim % 2 != 0 || _dim != _heads * _headDim || _channels <= 0 ||
            _textDim <= 0 || _layers <= 0 || blocks.Length != _layers || !float.IsFinite(_eps) || _eps <= 0 || _headDim > 512)
            throw new ArgumentException("Invalid Qwen-Image-2.1 transformer shape.");
        _ff = checked((int)blocks[0].Down.Ne0);
        Check(weights.ImageIn, _channels, _dim, "img_in");
        Check(weights.TextIn, _textDim, _dim, "txt_in.in_layer");
        Check(weights.TextOut, _dim, _dim, "txt_in.out_layer");
        Check(weights.TimeIn, 256, _dim, "timestep_embedder.linear_1");
        Check(weights.TimeOut, _dim, _dim, "timestep_embedder.linear_2");
        Check(weights.Modulation, _dim, 4L * _dim, "modulation.1");
        Check(weights.NormOut, _dim, _dim, "norm_out.linear");
        Check(weights.ProjOut, _dim, _channels, "proj_out");
        if (weights.TextNorm == IntPtr.Zero) throw new ArgumentException("Qwen-Image-2.1 needs txt_in.text_norm.");
        foreach (var b in blocks)
        {
            Check(b.Q, _dim, _dim, "attn.to_q"); Check(b.K, _dim, _dim, "attn.to_k");
            Check(b.V, _dim, _dim, "attn.to_v"); Check(b.Out, _dim, _dim, "attn.to_out.0");
            Check(b.Gate, _dim, b.Up.Data == IntPtr.Zero ? 2L * _ff : _ff, "img_mlp gate");
            if (b.Up.Data != IntPtr.Zero) Check(b.Up, _dim, _ff, "img_mlp up");
            Check(b.Down, _ff, _dim, "img_mlp.out");
            if (b.NormQ == IntPtr.Zero || b.NormK == IntPtr.Zero) throw new ArgumentException("Qwen-Image-2.1 needs attn.norm_q/norm_k.");
        }
    }

    private static void Check(in QwenImage21Weight w, long input, long output, string name)
    {
        if (w.Data == IntPtr.Zero || w.Ne0 != input || w.Ne1 != output || w.Ne0 > int.MaxValue || w.Ne1 > int.MaxValue)
            throw new ArgumentException($"Qwen-Image-2.1 {name} must be [{input},{output}].");
        if (!ManagedQuantizedOps.SupportsDequantization((GgmlTensorType)w.Type))
            throw new NotSupportedException($"The pure-C# cpu backend cannot run Qwen-Image-2.1 {name} stored as {(GgmlTensorType)w.Type}.");
    }

    /// <summary>
    /// One velocity prediction. <paramref name="images"/> holds the reference latents followed
    /// by the target's, token-major; <paramref name="time"/> the two sinusoidal rows (target
    /// timestep, t=0); <paramref name="cos"/>/<paramref name="sin"/> [totalSeq, headDim/2].
    /// </summary>
    internal QwenImage21ForwardPath Forward(float[] images, int imageSeq, float[] text, int textSeq, float[] time,
        float[] cos, float[] sin, QwenImage21Segment[] segments, int prefixSeq, int totalSeq, IntPtr adapter,
        QwenImage21ManagedPrefix cache, float[] output)
    {
        Validate(images, imageSeq, text, textSeq, time, cos, sin, segments, prefixSeq, totalSeq, output);
        lock (_gate)
        {
            var lora = AdapterFor(adapter);
            var path = QwenImage21ForwardPath.Full;
            if (cache != null && prefixSeq > 0)
            {
                cache.Bind(this, adapter, segments, prefixSeq, totalSeq, imageSeq, textSeq, _layers, _dim);
                path = cache.Declined ? QwenImage21ForwardPath.Declined
                    : cache.Filled ? QwenImage21ForwardPath.Cached : QwenImage21ForwardPath.Extract;
            }
            var stages = Profile ? new Stages() : null;
            fixed (float* imgs = images, txt = text, tm = time, cs = cos, sn = sin, outp = output)
                Run(imgs, txt, textSeq, tm, cs, sn, segments, prefixSeq, totalSeq, lora,
                    path is QwenImage21ForwardPath.Extract or QwenImage21ForwardPath.Cached ? cache : null, path, outp, stages);
            if (path == QwenImage21ForwardPath.Extract) cache.Filled = true;
            stages?.Report(path, totalSeq - (path == QwenImage21ForwardPath.Cached ? prefixSeq : 0));
            return path;
        }
    }

    private void Validate(float[] images, int imageSeq, float[] text, int textSeq, float[] time, float[] cos, float[] sin,
        QwenImage21Segment[] segments, int prefixSeq, int totalSeq, float[] output)
    {
        ArgumentNullException.ThrowIfNull(segments);
        if (images == null || imageSeq <= 0 || images.Length < (long)imageSeq * _channels ||
            text == null || textSeq <= 0 || text.Length < (long)textSeq * _textDim ||
            time == null || time.Length < 512 || prefixSeq < 0 || totalSeq <= prefixSeq || segments.Length == 0 ||
            cos == null || sin == null || cos.Length < (long)totalSeq * (_headDim / 2) || sin.Length < (long)totalSeq * (_headDim / 2))
            throw new ArgumentException("QwenImage21: invalid forward inputs");
        int end = 0;
        foreach (var s in segments)
        {
            if (s.Start != end || s.End <= s.Start || s.End > totalSeq || s.SourceStart < 0 ||
                (long)s.SourceStart + s.End - s.Start > (s.IsImage != 0 ? imageSeq : textSeq))
                throw new ArgumentException("QwenImage21: invalid sequence segment");
            end = s.End;
        }
        var target = segments[^1];
        if (end != totalSeq || target.Start != prefixSeq || target.IsImage == 0)
            throw new ArgumentException("QwenImage21: missing target image segment");
        if (output == null || output.Length < (long)(target.End - target.Start) * _channels)
            throw new ArgumentException("QwenImage21: output is too small");
    }

    private void Run(float* images, float* text, int textSeq, float* time, float* cos, float* sin,
        QwenImage21Segment[] segments, int prefixSeq, int totalSeq, AdapterF32 lora, QwenImage21ManagedPrefix cache,
        QwenImage21ForwardPath path, float* output, Stages stages)
    {
        bool cached = path == QwenImage21ForwardPath.Cached;
        int dim = _dim, first = cached ? prefixSeq : 0, rows = totalSeq - first, prefixRows = cached ? 0 : prefixSeq;
        var target = segments[^1];

        // ---- time embedding and the shared AdaLN modulation (row 0: target t, row 1: t=0) ----
        float* t1 = Get(Slot.TimeA, 2L * dim), t2 = Get(Slot.TimeB, 2L * dim);
        float* mod = Get(Slot.Modulation, 8L * dim), normOut = Get(Slot.NormOut, dim);
        Project(_w.TimeIn, time, 256, 2, t1, dim, lora?.TimeIn);
        for (int i = 0; i < 2 * dim; i++) t1[i] = QwenImage21CpuKernels.Silu(t1[i]);
        Project(_w.TimeOut, t1, dim, 2, t2, dim, lora?.TimeOut);
        for (int i = 0; i < 2 * dim; i++) t2[i] = QwenImage21CpuKernels.Silu(t2[i]);
        Project(_w.Modulation, t2, dim, 2, mod, 4 * dim, lora?.Modulation);
        Project(_w.NormOut, t2, dim, 1, normOut, dim, lora?.NormOut);
        // ada[(chunk * 2 + row) * dim]: chunks 0/2 are (1 + scale), 1/3 are tanh(gate).
        float* ada = Get(Slot.Ada, 8L * dim);
        for (int row = 0; row < 2; row++)
            for (int chunk = 0; chunk < 4; chunk++)
            {
                float* src = mod + (long)row * 4 * dim + (long)chunk * dim, dst = ada + (long)(chunk * 2 + row) * dim;
                for (int i = 0; i < dim; i++) dst[i] = chunk % 2 == 0 ? src[i] + 1f : MathF.Tanh(src[i]);
            }
        for (int i = 0; i < dim; i++) normOut[i] += 1f;
        float* scale0T = ada, scale0P = ada + dim, gate1T = ada + 2L * dim, gate1P = ada + 3L * dim;
        float* scale2T = ada + 4L * dim, scale2P = ada + 5L * dim, gate3T = ada + 6L * dim, gate3P = ada + 7L * dim;

        // ---- the joint sequence ----
        float* joint = Get(Slot.Joint, (long)rows * dim);
        if (cached)
        {
            Project(_w.ImageIn, images + (long)target.SourceStart * _channels, _channels, target.End - target.Start,
                joint, dim, lora?.ImageIn);
        }
        else
        {
            float* normalized = Get(Slot.TextA, (long)textSeq * _textDim);
            float* hidden = Get(Slot.TextB, (long)textSeq * dim), embedded = Get(Slot.TextC, (long)textSeq * dim);
            float* gain = Get(Slot.Head, _textDim);
            for (int i = 0; i < _textDim; i++) gain[i] = ((float*)_w.TextNorm)[i] + 1f;
            QwenImage21CpuKernels.RmsNormGain(text, normalized, textSeq, _textDim, _eps, gain);
            Project(_w.TextIn, normalized, _textDim, textSeq, hidden, dim, lora?.TextIn);
            bool fp16 = GeluFp16;
            nint ha = (nint)hidden;
            long count = (long)textSeq * dim;
            int per = 1 << 16;
            QwenImage21CpuKernels.For((int)((count + per - 1) / per), b =>
            {
                float* x = (float*)ha;
                long end = Math.Min(count, (long)(b + 1) * per);
                for (long i = (long)b * per; i < end; i++) x[i] = QwenImage21CpuKernels.Gelu(x[i], fp16);
            });
            Project(_w.TextOut, hidden, dim, textSeq, embedded, dim, lora?.TextOut);
            foreach (var s in segments)
            {
                if (s.IsImage != 0)
                    Project(_w.ImageIn, images + (long)s.SourceStart * _channels, _channels, s.End - s.Start,
                        joint + (long)s.Start * dim, dim, lora?.ImageIn);
                else
                    Buffer.MemoryCopy(embedded + (long)s.SourceStart * dim, joint + (long)s.Start * dim,
                        (long)(s.End - s.Start) * dim * sizeof(float), (long)(s.End - s.Start) * dim * sizeof(float));
            }
        }
        stages?.Lap(ref stages.Embed);

        // ---- blocks ----
        float* h = Get(Slot.Hidden, (long)rows * dim), q = Get(Slot.Query, (long)rows * dim);
        float* k = Get(Slot.Key, (long)totalSeq * dim), v = Get(Slot.Value, (long)totalSeq * dim);
        float* o = Get(Slot.Output, (long)rows * dim);
        float* kt = Get(Slot.KeysT, (long)_heads * _headDim * QwenImage21CpuKernels.KeyStride(totalSeq));
        float* vh = GatherV ? Get(Slot.ValuesH, (long)totalSeq * dim) : null;
        float* kRows = k + (long)first * dim, vRows = v + (long)first * dim;
        // Equal MLP chunks: 1047 rows run as 2 x 524, not 1024 + a 23-row chunk that would
        // dequantize both MLP weights again for a handful of rows.
        int half = _headDim / 2, chunkRows = (rows + (rows + MlpRows - 1) / MlpRows - 1) / ((rows + MlpRows - 1) / MlpRows);
        float* gu = Get(Slot.GateUp, (long)chunkRows * 2 * _ff);
        float scale = 1f / MathF.Sqrt(_headDim);
        for (int layer = 0; layer < _layers; layer++)
        {
            var w = _blocks[layer];
            var l = lora?.Blocks[layer];
            QwenImage21CpuKernels.LayerNormScale(joint, h, rows, dim, _eps, scale0T, scale0P, prefixRows);
            stages?.Lap(ref stages.Norm);
            // Q, K and V read one input: a single stacked LoRA shrink serves all three.
            float* qkv = Shrink(l?.Qkv, h, dim, rows, Slot.ShrinkGroup);
            stages?.Lap(ref stages.Lora);
            Project(w.Q, h, dim, rows, q, dim, l?.Q, qkv, stages);
            Project(w.K, h, dim, rows, kRows, dim, l?.K, qkv, stages);
            Project(w.V, h, dim, rows, vRows, dim, l?.V, qkv, stages);
            QwenImage21CpuKernels.RmsNormRope(q, rows, dim, _heads, _headDim, (float*)w.NormQ, cos + (long)first * half, sin + (long)first * half, _eps);
            QwenImage21CpuKernels.RmsNormRope(kRows, rows, dim, _heads, _headDim, (float*)w.NormK, cos + (long)first * half, sin + (long)first * half, _eps);
            if (cache != null && !cached) cache.Store(layer, k, v, dim);
            else if (cache != null) cache.Load(layer, k, v, dim);
            QwenImage21CpuKernels.TransposeKeys(k, dim, totalSeq, _heads, _headDim, kt);
            if (vh != null) QwenImage21CpuKernels.GatherValues(v, dim, totalSeq, _heads, _headDim, vh);
            stages?.Lap(ref stages.Norm);
            QwenImage21CpuKernels.Attention(q, kt, totalSeq, vh != null ? vh : v, vh != null ? _headDim : dim,
                vh != null ? (long)totalSeq * _headDim : _headDim, o, dim, _heads, _headDim, segments,
                cached ? segments.Length - 1 : 0, first, scale);
            stages?.Lap(ref stages.Attention);
            // The attention output projection reuses Q's rows.
            Project(w.Out, o, dim, rows, q, dim, l?.Out, Shrink(l?.OutGroup, o, dim, rows, Slot.ShrinkGroup), stages);
            QwenImage21CpuKernels.GatedAdd(joint, q, dim, rows, dim, gate1T, gate1P, prefixRows);
            QwenImage21CpuKernels.LayerNormScale(joint, h, rows, dim, _eps, scale2T, scale2P, prefixRows);
            stages?.Lap(ref stages.Norm);
            for (int c0 = 0; c0 < rows; c0 += chunkRows)
            {
                int cn = Math.Min(chunkRows, rows - c0);
                float* x = h + (long)c0 * dim;
                if (w.Up.Data == IntPtr.Zero)
                {
                    // The checkpoint's fused [gate | up]; a LoRA on it spans both halves.
                    Project(w.Gate, x, dim, cn, gu, 2 * _ff, l?.Gate, Shrink(l?.GateUpGroup, x, dim, cn, Slot.ShrinkGroup), stages);
                }
                else
                {
                    float* shrunk = Shrink(l?.GateUpGroup, x, dim, cn, Slot.ShrinkGroup);
                    Project(w.Gate, x, dim, cn, gu, 2 * _ff, l?.Gate, shrunk, stages);
                    Project(w.Up, x, dim, cn, gu + _ff, 2 * _ff, l?.Up, shrunk, stages);
                }
                QwenImage21CpuKernels.SwiGlu(gu, 2 * _ff, cn, _ff);
                stages?.Lap(ref stages.Norm);
                Project(w.Down, gu, 2 * _ff, cn, q + (long)c0 * dim, dim, l?.Down,
                    Shrink(l?.DownGroup, gu, 2 * _ff, cn, Slot.ShrinkGroup), stages);
            }
            QwenImage21CpuKernels.GatedAdd(joint, q, dim, rows, dim, gate3T, gate3P, prefixRows);
            stages?.Lap(ref stages.Norm);
        }

        // ---- output: LayerNorm * (1 + norm_out(t)) on the target rows, then proj_out or a step head ----
        int targetRows = rows - prefixRows;
        QwenImage21CpuKernels.LayerNormScale(joint + (long)prefixRows * dim, h, targetRows, dim, _eps, normOut, normOut, 0);
        if (lora is { Source: var source } && ((QwenImage21Adapter*)source)->OutputHead != IntPtr.Zero)
        {
            var a = (QwenImage21Adapter*)source;
            float* head = (float*)a->OutputHead;
            if (a->OutputHeadType == 1)
            {
                head = Get(Slot.Head, (long)_channels * dim);
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<System.Half>((void*)a->OutputHead, _channels * dim), new Span<float>(head, _channels * dim));
            }
            QwenImage21CpuKernels.GemmNT(h, dim, head, dim, output, _channels, targetRows, _channels, dim);
        }
        else
            Project(_w.ProjOut, h, dim, targetRows, output, _channels, lora?.ProjOut);
        stages?.Lap(ref stages.Embed);
    }

    // ---- projections -----------------------------------------------------------------

    /// <summary>y = W x (+ a LoRA update: row scale on the base, then up * shrunk). Weights with
    /// an integer-dot plan use <see cref="ManagedQuantizedOps"/>; the rest (F32/F16/BF16, quant
    /// types without a plan) dequantize tiles in <see cref="QwenImage21CpuKernels.GemmDequantNT"/>.</summary>
    private void Project(in QwenImage21Weight w, float* x, int xStride, int rows, float* y, int yStride,
        Update l, float* shrunk = null, Stages stages = null)
    {
        if (rows <= 0) return;
        stages?.Lap(ref stages.Norm);
        int input = (int)w.Ne0, outputs = (int)w.Ne1;
        var type = (GgmlTensorType)w.Type;
        if (!F32Matmul && type is not (GgmlTensorType.F32 or GgmlTensorType.F16 or GgmlTensorType.BF16) &&
            ManagedQuantizedOps.TryGetActivationPlan(type, input, out _))
            ManagedQuantizedOps.AddmmQuantizedToFloat32(w.Type, w.Data, w.Ne0, w.Ne1, x, xStride, rows, y, yStride);
        else
        {
            float* source = x;
            int sourceStride = xStride;
            if (RoundActivations && type is GgmlTensorType.F16 or GgmlTensorType.BF16)
            {
                source = Get(Slot.Rounded, (long)rows * input);
                sourceStride = input;
                for (int r = 0; r < rows; r++)
                    for (int i = 0; i < input; i++)
                    {
                        float v = x[(long)r * xStride + i];
                        source[(long)r * input + i] = type == GgmlTensorType.BF16 ? QwenImage21DiT.RoundBf16(v) : (float)(System.Half)v;
                    }
            }
            QwenImage21CpuKernels.GemmDequantNT(source, sourceStride, rows, w.Type, (byte*)w.Data, w.Bytes / w.Ne1, input, outputs, y, yStride);
        }
        stages?.Lap(ref stages.Matmul);
        if (l == null) return;
        if (l.RowScale != null)
        {
            nint ya = (nint)y, ra = (nint)l.RowScale;
            QwenImage21CpuKernels.For(rows, r =>
            {
                var row = new Span<float>((float*)ya + (long)r * yStride, outputs);
                TensorPrimitives.Multiply(row, new ReadOnlySpan<float>((float*)ra, outputs), row);
            });
        }
        if (l.Rank > 0)
        {
            int stride;
            if (shrunk == null)
            {
                // A projection outside the block groups (the global adapters).
                shrunk = Get(Slot.ShrinkOwn, (long)rows * l.Rank);
                QwenImage21CpuKernels.GemmNT(x, xStride, l.Down, input, shrunk, l.Rank, rows, l.Rank, input);
                stride = l.Rank;
            }
            else
            {
                stride = l.GroupRank;
                shrunk += l.Offset;
            }
            QwenImage21CpuKernels.GemmNNAccumulate(shrunk, stride, l.UpT, y, yStride, rows, outputs, l.Rank);
        }
        stages?.Lap(ref stages.Lora);
    }

    /// <summary>The stacked down projections of a group over <paramref name="rows"/> inputs,
    /// [rows, group rank], or null without a low-rank term.</summary>
    private float* Shrink(Group group, float* x, int xStride, int rows, Slot slot)
    {
        if (group == null || group.Rank == 0 || rows <= 0) return null;
        float* result = Get(slot, (long)rows * group.Rank);
        QwenImage21CpuKernels.GemmNT(x, xStride, group.Down, group.In, result, group.Rank, rows, group.Rank, group.In);
        return result;
    }

    private float* Get(Slot slot, long floats)
    {
        int i = (int)slot;
        long bytes = Math.Max(1, floats) * sizeof(float);
        if (_scratchBytes[i] < bytes)
        {
            if (_scratch[i] != IntPtr.Zero) NativeMemory.AlignedFree((void*)_scratch[i]);
            _scratch[i] = IntPtr.Zero;
            _scratchBytes[i] = 0;
            void* p = NativeMemory.AlignedAlloc((nuint)bytes, 64);
            if (p == null) throw new OutOfMemoryException($"Qwen-Image-2.1 cpu scratch of {bytes} bytes.");
            _scratch[i] = (IntPtr)p;
            _scratchBytes[i] = bytes;
        }
        return (float*)_scratch[i];
    }

    /// <summary>Frees the activation scratch (kept between denoising steps).</summary>
    internal void ReleaseScratch()
    {
        lock (_gate)
            for (int i = 0; i < _scratch.Length; i++)
            {
                if (_scratch[i] != IntPtr.Zero) NativeMemory.AlignedFree((void*)_scratch[i]);
                _scratch[i] = IntPtr.Zero;
                _scratchBytes[i] = 0;
            }
    }

    public void Dispose()
    {
        ReleaseScratch();
        lock (_gate)
        {
            _adapter?.Dispose();
            _adapter = null;
        }
    }

    // ---- LoRA factors in F32 ----------------------------------------------------------

    /// <summary>One projection's update: y = RowScale * (W x) + UpT^T (Down x).</summary>
    private sealed class Update
    {
        internal float* RowScale;   // [out], or null
        internal float* UpT;        // [rank, out] in 64-column panels: up transposed, so the expand is a row of FMAs per rank step
        internal float* Down;       // [rank, in]: this update's rows of its group's stack
        internal int Rank, Offset, GroupRank;
    }

    /// <summary>Updates that read one input; their down factors are stacked so the input is
    /// read once (the native graph's stacked shrink). Updates that share one factor (the two
    /// halves of a fused gate_up LoRA) share its columns.</summary>
    private sealed class Group
    {
        internal float* Down;       // [rank, in]
        internal int Rank, In;
    }

    private sealed class BlockUpdates
    {
        internal Update Q, K, V, Out, Gate, Up, Down;
        internal Group Qkv, OutGroup, GateUpGroup, DownGroup;
    }

    /// <summary>A <see cref="QwenImage21Adapter"/>'s factors converted to F32 once.</summary>
    private sealed class AdapterF32 : IDisposable
    {
        internal IntPtr Source;
        internal Update ImageIn, TextIn, TextOut, TimeIn, TimeOut, Modulation, NormOut, ProjOut;
        internal BlockUpdates[] Blocks;
        private readonly List<IntPtr> _allocations = new();

        internal float* Allocate(long floats)
        {
            void* p = NativeMemory.AlignedAlloc((nuint)(Math.Max(1, floats) * sizeof(float)), 64);
            if (p == null) throw new OutOfMemoryException("Qwen-Image-2.1 LoRA factors.");
            _allocations.Add((IntPtr)p);
            return (float*)p;
        }

        public void Dispose()
        {
            foreach (var p in _allocations) NativeMemory.AlignedFree((void*)p);
            _allocations.Clear();
        }
    }

    private AdapterF32 AdapterFor(IntPtr adapter)
    {
        if (adapter == IntPtr.Zero) return null;
        if (_adapter?.Source == adapter) return _adapter;
        _adapter?.Dispose();
        _adapter = null;
        var a = (QwenImage21Adapter*)adapter;
        if (a->StructBytes != sizeof(QwenImage21Adapter) || a->NumLayers != _layers)
            throw new ArgumentException("QwenImage21: invalid LoRA adapter descriptor");
        if (a->OutputHead != IntPtr.Zero && ((a->OutputHeadType != 0 && a->OutputHeadType != 1) || HasUpdate(a->ProjOut)))
            throw new ArgumentException("QwenImage21: an output head replaces proj_out and cannot carry a LoRA");
        var result = new AdapterF32 { Source = adapter };
        try
        {
            Update Global(in QwenImage21Lora l, in QwenImage21Weight w, string name)
            {
                var g = BuildGroup(result, w.Ne0, name, (l, w.Ne1));
                return g.Updates[0];
            }
            result.ImageIn = Global(a->ImageIn, _w.ImageIn, "img_in");
            result.TextIn = Global(a->TextIn, _w.TextIn, "txt_in.in_layer");
            result.TextOut = Global(a->TextOut, _w.TextOut, "txt_in.out_layer");
            result.TimeIn = Global(a->TimeIn, _w.TimeIn, "timestep_embedder.linear_1");
            result.TimeOut = Global(a->TimeOut, _w.TimeOut, "timestep_embedder.linear_2");
            result.Modulation = Global(a->Modulation, _w.Modulation, "modulation.1");
            result.NormOut = Global(a->NormOut, _w.NormOut, "norm_out.linear");
            result.ProjOut = Global(a->ProjOut, _w.ProjOut, "proj_out");
            result.Blocks = new BlockUpdates[_layers];
            var blocks = (QwenImage21BlockLora*)a->Blocks;
            for (int i = 0; i < _layers; i++)
            {
                var u = result.Blocks[i] = new BlockUpdates();
                if (blocks == null) continue;
                var l = blocks[i];
                var w = _blocks[i];
                var qkv = BuildGroup(result, _dim, "attn.to_q/k/v", (l.Q, w.Q.Ne1), (l.K, w.K.Ne1), (l.V, w.V.Ne1));
                (u.Q, u.K, u.V, u.Qkv) = (qkv.Updates[0], qkv.Updates[1], qkv.Updates[2], qkv.Group);
                var output = BuildGroup(result, _dim, "attn.to_out.0", (l.Out, w.Out.Ne1));
                (u.Out, u.OutGroup) = (output.Updates[0], output.Group);
                if (w.Up.Data == IntPtr.Zero && HasUpdate(l.Up))
                    throw new ArgumentException("QwenImage21: a fused gate_up weight takes its LoRA as one gate update");
                var gateUp = w.Up.Data == IntPtr.Zero
                    ? BuildGroup(result, _dim, "img_mlp gate", (l.Gate, w.Gate.Ne1))
                    : BuildGroup(result, _dim, "img_mlp gate/up", (l.Gate, w.Gate.Ne1), (l.Up, w.Up.Ne1));
                u.Gate = gateUp.Updates[0];
                u.Up = gateUp.Updates.Length > 1 ? gateUp.Updates[1] : null;
                u.GateUpGroup = gateUp.Group;
                var down = BuildGroup(result, _ff, "img_mlp.out", (l.Down, w.Down.Ne1));
                (u.Down, u.DownGroup) = (down.Updates[0], down.Group);
            }
        }
        catch
        {
            result.Dispose();
            throw;
        }
        return _adapter = result;
    }

    private static bool HasUpdate(in QwenImage21Lora l) => l.Rank > 0 || l.RowScale != IntPtr.Zero;

    /// <summary>Converts the updates of projections that read one input of width
    /// <paramref name="input"/>: their down factors are stacked (a shared factor once).</summary>
    private static (Update[] Updates, Group Group) BuildGroup(AdapterF32 owner, long input, string name,
        params (QwenImage21Lora Lora, long Out)[] members)
    {
        var updates = new Update[members.Length];
        var distinct = new List<(IntPtr Down, int Rank, int Type, int Offset)>();
        int total = 0;
        for (int i = 0; i < members.Length; i++)
        {
            var (l, outputs) = members[i];
            if (!HasUpdate(l)) continue;
            bool typed = l.Type is 0 or 1;
            if (l.Rank < 0 || l.In != input || l.Out != outputs || (l.Rank > 0 && (!typed || l.Down == IntPtr.Zero || l.Up == IntPtr.Zero)))
                throw new ArgumentException($"QwenImage21: LoRA update does not fit {name}");
            var u = updates[i] = new Update { Rank = l.Rank };
            if (l.RowScale != IntPtr.Zero)
            {
                u.RowScale = owner.Allocate(l.Out);
                new ReadOnlySpan<float>((void*)l.RowScale, (int)l.Out).CopyTo(new Span<float>(u.RowScale, (int)l.Out));
            }
            if (l.Rank == 0) continue;
            int found = distinct.FindIndex(d => d.Down == l.Down && d.Rank == l.Rank && d.Type == l.Type);
            if (found < 0)
            {
                distinct.Add((l.Down, l.Rank, l.Type, total));
                found = distinct.Count - 1;
                total += l.Rank;
            }
            u.Offset = distinct[found].Offset;
            // up is [out, rank]; store it as [rank, out] in 64-column panels (PackPanels' layout).
            const int panel = QwenImage21CpuKernels.PanelWidth;
            long floats = QwenImage21CpuKernels.PanelFloats(l.Rank, (int)l.Out);
            u.UpT = owner.Allocate(floats);
            new Span<float>(u.UpT, checked((int)floats)).Clear();
            var row = new float[l.Rank];
            for (long o = 0; o < l.Out; o++)
            {
                Convert(l.Up + (nint)(o * l.Rank * (l.Type == 0 ? 4 : 2)), l.Type, row, l.Rank);
                float* column = u.UpT + (o / panel) * l.Rank * panel + o % panel;
                for (int r = 0; r < l.Rank; r++) column[(long)r * panel] = row[r];
            }
        }
        if (total == 0) return (updates, null);
        var group = new Group { Rank = total, In = (int)input, Down = owner.Allocate((long)total * input) };
        foreach (var d in distinct)
            Convert(d.Down, d.Type, new Span<float>(group.Down + (long)d.Offset * input, checked((int)(d.Rank * input))));
        foreach (var u in updates)
            if (u is { Rank: > 0 })
            {
                u.GroupRank = total;
                u.Down = group.Down + (long)u.Offset * input;
            }
        return (updates, group);
    }

    private static void Convert(IntPtr source, int type, float[] destination, int count) =>
        Convert(source, type, destination.AsSpan(0, count));

    private static void Convert(IntPtr source, int type, Span<float> destination)
    {
        if (type == 0) new ReadOnlySpan<float>((void*)source, destination.Length).CopyTo(destination);
        else TensorPrimitives.ConvertToSingle(new ReadOnlySpan<System.Half>((void*)source, destination.Length), destination);
    }

    // ---- stage timing (TS_QWEN21_CPU_PROFILE=1) ------------------------------------------

    internal sealed class Stages
    {
        // One timeline: each Lap charges the time since the previous one to a stage.
        internal long Embed, Norm, Matmul, Lora, Attention;
        private long _mark = Stopwatch.GetTimestamp();
        private readonly long _start = Stopwatch.GetTimestamp();

        internal void Lap(ref long counter)
        {
            long now = Stopwatch.GetTimestamp();
            counter += now - _mark;
            _mark = now;
        }

        internal void Report(QwenImage21ForwardPath path, int rows)
        {
            double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"  [qwen21-cpu] {path.ToString().ToLowerInvariant()} rows={rows} total={Ms(Stopwatch.GetTimestamp() - _start):F0}ms " +
                $"matmul={Ms(Matmul):F0} lora={Ms(Lora):F0} attention={Ms(Attention):F0} " +
                $"norm/rope/glu={Ms(Norm):F0} embed/head={Ms(Embed):F0}");
        }
    }
}

/// <summary>
/// The pure-C# transformer's prefix KV cache of one request (see <see cref="QwenImage21DiT.PrefixCache"/>):
/// per layer, the K (after norm and RoPE) and V rows of the text and reference tokens.
/// <see cref="QwenImage21PrefixCacheType.Auto"/> stores what attention reads (F32), so cached
/// steps reproduce uncached ones; F16 and Q8_0 trade that for memory as on the native path.
/// </summary>
internal sealed unsafe class QwenImage21ManagedPrefix : IDisposable
{
    private const int GgmlF32 = 0, GgmlF16 = 1, GgmlQ8_0 = 8;
    private IntPtr[] _k = Array.Empty<IntPtr>(), _v = Array.Empty<IntPtr>();
    private object _owner;
    private IntPtr _adapter;
    private QwenImage21Segment[] _segments;
    private int _total, _imageSeq, _textSeq, _dim;

    internal QwenImage21ManagedPrefix(QwenImage21PrefixCacheType requested) => Requested = requested;

    internal QwenImage21PrefixCacheType Requested { get; }
    internal int KeyType { get; private set; }
    internal int ValueType { get; private set; }
    internal int Tokens { get; private set; }
    internal long Bytes { get; private set; }
    internal bool Filled { get; set; }
    internal bool Declined { get; private set; }

    internal QwenImage21PrefixCacheInfo Info => new()
    {
        State = Declined ? 2 : Filled ? 1 : 0,
        KeyType = KeyType, ValueType = ValueType, Tokens = Tokens, Bytes = Bytes,
    };

    /// <summary>Keeps the stored values when they were computed for this transformer, adapter
    /// and layout; otherwise drops them and allocates (or declines) storage for a new fill.</summary>
    internal void Bind(object owner, IntPtr adapter, QwenImage21Segment[] segments, int prefix, int total,
        int imageSeq, int textSeq, int layers, int dim)
    {
        if (ReferenceEquals(_owner, owner) && _adapter == adapter && Tokens == prefix && _total == total &&
            _imageSeq == imageSeq && _textSeq == textSeq && _dim == dim && _k.Length == (Declined ? 0 : layers) &&
            _segments.AsSpan().SequenceEqual(segments))
            return;
        Release();
        (_owner, _adapter, _segments, Tokens, _total, _imageSeq, _textSeq, _dim) =
            (owner, adapter, (QwenImage21Segment[])segments.Clone(), prefix, total, imageSeq, textSeq, dim);
        (KeyType, ValueType) = Requested switch
        {
            QwenImage21PrefixCacheType.F16 => (GgmlF16, GgmlF16),
            QwenImage21PrefixCacheType.Q8_0 => (GgmlQ8_0, GgmlQ8_0),
            // Attention logits are more sensitive to K than the output is to V.
            QwenImage21PrefixCacheType.Q8_0V => (GgmlF32, GgmlQ8_0),
            _ => (GgmlF32, GgmlF32),
        };
        long elements = (long)prefix * dim;
        long perLayer = SizeOf(KeyType, elements) + SizeOf(ValueType, elements);
        Bytes = perLayer * layers;
        // The native rule: at most half of what is free, and TS_QWEN21_PREFIX_CACHE_MAX_MIB.
        var memory = GC.GetGCMemoryInfo();
        long free = memory.TotalAvailableMemoryBytes > 0 ? Math.Max(0, memory.TotalAvailableMemoryBytes - memory.MemoryLoadBytes) : long.MaxValue;
        if (Bytes > free / 2 || !WithinUserCap(Bytes))
        {
            Declined = true;
            Console.Error.WriteLine($"[qwen21] prefix KV cache declined: {prefix} prefix tokens need {Bytes / (1024.0 * 1024.0):F1} MiB, " +
                $"about {(free == long.MaxValue ? -1 : free) / (1024.0 * 1024.0):F1} MiB is free (the cache may use half; " +
                "TS_QWEN21_PREFIX_CACHE_MAX_MIB caps it further); this request recomputes the prefix every step.");
            return;
        }
        _k = new IntPtr[layers];
        _v = new IntPtr[layers];
        try
        {
            for (int i = 0; i < layers; i++)
            {
                _k[i] = Alloc(SizeOf(KeyType, elements));
                _v[i] = Alloc(SizeOf(ValueType, elements));
            }
        }
        catch
        {
            Release();
            throw;
        }
    }

    private static long SizeOf(int type, long elements) =>
        type switch { GgmlF32 => elements * 4, GgmlF16 => elements * 2, _ => elements / 32 * 34 };

    private static bool WithinUserCap(long bytes)
    {
        string value = Environment.GetEnvironmentVariable("TS_QWEN21_PREFIX_CACHE_MAX_MIB");
        if (string.IsNullOrWhiteSpace(value)) return true;
        return double.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double mib) &&
            double.IsFinite(mib) && mib >= 0 && bytes <= mib * 1024 * 1024;
    }

    private static IntPtr Alloc(long bytes)
    {
        void* p = NativeMemory.AlignedAlloc((nuint)Math.Max(1, bytes), 64);
        if (p == null) throw new OutOfMemoryException($"Qwen-Image-2.1 prefix cache of {bytes} bytes.");
        return (IntPtr)p;
    }

    /// <summary>Stores rows [0, Tokens) of this layer's K and V.</summary>
    internal void Store(int layer, float* k, float* v, int dim)
    {
        Convert(k, _k[layer], KeyType, toStorage: true);
        Convert(v, _v[layer], ValueType, toStorage: true);
    }

    /// <summary>Writes the stored prefix rows of this layer into rows [0, Tokens) of K and V.</summary>
    internal void Load(int layer, float* k, float* v, int dim)
    {
        Convert(k, _k[layer], KeyType, toStorage: false);
        Convert(v, _v[layer], ValueType, toStorage: false);
    }

    private void Convert(float* rows, IntPtr stored, int type, bool toStorage)
    {
        // Row blocks of whole 32-element groups (dim is a multiple of 32 for Q8_0).
        int dim = _dim, count = Tokens, per = Math.Max(1, (1 << 16) / dim);
        nint ra = (nint)rows, sa = stored;
        QwenImage21CpuKernels.For((count + per - 1) / per, b =>
        {
            int r0 = b * per, rn = Math.Min(per, count - r0);
            long first = (long)r0 * dim, n = (long)rn * dim;
            float* f = (float*)ra + first;
            switch (type)
            {
                case GgmlF32:
                    if (toStorage) Buffer.MemoryCopy(f, (float*)sa + first, n * 4, n * 4);
                    else Buffer.MemoryCopy((float*)sa + first, f, n * 4, n * 4);
                    break;
                case GgmlF16:
                    if (toStorage) TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(f, (int)n), new Span<System.Half>((System.Half*)sa + first, (int)n));
                    else TensorPrimitives.ConvertToSingle(new ReadOnlySpan<System.Half>((System.Half*)sa + first, (int)n), new Span<float>(f, (int)n));
                    break;
                default:
                    byte* q = (byte*)sa + first / 32 * 34;
                    if (toStorage) ManagedQuantizedOps.QuantizeRowFromFloat32(GgmlQ8_0, f, (IntPtr)q, n);
                    else ManagedQuantizedOps.DequantizeRowToFloat32(GgmlQ8_0, (IntPtr)q, f, n);
                    break;
            }
        });
    }

    private void Release()
    {
        foreach (var p in _k) if (p != IntPtr.Zero) NativeMemory.AlignedFree((void*)p);
        foreach (var p in _v) if (p != IntPtr.Zero) NativeMemory.AlignedFree((void*)p);
        _k = Array.Empty<IntPtr>();
        _v = Array.Empty<IntPtr>();
        Filled = false;
        Declined = false;
        _owner = null;
    }

    public void Dispose() => Release();
}
