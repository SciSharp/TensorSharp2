// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using TensorSharp.Cpu;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage;

/// <summary>
/// Kernels of the pure-C# Qwen-Image-2.1 transformer (<see cref="QwenImage21ManagedDiT"/>).
/// </summary>
/// <remarks>
/// <para>The row-wise AdaLN/RMSNorm/RoPE/SwiGLU passes are memory bound and use
/// <see cref="TensorPrimitives"/> (itself Vector512 where available). The compute-bound
/// kernels - the projections (<see cref="GemmDequantNT"/>: F32 activations against
/// dequantized weight tiles), the segmented flash attention and the LoRA shrink/expand
/// products - come in three widths: AVX-512 (16 lanes), AVX2+FMA (8) and a portable tier
/// (TensorPrimitives/Vector128 row operations; AdvSimd on ARM64).</para>
/// <para>On this generation of client cores a 512-bit FMA issues once per cycle and a 256-bit
/// one twice, so AVX-512 does not raise the FMA ceiling; it pays through its 32 registers,
/// which hold a 6x4 dot tile or a 4x64 score/output tile (24 or 16 accumulators) plus its
/// operands without spills, against 4x2 and 4x16 for AVX2.</para>
/// </remarks>
internal static unsafe class QwenImage21CpuKernels
{
    /// <summary>Lanes of the compute-bound kernels: 16 (AVX-512), 8 (AVX2+FMA) or 1 (portable:
    /// <see cref="TensorPrimitives"/> and Vector128, which is AdvSimd on ARM64).
    /// TS_CPU_DISABLE_AVX512=1 selects the AVX2 kernels on an AVX-512 host so both paths are
    /// exercised on one machine; tests assign it directly. It narrows only this file's
    /// hand-written kernels: TensorPrimitives (the memory-bound row passes, conversions) keeps
    /// using Vector512. To measure an AVX2-only host, disable AVX-512 in the runtime instead:
    /// DOTNET_EnableAVX512=0 (.NET 10 ignores the older DOTNET_EnableAVX512F=0), which also
    /// turns Vector512.IsHardwareAccelerated off; DOTNET_EnableAVX2=0 leaves only Vector128,
    /// the portable width-1 tier.</summary>
    internal static int Width { get; set; } = DefaultWidth();

    internal static int DefaultWidth()
    {
        if (CpuIsa.Avx512) return 16;
        if (CpuIsa.Avx2Fma) return 8;
        return 1;
    }

    // The shared CPU pool, or Parallel.For under TS_CPU_POOL=0 (see CpuWorkers).
    internal static int Workers => CpuWorkers.Shared.ThreadCount;

    /// <summary>Runs <paramref name="body"/> for every block on the shared CPU pool.</summary>
    internal static void For(int blocks, Action<int> body) => CpuWorkers.Shared.For(blocks, body);

    /// <summary>Rows per task of a row-wise pass: a few tasks per worker, and one task when
    /// the whole pass is too small to be worth a dispatch.</summary>
    private static int RowsPerTask(int rows, long rowElements)
    {
        if ((long)rows * rowElements < 32 * 1024) return Math.Max(1, rows);
        int tasks = Math.Max(1, Workers * 4);
        return Math.Max(1, (rows + tasks - 1) / tasks);
    }

    // ---- row-wise passes ---------------------------------------------------------------

    /// <summary>y = LayerNorm(x) (no affine) * scale, where rows below <paramref name="prefixRows"/>
    /// take <paramref name="prefixScale"/> (the t=0 modulation) and the rest <paramref name="targetScale"/>.
    /// Same order of operations as ggml_norm followed by ggml_mul.</summary>
    internal static void LayerNormScale(float* x, float* y, int rows, int dim, float eps,
        float* targetScale, float* prefixScale, int prefixRows)
    {
        nint xa = (nint)x, ya = (nint)y, ta = (nint)targetScale, pa = (nint)prefixScale;
        int per = RowsPerTask(rows, dim);
        For((rows + per - 1) / per, b =>
        {
            int end = Math.Min(rows, (b + 1) * per);
            for (int r = b * per; r < end; r++)
            {
                var src = new ReadOnlySpan<float>((float*)xa + (long)r * dim, dim);
                var dst = new Span<float>((float*)ya + (long)r * dim, dim);
                float mean = TensorPrimitives.Sum(src) / dim;
                TensorPrimitives.Subtract(src, mean, dst);
                float variance = TensorPrimitives.SumOfSquares(dst) / dim;
                TensorPrimitives.Multiply(dst, 1f / MathF.Sqrt(variance + eps), dst);
                TensorPrimitives.Multiply(dst, new ReadOnlySpan<float>((float*)(r < prefixRows ? pa : ta), dim), dst);
            }
        });
    }

    /// <summary>y = RMSNorm(x) * gain per row (ggml_rms_norm then ggml_mul); x and y may alias.</summary>
    internal static void RmsNormGain(float* x, float* y, int rows, int dim, float eps, float* gain)
    {
        nint xa = (nint)x, ya = (nint)y, ga = (nint)gain;
        int per = RowsPerTask(rows, dim);
        For((rows + per - 1) / per, b =>
        {
            int end = Math.Min(rows, (b + 1) * per);
            for (int r = b * per; r < end; r++)
            {
                var src = new ReadOnlySpan<float>((float*)xa + (long)r * dim, dim);
                var dst = new Span<float>((float*)ya + (long)r * dim, dim);
                float scale = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(src) / dim + eps);
                TensorPrimitives.Multiply(src, scale, dst);
                TensorPrimitives.Multiply(dst, new ReadOnlySpan<float>((float*)ga, dim), dst);
            }
        });
    }

    /// <summary>joint += h * gate per row, rows below <paramref name="prefixRows"/> gated by
    /// <paramref name="prefixGate"/> (tanh of the t=0 modulation).</summary>
    internal static void GatedAdd(float* joint, float* h, int hStride, int rows, int dim,
        float* targetGate, float* prefixGate, int prefixRows)
    {
        nint ja = (nint)joint, ha = (nint)h, ta = (nint)targetGate, pa = (nint)prefixGate;
        int per = RowsPerTask(rows, dim);
        For((rows + per - 1) / per, b =>
        {
            int end = Math.Min(rows, (b + 1) * per);
            for (int r = b * per; r < end; r++)
            {
                var dst = new Span<float>((float*)ja + (long)r * dim, dim);
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>((float*)ha + (long)r * hStride, dim),
                    new ReadOnlySpan<float>((float*)(r < prefixRows ? pa : ta), dim), dst, dst);
            }
        });
    }

    /// <summary>
    /// Per-head RMSNorm with gains followed by the 3-axis RoPE, in place, exactly as the native
    /// Builder::rope: each head's pairs (x[2i], x[2i+1]) rotate by (cos[i], sin[i]) and are
    /// written as concat(even', odd'). Q and K share that permutation, so their dot products
    /// are those of the interleaved rotation; V is not rotated and keeps its layout.
    /// </summary>
    /// <param name="cos">[rows, headDim/2] angles of these rows.</param>
    internal static void RmsNormRope(float* x, int rows, int stride, int heads, int headDim, float* gain,
        float* cos, float* sin, float eps)
    {
        nint xa = (nint)x, ga = (nint)gain, ca = (nint)cos, sa = (nint)sin;
        int half = headDim / 2;
        int per = RowsPerTask(rows, (long)heads * headDim);
        For((rows + per - 1) / per, b =>
        {
            int end = Math.Min(rows, (b + 1) * per);
            float* g = (float*)ga;
            float* tmp = stackalloc float[headDim];
            for (int r = b * per; r < end; r++)
            {
                float* c = (float*)ca + (long)r * half, s = (float*)sa + (long)r * half;
                for (int h = 0; h < heads; h++)
                {
                    float* v = (float*)xa + (long)r * stride + h * headDim;
                    float scale = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(new ReadOnlySpan<float>(v, headDim)) / headDim + eps);
                    for (int i = 0; i < half; i++)
                    {
                        float e = v[2 * i] * scale * g[2 * i];
                        float o = v[2 * i + 1] * scale * g[2 * i + 1];
                        tmp[i] = e * c[i] - o * s[i];
                        tmp[half + i] = o * c[i] + e * s[i];
                    }
                    new ReadOnlySpan<float>(tmp, headDim).CopyTo(new Span<float>(v, headDim));
                }
            }
        });
    }

    /// <summary>Rows hold [gate | up] (<paramref name="ff"/> each); the gate half becomes
    /// silu(gate) * up (ggml_swiglu on the fused projection, or swiglu_split on the halves).</summary>
    internal static void SwiGlu(float* gu, int stride, int rows, int ff)
    {
        nint ga = (nint)gu;
        int width = Width;
        int per = RowsPerTask(rows, 2L * ff);
        For((rows + per - 1) / per, b =>
        {
            int end = Math.Min(rows, (b + 1) * per);
            for (int r = b * per; r < end; r++)
            {
                float* g = (float*)ga + (long)r * stride, u = g + ff;
                int i = 0;
                if (width == 16)
                {
                    var one = Vector512.Create(1f);
                    for (; i + 16 <= ff; i += 16)
                    {
                        var x = Vector512.Load(g + i);
                        (x / (one + Vector512.Exp(-x)) * Vector512.Load(u + i)).Store(g + i);
                    }
                }
                else if (width == 8)
                {
                    var one = Vector256.Create(1f);
                    for (; i + 8 <= ff; i += 8)
                    {
                        var x = Vector256.Load(g + i);
                        (x / (one + Vector256.Exp(-x)) * Vector256.Load(u + i)).Store(g + i);
                    }
                }
                else if (Vector128.IsHardwareAccelerated)
                {
                    var one = Vector128.Create(1f);
                    for (; i + 4 <= ff; i += 4)
                    {
                        var x = Vector128.Load(g + i);
                        (x / (one + Vector128.Exp(-x)) * Vector128.Load(u + i)).Store(g + i);
                    }
                }
                for (; i < ff; i++) g[i] = g[i] / (1f + MathF.Exp(-g[i])) * u[i];
            }
        });
    }

    internal static float Silu(float x) => x / (1f + MathF.Exp(-x));

    /// <summary>ggml_gelu's tanh approximation. <paramref name="fp16Table"/> reproduces ggml-cpu,
    /// which looks the value up in an F16 table (input and output rounded to F16).</summary>
    internal static float Gelu(float x, bool fp16Table)
    {
        const float a = 0.044715f, sqrt2OverPi = 0.79788456080286535587989211986876f;
        if (!fp16Table) return 0.5f * x * (1f + MathF.Tanh(sqrt2OverPi * x * (1f + a * x * x)));
        if (x <= -10f) return 0f;
        if (x >= 10f) return x;
        float h = (float)(System.Half)x;
        return (float)(System.Half)(0.5f * h * (1f + MathF.Tanh(sqrt2OverPi * h * (1f + a * h * h))));
    }

    // ---- K transposition ---------------------------------------------------------------

    private const int KeyTile = 64;

    /// <summary>Key stride of <see cref="TransposeKeys"/>'s output: whole 64-key tiles, plus one
    /// cache line when that is a multiple of 4 KB, whose rows would share L1 sets.</summary>
    internal static int KeyStride(int keys)
    {
        int stride = (keys + KeyTile - 1) / KeyTile * KeyTile;
        return stride % 1024 == 0 ? stride + 16 : stride;
    }

    /// <summary>kt[h, d, j] = k[j, h * headDim + d] for j &lt; keys, zero up to <see cref="KeyStride"/>.
    /// The score kernel then reads one contiguous run of keys per head dimension.</summary>
    internal static void TransposeKeys(float* k, int kStride, int keys, int heads, int headDim, float* kt)
    {
        nint ka = (nint)k, ta = (nint)kt;
        int stride = KeyStride(keys), tiles = stride / KeyTile;
        For(heads * tiles, task =>
        {
            int h = task / tiles, j0 = task % tiles * KeyTile, jn = Math.Min(KeyTile, keys - j0);
            float* src = (float*)ka + h * headDim;
            float* dst = (float*)ta + (long)h * headDim * stride + j0;
            for (int d = 0; d < headDim; d++)
            {
                float* row = dst + (long)d * stride;
                float* col = src + (long)j0 * kStride + d;
                int j = 0;
                for (; j < jn; j++) row[j] = col[(long)j * kStride];
                for (; j < KeyTile; j++) row[j] = 0f;
            }
        });
    }

    /// <summary>vh[h, j, :] = v[j, h * headDim ..]: each head's values contiguous. Token-major
    /// rows are 16 KB apart at 4096 wide, so the 64 rows of one head's value tile all map to the
    /// same few L1 sets and evict each other; packed, the tile is 32 KB of sequential lines.</summary>
    internal static void GatherValues(float* v, int vStride, int keys, int heads, int headDim, float* vh)
    {
        nint va = (nint)v, ha = (nint)vh;
        int tiles = (keys + KeyTile - 1) / KeyTile;
        For(heads * tiles, task =>
        {
            int h = task / tiles, j0 = task % tiles * KeyTile, jn = Math.Min(KeyTile, keys - j0);
            long bytes = headDim * sizeof(float);
            for (int j = j0; j < j0 + jn; j++)
                Buffer.MemoryCopy((float*)va + (long)j * vStride + h * headDim, (float*)ha + ((long)h * keys + j) * headDim, bytes, bytes);
        });
    }

    // ---- segmented flash attention -----------------------------------------------------

    private const int QueryTile = 64;

    /// <summary>
    /// Attention of the queries in <paramref name="segments"/> (from <paramref name="firstSegment"/>)
    /// over the keys of the whole sequence, as the native graph: a text segment is causal over
    /// keys [0, position], an image segment bidirectional over [0, segment end).
    /// </summary>
    /// <param name="q">Rows of global positions [queryBase, ...), <paramref name="dim"/> apart, head h at h * headDim.</param>
    /// <param name="kt"><see cref="TransposeKeys"/> of the whole sequence's keys.</param>
    /// <param name="v">Values of the whole sequence: key j of head h at v + h * vHeadStride + j * vRowStride
    /// (token-major [keys, dim], or head-major after <see cref="GatherValues"/>).</param>
    /// <param name="o">Output rows, laid out as <paramref name="q"/>.</param>
    /// <remarks>Tasks are (head, 64-query tile) pairs; each streams 64-key tiles through an online
    /// softmax, so a query tile's keys are read once and its scores never leave L1.</remarks>
    internal static void Attention(float* q, float* kt, int keys, float* v, int vRowStride, long vHeadStride, float* o,
        int dim, int heads, int headDim, QwenImage21Segment[] segments, int firstSegment, int queryBase, float scale)
    {
        int ktStride = KeyStride(keys);
        int width = Width;
        if (headDim % 64 != 0) width = 1;
        // Flatten (segment, query tile) pairs; the target image usually dominates, and the
        // pool's atomic claiming balances the rest.
        int tilesTotal = 0;
        for (int s = firstSegment; s < segments.Length; s++)
            tilesTotal += (segments[s].End - segments[s].Start + QueryTile - 1) / QueryTile;
        var tileSegment = new int[tilesTotal];
        var tileStart = new int[tilesTotal];
        int t = 0;
        for (int s = firstSegment; s < segments.Length; s++)
            for (int start = segments[s].Start; start < segments[s].End; start += QueryTile, t++)
            {
                tileSegment[t] = s;
                tileStart[t] = start;
            }
        nint qa = (nint)q, ka = (nint)kt, va = (nint)v, oa = (nint)o;
        For(tilesTotal * heads, task =>
        {
            int h = task % heads, tile = task / heads;
            var seg = segments[tileSegment[tile]];
            int g0 = tileStart[tile], rows = Math.Min(QueryTile, seg.End - g0);
            long rowOffset = (long)(g0 - queryBase) * dim + h * headDim;
            AttendTile(width, (float*)qa + rowOffset, dim, (float*)ka + (long)h * headDim * ktStride, ktStride,
                (float*)va + h * vHeadStride, vRowStride, (float*)oa + rowOffset, dim, rows, headDim,
                seg.End, seg.IsImage != 0 ? -1 : g0, scale);
        });
    }

    /// <summary>One (head, query tile): keys [0, nk), causal from global position
    /// <paramref name="causalBase"/> for query row 0 when it is not negative.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AttendTile(int width, float* q, int qStride, float* kt, int ktStride, float* v, int vStride,
        float* o, int oStride, int rows, int headDim, int nk, int causalBase, float scale)
    {
        float* s = stackalloc float[QueryTile * KeyTile];
        float* acc = stackalloc float[QueryTile * headDim];
        float* rowMax = stackalloc float[QueryTile];
        float* rowSum = stackalloc float[QueryTile];
        new Span<float>(acc, QueryTile * headDim).Clear();
        for (int i = 0; i < QueryTile; i++) { rowMax[i] = float.NegativeInfinity; rowSum[i] = 0f; }
        int kEnd = causalBase < 0 ? nk : Math.Min(nk, causalBase + rows);
        for (int k0 = 0; k0 < kEnd; k0 += KeyTile)
        {
            int kn = Math.Min(KeyTile, kEnd - k0);
            for (int i = 0; i < rows; i += 4)
            {
                int n = Math.Min(4, rows - i);
                if (width == 16) Scores16(q + (long)i * qStride, qStride, n, kt + k0, ktStride, headDim, s + i * KeyTile);
                else if (width == 8) Scores8(q + (long)i * qStride, qStride, n, kt + k0, ktStride, headDim, s + i * KeyTile);
                else ScoresScalar(q + (long)i * qStride, qStride, n, kt + k0, ktStride, headDim, s + i * KeyTile);
            }
            for (int i = 0; i < rows; i++)
            {
                int limit = (causalBase < 0 ? nk : Math.Min(nk, causalBase + i + 1)) - k0;
                float* row = s + i * KeyTile;
                if (limit <= 0)
                {
                    // Every key of this tile is in this text row's future.
                    new Span<float>(row, KeyTile).Clear();
                    continue;
                }
                float corr = SoftmaxTile(width, row, Math.Min(limit, kn), scale, ref rowMax[i], ref rowSum[i]);
                if (corr != 1f)
                {
                    var a = new Span<float>(acc + i * headDim, headDim);
                    TensorPrimitives.Multiply(a, corr, a);
                }
            }
            for (int i = 0; i < rows; i += 4)
            {
                if (width == 16) Values16(s + i * KeyTile, v + (long)k0 * vStride, vStride, kn, acc + i * headDim, headDim);
                else if (width == 8) Values8(s + i * KeyTile, v + (long)k0 * vStride, vStride, kn, acc + i * headDim, headDim);
                else ValuesScalar(s + i * KeyTile, Math.Min(4, rows - i), v + (long)k0 * vStride, vStride, kn, acc + i * headDim, headDim);
            }
        }
        for (int i = 0; i < rows; i++)
        {
            // ggml's flash attention also multiplies by the reciprocal of the sum.
            float inv = rowSum[i] == 0f ? 0f : 1f / rowSum[i];
            TensorPrimitives.Multiply(new ReadOnlySpan<float>(acc + i * headDim, headDim), inv, new Span<float>(o + (long)i * oStride, headDim));
        }
    }

    private const float FlushBelow = -87f;

    /// <summary>Online-softmax update of one row of a score tile: scales the first
    /// <paramref name="valid"/> scores, replaces them by exp(score - newMax) (the rest of the
    /// tile by 0), and returns the factor that rescales the row's earlier accumulation.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float SoftmaxTile(int width, float* row, int valid, float scale, ref float max, ref float sum)
    {
        float tileMax = float.NegativeInfinity;
        int j = 0;
        if (width == 16)
        {
            var m = Vector512.Create(float.NegativeInfinity);
            var sc = Vector512.Create(scale);
            for (; j + 16 <= valid; j += 16)
            {
                var x = Vector512.Load(row + j) * sc;
                x.Store(row + j);
                m = Vector512.Max(m, x);
            }
            tileMax = HorizontalMax(m);
        }
        else if (width == 8)
        {
            var m = Vector256.Create(float.NegativeInfinity);
            var sc = Vector256.Create(scale);
            for (; j + 8 <= valid; j += 8)
            {
                var x = Vector256.Load(row + j) * sc;
                x.Store(row + j);
                m = Vector256.Max(m, x);
            }
            tileMax = HorizontalMax(m);
        }
        for (; j < valid; j++)
        {
            row[j] *= scale;
            if (row[j] > tileMax) tileMax = row[j];
        }
        float newMax = MathF.Max(max, tileMax);
        // Probabilities below e^-87 (and a rescale by one) would be denormal: numerically
        // nothing next to the row's maximum term of 1, but each FMA that reads a denormal takes
        // a microcode assist that costs ~100 cycles. Flush them to zero instead.
        float corr = max - newMax < FlushBelow ? 0f : MathF.Exp(max - newMax);
        float tileSum = 0f;
        j = 0;
        if (width == 16)
        {
            var nm = Vector512.Create(newMax);
            var floor = Vector512.Create(FlushBelow);
            var acc = Vector512<float>.Zero;
            for (; j + 16 <= valid; j += 16)
            {
                var x = Vector512.Load(row + j) - nm;
                var p = Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(x, floor), Vector512.Exp(x), Vector512<float>.Zero);
                p.Store(row + j);
                acc += p;
            }
            tileSum = Vector512.Sum(acc);
        }
        else if (width == 8)
        {
            var nm = Vector256.Create(newMax);
            var floor = Vector256.Create(FlushBelow);
            var acc = Vector256<float>.Zero;
            for (; j + 8 <= valid; j += 8)
            {
                var x = Vector256.Load(row + j) - nm;
                var p = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(x, floor), Vector256.Exp(x), Vector256<float>.Zero);
                p.Store(row + j);
                acc += p;
            }
            tileSum = Vector256.Sum(acc);
        }
        for (; j < valid; j++)
        {
            float x = row[j] - newMax;
            row[j] = x < FlushBelow ? 0f : MathF.Exp(x);
            tileSum += row[j];
        }
        for (; j < KeyTile; j++) row[j] = 0f;
        sum = sum * corr + tileSum;
        max = newMax;
        return corr;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMax(Vector512<float> v) =>
        HorizontalMax(Vector256.Max(v.GetLower(), v.GetUpper()));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMax(Vector256<float> v)
    {
        var m = Vector128.Max(v.GetLower(), v.GetUpper());
        return MathF.Max(MathF.Max(m[0], m[1]), MathF.Max(m[2], m[3]));
    }

    /// <summary>Scores of 4 query rows (rows past <paramref name="rows"/> repeat the last one)
    /// against a 64-key tile of the transposed keys: 16 accumulators of 16 lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Scores16(float* q, int qStride, int rows, float* kt, int ktStride, int headDim, float* s)
    {
        float* q0 = q, q1 = rows > 1 ? q0 + qStride : q0, q2 = rows > 2 ? q1 + qStride : q1, q3 = rows > 3 ? q2 + qStride : q2;
        Vector512<float> a00 = default, a01 = default, a02 = default, a03 = default;
        Vector512<float> a10 = default, a11 = default, a12 = default, a13 = default;
        Vector512<float> a20 = default, a21 = default, a22 = default, a23 = default;
        Vector512<float> a30 = default, a31 = default, a32 = default, a33 = default;
        float* k = kt;
        for (int d = 0; d < headDim; d++, k += ktStride)
        {
            var k0 = Vector512.Load(k);
            var k1 = Vector512.Load(k + 16);
            var k2 = Vector512.Load(k + 32);
            var k3 = Vector512.Load(k + 48);
            var b = Vector512.Create(q0[d]);
            a00 = Vector512.FusedMultiplyAdd(b, k0, a00); a01 = Vector512.FusedMultiplyAdd(b, k1, a01);
            a02 = Vector512.FusedMultiplyAdd(b, k2, a02); a03 = Vector512.FusedMultiplyAdd(b, k3, a03);
            b = Vector512.Create(q1[d]);
            a10 = Vector512.FusedMultiplyAdd(b, k0, a10); a11 = Vector512.FusedMultiplyAdd(b, k1, a11);
            a12 = Vector512.FusedMultiplyAdd(b, k2, a12); a13 = Vector512.FusedMultiplyAdd(b, k3, a13);
            b = Vector512.Create(q2[d]);
            a20 = Vector512.FusedMultiplyAdd(b, k0, a20); a21 = Vector512.FusedMultiplyAdd(b, k1, a21);
            a22 = Vector512.FusedMultiplyAdd(b, k2, a22); a23 = Vector512.FusedMultiplyAdd(b, k3, a23);
            b = Vector512.Create(q3[d]);
            a30 = Vector512.FusedMultiplyAdd(b, k0, a30); a31 = Vector512.FusedMultiplyAdd(b, k1, a31);
            a32 = Vector512.FusedMultiplyAdd(b, k2, a32); a33 = Vector512.FusedMultiplyAdd(b, k3, a33);
        }
        // The score tile always has room for 4 rows (QueryTile is a multiple of 4).
        a00.Store(s); a01.Store(s + 16); a02.Store(s + 32); a03.Store(s + 48);
        s += KeyTile;
        a10.Store(s); a11.Store(s + 16); a12.Store(s + 32); a13.Store(s + 48);
        s += KeyTile;
        a20.Store(s); a21.Store(s + 16); a22.Store(s + 32); a23.Store(s + 48);
        s += KeyTile;
        a30.Store(s); a31.Store(s + 16); a32.Store(s + 32); a33.Store(s + 48);
    }

    /// <summary>AVX2 scores: 4 rows x 16 keys (8 accumulators) at a time over the 64-key tile.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Scores8(float* q, int qStride, int rows, float* kt, int ktStride, int headDim, float* s)
    {
        float* q0 = q, q1 = rows > 1 ? q0 + qStride : q0, q2 = rows > 2 ? q1 + qStride : q1, q3 = rows > 3 ? q2 + qStride : q2;
        for (int c = 0; c < KeyTile; c += 16)
        {
            Vector256<float> a00 = default, a01 = default, a10 = default, a11 = default;
            Vector256<float> a20 = default, a21 = default, a30 = default, a31 = default;
            float* k = kt + c;
            for (int d = 0; d < headDim; d++, k += ktStride)
            {
                var k0 = Vector256.Load(k);
                var k1 = Vector256.Load(k + 8);
                var b = Vector256.Create(q0[d]);
                a00 = Fma.MultiplyAdd(b, k0, a00); a01 = Fma.MultiplyAdd(b, k1, a01);
                b = Vector256.Create(q1[d]);
                a10 = Fma.MultiplyAdd(b, k0, a10); a11 = Fma.MultiplyAdd(b, k1, a11);
                b = Vector256.Create(q2[d]);
                a20 = Fma.MultiplyAdd(b, k0, a20); a21 = Fma.MultiplyAdd(b, k1, a21);
                b = Vector256.Create(q3[d]);
                a30 = Fma.MultiplyAdd(b, k0, a30); a31 = Fma.MultiplyAdd(b, k1, a31);
            }
            float* r = s + c;
            a00.Store(r); a01.Store(r + 8); r += KeyTile;
            a10.Store(r); a11.Store(r + 8); r += KeyTile;
            a20.Store(r); a21.Store(r + 8); r += KeyTile;
            a30.Store(r); a31.Store(r + 8);
        }
    }

    /// <summary>Portable scores: per head dimension, one contiguous 64-key row of the transposed
    /// keys scaled into the score row (TensorPrimitives vectorizes it; the head dimension
    /// innermost would stride <paramref name="ktStride"/> floats per step).</summary>
    private static void ScoresScalar(float* q, int qStride, int rows, float* kt, int ktStride, int headDim, float* s)
    {
        for (int i = 0; i < rows; i++)
        {
            var row = new Span<float>(s + i * KeyTile, KeyTile);
            row.Clear();
            float* qi = q + (long)i * qStride;
            for (int d = 0; d < headDim; d++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(kt + (long)d * ktStride, KeyTile), qi[d], row, row);
        }
    }

    /// <summary>acc[4 rows] += p[4 rows, kn] * v[kn, headDim], 64 output lanes per pass.
    /// Rows past the tile's valid rows only feed accumulator rows that are never written out.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Values16(float* p, float* v, int vStride, int kn, float* acc, int headDim)
    {
        float* p0 = p, p1 = p + KeyTile, p2 = p1 + KeyTile, p3 = p2 + KeyTile;
        for (int d0 = 0; d0 < headDim; d0 += 64)
        {
            float* c0 = acc + d0, c1 = c0 + headDim, c2 = c1 + headDim, c3 = c2 + headDim;
            var a00 = Vector512.Load(c0); var a01 = Vector512.Load(c0 + 16); var a02 = Vector512.Load(c0 + 32); var a03 = Vector512.Load(c0 + 48);
            var a10 = Vector512.Load(c1); var a11 = Vector512.Load(c1 + 16); var a12 = Vector512.Load(c1 + 32); var a13 = Vector512.Load(c1 + 48);
            var a20 = Vector512.Load(c2); var a21 = Vector512.Load(c2 + 16); var a22 = Vector512.Load(c2 + 32); var a23 = Vector512.Load(c2 + 48);
            var a30 = Vector512.Load(c3); var a31 = Vector512.Load(c3 + 16); var a32 = Vector512.Load(c3 + 32); var a33 = Vector512.Load(c3 + 48);
            float* vr = v + d0;
            for (int j = 0; j < kn; j++, vr += vStride)
            {
                var v0 = Vector512.Load(vr);
                var v1 = Vector512.Load(vr + 16);
                var v2 = Vector512.Load(vr + 32);
                var v3 = Vector512.Load(vr + 48);
                var b = Vector512.Create(p0[j]);
                a00 = Vector512.FusedMultiplyAdd(b, v0, a00); a01 = Vector512.FusedMultiplyAdd(b, v1, a01);
                a02 = Vector512.FusedMultiplyAdd(b, v2, a02); a03 = Vector512.FusedMultiplyAdd(b, v3, a03);
                b = Vector512.Create(p1[j]);
                a10 = Vector512.FusedMultiplyAdd(b, v0, a10); a11 = Vector512.FusedMultiplyAdd(b, v1, a11);
                a12 = Vector512.FusedMultiplyAdd(b, v2, a12); a13 = Vector512.FusedMultiplyAdd(b, v3, a13);
                b = Vector512.Create(p2[j]);
                a20 = Vector512.FusedMultiplyAdd(b, v0, a20); a21 = Vector512.FusedMultiplyAdd(b, v1, a21);
                a22 = Vector512.FusedMultiplyAdd(b, v2, a22); a23 = Vector512.FusedMultiplyAdd(b, v3, a23);
                b = Vector512.Create(p3[j]);
                a30 = Vector512.FusedMultiplyAdd(b, v0, a30); a31 = Vector512.FusedMultiplyAdd(b, v1, a31);
                a32 = Vector512.FusedMultiplyAdd(b, v2, a32); a33 = Vector512.FusedMultiplyAdd(b, v3, a33);
            }
            a00.Store(c0); a01.Store(c0 + 16); a02.Store(c0 + 32); a03.Store(c0 + 48);
            a10.Store(c1); a11.Store(c1 + 16); a12.Store(c1 + 32); a13.Store(c1 + 48);
            a20.Store(c2); a21.Store(c2 + 16); a22.Store(c2 + 32); a23.Store(c2 + 48);
            a30.Store(c3); a31.Store(c3 + 16); a32.Store(c3 + 32); a33.Store(c3 + 48);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Values8(float* p, float* v, int vStride, int kn, float* acc, int headDim)
    {
        float* p0 = p, p1 = p + KeyTile, p2 = p1 + KeyTile, p3 = p2 + KeyTile;
        for (int d0 = 0; d0 < headDim; d0 += 16)
        {
            float* c0 = acc + d0, c1 = c0 + headDim, c2 = c1 + headDim, c3 = c2 + headDim;
            var a00 = Vector256.Load(c0); var a01 = Vector256.Load(c0 + 8);
            var a10 = Vector256.Load(c1); var a11 = Vector256.Load(c1 + 8);
            var a20 = Vector256.Load(c2); var a21 = Vector256.Load(c2 + 8);
            var a30 = Vector256.Load(c3); var a31 = Vector256.Load(c3 + 8);
            float* vr = v + d0;
            for (int j = 0; j < kn; j++, vr += vStride)
            {
                var v0 = Vector256.Load(vr);
                var v1 = Vector256.Load(vr + 8);
                var b = Vector256.Create(p0[j]);
                a00 = Fma.MultiplyAdd(b, v0, a00); a01 = Fma.MultiplyAdd(b, v1, a01);
                b = Vector256.Create(p1[j]);
                a10 = Fma.MultiplyAdd(b, v0, a10); a11 = Fma.MultiplyAdd(b, v1, a11);
                b = Vector256.Create(p2[j]);
                a20 = Fma.MultiplyAdd(b, v0, a20); a21 = Fma.MultiplyAdd(b, v1, a21);
                b = Vector256.Create(p3[j]);
                a30 = Fma.MultiplyAdd(b, v0, a30); a31 = Fma.MultiplyAdd(b, v1, a31);
            }
            a00.Store(c0); a01.Store(c0 + 8);
            a10.Store(c1); a11.Store(c1 + 8);
            a20.Store(c2); a21.Store(c2 + 8);
            a30.Store(c3); a31.Store(c3 + 8);
        }
    }

    private static void ValuesScalar(float* p, int rows, float* v, int vStride, int kn, float* acc, int headDim)
    {
        for (int i = 0; i < rows; i++)
        {
            var a = new Span<float>(acc + i * headDim, headDim);
            for (int j = 0; j < kn; j++)
            {
                float w = p[i * KeyTile + j];
                if (w == 0f) continue;
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(v + (long)j * vStride, headDim), w, a, a);
            }
        }
    }

    // ---- float products (projections, LoRA factors, output heads) ---------------------

    // Depth of one pass of the dot tiles: a 6-row activation panel (24 KB) stays in L1 while
    // the tile walks a task's weight columns, whose 1024-deep slices (128 KB for 32 columns)
    // stream from L2. Without it the panel (6 x 16 KB at 4096 wide) fell out of L1 for every
    // column tile and the L2 traffic, not the FMAs, bounded the product. Measured at 1047 rows:
    // 512 is ~20% slower (twice the horizontal sums), 2048 on par. TS_QWEN21_CPU_DEPTH
    // overrides it (a multiple of 16) for tuning.
    private static readonly int DepthBlock =
        int.TryParse(Environment.GetEnvironmentVariable("TS_QWEN21_CPU_DEPTH"), out int depth) && depth >= 16 ? depth / 16 * 16 : 1024;

    /// <summary>c[i, j] = dot(a[i, :k], b[j, :k]) for i &lt; m, j &lt; n (c is overwritten).
    /// The dot form suits a long k: a LoRA shrink (k = 4096..12288, n = its rank) or an output
    /// head; its horizontal sums are paid once per output and depth block.</summary>
    internal static void GemmNT(float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k)
    {
        if (m <= 0 || n <= 0) return;
        const int rowBlock = 12, colBlock = 16;
        int rowTiles = (m + rowBlock - 1) / rowBlock, colTiles = (n + colBlock - 1) / colBlock;
        nint aa = (nint)a, ba = (nint)b, ca = (nint)c;
        int width = Width;
        Action<int> body = task =>
        {
            int r0 = task / colTiles * rowBlock, c0 = task % colTiles * colBlock;
            DotRange(width, (float*)aa, lda, (float*)ba, ldb, (float*)ca, ldc,
                r0, Math.Min(m, r0 + rowBlock), c0, Math.Min(n, c0 + colBlock), k);
        };
        if ((long)m * n * k < 1 << 18) { for (int task = 0; task < rowTiles * colTiles; task++) body(task); return; }
        For(rowTiles * colTiles, body);
    }

    /// <summary>Rows [r0, r1) x columns [c0, c1) of <see cref="GemmNT"/> on the calling thread,
    /// in <see cref="DepthBlock"/> passes: 6x4 tiles (24 accumulators) on AVX-512, 4x2 on AVX2.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DotRange(int width, float* a, int lda, float* b, int ldb, float* c, int ldc,
        int r0, int r1, int c0, int c1, int k)
    {
        if (width == 1)
        {
            // Portable: TensorPrimitives.Dot is Vector128 (AdvSimd on ARM64) with several
            // accumulators, where a scalar loop is one serial dependency chain.
            for (int i = r0; i < r1; i++)
                for (int j = c0; j < c1; j++)
                    c[(long)i * ldc + j] = TensorPrimitives.Dot(new ReadOnlySpan<float>(a + (long)i * lda, k),
                        new ReadOnlySpan<float>(b + (long)j * ldb, k));
            return;
        }
        int tileRows = width == 16 ? 6 : 4, tileCols = width == 16 ? 4 : 2;
        for (int k0 = 0; k0 < k; k0 += DepthBlock)
        {
            int kn = Math.Min(DepthBlock, k - k0);
            bool accumulate = k0 > 0;
            for (int i = r0; i < r1; i += tileRows)
            {
                int rows = Math.Min(tileRows, r1 - i);
                float* ai = a + (long)i * lda + k0;
                for (int j = c0; j < c1; j += tileCols)
                {
                    float* bj = b + (long)j * ldb + k0;
                    float* cij = c + (long)i * ldc + j;
                    if (width == 16) DotTile16(ai, lda, rows, bj, ldb, Math.Min(4, c1 - j), cij, ldc, kn, accumulate);
                    else DotTile8(ai, lda, rows, bj, ldb, Math.Min(2, c1 - j), cij, ldc, kn, accumulate);
                }
            }
        }
    }

    /// <summary>
    /// y[i, j] = dot(x[i], dequant(w row j)): the F32 projection of the pure-C# transformer.
    /// Each task dequantizes a block of weight rows (<see cref="QwenImage21CpuDequant"/>, the
    /// same values as ggml's dequantizers) into an L2-resident scratch and runs every input row
    /// against it, so the weights stay as stored (file-mapped, quantized) and are expanded once
    /// per projection. The activations are not quantized, unlike ggml's integer dot.
    /// </summary>
    internal static void GemmDequantNT(float* x, int ldx, int rows, int ggmlType, byte* w, long rowBytes, int k, int n,
        float* y, int ldy)
    {
        if (rows <= 0 || n <= 0) return;
        // 32 weight rows per task: 128 KB of each depth block stays in L2, and even a 4096-wide
        // projection still yields 16 tasks per pool worker.
        const int colBlock = 32;
        nint xa = (nint)x, wa = (nint)w, ya = (nint)y;
        int width = Width;
        For((n + colBlock - 1) / colBlock, task =>
        {
            int c0 = task * colBlock, cn = Math.Min(colBlock, n - c0);
            float[] rented = System.Buffers.ArrayPool<float>.Shared.Rent(colBlock * k);
            try
            {
                fixed (float* block = rented)
                {
                    for (int c = 0; c < cn; c++)
                        QwenImage21CpuDequant.Row(width, ggmlType, (byte*)wa + (c0 + c) * rowBytes, block + (long)c * k, k);
                    DotRange(width, (float*)xa, ldx, block, k, (float*)ya + c0, ldy, 0, rows, 0, cn, k);
                }
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(rented); }
        });
    }

    /// <summary>6 rows x 4 columns of dots over <paramref name="k"/> (rows and columns past the
    /// valid ones repeat the last valid pointer; their sums are not stored).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DotTile16(float* a, int lda, int rows, float* b, int ldb, int cols, float* c, int ldc, int k, bool accumulate)
    {
        float* a0 = a, a1 = rows > 1 ? a0 + lda : a0, a2 = rows > 2 ? a1 + lda : a1;
        float* a3 = rows > 3 ? a2 + lda : a2, a4 = rows > 4 ? a3 + lda : a3, a5 = rows > 5 ? a4 + lda : a4;
        float* b0 = b, b1 = cols > 1 ? b0 + ldb : b0, b2 = cols > 2 ? b1 + ldb : b1, b3 = cols > 3 ? b2 + ldb : b2;
        Vector512<float> s00 = default, s01 = default, s02 = default, s03 = default;
        Vector512<float> s10 = default, s11 = default, s12 = default, s13 = default;
        Vector512<float> s20 = default, s21 = default, s22 = default, s23 = default;
        Vector512<float> s30 = default, s31 = default, s32 = default, s33 = default;
        Vector512<float> s40 = default, s41 = default, s42 = default, s43 = default;
        Vector512<float> s50 = default, s51 = default, s52 = default, s53 = default;
        int kk = 0;
        for (; kk + 16 <= k; kk += 16)
        {
            var y0 = Vector512.Load(b0 + kk); var y1 = Vector512.Load(b1 + kk);
            var y2 = Vector512.Load(b2 + kk); var y3 = Vector512.Load(b3 + kk);
            var x = Vector512.Load(a0 + kk);
            s00 = Vector512.FusedMultiplyAdd(x, y0, s00); s01 = Vector512.FusedMultiplyAdd(x, y1, s01);
            s02 = Vector512.FusedMultiplyAdd(x, y2, s02); s03 = Vector512.FusedMultiplyAdd(x, y3, s03);
            x = Vector512.Load(a1 + kk);
            s10 = Vector512.FusedMultiplyAdd(x, y0, s10); s11 = Vector512.FusedMultiplyAdd(x, y1, s11);
            s12 = Vector512.FusedMultiplyAdd(x, y2, s12); s13 = Vector512.FusedMultiplyAdd(x, y3, s13);
            x = Vector512.Load(a2 + kk);
            s20 = Vector512.FusedMultiplyAdd(x, y0, s20); s21 = Vector512.FusedMultiplyAdd(x, y1, s21);
            s22 = Vector512.FusedMultiplyAdd(x, y2, s22); s23 = Vector512.FusedMultiplyAdd(x, y3, s23);
            x = Vector512.Load(a3 + kk);
            s30 = Vector512.FusedMultiplyAdd(x, y0, s30); s31 = Vector512.FusedMultiplyAdd(x, y1, s31);
            s32 = Vector512.FusedMultiplyAdd(x, y2, s32); s33 = Vector512.FusedMultiplyAdd(x, y3, s33);
            x = Vector512.Load(a4 + kk);
            s40 = Vector512.FusedMultiplyAdd(x, y0, s40); s41 = Vector512.FusedMultiplyAdd(x, y1, s41);
            s42 = Vector512.FusedMultiplyAdd(x, y2, s42); s43 = Vector512.FusedMultiplyAdd(x, y3, s43);
            x = Vector512.Load(a5 + kk);
            s50 = Vector512.FusedMultiplyAdd(x, y0, s50); s51 = Vector512.FusedMultiplyAdd(x, y1, s51);
            s52 = Vector512.FusedMultiplyAdd(x, y2, s52); s53 = Vector512.FusedMultiplyAdd(x, y3, s53);
        }
        float* sums = stackalloc float[24]
        {
            Vector512.Sum(s00), Vector512.Sum(s01), Vector512.Sum(s02), Vector512.Sum(s03),
            Vector512.Sum(s10), Vector512.Sum(s11), Vector512.Sum(s12), Vector512.Sum(s13),
            Vector512.Sum(s20), Vector512.Sum(s21), Vector512.Sum(s22), Vector512.Sum(s23),
            Vector512.Sum(s30), Vector512.Sum(s31), Vector512.Sum(s32), Vector512.Sum(s33),
            Vector512.Sum(s40), Vector512.Sum(s41), Vector512.Sum(s42), Vector512.Sum(s43),
            Vector512.Sum(s50), Vector512.Sum(s51), Vector512.Sum(s52), Vector512.Sum(s53),
        };
        float** ar = stackalloc float*[6] { a0, a1, a2, a3, a4, a5 };
        float** br = stackalloc float*[4] { b0, b1, b2, b3 };
        StoreTile(sums, 4, ar, br, rows, cols, c, ldc, kk, k, accumulate);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DotTile8(float* a, int lda, int rows, float* b, int ldb, int cols, float* c, int ldc, int k, bool accumulate)
    {
        float* a0 = a, a1 = rows > 1 ? a0 + lda : a0, a2 = rows > 2 ? a1 + lda : a1, a3 = rows > 3 ? a2 + lda : a2;
        float* b0 = b, b1 = cols > 1 ? b0 + ldb : b0;
        Vector256<float> s00 = default, s01 = default, s10 = default, s11 = default;
        Vector256<float> s20 = default, s21 = default, s30 = default, s31 = default;
        int kk = 0;
        for (; kk + 8 <= k; kk += 8)
        {
            var y0 = Vector256.Load(b0 + kk); var y1 = Vector256.Load(b1 + kk);
            var x = Vector256.Load(a0 + kk);
            s00 = Fma.MultiplyAdd(x, y0, s00); s01 = Fma.MultiplyAdd(x, y1, s01);
            x = Vector256.Load(a1 + kk);
            s10 = Fma.MultiplyAdd(x, y0, s10); s11 = Fma.MultiplyAdd(x, y1, s11);
            x = Vector256.Load(a2 + kk);
            s20 = Fma.MultiplyAdd(x, y0, s20); s21 = Fma.MultiplyAdd(x, y1, s21);
            x = Vector256.Load(a3 + kk);
            s30 = Fma.MultiplyAdd(x, y0, s30); s31 = Fma.MultiplyAdd(x, y1, s31);
        }
        float* sums = stackalloc float[8]
        {
            Vector256.Sum(s00), Vector256.Sum(s01), Vector256.Sum(s10), Vector256.Sum(s11),
            Vector256.Sum(s20), Vector256.Sum(s21), Vector256.Sum(s30), Vector256.Sum(s31),
        };
        float** ar = stackalloc float*[4] { a0, a1, a2, a3 };
        float** br = stackalloc float*[2] { b0, b1 };
        StoreTile(sums, 2, ar, br, rows, cols, c, ldc, kk, k, accumulate);
    }

    /// <summary>Adds the scalar tail [kk, k) and stores (or accumulates) the valid part of a dot tile.</summary>
    private static void StoreTile(float* sums, int tileCols, float** ar, float** br, int rows, int cols,
        float* c, int ldc, int kk, int k, bool accumulate)
    {
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < cols; j++)
            {
                float sum = sums[r * tileCols + j];
                for (int t = kk; t < k; t++) sum += ar[r][t] * br[j][t];
                float* dst = c + (long)r * ldc + j;
                *dst = accumulate ? *dst + sum : sum;
            }
    }

    internal const int PanelWidth = 64;

    /// <summary>Floats of <see cref="PackPanels"/>' output for an [r, n] matrix.</summary>
    internal static long PanelFloats(int r, int n) => (long)((n + PanelWidth - 1) / PanelWidth) * r * PanelWidth;

    /// <summary>panels[p, t, c] = b[t, p * 64 + c], zero past n: an expand's right factor as
    /// 64-column panels of r x 256 contiguous bytes. Row-major rows of a 4096-wide factor are
    /// 16 KB apart, so a 64-column tile's r rows all fell into the same few L1 sets.</summary>
    internal static void PackPanels(float* b, int ldb, int r, int n, float* panels)
    {
        for (int p = 0, c0 = 0; c0 < n; p++, c0 += PanelWidth)
        {
            int cn = Math.Min(PanelWidth, n - c0);
            for (int t = 0; t < r; t++)
            {
                float* dst = panels + ((long)p * r + t) * PanelWidth;
                for (int c = 0; c < cn; c++) dst[c] = b[(long)t * ldb + c0 + c];
                for (int c = cn; c < PanelWidth; c++) dst[c] = 0f;
            }
        }
    }

    /// <summary>c[i, j] += sum_t a[i, t] * b[t, j] for i &lt; m, j &lt; n, with b as
    /// <see cref="PackPanels"/> panels: a LoRA expand (b = up transposed to [rank, out]), where
    /// each rank step is a broadcast and a row of FMAs.</summary>
    internal static void GemmNNAccumulate(float* a, int lda, float* panels, float* c, int ldc, int m, int n, int r)
    {
        if (m <= 0 || n <= 0 || r <= 0) return;
        const int rowBlock = 32, panelsPerTask = 4;
        int panelCount = (n + PanelWidth - 1) / PanelWidth;
        int rowTiles = (m + rowBlock - 1) / rowBlock, colTiles = (panelCount + panelsPerTask - 1) / panelsPerTask;
        nint aa = (nint)a, pa = (nint)panels, ca = (nint)c;
        int width = Width;
        Action<int> body = task =>
        {
            int r0 = task / colTiles * rowBlock, p0 = task % colTiles * panelsPerTask;
            int r1 = Math.Min(m, r0 + rowBlock), p1 = Math.Min(panelCount, p0 + panelsPerTask);
            for (int i = r0; i < r1; i += 4)
            {
                int rows = Math.Min(4, r1 - i);
                float* ai = (float*)aa + (long)i * lda;
                for (int p = p0; p < p1; p++)
                {
                    int j0 = p * PanelWidth, valid = Math.Min(PanelWidth, n - j0);
                    float* bp = (float*)pa + (long)p * r * PanelWidth;
                    float* ci = (float*)ca + (long)i * ldc + j0;
                    int jj = 0;
                    if (width == 16 && valid == PanelWidth)
                    {
                        OuterTile16(ai, lda, rows, bp, PanelWidth, ci, ldc, r);
                        jj = PanelWidth;
                    }
                    else if (width >= 8)
                        for (; jj + 16 <= valid; jj += 16) OuterTile8(ai, lda, rows, bp + jj, PanelWidth, ci + jj, ldc, r);
                    // The rest (all of it at width 1): contiguous panel rows, same summation order.
                    if (jj < valid)
                        for (int row = 0; row < rows; row++)
                        {
                            var dst = new Span<float>(ci + (long)row * ldc + jj, valid - jj);
                            for (int t = 0; t < r; t++)
                                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(bp + (long)t * PanelWidth + jj, valid - jj),
                                    ai[(long)row * lda + t], dst, dst);
                        }
                }
            }
        };
        if ((long)m * n * r < 1 << 18) { for (int task = 0; task < rowTiles * colTiles; task++) body(task); return; }
        For(rowTiles * colTiles, body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void OuterTile16(float* a, int lda, int rows, float* b, int ldb, float* c, int ldc, int r)
    {
        // Rows past `rows` repeat the last valid row (same inputs, same sums); their stores are skipped.
        float* a0 = a, a1 = rows > 1 ? a0 + lda : a0, a2 = rows > 2 ? a1 + lda : a1, a3 = rows > 3 ? a2 + lda : a2;
        float* c0 = c, c1 = rows > 1 ? c0 + ldc : c0, c2 = rows > 2 ? c1 + ldc : c1, c3 = rows > 3 ? c2 + ldc : c2;
        var s00 = Vector512.Load(c0); var s01 = Vector512.Load(c0 + 16); var s02 = Vector512.Load(c0 + 32); var s03 = Vector512.Load(c0 + 48);
        var s10 = Vector512.Load(c1); var s11 = Vector512.Load(c1 + 16); var s12 = Vector512.Load(c1 + 32); var s13 = Vector512.Load(c1 + 48);
        var s20 = Vector512.Load(c2); var s21 = Vector512.Load(c2 + 16); var s22 = Vector512.Load(c2 + 32); var s23 = Vector512.Load(c2 + 48);
        var s30 = Vector512.Load(c3); var s31 = Vector512.Load(c3 + 16); var s32 = Vector512.Load(c3 + 32); var s33 = Vector512.Load(c3 + 48);
        float* bt = b;
        for (int t = 0; t < r; t++, bt += ldb)
        {
            var y0 = Vector512.Load(bt); var y1 = Vector512.Load(bt + 16);
            var y2 = Vector512.Load(bt + 32); var y3 = Vector512.Load(bt + 48);
            var x = Vector512.Create(a0[t]);
            s00 = Vector512.FusedMultiplyAdd(x, y0, s00); s01 = Vector512.FusedMultiplyAdd(x, y1, s01);
            s02 = Vector512.FusedMultiplyAdd(x, y2, s02); s03 = Vector512.FusedMultiplyAdd(x, y3, s03);
            x = Vector512.Create(a1[t]);
            s10 = Vector512.FusedMultiplyAdd(x, y0, s10); s11 = Vector512.FusedMultiplyAdd(x, y1, s11);
            s12 = Vector512.FusedMultiplyAdd(x, y2, s12); s13 = Vector512.FusedMultiplyAdd(x, y3, s13);
            x = Vector512.Create(a2[t]);
            s20 = Vector512.FusedMultiplyAdd(x, y0, s20); s21 = Vector512.FusedMultiplyAdd(x, y1, s21);
            s22 = Vector512.FusedMultiplyAdd(x, y2, s22); s23 = Vector512.FusedMultiplyAdd(x, y3, s23);
            x = Vector512.Create(a3[t]);
            s30 = Vector512.FusedMultiplyAdd(x, y0, s30); s31 = Vector512.FusedMultiplyAdd(x, y1, s31);
            s32 = Vector512.FusedMultiplyAdd(x, y2, s32); s33 = Vector512.FusedMultiplyAdd(x, y3, s33);
        }
        if (rows > 3) { s30.Store(c3); s31.Store(c3 + 16); s32.Store(c3 + 32); s33.Store(c3 + 48); }
        if (rows > 2) { s20.Store(c2); s21.Store(c2 + 16); s22.Store(c2 + 32); s23.Store(c2 + 48); }
        if (rows > 1) { s10.Store(c1); s11.Store(c1 + 16); s12.Store(c1 + 32); s13.Store(c1 + 48); }
        s00.Store(c0); s01.Store(c0 + 16); s02.Store(c0 + 32); s03.Store(c0 + 48);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void OuterTile8(float* a, int lda, int rows, float* b, int ldb, float* c, int ldc, int r)
    {
        float* a0 = a, a1 = rows > 1 ? a0 + lda : a0, a2 = rows > 2 ? a1 + lda : a1, a3 = rows > 3 ? a2 + lda : a2;
        float* c0 = c, c1 = rows > 1 ? c0 + ldc : c0, c2 = rows > 2 ? c1 + ldc : c1, c3 = rows > 3 ? c2 + ldc : c2;
        var s00 = Vector256.Load(c0); var s01 = Vector256.Load(c0 + 8);
        var s10 = Vector256.Load(c1); var s11 = Vector256.Load(c1 + 8);
        var s20 = Vector256.Load(c2); var s21 = Vector256.Load(c2 + 8);
        var s30 = Vector256.Load(c3); var s31 = Vector256.Load(c3 + 8);
        float* bt = b;
        for (int t = 0; t < r; t++, bt += ldb)
        {
            var y0 = Vector256.Load(bt); var y1 = Vector256.Load(bt + 8);
            var x = Vector256.Create(a0[t]);
            s00 = Fma.MultiplyAdd(x, y0, s00); s01 = Fma.MultiplyAdd(x, y1, s01);
            x = Vector256.Create(a1[t]);
            s10 = Fma.MultiplyAdd(x, y0, s10); s11 = Fma.MultiplyAdd(x, y1, s11);
            x = Vector256.Create(a2[t]);
            s20 = Fma.MultiplyAdd(x, y0, s20); s21 = Fma.MultiplyAdd(x, y1, s21);
            x = Vector256.Create(a3[t]);
            s30 = Fma.MultiplyAdd(x, y0, s30); s31 = Fma.MultiplyAdd(x, y1, s31);
        }
        if (rows > 3) { s30.Store(c3); s31.Store(c3 + 8); }
        if (rows > 2) { s20.Store(c2); s21.Store(c2 + 8); }
        if (rows > 1) { s10.Store(c1); s11.Store(c1 + 8); }
        s00.Store(c0); s01.Store(c0 + 8);
    }
}
