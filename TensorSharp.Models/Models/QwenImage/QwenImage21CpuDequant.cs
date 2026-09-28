// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage;

/// <summary>
/// Row dequantizers for the pure-C# transformer's F32 projections
/// (<see cref="QwenImage21CpuKernels.GemmDequantNT"/>), vectorized for the types the
/// Qwen-Image-2.1 GGUFs use: Q4_K and Q6_K (Q4_K_M), Q8_0, BF16 and F16. Every value is
/// computed with the same operations in the same order as ggml's dequantize_row_* (and
/// <see cref="ManagedQuantizedOps"/>), so the result is bit-identical to the scalar code;
/// other types fall back to it.
/// </summary>
/// <remarks>At a few hundred activation rows the dequantization of every weight row is a
/// tenth of a projection when done a value at a time; 16 lanes make it negligible, and they
/// are what lets a 2-row time-embedding projection take the F32 path too.</remarks>
internal static unsafe class QwenImage21CpuDequant
{
    private const int QK = 256, Q4KBytes = 144, Q6KBytes = 210, Q8_0Bytes = 34;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Row(int width, int ggmlType, byte* src, float* dst, int n)
    {
        var type = (GgmlTensorType)ggmlType;
        if (width >= 8)
        {
            switch (type)
            {
                case GgmlTensorType.Q4_K when n % QK == 0:
                    for (int b = 0; b < n / QK; b++) Q4K(width, src + b * Q4KBytes, dst + b * QK);
                    return;
                case GgmlTensorType.Q6_K when n % QK == 0:
                    for (int b = 0; b < n / QK; b++) Q6K(width, src + b * Q6KBytes, dst + b * QK);
                    return;
                case GgmlTensorType.Q8_0 when n % 32 == 0:
                    for (int b = 0; b < n / 32; b++) Q8_0(width, src + b * Q8_0Bytes, dst + b * 32);
                    return;
                case GgmlTensorType.BF16:
                    Bf16(width, (ushort*)src, dst, n);
                    return;
                case GgmlTensorType.F16:
                    TensorPrimitives.ConvertToSingle(new ReadOnlySpan<System.Half>(src, n), new Span<float>(dst, n));
                    return;
                case GgmlTensorType.F32:
                    Buffer.MemoryCopy(src, dst, n * 4L, n * 4L);
                    return;
            }
        }
        ManagedQuantizedOps.DequantizeRowToFloat32(ggmlType, (IntPtr)src, dst, n);
    }

    /// <summary>16 unsigned bytes widened to floats: two halves of 8 on AVX2, one on AVX-512.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store16(int width, Vector128<byte> v, float scale, float offset, float* y)
    {
        // y = scale * v - offset, with the product rounded before the subtraction as ggml does.
        if (width == 16)
        {
            var f = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(v));
            (f * Vector512.Create(scale) - Vector512.Create(offset)).Store(y);
        }
        else
        {
            var lo = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(v));
            var hi = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(v, 8)));
            (lo * Vector256.Create(scale) - Vector256.Create(offset)).Store(y);
            (hi * Vector256.Create(scale) - Vector256.Create(offset)).Store(y + 8);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GetScaleMinK4(int j, byte* q, out int scale, out int min)
    {
        if (j < 4)
        {
            scale = q[j] & 63;
            min = q[j + 4] & 63;
        }
        else
        {
            scale = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            min = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }

    /// <summary>ggml dequantize_row_q4_K: y = d * scale * q - dmin * min per 32-value sub-block.</summary>
    private static void Q4K(int width, byte* block, float* y)
    {
        float d = (float)*(System.Half*)block, dmin = (float)*(System.Half*)(block + 2);
        byte* scales = block + 4, q = block + 16;
        var mask = Vector128.Create((byte)0x0F);
        for (int j = 0, sub = 0; j < QK; j += 64, sub += 2, q += 32)
        {
            GetScaleMinK4(sub, scales, out int sc1, out int m1);
            GetScaleMinK4(sub + 1, scales, out int sc2, out int m2);
            float d1 = d * sc1, min1 = dmin * m1, d2 = d * sc2, min2 = dmin * m2;
            for (int l = 0; l < 32; l += 16)
            {
                var bytes = Vector128.Load(q + l);
                Store16(width, bytes & mask, d1, min1, y + j + l);
                Store16(width, Vector128.ShiftRightLogical(bytes, 4), d2, min2, y + j + 32 + l);
            }
        }
    }

    /// <summary>ggml dequantize_row_q6_K: y = d * scale * (q - 32), q = 4 low bits | 2 high bits.</summary>
    private static void Q6K(int width, byte* block, float* y)
    {
        byte* ql = block, qh = block + 128;
        sbyte* sc = (sbyte*)(block + 192);
        float d = (float)*(System.Half*)(block + 208);
        var low = Vector128.Create((byte)0x0F);
        var two = Vector128.Create((byte)0x03);
        for (int n = 0; n < QK; n += 128, y += 128, ql += 64, qh += 32, sc += 8)
            for (int l = 0; l < 32; l += 16)
            {
                int s = l / 16;
                var a = Vector128.Load(ql + l);
                var b = Vector128.Load(ql + l + 32);
                var h = Vector128.Load(qh + l);
                // The -32 zero point is folded into the offset: d * sc * (q - 32) computed as
                // (d * sc) * q - (d * sc) * 32 would round differently, so subtract it as an
                // integer before converting, exactly as the scalar code does.
                Q6Store(width, (a & low) | ((h & two) << 4), d * sc[s], y + l);
                Q6Store(width, (b & low) | ((Vector128.ShiftRightLogical(h, 2) & two) << 4), d * sc[s + 2], y + l + 32);
                Q6Store(width, Vector128.ShiftRightLogical(a, 4) | ((Vector128.ShiftRightLogical(h, 4) & two) << 4), d * sc[s + 4], y + l + 64);
                Q6Store(width, Vector128.ShiftRightLogical(b, 4) | (Vector128.ShiftRightLogical(h, 6) << 4), d * sc[s + 6], y + l + 96);
            }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q6Store(int width, Vector128<byte> q, float scale, float* y)
    {
        if (width == 16)
        {
            var i = Avx512F.ConvertToVector512Int32(q) - Vector512.Create(32);
            (Avx512F.ConvertToVector512Single(i) * Vector512.Create(scale)).Store(y);
        }
        else
        {
            var lo = Avx2.ConvertToVector256Int32(q) - Vector256.Create(32);
            var hi = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(q, 8)) - Vector256.Create(32);
            (Avx.ConvertToVector256Single(lo) * Vector256.Create(scale)).Store(y);
            (Avx.ConvertToVector256Single(hi) * Vector256.Create(scale)).Store(y + 8);
        }
    }

    /// <summary>ggml dequantize_row_q8_0: y = d * q.</summary>
    private static void Q8_0(int width, byte* block, float* y)
    {
        float d = (float)*(System.Half*)block;
        sbyte* q = (sbyte*)(block + 2);
        for (int l = 0; l < 32; l += 16)
        {
            var v = Vector128.Load(q + l);
            if (width == 16)
                (Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(v)) * Vector512.Create(d)).Store(y + l);
            else
            {
                (Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(v)) * Vector256.Create(d)).Store(y + l);
                (Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(v, 8))) * Vector256.Create(d)).Store(y + l + 8);
            }
        }
    }

    /// <summary>BF16 is the upper half of an F32: widen and shift.</summary>
    private static void Bf16(int width, ushort* src, float* dst, int n)
    {
        int i = 0;
        if (width == 16)
            for (; i + 16 <= n; i += 16)
                Avx512F.ShiftLeftLogical(Avx512F.ConvertToVector512UInt32(Vector256.Load(src + i)), 16).AsSingle().Store(dst + i);
        else
            for (; i + 8 <= n; i += 8)
                Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(Vector128.Load(src + i)), 16).AsSingle().Store(dst + i);
        for (; i < n; i++) dst[i] = BitConverter.Int32BitsToSingle(src[i] << 16);
    }
}
