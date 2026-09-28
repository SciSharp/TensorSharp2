// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// B operand of the packed GEMM read straight from a GGUF-quantized weight matrix: for a linear
// y = x W^T with W [out, in] stored as quantized rows, B[k, n] = W[n, k]. Each task dequantizes
// only the KC x NT tile it is about to multiply (block-aligned KC), so a weight element is
// decoded once per forward instead of once per activation row, and the F32 copy never exists.
// Q4_K and Q6_K (the Qwen3-VL-8B text encoder's types) decode with AVX2; every other type goes
// through ManagedQuantizedOps' scalar dequant. Both give the exact scalar-dequant floats, which
// reach the panel through 8x8 register transposes. Activations stay F32 (no 8-bit rounding),
// so a projection matches a double-precision reference to ~1e-6 (ManagedQuantizedOps' q8
// activation path: ~4e-3), at 3-6x its speed for a 37-token prompt.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Models.QwenImage
{
    internal readonly unsafe struct QuantRowsPanelSource : IGemmPanelSource
    {
        private const int QK_K = 256;
        private readonly nint _w;
        private readonly long _rowBytes;
        private readonly int _type, _rows, _blockElements, _blockBytes;

        internal QuantRowsPanelSource(IntPtr weights, int ggmlType, long ne0, long ne1)
        {
            _w = weights; _type = ggmlType; _rows = checked((int)ne1);
            var type = (GgmlTensorType)ggmlType;
            _blockElements = (int)GgufFile.GetBlockSize(type);
            _blockBytes = (int)GgufFile.GetTypeSize(type);
            _rowBytes = ManagedQuantizedOps.RowSize(ggmlType, ne0);
        }

        /// <summary>The weight can feed the GEMM: a managed-dequant type whose block tiles every
        /// K chunk (the driver's chunks are 256-element multiples when K is).</summary>
        internal static bool Supports(int ggmlType, long ne0)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!ManagedQuantizedOps.SupportsCpuQuantizedStorage(type) && type is not (GgmlTensorType.F32 or GgmlTensorType.F16 or GgmlTensorType.BF16))
                return false;
            long block = GgufFile.GetBlockSize(type);
            return block > 0 && QK_K % block == 0 && ne0 % QK_K == 0;
        }

        public float* Panels(int k0, int kc, int n0, int count, int nr, float* scratch, out long panelStride)
        {
            if (k0 % _blockElements != 0 || kc % _blockElements != 0)
                throw new InvalidOperationException($"K chunk [{k0}, +{kc}) is not aligned to {_blockElements}-element blocks.");
            panelStride = (long)kc * nr;
            // Eight weight rows at a time: dequantize them contiguously, then write them into the
            // [kc][nr] panel with 8x8 register transposes (one 32-byte store per 8 values) rather
            // than one strided scalar store per value.
            float* rows = stackalloc float[8 * kc];
            byte* blocks0 = (byte*)_w + (long)(k0 / _blockElements) * _blockBytes;
            for (int q = 0; q * nr < count; q++)
            {
                float* dst = scratch + q * panelStride;
                for (int g = 0; g < nr; g += 8)
                {
                    for (int r = 0; r < 8; r++)
                    {
                        int n = n0 + q * nr + g + r;
                        if (n < _rows) Dequantize(blocks0 + n * _rowBytes, rows + r * kc, kc);
                        else new Span<float>(rows + r * kc, kc).Clear();
                    }
                    if (Avx.IsSupported && kc % 8 == 0)
                        for (int t = 0; t < kc; t += 8) Transpose8(rows + t, kc, dst + (long)t * nr + g, nr);
                    else
                        for (int r = 0; r < 8; r++)
                            for (int t = 0; t < kc; t++) dst[(long)t * nr + g + r] = rows[r * kc + t];
                }
            }
            return scratch;
        }

        // dst[t * dstStride + r] = src[r * srcStride + t] for an 8x8 block.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Transpose8(float* src, int srcStride, float* dst, int dstStride)
        {
            var r0 = Avx.LoadVector256(src); var r1 = Avx.LoadVector256(src + srcStride);
            var r2 = Avx.LoadVector256(src + 2 * srcStride); var r3 = Avx.LoadVector256(src + 3 * srcStride);
            var r4 = Avx.LoadVector256(src + 4 * srcStride); var r5 = Avx.LoadVector256(src + 5 * srcStride);
            var r6 = Avx.LoadVector256(src + 6 * srcStride); var r7 = Avx.LoadVector256(src + 7 * srcStride);
            var t0 = Avx.UnpackLow(r0, r1); var t1 = Avx.UnpackHigh(r0, r1);
            var t2 = Avx.UnpackLow(r2, r3); var t3 = Avx.UnpackHigh(r2, r3);
            var t4 = Avx.UnpackLow(r4, r5); var t5 = Avx.UnpackHigh(r4, r5);
            var t6 = Avx.UnpackLow(r6, r7); var t7 = Avx.UnpackHigh(r6, r7);
            var u0 = Avx.Shuffle(t0, t2, 0x44); var u1 = Avx.Shuffle(t0, t2, 0xEE);
            var u2 = Avx.Shuffle(t1, t3, 0x44); var u3 = Avx.Shuffle(t1, t3, 0xEE);
            var u4 = Avx.Shuffle(t4, t6, 0x44); var u5 = Avx.Shuffle(t4, t6, 0xEE);
            var u6 = Avx.Shuffle(t5, t7, 0x44); var u7 = Avx.Shuffle(t5, t7, 0xEE);
            Avx.Store(dst, Avx.Permute2x128(u0, u4, 0x20));
            Avx.Store(dst + dstStride, Avx.Permute2x128(u1, u5, 0x20));
            Avx.Store(dst + 2 * dstStride, Avx.Permute2x128(u2, u6, 0x20));
            Avx.Store(dst + 3 * dstStride, Avx.Permute2x128(u3, u7, 0x20));
            Avx.Store(dst + 4 * dstStride, Avx.Permute2x128(u0, u4, 0x31));
            Avx.Store(dst + 5 * dstStride, Avx.Permute2x128(u1, u5, 0x31));
            Avx.Store(dst + 6 * dstStride, Avx.Permute2x128(u2, u6, 0x31));
            Avx.Store(dst + 7 * dstStride, Avx.Permute2x128(u3, u7, 0x31));
        }

        private void Dequantize(byte* src, float* dst, int count)
        {
            if (Avx2.IsSupported)
            {
                if (_type == (int)GgmlTensorType.Q4_K) { for (int b = 0; b < count; b += QK_K) DequantQ4K(src + b / QK_K * 144, dst + b); return; }
                if (_type == (int)GgmlTensorType.Q6_K) { for (int b = 0; b < count; b += QK_K) DequantQ6K(src + b / QK_K * 210, dst + b); return; }
            }
            ManagedQuantizedOps.DequantizeRowToFloat32(_type, (IntPtr)src, dst, count);
        }

        // y = (d*sc) * q - dmin*m per 32-value sub-block, as the scalar DequantizeQ4K.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DequantQ4K(byte* block, float* y)
        {
            float d = (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(block));
            float min = (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(block + 2));
            byte* scales = block + 4;
            byte* q = block + 16;
            var mask = Vector256.Create((byte)0x0F);
            for (int g = 0; g < 4; g++)
            {
                ScaleMin(2 * g, scales, out int sc1, out int m1);
                ScaleMin(2 * g + 1, scales, out int sc2, out int m2);
                var raw = Vector256.Load(q + 32 * g);
                Store32(raw & mask, y + 64 * g, d * sc1, min * m1);
                Store32(Vector256.ShiftRightLogical(raw.AsUInt16(), 4).AsByte() & mask, y + 64 * g + 32, d * sc2, min * m2);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ScaleMin(int j, byte* q, out int d, out int m)
        {
            if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; return; }
            d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }

        // 32 unsigned bytes -> scale * value - offset.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store32(Vector256<byte> v, float* y, float scale, float offset)
        {
            var vs = Vector256.Create(scale);
            var vo = Vector256.Create(offset);
            var (lo, hi) = Vector256.Widen(v);
            var (a, b) = Vector256.Widen(lo);
            var (c, e) = Vector256.Widen(hi);
            Vector256.Store(Vector256.ConvertToSingle(a.AsInt32()) * vs - vo, y);
            Vector256.Store(Vector256.ConvertToSingle(b.AsInt32()) * vs - vo, y + 8);
            Vector256.Store(Vector256.ConvertToSingle(c.AsInt32()) * vs - vo, y + 16);
            Vector256.Store(Vector256.ConvertToSingle(e.AsInt32()) * vs - vo, y + 24);
        }

        // y = (d*scale[is]) * (q - 32) per 16 values, as the scalar DequantizeQ6K.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DequantQ6K(byte* block, float* y)
        {
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* sc = (sbyte*)(block + 192);
            float d = (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(block + 208));
            var m4 = Vector256.Create((byte)0x0F);
            var m2 = Vector256.Create((byte)0x03);
            for (int n = 0; n < QK_K; n += 128)
            {
                var l0 = Vector256.Load(ql);
                var l1 = Vector256.Load(ql + 32);
                var h = Vector256.Load(qh);
                var q1 = (l0 & m4) | Vector256.ShiftLeft((h & m2).AsUInt16(), 4).AsByte();
                var q2 = (l1 & m4) | Vector256.ShiftLeft((Vector256.ShiftRightLogical(h.AsUInt16(), 2).AsByte() & m2).AsUInt16(), 4).AsByte();
                var q3 = (Vector256.ShiftRightLogical(l0.AsUInt16(), 4).AsByte() & m4)
                    | Vector256.ShiftLeft((Vector256.ShiftRightLogical(h.AsUInt16(), 4).AsByte() & m2).AsUInt16(), 4).AsByte();
                var q4 = (Vector256.ShiftRightLogical(l1.AsUInt16(), 4).AsByte() & m4)
                    | Vector256.ShiftLeft((Vector256.ShiftRightLogical(h.AsUInt16(), 6).AsByte() & m2).AsUInt16(), 4).AsByte();
                Store32Q6(q1, y + n, d * sc[0], d * sc[1]);
                Store32Q6(q2, y + n + 32, d * sc[2], d * sc[3]);
                Store32Q6(q3, y + n + 64, d * sc[4], d * sc[5]);
                Store32Q6(q4, y + n + 96, d * sc[6], d * sc[7]);
                ql += 64; qh += 32; sc += 8;
            }
        }

        // 32 six-bit values (0..63, stored +32): first 16 use scale0, last 16 scale1.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store32Q6(Vector256<byte> v, float* y, float scale0, float scale1)
        {
            var bias = Vector256.Create(32f);
            var s0 = Vector256.Create(scale0);
            var s1 = Vector256.Create(scale1);
            var (lo, hi) = Vector256.Widen(v);
            var (a, b) = Vector256.Widen(lo);
            var (c, e) = Vector256.Widen(hi);
            Vector256.Store(s0 * (Vector256.ConvertToSingle(a.AsInt32()) - bias), y);
            Vector256.Store(s0 * (Vector256.ConvertToSingle(b.AsInt32()) - bias), y + 8);
            Vector256.Store(s1 * (Vector256.ConvertToSingle(c.AsInt32()) - bias), y + 16);
            Vector256.Store(s1 * (Vector256.ConvertToSingle(e.AsInt32()) - bias), y + 24);
        }
    }
}
