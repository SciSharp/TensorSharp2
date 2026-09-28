// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // Column-pair decoders for the multi-row GEMM.
    //
    // A pair of weight rows A, B is expanded into one scratch where every
    // 32-element chunk is a 64-byte line [A: 32 values | B: 32 values]. The
    // AVX-512 kernel uses the line as one zmm (two output columns per
    // vpmaddubsw); the AVX2 kernel reads either half. Everything that the
    // per-row kernels recompute on every dot - nibble/bit unpacking, 6-bit
    // scale unpacking, fp16 -> fp32 scales, the broadcast int16 scale vectors -
    // is done here once per (pair, row block).
    //
    // K-quant layout, per 256-element super-block (QKSbStride bytes):
    //   +0     values : 8 chunks x 64 B, unsigned (Q4_K 0..15, Q5_K 0..31, Q6_K 0..63)
    //   +512   scales : 8 chunks x 64 B, int16 multiplier of each maddubs lane
    //   +1024  d      : [A: d x8 | B: d x8] floats
    //   +1088  aux    : Q4_K/Q5_K [A: dmin*min_0..7 | B: ...] floats
    //                   Q6_K      [A: 32*scale_0..15 | B: ...] int16
    // Q4_0/Q5_0/Q8_0 layout: blocks in groups of four (Q0GrpStride bytes),
    //   +64*i  values : block i, [A: 32 | B: 32] (unsigned for Q4_0/Q5_0, signed for Q8_0)
    //   +256   d      : [A: d0 d1 d2 d3 d0 d1 d2 d3 | B: ...] floats, the lane order
    //                   the kernels' four-block reduction produces
    // followed by the inDim/32 % 4 leftover blocks (Q0BlkStride bytes each):
    //   +0     values : [A: 32 | B: 32]
    //   +64    d      : [A: d x8 | B: d x8] floats
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        private const int QKSbStride = 1152;
        private const int Q0BlkStride = 128;
        private const int Q0GrpStride = 4 * 64 + 64;

        /// <summary>Scratch bytes of a decoded Q4_0/Q5_0/Q8_0 pair.</summary>
        private static int Q0PairScratchBytes(int nb) => (nb >> 2) * Q0GrpStride + (nb & 3) * Q0BlkStride;

        /// <summary>Store one decoded block (values + fp32 scale) of one column
        /// half at its place in the grouped Q0 layout.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StoreQ0Block(byte* dst, int b, int nb, Vector256<byte> values, float d)
        {
            int grouped = nb & ~3;
            if (b < grouped)
            {
                byte* g = dst + (b >> 2) * Q0GrpStride;
                values.Store(g + (b & 3) * 64);
                float* dp = (float*)(g + 256) + (b & 3);
                dp[0] = d;
                dp[4] = d;
            }
            else
            {
                byte* l = dst + (grouped >> 2) * Q0GrpStride + (b - grouped) * Q0BlkStride;
                values.Store(l);
                Vector256.Create(d).Store((float*)(l + 64));
            }
        }

        private static unsafe void DecodeQGemmPair(GgmlTensorType type, byte* rowA, byte* rowB, int inDim, byte* dst)
        {
            switch (type)
            {
                case GgmlTensorType.Q4_K:
                    DecodeQ4KHalf(rowA, inDim / QK_K, dst);
                    DecodeQ4KHalf(rowB, inDim / QK_K, dst + 32);
                    break;
                case GgmlTensorType.Q5_K:
                    DecodeQ5KHalf(rowA, inDim / QK_K, dst);
                    DecodeQ5KHalf(rowB, inDim / QK_K, dst + 32);
                    break;
                case GgmlTensorType.Q6_K:
                    DecodeQ6KHalf(rowA, inDim / QK_K, dst);
                    DecodeQ6KHalf(rowB, inDim / QK_K, dst + 32);
                    break;
                case GgmlTensorType.Q4_0:
                    DecodeQ40Half(rowA, inDim / QK4_0, dst);
                    DecodeQ40Half(rowB, inDim / QK4_0, dst + 32);
                    break;
                case GgmlTensorType.Q5_0:
                    DecodeQ50Half(rowA, inDim / QK5_0, dst);
                    DecodeQ50Half(rowB, inDim / QK5_0, dst + 32);
                    break;
                case GgmlTensorType.Q8_0:
                    DecodeQ80Half(rowA, inDim / QK8_0, dst);
                    DecodeQ80Half(rowB, inDim / QK8_0, dst + 32);
                    break;
                default:
                    throw new NotSupportedException($"No GEMM decoder for {type}.");
            }
        }

        /// <summary>
        /// The eight 6-bit scales and mins of a Q4_K/Q5_K super-block as two
        /// little-endian byte octets (ggml's kmask unpacking; equivalent to
        /// <see cref="GetScaleMinK4"/> for j = 0..7).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void UnpackKScales(byte* packed, out ulong scales, out ulong mins)
        {
            const uint kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
            uint u0 = ReadUInt32(packed), u1 = ReadUInt32(packed + 4), u2 = ReadUInt32(packed + 8);
            uint u3 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
            uint uaux = u1 & kmask1;
            u1 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
            u2 = uaux;
            u0 &= kmask1;
            scales = u0 | ((ulong)u1 << 32);
            mins = u2 | ((ulong)u3 << 32);
        }

        /// <summary>Eight unsigned bytes (little-endian in a ulong) as floats.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> BytesToFloat8(ulong bytes)
            => Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.CreateScalar(bytes).AsByte()));

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DecodeQ4KHalf(byte* row, int nsb, byte* dst)
        {
            Vector256<byte> m4 = Vector256.Create((byte)0x0F);
            for (int sb = 0; sb < nsb; sb++)
            {
                byte* blk = row + sb * Q4_KBlockBytes;
                byte* o = dst + sb * QKSbStride;
                float d = HalfToSingle(ReadUInt16(blk));
                float dmin = HalfToSingle(ReadUInt16(blk + 2));
                UnpackKScales(blk + 4, out ulong sc, out ulong mn);
                byte* qs = blk + 16;
                for (int p = 0; p < 4; p++)
                {
                    Vector256<byte> q = Vector256.Load(qs + p * 32);
                    (q & m4).Store(o + (2 * p) * 64);
                    (Vector256.ShiftRightLogical(q.AsUInt16(), 4).AsByte() & m4).Store(o + (2 * p + 1) * 64);
                }
                for (int j = 0; j < 8; j++)
                    Vector256.Create((short)(byte)(sc >> (8 * j))).Store((short*)(o + 512 + j * 64));
                Vector256.Create(d).Store((float*)(o + 1024));
                (BytesToFloat8(mn) * Vector256.Create(dmin)).Store((float*)(o + 1088));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DecodeQ5KHalf(byte* row, int nsb, byte* dst)
        {
            Vector256<byte> m4 = Vector256.Create((byte)0x0F);
            Vector256<byte> bit4 = Vector256.Create((byte)0x10);
            for (int sb = 0; sb < nsb; sb++)
            {
                byte* blk = row + sb * Q5_KBlockBytes;
                byte* o = dst + sb * QKSbStride;
                float d = HalfToSingle(ReadUInt16(blk));
                float dmin = HalfToSingle(ReadUInt16(blk + 2));
                UnpackKScales(blk + 4, out ulong sc, out ulong mn);
                Vector256<byte> qh = Vector256.Load(blk + 16);
                byte* qs = blk + 48;
                for (int p = 0; p < 4; p++)
                {
                    Vector256<byte> q = Vector256.Load(qs + p * 32);
                    // Sub-block j takes its 5th bit from bit j of qh[l].
                    Vector256<byte> bLo = Vector256.Create((byte)(1 << (2 * p)));
                    Vector256<byte> bHi = Vector256.Create((byte)(2 << (2 * p)));
                    Vector256<byte> hLo = Vector256.Equals(qh & bLo, bLo) & bit4;
                    Vector256<byte> hHi = Vector256.Equals(qh & bHi, bHi) & bit4;
                    ((q & m4) | hLo).Store(o + (2 * p) * 64);
                    ((Vector256.ShiftRightLogical(q.AsUInt16(), 4).AsByte() & m4) | hHi).Store(o + (2 * p + 1) * 64);
                }
                for (int j = 0; j < 8; j++)
                    Vector256.Create((short)(byte)(sc >> (8 * j))).Store((short*)(o + 512 + j * 64));
                Vector256.Create(d).Store((float*)(o + 1024));
                (BytesToFloat8(mn) * Vector256.Create(dmin)).Store((float*)(o + 1088));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DecodeQ6KHalf(byte* row, int nsb, byte* dst)
        {
            Vector256<byte> m4 = Vector256.Create((byte)0x0F);
            Vector256<byte> m30 = Vector256.Create((byte)0x30);
            for (int sb = 0; sb < nsb; sb++)
            {
                byte* blk = row + sb * Q6_KBlockBytes;
                byte* o = dst + sb * QKSbStride;
                byte* ql = blk;
                byte* qh = blk + QK_K / 2;
                sbyte* scales = (sbyte*)(blk + QK_K / 2 + QK_K / 4);
                float d = HalfToSingle(ReadUInt16(blk + QK_K / 2 + QK_K / 4 + QK_K / 16));

                // Chunk 4h+k of half h: low nibble / high nibble of ql[64h (+32)],
                // high two bits = bits 2k..2k+1 of qh[32h + l] (see DequantizeQ6K).
                // The 16-bit shifts only move bits across a byte boundary into
                // positions the 0x30 mask clears.
                for (int h = 0; h < 2; h++)
                {
                    Vector256<byte> ql0 = Vector256.Load(ql + h * 64);
                    Vector256<byte> ql1 = Vector256.Load(ql + h * 64 + 32);
                    Vector256<ushort> qhv = Vector256.Load(qh + h * 32).AsUInt16();
                    byte* c = o + h * 4 * 64;
                    ((ql0 & m4) | (Vector256.ShiftLeft(qhv, 4).AsByte() & m30)).Store(c);
                    ((ql1 & m4) | (Vector256.ShiftLeft(qhv, 2).AsByte() & m30)).Store(c + 64);
                    ((Vector256.ShiftRightLogical(ql0.AsUInt16(), 4).AsByte() & m4) | (qhv.AsByte() & m30)).Store(c + 128);
                    ((Vector256.ShiftRightLogical(ql1.AsUInt16(), 4).AsByte() & m4)
                        | (Vector256.ShiftRightLogical(qhv, 2).AsByte() & m30)).Store(c + 192);
                }
                // Chunk k covers sub-blocks 2k (maddubs lanes 0..7) and 2k+1 (8..15).
                for (int k = 0; k < 8; k++)
                    Vector256.Create(Vector128.Create((short)scales[2 * k]), Vector128.Create((short)scales[2 * k + 1]))
                        .Store((short*)(o + 512 + k * 64));
                Vector256.Create(d).Store((float*)(o + 1024));
                Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int16(Vector128.Load(scales)), 5).Store((short*)(o + 1088));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DecodeQ40Half(byte* row, int nb, byte* dst)
        {
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            for (int b = 0; b < nb; b++)
            {
                byte* blk = row + b * Q4_0BlockBytes;
                Vector128<byte> q = Vector128.Load(blk + 2);
                StoreQ0Block(dst, b, nb,
                    Vector256.Create(q & m4, Vector128.ShiftRightLogical(q.AsUInt16(), 4).AsByte() & m4),
                    HalfToSingle(ReadUInt16(blk)));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DecodeQ50Half(byte* row, int nb, byte* dst)
        {
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector256<byte> bit4 = Vector256.Create((byte)0x10);
            // Byte e of the spread picks qh byte e/8; the mask then isolates bit e%8.
            Vector256<byte> spread = Vector256.Create(
                (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
            Vector256<byte> bits = Vector256.Create(
                (byte)1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
            for (int b = 0; b < nb; b++)
            {
                byte* blk = row + b * Q5_0BlockBytes;
                Vector128<byte> q = Vector128.Load(blk + 6);
                Vector256<byte> v = Vector256.Create(q & m4, Vector128.ShiftRightLogical(q.AsUInt16(), 4).AsByte() & m4);
                Vector256<byte> qh = Avx2.Shuffle(Vector256.Create(ReadUInt32(blk + 2)).AsByte(), spread);
                v |= Vector256.Equals(qh & bits, bits) & bit4;
                StoreQ0Block(dst, b, nb, v, HalfToSingle(ReadUInt16(blk)));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DecodeQ80Half(byte* row, int nb, byte* dst)
        {
            for (int b = 0; b < nb; b++)
            {
                byte* blk = row + b * Q8_0BlockBytes;
                StoreQ0Block(dst, b, nb, Vector256.Load(blk + 2), HalfToSingle(ReadUInt16(blk)));
            }
        }
    }
}
