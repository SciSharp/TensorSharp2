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
    // GEMM microkernels over a decoded column pair (layout in
    // ManagedQuantGemm.Decode.cs) and a tile of GEMM-layout activation rows.
    //
    // The row count of a tile is a generic struct parameter, so the JIT
    // compiles one body per tile height with the unused rows' accumulators and
    // loads folded away - the accumulators stay in registers without eight
    // hand-written copies of every kernel.
    //
    // AVX-512 (8 rows x 2 columns): a zmm holds 32 values of column A and 32 of
    // column B; the activation chunk is broadcast to both halves with a
    // load-only vbroadcasti32x8, so each vpmaddubsw produces both columns and no
    // shuffle is needed. On client cores (one 512-bit multiply port) this runs
    // at the same int8 rate as two 256-bit ports; the win over AVX2 is the 32
    // registers, which is what lets 8 rows x 2 columns share one weight load.
    //
    // AVX2 (4 rows x 1 column, 16 ymm registers): each column of the pair is a
    // separate pass over the same (L1-hot) activation rows.
    //
    // Integer sums per sub-block are exact; each super-block (K-quants) or
    // block (Q4_0/Q5_0/Q8_0) is scaled by its own float scales exactly as the
    // per-row kernels do, only summed in a different order.
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        private interface IQGemmRows { static abstract int Count { get; } }
        private struct QRows1 : IQGemmRows { public static int Count => 1; }
        private struct QRows2 : IQGemmRows { public static int Count => 2; }
        private struct QRows3 : IQGemmRows { public static int Count => 3; }
        private struct QRows4 : IQGemmRows { public static int Count => 4; }
        private struct QRows5 : IQGemmRows { public static int Count => 5; }
        private struct QRows6 : IQGemmRows { public static int Count => 6; }
        private struct QRows7 : IQGemmRows { public static int Count => 7; }
        private struct QRows8 : IQGemmRows { public static int Count => 8; }

        /// <summary>Compile-time flag: K family = Q6_K offset (true) vs
        /// Q4_K/Q5_K min (false); Q0 family = signed Q8_0 (true) vs zero-point
        /// Q4_0/Q5_0 (false).</summary>
        private interface IQGemmFlag { static abstract bool On { get; } }
        private struct QFlagOff : IQGemmFlag { public static bool On => false; }
        private struct QFlagOn : IQGemmFlag { public static bool On => true; }

        private static unsafe void RunQGemmKernel(
            QGemmFamily family, bool avx512, int rows, byte* w, byte* act, int actStride, int inDim,
            float* outp, int outStride, bool second)
        {
            if (avx512)
            {
                switch (family)
                {
                    case QGemmFamily.KMin: QGemmK512Rows<QFlagOff>(rows, w, act, actStride, inDim, outp, outStride, second); break;
                    case QGemmFamily.KOfs: QGemmK512Rows<QFlagOn>(rows, w, act, actStride, inDim, outp, outStride, second); break;
                    case QGemmFamily.Q0Unsigned: QGemmQ0512Rows<QFlagOff>(rows, w, act, actStride, inDim, outp, outStride, second); break;
                    case QGemmFamily.Q0Signed: QGemmQ0512Rows<QFlagOn>(rows, w, act, actStride, inDim, outp, outStride, second); break;
                    default: throw new NotSupportedException(family.ToString());
                }
                return;
            }

            // AVX2: one column per pass; the second pass finds the rows in L1.
            for (int half = 0; half < (second ? 2 : 1); half++)
            {
                byte* wh = w + half * 32;
                float* oh = outp + half;
                switch (family)
                {
                    case QGemmFamily.KMin: QGemmK256Rows<QFlagOff>(rows, wh, act, actStride, inDim, oh, outStride); break;
                    case QGemmFamily.KOfs: QGemmK256Rows<QFlagOn>(rows, wh, act, actStride, inDim, oh, outStride); break;
                    case QGemmFamily.Q0Unsigned: QGemmQ0256Rows<QFlagOff>(rows, wh, act, actStride, inDim, oh, outStride); break;
                    case QGemmFamily.Q0Signed: QGemmQ0256Rows<QFlagOn>(rows, wh, act, actStride, inDim, oh, outStride); break;
                    default: throw new NotSupportedException(family.ToString());
                }
            }
        }

        private static unsafe void QGemmK512Rows<TF>(int rows, byte* w, byte* act, int actStride, int inDim, float* outp, int outStride, bool second)
            where TF : struct, IQGemmFlag
        {
            switch (rows)
            {
                case 1: QGemmK512<QRows1, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 2: QGemmK512<QRows2, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 3: QGemmK512<QRows3, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 4: QGemmK512<QRows4, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 5: QGemmK512<QRows5, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 6: QGemmK512<QRows6, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 7: QGemmK512<QRows7, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                default: QGemmK512<QRows8, TF>(w, act, actStride, inDim, outp, outStride, second); break;
            }
        }

        private static unsafe void QGemmQ0512Rows<TF>(int rows, byte* w, byte* act, int actStride, int inDim, float* outp, int outStride, bool second)
            where TF : struct, IQGemmFlag
        {
            switch (rows)
            {
                case 1: QGemmQ0512<QRows1, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 2: QGemmQ0512<QRows2, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 3: QGemmQ0512<QRows3, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 4: QGemmQ0512<QRows4, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 5: QGemmQ0512<QRows5, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 6: QGemmQ0512<QRows6, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                case 7: QGemmQ0512<QRows7, TF>(w, act, actStride, inDim, outp, outStride, second); break;
                default: QGemmQ0512<QRows8, TF>(w, act, actStride, inDim, outp, outStride, second); break;
            }
        }

        private static unsafe void QGemmK256Rows<TF>(int rows, byte* w, byte* act, int actStride, int inDim, float* outp, int outStride)
            where TF : struct, IQGemmFlag
        {
            switch (rows)
            {
                case 1: QGemmK256<QRows1, TF>(w, act, actStride, inDim, outp, outStride); break;
                case 2: QGemmK256<QRows2, TF>(w, act, actStride, inDim, outp, outStride); break;
                case 3: QGemmK256<QRows3, TF>(w, act, actStride, inDim, outp, outStride); break;
                default: QGemmK256<QRows4, TF>(w, act, actStride, inDim, outp, outStride); break;
            }
        }

        private static unsafe void QGemmQ0256Rows<TF>(int rows, byte* w, byte* act, int actStride, int inDim, float* outp, int outStride)
            where TF : struct, IQGemmFlag
        {
            switch (rows)
            {
                case 1: QGemmQ0256<QRows1, TF>(w, act, actStride, inDim, outp, outStride); break;
                case 2: QGemmQ0256<QRows2, TF>(w, act, actStride, inDim, outp, outStride); break;
                case 3: QGemmQ0256<QRows3, TF>(w, act, actStride, inDim, outp, outStride); break;
                default: QGemmQ0256<QRows4, TF>(w, act, actStride, inDim, outp, outStride); break;
            }
        }

        // ---- AVX-512 helpers ----------------------------------------------

        // 32 bytes into both 256-bit halves (a pure load uop). This must be the
        // AVX512DQ vbroadcasti32x8/vbroadcastf32x8 form: .NET 10.0.8 mis-encodes
        // the EVEX compressed displacement of the AVX512F 64x4 forms
        // (Avx512F.BroadcastVector256ToVector512(long*/double*)), so [p + 32]
        // loads from p + 64. The Q0 kernels address blocks at constant offsets
        // (x + 32/64/96) and read the wrong activations through it.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<sbyte> Bcast256x2(byte* p)
            => Avx512DQ.BroadcastVector256ToVector512((int*)p).AsSByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float> Bcast256x2F(byte* p)
            => Avx512DQ.BroadcastVector256ToVector512((float*)p);

        /// <summary>acc += maddwd(maddubs(w, x), scale): one 32-element chunk of
        /// both columns, the int16 products weighted by their sub-block scale.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<int> KStep512(Vector512<int> acc, Vector512<byte> w, Vector512<short> scale, byte* x)
            => Avx512F.Add(acc, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(w, Bcast256x2(x)), scale));

        /// <summary>Q6_K: subtract 32 * sum(scale * bsum), exactly, in the integer domain.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<int> KOfs512(Vector512<int> acc, Vector512<short> ofs, byte* bsums)
            => Avx512F.Subtract(acc, Avx512BW.MultiplyAddAdjacent(Bcast256x2(bsums).AsInt16(), ofs));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float> KScale512(Vector512<float> acc, Vector512<int> isum, Vector512<float> d, byte* d8)
            => Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single(isum),
                Avx512F.Multiply(d, Vector512.Create(*(float*)d8)), acc);

        /// <summary>Q4_K/Q5_K: acc -= (dmin * min_j) * (d8 * bsum_j), eight
        /// sub-blocks per column half.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float> KMin512(Vector512<float> acc, Vector512<float> mins, byte* bsF)
            => Avx512F.FusedMultiplyAddNegated(mins, Bcast256x2F(bsF), acc);

        /// <summary>Horizontal sums of the two 8-lane halves (columns A and B).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StorePair(Vector512<float> acc, float* o, bool second)
        {
            Vector256<float> h = Avx.HorizontalAdd(acc.GetLower(), acc.GetUpper());
            Vector128<float> s = Sse.Add(h.GetLower(), h.GetUpper());
            s = Sse3.HorizontalAdd(s, s);
            o[0] = s.ToScalar();
            if (second)
                o[1] = s.GetElement(1);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QGemmK512<TR, TOfs>(byte* w, byte* act, int actStride, int inDim, float* outp, int outStride, bool second)
            where TR : struct, IQGemmRows
            where TOfs : struct, IQGemmFlag
        {
            int rows = TR.Count;
            int nsb = inDim / QK_K;
            byte* a0 = act;
            byte* a1 = act + actStride;
            byte* a2 = act + 2 * actStride;
            byte* a3 = act + 3 * actStride;
            byte* a4 = act + 4 * actStride;
            byte* a5 = act + 5 * actStride;
            byte* a6 = act + 6 * actStride;
            byte* a7 = act + 7 * actStride;
            Vector512<float> f0 = default, f1 = default, f2 = default, f3 = default;
            Vector512<float> f4 = default, f5 = default, f6 = default, f7 = default;

            for (int sb = 0; sb < nsb; sb++)
            {
                byte* ws = w + sb * QKSbStride;
                Vector512<int> i0 = default, i1 = default, i2 = default, i3 = default;
                Vector512<int> i4 = default, i5 = default, i6 = default, i7 = default;
                for (int j = 0; j < 8; j++)
                {
                    Vector512<byte> wv = Vector512.Load(ws + j * 64);
                    Vector512<short> sv = Vector512.Load((short*)(ws + 512 + j * 64));
                    int o = sb * QK_K + j * 32;
                    i0 = KStep512(i0, wv, sv, a0 + o);
                    if (rows > 1) i1 = KStep512(i1, wv, sv, a1 + o);
                    if (rows > 2) i2 = KStep512(i2, wv, sv, a2 + o);
                    if (rows > 3) i3 = KStep512(i3, wv, sv, a3 + o);
                    if (rows > 4) i4 = KStep512(i4, wv, sv, a4 + o);
                    if (rows > 5) i5 = KStep512(i5, wv, sv, a5 + o);
                    if (rows > 6) i6 = KStep512(i6, wv, sv, a6 + o);
                    if (rows > 7) i7 = KStep512(i7, wv, sv, a7 + o);
                }

                int ao = inDim + sb * 32;
                if (TOfs.On)
                {
                    Vector512<short> ov = Vector512.Load((short*)(ws + 1088));
                    i0 = KOfs512(i0, ov, a0 + ao);
                    if (rows > 1) i1 = KOfs512(i1, ov, a1 + ao);
                    if (rows > 2) i2 = KOfs512(i2, ov, a2 + ao);
                    if (rows > 3) i3 = KOfs512(i3, ov, a3 + ao);
                    if (rows > 4) i4 = KOfs512(i4, ov, a4 + ao);
                    if (rows > 5) i5 = KOfs512(i5, ov, a5 + ao);
                    if (rows > 6) i6 = KOfs512(i6, ov, a6 + ao);
                    if (rows > 7) i7 = KOfs512(i7, ov, a7 + ao);
                }

                Vector512<float> dv = Vector512.Load((float*)(ws + 1024));
                int d8o = inDim + nsb * 32 + sb * 4;
                f0 = KScale512(f0, i0, dv, a0 + d8o);
                if (rows > 1) f1 = KScale512(f1, i1, dv, a1 + d8o);
                if (rows > 2) f2 = KScale512(f2, i2, dv, a2 + d8o);
                if (rows > 3) f3 = KScale512(f3, i3, dv, a3 + d8o);
                if (rows > 4) f4 = KScale512(f4, i4, dv, a4 + d8o);
                if (rows > 5) f5 = KScale512(f5, i5, dv, a5 + d8o);
                if (rows > 6) f6 = KScale512(f6, i6, dv, a6 + d8o);
                if (rows > 7) f7 = KScale512(f7, i7, dv, a7 + d8o);

                if (!TOfs.On)
                {
                    Vector512<float> mv = Vector512.Load((float*)(ws + 1088));
                    f0 = KMin512(f0, mv, a0 + ao);
                    if (rows > 1) f1 = KMin512(f1, mv, a1 + ao);
                    if (rows > 2) f2 = KMin512(f2, mv, a2 + ao);
                    if (rows > 3) f3 = KMin512(f3, mv, a3 + ao);
                    if (rows > 4) f4 = KMin512(f4, mv, a4 + ao);
                    if (rows > 5) f5 = KMin512(f5, mv, a5 + ao);
                    if (rows > 6) f6 = KMin512(f6, mv, a6 + ao);
                    if (rows > 7) f7 = KMin512(f7, mv, a7 + ao);
                }
            }

            StorePair(f0, outp, second);
            if (rows > 1) StorePair(f1, outp + outStride, second);
            if (rows > 2) StorePair(f2, outp + 2 * (long)outStride, second);
            if (rows > 3) StorePair(f3, outp + 3 * (long)outStride, second);
            if (rows > 4) StorePair(f4, outp + 4 * (long)outStride, second);
            if (rows > 5) StorePair(f5, outp + 5 * (long)outStride, second);
            if (rows > 6) StorePair(f6, outp + 6 * (long)outStride, second);
            if (rows > 7) StorePair(f7, outp + 7 * (long)outStride, second);
        }

        // ---- Q4_0 / Q5_0 / Q8_0 (AVX-512) ------------------------------------
        //
        // These carry a float scale per 32-element block (weight fp16 d times
        // activation fp16 d), so unlike the K-quants the integer sums cannot run
        // over a super-block. Converting and scaling every block costs three
        // multiply-port uops (cvt, mul, fma) on top of the two integer ones. Four
        // consecutive blocks' int32 partials are therefore first folded into one
        // vector on the shuffle port (vpunpck[lh]qdq + add, then vshufps + add),
        // leaving one lane per (column half, block): [b0 b1 b2 b3] in every
        // 128-bit lane. One cvt/mul/fma then scales four blocks, against a scale
        // vector laid out the same way (decoder: [d0 d1 d2 d3] x2 per column;
        // activation: its four contiguous dx values, a vbroadcastf32x4 load).

        /// <summary>Integer dot of one block of both columns with one row:
        /// 16 int32 lanes [A: 8 | B: 8], Q8_0's sign moved onto x.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<int> Q0Dot512<TSigned>(Vector512<byte> w, Vector512<sbyte> neg, Vector512<short> ones, byte* x)
            where TSigned : struct, IQGemmFlag
        {
            Vector512<sbyte> xv = Bcast256x2(x);
            // Signed weights: vpmaddubsw needs an unsigned left operand, so the
            // kernel multiplies |w| by x with w's sign moved onto x
            // ((x ^ m) - m negates where m = -1). AVX-512BW has no vpsignb.
            if (TSigned.On)
                xv = Avx512BW.Subtract(Avx512F.Xor(xv, neg), neg);
            return Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(w, xv), ones);
        }

        /// <summary>Fold four blocks' [A: 8 | B: 8] partials into [b0 b1 b2 b3]
        /// per 128-bit lane (lanes 0-1 column A, 2-3 column B).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<int> Reduce4Blocks512(Vector512<int> i0, Vector512<int> i1, Vector512<int> i2, Vector512<int> i3)
        {
            Vector512<int> u01 = Avx512F.Add(
                Avx512F.UnpackLow(i0.AsInt64(), i1.AsInt64()).AsInt32(),
                Avx512F.UnpackHigh(i0.AsInt64(), i1.AsInt64()).AsInt32());   // [b0 b0 b1 b1]
            Vector512<int> u23 = Avx512F.Add(
                Avx512F.UnpackLow(i2.AsInt64(), i3.AsInt64()).AsInt32(),
                Avx512F.UnpackHigh(i2.AsInt64(), i3.AsInt64()).AsInt32());   // [b2 b2 b3 b3]
            Vector512<float> a = u01.AsSingle(), b = u23.AsSingle();
            return Avx512F.Add(Avx512F.Shuffle(a, b, 0x88).AsInt32(), Avx512F.Shuffle(a, b, 0xDD).AsInt32());
        }

        /// <summary>Four blocks of both columns for one row, scaled and accumulated.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float> Q0Group512<TSigned>(
            Vector512<float> acc,
            Vector512<byte> w0, Vector512<byte> w1, Vector512<byte> w2, Vector512<byte> w3,
            Vector512<sbyte> n0, Vector512<sbyte> n1, Vector512<sbyte> n2, Vector512<sbyte> n3,
            Vector512<float> dw, Vector512<short> ones, byte* x, byte* dx, byte* sx)
            where TSigned : struct, IQGemmFlag
        {
            Vector512<int> v = Reduce4Blocks512(
                Q0Dot512<TSigned>(w0, n0, ones, x), Q0Dot512<TSigned>(w1, n1, ones, x + 32),
                Q0Dot512<TSigned>(w2, n2, ones, x + 64), Q0Dot512<TSigned>(w3, n3, ones, x + 96));
            // Zero point: two lanes per (column, block) remain, each takes half.
            if (!TSigned.On)
                v = Avx512F.Subtract(v, Avx512F.BroadcastVector128ToVector512((int*)sx));
            return Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single(v),
                Avx512F.Multiply(dw, Avx512F.BroadcastVector128ToVector512((float*)dx)), acc);
        }

        /// <summary>One leftover block (inDim/32 % 4) of both columns for one row.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float> Q0Step512<TSigned>(
            Vector512<float> acc, Vector512<byte> w, Vector512<sbyte> neg, Vector512<float> dw, Vector512<short> ones,
            byte* x, byte* dx, byte* sx)
            where TSigned : struct, IQGemmFlag
        {
            Vector512<int> isum = Q0Dot512<TSigned>(w, neg, ones, x);
            // Eight lanes per (column, block): a quarter of the stored half share.
            if (!TSigned.On)
                isum = Avx512F.Subtract(isum, Vector512.Create(*(int*)sx >> 2));
            return Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single(isum),
                Avx512F.Multiply(dw, Vector512.Create(*(float*)dx)), acc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<byte> LoadQ0Weights512<TSigned>(byte* p, out Vector512<sbyte> neg)
            where TSigned : struct, IQGemmFlag
        {
            Vector512<byte> w = Vector512.Load(p);
            if (!TSigned.On)
            {
                neg = default;
                return w;
            }
            neg = Vector512.LessThan(w.AsSByte(), Vector512<sbyte>.Zero);
            return Vector512.Abs(w.AsSByte()).AsByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QGemmQ0512<TR, TSigned>(byte* w, byte* act, int actStride, int inDim, float* outp, int outStride, bool second)
            where TR : struct, IQGemmRows
            where TSigned : struct, IQGemmFlag
        {
            int rows = TR.Count;
            int nb = inDim / QK8_0;
            int groups = nb >> 2;
            byte* a0 = act;
            byte* a1 = act + actStride;
            byte* a2 = act + 2 * actStride;
            byte* a3 = act + 3 * actStride;
            byte* a4 = act + 4 * actStride;
            byte* a5 = act + 5 * actStride;
            byte* a6 = act + 6 * actStride;
            byte* a7 = act + 7 * actStride;
            int sxBase = inDim + nb * 4;
            Vector512<short> ones = Vector512.Create((short)1);
            Vector512<float> f0 = default, f1 = default, f2 = default, f3 = default;
            Vector512<float> f4 = default, f5 = default, f6 = default, f7 = default;

            for (int g = 0; g < groups; g++)
            {
                byte* wg = w + g * Q0GrpStride;
                Vector512<byte> w0 = LoadQ0Weights512<TSigned>(wg, out Vector512<sbyte> n0);
                Vector512<byte> w1 = LoadQ0Weights512<TSigned>(wg + 64, out Vector512<sbyte> n1);
                Vector512<byte> w2 = LoadQ0Weights512<TSigned>(wg + 128, out Vector512<sbyte> n2);
                Vector512<byte> w3 = LoadQ0Weights512<TSigned>(wg + 192, out Vector512<sbyte> n3);
                Vector512<float> dw = Vector512.Load((float*)(wg + 256));
                int qo = g * 4 * QK8_0;
                int xo = inDim + g * 16;
                int so = sxBase + g * 16;
                f0 = Q0Group512<TSigned>(f0, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a0 + qo, a0 + xo, a0 + so);
                if (rows > 1) f1 = Q0Group512<TSigned>(f1, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a1 + qo, a1 + xo, a1 + so);
                if (rows > 2) f2 = Q0Group512<TSigned>(f2, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a2 + qo, a2 + xo, a2 + so);
                if (rows > 3) f3 = Q0Group512<TSigned>(f3, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a3 + qo, a3 + xo, a3 + so);
                if (rows > 4) f4 = Q0Group512<TSigned>(f4, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a4 + qo, a4 + xo, a4 + so);
                if (rows > 5) f5 = Q0Group512<TSigned>(f5, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a5 + qo, a5 + xo, a5 + so);
                if (rows > 6) f6 = Q0Group512<TSigned>(f6, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a6 + qo, a6 + xo, a6 + so);
                if (rows > 7) f7 = Q0Group512<TSigned>(f7, w0, w1, w2, w3, n0, n1, n2, n3, dw, ones, a7 + qo, a7 + xo, a7 + so);
            }

            byte* wl = w + groups * Q0GrpStride;
            for (int b = groups * 4; b < nb; b++)
            {
                byte* wb = wl + (b - groups * 4) * Q0BlkStride;
                Vector512<byte> wv = LoadQ0Weights512<TSigned>(wb, out Vector512<sbyte> neg);
                Vector512<float> dw = Vector512.Load((float*)(wb + 64));
                int qo = b * QK8_0;
                int xo = inDim + b * 4;
                int so = sxBase + b * 4;
                f0 = Q0Step512<TSigned>(f0, wv, neg, dw, ones, a0 + qo, a0 + xo, a0 + so);
                if (rows > 1) f1 = Q0Step512<TSigned>(f1, wv, neg, dw, ones, a1 + qo, a1 + xo, a1 + so);
                if (rows > 2) f2 = Q0Step512<TSigned>(f2, wv, neg, dw, ones, a2 + qo, a2 + xo, a2 + so);
                if (rows > 3) f3 = Q0Step512<TSigned>(f3, wv, neg, dw, ones, a3 + qo, a3 + xo, a3 + so);
                if (rows > 4) f4 = Q0Step512<TSigned>(f4, wv, neg, dw, ones, a4 + qo, a4 + xo, a4 + so);
                if (rows > 5) f5 = Q0Step512<TSigned>(f5, wv, neg, dw, ones, a5 + qo, a5 + xo, a5 + so);
                if (rows > 6) f6 = Q0Step512<TSigned>(f6, wv, neg, dw, ones, a6 + qo, a6 + xo, a6 + so);
                if (rows > 7) f7 = Q0Step512<TSigned>(f7, wv, neg, dw, ones, a7 + qo, a7 + xo, a7 + so);
            }

            StorePair(f0, outp, second);
            if (rows > 1) StorePair(f1, outp + outStride, second);
            if (rows > 2) StorePair(f2, outp + 2 * (long)outStride, second);
            if (rows > 3) StorePair(f3, outp + 3 * (long)outStride, second);
            if (rows > 4) StorePair(f4, outp + 4 * (long)outStride, second);
            if (rows > 5) StorePair(f5, outp + 5 * (long)outStride, second);
            if (rows > 6) StorePair(f6, outp + 6 * (long)outStride, second);
            if (rows > 7) StorePair(f7, outp + 7 * (long)outStride, second);
        }

        // ---- AVX2 kernels (one column: w points at the pair's A or B half) ----

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<int> KStep256(Vector256<int> acc, Vector256<byte> w, Vector256<short> scale, byte* x)
            => Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(w, Vector256.Load((sbyte*)x)), scale));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<int> KOfs256(Vector256<int> acc, Vector256<short> ofs, byte* bsums)
            => Avx2.Subtract(acc, Avx2.MultiplyAddAdjacent(Vector256.Load((short*)bsums), ofs));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<float> KScale256(Vector256<float> acc, Vector256<int> isum, Vector256<float> d, byte* d8)
            => Fma.MultiplyAdd(Avx.ConvertToVector256Single(isum), Avx.Multiply(d, Vector256.Create(*(float*)d8)), acc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<float> KMin256(Vector256<float> acc, Vector256<float> mins, byte* bsF)
            => Fma.MultiplyAddNegated(mins, Vector256.Load((float*)bsF), acc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HSum256(Vector256<float> v)
        {
            Vector128<float> s = Sse.Add(v.GetLower(), v.GetUpper());
            s = Sse3.HorizontalAdd(s, s);
            s = Sse3.HorizontalAdd(s, s);
            return s.ToScalar();
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QGemmK256<TR, TOfs>(byte* w, byte* act, int actStride, int inDim, float* outp, int outStride)
            where TR : struct, IQGemmRows
            where TOfs : struct, IQGemmFlag
        {
            int rows = TR.Count;
            int nsb = inDim / QK_K;
            byte* a0 = act;
            byte* a1 = act + actStride;
            byte* a2 = act + 2 * actStride;
            byte* a3 = act + 3 * actStride;
            Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;

            for (int sb = 0; sb < nsb; sb++)
            {
                byte* ws = w + sb * QKSbStride;
                Vector256<int> i0 = default, i1 = default, i2 = default, i3 = default;
                for (int j = 0; j < 8; j++)
                {
                    Vector256<byte> wv = Vector256.Load(ws + j * 64);
                    Vector256<short> sv = Vector256.Load((short*)(ws + 512 + j * 64));
                    int o = sb * QK_K + j * 32;
                    i0 = KStep256(i0, wv, sv, a0 + o);
                    if (rows > 1) i1 = KStep256(i1, wv, sv, a1 + o);
                    if (rows > 2) i2 = KStep256(i2, wv, sv, a2 + o);
                    if (rows > 3) i3 = KStep256(i3, wv, sv, a3 + o);
                }

                int ao = inDim + sb * 32;
                if (TOfs.On)
                {
                    Vector256<short> ov = Vector256.Load((short*)(ws + 1088));
                    i0 = KOfs256(i0, ov, a0 + ao);
                    if (rows > 1) i1 = KOfs256(i1, ov, a1 + ao);
                    if (rows > 2) i2 = KOfs256(i2, ov, a2 + ao);
                    if (rows > 3) i3 = KOfs256(i3, ov, a3 + ao);
                }

                Vector256<float> dv = Vector256.Load((float*)(ws + 1024));
                int d8o = inDim + nsb * 32 + sb * 4;
                f0 = KScale256(f0, i0, dv, a0 + d8o);
                if (rows > 1) f1 = KScale256(f1, i1, dv, a1 + d8o);
                if (rows > 2) f2 = KScale256(f2, i2, dv, a2 + d8o);
                if (rows > 3) f3 = KScale256(f3, i3, dv, a3 + d8o);

                if (!TOfs.On)
                {
                    Vector256<float> mv = Vector256.Load((float*)(ws + 1088));
                    f0 = KMin256(f0, mv, a0 + ao);
                    if (rows > 1) f1 = KMin256(f1, mv, a1 + ao);
                    if (rows > 2) f2 = KMin256(f2, mv, a2 + ao);
                    if (rows > 3) f3 = KMin256(f3, mv, a3 + ao);
                }
            }

            outp[0] = HSum256(f0);
            if (rows > 1) outp[outStride] = HSum256(f1);
            if (rows > 2) outp[2 * (long)outStride] = HSum256(f2);
            if (rows > 3) outp[3 * (long)outStride] = HSum256(f3);
        }

        /// <summary>Integer dot of one block of one column with one row (8 int32
        /// lanes); Q8_0's sign moved onto x with vpsignb.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<int> Q0Dot256<TSigned>(Vector256<sbyte> w, Vector256<short> ones, byte* x)
            where TSigned : struct, IQGemmFlag
        {
            Vector256<sbyte> xv = Vector256.Load((sbyte*)x);
            Vector256<byte> wu;
            if (TSigned.On)
            {
                xv = Avx2.Sign(xv, w);
                wu = Avx2.Abs(w);
            }
            else
            {
                wu = w.AsByte();
            }
            return Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(wu, xv), ones);
        }

        /// <summary>AVX2 form of <see cref="Reduce4Blocks512"/>: [b0 b1 b2 b3] per 128-bit lane.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> Reduce4Blocks256(Vector256<int> i0, Vector256<int> i1, Vector256<int> i2, Vector256<int> i3)
        {
            Vector256<int> u01 = Avx2.Add(
                Avx2.UnpackLow(i0.AsInt64(), i1.AsInt64()).AsInt32(),
                Avx2.UnpackHigh(i0.AsInt64(), i1.AsInt64()).AsInt32());
            Vector256<int> u23 = Avx2.Add(
                Avx2.UnpackLow(i2.AsInt64(), i3.AsInt64()).AsInt32(),
                Avx2.UnpackHigh(i2.AsInt64(), i3.AsInt64()).AsInt32());
            Vector256<float> a = u01.AsSingle(), b = u23.AsSingle();
            return Avx2.Add(Avx.Shuffle(a, b, 0x88).AsInt32(), Avx.Shuffle(a, b, 0xDD).AsInt32());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<float> Q0Group256<TSigned>(
            Vector256<float> acc, Vector256<sbyte> w0, Vector256<sbyte> w1, Vector256<sbyte> w2, Vector256<sbyte> w3,
            Vector256<float> dw, Vector256<short> ones, byte* x, byte* dx, byte* sx)
            where TSigned : struct, IQGemmFlag
        {
            Vector256<int> v = Reduce4Blocks256(
                Q0Dot256<TSigned>(w0, ones, x), Q0Dot256<TSigned>(w1, ones, x + 32),
                Q0Dot256<TSigned>(w2, ones, x + 64), Q0Dot256<TSigned>(w3, ones, x + 96));
            if (!TSigned.On)
                v = Avx2.Subtract(v, Avx2.BroadcastVector128ToVector256((int*)sx));
            return Fma.MultiplyAdd(Avx.ConvertToVector256Single(v),
                Avx.Multiply(dw, Avx.BroadcastVector128ToVector256((float*)dx)), acc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<float> Q0Step256<TSigned>(
            Vector256<float> acc, Vector256<sbyte> w, Vector256<float> dw, Vector256<short> ones,
            byte* x, byte* dx, byte* sx)
            where TSigned : struct, IQGemmFlag
        {
            Vector256<int> isum = Q0Dot256<TSigned>(w, ones, x);
            if (!TSigned.On)
                isum = Avx2.Subtract(isum, Vector256.Create(*(int*)sx >> 2));
            return Fma.MultiplyAdd(Avx.ConvertToVector256Single(isum), Avx.Multiply(dw, Vector256.Create(*(float*)dx)), acc);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QGemmQ0256<TR, TSigned>(byte* w, byte* act, int actStride, int inDim, float* outp, int outStride)
            where TR : struct, IQGemmRows
            where TSigned : struct, IQGemmFlag
        {
            int rows = TR.Count;
            int nb = inDim / QK8_0;
            int groups = nb >> 2;
            byte* a0 = act;
            byte* a1 = act + actStride;
            byte* a2 = act + 2 * actStride;
            byte* a3 = act + 3 * actStride;
            int sxBase = inDim + nb * 4;
            Vector256<short> ones = Vector256.Create((short)1);
            Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;

            for (int g = 0; g < groups; g++)
            {
                byte* wg = w + g * Q0GrpStride;
                Vector256<sbyte> w0 = Vector256.Load((sbyte*)wg);
                Vector256<sbyte> w1 = Vector256.Load((sbyte*)(wg + 64));
                Vector256<sbyte> w2 = Vector256.Load((sbyte*)(wg + 128));
                Vector256<sbyte> w3 = Vector256.Load((sbyte*)(wg + 192));
                Vector256<float> dw = Vector256.Load((float*)(wg + 256));
                int qo = g * 4 * QK8_0;
                int xo = inDim + g * 16;
                int so = sxBase + g * 16;
                f0 = Q0Group256<TSigned>(f0, w0, w1, w2, w3, dw, ones, a0 + qo, a0 + xo, a0 + so);
                if (rows > 1) f1 = Q0Group256<TSigned>(f1, w0, w1, w2, w3, dw, ones, a1 + qo, a1 + xo, a1 + so);
                if (rows > 2) f2 = Q0Group256<TSigned>(f2, w0, w1, w2, w3, dw, ones, a2 + qo, a2 + xo, a2 + so);
                if (rows > 3) f3 = Q0Group256<TSigned>(f3, w0, w1, w2, w3, dw, ones, a3 + qo, a3 + xo, a3 + so);
            }

            byte* wl = w + groups * Q0GrpStride;
            for (int b = groups * 4; b < nb; b++)
            {
                byte* wb = wl + (b - groups * 4) * Q0BlkStride;
                Vector256<sbyte> wv = Vector256.Load((sbyte*)wb);
                Vector256<float> dw = Vector256.Load((float*)(wb + 64));
                int qo = b * QK8_0;
                int xo = inDim + b * 4;
                int so = sxBase + b * 4;
                f0 = Q0Step256<TSigned>(f0, wv, dw, ones, a0 + qo, a0 + xo, a0 + so);
                if (rows > 1) f1 = Q0Step256<TSigned>(f1, wv, dw, ones, a1 + qo, a1 + xo, a1 + so);
                if (rows > 2) f2 = Q0Step256<TSigned>(f2, wv, dw, ones, a2 + qo, a2 + xo, a2 + so);
                if (rows > 3) f3 = Q0Step256<TSigned>(f3, wv, dw, ones, a3 + qo, a3 + xo, a3 + so);
            }

            outp[0] = HSum256(f0);
            if (rows > 1) outp[outStride] = HSum256(f1);
            if (rows > 2) outp[2 * (long)outStride] = HSum256(f2);
            if (rows > 3) outp[3 * (long)outStride] = HSum256(f3);
        }
    }
}
