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
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // Float GEMM for weights without an int8 kernel (F16/BF16/F32 and the
    // quant types that only have a dequantizer).
    //
    // DequantMatMulColumns dequantizes one weight row and dots it with four
    // activation rows at a time, which re-reads each activation panel once per
    // output column and keeps one weight vector per four FMAs. Here four weight
    // rows are dequantized into an L2-resident F32 panel once per row block,
    // and a 4x4 (AVX-512) or 2x4 (AVX2) register tile of dot products runs over
    // it - four FMAs per activation load and four per weight load. Activations
    // stay F32, so this is the same math as before (F32 weights x F32
    // activations), only summed in a different order.
    //
    // Dequantization is always the managed implementation here (bit-exact with
    // ggml's), so the pure-C# backend never reaches native code on this path.
    // A/B: TS_CPU_FGEMM=0 (or TS_CPU_QGEMM=0) restores DequantMatMulColumns.
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        private static readonly bool FGemmEnabled =
            Environment.GetEnvironmentVariable("TS_CPU_FGEMM") != "0";

        private const int FGemmCols = 4;

        [ThreadStatic] private static float[] _fgemmScratch;

        private static unsafe float* FGemmScratch(int floats)
        {
            float[] buf = _fgemmScratch;
            if (buf == null || buf.Length < floats + 16)
            {
                buf = GC.AllocateUninitializedArray<float>(Math.Max(floats + 16, 16 * 1024), pinned: true);
                _fgemmScratch = buf;
            }
            float* p = (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buf));
            return (float*)(((nint)p + 63) & ~(nint)63);
        }

        /// <summary>
        /// Float-panel GEMM over any dequantizable weight type. Returns false when
        /// the path is disabled or the ISA is missing (the caller then runs
        /// <see cref="DequantMatMulColumns"/>).
        /// </summary>
        internal static unsafe bool TryFloatPanelGemm(
            int ggmlType, byte* weights, long rowBytes, int inDim, int outDim,
            float* input, int inputRowStride, int rowCount, float* output, int outputRowStride,
            ParallelOptions options, QGemmIsa isa)
        {
            if (isa == QGemmIsa.Auto && !FGemmEnabled)
                return false;
            isa = ResolveQGemmIsa(isa);
            if (isa == QGemmIsa.Legacy || rowCount <= 0 || outDim <= 0 || inDim <= 0)
                return false;
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                return false;

            bool avx512 = isa == QGemmIsa.Avx512;
            int dop = ResolveDop(options);
            int groups = (outDim + FGemmCols - 1) / FGemmCols;

            // Row blocks keep the activation panel in L2 next to the 4-row weight panel.
            int rowsPerBlock = rowCount;
            long blockRows = QGemmRowBlockBytes / Math.Max(1L, (long)inDim * sizeof(float));
            if (blockRows < rowCount)
                rowsPerBlock = (int)Math.Max(4, blockRows & ~3L);
            int rowBlocks = (rowCount + rowsPerBlock - 1) / rowsPerBlock;
            long macs = (long)rowCount * outDim * inDim;
            long wantTasks = QGemmWantTasks(macs, rowBytes * outDim, rowCount, dop);
            long colBlocks = Math.Clamp((wantTasks + rowBlocks - 1) / rowBlocks, 1, groups);
            int groupsPerBlock = (int)((groups + colBlocks - 1) / colBlocks);
            int colBlockCount = (groups + groupsPerBlock - 1) / groupsPerBlock;

            nint w = (nint)weights, x = (nint)input, o = (nint)output;
            int inStride = inputRowStride, outStride = outputRowStride;
            void RunTask(int t)
            {
                int cb = t % colBlockCount;
                int rb = t / colBlockCount;
                int g0 = cb * groupsPerBlock, g1 = Math.Min(groups, g0 + groupsPerBlock);
                int r0 = rb * rowsPerBlock, r1 = Math.Min(rowCount, r0 + rowsPerBlock);
                float* panel = FGemmScratch(FGemmCols * inDim);
                int tileRows = avx512 ? 4 : 2;
                for (int g = g0; g < g1; g++)
                {
                    int c0 = g * FGemmCols;
                    int cols = Math.Min(FGemmCols, outDim - c0);
                    for (int c = 0; c < cols; c++)
                        DequantizeToFloat32(type, (byte*)w + (c0 + c) * rowBytes, panel + (long)c * inDim, inDim);
                    for (int r = r0; r < r1; r += tileRows)
                    {
                        int rows = Math.Min(tileRows, r1 - r);
                        float* xr = (float*)x + (long)r * inStride;
                        float* orow = (float*)o + (long)r * outStride + c0;
                        if (avx512)
                            FGemm512(rows, cols, panel, inDim, xr, inStride, orow, outStride);
                        else
                            FGemm256(rows, cols, panel, inDim, xr, inStride, orow, outStride);
                    }
                }
            }

            int tasks = rowBlocks * colBlockCount;
            if (tasks > 1 && dop > 1)
                RunParallelBlocks(tasks, options, RunTask);
            else
                for (int t = 0; t < tasks; t++) RunTask(t);
            return true;
        }

        private static unsafe void FGemm512(int rows, int cols, float* panel, int k, float* x, int xStride, float* o, int oStride)
        {
            switch (rows)
            {
                case 1: FGemm512Rows<QRows1>(cols, panel, k, x, xStride, o, oStride); break;
                case 2: FGemm512Rows<QRows2>(cols, panel, k, x, xStride, o, oStride); break;
                case 3: FGemm512Rows<QRows3>(cols, panel, k, x, xStride, o, oStride); break;
                default: FGemm512Rows<QRows4>(cols, panel, k, x, xStride, o, oStride); break;
            }
        }

        private static unsafe void FGemm512Rows<TR>(int cols, float* panel, int k, float* x, int xStride, float* o, int oStride)
            where TR : struct, IQGemmRows
        {
            switch (cols)
            {
                case 1: FGemmTile512<TR, QRows1>(panel, k, x, xStride, o, oStride); break;
                case 2: FGemmTile512<TR, QRows2>(panel, k, x, xStride, o, oStride); break;
                case 3: FGemmTile512<TR, QRows3>(panel, k, x, xStride, o, oStride); break;
                default: FGemmTile512<TR, QRows4>(panel, k, x, xStride, o, oStride); break;
            }
        }

        private static unsafe void FGemm256(int rows, int cols, float* panel, int k, float* x, int xStride, float* o, int oStride)
        {
            if (rows == 1)
            {
                switch (cols)
                {
                    case 1: FGemmTile256<QRows1, QRows1>(panel, k, x, xStride, o, oStride); break;
                    case 2: FGemmTile256<QRows1, QRows2>(panel, k, x, xStride, o, oStride); break;
                    case 3: FGemmTile256<QRows1, QRows3>(panel, k, x, xStride, o, oStride); break;
                    default: FGemmTile256<QRows1, QRows4>(panel, k, x, xStride, o, oStride); break;
                }
                return;
            }
            switch (cols)
            {
                case 1: FGemmTile256<QRows2, QRows1>(panel, k, x, xStride, o, oStride); break;
                case 2: FGemmTile256<QRows2, QRows2>(panel, k, x, xStride, o, oStride); break;
                case 3: FGemmTile256<QRows2, QRows3>(panel, k, x, xStride, o, oStride); break;
                default: FGemmTile256<QRows2, QRows4>(panel, k, x, xStride, o, oStride); break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HSum512(Vector512<float> v)
            => HSum256(Avx.Add(v.GetLower(), v.GetUpper()));

        /// <summary>Scalar tail of one (row, column) dot, for inDim % 16.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotTail(float* a, float* b, int start, int end)
        {
            float s = 0f;
            for (int i = start; i < end; i++) s += a[i] * b[i];
            return s;
        }

        // TR = activation rows (1..4), TC = weight columns (1..4) of the tile.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void FGemmTile512<TR, TC>(float* panel, int k, float* x, int xStride, float* o, int oStride)
            where TR : struct, IQGemmRows
            where TC : struct, IQGemmRows
        {
            int rows = TR.Count, cols = TC.Count;
            float* w0 = panel, w1 = panel + k, w2 = panel + 2 * k, w3 = panel + 3 * k;
            float* x0 = x, x1 = x + xStride, x2 = x + 2 * (long)xStride, x3 = x + 3 * (long)xStride;
            Vector512<float> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector512<float> a10 = default, a11 = default, a12 = default, a13 = default;
            Vector512<float> a20 = default, a21 = default, a22 = default, a23 = default;
            Vector512<float> a30 = default, a31 = default, a32 = default, a33 = default;
            int i = 0;
            for (; i <= k - 16; i += 16)
            {
                Vector512<float> v0 = Vector512.Load(w0 + i);
                Vector512<float> v1 = cols > 1 ? Vector512.Load(w1 + i) : default;
                Vector512<float> v2 = cols > 2 ? Vector512.Load(w2 + i) : default;
                Vector512<float> v3 = cols > 3 ? Vector512.Load(w3 + i) : default;

                Vector512<float> xv = Vector512.Load(x0 + i);
                a00 = Avx512F.FusedMultiplyAdd(xv, v0, a00);
                if (cols > 1) a01 = Avx512F.FusedMultiplyAdd(xv, v1, a01);
                if (cols > 2) a02 = Avx512F.FusedMultiplyAdd(xv, v2, a02);
                if (cols > 3) a03 = Avx512F.FusedMultiplyAdd(xv, v3, a03);
                if (rows > 1)
                {
                    xv = Vector512.Load(x1 + i);
                    a10 = Avx512F.FusedMultiplyAdd(xv, v0, a10);
                    if (cols > 1) a11 = Avx512F.FusedMultiplyAdd(xv, v1, a11);
                    if (cols > 2) a12 = Avx512F.FusedMultiplyAdd(xv, v2, a12);
                    if (cols > 3) a13 = Avx512F.FusedMultiplyAdd(xv, v3, a13);
                }
                if (rows > 2)
                {
                    xv = Vector512.Load(x2 + i);
                    a20 = Avx512F.FusedMultiplyAdd(xv, v0, a20);
                    if (cols > 1) a21 = Avx512F.FusedMultiplyAdd(xv, v1, a21);
                    if (cols > 2) a22 = Avx512F.FusedMultiplyAdd(xv, v2, a22);
                    if (cols > 3) a23 = Avx512F.FusedMultiplyAdd(xv, v3, a23);
                }
                if (rows > 3)
                {
                    xv = Vector512.Load(x3 + i);
                    a30 = Avx512F.FusedMultiplyAdd(xv, v0, a30);
                    if (cols > 1) a31 = Avx512F.FusedMultiplyAdd(xv, v1, a31);
                    if (cols > 2) a32 = Avx512F.FusedMultiplyAdd(xv, v2, a32);
                    if (cols > 3) a33 = Avx512F.FusedMultiplyAdd(xv, v3, a33);
                }
            }

            o[0] = HSum512(a00) + DotTail(x0, w0, i, k);
            if (cols > 1) o[1] = HSum512(a01) + DotTail(x0, w1, i, k);
            if (cols > 2) o[2] = HSum512(a02) + DotTail(x0, w2, i, k);
            if (cols > 3) o[3] = HSum512(a03) + DotTail(x0, w3, i, k);
            if (rows > 1)
            {
                float* r = o + oStride;
                r[0] = HSum512(a10) + DotTail(x1, w0, i, k);
                if (cols > 1) r[1] = HSum512(a11) + DotTail(x1, w1, i, k);
                if (cols > 2) r[2] = HSum512(a12) + DotTail(x1, w2, i, k);
                if (cols > 3) r[3] = HSum512(a13) + DotTail(x1, w3, i, k);
            }
            if (rows > 2)
            {
                float* r = o + 2 * (long)oStride;
                r[0] = HSum512(a20) + DotTail(x2, w0, i, k);
                if (cols > 1) r[1] = HSum512(a21) + DotTail(x2, w1, i, k);
                if (cols > 2) r[2] = HSum512(a22) + DotTail(x2, w2, i, k);
                if (cols > 3) r[3] = HSum512(a23) + DotTail(x2, w3, i, k);
            }
            if (rows > 3)
            {
                float* r = o + 3 * (long)oStride;
                r[0] = HSum512(a30) + DotTail(x3, w0, i, k);
                if (cols > 1) r[1] = HSum512(a31) + DotTail(x3, w1, i, k);
                if (cols > 2) r[2] = HSum512(a32) + DotTail(x3, w2, i, k);
                if (cols > 3) r[3] = HSum512(a33) + DotTail(x3, w3, i, k);
            }
        }

        // AVX2: 16 ymm registers, so 2 rows x 4 columns (8 accumulators).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void FGemmTile256<TR, TC>(float* panel, int k, float* x, int xStride, float* o, int oStride)
            where TR : struct, IQGemmRows
            where TC : struct, IQGemmRows
        {
            int rows = TR.Count, cols = TC.Count;
            float* w0 = panel, w1 = panel + k, w2 = panel + 2 * k, w3 = panel + 3 * k;
            float* x0 = x, x1 = x + xStride;
            Vector256<float> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector256<float> a10 = default, a11 = default, a12 = default, a13 = default;
            int i = 0;
            for (; i <= k - 8; i += 8)
            {
                Vector256<float> v0 = Vector256.Load(w0 + i);
                Vector256<float> v1 = cols > 1 ? Vector256.Load(w1 + i) : default;
                Vector256<float> v2 = cols > 2 ? Vector256.Load(w2 + i) : default;
                Vector256<float> v3 = cols > 3 ? Vector256.Load(w3 + i) : default;

                Vector256<float> xv = Vector256.Load(x0 + i);
                a00 = Fma.MultiplyAdd(xv, v0, a00);
                if (cols > 1) a01 = Fma.MultiplyAdd(xv, v1, a01);
                if (cols > 2) a02 = Fma.MultiplyAdd(xv, v2, a02);
                if (cols > 3) a03 = Fma.MultiplyAdd(xv, v3, a03);
                if (rows > 1)
                {
                    xv = Vector256.Load(x1 + i);
                    a10 = Fma.MultiplyAdd(xv, v0, a10);
                    if (cols > 1) a11 = Fma.MultiplyAdd(xv, v1, a11);
                    if (cols > 2) a12 = Fma.MultiplyAdd(xv, v2, a12);
                    if (cols > 3) a13 = Fma.MultiplyAdd(xv, v3, a13);
                }
            }

            o[0] = HSum256(a00) + DotTail(x0, w0, i, k);
            if (cols > 1) o[1] = HSum256(a01) + DotTail(x0, w1, i, k);
            if (cols > 2) o[2] = HSum256(a02) + DotTail(x0, w2, i, k);
            if (cols > 3) o[3] = HSum256(a03) + DotTail(x0, w3, i, k);
            if (rows > 1)
            {
                float* r = o + oStride;
                r[0] = HSum256(a10) + DotTail(x1, w0, i, k);
                if (cols > 1) r[1] = HSum256(a11) + DotTail(x1, w1, i, k);
                if (cols > 2) r[2] = HSum256(a12) + DotTail(x1, w2, i, k);
                if (cols > 3) r[3] = HSum256(a13) + DotTail(x1, w3, i, k);
            }
        }
    }
}
