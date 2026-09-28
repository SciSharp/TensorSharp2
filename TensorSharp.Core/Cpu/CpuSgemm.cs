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
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Packed, cache-blocked F32 GEMM for the pure-C# CPU backend:
    /// <c>C = alpha * op(A) * op(B) + beta * C</c> over arbitrary element strides.
    ///
    /// Layout: A(i,p) = a[i*aRowStride + p*aColStride], B(p,j) = b[p*bRowStride + j*bColStride],
    /// C(i,j) = c[i*ldc + j]. Transposes are just swapped strides, so one driver covers
    /// NN/NT/TN/TT and every tensor view Ops.Addmm hands it (row- or column-major, narrowed rows).
    ///
    /// Structure (BLIS): C is split into independent tiles run in parallel on CpuParallel; each
    /// tile loops NC (columns) -> KC (depth, B block packed into NR-wide panels) -> MC (rows, A
    /// block packed into MR-tall panels) -> an MR x NR register-tile microkernel. Every tile packs
    /// its own panels into thread-local buffers, so there are no barriers and no shared scratch;
    /// the partition keeps tiles square-ish so the re-packing that costs is ~1/tile-dim of the
    /// arithmetic. Microkernels: AVX-512 8x32 (16 zmm accumulators), AVX2+FMA 6x16 (12 ymm),
    /// and a portable Vector&lt;T&gt; 4x(2*lanes) for everything else (ARM64 included). Skinny
    /// products (M &lt;= 4) skip packing: packing B costs as much as their whole arithmetic.
    ///
    /// Knobs: TS_CPU_SGEMM=0 routes MatrixMultiplication/DirectOps back to their previous
    /// loops; TS_CPU_DISABLE_AVX512=1 forces the AVX2 kernel (so it is testable on an AVX-512
    /// host); TS_CPU_SGEMM_KC / _MC / _NC override the cache blocking for tuning.
    /// </summary>
    public static unsafe class CpuSgemm
    {
        /// <summary>Microkernel families. Avx2Wide is the 8x24 ymm tile that needs the 32-register
        /// EVEX file, so it exists only on AVX-512 hardware.</summary>
        internal enum KernelKind { Portable = 0, Avx2 = 1, Avx2Wide = 2, Avx512 = 3 }

        /// <summary>False when TS_CPU_SGEMM=0: callers keep their pre-existing GEMM loops.</summary>
        public static bool Enabled { get; } = Environment.GetEnvironmentVariable("TS_CPU_SGEMM") != "0";

        internal static readonly bool Avx512DisabledByEnv =
            Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") == "1";

        private static KernelKind _kernel = DefaultKernel();

        /// <summary>The microkernel family in use (tests/benchmarks may pin another supported one).</summary>
        internal static KernelKind ActiveKernel
        {
            get => _kernel;
            set
            {
                if (!IsSupported(value))
                    throw new PlatformNotSupportedException($"{value} kernel is not supported on this CPU.");
                _kernel = value;
            }
        }

        /// <summary>Human-readable name of the active microkernel.</summary>
        public static string ActiveKernelName => _kernel switch
        {
            KernelKind.Avx512 => "avx512-8x32",
            KernelKind.Avx2Wide => "avx2-evex-8x24",
            KernelKind.Avx2 => "avx2-6x16",
            _ => $"portable-4x{2 * Vector<float>.Count}",
        };

        internal static bool IsSupported(KernelKind kind) => kind switch
        {
            KernelKind.Avx512 => Avx512F.IsSupported && Vector512.IsHardwareAccelerated,
            KernelKind.Avx2Wide => Avx512F.VL.IsSupported && Fma.IsSupported,
            KernelKind.Avx2 => Avx2.IsSupported && Fma.IsSupported,
            _ => true,
        };

        /// <summary>
        /// TS_CPU_SGEMM_KERNEL=avx512|avx2wide|avx2|portable pins a kernel; otherwise the widest
        /// supported one, or the AVX2 6x16 tile when TS_CPU_DISABLE_AVX512=1 (the path a CPU
        /// without AVX-512 takes).
        /// </summary>
        private static KernelKind DefaultKernel()
        {
            KernelKind? pinned = Environment.GetEnvironmentVariable("TS_CPU_SGEMM_KERNEL")?.ToLowerInvariant() switch
            {
                "avx512" => KernelKind.Avx512,
                "avx2wide" => KernelKind.Avx2Wide,
                "avx2" => KernelKind.Avx2,
                "portable" => KernelKind.Portable,
                _ => null,
            };
            if (pinned.HasValue && IsSupported(pinned.Value) &&
                !(Avx512DisabledByEnv && pinned.Value >= KernelKind.Avx2Wide))
            {
                return pinned.Value;
            }

            if (!Avx512DisabledByEnv && IsSupported(KernelKind.Avx512)) return KernelKind.Avx512;
            if (IsSupported(KernelKind.Avx2)) return KernelKind.Avx2;
            return KernelKind.Portable;
        }

        // Rows handled by the unpacked skinny path. Up to here the arithmetic per B element
        // (M FMAs) is below the cost of copying that element into a panel.
        private const int SkinnyMaxM = 4;

        // Below this much arithmetic a fork/join costs more than it saves.
        private const double ParallelFlopThreshold = 4e6;

        // Arithmetic per parallel tile we aim for at least (keeps dispatch < a few % of the tile).
        private const double MinTileFlops = 1e6;

        private static readonly int EnvKc = EnvInt("TS_CPU_SGEMM_KC");
        private static readonly int EnvMc = EnvInt("TS_CPU_SGEMM_MC");
        private static readonly int EnvNc = EnvInt("TS_CPU_SGEMM_NC");

        private static int EnvInt(string name)
            => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : 0;

        private readonly struct Blocking
        {
            public readonly KernelKind Kind;
            public readonly int MR, NR, KC, MC, NC;

            public Blocking(KernelKind kind, int mr, int nr, int kc, int mc, int nc)
            {
                Kind = kind;
                MR = mr;
                NR = nr;
                KC = EnvKc > 0 ? EnvKc : kc;
                // MC/NC are kept multiples of the register tile.
                MC = RoundUp(EnvMc > 0 ? EnvMc : mc, mr);
                NC = RoundUp(EnvNc > 0 ? EnvNc : nc, nr);
            }
        }

        // KC x NR B micro-panel (32 KB at 256 x 32) stays in the 48 KB L1D across the MR
        // loop; the MC x KC A block (144 KB) stays in L2 across the NR loop; the KC x NC
        // B block (1 MB) lives in L2/L3 for the whole MC loop.
        private static Blocking GetBlocking(KernelKind kind) => kind switch
        {
            KernelKind.Avx512 => new Blocking(kind, 8, 32, 256, 144, 1024),
            KernelKind.Avx2Wide => new Blocking(kind, 8, 24, 256, 144, 1008),
            KernelKind.Avx2 => new Blocking(kind, 6, 16, 256, 144, 1024),
            _ => new Blocking(kind, 4, 2 * Vector<float>.Count, 256, 128, 1024),
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CeilDiv(int value, int divisor) => (value + divisor - 1) / divisor;

        // ------------------------------------------------------------------------------------
        //  Public entry points
        // ------------------------------------------------------------------------------------

        /// <summary>
        /// C[m,n] = alpha * A * B + beta * C with A(i,p) = a[i*aRowStride + p*aColStride],
        /// B(p,j) = b[p*bRowStride + j*bColStride], C(i,j) = c[i*ldc + j]. beta == 0 never reads C.
        /// </summary>
        public static void Gemm(int m, int n, int k, float alpha,
            float* a, long aRowStride, long aColStride,
            float* b, long bRowStride, long bColStride,
            float beta, float* c, long ldc, bool allowParallel = true)
        {
            GemmBatched(1, m, n, k, alpha, a, 0, aRowStride, aColStride, b, 0, bRowStride, bColStride,
                beta, c, 0, ldc, allowParallel);
        }

        /// <summary>Strided-batch form of <see cref="Gemm"/> (batch offsets in elements).</summary>
        public static void GemmBatched(int batch, int m, int n, int k, float alpha,
            float* a, long aBatchStride, long aRowStride, long aColStride,
            float* b, long bBatchStride, long bRowStride, long bColStride,
            float beta, float* c, long cBatchStride, long ldc, bool allowParallel = true)
        {
            if (batch <= 0 || m <= 0 || n <= 0) return;

            if (k <= 0 || alpha == 0f)
            {
                for (int bi = 0; bi < batch; bi++)
                    ScaleC(m, n, beta, c + bi * cBatchStride, ldc);
                return;
            }

            Blocking blk = GetBlocking(_kernel);
            double itemFlops = 2.0 * m * n * k;
            double totalFlops = itemFlops * batch;
            int threads = allowParallel ? CpuParallel.DegreeOfParallelism : 1;
            bool skinny = m <= SkinnyMaxM && (bColStride == 1 || (bRowStride == 1 && aColStride == 1));

            if (threads <= 1 || totalFlops < ParallelFlopThreshold)
            {
                for (int bi = 0; bi < batch; bi++)
                {
                    RunTile(blk, skinny, m, n, k, alpha,
                        a + bi * aBatchStride, aRowStride, aColStride,
                        b + bi * bBatchStride, bRowStride, bColStride,
                        beta, c + bi * cBatchStride, ldc);
                }
                return;
            }

            // Never cut tiles so small that dispatch dominates them, nor into more blocks than
            // a few per thread (each extra tile re-packs its panels).
            int maxTotal = (int)Math.Max(1, Math.Min(8.0 * threads, totalFlops / MinTileFlops));
            ChoosePartition(blk, skinny, m, n, batch, threads, maxTotal, out int mSplits, out int nSplits,
                out int rowsPerTile, out int colsPerTile);
            int tilesPerItem = mSplits * nSplits;

            CpuParallel.For(batch * tilesPerItem, t =>
            {
                int bi = t / tilesPerItem;
                int tile = t - bi * tilesPerItem;
                int ti = tile / nSplits;
                int tj = tile - ti * nSplits;
                int r0 = ti * rowsPerTile;
                int c0 = tj * colsPerTile;
                int rows = Math.Min(rowsPerTile, m - r0);
                int cols = Math.Min(colsPerTile, n - c0);
                if (rows <= 0 || cols <= 0) return;

                RunTile(blk, skinny, rows, cols, k, alpha,
                    a + bi * aBatchStride + r0 * aRowStride, aRowStride, aColStride,
                    b + bi * bBatchStride + c0 * bColStride, bRowStride, bColStride,
                    beta, c + bi * cBatchStride + r0 * ldc + c0, ldc);
            });
        }

        /// <summary>
        /// Pick the mSplits x nSplits grid of tiles per batch item (tile edges multiples of the
        /// register tile) that minimizes estimated time. Every tile re-packs the A rows and B
        /// columns it touches, which costs about 5/tileCols + 5/tileRows of the arithmetic (a
        /// packed element takes ~0.3 cycles; one FMA lane of arithmetic ~1/16), and the job ends
        /// with its slowest thread, so the estimate is (1 + packing) / balance efficiency, where
        /// efficiency compares the ideal per-thread work with rounds x largest tile. Grids with
        /// fewer than ~2 tiles per thread are only taken when the shape allows no more, so a
        /// descheduled worker cannot stall the whole job.
        /// </summary>
        private static void ChoosePartition(in Blocking blk, bool skinny, int m, int n, int batch, int threads, int maxTotal,
            out int mSplits, out int nSplits, out int rowsPerTile, out int colsPerTile)
        {
            int rowUnit = skinny ? m : blk.MR;
            int colUnit = skinny ? Math.Max(64, 2 * Vector<float>.Count) : blk.NR;
            int mUnits = skinny ? 1 : CeilDiv(m, rowUnit);
            int nUnits = CeilDiv(n, colUnit);
            int maxPerItem = Math.Max(1, maxTotal / batch);
            int minTotal = Math.Min(maxTotal, 2 * threads);

            int bestRowsUnits = mUnits, bestColsUnits = nUnits;
            double bestScore = double.MaxValue;
            bool bestMeetsMin = false;
            for (int ms = 1; ms <= Math.Min(mUnits, maxPerItem); ms++)
            {
                int rowsUnits = CeilDiv(mUnits, ms);
                int realM = CeilDiv(mUnits, rowsUnits);
                if (realM != ms) continue; // same grid as a smaller ms
                for (int ns = 1; ns <= Math.Min(nUnits, maxPerItem / ms); ns++)
                {
                    int colsUnits = CeilDiv(nUnits, ns);
                    int realN = CeilDiv(nUnits, colsUnits);
                    if (realN != ns) continue;

                    int tiles = batch * realM * realN;
                    double tileRows = Math.Min(m, (double)rowsUnits * rowUnit);
                    double tileCols = Math.Min(n, (double)colsUnits * colUnit);
                    int rounds = CeilDiv(tiles, threads);
                    double efficiency = ((double)batch * m * n / threads) / (rounds * tileRows * tileCols);
                    double score = (1.0 + 5.0 / tileRows + 5.0 / tileCols) / Math.Min(1.0, efficiency);
                    bool meetsMin = tiles >= minTotal;
                    if ((meetsMin && !bestMeetsMin) || (meetsMin == bestMeetsMin && score < bestScore - 1e-12))
                    {
                        bestScore = score;
                        bestMeetsMin = meetsMin;
                        bestRowsUnits = rowsUnits;
                        bestColsUnits = colsUnits;
                    }
                }
            }

            rowsPerTile = skinny ? m : bestRowsUnits * rowUnit;
            colsPerTile = bestColsUnits * colUnit;
            mSplits = CeilDiv(m, rowsPerTile);
            nSplits = CeilDiv(n, colsPerTile);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void RunTile(in Blocking blk, bool skinny, int m, int n, int k, float alpha,
            float* a, long ars, long acs, float* b, long brs, long bcs, float beta, float* c, long ldc)
        {
            if (skinny)
            {
                if (bcs == 1) SkinnyRowB(m, n, k, alpha, a, ars, acs, b, brs, beta, c, ldc);
                else SkinnyDot(m, n, k, alpha, a, ars, b, bcs, beta, c, ldc);
                return;
            }

            GemmSerial(blk, m, n, k, alpha, a, ars, acs, b, brs, bcs, beta, c, ldc);
        }

        private static void ScaleC(int m, int n, float beta, float* c, long ldc)
        {
            if (beta == 1f) return;
            for (int i = 0; i < m; i++)
            {
                Span<float> row = new Span<float>(c + i * ldc, n);
                if (beta == 0f) row.Clear();
                else System.Numerics.Tensors.TensorPrimitives.Multiply(row, beta, row);
            }
        }

        // ------------------------------------------------------------------------------------
        //  Serial blocked driver (one tile)
        // ------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void GemmSerial(in Blocking blk, int m, int n, int k, float alpha,
            float* a, long ars, long acs, float* b, long brs, long bcs, float beta, float* c, long ldc)
        {
            int MR = blk.MR, NR = blk.NR;

            // Even K blocks: a 256 + 44 split would run the tail block at a fraction of peak
            // (the C tile load/store is amortized over only 44 steps).
            int kBlocks = CeilDiv(k, blk.KC);
            int KC = CeilDiv(k, kBlocks);
            int MC = Math.Min(blk.MC, RoundUp(m, MR));
            int NC = Math.Min(blk.NC, RoundUp(n, NR));

            float* pa = ThreadBuffers.GetA((long)MC * KC);
            float* pb = ThreadBuffers.GetB((long)NC * KC);
            float* edge = stackalloc float[8 * 32];

            for (int jc = 0; jc < n; jc += NC)
            {
                int nc = Math.Min(NC, n - jc);
                for (int pc = 0; pc < k; pc += KC)
                {
                    int kc = Math.Min(KC, k - pc);
                    float betaEff = pc == 0 ? beta : 1f;
                    PackB(kc, nc, b + pc * brs + jc * bcs, brs, bcs, pb, NR);

                    for (int ic = 0; ic < m; ic += MC)
                    {
                        int mc = Math.Min(MC, m - ic);
                        PackA(kc, mc, a + ic * ars + pc * acs, ars, acs, pa, MR);

                        for (int jr = 0; jr < nc; jr += NR)
                        {
                            int nr = Math.Min(NR, nc - jr);
                            float* bPanel = pb + (long)(jr / NR) * NR * kc;
                            for (int ir = 0; ir < mc; ir += MR)
                            {
                                int mr = Math.Min(MR, mc - ir);
                                float* aPanel = pa + (long)(ir / MR) * MR * kc;
                                float* cTile = c + (ic + ir) * ldc + jc + jr;
                                if (mr == MR && nr == NR)
                                {
                                    Micro(blk.Kind, kc, aPanel, bPanel, cTile, ldc, alpha, betaEff);
                                }
                                else
                                {
                                    // Edge tile: full register tile into scratch, then merge the
                                    // valid part (panels are zero padded, so the rest is 0).
                                    Micro(blk.Kind, kc, aPanel, bPanel, edge, NR, 1f, 0f);
                                    MergeEdge(edge, NR, cTile, ldc, mr, nr, alpha, betaEff);
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void MergeEdge(float* tmp, int tmpStride, float* c, long ldc, int mr, int nr, float alpha, float beta)
        {
            for (int i = 0; i < mr; i++)
            {
                float* src = tmp + i * tmpStride;
                float* dst = c + i * ldc;
                if (beta == 0f)
                {
                    for (int j = 0; j < nr; j++) dst[j] = alpha * src[j];
                }
                else if (beta == 1f)
                {
                    for (int j = 0; j < nr; j++) dst[j] += alpha * src[j];
                }
                else
                {
                    for (int j = 0; j < nr; j++) dst[j] = alpha * src[j] + beta * dst[j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Micro(KernelKind kind, int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            if (kind == KernelKind.Avx512) Kernel8x32(kc, pa, pb, c, ldc, alpha, beta);
            else if (kind == KernelKind.Avx2Wide) Kernel8x24(kc, pa, pb, c, ldc, alpha, beta);
            else if (kind == KernelKind.Avx2) Kernel6x16(kc, pa, pb, c, ldc, alpha, beta);
            else KernelPortable(kc, pa, pb, c, ldc, alpha, beta);
        }

        // ------------------------------------------------------------------------------------
        //  Packing. A block -> MR-row panels laid out [p][r]; B block -> NR-column panels [p][j].
        //  Both zero padded to the full register tile. The layout the operand already has picks
        //  the loop: a contiguous run along the packed dimension is a straight vector copy, a
        //  run along K (row-major A, or B given as its transpose) goes through 8x8 transposes.
        // ------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackA(int kc, int mc, float* a, long ars, long acs, float* dst, int MR)
        {
            for (int i = 0; i < mc; i += MR)
            {
                int mr = Math.Min(MR, mc - i);
                float* src = a + i * ars;
                if (acs == 1 && mr == MR && (MR & 7) == 0 && Avx.IsSupported)
                {
                    PackTransposed(kc, src, ars, MR, dst, MR);
                }
                else if (ars == 1)
                {
                    for (int p = 0; p < kc; p++)
                    {
                        float* s = src + p * acs;
                        float* d = dst + p * MR;
                        int r = 0;
                        for (; r < mr; r++) d[r] = s[r];
                        for (; r < MR; r++) d[r] = 0f;
                    }
                }
                else
                {
                    for (int r = 0; r < mr; r++)
                    {
                        float* s = src + r * ars;
                        float* d = dst + r;
                        if (acs == 1)
                        {
                            for (int p = 0; p < kc; p++) d[p * MR] = s[p];
                        }
                        else
                        {
                            for (int p = 0; p < kc; p++) d[p * MR] = s[p * acs];
                        }
                    }
                    for (int r = mr; r < MR; r++)
                    {
                        float* d = dst + r;
                        for (int p = 0; p < kc; p++) d[p * MR] = 0f;
                    }
                }
                dst += (long)MR * kc;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackB(int kc, int nc, float* b, long brs, long bcs, float* dst, int NR)
        {
            for (int j = 0; j < nc; j += NR)
            {
                int nr = Math.Min(NR, nc - j);
                float* src = b + j * bcs;
                if (bcs == 1)
                {
                    if (nr == NR)
                    {
                        for (int p = 0; p < kc; p++)
                            CopyPanelRow(src + p * brs, dst + p * NR, NR);
                    }
                    else
                    {
                        for (int p = 0; p < kc; p++)
                        {
                            float* s = src + p * brs;
                            float* d = dst + p * NR;
                            int q = 0;
                            for (; q < nr; q++) d[q] = s[q];
                            for (; q < NR; q++) d[q] = 0f;
                        }
                    }
                }
                else if (brs == 1 && nr == NR && (NR & 7) == 0 && Avx.IsSupported)
                {
                    PackTransposed(kc, src, bcs, NR, dst, NR);
                }
                else
                {
                    for (int q = 0; q < nr; q++)
                    {
                        float* s = src + q * bcs;
                        float* d = dst + q;
                        if (brs == 1)
                        {
                            for (int p = 0; p < kc; p++) d[p * NR] = s[p];
                        }
                        else
                        {
                            for (int p = 0; p < kc; p++) d[p * NR] = s[p * brs];
                        }
                    }
                    for (int q = nr; q < NR; q++)
                    {
                        float* d = dst + q;
                        for (int p = 0; p < kc; p++) d[p * NR] = 0f;
                    }
                }
                dst += (long)NR * kc;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopyPanelRow(float* src, float* dst, int count)
        {
            int q = 0;
            if (Vector512.IsHardwareAccelerated)
            {
                for (; q + 16 <= count; q += 16)
                    Vector512.Store(Vector512.Load(src + q), dst + q);
            }
            if (Vector256.IsHardwareAccelerated)
            {
                for (; q + 8 <= count; q += 8)
                    Vector256.Store(Vector256.Load(src + q), dst + q);
            }
            if (Vector128.IsHardwareAccelerated)
            {
                for (; q + 4 <= count; q += 4)
                    Vector128.Store(Vector128.Load(src + q), dst + q);
            }
            for (; q < count; q++)
                dst[q] = src[q];
        }

        /// <summary>
        /// <paramref name="lines"/> source lines (line q at src + q*lineStride, contiguous along p)
        /// -> dst[p*width + q]: the panel layout for a row-major A block or a transposed B block.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackTransposed(int kc, float* src, long lineStride, int lines, float* dst, int width)
        {
            for (int q0 = 0; q0 < lines; q0 += 8)
            {
                float* s0 = src + q0 * lineStride;
                float* s1 = s0 + lineStride;
                float* s2 = s1 + lineStride;
                float* s3 = s2 + lineStride;
                float* s4 = s3 + lineStride;
                float* s5 = s4 + lineStride;
                float* s6 = s5 + lineStride;
                float* s7 = s6 + lineStride;
                float* d = dst + q0;
                int p = 0;
                for (; p + 8 <= kc; p += 8)
                {
                    Transpose8x8(s0 + p, s1 + p, s2 + p, s3 + p, s4 + p, s5 + p, s6 + p, s7 + p, d + p * width, width);
                }
                for (; p < kc; p++)
                {
                    float* dp = d + p * width;
                    dp[0] = s0[p]; dp[1] = s1[p]; dp[2] = s2[p]; dp[3] = s3[p];
                    dp[4] = s4[p]; dp[5] = s5[p]; dp[6] = s6[p]; dp[7] = s7[p];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Transpose8x8(float* s0, float* s1, float* s2, float* s3,
            float* s4, float* s5, float* s6, float* s7, float* d, long dstStride)
        {
            Vector256<float> r0 = Avx.LoadVector256(s0), r1 = Avx.LoadVector256(s1);
            Vector256<float> r2 = Avx.LoadVector256(s2), r3 = Avx.LoadVector256(s3);
            Vector256<float> r4 = Avx.LoadVector256(s4), r5 = Avx.LoadVector256(s5);
            Vector256<float> r6 = Avx.LoadVector256(s6), r7 = Avx.LoadVector256(s7);

            Vector256<float> t0 = Avx.UnpackLow(r0, r1), t1 = Avx.UnpackHigh(r0, r1);
            Vector256<float> t2 = Avx.UnpackLow(r2, r3), t3 = Avx.UnpackHigh(r2, r3);
            Vector256<float> t4 = Avx.UnpackLow(r4, r5), t5 = Avx.UnpackHigh(r4, r5);
            Vector256<float> t6 = Avx.UnpackLow(r6, r7), t7 = Avx.UnpackHigh(r6, r7);

            Vector256<float> u0 = Avx.Shuffle(t0, t2, 0x44), u1 = Avx.Shuffle(t0, t2, 0xEE);
            Vector256<float> u2 = Avx.Shuffle(t1, t3, 0x44), u3 = Avx.Shuffle(t1, t3, 0xEE);
            Vector256<float> u4 = Avx.Shuffle(t4, t6, 0x44), u5 = Avx.Shuffle(t4, t6, 0xEE);
            Vector256<float> u6 = Avx.Shuffle(t5, t7, 0x44), u7 = Avx.Shuffle(t5, t7, 0xEE);

            Avx.Store(d, Avx.Permute2x128(u0, u4, 0x20));
            Avx.Store(d + dstStride, Avx.Permute2x128(u1, u5, 0x20));
            Avx.Store(d + 2 * dstStride, Avx.Permute2x128(u2, u6, 0x20));
            Avx.Store(d + 3 * dstStride, Avx.Permute2x128(u3, u7, 0x20));
            Avx.Store(d + 4 * dstStride, Avx.Permute2x128(u0, u4, 0x31));
            Avx.Store(d + 5 * dstStride, Avx.Permute2x128(u1, u5, 0x31));
            Avx.Store(d + 6 * dstStride, Avx.Permute2x128(u2, u6, 0x31));
            Avx.Store(d + 7 * dstStride, Avx.Permute2x128(u3, u7, 0x31));
        }

        // ------------------------------------------------------------------------------------
        //  Microkernels: C[MR x NR] (op)= alpha * sum_p Apanel[p][0..MR) (x) Bpanel[p][0..NR).
        //  beta == 0 stores without reading C, beta == 1 accumulates (later K blocks).
        // ------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void Kernel8x32(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            Vector512<float> c00 = Vector512<float>.Zero, c01 = Vector512<float>.Zero;
            Vector512<float> c10 = Vector512<float>.Zero, c11 = Vector512<float>.Zero;
            Vector512<float> c20 = Vector512<float>.Zero, c21 = Vector512<float>.Zero;
            Vector512<float> c30 = Vector512<float>.Zero, c31 = Vector512<float>.Zero;
            Vector512<float> c40 = Vector512<float>.Zero, c41 = Vector512<float>.Zero;
            Vector512<float> c50 = Vector512<float>.Zero, c51 = Vector512<float>.Zero;
            Vector512<float> c60 = Vector512<float>.Zero, c61 = Vector512<float>.Zero;
            Vector512<float> c70 = Vector512<float>.Zero, c71 = Vector512<float>.Zero;

            for (int p = 0; p < kc; p++)
            {
                Vector512<float> b0 = Avx512F.LoadVector512(pb);
                Vector512<float> b1 = Avx512F.LoadVector512(pb + 16);
                Vector512<float> av = Vector512.Create(pa[0]);
                c00 = Avx512F.FusedMultiplyAdd(av, b0, c00);
                c01 = Avx512F.FusedMultiplyAdd(av, b1, c01);
                av = Vector512.Create(pa[1]);
                c10 = Avx512F.FusedMultiplyAdd(av, b0, c10);
                c11 = Avx512F.FusedMultiplyAdd(av, b1, c11);
                av = Vector512.Create(pa[2]);
                c20 = Avx512F.FusedMultiplyAdd(av, b0, c20);
                c21 = Avx512F.FusedMultiplyAdd(av, b1, c21);
                av = Vector512.Create(pa[3]);
                c30 = Avx512F.FusedMultiplyAdd(av, b0, c30);
                c31 = Avx512F.FusedMultiplyAdd(av, b1, c31);
                av = Vector512.Create(pa[4]);
                c40 = Avx512F.FusedMultiplyAdd(av, b0, c40);
                c41 = Avx512F.FusedMultiplyAdd(av, b1, c41);
                av = Vector512.Create(pa[5]);
                c50 = Avx512F.FusedMultiplyAdd(av, b0, c50);
                c51 = Avx512F.FusedMultiplyAdd(av, b1, c51);
                av = Vector512.Create(pa[6]);
                c60 = Avx512F.FusedMultiplyAdd(av, b0, c60);
                c61 = Avx512F.FusedMultiplyAdd(av, b1, c61);
                av = Vector512.Create(pa[7]);
                c70 = Avx512F.FusedMultiplyAdd(av, b0, c70);
                c71 = Avx512F.FusedMultiplyAdd(av, b1, c71);
                pa += 8;
                pb += 32;
            }

            Vector512<float> va = Vector512.Create(alpha);
            if (beta == 0f)
            {
                Store512x2(c, c00, c01, va); Store512x2(c + ldc, c10, c11, va);
                Store512x2(c + 2 * ldc, c20, c21, va); Store512x2(c + 3 * ldc, c30, c31, va);
                Store512x2(c + 4 * ldc, c40, c41, va); Store512x2(c + 5 * ldc, c50, c51, va);
                Store512x2(c + 6 * ldc, c60, c61, va); Store512x2(c + 7 * ldc, c70, c71, va);
            }
            else
            {
                Vector512<float> vb = Vector512.Create(beta);
                bool one = beta == 1f;
                Update512x2(c, c00, c01, va, vb, one); Update512x2(c + ldc, c10, c11, va, vb, one);
                Update512x2(c + 2 * ldc, c20, c21, va, vb, one); Update512x2(c + 3 * ldc, c30, c31, va, vb, one);
                Update512x2(c + 4 * ldc, c40, c41, va, vb, one); Update512x2(c + 5 * ldc, c50, c51, va, vb, one);
                Update512x2(c + 6 * ldc, c60, c61, va, vb, one); Update512x2(c + 7 * ldc, c70, c71, va, vb, one);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store512x2(float* c, Vector512<float> v0, Vector512<float> v1, Vector512<float> va)
        {
            Avx512F.Store(c, Avx512F.Multiply(v0, va));
            Avx512F.Store(c + 16, Avx512F.Multiply(v1, va));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Update512x2(float* c, Vector512<float> v0, Vector512<float> v1,
            Vector512<float> va, Vector512<float> vb, bool betaIsOne)
        {
            Vector512<float> o0 = Avx512F.LoadVector512(c);
            Vector512<float> o1 = Avx512F.LoadVector512(c + 16);
            if (!betaIsOne)
            {
                o0 = Avx512F.Multiply(o0, vb);
                o1 = Avx512F.Multiply(o1, vb);
            }
            Avx512F.Store(c, Avx512F.FusedMultiplyAdd(v0, va, o0));
            Avx512F.Store(c + 16, Avx512F.FusedMultiplyAdd(v1, va, o1));
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void Kernel6x16(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            Vector256<float> c00 = Vector256<float>.Zero, c01 = Vector256<float>.Zero;
            Vector256<float> c10 = Vector256<float>.Zero, c11 = Vector256<float>.Zero;
            Vector256<float> c20 = Vector256<float>.Zero, c21 = Vector256<float>.Zero;
            Vector256<float> c30 = Vector256<float>.Zero, c31 = Vector256<float>.Zero;
            Vector256<float> c40 = Vector256<float>.Zero, c41 = Vector256<float>.Zero;
            Vector256<float> c50 = Vector256<float>.Zero, c51 = Vector256<float>.Zero;

            for (int p = 0; p < kc; p++)
            {
                Vector256<float> b0 = Avx.LoadVector256(pb);
                Vector256<float> b1 = Avx.LoadVector256(pb + 8);
                Vector256<float> av = Vector256.Create(pa[0]);
                c00 = Fma.MultiplyAdd(av, b0, c00);
                c01 = Fma.MultiplyAdd(av, b1, c01);
                av = Vector256.Create(pa[1]);
                c10 = Fma.MultiplyAdd(av, b0, c10);
                c11 = Fma.MultiplyAdd(av, b1, c11);
                av = Vector256.Create(pa[2]);
                c20 = Fma.MultiplyAdd(av, b0, c20);
                c21 = Fma.MultiplyAdd(av, b1, c21);
                av = Vector256.Create(pa[3]);
                c30 = Fma.MultiplyAdd(av, b0, c30);
                c31 = Fma.MultiplyAdd(av, b1, c31);
                av = Vector256.Create(pa[4]);
                c40 = Fma.MultiplyAdd(av, b0, c40);
                c41 = Fma.MultiplyAdd(av, b1, c41);
                av = Vector256.Create(pa[5]);
                c50 = Fma.MultiplyAdd(av, b0, c50);
                c51 = Fma.MultiplyAdd(av, b1, c51);
                pa += 6;
                pb += 16;
            }

            Vector256<float> va = Vector256.Create(alpha);
            if (beta == 0f)
            {
                Store256x2(c, c00, c01, va); Store256x2(c + ldc, c10, c11, va);
                Store256x2(c + 2 * ldc, c20, c21, va); Store256x2(c + 3 * ldc, c30, c31, va);
                Store256x2(c + 4 * ldc, c40, c41, va); Store256x2(c + 5 * ldc, c50, c51, va);
            }
            else
            {
                Vector256<float> vb = Vector256.Create(beta);
                bool one = beta == 1f;
                Update256x2(c, c00, c01, va, vb, one); Update256x2(c + ldc, c10, c11, va, vb, one);
                Update256x2(c + 2 * ldc, c20, c21, va, vb, one); Update256x2(c + 3 * ldc, c30, c31, va, vb, one);
                Update256x2(c + 4 * ldc, c40, c41, va, vb, one); Update256x2(c + 5 * ldc, c50, c51, va, vb, one);
            }
        }

        /// <summary>
        /// 8x24 tile on 256-bit vectors: 24 ymm accumulators, which needs the 32-register EVEX
        /// file (AVX-512VL hardware). Same register-tile volume as the zmm kernel, but 256-bit
        /// FMAs keep client cores in their AVX2 turbo licence.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void Kernel8x24(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            Vector256<float> c00 = Vector256<float>.Zero, c01 = Vector256<float>.Zero, c02 = Vector256<float>.Zero;
            Vector256<float> c10 = Vector256<float>.Zero, c11 = Vector256<float>.Zero, c12 = Vector256<float>.Zero;
            Vector256<float> c20 = Vector256<float>.Zero, c21 = Vector256<float>.Zero, c22 = Vector256<float>.Zero;
            Vector256<float> c30 = Vector256<float>.Zero, c31 = Vector256<float>.Zero, c32 = Vector256<float>.Zero;
            Vector256<float> c40 = Vector256<float>.Zero, c41 = Vector256<float>.Zero, c42 = Vector256<float>.Zero;
            Vector256<float> c50 = Vector256<float>.Zero, c51 = Vector256<float>.Zero, c52 = Vector256<float>.Zero;
            Vector256<float> c60 = Vector256<float>.Zero, c61 = Vector256<float>.Zero, c62 = Vector256<float>.Zero;
            Vector256<float> c70 = Vector256<float>.Zero, c71 = Vector256<float>.Zero, c72 = Vector256<float>.Zero;

            for (int p = 0; p < kc; p++)
            {
                Vector256<float> b0 = Avx.LoadVector256(pb);
                Vector256<float> b1 = Avx.LoadVector256(pb + 8);
                Vector256<float> b2 = Avx.LoadVector256(pb + 16);
                Vector256<float> av = Vector256.Create(pa[0]);
                c00 = Fma.MultiplyAdd(av, b0, c00); c01 = Fma.MultiplyAdd(av, b1, c01); c02 = Fma.MultiplyAdd(av, b2, c02);
                av = Vector256.Create(pa[1]);
                c10 = Fma.MultiplyAdd(av, b0, c10); c11 = Fma.MultiplyAdd(av, b1, c11); c12 = Fma.MultiplyAdd(av, b2, c12);
                av = Vector256.Create(pa[2]);
                c20 = Fma.MultiplyAdd(av, b0, c20); c21 = Fma.MultiplyAdd(av, b1, c21); c22 = Fma.MultiplyAdd(av, b2, c22);
                av = Vector256.Create(pa[3]);
                c30 = Fma.MultiplyAdd(av, b0, c30); c31 = Fma.MultiplyAdd(av, b1, c31); c32 = Fma.MultiplyAdd(av, b2, c32);
                av = Vector256.Create(pa[4]);
                c40 = Fma.MultiplyAdd(av, b0, c40); c41 = Fma.MultiplyAdd(av, b1, c41); c42 = Fma.MultiplyAdd(av, b2, c42);
                av = Vector256.Create(pa[5]);
                c50 = Fma.MultiplyAdd(av, b0, c50); c51 = Fma.MultiplyAdd(av, b1, c51); c52 = Fma.MultiplyAdd(av, b2, c52);
                av = Vector256.Create(pa[6]);
                c60 = Fma.MultiplyAdd(av, b0, c60); c61 = Fma.MultiplyAdd(av, b1, c61); c62 = Fma.MultiplyAdd(av, b2, c62);
                av = Vector256.Create(pa[7]);
                c70 = Fma.MultiplyAdd(av, b0, c70); c71 = Fma.MultiplyAdd(av, b1, c71); c72 = Fma.MultiplyAdd(av, b2, c72);
                pa += 8;
                pb += 24;
            }

            Vector256<float> va = Vector256.Create(alpha);
            Vector256<float> vb = Vector256.Create(beta);
            Row256x3(c, c00, c01, c02, va, vb, beta);
            Row256x3(c + ldc, c10, c11, c12, va, vb, beta);
            Row256x3(c + 2 * ldc, c20, c21, c22, va, vb, beta);
            Row256x3(c + 3 * ldc, c30, c31, c32, va, vb, beta);
            Row256x3(c + 4 * ldc, c40, c41, c42, va, vb, beta);
            Row256x3(c + 5 * ldc, c50, c51, c52, va, vb, beta);
            Row256x3(c + 6 * ldc, c60, c61, c62, va, vb, beta);
            Row256x3(c + 7 * ldc, c70, c71, c72, va, vb, beta);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Row256x3(float* c, Vector256<float> v0, Vector256<float> v1, Vector256<float> v2,
            Vector256<float> va, Vector256<float> vb, float beta)
        {
            if (beta == 0f)
            {
                Avx.Store(c, Avx.Multiply(v0, va));
                Avx.Store(c + 8, Avx.Multiply(v1, va));
                Avx.Store(c + 16, Avx.Multiply(v2, va));
                return;
            }

            Vector256<float> o0 = Avx.LoadVector256(c);
            Vector256<float> o1 = Avx.LoadVector256(c + 8);
            Vector256<float> o2 = Avx.LoadVector256(c + 16);
            if (beta != 1f)
            {
                o0 = Avx.Multiply(o0, vb);
                o1 = Avx.Multiply(o1, vb);
                o2 = Avx.Multiply(o2, vb);
            }
            Avx.Store(c, Fma.MultiplyAdd(v0, va, o0));
            Avx.Store(c + 8, Fma.MultiplyAdd(v1, va, o1));
            Avx.Store(c + 16, Fma.MultiplyAdd(v2, va, o2));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store256x2(float* c, Vector256<float> v0, Vector256<float> v1, Vector256<float> va)
        {
            Avx.Store(c, Avx.Multiply(v0, va));
            Avx.Store(c + 8, Avx.Multiply(v1, va));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Update256x2(float* c, Vector256<float> v0, Vector256<float> v1,
            Vector256<float> va, Vector256<float> vb, bool betaIsOne)
        {
            Vector256<float> o0 = Avx.LoadVector256(c);
            Vector256<float> o1 = Avx.LoadVector256(c + 8);
            if (!betaIsOne)
            {
                o0 = Avx.Multiply(o0, vb);
                o1 = Avx.Multiply(o1, vb);
            }
            Avx.Store(c, Fma.MultiplyAdd(v0, va, o0));
            Avx.Store(c + 8, Fma.MultiplyAdd(v1, va, o1));
        }

        /// <summary>Portable 4 x (2 * Vector&lt;float&gt;.Count) tile (ARM64 NEON, pre-AVX2 x64).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void KernelPortable(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            int w = Vector<float>.Count;
            Vector<float> c00 = Vector<float>.Zero, c01 = Vector<float>.Zero;
            Vector<float> c10 = Vector<float>.Zero, c11 = Vector<float>.Zero;
            Vector<float> c20 = Vector<float>.Zero, c21 = Vector<float>.Zero;
            Vector<float> c30 = Vector<float>.Zero, c31 = Vector<float>.Zero;
            for (int p = 0; p < kc; p++)
            {
                Vector<float> b0 = Unsafe.ReadUnaligned<Vector<float>>(pb);
                Vector<float> b1 = Unsafe.ReadUnaligned<Vector<float>>(pb + w);
                Vector<float> av = new Vector<float>(pa[0]);
                c00 += av * b0; c01 += av * b1;
                av = new Vector<float>(pa[1]);
                c10 += av * b0; c11 += av * b1;
                av = new Vector<float>(pa[2]);
                c20 += av * b0; c21 += av * b1;
                av = new Vector<float>(pa[3]);
                c30 += av * b0; c31 += av * b1;
                pa += 4;
                pb += 2 * w;
            }

            StorePortable(c, c00, c01, alpha, beta, w);
            StorePortable(c + ldc, c10, c11, alpha, beta, w);
            StorePortable(c + 2 * ldc, c20, c21, alpha, beta, w);
            StorePortable(c + 3 * ldc, c30, c31, alpha, beta, w);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StorePortable(float* c, Vector<float> v0, Vector<float> v1, float alpha, float beta, int w)
        {
            Vector<float> va = new Vector<float>(alpha);
            v0 *= va;
            v1 *= va;
            if (beta != 0f)
            {
                Vector<float> o0 = Unsafe.ReadUnaligned<Vector<float>>(c);
                Vector<float> o1 = Unsafe.ReadUnaligned<Vector<float>>(c + w);
                if (beta == 1f)
                {
                    v0 += o0;
                    v1 += o1;
                }
                else
                {
                    Vector<float> vb = new Vector<float>(beta);
                    v0 += o0 * vb;
                    v1 += o1 * vb;
                }
            }
            Unsafe.WriteUnaligned(c, v0);
            Unsafe.WriteUnaligned(c + w, v1);
        }

        // ------------------------------------------------------------------------------------
        //  Skinny products (M <= 4): memory bound on B, so read B where it lies.
        // ------------------------------------------------------------------------------------

        /// <summary>B row-major (contiguous along N): C rows += A(i,p) * B[p, :] over column chunks
        /// kept L1-resident, B streamed once per chunk.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void SkinnyRowB(int m, int n, int k, float alpha, float* a, long ars, long acs,
            float* b, long brs, float beta, float* c, long ldc)
        {
            const int Chunk = 512;
            int w = Vector<float>.Count;
            for (int j0 = 0; j0 < n; j0 += Chunk)
            {
                int nc = Math.Min(Chunk, n - j0);
                for (int i = 0; i < m; i++)
                {
                    Span<float> row = new Span<float>(c + i * ldc + j0, nc);
                    if (beta == 0f) row.Clear();
                    else if (beta != 1f) System.Numerics.Tensors.TensorPrimitives.Multiply(row, beta, row);
                }

                for (int p = 0; p < k; p++)
                {
                    float* bp = b + p * brs + j0;
                    for (int i = 0; i < m; i++)
                    {
                        float s = alpha * a[i * ars + p * acs];
                        if (s == 0f) continue;
                        float* cp = c + i * ldc + j0;
                        Vector<float> vs = new Vector<float>(s);
                        int j = 0;
                        for (; j + w <= nc; j += w)
                        {
                            Vector<float> acc = Unsafe.ReadUnaligned<Vector<float>>(cp + j);
                            acc += vs * Unsafe.ReadUnaligned<Vector<float>>(bp + j);
                            Unsafe.WriteUnaligned(cp + j, acc);
                        }
                        for (; j < nc; j++) cp[j] += s * bp[j];
                    }
                }
            }
        }

        /// <summary>B given as its transpose (B^T rows contiguous along K) and A rows contiguous:
        /// dot products, two B^T rows at a time against every A row.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void SkinnyDot(int m, int n, int k, float alpha, float* a, long ars,
            float* b, long bcs, float beta, float* c, long ldc)
        {
            int j = 0;
            for (; j + 2 <= n; j += 2)
            {
                float* b0 = b + j * bcs;
                float* b1 = b0 + bcs;
                for (int i = 0; i < m; i++)
                {
                    float* ar = a + i * ars;
                    Dot2(ar, b0, b1, k, out float d0, out float d1);
                    float* cp = c + i * ldc + j;
                    if (beta == 0f)
                    {
                        cp[0] = alpha * d0;
                        cp[1] = alpha * d1;
                    }
                    else
                    {
                        cp[0] = alpha * d0 + beta * cp[0];
                        cp[1] = alpha * d1 + beta * cp[1];
                    }
                }
            }
            for (; j < n; j++)
            {
                float* b0 = b + j * bcs;
                for (int i = 0; i < m; i++)
                {
                    float d0 = System.Numerics.Tensors.TensorPrimitives.Dot(
                        new ReadOnlySpan<float>(a + i * ars, k), new ReadOnlySpan<float>(b0, k));
                    float* cp = c + i * ldc + j;
                    cp[0] = beta == 0f ? alpha * d0 : alpha * d0 + beta * cp[0];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Dot2(float* a, float* b0, float* b1, int k, out float d0, out float d1)
        {
            int w = Vector<float>.Count;
            Vector<float> s00 = Vector<float>.Zero, s01 = Vector<float>.Zero;
            Vector<float> s10 = Vector<float>.Zero, s11 = Vector<float>.Zero;
            int p = 0;
            for (; p + 2 * w <= k; p += 2 * w)
            {
                Vector<float> x0 = Unsafe.ReadUnaligned<Vector<float>>(a + p);
                Vector<float> x1 = Unsafe.ReadUnaligned<Vector<float>>(a + p + w);
                s00 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b0 + p);
                s01 += x1 * Unsafe.ReadUnaligned<Vector<float>>(b0 + p + w);
                s10 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b1 + p);
                s11 += x1 * Unsafe.ReadUnaligned<Vector<float>>(b1 + p + w);
            }
            for (; p + w <= k; p += w)
            {
                Vector<float> x0 = Unsafe.ReadUnaligned<Vector<float>>(a + p);
                s00 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b0 + p);
                s10 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b1 + p);
            }
            float r0 = Vector.Sum(s00 + s01);
            float r1 = Vector.Sum(s10 + s11);
            for (; p < k; p++)
            {
                r0 += a[p] * b0[p];
                r1 += a[p] * b1[p];
            }
            d0 = r0;
            d1 = r1;
        }

        // ------------------------------------------------------------------------------------
        //  Per-thread packing buffers (64-byte aligned, grown on demand, kept for the thread's
        //  lifetime: the pool's workers run GEMM after GEMM with the same shapes).
        // ------------------------------------------------------------------------------------

        private static class ThreadBuffers
        {
            [ThreadStatic] private static Holder? _a;
            [ThreadStatic] private static Holder? _b;

            public static float* GetA(long floats) => (_a ??= new Holder()).Get(floats);
            public static float* GetB(long floats) => (_b ??= new Holder()).Get(floats);

            private sealed class Holder
            {
                private float* _ptr;
                private long _capacity;

                public float* Get(long floats)
                {
                    if (floats > _capacity)
                    {
                        if (_ptr != null) NativeMemory.AlignedFree(_ptr);
                        long cap = Math.Max(floats, _capacity * 2);
                        _ptr = (float*)NativeMemory.AlignedAlloc((nuint)(cap * sizeof(float)), 64);
                        _capacity = cap;
                    }
                    return _ptr;
                }

                ~Holder()
                {
                    if (_ptr != null) NativeMemory.AlignedFree(_ptr);
                }
            }
        }
    }
}
