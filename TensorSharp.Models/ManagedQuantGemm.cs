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
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // Multi-row quantized GEMM for the pure-C# CPU backend.
    //
    // The per-row path (TryAddmmQuantizedToFloat32's DotQuantized loop) calls
    // one weight-row x activation-row dot per output element, so every weight
    // block is re-decoded (nibble unpack, 6-bit scale unpack, fp16 scales) for
    // EVERY activation row. At prefill widths that decode is most of the work:
    // DiffusionGemma's dense MLP measured ~36 GFLOPS that way.
    //
    // Here a pair of weight rows ("columns" of the output) is decoded ONCE into
    // an L1-resident scratch - unsigned int8 values in natural element order,
    // pre-broadcast int16 sub-block scales, float block scales - and then run
    // against every activation row of the tile with a register-blocked
    // microkernel: 8 rows x 2 columns in one zmm (AVX-512BW) or 4 rows x 1
    // column (AVX2). The integer arithmetic is the one the per-row kernels use
    // (vpmaddubsw into int16, vpmaddwd against the sub-block scale into int32),
    // so the per-sub-block integer sums are exact and only the float scaling is
    // re-associated.
    //
    // Activations are quantized exactly as before (same Q8_K / Q8_0 values and
    // scales); only their layout differs: all int8 values of a row first
    // (32-byte aligned chunks, so no load splits a cache line), then the side
    // data the kernels want pre-computed once per row instead of once per
    // (row, column) pair - float scales, and for Q4_K/Q5_K the d8 * bsum
    // products that carry the K-quant min term.
    //
    // A/B: TS_CPU_QGEMM=0 restores the per-row path for everything this file
    // handles; TS_CPU_DISABLE_AVX512=1 forces the AVX2 kernels on an AVX-512
    // machine. Hosts without AVX2+FMA (and ARM64) keep the per-row path.
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        /// <summary>Kernel selection for the multi-row GEMM. <see cref="Auto"/>
        /// honours the TS_CPU_QGEMM / TS_CPU_DISABLE_AVX512 switches; the others
        /// exist so tests and benchmarks can compare paths in one process.</summary>
        internal enum QGemmIsa
        {
            Auto = 0,
            Legacy = 1,
            Avx2 = 2,
            Avx512 = 3,
        }

        private enum QGemmFamily
        {
            None,
            /// <summary>Q4_K / Q5_K: unsigned values, 32-element sub-blocks with a
            /// 6-bit scale and a 6-bit min (min term via Q8_K bsums).</summary>
            KMin,
            /// <summary>Q6_K: unsigned 0..63 values, 16-element int8 scales, -32
            /// zero point folded in as an exact integer correction off the bsums.</summary>
            KOfs,
            /// <summary>Q4_0 / Q5_0: unsigned values with a zero point, folded in
            /// through the activation block sums.</summary>
            Q0Unsigned,
            /// <summary>Q8_0: signed values (sign moved onto the activation).</summary>
            Q0Signed,
        }

        private static readonly bool QGemmEnabled =
            Environment.GetEnvironmentVariable("TS_CPU_QGEMM") != "0";
        private static readonly bool QGemmAvx512Disabled =
            Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") == "1";
        // Smallest row count that takes the GEMM (TS_CPU_QGEMM_MIN_ROWS overrides
        // it for every type). Even one row gains where the per-row dot spends
        // more on decoding than on reading: DRAM-bound 4096x14336 matvecs measured
        // Q4_K 1.4-1.6x, Q6_K 1.7-1.8x, Q5_K 1.3x, Q5_0 1.3-1.7x, BF16 1.5x faster
        // through the GEMM. Q4_0/Q8_0 single rows were a wash (0.96-1.24x), so they
        // keep the per-row kernels.
        private static readonly int QGemmMinRowsOverride = (int)Math.Clamp(EnvLong("TS_CPU_QGEMM_MIN_ROWS", 0), 0, int.MaxValue);

        private static int QGemmMinRows(GgmlTensorType type)
        {
            if (QGemmMinRowsOverride > 0) return QGemmMinRowsOverride;
            return type is GgmlTensorType.Q4_0 or GgmlTensorType.Q8_0 ? 2 : 1;
        }
        // Minimum multiply-accumulates per parallel task. ~1M MACs is ~50 us on
        // one core here - big enough to bury the pool dispatch, small enough that
        // an MoE expert (4 rows x 2816 x 1408 = 16M MACs) still fans out.
        private static readonly long QGemmMinTaskMacs = EnvLong("TS_CPU_QGEMM_TASK_MACS", 1L << 20);
        // Activation bytes one row block may hold. The block is re-read once per
        // decoded column pair, so it has to stay in L2 next to the pair scratch
        // (~18 KB at K=4096); 1.25 MB L2 parts keep ~40% of it for this.
        private static readonly long QGemmRowBlockBytes = EnvLong("TS_CPU_QGEMM_L2_BYTES", 512 * 1024);

        // Diagnostic: TS_CPU_QGEMM_VERIFY=1 re-runs every GEMM through the per-row
        // path and reports the largest relative difference seen so far (stderr),
        // which checks the kernels on a real model's weights and activations.
        private static readonly bool QGemmVerify =
            Environment.GetEnvironmentVariable("TS_CPU_QGEMM_VERIFY") == "1";
        private static double _qgemmVerifyWorst;
        private static readonly object QGemmVerifyLock = new object();

        internal static bool QGemmAvx2Supported => Avx2.IsSupported && Fma.IsSupported;

        internal static bool QGemmAvx512Supported =>
            QGemmAvx2Supported && Avx512F.IsSupported && Avx512BW.IsSupported && Avx512DQ.IsSupported
            && Vector512.IsHardwareAccelerated;

        /// <summary>Concrete kernel set for a request: Legacy when the GEMM is
        /// switched off or the ISA is missing.</summary>
        internal static QGemmIsa ResolveQGemmIsa(QGemmIsa requested)
        {
            switch (requested)
            {
                case QGemmIsa.Legacy:
                    return QGemmIsa.Legacy;
                case QGemmIsa.Avx512:
                    return QGemmAvx512Supported ? QGemmIsa.Avx512 : QGemmAvx2Supported ? QGemmIsa.Avx2 : QGemmIsa.Legacy;
                case QGemmIsa.Avx2:
                    return QGemmAvx2Supported ? QGemmIsa.Avx2 : QGemmIsa.Legacy;
                default:
                    if (!QGemmEnabled || !QGemmAvx2Supported) return QGemmIsa.Legacy;
                    return QGemmAvx512Supported && !QGemmAvx512Disabled ? QGemmIsa.Avx512 : QGemmIsa.Avx2;
            }
        }

        private static QGemmFamily GetQGemmFamily(GgmlTensorType type, int inDim)
        {
            switch (type)
            {
                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                    return inDim % QK_K == 0 ? QGemmFamily.KMin : QGemmFamily.None;
                case GgmlTensorType.Q6_K:
                    return inDim % QK_K == 0 ? QGemmFamily.KOfs : QGemmFamily.None;
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q5_0:
                    return inDim % QK8_0 == 0 ? QGemmFamily.Q0Unsigned : QGemmFamily.None;
                case GgmlTensorType.Q8_0:
                    return inDim % QK8_0 == 0 ? QGemmFamily.Q0Signed : QGemmFamily.None;
                default:
                    return QGemmFamily.None;
            }
        }

        private static bool IsKFamily(QGemmFamily family) => family == QGemmFamily.KMin || family == QGemmFamily.KOfs;

        // GEMM activation row: int8 values [inDim] | per-block side data, padded
        // to 64 bytes so every row (and every 32-value chunk) stays aligned.
        //   K family : aux[nsb] x 32 B (KMin: 8 floats d8*bsum-pair, KOfs: 16 int16 bsums) | d8[nsb] floats
        //   Q0 family: dx[nb] floats (fp16-rounded block scale) | sxAdj[nb] int32 (zero point * block sum / 2)
        private static int QGemmActRowBytes(QGemmFamily family, int inDim)
        {
            long bytes = IsKFamily(family)
                ? inDim + (long)(inDim / QK_K) * 36
                : inDim + (long)(inDim / QK8_0) * 8;
            return checked((int)((bytes + 63) & ~63L));
        }

        /// <summary>Scratch bytes one decoded column pair needs.</summary>
        private static int QGemmPairScratchBytes(QGemmFamily family, int inDim)
            => IsKFamily(family) ? inDim / QK_K * QKSbStride : Q0PairScratchBytes(inDim / QK8_0);

        // Per-thread, 64-byte aligned decode scratch. Pinned on the POH so the
        // address stays valid for the lifetime of the array, and reclaimed with
        // the thread (the pool's workers live for the process).
        [ThreadStatic] private static byte[] _qgemmScratch;

        private static unsafe byte* QGemmScratch(int bytes)
        {
            byte[] buf = _qgemmScratch;
            if (buf == null || buf.Length < bytes + 64)
            {
                buf = GC.AllocateUninitializedArray<byte>(Math.Max(bytes + 64, 64 * 1024), pinned: true);
                _qgemmScratch = buf;
            }
            byte* p = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buf));
            return (byte*)(((nint)p + 63) & ~(nint)63);
        }

        /// <summary>
        /// Multi-row GEMM entry: <c>output[r, c] = dot(weightRow(c), input[r])</c>
        /// for the types the tiled kernels cover. Returns false (and writes
        /// nothing) when the type/shape/ISA is not handled, so the caller can run
        /// the per-row path.
        /// </summary>
        internal static unsafe bool TryQGemm(
            int ggmlType,
            IntPtr weights,
            int inDim,
            int outDim,
            float* input,
            int inputRowStride,
            int rowCount,
            float* output,
            int outputRowStride,
            ParallelOptions options,
            QGemmIsa isa)
        {
            var type = (GgmlTensorType)ggmlType;
            QGemmFamily family = GetQGemmFamily(type, inDim);
            if (family == QGemmFamily.None || rowCount <= 0 || outDim <= 0)
                return false;
            isa = ResolveQGemmIsa(isa);
            if (isa == QGemmIsa.Legacy)
                return false;

            int actStride = QGemmActRowBytes(family, inDim);
            long actBytes = (long)rowCount * actStride + 64;
            if (actBytes > int.MaxValue)
                return false;

            int dop = ResolveDop(options);
            byte[] rented = ArrayPool<byte>.Shared.Rent((int)actBytes);
            try
            {
                fixed (byte* rentedBase = rented)
                {
                    byte* act = (byte*)(((nint)rentedBase + 63) & ~(nint)63);
                    QuantizeRowsQGemm(type, family, input, inputRowStride, rowCount, inDim, act, actStride, options, dop);

                    int rowBytes = (int)RowSize(ggmlType, inDim);
                    QGemmPlan plan = PlanQGemm(rowCount, inDim, outDim, rowBytes, actStride, dop);
                    nint w = weights, a = (nint)act, o = (nint)output;
                    bool avx512 = isa == QGemmIsa.Avx512;
                    int outStride = outputRowStride;
                    void RunTask(int t)
                    {
                        int cb = t % plan.ColBlocks;
                        int rb = t / plan.ColBlocks;
                        int col0 = cb * plan.PairsPerBlock * 2;
                        int col1 = Math.Min(outDim, col0 + plan.PairsPerBlock * 2);
                        int row0 = rb * plan.RowsPerBlock;
                        int row1 = Math.Min(rowCount, row0 + plan.RowsPerBlock);
                        byte* scratch = QGemmScratch(QGemmPairScratchBytes(family, inDim));
                        QGemmTile(type, family, avx512, (byte*)w, rowBytes, inDim, outDim,
                            (byte*)a, actStride, (float*)o, outStride, col0, col1, row0, row1, scratch);
                    }

                    int tasks = plan.RowBlocks * plan.ColBlocks;
                    if (tasks > 1 && dop > 1)
                        RunParallelBlocks(tasks, options, RunTask);
                    else
                        for (int t = 0; t < tasks; t++) RunTask(t);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
            return true;
        }

        /// <summary>TS_CPU_QGEMM_VERIFY: compare a finished GEMM with the per-row path.</summary>
        private static unsafe void VerifyQGemmAgainstLegacy(
            string what, int ggmlType, IntPtr weights, int inDim, int outDim, float* input, int inputRowStride,
            int rowCount, float* output, int outputRowStride, bool floatPanel)
        {
            float[] reference = new float[(long)rowCount * outDim];
            fixed (float* r = reference)
            {
                if (floatPanel)
                {
                    long rowBytes = RowSize(ggmlType, inDim);
                    float[] scratch = new float[inDim];
                    fixed (float* s = scratch)
                        DequantMatMulColumns(ggmlType, (byte*)weights, rowBytes, inDim, outDim,
                            input, inputRowStride, rowCount, r, outDim, 0, outDim, s);
                }
                else
                {
                    TryAddmmQuantizedToFloat32(ggmlType, weights, inDim, outDim, input, inputRowStride, rowCount,
                        r, outDim, null, QGemmIsa.Legacy);
                }
            }
            double maxRef = 1e-30, maxDiff = 0;
            for (int row = 0; row < rowCount; row++)
                for (int c = 0; c < outDim; c++)
                {
                    double e = reference[(long)row * outDim + c];
                    double a = output[(long)row * outputRowStride + c];
                    maxRef = Math.Max(maxRef, Math.Abs(e));
                    maxDiff = Math.Max(maxDiff, double.IsNaN(a) || double.IsNaN(e) ? double.PositiveInfinity : Math.Abs(a - e));
                }
            double rel = maxDiff / maxRef;
            lock (QGemmVerifyLock)
            {
                if (rel > _qgemmVerifyWorst || rel > 1e-4)
                {
                    _qgemmVerifyWorst = Math.Max(_qgemmVerifyWorst, rel);
                    Console.Error.WriteLine(
                        $"[qgemm-verify] {what} {(GgmlTensorType)ggmlType} M={rowCount} K={inDim} N={outDim}: " +
                        $"max|new-legacy|/max|legacy| = {rel:E2} (worst so far {_qgemmVerifyWorst:E2})");
                }
            }
        }

        private readonly struct QGemmPlan
        {
            public readonly int RowsPerBlock;
            public readonly int RowBlocks;
            public readonly int PairsPerBlock;
            public readonly int ColBlocks;

            public QGemmPlan(int rowsPerBlock, int rowBlocks, int pairsPerBlock, int colBlocks)
            {
                RowsPerBlock = rowsPerBlock;
                RowBlocks = rowBlocks;
                PairsPerBlock = pairsPerBlock;
                ColBlocks = colBlocks;
            }
        }

        /// <summary>
        /// How many tasks a matmul is worth: enough arithmetic per task to bury
        /// the dispatch, or - for the few-row, bandwidth-bound shapes - enough
        /// weight bytes per task by the same rule as ParallelColumnBlock; never
        /// more than a few per worker.
        /// </summary>
        private static long QGemmWantTasks(long macs, long weightBytes, long rows, int dop)
        {
            if (dop <= 1) return 1;
            long perTaskBytes = Math.Max(MinBytesPerParallelTaskFloor, MinBytesPerParallelTask / Math.Clamp(rows, 1, 8));
            long byWork = Math.Max(macs / Math.Max(1, QGemmMinTaskMacs), weightBytes / Math.Max(1, perTaskBytes));
            return Math.Clamp(byWork, 1, (long)dop * TasksPerWorker);
        }

        /// <summary>
        /// Tile the (rows x column pairs) space. Rows are cut only when the
        /// activation block would fall out of L2 (every decoded pair re-reads
        /// it); columns are cut so there are enough tasks for the pool, sized by
        /// work rather than by thread count (see ParallelColumnBlock).
        /// </summary>
        private static QGemmPlan PlanQGemm(int rowCount, int inDim, int outDim, long weightRowBytes, int actStride, int dop)
        {
            int pairs = (outDim + 1) / 2;
            int rowsPerBlock = rowCount;
            long blockRows = QGemmRowBlockBytes / Math.Max(1, actStride);
            if (blockRows < rowCount)
                rowsPerBlock = (int)Math.Max(8, blockRows & ~7L);
            int rowBlocks = (rowCount + rowsPerBlock - 1) / rowsPerBlock;

            long wantTasks = QGemmWantTasks((long)rowCount * outDim * inDim, weightRowBytes * outDim, rowCount, dop);
            long colBlocks = Math.Clamp((wantTasks + rowBlocks - 1) / rowBlocks, 1, pairs);
            int pairsPerBlock = (int)((pairs + colBlocks - 1) / colBlocks);
            colBlocks = (pairs + pairsPerBlock - 1) / pairsPerBlock;
            return new QGemmPlan(rowsPerBlock, rowBlocks, pairsPerBlock, (int)colBlocks);
        }

        /// <summary>One task: decode each column pair of [col0, col1) once, then
        /// run it against every row tile of [row0, row1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QGemmTile(
            GgmlTensorType type, QGemmFamily family, bool avx512,
            byte* weights, int rowBytes, int inDim, int outDim,
            byte* act, int actStride, float* output, int outStride,
            int col0, int col1, int row0, int row1, byte* scratch)
        {
            int tileRows = avx512 ? 8 : 4;
            for (int c = col0; c < col1; c += 2)
            {
                bool second = c + 1 < outDim;
                byte* wa = weights + (long)c * rowBytes;
                DecodeQGemmPair(type, wa, second ? wa + rowBytes : wa, inDim, scratch);
                for (int r = row0; r < row1; r += tileRows)
                {
                    int rows = Math.Min(tileRows, row1 - r);
                    RunQGemmKernel(family, avx512, rows, scratch, act + (long)r * actStride, actStride, inDim,
                        output + (long)r * outStride + c, outStride, second);
                }
            }
        }

        // ------------------------------------------------------------------
        // Batched (MoE) GEMM: every job's column pairs in one flat task space.
        // ------------------------------------------------------------------

        private static unsafe bool TryQGemmBatch(
            GgmlTensorType type,
            int inDim,
            int inputRowStride,
            ReadOnlySpan<QuantMatMulJob> jobs,
            ParallelOptions options,
            QGemmIsa isa)
        {
            QGemmFamily family = GetQGemmFamily(type, inDim);
            if (family == QGemmFamily.None)
                return false;
            isa = ResolveQGemmIsa(isa);
            if (isa == QGemmIsa.Legacy)
                return false;

            // --- quantize each DISTINCT input block once (same rule as the per-row batch) ---
            int n = jobs.Length;
            var actOf = new int[n];
            var blockInput = new IntPtr[n];
            var blockRows = new int[n];
            var blockOffset = new long[n];
            int nBlocks = 0;
            long totalRows = 0;
            for (int j = 0; j < n; j++)
            {
                int found = -1;
                for (int b = 0; b < nBlocks; b++)
                    if (blockInput[b] == jobs[j].Input && blockRows[b] == jobs[j].RowCount)
                    {
                        found = b;
                        break;
                    }
                if (found < 0)
                {
                    found = nBlocks++;
                    blockInput[found] = jobs[j].Input;
                    blockRows[found] = jobs[j].RowCount;
                    blockOffset[found] = totalRows;
                    totalRows += jobs[j].RowCount;
                }
                actOf[j] = found;
            }

            int actStride = QGemmActRowBytes(family, inDim);
            long actBytes = totalRows * actStride + 64;
            if (actBytes > int.MaxValue)
                return false;

            int dop = ResolveDop(options);
            byte[] rented = ArrayPool<byte>.Shared.Rent((int)Math.Max(64, actBytes));
            try
            {
                fixed (byte* rentedBase = rented)
                {
                    byte* act = (byte*)(((nint)rentedBase + 63) & ~(nint)63);
                    if (dop > 1 && totalRows >= 8 && totalRows * inDim >= 64 * 1024)
                    {
                        // A prefill-sized MoE batch gathers thousands of rows; quantize
                        // them in parallel, flattened across the distinct input blocks.
                        var rowSrc = new nint[totalRows];
                        for (int b = 0; b < nBlocks; b++)
                            for (int row = 0; row < blockRows[b]; row++)
                                rowSrc[blockOffset[b] + row] = blockInput[b] + (nint)((long)row * inputRowStride * sizeof(float));
                        int rowsPerTask = (int)Math.Max(4, (totalRows + dop * TasksPerWorker - 1) / (dop * TasksPerWorker));
                        int quantTasks = (int)((totalRows + rowsPerTask - 1) / rowsPerTask);
                        nint dstBase = (nint)act;
                        int width = inDim;
                        RunParallelBlocks(quantTasks, options, t =>
                        {
                            long r1 = Math.Min(rowSrc.Length, (long)(t + 1) * rowsPerTask);
                            for (long r = (long)t * rowsPerTask; r < r1; r++)
                                QuantizeRowQGemm(type, family, (float*)rowSrc[r], (byte*)dstBase + r * actStride, width);
                        });
                    }
                    else
                    {
                        for (int b = 0; b < nBlocks; b++)
                        {
                            float* src = (float*)blockInput[b];
                            byte* dst = act + blockOffset[b] * actStride;
                            for (int row = 0; row < blockRows[b]; row++)
                                QuantizeRowQGemm(type, family, src + (long)row * inputRowStride, dst + (long)row * actStride, inDim);
                        }
                    }

                    // Pairs per task from the total work, as in PlanQGemm.
                    int rowBytes = (int)RowSize((int)type, inDim);
                    long macs = 0, totalPairs = 0, totalCols = 0;
                    for (int j = 0; j < n; j++)
                    {
                        if (jobs[j].RowCount <= 0) continue;
                        macs += (long)jobs[j].RowCount * jobs[j].OutDim * inDim;
                        totalPairs += (jobs[j].OutDim + 1) / 2;
                        totalCols += jobs[j].OutDim;
                    }
                    // Rows per weight byte = the jobs' average row count.
                    long avgRows = macs / Math.Max(1, totalCols * inDim);
                    long wantTasks = QGemmWantTasks(macs, totalCols * rowBytes, avgRows, dop);
                    int pairsPerTask = (int)Math.Max(1, (totalPairs + wantTasks - 1) / wantTasks);

                    var taskStart = new int[n + 1];
                    int totalTasks = 0;
                    for (int j = 0; j < n; j++)
                    {
                        taskStart[j] = totalTasks;
                        int jobPairs = (jobs[j].OutDim + 1) / 2;
                        if (jobs[j].RowCount > 0)
                            totalTasks += (jobPairs + pairsPerTask - 1) / pairsPerTask;
                    }
                    taskStart[n] = totalTasks;

                    // Span cannot cross the lambda, so copy the jobs' raw fields.
                    var wPtr = new nint[n];
                    var oPtr = new nint[n];
                    var outDims = new int[n];
                    var rowCnt = new int[n];
                    var outStrides = new int[n];
                    var actOff = new long[n];
                    for (int j = 0; j < n; j++)
                    {
                        wPtr[j] = jobs[j].Weights;
                        oPtr[j] = jobs[j].Output;
                        outDims[j] = jobs[j].OutDim;
                        rowCnt[j] = jobs[j].RowCount;
                        outStrides[j] = jobs[j].OutputRowStride;
                        actOff[j] = blockOffset[actOf[j]] * actStride;
                    }
                    nint actAddr = (nint)act;
                    bool avx512 = isa == QGemmIsa.Avx512;
                    int scratchBytes = QGemmPairScratchBytes(family, inDim);

                    void RunTask(int t)
                    {
                        int j = 0;
                        while (j + 1 < n && taskStart[j + 1] <= t) j++;
                        int local = t - taskStart[j];
                        int col0 = local * pairsPerTask * 2;
                        int col1 = Math.Min(outDims[j], col0 + pairsPerTask * 2);
                        byte* scratch = QGemmScratch(scratchBytes);
                        QGemmTile(type, family, avx512, (byte*)wPtr[j], rowBytes, inDim, outDims[j],
                            (byte*)actAddr + actOff[j], actStride, (float*)oPtr[j], outStrides[j],
                            col0, col1, 0, rowCnt[j], scratch);
                    }

                    if (totalTasks > 1 && dop > 1)
                        RunParallelBlocks(totalTasks, options, RunTask);
                    else
                        for (int t = 0; t < totalTasks; t++) RunTask(t);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
            return true;
        }

        // ------------------------------------------------------------------
        // Activation quantization into the GEMM layout.
        // ------------------------------------------------------------------

        private static unsafe void QuantizeRowsQGemm(
            GgmlTensorType type, QGemmFamily family, float* input, int inputRowStride, int rowCount, int inDim,
            byte* act, int actStride, ParallelOptions options, int dop)
        {
            // Rows are independent; hand them out in chunks sized by elements so a
            // 4096-row prefill is a few dozen tasks, not thousands.
            long elements = (long)rowCount * inDim;
            if (dop > 1 && rowCount >= 8 && elements >= 64 * 1024)
            {
                int rowsPerTask = (int)Math.Max(4, Math.Min(rowCount, (32 * 1024 + inDim - 1) / inDim));
                int tasks = Math.Min((rowCount + rowsPerTask - 1) / rowsPerTask, dop * TasksPerWorker);
                rowsPerTask = (rowCount + tasks - 1) / tasks;
                nint src = (nint)input, dst = (nint)act;
                RunParallelBlocks(tasks, options, t =>
                {
                    int r0 = t * rowsPerTask, r1 = Math.Min(rowCount, r0 + rowsPerTask);
                    for (int r = r0; r < r1; r++)
                        QuantizeRowQGemm(type, family, (float*)src + (long)r * inputRowStride,
                            (byte*)dst + (long)r * actStride, inDim);
                });
                return;
            }
            for (int r = 0; r < rowCount; r++)
                QuantizeRowQGemm(type, family, input + (long)r * inputRowStride, act + (long)r * actStride, inDim);
        }

        /// <summary>
        /// Quantize one activation row into the GEMM layout. The int8 values and
        /// scales are bit-identical to QuantizeF32ToQ8_K / QuantizeF32ToQ8_0
        /// (same max-abs scale, same round-half-to-even, same +-127 clamp).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QuantizeRowQGemm(GgmlTensorType type, QGemmFamily family, float* src, byte* dst, int inDim)
        {
            sbyte* qs = (sbyte*)dst;
            if (IsKFamily(family))
            {
                int nsb = inDim / QK_K;
                byte* aux = dst + inDim;
                float* d8 = (float*)(aux + nsb * 32);
                int* groupSums = stackalloc int[QK_K / 16];
                for (int sb = 0; sb < nsb; sb++)
                {
                    float* x = src + sb * QK_K;
                    float maxAbs = MaxAbs(x, QK_K);
                    float scale = maxAbs / 127.0f;
                    d8[sb] = scale;
                    sbyte* q = qs + sb * QK_K;
                    byte* a = aux + sb * 32;
                    if (scale == 0.0f)
                    {
                        Unsafe.InitBlockUnaligned(q, 0, QK_K);
                        Unsafe.InitBlockUnaligned(a, 0, 32);
                        continue;
                    }

                    QuantizeInt8Groups16(x, 1.0f / scale, q, QK_K, groupSums);
                    if (family == QGemmFamily.KMin)
                    {
                        float* bsF = (float*)a;
                        for (int j = 0; j < 8; j++)
                            bsF[j] = scale * (groupSums[2 * j] + groupSums[2 * j + 1]);
                    }
                    else
                    {
                        short* bs = (short*)a;
                        for (int g = 0; g < 16; g++)
                            bs[g] = (short)groupSums[g];
                    }
                }
                return;
            }

            int nb = inDim / QK8_0;
            float* dx = (float*)(dst + inDim);
            int* sxAdj = (int*)(dst + inDim + nb * 4);
            // Q4_0's zero point is 8 and Q5_0's 16. The kernels fold it in as
            // an integer correction: their four-block reduction leaves two
            // int32 lanes per (column, block), so each lane gets half of
            // zeroPoint * sum(x); the single-block tail (eight lanes) a quarter
            // of that.
            int zeroShare = type == GgmlTensorType.Q5_0 ? 8 : 4;
            int* groupSums2 = stackalloc int[2];
            for (int b = 0; b < nb; b++)
            {
                float* x = src + b * QK8_0;
                float maxAbs = MaxAbs(x, QK8_0);
                float scale = maxAbs / 127.0f;
                // The per-row kernels read the block scale back from its fp16
                // encoding, so the GEMM uses that same rounded value.
                dx[b] = (float)(System.Half)scale;
                sbyte* q = qs + b * QK8_0;
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(q, 0, QK8_0);
                    sxAdj[b] = 0;
                    continue;
                }

                QuantizeInt8Groups16(x, 1.0f / scale, q, QK8_0, groupSums2);
                sxAdj[b] = zeroShare * (groupSums2[0] + groupSums2[1]);
            }
        }

        /// <summary>
        /// <c>q[i] = clamp(round_half_even(x[i] * invScale), -127, 127)</c> for
        /// <paramref name="n"/> (a multiple of 16) values, plus the sum of each
        /// 16-value group. Bit-identical to the scalar
        /// <c>ClampToInt8(MathF.Round(x * invScale))</c>: the product is the same
        /// IEEE multiply, vroundps/vrndscaleps round half to even like
        /// MathF.Round, and the rounded value converts to int32 exactly.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void QuantizeInt8Groups16(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            if (Avx512F.IsSupported && Avx512BW.IsSupported && !QGemmAvx512Disabled)
                QuantizeInt8Groups16Avx512(x, invScale, q, n, groupSums);
            else if (Avx2.IsSupported)
                QuantizeInt8Groups16Avx2(x, invScale, q, n, groupSums);
            else
                QuantizeInt8Groups16Scalar(x, invScale, q, n, groupSums);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void QuantizeInt8Groups16Avx512(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            Vector512<float> inv = Vector512.Create(invScale);
            Vector512<int> lo = Vector512.Create(-127), hi = Vector512.Create(127);
            for (int i = 0, g = 0; i < n; i += 16, g++)
            {
                Vector512<float> v = Avx512F.Multiply(Avx512F.LoadVector512(x + i), inv);
                Vector512<int> r = Avx512F.ConvertToVector512Int32(Avx512F.RoundScale(v, 0));
                r = Avx512F.Min(Avx512F.Max(r, lo), hi);
                Avx512F.ConvertToVector128SByte(r).Store(q + i);
                groupSums[g] = Vector512.Sum(r);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void QuantizeInt8Groups16Avx2(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            Vector256<float> inv = Vector256.Create(invScale);
            Vector256<int> lo = Vector256.Create(-127), hi = Vector256.Create(127);
            for (int i = 0, g = 0; i < n; i += 16, g++)
            {
                Vector256<int> r0 = Avx.ConvertToVector256Int32(
                    Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i), inv)));
                Vector256<int> r1 = Avx.ConvertToVector256Int32(
                    Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i + 8), inv)));
                r0 = Avx2.Min(Avx2.Max(r0, lo), hi);
                r1 = Avx2.Min(Avx2.Max(r1, lo), hi);
                // int32 -> int16 -> int8 packs work per 128-bit lane:
                // [r0.lo r1.lo | r0.hi r1.hi] -> bytes need a dword permute.
                Vector256<short> s = Avx2.PackSignedSaturate(r0, r1);
                Vector256<sbyte> b = Avx2.PackSignedSaturate(s, s);
                b = Avx2.PermuteVar8x32(b.AsInt32(), Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7)).AsSByte();
                b.GetLower().Store(q + i);
                groupSums[g] = Vector256.Sum(Avx2.Add(r0, r1));
            }
        }

        internal static unsafe void QuantizeInt8Groups16Scalar(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            for (int i = 0, g = 0; i < n; i += 16, g++)
            {
                int sum = 0;
                for (int k = 0; k < 16; k++)
                {
                    sbyte v = ClampToInt8(MathF.Round(x[i + k] * invScale));
                    q[i + k] = v;
                    sum += v;
                }
                groupSums[g] = sum;
            }
        }

        // ---- test hooks ----------------------------------------------------

        /// <summary>GEMM activation row size for <paramref name="type"/>, or 0 when
        /// the type has no multi-row kernel.</summary>
        internal static int QGemmActivationRowBytes(GgmlTensorType type, int inDim)
        {
            QGemmFamily family = GetQGemmFamily(type, inDim);
            return family == QGemmFamily.None ? 0 : QGemmActRowBytes(family, inDim);
        }

        /// <summary>Quantize one row into the GEMM layout (see QuantizeRowQGemm).</summary>
        internal static unsafe void QGemmQuantizeActivationRow(GgmlTensorType type, float* src, byte* dst, int inDim)
            => QuantizeRowQGemm(type, GetQGemmFamily(type, inDim), src, dst, inDim);
    }
}
