// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Packed, register-blocked F32 GEMM for the pure-C# (BackendType.Cpu) Qwen-Image-2.1 VAE and
// vision tower: C[m, n] = init + sum_k A[m, k] * B[k, n].
//
// GotoBLAS structure. A is packed once into MR-row panels ([K][MR] per panel) - the VAE and
// vision weights are packed once and cached, so the pack is free at run time. B is produced
// on the fly, KC rows x one N tile at a time, into NR-column panels ([KC][NR] per panel) in a
// thread-local scratch that stays in L2; for a convolution that pack IS the im2col, so no
// im2col matrix is ever materialized (a 2048x2048 decoder conv would need 10+ GB). The
// micro-kernel keeps an MR x NR tile of C in registers across a KC chunk and streams one
// broadcast of A and NR/lanes loads of B per k:
//
//     AVX-512 : 12 x 32 (24 zmm accumulators, 2 B loads, broadcast folded into the FMA)
//     AVX2+FMA:  6 x 16 (12 ymm accumulators; 16 registers leave no room for more)
//     portable:  4 x 8  (Vector128: SSE / AdvSimd)
//
// This client part has one 512-bit FMA port, so AVX-512 and AVX2 have the same peak here;
// AVX-512 wins on fewer instructions and a larger C tile (less B traffic per FMA).
// TS_CPU_DISABLE_AVX512=1 selects the AVX2 kernel so it can be A/B'd on AVX-512 hardware.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Models.QwenImage
{
    internal enum CpuGemmIsa { Portable, Avx2, Avx512 }

    /// <summary>Source of packed NR-column panels of B for rows [k0, k0+kc) and columns
    /// [n0, n0+count). Either packs into <paramref name="scratch"/> (panel stride kc*NR) or
    /// returns a pointer into data packed ahead of time (its own panel stride).</summary>
    internal unsafe interface IGemmPanelSource
    {
        float* Panels(int k0, int kc, int n0, int count, int nr, float* scratch, out long panelStride);

        /// <summary>Every K chunk the driver asks for starts and ends on a multiple of this
        /// (a quantized row decodes whole blocks); K itself must be a multiple of it.</summary>
        int KAlignment { get; }
    }

    /// <summary>A matrix packed into MR-row (A) or NR-column (B) panels, 64-byte aligned on the
    /// pinned object heap so the GC never moves it and the pointer stays valid.</summary>
    internal sealed unsafe class PackedPanels
    {
        private readonly float[] _raw;
        internal readonly nint Base;
        internal readonly int Rows;     // M (A) or N (B), unpadded
        internal readonly int K;
        internal readonly int Panel;    // MR (A) or NR (B)
        internal readonly int Panels;
        internal readonly CpuGemmIsa Isa;

        /// <summary>A pack of the given shape. With <paramref name="reuse"/> the storage of that
        /// earlier pack is taken over when it is large enough (a per-layer operand repacked
        /// every call); the earlier pack must no longer be read.</summary>
        internal PackedPanels(int rows, int k, int panel, CpuGemmIsa isa, PackedPanels reuse = null)
        {
            Rows = rows; K = k; Panel = panel; Isa = isa;
            Panels = (rows + panel - 1) / panel;
            long floats = (long)Panels * panel * k;
            _raw = reuse != null && reuse._raw.LongLength >= floats + 16
                ? reuse._raw
                : GC.AllocateUninitializedArray<float>(checked((int)(floats + 16)), pinned: true);
            nint p = (nint)Unsafe.AsPointer(ref _raw[0]);
            Base = (p + 63) & ~(nint)63;
        }

        internal float* Ptr => (float*)Base;
        internal long Floats => (long)Panels * Panel * K;
    }

    internal static unsafe class CpuPackedGemm
    {
        /// <summary>The kernel family new packs are made for. Settable so tests can exercise every
        /// kernel on one machine; a pack remembers its ISA, so changing this never mixes shapes.</summary>
        internal static CpuGemmIsa Isa { get; set; } = DetectIsa();

        internal static CpuGemmIsa DetectIsa()
        {
            bool no512 = Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") == "1";
            if (!no512 && Avx512F.IsSupported && Vector512.IsHardwareAccelerated) return CpuGemmIsa.Avx512;
            if (Avx2.IsSupported && Fma.IsSupported) return CpuGemmIsa.Avx2;
            return CpuGemmIsa.Portable;
        }

        internal static bool IsaSupported(CpuGemmIsa isa) => isa switch
        {
            CpuGemmIsa.Avx512 => Avx512F.IsSupported && Vector512.IsHardwareAccelerated,
            CpuGemmIsa.Avx2 => Avx2.IsSupported && Fma.IsSupported,
            _ => true,
        };

        internal static int Mr(CpuGemmIsa isa) => isa switch { CpuGemmIsa.Avx512 => 12, CpuGemmIsa.Avx2 => 6, _ => 4 };
        internal static int Nr(CpuGemmIsa isa) => isa switch { CpuGemmIsa.Avx512 => 32, CpuGemmIsa.Avx2 => 16, _ => 8 };

        // KC x NT floats of B are packed per task: 256 x 256 x 4 B = 256 KB sits in the 1.25 MB L2
        // next to the streamed A panels, and a KC of 256 keeps an A panel (12 x 256 x 4 = 12 KB)
        // plus one B panel (32 KB) inside the 48 KB L1 for the micro-kernel. The overrides are
        // A/B knobs, clamped so a stray value cannot size a multi-GB scratch. KcBlock is settable
        // for tests; the result never depends on it (C round-trips through memory as the same
        // float between K chunks).
        internal static int KcBlock { get; set; } = Math.Clamp(EnvInt("TS_CPU_GEMM_KC", 256), 16, 4096);
        private static readonly int NtBlock = Math.Clamp(EnvInt("TS_CPU_GEMM_NT", 256), 32, 2048);

        // Worker pools. By default the kernels run on CpuWorkerPool.Shared (cores/2 here), like
        // the rest of the pure-C# backend. A caller whose whole pipeline is these kernels (the
        // VAE: convolutions plus its own elementwise passes) can pass WidePool, one thread per
        // logical processor: the conv GEMMs are FMA-bound, and two SMT threads per core keep the
        // single 512-bit FMA port busier (512x512 decode 7.9 -> 7.0 s on the 8-core/16-thread
        // i7-11800H). Mixed with ThreadPool work (text encoder, vision tower) the extra spinning
        // workers cost more than they give (TE 1.6 -> 1.8 s, vision 0.72 -> 0.97 s), so those
        // stay on the shared pool. Default width: every logical CPU, at most 64;
        // TS_CPU_GEMM_THREADS overrides it within the pool's own limit (1..512).
        private static readonly Lazy<CpuWorkerPool> s_widePool = new(() => new CpuWorkerPool(
            Math.Clamp(EnvInt("TS_CPU_GEMM_THREADS", Math.Min(Environment.ProcessorCount, 64)), 1, 512)));

        internal static CpuWorkerPool WidePool => s_widePool.Value;

        // Times the driver packed all of B once and shared it between row blocks (a test hook:
        // the path only runs at shapes a unit test has to aim for).
        private static long s_sharedPrepackRuns;
        internal static long SharedPrepackRuns => System.Threading.Interlocked.Read(ref s_sharedPrepackRuns);

        private static int EnvInt(string name, int fallback)
            => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;

        // ---- packing -----------------------------------------------------------------

        /// <summary>Pack A[m, k] (element (i, j) at src[i*strideM + j*strideK]) into MR-row panels
        /// (into <paramref name="reuse"/>'s storage when it is large enough; see PackedPanels).
        /// With <paramref name="serial"/> the pack runs on the calling thread (a pool task).</summary>
        internal static PackedPanels PackA(float* src, int m, int k, long strideM, long strideK, CpuGemmIsa isa,
            CpuWorkerPool pool = null, PackedPanels reuse = null, bool serial = false)
        {
            var packed = new PackedPanels(m, k, Mr(isa), isa, reuse);
            int mr = packed.Panel;
            nint srcL = (nint)src, dstL = packed.Base;
            void PackPanel(int p)
            {
                float* s = (float*)srcL;
                float* d = (float*)dstL + (long)p * k * mr;
                int r0 = p * mr, rows = Math.Min(mr, m - r0);
                if (strideM == 1)
                {
                    // Column access is contiguous (e.g. Q^T from a planar [C, hw] map).
                    for (int j = 0; j < k; j++)
                    {
                        float* sj = s + j * strideK + r0;
                        float* dj = d + (long)j * mr;
                        int r = 0;
                        for (; r < rows; r++) dj[r] = sj[r];
                        for (; r < mr; r++) dj[r] = 0f;
                    }
                }
                else
                {
                    for (int r = 0; r < mr; r++)
                    {
                        float* dr = d + r;
                        if (r >= rows) { for (int j = 0; j < k; j++) dr[(long)j * mr] = 0f; continue; }
                        float* sr = s + (r0 + r) * strideM;
                        for (int j = 0; j < k; j++) dr[(long)j * mr] = sr[j * strideK];
                    }
                }
            }
            ForEach(packed.Panels, !serial && (long)m * k >= 1 << 18, PackPanel, pool);
            return packed;
        }

        /// <summary>Pack B[k, n] (element (j, i) at src[j*strideK + i*strideN]) into NR-column panels
        /// over the full K, for a B operand that is reused across calls (a linear layer's W^T).</summary>
        internal static PackedPanels PackB(float* src, int k, int n, long strideK, long strideN, CpuGemmIsa isa,
            CpuWorkerPool pool = null)
        {
            var packed = new PackedPanels(n, k, Nr(isa), isa);
            int nr = packed.Panel;
            nint srcL = (nint)src, dstL = packed.Base;
            ForEach(packed.Panels, (long)n * k >= 1 << 18, p =>
            {
                var source = new StridedPanelSource((float*)srcL, strideK, strideN, n);
                source.Pack(0, k, p * nr, nr, (float*)dstL + (long)p * k * nr);
            }, pool);
            return packed;
        }

        // ---- driver -------------------------------------------------------------------

        /// <summary>
        /// C = init + A x B over A's rows [panel0*MR, (panel0+panelCount)*MR) and B's n columns;
        /// <paramref name="c"/> points at the first of those rows (row stride ldc). init: the
        /// existing C when <paramref name="accumulate"/>, else biasM[row] (per absolute A row),
        /// biasN[col] (per column) or zero. The row sub-range lets a caller block over A.
        /// <paramref name="serial"/> runs the whole product on the calling thread with no shared
        /// state, for callers that parallelize over many small GEMMs themselves (attention heads).
        /// </summary>
        internal static void Gemm<TSource>(PackedPanels a, TSource b, int n, float* c, long ldc,
            float* biasM = null, float* biasN = null, bool accumulate = false,
            int panel0 = 0, int panelCount = -1, CpuWorkerPool pool = null, bool serial = false)
            where TSource : struct, IGemmPanelSource
        {
            pool ??= CpuWorkerPool.Shared;
            CpuGemmIsa isa = a.Isa;
            int mr = a.Panel, nr = Nr(isa), k = a.K, m = a.Rows;
            if (panelCount < 0) panelCount = a.Panels - panel0;
            if (n <= 0 || panelCount <= 0) return;
            if (k == 0)
            {
                for (int p = panel0; p < panel0 + panelCount; p++)
                    for (int r = p * mr; r < Math.Min(m, (p + 1) * mr); r++)
                        if (!accumulate)
                            for (int j = 0; j < n; j++)
                                c[(r - panel0 * mr) * ldc + j] = (biasM != null ? biasM[r] : 0f) + (biasN != null ? biasN[j] : 0f);
                return;
            }

            int ntPanels = Math.Max(1, NtBlock / nr);
            int nPanels = (n + nr - 1) / nr;
            int nTiles = (nPanels + ntPanels - 1) / ntPanels;
            int threads = serial ? 1 : pool.ThreadCount;
            // Narrow the column tiles when even the widest row split leaves threads idle
            // (a 1024-wide projection of a 40-token prompt is 4 tiles of 256 and one row block).
            while (!serial && ntPanels > 1 && nTiles * Math.Max(1, panelCount / 4) < 2 * threads)
            {
                ntPanels = (ntPanels + 1) / 2;
                nTiles = (nPanels + ntPanels - 1) / ntPanels;
            }
            // Split rows too when there are too few pixel tiles to feed every thread (the 16x16
            // and 32x32 decoder stages have 1-4 tiles but 1152 output channels). Keep >= 4 A
            // panels per task so a B tile is shared by 48+ rows.
            int mBlocks = 1;
            int wanted = threads * 3;
            if (!serial && nTiles < wanted)
                mBlocks = Math.Max(1, Math.Min((wanted + nTiles - 1) / nTiles, panelCount / 4));
            int mbPanels = (panelCount + mBlocks - 1) / mBlocks;
            mBlocks = (panelCount + mbPanels - 1) / mbPanels;
            int tasks = nTiles * mBlocks;
            // Even K chunks: 1296 = 5 x 256 + 16 would pay a whole chunk's C traffic for 16 k.
            // Then rounded up to the source's block alignment (a quantized row decodes whole
            // 256-value super-blocks, whatever TS_CPU_GEMM_KC says).
            int kChunks = (k + KcBlock - 1) / KcBlock;
            int kc = (k + kChunks - 1) / kChunks;
            int align = Math.Max(1, b.KAlignment);
            if (align > 1)
            {
                kc = (int)Math.Min(k, ((long)kc + align - 1) / align * align);
                kChunks = (k + kc - 1) / kc;
            }
            bool parallel = !serial && tasks > 1 && (long)panelCount * mr * n * k >= 1L << 21;

            // When several row blocks share each B tile, pack all of B once, in parallel, and let
            // every task read the shared copy: re-packing per row block made the pack cost more
            // than the FMAs at 16x16 (24 tasks re-packing the same 10 MB).
            float[] shared = null;
            if (parallel && mBlocks >= 3 && b is not PrepackedPanelSource)
            {
                long floats = (long)nPanels * nr * k;
                if (floats * sizeof(float) <= PrepackBudgetBytes)
                {
                    System.Threading.Interlocked.Increment(ref s_sharedPrepackRuns);
                    shared = RentShared(floats);
                    nint sharedBase = Align64(shared);
                    TSource packer = b;
                    CpuPackedGemm.ForEach(nPanels * kChunks, true, u =>
                    {
                        int q = u / kChunks, chunk = u - q * kChunks;
                        int k0 = chunk * kc, kLen = Math.Min(kc, k - k0);
                        TSource local = packer;
                        local.Panels(k0, kLen, q * nr, Math.Min(nr, n - q * nr), nr,
                            (float*)sharedBase + ((long)q * k + k0) * nr, out _);
                    }, pool);
                    Run(a, new PrepackedPanelSource(sharedBase, k), n, c, ldc, biasM, biasN, accumulate, panel0, panelCount,
                        ntPanels, mBlocks, mbPanels, tasks, kc, parallel, pool);
                    ReturnShared(shared);
                    return;
                }
            }
            Run(a, b, n, c, ldc, biasM, biasN, accumulate, panel0, panelCount, ntPanels, mBlocks, mbPanels, tasks, kc, parallel, pool);
        }

        // B packed for the whole GEMM is capped at this; bigger products pack per task.
        private const long PrepackBudgetBytes = 128L * 1024 * 1024;

        private static void Run<TSource>(PackedPanels a, TSource b, int n, float* c, long ldc, float* biasM, float* biasN,
            bool accumulate, int panel0, int panelCount, int ntPanels, int mBlocks, int mbPanels, int tasks, int kc, bool parallel,
            CpuWorkerPool pool) where TSource : struct, IGemmPanelSource
        {
            CpuGemmIsa isa = a.Isa;
            int mr = a.Panel, nr = Nr(isa), k = a.K, m = a.Rows;
            int scratchFloats = kc * ntPanels * nr;
            nint aL = a.Base, cL = (nint)c, bmL = (nint)biasM, bnL = (nint)biasN;

            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            void Task(int t)
            {
                int tile = t / mBlocks, block = t - tile * mBlocks;
                int n0 = tile * ntPanels * nr;
                int count = Math.Min(ntPanels * nr, n - n0);
                int p0 = panel0 + block * mbPanels;
                int p1 = Math.Min(panel0 + panelCount, p0 + mbPanels);
                float* scratch = Scratch(scratchFloats);
                TSource source = b;
                for (int k0 = 0; k0 < k; k0 += kc)
                {
                    int kLen = Math.Min(kc, k - k0);
                    float* panels = source.Panels(k0, kLen, n0, count, nr, scratch, out long panelStride);
                    bool first = k0 == 0 && !accumulate;
                    for (int p = p0; p < p1; p++)
                    {
                        float* ap = (float*)aL + ((long)p * k + k0) * mr;
                        int row0 = p * mr, rows = Math.Min(mr, m - row0);
                        for (int q = 0; q * nr < count; q++)
                        {
                            int col0 = n0 + q * nr, cols = Math.Min(nr, n - col0);
                            float* cp = (float*)cL + (row0 - panel0 * mr) * ldc + col0;
                            float* bp = panels + q * panelStride;
                            if (rows == mr && cols == nr)
                            {
                                if (first) InitTile(cp, ldc, mr, nr, (float*)bmL, (float*)bnL, row0, col0);
                                Kernel(isa, ap, bp, kLen, cp, ldc);
                            }
                            else
                                EdgeTile(isa, ap, bp, kLen, cp, ldc, rows, cols, mr, nr,
                                    first, (float*)bmL, (float*)bnL, row0, col0);
                        }
                    }
                }
            }

            if (!parallel)
                for (int t = 0; t < tasks; t++) Task(t);
            else
                pool.For(tasks, Task);
        }

        private static void InitTile(float* c, long ldc, int mr, int nr, float* biasM, float* biasN, int row0, int col0)
        {
            for (int r = 0; r < mr; r++)
            {
                float* cr = c + r * ldc;
                float bm = biasM != null ? biasM[row0 + r] : 0f;
                if (biasN != null) for (int j = 0; j < nr; j++) cr[j] = bm + biasN[col0 + j];
                else new Span<float>(cr, nr).Fill(bm);
            }
        }

        // Partial edge tiles go through a stack copy so the kernel always sees a full,
        // in-bounds tile.
        private static void EdgeTile(CpuGemmIsa isa, float* a, float* b, int kc, float* c, long ldc,
            int rows, int cols, int mr, int nr, bool init, float* biasM, float* biasN, int row0, int col0)
        {
            // A short last row panel (a 37-token prompt is 3 x 12 + 1 rows) runs a narrower
            // kernel over just its rows instead of the full MR-row one on zero padding.
            int sub = isa == CpuGemmIsa.Avx512 ? 4 : isa == CpuGemmIsa.Avx2 ? 3 : 0;
            if (sub > 0 && cols == nr && rows <= mr - sub)
            {
                float* part = stackalloc float[4 * 32];
                for (int r = 0; r < rows; r += sub)
                {
                    int n = Math.Min(sub, rows - r);
                    bool direct = n == sub;
                    float* cp = direct ? c + r * ldc : part;
                    long ld = direct ? ldc : nr;
                    for (int i = 0; i < sub; i++)
                        for (int j = 0; j < nr; j++)
                        {
                            if (i >= n) { if (!direct) part[i * nr + j] = 0f; continue; }
                            if (init) cp[i * ld + j] = (biasM != null ? biasM[row0 + r + i] : 0f) + (biasN != null ? biasN[col0 + j] : 0f);
                            else if (!direct) part[i * nr + j] = c[(r + i) * ldc + j];
                        }
                    if (isa == CpuGemmIsa.Avx512) Kernel4x32(a + r, b, kc, cp, ld, mr);
                    else Kernel3x16(a + r, b, kc, cp, ld, mr);
                    if (!direct)
                        for (int i = 0; i < n; i++)
                            for (int j = 0; j < nr; j++) c[(r + i) * ldc + j] = part[i * nr + j];
                }
                return;
            }

            float* tmp = stackalloc float[12 * 32];
            for (int r = 0; r < mr; r++)
                for (int j = 0; j < nr; j++)
                {
                    float v = 0f;
                    if (r < rows && j < cols)
                    {
                        if (!init) v = c[r * ldc + j];
                        else v = (biasM != null ? biasM[row0 + r] : 0f) + (biasN != null ? biasN[col0 + j] : 0f);
                    }
                    tmp[r * nr + j] = v;
                }
            Kernel(isa, a, b, kc, tmp, nr);
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < cols; j++) c[r * ldc + j] = tmp[r * nr + j];
        }

        // One shared B buffer, handed out lock-free: GEMMs normally run one at a time, and a
        // concurrent caller that finds it taken just allocates its own.
        private static float[] s_shared;

        private static float[] RentShared(long floats)
        {
            float[] buffer = System.Threading.Interlocked.Exchange(ref s_shared, null);
            if (buffer == null || buffer.LongLength < floats + 16)
                buffer = GC.AllocateUninitializedArray<float>(checked((int)(floats + 16)), pinned: true);
            return buffer;
        }

        private static void ReturnShared(float[] buffer) => System.Threading.Interlocked.Exchange(ref s_shared, buffer);

        private static nint Align64(float[] pinned) => ((nint)Unsafe.AsPointer(ref pinned[0]) + 63) & ~(nint)63;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Kernel(CpuGemmIsa isa, float* a, float* b, int kc, float* c, long ldc)
        {
            if (isa == CpuGemmIsa.Avx512) Kernel12x32(a, b, kc, c, ldc);
            else if (isa == CpuGemmIsa.Avx2) Kernel6x16(a, b, kc, c, ldc);
            else Kernel4x8(a, b, kc, c, ldc);
        }

        // ---- micro-kernels (C tile += A panel x B panel) --------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Kernel12x32(float* a, float* b, int kc, float* c, long ldc)
        {
            float* c0 = c, c1 = c + ldc, c2 = c + 2 * ldc, c3 = c + 3 * ldc, c4 = c + 4 * ldc, c5 = c + 5 * ldc;
            float* c6 = c + 6 * ldc, c7 = c + 7 * ldc, c8 = c + 8 * ldc, c9 = c + 9 * ldc, c10 = c + 10 * ldc, c11 = c + 11 * ldc;
            Vector512<float> x0a = Vector512.Load(c0), x0b = Vector512.Load(c0 + 16);
            Vector512<float> x1a = Vector512.Load(c1), x1b = Vector512.Load(c1 + 16);
            Vector512<float> x2a = Vector512.Load(c2), x2b = Vector512.Load(c2 + 16);
            Vector512<float> x3a = Vector512.Load(c3), x3b = Vector512.Load(c3 + 16);
            Vector512<float> x4a = Vector512.Load(c4), x4b = Vector512.Load(c4 + 16);
            Vector512<float> x5a = Vector512.Load(c5), x5b = Vector512.Load(c5 + 16);
            Vector512<float> x6a = Vector512.Load(c6), x6b = Vector512.Load(c6 + 16);
            Vector512<float> x7a = Vector512.Load(c7), x7b = Vector512.Load(c7 + 16);
            Vector512<float> x8a = Vector512.Load(c8), x8b = Vector512.Load(c8 + 16);
            Vector512<float> x9a = Vector512.Load(c9), x9b = Vector512.Load(c9 + 16);
            Vector512<float> x10a = Vector512.Load(c10), x10b = Vector512.Load(c10 + 16);
            Vector512<float> x11a = Vector512.Load(c11), x11b = Vector512.Load(c11 + 16);
            for (int k = 0; k < kc; k++)
            {
                Vector512<float> b0 = Vector512.Load(b), b1 = Vector512.Load(b + 16);
                Vector512<float> av = Vector512.Create(a[0]);
                x0a = Avx512F.FusedMultiplyAdd(av, b0, x0a); x0b = Avx512F.FusedMultiplyAdd(av, b1, x0b);
                av = Vector512.Create(a[1]);
                x1a = Avx512F.FusedMultiplyAdd(av, b0, x1a); x1b = Avx512F.FusedMultiplyAdd(av, b1, x1b);
                av = Vector512.Create(a[2]);
                x2a = Avx512F.FusedMultiplyAdd(av, b0, x2a); x2b = Avx512F.FusedMultiplyAdd(av, b1, x2b);
                av = Vector512.Create(a[3]);
                x3a = Avx512F.FusedMultiplyAdd(av, b0, x3a); x3b = Avx512F.FusedMultiplyAdd(av, b1, x3b);
                av = Vector512.Create(a[4]);
                x4a = Avx512F.FusedMultiplyAdd(av, b0, x4a); x4b = Avx512F.FusedMultiplyAdd(av, b1, x4b);
                av = Vector512.Create(a[5]);
                x5a = Avx512F.FusedMultiplyAdd(av, b0, x5a); x5b = Avx512F.FusedMultiplyAdd(av, b1, x5b);
                av = Vector512.Create(a[6]);
                x6a = Avx512F.FusedMultiplyAdd(av, b0, x6a); x6b = Avx512F.FusedMultiplyAdd(av, b1, x6b);
                av = Vector512.Create(a[7]);
                x7a = Avx512F.FusedMultiplyAdd(av, b0, x7a); x7b = Avx512F.FusedMultiplyAdd(av, b1, x7b);
                av = Vector512.Create(a[8]);
                x8a = Avx512F.FusedMultiplyAdd(av, b0, x8a); x8b = Avx512F.FusedMultiplyAdd(av, b1, x8b);
                av = Vector512.Create(a[9]);
                x9a = Avx512F.FusedMultiplyAdd(av, b0, x9a); x9b = Avx512F.FusedMultiplyAdd(av, b1, x9b);
                av = Vector512.Create(a[10]);
                x10a = Avx512F.FusedMultiplyAdd(av, b0, x10a); x10b = Avx512F.FusedMultiplyAdd(av, b1, x10b);
                av = Vector512.Create(a[11]);
                x11a = Avx512F.FusedMultiplyAdd(av, b0, x11a); x11b = Avx512F.FusedMultiplyAdd(av, b1, x11b);
                a += 12; b += 32;
            }
            x0a.Store(c0); x0b.Store(c0 + 16); x1a.Store(c1); x1b.Store(c1 + 16);
            x2a.Store(c2); x2b.Store(c2 + 16); x3a.Store(c3); x3b.Store(c3 + 16);
            x4a.Store(c4); x4b.Store(c4 + 16); x5a.Store(c5); x5b.Store(c5 + 16);
            x6a.Store(c6); x6b.Store(c6 + 16); x7a.Store(c7); x7b.Store(c7 + 16);
            x8a.Store(c8); x8b.Store(c8 + 16); x9a.Store(c9); x9b.Store(c9 + 16);
            x10a.Store(c10); x10b.Store(c10 + 16); x11a.Store(c11); x11b.Store(c11 + 16);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Kernel6x16(float* a, float* b, int kc, float* c, long ldc)
        {
            float* c0 = c, c1 = c + ldc, c2 = c + 2 * ldc, c3 = c + 3 * ldc, c4 = c + 4 * ldc, c5 = c + 5 * ldc;
            Vector256<float> x0a = Vector256.Load(c0), x0b = Vector256.Load(c0 + 8);
            Vector256<float> x1a = Vector256.Load(c1), x1b = Vector256.Load(c1 + 8);
            Vector256<float> x2a = Vector256.Load(c2), x2b = Vector256.Load(c2 + 8);
            Vector256<float> x3a = Vector256.Load(c3), x3b = Vector256.Load(c3 + 8);
            Vector256<float> x4a = Vector256.Load(c4), x4b = Vector256.Load(c4 + 8);
            Vector256<float> x5a = Vector256.Load(c5), x5b = Vector256.Load(c5 + 8);
            for (int k = 0; k < kc; k++)
            {
                Vector256<float> b0 = Vector256.Load(b), b1 = Vector256.Load(b + 8);
                Vector256<float> av = Vector256.Create(a[0]);
                x0a = Fma.MultiplyAdd(av, b0, x0a); x0b = Fma.MultiplyAdd(av, b1, x0b);
                av = Vector256.Create(a[1]);
                x1a = Fma.MultiplyAdd(av, b0, x1a); x1b = Fma.MultiplyAdd(av, b1, x1b);
                av = Vector256.Create(a[2]);
                x2a = Fma.MultiplyAdd(av, b0, x2a); x2b = Fma.MultiplyAdd(av, b1, x2b);
                av = Vector256.Create(a[3]);
                x3a = Fma.MultiplyAdd(av, b0, x3a); x3b = Fma.MultiplyAdd(av, b1, x3b);
                av = Vector256.Create(a[4]);
                x4a = Fma.MultiplyAdd(av, b0, x4a); x4b = Fma.MultiplyAdd(av, b1, x4b);
                av = Vector256.Create(a[5]);
                x5a = Fma.MultiplyAdd(av, b0, x5a); x5b = Fma.MultiplyAdd(av, b1, x5b);
                a += 6; b += 16;
            }
            x0a.Store(c0); x0b.Store(c0 + 8); x1a.Store(c1); x1b.Store(c1 + 8);
            x2a.Store(c2); x2b.Store(c2 + 8); x3a.Store(c3); x3b.Store(c3 + 8);
            x4a.Store(c4); x4b.Store(c4 + 8); x5a.Store(c5); x5b.Store(c5 + 8);
        }

        // Row-edge kernels: the first 4 (3) rows of an MR-row A panel (A stride aStride per k).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Kernel4x32(float* a, float* b, int kc, float* c, long ldc, int aStride)
        {
            float* c0 = c, c1 = c + ldc, c2 = c + 2 * ldc, c3 = c + 3 * ldc;
            Vector512<float> x0a = Vector512.Load(c0), x0b = Vector512.Load(c0 + 16);
            Vector512<float> x1a = Vector512.Load(c1), x1b = Vector512.Load(c1 + 16);
            Vector512<float> x2a = Vector512.Load(c2), x2b = Vector512.Load(c2 + 16);
            Vector512<float> x3a = Vector512.Load(c3), x3b = Vector512.Load(c3 + 16);
            for (int k = 0; k < kc; k++)
            {
                Vector512<float> b0 = Vector512.Load(b), b1 = Vector512.Load(b + 16);
                Vector512<float> av = Vector512.Create(a[0]);
                x0a = Avx512F.FusedMultiplyAdd(av, b0, x0a); x0b = Avx512F.FusedMultiplyAdd(av, b1, x0b);
                av = Vector512.Create(a[1]);
                x1a = Avx512F.FusedMultiplyAdd(av, b0, x1a); x1b = Avx512F.FusedMultiplyAdd(av, b1, x1b);
                av = Vector512.Create(a[2]);
                x2a = Avx512F.FusedMultiplyAdd(av, b0, x2a); x2b = Avx512F.FusedMultiplyAdd(av, b1, x2b);
                av = Vector512.Create(a[3]);
                x3a = Avx512F.FusedMultiplyAdd(av, b0, x3a); x3b = Avx512F.FusedMultiplyAdd(av, b1, x3b);
                a += aStride; b += 32;
            }
            x0a.Store(c0); x0b.Store(c0 + 16); x1a.Store(c1); x1b.Store(c1 + 16);
            x2a.Store(c2); x2b.Store(c2 + 16); x3a.Store(c3); x3b.Store(c3 + 16);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Kernel3x16(float* a, float* b, int kc, float* c, long ldc, int aStride)
        {
            float* c0 = c, c1 = c + ldc, c2 = c + 2 * ldc;
            Vector256<float> x0a = Vector256.Load(c0), x0b = Vector256.Load(c0 + 8);
            Vector256<float> x1a = Vector256.Load(c1), x1b = Vector256.Load(c1 + 8);
            Vector256<float> x2a = Vector256.Load(c2), x2b = Vector256.Load(c2 + 8);
            for (int k = 0; k < kc; k++)
            {
                Vector256<float> b0 = Vector256.Load(b), b1 = Vector256.Load(b + 8);
                Vector256<float> av = Vector256.Create(a[0]);
                x0a = Fma.MultiplyAdd(av, b0, x0a); x0b = Fma.MultiplyAdd(av, b1, x0b);
                av = Vector256.Create(a[1]);
                x1a = Fma.MultiplyAdd(av, b0, x1a); x1b = Fma.MultiplyAdd(av, b1, x1b);
                av = Vector256.Create(a[2]);
                x2a = Fma.MultiplyAdd(av, b0, x2a); x2b = Fma.MultiplyAdd(av, b1, x2b);
                a += aStride; b += 16;
            }
            x0a.Store(c0); x0b.Store(c0 + 8); x1a.Store(c1); x1b.Store(c1 + 8); x2a.Store(c2); x2b.Store(c2 + 8);
        }

        // Portable (SSE / AdvSimd). MultiplyAddEstimate is a fused FMA where the hardware has
        // one and a separate multiply + add otherwise.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Kernel4x8(float* a, float* b, int kc, float* c, long ldc)
        {
            float* c0 = c, c1 = c + ldc, c2 = c + 2 * ldc, c3 = c + 3 * ldc;
            Vector128<float> x0a = Vector128.Load(c0), x0b = Vector128.Load(c0 + 4);
            Vector128<float> x1a = Vector128.Load(c1), x1b = Vector128.Load(c1 + 4);
            Vector128<float> x2a = Vector128.Load(c2), x2b = Vector128.Load(c2 + 4);
            Vector128<float> x3a = Vector128.Load(c3), x3b = Vector128.Load(c3 + 4);
            for (int k = 0; k < kc; k++)
            {
                Vector128<float> b0 = Vector128.Load(b), b1 = Vector128.Load(b + 4);
                Vector128<float> av = Vector128.Create(a[0]);
                x0a = Vector128.MultiplyAddEstimate(av, b0, x0a); x0b = Vector128.MultiplyAddEstimate(av, b1, x0b);
                av = Vector128.Create(a[1]);
                x1a = Vector128.MultiplyAddEstimate(av, b0, x1a); x1b = Vector128.MultiplyAddEstimate(av, b1, x1b);
                av = Vector128.Create(a[2]);
                x2a = Vector128.MultiplyAddEstimate(av, b0, x2a); x2b = Vector128.MultiplyAddEstimate(av, b1, x2b);
                av = Vector128.Create(a[3]);
                x3a = Vector128.MultiplyAddEstimate(av, b0, x3a); x3b = Vector128.MultiplyAddEstimate(av, b1, x3b);
                a += 4; b += 8;
            }
            x0a.Store(c0); x0b.Store(c0 + 4); x1a.Store(c1); x1b.Store(c1 + 4);
            x2a.Store(c2); x2b.Store(c2 + 4); x3a.Store(c3); x3b.Store(c3 + 4);
        }

        // ---- helpers ------------------------------------------------------------------

        [ThreadStatic] private static float[] t_scratchRaw;
        [ThreadStatic] private static nint t_scratch;
        [ThreadStatic] private static int t_scratchFloats;

        /// <summary>Thread-local, 64-byte aligned, pinned scratch of at least <paramref name="floats"/>.</summary>
        internal static float* Scratch(int floats)
        {
            if (t_scratchFloats < floats)
            {
                t_scratchRaw = GC.AllocateUninitializedArray<float>(floats + 16, pinned: true);
                nint p = (nint)Unsafe.AsPointer(ref t_scratchRaw[0]);
                t_scratch = (p + 63) & ~(nint)63;
                t_scratchFloats = floats;
            }
            return (float*)t_scratch;
        }

        internal static void ForEach(int count, bool parallel, Action<int> body, CpuWorkerPool pool = null)
        {
            if (!parallel || count <= 1) { for (int i = 0; i < count; i++) body(i); return; }
            (pool ?? CpuWorkerPool.Shared).For(count, body);
        }
    }

    /// <summary>B given as a strided matrix: element (k, n) at <c>B[k*StrideK + n*StrideN]</c>.</summary>
    internal readonly unsafe struct StridedPanelSource : IGemmPanelSource
    {
        private readonly nint _b;
        private readonly long _strideK, _strideN;
        private readonly int _n;

        internal StridedPanelSource(float* b, long strideK, long strideN, int n)
        {
            _b = (nint)b; _strideK = strideK; _strideN = strideN; _n = n;
        }

        public int KAlignment => 1;

        public float* Panels(int k0, int kc, int n0, int count, int nr, float* scratch, out long panelStride)
        {
            panelStride = (long)kc * nr;
            for (int q = 0; q * nr < count; q++)
                Pack(k0, kc, n0 + q * nr, nr, scratch + q * panelStride);
            return scratch;
        }

        /// <summary>Pack one NR-column panel starting at column <paramref name="col0"/>.</summary>
        internal void Pack(int k0, int kc, int col0, int nr, float* dst)
        {
            float* b = (float*)_b;
            int cols = Math.Min(nr, _n - col0);
            if (_strideN == 1)
            {
                for (int j = 0; j < kc; j++)
                {
                    float* s = b + (k0 + j) * _strideK + col0;
                    float* d = dst + (long)j * nr;
                    if (cols == nr) Buffer.MemoryCopy(s, d, nr * sizeof(float), nr * sizeof(float));
                    else
                    {
                        int i = 0;
                        for (; i < cols; i++) d[i] = s[i];
                        for (; i < nr; i++) d[i] = 0f;
                    }
                }
                return;
            }
            // Column-major source (k contiguous): walk each column once, scattering by NR.
            for (int i = 0; i < nr; i++)
            {
                float* d = dst + i;
                if (i >= cols) { for (int j = 0; j < kc; j++) d[(long)j * nr] = 0f; continue; }
                float* s = b + (col0 + i) * _strideN + k0 * _strideK;
                for (int j = 0; j < kc; j++) d[(long)j * nr] = s[j * _strideK];
            }
        }
    }

    /// <summary>B packed ahead of time over the full K (a reused operand such as W^T).</summary>
    internal readonly unsafe struct PrepackedPanelSource : IGemmPanelSource
    {
        private readonly nint _base;
        private readonly int _k;

        internal PrepackedPanelSource(PackedPanels packed) { _base = packed.Base; _k = packed.K; }
        internal PrepackedPanelSource(nint panels, int k) { _base = panels; _k = k; }

        public int KAlignment => 1;

        public float* Panels(int k0, int kc, int n0, int count, int nr, float* scratch, out long panelStride)
        {
            panelStride = (long)_k * nr;
            return (float*)_base + ((long)(n0 / nr) * _k + k0) * nr;
        }
    }

    /// <summary>
    /// Implicit im2col over a planar CHW feature map: B[(ic, ky, kx), (oy, ox)] =
    /// x[ic, oy*sh + ky - padT, ox*sw + kx - padL], zero outside the map (which also realizes
    /// any bottom/right padding). Packing a KC x NT tile is the only im2col that ever exists.
    /// With upsample2x the map is read through a nearest 2x upsample (x[iy/2, ix/2]).
    /// </summary>
    internal readonly unsafe struct Im2colPanelSource : IGemmPanelSource
    {
        private readonly nint _x;
        private readonly int _h, _w, _srcW, _kh, _kw, _sh, _sw, _padT, _padL, _wo, _n, _up;

        internal Im2colPanelSource(float* x, int h, int w, int kh, int kw, int sh, int sw, int padT, int padL,
            int ho, int wo, bool upsample2x = false)
        {
            _up = upsample2x ? 1 : 0;
            _x = (nint)x; _h = h << _up; _w = w << _up; _srcW = w;
            _kh = kh; _kw = kw; _sh = sh; _sw = sw; _padT = padT; _padL = padL;
            _wo = wo; _n = ho * wo;
        }

        public int KAlignment => 1;

        public float* Panels(int k0, int kc, int n0, int count, int nr, float* scratch, out long panelStride)
        {
            panelStride = (long)kc * nr;
            // 1x1 stride-1 unpadded: output pixel p reads input pixel p, rows of B are
            // contiguous input planes regardless of where panels cross image rows.
            bool pointwise = _up == 0 && _kh == 1 && _kw == 1 && _sh == 1 && _sw == 1 && _padT == 0 && _padL == 0
                && _n == _h * _w;
            for (int q = 0; q * nr < count; q++)
            {
                int col0 = n0 + q * nr;
                float* dst = scratch + q * panelStride;
                if (pointwise) PackPointwise(k0, kc, col0, nr, dst);
                else PackPanel(k0, kc, col0, nr, dst);
            }
            return scratch;
        }

        private void PackPointwise(int k0, int kc, int col0, int nr, float* dst)
        {
            float* x = (float*)_x;
            long plane = (long)_h * _w;
            int cols = Math.Min(nr, _n - col0);
            for (int j = 0; j < kc; j++)
            {
                float* s = x + (k0 + j) * plane + col0;
                float* d = dst + (long)j * nr;
                if (cols == nr) Buffer.MemoryCopy(s, d, nr * sizeof(float), nr * sizeof(float));
                else
                {
                    int i = 0;
                    for (; i < cols; i++) d[i] = s[i];
                    for (; i < nr; i++) d[i] = 0f;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void PackPanel(int k0, int kc, int col0, int nr, float* dst)
        {
            float* x = (float*)_x;
            int h = _h, w = _w, srcW = _srcW, up = _up, kw = _kw, khw = _kh * _kw, sw = _sw;
            long planeSize = (long)(h >> up) * srcW;
            int cols = Math.Min(nr, _n - col0);
            // Split the panel into runs that stay on one output row (one run unless the
            // panel wraps, e.g. 32 columns over a 16-wide map): each run of each B row is
            // then a shifted slice of one input row, copied rather than gathered.
            int* segStart = stackalloc int[nr + 1];
            int* segRowBase = stackalloc int[nr];
            int* segColBase = stackalloc int[nr];
            int segments = 0;
            for (int i = 0; i < cols;)
            {
                int p = col0 + i, y = p / _wo, ox = p - y * _wo;
                int len = Math.Min(cols - i, _wo - ox);
                segStart[segments] = i;
                segRowBase[segments] = y * _sh - _padT;
                segColBase[segments] = ox * sw - _padL;
                segments++;
                i += len;
            }
            segStart[segments] = cols;

            int ic = k0 / khw, rem = k0 - ic * khw, ky = rem / kw, kx = rem - ky * kw;
            for (int j = 0; j < kc; j++)
            {
                float* d = dst + (long)j * nr;
                float* plane = x + ic * planeSize;
                for (int g = 0; g < segments; g++)
                {
                    int i0 = segStart[g], len = segStart[g + 1] - i0;
                    int iy = segRowBase[g] + ky;
                    if ((uint)iy >= (uint)h) { new Span<float>(d + i0, len).Clear(); continue; }
                    float* s = plane + (long)(iy >> up) * srcW;
                    int ix0 = segColBase[g] + kx;
                    if (sw == 1 && up == 0)
                    {
                        // In-range part copied, out-of-range (padding) columns zeroed.
                        int lo = Math.Clamp(-ix0, 0, len), hi = Math.Clamp(w - ix0, lo, len);
                        if (lo > 0) new Span<float>(d + i0, lo).Clear();
                        if (hi > lo) Buffer.MemoryCopy(s + ix0 + lo, d + i0 + lo, (hi - lo) * sizeof(float), (hi - lo) * sizeof(float));
                        if (hi < len) new Span<float>(d + i0 + hi, len - hi).Clear();
                    }
                    else
                        for (int i = 0; i < len; i++)
                        {
                            int ix = ix0 + i * sw;
                            d[i0 + i] = (uint)ix < (uint)w ? s[ix >> up] : 0f;
                        }
                }
                if (cols < nr) new Span<float>(d + cols, nr - cols).Clear();
                if (++kx == kw) { kx = 0; if (++ky == _kh) { ky = 0; ic++; } }
            }
        }
    }
}
