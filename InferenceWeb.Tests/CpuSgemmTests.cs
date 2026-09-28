using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Direct;

namespace InferenceWeb.Tests;

/// <summary>
/// Kernel-selection knobs (CpuSgemm.ActiveKernel, CpuKernels.Use512) are process-wide, so the
/// tests that flip them run alone.
/// </summary>
[CollectionDefinition("CPU kernel selection", DisableParallelization = true)]
public sealed class CpuKernelSelectionCollection { }

/// <summary>
/// Packed SGEMM (CpuSgemm) against a double-precision reference: every microkernel the host
/// supports (AVX-512 8x32, the EVEX 8x24, AVX2 6x16, portable), every transpose/stride layout
/// Ops.Addmm and Ops.AddmmBatch can hand it, alpha/beta, edge tiles, K=1, skinny M, the
/// parallel tile split, and the DirectOps orientations.
/// </summary>
[Collection("CPU kernel selection")]
public unsafe class CpuSgemmTests
{
    private readonly IAllocator _alloc = new CpuAllocator(BlasEnum.DotNet);

    public static IEnumerable<object[]> Kernels()
    {
        foreach (CpuSgemm.KernelKind kind in Enum.GetValues<CpuSgemm.KernelKind>())
        {
            if (CpuSgemm.IsSupported(kind))
                yield return new object[] { kind.ToString() };
        }
    }

    private static IDisposable UseKernel(string kind)
    {
        CpuSgemm.KernelKind previous = CpuSgemm.ActiveKernel;
        CpuSgemm.ActiveKernel = Enum.Parse<CpuSgemm.KernelKind>(kind);
        return new Restore(() => CpuSgemm.ActiveKernel = previous);
    }

    private sealed class Restore : IDisposable
    {
        private readonly Action _undo;
        public Restore(Action undo) => _undo = undo;
        public void Dispose() => _undo();
    }

    private Tensor Random(int seed, params long[] sizes)
    {
        var t = new Tensor(_alloc, DType.Float32, sizes);
        var rng = new Random(seed);
        float[] values = new float[t.ElementCount()];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        t.SetElementsAsFloat(values);
        return t;
    }

    private static float Get(Tensor t, long i, long j)
    {
        float* p = (float*)CpuNativeHelpers.GetBufferStart(t);
        return p[i * t.Strides[0] + j * t.Strides[1]];
    }

    /// <summary>
    /// Asserts C == alpha * A B + beta * C0 elementwise within a K-scaled bound on the
    /// magnitude of the products (float accumulation error grows ~ K * eps * sum|a b|).
    /// </summary>
    private static void AssertGemm(Tensor a, Tensor b, Tensor cBefore, Tensor c, float alpha, float beta)
    {
        long m = c.Sizes[0], n = c.Sizes[1], k = a.Sizes[1];
        double tolFactor = 2e-7 * (k + 16);
        for (long i = 0; i < m; i++)
        {
            for (long j = 0; j < n; j++)
            {
                double acc = 0, mag = 0;
                for (long p = 0; p < k; p++)
                {
                    double prod = (double)Get(a, i, p) * Get(b, p, j);
                    acc += prod;
                    mag += Math.Abs(prod);
                }
                double c0 = beta == 0f ? 0 : Get(cBefore, i, j);
                double expected = alpha * acc + beta * c0;
                double actual = Get(c, i, j);
                double tol = tolFactor * (Math.Abs(alpha) * mag + Math.Abs(beta * c0)) + 1e-6;
                Assert.True(Math.Abs(actual - expected) <= tol,
                    $"C[{i},{j}] = {actual}, expected {expected} (tol {tol:E2}), m={m} n={n} k={k}");
            }
        }
    }

    /// <summary>Contiguous copy of a view (to snapshot C before an in-place GEMM).</summary>
    private Tensor Snapshot(Tensor t)
    {
        var copy = new Tensor(_alloc, DType.Float32, t.Sizes);
        Ops.Copy(copy, t);
        return copy;
    }

    public static IEnumerable<object[]> Shapes()
    {
        string[] kernels = Kernels().Select(k => (string)k[0]).ToArray();
        int[][] shapes =
        {
            new[] { 1, 1, 1 }, new[] { 1, 37, 64 }, new[] { 3, 5, 7 }, new[] { 4, 300, 129 },
            new[] { 5, 1, 33 }, new[] { 8, 32, 1 }, new[] { 9, 33, 17 }, new[] { 13, 47, 256 },
            new[] { 31, 17, 300 }, new[] { 70, 90, 128 }, new[] { 129, 65, 513 }, new[] { 200, 200, 120 },
            // Serial (< 4 MFLOP, one tile) shapes past the MC row block and the NC column block,
            // so the packed driver's ic > 0 / jc > 0 loops run (in NT too for N = 68, above every
            // kernel's narrow-dot limit); in NT, N = 40 and N = 5 take the narrow dot path instead
            // (ragged rows, a K tail, a long unsplit K).
            new[] { 301, 40, 100 }, new[] { 290, 68, 100 }, new[] { 8, 3000, 64 }, new[] { 37, 5, 4133 },
            // Parallel narrow products (row chunks of the dot path in NT); the last one's B^T
            // (40 x 3500 floats) exceeds the dot path's L2 budget, so its K is split in two blocks.
            new[] { 1001, 3, 1152 }, new[] { 403, 29, 700 }, new[] { 21, 40, 3500 },
        };
        foreach (string kernel in kernels)
            foreach (int[] s in shapes)
                foreach (string layout in new[] { "NN", "NT", "TN", "TT" })
                    yield return new object[] { kernel, s[0], s[1], s[2], layout };
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Addmm_MatchesDoubleReference_AllLayouts(string kernel, int m, int n, int k, string layout)
    {
        using var _ = UseKernel(kernel);
        bool ta = layout[0] == 'T', tb = layout[1] == 'T';
        using Tensor aStore = ta ? Random(1, k, m) : Random(1, m, k);
        using Tensor bStore = tb ? Random(2, n, k) : Random(2, k, n);
        using Tensor a = ta ? aStore.Transpose() : aStore.CopyRef();
        using Tensor b = tb ? bStore.Transpose() : bStore.CopyRef();

        using Tensor c = Random(3, m, n);
        using Tensor before = Snapshot(c);
        Ops.Addmm(c, 0f, c, 1f, a, b);
        AssertGemm(a, b, before, c, 1f, 0f);

        using Tensor c2 = Random(4, m, n);
        using Tensor before2 = Snapshot(c2);
        Ops.Addmm(c2, 0.25f, c2, -0.5f, a, b);
        AssertGemm(a, b, before2, c2, -0.5f, 0.25f);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void Addmm_BetaOne_Accumulates_AndAlphaZero_OnlyScales(string kernel)
    {
        using var _ = UseKernel(kernel);
        using Tensor a = Random(5, 37, 300);
        using Tensor b = Random(6, 300, 45);
        using Tensor c = Random(7, 37, 45);
        using Tensor before = Snapshot(c);
        Ops.Addmm(c, 1f, c, 1f, a, b);
        AssertGemm(a, b, before, c, 1f, 1f);

        using Tensor d = Random(8, 37, 45);
        using Tensor dBefore = Snapshot(d);
        Ops.Addmm(d, 2f, d, 0f, a, b);
        float[] got = d.GetElementsAsFloat(37 * 45);
        float[] was = dBefore.GetElementsAsFloat(37 * 45);
        for (int i = 0; i < got.Length; i++)
            Assert.Equal(2f * was[i], got[i]);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void Addmm_BetaZero_IgnoresNaNInOutput(string kernel)
    {
        using var _ = UseKernel(kernel);
        using Tensor a = Random(9, 19, 40);
        using Tensor b = Random(10, 40, 35);
        using var c = new Tensor(_alloc, DType.Float32, 19, 35);
        Ops.Fill(c, float.NaN);
        Ops.Addmm(c, 0f, c, 1f, a, b);
        Assert.DoesNotContain(c.GetElementsAsFloat(19 * 35), float.IsNaN);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void Addmm_NonContiguousViews(string kernel)
    {
        using var _ = UseKernel(kernel);
        // A, B and C are column-narrowed windows of wider matrices (row stride > cols), and a
        // second case writes a column-major C (the C^T = B^T A^T route).
        using Tensor aWide = Random(11, 45, 90);
        using Tensor bWide = Random(12, 70, 80);
        using Tensor cWide = Random(13, 45, 100);
        using Tensor a = aWide.Narrow(1, 7, 61);
        using Tensor bRows = bWide.Narrow(0, 3, 61);
        using Tensor b = bRows.Narrow(1, 5, 53);
        using Tensor c = cWide.Narrow(1, 11, 53);
        using Tensor before = Snapshot(c);
        Ops.Addmm(c, 0.5f, c, 1.5f, a, b);
        AssertGemm(a, b, before, c, 1.5f, 0.5f);

        // Everything outside the C window is untouched.
        using Tensor pristine = Random(13, 45, 100);
        float[] now = cWide.GetElementsAsFloat(45 * 100);
        float[] was = pristine.GetElementsAsFloat(45 * 100);
        for (int i = 0; i < 45; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                if (j < 11 || j >= 64)
                    Assert.Equal(was[i * 100 + j], now[i * 100 + j]);
            }
        }

        using Tensor cStore = Random(14, 53, 45);
        using Tensor cColMajor = cStore.Transpose();          // [45, 53], strides [1, 45]
        using Tensor before2 = Snapshot(cColMajor);
        Ops.Addmm(cColMajor, 0.25f, cColMajor, 1f, a, b);
        AssertGemm(a, b, before2, cColMajor, 1f, 0.25f);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void Gemm_ArbitraryElementStrides(string kernel)
    {
        using var _ = UseKernel(kernel);
        // Neither operand has a unit stride (every other element), so the generic packers run.
        int m = 23, n = 41, k = 57;
        float[] aBuf = new float[m * k * 4];
        float[] bBuf = new float[k * n * 6];
        var rng = new Random(15);
        for (int i = 0; i < aBuf.Length; i++) aBuf[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bBuf.Length; i++) bBuf[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] c = new float[m * n];
        long ars = 2 * k, acs = 2, brs = 3, bcs = 3 * k;
        fixed (float* ap = aBuf, bp = bBuf, cp = c)
        {
            CpuSgemm.Gemm(m, n, k, 1f, ap, ars, acs, bp, brs, bcs, 0f, cp, n);
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double acc = 0, mag = 0;
                for (int p = 0; p < k; p++)
                {
                    double prod = (double)aBuf[i * ars + p * acs] * bBuf[p * brs + j * bcs];
                    acc += prod;
                    mag += Math.Abs(prod);
                }
                Assert.True(Math.Abs(c[i * n + j] - acc) <= 2e-7 * (k + 16) * mag + 1e-6, $"[{i},{j}] {c[i * n + j]} vs {acc}");
            }
        }
    }

    public static IEnumerable<object[]> BatchCases()
    {
        foreach (object[] k in Kernels())
        {
            yield return new object[] { k[0], 3, 23, 23, 128 };
            yield return new object[] { k[0], 16, 70, 70, 256 };
            yield return new object[] { k[0], 5, 1, 17, 33 };
            yield return new object[] { k[0], 2, 130, 97, 65 };
            // Narrow score GEMMs (few keys) on the dot path, serial and parallel over the batch.
            yield return new object[] { k[0], 8, 45, 30, 72 };
            yield return new object[] { k[0], 12, 200, 40, 136 };
        }
    }

    [Theory]
    [MemberData(nameof(BatchCases))]
    public void AddmmBatch_AttentionLayouts(string kernel, int batch, int m, int n, int k)
    {
        using var _ = UseKernel(kernel);
        using Tensor q = Random(16, batch, m, k);
        using Tensor keys = Random(17, batch, n, k);
        using Tensor kT = keys.Transpose(1, 2);               // [batch, k, n] column-major per item
        using Tensor scores = Random(18, batch, m, n);
        Ops.AddmmBatch(scores, 0f, scores, 0.125f, q, kT);

        using Tensor v = Random(19, batch, n, k);
        using Tensor o = Random(20, batch, m, k);
        using Tensor oBefore = Snapshot(o);
        Ops.AddmmBatch(o, 0.5f, o, 1f, scores, v);

        for (int bi = 0; bi < batch; bi++)
        {
            using Tensor qi = q.Select(0, bi);
            using Tensor kti = kT.Select(0, bi);
            using Tensor si = scores.Select(0, bi);
            using var zero = new Tensor(_alloc, DType.Float32, m, n);
            Ops.Fill(zero, 0f);
            AssertGemm(qi, kti, zero, si, 0.125f, 0f);

            using Tensor vi = v.Select(0, bi);
            using Tensor oi = o.Select(0, bi);
            using Tensor obi = oBefore.Select(0, bi);
            AssertGemm(si, vi, obi, oi, 1f, 0.5f);
        }
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void DirectOps_CpuGemmOrientations(string kernel)
    {
        using var _ = UseKernel(kernel);
        // Wide enough to take the parallel tile split, with ragged edges.
        int m = 301, n = 67, k = 200;
        using Tensor x = Random(21, m, k);
        using Tensor w = Random(22, n, k);
        using Tensor c = Random(23, m, n);
        using Tensor before = Snapshot(c);
        DirectOps.CpuGemmABt(x, w, c, 0.75f, 0.5f);
        using Tensor wT = w.Transpose();
        AssertGemm(x, wT, before, c, 0.75f, 0.5f);

        using Tensor bKn = Random(24, k, n);
        using Tensor c2 = new Tensor(_alloc, DType.Float32, m, n);
        using var zeros = new Tensor(_alloc, DType.Float32, m, n);
        Ops.Fill(zeros, 0f);
        DirectOps.CpuGemmAB(x, bKn, c2, 1f, 0f);
        AssertGemm(x, bKn, zeros, c2, 1f, 0f);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void NarrowDot_TailMask_DoesNotTurnInfIntoNaN(string kernel)
    {
        using var _ = UseKernel(kernel);
        // K = 20: the dot path's tail step re-reads the last full vector with the lanes it already
        // summed zeroed in BOTH operands. Column 13 lies in that re-read window for 256- and
        // 512-bit vectors, so a +Inf there must stay +Inf (0 * Inf would be NaN).
        int m = 9, n = 5, k = 20;
        float[] a = new float[m * k];
        float[] bT = new float[n * k];
        var rng = new Random(31);
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble() + 0.5f;
        for (int i = 0; i < bT.Length; i++) bT[i] = (float)(rng.NextDouble() * 2 - 1);
        bT[2 * k + 13] = float.PositiveInfinity;
        float[] c = new float[m * n];
        fixed (float* ap = a, bp = bT, cp = c)
        {
            CpuSgemm.Gemm(m, n, k, 1f, ap, k, 1, bp, 1, k, 0f, cp, n);
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j == 2)
                {
                    Assert.Equal(float.PositiveInfinity, c[i * n + j]);
                    continue;
                }
                double acc = 0, mag = 0;
                for (int p = 0; p < k; p++)
                {
                    double prod = (double)a[i * k + p] * bT[j * k + p];
                    acc += prod;
                    mag += Math.Abs(prod);
                }
                Assert.True(Math.Abs(c[i * n + j] - acc) <= 2e-7 * (k + 16) * mag + 1e-6, $"[{i},{j}] {c[i * n + j]} vs {acc}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void NarrowDot_MatchesPackedPath(string kernel)
    {
        using var _ = UseKernel(kernel);
        // Same product through the packed tile (dot path forced off) and the dot path (forced
        // on, whatever the kernel's default limit): both within the fp64 bound.
        int m = 157, n = 13, k = 333;
        using Tensor x = Random(32, m, k);
        using Tensor w = Random(33, n, k);
        using Tensor wT = w.Transpose();
        using Tensor packed = new Tensor(_alloc, DType.Float32, m, n);
        using Tensor dot = new Tensor(_alloc, DType.Float32, m, n);
        using var zeros = new Tensor(_alloc, DType.Float32, m, n);
        Ops.Fill(zeros, 0f);
        try
        {
            CpuSgemm.NarrowDotMaxN = 0;
            DirectOps.CpuGemmABt(x, w, packed, 1f, 0f);
            CpuSgemm.NarrowDotMaxN = 64;
            DirectOps.CpuGemmABt(x, w, dot, 1f, 0f);
        }
        finally
        {
            CpuSgemm.NarrowDotMaxN = -1;
        }
        AssertGemm(x, wT, zeros, packed, 1f, 0f);
        AssertGemm(x, wT, zeros, dot, 1f, 0f);
    }

    [Fact]
    public void KernelSelection_HonoursSupport()
    {
        Assert.True(CpuSgemm.IsSupported(CpuSgemm.KernelKind.Portable));
        Assert.True(CpuSgemm.IsSupported(CpuSgemm.ActiveKernel));
        if (CpuIsa.Avx512DisabledByEnv)
            Assert.True(CpuSgemm.ActiveKernel <= CpuSgemm.KernelKind.Avx2);
    }
}
