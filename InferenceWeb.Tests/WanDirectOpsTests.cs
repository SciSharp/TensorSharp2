using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Models.WanVideo;

namespace InferenceWeb.Tests;

// Numeric tests for the direct-backend Wan primitives (WanDirectOps /
// CudaWanOps): the managed parallel GEMM kernels, the generic attention
// fallback, rowwise modulate/gate/bias ops and interleaved RoPE — each against
// a naive reference, and the CUDA kernels against the CPU implementations.
public class WanDirectOpsTests
{
    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1);
        return a;
    }

    private static void AssertClose(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            float mag = Math.Max(1f, Math.Abs(expected[i]));
            Assert.True(diff <= tol * mag, $"[{i}] expected {expected[i]}, got {actual[i]} (diff {diff})");
        }
    }

    [Fact]
    public void CpuGemmABt_MatchesNaive()
    {
        const int m = 7, k = 37, n = 11;
        var alloc = new CpuAllocator(BlasEnum.DotNet);
        float[] a = Rand(m * k, 1), b = Rand(n * k, 2);
        using var at = Tensor.FromArray(alloc, a); using var atv = at.View(m, k);
        using var bt = Tensor.FromArray(alloc, b); using var btv = bt.View(n, k);
        using var ct = new Tensor(alloc, DType.Float32, m, n);
        WanDirectOps.CpuGemmABt(atv, btv, ct, 0.5f, 0f);

        var expect = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int t = 0; t < k; t++) s += a[i * k + t] * b[j * k + t];
                expect[i * n + j] = (float)(0.5 * s);
            }
        AssertClose(expect, ct.GetElementsAsFloat(m * n));
    }

    [Fact]
    public void CpuGemmAB_MatchesNaive_WithBetaAccumulate()
    {
        const int m = 5, k = 23, n = 17;
        var alloc = new CpuAllocator(BlasEnum.DotNet);
        float[] a = Rand(m * k, 3), b = Rand(k * n, 4), c0 = Rand(m * n, 5);
        using var at = Tensor.FromArray(alloc, a); using var atv = at.View(m, k);
        using var bt = Tensor.FromArray(alloc, b); using var btv = bt.View(k, n);
        using var ct = Tensor.FromArray(alloc, (float[])c0.Clone()); using var ctv = ct.View(m, n);
        WanDirectOps.CpuGemmAB(atv, btv, ctv, 1f, 1f);

        var expect = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = c0[i * n + j];
                for (int t = 0; t < k; t++) s += a[i * k + t] * b[t * n + j];
                expect[i * n + j] = (float)s;
            }
        AssertClose(expect, ctv.GetElementsAsFloat(m * n));
    }

    // Naive reference attention over token-major [sq, heads*hd] with optional bias.
    private static float[] NaiveAttention(float[] q, float[] k, float[] v, float[] bias,
        int heads, int hd, int sq, int sk, float scale)
    {
        var outT = new float[sq * heads * hd];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < sq; i++)
            {
                var scores = new double[sk];
                double max = double.NegativeInfinity;
                for (int j = 0; j < sk; j++)
                {
                    double s = 0;
                    for (int d = 0; d < hd; d++)
                        s += q[(i * heads + h) * hd + d] * k[(j * heads + h) * hd + d];
                    s *= scale;
                    if (bias != null) s += bias[((long)h * sq + i) * sk + j];
                    scores[j] = s;
                    if (s > max) max = s;
                }
                double denom = 0;
                for (int j = 0; j < sk; j++) { scores[j] = Math.Exp(scores[j] - max); denom += scores[j]; }
                for (int d = 0; d < hd; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < sk; j++) acc += scores[j] / denom * v[(j * heads + h) * hd + d];
                    outT[(i * heads + h) * hd + d] = (float)acc;
                }
            }
        return outT;
    }

    [Theory]
    [InlineData(2, 32, 40, 40, false)]
    [InlineData(3, 16, 33, 9, false)]
    [InlineData(2, 16, 20, 20, true)]
    public void CpuAttention_MatchesNaive(int heads, int hd, int sq, int sk, bool withBias)
    {
        var alloc = new CpuAllocator(BlasEnum.DotNet);
        var ctx = new WanDirectContext(alloc);
        float[] q = Rand(sq * heads * hd, 10), k = Rand(sk * heads * hd, 11), v = Rand(sk * heads * hd, 12);
        float[] bias = withBias ? Rand(heads * sq * sk, 13) : null;
        float scale = 1f / MathF.Sqrt(hd);

        using var qt = ctx.FromFloats(q, sq, heads * hd);
        using var kt = ctx.FromFloats(k, sk, heads * hd);
        using var vt = ctx.FromFloats(v, sk, heads * hd);
        using var bt = bias != null ? ctx.FromFloats(bias, heads, sq, sk) : null;
        using var ot = WanDirectOps.Attention(ctx, qt, kt, vt, heads, hd, scale, bt);

        AssertClose(NaiveAttention(q, k, v, bias, heads, hd, sq, sk, scale),
                    ot.GetElementsAsFloat(sq * heads * hd), 2e-4f);
    }

    [Fact]
    public void CpuRopeModulateGateBias_MatchReference()
    {
        const int seq = 6, heads = 2, hd = 8, dim = heads * hd;
        var alloc = new CpuAllocator(BlasEnum.DotNet);
        var ctx = new WanDirectContext(alloc);
        float[] x = Rand(seq * dim, 20), cos = Rand(seq * hd, 21), sin = Rand(seq * hd, 22);
        // pair-duplicate the tables (WanRope layout)
        for (int t = 0; t < seq; t++)
            for (int p = 0; p < hd / 2; p++)
            {
                cos[t * hd + 2 * p + 1] = cos[t * hd + 2 * p];
                sin[t * hd + 2 * p + 1] = sin[t * hd + 2 * p];
            }

        using var xt = ctx.FromFloats((float[])x.Clone(), seq, dim);
        using var ct = ctx.FromFloats(cos, seq, hd);
        using var st = ctx.FromFloats(sin, seq, hd);
        WanDirectOps.RopeInterleaved(ctx, xt, ct, st, heads, hd);
        var got = xt.GetElementsAsFloat(seq * dim);
        for (int t = 0; t < seq; t++)
            for (int h = 0; h < heads; h++)
                for (int p = 0; p < hd / 2; p++)
                {
                    float c = cos[t * hd + 2 * p], s = sin[t * hd + 2 * p];
                    int e = (t * heads + h) * hd + 2 * p;
                    float ev = x[e] * c - x[e + 1] * s;
                    float ov = x[e + 1] * c + x[e] * s;
                    Assert.True(Math.Abs(got[e] - ev) < 1e-5f && Math.Abs(got[e + 1] - ov) < 1e-5f,
                        $"rope mismatch at t={t} h={h} p={p}");
                }

        // modulate rows [1, seq): y = x*(1+scale)+shift
        float[] shift = Rand(dim, 23), scaleV = Rand(dim, 24);
        using var mt = ctx.FromFloats((float[])x.Clone(), seq, dim);
        using var sh = ctx.FromFloats(shift, dim);
        using var sc = ctx.FromFloats(scaleV, dim);
        WanDirectOps.ModulateRows(ctx, mt, mt, sh, sc, 1, seq - 1);
        var mgot = mt.GetElementsAsFloat(seq * dim);
        for (int i = 0; i < seq * dim; i++)
        {
            float expect = i < dim ? x[i] : x[i] * (1 + scaleV[i % dim]) + shift[i % dim];
            Assert.True(Math.Abs(mgot[i] - expect) < 1e-5f, $"modulate mismatch at {i}");
        }

        // gated add: x += v*gate over all rows
        float[] vv = Rand(seq * dim, 25), gate = Rand(dim, 26);
        using var gt = ctx.FromFloats((float[])x.Clone(), seq, dim);
        using var vt = ctx.FromFloats(vv, seq, dim);
        using var ga = ctx.FromFloats(gate, dim);
        WanDirectOps.GateAddRows(ctx, gt, vt, ga, 0, seq);
        var ggot = gt.GetElementsAsFloat(seq * dim);
        for (int i = 0; i < seq * dim; i++)
            Assert.True(Math.Abs(ggot[i] - (x[i] + vv[i] * gate[i % dim])) < 1e-5f, $"gate mismatch at {i}");
    }

    // Manual perf probe (TS_WAN_ATTN_BENCH=1): raw streaming-attention kernel
    // throughput at the 832x480x33f DiT shape, printed as GFLOPS.
    [Fact]
    public void WanAttentionThroughput()
    {
        if (Environment.GetEnvironmentVariable("TS_WAN_ATTN_BENCH") != "1" || !CudaBackend.IsAvailable())
            return;
        const int seq = 14040, heads = 12, hd = 128;
        using var alloc = new CudaAllocator();
        var ctx = new WanDirectContext(alloc);
        using var q = ctx.FromFloats(Rand(seq * heads * hd, 50), seq, heads * hd);
        using var k = ctx.FromFloats(Rand(seq * heads * hd, 51), seq, heads * hd);
        using var v = ctx.FromFloats(Rand(seq * heads * hd, 52), seq, heads * hd);
        float scale = 1f / MathF.Sqrt(hd);
        using (var warm = WanDirectOps.Attention(ctx, q, k, v, heads, hd, scale)) warm.GetElementsAsFloat(1);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        const int iters = 5;
        for (int i = 0; i < iters; i++)
        {
            using var o = WanDirectOps.Attention(ctx, q, k, v, heads, hd, scale);
            o.GetElementsAsFloat(1);
        }
        sw.Stop();
        double flop = 4.0 * seq * seq * hd * heads;
        Console.WriteLine($"[wan-attn-bench] {sw.Elapsed.TotalMilliseconds / iters:F1} ms/iter, " +
                          $"{flop / (sw.Elapsed.TotalSeconds / iters) / 1e9:F0} GFLOPS");
        ctx.Dispose();
    }

    [Fact]
    public void CudaWanKernels_MatchCpu()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int heads = 2, hd = 128, sq = 300, sk = 70;
        var cpuAlloc = new CpuAllocator(BlasEnum.DotNet);
        using var cudaAlloc = new CudaAllocator();
        var cpuCtx = new WanDirectContext(cpuAlloc);
        var cudaCtx = new WanDirectContext(cudaAlloc);

        float[] q = Rand(sq * heads * hd, 30), k = Rand(sk * heads * hd, 31), v = Rand(sk * heads * hd, 32);
        float[] bias = Rand(heads * sq * sk, 33);
        float scale = 1f / MathF.Sqrt(hd);

        float[] Run(WanDirectContext ctx)
        {
            using var qt = ctx.FromFloats(q, sq, heads * hd);
            using var kt = ctx.FromFloats(k, sk, heads * hd);
            using var vt = ctx.FromFloats(v, sk, heads * hd);
            using var bt = ctx.FromFloats(bias, heads, sq, sk);
            using var ot = WanDirectOps.Attention(ctx, qt, kt, vt, heads, hd, scale, bt);
            return ot.GetElementsAsFloat(sq * heads * hd);
        }
        AssertClose(Run(cpuCtx), Run(cudaCtx), 5e-4f);

        // rope + modulate + gate + zero on CUDA vs CPU
        const int seq = 40, dim = heads * hd;
        float[] x = Rand(seq * dim, 34), cos = Rand(seq * hd, 35), sin = Rand(seq * hd, 36);
        float[] shift = Rand(dim, 37), scaleV = Rand(dim, 38), gate = Rand(dim, 39), vv = Rand(seq * dim, 40);

        float[] RunSmall(WanDirectContext ctx)
        {
            using var xt = ctx.FromFloats((float[])x.Clone(), seq, dim);
            using var ct = ctx.FromFloats(cos, seq, hd);
            using var st = ctx.FromFloats(sin, seq, hd);
            WanDirectOps.RopeInterleaved(ctx, xt, ct, st, heads, hd);
            using var sh = ctx.FromFloats(shift, dim);
            using var sc = ctx.FromFloats(scaleV, dim);
            WanDirectOps.ModulateRows(ctx, xt, xt, sh, sc, 2, seq - 2);
            using var ga = ctx.FromFloats(gate, dim);
            using var vt = ctx.FromFloats(vv, seq, dim);
            WanDirectOps.GateAddRows(ctx, xt, vt, ga, 0, seq);
            return xt.GetElementsAsFloat(seq * dim);
        }
        AssertClose(RunSmall(cpuCtx), RunSmall(cudaCtx), 1e-4f);

        using (var z = cudaCtx.NewF32(17, 9))
        {
            WanDirectOps.Zero(cudaCtx, z);
            foreach (float f in z.GetElementsAsFloat(17 * 9)) Assert.Equal(0f, f);
        }

        cudaCtx.Dispose();
        cpuCtx.Dispose();
    }
}
