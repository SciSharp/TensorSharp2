using System.Runtime.Intrinsics;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

/// <summary>
/// SIMD + parallel fast paths of the CPU elementwise / norm / softmax / copy / RoPE ops
/// (CpuKernels behind TensorApplyCPU), on both vector widths, against scalar references:
/// arithmetic ops bit-exact, transcendental ones within a few ULP of a double reference,
/// reductions within a K-scaled bound. Lengths cover vector tails and the parallel split.
/// </summary>
[Collection("CPU kernel selection")]
public unsafe class CpuSimdKernelTests
{
    private readonly IAllocator _alloc = new CpuAllocator(BlasEnum.DotNet);

    public static IEnumerable<object[]> Widths()
    {
        yield return new object[] { false };
        if (Vector512.IsHardwareAccelerated)
            yield return new object[] { true };
    }

    private static IDisposable UseWidth(bool use512)
    {
        bool previous = CpuKernels.Use512;
        CpuKernels.Use512 = use512;
        return new Restore(() => CpuKernels.Use512 = previous);
    }

    private sealed class Restore : IDisposable
    {
        private readonly Action _undo;
        public Restore(Action undo) => _undo = undo;
        public void Dispose() => _undo();
    }

    private static readonly int[] Lengths = { 1, 7, 15, 16, 17, 31, 33, 100, 1000, 300_007 };

    private Tensor FromArray(float[] values, params long[] sizes)
    {
        var t = new Tensor(_alloc, DType.Float32, sizes);
        t.SetElementsAsFloat(values);
        return t;
    }

    private static float[] RandomArray(int seed, int n, float lo = -4f, float hi = 4f)
    {
        var rng = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(lo + (hi - lo) * rng.NextDouble());
        return v;
    }

    private static float[] Read(Tensor t) => t.GetElementsAsFloat((int)t.ElementCount());

    private static void AssertNear(double expected, float actual, double rel, double abs, string what)
    {
        Assert.True(Math.Abs(actual - expected) <= rel * Math.Abs(expected) + abs,
            $"{what}: got {actual}, expected {expected}");
    }

    // ---------------------------------------------------------------- binary / scalar ops

    [Theory]
    [MemberData(nameof(Widths))]
    public void BinaryOps_AreBitExact_IncludingInPlace(bool use512)
    {
        using var _ = UseWidth(use512);
        foreach (int n in Lengths)
        {
            float[] x = RandomArray(n, n);
            float[] y = RandomArray(n + 1, n, 0.5f, 3f);
            using Tensor tx = FromArray(x, n);
            using Tensor ty = FromArray(y, n);
            using var r = new Tensor(_alloc, DType.Float32, n);

            Ops.Add(r, tx, ty); Assert.Equal(x.Zip(y, (a, b) => a + b), Read(r));
            Ops.Sub(r, tx, ty); Assert.Equal(x.Zip(y, (a, b) => a - b), Read(r));
            Ops.Mul(r, tx, ty); Assert.Equal(x.Zip(y, (a, b) => a * b), Read(r));
            Ops.Div(r, tx, ty); Assert.Equal(x.Zip(y, (a, b) => a / b), Read(r));

            Ops.Add(r, tx, 1.25f); Assert.Equal(x.Select(a => a + 1.25f), Read(r));
            Ops.Sub(r, tx, 1.25f); Assert.Equal(x.Select(a => a - 1.25f), Read(r));
            Ops.Sub(r, 1.25f, tx); Assert.Equal(x.Select(a => 1.25f - a), Read(r));
            Ops.Mul(r, tx, 0.3f); Assert.Equal(x.Select(a => a * 0.3f), Read(r));
            Ops.Div(r, tx, 0.3f); Assert.Equal(x.Select(a => a / 0.3f), Read(r));
            Ops.Div(r, 0.3f, ty); Assert.Equal(y.Select(a => 0.3f / a), Read(r));

            using Tensor inPlace = FromArray(x, n);
            Ops.Mul(inPlace, inPlace, ty);
            Assert.Equal(x.Zip(y, (a, b) => a * b), Read(inPlace));
        }
    }

    [Theory]
    [MemberData(nameof(Widths))]
    public void RowBroadcast_IsBitExact(bool use512)
    {
        using var _ = UseWidth(use512);
        foreach (var (rows, cols) in new[] { (3, 1), (2, 7), (5, 16), (3, 33), (200, 2816) })
        {
            float[] x = RandomArray(rows * 31 + cols, rows * cols);
            float[] b = RandomArray(cols, cols, 0.5f, 2f);
            using Tensor tx = FromArray(x, rows, cols);
            using Tensor tb = FromArray(b, cols);
            using var r = new Tensor(_alloc, DType.Float32, rows, cols);
            Ops.Add(r, tx, tb);
            Assert.Equal(Enumerable.Range(0, rows * cols).Select(i => x[i] + b[i % cols]), Read(r));
            Ops.Div(r, tx, tb);
            Assert.Equal(Enumerable.Range(0, rows * cols).Select(i => x[i] / b[i % cols]), Read(r));
        }
    }

    // ---------------------------------------------------------------- activations

    private static double Gelu(double x) => 0.5 * x * (1 + Math.Tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    [Theory]
    [MemberData(nameof(Widths))]
    public void Activations_MatchDoubleReference(bool use512)
    {
        using var _ = UseWidth(use512);
        foreach (int n in Lengths)
        {
            float[] x = RandomArray(3 * n, n, -8f, 8f);
            float[] y = RandomArray(3 * n + 1, n, -3f, 3f);
            using Tensor tx = FromArray(x, n);
            using Tensor ty = FromArray(y, n);
            using var r = new Tensor(_alloc, DType.Float32, n);

            void Check(Func<int, double> expected, string what, double rel = 4e-6, double abs = 2e-7)
            {
                float[] got = Read(r);
                for (int i = 0; i < n; i++)
                    AssertNear(expected(i), got[i], rel, abs * (1 + Math.Abs(x[i])), $"{what}[{i}] x={x[i]}");
            }

            Ops.GELU(r, tx); Check(i => Gelu(x[i]), "gelu");
            Ops.GELUMul(r, tx, ty); Check(i => Gelu(x[i]) * y[i], "gelumul", 4e-6, 4e-7);
            Ops.SiLU(r, tx); Check(i => x[i] * Sigmoid(x[i]), "silu");
            Ops.SiLUMul(r, tx, ty); Check(i => x[i] * Sigmoid(x[i]) * y[i], "silumul", 4e-6, 4e-7);
            Ops.SigmoidMul(r, ty, tx); Check(i => y[i] * Sigmoid(x[i]), "sigmoidmul", 4e-6, 4e-7);
            Ops.Sigmoid(r, tx); Check(i => Sigmoid(x[i]), "sigmoid");
            Ops.Tanh(r, tx); Check(i => Math.Tanh(x[i]), "tanh");
            Ops.Exp(r, ty); Check(i => Math.Exp(y[i]), "exp");
            Ops.SiLUMulClamp(r, tx, ty, 2.5f);
            Check(i =>
            {
                double g = Math.Min(x[i], 2.5), u = Math.Clamp(y[i], -2.5, 2.5);
                return g * Sigmoid(g) * u;
            }, "silumulclamp", 4e-6, 4e-7);

            // In place (result aliases the gate) is how the FFNs call these.
            using Tensor g2 = FromArray(x, n);
            Ops.GELUMul(g2, g2, ty);
            float[] inPlace = Read(g2);
            for (int i = 0; i < n; i++)
                AssertNear(Gelu(x[i]) * y[i], inPlace[i], 4e-6, 4e-7 * (1 + Math.Abs(x[i])), $"gelumul in-place[{i}]");
        }
    }

    // ---------------------------------------------------------------- row ops

    [Theory]
    [MemberData(nameof(Widths))]
    public void Norms_MatchDoubleReference(bool use512)
    {
        using var _ = UseWidth(use512);
        foreach (var (rows, cols) in new[] { (1, 7), (5, 16), (3, 33), (70, 2816), (300, 2112) })
        {
            float[] x = RandomArray(rows + cols, rows * cols, -3f, 3f);
            float[] g = RandomArray(cols + 5, cols, 0.5f, 1.5f);
            float[] b = RandomArray(cols + 6, cols, -0.5f, 0.5f);
            using Tensor tx = FromArray(x, rows, cols);
            using Tensor tg = FromArray(g, cols);
            using Tensor tb = FromArray(b, cols);
            using var r = new Tensor(_alloc, DType.Float32, rows, cols);
            const float eps = 1e-6f;

            Ops.RMSNorm(r, tx, tg, null, eps);
            float[] rms = Read(r);
            Ops.RMSNorm(r, tx, tg, tb, eps);
            float[] rmsBias = Read(r);
            Ops.LayerNorm(r, tx, tg, tb, eps);
            float[] ln = Read(r);

            for (int i = 0; i < rows; i++)
            {
                double sq = 0, sum = 0;
                for (int j = 0; j < cols; j++) { sq += (double)x[i * cols + j] * x[i * cols + j]; sum += x[i * cols + j]; }
                double inv = 1 / Math.Sqrt(sq / cols + eps);
                double mean = sum / cols, var = 0;
                for (int j = 0; j < cols; j++) { double d = x[i * cols + j] - mean; var += d * d; }
                double sigma = Math.Sqrt(eps + var / cols);
                for (int j = 0; j < cols; j++)
                {
                    int idx = i * cols + j;
                    AssertNear(x[idx] * inv * g[j], rms[idx], 5e-6, 1e-6, $"rmsnorm[{i},{j}]");
                    AssertNear(x[idx] * inv * g[j] + b[j], rmsBias[idx], 5e-6, 1e-6, $"rmsnorm+b[{i},{j}]");
                    AssertNear(g[j] * ((x[idx] - mean) / sigma) + b[j], ln[idx], 5e-6, 2e-6, $"layernorm[{i},{j}]");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Widths))]
    public void Softmax_MatchesDoubleReference_WithMaskedEntries(bool use512)
    {
        using var _ = UseWidth(use512);
        foreach (var (rows, cols) in new[] { (1, 1), (4, 5), (16, 70), (8, 1024), (3, 5000) })
        {
            float[] x = RandomArray(rows * 7 + cols, rows * cols, -10f, 10f);
            for (int i = 0; i < x.Length; i += 3) x[i] = float.NegativeInfinity;
            for (int i = 0; i < rows; i++) x[i * cols + cols - 1] = 1.5f; // every row keeps a finite logit
            using Tensor tx = FromArray(x, rows, cols);
            using var r = new Tensor(_alloc, DType.Float32, rows, cols);
            Ops.Softmax(r, tx);
            float[] got = Read(r);
            for (int i = 0; i < rows; i++)
            {
                double max = double.NegativeInfinity;
                for (int j = 0; j < cols; j++) max = Math.Max(max, x[i * cols + j]);
                double sum = 0;
                for (int j = 0; j < cols; j++) sum += Math.Exp(x[i * cols + j] - max);
                for (int j = 0; j < cols; j++)
                    AssertNear(Math.Exp(x[i * cols + j] - max) / sum, got[i * cols + j], 1e-5, 1e-8, $"softmax[{i},{j}]");
            }

            using Tensor inPlace = FromArray(x, rows, cols);
            Ops.Softmax(inPlace, inPlace);
            Assert.Equal(got, Read(inPlace));
        }
    }

    // ---------------------------------------------------------------- copies, masks, gathers

    [Fact]
    public void StridedCopy_PermuteNarrowAndTranspose_AreExact()
    {
        float[] x = RandomArray(41, 70 * 16 * 256);
        using Tensor tx = FromArray(x, 70, 16, 256);

        // [seq, heads, hd] -> [heads, seq, hd] (head split).
        using Tensor perm = tx.Transpose(0, 1);
        using var r = new Tensor(_alloc, DType.Float32, 16, 70, 256);
        Ops.Copy(r, perm);
        float[] got = Read(r);
        for (int h = 0; h < 16; h++)
            for (int t = 0; t < 70; t++)
                for (int e = 0; e < 256; e += 17)
                    Assert.Equal(x[(t * 16 + h) * 256 + e], got[(h * 70 + t) * 256 + e]);

        // Narrowed columns into a contiguous tensor, and back into a narrowed destination.
        using Tensor flat = FromArray(x, 1120, 256);
        using Tensor window = flat.Narrow(1, 3, 100);
        using var dense = new Tensor(_alloc, DType.Float32, 1120, 100);
        Ops.Copy(dense, window);
        float[] d = Read(dense);
        for (int i = 0; i < 1120; i += 7)
            for (int j = 0; j < 100; j++)
                Assert.Equal(x[i * 256 + 3 + j], d[i * 100 + j]);

        using var target = new Tensor(_alloc, DType.Float32, 1120, 256);
        Ops.Fill(target, -1f);
        using Tensor targetWindow = target.Narrow(1, 50, 100);
        Ops.Copy(targetWindow, dense);
        float[] t2 = Read(target);
        for (int i = 0; i < 1120; i += 5)
        {
            for (int j = 0; j < 256; j++)
            {
                float expected = j >= 50 && j < 150 ? x[i * 256 + 3 + (j - 50)] : -1f;
                Assert.Equal(expected, t2[i * 256 + j]);
            }
        }

        // Full 2D transpose (inner strides differ).
        using Tensor small = FromArray(RandomArray(42, 37 * 53), 37, 53);
        using Tensor smallT = small.Transpose();
        using var rt = new Tensor(_alloc, DType.Float32, 53, 37);
        Ops.Copy(rt, smallT);
        float[] s = Read(small), sT = Read(rt);
        for (int i = 0; i < 37; i++)
            for (int j = 0; j < 53; j++)
                Assert.Equal(s[i * 53 + j], sT[j * 37 + i]);
    }

    [Fact]
    public void CausalMask_AndIndexSelect_AreExact()
    {
        int heads = 3, seq = 70, cols = 90, startPos = 20;
        float[] x = RandomArray(43, heads * seq * cols);
        using Tensor tx = FromArray(x, heads, seq, cols);
        Ops.AddCausalMask(tx, seq, startPos, float.NegativeInfinity);
        float[] got = Read(tx);
        for (int h = 0; h < heads; h++)
            for (int t = 0; t < seq; t++)
                for (int s = 0; s < cols; s++)
                {
                    int idx = (h * seq + t) * cols + s;
                    Assert.Equal(s > startPos + t ? float.NegativeInfinity : x[idx], got[idx]);
                }

        float[] table = RandomArray(44, 50 * 2816);
        using Tensor tt = FromArray(table, 50, 2816);
        int[] ids = Enumerable.Range(0, 64).Select(i => (i * 7) % 50).ToArray();
        using var idx2 = new Tensor(_alloc, DType.Int32, ids.Length);
        idx2.SetElementsAsInt(ids);
        using Tensor rows = Ops.IndexSelect(null, tt, idx2);
        float[] r = Read(rows);
        for (int i = 0; i < ids.Length; i++)
            for (int j = 0; j < 2816; j += 13)
                Assert.Equal(table[ids[i] * 2816 + j], r[i * 2816 + j]);
    }

    [Theory]
    [InlineData(2, 0f)]
    [InlineData(0, 0f)]
    [InlineData(2, 1f)]
    public void RoPEEx_MatchesScalarFormula(int mode, float extFactor)
    {
        int seq = 37, heads = 5, hd = 40;
        float freqBase = 10000f, freqScale = 0.5f, attnFactor = 1f, betaFast = 32f, betaSlow = 1f;
        int nCtxOrig = extFactor != 0 ? 4096 : 0;
        float[] x = RandomArray(45, seq * heads * hd);
        using Tensor tx = FromArray(x, 1, seq, heads, hd);
        using var pos = new Tensor(_alloc, DType.Int32, seq * heads);
        int[] p = new int[seq * heads];
        for (int t = 0; t < seq; t++) for (int h = 0; h < heads; h++) p[t * heads + h] = 3 + t;
        pos.SetElementsAsInt(p);
        using Tensor r = Ops.RoPEEx(null, tx, pos, hd, mode, nCtxOrig, freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow);
        float[] got = Read(r);

        // The legacy per-row formula, verbatim (NeoX halves or interleaved pairs, YaRN ramp).
        bool neox = (mode & 2) != 0;
        int pairs = hd / 2;
        float corrLow = 0, corrHigh = 0, mscale = attnFactor;
        if (extFactor != 0)
        {
            float CorrDim(float nRot) => hd * MathF.Log(nCtxOrig / (nRot * 2.0f * MathF.PI)) / (2.0f * MathF.Log(freqBase));
            corrLow = MathF.Max(0, MathF.Floor(CorrDim(betaFast)));
            corrHigh = MathF.Min(hd / 2 - 1, MathF.Ceiling(CorrDim(betaSlow)));
            mscale *= 1.0f + 0.1f * MathF.Log(1.0f / freqScale);
        }
        for (int row = 0; row < seq * heads; row++)
        {
            for (int i = 0; i < pairs; i++)
            {
                float theta = p[row] * MathF.Pow(freqBase, -2.0f * i / hd);
                float c, s;
                if (extFactor != 0)
                {
                    float interp = freqScale * theta;
                    float rampY = (i - corrLow) / MathF.Max(0.001f, corrHigh - corrLow);
                    float mix = (1.0f - MathF.Min(1.0f, MathF.Max(0.0f, rampY))) * extFactor;
                    float th = interp * (1.0f - mix) + theta * mix;
                    c = MathF.Cos(th) * mscale;
                    s = MathF.Sin(th) * mscale;
                }
                else
                {
                    c = MathF.Cos(theta * freqScale);
                    s = MathF.Sin(theta * freqScale);
                }
                int li = neox ? i : 2 * i, ri = neox ? i + pairs : 2 * i + 1;
                float left = x[row * hd + li], right = x[row * hd + ri];
                Assert.Equal(left * c - right * s, got[row * hd + li]);
                Assert.Equal(right * c + left * s, got[row * hd + ri]);
            }
        }
    }

    // ---------------------------------------------------------------- DirectOps CPU paths

    [Fact]
    public void DirectOps_RowOps_AreBitExact()
    {
        using var ctx = new TensorSharp.Models.Direct.DirectContext(_alloc);
        int rows = 37, cols = 2817;
        float[] x = RandomArray(51, rows * cols);
        float[] v = RandomArray(52, rows * cols);
        float[] shift = RandomArray(53, cols, -1f, 1f);
        float[] scale = RandomArray(54, cols, -0.5f, 0.5f);

        using Tensor tx = FromArray(x, rows, cols);
        using Tensor ty = new Tensor(_alloc, DType.Float32, rows, cols);
        using Tensor tShift = FromArray(shift, cols);
        using Tensor tScale = FromArray(scale, cols);
        Ops.Fill(ty, 0f);
        TensorSharp.Models.Direct.DirectOps.ModulateRows(ctx, ty, tx, tShift, tScale, 3, 30);
        float[] got = Read(ty);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.Equal(r >= 3 && r < 33 ? x[r * cols + c] * (1f + scale[c]) + shift[c] : 0f, got[r * cols + c]);

        using Tensor tv = FromArray(v, rows, cols);
        TensorSharp.Models.Direct.DirectOps.GateAddRows(ctx, tx, tv, tScale, 0, rows);
        got = Read(tx);
        for (int i = 0; i < rows * cols; i++)
            Assert.Equal(x[i] + v[i] * scale[i % cols], got[i]);

        using Tensor tb = FromArray(x, rows, cols);
        TensorSharp.Models.Direct.DirectOps.AddBiasRows(ctx, tb, tShift);
        got = Read(tb);
        for (int i = 0; i < rows * cols; i++)
            Assert.Equal(x[i] + shift[i % cols], got[i]);

        using Tensor tm = FromArray(x, rows, cols);
        TensorSharp.Models.Direct.DirectOps.MulColsRows(ctx, tm, tScale);
        got = Read(tm);
        for (int i = 0; i < rows * cols; i++)
            Assert.Equal(x[i] * scale[i % cols], got[i]);
    }

    [Theory]
    [InlineData(4, 70, 90, 64, false)]
    [InlineData(3, 300, 300, 40, true)]
    [InlineData(1, 1025, 777, 96, false)]
    // Enough (head, query-block) items for the pool (>= 16 even at TS_CPU_POOL=0): every block
    // runs single-threaded GEMMs and the per-row softmax loop, the branch DiT/VAE attention takes.
    [InlineData(16, 300, 300, 64, false)]
    [InlineData(24, 64, 80, 32, true)]
    public void DirectOps_CpuAttention_MatchesDoubleReference(int heads, int sq, int sk, int hd, bool withBias)
    {
        using var ctx = new TensorSharp.Models.Direct.DirectContext(_alloc);
        float[] q = RandomArray(61, sq * heads * hd, -1f, 1f);
        float[] k = RandomArray(62, sk * heads * hd, -1f, 1f);
        float[] v = RandomArray(63, sk * heads * hd, -1f, 1f);
        float[] bias = withBias ? RandomArray(64, heads * sq * sk, -2f, 2f) : null;
        using Tensor tq = FromArray(q, sq, heads * hd);
        using Tensor tk = FromArray(k, sk, heads * hd);
        using Tensor tv = FromArray(v, sk, heads * hd);
        using Tensor tb = withBias ? FromArray(bias, heads, sq, sk) : null;
        float scale = 1f / MathF.Sqrt(hd);
        using Tensor o = TensorSharp.Models.Direct.DirectOps.Attention(ctx, tq, tk, tv, heads, hd, scale, tb);
        float[] got = Read(o);

        var rng = new Random(65);
        var s = new double[sk];
        for (int t = 0; t < 24; t++)
        {
            int h = rng.Next(heads), i = rng.Next(sq);
            double max = double.NegativeInfinity;
            for (int j = 0; j < sk; j++)
            {
                double d = 0;
                for (int e = 0; e < hd; e++) d += (double)q[(i * heads + h) * hd + e] * k[(j * heads + h) * hd + e];
                s[j] = d * scale + (withBias ? bias[(h * sq + i) * sk + j] : 0);
                max = Math.Max(max, s[j]);
            }
            double sum = 0;
            for (int j = 0; j < sk; j++) { s[j] = Math.Exp(s[j] - max); sum += s[j]; }
            for (int e = 0; e < hd; e++)
            {
                double acc = 0;
                for (int j = 0; j < sk; j++) acc += s[j] / sum * v[(j * heads + h) * hd + e];
                AssertNear(acc, got[(i * heads + h) * hd + e], 1e-4, 2e-6, $"attn h={h} i={i} e={e}");
            }
        }
    }

    [Fact]
    public void DirectOps_CpuAttention_EmptyQuery_ReturnsEmpty()
    {
        using var ctx = new TensorSharp.Models.Direct.DirectContext(_alloc);
        using var tq = new Tensor(_alloc, DType.Float32, 0, 2 * 32);
        using Tensor tk = FromArray(RandomArray(71, 5 * 2 * 32), 5, 2 * 32);
        using Tensor tv = FromArray(RandomArray(72, 5 * 2 * 32), 5, 2 * 32);
        using Tensor o = TensorSharp.Models.Direct.DirectOps.Attention(ctx, tq, tk, tv, 2, 32, 0.25f);
        Assert.Equal(new long[] { 0, 64 }, o.Sizes);
    }

    [Fact]
    public void DirectOps_CpuAttention_RejectsShapesItWouldReadPast()
    {
        // The blocked path indexes Q/K/V/bias with raw pointers: a head-shared [1, sq, sk] bias
        // with 2 heads, or Q narrower than heads*hd, must throw rather than read out of bounds.
        using var ctx = new TensorSharp.Models.Direct.DirectContext(_alloc);
        int heads = 2, sq = 6, sk = 5, hd = 16;
        using Tensor tq = FromArray(RandomArray(73, sq * heads * hd), sq, heads * hd);
        using Tensor tk = FromArray(RandomArray(74, sk * heads * hd), sk, heads * hd);
        using Tensor tv = FromArray(RandomArray(75, sk * heads * hd), sk, heads * hd);
        using Tensor shared = FromArray(RandomArray(76, sq * sk), 1, sq, sk);
        Assert.ThrowsAny<ArgumentException>(() =>
            TensorSharp.Models.Direct.DirectOps.Attention(ctx, tq, tk, tv, heads, hd, 0.25f, shared).Dispose());

        using Tensor narrowQ = FromArray(RandomArray(77, sq * hd), sq, hd);
        Assert.ThrowsAny<ArgumentException>(() =>
            TensorSharp.Models.Direct.DirectOps.Attention(ctx, narrowQ, tk, tv, heads, hd, 0.25f).Dispose());

        using Tensor shortV = FromArray(RandomArray(78, (sk - 1) * heads * hd), sk - 1, heads * hd);
        Assert.ThrowsAny<ArgumentException>(() =>
            TensorSharp.Models.Direct.DirectOps.Attention(ctx, tq, tk, shortV, heads, hd, 0.25f).Dispose());
    }

    // ---------------------------------------------------------------- CpuParallel hook

    [Fact]
    public void CpuParallel_ForRange_CoversEveryIndexOnce()
    {
        foreach (long count in new long[] { 0, 1, 5, 1000, 1_000_003 })
        {
            foreach (long minChunk in new long[] { 1, 7, 4096, 10_000_000 })
            {
                var hits = new int[Math.Max(1, count)];
                CpuParallel.ForRange(count, minChunk, (s, e) =>
                {
                    Assert.True(s < e);
                    for (long i = s; i < e; i++) Interlocked.Increment(ref hits[i]);
                });
                for (long i = 0; i < count; i++) Assert.Equal(1, hits[i]);
            }
        }
    }

    [Fact]
    public void CpuParallel_PropagatesExceptions_AndAllowsNesting()
    {
        var ex = Assert.ThrowsAny<Exception>(() => CpuParallel.For(16, i =>
        {
            if (i == 5) throw new InvalidOperationException("boom");
        }));
        Assert.Contains("boom", ex is AggregateException agg ? agg.InnerExceptions[0].Message : ex.Message);

        int total = 0;
        CpuParallel.For(8, i => CpuParallel.For(8, j => Interlocked.Increment(ref total)));
        Assert.Equal(64, total);
    }

    [Fact]
    public void CpuParallel_IsBoundToWorkerPool_UnlessDisabled()
    {
        // Loading TensorSharp.Models installs its CpuWorkerPool as the Core runner.
        System.Runtime.CompilerServices.RuntimeHelpers.RunModuleConstructor(typeof(TensorSharp.Models.ModelBase).Module.ModuleHandle);
        bool disabled = Environment.GetEnvironmentVariable("TS_CPU_POOL") == "0";
        Assert.Equal(!disabled, CpuParallel.HasCustomRunner);
        Assert.True(CpuParallel.DegreeOfParallelism >= 1);
    }
}
