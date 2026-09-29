// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Kernel-level contracts of the DiffusionGemma pure-C# (cpu backend) forward. The model is so
// sensitive to last-bit differences (Q8 activation quantization + top-8 routing) that its kernels
// pin their float arithmetic, so each one is held to BITWISE equality with a reference loop written
// out here (the arithmetic of the Ops chain or legacy loop it replaced, as of this change) and to a
// float64 reference for the math itself. The references are local on purpose: Ops.RMSNorm,
// Ops.GELUMul and the F32 GEMM are free to change their own arithmetic without these kernels
// following. The opt-in FMA attention tiles (DIFFUSION_CPU_ATTN_FAST=1) are held to float64 only,
// on both the Vector<T> (AVX2 / AdvSimd) and, where the hardware has it, the AVX-512 path.
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public sealed unsafe class DiffusionGemmaCpuKernelTests
{
    public static IEnumerable<object[]> AttentionKernels()
    {
        // passed as int: xUnit test methods are public and the kernel enum is internal
        yield return new object[] { (int)DiffusionAttnKernel.Exact };
        yield return new object[] { (int)DiffusionAttnKernel.Fma };
        if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported)
            yield return new object[] { (int)DiffusionAttnKernel.Fma512 };
    }

    private static float[] Random(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static void AssertSameBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"element {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    // one Vector<float> accumulator of x*x (multiply, then add) and its lane sum; returns where the
    // scalar tail starts
    private static float VectorSumOfSquares(float* x, int n, out int tail)
    {
        int vLen = Vector<float>.Count, i = 0;
        Vector<float> acc = Vector<float>.Zero;
        for (; i <= n - vLen; i += vLen)
        {
            Vector<float> v = Unsafe.ReadUnaligned<Vector<float>>(x + i);
            acc += v * v;
        }
        tail = i;
        return Vector.Sum(acc);
    }

    // ---- fused per-head RMSNorm + NeoX RoPE: the (pre-SIMD) Ops.RMSNorm arithmetic, then the scalar RoPE ----
    [Theory]
    [InlineData(8, 256, true, true)]
    [InlineData(2, 512, true, true)]
    [InlineData(8, 256, false, false)]
    [InlineData(3, 40, true, true)]      // head dim that is not a vector multiple
    public void HeadNormRope_MatchesReferenceArithmetic_Bitwise(int heads, int hd, bool weighted, bool rope)
    {
        const float eps = 1e-6f;
        int rows = 5, half = hd / 2;
        float[] x = Random(rows * heads * hd, 1, 3f);
        float[] w = weighted ? Random(hd, 2, 2f) : null;
        var cos = new float[rows * half];
        var sin = new float[rows * half];
        for (int p = 0; p < rows; p++)
            for (int j = 0; j < half; j++)
            {
                float angle = (p + 7) * (float)(1.0 / Math.Pow(10000, 2.0 * j / hd));
                cos[p * half + j] = MathF.Cos(angle);
                sin[p * half + j] = MathF.Sin(angle);
            }

        // reference: sum of squares (vector accumulator + in-order tail), x * (1/sqrt(mean + eps)), then
        // * gamma (ones for the unweighted norm, as the model does), then (x0*c - x1*s, x0*s + x1*c)
        var expected = new float[x.Length];
        var exact = new double[x.Length];
        fixed (float* xp = x)
            for (int r = 0; r < rows; r++)
                for (int h = 0; h < heads; h++)
                {
                    int b = (r * heads + h) * hd;
                    float sq = VectorSumOfSquares(xp + b, hd, out int t);
                    for (; t < hd; t++) sq += x[b + t] * x[b + t];
                    float inv = 1.0f / MathF.Sqrt(sq / hd + eps);
                    double sqD = 0;
                    for (int i = 0; i < hd; i++) sqD += (double)x[b + i] * x[b + i];
                    double invD = 1.0 / Math.Sqrt(sqD / hd + eps);
                    for (int i = 0; i < hd; i++)
                    {
                        expected[b + i] = x[b + i] * inv * (w != null ? w[i] : 1f);
                        exact[b + i] = x[b + i] * invD * (w != null ? w[i] : 1.0);
                    }
                    if (!rope) continue;
                    for (int j = 0; j < half; j++)
                    {
                        float c = cos[r * half + j], sn = sin[r * half + j];
                        float x0 = expected[b + j], x1 = expected[b + j + half];
                        expected[b + j] = x0 * c - x1 * sn;
                        expected[b + j + half] = x0 * sn + x1 * c;
                        double d0 = exact[b + j], d1 = exact[b + j + half];
                        exact[b + j] = d0 * c - d1 * sn;
                        exact[b + j + half] = d0 * sn + d1 * c;
                    }
                }

        var actual = (float[])x.Clone();
        fixed (float* a = actual) fixed (float* wp = w) fixed (float* cp = cos) fixed (float* sp = sin)
            for (int r = 0; r < rows; r++)
                DiffusionGemmaCpuKernels.HeadNormRopeRow(a + r * heads * hd, a + r * heads * hd, heads, hd, wp, eps,
                    rope ? cp + r * half : null, rope ? sp + r * half : null);
        AssertSameBits(expected, actual);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - exact[i]) <= 1e-5 * Math.Max(1.0, Math.Abs(exact[i])), $"element {i}: {actual[i]} vs {exact[i]}");
    }

    // ---- GELU(gate)*up: the tanh approximation with a double-precision tanh, bit for bit ----
    [Fact]
    public void GeluMul_MatchesDoubleTanhFormula_Bitwise()
    {
        int n = 4099;
        float[] gate = Random(n, 3, 8f), up = Random(n, 4, 3f);
        gate[0] = 0f; gate[1] = -0f; gate[2] = 40f; gate[3] = -40f; gate[4] = 1e-30f;
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            float g = gate[i];
            expected[i] = 0.5f * g * (1.0f + (float)Math.Tanh(0.7978845608f * (g + 0.044715f * g * g * g))) * up[i];
            double gd = g;
            double exact = 0.5 * gd * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (gd + 0.044715 * gd * gd * gd))) * up[i];
            Assert.True(Math.Abs(expected[i] - exact) <= 1e-5 * Math.Max(1.0, Math.Abs(exact)), $"element {i}");
        }
        var actual = new float[n];
        fixed (float* gp = gate) fixed (float* upp = up) fixed (float* ap = actual)
            DiffusionGemmaCpuKernels.GeluMulRow(gp, upp, ap, n);
        AssertSameBits(expected, actual);
    }

    // ---- MoE combine == the reference loop (zero-fill, dst += w * (scale * y) in order) ----
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void WeightedRowSum_MatchesReferenceLoop_Bitwise(bool withScales)
    {
        const int n = 2816 + 5, k = 8;
        var ys = Enumerable.Range(0, k).Select(i => Random(n, 10 + i, 4f)).ToArray();
        float[] weights = Random(k, 30), scales = Random(k, 31, 2f);
        var expected = new float[n];
        for (int e = 0; e < k; e++)
            for (int d = 0; d < n; d++)
            {
                float src = withScales ? ys[e][d] * scales[e] : ys[e][d];
                expected[d] += weights[e] * src;
            }
        var actual = new float[n];
        var handles = ys.Select(y => GCHandle.Alloc(y, GCHandleType.Pinned)).ToArray();
        try
        {
            float** rows = stackalloc float*[k];
            for (int e = 0; e < k; e++) rows[e] = (float*)handles[e].AddrOfPinnedObject();
            fixed (float* a = actual) fixed (float* w = weights) fixed (float* s = scales)
                DiffusionGemmaCpuKernels.WeightedRowSum(a, n, rows, w, withScales ? s : null, k);
        }
        finally { foreach (var h in handles) h.Free(); }
        AssertSameBits(expected, actual);
    }

    // ---- router dot: the 4x4 F32 GEMM kernel's arithmetic, for every row, and close to float64 ----
    [Fact]
    public void RouterDot_MatchesReferenceArithmetic_Bitwise()
    {
        const int rows = 7, experts = 128, dim = 2816 + 3;   // odd row count and a scalar tail
        float[] x = Random(rows * dim, 40, 2f), w = Random(experts * dim, 41, 0.1f);
        fixed (float* xp = x) fixed (float* wp = w)
            for (int r = 0; r < rows; r++)
                for (int e = 0; e < experts; e++)
                {
                    float* a = xp + r * dim, b = wp + e * dim;
                    // one Vector<float> accumulator of a*b (multiply, then add), lane sum, in-order scalar tail
                    int vLen = Vector<float>.Count, i = 0;
                    Vector<float> acc = Vector<float>.Zero;
                    for (; i <= dim - vLen; i += vLen)
                        acc += Unsafe.ReadUnaligned<Vector<float>>(a + i) * Unsafe.ReadUnaligned<Vector<float>>(b + i);
                    float expected = Vector.Sum(acc);
                    for (; i < dim; i++) expected += a[i] * b[i];
                    double exact = 0;
                    for (int k = 0; k < dim; k++) exact += (double)a[k] * b[k];

                    float actual = DiffusionGemmaCpuKernels.RouterDot(a, b, dim);
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual));
                    Assert.True(Math.Abs(actual - exact) <= 1e-4, $"row {r} expert {e}: {actual} vs {exact}");
                }
    }

    // ---- attention ----

    /// <summary>The legacy CPU kernel (DiffusionGemmaModel.AttentionRegionAware) for one head/query:
    /// VecDot per key, max, scalar exp, in-order sum, then VecScaleAdd of V by e*inv.</summary>
    private static float[] LegacyAttention(float[] q, float[] keys, float[] values, int rows, int qHeads, int kvHeads,
        int hd, int[] lo, int[] hi)
    {
        int group = qHeads / kvHeads;
        var o = new float[rows * qHeads * hd];
        var scores = new float[keys.Length / (kvHeads * hd)];
        fixed (float* qp = q) fixed (float* kp = keys) fixed (float* vp = values) fixed (float* opp = o)
            for (int h = 0; h < qHeads; h++)
            {
                int kvHead = h / group;
                for (int qi = 0; qi < rows; qi++)
                {
                    int klo = lo[qi], khi = Math.Max(hi[qi], lo[qi] + 1);
                    float* qVec = qp + ((long)qi * qHeads + h) * hd;
                    float max = float.NegativeInfinity;
                    for (int kj = klo; kj < khi; kj++)
                    {
                        float dot = TensorComputePrimitives.Dot(qVec, kp + ((long)kj * kvHeads + kvHead) * hd, hd);
                        scores[kj] = dot;
                        if (dot > max) max = dot;
                    }
                    float sum = 0f;
                    for (int kj = klo; kj < khi; kj++)
                    {
                        float e = MathF.Exp(scores[kj] - max);
                        scores[kj] = e;
                        sum += e;
                    }
                    float inv = sum > 0f ? 1f / sum : 0f;
                    float* oVec = opp + ((long)qi * qHeads + h) * hd;
                    TensorComputePrimitives.Zero(oVec, hd);
                    for (int kj = klo; kj < khi; kj++)
                        TensorComputePrimitives.ScaleAdd(oVec, vp + ((long)kj * kvHeads + kvHead) * hd, scores[kj] * inv, hd);
                }
            }
        return o;
    }

    private static double[] DoubleAttention(float[] q, float[] keys, float[] values, int rows, int qHeads, int kvHeads,
        int hd, int[] lo, int[] hi)
    {
        int group = qHeads / kvHeads;
        var o = new double[rows * qHeads * hd];
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < qHeads; h++)
            {
                int g = h / group;
                int a = lo[r], e = Math.Max(hi[r], lo[r] + 1);
                var s = new double[e - a];
                double max = double.NegativeInfinity;
                for (int j = a; j < e; j++)
                {
                    double dot = 0;
                    for (int i = 0; i < hd; i++) dot += (double)q[(r * qHeads + h) * hd + i] * keys[(j * kvHeads + g) * hd + i];
                    s[j - a] = dot;
                    max = Math.Max(max, dot);
                }
                double sum = 0;
                for (int j = 0; j < s.Length; j++) { s[j] = Math.Exp(s[j] - max); sum += s[j]; }
                for (int j = a; j < e; j++)
                    for (int i = 0; i < hd; i++)
                        o[(r * qHeads + h) * hd + i] += s[j - a] / sum * values[(j * kvHeads + g) * hd + i];
            }
        return o;
    }

    private static float[] RunAttention(float[] q, int rows, int qHeads, int kvHeads, int hd, DiffusionCpuAttnGroup[] groups,
        DiffusionAttnKernel kernel)
    {
        var o = new float[rows * qHeads * hd];
        int items = DiffusionGemmaCpuKernels.CountAttendItems(groups, qHeads, out int[] starts);
        fixed (float* qp = q) fixed (float* op = o)
        {
            nint qa = (nint)qp, oa = (nint)op;
            Parallel.For(0, items, item => DiffusionGemmaCpuKernels.AttendItem(item, groups, starts,
                (float*)qa, (float*)oa, qHeads, kvHeads, hd, kernel));
        }
        return o;
    }

    private static (float[] q, float[] k, float[] v, int[] lo, int[] hi) AttentionCase(int qHeads, int kvHeads, int hd,
        int P, int C, int swa, int seed)
    {
        int N = P + C;
        float[] q = Random(N * qHeads * hd, seed, 0.4f);
        float[] k = Random(N * kvHeads * hd, seed + 1, 0.4f);
        float[] v = Random(N * kvHeads * hd, seed + 2, 2f);
        var lo = new int[N];
        var hi = new int[N];
        for (int i = 0; i < N; i++)
        {
            bool canvas = i >= P;
            lo[i] = canvas ? Math.Max(0, P - swa + 1) : Math.Max(0, i - swa + 1);
            hi[i] = canvas ? N : i + 1;
            if (!canvas && i >= 10 && i < 14) hi[i] = Math.Max(hi[i], 14);   // an image-style span
        }
        return (q, k, v, lo, hi);
    }

    [Theory]
    [MemberData(nameof(AttentionKernels))]
    public void Attention_PromptCausalSwaAndCanvasBidirectional_MatchesReference(int kernelId)
    {
        var kernel = (DiffusionAttnKernel)kernelId;
        foreach (var (qHeads, kvHeads, hd, swa) in new[] { (16, 8, 256, 24), (16, 2, 512, 1 << 20), (4, 4, 40, 9), (2, 1, 12, 5) })
        {
            const int P = 37, C = 11, N = P + C;
            var (q, k, v, lo, hi) = AttentionCase(qHeads, kvHeads, hd, P, C, swa, 100 + hd);
            float[] got;
            fixed (float* kp = k) fixed (float* vp = v)
                got = RunAttention(q, N, qHeads, kvHeads, hd,
                    new[] { new DiffusionCpuAttnGroup { QStart = 0, QCount = N, KA = (nint)kp, VA = (nint)vp, LenA = N, Klo = lo, Khi = hi } },
                    kernel);
            if (kernel == DiffusionAttnKernel.Exact)
            {
                // blocked and parallel, but the legacy per-pair arithmetic exactly
                AssertSameBits(LegacyAttention(q, k, v, N, qHeads, kvHeads, hd, lo, hi), got);
                continue;
            }
            double[] expected = DoubleAttention(q, k, v, N, qHeads, kvHeads, hd, lo, hi);
            double worst = 0;
            for (int i = 0; i < got.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expected[i]));
            Assert.True(worst <= 2e-5, $"{kernel} qHeads={qHeads} kvHeads={kvHeads} hd={hd}: max |err| {worst:G4}");
        }
    }

    // The property the prompt-KV decode rests on: a canvas query attending [cached prompt | fresh
    // canvas] as two separate segments gives BITWISE the unified result, even though the unified run
    // groups it with prompt queries (P is not a multiple of the micro-block) and the decode keeps only
    // the sliding-window tail of the prompt.
    [Theory]
    [MemberData(nameof(AttentionKernels))]
    public void Attention_SplitPromptCacheSegment_IsBitwiseTheUnifiedResult(int kernelId)
    {
        var kernel = (DiffusionAttnKernel)kernelId;
        const int qHeads = 16, kvHeads = 8, hd = 256, P = 45, C = 13, N = P + C, swa = 20;
        var (q, k, v, lo, hi) = AttentionCase(qHeads, kvHeads, hd, P, C, swa, 7);
        int kvStride = kvHeads * hd, qStride = qHeads * hd;
        float[] unified, split;
        fixed (float* kp = k) fixed (float* vp = v)
            unified = RunAttention(q, N, qHeads, kvHeads, hd,
                new[] { new DiffusionCpuAttnGroup { QStart = 0, QCount = N, KA = (nint)kp, VA = (nint)vp, LenA = N, Klo = lo, Khi = hi } },
                kernel);
        int stored = swa - 1;
        float[] promptK = k.AsSpan((P - stored) * kvStride, stored * kvStride).ToArray();
        float[] promptV = v.AsSpan((P - stored) * kvStride, stored * kvStride).ToArray();
        float[] canvasK = k.AsSpan(P * kvStride, C * kvStride).ToArray();
        float[] canvasV = v.AsSpan(P * kvStride, C * kvStride).ToArray();
        float[] canvasQ = q.AsSpan(P * qStride, C * qStride).ToArray();
        fixed (float* pk = promptK) fixed (float* pv = promptV) fixed (float* ck = canvasK) fixed (float* cv = canvasV)
            split = RunAttention(canvasQ, C, qHeads, kvHeads, hd, new[]
            {
                new DiffusionCpuAttnGroup
                {
                    QStart = 0, QCount = C, KA = (nint)pk, VA = (nint)pv, LenA = stored, KB = (nint)ck, VB = (nint)cv, LenB = C,
                    UniformLo = 0, UniformHi = stored + C,
                },
            }, kernel);
        AssertSameBits(unified.AsSpan(P * qStride).ToArray(), split);
    }
}
