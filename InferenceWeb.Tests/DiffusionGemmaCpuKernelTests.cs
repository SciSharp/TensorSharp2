// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Kernel-level contracts of the DiffusionGemma pure-C# (cpu backend) forward. The model is so
// sensitive to last-bit differences (Q8 activation quantization + top-8 routing) that end-to-end
// comparisons against the legacy CPU path only work bitwise, so every kernel that replaces an
// existing Ops chain or legacy loop is held to BITWISE equality with it here. The opt-in FMA
// attention tiles (DIFFUSION_CPU_ATTN_FAST=1) are held to a float64 reference instead, on both the
// Vector<T> (AVX2 / AdvSimd) and, where the hardware has it, the AVX-512 path.
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public sealed unsafe class DiffusionGemmaCpuKernelTests
{
    private readonly IAllocator _alloc = new CpuAllocator(BlasEnum.DotNet);

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

    private Tensor TensorFrom(float[] data, params long[] sizes)
    {
        var t = new Tensor(_alloc, DType.Float32, sizes);
        t.SetElementsAsFloat(data);
        return t;
    }

    private static void AssertSameBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"element {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    // ---- fused per-head RMSNorm + NeoX RoPE == Ops.RMSNorm then the scalar RoPE, bit for bit ----
    [Theory]
    [InlineData(8, 256, true, true)]
    [InlineData(2, 512, true, true)]
    [InlineData(8, 256, false, false)]
    [InlineData(3, 40, true, true)]      // head dim that is not a vector multiple
    public void HeadNormRope_MatchesOpsRmsNormThenScalarRope_Bitwise(int heads, int hd, bool weighted, bool rope)
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

        // reference: Ops.RMSNorm over [rows*heads, hd] (ones for the unweighted norm, as the model does)
        using var refT = TensorFrom(x, rows * heads, hd);
        using var gamma = TensorFrom(w ?? Enumerable.Repeat(1f, hd).ToArray(), hd);
        Ops.RMSNorm(refT, refT, gamma, null, eps);
        float[] expected = refT.GetElementsAsFloat(rows * heads * hd);
        if (rope)
        {
            for (int r = 0; r < rows; r++)
                for (int h = 0; h < heads; h++)
                    for (int j = 0; j < half; j++)
                    {
                        int b = (r * heads + h) * hd;
                        float c = cos[r * half + j], s = sin[r * half + j];
                        float x0 = expected[b + j], x1 = expected[b + j + half];
                        expected[b + j] = x0 * c - x1 * s;
                        expected[b + j + half] = x0 * s + x1 * c;
                    }
        }

        var actual = (float[])x.Clone();
        fixed (float* a = actual) fixed (float* wp = w) fixed (float* cp = cos) fixed (float* sp = sin)
            for (int r = 0; r < rows; r++)
                DiffusionGemmaCpuKernels.HeadNormRopeRow(a + r * heads * hd, a + r * heads * hd, heads, hd, wp, eps,
                    rope ? cp + r * half : null, rope ? sp + r * half : null);
        AssertSameBits(expected, actual);
    }

    // ---- GELU(gate)*up == Ops.GELUMul, bit for bit ----
    [Fact]
    public void GeluMul_MatchesOpsGeluMul_Bitwise()
    {
        int n = 4099;
        float[] gate = Random(n, 3, 8f), up = Random(n, 4, 3f);
        gate[0] = 0f; gate[1] = -0f; gate[2] = 40f; gate[3] = -40f; gate[4] = 1e-30f;
        using var g = TensorFrom(gate, n);
        using var u = TensorFrom(up, n);
        using var r = new Tensor(_alloc, DType.Float32, n);
        Ops.GELUMul(r, g, u);
        float[] expected = r.GetElementsAsFloat(n);
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

    // ---- router dot == the legacy linear (Ops.Addmm against the transposed F32 router weight) ----
    [Fact]
    public void RouterDot_MatchesLegacyF32Linear_ForFullRowBlocks_Bitwise()
    {
        const int rows = 8, experts = 128, dim = 2816;   // 8 rows: two full 4-row GEMM blocks
        float[] x = Random(rows * dim, 40, 2f), w = Random(experts * dim, 41, 0.1f);
        using var input = TensorFrom(x, rows, dim);
        using var weight = TensorFrom(w, experts, dim);
        using var result = new Tensor(_alloc, DType.Float32, rows, experts);
        using (var wT = weight.Transpose())
            Ops.Addmm(result, 0, result, 1.0f, input, wT);
        float[] expected = result.GetElementsAsFloat(rows * experts);
        var actual = new float[rows * experts];
        fixed (float* xp = x) fixed (float* wp = w)
            for (int r = 0; r < rows; r++)
                for (int e = 0; e < experts; e++)
                    actual[r * experts + e] = DiffusionGemmaCpuKernels.RouterDot(xp + r * dim, wp + e * dim, dim);
        AssertSameBits(expected, actual);
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
