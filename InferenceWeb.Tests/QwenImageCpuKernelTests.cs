// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.Intrinsics;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// The pure-C# (cpu backend) Qwen-Image-2.1 kernels against their scalar definitions: the
/// packed SGEMM (every kernel family this machine can run, so the AVX2 and portable kernels
/// are covered on AVX-512 hardware too), the implicit-im2col convolution, the VAE attention,
/// norm/SiLU/resampling passes, the text encoder's causal GQA and RoPE tables, and the vision
/// tower's erf GELU.
/// </summary>
[Collection("CpuPackedGemmIsa")]
public sealed unsafe class QwenImageCpuKernelTests
{
    // Kernel families by name (the enum is internal): every one this machine can execute.
    public static TheoryData<string> Isas()
    {
        var data = new TheoryData<string>();
        foreach (var isa in new[] { CpuGemmIsa.Avx512, CpuGemmIsa.Avx2, CpuGemmIsa.Portable })
            if (CpuPackedGemm.IsaSupported(isa)) data.Add(isa.ToString());
        return data;
    }

    private static float[] Random(Random rng, long n, float scale = 1f)
    {
        var v = new float[n];
        for (long i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return v;
    }

    private static void AssertClose(float[] expected, float[] actual, double relL2Limit, double normalizedMaxLimit, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sq = 0, ref2 = 0, maxErr = 0, maxRef = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}: non-finite at {i}");
            double d = actual[i] - expected[i];
            sq += d * d; ref2 += (double)expected[i] * expected[i];
            maxErr = Math.Max(maxErr, Math.Abs(d)); maxRef = Math.Max(maxRef, Math.Abs(expected[i]));
        }
        double relL2 = Math.Sqrt(sq / Math.Max(ref2, 1e-300)), normMax = maxErr / Math.Max(maxRef, 1e-30);
        Assert.True(relL2 <= relL2Limit, $"{what}: relL2 {relL2:E3} > {relL2Limit:E1}");
        Assert.True(normMax <= normalizedMaxLimit, $"{what}: max error / max |ref| {normMax:E3} > {normalizedMaxLimit:E1}");
    }

    private static T WithIsa<T>(CpuGemmIsa isa, Func<T> body)
    {
        CpuGemmIsa saved = CpuPackedGemm.Isa;
        try { CpuPackedGemm.Isa = isa; return body(); }
        finally { CpuPackedGemm.Isa = saved; }
    }

    // ---- GEMM ----------------------------------------------------------------------

    [Theory]
    [MemberData(nameof(Isas))]
    public void PackedGemmMatchesDoubleReferenceForEveryOperandForm(string isaName)
    {
        var isa = Enum.Parse<CpuGemmIsa>(isaName);
        var rng = new Random(11);
        // Odd sizes: partial row panels, partial column panels, uneven K chunks (K > KC).
        foreach (var (m, n, k) in new[] { (1, 1, 1), (13, 33, 7), (37, 70, 300), (50, 530, 523), (130, 9, 1025) })
        {
            float[] a = Random(rng, (long)m * k), b = Random(rng, (long)k * n), biasM = Random(rng, m), biasN = Random(rng, n);
            var expected = new float[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = biasM[i];
                    for (int t = 0; t < k; t++) acc += (double)a[i * k + t] * b[t * n + j];
                    expected[i * n + j] = (float)acc;
                }
            var bt = new float[n * k];   // B^T, i.e. B read column-major
            for (int t = 0; t < k; t++) for (int j = 0; j < n; j++) bt[j * k + t] = b[t * n + j];

            fixed (float* pa = a, pb = b, pbt = bt, pbm = biasM, pbn = biasN)
            {
                var packedA = CpuPackedGemm.PackA(pa, m, k, k, 1, isa);
                // B row-major, per-row bias.
                var c1 = new float[m * n];
                fixed (float* pc = c1) CpuPackedGemm.Gemm(packedA, new StridedPanelSource(pb, n, 1, n), n, pc, n, biasM: pbm);
                AssertClose(expected, c1, 1e-5, 1e-5, $"{isa} {m}x{n}x{k} row-major B");

                // B column-major (B^T given), prepacked once.
                var packedB = CpuPackedGemm.PackB(pbt, k, n, 1, k, isa);
                var c2 = new float[m * n];
                fixed (float* pc = c2) CpuPackedGemm.Gemm(packedA, new PrepackedPanelSource(packedB), n, pc, n, biasM: pbm);
                AssertClose(expected, c2, 1e-5, 1e-5, $"{isa} {m}x{n}x{k} prepacked B");

                // A given transposed (column access), per-column bias, then accumulate twice.
                var at = new float[k * m];
                for (int i = 0; i < m; i++) for (int t = 0; t < k; t++) at[t * m + i] = a[i * k + t];
                fixed (float* pat = at)
                {
                    var packedAt = CpuPackedGemm.PackA(pat, m, k, 1, m, isa);
                    var c3 = new float[m * n];
                    fixed (float* pc = c3)
                    {
                        CpuPackedGemm.Gemm(packedAt, new StridedPanelSource(pbt, 1, k, n), n, pc, n, biasN: pbn);
                        CpuPackedGemm.Gemm(packedAt, new StridedPanelSource(pbt, 1, k, n), n, pc, n, accumulate: true);
                    }
                    var expected3 = new float[m * n];
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                            expected3[i * n + j] = 2 * (expected[i * n + j] - biasM[i]) + biasN[j];
                    AssertClose(expected3, c3, 1e-5, 1e-5, $"{isa} {m}x{n}x{k} transposed A + accumulate");
                }

                // Row sub-range (whole panels), C pointing at the first selected row.
                int mr = CpuPackedGemm.Mr(isa);
                if (m > mr)
                {
                    int panels = (m - mr + mr - 1) / mr;
                    var c4 = new float[m * n];
                    fixed (float* pc = c4)
                        CpuPackedGemm.Gemm(packedA, new StridedPanelSource(pb, n, 1, n), n, pc + mr * n, n, biasM: pbm,
                            panel0: 1, panelCount: panels);
                    for (int i = 0; i < mr; i++) for (int j = 0; j < n; j++) Assert.Equal(0f, c4[i * n + j]);
                    AssertClose(expected[(mr * n)..], c4[(mr * n)..], 1e-5, 1e-5, $"{isa} {m}x{n}x{k} row sub-range");
                }
            }
        }
    }

    // ---- convolution ---------------------------------------------------------------

    [Theory]
    [MemberData(nameof(Isas))]
    public void PackedConvolutionMatchesScalarConvolution(string isaName)
    {
        var isa = Enum.Parse<CpuGemmIsa>(isaName);
        // (ic, oc, h, w, k, stride, padT, padB, upsample)
        var cases = new[]
        {
            (8, 12, 16, 16, 3, 1, 1, 1, false),    // panel wraps image rows (16-wide map)
            (5, 13, 9, 37, 3, 1, 1, 1, false),     // odd width, partial panels everywhere
            (6, 7, 37, 41, 1, 1, 0, 0, false),     // pointwise
            (9, 11, 38, 40, 3, 2, 0, 1, false),    // encoder downsample: stride 2, pad (0,1)
            (7, 9, 11, 13, 3, 1, 1, 1, true),      // decoder resample: reads through a 2x upsample
            (144, 4, 20, 70, 3, 1, 1, 1, false),   // head conv: OC below one row panel
            (40, 24, 3, 5, 3, 1, 1, 1, false),     // K = 360 > KC and tiny maps
        };
        var rng = new Random(5);
        foreach (var (ic, oc, h, w, kk, stride, padT, padB, up) in cases)
        {
            var x = new Feature(ic, h, w, Random(rng, (long)ic * h * w));
            float[] weight = Random(rng, (long)oc * ic * kk * kk, 0.3f), bias = Random(rng, oc);
            var source = up ? Upsample(x) : x;
            var reference = VaeReferenceMath.Conv2dScalar(source, weight, oc, ic, kk, kk, bias, stride, stride, padT, padB, padT, padB);
            var actual = WithIsa(isa, () => VaeReferenceMath.Conv2dCpu(x, VaeReferenceMath.PackConvWeight(weight, oc, ic * kk * kk),
                bias, oc, kk, kk, stride, stride, padT, padB, padT, padB, up));
            Assert.Equal((reference.C, reference.H, reference.W), (actual.C, actual.H, actual.W));
            AssertClose(reference.D, actual.D, 1e-5, 1e-5, $"{isa} conv ic={ic} oc={oc} {h}x{w} k={kk} s={stride} up={up}");
        }
    }

    [Avx512AndAvx2Fact]
    public void KernelWidthDoesNotChangeTheBits()
    {
        // Every output accumulates one FMA per k in k order whatever the tile shape, so the
        // AVX-512 and AVX2 kernels must agree exactly (PoolWidthDoesNotChangeTheBits covers
        // the thread count).
        var rng = new Random(41);
        var x = new Feature(40, 21, 35, Random(rng, 40 * 21 * 35));
        float[] weight = Random(rng, 50 * 40 * 9, 0.2f), bias = Random(rng, 50);
        Feature Run(CpuGemmIsa isa) => WithIsa(isa, () => VaeReferenceMath.Conv2dCpu(x,
            VaeReferenceMath.PackConvWeight(weight, 50, 360), bias, 50, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.Equal(Run(CpuGemmIsa.Avx512).D, Run(CpuGemmIsa.Avx2).D);
    }

    [Fact]
    public void Conv2dDispatchesToThePackedPathAndHonoursTheScalarSwitch()
    {
        var rng = new Random(3);
        var x = new Feature(6, 10, 12, Random(rng, 6 * 10 * 12));
        float[] weight = Random(rng, 8 * 6 * 9), bias = Random(rng, 8);
        bool savedGpu = VaeReferenceMath.UseGpuConv, savedScalar = VaeReferenceMath.UseScalarCpu;
        try
        {
            VaeReferenceMath.UseGpuConv = false;
            VaeReferenceMath.UseScalarCpu = true;
            var scalar = VaeReferenceMath.Conv2d(x, weight, 8, 6, 3, 3, bias, 1, 1, 1, 1, 1, 1);
            Assert.Equal(VaeReferenceMath.Conv2dScalar(x, weight, 8, 6, 3, 3, bias, 1, 1, 1, 1, 1, 1).D, scalar.D);
            VaeReferenceMath.UseScalarCpu = false;
            var packed = VaeReferenceMath.Conv2d(x, weight, 8, 6, 3, 3, bias, 1, 1, 1, 1, 1, 1);
            AssertClose(scalar.D, packed.D, 1e-5, 1e-5, "Conv2d packed");
        }
        finally
        {
            VaeReferenceMath.UseGpuConv = savedGpu;
            VaeReferenceMath.UseScalarCpu = savedScalar;
        }
    }

    private static Feature Upsample(Feature x)
    {
        var y = new Feature(x.C, 2 * x.H, 2 * x.W);
        for (int c = 0; c < x.C; c++)
            for (int oy = 0; oy < y.H; oy++)
                for (int ox = 0; ox < y.W; ox++)
                    y.D[(c * y.H + oy) * y.W + ox] = x.D[(c * x.H + oy / 2) * x.W + ox / 2];
        return y;
    }

    // ---- VAE attention / elementwise ----------------------------------------------

    [Theory]
    [MemberData(nameof(Isas))]
    public void BlockedVaeAttentionMatchesScalarAttention(string isaName)
    {
        var isa = Enum.Parse<CpuGemmIsa>(isaName);
        var rng = new Random(9);
        foreach (var (c, h, w) in new[] { (16, 5, 7), (48, 9, 13), (24, 16, 17) })
        {
            int hw = h * w;
            float[] qkv = Random(rng, 3L * c * hw, 2f);
            // The original scalar loop of VaeReferenceMath.AttentionBlock.
            var expected = new float[c * hw];
            float scale = 1f / MathF.Sqrt(c);
            for (int i = 0; i < hw; i++)
            {
                var scores = new double[hw];
                double mx = double.NegativeInfinity;
                for (int j = 0; j < hw; j++)
                {
                    double s = 0;
                    for (int ch = 0; ch < c; ch++) s += (double)qkv[ch * hw + i] * qkv[(c + ch) * hw + j];
                    scores[j] = s * scale;
                    mx = Math.Max(mx, scores[j]);
                }
                double sum = 0;
                for (int j = 0; j < hw; j++) { scores[j] = Math.Exp(scores[j] - mx); sum += scores[j]; }
                for (int ch = 0; ch < c; ch++)
                {
                    double acc = 0;
                    for (int j = 0; j < hw; j++) acc += scores[j] * qkv[(2 * c + ch) * hw + j];
                    expected[ch * hw + i] = (float)(acc / sum);
                }
            }
            var actual = WithIsa(isa, () => VaeReferenceMath.AttentionCpu(qkv, c, h, w));
            AssertClose(expected, actual.D, 2e-6, 2e-5, $"{isa} attention C={c} {h}x{w}");
        }
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(144, 1, 1025)]
    [InlineData(96, 33, 70)]
    public void FusedNormSiluMatchesScalarNormThenSilu(int channels, int height, int width)
    {
        var rng = new Random(17);
        var x = new Feature(channels, height, width, Random(rng, (long)channels * height * width, 30f));
        float[] gamma = Random(rng, channels, 2f);
        var exact = VaeReferenceMath.RmsNormChannel(x, gamma);
        var expected = (float[])exact.D.Clone();
        VaeReferenceMath.SiluInPlace(expected);
        var fused = VaeReferenceMath.RmsNormChannelFast(x, gamma, silu: true);
        AssertClose(expected, fused.D, 1e-6, 1e-6, "norm+silu");
        var fast = (float[])exact.D.Clone();
        VaeReferenceMath.SiluInPlaceFast(fast);
        AssertClose(expected, fast, 1e-6, 1e-6, "silu");
        // Extreme inputs saturate the same way as the scalar formula.
        float[] edges = { -1000f, -100f, -88f, -20f, -0f, 0f, 1e-30f, 20f, 88f, 100f, 1000f };
        var edgeFast = (float[])edges.Clone();
        var edgeExact = (float[])edges.Clone();
        VaeReferenceMath.SiluInPlaceFast(edgeFast);
        VaeReferenceMath.SiluInPlace(edgeExact);
        for (int i = 0; i < edges.Length; i++)
            Assert.True(Math.Abs(edgeFast[i] - edgeExact[i]) <= 1e-6 * Math.Max(1, Math.Abs(edgeExact[i])), $"silu({edges[i]})");
    }

    [Theory]
    [InlineData(4, 1, 2, 2, 12, 20)]      // decoder shortcut: channel repeat + 2x spatial
    [InlineData(6, 2, 2, 12, 8, 10)]      // time factor 2
    [InlineData(8, 1, 1, 8, 5, 7)]
    public void ParallelResamplingMatchesSerialDefinitionExactly(int inChannels, int timeFactor, int spatial, int outChannels, int h, int w)
    {
        var rng = new Random(23);
        var x = new Feature(inChannels, h, w, Random(rng, (long)inChannels * h * w));
        // DuplicateUp21 as originally written (serial).
        int repeats = outChannels * timeFactor * spatial * spatial / inChannels;
        var up = new Feature(outChannels, h * spatial, w * spatial);
        for (int oc = 0; oc < outChannels; oc++)
            for (int yy = 0; yy < up.H; yy++)
                for (int xx = 0; xx < up.W; xx++)
                {
                    int packed = ((oc * timeFactor + timeFactor - 1) * spatial + yy % spatial) * spatial + xx % spatial;
                    up.D[(oc * up.H + yy) * up.W + xx] = x.D[(packed / repeats * h + yy / spatial) * w + xx / spatial];
                }
        Assert.Equal(up.D, VaeReferenceMath.DuplicateUp21(x, outChannels, timeFactor, spatial).D);

        // AverageDown21 as originally written, applied to the upsampled map.
        int downOut = inChannels;
        int group = up.C * timeFactor * spatial * spatial / downOut;
        var down = new Feature(downOut, up.H / spatial, up.W / spatial);
        for (int oc = 0; oc < downOut; oc++)
            for (int g = 0; g < group; g++)
            {
                int packed = oc * group + g;
                int ic = packed / (timeFactor * spatial * spatial);
                int t = packed / (spatial * spatial) % timeFactor;
                if (t != timeFactor - 1) continue;
                int dy = packed / spatial % spatial, dx = packed % spatial;
                for (int yy = 0; yy < down.H; yy++)
                    for (int xx = 0; xx < down.W; xx++)
                        down.D[(oc * down.H + yy) * down.W + xx] += up.D[(ic * up.H + yy * spatial + dy) * up.W + xx * spatial + dx] / group;
            }
        Assert.Equal(down.D, VaeReferenceMath.AverageDown21(up, downOut, timeFactor, spatial).D);
    }

    // ---- quantized weight panels ---------------------------------------------------

    // Random but well-formed blocks: every byte random except the f16 scales, kept small and finite.
    private static byte[] RandomQuantRows(Random rng, GgmlTensorType type, int rows, int cols)
    {
        long rowBytes = ManagedQuantizedOps.RowSize((int)type, cols);
        var data = new byte[rowBytes * rows];
        rng.NextBytes(data);
        int blockBytes = (int)TensorSharp.Runtime.GgufFile.GetTypeSize(type);
        int[] halfOffsets = type switch
        {
            GgmlTensorType.Q4_K => new[] { 0, 2 },
            GgmlTensorType.Q6_K => new[] { 208 },
            GgmlTensorType.Q8_0 => new[] { 0 },
            _ => throw new NotSupportedException(type.ToString()),
        };
        for (long b = 0; b < data.Length; b += blockBytes)
            foreach (int off in halfOffsets)
                BitConverter.TryWriteBytes(data.AsSpan((int)(b + off), 2), (System.Half)((float)rng.NextDouble() * 0.01f + 0.001f));
        return data;
    }

    [Theory]
    [InlineData(GgmlTensorType.Q4_K)]
    [InlineData(GgmlTensorType.Q6_K)]
    [InlineData(GgmlTensorType.Q8_0)]
    public void QuantizedRowPanelsHoldTheScalarDequantAndFeedTheGemm(GgmlTensorType type)
    {
        var rng = new Random((int)type);
        int outDim = 45, inDim = 512, seq = 13;   // 45: a partial 8-row group and panel
        byte[] weights = RandomQuantRows(rng, type, outDim, inDim);
        long rowBytes = ManagedQuantizedOps.RowSize((int)type, inDim);
        var dense = new float[outDim * inDim];
        fixed (byte* wp = weights)
        fixed (float* dp = dense)
            for (int o = 0; o < outDim; o++)
                ManagedQuantizedOps.DequantizeRowToFloat32((int)type, (IntPtr)(wp + o * rowBytes), dp + o * inDim, inDim);

        foreach (var isa in new[] { CpuGemmIsa.Avx512, CpuGemmIsa.Avx2, CpuGemmIsa.Portable })
        {
            if (!CpuPackedGemm.IsaSupported(isa)) continue;
            int nr = CpuPackedGemm.Nr(isa);
            fixed (byte* wp = weights)
            {
                var source = new QuantRowsPanelSource((IntPtr)wp, (int)type, inDim, outDim);
                // Panels for rows [256, 512) of K and every column: exactly the scalar dequant.
                int panels = (outDim + nr - 1) / nr;
                var scratch = new float[panels * 256 * nr + 16];
                fixed (float* sp = scratch)
                {
                    float* p = source.Panels(256, 256, 0, outDim, nr, sp, out long stride);
                    for (int n = 0; n < panels * nr; n++)
                        for (int k = 0; k < 256; k++)
                        {
                            float expected = n < outDim ? dense[n * inDim + 256 + k] : 0f;
                            Assert.Equal(expected, p[n / nr * stride + (long)k * nr + n % nr]);
                        }
                }

                float[] x = Random(rng, (long)seq * inDim);
                var expectedY = new float[seq * outDim];
                for (int r = 0; r < seq; r++)
                    for (int o = 0; o < outDim; o++)
                    {
                        double acc = 0;
                        for (int k = 0; k < inDim; k++) acc += (double)x[r * inDim + k] * dense[o * inDim + k];
                        expectedY[r * outDim + o] = (float)acc;
                    }
                var y = new float[seq * outDim];
                fixed (float* xp = x, yp = y)
                    CpuPackedGemm.Gemm(CpuPackedGemm.PackA(xp, seq, inDim, inDim, 1, isa), source, outDim, yp, outDim);
                AssertClose(expectedY, y, 1e-5, 1e-5, $"{isa} {type} linear");
            }
        }
    }

    // The row-block split, the shared B pack and the K chunking all change with the pool width
    // and TS_CPU_GEMM_KC; none may change a bit (C round-trips through memory as the same float).
    [Theory]
    [MemberData(nameof(Isas))]
    public void QuantizedRowGemmKeepsItsBitsThroughTheSharedPackAndAnyKChunk(string isaName)
    {
        var isa = Enum.Parse<CpuGemmIsa>(isaName);
        // 150 rows x 64 outputs: few column tiles, many row panels, so a 4-thread pool splits
        // rows into >= 3 blocks and packs B once for all of them (the path attn_k / attn_v take
        // for a 133+ token prompt); the single-block serial run is the reference.
        int outDim = 64, inDim = 512, seq = 150;
        using var pool = new CpuWorkerPool(4);
        int savedKc = CpuPackedGemm.KcBlock;
        try
        {
            foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q6_K, GgmlTensorType.Q8_0 })
            {
                var rng = new Random(100 + (int)type);
                byte[] weights = RandomQuantRows(rng, type, outDim, inDim);
                float[] x = Random(rng, (long)seq * inDim);
                long rowBytes = ManagedQuantizedOps.RowSize((int)type, inDim);
                var dense = new float[outDim * inDim];
                fixed (byte* wp = weights)
                fixed (float* dp = dense)
                    for (int o = 0; o < outDim; o++)
                        ManagedQuantizedOps.DequantizeRowToFloat32((int)type, (IntPtr)(wp + o * rowBytes), dp + o * inDim, inDim);
                var expected = new float[seq * outDim];
                for (int r = 0; r < seq; r++)
                    for (int o = 0; o < outDim; o++)
                    {
                        double acc = 0;
                        for (int k = 0; k < inDim; k++) acc += (double)x[r * inDim + k] * dense[o * inDim + k];
                        expected[r * outDim + o] = (float)acc;
                    }

                float[] Run(bool serial)
                {
                    var y = new float[seq * outDim];
                    fixed (byte* wp = weights)
                    fixed (float* xp = x, yp = y)
                        CpuPackedGemm.Gemm(CpuPackedGemm.PackA(xp, seq, inDim, inDim, 1, isa),
                            new QuantRowsPanelSource((IntPtr)wp, (int)type, inDim, outDim), outDim, yp, outDim,
                            pool: pool, serial: serial);
                    return y;
                }

                CpuPackedGemm.KcBlock = 256;
                float[] reference = Run(serial: true);
                AssertClose(expected, reference, 1e-5, 1e-5, $"{isa} {type} serial");
                long sharedBefore = CpuPackedGemm.SharedPrepackRuns;
                Assert.Equal(reference, Run(serial: false));
                Assert.True(CpuPackedGemm.SharedPrepackRuns > sharedBefore, $"{isa} {type}: the shared B pack did not run");
                // TS_CPU_GEMM_KC values whose balanced chunk is not block-aligned (128, and 384,
                // which balances K = 512 to 256 + 256 only after rounding) used to throw from the
                // quantized panel source.
                foreach (int kc in new[] { 16, 128, 384, 4096 })
                {
                    CpuPackedGemm.KcBlock = kc;
                    Assert.Equal(reference, Run(serial: false));
                    Assert.Equal(reference, Run(serial: true));
                }
                CpuPackedGemm.KcBlock = savedKc;
            }
        }
        finally
        {
            CpuPackedGemm.KcBlock = savedKc;
        }
    }

    [Theory]
    [MemberData(nameof(Isas))]
    public void PoolWidthDoesNotChangeTheBits(string isaName)
    {
        var isa = Enum.Parse<CpuGemmIsa>(isaName);
        var rng = new Random(43);
        // A 3x3 conv through the implicit im2col, and a strided linear with more row panels
        // than column tiles (row blocks and the shared B pack at 3+ threads).
        int ic = 24, oc = 96, h = 20, w = 30;
        float[] x = Random(rng, (long)ic * h * w), weight = Random(rng, (long)oc * ic * 9, 0.2f), bias = Random(rng, oc);
        int m = 200, n = 40, k = 300;
        float[] a = Random(rng, (long)m * k), b = Random(rng, (long)k * n);
        using var pool3 = new CpuWorkerPool(3);
        using var pool4 = new CpuWorkerPool(4);
        fixed (float* xp = x, wp = weight, bp = bias, ap = a, bbp = b)
        {
            var packedW = CpuPackedGemm.PackA(wp, oc, ic * 9, ic * 9, 1, isa);
            var packedA = CpuPackedGemm.PackA(ap, m, k, k, 1, isa);
            nint xL = (nint)xp, bL = (nint)bp, bbL = (nint)bbp;
            float[] Conv(CpuWorkerPool p, bool serial)
            {
                var y = new float[oc * h * w];
                fixed (float* yp = y)
                    CpuPackedGemm.Gemm(packedW, new Im2colPanelSource((float*)xL, h, w, 3, 3, 1, 1, 1, 1, h, w), h * w, yp, h * w,
                        biasM: (float*)bL, pool: p, serial: serial);
                return y;
            }
            float[] Linear(CpuWorkerPool p, bool serial)
            {
                var y = new float[m * n];
                fixed (float* yp = y)
                    CpuPackedGemm.Gemm(packedA, new StridedPanelSource((float*)bbL, n, 1, n), n, yp, n, pool: p, serial: serial);
                return y;
            }
            float[] conv = Conv(pool3, serial: true), linear = Linear(pool3, serial: true);
            foreach (var p in new[] { pool3, pool4, CpuWorkerPool.Shared })
            {
                Assert.Equal(conv, Conv(p, serial: false));
                Assert.Equal(linear, Linear(p, serial: false));
            }
        }
    }

    // ---- text encoder / vision tower pieces ----------------------------------------

    [Theory]
    [InlineData(1, 4, 2, 8)]
    [InlineData(37, 8, 2, 128)]
    [InlineData(70, 32, 8, 128)]
    public void CausalGqaAttentionMatchesMaskedSoftmaxReference(int seq, int heads, int kvHeads, int dim)
    {
        var rng = new Random(29);
        float[] q = Random(rng, (long)seq * heads * dim, 3f), k = Random(rng, (long)seq * kvHeads * dim, 3f),
            v = Random(rng, (long)seq * kvHeads * dim);
        float scale = 1f / MathF.Sqrt(dim);
        var expected = new float[seq * heads * dim];
        int group = heads / kvHeads;
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < seq; i++)
            {
                int kvh = h / group;   // RepeatInterleave of the KV heads
                var s = new double[i + 1];
                double mx = double.NegativeInfinity, sum = 0;
                for (int j = 0; j <= i; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < dim; d++) dot += (double)q[(i * heads + h) * dim + d] * k[(j * kvHeads + kvh) * dim + d];
                    s[j] = dot * scale; mx = Math.Max(mx, s[j]);
                }
                for (int j = 0; j <= i; j++) { s[j] = Math.Exp(s[j] - mx); sum += s[j]; }
                for (int d = 0; d < dim; d++)
                {
                    double acc = 0;
                    for (int j = 0; j <= i; j++) acc += s[j] * v[(j * kvHeads + kvh) * dim + d];
                    expected[(i * heads + h) * dim + d] = (float)(acc / sum);
                }
            }
        var actual = new float[expected.Length];
        fixed (float* pq = q, pk = k, pv = v, po = actual)
            QwenImageTextEncoder.CausalGqaAttention(pq, pk, pv, po, seq, heads, kvHeads, dim, scale);
        AssertClose(expected, actual, 2e-6, 2e-5, $"gqa seq={seq}");
    }

    [Fact]
    public void RopeTablesHoldTheInlineAngles()
    {
        var imgs = new[] { new ImageCond { Start = 3, Count = 4, GridH = 4, GridW = 4 } };
        int seq = 12, headDim = 128;
        int[] pos = QwenImageTextEncoder.BuildPositions(seq, imgs);
        var (cos, sin) = QwenImageTextEncoder.BuildRopeTables(pos, seq, headDim, 5_000_000f);
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < headDim / 2; i++)
            {
                // The per-element expression the per-op path evaluated before the table.
                float freq = (float)Math.Pow(5_000_000f, -2.0 * i / headDim);
                float ang = pos[QwenImageTextEncoder.InterleavedRopeAxis(i) * seq + s] * freq;
                Assert.Equal(MathF.Cos(ang), cos[s * (headDim / 2) + i]);
                Assert.Equal(MathF.Sin(ang), sin[s * (headDim / 2) + i]);
            }
    }

    [Theory]
    [MemberData(nameof(Isas))]
    public void VisionAttentionMatchesSoftmaxReference(string isaName)
    {
        var isa = Enum.Parse<CpuGemmIsa>(isaName);
        var rng = new Random(37);
        using var pool = new CpuWorkerPool(3);
        var workspace = new CpuAttentionWorkspace();
        // 72 is the Qwen3-VL vision head dim; 130 and 37 leave partial query blocks and partial
        // row panels, 13 a dim that is no whole number of panels; the workspace is reused across
        // the growing, then shrinking, calls.
        foreach (var (n, heads, dim) in new[] { (1, 2, 8), (37, 3, 72), (130, 4, 72), (64, 2, 13) })
        {
            float[] q = Random(rng, (long)n * heads * dim, 2f), k = Random(rng, (long)n * heads * dim, 2f),
                v = Random(rng, (long)n * heads * dim);
            float scale = 1f / MathF.Sqrt(dim);
            int stride = heads * dim;
            var expected = new float[n * stride];
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < n; i++)
                {
                    var s = new double[n];
                    double mx = double.NegativeInfinity, sum = 0;
                    for (int j = 0; j < n; j++)
                    {
                        double dot = 0;
                        for (int d = 0; d < dim; d++) dot += (double)q[i * stride + h * dim + d] * k[j * stride + h * dim + d];
                        s[j] = dot * scale; mx = Math.Max(mx, s[j]);
                    }
                    for (int j = 0; j < n; j++) { s[j] = Math.Exp(s[j] - mx); sum += s[j]; }
                    for (int d = 0; d < dim; d++)
                    {
                        double acc = 0;
                        for (int j = 0; j < n; j++) acc += s[j] * v[j * stride + h * dim + d];
                        expected[i * stride + h * dim + d] = (float)(acc / sum);
                    }
                }
            var actual = new float[expected.Length];
            WithIsa(isa, () =>
            {
                fixed (float* pq = q, pk = k, pv = v, po = actual)
                    CpuFullAttention.Run(pq, pk, pv, po, n, heads, dim, scale, workspace, pool);
                return 0;
            });
            AssertClose(expected, actual, 3e-6, 3e-5, $"{isa} vision attention n={n} heads={heads} dim={dim}");
        }
    }

    [Fact]
    public void SoftmaxExpStaysWithinAFewUlpOfMathFExp()
    {
        // The attention softmax's polynomial exp over its whole domain (x <= 0), both widths.
        double worst = 0;
        var lanes = new float[16];
        for (int step = 0; step <= 1_000_000; step += 16)
        {
            for (int l = 0; l < 16; l++) lanes[l] = -87f * Math.Min(step + l, 1_000_000) / 1_000_000f;   // [-87, 0]
            fixed (float* lp = lanes)
            {
                var e256 = CpuFullAttention.ExpNonPositive(Vector256.Load(lp));
                var e512 = CpuFullAttention.ExpNonPositive(Vector512.Load(lp));
                for (int l = 0; l < 16; l++)
                {
                    double exact = Math.Exp(lanes[l]);
                    worst = Math.Max(worst, Math.Abs(e512.GetElement(l) - exact) / exact);
                    if (l < 8) worst = Math.Max(worst, Math.Abs(e256.GetElement(l) - exact) / exact);
                }
            }
        }
        Assert.True(worst < 4e-7, $"worst relative error {worst:E2}");
        // Saturation: far below the range it stays a tiny normal float, 0 maps to exactly 1,
        // and NaN propagates.
        var edge = CpuFullAttention.ExpNonPositive(Vector256.Create(-1000f, -88f, -87f, 0f, -0f, float.NaN, -1e-30f, -20f));
        Assert.InRange(edge.GetElement(0), 0f, 2e-38f);
        Assert.InRange(edge.GetElement(1), 0f, 2e-38f);
        Assert.Equal(1f, edge.GetElement(3));
        Assert.Equal(1f, edge.GetElement(4));
        Assert.True(float.IsNaN(edge.GetElement(5)));
        Assert.Equal(1f, edge.GetElement(6));
    }

    [Fact]
    public void RawArrayConvolutionRejectsMismatchedWeightsAndBias()
    {
        // The packed path reads these arrays through raw pointers, so a short weight, bias,
        // feature or qkv must fail loudly (as the bounds-checked scalar loops did).
        var x = new Feature(4, 5, 6, new float[4 * 5 * 6]);
        Assert.Throws<ArgumentException>(() => VaeReferenceMath.PackConvWeight(new float[8 * 4 * 9 - 1], 8, 4 * 9));
        var packed = VaeReferenceMath.PackConvWeight(new float[8 * 4 * 9], 8, 4 * 9);
        Assert.Throws<ArgumentException>(() => VaeReferenceMath.Conv2dCpu(x, packed, new float[7], 8, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.Throws<ArgumentException>(() => VaeReferenceMath.Conv2dCpu(new Feature(4, 5, 6, new float[4 * 5 * 6 - 1]),
            packed, new float[8], 8, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.Equal(8 * 5 * 6, VaeReferenceMath.Conv2dCpu(x, packed, null, 8, 3, 3, 1, 1, 1, 1, 1, 1).D.Length);
        Assert.Throws<ArgumentException>(() => VaeReferenceMath.AttentionCpu(new float[3 * 4 * 30 - 1], 4, 5, 6));
    }

    [Fact]
    public void VectorizedErfGeluMatchesHostLoop()
    {
        var rng = new Random(31);
        float[] values = Random(rng, 50_003, 8f);
        values[0] = 0f; values[1] = -0f; values[2] = 30f; values[3] = -30f; values[4] = 1e-30f;
        var expected = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            // Qwen35VisionEncoder.ApplyVisionGelu's host loop.
            double x = values[i] / Math.Sqrt(2.0);
            double sign = x < 0 ? -1.0 : 1.0;
            x = Math.Abs(x);
            double t = 1.0 / (1.0 + 0.3275911 * x);
            double p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
            double erf = sign * (1.0 - p * Math.Exp(-x * x));
            expected[i] = (float)(0.5 * values[i] * (1.0 + erf));
        }
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var tensor = new Tensor(allocator, DType.Float32, values.Length);
        tensor.SetElementsAsFloat(values);
        Qwen35VisionEncoder.GeluErfInPlace(tensor);
        float[] actual = tensor.GetElementsAsFloat(values.Length);
        int different = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (expected[i] != actual[i]) different++;
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f * Math.Max(1f, Math.Abs(expected[i])), $"gelu({values[i]})");
        }
        // Only the double-precision exp differs; almost every value rounds to the same float.
        Assert.True(different <= values.Length / 1000, $"{different} values differ");
    }
}

[CollectionDefinition("CpuPackedGemmIsa", DisableParallelization = true)]
public sealed class CpuPackedGemmIsaCollection { }

/// <summary>Skipped (reported as such, not passed) on hardware without both AVX-512 and AVX2+FMA.</summary>
public sealed class Avx512AndAvx2FactAttribute : FactAttribute
{
    public Avx512AndAvx2FactAttribute()
    {
        if (!CpuPackedGemm.IsaSupported(CpuGemmIsa.Avx512) || !CpuPackedGemm.IsaSupported(CpuGemmIsa.Avx2))
            Skip = "Compares the AVX-512 and AVX2 GEMM kernels; this CPU lacks one of them.";
    }
}
