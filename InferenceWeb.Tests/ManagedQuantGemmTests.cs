using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Xunit.Abstractions;
using QGemmIsa = TensorSharp.Models.ManagedQuantizedOps.QGemmIsa;

namespace InferenceWeb.Tests;

/// <summary>
/// The multi-row quantized GEMM (ManagedQuantGemm*.cs) against the per-row dot
/// path it replaces and against a dequantize-then-dot reference, on both the
/// AVX-512 and the AVX2 kernels.
///
/// The new path quantizes activations to the same int8 values and scales and
/// computes the same integer sub-block sums, so it may differ from the per-row
/// path only by float re-association: the tolerance is 2e-5 of the output's
/// max magnitude (observed ~1e-6). The dequant reference differs by the Q8
/// activation quantization itself, so it gets a loose tolerance and only
/// guards against a kernel that is consistently wrong on both paths.
/// </summary>
public class ManagedQuantGemmTests
{
    private const float LegacyRelTol = 2e-5f;
    private readonly ITestOutputHelper _output;

    public ManagedQuantGemmTests(ITestOutputHelper output) => _output = output;

    private static readonly int[] RowCounts = { 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64, 70, 257 };

    public static IEnumerable<object[]> GemmCases()
    {
        foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q5_K, GgmlTensorType.Q6_K })
            foreach (int k in new[] { 256, 2816 })
                foreach (int rows in RowCounts)
                    yield return new object[] { type, rows, k, 37 };
        foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q6_K })
            foreach (int rows in new[] { 1, 8, 70 })
                yield return new object[] { type, rows, 4096, 21 };
        foreach (var type in new[] { GgmlTensorType.Q8_0, GgmlTensorType.Q5_0, GgmlTensorType.Q4_0 })
            foreach (int k in new[] { 32, 704, 2112 })
                foreach (int rows in RowCounts)
                    yield return new object[] { type, rows, k, 37 };
        foreach (var type in new[] { GgmlTensorType.Q8_0, GgmlTensorType.Q5_0 })
            foreach (int rows in new[] { 4, 70 })
                yield return new object[] { type, rows, 4096, 21 };
        // K = 12288 (the Qwen-Image DiT's MLP down projection): the pair scratch
        // (48 super-blocks) outgrows the L1 and the rows split into L2 blocks.
        foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q5_K, GgmlTensorType.Q6_K,
                                     GgmlTensorType.Q8_0, GgmlTensorType.Q5_0, GgmlTensorType.Q4_0 })
            foreach (int rows in new[] { 1, 8, 70 })
                yield return new object[] { type, rows, 12288, 5 };
        // Q0 four-block groups with 0 and 3 leftover blocks (K/32 = 4, 7, 128).
        foreach (var type in new[] { GgmlTensorType.Q4_0, GgmlTensorType.Q5_0, GgmlTensorType.Q8_0 })
            foreach (int k in new[] { 128, 224, 4096 })
                foreach (int rows in new[] { 1, 4, 9 })
                    yield return new object[] { type, rows, k, 7 };
        // single-column and two-column outputs (pair tail / no tail)
        foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q6_K, GgmlTensorType.Q8_0, GgmlTensorType.Q5_0 })
            foreach (int n in new[] { 1, 2 })
                yield return new object[] { type, 9, type == GgmlTensorType.Q8_0 || type == GgmlTensorType.Q5_0 ? 704 : 512, n };
    }

    [Theory]
    [MemberData(nameof(GemmCases))]
    public unsafe void QGemm_MatchesPerRowPathAndDequantReference(GgmlTensorType type, int rows, int k, int n)
    {
        var rng = new Random(20260927 + (int)type * 131 + rows * 7 + k);
        byte[] weights = BuildRandomWeights(rng, type, n, k);
        int inStride = k + 5, outStride = n + 3;
        float[] input = BuildInput(rng, rows, k, inStride);

        float[] legacy = RunAddmm(type, weights, k, n, input, inStride, rows, outStride, QGemmIsa.Legacy);
        AssertWithinActivationQuantBound(type, weights, k, n, input, inStride, rows, outStride, legacy);

        float legacyScale = MaxAbs(legacy) + 1e-6f;
        foreach (var isa in AvailableGemmIsas())
        {
            float[] actual = RunAddmm(type, weights, k, n, input, inStride, rows, outStride, isa);
            float err = MaxAbsDiff(legacy, actual) / legacyScale;
            _output.WriteLine($"{type} rows={rows} K={k} N={n} {isa}: max |gemm - legacy| / max|legacy| = {err:E2}");
            Assert.True(err <= LegacyRelTol, $"{type} rows={rows} K={k} N={n} {isa}: relative error {err:E2}");
            AssertPaddingUntouched(actual, rows, n, outStride);
        }

        // The public entry (Auto) must agree too, whichever path it picks.
        float[] auto = RunAddmm(type, weights, k, n, input, inStride, rows, outStride, QGemmIsa.Auto);
        Assert.True(MaxAbsDiff(legacy, auto) / legacyScale <= LegacyRelTol);
    }

    /// <summary>Every output is computed by one kernel call whose summation
    /// order does not depend on the tile it lands in, so the task partitioning
    /// (single-threaded vs the pool) must not change a single bit.</summary>
    [Theory]
    [InlineData(GgmlTensorType.Q4_K, 70, 2816, 301)]
    [InlineData(GgmlTensorType.Q6_K, 257, 2816, 64)]
    [InlineData(GgmlTensorType.Q8_0, 70, 2112, 301)]
    [InlineData(GgmlTensorType.Q5_0, 33, 704, 97)]
    public unsafe void QGemm_ResultIsIndependentOfPartitioning(GgmlTensorType type, int rows, int k, int n)
    {
        var rng = new Random(77 + (int)type);
        byte[] weights = BuildRandomWeights(rng, type, n, k);
        float[] input = BuildInput(rng, rows, k, k);
        foreach (var isa in AvailableGemmIsas())
        {
            float[] pooled = RunAddmm(type, weights, k, n, input, k, rows, n, isa);
            float[] serial = RunAddmm(type, weights, k, n, input, k, rows, n, isa,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 1 });
            Assert.Equal(serial, pooled);
        }
    }

    [Theory]
    [InlineData(GgmlTensorType.Q4_K, 2816)]
    [InlineData(GgmlTensorType.Q5_K, 512)]
    [InlineData(GgmlTensorType.Q6_K, 2816)]
    [InlineData(GgmlTensorType.Q8_0, 704)]
    [InlineData(GgmlTensorType.Q5_0, 2112)]
    [InlineData(GgmlTensorType.Q4_0, 704)]
    public unsafe void QGemmBatch_MatchesPerJobPerRowPath(GgmlTensorType type, int k)
    {
        var rng = new Random(4242 + (int)type + k);
        int[] jobRows = { 1, 3, 8, 17, 3, 2, 0, 5 };
        int[] jobOut = { 64, 37, 64, 5, 37, 64, 16, 1 };
        int jobs = jobRows.Length;
        int inStride = k + 3;
        var weights = new byte[jobs][];
        var inputs = new float[jobs][];
        for (int j = 0; j < jobs; j++)
        {
            weights[j] = BuildRandomWeights(rng, type, jobOut[j], k);
            inputs[j] = BuildInput(rng, Math.Max(1, jobRows[j]), k, inStride);
        }
        // jobs 1 and 4 share one input block (the gate/up pattern)
        inputs[4] = inputs[1];

        var expected = new float[jobs][];
        for (int j = 0; j < jobs; j++)
            expected[j] = jobRows[j] == 0
                ? new float[jobOut[j] + 2]
                : RunAddmm(type, weights[j], k, jobOut[j], inputs[j], inStride, jobRows[j], jobOut[j] + 2, QGemmIsa.Legacy);

        foreach (var isa in AvailableGemmIsas().Append(QGemmIsa.Auto))
        {
            var handles = new List<GCHandle>();
            var outputs = new float[jobs][];
            try
            {
                var batch = new ManagedQuantizedOps.QuantMatMulJob[jobs];
                for (int j = 0; j < jobs; j++)
                {
                    outputs[j] = new float[Math.Max(1, jobRows[j]) * (jobOut[j] + 2)];
                    Array.Fill(outputs[j], float.NaN);
                    var hw = GCHandle.Alloc(weights[j], GCHandleType.Pinned);
                    var hi = GCHandle.Alloc(inputs[j], GCHandleType.Pinned);
                    var ho = GCHandle.Alloc(outputs[j], GCHandleType.Pinned);
                    handles.Add(hw); handles.Add(hi); handles.Add(ho);
                    batch[j] = new ManagedQuantizedOps.QuantMatMulJob(hw.AddrOfPinnedObject(), hi.AddrOfPinnedObject(),
                        ho.AddrOfPinnedObject(), jobOut[j], jobRows[j], jobOut[j] + 2);
                }
                Assert.True(ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, inStride, batch, null, isa));
            }
            finally
            {
                foreach (var h in handles) h.Free();
            }

            for (int j = 0; j < jobs; j++)
            {
                if (jobRows[j] == 0) continue;
                float scale = MaxAbs(expected[j]) + 1e-6f;
                float err = MaxAbsDiff(expected[j], outputs[j]) / scale;
                _output.WriteLine($"{type} K={k} job {j} rows={jobRows[j]} {isa}: rel err {err:E2}");
                Assert.True(err <= LegacyRelTol, $"{type} job {j} {isa}: relative error {err:E2}");
            }
        }
    }

    /// <summary>The GEMM activation layout must carry exactly the Q8_K / Q8_0
    /// values, scales and block sums the per-row path produces.</summary>
    [Theory]
    [InlineData(GgmlTensorType.Q4_K, 1024)]
    [InlineData(GgmlTensorType.Q6_K, 1024)]
    [InlineData(GgmlTensorType.Q8_0, 704)]
    [InlineData(GgmlTensorType.Q5_0, 704)]
    [InlineData(GgmlTensorType.Q4_0, 96)]
    public unsafe void QGemmActivationLayout_IsBitIdenticalToPerRowQuantization(GgmlTensorType type, int k)
    {
        var rng = new Random(99 + (int)type);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() - 0.5) * (i % 97 == 0 ? 40f : 2f);
        // an all-zero block (scale 0 path) and exact .5 ties (round half to even)
        for (int i = 32; i < 64; i++) x[i] = 0f;
        for (int i = 256; i < 288 && i < k; i++) x[i] = (i % 2 == 0 ? 2.5f : -3.5f);
        if (k > 288) x[288] = 127f;

        bool kFamily = type is GgmlTensorType.Q4_K or GgmlTensorType.Q6_K;
        Assert.True(ManagedQuantizedOps.TryGetActivationPlan(type, k, out int stdBytes));
        byte[] std = new byte[stdBytes];
        byte[] gemm = new byte[ManagedQuantizedOps.QGemmActivationRowBytes(type, k)];
        fixed (float* xp = x)
        fixed (byte* sp = std)
        fixed (byte* gp = gemm)
        {
            ManagedQuantizedOps.QuantizeActivationRow(type, xp, sp, k);
            ManagedQuantizedOps.QGemmQuantizeActivationRow(type, xp, gp, k);
        }

        if (kFamily)
        {
            int nsb = k / 256;
            for (int sb = 0; sb < nsb; sb++)
            {
                int s = sb * 292;
                float d = BitConverter.ToSingle(std, s);
                Assert.Equal(d, BitConverter.ToSingle(gemm, k + nsb * 32 + sb * 4));
                for (int i = 0; i < 256; i++)
                    Assert.Equal(std[s + 4 + i], gemm[sb * 256 + i]);
                for (int g = 0; g < 16; g++)
                {
                    short bs = BinaryPrimitives.ReadInt16LittleEndian(std.AsSpan(s + 260 + g * 2));
                    if (type == GgmlTensorType.Q6_K)
                        Assert.Equal(bs, BinaryPrimitives.ReadInt16LittleEndian(gemm.AsSpan(k + sb * 32 + g * 2)));
                    else if (g % 2 == 0)
                    {
                        short bs1 = BinaryPrimitives.ReadInt16LittleEndian(std.AsSpan(s + 260 + g * 2 + 2));
                        Assert.Equal(d * (bs + bs1), BitConverter.ToSingle(gemm, k + sb * 32 + g / 2 * 4));
                    }
                }
            }
        }
        else
        {
            int nb = k / 32;
            int share = type == GgmlTensorType.Q5_0 ? 8 : 4;   // zeroPoint / 2
            for (int b = 0; b < nb; b++)
            {
                int s = b * 34;
                float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(std.AsSpan(s)));
                Assert.Equal(d, BitConverter.ToSingle(gemm, k + b * 4));
                int sum = 0;
                for (int i = 0; i < 32; i++)
                {
                    Assert.Equal(std[s + 2 + i], gemm[b * 32 + i]);
                    sum += (sbyte)std[s + 2 + i];
                }
                if (type != GgmlTensorType.Q8_0)
                    Assert.Equal(share * sum, BitConverter.ToInt32(gemm, k + nb * 4 + b * 4));
            }
        }
    }

    [Fact]
    public unsafe void Int8Quantizer_SimdVariantsAreBitIdenticalToScalar()
    {
        var rng = new Random(5);
        const int n = 4096;
        float[] x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = i % 5 == 0 ? (i % 2 == 0 ? 0.5f : -0.5f) * (i % 7) : (float)(rng.NextDouble() - 0.5) * 300f;
        // Non-finite and out-of-int-range products: the scalar (int) cast
        // saturates (NaN -> 0, +Inf -> +127), raw vcvtps2dq would give -127.
        float[] special = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1e30f, -1e30f, 3e9f, -3e9f,
                            0f, -0f, 1e-45f, -1e-45f, 1e-38f, 127.5f, -127.5f, 126.5f, 128f };
        for (int i = 0; i < special.Length; i++)
            for (int rep = 0; rep < 8; rep++)
                x[1024 + rep * 97 + i] = special[i];
        // 1/scale overflows to +Inf when a block's max|x| is below ~3.7e-37,
        // which makes 0 * inv a NaN as well.
        foreach (float inv in new[] { 1f, 0.25f, 1f / 3f, 0.4231f, float.PositiveInfinity, float.MaxValue })
        {
            sbyte[] qs = new sbyte[n], q2 = new sbyte[n], q5 = new sbyte[n];
            int[] ss = new int[n / 16], s2 = new int[n / 16], s5 = new int[n / 16];
            fixed (float* xp = x)
            fixed (sbyte* a = qs) fixed (sbyte* b = q2) fixed (sbyte* c = q5)
            fixed (int* sa = ss) fixed (int* sb = s2) fixed (int* sc = s5)
            {
                ManagedQuantizedOps.QuantizeInt8Groups16Scalar(xp, inv, a, n, sa);
                if (Avx2.IsSupported)
                {
                    ManagedQuantizedOps.QuantizeInt8Groups16Avx2(xp, inv, b, n, sb);
                    Assert.Equal(qs, q2);
                    Assert.Equal(ss, s2);
                }
                if (Avx512F.IsSupported && Avx512BW.IsSupported)
                {
                    ManagedQuantizedOps.QuantizeInt8Groups16Avx512(xp, inv, c, n, sc);
                    Assert.Equal(qs, q5);
                    Assert.Equal(ss, s5);
                }
            }
            // the scalar reference itself: NaN -> 0, +Inf -> 127, -Inf -> -127
            Assert.Equal(0, qs[1024]);
            Assert.Equal(127, qs[1025]);
            Assert.Equal(-127, qs[1026]);
        }
    }

    /// <summary>A degenerate block (max|x| ~ 1e-39, so 1/scale is +Inf) goes
    /// through the per-row and GEMM quantizers identically.</summary>
    [Theory]
    [InlineData(GgmlTensorType.Q4_K, 512)]
    [InlineData(GgmlTensorType.Q6_K, 512)]
    [InlineData(GgmlTensorType.Q8_0, 64)]
    [InlineData(GgmlTensorType.Q4_0, 64)]
    public unsafe void QGemmActivationLayout_TinyScaleBlockMatchesPerRowQuantization(GgmlTensorType type, int k)
    {
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = i % 3 == 0 ? 0f : (i % 2 == 0 ? 1e-39f : -2e-39f);
        Assert.True(ManagedQuantizedOps.TryGetActivationPlan(type, k, out int stdBytes));
        byte[] std = new byte[stdBytes];
        byte[] gemm = new byte[ManagedQuantizedOps.QGemmActivationRowBytes(type, k)];
        bool kFamily = type is GgmlTensorType.Q4_K or GgmlTensorType.Q6_K;
        int block = kFamily ? 256 : 32, header = kFamily ? 4 : 2, blockBytes = kFamily ? 292 : 34;
        foreach (bool avx512 in new[] { true, false })
        {
            fixed (float* xp = x)
            fixed (byte* sp = std)
            fixed (byte* gp = gemm)
            {
                ManagedQuantizedOps.QuantizeActivationRow(type, xp, sp, k);
                ManagedQuantizedOps.QGemmQuantizeActivationRow(type, xp, gp, k, avx512);
            }
            for (int b = 0; b < k / block; b++)
                for (int i = 0; i < block; i++)
                {
                    sbyte expected = (sbyte)std[b * blockBytes + header + i];
                    Assert.Equal(expected, (sbyte)gemm[b * block + i]);
                    if (x[b * block + i] == 0f) Assert.Equal(0, expected);   // 0 * Inf is NaN -> 0
                    else Assert.Equal(x[b * block + i] > 0 ? 127 : -127, expected);
                }
        }
    }

    public static IEnumerable<object[]> InvarianceCases()
    {
        foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q5_K, GgmlTensorType.Q6_K })
            yield return new object[] { type, 2816 };
        foreach (var type in new[] { GgmlTensorType.Q4_0, GgmlTensorType.Q5_0, GgmlTensorType.Q8_0 })
            yield return new object[] { type, 704 };
        foreach (var type in new[] { GgmlTensorType.BF16, GgmlTensorType.F16, GgmlTensorType.F32 })
            yield return new object[] { type, 520 };
    }

    /// <summary>
    /// With the default routing a row's output must not depend on how many rows
    /// share the call - decode (M = 1) vs a speculative verify batch vs
    /// continuous batching, or a dense call vs a one-row MoE batch job. The
    /// GEMM sums every output the same way at any tile height, and single rows
    /// take it for every type, so this is bit-exact.
    /// </summary>
    [QGemmDefaultRoutingTheory]
    [MemberData(nameof(InvarianceCases))]
    public unsafe void AutoRouting_RowResultIsIndependentOfBatchSize(GgmlTensorType type, int k)
    {
        const int rows = 11, n = 37;
        var rng = new Random(606 + (int)type);
        byte[] weights = BuildRandomWeights(rng, type, n, k);
        float[] input = BuildInput(rng, rows, k, k);
        float[] all = RunAddmm(type, weights, k, n, input, k, rows, n, QGemmIsa.Auto);

        for (int r = 0; r < rows; r++)
        {
            float[] one = RunAddmm(type, weights, k, n, input.AsSpan(r * k, k).ToArray(), k, 1, n, QGemmIsa.Auto);
            Assert.Equal(all.AsSpan(r * n, n).ToArray(), one);
        }
        float[] middle = RunAddmm(type, weights, k, n, input.AsSpan(4 * k, 5 * k).ToArray(), k, 5, n, QGemmIsa.Auto);
        Assert.Equal(all.AsSpan(4 * n, 5 * n).ToArray(), middle);

        if (!ManagedQuantizedOps.TryGetActivationPlan(type, k, out _))
            return;   // float types have no MoE batch entry
        // Every row as its own one-row batch job (top-k routing of single tokens).
        float[] batched = new float[rows * n];
        fixed (byte* w = weights)
        fixed (float* x = input)
        fixed (float* o = batched)
        {
            var jobs = new ManagedQuantizedOps.QuantMatMulJob[rows];
            for (int r = 0; r < rows; r++)
                jobs[r] = new ManagedQuantizedOps.QuantMatMulJob((IntPtr)w, (IntPtr)(x + r * k), (IntPtr)(o + r * n), n, 1, n);
            Assert.True(ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, jobs, null, QGemmIsa.Auto));
        }
        Assert.Equal(all, batched);
    }

    /// <summary>
    /// TS_CPU_QGEMM_MIN_ROWS sends small calls to the per-row path; the MoE batch
    /// applies the same threshold per job, so each job still matches the dense
    /// call with its row count bit for bit.
    /// </summary>
    [Theory]
    [InlineData(GgmlTensorType.Q4_K, 512)]
    [InlineData(GgmlTensorType.Q8_0, 704)]
    [InlineData(GgmlTensorType.Q4_0, 224)]
    public unsafe void QGemmBatch_MinRowsSplitMatchesDenseRoutingPerJob(GgmlTensorType type, int k)
    {
        const int minRows = 4;
        var rng = new Random(313 + (int)type);
        int[] jobRows = { 1, 3, 4, 9, 0, 2, 5 };
        int jobs = jobRows.Length, n = 19;
        var weights = new byte[jobs][];
        var inputs = new float[jobs][];
        for (int j = 0; j < jobs; j++)
        {
            weights[j] = BuildRandomWeights(rng, type, n, k);
            inputs[j] = BuildInput(rng, Math.Max(1, jobRows[j]), k, k);
        }
        QGemmIsa gemmIsa = ManagedQuantizedOps.ResolveQGemmIsa(QGemmIsa.Auto);
        var outputs = new float[jobs][];
        var handles = new List<GCHandle>();
        try
        {
            var batch = new ManagedQuantizedOps.QuantMatMulJob[jobs];
            for (int j = 0; j < jobs; j++)
            {
                outputs[j] = new float[Math.Max(1, jobRows[j]) * n];
                var hw = GCHandle.Alloc(weights[j], GCHandleType.Pinned);
                var hi = GCHandle.Alloc(inputs[j], GCHandleType.Pinned);
                var ho = GCHandle.Alloc(outputs[j], GCHandleType.Pinned);
                handles.Add(hw); handles.Add(hi); handles.Add(ho);
                batch[j] = new ManagedQuantizedOps.QuantMatMulJob(hw.AddrOfPinnedObject(), hi.AddrOfPinnedObject(),
                    ho.AddrOfPinnedObject(), n, jobRows[j], n);
            }
            Assert.True(ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, batch, null, QGemmIsa.Auto, minRows));
        }
        finally
        {
            foreach (var h in handles) h.Free();
        }

        for (int j = 0; j < jobs; j++)
        {
            if (jobRows[j] == 0) continue;
            QGemmIsa isa = jobRows[j] < minRows ? QGemmIsa.Legacy : gemmIsa;
            float[] expected = RunAddmm(type, weights[j], k, n, inputs[j], k, jobRows[j], n, isa);
            Assert.Equal(expected, outputs[j]);
        }
    }

    [Fact]
    public void ResolveQGemmIsa_HonoursSwitchesAndHostIsa()
    {
        static QGemmIsa R(QGemmIsa req, bool enabled, bool noAvx512, bool avx2, bool avx512)
            => ManagedQuantizedOps.ResolveQGemmIsa(req, enabled, noAvx512, avx2, avx512);

        Assert.Equal(QGemmIsa.Avx512, R(QGemmIsa.Auto, true, false, true, true));
        Assert.Equal(QGemmIsa.Avx2, R(QGemmIsa.Auto, true, true, true, true));      // TS_CPU_DISABLE_AVX512=1
        Assert.Equal(QGemmIsa.Legacy, R(QGemmIsa.Auto, false, false, true, true));  // TS_CPU_QGEMM=0
        Assert.Equal(QGemmIsa.Avx2, R(QGemmIsa.Auto, true, false, true, false));    // AVX2-only host
        Assert.Equal(QGemmIsa.Legacy, R(QGemmIsa.Auto, true, false, false, false)); // ARM64 / no AVX2
        // explicit requests (tests, benchmarks) ignore the switches, not the host
        Assert.Equal(QGemmIsa.Avx512, R(QGemmIsa.Avx512, false, true, true, true));
        Assert.Equal(QGemmIsa.Avx2, R(QGemmIsa.Avx512, true, false, true, false));
        Assert.Equal(QGemmIsa.Legacy, R(QGemmIsa.Avx2, true, false, false, false));
        Assert.Equal(QGemmIsa.Legacy, R(QGemmIsa.Legacy, true, false, true, true));
    }

    // .NET 10.0.8 encodes Avx512F.BroadcastVector256ToVector512(long*/double*)
    // with a constant displacement as base + 2 * disp. The kernels' broadcast
    // helpers use the AVX512DQ forms instead; these probes fold constant
    // displacements into the helpers' memory operands, as the Q0 kernels do.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector512<sbyte> Bcast32(byte* p) => ManagedQuantizedOps.Bcast256x2(p + 32);
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector512<sbyte> Bcast64(byte* p) => ManagedQuantizedOps.Bcast256x2(p + 64);
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector512<sbyte> Bcast96(byte* p) => ManagedQuantizedOps.Bcast256x2(p + 96);
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector512<float> BcastF32(byte* p) => ManagedQuantizedOps.Bcast256x2F(p + 32);
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector512<float> BcastF96(byte* p) => ManagedQuantizedOps.Bcast256x2F(p + 96);

    [Avx512Fact]
    public unsafe void BroadcastHelpers_LoadTheAddressedBytesAtConstantDisplacements()
    {
        byte* buf = (byte*)NativeMemory.AlignedAlloc(512, 64);
        try
        {
            for (int i = 0; i < 512; i++) buf[i] = (byte)i;
            foreach (var (v, at) in new[] { (Bcast32(buf), 32), (Bcast64(buf), 64), (Bcast96(buf), 96) })
                for (int i = 0; i < 64; i++)
                    Assert.Equal((byte)(at + i % 32), (byte)v.GetElement(i));
            foreach (var (v, at) in new[] { (BcastF32(buf), 32), (BcastF96(buf), 96) })
                for (int i = 0; i < 16; i++)
                    Assert.Equal(BitConverter.ToSingle(new[] { (byte)(at + i % 8 * 4), (byte)(at + i % 8 * 4 + 1),
                        (byte)(at + i % 8 * 4 + 2), (byte)(at + i % 8 * 4 + 3) }), v.GetElement(i));
        }
        finally
        {
            NativeMemory.AlignedFree(buf);
        }
    }

    /// <summary>The float panel reads F32 weights in place (no dequant copy);
    /// check it against a float64 reference.</summary>
    [Fact]
    public void FloatPanel_UsesInPlaceF32Weights()
    {
        var rng = new Random(8);
        const int k = 96, n = 9, rows = 6;
        byte[] weights = BuildRandomWeights(rng, GgmlTensorType.F32, n, k);
        float[] input = BuildInput(rng, rows, k, k);
        float[] expected = new float[rows * n];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < n; c++)
            {
                double s = 0;
                for (int i = 0; i < k; i++) s += (double)BitConverter.ToSingle(weights, (c * k + i) * 4) * input[r * k + i];
                expected[r * n + c] = (float)s;
            }
        foreach (var isa in AvailableGemmIsas().Append(QGemmIsa.Auto))
        {
            float[] actual = RunAddmm(GgmlTensorType.F32, weights, k, n, input, k, rows, n, isa);
            Assert.True(MaxAbsDiff(expected, actual) <= 1e-5f * (MaxAbs(expected) + 1e-6f), $"F32 {isa}");
        }
    }

    public static IEnumerable<object[]> FloatPanelCases()
    {
        foreach (var type in new[] { GgmlTensorType.BF16, GgmlTensorType.F16, GgmlTensorType.F32 })
            foreach (int k in new[] { 64, 100, 4096 })
                foreach (int rows in new[] { 1, 2, 3, 4, 5, 9, 17 })
                    yield return new object[] { type, rows, k, 23 };
        foreach (int rows in new[] { 1, 3, 4, 17 })
        {
            yield return new object[] { GgmlTensorType.Q3_K, rows, 512, 19 };
            yield return new object[] { GgmlTensorType.IQ4_XS, rows, 512, 19 };
        }
        yield return new object[] { GgmlTensorType.BF16, 70, 256, 1 };
        yield return new object[] { GgmlTensorType.BF16, 70, 256, 6 };
        // K = 12288: F16/BF16 rebuild the panel per L2-sized row block,
        // the dequant-only types build it once for all rows.
        foreach (var type in new[] { GgmlTensorType.BF16, GgmlTensorType.F16, GgmlTensorType.Q3_K, GgmlTensorType.IQ4_XS })
            yield return new object[] { type, 70, 12288, 5 };
    }

    [Theory]
    [MemberData(nameof(FloatPanelCases))]
    public unsafe void FloatPanelGemm_MatchesDequantColumnPath(GgmlTensorType type, int rows, int k, int n)
    {
        var rng = new Random(31 + (int)type * 3 + rows + k);
        byte[] weights = BuildRandomWeights(rng, type, n, k);
        int inStride = k + 7, outStride = n + 2;
        float[] input = BuildInput(rng, rows, k, inStride);

        float[] legacy = RunAddmm(type, weights, k, n, input, inStride, rows, outStride, QGemmIsa.Legacy);
        float scale = MaxAbs(legacy) + 1e-6f;
        foreach (var isa in AvailableGemmIsas().Append(QGemmIsa.Auto))
        {
            float[] actual = RunAddmm(type, weights, k, n, input, inStride, rows, outStride, isa);
            float err = MaxAbsDiff(legacy, actual) / scale;
            _output.WriteLine($"{type} rows={rows} K={k} N={n} {isa}: rel err {err:E2}");
            Assert.True(err <= LegacyRelTol, $"{type} rows={rows} K={k} {isa}: relative error {err:E2}");
            AssertPaddingUntouched(actual, rows, n, outStride);
        }
    }

    /// <summary>The AVX2 Q5_0 x Q8_0 dot (formerly scalar-only) against an
    /// exact integer reference built from the dequantized block values.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(22)]
    [InlineData(128)]
    public unsafe void Q50Dot_MatchesIntegerReference(int blocks)
    {
        var rng = new Random(blocks);
        int k = blocks * 32;
        byte[] w = BuildRandomWeights(rng, GgmlTensorType.Q5_0, 1, k);
        float[] x = BuildInput(rng, 1, k, k);
        Assert.True(ManagedQuantizedOps.TryGetActivationPlan(GgmlTensorType.Q5_0, k, out int actBytes));
        byte[] act = new byte[actBytes];
        float[] wf = new float[k];
        ManagedQuantizedOps.DequantizeToFloat32((int)GgmlTensorType.Q5_0, w, 0, wf, 0, k);
        double expected = 0;
        fixed (float* xp = x) fixed (byte* ap = act) fixed (byte* wp = w)
        {
            ManagedQuantizedOps.QuantizeActivationRow(GgmlTensorType.Q5_0, xp, ap, k);
            for (int b = 0; b < blocks; b++)
            {
                float dw = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(w.AsSpan(b * 22)));
                float dx = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(act.AsSpan(b * 34)));
                long isum = 0;
                for (int i = 0; i < 32; i++)
                    isum += (long)MathF.Round(wf[b * 32 + i] / dw) * (sbyte)act[b * 34 + 2 + i];
                expected += (double)dw * dx * isum;
            }
            float actual = ManagedQuantizedOps.DotQuantizedRow(GgmlTensorType.Q5_0, wp, ap, k);
            Assert.InRange(Math.Abs(actual - expected), 0, 1e-5 * Math.Max(1, Math.Abs(expected)) + 1e-5);
        }
    }

    // ---- helpers ----------------------------------------------------------

    private static IEnumerable<QGemmIsa> AvailableGemmIsas()
    {
        var list = new List<QGemmIsa>();
        if (ManagedQuantizedOps.QGemmAvx512Supported) list.Add(QGemmIsa.Avx512);
        if (ManagedQuantizedOps.QGemmAvx2Supported) list.Add(QGemmIsa.Avx2);
        return list;
    }

    private static unsafe float[] RunAddmm(GgmlTensorType type, byte[] weights, int k, int n, float[] input, int inStride,
        int rows, int outStride, QGemmIsa isa, System.Threading.Tasks.ParallelOptions options = null)
    {
        float[] output = new float[rows * outStride];
        Array.Fill(output, float.NaN);
        fixed (byte* w = weights)
        fixed (float* x = input)
        fixed (float* o = output)
        {
            ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, x, inStride, rows, o, outStride, options, isa);
        }
        return output;
    }

    /// <summary>
    /// The per-row path against a float64 dequantize-then-dot reference, with
    /// the rigorous bound of the Q8 activation quantization: every activation
    /// element moves by at most half its block's step (max|block| / 127 / 2),
    /// so |out - ref| &lt;= sum_i |w_i| * step_i / 2, plus float rounding.
    /// </summary>
    private static void AssertWithinActivationQuantBound(GgmlTensorType type, byte[] weights, int k, int n, float[] input,
        int inStride, int rows, int outStride, float[] actual)
    {
        int block = type is GgmlTensorType.Q4_K or GgmlTensorType.Q5_K or GgmlTensorType.Q6_K ? 256 : 32;
        long rowBytes = ManagedQuantizedOps.RowSize((int)type, k);
        float[] wrow = new float[k];
        double[] halfStep = new double[k];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < k; b += block)
            {
                double m = 0;
                for (int i = b; i < b + block; i++) m = Math.Max(m, Math.Abs(input[r * inStride + i]));
                for (int i = b; i < b + block; i++) halfStep[i] = m / 127.0 / 2.0;
            }
            for (int c = 0; c < n; c++)
            {
                ManagedQuantizedOps.DequantizeToFloat32((int)type, weights, (int)(c * rowBytes), wrow, 0, k);
                double s = 0, bound = 0, mag = 0;
                for (int i = 0; i < k; i++)
                {
                    double p = (double)wrow[i] * input[r * inStride + i];
                    s += p;
                    mag += Math.Abs(p);
                    bound += Math.Abs(wrow[i]) * halfStep[i];
                }
                double err = Math.Abs(actual[r * outStride + c] - s);
                Assert.True(err <= bound * 1.01 + 1e-5 * mag + 1e-6,
                    $"{type} row {r} col {c}: |legacy - reference| = {err} exceeds the activation quantization bound {bound}");
            }
        }
    }

    private static float[] BuildInput(Random rng, int rows, int k, int stride)
    {
        float[] x = new float[rows * stride];
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < stride; i++)
            {
                if (i >= k) { x[r * stride + i] = 1e30f; continue; }   // stride padding must never be read
                float v = (float)(rng.NextDouble() - 0.5) * 0.4f + 0.05f * MathF.Sin(i * 0.013f + r);
                if (i % 211 == 3) v *= 25f;                             // outliers
                if (r % 3 == 1 && i < 256 && k >= 512) v = 0f;          // an all-zero leading block
                x[r * stride + i] = v;
            }
        return x;
    }

    private static byte[] BuildRandomWeights(Random rng, GgmlTensorType type, int outDim, int inDim)
    {
        int blockBytes = (int)GgufFile.GetTypeSize(type);
        int blockSize = (int)GgufFile.GetBlockSize(type);
        int blocksPerRow = inDim / blockSize;
        byte[] raw = new byte[(long)outDim * blocksPerRow * blockBytes];
        rng.NextBytes(raw);
        for (long o = 0; o < raw.Length; o += blockBytes)
        {
            switch (type)
            {
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q5_0:
                case GgmlTensorType.Q8_0:
                    WriteHalf(raw, o, 0.01f + 0.03f * (float)rng.NextDouble());
                    break;
                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                    WriteHalf(raw, o, 0.01f + 0.03f * (float)rng.NextDouble());
                    WriteHalf(raw, o + 2, 0.005f + 0.02f * (float)rng.NextDouble());
                    break;
                case GgmlTensorType.Q6_K:
                    WriteHalf(raw, o + blockBytes - 2, 0.005f + 0.02f * (float)rng.NextDouble());
                    break;
                case GgmlTensorType.Q3_K:
                    WriteHalf(raw, o + blockBytes - 2, 0.005f + 0.02f * (float)rng.NextDouble());
                    break;
                case GgmlTensorType.IQ4_XS:
                    WriteHalf(raw, o, 0.005f + 0.02f * (float)rng.NextDouble());
                    break;
                case GgmlTensorType.F16:
                    WriteHalf(raw, o, (float)(rng.NextDouble() - 0.5) * 0.1f);
                    break;
                case GgmlTensorType.BF16:
                {
                    uint bits = BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() - 0.5) * 0.1f);
                    raw[o] = (byte)(bits >> 16);
                    raw[o + 1] = (byte)(bits >> 24);
                    break;
                }
                case GgmlTensorType.F32:
                    BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan((int)o), (float)(rng.NextDouble() - 0.5) * 0.1f);
                    break;
                default:
                    throw new NotSupportedException(type.ToString());
            }
        }
        return raw;
    }

    private static void WriteHalf(byte[] buffer, long offset, float value)
        => BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan((int)offset), BitConverter.HalfToUInt16Bits((Half)value));

    private static float MaxAbs(float[] a)
    {
        float m = 0f;
        foreach (float v in a)
            if (!float.IsNaN(v)) m = MathF.Max(m, MathF.Abs(v));
        return m;
    }

    /// <summary>Max |a - b| over the written outputs (NaN sentinels in both
    /// arrays mark stride padding and are skipped only when both are NaN).</summary>
    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            if (float.IsNaN(a[i]) && float.IsNaN(b[i])) continue;
            float d = MathF.Abs(a[i] - b[i]);
            if (float.IsNaN(d)) return float.PositiveInfinity;
            m = MathF.Max(m, d);
        }
        return m;
    }

    private static void AssertPaddingUntouched(float[] output, int rows, int n, int outStride)
    {
        for (int r = 0; r < rows; r++)
            for (int c = n; c < outStride; c++)
                Assert.True(float.IsNaN(output[r * outStride + c]), $"output padding row {r} col {c} was written");
    }
}

/// <summary>Theory that runs only under the default GEMM routing: batch-size
/// invariance is what that routing promises, and TS_CPU_QGEMM=0,
/// TS_CPU_FGEMM=0, TS_CPU_QGEMM_MIN_ROWS or a host without AVX2 give it up.</summary>
public sealed class QGemmDefaultRoutingTheoryAttribute : TheoryAttribute
{
    public QGemmDefaultRoutingTheoryAttribute()
    {
        if (!ManagedQuantizedOps.QGemmRoutingIsBatchInvariant)
            Skip = "The GEMM routing is not the default (TS_CPU_QGEMM / TS_CPU_FGEMM / TS_CPU_QGEMM_MIN_ROWS, or no AVX2).";
    }
}

/// <summary>Fact that needs the AVX-512 GEMM kernels' instruction sets.</summary>
public sealed class Avx512FactAttribute : FactAttribute
{
    public Avx512FactAttribute()
    {
        if (!ManagedQuantizedOps.QGemmAvx512Supported)
            Skip = "Requires AVX-512 F/BW/DQ.";
    }
}
