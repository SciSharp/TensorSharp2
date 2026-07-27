using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>
/// Op-level coverage for the pieces the direct-CUDA vision-encoder path depends
/// on. Each op is checked against an independent host reference, because the
/// defects these guard against were all silent: the encoder produced plausible
/// but wrong embeddings rather than throwing.
/// </summary>
public class VisionEncoderCudaTests
{
    // Ops.Add(x[rows, cols], bias[cols]) is how every linear layer in the vision
    // encoders applies its bias. The elementwise CUDA kernel requires identical
    // shapes, so this used to fall through to the CUDA CPU fallback, whose
    // Apply3 advances all three iterators together and stops at the shortest
    // operand's last block — biasing row 0 and leaving every other row untouched.
    [Theory]
    [InlineData(1, 8)]
    [InlineData(7, 3)]
    [InlineData(64, 1152)]
    [InlineData(257, 4304)]
    public void CudaAdd_RowBroadcastBias_AppliesToEveryRow(int rows, int cols)
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        float[] baseValues = MakePattern(rows * cols, 0.7f, 0.1f);
        float[] biasValues = MakePattern(cols, 1.3f, -0.4f);

        using var tensor = new Tensor(allocator, DType.Float32, rows, cols);
        tensor.SetElementsAsFloat(baseValues);
        using var bias = new Tensor(allocator, DType.Float32, cols);
        bias.SetElementsAsFloat(biasValues);

        Ops.Add(tensor, tensor, bias);

        float[] expected = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                expected[r * cols + c] = baseValues[r * cols + c] + biasValues[c];

        AssertClose(expected, tensor.GetElementsAsFloat(rows * cols));
    }

    // Same defect on the CPU allocator: TensorApplyCPU.Add reaches the same
    // Apply3 for a trailing-dimension rhs.
    [Fact]
    public void CpuAdd_RowBroadcastBias_AppliesToEveryRow()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        const int rows = 33;
        const int cols = 129;
        float[] baseValues = MakePattern(rows * cols, 0.7f, 0.1f);
        float[] biasValues = MakePattern(cols, 1.3f, -0.4f);

        using var tensor = new Tensor(allocator, DType.Float32, rows, cols);
        tensor.SetElementsAsFloat(baseValues);
        using var bias = new Tensor(allocator, DType.Float32, cols);
        bias.SetElementsAsFloat(biasValues);

        Ops.Add(tensor, tensor, bias);

        float[] expected = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                expected[r * cols + c] = baseValues[r * cols + c] + biasValues[c];

        AssertClose(expected, tensor.GetElementsAsFloat(rows * cols));
    }

    // A same-shape add must keep using the elementwise kernel; the broadcast
    // detection above must not capture it.
    [Fact]
    public void CudaAdd_SameShape_StillElementwise()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        const int rows = 5, cols = 9;
        float[] a = MakePattern(rows * cols, 0.9f, 0.2f);
        float[] b = MakePattern(rows * cols, 0.4f, -1.1f);

        using var ta = new Tensor(allocator, DType.Float32, rows, cols);
        ta.SetElementsAsFloat(a);
        using var tb = new Tensor(allocator, DType.Float32, rows, cols);
        tb.SetElementsAsFloat(b);
        using var result = new Tensor(allocator, DType.Float32, rows, cols);

        Ops.Add(result, ta, tb);

        float[] expected = new float[rows * cols];
        for (int i = 0; i < expected.Length; i++)
            expected[i] = a[i] + b[i];

        AssertClose(expected, result.GetElementsAsFloat(rows * cols));
    }

    // A [cols]-sized 2D tensor is a single row, not a broadcast source.
    [Fact]
    public void CudaAdd_SingleRowSameShape_IsNotTreatedAsBroadcast()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var a = new Tensor(allocator, DType.Float32, 1, 6);
        a.SetElementsAsFloat(new[] { 1f, 2f, 3f, 4f, 5f, 6f });
        using var b = new Tensor(allocator, DType.Float32, 1, 6);
        b.SetElementsAsFloat(new[] { 10f, 20f, 30f, 40f, 50f, 60f });
        using var result = new Tensor(allocator, DType.Float32, 1, 6);

        Ops.Add(result, a, b);

        AssertClose(new[] { 11f, 22f, 33f, 44f, 55f, 66f }, result.GetElementsAsFloat(6));
    }

    [Theory]
    [InlineData(4, 16, true)]
    [InlineData(64, 1152, true)]
    [InlineData(31, 1153, true)]
    [InlineData(8, 512, false)]
    public void CudaLayerNorm_MatchesHostReference(int rows, int cols, bool withBias)
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        const float eps = 1e-6f;
        float[] input = MakePattern(rows * cols, 2.5f, 0.3f);
        float[] alpha = MakePattern(cols, 0.6f, 1.0f);
        float[] beta = MakePattern(cols, 0.2f, -0.1f);

        using var src = new Tensor(allocator, DType.Float32, rows, cols);
        src.SetElementsAsFloat(input);
        using var alphaT = new Tensor(allocator, DType.Float32, cols);
        alphaT.SetElementsAsFloat(alpha);
        Tensor betaT = null;
        if (withBias)
        {
            betaT = new Tensor(allocator, DType.Float32, cols);
            betaT.SetElementsAsFloat(beta);
        }

        using var result = Ops.LayerNorm(null, src, alphaT, betaT, eps);
        betaT?.Dispose();

        float[] expected = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++)
                sum += input[r * cols + c];
            double mean = sum / cols;

            double sq = 0;
            for (int c = 0; c < cols; c++)
            {
                double d = input[r * cols + c] - mean;
                sq += d * d;
            }
            double sigma = Math.Sqrt(eps + sq / cols);

            for (int c = 0; c < cols; c++)
            {
                double v = (input[r * cols + c] - mean) / sigma * alpha[c];
                if (withBias)
                    v += beta[c];
                expected[r * cols + c] = (float)v;
            }
        }

        AssertClose(expected, result.GetElementsAsFloat(rows * cols), 2e-3f);
    }

    // The bidirectional flash-shaped kernel used for vision attention. Checked
    // against a direct softmax(QK^T/sqrt(d))V reference, including the
    // seq % 128 != 0 tail and a sequence long enough to span many K tiles.
    [Theory]
    [InlineData(1, 1, 72)]
    [InlineData(37, 2, 72)]
    [InlineData(128, 4, 64)]
    [InlineData(300, 3, 80)]
    [InlineData(521, 16, 72)]
    public void CudaVisionAttention_MatchesHostReference(int seq, int heads, int headDim)
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        int count = seq * heads * headDim;
        float[] q = MakePattern(count, 0.8f, 0.05f);
        float[] k = MakePattern(count, 0.6f, -0.2f);
        float[] v = MakePattern(count, 1.1f, 0.4f);
        float scale = 1f / MathF.Sqrt(headDim);

        using var qt = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        qt.SetElementsAsFloat(q);
        using var kt = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        kt.SetElementsAsFloat(k);
        using var vt = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        vt.SetElementsAsFloat(v);
        using var outT = new Tensor(allocator, DType.Float32, seq, heads * headDim);

        Assert.True(CudaFusedOps.TryVisionAttention(outT, qt, kt, vt, heads, seq, headDim, scale));

        float[] expected = ReferenceAttention(q, k, v, seq, heads, headDim, scale);
        AssertClose(expected, outT.GetElementsAsFloat(count), 2e-3f);
    }

    // The kernel and the generic SDPA path must agree; the encoder falls back to
    // SDPA for head dims without a specialized instantiation.
    [Fact]
    public void CudaVisionAttention_MatchesScaledDotProductAttention()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int seq = 200, heads = 4, headDim = 72;
        using var allocator = new CudaAllocator();
        int count = seq * heads * headDim;
        float[] q = MakePattern(count, 0.8f, 0.05f);
        float[] k = MakePattern(count, 0.6f, -0.2f);
        float[] v = MakePattern(count, 1.1f, 0.4f);
        float scale = 1f / MathF.Sqrt(headDim);

        using var qt = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        qt.SetElementsAsFloat(q);
        using var kt = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        kt.SetElementsAsFloat(k);
        using var vt = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        vt.SetElementsAsFloat(v);

        using var fast = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        Assert.True(CudaFusedOps.TryVisionAttention(fast, qt, kt, vt, heads, seq, headDim, scale));

        using var q4 = qt.View(1, seq, heads, headDim);
        using var k4 = kt.View(1, seq, heads, headDim);
        using var v4 = vt.View(1, seq, heads, headDim);
        using var generic = Ops.ScaledDotProductAttention(null, q4, k4, v4, null, scale);

        AssertClose(generic.GetElementsAsFloat(count), fast.GetElementsAsFloat(count), 2e-3f);
    }

    // Vision RoPE: NeoX rotation over [seq, heads, headDim] with per-position
    // cos/sin tables. Guards the layout the encoder relies on (tables indexed by
    // position, shared across heads; the rotated pair is (d, d + half)).
    [Theory]
    [InlineData(5, 3, 72)]
    [InlineData(256, 16, 72)]
    public void CudaNeoxRopeFlat_MatchesHostReference(int seq, int heads, int headDim)
    {
        if (!CudaBackend.IsAvailable())
            return;

        int half = headDim / 2;
        using var allocator = new CudaAllocator();
        int count = seq * heads * headDim;
        float[] data = MakePattern(count, 0.9f, 0.15f);
        float[] cos = new float[seq * half];
        float[] sin = new float[seq * half];
        for (int p = 0; p < seq; p++)
        {
            for (int j = 0; j < half; j++)
            {
                float angle = p * 0.017f + j * 0.031f;
                cos[p * half + j] = MathF.Cos(angle);
                sin[p * half + j] = MathF.Sin(angle);
            }
        }

        using var tensor = new Tensor(allocator, DType.Float32, seq, heads * headDim);
        tensor.SetElementsAsFloat(data);
        using var cosT = new Tensor(allocator, DType.Float32, seq, half);
        cosT.SetElementsAsFloat(cos);
        using var sinT = new Tensor(allocator, DType.Float32, seq, half);
        sinT.SetElementsAsFloat(sin);

        Assert.True(CudaFusedOps.TryNeoxRopeFlat(tensor, cosT, sinT, heads, seq, headDim, half));

        float[] expected = (float[])data.Clone();
        for (int p = 0; p < seq; p++)
        {
            for (int h = 0; h < heads; h++)
            {
                long baseIdx = ((long)p * heads + h) * headDim;
                for (int j = 0; j < half; j++)
                {
                    float x0 = data[baseIdx + j];
                    float x1 = data[baseIdx + j + half];
                    float c = cos[p * half + j];
                    float s = sin[p * half + j];
                    expected[baseIdx + j] = x0 * c - x1 * s;
                    expected[baseIdx + j + half] = x0 * s + x1 * c;
                }
            }
        }

        AssertClose(expected, tensor.GetElementsAsFloat(count));
    }

    private static float[] ReferenceAttention(float[] q, float[] k, float[] v,
        int seq, int heads, int headDim, float scale)
    {
        float[] output = new float[seq * heads * headDim];
        var scores = new double[seq];
        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < seq; i++)
            {
                long qBase = ((long)i * heads + h) * headDim;
                double max = double.NegativeInfinity;
                for (int j = 0; j < seq; j++)
                {
                    long kBase = ((long)j * heads + h) * headDim;
                    double dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += (double)q[qBase + d] * k[kBase + d];
                    scores[j] = dot * scale;
                    if (scores[j] > max)
                        max = scores[j];
                }

                double sum = 0;
                for (int j = 0; j < seq; j++)
                {
                    scores[j] = Math.Exp(scores[j] - max);
                    sum += scores[j];
                }

                long oBase = ((long)i * heads + h) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < seq; j++)
                        acc += scores[j] * v[((long)j * heads + h) * headDim + d];
                    output[oBase + d] = (float)(acc / sum);
                }
            }
        }

        return output;
    }

    private static float[] MakePattern(int length, float scale, float offset)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = offset + MathF.Sin(i * 0.37f) * scale + MathF.Cos(i * 0.13f) * (scale * 0.5f);
        return values;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(MathF.Abs(actual[i] - expected[i]), 0.0f, tolerance);
    }
}
