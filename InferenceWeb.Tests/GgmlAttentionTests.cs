// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public class GgmlAttentionTests
{
    [Fact]
    public void ScaledDotProductAttention_HeadDim96_NarrowedViewsMatchReference()
    {
        const int batch = 1;
        const int queryLength = 17;
        // Deliberately different from queryLength: a previous native validator
        // compared the sequence axes while claiming to validate head count,
        // which silently excluded otherwise-valid cross attention.
        const int keyLength = 19;
        const int heads = 2;
        const int headDim = 96;
        const int offset = 1;
        float scale = 1.0f / MathF.Sqrt(headDim);

        float[,,,] qBacking = BuildInput(batch, queryLength + 2, heads, headDim, 0.013f, useCos: false);
        float[,,,] kBacking = BuildInput(batch, keyLength + 2, heads, headDim, 0.017f, useCos: true);
        float[,,,] vBacking = BuildInput(batch, keyLength + 2, heads, headDim, 0.019f, useCos: false);
        float[,,,] outputSeed = new float[batch, queryLength + 2, heads, headDim];
        Fill(outputSeed, -777f);

        string configuredBackend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu")
            .Trim().ToLowerInvariant();
        GgmlBackendType backend = configuredBackend switch
        {
            "cuda" => GgmlBackendType.Cuda,
            "metal" => GgmlBackendType.Metal,
            "vulkan" => GgmlBackendType.Vulkan,
            _ => GgmlBackendType.Cpu,
        };

        var context = new GgmlContext(new[] { 0 }, backend);
        var allocator = new GgmlAllocator(context, 0);
        using var qBase = Tensor.FromArray(allocator, qBacking);
        using var kBase = Tensor.FromArray(allocator, kBacking);
        using var vBase = Tensor.FromArray(allocator, vBacking);
        using var outputBase = Tensor.FromArray(allocator, outputSeed);
        using var q = qBase.Narrow(1, offset, queryLength);
        using var k = kBase.Narrow(1, offset, keyLength);
        using var v = vBase.Narrow(1, offset, keyLength);
        using var output = outputBase.Narrow(1, offset, queryLength);

        Tensor actualTensor = Ops.ScaledDotProductAttention(output, q, k, v, null, scale);
        try
        {
            float[] expected = Reference(qBacking, kBacking, vBacking, offset, queryLength, keyLength, scale);
            float[] actual = actualTensor.GetElementsAsFloat(expected.Length);
            AssertClose(expected, actual, tolerance: 2e-3f);

            // The destination itself is a non-zero-offset view. Guard the rows on
            // both sides so an incorrect binding offset cannot pass numerically.
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < headDim; d++)
                {
                    Assert.Equal(-777f, outputBase.GetElementAsFloat(0, 0, h, d));
                    Assert.Equal(-777f, outputBase.GetElementAsFloat(0, queryLength + 1, h, d));
                }
        }
        finally
        {
            if (!ReferenceEquals(actualTensor, output))
                actualTensor.Dispose();
        }
    }

    private static float[,,,] BuildInput(int batch, int sequence, int heads, int dim, float frequency, bool useCos)
    {
        var values = new float[batch, sequence, heads, dim];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < sequence; t++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < dim; d++)
                    {
                        float angle = (b + 1) * (t + 2) * (h + 2) * (d + 1) * frequency;
                        values[b, t, h, d] = useCos ? MathF.Cos(angle) : MathF.Sin(angle);
                    }
        return values;
    }

    private static void Fill(float[,,,] values, float value)
    {
        for (int b = 0; b < values.GetLength(0); b++)
            for (int t = 0; t < values.GetLength(1); t++)
                for (int h = 0; h < values.GetLength(2); h++)
                    for (int d = 0; d < values.GetLength(3); d++)
                        values[b, t, h, d] = value;
    }

    private static float[] Reference(
        float[,,,] q,
        float[,,,] k,
        float[,,,] v,
        int offset,
        int queryLength,
        int keyLength,
        float scale)
    {
        int batch = q.GetLength(0);
        int heads = q.GetLength(2);
        int dim = q.GetLength(3);
        var result = new float[batch * queryLength * heads * dim];
        var logits = new float[keyLength];
        int outputIndex = 0;

        for (int b = 0; b < batch; b++)
            for (int tq = 0; tq < queryLength; tq++)
                for (int h = 0; h < heads; h++)
                {
                    float max = float.NegativeInfinity;
                    for (int tk = 0; tk < keyLength; tk++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < dim; d++)
                            dot += q[b, tq + offset, h, d] * k[b, tk + offset, h, d];
                        logits[tk] = dot * scale;
                        max = Math.Max(max, logits[tk]);
                    }

                    float denominator = 0f;
                    for (int tk = 0; tk < keyLength; tk++)
                    {
                        logits[tk] = MathF.Exp(logits[tk] - max);
                        denominator += logits[tk];
                    }

                    for (int d = 0; d < dim; d++, outputIndex++)
                    {
                        float sum = 0f;
                        for (int tk = 0; tk < keyLength; tk++)
                            sum += logits[tk] / denominator * v[b, tk + offset, h, d];
                        result[outputIndex] = sum;
                    }
                }

        return result;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"Mismatch at {i}: expected {expected[i]:R}, actual {actual[i]:R}, tolerance {tolerance:R}");
    }
}
