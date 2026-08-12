// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class GgmlMuseGlimmerVisionBlockTests
{
    private readonly ITestOutputHelper _output;

    public GgmlMuseGlimmerVisionBlockTests(ITestOutputHelper output) => _output = output;

    private const int Q8_0 = 8;
    private const int Hidden = 192;
    private const int Heads = 2;
    private const int HeadDim = 96;
    private const int Intermediate = 256;
    private const int Tokens = 17;
    private const float Eps = 1e-5f;
    private const float RopeTheta = 10_000f;

    [Fact]
    public void MuseGlimmerVisionBlock_Q80_GlobalAndPackedWindowsMatchHostReference()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND"), "cuda",
                StringComparison.OrdinalIgnoreCase))
        {
            // GGML has one process-global backend. This test is deliberately
            // opt-in so the default CPU suite remains deterministic.
            return;
        }

        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cuda);
        var allocator = new GgmlAllocator(context, 0);

        float[] initial = Values(Tokens * Hidden, 0.43f, 0.071f);
        float[] ln1W = Values(Hidden, 0.11f, 0.031f, add: 1f);
        float[] ln1B = Values(Hidden, 0.07f, 0.017f);
        float[] qBias = Values(Hidden, 0.05f, 0.019f);
        float[] kBias = Values(Hidden, 0.04f, 0.023f);
        float[] vBias = Values(Hidden, 0.06f, 0.029f);
        float[] outBias = Values(Hidden, 0.03f, 0.037f);
        float[] ln2W = Values(Hidden, 0.09f, 0.041f, add: 1f);
        float[] ln2B = Values(Hidden, 0.05f, 0.043f);
        float[] upBias = Values(Intermediate, 0.025f, 0.047f);
        float[] downBias = Values(Hidden, 0.02f, 0.053f);

        using QuantizedWeight q = Weight(Hidden, Hidden, 0.038f, 0.011f);
        using QuantizedWeight k = Weight(Hidden, Hidden, 0.035f, 0.013f);
        using QuantizedWeight v = Weight(Hidden, Hidden, 0.041f, 0.017f);
        using QuantizedWeight o = Weight(Hidden, Hidden, 0.032f, 0.019f);
        using QuantizedWeight up = Weight(Hidden, Intermediate, 0.027f, 0.007f);
        using QuantizedWeight down = Weight(Intermediate, Hidden, 0.024f, 0.009f);

        float[] qF32 = Dequantize(q);
        float[] kF32 = Dequantize(k);
        float[] vF32 = Dequantize(v);
        float[] oF32 = Dequantize(o);
        float[] upF32 = Dequantize(up);
        float[] downF32 = Dequantize(down);

        int[] posW = Enumerable.Range(0, Tokens).Select(i => i % 5 + 1).ToArray();
        int[] posH = Enumerable.Range(0, Tokens).Select(i => i / 5 + 1).ToArray();
        int[] packedWindows = { 0, 8, Tokens };
        float[][] actualByMode = new float[2][];

        for (int mode = 0; mode < 2; mode++)
        {
            bool global = mode == 0;
            int[] referenceOffsets = global ? new[] { 0, Tokens } : packedWindows;
            float[] expected = ReferenceBlock(initial, ln1W, ln1B,
                qF32, qBias, kF32, kBias, vF32, vBias, oF32, outBias,
                ln2W, ln2B, upF32, upBias, downF32, downBias,
                posW, posH, referenceOffsets);

            using Tensor hidden = Tensor.FromArray(allocator, ToMatrix(initial, Tokens, Hidden));
            using Tensor ln1WT = Tensor.FromArray(allocator, ln1W);
            using Tensor ln1BT = Tensor.FromArray(allocator, ln1B);
            using Tensor qBT = Tensor.FromArray(allocator, qBias);
            using Tensor kBT = Tensor.FromArray(allocator, kBias);
            using Tensor vBT = Tensor.FromArray(allocator, vBias);
            using Tensor outBT = Tensor.FromArray(allocator, outBias);
            using Tensor ln2WT = Tensor.FromArray(allocator, ln2W);
            using Tensor ln2BT = Tensor.FromArray(allocator, ln2B);
            using Tensor upBT = Tensor.FromArray(allocator, upBias);
            using Tensor downBT = Tensor.FromArray(allocator, downBias);

            bool fused = GgmlBasicOps.TryMuseGlimmerVisionBlock(
                hidden,
                ln1WT, ln1BT,
                q.Data, q.GgmlType, q.Ne0, q.Ne1, q.RawBytes, qBT,
                k.Data, k.GgmlType, k.Ne0, k.Ne1, k.RawBytes, kBT,
                v.Data, v.GgmlType, v.Ne0, v.Ne1, v.RawBytes, vBT,
                o.Data, o.GgmlType, o.Ne0, o.Ne1, o.RawBytes, outBT,
                ln2WT, ln2BT,
                up.Data, up.GgmlType, up.Ne0, up.Ne1, up.RawBytes, upBT,
                down.Data, down.GgmlType, down.Ne0, down.Ne1, down.RawBytes, downBT,
                posW, posH, packedWindows, global,
                Heads, HeadDim, Eps, RopeTheta);

            Assert.True(fused, $"CUDA fused Muse-Glimmer block rejected the {(global ? "global" : "packed-window")} geometry: {LastNativeError()}");
            float[] actual = hidden.GetElementsAsFloat(Tokens * Hidden);
            var error = AssertClose(expected, actual, absoluteTolerance: 6e-3f, relativeTolerance: 2e-3f);
            _output.WriteLine($"{(global ? "global" : "packed windows")}: max abs={error.maxAbsolute:R}, max relative={error.maxRelative:R}");
            actualByMode[mode] = actual;
        }

        // This makes the sparse-window half of the test meaningful: if the
        // offsets were ignored, the two executions would be indistinguishable.
        Assert.True(MaxAbsDifference(actualByMode[0], actualByMode[1]) > 1e-3f,
            "Global and packed-window attention unexpectedly produced the same block output.");
    }

    private static QuantizedWeight Weight(int input, int output, float amplitude, float frequency)
    {
        float[] values = new float[input * output];
        for (int row = 0; row < output; row++)
            for (int col = 0; col < input; col++)
                values[row * input + col] = amplitude * MathF.Sin((row + 1) * (col + 3) * frequency);

        byte[] raw = new byte[checked(values.Length / 32 * 34)];
        ManagedQuantizedOps.QuantizeRowFromFloat32(Q8_0, values, 0, raw, 0, values.Length);
        return new QuantizedWeight(raw, Q8_0, input, output);
    }

    private static unsafe float[] Dequantize(QuantizedWeight weight)
    {
        float[] values = new float[checked((int)(weight.Ne0 * weight.Ne1))];
        fixed (float* dst = values)
            ManagedQuantizedOps.DequantizeToFloat32Native(
                weight.GgmlType, weight.Data, (IntPtr)dst, values.Length);
        return values;
    }

    private static float[] ReferenceBlock(
        float[] initial,
        float[] ln1W, float[] ln1B,
        float[] qW, float[] qB,
        float[] kW, float[] kB,
        float[] vW, float[] vB,
        float[] outW, float[] outB,
        float[] ln2W, float[] ln2B,
        float[] upW, float[] upB,
        float[] downW, float[] downB,
        int[] posW, int[] posH, int[] offsets)
    {
        float[] ln1 = LayerNorm(initial, Tokens, Hidden, ln1W, ln1B);
        float[] q = MatMul(ln1, Tokens, Hidden, qW, Hidden, qB);
        float[] k = MatMul(ln1, Tokens, Hidden, kW, Hidden, kB);
        float[] v = MatMul(ln1, Tokens, Hidden, vW, Hidden, vB);
        ApplyRope(q, posW, posH);
        ApplyRope(k, posW, posH);

        // The fused CUDA graph casts K/V to F16 before flash attention, just as
        // llama.cpp's CLIP graph does. Mirror that deliberate precision choice.
        for (int i = 0; i < k.Length; i++) k[i] = (float)(System.Half)k[i];
        for (int i = 0; i < v.Length; i++) v[i] = (float)(System.Half)v[i];

        float[] attended = Attention(q, k, v, offsets);
        float[] projected = MatMul(attended, Tokens, Hidden, outW, Hidden, outB);
        float[] residual1 = Add(initial, projected);
        float[] ln2 = LayerNorm(residual1, Tokens, Hidden, ln2W, ln2B);
        float[] fc1 = MatMul(ln2, Tokens, Hidden, upW, Intermediate, upB);
        for (int i = 0; i < fc1.Length; i++)
            fc1[i] = 0.5f * fc1[i] * (1f + Erf(fc1[i] * 0.7071067811865475f));
        float[] fc2 = MatMul(fc1, Tokens, Intermediate, downW, Hidden, downB);
        return Add(residual1, fc2);
    }

    private static float[] LayerNorm(float[] input, int rows, int cols, float[] weight, float[] bias)
    {
        float[] result = new float[input.Length];
        for (int row = 0; row < rows; row++)
        {
            int start = row * cols;
            float mean = 0f;
            for (int col = 0; col < cols; col++) mean += input[start + col];
            mean /= cols;
            float variance = 0f;
            for (int col = 0; col < cols; col++)
            {
                float delta = input[start + col] - mean;
                variance += delta * delta;
            }
            float inverseStdDev = 1f / MathF.Sqrt(variance / cols + Eps);
            for (int col = 0; col < cols; col++)
                result[start + col] = (input[start + col] - mean) * inverseStdDev * weight[col] + bias[col];
        }
        return result;
    }

    private static float[] MatMul(float[] input, int rows, int inputDim, float[] weight, int outputDim, float[] bias)
    {
        float[] result = new float[rows * outputDim];
        for (int row = 0; row < rows; row++)
            for (int output = 0; output < outputDim; output++)
            {
                float sum = bias[output];
                int inputStart = row * inputDim;
                int weightStart = output * inputDim;
                for (int col = 0; col < inputDim; col++)
                    sum += input[inputStart + col] * weight[weightStart + col];
                result[row * outputDim + output] = sum;
            }
        return result;
    }

    private static void ApplyRope(float[] values, int[] posW, int[] posH)
    {
        int ropeDim = HeadDim / 2;
        int pairsPerHalf = ropeDim / 2;
        for (int token = 0; token < Tokens; token++)
            for (int head = 0; head < Heads; head++)
                for (int half = 0; half < 2; half++)
                    for (int pair = 0; pair < pairsPerHalf; pair++)
                    {
                        int position = half == 0 ? posW[token] : posH[token];
                        float angle = position * MathF.Pow(RopeTheta, -2f * pair / ropeDim);
                        float cosine = MathF.Cos(angle);
                        float sine = MathF.Sin(angle);
                        int index = token * Hidden + head * HeadDim + half * ropeDim + pair * 2;
                        float x0 = values[index];
                        float x1 = values[index + 1];
                        values[index] = x0 * cosine - x1 * sine;
                        values[index + 1] = x1 * cosine + x0 * sine;
                    }
    }

    private static float[] Attention(float[] q, float[] k, float[] v, int[] offsets)
    {
        float[] result = new float[q.Length];
        float scale = 1f / MathF.Sqrt(HeadDim);
        for (int window = 0; window < offsets.Length - 1; window++)
            for (int queryToken = offsets[window]; queryToken < offsets[window + 1]; queryToken++)
                for (int head = 0; head < Heads; head++)
                {
                    int length = offsets[window + 1] - offsets[window];
                    float[] logits = new float[length];
                    float max = float.NegativeInfinity;
                    for (int keyToken = offsets[window]; keyToken < offsets[window + 1]; keyToken++)
                    {
                        float dot = 0f;
                        int qStart = queryToken * Hidden + head * HeadDim;
                        int kStart = keyToken * Hidden + head * HeadDim;
                        for (int dim = 0; dim < HeadDim; dim++) dot += q[qStart + dim] * k[kStart + dim];
                        logits[keyToken - offsets[window]] = dot * scale;
                        max = Math.Max(max, dot * scale);
                    }
                    float denominator = 0f;
                    for (int i = 0; i < length; i++)
                    {
                        logits[i] = MathF.Exp(logits[i] - max);
                        denominator += logits[i];
                    }
                    int outputStart = queryToken * Hidden + head * HeadDim;
                    for (int dim = 0; dim < HeadDim; dim++)
                    {
                        float sum = 0f;
                        for (int keyToken = offsets[window]; keyToken < offsets[window + 1]; keyToken++)
                        {
                            int vIndex = keyToken * Hidden + head * HeadDim + dim;
                            sum += logits[keyToken - offsets[window]] / denominator * v[vIndex];
                        }
                        result[outputStart + dim] = sum;
                    }
                }
        return result;
    }

    private static float Erf(float value)
    {
        double x = value;
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        if (x > 6) return (float)sign;
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double polynomial = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                              - 0.284496736) * t + 0.254829592) * t;
        return (float)(sign * (1.0 - polynomial * Math.Exp(-x * x)));
    }

    private static float[] Add(float[] left, float[] right)
    {
        float[] result = new float[left.Length];
        for (int i = 0; i < result.Length; i++) result[i] = left[i] + right[i];
        return result;
    }

    private static float[] Values(int count, float amplitude, float frequency, float add = 0f)
    {
        float[] result = new float[count];
        for (int i = 0; i < count; i++) result[i] = add + amplitude * MathF.Sin((i + 1) * frequency);
        return result;
    }

    private static float[,] ToMatrix(float[] values, int rows, int cols)
    {
        float[,] matrix = new float[rows, cols];
        Buffer.BlockCopy(values, 0, matrix, 0, values.Length * sizeof(float));
        return matrix;
    }

    private static (float maxAbsolute, float maxRelative) AssertClose(
        float[] expected, float[] actual, float absoluteTolerance, float relativeTolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxAbsolute = 0f;
        float maxRelative = 0f;
        int worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float absolute = Math.Abs(expected[i] - actual[i]);
            float relative = absolute / Math.Max(1e-5f, Math.Abs(expected[i]));
            if (absolute > maxAbsolute) { maxAbsolute = absolute; worst = i; }
            maxRelative = Math.Max(maxRelative, relative);
            Assert.True(absolute <= absoluteTolerance + relativeTolerance * Math.Abs(expected[i]),
                $"Mismatch at {i}: expected={expected[i]:R}, actual={actual[i]:R}, abs={absolute:R}, " +
                $"maxAbs={maxAbsolute:R}, maxRel={maxRelative:R}, worst={worst}.");
        }
        return (maxAbsolute, maxRelative);
    }

    private static float MaxAbsDifference(float[] left, float[] right)
    {
        float max = 0f;
        for (int i = 0; i < left.Length; i++) max = Math.Max(max, Math.Abs(left[i] - right[i]));
        return max;
    }

    private static string LastNativeError()
    {
        Type native = typeof(GgmlBasicOps).Assembly.GetType("TensorSharp.GGML.GgmlNative");
        var method = native?.GetMethod("LastNativeError", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        return method?.Invoke(null, new object[] { "(no native error)" }) as string ?? "(unavailable)";
    }
}
