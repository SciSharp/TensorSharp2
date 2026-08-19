// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Numerical parity tests for the native MLX (Apple Metal) IQ3_XXS kernels
// (tensorsharp_dequant_iq3_xxs + its matmul / simdgroup-matmul / get_rows
// wrappers) against the managed CPU reference dequant in
// TensorSharp.Models.ManagedQuantizedOps (a verbatim port of ggml's
// dequantize_row_iq3_xxs).
//
// Why this matters: Unsloth's "UD" mixed quants put IQ3_XXS on every
// blk.N.ffn_down.weight — the single largest matmul per layer. Before the
// native kernel existed those tensors stayed file-backed and every token paid
// the C# row-dequant fallback (Muse-Glimmer-30B-UD-IQ2_XXS decoded at
// 2.24 tok/s on MLX vs 20.6 tok/s on ggml_metal). A kernel that compiles but
// decodes wrong produces fluent-looking garbage rather than an error, so the
// codebook indexing / sign unpacking / scale math is pinned here bit for bit.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class MlxIq3XxsKernelTests
{
    private const int QK_K = 256;
    private const int Iq3XxsBlockBytes = 98; // d (half) + qs[3*QK_K/8]

    private readonly ITestOutputHelper _output;

    public MlxIq3XxsKernelTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // Runs without MLX: the preload gate is what routes IQ3_XXS weights to the
    // device instead of the file-backed CPU fallback.
    [Fact]
    public void MlxPreloadGate_AcceptsIq3Xxs()
    {
        Assert.True(MlxQuantizedOps.CanPreloadQuantizedType((int)GgmlTensorType.IQ3_XXS));
        // Raw zero-copy wrap of the GGUF mmap, not a repack.
        Assert.False(MlxQuantizedOps.PreloadDuplicatesHostMemory((int)GgmlTensorType.IQ3_XXS));
    }

    // rows = 3 exercises the per-output ("decode") kernel, which accumulates in
    // fp32 — parity there is fp32-rounding tight. rows = 8 hits the
    // simdgroup_matrix threshold (TS_MLX_IQ2XXS_SG_MIN_ROWS, default 8) and so
    // exercises the prefill kernel variant, whose shared
    // BuildIQuantMatmulSimdgroupSource template stages both the dequantized
    // weights and the activations as `half` before simdgroup_multiply_accumulate
    // — hence the looser fp16-scale tolerance there (same template all the other
    // i-quants use; not IQ3_XXS-specific).
    [MlxTheory]
    [InlineData(3, 1e-4f)]
    [InlineData(8, 1e-2f)]
    public void MlxQuantizedMatmul_Iq3XxsMatchesManagedDequantReference(int rows, float tolerance)
    {
        const int inDim = 512;
        const int outDim = 6;
        byte[] weights = CreateIq3XxsRows(outDim, inDim, seed: 0x51F3);

        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 2) * (c + 5) * 0.011f) * 0.75f;

        float[] expected = ManagedDequantMatmul(weights, outDim, inDim, input);

        using var allocator = new MlxAllocator();
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x3A3011 + rows);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            MlxQuantizedOps.PreloadQuantizedWeight(
                allocator, cacheKey, host, (int)GgmlTensorType.IQ3_XXS, inDim, outDim, weights.Length);

            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.IQ3_XXS,
                inDim,
                outDim,
                weights.Length));

            float[] actual = outputTensor.GetElementsAsFloat(rows * outDim);
            AssertCloseWithReport($"IQ3_XXS matmul rows={rows}", expected, actual, tolerance);
        }
        finally
        {
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            Marshal.FreeHGlobal(host);
        }
    }

    // get_rows shares tensorsharp_dequant_iq3_xxs with the matmul kernel but
    // reaches every element individually, so it pins the full 256-element decode
    // (not just its dot product).
    [MlxFact]
    public void MlxQuantizedRows_Iq3XxsMatchesManagedDequantReference()
    {
        const int inDim = 512;
        const int outDim = 6;
        int[] rows = { 5, 0, 3, 3 };
        byte[] weights = CreateIq3XxsRows(outDim, inDim, seed: 0x77A1);

        float[] expected = new float[rows.Length * inDim];
        for (int i = 0; i < rows.Length; i++)
            DequantizeRow(weights, rows[i], inDim, expected, i * inDim);

        using var allocator = new MlxAllocator();
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x6A3011);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            MlxQuantizedOps.PreloadQuantizedWeight(
                allocator, cacheKey, host, (int)GgmlTensorType.IQ3_XXS, inDim, outDim, weights.Length);

            using var indices = Tensor.FromArray(allocator, rows);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows.Length, inDim);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.IQ3_XXS,
                inDim,
                outDim,
                weights.Length,
                indices));

            float[] actual = outputTensor.GetElementsAsFloat(rows.Length * inDim);
            // Element-wise dequant is exact up to fp32 rounding of the same
            // product — the tolerance only absorbs half->float differences.
            AssertCloseWithReport("IQ3_XXS get_rows", expected, actual, 1e-6f);
        }
        finally
        {
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            Marshal.FreeHGlobal(host);
        }
    }

    private void AssertCloseWithReport(string label, float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxAbs = 0;
        double sumAbs = 0;
        double maxMagnitude = 0;
        int worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{label}: non-finite at {i}");
            double err = Math.Abs((double)expected[i] - actual[i]);
            sumAbs += err;
            maxMagnitude = Math.Max(maxMagnitude, Math.Abs((double)expected[i]));
            if (err > maxAbs)
            {
                maxAbs = err;
                worst = i;
            }
        }

        _output.WriteLine(
            $"{label}: n={expected.Length} maxAbsErr={maxAbs:E3} meanAbsErr={sumAbs / expected.Length:E3} " +
            $"maxRefMagnitude={maxMagnitude:F5} worstIndex={worst} " +
            $"(expected {expected[worst]:F7}, actual {actual[worst]:F7})");

        Assert.True(
            maxAbs <= tolerance,
            $"{label}: max abs error {maxAbs:E3} > {tolerance:E3} at index {worst} " +
            $"(expected {expected[worst]}, actual {actual[worst]})");
    }

    // Reference matmul: managed (ggml-verbatim) dequant of each weight row,
    // then a plain fp32 dot product.
    private static float[] ManagedDequantMatmul(byte[] weights, int outDim, int inDim, float[,] input)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeRow(weights, o, inDim, dequantizedRow, 0);
            for (int r = 0; r < rows; r++)
            {
                float sum = 0;
                for (int c = 0; c < inDim; c++)
                    sum += input[r, c] * dequantizedRow[c];
                expected[r * outDim + o] = sum;
            }
        }

        return expected;
    }

    private static void DequantizeRow(byte[] weights, int row, int inDim, float[] dst, int dstOffset)
    {
        long rowBytes = ManagedQuantizedOps.RowSize((int)GgmlTensorType.IQ3_XXS, inDim);
        Assert.Equal((long)(inDim / QK_K) * Iq3XxsBlockBytes, rowBytes);
        ManagedQuantizedOps.DequantizeToFloat32(
            (int)GgmlTensorType.IQ3_XXS, weights, checked((int)(row * rowBytes)), dst, dstOffset, inDim);
    }

    // Synthetic IQ3_XXS super-blocks. Every bit pattern in qs is a legal
    // IQ3_XXS payload (8-bit codebook indices, a 4-bit scale and four 7-bit
    // sign selectors per group), so a deterministic PRNG fill sweeps the whole
    // codebook and every sign/scale combination.
    private static byte[] CreateIq3XxsRows(int rows, int cols, int seed)
    {
        Assert.Equal(0, cols % QK_K);
        int blocksPerRow = cols / QK_K;
        byte[] raw = new byte[(long)rows * blocksPerRow * Iq3XxsBlockBytes];
        uint state = (uint)seed | 1u;
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * Iq3XxsBlockBytes;
                // Exactly representable in fp16 so the scale itself contributes
                // no rounding difference between the reference and the kernel.
                WriteHalf(raw, offset, 0.001953125f + r * 0.000244140625f + b * 0.0001220703125f);
                for (int i = 2; i < Iq3XxsBlockBytes; i++)
                {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    raw[offset + i] = (byte)(state >> 7);
                }
            }
        }

        return raw;
    }

    private static void WriteHalf(byte[] raw, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((System.Half)value);
        raw[offset] = (byte)bits;
        raw[offset + 1] = (byte)(bits >> 8);
    }
}
