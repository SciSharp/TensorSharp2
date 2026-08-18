// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Numerical parity for the DECODE / short-batch MLX matmul kernels of the
// i-quant family (IQ2_XXS, IQ2_S, IQ3_S) against the managed CPU reference
// dequant in TensorSharp.Models.ManagedQuantizedOps, which is a verbatim port
// of ggml's dequantize_row_* routines. (IQ3_XXS has its own dedicated suite in
// MlxIq3XxsKernelTests.)
//
// Why this exists: those kernels were rewritten from a per-ELEMENT dequant
// (one super-block header load + one codebook lookup per weight value) to a
// block-amortized form that decodes 8 consecutive weights per call, because the
// per-element form ran Muse-Glimmer-30B-UD-IQ2_XXS decode at ~35 GB/s against a
// ~205 GB/s roofline (3.6 tok/s on MLX vs 21.3 on ggml_metal, M5 Pro). A
// mis-indexed codebook or sign group in the rewritten form produces
// fluent-looking garbage rather than an error, so the arithmetic is pinned here
// against the reference.
//
// TS_MLX_IQ_DECODE_DOT8=0 selects the legacy per-element body; both must agree
// with the reference, which is what makes that switch a usable A/B.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class MlxIQuantDecodeMatmulTests
{
    private const int QK_K = 256;

    private readonly ITestOutputHelper _output;

    public MlxIQuantDecodeMatmulTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // rows = 1 is the decode shape the pipelined path issues; rows = 3 stays
    // below the simdgroup_matrix threshold (TS_MLX_IQ2XXS_SG_MIN_ROWS, default
    // 8) so both exercise the per-output kernel this suite is about. The
    // tolerance is loose enough for the block-scale factoring the amortized
    // form does (sum(d*g*x) -> d*sum(g*x)) but far tighter than any indexing
    // mistake could survive.
    [MlxTheory]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 1)]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 3)]
    [InlineData((int)GgmlTensorType.IQ2_S, 1)]
    [InlineData((int)GgmlTensorType.IQ2_S, 3)]
    [InlineData((int)GgmlTensorType.IQ3_S, 1)]
    [InlineData((int)GgmlTensorType.IQ3_S, 3)]
    public void DecodeMatmul_MatchesManagedDequantReference(int ggmlType, int rows)
    {
        const int inDim = 512;   // 2 super-blocks per row
        const int outDim = 6;
        byte[] weights = CreateRows(ggmlType, outDim, inDim, seed: 0x2C51 + ggmlType);

        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 3) * (c + 7) * 0.013f) * 0.6f;

        float[] expected = ManagedDequantMatmul(ggmlType, weights, outDim, inDim, input);

        using var allocator = new MlxAllocator();
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x5D0000 + ggmlType * 16 + rows);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            MlxQuantizedOps.PreloadQuantizedWeight(
                allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);

            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor, inputTensor, cacheKey, IntPtr.Zero,
                ggmlType, inDim, outDim, weights.Length));

            float[] actual = outputTensor.GetElementsAsFloat(rows * outDim);
            AssertClose($"{(GgmlTensorType)ggmlType} decode matmul rows={rows}", expected, actual, 1e-4f);
        }
        finally
        {
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            Marshal.FreeHGlobal(host);
        }
    }

    // The pipelined step's next-token embedding gather goes through get_rows,
    // which still uses the per-ELEMENT dequant helper - so this pins that the
    // rewrite left the scalar path intact and the two agree element for
    // element.
    [MlxTheory]
    [InlineData((int)GgmlTensorType.IQ2_XXS)]
    [InlineData((int)GgmlTensorType.IQ2_S)]
    [InlineData((int)GgmlTensorType.IQ3_S)]
    public void GetRows_MatchesManagedDequantReference(int ggmlType)
    {
        const int inDim = 512;
        const int outDim = 6;
        int[] rowIds = { 4, 0, 2, 2 };
        byte[] weights = CreateRows(ggmlType, outDim, inDim, seed: 0x91B7 + ggmlType);

        float[] expected = new float[rowIds.Length * inDim];
        for (int i = 0; i < rowIds.Length; i++)
            DequantizeRow(ggmlType, weights, rowIds[i], inDim, expected, i * inDim);

        using var allocator = new MlxAllocator();
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x5E0000 + ggmlType);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            MlxQuantizedOps.PreloadQuantizedWeight(
                allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);

            using var indices = Tensor.FromArray(allocator, rowIds);
            using var outputTensor = new Tensor(allocator, DType.Float32, rowIds.Length, inDim);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor, cacheKey, IntPtr.Zero,
                ggmlType, inDim, outDim, weights.Length, indices));

            float[] actual = outputTensor.GetElementsAsFloat(rowIds.Length * inDim);
            AssertClose($"{(GgmlTensorType)ggmlType} get_rows", expected, actual, 1e-6f);
        }
        finally
        {
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            Marshal.FreeHGlobal(host);
        }
    }

    private void AssertClose(string label, float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxAbs = 0;
        int worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{label}: non-finite at {i}");
            double err = Math.Abs((double)expected[i] - actual[i]);
            if (err > maxAbs)
            {
                maxAbs = err;
                worst = i;
            }
        }

        _output.WriteLine($"{label}: n={expected.Length} maxAbsErr={maxAbs:E3} worstIndex={worst} " +
            $"(expected {expected[worst]:F7}, actual {actual[worst]:F7})");
        Assert.True(maxAbs <= tolerance,
            $"{label}: max abs error {maxAbs:E3} > {tolerance:E3} at index {worst} " +
            $"(expected {expected[worst]}, actual {actual[worst]})");
    }

    private static float[] ManagedDequantMatmul(int ggmlType, byte[] weights, int outDim, int inDim, float[,] input)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeRow(ggmlType, weights, o, inDim, dequantizedRow, 0);
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

    private static void DequantizeRow(int ggmlType, byte[] weights, int row, int inDim, float[] dst, int dstOffset)
    {
        long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, inDim);
        ManagedQuantizedOps.DequantizeToFloat32(
            ggmlType, weights, checked((int)(row * rowBytes)), dst, dstOffset, inDim);
    }

    /// <summary>
    /// Synthetic super-blocks: an fp16 scale that is exactly representable (so
    /// the scale itself contributes no rounding difference between reference
    /// and kernel) followed by a deterministic xorshift fill of the payload.
    /// Every bit pattern in these formats' payloads is legal - they are all
    /// codebook indices, packed sign selectors and 4-bit scales - so the fill
    /// sweeps the whole codebook and every sign/scale combination.
    /// </summary>
    private static byte[] CreateRows(int ggmlType, int rows, int cols, int seed)
    {
        Assert.Equal(0, cols % QK_K);
        int blocksPerRow = cols / QK_K;
        int blockBytes = checked((int)ManagedQuantizedOps.RowSize(ggmlType, QK_K));
        byte[] raw = new byte[(long)rows * blocksPerRow * blockBytes];

        // IQ2_XXS / IQ2_S / IQ3_S all start their super-block with the fp16
        // scale EXCEPT IQ3_S, whose layout is d, qs[64], qh[8], signs[32],
        // scales[16] - d is still first. So one offset works for all three.
        uint state = (uint)seed | 1u;
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                WriteHalf(raw, offset, 0.001953125f + r * 0.000244140625f + b * 0.0001220703125f);
                for (int i = 2; i < blockBytes; i++)
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
