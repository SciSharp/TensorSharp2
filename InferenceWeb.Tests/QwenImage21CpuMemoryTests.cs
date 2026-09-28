// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>
/// The cpu backend's up-front memory estimate for Qwen-Image-2.1 (QwenImage21CpuMemory): the
/// VAE decode term is the decoder's measured live set per pixel (the real-VAE test in
/// QwenImage21VaeMemoryTests pins it), and a size that cannot fit is refused with the largest
/// size that can.
/// </summary>
public sealed class QwenImage21CpuMemoryTests
{
    private const long Gib = 1L << 30;
    private const long Q4TransformerBytes = 4_189_343_904;   // qwen_image_2.1_Q4_K_M.gguf

    [Fact]
    public void DecodeEstimateGrowsWithThePixelCount()
    {
        Assert.Equal(576 * sizeof(float), QwenImage21CpuMemory.DecodeFeatureBytesPerPixel);
        long oneK = QwenImage21CpuMemory.EstimateDecodeBytes(1024, 1024), twoK = QwenImage21CpuMemory.EstimateDecodeBytes(2048, 2048);
        Assert.Equal(QwenImage21CpuMemory.DecodeFeatureBytesPerPixel * (2048L * 2048 - 1024L * 1024), twoK - oneK);
        // The 2K decode is the request's peak on the cpu backend, the denoise's is below it.
        Assert.True(twoK > QwenImage21CpuMemory.EstimateDenoiseBytes(2048, 2048, 256, Q4TransformerBytes));
        Assert.InRange(twoK, 10 * Gib, 14 * Gib);
    }

    [Fact]
    public void TwoKFitsThirtyTwoGigabytesAndIsRefusedWithAFittingSizeOnEight()
    {
        Assert.Null(QwenImage21CpuMemory.Refusal(2048, 2048, 256, Q4TransformerBytes, 32 * Gib));
        string refusal = QwenImage21CpuMemory.Refusal(2048, 2048, 256, Q4TransformerBytes, 8 * Gib);
        Assert.NotNull(refusal);
        Assert.Contains("2048x2048", refusal, StringComparison.Ordinal);
        Assert.Contains("8.0 GiB", refusal, StringComparison.Ordinal);
        Assert.Contains("VAE decode", refusal, StringComparison.Ordinal);
        Assert.Contains(QwenImage21CpuMemory.CheckVariable + "=0", refusal, StringComparison.Ordinal);
        int side = QwenImage21CpuMemory.LargestSquareSide(256, Q4TransformerBytes, 8 * Gib);
        Assert.Contains($"at most about {side}x{side}", refusal, StringComparison.Ordinal);
        Assert.InRange(side, 1024, 2016);
        Assert.Equal(0, side % 32);
        Assert.Null(QwenImage21CpuMemory.Refusal(side, side, 256, Q4TransformerBytes, 8 * Gib));
        Assert.NotNull(QwenImage21CpuMemory.Refusal(side + 32, side + 32, 256, Q4TransformerBytes, 8 * Gib));
    }

    [Fact]
    public void ATransformerLargerThanMemoryIsNamedAsTheCause()
    {
        string refusal = QwenImage21CpuMemory.Refusal(512, 512, 256, 20 * Gib, 16 * Gib);
        Assert.NotNull(refusal);
        Assert.Contains("transformer", refusal, StringComparison.Ordinal);
        Assert.Contains("cannot hold even a small image", refusal, StringComparison.Ordinal);
        Assert.Equal(0, QwenImage21CpuMemory.LargestSquareSide(256, 20 * Gib, 16 * Gib));
    }

    [Fact]
    public void UnknownMemoryIsNeverRefused() =>
        Assert.Null(QwenImage21CpuMemory.Refusal(8192, 8192, 256, Q4TransformerBytes, 0));
}
