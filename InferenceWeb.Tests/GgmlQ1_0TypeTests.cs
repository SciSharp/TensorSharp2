using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The upstream GGML Q1_0 tensor type (type 41): a 128-value block of one F16 scale and
/// 128 sign bits. The GGUF reader, the native bindings and the managed fallback must all
/// agree on that layout without loading a checkpoint.
/// </summary>
public sealed class GgmlQ1_0TypeTests
{
    [Fact]
    public void Q10_UsesTheUpstreamGgmlTypeAndRowLayout()
    {
        // GGML_TYPE_Q1_0 is part of the file format. Renumbering it or treating it as
        // a 256-value K-quant makes GGUF offsets wrong for every following tensor.
        Assert.Equal(41u, (uint)GgmlTensorType.Q1_0);
        Assert.Equal(128, GgufFile.GetBlockSize(GgmlTensorType.Q1_0));
        Assert.Equal(18, GgufFile.GetTypeSize(GgmlTensorType.Q1_0));

        Assert.Equal(18, ManagedQuantizedOps.RowSize((int)GgmlTensorType.Q1_0, 128));
        Assert.Equal(36, GgmlGgufTensorDequant.GetRowSizeBytes((int)GgmlTensorType.Q1_0, 256));
    }

    [Fact]
    public void ManagedQ10_MatchesNativeGgmlForAHandBuiltBlock()
    {
        const int valuesPerBlock = 128;
        const float scale = 0.75f;
        byte[] block = new byte[18];

        ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)scale);
        block[0] = (byte)scaleBits;
        block[1] = (byte)(scaleBits >> 8);

        // Q1_0 stores one sign bit per value, least-significant bit first. Use a
        // deliberately asymmetric pattern so byte order and bit order are both tested.
        block[2] = 0b_1001_0110;
        for (int i = 1; i < 16; i++)
            block[2 + i] = (byte)(0xA5 ^ (i * 29));

        float[] managed = new float[valuesPerBlock];
        ManagedQuantizedOps.DequantizeToFloat32(
            (int)GgmlTensorType.Q1_0, block, 0, managed, 0, valuesPerBlock);

        float[] native = new float[valuesPerBlock];
        GgmlGgufTensorDequant.DequantizeToFloat32(
            (int)GgmlTensorType.Q1_0, block, 0, native, 0, valuesPerBlock);

        Assert.Equal(-scale, managed[0]);
        Assert.Equal(scale, managed[1]);
        Assert.Equal(scale, managed[2]);
        Assert.Equal(-scale, managed[3]);
        Assert.Equal(scale, managed[7]);
        Assert.Equal(native, managed);
    }
}
