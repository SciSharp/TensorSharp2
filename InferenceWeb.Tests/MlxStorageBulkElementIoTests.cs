using TensorSharp;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

/// <summary>
/// Regression tests for MlxStorage's BULK element IO
/// (<see cref="Storage.GetElementsAsFloat"/> / <see cref="Storage.SetElementsAsFloat"/>).
///
/// Both used to be a per-element loop, and every element went through
/// EnsureHostReadable() -> ThrowIfDestroyed() + lock (sync). The two hot callers
/// are the Muse-Glimmer mmproj upload (~1.85e9 parameters when the tower is
/// dequantized to F32) and ModelBase.TensorToFloatArray on the LM head, which is
/// 202,048 locked per-element reads for every decoded token. They now do one
/// host sync plus one typed copy loop, so these tests pin the observable
/// contract: identical values to the per-element path, and unchanged
/// host/device dirty semantics.
/// </summary>
public class MlxStorageBulkElementIoTests
{
    [Fact]
    public void MlxBulkFloat32Io_MatchesPerElementPath()
    {
        if (!MlxBackend.IsAvailable())
            return;

        AssertBulkMatchesPerElement(DType.Float32, tolerance: 0f);
    }

    [Fact]
    public void MlxBulkFloat16Io_MatchesPerElementPath()
    {
        if (!MlxBackend.IsAvailable())
            return;

        // F16 loses precision on the way in; the bulk path must lose EXACTLY the
        // same precision as the per-element path (asserted bit-for-bit below),
        // and the round trip must still land within half's ~2^-11 relative step
        // for the +/-10 range used here.
        AssertBulkMatchesPerElement(DType.Float16, tolerance: 0.01f);
    }

    /// <summary>
    /// A bulk WRITE must mark the storage host-dirty, exactly as the per-element
    /// path did, or the next kernel silently reads the stale device array.
    /// </summary>
    [Fact]
    public void MlxBulkSetElementsAsFloat_IsVisibleToTheNextDeviceOp()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var source = new Tensor(allocator, DType.Float32, 2, 3);

        // Fill on the device first, so the storage holds a device array and the
        // host buffer is the stale copy.
        Ops.Fill(source, 0f);
        source.Storage.SetElementsAsFloat(source.StorageOffset, new[] { 1f, 2f, 3f, 4f, 5f, 6f });

        using var doubled = new Tensor(allocator, DType.Float32, 2, 3);
        Ops.Mul(doubled, source, 2f);

        float[] actual = doubled.GetElementsAsFloat(6);
        for (int i = 0; i < actual.Length; i++)
            AssertClose(2f * (i + 1), actual[i]);
    }

    /// <summary>
    /// A bulk READ must first pull the device array back to the host, and must
    /// keep doing so after each subsequent device op.
    /// </summary>
    [Fact]
    public void MlxBulkGetElementsAsFloat_SeesDeviceComputedValues()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 2, 3);

        Ops.Fill(tensor, 1.25f);
        float[] filled = tensor.Storage.GetElementsAsFloat(tensor.StorageOffset, 6);
        Assert.All(filled, v => AssertClose(1.25f, v));

        Ops.Mul(tensor, tensor, 4f);
        float[] scaled = tensor.Storage.GetElementsAsFloat(tensor.StorageOffset, 6);
        Assert.All(scaled, v => AssertClose(5f, v));
    }

    private static void AssertBulkMatchesPerElement(DType elementType, float tolerance)
    {
        // A non-zero start index and a length that stops short of the end so the
        // bulk pointer arithmetic and the untouched-region checks both bite.
        const int count = 4096;
        const int offset = 7;
        const int length = 4000;

        var random = new Random(20260812);
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (float)(random.NextDouble() * 20.0 - 10.0);

        using var allocator = new MlxAllocator();
        using var bulk = new Tensor(allocator, elementType, count);
        using var perElement = new Tensor(allocator, elementType, count);

        bulk.Storage.SetElementsAsFloat(offset, values);
        for (int i = 0; i < length; i++)
            perElement.Storage.SetElementAsFloat(offset + i, values[i]);

        float[] bulkRead = bulk.Storage.GetElementsAsFloat(offset, length);
        float[] bulkReadOfPerElementWrite = perElement.Storage.GetElementsAsFloat(offset, length);
        Assert.Equal(length, bulkRead.Length);

        for (int i = 0; i < length; i++)
        {
            float perElementRead = perElement.Storage.GetElementAsFloat(offset + i);

            // Bulk write + bulk read == per-element write + per-element read,
            // bit for bit (same dtype conversion, so no tolerance is warranted).
            Assert.Equal(perElementRead, bulkRead[i]);
            // Cross-check the two halves independently: bulk write read back one
            // element at a time, and per-element write read back in bulk.
            Assert.Equal(perElementRead, bulk.Storage.GetElementAsFloat(offset + i));
            Assert.Equal(perElementRead, bulkReadOfPerElementWrite[i]);
            // And the values actually survive the round trip.
            Assert.True(Math.Abs(values[i] - bulkRead[i]) <= tolerance,
                $"index {i}: wrote {values[i]}, read back {bulkRead[i]}");
        }

        // Nothing outside [offset, offset + length) was touched.
        float[] head = bulk.Storage.GetElementsAsFloat(0, offset);
        Assert.All(head, v => Assert.Equal(0f, v));
        float[] tail = bulk.Storage.GetElementsAsFloat(offset + length, count - offset - length);
        Assert.All(tail, v => Assert.Equal(0f, v));

        // Degenerate range: no throw, no allocation surprises.
        Assert.Empty(bulk.Storage.GetElementsAsFloat(offset, 0));
        bulk.Storage.SetElementsAsFloat(offset, Array.Empty<float>());
    }

    private static void AssertClose(float expected, float actual, float tolerance = 1e-4f)
    {
        Assert.True(Math.Abs(expected - actual) <= tolerance,
            $"expected {expected}, actual {actual}");
    }
}
