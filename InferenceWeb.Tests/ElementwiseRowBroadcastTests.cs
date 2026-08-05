using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

/// <summary>
/// Row-broadcast elementwise ops: <c>x[rows, cols] OP bias[cols]</c>.
///
/// The generic Apply3 path underneath these ops advances all three iterators
/// together and stops at the shortest operand's last block, so a [cols] right
/// operand touched row 0 and silently left every other row unchanged. Add had a
/// dedicated fast path; Sub, Mul and Div did not, and Gemma 4's vision projector
/// standardises its pooled patches with exactly that shape (Sub then Mul) — on
/// the CUDA backend, whose fallback routes here, only the first patch was being
/// standardised and the model hallucinated the image content.
/// </summary>
public class ElementwiseRowBroadcastTests
{
    private readonly IAllocator _alloc = new CpuAllocator(BlasEnum.DotNet);

    private Tensor TensorFrom(float[] values, params long[] sizes)
    {
        var tensor = new Tensor(_alloc, DType.Float32, sizes);
        tensor.SetElementsAsFloat(values);
        return tensor;
    }

    private const int Rows = 5;
    private const int Cols = 9;   // deliberately not a multiple of the SIMD width

    private static float[] Matrix()
    {
        var m = new float[Rows * Cols];
        for (int i = 0; i < m.Length; i++) m[i] = 1.0f + i * 0.25f;
        return m;
    }

    private static float[] Bias()
    {
        var b = new float[Cols];
        for (int c = 0; c < Cols; c++) b[c] = 0.5f + c;
        return b;
    }

    private void AssertBroadcast(Func<Tensor, Tensor, Tensor> op, Func<float, float, float> scalar)
    {
        float[] m = Matrix();
        float[] b = Bias();
        using var lhs = TensorFrom(m, Rows, Cols);
        using var rhs = TensorFrom(b, Cols);
        using var result = op(lhs, rhs);

        float[] observed = result.GetElementsAsFloat(Rows * Cols);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                float expected = scalar(m[r * Cols + c], b[c]);
                int i = r * Cols + c;
                Assert.True(MathF.Abs(expected - observed[i]) <= 1e-5f,
                    $"row {r} col {c}: expected {expected}, observed {observed[i]}");
            }
        }
    }

    [Fact]
    public void Add_BroadcastsBiasToEveryRow()
        => AssertBroadcast((l, r) => Ops.Add(null, l, r), (x, y) => x + y);

    [Fact]
    public void Sub_BroadcastsBiasToEveryRow()
        => AssertBroadcast((l, r) => Ops.Sub(null, l, r), (x, y) => x - y);

    [Fact]
    public void Mul_BroadcastsScaleToEveryRow()
        => AssertBroadcast((l, r) => Ops.Mul(null, l, r), (x, y) => x * y);

    [Fact]
    public void Div_BroadcastsScaleToEveryRow()
        => AssertBroadcast((l, r) => Ops.Div(null, l, r), (x, y) => x / y);

    [Fact]
    public void Mul_InPlaceBroadcast_UpdatesEveryRow()
    {
        float[] m = Matrix();
        float[] b = Bias();
        using var lhs = TensorFrom(m, Rows, Cols);
        using var rhs = TensorFrom(b, Cols);

        Ops.Mul(lhs, lhs, rhs);

        float[] observed = lhs.GetElementsAsFloat(Rows * Cols);
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                Assert.True(MathF.Abs(m[r * Cols + c] * b[c] - observed[r * Cols + c]) <= 1e-5f,
                    $"row {r} col {c} was not scaled");
    }

    [Fact]
    public void SameShapeOperands_AreUnaffected()
    {
        float[] a = Matrix();
        float[] b = Matrix();
        for (int i = 0; i < b.Length; i++) b[i] = b[i] * 0.5f + 1.0f;

        using var lhs = TensorFrom(a, Rows, Cols);
        using var rhs = TensorFrom(b, Rows, Cols);
        using var sub = Ops.Sub(null, lhs, rhs);
        using var mul = Ops.Mul(null, lhs, rhs);

        float[] subObserved = sub.GetElementsAsFloat(a.Length);
        float[] mulObserved = mul.GetElementsAsFloat(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(MathF.Abs((a[i] - b[i]) - subObserved[i]) <= 1e-5f, $"sub index {i}");
            Assert.True(MathF.Abs((a[i] * b[i]) - mulObserved[i]) <= 1e-5f, $"mul index {i}");
        }
    }
}
