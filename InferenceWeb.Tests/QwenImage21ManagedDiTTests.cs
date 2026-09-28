// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// The pure-C# Qwen-Image-2.1 transformer (the cpu backend) on a tiny synthetic model with
/// F32 weights, against an independent double-precision implementation of the native graph
/// (ggml_ops_qwen_image21.cpp): time MLP and shared AdaLN, text conditioner, segmented joint
/// sequence with a reference image, per-head RMSNorm + half-split RoPE, causal text and
/// bidirectional image attention, gated residuals, SwiGLU, output head; LoRA descriptors as
/// QwenImage21LoraSet packs them; and the prefix KV cache. Every kernel width this host
/// supports (AVX-512, AVX2, scalar) is exercised. No native library is involved.
/// </summary>
public sealed unsafe class QwenImage21ManagedDiTTests
{
    public static IEnumerable<object[]> Widths()
    {
        if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported) yield return new object[] { 16 };
        if (Vector256.IsHardwareAccelerated && Avx2.IsSupported && Fma.IsSupported) yield return new object[] { 8 };
        yield return new object[] { 1 };
    }

    private static T WithWidth<T>(int width, Func<T> body)
    {
        int saved = QwenImage21CpuKernels.Width;
        QwenImage21CpuKernels.Width = width;
        try { return body(); }
        finally { QwenImage21CpuKernels.Width = saved; }
    }

    // ---- the synthetic model ------------------------------------------------------------

    private sealed class Tiny : IDisposable
    {
        internal readonly int Heads, HeadDim, Dim, Channels = 16, TextDim = 48, Layers = 2, Ff = 160;
        internal const float Eps = 1e-6f;
        internal readonly Dictionary<string, float[]> W = new(StringComparer.Ordinal);
        internal readonly List<IntPtr> Memory = new();
        internal QwenImage21ForwardArgs Args;
        internal QwenImage21Block[] Blocks;
        // Block 0 keeps separate gate/up projections, block 1 the fused [gate | up] layout.
        internal readonly bool[] Fused = { false, true };

        internal Tiny(int heads, int headDim, int seed = 1)
        {
            Heads = heads; HeadDim = headDim; Dim = heads * headDim;
            var rng = new Random(seed);
            Args = new QwenImage21ForwardArgs
            {
                ImageIn = Weight("img_in", Channels, Dim, rng),
                TextIn = Weight("txt_in.in_layer", TextDim, Dim, rng),
                TextOut = Weight("txt_in.out_layer", Dim, Dim, rng),
                TimeIn = Weight("time_in", 256, Dim, rng),
                TimeOut = Weight("time_out", Dim, Dim, rng),
                Modulation = Weight("modulation", Dim, 4 * Dim, rng, 0.5f),
                NormOut = Weight("norm_out", Dim, Dim, rng, 0.5f),
                ProjOut = Weight("proj_out", Dim, Channels, rng),
                TextNorm = Gain("text_norm", TextDim, rng, 0f),
                Dim = Dim, Heads = heads, HeadDim = headDim, Channels = Channels, TextDim = TextDim,
                NumLayers = Layers, Eps = Eps,
            };
            Blocks = new QwenImage21Block[Layers];
            for (int i = 0; i < Layers; i++)
            {
                string p = $"{i}.";
                Blocks[i] = new QwenImage21Block
                {
                    Q = Weight(p + "q", Dim, Dim, rng), K = Weight(p + "k", Dim, Dim, rng),
                    V = Weight(p + "v", Dim, Dim, rng), Out = Weight(p + "out", Dim, Dim, rng),
                    Gate = Weight(p + "gate", Dim, Fused[i] ? 2 * Ff : Ff, rng),
                    Up = Fused[i] ? default : Weight(p + "up", Dim, Ff, rng),
                    Down = Weight(p + "down", Ff, Dim, rng),
                    NormQ = Gain(p + "norm_q", headDim, rng, 1f), NormK = Gain(p + "norm_k", headDim, rng, 1f),
                };
            }
        }

        internal IntPtr Copy(float[] values)
        {
            var p = (IntPtr)NativeMemory.AlignedAlloc((nuint)(values.Length * 4), 64);
            Memory.Add(p);
            Marshal.Copy(values, 0, p, values.Length);
            return p;
        }

        internal IntPtr Zeroed(int bytes)
        {
            var p = (IntPtr)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            Memory.Add(p);
            new Span<byte>((void*)p, bytes).Clear();
            return p;
        }

        internal IntPtr CopyHalf(float[] values)
        {
            var p = (IntPtr)NativeMemory.AlignedAlloc((nuint)(values.Length * 2), 64);
            Memory.Add(p);
            for (int i = 0; i < values.Length; i++) ((Half*)p)[i] = (Half)values[i];
            return p;
        }

        private QwenImage21Weight Weight(string name, int input, int output, Random rng, float gain = 1f)
        {
            var data = new float[input * output];
            float scale = gain * 1.5f / MathF.Sqrt(input);
            for (int i = 0; i < data.Length; i++) data[i] = scale * (float)(rng.NextDouble() * 2 - 1);
            W[name] = data;
            return new QwenImage21Weight { Data = Copy(data), Type = 0, Ne0 = input, Ne1 = output, Bytes = data.Length * 4L };
        }

        private IntPtr Gain(string name, int n, Random rng, float offset)
        {
            var data = new float[n];
            for (int i = 0; i < n; i++) data[i] = offset + 0.25f * (float)(rng.NextDouble() * 2 - 1);
            W[name] = data;
            return Copy(data);
        }

        public void Dispose()
        {
            foreach (var p in Memory) NativeMemory.AlignedFree((void*)p);
            Memory.Clear();
        }
    }

    /// <summary>A projection's update in reference form (values as the engine reads them).</summary>
    private sealed class LoraRef
    {
        internal float[] Down, Up, RowScale;   // down [rank, in], up [out, rank]
        internal int Rank;
    }

    /// <summary>Inputs of one prediction: an edit layout (text, a 4x6 reference, text, a 10x10 target).
    /// The conditioning (text, reference latents) is the same for every seed; the target latents
    /// follow <c>seed</c>, as successive denoising steps of one request.</summary>
    private sealed class Inputs
    {
        internal float[] Images, Text, Time, Cos, Sin;
        internal QwenImage21Segment[] Segments;
        internal int ImageSeq, TextSeq, Prefix, Total;

        internal static Inputs Create(Tiny m, float timestep, int seed = 7, bool edit = true)
        {
            var conditioning = new Random(1234);
            var rng = new Random(seed);
            int textSeq = 12;
            int[] slots = new int[textSeq];
            var shapes = new List<(int Height, int Width)>();
            if (edit)
            {
                for (int i = 3; i < 9; i++) slots[i] = 1;
                shapes.Add((4, 6));
            }
            shapes.Add((10, 10));
            var layout = QwenImage21DiT.BuildLayout(textSeq, edit ? slots : null, shapes.ToArray());
            int total = layout.Cos.Length / 64, imageSeq = shapes.Sum(s => s.Height * s.Width);
            var x = new Inputs
            {
                Segments = layout.Segments, Prefix = layout.Prefix, Total = total, ImageSeq = imageSeq, TextSeq = textSeq,
                Text = Random(textSeq * m.TextDim, conditioning, 2f),
            };
            int referenceTokens = imageSeq - 100;
            x.Images = Random(referenceTokens * m.Channels, conditioning, 1f).Concat(Random(100 * m.Channels, rng, 1f)).ToArray();
            if (m.HeadDim == 128)
            {
                x.Cos = layout.Cos;
                x.Sin = layout.Sin;
            }
            else
            {
                x.Cos = new float[total * m.HeadDim / 2];
                x.Sin = new float[x.Cos.Length];
                for (int i = 0; i < x.Cos.Length; i++)
                {
                    double angle = rng.NextDouble() * 6.3;
                    x.Cos[i] = (float)Math.Cos(angle);
                    x.Sin[i] = (float)Math.Sin(angle);
                }
            }
            x.Time = new float[512];
            for (int i = 0; i < 128; ++i)
            {
                float angle = timestep * 1000f * MathF.Exp(-MathF.Log(10000f) * i / 128);
                x.Time[i] = MathF.Cos(angle); x.Time[i + 128] = MathF.Sin(angle);
                x.Time[256 + i] = 1f;
            }
            return x;
        }

        private static float[] Random(int n, Random rng, float scale)
        {
            var v = new float[n];
            for (int i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
            return v;
        }
    }

    private static float[] Run(QwenImage21ManagedDiT engine, Tiny m, Inputs x, IntPtr adapter = default,
        QwenImage21ManagedPrefix cache = null, List<QwenImage21ForwardPath> paths = null)
    {
        var target = x.Segments[^1];
        var output = new float[(target.End - target.Start) * m.Channels];
        var path = engine.Forward(x.Images, x.ImageSeq, x.Text, x.TextSeq, x.Time, x.Cos, x.Sin, x.Segments,
            x.Prefix, x.Total, adapter, cache, output);
        paths?.Add(path);
        return output;
    }

    // ---- the reference: the native graph in double precision -------------------------------

    private static double[] Linear(float[] w, int input, int output, double[] x, int rows, LoraRef l)
    {
        var y = new double[rows * output];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < output; o++)
            {
                double sum = 0;
                for (int i = 0; i < input; i++) sum += w[o * input + i] * x[r * input + i];
                y[r * output + o] = sum;
            }
        if (l == null) return y;
        if (l.RowScale != null)
            for (int r = 0; r < rows; r++)
                for (int o = 0; o < output; o++) y[r * output + o] *= l.RowScale[o];
        if (l.Rank > 0)
            for (int r = 0; r < rows; r++)
            {
                var s = new double[l.Rank];
                for (int k = 0; k < l.Rank; k++)
                    for (int i = 0; i < input; i++) s[k] += l.Down[k * input + i] * x[r * input + i];
                for (int o = 0; o < output; o++)
                    for (int k = 0; k < l.Rank; k++) y[r * output + o] += l.Up[o * l.Rank + k] * s[k];
            }
        return y;
    }

    private static double Silu(double x) => x / (1 + Math.Exp(-x));

    private static double[] LayerNorm(double[] x, int rows, int dim)
    {
        var y = new double[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double mean = 0, variance = 0;
            for (int i = 0; i < dim; i++) mean += x[r * dim + i];
            mean /= dim;
            for (int i = 0; i < dim; i++) variance += Math.Pow(x[r * dim + i] - mean, 2);
            double scale = 1 / Math.Sqrt(variance / dim + Tiny.Eps);
            for (int i = 0; i < dim; i++) y[r * dim + i] = (x[r * dim + i] - mean) * scale;
        }
        return y;
    }

    private static float[] Reference(Tiny m, Inputs x, Dictionary<string, LoraRef> lora = null, float[] head = null)
    {
        LoraRef L(string name) => lora != null && lora.TryGetValue(name, out var l) ? l : null;
        int d = m.Dim, hd = m.HeadDim, half = hd / 2;
        var time = x.Time.Select(v => (double)v).ToArray();
        var t1 = Linear(m.W["time_in"], 256, d, time, 2, L("time_in")).Select(Silu).ToArray();
        var t2 = Linear(m.W["time_out"], d, d, t1, 2, L("time_out")).Select(Silu).ToArray();
        var mod = Linear(m.W["modulation"], d, 4 * d, t2, 2, L("modulation"));
        var normOut = Linear(m.W["norm_out"], d, d, t2.Take(d).ToArray(), 1, L("norm_out"));
        double Mod(int row, int chunk, int i) => mod[(row < x.Prefix ? 1 : 0) * 4 * d + chunk * d + i];

        // Text conditioner.
        var text = new double[x.TextSeq * m.TextDim];
        for (int r = 0; r < x.TextSeq; r++)
        {
            double ms = 0;
            for (int i = 0; i < m.TextDim; i++) ms += (double)x.Text[r * m.TextDim + i] * x.Text[r * m.TextDim + i];
            double s = 1 / Math.Sqrt(ms / m.TextDim + Tiny.Eps);
            for (int i = 0; i < m.TextDim; i++) text[r * m.TextDim + i] = x.Text[r * m.TextDim + i] * s * (1 + m.W["text_norm"][i]);
        }
        var hidden = Linear(m.W["txt_in.in_layer"], m.TextDim, d, text, x.TextSeq, L("txt_in.in_layer"))
            .Select(v => 0.5 * v * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * v * (1 + 0.044715 * v * v)))).ToArray();
        var textEmb = Linear(m.W["txt_in.out_layer"], d, d, hidden, x.TextSeq, L("txt_in.out_layer"));
        var imgEmb = Linear(m.W["img_in"], m.Channels, d, x.Images.Select(v => (double)v).ToArray(), x.ImageSeq, L("img_in"));
        int n = x.Total;
        var joint = new double[n * d];
        foreach (var s in x.Segments)
            Array.Copy(s.IsImage != 0 ? imgEmb : textEmb, s.SourceStart * d, joint, s.Start * d, (s.End - s.Start) * d);

        for (int layer = 0; layer < m.Layers; layer++)
        {
            string p = $"{layer}.";
            var h = LayerNorm(joint, n, d);
            for (int r = 0; r < n; r++) for (int i = 0; i < d; i++) h[r * d + i] *= 1 + Mod(r, 0, i);
            var q = Linear(m.W[p + "q"], d, d, h, n, L(p + "q"));
            var k = Linear(m.W[p + "k"], d, d, h, n, L(p + "k"));
            var v = Linear(m.W[p + "v"], d, d, h, n, L(p + "v"));
            foreach (var (t, gain) in new[] { (q, m.W[p + "norm_q"]), (k, m.W[p + "norm_k"]) })
                for (int r = 0; r < n; r++)
                    for (int hi = 0; hi < m.Heads; hi++)
                    {
                        int o = r * d + hi * hd;
                        double ms = 0;
                        for (int i = 0; i < hd; i++) ms += t[o + i] * t[o + i];
                        double s = 1 / Math.Sqrt(ms / hd + Tiny.Eps);
                        var rotated = new double[hd];
                        for (int i = 0; i < half; i++)
                        {
                            double e = t[o + 2 * i] * s * gain[2 * i], od = t[o + 2 * i + 1] * s * gain[2 * i + 1];
                            double c = x.Cos[r * half + i], sn = x.Sin[r * half + i];
                            rotated[i] = e * c - od * sn;
                            rotated[half + i] = od * c + e * sn;
                        }
                        Array.Copy(rotated, 0, t, o, hd);
                    }
            var attn = new double[n * d];
            foreach (var seg in x.Segments)
                for (int row = seg.Start; row < seg.End; row++)
                {
                    int keys = seg.IsImage != 0 ? seg.End : row + 1;
                    for (int hi = 0; hi < m.Heads; hi++)
                    {
                        var scores = new double[keys];
                        for (int j = 0; j < keys; j++)
                        {
                            double dot = 0;
                            for (int i = 0; i < hd; i++) dot += q[row * d + hi * hd + i] * k[j * d + hi * hd + i];
                            scores[j] = dot / Math.Sqrt(hd);
                        }
                        double max = scores.Max(), sum = scores.Sum(sv => Math.Exp(sv - max));
                        for (int j = 0; j < keys; j++)
                        {
                            double w = Math.Exp(scores[j] - max) / sum;
                            for (int i = 0; i < hd; i++) attn[row * d + hi * hd + i] += w * v[j * d + hi * hd + i];
                        }
                    }
                }
            var projected = Linear(m.W[p + "out"], d, d, attn, n, L(p + "out"));
            for (int r = 0; r < n; r++) for (int i = 0; i < d; i++) joint[r * d + i] += Math.Tanh(Mod(r, 1, i)) * projected[r * d + i];
            h = LayerNorm(joint, n, d);
            for (int r = 0; r < n; r++) for (int i = 0; i < d; i++) h[r * d + i] *= 1 + Mod(r, 2, i);
            double[] gate, up;
            if (m.Fused[layer])
            {
                var gu = Linear(m.W[p + "gate"], d, 2 * m.Ff, h, n, L(p + "gate"));
                gate = new double[n * m.Ff];
                up = new double[n * m.Ff];
                for (int r = 0; r < n; r++)
                {
                    Array.Copy(gu, r * 2 * m.Ff, gate, r * m.Ff, m.Ff);
                    Array.Copy(gu, r * 2 * m.Ff + m.Ff, up, r * m.Ff, m.Ff);
                }
            }
            else
            {
                gate = Linear(m.W[p + "gate"], d, m.Ff, h, n, L(p + "gate"));
                up = Linear(m.W[p + "up"], d, m.Ff, h, n, L(p + "up"));
            }
            var act = gate.Select((g, i) => Silu(g) * up[i]).ToArray();
            var down = Linear(m.W[p + "down"], m.Ff, d, act, n, L(p + "down"));
            for (int r = 0; r < n; r++) for (int i = 0; i < d; i++) joint[r * d + i] += Math.Tanh(Mod(r, 3, i)) * down[r * d + i];
        }
        int targetRows = n - x.Prefix;
        var final = LayerNorm(joint.Skip(x.Prefix * d).ToArray(), targetRows, d);
        for (int r = 0; r < targetRows; r++) for (int i = 0; i < d; i++) final[r * d + i] *= 1 + normOut[i];
        var result = head != null
            ? Linear(head, d, m.Channels, final, targetRows, null)
            : Linear(m.W["proj_out"], d, m.Channels, final, targetRows, L("proj_out"));
        return result.Select(v => (float)v).ToArray();
    }

    private static (double Cosine, double RelMax) Compare(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double dot = 0, ee = 0, aa = 0, maxDiff = 0, maxAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"non-finite output at {i}");
            dot += (double)expected[i] * actual[i]; ee += (double)expected[i] * expected[i]; aa += (double)actual[i] * actual[i];
            maxDiff = Math.Max(maxDiff, Math.Abs(expected[i] - actual[i]));
            maxAbs = Math.Max(maxAbs, Math.Abs(expected[i]));
        }
        return (dot / Math.Sqrt(ee * aa), maxDiff / maxAbs);
    }

    // ---- tests ----------------------------------------------------------------------------

    [Theory]
    [MemberData(nameof(Widths))]
    public void ForwardMatchesTheNativeGraphReference(int width)
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        var x = Inputs.Create(m, 0.73f);
        var expected = Reference(m, x);
        var actual = WithWidth(width, () => Run(engine, m, x));
        var (cosine, relMax) = Compare(expected, actual);
        Assert.True(cosine > 0.999999 && relMax < 2e-4, $"width {width}: cosine {cosine}, rel max {relMax}");
    }

    [Theory]
    [MemberData(nameof(Widths))]
    public void HeadDimensionsOffTheSimdTileUseThePortableAttention(int width)
    {
        // headDim 32 is not a multiple of the 64-lane value tile: attention runs the scalar
        // kernel while the projections keep their vector paths.
        using var m = new Tiny(heads: 3, headDim: 32);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        var x = Inputs.Create(m, 0.2f);
        var (cosine, relMax) = Compare(Reference(m, x), WithWidth(width, () => Run(engine, m, x)));
        Assert.True(cosine > 0.999999 && relMax < 2e-4, $"width {width}: cosine {cosine}, rel max {relMax}");
    }

    [Fact]
    public void TextToImageLayoutHasOnlyATextPrefix()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        var x = Inputs.Create(m, 1f, edit: false);
        Assert.Equal(2, x.Segments.Length);
        var (cosine, relMax) = Compare(Reference(m, x), Run(engine, m, x));
        Assert.True(cosine > 0.999999 && relMax < 2e-4, $"cosine {cosine}, rel max {relMax}");
    }

    [Theory]
    [MemberData(nameof(Widths))]
    public void F32PrefixCacheReproducesTheUncachedPredictionExactly(int width)
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        using var cache = new QwenImage21ManagedPrefix(QwenImage21PrefixCacheType.Auto);
        var paths = new List<QwenImage21ForwardPath>();
        WithWidth(width, () =>
        {
            int step = 0;
            foreach (float t in new[] { 1f, 0.6f, 0.25f })
            {
                // Same conditioning, a new timestep and new target latents every step.
                var x = Inputs.Create(m, t, seed: 20 + step++);
                var uncached = Run(engine, m, x);
                var viaCache = Run(engine, m, x, cache: cache, paths: paths);
                Assert.Equal(uncached, viaCache);
            }
            return 0;
        });
        Assert.Equal(new[] { QwenImage21ForwardPath.Extract, QwenImage21ForwardPath.Cached, QwenImage21ForwardPath.Cached }, paths);
        var info = cache.Info;
        Assert.Equal(1, info.State);
        Assert.Equal(30, info.Tokens);
        Assert.Equal(0, info.KeyType);
        Assert.Equal(2L * m.Layers * 30 * m.Dim * 4, info.Bytes);
    }

    [Theory]
    [InlineData(QwenImage21PrefixCacheType.F16, 1, 1)]
    [InlineData(QwenImage21PrefixCacheType.Q8_0, 8, 8)]
    [InlineData(QwenImage21PrefixCacheType.Q8_0V, 0, 8)]
    public void CompactPrefixCachesRoundOnlyThePrefix(QwenImage21PrefixCacheType type, int keyType, int valueType)
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        using var cache = new QwenImage21ManagedPrefix(type);
        var x = Inputs.Create(m, 0.5f);
        var uncached = Run(engine, m, x);
        Assert.Equal(uncached, Run(engine, m, x, cache: cache));   // the extract step is exact
        var cached = Run(engine, m, x, cache: cache);
        var (cosine, _) = Compare(uncached, cached);
        Assert.True(cosine > 0.9999, $"{type}: cosine {cosine}");
        Assert.Equal(keyType, cache.Info.KeyType);
        Assert.Equal(valueType, cache.Info.ValueType);
    }

    [Fact]
    public void PrefixCacheRefillsWhenTheLayoutChanges()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        using var cache = new QwenImage21ManagedPrefix(QwenImage21PrefixCacheType.Auto);
        var paths = new List<QwenImage21ForwardPath>();
        var edit = Inputs.Create(m, 0.5f);
        var textOnly = Inputs.Create(m, 0.5f, edit: false);
        Run(engine, m, edit, cache: cache, paths: paths);
        Assert.Equal(Run(engine, m, textOnly), Run(engine, m, textOnly, cache: cache, paths: paths));
        Assert.Equal(new[] { QwenImage21ForwardPath.Extract, QwenImage21ForwardPath.Extract }, paths);
    }

    [Fact]
    public void PrefixCacheDeclinedByTheUserCapRecomputesThePrefix()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        using var cache = new QwenImage21ManagedPrefix(QwenImage21PrefixCacheType.Auto);
        var x = Inputs.Create(m, 0.5f);
        string saved = Environment.GetEnvironmentVariable("TS_QWEN21_PREFIX_CACHE_MAX_MIB");
        Environment.SetEnvironmentVariable("TS_QWEN21_PREFIX_CACHE_MAX_MIB", "0");
        try
        {
            var paths = new List<QwenImage21ForwardPath>();
            var expected = Run(engine, m, x);
            Assert.Equal(expected, Run(engine, m, x, cache: cache, paths: paths));
            Assert.Equal(expected, Run(engine, m, x, cache: cache, paths: paths));
            Assert.Equal(new[] { QwenImage21ForwardPath.Declined, QwenImage21ForwardPath.Declined }, paths);
            Assert.Equal(2, cache.Info.State);
        }
        finally { Environment.SetEnvironmentVariable("TS_QWEN21_PREFIX_CACHE_MAX_MIB", saved); }
    }

    /// <summary>NativeMemory throws on a failed allocation: the request must recompute the
    /// prefix (the native cache's decline), not fail, and the partial storage must be freed.</summary>
    [Fact]
    public void PrefixCacheThatCannotBeAllocatedIsDeclined()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        using var cache = new QwenImage21ManagedPrefix(QwenImage21PrefixCacheType.Auto);
        var live = new List<IntPtr>();
        int calls = 0;
        // Succeed for layer 0's K and V and layer 1's K, then fail.
        cache.Allocator = bytes =>
        {
            if (++calls > 3) throw new OutOfMemoryException("test");
            var p = (IntPtr)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            live.Add(p);
            return p;
        };
        var x = Inputs.Create(m, 0.5f);
        var expected = Run(engine, m, x);
        var paths = new List<QwenImage21ForwardPath>();
        Assert.Equal(expected, Run(engine, m, x, cache: cache, paths: paths));
        Assert.Equal(expected, Run(engine, m, x, cache: cache, paths: paths));
        Assert.Equal(new[] { QwenImage21ForwardPath.Declined, QwenImage21ForwardPath.Declined }, paths);
        Assert.Equal(4, calls);   // the second step keeps the decline, it does not retry
        var info = cache.Info;
        Assert.Equal(2, info.State);
        Assert.Equal(30, info.Tokens);
        Assert.Equal(2L * m.Layers * 30 * m.Dim * 4, info.Bytes);
    }

    [Fact]
    public void DisposedTransformerRefusesToRun()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        var x = Inputs.Create(m, 0.5f);
        Run(engine, m, x);
        engine.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Run(engine, m, x));
        engine.Dispose();
    }

    // ---- LoRA ---------------------------------------------------------------------------------

    /// <summary>Descriptors laid out as QwenImage21LoraSet packs them: Q/K/V downs back to back
    /// (one stacked shrink), gate and up sharing one down (a fused LoRA's halves), F16 and F32
    /// factors, DoRA row scales with and without a low-rank term, and global adapters.</summary>
    private static (IntPtr Adapter, Dictionary<string, LoraRef> Reference) BuildAdapter(Tiny m, bool withHead, out float[] head)
    {
        var rng = new Random(99);
        var reference = new Dictionary<string, LoraRef>(StringComparer.Ordinal);
        float[] Rand(int n, float scale) => Enumerable.Range(0, n).Select(_ => scale * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] RoundHalf(float[] v) => v.Select(f => (float)(Half)f).ToArray();

        QwenImage21Lora Make(string name, int input, int output, int rank, int type, bool rowScale, IntPtr sharedDown = default,
            float[] sharedDownValues = null, float[] downValues = null)
        {
            var l = new QwenImage21Lora { Rank = rank, In = input, Out = output, Type = type };
            var r = new LoraRef { Rank = rank };
            if (rank > 0)
            {
                var down = sharedDownValues ?? downValues ?? Rand(rank * input, 0.3f);
                var up = Rand(output * rank, 0.3f);
                if (type == 1) { down = RoundHalf(down); up = RoundHalf(up); }
                l.Down = sharedDown != IntPtr.Zero ? sharedDown : downValues != null ? IntPtr.Zero : type == 1 ? m.CopyHalf(down) : m.Copy(down);
                l.Up = type == 1 ? m.CopyHalf(up) : m.Copy(up);
                (r.Down, r.Up) = (down, up);
            }
            if (rowScale)
            {
                var s = Rand(output, 0.2f).Select(v => 1f + v).ToArray();
                l.RowScale = m.Copy(s);
                r.RowScale = s;
            }
            reference[name] = r;
            return l;
        }

        var blocks = (QwenImage21BlockLora*)m.Zeroed(sizeof(QwenImage21BlockLora) * m.Layers);
        // Block 0 (separate gate/up): Q/K/V F16 downs stacked in one allocation.
        {
            int d = m.Dim;
            var downs = new[] { Rand(16 * d, 0.3f), Rand(8 * d, 0.3f), Rand(16 * d, 0.3f) }.Select(RoundHalf).ToArray();
            IntPtr stacked = m.CopyHalf(downs.SelectMany(v => v).ToArray());
            var q = Make("0.q", d, d, 16, 1, false, stacked, downs[0]);
            var k = Make("0.k", d, d, 8, 1, true, stacked + 16 * d * 2, downs[1]);
            var v = Make("0.v", d, d, 16, 1, false, stacked + 24 * d * 2, downs[2]);
            var o = Make("0.out", d, d, 0, 0, true);
            var sharedValues = RoundHalf(Rand(8 * d, 0.3f));
            IntPtr shared = m.CopyHalf(sharedValues);
            var gate = Make("0.gate", d, m.Ff, 8, 1, false, shared, sharedValues);
            var up = Make("0.up", d, m.Ff, 8, 1, true, shared, sharedValues);
            var down = Make("0.down", m.Ff, d, 4, 0, true);
            blocks[0] = new QwenImage21BlockLora { Q = q, K = k, V = v, Out = o, Gate = gate, Up = up, Down = down };
        }
        // Block 1 (fused gate_up): K alone (F32) and one update over both MLP halves.
        blocks[1] = new QwenImage21BlockLora
        {
            K = Make("1.k", m.Dim, m.Dim, 8, 0, false),
            Gate = Make("1.gate", m.Dim, 2 * m.Ff, 8, 1, true),
        };
        var adapter = (QwenImage21Adapter*)m.Zeroed(sizeof(QwenImage21Adapter));
        *adapter = new QwenImage21Adapter
        {
            StructBytes = sizeof(QwenImage21Adapter), NumLayers = m.Layers,
            ImageIn = Make("img_in", m.Channels, m.Dim, 4, 0, false),
            TextIn = Make("txt_in.in_layer", m.TextDim, m.Dim, 8, 1, false),
            TimeIn = Make("time_in", 256, m.Dim, 4, 0, false),
            Modulation = Make("modulation", m.Dim, 4 * m.Dim, 4, 0, true),
            NormOut = Make("norm_out", m.Dim, m.Dim, 4, 1, false),
            ProjOut = withHead ? default : Make("proj_out", m.Dim, m.Channels, 4, 0, true),
            Blocks = (IntPtr)blocks,
        };
        head = null;
        if (withHead)
        {
            head = Rand(m.Channels * m.Dim, 0.1f);
            adapter->OutputHead = m.Copy(head);
            adapter->OutputHeadType = 0;
        }
        return ((IntPtr)adapter, reference);
    }

    [Theory]
    [MemberData(nameof(Widths))]
    public void LoraUpdatesMatchTheReference(int width)
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        var (adapter, reference) = BuildAdapter(m, withHead: false, out _);
        var x = Inputs.Create(m, 0.4f);
        var expected = Reference(m, x, reference);
        var actual = WithWidth(width, () => Run(engine, m, x, adapter));
        var (cosine, relMax) = Compare(expected, actual);
        Assert.True(cosine > 0.999999 && relMax < 2e-4, $"width {width}: cosine {cosine}, rel max {relMax}");
        // The adapter really changes the prediction.
        var (baseCosine, _) = Compare(expected, Reference(m, x));
        Assert.True(baseCosine < 0.999, $"the adapter barely changes the output (cosine {baseCosine})");
    }

    [Fact]
    public void PerStepOutputHeadReplacesProjOutAndTheCacheStillHolds()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        using var cache = new QwenImage21ManagedPrefix(QwenImage21PrefixCacheType.Auto);
        var (adapter, reference) = BuildAdapter(m, withHead: true, out float[] head);
        var x = Inputs.Create(m, 0.9f);
        var expected = Reference(m, x, reference, head);
        var (cosine, relMax) = Compare(expected, Run(engine, m, x, adapter, cache));
        Assert.True(cosine > 0.999999 && relMax < 2e-4, $"cosine {cosine}, rel max {relMax}");
        // A later step selects another head; the stored prefix K/V do not depend on it.
        var nextHead = head.Select(v => -0.5f * v).ToArray();
        ((QwenImage21Adapter*)adapter)->OutputHead = m.Copy(nextHead);
        Assert.Equal(Run(engine, m, x, adapter), Run(engine, m, x, adapter, cache));
        (cosine, relMax) = Compare(Reference(m, x, reference, nextHead), Run(engine, m, x, adapter, cache));
        Assert.True(cosine > 0.999999 && relMax < 2e-4, $"cosine {cosine}, rel max {relMax}");
    }

    /// <summary>ReleaseScratch (called before the VAE decode and after every request) also drops
    /// the F32 LoRA factors; the next forward rebuilds them from the descriptor and the prefix
    /// cache, bound to the descriptor rather than to the F32 copy, stays valid.</summary>
    [Fact]
    public void ReleasedLoraFactorsAreRebuiltAndTheCacheStillHolds()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        using var cache = new QwenImage21ManagedPrefix(QwenImage21PrefixCacheType.Auto);
        var (adapter, _) = BuildAdapter(m, withHead: false, out _);
        var x = Inputs.Create(m, 0.7f);
        var paths = new List<QwenImage21ForwardPath>();
        var expected = Run(engine, m, x, adapter, cache, paths);
        engine.ReleaseScratch();
        Assert.Equal(expected, Run(engine, m, x, adapter, cache, paths));
        engine.ReleaseScratch();
        Assert.Equal(expected, Run(engine, m, x, adapter));
        Assert.Equal(new[] { QwenImage21ForwardPath.Extract, QwenImage21ForwardPath.Cached }, paths);
    }

    [Fact]
    public void AnOutputHeadWithAProjOutUpdateIsRefused()
    {
        using var m = new Tiny(heads: 2, headDim: 128);
        using var engine = new QwenImage21ManagedDiT(m.Args, m.Blocks);
        var (adapter, _) = BuildAdapter(m, withHead: false, out _);
        ((QwenImage21Adapter*)adapter)->OutputHead = m.Copy(new float[m.Channels * m.Dim]);
        var error = Assert.Throws<ArgumentException>(() => Run(engine, m, Inputs.Create(m, 0.5f), adapter));
        Assert.Contains("output head", error.Message);
    }

    // ---- kernels --------------------------------------------------------------------------------

    [Theory]
    [MemberData(nameof(Widths))]
    public void FloatProductsMatchAcrossWidths(int width)
    {
        var rng = new Random(3);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        // Odd sizes: row, column and k tails of every tile shape, and a k over two depth blocks.
        foreach (var (mRows, n, k) in new[] { (1, 1, 1), (7, 13, 37), (33, 70, 130), (5, 150, 64), (13, 9, 2100) })
        {
            float[] a = Rand(mRows * k), b = Rand(n * k), c = new float[mRows * n];
            fixed (float* pa = a, pb = b, pc = c)
            {
                nint aa = (nint)pa, ba = (nint)pb, ca = (nint)pc;
                WithWidth(width, () => { QwenImage21CpuKernels.GemmNT((float*)aa, k, (float*)ba, k, (float*)ca, n, mRows, n, k); return 0; });
            }
            for (int i = 0; i < mRows; i++)
                for (int j = 0; j < n; j++)
                {
                    double expected = 0;
                    for (int t = 0; t < k; t++) expected += (double)a[i * k + t] * b[j * k + t];
                    Assert.True(Math.Abs(expected - c[i * n + j]) < 1e-4 * Math.Sqrt(k), $"NT {mRows}x{n}x{k} [{i},{j}]");
                }
            // Expand: c += a[m, r] * bt[r, n] with a column stride wider than n.
            int ldc = n + 3;
            float[] bt = Rand(k * n), acc = Rand(mRows * ldc), start = (float[])acc.Clone();
            var panels = new float[QwenImage21CpuKernels.PanelFloats(k, n)];
            fixed (float* pa = a, pb = bt, pc = acc, pp = panels)
            {
                nint aa = (nint)pa, ca = (nint)pc, pan = (nint)pp;
                QwenImage21CpuKernels.PackPanels(pb, n, k, n, pp);
                WithWidth(width, () => { QwenImage21CpuKernels.GemmNNAccumulate((float*)aa, k, (float*)pan, (float*)ca, ldc, mRows, n, k); return 0; });
            }
            for (int i = 0; i < mRows; i++)
                for (int j = 0; j < ldc; j++)
                {
                    double expected = start[i * ldc + j];
                    if (j < n) for (int t = 0; t < k; t++) expected += (double)a[i * k + t] * bt[t * n + j];
                    Assert.True(Math.Abs(expected - acc[i * ldc + j]) < 1e-4 * Math.Sqrt(k), $"NN {mRows}x{n}x{k} [{i},{j}]");
                }
        }
    }

    /// <summary>Random blocks of every type the vectorized dequantizers handle: the values must be
    /// bit-identical to <see cref="ManagedQuantizedOps"/>'s scalar ggml ports.</summary>
    [Theory]
    [MemberData(nameof(Widths))]
    public void VectorDequantizationIsBitIdenticalToTheScalarDequantizers(int width)
    {
        var rng = new Random(17);
        foreach (var (type, blockBytes, blockValues) in new[]
        {
            (GgmlTensorType.Q4_K, 144, 256), (GgmlTensorType.Q6_K, 210, 256), (GgmlTensorType.Q8_0, 34, 32),
            (GgmlTensorType.BF16, 2, 1), (GgmlTensorType.F16, 2, 1), (GgmlTensorType.F32, 4, 1),
        })
        {
            int blocks = 3 * 256 / blockValues, n = blocks * blockValues;
            var raw = new byte[blocks * blockBytes];
            rng.NextBytes(raw);
            // Finite scales: F16 d (and dmin) fields get small normal values.
            void Half(int offset) => BitConverter.TryWriteBytes(raw.AsSpan(offset), (Half)(0.001f + 0.05f * (float)rng.NextDouble()));
            for (int b = 0; b < blocks; b++)
                switch (type)
                {
                    case GgmlTensorType.Q4_K: Half(b * 144); Half(b * 144 + 2); break;
                    case GgmlTensorType.Q6_K: Half(b * 210 + 208); break;
                    case GgmlTensorType.Q8_0: Half(b * 34); break;
                    case GgmlTensorType.F16: Half(b * 2); break;
                    case GgmlTensorType.BF16: BitConverter.TryWriteBytes(raw.AsSpan(b * 2), (ushort)(BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() * 4 - 2)) >> 16)); break;
                    case GgmlTensorType.F32: BitConverter.TryWriteBytes(raw.AsSpan(b * 4), (float)(rng.NextDouble() * 4 - 2)); break;
                }
            var expected = new float[n];
            var actual = new float[n];
            fixed (byte* src = raw)
            fixed (float* e = expected, a = actual)
            {
                ManagedQuantizedOps.DequantizeRowToFloat32((int)type, (IntPtr)src, e, n);
                QwenImage21CpuDequant.Row(width, (int)type, src, a, n);
            }
            for (int i = 0; i < n; i++)
                Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                    $"{type} width {width} [{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    public static IEnumerable<object[]> WidthsAndValueLayouts() =>
        Widths().SelectMany(w => new[] { new object[] { w[0], false }, new object[] { w[0], true } });

    [Theory]
    [MemberData(nameof(WidthsAndValueLayouts))]
    public void SegmentedAttentionMatchesDirectSoftmax(int width, bool headMajorValues)
    {
        // A causal text run, an image segment and a target whose lengths straddle the
        // 64-query and 64-key tiles; values read in place or from the per-head copy.
        const int heads = 2, hd = 128, dim = heads * hd;
        var segments = new[]
        {
            new QwenImage21Segment { Start = 0, End = 70, IsImage = 0 },
            new QwenImage21Segment { Start = 70, End = 75, IsImage = 1 },
            new QwenImage21Segment { Start = 75, End = 206, IsImage = 1 },
        };
        int total = 206;
        var rng = new Random(5);
        float[] Rand(int n, float s) => Enumerable.Range(0, n).Select(_ => s * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        // Large logits: most probabilities of a row underflow and are flushed to zero.
        float[] q = Rand(total * dim, 4f), k = Rand(total * dim, 4f), v = Rand(total * dim, 1f), o = new float[total * dim];
        var kt = new float[heads * hd * QwenImage21CpuKernels.KeyStride(total)];
        var vh = new float[total * dim];
        float scale = 1f / MathF.Sqrt(hd);
        fixed (float* pq = q, pk = k, pv = v, po = o, pt = kt, ph = vh)
        {
            nint qa = (nint)pq, ka = (nint)pk, va = (nint)pv, oa = (nint)po, ta = (nint)pt, ha = (nint)ph;
            WithWidth(width, () =>
            {
                QwenImage21CpuKernels.TransposeKeys((float*)ka, dim, total, heads, hd, (float*)ta);
                if (headMajorValues)
                {
                    QwenImage21CpuKernels.GatherValues((float*)va, dim, total, heads, hd, (float*)ha);
                    QwenImage21CpuKernels.Attention((float*)qa, (float*)ta, total, (float*)ha, hd, (long)total * hd, (float*)oa,
                        dim, heads, hd, segments, 0, 0, scale);
                }
                else
                    QwenImage21CpuKernels.Attention((float*)qa, (float*)ta, total, (float*)va, dim, hd, (float*)oa,
                        dim, heads, hd, segments, 0, 0, scale);
                return 0;
            });
        }
        double worst = 0;
        foreach (var seg in segments)
            for (int row = seg.Start; row < seg.End; row++)
                for (int h = 0; h < heads; h++)
                {
                    int keys = seg.IsImage != 0 ? seg.End : row + 1;
                    var s = new double[keys];
                    for (int j = 0; j < keys; j++)
                    {
                        double dot = 0;
                        for (int i = 0; i < hd; i++) dot += (double)q[row * dim + h * hd + i] * k[j * dim + h * hd + i];
                        s[j] = dot * scale;
                    }
                    double max = s.Max(), sum = s.Sum(x => Math.Exp(x - max));
                    for (int i = 0; i < hd; i++)
                    {
                        double expected = 0;
                        for (int j = 0; j < keys; j++) expected += Math.Exp(s[j] - max) / sum * v[j * dim + h * hd + i];
                        worst = Math.Max(worst, Math.Abs(expected - o[row * dim + h * hd + i]));
                    }
                }
        Assert.True(worst < 2e-5, $"width {width}: max abs error {worst}");
    }

    [Theory]
    [MemberData(nameof(Widths))]
    public void SwiGluMatchesTheFormula(int width)
    {
        const int rows = 3, ff = 37;
        var rng = new Random(11);
        var gu = Enumerable.Range(0, rows * 2 * ff).Select(_ => 8f * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var input = (float[])gu.Clone();
        fixed (float* p = gu)
        {
            nint a = (nint)p;
            WithWidth(width, () => { QwenImage21CpuKernels.SwiGlu((float*)a, 2 * ff, rows, ff); return 0; });
        }
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < ff; i++)
            {
                double g = input[r * 2 * ff + i], u = input[r * 2 * ff + ff + i];
                Assert.True(Math.Abs(g / (1 + Math.Exp(-g)) * u - gu[r * 2 * ff + i]) < 1e-5 * (1 + Math.Abs(g * u)));
                Assert.Equal(input[r * 2 * ff + ff + i], gu[r * 2 * ff + ff + i]);
            }
    }

    [Fact]
    public void GeluFp16TableModeRoundsLikeGgmlCpu()
    {
        Assert.Equal(0f, QwenImage21CpuKernels.Gelu(-12f, fp16Table: true));
        Assert.Equal(12f, QwenImage21CpuKernels.Gelu(12f, fp16Table: true));
        float exact = QwenImage21CpuKernels.Gelu(0.3f, fp16Table: false);
        float table = QwenImage21CpuKernels.Gelu(0.3f, fp16Table: true);
        Assert.Equal((float)(Half)table, table);
        Assert.True(Math.Abs(exact - table) < 1e-3f);
    }

    // ---- the pure-C# route through the model classes -----------------------------------------------

    [Fact]
    public void ManagedPrefixCacheReportsAndReleasesWithoutNativeCalls()
    {
        // Counted rather than left to throw: from a repo checkout GgmlOps is always loadable
        // (the resolver walks up to the repo root), so a stray native call would succeed.
        var cache = QwenImage21DiT.CreatePrefixCache(new float[4], null, null, null, "f16");
        cache.Managed = new QwenImage21ManagedPrefix(cache.Type);
        Assert.Equal(0, cache.Info.State);
        cache.Dispose();
        cache.Dispose();
        Assert.Equal(0, cache.NativeCalls);
        var unused = QwenImage21DiT.CreatePrefixCache(new float[4], null, null, null, null);
        Assert.Equal(default, unused.Info);
        unused.Dispose();
        Assert.Equal(0, unused.NativeCalls);
    }

    [Fact]
    public void CpuBackendRefusesTensorParallelismByName()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ts-qi21-cpu-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        bool preferManaged = NativeDequant.PreferManaged;
        try
        {
            string path = Path.Combine(dir, "qwen-image-2.1-header.gguf");
            WriteHeaderOnlyGguf(path, new[] { ("txt_in.text_norm.weight", new ulong[] { 4096 }) });
            var error = Assert.Throws<ModelLoadRefusedException>(() => new QwenImageModel(path, BackendType.Cpu, tpDegree: 2));
            Assert.Contains("--tp 2", error.Message);
            Assert.Contains("cpu backend", error.Message);
        }
        finally
        {
            NativeDequant.PreferManaged = preferManaged;
            try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
        }
    }

    /// <summary>The real checkpoint on the cpu backend (64x64, 12 random text rows): finite
    /// velocities from both matmul modes, the prefix cache's extract and cached steps equal
    /// to the uncached prediction bit for bit without a native prefix-cache call, and a
    /// disposed transformer that refuses to predict (instead of reaching the native graph).</summary>
    [ModelFact("TENSORSHARP_QWEN21_DIT")]
    public void RealTransformerRunsOnTheCpuBackendWithAnExactPrefixCache()
    {
        bool preferManaged = NativeDequant.PreferManaged, f32 = QwenImage21ManagedDiT.F32Matmul;
        try
        {
            var dit = new QwenImage21DiT(Environment.GetEnvironmentVariable("TENSORSHARP_QWEN21_DIT")!, BackendType.Cpu);
            var rng = new Random(4);
            var text = Enumerable.Range(0, 12 * 4096).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            var latents = Enumerable.Range(0, 16 * 64).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            try
            {
                Assert.True(dit.IsManaged);
                foreach (bool mode in new[] { true, false })
                {
                    QwenImage21ManagedDiT.F32Matmul = mode;
                    var full = dit.Predict(latents, 4, 4, text, 12, 0.5f);
                    Assert.Contains(full, v => v != 0f);
                    var cache = QwenImage21DiT.CreatePrefixCache(text, null, null, "1", null);
                    Assert.Equal(full, dit.Predict(latents, 4, 4, text, 12, 0.5f, prefixCache: cache));
                    Assert.Equal(QwenImage21ForwardPath.Extract, cache.LastPath);
                    Assert.Equal(full, dit.Predict(latents, 4, 4, text, 12, 0.5f, prefixCache: cache));
                    Assert.Equal(QwenImage21ForwardPath.Cached, cache.LastPath);
                    Assert.Equal(1, cache.Info.State);
                    Assert.Equal(12, cache.Info.Tokens);
                    cache.Dispose();
                    Assert.Equal(0, cache.NativeCalls);
                }
            }
            finally { dit.Dispose(); }
            Assert.True(dit.IsManaged);
            Assert.Throws<ObjectDisposedException>(() => dit.Predict(latents, 4, 4, text, 12, 0.5f));
        }
        finally
        {
            QwenImage21ManagedDiT.F32Matmul = f32;
            NativeDequant.PreferManaged = preferManaged;
        }
    }

    private static void WriteHeaderOnlyGguf(string path, IEnumerable<(string Name, ulong[] Shape)> tensors)
    {
        var list = tensors.ToList();
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write((ulong)list.Count);
        writer.Write(0UL);
        foreach (var (name, shape) in list)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(name);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
            writer.Write((uint)shape.Length);
            foreach (ulong d in shape) writer.Write(d);
            writer.Write((uint)GgmlTensorType.F32);
            writer.Write(0UL);
        }
    }
}
