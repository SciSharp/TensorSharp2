// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// The Qwen-Image-2.1 VAE's memory work on the pure-C# path: feature maps recycled through
/// <see cref="VaeFeaturePool"/> as soon as they are read, in-place norms, and the decoder's
/// stage shortcut added straight from the stage input. A tiny VAE with the released
/// architecture (every block, shortcut conv, attention and resampling kind; narrow channels)
/// runs the production drivers against the drivers as they were written before (materialized
/// shortcuts, nothing released) on the same scalar primitives: bit-identical. The fast path with
/// every released map poisoned (NaN) must equal the unpooled fast path bit for bit and release
/// every map it rents.
/// </summary>
[Collection("CpuPackedGemmIsa")]
public sealed class QwenImage21VaeMemoryTests
{
    // ---- a tiny VAE with the released layout -----------------------------------------------

    private sealed class TinyVaeStore : IFloatTensorStore
    {
        private readonly Dictionary<string, (long[] Shape, float[] Data)> _tensors = new();
        private readonly Random _rng;

        // Decoder: conv1 64->16, stages 16->16, 16->16, 16->8, 8->4, 4->2 (stage 2-4 open with a
        // shortcut conv, like 1152->576, 576->288, 288->144), head 2->4. Encoder: conv1 4->4,
        // stages 4->4, 4->8, 8->16, 16->32, 32->32, head 32->128.
        internal static readonly int[] DecoderIn = { 16, 16, 16, 8, 4 }, DecoderOut = { 16, 16, 8, 4, 2 };
        internal static readonly int[] EncoderIn = { 4, 4, 8, 16, 32 }, EncoderOut = { 4, 8, 16, 32, 32 };

        public TinyVaeStore(int seed)
        {
            _rng = new Random(seed);
            Conv("conv2", 64, 64, 1, causal: true);
            Conv("decoder.conv1", DecoderIn[0], 64, 3, causal: true);
            Middle("decoder", DecoderIn[0]);
            for (int s = 0; s < 5; s++)
            {
                string p = $"decoder.upsamples.{s}.upsamples";
                Residual(p + ".0", DecoderIn[s], DecoderOut[s]);
                Residual(p + ".1", DecoderOut[s], DecoderOut[s]);
                Residual(p + ".2", DecoderOut[s], DecoderOut[s]);
                if (s < 4) Conv(p + ".3.resample.1", DecoderOut[s], DecoderOut[s], 3, causal: false);
            }
            Gamma("decoder.head.0.gamma", DecoderOut[4], 4);
            Conv("decoder.head.2", 4, DecoderOut[4], 3, causal: true);

            Conv("encoder.conv1", EncoderIn[0], 4, 3, causal: true);
            for (int s = 0; s < 5; s++)
            {
                string p = $"encoder.downsamples.{s}.downsamples";
                Residual(p + ".0", EncoderIn[s], EncoderOut[s]);
                Residual(p + ".1", EncoderOut[s], EncoderOut[s]);
                if (s < 4) Conv(p + ".2.resample.1", EncoderOut[s], EncoderOut[s], 3, causal: false);
            }
            Middle("encoder", EncoderOut[4]);
            Gamma("encoder.head.0.gamma", EncoderOut[4], 4);
            Conv("encoder.head.2", 128, EncoderOut[4], 3, causal: true);
            Conv("conv1", 128, 128, 1, causal: true);
        }

        private float[] Random(long n, float scale, float offset = 0f)
        {
            var v = new float[n];
            for (long i = 0; i < n; i++) v[i] = offset + (float)(_rng.NextDouble() * 2 - 1) * scale;
            return v;
        }

        // Causal convs carry the unit time axis ([OC, IC, 1, KH, KW]); attention projections and
        // spatial resampling are plain 4-D convs, as QwenImage21VaeTensorStore presents them.
        private void Conv(string prefix, int oc, int ic, int k, bool causal)
        {
            long[] shape = causal ? new long[] { oc, ic, 1, k, k } : new long[] { oc, ic, k, k };
            _tensors[prefix + ".weight"] = (shape, Random((long)oc * ic * k * k, 1.5f / MathF.Sqrt(ic * k * k)));
            _tensors[prefix + ".bias"] = (new long[] { oc }, Random(oc, 0.1f));
        }

        private void Gamma(string name, int c, int rank) =>
            _tensors[name] = (rank == 4 ? new long[] { c, 1, 1, 1 } : new long[] { c, 1, 1 }, Random(c, 0.2f, 1f));

        private void Residual(string prefix, int ic, int oc)
        {
            Gamma(prefix + ".residual.0.gamma", ic, 4);
            Conv(prefix + ".residual.2", oc, ic, 3, causal: true);
            Gamma(prefix + ".residual.3.gamma", oc, 4);
            Conv(prefix + ".residual.6", oc, oc, 3, causal: true);
            if (ic != oc) Conv(prefix + ".shortcut", oc, ic, 1, causal: true);
        }

        private void Middle(string side, int c)
        {
            Residual(side + ".middle.0", c, c);
            Gamma(side + ".middle.1.norm.gamma", c, 3);
            Conv(side + ".middle.1.to_qkv", 3 * c, c, 1, causal: false);
            Conv(side + ".middle.1.proj", c, c, 1, causal: false);
            Residual(side + ".middle.2", c, c);
        }

        public bool HasTensor(string name) => _tensors.ContainsKey(name);
        public float[] ReadFloat32(string name) => (float[])_tensors[name].Data.Clone();
        public long[] TensorShape(string name) => _tensors.TryGetValue(name, out var t) ? t.Shape : Array.Empty<long>();
    }

    // ---- the drivers as written before the pool, on the scalar primitives -----------------

    private sealed class ReferenceVae
    {
        private readonly TinyVaeStore _w;
        public ReferenceVae(TinyVaeStore w) => _w = w;

        private Feature Conv(string prefix, Feature x, int pad, bool upsample = false, int stride = 1, bool downsamplePad = false)
        {
            long[] shape = _w.TensorShape(prefix + ".weight");
            int oc = (int)shape[0], ic = (int)shape[1], kh = (int)shape[^2], kw = (int)shape[^1];
            int kd = shape.Length == 5 ? (int)shape[2] : 1;
            float[] weight = VaeReferenceMath.LastTemporalSlice(_w.ReadFloat32(prefix + ".weight"), oc, ic, kd, kh, kw);
            var input = upsample ? Upsample(x) : x;
            return downsamplePad
                ? VaeReferenceMath.Conv2dScalar(input, weight, oc, ic, kh, kw, _w.ReadFloat32(prefix + ".bias"), stride, stride, 0, 1, 0, 1)
                : VaeReferenceMath.Conv2dScalar(input, weight, oc, ic, kh, kw, _w.ReadFloat32(prefix + ".bias"), stride, stride, pad, pad, pad, pad);
        }

        private static Feature Upsample(Feature x)
        {
            var y = new Feature(x.C, 2 * x.H, 2 * x.W);
            for (int c = 0; c < x.C; c++)
                for (int oy = 0; oy < y.H; oy++)
                    for (int ox = 0; ox < y.W; ox++)
                        y.D[(c * y.H + oy) * y.W + ox] = x.D[(c * x.H + oy / 2) * x.W + ox / 2];
            return y;
        }

        private static Feature Add(Feature a, Feature b)
        {
            for (int i = 0; i < a.D.Length; i++) a.D[i] = a.D[i] + b.D[i];
            return a;
        }

        private Feature NormSilu(string gamma, Feature x)
        {
            var t = VaeReferenceMath.RmsNormChannel(x, _w.ReadFloat32(gamma));
            VaeReferenceMath.SiluInPlace(t.D);
            return t;
        }

        private Feature Residual(string prefix, Feature x)
        {
            var residual = _w.HasTensor(prefix + ".shortcut.weight") ? Conv(prefix + ".shortcut", x, 0) : x;
            var t = NormSilu(prefix + ".residual.0.gamma", x);
            t = Conv(prefix + ".residual.2", t, 1);
            t = NormSilu(prefix + ".residual.3.gamma", t);
            t = Conv(prefix + ".residual.6", t, 1);
            return Add(t, residual);
        }

        // VaeReferenceMath.AttentionBlock's scalar branch, verbatim.
        private Feature Attention(string prefix, Feature x)
        {
            int C = x.C, H = x.H, W = x.W, hw = H * W;
            var xn = VaeReferenceMath.RmsNormChannel(x, _w.ReadFloat32(prefix + ".norm.gamma"));
            var qkv = VaeReferenceMath.Conv2dScalar(xn, _w.ReadFloat32(prefix + ".to_qkv.weight"), 3 * C, C, 1, 1,
                _w.ReadFloat32(prefix + ".to_qkv.bias"), 1, 1, 0, 0, 0, 0);
            float scale = 1f / MathF.Sqrt(C);
            var outp = new Feature(C, H, W);
            for (int i = 0; i < hw; i++)
            {
                var scores = new float[hw];
                float mx = float.NegativeInfinity;
                for (int j = 0; j < hw; j++)
                {
                    float s = 0;
                    for (int c = 0; c < C; c++) s += qkv.D[c * hw + i] * qkv.D[(C + c) * hw + j];
                    s *= scale;
                    scores[j] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0;
                for (int j = 0; j < hw; j++) { float e = MathF.Exp(scores[j] - mx); scores[j] = e; sum += e; }
                float invSum = 1f / sum;
                for (int c = 0; c < C; c++)
                {
                    float acc = 0;
                    int vbase = (2 * C + c) * hw;
                    for (int j = 0; j < hw; j++) acc += scores[j] * qkv.D[vbase + j];
                    outp.D[c * hw + i] = acc * invSum;
                }
            }
            var proj = VaeReferenceMath.Conv2dScalar(outp, _w.ReadFloat32(prefix + ".proj.weight"), C, C, 1, 1,
                _w.ReadFloat32(prefix + ".proj.bias"), 1, 1, 0, 0, 0, 0);
            return Add(proj, x);
        }

        private Feature Middle(string prefix, Feature x) =>
            Residual(prefix + ".2", Attention(prefix + ".1", Residual(prefix + ".0", x)));

        public RgbImage Decode(VaeLatent latent)
        {
            int pixels = latent.Height * latent.Width;
            var x = new Feature(64, latent.Height, latent.Width);
            for (int c = 0; c < 64; c++)
                for (int i = 0; i < pixels; i++)
                    x.D[c * pixels + i] = latent.Data[c * pixels + i] * VaeReferenceMath.Qwen21Std[c] + VaeReferenceMath.Qwen21Mean[c];
            x = Conv("conv2", x, 0);
            x = Conv("decoder.conv1", x, 1);
            x = Middle("decoder.middle", x);
            for (int stage = 0; stage < 5; stage++)
            {
                string prefix = $"decoder.upsamples.{stage}.upsamples";
                var shortcut = stage < 4
                    ? VaeReferenceMath.DuplicateUp21(x, TinyVaeStore.DecoderOut[stage], stage < 3 ? 2 : 1, 2)
                    : null;
                for (int j = 0; j < 3; j++) x = Residual(prefix + $".{j}", x);
                if (stage < 4) x = Add(Conv(prefix + ".3.resample.1", x, 1, upsample: true), shortcut);
            }
            x = NormSilu("decoder.head.0.gamma", x);
            x = Conv("decoder.head.2", x, 1);
            pixels = x.H * x.W;
            var rgb = new float[3 * pixels];
            var alpha = new float[pixels];
            for (int i = 0; i < pixels; i++)
            {
                alpha[i] = Math.Clamp((x.D[3 * pixels + i] + 1f) * .5f, 0f, 1f);
                for (int c = 0; c < 3; c++) rgb[i * 3 + c] = Math.Clamp((x.D[c * pixels + i] + 1f) * .5f, 0f, 1f);
            }
            return new RgbImage(x.W, x.H, rgb, alpha);
        }

        public float[] Encode(RgbImage image)
        {
            int hw = image.Width * image.Height;
            var x = new Feature(4, image.Height, image.Width);
            for (int i = 0; i < hw; i++)
            {
                for (int c = 0; c < 3; c++) x.D[c * hw + i] = image.Pixels[i * 3 + c] * 2f - 1f;
                x.D[3 * hw + i] = image.Alpha == null ? 1f : image.Alpha[i] * 2f - 1f;
            }
            x = Conv("encoder.conv1", x, 1);
            for (int stage = 0; stage < 5; stage++)
            {
                string prefix = $"encoder.downsamples.{stage}.downsamples";
                var shortcut = VaeReferenceMath.AverageDown21(x, TinyVaeStore.EncoderOut[stage],
                    stage is >= 1 and <= 3 ? 2 : 1, stage < 4 ? 2 : 1);
                x = Residual(prefix + ".0", x);
                x = Residual(prefix + ".1", x);
                if (stage < 4) x = Conv(prefix + ".2.resample.1", x, 0, stride: 2, downsamplePad: true);
                x = Add(x, shortcut);
            }
            x = Middle("encoder.middle", x);
            x = NormSilu("encoder.head.0.gamma", x);
            x = Conv("encoder.head.2", x, 1);
            x = Conv("conv1", x, 0);
            int pixels = x.H * x.W;
            var latent = new float[64 * pixels];
            for (int c = 0; c < 64; c++)
                for (int i = 0; i < pixels; i++)
                    latent[c * pixels + i] = (x.D[c * pixels + i] - VaeReferenceMath.Qwen21Mean[c]) / VaeReferenceMath.Qwen21Std[c];
            return latent;
        }
    }

    // ---- helpers -----------------------------------------------------------------------------

    private static VaeLatent RandomLatent(int seed, int height, int width)
    {
        var rng = new Random(seed);
        var data = new float[64 * height * width];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 4 - 2);
        return new VaeLatent(64, height, width, data);
    }

    private static RgbImage RandomImage(int seed, int width, int height)
    {
        var rng = new Random(seed);
        var pixels = new float[width * height * 3];
        var alpha = new float[width * height];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = (float)rng.NextDouble();
        for (int i = 0; i < alpha.Length; i++) alpha[i] = (float)rng.NextDouble();
        return new RgbImage(width, height, pixels, alpha);
    }

    /// <summary>Runs <paramref name="body"/> with the VAE switches set, restoring them after.</summary>
    private static T With<T>(bool scalar, bool pool, bool poison, Func<T> body)
    {
        bool gpu = VaeReferenceMath.UseGpuConv, savedScalar = VaeReferenceMath.UseScalarCpu;
        bool enabled = VaeFeaturePool.Enabled, savedPoison = VaeFeaturePool.PoisonReturned;
        try
        {
            VaeReferenceMath.UseGpuConv = false;
            VaeReferenceMath.UseScalarCpu = scalar;
            VaeFeaturePool.Enabled = pool;
            VaeFeaturePool.PoisonReturned = poison;
            return body();
        }
        finally
        {
            VaeReferenceMath.UseGpuConv = gpu;
            VaeReferenceMath.UseScalarCpu = savedScalar;
            VaeFeaturePool.Enabled = enabled;
            VaeFeaturePool.PoisonReturned = savedPoison;
        }
    }

    private static void AssertClose(float[] expected, float[] actual, double relL2Limit, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sq = 0, ref2 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}: non-finite at {i}");
            double d = actual[i] - expected[i];
            sq += d * d;
            ref2 += (double)expected[i] * expected[i];
        }
        double relL2 = Math.Sqrt(sq / Math.Max(ref2, 1e-300));
        Assert.True(relL2 <= relL2Limit, $"{what}: relL2 {relL2:E3} > {relL2Limit:E1}");
    }

    // ---- the drivers -------------------------------------------------------------------------

    [Fact]
    public void ScalarDecodeMatchesTheDriverAsOriginallyWritten()
    {
        var store = new TinyVaeStore(11);
        var latent = RandomLatent(3, 3, 4);
        var expected = new ReferenceVae(store).Decode(latent);
        var actual = With(scalar: true, pool: true, poison: true, () => VaeReferenceMath.Decode21(VaeWeights.Load(store), latent));
        Assert.Equal(64, actual.Width);
        Assert.Equal(48, actual.Height);
        Assert.Equal(expected.Pixels, actual.Pixels);
        Assert.Equal(expected.Alpha, actual.Alpha);
    }

    [Fact]
    public void ScalarEncodeMatchesTheDriverAsOriginallyWritten()
    {
        var store = new TinyVaeStore(12);
        var image = RandomImage(4, 64, 48);
        float[] expected = new ReferenceVae(store).Encode(image);
        var actual = With(scalar: true, pool: true, poison: true, () => VaeReferenceMath.Encode21(VaeWeights.Load(store), image));
        Assert.Equal(3, actual.Height);
        Assert.Equal(4, actual.Width);
        Assert.Equal(expected, actual.Data);
    }

    [Fact]
    public void PooledFastDecodeIsBitIdenticalAndReleasesEveryMap()
    {
        var store = new TinyVaeStore(13);
        var latent = RandomLatent(5, 3, 4);
        var unpooled = With(scalar: false, pool: false, poison: false, () => VaeReferenceMath.Decode21(VaeWeights.Load(store), latent));
        VaeFeaturePool.LastCompleted = null;
        var pooled = With(scalar: false, pool: true, poison: true, () => VaeReferenceMath.Decode21(VaeWeights.Load(store), latent));
        Assert.Equal(unpooled.Pixels, pooled.Pixels);
        Assert.Equal(unpooled.Alpha, pooled.Alpha);
        var stats = VaeFeaturePool.LastCompleted;
        Assert.NotNull(stats);
        Assert.True(stats.Hits > 0, stats.Describe());
        Assert.Equal(0, stats.LiveBytesAtDispose);
        // The fast path against the scalar oracle: GEMM and vectorized-exp rounding only.
        var reference = new ReferenceVae(store).Decode(latent);
        AssertClose(reference.Pixels, pooled.Pixels, 1e-4, "fast decode vs reference");
    }

    [Fact]
    public void PooledFastEncodeIsBitIdenticalAndReleasesEveryMap()
    {
        var store = new TinyVaeStore(14);
        var image = RandomImage(6, 64, 48);
        var unpooled = With(scalar: false, pool: false, poison: false, () => VaeReferenceMath.Encode21(VaeWeights.Load(store), image));
        VaeFeaturePool.LastCompleted = null;
        var pooled = With(scalar: false, pool: true, poison: true, () => VaeReferenceMath.Encode21(VaeWeights.Load(store), image));
        Assert.Equal(unpooled.Data, pooled.Data);
        var stats = VaeFeaturePool.LastCompleted;
        Assert.NotNull(stats);
        Assert.True(stats.Hits > 0, stats.Describe());
        Assert.Equal(0, stats.LiveBytesAtDispose);
        AssertClose(new ReferenceVae(store).Encode(image), pooled.Data, 1e-4, "fast encode vs reference");
    }

    [Fact]
    public void NoPoolIsActiveOffTheManagedFastPath()
    {
        var store = new TinyVaeStore(15);
        var latent = RandomLatent(7, 3, 4);
        VaeFeaturePool.LastCompleted = null;
        With(scalar: true, pool: true, poison: false, () => VaeReferenceMath.Decode21(VaeWeights.Load(store), latent));
        Assert.Null(VaeFeaturePool.LastCompleted);
        Assert.Null(VaeFeaturePool.Current);
    }

    // ---- the fused shortcut and the in-place norm --------------------------------------------

    [Theory]
    [InlineData(16, 16, 2)]   // decoder stages 0-1: 1152 -> 1152, time factor 2 (8 repeats)
    [InlineData(16, 8, 2)]    // stage 2: 1152 -> 576 (4 repeats)
    [InlineData(8, 4, 1)]     // stage 3: 576 -> 288, time factor 1 (2 repeats)
    public void AddDuplicateUpEqualsAddingTheMaterializedShortcut(int inChannels, int outChannels, int timeFactor)
    {
        var rng = new Random(inChannels * 31 + outChannels);
        var x = new Feature(inChannels, 5, 7);
        for (int i = 0; i < x.D.Length; i++) x.D[i] = (float)(rng.NextDouble() * 6 - 3);
        var a = new Feature(outChannels, 10, 14);
        for (int i = 0; i < a.D.Length; i++) a.D[i] = (float)(rng.NextDouble() * 6 - 3);
        var shortcut = VaeReferenceMath.DuplicateUp21(x, outChannels, timeFactor, 2);
        var expected = new float[a.D.Length];
        for (int i = 0; i < expected.Length; i++) expected[i] = a.D[i] + shortcut.D[i];
        var result = VaeReferenceMath.AddDuplicateUp21InPlace(a, x, timeFactor, 2);
        Assert.Same(a, result);
        Assert.Equal(expected, a.D);
        Assert.Throws<ArgumentException>(() => VaeReferenceMath.AddDuplicateUp21InPlace(new Feature(outChannels, 9, 14), x, timeFactor, 2));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void InPlaceChannelNormIsBitIdentical(bool silu)
    {
        var rng = new Random(silu ? 1 : 2);
        var x = new Feature(37, 13, 91);   // a partial last tile and a scalar tail
        for (int i = 0; i < x.D.Length; i++) x.D[i] = (float)(rng.NextDouble() * 40 - 20);
        var gamma = new float[37];
        for (int i = 0; i < gamma.Length; i++) gamma[i] = (float)(rng.NextDouble() + .5);
        var expected = VaeReferenceMath.RmsNormChannelFast(x, gamma, silu).D;
        var inPlace = VaeReferenceMath.RmsNormChannelFast(x, gamma, silu, inPlace: true);
        Assert.Same(x, inPlace);
        Assert.Equal(expected, x.D);
    }

    // ---- the pool itself ------------------------------------------------------------------------

    [Fact]
    public void PoolRecyclesExactLengthsAndDropsOthersOnAMiss()
    {
        bool enabled = VaeFeaturePool.Enabled;
        try
        {
            VaeFeaturePool.Enabled = true;
            Assert.Null(VaeFeaturePool.Enter(active: false));
            using (var pool = VaeFeaturePool.Enter(active: true))
            {
                Assert.Same(pool, VaeFeaturePool.Current);
                var a = Feature.Uninitialized(2, 3, 4);
                var b = Feature.Uninitialized(2, 3, 4);
                float[] aData = a.D, bData = b.D;
                Assert.NotSame(aData, bData);
                VaeFeaturePool.Release(a);
                Assert.Null(a.D);   // detached: a stale local no longer keeps the buffer alive
                Assert.Throws<InvalidOperationException>(() => VaeFeaturePool.Release(a));
                var c = Feature.Uninitialized(4, 3, 2);   // same length: reused
                Assert.Same(aData, c.D);
                VaeFeaturePool.Release(b);
                Assert.Throws<InvalidOperationException>(() => pool.Return(bData));
                var d = Feature.Uninitialized(5, 5, 5);   // a miss: b's length is dropped
                var e = Feature.Uninitialized(2, 3, 4);
                Assert.NotSame(bData, d.D);
                Assert.NotSame(bData, e.D);
                Assert.Equal(5, pool.Rents);
                Assert.Equal(1, pool.Hits);
                Assert.Equal((24 + 24 + 125) * sizeof(float), pool.PeakLiveBytes);   // c, d, e
                // A map the pool did not rent is adopted.
                var foreign = new Feature(1, 1, 7);
                float[] foreignData = foreign.D;
                VaeFeaturePool.Release(foreign);
                Assert.Same(foreignData, Feature.Uninitialized(7, 1, 1).D);
                using (var inner = VaeFeaturePool.Enter(active: true))
                    Assert.Same(inner, VaeFeaturePool.Current);
                Assert.Same(pool, VaeFeaturePool.Current);
            }
            Assert.Null(VaeFeaturePool.Current);
            var unpooled = new Feature(1, 1, 1);
            VaeFeaturePool.Release(unpooled);   // no pool: a no-op
            Assert.NotNull(unpooled.D);
            VaeFeaturePool.Enabled = false;
            Assert.Null(VaeFeaturePool.Enter(active: true));
        }
        finally { VaeFeaturePool.Enabled = enabled; }
    }

    [Fact]
    public void PoisonMarksReleasedMaps()
    {
        bool poison = VaeFeaturePool.PoisonReturned, enabled = VaeFeaturePool.Enabled;
        try
        {
            VaeFeaturePool.PoisonReturned = true;
            VaeFeaturePool.Enabled = true;
            using var pool = VaeFeaturePool.Enter(active: true);
            var a = Feature.Uninitialized(1, 2, 3);
            float[] data = a.D;
            Array.Fill(data, 1f);
            VaeFeaturePool.Release(a);
            Assert.All(data, v => Assert.True(float.IsNaN(v)));
        }
        finally
        {
            VaeFeaturePool.PoisonReturned = poison;
            VaeFeaturePool.Enabled = enabled;
        }
    }

    // ---- the real VAE: the decoder's live set is what the memory estimate assumes ------------

    /// <summary>A 256x256 decode with the released VAE: its largest live set of feature maps is
    /// exactly QwenImage21CpuMemory.DecodeFeatureBytesPerPixel per output pixel (stage 4's first
    /// residual block), and every map is released.</summary>
    [ModelFact("TENSORSHARP_QWEN21_DIT")]
    public void RealDecoderPeakLiveSetMatchesTheMemoryEstimate()
    {
        string directory = Path.GetDirectoryName(Environment.GetEnvironmentVariable("TENSORSHARP_QWEN21_DIT"))!;
        string path = Path.Combine(directory, "qwen_image_2.1_vae_bf16.safetensors");
        if (!File.Exists(path)) return;
        using var file = SafetensorsModel.Open(path);
        var weights = VaeWeights.Load(new QwenImage21VaeTensorStore(file));
        VaeFeaturePool.LastCompleted = null;
        var image = With(scalar: false, pool: true, poison: false, () => VaeReferenceMath.Decode21(weights, RandomLatent(9, 16, 16)));
        Assert.Equal(256, image.Width);
        var stats = VaeFeaturePool.LastCompleted;
        Assert.NotNull(stats);
        Assert.Equal(QwenImage21CpuMemory.DecodeFeatureBytesPerPixel * 256 * 256, stats.PeakLiveBytes);
        Assert.Equal(0, stats.LiveBytesAtDispose);
    }
}
