// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 audio VAE decoder: the BigVGAN-style vocoder of
// TSGgml_MiniMaxH3AudioVaeDecode in managed code, so BackendType.Cpu (the 100%
// pure-C# backend) can turn audio latents into samples with no ggml.
//
// Everything here works on plain [channel][time] float arrays rather than
// Tensors: the ops are 1-D convolutions, a transposed convolution, replicate
// padding and a per-channel activation, none of which the tensor layer has, and
// the channel counts are small enough that a direct loop is the clearest and
// fastest expression of them.
//
// Conventions that MUST match ggml, because the weights were laid out for it:
//   - conv:  out[oc][t] = sum_ic sum_kk x[ic][t*s - p + kk*d] * w[oc][ic][kk]
//            (cross-correlation, zero outside), which is ggml_conv_1d.
//   - convT: out[oc][j*s + kk] += x[ic][j] * w[ic][oc][kk], no padding,
//            which is ggml_conv_transpose_1d.
//   - the alias-free activation upsamples by zero-stuffing and convolving with a
//     REVERSED filter, so the caller hands us the up filter already reversed.
using System;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed MiniMax-H3 audio VAE decoder (no ggml).</summary>
    internal sealed class MiniMaxH3DirectAudioVae
    {
        internal sealed class Conv
        {
            public float[] W;          // [oc][ic][k], or [ic][oc][k] when transposed
            public float[] B;
            public int K, Ic, Oc, Stride, Pad, Dilation;
            public bool Transposed;
        }

        internal sealed class Act
        {
            public float[] Alpha;      // already exponentiated
            public float[] Beta;       // already exponentiated, with the zero guard folded in
            public float[] UpFilter;   // already reversed
            public float[] DownFilter;
            public int Channels, Kernel;
        }

        private readonly Conv _decInProj, _convPre, _convPost;
        private readonly Conv[] _convs;
        private readonly Act[] _acts;
        private readonly int[] _rates;
        private readonly int _ampsPerStage;

        public MiniMaxH3DirectAudioVae(Conv decInProj, Conv convPre, Conv convPost,
                                       Conv[] convs, Act[] acts, int[] rates, int ampsPerStage)
        {
            _decInProj = decInProj; _convPre = convPre; _convPost = convPost;
            _convs = convs; _acts = acts; _rates = rates; _ampsPerStage = ampsPerStage;
        }

        // ---- primitives ------------------------------------------------------

        /// <summary>ggml_conv_1d over x[ic][T].</summary>
        private static float[] Conv1d(float[] x, int ic, int t, Conv c, out int outT)
        {
            int k = c.K, oc = c.Oc, s = c.Stride, p = c.Pad, dl = c.Dilation;
            outT = (t + 2 * p - dl * (k - 1) - 1) / s + 1;
            if (outT <= 0) throw new InvalidOperationException("MiniMax-H3 audio VAE: empty conv output.");
            var y = new float[(long)oc * outT];
            int outLen = outT;
            DirectParallel.For(oc, o =>
            {
                long yo = (long)o * outLen;
                for (int i = 0; i < ic; i++)
                {
                    long wo = ((long)o * ic + i) * k;
                    long xo = (long)i * t;
                    for (int kk = 0; kk < k; kk++)
                    {
                        float wv = c.W[wo + kk];
                        if (wv == 0f) continue;
                        int off = kk * dl - p;
                        for (int ot = 0; ot < outLen; ot++)
                        {
                            int xi = ot * s + off;
                            if ((uint)xi < (uint)t) y[yo + ot] += wv * x[xo + xi];
                        }
                    }
                }
                if (c.B != null)
                    for (int ot = 0; ot < outLen; ot++) y[yo + ot] += c.B[o];
            });
            return y;
        }

        /// <summary>ggml_conv_transpose_1d over x[ic][T], stride s, no padding.</summary>
        private static float[] ConvTranspose1d(float[] x, int ic, int t, Conv c, int stride, out int outT)
        {
            // Torch stores a transposed kernel as [in, out, k], so c.Oc is the INPUT
            // channel count here and c.Ic is the output one - the same swap the
            // GGML path relies on for its bias length.
            int k = c.K, oc = c.Ic;
            outT = (t - 1) * stride + k;
            var y = new float[(long)oc * outT];
            int outLen = outT;
            DirectParallel.For(oc, o =>
            {
                long yo = (long)o * outLen;
                for (int i = 0; i < ic; i++)
                {
                    long wo = ((long)i * oc + o) * k;
                    long xo = (long)i * t;
                    for (int j = 0; j < t; j++)
                    {
                        float xv = x[xo + j];
                        if (xv == 0f) continue;
                        int b = j * stride;
                        for (int kk = 0; kk < k; kk++) y[yo + b + kk] += xv * c.W[wo + kk];
                    }
                }
            });
            return y;
        }

        /// <summary>Edge-replicating pad on the time axis of x[c][T].</summary>
        private static float[] ReplicatePad(float[] x, int c, int t, int left, int right)
        {
            int outT = t + left + right;
            var y = new float[(long)c * outT];
            DirectParallel.For(c, ch =>
            {
                long xo = (long)ch * t, yo = (long)ch * outT;
                for (int i = 0; i < left; i++) y[yo + i] = x[xo];
                Array.Copy(x, xo, y, yo + left, t);
                for (int i = 0; i < right; i++) y[yo + left + t + i] = x[xo + t - 1];
            });
            return y;
        }

        /// <summary>SnakeBeta in place: x + sin(alpha*x)^2 / beta, per channel.</summary>
        private static void SnakeBeta(float[] x, int c, int t, float[] alpha, float[] beta)
        {
            DirectParallel.For(c, ch =>
            {
                float a = alpha[ch], invB = 1f / beta[ch];
                long o = (long)ch * t;
                for (int i = 0; i < t; i++)
                {
                    float s = MathF.Sin(a * x[o + i]);
                    x[o + i] += s * s * invB;
                }
            });
        }

        /// <summary>Per-channel convolution with one shared filter (the resamplers).</summary>
        private static float[] FilterPerChannel(float[] x, int c, int t, float[] filter,
                                                int stride, int pad, out int outT)
        {
            int k = filter.Length;
            outT = (t + 2 * pad - (k - 1) - 1) / stride + 1;
            var y = new float[(long)c * outT];
            int outLen = outT;
            DirectParallel.For(c, ch =>
            {
                long xo = (long)ch * t, yo = (long)ch * outLen;
                for (int ot = 0; ot < outLen; ot++)
                {
                    float sum = 0f;
                    int baseIdx = ot * stride - pad;
                    for (int kk = 0; kk < k; kk++)
                    {
                        int xi = baseIdx + kk;
                        if ((uint)xi < (uint)t) sum += filter[kk] * x[xo + xi];
                    }
                    y[yo + ot] = sum;
                }
            });
            return y;
        }

        /// <summary>Alias-free activation: 2x up, snake, 2x down.</summary>
        private static float[] Activation1d(float[] x, int c, int t, Act a, out int outT)
        {
            const int ratio = 2;
            int k = a.Kernel;
            int upPad = k / ratio - 1;
            int padLeft = upPad * ratio + (k - ratio) / 2;
            int padRight = upPad * ratio + (k - ratio + 1) / 2;

            float[] h = ReplicatePad(x, c, t, upPad, upPad);
            int len = t + 2 * upPad;

            // Zero-stuff by `ratio`, then convolve with the pre-reversed filter: the
            // same operation as a transposed convolution, and it batches over channels.
            int stuffedLen = len * ratio;
            var stuffed = new float[(long)c * stuffedLen];
            DirectParallel.For(c, ch =>
            {
                long ho = (long)ch * len, so = (long)ch * stuffedLen;
                for (int i = 0; i < len; i++) stuffed[so + i * ratio] = h[ho + i];
            });

            float[] up = FilterPerChannel(stuffed, c, stuffedLen, a.UpFilter, 1, k - 1, out int upLen);
            int outTime = (len - 1) * ratio + k;
            if (upLen < outTime) throw new InvalidOperationException(
                "MiniMax-H3 audio VAE: upsample produced a short signal.");

            // Scale by the ratio and drop the padding the up-convolution introduced.
            int keep = outTime - padLeft - padRight;
            var mid = new float[(long)c * keep];
            DirectParallel.For(c, ch =>
            {
                long uo = (long)ch * upLen + padLeft, mo = (long)ch * keep;
                for (int i = 0; i < keep; i++) mid[mo + i] = up[uo + i] * ratio;
            });

            SnakeBeta(mid, c, keep, a.Alpha, a.Beta);

            int downLeft = k / 2 - (k % 2 == 0 ? 1 : 0);
            int downRight = k / 2;
            float[] padded = ReplicatePad(mid, c, keep, downLeft, downRight);
            float[] down = FilterPerChannel(padded, c, keep + downLeft + downRight,
                                            a.DownFilter, ratio, 0, out outT);
            return down;
        }

        // ---- decode ----------------------------------------------------------

        /// <summary>
        /// Decode one latent plane. latent is [latentChannels][latentLen]; the
        /// result is the first <paramref name="samples"/> waveform samples.
        /// </summary>
        public float[] Decode(float[] latent, int latentChannels, int latentLen, int samples)
        {
            float[] h = Conv1d(latent, latentChannels, latentLen, _decInProj, out int t);
            h = Conv1d(h, _decInProj.Oc, t, _convPre, out t);
            int c = _convPre.Oc;

            int convIdx = _rates.Length;   // the transposed upsamples occupy the first slots
            int actIdx = 0;
            for (int st = 0; st < _rates.Length; st++)
            {
                Conv up = _convs[st];
                int rate = _rates[st];
                int trim = (up.K - rate) / 2;

                h = ConvTranspose1d(h, c, t, up, rate, out t);
                c = up.Ic;                                   // transposed: out channels live in Ic
                if (trim > 0)
                {
                    int keep = t - 2 * trim;
                    var cut = new float[(long)c * keep];
                    for (int ch = 0; ch < c; ch++)
                        Array.Copy(h, (long)ch * t + trim, cut, (long)ch * keep, keep);
                    h = cut;
                    t = keep;
                }
                if (up.B != null)
                    for (int ch = 0; ch < c; ch++)
                    {
                        float b = up.B[ch];
                        long o = (long)ch * t;
                        for (int i = 0; i < t; i++) h[o + i] += b;
                    }

                // Multi-receptive-field fusion: the AMP blocks are AVERAGED, not summed.
                float[] acc = null;
                for (int am = 0; am < _ampsPerStage; am++)
                {
                    float[] x = h;
                    int xt = t;
                    for (int dl = 0; dl < 3; dl++)
                    {
                        float[] a1 = Activation1d(x, c, xt, _acts[actIdx + dl * 2], out int at);
                        float[] r = Conv1d(a1, c, at, _convs[convIdx + dl], out int rt);
                        float[] a2 = Activation1d(r, c, rt, _acts[actIdx + dl * 2 + 1], out int at2);
                        r = Conv1d(a2, c, at2, _convs[convIdx + 3 + dl], out rt);
                        if (rt != xt) throw new InvalidOperationException(
                            $"MiniMax-H3 audio VAE: residual length {rt} does not match {xt}.");
                        var sum = new float[(long)c * xt];
                        for (long i = 0; i < sum.LongLength; i++) sum[i] = x[i] + r[i];
                        x = sum;
                    }
                    convIdx += 6;
                    actIdx += 6;
                    if (acc == null) acc = x;
                    else for (long i = 0; i < acc.LongLength; i++) acc[i] += x[i];
                }
                float inv = 1f / _ampsPerStage;
                for (long i = 0; i < acc.LongLength; i++) acc[i] *= inv;
                h = acc;
            }

            h = Activation1d(h, c, t, _acts[actIdx], out t);
            h = Conv1d(h, c, t, _convPost, out t);

            // use_tanh_at_final is false for this checkpoint, so the output is clamped.
            var outp = new float[samples];
            int n = Math.Min(samples, t);
            for (int i = 0; i < n; i++) outp[i] = Math.Clamp(h[i], -1f, 1f);
            return outp;
        }
    }

    /// <summary>Thin adapter so this file can use the shared CPU pool without
    /// dragging in the tensor types.</summary>
    internal static class DirectParallel
    {
        public static void For(int count, Action<int> body) => CpuWorkerPool.Shared.For(count, body);
    }
}
