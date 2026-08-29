// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 audio VAE ENCODER: the DAC-style analysis trunk and
// the causal attention projection of TSGgml_MiniMaxH3AudioVaeEncode in managed
// code, so BackendType.Cpu (the 100% pure-C# backend) can turn reference audio
// into latents with no ggml. Without it, reference/i2v/fl2v conditioning has no
// way to encode its audio track on that backend at all.
//
// The encoder is NOT the decoder run backwards, and beyond the conv descriptor
// the two share nothing:
//   - the activation is Snake1d, x + sin(alpha*x)^2 / (alpha + 1e-9), with alpha
//     stored LINEARLY, where the decoder's SnakeBeta keeps alpha and beta in log
//     scale and exponentiates them at load;
//   - the convolutions ZERO-pad through ggml_conv_1d's own padding argument,
//     where the decoder replicate-pads around its resamplers;
//   - the downsample strides are {2, 4, 4, 5, 5} with kernel 2*stride and padding
//     ceil(stride/2) -- the DAC convention, not the usual (k-1)/2 -- which is the
//     only reason the frame count closes exactly at the end.
//
// The whole stage turns on one layout seam. The conv trunk is CHANNEL-major
// (flat c*T + t, ggml ne = [T, C]) and the attention block is TOKEN-major (flat
// t*C + c, ggml ne = [C, T]); the two ggml_cont(ggml_transpose(...)) calls in the
// native op are real data movements, and they are real copies here too. The
// result goes back out channel-major, because that is what NormalizePlane and
// the pipeline's re-pack loop index -- and both would happily consume a
// token-major array of the same length instead, producing a full-size, finite,
// spectrally plausible latent that is simply wrong.
using System;
using System.Numerics;

using TensorSharp;
using TensorSharp.Models.Direct;

// The decoder's conv descriptor already carries exactly the fields a 1-D
// convolution needs, and MiniMaxH3AudioVae builds both halves with one ReadConv
// helper, so the encoder reuses it rather than declaring a second copy.
using Conv = TensorSharp.Models.MiniMaxH3.MiniMaxH3DirectAudioVae.Conv;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed MiniMax-H3 audio VAE encoder (no ggml).</summary>
    internal sealed class MiniMaxH3DirectAudioVaeEncoder
    {
        /// <summary>One encoder Snake1d activation.</summary>
        internal sealed class Snake
        {
            /// <summary>
            /// Wrap one alpha tensor. <paramref name="alpha"/> is the checkpoint
            /// tensor as stored: LINEAR scale, and exponentiating it -- as the
            /// decoder's SnakeBeta requires -- is the trap this type exists to
            /// close. Note also that only the DENOMINATOR carries the 1e-9 guard,
            /// which is why the native descriptor ships alpha and alpha+1e-9 as
            /// two separate host arrays; taking alpha alone here makes that
            /// impossible to collapse by accident.
            /// </summary>
            public Snake(float[] alpha)
            {
                if (alpha is null) throw new ArgumentNullException(nameof(alpha));
                Alpha = alpha;
                Inv = new float[alpha.Length];
                for (int i = 0; i < alpha.Length; i++) Inv[i] = 1f / (alpha[i] + 1e-9f);
            }

            /// <summary>Raw per-channel alpha, the sine's frequency.</summary>
            public float[] Alpha { get; }

            /// <summary>1 / (alpha + 1e-9), the divisor.</summary>
            public float[] Inv { get; }

            /// <summary>Channels the activation covers. The tensor is stored rank-3
            /// as torch [1, C, 1], so a reader that takes Shape[0] gets 1 -- read it
            /// by length, exactly as MiniMaxH3AudioVae.Snake does.</summary>
            public int Channels => Alpha.Length;
        }

        /// <summary>One 2-D linear, weights kept as the raw torch [out, in] floats.</summary>
        internal sealed class Lin
        {
            /// <summary>Weight. torch stores [out, in] row-major, which IS the ggml
            /// [in, out] layout the native binds, so the bytes need no rearranging.</summary>
            public float[] W;
            /// <summary>Bias [out], or null.</summary>
            public float[] B;
            /// <summary>ggml ne0 = in_features (torch Shape[1]).</summary>
            public int Ne0;
            /// <summary>ggml ne1 = out_features (torch Shape[0]).</summary>
            public int Ne1;
        }

        /// <summary>One residual unit of an encoder block.</summary>
        internal sealed class ResUnit
        {
            /// <summary>Snake on the unit's input; the input itself is the residual.</summary>
            public Snake Act1;
            /// <summary>k7, stride 1, dilation d, padding 3*d -- length preserving.</summary>
            public Conv Conv1;
            public Snake Act2;
            /// <summary>k1, a pure per-timestep projection.</summary>
            public Conv Conv2;
        }

        /// <summary>One encoder block: three residual units, a Snake, a downsample.</summary>
        internal sealed class Block
        {
            /// <summary>The three units, dilations 1, 3, 9 in order.</summary>
            public ResUnit[] Units;
            /// <summary>The pre-downsample Snake.</summary>
            public Snake Act;
            /// <summary>k = 2*stride, stride, padding = ceil(stride/2); channels double.</summary>
            public Conv Down;
        }

        /// <summary>The causal attention block that takes the 2048-wide trunk to the
        /// 32-wide latent.</summary>
        internal sealed class AttnBlock
        {
            /// <summary>LayerNorm feeding the QKV projection, [trunk].</summary>
            public float[] Norm1W, Norm1B;
            /// <summary>A SECOND LayerNorm over the SAME pre-attention state, feeding
            /// <see cref="Proj"/>. Not a post-attention norm.</summary>
            public float[] Norm3W, Norm3B;
            /// <summary>LayerNorm on the 32-wide latent, [latent].</summary>
            public float[] Norm2W, Norm2B;
            /// <summary>The MLP's own LayerNorm, applied back-to-back after
            /// <see cref="Norm2W"/>. Both run; dropping either is a silent error.</summary>
            public float[] MlpNormW, MlpNormB;
            /// <summary>Fused QKV, [trunk, 3*trunk]. The checkpoint has no bias for
            /// it -- see <see cref="QkvBias"/>.</summary>
            public Lin Qkv;
            /// <summary>The 3*trunk-wide QKV bias, built on the host as
            /// concat(q_bias, ZEROS[trunk], v_bias): there is no key bias in the
            /// checkpoint, and pre_block.attn.zero_k_bias is never read.</summary>
            public float[] QkvBias;
            /// <summary>32 -> 32, applied AFTER the head mean and the adaptive pool.
            /// It is not a 2048 -> 32 output projection.</summary>
            public Lin AttnProj;
            /// <summary>2048 -> 32, the ONLY thing carrying the residual across the
            /// width change; there is no identity skip.</summary>
            public Lin Proj;
            /// <summary>GeGLU gate branch (w0), value branch (w1), down projection (w2).
            /// w0 and w1 are separate matrices, not one fused 32 -> 128 split.</summary>
            public Lin W0, W1, W2;
            /// <summary>Attention heads. A C# constant on the driver
            /// (MiniMaxH3AudioVae.AttnHeads), not something the checkpoint pins.</summary>
            public int Heads;
        }

        private readonly Conv _convIn, _finalConv, _meanProj;
        private readonly Block[] _blocks;
        private readonly Snake _finalAct;
        private readonly AttnBlock _pre;
        private readonly int _latentChannels, _trunkChannels, _headDim;
        private readonly float _eps;

        /// <summary>
        /// Build the managed encoder. The arguments mirror
        /// TSGgmlMiniMaxH3AudioVaeEncodeDesc field for field, so the driver fills
        /// them from the same tensor names, padding and stride arithmetic it
        /// already uses for the native descriptor.
        /// </summary>
        /// <param name="convIn">encoder.block.0, k7 s1 p3, the only conv with IC = 1.</param>
        /// <param name="blocks">The five downsampling blocks, in order.</param>
        /// <param name="finalAct">encoder.block.6, the Snake on the 2048-wide trunk.</param>
        /// <param name="finalConv">encoder.block.7, k3 s1 p1.</param>
        /// <param name="pre">The pre_block causal attention projection.</param>
        /// <param name="meanProj">mean_proj, k1 -- the deterministic output conv. There
        /// is no logvar branch and no sampling.</param>
        /// <param name="latentChannels">32.</param>
        /// <param name="trunkChannels">2048.</param>
        /// <param name="eps">LayerNorm epsilon; 1e-5f, hardcoded by the driver rather
        /// than read from any config, and NOT the 1e-6 other H3 stages use.</param>
        /// <param name="allocator">Optional, only to reject a CUDA allocator. The
        /// audio VAE is constructed without one (MiniMaxH3Model.CreateAudioVae passes
        /// just the path and the backend), so the guard is a no-op on that route.</param>
        public MiniMaxH3DirectAudioVaeEncoder(Conv convIn, Block[] blocks, Snake finalAct,
                                              Conv finalConv, AttnBlock pre, Conv meanProj,
                                              int latentChannels, int trunkChannels, float eps,
                                              IAllocator allocator = null)
        {
            // The same guard the other direct MiniMax-H3 stages carry in their
            // constructors: everything below is a managed CPU kernel, so a caller
            // holding a CUDA allocator would silently get CPU math on device data.
            // The context owns nothing and is dropped once it has answered.
            var ctx = new DirectContext(allocator);
            if (ctx.IsCuda)
                throw new NotSupportedException(
                    "MiniMaxH3DirectAudioVaeEncoder is the managed CPU path; MiniMax-H3 on " +
                    "BackendType.Cuda is not implemented (use a GGML backend, or BackendType.Cpu).");

            _convIn = convIn ?? throw new ArgumentNullException(nameof(convIn));
            _blocks = blocks ?? throw new ArgumentNullException(nameof(blocks));
            _finalAct = finalAct ?? throw new ArgumentNullException(nameof(finalAct));
            _finalConv = finalConv ?? throw new ArgumentNullException(nameof(finalConv));
            _pre = pre ?? throw new ArgumentNullException(nameof(pre));
            _meanProj = meanProj ?? throw new ArgumentNullException(nameof(meanProj));
            _latentChannels = latentChannels;
            _trunkChannels = trunkChannels;
            _eps = eps;

            if (_blocks.Length == 0)
                throw new ArgumentException("the encoder needs at least one block.", nameof(blocks));
            foreach (Block b in _blocks)
            {
                if (b?.Units == null || b.Units.Length == 0 || b.Act == null || b.Down == null)
                    throw new ArgumentException(
                        "every encoder block needs its residual units, its Snake and its downsample.",
                        nameof(blocks));
                foreach (ResUnit u in b.Units)
                    if (u?.Act1 == null || u.Conv1 == null || u.Act2 == null || u.Conv2 == null)
                        throw new ArgumentException(
                            "every residual unit needs two Snakes and two convolutions.", nameof(blocks));
            }
            if (_pre.AttnProj == null || _pre.Proj == null ||
                _pre.W0 == null || _pre.W1 == null || _pre.W2 == null)
                throw new ArgumentException(
                    "the attention block needs attn_proj, proj and the three MLP matrices.", nameof(pre));
            if (latentChannels <= 0 || trunkChannels <= 0)
                throw new ArgumentOutOfRangeException(nameof(trunkChannels),
                    "latent and trunk widths must be positive.");
            // The native validates exactly these two ratios, and for the same
            // reason: the 2048 -> 32 projection is a pair of means whose group
            // sizes are trunk/heads and (trunk/heads)/latent.
            if (_pre.Heads <= 0 || trunkChannels % _pre.Heads != 0)
                throw new ArgumentException("bad head geometry.", nameof(pre));
            _headDim = trunkChannels / _pre.Heads;
            if (_headDim % latentChannels != 0)
                throw new ArgumentException(
                    "head dim is not a whole multiple of the latent width.", nameof(pre));
            if (_pre.QkvBias == null || _pre.QkvBias.Length != 3 * trunkChannels)
                throw new ArgumentException(
                    $"the fused QKV bias must be {3 * trunkChannels} floats " +
                    "(q_bias | zeros | v_bias).", nameof(pre));
            if (_pre.Qkv == null || _pre.Qkv.Ne0 != trunkChannels || _pre.Qkv.Ne1 != 3 * trunkChannels)
                throw new ArgumentException(
                    $"the QKV weight must be [{3 * trunkChannels}, {trunkChannels}].", nameof(pre));
            if (_finalConv.Oc != trunkChannels)
                throw new ArgumentException(
                    $"final_conv emits {_finalConv.Oc} channels, expected {trunkChannels}.",
                    nameof(finalConv));
            if (_meanProj.Oc != latentChannels || _meanProj.Ic != latentChannels)
                throw new ArgumentException(
                    $"mean_proj must be [{latentChannels}, {latentChannels}, 1].", nameof(meanProj));
        }

        // ---- encode ----------------------------------------------------------

        /// <summary>
        /// Encode one already-truncated mono plane of 32 kHz PCM.
        /// <paramref name="wave"/> holds exactly <paramref name="frames"/> * 800
        /// samples -- the trunk's strided convolutions only land on a whole latent
        /// frame when the input is a whole multiple of the hop, which is why the
        /// caller does the truncation and passes the frame count it derived.
        ///
        /// <para>The result is [channel][time] with time contiguous, i.e. flat
        /// index c*frames + t, and in the VAE's own scale: the pipeline calls
        /// NormalizePlane afterwards, exactly as it does for the native op, so
        /// folding latents_mean/latents_std in here would double-apply them.</para>
        /// </summary>
        public float[] Encode(float[] wave, int frames)
        {
            if (wave is null) throw new ArgumentNullException(nameof(wave));
            if (frames <= 0)
                throw new ArgumentOutOfRangeException(nameof(frames), "at least one latent frame is required.");
            if (wave.Length == 0)
                throw new ArgumentException("the waveform is empty.", nameof(wave));

            // ---- conv trunk, in [channel][time] ----
            float[] h = Conv1d(wave, 1, wave.Length, _convIn, out int t);
            int c = _convIn.Oc;
            foreach (Block b in _blocks)
            {
                foreach (ResUnit u in b.Units)
                {
                    // act1 must not touch h: h IS the residual.
                    float[] y = new float[(long)c * t];
                    Snake1d(h, y, c, t, u.Act1);
                    y = Conv1d(y, c, t, u.Conv1, out int yt);
                    Snake1d(y, y, u.Conv1.Oc, yt, u.Act2);
                    y = Conv1d(y, u.Conv1.Oc, yt, u.Conv2, out int rt);
                    if (rt != t || u.Conv2.Oc != c)
                        throw new InvalidOperationException(
                            $"MiniMax-H3 audio VAE encoder: residual [{u.Conv2.Oc}, {rt}] " +
                            $"does not match [{c}, {t}].");
                    AddInto(h, y);
                }
                Snake1d(h, h, c, t, b.Act);
                h = Conv1d(h, c, t, b.Down, out t);
                c = b.Down.Oc;
            }
            Snake1d(h, h, c, t, _finalAct);
            h = Conv1d(h, c, t, _finalConv, out t);
            // h is now [trunk][frames]: Conv1d checked final_conv's input width and
            // the constructor checked that its output width is the trunk.

            // The cheapest guard there is against a padding or stride slip
            // upstream; the native aborts on the same condition.
            if (t != frames)
                throw new InvalidOperationException(
                    $"MiniMax-H3 audio VAE encoder: trunk produced {t} frames, expected {frames}.");

            // ---- causal attention projection, in [time][channel] ----
            int seq = frames, trunk = _trunkChannels, lat = _latentChannels;
            int heads = _pre.Heads, hd = _headDim;
            // The scale comes from the HEAD dim, not the trunk width.
            float scale = 1f / MathF.Sqrt(hd);

            // A real data move, not a view: everything from here to the transpose
            // back is token-major rows of `trunk` (then `lat`) values.
            float[] x = ToTokenMajor(h, trunk, seq);

            float[] n1 = LayerNorm(x, seq, trunk, _pre.Norm1W, _pre.Norm1B, _eps);
            float[] qkv = Linear(n1, seq, _pre.Qkv, _pre.QkvBias);
            float[] pooled = AttendAndPool(qkv, seq, trunk, heads, hd, lat, scale);
            float[] attn = Linear(pooled, seq, _pre.AttnProj, _pre.AttnProj.B);

            // norm3 reads the SAME unmodified pre-attention x, and proj(n3) is the
            // block's only residual across the 2048 -> 32 width change.
            float[] n3 = LayerNorm(x, seq, trunk, _pre.Norm3W, _pre.Norm3B, _eps);
            float[] state = Linear(n3, seq, _pre.Proj, _pre.Proj.B);
            AddInto(state, attn);

            // Two LayerNorms back to back on the 32-wide latent, both affine. The
            // native says so explicitly; dropping either is a plausible silent error.
            float[] mh = LayerNorm(state, seq, lat, _pre.Norm2W, _pre.Norm2B, _eps);
            mh = LayerNorm(mh, seq, lat, _pre.MlpNormW, _pre.MlpNormB, _eps);

            // GeGLU: w0 is the GATE and w1 is the VALUE. Swapping them is silent.
            float[] gate = Linear(mh, seq, _pre.W0, _pre.W0.B);
            GeluInPlace(gate);
            float[] val = Linear(mh, seq, _pre.W1, _pre.W1.B);
            MulInto(gate, val);
            AddInto(state, Linear(gate, seq, _pre.W2, _pre.W2.B));

            // ---- back to [channel][time] for the 1-wide output convolution ----
            float[] z = ToChannelMajor(state, seq, lat);
            float[] outp = Conv1d(z, lat, seq, _meanProj, out int outFrames);
            if (outFrames != frames)
                throw new InvalidOperationException(
                    $"MiniMax-H3 audio VAE encoder: mean_proj produced {outFrames} frames, " +
                    $"expected {frames}.");
            return outp;
        }

        // ---- conv primitive ---------------------------------------------------

        /// <summary>
        /// ggml_conv_1d over x[ic][T]:
        /// out[oc][t] = sum_ic sum_kk x[ic][t*s - p + kk*d] * w[oc][ic][kk],
        /// zero outside the signal. Cross-correlation -- nothing is flipped, and the
        /// padding is ggml's own zero padding, never the decoder's replicate pad.
        /// </summary>
        private static unsafe float[] Conv1d(float[] x, int ic, int t, Conv c, out int outT)
        {
            int k = c.K, oc = c.Oc, s = c.Stride, p = c.Pad, dl = c.Dilation;
            if (c.Ic != ic)
                throw new InvalidOperationException(
                    $"MiniMax-H3 audio VAE encoder: conv expects {c.Ic} input channels, got {ic}.");
            outT = (t + 2 * p - dl * (k - 1) - 1) / s + 1;
            if (outT <= 0)
                throw new InvalidOperationException("MiniMax-H3 audio VAE encoder: empty conv output.");

            var y = new float[(long)oc * outT];
            int outLen = outT;
            int vw = Vector<float>.Count;
            fixed (float* px0 = x, pw0 = c.W, py0 = y, pb0 = c.B)
            {
                float* px = px0, pw = pw0, py = py0, pb = pb0;
                DirectParallel.For(oc, o =>
                {
                    float* yr = py + (long)o * outLen;
                    for (int i = 0; i < ic; i++)
                    {
                        float* wr = pw + ((long)o * ic + i) * k;
                        float* xr = px + (long)i * t;
                        for (int kk = 0; kk < k; kk++)
                        {
                            float wv = wr[kk];
                            if (wv == 0f) continue;
                            int off = kk * dl - p;
                            // Solve the "is this tap inside the signal" test once
                            // per tap instead of testing it per output sample: with
                            // the test in the innermost loop nothing vectorizes, and
                            // this trunk is ~400 GFLOP for ten seconds of audio.
                            int lo = Math.Max(0, CeilDiv(-off, s));
                            int hi = Math.Min(outLen, FloorDiv(t - 1 - off, s) + 1);
                            if (hi <= lo) continue;
                            if (s == 1)
                            {
                                float* yp = yr + lo;
                                float* xp = xr + lo + off;
                                int n = hi - lo, j = 0;
                                var vwv = new Vector<float>(wv);
                                for (; j + vw <= n; j += vw)
                                {
                                    var vy = new Vector<float>(new ReadOnlySpan<float>(yp + j, vw));
                                    var vx = new Vector<float>(new ReadOnlySpan<float>(xp + j, vw));
                                    (vy + vwv * vx).CopyTo(new Span<float>(yp + j, vw));
                                }
                                for (; j < n; j++) yp[j] += wv * xp[j];
                            }
                            else
                            {
                                for (int ot = lo; ot < hi; ot++) yr[ot] += wv * xr[ot * s + off];
                            }
                        }
                    }
                    // ggml adds the bias as a separate broadcast op AFTER the
                    // convolution; folding it into the accumulator would reorder
                    // the rounding for no gain.
                    if (pb != null)
                        for (int ot = 0; ot < outLen; ot++) yr[ot] += pb[o];
                });
            }
            return y;
        }

        /// <summary>Smallest integer >= a/b, for b > 0 (C# division truncates).</summary>
        private static int CeilDiv(int a, int b) => a >= 0 ? (a + b - 1) / b : -((-a) / b);

        /// <summary>Largest integer &lt;= a/b, for b > 0.</summary>
        private static int FloorDiv(int a, int b) => a >= 0 ? a / b : -(((-a) + b - 1) / b);

        // ---- trunk elementwise ------------------------------------------------

        /// <summary>Snake1d: dst[c][t] = src[c][t] + sin(alpha[c]*src)^2 / (alpha[c] + 1e-9).
        /// src and dst may be the same array. There is no beta term -- this is not
        /// the decoder's SnakeBeta -- and the 1e-9 guard is only in the DIVISOR.</summary>
        private static void Snake1d(float[] src, float[] dst, int channels, int t, Snake s)
        {
            if (s.Channels != channels)
                throw new InvalidOperationException(
                    $"MiniMax-H3 audio VAE encoder: snake covers {s.Channels} channels, got {channels}.");
            DirectParallel.For(channels, ch =>
            {
                float a = s.Alpha[ch], inv = s.Inv[ch];
                long o = (long)ch * t;
                for (int i = 0; i < t; i++)
                {
                    float v = src[o + i];
                    float sn = MathF.Sin(a * v);
                    dst[o + i] = v + sn * sn * inv;
                }
            });
        }

        /// <summary>dst += src, elementwise over the whole plane.</summary>
        private static void AddInto(float[] dst, float[] src)
        {
            SameLength(dst, src);
            ForRanges(dst.LongLength, (lo, hi) =>
            {
                for (long i = lo; i < hi; i++) dst[i] += src[i];
            });
        }

        /// <summary>dst *= src, elementwise (the GeGLU gate).</summary>
        private static void MulInto(float[] dst, float[] src)
        {
            SameLength(dst, src);
            ForRanges(dst.LongLength, (lo, hi) =>
            {
                for (long i = lo; i < hi; i++) dst[i] *= src[i];
            });
        }

        /// <summary>GELU in place, the tanh approximation ggml_gelu uses (not erf).</summary>
        private static void GeluInPlace(float[] v)
        {
            ForRanges(v.LongLength, (lo, hi) =>
            {
                for (long i = lo; i < hi; i++)
                {
                    float x = v[i];
                    v[i] = 0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                }
            });
        }

        private static void SameLength(float[] a, float[] b)
        {
            if (a.LongLength != b.LongLength)
                throw new InvalidOperationException(
                    $"MiniMax-H3 audio VAE encoder: elementwise length mismatch " +
                    $"({a.LongLength} vs {b.LongLength}).");
        }

        /// <summary>Split [0, n) into contiguous ranges across the shared pool.</summary>
        private static void ForRanges(long n, Action<long, long> body)
        {
            if (n <= 0) return;
            int tasks = (int)Math.Min(n, Math.Max(1, CpuWorkerPool.Shared.ThreadCount * 4));
            long chunk = (n + tasks - 1) / tasks;
            DirectParallel.For(tasks, b =>
            {
                long lo = b * chunk;
                if (lo < n) body(lo, Math.Min(n, lo + chunk));
            });
        }

        // ---- layout seam ------------------------------------------------------

        /// <summary>Channel-major [C][T] -> token-major [T][C]: x[t*C + c] = h[c*T + t].</summary>
        private static float[] ToTokenMajor(float[] h, int channels, int t)
        {
            var x = new float[(long)channels * t];
            DirectParallel.For(t, i =>
            {
                long o = (long)i * channels;
                for (int ch = 0; ch < channels; ch++) x[o + ch] = h[(long)ch * t + i];
            });
            return x;
        }

        /// <summary>Token-major [T][C] -> channel-major [C][T]: z[c*T + t] = x[t*C + c].</summary>
        private static float[] ToChannelMajor(float[] x, int t, int channels)
        {
            var z = new float[(long)channels * t];
            DirectParallel.For(channels, ch =>
            {
                long o = (long)ch * t;
                for (int i = 0; i < t; i++) z[o + i] = x[(long)i * channels + ch];
            });
            return z;
        }

        // ---- attention-block primitives ---------------------------------------

        /// <summary>
        /// ggml_norm + affine over the last dim: a MEAN-CENTRED LayerNorm with the
        /// biased variance and eps INSIDE the sqrt. Not an RMS norm -- nothing in
        /// this stage is.
        /// </summary>
        private static unsafe float[] LayerNorm(float[] x, int rows, int cols,
                                                float[] gamma, float[] beta, float eps)
        {
            var y = new float[(long)rows * cols];
            fixed (float* px0 = x, py0 = y, pg0 = gamma, pb0 = beta)
            {
                float* px = px0, py = py0, pg = pg0, pb = pb0;
                DirectParallel.For(rows, r =>
                {
                    float* xr = px + (long)r * cols;
                    float* yr = py + (long)r * cols;
                    // Double accumulators, as ggml_norm uses: at 2048 columns a
                    // float sum of the squares loses several bits of the variance.
                    double sum = 0.0;
                    for (int i = 0; i < cols; i++) sum += xr[i];
                    float mean = (float)(sum / cols);
                    double sum2 = 0.0;
                    for (int i = 0; i < cols; i++)
                    {
                        float v = xr[i] - mean;
                        yr[i] = v;
                        sum2 += (double)v * v;
                    }
                    float scale = 1f / MathF.Sqrt((float)(sum2 / cols) + eps);
                    for (int i = 0; i < cols; i++)
                    {
                        float v = yr[i] * scale;
                        if (pg != null) v *= pg[i];
                        if (pb != null) v += pb[i];
                        yr[i] = v;
                    }
                });
            }
            return y;
        }

        /// <summary>
        /// y[r][j] = sum_c W[j][c] * x[r][c] + bias[j], over token-major rows.
        /// <paramref name="bias"/> is passed separately because the fused QKV
        /// projection's bias is synthesized on the host rather than read from the
        /// checkpoint.
        /// </summary>
        private static unsafe float[] Linear(float[] x, int rows, Lin w, float[] bias)
        {
            int inDim = w.Ne0, outDim = w.Ne1;
            if ((long)inDim * outDim != w.W.LongLength)
                throw new InvalidOperationException(
                    $"MiniMax-H3 audio VAE encoder: linear weight holds {w.W.LongLength} floats, " +
                    $"expected {(long)inDim * outDim}.");
            var y = new float[(long)rows * outDim];
            // Blocking the output columns keeps the pool busy even for the short
            // sequences a one-second reference produces, where a row-per-task split
            // would hand most workers nothing.
            const int ColBlock = 32;
            int blocks = (outDim + ColBlock - 1) / ColBlock;
            fixed (float* px0 = x, pw0 = w.W, py0 = y, pb0 = bias)
            {
                float* px = px0, pw = pw0, py = py0, pb = pb0;
                DirectParallel.For(rows * blocks, idx =>
                {
                    int r = idx / blocks;
                    int j0 = (idx - r * blocks) * ColBlock;
                    int j1 = Math.Min(outDim, j0 + ColBlock);
                    float* xr = px + (long)r * inDim;
                    float* yr = py + (long)r * outDim;
                    for (int j = j0; j < j1; j++)
                    {
                        float s = Dot(xr, pw + (long)j * inDim, inDim);
                        yr[j] = pb != null ? s + pb[j] : s;
                    }
                });
            }
            return y;
        }

        /// <summary>
        /// The whole attention branch: causal single-head attention, the mean over
        /// heads and the adaptive average pool that together take trunk -> latent.
        ///
        /// <para>The native materializes a [T, T, heads] score tensor and a [T, T]
        /// host mask of -inf above the diagonal. Restricting the key loop to
        /// k &lt;= q is exactly equivalent -- ggml_soft_max_ext scales first, then
        /// adds the mask, then softmaxes -- and is O(T) memory with no infinities in
        /// flight.</para>
        /// </summary>
        private static unsafe float[] AttendAndPool(float[] qkv, int seq, int trunk,
                                                    int heads, int hd, int lat, float scale)
        {
            int group = hd / lat;
            int row = 3 * trunk;
            var pooled = new float[(long)seq * lat];
            // Causal rows get monotonically more expensive, so hand out many small
            // chunks and let the pool's atomic claim balance the tail.
            int tasks = Math.Min(seq, Math.Max(1, CpuWorkerPool.Shared.ThreadCount * 8));
            int chunk = (seq + tasks - 1) / tasks;
            int nchunks = (seq + chunk - 1) / chunk;
            fixed (float* pq0 = qkv, pp0 = pooled)
            {
                float* pq = pq0, pp = pp0;
                DirectParallel.For(nchunks, ci =>
                {
                    var probs = new float[seq];
                    var acc = new float[hd];
                    var head = new float[hd];
                    fixed (float* sp = probs, ap = acc, hp = head)
                    {
                        int q0 = ci * chunk, q1 = Math.Min(seq, q0 + chunk);
                        for (int q = q0; q < q1; q++)
                        {
                            for (int e = 0; e < hd; e++) ap[e] = 0f;
                            for (int h = 0; h < heads; h++)
                            {
                                // Q | K | V are BLOCK-contiguous in the fused
                                // projection, then split per head inside each block.
                                float* qr = pq + (long)q * row + (long)h * hd;
                                float maxs = float.NegativeInfinity;
                                for (int kk = 0; kk <= q; kk++)
                                {
                                    float s = Dot(qr, pq + (long)kk * row + trunk + (long)h * hd, hd) * scale;
                                    sp[kk] = s;
                                    if (s > maxs) maxs = s;
                                }
                                float sum = 0f;
                                for (int kk = 0; kk <= q; kk++)
                                {
                                    float e2 = MathF.Exp(sp[kk] - maxs);
                                    sp[kk] = e2;
                                    sum += e2;
                                }
                                float inv = 1f / sum;
                                // Sum this head on its own before folding it in, so
                                // the accumulation order matches the native's
                                // per-head mul_mat followed by ggml_mean.
                                for (int e = 0; e < hd; e++) hp[e] = 0f;
                                for (int kk = 0; kk <= q; kk++)
                                    Axpy(hp, pq + (long)kk * row + 2 * trunk + (long)h * hd,
                                         sp[kk] * inv, hd);
                                for (int e = 0; e < hd; e++) ap[e] += hp[e];
                            }
                            // trunk -> latent is TWO MEANS, not a linear: first
                            // across the heads, then over `group` CONSECUTIVE
                            // head-dim positions of the already head-reduced vector.
                            // Pooling the flattened head-concatenated vector into
                            // `lat` groups of heads*group instead would group WITHIN
                            // a head and give different numbers.
                            float* outr = pp + (long)q * lat;
                            for (int cch = 0; cch < lat; cch++)
                            {
                                int b = cch * group;
                                double s2 = 0.0;
                                for (int j = 0; j < group; j++) s2 += ap[b + j] / heads;
                                outr[cch] = (float)(s2 / group);
                            }
                        }
                    }
                });
            }
            return pooled;
        }

        private static unsafe float Dot(float* a, float* b, int n)
        {
            int vw = Vector<float>.Count;
            var acc = Vector<float>.Zero;
            int i = 0;
            for (; i + vw <= n; i += vw)
                acc += new Vector<float>(new ReadOnlySpan<float>(a + i, vw))
                     * new Vector<float>(new ReadOnlySpan<float>(b + i, vw));
            float sum = Vector.Dot(acc, Vector<float>.One);
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }

        private static unsafe void Axpy(float* y, float* x, float a, int n)
        {
            int vw = Vector<float>.Count;
            var va = new Vector<float>(a);
            int i = 0;
            for (; i + vw <= n; i += vw)
            {
                var vy = new Vector<float>(new ReadOnlySpan<float>(y + i, vw));
                var vx = new Vector<float>(new ReadOnlySpan<float>(x + i, vw));
                (vy + va * vx).CopyTo(new Span<float>(y + i, vw));
            }
            for (; i < n; i++) y[i] += a * x[i];
        }
    }
}
