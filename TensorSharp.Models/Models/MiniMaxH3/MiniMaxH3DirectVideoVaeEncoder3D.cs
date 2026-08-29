// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 video VAE ENCODER: the causal 3-D convolutional
// network of TSGgml_MiniMaxH3VideoVaeEncode3D in managed code, so
// BackendType.Cpu (the 100% pure-C# backend) can turn frames into latents with
// no ggml. That is what unblocks i2v, fl2v and reference conditioning there --
// the decoder only ever needed latents to already exist.
//
// Activations are CHANNELS-LAST [T, H, W, C], because that is the layout in
// which every operation here is a primitive the direct path already has:
//   - a convolution is im2col ([rows*OW, Ic*K*K]) times a flattened kernel
//     [Oc, Ic*K*K], i.e. one DirectOps.CpuGemmABt per temporal tap,
//   - the bias add is DirectOps.AddBiasRows over one frame's rows,
//   - the group norm reduces a contiguous (C/groups)-wide column block over one
//     frame's H*W rows.
// The ggml graph runs [W, H, C, T] instead, but it holds ONE layout throughout
// too -- there is no permute node anywhere in it, the group norm being a
// reshape. So the two layouts differ only at the boundaries: the planar
// [T][C][H][W] pixel buffer is permuted in, the [T][C][H][W] latent is permuted
// out, and nothing in between moves.
//
// Conventions that MUST match ggml, because the weights were laid out for it:
//   - torch [OC, IC, KD, KH, KW] repacks into KD contiguous [OC][IC][KH][KW]
//     blocks, and each block is ALREADY the flattened GEMM kernel
//     [Oc, Ic*Kh*Kw] -- transposing it again is the classic ne-order inversion.
//   - conv is CROSS-CORRELATION: out[oc][oy][ox] = sum_ic sum_kh sum_kw
//     in[ic][oy*s - pad + kh][ox*s - pad + kw] * w[oc][ic][kh][kw].
//   - temporal tap j reads padded frame o*timeStride + j, so j = 0 is the
//     OLDEST frame and j = KD-1 the current one. Reversing that still produces
//     plausible-looking latents, which is why it goes unnoticed.
//   - causal padding is KD-1 ZERO frames at the FRONT and nothing at the back,
//     never edge replication -- which is exactly why a one-frame clip reduces to
//     the kernel's last temporal slice, and why this port gets that reduction
//     for free instead of needing a second code path.
//   - spatial padding is PyTorch 'reflect' (index -1 maps to 1, index n to
//     n-2), and the downsample reflects RIGHT and BOTTOM ONLY.
//   - the group norm is per FRAME with groups=32 and eps=1e-6, and accumulates
//     in double like ggml's own kernel does.
using System;
using System.IO;

using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed MiniMax-H3 causal 3-D video VAE encoder (no ggml).</summary>
    internal sealed class MiniMaxH3DirectVideoVaeEncoder3D : IDisposable
    {
        /// <summary>One 3-D convolution, held as its KD separate 2-D taps: tap j is
        /// the flattened kernel [Oc, Ic*Kh*Kw] the im2col GEMM contracts against.
        /// A k=3 temporal convolution is three 2-D convolutions of shifted frames
        /// summed, which is the same decomposition the native graph uses.</summary>
        private sealed class Conv3T
        {
            public int Kd, Kh, Kw, Ic, Oc;
            public Tensor[] Taps;
            public Tensor Bias;          // null when the checkpoint has no bias
        }

        private sealed class ResBlock3T
        {
            public Tensor Norm1W, Norm1B, Norm2W, Norm2B;
            public Conv3T Conv1, Conv2;

            /// <summary>Present only on the three channel-widening blocks (level 1,
            /// 3 and 5, block 0). Everywhere else the skip is the identity and the
            /// tensor is absent -- do NOT synthesize a 1x1 conv in its place.</summary>
            public Conv3T Shortcut;
        }

        private sealed class Level3T
        {
            public ResBlock3T Block0, Block1;
            public Conv3T Downsample;    // null on the levels that do not downsample
            public int SpaceStride, TimeStride;
        }

        // Cap on one band of materialized im2col columns. Level 0 of a 256x256
        // tile would otherwise build 256*256*1152 floats (1.1 GB) in one piece;
        // banding the output rows bounds the scratch without changing the GEMM.
        private const long GemmBandBytes = 64L << 20;

        private readonly DirectContext _ctx;
        private readonly Conv3T _convIn, _convOut, _quantConv;
        private readonly Level3T[] _levels;
        private readonly Tensor _normOutW, _normOutB;
        private readonly int _groups, _latentChannels;
        private readonly float _eps;
        private bool _disposed;

        /// <param name="allocator">Allocator the activations live on.</param>
        /// <param name="read">Reads a VAE tensor by name as F32, or null when absent.</param>
        /// <param name="shape">Returns a tensor's dimensions in torch order, or null
        /// when absent -- this is how the optional nin_shortcut and downsample
        /// kernels are detected.</param>
        /// <param name="timeDown">Per-level temporal stride. All four downsample
        /// kernels are 3x3x3, so the checkpoint cannot say which axis they stride;
        /// this table (from source_config.json) is the only thing that can.</param>
        /// <param name="spaceDown">Per-level spatial stride.</param>
        /// <param name="levels">Encoder levels in the checkpoint.</param>
        /// <param name="latentChannels">Channels of the posterior MEAN, i.e. half of
        /// what conv_out emits.</param>
        /// <param name="groups">Group-norm groups.</param>
        /// <param name="eps">Group-norm epsilon -- 1e-6 here, NOT the 1e-5 the
        /// decoder's LayerNorms use.</param>
        public MiniMaxH3DirectVideoVaeEncoder3D(
            IAllocator allocator,
            Func<string, float[]> read,
            Func<string, long[]> shape,
            int[] timeDown, int[] spaceDown, int levels,
            int latentChannels, int groups = 32, float eps = 1e-6f)
        {
            if (read is null) throw new ArgumentNullException(nameof(read));
            if (shape is null) throw new ArgumentNullException(nameof(shape));
            if (levels <= 0) throw new ArgumentOutOfRangeException(nameof(levels));
            if (timeDown is null || timeDown.Length != levels)
                throw new ArgumentException($"the stride table must have {levels} entries.", nameof(timeDown));
            if (spaceDown is null || spaceDown.Length != levels)
                throw new ArgumentException($"the stride table must have {levels} entries.", nameof(spaceDown));
            if (latentChannels <= 0) throw new ArgumentOutOfRangeException(nameof(latentChannels));
            if (groups <= 0) throw new ArgumentOutOfRangeException(nameof(groups));

            _ctx = new DirectContext(allocator);
            if (_ctx.IsCuda)
                throw new NotSupportedException(
                    "MiniMaxH3DirectVideoVaeEncoder3D is the managed CPU path; MiniMax-H3 on " +
                    "BackendType.Cuda is not implemented (use a GGML backend, or BackendType.Cpu).");

            _groups = groups; _eps = eps; _latentChannels = latentChannels;

            _convIn = LoadConv("encoder.conv_in", read, shape, required: true);
            _convOut = LoadConv("encoder.conv_out", read, shape, required: true);
            // quant_conv is a TOP-LEVEL tensor, not an 'encoder.' one.
            _quantConv = LoadConv("quant_conv", read, shape, required: true);
            _normOutW = Vec(read("encoder.norm_out.weight"));
            _normOutB = Vec(read("encoder.norm_out.bias"));

            _levels = new Level3T[levels];
            for (int l = 0; l < levels; l++)
            {
                string p = "encoder.down." + l + ".";
                _levels[l] = new Level3T
                {
                    Block0 = LoadBlock(p + "block.0.", read, shape),
                    Block1 = LoadBlock(p + "block.1.", read, shape),
                    Downsample = LoadConv(p + "downsample.conv", read, shape, required: false),
                    SpaceStride = spaceDown[l],
                    TimeStride = timeDown[l],
                };
            }

            // conv_out emits mean | logvar concatenated on the channel axis and
            // quant_conv mixes all of it; only the leading half survives.
            if (_quantConv.Oc < latentChannels)
                throw new InvalidOperationException(
                    $"quant_conv emits {_quantConv.Oc} channels, fewer than the {latentChannels} " +
                    "the posterior mean needs.");
        }

        // ---- weight loading --------------------------------------------------

        private Tensor Vec(float[] v) => v is null ? null : _ctx.Own(_ctx.FromFloats(v, v.LongLength));

        private ResBlock3T LoadBlock(string prefix, Func<string, float[]> read, Func<string, long[]> shape)
            => new ResBlock3T
            {
                Norm1W = Vec(read(prefix + "norm1.weight")),
                Norm1B = Vec(read(prefix + "norm1.bias")),
                Norm2W = Vec(read(prefix + "norm2.weight")),
                Norm2B = Vec(read(prefix + "norm2.bias")),
                Conv1 = LoadConv(prefix + "conv1", read, shape, required: true),
                Conv2 = LoadConv(prefix + "conv2", read, shape, required: true),
                Shortcut = LoadConv(prefix + "nin_shortcut", read, shape, required: false),
            };

        /// <summary>Load a 3-D convolution, hoisting the depth axis to the outside so
        /// slice j is a contiguous [OC][IC][KH][KW] block.
        ///
        /// <para>That block IS the flattened GEMM kernel [Oc, Ic*Kh*Kw] -- the same
        /// repack <c>MiniMaxH3VideoVae.Conv3d</c> performs for the native path, which
        /// is why the two paths can share one interpretation of these weights. Unlike
        /// that one this keeps no pinned host copy and registers nothing with the
        /// ggml device-buffer cache: that bookkeeping exists only to tell the cache
        /// when host pointers die, and paying for it here would double the 721 MB the
        /// encoder's F32 weights already cost.</para></summary>
        private Conv3T LoadConv(string prefix, Func<string, float[]> read, Func<string, long[]> shape,
                                bool required)
        {
            string name = prefix + ".weight";
            long[] dims = shape(name);
            if (dims is null)
            {
                if (required) throw new InvalidOperationException($"missing tensor '{name}'");
                return null;
            }
            if (dims.Length != 5)
                throw new InvalidOperationException(
                    $"'{name}' has rank {dims.Length}; expected a 3-D conv kernel.");

            int oc = (int)dims[0], ic = (int)dims[1], kd = (int)dims[2], kh = (int)dims[3], kw = (int)dims[4];
            if (kh != kw)
                throw new NotSupportedException(
                    $"'{name}' is {kh}x{kw} in space; this encoder assumes square kernels.");
            float[] src = read(name)
                ?? throw new InvalidOperationException($"missing tensor '{name}'");
            long spatial = (long)kh * kw;
            if (src.LongLength != (long)oc * ic * kd * spatial)
                throw new InvalidOperationException(
                    $"'{name}' has {src.LongLength} elements, which does not match " +
                    $"[{oc}, {ic}, {kd}, {kh}, {kw}].");

            var c = new Conv3T { Kd = kd, Kh = kh, Kw = kw, Ic = ic, Oc = oc, Taps = new Tensor[kd] };
            // One staging buffer for all KD slices: FromFloats copies, so it can be
            // refilled rather than allocating kd temporaries of a 113 MB kernel.
            var tap = new float[(long)oc * ic * spatial];
            for (int j = 0; j < kd; j++)
            {
                for (long oi = 0; oi < (long)oc * ic; oi++)
                    Array.Copy(src, (oi * kd + j) * spatial, tap, oi * spatial, spatial);
                c.Taps[j] = _ctx.Own(_ctx.FromFloats(tap, oc, (long)ic * spatial));
            }
            c.Bias = Vec(read(prefix + ".bias"));
            return c;
        }

        // ---- encode ----------------------------------------------------------

        /// <summary>
        /// Encode one chunk of pixel frames through the causal 3-D encoder.
        ///
        /// <para><paramref name="video"/> is the driver's already ImageNet-normalized
        /// buffer laid out [T][C=3][H][W] (ggml ne [W, H, 3, T]) -- NOT the [-1,1]
        /// rescale most VAEs use, and not channels-last. <paramref name="outp"/> is
        /// filled with the posterior mean in the matching [T][C][H][W] order, which
        /// is what the driver's own token-order transpose expects; emitting token
        /// order here instead would come back scrambled with the right L2 norm.</para>
        /// </summary>
        public void Encode(float[] video, int width, int height, int frames,
                           int latentFrames, float[] outp)
        {
            if (video is null) throw new ArgumentNullException(nameof(video));
            if (outp is null) throw new ArgumentNullException(nameof(outp));
            if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
            if (latentFrames <= 0) throw new ArgumentOutOfRangeException(nameof(latentFrames));
            if (video.LongLength < (long)width * height * 3 * frames)
                throw new ArgumentException(
                    $"video holds {video.LongLength} floats, fewer than the " +
                    $"{(long)width * height * 3 * frames} a {width}x{height}x{frames} clip needs.",
                    nameof(video));

            // Parity aid mirroring the native's TS_H3_ENC3D_STOP: cut the network off
            // after one stage so a mismatch localizes to that stage rather than to
            // "the encoder is wrong". As on the native side a stopped run returns
            // WITHOUT filling outp, so the caller sees a zero latent.
            int stopAfter = int.TryParse(
                Environment.GetEnvironmentVariable("TS_H3_ENC3D_STOP"), out int raw) ? raw : -1;

            Tensor h = BuildInput(video, width, height, frames);
            try
            {
                Replace(ref h, CausalConv(_convIn, h, 1, 1, spatialPad: true));
                if (stopAfter == 0) { DumpProbe(h); return; }

                for (int l = 0; l < _levels.Length; l++)
                {
                    Level3T lv = _levels[l];
                    for (int bi = 0; bi < 2; bi++)
                    {
                        ResBlock3T rb = bi == 0 ? lv.Block0 : lv.Block1;

                        // h feeds the skip as well, so the first norm cannot run in
                        // place; the second consumes a dead temporary and can.
                        Tensor t = _ctx.NewF32(h.Sizes);
                        try
                        {
                            GroupNormSilu(t, h, rb.Norm1W, rb.Norm1B);
                            Replace(ref t, CausalConv(rb.Conv1, t, 1, 1, spatialPad: true));
                            GroupNormSilu(t, t, rb.Norm2W, rb.Norm2B);
                            Replace(ref t, CausalConv(rb.Conv2, t, 1, 1, spatialPad: true));
                            if (rb.Shortcut != null) Replace(ref h, Pointwise(rb.Shortcut, h));
                            AddInPlace(h, t);   // the residual add is bare: no scaling, no activation
                        }
                        finally { t.Dispose(); }
                    }
                    if (lv.Downsample != null)
                        Replace(ref h, CausalConv(lv.Downsample, h, lv.SpaceStride, lv.TimeStride,
                                                  spatialPad: false));
                    if (stopAfter == l + 1) { DumpProbe(h); return; }
                }

                GroupNormSilu(h, h, _normOutW, _normOutB);
                if (stopAfter == 100) { DumpProbe(h); return; }
                Replace(ref h, CausalConv(_convOut, h, 1, 1, spatialPad: true));
                if (stopAfter == 101) { DumpProbe(h); return; }

                Replace(ref h, Pointwise(_quantConv, h));

                // The native hard-errors here for the same reason: this check is the
                // only thing between a mistyped stride table and a silently mis-sized
                // latent that the tiling loop would then index out of.
                if (h.Sizes[0] != latentFrames)
                    throw new InvalidOperationException(
                        "MiniMax-H3 3-D video encode: temporal arithmetic produced an unexpected " +
                        $"latent frame count ({h.Sizes[0]}, expected {latentFrames}).");
                long need = h.Sizes[1] * h.Sizes[2] * latentFrames * _latentChannels;
                if (outp.LongLength < need)
                    throw new ArgumentException(
                        $"outp holds {outp.LongLength} floats, fewer than the {need} this latent needs.",
                        nameof(outp));

                WriteMean(h, outp, latentFrames);
            }
            finally { h.Dispose(); }
        }

        /// <summary>Permute the driver's planar [T][C=3][H][W] pixels into the
        /// channels-last [T, H, W, 3] this encoder works in. One of exactly two
        /// permutes in the whole stage; the interior never moves an axis.</summary>
        private unsafe Tensor BuildInput(float[] video, int width, int height, int frames)
        {
            Tensor x = _ctx.NewF32(frames, height, width, 3);
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            long w = width, hgt = height, plane = w * hgt;
            fixed (float* pinned = video)
            {
                // A fixed local cannot be captured by a lambda; a plain pointer copy
                // can, and the pin outlives the (synchronous) parallel loop.
                float* src = pinned;
                DirectOps.RowsParallel((long)frames * hgt, fy =>
                {
                    long f = fy / hgt, y = fy % hgt;
                    float* dst = px + ((f * hgt + y) * w) * 3;
                    for (long c = 0; c < 3; c++)
                    {
                        float* s = src + (f * 3 + c) * plane + y * w;
                        for (long i = 0; i < w; i++) dst[i * 3 + c] = s[i];
                    }
                });
            }
            return x;
        }

        /// <summary>Copy the posterior MEAN -- the leading channels of the 48 that
        /// conv_out emits -- into the driver's [T][C][H][W] buffer.
        ///
        /// <para>The logvar half is computed and thrown away, exactly as the reference
        /// does at inference (the posterior is never sampled). It cannot be skipped
        /// earlier because quant_conv mixes all 48 channels.</para></summary>
        private unsafe void WriteMean(Tensor h, float[] outp, int latentFrames)
        {
            long lh = h.Sizes[1], lw = h.Sizes[2], c = h.Sizes[3], plane = lh * lw;
            long lc = _latentChannels;
            float* ph = (float*)CpuNativeHelpers.GetBufferStart(h);
            fixed (float* pinned = outp)
            {
                float* dst = pinned;
                DirectOps.RowsParallel((long)latentFrames * lc, tc =>
                {
                    long t = tc / lc, ch = tc % lc;
                    float* d = dst + (t * lc + ch) * plane;
                    float* s = ph + t * plane * c + ch;
                    for (long i = 0; i < plane; i++) d[i] = s[i * c];
                });
            }
        }

        // ---- layers ----------------------------------------------------------

        /// <summary>
        /// A CausalConv3d: reflect in space, KD-1 zero frames at the FRONT in time.
        ///
        /// <para><paramref name="spatialPad"/> false is the downsample's asymmetric
        /// (right, bottom) reflect, NOT the symmetric pad the residual convs use --
        /// padding it symmetrically shifts the whole latent grid by half a voxel and
        /// every downstream latent comes out subtly wrong.</para></summary>
        private Tensor CausalConv(Conv3T c, Tensor x, int spaceStride, int timeStride, bool spatialPad)
        {
            int padL = 0, padR = 0, padT = 0, padB = 0;
            if (spatialPad) { padL = padR = padT = padB = 1; }
            else if (spaceStride == 2) { padR = padB = 1; }

            // With the KD-1 causal zeros in front the frame count is T + KD - 1, so
            // the output count is the (T - 1) / stride + 1 that the driver's
            // ChunkLatentFrames independently computes.
            long paddedT = x.Sizes[0] + c.Kd - 1;
            long outFrames = (paddedT - c.Kd) / timeStride + 1;
            if (outFrames <= 0)
                throw new InvalidOperationException(
                    "MiniMax-H3 3-D video encode: a temporal convolution consumed every frame.");
            return Conv3dFrames(c, x, outFrames, spaceStride, timeStride, padL, padR, padT, padB);
        }

        /// <summary>A 1x1x1 convolution: no spatial padding, and no causal padding
        /// because KD-1 is zero. nin_shortcut and quant_conv both come through here;
        /// routing them through <see cref="CausalConv"/> is the easiest mistake in
        /// this stage.</summary>
        private Tensor Pointwise(Conv3T c, Tensor x) => Conv3dFrames(c, x, x.Sizes[0], 1, 1, 0, 0, 0, 0);

        /// <summary>
        /// The temporal convolution itself: KD banded 2-D convolutions of shifted
        /// frames, summed, with the bias added once afterwards.
        ///
        /// <para>Output frame o reads source frame <c>o*timeStride + j - (KD-1)</c>
        /// for tap j. A negative index is one of the causal zero frames, whose whole
        /// contribution is zero, so the tap is skipped rather than materialized --
        /// which is precisely why a one-frame clip runs one GEMM per layer instead of
        /// three, with no separate single-frame code path.</para></summary>
        private Tensor Conv3dFrames(Conv3T c, Tensor x, long outFrames, int stride, int timeStride,
                                    int padLeft, int padRight, int padTop, int padBottom)
        {
            long h = x.Sizes[1], w = x.Sizes[2];
            int k = c.Kw, ic = c.Ic;
            long oh = (h + padTop + padBottom - k) / stride + 1;
            long ow = (w + padLeft + padRight - k) / stride + 1;
            if (oh <= 0 || ow <= 0)
                throw new InvalidOperationException(
                    $"MiniMax-H3 3-D video encode: a {k}x{k} stride-{stride} convolution of " +
                    $"{w}x{h} has no output.");
            if (x.Sizes[3] != ic)
                throw new InvalidOperationException(
                    $"MiniMax-H3 3-D video encode: kernel expects {ic} channels, got {x.Sizes[3]}.");

            long kdim = (long)ic * c.Kh * c.Kw;
            bool direct = k == 1 && stride == 1 &&
                          padLeft == 0 && padRight == 0 && padTop == 0 && padBottom == 0;
            long bandRows = direct
                ? oh
                : Math.Max(1, Math.Min(oh, GemmBandBytes / (4 * Math.Max(1, ow * kdim))));
            int causal = c.Kd - 1;

            Tensor outT = _ctx.NewF32(outFrames, oh, ow, c.Oc);
            try
            {
                for (long o = 0; o < outFrames; o++)
                {
                    using Tensor oFrame = outT.Select(0, o);
                    using Tensor o2 = oFrame.View(oh * ow, c.Oc);
                    for (long y0 = 0; y0 < oh; y0 += bandRows)
                    {
                        long rows = Math.Min(bandRows, oh - y0);
                        using Tensor dst = o2.Narrow(0, y0 * ow, rows * ow);
                        bool wrote = false;
                        for (int j = 0; j < c.Kd; j++)
                        {
                            long sf = o * timeStride + j - causal;
                            if (sf < 0) continue;                    // a causal zero frame
                            using Tensor src = x.Select(0, sf);      // [h, w, ic] contiguous
                            if (direct)
                            {
                                using Tensor col = src.View(h * w, ic);
                                using Tensor band = col.Narrow(0, y0 * ow, rows * ow);
                                DirectOps.CpuGemmABt(band, c.Taps[j], dst, 1f, wrote ? 1f : 0f);
                            }
                            else
                            {
                                using Tensor col = _ctx.NewF32(rows * ow, kdim);
                                Im2ColReflect(col, src, ic, h, w, k, ow, padLeft, padTop, stride, y0, rows);
                                DirectOps.CpuGemmABt(col, c.Taps[j], dst, 1f, wrote ? 1f : 0f);
                            }
                            wrote = true;
                        }
                        // Unreachable while KD-1 causal zeros are prepended (tap KD-1
                        // always lands on a real frame), but a band that wrote nothing
                        // would otherwise pass on whatever the allocator handed back.
                        if (!wrote) DirectOps.Zero(_ctx, dst);
                    }
                    // Broadcast over W, H and T: one add per output frame, after the
                    // tap sum, never per tap.
                    if (c.Bias != null) DirectOps.AddBiasRows(_ctx, o2, c.Bias);
                }
            }
            catch { outT.Dispose(); throw; }
            return outT;
        }

        /// <summary>
        /// Materialize one band of a frame's im2col matrix [rows*OW, Ic*Kh*Kw], with
        /// PyTorch 'reflect' padding folded into the index map.
        ///
        /// <para>The native reflect-pads the whole tensor into a fresh buffer instead
        /// (h3_reflect_pad_4d), which at level 0 of a 17-frame 256px tile is another
        /// 578 MB. gallocr aliases that away for it; a managed port has no such
        /// allocator, and the index map costs nothing that a bounds check was not
        /// already paying.</para>
        ///
        /// <para>Column order is ic, then kh, then kw -- exactly the flattened kernel
        /// row [OC][IC][KH][KW] the repack produced.</para></summary>
        private static unsafe void Im2ColReflect(Tensor col, Tensor frame, int ic, long h, long w,
                                                 int k, long ow, int padLeft, int padTop,
                                                 int stride, long y0, long rows)
        {
            float* src = (float*)CpuNativeHelpers.GetBufferStart(frame);
            float* dst = (float*)CpuNativeHelpers.GetBufferStart(col);
            long kdim = (long)ic * k * k;
            DirectOps.RowsParallel(rows * ow, p =>
            {
                long oy = y0 + p / ow, ox = p % ow;
                float* drow = dst + p * kdim;
                long ki = 0;
                for (long cc = 0; cc < ic; cc++)
                    for (long kh = 0; kh < k; kh++)
                    {
                        long iy = Reflect(oy * stride - padTop + kh, h);
                        for (long kw = 0; kw < k; kw++, ki++)
                        {
                            long ix = Reflect(ox * stride - padLeft + kw, w);
                            drow[ki] = src[(iy * w + ix) * ic + cc];
                        }
                    }
            });
        }

        /// <summary>PyTorch 'reflect': mirror about the edge WITHOUT repeating it, so
        /// with pad 1 the new first column is old column 1 and the new last column is
        /// old column n-2. Reflecting each axis independently is equivalent to the
        /// native's pad-W-then-pad-H-on-the-already-padded-tensor, corners included.</summary>
        private static long Reflect(long i, long n)
        {
            if (n <= 1) return 0;
            if (i < 0) return -i;
            if (i >= n) return 2 * (n - 1) - i;
            return i;
        }

        /// <summary>
        /// Per-FRAME group norm followed by SiLU, written into <paramref name="dst"/>
        /// (which may be <paramref name="x"/> itself).
        ///
        /// <para>Time is the BATCH and never part of the reduction: one group spans
        /// H*W positions times a contiguous (C/groups)-wide channel block of ONE
        /// frame. Folding T in would still produce plausible latents while silently
        /// coupling frames the model expects to be independent.</para>
        ///
        /// <para>Both accumulators are double, matching ggml's ggml_float: at level 0
        /// one group reduces 256*256*4 elements, where a float accumulator drifts
        /// measurably against the reference. Note the variance comes from the CENTRED
        /// values, not E[x^2] - E[x]^2, and eps sits INSIDE the square root.</para></summary>
        private unsafe void GroupNormSilu(Tensor dst, Tensor x, Tensor gain, Tensor bias)
        {
            long t = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2], c = x.Sizes[3];
            if (c % _groups != 0)
                throw new InvalidOperationException(
                    $"MiniMax-H3 3-D video encode: {c} channels do not divide into {_groups} groups.");
            long cpg = c / _groups, rows = h * w, n = rows * cpg;
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            float* pd = (float*)CpuNativeHelpers.GetBufferStart(dst);
            float* pw = null, pb = null;
            if (gain != null) pw = (float*)CpuNativeHelpers.GetBufferStart(gain);
            if (bias != null) pb = (float*)CpuNativeHelpers.GetBufferStart(bias);
            float eps = _eps;
            long groups = _groups;

            DirectOps.RowsParallel(t * groups, tg =>
            {
                long frame = tg / groups, g = tg % groups, c0 = g * cpg;
                float* xb = px + frame * rows * c + c0;
                float* db = pd + frame * rows * c + c0;

                double sum = 0.0;
                for (long r = 0; r < rows; r++)
                {
                    float* s = xb + r * c;
                    for (long i = 0; i < cpg; i++) sum += s[i];
                }
                float mean = (float)(sum / n);

                double sum2 = 0.0;
                for (long r = 0; r < rows; r++)
                {
                    float* s = xb + r * c;
                    float* o = db + r * c;
                    for (long i = 0; i < cpg; i++)
                    {
                        float v = s[i] - mean;
                        o[i] = v;
                        sum2 += v * v;
                    }
                }
                float scale = 1.0f / MathF.Sqrt((float)(sum2 / n) + eps);

                for (long r = 0; r < rows; r++)
                {
                    float* o = db + r * c;
                    for (long i = 0; i < cpg; i++)
                    {
                        float v = o[i] * scale;
                        if (pw != null) v *= pw[c0 + i];
                        if (pb != null) v += pb[c0 + i];
                        o[i] = v / (1f + MathF.Exp(-v));      // SiLU, as ggml_silu
                    }
                }
            });
        }

        // ---- plumbing --------------------------------------------------------

        /// <summary>h += t elementwise. Ops.Add would do it, but its contiguous fast
        /// path is single-threaded and counts elements in an int, and one level-0
        /// residual here is 142M of them.</summary>
        private static unsafe void AddInPlace(Tensor h, Tensor t)
        {
            if (h.Sizes[3] != t.Sizes[3])
                throw new InvalidOperationException(
                    $"MiniMax-H3 3-D video encode: residual channel mismatch ({h.Sizes[3]} vs " +
                    $"{t.Sizes[3]}); a channel-widening block is missing its nin_shortcut.");
            long rows = h.Sizes[0] * h.Sizes[1] * h.Sizes[2], cols = h.Sizes[3];
            float* ph = (float*)CpuNativeHelpers.GetBufferStart(h);
            float* pt = (float*)CpuNativeHelpers.GetBufferStart(t);
            int vw = System.Numerics.Vector<float>.Count;
            DirectOps.RowsParallel(rows, r =>
            {
                float* a = ph + r * cols;
                float* b = pt + r * cols;
                long i = 0;
                for (; i + vw <= cols; i += vw)
                    (new System.Numerics.Vector<float>(new ReadOnlySpan<float>(a + i, vw)) +
                     new System.Numerics.Vector<float>(new ReadOnlySpan<float>(b + i, vw)))
                        .CopyTo(new Span<float>(a + i, vw));
                for (; i < cols; i++) a[i] += b[i];
            });
        }

        private static void Replace(ref Tensor slot, Tensor next)
        {
            Tensor old = slot;
            slot = next;
            old.Dispose();
        }

        /// <summary>
        /// Write the stage TS_H3_ENC3D_STOP selected to TS_H3_ENC3D_DUMP: an
        /// int32[4] ne header followed by the F32 payload, both in the native's
        /// [W, H, C, T] order so the two dumps line up. They will not agree bit for
        /// bit -- ggml_conv_2d runs its im2col and kernel in F16, so this managed F32
        /// path lands closer to the PyTorch reference than to the GGML one -- but a
        /// stage-by-stage diff is the only practical way to localize a mismatch.
        /// </summary>
        private static unsafe void DumpProbe(Tensor probe)
        {
            string path = Environment.GetEnvironmentVariable("TS_H3_ENC3D_DUMP");
            if (string.IsNullOrEmpty(path)) return;

            long t = probe.Sizes[0], h = probe.Sizes[1], c = probe.Sizes[3];
            int w = checked((int)probe.Sizes[2]);
            float* p = (float*)CpuNativeHelpers.GetBufferStart(probe);
            var row = new float[w];
            var bytes = new byte[(long)w * sizeof(float)];
            var header = new byte[16];
            Buffer.BlockCopy(new[] { w, (int)h, (int)c, (int)t }, 0, header, 0, header.Length);

            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            fs.Write(header, 0, header.Length);
            for (long ti = 0; ti < t; ti++)
                for (long ci = 0; ci < c; ci++)
                    for (long yi = 0; yi < h; yi++)
                    {
                        float* s = p + ((ti * h + yi) * w) * c + ci;
                        for (int xi = 0; xi < w; xi++) row[xi] = s[xi * c];
                        Buffer.BlockCopy(row, 0, bytes, 0, bytes.Length);
                        fs.Write(bytes, 0, bytes.Length);
                    }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _ctx.Dispose();     // every weight tensor is context-owned
        }
    }
}
