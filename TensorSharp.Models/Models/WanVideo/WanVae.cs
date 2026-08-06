// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan causal 3D video VAE (decoder + encoder), covering both generations:
//   - Wan 2.1 (wan_2.1_vae.safetensors): z=16, 8x spatial / 4x temporal.
//   - Wan 2.2 TI2V (Wan2.2_VAE.safetensors): z=48, 16x spatial (8x conv + 2x2
//     pixel patchify) / 4x temporal, residual AvgDown/DupUp scale shortcuts.
// Weights load from the original safetensors (BF16/F16, upcast once into stable
// unmanaged F32 buffers); each causal 3D conv is pre-sliced into its temporal
// taps so the native graphs (TSGgml_WanVaeDecode / TSGgml_WanVaeEncode) run
// them as batched 2D convolutions. Decode/encode iterate their temporal chunks
// with the causal feature caches inside the one native graph (port of
// stable-diffusion.cpp wan_vae.hpp / diffusers AutoencoderKLWan).
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanVae : IDisposable
    {
        public const int TemporalScale = 4;

        /// <summary>1 = Wan 2.1, 2 = Wan 2.2 TI2V.</summary>
        public int Version { get; }
        public int ZDim { get; }
        /// <summary>Total pixels per latent cell (conv scale x pixel patchify).</summary>
        public int SpatialScale { get; }
        /// <summary>Pixel patchify factor at the encoder input / decoder output (2 for Wan 2.2).</summary>
        public int PatchSize { get; }
        public bool HasEncoder { get; }

        // Wan 2.1 VAE latent normalization (diffusers AutoencoderKLWan config).
        internal static readonly float[] LatentsMean21 = {
            -0.7571f,-0.7089f,-0.9113f,0.1075f,-0.1745f,0.9653f,-0.1517f,1.5508f,
            0.4134f,-0.0715f,0.5517f,-0.3632f,-0.1922f,-0.9497f,0.2503f,-0.2921f };
        internal static readonly float[] LatentsStd21 = {
            2.8184f,1.4541f,2.3275f,2.6558f,1.2196f,1.7708f,2.6052f,2.0743f,
            3.2687f,2.1526f,2.8652f,1.5579f,1.6382f,1.1253f,2.8251f,1.916f };

        // Wan 2.2 TI2V 48-channel latent normalization (Wan2.2-TI2V-5B VAE config).
        internal static readonly float[] LatentsMean22 = {
            -0.2289f,-0.0052f,-0.1323f,-0.2339f,-0.2799f,0.0174f,0.1838f,0.1557f,
            -0.1382f,0.0542f,0.2813f,0.0891f,0.157f,-0.0098f,0.0375f,-0.1825f,
            -0.2246f,-0.1207f,-0.0698f,0.5109f,0.2665f,-0.2108f,-0.2158f,0.2502f,
            -0.2055f,-0.0322f,0.1109f,0.1567f,-0.0729f,0.0899f,-0.2799f,-0.123f,
            -0.0313f,-0.1649f,0.0117f,0.0723f,-0.2839f,-0.2083f,-0.052f,0.3748f,
            0.0152f,0.1957f,0.1433f,-0.2944f,0.3573f,-0.0548f,-0.1681f,-0.0667f };
        internal static readonly float[] LatentsStd22 = {
            0.4765f,1.0364f,0.4514f,1.1677f,0.5313f,0.499f,0.4818f,0.5013f,
            0.8158f,1.0344f,0.5894f,1.0901f,0.6885f,0.6165f,0.8454f,0.4978f,
            0.5759f,0.3523f,0.7135f,0.6804f,0.5833f,1.4146f,0.8986f,0.5659f,
            0.7069f,0.5338f,0.4889f,0.4917f,0.4069f,0.4999f,0.6866f,0.4093f,
            0.5709f,0.6065f,0.6415f,0.4944f,0.5726f,1.2042f,0.5458f,1.6887f,
            0.3971f,1.06f,0.3943f,0.5537f,0.5444f,0.4089f,0.7468f,0.7744f };

        private readonly float[] _mean;
        private readonly float[] _std;
        private readonly List<IntPtr> _allocs = new();
        private readonly WanVaeDecodeArgsTemplate _t;
        private readonly WanVaeEncodeArgsTemplate _e;

        // The descriptors minus the per-call fields, built once at load.
        private struct WanVaeDecodeArgsTemplate
        {
            public WanVaeConv Conv2, Conv1;
            public WanVaeResBlockW Mid0, Mid2;
            public WanVaeAttnW Mid1;
            public WanVaeResBlockW[] Res;   // 12
            public WanVaeUpsampleW[] Up;    // 3
            public WanVaeNorm HeadNorm;
            public WanVaeConv HeadConv;
        }

        private struct WanVaeEncodeArgsTemplate
        {
            public WanVaeConv Stem;
            public WanVaeResBlockW[] Res;   // 8
            public WanVaeDownW[] Down;      // 3
            public WanVaeResBlockW Mid0, Mid2;
            public WanVaeAttnW Mid1;
            public WanVaeNorm HeadNorm;
            public WanVaeConv HeadConv;
            public WanVaeConv Quant;
        }

        public WanVae(string vaePath)
        {
            if (!vaePath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"Wan VAE must be an original safetensors checkpoint (wan_2.1_vae.safetensors / Wan2.2_VAE.safetensors); got '{vaePath}'.");
            using var st = new SafetensorsFile(vaePath);

            // Wan 2.2 nests each scale's blocks inside Up_/Down_ResidualBlock groups.
            Version = st.HasTensor("decoder.upsamples.0.upsamples.0.residual.0.gamma") ? 2 : 1;
            ZDim = Version == 2 ? 48 : 16;
            PatchSize = Version == 2 ? 2 : 1;
            SpatialScale = 8 * PatchSize;
            _mean = Version == 2 ? LatentsMean22 : LatentsMean21;
            _std = Version == 2 ? LatentsStd22 : LatentsStd21;

            WanVaeConv Conv(string prefix, int kd, int k, int ic, int oc)
            {
                float[] w = st.ReadFloat32(prefix + ".weight");   // torch [oc, ic, kd, kh, kw] (or [oc, ic, kh, kw])
                float[] b = st.HasTensor(prefix + ".bias") ? st.ReadFloat32(prefix + ".bias") : null;
                var c = new WanVaeConv { Kd = kd, K = k, Ic = ic, Oc = oc, Bias = b != null ? Reg(b) : IntPtr.Zero };
                int khw = k * k;
                for (int tap = 0; tap < kd; tap++)
                {
                    // torch element (oc, ic, kd, kh, kw) -> ggml tap kernel [kw, kh, ic, oc]
                    // (identical memory nesting once the kd plane is extracted).
                    var slice = new float[(long)oc * ic * khw];
                    int kdhw = kd * khw;
                    int tapLocal = tap;
                    Parallel.For(0, oc, o =>
                    {
                        for (int i = 0; i < ic; i++)
                        {
                            long src = ((long)o * ic + i) * kdhw + (long)tapLocal * khw;
                            long dst = ((long)o * ic + i) * khw;
                            Array.Copy(w, src, slice, dst, khw);
                        }
                    });
                    IntPtr p = Reg(slice);
                    if (tap == 0) c.Tap0 = p; else if (tap == 1) c.Tap1 = p; else c.Tap2 = p;
                }
                return c;
            }
            WanVaeNorm Norm(string name, int ch) => new WanVaeNorm { Gamma = Reg(st.ReadFloat32(name)), C = ch };
            WanVaeResBlockW Res(string prefix, int inDim, int outDim)
            {
                var r = new WanVaeResBlockW
                {
                    N0 = Norm(prefix + ".residual.0.gamma", inDim),
                    C2 = Conv(prefix + ".residual.2", 3, 3, inDim, outDim),
                    N3 = Norm(prefix + ".residual.3.gamma", outDim),
                    C6 = Conv(prefix + ".residual.6", 3, 3, outDim, outDim),
                };
                if (inDim != outDim)
                    r.Shortcut = Conv(prefix + ".shortcut", 1, 1, inDim, outDim);
                return r;
            }
            WanVaeAttnW Attn(string prefix, int dim) => new WanVaeAttnW
            {
                Norm = Norm(prefix + ".norm.gamma", dim),
                QkvW = Reg(st.ReadFloat32(prefix + ".to_qkv.weight")),
                QkvB = Reg(st.ReadFloat32(prefix + ".to_qkv.bias")),
                ProjW = Reg(st.ReadFloat32(prefix + ".proj.weight")),
                ProjB = Reg(st.ReadFloat32(prefix + ".proj.bias")),
                C = dim,
            };

            // ---- decoder ----
            if (Version == 1)
            {
                _t.Conv2 = Conv("conv2", 1, 1, ZDim, ZDim);
                _t.Conv1 = Conv("decoder.conv1", 3, 3, ZDim, 384);
                _t.Mid0 = Res("decoder.middle.0", 384, 384);
                _t.Mid2 = Res("decoder.middle.2", 384, 384);
                _t.Mid1 = Attn("decoder.middle.1", 384);

                // decoder.upsamples.N layout (wan2.1): per scale i in 0..3, three residual
                // blocks then (i < 3) a Resample; the first block of scales 1..3 changes
                // the channel count (the preceding Resample halved it).
                int[] resIn = { 384, 384, 384, 192, 384, 384, 192, 192, 192, 96, 96, 96 };
                int[] resOut = { 384, 384, 384, 384, 384, 384, 192, 192, 192, 96, 96, 96 };
                _t.Res = new WanVaeResBlockW[12];
                _t.Up = new WanVaeUpsampleW[3];
                int idx = 0, ri = 0;
                for (int scale = 0; scale < 4; scale++)
                {
                    for (int rblk = 0; rblk < 3; rblk++)
                    {
                        _t.Res[ri] = Res($"decoder.upsamples.{idx++}", resIn[ri], resOut[ri]);
                        ri++;
                    }
                    if (scale < 3)
                    {
                        string p = $"decoder.upsamples.{idx++}";
                        int dim = resOut[ri - 1];
                        var up = new WanVaeUpsampleW
                        {
                            SConv = Conv(p + ".resample.1", 1, 3, dim, dim / 2),
                        };
                        if (st.HasTensor(p + ".time_conv.weight"))
                            up.TimeConv = Conv(p + ".time_conv", 3, 1, dim, dim * 2);
                        _t.Up[scale] = up;
                    }
                }

                _t.HeadNorm = Norm("decoder.head.0.gamma", 96);
                _t.HeadConv = Conv("decoder.head.2", 3, 3, 96, 3);
            }
            else
            {
                _t.Conv2 = Conv("conv2", 1, 1, ZDim, ZDim);
                _t.Conv1 = Conv("decoder.conv1", 3, 3, ZDim, 1024);
                _t.Mid0 = Res("decoder.middle.0", 1024, 1024);
                _t.Mid2 = Res("decoder.middle.2", 1024, 1024);
                _t.Mid1 = Attn("decoder.middle.1", 1024);

                // wan2.2 decoder: 4 Up_ResidualBlock groups of 3 residual blocks, with a
                // channel-preserving Resample (upsamples.3) after groups 0..2; the DupUp3D
                // shortcuts around each group are weight-free (built in the native graph).
                int[] resIn = { 1024, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256 };
                int[] resOut = { 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256, 256 };
                _t.Res = new WanVaeResBlockW[12];
                _t.Up = new WanVaeUpsampleW[3];
                for (int scale = 0; scale < 4; scale++)
                {
                    for (int rblk = 0; rblk < 3; rblk++)
                    {
                        int ri = scale * 3 + rblk;
                        _t.Res[ri] = Res($"decoder.upsamples.{scale}.upsamples.{rblk}", resIn[ri], resOut[ri]);
                    }
                    if (scale < 3)
                    {
                        string p = $"decoder.upsamples.{scale}.upsamples.3";
                        int dim = resOut[scale * 3 + 2];
                        var up = new WanVaeUpsampleW
                        {
                            SConv = Conv(p + ".resample.1", 1, 3, dim, dim),
                        };
                        if (st.HasTensor(p + ".time_conv.weight"))
                            up.TimeConv = Conv(p + ".time_conv", 3, 1, dim, dim * 2);
                        _t.Up[scale] = up;
                    }
                }

                _t.HeadNorm = Norm("decoder.head.0.gamma", 256);
                _t.HeadConv = Conv("decoder.head.2", 3, 3, 256, 12);
            }

            // ---- encoder (image/video -> latent; used for I2V conditioning) ----
            HasEncoder = st.HasTensor("encoder.conv1.weight");
            if (HasEncoder)
            {
                _e.Res = new WanVaeResBlockW[8];
                _e.Down = new WanVaeDownW[3];
                if (Version == 1)
                {
                    _e.Stem = Conv("encoder.conv1", 3, 3, 3, 96);
                    int[] resIn = { 96, 96, 96, 192, 192, 384, 384, 384 };
                    int[] resOut = { 96, 96, 192, 192, 384, 384, 384, 384 };
                    int idx = 0;
                    for (int scale = 0; scale < 4; scale++)
                    {
                        for (int rblk = 0; rblk < 2; rblk++)
                        {
                            int ri = scale * 2 + rblk;
                            _e.Res[ri] = Res($"encoder.downsamples.{idx++}", resIn[ri], resOut[ri]);
                        }
                        if (scale < 3)
                        {
                            string p = $"encoder.downsamples.{idx++}";
                            int dim = resOut[scale * 2 + 1];
                            var dn = new WanVaeDownW { SConv = Conv(p + ".resample.1", 1, 3, dim, dim) };
                            if (st.HasTensor(p + ".time_conv.weight"))
                                dn.TConv = Conv(p + ".time_conv", 3, 1, dim, dim);
                            _e.Down[scale] = dn;
                        }
                    }
                    _e.Mid0 = Res("encoder.middle.0", 384, 384);
                    _e.Mid2 = Res("encoder.middle.2", 384, 384);
                    _e.Mid1 = Attn("encoder.middle.1", 384);
                    _e.HeadNorm = Norm("encoder.head.0.gamma", 384);
                    _e.HeadConv = Conv("encoder.head.2", 3, 3, 384, 2 * ZDim);
                    _e.Quant = Conv("conv1", 1, 1, 2 * ZDim, 2 * ZDim);
                }
                else
                {
                    _e.Stem = Conv("encoder.conv1", 3, 3, 12, 160);
                    int[] resIn = { 160, 160, 160, 320, 320, 640, 640, 640 };
                    int[] resOut = { 160, 160, 320, 320, 640, 640, 640, 640 };
                    for (int scale = 0; scale < 4; scale++)
                    {
                        for (int rblk = 0; rblk < 2; rblk++)
                        {
                            int ri = scale * 2 + rblk;
                            _e.Res[ri] = Res($"encoder.downsamples.{scale}.downsamples.{rblk}", resIn[ri], resOut[ri]);
                        }
                        if (scale < 3)
                        {
                            string p = $"encoder.downsamples.{scale}.downsamples.2";
                            int dim = resOut[scale * 2 + 1];
                            var dn = new WanVaeDownW { SConv = Conv(p + ".resample.1", 1, 3, dim, dim) };
                            if (st.HasTensor(p + ".time_conv.weight"))
                                dn.TConv = Conv(p + ".time_conv", 3, 1, dim, dim);
                            _e.Down[scale] = dn;
                        }
                    }
                    _e.Mid0 = Res("encoder.middle.0", 640, 640);
                    _e.Mid2 = Res("encoder.middle.2", 640, 640);
                    _e.Mid1 = Attn("encoder.middle.1", 640);
                    _e.HeadNorm = Norm("encoder.head.0.gamma", 640);
                    _e.HeadConv = Conv("encoder.head.2", 3, 3, 640, 2 * ZDim);
                    _e.Quant = Conv("conv1", 1, 1, 2 * ZDim, 2 * ZDim);
                }
            }
        }

        private IntPtr Reg(float[] data)
        {
            long bytes = data.LongLength * sizeof(float);
            IntPtr p = Marshal.AllocHGlobal((IntPtr)bytes);
            _allocs.Add(p);
            Marshal.Copy(data, 0, p, data.Length);
            return p;
        }

        /// <summary>
        /// Decode diffusion latents [ZDim, t, lh, lw] (planar c,t,h,w) into RGB frames.
        /// Above ~0.5 MP the decode is spatially tiled into full-width horizontal bands
        /// (each ~0.4 MP, the scale the whole-graph kernel is fast at) blended over an
        /// 8-latent-row overlap — the diffusers AutoencoderKLWan enable_tiling approach.
        /// A 720p plane's activation planes + cross-chunk causal caches otherwise hold
        /// ~12 GB device-resident, which pushes 16 GB WDDM cards into shared-memory
        /// paging (~20x slower). TS_WAN_VAE_TILE=0 disables tiling.
        /// </summary>
        public RgbImage[] Decode(float[] latent, int t, int lh, int lw)
        {
            int outT = 1 + (t - 1) * TemporalScale;
            int W = lw * SpatialScale, H = lh * SpatialScale;

            // de-normalize (z * std + mean) and repack planar [c,t,h,w] -> [W,H,C,T] memory
            var z = new float[(long)lw * lh * ZDim * t];
            for (int c = 0; c < ZDim; c++)
            {
                float std = _std[c], mean = _mean[c];
                for (int tt = 0; tt < t; tt++)
                {
                    long src = ((long)c * t + tt) * lh * lw;
                    long dst = ((long)tt * ZDim + c) * lh * lw;
                    for (int i = 0; i < lh * lw; i++)
                        z[dst + i] = latent[src + i] * std + mean;
                }
            }

            const int OverlapLat = 8;   // latent rows blended between bands
            int bandLat = Math.Max(16, (int)Math.Round(420_000.0 / ((double)W * SpatialScale)));
            bool tile = Environment.GetEnvironmentVariable("TS_WAN_VAE_TILE") != "0"
                        && (long)W * H > 640_000 && lh > bandLat;

            var pixels = new float[(long)W * H * 3 * outT];
            if (!tile)
            {
                DecodeNative(z, t, lh, lw, pixels);
            }
            else
            {
                // Band starts: stride bandLat - OverlapLat, last band pinned to the end.
                var starts = new System.Collections.Generic.List<int>();
                for (int y = 0; ; y += bandLat - OverlapLat)
                {
                    if (y + bandLat >= lh) { starts.Add(Math.Max(0, lh - bandLat)); break; }
                    starts.Add(y);
                }
                Console.WriteLine($"  [wan] VAE decode tiled into {starts.Count} horizontal bands " +
                                  $"({W}x{bandLat * SpatialScale} px, {OverlapLat * SpatialScale} px blend)");
                var band = new float[(long)lw * bandLat * ZDim * t];
                var bandPx = new float[(long)W * (bandLat * SpatialScale) * 3 * outT];
                for (int bi = 0; bi < starts.Count; bi++)
                {
                    int y0 = starts[bi];
                    // extract latent rows y0..y0+bandLat (contiguous per (t,c) slab)
                    for (int tt = 0; tt < t; tt++)
                        for (int c = 0; c < ZDim; c++)
                            Array.Copy(z, (((long)tt * ZDim + c) * lh + y0) * lw,
                                       band, ((long)tt * ZDim + c) * bandLat * lw,
                                       (long)bandLat * lw);
                    DecodeNative(band, t, bandLat, lw, bandPx);
                    // blend the band into the canvas: cross-fade the overlap rows
                    int py0 = y0 * SpatialScale;
                    int blendPx = bi == 0 ? 0 : (starts[bi - 1] + bandLat - y0) * SpatialScale;
                    int bandH = bandLat * SpatialScale;
                    Parallel.For(0, outT, f =>
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            long srcPlane = ((long)f * 3 + c) * bandH * W;
                            long dstPlane = ((long)f * 3 + c) * H * W;
                            for (int r = 0; r < bandH; r++)
                            {
                                long src = srcPlane + (long)r * W;
                                long dst = dstPlane + (long)(py0 + r) * W;
                                if (r < blendPx)
                                {
                                    float a = (r + 1f) / (blendPx + 1f);
                                    for (int x = 0; x < W; x++)
                                        pixels[dst + x] = pixels[dst + x] * (1f - a) + bandPx[src + x] * a;
                                }
                                else
                                {
                                    Array.Copy(bandPx, src, pixels, dst, W);
                                }
                            }
                        }
                    });
                }
            }

            // [W,H,3,T] memory -> per-frame RGB images ([-1,1] -> [0,1])
            var frames = new RgbImage[outT];
            long hw = (long)W * H;
            Parallel.For(0, outT, f =>
            {
                var chw = new float[3 * hw];
                for (int c = 0; c < 3; c++)
                {
                    long src = ((long)f * 3 + c) * hw;
                    long dst = (long)c * hw;
                    for (long i = 0; i < hw; i++)
                    {
                        float v = (pixels[src + i] + 1f) * 0.5f;
                        chw[dst + i] = v < 0f ? 0f : v > 1f ? 1f : v;
                    }
                }
                frames[f] = RgbImage.FromPlanarChw(W, H, chw);
            });
            return frames;
        }

        // One whole-graph native decode of latent z [lw, latRows, ZDim, t] (native
        // [W,H,C,T] memory order) writing [-1,1] pixels [lw*scale, latRows*scale, 3, outT].
        private unsafe void DecodeNative(float[] z, int t, int latRows, int lw, float[] pixels)
        {
            long outLen = (long)(lw * SpatialScale) * (latRows * SpatialScale) * 3 * (1 + (t - 1) * TemporalScale);
            fixed (float* zp = z, op = pixels)
            {
                var d = new WanVaeDecodeArgs
                {
                    Z = (IntPtr)zp,
                    Out = (IntPtr)op,
                    OutLen = outLen,
                    Conv2 = _t.Conv2,
                    Conv1 = _t.Conv1,
                    Mid0 = _t.Mid0,
                    Mid2 = _t.Mid2,
                    Mid1 = _t.Mid1,
                    Res0 = _t.Res[0], Res1 = _t.Res[1], Res2 = _t.Res[2], Res3 = _t.Res[3],
                    Res4 = _t.Res[4], Res5 = _t.Res[5], Res6 = _t.Res[6], Res7 = _t.Res[7],
                    Res8 = _t.Res[8], Res9 = _t.Res[9], Res10 = _t.Res[10], Res11 = _t.Res[11],
                    Up0 = _t.Up[0], Up1 = _t.Up[1], Up2 = _t.Up[2],
                    HeadNorm = _t.HeadNorm,
                    HeadConv = _t.HeadConv,
                    Zw = lw, Zh = latRows, Zt = t, Zc = ZDim,
                    Version = Version,
                    PatchSize = PatchSize,
                    StructBytes = Marshal.SizeOf<WanVaeDecodeArgs>(),
                };
                if (!GgmlBasicOps.TryWanVaeDecode(in d))
                    throw new InvalidOperationException("Wan VAE decode failed (see native error above).");
            }
        }

        /// <summary>
        /// Encode pixel frames (planar [3, t, H, W], values in [-1,1]) into NORMALIZED
        /// diffusion latents [ZDim, lt, lh, lw] (planar c,t,h,w) — the posterior mean,
        /// scaled to the space the DiT denoises in ((mu - mean) / std).
        /// </summary>
        public unsafe float[] Encode(float[] pixels, int t, int H, int W)
        {
            if (!HasEncoder)
                throw new NotSupportedException("This Wan VAE checkpoint has no encoder weights (decode-only).");
            if (W % SpatialScale != 0 || H % SpatialScale != 0 || (t - 1) % TemporalScale != 0)
                throw new ArgumentException($"Wan VAE encode needs W/H multiples of {SpatialScale} and t = 4k+1; got {W}x{H}x{t}.");

            int p = PatchSize;
            int pw = W / p, ph = H / p, pc = 3 * p * p;
            int lw = W / SpatialScale, lh = H / SpatialScale, lt = 1 + (t - 1) / TemporalScale;

            // planar [3, t, H, W] -> native [pw, ph, pc, t] slabs; for wan2.2 the 2x2
            // pixel patchify interleaves as channel c*4 + xoff*2 + yoff (diffusers order).
            var x = new float[(long)pw * ph * pc * t];
            long phw = (long)pw * ph;
            Parallel.For(0, t, tt =>
            {
                for (int c = 0; c < 3; c++)
                {
                    long src = ((long)c * t + tt) * H * W;
                    if (p == 1)
                    {
                        long dst = ((long)tt * pc + c) * phw;
                        Array.Copy(pixels, src, x, dst, phw);
                    }
                    else
                    {
                        for (int yo = 0; yo < p; yo++)
                            for (int xo = 0; xo < p; xo++)
                            {
                                int c2 = c * p * p + xo * p + yo;
                                long dst = ((long)tt * pc + c2) * phw;
                                for (int h = 0; h < ph; h++)
                                {
                                    long srow = src + (long)(h * p + yo) * W + xo;
                                    long drow = dst + (long)h * pw;
                                    for (int w = 0; w < pw; w++)
                                        x[drow + w] = pixels[srow + w * p];
                                }
                            }
                    }
                }
            });

            long outLen = (long)lw * lh * ZDim * lt;
            var mu = new float[outLen];
            fixed (float* xp = x, op = mu)
            {
                var d = new WanVaeEncodeArgs
                {
                    X = (IntPtr)xp,
                    Out = (IntPtr)op,
                    OutLen = outLen,
                    Stem = _e.Stem,
                    Res0 = _e.Res[0], Res1 = _e.Res[1], Res2 = _e.Res[2], Res3 = _e.Res[3],
                    Res4 = _e.Res[4], Res5 = _e.Res[5], Res6 = _e.Res[6], Res7 = _e.Res[7],
                    Down0 = _e.Down[0], Down1 = _e.Down[1], Down2 = _e.Down[2],
                    Mid0 = _e.Mid0, Mid2 = _e.Mid2, Mid1 = _e.Mid1,
                    HeadNorm = _e.HeadNorm,
                    HeadConv = _e.HeadConv,
                    Quant = _e.Quant,
                    PxW = pw, PxH = ph, PxC = pc, PxT = t,
                    ZDim = ZDim,
                    Version = Version,
                    StructBytes = Marshal.SizeOf<WanVaeEncodeArgs>(),
                };
                if (!GgmlBasicOps.TryWanVaeEncode(in d))
                    throw new InvalidOperationException("Wan VAE encode failed (see native error above).");
            }

            // native [lw, lh, z, lt] -> planar [z, lt, lh, lw], normalized
            var latent = new float[(long)ZDim * lt * lh * lw];
            long lhw = (long)lw * lh;
            for (int c = 0; c < ZDim; c++)
            {
                float inv = 1f / _std[c];
                float mean = _mean[c];
                for (int tt = 0; tt < lt; tt++)
                {
                    long src = ((long)tt * ZDim + c) * lhw;
                    long dst = ((long)c * lt + tt) * lhw;
                    for (long i = 0; i < lhw; i++)
                        latent[dst + i] = (mu[src + i] - mean) * inv;
                }
            }
            return latent;
        }

        public void Dispose()
        {
            foreach (var p in _allocs)
            {
                // The native side resident-caches device copies by host pointer; a later
                // allocation reusing this address must not hit a stale device buffer.
                GgmlBasicOps.InvalidateHostBuffer(p);
                Marshal.FreeHGlobal(p);
            }
            _allocs.Clear();
        }
    }
}
