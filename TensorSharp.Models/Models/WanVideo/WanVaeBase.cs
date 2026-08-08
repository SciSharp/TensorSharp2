// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Backend-independent half of the Wan causal 3D video VAE: latent
// (de-)normalization, the spatial band tiling that keeps full-resolution
// decodes inside device memory, pixel (un)patchify and RGB frame assembly.
// The network itself runs behind DecodeNative/EncodeNative — the GGML
// whole-graph kernels (WanVae) or the direct Cuda/Cpu tensors (WanDirectVae).
using System;
using System.Threading.Tasks;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.WanVideo
{
    internal abstract class WanVaeBase : IDisposable
    {
        public const int TemporalScale = 4;

        /// <summary>1 = Wan 2.1, 2 = Wan 2.2 TI2V.</summary>
        public int Version { get; protected set; }
        public int ZDim { get; protected set; }
        /// <summary>Total pixels per latent cell (conv scale x pixel patchify).</summary>
        public int SpatialScale { get; protected set; }
        /// <summary>Pixel patchify factor at the encoder input / decoder output (2 for Wan 2.2).</summary>
        public int PatchSize { get; protected set; }
        public bool HasEncoder { get; protected set; }

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

        protected float[] _mean;
        protected float[] _std;

        protected void InitFromWeights(WanVaeWeights wts)
        {
            Version = wts.Version;
            ZDim = wts.ZDim;
            PatchSize = wts.PatchSize;
            SpatialScale = wts.SpatialScale;
            HasEncoder = wts.HasEncoder;
            _mean = Version == 2 ? LatentsMean22 : LatentsMean21;
            _std = Version == 2 ? LatentsStd22 : LatentsStd21;
        }

        /// <summary>One network decode of de-normalized latents z [lw, latRows, ZDim, t]
        /// (native [W,H,C,T] memory order) writing [-1,1] pixels
        /// [lw*scale, latRows*scale, 3, 1+(t-1)*4].</summary>
        protected abstract void DecodeNative(float[] z, int t, int latRows, int lw, float[] pixels);

        /// <summary>One network encode of pixels x [pw, ph, pc, t] (native memory order,
        /// pre-patchified for Wan 2.2) writing the posterior mean [lw, lh, ZDim, lt].</summary>
        protected abstract void EncodeNative(float[] x, int t, int ph, int pw, int pc, float[] mu);

        /// <summary>Pixel area above which Decode splits into blended horizontal bands.
        /// The GGML path handles ~0.6 MP planes in one graph; the direct path keeps
        /// full-precision activations and caches, so it tiles earlier.</summary>
        protected virtual long TilePixelThreshold => 640_000;

        public abstract void Dispose();

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
            // Band height targets ~2/3 of the tile threshold per band (so a plane
            // just over the threshold still splits into at least two bands).
            int bandLat = Math.Max(16, (int)Math.Round(TilePixelThreshold * 0.66 / ((double)W * SpatialScale)));
            bool tile = Environment.GetEnvironmentVariable("TS_WAN_VAE_TILE") != "0"
                        && (long)W * H > TilePixelThreshold && lh > bandLat;

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

        /// <summary>
        /// Encode pixel frames (planar [3, t, H, W], values in [-1,1]) into NORMALIZED
        /// diffusion latents [ZDim, lt, lh, lw] (planar c,t,h,w) — the posterior mean,
        /// scaled to the space the DiT denoises in ((mu - mean) / std).
        /// </summary>
        public float[] Encode(float[] pixels, int t, int H, int W)
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
            EncodeNative(x, t, ph, pw, pc, mu);

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
    }
}
