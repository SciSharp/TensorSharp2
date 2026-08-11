// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.IO;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Preprocessing for the Muse-Glimmer vision tower, ported from llama.cpp
    /// (tools/mtmd/mtmd-image.cpp: muse_glimmer_grid_size +
    /// mtmd_image_preprocessor_muse_glimmer::preprocess).
    ///
    /// The pipeline is deliberately simple and has no tiling, no padding and no
    /// aspect-ratio preservation:
    ///   1. pick a token grid (nph x npw) whose aspect ratio best matches the
    ///      source image and whose token count stays under maxTokens;
    ///   2. STRETCH the image to (npw * 28, nph * 28) with a Pillow-compatible
    ///      Lanczos-3 filter (llama.cpp RESIZE_ALGO_LANCZOS, PAD_NONE);
    ///   3. u8 / 255, then (x - mean) / std with the mmproj's image_mean /
    ///      image_std (0.5 / 0.5 for this model);
    ///   4. emit channel-first [3, H, W].
    ///
    /// The resize kernel is a first-order source of divergence from the reference
    /// implementation, so the fixed-point two-pass Pillow filter is ported here
    /// verbatim rather than reusing the repo's bilinear helper. Only
    /// <see cref="Gemma3ImageProcessor.DecodeImageToRGBA"/> is shared.
    /// </summary>
    public sealed class MuseGlimmerImageProcessor
    {
        /// <summary>ViT patch size (clip.vision.patch_size).</summary>
        public int PatchSize { get; }

        /// <summary>Pixel-shuffle merge factor (clip.vision.spatial_merge_size).</summary>
        public int MergeSize { get; }

        /// <summary>
        /// Upper bound on output (merged) tokens. llama.cpp calls
        /// set_limit_image_tokens(1, 4096), which sets image_max_pixels to
        /// 4096 * patch_area, so max_tokens = image_max_pixels / patch_area = 4096.
        /// </summary>
        public int MaxTokens { get; }

        /// <summary>patch_size * merge_size: the pixel side of one output token.</summary>
        public int PatchHW => PatchSize * MergeSize;

        private readonly float[] _imageMean;
        private readonly float[] _imageStd;

        public MuseGlimmerImageProcessor(int patchSize = 14, int mergeSize = 2, int maxTokens = 4096,
            float[] imageMean = null, float[] imageStd = null)
        {
            if (patchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(patchSize));
            if (mergeSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(mergeSize));
            if (maxTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxTokens));

            PatchSize = patchSize;
            MergeSize = mergeSize;
            MaxTokens = maxTokens;
            _imageMean = (imageMean != null && imageMean.Length >= 3)
                ? new[] { imageMean[0], imageMean[1], imageMean[2] }
                : new[] { 0.5f, 0.5f, 0.5f };
            _imageStd = (imageStd != null && imageStd.Length >= 3)
                ? new[] { imageStd[0], imageStd[1], imageStd[2] }
                : new[] { 0.5f, 0.5f, 0.5f };
        }

        public static (int width, int height) ReadImageDimensions(string path)
        {
            return Gemma3ImageProcessor.ReadImageDimensions(path);
        }

        /// <summary>
        /// Verbatim port of muse_glimmer_grid_size (llama.cpp), which itself
        /// replicates transformers' get_aspect_ratio_preserving_size. Returns the
        /// stretched target resolution in pixels.
        /// </summary>
        public (int width, int height) ComputeTargetSize(int imgWidth, int imgHeight)
        {
            if (imgWidth <= 0 || imgHeight <= 0)
                throw new ArgumentException($"Invalid image size {imgWidth}x{imgHeight}");

            int patchHw = PatchHW;
            double iNph = (double)imgHeight / patchHw;
            double iNpw = (double)imgWidth / patchHw;
            double ratio = iNph > 0.0 ? iNpw / iNph : 1.0;
            if (iNph * iNpw > (double)MaxTokens)
            {
                iNph = Math.Sqrt((double)MaxTokens / ratio);
                iNpw = iNph * ratio;
            }

            int[] hs = { (int)Math.Floor(iNph), (int)Math.Ceiling(iNph) };
            int[] ws = { (int)Math.Floor(iNpw), (int)Math.Ceiling(iNpw) };
            double targetAr = (double)imgHeight / (double)imgWidth;

            int bestNph = -1;
            int bestNpw = -1;
            double bestD = 0.0;
            for (int a = 0; a < 2; a++)
            {
                for (int b = 0; b < 2; b++)
                {
                    int nph = hs[a];
                    int npw = ws[b];
                    if (nph < 1 || npw < 1 || nph * npw > MaxTokens)
                        continue;

                    double d = Math.Abs((double)nph / (double)npw - targetAr);
                    int nTokens = nph * npw;
                    int bestNTokens = bestNph * bestNpw;
                    if (bestNph < 0 || d < bestD || (d == bestD && nTokens > bestNTokens))
                    {
                        bestNph = nph;
                        bestNpw = npw;
                        bestD = d;
                    }
                }
            }

            if (bestNph < 0)
            {
                // No candidate fit under the cap: round and clamp (std::lround
                // rounds halves away from zero).
                bestNph = Math.Max(1, (int)Math.Round(iNph, MidpointRounding.AwayFromZero));
                bestNpw = Math.Max(1, (int)Math.Round(iNpw, MidpointRounding.AwayFromZero));
            }

            return (bestNpw * patchHw, bestNph * patchHw);
        }

        /// <summary>
        /// Number of merged vision tokens the encoder emits for this image:
        /// (targetW / patch / merge) * (targetH / patch / merge).
        /// </summary>
        public int ComputeTokenCount(int imgWidth, int imgHeight)
        {
            var (w, h) = ComputeTargetSize(imgWidth, imgHeight);
            return (w / PatchSize / MergeSize) * (h / PatchSize / MergeSize);
        }

        public (float[] pixels, int width, int height) ProcessImage(string path)
        {
            return ProcessImageBytes(File.ReadAllBytes(path));
        }

        public (float[] pixels, int width, int height) ProcessImageBytes(byte[] fileBytes)
        {
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(fileBytes, out int srcW, out int srcH);
            var (dstW, dstH) = ComputeTargetSize(srcW, srcH);

            // llama.cpp operates on a 3-channel u8 image. The repo's decoders emit
            // RGBA, so alpha is composited over white first (the repo-wide
            // convention; stb_image, which llama.cpp uses, simply drops alpha, so
            // the two agree for every opaque image).
            byte[] rgb = RgbaToRgbOverWhite(rgba, srcW, srcH);
            byte[] resized = (dstW == srcW && dstH == srcH)
                ? rgb
                : ResizeLanczosPillow(rgb, srcW, srcH, dstW, dstH);

            int plane = dstW * dstH;
            float[] pixels = new float[3 * plane];
            float m0 = _imageMean[0], m1 = _imageMean[1], m2 = _imageMean[2];
            float s0 = _imageStd[0], s1 = _imageStd[1], s2 = _imageStd[2];
            Parallel.For(0, dstH, y =>
            {
                int rowBase = y * dstW;
                for (int x = 0; x < dstW; x++)
                {
                    int i = rowBase + x;
                    int p = i * 3;
                    pixels[i] = (resized[p] / 255f - m0) / s0;
                    pixels[plane + i] = (resized[p + 1] / 255f - m1) / s1;
                    pixels[2 * plane + i] = (resized[p + 2] / 255f - m2) / s2;
                }
            });

            return (pixels, dstW, dstH);
        }

        private static byte[] RgbaToRgbOverWhite(byte[] rgba, int width, int height)
        {
            byte[] rgb = new byte[(long)width * height * 3];
            Parallel.For(0, height, y =>
            {
                int rowBase = y * width;
                for (int x = 0; x < width; x++)
                {
                    int src = (rowBase + x) * 4;
                    int dst = (rowBase + x) * 3;
                    float alpha = rgba[src + 3] / 255f;
                    for (int c = 0; c < 3; c++)
                    {
                        float composited = rgba[src + c] * alpha + 255f * (1f - alpha);
                        rgb[dst + c] = (byte)Math.Clamp((int)(composited + 0.5f), 0, 255);
                    }
                }
            });
            return rgb;
        }

        // ------------------------------------------------------------------
        // Pillow-compatible Lanczos-3 resampling, ported from llama.cpp
        // tools/mtmd/mtmd-image.cpp img_tool::resize_pillow(use_lanczos = true),
        // which is itself adapted from Pillow's src/libImaging/Resample.c.
        //
        // Properties that must be preserved for parity:
        //   - separable: horizontal pass then vertical pass, each over u8 pixels;
        //   - per-output-pixel normalized filter coefficients;
        //   - 22-bit fixed-point integer accumulation with a rounding bias.
        // ------------------------------------------------------------------
        private const int PrecisionBits = 32 - 8 - 2;   // 22
        private const double FilterSupport = 3.0;        // Lanczos-3

        private static double Sinc(double v)
        {
            if (v == 0.0)
                return 1.0;
            double piV = v * 3.141592653589793238462643383279502884;
            return Math.Sin(piV) / piV;
        }

        private static double Lanczos3(double x)
        {
            if (-3.0 <= x && x < 3.0)
                return Sinc(x) * Sinc(x / 3.0);
            return 0.0;
        }

        /// <summary>
        /// Precompute the filter coefficients for one dimension. Returns the kernel
        /// size; bounds[xx*2+0] is the first contributing input index and
        /// bounds[xx*2+1] the contributing count for output pixel xx.
        /// </summary>
        private static int PrecomputeWeights(int inSize, int outSize, out int[] bounds, out int[] weights)
        {
            if (inSize <= 0 || outSize <= 0)
                throw new ArgumentException("PrecomputeWeights requires positive sizes");

            double scale = (double)inSize / outSize;
            double filterScale = scale;
            if (filterScale < 1.0)
                filterScale = 1.0;

            double support = FilterSupport * filterScale;
            int ksize = (int)Math.Ceiling(support) * 2 + 1;

            double[] pre = new double[(long)outSize * ksize];
            bounds = new int[outSize * 2];

            for (int xx = 0; xx < outSize; xx++)
            {
                double center = (xx + 0.5) * scale;
                double ww = 0.0;
                double ss = 1.0 / filterScale;

                int xmin = (int)(center - support + 0.5);
                if (xmin < 0)
                    xmin = 0;

                int xmax = (int)(center + support + 0.5);
                if (xmax > inSize)
                    xmax = inSize;

                xmax -= xmin;

                int x;
                for (x = 0; x < xmax; x++)
                {
                    double w = Lanczos3((x + xmin - center + 0.5) * ss);
                    pre[(long)xx * ksize + x] = w;
                    ww += w;
                }

                for (x = 0; x < xmax; x++)
                {
                    if (ww != 0.0)
                        pre[(long)xx * ksize + x] /= ww;
                }

                for (; x < ksize; x++)
                    pre[(long)xx * ksize + x] = 0;

                bounds[xx * 2 + 0] = xmin;
                bounds[xx * 2 + 1] = xmax;
            }

            weights = new int[(long)outSize * ksize];
            double fxpScale = Math.ScaleB(1.0, PrecisionBits);
            for (long i = 0; i < (long)outSize * ksize; i++)
            {
                // Pillow adds +/- 0.5 then truncates toward zero (a plain round
                // would round twice).
                double rounded = pre[i] * fxpScale + (pre[i] < 0 ? -0.5 : 0.5);
                weights[i] = (int)rounded;
            }

            return ksize;
        }

        private static byte Clip8(int value)
        {
            if (value < 0) return 0;
            if (value > 255) return 255;
            return (byte)value;
        }

        private static byte[] ResampleHorizontal(byte[] src, int srcW, int srcH, int outW,
            int ksize, int[] bounds, int[] weights)
        {
            byte[] dst = new byte[(long)outW * srcH * 3];
            Parallel.For(0, srcH, yy =>
            {
                long srcRow = (long)yy * srcW * 3;
                long dstRow = (long)yy * outW * 3;
                for (int xx = 0; xx < outW; xx++)
                {
                    int xmin = bounds[xx * 2 + 0];
                    int xcnt = bounds[xx * 2 + 1];

                    int ss0 = 1 << (PrecisionBits - 1);
                    int ss1 = 1 << (PrecisionBits - 1);
                    int ss2 = 1 << (PrecisionBits - 1);

                    long wBase = (long)xx * ksize;
                    for (int x = 0; x < xcnt; x++)
                    {
                        long p = srcRow + (long)(x + xmin) * 3;
                        int w = weights[wBase + x];
                        ss0 += src[p] * w;
                        ss1 += src[p + 1] * w;
                        ss2 += src[p + 2] * w;
                    }

                    long q = dstRow + (long)xx * 3;
                    dst[q] = Clip8(ss0 >> PrecisionBits);
                    dst[q + 1] = Clip8(ss1 >> PrecisionBits);
                    dst[q + 2] = Clip8(ss2 >> PrecisionBits);
                }
            });
            return dst;
        }

        private static byte[] ResampleVertical(byte[] src, int srcW, int srcH, int outH,
            int ksize, int[] bounds, int[] weights)
        {
            byte[] dst = new byte[(long)srcW * outH * 3];
            Parallel.For(0, outH, yy =>
            {
                int ymin = bounds[yy * 2 + 0];
                int ycnt = bounds[yy * 2 + 1];
                long dstRow = (long)yy * srcW * 3;
                long wBase = (long)yy * ksize;

                for (int xx = 0; xx < srcW; xx++)
                {
                    int ss0 = 1 << (PrecisionBits - 1);
                    int ss1 = 1 << (PrecisionBits - 1);
                    int ss2 = 1 << (PrecisionBits - 1);

                    for (int y = 0; y < ycnt; y++)
                    {
                        long p = ((long)(y + ymin) * srcW + xx) * 3;
                        int w = weights[wBase + y];
                        ss0 += src[p] * w;
                        ss1 += src[p + 1] * w;
                        ss2 += src[p + 2] * w;
                    }

                    long q = dstRow + (long)xx * 3;
                    dst[q] = Clip8(ss0 >> PrecisionBits);
                    dst[q + 1] = Clip8(ss1 >> PrecisionBits);
                    dst[q + 2] = Clip8(ss2 >> PrecisionBits);
                }
            });
            return dst;
        }

        /// <summary>
        /// Two-pass separable Lanczos-3 resize over a packed RGB u8 buffer.
        /// Mirrors img_tool::resize_pillow's need_horizontal / need_vertical logic.
        /// </summary>
        internal static byte[] ResizeLanczosPillow(byte[] src, int srcW, int srcH, int dstW, int dstH)
        {
            if (dstW <= 0 || dstH <= 0)
                throw new ArgumentException($"Invalid target size {dstW}x{dstH}");

            bool needHorizontal = dstW != srcW;
            bool needVertical = dstH != srcH;

            if (needHorizontal && needVertical)
            {
                int kH = PrecomputeWeights(srcW, dstW, out int[] bH, out int[] wH);
                int kV = PrecomputeWeights(srcH, dstH, out int[] bV, out int[] wV);
                byte[] temp = ResampleHorizontal(src, srcW, srcH, dstW, kH, bH, wH);
                return ResampleVertical(temp, dstW, srcH, dstH, kV, bV, wV);
            }

            if (needHorizontal)
            {
                int kH = PrecomputeWeights(srcW, dstW, out int[] bH, out int[] wH);
                return ResampleHorizontal(src, srcW, srcH, dstW, kH, bH, wH);
            }

            if (needVertical)
            {
                int kV = PrecomputeWeights(srcH, dstH, out int[] bV, out int[] wV);
                return ResampleVertical(src, srcW, srcH, dstH, kV, bV, wV);
            }

            return src;
        }
    }
}
