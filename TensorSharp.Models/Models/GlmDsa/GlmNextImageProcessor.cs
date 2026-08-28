// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// GLM-5.3-Flash (glm5next) image preprocessing, mirroring llama.cpp's
// mtmd_image_preprocessor_glm5next (itself a port of the HF Glm5NextProcessor):
//
//   1. smart_resize: CEIL-align both edges to factor = patch * merge (28). An
//      image below the minimum token budget is upscaled by sqrt(min/area) and
//      re-aligned; one above the maximum is shrunk by BINARY SEARCH over the
//      content height, because aligning both edges is not monotone in the
//      Qwen-style sqrt scale and that scale leaves budget unspent.
//   2. The CONTENT keeps its aspect ratio inside the canvas: scale =
//      min(canvasH/h, canvasW/w), never upscaling an image that already spends
//      the minimum budget. It is composited at the TOP-LEFT (not centred) and
//      the remainder is black padding.
//   3. Bicubic (PyTorch-equivalent) resize, CLIP mean/std normalization,
//      channel-first output.
using System;

namespace TensorSharp.Models
{
    public class GlmNextImageProcessor
    {
        public int PatchSize { get; }
        public int MergeSize { get; }
        public int Factor { get; }
        /// <summary>Minimum pixel budget (16 output tokens' worth).</summary>
        public long MinPixels { get; }
        /// <summary>Maximum pixel budget (8000 output tokens' worth).</summary>
        public long MaxPixels { get; }

        private static readonly float[] Mean = { 0.48145467f, 0.4578275f, 0.40821072f };
        private static readonly float[] Std = { 0.26862955f, 0.2613026f, 0.2757771f };

        public GlmNextImageProcessor(int patchSize = 14, int mergeSize = 2,
            int minTokens = 16, int maxTokens = 8000)
        {
            PatchSize = patchSize;
            MergeSize = mergeSize;
            Factor = patchSize * mergeSize;
            MinPixels = (long)minTokens * Factor * Factor;
            MaxPixels = (long)maxTokens * Factor * Factor;
        }

        public static (int width, int height) ReadImageDimensions(string path)
            => Gemma3ImageProcessor.ReadImageDimensions(path);

        private int Align(long v) => (int)((v + Factor - 1) / Factor * Factor);

        /// <summary>Canvas (aligned) size for an input of the given size.</summary>
        public (int width, int height) SmartResize(int width, int height)
        {
            if (width <= 0 || height <= 0)
                return (0, 0);

            int alignedH = Align(height);
            int alignedW = Align(width);

            // upscale an image that is too small to spend the minimum token budget
            if ((long)alignedH * alignedW < MinPixels)
            {
                double scale = Math.Sqrt((double)MinPixels / ((double)height * width));
                alignedH = Align(Math.Max(1L, (long)Math.Ceiling(height * scale)));
                alignedW = Align(Math.Max(1L, (long)Math.Ceiling(width * scale)));
            }

            if ((long)alignedH * alignedW > MaxPixels)
            {
                // binary search the tallest content height whose aligned canvas
                // still fits the budget
                int low = 1, high = height;
                alignedH = Factor;
                alignedW = Factor;
                while (low <= high)
                {
                    int contentH = (low + high) / 2;
                    int contentW = Math.Max(1, (int)Math.Floor((double)width * contentH / height));
                    int candH = Align(contentH);
                    int candW = Align(contentW);
                    if ((long)candH * candW <= MaxPixels)
                    {
                        alignedH = candH;
                        alignedW = candW;
                        low = contentH + 1;
                    }
                    else
                    {
                        high = contentH - 1;
                    }
                }
            }

            return (alignedW, alignedH);
        }

        /// <summary>Content size (aspect-preserving) inside the canvas.</summary>
        public (int width, int height) ContentSize(int width, int height, int canvasW, int canvasH)
        {
            if (canvasW == 0 || canvasH == 0)
                return (0, 0);
            double scale = Math.Min((double)canvasH / height, (double)canvasW / width);
            if ((long)height * width >= MinPixels)
                scale = Math.Min(1.0, scale);
            return (Math.Max(1, Math.Min(canvasW, (int)Math.Floor(width * scale))),
                    Math.Max(1, Math.Min(canvasH, (int)Math.Floor(height * scale))));
        }

        public int ComputeImageTokenCount(int origWidth, int origHeight)
        {
            var (cw, ch) = SmartResize(origWidth, origHeight);
            return (cw / PatchSize / MergeSize) * (ch / PatchSize / MergeSize);
        }

        public int ComputeImageTokenCount(string imagePath)
        {
            var (w, h) = ReadImageDimensions(imagePath);
            return ComputeImageTokenCount(w, h);
        }

        /// <summary>
        /// Full pipeline: decode, smart-resize, bicubic content resize, top-left
        /// composite on a black canvas, CLIP-normalize. Returns channel-first
        /// [3, canvasH, canvasW] floats plus the canvas geometry.
        /// </summary>
        public (float[] pixels, int canvasH, int canvasW) ProcessImage(string imagePath)
        {
            byte[] fileBytes = System.IO.File.ReadAllBytes(imagePath);
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(fileBytes, out int origW, out int origH);

            var (canvasW, canvasH) = SmartResize(origW, origH);
            if (canvasW == 0 || canvasH == 0)
                throw new ArgumentException($"Image {imagePath} is empty ({origW}x{origH}).");
            var (contentW, contentH) = ContentSize(origW, origH, canvasW, canvasH);

            float[] content = ResizeBicubicCHW(rgba, origW, origH, contentW, contentH);

            // Composite at the top-left of a black canvas, then normalize. The pad
            // color is 0 (llama.cpp's default for this projector), so the padded
            // area normalizes to -mean/std per channel.
            var pixels = new float[3L * canvasH * canvasW];
            long plane = (long)canvasH * canvasW;
            long contentPlane = (long)contentH * contentW;
            for (int c = 0; c < 3; c++)
            {
                float mean = Mean[c], std = Std[c];
                float pad = (0f - mean) / std;
                long cBase = c * plane;
                long sBase = c * contentPlane;
                for (int y = 0; y < canvasH; y++)
                {
                    long row = cBase + (long)y * canvasW;
                    if (y < contentH)
                    {
                        long srow = sBase + (long)y * contentW;
                        for (int x = 0; x < contentW; x++)
                            pixels[row + x] = (content[srow + x] - mean) / std;
                        for (int x = contentW; x < canvasW; x++)
                            pixels[row + x] = pad;
                    }
                    else
                    {
                        for (int x = 0; x < canvasW; x++)
                            pixels[row + x] = pad;
                    }
                }
            }

            return (pixels, canvasH, canvasW);
        }

        // ---- PyTorch-equivalent bicubic (align_corners=False, no antialias) ----

        private static float ClampUnit(double v) => (float)Math.Max(0.0, Math.Min(1.0, v));

        private static int ClampIndex(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

        private static void TorchBicubicWeights(double t, Span<double> w)
        {
            // a = -0.75, the PyTorch/OpenCV constant.
            const double a = -0.75;
            double t2 = t * t, t3 = t2 * t;
            w[0] = a * (t3 - 2 * t2 + t);
            w[1] = (a + 2) * t3 - (a + 3) * t2 + 1;
            w[2] = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t;
            w[3] = -a * (t3 - t2);
        }

        private static float[] ResizeBicubicCHW(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
        {
            long srcPlane = (long)srcW * srcH;
            var src = new float[3 * srcPlane];
            System.Threading.Tasks.Parallel.For(0, srcH, y =>
            {
                for (int x = 0; x < srcW; x++)
                {
                    int idx = (y * srcW + x) * 4;
                    long d = (long)y * srcW + x;
                    src[d] = rgba[idx] / 255.0f;
                    src[srcPlane + d] = rgba[idx + 1] / 255.0f;
                    src[2 * srcPlane + d] = rgba[idx + 2] / 255.0f;
                }
            });

            if (dstW == srcW && dstH == srcH)
                return src;

            long dstPlane = (long)dstW * dstH;
            var dst = new float[3 * dstPlane];
            double scaleX = (double)srcW / dstW;
            double scaleY = (double)srcH / dstH;

            System.Threading.Tasks.Parallel.For(0, dstH, oy =>
            {
                double srcY = scaleY * (oy + 0.5) - 0.5;
                int yBase = (int)Math.Floor(srcY);
                double yFrac = ClampUnit(srcY - yBase);
                Span<double> wy = stackalloc double[4];
                TorchBicubicWeights(yFrac, wy);

                Span<double> wx = stackalloc double[4];
                for (int ox = 0; ox < dstW; ox++)
                {
                    double srcX = scaleX * (ox + 0.5) - 0.5;
                    int xBase = (int)Math.Floor(srcX);
                    double xFrac = ClampUnit(srcX - xBase);
                    TorchBicubicWeights(xFrac, wx);

                    for (int c = 0; c < 3; c++)
                    {
                        double sum = 0;
                        long channelBase = c * srcPlane;
                        for (int ky = 0; ky < 4; ky++)
                        {
                            int iy = ClampIndex(yBase - 1 + ky, 0, srcH - 1);
                            long rowBase = channelBase + (long)iy * srcW;
                            for (int kx = 0; kx < 4; kx++)
                            {
                                int ix = ClampIndex(xBase - 1 + kx, 0, srcW - 1);
                                sum += src[rowBase + ix] * wy[ky] * wx[kx];
                            }
                        }
                        dst[c * dstPlane + (long)oy * dstW + ox] = (float)sum;
                    }
                }
            });

            return dst;
        }
    }
}
