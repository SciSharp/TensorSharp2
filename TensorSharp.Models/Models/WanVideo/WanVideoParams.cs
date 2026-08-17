// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.WanVideo
{
    /// <summary>A progress report from a running Wan generation.</summary>
    public sealed class WanProgress
    {
        /// <summary>"text-encode", "image-encode", "denoise", "vae-decode" or "done".</summary>
        public string Phase { get; init; }
        /// <summary>Completed denoising steps (0-based count of finished steps).</summary>
        public int Step { get; init; }
        /// <summary>Total denoising steps for this request.</summary>
        public int TotalSteps { get; init; }
        /// <summary>Human-readable detail, e.g. "cond pass" or "band 3/7".</summary>
        public string Detail { get; init; }
        /// <summary>Seconds since generation started.</summary>
        public double ElapsedSeconds { get; init; }
        /// <summary>Estimated seconds still to run, or -1 before the first pass finishes.</summary>
        public double EtaSeconds { get; init; } = -1;
        /// <summary>True when this report is a periodic "still working" tick rather
        /// than a phase transition.</summary>
        public bool Heartbeat { get; init; }
    }

    /// <summary>Sampling parameters for Wan video generation (text-to-video, or
    /// image-to-video on the Wan 2.2 models when a conditioning image is supplied).</summary>
    public sealed class WanVideoParams
    {
        /// <summary>Output width in pixels (rounded to the model grid: 16 px, or 32 for
        /// Wan2.2-TI2V). 0 = default 832 (or the conditioning image's aspect for I2V).</summary>
        public int Width { get; set; }

        /// <summary>Output height in pixels (rounded to the model grid). 0 = default 480
        /// (or the conditioning image's aspect for I2V).</summary>
        public int Height { get; set; }

        /// <summary>Number of output frames; snapped to 4k+1 (the VAE's temporal grid).
        /// 0 = model default (33; 49 for Wan2.2-TI2V). 1 = a still image.</summary>
        public int Frames { get; set; }

        /// <summary>Denoising steps. 0 = model default (30 for Wan 2.1; the official 40
        /// for Wan2.2-A14B / 50 for Wan2.2-TI2V).</summary>
        public int Steps { get; set; }

        /// <summary>Classifier-free guidance scale. 0 = model default (6.0 for Wan 2.1 T2V,
        /// 5.0 for Wan2.2-TI2V, 3.5/4.0 for Wan2.2-A14B); 1.0 disables the negative pass.</summary>
        public float CfgScale { get; set; }

        /// <summary>Wan2.2-A14B only: guidance scale for the low-noise expert phase.
        /// 0 = model default (3.5 for I2V, 3.0 for T2V).</summary>
        public float CfgScale2 { get; set; }

        /// <summary>RNG seed; &lt; 0 picks a random seed.</summary>
        public long Seed { get; set; } = -1;

        /// <summary>FlowMatch timestep shift. 0 = model default (Wan 2.1: 3.0/5.0/8.0 recipe;
        /// Wan 2.2: 5.0, or 12.0 for A14B T2V).</summary>
        public float FlowShift { get; set; }

        /// <summary>Negative prompt; null = the official Wan default negative prompt.</summary>
        public string NegativePrompt { get; set; }

        /// <summary>Sampler: "unipc" (the official Wan sampler, default) or "euler".</summary>
        public string Sampler { get; set; }

        /// <summary>Playback frame rate for the saved video. 0 = model default (16;
        /// 24 for Wan2.2-TI2V, its training rate).</summary>
        public int Fps { get; set; }

        /// <summary>Conditioning image for image-to-video (Wan 2.2 models): becomes the
        /// first frame of the generated video, with the text prompt controlling motion,
        /// camera and scene evolution. Highest precedence of the three image inputs.</summary>
        public RgbImage Image { get; set; }

        /// <summary>Conditioning image as an encoded file (PNG/JPEG/WebP/...).</summary>
        public byte[] ImageBytes { get; set; }

        /// <summary>Conditioning image as a file path.</summary>
        public string ImagePath { get; set; }

        /// <summary>Wan2.2-A14B only: high/low-noise expert boundary as a fraction of the
        /// 1000-step timestep range. 0 = model default (0.9 for I2V, 0.875 for T2V).</summary>
        public float BoundaryRatio { get; set; }

        /// <summary>
        /// Classifier-free guidance cache stride. Every step normally costs TWO DiT
        /// passes (conditional + unconditional). Writing the update as
        /// <c>v = v_cond + (cfg-1) * d</c> with <c>d = v_cond - v_uncond</c> isolates
        /// the guidance direction <c>d</c>, which changes far more slowly across the
        /// schedule than <c>v</c> itself. With a stride of N the unconditional pass
        /// runs on one step in N (plus a warm-up and the final step) and the cached
        /// <c>d</c> covers the rest. At the 50-step TI2V recipe, stride 2 runs 77 of
        /// the 100 passes (1.30x faster) and stride 3 runs 70 (1.43x).
        /// <para>0 or 1 = off (default): every step runs both passes exactly as the
        /// reference pipelines do. This is an approximation — leave it off when
        /// matching a reference sample matters.</para>
        /// </summary>
        public int CfgCacheStride { get; set; }

        /// <summary>Per-step progress callback: (stepIndex, totalSteps).</summary>
        public Action<int, int> OnStep { get; set; }

        /// <summary>
        /// Fine-grained progress. Fires on every phase transition AND on a periodic
        /// heartbeat while a single DiT pass is running, so a caller can show that
        /// work is advancing during the minutes one 720p/121-frame pass takes.
        /// </summary>
        public Action<WanProgress> OnProgress { get; set; }

        /// <summary>Resolve the conditioning image from whichever input was supplied.</summary>
        internal RgbImage ResolveImage()
        {
            if (Image != null) return Image;
            if (ImageBytes is { Length: > 0 }) return ImageIO.Decode(ImageBytes);
            if (!string.IsNullOrWhiteSpace(ImagePath)) return ImageIO.Load(ImagePath);
            return null;
        }
    }
}
