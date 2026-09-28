// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Qwen-Image-2.1 sampling parameters.
    /// Zero-valued step/CFG settings select the model's defaults.
    /// </summary>
    public sealed class QwenImageParams
    {
        /// <summary>Number of denoising (FlowMatch Euler) steps. 0 = auto: 40 steps.</summary>
        public int Steps { get; set; } = 0;

        /// <summary>
        /// Classifier-free guidance scale; &lt;= 1 disables the negative pass (single forward/step).
        /// 0 = auto: 1.0 (the checkpoint's recommended unguided sampling).
        /// Explicit guidance above 1 uses standard CFG without per-token renormalization.
        /// </summary>
        public float CfgScale { get; set; } = 0f;

        /// <summary>Negative prompt for the CFG pass (empty = unconditional).</summary>
        public string NegativePrompt { get; set; } = " ";

        public long Seed { get; set; } = 0;

        /// <summary>
        /// Target output area in pixels (aspect ratio follows the input image).
        /// 0 = the model's native 2048² area, except on the host-CPU backends (cpu, ggml_cpu),
        /// where the automatic area is 1024² (see QwenImage21Pipeline.HostCpuAutomaticArea).
        /// Dimensions are snapped to multiples of 32.
        /// An explicit positive area takes precedence over the model default; a request that
        /// passes exactly the native 2048² area cannot be told from an automatic one (the
        /// server resolves omitted areas to it), so ask for 2048² on a CPU with width/height.
        /// </summary>
        public long TargetArea { get; set; } = 0;

        /// <summary>Resolve the automatic output area, retaining an explicit area.</summary>
        public long ResolveTargetArea() =>
            TargetArea > 0 ? TargetArea : 2048L * 2048;

        /// <summary>Optional explicit output width/height override (0 = derive from input + TargetArea).</summary>
        public int Width { get; set; } = 0;
        public int Height { get; set; } = 0;

        /// <summary>
        /// Optional per-step progress callback for live UI feedback during the denoise loop.
        /// Invoked once after every step as <c>(step, totalSteps, preview)</c> where <c>step</c> is
        /// 1-based and <c>preview</c> is a decoded RGB snapshot of the current (partially denoised)
        /// latent on throttled steps, or <c>null</c> on the steps in between (a progress-only tick).
        /// </summary>
        public Action<int, int, RgbImage> OnStep { get; set; }

        /// <summary>
        /// How many decoded image previews to emit across the denoise loop (0 = progress ticks only,
        /// no decode). Previews are spaced evenly and decoded at reduced resolution to keep the
        /// per-preview VAE cost (and VRAM) small relative to the denoise itself.
        /// </summary>
        public int PreviewCount { get; set; } = 0;
    }
}
