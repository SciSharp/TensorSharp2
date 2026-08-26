// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Progress reporting shared by every video-generation model.
namespace TensorSharp.Models.Video
{
    /// <summary>A progress report from a running video generation.</summary>
    public class VideoGenerationProgress
    {
        /// <summary>"text-encode", "image-encode", "denoise", "vae-decode",
        /// "audio-decode" or "done".</summary>
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
}
