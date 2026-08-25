// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Back-compat shims. The video-generation parameter and progress types moved to
// TensorSharp.Models.Video when a second video model (MiniMax-H3) arrived and the
// knobs stopped being Wan-specific. These aliases keep existing source compiling.
using TensorSharp.Models.Video;

namespace TensorSharp.Models.WanVideo
{
    /// <summary>A progress report from a running Wan generation.</summary>
    /// <remarks>Prefer <see cref="VideoGenerationProgress"/>, which every video model reports.</remarks>
    public sealed class WanProgress : VideoGenerationProgress
    {
    }

    /// <summary>Sampling parameters for Wan video generation.</summary>
    /// <remarks>Prefer <see cref="VideoGenerationParams"/>, which every video model accepts.
    /// This type adds nothing and exists so existing callers keep compiling.</remarks>
    public sealed class WanVideoParams : VideoGenerationParams
    {
    }
}
