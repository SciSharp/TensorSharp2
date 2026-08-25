// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The seam every video-generation model sits behind, so the CLI and the server can
// drive Wan, MiniMax-H3 and anything added later through one path instead of
// type-testing each concrete model.
using TensorSharp.Models.WanVideo;

namespace TensorSharp.Models.Video
{
    /// <summary>A model that turns a text prompt (plus optional conditioning) into video.</summary>
    public interface IVideoGenerationModel
    {
        /// <summary>Generate a clip. <paramref name="p"/> null = all model defaults.</summary>
        GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams p = null);

        /// <summary>Short family id for UI and telemetry, e.g. "wan" or "minimax-h3".</summary>
        string VideoModelFamily { get; }

        /// <summary>True when the model generates an audio track jointly with the video.</summary>
        bool SupportsAudio { get; }

        /// <summary>True when the model accepts a first-frame conditioning image.</summary>
        bool SupportsImageConditioning { get; }

        /// <summary>True when the model accepts a last-frame conditioning image.</summary>
        bool SupportsEndImageConditioning { get; }

        /// <summary>True when the model accepts reference images/videos/audio.</summary>
        bool SupportsReferenceConditioning { get; }
    }
}
