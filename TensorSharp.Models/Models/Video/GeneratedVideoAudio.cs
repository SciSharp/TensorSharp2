// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A decoded audio track accompanying a generated video. Models that generate audio
// jointly with video (MiniMax-H3) return one of these; video-only models return null.
namespace TensorSharp.Models.Video
{
    /// <summary>A decoded audio track: planar PCM, one array per channel, samples in [-1, 1].</summary>
    public sealed class GeneratedVideoAudio
    {
        /// <summary>Per-channel planar PCM. Index 0 is left for stereo content.</summary>
        public float[][] Channels { get; init; }

        /// <summary>Sample rate in Hz.</summary>
        public int SampleRate { get; init; }

        /// <summary>Number of channels (2 for stereo).</summary>
        public int ChannelCount => Channels?.Length ?? 0;

        /// <summary>Samples per channel.</summary>
        public int SampleCount => Channels is { Length: > 0 } ? Channels[0].Length : 0;

        /// <summary>Track duration in seconds.</summary>
        public double DurationSeconds => SampleRate > 0 ? SampleCount / (double)SampleRate : 0;
    }
}
