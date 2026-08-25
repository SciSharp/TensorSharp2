// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 request geometry: the pixel/latent/token shape arithmetic that ties a
// user request (width, height, frame count) to the packed audio-video latent the
// joint diffusion transformer denoises.
//
// H3 is unusual in three ways and every one of them shows up here:
//   * the video VAE compresses 16x spatially and 4x temporally, but its temporal
//     chunking makes the *pixel* frame count live on a 17k+5 grid rather than the
//     4k+1 grid a plain causal VAE would give;
//   * audio is generated jointly, at a fixed 40 Hz latent rate that is independent
//     of the video latent rate, so the two streams have unrelated lengths; and
//   * both streams are carried in ONE tensor -- the audio latent is flat-copied
//     into extra channels appended to the video latent (a byte-level pack, not a
//     semantic channel layout), so a single sampler can step both at once.
using System;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Shape arithmetic for a MiniMax-H3 generation request. Pure functions;
    /// no weights and no tensors, so every rule here is directly unit-testable.</summary>
    public static class MiniMaxH3Geometry
    {
        /// <summary>H3 is trained at 24 fps and any other rate is overridden.</summary>
        public const int Fps = 24;

        /// <summary>Video VAE spatial compression.</summary>
        public const int VaeSpatialRatio = 16;

        /// <summary>Video VAE temporal compression.</summary>
        public const int VaeTemporalRatio = 4;

        /// <summary>Transformer patch size over the latent (t, h, w) = (1, 2, 2).</summary>
        public const int PatchT = 1;
        public const int PatchH = 2;
        public const int PatchW = 2;

        /// <summary>Width/height are aligned up to VAE ratio (16) x transformer down factor (2).</summary>
        public const int SpatialMultiple = VaeSpatialRatio * PatchH; // 32

        /// <summary>Video latent channel count.</summary>
        public const int VideoLatentChannels = 24;

        /// <summary>Audio latent channel count.</summary>
        public const int AudioLatentChannels = 32;

        /// <summary>Audio latents per second: 32 kHz / 800 (the audio VAE's total stride).</summary>
        public const int AudioLatentsPerSecond = 40;

        /// <summary>Stereo: the audio latent carries two channel planes.</summary>
        public const int AudioChannels = 2;

        /// <summary>Shortest clip the 17k+5 frame grid allows.</summary>
        public const int MinFrames = 5;

        /// <summary>Round a pixel dimension up to the 32-px grid the model requires.</summary>
        public static int AlignSpatial(int size)
        {
            if (size <= 0) return SpatialMultiple;
            int r = size % SpatialMultiple;
            return r == 0 ? size : size + (SpatialMultiple - r);
        }

        /// <summary>Round a requested frame count up onto H3's 17k+5 grid (minimum 5).
        /// The grid comes from the video VAE's decode chunking: each 5-latent-frame
        /// chunk yields 17 pixel frames, plus a 5-frame lead-in.</summary>
        public static int AlignFrames(int frames)
        {
            int f = Math.Max(frames, 5);
            while (f % 17 != 5) f++;
            return f;
        }

        /// <summary>Latent temporal length for an aligned pixel frame count.</summary>
        public static int VideoFramesToLatentFrames(int frames)
        {
            if (frames <= 5) return 2;
            return ((frames - 5) / 17) * 5 + 2;
        }

        /// <summary>Inverse of <see cref="VideoFramesToLatentFrames"/>.</summary>
        public static int LatentFramesToVideoFrames(int latentFrames)
        {
            if (latentFrames <= 2) return 5;
            return ((latentFrames - 2) / 5) * 17 + 5;
        }

        /// <summary>Number of audio latent frames for a clip. H3 pins fps to 24, so this
        /// is frames * 5/3, but the general form is kept so the rule stays explicit.</summary>
        public static int AudioLatentCount(int frames, int fps = Fps)
        {
            if (fps <= 0) fps = Fps;
            long n = (long)Math.Round(frames * (double)AudioLatentsPerSecond / fps, MidpointRounding.AwayFromZero);
            return (int)Math.Max(1, n);
        }

        /// <summary>Extra channels needed to carry the audio latent alongside the video
        /// latent in one packed tensor: the audio values are flat-copied into channels
        /// appended after the 24 video channels, with a zero-padded tail.</summary>
        public static int PackedExtraChannels(int latentW, int latentH, int latentT, int audioLatentCount)
        {
            long spatial = (long)latentW * latentH * latentT;
            if (spatial <= 0) return 0;
            long audioValues = (long)audioLatentCount * AudioChannels * AudioLatentChannels;
            return (int)((audioValues + spatial - 1) / spatial);
        }

        /// <summary>Resolve a full request geometry from the user's pixel request.</summary>
        public static MiniMaxH3Shape Resolve(int width, int height, int frames, int fps = Fps)
        {
            int w = AlignSpatial(width);
            int h = AlignSpatial(height);
            int f = AlignFrames(frames);
            int latentW = w / VaeSpatialRatio;
            int latentH = h / VaeSpatialRatio;
            int latentT = VideoFramesToLatentFrames(f);
            int audio = AudioLatentCount(f, fps);
            return new MiniMaxH3Shape
            {
                Width = w,
                Height = h,
                Frames = f,
                Fps = Fps,
                LatentWidth = latentW,
                LatentHeight = latentH,
                LatentFrames = latentT,
                AudioLatentFrames = audio,
                PackedExtraChannels = PackedExtraChannels(latentW, latentH, latentT, audio),
            };
        }
    }

    /// <summary>Resolved shapes for one MiniMax-H3 request.</summary>
    public sealed class MiniMaxH3Shape
    {
        /// <summary>Aligned output width in pixels (multiple of 32).</summary>
        public int Width { get; init; }
        /// <summary>Aligned output height in pixels (multiple of 32).</summary>
        public int Height { get; init; }
        /// <summary>Aligned output frame count (17k+5).</summary>
        public int Frames { get; init; }
        /// <summary>Playback rate; always 24 for H3.</summary>
        public int Fps { get; init; }
        /// <summary>Video latent width (Width / 16).</summary>
        public int LatentWidth { get; init; }
        /// <summary>Video latent height (Height / 16).</summary>
        public int LatentHeight { get; init; }
        /// <summary>Video latent temporal length (5k+2).</summary>
        public int LatentFrames { get; init; }
        /// <summary>Audio latent temporal length at 40 Hz.</summary>
        public int AudioLatentFrames { get; init; }
        /// <summary>Channels appended to the 24 video channels to carry the audio latent.</summary>
        public int PackedExtraChannels { get; init; }

        /// <summary>Transformer token grid width per latent frame (LatentWidth / 2).</summary>
        public int TokenGridWidth => LatentWidth / MiniMaxH3Geometry.PatchW;
        /// <summary>Transformer token grid height per latent frame (LatentHeight / 2).</summary>
        public int TokenGridHeight => LatentHeight / MiniMaxH3Geometry.PatchH;
        /// <summary>Video tokens in the packed sequence.</summary>
        public int VideoTokenCount => LatentFrames * TokenGridWidth * TokenGridHeight;
        /// <summary>Audio tokens in the packed sequence: one per latent frame per stereo channel.</summary>
        public int AudioTokenCount => AudioLatentFrames * MiniMaxH3Geometry.AudioChannels;
        /// <summary>Total channels of the packed audio-video latent.</summary>
        public int PackedChannels => MiniMaxH3Geometry.VideoLatentChannels + PackedExtraChannels;
        /// <summary>Element count of the video latent.</summary>
        public long VideoLatentLength =>
            (long)LatentWidth * LatentHeight * LatentFrames * MiniMaxH3Geometry.VideoLatentChannels;
        /// <summary>Element count of the audio latent.</summary>
        public long AudioLatentLength =>
            (long)AudioLatentFrames * MiniMaxH3Geometry.AudioChannels * MiniMaxH3Geometry.AudioLatentChannels;
        /// <summary>Element count of the packed latent (video + audio + zero padding).</summary>
        public long PackedLatentLength =>
            (long)LatentWidth * LatentHeight * LatentFrames * PackedChannels;
        /// <summary>Clip duration in seconds at 24 fps.</summary>
        public double DurationSeconds => Frames / (double)Fps;
    }
}
