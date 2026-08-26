// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 packed-sequence layout.
//
// H3 is a single-stream transformer: text, conditioning frames, target audio and
// target video are ONE token sequence attended over bidirectionally, with no
// cross-attention anywhere. Two things therefore have to be decided per token
// rather than per tensor:
//
//   1. its 3-axis RoPE position. These are CONTINUOUS floats, not indices. The
//      time axis is measured in audio-latent units (1 unit = 1/40 s) so that video
//      frames and audio frames share one timeline, and the two spatial axes are
//      normalized by the frame's aspect so that a 54x30 latent and a 30x54 latent
//      get comparable coordinates.
//
//   2. its AdaLN modulation row. The model emits 6 modulation vectors x 3
//      modalities per distinct timestep; a token picks (timestepRow * 3 + modality)
//      with modality 0 = visual, 1 = text, 2 = audio. Because conditioning frames
//      sit at a near-clean timestep while the target sits at the current noisy one,
//      several distinct timesteps are live within a single forward pass.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>What a run of tokens in the packed sequence represents.</summary>
    public enum MiniMaxH3SegmentKind
    {
        Text,
        ConditionVideo,
        ConditionAudio,
        TargetAudio,
        TargetVideo,
    }

    /// <summary>AdaLN modality index. The model's modulation projection is laid out
    /// as [timestep][modality][6 vectors].</summary>
    public static class MiniMaxH3Modality
    {
        public const int Visual = 0;
        public const int Text = 1;
        public const int Audio = 2;
        /// <summary>Modalities per timestep row.</summary>
        public const int Count = 3;
        /// <summary>Modulation vectors per (timestep, modality): shift/scale/gate for the
        /// attention branch and again for the MLP branch.</summary>
        public const int VectorsPerRow = 6;
    }

    /// <summary>A half-open token range and the AdaLN row it modulates with.</summary>
    public readonly struct MiniMaxH3ModulationSpan
    {
        public MiniMaxH3ModulationSpan(int start, int end, int row)
        {
            Start = start; End = end; Row = row;
        }
        public int Start { get; }
        public int End { get; }
        /// <summary>Row into the flattened [timestep * 3 + modality] modulation table.</summary>
        public int Row { get; }
        public int Length => End - Start;
    }

    /// <summary>A half-open token range and what it holds.</summary>
    public readonly struct MiniMaxH3SequenceSegment
    {
        public MiniMaxH3SequenceSegment(int start, int end, MiniMaxH3SegmentKind kind, int sourceIndex = -1)
        {
            Start = start; End = end; Kind = kind; SourceIndex = sourceIndex;
        }
        public int Start { get; }
        public int End { get; }
        public MiniMaxH3SegmentKind Kind { get; }
        /// <summary>Index of the conditioning input this segment came from, or -1.</summary>
        public int SourceIndex { get; }
        public int Length => End - Start;
    }

    /// <summary>The resolved packed sequence for one forward pass.</summary>
    /// <summary>What one Ref2VA reference block contributes to the sequence.</summary>
    public enum MiniMaxH3ReferenceKind
    {
        /// <summary>A single still frame — an identity or product reference.</summary>
        Image = 0,
        /// <summary>A clip of latent frames with no soundtrack.</summary>
        Video = 1,
        /// <summary>A clip and its soundtrack, which share one stretch of the timeline.</summary>
        VideoAudio = 2,
        /// <summary>A soundtrack on its own.</summary>
        Audio = 3,
    }

    /// <summary>One Ref2VA reference, as the packed layout sees it: which encoded
    /// latents it draws on and how much of each axis they span.
    ///
    /// <para>Blocks are laid out head to tail on a shared timeline that the target
    /// video then continues from, so a reference does not merely sit beside the
    /// generated clip — it occupies time before it.</para></summary>
    public sealed class MiniMaxH3ReferenceBlock
    {
        public MiniMaxH3ReferenceKind Kind { get; init; }

        /// <summary>Index into the encoded video latents, or -1 for an audio-only block.</summary>
        public int VideoIndex { get; init; } = -1;

        /// <summary>Index into the encoded audio latents, or -1 when there is no sound.</summary>
        public int AudioIndex { get; init; } = -1;

        /// <summary>Latent frames in the referenced clip (1 for a still image).</summary>
        public int LatentFrames { get; init; } = 1;

        /// <summary>Latent height of the referenced clip. References keep their own
        /// aspect ratio, so this need not match the generated canvas.</summary>
        public int LatentHeight { get; init; }
        public int LatentWidth { get; init; }

        /// <summary>Audio latent frames, when this block carries sound.</summary>
        public int AudioLatentFrames { get; init; }
    }

    public sealed class MiniMaxH3PackedLayout
    {
        /// <summary>Flat (t, h, w) triples, 3 floats per token.</summary>
        public float[] Positions { get; init; }
        /// <summary>Per-token-run AdaLN rows, covering the sequence without gaps.</summary>
        public IReadOnlyList<MiniMaxH3ModulationSpan> ModulationSpans { get; init; }
        /// <summary>Per-token-run segment kinds.</summary>
        public IReadOnlyList<MiniMaxH3SequenceSegment> Segments { get; init; }
        /// <summary>Distinct timestep values referenced by <see cref="ModulationSpans"/>.</summary>
        public float[] Timesteps { get; init; }
        /// <summary>Total tokens in the packed sequence.</summary>
        public int TokenCount { get; init; }
        /// <summary>Token range of the denoised audio stream.</summary>
        public MiniMaxH3ModulationSpan AudioSpan { get; init; }
        /// <summary>Token range of the denoised video stream.</summary>
        public MiniMaxH3ModulationSpan VideoSpan { get; init; }
    }

    /// <summary>Builds the packed token layout: RoPE positions plus AdaLN modulation rows.</summary>
    public static class MiniMaxH3Layout
    {
        /// <summary>Pixel frames per audio latent: 24 fps / 40 Hz. Multiplying a pixel-frame
        /// span by this converts it into the shared audio-latent time unit.</summary>
        public const float FrameRescale = 5f / 3f;

        /// <summary>Spatial coordinates are spread over a nominal 32-unit extent.</summary>
        public const float SpatialExtent = 32f;

        // Latent frame k of every 5 covers this many pixel frames. The leading 1 is the
        // causal encoder's lone first frame; 1 + 4*4 = 17 is exactly the frame grid.
        private static readonly int[] SpanTable = { 1, 4, 4, 4, 4 };

        /// <summary>Pixel-frame span of latent frame <paramref name="frame"/>, in audio-latent units.</summary>
        public static float VideoSpan(int frame) => FrameRescale * SpanTable[frame % 5];

        /// <summary>Total time extent of a latent clip, in audio-latent units.</summary>
        public static float VideoDuration(int latentFrames)
        {
            float d = 0f;
            for (int t = 0; t < latentFrames; t++) d += VideoSpan(t);
            return d;
        }

        /// <summary>Post-patch coordinates along one spatial axis. <paramref name="latentDim"/>
        /// is the pre-patch latent extent; the axis has latentDim/2 entries because the
        /// transformer patch is 2x2. Coordinates are normalized by the frame's geometric mean
        /// extent so aspect ratio, not absolute size, sets the spacing.</summary>
        public static float[] SpatialAxis(int latentDim, double sqrtArea)
        {
            int count = latentDim / MiniMaxH3Geometry.PatchH;
            var result = new float[count];
            if (count == 0) return result;
            double ratio = latentDim / sqrtArea;
            for (int i = 0; i < count; i++)
                result[i] = (float)((i * (ratio / count) + (1.0 - ratio) * 0.5) * SpatialExtent);
            return result;
        }

        /// <summary>Spatial axes (height, width) for a latent frame of the given extents.</summary>
        public static (float[] H, float[] W) SpatialAxes(int latentHeight, int latentWidth)
        {
            double sqrtArea = Math.Sqrt((double)latentHeight * latentWidth);
            return (SpatialAxis(latentHeight, sqrtArea), SpatialAxis(latentWidth, sqrtArea));
        }

        /// <summary>Build the packed layout for a text-to-video / image-to-video request.
        /// <paramref name="conditionFrames"/> gives the latent frame count of each conditioning
        /// clip (1 for a keyframe image), and <paramref name="keyframeAtEnd"/> marks the ones
        /// anchored to the end of the timeline rather than the start.</summary>
        public static MiniMaxH3PackedLayout Build(
            int textLength,
            MiniMaxH3Shape shape,
            float sigma,
            IReadOnlyList<int> conditionFrames = null,
            IReadOnlyList<bool> keyframeAtEnd = null,
            IReadOnlyList<int> textTokenModalities = null,
            float videoShift = MiniMaxH3Scheduler.VideoShift,
            float audioShift = MiniMaxH3Scheduler.AudioShift,
            IReadOnlyList<MiniMaxH3ReferenceBlock> references = null)
        {
            if (shape is null) throw new ArgumentNullException(nameof(shape));
            if (textLength < 0) throw new ArgumentOutOfRangeException(nameof(textLength));

            float s = MiniMaxH3Scheduler.ClampSigma(sigma);
            float videoT = MiniMaxH3Scheduler.VideoTimestep(s);
            float audioT = MiniMaxH3Scheduler.AudioTimestep(s, videoShift, audioShift);

            var timesteps = new List<float>();
            int videoRow = AddTimestep(timesteps, videoT);
            int audioRow = AddTimestep(timesteps, audioT);
            int condRow = AddTimestep(timesteps, Math.Max(videoT, MiniMaxH3Scheduler.VisualConditionTimestep));
            int audioCondRow = AddTimestep(timesteps, Math.Max(audioT, 1f));

            var (hAxis, wAxis) = SpatialAxes(shape.LatentHeight, shape.LatentWidth);
            int frameRows = hAxis.Length * wAxis.Length;

            var positions = new List<float>();
            var spans = new List<MiniMaxH3ModulationSpan>();
            var segments = new List<MiniMaxH3SequenceSegment>();

            void Append(float t, float h, float w)
            {
                positions.Add(t); positions.Add(h); positions.Add(w);
            }

            // --- text -------------------------------------------------------------
            for (int i = 0; i < textLength; i++) Append(i, 0f, 0f);
            if (textLength > 0)
                segments.Add(new MiniMaxH3SequenceSegment(0, textLength, MiniMaxH3SegmentKind.Text));

            // Text tokens carry a per-token modality: plain text is 1, but tokens standing in
            // for an injected image (the <|vision_start|>..<|vision_end|> span) are 0 and get
            // modulated on the VISUAL stream. Emit one span per run of equal modality.
            if (textLength > 0)
            {
                int runStart = 0;
                int current = textTokenModalities is { Count: > 0 }
                    ? textTokenModalities[0] : MiniMaxH3Modality.Text;
                for (int i = 1; i <= textLength; i++)
                {
                    // -1 is a run terminator, used only past the end of the text. Letting
                    // it stand in for "no modalities supplied" would split the run and
                    // leave the final span carrying -1 as its modality.
                    int tag = i == textLength
                        ? -1
                        : textTokenModalities is { Count: > 0 }
                            ? textTokenModalities[i]
                            : MiniMaxH3Modality.Text;
                    if (i == textLength || tag != current)
                    {
                        spans.Add(new MiniMaxH3ModulationSpan(
                            runStart, i, videoRow * MiniMaxH3Modality.Count + current));
                        runStart = i;
                        current = tag;
                    }
                }
            }

            int row = textLength;
            float cursor = textLength;

            // --- Ref2VA reference blocks -----------------------------------------
            // Unlike keyframes, references are NOT pinned to the output timeline: each
            // block consumes a stretch of the cursor, and the generated clip starts
            // after the last one. A reference therefore reads as material that came
            // before the shot rather than as a frame the shot must reproduce, which is
            // exactly the difference between Ref2VA and FL2VA.
            if (references is { Count: > 0 })
            {
                foreach (var block in references)
                {
                    float blockEnd = cursor;

                    if (block.Kind is MiniMaxH3ReferenceKind.Audio or MiniMaxH3ReferenceKind.VideoAudio)
                    {
                        if (block.AudioIndex < 0 || block.AudioLatentFrames <= 0)
                            throw new ArgumentException(
                                "an audio reference block needs encoded audio latents.", nameof(references));
                        // Sound rides the width axis of its own clip when it has one, so a
                        // reference video and its soundtrack stay spatially aligned.
                        float low = wAxis.Length > 0 ? wAxis[0] : 0f;
                        float high = wAxis.Length > 0 ? wAxis[^1] : 0f;
                        if (block.VideoIndex >= 0)
                        {
                            var (_, refW) = SpatialAxes(block.LatentHeight, block.LatentWidth);
                            low = refW.Length > 0 ? refW[0] : low;
                            high = refW.Length > 0 ? refW[^1] : high;
                        }
                        int count = block.AudioLatentFrames * MiniMaxH3Geometry.AudioChannels;
                        AppendAudioPositions(Append, block.AudioLatentFrames, cursor, low, high);
                        segments.Add(new MiniMaxH3SequenceSegment(
                            row, row + count, MiniMaxH3SegmentKind.ConditionAudio, block.AudioIndex));
                        spans.Add(new MiniMaxH3ModulationSpan(row, row + count,
                            audioCondRow * MiniMaxH3Modality.Count + MiniMaxH3Modality.Audio));
                        row += count;
                        blockEnd = Math.Max(blockEnd, cursor + block.AudioLatentFrames);
                    }

                    if (block.Kind != MiniMaxH3ReferenceKind.Audio)
                    {
                        if (block.VideoIndex < 0 || block.LatentHeight <= 0 || block.LatentWidth <= 0)
                            throw new ArgumentException(
                                "a visual reference block needs encoded video latents.", nameof(references));
                        var (refH, refW) = SpatialAxes(block.LatentHeight, block.LatentWidth);
                        int frames = Math.Max(1, block.LatentFrames);
                        int count = frames * refH.Length * refW.Length;
                        float videoCursor = cursor;
                        for (int t = 0; t < frames; t++)
                        {
                            foreach (float h in refH)
                                foreach (float w in refW)
                                    Append(videoCursor, h, w);
                            videoCursor += VideoSpan(t);
                        }
                        segments.Add(new MiniMaxH3SequenceSegment(
                            row, row + count, MiniMaxH3SegmentKind.ConditionVideo, block.VideoIndex));
                        spans.Add(new MiniMaxH3ModulationSpan(row, row + count,
                            condRow * MiniMaxH3Modality.Count + MiniMaxH3Modality.Visual));
                        row += count;
                        // A still image costs one unit of timeline no matter how many
                        // tokens it takes; a clip costs its whole duration.
                        blockEnd = block.Kind == MiniMaxH3ReferenceKind.Image
                            ? Math.Max(blockEnd, cursor + 1f)
                            : Math.Max(blockEnd, videoCursor);
                    }

                    cursor = blockEnd;
                }
            }

            // --- conditioning frames ---------------------------------------------
            // Keyframes are ordinary extra tokens at a near-clean timestep, not a mask:
            // a start frame is pinned to the beginning of the timeline and an end frame to
            // one video-span before the end.
            else if (conditionFrames is { Count: > 0 })
            {
                float duration = VideoDuration(shape.LatentFrames);
                for (int index = 0; index < conditionFrames.Count; index++)
                {
                    int frames = Math.Max(1, conditionFrames[index]);
                    bool atEnd = keyframeAtEnd is { Count: > 0 }
                                 && index < keyframeAtEnd.Count && keyframeAtEnd[index];
                    float keyframeT = atEnd ? textLength + duration - FrameRescale : textLength;
                    int count = frames * frameRows;
                    for (int t = 0; t < frames; t++)
                        foreach (float h in hAxis)
                            foreach (float w in wAxis)
                                Append(keyframeT, h, w);
                    segments.Add(new MiniMaxH3SequenceSegment(
                        row, row + count, MiniMaxH3SegmentKind.ConditionVideo, index));
                    spans.Add(new MiniMaxH3ModulationSpan(
                        row, row + count, condRow * MiniMaxH3Modality.Count + MiniMaxH3Modality.Visual));
                    row += count;
                }
            }

            // --- target audio (before video in the sequence) ----------------------
            int audioStart = row;
            int audioTokens = shape.AudioTokenCount;
            AppendAudioPositions(Append, shape.AudioLatentFrames, cursor,
                wAxis.Length > 0 ? wAxis[0] : 0f,
                wAxis.Length > 0 ? wAxis[wAxis.Length - 1] : 0f);
            var audioSpan = new MiniMaxH3ModulationSpan(audioStart, audioStart + audioTokens, audioRow);
            segments.Add(new MiniMaxH3SequenceSegment(
                audioStart, audioStart + audioTokens, MiniMaxH3SegmentKind.TargetAudio));
            spans.Add(new MiniMaxH3ModulationSpan(audioStart, audioStart + audioTokens,
                audioRow * MiniMaxH3Modality.Count + MiniMaxH3Modality.Audio));
            row += audioTokens;

            // --- target video -----------------------------------------------------
            int videoStart = row;
            for (int t = 0; t < shape.LatentFrames; t++)
            {
                foreach (float h in hAxis)
                    foreach (float w in wAxis)
                        Append(cursor, h, w);
                cursor += VideoSpan(t);
            }
            int videoTokens = shape.LatentFrames * frameRows;
            var videoSpan = new MiniMaxH3ModulationSpan(videoStart, videoStart + videoTokens, videoRow);
            segments.Add(new MiniMaxH3SequenceSegment(
                videoStart, videoStart + videoTokens, MiniMaxH3SegmentKind.TargetVideo));
            spans.Add(new MiniMaxH3ModulationSpan(videoStart, videoStart + videoTokens,
                videoRow * MiniMaxH3Modality.Count + MiniMaxH3Modality.Visual));
            row += videoTokens;

            return new MiniMaxH3PackedLayout
            {
                Positions = positions.ToArray(),
                ModulationSpans = spans,
                Segments = segments,
                Timesteps = timesteps.ToArray(),
                TokenCount = row,
                AudioSpan = audioSpan,
                VideoSpan = videoSpan,
            };
        }

        // Stereo is laid out as two consecutive runs of length audioFrames, one per channel,
        // anchored to the left and right edge of the video's width axis so the audio tokens
        // sit spatially "across" the frame rather than at a single point.
        private static void AppendAudioPositions(
            Action<float, float, float> append, int audioFrames, float cursor, float wLow, float wHigh)
        {
            for (int channel = 0; channel < MiniMaxH3Geometry.AudioChannels; channel++)
            {
                float w = channel == 0 ? wLow : wHigh;
                for (int t = 0; t < audioFrames; t++) append(cursor + t, 0f, w);
            }
        }

        private static int AddTimestep(List<float> values, float value)
        {
            for (int i = 0; i < values.Count; i++)
                if (values[i].Equals(value)) return i;
            values.Add(value);
            return values.Count - 1;
        }

        /// <summary>Resolve a timestep in [0,1] onto the AdaLN curve table's grid, returning the
        /// lower index and the interpolation fraction. The checkpoint stores a learned curve
        /// instead of a sinusoidal time embedding, so the "time embedding" is a table lookup.</summary>
        public static (int Lower, int Upper, float Fraction) CurveIndex(float timestep, int gridSize)
        {
            if (gridSize < 2) return (0, 0, 0f);
            double position = Math.Clamp(timestep, 0f, 1f) * (gridSize - 1);
            int lower = Math.Clamp((int)Math.Floor(position), 0, gridSize - 2);
            return (lower, lower + 1, (float)(position - lower));
        }
    }
}
