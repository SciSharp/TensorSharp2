// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.IO;
using OpenCvSharp;

namespace InferenceWeb.Tests;

/// <summary>
/// Synthesises the short clips the video tests run against, so nothing depends on a
/// checked-in binary fixture or on the operator's own footage.
///
/// <para>Frames carry a large moving block on a per-frame colour, which gives three
/// things the tests need: real entropy (so PNG compression is not degenerate), visibly
/// different consecutive frames (so "did sampling actually advance?" is answerable from
/// the pixels), and a per-frame colour that identifies WHICH source frame an extracted
/// PNG came from.</para>
/// </summary>
internal static class VideoFixture
{
    /// <summary>FourCC for an inter-coded clip (H.264), the shape of a phone/screen recording.</summary>
    public const string InterCoded = "avc1";

    /// <summary>FourCC for an all-intra clip (MJPEG): every frame a keyframe.</summary>
    public const string AllIntra = "MJPG";

    /// <summary>
    /// Writes a clip and returns its path, or null when this machine's OpenCV build
    /// cannot encode video (the gate in <c>TestGates.VideoSkip</c> turns that into a
    /// visible skip rather than a failure).
    /// </summary>
    public static string TryWrite(
        string path, int frames, double fps = 24.0, int width = 128, int height = 96, string fourcc = InterCoded)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
        try
        {
            using (var writer = new VideoWriter(path, FourCC.FromString(fourcc), fps, new Size(width, height)))
            {
                if (!writer.IsOpened())
                    return null;

                using var mat = new Mat(height, width, MatType.CV_8UC3);
                for (int i = 0; i < frames; i++)
                {
                    mat.SetTo(FrameColor(i));
                    Cv2.Rectangle(
                        mat,
                        new Rect((i * 7) % Math.Max(1, width - 24), (i * 5) % Math.Max(1, height - 24), 24, 24),
                        new Scalar(255, 255, 255), thickness: -1);
                    writer.Write(mat);
                }
            }
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return null;
        }

        var info = new FileInfo(path);
        return info.Exists && info.Length > 0 ? path : null;
    }

    /// <summary>The flat colour source frame <paramref name="i"/> is painted with, as BGR.</summary>
    public static Scalar FrameColor(int i) => new(i % 256, (i * 3) % 256, (i * 7) % 256);
}
