// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using StbImageSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Covers <see cref="MediaHelper.ExtractVideoFrames(string, string, string, int, double)"/>
/// and the contract the Web UI upload endpoint depends on.
///
/// <para>The regression these pin down: /api/upload used to extract a clip's frames into a
/// private temp directory while returning only their BARE FILE NAMES to the browser. Every
/// downstream consumer resolves a bare name against the upload root, so the next chat turn
/// died on <c>FileNotFoundException: .../uploads/frame_0001.png</c> and the frame thumbnails
/// 404'd. Frames now land in the caller's directory under the upload's own GUID, which also
/// stops two clips from both claiming <c>frame_0001.png</c>.</para>
/// </summary>
public class VideoFrameExtractionTests : IDisposable
{
    private readonly string _baseDir;
    private readonly string _uploadRoot;

    public VideoFrameExtractionTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-video-frames-" + Guid.NewGuid().ToString("N"));
        _uploadRoot = Path.Combine(_baseDir, "uploads");
        Directory.CreateDirectory(_uploadRoot);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    private string WriteClip(string name = "clip.mp4", int frames = 96, double fps = 24.0,
        int width = 128, int height = 96, string fourcc = VideoFixture.InterCoded)
        => VideoFixture.TryWrite(Path.Combine(_baseDir, name), frames, fps, width, height, fourcc);

    // ---- the reported failure ------------------------------------------------

    /// <summary>
    /// The exact shape of the reported bug: take what /api/upload hands the browser for a
    /// video (bare frame names), push it back through the chat request parser the way the
    /// browser does, and require every resolved path to exist. Before the fix the names
    /// resolved into the upload root while the bytes sat in %TEMP%, so this failed on the
    /// first frame.
    /// </summary>
    [VideoFact]
    public void FrameNamesReturnedByUpload_ResolveToRealFilesUnderTheUploadRoot()
    {
        string clip = WriteClip();
        Assert.NotNull(clip);

        // What UploadAsync does: save as a GUID, then extract frames beside it.
        string safeFileName = Guid.NewGuid().ToString("N") + ".mp4";
        List<string> frames = MediaHelper.ExtractVideoFrames(
            clip, _uploadRoot, Path.GetFileNameWithoutExtension(safeFileName), maxFrames: 0, fps: 1.0);
        Assert.NotEmpty(frames);

        // What the browser posts back: the bare names, exactly as the endpoint reports them.
        var imagePaths = frames.Select(Path.GetFileName).ToList();
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "视频中的两个人正在干嘛？", ImagePaths = imagePaths, IsVideo = true },
        };

        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(history, _uploadRoot));
        foreach (string resolved in history[0].ImagePaths)
        {
            Assert.True(File.Exists(resolved), $"resolved attachment does not exist: {resolved}");
            Assert.Equal(Path.GetFullPath(_uploadRoot), Path.GetDirectoryName(resolved));
        }
    }

    /// <summary>
    /// The failure surfaced from the image processor, not the parser, so drive that too:
    /// every extracted frame must survive the vision preprocessor the Gemma 4 path runs.
    /// </summary>
    [VideoFact]
    public void ExtractedFrames_FeedTheVisionImageProcessor()
    {
        string clip = WriteClip(frames: 48);
        Assert.NotNull(clip);

        var frames = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "vid", maxFrames: 0, fps: 1.0);
        Assert.NotEmpty(frames);

        var processor = new Gemma4ImageProcessor();
        foreach (string frame in frames)
        {
            (float[] pixels, int width, int height) = processor.ProcessImage(frame);
            Assert.True(width > 0 && height > 0);
            Assert.Equal(3 * width * height, pixels.Length);
            Assert.All(pixels, p => Assert.False(float.IsNaN(p) || float.IsInfinity(p)));
        }
    }

    // ---- placement and naming ------------------------------------------------

    [VideoFact]
    public void FramesLandInTheRequestedDirectoryUnderTheRequestedPrefix()
    {
        string clip = WriteClip(frames: 72);
        Assert.NotNull(clip);

        var frames = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "abc123", maxFrames: 0, fps: 1.0);

        Assert.NotEmpty(frames);
        Assert.Equal(
            frames.Select((_, i) => $"abc123_{i + 1:D4}.png").ToList(),
            frames.Select(Path.GetFileName).ToList());
        Assert.All(frames, f => Assert.Equal(Path.GetFullPath(_uploadRoot), Path.GetDirectoryName(f)));
        Assert.All(frames, f => Assert.True(File.Exists(f)));
    }

    /// <summary>
    /// Two clips uploaded to the same directory must not overwrite each other — the whole
    /// point of naming frames after the upload's GUID rather than <c>frame_0001.png</c>.
    /// </summary>
    [VideoFact]
    public void TwoClipsInOneDirectoryDoNotCollide()
    {
        string first = WriteClip("first.mp4", frames: 48);
        string second = WriteClip("second.mp4", frames: 48, fps: 24.0, width: 160, height: 120);
        Assert.NotNull(first);
        Assert.NotNull(second);

        var a = MediaHelper.ExtractVideoFrames(first, _uploadRoot, "aaa", maxFrames: 0, fps: 1.0);
        var b = MediaHelper.ExtractVideoFrames(second, _uploadRoot, "bbb", maxFrames: 0, fps: 1.0);

        Assert.Empty(a.Select(Path.GetFileName).Intersect(b.Select(Path.GetFileName)));
        Assert.All(a.Concat(b), f => Assert.True(File.Exists(f)));

        // The second extraction must not have clobbered the first's pixels.
        ImageResult firstFrame = Decode(a[0]);
        Assert.Equal(128, firstFrame.Width);
        Assert.Equal(96, firstFrame.Height);
        ImageResult secondFrame = Decode(b[0]);
        Assert.Equal(160, secondFrame.Width);
        Assert.Equal(120, secondFrame.Height);
    }

    /// <summary>A prefix carrying path separators must not steer frames out of the directory.</summary>
    [VideoFact]
    public void PrefixIsSanitizedSoFramesCannotEscapeTheDirectory()
    {
        string clip = WriteClip(frames: 24);
        Assert.NotNull(clip);

        var frames = MediaHelper.ExtractVideoFrames(
            clip, _uploadRoot, "../../etc/pa sswd", maxFrames: 0, fps: 1.0);

        Assert.NotEmpty(frames);
        Assert.All(frames, f => Assert.Equal(Path.GetFullPath(_uploadRoot), Path.GetDirectoryName(f)));
        // And the sanitized names are still valid upload references.
        var refs = frames.Select(Path.GetFileName).ToList();
        var history = new List<ChatMessage> { new() { Role = "user", ImagePaths = refs } };
        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(history, _uploadRoot));
        Assert.All(history[0].ImagePaths, p => Assert.True(File.Exists(p)));
    }

    /// <summary>The temp-directory overload the CLI uses keeps its historical naming.</summary>
    [VideoFact]
    public void DefaultOverloadStillWritesFrameNNNNIntoAFreshTempDirectory()
    {
        string clip = WriteClip(frames: 72);
        Assert.NotNull(clip);

        var frames = MediaHelper.ExtractVideoFrames(clip, maxFrames: 0, fps: 1.0);
        try
        {
            Assert.NotEmpty(frames);
            Assert.Equal("frame_0001.png", Path.GetFileName(frames[0]));
            string dir = Path.GetDirectoryName(frames[0]);
            Assert.StartsWith("frames_", Path.GetFileName(dir));
            Assert.All(frames, f => Assert.True(File.Exists(f)));
        }
        finally
        {
            try { Directory.Delete(Path.GetDirectoryName(frames[0]), recursive: true); } catch { }
        }
    }

    // ---- image quality -------------------------------------------------------

    /// <summary>
    /// Every frame must be a PNG a decoder accepts, at the source resolution, fully opaque.
    /// The encoder writes PNG by hand (no imgcodecs in the shipped OpenCV runtime), and the
    /// deflate/Adler-32 trailer it emits is exactly what a truncated or mis-checksummed
    /// stream would get wrong.
    /// </summary>
    [VideoFact]
    public void EveryFrameIsAValidOpaquePngAtSourceResolution()
    {
        string clip = WriteClip(frames: 72, width: 160, height: 120);
        Assert.NotNull(clip);

        var frames = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "q", maxFrames: 0, fps: 2.0);
        Assert.NotEmpty(frames);

        foreach (string frame in frames)
        {
            ImageResult img = Decode(frame);
            Assert.Equal(160, img.Width);
            Assert.Equal(120, img.Height);
            Assert.Equal(ColorComponents.RedGreenBlueAlpha, img.Comp);
            for (int i = 3; i < img.Data.Length; i += 4)
                Assert.Equal(255, img.Data[i]);
        }
    }

    /// <summary>
    /// Sampling must actually advance through the clip. The fixture paints each source
    /// frame a different colour, so consecutive extracted frames differing proves the
    /// decode loop is not handing back the same frame repeatedly — the failure mode a
    /// broken seek/step cursor would produce.
    /// </summary>
    [VideoFact]
    public void ConsecutiveFramesComeFromDifferentPointsInTheClip()
    {
        string clip = WriteClip(frames: 96);
        Assert.NotNull(clip);

        var frames = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "adv", maxFrames: 0, fps: 2.0);
        Assert.True(frames.Count >= 4, $"expected several frames, got {frames.Count}");

        var seen = new List<string>();
        foreach (string frame in frames)
        {
            ImageResult img = Decode(frame);
            // Corner pixel: away from the moving white block's start, so it carries the
            // per-frame background colour.
            int last = img.Data.Length - 4;
            seen.Add($"{img.Data[last]},{img.Data[last + 1]},{img.Data[last + 2]}");
        }

        Assert.True(seen.Distinct().Count() > 1,
            "every extracted frame had the same corner colour: sampling never advanced");
    }

    /// <summary>
    /// Stepping and seeking must select the same frames. Extraction switches strategy on
    /// the gap between wanted frames, so a clip sampled either side of that threshold has
    /// to yield the same pixels or the choice would silently change what the model sees.
    /// </summary>
    [VideoFact]
    public void SteppingAndSeekingSelectTheSameFrames()
    {
        // 96 frames at 24 fps. fps=4 -> gap 6 (steps); fps=0.5 -> gap 48 (seeks). Ask for
        // the same 2 frames both ways via the cap, which down-selects to the same indices.
        string clip = WriteClip(frames: 96);
        Assert.NotNull(clip);

        var stepped = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "step", maxFrames: 0, fps: 24.0);
        var sought = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "seek", maxFrames: 0, fps: 0.5);
        Assert.NotEmpty(stepped);
        Assert.NotEmpty(sought);

        // fps=24 keeps every frame (gap 1, stepped); fps=0.5 keeps frames 0 and 48 (gap 48,
        // sought). Frame 0 and frame 48 of the stepped run must match the sought run byte
        // for byte — same source frames, same encoder.
        Assert.True(stepped.Count > 48, $"expected the full clip, got {stepped.Count} frames");
        Assert.Equal(File.ReadAllBytes(stepped[0]), File.ReadAllBytes(sought[0]));
        if (sought.Count > 1)
            Assert.Equal(File.ReadAllBytes(stepped[48]), File.ReadAllBytes(sought[1]));
    }

    /// <summary>All-intra footage takes the other branch of the seek/step choice.</summary>
    [VideoFact]
    public void AllIntraFootageExtractsToo()
    {
        string clip = WriteClip("intra.avi", frames: 60, fps: 30.0, fourcc: VideoFixture.AllIntra);
        if (clip == null)
            return; // MJPEG unavailable on this OpenCV build; the inter-coded cases still cover the path.

        var frames = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "intra", maxFrames: 0, fps: 1.0);

        Assert.NotEmpty(frames);
        Assert.All(frames, f => Assert.True(File.Exists(f)));
        Assert.All(frames, f => Assert.Equal(128, Decode(f).Width));
    }

    // ---- bounds and arguments ------------------------------------------------

    [VideoFact]
    public void MaxFramesCapsTheResultAndStillNumbersContiguously()
    {
        string clip = WriteClip(frames: 240);
        Assert.NotNull(clip);

        var frames = MediaHelper.ExtractVideoFrames(clip, _uploadRoot, "cap", maxFrames: 3, fps: 4.0);

        Assert.Equal(3, frames.Count);
        Assert.Equal(
            new[] { "cap_0001.png", "cap_0002.png", "cap_0003.png" },
            frames.Select(Path.GetFileName).ToArray());
        Assert.All(frames, f => Assert.True(File.Exists(f)));
    }

    [VideoFact]
    public void OutputDirectoryIsCreatedWhenMissing()
    {
        string clip = WriteClip(frames: 24);
        Assert.NotNull(clip);
        string nested = Path.Combine(_baseDir, "not", "there", "yet");

        var frames = MediaHelper.ExtractVideoFrames(clip, nested, "n", maxFrames: 0, fps: 1.0);

        Assert.NotEmpty(frames);
        Assert.All(frames, f => Assert.True(File.Exists(f)));
    }

    [Fact]
    public void NullOrEmptyArgumentsAreRejected()
    {
        Assert.Throws<ArgumentNullException>(() => MediaHelper.ExtractVideoFrames(null, _uploadRoot, "p"));
        Assert.Throws<ArgumentNullException>(() => MediaHelper.ExtractVideoFrames("  ", _uploadRoot, "p"));
        Assert.Throws<ArgumentNullException>(() => MediaHelper.ExtractVideoFrames("clip.mp4", null, "p"));
        Assert.Throws<ArgumentNullException>(() => MediaHelper.ExtractVideoFrames("clip.mp4", "  ", "p"));
    }

    /// <summary>An unreadable file must fail loudly, and must not leave frames behind.</summary>
    [Fact]
    public void NonVideoInputThrowsAndWritesNothing()
    {
        string notAVideo = Path.Combine(_baseDir, "notavideo.mp4");
        File.WriteAllText(notAVideo, "this is not a video");

        Assert.ThrowsAny<Exception>(() => MediaHelper.ExtractVideoFrames(notAVideo, _uploadRoot, "bad"));
        Assert.Empty(Directory.GetFiles(_uploadRoot, "bad_*.png"));
    }

    private static ImageResult Decode(string pngPath) =>
        ImageResult.FromMemory(File.ReadAllBytes(pngPath), ColorComponents.RedGreenBlueAlpha);
}
