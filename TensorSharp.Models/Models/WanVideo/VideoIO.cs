// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MP4 writing for generated video. Preference order:
//   1. ffmpeg on PATH — H.264 yuv420p, the container every browser plays.
//   2. OpenCvSharp VideoWriter 'avc1' — H.264 via the OS codec (Media Foundation
//      on Windows; often unavailable on Linux images).
//   3. OpenCvSharp VideoWriter 'mp4v' — MPEG-4 Part 2. Plays in VLC/ffmpeg-based
//      players but NOT in browsers, so callers that need in-browser playback
//      should check the returned codec.
using System;
using System.Diagnostics;
using System.IO;
using OpenCvSharp;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.WanVideo
{
    public static class VideoIO
    {
        /// <summary>Write frames as an MP4. Returns the codec actually used
        /// ("h264" or "mp4v").</summary>
        public static string SaveMp4(string path, RgbImage[] frames, int fps)
        {
            if (frames == null || frames.Length == 0)
                throw new ArgumentException("no frames to save", nameof(frames));
            if (fps <= 0) fps = 16;

            string full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full) ?? ".");

            if (TrySaveWithFfmpeg(full, frames, fps)) return "h264";
            if (TrySaveWithOpenCv(full, frames, fps, "avc1"))
            {
                Console.WriteLine("  [video] encoded H.264 via the OS encoder at its DEFAULT bitrate, visibly " +
                                  "lower quality than the frames the model produced. Install ffmpeg on PATH " +
                                  "(or set TS_FFMPEG=<path-to-ffmpeg.exe>) for near-lossless output.");
                return "h264";
            }
            if (TrySaveWithOpenCv(full, frames, fps, "mp4v"))
            {
                Console.WriteLine("  [video] H.264 unavailable (no ffmpeg on PATH, no OS encoder); wrote MPEG-4 " +
                                  "Part 2 — plays in VLC but not in browsers. Install ffmpeg for browser-ready MP4s.");
                return "mp4v";
            }
            throw new InvalidOperationException(
                "Could not encode MP4: no ffmpeg on PATH and OpenCV VideoWriter failed to open. " +
                "Install ffmpeg (https://ffmpeg.org) to enable video export.");
        }

        /// <summary>Encode to an in-memory MP4 (for server responses). Returns null codec info
        /// via <paramref name="codec"/> as in <see cref="SaveMp4"/>.</summary>
        public static byte[] EncodeMp4(RgbImage[] frames, int fps, out string codec)
        {
            string tmp = Path.Combine(Path.GetTempPath(), $"tswan-{Guid.NewGuid():N}.mp4");
            try
            {
                codec = SaveMp4(tmp, frames, fps);
                return File.ReadAllBytes(tmp);
            }
            finally
            {
                try { if (File.Exists(tmp)) File.Delete(tmp); } catch { /* best effort */ }
            }
        }

        private static bool TrySaveWithFfmpeg(string path, RgbImage[] frames, int fps)
        {
            string ffmpeg = FindFfmpeg();
            if (ffmpeg == null) return false;

            string tmpDir = Path.Combine(Path.GetTempPath(), $"tswan-frames-{Guid.NewGuid():N}");
            Directory.CreateDirectory(tmpDir);
            try
            {
                for (int i = 0; i < frames.Length; i++)
                    ImageIO.SavePng(Path.Combine(tmpDir, $"f{i:D5}.png"), frames[i]);

                var psi = new ProcessStartInfo
                {
                    FileName = ffmpeg,
                    Arguments = $"-y -loglevel error -framerate {fps} -i \"{Path.Combine(tmpDir, "f%05d.png")}\" " +
                                $"-c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p -movflags +faststart \"{path}\"",
                    UseShellExecute = false,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };
                using var proc = Process.Start(psi);
                string err = proc.StandardError.ReadToEnd();
                proc.WaitForExit();
                if (proc.ExitCode != 0)
                {
                    Console.WriteLine($"  [video] ffmpeg failed ({proc.ExitCode}): {err.Trim()}");
                    return false;
                }
                return File.Exists(path);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [video] ffmpeg path failed: {ex.Message}");
                return false;
            }
            finally
            {
                try { Directory.Delete(tmpDir, recursive: true); } catch { /* best effort */ }
            }
        }

        private static string FindFfmpeg()
        {
            string env = Environment.GetEnvironmentVariable("TS_FFMPEG");
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            string exe = OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg";
            foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
            {
                try
                {
                    if (!string.IsNullOrWhiteSpace(dir) && File.Exists(Path.Combine(dir, exe)))
                        return Path.Combine(dir, exe);
                }
                catch { /* malformed PATH entry */ }
            }
            // A copy dropped next to the app (or in an ffmpeg/bin subfolder) also counts.
            string baseDir = AppContext.BaseDirectory;
            foreach (string cand in new[]
            {
                Path.Combine(baseDir, exe),
                Path.Combine(baseDir, "ffmpeg", exe),
                Path.Combine(baseDir, "ffmpeg", "bin", exe),
            })
            {
                if (File.Exists(cand)) return cand;
            }
            return null;
        }

        private static bool TrySaveWithOpenCv(string path, RgbImage[] frames, int fps, string fourcc)
        {
            try
            {
                int w = frames[0].Width, h = frames[0].Height;
                // On Windows, go straight to Media Foundation: OpenCV's default backend
                // order tries FFMPEG first, whose H.264 encoder needs a Cisco openh264
                // DLL that is not bundled — it fails with loud stderr errors before the
                // MSMF fallback silently writes the actual file. Selecting MSMF up
                // front produces the same (valid) H.264 without the error spew.
                using (var writer = OperatingSystem.IsWindows()
                    ? new VideoWriter(path, VideoCaptureAPIs.MSMF, FourCC.FromString(fourcc), fps, new Size(w, h))
                    : new VideoWriter(path, FourCC.FromString(fourcc), fps, new Size(w, h)))
                {
                    if (!writer.IsOpened()) return false;
                    using var mat = new Mat(h, w, MatType.CV_8UC3);
                    foreach (var frame in frames)
                    {
                        WriteFrameBgr(mat, frame);
                        writer.Write(mat);
                    }
                }
                // OpenCV can "successfully" write a broken stream when the codec half
                // initializes (observed with the bundled openh264 stub). Trust only a
                // file that reads back.
                using (var check = new VideoCapture(path))
                {
                    using var probe = new Mat();
                    if (!check.IsOpened() || !check.Read(probe) || probe.Empty())
                    {
                        Console.WriteLine($"  [video] OpenCV VideoWriter ({fourcc}) produced an unreadable file; trying the next codec.");
                        try { File.Delete(path); } catch { /* best effort */ }
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [video] OpenCV VideoWriter ({fourcc}) failed: {ex.Message}");
                return false;
            }
        }

        private static unsafe void WriteFrameBgr(Mat mat, RgbImage frame)
        {
            int w = frame.Width, h = frame.Height;
            var px = frame.Pixels;   // interleaved RGB in [0,1]
            byte* dst = (byte*)mat.Data;
            long step = mat.Step();
            for (int y = 0; y < h; y++)
            {
                byte* row = dst + y * step;
                int src = y * w * 3;
                for (int x = 0; x < w; x++)
                {
                    row[x * 3 + 0] = ToByte(px[src + x * 3 + 2]);
                    row[x * 3 + 1] = ToByte(px[src + x * 3 + 1]);
                    row[x * 3 + 2] = ToByte(px[src + x * 3 + 0]);
                }
            }
        }

        private static byte ToByte(float v)
        {
            int i = (int)(v * 255f + 0.5f);
            return (byte)(i < 0 ? 0 : i > 255 ? 255 : i);
        }
    }
}
