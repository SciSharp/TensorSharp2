// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace TensorSharp.Models
{
    public static class MediaHelper
    {
        /// <summary>
        /// Frames sampled per second of video when no explicit rate is supplied.
        /// Overridable via the <c>VIDEO_SAMPLE_FPS</c> environment variable.
        /// </summary>
        public const double DefaultVideoSampleFps = 1.0;

        /// <summary>
        /// Default upper bound on the number of extracted frames. <c>0</c> means
        /// "no cap": extraction is purely time-based at the sampling fps. Set the
        /// <c>VIDEO_MAX_FRAMES</c> environment variable to a positive value to
        /// bound long videos.
        /// </summary>
        public const int DefaultVideoMaxFrames = 0;

        /// <summary>
        /// Resolves the sampling rate (frames per second of video) from the
        /// <c>VIDEO_SAMPLE_FPS</c> environment variable, falling back to
        /// <see cref="DefaultVideoSampleFps"/> when unset or invalid.
        /// </summary>
        public static double GetConfiguredVideoSampleFps(double fallback = DefaultVideoSampleFps)
        {
            string raw = Environment.GetEnvironmentVariable("VIDEO_SAMPLE_FPS");
            if (!string.IsNullOrWhiteSpace(raw) &&
                double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) &&
                parsed > 0)
            {
                return parsed;
            }

            return fallback > 0 ? fallback : DefaultVideoSampleFps;
        }

        /// <summary>
        /// Resolves the optional upper bound on extracted frames from the
        /// <c>VIDEO_MAX_FRAMES</c> environment variable. Returns <c>0</c> (no cap)
        /// when unset or invalid so that extraction stays purely time-based.
        /// </summary>
        public static int GetConfiguredMaxVideoFrames(int fallback = DefaultVideoMaxFrames)
        {
            string raw = Environment.GetEnvironmentVariable("VIDEO_MAX_FRAMES");
            if (!string.IsNullOrWhiteSpace(raw) &&
                int.TryParse(raw, out int parsed) &&
                parsed > 0)
            {
                return parsed;
            }

            return fallback > 0 ? fallback : 0;
        }

        /// <summary>
        /// Extracts frames from a video using time-based sampling: one frame every
        /// <c>1 / fps</c> seconds (default 1 fps), so the frame count scales with the
        /// clip's duration. When <paramref name="maxFrames"/> resolves to a positive
        /// value (via the <c>VIDEO_MAX_FRAMES</c> env var or an explicit argument) it
        /// acts as an upper bound, evenly down-selecting; otherwise every sampled
        /// frame is kept.
        /// </summary>
        /// <param name="videoPath">Path to the source video file.</param>
        /// <param name="maxFrames">
        /// Optional cap on the number of frames. <c>&lt;= 0</c> resolves from
        /// <c>VIDEO_MAX_FRAMES</c> (default: no cap).
        /// </param>
        /// <param name="fps">
        /// Sampling rate in frames per second of video. <c>&lt;= 0</c> resolves from
        /// <c>VIDEO_SAMPLE_FPS</c> (default: 1 fps).
        /// </param>
        public static List<string> ExtractVideoFrames(string videoPath, int maxFrames = 0, double fps = 0.0)
        {
            if (maxFrames <= 0)
                maxFrames = GetConfiguredMaxVideoFrames();
            if (fps <= 0)
                fps = GetConfiguredVideoSampleFps();

            string tempDir = Path.Combine(Path.GetTempPath(), $"frames_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);

            using var capture = new VideoCapture(videoPath);
            if (!capture.IsOpened())
                throw new Exception($"Failed to open video file: {videoPath}");

            double videoFps = capture.Get(VideoCaptureProperties.Fps);
            int totalFrames = (int)capture.Get(VideoCaptureProperties.FrameCount);
            if (videoFps <= 0 || totalFrames <= 0)
                throw new Exception($"Invalid video: fps={videoFps}, frames={totalFrames}");

            // Time-based candidate sampling: pick one frame every (videoFps / fps)
            // source frames, i.e. one frame per (1 / fps) seconds of wall-clock time.
            int frameInterval = Math.Max(1, (int)Math.Round(videoFps / fps));
            var candidateFrames = new List<int>();
            for (int frameIdx = 0; frameIdx < totalFrames; frameIdx += frameInterval)
                candidateFrames.Add(frameIdx);

            // Keep every time-sampled frame by default. VIDEO_MAX_FRAMES (when > 0)
            // is an optional upper bound that evenly down-selects to protect against
            // runaway frame counts / context blow-up on very long clips.
            List<int> selectedPositions;
            if (maxFrames > 0 && candidateFrames.Count > maxFrames)
            {
                selectedPositions = SelectEvenlySpacedIndices(candidateFrames.Count, maxFrames);
            }
            else
            {
                selectedPositions = new List<int>(candidateFrames.Count);
                for (int i = 0; i < candidateFrames.Count; i++)
                    selectedPositions.Add(i);
            }

            var frames = new List<string>();
            using var mat = new Mat();

            foreach (int pos in selectedPositions)
            {
                int frameIdx = candidateFrames[pos];

                capture.Set(VideoCaptureProperties.PosFrames, frameIdx);
                if (!capture.Read(mat) || mat.Empty())
                    break;

                string framePath = Path.Combine(tempDir, $"frame_{frames.Count + 1:D4}.png");
                SaveMatAsPng(mat, framePath);
                frames.Add(framePath);
            }

            return frames;
        }

        /// <summary>Sample a video onto a fixed frame rate, returning the extracted
        /// frame files and the rate the source carried.
        ///
        /// <para>Unlike <see cref="ExtractVideoFrames"/>, which samples sparsely for a
        /// vision-language model, this reproduces a clip on a target timeline: frame
        /// <c>i</c> of the result is the source frame at <c>i * sourceFps /
        /// targetFps</c>, so a 30 fps source played onto a 24 fps grid holds frames
        /// rather than dropping content. A generative video model conditions on
        /// motion, and sparse sampling would misrepresent it.</para>
        ///
        /// <para>Also accepts a DIRECTORY of image files, which is how reference clips
        /// are usually delivered; the files are taken in sorted order and assumed to be
        /// at <paramref name="targetFps"/> already unless <paramref name="sourceFpsHint"/>
        /// says otherwise.</para></summary>
        /// <param name="maxFrames">Upper bound on the returned count. 0 = no bound.</param>
        public static (List<string> Frames, double SourceFps) ExtractFramesAtRate(
            string videoPath, double targetFps, int maxFrames = 0, double sourceFpsHint = 0)
        {
            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentNullException(nameof(videoPath));
            if (targetFps <= 0) throw new ArgumentOutOfRangeException(nameof(targetFps));

            if (Directory.Exists(videoPath))
            {
                var files = new List<string>(Directory.GetFiles(videoPath));
                files.RemoveAll(f => !IsImageFile(f));
                files.Sort(StringComparer.Ordinal);
                if (files.Count == 0)
                    throw new InvalidOperationException(
                        $"'{videoPath}' contains no image files to use as video frames.");
                double dirFps = sourceFpsHint > 0 ? sourceFpsHint : targetFps;
                return (Resample(files, dirFps, targetFps, maxFrames), dirFps);
            }

            if (!File.Exists(videoPath))
                throw new FileNotFoundException($"video not found: {videoPath}", videoPath);

            using var capture = new VideoCapture(videoPath);
            if (!capture.IsOpened())
                throw new InvalidOperationException($"failed to open video file: {videoPath}");

            double sourceFps = capture.Get(VideoCaptureProperties.Fps);
            if (sourceFpsHint > 0) sourceFps = sourceFpsHint;
            int totalFrames = (int)capture.Get(VideoCaptureProperties.FrameCount);
            if (sourceFps <= 0 || totalFrames <= 0)
                throw new InvalidOperationException(
                    $"invalid video: fps={sourceFps}, frames={totalFrames}");

            int wanted = (int)Math.Round(totalFrames * targetFps / sourceFps);
            if (maxFrames > 0) wanted = Math.Min(wanted, maxFrames);
            wanted = Math.Max(1, wanted);

            string tempDir = Path.Combine(Path.GetTempPath(), $"refvid_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);
            var frames = new List<string>(wanted);
            using var mat = new Mat();
            for (int i = 0; i < wanted; i++)
            {
                int sourceIndex = Math.Min(totalFrames - 1,
                    (int)Math.Floor(i * sourceFps / targetFps));
                capture.Set(VideoCaptureProperties.PosFrames, sourceIndex);
                if (!capture.Read(mat) || mat.Empty()) break;
                string framePath = Path.Combine(tempDir, $"frame_{frames.Count + 1:D5}.png");
                SaveMatAsPng(mat, framePath);
                frames.Add(framePath);
            }
            if (frames.Count == 0)
                throw new InvalidOperationException($"no frames could be read from {videoPath}");
            return (frames, sourceFps);
        }

        private static bool IsImageFile(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            return ext is ".png" or ".jpg" or ".jpeg" or ".bmp" or ".webp";
        }

        // Hold-and-drop resampling of an already-extracted frame list.
        private static List<string> Resample(List<string> files, double fromFps, double toFps, int maxFrames)
        {
            int wanted = (int)Math.Round(files.Count * toFps / fromFps);
            if (maxFrames > 0) wanted = Math.Min(wanted, maxFrames);
            wanted = Math.Max(1, wanted);
            var outp = new List<string>(wanted);
            for (int i = 0; i < wanted; i++)
                outp.Add(files[Math.Min(files.Count - 1, (int)Math.Floor(i * fromFps / toFps))]);
            return outp;
        }

        public static List<int> SelectEvenlySpacedIndices(int count, int maxCount)
        {
            var indices = new List<int>();
            if (count <= 0 || maxCount <= 0)
                return indices;

            if (count <= maxCount)
            {
                for (int i = 0; i < count; i++)
                    indices.Add(i);
                return indices;
            }

            if (maxCount == 1)
            {
                indices.Add(count / 2);
                return indices;
            }

            double step = (double)(count - 1) / (maxCount - 1);
            int previous = -1;
            for (int i = 0; i < maxCount; i++)
            {
                int idx = (int)Math.Round(i * step);
                if (idx <= previous)
                    idx = previous + 1;
                if (idx >= count)
                    idx = count - 1;

                indices.Add(idx);
                previous = idx;
            }

            return indices;
        }

        private static void SaveMatAsPng(Mat mat, string path)
        {
            int width = mat.Cols;
            int height = mat.Rows;
            int channels = mat.Channels();
            int step = (int)mat.Step();

            int rowStride = 1 + width * 4;
            byte[] rawRows = new byte[height * rowStride];

            unsafe
            {
                byte* src = (byte*)mat.Data;
                for (int y = 0; y < height; y++)
                {
                    int dstRowStart = y * rowStride;
                    rawRows[dstRowStart] = 0; // PNG filter: None
                    byte* row = src + y * step;
                    for (int x = 0; x < width; x++)
                    {
                        int dstOff = dstRowStart + 1 + x * 4;
                        int srcOff = x * channels;
                        rawRows[dstOff]     = row[srcOff + 2]; // R (BGR→RGB)
                        rawRows[dstOff + 1] = row[srcOff + 1]; // G
                        rawRows[dstOff + 2] = row[srcOff];     // B
                        rawRows[dstOff + 3] = channels >= 4 ? row[srcOff + 3] : (byte)255;
                    }
                }
            }

            byte[] compressed;
            using (var ms = new MemoryStream())
            {
                ms.WriteByte(0x78); // zlib header
                ms.WriteByte(0x01);
                using (var deflate = new DeflateStream(ms, CompressionLevel.Fastest, true))
                    deflate.Write(rawRows, 0, rawRows.Length);

                uint a = 1, b = 0;
                for (int i = 0; i < rawRows.Length; i++)
                {
                    a = (a + rawRows[i]) % 65521;
                    b = (b + a) % 65521;
                }
                uint adler = (b << 16) | a;
                ms.WriteByte((byte)(adler >> 24));
                ms.WriteByte((byte)(adler >> 16));
                ms.WriteByte((byte)(adler >> 8));
                ms.WriteByte((byte)adler);
                compressed = ms.ToArray();
            }

            using var fs = File.Create(path);
            fs.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
            WritePngChunk(fs, "IHDR", BuildIHDR(width, height));
            WritePngChunk(fs, "IDAT", compressed);
            WritePngChunk(fs, "IEND", Array.Empty<byte>());
        }

        private static byte[] BuildIHDR(int width, int height)
        {
            byte[] ihdr = new byte[13];
            ihdr[0] = (byte)(width >> 24); ihdr[1] = (byte)(width >> 16);
            ihdr[2] = (byte)(width >> 8);  ihdr[3] = (byte)width;
            ihdr[4] = (byte)(height >> 24); ihdr[5] = (byte)(height >> 16);
            ihdr[6] = (byte)(height >> 8);  ihdr[7] = (byte)height;
            ihdr[8] = 8;  // bit depth
            ihdr[9] = 6;  // color type RGBA
            return ihdr;
        }

        private static void WritePngChunk(Stream s, string type, byte[] data)
        {
            byte[] lenBuf = { (byte)(data.Length >> 24), (byte)(data.Length >> 16),
                              (byte)(data.Length >> 8),  (byte)data.Length };
            byte[] typeBuf = System.Text.Encoding.ASCII.GetBytes(type);
            s.Write(lenBuf, 0, 4);
            s.Write(typeBuf, 0, 4);
            if (data.Length > 0) s.Write(data, 0, data.Length);

            uint crc = Crc32Png(typeBuf, data);
            byte[] crcBuf = { (byte)(crc >> 24), (byte)(crc >> 16),
                              (byte)(crc >> 8),  (byte)crc };
            s.Write(crcBuf, 0, 4);
        }

        private static readonly uint[] Crc32Table = BuildCrc32Table();
        private static uint[] BuildCrc32Table()
        {
            var t = new uint[256];
            for (uint n = 0; n < 256; n++)
            {
                uint c = n;
                for (int k = 0; k < 8; k++)
                    c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
                t[n] = c;
            }
            return t;
        }

        private static uint Crc32Png(byte[] type, byte[] data)
        {
            uint crc = 0xFFFFFFFF;
            foreach (byte b in type)
                crc = Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
            foreach (byte b in data)
                crc = Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
            return crc ^ 0xFFFFFFFF;
        }
    }
}

