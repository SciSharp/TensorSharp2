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
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
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
        ///
        /// <para>Frames land in a fresh temp directory as <c>frame_0001.png</c>, ... —
        /// the form the CLI wants, where the returned full paths are handed straight to
        /// the image processor. A caller that must SERVE the frames (the web upload
        /// endpoint) has to place them somewhere addressable instead; see the
        /// <see cref="ExtractVideoFrames(string, string, string, int, double)"/>
        /// overload.</para>
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
            string tempDir = Path.Combine(Path.GetTempPath(), $"frames_{Guid.NewGuid():N}");
            return ExtractVideoFrames(videoPath, tempDir, DefaultFramePrefix, maxFrames, fps);
        }

        /// <summary>Default base name for extracted frames, giving <c>frame_0001.png</c>.</summary>
        internal const string DefaultFramePrefix = "frame";

        /// <summary>
        /// <see cref="ExtractVideoFrames(string, int, double)"/> writing into a caller-chosen
        /// directory under a caller-chosen base name, so the frames land somewhere the caller
        /// can address them.
        ///
        /// <para>This is what the web upload endpoint needs: a chat attachment is referenced by
        /// its bare file name and resolved against the upload directory, and the same name is
        /// what <c>/uploads/&lt;name&gt;</c> serves. Frames written to a private temp directory
        /// satisfy neither — the reference does not resolve, the thumbnail 404s, and the bytes
        /// escape both the upload quota and its TTL sweep. Naming them after the upload's own
        /// GUID also keeps two clips from colliding on <c>frame_0001.png</c>.</para>
        /// </summary>
        /// <param name="videoPath">Path to the source video file.</param>
        /// <param name="outputDirectory">Directory the PNGs are written to (created if missing).</param>
        /// <param name="namePrefix">Base name for the emitted files (sanitized); frames are <c>{prefix}_0001.png</c>, ...</param>
        /// <param name="maxFrames">Optional cap on the number of frames. <c>&lt;= 0</c> resolves from <c>VIDEO_MAX_FRAMES</c>.</param>
        /// <param name="fps">Sampling rate in frames per second of video. <c>&lt;= 0</c> resolves from <c>VIDEO_SAMPLE_FPS</c>.</param>
        public static List<string> ExtractVideoFrames(
            string videoPath, string outputDirectory, string namePrefix,
            int maxFrames = 0, double fps = 0.0)
        {
            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentNullException(nameof(videoPath));
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            if (maxFrames <= 0)
                maxFrames = GetConfiguredMaxVideoFrames();
            if (fps <= 0)
                fps = GetConfiguredVideoSampleFps();

            Directory.CreateDirectory(outputDirectory);
            string prefix = SanitizeName(namePrefix);

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

            var wanted = new List<int>(selectedPositions.Count);
            foreach (int pos in selectedPositions)
                wanted.Add(candidateFrames[pos]);

            return DecodeAndEncodeFrames(capture, wanted, outputDirectory, prefix);
        }

        /// <summary>
        /// Decoding is strictly sequential (one <see cref="VideoCapture"/>, one reused
        /// <see cref="Mat"/>), but PNG encoding is not: deflate + Adler-32 + the write cost
        /// about as much as the decode itself and depend on nothing but their own frame. Each
        /// decoded frame is therefore copied out to a private scanline buffer on this thread
        /// and encoded on the pool while the next frame decodes, with a semaphore bounding how
        /// many raw frames are in flight so a long clip cannot balloon the heap.
        /// </summary>
        private static List<string> DecodeAndEncodeFrames(
            VideoCapture capture, List<int> wantedFrames, string outputDirectory, string prefix)
        {
            var encodes = new List<Task>();
            var frames = new List<string>();
            SemaphoreSlim slots = null;

            try
            {
                using (var mat = new Mat())
                {
                    // Walking forward with Grab() skips a frame for a fraction of the cost of a
                    // decode, while Set(PosFrames) pays a keyframe seek plus decode-forward that
                    // is roughly constant in distance. Below the crossover, stepping is both
                    // cheaper and exact — it never depends on the container's index being
                    // trustworthy. Above it, seeking wins and is the only sane option on a long
                    // clip sampled sparsely. The two are never mixed within one extraction: a
                    // seek that lands off-by-a-frame would then silently skew every subsequent
                    // step.
                    bool stepForward = MaxGap(wantedFrames) <= SequentialStepMaxGap;
                    int cursor = 0;

                    foreach (int frameIdx in wantedFrames)
                    {
                        if (stepForward)
                        {
                            bool exhausted = false;
                            while (cursor < frameIdx)
                            {
                                if (!capture.Grab()) { exhausted = true; break; }
                                cursor++;
                            }
                            if (exhausted)
                                break;
                        }
                        else
                        {
                            capture.Set(VideoCaptureProperties.PosFrames, frameIdx);
                        }

                        if (!capture.Read(mat) || mat.Empty())
                            break;
                        cursor = frameIdx + 1;

                        int width = mat.Cols, height = mat.Rows;
                        if (slots == null)
                        {
                            // The frame size is only known once one has been decoded, and it
                            // is what the in-flight bound is priced against.
                            int limit = InFlightLimit(width, height);
                            slots = new SemaphoreSlim(limit, limit);
                        }

                        string framePath = Path.Combine(outputDirectory, $"{prefix}_{frames.Count + 1:D4}.png");
                        byte[] scanlines = BuildRgbaScanlines(mat);
                        frames.Add(framePath);

                        slots.Wait();
                        encodes.Add(Task.Run(() =>
                        {
                            try { File.WriteAllBytes(framePath, EncodePng(scanlines, width, height)); }
                            finally { slots.Release(); }
                        }));
                    }
                }

                try
                {
                    Task.WaitAll(encodes.ToArray());
                }
                catch (AggregateException ex)
                {
                    // A half-written frame is worse than none: the caller would hand the model a
                    // truncated PNG. Drop everything this call produced and surface the cause.
                    foreach (string path in frames)
                    {
                        try { File.Delete(path); } catch { /* best effort */ }
                    }
                    ExceptionDispatchInfo.Capture(ex.InnerException ?? ex).Throw();
                    throw;
                }

                return frames;
            }
            finally
            {
                slots?.Dispose();
            }
        }

        /// <summary>
        /// How many frames may be awaiting encoding at once. One per core keeps the pool
        /// fed, but each in-flight frame holds a <c>width * height * 4</c> scanline buffer,
        /// and that term is the one that runs away: at 4K it is 33 MB a frame, so a
        /// core-count-only bound would put a third of a gigabyte on the heap for a clip
        /// whose frames are 8x bigger than 1080p. Cap by a memory budget as well, so the
        /// footprint stays flat in resolution and only the parallelism gives way.
        /// </summary>
        private static int InFlightLimit(int width, int height)
        {
            long frameBytes = Math.Max(1L, (long)width * height * 4);
            long affordable = InFlightScanlineBudgetBytes / frameBytes;
            int byBudget = (int)Math.Min(int.MaxValue, Math.Max(1, affordable));
            return Math.Max(1, Math.Min(Environment.ProcessorCount, byBudget));
        }

        /// <summary>Heap budget for scanline buffers awaiting PNG encoding. Sized so even a
        /// 4K clip keeps several frames in flight while staying well inside a server's
        /// working set.</summary>
        private const long InFlightScanlineBudgetBytes = 256L * 1024 * 1024;

        /// <summary>
        /// Largest step between consecutive wanted frames (counting the first from frame 0).
        /// Governs whether stepping forward or seeking is the cheaper way to reach them.
        /// </summary>
        private static int MaxGap(List<int> wantedFrames)
        {
            int max = 0, previous = 0;
            foreach (int frameIdx in wantedFrames)
            {
                int gap = frameIdx - previous;
                if (gap > max) max = gap;
                previous = frameIdx;
            }
            return max;
        }

        /// <summary>
        /// Frame distance past which seeking beats stepping.
        ///
        /// <para>Measured decode-only cost per sampled frame, three clips, gaps 1..96:
        /// stepping wins by 1.9x on 768x576 H.264 and 1.8x on a 3000-frame 640x480 H.264
        /// at a gap of 12, and the crossover sits at a gap of 24-32 for both. The binding
        /// constraint is all-intra footage (MJPEG), where every frame is a keyframe and a
        /// seek has nothing to decode forward over: there the crossover falls to ~12. A
        /// gap of 12 is therefore the largest value that is still a clear win on the
        /// inter-coded clips people actually upload while costing all-intra footage at
        /// most 6%. Sampling at the default 1 fps puts the gap at the source frame rate —
        /// past the threshold, onto the seek path — which the same measurements show is
        /// within a few percent of stepping either way.</para>
        /// </summary>
        private const int SequentialStepMaxGap = 12;

        private static string SanitizeName(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return DefaultFramePrefix;

            name = Path.GetFileNameWithoutExtension(name);
            var sb = new System.Text.StringBuilder(name.Length);
            foreach (char c in name)
                sb.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');

            string cleaned = sb.ToString().Trim('_');
            return cleaned.Length == 0 ? DefaultFramePrefix : cleaned;
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
            File.WriteAllBytes(path, EncodePng(BuildRgbaScanlines(mat), mat.Cols, mat.Rows));
        }

        /// <summary>
        /// Copies a decoded BGR(A) frame out into PNG scanline form — RGBA rows each
        /// prefixed with a filter-type byte — severing every tie to the
        /// <see cref="Mat"/>, which the decode loop immediately reuses for the next
        /// frame. Must run on the thread that owns the capture; everything downstream
        /// of it is pure computation over this buffer.
        /// </summary>
        private static byte[] BuildRgbaScanlines(Mat mat)
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

            return rawRows;
        }

        /// <summary>Assembles a complete PNG file from RGBA scanlines. Thread-safe: it
        /// touches nothing but its arguments.</summary>
        private static byte[] EncodePng(byte[] rawRows, int width, int height)
        {
            byte[] compressed;
            using (var ms = new MemoryStream())
            {
                ms.WriteByte(0x78); // zlib header
                ms.WriteByte(0x01);
                using (var deflate = new DeflateStream(ms, CompressionLevel.Fastest, true))
                    deflate.Write(rawRows, 0, rawRows.Length);

                uint adler = Adler32(rawRows);
                ms.WriteByte((byte)(adler >> 24));
                ms.WriteByte((byte)(adler >> 16));
                ms.WriteByte((byte)(adler >> 8));
                ms.WriteByte((byte)adler);
                compressed = ms.ToArray();
            }

            using var png = new MemoryStream(compressed.Length + 128);
            png.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
            WritePngChunk(png, "IHDR", BuildIHDR(width, height));
            WritePngChunk(png, "IDAT", compressed);
            WritePngChunk(png, "IEND", Array.Empty<byte>());
            return png.ToArray();
        }

        /// <summary>
        /// Adler-32 over the raw scanlines, for the zlib trailer.
        ///
        /// <para>Reducing modulo 65521 on every byte costs two divisions per byte, which on a
        /// 1080p frame is more time than the deflate that follows it. The sums instead run
        /// unreduced for <see cref="AdlerNmax"/> bytes — the largest block for which
        /// <c>b</c> provably cannot overflow 32 bits — and are reduced once per block, the
        /// standard zlib formulation. The result is bit-identical.</para>
        /// </summary>
        private static uint Adler32(byte[] data)
        {
            const uint Base = 65521;
            uint a = 1, b = 0;
            int offset = 0, remaining = data.Length;

            while (remaining > 0)
            {
                int block = remaining < AdlerNmax ? remaining : AdlerNmax;
                remaining -= block;
                int end = offset + block;
                for (; offset < end; offset++)
                {
                    a += data[offset];
                    b += a;
                }
                a %= Base;
                b %= Base;
            }

            return (b << 16) | a;
        }

        /// <summary>Largest byte count for which the unreduced Adler-32 <c>b</c> accumulator
        /// stays inside 32 bits; the constant zlib uses.</summary>
        private const int AdlerNmax = 5552;

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

