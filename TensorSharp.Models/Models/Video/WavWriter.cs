// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Minimal RIFF/WAVE writer for the audio track of a generated video.
//
// Models that generate audio jointly with video hand back planar float PCM. We write
// it as 16-bit little-endian WAV: universally playable, and muxable into the MP4 later
// with a one-line ffmpeg call. Writing a sidecar rather than muxing in-process keeps
// video export working on machines with no encoder installed.
using System;
using System.IO;

namespace TensorSharp.Models.Video
{
    /// <summary>Writes planar float PCM as a 16-bit RIFF/WAVE file.</summary>
    public static class WavWriter
    {
        private const int BitsPerSample = 16;

        /// <summary>Write <paramref name="channels"/> (planar, samples in [-1, 1]) to
        /// <paramref name="path"/>. Channels are interleaved on the way out; a ragged
        /// input is truncated to the shortest channel.</summary>
        public static void Write(string path, float[][] channels, int sampleRate)
        {
            using var stream = File.Create(path);
            Write(stream, channels, sampleRate);
        }

        /// <summary>Encode to a byte array (for serving over HTTP without touching disk).</summary>
        public static byte[] Encode(float[][] channels, int sampleRate)
        {
            using var ms = new MemoryStream();
            Write(ms, channels, sampleRate);
            return ms.ToArray();
        }

        /// <summary>Write planar float PCM to an open stream.</summary>
        public static void Write(Stream stream, float[][] channels, int sampleRate)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            if (channels is null || channels.Length == 0)
                throw new ArgumentException("at least one channel is required", nameof(channels));
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));

            int channelCount = channels.Length;
            int frames = int.MaxValue;
            foreach (var c in channels)
            {
                if (c is null) throw new ArgumentException("channel data must not be null", nameof(channels));
                frames = Math.Min(frames, c.Length);
            }
            if (frames == int.MaxValue) frames = 0;

            int blockAlign = channelCount * (BitsPerSample / 8);
            int dataBytes = frames * blockAlign;

            using var w = new BinaryWriter(stream, System.Text.Encoding.ASCII, leaveOpen: true);
            w.Write(new[] { 'R', 'I', 'F', 'F' });
            w.Write(36 + dataBytes);              // RIFF chunk size
            w.Write(new[] { 'W', 'A', 'V', 'E' });
            w.Write(new[] { 'f', 'm', 't', ' ' });
            w.Write(16);                          // PCM fmt chunk size
            w.Write((short)1);                    // PCM
            w.Write((short)channelCount);
            w.Write(sampleRate);
            w.Write(sampleRate * blockAlign);     // byte rate
            w.Write((short)blockAlign);
            w.Write((short)BitsPerSample);
            w.Write(new[] { 'd', 'a', 't', 'a' });
            w.Write(dataBytes);

            var buffer = new byte[dataBytes];
            int offset = 0;
            for (int i = 0; i < frames; i++)
            {
                for (int c = 0; c < channelCount; c++)
                {
                    // Clamp before scaling: a model can overshoot [-1,1] slightly and
                    // wrapping the int16 would turn that into a loud click.
                    float v = Math.Clamp(channels[c][i], -1f, 1f);
                    short s = (short)Math.Round(v * short.MaxValue, MidpointRounding.AwayFromZero);
                    buffer[offset++] = (byte)(s & 0xFF);
                    buffer[offset++] = (byte)((s >> 8) & 0xFF);
                }
            }
            w.Write(buffer);
            w.Flush();
        }
    }
}
