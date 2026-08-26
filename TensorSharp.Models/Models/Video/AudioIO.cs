// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Audio decoding for video models that take a reference soundtrack.
//
// Deliberately channel-preserving and rate-parameterized, unlike the per-model
// preprocessors elsewhere in the tree that decode straight to 16 kHz mono: a
// joint audio-video model conditions on STEREO at its own sample rate, and
// folding the channels on the way in would throw away half of what it is being
// asked to reference.
using System;
using System.Collections.Generic;
using System.IO;
using NLayer;
using NVorbis;

namespace TensorSharp.Models.Video
{
    /// <summary>Decoded PCM: one plane per channel, float samples in [-1, 1].</summary>
    public sealed class DecodedAudio
    {
        public float[][] Channels { get; init; }
        public int SampleRate { get; init; }
        public int ChannelCount => Channels?.Length ?? 0;
        public int SampleCount => Channels is { Length: > 0 } ? Channels[0].Length : 0;
        public double DurationSeconds => SampleRate > 0 ? SampleCount / (double)SampleRate : 0;
    }

    /// <summary>Reads WAV, MP3 and Ogg Vorbis into planar float PCM.</summary>
    public static class AudioIO
    {
        /// <summary>Decode an audio file, resampled to <paramref name="targetRate"/> and
        /// folded or duplicated to <paramref name="targetChannels"/>.
        ///
        /// <para>A mono source asked for stereo is duplicated rather than left silent
        /// on one side; a multi-channel source asked for fewer is averaged down.</para></summary>
        public static DecodedAudio Decode(string path, int targetRate, int targetChannels)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
            if (!File.Exists(path)) throw new FileNotFoundException($"audio file not found: {path}", path);
            if (targetRate <= 0) throw new ArgumentOutOfRangeException(nameof(targetRate));
            if (targetChannels <= 0) throw new ArgumentOutOfRangeException(nameof(targetChannels));

            string ext = Path.GetExtension(path).ToLowerInvariant();
            DecodedAudio raw = ext switch
            {
                ".wav" or ".wave" => DecodeWav(File.ReadAllBytes(path)),
                ".mp3" => DecodeMp3(path),
                ".ogg" or ".oga" => DecodeOgg(path),
                _ => throw new NotSupportedException(
                    $"audio format '{ext}' is not supported; use .wav, .mp3 or .ogg."),
            };
            return Conform(raw, targetRate, targetChannels);
        }

        /// <summary>Resample and re-channel already-decoded PCM.</summary>
        public static DecodedAudio Conform(DecodedAudio audio, int targetRate, int targetChannels)
        {
            if (audio is not { ChannelCount: > 0, SampleCount: > 0 })
                throw new ArgumentException("no audio samples", nameof(audio));

            float[][] planes = audio.Channels;
            if (audio.ChannelCount != targetChannels)
            {
                var mapped = new float[targetChannels][];
                if (audio.ChannelCount == 1)
                {
                    for (int c = 0; c < targetChannels; c++) mapped[c] = planes[0];
                }
                else if (targetChannels == 1)
                {
                    var mono = new float[audio.SampleCount];
                    for (int i = 0; i < mono.Length; i++)
                    {
                        double sum = 0;
                        for (int c = 0; c < audio.ChannelCount; c++) sum += planes[c][i];
                        mono[i] = (float)(sum / audio.ChannelCount);
                    }
                    mapped[0] = mono;
                }
                else
                {
                    // Take the leading channels and pad by repeating the last one; a
                    // surround downmix is not something to guess at silently.
                    for (int c = 0; c < targetChannels; c++)
                        mapped[c] = planes[Math.Min(c, audio.ChannelCount - 1)];
                }
                planes = mapped;
            }

            if (audio.SampleRate != targetRate)
            {
                var resampled = new float[planes.Length][];
                for (int c = 0; c < planes.Length; c++)
                    resampled[c] = Resample(planes[c], audio.SampleRate, targetRate);
                planes = resampled;
            }
            return new DecodedAudio { Channels = planes, SampleRate = targetRate };
        }

        /// <summary>Linear resampling. Good enough for a conditioning reference, whose
        /// content the model summarizes rather than reproduces sample for sample.</summary>
        private static float[] Resample(float[] samples, int fromRate, int toRate)
        {
            if (fromRate == toRate || samples.Length == 0) return samples;
            long n = (long)Math.Round(samples.LongLength * (double)toRate / fromRate);
            if (n <= 1) return new[] { samples[0] };
            var outp = new float[n];
            double step = (samples.Length - 1) / (double)(n - 1);
            for (long i = 0; i < n; i++)
            {
                double pos = i * step;
                int idx = (int)pos;
                float frac = (float)(pos - idx);
                outp[i] = idx + 1 < samples.Length
                    ? samples[idx] * (1 - frac) + samples[idx + 1] * frac
                    : samples[^1];
            }
            return outp;
        }

        private static DecodedAudio DecodeMp3(string path)
        {
            using var stream = File.OpenRead(path);
            var reader = new MpegFile(stream);
            return Deinterleave(ReadAll(reader.ReadSamples), reader.Channels, reader.SampleRate);
        }

        private static DecodedAudio DecodeOgg(string path)
        {
            using var vorbis = new VorbisReader(path);
            return Deinterleave(ReadAll(vorbis.ReadSamples), vorbis.Channels, vorbis.SampleRate);
        }

        private static float[] ReadAll(Func<float[], int, int, int> read)
        {
            var all = new List<float>();
            var buf = new float[8192];
            int n;
            while ((n = read(buf, 0, buf.Length)) > 0)
                for (int i = 0; i < n; i++) all.Add(buf[i]);
            return all.ToArray();
        }

        private static DecodedAudio Deinterleave(float[] interleaved, int channels, int rate)
        {
            int frames = channels > 0 ? interleaved.Length / channels : 0;
            var planes = new float[channels][];
            for (int c = 0; c < channels; c++)
            {
                planes[c] = new float[frames];
                for (int i = 0; i < frames; i++) planes[c][i] = interleaved[i * channels + c];
            }
            return new DecodedAudio { Channels = planes, SampleRate = rate };
        }

        /// <summary>Parse a RIFF/WAVE file: PCM integer or IEEE float, any channel count.</summary>
        public static DecodedAudio DecodeWav(byte[] data)
        {
            if (data.Length < 12 ||
                System.Text.Encoding.ASCII.GetString(data, 0, 4) != "RIFF" ||
                System.Text.Encoding.ASCII.GetString(data, 8, 4) != "WAVE")
                throw new InvalidDataException("not a RIFF/WAVE file");

            ushort format = 0;
            int channels = 0, rate = 0, bits = 0;
            byte[] payload = null;

            int offset = 12;
            while (offset + 8 <= data.Length)
            {
                string id = System.Text.Encoding.ASCII.GetString(data, offset, 4);
                int size = BitConverter.ToInt32(data, offset + 4);
                int end = (int)Math.Min((long)offset + 8 + size, data.Length);
                if (id == "fmt " && end - offset - 8 >= 16)
                {
                    format = BitConverter.ToUInt16(data, offset + 8);
                    channels = BitConverter.ToUInt16(data, offset + 10);
                    rate = BitConverter.ToInt32(data, offset + 12);
                    bits = BitConverter.ToUInt16(data, offset + 22);
                    // WAVE_FORMAT_EXTENSIBLE hides the real format in its sub-GUID.
                    if (format == 0xFFFE && end - offset - 8 >= 26)
                        format = BitConverter.ToUInt16(data, offset + 32);
                }
                else if (id == "data")
                {
                    payload = new byte[end - offset - 8];
                    Array.Copy(data, offset + 8, payload, 0, payload.Length);
                }
                offset += 8 + size;
                if (size % 2 != 0) offset++;   // chunks are word-aligned
            }

            if (channels <= 0 || rate <= 0) throw new InvalidDataException("WAV has no fmt chunk");
            if (payload == null) throw new InvalidDataException("WAV has no data chunk");
            if (format != 1 && format != 3)
                throw new NotSupportedException($"unsupported WAV format tag {format}; expected PCM or float.");

            int bytesPerSample = bits / 8;
            if (bytesPerSample <= 0) throw new InvalidDataException("WAV declares zero bits per sample");
            int frames = payload.Length / (bytesPerSample * channels);
            var planes = new float[channels][];
            for (int c = 0; c < channels; c++) planes[c] = new float[frames];

            for (int i = 0; i < frames; i++)
                for (int c = 0; c < channels; c++)
                {
                    int off = (i * channels + c) * bytesPerSample;
                    planes[c][i] = format switch
                    {
                        3 when bits == 32 => BitConverter.ToSingle(payload, off),
                        3 when bits == 64 => (float)BitConverter.ToDouble(payload, off),
                        1 when bits == 8 => (payload[off] - 128) / 128f,
                        1 when bits == 16 => BitConverter.ToInt16(payload, off) / 32768f,
                        1 when bits == 24 => ((payload[off] | (payload[off + 1] << 8)
                                               | ((sbyte)payload[off + 2] << 16))) / 8388608f,
                        1 when bits == 32 => BitConverter.ToInt32(payload, off) / 2147483648f,
                        _ => throw new NotSupportedException($"unsupported WAV depth {bits} bits"),
                    };
                }
            return new DecodedAudio { Channels = planes, SampleRate = rate };
        }
    }
}
