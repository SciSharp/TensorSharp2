// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Backend PARITY for the Wan 2.1 video VAE decode. Decodes a deterministic latent
// of a configurable size and dumps the RGB planes, so the same binary can be run
// once per backend and the dumps diffed. ggml backend init is process-global,
// hence one backend per invocation.
//
// Why: Wan renders a uniformly BLACK frame on ggml_metal at small spatial sizes
// (32x32 latent -> 256x256 px) while ggml_cpu renders a picture from the SAME
// latent — the denoise loop is provably fine (final-latent cosine 0.993 between
// the two backends), so the fault is inside the VAE decode. A 60x60 latent
// (480x480 px) decodes correctly on both.
//
// Opt in with TS_WAN_VAE_PARITY_VAE=<wan_2.1_vae.safetensors>
//             TS_WAN_VAE_PARITY_OUT=<dump.bin>
//            [TS_WAN_VAE_PARITY_BACKEND=metal|cpu|cuda]
//            [TS_WAN_VAE_PARITY_LH=32] [TS_WAN_VAE_PARITY_LW=32] [TS_WAN_VAE_PARITY_T=1]

using System;
using System.IO;
using TensorSharp.GGML;
using TensorSharp.Models.WanVideo;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class WanVaeBackendParityTests
{
    private readonly ITestOutputHelper _output;
    public WanVaeBackendParityTests(ITestOutputHelper output) { _output = output; }

    [ModelFact("TS_WAN_VAE_PARITY_VAE")]
    public void DumpVaeDecodeForBackendParity()
    {
        string vaePath = Environment.GetEnvironmentVariable("TS_WAN_VAE_PARITY_VAE");
        string outPath = Environment.GetEnvironmentVariable("TS_WAN_VAE_PARITY_OUT");
        if (string.IsNullOrEmpty(vaePath) || string.IsNullOrEmpty(outPath) || !File.Exists(vaePath))
        {
            _output.WriteLine("[wan-vae-parity] env unset; skipping");
            return;
        }

        var backend = (Environment.GetEnvironmentVariable("TS_WAN_VAE_PARITY_BACKEND") ?? "cpu").ToLowerInvariant() switch
        {
            "metal" => GgmlBackendType.Metal,
            "cuda" => GgmlBackendType.Cuda,
            _ => GgmlBackendType.Cpu,
        };
        GgmlBasicOps.EnsureBackendAvailable(backend);

        int lh = ReadInt("TS_WAN_VAE_PARITY_LH", 32);
        int lw = ReadInt("TS_WAN_VAE_PARITY_LW", 32);
        int t = ReadInt("TS_WAN_VAE_PARITY_T", 1);
        const int zdim = 16;   // Wan 2.1 latent channels

        var latent = new float[(long)zdim * t * lh * lw];
        // TS_WAN_VAE_PARITY_LATENT lets a REAL denoised latent (the pipeline's
        // latent_final.bin) be replayed instead of synthetic noise — activation
        // magnitudes differ enough between the two to expose different kernels.
        string latentFile = Environment.GetEnvironmentVariable("TS_WAN_VAE_PARITY_LATENT");
        if (!string.IsNullOrEmpty(latentFile) && File.Exists(latentFile))
        {
            byte[] raw = File.ReadAllBytes(latentFile);
            if (raw.Length != latent.Length * sizeof(float))
                throw new InvalidOperationException(
                    $"latent file has {raw.Length / 4} floats, expected {latent.Length} for {zdim}x{t}x{lh}x{lw}");
            Buffer.BlockCopy(raw, 0, latent, 0, raw.Length);
            _output.WriteLine($"[wan-vae-parity] latent loaded from {latentFile}");
        }
        else
        {
            Fill(latent, 0xA11CEu);
        }

        using var vae = new WanVae(vaePath);
        var frames = vae.Decode(latent, t, lh, lw);
        _output.WriteLine($"[wan-vae-parity] backend={backend} latent={zdim}x{t}x{lh}x{lw} frames={frames.Length} " +
                          $"{frames[0].Width}x{frames[0].Height}");

        // Flatten every frame's pixels into one float dump for the diff.
        int perFrame = frames[0].Width * frames[0].Height * 3;
        var all = new float[(long)frames.Length * perFrame];
        long o = 0;
        double sum = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var f in frames)
        {
            Array.Copy(f.Pixels, 0, all, o, perFrame);
            o += perFrame;
            foreach (float px in f.Pixels) { sum += px; if (px < min) min = px; if (px > max) max = px; }
        }
        _output.WriteLine($"[wan-vae-parity] pixels mean={sum / all.LongLength:F5} min={min:F5} max={max:F5}");

        var bytes = new byte[all.Length * sizeof(float)];
        Buffer.BlockCopy(all, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(outPath, bytes);

        // A VAE that decodes every pixel to the same value produced no image at
        // all - that is the ggml_metal failure this test exists to catch, and it
        // is invisible to a cosine check that only looks at magnitudes.
        Assert.True(max - min > 1e-3f,
            $"VAE decode produced a CONSTANT image (all pixels ~= {min:F5}); the decode collapsed");
    }

    private static int ReadInt(string name, int fallback)
    {
        string v = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrWhiteSpace(v) && int.TryParse(v, out int p) && p > 0 ? p : fallback;
    }

    /// <summary>Deterministic pseudo-random fill in [-2, 2] — the range a denoised
    /// Wan latent actually occupies.</summary>
    private static void Fill(float[] dst, uint seed)
    {
        uint s = seed;
        for (int i = 0; i < dst.Length; i++)
        {
            s = s * 1664525u + 1013904223u;
            dst[i] = (((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * 2.0f;
        }
    }
}
