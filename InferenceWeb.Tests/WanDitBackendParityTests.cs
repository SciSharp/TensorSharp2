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
// Backend PARITY for the Wan DiT: run one forward with deterministic inputs and
// dump v to a file, so the same test binary can be invoked once per backend and
// the two dumps diffed. ggml backend init is process-global, which is why this
// cannot compare two backends inside a single test.
//
// It exists because Wan2.1-T2V-1.3B (the only F16 DiT in the family) rendered a
// uniformly BLACK frame on ggml_metal while ggml_cpu rendered a picture from the
// identical prompt/seed/steps — the quantized 5B/14B/A14B DiTs were all fine.
//
// Opt in with TS_WAN_PARITY_MODEL=<dit.gguf> TS_WAN_PARITY_OUT=<dump.bin>
// [TS_WAN_PARITY_BACKEND=metal|cpu|cuda]. The default latent grid is the small
// T=2, H=8, W=8 fixture; TS_WAN_PARITY_T / _HH / _WW optionally override it
// for allocation-pressure probes without changing the normal parity workload.

using System;
using System.IO;
using TensorSharp.GGML;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class WanDitBackendParityTests
{
    private readonly ITestOutputHelper _output;
    public WanDitBackendParityTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void DumpDitForwardForBackendParity()
    {
        string model = Environment.GetEnvironmentVariable("TS_WAN_PARITY_MODEL");
        string outPath = Environment.GetEnvironmentVariable("TS_WAN_PARITY_OUT");
        if (string.IsNullOrEmpty(model) || string.IsNullOrEmpty(outPath) || !File.Exists(model))
        {
            _output.WriteLine("[wan-parity] TS_WAN_PARITY_MODEL / TS_WAN_PARITY_OUT unset; skipping");
            return;
        }

        var backend = (Environment.GetEnvironmentVariable("TS_WAN_PARITY_BACKEND") ?? "cpu").ToLowerInvariant() switch
        {
            "metal" => GgmlBackendType.Metal,
            "cuda" => GgmlBackendType.Cuda,
            _ => GgmlBackendType.Cpu,
        };
        GgmlBasicOps.EnsureBackendAvailable(backend);

        // Small default problem: 2 latent frames of 8x8 -> 2*4*4 = 32 tokens.
        // The optional dimensions let an opt-in run reproduce production graph
        // allocation pressure with a smaller checkpoint. For example, the
        // 736x544x245 A14B request maps to T=62, Hh=68, Ww=92 (96,968 tokens).
        int T = ReadPositiveInt("TS_WAN_PARITY_T", 2);
        int Hh = ReadPositiveInt("TS_WAN_PARITY_HH", 8);
        int Ww = ReadPositiveInt("TS_WAN_PARITY_WW", 8);
        if (Hh % WanDiT.PatchH != 0 || Ww % WanDiT.PatchW != 0)
            throw new InvalidOperationException(
                $"TS_WAN_PARITY_HH/WW must be divisible by the Wan patch grid " +
                $"{WanDiT.PatchH}x{WanDiT.PatchW}; got {Hh}x{Ww}.");
        int hLen = Hh / WanDiT.PatchH, wLen = Ww / WanDiT.PatchW;
        int seq = checked(T * hLen * wLen);

        using var gguf = new GgufFile(model);
        using var dit = new WanDiT(gguf);
        _output.WriteLine($"[wan-parity] backend={backend} inDim={dit.InDim} inTok={dit.InTok} " +
                          $"outTok={dit.OutTok} latent={T}x{Hh}x{Ww} seq={seq}");

        var x = new float[(long)dit.InTok * seq];
        Fill(x, 0x5EEDu);
        // Context is the zero-padded UMT5 conditioning the DiT expects:
        // [TextDim * 512]. Filling it deterministically exercises cross-attention
        // without standing up the text encoder.
        var ctx = new float[(long)WanDiT.TextDim * 512];
        Fill(ctx, 0xC0FFEEu);

        var rope = WanRope.Build(T, hLen, wLen);
        float[] v = dit.Predict(x, ctx, 500f, rope, seq);

        Assert.Equal((long)dit.OutTok * seq, v.LongLength);
        double sumAbs = 0;
        bool anyNonFinite = false;
        foreach (float f in v)
        {
            if (float.IsNaN(f) || float.IsInfinity(f)) anyNonFinite = true;
            sumAbs += Math.Abs(f);
        }
        _output.WriteLine($"[wan-parity] v: n={v.Length} meanAbs={sumAbs / v.Length:E6} nonFinite={anyNonFinite}");

        var bytes = new byte[v.Length * sizeof(float)];
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(outPath, bytes);

        // A DiT that emits all-zero or non-finite velocity cannot produce a picture:
        // the sampler integrates it into a constant latent and the VAE decodes a flat
        // frame. Catch that here rather than 30 s later in an image nobody diffs.
        Assert.False(anyNonFinite, "DiT velocity contains NaN/Inf");
        Assert.True(sumAbs / v.Length > 1e-8,
            $"DiT velocity is degenerate (meanAbs={sumAbs / v.Length:E6}); the VAE will decode a flat frame");
    }

    private static int ReadPositiveInt(string name, int fallback)
    {
        string raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
            return fallback;
        if (!int.TryParse(raw, out int value) || value <= 0)
            throw new InvalidOperationException($"{name} must be a positive integer; got '{raw}'.");
        return value;
    }

    /// <summary>Deterministic, well-conditioned pseudo-random fill in [-1, 1].</summary>
    private static void Fill(float[] dst, uint seed)
    {
        uint s = seed;
        for (int i = 0; i < dst.Length; i++)
        {
            s = s * 1664525u + 1013904223u;
            dst[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
    }
}
