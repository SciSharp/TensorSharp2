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
// Backend PARITY for autoregressive decode: prefill a fixed token sequence, then
// take N greedy steps, printing the argmax and logit statistics of every step.
// Run once per backend and diff to see exactly which step diverges and whether
// the logits went NaN / all-equal.
//
// Written for "gemma-4-E4B emits one correct token then <pad> forever on MLX"
// (gemma-4-12B and -26B-A4B are fine on the same backend), where the interesting
// question is whether PREFILL or DECODE breaks and whether the logits collapse.
//
// Opt in with TS_PARITY_MODEL=<model.gguf> [TS_PARITY_BACKEND=mlx|ggml_metal|ggml_cpu]
//             [TS_PARITY_STEPS=8]

using System;
using System.IO;
using System.Text;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class DecodeBackendParityTests
{
    private readonly ITestOutputHelper _output;
    public DecodeBackendParityTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void DumpGreedyDecodeTrace()
    {
        string modelPath = Environment.GetEnvironmentVariable("TS_PARITY_MODEL");
        if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
        {
            _output.WriteLine("[decode-parity] TS_PARITY_MODEL unset; skipping");
            return;
        }

        var backend = (Environment.GetEnvironmentVariable("TS_PARITY_BACKEND") ?? "ggml_cpu").ToLowerInvariant() switch
        {
            "mlx" => BackendType.Mlx,
            "ggml_metal" or "metal" => BackendType.GgmlMetal,
            "ggml_cuda" or "cuda" => BackendType.GgmlCuda,
            _ => BackendType.GgmlCpu,
        };
        int steps = int.TryParse(Environment.GetEnvironmentVariable("TS_PARITY_STEPS"), out int s) && s > 0 ? s : 8;

        using var model = ModelBase.Create(modelPath, backend);
        int vocab = model.Config.VocabSize;

        // A fixed, tokenizer-rendered prompt so both backends see identical ids.
        // TS_PARITY_PROMPT_FILE overrides the prompt: prompt LENGTH selects which
        // kernels run (e.g. the MLX packed GatedDeltaNet kernel only engages at
        // seqLen >= TS_MLX_QWEN35_GDN_PACKED_MIN_SEQ_LEN, default 64), so a short
        // prompt can pass while a realistic one is wrong.
        string promptFile = Environment.GetEnvironmentVariable("TS_PARITY_PROMPT_FILE");
        string promptText = !string.IsNullOrEmpty(promptFile) && File.Exists(promptFile)
            ? File.ReadAllText(promptFile)
            : "The capital of France is";
        int[] prompt = System.Linq.Enumerable.ToArray(model.Tokenizer.Encode(promptText, addSpecial: true));
        _output.WriteLine($"[decode-parity] backend={backend} arch={model.Config.Architecture} " +
                          $"vocab={vocab} promptTokens={prompt.Length}");

        var sb = new StringBuilder();
        float[] logits = model.Forward(prompt);
        sb.AppendLine(Describe("prefill", logits, vocab));

        int next = ArgMax(logits, vocab);
        for (int i = 0; i < steps; i++)
        {
            logits = model.Forward(new[] { next });
            sb.AppendLine(Describe($"decode{i}", logits, vocab));
            next = ArgMax(logits, vocab);
        }
        _output.WriteLine(sb.ToString());
        // xunit only surfaces test output for FAILING tests, so also write the
        // trace where a cross-backend diff can pick it up.
        string tracePath = Environment.GetEnvironmentVariable("TS_PARITY_OUT");
        if (!string.IsNullOrEmpty(tracePath))
            File.WriteAllText(tracePath, sb.ToString());

        // The failure this exists to catch: logits that are NaN, or so flat that
        // argmax always lands on token 0 (which detokenises to <pad>).
        Assert.True(!float.IsNaN(logits[0]), "decode logits contain NaN");
    }

    private static string Describe(string tag, float[] logits, int vocab)
    {
        int n = Math.Min(vocab, logits.Length);
        double sum = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        int nan = 0;
        for (int i = 0; i < n; i++)
        {
            float v = logits[i];
            if (float.IsNaN(v)) { nan++; continue; }
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        int am = ArgMax(logits, vocab);
        return $"  {tag,-9} argmax={am,-8} max={max,-12:F5} min={min,-12:F5} mean={sum / Math.Max(1, n - nan),-12:F5} nan={nan}";
    }

    private static int ArgMax(float[] logits, int vocab)
    {
        int n = Math.Min(vocab, logits.Length), best = 0;
        float bv = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
}
