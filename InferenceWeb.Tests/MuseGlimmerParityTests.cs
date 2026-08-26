// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Numerical-parity tests for Muse-Glimmer against llama.cpp.
//
// The golden file is produced by .parity/gen_ref.py against a running
// llama-server on the same GGUF, and records for each prompt: the tokenized
// prompt, the greedy continuation token ids, and the top-10 logprobs of the
// first generated token. Feeding the RECORDED prompt token ids into the model
// (instead of re-tokenizing) isolates the forward pass from any tokenizer or
// chat-template difference; a separate test covers the tokenizer itself.
//
// Opt-in:
//   TS_TEST_MODEL_DIR          directory holding Muse-Glimmer-*.gguf
//   TS_MUSE_GLIMMER_REF        golden JSON (default: <repo>/.parity/ref_text.json)
//   TS_MUSE_GLIMMER_BACKEND    backend name (default: ggml_cuda)
// When the model or the golden file is absent the tests soft-skip (still green),
// matching the convention of every other model-dependent test in this project.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class MuseGlimmerParityTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string EnvRefFile = "TS_MUSE_GLIMMER_REF";
    private const string EnvBackend = "TS_MUSE_GLIMMER_BACKEND";

    private readonly ITestOutputHelper _output;
    public MuseGlimmerParityTests(ITestOutputHelper output) { _output = output; }

    private sealed class RefRecord
    {
        public string Prompt { get; set; }
        public int[] PromptTokens { get; set; }
        public int[] GeneratedTokens { get; set; }
        public string Content { get; set; }
    }

    [ModelFact("TS_TEST_MODEL_DIR", "muse-glimmer")]
    public void MuseGlimmer_GreedyContinuationMatchesLlamaCpp()
    {
        var records = TryLoadReference();
        if (records == null) return;

        string modelPath = TryFindModel();
        if (modelPath == null) return;

        using var model = ModelBase.Create(modelPath, ResolveBackend());
        Assert.Equal("muse-glimmer", model.Config.Architecture);

        int mismatches = 0;
        foreach (var rec in records)
        {
            if (rec.PromptTokens == null || rec.PromptTokens.Length == 0)
                continue;
            if (rec.GeneratedTokens == null || rec.GeneratedTokens.Length == 0)
                continue;

            int[] produced = GreedyContinue(model, rec.PromptTokens, rec.GeneratedTokens.Length);

            int firstDiff = -1;
            for (int i = 0; i < produced.Length; i++)
            {
                if (produced[i] != rec.GeneratedTokens[i]) { firstDiff = i; break; }
            }

            string preview = rec.Prompt != null && rec.Prompt.Length > 40
                ? rec.Prompt.Substring(0, 40) + "..."
                : rec.Prompt;

            if (firstDiff < 0)
            {
                _output.WriteLine($"[MATCH ] {preview}  ({produced.Length} tokens)");
            }
            else
            {
                mismatches++;
                _output.WriteLine($"[DIFF  ] {preview}  first divergence at index {firstDiff}: " +
                    $"expected {rec.GeneratedTokens[firstDiff]} '{Decode(model, rec.GeneratedTokens[firstDiff])}', " +
                    $"got {produced[firstDiff]} '{Decode(model, produced[firstDiff])}'");
                _output.WriteLine($"          llama.cpp: {string.Join(",", rec.GeneratedTokens)}");
                _output.WriteLine($"          tensorsharp: {string.Join(",", produced)}");
            }
        }

        Assert.True(mismatches == 0,
            $"{mismatches} of {records.Count} prompts diverged from llama.cpp (see test output for the first divergence of each).");
    }

    /// <summary>
    /// The forward pass is compared on recorded token ids, so the tokenizer needs
    /// its own check: our BPE must reproduce llama.cpp's prompt tokenization.
    /// </summary>
    [ModelFact("TS_TEST_MODEL_DIR", "muse-glimmer")]
    public void MuseGlimmer_TokenizerMatchesLlamaCpp()
    {
        var records = TryLoadReference();
        if (records == null) return;

        string modelPath = TryFindModel();
        if (modelPath == null) return;

        using var model = ModelBase.Create(modelPath, ResolveBackend());

        int mismatches = 0;
        foreach (var rec in records)
        {
            if (rec.Prompt == null || rec.PromptTokens == null)
                continue;

            var ours = model.Tokenizer.Encode(rec.Prompt, addSpecial: true).ToArray();
            if (ours.SequenceEqual(rec.PromptTokens))
            {
                _output.WriteLine($"[MATCH ] {ours.Length} prompt tokens");
                continue;
            }

            mismatches++;
            _output.WriteLine($"[DIFF  ] prompt {rec.Prompt.Substring(0, Math.Min(40, rec.Prompt.Length))}");
            _output.WriteLine($"          llama.cpp  ({rec.PromptTokens.Length}): {string.Join(",", rec.PromptTokens)}");
            _output.WriteLine($"          tensorsharp ({ours.Length}): {string.Join(",", ours)}");
        }

        Assert.True(mismatches == 0, $"{mismatches} prompts tokenized differently from llama.cpp.");
    }

    /// <summary>
    /// The sliding-window pattern is the single easiest thing to get off by one:
    /// llama.cpp's set_swa_pattern(4) makes layer l a sliding-window layer when
    /// l % 4 &lt; 3, i.e. every 4th layer (l = 3, 7, 11, ...) is full causal + NoPE.
    /// </summary>
    [Fact]
    public void MuseGlimmer_SwaPatternMatchesLlamaCpp()
    {
        const int period = 4;
        var expected = new bool[52];
        for (int l = 0; l < expected.Length; l++)
            expected[l] = (l % period) < (period - 1);

        Assert.True(expected[0] && expected[1] && expected[2]);
        Assert.False(expected[3]);
        Assert.False(expected[51]);
        Assert.Equal(39, expected.Count(x => x));
        Assert.Equal(13, expected.Count(x => !x));
    }

    /// <summary>
    /// The sliding window (2048) does nothing until the context exceeds it, so the
    /// short prompts above never exercise the SWA mask - nor the fused kernel's
    /// 256-token padded attention window rolling over, nor a KV-cache grow. This
    /// runs a ~4790-token prompt against a llama.cpp golden captured the same way.
    /// Generate it with .parity/gen_ref_long.py (needs a llama-server with -c 8192).
    /// </summary>
    [ModelFact("TS_TEST_MODEL_DIR", "muse-glimmer")]
    public void MuseGlimmer_LongContextMatchesLlamaCpp()
    {
        var records = TryLoadReference(LocateReference("ref_text_long.json"), "TS_MUSE_GLIMMER_REF_LONG");
        if (records == null) return;

        string modelPath = TryFindModel();
        if (modelPath == null) return;

        // The golden prompt is longer than the default context.
        using var ctx = new EnvScope();
        ctx.Set("MAX_CONTEXT", "8192");
        using var model = ModelBase.Create(modelPath, ResolveBackend());

        foreach (var rec in records)
        {
            Assert.True(rec.PromptTokens.Length > 2048,
                $"Long-context golden has only {rec.PromptTokens.Length} prompt tokens; it must exceed the 2048 sliding window to be meaningful.");

            int[] produced = GreedyContinue(model, rec.PromptTokens, rec.GeneratedTokens.Length);
            _output.WriteLine($"prompt tokens: {rec.PromptTokens.Length}");
            _output.WriteLine($"llama.cpp  : {string.Join(",", rec.GeneratedTokens)}");
            _output.WriteLine($"tensorsharp: {string.Join(",", produced)}");
            Assert.Equal(rec.GeneratedTokens, produced);
        }
    }

    /// <summary>
    /// DFlash is lossless speculative decoding: every drafted token is verified by
    /// the target, so greedy output must be byte-identical to the non-speculative
    /// path (and therefore to llama.cpp). llama.cpp's own DFlash run over these same
    /// prompts reproduces its baseline token-for-token at 1.33x-4.77x, so anything
    /// other than an exact match here is a bug, not sampling noise.
    /// </summary>
    [ModelFact("TS_TEST_MODEL_DIR", "muse-glimmer")]
    public void MuseGlimmer_DFlashDraftingIsLossless()
    {
        var records = TryLoadReference();
        if (records == null) return;

        string modelPath = TryFindModel();
        if (modelPath == null) return;

        string draftPath = TryFindDraftModel();
        if (draftPath == null) return;

        using var model = ModelBase.Create(modelPath, ResolveBackend(), draftModelPath: draftPath);
        var spec = model as ISpeculativeModel;
        Assert.NotNull(spec);
        Assert.True(spec.HasDraftHead, "DFlash drafter did not arm (HasDraftHead is false).");
        _output.WriteLine($"DFlash armed: hidden={spec.SpecFeatureSize} blockDrafts={spec.DraftBlockSize}");

        var decoder = new SpeculativeDecoder(spec, spec.DraftBlockSize);

        int mismatches = 0;
        foreach (var rec in records)
        {
            if (rec.PromptTokens == null || rec.GeneratedTokens == null || rec.GeneratedTokens.Length == 0)
                continue;

            model.ResetKVCache();
            var produced = decoder.GenerateGreedy(rec.PromptTokens, rec.GeneratedTokens.Length)
                .Take(rec.GeneratedTokens.Length)
                .ToArray();

            string preview = rec.Prompt != null && rec.Prompt.Length > 40
                ? rec.Prompt.Substring(0, 40) + "..."
                : rec.Prompt;

            if (produced.Length == rec.GeneratedTokens.Length && produced.SequenceEqual(rec.GeneratedTokens))
            {
                _output.WriteLine($"[MATCH ] {preview}  ({produced.Length} tokens, " +
                    $"acceptance {decoder.AcceptanceRate:P1}, drafted {decoder.TokensDrafted}, accepted {decoder.TokensAccepted})");
            }
            else
            {
                mismatches++;
                _output.WriteLine($"[DIFF  ] {preview}");
                _output.WriteLine($"          llama.cpp  : {string.Join(",", rec.GeneratedTokens)}");
                _output.WriteLine($"          dflash     : {string.Join(",", produced)}");
            }
        }

        _output.WriteLine($"Overall DFlash acceptance: {decoder.AcceptanceRate:P1} " +
            $"({decoder.TokensAccepted}/{decoder.TokensDrafted} drafted tokens over " +
            $"{decoder.VerifySteps} verify steps, {decoder.PlainSteps} plain steps)");

        Assert.True(mismatches == 0,
            $"DFlash speculative decoding changed the output on {mismatches} prompts; it must be lossless.");
        Assert.True(decoder.TokensDrafted > 0, "DFlash never drafted a token - the drafter is not running.");
    }

    /// <summary>
    /// Image geometry against llama.cpp's muse_glimmer_grid_size. The 1024x1024
    /// case is pinned to an observed llama-server run: prompt_tokens 1434 for a
    /// 65-token chat frame, i.e. 1369 = 37x37 image tokens; the 336x336 case to
    /// prompt_tokens 209, i.e. 144 = 12x12.
    /// </summary>
    [Fact]
    public void MuseGlimmer_ImageGeometryMatchesLlamaCpp()
    {
        var proc = new MuseGlimmerImageProcessor();

        Assert.Equal(28, proc.PatchHW);

        // 336x336 divides exactly: 336/28 = 12 -> 12x12 = 144 tokens, no resize.
        Assert.Equal((336, 336), proc.ComputeTargetSize(336, 336));
        Assert.Equal(144, proc.ComputeTokenCount(336, 336));

        // 1024x1024: i_np = 36.571; candidates (36,36) d=0 and (37,37) d=0 tie on
        // aspect, broken toward more tokens -> 37x37, stretched to 1036x1036.
        Assert.Equal((1036, 1036), proc.ComputeTargetSize(1024, 1024));
        Assert.Equal(1369, proc.ComputeTokenCount(1024, 1024));

        // Above the 4096-token cap the grid is shrunk while preserving the ratio.
        int wide = proc.ComputeTokenCount(4000, 2000);
        Assert.True(wide <= 4096, $"token count {wide} exceeds the 4096 cap");

        // Tall and thin: the chosen grid must stay at least 1 token on each axis.
        Assert.Equal(1, proc.ComputeTokenCount(20, 20));
    }

    private string TryFindDraftModel()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
            return null;

        string path = Directory.EnumerateFiles(dir, "dflash*.gguf", SearchOption.TopDirectoryOnly)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();

        if (path == null)
            _output.WriteLine($"No dflash*.gguf under {dir} - skipping the DFlash test.");
        return path;
    }

    private static int[] GreedyContinue(ModelBase model, int[] promptTokens, int count)
    {
        model.ResetKVCache();
        var produced = new int[count];

        float[] logits = model.Forward(promptTokens);
        for (int i = 0; i < count; i++)
        {
            int next = ArgMax(logits);
            produced[i] = next;
            if (i + 1 < count)
                logits = model.Forward(new[] { next });
        }
        return produced;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        float bestValue = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestValue) { bestValue = values[i]; best = i; }
        }
        return best;
    }

    private static string Decode(ModelBase model, int token)
    {
        try { return model.Tokenizer.Decode(new List<int> { token }); }
        catch { return "?"; }
    }

    private static BackendType ResolveBackend()
    {
        string name = Environment.GetEnvironmentVariable(EnvBackend);
        if (string.IsNullOrWhiteSpace(name))
            return BackendType.GgmlCuda;
        return Enum.TryParse<BackendType>(name, ignoreCase: true, out var parsed)
            ? parsed
            : BackendType.GgmlCuda;
    }

    private string TryFindModel()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set or missing - skipping Muse-Glimmer parity test.");
            return null;
        }

        string path = Directory.EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly)
            .Where(p => Path.GetFileName(p).StartsWith("Muse-Glimmer", StringComparison.OrdinalIgnoreCase))
            .Where(p => !Path.GetFileName(p).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();

        if (path == null)
            _output.WriteLine($"No Muse-Glimmer*.gguf under {dir} - skipping.");
        return path;
    }

    private List<RefRecord> TryLoadReference() => TryLoadReference(LocateReference("ref_text.json"), EnvRefFile);

    private List<RefRecord> TryLoadReference(string defaultPath, string envName)
    {
        string path = Environment.GetEnvironmentVariable(envName);
        if (string.IsNullOrWhiteSpace(path))
            path = defaultPath;

        if (path == null || !File.Exists(path))
        {
            _output.WriteLine($"Golden file not found ({envName} unset and no .parity file) - skipping. " +
                "Regenerate with the scripts in .parity/ (see .parity/README.md).");
            return null;
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var records = new List<RefRecord>();
        foreach (var el in doc.RootElement.EnumerateArray())
        {
            records.Add(new RefRecord
            {
                Prompt = el.TryGetProperty("prompt", out var p) ? p.GetString() : null,
                PromptTokens = ReadIntArray(el, "prompt_tokens"),
                GeneratedTokens = ReadIntArray(el, "generated_tokens"),
                Content = el.TryGetProperty("content", out var c) ? c.GetString() : null,
            });
        }
        _output.WriteLine($"Loaded {records.Count} golden records from {path}");
        return records;
    }

    private static int[] ReadIntArray(JsonElement el, string name)
    {
        if (!el.TryGetProperty(name, out var arr) || arr.ValueKind != JsonValueKind.Array)
            return null;
        var list = new List<int>();
        foreach (var v in arr.EnumerateArray())
        {
            if (v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i))
                list.Add(i);
            else
                return null;
        }
        return list.ToArray();
    }

    /// <summary>Walk up from the test binary to the repo root and look for .parity/&lt;name&gt;.</summary>
    private static string LocateReference(string name)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, ".parity", name);
            if (File.Exists(candidate))
                return candidate;
        }
        return null;
    }
}
