// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Numerical-parity tests for GLM-5.x (glm-dsa) against llama.cpp.
//
// The golden file is produced by .parity/gen_ref_glm.py against a running
// llama-server on the same split GGUF, and records for each prompt: the
// tokenized prompt and the greedy continuation token ids. Feeding the RECORDED
// prompt token ids in (instead of re-tokenizing) isolates the forward pass from
// any tokenizer or chat-template difference; a separate test covers the
// tokenizer itself.
//
// The interesting record is the long one: DeepSeek Sparse Attention only starts
// dropping cells once the context passes attention.indexer.top_k (2048 for
// GLM-5.2), so a short prompt never exercises the lightning indexer, the top-k
// mask, or the reuse of a selection across the layers that share it.
//
// Opt-in:
//   TS_TEST_MODEL_DIR    directory holding GLM-*.gguf (first shard is enough)
//   TS_GLM_REF           golden JSON (default: <repo>/.parity/ref_glm.json)
//   TS_GLM_BACKEND       backend name (default: ggml_cuda)
// When the model or the golden file is absent the tests soft-skip (still green),
// matching the convention of every other model-dependent test in this project.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Trait("Requires", "Models")]
public class GlmDsaParityTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string EnvRefFile = "TS_GLM_REF";
    private const string EnvBackend = "TS_GLM_BACKEND";

    /// <summary>GLM-5.2's attention.indexer.top_k: below this the attention is dense.</summary>
    private const int IndexerTopK = 2048;

    private readonly ITestOutputHelper _output;
    public GlmDsaParityTests(ITestOutputHelper output) { _output = output; }

    private sealed class RefRecord
    {
        public string Prompt { get; set; }
        public int[] PromptTokens { get; set; }
        public int[] GeneratedTokens { get; set; }
    }

    [Fact]
    public void GlmDsa_GreedyContinuationMatchesLlamaCpp()
    {
        var records = TryLoadReference();
        if (records == null) return;

        string modelPath = TryFindModel();
        if (modelPath == null) return;

        // One golden prompt runs past the indexer's top-k, so the context has to
        // be big enough to hold it or that record would silently test nothing.
        using var ctx = new EnvScope();
        ctx.Set("MAX_CONTEXT", "8192");

        using var model = ModelBase.Create(modelPath, ResolveBackend());
        Assert.Equal("glm-dsa", model.Config.Architecture);

        int mismatches = 0;
        int sparse = 0;
        foreach (var rec in records)
        {
            if (rec.PromptTokens == null || rec.PromptTokens.Length == 0) continue;
            if (rec.GeneratedTokens == null || rec.GeneratedTokens.Length == 0) continue;

            int[] produced = GreedyContinue(model, rec.PromptTokens, rec.GeneratedTokens.Length);

            int firstDiff = -1;
            for (int i = 0; i < produced.Length; i++)
                if (produced[i] != rec.GeneratedTokens[i]) { firstDiff = i; break; }

            bool isSparse = rec.PromptTokens.Length > IndexerTopK;
            if (isSparse) sparse++;

            string preview = rec.Prompt != null && rec.Prompt.Length > 40
                ? rec.Prompt.Substring(0, 40) + "..."
                : rec.Prompt;
            string kind = isSparse ? $"sparse, {rec.PromptTokens.Length} prompt tokens" : "dense";

            if (firstDiff < 0)
            {
                _output.WriteLine($"[MATCH ] {preview}  ({produced.Length} tokens, {kind})");
            }
            else
            {
                mismatches++;
                _output.WriteLine($"[DIFF  ] {preview}  ({kind}) first divergence at index {firstDiff}: " +
                    $"expected {rec.GeneratedTokens[firstDiff]} '{Decode(model, rec.GeneratedTokens[firstDiff])}', " +
                    $"got {produced[firstDiff]} '{Decode(model, produced[firstDiff])}'");
                _output.WriteLine($"          llama.cpp  : {string.Join(",", rec.GeneratedTokens)}");
                _output.WriteLine($"          tensorsharp: {string.Join(",", produced)}");
            }
        }

        Assert.True(sparse > 0,
            $"No golden prompt exceeds the {IndexerTopK}-token indexer top-k, so this run never took the sparse " +
            "path. Regenerate the goldens with .parity/gen_ref_glm.py, which includes a long-context prompt.");
        Assert.True(mismatches == 0,
            $"{mismatches} of {records.Count} prompts diverged from llama.cpp (see test output for the first divergence of each).");
    }

    /// <summary>
    /// The forward pass is compared on recorded token ids, so the tokenizer needs
    /// its own check: our BPE must reproduce llama.cpp's prompt tokenization.
    /// </summary>
    [Fact]
    public void GlmDsa_TokenizerMatchesLlamaCpp()
    {
        var records = TryLoadReference();
        if (records == null) return;

        string modelPath = TryFindModel();
        if (modelPath == null) return;

        using var model = ModelBase.Create(modelPath, ResolveBackend());

        int mismatches = 0;
        foreach (var rec in records)
        {
            if (rec.Prompt == null || rec.PromptTokens == null) continue;

            var ours = model.Tokenizer.Encode(rec.Prompt, addSpecial: true).ToArray();
            if (ours.SequenceEqual(rec.PromptTokens))
            {
                _output.WriteLine($"[MATCH ] {ours.Length} prompt tokens");
                continue;
            }

            mismatches++;
            _output.WriteLine($"[DIFF  ] prompt {rec.Prompt.Substring(0, Math.Min(40, rec.Prompt.Length))}");
            _output.WriteLine($"          llama.cpp   ({rec.PromptTokens.Length}): {string.Join(",", rec.PromptTokens)}");
            _output.WriteLine($"          tensorsharp ({ours.Length}): {string.Join(",", ours)}");
        }

        Assert.True(mismatches == 0, $"{mismatches} prompts tokenized differently from llama.cpp.");
    }

    /// <summary>
    /// Which layers recompute the indexer selection, from llama.cpp's
    /// <c>src/models/glm-dsa.cpp</c>: the first three, then every fourth. The
    /// layers in between attend through the previous full layer's selection, so
    /// an off-by-one here changes what 57 of 78 layers are allowed to look at
    /// without changing any shape.
    /// </summary>
    [Fact]
    public void GlmDsa_IndexerLayerPatternMatchesLlamaCpp()
    {
        const int layers = 78;
        var full = new bool[layers];
        for (int i = 0; i < layers; i++)
            full[i] = i < 3 || (i - 2) % 4 == 0;

        Assert.True(full[0] && full[1] && full[2]);
        Assert.True(full[6] && full[10] && full[14]);
        Assert.False(full[3] || full[4] || full[5]);
        Assert.Equal(21, full.Count(x => x));
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
            if (values[i] > bestValue) { bestValue = values[i]; best = i; }
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
            _output.WriteLine($"{EnvModelDir} not set or missing - skipping GLM parity test.");
            return null;
        }

        // A split GGUF is opened through its first shard; the rest are siblings.
        string path = Directory.EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly)
            .Where(p => Path.GetFileName(p).StartsWith("GLM", StringComparison.OrdinalIgnoreCase))
            .Where(p => !Path.GetFileName(p).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase))
            .Where(p => !System.Text.RegularExpressions.Regex.IsMatch(
                Path.GetFileName(p), @"-(?!00001)\d{5}-of-\d{5}\.gguf$"))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();

        if (path == null)
            _output.WriteLine($"No GLM*.gguf under {dir} - skipping.");
        return path;
    }

    private List<RefRecord> TryLoadReference()
    {
        string path = Environment.GetEnvironmentVariable(EnvRefFile);
        if (string.IsNullOrWhiteSpace(path))
            path = LocateReference("ref_glm.json");

        if (path == null || !File.Exists(path))
        {
            _output.WriteLine($"Golden file not found ({EnvRefFile} unset and no .parity file) - skipping. " +
                "Regenerate with .parity/gen_ref_glm.py (see .parity/README.md).");
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
