// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Regression: "two prompts in parallel, then New Chat, then a prompt -> <pad>
// forever".
//
// Mechanism. Gemma 4 serves N>=2 concurrent requests from per-request KV-cache
// holders. On the first multi-sequence step the executor calls
// AdoptPrimaryCacheToFused, which hands the live PRIMARY cache to the running
// request's holder and mints a REPLACEMENT primary for later single-stream use.
// That replacement is the cache the next N==1 request runs on - and it was
// allocated without being zeroed, while its sibling CreateFreshHolder zeroed
// its own allocation for exactly this reason.
//
// It matters because the fused decode graph reads a FIXED 256-padded attention
// window: rows past the written length are masked with -inf, which cancels a
// FINITE score but turns a non-finite one into NaN, and one NaN takes the whole
// softmax row -> every logit NaN -> argmax returns index 0 -> Gemma's <pad>,
// forever. ModelBase.InitializeCacheTensor skips the fill on GgmlCuda/Mlx and
// the pool hands back recycled (not zeroed) blocks, so on those backends the
// padding was live activation garbage.
//
// The fix moved the zero-fill INTO AllocateKvCacheArrays so no allocation site
// can skip it. This test drives the exact user-visible sequence.
//
// Opt-in, and it needs TWO env vars:
//   TS_TEST_MODEL_DIR=<dir containing gemma-4-E4B*.gguf>
//   TS_TEST_GGML_BACKEND=cuda
// The second one matters. GGML allows exactly ONE backend per process and
// GgmlBackendTestInitializer pins CPU by default, so without it this test can
// only report "cannot load on GgmlCuda ... Skipping". And CPU would prove
// nothing: the defect exists only where cache allocation is not zero-filled.
//
// Verified to catch the bug: with the zero-fill reverted this fails with
// "solo request after the concurrent burst emitted 47 <pad> tokens".
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4ConcurrentThenSoloRegressionTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private readonly ITestOutputHelper _output;

    public Gemma4ConcurrentThenSoloRegressionTests(ITestOutputHelper output) { _output = output; }

    [ModelFact("TS_TEST_MODEL_DIR", "gemma-4-e4b")]
    public async Task SoloRequestAfterConcurrentBurst_DoesNotDegenerate()
    {
        string modelPath = FindGemma4();
        if (modelPath == null) { _output.WriteLine("no gemma-4-e4b model; skipping"); return; }

        // The defect only exists on a backend whose cache allocation is NOT
        // zero-filled, so a green run on ggml_cpu would prove nothing. Try the GPU
        // backend and treat "cannot create" as a skip rather than a false pass.
        BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCuda;
        TensorSharp.Models.ModelBase model;
        try
        {
            model = TensorSharp.Models.ModelBase.Create(modelPath, backend);
        }
        catch (Exception ex)
        {
            _output.WriteLine($"cannot load on {backend} ({ex.GetType().Name}: {ex.Message}); "
                              + "this defect only reproduces on a non-zero-filling GPU backend. Skipping.");
            return;
        }
        using var _model = model;
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        const int blockSize = 256;
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 4096,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 1024,
            NumBlocks = 128,
            BlockSize = blockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = blockSize,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        // 1. Two requests in flight together. This is what makes the executor
        //    hand the primary cache away and mint the replacement.
        var a = Generate(engine, model, renderer, "Describe the game Final Fantasy VII in detail.", "par-a", 48);
        await Task.Delay(300);   // let A start before B joins, as two browser tabs would
        var b = Generate(engine, model, renderer, "Describe the book A Brief History of Time in detail.", "par-b", 48);
        string outA = await a, outB = await b;
        _output.WriteLine($"[parallel A] {Trim(outA)}");
        _output.WriteLine($"[parallel B] {Trim(outB)}");

        // 2. Both finished -> holders released -> the replacement primary becomes
        //    the active cache. "New Chat" then sends a plain single-stream request,
        //    which is the one that used to emit nothing but <pad>.
        string solo = await Generate(engine, model, renderer,
            "Describe the game Final Fantasy VII in detail.", "solo-after", 48);
        _output.WriteLine($"[solo after burst] {Trim(solo)}");

        AssertNotDegenerate(outA, "first parallel request");
        AssertNotDegenerate(outB, "second parallel request");
        AssertNotDegenerate(solo, "solo request after the concurrent burst");
    }

    /// <summary>Degenerate == the sampler is being fed dead logits: the reply is
    /// empty, is one token repeated, or is mostly Gemma's &lt;pad&gt;.</summary>
    private static void AssertNotDegenerate(string text, string what)
    {
        Assert.False(string.IsNullOrWhiteSpace(text), $"{what} produced no text at all.");

        int pads = CountOccurrences(text, "<pad>");
        Assert.True(pads <= 2,
            $"{what} emitted {pads} <pad> tokens - the NaN-logits signature. Text: {Trim(text)}");

        // A healthy 48-token reply has many distinct words; a fixed-point loop has
        // one or two. This also catches placeholder-token streams (<unusedNN>).
        var words = text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length >= 8)
        {
            int distinct = words.Distinct(StringComparer.Ordinal).Count();
            Assert.True(distinct > 3,
                $"{what} repeated the same {distinct} token(s) for its whole reply. Text: {Trim(text)}");
        }
    }

    private static int CountOccurrences(string haystack, string needle)
    {
        int n = 0, i = 0;
        while ((i = haystack.IndexOf(needle, i, StringComparison.Ordinal)) >= 0) { n++; i += needle.Length; }
        return n;
    }

    private static string Trim(string s)
        => s.Length <= 200 ? s : s.Substring(0, 200) + "...";

    private static async Task<string> Generate(
        InferenceEngine engine, TensorSharp.Models.ModelBase model,
        KVCachePromptRenderer renderer, string prompt, string reqId, int maxNewTokens)
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
        var tokens = renderer.RenderToTokens(
            model.Tokenizer,
            model.Config?.ChatTemplate,
            history,
            model.Config?.Architecture ?? string.Empty,
            addGenerationPrompt: true,
            tools: null,
            enableThinking: false);

        var seq = new SequenceState(reqId, tokens, maxNewTokens, 256, SamplingConfig.Default);
        var handle = engine.SubmitRequest(seq);
        var sb = new System.Text.StringBuilder();
        try
        {
            await foreach (var tok in handle.Tokens.ReadAllAsync())
                sb.Append(model.Tokenizer.Decode(new List<int> { tok }));
        }
        catch (Exception ex)
        {
            throw new Xunit.Sdk.XunitException($"generation for {reqId} failed: {ex}");
        }
        await handle.Completion;
        return sb.ToString();
    }

    private static string FindGemma4()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        return Directory.GetFiles(dir, "*.gguf").FirstOrDefault(p =>
        {
            var n = Path.GetFileName(p).ToLowerInvariant();
            return n.Contains("gemma-4-e4b") && !n.Contains("mmproj") && !n.Contains("assistant");
        });
    }
}
