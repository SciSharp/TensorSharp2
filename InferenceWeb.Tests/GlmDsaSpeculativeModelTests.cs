// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// NextN/MTP speculative decoding on a REAL GLM-5.x checkpoint.
//
// GlmDsaSpeculativeTests covers the architecture on a 1.9 MB synthetic model, which
// proves the wiring but says nothing about either thing that actually matters
// here: whether a draft head trained on this model predicts well enough to be
// accepted, and whether the fused verify beats plain decode. Both need the real
// weights, so this is opt-in:
//
//   TS_TEST_MODEL_DIR   directory holding GLM-*.gguf (first shard is enough)
//   TS_GLM_BACKEND      backend name (default: ggml_cuda)
//   TS_GLM_MTP_TOKENS   tokens to generate per run (default: 96)
//   TS_N_CPU_MOE        routed-expert offload, as on the server
//
// Everything happens in ONE process against ONE loaded model: a 175 GiB load
// takes four minutes, and an A/B across two processes would compare two
// page-cache states as much as two decode paths.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Trait("Requires", "Models")]
public class GlmDsaSpeculativeModelTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string EnvBackend = "TS_GLM_BACKEND";
    private const string EnvTokens = "TS_GLM_MTP_TOKENS";

    private readonly ITestOutputHelper _output;
    public GlmDsaSpeculativeModelTests(ITestOutputHelper output) { _output = output; }

    private const string Prompt =
        "Explain, in a single paragraph, why speculative decoding does not change " +
        "the output of a language model.";

    /// <summary>
    /// One load, every measurement. Covers, in order:
    ///
    ///   1. the draft block loaded at all;
    ///   2. the batched trunk forward the verify pass relies on agrees with
    ///      per-token decode (this is what says the verify MATH is right, and it
    ///      involves no MTP code whatsoever);
    ///   3. driving the speculative loop with drafting suppressed reproduces
    ///      greedy EXACTLY (this is what says the plain / catch-up / cache
    ///      bookkeeping is right);
    ///   4. drafting for real: acceptance, the emitted stream, and throughput
    ///      across draft-confidence gates.
    ///
    /// Splitting 2 and 3 out is what makes a divergence in 4 diagnosable instead
    /// of a mystery: on a 2-bit 256-expert MoE a last-bit difference in a router
    /// logit changes which experts run, so batched-vs-sequential arithmetic can
    /// flip a near-tied token no matter how correct the speculation is.
    /// </summary>
    [Fact]
    public void GlmDsa_MtpSpeculativeDecoding_Report()
    {
        string modelPath = TryFindModel();
        if (modelPath == null) return;

        int maxNew = EnvInt(EnvTokens, 96);

        using var env = new EnvScope();
        // The native loader only pages in the draft block when speculation was
        // asked for; --mtp-spec sets this before the server loads its model.
        env.Set("TS_MTP_SPEC", "1");
        // GLM-5.2 does not fit two 97 GiB cards without offload, and the server
        // applies this from --n-cpu-moe before its model loads.
        MoeCpuOffloadConfig.ConfigureFromEnvironment();

        using var model = ModelBase.Create(modelPath, ResolveBackend());
        Assert.Equal("glm-dsa", model.Config.Architecture);

        var spec = Assert.IsAssignableFrom<ISpeculativeModel>(model);
        Assert.True(spec.HasDraftHead,
            "the NextN/MTP draft block did not load — check the loader's stderr for why " +
            "(a trunk-only checkpoint, or no room on the device)");
        Assert.Equal(model.Config.HiddenSize, spec.SpecFeatureSize);

        int[] prompt = model.Tokenizer.Encode(Prompt, addSpecial: true).ToArray();
        int vocab = model.Config.VocabSize;
        _output.WriteLine($"prompt {prompt.Length} tokens, generating {maxNew}, vocab {vocab}");

        // ---- 1. plain greedy: the reference stream and the baseline speed ---
        model.ResetKVCache();
        var swPrefill = Stopwatch.StartNew();
        float[] logits = model.ForwardRefill(prompt);
        swPrefill.Stop();

        var plain = new int[maxNew];
        var plainRowLogits = new float[maxNew][];
        var swPlain = Stopwatch.StartNew();
        for (int i = 0; i < maxNew; i++)
        {
            plainRowLogits[i] = (float[])logits.Clone();
            plain[i] = Argmax(logits);
            if (i + 1 < maxNew) logits = model.Forward(new[] { plain[i] });
        }
        swPlain.Stop();
        // maxNew tokens are emitted but only maxNew-1 forwards ran (the last is
        // drawn from the previous step's logits and never fed back).
        var plainSamples = new List<double> { (maxNew - 1) / swPlain.Elapsed.TotalSeconds };

        // Repeat the baseline. It is the noisy half of this comparison: 20 of 78
        // layers run their experts on the host under --n-cpu-moe, and that
        // matmul contends with whatever else the box is doing, so a single
        // sample has moved by 20% between otherwise identical runs while the
        // speculative numbers stayed inside 2%. The median of three is what the
        // speedups below are quoted against.
        for (int rep = 0; rep < 2; rep++)
        {
            model.ResetKVCache();
            float[] l = model.ForwardRefill(prompt);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < maxNew; i++)
            {
                int t = Argmax(l);
                if (i + 1 < maxNew) l = model.Forward(new[] { t });
            }
            sw.Stop();
            plainSamples.Add((maxNew - 1) / sw.Elapsed.TotalSeconds);
        }
        plainSamples.Sort();
        double plainTps = plainSamples[plainSamples.Count / 2];
        _output.WriteLine($"[1] plain greedy: prefill {prompt.Length / swPrefill.Elapsed.TotalSeconds:F1} tok/s, " +
                          $"decode {plainTps:F2} tok/s (median of {plainSamples.Count}: " +
                          $"{string.Join(", ", plainSamples.Select(x => x.ToString("F2")))})");

        // ---- 2a. does capturing h change the trunk at all? -----------------
        // SpecForward reorders the tail of the trunk graph: the final norm runs
        // over every row and the LM head selects from the NORMED rows, instead
        // of selecting first and norming one row. RMS norm is row-wise, so that
        // is the same arithmetic per row — and at nt = 1 it is the ONLY thing
        // that differs from a plain Forward, which makes this the clean control
        // for the batched comparison below.
        model.ResetKVCache();
        model.ForwardRefill(prompt);
        var h1 = new float[spec.SpecFeatureSize];
        var specRow = new float[vocab];
        spec.SpecForward(new[] { plain[0] }, h1, specRow, allLogitsRows: false);

        double soloWorst = 0;
        for (int v = 0; v < vocab; v++)
            soloWorst = Math.Max(soloWorst, Math.Abs(specRow[v] - plainRowLogits[1][v]));
        _output.WriteLine($"[2a] h-capturing forward at nt=1 vs plain forward: " +
                          $"argmax {(Argmax(specRow) == Argmax(plainRowLogits[1]) ? "same" : "DIFFERENT")}, " +
                          $"max |delta| {soloWorst:G4}");
        Assert.Equal(Argmax(plainRowLogits[1]), Argmax(specRow));
        Assert.True(soloWorst < 1e-2,
            $"capturing the hidden state changed a single-token forward by {soloWorst:G4}; it must be a pure " +
            "read-out of a value the trunk already computes");

        // ---- 2b. batched trunk vs per-token decode -------------------------
        // Replay the first W generated tokens as ONE batch from the same cache
        // position the per-token run started from, and compare row by row. Any
        // difference here is the trunk's own batched-vs-sequential arithmetic;
        // MTP is not involved and cannot cause or cure it.
        //
        // Note the shift. plainRowLogits[i] is the distribution token i was drawn
        // FROM, so it was produced by the forward of token i-1 (or the prompt,
        // for i = 0). Batched row r consumes plain[r], so it predicts plain[r+1]
        // and lines up with plainRowLogits[r+1] — there is no batched row
        // corresponding to plainRowLogits[0] at all.
        // Sweep the WHOLE generated continuation in windows, not just the first
        // one: a flip rate needs a denominator. Each window replays from the
        // position the per-token run was at, so every batched row has an exact
        // per-token counterpart.
        const int W = 8;
        int argmaxFlips = 0, compared = 0, top5Mismatch = 0;
        double worstAbs = 0;
        var hAll = new float[(long)W * spec.SpecFeatureSize];
        var rowLogits = new float[(long)W * vocab];
        for (int start = 0; start + 1 < maxNew; start += W)
        {
            int w = Math.Min(W, maxNew - start);
            if (w < 2) break;
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            if (start > 0)
            {
                // Re-establish the cache with the tokens that precede the window,
                // one at a time, so the window's rows see exactly the state the
                // per-token run left behind.
                for (int i = 0; i < start; i++) model.Forward(new[] { plain[i] });
            }
            var batch = new int[w];
            Array.Copy(plain, start, batch, 0, w);
            spec.SpecForward(batch, hAll, rowLogits, allLogitsRows: true);

            for (int r = 0; r + 1 < w; r++)
            {
                var row = new float[vocab];
                Array.Copy(rowLogits, (long)r * vocab, row, 0, vocab);
                float[] reference = plainRowLogits[start + r + 1];
                compared++;
                if (Argmax(row) != Argmax(reference)) argmaxFlips++;
                if (!TopK(row, 5).SequenceEqual(TopK(reference, 5))) top5Mismatch++;
                for (int v = 0; v < vocab; v++)
                    worstAbs = Math.Max(worstAbs, Math.Abs(row[v] - reference[v]));
            }
        }
        _output.WriteLine($"[2b] batched trunk ({W}-row windows) vs per-token decode over {compared} rows: " +
                          $"argmax flips {argmaxFlips} ({(double)argmaxFlips / Math.Max(1, compared):P2}), " +
                          $"top-5 reorders {top5Mismatch}, max |delta| {worstAbs:G4}");
        // What must hold is that the batch computes the same FUNCTION, not the
        // same bits. It cannot be the same bits: a K-row GEMM reduces in a
        // different order than a 1-row one, and with 256 experts at top-8 a
        // last-bit difference in a router logit changes which experts run, which
        // is a discrete change that 78 layers then amplify. So the check is on
        // what actually determines the token stream — the ranking. A wrong
        // hidden state, mask or cache row flips the argmax on every row (that is
        // exactly what an off-by-one in this comparison produced: 8/8 flips and
        // |delta| 21), which is unmissable here.
        // A handful of flips across ~80 rows is the near-tie rate on a 2-bit MoE
        // and is what makes a greedy speculative run eventually take a different
        // (equally valid) branch. A WRONG computation flips essentially every
        // row — an off-by-one in this very comparison produced 8/8 flips and
        // |delta| 21 — so the threshold separates the two cases by an order of
        // magnitude and does not need to be tight.
        Assert.True(argmaxFlips <= Math.Max(2, compared / 10),
            $"{argmaxFlips}/{compared} batched rows pick a different top token than per-token decode — " +
            "that is a different computation, not a different rounding");
        Assert.True(top5Mismatch <= compared / 2,
            $"{top5Mismatch}/{compared} batched rows reorder the top 5 — beyond what near-ties explain");

        // ---- 2c. what a verify actually costs ------------------------------
        // Speculation is only ever worth it when a K+1-row trunk pass costs less
        // than K+1 one-row passes. On a sparse MoE it normally does — the routed
        // experts are read once for the whole batch — but --n-cpu-moe moves some
        // of that work to a host matmul that scales with rows, so the curve is
        // worth having in front of you before reading the throughput numbers.
        var costs = new List<string>();
        double cost1 = 0;
        foreach (int rows in new[] { 1, 2, 3, 5, 9 })
        {
            var toks = new int[rows];
            for (int i = 0; i < rows; i++) toks[i] = plain[i % maxNew];
            var hBuf = new float[(long)rows * spec.SpecFeatureSize];
            var lBuf = new float[(long)rows * vocab];

            model.ResetKVCache();
            model.ForwardRefill(prompt);
            spec.SpecForward(toks, hBuf, lBuf, allLogitsRows: true);   // warm the graph
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            var sw = Stopwatch.StartNew();
            spec.SpecForward(toks, hBuf, lBuf, allLogitsRows: true);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            if (rows == 1) cost1 = ms;
            costs.Add($"{rows} rows {ms,7:F1} ms ({ms / cost1:F2}x a 1-row pass, {ms / rows / cost1:F2}x per token)");
        }
        _output.WriteLine("[2c] trunk verify cost:");
        foreach (string c in costs) _output.WriteLine("     " + c);

        // ---- 3. the speculative loop with drafting suppressed --------------
        // A gate no finite confidence can clear (1.0f itself is reachable: the
        // top-1 probability over the top-10 logits rounds to exactly 1.0f for a
        // sufficiently peaked head) makes every step degrade to a plain decode
        // THROUGH the speculative code: same cache bookkeeping, same DraftCatchUp
        // calls, same rewinds. If this does not reproduce greedy exactly, the
        // bug is in this file's subject and not in floating point.
        var noDraft = new SpeculativeDecoder(spec, maxDraftTokens: 4)
        {
            AdaptiveSpeculation = false,
            MinDraftProb = float.MaxValue,
        };
        List<int> noDraftTokens = noDraft.GenerateGreedy(prompt, maxNew);
        Assert.Equal(0, noDraft.VerifySteps);
        int noDraftDiff = FirstDifference(plain, noDraftTokens);
        _output.WriteLine($"[3] speculative loop, drafting suppressed: " +
                          (noDraftDiff < 0 ? "identical to greedy" : $"DIVERGED at {noDraftDiff}"));
        Assert.True(noDraftDiff < 0,
            $"the speculative code path diverged from greedy at token {noDraftDiff} with no drafting at all — " +
            "the trunk/catch-up/cache bookkeeping is wrong, independently of any draft quality");

        // ---- 4. drafting for real -----------------------------------------
        // Two knobs interact. `k` bounds the window; `pMin` stops the chain at
        // the first token the head is not sure about. A 1-token window with the
        // gate open is the interesting extreme: it never wastes more than one
        // verify row, and [2c] says that row is nearly free.
        (int k, float pMin)[] configs =
        {
            (8, 0.75f),   // the shared per-token default
            (8, 0.60f),   // glm-dsa's own gate, full window
            (4, 0.60f),   // glm-dsa's own gate, capped window
            (4, 0.55f),
            (2, 0.30f),
        };

        var results = new List<string>();
        double bestTps = 0;
        string bestConfig = "none";
        double defaultAcceptance = 0;
        long defaultDrafted = 0;
        foreach (var (k, pMin) in configs)
        {
            var decoder = new SpeculativeDecoder(spec, maxDraftTokens: k)
            {
                // Measure the drafter, not the governor's opinion of it: the
                // governor is exercised on the server path, and here it would
                // park drafting on an 8-step sample and leave nothing to compare.
                AdaptiveSpeculation = false,
                MinDraftProb = pMin,
            };
            List<int> produced = decoder.GenerateGreedy(prompt, maxNew);
            double tps = (maxNew - 1) / decoder.LastDecodeSeconds;
            int diff = FirstDifference(plain, produced);
            double perVerify = decoder.VerifySteps > 0 ? (double)decoder.TokensDrafted / decoder.VerifySteps : 0;

            results.Add($"     k={k} pMin={pMin:F2}  {tps,6:F2} tok/s ({tps / plainTps:F2}x)  " +
                        $"acceptance {decoder.AcceptanceRate,6:P1}  drafted/verify {perVerify:F2}  " +
                        $"verify {decoder.VerifySteps,3} plain {decoder.PlainSteps,3} rollback {decoder.RollbackSteps,3}  " +
                        $"greedy-match {(diff < 0 ? maxNew : diff)}/{maxNew}");

            if (tps > bestTps) { bestTps = tps; bestConfig = $"k={k} pMin={pMin:F2}"; }
            if (k == 8 && Math.Abs(pMin - 0.75f) < 1e-6)
            {
                defaultAcceptance = decoder.AcceptanceRate;
                defaultDrafted = decoder.TokensDrafted;
                _output.WriteLine("     text: " + Preview(model, produced.ToArray()));
            }
        }

        // Print everything BEFORE asserting: a run that fails an assertion is
        // exactly the run whose numbers you need.
        _output.WriteLine("[4] drafting:");
        foreach (string r in results) _output.WriteLine(r);
        _output.WriteLine($"     best: {bestConfig} at {bestTps:F2} tok/s ({bestTps / plainTps:F2}x over plain)");

        Assert.True(defaultDrafted > 0, "the draft head never proposed a token at the default gate");
        // A NextN head trained with the model predicts its own trunk well; the
        // gate is what keeps that number high, so this is asserted at the
        // DEFAULT gate rather than at whatever the sweep's most permissive
        // setting happens to be.
        Assert.True(defaultAcceptance > 0.7,
            $"acceptance {defaultAcceptance:P1} at the default gate is far below what a trained NextN head " +
            "reaches — the draft block is probably reading the wrong hidden state or the wrong weights");
        Assert.True(bestTps > plainTps,
            $"no speculative configuration beat plain decode ({bestTps:F2} vs {plainTps:F2} tok/s)");
    }

    /// <summary>
    /// Loading without speculation must leave the model exactly as it was, and
    /// must not page in the draft block: it is ~3 GiB that would otherwise come
    /// out of the context the loader can size against the free VRAM.
    /// </summary>
    [Fact]
    public void GlmDsa_WithoutMtpSpecTheDraftBlockIsNotLoaded()
    {
        string modelPath = TryFindModel();
        if (modelPath == null) return;

        using var env = new EnvScope();
        env.Set("TS_MTP_SPEC", "0");
        env.Set("TS_GLM_MTP", "0");
        MoeCpuOffloadConfig.ConfigureFromEnvironment();

        using var model = ModelBase.Create(modelPath, ResolveBackend());
        var spec = (ISpeculativeModel)model;
        Assert.False(spec.HasDraftHead, "the draft block must stay unloaded unless speculation was requested");

        int[] prompt = model.Tokenizer.Encode(Prompt, addSpecial: true).ToArray();
        model.ResetKVCache();
        float[] logits = model.ForwardRefill(prompt);
        Assert.All(logits.Take(64), v => Assert.False(float.IsNaN(v)));
    }

    private static int FirstDifference(int[] reference, IReadOnlyList<int> other)
    {
        int n = Math.Min(reference.Length, other.Count);
        for (int i = 0; i < n; i++)
            if (reference[i] != other[i]) return i;
        return reference.Length == other.Count ? -1 : n;
    }

    /// <summary>The k highest-scoring indices, best first.</summary>
    private static int[] TopK(float[] v, int k) =>
        Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).ThenBy(i => i).Take(k).ToArray();

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static string Preview(ModelBase model, int[] tokens)
    {
        try
        {
            string s = model.Tokenizer.Decode(tokens.ToList()).Replace("\n", "\\n");
            return s.Length > 200 ? s.Substring(0, 200) + "..." : s;
        }
        catch { return string.Join(",", tokens.Take(24)); }
    }

    private static int EnvInt(string name, int fallback)
    {
        string raw = Environment.GetEnvironmentVariable(name);
        return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
    }

    private static BackendType ResolveBackend()
    {
        string name = Environment.GetEnvironmentVariable(EnvBackend);
        if (string.IsNullOrWhiteSpace(name))
            return BackendType.GgmlCuda;
        return Enum.TryParse<BackendType>(name, ignoreCase: true, out var parsed) ? parsed : BackendType.GgmlCuda;
    }

    private string TryFindModel()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set or missing - skipping GLM MTP model test.");
            return null;
        }
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
}
