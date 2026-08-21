// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// NextN/MTP speculative-decoding tests for glm-dsa, on the synthetic model so
// they run anywhere with no download.
//
// The load-bearing invariant is the LAST one: speculation is a speed
// optimization and nothing else, so a greedy speculative run must emit exactly
// the token stream plain greedy decoding emits. Everything else here exists to
// make a failure of that assertion diagnosable — a draft head that never drafts
// and a draft head that drafts garbage both produce a correct stream, just
// slowly, and only the acceptance counters tell them apart.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class GlmDsaMtpTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir;

    public GlmDsaMtpTests(ITestOutputHelper output)
    {
        _output = output;
        _dir = Path.Combine(Path.GetTempPath(), "ts-glm-mtp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string BuildModel(bool quantize = false) =>
        GlmDsaSyntheticModelBuilder.Write(Path.Combine(_dir, "tiny-glm-dsa.gguf"), null, quantize);

    /// <summary>24 tokens; with indexer top_k = 8 the trunk is on the SPARSE path,
    /// so the draft block's dense attention is genuinely a different code path.</summary>
    private static int[] Prompt24() => Enumerable.Range(0, 24).Select(i => 65 + (i * 7) % 50).ToArray();

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>Plain greedy decode — the reference stream speculation must reproduce.</summary>
    private static int[] PlainGreedy(ModelBase model, int[] prompt, int count)
    {
        model.ResetKVCache();
        float[] logits = model.ForwardRefill(prompt);
        var produced = new int[count];
        for (int i = 0; i < count; i++)
        {
            produced[i] = Argmax(logits);
            if (i + 1 < count) logits = model.Forward(new[] { produced[i] });
        }
        return produced;
    }

    [Fact]
    public void GlmDsa_MtpDraftHeadIsDetected()
    {
        using var model = ModelBase.Create(BuildModel(), BackendType.Cpu);
        var spec = Assert.IsAssignableFrom<IMtpSpeculativeModel>(model);

        Assert.True(spec.HasMtp, "the synthetic glm-dsa model ships a complete NextN block");
        Assert.Equal(GlmDsaSyntheticModelBuilder.Hidden, spec.MtpHiddenSize);
        // Per-token drafting, not a DSpark-style block drafter.
        Assert.Equal(0, spec.MtpDraftBlockSize);
        Assert.True(spec.MtpVerifyPersistsAcceptedKv,
            "glm-dsa has no recurrent state, so a verify's KV writes survive a partial rejection");
        Assert.True(spec.MtpSpeculationProfitable);
    }

    [Fact]
    public void GlmDsa_SpecForwardMatchesPlainForward()
    {
        string path = BuildModel();
        int[] prompt = Prompt24();
        int hidden = GlmDsaSyntheticModelBuilder.Hidden;
        int vocab = GlmDsaSyntheticModelBuilder.VocabSize;

        float[] plainLogits;
        using (var model = ModelBase.Create(path, BackendType.Cpu))
        {
            model.ResetKVCache();
            plainLogits = (float[])model.ForwardRefill(prompt).Clone();
        }

        using (var model = ModelBase.Create(path, BackendType.Cpu))
        {
            var spec = (IMtpSpeculativeModel)model;
            model.ResetKVCache();
            var h = new float[(long)prompt.Length * hidden];
            var logits = new float[vocab];
            spec.SpecForward(prompt, h, logits, allLogitsRows: false);

            // Same trunk, same caches, same advance — SpecForward only additionally
            // reads out the hidden states.
            Assert.Equal(prompt.Length, spec.CacheSeqLen);
            for (int v = 0; v < vocab; v++)
                Assert.Equal(plainLogits[v], logits[v], 4);

            // h_nextn must be a real post-norm hidden state, not a zeroed buffer.
            double norm = 0;
            for (int i = 0; i < hidden; i++) norm += h[i] * h[i];
            Assert.True(norm > 1e-6, "captured h_nextn row 0 is all zeros");
        }
    }

    [Fact]
    public void GlmDsa_SpecForwardAllRowsLogitsMatchPerRowDecode()
    {
        string path = BuildModel();
        int[] prompt = Prompt24();
        int vocab = GlmDsaSyntheticModelBuilder.VocabSize;
        int hidden = GlmDsaSyntheticModelBuilder.Hidden;

        // Row-by-row: forward one token at a time and keep each row's logits.
        var perRow = new float[prompt.Length][];
        using (var model = ModelBase.Create(path, BackendType.Cpu))
        {
            model.ResetKVCache();
            for (int i = 0; i < prompt.Length; i++)
                perRow[i] = (float[])model.Forward(new[] { prompt[i] }).Clone();
        }

        // One batch, all rows.
        using (var model = ModelBase.Create(path, BackendType.Cpu))
        {
            var spec = (IMtpSpeculativeModel)model;
            model.ResetKVCache();
            var h = new float[(long)prompt.Length * hidden];
            var logits = new float[(long)prompt.Length * vocab];
            spec.SpecForward(prompt, h, logits, allLogitsRows: true);

            for (int r = 0; r < prompt.Length; r++)
            {
                var row = new float[vocab];
                Array.Copy(logits, (long)r * vocab, row, 0, vocab);
                Assert.Equal(Argmax(perRow[r]), Argmax(row));
            }
        }
    }

    [Fact]
    public void GlmDsa_MtpDraftStepProducesUsableLogits()
    {
        using var model = ModelBase.Create(BuildModel(), BackendType.Cpu);
        var spec = (IMtpSpeculativeModel)model;
        int hidden = spec.MtpHiddenSize;
        int vocab = model.Config.VocabSize;
        int[] prompt = Prompt24();

        model.ResetKVCache();
        var hAll = new float[(long)prompt.Length * hidden];
        var promptLogits = new float[vocab];
        spec.SpecForward(prompt, hAll, promptLogits, allLogitsRows: false);

        // Catch the draft block up over the prompt, pairing token k with the
        // hidden state of token k-1 (row 0 carries in zeros, as at generation start).
        var paired = new float[(long)prompt.Length * hidden];
        Array.Copy(hAll, 0, paired, hidden, (long)(prompt.Length - 1) * hidden);
        spec.MtpCatchUp(prompt, paired, 0);

        int next = Argmax(promptLogits);
        var lastH = new float[hidden];
        Array.Copy(hAll, (long)(prompt.Length - 1) * hidden, lastH, 0, hidden);

        var draftLogits = new float[vocab];
        var draftH = new float[hidden];
        spec.MtpDraftStep(next, lastH, prompt.Length, draftLogits, draftH);

        Assert.All(draftLogits, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        double hNorm = 0;
        for (int i = 0; i < hidden; i++) hNorm += draftH[i] * draftH[i];
        Assert.True(hNorm > 1e-6, "the draft block returned a zero hidden state");

        int drafted = Argmax(draftLogits);
        Assert.InRange(drafted, 0, vocab - 1);
        _output.WriteLine($"prompt -> {next}, draft head proposes {drafted}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GlmDsa_SpeculativeGreedyMatchesPlainGreedy(bool quantize)
    {
        string path = BuildModel(quantize);
        const int maxNew = 24;
        int[] prompt = Prompt24();

        int[] plain;
        using (var model = ModelBase.Create(path, BackendType.Cpu))
            plain = PlainGreedy(model, prompt, maxNew);

        using (var model = ModelBase.Create(path, BackendType.Cpu))
        {
            var spec = (IMtpSpeculativeModel)model;
            Assert.True(spec.HasMtp);
            var decoder = new MtpSpeculativeDecoder(spec, maxDraftTokens: 4)
            {
                // Force drafting on: the cost governor exists to protect
                // throughput, and on a 1.9 MB model it would measure noise.
                AdaptiveSpeculation = false,
                // The synthetic weights are random, so the draft head's
                // distribution over 128 tokens is near-uniform and the production
                // 0.75 confidence gate would stop every window at zero tokens.
                // A low gate is what makes this test exercise draft/verify at all;
                // it cannot make the RESULT wrong, which is the whole point.
                MinDraftProb = 0.02f,
            };
            List<int> speculative = decoder.GenerateGreedy(prompt, maxNew);

            _output.WriteLine($"quantized={quantize} drafted={decoder.TokensDrafted} " +
                              $"accepted={decoder.TokensAccepted} acceptance={decoder.AcceptanceRate:P1} " +
                              $"verify={decoder.VerifySteps} plain={decoder.PlainSteps} " +
                              $"rollback={decoder.RollbackSteps}");
            _output.WriteLine("plain: " + string.Join(",", plain));
            _output.WriteLine("spec : " + string.Join(",", speculative));

            Assert.Equal(maxNew, speculative.Count);
            Assert.Equal(plain, speculative.ToArray());
            Assert.True(decoder.TokensDrafted > 0, "the draft head never proposed a token");
        }
    }

    [Fact]
    public void GlmDsa_SpeculationLeavesTheTrunkCacheWhereGreedyWouldHave()
    {
        // Rejected speculative tokens must not survive in the trunk cache. The
        // check that actually proves it is the CONTINUATION: feed the cache the
        // one token greedy decoding would not yet have forwarded and compare the
        // logits against a plain run at the same point. Stale KV rows from a
        // rejected window change those logits; a length check alone would not
        // notice.
        string path = BuildModel();
        const int maxNew = 16;
        int[] prompt = Prompt24();

        int[] plain;
        float[] plainNext;
        using (var model = ModelBase.Create(path, BackendType.Cpu))
        {
            plain = PlainGreedy(model, prompt, maxNew);
            plainNext = (float[])model.Forward(new[] { plain[^1] }).Clone();
        }

        using var specModel = ModelBase.Create(path, BackendType.Cpu);
        var spec = (IMtpSpeculativeModel)specModel;
        var decoder = new MtpSpeculativeDecoder(spec, maxDraftTokens: 4)
        {
            AdaptiveSpeculation = false,
            MinDraftProb = 0.02f,
        };
        List<int> produced = decoder.GenerateGreedy(prompt, maxNew);
        Assert.Equal(plain, produced.ToArray());

        // Greedy decoding forwards every emitted token except the last (it is
        // sampled from the previous step's logits and fed on the NEXT step), so
        // the trunk holds prompt + N - 1.
        Assert.Equal(prompt.Length + produced.Count - 1, spec.CacheSeqLen);

        float[] specNext = specModel.Forward(new[] { produced[^1] });
        Assert.Equal(prompt.Length + produced.Count, spec.CacheSeqLen);
        for (int v = 0; v < specNext.Length; v++)
            Assert.Equal(plainNext[v], specNext[v], 3);
    }

    [Fact]
    public void GlmDsa_PartialRejectionKeepsTheAcceptedPrefixKv()
    {
        // Drive the window deliberately wide so rejections are frequent, then
        // check the emitted stream still matches greedy. This is the path
        // MtpVerifyPersistsAcceptedKv takes: no rollback re-forward, just a
        // position rewind, so a bug here shows up as a corrupted continuation
        // rather than a crash.
        string path = BuildModel();
        const int maxNew = 20;
        int[] prompt = Prompt24();

        int[] plain;
        using (var model = ModelBase.Create(path, BackendType.Cpu))
            plain = PlainGreedy(model, prompt, maxNew);

        using (var model = ModelBase.Create(path, BackendType.Cpu))
        {
            var spec = (IMtpSpeculativeModel)model;
            var decoder = new MtpSpeculativeDecoder(spec, maxDraftTokens: 8)
            {
                AdaptiveSpeculation = false,
                // Draft regardless of confidence so the window is always full and
                // rejections are guaranteed.
                MinDraftProb = 0f,
                // (see GlmDsa_SpeculativeGreedyMatchesPlainGreedy on why the
                // synthetic model needs the gate opened)
            };
            List<int> speculative = decoder.GenerateGreedy(prompt, maxNew);
            _output.WriteLine($"drafted={decoder.TokensDrafted} accepted={decoder.TokensAccepted} " +
                              $"rollbacks={decoder.RollbackSteps}");

            Assert.True(decoder.RollbackSteps > 0, "expected at least one partially-rejected window");
            Assert.Equal(plain, speculative.ToArray());
        }
    }

    [Fact]
    public void GlmDsa_TrunkGoldensAreUnchangedByTheMtpBlock()
    {
        // The MTP block adds a cache layer and a set of weight handles. The trunk
        // must be bit-for-bit what it was: these are the llama.cpp goldens the
        // architecture tests pin.
        using var model = ModelBase.Create(BuildModel(), BackendType.Cpu);
        int[] produced = PlainGreedy(model, Prompt24(), 8);
        Assert.Equal(new[] { 84, 46, 77, 48, 3, 37, 9, 3 }, produced);
    }
}
