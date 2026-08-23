// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// End-to-end proof that the algorithm layer is decoupled from the model layer:
// a model that implements ONLY ISpeculativeTarget — no draft head, no
// speculator weights, nothing a checkpoint has to ship — is driven through the
// SAME SpeculativeExecution loop that serves a NextN/MTP head, and the emitted
// stream is identical to plain decoding.
using System;
using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

public class NGramSpeculativeDecodeTests
{
    private const int VocabSize = 64;

    [Fact]
    public void NGramSpeculation_OnAModelWithNoDraftHead_MatchesPlainGreedyExactly()
    {
        // The trunk's continuation is periodic (the model "quotes" itself), which
        // is the regime prompt-lookup decoding exists for: summarizing, editing
        // or answering about a document, repetitive structured output, agentic
        // loops. On free-form text it simply finds nothing and every step
        // degrades to a plain decode.
        var plainModel = new PeriodicTrunk(period: 7);
        var plain = PlainGreedy(plainModel, promptLen: 12, count: 40);

        var specModel = new PeriodicTrunk(period: 7);
        var exec = NewNGramExec(specModel, maxDraftTokens: 8);
        var speculative = SpeculativeGreedy(specModel, exec, promptLen: 12, count: 40);

        Assert.Equal(plain, speculative);
        Assert.True(exec.Stats.TokensAccepted > 0,
            $"n-gram never had a draft accepted (drafted={exec.Stats.TokensDrafted})");
        // The whole point: fewer trunk passes for the same tokens.
        Assert.True(exec.Stats.VerifySteps + exec.Stats.PlainSteps < 40,
            "speculation took no fewer trunk passes than plain decoding");
        Assert.Empty(specModel.ProtocolViolations);
    }

    [Fact]
    public void NGramSpeculation_NonRepeatingTrunk_StillMatchesPlainGreedy()
    {
        // No suffix ever repeats, so every step degrades to a plain decode. The
        // stream must be identical and nothing must break.
        var plainModel = new PeriodicTrunk(period: 0);
        var plain = PlainGreedy(plainModel, promptLen: 8, count: 24);

        var specModel = new PeriodicTrunk(period: 0);
        var exec = NewNGramExec(specModel, maxDraftTokens: 8);
        var speculative = SpeculativeGreedy(specModel, exec, promptLen: 8, count: 24);

        Assert.Equal(plain, speculative);
        Assert.Empty(specModel.ProtocolViolations);
    }

    [Fact]
    public void NGramSpeculation_NeedsNoHiddenStateCapture()
    {
        // A weight-free speculator lets the loop skip the trunk's hidden-state
        // tap entirely — the model must never be asked for one.
        var model = new PeriodicTrunk(period: 5);
        var exec = NewNGramExec(model, maxDraftTokens: 8);
        SpeculativeGreedy(model, exec, promptLen: 10, count: 30);

        Assert.Equal(0, model.HiddenStateRequests);
    }

    private static SpeculativeExecution NewNGramExec(PeriodicTrunk model, int maxDraftTokens)
    {
        var speculator = SpeculatorRegistry.Create(
            model,
            new SpeculationOptions
            {
                Enabled = true,
                SpeculatorName = SpeculatorRegistry.NGram,
                MaxDraftTokens = maxDraftTokens,
            },
            out string decline);
        Assert.True(speculator != null, decline);
        var exec = new SpeculativeExecution(model, speculator);
        // Measure the algorithm, not the governor: on a fake model every step
        // costs the same, so the cost governor's verdict is noise.
        exec.AdaptiveSpeculation = false;
        return exec;
    }

    private static List<int> PlainGreedy(PeriodicTrunk model, int promptLen, int count)
    {
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();
        float[] logits = model.Forward(prompt);
        var output = new List<int>();
        int t = Argmax(logits);
        output.Add(t);
        for (int i = 1; i < count; i++)
        {
            logits = model.Forward(new[] { t });
            t = Argmax(logits);
            output.Add(t);
        }
        return output;
    }

    private static List<int> SpeculativeGreedy(PeriodicTrunk model, SpeculativeExecution exec,
        int promptLen, int count)
    {
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();
        float[] logits = exec.PrefillStep(prompt, 0);

        var output = new List<int>();
        int last = Argmax(logits);
        output.Add(last);
        int pos = promptLen;

        while (output.Count < count)
        {
            var accepted = new List<int>();
            var outcome = exec.DecodeStep(last, pos, count - output.Count,
                drawNext: Argmax, onDraftAccepted: accepted.Add);

            if (!outcome.UsedSpeculation)
            {
                last = Argmax(outcome.NextLogits);
                pos += 1;
                output.Add(last);
                continue;
            }

            pos += outcome.AcceptedCount + 1;
            last = outcome.NextToken;
            foreach (int d in accepted)
            {
                if (output.Count >= count) return output;
                output.Add(d);
            }
            if (output.Count < count)
                output.Add(outcome.NextToken);
        }
        return output;
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>
    /// A trunk with no draft head whatsoever. Its next-token function repeats
    /// with <c>period</c> (0 = never repeats), so an n-gram drafter can find the
    /// continuation in the context exactly the way it does on real repetitive
    /// text. Records any request for hidden states, which a weight-free
    /// speculator must never trigger.
    /// </summary>
    private sealed class PeriodicTrunk : ISpeculativeTarget
    {
        private readonly int _period;
        private readonly List<int> _trunk = new();

        public PeriodicTrunk(int period) => _period = period;

        public List<string> ProtocolViolations { get; } = new();
        public int HiddenStateRequests { get; private set; }

        /// <summary>The trunk's argmax for the row of the token at position p.</summary>
        private int Next(int p) => _period > 0
            ? 3 + (p % _period)
            : 3 + ((p * 13 + 7) % (VocabSize - 4));

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize, HiddenSize = 4 };
        public ITokenizer Tokenizer => null;
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public void TruncateKVCache(int tokenCount) { }
        public void ResetKVCache() => _trunk.Clear();
        public void Dispose() { }

        public float[] Forward(int[] tokens)
        {
            int start = _trunk.Count;
            _trunk.AddRange(tokens);
            var logits = new float[VocabSize];
            logits[Next(start + tokens.Length - 1)] = 10f;
            return logits;
        }

        public int CacheSeqLen => _trunk.Count;
        public int MaxContextLength => 4096;

        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (hAllOut != null)
                HiddenStateRequests++;
            int start = _trunk.Count;
            _trunk.AddRange(tokens);
            if (allLogitsRows)
            {
                for (int i = 0; i < tokens.Length; i++)
                {
                    Array.Clear(logitsOut, i * VocabSize, VocabSize);
                    logitsOut[i * VocabSize + Next(start + i)] = 10f;
                }
            }
            else
            {
                Array.Clear(logitsOut, 0, VocabSize);
                logitsOut[Next(start + tokens.Length - 1)] = 10f;
            }
        }

        public void SpecEnsureCapacity(int requiredSeqLen) { }
        public void SpecSnapshotRecurrentState() { }
        public void SpecRestoreRecurrentState() { }

        public void SpecRewindCache(int length)
        {
            if (length < 0 || length > _trunk.Count)
            {
                ProtocolViolations.Add($"rewind to {length} outside [0, {_trunk.Count}]");
                return;
            }
            _trunk.RemoveRange(length, _trunk.Count - length);
        }
    }
}
