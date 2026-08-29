// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Gemma 3's vLLM-style batched paged forward must agree with the per-sequence
// path it replaces. Gemma 3 is the architecture where this is easiest to get
// subtly wrong: five of every six layers are sliding-window, Q is pre-scaled
// before attention rather than inside it, and RoPE switches frequency family
// between global and local layers. Any of those going astray still produces
// fluent-looking logits, so compare the vectors, not the vibes.
//
// Opt-in via TS_TEST_MODEL_DIR containing a gemma-3-*.gguf.
using System.Linq;

using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma3BatchedForwardTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    private readonly ITestOutputHelper _output;
    public Gemma3BatchedForwardTests(ITestOutputHelper output) { _output = output; }

    [ModelFact("TS_TEST_MODEL_DIR", "gemma-3")]
    public void Gemma3_BatchSize1_ForwardBatchMatchesPerSequence()
    {
        var model = TryLoadGemma3();
        if (model == null) return;
        try
        {
            var prompt = model.Tokenizer.Encode("The capital of France is", addSpecial: true).ToArray();
            const int blockSize = 16;

            model.ResetKVCache();
            float[] legacyLogits = model.Forward(prompt);
            int legacyTop1 = ArgMax(legacyLogits);

            model.ResetKVCache();
            var pool = new BlockPool(numBlocks: 16, blockSize: blockSize,
                blockByteSize: model.ComputeKVBlockByteSize(blockSize));
            var seq = new SequenceState("r0", prompt, maxNewTokens: 1, blockSize: blockSize,
                samplingConfig: SamplingConfig.Default);
            foreach (var b in pool.AllocateNew(BlocksFor(prompt.Length, blockSize)))
                seq.BlockTable.AppendBlock(b);

            var batched = (IBatchedPagedModel)model;
            Assert.True(batched.BatchedForwardAvailable,
                "Gemma 3 should offer its batched paged path on a single-device run.");

            var perSeqLogits = batched.ForwardBatch(BuildContext(seq, prompt, blockSize));
            Assert.Single(perSeqLogits);

            float[] batchedVec = perSeqLogits[0];
            Assert.Equal(legacyLogits.Length, batchedVec.Length);

            int batchedTop1 = ArgMax(batchedVec);
            _output.WriteLine($"[gemma3] legacy top-1={legacyTop1} '{model.Tokenizer.Decode(new List<int> { legacyTop1 })}', " +
                              $"batched top-1={batchedTop1} '{model.Tokenizer.Decode(new List<int> { batchedTop1 })}'");

            double dot = 0, nL = 0, nB = 0;
            for (int i = 0; i < legacyLogits.Length; i++)
            {
                dot += (double)legacyLogits[i] * batchedVec[i];
                nL += (double)legacyLogits[i] * legacyLogits[i];
                nB += (double)batchedVec[i] * batchedVec[i];
            }
            double cosine = dot / (Math.Sqrt(nL) * Math.Sqrt(nB) + 1e-12);
            _output.WriteLine($"[gemma3] logit cosine similarity = {cosine:F6}");

            // The two paths run different kernels over the same math, so exact
            // equality is not expected; a real bug (wrong RoPE family on a layer
            // class, an unmasked sliding window, a missing Q pre-scale) drops this
            // far below 0.99, while FP reassociation alone stays above it.
            Assert.True(cosine >= 0.99, $"logit cosine was {cosine:F6} (< 0.99) - the batched path diverges");
            Assert.Equal(legacyTop1, batchedTop1);
        }
        finally
        {
            model.Dispose();
        }
    }

    /// <summary>
    /// Two sequences of DIFFERENT lengths in one batch, which is the case that
    /// actually exercises the block tables: sequence 1's keys must not leak into
    /// sequence 0's attention, and each sequence's sliding window is measured from
    /// its OWN position, not the batch's.
    /// </summary>
    [ModelFact("TS_TEST_MODEL_DIR", "gemma-3")]
    public void Gemma3_TwoSequences_EachMatchesItsOwnPerSequenceRun()
    {
        var model = TryLoadGemma3();
        if (model == null) return;
        try
        {
            var promptA = model.Tokenizer.Encode("The capital of France is", addSpecial: true).ToArray();
            var promptB = model.Tokenizer.Encode("Water boils at a temperature of", addSpecial: true).ToArray();
            const int blockSize = 16;

            model.ResetKVCache();
            float[] refA = (float[])model.Forward(promptA).Clone();
            model.ResetKVCache();
            float[] refB = (float[])model.Forward(promptB).Clone();

            model.ResetKVCache();
            var pool = new BlockPool(numBlocks: 64, blockSize: blockSize,
                blockByteSize: model.ComputeKVBlockByteSize(blockSize));
            var seqA = new SequenceState("rA", promptA, maxNewTokens: 1, blockSize: blockSize,
                samplingConfig: SamplingConfig.Default);
            var seqB = new SequenceState("rB", promptB, maxNewTokens: 1, blockSize: blockSize,
                samplingConfig: SamplingConfig.Default);
            foreach (var b in pool.AllocateNew(BlocksFor(promptA.Length, blockSize))) seqA.BlockTable.AppendBlock(b);
            foreach (var b in pool.AllocateNew(BlocksFor(promptB.Length, blockSize))) seqB.BlockTable.AppendBlock(b);

            var ctx = BuildContext2(seqA, promptA, seqB, promptB, blockSize);
            var got = ((IBatchedPagedModel)model).ForwardBatch(ctx);
            Assert.Equal(2, got.Count);

            AssertCosine(refA, got[0], "sequence A");
            AssertCosine(refB, got[1], "sequence B");
            Assert.Equal(ArgMax(refA), ArgMax(got[0]));
            Assert.Equal(ArgMax(refB), ArgMax(got[1]));
        }
        finally
        {
            model.Dispose();
        }
    }

    private void AssertCosine(float[] reference, float[] actual, string label)
    {
        Assert.Equal(reference.Length, actual.Length);
        double dot = 0, nR = 0, nA = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            dot += (double)reference[i] * actual[i];
            nR += (double)reference[i] * reference[i];
            nA += (double)actual[i] * actual[i];
        }
        double cosine = dot / (Math.Sqrt(nR) * Math.Sqrt(nA) + 1e-12);
        _output.WriteLine($"[gemma3] {label} cosine = {cosine:F6}");
        Assert.True(cosine >= 0.99, $"{label}: logit cosine was {cosine:F6} (< 0.99)");
    }

    private static int BlocksFor(int tokens, int blockSize) => (tokens + blockSize - 1) / blockSize;

    private static BatchedForwardContext BuildContext(SequenceState seq, int[] tokens, int blockSize)
    {
        var ctx = new BatchedForwardContext
        {
            Sequences = new List<SequenceState> { seq },
            NumScheduledTokens = new List<int> { tokens.Length },
            QueryStartLoc = new List<int> { 0, tokens.Length },
            Positions = Enumerable.Range(0, tokens.Length).ToList(),
            SlotMapping = new List<int>(),
            BlockTables = new int[1][],
            MaxQueryLen = tokens.Length,
            MaxSeqLen = tokens.Length,
        };
        var ids = new int[seq.BlockTable.NumBlocks];
        for (int b = 0; b < ids.Length; b++) ids[b] = seq.BlockTable.Blocks[b].Id;
        ctx.BlockTables[0] = ids;
        for (int t = 0; t < tokens.Length; t++)
            ctx.SlotMapping.Add(ids[t / blockSize] * blockSize + (t % blockSize));
        return ctx;
    }

    private static BatchedForwardContext BuildContext2(
        SequenceState a, int[] ta, SequenceState b, int[] tb, int blockSize)
    {
        var ctx = new BatchedForwardContext
        {
            Sequences = new List<SequenceState> { a, b },
            NumScheduledTokens = new List<int> { ta.Length, tb.Length },
            QueryStartLoc = new List<int> { 0, ta.Length, ta.Length + tb.Length },
            Positions = Enumerable.Range(0, ta.Length).Concat(Enumerable.Range(0, tb.Length)).ToList(),
            SlotMapping = new List<int>(),
            BlockTables = new int[2][],
            MaxQueryLen = Math.Max(ta.Length, tb.Length),
            MaxSeqLen = Math.Max(ta.Length, tb.Length),
        };
        foreach (var (seq, toks, idx) in new[] { (a, ta, 0), (b, tb, 1) })
        {
            var ids = new int[seq.BlockTable.NumBlocks];
            for (int i = 0; i < ids.Length; i++) ids[i] = seq.BlockTable.Blocks[i].Id;
            ctx.BlockTables[idx] = ids;
            for (int t = 0; t < toks.Length; t++)
                ctx.SlotMapping.Add(ids[t / blockSize] * blockSize + (t % blockSize));
        }
        return ctx;
    }

    private Gemma3Model TryLoadGemma3()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set; skipping");
            return null;
        }
        string path = Directory.GetFiles(dir, "*.gguf").FirstOrDefault(p =>
        {
            string n = Path.GetFileName(p).ToLowerInvariant();
            return n.Contains("gemma-3") && !n.Contains("mmproj");
        });
        if (path == null)
        {
            _output.WriteLine("No Gemma 3 GGUF available; skipping");
            return null;
        }
        _output.WriteLine($"[gemma3] loading {Path.GetFileName(path)}");
        try
        {
            BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
            return (Gemma3Model)ModelBase.Create(path, backend);
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Failed to load: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}
