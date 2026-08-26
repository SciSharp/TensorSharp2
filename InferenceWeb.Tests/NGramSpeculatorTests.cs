// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The weight-free speculator: suffix matching over the sequence's own tokens.
// It needs no draft head, so these tests are the proof that the algorithm layer
// is genuinely decoupled from the model layer — the same shared loop that
// drives a NextN head drives this with no model support beyond SpecForward.
using System;
using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

public class NGramSpeculatorTests
{
    private static DraftContext Ctx(int lastToken, int position, int maxTokens) => new()
    {
        LastToken = lastToken,
        Position = position,
        MaxTokens = maxTokens,
    };

    [Fact]
    public void Propose_RepeatedPhrase_DraftsTheContinuationThatFollowedItBefore()
    {
        var spec = new NGramSpeculator(maxDraftTokens: 8);
        // "the quick brown fox jumped" appears once; the sequence then repeats
        // "the quick brown" and the drafter should propose "fox jumped".
        int[] corpus = { 10, 20, 30, 40, 50, 60, 70, 10, 20 };
        spec.Commit(corpus, null, 0);

        var drafts = new List<int>();
        // Last committed token is 20 (index 8); the next token is 30 — which is
        // the pattern (10, 20, 30) seen at index 0..2.
        int n = spec.Propose(Ctx(lastToken: 30, position: corpus.Length, maxTokens: 8), drafts);

        Assert.Equal(new[] { 40, 50, 60, 70, 10, 20 }, drafts);
        Assert.Equal(drafts.Count, n);
    }

    [Fact]
    public void Propose_NoRepetition_DraftsNothingAndCostsNoVerifyRow()
    {
        var spec = new NGramSpeculator(maxDraftTokens: 8);
        spec.Commit(Enumerable.Range(1, 40).ToArray(), null, 0);

        var drafts = new List<int>();
        // 999 has never been seen, so no suffix ending in it can match.
        Assert.Equal(0, spec.Propose(Ctx(lastToken: 999, position: 40, maxTokens: 8), drafts));
        Assert.Empty(drafts);
    }

    [Fact]
    public void Propose_PrefersTheLongestMatchingSuffix()
    {
        var spec = new NGramSpeculator(maxDraftTokens: 4);
        // (1,2) is followed by 3 early on and by 99 later. The longer context
        // (7,1,2) only ever occurred before the 99, so the longer match wins.
        spec.Commit(new[] { 1, 2, 3, 4, 5, 7, 1, 2, 99, 98, 5, 6, 7, 1 }, null, 0);

        var drafts = new List<int>();
        spec.Propose(Ctx(lastToken: 2, position: 14, maxTokens: 4), drafts);

        Assert.Equal(new[] { 99, 98, 5, 6 }, drafts);
    }

    [Fact]
    public void Propose_HonoursTheWindowCap()
    {
        var spec = new NGramSpeculator(maxDraftTokens: 8);
        spec.Commit(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1 }, null, 0);

        var drafts = new List<int>();
        spec.Propose(Ctx(lastToken: 2, position: 10, maxTokens: 3), drafts);

        Assert.Equal(new[] { 3, 4, 5 }, drafts);
    }

    [Fact]
    public void MinDraftProb_ScalesTheRequiredMatchLength()
    {
        // A 2-token match exists; nothing longer does. The default gate accepts
        // it; a gate demanding a full MaxNgram match does not.
        int[] corpus = { 5, 6, 7, 8, 9, 40, 41, 42, 5 };

        var loose = new NGramSpeculator(maxDraftTokens: 4, minNgram: 2, maxNgram: 8);
        loose.Commit(corpus, null, 0);
        var loosely = new List<int>();
        loose.Propose(Ctx(lastToken: 6, position: corpus.Length, maxTokens: 4), loosely);
        Assert.Equal(new[] { 7, 8, 9, 40 }, loosely);

        var strict = new NGramSpeculator(maxDraftTokens: 4, minNgram: 2, maxNgram: 8)
        {
            MinDraftProb = 1f,      // demand a full 8-token suffix match
        };
        strict.Commit(corpus, null, 0);
        var strictly = new List<int>();
        Assert.Equal(0, strict.Propose(Ctx(lastToken: 6, position: corpus.Length, maxTokens: 4), strictly));
    }

    [Fact]
    public void Propose_PositionOutOfSyncWithTheCorpus_DraftsNothing()
    {
        // Attached mid-sequence: the corpus is not this sequence's history, so
        // mining it would propose text from somewhere else. Draft nothing.
        var spec = new NGramSpeculator(maxDraftTokens: 4);
        spec.Commit(new[] { 1, 2, 3, 1, 2 }, null, 0);

        var drafts = new List<int>();
        Assert.Equal(0, spec.Propose(Ctx(lastToken: 3, position: 999, maxTokens: 4), drafts));
    }

    [Fact]
    public void Reset_DropsTheCorpus()
    {
        var spec = new NGramSpeculator(maxDraftTokens: 4);
        spec.Commit(new[] { 1, 2, 3, 4, 1, 2 }, null, 0);
        spec.Reset();

        var drafts = new List<int>();
        Assert.Equal(0, spec.Propose(Ctx(lastToken: 3, position: 0, maxTokens: 4), drafts));
    }

    [Fact]
    public void Commit_IncrementalAndBulk_ProduceTheSameProposals()
    {
        // The engine commits a few tokens per step; the standalone decoder
        // commits a whole prompt chunk. The index must not care.
        int[] tokens = { 4, 8, 15, 16, 23, 42, 4, 8, 15 };

        var bulk = new NGramSpeculator(maxDraftTokens: 4);
        bulk.Commit(tokens, null, 0);

        var incremental = new NGramSpeculator(maxDraftTokens: 4);
        for (int i = 0; i < tokens.Length; i++)
            incremental.Commit(new[] { tokens[i] }, null, i);

        var a = new List<int>();
        var b = new List<int>();
        bulk.Propose(Ctx(lastToken: 16, position: tokens.Length, maxTokens: 4), a);
        incremental.Propose(Ctx(lastToken: 16, position: tokens.Length, maxTokens: 4), b);

        Assert.Equal(new[] { 23, 42, 4, 8 }, a);
        Assert.Equal(a, b);
    }

    [Fact]
    public void Describe_NamesTheAlgorithmAndItsOrders()
    {
        var spec = new NGramSpeculator(maxDraftTokens: 4, minNgram: 3, maxNgram: 6);
        Assert.Equal(SpeculatorRegistry.NGram, spec.Name);
        Assert.Equal("ngram(n=3..6)", spec.Describe());
        Assert.False(spec.NeedsHiddenState);
        Assert.False(spec.HandlesOwnPrefill);
    }
}
