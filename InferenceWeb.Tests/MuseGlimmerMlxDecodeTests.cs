// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Coverage for the Muse-Glimmer MLX pipelined greedy decode path
// (MuseGlimmerModel.MlxDecode.cs) and the batched-graph helper it is built on
// (MlxFusedOps.RunDecodeGraphBatched).
//
// Everything here runs without the 10 GB GGUF: the eligibility gate is a pure
// predicate on backend/state, and the batching helper only needs the MLX worker
// thread (not a model). The numerical half of the path — that a pipelined step
// reproduces ggml_metal token for token — is verified by the CLI parity runs
// documented in MuseGlimmerModel.MlxDecode.cs, since it needs real weights.
using System;
using System.Threading;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class MuseGlimmerMlxDecodeTests
{
    // The pipelined step computes argmax and the next embedding ON DEVICE with
    // MLX-only entry points, so every non-MLX backend has to fall through to
    // the standard Forward + host-sample loop (GGML/CUDA have their own
    // whole-model fused graph instead — see MuseGlimmerModel.Fused.cs).
    [Theory]
    [InlineData(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlVulkan)]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Cpu)]
    public void PipelinedGreedy_IsRefusedOffMlx(BackendType backend)
    {
        Assert.False(MuseGlimmerModel.IsPipelinedGreedyEligible(
            backend, isTensorParallel: false, hasPendingVisionEmbeddings: false, kvSwaRows: 0));
    }

    [Fact]
    public void PipelinedGreedy_IsAllowedOnMlx()
    {
        Assert.True(MuseGlimmerModel.IsPipelinedGreedyEligible(
            BackendType.Mlx, isTensorParallel: false, hasPendingVisionEmbeddings: false, kvSwaRows: 0));
    }

    // Under tensor parallelism the KV caches live in the per-rank TP arrays and
    // every layer ends in a collective, so a "no host sync" step is not
    // expressible.
    [Fact]
    public void PipelinedGreedy_IsRefusedUnderTensorParallelism()
    {
        Assert.False(MuseGlimmerModel.IsPipelinedGreedyEligible(
            BackendType.Mlx, isTensorParallel: true, hasPendingVisionEmbeddings: false, kvSwaRows: 0));
    }

    // Pending vision embeddings have to be spliced into a MULTI-row forward
    // before any decode step runs; a single-row decode would silently drop the
    // image.
    [Fact]
    public void PipelinedGreedy_IsRefusedWithPendingVisionEmbeddings()
    {
        Assert.False(MuseGlimmerModel.IsPipelinedGreedyEligible(
            BackendType.Mlx, isTensorParallel: false, hasPendingVisionEmbeddings: true, kvSwaRows: 0));
    }

    // A sliding-window RING row is not an absolute position, and only the fused
    // GGML kernel can read that layout (MuseGlimmerModel.RequireUniformCache).
    // The ring is never armed on MLX today; the guard is what keeps a future
    // change from turning into silently wrong attention instead of a throw.
    [Fact]
    public void PipelinedGreedy_IsRefusedWhenSlidingWindowRingIsArmed()
    {
        Assert.False(MuseGlimmerModel.IsPipelinedGreedyEligible(
            BackendType.Mlx, isTensorParallel: false, hasPendingVisionEmbeddings: false, kvSwaRows: 4352));
    }

    // ================================================================
    // MlxFusedOps.RunDecodeGraphBatched
    // ================================================================
    //
    // The whole point of the wrapper is that the callback runs ON the MLX
    // worker thread, because that is what makes MlxWorker.Invoke short-circuit
    // to a direct call (MlxWorker.IsOnWorkerThread) for the ~1150 nested MLX
    // calls a 52-layer decode step issues.

    [MlxFact]
    public void RunDecodeGraphBatched_RunsCallbackOnTheMlxWorkerThread()
    {
        Assert.False(MlxFusedOps.IsBuildingBatchedGraph);

        int callerThread = Thread.CurrentThread.ManagedThreadId;
        var (insideBatched, insideThread) = MlxFusedOps.RunDecodeGraphBatched(
            () => (MlxFusedOps.IsBuildingBatchedGraph, Thread.CurrentThread.ManagedThreadId));

        Assert.True(insideBatched);
        Assert.NotEqual(callerThread, insideThread);
        Assert.False(MlxFusedOps.IsBuildingBatchedGraph);
    }

    // Nesting has to be free AND ordered: a decode layer calls straight back
    // into MlxFusedOps / MlxQuantizedOps entry points that Invoke on their own.
    [MlxFact]
    public void RunDecodeGraphBatched_NestedInvokesRunInlineOnTheSameThread()
    {
        var (outer, inner) = MlxFusedOps.RunDecodeGraphBatched(() =>
        {
            int outerThread = Thread.CurrentThread.ManagedThreadId;
            int innerThread = MlxFusedOps.RunDecodeGraphBatched(() => Thread.CurrentThread.ManagedThreadId);
            return (outerThread, innerThread);
        });

        Assert.Equal(outer, inner);
    }

    // A failure inside the graph build must surface at the call site rather
    // than being swallowed on the worker (which would leave the caller holding
    // a half-built step and no error).
    [MlxFact]
    public void RunDecodeGraphBatched_PropagatesExceptions()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => MlxFusedOps.RunDecodeGraphBatched<int>(() => throw new InvalidOperationException("boom")));
        Assert.Equal("boom", ex.Message);
    }

    [Fact]
    public void RunDecodeGraphBatched_RejectsNullCallback()
    {
        Assert.Throws<ArgumentNullException>(() => MlxFusedOps.RunDecodeGraphBatched<int>(null));
        Assert.Throws<ArgumentNullException>(() => MlxFusedOps.RunDecodeGraphBatched((Action)null));
    }
}
