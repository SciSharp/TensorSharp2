// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Linq;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for <see cref="ExecutionPlanner"/> — the single decision point
/// that maps (model+backend capabilities, operator overrides, engine config,
/// step features) to an execution plan. Because the planner is a pure
/// function, every path combination that used to be emergent control flow in
/// BatchExecutor.ExecuteStep is directly assertable here, without a model.
/// </summary>
public class ExecutionPlannerTests
{
    private static readonly SchedulerConfig PlainConfig = new();
    private static readonly SchedulerConfig SpecConfig =
        new() { Speculation = new SpeculationOptions { Enabled = true } };

    /// <summary>A model+backend that supports the full batched contract
    /// (like Qwen 3.5 on CUDA), used as the baseline for the tests.</summary>
    private static ExecutionCapabilities BatchedCaps() => new()
    {
        SupportsBatchedPagedAttention = true,
        BatchedForwardAvailable = true,
        SupportsBatchedMultimodal = false,
        SupportsPerSequenceFusedForward = false,
        SupportsLinearKvMigration = true,
        SupportsKvStateSnapshot = true,
        SupportsCrossSequenceKvReuse = true,
    };

    private static ExecutionStepFeatures Seqs(int count) => new() { SequenceCount = count };

    // ----- batched vs fallback -----

    [Fact]
    public void MultiSequenceTextStep_SelectsBatchedPaged_WithPerSeqFallback()
    {
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), ExecutionOptions.Default, PlainConfig, Seqs(3));

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
        // BatchedPaged is declinable (migration failure / model refusal), so
        // the chain must terminate in the universal per-seq fallback.
        Assert.Equal(ExecutionPathKind.PerSequence, plan.Candidates[^1]);
    }

    [Fact]
    public void NonBatchedModel_AlwaysSelectsPerSequence()
    {
        var caps = new ExecutionCapabilities { SupportsKvStateSnapshot = true };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, Seqs(2));

        Assert.Equal(ExecutionPathKind.PerSequence, plan.Selected);
        Assert.Single(plan.Candidates);
    }

    [Fact]
    public void BatchedPathDisabledOverride_ForcesPerSequence_AndRecordsReason()
    {
        var options = ExecutionOptions.Default with { BatchedPathDisabled = true };
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), options, PlainConfig, Seqs(3));

        Assert.Equal(ExecutionPathKind.PerSequence, plan.Selected);
        var rejection = Assert.Single(plan.Rejections, r => r.Path == ExecutionPathKind.BatchedPaged);
        Assert.Contains("TS_SCHED_DISABLE_BATCHED", rejection.Reason);
    }

    [Fact]
    public void ModelDeclaredBatchedOptOut_SkipsBatchedPaged_WithoutExceptionFallback()
    {
        // A model whose batched path is opted out (e.g. TS_QWEN35_BATCHED=0)
        // declares BatchedForwardAvailable=false; the planner must route
        // straight to per-seq instead of relying on ForwardBatch throwing.
        var caps = BatchedCaps() with { BatchedForwardAvailable = false };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, Seqs(2));

        Assert.Equal(ExecutionPathKind.PerSequence, plan.Selected);
        Assert.Contains(plan.Rejections, r => r.Path == ExecutionPathKind.BatchedPaged);
    }

    [Fact]
    public void EmptyStep_SelectsPerSequence()
    {
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), ExecutionOptions.Default, PlainConfig, Seqs(0));
        Assert.Equal(ExecutionPathKind.PerSequence, plan.Selected);
    }

    // ----- N=1 fused fast path -----

    [Fact]
    public void SoloSequenceInLinearCache_SelectsSingleSequenceFusedFastPath()
    {
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), ExecutionOptions.Default, PlainConfig, Seqs(1));

        Assert.Equal(ExecutionPathKind.SingleSequenceFused, plan.Selected);
        // Terminal: the per-seq executor cannot decline.
        Assert.Single(plan.Candidates);
    }

    [Fact]
    public void SoloSequenceAlreadyInPagedStorage_StaysOnBatchedPaged()
    {
        var features = Seqs(1) with { SoloKvInPagedStorage = true };
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), ExecutionOptions.Default, PlainConfig, features);

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
        var rejection = Assert.Single(plan.Rejections, r => r.Path == ExecutionPathKind.SingleSequenceFused);
        Assert.Contains("paged storage", rejection.Reason);
    }

    [Fact]
    public void SoloWithoutLinearKvMigration_RejectsFastPath()
    {
        // Without linear->paged migration a second concurrent request would
        // corrupt the first sequence's attention, so the fast path must not
        // engage (the original Gemma 4 token-repeat bug).
        var caps = BatchedCaps() with { SupportsLinearKvMigration = false };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, Seqs(1));

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
        Assert.Contains(plan.Rejections, r => r.Path == ExecutionPathKind.SingleSequenceFused);
    }

    [Fact]
    public void SoloRequiringOwnershipSwap_WithoutSnapshot_RejectsFastPath()
    {
        var caps = BatchedCaps() with { SupportsKvStateSnapshot = false };
        var features = Seqs(1) with { SoloRequiresOwnershipSwap = true };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, features);

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
    }

    [Fact]
    public void N1FastPathDisabledOverride_StaysOnBatchedPaged()
    {
        var options = ExecutionOptions.Default with { BatchedN1FastPathEnabled = false };
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), options, PlainConfig, Seqs(1));

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
        var rejection = Assert.Single(plan.Rejections, r => r.Path == ExecutionPathKind.SingleSequenceFused);
        Assert.Contains("TS_BATCHED_N1_FAST_PATH", rejection.Reason);
    }

    // ----- per-sequence fused concurrency -----

    [Fact]
    public void FusedCapableModel_ConcurrentStep_SelectsPerSequenceFused()
    {
        var caps = BatchedCaps() with { SupportsPerSequenceFusedForward = true };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, Seqs(4));

        Assert.Equal(ExecutionPathKind.PerSequenceFused, plan.Selected);
        Assert.Single(plan.Candidates);
    }

    [Fact]
    public void FusedResidentSoloSequence_MustStayFused()
    {
        // A sequence whose tail K/V lives only in its per-request fused holder
        // cannot drop back to the single-stream or batched paths.
        var caps = BatchedCaps() with { SupportsPerSequenceFusedForward = true };
        var features = Seqs(1) with { SoloHasFusedCache = true };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, features);

        Assert.Equal(ExecutionPathKind.PerSequenceFused, plan.Selected);
    }

    [Fact]
    public void FusedCapableSoloOwnershipSwap_UsesPerRequestHolder()
    {
        var caps = BatchedCaps() with
        {
            SupportsPerSequenceFusedForward = true,
            SupportsCrossSequenceKvReuse = false,
        };
        var features = Seqs(1) with { SoloRequiresOwnershipSwap = true };

        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, features);

        Assert.Equal(ExecutionPathKind.PerSequenceFused, plan.Selected);
    }

    [Fact]
    public void FusedCapableModel_NeverFusedSolo_KeepsN1FastPath()
    {
        var caps = BatchedCaps() with { SupportsPerSequenceFusedForward = true };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, Seqs(1));

        Assert.Equal(ExecutionPathKind.SingleSequenceFused, plan.Selected);
    }

    [Fact]
    public void PerSeqFusedDisabledOverride_FallsBackToBatchedPaged()
    {
        var caps = BatchedCaps() with { SupportsPerSequenceFusedForward = true };
        var options = ExecutionOptions.Default with { PerSeqFusedEnabled = false };
        var plan = ExecutionPlanner.PlanStep(caps, options, PlainConfig, Seqs(3));

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
        var rejection = Assert.Single(plan.Rejections, r => r.Path == ExecutionPathKind.PerSequenceFused);
        Assert.Contains("TS_PER_SEQ_FUSED", rejection.Reason);
    }

    // ----- multimodal routing -----

    [Fact]
    public void MixedMultimodalAndTextStep_SelectsSplit()
    {
        var features = Seqs(3) with { MultimodalPendingCount = 1 };
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), ExecutionOptions.Default, PlainConfig, features);

        Assert.Equal(ExecutionPathKind.MixedMultimodalSplit, plan.Selected);
    }

    [Fact]
    public void AllMultimodalStep_WithoutBatchedMultimodal_FallsBackToPerSequence()
    {
        var features = Seqs(2) with { MultimodalPendingCount = 2 };
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), ExecutionOptions.Default, PlainConfig, features);

        Assert.Equal(ExecutionPathKind.PerSequence, plan.Selected);
        Assert.Contains(plan.Rejections, r => r.Path == ExecutionPathKind.BatchedPaged);
    }

    [Fact]
    public void BatchedMultimodalModel_KeepsMultimodalInBatchedPath()
    {
        // Qwen 3.5-style: SupportsBatchedMultimodal means no split; the whole
        // step (multimodal + text) goes through ForwardBatch.
        var caps = BatchedCaps() with { SupportsBatchedMultimodal = true };
        var features = Seqs(3) with { MultimodalPendingCount = 1 };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, PlainConfig, features);

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
    }

    // ----- speculative decoding (NextN/MTP) -----

    private static ExecutionCapabilities SpecLinearCaps() => BatchedCaps() with
    {
        SupportsSpeculativeTrunk = true,
        HasDraftHead = true,
        SpeculationProfitable = true,
        SupportsBatchedSpecTrunk = false,
    };

    private static ExecutionCapabilities SpecBatchedTrunkCaps() => BatchedCaps() with
    {
        SupportsSpeculativeTrunk = true,
        HasDraftHead = true,
        SpeculationProfitable = true,
        SupportsBatchedSpecTrunk = true,
    };

    [Fact]
    public void SpecRequested_LinearTrunkModel_RoutesSoloThroughSpecPerSequence()
    {
        var plan = ExecutionPlanner.PlanStep(SpecLinearCaps(), ExecutionOptions.Default, SpecConfig, Seqs(1));

        Assert.Equal(ExecutionPathKind.SpeculativePerSequence, plan.Selected);
        Assert.Single(plan.Candidates); // terminal: plain per-seq serves if arming fails
    }

    [Fact]
    public void SpecRequested_BatchedTrunkModel_PutsTrunkFirstWithFallbackChain()
    {
        var plan = ExecutionPlanner.PlanStep(SpecBatchedTrunkCaps(), ExecutionOptions.Default, SpecConfig, Seqs(1));

        Assert.Equal(ExecutionPathKind.SpeculativeBatchedTrunk, plan.Selected);
        // Declinable (arming/continuity), so a non-speculative path must follow.
        Assert.True(plan.Candidates.Count >= 2);
        Assert.NotEqual(ExecutionPathKind.SpeculativeBatchedTrunk, plan.Candidates[1]);
    }

    [Fact]
    public void SpecRequested_MultiSequenceStep_ServesNormalPaths()
    {
        var plan = ExecutionPlanner.PlanStep(SpecBatchedTrunkCaps(), ExecutionOptions.Default, SpecConfig, Seqs(2));

        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
        Assert.Contains(plan.Rejections, r => r.Path == ExecutionPathKind.SpeculativeBatchedTrunk);
    }

    [Fact]
    public void SpecRequested_SoloKvInPagedStorage_LinearTrunkDeclines()
    {
        var features = Seqs(1) with { SoloKvInPagedStorage = true };
        var plan = ExecutionPlanner.PlanStep(SpecLinearCaps(), ExecutionOptions.Default, SpecConfig, features);

        // Forward would attend against an empty linear cache; must stay batched.
        Assert.Equal(ExecutionPathKind.BatchedPaged, plan.Selected);
        Assert.Contains(plan.Rejections, r => r.Path == ExecutionPathKind.SpeculativePerSequence);
    }

    [Fact]
    public void SpecRequested_ButUnprofitableOnBackend_FlagsNoticeAndServesStandardDecode()
    {
        var caps = SpecLinearCaps() with { SpeculationProfitable = false };
        var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, SpecConfig, Seqs(1));

        Assert.True(plan.SpeculationUnprofitable);
        Assert.Equal(ExecutionPathKind.SingleSequenceFused, plan.Selected);
    }

    [Fact]
    public void SpecRequested_NoDraftHead_ServesStandardDecodeWithoutNotice()
    {
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), ExecutionOptions.Default, SpecConfig, Seqs(1));

        Assert.False(plan.SpeculationUnprofitable);
        Assert.Equal(ExecutionPathKind.SingleSequenceFused, plan.Selected);
    }

    [Fact]
    public void SpecBatchedTrunk_EngagesEvenWhenBatchedPathDisabled()
    {
        // Historical behaviour preserved: the MTP routes predate
        // TS_SCHED_DISABLE_BATCHED and must keep engaging under it; on
        // decline the step falls to per-seq (batched is disabled).
        var options = ExecutionOptions.Default with { BatchedPathDisabled = true };
        var plan = ExecutionPlanner.PlanStep(SpecBatchedTrunkCaps(), options, SpecConfig, Seqs(1));

        Assert.Equal(ExecutionPathKind.SpeculativeBatchedTrunk, plan.Selected);
        Assert.Equal(ExecutionPathKind.PerSequence, plan.Candidates[^1]);
    }

    // ----- plan invariants and reporting -----

    [Fact]
    public void EveryPlan_TerminatesWithANonDecliningCandidate()
    {
        // Exhaustively sweep the boolean feature/capability space (bounded:
        // this is exactly the combination explosion the planner exists to
        // make enumerable) and assert the chain always ends in a terminal
        // path, so no step can ever go unserved.
        foreach (var batchedImpl in new[] { false, true })
        foreach (var available in new[] { false, true })
        foreach (var fusedCap in new[] { false, true })
        foreach (var migration in new[] { false, true })
        foreach (var snapshot in new[] { false, true })
        foreach (var mtp in new[] { false, true })
        foreach (var trunk in new[] { false, true })
        foreach (var disabled in new[] { false, true })
        foreach (var count in new[] { 0, 1, 2 })
        foreach (var mmCount in new[] { 0, 1 })
        {
            if (mmCount > count) continue;
            var caps = new ExecutionCapabilities
            {
                SupportsBatchedPagedAttention = batchedImpl,
                BatchedForwardAvailable = batchedImpl && available,
                SupportsPerSequenceFusedForward = batchedImpl && fusedCap,
                SupportsLinearKvMigration = batchedImpl && migration,
                SupportsKvStateSnapshot = snapshot,
                SupportsCrossSequenceKvReuse = snapshot,
                SupportsSpeculativeTrunk = mtp,
                HasDraftHead = mtp,
                SpeculationProfitable = mtp,
                SupportsBatchedSpecTrunk = mtp && trunk,
            };
            var options = ExecutionOptions.Default with { BatchedPathDisabled = disabled };
            var features = new ExecutionStepFeatures
            {
                SequenceCount = count,
                MultimodalPendingCount = mmCount,
                SoloHasPendingMultimodal = count == 1 && mmCount == 1,
            };
            var plan = ExecutionPlanner.PlanStep(caps, options, SpecConfig, features);

            Assert.NotEmpty(plan.Candidates);
            var last = plan.Candidates[^1];
            Assert.True(
                last != ExecutionPathKind.SpeculativeBatchedTrunk && last != ExecutionPathKind.BatchedPaged,
                $"plan may end with declinable path {last}: {plan.Describe()}");
            // No duplicate candidates (each path tried at most once).
            Assert.Equal(plan.Candidates.Count, plan.Candidates.Distinct().Count());
        }
    }

    [Fact]
    public void PlanDescribe_ListsSelectedFallbacksAndRejections()
    {
        var options = ExecutionOptions.Default with { BatchedN1FastPathEnabled = false };
        var plan = ExecutionPlanner.PlanStep(BatchedCaps(), options, PlainConfig, Seqs(1));

        string desc = plan.Describe();
        Assert.Contains("BatchedPaged", desc);
        Assert.Contains("PerSequence", desc);
        Assert.Contains("TS_BATCHED_N1_FAST_PATH", desc);
    }

    [Fact]
    public void CapabilityReport_ExplainsUnavailablePaths()
    {
        var caps = new ExecutionCapabilities { SupportsKvStateSnapshot = true };
        string report = ExecutionPlanner.BuildCapabilityReport(caps, ExecutionOptions.Default, PlainConfig);

        Assert.Contains("does not implement IBatchedPagedModel", report);
        Assert.Contains("off (not requested)", report);
    }

    [Fact]
    public void ExecutionOptions_FromEnvironment_ParsesLegacyFlagSemantics()
    {
        string prevDisable = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
        string prevFused = Environment.GetEnvironmentVariable("TS_BATCHED_FUSED_DECODE");
        try
        {
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
            // Strict opt-in flag: an arbitrary non-"1"/"true" value stays off
            // (historical parse rule preserved).
            Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", "yes");

            var options = ExecutionOptions.FromEnvironment();
            Assert.True(options.BatchedPathDisabled);
            Assert.False(options.BatchedFusedDecodeEnabled);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", prevDisable);
            Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", prevFused);
        }
    }
}
