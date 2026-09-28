// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Explicit tensor-parallel and layer-placement requests are checked against
// each architecture's descriptor. Unsupported requests must fail, never turn
// into a different execution mode or silently leave extra GPUs idle.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Models.Architecture;
using Xunit;

namespace InferenceWeb.Tests;

public class TensorParallelSupportGateTests
{
    private static ModelArchitectureDescriptor Arch(string id)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(id, out var descriptor),
            $"architecture '{id}' is not registered");
        return descriptor;
    }

    private static int Resolve(ModelArchitectureDescriptor arch, BackendType backend, int tpDegree,
        ref ITensorParallelGroup group, out int layerSplit)
        => TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            arch, backend, tpDegree, ref group, out layerSplit);

    private static int Resolve(string arch, BackendType backend, int tpDegree,
        ref ITensorParallelGroup group, out int layerSplit)
        => Resolve(Arch(arch), backend, tpDegree, ref group, out layerSplit);

    [Fact]
    public void LayerSplitArchitecture_RequiresExplicitLayerSplit()
    {
        ITensorParallelGroup group = null;
        var error = Assert.Throws<NotSupportedException>(() =>
            Resolve("qwen4exp", BackendType.GgmlCuda, 2, ref group, out _));
        Assert.Contains("--layer-split", error.Message);
        int tp = TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            Arch("qwen4exp"), BackendType.GgmlCuda, 1, ref group, out int layerSplit, 2);

        // No tensor-parallel group: IsTensorParallel gates weight sharding and the
        // AllReduce machinery, none of which a layer split uses.
        Assert.Equal(1, tp);
        Assert.Null(group);
        // ...but both GPUs are used, by layers.
        Assert.Equal(2, layerSplit);
    }

    [Fact]
    public void LayerSplit_OnlyOnBackendsThatHaveSeveralDevices()
    {
        ITensorParallelGroup group = null;
        Assert.Throws<NotSupportedException>(() => TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            Arch("qwen4exp"), BackendType.GgmlCpu, 1, ref group, out _, 2));
    }

    [Fact]
    public void DistributedTpOnUnsupportedArchitecture_Throws()
    {
        // A distributed group cannot be downgraded on one node: the other nodes
        // would still be waiting on collectives this rank will never issue. A
        // layer split is single-process, so it is not an answer here either.
        ITensorParallelGroup group = new StubTpGroup();
        var ex = Assert.Throws<NotSupportedException>(
            () => Resolve("qwen4exp", BackendType.GgmlCuda, 2, ref group, out _));
        Assert.Contains("qwen4exp", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("qwen35")]
    [InlineData("gemma4")]
    [InlineData("muse-glimmer")]
    [InlineData("glm-dsa")]
    [InlineData("glm5next")]
    public void TpCapableArchitectures_AreUntouched(string arch)
    {
        ITensorParallelGroup group = null;
        Assert.Equal(4, Resolve(arch, BackendType.GgmlCuda, 4, ref group, out int layerSplit));
        Assert.Equal(1, layerSplit);
    }

    [Fact]
    public void GlmNativeTensorParallelism_RejectsDistributedGroupBeforeModelConstruction()
    {
        var context = new ModelCreateContext(
            "/model-is-not-opened.gguf", BackendType.GgmlCuda, probe: null,
            tpDegree: 2, tpGroup: new StubTpGroup());

        var error = Assert.Throws<NotSupportedException>(() => Arch("glm5next").Factory(context));
        Assert.Contains("local/single-process", error.Message, StringComparison.Ordinal);
        Assert.Contains("--tp-node-id/--tp-peers", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeepSeek41NativeExecutor_RejectsDistributedGroupBeforeSidecarOrWeights()
    {
        var context = new ModelCreateContext(
            "/model-is-not-opened.gguf", BackendType.GgmlCuda, probe: null,
            tpDegree: 2, tpGroup: new StubTpGroup());

        var error = Assert.Throws<NotSupportedException>(() => Arch("deepseek41").ApplyNativeTunables(context));
        Assert.Contains("single-process", error.Message, StringComparison.Ordinal);
        Assert.Contains("--tp-node-id/--tp-peers", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NoTpRequested_IsAlwaysAPassthrough()
    {
        // The gate must not fire on ordinary single-GPU runs of the very
        // architectures it knows about.
        ITensorParallelGroup group = null;
        Assert.Equal(1, Resolve("qwen4exp", BackendType.GgmlCuda, 1, ref group, out int layerSplit));
        Assert.Equal(1, layerSplit);
        Assert.Null(group);
    }

    [Fact]
    public void ArchitectureThatDeclaresNothing_IsNotBlocked()
    {
        // The gate is opt-in per architecture, not an allow-list: a family that says
        // nothing about multi-GPU gets plain tensor parallelism, unchanged.
        var brandNew = new ModelArchitectureDescriptor
        {
            Id = "brand-new-arch",
            Aliases = new[] { "brand-new-arch" },
            Factory = _ => throw new NotSupportedException("not constructed by this test"),
        };
        ITensorParallelGroup group = null;
        Assert.Equal(2, Resolve(brandNew, BackendType.GgmlCuda, 2, ref group, out int layerSplit));
        Assert.Equal(1, layerSplit);
    }

    [Fact]
    public void EveryDegradedArchitectureExplainsItself()
    {
        // The message is the whole value of the gate - it is what tells the operator
        // why the second GPU is idle, or why it holds whole layers instead of shards.
        foreach (var arch in ModelArchitectureRegistry.All.Where(a => a.MultiGpu != MultiGpuMode.TensorParallel))
        {
            Assert.False(string.IsNullOrWhiteSpace(arch.MultiGpuLimitation), $"'{arch.Id}' has no explanation.");
            Assert.Contains(arch.Id, arch.MultiGpuLimitation, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void DeclaringADegradedModeWithoutAReasonIsRejectedAtRegistration()
    {
        // Validate() is what makes "every entry explains itself" true by construction,
        // so a family cannot land a silent degrade the way the old name tables allowed.
        var silent = new ModelArchitectureDescriptor
        {
            Id = "silent-arch",
            Aliases = new[] { "silent-arch" },
            Factory = _ => throw new NotSupportedException(),
            MultiGpu = MultiGpuMode.LayerSplit,
        };
        var ex = Assert.Throws<InvalidOperationException>(() => ModelArchitectureRegistry.Register(silent));
        Assert.Contains("MultiGpuLimitation", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryBuiltInArchitectureIsWellFormed()
    {
        var all = ModelArchitectureRegistry.All;
        Assert.NotEmpty(all);
        foreach (var arch in all)
        {
            Assert.NotNull(arch.Factory);
            Assert.Contains(arch.Id, arch.Aliases, StringComparer.OrdinalIgnoreCase);
            foreach (string alias in arch.Aliases)
                Assert.Same(arch, Arch(alias));
        }

        // Aliases are the routing key; two families claiming one would silently shadow.
        var aliases = all.SelectMany(a => a.Aliases).Select(a => a.ToLowerInvariant()).ToList();
        Assert.Equal(aliases.Count, aliases.Distinct().Count());
    }

    /// <summary>Minimal live group: the gate only reads whether one exists.</summary>
    private sealed class StubTpGroup : ITensorParallelGroup
    {
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
