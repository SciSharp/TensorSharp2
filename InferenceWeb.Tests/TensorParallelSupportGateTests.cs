// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// How `--tp N` is resolved per architecture.
//
// Two regressions live here. First: `--tp 2` on an architecture with no
// tensor-parallel implementation used to be accepted in full silence - a real
// multi-GPU context and NCCL group were built, the banner announced "Tensor
// parallelism: 2 GPUs", and then every weight was uploaded through rank 0 and
// the model ran on GPU 0, because sharding is opt-in per model class and
// qwen4exp never opted in. Second: refusing outright then threw the second GPU
// away for an architecture that CAN use it - just not by sharding. qwen4exp now
// resolves --tp N to a LAYER SPLIT (each GPU holds a contiguous run of whole
// layers), which is the same and only multi-GPU mode llama.cpp offers for it.
//
// The mode now lives on each architecture's own descriptor rather than in two
// name tables inside ModelBase, so the last two facts the tables used to be
// checked for - "every entry explains itself" and "every layer-split arch is
// also declared non-tensor-parallel" - are structural invariants of
// ModelArchitectureDescriptor.Validate() instead, asserted here over the whole
// registered set.
using System;
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
    public void LayerSplitArchitecture_ResolvesToASplit_NotTensorParallelism()
    {
        ITensorParallelGroup group = null;
        int tp = Resolve("qwen4exp", BackendType.GgmlCuda, 2, ref group, out int layerSplit);

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
        // ggml_cpu exposes one device; there is nothing to split across, so this
        // must fall back to the loud single-GPU degrade rather than claim a split.
        ITensorParallelGroup group = null;
        int tp = Resolve("qwen4exp", BackendType.GgmlCpu, 2, ref group, out int layerSplit);
        Assert.Equal(1, tp);
        Assert.Equal(1, layerSplit);
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
    [InlineData("deepseek4")]   // multi-GPU through its own executor, not the TP group
    public void TpCapableArchitectures_AreUntouched(string arch)
    {
        ITensorParallelGroup group = null;
        Assert.Equal(4, Resolve(arch, BackendType.GgmlCuda, 4, ref group, out int layerSplit));
        Assert.Equal(1, layerSplit);
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
