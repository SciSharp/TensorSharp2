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
using System;
using TensorSharp;
using Xunit;

namespace InferenceWeb.Tests;

public class TensorParallelSupportGateTests
{
    private static int Resolve(string arch, BackendType backend, int tpDegree,
        ref ITensorParallelGroup group, out int layerSplit)
        => TensorSharp.Models.ModelBase.ResolveTensorParallelSupport(
            arch, backend, tpDegree, ref group, out layerSplit);

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
    public void UnknownArchitecture_IsNotBlocked()
    {
        // The table is a deny-list of known-unsupported architectures, not an
        // allow-list: a new arch must not be refused just for being absent.
        ITensorParallelGroup group = null;
        Assert.Equal(2, Resolve("brand-new-arch", BackendType.GgmlCuda, 2, ref group, out _));
    }

    [Fact]
    public void EveryEntryExplainsItself()
    {
        // The message is the whole value of the gate - it is what tells the
        // operator why the second GPU is idle. An empty one is a bug.
        Assert.NotEmpty(TensorSharp.Models.ModelBase.ArchitecturesWithoutTensorParallel);
        foreach (var kv in TensorSharp.Models.ModelBase.ArchitecturesWithoutTensorParallel)
        {
            Assert.False(string.IsNullOrWhiteSpace(kv.Value), $"'{kv.Key}' has no explanation.");
            Assert.Contains(kv.Key, kv.Value, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void EveryLayerSplitArchitectureIsAlsoDeclaredNonTensorParallel()
    {
        // The split list is consulted only after the no-TP list matches. An arch in
        // one and not the other would silently never split.
        foreach (string arch in TensorSharp.Models.ModelBase.ArchitecturesWithLayerSplit)
        {
            Assert.True(TensorSharp.Models.ModelBase.ArchitecturesWithoutTensorParallel.ContainsKey(arch),
                $"'{arch}' can layer-split but is not listed as lacking tensor parallelism, so the "
                + "split branch is unreachable for it.");
        }
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
