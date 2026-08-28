// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The layer -> GPU assignment behind qwen4exp's multi-GPU layer split.
//
// The properties asserted here are not cosmetic. Each device boundary inside a
// token is a span cut and a residual hand-off through host memory, so:
//   * runs must be CONTIGUOUS - an interleaved map costs a seam per interleave;
//   * device indices must be MONOTONIC - the residual only travels forwards;
//   * no device may be left empty - that is a seam bought for nothing.
// And the map must be stable for the model's lifetime: it decides which GPU each
// weight is uploaded to, and the preload frees the host copy right afterwards.
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class Qwen4ExpLayerSplitTests
{
    private static long[] Uniform(int n, long bytes) => Enumerable.Repeat(bytes, n).ToArray();

    [Fact]
    public void SingleDevice_PutsEverythingOnGpu0()
    {
        int[] map = Qwen4ExpModel.PackLayersOntoDevices(Uniform(48, 1000), 0, 0, 1);
        Assert.All(map, d => Assert.Equal(0, d));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void RunsAreContiguousMonotonicAndCoverEveryDevice(int devices)
    {
        int[] map = Qwen4ExpModel.PackLayersOntoDevices(Uniform(48, 1000), 0, 0, devices);

        Assert.Equal(48, map.Length);
        Assert.Equal(0, map[0]);
        for (int i = 1; i < map.Length; i++)
        {
            // Monotonic, and never skipping a device: both are what makes the
            // number of seams exactly devices-1.
            Assert.True(map[i] == map[i - 1] || map[i] == map[i - 1] + 1,
                $"layer {i} jumped from device {map[i - 1]} to {map[i]}");
        }
        Assert.Equal(devices - 1, map[^1]);
        Assert.Equal(devices, map.Distinct().Count());
    }

    [Fact]
    public void SharedBytesShiftLayersOffDevice0()
    {
        // Device 0 also carries the token embedding (the head group is charged to
        // the LAST device instead - see below; the vision tower is not modelled at
        // all, see BuildLayerDeviceMap). Charging those bytes is the whole point: an
        // equal LAYER count would make gpu0 the one that runs out of VRAM.
        int[] even = Qwen4ExpModel.PackLayersOntoDevices(Uniform(48, 1000), 0, 0, 2);
        int[] loaded = Qwen4ExpModel.PackLayersOntoDevices(Uniform(48, 1000), 12_000, 0, 2);

        int evenOnGpu0 = even.Count(d => d == 0);
        int loadedOnGpu0 = loaded.Count(d => d == 0);
        Assert.True(loadedOnGpu0 < evenOnGpu0,
            $"shared bytes should move layers off gpu0, but it kept {loadedOnGpu0} vs {evenOnGpu0}");
        Assert.True(loadedOnGpu0 >= 1, "gpu0 must still run at least one layer - it holds the embedding.");
    }

    [Fact]
    public void UnevenLayerCostsStillBalanceBytes()
    {
        // Real models are not uniform: qwen4exp's dense layers are far smaller than
        // its 512-expert MoE layers. Balance is on BYTES, not layer count.
        var bytes = new long[8];
        for (int i = 0; i < 8; i++) bytes[i] = i < 4 ? 100 : 900;

        int[] map = Qwen4ExpModel.PackLayersOntoDevices(bytes, 0, 0, 2);

        long gpu0 = 0, gpu1 = 0;
        for (int i = 0; i < 8; i++) { if (map[i] == 0) gpu0 += bytes[i]; else gpu1 += bytes[i]; }
        long total = gpu0 + gpu1;
        // Each side within 40% of half: a layer is indivisible, so exact halves are
        // not reachable, but a count-based split (4/4 = 400/3600) would be far worse.
        Assert.True(gpu0 > total * 0.1 && gpu1 > total * 0.1,
            $"bytes are badly balanced: gpu0={gpu0} gpu1={gpu1}");
    }

    [Fact]
    public void MoreDevicesThanLayers_LeavesNoDeviceWithoutWork()
    {
        // Degenerate but reachable (--tp 4 on a tiny test model). Every device that
        // appears in the map must own at least one layer; the packing must not
        // advance past a device it never used.
        int[] map = Qwen4ExpModel.PackLayersOntoDevices(Uniform(3, 1000), 0, 0, 4);
        Assert.Equal(3, map.Length);
        foreach (int d in map.Distinct())
            Assert.Contains(d, map);
        for (int i = 1; i < map.Length; i++)
            Assert.True(map[i] == map[i - 1] || map[i] == map[i - 1] + 1);
    }


    [Fact]
    public void HeadBytesShiftLayersOffTheLastDevice()
    {
        // The final mixer + LM head ride the LAST span, so they are charged to the
        // LAST device. Charging them to device 0 (as the first cut of this code did)
        // made the last GPU the one that ran out of VRAM.
        int[] even = Qwen4ExpModel.PackLayersOntoDevices(Uniform(48, 1000), 0, 0, 2);
        int[] headed = Qwen4ExpModel.PackLayersOntoDevices(Uniform(48, 1000), 0, 12_000, 2);

        int evenOnLast = even.Count(d => d == 1);
        int headedOnLast = headed.Count(d => d == 1);
        Assert.True(headedOnLast < evenOnLast,
            $"head bytes should move layers off the last gpu, but it kept {headedOnLast} vs {evenOnLast}");
        Assert.True(headedOnLast >= 1, "the last gpu must still run at least one layer - it holds the head.");
    }

    [Fact]
    public void SharedAndHeadBytesPullFromOppositeEnds()
    {
        // Both ends loaded: device 0 carries the embedding, the last carries the head,
        // so the middle devices should take more layers than either end.
        int[] map = Qwen4ExpModel.PackLayersOntoDevices(Uniform(60, 1000), 15_000, 15_000, 3);
        int first = map.Count(d => d == 0), mid = map.Count(d => d == 1), last = map.Count(d => d == 2);
        Assert.Equal(60, first + mid + last);
        Assert.True(mid > first && mid > last,
            $"middle device should take the most layers: {first}/{mid}/{last}");
    }


    // ---- TS_Q4E_LAYER_SPLIT override ----------------------------------------

    [Fact]
    public void Override_Unset_UsesTheAutomaticBalance()
    {
        Assert.Null(Qwen4ExpModel.ParseLayerSplitOverride(null, 48, 2));
        Assert.Null(Qwen4ExpModel.ParseLayerSplitOverride("", 48, 2));
        Assert.Null(Qwen4ExpModel.ParseLayerSplitOverride("   ", 48, 2));
    }

    [Fact]
    public void Override_IgnoredWithoutASplit()
    {
        Assert.Null(Qwen4ExpModel.ParseLayerSplitOverride("48", 48, 1));
    }

    [Fact]
    public void Override_AssignsExactlyTheRequestedCounts()
    {
        int[] map = Qwen4ExpModel.ParseLayerSplitOverride("20,28", 48, 2);
        Assert.NotNull(map);
        Assert.Equal(20, map.Count(d => d == 0));
        Assert.Equal(28, map.Count(d => d == 1));
        // Still contiguous and monotonic - the seam count must not change.
        for (int i = 1; i < map.Length; i++)
            Assert.True(map[i] == map[i - 1] || map[i] == map[i - 1] + 1);
    }

    [Theory]
    [InlineData("20,29", 48, 2)]      // counts do not sum to the layer count
    [InlineData("20", 48, 2)]         // too few devices
    [InlineData("10,20,18", 48, 2)]   // too many devices
    [InlineData("0,48", 48, 2)]       // a GPU with no layers is a seam for nothing
    [InlineData("-1,49", 48, 2)]      // negative
    [InlineData("twenty,28", 48, 2)]  // unparseable
    public void Override_RejectsAnythingItCannotHonour(string spec, int layers, int devices)
    {
        // Throwing matters: silently falling back to the automatic balance would let
        // an operator believe they had placed the layers when they had not.
        Assert.Throws<System.ArgumentException>(
            () => Qwen4ExpModel.ParseLayerSplitOverride(spec, layers, devices));
    }

    [Fact]
    public void EmptyModel_DoesNotThrow()
    {
        Assert.Empty(Qwen4ExpModel.PackLayersOntoDevices(System.Array.Empty<long>(), 0, 0, 2));
    }
}
