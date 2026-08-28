// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The two pieces of arithmetic DFlash2 adds on top of DFlash, checked against
/// independent references written from the algorithm rather than from the
/// implementation:
///
///   * the grouped dynamic depthwise convolution that wraps every attention and
///     every FFN sublayer, and
///   * the greedy walk through the candidate selector's transition lattice.
///
/// These run without weights, a GPU, or a native library, so they hold on every
/// machine - unlike the end-to-end parity runs, which need the checkpoints.
/// </summary>
public class DFlash2MathTests
{
    /// <summary>
    /// out[r][c] = sum_t (base[side][t][c] + delta[r][t][c / groupSize]) * x[r-t][c],
    /// tap t masked off for r &lt; t. Written the slow, obvious way.
    /// </summary>
    private static float[] ReferenceConvolve(
        float[] x, float[] baseKernel, float[] delta,
        int rows, int hidden, int taps, int groupSize, int deltaStride, int deltaOffset, int side)
    {
        int groups = hidden / groupSize;
        var outp = new float[rows * hidden];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < hidden; c++)
            {
                float acc = 0f;
                for (int t = 0; t < taps; t++)
                {
                    if (r - t < 0)
                        continue;                     // the block-boundary mask
                    float b = baseKernel[(side * taps + t) * hidden + c];
                    float d = delta[r * deltaStride + deltaOffset + t * groups + (c / groupSize)];
                    acc += (b + d) * x[(r - t) * hidden + c];
                }
                outp[r * hidden + c] = acc;
            }
        }
        return outp;
    }

    [Theory]
    [InlineData(1, 8, 2, 2)]      // one row: only tap 0 survives the mask
    [InlineData(8, 16, 2, 4)]     // the shipped shape (2 taps)
    [InlineData(5, 12, 3, 3)]     // a wider kernel than anything shipped
    [InlineData(16, 64, 2, 16)]   // the shipped group size
    public void GroupedConvolve_MatchesReference(int rows, int hidden, int taps, int groupSize)
    {
        int groups = hidden / groupSize;
        int deltaStride = 2 * taps * groups;   // both sides, as the projection emits them
        var rng = new Random(1234 + rows * 31 + hidden);

        float[] Rand(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++)
                a[i] = (float)(rng.NextDouble() * 2 - 1);
            return a;
        }

        float[] x = Rand(rows * hidden);
        float[] baseKernel = Rand(2 * taps * hidden);
        float[] delta = Rand(rows * deltaStride);

        for (int side = 0; side < 2; side++)
        {
            int deltaOffset = side * taps * groups;
            float[] expected = ReferenceConvolve(x, baseKernel, delta, rows, hidden, taps, groupSize, deltaStride, deltaOffset, side);
            float[] actual = ModelBase.DFlashGroupedConvolve(x, baseKernel, delta, rows, hidden, taps, groupSize, deltaStride, deltaOffset, side);

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f, $"side {side} element {i}: {expected[i]} != {actual[i]}");
        }
    }

    /// <summary>
    /// A checkpoint's conv_base starts (and stays close to) the identity - tap 0 all
    /// ones, later taps zero - so an all-zero delta must leave the rows untouched.
    /// That is the property that makes the convolution safe to add to a trained
    /// backbone, and the one a layout mistake would break first.
    /// </summary>
    [Fact]
    public void GroupedConvolve_IdentityKernelIsAPassThrough()
    {
        const int rows = 8, hidden = 32, taps = 2, groupSize = 8;
        int groups = hidden / groupSize;
        int deltaStride = 2 * taps * groups;

        var x = new float[rows * hidden];
        for (int i = 0; i < x.Length; i++)
            x[i] = i + 1;

        var baseKernel = new float[2 * taps * hidden];
        for (int side = 0; side < 2; side++)
            for (int c = 0; c < hidden; c++)
                baseKernel[(side * taps + 0) * hidden + c] = 1f;   // tap 0 = identity

        var delta = new float[rows * deltaStride];                  // no dynamic part

        float[] actual = ModelBase.DFlashGroupedConvolve(x, baseKernel, delta, rows, hidden, taps, groupSize, deltaStride, 0, side: 0);
        Assert.Equal(x, actual);
    }

    /// <summary>Tap 1 must never reach across the block boundary: row 0 sees only
    /// itself no matter what the tap-1 coefficient is.</summary>
    [Fact]
    public void GroupedConvolve_FirstRowIgnoresTheShiftedTap()
    {
        const int rows = 3, hidden = 4, taps = 2, groupSize = 2;
        int groups = hidden / groupSize;
        int deltaStride = 2 * taps * groups;

        var x = new float[] { 1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400 };
        var baseKernel = new float[2 * taps * hidden];
        for (int c = 0; c < hidden; c++)
        {
            baseKernel[(0 * taps + 0) * hidden + c] = 1f;   // tap 0 identity
            baseKernel[(0 * taps + 1) * hidden + c] = 5f;   // tap 1 deliberately huge
        }
        var delta = new float[rows * deltaStride];

        float[] actual = ModelBase.DFlashGroupedConvolve(x, baseKernel, delta, rows, hidden, taps, groupSize, deltaStride, 0, side: 0);

        // Row 0: x[0] only. Row r>0: x[r] + 5*x[r-1].
        Assert.Equal(new float[] { 1, 2, 3, 4 }, actual[0..4]);
        Assert.Equal(new float[] { 10 + 5 * 1, 20 + 5 * 2, 30 + 5 * 3, 40 + 5 * 4 }, actual[4..8]);
        Assert.Equal(new float[] { 100 + 5 * 10, 200 + 5 * 20, 300 + 5 * 30, 400 + 5 * 40 }, actual[8..12]);
    }

    [Fact]
    public void TopK_SelectsTheLargestValuesRegardlessOfOrder()
    {
        var rng = new Random(7);
        var row = new float[4096];
        for (int i = 0; i < row.Length; i++)
            row[i] = (float)(rng.NextDouble() * 100 - 50);

        const int k = 16;
        var ids = new int[k];
        var vals = new float[k];
        ModelBase.DFlashTopK(row, k, ids, vals, 0);

        // Every reported (id, value) pair must be consistent...
        for (int i = 0; i < k; i++)
            Assert.Equal(row[ids[i]], vals[i]);
        // ...the ids distinct...
        Assert.Equal(k, new HashSet<int>(ids).Count);
        // ...and the set must be exactly the k largest (the top-k is unsorted, so
        // compare as sets against a sorted reference).
        var sorted = (float[])row.Clone();
        Array.Sort(sorted);
        float cutoff = sorted[^k];
        foreach (float v in vals)
            Assert.True(v >= cutoff, $"{v} < cutoff {cutoff}");
    }

    /// <summary>
    /// The lattice walk: position 0 takes the argmax of the anchor row, and every
    /// later position takes the argmax of the row selected by the PREVIOUS
    /// position's choice. Getting that indirection wrong is invisible in the output
    /// (the trunk still corrects every token) and shows up only as lost acceptance,
    /// so it is worth pinning exactly.
    /// </summary>
    [Fact]
    public void WalkLattice_FollowsThePreviousChoicesRow()
    {
        const int gamma = 3, k = 4;
        // scores layout: [k] anchor row, then one [k(pred), k(cand)] matrix per
        // following position, candidate-fastest.
        var scores = new float[k + k * k * (gamma - 1)];

        // Anchor row: candidate 2 wins.
        scores[0] = 0.1f; scores[1] = 0.2f; scores[2] = 9.0f; scores[3] = 0.3f;

        // Position 1, predecessor rows. Only row 2 should ever be read; make the
        // others point somewhere else so a wrong index is caught.
        float[,] p1 = { { 5, 0, 0, 0 }, { 0, 5, 0, 0 }, { 0, 0, 0, 7 }, { 0, 0, 5, 0 } };
        for (int p = 0; p < k; p++)
            for (int c = 0; c < k; c++)
                scores[k + (p * k) + c] = p1[p, c];

        // Position 2: only predecessor row 3 matters (position 1 chose candidate 3).
        float[,] p2 = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 8, 0, 0 } };
        for (int p = 0; p < k; p++)
            for (int c = 0; c < k; c++)
                scores[k + k * k + (p * k) + c] = p2[p, c];

        // Distinct token ids so the gather is checked too.
        var cand = new int[gamma * k];
        for (int e = 0; e < gamma; e++)
            for (int c = 0; c < k; c++)
                cand[e * k + c] = 1000 * (e + 1) + c;

        var draft = new int[gamma];
        var conf = new float[gamma];
        int n = ModelBase.DFlashWalkLattice(gamma, k, scores, cand, draft, conf);

        Assert.Equal(gamma, n);
        Assert.Equal(new[] { 1002, 2003, 3001 }, draft);
        // Each confidence is the softmax of the row it was chosen from.
        Assert.Equal(ModelBase.DFlashSoftmaxAt(new[] { 0.1f, 0.2f, 9.0f, 0.3f }, 2), conf[0], 5);
        Assert.Equal(ModelBase.DFlashSoftmaxAt(new[] { 0f, 0f, 0f, 7f }, 3), conf[1], 5);
        Assert.Equal(ModelBase.DFlashSoftmaxAt(new[] { 0f, 8f, 0f, 0f }, 1), conf[2], 5);
    }

    /// <summary>A one-position block has no transitions at all: the walk must read
    /// only the anchor row and never index past it.</summary>
    [Fact]
    public void WalkLattice_SinglePositionUsesOnlyTheAnchorRow()
    {
        const int gamma = 1, k = 3;
        var scores = new float[] { 1f, 4f, 2f };
        var cand = new[] { 11, 22, 33 };
        var draft = new int[gamma];
        var conf = new float[gamma];

        Assert.Equal(1, ModelBase.DFlashWalkLattice(gamma, k, scores, cand, draft, conf));
        Assert.Equal(22, draft[0]);
        Assert.Equal(ModelBase.DFlashSoftmaxAt(new[] { 1f, 4f, 2f }, 1), conf[0], 5);
    }
}
