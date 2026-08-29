// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.GGML;

namespace TensorSharp.Models.Paged
{
    /// <summary>
    /// The batched paged path's K/V pool, held on the DEVICE.
    ///
    /// The managed <c>float[][]</c> pool it replaces is what capped continuous
    /// batching: every layer of every step gathered the whole sequence history out
    /// of host memory and pushed it across the bus, so the batched path cost
    /// O(history x layers) per step. Measured on gemma-4-E4B / 1x Blackwell that
    /// made it 7.7x slower per token than per-sequence fused decode and pinned
    /// aggregate throughput at ~69 tok/s however many sequences were in flight -
    /// i.e. concurrency bought nothing.
    ///
    /// Here the pool is backend tensors that never move. Writing a step's K/V is
    /// <c>ggml_set_rows</c> over just this step's tokens; reading a sequence's
    /// history is <c>ggml_get_rows</c> inside the flash-attention graph. Only the
    /// index vectors are host data.
    ///
    /// Falls back cleanly: <see cref="TryCreate"/> returns null when the backend
    /// cannot allocate (no GGML backend, or out of VRAM) and callers keep the host
    /// pool, so this is an optimisation and never a correctness dependency.
    /// </summary>
    internal sealed class DevicePagedKvCache : IDisposable
    {
        private IntPtr _handle;

        private DevicePagedKvCache(IntPtr handle, int numLayers, int numBlocks,
            int blockSize, int numKvHeads, int headDim)
        {
            _handle = handle;
            NumLayers = numLayers;
            NumBlocks = numBlocks;
            BlockSize = blockSize;
            NumKvHeads = numKvHeads;
            HeadDim = headDim;
        }

        public int NumLayers { get; }
        public int NumBlocks { get; private set; }
        public int BlockSize { get; }
        public int NumKvHeads { get; }
        public int HeadDim { get; }

        public bool IsAlive => _handle != IntPtr.Zero;

        /// <summary>Device memory the pool occupies, for VRAM reporting.</summary>
        public long DeviceBytes => GgmlBasicOps.PagedKvPoolBytes(_handle);

        /// <summary>Allocate the pool, or return null when it cannot be had.</summary>
        public static DevicePagedKvCache TryCreate(
            int numLayers, int numBlocks, int blockSize, int numKvHeads, int headDim)
        {
            if (numLayers <= 0 || numBlocks <= 0 || blockSize <= 0 || numKvHeads <= 0 || headDim <= 0)
                return null;
            try
            {
                IntPtr handle = GgmlBasicOps.PagedKvPoolCreate(
                    numLayers, numBlocks, blockSize, numKvHeads, headDim);
                if (handle == IntPtr.Zero)
                    return null;
                return new DevicePagedKvCache(handle, numLayers, numBlocks, blockSize, numKvHeads, headDim);
            }
            catch (DllNotFoundException) { return null; }
            catch (EntryPointNotFoundException) { return null; }
            catch (Exception) { return null; }
        }

        /// <summary>
        /// Make room for <paramref name="requiredBlocks"/>. False means the pool
        /// kept its old size (typically out of VRAM); the K/V already written is
        /// still valid, but the caller must run this step on the host pool rather
        /// than write past the end.
        /// </summary>
        public bool EnsureCapacity(int requiredBlocks)
        {
            if (!IsAlive) return false;
            if (requiredBlocks <= NumBlocks) return true;

            // Doubling, like the host pool: growth means an on-device copy of
            // everything already written, so it must be amortised.
            int target = Math.Max(requiredBlocks, NumBlocks * 2);
            try
            {
                if (!GgmlBasicOps.PagedKvPoolGrow(_handle, target))
                    return false;
                NumBlocks = target;
                return true;
            }
            catch (Exception) { return false; }
        }

        /// <summary>Write this step's K and V into the mapped slots.</summary>
        public void Scatter(int layer, float[] k, float[] v, int[] slotMapping, int numTokens)
            => GgmlBasicOps.PagedKvPoolScatter(_handle, layer, k, v, slotMapping, numTokens);

        /// <summary>
        /// Batched flash attention against the pool. <paramref name="output"/> is
        /// filled in place with [numTokens, numHeads * headDim].
        /// </summary>
        public void Attention(
            int layer, float[] q, float[] output,
            int[] queryStartLoc, int[] seqLens, int[] positions,
            int[][] blockTables, int numSeqs, int numTokens, int numHeads,
            float scale, int slidingWindow = 0)
        {
            FlattenBlockTables(blockTables, numSeqs, out int[] flat, out int[] offsets);
            GgmlBasicOps.PagedKvPoolAttention(
                _handle, layer, q, output,
                queryStartLoc, seqLens, positions, flat, offsets,
                numSeqs, numTokens, numHeads, scale, slidingWindow);
        }

        /// <summary>Ragged block tables into one flat array plus per-sequence
        /// offsets, which is the shape the native entry point takes.</summary>
        private static void FlattenBlockTables(int[][] tables, int numSeqs,
            out int[] flat, out int[] offsets)
        {
            offsets = new int[numSeqs + 1];
            int total = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                offsets[s] = total;
                total += tables[s]?.Length ?? 0;
            }
            offsets[numSeqs] = total;

            flat = new int[total];
            int w = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                int[] t = tables[s];
                if (t == null) continue;
                Array.Copy(t, 0, flat, w, t.Length);
                w += t.Length;
            }
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                GgmlBasicOps.PagedKvPoolFree(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }
}
