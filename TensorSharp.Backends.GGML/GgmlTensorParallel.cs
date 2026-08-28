// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Native bindings for GGML tensor parallelism: device enumeration, per-thread
// rank selection, the cross-GPU AllReduce, and the fused multi-rank matmul that
// backs column-/row-parallel linear layers.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    internal static partial class GgmlNative
    {
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetGpuDeviceCount(int backendType);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetGpuDeviceDescription(int backendType, int deviceIndex, byte[] description, int descriptionSize);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_TensorParallelInit(int backendType, int[] deviceIndices, int count, int concurrentRanks);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MultiDeviceInit(int backendType, int[] deviceIndices, int count);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_SetActiveDevice(int rank);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetActiveDevice();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetTensorParallelDegree();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_TensorParallelHasDeviceAllReduce();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_TensorParallelAllReduceHost(float** buffers, int rankCount, long count);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_TensorParallelAllReduceDevice(float** buffers, int rankCount, long count);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_TensorParallelFusedAvailable(int rankCount);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_TensorParallelExecutePlans(IntPtr[] plans, int rankCount);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe partial int TSGgml_TensorParallelMatmul(
            GgmlTensorView2D* results,
            GgmlTensorView2D* inputs,
            IntPtr* weightData,
            int* weightTypes,
            long* weightNe0,
            long* weightNe1,
            long* weightRawBytes,
            int rankCount,
            int allReduce);

        // The native active rank is thread-local, so mirror it here and skip the
        // interop call when it has not changed: the per-op dispatch hook runs on
        // literally every op.
        [ThreadStatic] private static int s_cachedRank;
        [ThreadStatic] private static bool s_cachedRankValid;

        // Set while a thread is executing one rank's share of a RunPerRank body.
        //
        // ggml-cuda's VMM pool (ggml_cuda_pool_vmm) is a bump allocator with no
        // lock that asserts every free is the exact reverse of the matching
        // alloc. It lives on the backend context, so two threads inside
        // graph_compute on the SAME backend corrupt it —
        // "GGML_ASSERT(ptr == pool_addr + pool_used) failed".
        //
        // Rank fan-out is the only place this bridge runs concurrently, and it
        // is safe precisely because each thread owns a different backend. The
        // per-op dispatch hook would break that: an op whose result happens to
        // live on another rank would drag the worker onto that rank's backend
        // while its owner is still using it. So while a rank is pinned the hook
        // leaves the thread alone — the op then runs on the pinned rank's GPU,
        // which is correct (GGML tensors are host memory, any GPU computes the
        // same values) and is the placement the fan-out intended anyway.
        [ThreadStatic] private static bool s_rankPinned;

        /// <summary>Number of GPUs the given GGML backend can address.</summary>
        public static int GetGpuDeviceCount(GgmlBackendType backendType)
        {
            try { return Math.Max(0, TSGgml_GetGpuDeviceCount((int)backendType)); }
            catch (DllNotFoundException) { return 0; }
            catch (EntryPointNotFoundException) { return 0; }
        }

        /// <summary>Adapter name of a GPU, or null when unavailable.</summary>
        public static string GetGpuDeviceDescription(GgmlBackendType backendType, int deviceIndex)
        {
            try
            {
                var buffer = new byte[256];
                if (TSGgml_GetGpuDeviceDescription((int)backendType, deviceIndex, buffer, buffer.Length) == 0)
                    return null;
                int len = Array.IndexOf(buffer, (byte)0);
                if (len < 0) len = buffer.Length;
                return System.Text.Encoding.UTF8.GetString(buffer, 0, len);
            }
            catch (DllNotFoundException) { return null; }
            catch (EntryPointNotFoundException) { return null; }
        }

        /// <summary>
        /// Bring up one ggml backend per listed GPU; rank r maps to
        /// <paramref name="deviceIndices"/>[r].
        /// </summary>
        public static void TensorParallelInit(GgmlBackendType backendType, int[] deviceIndices, bool concurrentRanks)
        {
            if (deviceIndices == null || deviceIndices.Length == 0)
                throw new ArgumentException("At least one device index is required.", nameof(deviceIndices));

            if (TSGgml_TensorParallelInit((int)backendType, deviceIndices, deviceIndices.Length, concurrentRanks ? 1 : 0) == 0)
            {
                throw new InvalidOperationException(GetLastErrorMessage(
                    $"Failed to initialize GGML tensor parallelism across {deviceIndices.Length} device(s)."));
            }
            s_cachedRankValid = false;
        }

        /// <summary>
        /// Bring up one ggml backend per listed GPU for a LAYER SPLIT: same device
        /// setup as <see cref="TensorParallelInit"/>, but no cross-device collective
        /// is created.
        ///
        /// A layer split never reduces across devices - each GPU owns a contiguous
        /// run of layers and only the residual crosses a boundary, through host
        /// memory - so initialising NCCL/P2P would spend the startup time and take
        /// on the lying-P2P first-collective hang risk for machinery that is never
        /// used.
        /// </summary>
        public static void MultiDeviceInit(GgmlBackendType backendType, int[] deviceIndices)
        {
            if (deviceIndices == null || deviceIndices.Length == 0)
                throw new ArgumentException("At least one device index is required.", nameof(deviceIndices));

            if (TSGgml_MultiDeviceInit((int)backendType, deviceIndices, deviceIndices.Length) == 0)
            {
                throw new InvalidOperationException(GetLastErrorMessage(
                    $"Failed to initialize {deviceIndices.Length} GGML device(s) for a layer split."));
            }
            s_cachedRankValid = false;
        }

        /// <summary>Number of ranks the native bridge currently has initialized.</summary>
        public static int TensorParallelDegree()
        {
            try { return Math.Max(1, TSGgml_GetTensorParallelDegree()); }
            catch (EntryPointNotFoundException) { return 1; }
        }

        /// <summary>True when AllReduce runs on-device (NCCL / P2P) rather than through host memory.</summary>
        public static bool TensorParallelHasDeviceAllReduce()
        {
            try { return TSGgml_TensorParallelHasDeviceAllReduce() != 0; }
            catch (EntryPointNotFoundException) { return false; }
        }

        /// <summary>
        /// Select the GPU that subsequent ops on this thread run on. Cheap and
        /// idempotent — safe to call once per op.
        /// </summary>
        public static void SetActiveRank(int rank)
        {
            if (s_cachedRankValid && s_cachedRank == rank)
                return;
            if (TSGgml_SetActiveDevice(rank) == 0)
                throw new ArgumentOutOfRangeException(nameof(rank), GetLastErrorMessage($"Invalid GGML rank {rank}."));
            s_cachedRank = rank;
            s_cachedRankValid = true;
        }

        /// <summary>Current rank for this thread.</summary>
        public static int GetActiveRank()
        {
            if (s_cachedRankValid) return s_cachedRank;
            s_cachedRank = TSGgml_GetActiveDevice();
            s_cachedRankValid = true;
            return s_cachedRank;
        }

        /// <summary>
        /// Rank selection that yields to an active pin. Used by the per-op
        /// dispatch hook so it can place ops by result tensor in single-threaded
        /// code without ever moving a rank worker off its own backend.
        /// </summary>
        public static void SetActiveRankIfUnpinned(int rank)
        {
            if (s_rankPinned) return;
            SetActiveRank(rank);
        }

        /// <summary>
        /// Pin this thread to <paramref name="rank"/> for the duration of a
        /// rank fan-out. Returns the previous pin state so the caller can restore
        /// it (fan-outs can nest — a TP linear inside a TP block).
        /// </summary>
        public static bool PinRank(int rank)
        {
            bool previous = s_rankPinned;
            s_rankPinned = false;
            SetActiveRank(rank);
            s_rankPinned = true;
            return previous;
        }

        /// <summary>Restore the pin state returned by <see cref="PinRank"/>.</summary>
        public static void RestorePin(bool previous, int previousRank)
        {
            s_rankPinned = false;
            SetActiveRank(previousRank);
            s_rankPinned = previous;
        }

        /// <summary>In-place host AllReduce (sum) over one buffer per rank.</summary>
        public static unsafe void TensorParallelAllReduceHost(float** buffers, int rankCount, long count)
        {
            if (TSGgml_TensorParallelAllReduceHost(buffers, rankCount, count) == 0)
                throw new InvalidOperationException(GetLastErrorMessage("GGML tensor-parallel host AllReduce failed."));
        }

        /// <summary>
        /// In-place device AllReduce (sum) over one host buffer per rank, staged
        /// through VRAM and reduced with the backend collective. Returns false
        /// when the collective is unavailable for this payload.
        /// </summary>
        public static unsafe bool TensorParallelAllReduceDevice(float** buffers, int rankCount, long count)
        {
            return TSGgml_TensorParallelAllReduceDevice(buffers, rankCount, count) != 0;
        }

        /// <summary>
        /// True when a fused whole-model tensor-parallel graph can run: more
        /// than one rank and a device collective to reduce the per-layer
        /// partials with. False means the caller should keep its per-op forward,
        /// because every segment boundary would otherwise cost a host round trip.
        /// </summary>
        public static bool TensorParallelFusedAvailable(int rankCount)
        {
            try { return TSGgml_TensorParallelFusedAvailable(rankCount) != 0; }
            catch (EntryPointNotFoundException) { return false; }
        }

        /// <summary>
        /// Run one fused per-rank graph plan per rank: segment k is submitted on
        /// every GPU asynchronously, its row-parallel partials are AllReduced in
        /// VRAM, then segment k+1 follows. This is the schedule llama.cpp's meta
        /// backend uses for <c>--split-mode tensor</c>, and it is what keeps the
        /// activations off the PCIe bus.
        /// </summary>
        public static void TensorParallelExecutePlans(IntPtr[] plans)
        {
            if (plans == null || plans.Length == 0)
                throw new ArgumentException("At least one plan is required.", nameof(plans));
            if (TSGgml_TensorParallelExecutePlans(plans, plans.Length) == 0)
                throw new InvalidOperationException(GetLastErrorMessage("GGML tensor-parallel plan execution failed."));
        }

        /// <summary>
        /// One column-parallel (<paramref name="allReduce"/> false) or
        /// row-parallel (true) linear across every rank, submitted as N
        /// concurrent device graphs and synchronized once.
        /// </summary>
        public static unsafe void TensorParallelMatmul(
            GgmlTensorView2D[] results,
            GgmlTensorView2D[] inputs,
            IntPtr[] weightData,
            int[] weightTypes,
            long[] weightNe0,
            long[] weightNe1,
            long[] weightRawBytes,
            bool allReduce)
        {
            int n = results.Length;
            fixed (GgmlTensorView2D* r = results)
            fixed (GgmlTensorView2D* i = inputs)
            fixed (IntPtr* wd = weightData)
            fixed (int* wt = weightTypes)
            fixed (long* w0 = weightNe0)
            fixed (long* w1 = weightNe1)
            fixed (long* wb = weightRawBytes)
            {
                if (TSGgml_TensorParallelMatmul(r, i, wd, wt, w0, w1, wb, n, allReduce ? 1 : 0) == 0)
                    throw new InvalidOperationException(GetLastErrorMessage("GGML tensor-parallel matmul failed."));
            }
        }
    }
}
