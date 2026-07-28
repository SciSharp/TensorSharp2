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
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    internal static partial class GgmlNative
    {
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetGpuDeviceCount(int backendType);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetGpuDeviceDescription(int backendType, int deviceIndex, byte[] description, int descriptionSize);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_TensorParallelInit(int backendType, int[] deviceIndices, int count, int concurrentRanks);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SetActiveDevice(int rank);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetActiveDevice();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetTensorParallelDegree();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_TensorParallelHasDeviceAllReduce();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern unsafe int TSGgml_TensorParallelAllReduceHost(float** buffers, int rankCount, long count);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern unsafe int TSGgml_TensorParallelAllReduceDevice(float** buffers, int rankCount, long count);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern unsafe int TSGgml_TensorParallelMatmul(
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
