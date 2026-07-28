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
using System.Reflection;

namespace TensorSharp.GGML
{
    public sealed class GgmlContext
    {
        internal GgmlMemoryPool MemoryPool { get; }

        public GgmlContext(int[] deviceIds, GgmlBackendType backendType)
        {
            if (deviceIds == null || deviceIds.Length == 0)
            {
                throw new ArgumentException("At least one device id is required for the GGML backend.", nameof(deviceIds));
            }

            DeviceIds = (int[])deviceIds.Clone();
            DeviceId = deviceIds[0];
            BackendType = backendType;
            MemoryPool = new GgmlMemoryPool(backendType);
            MemoryPool.EnsureInitialBlocks();
            GgmlNative.EnsureAvailable(backendType);

            if (deviceIds.Length > 1)
            {
                // Tensor parallelism: bring up one ggml backend per GPU. Ops then
                // select a rank with GgmlNative.SetActiveDevice; tensors carry
                // their rank through GgmlAllocator.DeviceId.
                if (backendType != GgmlBackendType.Cuda && backendType != GgmlBackendType.Vulkan)
                {
                    throw new NotSupportedException(
                        $"The GGML {backendType} backend exposes a single device; tensor parallelism requires the CUDA or Vulkan backend.");
                }
                // The native side needs to know whether the ranks will be driven
                // concurrently: that decides whether ggml-cuda's graph capture
                // (which is process-wide and breaks under concurrent CUDA calls)
                // has to be turned off for the run.
                GgmlNative.TensorParallelInit(backendType, DeviceIds, GgmlTensorParallelGroup.ParallelRanks);
                HasDeviceAllReduce = GgmlNative.TensorParallelHasDeviceAllReduce();
            }
            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());

            // On Metal, default to async (lazy) GPU dispatch. This is the same model
            // llama.cpp uses: per-op `ggml_metal_graph_compute` commits its command
            // buffer and returns immediately; only the final `ggml_backend_synchronize`
            // (or a host-side data read via TensorComputePrimitives.GetFloatPointer)
            // actually waits on the GPU. With this enabled we avoid one
            // `[cmd_buf waitUntilCompleted]` (~30-100µs of driver/IPC round-trip on
            // M-series Macs) per op submitted, which dominates prefill on long
            // prompts where TensorSharp's per-op driving model would otherwise
            // submit hundreds of command buffers serially.
            //
            // Set TS_GGML_ASYNC_COMPUTE=0 to disable and fall back to the legacy
            // eager-sync behaviour for debugging.
            var disableAsync = Environment.GetEnvironmentVariable("TS_GGML_ASYNC_COMPUTE");
            bool enableAsync = backendType == GgmlBackendType.Metal &&
                               !string.Equals(disableAsync, "0", StringComparison.Ordinal);
            GgmlNative.SetAsyncCompute(enableAsync);
        }

        /// <summary>Physical device index backing rank 0.</summary>
        public int DeviceId { get; }

        /// <summary>Physical device index per rank; length == the TP degree.</summary>
        public int[] DeviceIds { get; }

        /// <summary>Number of GPUs this context spans (1 = no tensor parallelism).</summary>
        public int Degree => DeviceIds.Length;

        /// <summary>
        /// True when cross-GPU AllReduce runs entirely on the devices (NCCL or
        /// P2P via ggml's CUDA comm backend) rather than through host memory.
        /// </summary>
        public bool HasDeviceAllReduce { get; }

        public GgmlBackendType BackendType { get; }
    }
}
