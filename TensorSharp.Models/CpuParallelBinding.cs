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
using System.Runtime.CompilerServices;
using TensorSharp.Cpu;

namespace TensorSharp.Models
{
    /// <summary>
    /// Routes the Core CPU kernels' parallel loops (F32 GEMM, elementwise, norm
    /// and softmax ops behind Ops.* on CpuStorage) onto the same persistent
    /// spinning pool the quantized matmuls use, instead of the ThreadPool.
    /// A decode/diffusion step issues hundreds of these ops, each worth tens of
    /// microseconds per core, and a ThreadPool fork/join (waking parked threads)
    /// costs about as much as the work. Sharing the pool also keeps the process
    /// from running two sets of hot threads (the pool's spinners plus the
    /// ThreadPool's workers) that would fight over the same cores.
    ///
    /// The pool is created lazily on the first parallel Core call, so merely
    /// loading this assembly (GGML/CUDA backends) starts no threads.
    /// TS_CPU_POOL=0 keeps Core on Parallel.For, same knob as the quantized path.
    /// </summary>
    internal static class CpuParallelBinding
    {
        internal static readonly bool Enabled =
            Environment.GetEnvironmentVariable("TS_CPU_POOL") != "0";

#pragma warning disable CA2255 // intentional: bind before any model code runs a Core op
        [ModuleInitializer]
#pragma warning restore CA2255
        internal static void Bind()
        {
            if (!Enabled) return;
            CpuParallel.SetRunner(
                (blockCount, body) => CpuWorkerPool.Shared.For(blockCount, body),
                () => CpuWorkerPool.Shared.ThreadCount);
        }
    }
}
