// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Host-side quantized matmul the DSV4 engine uses for the routed experts of
    /// <c>--n-cpu-moe</c> layers.
    /// </summary>
    /// <remarks>
    /// Inverted like <see cref="IDsv4WeightSource"/>: the managed quantized
    /// kernels live in TensorSharp.Models, which references this assembly and
    /// not the other way round, so the model layer supplies the implementation.
    /// Called concurrently from the loader only; inference calls it from the
    /// single engine thread.
    /// </remarks>
    public interface IDsv4HostMatMul
    {
        /// <summary>One <c>output = input * weights^T</c> of a batch.</summary>
        public readonly struct Job
        {
            /// <summary>Quantized <c>[OutDim, inDim]</c> block matrix.</summary>
            public readonly IntPtr Weights;
            /// <summary><c>float*</c>, <c>[RowCount, inDim]</c>.</summary>
            public readonly IntPtr Input;
            /// <summary><c>float*</c>, <c>[RowCount, OutDim]</c>.</summary>
            public readonly IntPtr Output;
            public readonly int OutDim;
            public readonly int RowCount;
            public readonly int OutputRowStride;

            public Job(IntPtr weights, IntPtr input, IntPtr output, int outDim, int rowCount, int outputRowStride)
            {
                Weights = weights;
                Input = input;
                Output = output;
                OutDim = outDim;
                RowCount = rowCount;
                OutputRowStride = outputRowStride;
            }
        }

        /// <summary>
        /// Run every job under ONE parallel dispatch. A MoE FFN on the host is a
        /// pile of tiny matmuls (one matvec per selected expert per projection
        /// at decode width), and dispatching each separately costs more in
        /// scheduler ramp than the arithmetic it schedules — batching them is
        /// worth ~2x on a DeepSeek V4 token.
        ///
        /// All jobs share <paramref name="ggmlType"/>, <paramref name="inDim"/>
        /// and <paramref name="inputRowStride"/>. Returns false when the
        /// quantized type has no direct host kernel, which the caller reports as
        /// an unsupported-offload error rather than silently producing wrong
        /// output.
        /// </summary>
        bool TryMatMulBatch(int ggmlType, int inDim, int inputRowStride, ReadOnlySpan<Job> jobs);
    }
}
