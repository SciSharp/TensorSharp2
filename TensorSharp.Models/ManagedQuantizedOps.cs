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
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    internal static class ManagedQuantizedOps
    {
        private const int QK4_0 = 32;
        private const int QK4_1 = 32;
        private const int QK5_0 = 32;
        private const int QK5_1 = 32;
        private const int QK8_0 = 32;
        private const int QK8_1 = 32;
        private const int QK4_NL = 32;
        private const int QK_MXFP4 = 32;
        private const int QK_K = 256;
        private const int K_SCALE_SIZE = 12;
        private const int Q4_0BlockBytes = 2 + QK4_0 / 2;
        private const int Q4_1BlockBytes = 4 + QK4_1 / 2;
        private const int Q5_0BlockBytes = 2 + 4 + QK5_0 / 2;
        private const int Q5_1BlockBytes = 4 + 4 + QK5_1 / 2;
        private const int Q8_0BlockBytes = 2 + QK8_0;
        private const int Q8_1BlockBytes = 4 + QK8_1;
        private const int Q4_KBlockBytes = 4 + K_SCALE_SIZE + QK_K / 2;
        private const int Q5_KBlockBytes = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
        private const int Q6_KBlockBytes = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
        private const int Q8_KBlockBytes = 4 + QK_K + 2 * (QK_K / 16);

        private static readonly sbyte[] Iq4NlValues =
        {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        };

        private static readonly sbyte[] Mxfp4Values =
        {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
        };

        public static bool SupportsCpuQuantizedStorage(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F16 => true,
                GgmlTensorType.BF16 => true,
                GgmlTensorType.Q4_0 => true,
                GgmlTensorType.Q4_1 => true,
                GgmlTensorType.Q5_0 => true,
                GgmlTensorType.Q5_1 => true,
                GgmlTensorType.Q8_0 => true,
                GgmlTensorType.Q8_1 => true,
                GgmlTensorType.Q2_K => true,
                GgmlTensorType.Q3_K => true,
                GgmlTensorType.Q4_K => true,
                GgmlTensorType.Q5_K => true,
                GgmlTensorType.Q6_K => true,
                GgmlTensorType.IQ4_NL => true,
                GgmlTensorType.IQ2_XXS => true,
                GgmlTensorType.IQ2_S => true,
                GgmlTensorType.IQ3_S => true,
                GgmlTensorType.MXFP4 => true,
                _ => false,
            };
        }

        public static bool SupportsDequantization(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F32 => true,
                GgmlTensorType.F16 => true,
                GgmlTensorType.BF16 => true,
                GgmlTensorType.I8 => true,
                GgmlTensorType.I16 => true,
                GgmlTensorType.I32 => true,
                GgmlTensorType.I64 => true,
                GgmlTensorType.F64 => true,
                _ => SupportsCpuQuantizedStorage(type),
            };
        }

        public static long RowSize(int ggmlType, long ne)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            long blockSize = GgufFile.GetBlockSize(type);
            if (ne % blockSize != 0)
                throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {ne}.");

            return (ne / blockSize) * GgufFile.GetTypeSize(type);
        }

        public static unsafe void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (byte* srcBase = src)
            fixed (float* dstBase = dst)
            {
                DequantizeToFloat32(type, srcBase + srcOffset, dstBase + dstOffset, numElements);
            }
        }

        public static unsafe void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (float* dstBase = dst)
            {
                DequantizeToFloat32(type, (byte*)src.ToPointer(), dstBase + dstOffset, numElements);
            }
        }

        public static unsafe void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            DequantizeToFloat32(type, (byte*)src.ToPointer(), (float*)dst.ToPointer(), numElements);
        }

        public static unsafe void DequantizeRowToFloat32(int ggmlType, IntPtr src, float* dst, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            DequantizeToFloat32(type, (byte*)src.ToPointer(), dst, numElements);
        }

        /// <summary>
        /// Quantize a contiguous run of <paramref name="numElements"/> F32 values into a
        /// block-quantized buffer (Q4_0 or Q8_0), matching ggml's reference block layout
        /// (fp16 block scale + packed quants). Used by the managed KV-cache write path so
        /// that block-quantized caches (<c>--kv-cache-dtype q4_0/q8_0</c>) can be appended
        /// to from the per-op prefill path; the bytes it produces are dequantized
        /// identically by ggml's native kernels on the subsequent fused decode read.
        /// <paramref name="numElements"/> must be a multiple of the 32-element block size.
        /// </summary>
        public static unsafe void QuantizeRowFromFloat32(int ggmlType, float* src, IntPtr dst, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            byte* d = (byte*)dst.ToPointer();
            switch (type)
            {
                case GgmlTensorType.Q4_0:
                    if (numElements % QK4_0 != 0)
                        throw new NotSupportedException($"Q4_0 requires {QK4_0}-element alignment, got {numElements}.");
                    QuantizeF32ToQ4_0(src, d, (int)numElements);
                    break;
                case GgmlTensorType.Q8_0:
                    if (numElements % QK8_0 != 0)
                        throw new NotSupportedException($"Q8_0 requires {QK8_0}-element alignment, got {numElements}.");
                    QuantizeF32ToQ8_0(src, d, (int)numElements);
                    break;
                default:
                    throw new NotSupportedException($"QuantizeRowFromFloat32 does not support GGUF tensor type {type}.");
            }
        }

        public static unsafe void QuantizeRowFromFloat32(int ggmlType, float[] src, int srcOffset, byte[] dst, int dstOffset, long numElements)
        {
            fixed (float* s = src)
            fixed (byte* d = dst)
            {
                QuantizeRowFromFloat32(ggmlType, s + srcOffset, (IntPtr)(d + dstOffset), numElements);
            }
        }

        public static unsafe void DotRowBatchToFloat32(int ggmlType, byte[] src, int srcOffset,
            float[] inputs, int inputOffset, int inputRowStride, int rowCount, long numElements,
            float[] outputs, int outputOffset)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (byte* srcBase = src)
            fixed (float* inputBase = inputs)
            fixed (float* outputBase = outputs)
            {
                DotRowBatchToFloat32(
                    ggmlType,
                    (IntPtr)(srcBase + srcOffset),
                    inputBase + inputOffset,
                    inputRowStride,
                    rowCount,
                    numElements,
                    outputBase + outputOffset);
            }
        }

        public static unsafe void DotRowBatchToFloat32(int ggmlType, IntPtr src, float* inputs,
            int inputRowStride, int rowCount, long numElements, float* outputs)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");
            if (rowCount < 1)
                throw new ArgumentOutOfRangeException(nameof(rowCount));
            if (inputRowStride < numElements)
                throw new ArgumentOutOfRangeException(nameof(inputRowStride));

            long blockSize = GgufFile.GetBlockSize(type);
            if (numElements % blockSize != 0)
                throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {numElements}.");

            for (int row = 0; row < rowCount; row++)
                outputs[row] = 0.0f;

            if (type == GgmlTensorType.F32)
            {
                float* weight = (float*)src.ToPointer();
                for (int row = 0; row < rowCount; row++)
                    outputs[row] = DotFloat(inputs + (long)row * inputRowStride, weight, (int)numElements);
                return;
            }

            float* scratch = stackalloc float[QK_K];
            byte* chunkPtr = (byte*)src.ToPointer();
            long elementOffset = 0;

            while (elementOffset < numElements)
            {
                int chunkElements = GetDotChunkSize(type, numElements - elementOffset);
                DequantizeToFloat32(type, chunkPtr, scratch, chunkElements);

                float* inputChunk = inputs + elementOffset;
                for (int row = 0; row < rowCount; row++)
                {
                    outputs[row] += DotFloat(inputChunk + (long)row * inputRowStride, scratch, chunkElements);
                }

                chunkPtr += GetDotChunkBytes(type, chunkElements);
                elementOffset += chunkElements;
            }
        }


        /// <summary>
        /// The pure-C# quantized/dense linear:
        /// output[row, col] = dot(weightRow(col), input[row]) for every row/col.
        ///
        /// One implementation for every managed caller (ModelBase's linear layers,
        /// the DeepSeek V4 CPU executor, the quant-matmul bench): weight types with
        /// an integer dot kernel quantize the activations once and run
        /// <see cref="TryAddmmQuantizedToFloat32(int, IntPtr, long, long, float*, int, int, float*, int)"/>;
        /// everything else (F16/BF16/F32 and any quant without a direct kernel)
        /// dequantizes each weight row ONCE into an L1-hot scratch and dots it with
        /// four activation rows at a time.
        ///
        /// Row strides let callers point at a sub-block of a wider scratch buffer,
        /// and <paramref name="options"/> caps the degree of parallelism (the DSV4
        /// executor runs under a CPU quota and sets it explicitly).
        /// </summary>
        public static unsafe void AddmmQuantizedToFloat32(
            int ggmlType,
            IntPtr weights,
            long ne0,
            long ne1,
            float* input,
            int inputRowStride,
            int rowCount,
            float* output,
            int outputRowStride,
            ParallelOptions options = null)
        {
            if (TryAddmmQuantizedToFloat32(
                    ggmlType, weights, ne0, ne1, input, inputRowStride, rowCount, output, outputRowStride, options))
            {
                return;
            }

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            long rowBytes = RowSize(ggmlType, ne0);
            byte* weightBase = (byte*)weights.ToPointer();
            int dop = options?.MaxDegreeOfParallelism > 0 ? options.MaxDegreeOfParallelism : Environment.ProcessorCount;

            void RunRange(int start, int end, float* w)
                => DequantMatMulColumns(ggmlType, weightBase, rowBytes, inDim, outDim,
                    input, inputRowStride, rowCount, output, outputRowStride, start, end, w);

            if (outDim < 128 || (long)rowCount * outDim < 512 || dop <= 1)
            {
                float[] scratch = ArrayPool<float>.Shared.Rent(inDim);
                try
                {
                    fixed (float* w = scratch)
                        RunRange(0, outDim, w);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scratch);
                }
                return;
            }

            // Blocked column ranges, not one task per column: a single dequant+dot
            // column is ~microseconds, so per-column tasks are dominated by
            // scheduler overhead and stop scaling after a handful of cores.
            int colBlock = Math.Max(4, outDim / (dop * 8));
            int nBlocks = (outDim + colBlock - 1) / colBlock;
            Parallel.For(0, nBlocks, options ?? new ParallelOptions(), b =>
            {
                float[] scratch = ArrayPool<float>.Shared.Rent(inDim);
                try
                {
                    fixed (float* w = scratch)
                        RunRange(b * colBlock, Math.Min(outDim, b * colBlock + colBlock), w);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scratch);
                }
            });
        }

        // Core of the pure-C# quantized linear (shared by AddmmQuantManaged and the
        // quant-matmul benchmark/self-test): for each output column in [startCol,endCol),
        // dequantize its weight row into <paramref name="wScratch"/> (inDim floats, hot in L1)
        // and dot it with every activation row using register-blocked VecDot4. Dequant honours
        // NativeDequant.PreferManaged (managed on the pure-C# CPU backend).
        public static unsafe void DequantMatMulColumns(
            int ggmlType, byte* weightBase, long rowBytes, int inDim, int outDim,
            float* inputPtr, int inputRowStride, int seqLen, float* resultPtr, int outputRowStride,
            int startCol, int endCol, float* wScratch)
        {
            for (int col = startCol; col < endCol; col++)
            {
                byte* rowPtr = weightBase + (long)col * rowBytes;
                NativeDequant.DequantizeToFloat32Native(ggmlType, (IntPtr)rowPtr, (IntPtr)wScratch, inDim);

                int row = 0;
                for (; row + 4 <= seqLen; row += 4)
                {
                    TensorComputePrimitives.Dot4(
                        inputPtr + (long)row * inputRowStride,
                        inputPtr + (long)(row + 1) * inputRowStride,
                        inputPtr + (long)(row + 2) * inputRowStride,
                        inputPtr + (long)(row + 3) * inputRowStride,
                        wScratch, inDim, out float r0, out float r1, out float r2, out float r3);
                    resultPtr[(long)row * outputRowStride + col] = r0;
                    resultPtr[(long)(row + 1) * outputRowStride + col] = r1;
                    resultPtr[(long)(row + 2) * outputRowStride + col] = r2;
                    resultPtr[(long)(row + 3) * outputRowStride + col] = r3;
                }
                for (; row < seqLen; row++)
                    resultPtr[(long)row * outputRowStride + col] = TensorComputePrimitives.Dot(inputPtr + (long)row * inputRowStride, wScratch, inDim);
            }
        }

        /// <summary>
        /// One <c>output = input * weights^T</c> in a batch that shares a single
        /// parallel dispatch. See <see cref="TryAddmmQuantizedBatch"/>.
        /// </summary>
        public readonly struct QuantMatMulJob
        {
            /// <summary>Quantized <c>[OutDim, InDim]</c> block matrix.</summary>
            public readonly IntPtr Weights;
            /// <summary><c>float*</c>, <c>[RowCount, inDim]</c>.</summary>
            public readonly IntPtr Input;
            /// <summary><c>float*</c>, <c>[RowCount, OutDim]</c>.</summary>
            public readonly IntPtr Output;
            public readonly int OutDim;
            public readonly int RowCount;
            public readonly int OutputRowStride;

            public QuantMatMulJob(IntPtr weights, IntPtr input, IntPtr output, int outDim, int rowCount,
                int outputRowStride)
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
        /// Several small quantized matmuls under ONE parallel dispatch.
        ///
        /// <para>A Mixture-of-Experts FFN on the host is a pile of tiny matmuls —
        /// at decode width, <c>n_expert_used</c> matvecs per projection per
        /// layer. Run through <see cref="TryAddmmQuantizedToFloat32"/> one at a
        /// time, each pays a <see cref="Parallel.For"/> ramp that is comparable
        /// to the arithmetic it schedules; a DeepSeek V4 token with 13 offloaded
        /// layers issued 234 of them and spent most of the time in the
        /// scheduler. Batching flattens every job's column blocks into one index
        /// space, so a layer costs one dispatch per projection stage instead of
        /// one per (expert, projection).</para>
        ///
        /// <para>All jobs share <paramref name="ggmlType"/>, <paramref name="inDim"/>
        /// and <paramref name="inputRowStride"/>. Jobs whose <c>Input</c> pointer
        /// repeats (a gate and an up projection over the same rows) quantize
        /// those activations once.</para>
        /// </summary>
        public static unsafe bool TryAddmmQuantizedBatch(
            int ggmlType,
            int inDim,
            int inputRowStride,
            ReadOnlySpan<QuantMatMulJob> jobs,
            ParallelOptions options = null)
        {
            if (jobs.Length == 0)
                return true;
            var type = (GgmlTensorType)ggmlType;
            if (!TryGetDirectMatMulPlan(type, inDim, out ActivationQuantKind activationKind, out int activationRowBytes))
                return false;

            // --- quantize each DISTINCT input block once ---
            int n = jobs.Length;
            var actOf = new int[n];          // job -> index into the activation blocks
            var blockInput = new IntPtr[n];
            var blockRows = new int[n];
            var blockOffset = new long[n];
            int nBlocks = 0;
            long totalRows = 0;
            for (int j = 0; j < n; j++)
            {
                int found = -1;
                for (int b = 0; b < nBlocks; b++)
                    if (blockInput[b] == jobs[j].Input && blockRows[b] == jobs[j].RowCount)
                    {
                        found = b;
                        break;
                    }
                if (found < 0)
                {
                    found = nBlocks++;
                    blockInput[found] = jobs[j].Input;
                    blockRows[found] = jobs[j].RowCount;
                    blockOffset[found] = totalRows;
                    totalRows += jobs[j].RowCount;
                }
                actOf[j] = found;
            }

            long totalActivationBytes = totalRows * activationRowBytes;
            if (totalActivationBytes > int.MaxValue)
                return false;

            int dop = options?.MaxDegreeOfParallelism > 0 ? options.MaxDegreeOfParallelism : Environment.ProcessorCount;
            byte[] rented = ArrayPool<byte>.Shared.Rent((int)totalActivationBytes);
            try
            {
                fixed (byte* activationBase = rented)
                {
                    for (int b = 0; b < nBlocks; b++)
                    {
                        float* src = (float*)blockInput[b];
                        byte* dst = activationBase + blockOffset[b] * activationRowBytes;
                        for (int row = 0; row < blockRows[b]; row++)
                            QuantizeActivation(src + (long)row * inputRowStride,
                                dst + (long)row * activationRowBytes, inDim, activationKind);
                    }

                    // --- flatten every job's output columns into one block space ---
                    int weightRowBytes = (int)RowSize(ggmlType, inDim);
                    var blockStart = new int[n + 1];
                    int colBlock = 0;
                    {
                        long totalCols = 0;
                        for (int j = 0; j < n; j++) totalCols += jobs[j].OutDim;
                        // Aim for a few chunks per worker: fewer and the tail
                        // straggles, more and the per-chunk bookkeeping shows up.
                        colBlock = (int)Math.Max(8, totalCols / Math.Max(1, dop * 8));
                    }
                    int totalBlocks = 0;
                    for (int j = 0; j < n; j++)
                    {
                        blockStart[j] = totalBlocks;
                        totalBlocks += (jobs[j].OutDim + colBlock - 1) / colBlock;
                    }
                    blockStart[n] = totalBlocks;

                    // Span cannot cross the lambda, so copy the jobs' raw fields.
                    var wPtr = new nint[n];
                    var oPtr = new nint[n];
                    var outDim = new int[n];
                    var rowCnt = new int[n];
                    var outStride = new int[n];
                    var actOff = new long[n];
                    for (int j = 0; j < n; j++)
                    {
                        wPtr[j] = jobs[j].Weights;
                        oPtr[j] = jobs[j].Output;
                        outDim[j] = jobs[j].OutDim;
                        rowCnt[j] = jobs[j].RowCount;
                        outStride[j] = jobs[j].OutputRowStride;
                        actOff[j] = blockOffset[actOf[j]] * activationRowBytes;
                    }
                    nint actAddr = (nint)activationBase;
                    int width = inDim;
                    int actBytes = activationRowBytes;

                    void RunBlock(int block)
                    {
                        // locate the job owning this block (n is small: experts
                        // per layer, at most a few hundred)
                        int j = 0;
                        while (j + 1 < n && blockStart[j + 1] <= block) j++;
                        int local = block - blockStart[j];
                        int startCol = local * colBlock;
                        int endCol = Math.Min(outDim[j], startCol + colBlock);

                        byte* act = (byte*)actAddr + actOff[j];
                        byte* w = (byte*)wPtr[j];
                        float* outp = (float*)oPtr[j];
                        for (int col = startCol; col < endCol; col++)
                        {
                            byte* weightRow = w + (long)col * weightRowBytes;
                            for (int row = 0; row < rowCnt[j]; row++)
                                outp[(long)row * outStride[j] + col] =
                                    DotQuantized(type, weightRow, act + (long)row * actBytes, width);
                        }
                    }

                    if (totalBlocks > 1 && dop > 1)
                        Parallel.For(0, totalBlocks, options ?? new ParallelOptions(), RunBlock);
                    else
                        for (int b = 0; b < totalBlocks; b++) RunBlock(b);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }

            return true;
        }

        public static unsafe bool TryAddmmQuantizedToFloat32(
            int ggmlType,
            IntPtr weights,
            long ne0,
            long ne1,
            float* input,
            int inputRowStride,
            int rowCount,
            float* output,
            int outputRowStride,
            ParallelOptions options = null)
        {
            var type = (GgmlTensorType)ggmlType;
            if (ne0 > int.MaxValue || ne1 > int.MaxValue)
                return false;

            if (!TryGetDirectMatMulPlan(type, (int)ne0, out ActivationQuantKind activationKind, out int activationRowBytes))
                return false;

            if (weights == IntPtr.Zero)
                throw new ArgumentException("Quantized weights pointer cannot be null.", nameof(weights));
            if (inputRowStride < ne0)
                throw new ArgumentOutOfRangeException(nameof(inputRowStride));
            if (outputRowStride < ne1)
                throw new ArgumentOutOfRangeException(nameof(outputRowStride));

            long totalActivationBytes = (long)rowCount * activationRowBytes;
            if (totalActivationBytes > int.MaxValue)
                return false;

            byte[] rented = ArrayPool<byte>.Shared.Rent((int)totalActivationBytes);
            try
            {
                fixed (byte* activationBase = rented)
                {
                    int dop = options?.MaxDegreeOfParallelism > 0 ? options.MaxDegreeOfParallelism : Environment.ProcessorCount;
                    if (rowCount >= 8 && dop > 1)
                    {
                        // Prefill batches quantize thousands of activation rows;
                        // serially that is a measurable share of the matmul.
                        nint actAddr = (nint)activationBase;
                        nint inAddr = (nint)input;
                        int actBytes = activationRowBytes;
                        int width = (int)ne0;
                        int stride = inputRowStride;
                        var kind = activationKind;
                        Parallel.For(0, rowCount, options ?? new ParallelOptions(), row =>
                            QuantizeActivation((float*)inAddr + (long)row * stride,
                                (byte*)actAddr + (long)row * actBytes, width, kind));
                    }
                    else
                    {
                        for (int row = 0; row < rowCount; row++)
                        {
                            byte* dst = activationBase + (long)row * activationRowBytes;
                            float* src = input + (long)row * inputRowStride;
                            QuantizeActivation(src, dst, (int)ne0, activationKind);
                        }
                    }

                    byte* weightBase = (byte*)weights.ToPointer();
                    int weightRowBytes = (int)RowSize(ggmlType, ne0);
                    int outDim = (int)ne1;
                    int inDim = (int)ne0;
                    nint activationAddress = (nint)activationBase;
                    nint weightAddress = (nint)weightBase;
                    nint outputAddress = (nint)output;

                    void ComputeColumnRange(int startCol, int endCol)
                    {
                        byte* activationPtr = (byte*)activationAddress;
                        byte* weightPtr = (byte*)weightAddress;
                        float* outputPtr = (float*)outputAddress;

                        for (int col = startCol; col < endCol; col++)
                        {
                            byte* weightRow = weightPtr + (long)col * weightRowBytes;
                            for (int row = 0; row < rowCount; row++)
                            {
                                byte* activationRow = activationPtr + (long)row * activationRowBytes;
                                outputPtr[(long)row * outputRowStride + col] =
                                    DotQuantized(type, weightRow, activationRow, inDim);
                            }
                        }
                    }

                    bool useParallel = outDim >= 128 && (long)rowCount * outDim >= 512 && dop > 1;
                    if (useParallel)
                    {
                        // Blocked ranges: per-column tasks make the scheduler
                        // overhead comparable to the ~0.5-2us dot itself and cap
                        // scaling at a handful of cores.
                        int colBlock = Math.Max(8, outDim / (dop * 8));
                        int nBlocks = (outDim + colBlock - 1) / colBlock;
                        Parallel.For(0, nBlocks, options ?? new ParallelOptions(), b =>
                            ComputeColumnRange(b * colBlock, Math.Min(outDim, b * colBlock + colBlock)));
                    }
                    else
                    {
                        ComputeColumnRange(0, outDim);
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }

            return true;
        }

        public static unsafe bool TryAddmmQuantizedToFloat32(
            int ggmlType,
            byte[] weights,
            int weightsOffset,
            long ne0,
            long ne1,
            float[] input,
            int inputOffset,
            int inputRowStride,
            int rowCount,
            float[] output,
            int outputOffset,
            int outputRowStride)
        {
            if (weights == null)
                throw new ArgumentNullException(nameof(weights));
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (output == null)
                throw new ArgumentNullException(nameof(output));

            fixed (byte* weightPtr = weights)
            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                return TryAddmmQuantizedToFloat32(
                    ggmlType,
                    (IntPtr)(weightPtr + weightsOffset),
                    ne0,
                    ne1,
                    inputPtr + inputOffset,
                    inputRowStride,
                    rowCount,
                    outputPtr + outputOffset,
                    outputRowStride);
            }
        }

        private enum ActivationQuantKind
        {
            Q8_0,
            Q8_1,
            Q8_K,
        }

        /// <summary>
        /// Exposes the direct quantized-dot plan so callers that drive their own
        /// parallel loops (e.g. the DeepSeek4 CPU executor) can quantize the
        /// activations once and then dot weight rows in whatever blocking works
        /// best for their shapes. Returns false when <paramref name="type"/> has
        /// no integer fast path (callers fall back to dequant + float dot).
        /// </summary>
        internal static bool TryGetActivationPlan(GgmlTensorType type, int elementCount, out int activationRowBytes)
        {
            bool ok = TryGetDirectMatMulPlan(type, elementCount, out _, out activationRowBytes);
            return ok;
        }

        internal static unsafe void QuantizeActivationRow(GgmlTensorType weightType, float* src, byte* dst, int elementCount)
        {
            if (!TryGetDirectMatMulPlan(weightType, elementCount, out ActivationQuantKind kind, out _))
                throw new NotSupportedException($"No direct activation plan for {weightType}.");
            QuantizeActivation(src, dst, elementCount, kind);
        }

        internal static unsafe float DotQuantizedRow(GgmlTensorType type, byte* weightRow, byte* activationRow, int elementCount)
        {
            return DotQuantized(type, weightRow, activationRow, elementCount);
        }

        private static bool TryGetDirectMatMulPlan(
            GgmlTensorType type,
            int elementCount,
            out ActivationQuantKind activationKind,
            out int activationRowBytes)
        {
            activationKind = default;
            activationRowBytes = 0;

            switch (type)
            {
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q5_0:
                case GgmlTensorType.Q8_0:
                case GgmlTensorType.Q8_1:
                    if (elementCount % QK8_0 != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_0;
                    activationRowBytes = elementCount / QK8_0 * Q8_0BlockBytes;
                    return true;

                case GgmlTensorType.Q4_1:
                case GgmlTensorType.Q5_1:
                    if (elementCount % QK8_1 != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_1;
                    activationRowBytes = elementCount / QK8_1 * Q8_1BlockBytes;
                    return true;

                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                case GgmlTensorType.Q6_K:
                case GgmlTensorType.IQ3_S:
                    if (elementCount % QK_K != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_K;
                    activationRowBytes = elementCount / QK_K * Q8_KBlockBytes;
                    return true;

                case GgmlTensorType.MXFP4:
                    if (elementCount % QK_MXFP4 != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_0;
                    activationRowBytes = elementCount / QK8_0 * Q8_0BlockBytes;
                    return true;

                default:
                    return false;
            }
        }

        private static unsafe void QuantizeActivation(float* src, byte* dst, int elementCount, ActivationQuantKind kind)
        {
            switch (kind)
            {
                case ActivationQuantKind.Q8_0:
                    QuantizeF32ToQ8_0(src, dst, elementCount);
                    return;
                case ActivationQuantKind.Q8_1:
                    QuantizeF32ToQ8_1(src, dst, elementCount);
                    return;
                case ActivationQuantKind.Q8_K:
                    QuantizeF32ToQ8_K(src, dst, elementCount);
                    return;
                default:
                    throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
            }
        }

        private static unsafe float DotQuantized(GgmlTensorType type, byte* weightRow, byte* activationRow, int elementCount)
        {
            return type switch
            {
                GgmlTensorType.Q4_0 => VecDotQ4_0Q8_0(weightRow, activationRow, elementCount / QK4_0),
                GgmlTensorType.Q4_1 => VecDotQ4_1Q8_1(weightRow, activationRow, elementCount / QK4_1),
                GgmlTensorType.Q5_0 => VecDotQ5_0Q8_0(weightRow, activationRow, elementCount / QK5_0),
                GgmlTensorType.Q5_1 => VecDotQ5_1Q8_1(weightRow, activationRow, elementCount / QK5_1),
                GgmlTensorType.Q8_0 => VecDotQ8_0Q8_0(weightRow, activationRow, elementCount / QK8_0),
                GgmlTensorType.Q8_1 => VecDotQ8_1Q8_0(weightRow, activationRow, elementCount / QK8_1),
                GgmlTensorType.Q4_K => VecDotQ4_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q5_K => VecDotQ5_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q6_K => VecDotQ6_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.IQ3_S => VecDotIq3SQ8K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.MXFP4 => VecDotMxfp4Q8_0(weightRow, activationRow, elementCount / QK_MXFP4),
                _ => throw new NotSupportedException($"Direct managed quantized matmul does not support {type}."),
            };
        }

        private static unsafe void DequantizeToFloat32(GgmlTensorType type, byte* src, float* dst, long numElements)
        {
            switch (type)
            {
                case GgmlTensorType.F32:
                    Buffer.MemoryCopy(src, dst, numElements * sizeof(float), numElements * sizeof(float));
                    return;
                case GgmlTensorType.F16:
                    DequantizeF16(src, dst, numElements);
                    return;
                case GgmlTensorType.BF16:
                    DequantizeBf16(src, dst, numElements);
                    return;
                case GgmlTensorType.I8:
                    DequantizeI8(src, dst, numElements);
                    return;
                case GgmlTensorType.I16:
                    DequantizeI16(src, dst, numElements);
                    return;
                case GgmlTensorType.I32:
                    DequantizeI32(src, dst, numElements);
                    return;
                case GgmlTensorType.I64:
                    DequantizeI64(src, dst, numElements);
                    return;
                case GgmlTensorType.F64:
                    DequantizeF64(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_0:
                    DequantizeQ40(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_1:
                    DequantizeQ41(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_0:
                    DequantizeQ50(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_1:
                    DequantizeQ51(src, dst, numElements);
                    return;
                case GgmlTensorType.Q8_0:
                    DequantizeQ80(src, dst, numElements);
                    return;
                case GgmlTensorType.Q8_1:
                    DequantizeQ81(src, dst, numElements);
                    return;
                case GgmlTensorType.Q2_K:
                    DequantizeQ2K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q3_K:
                    DequantizeQ3K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_K:
                    DequantizeQ4K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_K:
                    DequantizeQ5K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q6_K:
                    DequantizeQ6K(src, dst, numElements);
                    return;
                case GgmlTensorType.IQ4_NL:
                    DequantizeIq4Nl(src, dst, numElements);
                    return;
                case GgmlTensorType.IQ2_XXS:
                    DequantizeIq2Xxs(src, dst, numElements);
                    return;
                case GgmlTensorType.IQ2_S:
                    DequantizeIq2S(src, dst, numElements);
                    return;
                case GgmlTensorType.IQ3_S:
                    DequantizeIq3S(src, dst, numElements);
                    return;
                case GgmlTensorType.MXFP4:
                    DequantizeMxfp4(src, dst, numElements);
                    return;
                default:
                    throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");
            }
        }

        private static unsafe void DequantizeF16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = HalfToSingle(ReadUInt16(src + i * 2));
        }

        private static unsafe void DequantizeBf16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
            {
                uint bits = (uint)ReadUInt16(src + i * 2) << 16;
                dst[i] = BitConverter.Int32BitsToSingle((int)bits);
            }
        }

        private static unsafe void DequantizeI8(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ((sbyte*)src)[i];
        }

        private static unsafe void DequantizeI16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = (short)ReadUInt16(src + i * 2);
        }

        private static unsafe void DequantizeI32(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ReadInt32(src + i * 4);
        }

        private static unsafe void DequantizeI64(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ReadInt64(src + i * 8);
        }

        private static unsafe void DequantizeF64(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = (float)ReadDouble(src + i * 8);
        }

        private static unsafe void DequantizeQ40(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_0 != 0)
                throw new NotSupportedException($"Q4_0 requires {QK4_0}-element alignment, got {numElements}.");

            int nb = (int)(numElements / QK4_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * (2 + QK4_0 / 2);
                float d = HalfToSingle(ReadUInt16(block));
                byte* qs = block + 2;
                float* y = dst + i * QK4_0;
                for (int j = 0; j < QK4_0 / 2; j++)
                {
                    int x0 = (qs[j] & 0x0F) - 8;
                    int x1 = (qs[j] >> 4) - 8;
                    y[j] = x0 * d;
                    y[j + QK4_0 / 2] = x1 * d;
                }
            }
        }

        private static unsafe void DequantizeQ41(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_1 != 0)
                throw new NotSupportedException($"Q4_1 requires {QK4_1}-element alignment, got {numElements}.");

            int nb = (int)(numElements / QK4_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * (4 + QK4_1 / 2);
                float d = HalfToSingle(ReadUInt16(block));
                float m = HalfToSingle(ReadUInt16(block + 2));
                byte* qs = block + 4;
                float* y = dst + i * QK4_1;
                for (int j = 0; j < QK4_1 / 2; j++)
                {
                    int x0 = qs[j] & 0x0F;
                    int x1 = qs[j] >> 4;
                    y[j] = x0 * d + m;
                    y[j + QK4_1 / 2] = x1 * d + m;
                }
            }
        }

        private static unsafe void DequantizeQ50(byte* src, float* dst, long numElements)
        {
            if (numElements % QK5_0 != 0)
                throw new NotSupportedException($"Q5_0 requires {QK5_0}-element alignment, got {numElements}.");

            int blockBytes = 2 + 4 + QK5_0 / 2;
            int nb = (int)(numElements / QK5_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                uint qh = ReadUInt32(block + 2);
                byte* qs = block + 6;
                float* y = dst + i * QK5_0;
                for (int j = 0; j < QK5_0 / 2; j++)
                {
                    int xh0 = (int)(((qh >> j) << 4) & 0x10);
                    int xh1 = (int)((qh >> (j + 12)) & 0x10);
                    int x0 = ((qs[j] & 0x0F) | xh0) - 16;
                    int x1 = ((qs[j] >> 4) | xh1) - 16;
                    y[j] = x0 * d;
                    y[j + QK5_0 / 2] = x1 * d;
                }
            }
        }

        private static unsafe void DequantizeQ51(byte* src, float* dst, long numElements)
        {
            if (numElements % QK5_1 != 0)
                throw new NotSupportedException($"Q5_1 requires {QK5_1}-element alignment, got {numElements}.");

            int blockBytes = 4 + 4 + QK5_1 / 2;
            int nb = (int)(numElements / QK5_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float m = HalfToSingle(ReadUInt16(block + 2));
                uint qh = ReadUInt32(block + 4);
                byte* qs = block + 8;
                float* y = dst + i * QK5_1;
                for (int j = 0; j < QK5_1 / 2; j++)
                {
                    int xh0 = (int)(((qh >> j) << 4) & 0x10);
                    int xh1 = (int)((qh >> (j + 12)) & 0x10);
                    int x0 = (qs[j] & 0x0F) | xh0;
                    int x1 = (qs[j] >> 4) | xh1;
                    y[j] = x0 * d + m;
                    y[j + QK5_1 / 2] = x1 * d + m;
                }
            }
        }

        private static unsafe void DequantizeQ80(byte* src, float* dst, long numElements)
        {
            if (numElements % QK8_0 != 0)
                throw new NotSupportedException($"Q8_0 requires {QK8_0}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK8_0;
            int nb = (int)(numElements / QK8_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                sbyte* qs = (sbyte*)(block + 2);
                float* y = dst + i * QK8_0;
                for (int j = 0; j < QK8_0; j++)
                    y[j] = qs[j] * d;
            }
        }

        private static unsafe void DequantizeQ81(byte* src, float* dst, long numElements)
        {
            if (numElements % QK8_1 != 0)
                throw new NotSupportedException($"Q8_1 requires {QK8_1}-element alignment, got {numElements}.");

            int blockBytes = 4 + QK8_1;
            int nb = (int)(numElements / QK8_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                sbyte* qs = (sbyte*)(block + 4);
                float* y = dst + i * QK8_1;
                for (int j = 0; j < QK8_1; j++)
                    y[j] = qs[j] * d;
            }
        }

        private static unsafe void DequantizeQ4K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q4_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 4 + K_SCALE_SIZE + QK_K / 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float min = HalfToSingle(ReadUInt16(block + 2));
                byte* scales = block + 4;
                byte* q = block + 4 + K_SCALE_SIZE;
                float* y = dst + i * QK_K;
                int isIdx = 0;
                for (int j = 0; j < QK_K; j += 64)
                {
                    GetScaleMinK4(isIdx, scales, out byte sc1, out byte m1q);
                    GetScaleMinK4(isIdx + 1, scales, out byte sc2, out byte m2q);
                    float d1 = d * sc1;
                    float d2 = d * sc2;
                    float m1 = min * m1q;
                    float m2 = min * m2q;
                    for (int l = 0; l < 32; l++)
                        y[j + l] = d1 * (q[l] & 0x0F) - m1;
                    for (int l = 0; l < 32; l++)
                        y[j + l + 32] = d2 * (q[l] >> 4) - m2;
                    q += 32;
                    isIdx += 2;
                }
            }
        }

        private static unsafe void DequantizeQ5K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q5_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float min = HalfToSingle(ReadUInt16(block + 2));
                byte* scales = block + 4;
                byte* qh = block + 4 + K_SCALE_SIZE;
                byte* ql = qh + QK_K / 8;
                float* y = dst + i * QK_K;
                int isIdx = 0;
                byte u1 = 1;
                byte u2 = 2;
                for (int j = 0; j < QK_K; j += 64)
                {
                    GetScaleMinK4(isIdx, scales, out byte sc1, out byte m1q);
                    GetScaleMinK4(isIdx + 1, scales, out byte sc2, out byte m2q);
                    float d1 = d * sc1;
                    float d2 = d * sc2;
                    float m1 = min * m1q;
                    float m2 = min * m2q;
                    for (int l = 0; l < 32; l++)
                        y[j + l] = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++)
                        y[j + l + 32] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
                    ql += 32;
                    isIdx += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }

        private static unsafe void DequantizeQ6K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q6_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                byte* ql = block;
                byte* qh = ql + QK_K / 2;
                sbyte* scales = (sbyte*)(qh + QK_K / 4);
                float d = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float* y = dst + i * QK_K;

                for (int n = 0; n < QK_K; n += 128)
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int isIdx = l / 16;
                        sbyte q1 = (sbyte)(((ql[l] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) - 32);
                        sbyte q2 = (sbyte)(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32);
                        sbyte q3 = (sbyte)(((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32);
                        sbyte q4 = (sbyte)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32);
                        y[n + l] = d * scales[isIdx] * q1;
                        y[n + l + 32] = d * scales[isIdx + 2] * q2;
                        y[n + l + 64] = d * scales[isIdx + 4] * q3;
                        y[n + l + 96] = d * scales[isIdx + 6] * q4;
                    }

                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
        }

        // Q2_K: 16 sub-blocks of 16. Ported verbatim from ggml dequantize_row_q2_K.
        // block layout: scales[16] | qs[64] | d(fp16) | dmin(fp16) = 84 bytes.
        private static unsafe void DequantizeQ2K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q2_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = QK_K / 16 + QK_K / 4 + 2 + 2;   // 16 + 64 + 2 + 2 = 84
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                byte* scales = block;                 // [16]
                byte* q = block + QK_K / 16;           // qs [64]
                float d = HalfToSingle(ReadUInt16(block + QK_K / 16 + QK_K / 4));       // +80
                float min = HalfToSingle(ReadUInt16(block + QK_K / 16 + QK_K / 4 + 2)); // +82
                float* y = dst + i * QK_K;

                int si = 0;
                for (int n = 0; n < QK_K; n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        byte sc = scales[si++];
                        float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                        for (int l = 0; l < 16; ++l) *y++ = dl * ((sbyte)((q[l] >> shift) & 3)) - ml;

                        sc = scales[si++];
                        dl = d * (sc & 0xF); ml = min * (sc >> 4);
                        for (int l = 0; l < 16; ++l) *y++ = dl * ((sbyte)((q[l + 16] >> shift) & 3)) - ml;

                        shift += 2;
                    }
                    q += 32;
                }
            }
        }

        // Q3_K: 16 sub-blocks of 16, 6-bit scales packed in 12 bytes, high bit in hmask.
        // Ported verbatim from ggml dequantize_row_q3_K.
        // block layout: hmask[32] | qs[64] | scales[12] | d(fp16) = 110 bytes.
        private static unsafe void DequantizeQ3K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q3_K requires {QK_K}-element alignment, got {numElements}.");

            const uint kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
            int blockBytes = QK_K / 8 + QK_K / 4 + 12 + 2;   // 32 + 64 + 12 + 2 = 110
            int nb = (int)(numElements / QK_K);
            uint* aux = stackalloc uint[4];
            sbyte* scales = (sbyte*)aux;
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                byte* hm = block;                     // hmask [32]
                byte* q = block + QK_K / 8;           // qs [64]
                byte* sc = block + QK_K / 8 + QK_K / 4;   // scales [12]
                float dAll = HalfToSingle(ReadUInt16(sc + 12));   // d at +108

                aux[0] = ReadUInt32(sc + 0);
                aux[1] = ReadUInt32(sc + 4);
                aux[2] = ReadUInt32(sc + 8);
                uint tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

                float* y = dst + i * QK_K;
                int si = 0; byte m = 1; byte* qq = q;
                for (int n = 0; n < QK_K; n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        float dl = dAll * (scales[si++] - 32);
                        for (int l = 0; l < 16; ++l)
                            *y++ = dl * ((sbyte)((qq[l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4));

                        dl = dAll * (scales[si++] - 32);
                        for (int l = 0; l < 16; ++l)
                            *y++ = dl * ((sbyte)((qq[l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4));

                        shift += 2; m <<= 1;
                    }
                    qq += 32;
                }
            }
        }

        // IQ2_XXS: 2.0625 bpw codebook quant. Ported verbatim from ggml dequantize_row_iq2_xxs.
        // block layout: d(fp16) | qs[32] (uint16) = 66 bytes. grid/sign tables in IQuantGrids.
        private static unsafe void DequantizeIq2Xxs(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"IQ2_XXS requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 2 + (QK_K / 8) * 2;   // 2 + 32*2 = 66
            int nb = (int)(numElements / QK_K);
            fixed (ulong* grid = IQuantGrids.iq2xxs_grid)
            fixed (byte* ksigns = IQuantGrids.ksigns_iq2xs)
            fixed (byte* kmask = IQuantGrids.kmask_iq2xs)
            {
                uint* aux32 = stackalloc uint[2];
                byte* aux8 = (byte*)aux32;
                for (int i = 0; i < nb; i++)
                {
                    byte* block = src + i * blockBytes;
                    float d = HalfToSingle(ReadUInt16(block));
                    byte* qs = block + 2;
                    float* y = dst + i * QK_K;
                    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32)
                    {
                        byte* p = qs + 8 * ib32;   // 4 uint16 = 8 bytes per ib32
                        aux32[0] = ReadUInt32(p);
                        aux32[1] = ReadUInt32(p + 4);
                        float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
                        for (int l = 0; l < 4; ++l)
                        {
                            byte* g = (byte*)(grid + aux8[l]);
                            byte signs = ksigns[(aux32[1] >> (7 * l)) & 127];
                            for (int j = 0; j < 8; ++j)
                                y[j] = db * g[j] * ((signs & kmask[j]) != 0 ? -1f : 1f);
                            y += 8;
                        }
                    }
                }
            }
        }

        // IQ2_S: 2.5625 bpw codebook quant. Ported verbatim from ggml dequantize_row_iq2_s.
        // block layout: d(fp16) | qs[64] | qh[8] | scales[8] = 82 bytes; signs are qs[32..63].
        private static unsafe void DequantizeIq2S(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"IQ2_S requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32;   // 2 + 64 + 8 + 8 = 82
            int nb = (int)(numElements / QK_K);
            fixed (ulong* grid = IQuantGrids.iq2s_grid)
            fixed (byte* kmask = IQuantGrids.kmask_iq2xs)
            {
                for (int i = 0; i < nb; i++)
                {
                    byte* block = src + i * blockBytes;
                    float d = HalfToSingle(ReadUInt16(block));
                    byte* qs = block + 2;                  // [64]
                    byte* qh = qs + QK_K / 4;              // [8]
                    byte* scales = qh + QK_K / 32;         // [8]
                    byte* signs = qs + QK_K / 8;           // qs + 32
                    float* y = dst + i * QK_K;
                    byte* qsp = qs, signsp = signs;
                    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32)
                    {
                        float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
                        float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                        for (int l = 0; l < 4; ++l)
                        {
                            float dl = l < 2 ? db0 : db1;
                            int idx = qsp[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                            byte* g = (byte*)(grid + idx);
                            byte sgn = signsp[l];
                            for (int j = 0; j < 8; ++j)
                                y[j] = dl * g[j] * ((sgn & kmask[j]) != 0 ? -1f : 1f);
                            y += 8;
                        }
                        qsp += 4; signsp += 4;
                    }
                }
            }
        }

        // IQ3_S: codebook quant, 4-byte grid entries. Ported verbatim from ggml dequantize_row_iq3_s.
        // block layout: d(fp16) | qs[64] | qh[8] | signs[32] | scales[4] = 110 bytes.
        private static unsafe void DequantizeIq3S(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"IQ3_S requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64;   // 2+64+8+32+4 = 110
            int nb = (int)(numElements / QK_K);
            fixed (uint* grid = IQuantGrids.iq3s_grid)
            fixed (byte* kmask = IQuantGrids.kmask_iq2xs)
            {
                for (int i = 0; i < nb; i++)
                {
                    byte* block = src + i * blockBytes;
                    float d = HalfToSingle(ReadUInt16(block));
                    byte* qs = block + 2;                  // [64]
                    byte* qh = qs + QK_K / 4;              // [8]
                    byte* signs = qh + QK_K / 32;          // [32]
                    byte* scales = signs + QK_K / 8;       // [4]
                    float* y = dst + i * QK_K;
                    byte* qsp = qs, qhp = qh, signsp = signs;
                    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2)
                    {
                        float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
                        float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
                        for (int l = 0; l < 4; ++l)
                        {
                            byte* g1 = (byte*)(grid + (qsp[2 * l + 0] | ((qhp[0] << (8 - 2 * l)) & 256)));
                            byte* g2 = (byte*)(grid + (qsp[2 * l + 1] | ((qhp[0] << (7 - 2 * l)) & 256)));
                            byte sgn = signsp[l];
                            for (int j = 0; j < 4; ++j)
                            {
                                y[j + 0] = db1 * g1[j] * ((sgn & kmask[j + 0]) != 0 ? -1f : 1f);
                                y[j + 4] = db1 * g2[j] * ((sgn & kmask[j + 4]) != 0 ? -1f : 1f);
                            }
                            y += 8;
                        }
                        qsp += 8; signsp += 4;
                        for (int l = 0; l < 4; ++l)
                        {
                            byte* g1 = (byte*)(grid + (qsp[2 * l + 0] | ((qhp[1] << (8 - 2 * l)) & 256)));
                            byte* g2 = (byte*)(grid + (qsp[2 * l + 1] | ((qhp[1] << (7 - 2 * l)) & 256)));
                            byte sgn = signsp[l];
                            for (int j = 0; j < 4; ++j)
                            {
                                y[j + 0] = db2 * g1[j] * ((sgn & kmask[j + 0]) != 0 ? -1f : 1f);
                                y[j + 4] = db2 * g2[j] * ((sgn & kmask[j + 4]) != 0 ? -1f : 1f);
                            }
                            y += 8;
                        }
                        qhp += 2; qsp += 8; signsp += 4;
                    }
                }
            }
        }

        private static unsafe void DequantizeIq4Nl(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_NL != 0)
                throw new NotSupportedException($"IQ4_NL requires {QK4_NL}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK4_NL / 2;
            int nb = (int)(numElements / QK4_NL);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                byte* qs = block + 2;
                float* y = dst + i * QK4_NL;
                for (int j = 0; j < QK4_NL / 2; j++)
                {
                    y[j] = d * Iq4NlValues[qs[j] & 0x0F];
                    y[j + QK4_NL / 2] = d * Iq4NlValues[qs[j] >> 4];
                }
            }
        }

        private static unsafe void DequantizeMxfp4(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_MXFP4 != 0)
                throw new NotSupportedException($"MXFP4 requires {QK_MXFP4}-element alignment, got {numElements}.");

            int blockBytes = 1 + QK_MXFP4 / 2;
            int nb = (int)(numElements / QK_MXFP4);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = E8M0ToFp32Half(block[0]);
                byte* qs = block + 1;
                float* y = dst + i * QK_MXFP4;
                for (int j = 0; j < QK_MXFP4 / 2; j++)
                {
                    y[j] = d * Mxfp4Values[qs[j] & 0x0F];
                    y[j + QK_MXFP4 / 2] = d * Mxfp4Values[qs[j] >> 4];
                }
            }
        }

        private static unsafe void QuantizeF32ToQ8_0(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK8_0;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK8_0;
                byte* blockDst = dst + block * Q8_0BlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK8_0);
                float scale = maxAbs / 127.0f;
                WriteHalf(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 2);
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK8_0);
                    continue;
                }

                float invScale = 1.0f / scale;
                for (int i = 0; i < QK8_0; i++)
                    qs[i] = ClampToInt8(MathF.Round(blockSrc[i] * invScale));
            }
        }

        // Mirror of ggml's quantize_row_q4_0_ref: per 32-element block, d = max/-8
        // (max = the element with the largest magnitude, sign included), stored as
        // fp16, then 4-bit quants qi = MIN(15, (int)(x/d + 8.5)). Low nibble holds
        // element j, high nibble holds element j+16 (matches DequantizeQ40 above).
        private static unsafe void QuantizeF32ToQ4_0(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK4_0;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK4_0;
                byte* blockDst = dst + block * Q4_0BlockBytes;

                float amax = 0.0f, max = 0.0f;
                for (int j = 0; j < QK4_0; j++)
                {
                    float v = blockSrc[j];
                    float av = MathF.Abs(v);
                    if (av > amax) { amax = av; max = v; }
                }

                float d = max / -8.0f;
                WriteHalf(blockDst, d);

                byte* qs = blockDst + 2;
                if (d == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK4_0 / 2);
                    continue;
                }

                float id = 1.0f / d;
                for (int j = 0; j < QK4_0 / 2; j++)
                {
                    float x0 = blockSrc[j] * id;
                    float x1 = blockSrc[j + QK4_0 / 2] * id;
                    int xi0 = Math.Min(15, (int)(x0 + 8.5f));
                    int xi1 = Math.Min(15, (int)(x1 + 8.5f));
                    qs[j] = (byte)(xi0 | (xi1 << 4));
                }
            }
        }

        private static unsafe void QuantizeF32ToQ8_1(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK8_1;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK8_1;
                byte* blockDst = dst + block * Q8_1BlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK8_1);
                float scale = maxAbs / 127.0f;
                WriteHalf(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 4);
                int sum = 0;
                if (scale != 0.0f)
                {
                    float invScale = 1.0f / scale;
                    for (int i = 0; i < QK8_1; i++)
                    {
                        sbyte q = ClampToInt8(MathF.Round(blockSrc[i] * invScale));
                        qs[i] = q;
                        sum += q;
                    }
                }
                else
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK8_1);
                }

                WriteHalf(blockDst + 2, scale * sum);
            }
        }

        private static unsafe void QuantizeF32ToQ8_K(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK_K;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK_K;
                byte* blockDst = dst + block * Q8_KBlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK_K);
                float scale = maxAbs / 127.0f;
                Unsafe.WriteUnaligned(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 4);
                short* bsums = (short*)(blockDst + 4 + QK_K);
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK_K);
                    Unsafe.InitBlockUnaligned(bsums, 0, QK_K / 16 * sizeof(short));
                    continue;
                }

                float invScale = 1.0f / scale;
                for (int group = 0; group < QK_K / 16; group++)
                {
                    int sum = 0;
                    int offset = group * 16;
                    for (int i = 0; i < 16; i++)
                    {
                        sbyte q = ClampToInt8(MathF.Round(blockSrc[offset + i] * invScale));
                        qs[offset + i] = q;
                        sum += q;
                    }

                    bsums[group] = (short)sum;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void WriteHalf(byte* dst, float value)
        {
            Unsafe.WriteUnaligned(dst, BitConverter.HalfToUInt16Bits((System.Half)value));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static sbyte ClampToInt8(float value)
        {
            int rounded = (int)value;
            if (rounded > 127) return 127;
            if (rounded < -127) return -127;
            return (sbyte)rounded;
        }

        private static unsafe float MaxAbs(float* src, int length)
        {
            if (Avx512F.IsSupported && length >= 16)
            {
                Vector512<float> max = Vector512<float>.Zero;
                int i = 0;
                for (; i <= length - 16; i += 16)
                    max = Avx512F.Max(max, Vector512.Abs(Avx512F.LoadVector512(src + i)));

                float result = HorizontalMax(max);
                for (; i < length; i++)
                {
                    float abs = MathF.Abs(src[i]);
                    if (abs > result) result = abs;
                }

                return result;
            }

            int vectorSize = Vector<float>.Count;
            Vector<float> maxVec = Vector<float>.Zero;
            int j = 0;
            for (; j <= length - vectorSize; j += vectorSize)
                maxVec = Vector.Max(maxVec, Vector.Abs(LoadVec(src + j)));

            float maxAbs = 0.0f;
            for (int lane = 0; lane < Vector<float>.Count; lane++)
                if (maxVec[lane] > maxAbs) maxAbs = maxVec[lane];

            for (; j < length; j++)
            {
                float abs = MathF.Abs(src[j]);
                if (abs > maxAbs) maxAbs = abs;
            }

            return maxAbs;
        }


        private static unsafe float VecDotQ4_0Q8_0(byte* q4, byte* q8, int blockCount)
        {
            if (Avx512F.IsSupported && Avx512BW.IsSupported)
                return VecDotQ4_0Q8_0Avx512(q4, q8, blockCount);
            if (Avx2.IsSupported)
                return VecDotQ4_0Q8_0Avx2(q4, q8, blockCount);

            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q4Block = q4 + block * Q4_0BlockBytes;
                byte* q8Block = q8 + block * Q8_0BlockBytes;
                float d4 = HalfToSingle(ReadUInt16(q4Block));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                byte* qs = q4Block + 2;
                sbyte* qx = (sbyte*)(q8Block + 2);

                int isum = 0;
                for (int i = 0; i < QK4_0 / 2; i++)
                {
                    int low = (qs[i] & 0x0F) - 8;
                    int high = (qs[i] >> 4) - 8;
                    isum += low * qx[i] + high * qx[i + QK4_0 / 2];
                }

                sum += d4 * d8 * isum;
            }

            return sum;
        }

        // Unpack a Q4_0 block's 16 packed nibble bytes into 32 signed bytes in the
        // ggml dequant order [low0..low15, high0..high15] (matching the Q8_0
        // activation layout qx[0..31]), with the -8 zero-point offset already
        // applied so the result is the signed weight value.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<sbyte> UnpackQ40Nibbles(byte* qs, Vector256<sbyte> offset8)
        {
            Vector128<byte> packed = Unsafe.ReadUnaligned<Vector128<byte>>(qs);
            Vector128<byte> mask = Vector128.Create((byte)0x0F);
            Vector128<byte> low = Sse2.And(packed, mask);
            Vector128<byte> high = Sse2.And(Sse2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), mask);
            return Avx2.Subtract(Vector256.Create(low, high).AsSByte(), offset8);
        }

        private static unsafe float VecDotQ4_0Q8_0Avx512(byte* q4, byte* q8, int blockCount)
        {
            // Two independent FMA accumulators break the loop-carried dependency
            // on `acc` so the int8 widen/madd pipeline isn't stalled on FMA
            // latency; this lifts the Q4_0 matmul closer to the memory wall.
            Vector512<float> acc0 = Vector512<float>.Zero;
            Vector512<float> acc1 = Vector512<float>.Zero;
            Vector512<short> ones = Vector512.Create((short)1);
            Vector256<sbyte> offset8 = Vector256.Create((sbyte)8);

            int block = 0;
            int pairEnd = blockCount & ~1;
            for (; block < pairEnd; block += 2)
            {
                acc0 = AccumQ40BlockAvx512(q4, q8, block, ones, offset8, acc0);
                acc1 = AccumQ40BlockAvx512(q4, q8, block + 1, ones, offset8, acc1);
            }
            if (block < blockCount)
                acc0 = AccumQ40BlockAvx512(q4, q8, block, ones, offset8, acc0);

            return HorizontalSum(acc0 + acc1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float> AccumQ40BlockAvx512(
            byte* q4, byte* q8, int block, Vector512<short> ones, Vector256<sbyte> offset8, Vector512<float> acc)
        {
            byte* wb = q4 + block * Q4_0BlockBytes;
            byte* xb = q8 + block * Q8_0BlockBytes;
            float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

            Vector256<sbyte> qwBytes = UnpackQ40Nibbles(wb + 2, offset8);
            Vector256<sbyte> qxBytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
            Vector512<short> qw = Avx512BW.ConvertToVector512Int16(qwBytes);
            Vector512<short> qx = Avx512BW.ConvertToVector512Int16(qxBytes);
            Vector512<short> products = Avx512BW.MultiplyLow(qw, qx);
            Vector512<int> pairSums = Avx512BW.MultiplyAddAdjacent(products, ones);
            Vector512<float> dotParts = Avx512F.ConvertToVector512Single(pairSums);
            return Avx512F.FusedMultiplyAdd(Vector512.Create(scale), dotParts, acc);
        }

        private static unsafe float VecDotQ4_0Q8_0Avx2(byte* q4, byte* q8, int blockCount)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<short> ones = Vector256.Create((short)1);
            Vector256<sbyte> offset8 = Vector256.Create((sbyte)8);

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q4 + block * Q4_0BlockBytes;
                byte* xb = q8 + block * Q8_0BlockBytes;
                float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

                Vector256<sbyte> qw = UnpackQ40Nibbles(wb + 2, offset8);
                Vector256<sbyte> qx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
                // signed*signed dot via maddubs(|w|, sign(w)*x): see VecDotQ8_0Q8_0Avx2.
                Vector256<sbyte> absW = Avx2.Sign(qw, qw);
                Vector256<sbyte> signedX = Avx2.Sign(qx, qw);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absW.AsByte(), signedX);
                Vector256<int> pairSums = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> dotParts = Avx.ConvertToVector256Single(pairSums);
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(Vector256.Create(scale), dotParts, acc)
                    : Avx.Add(acc, Avx.Multiply(Vector256.Create(scale), dotParts));
            }

            return HorizontalSum(acc);
        }

        private static unsafe float VecDotQ4_1Q8_1(byte* q4, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q4Block = q4 + block * Q4_1BlockBytes;
                byte* q8Block = q8 + block * Q8_1BlockBytes;
                float d4 = HalfToSingle(ReadUInt16(q4Block));
                float m4 = HalfToSingle(ReadUInt16(q4Block + 2));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                float s8 = HalfToSingle(ReadUInt16(q8Block + 2));
                byte* qs = q4Block + 4;
                sbyte* qx = (sbyte*)(q8Block + 4);

                int isum = 0;
                for (int i = 0; i < QK4_1 / 2; i++)
                    isum += (qs[i] & 0x0F) * qx[i] + (qs[i] >> 4) * qx[i + QK4_1 / 2];

                sum += d4 * d8 * isum + m4 * s8;
            }

            return sum;
        }

        private static unsafe float VecDotQ5_0Q8_0(byte* q5, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q5Block = q5 + block * Q5_0BlockBytes;
                byte* q8Block = q8 + block * Q8_0BlockBytes;
                float d5 = HalfToSingle(ReadUInt16(q5Block));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                uint qh = ReadUInt32(q5Block + 2);
                byte* qs = q5Block + 6;
                sbyte* qx = (sbyte*)(q8Block + 2);

                int isum = 0;
                for (int i = 0; i < QK5_0 / 2; i++)
                {
                    int xh0 = (int)(((qh >> i) << 4) & 0x10);
                    int xh1 = (int)((qh >> (i + 12)) & 0x10);
                    int x0 = ((qs[i] & 0x0F) | xh0) - 16;
                    int x1 = ((qs[i] >> 4) | xh1) - 16;
                    isum += x0 * qx[i] + x1 * qx[i + QK5_0 / 2];
                }

                sum += d5 * d8 * isum;
            }

            return sum;
        }

        private static unsafe float VecDotQ5_1Q8_1(byte* q5, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q5Block = q5 + block * Q5_1BlockBytes;
                byte* q8Block = q8 + block * Q8_1BlockBytes;
                float d5 = HalfToSingle(ReadUInt16(q5Block));
                float m5 = HalfToSingle(ReadUInt16(q5Block + 2));
                uint qh = ReadUInt32(q5Block + 4);
                byte* qs = q5Block + 8;
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                float s8 = HalfToSingle(ReadUInt16(q8Block + 2));
                sbyte* qx = (sbyte*)(q8Block + 4);

                int isum = 0;
                for (int i = 0; i < QK5_1 / 2; i++)
                {
                    int xh0 = (int)(((qh >> i) << 4) & 0x10);
                    int xh1 = (int)((qh >> (i + 12)) & 0x10);
                    int x0 = (qs[i] & 0x0F) | xh0;
                    int x1 = (qs[i] >> 4) | xh1;
                    isum += x0 * qx[i] + x1 * qx[i + QK5_1 / 2];
                }

                sum += d5 * d8 * isum + m5 * s8;
            }

            return sum;
        }

        private static unsafe float VecDotQ8_0Q8_0(byte* q8w, byte* q8x, int blockCount)
        {
            if (Avx512F.IsSupported && Avx512BW.IsSupported)
                return VecDotQ8_0Q8_0Avx512(q8w, q8x, blockCount);
            if (Avx2.IsSupported)
                return VecDotQ8_0Q8_0Avx2(q8w, q8x, blockCount);

            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float dw = HalfToSingle(ReadUInt16(wb));
                float dx = HalfToSingle(ReadUInt16(xb));
                sbyte* qw = (sbyte*)(wb + 2);
                sbyte* qx = (sbyte*)(xb + 2);

                int isum = 0;
                for (int i = 0; i < QK8_0; i++)
                    isum += qw[i] * qx[i];
                sum += dw * dx * isum;
            }

            return sum;
        }

        private static unsafe float VecDotQ8_1Q8_0(byte* q8w, byte* q8x, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_1BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float dw = HalfToSingle(ReadUInt16(wb));
                float dx = HalfToSingle(ReadUInt16(xb));
                sbyte* qw = (sbyte*)(wb + 4);
                sbyte* qx = (sbyte*)(xb + 2);

                int isum = 0;
                for (int i = 0; i < QK8_1; i++)
                    isum += qw[i] * qx[i];
                sum += dw * dx * isum;
            }

            return sum;
        }

        private static unsafe float VecDotQ8_0Q8_0Avx512(byte* q8w, byte* q8x, int blockCount)
        {
            Vector512<float> acc = Vector512<float>.Zero;
            Vector512<short> ones = Vector512.Create((short)1);

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

                Vector256<sbyte> qwBytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb + 2);
                Vector256<sbyte> qxBytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
                Vector512<short> qw = Avx512BW.ConvertToVector512Int16(qwBytes);
                Vector512<short> qx = Avx512BW.ConvertToVector512Int16(qxBytes);
                Vector512<short> products = Avx512BW.MultiplyLow(qw, qx);
                Vector512<int> pairSums = Avx512BW.MultiplyAddAdjacent(products, ones);
                Vector512<float> dotParts = Avx512F.ConvertToVector512Single(pairSums);

                acc = Avx512F.FusedMultiplyAdd(Vector512.Create(scale), dotParts, acc);
            }

            return HorizontalSum(acc);
        }

        private static unsafe float VecDotQ8_0Q8_0Avx2(byte* q8w, byte* q8x, int blockCount)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<short> ones = Vector256.Create((short)1);

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

                Vector256<sbyte> qw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb + 2);
                Vector256<sbyte> qx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
                Vector256<sbyte> absW = Avx2.Sign(qw, qw);
                Vector256<sbyte> signedX = Avx2.Sign(qx, qw);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absW.AsByte(), signedX);
                Vector256<int> pairSums = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> dotParts = Avx.ConvertToVector256Single(pairSums);
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(Vector256.Create(scale), dotParts, acc)
                    : Avx.Add(acc, Avx.Multiply(Vector256.Create(scale), dotParts));
            }

            return HorizontalSum(acc);
        }

        private static unsafe float VecDotQ4_KQ8_K(byte* q4k, byte* q8k, int superBlockCount)
        {
            if (Avx2.IsSupported)
                return VecDotQ4_KQ8_KAvx2(q4k, q8k, superBlockCount);

            return VecDotQ4_KQ8_KScalar(q4k, q8k, superBlockCount);
        }

        // AVX2 Q4_K x Q8_K. The 8 sub-block dots are kept scaled in vector lanes
        // (maddubs to int16, then madd against the broadcast 6-bit scale) so the
        // whole super-block needs a single horizontal sum. The K-quant min term
        // (dmin * sum_j min_j * q8_bsum_j) is folded into a scalar correction off
        // the Q8_K bsums — the same factoring llama.cpp uses.
        private static unsafe float VecDotQ4_KQ8_KAvx2(byte* q4k, byte* q8k, int superBlockCount)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<byte> loMask = Vector256.Create((byte)0x0F);
            float minTotal = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d4 = HalfToSingle(ReadUInt16(q4k));
                float dmin = HalfToSingle(ReadUInt16(q4k + 2));
                UnpackQ4Q5Scales(q4k + 4, scBuf, mnBuf);
                byte* qs = q4k + 16;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                Vector256<int> sumi = Vector256<int>.Zero;
                for (int p = 0; p < 4; p++)
                {
                    Vector256<byte> q4bits = Unsafe.ReadUnaligned<Vector256<byte>>(qs + p * 32);
                    Vector256<byte> low = Avx2.And(q4bits, loMask);
                    Vector256<byte> high = Avx2.And(Avx2.ShiftRightLogical(q4bits.AsUInt16(), 4).AsByte(), loMask);
                    Vector256<sbyte> q8lo = Unsafe.ReadUnaligned<Vector256<sbyte>>((byte*)(q8Values + (2 * p) * 32));
                    Vector256<sbyte> q8hi = Unsafe.ReadUnaligned<Vector256<sbyte>>((byte*)(q8Values + (2 * p + 1) * 32));
                    Vector256<short> p16lo = Avx2.MultiplyAddAdjacent(low, q8lo);
                    Vector256<short> p16hi = Avx2.MultiplyAddAdjacent(high, q8hi);
                    sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(p16lo, Vector256.Create((short)scBuf[2 * p])));
                    sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(p16hi, Vector256.Create((short)scBuf[2 * p + 1])));
                }

                int msum = 0;
                for (int j = 0; j < 8; j++)
                    msum += mnBuf[j] * (bsums[2 * j] + bsums[2 * j + 1]);
                minTotal += d8 * dmin * msum;

                float scale = d4 * d8;
                Vector256<float> prod = Avx.ConvertToVector256Single(sumi);
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(Vector256.Create(scale), prod, acc)
                    : Avx.Add(acc, Avx.Multiply(Vector256.Create(scale), prod));

                q4k += Q4_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return HorizontalSum(acc) - minTotal;
        }

        private static unsafe float VecDotQ4_KQ8_KScalar(byte* q4k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d4 = HalfToSingle(ReadUInt16(q4k));
                float dmin = HalfToSingle(ReadUInt16(q4k + 2));
                UnpackQ4Q5Scales(q4k + 4, scBuf, mnBuf);
                byte* qs = q4k + 16;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                for (int j = 0; j < 8; j++)
                {
                    int pairIndex = j / 2;
                    bool highNibble = (j & 1) != 0;
                    sbyte* q8Vals = q8Values + j * 32;
                    int prodSum = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int raw = qs[pairIndex * 32 + i];
                        int q = highNibble ? raw >> 4 : raw & 0x0F;
                        prodSum += q * q8Vals[i];
                    }

                    int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                    sum += d8 * (d4 * scBuf[j] * prodSum - dmin * mnBuf[j] * q8Sum);
                }

                q4k += Q4_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        private static unsafe float VecDotQ5_KQ8_K(byte* q5k, byte* q8k, int superBlockCount)
        {
            if (Avx2.IsSupported)
                return VecDotQ5_KQ8_KAvx2(q5k, q8k, superBlockCount);

            return VecDotQ5_KQ8_KScalar(q5k, q8k, superBlockCount);
        }

        // AVX2 Q5_K x Q8_K. Same lane-scaled accumulation as Q4_K, with the 5th
        // bit pulled from qh: for sub-block j the high bit is qh[i] bit j, so the
        // low/high nibble of each qs byte (sub-blocks 2p / 2p+1) gets bit 2p / 2p+1
        // of qh added at weight 16 before the maddubs.
        private static unsafe float VecDotQ5_KQ8_KAvx2(byte* q5k, byte* q8k, int superBlockCount)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<byte> loMask = Vector256.Create((byte)0x0F);
            Vector256<byte> oneByte = Vector256.Create((byte)1);
            float minTotal = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d5 = HalfToSingle(ReadUInt16(q5k));
                float dmin = HalfToSingle(ReadUInt16(q5k + 2));
                UnpackQ4Q5Scales(q5k + 4, scBuf, mnBuf);
                byte* qh = q5k + 16;
                byte* qs = q5k + 48;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                Vector256<byte> qhbits = Unsafe.ReadUnaligned<Vector256<byte>>(qh);

                Vector256<int> sumi = Vector256<int>.Zero;
                for (int p = 0; p < 4; p++)
                {
                    Vector256<byte> q4bits = Unsafe.ReadUnaligned<Vector256<byte>>(qs + p * 32);
                    Vector256<byte> low = Avx2.And(q4bits, loMask);
                    Vector256<byte> high = Avx2.And(Avx2.ShiftRightLogical(q4bits.AsUInt16(), 4).AsByte(), loMask);

                    // bit (2p) and (2p+1) of each qh byte -> 0/1, shifted to weight 16.
                    Vector256<byte> hbitLo = Avx2.And(Avx2.ShiftRightLogical(qhbits.AsUInt16(), (byte)(2 * p)).AsByte(), oneByte);
                    Vector256<byte> hbitHi = Avx2.And(Avx2.ShiftRightLogical(qhbits.AsUInt16(), (byte)(2 * p + 1)).AsByte(), oneByte);
                    low = Avx2.Add(low, Avx2.ShiftLeftLogical(hbitLo.AsUInt16(), 4).AsByte());
                    high = Avx2.Add(high, Avx2.ShiftLeftLogical(hbitHi.AsUInt16(), 4).AsByte());

                    Vector256<sbyte> q8lo = Unsafe.ReadUnaligned<Vector256<sbyte>>((byte*)(q8Values + (2 * p) * 32));
                    Vector256<sbyte> q8hi = Unsafe.ReadUnaligned<Vector256<sbyte>>((byte*)(q8Values + (2 * p + 1) * 32));
                    Vector256<short> p16lo = Avx2.MultiplyAddAdjacent(low, q8lo);
                    Vector256<short> p16hi = Avx2.MultiplyAddAdjacent(high, q8hi);
                    sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(p16lo, Vector256.Create((short)scBuf[2 * p])));
                    sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(p16hi, Vector256.Create((short)scBuf[2 * p + 1])));
                }

                int msum = 0;
                for (int j = 0; j < 8; j++)
                    msum += mnBuf[j] * (bsums[2 * j] + bsums[2 * j + 1]);
                minTotal += d8 * dmin * msum;

                float scale = d5 * d8;
                Vector256<float> prod = Avx.ConvertToVector256Single(sumi);
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(Vector256.Create(scale), prod, acc)
                    : Avx.Add(acc, Avx.Multiply(Vector256.Create(scale), prod));

                q5k += Q5_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return HorizontalSum(acc) - minTotal;
        }

        private static unsafe float VecDotQ5_KQ8_KScalar(byte* q5k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d5 = HalfToSingle(ReadUInt16(q5k));
                float dmin = HalfToSingle(ReadUInt16(q5k + 2));
                UnpackQ4Q5Scales(q5k + 4, scBuf, mnBuf);
                byte* qh = q5k + 16;
                byte* qs = q5k + 48;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                for (int j = 0; j < 8; j++)
                {
                    int pairIndex = j / 2;
                    bool highNibble = (j & 1) != 0;
                    sbyte* q8Vals = q8Values + j * 32;
                    int prodSum = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int raw = qs[pairIndex * 32 + i];
                        int lo4 = highNibble ? raw >> 4 : raw & 0x0F;
                        int bit5 = (qh[i] >> j) & 1;
                        prodSum += (lo4 | (bit5 << 4)) * q8Vals[i];
                    }

                    int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                    sum += d8 * (d5 * scBuf[j] * prodSum - dmin * mnBuf[j] * q8Sum);
                }

                q5k += Q5_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        private static unsafe float VecDotQ6_KQ8_K(byte* q6k, byte* q8k, int superBlockCount)
        {
            if (Avx2.IsSupported)
                return VecDotQ6_KQ8_KAvx2(q6k, q8k, superBlockCount);
            if (Ssse3.IsSupported)
                return VecDotQ6_KQ8_KSse(q6k, q8k, superBlockCount);

            return VecDotQ6_KQ8_KScalar(q6k, q8k, superBlockCount);
        }

        // AVX2 Q6_K x Q8_K. Consecutive sub-block pairs (2m, 2m+1) read
        // contiguous 32-byte ql / qh / q8 spans (only their qh offset differs),
        // so each pair is one 256-bit maddubs over the unsigned 0..63
        // reconstruction; the two halves carry the two sub-blocks' int8 scales in
        // the low/high 128-bit lanes of the madd multiplier. The -32 zero-point
        // is a scalar correction off the Q8_K per-16 bsums (see the SSE variant).
        private static unsafe float VecDotQ6_KQ8_KAvx2(byte* q6k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            Vector256<byte> loMask = Vector256.Create((byte)0x0F);
            Vector256<byte> hi2Mask = Vector256.Create((byte)0x03);

            for (int block = 0; block < superBlockCount; block++)
            {
                byte* ql = q6k;
                byte* qh = q6k + QK_K / 2;
                sbyte* scales = (sbyte*)(q6k + QK_K / 2 + QK_K / 4);
                float d6 = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                Vector256<int> sumi = Vector256<int>.Zero;
                for (int pair = 0; pair < 8; pair++)
                {
                    int half = pair / 4;
                    int pm = pair % 4;
                    int qlOff = half * 64 + (pm % 2) * 32;
                    bool isUpper = pm >= 2;
                    int qhOff = half * 32;
                    int qhShift = pm * 2;

                    Vector256<byte> ql32 = Unsafe.ReadUnaligned<Vector256<byte>>(ql + qlOff);
                    Vector256<byte> lo4 = isUpper
                        ? Avx2.And(Avx2.ShiftRightLogical(ql32.AsUInt16(), 4).AsByte(), loMask)
                        : Avx2.And(ql32, loMask);

                    Vector256<byte> qh32 = Unsafe.ReadUnaligned<Vector256<byte>>(qh + qhOff);
                    Vector256<byte> hi2 = Avx2.And(Avx2.ShiftRightLogical(qh32.AsUInt16(), (byte)qhShift).AsByte(), hi2Mask);
                    Vector256<byte> qval = Avx2.Add(lo4, Avx2.ShiftLeftLogical(hi2.AsUInt16(), 4).AsByte());

                    Vector256<sbyte> q8v = Unsafe.ReadUnaligned<Vector256<sbyte>>((byte*)(q8Values + pair * 32));
                    Vector256<short> p16 = Avx2.MultiplyAddAdjacent(qval, q8v);

                    // Low 128-bit lanes weight sub-block 2*pair, high lanes 2*pair+1.
                    Vector256<short> scaleVec = Vector256.Create(
                        Vector128.Create((short)scales[2 * pair]),
                        Vector128.Create((short)scales[2 * pair + 1]));
                    sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(p16, scaleVec));
                }

                int corr = 0;
                for (int sub = 0; sub < 16; sub++)
                    corr += scales[sub] * bsums[sub];

                float scaleBase = d6 * d8;
                sum += scaleBase * (HorizontalSum128(Sse2.Add(sumi.GetLower(), sumi.GetUpper())) - 32 * corr);

                q6k += Q6_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        // SSSE3 Q6_K x Q8_K. Q6_K has 16 sub-blocks of 16 elements with per-16
        // int8 scales. The inner 16-element dot is done with one 128-bit maddubs
        // on the unsigned 0..63 reconstruction (low nibble | high 2 bits << 4),
        // kept scaled in lanes (madd against the broadcast scale). The Q6_K -32
        // zero-point becomes a scalar correction off the Q8_K per-16 bsums:
        //   sum_sub scale_sub*(q6-32)*q8 = sum_sub scale_sub*q6unsigned*q8 - 32*sum_sub scale_sub*bsum_sub.
        private static unsafe float VecDotQ6_KQ8_KSse(byte* q6k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            Vector128<byte> loMask = Vector128.Create((byte)0x0F);
            Vector128<byte> hi2Mask = Vector128.Create((byte)0x03);

            for (int block = 0; block < superBlockCount; block++)
            {
                byte* ql = q6k;
                byte* qh = q6k + QK_K / 2;
                sbyte* scales = (sbyte*)(q6k + QK_K / 2 + QK_K / 4);
                float d6 = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                Vector128<int> sumi = Vector128<int>.Zero;
                int corr = 0;
                for (int sub = 0; sub < 16; sub++)
                {
                    int half = sub / 8;
                    int sh = sub % 8;
                    int qlOffset = half * 64 + (sh % 4) * 16;
                    bool isUpper = sh >= 4;
                    int qhOffset = half * 32 + (sh % 2) * 16;
                    int qhShift = (sh / 2) * 2;
                    int s = scales[sub];

                    Vector128<byte> qlBytes = Unsafe.ReadUnaligned<Vector128<byte>>(ql + qlOffset);
                    Vector128<byte> lo4 = isUpper
                        ? Sse2.And(Sse2.ShiftRightLogical(qlBytes.AsUInt16(), 4).AsByte(), loMask)
                        : Sse2.And(qlBytes, loMask);

                    Vector128<byte> qhBytes = Unsafe.ReadUnaligned<Vector128<byte>>(qh + qhOffset);
                    Vector128<byte> hi2 = Sse2.And(Sse2.ShiftRightLogical(qhBytes.AsUInt16(), (byte)qhShift).AsByte(), hi2Mask);
                    Vector128<byte> qval = Sse2.Add(lo4, Sse2.ShiftLeftLogical(hi2.AsUInt16(), 4).AsByte());

                    Vector128<sbyte> q8v = Unsafe.ReadUnaligned<Vector128<sbyte>>((byte*)(q8Values + sub * 16));
                    Vector128<short> p16 = Ssse3.MultiplyAddAdjacent(qval, q8v);
                    sumi = Sse2.Add(sumi, Sse2.MultiplyAddAdjacent(p16, Vector128.Create((short)s)));

                    corr += s * bsums[sub];
                }

                float scaleBase = d6 * d8;
                sum += scaleBase * (HorizontalSum128(sumi) - 32 * corr);

                q6k += Q6_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        private static unsafe float VecDotQ6_KQ8_KScalar(byte* q6k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;

            for (int block = 0; block < superBlockCount; block++)
            {
                byte* ql = q6k;
                byte* qh = q6k + QK_K / 2;
                sbyte* scales = (sbyte*)(q6k + QK_K / 2 + QK_K / 4);
                float d6 = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                float scaleBase = d6 * d8;

                for (int sub = 0; sub < 16; sub++)
                {
                    float scale = scaleBase * scales[sub];
                    sbyte* q8Vals = q8Values + sub * 16;
                    int half = sub / 8;
                    int sh = sub % 8;
                    int qlOffset = half * 64 + (sh % 4) * 16;
                    bool isUpper = sh >= 4;
                    int qhOffset = half * 32 + (sh % 2) * 16;
                    int qhShift = (sh / 2) * 2;

                    int isum = 0;
                    for (int i = 0; i < 16; i++)
                    {
                        int lo4 = isUpper ? (ql[qlOffset + i] >> 4) & 0x0F : ql[qlOffset + i] & 0x0F;
                        int hi2 = (qh[qhOffset + i] >> qhShift) & 0x03;
                        int q6 = (lo4 | (hi2 << 4)) - 32;
                        isum += q6 * q8Vals[i];
                    }

                    sum += scale * isum;
                }

                q6k += Q6_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        private static unsafe void UnpackQ4Q5Scales(byte* packed, byte* scales, byte* mins)
        {
            for (int i = 0; i < 8; i++)
                GetScaleMinK4(i, packed, out scales[i], out mins[i]);
        }

        // ------------------------------------------------------------------
        // IQ3_S x Q8_K (mirrors ggml_vec_dot_iq3_s_q8_K_generic)
        // ------------------------------------------------------------------

        // Per-byte sign expansion: bit j of the signs byte selects -1 (0xFF) or
        // +1 (0x01) for element j of an 8-element group.
        private static readonly ulong[] Iq3SignTab = BuildIq3SignTab();

        private static ulong[] BuildIq3SignTab()
        {
            var tab = new ulong[256];
            for (int b = 0; b < 256; b++)
            {
                ulong v = 0;
                for (int j = 0; j < 8; j++)
                {
                    byte lane = ((b >> j) & 1) != 0 ? (byte)0xFF : (byte)0x01;
                    v |= (ulong)lane << (8 * j);
                }
                tab[b] = v;
            }
            return tab;
        }

        private const int Iq3SBlockBytes = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64; // 110

        private static unsafe float VecDotIq3SQ8K(byte* iq3, byte* q8k, int superBlockCount)
        {
            if (Avx2.IsSupported)
                return VecDotIq3SQ8KAvx2(iq3, q8k, superBlockCount);
            return VecDotIq3SQ8KScalar(iq3, q8k, superBlockCount);
        }

        private static unsafe float VecDotIq3SQ8KAvx2(byte* iq3, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            fixed (uint* grid = IQuantGrids.iq3s_grid)
            fixed (ulong* signTab = Iq3SignTab)
            {
                for (int block = 0; block < superBlockCount; block++)
                {
                    byte* x = iq3 + block * Iq3SBlockBytes;
                    float d3 = HalfToSingle(ReadUInt16(x));
                    byte* qs = x + 2;                       // [64]
                    byte* qh = qs + QK_K / 4;               // [8]
                    byte* signs = qh + QK_K / 32;           // [32]
                    byte* scales = signs + QK_K / 8;        // [4]

                    byte* y = q8k + block * Q8_KBlockBytes;
                    float d8 = ReadSingle(y);
                    sbyte* q8 = (sbyte*)(y + 4);

                    Vector256<int> bsumVec = Vector256<int>.Zero;
                    for (int ib32 = 0; ib32 < QK_K / 32; ib32++)
                    {
                        int ls = 2 * ((ib32 & 1) == 0 ? scales[ib32 / 2] & 0xF : scales[ib32 / 2] >> 4) + 1;
                        int qhv = qh[ib32];

                        Vector256<int> acc = Vector256<int>.Zero;
                        for (int l = 0; l < 4; l += 2)
                        {
                            // 16 elements: groups l and l+1 (8 grid magnitudes each)
                            Vector128<byte> gridV = Vector128.Create(
                                grid[qs[2 * l + 0] | ((qhv << (8 - 2 * l)) & 256)],
                                grid[qs[2 * l + 1] | ((qhv << (7 - 2 * l)) & 256)],
                                grid[qs[2 * l + 2] | ((qhv << (8 - 2 * (l + 1))) & 256)],
                                grid[qs[2 * l + 3] | ((qhv << (7 - 2 * (l + 1))) & 256)]).AsByte();
                            Vector128<sbyte> signV = Vector128.Create(
                                signTab[signs[l]], signTab[signs[l + 1]]).AsSByte();

                            Vector256<short> g16 = Avx2.ConvertToVector256Int16(gridV);
                            Vector256<short> s16 = Avx2.ConvertToVector256Int16(signV);
                            Vector256<short> gs = Avx2.MultiplyLow(g16, s16);
                            Vector256<short> q16 = Avx2.ConvertToVector256Int16(
                                Unsafe.ReadUnaligned<Vector128<sbyte>>(q8 + 8 * l));
                            acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(gs, q16));
                        }
                        bsumVec = Avx2.Add(bsumVec, Avx2.MultiplyLow(acc, Vector256.Create(ls)));

                        qs += 8;
                        signs += 4;
                        q8 += 32;
                    }

                    int bsum = HorizontalSum128(Sse2.Add(bsumVec.GetLower(), bsumVec.GetUpper()));
                    sum += d3 * d8 * bsum;
                }
            }
            return sum;
        }

        private static unsafe float VecDotIq3SQ8KScalar(byte* iq3, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            fixed (uint* grid = IQuantGrids.iq3s_grid)
            fixed (byte* kmask = IQuantGrids.kmask_iq2xs)
            {
                for (int block = 0; block < superBlockCount; block++)
                {
                    byte* x = iq3 + block * Iq3SBlockBytes;
                    float d3 = HalfToSingle(ReadUInt16(x));
                    byte* qs = x + 2;
                    byte* qh = qs + QK_K / 4;
                    byte* signs = qh + QK_K / 32;
                    byte* scales = signs + QK_K / 8;

                    byte* y = q8k + block * Q8_KBlockBytes;
                    float d8 = ReadSingle(y);
                    sbyte* q8 = (sbyte*)(y + 4);

                    int bsum = 0;
                    for (int ib32 = 0; ib32 < QK_K / 32; ib32++)
                    {
                        int ls = 2 * ((ib32 & 1) == 0 ? scales[ib32 / 2] & 0xF : scales[ib32 / 2] >> 4) + 1;
                        int qhv = qh[ib32];
                        int sumi = 0;
                        for (int l = 0; l < 4; ++l)
                        {
                            byte* grid1 = (byte*)(grid + (qs[2 * l + 0] | ((qhv << (8 - 2 * l)) & 256)));
                            byte* grid2 = (byte*)(grid + (qs[2 * l + 1] | ((qhv << (7 - 2 * l)) & 256)));
                            byte sgn = signs[l];
                            for (int j = 0; j < 4; ++j)
                            {
                                sumi += grid1[j] * q8[j + 0] * ((sgn & kmask[j + 0]) != 0 ? -1 : 1);
                                sumi += grid2[j] * q8[j + 4] * ((sgn & kmask[j + 4]) != 0 ? -1 : 1);
                            }
                            q8 += 8;
                        }
                        bsum += sumi * ls;
                        qs += 8;
                        signs += 4;
                    }
                    sum += d3 * d8 * bsum;
                }
            }
            return sum;
        }

        // ------------------------------------------------------------------
        // MXFP4 x Q8_0 (mirrors ggml_vec_dot_mxfp4_q8_0_generic)
        // ------------------------------------------------------------------

        private const int Mxfp4BlockBytes = 1 + QK_MXFP4 / 2; // 17

        private static unsafe float VecDotMxfp4Q8_0(byte* mx, byte* q8, int blockCount)
        {
            if (Avx2.IsSupported)
                return VecDotMxfp4Q8_0Avx2(mx, q8, blockCount);

            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* xb = mx + block * Mxfp4BlockBytes;
                byte* yb = q8 + block * Q8_0BlockBytes;
                float d = HalfToSingle(ReadUInt16(yb)) * E8M0ToFp32Half(xb[0]);
                byte* qs = xb + 1;
                sbyte* q8v = (sbyte*)(yb + 2);

                int sumi1 = 0, sumi2 = 0;
                for (int j = 0; j < QK_MXFP4 / 2; ++j)
                {
                    sumi1 += q8v[j] * Mxfp4Values[qs[j] & 0xF];
                    sumi2 += q8v[j + QK_MXFP4 / 2] * Mxfp4Values[qs[j] >> 4];
                }
                sum += d * (sumi1 + sumi2);
            }
            return sum;
        }

        private static unsafe float VecDotMxfp4Q8_0Avx2(byte* mx, byte* q8, int blockCount)
        {
            Vector128<byte> loMask = Vector128.Create((byte)0x0F);
            Vector128<byte> table;
            fixed (sbyte* tbl = Mxfp4Values)
            {
                table = Unsafe.ReadUnaligned<Vector128<byte>>(tbl);
            }

            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* xb = mx + block * Mxfp4BlockBytes;
                byte* yb = q8 + block * Q8_0BlockBytes;
                float d = HalfToSingle(ReadUInt16(yb)) * E8M0ToFp32Half(xb[0]);

                Vector128<byte> qsBytes = Unsafe.ReadUnaligned<Vector128<byte>>(xb + 1);
                Vector128<byte> lo = Sse2.And(qsBytes, loMask);
                Vector128<byte> hi = Sse2.And(Sse2.ShiftRightLogical(qsBytes.AsUInt16(), 4).AsByte(), loMask);

                Vector128<sbyte> vlo = Ssse3.Shuffle(table, lo).AsSByte();
                Vector128<sbyte> vhi = Ssse3.Shuffle(table, hi).AsSByte();

                // lo nibbles are elements 0..15, hi nibbles elements 16..31
                Vector256<short> w16lo = Avx2.ConvertToVector256Int16(vlo);
                Vector256<short> w16hi = Avx2.ConvertToVector256Int16(vhi);
                Vector256<short> q16lo = Avx2.ConvertToVector256Int16(Unsafe.ReadUnaligned<Vector128<sbyte>>(yb + 2));
                Vector256<short> q16hi = Avx2.ConvertToVector256Int16(Unsafe.ReadUnaligned<Vector128<sbyte>>(yb + 2 + 16));

                Vector256<int> prod = Avx2.Add(
                    Avx2.MultiplyAddAdjacent(w16lo, q16lo),
                    Avx2.MultiplyAddAdjacent(w16hi, q16hi));
                int sumi = HorizontalSum128(Sse2.Add(prod.GetLower(), prod.GetUpper()));
                sum += d * sumi;
            }
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetDotChunkSize(GgmlTensorType type, long remaining)
        {
            return type switch
            {
                GgmlTensorType.F16 or GgmlTensorType.BF16 or
                GgmlTensorType.I8 or GgmlTensorType.I16 or GgmlTensorType.I32 or
                GgmlTensorType.I64 or GgmlTensorType.F64 => (int)Math.Min(remaining, QK_K),
                _ => (int)Math.Min(remaining, GgufFile.GetBlockSize(type)),
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetDotChunkBytes(GgmlTensorType type, int chunkElements)
        {
            return type switch
            {
                GgmlTensorType.F32 => chunkElements * sizeof(float),
                GgmlTensorType.F16 or GgmlTensorType.BF16 => chunkElements * sizeof(ushort),
                GgmlTensorType.I8 => chunkElements,
                GgmlTensorType.I16 => chunkElements * sizeof(short),
                GgmlTensorType.I32 => chunkElements * sizeof(int),
                GgmlTensorType.I64 or GgmlTensorType.F64 => chunkElements * sizeof(long),
                _ => (int)GgufFile.GetTypeSize(type),
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LoadVec(float* ptr) => Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)ptr);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotFloat(float* lhs, float* rhs, int length)
        {
            return TensorPrimitives.Dot(
                new ReadOnlySpan<float>(lhs, length),
                new ReadOnlySpan<float>(rhs, length));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ushort ReadUInt16(byte* p) => Unsafe.ReadUnaligned<ushort>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe uint ReadUInt32(byte* p) => Unsafe.ReadUnaligned<uint>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe int ReadInt32(byte* p) => Unsafe.ReadUnaligned<int>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ReadInt64(byte* p) => Unsafe.ReadUnaligned<long>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe double ReadDouble(byte* p) => Unsafe.ReadUnaligned<double>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float ReadSingle(byte* p) => Unsafe.ReadUnaligned<float>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HalfToSingle(ushort value) => (float)BitConverter.UInt16BitsToHalf(value);

        private static unsafe float HorizontalSum(Vector256<float> v)
        {
            float* tmp = stackalloc float[8];
            Avx.Store(tmp, v);
            float sum = 0.0f;
            for (int i = 0; i < 8; i++)
                sum += tmp[i];
            return sum;
        }

        private static unsafe float HorizontalSum(Vector512<float> v)
        {
            float* tmp = stackalloc float[16];
            Avx512F.Store(tmp, v);
            float sum = 0.0f;
            for (int i = 0; i < 16; i++)
                sum += tmp[i];
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HorizontalSum128(Vector128<int> v)
        {
            Vector128<int> hi = Sse2.Add(v, Sse2.Shuffle(v, 0x4E)); // [2,3,0,1]
            hi = Sse2.Add(hi, Sse2.Shuffle(hi, 0xB1));               // [1,0,3,2]
            return hi.ToScalar();
        }

        private static unsafe float HorizontalMax(Vector512<float> v)
        {
            float* tmp = stackalloc float[16];
            Avx512F.Store(tmp, v);
            float max = tmp[0];
            for (int i = 1; i < 16; i++)
                if (tmp[i] > max) max = tmp[i];
            return max;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float E8M0ToFp32Half(byte value)
        {
            uint bits = value < 2 ? 0x00200000u << value : ((uint)value - 1u) << 23;
            return BitConverter.Int32BitsToSingle((int)bits);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
        {
            if (j < 4)
            {
                d = (byte)(q[j] & 63);
                m = (byte)(q[j + 4] & 63);
                return;
            }

            d = (byte)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }
}

