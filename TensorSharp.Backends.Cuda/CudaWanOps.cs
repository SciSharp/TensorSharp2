// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Tensor-level wrappers for the Wan video direct-backend kernels
// (ts_wan_* in tensorsharp_kernels.cu). Every method returns false when the
// tensors are not device-resident contiguous F32 or the kernel is missing from
// the loaded PTX (built before these kernels existed) — callers fall back to
// their generic path.
using System;
using TensorSharp.Cuda;

namespace TensorSharp.Cuda
{
    public static class CudaWanOps
    {
        /// <summary>Streaming multi-head attention over token-major [seq, heads, headDim]
        /// q/k/v with independent kv length and optional additive [heads, sq, sk] bias.</summary>
        public static bool TryAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor bias,
            int numHeads, int seqQ, int seqK, int headDim, float scale)
        {
            long expQ = (long)seqQ * numHeads * headDim;
            long expK = (long)seqK * numHeads * headDim;
            if (!CudaKernelOps.TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !CudaKernelOps.TryGetContiguousFloat(query, out CudaStorage queryStorage, out IntPtr queryPtr, out int queryCount) ||
                !CudaKernelOps.TryGetContiguousFloat(key, out CudaStorage keyStorage, out IntPtr keyPtr, out int keyCount) ||
                !CudaKernelOps.TryGetContiguousFloat(value, out CudaStorage valueStorage, out IntPtr valuePtr, out int valueCount) ||
                seqQ <= 0 || seqK <= 0 || numHeads <= 0 || headDim <= 0 ||
                resultCount != expQ || queryCount != expQ || keyCount != expK || valueCount != expK)
            {
                return false;
            }

            IntPtr biasPtr = IntPtr.Zero;
            CudaStorage biasStorage = null;
            if (bias != null)
            {
                if (!CudaKernelOps.TryGetContiguousFloat(bias, out biasStorage, out biasPtr, out int biasCount) ||
                    biasCount != (long)numHeads * seqQ * seqK)
                    return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanAttentionF32(headDim))
                return false;

            queryStorage.EnsureDeviceCurrent();
            keyStorage.EnsureDeviceCurrent();
            valueStorage.EnsureDeviceCurrent();
            biasStorage?.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanAttentionF32(queryPtr, keyPtr, valuePtr, biasPtr, resultPtr,
                numHeads, seqQ, seqK, headDim, scale, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>In-place interleaved-pair RoPE over x [seq, heads*headDim] with
        /// pair-duplicated [seq, headDim] cos/sin tables.</summary>
        public static bool TryRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int heads, int seq, int headDim)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !CudaKernelOps.TryGetContiguousFloat(cos, out CudaStorage cosStorage, out IntPtr cosPtr, out int cosCount) ||
                !CudaKernelOps.TryGetContiguousFloat(sin, out CudaStorage sinStorage, out IntPtr sinPtr, out int sinCount) ||
                xCount != (long)seq * heads * headDim ||
                cosCount != (long)seq * headDim || sinCount != (long)seq * headDim)
            {
                return false;
            }
            CudaAllocator allocator = xStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanRopeF32)
                return false;
            xStorage.EnsureDeviceCurrent();
            cosStorage.EnsureDeviceCurrent();
            sinStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanRopeF32(xPtr, cosPtr, sinPtr, heads, seq, headDim, allocator.Stream.Handle);
            xStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>y[r, c] = x[r, c] * (1 + scale[c]) + shift[c] over rows [rowFrom, rowFrom+rowCount).
        /// y and x may be the same tensor.</summary>
        public static bool TryModulateRows(Tensor y, Tensor x, Tensor shift, Tensor scale, long rowFrom, long rowCount)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(y, out CudaStorage yStorage, out IntPtr yPtr, out int yCount) ||
                !CudaKernelOps.TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !CudaKernelOps.TryGetContiguousFloat(shift, out CudaStorage shiftStorage, out IntPtr shiftPtr, out int shiftCount) ||
                !CudaKernelOps.TryGetContiguousFloat(scale, out CudaStorage scaleStorage, out IntPtr scalePtr, out int scaleCount) ||
                x.DimensionCount != 2 || yCount != xCount)
            {
                return false;
            }
            int cols = checked((int)x.Sizes[1]);
            if (shiftCount < cols || scaleCount < cols || rowFrom < 0 || rowFrom + rowCount > x.Sizes[0])
                return false;
            CudaAllocator allocator = yStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanModulateF32)
                return false;
            xStorage.EnsureDeviceCurrent();
            shiftStorage.EnsureDeviceCurrent();
            scaleStorage.EnsureDeviceCurrent();
            if (!ReferenceEquals(yStorage, xStorage)) yStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanModulateF32(yPtr, xPtr, shiftPtr, scalePtr, rowFrom, rowCount, cols, allocator.Stream.Handle);
            yStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>x[r, c] += v[r, c] * gate[c] over rows [rowFrom, rowFrom+rowCount).</summary>
        public static bool TryGateAddRows(Tensor x, Tensor v, Tensor gate, long rowFrom, long rowCount)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !CudaKernelOps.TryGetContiguousFloat(v, out CudaStorage vStorage, out IntPtr vPtr, out int vCount) ||
                !CudaKernelOps.TryGetContiguousFloat(gate, out CudaStorage gateStorage, out IntPtr gatePtr, out int gateCount) ||
                x.DimensionCount != 2 || vCount != xCount)
            {
                return false;
            }
            int cols = checked((int)x.Sizes[1]);
            if (gateCount < cols || rowFrom < 0 || rowFrom + rowCount > x.Sizes[0])
                return false;
            CudaAllocator allocator = xStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanGateAddF32)
                return false;
            xStorage.EnsureDeviceCurrent();
            vStorage.EnsureDeviceCurrent();
            gateStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanGateAddF32(xPtr, vPtr, gatePtr, rowFrom, rowCount, cols, allocator.Stream.Handle);
            xStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>In-place x[r, c] *= gain[c].</summary>
        public static bool TryScaleCols(Tensor x, Tensor gain)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !CudaKernelOps.TryGetContiguousFloat(gain, out CudaStorage gainStorage, out IntPtr gainPtr, out int gainCount) ||
                x.DimensionCount != 2)
            {
                return false;
            }
            int cols = checked((int)x.Sizes[1]);
            if (gainCount < cols)
                return false;
            CudaAllocator allocator = xStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanScaleColsF32)
                return false;
            xStorage.EnsureDeviceCurrent();
            gainStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanScaleColsF32(xPtr, gainPtr, x.Sizes[0], cols, allocator.Stream.Handle);
            xStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>im2col over a [C, H, W] frame into columns [C*KH*KW, bandOh*OW] for
        /// the output row band [oy0, oy0+bandOh).</summary>
        public static bool TryIm2Col(Tensor col, Tensor input, int c, int h, int w, int kh, int kw,
            int oh, int ow, int padH, int padW, int strideH, int strideW, int oy0, int bandOh)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(col, out CudaStorage colStorage, out IntPtr colPtr, out int colCount) ||
                !CudaKernelOps.TryGetContiguousFloat(input, out CudaStorage inputStorage, out IntPtr inputPtr, out int inputCount) ||
                colCount != (long)c * kh * kw * bandOh * ow || inputCount < (long)c * h * w)
            {
                return false;
            }
            CudaAllocator allocator = colStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanIm2colF32)
                return false;
            inputStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanIm2colF32(inputPtr, colPtr, c, h, w, kh, kw, oh, ow,
                padH, padW, strideH, strideW, oy0, bandOh, allocator.Stream.Handle);
            colStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>In-place channel RMS norm over a [C, HW] frame with per-channel gain.</summary>
        public static bool TryChannelRms(Tensor x, Tensor gamma, int c, long hw, float eps)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !CudaKernelOps.TryGetContiguousFloat(gamma, out CudaStorage gammaStorage, out IntPtr gammaPtr, out int gammaCount) ||
                xCount < (long)c * hw || gammaCount < c)
            {
                return false;
            }
            CudaAllocator allocator = xStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanChanRmsF32)
                return false;
            xStorage.EnsureDeviceCurrent();
            gammaStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanChanRmsF32(xPtr, gammaPtr, c, hw, eps, allocator.Stream.Handle);
            xStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>Device zero-fill of a contiguous F32 tensor (Ops.Fill routes CUDA
        /// storage through the elementwise CPU fallback, which round-trips the buffer).</summary>
        public static bool TryZero(Tensor t)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(t, out CudaStorage storage, out IntPtr ptr, out int count))
                return false;
            CudaAllocator allocator = storage.AllocatorImpl;
            storage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            Interop.CudaDriverApi.cuMemsetD8Async(ptr, 0, (UIntPtr)((long)count * sizeof(float)), allocator.Stream.Handle);
            storage.MarkDeviceModified();
            return true;
        }

        /// <summary>Nearest x2 spatial upsample [C, H, W] -> [C, 2H, 2W].</summary>
        public static bool TryUpsample2x(Tensor dst, Tensor src, int c, int h, int w)
        {
            if (!CudaKernelOps.TryGetContiguousFloat(dst, out CudaStorage dstStorage, out IntPtr dstPtr, out int dstCount) ||
                !CudaKernelOps.TryGetContiguousFloat(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcCount) ||
                dstCount != (long)c * 4 * h * w || srcCount < (long)c * h * w)
            {
                return false;
            }
            CudaAllocator allocator = dstStorage.AllocatorImpl;
            if (!CudaKernelOps.TryGetKernels(allocator, out CudaKernels kernels) || !kernels.HasWanUpsample2xF32)
                return false;
            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchWanUpsample2xF32(srcPtr, dstPtr, c, h, w, allocator.Stream.Handle);
            dstStorage.MarkDeviceModified();
            return true;
        }
    }
}
