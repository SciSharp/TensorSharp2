using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Multi-GPU collective operations using CUDA peer-to-peer memory access.
    /// Works on any platform with CUDA (Windows, Linux) without requiring NCCL.
    /// 
    /// AllReduce algorithm (reduce-to-zero + broadcast):
    ///   1. Each non-zero GPU copies its partial to GPU 0's staging buffer
    ///   2. GPU 0 accumulates: buffer[0] += staging (via elementwise-add kernel)
    ///   3. GPU 0 broadcasts the reduced result to all other GPUs
    /// 
    /// For N=2 this is optimal (1 copy + 1 add + 1 broadcast). For N≤8 the
    /// serial reduction on GPU 0 is bandwidth-bound on the P2P links, which
    /// matches ring-allreduce throughput when NVLink is not available.
    /// </summary>
    internal sealed class CudaP2PCommunicator : IDisposable
    {
        private readonly CudaAllocator[] _allocators;
        private readonly int _worldSize;
        private readonly bool[] _p2pEnabled;

        // Staging buffer on GPU 0 for accumulation during AllReduce.
        private IntPtr _stagingBuffer;
        private long _stagingBufferBytes;

        public CudaP2PCommunicator(CudaAllocator[] allocators)
        {
            _allocators = allocators ?? throw new ArgumentNullException(nameof(allocators));
            _worldSize = allocators.Length;

            if (_worldSize < 2)
                throw new ArgumentException("P2P communicator requires at least 2 GPUs.", nameof(allocators));

            _p2pEnabled = EnablePeerAccess(allocators);
        }

        public int WorldSize => _worldSize;

        // Device pointer to a tensor's storage. NOTE: Storage.PtrAtElement returns
        // a HOST pointer and marks the storage host-dirty (it's the "raw host write"
        // checkout), so it must NOT be used for device-to-device / peer copies —
        // doing so passes a host address where CUDA expects a device address
        // (CUDA_ERROR_INVALID_VALUE) and corrupts the tensor's dirty state,
        // clobbering the reduced result on the next EnsureDeviceCurrent.
        private static IntPtr DevicePtr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(0);

        private static bool[] EnablePeerAccess(CudaAllocator[] allocators)
        {
            int n = allocators.Length;
            var enabled = new bool[n * n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;

                    allocators[i].Context.MakeCurrent();
                    CudaDriverApi.cuDeviceCanAccessPeer(out int canAccess,
                        allocators[i].DeviceId, allocators[j].DeviceId).ThrowOnError();

                    if (canAccess == 1)
                    {
                        // cuCtxEnablePeerAccess returns CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                        // (704) if already enabled — treat that as success.
                        int result = CudaDriverApi.cuCtxEnablePeerAccess(allocators[j].Context.Handle, 0);
                        if (result == 0 || result == 704)
                            enabled[i * n + j] = true;
                        else
                            result.ThrowOnError();
                    }
                }
            }

            return enabled;
        }

        private bool CanAccessPeer(int from, int to) => _p2pEnabled[from * _worldSize + to];

        private void EnsureStagingBuffer(long requiredBytes)
        {
            if (_stagingBufferBytes >= requiredBytes)
                return;

            if (_stagingBuffer != IntPtr.Zero)
            {
                _allocators[0].Context.MakeCurrent();
                CudaDriverApi.cuMemFree(_stagingBuffer);
            }

            _allocators[0].Context.MakeCurrent();
            CudaDriverApi.cuMemAlloc(out _stagingBuffer, new UIntPtr((ulong)requiredBytes)).ThrowOnError();
            _stagingBufferBytes = requiredBytes;
        }

        /// <summary>
        /// In-place AllReduce (sum) across all GPUs. After this call, every
        /// tensor in <paramref name="tensors"/> holds the element-wise sum of
        /// all input values. tensors[i] must reside on GPU i.
        /// </summary>
        public void AllReduce(Tensor[] tensors)
        {
            if (tensors == null || tensors.Length != _worldSize)
                throw new ArgumentException($"Expected {_worldSize} tensors, got {tensors?.Length ?? 0}.");

            long byteCount = tensors[0].Storage.ByteLength;
            int elementCount = (int)tensors[0].Storage.ElementCount;

            // Synchronize all GPUs to ensure partial results are complete.
            for (int i = 0; i < _worldSize; i++)
                _allocators[i].Synchronize();

            // Phase 1: Reduce to GPU 0.
            _allocators[0].Context.MakeCurrent();
            for (int i = 1; i < _worldSize; i++)
            {
                IntPtr srcPtr = DevicePtr(tensors[i]);
                IntPtr dstPtr = DevicePtr(tensors[0]);

                if (CanAccessPeer(0, i))
                {
                    // P2P copy: GPU i → GPU 0 staging, then add.
                    EnsureStagingBuffer(byteCount);
                    CudaDriverApi.cuMemcpyPeerAsync(
                        _stagingBuffer, _allocators[0].Context.Handle,
                        srcPtr, _allocators[i].Context.Handle,
                        new UIntPtr((ulong)byteCount),
                        _allocators[0].Stream.Handle).ThrowOnError();

                    _allocators[0].Kernels.LaunchBinaryF32(
                        dstPtr, _stagingBuffer, dstPtr,
                        elementCount, 0, // op=0 → Add
                        _allocators[0].Stream.Handle);
                }
                else
                {
                    // Fallback: stage through host memory.
                    AllReduceViaHost(tensors[0], tensors[i], elementCount, byteCount);
                }
            }

            _allocators[0].Synchronize();

            // Phase 2: Broadcast from GPU 0 to all other GPUs.
            IntPtr resultPtr = DevicePtr(tensors[0]);
            for (int i = 1; i < _worldSize; i++)
            {
                _allocators[i].Context.MakeCurrent();
                IntPtr dstPtr = DevicePtr(tensors[i]);

                if (CanAccessPeer(i, 0))
                {
                    CudaDriverApi.cuMemcpyPeerAsync(
                        dstPtr, _allocators[i].Context.Handle,
                        resultPtr, _allocators[0].Context.Handle,
                        new UIntPtr((ulong)byteCount),
                        _allocators[i].Stream.Handle).ThrowOnError();
                }
                else
                {
                    BroadcastViaHost(tensors[0], tensors[i], byteCount);
                }
            }

            // Final sync so all GPUs have the reduced result.
            for (int i = 0; i < _worldSize; i++)
                _allocators[i].Synchronize();
        }

        private unsafe void AllReduceViaHost(Tensor dst, Tensor src, int elementCount, long byteCount)
        {
            // Copy src (GPU i) → host → dst (GPU 0), then add on GPU 0.
            var hostBuf = new float[elementCount];
            fixed (float* hostPtr = hostBuf)
            {
                _allocators[src.Storage.Allocator.DeviceId].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)hostPtr, DevicePtr(src),
                    new UIntPtr((ulong)byteCount)).ThrowOnError();

                _allocators[0].Context.MakeCurrent();
                EnsureStagingBuffer(byteCount);
                CudaDriverApi.cuMemcpyHtoD(_stagingBuffer, (IntPtr)hostPtr,
                    new UIntPtr((ulong)byteCount)).ThrowOnError();

                IntPtr dstDevPtr = DevicePtr(dst);
                _allocators[0].Kernels.LaunchBinaryF32(
                    dstDevPtr, _stagingBuffer,
                    dstDevPtr,
                    elementCount, 0,
                    _allocators[0].Stream.Handle);
            }
        }

        private unsafe void BroadcastViaHost(Tensor src, Tensor dst, long byteCount)
        {
            int elementCount = (int)src.Storage.ElementCount;
            var hostBuf = new float[elementCount];
            fixed (float* hostPtr = hostBuf)
            {
                _allocators[0].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)hostPtr, DevicePtr(src),
                    new UIntPtr((ulong)byteCount)).ThrowOnError();

                int dstDevice = dst.Storage.Allocator.DeviceId;
                _allocators[dstDevice].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyHtoD(DevicePtr(dst), (IntPtr)hostPtr,
                    new UIntPtr((ulong)byteCount)).ThrowOnError();
            }
        }

        public void Dispose()
        {
            if (_stagingBuffer != IntPtr.Zero)
            {
                _allocators[0].Context.MakeCurrent();
                CudaDriverApi.cuMemFree(_stagingBuffer);
                _stagingBuffer = IntPtr.Zero;
                _stagingBufferBytes = 0;
            }
        }
    }
}
