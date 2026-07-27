using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaDriverApi
    {
        private const string LibName = "cuda";

        [DllImport(LibName)]
        public static extern int cuInit(uint flags);

        [DllImport(LibName)]
        public static extern int cuDeviceGet(out int device, int ordinal);

        [DllImport(LibName)]
        public static extern int cuDeviceGetCount(out int count);

        [DllImport(LibName)]
        public static extern int cuDeviceGetName(byte[] name, int len, int device);

        [DllImport(LibName, EntryPoint = "cuDeviceTotalMem_v2")]
        public static extern int cuDeviceTotalMem(out UIntPtr bytes, int device);

        [DllImport(LibName, EntryPoint = "cuMemGetInfo_v2")]
        public static extern int cuMemGetInfo(out UIntPtr free, out UIntPtr total);

        [DllImport(LibName)]
        public static extern int cuDeviceGetAttribute(out int value, int attribute, int device);

        [DllImport(LibName, EntryPoint = "cuCtxCreate_v2")]
        public static extern int cuCtxCreate(out IntPtr ctx, uint flags, int device);

        [DllImport(LibName, EntryPoint = "cuCtxDestroy_v2")]
        public static extern int cuCtxDestroy(IntPtr ctx);

        [DllImport(LibName)]
        public static extern int cuCtxSetCurrent(IntPtr ctx);

        [DllImport(LibName)]
        public static extern int cuCtxGetCurrent(out IntPtr ctx);

        [DllImport(LibName)]
        public static extern int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device);

        [DllImport(LibName)]
        public static extern int cuDevicePrimaryCtxRelease(int device);

        [DllImport(LibName, EntryPoint = "cuMemAlloc_v2")]
        public static extern int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize);

        [DllImport(LibName, EntryPoint = "cuMemFree_v2")]
        public static extern int cuMemFree(IntPtr devicePtr);

        /// <summary>Page-locked (pinned) host allocation. A captured HtoD memcpy
        /// node whose source is pinned host memory re-reads the buffer on every
        /// graph launch, which is how decode-graph replays receive per-token
        /// parameters (see CudaDecodeDynParams).</summary>
        [DllImport(LibName, EntryPoint = "cuMemHostAlloc")]
        public static extern int cuMemHostAlloc(out IntPtr hostPtr, UIntPtr byteSize, uint flags);

        [DllImport(LibName, EntryPoint = "cuMemFreeHost")]
        public static extern int cuMemFreeHost(IntPtr hostPtr);

        [DllImport(LibName, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount);

        [DllImport(LibName, EntryPoint = "cuMemcpyHtoDAsync_v2")]
        public static extern int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoHAsync_v2")]
        public static extern int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoD_v2")]
        public static extern int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount);

        [DllImport(LibName, EntryPoint = "cuMemcpyDtoDAsync_v2")]
        public static extern int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        [DllImport(LibName, EntryPoint = "cuMemsetD8_v2")]
        public static extern int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count);

        [DllImport(LibName)]
        public static extern int cuModuleLoadData(out IntPtr module, IntPtr image);

        [DllImport(LibName)]
        public static extern int cuModuleGetFunction(out IntPtr function, IntPtr module, string name);

        [DllImport(LibName)]
        public static extern int cuModuleUnload(IntPtr module);

        [DllImport(LibName)]
        public static extern int cuFuncSetAttribute(IntPtr function, int attribute, int value);

        // CUfunction_attribute: opt-in cap for dynamic shared memory per block
        // (default launches are limited to 48 KB without it).
        public const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;

        [DllImport(LibName)]
        public static extern int cuLaunchKernel(
            IntPtr function,
            uint gridDimX,
            uint gridDimY,
            uint gridDimZ,
            uint blockDimX,
            uint blockDimY,
            uint blockDimZ,
            uint sharedMemBytes,
            IntPtr stream,
            IntPtr kernelParams,
            IntPtr extra);

        [DllImport(LibName)]
        public static extern int cuStreamCreate(out IntPtr stream, uint flags);

        [DllImport(LibName, EntryPoint = "cuStreamDestroy_v2")]
        public static extern int cuStreamDestroy(IntPtr stream);

        [DllImport(LibName)]
        public static extern int cuStreamSynchronize(IntPtr stream);

        // ---- CUDA Graphs (stream capture) ----
        [DllImport(LibName, EntryPoint = "cuStreamBeginCapture_v2")]
        public static extern int cuStreamBeginCapture(IntPtr stream, int mode);

        [DllImport(LibName)]
        public static extern int cuStreamEndCapture(IntPtr stream, out IntPtr graph);

        [DllImport(LibName)]
        public static extern int cuGraphInstantiateWithFlags(out IntPtr graphExec, IntPtr graph, ulong flags);

        [DllImport(LibName)]
        public static extern int cuGraphLaunch(IntPtr graphExec, IntPtr stream);

        [DllImport(LibName)]
        public static extern int cuGraphExecDestroy(IntPtr graphExec);

        [DllImport(LibName)]
        public static extern int cuGraphDestroy(IntPtr graph);

        // CUstreamCaptureMode: 0 = GLOBAL, 1 = THREAD_LOCAL, 2 = RELAXED.
        public const int CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;
        // Relaxed capture permits API calls that are unsafe-by-default during
        // capture (notably cuMemAlloc for pool misses); the capture owner takes
        // responsibility for the referenced memory's lifetime.
        public const int CU_STREAM_CAPTURE_MODE_RELAXED = 2;

        [DllImport(LibName)]
        public static extern int cuGetErrorString(int error, out IntPtr str);

        // ---- CUDA Events ----
        [DllImport(LibName)]
        public static extern int cuEventCreate(out IntPtr phEvent, uint flags);

        [DllImport(LibName, EntryPoint = "cuEventDestroy_v2")]
        public static extern int cuEventDestroy(IntPtr hEvent);

        [DllImport(LibName, EntryPoint = "cuEventRecord")]
        public static extern int cuEventRecord(IntPtr hEvent, IntPtr hStream);

        [DllImport(LibName)]
        public static extern int cuEventSynchronize(IntPtr hEvent);

        [DllImport(LibName)]
        public static extern int cuStreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags);

        // ---- Peer-to-Peer (multi-GPU) ----
        [DllImport(LibName)]
        public static extern int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev);

        [DllImport(LibName)]
        public static extern int cuCtxEnablePeerAccess(IntPtr peerContext, uint flags);

        [DllImport(LibName)]
        public static extern int cuMemcpyPeerAsync(
            IntPtr dstDevice, IntPtr dstContext,
            IntPtr srcDevice, IntPtr srcContext,
            UIntPtr byteCount, IntPtr hStream);

        // CUstream flags
        public const uint CU_STREAM_DEFAULT = 0;
        public const uint CU_STREAM_NON_BLOCKING = 1;

        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
        public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    }
}
