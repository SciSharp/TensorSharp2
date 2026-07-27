using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaEvent : IDisposable
    {
        private IntPtr handle;

        private CudaEvent(IntPtr handle)
        {
            this.handle = handle;
        }

        public IntPtr Handle => handle;

        public static CudaEvent Create(bool disableTiming = true)
        {
            // CU_EVENT_DISABLE_TIMING = 0x2 — cheaper when only used for synchronization.
            uint flags = disableTiming ? 0x2u : 0u;
            CudaDriverApi.cuEventCreate(out IntPtr ev, flags).ThrowOnError();
            return new CudaEvent(ev);
        }

        public void Record(CudaStream stream)
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaEvent));

            CudaDriverApi.cuEventRecord(handle, stream.Handle).ThrowOnError();
        }

        public void Synchronize()
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaEvent));

            CudaDriverApi.cuEventSynchronize(handle).ThrowOnError();
        }

        public static void MakeStreamWait(CudaStream stream, CudaEvent ev)
        {
            CudaDriverApi.cuStreamWaitEvent(stream.Handle, ev.handle, 0).ThrowOnError();
        }

        public void Dispose()
        {
            IntPtr h = Interlocked.Exchange(ref handle, IntPtr.Zero);
            if (h != IntPtr.Zero)
                CudaDriverApi.cuEventDestroy(h);
        }
    }
}
