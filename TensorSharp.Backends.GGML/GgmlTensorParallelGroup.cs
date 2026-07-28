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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.GGML
{
    /// <summary>
    /// Tensor-parallel group backed by the GGML backends — one ggml backend per
    /// GPU inside a single process, the counterpart of
    /// <c>TensorSharp.Cuda.TensorParallelGroup</c> for the ggml path.
    ///
    /// Three things make this group work with TensorSharp's host-resident tensor
    /// model:
    ///
    ///  * <see cref="GetAllocator"/> hands out a per-rank
    ///    <see cref="GgmlAllocator"/>, so every tensor records the GPU that owns
    ///    the work on it.
    ///  * A dispatch hook on <see cref="TensorSharp.OpRegistry"/> selects that
    ///    GPU from an op's result tensor, so ordinary <c>Ops.*</c> calls in the
    ///    per-model TP forwards land on the right device without each of them
    ///    having to say so.
    ///  * <see cref="RunPerRank"/> fans the ranks out across worker threads.
    ///    GGML ops submit and synchronize in one call, so a sequential rank loop
    ///    would run the GPUs one after another and make TP slower than a single
    ///    device; the fan-out is what actually overlaps them.
    /// </summary>
    public sealed class GgmlTensorParallelGroup : ITensorParallelGroup
    {
        private readonly GgmlAllocator[] _allocators;
        private readonly GgmlContext _context;
        private readonly RankWorkerPool _workers;
        private bool _disposed;

        /// <summary>
        /// Set to 0 to run the per-rank loops sequentially on the calling thread.
        /// Diagnostic escape hatch: it isolates concurrency from correctness when
        /// a TP result looks wrong.
        /// </summary>
        internal static readonly bool ParallelRanks =
            !string.Equals(Environment.GetEnvironmentVariable("TS_GGML_TP_PARALLEL"), "0", StringComparison.Ordinal);

        public GgmlTensorParallelGroup(GgmlContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));

            Degree = context.Degree;
            _allocators = new GgmlAllocator[Degree];
            for (int r = 0; r < Degree; r++)
                _allocators[r] = new GgmlAllocator(context, r);

            if (Degree > 1)
            {
                _workers = ParallelRanks ? new RankWorkerPool(Degree) : null;
                InstallDispatchHook();

                var names = new List<string>(Degree);
                for (int r = 0; r < Degree; r++)
                {
                    string desc = GgmlNative.GetGpuDeviceDescription(context.BackendType, context.DeviceIds[r]);
                    names.Add(desc ?? $"device {context.DeviceIds[r]}");
                }
                Console.WriteLine($"Tensor parallelism (GGML {context.BackendType}): {Degree} GPUs ({string.Join(", ", names)})");
                Console.WriteLine($"  AllReduce: {(context.HasDeviceAllReduce ? "device collective (NCCL / P2P)" : "host reduction")}" +
                                  $"; rank dispatch: {(_workers != null ? "parallel" : "sequential")}");
            }
        }

        /// <summary>Number of GPUs in this group (local to this process).</summary>
        public int Degree { get; }

        /// <summary>The multi-device GGML context this group drives.</summary>
        public GgmlContext Context => _context;

        public bool IsActive => Degree > 1;

        /// <summary>Single-node: global degree equals local degree.</summary>
        public int GlobalDegree => Degree;

        /// <summary>Single-node: no rank offset.</summary>
        public int GlobalRankOffset => 0;

        /// <summary>Single-node: one node.</summary>
        public int NodeCount => 1;

        public IAllocator GetAllocator(int rank)
        {
            if ((uint)rank >= (uint)Degree)
                throw new ArgumentOutOfRangeException(nameof(rank));
            return _allocators[rank];
        }

        /// <summary>
        /// Run <paramref name="body"/> once per rank, concurrently when the
        /// worker pool is enabled. Each worker pins itself to its rank's GPU for
        /// the duration of the call, and the first exception thrown by any rank
        /// is rethrown here once every rank has finished.
        /// </summary>
        public void RunPerRank(Action<int> body)
        {
            if (Degree == 1)
            {
                body(0);
                return;
            }

            if (_workers == null)
            {
                for (int r = 0; r < Degree; r++)
                    RunPinned(r, body);
                return;
            }

            _workers.Run(body);
        }

        /// <summary>
        /// Run one rank's share of a fan-out with this thread pinned to that
        /// rank's GPU, so the per-op dispatch hook cannot move it onto a backend
        /// another rank is concurrently using.
        /// </summary>
        private static void RunPinned(int rank, Action<int> body)
        {
            int previousRank = GgmlNative.GetActiveRank();
            bool previousPin = GgmlNative.PinRank(rank);
            try
            {
                body(rank);
            }
            finally
            {
                GgmlNative.RestorePin(previousPin, previousRank);
            }
        }

        /// <summary>
        /// In-place AllReduce (sum) across all GPUs. Every tensor is a
        /// host-resident F32 buffer, so the sum happens where the data already
        /// is — on the CPU for small payloads, staged through the device
        /// collective for large ones.
        /// </summary>
        public void AllReduce(Tensor[] tensors)
        {
            if (!IsActive) return;
            if (tensors == null || tensors.Length != Degree)
                throw new ArgumentException($"Expected {Degree} tensors, got {tensors?.Length ?? 0}.");

            long count = tensors[0].Storage.ElementCount;
            for (int r = 1; r < Degree; r++)
            {
                if (tensors[r].Storage.ElementCount != count)
                    throw new ArgumentException("AllReduce requires identically sized tensors on every rank.");
            }
            if (count <= 0) return;

            // Drain any deferred GPU work so the host buffers hold final values.
            GgmlBasicOps.HostReadBarrier();

            unsafe
            {
                float** ptrs = stackalloc float*[Degree];
                for (int r = 0; r < Degree; r++)
                {
                    if (tensors[r].ElementType != DType.Float32)
                        throw new NotSupportedException("GGML tensor-parallel AllReduce requires Float32 tensors.");
                    ptrs[r] = (float*)tensors[r].Storage.PtrAtElement(tensors[r].StorageOffset);
                }

                if (count >= DeviceAllReduceThreshold && _context.HasDeviceAllReduce &&
                    GgmlNative.TensorParallelAllReduceDevice(ptrs, Degree, count))
                {
                    return;
                }
                GgmlNative.TensorParallelAllReduceHost(ptrs, Degree, count);
            }
        }

        /// <summary>
        /// Element count above which the device collective beats the host sum.
        /// Below it, staging through VRAM costs two PCIe crossings that dwarf a
        /// few microseconds of vectorized addition; above it the collective wins.
        /// Override with TS_GGML_TP_DEVICE_AR_THRESHOLD.
        /// </summary>
        public static readonly long DeviceAllReduceThreshold = ReadThreshold();

        private static long ReadThreshold()
        {
            string raw = Environment.GetEnvironmentVariable("TS_GGML_TP_DEVICE_AR_THRESHOLD");
            if (raw != null && long.TryParse(raw, out long parsed) && parsed >= 0)
                return parsed;
            return 256 * 1024;
        }

        /// <summary>All GGML work is synchronous per op; nothing is left in flight.</summary>
        public void Synchronize() => GgmlBasicOps.HostReadBarrier();

        /// <summary>Single-node group: no cross-node rendezvous.</summary>
        public void Barrier() { }

        public void BroadcastControl(int op, int[] payload) =>
            throw new NotSupportedException("Single-node GGML TP groups have no worker nodes.");

        public (int op, int[] payload) ReceiveControl() =>
            throw new NotSupportedException("Single-node GGML TP groups have no worker nodes.");

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _workers?.Dispose();
            if (Degree > 1)
                OpRegistry.PreInvokeHook = null;
        }

        // --- Per-op device routing -----------------------------------------

        /// <summary>
        /// Route each op to the GPU that owns its result. TensorSharp ops take
        /// the destination tensor first, so the result's allocator identifies the
        /// rank; ops whose first argument is not a GGML tensor keep whatever rank
        /// the enclosing <see cref="RunPerRank"/> scope selected.
        /// </summary>
        private static void InstallDispatchHook()
        {
            OpRegistry.PreInvokeHook = args =>
            {
                if (args.Length > 0 && args[0] is Tensor t && t.Storage is GgmlStorage s)
                    GgmlNative.SetActiveRankIfUnpinned(s.DeviceId);
            };
        }

        /// <summary>Scoped rank switch for the calling thread.</summary>
        internal readonly struct RankScope : IDisposable
        {
            private readonly int _previous;
            private readonly bool _changed;

            public RankScope(int rank)
            {
                _previous = GgmlNative.GetActiveRank();
                _changed = _previous != rank;
                if (_changed) GgmlNative.SetActiveRank(rank);
            }

            public void Dispose()
            {
                if (_changed) GgmlNative.SetActiveRank(_previous);
            }
        }

        // --- Rank worker pool ------------------------------------------------

        /// <summary>
        /// One long-lived thread per rank &gt; 0 (rank 0 runs on the caller, which
        /// avoids a hand-off on the critical path). Threads are persistent so the
        /// per-thread native rank selection and ggml's per-thread CUDA context
        /// binding are established once rather than per layer.
        /// </summary>
        private sealed class RankWorkerPool : IDisposable
        {
            private readonly Thread[] _threads;
            private readonly SemaphoreSlim[] _start;
            private readonly SemaphoreSlim[] _done;
            private readonly Exception[] _errors;
            private Action<int> _body;
            private volatile bool _shutdown;

            public RankWorkerPool(int degree)
            {
                int workers = degree - 1;
                _threads = new Thread[workers];
                _start = new SemaphoreSlim[workers];
                _done = new SemaphoreSlim[workers];
                _errors = new Exception[degree];

                for (int i = 0; i < workers; i++)
                {
                    int rank = i + 1;
                    int slot = i;
                    _start[slot] = new SemaphoreSlim(0, 1);
                    _done[slot] = new SemaphoreSlim(0, 1);
                    _threads[slot] = new Thread(() => WorkerLoop(rank, slot))
                    {
                        IsBackground = true,
                        Name = $"ggml-tp-rank{rank}",
                    };
                    _threads[slot].Start();
                }
            }

            private void WorkerLoop(int rank, int slot)
            {
                // Pin this thread to its rank once; every op it issues from here
                // on runs on that GPU.
                GgmlNative.SetActiveRank(rank);
                while (true)
                {
                    _start[slot].Wait();
                    if (_shutdown) return;
                    try
                    {
                        RunPinned(rank, _body);
                    }
                    catch (Exception ex)
                    {
                        _errors[rank] = ex;
                    }
                    finally
                    {
                        _done[slot].Release();
                    }
                }
            }

            public void Run(Action<int> body)
            {
                _body = body;
                Array.Clear(_errors, 0, _errors.Length);

                for (int i = 0; i < _start.Length; i++)
                    _start[i].Release();

                // Rank 0 on the calling thread, overlapping the workers.
                try
                {
                    RunPinned(0, body);
                }
                catch (Exception ex)
                {
                    _errors[0] = ex;
                }

                for (int i = 0; i < _done.Length; i++)
                    _done[i].Wait();

                for (int r = 0; r < _errors.Length; r++)
                {
                    if (_errors[r] != null)
                    {
                        var error = _errors[r];
                        _errors[r] = null;
                        throw new AggregateException($"GGML tensor-parallel rank {r} failed.", error);
                    }
                }
            }

            public void Dispose()
            {
                _shutdown = true;
                for (int i = 0; i < _start.Length; i++)
                    _start[i].Release();
                for (int i = 0; i < _threads.Length; i++)
                    _threads[i]?.Join(TimeSpan.FromSeconds(5));
            }
        }
    }
}
