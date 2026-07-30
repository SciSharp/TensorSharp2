// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Persistent fork-join worker pool for the DeepSeek4 CPU executor.
//
// A decode step dispatches hundreds of small parallel regions per token
// (matmuls, compressors, attention). TPL's Parallel.For pays a fork/join
// latency in the milliseconds on busy multi-tenant hosts (thread-pool workers
// must be rescheduled by the OS for every region), which dwarfs the actual
// work. This pool keeps its workers alive across regions: they spin briefly
// waiting for the next work epoch before parking on a semaphore, so a region
// costs microseconds when regions arrive back to back — the same design as
// ggml's compute threadpool.
//
// Chunk claims go through a single 64-bit cursor whose high word is the work
// epoch, so a straggler from a previous region can never claim (or double-run)
// chunks of the current one. Not reentrant: regions are dispatched one at a
// time from the executor's single coordinating thread.
using System;
using System.Threading;

namespace TensorSharp.Models
{
    internal sealed class Dsv4SpinPool : IDisposable
    {
        private readonly Thread[] _workers;
        private readonly SemaphoreSlim _wake = new SemaphoreSlim(0);

        // work descriptor; fields are published by the Volatile.Write to _cursor
        private Action<int, int> _body;      // (startInclusive, endExclusive)
        private int _items;
        private int _grain;
        private int _chunks;

        // (epoch << 32) | nextStartIndex. Claims CAS this forward and are epoch-checked.
        private long _cursor;
        private int _completed;              // chunks completed in the current epoch
        private int _epochCounter;
        private volatile bool _stop;
        private volatile int _parked;

        // Spin budget before a worker parks. Large enough to bridge the serial
        // gaps between a forward pass's regions, small enough not to burn the
        // CPU quota for long when the model goes idle.
        private const int SpinRounds = 2048;

        public int Width { get; }

        public Dsv4SpinPool(int threads)
        {
            Width = Math.Max(1, threads);
            _workers = new Thread[Width - 1];
            for (int i = 0; i < _workers.Length; i++)
            {
                _workers[i] = new Thread(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"dsv4-pool-{i}",
                };
                _workers[i].Start();
            }
        }

        /// <summary>Runs body over chunked ranges of [0, items); blocks until all chunks completed.</summary>
        public void For(int items, Action<int, int> body)
        {
            if (items <= 0)
                return;

            int grain = Math.Max(1, items / (Width * 4));
            int chunks = (items + grain - 1) / grain;
            if (chunks == 1 || Width == 1)
            {
                body(0, items);
                return;
            }

            _body = body;
            _items = items;
            _grain = grain;
            _chunks = chunks;
            Volatile.Write(ref _completed, 0);
            int epoch = ++_epochCounter;
            // publishes the descriptor fields (release) and opens the epoch
            Volatile.Write(ref _cursor, (long)epoch << 32);

            int parked = _parked;
            if (parked > 0)
                _wake.Release(parked);

            RunChunks(epoch, body, items, grain);

            var sw = new SpinWait();
            while (Volatile.Read(ref _completed) < chunks)
                sw.SpinOnce();

            _body = null;
        }

        /// <summary>Convenience wrapper: per-index body.</summary>
        public void For(int items, Action<int> body)
        {
            For(items, (s, e) =>
            {
                for (int i = s; i < e; i++)
                    body(i);
            });
        }

        private void RunChunks(int epoch, Action<int, int> body, int items, int grain)
        {
            while (true)
            {
                long cur = Volatile.Read(ref _cursor);
                if ((int)(cur >> 32) != epoch)
                    return;                              // epoch closed
                int start = (int)cur;
                if (start >= items)
                    return;                              // drained
                long next = ((long)epoch << 32) | (uint)(start + grain);
                if (Interlocked.CompareExchange(ref _cursor, next, cur) != cur)
                    continue;                            // lost the claim; retry

                int end = Math.Min(start + grain, items);
                body(start, end);
                Interlocked.Increment(ref _completed);
            }
        }

        private void WorkerLoop()
        {
            int seenEpoch = 0;
            while (!_stop)
            {
                bool got = false;
                for (int spin = 0; spin < SpinRounds; spin++)
                {
                    if ((int)(Volatile.Read(ref _cursor) >> 32) != seenEpoch)
                    {
                        got = true;
                        break;
                    }
                    Thread.SpinWait(64);
                    if ((spin & 127) == 127)
                        Thread.Yield();
                }

                if (!got)
                {
                    Interlocked.Increment(ref _parked);
                    try
                    {
                        if ((int)(Volatile.Read(ref _cursor) >> 32) == seenEpoch && !_stop)
                            _wake.Wait(50);
                    }
                    finally
                    {
                        Interlocked.Decrement(ref _parked);
                    }
                    continue;
                }

                long cur = Volatile.Read(ref _cursor);
                seenEpoch = (int)(cur >> 32);
                var body = _body;                        // safe: published before the cursor write
                int items = _items;
                int grain = _grain;
                if (body != null)
                    RunChunks(seenEpoch, body, items, grain);
            }
        }

        public void Dispose()
        {
            _stop = true;
            _wake.Release(_workers.Length + 1);
            foreach (var w in _workers)
                w.Join(200);
            _wake.Dispose();
        }
    }
}
