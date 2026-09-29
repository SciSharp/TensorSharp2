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
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Parallel-for hook for the managed CPU kernels in this assembly (GEMM,
    /// elementwise and row ops). Core has no thread pool of its own, so by
    /// default it forks on <see cref="Parallel.For(int, int, Action{int})"/>;
    /// a host that owns a cheaper fork/join (TensorSharp.Models binds its
    /// persistent spinning CpuWorkerPool here at module load) installs it with
    /// <see cref="SetRunner"/>. A ThreadPool fork/join costs ~20-60 us to wake
    /// parked threads, which is the whole budget of a typical per-op call.
    ///
    /// Runner contract: run body(i) for every i in [0, blockCount) and return
    /// once all have completed, rethrowing a failure (an AggregateException is
    /// fine); nested or concurrent calls must be safe (running inline is fine).
    /// </summary>
    internal static class CpuParallel
    {
        private sealed class Runner
        {
            public readonly Action<int, Action<int>> For;
            public readonly Func<int>? Degree;
            public Runner(Action<int, Action<int>> forImpl, Func<int>? degree)
            {
                For = forImpl;
                Degree = degree;
            }
        }

        private static Runner? _runner;
        private static int _cachedDegree;

        /// <summary>
        /// Install the fork/join used by every Core CPU kernel. <paramref name="degreeOfParallelism"/>
        /// is evaluated lazily (at the first parallel call) so installing a runner never has to spin
        /// up its threads. Pass null for <paramref name="runner"/> to restore the ThreadPool default.
        /// </summary>
        public static void SetRunner(Action<int, Action<int>>? runner, Func<int>? degreeOfParallelism)
        {
            Volatile.Write(ref _runner, runner == null ? null : new Runner(runner, degreeOfParallelism));
            Volatile.Write(ref _cachedDegree, 0);
        }

        /// <summary>True when a host runner (not the ThreadPool default) is installed.</summary>
        public static bool HasCustomRunner => Volatile.Read(ref _runner) != null;

        /// <summary>
        /// Number of threads a parallel call runs on; kernels size their work split
        /// from this (a few blocks per thread) rather than from the core count.
        /// </summary>
        public static int DegreeOfParallelism
        {
            get
            {
                int d = Volatile.Read(ref _cachedDegree);
                if (d > 0) return d;
                Runner? r = Volatile.Read(ref _runner);
                d = r?.Degree != null ? r.Degree() : Environment.ProcessorCount;
                d = Math.Max(1, d);
                Volatile.Write(ref _cachedDegree, d);
                return d;
            }
        }

        /// <summary>Run <paramref name="body"/> for every block in [0, blockCount).</summary>
        public static void For(int blockCount, Action<int> body)
        {
            if (blockCount <= 0) return;
            if (blockCount == 1)
            {
                body(0);
                return;
            }

            Runner? r = Volatile.Read(ref _runner);
            if (r != null) r.For(blockCount, body);
            else Parallel.For(0, blockCount, body);
        }

        /// <summary>
        /// Split [0, count) into contiguous chunks of at least <paramref name="minChunk"/> items and
        /// run <paramref name="body"/>(start, end) for each. The chunk count follows the WORK: small
        /// ranges stay on the calling thread, large ones get a few chunks per worker so the tail
        /// balances. Inline when the range is a single chunk.
        /// </summary>
        public static void ForRange(long count, long minChunk, Action<long, long> body)
        {
            if (count <= 0) return;
            minChunk = Math.Max(1, minChunk);
            if (count <= minChunk)
            {
                body(0, count);
                return;
            }

            int degree = DegreeOfParallelism;
            long maxBlocks = Math.Max(1, (long)degree * 4);
            long blocks = Math.Min(maxBlocks, (count + minChunk - 1) / minChunk);
            if (blocks <= 1)
            {
                body(0, count);
                return;
            }

            long chunk = (count + blocks - 1) / blocks;
            blocks = (count + chunk - 1) / chunk;
            For((int)blocks, b =>
            {
                long start = b * chunk;
                long end = Math.Min(count, start + chunk);
                if (start < end) body(start, end);
            });
        }
    }
}
