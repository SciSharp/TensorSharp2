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
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Where a pure-C# kernel's parallel loop runs: a persistent spinning <see cref="CpuWorkerPool"/>
    /// or, under TS_CPU_POOL=0 (hosts that cannot afford dedicated spinning threads), the
    /// ThreadPool's Parallel.For. The pure-C# kernels fork through this class, the quantized
    /// matmuls' RunParallelBlocks or the Core kernels' CpuParallel hook, all of which read the
    /// same switch, so under TS_CPU_POOL=0 none of them starts a spinning thread. (Loops that
    /// still call CpuWorkerPool.Shared directly - DirectOps.RowsParallel, MiniMax-H3's own
    /// kernels - do not follow the switch.)
    ///
    /// Widths: <see cref="Shared"/> is <see cref="CpuWorkerPool.Shared"/> (TS_CPU_THREADS threads;
    /// by default every logical processor up to 8, half of them above that), or an uncapped
    /// Parallel.For sized as <see cref="Environment.ProcessorCount"/> under TS_CPU_POOL=0. A
    /// <see cref="Dedicated"/> set (the Qwen-Image VAE's wide pool, TS_CPU_GEMM_THREADS) keeps its
    /// width either way: its own pool, or Parallel.For capped at that many threads.
    /// </summary>
    internal sealed class CpuWorkers
    {
        /// <summary>False under TS_CPU_POOL=0: every loop forks on Parallel.For instead.</summary>
        internal static readonly bool PoolEnabled = Environment.GetEnvironmentVariable("TS_CPU_POOL") != "0";

        /// <summary>The default for every managed CPU kernel. Creating it starts no thread; the
        /// shared pool starts on the first parallel loop.</summary>
        internal static CpuWorkers Shared { get; } = PoolEnabled
            ? new CpuWorkers(() => CpuWorkerPool.Shared)
            : new CpuWorkers(Environment.ProcessorCount, new ParallelOptions());

        private readonly Lazy<CpuWorkerPool> _pool;   // null: Parallel.For
        private readonly ParallelOptions _options;
        private readonly int _width;

        private CpuWorkers(Func<CpuWorkerPool> pool) => _pool = new Lazy<CpuWorkerPool>(pool);

        private CpuWorkers(int width, ParallelOptions options)
        {
            _width = Math.Max(1, width);
            _options = options;
        }

        /// <summary>A pool of its own, <paramref name="threads"/> wide (created on the first
        /// parallel loop); Parallel.For capped at that width under TS_CPU_POOL=0.</summary>
        internal static CpuWorkers Dedicated(int threads) => PoolEnabled
            ? new CpuWorkers(() => new CpuWorkerPool(threads))
            : OnThreadPool(threads);

        /// <summary>Always <paramref name="pool"/>, whatever TS_CPU_POOL says (tests pin widths).</summary>
        internal static CpuWorkers On(CpuWorkerPool pool)
        {
            ArgumentNullException.ThrowIfNull(pool);
            return new CpuWorkers(() => pool);
        }

        /// <summary>Always Parallel.For capped at <paramref name="threads"/> (the TS_CPU_POOL=0 path).</summary>
        internal static CpuWorkers OnThreadPool(int threads)
            => new CpuWorkers(threads, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, threads) });

        /// <summary>Threads a loop runs on; callers size their task split from it.</summary>
        internal int ThreadCount => _pool != null ? _pool.Value.ThreadCount : _width;

        /// <summary>Run <paramref name="body"/> for every index in [0, count) and return once all
        /// have completed (inline for a single index). A failure is rethrown as an AggregateException.</summary>
        internal void For(int count, Action<int> body)
        {
            if (count <= 0) return;
            if (count == 1) { body(0); return; }
            if (_pool != null) _pool.Value.For(count, body);
            else Parallel.For(0, count, _options, body);
        }
    }
}
