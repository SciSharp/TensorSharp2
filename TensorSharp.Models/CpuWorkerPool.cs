using System;
using System.Threading;

namespace TensorSharp.Models
{
    /// <summary>
    /// Persistent spin-then-park worker pool for the pure-C# CPU backend.
    ///
    /// Why not <see cref="System.Threading.Tasks.Parallel"/>: a decoded token
    /// issues a few hundred matmuls, each only ~1 ms of work spread over every
    /// core, so each one is its own fork/join. On the ThreadPool that means
    /// waking parked threads through the kernel and parking them again, and the
    /// wakeup latency for a large pool is the same order as the work itself.
    /// Measured on a 122-core allocation, a 4096x4096 q8_0 matvec went
    /// 2.3 -> 11.7 GB/s from 1 to 16 threads and then FLATLINED at ~13 through
    /// 122, while q4_0 went backwards (4.8 at 16 -> 3.3 at 122). End-to-end
    /// decode showed the same ceiling: 0.5 tok/s at 1 thread, 2.3 by 8, and no
    /// further gain from the remaining 114 cores.
    ///
    /// This pool keeps its workers alive across jobs and hands them work through
    /// a generation counter they spin on, so submitting costs an interlocked
    /// write instead of a kernel round trip. Blocks are claimed with an atomic
    /// counter, which also balances the tail without extra tasks. Workers spin
    /// for a bounded interval and only then park on a monitor, so an idle
    /// process does not burn cores, and the park happens under the same lock the
    /// submitter pulses so a wakeup can never be missed.
    /// </summary>
    internal sealed class CpuWorkerPool
    {
        public static readonly CpuWorkerPool Shared = new CpuWorkerPool(DefaultThreadCount());

        // Waking a parked worker is expensive at this width, whatever the
        // primitive, so the design spins long enough that the steady state never
        // parks at all. Measured on a 122-core allocation, decode tok/s:
        //
        //     spin 4096, monitor park : 7.0   <- this
        //     spin 256,  monitor park : 0.1
        //     spin 512,  event park   : 0.3
        //
        // Parking early loses because a job has to wake every worker, and 121
        // kernel wakeups cost far more than the ~60 us of work being handed out.
        // Spinning is not free either - the pool holds a thread per core while the
        // REST of the CPU path (attention, elementwise ops in the core assembly)
        // still runs on the ThreadPool, so spinners burn the CPU quota that work
        // needs. That is what ThreadCount is for: the pool deliberately does NOT
        // take every core.
        private static readonly int SpinsBeforePark = EnvInt("TS_CPU_SPIN", 4096);
        private const int ParkPollMs = 5;

        private static int EnvInt(string name, int fallback)
            => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v >= 0
                ? v
                : fallback;

        // Set on ANY thread that is inside a job - the submitter included, since
        // Monitor is recursive and a nested For() from the submitting thread would
        // otherwise re-enter _submitLock and clobber the job in flight.
        [ThreadStatic] private static bool _inJob;

        private readonly object _submitLock = new object();
        private readonly object _sleepLock = new object();
        private readonly int _workers;

        private Action<int> _body;
        private int _blockCount;
        private int _nextBlock;
        private int _activeCount;
        private int _parked;
        private int _ready;
        private bool _started;   // guarded by _submitLock
        private long _generation;
        private Exception _error;
        private volatile bool _shutdown;

        private CpuWorkerPool(int totalThreads)
        {
            _workers = Math.Max(0, totalThreads - 1);
            for (int i = 0; i < _workers; i++)
            {
                var t = new Thread(WorkerLoop, 512 * 1024)
                {
                    IsBackground = true,
                    Name = "ts-cpu-" + i,
                };
                t.Start();
            }
            // NB: the workers are NOT waited for here. WorkerLoop touches this
            // type's statics, so a thread that reaches them while the type
            // initializer is still running blocks on it - and the initializer is
            // what runs this constructor. Waiting here deadlocked every worker
            // against the cctor. The handshake happens on the first For()
            // instead, by which point the initializer has completed.
        }

        private static int DefaultThreadCount()
        {
            string s = Environment.GetEnvironmentVariable("TS_CPU_THREADS");
            if (int.TryParse(s, out int n) && n > 0) return n;
            return DefaultThreadCountFor(Environment.ProcessorCount);
        }

        /// <summary>
        /// How wide the pool should be. NOT every core: the quantized matmuls this
        /// pool runs are only part of the CPU path, the rest still uses the
        /// ThreadPool, and workers spin between jobs. Taking every core made the
        /// spinners starve that other work and regressed prefill. It also buys
        /// nothing - past about a quarter of this machine the matmuls stop scaling,
        /// so the extra threads only add spin.
        /// </summary>
        internal static int DefaultThreadCountFor(int processors)
            => processors <= 8 ? Math.Max(1, processors) : Math.Max(8, processors / 2);

        /// <summary>Threads that will run a job, including the submitting one.</summary>
        public int ThreadCount => _workers + 1;

        /// <summary>
        /// Run <paramref name="body"/> for every block index in [0, blockCount)
        /// across this thread and the pool's, returning once all have completed.
        /// </summary>
        public void For(int blockCount, Action<int> body)
        {
            if (blockCount <= 0) return;

            // Nested calls, and a job already in flight, both run inline: this is
            // a flat one-job-at-a-time pool, and recursing into it would either
            // deadlock or oversubscribe.
            if (blockCount == 1 || _workers == 0 || _inJob || !Monitor.TryEnter(_submitLock))
            {
                for (int i = 0; i < blockCount; i++) body(i);
                return;
            }

            _inJob = true;
            try
            {
                // Every worker must have latched a generation before the first
                // job is published. One that started late would otherwise latch a
                // generation already in flight, skip that job, and never
                // decrement the completion count - hanging the submitter.
                if (!_started)
                {
                    var startSpin = new SpinWait();
                    while (Volatile.Read(ref _ready) != _workers) startSpin.SpinOnce(-1);
                    _started = true;
                }

                _body = body;
                _blockCount = blockCount;
                _error = null;
                Volatile.Write(ref _nextBlock, 0);
                Volatile.Write(ref _activeCount, _workers);

                // Full fence: publishes the fields above to every worker that
                // subsequently observes the new generation.
                Interlocked.Increment(ref _generation);

                // Only pay for a wakeup when someone is actually parked. The
                // increment above is a full fence and a worker publishes _parked
                // before re-checking the generation, so a worker this misses is
                // guaranteed to see the new generation and not wait.
                if (Volatile.Read(ref _parked) > 0)
                {
                    lock (_sleepLock) Monitor.PulseAll(_sleepLock);
                }

                RunBlocks();

                var spin = new SpinWait();
                while (Volatile.Read(ref _activeCount) != 0) spin.SpinOnce(-1);

                _body = null;
                Exception error = _error;
                if (error != null)
                {
                    _error = null;
                    throw new AggregateException(error);
                }
            }
            finally
            {
                _inJob = false;
                Monitor.Exit(_submitLock);
            }
        }

        private void RunBlocks()
        {
            Action<int> body = _body;
            int count = _blockCount;
            try
            {
                while (true)
                {
                    int i = Interlocked.Increment(ref _nextBlock) - 1;
                    if (i >= count) return;
                    body(i);
                }
            }
            catch (Exception ex)
            {
                // Claim the rest of the range so the other threads wind down,
                // and hand the first failure back to the submitter.
                Interlocked.Exchange(ref _nextBlock, count);
                Interlocked.CompareExchange(ref _error, ex, null);
            }
        }

        private void WorkerLoop()
        {
            long seen = Volatile.Read(ref _generation);
            Interlocked.Increment(ref _ready);
            _inJob = true;
            int spins = 0;

            while (!_shutdown)
            {
                long gen = Volatile.Read(ref _generation);
                if (gen != seen)
                {
                    seen = gen;
                    RunBlocks();
                    Interlocked.Decrement(ref _activeCount);
                    spins = 0;
                    continue;
                }

                if (++spins < SpinsBeforePark)
                {
                    Thread.SpinWait(64);
                    continue;
                }

                // Park under the lock the submitter pulses, so check-then-wait
                // cannot race with a job being published.
                Interlocked.Increment(ref _parked);
                lock (_sleepLock)
                {
                    if (Volatile.Read(ref _generation) == seen && !_shutdown)
                        Monitor.Wait(_sleepLock, ParkPollMs);
                }
                Interlocked.Decrement(ref _parked);
                spins = 0;
            }
        }
    }
}
