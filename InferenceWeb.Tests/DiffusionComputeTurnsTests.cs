// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// DiffusionComputeTurns: the diffusion scheduler holds GpuComputeLock for a whole denoising block
// and steps aside before a forward when a Jev read or an image encode is waiting. These pin the
// turn-taking itself with a plain lock object and sleeping "forwards", so they need no weights.
using System.Collections.Concurrent;

namespace InferenceWeb.Tests;

public sealed class DiffusionComputeTurnsTests
{
    private const int ForwardMs = 40;

    /// <summary>A block of <paramref name="forwards"/> forwards under the lock, yielding before each,
    /// the way DiffusionBatchScheduler drives RunBlockBatched.</summary>
    private static Thread StartBlock(object gpu, DiffusionComputeTurns turns, int forwards, Action<int> onForward,
        ManualResetEventSlim? started = null)
    {
        var thread = new Thread(() =>
        {
            lock (gpu)
            {
                started?.Set();
                for (int f = 0; f < forwards; f++)
                {
                    turns.Yield();
                    onForward(f);
                    Thread.Sleep(ForwardMs);
                }
            }
        }) { IsBackground = true };
        thread.Start();
        return thread;
    }

    [Fact]
    public void Yield_WithNothingRegistered_KeepsTheLock()
    {
        var gpu = new object();
        var turns = new DiffusionComputeTurns(gpu);
        Assert.False(turns.Yield());   // not held: nothing to yield
        lock (gpu)
        {
            Assert.False(turns.Yield());
            Assert.True(Monitor.IsEntered(gpu));
        }
        Assert.False(turns.HasJobs);
    }

    // The regression this exists for: a Jev read issued mid-chat used to wait for every remaining
    // forward of the block (~100 s on the cpu backend). It must now wait for the forward in progress.
    [Fact]
    public void WaitingJob_RunsBeforeTheNextForward()
    {
        var gpu = new object();
        var turns = new DiffusionComputeTurns(gpu);
        const int forwards = 20;
        int completed = 0;
        using var midBlock = new ManualResetEventSlim();
        Thread block = StartBlock(gpu, turns, forwards, f =>
        {
            Volatile.Write(ref completed, f);
            if (f == 3) midBlock.Set();
        });
        Assert.True(midBlock.Wait(TimeSpan.FromSeconds(10)));
        int issuedAt = Volatile.Read(ref completed);
        int ranAt;
        turns.Enter();
        try { ranAt = Volatile.Read(ref completed); }
        finally { turns.Exit(); }
        Assert.True(block.Join(TimeSpan.FromSeconds(30)));
        // The forward in progress when the job registered finishes; the next one waits for the job.
        Assert.InRange(ranAt - issuedAt, 0, 1);
        Assert.False(turns.HasJobs);
    }

    // Many chats share one block, so "fair to Jev" is per forward, not per block; the reverse is
    // that a queue of jobs arriving back to back must still let the block move between them.
    [Fact]
    public void BackToBackJobs_AlternateWithTheBlock()
    {
        var gpu = new object();
        var turns = new DiffusionComputeTurns(gpu);
        var log = new ConcurrentQueue<char>();
        using var started = new ManualResetEventSlim();
        const int forwards = 8;
        Thread block = StartBlock(gpu, turns, forwards, _ => log.Enqueue('F'), started);
        Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
        int jobs = 0;
        while (block.IsAlive && jobs < 200)
        {
            turns.Enter();
            try
            {
                log.Enqueue('J');
                Thread.Sleep(5);
                jobs++;
            }
            finally { turns.Exit(); }
        }
        Assert.True(block.Join(TimeSpan.FromSeconds(30)));
        string order = new(log.ToArray());
        Assert.Equal(forwards, order.Count(c => c == 'F'));
        // Until the block ends, every job is followed by a forward before the next job.
        string duringBlock = order[..(order.LastIndexOf('F') + 1)];
        Assert.DoesNotContain("JJ", duringBlock);
        Assert.Contains("FJF", duringBlock);
    }

    [Fact]
    public void CancelledWhileWaiting_ThrowsWithoutTheLockAndUnregisters()
    {
        var gpu = new object();
        var turns = new DiffusionComputeTurns(gpu);
        using var holding = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        bool yieldedAfterCancel = true;
        var holder = new Thread(() =>
        {
            lock (gpu)
            {
                holding.Set();
                release.Wait();   // one long forward, no yield point
                yieldedAfterCancel = turns.Yield();
            }
        }) { IsBackground = true };
        holder.Start();
        Assert.True(holding.Wait(TimeSpan.FromSeconds(10)));
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        Assert.ThrowsAny<OperationCanceledException>(() => turns.Enter(cts.Token));
        Assert.False(Monitor.IsEntered(gpu));
        Assert.False(turns.HasJobs);
        release.Set();
        Assert.True(holder.Join(TimeSpan.FromSeconds(10)));
        Assert.False(yieldedAfterCancel);
    }

    [Fact]
    public void JobQueuedBehindAYieldingBlock_CanBeCancelled()
    {
        var gpu = new object();
        var turns = new DiffusionComputeTurns(gpu);
        using var started = new ManualResetEventSlim();
        Thread block = StartBlock(gpu, turns, 4, _ => { }, started);
        Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
        turns.Enter();   // the block steps aside for this job and waits to resume
        try
        {
            var second = Task.Run(() =>
            {
                using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
                turns.Enter(cts.Token);   // defers to the resuming block, then gives up
                turns.Exit();
            });
            Assert.ThrowsAny<OperationCanceledException>(() => second.GetAwaiter().GetResult());
        }
        finally { turns.Exit(); }
        Assert.True(block.Join(TimeSpan.FromSeconds(30)));
        Assert.False(turns.HasJobs);
    }

    // The vision encoder releases the lock between its blocks (ModelBase.YieldGpuComputeLock). A
    // block that stepped aside for that job must not slip a forward into each of those gaps: on the
    // cpu backend a forward is a second or more, and an image encode has dozens of gaps.
    [Fact]
    public void JobSteppingOutOfTheLock_IsNotInterruptedByTheBlock()
    {
        var gpu = new object();
        var turns = new DiffusionComputeTurns(gpu);
        var log = new ConcurrentQueue<char>();
        using var midBlock = new ManualResetEventSlim();
        const int forwards = 6;
        Thread block = StartBlock(gpu, turns, forwards, f =>
        {
            log.Enqueue('F');
            if (f == 1) midBlock.Set();
        });
        Assert.True(midBlock.Wait(TimeSpan.FromSeconds(10)));
        turns.Enter();
        try
        {
            for (int b = 0; b < 5; b++)
            {
                log.Enqueue('E');
                Monitor.Exit(gpu);   // what YieldGpuComputeLock does between encoder blocks
                Thread.Sleep(10);
                Monitor.Enter(gpu);
            }
        }
        finally { turns.Exit(); }
        Assert.True(block.Join(TimeSpan.FromSeconds(30)));
        string order = new(log.ToArray());
        Assert.Contains("EEEEE", order);
        Assert.Equal(forwards, order.Count(c => c == 'F'));
    }

    [Fact]
    public void WithoutABlock_JobsStillExcludeEachOther()
    {
        var gpu = new object();
        var turns = new DiffusionComputeTurns(gpu);
        int inside = 0, overlaps = 0;
        Parallel.For(0, 32, _ =>
        {
            turns.Enter();
            try
            {
                if (Interlocked.Increment(ref inside) != 1) Interlocked.Increment(ref overlaps);
                Thread.SpinWait(2000);
                Interlocked.Decrement(ref inside);
            }
            finally { turns.Exit(); }
        });
        Assert.Equal(0, overlaps);
        Assert.False(turns.HasJobs);
    }
}
