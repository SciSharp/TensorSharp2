// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) whole-model engine for the direct-CUDA backend.
//
// This is the GPU counterpart of DeepSeek4CpuExecutor: an imperative,
// single-sequence executor that keeps the quantized weights resident in
// device memory (layer-split across all visible GPUs by cumulative weight
// bytes, so a model larger than one GPU's VRAM is hosted across several) and
// drives the DSV4-specific kernels in tensorsharp_dsv4_kernels.cu plus the
// generic quantized matmul kernels in tensorsharp_kernels.cu, entirely
// through the CUDA driver API + cuBLAS. It has no dependency on ggml.
//
// The model-side executor (DeepSeek4CudaExecutor in TensorSharp.Models)
// parses the split-GGUF shards and hands this engine host pointers into the
// mapped tensor data plus pre-dequantized F32 copies of the small tensors
// (norms, gates, sinks, APE tables, the BF16 router) and precomputed RoPE
// cos/sin tables; this engine owns everything device-side.
//
// Hidden state flows device to device at layer-group boundaries via
// cuMemcpyPeerAsync ordered with events, so prefill chunks pipeline across
// GPUs naturally. Caches are F16 (parity with the native executor and
// llama.cpp); compressor state rings are F32.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe class Dsv4CudaEngine : IDisposable
    {
        private const int HC = 4;
        private const int HcMixDim = (2 + HC) * HC; // 24
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;
        private const int Q81BlockBytes = 36;

        // ggml type ids this engine dispatches on
        private const int TF32 = 0, TF16 = 1, TQ8_0 = 8, TQ6_K = 14, TIQ3_S = 21, TBF16 = 30, TMXFP4 = 39;

        public struct QuantWeightDesc
        {
            /// <summary>Host staging pointer. Zero when the weight streams
            /// straight from <see cref="Source"/> into VRAM.</summary>
            public IntPtr HostPtr;
            /// <summary>Byte source the upload reads from when
            /// <see cref="HostPtr"/> is zero (the GGUF shard, normally).</summary>
            public IDsv4WeightSource Source;
            public long SourceOffset;
            public int GgmlType;
            public int Ne0;
            public int Ne1;
            public int Ne2;
            public long RowBytes;
            public string Name;
            public bool IsValid => HostPtr != IntPtr.Zero || Source != null;
            public long TotalBytes => (long)Ne1 * Math.Max(1, Ne2) * RowBytes;
        }

        public sealed class LayerDesc
        {
            public int Ratio;
            public float ClampExp, ClampShexp;
            public QuantWeightDesc WqA, WqB, Wkv, WoA, WoB;
            public QuantWeightDesc CompWkv, CompWgate, IdxProj, IdxQB, IdxCompWkv, IdxCompWgate;
            public QuantWeightDesc GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
            public float[] AttnNorm, QANorm, KvNorm, Sinks;
            public float[] HcAttnFn, HcAttnScale, HcAttnBase, HcFfnFn, HcFfnScale, HcFfnBase;
            public float[] CompApe, CompNorm, IdxCompApe, IdxCompNorm;
            public float[] GateInp, ExpProbsBias, FfnNorm;
            public int[] Tid2Eid;
        }

        public sealed class ModelDesc
        {
            public int NLayer, NEmbd, NHead, HeadDim, NRot, QLoraRank, OGroups, OLoraRank, NSwa, NVocab;
            public int NExpert, NExpertUsed, NFfExp, HashLayerCount;
            public int IdxNHead, IdxHeadSize, IdxTopK;
            public int HcSinkhornIters;
            public float RmsEps, HcEps, ExpertWeightsScale;
            public bool ExpertWeightsNorm;
            public int NCtx, NUbatch;
            public QuantWeightDesc TokEmbd, Output;
            public float[] OutputNorm, HcHeadFn, HcHeadScale, HcHeadBase;
            public float[] RopeRawTable, RopeCompTable; // [nCtx * nRot] interleaved cos/sin
            public LayerDesc[] Layers;
        }

        internal sealed class UploadJob
        {
            public IntPtr Dst;
            public IDsv4WeightSource Src;
            public long SrcOffset;
            public long Bytes;
        }

        private struct DevQW
        {
            public IntPtr Ptr;
            public int Type;
            public int Ne0;
            public int Ne1;
            public long RowBytes;
        }

        private sealed class DevLayer
        {
            public int Device;
            public int Ratio;
            public float ClampExp, ClampShexp;
            public DevQW WqA, WqB, Wkv, WoA, WoB;
            public DevQW CompWkv, CompWgate, IdxProj, IdxQB, IdxCompWkv, IdxCompWgate;
            public DevQW GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
            public IntPtr AttnNorm, QANorm, KvNorm, Sinks;
            public IntPtr HcAttnFn, HcAttnScale, HcAttnBase, HcFfnFn, HcFfnScale, HcFfnBase;
            public IntPtr CompApe, CompNorm, IdxCompApe, IdxCompNorm;
            public IntPtr GateInp, ExpProbsBias, FfnNorm, Tid2Eid;
            public IntPtr RingK, CompK, LidK;
            public IntPtr HistKv, HistScore, LidHistKv, LidHistScore;
            public int ShFf;
        }

        private sealed class Dev
        {
            public int Ordinal;
            public CudaAllocator Alloc;
            public Dsv4Kernels DK;
            public IntPtr Event;
            public IntPtr Arena;
            public long ArenaBytes;
            public long ArenaUsed;
            public IntPtr RopeRaw, RopeComp;
            public bool NeedsTokens;
            public IntPtr TokensDev0, TokensDev1;
            public IntPtr TokEv0, TokEv1; // guards pinned-buffer reuse per parity
            // Layer-boundary handoff staging: this device's outgoing hidden
            // streams go DtoH into BoundaryPinned on this device's stream, then
            // HtoD on the next device's stream. Direct cuMemcpyPeerAsync is NOT
            // used: on several cloud PCIe topologies peer DMA reports available
            // but silently transfers corrupt data (see CudaP2PCommunicator's
            // round-trip self-test for the same failure mode).
            public IntPtr BoundaryPinned;
            public IntPtr XsReadyEv;   // recorded on this stream after the DtoH
            public IntPtr CopyDoneEv;  // recorded on the DST stream after the HtoD

            // scratch
            public IntPtr Xs, XsOut, Cur, Inv, Mixes, Pre, Post, Comb;
            public IntPtr Qr, Q, KvRaw, StKv, StScore, LidStKv, LidStScore;
            public IntPtr Iq, Iw, IdxScores, TopkIdx, TopkCnt;
            public IntPtr AttnO, OGrouped, OGroupedOut, OG, AttnOut, FfnOut;
            public IntPtr RouterLogits, Sel, SelW, Counts, Offsets, Cursors, RowOfSlot, SlotToken;
            public IntPtr ActQ8A, ActQ8B, ExpGate, ExpUp, ExpDown, ShGate, ShUp, ShDown;
            // dense split-q8_1 activation scratch for the register-staged expert
            // kernels (A = per-token rows into gate/up, B = packed rows into down)
            public IntPtr SplitQsA, SplitDA, SplitQsB, SplitDB;
            public IntPtr WF16, AF16;
            public long WF16Elems;
            public List<IntPtr> OwnedAllocs = new List<IntPtr>();
            // Weights that stream from disk: planned during the (I/O-free)
            // arena-layout pass, then transferred by the loader thread pool.
            public List<UploadJob> UploadPlan = new List<UploadJob>();
            public UploadJob[] UploadJobs;
            public int UploadCursor;

            public IntPtr Stream => Alloc.Stream.Handle;
            public void MakeCurrent() => Alloc.Context.MakeCurrent();
        }

        private readonly ModelDesc _m;
        private readonly Dev[] _devs;
        private readonly DevLayer[] _layers;
        private readonly int _ringRaw;
        private readonly int _compRowsCsa;
        private readonly int _compRowsHca;
        private readonly int _lastDev;
        private IntPtr _outputW;          // lm head (device ptr on last dev)
        private DevQW _outputQW;
        private DevQW _tokEmbdQW;
        private IntPtr _outputNorm, _hcHeadFn, _hcHeadScale, _hcHeadBase;
        private IntPtr _logitsDev;
        private IntPtr _pinnedTokens0, _pinnedTokens1;
        private int _chunkParity;
        private readonly int _perf;
        private readonly bool _syncDebug;
        private readonly float[] _logitsHost;

        public int NPast { get; private set; }
        public int ContextSize => _m.NCtx;

        public Dsv4CudaEngine(ModelDesc m, int nGpu)
        {
            _m = m ?? throw new ArgumentNullException(nameof(m));
            if (m.HeadDim != 512)
                throw new NotSupportedException($"DSV4 CUDA engine requires head_dim 512, got {m.HeadDim}.");
            if (m.IdxNHead > 0 && m.IdxHeadSize != 128)
                throw new NotSupportedException($"DSV4 CUDA engine requires indexer key_length 128, got {m.IdxHeadSize}.");
            if (m.NExpertUsed > 16)
                throw new NotSupportedException($"DSV4 CUDA engine supports at most 16 experts per token, got {m.NExpertUsed}.");

            _perf = EnvInt("TS_DSV4_PERF", 0);
            _syncDebug = EnvInt("TS_DSV4_CUDA_SYNCDBG", 0) != 0;
            _logitsHost = new float[m.NVocab];

            CudaDriverApi.cuInit(0);
            CudaDriverApi.cuDeviceGetCount(out int devCount).ThrowOnError();
            int useDevs = nGpu > 0 ? Math.Min(nGpu, devCount) : devCount;
            if (useDevs < 1)
                throw new InvalidOperationException("No CUDA devices available for the DSV4 engine.");

            _ringRaw = Pad(m.NSwa + m.NUbatch, 256);
            _compRowsCsa = m.NCtx / CsaRatio + 1;
            _compRowsHca = m.NCtx / HcaRatio + 1;

            // ---- layer placement: contiguous ranges balanced by quantized bytes ----
            var layerBytes = new long[m.NLayer];
            long totalBytes = 0;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = m.Layers[il];
                long b = 0;
                foreach (var qw in EnumerateQuantWeights(L))
                    b += qw.TotalBytes;
                layerBytes[il] = b;
                totalBytes += b;
            }

            var assignment = new int[m.NLayer];
            {
                long target = totalBytes / useDevs;
                int dev = 0;
                long acc = 0;
                for (int il = 0; il < m.NLayer; il++)
                {
                    assignment[il] = dev;
                    acc += layerBytes[il];
                    if (acc >= target && dev < useDevs - 1 && il < m.NLayer - 1)
                    {
                        dev++;
                        acc = 0;
                    }
                }
            }
            _lastDev = useDevs - 1;

            // ---- devices ----
            _devs = new Dev[useDevs];
            for (int d = 0; d < useDevs; d++)
            {
                var dev = new Dev { Ordinal = d };
                dev.Alloc = new CudaAllocator(d);
                dev.MakeCurrent();
                dev.DK = Dsv4Kernels.Create();
                CudaDriverApi.cuEventCreate(out IntPtr ev, 0x02 /*CU_EVENT_DISABLE_TIMING*/).ThrowOnError();
                dev.Event = ev;
                _devs[d] = dev;
            }

            // per-device boundary staging (pinned + events)
            long xsBytes = (long)m.NUbatch * HC * m.NEmbd * 4;
            for (int d = 0; d < useDevs; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                CudaDriverApi.cuMemHostAlloc(out IntPtr pinnedXs, new UIntPtr((ulong)xsBytes), 0x1 /*PORTABLE*/).ThrowOnError();
                dev.BoundaryPinned = pinnedXs;
                CudaDriverApi.cuEventCreate(out IntPtr xr, 0x02).ThrowOnError();
                CudaDriverApi.cuEventCreate(out IntPtr cd, 0x02).ThrowOnError();
                dev.XsReadyEv = xr;
                dev.CopyDoneEv = cd;
            }

            // ---- arena sizing per device (weights + small tensors + caches + rope) ----
            var arenaNeed = new long[useDevs];
            for (int il = 0; il < m.NLayer; il++)
            {
                int d = assignment[il];
                var L = m.Layers[il];
                foreach (var qw in EnumerateQuantWeights(L))
                    arenaNeed[d] += Align(qw.TotalBytes);
                foreach (var arr in EnumerateF32Arrays(L))
                    arenaNeed[d] += Align((long)(arr?.Length ?? 0) * 4);
                if (L.Tid2Eid != null)
                    arenaNeed[d] += Align((long)L.Tid2Eid.Length * 4);
                arenaNeed[d] += CacheBytes(L.Ratio);
            }
            arenaNeed[0] += Align(m.TokEmbd.TotalBytes);
            arenaNeed[_lastDev] += Align(m.Output.TotalBytes)
                + Align((long)m.OutputNorm.Length * 4) + Align((long)m.HcHeadFn.Length * 4)
                + Align((long)m.HcHeadScale.Length * 4) + Align((long)m.HcHeadBase.Length * 4);
            for (int d = 0; d < useDevs; d++)
                arenaNeed[d] += Align((long)m.RopeRawTable.Length * 4) + Align((long)m.RopeCompTable.Length * 4);

            for (int d = 0; d < useDevs; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                dev.ArenaBytes = arenaNeed[d];
                CudaDriverApi.cuMemAlloc(out IntPtr arena, new UIntPtr((ulong)Math.Max(arenaNeed[d], 256))).ThrowOnError();
                dev.Arena = arena;
                dev.ArenaUsed = 0;
            }

            // ---- upload weights (parallel across devices) ----
            _layers = new DevLayer[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
                _layers[il] = new DevLayer { Device = assignment[il], Ratio = m.Layers[il].Ratio };

            var perDevLayers = new List<int>[useDevs];
            for (int d = 0; d < useDevs; d++)
                perDevLayers[d] = new List<int>();
            for (int il = 0; il < m.NLayer; il++)
                perDevLayers[assignment[il]].Add(il);

            var uploadSw = Stopwatch.StartNew();
            Parallel.For(0, useDevs, d =>
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                foreach (int il in perDevLayers[d])
                    UploadLayer(dev, il);
                if (d == 0)
                    _tokEmbdQW = UploadQuant(dev, m.TokEmbd);
                if (d == _lastDev)
                {
                    _outputQW = UploadQuant(dev, m.Output);
                    _outputW = _outputQW.Ptr;
                    _outputNorm = UploadF32(dev, m.OutputNorm);
                    _hcHeadFn = UploadF32(dev, m.HcHeadFn);
                    _hcHeadScale = UploadF32(dev, m.HcHeadScale);
                    _hcHeadBase = UploadF32(dev, m.HcHeadBase);
                }
                dev.RopeRaw = UploadF32(dev, m.RopeRawTable);
                dev.RopeComp = UploadF32(dev, m.RopeCompTable);
            });

            // Weights sourced from disk were only *placed* above; move the bytes
            // now, with the reader concurrency the filesystem actually likes.
            RunStreamedUploads();

            // ---- scratch + tokens + logits ----
            for (int d = 0; d < useDevs; d++)
                AllocateScratch(_devs[d]);

            _devs[0].NeedsTokens = true;
            for (int il = 0; il < Math.Min(m.HashLayerCount, m.NLayer); il++)
                _devs[assignment[il]].NeedsTokens = true;
            foreach (var dev in _devs)
            {
                if (!dev.NeedsTokens)
                    continue;
                dev.MakeCurrent();
                dev.TokensDev0 = AllocDev(dev, (long)m.NUbatch * 4);
                dev.TokensDev1 = AllocDev(dev, (long)m.NUbatch * 4);
                CudaDriverApi.cuEventCreate(out IntPtr te0, 0x02).ThrowOnError();
                CudaDriverApi.cuEventCreate(out IntPtr te1, 0x02).ThrowOnError();
                dev.TokEv0 = te0;
                dev.TokEv1 = te1;
            }

            var last = _devs[_lastDev];
            last.MakeCurrent();
            _logitsDev = AllocDev(last, (long)m.NVocab * 4);

            _devs[0].MakeCurrent();
            // CU_MEMHOSTALLOC_PORTABLE (0x1): the pinned token buffers are copied
            // from by every device that needs token ids, not just device 0.
            CudaDriverApi.cuMemHostAlloc(out _pinnedTokens0, new UIntPtr((ulong)m.NUbatch * 4), 0x1).ThrowOnError();
            CudaDriverApi.cuMemHostAlloc(out _pinnedTokens1, new UIntPtr((ulong)m.NUbatch * 4), 0x1).ThrowOnError();

            Reset();

            double gib = totalBytes / (1024.0 * 1024 * 1024);
            Console.Error.WriteLine(
                $"[dsv4-cuda] {gib:F1} GiB of weights resident across {useDevs} GPU(s) " +
                $"(layer split {string.Join("/", CountPerDev(assignment, useDevs))}), uploaded in {uploadSw.Elapsed.TotalSeconds:F1}s " +
                $"(n_ctx={m.NCtx}, ubatch={m.NUbatch})");
        }

        private static int[] CountPerDev(int[] assignment, int nDev)
        {
            var counts = new int[nDev];
            foreach (int d in assignment)
                counts[d]++;
            return counts;
        }

        private static int EnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        private static int Pad(int v, int p) => (v + p - 1) / p * p;
        private static long Align(long v) => (v + 255) & ~255L;

        private IEnumerable<QuantWeightDesc> EnumerateQuantWeights(LayerDesc l)
        {
            yield return l.WqA;
            yield return l.WqB;
            yield return l.Wkv;
            yield return l.WoA;
            yield return l.WoB;
            if (l.CompWkv.IsValid) yield return l.CompWkv;
            if (l.CompWgate.IsValid) yield return l.CompWgate;
            if (l.IdxProj.IsValid) yield return l.IdxProj;
            if (l.IdxQB.IsValid) yield return l.IdxQB;
            if (l.IdxCompWkv.IsValid) yield return l.IdxCompWkv;
            if (l.IdxCompWgate.IsValid) yield return l.IdxCompWgate;
            yield return l.GateExps;
            yield return l.UpExps;
            yield return l.DownExps;
            yield return l.GateShexp;
            yield return l.UpShexp;
            yield return l.DownShexp;
        }

        private static IEnumerable<float[]> EnumerateF32Arrays(LayerDesc l)
        {
            yield return l.AttnNorm; yield return l.QANorm; yield return l.KvNorm; yield return l.Sinks;
            yield return l.HcAttnFn; yield return l.HcAttnScale; yield return l.HcAttnBase;
            yield return l.HcFfnFn; yield return l.HcFfnScale; yield return l.HcFfnBase;
            yield return l.CompApe; yield return l.CompNorm; yield return l.IdxCompApe; yield return l.IdxCompNorm;
            yield return l.GateInp; yield return l.ExpProbsBias; yield return l.FfnNorm;
        }

        private long CacheBytes(int ratio)
        {
            int hd = _m.HeadDim;
            long b = Align((long)_ringRaw * hd * 2);
            if (ratio == CsaRatio)
            {
                b += Align((long)_compRowsCsa * hd * 2);
                b += Align((long)_compRowsCsa * _m.IdxHeadSize * 2);
                b += 2 * Align(2L * CsaRatio * 2 * hd * 4);
                b += 2 * Align(2L * CsaRatio * 2 * _m.IdxHeadSize * 4);
            }
            else if (ratio == HcaRatio)
            {
                b += Align((long)_compRowsHca * hd * 2);
                b += 2 * Align((long)HcaRatio * hd * 4);
            }
            return b;
        }

        // ---- arena sub-allocation + upload helpers (device context must be current) ----

        private static IntPtr ArenaTake(Dev dev, long bytes)
        {
            long aligned = Align(bytes);
            if (dev.ArenaUsed + aligned > dev.ArenaBytes)
                throw new InvalidOperationException($"[dsv4-cuda] arena overflow on device {dev.Ordinal}");
            IntPtr p = (IntPtr)((long)dev.Arena + dev.ArenaUsed);
            dev.ArenaUsed += aligned;
            return p;
        }

        private static DevQW UploadQuant(Dev dev, in QuantWeightDesc qw)
        {
            if (!qw.IsValid)
                return default;
            long bytes = qw.TotalBytes;
            IntPtr p = ArenaTake(dev, bytes);
            if (qw.HostPtr != IntPtr.Zero)
            {
                CudaDriverApi.cuMemcpyHtoD(p, qw.HostPtr, new UIntPtr((ulong)bytes)).ThrowOnError();
            }
            else
            {
                // Planned now, transferred later by RunStreamedUploads so that
                // the (slow) reads run wide instead of one tensor at a time.
                // Split into segments so several loader threads can share one
                // multi-hundred-MB expert stack.
                for (long off = 0; off < bytes; off += UploadSegmentBytes)
                {
                    dev.UploadPlan.Add(new UploadJob
                    {
                        Dst = (IntPtr)((long)p + off),
                        Src = qw.Source,
                        SrcOffset = qw.SourceOffset + off,
                        Bytes = Math.Min(UploadSegmentBytes, bytes - off),
                    });
                }
            }
            return new DevQW { Ptr = p, Type = qw.GgmlType, Ne0 = qw.Ne0, Ne1 = qw.Ne1, RowBytes = qw.RowBytes };
        }

        // Loader tuning. The read side is the bottleneck on network filesystems
        // and its throughput is *not* monotonic in thread count -- MooseFS/FUSE
        // peaks around 8-32 concurrent readers and falls off a cliff above that
        // (2.4 GB/s at 16 threads vs 1.0 GB/s at 96), so the pool is explicitly
        // bounded rather than left to the thread pool's discretion.
        private static readonly int LoaderThreads = Math.Max(1, EnvInt("TS_DSV4_LOAD_THREADS", 16));
        private static readonly int LoaderChunkBytes = Math.Max(1 << 20, EnvInt("TS_DSV4_LOAD_CHUNK_MB", 16) << 20);
        private const long UploadSegmentBytes = 128L << 20;
        private static readonly bool LoaderStats = EnvInt("TS_DSV4_LOAD_STATS", 0) != 0;
        private static long _loaderReadTicks;
        private static long _loaderCopyTicks;

        /// <summary>
        /// Streams every planned weight from its GGUF shard into VRAM: bounded
        /// pool of reader threads, each double-buffering through pinned host
        /// chunks so the file read of chunk N+1 overlaps the HtoD of chunk N.
        /// </summary>
        private void RunStreamedUploads()
        {
            long total = 0;
            int jobCount = 0;
            foreach (var dev in _devs)
            {
                dev.UploadJobs = dev.UploadPlan.ToArray();
                dev.UploadPlan = null;
                dev.UploadCursor = 0;
                jobCount += dev.UploadJobs.Length;
                foreach (var j in dev.UploadJobs)
                    total += j.Bytes;
            }
            if (jobCount == 0)
                return;

            var sw = Stopwatch.StartNew();
            int perDev = Math.Max(1, LoaderThreads / _devs.Length);
            var errors = new List<Exception>();
            var threads = new List<System.Threading.Thread>(perDev * _devs.Length);

            foreach (var devLocal in _devs)
            {
                var dev = devLocal;
                if (dev.UploadJobs.Length == 0)
                    continue;
                for (int w = 0; w < perDev; w++)
                {
                    var t = new System.Threading.Thread(() =>
                    {
                        try
                        {
                            StreamUploadWorker(dev);
                        }
                        catch (Exception ex)
                        {
                            lock (errors)
                                errors.Add(ex);
                        }
                    });
                    t.IsBackground = true;
                    threads.Add(t);
                    t.Start();
                }
            }
            foreach (var t in threads)
                t.Join();
            if (errors.Count > 0)
                throw new AggregateException("[dsv4-cuda] weight streaming failed", errors);

            double gib = total / (1024.0 * 1024 * 1024);
            Console.Error.WriteLine($"[dsv4-cuda] streamed {gib:F1} GiB of weights into VRAM in " +
                $"{sw.Elapsed.TotalSeconds:F1}s ({gib / Math.Max(0.001, sw.Elapsed.TotalSeconds):F2} GiB/s, " +
                $"{threads.Count} readers)");
            if (LoaderStats)
            {
                double readS = System.Threading.Interlocked.Read(ref _loaderReadTicks) / (double)Stopwatch.Frequency;
                double copyS = System.Threading.Interlocked.Read(ref _loaderCopyTicks) / (double)Stopwatch.Frequency;
                Console.Error.WriteLine($"[dsv4-cuda]   thread-seconds: read {readS:F1}s " +
                    $"({gib / Math.Max(0.001, readS) * threads.Count:F2} GiB/s aggregate), copy-wait {copyS:F1}s");
            }
        }

        private static void StreamUploadWorker(Dev dev)
        {
            dev.MakeCurrent();
            int chunk = LoaderChunkBytes;
            IntPtr stream = IntPtr.Zero;
            var bufs = new IntPtr[2];
            var evs = new IntPtr[2];
            var pending = new bool[2];
            try
            {
                CudaDriverApi.cuStreamCreate(out stream, 0x1 /*NON_BLOCKING*/).ThrowOnError();
                for (int i = 0; i < 2; i++)
                {
                    CudaDriverApi.cuMemHostAlloc(out bufs[i], new UIntPtr((ulong)chunk), 0x1 /*PORTABLE*/).ThrowOnError();
                    CudaDriverApi.cuEventCreate(out evs[i], 0x02 /*DISABLE_TIMING*/).ThrowOnError();
                }

                var jobs = dev.UploadJobs;
                int slot = 0;
                while (true)
                {
                    int idx = System.Threading.Interlocked.Increment(ref dev.UploadCursor) - 1;
                    if (idx >= jobs.Length)
                        break;
                    var job = jobs[idx];
                    for (long done = 0; done < job.Bytes; )
                    {
                        long n = Math.Min(chunk, job.Bytes - done);
                        // Reclaim the staging buffer only once its copy retired.
                        if (pending[slot])
                        {
                            long tc = LoaderStats ? Stopwatch.GetTimestamp() : 0;
                            CudaDriverApi.cuEventSynchronize(evs[slot]).ThrowOnError();
                            if (LoaderStats)
                                System.Threading.Interlocked.Add(ref _loaderCopyTicks, Stopwatch.GetTimestamp() - tc);
                            pending[slot] = false;
                        }
                        long t0 = LoaderStats ? Stopwatch.GetTimestamp() : 0;
                        job.Src.Read(job.SrcOffset + done, bufs[slot], n);
                        if (LoaderStats)
                            System.Threading.Interlocked.Add(ref _loaderReadTicks, Stopwatch.GetTimestamp() - t0);
                        CudaDriverApi.cuMemcpyHtoDAsync((IntPtr)((long)job.Dst + done), bufs[slot],
                            new UIntPtr((ulong)n), stream).ThrowOnError();
                        CudaDriverApi.cuEventRecord(evs[slot], stream).ThrowOnError();
                        pending[slot] = true;
                        done += n;
                        slot ^= 1;
                    }
                }
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            }
            finally
            {
                for (int i = 0; i < 2; i++)
                {
                    if (evs[i] != IntPtr.Zero) CudaDriverApi.cuEventDestroy(evs[i]);
                    if (bufs[i] != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(bufs[i]);
                }
                if (stream != IntPtr.Zero) CudaDriverApi.cuStreamDestroy(stream);
            }
        }

        private static IntPtr UploadF32(Dev dev, float[] data)
        {
            if (data == null || data.Length == 0)
                return IntPtr.Zero;
            IntPtr p = ArenaTake(dev, (long)data.Length * 4);
            fixed (float* src = data)
                CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, new UIntPtr((ulong)data.Length * 4)).ThrowOnError();
            return p;
        }

        private static IntPtr UploadI32(Dev dev, int[] data)
        {
            if (data == null || data.Length == 0)
                return IntPtr.Zero;
            IntPtr p = ArenaTake(dev, (long)data.Length * 4);
            fixed (int* src = data)
                CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, new UIntPtr((ulong)data.Length * 4)).ThrowOnError();
            return p;
        }

        private void UploadLayer(Dev dev, int il)
        {
            var src = _m.Layers[il];
            var dst = _layers[il];
            dst.ClampExp = src.ClampExp;
            dst.ClampShexp = src.ClampShexp;

            dst.WqA = UploadQuant(dev, src.WqA);
            dst.WqB = UploadQuant(dev, src.WqB);
            dst.Wkv = UploadQuant(dev, src.Wkv);
            dst.WoA = UploadQuant(dev, src.WoA);
            dst.WoB = UploadQuant(dev, src.WoB);
            dst.CompWkv = UploadQuant(dev, src.CompWkv);
            dst.CompWgate = UploadQuant(dev, src.CompWgate);
            dst.IdxProj = UploadQuant(dev, src.IdxProj);
            dst.IdxQB = UploadQuant(dev, src.IdxQB);
            dst.IdxCompWkv = UploadQuant(dev, src.IdxCompWkv);
            dst.IdxCompWgate = UploadQuant(dev, src.IdxCompWgate);
            dst.GateExps = UploadQuant(dev, src.GateExps);
            dst.UpExps = UploadQuant(dev, src.UpExps);
            dst.DownExps = UploadQuant(dev, src.DownExps);
            dst.GateShexp = UploadQuant(dev, src.GateShexp);
            dst.UpShexp = UploadQuant(dev, src.UpShexp);
            dst.DownShexp = UploadQuant(dev, src.DownShexp);
            dst.ShFf = src.UpShexp.Ne1;

            dst.AttnNorm = UploadF32(dev, src.AttnNorm);
            dst.QANorm = UploadF32(dev, src.QANorm);
            dst.KvNorm = UploadF32(dev, src.KvNorm);
            dst.Sinks = UploadF32(dev, src.Sinks);
            dst.HcAttnFn = UploadF32(dev, src.HcAttnFn);
            dst.HcAttnScale = UploadF32(dev, src.HcAttnScale);
            dst.HcAttnBase = UploadF32(dev, src.HcAttnBase);
            dst.HcFfnFn = UploadF32(dev, src.HcFfnFn);
            dst.HcFfnScale = UploadF32(dev, src.HcFfnScale);
            dst.HcFfnBase = UploadF32(dev, src.HcFfnBase);
            dst.CompApe = UploadF32(dev, src.CompApe);
            dst.CompNorm = UploadF32(dev, src.CompNorm);
            dst.IdxCompApe = UploadF32(dev, src.IdxCompApe);
            dst.IdxCompNorm = UploadF32(dev, src.IdxCompNorm);
            dst.GateInp = UploadF32(dev, src.GateInp);
            dst.ExpProbsBias = UploadF32(dev, src.ExpProbsBias);
            dst.FfnNorm = UploadF32(dev, src.FfnNorm);
            dst.Tid2Eid = UploadI32(dev, src.Tid2Eid);

            int hd = _m.HeadDim;
            dst.RingK = ArenaTake(dev, (long)_ringRaw * hd * 2);
            if (dst.Ratio == CsaRatio)
            {
                dst.CompK = ArenaTake(dev, (long)_compRowsCsa * hd * 2);
                dst.LidK = ArenaTake(dev, (long)_compRowsCsa * _m.IdxHeadSize * 2);
                dst.HistKv = ArenaTake(dev, 2L * CsaRatio * 2 * hd * 4);
                dst.HistScore = ArenaTake(dev, 2L * CsaRatio * 2 * hd * 4);
                dst.LidHistKv = ArenaTake(dev, 2L * CsaRatio * 2 * _m.IdxHeadSize * 4);
                dst.LidHistScore = ArenaTake(dev, 2L * CsaRatio * 2 * _m.IdxHeadSize * 4);
            }
            else if (dst.Ratio == HcaRatio)
            {
                dst.CompK = ArenaTake(dev, (long)_compRowsHca * hd * 2);
                dst.HistKv = ArenaTake(dev, (long)HcaRatio * hd * 4);
                dst.HistScore = ArenaTake(dev, (long)HcaRatio * hd * 4);
            }
        }

        private static IntPtr AllocDev(Dev dev, long bytes)
        {
            CudaDriverApi.cuMemAlloc(out IntPtr p, new UIntPtr((ulong)Math.Max(bytes, 256))).ThrowOnError();
            dev.OwnedAllocs.Add(p);
            return p;
        }

        private void AllocateScratch(Dev dev)
        {
            dev.MakeCurrent();
            var m = _m;
            int nt = m.NUbatch;
            int e = m.NEmbd;
            int hd = m.HeadDim;
            int s = nt * m.NExpertUsed;

            dev.Xs = AllocDev(dev, (long)nt * HC * e * 4);
            dev.XsOut = AllocDev(dev, (long)nt * HC * e * 4);
            dev.Cur = AllocDev(dev, (long)nt * e * 4);
            dev.Inv = AllocDev(dev, (long)nt * 4);
            dev.Mixes = AllocDev(dev, (long)nt * HcMixDim * 4);
            dev.Pre = AllocDev(dev, (long)nt * HC * 4);
            dev.Post = AllocDev(dev, (long)nt * HC * 4);
            dev.Comb = AllocDev(dev, (long)nt * HC * HC * 4);
            dev.Qr = AllocDev(dev, (long)nt * m.QLoraRank * 4);
            dev.Q = AllocDev(dev, (long)nt * m.NHead * hd * 4);
            dev.KvRaw = AllocDev(dev, (long)nt * hd * 4);
            dev.StKv = AllocDev(dev, (long)nt * 2 * hd * 4);
            dev.StScore = AllocDev(dev, (long)nt * 2 * hd * 4);
            dev.LidStKv = AllocDev(dev, (long)nt * 2 * m.IdxHeadSize * 4);
            dev.LidStScore = AllocDev(dev, (long)nt * 2 * m.IdxHeadSize * 4);
            dev.Iq = AllocDev(dev, (long)nt * m.IdxNHead * m.IdxHeadSize * 4);
            dev.Iw = AllocDev(dev, (long)nt * m.IdxNHead * 4);
            dev.IdxScores = AllocDev(dev, (long)nt * _compRowsCsa * 4);
            dev.TopkIdx = AllocDev(dev, (long)nt * m.IdxTopK * 4);
            dev.TopkCnt = AllocDev(dev, (long)nt * 4);
            dev.AttnO = AllocDev(dev, (long)nt * m.NHead * hd * 4);
            dev.OGrouped = AllocDev(dev, (long)nt * m.NHead * hd * 4);
            dev.OGroupedOut = AllocDev(dev, (long)m.OGroups * nt * m.OLoraRank * 4);
            dev.OG = AllocDev(dev, (long)nt * m.OGroups * m.OLoraRank * 4);
            dev.AttnOut = AllocDev(dev, (long)nt * e * 4);
            dev.FfnOut = AllocDev(dev, (long)nt * e * 4);
            dev.RouterLogits = AllocDev(dev, (long)nt * m.NExpert * 4);
            dev.Sel = AllocDev(dev, (long)s * 4);
            dev.SelW = AllocDev(dev, (long)s * 4);
            dev.Counts = AllocDev(dev, (long)m.NExpert * 4);
            dev.Offsets = AllocDev(dev, (long)m.NExpert * 4);
            dev.Cursors = AllocDev(dev, (long)m.NExpert * 4);
            dev.RowOfSlot = AllocDev(dev, (long)s * 4);
            dev.SlotToken = AllocDev(dev, (long)s * 4);

            // q8_1 activation scratch: A covers [max(nt, S)] rows of the widest
            // quantized input (oGroups*oLoraRank for wo_b); B covers the packed
            // expert hidden rows for the down projection.
            int maxIn = Math.Max(Math.Max(e, m.OGroups * m.OLoraRank), Math.Max(m.QLoraRank, m.NFfExp));
            long q8RowBytesA = (long)(maxIn / 32) * Q81BlockBytes;
            long rowsA = Math.Max(Math.Max(nt, s), (long)m.OGroups * nt);
            dev.ActQ8A = AllocDev(dev, rowsA * q8RowBytesA);
            dev.ActQ8B = AllocDev(dev, (long)s * (m.NFfExp / 32) * Q81BlockBytes);

            // split-layout twins for the staged expert kernels
            dev.SplitQsA = AllocDev(dev, (long)nt * e);
            dev.SplitDA = AllocDev(dev, (long)nt * (e / 32) * 4);
            dev.SplitQsB = AllocDev(dev, (long)s * m.NFfExp);
            dev.SplitDB = AllocDev(dev, (long)s * (m.NFfExp / 32) * 4);

            dev.ExpGate = AllocDev(dev, (long)s * m.NFfExp * 4);
            dev.ExpUp = AllocDev(dev, (long)s * m.NFfExp * 4);
            dev.ExpDown = AllocDev(dev, (long)s * e * 4);
            int shFf = 0;
            foreach (var dl in _layers)
                if (dl.Device == dev.Ordinal && dl.ShFf > shFf)
                    shFf = dl.ShFf;
            shFf = Math.Max(shFf, 1);
            dev.ShGate = AllocDev(dev, (long)nt * shFf * 4);
            dev.ShUp = AllocDev(dev, (long)nt * shFf * 4);
            dev.ShDown = AllocDev(dev, (long)nt * e * 4);

            // F16 dequant scratch for prefill GEMMs: sized to the largest dense
            // quantized weight on this device (+ activation panel).
            long maxWElems = 0;
            for (int il = 0; il < _m.NLayer; il++)
            {
                if (_layers[il].Device != dev.Ordinal)
                    continue;
                var L = _layers[il];
                foreach (var qw in new[] { L.WqA, L.WqB, L.Wkv, L.WoA, L.WoB, L.CompWkv, L.CompWgate, L.IdxProj, L.IdxQB, L.IdxCompWkv, L.IdxCompWgate, L.GateShexp, L.UpShexp, L.DownShexp })
                {
                    if (qw.Ptr == IntPtr.Zero)
                        continue;
                    // Only quantized weights are staged through WF16; F16/BF16/F32
                    // feed cuBLAS straight from the arena.
                    if (qw.Type == TF16 || qw.Type == TBF16 || qw.Type == TF32)
                        continue;
                    long elems = (long)qw.Ne0 * qw.Ne1;
                    if (elems > maxWElems)
                        maxWElems = elems;
                }
            }
            dev.WF16Elems = maxWElems;
            if (maxWElems > 0)
                dev.WF16 = AllocDev(dev, maxWElems * 2);
            long maxAElems = (long)Math.Max(nt, s) * maxIn;
            dev.AF16 = AllocDev(dev, maxAElems * 2);
        }

        // -------------------------------------------------------------------
        // Reset / caches
        // -------------------------------------------------------------------

        public void Reset()
        {
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream);
            }
            int hd = _m.HeadDim;
            for (int il = 0; il < _m.NLayer; il++)
            {
                var L = _layers[il];
                var dev = _devs[L.Device];
                dev.MakeCurrent();
                Memset0(L.RingK, (long)_ringRaw * hd * 2);
                if (L.Ratio == CsaRatio)
                {
                    Memset0(L.CompK, (long)_compRowsCsa * hd * 2);
                    Memset0(L.LidK, (long)_compRowsCsa * _m.IdxHeadSize * 2);
                    Memset0(L.HistKv, 2L * CsaRatio * 2 * hd * 4);
                    Memset0(L.HistScore, 2L * CsaRatio * 2 * hd * 4);
                    Memset0(L.LidHistKv, 2L * CsaRatio * 2 * _m.IdxHeadSize * 4);
                    Memset0(L.LidHistScore, 2L * CsaRatio * 2 * _m.IdxHeadSize * 4);
                }
                else if (L.Ratio == HcaRatio)
                {
                    Memset0(L.CompK, (long)_compRowsHca * hd * 2);
                    Memset0(L.HistKv, (long)HcaRatio * hd * 4);
                    Memset0(L.HistScore, (long)HcaRatio * hd * 4);
                }
            }
            NPast = 0;
        }

        private static void Memset0(IntPtr p, long bytes)
        {
            if (p != IntPtr.Zero && bytes > 0)
                CudaDriverApi.cuMemsetD8(p, 0, new UIntPtr((ulong)bytes)).ThrowOnError();
        }

        // -------------------------------------------------------------------
        // Forward
        // -------------------------------------------------------------------

        public void Forward(int[] tokens, float[] logitsOut)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            if (NPast + tokens.Length > _m.NCtx)
                throw new InvalidOperationException($"[dsv4-cuda] context overflow: n_past={NPast} + {tokens.Length} > n_ctx={_m.NCtx}");

            var sw = Stopwatch.StartNew();
            int done = 0;
            while (done < tokens.Length)
            {
                int nt = Math.Min(_m.NUbatch, tokens.Length - done);
                bool last = done + nt == tokens.Length;
                ForwardUbatch(tokens, done, nt, NPast, last ? logitsOut : null);
                NPast += nt;
                done += nt;
            }
            if (_perf > 0)
            {
                double secs = sw.Elapsed.TotalSeconds;
                Console.Error.WriteLine($"[dsv4-cuda] forward {tokens.Length} tokens in {secs:F3}s ({tokens.Length / secs:F1} tok/s)");
                ReportStages();
            }
        }

        private void CheckSync(Dev dev, string stage)
        {
            if (!_syncDebug)
                return;
            int rc = CudaDriverApi.cuStreamSynchronize(dev.Stream);
            if (rc != 0)
                throw new InvalidOperationException($"[dsv4-cuda] CUDA error {rc} after {stage} on device {dev.Ordinal}");
        }

        // TS_DSV4_PERF>=2: per-stage wall time. Each Stage() call synchronizes
        // the device stream, so the totals are serialized-stage times (they sum
        // to more than a pipelined step) -- use them to rank stages, not to
        // predict throughput.
        private static readonly string[] StageNames =
        {
            "embed", "hc", "attnproj", "comp", "idx", "attncore", "outproj",
            "router", "experts", "shexp", "lmhead", "boundary",
        };
        private readonly long[] _stageTicks = new long[12];
        private long _stageT0;

        private void StageBegin()
        {
            if (_perf >= 2)
                _stageT0 = Stopwatch.GetTimestamp();
        }

        private void StageEnd(Dev dev, int stage)
        {
            if (_perf < 2)
                return;
            CudaDriverApi.cuStreamSynchronize(dev.Stream);
            _stageTicks[stage] += Stopwatch.GetTimestamp() - _stageT0;
            _stageT0 = Stopwatch.GetTimestamp();
        }

        private void ReportStages()
        {
            if (_perf < 2)
                return;
            var sb = new System.Text.StringBuilder("[dsv4-cuda]   stages:");
            for (int i = 0; i < _stageTicks.Length; i++)
            {
                if (_stageTicks[i] == 0)
                    continue;
                sb.Append($" {StageNames[i]}={_stageTicks[i] * 1000.0 / Stopwatch.Frequency:F1}ms");
                _stageTicks[i] = 0;
            }
            Console.Error.WriteLine(sb.ToString());
        }

        // TS_DSV4_CUDA_DEBUG=1: print the first values of layer-0 stage outputs
        // (paired with TS_DSV4_CPU_DEBUG=1 prints in DeepSeek4CpuExecutor for
        // stage-by-stage A/B against the exact managed reference).
        private static readonly bool StageDebug = EnvInt("TS_DSV4_CUDA_DEBUG", 0) != 0;

        private void Dump(Dev dev, string label, IntPtr ptr, int n = 6)
        {
            if (!StageDebug || ptr == IntPtr.Zero)
                return;
            dev.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev.Stream);
            var tmp = new float[n];
            fixed (float* p = tmp)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, ptr, new UIntPtr((ulong)n * 4));
            Console.Error.WriteLine($"[dbg-cuda] {label}: {string.Join(" ", Array.ConvertAll(tmp, v => v.ToString("G6")))}");
        }

        private void DumpF16(Dev dev, string label, IntPtr ptr, int n = 6)
        {
            if (!StageDebug || ptr == IntPtr.Zero)
                return;
            dev.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev.Stream);
            var tmp = new ushort[n];
            fixed (ushort* p = tmp)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, ptr, new UIntPtr((ulong)n * 2));
            var vals = new float[n];
            for (int i = 0; i < n; i++)
                vals[i] = (float)BitConverter.UInt16BitsToHalf(tmp[i]);
            Console.Error.WriteLine($"[dbg-cuda] {label}: {string.Join(" ", Array.ConvertAll(vals, v => v.ToString("G6")))}");
        }

        private void ForwardUbatch(int[] tokens, int tokOff, int nt, int p0, float[] logitsOut)
        {
            var m = _m;
            int e = m.NEmbd;

            // tokens -> pinned buffer (parity-double-buffered) -> devices that need
            // them. Per-device parity events guard reuse of the pinned slot two
            // chunks later without forcing a stream sync.
            int parity = _chunkParity;
            _chunkParity ^= 1;
            IntPtr pinned = parity == 0 ? _pinnedTokens0 : _pinnedTokens1;
            foreach (var dev in _devs)
            {
                if (!dev.NeedsTokens)
                    continue;
                CudaDriverApi.cuEventSynchronize(parity == 0 ? dev.TokEv0 : dev.TokEv1);
            }
            Marshal.Copy(tokens, tokOff, pinned, nt);
            foreach (var dev in _devs)
            {
                if (!dev.NeedsTokens)
                    continue;
                dev.MakeCurrent();
                IntPtr dst = parity == 0 ? dev.TokensDev0 : dev.TokensDev1;
                CudaDriverApi.cuMemcpyHtoDAsync(dst, pinned, new UIntPtr((ulong)nt * 4), dev.Stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(parity == 0 ? dev.TokEv0 : dev.TokEv1, dev.Stream).ThrowOnError();
            }
            IntPtr TokensOf(Dev dev) => parity == 0 ? dev.TokensDev0 : dev.TokensDev1;

            // embedding on device 0
            var dev0 = _devs[0];
            dev0.MakeCurrent();
            StageBegin();
            dev0.DK.Embed(_tokEmbdQW.Ptr, TokensOf(dev0), dev0.Xs, _tokEmbdQW.Type, _tokEmbdQW.RowBytes, nt, e, dev0.Stream);
            StageEnd(dev0, 0);
            CheckSync(dev0, "embed");
            Dump(dev0, "embed.xs", dev0.Xs);

            int curDev = 0;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != curDev)
                {
                    // Hand the hidden streams to the next device, staged through
                    // pinned host memory (peer DMA lies on this hardware class —
                    // see the BoundaryPinned comment). Event chain: DtoH on the
                    // src stream -> dst waits XsReady -> HtoD on the dst stream
                    // -> src waits CopyDone before anything may overwrite its Xs
                    // or the pinned slice (next chunk's work).
                    var src = _devs[curDev];
                    var dst = _devs[L.Device];
                    var copyBytes = new UIntPtr((ulong)((long)nt * HC * e * 4));
                    // Each event is RECORDED on a stream in its own context (the
                    // driver rejects a cross-context record with "invalid
                    // resource handle"); only the WAITS cross contexts, which
                    // is the supported multi-GPU pattern.
                    src.MakeCurrent();
                    CudaDriverApi.cuMemcpyDtoHAsync(src.BoundaryPinned, src.Xs, copyBytes, src.Stream).ThrowOnError();
                    CudaDriverApi.cuEventRecord(src.XsReadyEv, src.Stream).ThrowOnError();
                    dst.MakeCurrent();
                    CudaDriverApi.cuStreamWaitEvent(dst.Stream, src.XsReadyEv, 0).ThrowOnError();
                    CudaDriverApi.cuMemcpyHtoDAsync(dst.Xs, src.BoundaryPinned, copyBytes, dst.Stream).ThrowOnError();
                    CudaDriverApi.cuEventRecord(dst.CopyDoneEv, dst.Stream).ThrowOnError();
                    src.MakeCurrent();
                    CudaDriverApi.cuStreamWaitEvent(src.Stream, dst.CopyDoneEv, 0).ThrowOnError();
                    curDev = L.Device;
                    StageEnd(dst, 11);
                }

                var dev = _devs[curDev];
                dev.MakeCurrent();

                // ---- attention super-block ----
                bool dbg = StageDebug && il == 0;
                HcPre(dev, L, nt, attn: true);
                if (dbg)
                {
                    Dump(dev, "L0.attn.mixes", dev.Mixes);
                    Dump(dev, "L0.attn.pre", dev.Pre, 4);
                    Dump(dev, "L0.attn.comb", dev.Comb, 8);
                    Dump(dev, "L0.attn.cur", dev.Cur);
                }
                dev.DK.RmsNorm(dev.Cur, L.AttnNorm, nt, e, m.RmsEps, dev.Stream);
                StageEnd(dev, 1);
                if (dbg)
                    Dump(dev, "L0.attn.cur_norm", dev.Cur);
                CheckSync(dev, $"hc_pre_attn L{il}");
                Attention(dev, L, il, nt, p0, TokensOf(dev));
                if (dbg)
                    Dump(dev, "L0.attn.out", dev.AttnOut);
                dev.DK.HcPost(dev.Xs, dev.AttnOut, dev.Post, dev.Comb, dev.XsOut, nt, e, dev.Stream);
                SwapXs(dev);
                StageEnd(dev, 1);
                if (dbg)
                    Dump(dev, "L0.attn.xs_post", dev.Xs);
                CheckSync(dev, $"attn L{il}");

                // ---- FFN super-block ----
                HcPre(dev, L, nt, attn: false);
                dev.DK.RmsNorm(dev.Cur, L.FfnNorm, nt, e, m.RmsEps, dev.Stream);
                StageEnd(dev, 1);
                MoeFfn(dev, L, il, nt, TokensOf(dev));
                if (dbg)
                    Dump(dev, "L0.ffn.out", dev.FfnOut);
                dev.DK.HcPost(dev.Xs, dev.FfnOut, dev.Post, dev.Comb, dev.XsOut, nt, e, dev.Stream);
                SwapXs(dev);
                StageEnd(dev, 1);
                if (StageDebug)
                    Dump(dev, $"L{il}.ffn.xs_post", dev.Xs);
                CheckSync(dev, $"ffn L{il}");
            }

            if (logitsOut != null)
            {
                var dev = _devs[curDev];
                if (curDev != _lastDev)
                    throw new InvalidOperationException("[dsv4-cuda] output head is not on the final layer device");
                dev.MakeCurrent();
                IntPtr lastX = (IntPtr)((long)dev.Xs + (long)(nt - 1) * HC * e * 4);
                dev.DK.HcHead(lastX, _hcHeadFn, _hcHeadScale, _hcHeadBase, dev.Cur, e,
                    m.HcHeadScale.Length, m.HcHeadBase.Length, m.RmsEps, dev.Stream);
                Dump(dev, "head.cur", dev.Cur);
                dev.DK.RmsNorm(dev.Cur, _outputNorm, 1, e, m.RmsEps, dev.Stream);
                MatMul(dev, _outputQW, dev.Cur, _logitsDev, 1);
                StageEnd(dev, 10);
                Dump(dev, "head.logits", _logitsDev, 8);
                fixed (float* dst = _logitsHost)
                {
                    CudaDriverApi.cuMemcpyDtoHAsync((IntPtr)dst, _logitsDev, new UIntPtr((ulong)m.NVocab * 4UL), dev.Stream).ThrowOnError();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                }
                Array.Copy(_logitsHost, logitsOut, m.NVocab);
            }
        }

        private static void SwapXs(Dev dev)
        {
            (dev.Xs, dev.XsOut) = (dev.XsOut, dev.Xs);
        }

        private void HcPre(Dev dev, DevLayer l, int nt, bool attn)
        {
            var m = _m;
            int flatDim = HC * m.NEmbd;
            IntPtr fn = attn ? l.HcAttnFn : l.HcFfnFn;
            IntPtr scale = attn ? l.HcAttnScale : l.HcFfnScale;
            IntPtr baseW = attn ? l.HcAttnBase : l.HcFfnBase;

            dev.DK.HcRms(dev.Xs, dev.Inv, nt, flatDim, m.RmsEps, dev.Stream);
            // mixes = hc_fn (F32 [24, flatDim]) x flat streams
            GemmF32(dev, fn, dev.Xs, dev.Mixes, flatDim, HcMixDim, nt);
            dev.DK.HcGatesComb(dev.Mixes, dev.Inv, scale, baseW, dev.Pre, dev.Post, dev.Comb,
                nt, m.HcSinkhornIters, m.HcEps, dev.Stream);
            dev.DK.HcCollapse(dev.Xs, dev.Pre, dev.Cur, nt, m.NEmbd, dev.Stream);
        }

        // -------------------------------------------------------------------
        // Attention
        // -------------------------------------------------------------------

        private void Attention(Dev dev, DevLayer l, int il, int nt, int p0, IntPtr tokensDev)
        {
            var m = _m;
            int e = m.NEmbd, nh = m.NHead, hd = m.HeadDim, rot = m.NRot;
            bool comp = l.Ratio != 0;
            IntPtr ropeTab = comp ? dev.RopeComp : dev.RopeRaw;

            // q = wq_b(rms(wq_a(cur))), kv = wkv(cur)
            bool dbg = StageDebug && il == 0;
            MatMul(dev, l.WqA, dev.Cur, dev.Qr, nt);
            dev.DK.RmsNorm(dev.Qr, l.QANorm, nt, m.QLoraRank, m.RmsEps, dev.Stream);
            if (dbg)
                Dump(dev, "L0.qr", dev.Qr);
            MatMul(dev, l.WqB, dev.Qr, dev.Q, nt);
            MatMul(dev, l.Wkv, dev.Cur, dev.KvRaw, nt);
            if (dbg)
                Dump(dev, "L0.kv_raw", dev.KvRaw);
            dev.DK.AttnPrep(dev.Q, dev.KvRaw, l.KvNorm, ropeTab, l.RingK, p0, _ringRaw, nh, hd, rot, m.RmsEps, nt, dev.Stream);
            StageEnd(dev, 2);
            if (dbg)
            {
                Dump(dev, "L0.q_prep", dev.Q);
                DumpF16(dev, "L0.ring0", l.RingK);
            }
            CheckSync(dev, $"attn_prep L{il}");

            int mode = 0;
            if (l.Ratio == CsaRatio)
            {
                int cw = 2 * hd;
                MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
                MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);
                dev.DK.ApeAdd(dev.StScore, l.CompApe, p0, CsaRatio, nt, cw, dev.Stream);
                RunCompressor(dev, nt, p0, CsaRatio, 2, hd, 2 * CsaRatio, cw,
                    dev.StKv, dev.StScore, l.HistKv, l.HistScore, l.CompNorm, l.CompK, dev.RopeComp);

                int lcw = 2 * m.IdxHeadSize;
                MatMul(dev, l.IdxCompWkv, dev.Cur, dev.LidStKv, nt);
                MatMul(dev, l.IdxCompWgate, dev.Cur, dev.LidStScore, nt);
                dev.DK.ApeAdd(dev.LidStScore, l.IdxCompApe, p0, CsaRatio, nt, lcw, dev.Stream);
                RunCompressor(dev, nt, p0, CsaRatio, 2, m.IdxHeadSize, 2 * CsaRatio, lcw,
                    dev.LidStKv, dev.LidStScore, l.LidHistKv, l.LidHistScore, l.IdxCompNorm, l.LidK, dev.RopeComp);
                StageEnd(dev, 3);
                CheckSync(dev, $"compress L{il}");

                int maxVis = (int)(((long)p0 + nt) / CsaRatio);
                if (maxVis > m.IdxTopK)
                {
                    // lightning indexer + top-k
                    MatMul(dev, l.IdxQB, dev.Qr, dev.Iq, nt);
                    MatMul(dev, l.IdxProj, dev.Cur, dev.Iw, nt);
                    float iwScale = 1.0f / MathF.Sqrt((float)m.IdxHeadSize * m.IdxNHead);
                    dev.DK.IdxPrep(dev.Iq, dev.Iw, dev.RopeComp, p0, m.IdxNHead, m.IdxHeadSize, rot, iwScale, nt, dev.Stream);
                    dev.DK.IdxScores(dev.Iq, dev.Iw, l.LidK, dev.IdxScores, p0, CsaRatio, m.IdxNHead, m.IdxHeadSize,
                        nt, _compRowsCsa, maxVis, dev.Stream);
                    dev.DK.TopK(dev.IdxScores, dev.TopkIdx, dev.TopkCnt, p0, CsaRatio, m.IdxTopK, _compRowsCsa, nt, dev.Stream);
                    StageEnd(dev, 4);
                    CheckSync(dev, $"indexer L{il}");
                    mode = 1;
                }
                else
                {
                    mode = 2; // every visible row is selected: skip the indexer entirely
                }
            }
            else if (l.Ratio == HcaRatio)
            {
                MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
                MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);
                dev.DK.ApeAdd(dev.StScore, l.CompApe, p0, HcaRatio, nt, hd, dev.Stream);
                RunCompressor(dev, nt, p0, HcaRatio, 1, hd, HcaRatio, hd,
                    dev.StKv, dev.StScore, l.HistKv, l.HistScore, l.CompNorm, l.CompK, dev.RopeComp);
                StageEnd(dev, 3);
                CheckSync(dev, $"compress L{il}");
                mode = 2;
            }

            float kqScale = 1.0f / MathF.Sqrt(hd);
            dev.DK.Attention(dev.Q, l.RingK, l.CompK, dev.TopkIdx, dev.TopkCnt, l.Sinks, dev.AttnO,
                p0, m.NSwa, _ringRaw, nh, hd, mode, l.Ratio == 0 ? 1 : l.Ratio, m.IdxTopK, kqScale, nt, dev.Stream);
            StageEnd(dev, 5);
            if (dbg)
                Dump(dev, "L0.attn_core", dev.AttnO);
            CheckSync(dev, $"attn_core L{il}");

            // inverse rope + grouped LoRA out-projection
            int hpg = nh / m.OGroups;
            dev.DK.AttnFinish(dev.AttnO, ropeTab, dev.OGrouped, p0, nh, hd, rot, hpg, nt, dev.Stream);

            int groupDim = hpg * hd;
            int oCat = m.OGroups * m.OLoraRank;
            // quantize the whole grouped layout once ([G*nt, groupDim] rows), then
            // run one matmul per group against the matching WoA row block.
            var woA = l.WoA;
            if (nt == 1 && woA.Type == TBF16 && Bf16MatvecEnabled && (groupDim & 7) == 0)
            {
                // Decode: the 8 group matvecs are ~11us each, so folding them
                // into one grid saves more in launch gaps than it costs.
                dev.Alloc.Kernels.LaunchMatvecBf16(woA.Ptr, dev.OGrouped, dev.OGroupedOut,
                    groupDim, m.OLoraRank, dev.Stream, m.OGroups);
            }
            else
            {
                for (int g = 0; g < m.OGroups; g++)
                {
                    var slice = woA;
                    slice.Ptr = (IntPtr)((long)woA.Ptr + (long)g * m.OLoraRank * woA.RowBytes);
                    slice.Ne1 = m.OLoraRank;
                    IntPtr input = (IntPtr)((long)dev.OGrouped + (long)g * nt * groupDim * 4);
                    IntPtr output = (IntPtr)((long)dev.OGroupedOut + (long)g * nt * m.OLoraRank * 4);
                    MatMul(dev, slice, input, output, nt);
                }
            }
            dev.DK.Regroup(dev.OGroupedOut, dev.OG, m.OGroups, nt, m.OLoraRank, dev.Stream);
            MatMul(dev, l.WoB, dev.OG, dev.AttnOut, nt);
            StageEnd(dev, 6);
            CheckSync(dev, $"out_proj L{il}");
        }

        private void RunCompressor(Dev dev, int nt, int p0, int ratio, int coff, int head, int stateSize, int cw,
            IntPtr stKv, IntPtr stScore, IntPtr histKv, IntPtr histScore, IntPtr normW, IntPtr cache, IntPtr ropeTab)
        {
            long firstBoundary = -1;
            for (long p = p0; p < (long)p0 + nt; p++)
            {
                if ((p + 1) % ratio == 0)
                {
                    firstBoundary = p;
                    break;
                }
            }
            if (firstBoundary >= 0)
            {
                int nBlocks = (int)(((long)p0 + nt - 1 - firstBoundary) / ratio) + 1;
                dev.DK.Compress(stKv, stScore, histKv, histScore, normW, ropeTab, cache,
                    firstBoundary, nBlocks, p0, ratio, coff, head, stateSize, _m.NRot, _m.RmsEps, dev.Stream);
            }
            dev.DK.Persist(stKv, stScore, histKv, histScore, p0, nt, stateSize, cw, dev.Stream);
        }

        // -------------------------------------------------------------------
        // MoE FFN
        // -------------------------------------------------------------------

        private void MoeFfn(Dev dev, DevLayer l, int il, int nt, IntPtr tokensDev)
        {
            var m = _m;
            int e = m.NEmbd, ff = m.NFfExp, nUsed = m.NExpertUsed, nEx = m.NExpert;
            int s = nt * nUsed;

            // router logits (gate_inp is F32) + selection/weights
            bool dbg = StageDebug && il == 0;
            GemmF32(dev, l.GateInp, dev.Cur, dev.RouterLogits, e, nEx, nt);
            dev.DK.MoeSelect(dev.RouterLogits, l.ExpProbsBias, l.Tid2Eid, tokensDev, dev.Sel, dev.SelW,
                nEx, nUsed, m.ExpertWeightsNorm ? 1 : 0, m.ExpertWeightsScale, nt, dev.Stream);
            StageEnd(dev, 7);
            if (dbg)
            {
                Dump(dev, "L0.router", dev.RouterLogits);
                Dump(dev, "L0.selw", dev.SelW, m.NExpertUsed);
            }
            CheckSync(dev, $"moe_select L{il}");

            // shared expert (dense) -> ShDown
            int shFf = l.ShFf;
            MatMul(dev, l.UpShexp, dev.Cur, dev.ShUp, nt);
            MatMul(dev, l.GateShexp, dev.Cur, dev.ShGate, nt);
            dev.DK.SwigluClamp(dev.ShGate, dev.ShUp, (long)nt * shFf, l.ClampShexp, dev.Stream);
            MatMul(dev, l.DownShexp, dev.ShGate, dev.ShDown, nt);
            StageEnd(dev, 9);
            CheckSync(dev, $"shexp L{il}");

            // routed experts
            int guType = l.GateExps.Type;
            int downType = l.DownExps.Type;
            if (l.UpExps.Type != guType)
                throw new NotSupportedException("[dsv4-cuda] expert gate/up quant types differ");
            RequireExpertType(guType);
            RequireExpertType(downType);

            bool staged = StagedExpertsEnabled && nt > 1
                && Dsv4Kernels.StagedSupports(e) && Dsv4Kernels.StagedSupports(ff);
            if (staged)
                QuantizeQ81Split(dev, dev.Cur, dev.SplitQsA, dev.SplitDA, e, nt);
            else
                QuantizeQ81(dev, dev.Cur, dev.ActQ8A, e, nt);
            if (nt == 1)
            {
                dev.DK.MoeGateUpDecode(l.GateExps.Ptr, l.UpExps.Ptr, dev.ActQ8A, dev.Sel,
                    dev.ExpGate, dev.ExpUp, guType, ff, e, l.GateExps.RowBytes, nUsed, dev.Stream);
                dev.DK.SwigluClamp(dev.ExpGate, dev.ExpUp, (long)nUsed * ff, l.ClampExp, dev.Stream);
                QuantizeQ81(dev, dev.ExpGate, dev.ActQ8B, ff, nUsed);
                dev.DK.MoeDownDecode(l.DownExps.Ptr, dev.ActQ8B, dev.Sel, dev.ExpDown,
                    downType, e, ff, l.DownExps.RowBytes, nUsed, dev.Stream);
                dev.DK.MoeScatterAdd(dev.ExpDown, IntPtr.Zero, dev.SelW, dev.ShDown, dev.FfnOut, 1, nUsed, e, dev.Stream);
            }
            else
            {
                // grouping plan (all on device; Counts zeroed via the fill kernel
                // so everything stays stream-ordered)
                dev.Alloc.Kernels.LaunchFillF32(dev.Counts, nEx, 0f, dev.Stream);
                dev.DK.MoeCount(dev.Sel, dev.Counts, s, dev.Stream);
                dev.DK.MoeScan(dev.Counts, dev.Offsets, dev.Cursors, nEx, dev.Stream);
                dev.DK.MoeScatter(dev.Sel, dev.Cursors, dev.RowOfSlot, dev.SlotToken, s, nUsed, dev.Stream);

                // Staged-weight kernels decode each expert weight row once into
                // registers and reuse it across the expert's member tokens; the
                // per-token kernels re-decode per (row, token), which dominated
                // prefill.
                if (staged)
                {
                    dev.DK.MoeGateUpStaged(l.GateExps.Ptr, l.UpExps.Ptr, dev.SplitQsA, dev.SplitDA,
                        dev.Counts, dev.Offsets, dev.SlotToken,
                        dev.ExpGate, dev.ExpUp, guType, ff, e, l.GateExps.RowBytes, nEx, dev.Stream);
                    dev.DK.SwigluClamp(dev.ExpGate, dev.ExpUp, (long)s * ff, l.ClampExp, dev.Stream);
                    QuantizeQ81Split(dev, dev.ExpGate, dev.SplitQsB, dev.SplitDB, ff, s);
                    dev.DK.MoeDownStaged(l.DownExps.Ptr, dev.SplitQsB, dev.SplitDB, dev.Counts, dev.Offsets, dev.ExpDown,
                        downType, e, ff, l.DownExps.RowBytes, nEx, dev.Stream);
                }
                else
                {
                    dev.DK.MoeGateUp(l.GateExps.Ptr, l.UpExps.Ptr, dev.ActQ8A, dev.Counts, dev.Offsets, dev.SlotToken,
                        dev.ExpGate, dev.ExpUp, guType, ff, e, l.GateExps.RowBytes, nEx, dev.Stream);
                    dev.DK.SwigluClamp(dev.ExpGate, dev.ExpUp, (long)s * ff, l.ClampExp, dev.Stream);
                    QuantizeQ81(dev, dev.ExpGate, dev.ActQ8B, ff, s);
                    dev.DK.MoeDown(l.DownExps.Ptr, dev.ActQ8B, dev.Counts, dev.Offsets, dev.ExpDown,
                        downType, e, ff, l.DownExps.RowBytes, nEx, dev.Stream);
                }
                dev.DK.MoeScatterAdd(dev.ExpDown, dev.RowOfSlot, dev.SelW, dev.ShDown, dev.FfnOut, nt, nUsed, e, dev.Stream);
            }
            StageEnd(dev, 8);
            CheckSync(dev, $"experts L{il}");
        }

        // Register-staged expert kernels for prefill (TS_DSV4_STAGED_EXPERTS=0
        // reverts to the per-token kernels for A/B).
        // TS_DSV4_BF16_MATVEC=0 reverts decode's dense BF16 projections to cuBLAS.
        private static readonly bool Bf16MatvecEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_DSV4_BF16_MATVEC"), "0", StringComparison.Ordinal);

        private static readonly bool StagedExpertsEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_DSV4_STAGED_EXPERTS"), "0", StringComparison.Ordinal);

        private static void RequireExpertType(int type)
        {
            if (type != TIQ3_S && type != TMXFP4 && type != TQ8_0 && type != TQ6_K)
                throw new NotSupportedException($"[dsv4-cuda] unsupported expert quant type {type} (supported: Q8_0, Q6_K, IQ3_S, MXFP4)");
        }

        // -------------------------------------------------------------------
        // dense matmul dispatch
        // -------------------------------------------------------------------

        private void QuantizeQ81(Dev dev, IntPtr input, IntPtr scratch, int inDim, int rows)
        {
            dev.Alloc.Kernels.LaunchQuantizeQ81Rows(input, scratch, inDim, rows, dev.Stream, warpCooperative: true);
        }

        /// <summary>Dense split q8_1 (contiguous int8 row + separate per-block
        /// scale). Values are bit-identical to the interleaved layout; the dense
        /// form is what makes the staged expert kernels' activation reads
        /// vectorizable.</summary>
        private void QuantizeQ81Split(Dev dev, IntPtr input, IntPtr qs, IntPtr d, int inDim, int rows)
        {
            dev.Alloc.Kernels.LaunchQuantizeQ81SplitRows(input, qs, d, inDim, rows, dev.Stream);
        }

        /// <summary>
        /// output[rows, w.Ne1] = input[rows, w.Ne0] x w^T for the resident
        /// quantized/dense weight. Decode uses the dp4a matvec kernels, mid-size
        /// row counts the Q8_0 MMQ GEMM, prefill-sized row counts a
        /// dequant-to-F16 + cuBLAS GEMM (mirroring CudaQuantizedOps' routing).
        /// </summary>
        private void MatMul(Dev dev, in DevQW w, IntPtr input, IntPtr output, int rows)
        {
            int inDim = w.Ne0;
            int outDim = w.Ne1;
            var k = dev.Alloc.Kernels;

            switch (w.Type)
            {
                case TQ8_0:
                    if (rows == 1)
                    {
                        QuantizeQ81(dev, input, dev.ActQ8A, inDim, 1);
                        k.LaunchQuantMatmulQ80Vec(w.Ptr, dev.ActQ8A, output, inDim, outDim, dev.Stream);
                    }
                    else if (rows <= 512 && inDim % 32 == 0)
                    {
                        QuantizeQ81(dev, input, dev.ActQ8A, inDim, rows);
                        k.LaunchQuantMatmulQ80Mmq(w.Ptr, dev.ActQ8A, output, inDim, outDim, rows, dev.Stream);
                    }
                    else
                    {
                        k.LaunchDequantWeightQ80F16(w.Ptr, dev.WF16, (long)inDim * outDim, dev.Stream);
                        k.LaunchConvertF32F16(input, dev.AF16, (long)rows * inDim, dev.Stream);
                        GemmF16(dev, dev.WF16, dev.AF16, output, inDim, outDim, rows);
                    }
                    break;

                case TQ6_K:
                    if (rows == 1)
                    {
                        QuantizeQ81(dev, input, dev.ActQ8A, inDim, 1);
                        k.LaunchQuantMatmulQ6KDp4a(w.Ptr, dev.ActQ8A, output, inDim, outDim, dev.Stream);
                    }
                    else
                    {
                        k.LaunchDequantWeightF16(w.Ptr, dev.WF16, w.Type, inDim, (long)inDim * outDim, dev.Stream);
                        k.LaunchConvertF32F16(input, dev.AF16, (long)rows * inDim, dev.Stream);
                        GemmF16(dev, dev.WF16, dev.AF16, output, inDim, outDim, rows);
                    }
                    break;

                case TF16:
                    k.LaunchConvertF32F16(input, dev.AF16, (long)rows * inDim, dev.Stream);
                    GemmF16(dev, w.Ptr, dev.AF16, output, inDim, outDim, rows);
                    break;

                case TBF16:
                    // Weights stay bit-exact BF16 in VRAM. Decode is one row, so
                    // it is bandwidth-bound and the dedicated matvec beats a
                    // cuBLAS GEMM with n = 1; prefill wants the tensor cores.
                    if (rows == 1 && Bf16MatvecEnabled && (inDim & 7) == 0)
                    {
                        k.LaunchMatvecBf16(w.Ptr, input, output, inDim, outDim, dev.Stream);
                    }
                    else
                    {
                        k.LaunchConvertF32Bf16(input, dev.AF16, (long)rows * inDim, dev.Stream);
                        GemmBf16(dev, w.Ptr, dev.AF16, output, inDim, outDim, rows);
                    }
                    break;

                case TF32:
                    GemmF32(dev, w.Ptr, input, output, inDim, outDim, rows);
                    break;

                default:
                    if (rows == 1 && CudaQuantizedOps.SupportsQuantizedType(w.Type))
                    {
                        k.LaunchQuantMatmulVecF32(w.Ptr, input, output, w.Type, inDim, outDim, dev.Stream);
                    }
                    else if (CudaQuantizedOps.SupportsQuantizedType(w.Type))
                    {
                        k.LaunchDequantWeightF16(w.Ptr, dev.WF16, w.Type, inDim, (long)inDim * outDim, dev.Stream);
                        k.LaunchConvertF32F16(input, dev.AF16, (long)rows * inDim, dev.Stream);
                        GemmF16(dev, dev.WF16, dev.AF16, output, inDim, outDim, rows);
                    }
                    else
                    {
                        throw new NotSupportedException($"[dsv4-cuda] unsupported dense weight type {w.Type}");
                    }
                    break;
            }
        }

        private static void GemmF16(Dev dev, IntPtr wF16, IntPtr aF16, IntPtr outF32, int inDim, int outDim, int rows)
        {
            dev.Alloc.Blas.SetStream(dev.Stream);
            float alpha = 1.0f, beta = 0.0f;
            CublasApi.cublasGemmEx(
                dev.Alloc.Blas.Handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                outDim, rows, inDim,
                ref alpha,
                wF16, CublasApi.CUDA_R_16F, inDim,
                aF16, CublasApi.CUDA_R_16F, inDim,
                ref beta,
                outF32, CublasApi.CUDA_R_32F, outDim,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }

        private static void GemmBf16(Dev dev, IntPtr wBf16, IntPtr aBf16, IntPtr outF32, int inDim, int outDim, int rows)
        {
            dev.Alloc.Blas.SetStream(dev.Stream);
            float alpha = 1.0f, beta = 0.0f;
            CublasApi.cublasGemmEx(
                dev.Alloc.Blas.Handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                outDim, rows, inDim,
                ref alpha,
                wBf16, CublasApi.CUDA_R_16BF, inDim,
                aBf16, CublasApi.CUDA_R_16BF, inDim,
                ref beta,
                outF32, CublasApi.CUDA_R_32F, outDim,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }

        private static void GemmF32(Dev dev, IntPtr wF32, IntPtr aF32, IntPtr outF32, int inDim, int outDim, int rows)
        {
            dev.Alloc.Blas.SetStream(dev.Stream);
            float alpha = 1.0f, beta = 0.0f;
            CublasApi.cublasGemmEx(
                dev.Alloc.Blas.Handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                outDim, rows, inDim,
                ref alpha,
                wF32, CublasApi.CUDA_R_32F, inDim,
                aF32, CublasApi.CUDA_R_32F, inDim,
                ref beta,
                outF32, CublasApi.CUDA_R_32F, outDim,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }

        public void Dispose()
        {
            foreach (var dev in _devs)
            {
                if (dev == null)
                    continue;
                try
                {
                    dev.MakeCurrent();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream);
                    foreach (var p in dev.OwnedAllocs)
                        CudaDriverApi.cuMemFree(p);
                    dev.OwnedAllocs.Clear();
                    if (dev.Arena != IntPtr.Zero)
                        CudaDriverApi.cuMemFree(dev.Arena);
                    if (dev.Event != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.Event);
                    if (dev.TokEv0 != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.TokEv0);
                    if (dev.TokEv1 != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.TokEv1);
                    if (dev.XsReadyEv != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.XsReadyEv);
                    if (dev.CopyDoneEv != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.CopyDoneEv);
                    if (dev.BoundaryPinned != IntPtr.Zero)
                        CudaDriverApi.cuMemFreeHost(dev.BoundaryPinned);
                    dev.DK?.Dispose();
                    dev.Alloc?.Dispose();
                }
                catch
                {
                    // teardown must not throw
                }
            }
            if (_pinnedTokens0 != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedTokens0);
            if (_pinnedTokens1 != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedTokens1);
        }
    }
}
