// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DSpark speculative decoding for DeepSeek V4 (Flash).
//
// DSpark (DeepSeek + PKU 2026, "Confidence-Scheduled Speculative Decoding with
// Semi-Autoregressive Generation") drafts a whole BLOCK of tokens per step from
// a small support module shipped alongside the target model: three DSV4 blocks
// (compress_ratio 0, i.e. plain sliding-window attention) that read the
// target's hidden states instead of running the target itself.
//
//   main_x     = main_norm(main_proj([h40; h41; h42]))   per COMMITTED position
//   ring[pos] <- rope(kv_norm(wkv(main_x)))              the drafter's history
//   block      = embed([last_token, noise x (B-1)])      B = dspark.block_size
//   ... 3 stages, each attending over [ring window | the block itself],
//       non-causally inside the block ...
//   logits(i)  = lm_head(norm(hc_head(x)))(i) + W2 . W1[prev(i)]   (Markov head)
//   conf(i)    = sigmoid(proj . [x(i); W1[prev(i)]])               (confidence)
//
// prev(0) is the anchor (last committed) token and prev(i) the drafter's own
// argmax at i-1, so the block is semi-autoregressive: one forward pass, but
// each position still conditions on the token before it. The confidence head
// truncates the block at the first position whose predicted acceptance falls
// below the threshold.
//
// The target then verifies [last_token, d1..dk] in ONE batched forward and the
// caller keeps the longest prefix its sampler would have drawn anyway, so the
// emitted token stream is identical to non-speculative greedy decoding.
//
// Everything here runs on the device that owns the output head and the three
// target feature layers; the drafter reuses the target's kernels (hyper
// connections, MoE, grouped out-projection) with its own weights, plus the
// DSpark-specific kernels in tensorsharp_dsv4_kernels.cu.
// ---------------------------------------------------------------------------
using System;
using System.Diagnostics;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Dsv4CudaEngine
    {
        /// <summary>Weights and hyper-parameters of the DSpark support module
        /// (a separate GGUF: <c>general.architecture = deepseek4-dspark</c>).</summary>
        public sealed class DsparkDesc
        {
            public int BlockSize;          // dspark.block_size (drafted tokens per step)
            public int NoiseTokenId;       // dspark.noise_token_id (the masked-slot token)
            public int MarkovRank;         // dspark.markov_rank
            public int[] TargetLayerIds;   // dspark.target_layer_ids (target blocks feeding main_proj)
            public LayerDesc[] Stages;     // one per dspark.n_layers, Ratio == 0

            public QuantWeightDesc MainProj;   // [n_target * n_embd, n_embd]
            public float[] MainNorm;           // [n_embd]
            public float[] Norm;               // final norm before the shared lm head
            public float[] HcHeadFn, HcHeadScale, HcHeadBase;

            public float[] MarkovW1;           // [n_vocab, rank], dequantized (gathered per token)
            public QuantWeightDesc MarkovW2;   // [rank, n_vocab]
            public float[] ConfProj;           // [n_embd + rank]
        }

        /// <summary>Device-resident DSpark state (one hosting device).</summary>
        private sealed class DsparkRt
        {
            public Dev Dev;
            public DsparkDesc Desc;
            public DevLayer[] Stages;
            public DevQW MainProj, MarkovW2;
            public Tensor MainNorm, Norm, HcHeadFn, HcHeadScale, HcHeadBase, MarkovW1, ConfProj;

            public int RingRows;           // drafter SWA ring modulus
            public int[] CaptureSlot;      // target layer id -> column block in CapH (-1: not captured)

            // scratch
            public Tensor CapH;            // [nUbatch, nTarget * n_embd] captured target features
            public Tensor MainH;           // [ringRows, nTarget * n_embd] host-fed features
            public Tensor MainX;           // [ringRows, n_embd] main_norm(main_proj(.))
            public Tensor BlockKv;         // [blockSize, head_dim] F16, the block's own keys
            public Tensor Head;            // [blockSize, n_embd] hc_head output (confidence input)
            public Tensor Logits;          // [blockSize, n_vocab] draft logits
            public Tensor Bias;            // [1, n_vocab] Markov bias
            public Tensor W1Rows;          // [blockSize, rank] W1[prev(i)]
            public Tensor Toks;            // [blockSize] drafted token ids
            public Tensor Conf;            // [blockSize] acceptance probabilities
            public Tensor BlockTokens;     // [blockSize] block input token ids

            public IntPtr PinnedCap;       // pinned staging for the feature readback
            public float[] HostConf;
            public int[] HostToks;
            public float[] HostBlockEmbd;  // staging when the embedding table is on another device
        }

        private DsparkRt _ds;

        /// <summary>TS_DSV4_DSPARK_CAPTURE=0 skips the target-feature capture
        /// (A/B knob: the drafter then reads stale features and its proposals
        /// are rejected, so only the timing is meaningful).</summary>
        private static readonly bool DsparkCaptureEnabled = EnvInt("TS_DSV4_DSPARK_CAPTURE", 1) != 0;

        // Set for the duration of a prefill ForwardSpec: each micro-batch feeds
        // the drafter's ring straight from the on-device capture.
        private bool _dsSelfCatchUp;

        // TS_DSV4_PERF>=1: cumulative wall time of the drafter's phases. Each
        // mark synchronizes the stream, so the numbers are serialized-phase
        // times -- use them to rank phases, not to predict throughput.
        private readonly double[] _dsPhaseMs = new double[6];
        private long _dsSteps;
        private Tensor _specLogits;        // [maxVerifyRows, n_vocab] on the last device
        private int _maxDraft;

        /// <summary>Draft block size, or 0 when the model has no DSpark module.</summary>
        public int DsparkBlockSize => _ds?.Desc.BlockSize ?? 0;

        /// <summary>Width of one row of <see cref="ForwardSpec"/>'s hidden capture
        /// (the concatenated target-layer features main_proj consumes).</summary>
        public int DsparkFeatureSize => _ds == null ? 0 : _ds.Desc.TargetLayerIds.Length * _m.NEmbd;

        // -------------------------------------------------------------------
        // setup
        // -------------------------------------------------------------------

        /// <summary>Bytes the drafter adds to its hosting device, so the layer
        /// split can leave room for them.</summary>
        private static long DsparkBytes(DsparkDesc d)
        {
            if (d == null)
                return 0;
            long b = 0;
            foreach (var s in d.Stages)
            {
                b += Align(s.WqA.TotalBytes) + Align(s.WqB.TotalBytes) + Align(s.Wkv.TotalBytes)
                   + Align(s.WoA.TotalBytes) + Align(s.WoB.TotalBytes)
                   + Align(s.GateExps.TotalBytes) + Align(s.UpExps.TotalBytes) + Align(s.DownExps.TotalBytes)
                   + Align(s.GateShexp.TotalBytes) + Align(s.UpShexp.TotalBytes) + Align(s.DownShexp.TotalBytes);
            }
            b += Align(d.MainProj.TotalBytes) + Align(d.MarkovW2.TotalBytes);
            return b;
        }

        /// <summary>
        /// Uploads the drafter onto the device that owns the output head, and
        /// allocates its rings and scratch. The three target feature layers must
        /// live on that same device: the drafter consumes their per-row stream
        /// means, and staging them across devices every ubatch would cost more
        /// than speculation saves.
        /// </summary>
        private void SetupDspark(DsparkDesc d)
        {
            var dev = _devs[_lastDev];
            dev.MakeCurrent();

            foreach (int lid in d.TargetLayerIds)
            {
                if (lid < 0 || lid >= _m.NLayer)
                    throw new InvalidOperationException($"[dsv4-cuda] DSpark target layer {lid} out of range");
                if (_layers[lid].Device != _lastDev)
                {
                    throw new NotSupportedException(
                        $"[dsv4-cuda] DSpark needs target layer {lid} on the output-head device " +
                        $"(it is on device {_layers[lid].Device}); reduce the GPU count or disable DSpark.");
                }
            }

            var rt = new DsparkRt
            {
                Dev = dev,
                Desc = d,
                Stages = new DevLayer[d.Stages.Length],
                RingRows = Pad(_m.NSwa + d.BlockSize + 1, 32),
                CaptureSlot = new int[_m.NLayer],
            };
            for (int i = 0; i < _m.NLayer; i++)
                rt.CaptureSlot[i] = -1;
            for (int i = 0; i < d.TargetLayerIds.Length; i++)
                rt.CaptureSlot[d.TargetLayerIds[i]] = i;

            int e = _m.NEmbd, hd = _m.HeadDim, b = d.BlockSize;
            int feat = d.TargetLayerIds.Length * e;

            for (int s = 0; s < d.Stages.Length; s++)
            {
                var src = d.Stages[s];
                var dst = new DevLayer { Device = _lastDev, Ratio = 0 };
                dst.ClampExp = src.ClampExp;
                dst.ClampShexp = src.ClampShexp;
                dst.WqA = UploadQuant(dev, src.WqA);
                dst.WqB = UploadQuant(dev, src.WqB);
                dst.Wkv = UploadQuant(dev, src.Wkv);
                dst.WoA = UploadQuant(dev, src.WoA);
                dst.WoB = UploadQuant(dev, src.WoB);
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
                dst.GateInp = UploadF32(dev, src.GateInp);
                dst.ExpProbsBias = UploadF32(dev, src.ExpProbsBias);
                dst.FfnNorm = UploadF32(dev, src.FfnNorm);
                dst.RingK = AllocT(dev, DType.Float16, rt.RingRows, hd);
                rt.Stages[s] = dst;
            }

            rt.MainProj = UploadQuant(dev, d.MainProj);
            rt.MarkovW2 = UploadQuant(dev, d.MarkovW2);
            rt.MainNorm = UploadF32(dev, d.MainNorm);
            rt.Norm = UploadF32(dev, d.Norm);
            rt.HcHeadFn = UploadF32(dev, d.HcHeadFn);
            rt.HcHeadScale = UploadF32(dev, d.HcHeadScale);
            rt.HcHeadBase = UploadF32(dev, d.HcHeadBase);
            rt.MarkovW1 = UploadF32(dev, d.MarkovW1);
            rt.ConfProj = UploadF32(dev, d.ConfProj);

            rt.CapH = AllocF32(dev, _m.NUbatch, feat);
            rt.MainH = AllocF32(dev, rt.RingRows, feat);
            rt.MainX = AllocF32(dev, rt.RingRows, e);
            rt.BlockKv = AllocT(dev, DType.Float16, b, hd);
            rt.Head = AllocF32(dev, b, e);
            rt.Logits = AllocF32(dev, b, _m.NVocab);
            rt.Bias = AllocF32(dev, 1, _m.NVocab);
            rt.W1Rows = AllocF32(dev, b, d.MarkovRank);
            rt.Toks = AllocI32(dev, b);
            rt.Conf = AllocF32(dev, b);
            rt.BlockTokens = AllocI32(dev, b);
            rt.HostConf = new float[b];
            rt.HostToks = new int[b];
            rt.HostBlockEmbd = new float[(long)b * HC * e];
            CudaDriverApi.cuMemHostAlloc(out rt.PinnedCap,
                new UIntPtr((ulong)((long)_m.NUbatch * feat * 4L)), 0x1 /*PORTABLE*/).ThrowOnError();

            _ds = rt;
            foreach (var st in rt.Stages)
                Memset0(st.RingK);

            Console.Error.WriteLine(
                $"[dsv4-cuda] DSpark drafter ready on device {dev.Ordinal}: {d.Stages.Length} stage(s), " +
                $"block_size={d.BlockSize}, markov_rank={d.MarkovRank}, " +
                $"target_layers=[{string.Join(",", d.TargetLayerIds)}], noise_token={d.NoiseTokenId}");
        }

        private void ResetDspark()
        {
            if (_ds == null)
                return;
            _ds.Dev.MakeCurrent();
            foreach (var st in _ds.Stages)
                Memset0(st.RingK);
        }

        // -------------------------------------------------------------------
        // trunk passes with feature capture
        // -------------------------------------------------------------------

        /// <summary>
        /// Trunk forward that also returns what speculation needs: the target
        /// features of every row (<paramref name="hAllOut"/>, rows of
        /// <see cref="DsparkFeatureSize"/> floats) and, with
        /// <paramref name="allLogitsRows"/>, the LM head over every row instead
        /// of only the last. Advances the KV caches exactly like
        /// <see cref="Forward"/>.
        /// </summary>
        /// <remarks>
        /// A hidden buffer too small to hold every row means PREFILL: the caller
        /// only wants the last row's features and the drafter replays its own key
        /// ring on-device. That keeps the whole prompt in ONE call with no host
        /// round trip between micro-batches -- a readback per chunk serializes the
        /// launch stream against the GPU and costs ~45% of prefill throughput.
        /// </remarks>
        public void ForwardSpec(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (_ds == null)
                throw new InvalidOperationException("[dsv4-cuda] no DSpark module loaded");
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            if (NPast + tokens.Length > _m.NCtx)
                throw new InvalidOperationException($"[dsv4-cuda] context overflow: n_past={NPast} + {tokens.Length} > n_ctx={_m.NCtx}");
            if (allLogitsRows && tokens.Length > _m.NUbatch)
                throw new NotSupportedException("[dsv4-cuda] per-row logits require a single ubatch");

            long feat = DsparkFeatureSize;
            bool lastRowOnly = hAllOut != null && hAllOut.LongLength < (long)tokens.Length * feat;
            _dsSelfCatchUp = lastRowOnly;

            var sw = _perf > 0 ? Stopwatch.StartNew() : null;
            int done = 0;
            int lastNt = 0;
            while (done < tokens.Length)
            {
                int nt = Math.Min(_m.NUbatch, tokens.Length - done);
                bool last = done + nt == tokens.Length;
                ForwardUbatch(tokens, done, nt, NPast, last ? logitsOut : null,
                    allLogitsRows, lastRowOnly ? null : hAllOut, done);
                NPast += nt;
                done += nt;
                lastNt = nt;
            }
            _dsSelfCatchUp = false;

            if (lastRowOnly && lastNt > 0)
            {
                var dev = _ds.Dev;
                dev.MakeCurrent();
                using Tensor row = Block(_ds.CapH, lastNt - 1, 1, feat);
                CudaDriverApi.cuMemcpyDtoHAsync(_ds.PinnedCap, Ptr(row), new UIntPtr((ulong)feat * 4), dev.Stream).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                fixed (float* dst = hAllOut)
                    Buffer.MemoryCopy((void*)_ds.PinnedCap, dst, hAllOut.LongLength * 4L, feat * 4L);
            }
            if (sw != null)
            {
                double secs = sw.Elapsed.TotalSeconds;
                Console.Error.WriteLine($"[dsv4-cuda] spec forward {tokens.Length} tokens in {secs:F3}s ({tokens.Length / secs:F1} tok/s)");
                ReportStages();
            }
        }

        /// <summary>
        /// Replays this micro-batch's committed positions into the drafter's key
        /// ring straight from the on-device capture (no host round trip). Only the
        /// last ringRows positions can survive the ring, and writing more than
        /// that would race two rows onto one slot.
        /// </summary>
        private void DsparkWriteRingFromCapture(int nt, int p0)
        {
            var rt = _ds;
            int keep = Math.Min(nt, rt.RingRows);
            int first = nt - keep;
            int feat = DsparkFeatureSize;

            using (Tensor rows = Block(rt.CapH, first, keep, feat))
            {
                MatMul(rt.Dev, rt.MainProj, rows, rt.MainX, keep);
            }
            RmsNorm(rt.Dev, rt.MainX, rt.MainNorm, keep);
            DsparkWriteRing(nt, p0, first);
        }

        /// <summary>Copies the captured features of this ubatch back to the host,
        /// through pinned staging (a direct copy into the managed array is staged
        /// by the driver in small synchronous chunks and costs ~0.4 s per prefill
        /// micro-batch).</summary>
        private void DsparkCaptureOut(int nt, float[] hAllOut, int rowOff)
        {
            var rt = _ds;
            long feat = DsparkFeatureSize;
            long count = (long)nt * feat;
            if ((long)rowOff * feat + count > hAllOut.LongLength)
                throw new ArgumentException("[dsv4-cuda] hidden capture buffer too small", nameof(hAllOut));
            rt.Dev.MakeCurrent();
            CudaDriverApi.cuMemcpyDtoHAsync(rt.PinnedCap, Ptr(rt.CapH), new UIntPtr((ulong)count * 4), rt.Dev.Stream).ThrowOnError();
            CudaDriverApi.cuStreamSynchronize(rt.Dev.Stream).ThrowOnError();
            fixed (float* dst = &hAllOut[(long)rowOff * feat])
                Buffer.MemoryCopy((void*)rt.PinnedCap, dst, count * 4L, count * 4L);
        }

        // -------------------------------------------------------------------
        // drafter passes
        // -------------------------------------------------------------------

        /// <summary>
        /// Uploads <paramref name="rows"/> rows of target features and projects
        /// them into <see cref="DsparkRt.MainX"/>, returning the first row index
        /// actually kept (rows older than the ring modulus are dropped, and
        /// dropping them is what keeps the ring writes collision-free).
        /// </summary>
        private int DsparkProject(float[] hRows, int rows, int hRowOff)
        {
            var rt = _ds;
            int feat = DsparkFeatureSize;
            int keep = Math.Min(rows, rt.RingRows);
            int first = rows - keep;

            rt.Dev.MakeCurrent();
            fixed (float* src = &hRows[(long)(hRowOff + first) * feat])
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(rt.MainH), (IntPtr)src,
                    new UIntPtr((ulong)((long)keep * feat) * 4), rt.Dev.Stream).ThrowOnError();
            }
            MatMul(rt.Dev, rt.MainProj, rt.MainH, rt.MainX, keep);
            RmsNorm(rt.Dev, rt.MainX, rt.MainNorm, keep);
            return first;
        }

        /// <summary>Writes the drafter's per-position keys for
        /// <paramref name="rows"/> committed positions starting at
        /// <paramref name="startPos"/>.</summary>
        private void DsparkWriteRing(int rows, int startPos, int firstKept)
        {
            var rt = _ds;
            var dev = rt.Dev;
            int keep = rows - firstKept;
            int pos0 = startPos + firstKept;

            foreach (var st in rt.Stages)
            {
                MatMul(dev, st.Wkv, rt.MainX, dev.KvRaw, keep);
                dev.DK.DsparkPrep(dev.Q, dev.KvRaw, Ptr(st.KvNorm), Ptr(dev.RopeRaw), Ptr(st.RingK),
                    pos0, pos0 % rt.RingRows, rt.RingRows, _m.NHead, _m.HeadDim, _m.NRot, _m.RmsEps,
                    keep, kvOnly: true, dev.Stream);
            }
        }

        /// <summary>
        /// Replays committed positions through the drafter so its key ring
        /// tracks the real context. Row k of <paramref name="hRows"/> holds the
        /// target features of position <paramref name="firstPos"/> + k; rows
        /// before position 0 (the zeroed "hidden state of the token before the
        /// prompt") are skipped.
        /// </summary>
        public void DsparkCatchUp(float[] hRows, int rows, int firstPos)
        {
            if (_ds == null || rows <= 0)
                return;

            long tc0 = _perf > 0 ? Stopwatch.GetTimestamp() : 0;

            int skip = firstPos < 0 ? Math.Min(-firstPos, rows) : 0;
            rows -= skip;
            firstPos += skip;
            if (rows <= 0)
                return;

            int kept = DsparkProject(hRows, rows, skip);
            DsparkWriteRing(rows, firstPos, kept);

            if (_perf > 0)
            {
                CudaDriverApi.cuStreamSynchronize(_ds.Dev.Stream);
                _dsPhaseMs[5] += (Stopwatch.GetTimestamp() - tc0) * 1000.0 / Stopwatch.Frequency;
                Console.Error.WriteLine($"[dsv4-cuda] dspark catch-up total {_dsPhaseMs[5]:F0}ms");
            }
        }

        /// <summary>
        /// Drafts one block. <paramref name="hPrev"/> holds the target features
        /// of the last FORWARDED position (<paramref name="position"/> - 1) and
        /// <paramref name="anchorToken"/> the token at <paramref name="position"/>
        /// (drawn but not yet forwarded). Returns the number of tokens written to
        /// <paramref name="draftOut"/>; <paramref name="confOut"/> receives their
        /// predicted acceptance probabilities.
        /// </summary>
        public int DsparkDraft(int anchorToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            var rt = _ds;
            if (rt == null || position <= 0)
                return 0;

            var dev = rt.Dev;
            var m = _m;
            int e = m.NEmbd, b = rt.Desc.BlockSize, rank = rt.Desc.MarkovRank;
            dev.MakeCurrent();

            bool timed = _perf > 0;
            long t0 = timed ? Stopwatch.GetTimestamp() : 0;
            void Phase(int i)
            {
                if (!timed)
                    return;
                CudaDriverApi.cuStreamSynchronize(dev.Stream);
                long now = Stopwatch.GetTimestamp();
                _dsPhaseMs[i] += (now - t0) * 1000.0 / Stopwatch.Frequency;
                t0 = now;
            }

            // The drafter's own key for the last committed position (the target
            // features of position-1), exactly like the reference's
            // kv_cache[start_pos % win] = main_kv.
            int kept = DsparkProject(hPrev, 1, 0);
            DsparkWriteRing(1, position - 1, kept);

            // Block input: [anchor, noise, ...] embedded and broadcast over the
            // hyper-connection streams.
            DsparkEmbedBlock(anchorToken);
            Phase(0);

            for (int s = 0; s < rt.Stages.Length; s++)
            {
                var st = rt.Stages[s];

                HcPre(dev, st, b, attn: true);
                RmsNorm(dev, dev.Cur, st.AttnNorm, b);
                DsparkAttention(st, position, b);
                dev.DK.HcPost(dev.Xs, dev.AttnOut, dev.Post, dev.Comb, dev.XsOut, b, e, dev.Stream);
                SwapXs(dev);

                HcPre(dev, st, b, attn: false);
                RmsNorm(dev, dev.Cur, st.FfnNorm, b);
                MoeFfn(dev, st, m.NLayer + s, b, null);
                dev.DK.HcPost(dev.Xs, dev.FfnOut, dev.Post, dev.Comb, dev.XsOut, b, e, dev.Stream);
                SwapXs(dev);
            }

            Phase(1);

            // Head: hc_head -> norm -> the TARGET's lm head.
            dev.DK.HcHead(dev.Xs, Ptr(rt.HcHeadFn), Ptr(rt.HcHeadScale), Ptr(rt.HcHeadBase), rt.Head, e,
                rt.Desc.HcHeadScale.Length, rt.Desc.HcHeadBase.Length, m.RmsEps, dev.Stream, b);
            {
                // The confidence head reads the pre-norm hc_head output, so norm
                // into Cur instead of in place.
                using Tensor cur = Rows(dev.Cur, b);
                using Tensor head = Rows(rt.Head, b);
                Ops.RMSNorm(cur, head, rt.Norm, null, m.RmsEps);
            }
            MatMul(dev, _outputQW, dev.Cur, rt.Logits, b);
            Phase(2);

            // Markov chain: position i is biased by W2 . W1[prev(i)] and its
            // argmax becomes prev(i+1).
            for (int i = 0; i < b; i++)
            {
                using Tensor w1Row = Block(rt.W1Rows, i, 1, rank);
                dev.DK.DsparkGather(Ptr(rt.MarkovW1), rt.Toks, i - 1, anchorToken, w1Row, rank, dev.Stream);
                MatMul(dev, rt.MarkovW2, w1Row, rt.Bias, 1);
                using Tensor lg = Block(rt.Logits, i, 1, m.NVocab);
                dev.DK.DsparkArgmax(lg, rt.Bias, rt.Toks, i, m.NVocab, dev.Stream);
            }

            Phase(3);

            dev.DK.DsparkConf(rt.Head, rt.W1Rows, Ptr(rt.ConfProj), rt.Conf, e, rank, b, dev.Stream);

            fixed (int* tp = rt.HostToks)
            fixed (float* cp = rt.HostConf)
            {
                CudaDriverApi.cuMemcpyDtoHAsync((IntPtr)tp, Ptr(rt.Toks), new UIntPtr((ulong)b * 4), dev.Stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoHAsync((IntPtr)cp, Ptr(rt.Conf), new UIntPtr((ulong)b * 4), dev.Stream).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }

            Phase(4);
            if (timed && (++_dsSteps % 64) == 0)
            {
                Console.Error.WriteLine(
                    $"[dsv4-cuda] dspark {_dsSteps} drafts: features={_dsPhaseMs[0]:F0}ms stages={_dsPhaseMs[1]:F0}ms " +
                    $"head={_dsPhaseMs[2]:F0}ms markov={_dsPhaseMs[3]:F0}ms conf+readback={_dsPhaseMs[4]:F0}ms");
            }

            int n = Math.Min(b, draftOut.Length);
            for (int i = 0; i < n; i++)
            {
                draftOut[i] = rt.HostToks[i];
                if (confOut != null && i < confOut.Length)
                    confOut[i] = rt.HostConf[i];
            }
            return n;
        }

        /// <summary>Embeds [anchor, noise x (B-1)] into the drafter's hyper-connection
        /// streams. The token embedding table lives on device 0, so when the drafter
        /// runs elsewhere the block (a few hundred KB) is staged through the host.</summary>
        private void DsparkEmbedBlock(int anchorToken)
        {
            var rt = _ds;
            var dev = rt.Dev;
            int b = rt.Desc.BlockSize, e = _m.NEmbd;
            var src = _devs[0];

            var ids = new int[b];
            ids[0] = anchorToken;
            for (int i = 1; i < b; i++)
                ids[i] = rt.Desc.NoiseTokenId;

            src.MakeCurrent();
            Tensor idsDev = ReferenceEquals(src, dev) ? rt.BlockTokens : src.TokensDev0;
            fixed (int* p = ids)
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(idsDev), (IntPtr)p, new UIntPtr((ulong)b * 4), src.Stream).ThrowOnError();
            src.DK.Embed(_tokEmbdQW.Ptr, idsDev, src.Xs, _tokEmbdQW.Type, _tokEmbdQW.RowBytes, b, e, src.Stream);

            if (ReferenceEquals(src, dev))
                return;

            long bytes = (long)b * HC * e * 4;
            fixed (float* h = rt.HostBlockEmbd)
            {
                CudaDriverApi.cuMemcpyDtoHAsync((IntPtr)h, Ptr(src.Xs), new UIntPtr((ulong)bytes), src.Stream).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(src.Stream).ThrowOnError();
                dev.MakeCurrent();
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Xs), (IntPtr)h, new UIntPtr((ulong)bytes), dev.Stream).ThrowOnError();
            }
        }

        /// <summary>
        /// Block attention for one drafter stage: queries at positions
        /// <paramref name="position"/> .. +B-1 over [committed window | the whole
        /// block], the block part non-causal (mode 3).
        /// </summary>
        private void DsparkAttention(DevLayer st, int position, int b)
        {
            var rt = _ds;
            var dev = rt.Dev;
            var m = _m;
            int nh = m.NHead, hd = m.HeadDim, rot = m.NRot;

            MatMul(dev, st.WqA, dev.Cur, dev.Qr, b);
            RmsNorm(dev, dev.Qr, st.QANorm, b);
            MatMul(dev, st.WqB, dev.Qr, dev.Q, b);
            MatMul(dev, st.Wkv, dev.Cur, dev.KvRaw, b);
            dev.DK.DsparkPrep(dev.Q, dev.KvRaw, Ptr(st.KvNorm), Ptr(dev.RopeRaw), Ptr(rt.BlockKv),
                position, 0, b, nh, hd, rot, m.RmsEps, b, kvOnly: false, dev.Stream);

            float kqScale = 1.0f / MathF.Sqrt(hd);
            dev.DK.Attention(dev.Q, Ptr(st.RingK), Ptr(rt.BlockKv), null, null, Ptr(st.Sinks), dev.AttnO,
                position - 1, m.NSwa, rt.RingRows, nh, hd, 3, 1, b, kqScale, b, dev.Stream);

            int hpg = nh / m.OGroups;
            dev.DK.AttnFinish(dev.AttnO, Ptr(dev.RopeRaw), dev.OGrouped, position, nh, hd, rot, hpg, b, dev.Stream);

            int groupDim = hpg * hd;
            for (int g = 0; g < m.OGroups; g++)
            {
                var slice = st.WoA;
                slice.Ptr = (IntPtr)((long)st.WoA.Ptr + (long)g * m.OLoraRank * st.WoA.RowBytes);
                slice.Ne1 = m.OLoraRank;
                using Tensor input = Block(dev.OGrouped, (long)g * b, b, groupDim);
                using Tensor output = Block(dev.OGroupedOut, (long)g * b, b, m.OLoraRank);
                MatMul(dev, slice, input, output, b);
            }
            dev.DK.Regroup(dev.OGroupedOut, dev.OG, m.OGroups, b, m.OLoraRank, dev.Stream);
            MatMul(dev, st.WoB, dev.OG, dev.AttnOut, b);
        }

        /// <summary>
        /// Drops the KV of rejected speculative tokens by rewinding the position
        /// counter. Nothing has to be restored: the raw ring and the compressor
        /// state rings are sized so a rejected tail can never alias a row the
        /// next pass still reads (see <c>StateRingExtra</c>), the compressed rows
        /// a rejected tail wrote are recomputed before they become visible, and
        /// the drafter's own ring only ever sees committed positions.
        /// </summary>
        public void Rewind(int nPast)
        {
            if (nPast < 0 || nPast > NPast)
                throw new ArgumentOutOfRangeException(nameof(nPast));
            NPast = nPast;
        }
    }
}
