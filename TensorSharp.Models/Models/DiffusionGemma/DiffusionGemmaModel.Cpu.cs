// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// The pure-C# CPU backend (BackendType.Cpu) forward for DiffusionGemma: prompt-KV caching on the
// host, a batched MoE, fused projections and the SIMD attention in DiffusionGemmaCpuKernels.
// Nothing here calls native code; every matmul goes through ManagedQuantizedOps.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    public sealed partial class DiffusionGemmaModel
    {
        // ---- A/B escape hatches (read once) ------------------------------------------------------
        // DIFFUSION_NO_PKV=1 (shared with the GPU backends) turns prompt-KV caching off, so every read
        // and every denoising step runs the unified [prompt|canvas] forward again. The switches below
        // restore the pre-existing CPU implementation of one stage each; DIFFUSION_CPU_LEGACY=1 does all.
        private static readonly bool CpuLegacyAll = Environment.GetEnvironmentVariable("DIFFUSION_CPU_LEGACY") == "1";
        private static readonly bool CpuLegacyMoe = CpuLegacyAll || Environment.GetEnvironmentVariable("DIFFUSION_CPU_LEGACY_MOE") == "1";
        private static readonly bool CpuLegacyProj = CpuLegacyAll || Environment.GetEnvironmentVariable("DIFFUSION_CPU_LEGACY_PROJ") == "1";
        private static readonly bool CpuLegacyAttn = CpuLegacyAll || Environment.GetEnvironmentVariable("DIFFUSION_CPU_LEGACY_ATTN") == "1";
        // The router is separate from the expert FFN so the batched FFN can be checked bitwise against
        // the reference loop under identical routing (the dot kernels round differently than the GEMM).
        private static readonly bool CpuLegacyRouter = CpuLegacyMoe || Environment.GetEnvironmentVariable("DIFFUSION_CPU_LEGACY_ROUTER") == "1";
        private static readonly bool CpuPoolDisabled = Environment.GetEnvironmentVariable("TS_CPU_POOL") == "0";

        // Tokens per batched-MoE pass. Bounds the gathered per-route scratch (a 4k-token prefill would
        // otherwise hold ~1 GB of routed rows) while still leaving ~32 rows per expert per pass.
        private static readonly int CpuMoeTokenChunk =
            int.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_CPU_MOE_CHUNK"), out int moeChunk) && moeChunk > 0 ? moeChunk : 512;

        /// <summary>True on the pure-C# CPU backend: prompt-KV caching runs on the host glue below
        /// (the device-glue backends keep their own implementation in DiffusionGemmaModel.cs).</summary>
        private bool UsesHostPromptKv => _backend == BackendType.Cpu;

        private bool CpuFastPaths => _backend == BackendType.Cpu;

        // growing host RoPE tables for absolute positions [0, _cpuRopeCap)
        private int _cpuRopeCap;
        private float[] _cpuCosLocal, _cpuSinLocal, _cpuCosGlobal, _cpuSinGlobal;
        private int[] _cpuIdentityPositions = Array.Empty<int>();

        // reusable batched-MoE scratch (pinned: the kernels take raw pointers across pool threads)
        private float[] _cpuMoeX, _cpuMoeGateUp, _cpuMoeAct, _cpuMoeY;
        // reusable canvas logits for the single-canvas paths (see the sampler's scBuffer contract)
        private float[] _cpuLogits;
        // stage timers for PrintForwardTiming (Stopwatch ticks)
        private long _tCpuQkv, _tCpuAttnCore, _tCpuMoeGateUp, _tCpuMoeDown;

        /// <summary>Zero the stage timers <see cref="PrintForwardTiming"/> reports (probes use it to
        /// profile warm reads without the first read's prefill).</summary>
        public void ResetForwardTiming()
        {
            _swForward.Reset();
            _tEmbed = _tAttn = _tMoe = _tDense = _tLmHead = _tSc = _tMoeRoute = _tMoeFfn = _tScTopK = _tScDevice = 0;
            _tCpuQkv = _tCpuAttnCore = _tCpuMoeGateUp = _tCpuMoeDown = 0;
        }

        private static void CpuFor(int count, Action<int> body)
        {
            if (count <= 0) return;
            if (count == 1) { body(0); return; }
            if (CpuPoolDisabled) Parallel.For(0, count, body);
            else CpuWorkerPool.Shared.For(count, body);
        }

        private static float[] GrowPinned(ref float[] buffer, long needed)
        {
            if (buffer == null || buffer.LongLength < needed)
                buffer = GC.AllocateUninitializedArray<float>(checked((int)Math.Max(needed, 1)), pinned: true);
            return buffer;
        }

        private void EnsureCpuRopeTables(int positions)
        {
            if (positions <= _cpuRopeCap) return;
            // BuildCosSin evaluates angle = p * freq per position, so a grown table holds exactly the
            // values a smaller one did - growth never changes an existing row. Bounded slack: a long
            // context would otherwise double into hundreds of MB of tables.
            int cap = Math.Max(positions, Math.Min(Math.Max(512, _cpuRopeCap * 2), positions + 2048));
            BuildCosSin(cap, _ropeFreqsLocal, out _cpuCosLocal, out _cpuSinLocal);
            BuildCosSin(cap, _ropeFreqsGlobal, out _cpuCosGlobal, out _cpuSinGlobal);
            _cpuRopeCap = cap;
        }

        private int[] IdentityPositions(int n)
        {
            if (_cpuIdentityPositions.Length < n)
            {
                var p = new int[Math.Max(n, _cpuIdentityPositions.Length * 2)];
                for (int i = 0; i < p.Length; i++) p[i] = i;
                _cpuIdentityPositions = p;
            }
            return _cpuIdentityPositions;
        }

        // =========================================================================================
        //  Projections
        // =========================================================================================

        /// <summary>Several projections of the SAME input in as few dispatches as possible: outputs whose
        /// weights share a quant type run as one <see cref="ManagedQuantizedOps.TryAddmmQuantizedBatch"/>
        /// (activations quantized once, one pool fork/join), which is what the Q/K/V and gate/up pairs
        /// need. Per row the arithmetic is that of separate linears. Anything the batch declines takes
        /// the ordinary managed linear. Outputs are [rows, Ne1]; a null output is skipped.</summary>
        private unsafe void CpuLinearMulti(Tensor input, string[] names, Tensor[] outputs)
        {
            int rows = (int)input.Sizes[0];
            int inDim = (int)input.Sizes[1];
            float* inPtr = GetFloatPtr(input);
            var done = new bool[names.Length];
            for (int i = 0; i < names.Length; i++)
            {
                if (done[i] || outputs[i] == null) continue;
                if (!_quantWeights.TryGetValue(names[i], out QuantizedWeight first) || !first.HasHostData || CpuLegacyProj)
                {
                    LinearInto(input, names[i], outputs[i]);
                    done[i] = true;
                    continue;
                }
                int count = 0;
                var jobs = new ManagedQuantizedOps.QuantMatMulJob[names.Length];
                var members = new int[names.Length];
                for (int j = i; j < names.Length; j++)
                {
                    if (done[j] || outputs[j] == null) continue;
                    if (!_quantWeights.TryGetValue(names[j], out QuantizedWeight qw) || !qw.HasHostData
                        || qw.GgmlType != first.GgmlType || qw.Ne0 != inDim)
                        continue;
                    jobs[count] = new ManagedQuantizedOps.QuantMatMulJob(qw.Data, (IntPtr)inPtr,
                        (IntPtr)GetFloatPtr(outputs[j]), (int)qw.Ne1, rows, (int)qw.Ne1);
                    members[count++] = j;
                }
                bool batched = ManagedQuantizedOps.TryAddmmQuantizedBatch(first.GgmlType, inDim, inDim,
                    new ReadOnlySpan<ManagedQuantizedOps.QuantMatMulJob>(jobs, 0, count));
                for (int m = 0; m < count; m++)
                {
                    int j = members[m];
                    QuantizedWeight qw = _quantWeights[names[j]];
                    if (!batched) AddmmQuantManaged(outputs[j], input, qw);
                    if (qw.Scale != 1.0f) Ops.Mul(outputs[j], outputs[j], qw.Scale);
                    done[j] = true;
                }
            }
        }

        /// <summary><see cref="ModelBase.LinearForward"/> into a caller-owned output.</summary>
        private void LinearInto(Tensor input, string name, Tensor output)
        {
            using Tensor r = LinearForward(input, name)
                ?? throw new InvalidOperationException($"Missing weight '{name}'.");
            Ops.Copy(output, r);
        }

        /// <summary>Q/K/V projection + per-head Q/K RMSNorm (weighted) + unweighted V RMSNorm + NeoX RoPE
        /// at absolute positions <paramref name="rowPos"/>, as flat token-major [rows, heads*hd] tensors.
        /// Global layers have no V projection (V = unweighted norm of the RAW K). The norm+RoPE pass is
        /// fused per row and bitwise identical to the Ops.RMSNorm + ApplyNeoXRoPERaw chain.
        /// <paramref name="needQ"/> = false skips the Q projection (the last prefill layer only needs K/V).</summary>
        private unsafe void CpuProjectQkv(Tensor normed, int layer, string prefix, int[] rowPos, bool needQ,
            out Tensor q, out Tensor k, out Tensor v)
        {
            int rows = (int)normed.Sizes[0];
            int hd = _headDim[layer];
            int qHeads = Config.NumHeads;
            int kvHeads = _kvHeads[layer];
            bool local = _isLocal[layer];
            bool hasV = _hasVProj[layer];
            float eps = Config.Eps;

            long ts = Stopwatch.GetTimestamp();
            q = needQ ? new Tensor(_allocator, DType.Float32, rows, qHeads * hd) : null;
            k = new Tensor(_allocator, DType.Float32, rows, kvHeads * hd);
            v = new Tensor(_allocator, DType.Float32, rows, kvHeads * hd);
            try
            {
                CpuLinearMulti(normed,
                    new[] { $"{prefix}.attn_q.weight", $"{prefix}.attn_k.weight", hasV ? $"{prefix}.attn_v.weight" : null },
                    new[] { q, k, hasV ? v : null });

                int maxPos = 0;
                for (int r = 0; r < rows; r++) if (rowPos[r] > maxPos) maxPos = rowPos[r];
                EnsureCpuRopeTables(maxPos + 1);
                float[] cosT = local ? _cpuCosLocal : _cpuCosGlobal;
                float[] sinT = local ? _cpuSinLocal : _cpuSinGlobal;
                int half = hd / 2;

                nint qa = q != null ? (nint)GetFloatPtr(q) : 0;
                nint ka = (nint)GetFloatPtr(k), va = (nint)GetFloatPtr(v);
                nint qw = (nint)GetFloatPtr(_weights[$"{prefix}.attn_q_norm.weight"]);
                nint kw = (nint)GetFloatPtr(_weights[$"{prefix}.attn_k_norm.weight"]);
                int qStride = qHeads * hd, kvStride = kvHeads * hd;
                // A row is a few microseconds of work; batch rows so an item carries tens of them.
                int rowsPerItem = Math.Max(1, 16384 / Math.Max(1, qStride + 2 * kvStride));
                int items = (rows + rowsPerItem - 1) / rowsPerItem;
                CpuFor(items, it =>
                {
                    fixed (float* cosP = cosT)
                    fixed (float* sinP = sinT)
                    {
                        int r1 = Math.Min(rows, (it + 1) * rowsPerItem);
                        for (int r = it * rowsPerItem; r < r1; r++)
                        {
                            float* kRow = (float*)ka + (long)r * kvStride;
                            float* vRow = (float*)va + (long)r * kvStride;
                            float* c = cosP + (long)rowPos[r] * half;
                            float* s = sinP + (long)rowPos[r] * half;
                            // V first: on global layers it is read from the raw (un-normed) K.
                            DiffusionGemmaCpuKernels.HeadNormRopeRow(hasV ? vRow : kRow, vRow, kvHeads, hd, null, eps, null, null);
                            DiffusionGemmaCpuKernels.HeadNormRopeRow(kRow, kRow, kvHeads, hd, (float*)kw, eps, c, s);
                            if (qa != 0)
                            {
                                float* qRow = (float*)qa + (long)r * qStride;
                                DiffusionGemmaCpuKernels.HeadNormRopeRow(qRow, qRow, qHeads, hd, (float*)qw, eps, c, s);
                            }
                        }
                    }
                });
                _tCpuQkv += Stopwatch.GetTimestamp() - ts;
            }
            catch
            {
                q?.Dispose(); k.Dispose(); v.Dispose();
                throw;
            }
        }

        // =========================================================================================
        //  Attention
        // =========================================================================================

        private unsafe void CpuAttend(Tensor q, Tensor o, DiffusionCpuAttnGroup[] groups, int layer)
        {
            int qHeads = Config.NumHeads, kvHeads = _kvHeads[layer], hd = _headDim[layer];
            int items = DiffusionGemmaCpuKernels.CountAttendItems(groups, qHeads, out int[] starts);
            nint qa = (nint)GetFloatPtr(q), oa = (nint)GetFloatPtr(o);
            long ts = Stopwatch.GetTimestamp();
            DiffusionAttnKernel kernel = DiffusionGemmaCpuKernels.DefaultAttention;
            CpuFor(items, item => DiffusionGemmaCpuKernels.AttendItem(item, groups, starts,
                (float*)qa, (float*)oa, qHeads, kvHeads, hd, kernel));
            _tCpuAttnCore += Stopwatch.GetTimestamp() - ts;
        }

        /// <summary>The unified / prefill attention group: queries and keys are the same rows
        /// [0, n) of one sequence whose first <paramref name="promptLen"/> rows are prompt, with the
        /// per-query interval from <see cref="AllowedRange"/> (causal + SWA prompt, bidirectional
        /// canvas, image spans widened on local layers).</summary>
        private unsafe DiffusionCpuAttnGroup SelfAttnGroup(Tensor k, Tensor v, int n, int promptLen, bool local)
        {
            var g = new DiffusionCpuAttnGroup
            {
                QStart = 0, QCount = n,
                KA = (nint)GetFloatPtr(k), VA = (nint)GetFloatPtr(v), LenA = n,
                Klo = new int[n], Khi = new int[n],
            };
            var spans = _visionSpans;
            for (int qi = 0; qi < n; qi++)
            {
                AllowedRange(qi, qi >= promptLen, promptLen, n, local, _slidingWindow, spans, out int klo, out int khi);
                g.Klo[qi] = klo; g.Khi[qi] = khi;
            }
            return g;
        }

        /// <summary>Unified [prompt|canvas] attention block on the CPU fast path (the replacement for the
        /// AttentionRegionAware branch of <see cref="Attention"/>). Returns the attn_output projection.</summary>
        private Tensor CpuUnifiedAttention(Tensor input, int layer, string prefix, int N, int P)
        {
            CpuProjectQkv(input, layer, prefix, IdentityPositions(N), needQ: true, out Tensor q, out Tensor k, out Tensor v);
            using (q) using (k) using (v)
            {
                var groups = new[] { SelfAttnGroup(k, v, N, P, _isLocal[layer]) };
                using var attn = new Tensor(_allocator, DType.Float32, N, Config.NumHeads * _headDim[layer]);
                CpuAttend(q, attn, groups, layer);
                return LinearForward(attn, $"{prefix}.attn_output.weight");
            }
        }

        // =========================================================================================
        //  Prompt prefill (host prompt-KV)
        // =========================================================================================

        /// <summary>Host prompt prefill: every layer's prompt K/V (after norm + RoPE) is kept as a flat
        /// token-major [rows, kvHeads*hd] tensor. Canvas queries never see more than the last
        /// (sliding_window-1) prompt keys on a local layer, so for a longer prompt only those rows are
        /// kept (the decode addresses them from row 0); global layers keep all P rows. The last layer
        /// only needs K/V: its prompt attention output and FFN would feed nothing, so they are skipped
        /// (1/30 of the prefill).</summary>
        private int PrefillPromptIntoCpu(int[] promptTokens, Tensor[] outK, Tensor[] outV, CancellationToken cancellationToken)
        {
            int P = promptTokens.Length;
            int D = Config.HiddenSize;
            float eps = Config.Eps;
            int L = Config.NumLayers;
            int[] rowPos = IdentityPositions(P);

            long ts = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(promptTokens);
            try
            {
                Ops.Mul(hidden, hidden, MathF.Sqrt(D));   // prompt = embed*sqrt(n_embd) (no rms-norm, no SC)
                ApplyPendingVisionEmbeddings(hidden, P);
                _tEmbed += Stopwatch.GetTimestamp() - ts;

                for (int l = 0; l < L; l++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    string prefix = $"blk.{l}";
                    bool local = _isLocal[l];
                    bool last = l == L - 1;

                    long tA = Stopwatch.GetTimestamp();
                    Tensor q, k, v;
                    using (Tensor normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight"))
                        CpuProjectQkv(normed, l, prefix, rowPos, needQ: !last, out q, out k, out v);

                    Tensor attn = null;
                    try
                    {
                        if (!last)
                        {
                            var groups = new[] { SelfAttnGroup(k, v, P, P, local) };
                            attn = new Tensor(_allocator, DType.Float32, P, Config.NumHeads * _headDim[l]);
                            CpuAttend(q, attn, groups, l);
                        }
                        int keep = local ? Math.Min(P, _slidingWindow - 1) : P;
                        if (keep == P)
                        {
                            outK[l] = k; k = null;   // ownership moves to the prompt store
                            outV[l] = v; v = null;
                        }
                        else
                        {
                            outK[l] = CopyLastRows(k, keep);
                            outV[l] = CopyLastRows(v, keep);
                        }
                    }
                    catch
                    {
                        attn?.Dispose();
                        throw;
                    }
                    finally
                    {
                        q?.Dispose(); k?.Dispose(); v?.Dispose();
                    }
                    if (last) { _tAttn += Stopwatch.GetTimestamp() - tA; break; }

                    Tensor attnOut;
                    using (attn) attnOut = LinearForward(attn, $"{prefix}.attn_output.weight");
                    using (Tensor residual = hidden)
                    {
                        hidden = attnOut;
                        Ops.RMSNorm(hidden, hidden, _weights[$"{prefix}.post_attention_norm.weight"], null, eps);
                        Ops.Add(hidden, hidden, residual);
                    }
                    _tAttn += Stopwatch.GetTimestamp() - tA;

                    hidden = FeedForward(hidden, l, prefix, P);
                    if (_encScale[l] != 1f) Ops.Mul(hidden, hidden, _encScale[l]);   // encoder scalar
                }
                return P;
            }
            finally { hidden.Dispose(); }
        }

        private unsafe Tensor CopyLastRows(Tensor t, int keep)
        {
            long rows = t.Sizes[0], cols = t.Sizes[1];
            var r = new Tensor(_allocator, DType.Float32, keep, cols);
            long bytes = keep * cols * sizeof(float);
            Buffer.MemoryCopy(GetFloatPtr(t) + (rows - keep) * cols, GetFloatPtr(r), bytes, bytes);
            return r;
        }

        // =========================================================================================
        //  Canvas decode (host prompt-KV), one or several sequences per forward
        // =========================================================================================

        private sealed class CpuDecodeSeq
        {
            public Tensor[] PromptK, PromptV;
            public int PromptLen;
            public int[] Canvas;
            public float[] ScPrev;
            public float ScUse;
            public float PrevTempInv = 1f;
        }

        /// <summary>Canvas-only forward over every sequence's canvas rows at once: the weight-bound work
        /// (projections, dense MLP, MoE) runs over all rows, attention per sequence against its cached
        /// prompt K/V + its own fresh canvas K/V, with exactly the unified forward's canvas mask (all
        /// canvas keys; prompt keys from max(0, P-swa+1) on local layers, all on global ones). Canvas
        /// token i of a sequence with prompt length P sits at RoPE position P+i. Returns [sum C, D].
        ///
        /// <paramref name="keepRows"/> (one sequence only; sorted, distinct canvas rows) returns just
        /// those rows, in that order. Every row is still needed as a key in the last layer, but only
        /// the kept rows' queries, attention, output projection and FFN are - which is all a structured
        /// read looks at - so the last layer runs its Q projection and FFN over a couple of rows
        /// instead of the whole canvas. Row-independent, so the kept rows are bitwise the full result.</summary>
        private unsafe Tensor CpuDecodeHidden(CpuDecodeSeq[] seqs, CancellationToken cancellationToken, int[] keepRows = null)
        {
            if (keepRows != null && (seqs.Length != 1 || keepRows.Length == 0 || keepRows.Length >= seqs[0].Canvas.Length))
                keepRows = null;
            int B = seqs.Length;
            int D = Config.HiddenSize;
            float eps = Config.Eps;
            var rowStart = new int[B + 1];
            for (int b = 0; b < B; b++)
            {
                var s = seqs[b];
                if (s.PromptK == null || s.PromptV == null || s.PromptLen <= 0 || s.PromptK[0] == null)
                    throw new InvalidOperationException("No prompt has been prefilled for this canvas decode.");
                rowStart[b + 1] = rowStart[b] + s.Canvas.Length;
            }
            int R = rowStart[B];

            var tokens = new int[R];
            var rowPos = new int[R];
            for (int b = 0; b < B; b++)
            {
                Array.Copy(seqs[b].Canvas, 0, tokens, rowStart[b], seqs[b].Canvas.Length);
                for (int i = 0; i < seqs[b].Canvas.Length; i++) rowPos[rowStart[b] + i] = seqs[b].PromptLen + i;
            }

            long ts = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            try
            {
                Ops.Mul(hidden, hidden, MathF.Sqrt(D));
                for (int b = 0; b < B; b++)
                {
                    var s = seqs[b];
                    if (!_scEnabled || s.ScPrev == null || s.ScUse == 0f) continue;
                    long tsc = Stopwatch.GetTimestamp();
                    using var scSignal = ComputeSelfConditioning(s.ScPrev, s.Canvas.Length, s.PrevTempInv);
                    Ops.Mul(scSignal, scSignal, s.ScUse);
                    using var slice = hidden.Narrow(0, rowStart[b], s.Canvas.Length);
                    Ops.Add(slice, slice, scSignal);
                    _tSc += Stopwatch.GetTimestamp() - tsc;
                }
                Ops.RMSNorm(hidden, hidden, GetOnes(D), null, eps);   // canvas = rms_norm_noscale(embed [+ SC])
                _tEmbed += Stopwatch.GetTimestamp() - ts;

                for (int l = 0; l < Config.NumLayers; l++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    string prefix = $"blk.{l}";
                    bool local = _isLocal[l];
                    int hd = _headDim[l];
                    int kvStride = _kvHeads[l] * hd;
                    bool prune = keepRows != null && l == Config.NumLayers - 1;

                    long tA = Stopwatch.GetTimestamp();
                    Tensor q, k, v;
                    using (Tensor normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight"))
                    {
                        CpuProjectQkv(normed, l, prefix, rowPos, needQ: !prune, out q, out k, out v);
                        if (prune)
                        {
                            try
                            {
                                using Tensor normedKeep = CopyRowsCpu(normed, keepRows);
                                var keepPos = new int[keepRows.Length];
                                for (int i = 0; i < keepRows.Length; i++) keepPos[i] = rowPos[keepRows[i]];
                                q = CpuProjectQ(normedKeep, l, prefix, keepPos);
                            }
                            catch
                            {
                                k.Dispose(); v.Dispose();
                                throw;
                            }
                        }
                    }

                    Tensor attnOut;
                    using (q) using (k) using (v)
                    {
                        float* kp = GetFloatPtr(k), vp = GetFloatPtr(v);
                        var groups = new DiffusionCpuAttnGroup[B];
                        for (int b = 0; b < B; b++)
                        {
                            var s = seqs[b];
                            Tensor pk = s.PromptK[l], pv = s.PromptV[l];
                            int stored = (int)pk.Sizes[0];   // the prefill may keep only the SWA tail
                            int allowed = local ? Math.Min(s.PromptLen, _slidingWindow - 1) : s.PromptLen;
                            if (stored < allowed || pk.Sizes[1] != kvStride)
                                throw new InvalidOperationException(
                                    $"Cached prompt K/V for layer {l} is [{pk.Sizes[0]}, {pk.Sizes[1]}]; expected at least {allowed} x {kvStride}.");
                            int C = s.Canvas.Length;
                            groups[b] = new DiffusionCpuAttnGroup
                            {
                                QStart = prune ? 0 : rowStart[b], QCount = prune ? keepRows.Length : C,
                                KA = (nint)GetFloatPtr(pk), VA = (nint)GetFloatPtr(pv), LenA = stored,
                                KB = (nint)(kp + (long)rowStart[b] * kvStride), VB = (nint)(vp + (long)rowStart[b] * kvStride),
                                LenB = C,
                                UniformLo = stored - allowed, UniformHi = stored + C,
                            };
                        }
                        using var attn = new Tensor(_allocator, DType.Float32, prune ? keepRows.Length : R, Config.NumHeads * hd);
                        CpuAttend(q, attn, groups, l);
                        attnOut = LinearForward(attn, $"{prefix}.attn_output.weight");
                    }

                    using (Tensor residual = prune ? CopyRowsCpu(hidden, keepRows) : hidden)
                    {
                        if (prune) hidden.Dispose();
                        hidden = attnOut;
                        Ops.RMSNorm(hidden, hidden, _weights[$"{prefix}.post_attention_norm.weight"], null, eps);
                        Ops.Add(hidden, hidden, residual);
                    }
                    _tAttn += Stopwatch.GetTimestamp() - tA;

                    hidden = FeedForward(hidden, l, prefix, prune ? keepRows.Length : R);
                    if (_decScale[l] != 1f) Ops.Mul(hidden, hidden, _decScale[l]);   // decoder scalar
                }
                return hidden;
            }
            catch
            {
                hidden.Dispose();
                throw;
            }
        }

        private Tensor CpuDecodeHidden(Tensor[] pk, Tensor[] pv, int P, int[] canvasTokens,
            float[] scPrevLogits, float scUse, float prevTempInv, CancellationToken cancellationToken, int[] keepRows = null)
        {
            return CpuDecodeHidden(new[]
            {
                new CpuDecodeSeq
                {
                    PromptK = pk, PromptV = pv, PromptLen = P, Canvas = canvasTokens,
                    ScPrev = scPrevLogits, ScUse = scUse, PrevTempInv = prevTempInv,
                },
            }, cancellationToken, keepRows);
        }

        /// <summary>Q projection + weighted per-head RMSNorm + NeoX RoPE for a subset of rows (the pruned
        /// last decode layer); the same per-row arithmetic as <see cref="CpuProjectQkv"/>.</summary>
        private unsafe Tensor CpuProjectQ(Tensor normedRows, int layer, string prefix, int[] rowPos)
        {
            int rows = (int)normedRows.Sizes[0];
            int hd = _headDim[layer];
            int qHeads = Config.NumHeads;
            var q = new Tensor(_allocator, DType.Float32, rows, qHeads * hd);
            try
            {
                CpuLinearMulti(normedRows, new[] { $"{prefix}.attn_q.weight" }, new[] { q });
                int maxPos = 0;
                for (int r = 0; r < rows; r++) if (rowPos[r] > maxPos) maxPos = rowPos[r];
                EnsureCpuRopeTables(maxPos + 1);
                bool local = _isLocal[layer];
                float[] cosT = local ? _cpuCosLocal : _cpuCosGlobal;
                float[] sinT = local ? _cpuSinLocal : _cpuSinGlobal;
                int half = hd / 2;
                float* qw = GetFloatPtr(_weights[$"{prefix}.attn_q_norm.weight"]);
                float* qp = GetFloatPtr(q);
                fixed (float* cosP = cosT)
                fixed (float* sinP = sinT)
                    for (int r = 0; r < rows; r++)
                    {
                        float* row = qp + (long)r * qHeads * hd;
                        DiffusionGemmaCpuKernels.HeadNormRopeRow(row, row, qHeads, hd, qw, Config.Eps,
                            cosP + (long)rowPos[r] * half, sinP + (long)rowPos[r] * half);
                    }
                return q;
            }
            catch
            {
                q.Dispose();
                throw;
            }
        }

        private unsafe Tensor CopyRowsCpu(Tensor src, int[] rows)
        {
            long cols = src.Sizes[1];
            var r = new Tensor(_allocator, DType.Float32, rows.Length, cols);
            long bytes = cols * sizeof(float);
            float* s = GetFloatPtr(src), d = GetFloatPtr(r);
            for (int i = 0; i < rows.Length; i++)
                Buffer.MemoryCopy(s + rows[i] * cols, d + i * cols, bytes, bytes);
            return r;
        }

        /// <summary>Host lm_head tail: output_norm + tied lm_head + scale + final-logit softcap, written
        /// straight into a float[] (no [rows, vocab] tensor + second host copy; at a 256-token canvas each
        /// was 268 MB per step). <paramref name="pooled"/> reuses one buffer, under the same contract as
        /// the GGML paths' pooled logits: the caller consumes it before the next forward overwrites it.
        /// Returns null when the embedding is not a host quantized weight (caller uses the Ops chain).</summary>
        private unsafe float[] CpuLmHead(Tensor hidden, int rowOffset, int rows, bool pooled)
        {
            if (!_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight head) || !head.HasHostData)
                return null;
            long ts = Stopwatch.GetTimestamp();
            int D = Config.HiddenSize;
            int vocab = Config.VocabSize;
            long total = (long)rows * vocab;
            float[] logits = pooled ? GrowPinnedExact(ref _cpuLogits, total) : new float[total];
            using (Tensor slice = hidden.Narrow(0, rowOffset, rows))
            using (Tensor normed = RMSNormOp(slice, "output_norm.weight"))
            fixed (float* dst = logits)
            {
                ManagedQuantizedOps.AddmmQuantizedToFloat32(head.GgmlType, head.Data, head.Ne0, head.Ne1,
                    GetFloatPtr(normed), D, rows, dst, vocab);
            }
            // scale, then softcap = tanh(x * (1/cap)) * cap: the reference Ops.Mul / Ops.Tanh (scalar
            // MathF.Tanh) / Ops.Mul chain element for element, just across the pool instead of one
            // thread over 67M logits.
            float scale = head.Scale;
            float cap = _finalLogitSoftcap;
            if (scale != 1f || cap > 0f)
            {
                float inv = cap > 0f ? 1f / cap : 0f;
                CpuFor(rows, r =>
                {
                    Span<float> row = logits.AsSpan(r * vocab, vocab);
                    if (scale != 1f) TensorPrimitives.Multiply(row, scale, row);
                    if (cap > 0f)
                        for (int i = 0; i < row.Length; i++) row[i] = MathF.Tanh(row[i] * inv) * cap;
                });
            }
            _tLmHead += Stopwatch.GetTimestamp() - ts;
            return logits;
        }

        private static float[] GrowPinnedExact(ref float[] buffer, long needed)
        {
            // Callers index rows off the returned array and some read its Length, so it must be exact.
            if (buffer == null || buffer.LongLength != needed)
                buffer = GC.AllocateUninitializedArray<float>(checked((int)needed), pinned: true);
            return buffer;
        }

        /// <summary>Batched decode on the pure-C# backend: every sequence's canvas rows share one forward
        /// (each routed expert and each projection then runs over more rows per weight read), attention
        /// stays per sequence. Each sequence gets its own logits array.</summary>
        private float[][] CpuDecodeCanvasBatched(DiffusionSeqState[] seqs, int[][] canvases,
            float[][] scPrev, float[] scUse, float[] prevTempInv)
        {
            int B = seqs.Length;
            var input = new CpuDecodeSeq[B];
            for (int b = 0; b < B; b++)
            {
                input[b] = new CpuDecodeSeq
                {
                    PromptK = seqs[b].PromptK, PromptV = seqs[b].PromptV, PromptLen = seqs[b].PromptLen,
                    Canvas = canvases[b], ScPrev = scPrev?[b], ScUse = scUse[b], PrevTempInv = prevTempInv[b],
                };
            }
            _swForward.Start();
            try
            {
                using Tensor hidden = CpuDecodeHidden(input, CancellationToken.None);
                var results = new float[B][];
                int row = 0;
                for (int b = 0; b < B; b++)
                {
                    int C = canvases[b].Length;
                    results[b] = CpuLmHead(hidden, row, C, pooled: false) ?? OpsLmHeadFresh(hidden, row, C);
                    row += C;
                }
                return results;
            }
            finally { _swForward.Stop(); }
        }

        private float[] OpsLmHeadFresh(Tensor hidden, int rowOffset, int rows)
        {
            using Tensor slice = hidden.Narrow(0, rowOffset, rows);
            using Tensor contiguous = Ops.NewContiguous(slice);
            using Tensor normed = RMSNormOp(contiguous, "output_norm.weight");
            using Tensor logits = LinearForward(normed, "token_embd.weight");
            if (_finalLogitSoftcap > 0f)
            {
                Ops.Mul(logits, logits, 1f / _finalLogitSoftcap);
                Ops.Tanh(logits, logits);
                Ops.Mul(logits, logits, _finalLogitSoftcap);
            }
            return logits.GetElementsAsFloat(checked(rows * Config.VocabSize));
        }

        // =========================================================================================
        //  Dense MLP and MoE
        // =========================================================================================

        /// <summary>Dense gated-GELU MLP with gate and up in one batched dispatch and a parallel
        /// GELU*up. Returns null when the weights are not the expected quantized pair.</summary>
        private unsafe Tensor CpuDenseMlp(Tensor input, string prefix, int N)
        {
            string gateName = $"{prefix}.ffn_gate.weight", upName = $"{prefix}.ffn_up.weight";
            if (!_quantWeights.TryGetValue(gateName, out QuantizedWeight gw) || !_quantWeights.TryGetValue(upName, out QuantizedWeight uw)
                || gw.Ne1 != uw.Ne1)
                return null;
            int ff = (int)gw.Ne1;
            using var normed = RMSNormOp(input, $"{prefix}.ffn_norm.weight");
            var gate = new Tensor(_allocator, DType.Float32, N, ff);
            try
            {
                using var up = new Tensor(_allocator, DType.Float32, N, ff);
                CpuLinearMulti(normed, new[] { gateName, upName }, new[] { gate, up });
                GeluMulRows((nint)GetFloatPtr(gate), ff, (nint)GetFloatPtr(up), ff, (nint)GetFloatPtr(gate), ff, N, ff);
                return LinearForward(gate, $"{prefix}.ffn_down.weight");
            }
            finally { gate.Dispose(); }
        }

        /// <summary>dst[r] = gelu(gate[r]) * up[r] for <paramref name="rows"/> rows of <paramref name="n"/>,
        /// in parallel row blocks (dst may alias gate).</summary>
        private static unsafe void GeluMulRows(nint gate, int gateStride, nint up, int upStride, nint dst, int dstStride,
            int rows, int n)
        {
            int rowsPerItem = Math.Max(1, 8192 / Math.Max(1, n));
            int items = (rows + rowsPerItem - 1) / rowsPerItem;
            CpuFor(items, it =>
            {
                int r1 = Math.Min(rows, (it + 1) * rowsPerItem);
                for (int r = it * rowsPerItem; r < r1; r++)
                    DiffusionGemmaCpuKernels.GeluMulRow((float*)gate + (long)r * gateStride, (float*)up + (long)r * upStride,
                        (float*)dst + (long)r * dstStride, n);
            });
        }

        /// <summary>
        /// The 128-expert top-8 MoE FFN on the pure-C# backend, batched across experts.
        ///
        /// The reference loop (still behind DIFFUSION_CPU_LEGACY_MOE=1) ran each active expert on its
        /// own: two Tensor allocations, a row gather, two linear dispatches (each a separate pool
        /// fork/join and activation quantization), a GELU and a scalar scatter - ~2x128 matmul
        /// dispatches per layer, most of them one or two rows wide, and 59% of a Jev read. Here:
        ///  1. counting-sort the (token, slot) routes by expert and gather their rows once;
        ///  2. ALL experts' gate_up projections under ONE <see cref="ManagedQuantizedOps.TryAddmmQuantizedBatch"/>
        ///     over the stacked expert tensor, cut into cost-balanced column slices so a hot expert
        ///     does not become the straggler of the fork/join;
        ///  3. GELU(gate)*up, parallel (exactly Ops.GELUMul's arithmetic);
        ///  4. all down projections under one more batch;
        ///  5. per token, the routing-weighted sum of its experts' rows (scale folded exactly as the
        ///     reference: w * (s_e * y), ascending expert order, from zero), parallel over tokens.
        /// Row for row these are the same dot products as the reference.
        /// </summary>
        private unsafe bool CpuMoEFfn(Tensor moeInput, Tensor output, int[] selected, float[] routing, int layer, int N, int D)
        {
            StackedExpertWeights gu = _stackedGateUp[layer];
            StackedExpertWeights dn = _stackedDown[layer];
            if (gu == null || dn == null || gu.PerExpertNe0 != D || dn.PerExpertNe1 != D || gu.PerExpertNe1 % 2 != 0
                || gu.NumExperts != _numExperts || dn.NumExperts != _numExperts)
                return false;
            int F = (int)(gu.PerExpertNe1 / 2);
            if (dn.PerExpertNe0 != F) return false;

            float* inPtr = GetFloatPtr(moeInput);
            float* outPtr = GetFloatPtr(output);
            for (int t0 = 0; t0 < N; t0 += CpuMoeTokenChunk)
            {
                int n = Math.Min(CpuMoeTokenChunk, N - t0);
                CpuMoEChunk(inPtr + (long)t0 * D, outPtr + (long)t0 * D, selected, routing, t0, n, layer, D, F, gu, dn);
            }
            return true;
        }

        private unsafe void CpuMoEChunk(float* input, float* output, int[] selected, float[] routing, int tokenBase,
            int N, int layer, int D, int F, StackedExpertWeights gu, StackedExpertWeights dn)
        {
            int E = _numExperts, K = _numExpertsUsed;
            int NK = N * K;
            float[] expertScale = _perExpertScale[layer];

            // 1) counting sort of the routes by expert; within an expert, token order (as the reference).
            var offset = new int[E + 1];
            for (int p = 0; p < NK; p++) offset[selected[tokenBase * K + p] + 1]++;
            for (int e = 0; e < E; e++) offset[e + 1] += offset[e];
            var cursor = (int[])offset.Clone();
            var rowToken = new int[NK];
            var pairRow = new int[NK];
            for (int p = 0; p < NK; p++)
            {
                int r = cursor[selected[tokenBase * K + p]]++;
                rowToken[r] = p / K;
                pairRow[p] = r;
            }

            float[] xs = GrowPinned(ref _cpuMoeX, (long)NK * D);
            float[] gus = GrowPinned(ref _cpuMoeGateUp, (long)NK * 2 * F);
            float[] acts = GrowPinned(ref _cpuMoeAct, (long)NK * F);
            float[] ys = GrowPinned(ref _cpuMoeY, (long)NK * D);
            fixed (float* x = xs) fixed (float* gUp = gus) fixed (float* act = acts) fixed (float* y = ys)
            {
                nint xa = (nint)x, ia = (nint)input;
                long rowBytes = (long)D * sizeof(float);
                const int gatherRows = 64;
                CpuFor((NK + gatherRows - 1) / gatherRows, it =>
                {
                    int r1 = Math.Min(NK, (it + 1) * gatherRows);
                    for (int r = it * gatherRows; r < r1; r++)
                        Buffer.MemoryCopy((float*)ia + (long)rowToken[r] * D, (float*)xa + (long)r * D, rowBytes, rowBytes);
                });

                // 2) gate_up for every expert, one dispatch
                long ts = Stopwatch.GetTimestamp();
                RunExpertBatch(gu, offset, x, D, gUp, 2 * F);
                _tCpuMoeGateUp += Stopwatch.GetTimestamp() - ts;
                // 3) GEGLU: gate = first F columns, up = last F
                GeluMulRows((nint)gUp, 2 * F, (nint)(gUp + F), 2 * F, (nint)act, F, NK, F);
                // 4) down for every expert, one dispatch
                ts = Stopwatch.GetTimestamp();
                RunExpertBatch(dn, offset, act, F, y, D);
                _tCpuMoeDown += Stopwatch.GetTimestamp() - ts;

                // 5) weighted combine per token
                nint ya = (nint)y, oa = (nint)output;
                const int tokensPerItem = 16;
                CpuFor((N + tokensPerItem - 1) / tokensPerItem, it =>
                {
                    float** rows = stackalloc float*[K];
                    float* w = stackalloc float[K];
                    float* s = stackalloc float[K];
                    int* ex = stackalloc int[K];
                    int s1 = Math.Min(N, (it + 1) * tokensPerItem);
                    for (int t = it * tokensPerItem; t < s1; t++)
                    {
                        int pBase = (tokenBase + t) * K;
                        bool scaled = false;
                        for (int u = 0; u < K; u++)
                        {
                            // insertion by expert id: the reference accumulated in ascending expert order
                            int e = selected[pBase + u];
                            int at = u;
                            while (at > 0 && ex[at - 1] > e)
                            {
                                ex[at] = ex[at - 1]; rows[at] = rows[at - 1]; w[at] = w[at - 1]; s[at] = s[at - 1];
                                at--;
                            }
                            ex[at] = e;
                            rows[at] = (float*)ya + (long)pairRow[t * K + u] * D;
                            w[at] = routing[pBase + u];
                            s[at] = expertScale != null ? expertScale[e] : 1f;
                            scaled |= s[at] != 1f;
                        }
                        DiffusionGemmaCpuKernels.WeightedRowSum((float*)oa + (long)t * D, D, rows, w, scaled ? s : null, K);
                    }
                });
            }
        }

        /// <summary>All active experts' <c>out = in * W_e^T</c> for one stacked expert tensor, as one
        /// batched dispatch. Experts are cut into column slices sized from the batch's total work
        /// (rows x columns) so the fork/join has a few items per worker and no single hot expert
        /// dominates its tail. Falls back to one managed linear per expert when the weight type has no
        /// direct quantized-dot plan.</summary>
        private unsafe void RunExpertBatch(StackedExpertWeights w, int[] offset, float* input, int inDim,
            float* output, int outCols)
        {
            int E = _numExperts;
            long rowBytes = ManagedQuantizedOps.RowSize(w.GgmlType, inDim);
            long perExpert = w.PerExpertRawBytes;
            byte* baseW = (byte*)w.Data;

            long totalWork = 0;
            int active = 0;
            for (int e = 0; e < E; e++)
            {
                int cnt = offset[e + 1] - offset[e];
                if (cnt == 0) continue;
                totalWork += (long)cnt * outCols;
                active++;
            }
            if (active == 0) return;
            int threads = CpuPoolDisabled ? Environment.ProcessorCount : CpuWorkerPool.Shared.ThreadCount;
            long perItem = Math.Max(1, totalWork / Math.Max(1, threads * 4));

            var jobs = new List<ManagedQuantizedOps.QuantMatMulJob>(active * 2);
            for (int e = 0; e < E; e++)
            {
                int cnt = offset[e + 1] - offset[e];
                if (cnt == 0) continue;
                long work = (long)cnt * outCols;
                int pieces = (int)Math.Clamp((work + perItem - 1) / perItem, 1, Math.Max(1, outCols / 64));
                int colsPer = (outCols + pieces - 1) / pieces;
                float* inE = input + (long)offset[e] * inDim;
                float* outE = output + (long)offset[e] * outCols;
                for (int c0 = 0; c0 < outCols; c0 += colsPer)
                {
                    int cols = Math.Min(colsPer, outCols - c0);
                    jobs.Add(new ManagedQuantizedOps.QuantMatMulJob(
                        (IntPtr)(baseW + e * perExpert + c0 * rowBytes), (IntPtr)inE, (IntPtr)(outE + c0),
                        cols, cnt, outCols));
                }
            }
            if (ManagedQuantizedOps.TryAddmmQuantizedBatch(w.GgmlType, inDim, inDim, CollectionsMarshal.AsSpan(jobs)))
                return;
            for (int e = 0; e < E; e++)
            {
                int cnt = offset[e + 1] - offset[e];
                if (cnt == 0) continue;
                ManagedQuantizedOps.AddmmQuantizedToFloat32(w.GgmlType, (IntPtr)(baseW + e * perExpert),
                    inDim, outCols, input + (long)offset[e] * inDim, inDim, cnt,
                    output + (long)offset[e] * outCols, outCols);
            }
        }

        /// <summary>MoE router logits [N, E] for the pure-C# path: one <see cref="DiffusionGemmaCpuKernels.RouterDot"/>
        /// per (token, expert), parallel over tokens. It has the arithmetic the generic F32 GEMM gives
        /// every row of a full 4-row block, but for ALL rows: the GEMM switches to a differently-grouped
        /// dot for a trailing partial block, so there a token's scores depended on how many rows shared
        /// the call. The prompt-KV decode needs row independence to route each canvas token exactly as
        /// the unified forward does (a flipped top-8 expert moves the output far more than any
        /// rounding). Returns null for a non-F32 router weight (caller uses the linear).</summary>
        private unsafe float[] CpuRouterScores(Tensor normed, string prefix, int N)
        {
            if (!_weights.TryGetValue($"{prefix}.ffn_gate_inp.weight", out Tensor router)
                || router.DimensionCount != 2 || !router.IsContiguous() || router.Sizes[0] != _numExperts)
                return null;
            int E = _numExperts;
            int D = (int)router.Sizes[1];
            if (normed.Sizes[1] != D) return null;
            var scores = new float[(long)N * E];
            nint x = (nint)GetFloatPtr(normed), w = (nint)GetFloatPtr(router);
            const int rowsPerItem = 4;
            CpuFor((N + rowsPerItem - 1) / rowsPerItem, it =>
            {
                int r1 = Math.Min(N, (it + 1) * rowsPerItem);
                fixed (float* sp = scores)
                {
                    for (int r = it * rowsPerItem; r < r1; r++)
                    {
                        float* xr = (float*)x + (long)r * D;
                        for (int e = 0; e < E; e++)
                            sp[(long)r * E + e] = DiffusionGemmaCpuKernels.RouterDot(xr, (float*)w + (long)e * D, D);
                    }
                }
            });
            return scores;
        }

        /// <summary>Token-embedding rows on the host, dequantized in parallel. The self-conditioning
        /// soft-embedding gathers C*K (8192 at a full canvas) rows per step, which the generic
        /// single-threaded Embedding walked one row at a time.</summary>
        private unsafe Tensor CpuEmbeddingRows(int[] tokens)
        {
            if (!CpuFastPaths || CpuLegacyAll || !_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight qw) || !qw.HasHostData
                || !ManagedQuantizedOps.SupportsDequantization((GgmlTensorType)qw.GgmlType) || tokens.Length < 64)
                return Embedding(tokens);
            int dim = (int)qw.Ne0;
            long rowBytes = ManagedQuantizedOps.RowSize(qw.GgmlType, qw.Ne0);
            var result = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
            nint dst = (nint)GetFloatPtr(result);
            nint src = qw.Data;
            int type = qw.GgmlType;
            const int rowsPerItem = 32;
            CpuFor((tokens.Length + rowsPerItem - 1) / rowsPerItem, it =>
            {
                int r1 = Math.Min(tokens.Length, (it + 1) * rowsPerItem);
                for (int r = it * rowsPerItem; r < r1; r++)
                    ManagedQuantizedOps.DequantizeRowToFloat32(type, src + (nint)(tokens[r] * rowBytes),
                        (float*)dst + (long)r * dim, dim);
            });
            return result;
        }
    }
}
