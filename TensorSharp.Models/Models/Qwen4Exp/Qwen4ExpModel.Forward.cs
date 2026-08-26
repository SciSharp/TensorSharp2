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
using System.Diagnostics;
using TensorSharp.Core;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        // KV for the full-attention layers only; the GDN layers carry recurrent state
        // instead. Indexed by absolute layer, null on a recurrent one.
        private Tensor[] _kCache;
        private Tensor[] _vCache;

        // Indexer keys, cached RAW: the pooling that produces a block key happens
        // before the norm and the rotation, so a cached post-rotation key could not be
        // re-pooled at a different block boundary.
        private Tensor[] _idxKCache;

        // GDN state, host side. conv is a [convKernel-1, convDim] ring per layer and
        // delta is [numVHeads, headKDim, headVDim].
        private float[][] _gdnConvState;

        // PLE depthwise conv history: (kernel-1)*ngram rows of hcDim.
        private float[] _pleConvState;

        private int _convDim;

        // Rows currently allocated in the attention caches; grows on demand up to
        // _maxContextLength.
        private int _kvCacheCapacity;

        // Per-phase tick counters. Cheap (one timestamp per phase per layer) and the
        // only way to tell a slow recurrence from a slow MoE without a profiler.
        // Reported by TS_Q4E_PROFILE=1.
        internal long Q4ePleTicks, Q4eHcTicks, Q4eGdnTicks, Q4eAttnTicks, Q4eMoeTicks, Q4eHeadTicks;
        internal long Q4eGdnProjTicks, Q4eGdnPrepTicks, Q4eGdnKernelTicks, Q4eGdnOutTicks;
        internal long Q4eAttnProjTicks, Q4eAttnNormTicks, Q4eAttnRopeTicks, Q4eAttnCoreTicks, Q4eAttnOutTicks;
        private static readonly bool _q4eProfile =
            string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_PROFILE"), "1", StringComparison.Ordinal);

        private void ReportQ4eProfile(int tokens)
        {
            if (!_q4eProfile) return;
            double f = 1000.0 / Stopwatch.Frequency;
            Console.WriteLine(
                $"[q4e-profile] tokens={tokens} ple={Q4ePleTicks * f:F0}ms hc={Q4eHcTicks * f:F0}ms " +
                $"gdn={Q4eGdnTicks * f:F0}ms attn={Q4eAttnTicks * f:F0}ms moe={Q4eMoeTicks * f:F0}ms " +
                $"head={Q4eHeadTicks * f:F0}ms");
            Console.WriteLine(
                $"[q4e-profile]   gdn: proj={Q4eGdnProjTicks * f:F0} prep={Q4eGdnPrepTicks * f:F0} " +
                $"kernel={Q4eGdnKernelTicks * f:F0} out={Q4eGdnOutTicks * f:F0}");
            Console.WriteLine(
                $"[q4e-profile]   attn: proj={Q4eAttnProjTicks * f:F0} norm={Q4eAttnNormTicks * f:F0} " +
                $"rope={Q4eAttnRopeTicks * f:F0} core={Q4eAttnCoreTicks * f:F0} out={Q4eAttnOutTicks * f:F0}");
        }

        private void InitCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();

            int nLayer = Config.NumLayers;
            _kCache = new Tensor[nLayer];
            _vCache = new Tensor[nLayer];
            _idxKCache = new Tensor[nLayer];
            _gdnConvState = new float[nLayer][];

            int keyDim = _headKDim * _numKHeads;
            int valueDim = _headVDim * _numVHeads;
            _convDim = keyDim * 2 + valueDim;

            for (int l = 0; l < nLayer; l++)
            {
                if (_isRecurrent[l])
                {
                    _gdnConvState[l] = new float[(long)(_convKernel - 1) * _convDim];
                    continue;
                }

                _kCache[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                _vCache[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                InitializeCacheTensor(_kCache[l]);
                InitializeCacheTensor(_vCache[l]);

                if (UsesQsa(l))
                {
                    _idxKCache[l] = new Tensor(_allocator, DType.Float32, 1, initialSeqLen, _indexerHeadDim);
                    InitializeCacheTensor(_idxKCache[l]);
                }
            }

            if (_pleHeads > 0)
                _pleConvState = new float[(long)(_pleConvKernel - 1) * _pleNgram * _hcDim];

            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
            {
                throw new InvalidOperationException(
                    $"Sequence length {requiredSeqLen} exceeds the configured context length {_maxContextLength}.");
            }

            int newCapacity = Math.Min(_maxContextLength, Math.Max(requiredSeqLen, _kvCacheCapacity * 2));
            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) continue;
                GrowCache(ref _kCache[l], Config.NumKVHeads, newCapacity, Config.HeadDim, kvDtype);
                GrowCache(ref _vCache[l], Config.NumKVHeads, newCapacity, Config.HeadDim, kvDtype);
                if (_idxKCache[l] != null)
                    GrowCache(ref _idxKCache[l], 1, newCapacity, _indexerHeadDim, DType.Float32);
            }
            _kvCacheCapacity = newCapacity;
        }

        private void GrowCache(ref Tensor cache, int heads, int newLen, int dim, DType dtype)
        {
            var grown = new Tensor(_allocator, dtype, heads, newLen, dim);
            InitializeCacheTensor(grown);
            if (_cacheSeqLen > 0)
            {
                using var src = cache.Narrow(1, 0, _cacheSeqLen);
                using var dst = grown.Narrow(1, 0, _cacheSeqLen);
                Ops.Copy(dst, src);
            }
            cache.Dispose();
            cache = grown;
        }

        protected override void ResetKVCacheCore()
        {
            _cacheSeqLen = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_gdnConvState[l] != null) Array.Clear(_gdnConvState[l]);
                // The delta-net state lives in a device tensor now (the fused kernel
                // updates it in place), so clearing the old host array is not enough.
                if (_gdnStateT != null && _gdnStateT[l] != null)
                {
                    Ops.Fill(_gdnStateT[l], 0f);
                    InvalidateTensorDeviceCache(_gdnStateT[l]);
                }
                if (_gdnConvWriteIdx != null) _gdnConvWriteIdx[l] = 0;
            }
            if (_pleConvState != null) Array.Clear(_pleConvState);
            ResetPleHistory();
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            long tEmb = Stopwatch.GetTimestamp();
            Tensor embd = Embedding(tokens);           // [T, n_embd]
            _embTicks += Stopwatch.GetTimestamp() - tEmb;

            // The wide residual starts as hc identical copies of the embedding.
            Tensor res = BroadcastToStreams(embd, seqLen);
            embd.Dispose();

            for (int il = 0; il < Config.NumLayers; il++)
            {
                long t0 = Stopwatch.GetTimestamp();
                if (_isPle[il])
                    PleLayer(res, tokens, seqLen, startPos, il);
                long t1 = Stopwatch.GetTimestamp(); Q4ePleTicks += t1 - t0;

                Tensor inject;
                Tensor cur = HcMix(res, seqLen,
                    $"blk.{il}.hc_attn_norm.weight", $"blk.{il}.hc_attn_down.weight",
                    $"blk.{il}.hc_attn_up.weight", $"blk.{il}.hc_attn_inject.weight", out inject);
                long t2 = Stopwatch.GetTimestamp(); Q4eHcTicks += t2 - t1;

                Tensor blockOut = _isRecurrent[il]
                    ? GdnLayer(cur, il, seqLen)
                    : AttentionLayer(cur, il, seqLen, startPos);
                cur.Dispose();
                long t3 = Stopwatch.GetTimestamp();
                if (_isRecurrent[il]) Q4eGdnTicks += t3 - t2; else Q4eAttnTicks += t3 - t2;

                HcCombine(res, blockOut, inject, seqLen);
                blockOut.Dispose();
                inject.Dispose();

                cur = HcMix(res, seqLen,
                    $"blk.{il}.hc_ffn_norm.weight", $"blk.{il}.hc_ffn_down.weight",
                    $"blk.{il}.hc_ffn_up.weight", $"blk.{il}.hc_ffn_inject.weight", out inject);
                long t4 = Stopwatch.GetTimestamp(); Q4eHcTicks += t4 - t3;

                Tensor ffnOut = MoeFfn(cur, il, seqLen);
                cur.Dispose();
                long t5 = Stopwatch.GetTimestamp(); Q4eMoeTicks += t5 - t4;

                HcCombine(res, ffnOut, inject, seqLen);
                ffnOut.Dispose();
                inject.Dispose();
                Q4eHcTicks += Stopwatch.GetTimestamp() - t5;
            }

            // The final mixer IS the output norm - qwen4exp ships no separate one.
            Tensor normed = HcMix(res, seqLen,
                "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight",
                null, out _);
            res.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long tHead = Stopwatch.GetTimestamp();
            Tensor logits = LinearForward(lastHidden, "output.weight")
                            ?? LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - tHead;
            Q4eHeadTicks += Stopwatch.GetTimestamp() - tHead;
            lastHidden.Dispose();

            _logitsBuffer = TensorToFloatArray(logits);
            logits.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            ReportQ4eProfile(seqLen);
            return _logitsBuffer;
        }

        /// <summary>[T, n_embd] -> [T, hc*n_embd], every stream a copy.</summary>
        private unsafe Tensor BroadcastToStreams(Tensor x, int seqLen)
        {
            int n = Config.HiddenSize;
            var res = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            float* src = GetFloatPtr(x);
            float* dst = GetFloatPtr(res);
            for (int t = 0; t < seqLen; t++)
            {
                float* s = src + (long)t * n;
                for (int c = 0; c < _hc; c++)
                    Buffer.MemoryCopy(s, dst + (long)t * _hcDim + (long)c * n, n * 4L, n * 4L);
            }
            InvalidateTensorDeviceCache(res);
            return res;
        }

        /// <summary>
        /// Read the wide residual: grouped RMS norm over each stream, a low-rank
        /// sigmoid gate, then collapse the streams by their mean. Also produces the
        /// per-stream scatter weights the matching <see cref="HcCombine"/> needs.
        /// </summary>
        private unsafe Tensor HcMix(Tensor res, int seqLen,
            string normName, string downName, string upName, string injectName, out Tensor inject)
        {
            int n = Config.HiddenSize;

            // xn = groupRmsNorm(res) * w_norm. The converter folded the gammas to
            // (1 + w), so this is a plain multiply rather than an affine.
            var xn = new Tensor(_allocator, DType.Float32, seqLen, _hcDim);
            {
                float* src = GetFloatPtr(res);
                float* dst = GetFloatPtr(xn);
                float* w = GetFloatPtr(_weights[normName]);
                float eps = Config.Eps;
                for (int t = 0; t < seqLen; t++)
                {
                    float* r = src + (long)t * _hcDim;
                    float* o = dst + (long)t * _hcDim;
                    for (int c = 0; c < _hc; c++)
                    {
                        float* rc = r + (long)c * n;
                        float* oc = o + (long)c * n;
                        double ss = 0;
                        for (int i = 0; i < n; i++) ss += (double)rc[i] * rc[i];
                        float inv = (float)(1.0 / Math.Sqrt(ss / n + eps));
                        int b = c * n;
                        for (int i = 0; i < n; i++) oc[i] = rc[i] * inv * w[b + i];
                    }
                }
                InvalidateTensorDeviceCache(xn);
            }

            // gate = sigmoid(W_up @ silu(W_down @ xn / hc))
            //
            // The scale+SiLU on [T, 320] and the sigmoid on [T, 10240] are three more
            // GGML dispatches for a few thousand elements. At 96 mixers per token that
            // was 288 dispatches of pure launch overhead, on a path that is already
            // dispatch-bound, so they run on the host instead.
            Tensor lo = LinearForward(xn, downName);                 // [T, hcLowRank]
            if (seqLen > HostElementwiseMaxRows)
            {
                Ops.Mul(lo, lo, 1.0f / _hc);
                Ops.SiLU(lo, lo);
            }
            else
            {
                float* p = GetFloatPtr(lo);
                long count = (long)seqLen * _hcLowRank;
                float inv = 1.0f / _hc;
                for (long i = 0; i < count; i++)
                {
                    float x = p[i] * inv;
                    p[i] = x / (1.0f + MathF.Exp(-x));               // silu
                }
                InvalidateTensorDeviceCache(lo);
            }
            Tensor gate = LinearForward(lo, upName);                 // [T, hcDim]
            lo.Dispose();
            if (seqLen > HostElementwiseMaxRows)
            {
                Ops.Sigmoid(gate, gate);
            }
            else
            {
                float* p = GetFloatPtr(gate);
                long count = (long)seqLen * _hcDim;
                for (long i = 0; i < count; i++)
                    p[i] = 1.0f / (1.0f + MathF.Exp(-p[i]));
                InvalidateTensorDeviceCache(gate);
            }

            // mixed = mean over the streams of (xn * gate)
            var mixed = new Tensor(_allocator, DType.Float32, seqLen, n);
            {
                float* xp = GetFloatPtr(xn);
                float* gp = GetFloatPtr(gate);
                float* mp = GetFloatPtr(mixed);
                float invHc = 1.0f / _hc;
                for (int t = 0; t < seqLen; t++)
                {
                    float* x = xp + (long)t * _hcDim;
                    float* g = gp + (long)t * _hcDim;
                    float* m = mp + (long)t * n;
                    for (int i = 0; i < n; i++) m[i] = x[i] * g[i];
                    for (int c = 1; c < _hc; c++)
                    {
                        int b = c * n;
                        for (int i = 0; i < n; i++) m[i] += x[b + i] * g[b + i];
                    }
                    for (int i = 0; i < n; i++) m[i] *= invHc;
                }
                InvalidateTensorDeviceCache(mixed);
            }
            gate.Dispose();

            inject = injectName != null ? LinearForward(xn, injectName) : null;  // [T, hc]
            xn.Dispose();
            return mixed;
        }

        /// <summary>
        /// Write a block's output back into every residual stream, each scaled by its
        /// own gate. 2*sigmoid centres the weights on 1, so an untrained injection
        /// reproduces a plain residual add.
        /// </summary>
        private unsafe void HcCombine(Tensor res, Tensor blockOut, Tensor inject, int seqLen)
        {
            int n = Config.HiddenSize;
            float* rp = GetFloatPtr(res);
            float* bp = GetFloatPtr(blockOut);
            float* ip = GetFloatPtr(inject);
            float invHc = 1.0f / _hc;
            for (int t = 0; t < seqLen; t++)
            {
                float* r = rp + (long)t * _hcDim;
                float* b = bp + (long)t * n;
                float* inj = ip + (long)t * _hc;
                for (int c = 0; c < _hc; c++)
                {
                    float w = 2.0f / (1.0f + MathF.Exp(-inj[c] * invHc));
                    float* rc = r + (long)c * n;
                    for (int i = 0; i < n; i++) rc[i] += b[i] * w;
                }
            }
            InvalidateTensorDeviceCache(res);
        }

        /// <summary>
        /// 512-expert MoE: softmax over all experts, top-k, renormalise, plus a shared
        /// expert behind its own scalar sigmoid gate.
        /// </summary>
        private unsafe Tensor MoeFfn(Tensor input, int il, int seqLen)
        {
            int n = Config.HiddenSize;
            if (_fusedGateUpExperts)
            {
                throw new NotSupportedException(
                    "This qwen4exp GGUF stacks gate and up into blk.N.ffn_gate_up_exps; "
                    + "only the separate ffn_gate_exps / ffn_up_exps layout is wired up so far.");
            }
            var gateStack = _stackedExpertWeights[$"blk.{il}.ffn_gate_exps.weight"];
            var upStack = _stackedExpertWeights[$"blk.{il}.ffn_up_exps.weight"];
            var downStack = _stackedExpertWeights[$"blk.{il}.ffn_down_exps.weight"];

            Tensor routerLogits = LinearForward(input, $"blk.{il}.ffn_gate_inp.weight"); // [T, nExperts]

            // Host top-k routing for every token at once: softmax over all 512 experts,
            // keep the best n_expert_used, renormalise (llama.cpp's norm_w).
            int K = _numExpertsUsed;
            if (_moeSelExperts == null || _moeSelExperts.Length < (long)seqLen * K)
            {
                _moeSelExperts = new int[(long)seqLen * K];
                _moeRouteWts = new float[(long)seqLen * K];
            }
            var probs = _moeProbs ??= new float[_numExperts];

            float* rl = GetFloatPtr(routerLogits);
            for (int t = 0; t < seqLen; t++)
            {
                float* row = rl + (long)t * _numExperts;
                float max = float.NegativeInfinity;
                for (int e = 0; e < _numExperts; e++) if (row[e] > max) max = row[e];
                float sum = 0f;
                for (int e = 0; e < _numExperts; e++) { probs[e] = MathF.Exp(row[e] - max); sum += probs[e]; }
                float invSum = 1.0f / sum;
                for (int e = 0; e < _numExperts; e++) probs[e] *= invSum;

                float wSum = 0f;
                for (int k = 0; k < K; k++)
                {
                    int best = -1; float bestV = float.NegativeInfinity;
                    for (int e = 0; e < _numExperts; e++)
                        if (probs[e] > bestV) { bestV = probs[e]; best = e; }
                    probs[best] = float.NegativeInfinity;
                    _moeSelExperts[t * K + k] = best;
                    _moeRouteWts[t * K + k] = bestV;
                    wSum += bestV;
                }
                float invW = 1.0f / wSum;
                for (int k = 0; k < K; k++) _moeRouteWts[t * K + k] *= invW;
            }
            routerLogits.Dispose();

            var result = new Tensor(_allocator, DType.Float32, seqLen, n);
            int nFf = (int)gateStack.PerExpertNe1;

            if (IsGgmlBackend)
            {
                // ONE ggml_mul_mat_id over the whole batch. The per-token call this
                // replaced issued seqLen * n_layers native graph builds - 49k of them
                // for a 1024-token prefill - and was the dominant prefill cost once the
                // recurrence moved to the GPU.
                GgmlBasicOps.MoEFFNPrefill(
                    input, result, seqLen, n, nFf,
                    gateStack.NumExperts, K, _moeSelExperts, _moeRouteWts,
                    gateStack.Data, gateStack.GgmlType, gateStack.PerExpertNe0, gateStack.PerExpertNe1, gateStack.TotalRawBytes,
                    upStack.Data, upStack.GgmlType, upStack.PerExpertNe0, upStack.PerExpertNe1, upStack.TotalRawBytes,
                    downStack.Data, downStack.GgmlType, downStack.PerExpertNe0, downStack.PerExpertNe1, downStack.TotalRawBytes,
                    gateBias: null, upBias: null, downBias: null,
                    activation: GgmlBasicOps.MoEActivation.SwiGLUSplit);
            }
            else
            {
                // Portable path for the direct-CUDA and pure-C# backends, which have no
                // stacked-expert kernel: one expert at a time through the ordinary
                // quantized linear, accumulated by route weight.
                var ids = new int[K];
                var wts = new float[K];
                for (int t = 0; t < seqLen; t++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        ids[k] = _moeSelExperts[t * K + k];
                        wts[k] = _moeRouteWts[t * K + k];
                    }
                    using var tokenIn = input.Narrow(0, t, 1);
                    using var tokenOut = result.Narrow(0, t, 1);
                    MoeExpertsPortable(tokenOut, tokenIn, il, ids, wts);
                }
            }
            InvalidateTensorDeviceCache(result);

            // Shared expert, batched over all tokens, behind its own sigmoid gate.
            Tensor sg = LinearForward(input, $"blk.{il}.ffn_gate_shexp.weight");
            Tensor su = LinearForward(input, $"blk.{il}.ffn_up_shexp.weight");
            if (seqLen > HostElementwiseMaxRows)
            {
                Ops.SiLUMul(sg, sg, su);
            }
            else
            {
                // silu(gate) * up on [T, 640] - host, for the same reason as the mixer.
                float* gp = GetFloatPtr(sg);
                float* up = GetFloatPtr(su);
                long count = (long)seqLen * sg.Sizes[1];
                for (long i = 0; i < count; i++)
                    gp[i] = (gp[i] / (1.0f + MathF.Exp(-gp[i]))) * up[i];
                InvalidateTensorDeviceCache(sg);
            }
            su.Dispose();
            Tensor sd = LinearForward(sg, $"blk.{il}.ffn_down_shexp.weight");
            sg.Dispose();

            // The gate is ONE scalar per token and its weight is a bare [n_embd] vector
            // rather than a matrix - LinearForward would read that as a 2560-wide
            // projection - so the dot product is explicit.
            Tensor gateVec = _weights[$"blk.{il}.ffn_gate_inp_shexp.weight"];
            {
                float* gv = GetFloatPtr(gateVec);
                float* xp = GetFloatPtr(input);
                float* sp = GetFloatPtr(sd);
                float* rp = GetFloatPtr(result);
                for (int t = 0; t < seqLen; t++)
                {
                    float* x = xp + (long)t * n;
                    double dot = 0;
                    for (int i = 0; i < n; i++) dot += (double)x[i] * gv[i];
                    float g = 1.0f / (1.0f + MathF.Exp(-(float)dot));
                    float* sv = sp + (long)t * n;
                    float* r = rp + (long)t * n;
                    for (int i = 0; i < n; i++) r[i] += sv[i] * g;
                }
                InvalidateTensorDeviceCache(result);
            }
            sd.Dispose();
            return result;
        }

        /// <summary>
        /// Above this many rows the small elementwise steps go back to GGML.
        ///
        /// A decode step is dispatch-bound - ~850 GGML launches per token, each costing
        /// far more than the arithmetic - so running a 320- or 10240-element sigmoid on
        /// the host removes three launches per mixer, 288 per token. A 1024-row prefill
        /// is the opposite: the same code becomes 10.5 M host exp() calls per mixer and
        /// cost 2.7 s of prefill when it was applied unconditionally.
        /// </summary>
        private const int HostElementwiseMaxRows = 8;

        private int[] _moeSelExperts;
        private float[] _moeRouteWts;
        private float[] _moeProbs;

        // Per-expert non-owning views over the stacked tensors, built on first use.
        // Only the backends without a stacked-expert kernel need them.
        private QuantizedWeight[][] _expertGateView, _expertUpView, _expertDownView;
        private Tensor _moeScratchGate, _moeScratchUp, _moeScratchDown;

        private void MoeExpertsPortable(Tensor outRow, Tensor inRow, int il, int[] expertIds, float[] routeW)
        {
            EnsureExpertViews(il);
            int n = Config.HiddenSize;
            int ff = (int)_expertGateView[il][expertIds[0]].Ne1;

            _moeScratchGate ??= new Tensor(_allocator, DType.Float32, 1, ff);
            _moeScratchUp ??= new Tensor(_allocator, DType.Float32, 1, ff);
            _moeScratchDown ??= new Tensor(_allocator, DType.Float32, 1, n);

            Ops.Fill(outRow, 0f);
            for (int k = 0; k < expertIds.Length; k++)
            {
                int e = expertIds[k];
                AddmmQuantManaged(_moeScratchGate, inRow, _expertGateView[il][e]);
                AddmmQuantManaged(_moeScratchUp, inRow, _expertUpView[il][e]);
                Ops.SiLUMul(_moeScratchGate, _moeScratchGate, _moeScratchUp);
                AddmmQuantManaged(_moeScratchDown, _moeScratchGate, _expertDownView[il][e]);
                Ops.AddMulV(outRow, outRow, _moeScratchDown, routeW[k]);
            }
        }

        private void EnsureExpertViews(int il)
        {
            _expertGateView ??= new QuantizedWeight[Config.NumLayers][];
            _expertUpView ??= new QuantizedWeight[Config.NumLayers][];
            _expertDownView ??= new QuantizedWeight[Config.NumLayers][];
            if (_expertGateView[il] != null) return;

            var g = _stackedExpertWeights[$"blk.{il}.ffn_gate_exps.weight"];
            var u = _stackedExpertWeights[$"blk.{il}.ffn_up_exps.weight"];
            var d = _stackedExpertWeights[$"blk.{il}.ffn_down_exps.weight"];
            _expertGateView[il] = new QuantizedWeight[_numExperts];
            _expertUpView[il] = new QuantizedWeight[_numExperts];
            _expertDownView[il] = new QuantizedWeight[_numExperts];
            for (int e = 0; e < _numExperts; e++)
            {
                _expertGateView[il][e] = QuantizedWeight.CreateExpertView(g, e);
                _expertUpView[il][e] = QuantizedWeight.CreateExpertView(u, e);
                _expertDownView[il][e] = QuantizedWeight.CreateExpertView(d, e);
            }
        }

        public override void Dispose()
        {
            if (_kCache != null)
                foreach (var t in _kCache) t?.Dispose();
            if (_vCache != null)
                foreach (var t in _vCache) t?.Dispose();
            if (_idxKCache != null)
                foreach (var t in _idxKCache) t?.Dispose();
            base.Dispose();
        }
    }
}
