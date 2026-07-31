// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DeepSeek V4 CPU executor: attention, block compressors (CSA/HCA/LID),
// lightning indexer and RoPE. See DeepSeek4CpuExecutor.cs for the overview.
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    internal sealed unsafe partial class DeepSeek4CpuExecutor
    {
        // -------------------------------------------------------------------
        // RoPE (port of ggml's rope_yarn / ggml_rope_cache_init, NORMAL mode:
        // interleaved pairs (x[i0], x[i0+1]) with cache (cos, sin) at [i0]).
        // -------------------------------------------------------------------

        private void RopeCacheInit(long pos, bool comp, float* cache)
        {
            int nDims = _nRot;
            float freqBase = comp ? _compressRopeBase : _ropeFreqBase;
            float freqScale = comp ? _yarnFreqScale : 1f;
            float extFactor = comp ? _yarnExtFactor : 0f;
            // dsv4_rope_attn_factor: mscale is cancelled so the effective factor is
            // 1/(1 + 0.1 ln(1/freq_scale)); rope_yarn multiplies the inverse back.
            float attnFactor = extFactor == 0f ? 1f : 1f / (1f + 0.1f * MathF.Log(1f / freqScale));
            float thetaScale = MathF.Pow(freqBase, -2f / nDims);

            float theta = pos;
            for (int i0 = 0; i0 < nDims; i0 += 2)
            {
                float thetaInterp = freqScale * theta;
                float th = thetaInterp;
                float mscale = attnFactor;
                if (extFactor != 0f)
                {
                    float y = (i0 / 2 - _compCorr0) / MathF.Max(0.001f, _compCorr1 - _compCorr0);
                    float ramp = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                    th = thetaInterp * (1f - ramp) + theta * ramp;
                    mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);
                }
                cache[i0 + 0] = MathF.Cos(th) * mscale;
                cache[i0 + 1] = MathF.Sin(th) * mscale;
                theta *= thetaScale;
            }
        }

        /// <summary>Rotate interleaved pairs in place. sinSign=-1 gives the inverse rotation (rope_ext_back).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void RotatePairs(float* x, float* cache, int nDims, float sinSign)
        {
            for (int i0 = 0; i0 < nDims; i0 += 2)
            {
                float c = cache[i0 + 0];
                float s = cache[i0 + 1] * sinSign;
                float x0 = x[i0 + 0];
                float x1 = x[i0 + 1];
                x[i0 + 0] = x0 * c - x1 * s;
                x[i0 + 1] = x0 * s + x1 * c;
            }
        }

        // -------------------------------------------------------------------
        // Attention super-block
        // -------------------------------------------------------------------

        private void Attention(int il, int nt, int p0)
        {
            Layer L = _layers[il];
            int E = _nEmbd, NH = _nHead, HD = _headDim, ROT = _nRot;
            int NOPE = HD - ROT;
            bool comp = L.Ratio != 0;
            float* ropeCache = comp ? _ropeCompCache : _ropeRawCache;

            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            // q = wq_b(rms(wq_a(cur))) -> per-head RMS (no weight) -> RoPE on rope slice
            bool dbg = StageDebug && il == 0;
            MatMul(L.WqA, 0, _qLoraRank, _cur, E, nt, _qr, _qLoraRank);
            RmsNormRows(_qr, L.QANorm, nt, _qLoraRank);
            if (dbg)
                DumpDbg("L0.qr", _qr);
            MatMul(L.WqB, 0, NH * HD, _qr, _qLoraRank, nt, _q, NH * HD);
            RmsNormRows(_q, null, nt * NH, HD);

            PFor(nt, t =>
            {
                float* cache = ropeCache + (long)t * ROT;
                float* qt = _q + (long)t * NH * HD;
                for (int h = 0; h < NH; h++)
                    RotatePairs(qt + (long)h * HD + NOPE, cache, ROT, 1f);
            });
            if (dbg)
                DumpDbg("L0.q_prep", _q);

            // kv: single shared 512-dim K(=V) head; commit to the SWA ring (F16-rounded)
            MatMul(L.Wkv, 0, HD, _cur, E, nt, _kvRow, HD);
            if (dbg)
                DumpDbg("L0.kv_raw_preNorm", _kvRow);
            RmsNormRows(_kvRow, L.KvNorm, nt, HD);
            PFor(nt, t =>
            {
                float* kv = _kvRow + (long)t * HD;
                RotatePairs(kv + NOPE, ropeCache + (long)t * ROT, ROT, 1f);
                long p = p0 + t;
                float* dst = L.RawK + (p % _ringRaw) * HD;
                for (int d = 0; d < HD; d++)
                    dst[d] = F16(kv[d]);
            });
            if (dbg && p0 == 0)
                DumpDbg("L0.ring0", L.RawK);

            Tick(2, t0);
            t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            // compressors + indexer
            if (L.Ratio == CsaRatio)
            {
                int cw = 2 * HD;
                MatMul(L.CompWkv, 0, cw, _cur, E, nt, _stKv, cw);
                MatMul(L.CompWgate, 0, cw, _cur, E, nt, _stScore, cw);
                AddApe(_stScore, L.CompApe, nt, p0, CsaRatio, cw);
                RunCompressor(nt, p0, CsaRatio, 2, HD, 2 * CsaRatio, cw,
                    _stKv, _stScore, L.HistKv, L.HistScore, L.CompNorm, L.CompK);

                int lcw = 2 * _idxHeadSize;
                MatMul(L.IdxCompWkv, 0, lcw, _cur, E, nt, _lidStKv, lcw);
                MatMul(L.IdxCompWgate, 0, lcw, _cur, E, nt, _lidStScore, lcw);
                AddApe(_lidStScore, L.IdxCompApe, nt, p0, CsaRatio, lcw);
                RunCompressor(nt, p0, CsaRatio, 2, _idxHeadSize, 2 * CsaRatio, lcw,
                    _lidStKv, _lidStScore, L.LidHistKv, L.LidHistScore, L.IdxCompNorm, L.LidK);
                Tick(3, t0);

                t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                BuildIndexerTopK(il, nt, p0);
                Tick(4, t0);
            }
            else if (L.Ratio == HcaRatio)
            {
                MatMul(L.CompWkv, 0, HD, _cur, E, nt, _stKv, HD);
                MatMul(L.CompWgate, 0, HD, _cur, E, nt, _stScore, HD);
                AddApe(_stScore, L.CompApe, nt, p0, HcaRatio, HD);
                RunCompressor(nt, p0, HcaRatio, 1, HD, HcaRatio, HD,
                    _stKv, _stScore, L.HistKv, L.HistScore, L.CompNorm, L.CompK);
                Tick(3, t0);
            }
            else
            {
                Tick(3, t0);
            }

            t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            AttentionCore(L, nt, p0);
            if (dbg)
                DumpDbg("L0.attn_core", _attnO);
            Tick(5, t0);
            t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            // inverse RoPE on the rope slice of the attention output
            PFor(nt, t =>
            {
                float* cache = ropeCache + (long)t * ROT;
                float* ot = _attnO + (long)t * NH * HD;
                for (int h = 0; h < NH; h++)
                    RotatePairs(ot + (long)h * HD + NOPE, cache, ROT, -1f);
            });

            // grouped LoRA output projection
            int oGroupDim = NH / _oGroups * HD;
            int oCat = _oGroups * _oLoraRank;
            if (!TryGroupedOutProjFused(L, nt, oGroupDim, oCat))
            {
                for (int g = 0; g < _oGroups; g++)
                {
                    MatMul(L.WoA, (long)g * _oLoraRank, _oLoraRank,
                        _attnO + (long)g * oGroupDim, NH * HD, nt,
                        _oG + (long)g * _oLoraRank, oCat);
                }
            }
            MatMul(L.WoB, 0, E, _oG, oCat, nt, _attnOut, E);
            Tick(6, t0);
        }

        /// <summary>
        /// All oGroups per-group LoRA-A matmuls in one quantize region + one dot
        /// region (weight row r belongs to group r / oLoraRank and dots the
        /// matching per-group slice of the attention output).
        /// </summary>
        private bool TryGroupedOutProjFused(Layer L, int nt, int oGroupDim, int oCat)
        {
            WeightRef w = L.WoA;
            if (!ManagedQuantizedOps.TryGetActivationPlan(w.Type, w.Ne0, out int actBytes) || actBytes > _maxActRowBytes)
                return false;

            int NHHD = _nHead * _headDim;
            int G = _oGroups;
            byte* act = _actQuant;
            var wRef = w;
            PFor(nt * G, i =>
            {
                int t = i / G, g = i % G;
                ManagedQuantizedOps.QuantizeActivationRow(wRef.Type,
                    _attnO + (long)t * NHHD + (long)g * oGroupDim,
                    act + (long)i * actBytes, wRef.Ne0);
            });

            int totalRows = w.Ne1;
            const int RowsPerTask = 128;
            int nBlocks = (totalRows + RowsPerTask - 1) / RowsPerTask;
            PFor(nBlocks, bi =>
            {
                int r0 = bi * RowsPerTask, r1 = Math.Min(r0 + RowsPerTask, totalRows);
                for (int r = r0; r < r1; r++)
                {
                    byte* wr = wRef.Row(r);
                    int g = r / _oLoraRank;
                    for (int t = 0; t < nt; t++)
                        _oG[(long)t * oCat + r] = ManagedQuantizedOps.DotQuantizedRow(wRef.Type, wr,
                            act + (long)(t * G + g) * actBytes, wRef.Ne0);
                }
            });
            return true;
        }

        private void AddApe(float* stScore, float[] ape, int nt, int p0, int ratio, int cw)
        {
            PFor(nt, t =>
            {
                int r = (int)((long)(p0 + t) % ratio);
                float* row = stScore + (long)t * cw;
                var dst = new Span<float>(row, cw);
                TensorPrimitives.Add(new ReadOnlySpan<float>(ape, r * cw, cw), dst, dst);
            });
        }

        // -------------------------------------------------------------------
        // Block compressor (CSA overlap coff=2 / HCA coff=1 / LID)
        //
        // For every block boundary p in the ubatch ((p+1) % ratio == 0) the
        // window of W = coff*ratio token projections is reduced with a
        // per-dimension softmax over the window, RMS-normed, RoPE'd at the
        // block-start position and committed (F16-rounded) into `cache` at row
        // p/ratio. Overlap compressors read half 0 of a token's projection when
        // the token plays the "previous block" role and half 1 for the
        // "current block" role. Window positions before the first token use
        // kv=0 / score=-inf (the ggml port bakes these into an extra ring row).
        //
        // Afterwards the last min(nt, stateSize) token projections are
        // persisted into the state ring for the next ubatch.
        // -------------------------------------------------------------------

        private void RunCompressor(
            int nt, int p0, int ratio, int coff, int head, int stateSize, int cw,
            float* stKv, float* stScore, float* histKv, float* histScore,
            float[] normW, float* cache)
        {
            int W = coff * ratio;
            long firstBoundary = -1;
            for (long p = p0; p < p0 + nt; p++)
            {
                if ((p + 1) % ratio == 0) { firstBoundary = p; break; }
            }

            if (firstBoundary >= 0)
            {
                int nBlocks = (int)((p0 + nt - 1 - firstBoundary) / ratio) + 1;
                PFor(nBlocks, bi =>
                {
                    long p = firstBoundary + (long)bi * ratio;
                    long blockRow = p / ratio;
                    long start = p + 1 - ratio;

                    float* row = stackalloc float[head];
                    for (int d = 0; d < head; d++)
                    {
                        float m = float.NegativeInfinity;
                        for (int w = 0; w < W; w++)
                        {
                            long tw = p + 1 - W + w;
                            if (tw < 0) continue;
                            int half = (coff == 2 && w >= ratio) ? head : 0;
                            float sc = tw >= p0
                                ? stScore[(tw - p0) * cw + half + d]
                                : histScore[(tw % stateSize) * cw + half + d];
                            if (sc > m) m = sc;
                        }
                        float se = 0f, sv = 0f;
                        if (!float.IsNegativeInfinity(m))
                        {
                            for (int w = 0; w < W; w++)
                            {
                                long tw = p + 1 - W + w;
                                if (tw < 0) continue;
                                int half = (coff == 2 && w >= ratio) ? head : 0;
                                long off = tw >= p0 ? (tw - p0) * cw + half + d : (tw % stateSize) * cw + half + d;
                                float sc = tw >= p0 ? stScore[off] : histScore[off];
                                if (float.IsNegativeInfinity(sc)) continue;
                                float e = MathF.Exp(sc - m);
                                se += e;
                                sv += e * (tw >= p0 ? stKv[off] : histKv[off]);
                            }
                        }
                        row[d] = se > 0f ? sv / se : 0f;
                    }

                    // RMS norm + weight
                    double ss = 0;
                    for (int d = 0; d < head; d++) ss += (double)row[d] * row[d];
                    float inv = 1.0f / MathF.Sqrt((float)(ss / head) + _rmsEps);
                    for (int d = 0; d < head; d++) row[d] *= inv * normW[d];

                    // RoPE the rope slice at the block start position (compress params)
                    float* rc = stackalloc float[_nRot];
                    RopeCacheInit(start, comp: true, rc);
                    RotatePairs(row + (head - _nRot), rc, _nRot, 1f);

                    float* dst = cache + blockRow * head;
                    for (int d = 0; d < head; d++)
                        dst[d] = F16(row[d]);
                });
            }

            // persist the tail of the ubatch into the state ring
            int persistStart = Math.Max(0, nt - stateSize);
            for (int t = persistStart; t < nt; t++)
            {
                long slot = (long)(p0 + t) % stateSize;
                Buffer.MemoryCopy(stKv + (long)t * cw, histKv + slot * cw, cw * sizeof(float), cw * sizeof(float));
                Buffer.MemoryCopy(stScore + (long)t * cw, histScore + slot * cw, cw * sizeof(float), cw * sizeof(float));
            }
        }

        // -------------------------------------------------------------------
        // Lightning indexer: per token, score every visible LID row with
        // sum_h relu(q_h . k) * w_h and keep the top-k row indices.
        // -------------------------------------------------------------------

        private void BuildIndexerTopK(int il, int nt, int p0)
        {
            Layer L = _layers[il];
            int E = _nEmbd;
            int IH = _idxNHead, ID = _idxHeadSize, ROT = _nRot;
            int idxNope = ID - ROT;

            MatMul(L.IdxQB, 0, IH * ID, _qr, _qLoraRank, nt, _iq, IH * ID);
            MatMul(L.IdxProj, 0, IH, _cur, E, nt, _iw, IH);

            float iwScale = 1.0f / MathF.Sqrt((float)ID * IH);
            PFor(nt, t =>
            {
                float* cache = _ropeCompCache + (long)t * ROT;
                float* iqt = _iq + (long)t * IH * ID;
                for (int h = 0; h < IH; h++)
                    RotatePairs(iqt + (long)h * ID + idxNope, cache, ROT, 1f);
                float* iwt = _iw + (long)t * IH;
                for (int h = 0; h < IH; h++)
                    iwt[h] *= iwScale;
            });

            // scores, parallel over (token, row-chunk)
            const int RowChunk = 1024;
            int maxVis = (int)(((long)p0 + nt) / CsaRatio);
            int nChunks = Math.Max(1, (maxVis + RowChunk - 1) / RowChunk);
            PFor(nt * nChunks, work =>
            {
                int t = work / nChunks;
                int ci = work % nChunks;
                long p = p0 + t;
                int nVis = (int)((p + 1) / CsaRatio);
                int r0 = ci * RowChunk;
                if (r0 >= nVis) return;
                int r1 = Math.Min(r0 + RowChunk, nVis);

                float* iqt = _iq + (long)t * IH * ID;
                float* iwt = _iw + (long)t * IH;
                float* scores = _idxScores + (long)t * _compRowsCsa;
                for (int r = r0; r < r1; r++)
                {
                    float* k = L.LidK + (long)r * ID;
                    var kSpan = new ReadOnlySpan<float>(k, ID);
                    float score = 0f;
                    for (int h = 0; h < IH; h++)
                    {
                        float qk = TensorPrimitives.Dot(new ReadOnlySpan<float>(iqt + (long)h * ID, ID), kSpan);
                        if (qk > 0f)
                            score += qk * iwt[h];
                    }
                    scores[r] = score;
                }
            });

            // top-k per token (min-heap over the visible rows)
            PFor(nt, t =>
            {
                long p = p0 + t;
                int nVis = (int)((p + 1) / CsaRatio);
                int k = Math.Min(_idxTopK, nVis);
                int* outIdx = _topK + (long)t * _idxTopK;
                _topKCount[t] = k;
                if (k <= 0) return;

                float* scores = _idxScores + (long)t * _compRowsCsa;
                if (k == nVis)
                {
                    for (int r = 0; r < k; r++) outIdx[r] = r;
                    return;
                }

                // min-heap of size k on (score, idx)
                float* heapVal = stackalloc float[k];
                for (int r = 0; r < k; r++) { heapVal[r] = scores[r]; outIdx[r] = r; }
                // heapify
                for (int i = k / 2 - 1; i >= 0; i--) HeapSiftDown(heapVal, outIdx, i, k);
                for (int r = k; r < nVis; r++)
                {
                    if (scores[r] <= heapVal[0]) continue;
                    heapVal[0] = scores[r];
                    outIdx[0] = r;
                    HeapSiftDown(heapVal, outIdx, 0, k);
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void HeapSiftDown(float* val, int* idx, int i, int n)
        {
            while (true)
            {
                int l = 2 * i + 1, r = l + 1, s = i;
                if (l < n && val[l] < val[s]) s = l;
                if (r < n && val[r] < val[s]) s = r;
                if (s == i) return;
                (val[i], val[s]) = (val[s], val[i]);
                (idx[i], idx[s]) = (idx[s], idx[i]);
                i = s;
            }
        }

        // -------------------------------------------------------------------
        // Attention core: per (token, head) online softmax with sinks over the
        // raw SWA rows plus (for CSA) the indexer-selected compressed rows or
        // (for HCA) all visible compressed rows. K doubles as V.
        // -------------------------------------------------------------------

        private void AttentionCore(Layer L, int nt, int p0)
        {
            int NH = _nHead, HD = _headDim;
            float kqScale = 1.0f / MathF.Sqrt(HD);
            float[] sinks = L.Sinks;
            int ratio = L.Ratio;
            int ring = _ringRaw;
            int nSwa = _nSwa;
            int maxRows = nSwa + Math.Max(_idxTopK, _nCtx / HcaRatio + 1) + 8;

            PFor(nt * NH, work =>
            {
                int t = work / NH;
                int h = work % NH;
                long p = p0 + t;

                float* q = _q + (long)t * NH * HD + (long)h * HD;
                var qSpan = new ReadOnlySpan<float>(q, HD);

                float** rowPtr = stackalloc float*[maxRows];
                float* score = stackalloc float[maxRows];
                int n = 0;

                long trStart = Math.Max(0, p - nSwa + 1);
                for (long tr = trStart; tr <= p; tr++)
                    rowPtr[n++] = L.RawK + (tr % ring) * HD;

                if (ratio == CsaRatio)
                {
                    int cnt = _topKCount[t];
                    int* sel = _topK + (long)t * _idxTopK;
                    for (int i = 0; i < cnt; i++)
                        rowPtr[n++] = L.CompK + (long)sel[i] * HD;
                }
                else if (ratio == HcaRatio)
                {
                    int nVis = (int)((p + 1) / HcaRatio);
                    for (int r = 0; r < nVis; r++)
                        rowPtr[n++] = L.CompK + (long)r * HD;
                }

                float m = sinks[h];
                for (int i = 0; i < n; i++)
                {
                    float s = TensorPrimitives.Dot(qSpan, new ReadOnlySpan<float>(rowPtr[i], HD)) * kqScale;
                    score[i] = s;
                    if (s > m) m = s;
                }

                float sum = MathF.Exp(sinks[h] - m);
                float* acc = stackalloc float[HD];
                new Span<float>(acc, HD).Clear();
                var accSpan = new Span<float>(acc, HD);
                for (int i = 0; i < n; i++)
                {
                    float w = MathF.Exp(score[i] - m);
                    sum += w;
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(rowPtr[i], HD), w, accSpan, accSpan);
                }

                float invSum = 1.0f / sum;
                float* dst = _attnO + (long)t * NH * HD + (long)h * HD;
                for (int d = 0; d < HD; d++)
                    dst[d] = acc[d] * invSum;
            });
        }
    }
}
