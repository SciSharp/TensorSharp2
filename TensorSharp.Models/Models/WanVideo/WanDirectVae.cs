// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan causal 3D video VAE on the direct Cuda/Cpu backends. Same chunked
// algorithm as the GGML whole-graph kernels (decode iterates latent frames,
// encode 1+4k pixel-frame chunks, causal conv caches carried between chunks —
// see ggml_ops_wan.cpp), but with CHANNELS-LAST [T, H, W, C] activations:
//   - every conv is banded im2col ([rows*OW, C*KH*KW]) x flattened-kernel GEMM
//     writing straight into contiguous output row bands,
//   - the channel RMS norm is a plain rows-RMSNorm over [T*H*W, C],
//   - bias adds are rows-bias adds, and nearest-x2 upsampling is two concats,
// so the whole network runs on the existing GEMM/norm/elementwise device ops
// plus the single ts_wan_im2col_f32 kernel (managed loop on CPU).
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanDirectVae : WanVaeBase
    {
        private readonly DirectContext _ctx;
        private readonly WanVaeWeights _w;

        private sealed class ConvT
        {
            public int Kd, K, Ic, Oc;
            public Tensor[] Taps;      // [Oc, Ic*K*K] device tensors
            public Tensor[] TapsT;     // transposed views [Ic*K*K, Oc] (GEMM operand)
            public Tensor Bias;        // [Oc] or null
        }
        private sealed class NormT { public Tensor Gamma; public int C; }
        private sealed class ResT { public NormT N0, N3; public ConvT C2, C6, Shortcut; }
        private sealed class AttnT { public NormT Norm; public DirectLinear Qkv, Proj; public int C; }
        private sealed class UpT { public ConvT TimeConv, SConv; }
        private sealed class DownT { public ConvT SConv, TConv; }

        private bool _decLoaded, _encLoaded;
        private ConvT _conv2, _conv1, _headConv;
        private ResT _mid0, _mid2;
        private AttnT _mid1;
        private ResT[] _res;
        private UpT[] _up;
        private NormT _headNorm;

        private ConvT _encStem, _encHeadConv, _encQuant;
        private ResT[] _encRes;
        private DownT[] _encDown;
        private ResT _encMid0, _encMid2;
        private AttnT _encMid1;
        private NormT _encHeadNorm;

        // Cross-chunk causal caches, indexed by traversal order (same cursor
        // discipline as the ggml WanVaeBuild).
        private readonly Tensor[] _cache = new Tensor[64];
        private int _cursor;
        private int _chunk;

        private const float NormEps = 1e-12f;

        // Direct-path activations and cross-chunk caches are full F32 (the GGML
        // graphs cast caches to F16 and let gallocr alias intermediates), so tile
        // above ~0.3 MP to bound the working set.
        protected override long TilePixelThreshold => 300_000;

        public WanDirectVae(DirectContext ctx, string vaePath)
        {
            _ctx = ctx;
            _w = WanVaeWeights.Load(vaePath);
            InitFromWeights(_w);
        }

        // ---- weight upload ---------------------------------------------------------

        private ConvT Conv(WanVaeConvWeights c)
        {
            if (c == null) return null;
            var t = new ConvT { Kd = c.Kd, K = c.K, Ic = c.Ic, Oc = c.Oc };
            t.Taps = new Tensor[c.Kd];
            t.TapsT = new Tensor[c.Kd];
            long k = (long)c.Ic * c.K * c.K;
            for (int j = 0; j < c.Kd; j++)
            {
                t.Taps[j] = _ctx.Own(_ctx.FromFloats(c.Taps[j], c.Oc, k));
                t.TapsT[j] = t.Taps[j].Transpose();
            }
            if (c.Bias != null) t.Bias = _ctx.Own(_ctx.FromFloats(c.Bias, c.Oc));
            return t;
        }
        private NormT Norm(WanVaeNormWeights n) => new NormT { Gamma = _ctx.Own(_ctx.FromFloats(n.Gamma, n.C)), C = n.C };
        private ResT Res(WanVaeResBlockWeights r) => new ResT
        { N0 = Norm(r.N0), C2 = Conv(r.C2), N3 = Norm(r.N3), C6 = Conv(r.C6), Shortcut = Conv(r.Shortcut) };
        private AttnT Attn(WanVaeAttnWeights a) => new AttnT
        {
            Norm = Norm(a.Norm),
            Qkv = DirectLinear.FromFloats(_ctx, a.QkvW, a.C, 3L * a.C, a.QkvB),
            Proj = DirectLinear.FromFloats(_ctx, a.ProjW, a.C, a.C, a.ProjB),
            C = a.C,
        };

        private void EnsureDecoder()
        {
            if (_decLoaded) return;
            _conv2 = Conv(_w.Conv2);
            _conv1 = Conv(_w.Conv1);
            _mid0 = Res(_w.Mid0);
            _mid1 = Attn(_w.Mid1);
            _mid2 = Res(_w.Mid2);
            _res = new ResT[12];
            for (int i = 0; i < 12; i++) _res[i] = Res(_w.Res[i]);
            _up = new UpT[3];
            for (int i = 0; i < 3; i++)
                _up[i] = new UpT { SConv = Conv(_w.Up[i].SConv), TimeConv = Conv(_w.Up[i].TimeConv) };
            _headNorm = Norm(_w.HeadNorm);
            _headConv = Conv(_w.HeadConv);
            _decLoaded = true;
        }

        private void EnsureEncoder()
        {
            if (_encLoaded) return;
            _encStem = Conv(_w.EncStem);
            _encRes = new ResT[8];
            for (int i = 0; i < 8; i++) _encRes[i] = Res(_w.EncRes[i]);
            _encDown = new DownT[3];
            for (int i = 0; i < 3; i++)
                _encDown[i] = new DownT { SConv = Conv(_w.EncDown[i].SConv), TConv = Conv(_w.EncDown[i].TConv) };
            _encMid0 = Res(_w.EncMid0);
            _encMid1 = Attn(_w.EncMid1);
            _encMid2 = Res(_w.EncMid2);
            _encHeadNorm = Norm(_w.EncHeadNorm);
            _encHeadConv = Conv(_w.EncHeadConv);
            _encQuant = Conv(_w.EncQuant);
            _encLoaded = true;
        }

        // ---- primitives ------------------------------------------------------------

        private static long GemmBudgetBytes()
        {
            string e = Environment.GetEnvironmentVariable("TS_WAN_VAE_GEMM_MAX_MB");
            long mb = 256;
            if (e != null && long.TryParse(e, out long v) && v > 0) mb = v;
            return mb << 20;
        }

        /// <summary>2D conv of every frame: x [T, H, W, Ic] -> [T, OH, OW, Oc].
        /// tapFrame(t) supplies the source frame for temporal tap j of output frame t.
        /// Bands the materialized im2col to the TS_WAN_VAE_GEMM_MAX_MB budget.
        /// padRB adds right/bottom-only zero padding (the encoder Resample: the
        /// im2col bounds check supplies the zeros, but the output size must count
        /// the padded extent).</summary>
        private Tensor Conv2dFrames(ConvT c, long T, long h, long w, Func<long, int, Tensor> tapFrame,
                                    int pad, int stride, int padRB = 0)
        {
            long oh = (h + 2 * pad + padRB - c.K) / stride + 1;
            long ow = (w + 2 * pad + padRB - c.K) / stride + 1;
            long k = (long)c.Ic * c.K * c.K;
            var outT = _ctx.NewF32(T, oh, ow, c.Oc);
            long bandRows = c.K == 1 ? oh
                : Math.Max(1, Math.Min(oh, GemmBudgetBytes() / (4 * Math.Max(1, ow * k))));

            for (long t = 0; t < T; t++)
            {
                using var oFrame = outT.Select(0, t);
                using var o2 = oFrame.View(oh * ow, c.Oc);
                for (long y0 = 0; y0 < oh; y0 += bandRows)
                {
                    long rows = Math.Min(bandRows, oh - y0);
                    using var dst = o2.Narrow(0, y0 * ow, rows * ow);
                    for (int j = 0; j < c.Kd; j++)
                    {
                        Tensor src = tapFrame(t, j);           // [h, w, Ic] contiguous view
                        if (c.K == 1 && stride == 1)
                        {
                            using var col = src.View(h * w, c.Ic);
                            using var band = col.Narrow(0, y0 * ow, rows * ow);
                            if (_ctx.IsCuda) Ops.Addmm(dst, j == 0 ? 0f : 1f, dst, 1f, band, c.TapsT[j]);
                            else DirectOps.CpuGemmABt(band, c.Taps[j], dst, 1f, j == 0 ? 0f : 1f);
                        }
                        else
                        {
                            using var col = _ctx.NewF32(rows * ow, k);
                            Im2Col(col, src, c.Ic, h, w, c.K, oh, ow, pad, stride, y0, rows);
                            if (_ctx.IsCuda) Ops.Addmm(dst, j == 0 ? 0f : 1f, dst, 1f, col, c.TapsT[j]);
                            else DirectOps.CpuGemmABt(col, c.Taps[j], dst, 1f, j == 0 ? 0f : 1f);
                        }
                        src.Dispose();
                    }
                }
                if (c.Bias != null)
                    DirectOps.AddBiasRows(_ctx, o2, c.Bias);
            }
            return outT;
        }

        private void Im2Col(Tensor col, Tensor frame, int ic, long h, long w, int kk,
                            long oh, long ow, int pad, int stride, long y0, long rows)
        {
            if (_ctx.IsCuda)
            {
                if (!CudaWanOps.TryIm2Col(col, frame, ic, (int)h, (int)w, kk, kk, (int)oh, (int)ow,
                                          pad, pad, stride, stride, (int)y0, (int)rows))
                    throw new InvalidOperationException("CUDA im2col kernel unavailable (stale PTX?).");
                return;
            }
            unsafe
            {
                float* src = (float*)CpuNativeHelpers.GetBufferStart(frame);
                float* dst = (float*)CpuNativeHelpers.GetBufferStart(col);
                long k = (long)ic * kk * kk;
                Parallel.For(0, rows * ow, p =>
                {
                    long oy = y0 + p / ow;
                    long ox = p % ow;
                    float* drow = dst + p * k;
                    long ki = 0;
                    for (int cc = 0; cc < ic; cc++)
                        for (int kh = 0; kh < kk; kh++)
                        {
                            long iy = oy * stride - pad + kh;
                            for (int kw = 0; kw < kk; kw++, ki++)
                            {
                                long ix = ox * stride - pad + kw;
                                drow[ki] = (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                    ? src[(iy * w + ix) * ic + cc] : 0f;
                            }
                        }
                });
            }
        }

        /// <summary>Causal conv3d: x [T, H, W, Ic] -> [T, H', W', Oc]; kd == 3 taps read
        /// the 2-frame cross-chunk cache (or causal zero padding on the first chunk).</summary>
        private Tensor CausalConv(ConvT c, Tensor x)
        {
            long T = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2];
            int pad = c.K == 3 ? 1 : 0;

            if (c.Kd == 1)
                return Conv2dFrames(c, T, h, w, (t, j) => x.Select(0, t), pad, 1);

            int idx = _cursor++;
            Tensor prev = _cache[idx];
            Tensor xin;
            if (prev == null)
            {
                xin = PadFrontFrames(x, 2);
            }
            else
            {
                if (prev.Sizes[0] < 2)
                {
                    using var padded = PadFrontFrames(prev, 2 - (int)prev.Sizes[0]);
                    xin = ConcatFrames(padded, x);
                }
                else
                {
                    xin = ConcatFrames(prev, x);
                }
            }

            // new cache = trailing (up to 2) input frames of x, extended with the old
            // cache's last frame when x is a single frame (sd.cpp CACHE_T logic).
            Tensor nc;
            if (T >= 2)
            {
                using var tail = x.Narrow(0, T - 2, 2);
                nc = Ops.NewContiguous(tail);
            }
            else if (prev != null)
            {
                using var last = prev.Narrow(0, prev.Sizes[0] - 1, 1);
                nc = ConcatFrames(last, x);
            }
            else
            {
                nc = Ops.NewContiguous(x);
            }
            SetCache(idx, nc);

            var outT = Conv2dFrames(c, T, h, w, (t, j) => xin.Select(0, t + j), pad, 1);
            xin.Dispose();
            return outT;
        }

        private void SetCache(int idx, Tensor t)
        {
            _cache[idx]?.Dispose();
            _cache[idx] = t;
        }

        private void ResetCaches()
        {
            for (int i = 0; i < _cache.Length; i++) { _cache[i]?.Dispose(); _cache[i] = null; }
            _cursor = 0;
            _chunk = 0;
        }

        private Tensor PadFrontFrames(Tensor x, int n)
        {
            var sizes = x.Sizes.ToArray();
            long t = sizes[0];
            sizes[0] = t + n;
            var outT = _ctx.NewF32(sizes);
            using (var front = outT.Narrow(0, 0, n))
                DirectOps.Zero(_ctx, front);
            using var dst = outT.Narrow(0, n, t);
            Ops.Copy(dst, x);
            return outT;
        }

        private static Tensor ConcatFrames(Tensor a, Tensor b) => Ops.Concat(null, 0, a, b);

        /// <summary>Channel RMS norm: normalize over C at every (t, h, w).</summary>
        private Tensor ChanRms(NormT n, Tensor x)
        {
            long rows = x.Sizes[0] * x.Sizes[1] * x.Sizes[2];
            var outT = _ctx.NewF32(x.Sizes);
            using var x2 = x.View(rows, n.C);
            using var o2 = outT.View(rows, n.C);
            Ops.RMSNorm(o2, x2, n.Gamma, null, NormEps);
            return outT;
        }

        private Tensor ResBlock(ResT r, Tensor x)
        {
            Tensor h = r.Shortcut != null ? CausalConv(r.Shortcut, x) : x;
            var y = ChanRms(r.N0, x);
            Ops.SiLU(y, y);
            var y2 = CausalConv(r.C2, y);
            y.Dispose();
            var y3 = ChanRms(r.N3, y2);
            y2.Dispose();
            Ops.SiLU(y3, y3);
            var y4 = CausalConv(r.C6, y3);
            y3.Dispose();
            Ops.Add(y4, y4, h);
            if (!ReferenceEquals(h, x)) h.Dispose();
            return y4;
        }

        // Spatial single-head self-attention over each frame (mid block).
        private Tensor AttnBlock(AttnT a, Tensor x)
        {
            long T = x.Sizes[0], hw = x.Sizes[1] * x.Sizes[2];
            int C = a.C;
            float scale = 1.0f / MathF.Sqrt(C);
            for (long t = 0; t < T; t++)
            {
                using var frame = x.Select(0, t);
                using var f2 = frame.View(hw, C);
                using var n = _ctx.NewF32(hw, C);
                Ops.RMSNorm(n, f2, a.Norm.Gamma, null, NormEps);
                using var qkv = a.Qkv.Forward(n);
                using var qv = qkv.Narrow(1, 0, C);
                using var q = Ops.NewContiguous(qv);
                using var kv = qkv.Narrow(1, C, C);
                using var k = Ops.NewContiguous(kv);
                using var vv = qkv.Narrow(1, 2L * C, C);
                using var v = Ops.NewContiguous(vv);
                using var attn = DirectOps.Attention(_ctx, q, k, v, 1, C, scale);
                using var o = a.Proj.Forward(attn);
                Ops.Add(o, o, f2);
                Ops.Copy(f2, o);
            }
            return x;
        }

        // Nearest x2 spatial upsample via two duplicate-concat reshapes.
        private Tensor SpatialDup2x(Tensor x)
        {
            long T = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2], C = x.Sizes[3];
            using var wv = x.View(T * h * w, 1, C);
            using var wd = Ops.Concat(null, 1, wv, wv);          // [THW, 2, C]
            using var wide = wd.View(T, h, 2 * w, C);
            using var hv = wide.View(T * h, 1, 2 * w * C);
            using var hd = Ops.Concat(null, 1, hv, hv);          // [TH, 2, 2WC]
            using var outT = hd.View(T, 2 * h, 2 * w, C);
            return Ops.NewContiguous(outT);
        }

        // Resample: optional causal temporal doubling (time_conv, chunks >= 1) then
        // nearest x2 spatial upsample + 3x3 conv.
        private Tensor Upsample(UpT u, Tensor x)
        {
            if (u.TimeConv != null)
            {
                int idx = _cursor++;
                if (_chunk > 0)
                {
                    long T = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2], C = x.Sizes[3];
                    Tensor prev = _cache[idx];
                    Tensor xin = prev == null ? PadFrontFrames(x, 2) : ConcatFrames(prev, x);

                    Tensor nc;
                    if (T >= 2)
                    {
                        using var tail = x.Narrow(0, T - 2, 2);
                        nc = Ops.NewContiguous(tail);
                    }
                    else if (prev != null)
                    {
                        using var last = prev.Narrow(0, prev.Sizes[0] - 1, 1);
                        nc = ConcatFrames(last, x);
                    }
                    else
                    {
                        nc = PadFrontFrames(x, 1);
                    }
                    SetCache(idx, nc);

                    // temporal conv (kd=3, 1x1 spatial) over the assembled window
                    var acc = Conv2dFrames(u.TimeConv, T, h, w, (t, j) => xin.Select(0, t + j), 0, 1);
                    xin.Dispose();
                    x.Dispose();
                    // [T, H, W, 2C] -> [2T, H, W, C]: conv channel index is half*C + c
                    // (torch view(2, c, ...)), frame index becomes half + 2*t.
                    using var split = acc.View(T, h, w, 2, C);
                    using var perm = split.Permute(0, 3, 1, 2, 4);   // [T, 2, H, W, C]
                    using var cont = Ops.NewContiguous(perm);
                    using var cv = cont.View(2 * T, h, w, C);
                    x = Ops.NewContiguous(cv);
                    acc.Dispose();
                }
                // chunk 0 passes through untouched; the cache stays empty and the
                // NEXT chunk zero-pads (sd.cpp Resample upsample3d, chunk_idx == 0).
            }

            var up = SpatialDup2x(x);
            x.Dispose();
            var y = Conv2dFrames(u.SConv, up.Sizes[0], up.Sizes[1], up.Sizes[2],
                                 (t, j) => up.Select(0, t), 1, 1);
            up.Dispose();
            return y;
        }

        // DupUp3D (wan2.2 decoder shortcut): channel-grouped nearest upsample.
        private Tensor DupUp(Tensor x, int ft, bool firstChunk)
        {
            long T = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2], C = x.Sizes[3];
            if (ft == 2)
            {
                // temporal duplicate then spatial x2
                Tensor dup;
                using (var fv = x.View(T, 1, h * w * C))
                using (var fd = Ops.Concat(null, 1, fv, fv))
                using (var v2 = fd.View(2 * T, h, w, C))
                    dup = Ops.NewContiguous(v2);
                var up = SpatialDup2x(dup);
                dup.Dispose();
                if (firstChunk)
                {
                    // causal first chunk keeps only the last of the duplicated frames
                    using var trimmed = up.Narrow(0, 1, 2 * T - 1);
                    var outT = Ops.NewContiguous(trimmed);
                    up.Dispose();
                    return outT;
                }
                return up;
            }

            // ft == 1, C == 2*Co: out[co, 2y+ys, 2x+xs] = in[2*co + ys, y, x]
            long co = C / 2;
            Tensor rows;
            using (var split = x.View(T, h, w, co, 2))
            using (var perm = split.Permute(0, 1, 4, 2, 3))       // [T, H, 2(ys), W, Co]
            using (var cont = Ops.NewContiguous(perm))
            using (var cv = cont.View(T, 2 * h, w, co))
                rows = Ops.NewContiguous(cv);
            // duplicate columns
            using var wv = rows.View(T * 2 * h * w, 1, co);
            using var wd = Ops.Concat(null, 1, wv, wv);
            using var wdv = wd.View(T, 2 * h, 2 * w, co);
            var res = Ops.NewContiguous(wdv);
            rows.Dispose();
            return res;
        }

        // wan2.2 pixel unpatchify: [T, H, W, 12] -> [T, 2H, 2W, 3], channel c*4 + xoff*2 + yoff.
        private Tensor Unpatchify2(Tensor x)
        {
            long T = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2];
            using var split = x.View(T, h, w, 3, 2, 2);           // (c, xoff, yoff)
            using var perm = split.Permute(0, 1, 5, 2, 4, 3);     // [T, H, yoff, W, xoff, C]
            using var cont = Ops.NewContiguous(perm);
            using var cv = cont.View(T, 2 * h, 2 * w, 3);
            return Ops.NewContiguous(cv);
        }

        // ---- decoder ---------------------------------------------------------------

        private Tensor DecodeChunk(Tensor z1)
        {
            _cursor = 0;
            bool w22 = Version == 2;

            var x = CausalConv(_conv1, z1);
            x = SwapDispose(x, ResBlock(_mid0, x));
            x = AttnBlock(_mid1, x);
            x = SwapDispose(x, ResBlock(_mid2, x));

            for (int scale = 0; scale < 4; scale++)
            {
                Tensor xin = w22 ? x : null;
                bool xinAlive = xin != null;
                for (int r = 0; r < 3; r++)
                {
                    var nx = ResBlock(_res[scale * 3 + r], x);
                    if (!ReferenceEquals(x, xin)) x.Dispose();
                    x = nx;
                }
                if (scale < 3)
                {
                    x = Upsample(_up[scale], x);   // Upsample disposes its input
                    if (w22)
                    {
                        // temperal_upsample = [true, true, false]
                        int ft = scale < 2 ? 2 : 1;
                        using var sc = DupUp(xin, ft, _chunk == 0);
                        Ops.Add(x, x, sc);
                    }
                }
                if (xinAlive && !ReferenceEquals(xin, x)) xin.Dispose();
            }

            var hn = ChanRms(_headNorm, x);
            x.Dispose();
            Ops.SiLU(hn, hn);
            var pix = CausalConv(_headConv, hn);
            hn.Dispose();
            if (w22 && PatchSize == 2)
                pix = SwapDispose(pix, Unpatchify2(pix));
            return pix;   // [tOut, H, W, 3]
        }

        private static Tensor SwapDispose(Tensor old, Tensor next)
        {
            if (!ReferenceEquals(old, next)) old.Dispose();
            return next;
        }

        protected override void DecodeNative(float[] z, int t, int latRows, int lw, float[] pixels)
        {
            EnsureDecoder();
            ResetCaches();
            try
            {
                // planar [t, zc, lh, lw] -> channels-last [t, lh, lw, zc]
                var zl = new float[z.LongLength];
                long lhw = (long)latRows * lw;
                Parallel.For(0, t, tt =>
                {
                    long src = (long)tt * ZDim * lhw;
                    long dst = (long)tt * lhw * ZDim;
                    for (int c = 0; c < ZDim; c++)
                        for (long i = 0; i < lhw; i++)
                            zl[dst + i * ZDim + c] = z[src + c * lhw + i];
                });
                using var zT = _ctx.FromFloats(zl, t, latRows, lw, ZDim);
                using var z2 = CausalConv(_conv2, zT);   // post-quant 1x1 over the whole latent

                int outFrame = 0;
                int W = lw * SpatialScale, H = latRows * SpatialScale;
                for (int i = 0; i < t; i++)
                {
                    _chunk = i;
                    using var z1 = z2.Narrow(0, i, 1);
                    using var chunkPix = DecodeChunk(z1);       // [tOut, H, W, 3]
                    int tOut = (int)chunkPix.Sizes[0];
                    float[] host = DirectOps.ToArray(chunkPix);
                    long hwPix = (long)H * W;
                    Parallel.For(0, tOut, f =>
                    {
                        long src = (long)f * hwPix * 3;
                        long dst = (long)(outFrame + f) * 3 * hwPix;
                        for (long p = 0; p < hwPix; p++)
                        {
                            pixels[dst + p] = host[src + p * 3];
                            pixels[dst + hwPix + p] = host[src + p * 3 + 1];
                            pixels[dst + 2 * hwPix + p] = host[src + p * 3 + 2];
                        }
                    });
                    outFrame += tOut;
                }
            }
            finally
            {
                ResetCaches();
            }
        }

        // ---- encoder ---------------------------------------------------------------

        // Encoder Resample: implicit right/bottom zero-pad + 3x3 stride-2 spatial conv,
        // then (downsample3d) a stride-2 temporal conv with a 1-frame causal cache.
        private Tensor EncResample(DownT d, Tensor x)
        {
            long T = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2];
            // OH = (H+1-3)/2+1 = H/2: bottom/right zero pad only (the im2col
            // bounds check supplies the zeros; padRB counts them in the size).
            var y = Conv2dFrames(d.SConv, T, h, w, (t, j) => x.Select(0, t), 0, 2, padRB: 1);
            x.Dispose();
            if (d.TConv == null)
                return y;

            int idx = _cursor++;
            long T2in = y.Sizes[0];
            if (_cache[idx] == null)
            {
                // first chunk: store the (single) frame, temporal conv starts next chunk
                SetCache(idx, Ops.NewContiguous(y));
                return y;
            }

            Tensor prev = _cache[idx];
            Tensor xin;
            using (var last = prev.Narrow(0, prev.Sizes[0] - 1, 1))
                xin = ConcatFrames(last, y);                       // [T + 1, ...]
            using (var tail = y.Narrow(0, T2in - 1, 1))
                SetCache(idx, Ops.NewContiguous(tail));

            // stride-2 temporal conv: out[j] = sum_tap w_tap * xin[2j + tap]
            long tOut = (xin.Sizes[0] - 3) / 2 + 1;
            long hh = xin.Sizes[1], ww = xin.Sizes[2];
            var acc = Conv2dFrames(d.TConv, tOut, hh, ww, (t, j) =>
            {
                // frame 2t + j of xin (frames themselves are contiguous)
                return xin.Narrow(0, 2 * t + j, 1);
            }, 0, 1);
            xin.Dispose();
            y.Dispose();
            return acc;
        }

        // AvgDown3D (wan2.2 encoder shortcut): spatial 2x2 average pool plus (ft == 2)
        // a temporal-pair channel regroup (out ch = c*2 + ift, front zero-pad odd T).
        private Tensor AvgDown(Tensor x, int ft, int fs)
        {
            Tensor cur = x;
            bool own = false;
            if (fs == 2)
            {
                long T = cur.Sizes[0], h = cur.Sizes[1], w = cur.Sizes[2], C = cur.Sizes[3];
                using var split = cur.View(T, h / 2, 2, w / 2, 2, C);
                Tensor sum = null;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                    {
                        using var ny = split.Narrow(2, dy, 1);
                        using var nx = ny.Narrow(4, dx, 1);
                        var q = Ops.NewContiguous(nx);
                        if (sum == null) sum = q;
                        else { Ops.Add(sum, sum, q); q.Dispose(); }
                    }
                Ops.Mul(sum, sum, 0.25f);
                Tensor pooled;
                using (var sv = sum.View(T, h / 2, w / 2, C))
                    pooled = Ops.NewContiguous(sv);
                sum.Dispose();
                cur = pooled;
                own = true;
            }
            if (ft == 2)
            {
                if (cur.Sizes[0] % 2 == 1)
                {
                    var padded = PadFrontFrames(cur, 1);
                    if (own) cur.Dispose();
                    cur = padded;
                    own = true;
                }
                long T = cur.Sizes[0], h = cur.Sizes[1], w = cur.Sizes[2], C = cur.Sizes[3];
                using var split = cur.View(T / 2, 2, h * w, C);
                using var perm = split.Permute(0, 2, 3, 1);        // [T/2, HW, C, 2(ift)]
                using var cont = Ops.NewContiguous(perm);
                using var cv = cont.View(T / 2, h, w, 2 * C);
                var res = Ops.NewContiguous(cv);
                if (own) cur.Dispose();
                return res;
            }
            return own ? cur : Ops.NewContiguous(cur);
        }

        private Tensor EncodeChunk(Tensor xc)
        {
            _cursor = 0;
            bool w22 = Version == 2;

            var x = CausalConv(_encStem, xc);
            for (int scale = 0; scale < 4; scale++)
            {
                Tensor xin = w22 ? x : null;
                for (int r = 0; r < 2; r++)
                {
                    var nx = ResBlock(_encRes[scale * 2 + r], x);
                    if (!ReferenceEquals(x, xin)) x.Dispose();
                    x = nx;
                }
                if (scale < 3)
                    x = EncResample(_encDown[scale], x);   // disposes its input
                if (w22)
                {
                    // temperal_downsample = [false, true, true]; spatial down at scales 0..2
                    int ft = (scale == 1 || scale == 2) ? 2 : 1;
                    int fs = scale < 3 ? 2 : 1;
                    Tensor sc = (ft == 1 && fs == 1) ? xin : AvgDown(xin, ft, fs);
                    Ops.Add(x, x, sc);
                    if (!ReferenceEquals(sc, xin)) sc.Dispose();
                }
                if (xin != null && !ReferenceEquals(xin, x)) xin.Dispose();
            }

            x = SwapDispose(x, ResBlock(_encMid0, x));
            x = AttnBlock(_encMid1, x);
            x = SwapDispose(x, ResBlock(_encMid2, x));

            var hn = ChanRms(_encHeadNorm, x);
            x.Dispose();
            Ops.SiLU(hn, hn);
            var h2 = CausalConv(_encHeadConv, hn);
            hn.Dispose();
            var q = CausalConv(_encQuant, h2);
            h2.Dispose();

            // posterior mean = first ZDim channels
            using var muv = q.Narrow(3, 0, ZDim);
            var mu = Ops.NewContiguous(muv);
            q.Dispose();
            return mu;   // [tOut, lh, lw, ZDim]
        }

        protected override void EncodeNative(float[] x, int t, int ph, int pw, int pc, float[] mu)
        {
            EnsureEncoder();
            ResetCaches();
            try
            {
                // planar [t, pc, ph, pw] -> channels-last [t, ph, pw, pc]
                var xl = new float[x.LongLength];
                long phw = (long)ph * pw;
                Parallel.For(0, t, tt =>
                {
                    long src = (long)tt * pc * phw;
                    long dst = (long)tt * phw * pc;
                    for (int c = 0; c < pc; c++)
                        for (long i = 0; i < phw; i++)
                            xl[dst + i * pc + c] = x[src + c * phw + i];
                });
                using var xT = _ctx.FromFloats(xl, t, ph, pw, pc);

                int chunks = 1 + (t - 1 + 3) / 4;
                int outFrame = 0;
                for (int i = 0; i < chunks; i++)
                {
                    _chunk = i;
                    int from = i == 0 ? 0 : 1 + 4 * (i - 1);
                    int count = Math.Min(t, 1 + 4 * i) - from;
                    using var xc = xT.Narrow(0, from, count);
                    using var muC = EncodeChunk(xc);            // [tOut, lh, lw, z]
                    int tOut = (int)muC.Sizes[0];
                    long lh = muC.Sizes[1], lw = muC.Sizes[2];
                    float[] host = DirectOps.ToArray(muC);
                    long lhw = lh * lw;
                    // channels-last -> native [lw, lh, z, lt] (= planar [lt, z, lh, lw])
                    Parallel.For(0, tOut, f =>
                    {
                        long src = (long)f * lhw * ZDim;
                        long dst = (long)(outFrame + f) * ZDim * lhw;
                        for (long p = 0; p < lhw; p++)
                            for (int c = 0; c < ZDim; c++)
                                mu[dst + c * lhw + p] = host[src + p * ZDim + c];
                    });
                    outFrame += tOut;
                }
            }
            finally
            {
                ResetCaches();
            }
        }

        public override void Dispose()
        {
            ResetCaches();
            _mid1?.Qkv?.Dispose(); _mid1?.Proj?.Dispose();
            _encMid1?.Qkv?.Dispose(); _encMid1?.Proj?.Dispose();
            // Conv/norm tensors are owned by the shared context.
        }
    }
}
