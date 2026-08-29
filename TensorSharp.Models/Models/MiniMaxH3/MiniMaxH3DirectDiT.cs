// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 diffusion transformer: the same graph as
// TSGgml_MiniMaxH3DitForward, expressed against the shared DirectOps/DirectLinear
// primitives so it runs with no ggml at all. This is what makes BackendType.Cpu
// (the 100% pure-C# backend) reach MiniMax-H3, which was otherwise the one
// generation family whose every stage was an unconditional whole-model native
// call.
//
// Sequence order and packing are fixed by the caller: text, conditioning frames,
// target audio, target video. There is no attention mask anywhere in this model,
// and modulation is applied per segment of that packed sequence.
using System;
using System.Collections.Generic;

using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;          // H3DitSegment / H3CondChunk: plain layout structs, no native call
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed MiniMax-H3 DiT step (no ggml).</summary>
    internal sealed class MiniMaxH3DirectDiT : IDisposable
    {
        private sealed class BlockW
        {
            public Tensor Norm1, Norm2, QNorm, KNorm;
            public DirectLinear AdaLn, Qkv, Out, Fc1, Fc2;
        }

        private readonly DirectContext _ctx;
        private readonly MiniMaxH3Config _cfg;
        private readonly BlockW[] _blocks;
        private readonly BlockW[] _refiner;      // no AdaLn on refiner blocks
        private readonly DirectLinear _videoPatch, _audioPatch, _conditionProj;
        private readonly DirectLinear _finalAdaLn, _finalVideoOut, _finalAudioOut;
        private readonly Tensor _finalNorm, _refinerFinalNorm;
        private readonly List<DirectLinear> _owned = new();
        private bool _disposed;

        public MiniMaxH3DirectDiT(GgufFile gguf, MiniMaxH3Config config, IAllocator allocator)
        {
            _cfg = config;
            _ctx = new DirectContext(allocator);
            if (_ctx.IsCuda)
                throw new NotSupportedException(
                    "MiniMaxH3DirectDiT is the managed CPU path; MiniMax-H3 on BackendType.Cuda " +
                    "is not implemented (use a GGML backend, or BackendType.Cpu).");

            string p = config.Prefix;
            DirectLinear Lin(string w, string b)
            {
                var l = DirectLinear.FromGguf(_ctx, gguf, w, b);
                _owned.Add(l);
                return l;
            }
            Tensor Vec(string name) =>
                _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(gguf, name),
                                         (long)gguf.Tensors[name].NumElements));

            _videoPatch = Lin(p + "video_patch_proj.weight", p + "video_patch_proj.bias");
            _audioPatch = Lin(p + "audio_patch_proj.weight", p + "audio_patch_proj.bias");
            _conditionProj = Lin(p + "condition_proj.weight", p + "condition_proj.bias");
            _finalAdaLn = Lin(p + "final_layer.adaln_proj.linear.weight",
                              p + "final_layer.adaln_proj.linear.bias");
            _finalVideoOut = Lin(p + "final_layer.video_out.weight", p + "final_layer.video_out.bias");
            _finalAudioOut = Lin(p + "final_layer.audio_out.weight", p + "final_layer.audio_out.bias");
            _finalNorm = Vec(p + "final_layer.norm.weight");
            _refinerFinalNorm = Vec(p + "token_refiner.final_norm.weight");

            _refiner = new BlockW[config.TokenRefinerLayers];
            for (int i = 0; i < _refiner.Length; i++)
            {
                string b = p + "token_refiner.blocks." + i + ".";
                _refiner[i] = new BlockW
                {
                    Norm1 = Vec(b + "norm1.weight"),
                    Norm2 = Vec(b + "norm2.weight"),
                    QNorm = Vec(b + "attn.q_norm.weight"),
                    KNorm = Vec(b + "attn.k_norm.weight"),
                    Qkv = Lin(b + "attn.qkv_proj.weight", null),
                    Out = Lin(b + "attn.out_proj.weight", null),
                    Fc1 = Lin(b + "mlp.fc1.weight", null),
                    Fc2 = Lin(b + "mlp.fc2.weight", null),
                };
            }

            _blocks = new BlockW[config.NumLayers];
            for (int i = 0; i < _blocks.Length; i++)
            {
                string b = p + "blocks." + i + ".";
                _blocks[i] = new BlockW
                {
                    Norm1 = Vec(b + "norm1.weight"),
                    Norm2 = Vec(b + "norm2.weight"),
                    QNorm = Vec(b + "attn.q_norm.weight"),
                    KNorm = Vec(b + "attn.k_norm.weight"),
                    AdaLn = Lin(b + "adaln_proj.linear.weight", b + "adaln_proj.linear.bias"),
                    Qkv = Lin(b + "attn.qkv_proj.weight", null),
                    Out = Lin(b + "attn.out_proj.weight", null),
                    Fc1 = Lin(b + "mlp.fc1.weight", null),
                    Fc2 = Lin(b + "mlp.fc2.weight", null),
                };
            }
        }

        // ---- small helpers ---------------------------------------------------

        private static Tensor Rows(Tensor src, int from, int count)
        {
            using var view = src.Narrow(0, from, count);
            return Ops.NewContiguous(view);
        }

        private static Tensor Cols(Tensor src, int from, int count)
        {
            using var view = src.Narrow(1, from, count);
            return Ops.NewContiguous(view);
        }

        /// <summary>One row of a modulation table as a contiguous [cols] vector.</summary>
        private static Tensor ModRow(Tensor mods, int row)
        {
            using var view = mods.Narrow(0, row, 1);
            using var c = Ops.NewContiguous(view);
            return c.View(mods.Sizes[1]);
        }

        private static void CopyRows(Tensor dst, int dstRow, Tensor src)
        {
            using var target = dst.Narrow(0, dstRow, src.Sizes[0]);
            Ops.Copy(target, src);
        }

        /// <summary>
        /// Rotate-half RoPE over the leading <paramref name="rot"/> channels of every
        /// head, in place. cos/sin are per-token tables [seq, rot] over the packed
        /// sequence - NOT the pair-duplicated layout DirectOps.RopeInterleaved wants,
        /// which is why this model needs its own.
        /// </summary>
        private static unsafe void RopeHalf(Tensor x, float[] cos, float[] sin,
                                            int seq, int heads, int hd, int rot)
        {
            if (rot <= 0) return;
            int half = rot / 2;
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            fixed (float* pcos = cos, psin = sin)
            {
                float* pc = pcos, ps = psin;
                CpuWorkerPool.Shared.For(seq, t =>
                {
                    float* ct = pc + (long)t * rot;
                    float* st = ps + (long)t * rot;
                    float* row = px + (long)t * heads * hd;
                    for (int h = 0; h < heads; h++)
                    {
                        float* v = row + (long)h * hd;
                        for (int i = 0; i < half; i++)
                        {
                            // rotate_half(x) = [-x2, x1]
                            float a = v[i], b = v[i + half];
                            v[i] = a * ct[i] - b * st[i];
                            v[i + half] = b * ct[i + half] + a * st[i + half];
                        }
                    }
                });
            }
        }

        private Tensor Attention(Tensor h, BlockW w, int seq, float[] cos, float[] sin)
        {
            int heads = _cfg.NumAttentionHeads, hd = _cfg.AttentionHeadDim;
            int inner = _cfg.AttentionInnerDim, rot = _cfg.RopeRotaryDim;
            float scale = 1.0f / MathF.Sqrt(hd);

            Tensor q, k;
            Tensor v;
            using (Tensor qkv = w.Qkv.Forward(h))          // [seq, 3*inner]
            {
                q = Cols(qkv, 0, inner);
                k = Cols(qkv, inner, inner);
                v = Cols(qkv, 2 * inner, inner);
            }

            try
            {
                // q/k RMS norm is per HEAD, over the head dim.
                if (w.QNorm != null) q = NormHeads(q, w.QNorm, seq, heads, hd);
                if (w.KNorm != null) k = NormHeads(k, w.KNorm, seq, heads, hd);

                RopeHalf(q, cos, sin, seq, heads, hd, rot);
                RopeHalf(k, cos, sin, seq, heads, hd, rot);

                using Tensor merged = DirectOps.Attention(_ctx, q, k, v, heads, hd, scale);
                return w.Out.Forward(merged);
            }
            finally
            {
                q?.Dispose();
                k?.Dispose();
                v?.Dispose();
            }
        }

        private Tensor NormHeads(Tensor x, Tensor gain, int seq, int heads, int hd)
        {
            using (x)
            using (Tensor flat = x.View((long)seq * heads, hd))
            using (Tensor n = DirectOps.RmsNorm(_ctx, flat, gain, _cfg.NormEps))
                return n.View(seq, (long)heads * hd);
        }

        /// <summary>SwiGLU with the gate first: fc1 emits [gate | value].</summary>
        private Tensor Mlp(Tensor h, BlockW w)
        {
            int ffn = _cfg.FfnHiddenSize;
            using Tensor gv = w.Fc1.Forward(h);
            using Tensor gate = Cols(gv, 0, ffn);
            using Tensor val = Cols(gv, ffn, ffn);
            using Tensor act = DirectOps.Silu(_ctx, gate);
            Ops.Mul(act, act, val);
            return w.Fc2.Forward(act);
        }

        // ---- modulation over the packed sequence -----------------------------

        /// <summary>x = x * (1 + scale) + shift, one modulation row per segment.</summary>
        private void Modulate(Tensor h, Tensor mods, H3DitSegment[] segs, int shiftIdx, int scaleIdx)
        {
            foreach (var sg in segs)
            {
                int n = sg.End - sg.Start;
                if (n <= 0) continue;
                using Tensor sh = ModRow(mods, sg.Col + shiftIdx);
                using Tensor sc = ModRow(mods, sg.Col + scaleIdx);
                DirectOps.ModulateRows(_ctx, h, h, sh, sc, sg.Start, n);
            }
        }

        /// <summary>x += upd * gate, one gate row per segment.</summary>
        private void GatedResidual(Tensor x, Tensor upd, Tensor mods, H3DitSegment[] segs, int gateIdx)
        {
            foreach (var sg in segs)
            {
                int n = sg.End - sg.Start;
                if (n <= 0) continue;
                using Tensor g = ModRow(mods, sg.Col + gateIdx);
                DirectOps.GateAddRows(_ctx, x, upd, g, sg.Start, n);
            }
        }

        // ---- forward ---------------------------------------------------------

        /// <summary>
        /// One diffusion step over the packed sequence. The inputs are exactly the
        /// buffers the native descriptor is handed, so the two paths can be
        /// compared step for step.
        /// </summary>
        public (float[] Video, float[] Audio) Forward(
            float[] videoInput, int conditionCount, int videoCount,
            float[] audioLatent, int audioCount,
            float[] conditionAudio, int conditionAudioCount,
            IReadOnlyList<H3CondChunk> conditionChunks,
            float[] textHidden, int textCount,
            float[] timeEmbed, int nTimesteps,
            float[] cos, float[] sin,
            H3DitSegment[] segments,
            int nTok, int audioStart, int videoStart,
            int audioCol, int videoCol,
            float videoScale, float audioScale,
            int numBlocks)
        {
            int hidden = _cfg.HiddenSize;
            float eps = _cfg.NormEps;

            using Tensor temb = _ctx.FromFloats(timeEmbed, nTimesteps, _cfg.TimeEmbedDim);

            // ---- text: project, then refine ----
            Tensor ctxTok = null;
            if (textCount > 0 && textHidden != null)
            {
                using (Tensor tin = _ctx.FromFloats(textHidden, textCount, _cfg.TextDim))
                    ctxTok = _conditionProj.Forward(tin);

                // Text sits at the head of the packed sequence, so the refiner wants
                // the matching PREFIX of the rope tables - which is what indexing
                // them from token 0 gives.
                foreach (var rw in _refiner)
                {
                    using (Tensor n1 = DirectOps.RmsNorm(_ctx, ctxTok, rw.Norm1, eps))
                    using (Tensor a = Attention(n1, rw, textCount, cos, sin))
                        Ops.Add(ctxTok, ctxTok, a);
                    using (Tensor n2 = DirectOps.RmsNorm(_ctx, ctxTok, rw.Norm2, eps))
                    using (Tensor m = Mlp(n2, rw))
                        Ops.Add(ctxTok, ctxTok, m);
                }
                if (_refinerFinalNorm != null)
                {
                    using Tensor prev = ctxTok;
                    ctxTok = DirectOps.RmsNorm(_ctx, prev, _refinerFinalNorm, eps);
                }
            }

            // ---- patchify video and audio, then pack ----
            // Conditioning frames share the patch projection of the target, exactly
            // as the reference concatenates them before projecting once.
            Tensor allVideo, audioTok, condAudioTok = null;
            using (Tensor vin = _ctx.FromFloats(videoInput, conditionCount + videoCount, _cfg.VideoPatchDim))
                allVideo = _videoPatch.Forward(vin);
            using (Tensor ain = _ctx.FromFloats(audioLatent, audioCount, _cfg.AudioLatentChannels))
                audioTok = _audioPatch.Forward(ain);
            if (conditionAudio != null && conditionAudioCount > 0)
            {
                using Tensor acin = _ctx.FromFloats(conditionAudio, conditionAudioCount,
                                                    _cfg.AudioLatentChannels);
                condAudioTok = _audioPatch.Forward(acin);
            }

            Tensor condTok = null, videoTok = allVideo;
            if (conditionCount > 0)
            {
                condTok = Rows(allVideo, 0, conditionCount);
                videoTok = Rows(allVideo, conditionCount, videoCount);
                allVideo.Dispose();
            }

            // With a chunk list the conditioning run is rebuilt in the order the
            // caller gave, taking each chunk from whichever projection it named.
            if (conditionChunks != null && conditionChunks.Count > 0)
            {
                var rebuilt = _ctx.NewF32(conditionCount + conditionAudioCount, hidden);
                int dst = 0, videoRow = 0, audioRow = 0;
                foreach (var ch in conditionChunks)
                {
                    if (ch.Count <= 0) continue;
                    if (ch.Kind == 0)
                    {
                        if (condTok == null || videoRow + ch.Count > conditionCount)
                            throw new InvalidOperationException(
                                "MiniMax-H3 DiT: conditioning video chunks overrun.");
                        using Tensor part = Rows(condTok, videoRow, ch.Count);
                        CopyRows(rebuilt, dst, part);
                        videoRow += ch.Count;
                    }
                    else
                    {
                        if (condAudioTok == null || audioRow + ch.Count > conditionAudioCount)
                            throw new InvalidOperationException(
                                "MiniMax-H3 DiT: conditioning audio chunks overrun.");
                        using Tensor part = Rows(condAudioTok, audioRow, ch.Count);
                        CopyRows(rebuilt, dst, part);
                        audioRow += ch.Count;
                    }
                    dst += ch.Count;
                }
                if (videoRow != conditionCount || audioRow != conditionAudioCount)
                    throw new InvalidOperationException(
                        "MiniMax-H3 DiT: conditioning chunks do not cover every row.");
                condTok?.Dispose();
                condTok = rebuilt;
            }
            condAudioTok?.Dispose();

            // Sequence order is fixed: text, conditioning frames, target audio, target video.
            Tensor x = _ctx.NewF32(nTok, hidden);
            int row = 0;
            void Append(Tensor part)
            {
                if (part == null) return;
                CopyRows(x, row, part);
                row += (int)part.Sizes[0];
                part.Dispose();
            }
            Append(ctxTok);
            Append(condTok);
            Append(audioTok);
            Append(videoTok);
            if (row != nTok)
                throw new InvalidOperationException(
                    $"MiniMax-H3 DiT: packed {row} tokens but the layout declares {nTok}.");

            try
            {
                int nl = numBlocks > 0 ? Math.Min(numBlocks, _blocks.Length) : _blocks.Length;
                for (int l = 0; l < nl; l++)
                {
                    BlockW bw = _blocks[l];
                    // No SiLU: the curve-table variant feeds the interpolated
                    // embedding straight into the projection.
                    using Tensor modsRaw = bw.AdaLn.Forward(temb);          // [nts, 18*hidden]
                    using Tensor mods = modsRaw.View(18L * nTimesteps, hidden);

                    using (Tensor h = DirectOps.RmsNorm(_ctx, x, bw.Norm1, eps))
                    {
                        Modulate(h, mods, segments, 0, 1);
                        using Tensor a = Attention(h, bw, nTok, cos, sin);
                        GatedResidual(x, a, mods, segments, 2);
                    }
                    using (Tensor h = DirectOps.RmsNorm(_ctx, x, bw.Norm2, eps))
                    {
                        Modulate(h, mods, segments, 3, 4);
                        using Tensor m = Mlp(h, bw);
                        GatedResidual(x, m, mods, segments, 5);
                    }
                }

                // ---- final layer ----
                // The final projection emits only shift+scale, and the spans index
                // it by TIMESTEP row rather than timestep*3+modality, so the column
                // bases differ from the per-block ones.
                using Tensor fmodsRaw = _finalAdaLn.Forward(temb);          // [nts, 2*hidden]
                using Tensor fmods = fmodsRaw.View(2L * nTimesteps, hidden);

                Tensor FinalSlice(int start, int count, int col)
                {
                    Tensor part;
                    using (Tensor slice = Rows(x, start, count))
                        part = DirectOps.RmsNorm(_ctx, slice, _finalNorm, eps);
                    using Tensor sh = ModRow(fmods, col);
                    using Tensor sc = ModRow(fmods, col + 1);
                    DirectOps.ModulateRows(_ctx, part, part, sh, sc, 0, count);
                    return part;
                }

                // The model emits velocities: video negated, audio scaled by
                // d(sigma_audio)/d(sigma_video) so one Euler step advances both.
                float[] videoOut, audioOut;
                using (Tensor vs = FinalSlice(videoStart, videoCount, videoCol))
                using (Tensor vo = _finalVideoOut.Forward(vs))
                {
                    Ops.Mul(vo, vo, videoScale);
                    videoOut = DirectOps.ToArray(vo);
                }
                using (Tensor asl = FinalSlice(audioStart, audioCount, audioCol))
                using (Tensor ao = _finalAudioOut.Forward(asl))
                {
                    Ops.Mul(ao, ao, audioScale);
                    audioOut = DirectOps.ToArray(ao);
                }
                return (videoOut, audioOut);
            }
            finally
            {
                x.Dispose();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var l in _owned) l.Dispose();
            _owned.Clear();
            _ctx.Dispose();
        }
    }
}
