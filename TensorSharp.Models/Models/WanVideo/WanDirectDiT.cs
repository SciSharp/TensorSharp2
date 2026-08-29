// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan DiT on the direct Cuda/Cpu backends: one Predict call = one
// denoising-step velocity prediction, mirroring the GGML whole-graph kernel
// (TSGgml_WanDitForward) op for op. Activations are token-major [seq, dim]
// F32 tensors; weight matrices keep their GGUF quantization on CUDA. The
// AdaLN time modulation supports the Wan 2.2 TI2V two-segment timesteps
// (tokens [0, seq0) run at timestep0 — the conditioned first frame).
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanDirectDiT : IWanDit
    {
        private readonly DirectContext _ctx;
        private readonly GgufFile _gguf;
        private const float Eps = 1e-6f;

        public int Dim { get; }
        public int Layers { get; }
        public int Heads { get; }
        public int Ff { get; }
        public int InDim { get; }
        public int OutDim { get; }
        public int InTok => InDim * WanDiT.PatchH * WanDiT.PatchW;
        public int OutTok => OutDim * WanDiT.PatchH * WanDiT.PatchW;

        private sealed class Block
        {
            public Tensor Mod;                 // [1, 6*dim]
            public DirectLinear SQ, SK, SV, SO;
            public Tensor SNormQ, SNormK;      // [dim] RMS gains
            public Tensor Norm3W, Norm3B;
            public DirectLinear XQ, XK, XV, XO;
            public Tensor XNormQ, XNormK;
            public DirectLinear Ffn0, Ffn2;
        }

        private bool _loaded;
        private DirectLinear _patch, _text0, _text2, _time0, _time2, _tproj, _head;
        private Tensor _headMod;               // [2, dim]
        private Block[] _blocks;

        // RoPE table upload cache: the pipeline reuses one WanRope for the whole
        // denoise, so keep its device tensors across Predict calls.
        private WanRope _ropeKey;
        private Tensor _cosT, _sinT;

        // TS_WAN_DIRECT_TIMING=1: per-phase wall times for one Predict (each phase
        // ends with a device sync, so the numbers are upper bounds).
        private static readonly bool Timing = Environment.GetEnvironmentVariable("TS_WAN_DIRECT_TIMING") == "1";
        private readonly System.Diagnostics.Stopwatch _sw = new();
        private readonly System.Collections.Generic.Dictionary<string, double> _phaseMs = new();

        private void Phase(string name)
        {
            if (!Timing) return;
            if (_ctx.IsCuda)
            {
                using var probe = _ctx.NewF32(1);
                probe.GetElementsAsFloat(1);   // device sync
            }
            if (name != null)
            {
                _phaseMs.TryGetValue(name, out double ms);
                _phaseMs[name] = ms + _sw.Elapsed.TotalMilliseconds;
            }
            _sw.Restart();
        }

        private void PhaseReport()
        {
            if (!Timing) return;
            Phase(null);
            Console.WriteLine("  [wan-direct-dit] " + string.Join(" ", System.Linq.Enumerable.Select(
                System.Linq.Enumerable.OrderByDescending(_phaseMs, kv => kv.Value),
                kv => $"{kv.Key}={kv.Value:F0}ms")));
            _phaseMs.Clear();
        }

        public WanDirectDiT(DirectContext ctx, GgufFile ditGguf)
        {
            _ctx = ctx;
            _gguf = ditGguf;

            int layers = 0;
            foreach (var name in _gguf.Tensors.Keys)
            {
                if (!name.StartsWith("blocks.", StringComparison.Ordinal)) continue;
                int dot = name.IndexOf('.', 7);
                if (dot > 7 && int.TryParse(name.AsSpan(7, dot - 7), out int idx))
                    layers = Math.Max(layers, idx + 1);
            }
            if (layers == 0)
                throw new InvalidDataException("Wan DiT GGUF has no 'blocks.N.*' tensors.");
            Layers = layers;

            var q0 = _gguf.Tensors["blocks.0.self_attn.q.weight"];
            Dim = (int)q0.Shape[0];
            Heads = Dim / WanDiT.HeadDim;
            Ff = (int)_gguf.Tensors["blocks.0.ffn.0.weight"].Shape[1];
            var patch = _gguf.Tensors["patch_embedding.weight"];
            InDim = (int)patch.Shape[3];
            OutDim = (int)(_gguf.Tensors["head.head.weight"].Shape[1] / (WanDiT.PatchH * WanDiT.PatchW));
            if (_gguf.Tensors.ContainsKey("blocks.0.cross_attn.k_img.weight"))
                throw new NotSupportedException(
                    "This Wan DiT is a Wan 2.1 image-to-video checkpoint (CLIP-conditioned cross-attention), " +
                    "which TensorSharp does not support; use the Wan 2.2 models for image-to-video.");
            Console.WriteLine($"  Wan DiT (direct): {Layers} blocks, dim={Dim}, heads={Heads}, ffn={Ff}, in={InDim}");
        }

        private Tensor F32(string name, params long[] sizes)
            => _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(_gguf, name), sizes));

        private void EnsureWeights()
        {
            if (_loaded) return;
            _patch = DirectLinear.FromGgufPatch(_ctx, _gguf, "patch_embedding.weight", "patch_embedding.bias");
            _text0 = DirectLinear.FromGguf(_ctx, _gguf, "text_embedding.0.weight", "text_embedding.0.bias");
            _text2 = DirectLinear.FromGguf(_ctx, _gguf, "text_embedding.2.weight", "text_embedding.2.bias");
            _time0 = DirectLinear.FromGguf(_ctx, _gguf, "time_embedding.0.weight", "time_embedding.0.bias");
            _time2 = DirectLinear.FromGguf(_ctx, _gguf, "time_embedding.2.weight", "time_embedding.2.bias");
            _tproj = DirectLinear.FromGguf(_ctx, _gguf, "time_projection.1.weight", "time_projection.1.bias");
            _head = DirectLinear.FromGguf(_ctx, _gguf, "head.head.weight", "head.head.bias");
            _headMod = F32("head.modulation", 2, Dim);

            _blocks = new Block[Layers];
            for (int l = 0; l < Layers; l++)
            {
                string p = $"blocks.{l}.";
                _blocks[l] = new Block
                {
                    Mod = F32($"{p}modulation", 1, 6L * Dim),
                    SQ = DirectLinear.FromGguf(_ctx, _gguf, $"{p}self_attn.q.weight", $"{p}self_attn.q.bias"),
                    SK = DirectLinear.FromGguf(_ctx, _gguf, $"{p}self_attn.k.weight", $"{p}self_attn.k.bias"),
                    SV = DirectLinear.FromGguf(_ctx, _gguf, $"{p}self_attn.v.weight", $"{p}self_attn.v.bias"),
                    SO = DirectLinear.FromGguf(_ctx, _gguf, $"{p}self_attn.o.weight", $"{p}self_attn.o.bias"),
                    SNormQ = F32($"{p}self_attn.norm_q.weight", Dim),
                    SNormK = F32($"{p}self_attn.norm_k.weight", Dim),
                    Norm3W = F32($"{p}norm3.weight", Dim),
                    Norm3B = F32($"{p}norm3.bias", Dim),
                    XQ = DirectLinear.FromGguf(_ctx, _gguf, $"{p}cross_attn.q.weight", $"{p}cross_attn.q.bias"),
                    XK = DirectLinear.FromGguf(_ctx, _gguf, $"{p}cross_attn.k.weight", $"{p}cross_attn.k.bias"),
                    XV = DirectLinear.FromGguf(_ctx, _gguf, $"{p}cross_attn.v.weight", $"{p}cross_attn.v.bias"),
                    XO = DirectLinear.FromGguf(_ctx, _gguf, $"{p}cross_attn.o.weight", $"{p}cross_attn.o.bias"),
                    XNormQ = F32($"{p}cross_attn.norm_q.weight", Dim),
                    XNormK = F32($"{p}cross_attn.norm_k.weight", Dim),
                    Ffn0 = DirectLinear.FromGguf(_ctx, _gguf, $"{p}ffn.0.weight", $"{p}ffn.0.bias"),
                    Ffn2 = DirectLinear.FromGguf(_ctx, _gguf, $"{p}ffn.2.weight", $"{p}ffn.2.bias"),
                };
            }
            _loaded = true;
        }

        // time embedding: sinusoid [freq_dim] -> e2d [1, dim]; e0 = tproj(silu(e2d)) [1, 6*dim]
        private (Tensor e2d, Tensor e0) TimeEmbed(float timestep)
        {
            var tsin = new float[WanDiT.FreqDim];
            WanDiT.FillSinusoid(tsin, timestep);
            using var tsinT = _ctx.FromFloats(tsin, 1, WanDiT.FreqDim);
            using var e = _time0.Forward(tsinT);
            using var eS = DirectOps.Silu(_ctx, e);
            var e2d = _time2.Forward(eS);
            using var e2dS = DirectOps.Silu(_ctx, e2d);
            var e0 = _tproj.Forward(e2dS);
            return (e2d, e0);
        }

        // Apply the AdaLN modulate per token segment: rows [0, s0) with the B
        // (timestep0) parameters, rows [s0, seq) with the A parameters.
        private void SegModulate(Tensor y, Tensor x, Tensor shiftA, Tensor scaleA,
                                 Tensor shiftB, Tensor scaleB, int seq, int s0)
        {
            if (s0 > 0)
            {
                DirectOps.ModulateRows(_ctx, y, x, shiftB, scaleB, 0, s0);
                DirectOps.ModulateRows(_ctx, y, x, shiftA, scaleA, s0, seq - s0);
            }
            else
            {
                DirectOps.ModulateRows(_ctx, y, x, shiftA, scaleA, 0, seq);
            }
        }

        private void SegGateAdd(Tensor x, Tensor v, Tensor gateA, Tensor gateB, int seq, int s0)
        {
            if (s0 > 0)
            {
                DirectOps.GateAddRows(_ctx, x, v, gateB, 0, s0);
                DirectOps.GateAddRows(_ctx, x, v, gateA, s0, seq - s0);
            }
            else
            {
                DirectOps.GateAddRows(_ctx, x, v, gateA, 0, seq);
            }
        }

        /// <summary>Same contract as WanDiT.Predict (the GGML path).</summary>
        public float[] Predict(float[] xTokens, float[] context, float timestep, WanRope rope, int seq,
                               int seq0 = 0, float timestep0 = 0f)
        {
            EnsureWeights();
            int dim = Dim, heads = Heads, hd = WanDiT.HeadDim, cl = WanTextEncoder.TextLen;
            int s0 = seq0 > 0 && seq0 < seq ? seq0 : 0;
            float scale = 1.0f / MathF.Sqrt(hd);

            if (!ReferenceEquals(_ropeKey, rope))
            {
                _cosT?.Dispose(); _sinT?.Dispose();
                _cosT = _ctx.FromFloats(rope.Cos, seq, hd);
                _sinT = _ctx.FromFloats(rope.Sin, seq, hd);
                _ropeKey = rope;
            }

            Tensor x;
            using (var xIn = _ctx.FromFloats(xTokens, seq, InTok))
                x = _patch.Forward(xIn);                              // [seq, dim]

            var (e2d, e0) = TimeEmbed(timestep);
            Tensor e2dB = null, e0B = null;
            if (s0 > 0) (e2dB, e0B) = TimeEmbed(timestep0);

            Tensor txt;
            using (var ctxT = _ctx.FromFloats(context, cl, WanDiT.TextDim))
            using (var t0 = _text0.Forward(ctxT))
            using (var tg = DirectOps.Gelu(_ctx, t0))
                txt = _text2.Forward(tg);                             // [cl, dim]

            for (int l = 0; l < Layers; l++)
            {
                Block b = _blocks[l];

                using var eb = _ctx.NewF32(1, 6L * dim);
                Ops.Add(eb, e0, b.Mod);
                Tensor ebB = null;
                if (s0 > 0)
                {
                    ebB = _ctx.NewF32(1, 6L * dim);
                    Ops.Add(ebB, e0B, b.Mod);
                }
                Tensor Chunk(Tensor e6, int i) => e6?.Narrow(1, (long)i * dim, dim);
                using var eShiftA = Chunk(eb, 0); using var eScaleA = Chunk(eb, 1); using var eGateA = Chunk(eb, 2);
                using var eShiftM = Chunk(eb, 3); using var eScaleM = Chunk(eb, 4); using var eGateM = Chunk(eb, 5);
                using var eShiftAB = Chunk(ebB, 0); using var eScaleAB = Chunk(ebB, 1); using var eGateAB = Chunk(ebB, 2);
                using var eShiftMB = Chunk(ebB, 3); using var eScaleMB = Chunk(ebB, 4); using var eGateMB = Chunk(ebB, 5);

                // --- self-attention sub-layer ---
                Phase("misc");
                using (var n1 = DirectOps.LayerNorm(_ctx, x, null, null, Eps))
                {
                    SegModulate(n1, n1, eShiftA, eScaleA, eShiftAB, eScaleAB, seq, s0);
                    Phase("norm-mod");
                    using var qLin = b.SQ.Forward(n1);
                    using var q = DirectOps.RmsNorm(_ctx, qLin, b.SNormQ, Eps);
                    using var kLin = b.SK.Forward(n1);
                    using var k = DirectOps.RmsNorm(_ctx, kLin, b.SNormK, Eps);
                    using var v = b.SV.Forward(n1);
                    Phase("qkv-proj");
                    DirectOps.RopeInterleaved(_ctx, q, _cosT, _sinT, heads, hd);
                    DirectOps.RopeInterleaved(_ctx, k, _cosT, _sinT, heads, hd);
                    Phase("rope");
                    using var attn = DirectOps.Attention(_ctx, q, k, v, heads, hd, scale);
                    Phase("self-attn");
                    using var o = b.SO.Forward(attn);
                    SegGateAdd(x, o, eGateA, eGateAB, seq, s0);
                    Phase("o-proj");
                }

                // --- cross-attention sub-layer (affine LayerNorm, no gating) ---
                using (var n3 = DirectOps.LayerNorm(_ctx, x, b.Norm3W, b.Norm3B, Eps))
                using (var xqLin = b.XQ.Forward(n3))
                using (var xq = DirectOps.RmsNorm(_ctx, xqLin, b.XNormQ, Eps))
                using (var xkLin = b.XK.Forward(txt))
                using (var xk = DirectOps.RmsNorm(_ctx, xkLin, b.XNormK, Eps))
                using (var xv = b.XV.Forward(txt))
                {
                    Phase("xproj");
                    using var xattn = DirectOps.Attention(_ctx, xq, xk, xv, heads, hd, scale);
                    Phase("cross-attn");
                    using var xo = b.XO.Forward(xattn);
                    Ops.Add(x, x, xo);
                    Phase("xo-proj");
                }

                // --- FFN sub-layer ---
                using (var n2 = DirectOps.LayerNorm(_ctx, x, null, null, Eps))
                {
                    SegModulate(n2, n2, eShiftM, eScaleM, eShiftMB, eScaleMB, seq, s0);
                    Phase("norm-mod");
                    using var h = b.Ffn0.Forward(n2);
                    using var hg = DirectOps.Gelu(_ctx, h);
                    using var mlp = b.Ffn2.Forward(hg);
                    SegGateAdd(x, mlp, eGateM, eGateMB, seq, s0);
                    Phase("ffn");
                }

                ebB?.Dispose();
            }

            // ---- head: modulated norm + projection ----
            float[] outTokens;
            using (var hm0 = _headMod.Narrow(0, 0, 1))
            using (var hm1 = _headMod.Narrow(0, 1, 1))
            {
                Tensor HeadShift(Tensor e) { var t = _ctx.NewF32(1, dim); Ops.Add(t, hm0, e); return t; }
                Tensor HeadScale(Tensor e) { var t = _ctx.NewF32(1, dim); Ops.Add(t, hm1, e); return t; }
                using var hShift = HeadShift(e2d);
                using var hScale = HeadScale(e2d);
                using var hShiftB = s0 > 0 ? HeadShift(e2dB) : null;
                using var hScaleB = s0 > 0 ? HeadScale(e2dB) : null;
                using var hn = DirectOps.LayerNorm(_ctx, x, null, null, Eps);
                SegModulate(hn, hn, hShift, hScale, hShiftB, hScaleB, seq, s0);
                using var outT = _head.Forward(hn);
                Phase("head");
                outTokens = DirectOps.ToArray(outT);
                Phase("download");
            }
            PhaseReport();

            x.Dispose();
            txt.Dispose();
            e2d.Dispose(); e0.Dispose();
            e2dB?.Dispose(); e0B?.Dispose();

            return WanDiT.ReorderHeadOutput(outTokens, seq, OutDim);
        }

        public void Dispose()
        {
            _cosT?.Dispose(); _sinT?.Dispose();
            _cosT = null; _sinT = null; _ropeKey = null;
            if (_loaded)
            {
                _patch?.Dispose(); _text0?.Dispose(); _text2?.Dispose();
                _time0?.Dispose(); _time2?.Dispose(); _tproj?.Dispose(); _head?.Dispose();
                foreach (var b in _blocks)
                {
                    b.SQ?.Dispose(); b.SK?.Dispose(); b.SV?.Dispose(); b.SO?.Dispose();
                    b.XQ?.Dispose(); b.XK?.Dispose(); b.XV?.Dispose(); b.XO?.Dispose();
                    b.Ffn0?.Dispose(); b.Ffn2?.Dispose();
                }
                _loaded = false;
            }
            // _gguf is owned by WanVideoModel.
        }
    }
}
