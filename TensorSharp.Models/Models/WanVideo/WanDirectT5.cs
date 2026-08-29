// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// UMT5-XXL text encoder on the direct Cuda/Cpu backends. Tokenization and the
// Wan conditioning semantics (pad/truncate to 512 with EOS, -inf key mask,
// zeroed padded rows) are shared with the GGML-path WanTextEncoder; the forward
// itself runs through DirectOps: quantized-resident linears (with the
// activation prescale guard — UMT5 hidden states overflow ggml-style q8_1
// block sums), per-layer T5 relative-position bias built host-side and fed to
// the streaming attention kernel as an additive [heads, 512, 512] tensor.
using System;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanDirectT5 : IWanTextEncoder
    {
        private readonly DirectContext _ctx;
        private readonly WanTextEncoder _tok;   // tokenizer + header parse + GGUF mmap
        private readonly GgufFile _gguf;
        private const float Eps = 1e-6f;

        private bool _loaded;
        private DirectLinear[] _q, _k, _v, _o, _gate, _up, _down;
        private Tensor[] _attnNorm, _ffnNorm;
        private float[][] _relB;                // per layer [32 * heads] (bucket-major)
        private Tensor _finalNorm;
        private int[] _buckets;                 // [n*n] relative-position buckets

        public WanDirectT5(DirectContext ctx, string tePath)
        {
            _ctx = ctx;
            _tok = new WanTextEncoder(tePath);
            _gguf = _tok.Gguf;
        }

        private void EnsureWeights()
        {
            if (_loaded) return;
            int nl = _tok.Layers;
            _q = new DirectLinear[nl]; _k = new DirectLinear[nl]; _v = new DirectLinear[nl];
            _o = new DirectLinear[nl]; _gate = new DirectLinear[nl]; _up = new DirectLinear[nl];
            _down = new DirectLinear[nl];
            _attnNorm = new Tensor[nl]; _ffnNorm = new Tensor[nl];
            _relB = new float[nl][];
            for (int l = 0; l < nl; l++)
            {
                string p = $"enc.blk.{l}.";
                _q[l] = DirectLinear.FromGguf(_ctx, _gguf, $"{p}attn_q.weight", prescale: true);
                _k[l] = DirectLinear.FromGguf(_ctx, _gguf, $"{p}attn_k.weight", prescale: true);
                _v[l] = DirectLinear.FromGguf(_ctx, _gguf, $"{p}attn_v.weight", prescale: true);
                _o[l] = DirectLinear.FromGguf(_ctx, _gguf, $"{p}attn_o.weight", prescale: true);
                _gate[l] = DirectLinear.FromGguf(_ctx, _gguf, $"{p}ffn_gate.weight", prescale: true);
                _up[l] = DirectLinear.FromGguf(_ctx, _gguf, $"{p}ffn_up.weight", prescale: true);
                _down[l] = DirectLinear.FromGguf(_ctx, _gguf, $"{p}ffn_down.weight", prescale: true);
                _attnNorm[l] = _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(_gguf, $"{p}attn_norm.weight"), _tok.Dim));
                _ffnNorm[l] = _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(_gguf, $"{p}ffn_norm.weight"), _tok.Dim));
                _relB[l] = DirectOps.DequantTensor(_gguf, $"{p}attn_rel_b.weight");
            }
            _finalNorm = _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(_gguf, "enc.output_norm.weight"), _tok.Dim));
            _buckets = WanTextEncoder.ComputeRelativeBuckets(WanTextEncoder.TextLen);
            _loaded = true;
        }

        /// <summary>Encode a prompt to the Wan conditioning: [512, Dim] hidden states,
        /// zero at padded positions. Same contract as WanTextEncoder.Encode.</summary>
        public float[] Encode(string text)
        {
            int[] tokens = _tok.PrepareTokens(text, out int realLen);
            EnsureWeights();
            int n = WanTextEncoder.TextLen;
            int dim = _tok.Dim, heads = _tok.Heads, hd = _tok.HeadDim, nl = _tok.Layers;

            var emb = new float[(long)n * dim];
            DirectOps.DequantRows(_gguf, "token_embd.weight", tokens, emb);
            Tensor x = _ctx.FromFloats(emb, n, dim);

            for (int l = 0; l < nl; l++)
            {
                using (var h = DirectOps.RmsNorm(_ctx, x, _attnNorm[l], Eps))
                using (var q = _q[l].Forward(h))
                using (var k = _k[l].Forward(h))
                using (var v = _v[l].Forward(h))
                using (var bias = BuildBias(l, realLen))
                using (var attn = DirectOps.Attention(_ctx, q, k, v, heads, hd, 1.0f, bias))
                using (var o = _o[l].Forward(attn))
                {
                    Ops.Add(x, x, o);
                }

                using (var h2 = DirectOps.RmsNorm(_ctx, x, _ffnNorm[l], Eps))
                using (var g = _gate[l].Forward(h2))
                using (var gg = DirectOps.Gelu(_ctx, g))
                using (var u = _up[l].Forward(h2))
                {
                    Ops.Mul(gg, gg, u);
                    using var d = _down[l].Forward(gg);
                    Ops.Add(x, x, d);
                }
            }

            using (var xin = x)
            using (var final = DirectOps.RmsNorm(_ctx, xin, _finalNorm, Eps))
            {
                float[] outHidden = DirectOps.ToArray(final);
                // zero_out_masked: the DiT cross-attends over all 512 positions with
                // no mask, so padded positions must carry exact zeros.
                Array.Clear(outHidden, realLen * dim, (n - realLen) * dim);
                return outHidden;
            }
        }

        // Additive attention bias [heads, n, n]: the layer's relative-position bias
        // (rel_b memory layout is bucket-major [32, heads]) plus the -inf key mask
        // for padded positions.
        private Tensor BuildBias(int layer, int realLen)
        {
            int n = WanTextEncoder.TextLen, heads = _tok.Heads;
            float[] relB = _relB[layer];
            int[] buckets = _buckets;
            var bias = new float[(long)heads * n * n];
            Parallel.For(0, heads, h =>
            {
                long baseH = (long)h * n * n;
                for (int q = 0; q < n; q++)
                {
                    long row = baseH + (long)q * n;
                    int bucketRow = q * n;
                    for (int k = 0; k < n; k++)
                        bias[row + k] = relB[buckets[bucketRow + k] * heads + h];
                    for (int k = realLen; k < n; k++)
                        bias[row + k] = float.NegativeInfinity;
                }
            });
            return _ctx.FromFloats(bias, heads, n, n);
        }

        public void Dispose()
        {
            if (_loaded)
            {
                foreach (var arr in new[] { _q, _k, _v, _o, _gate, _up, _down })
                    foreach (var lin in arr) lin?.Dispose();
                _loaded = false;
            }
            _tok.Dispose();
        }
    }
}
