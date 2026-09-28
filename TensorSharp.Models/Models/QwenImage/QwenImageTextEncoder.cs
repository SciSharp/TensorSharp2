// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Qwen3-VL-8B text encoder for Qwen-Image-2.1. We only need a single forward
// pass over the (already-tokenized) prompt to produce the 4096-dim conditioning
// hidden states the DiT consumes — no generation, no KV cache, no logits.
//
// This is a Qwen3-VL decoder trunk: RMSNorm -> GQA attention (separate q/k/v
// projections, per-head Q/K RMS norms, interleaved M-RoPE) -> RMSNorm -> SwiGLU
// MLP, with the vision tower's DeepStack embeddings added after the first three
// blocks. The conditioning is the last block's output, before the final RMSNorm.
// It is its own ModelBase instance over the text-encoder GGUF so it reuses the
// fast quantized matmul / ggml primitives.
// ============================================================================
using System;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using TensorSharp.Core;
using TensorSharp.Runtime;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Image conditioning injected into the text encoder at the image-pad span.</summary>
    internal sealed class ImageCond
    {
        public int Start;          // index of first image-pad token
        public int Count;          // number of image-pad tokens (= llm_h * llm_w)
        public int GridH, GridW;   // vision patch grid (before 2x merge)
        public float[][] DeepStack; // additions after the first three language blocks
        public float[] Embeds;     // [Count, hidden] merged vision embeddings
    }

    internal sealed class QwenImageTextEncoder : ModelBase
    {
        private readonly int _numHeads, _numKVHeads, _headDim, _numLayers;
        private readonly float _ropeBase, _eps;
        private static readonly string TraceDirectory = Environment.GetEnvironmentVariable("TS_QWEN_TE_TRACE_DIR");
        private static readonly int TraceLayer = int.TryParse(Environment.GetEnvironmentVariable("TS_QWEN_TE_TRACE_LAYER"), out int layer) ? layer : 0;
        internal IAllocator ConditionerAllocator => _allocator;

        public int HiddenSize => Config.HiddenSize;

        public QwenImageTextEncoder(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            try { QwenImage21CompanionValidation.ValidateText(_gguf); }
            catch { base.Dispose(); throw; }
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") ?? "qwen3vl" };
            ParseBaseConfig();
            _numHeads = Config.NumHeads;
            _numKVHeads = Config.NumKVHeads;
            _headDim = Config.HeadDim;
            _numLayers = Config.NumLayers;
            _ropeBase = Config.RopeBase > 0 ? Config.RopeBase : 1000000f;
            _eps = Config.Eps > 0 ? Config.Eps : 1e-6f;
            ParseTokenizer();
            EnsureQuantBackendAvailable();
            LoadWeights();
        }

        // M-RoPE 3D positions [3*seq] = (t[seq], h[seq], w[seq]); for text-only all three equal.
        private int[] _mropePos;

        /// <summary>Text-only conditioning (M-RoPE degenerates to 1D RoPE).</summary>
        public float[] EncodeHidden(int[] tokens) => EncodeHidden(tokens, (ImageCond[])null);

        /// <summary>
        /// Run the trunk over <paramref name="tokens"/> and return row-major
        /// <c>[seqLen, hidden]</c> conditioning: the last block's output, before the final
        /// RMSNorm. Each <paramref name="imgs"/> entry (ordered by <see cref="ImageCond.Start"/>,
        /// non-overlapping) replaces its <c>&lt;|image_pad|&gt;</c> span's token embeddings with
        /// the vision encoder's merged embeds, applies 3D M-RoPE positions for that span
        /// (<c>get_rope_index</c> semantics) and adds its DeepStack embeddings after the first
        /// blocks. The caller drops the template prefix.
        /// </summary>
        public unsafe float[] EncodeHidden(int[] tokens, ImageCond[] imgs)
        {
            if (!string.IsNullOrEmpty(TraceDirectory)) System.IO.Directory.CreateDirectory(TraceDirectory);
            int seq = tokens.Length;
            if (imgs != null && imgs.Length == 0) imgs = null;
            _mropePos = BuildPositions(tokens.Length, imgs);
            _ropeCos = _ropeSin = null;

            // Fused whole-trunk path (TSGgml_QwenTeTrunk): all layers in ONE device graph
            // instead of ~10 host round-trips per layer. Falls back to the per-op loop below.
            if (TryFusedEncode(tokens, imgs, out float[] fusedOut))
                return fusedOut;

            long profileStart = System.Diagnostics.Stopwatch.GetTimestamp();
            _profileLinear = _profileAttention = _profileNorm = 0;
            Tensor hidden = Embedding(tokens);             // [seq, hidden]
            if (imgs != null)
            {
                // overwrite the image-pad rows with the merged vision embeddings
                float* hp = GetFloatPtr(hidden);
                int H = Config.HiddenSize;
                foreach (var img in imgs)
                    for (int i = 0; i < img.Count; i++)
                        fixed (float* src = &img.Embeds[(long)i * H])
                            Buffer.MemoryCopy(src, hp + (long)(img.Start + i) * H, (long)H * 4, (long)H * 4);
                InvalidateTensorDeviceCache(hidden);
            }

            for (int layer = 0; layer < _numLayers; layer++)
            {
                string p = $"blk.{layer}";
                TraceTensor(layer, "input", hidden);
                long normStart = System.Diagnostics.Stopwatch.GetTimestamp();
                Tensor normedAttn = RMSNormOp(hidden, $"{p}.attn_norm.weight");
                _profileNorm += System.Diagnostics.Stopwatch.GetTimestamp() - normStart;
                using (Tensor normed = normedAttn)
                using (Tensor attnOut = Attention(normed, p, seq, layer))
                {
                    TraceTensor(layer, "attn_out", attnOut);
                    Tensor res = Ops.Add(hidden, hidden, attnOut);
                    if (!ReferenceEquals(res, hidden)) { hidden.Dispose(); hidden = res; }
                }
                TraceTensor(layer, "attn_residual", hidden);
                normStart = System.Diagnostics.Stopwatch.GetTimestamp();
                Tensor normedFfn = RMSNormOp(hidden, $"{p}.ffn_norm.weight");
                _profileNorm += System.Diagnostics.Stopwatch.GetTimestamp() - normStart;
                using (Tensor normed2 = normedFfn)
                using (Tensor ffnOut = SwiGluFfn(normed2, p, layer))
                {
                    Tensor res = Ops.Add(hidden, hidden, ffnOut);
                    if (!ReferenceEquals(res, hidden)) { hidden.Dispose(); hidden = res; }
                }
                TraceTensor(layer, "output", hidden);
                if (imgs != null)
                {
                    float* hp = GetFloatPtr(hidden);
                    foreach (var img in imgs)
                    {
                        if (img.DeepStack == null || layer >= img.DeepStack.Length) continue;
                        var extra = img.DeepStack[layer];
                        for (int i = 0; i < img.Count; i++)
                            for (int d = 0; d < Config.HiddenSize; d++)
                                hp[(long)(img.Start + i) * Config.HiddenSize + d] += extra[(long)i * Config.HiddenSize + d];
                    }
                    InvalidateTensorDeviceCache(hidden);
                }
            }
            var output = TensorToHostFloat(hidden, (long)seq * Config.HiddenSize);
            hidden.Dispose();
            if (ProfileOn)
            {
                double ms(long ticks) => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                double total = ms(System.Diagnostics.Stopwatch.GetTimestamp() - profileStart);
                Console.WriteLine($"  [te-profile] {seq} tokens: total {total:F0} ms, linear {ms(_profileLinear):F0} ms, " +
                    $"attention {ms(_profileAttention):F0} ms, rms norms {ms(_profileNorm):F0} ms, " +
                    $"other {total - ms(_profileLinear) - ms(_profileAttention) - ms(_profileNorm):F0} ms");
            }
            return output; // Qwen-Image-2.1 uses the last block before output_norm.
        }

        // get_rope_index: text tokens get sequential positions (all 3 equal); image tokens get
        // (t=cur, h=cur+row, w=cur+col) over the llm grid; after each image, cur += max(llm_h,llm_w).
        internal static int[] BuildPositions(int seq, ImageCond[] imgs)
        {
            var pos = new int[3 * seq];   // [t(0..seq), h(seq..2seq), w(2seq..3seq)]
            int cur = 0, s = 0, next = 0;
            while (s < seq)
            {
                if (imgs != null && next < imgs.Length && s == imgs[next].Start)
                {
                    var img = imgs[next]; next++;
                    int lh = img.GridH / 2, lw = img.GridW / 2;
                    for (int r = 0; r < lh; r++)
                        for (int c = 0; c < lw; c++)
                        {
                            pos[s] = cur; pos[seq + s] = cur + r; pos[2 * seq + s] = cur + c; s++;
                        }
                    cur += Math.Max(lh, lw);
                }
                else
                {
                    pos[s] = cur; pos[seq + s] = cur; pos[2 * seq + s] = cur; cur++; s++;
                }
            }
            return pos;
        }

        private Tensor Attention(Tensor input, string prefix, int seq, int layer)
        {
            TraceTensor(layer, "norm1", input);
            int qDim = _numHeads * _headDim, kvDim = _numKVHeads * _headDim;
            float scale = 1.0f / MathF.Sqrt(_headDim);

            Tensor q = LinearWithBias(input, $"{prefix}.attn_q.weight", $"{prefix}.attn_q.bias");
            Tensor k = LinearWithBias(input, $"{prefix}.attn_k.weight", $"{prefix}.attn_k.bias");
            Tensor v = LinearWithBias(input, $"{prefix}.attn_v.weight", $"{prefix}.attn_v.bias");
            TraceTensor(layer, "q", q); TraceTensor(layer, "k", k); TraceTensor(layer, "v", v);

            NormalizeHeads(q, $"{prefix}.attn_q_norm.weight", _numHeads, seq);
            NormalizeHeads(k, $"{prefix}.attn_k_norm.weight", _numKVHeads, seq);
            TraceTensor(layer, "qnorm", q); TraceTensor(layer, "knorm", k);
            ApplyMRoPE(q, _numHeads, seq);
            ApplyMRoPE(k, _numKVHeads, seq);
            TraceTensor(layer, "qrope", q); TraceTensor(layer, "krope", k);

            if (UseManagedAttention)
            {
                // Pure-C# backend: causal GQA straight from the [seq, heads*dim] projections.
                // No head-first copies, no group-expanded K/V and no seq x seq score tensors.
                var merged = new Tensor(_allocator, DType.Float32, seq, qDim);
                long attentionStart = System.Diagnostics.Stopwatch.GetTimestamp();
                CausalGqaAttention(q, k, v, merged, seq, scale);
                _profileAttention += System.Diagnostics.Stopwatch.GetTimestamp() - attentionStart;
                q.Dispose(); k.Dispose(); v.Dispose();
                TraceTensor(layer, "merged", merged);
                using (merged)
                    return Linear(merged, $"{prefix}.attn_output.weight");
            }

            Tensor qHeads = ReshapeToHeads(q, _numHeads, seq, _headDim); q.Dispose();
            Tensor kHeads = ReshapeToHeads(k, _numKVHeads, seq, _headDim); k.Dispose();
            Tensor vHeads = ReshapeToHeads(v, _numKVHeads, seq, _headDim); v.Dispose();

            int groupSize = _numHeads / _numKVHeads;
            using Tensor kExp = ExpandKVHeads(kHeads, groupSize, seq); kHeads.Dispose();
            using Tensor vExp = ExpandKVHeads(vHeads, groupSize, seq); vHeads.Dispose();

            using Tensor kT = kExp.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, seq, seq);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            TraceTensor(layer, "scores", scores);
            qHeads.Dispose();

            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(scores, sinks: null,
                    numHeads: _numHeads, seqLen: seq, kvLen: seq,
                    maskStartPos: 0, slidingWindow: 0, scale: 1.0f);
            }
            else
            {
                Ops.AddCausalMask(scores, seq, 0, float.NegativeInfinity);
                Ops.Softmax(scores, scores);
            }
            TraceTensor(layer, "probs", scores);

            var attnOut = new Tensor(_allocator, DType.Float32, _numHeads, seq, _headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExp);
            scores.Dispose();

            using Tensor flat = ReshapeFromHeads(attnOut, _numHeads, seq, _headDim);
            attnOut.Dispose();
            TraceTensor(layer, "merged", flat);
            return LinearForward(flat, $"{prefix}.attn_output.weight");
        }

        private Tensor SwiGluFfn(Tensor input, string prefix, int layer)
        {
            TraceTensor(layer, "norm2", input);
            Tensor gate = Linear(input, $"{prefix}.ffn_gate.weight");
            TraceTensor(layer, "gate", gate);
            using (Tensor up = Linear(input, $"{prefix}.ffn_up.weight"))
            {
                TraceTensor(layer, "up", up);
                SiluMulInPlace(gate, up);
            }
            TraceTensor(layer, "activated", gate);
            Tensor down = Linear(gate, $"{prefix}.ffn_down.weight");
            TraceTensor(layer, "down", down);
            gate.Dispose();
            return down;
        }

        // Multimodal RoPE (rotate_half / NeoX) over [seq, numHeads*headDim]. Each of the
        // headDim/2 frequency indices is assigned a t/h/w axis by the interleaved layout
        // (InterleavedRopeAxis); the rotation angle uses that axis's 3D position component.
        // Text tokens have all three components equal, so this is identical to standard 1D RoPE.
        // The cos/sin of every (token, frequency) are built once per EncodeHidden and shared by
        // q and k of all layers (72 applications): the per-element Math.Pow + sincos was the
        // cost of this pass. The table holds exactly the values the old inline code computed,
        // and the rotation keeps its multiply/add order, so the result is bit-identical.
        private float[] _ropeCos, _ropeSin;

        private unsafe void ApplyMRoPE(Tensor data, int numHeads, int seq)
        {
            int half = _headDim / 2;     // 64
            if (_ropeCos == null) (_ropeCos, _ropeSin) = BuildRopeTables(_mropePos, seq, _headDim, _ropeBase);
            float* p = GetFloatPtr(data);
            int headDim = _headDim;
            fixed (float* cosBase = _ropeCos, sinBase = _ropeSin)
            {
                nint pL = (nint)p, cL = (nint)cosBase, sL = (nint)sinBase;
                Parallel.For(0, seq, s =>
                {
                    float* cs = (float*)cL + (long)s * half, sn = (float*)sL + (long)s * half;
                    for (int hh = 0; hh < numHeads; hh++)
                    {
                        float* head = (float*)pL + (long)s * (numHeads * headDim) + (long)hh * headDim;
                        int i = 0;
                        for (; i + 8 <= half; i += 8)
                        {
                            var x1 = Vector256.Load(head + i);
                            var x2 = Vector256.Load(head + half + i);
                            var c = Vector256.Load(cs + i);
                            var si = Vector256.Load(sn + i);
                            Vector256.Store(x1 * c - x2 * si, head + i);
                            Vector256.Store(x2 * c + x1 * si, head + half + i);
                        }
                        for (; i < half; i++)
                        {
                            float x1 = head[i], x2 = head[half + i];
                            head[i] = x1 * cs[i] - x2 * sn[i];
                            head[half + i] = x2 * cs[i] + x1 * sn[i];
                        }
                    }
                });
            }
            InvalidateTensorDeviceCache(data);
        }

        /// <summary>cos/sin tables [seq, headDim/2] of the interleaved M-RoPE angles.</summary>
        internal static (float[] Cos, float[] Sin) BuildRopeTables(int[] pos, int seq, int headDim, float ropeBase)
        {
            int half = headDim / 2;
            var freq = new float[half];
            var axis = new int[half];
            for (int i = 0; i < half; i++)
            {
                freq[i] = (float)Math.Pow(ropeBase, -2.0 * i / headDim);
                axis[i] = InterleavedRopeAxis(i);
            }
            var cos = new float[(long)seq * half];
            var sin = new float[(long)seq * half];
            Parallel.For(0, seq, s =>
            {
                for (int i = 0; i < half; i++)
                {
                    float ang = pos[axis[i] * seq + s] * freq[i];
                    cos[(long)s * half + i] = MathF.Cos(ang);
                    sin[(long)s * half + i] = MathF.Sin(ang);
                }
            });
            return (cos, sin);
        }

        // The managed attention below reads raw host pointers, so it is limited to the
        // pure-C# backend; every other backend keeps the tensor-op sequence.
        private bool UseManagedAttention => _backend == BackendType.Cpu &&
            Environment.GetEnvironmentVariable("TS_QWEN_TE_CPU_ATTN") != "0";

        /// <summary>
        /// Causal grouped-query attention over the row-major projections: q [seq, H*D],
        /// k and v [seq, KV*D] -> out [seq, H*D]. Query head h reads KV head h / (H/KV) by
        /// index (the RepeatInterleave expansion, without the copy), and each query row
        /// stops at its own position, so no score matrix and no mask are materialized.
        /// Same formula as scale*QK^T + causal -inf mask + softmax + PV.
        /// </summary>
        private unsafe void CausalGqaAttention(Tensor q, Tensor k, Tensor v, Tensor output, int seq, float scale) =>
            CausalGqaAttention(GetFloatPtr(q), GetFloatPtr(k), GetFloatPtr(v), GetFloatPtr(output),
                seq, _numHeads, _numKVHeads, _headDim, scale);

        internal static unsafe void CausalGqaAttention(float* q, float* k, float* v, float* output,
            int seq, int heads, int kvHeads, int dim, float scale)
        {
            int group = heads / kvHeads;
            long qStride = (long)heads * dim, kvStride = (long)kvHeads * dim;
            nint qL = (nint)q, kL = (nint)k, vL = (nint)v, oL = (nint)output;
            const int QueryBlock = 16;
            int blocks = (seq + QueryBlock - 1) / QueryBlock;
            CpuWorkerPool.Shared.For(heads * blocks, task =>
            {
                int h = task / blocks, b = task - h * blocks;
                int kvh = h / group;
                float* kBase = (float*)kL + (long)kvh * dim, vBase = (float*)vL + (long)kvh * dim;
                float[] rented = seq > 4096 ? System.Buffers.ArrayPool<float>.Shared.Rent(seq) : null;
                Span<float> scores = rented != null ? rented.AsSpan(0, seq) : stackalloc float[seq];
                Span<float> acc = stackalloc float[dim];
                fixed (float* sp = scores, ap = acc)
                {
                    for (int i = b * QueryBlock; i < Math.Min(seq, (b + 1) * QueryBlock); i++)
                    {
                        float* qi = (float*)qL + i * qStride + (long)h * dim;
                        float mx = float.NegativeInfinity;
                        for (int j = 0; j <= i; j++)
                        {
                            float sc = Dot(qi, kBase + j * kvStride, dim) * scale;
                            sp[j] = sc;
                            if (sc > mx) mx = sc;
                        }
                        float sum = 0f;
                        for (int j = 0; j <= i; j++) { float e = MathF.Exp(sp[j] - mx); sp[j] = e; sum += e; }
                        float inv = 1f / sum;
                        new Span<float>(ap, dim).Clear();
                        for (int j = 0; j <= i; j++) Axpy(sp[j] * inv, vBase + j * kvStride, ap, dim);
                        float* oi = (float*)oL + i * qStride + (long)h * dim;
                        Buffer.MemoryCopy(ap, oi, dim * sizeof(float), dim * sizeof(float));
                    }
                }
                if (rented != null) System.Buffers.ArrayPool<float>.Shared.Return(rented);
            });
        }

        private static unsafe float Dot(float* a, float* b, int n)
        {
            var acc = Vector256<float>.Zero;
            int i = 0;
            for (; i + 8 <= n; i += 8) acc += Vector256.Load(a + i) * Vector256.Load(b + i);
            float sum = Vector256.Sum(acc);
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }

        private static unsafe void Axpy(float alpha, float* x, float* y, int n)
        {
            var va = Vector256.Create(alpha);
            int i = 0;
            for (; i + 8 <= n; i += 8) Vector256.Store(Vector256.Load(y + i) + va * Vector256.Load(x + i), y + i);
            for (; i < n; i++) y[i] += alpha * x[i];
        }

        internal static int InterleavedRopeAxis(int frequency) =>
            frequency % 3 == 1 && frequency < 60 ? 1 : frequency % 3 == 2 && frequency < 60 ? 2 : 0;

        private unsafe void NormalizeHeads(Tensor x, string weight, int heads, int seq)
        {
            float* data = GetFloatPtr(x);
            float* gamma = GetFloatPtr(_weights[weight]);
            int headDim = _headDim;
            float eps = _eps;
            nint dL = (nint)data, gL = (nint)gamma;
            // One delegate per token (all its heads), not per head row.
            Parallel.For(0, seq, s =>
            {
                for (int hh = 0; hh < heads; hh++)
                {
                    float* p = (float*)dL + ((long)s * heads + hh) * headDim;
                    float* g = (float*)gL;
                    double ss = 0;
                    for (int d = 0; d < headDim; d++) ss += (double)p[d] * p[d];
                    float inv = 1f / MathF.Sqrt((float)(ss / headDim) + eps);
                    for (int d = 0; d < headDim; d++) p[d] *= inv * g[d];
                }
            });
            InvalidateTensorDeviceCache(x);
        }

        private unsafe Tensor LinearWithBias(Tensor input, string weightName, string biasName)
        {
            Tensor result = Linear(input, weightName);
            if (_weights.TryGetValue(biasName, out var bias))
            {
                int seq = (int)result.Sizes[0], outDim = (int)result.Sizes[1];
                float* rPtr = GetFloatPtr(result);
                float* bPtr = GetFloatPtr(bias);
                int dim = Math.Min(outDim, (int)bias.ElementCount());
                Parallel.For(0, seq, s =>
                {
                    float* row = rPtr + (long)s * outDim;
                    for (int d = 0; d < dim; d++) row[d] += bPtr[d];
                });
                InvalidateTensorDeviceCache(result);
            }
            return result;
        }

        // Pure-C# backend: every projection runs on the packed F32 GEMM, reading the quantized
        // weight through QuantRowsPanelSource (each tile dequantized once per forward, with
        // exact F32 activations). The generic path (ManagedQuantizedOps) instead quantizes the
        // activations to 8 bits and re-decodes every weight block for each activation row.
        // TS_QWEN_TE_CPU_GEMM=0 restores it, and the scalar SiLU with it, so that
        // TS_QWEN_TE_CPU_GEMM=0 TS_QWEN_TE_CPU_ATTN=0 reproduce the previous per-op path bit for
        // bit (the RoPE tables and the per-token QK norm are bit-identical to what they replaced).
        private static readonly bool CpuGemmOn = Environment.GetEnvironmentVariable("TS_QWEN_TE_CPU_GEMM") != "0";

        // Parity reference for the harness (QwenImageStagesBench text --f64-linear), never set in
        // production: every projection the packed GEMM would run is instead summed in double over
        // the exactly dequantized weights, so a whole forward can be compared with one whose
        // matmuls carry neither F32 accumulation nor 8-bit activation rounding.
        internal static bool ReferenceF64Linear;

        // TS_QWEN_TE_PROFILE=1: per-forward split of the per-op path (linear / attention / norms).
        private static readonly bool ProfileOn = Environment.GetEnvironmentVariable("TS_QWEN_TE_PROFILE") == "1";
        private long _profileLinear, _profileAttention, _profileNorm;

        private Tensor Linear(Tensor input, string weightName)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            Tensor result = LinearCore(input, weightName);
            _profileLinear += System.Diagnostics.Stopwatch.GetTimestamp() - start;
            return result;
        }

        private unsafe Tensor LinearCore(Tensor input, string weightName)
        {
            if (_backend != BackendType.Cpu || !CpuGemmOn || !input.IsContiguous() ||
                !_quantWeights.TryGetValue(weightName, out var qw) || !qw.HasHostData ||
                !QuantRowsPanelSource.Supports(qw.GgmlType, qw.Ne0) || input.Sizes[1] != qw.Ne0)
                return LinearForward(input, weightName);
            int seq = (int)input.Sizes[0], inDim = (int)qw.Ne0, outDim = (int)qw.Ne1;
            var result = new Tensor(_allocator, DType.Float32, seq, outDim);
            if (ReferenceF64Linear)
            {
                ReferenceLinearF64(GetFloatPtr(input), seq, inDim, qw, GetFloatPtr(result), outDim);
                return result;
            }
            var packedInput = CpuPackedGemm.PackA(GetFloatPtr(input), seq, inDim, inDim, 1, CpuPackedGemm.Isa);
            CpuPackedGemm.Gemm(packedInput, new QuantRowsPanelSource(qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1),
                outDim, GetFloatPtr(result), outDim);
            if (qw.Scale != 1.0f)
                Ops.Mul(result, result, qw.Scale);
            return result;
        }

        private static unsafe void ReferenceLinearF64(float* x, int seq, int inDim, QuantizedWeight qw, float* y, int outDim)
        {
            nint xL = (nint)x, yL = (nint)y, wL = qw.Data;
            long rowBytes = ManagedQuantizedOps.RowSize(qw.GgmlType, inDim);
            int type = qw.GgmlType;
            double scale = qw.Scale;
            Parallel.For(0, outDim, () => new float[inDim], (o, _, w) =>
            {
                fixed (float* wp = w)
                {
                    ManagedQuantizedOps.DequantizeRowToFloat32(type, wL + (nint)(o * rowBytes), wp, inDim);
                    for (int r = 0; r < seq; r++)
                        ((float*)yL)[(long)r * outDim + o] = (float)(DotF64((float*)xL + (long)r * inDim, wp, inDim) * scale);
                }
                return w;
            }, _ => { });
        }

        // float x float is exact in double, so only the (double) additions round.
        private static unsafe double DotF64(float* a, float* b, int n)
        {
            Vector256<double> s0 = default, s1 = default, s2 = default, s3 = default;
            int i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var (a0, a1) = Vector256.Widen(Vector256.Load(a + i));
                var (b0, b1) = Vector256.Widen(Vector256.Load(b + i));
                var (a2, a3) = Vector256.Widen(Vector256.Load(a + i + 8));
                var (b2, b3) = Vector256.Widen(Vector256.Load(b + i + 8));
                s0 += a0 * b0; s1 += a1 * b1; s2 += a2 * b2; s3 += a3 * b3;
            }
            double acc = Vector256.Sum((s0 + s1) + (s2 + s3));
            for (; i < n; i++) acc += (double)a[i] * b[i];
            return acc;
        }

        // gate = silu(gate) * up. Chunked (one delegate per 16K values, not per value) and, with
        // the packed GEMM on the pure-C# backend, vectorized: the vectorized exp is within an ulp
        // or two of MathF.Exp. Other backends (and TS_QWEN_TE_CPU_GEMM=0) keep the scalar formula.
        private unsafe void SiluMulInPlace(Tensor gate, Tensor up)
        {
            int n = (int)gate.ElementCount();
            nint gL = (nint)GetFloatPtr(gate), uL = (nint)GetFloatPtr(up);
            bool vectorized = _backend == BackendType.Cpu && CpuGemmOn;
            const int Chunk = 16 * 1024;
            Parallel.For(0, (n + Chunk - 1) / Chunk, c =>
            {
                float* g = (float*)gL, u = (float*)uL;
                int i = c * Chunk, end = Math.Min(n, i + Chunk);
                if (vectorized)
                    for (; i + 8 <= end; i += 8)
                    {
                        var x = Vector256.Load(g + i);
                        Vector256.Store(x / (Vector256<float>.One + Vector256.Exp(-x)) * Vector256.Load(u + i), g + i);
                    }
                for (; i < end; i++)
                {
                    float x = g[i];
                    g[i] = (x / (1f + MathF.Exp(-x))) * u[i];
                }
            });
            InvalidateTensorDeviceCache(gate);
        }

        private unsafe float[] TensorToHostFloat(Tensor t, long count)
        {
            var dst = new float[count];
            float* p = GetFloatPtr(t);
            fixed (float* d = dst)
                Buffer.MemoryCopy(p, d, count * sizeof(float), count * sizeof(float));
            return dst;
        }

        // Diagnostic-only snapshots. The native path writes matching planar
        // layouts; retaining intermediate outputs there can inhibit fusion.
        private void TraceTensor(int layer, string stage, Tensor tensor)
        {
            if (string.IsNullOrEmpty(TraceDirectory) || layer != TraceLayer) return;
            float[] values = TensorToHostFloat(tensor, tensor.ElementCount());
            byte[] bytes = new byte[values.Length * sizeof(float)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            System.IO.File.WriteAllBytes(System.IO.Path.Combine(TraceDirectory, $"unfused.L{layer:D2}.{stage}.f32"), bytes);
        }

        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("Use EncodeHidden().");
        protected override void ResetKVCacheCore() { }

        // ---- fused whole-trunk path (TSGgml_QwenTeTrunk) --------------------------------
        // The per-op loop above pays ~10 device<->host round-trips per layer (M-RoPE, head
        // norms, SiLU as host loops) x 36 layers. The fused path assembles the embeddings and
        // rotate-half M-RoPE tables on the host, then runs the whole causal GQA trunk as one
        // device graph (weights resident by GGUF mmap ptr). TS_QWEN_TE_FUSED=0 disables.
        private static readonly bool FusedTrunkOn =
            Environment.GetEnvironmentVariable("TS_QWEN_TE_FUSED") != "0";
        private QwenTeLayerW[] _fusedLayers;
        private readonly System.Collections.Generic.List<IntPtr> _fusedAllocs = new();
        private bool _fusedFailed;

        private unsafe bool TryFusedEncode(int[] tokens, ImageCond[] imgs, out float[] result)
        {
            result = null;
            if (!FusedTrunkOn || !IsGgmlBackend || _fusedFailed) return false;
            if (_fusedLayers == null && !BuildFusedLayers()) { _fusedFailed = true; return false; }

            int seq = tokens.Length, H = Config.HiddenSize, hd = _headDim, half = hd / 2;

            // input embeddings (host) + vision-embed injection
            Tensor emb = Embedding(tokens);
            float[] x = TensorToHostFloat(emb, (long)seq * H);
            emb.Dispose();
            if (imgs != null)
                foreach (var img in imgs)
                    for (int i = 0; i < img.Count; i++)
                        Array.Copy(img.Embeds, (long)i * H, x, (long)(img.Start + i) * H, H);

            // rotate-half M-RoPE tables [seq, head_dim] (duplicated halves), from the same
            // 3D positions/axes as ApplyMRoPE.
            var cos = new float[(long)seq * hd];
            var sin = new float[(long)seq * hd];
            int[] pos = _mropePos;
            Parallel.For(0, seq, s =>
            {
                long b = (long)s * hd;
                for (int i = 0; i < half; i++)
                {
                    float freq = (float)Math.Pow(_ropeBase, -2.0 * i / hd);
                    float ang = pos[InterleavedRopeAxis(i) * seq + s] * freq;
                    float c = MathF.Cos(ang), sn = MathF.Sin(ang);
                    cos[b + i] = c; cos[b + half + i] = c;
                    sin[b + i] = sn; sin[b + half + i] = sn;
                }
            });

            int deepCount = imgs != null ? 3 : 0;
            var deep = new float[(long)deepCount * seq * H];
            if (imgs != null)
                foreach (var img in imgs)
                    for (int l = 0; l < Math.Min(deepCount, img.DeepStack?.Length ?? 0); l++)
                        Array.Copy(img.DeepStack[l], 0, deep, ((long)l * seq + img.Start) * H, (long)img.Count * H);
            var outArr = new float[(long)seq * H];
            bool ok;
            fixed (float* xp = x, cp = cos, sp = sin, op = outArr, dp = deep)
            fixed (QwenTeLayerW* lp = _fusedLayers)
            {
                var a = new QwenTeTrunkArgs
                {
                    X = (IntPtr)xp, Out = (IntPtr)op, CosF = (IntPtr)cp, SinF = (IntPtr)sp,
                    Layers = (IntPtr)lp, NumLayers = _numLayers,
                    StructBytes = System.Runtime.InteropServices.Marshal.SizeOf<QwenTeTrunkArgs>(),
                    Hidden = H, Heads = _numHeads, KvHeads = _numKVHeads, HeadDim = hd, Seq = seq,
                    Eps = Config.Eps, DeepStack = (IntPtr)dp, DeepStackCount = deepCount,
                };
                ok = GgmlBasicOps.TryQwenTeTrunk(in a);
            }
            if (!ok)
            {
                // Surface WHY: "op unsupported by backend" (a backend gap worth
                // fixing) reads very differently from an allocation failure, and
                // the per-op fallback is several times slower.
                Console.WriteLine("  [te-fused] trunk kernel unavailable (" +
                    GgmlBasicOps.LastNativeError("no native error") + "); using the per-op path.");
                _fusedFailed = true;
                return false;
            }
            result = outArr;
            return true;
        }

        private unsafe bool BuildFusedLayers()
        {
            try
            {
                QImgAttnW W(string name, string biasName)
                {
                    var info = _gguf.Tensors[name];
                    _gguf.TryGetTensorDataPointer(info, out IntPtr p);
                    var w = new QImgAttnW
                    {
                        W = p,
                        Type = (int)info.Type,
                        Ne0 = (long)info.Shape[0],
                        Ne1 = info.Shape.Length > 1 ? (long)info.Shape[1] : 1,
                        Bytes = _gguf.GetTensorByteCount(info),
                    };
                    if (biasName != null && _gguf.Tensors.ContainsKey(biasName))
                        w.B = F32Stable(biasName);
                    return w;
                }

                var layers = new QwenTeLayerW[_numLayers];
                for (int l = 0; l < _numLayers; l++)
                {
                    string p = $"blk.{l}";
                    layers[l] = new QwenTeLayerW
                    {
                        Ln1 = F32Stable($"{p}.attn_norm.weight"),
                        Ln2 = F32Stable($"{p}.ffn_norm.weight"),
                        Q = W($"{p}.attn_q.weight", $"{p}.attn_q.bias"),
                        K = W($"{p}.attn_k.weight", $"{p}.attn_k.bias"),
                        V = W($"{p}.attn_v.weight", $"{p}.attn_v.bias"),
                        O = W($"{p}.attn_output.weight", null),
                        Gate = W($"{p}.ffn_gate.weight", null),
                        Up = W($"{p}.ffn_up.weight", null),
                        Down = W($"{p}.ffn_down.weight", null),
                        QNorm = F32Stable($"{p}.attn_q_norm.weight"),
                        KNorm = F32Stable($"{p}.attn_k_norm.weight"),
                    };
                }
                _fusedLayers = layers;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [te-fused] layer table build failed ({ex.Message}); using the per-op path.");
                return false;
            }
        }

        // Stable unmanaged F32 copy of a (small) GGUF tensor — norm weights / biases must
        // outlive the call and be F32 regardless of on-disk type.
        private unsafe IntPtr F32Stable(string name)
        {
            var info = _gguf.Tensors[name];
            long n = info.NumElements;
            var host = new float[n];
            byte[] raw = _gguf.ReadTensorData(info);
            NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, host, 0, n);
            IntPtr p = System.Runtime.InteropServices.Marshal.AllocHGlobal((IntPtr)(n * sizeof(float)));
            System.Runtime.InteropServices.Marshal.Copy(host, 0, p, (int)n);
            _fusedAllocs.Add(p);
            return p;
        }

        public override void Dispose()
        {
            base.Dispose();
            foreach (var p in _fusedAllocs) System.Runtime.InteropServices.Marshal.FreeHGlobal(p);
            _fusedAllocs.Clear();
            _fusedLayers = null;
        }
    }
}
