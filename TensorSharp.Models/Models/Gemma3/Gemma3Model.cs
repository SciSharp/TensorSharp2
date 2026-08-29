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
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Gemma 3 model architecture.
    /// Key features:
    /// - Alternating sliding-window (local) and full causal (global) attention every 6 layers
    /// - NeoX-style RoPE with different bases for local/global layers
    /// - GELU activation in MLP (GeGLU: GELU(Gate) * Up)
    /// - 4 RMSNorms per layer: attn_norm, post_attention_norm, ffn_norm, post_ffw_norm
    /// - QK-norm (per-head RMSNorm on Q and K)
    /// - Embedding scaling by sqrt(hidden_size)
    /// - Tied token embedding / output weights
    /// - Final logit softcapping (model-dependent)
    /// </summary>
    public partial class Gemma3Model : ModelBase
    {
        private const int GlobalCacheInterval = 6;

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private float[] _ropeFreqsLocal;
        private float[] _ropeFreqsGlobal;
        private int _slidingWindow;
        private int[] _cachedSWAMaskWidths;
        private int _cachedSWAMaskQueryLen;
        private int _cachedSWAMaskStartPos = -1;
        private float _ropeLocalBase;
        private float _ropeGlobalBase;
        private float _ropeScale;
        private float _finalLogitSoftcap;
        private int _attnKeyLen;
        private int _attnValLen;
        private bool _hasTiedOutput;

        private Gemma3VisionEncoder _visionEncoder;
        private List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        public Gemma3Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null) : base(ggufPath, backend, tpDegree, tpGroup)
        {
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") };
            ParseBaseConfig();

            _attnKeyLen = Config.KeyLength > 0 ? Config.KeyLength : 256;
            _attnValLen = Config.ValueLength > 0 ? Config.ValueLength : 256;
            _slidingWindow = (int)_gguf.GetUint32($"{Config.Architecture}.attention.sliding_window", 1024);
            Config.SlidingWindow = _slidingWindow;
            _ropeLocalBase = _gguf.GetFloat32($"{Config.Architecture}.rope.local.freq_base", 10000f);
            _ropeGlobalBase = Config.RopeBase;
            _ropeScale = Config.RopeScale;
            _finalLogitSoftcap = _gguf.GetFloat32($"{Config.Architecture}.final_logit_softcapping", 0f);

            if (Config.NumLayers == 34)
                _ropeScale = 8.0f;

            Console.WriteLine($"Model: {Config.Architecture}, Layers={Config.NumLayers}, " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"KeyLen={_attnKeyLen}, ValLen={_attnValLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE global={_ropeGlobalBase} local={_ropeLocalBase} scale={_ropeScale}");
            Console.WriteLine($"Sliding window={_slidingWindow}, Softcap={_finalLogitSoftcap}");

            int globalCount = 0;
            for (int i = 0; i < Config.NumLayers; i++)
                if (IsGlobalLayer(i)) globalCount++;
            Console.WriteLine($"Layer types: {globalCount} global (causal), {Config.NumLayers - globalCount} local (SWA)");

            ParseTokenizer();
            LoadWeights();

            _hasTiedOutput = !_weights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("output.weight");
            if (_hasTiedOutput)
                Console.WriteLine("  Output tied to token_embd.weight");

            FuseGateUpWeights();

            if (IsTensorParallel)
            {
                ShardGemma3WeightsForTP();
                PrepareCudaQuantizedWeightsForInferenceTP();
            }
            else
            {
                PrepareCudaQuantizedWeightsForInference();
            }

            PrecomputeRoPE();

            int maxContextLength = ResolveConfiguredContextLength();
            if (IsTensorParallel)
            {
                int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
                if (initialCacheLength < maxContextLength)
                    Console.WriteLine($"Initial {_backend} TP KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
                InitTpKVCache(initialCacheLength, maxContextLength);
            }
            else
            {
                InitKVCache(maxContextLength);
            }
        }

        private bool IsGlobalLayer(int layer) => (layer + 1) % GlobalCacheInterval == 0;

        private void PrecomputeRoPE()
        {
            int halfDim = _attnKeyLen / 2;
            _ropeFreqsLocal = new float[halfDim];
            _ropeFreqsGlobal = new float[halfDim];

            for (int i = 0; i < halfDim; i++)
            {
                double freqLocal = 1.0 / Math.Pow(_ropeLocalBase, 2.0 * i / _attnKeyLen);
                _ropeFreqsLocal[i] = (float)freqLocal;

                double freqGlobal = 1.0 / Math.Pow(_ropeGlobalBase, 2.0 * i / _attnKeyLen);
                _ropeFreqsGlobal[i] = (float)(freqGlobal / _ropeScale);
            }
        }

        private void InitKVCache(int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, maxSeqLen, _attnKeyLen);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, maxSeqLen, _attnValLen);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
            }
        }

        protected override void ResetKVCacheCore()
        {
            _cacheSeqLen = 0;
            if (_kvCacheK != null)
            {
                foreach (var k in _kvCacheK)
                {
                    ResetCacheTensor(k);
                }
                foreach (var v in _kvCacheV)
                {
                    ResetCacheTensor(v);
                }
            }
            if (_tpKvCacheK != null)
                foreach (var layer in _tpKvCacheK)
                    foreach (var k in layer) ResetCacheTensor(k);
            if (_tpKvCacheV != null)
                foreach (var layer in _tpKvCacheV)
                    foreach (var v in layer) ResetCacheTensor(v);
        }

        protected override void TruncateKVCacheCore(int tokenCount)
        {
            base.TruncateKVCacheCore(tokenCount);
            if (_kvCacheK != null)
            {
                foreach (var k in _kvCacheK)
                    InvalidateTensorDeviceCache(k);
                foreach (var v in _kvCacheV)
                    InvalidateTensorDeviceCache(v);
            }
            if (_tpKvCacheK != null)
                foreach (var layer in _tpKvCacheK)
                    foreach (var k in layer) InvalidateTensorDeviceCache(k);
            if (_tpKvCacheV != null)
                foreach (var layer in _tpKvCacheV)
                    foreach (var v in layer) InvalidateTensorDeviceCache(v);
        }

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        public override string KVStateFingerprint =>
            $"gemma3|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}|kL={_attnKeyLen}|vL={_attnValLen}|swa={_slidingWindow}|dtype={_kvCacheDtype.ToShortString()}";

        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokenCount);

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            return KvBlockTransfer.Extract(
                _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                startToken, tokenCount, destination);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            if (!KvBlockTransfer.Inject(
                    _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                    destToken, tokenCount, source))
            {
                return false;
            }
            _cacheSeqLen = destToken + tokenCount;
            foreach (var k in _kvCacheK)
                InvalidateTensorDeviceCache(k);
            foreach (var v in _kvCacheV)
                InvalidateTensorDeviceCache(v);
            return true;
        }

        public void LoadVisionEncoder(string mmProjPath)
        {
            _visionEncoder = new Gemma3VisionEncoder(mmProjPath, _allocator);
            _visionEncoder.SetHostModel(this);
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public Gemma3VisionEncoder VisionEncoder => _visionEncoder;

        protected override float[] ForwardCore(int[] tokens)
        {
            if (IsTensorParallel)
                return ForwardTP(tokens);

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            ScaleEmbedding(hidden);

            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hidden, embeddings, position, startPos);
                    embeddings.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            for (int l = 0; l < Config.NumLayers; l++)
            {
                hidden = TransformerBlock(hidden, l, seqLen, startPos);
            }

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var lastRow = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(lastRow);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            t0 = Stopwatch.GetTimestamp();
            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
            lastHidden.Dispose();

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            t0 = Stopwatch.GetTimestamp();
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];

            unsafe
            {
                float* ptr = GetFloatPtr(logitsTensor);
                fixed (float* dst = _logitsBuffer)
                    Buffer.MemoryCopy(ptr, dst, Config.VocabSize * 4, Config.VocabSize * 4);
            }
            logitsTensor.Dispose();
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        /// <summary>
        /// Replace token embeddings at image placeholder positions with vision encoder output.
        /// The vision embeddings tensor has shape [numTokens, projDim] where projDim == hiddenSize.
        /// </summary>
        private unsafe void InjectVisionEmbeddings(Tensor hidden, Tensor visionEmbeddings, int insertPos, int startPos)
        {
            int numVisionTokens = (int)visionEmbeddings.Sizes[0];
            int dim = Config.HiddenSize;
            float* hPtr = GetFloatPtr(hidden);
            float* vPtr = GetFloatPtr(visionEmbeddings);

            for (int t = 0; t < numVisionTokens; t++)
            {
                float* dst = hPtr + (long)(insertPos + t) * dim;
                float* src = vPtr + (long)t * dim;
                Buffer.MemoryCopy(src, dst, dim * sizeof(float), dim * sizeof(float));
            }

            // insertPos is the offset within the current prefill chunk; startPos
            // (the cached sequence length) makes the absolute position explicit so
            // the log reads monotonically across chunked multimodal prefill.
            Console.WriteLine($"Injected {numVisionTokens} vision tokens at chunk-offset {insertPos} (absolute position {startPos + insertPos})");
        }

        private void ScaleEmbedding(Tensor hidden)
        {
            float scale = MathF.Sqrt(Config.HiddenSize);
            Ops.Mul(hidden, hidden, scale);
        }

        private void ApplyLogitSoftcap(Tensor logits)
        {
            float cap = _finalLogitSoftcap;
            Ops.Mul(logits, logits, 1f / cap);
            Ops.Tanh(logits, logits);
            Ops.Mul(logits, logits, cap);
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string prefix = $"blk.{layer}";

            // Pre-attention norm
            using var attnNormed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");

            // Self-attention
            using var attnOut = Attention(attnNormed, layer, prefix, seqLen, startPos);

            // Post-attention norm
            using var postAttnNormed = RMSNormOp(attnOut, $"{prefix}.post_attention_norm.weight");

            // Residual connection
            Ops.Add(postAttnNormed, postAttnNormed, hidden);
            hidden.Dispose();

            // Pre-FFN norm
            using var ffnNormed = RMSNormOp(postAttnNormed, $"{prefix}.ffn_norm.weight");

            // FFN (GeGLU with GELU activation)
            using var ffnOut = FFNGelu(ffnNormed, $"{prefix}.ffn_gate_up.weight",
                $"{prefix}.ffn_down.weight", seqLen);

            // Post-FFN norm
            using var postFfnNormed = RMSNormOp(ffnOut, $"{prefix}.post_ffw_norm.weight");

            // Residual connection
            var result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
            Ops.Add(result, postAttnNormed, postFfnNormed);
            return result;
        }

        private Tensor FFNGelu(Tensor input, string gateUpWeightName, string downWeightName, int seqLen)
        {
            Tensor gateUp = LinearForward(input, gateUpWeightName);
            int intermSize = Config.IntermediateSize;
            int halfDim = intermSize > 0 ? intermSize : (int)(gateUp.Sizes[1] / 2);

            Tensor gate, up;
            if (seqLen == 1)
            {
                gate = gateUp.Narrow(1, 0, halfDim);
                up = gateUp.Narrow(1, halfDim, halfDim);
            }
            else
            {
                using (var gView = gateUp.Narrow(1, 0, halfDim))
                    gate = Ops.NewContiguous(gView);
                using (var uView = gateUp.Narrow(1, halfDim, halfDim))
                    up = Ops.NewContiguous(uView);
            }
            gateUp.Dispose();

            Ops.GELUMul(gate, gate, up);
            up.Dispose();

            Tensor down = LinearForward(gate, downWeightName);
            gate.Dispose();
            return down;
        }

        private Tensor Attention(Tensor input, int layer, string prefix, int seqLen, int startPos)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool isGlobal = IsGlobalLayer(layer);

            Tensor q = LinearForward(input, $"{prefix}.attn_q.weight");
            Tensor k = LinearForward(input, $"{prefix}.attn_k.weight");
            Tensor v = LinearForward(input, $"{prefix}.attn_v.weight");

            // QK norm
            if (seqLen == 1)
            {
                RMSNormInPlace(q, _weights[$"{prefix}.attn_q_norm.weight"], Config.NumHeads, _attnKeyLen, Config.Eps);
                RMSNormInPlace(k, _weights[$"{prefix}.attn_k_norm.weight"], Config.NumKVHeads, _attnKeyLen, Config.Eps);
            }
            else
            {
                q = ApplyBatchRMSNorm(q, $"{prefix}.attn_q_norm.weight", Config.NumHeads, seqLen, _attnKeyLen);
                k = ApplyBatchRMSNorm(k, $"{prefix}.attn_k_norm.weight", Config.NumKVHeads, seqLen, _attnKeyLen);
            }

            // Apply NeoX-style RoPE
            if (seqLen == 1)
            {
                float[] freqs = isGlobal ? _ropeFreqsGlobal : _ropeFreqsLocal;
                ApplyNeoXRoPEDecode(q, Config.NumHeads, _attnKeyLen, startPos, freqs);
                ApplyNeoXRoPEDecode(k, Config.NumKVHeads, _attnKeyLen, startPos, freqs);
            }
            else
            {
                float ropeBase = isGlobal ? _ropeGlobalBase : _ropeLocalBase;
                float freqScale = isGlobal ? (1.0f / _ropeScale) : 1.0f;
                q = ApplyRoPEPrefill(q, Config.NumHeads, _attnKeyLen, seqLen, startPos, ropeBase, freqScale);
                k = ApplyRoPEPrefill(k, Config.NumKVHeads, _attnKeyLen, seqLen, startPos, ropeBase, freqScale);
            }

            // Q scaling
            float qScale = 1f / MathF.Sqrt(_attnKeyLen);
            ScaleTensor(q, qScale);

            int totalSeqLen = startPos + seqLen;
            Tensor result;

            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v,
                    Config.NumKVHeads, _attnKeyLen, startPos);

                int attendLen = isGlobal ? totalSeqLen : Math.Min(totalSeqLen, _slidingWindow);
                int attendStart = totalSeqLen - attendLen;

                result = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * _attnValLen);
                AttentionDecodeWithWindow(q, _kvCacheK[layer], _kvCacheV[layer], result,
                    Config.NumHeads, Config.NumKVHeads, _attnKeyLen, _attnValLen,
                    attendStart, totalSeqLen, 1f);
            }
            else
            {
                Tensor qHeads = ReshapeToHeads(q, Config.NumHeads, seqLen, _attnKeyLen);
                Tensor kHeads = ReshapeToHeads(k, Config.NumKVHeads, seqLen, _attnKeyLen);
                Tensor vHeads = ReshapeToHeads(v, Config.NumKVHeads, seqLen, _attnValLen);

                CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                kHeads.Dispose();
                vHeads.Dispose();

                int groupSize = Config.NumHeads / Config.NumKVHeads;
                Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
                Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

                using var kT = kExpanded.Transpose(1, 2);
                var scores = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, totalSeqLen);
                Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                qHeads.Dispose();
                kExpanded.Dispose();

                int windowSize = isGlobal ? 0 : _slidingWindow;
                ApplyCausalMask(scores, seqLen, totalSeqLen, windowSize);
                Ops.Softmax(scores, scores);

                var attnOut = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, _attnValLen);
                Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                scores.Dispose();
                vExpanded.Dispose();

                result = ReshapeFromHeads(attnOut, Config.NumHeads, seqLen, _attnValLen);
                attnOut.Dispose();
            }

            q.Dispose();
            k.Dispose();
            v.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            using (result)
            {
                return LinearForward(result, $"{prefix}.attn_output.weight");
            }
        }

        private Tensor ApplyBatchRMSNorm(Tensor data, string weightName, int numHeads, int seqLen, int headDim)
        {
            var alpha = _weights[weightName];
            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();
            Tensor flat = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return flat;
        }

        /// <summary>
        /// NeoX-style RoPE: pairs (x[j], x[j + d/2]) — first half with second half.
        /// GGML GGML_ROPE_TYPE_NEOX uses n_offset = n_dims/2.
        /// </summary>
        private unsafe void ApplyNeoXRoPEDecode(Tensor data, int numHeads, int headDim, int position, float[] freqs)
        {
            float* ptr = GetFloatPtr(data);
            int halfDim = headDim / 2;

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int j = 0; j < halfDim; j++)
                {
                    float angle = position * freqs[j];
                    float cos = MathF.Cos(angle);
                    float sin = MathF.Sin(angle);
                    float x0 = head[j];
                    float x1 = head[j + halfDim];
                    head[j] = x0 * cos - x1 * sin;
                    head[j + halfDim] = x0 * sin + x1 * cos;
                }
            }
        }

        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int headDim,
            int seqLen, int startPos, float ropeBase, float freqScale)
        {
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensorOn(data.Storage.Allocator, positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, 2, 0,
                ropeBase, freqScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private void ScaleTensor(Tensor t, float scale)
        {
            Ops.Mul(t, t, scale);
        }

        /// <summary>
        /// Attention decode with optional sliding window.
        /// attendStart..totalSeqLen-1 is the window of positions to attend to.
        /// </summary>
        private unsafe void AttentionDecodeWithWindow(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int attendStart, int totalSeqLen, float scale)
        {
            // The cache is only F32 on some backends. On the pure-C# `cpu` backend
            // (and anywhere --kv-cache-dtype asks for it) it is F16 or block
            // quantized, and walking it as a flat float buffer threw
            // "Requires a Float32 tensor, but found Float16" out of GetFloatPtr -
            // Gemma 3 could not run on the cpu backend at all. Dequantize the
            // active window into compact F32 and re-enter, which is what
            // ModelBase.AttentionDecodePureCS does for the non-windowed case. The
            // compact copy's Sizes[1] == totalSeqLen doubles as its row stride, so
            // the maxSeqLen read below stays correct.
            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                using Tensor kF32 = ExpandKVHeadsF16(kCache, 1, totalSeqLen);
                using Tensor vF32 = ExpandKVHeadsF16(vCache, 1, totalSeqLen);
                AttentionDecodeWithWindow(q, kF32, vF32, result, numHeads, numKVHeads,
                    keyDim, valDim, attendStart, totalSeqLen, scale);
                return;
            }
            if (IsBlockQuantCacheDType(kCache.ElementType) && vCache.ElementType == kCache.ElementType)
            {
                using Tensor kF32 = ExpandKVHeadsBlockQuant(kCache, 1, totalSeqLen);
                using Tensor vF32 = ExpandKVHeadsBlockQuant(vCache, 1, totalSeqLen);
                AttentionDecodeWithWindow(q, kF32, vF32, result, numHeads, numKVHeads,
                    keyDim, valDim, attendStart, totalSeqLen, scale);
                return;
            }

            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtr(kCache);
            float* vPtr = GetFloatPtr(vCache);
            float* rPtr = GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;
            int attendLen = totalSeqLen - attendStart;

            float* scores = stackalloc float[attendLen];

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * keyDim;
                int kvHead = h / groupSize;
                float* kHead = kPtr + kvHead * maxSeqLen * keyDim;
                float* vHead = vPtr + kvHead * maxSeqLen * valDim;

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < attendLen; t++)
                {
                    float s = VecDot(qHead, kHead + (attendStart + t) * keyDim, keyDim) * scale;
                    scores[t] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = 0;
                for (int t = 0; t < attendLen; t++)
                {
                    float e = MathF.Exp(scores[t] - maxScore);
                    scores[t] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;
                for (int t = 0; t < attendLen; t++)
                    scores[t] *= invSum;

                float* rHead = rPtr + h * valDim;
                VecZero(rHead, valDim);
                for (int t = 0; t < attendLen; t++)
                    VecScaleAdd(rHead, vHead + (attendStart + t) * valDim, scores[t], valDim);
            }
        }

        /// <summary>
        /// Apply causal mask to attention scores with optional sliding window.
        /// For sliding window, only positions within windowSize of the query are attended to.
        /// </summary>
        private unsafe void ApplyCausalMask(Tensor scores, int queryLen, int totalKVLen, int windowSize)
        {
            int startPos = totalKVLen - queryLen;

            // Causal mask via native op (upper triangle)
            Ops.AddCausalMask(scores, queryLen, startPos, float.NegativeInfinity);

            // Sliding window mask with cached per-row widths + vectorized fill
            if (windowSize > 0)
            {
                if (_cachedSWAMaskWidths == null ||
                    _cachedSWAMaskQueryLen != queryLen ||
                    _cachedSWAMaskStartPos != startPos)
                {
                    _cachedSWAMaskWidths = new int[queryLen];
                    _cachedSWAMaskQueryLen = queryLen;
                    _cachedSWAMaskStartPos = startPos;
                    for (int q = 0; q < queryLen; q++)
                        _cachedSWAMaskWidths[q] = Math.Max(0, startPos + q - windowSize + 1);
                }

                float* sPtr = GetFloatPtr(scores);
                int numHeads = (int)scores.Sizes[0];
                int rowStride = queryLen * totalKVLen;

                for (int h = 0; h < numHeads; h++)
                {
                    float* headScores = sPtr + h * rowStride;
                    for (int q = 0; q < queryLen; q++)
                    {
                        int width = _cachedSWAMaskWidths[q];
                        if (width > 0)
                            new Span<float>(headScores + q * totalKVLen, width).Fill(float.NegativeInfinity);
                    }
                }
                InvalidateTensorDeviceCache(scores);
            }
        }

        public override void Dispose()
        {
            _visionEncoder?.Dispose();
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings?.Dispose();
            _pendingVisionEmbeddingsList.Clear();
            if (_kvCacheK != null)
                foreach (var k in _kvCacheK) k?.Dispose();
            if (_kvCacheV != null)
                foreach (var v in _kvCacheV) v?.Dispose();
            if (_tpKvCacheK != null)
                foreach (var layer in _tpKvCacheK)
                    foreach (var t in layer) t?.Dispose();
            if (_tpKvCacheV != null)
                foreach (var layer in _tpKvCacheV)
                    foreach (var t in layer) t?.Dispose();
            base.Dispose();
        }
    }
}
