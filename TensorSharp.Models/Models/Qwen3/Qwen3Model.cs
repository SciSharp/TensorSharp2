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
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    public partial class Qwen3Model : ModelBase
    {
        // Bound the MLX lazy-graph depth across the per-layer dispatch loop.
        // Mirrors Qwen35's pattern; override via TS_MLX_EVAL_EVERY_N_LAYERS.
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();
        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 16;
        }

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;

        private string[][] _layerWeightNames;
        private int[] _decodeQPositions;
        private int[] _decodeKPositions;
        private float[] _ropeFreqs;

        // Qwen2 / Qwen2.5-VL ("qwen2vl") share Qwen3's block layout but swap one
        // detail: they add a bias to the Q/K/V projections and have no per-head
        // QK RMSNorm. Both are detected from the weights rather than hardcoded per
        // architecture, so a Qwen2-style GGUF loads through the same fast paths.
        private bool _hasQkvBias;
        private bool _hasQkNorm;

        private ModelDecodeArrays _modelDecodeArrays;
        private bool _canUseNativeLayerDecode;
        private bool _kvCacheHostDirty;

        public Qwen3Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            string arch = _gguf.GetString("general.architecture") ?? "qwen3";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            Config.NumKVHeads = (int)_gguf.GetUint32($"{arch}.attention.head_count_kv");

            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, eps={Config.Eps}");

            LoadWeights();
            _hasQkNorm = _weights.ContainsKey("blk.0.attn_q_norm.weight");
            _hasQkvBias = _weights.ContainsKey("blk.0.attn_q.bias");
            if (_hasQkvBias)
                FuseQKVBiases();
            FuseQKVWeights();
            FuseGateUpWeights();
            if (!_hasQkNorm || _hasQkvBias)
                Console.WriteLine($"  Attention variant: qkNorm={_hasQkNorm}, qkvBias={_hasQkvBias}");

            if (IsTensorParallel)
            {
                ShardQwen3WeightsForTP();
                PrepareCudaQuantizedWeightsForInferenceTP();
            }
            else
            {
                PrepareCudaQuantizedWeightsForInference();
            }

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");

            if (IsTensorParallel)
                InitTpKVCache(initialCacheLength, maxContextLength);
            else
                InitKVCache(initialCacheLength, maxContextLength);

            PrecomputeConstants();
            BuildModelDecodeArrays();
            DetermineNativeLayerDecodeAvailability();
        }

        // Concatenate the separate q/k/v bias vectors into one Q|K|V bias that
        // matches the layout of the fused attn_qkv weight, so the bias is a single
        // ggml_add on the fused projection output instead of three.
        private void FuseQKVBiases()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string qName = $"blk.{l}.attn_q.bias";
                string kName = $"blk.{l}.attn_k.bias";
                string vName = $"blk.{l}.attn_v.bias";

                if (!_weights.TryGetValue(qName, out var qb) ||
                    !_weights.TryGetValue(kName, out var kb) ||
                    !_weights.TryGetValue(vName, out var vb))
                {
                    _hasQkvBias = false;
                    return;
                }

                int qDim = (int)qb.ElementCount();
                int kDim = (int)kb.ElementCount();
                int vDim = (int)vb.ElementCount();
                var fused = new Tensor(_allocator, DType.Float32, qDim + kDim + vDim);
                using (var s0 = fused.Narrow(0, 0, qDim)) Ops.Copy(s0, qb.View(qDim));
                using (var s1 = fused.Narrow(0, qDim, kDim)) Ops.Copy(s1, kb.View(kDim));
                using (var s2 = fused.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, vb.View(vDim));

                _weights[$"blk.{l}.attn_qkv.bias"] = fused;
                _weights.Remove(qName); qb.Dispose();
                _weights.Remove(kName); kb.Dispose();
                _weights.Remove(vName); vb.Dispose();
            }
        }

        private unsafe void FuseQKVWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string qName = $"blk.{l}.attn_q.weight";
                string kName = $"blk.{l}.attn_k.weight";
                string vName = $"blk.{l}.attn_v.weight";
                string qkvName = $"blk.{l}.attn_qkv.weight";

                if (_quantWeights.TryGetValue(qName, out var qw) &&
                    _quantWeights.TryGetValue(kName, out var kw) &&
                    _quantWeights.TryGetValue(vName, out var vw) &&
                    qw.GgmlType == kw.GgmlType && kw.GgmlType == vw.GgmlType &&
                    qw.Ne0 == kw.Ne0 && kw.Ne0 == vw.Ne0)
                {
                    if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, qw, kw, vw))
                        continue;

                    _quantWeights[qkvName] = fusedWeight;
                    _quantWeights.Remove(qName); qw.Dispose();
                    _quantWeights.Remove(kName); kw.Dispose();
                    _quantWeights.Remove(vName); vw.Dispose();
                    fused++;
                }
                else if (_weights.TryGetValue(qName, out var qf) &&
                         _weights.TryGetValue(kName, out var kf) &&
                         _weights.TryGetValue(vName, out var vf))
                {
                    int qDim = (int)qf.Sizes[0], kDim = (int)kf.Sizes[0], vDim = (int)vf.Sizes[0];
                    int inDim = (int)qf.Sizes[1];
                    var fusedTensor = new Tensor(_allocator, DType.Float32, qDim + kDim + vDim, inDim);
                    using (var s0 = fusedTensor.Narrow(0, 0, qDim)) Ops.Copy(s0, qf);
                    using (var s1 = fusedTensor.Narrow(0, qDim, kDim)) Ops.Copy(s1, kf);
                    using (var s2 = fusedTensor.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, vf);
                    _weights[qkvName] = fusedTensor;
                    _weights.Remove(qName); qf.Dispose();
                    _weights.Remove(kName); kf.Dispose();
                    _weights.Remove(vName); vf.Dispose();
                    fused++;
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} QKV");
        }

        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            int headDim = Config.HeadDim;

            _layerWeightNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                _layerWeightNames[l] = new[]
                {
                    p + "attn_norm.weight",
                    p + "attn_qkv.weight",
                    p + "attn_q_norm.weight",
                    p + "attn_k_norm.weight",
                    p + "attn_output.weight",
                    p + "ffn_norm.weight",
                    p + "ffn_gate_up.weight",
                    p + "ffn_down.weight",
                    p + "attn_qkv.bias",
                };
            }

            _decodeQPositions = new int[Config.NumHeads];
            _decodeKPositions = new int[Config.NumKVHeads];

            int halfDim = headDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / headDim);
        }

        private int _kvCacheCapacity;

        private void InitKVCache(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, numKVHeads, initialSeqLen, headDim);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, numKVHeads, initialSeqLen, headDim);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                var newK = new Tensor(_allocator, kvDtype, numKVHeads, newCapacity, headDim);
                var newV = new Tensor(_allocator, kvDtype, numKVHeads, newCapacity, headDim);
                InitializeCacheTensor(newK);
                InitializeCacheTensor(newV);

                if (_cacheSeqLen > 0)
                {
                    using var srcK = _kvCacheK[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstK, srcK);

                    using var srcV = _kvCacheV[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstV = newV.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstV, srcV);
                }

                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded Qwen3 attention cache to {newCapacity} tokens.");
        }

        protected override void ResetKVCacheCore()
        {
            // Setting _cacheSeqLen = 0 is the functional reset. Under TP the non-TP
            // _kvCacheK/_kvCacheV arrays are null (TP uses _tpKvCacheK/_tpKvCacheV,
            // overwritten on the next forward), so guard the tensor loop against null.
            _cacheSeqLen = 0;
            _kvCacheHostDirty = false;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
            if (_kvCacheK == null) return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                ResetCacheTensor(_kvCacheK[l]);
                ResetCacheTensor(_kvCacheV[l]);
            }
        }

        protected override void TruncateKVCacheCore(int tokenCount)
        {
            EnsureKvCacheHostSynchronized();
            base.TruncateKVCacheCore(tokenCount);
            _kvCacheHostDirty = false;
            if (_kvCacheK == null) return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
        }

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        public override string KVStateFingerprint =>
            $"qwen3|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}|D={Config.HeadDim}|dtype={_kvCacheDtype.ToShortString()}";

        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokenCount);

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            EnsureKvCacheHostSynchronized();
            return KvBlockTransfer.Extract(
                _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                startToken, tokenCount, destination);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            EnsureCacheCapacity(destToken + tokenCount);
            EnsureKvCacheHostSynchronized();
            if (!KvBlockTransfer.Inject(
                    _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                    destToken, tokenCount, source))
            {
                return false;
            }
            _cacheSeqLen = destToken + tokenCount;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            _kvCacheHostDirty = false;
            return true;
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            if (IsTensorParallel)
                return ForwardTP(tokens);

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);
            bool useNativeModelDecode = seqLen == 1 && IsGgmlBackend && _modelDecodeArrays != null;
            bool useNativeDecode = seqLen == 1 && IsGgmlBackend && (_modelDecodeArrays != null || _canUseNativeLayerDecode);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            if (!useNativeDecode)
                EnsureKvCacheHostSynchronized();

            if (useNativeModelDecode)
            {
                long t0 = Stopwatch.GetTimestamp();
                NativeTransformerModelDecode(hidden, startPos);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                _kvCacheHostDirty = true;
            }
            else
            {
                for (int layer = 0; layer < Config.NumLayers; layer++)
                {
                    hidden = TransformerBlock(hidden, layer, seqLen, startPos);
                    if (_backend == BackendType.Mlx && (layer + 1) % MlxEvalEveryNLayers == 0
                        && layer + 1 != Config.NumLayers && hidden != null)
                    {
                        MlxFusedOps.TryAsyncEvaluate(hidden);
                    }
                }
            }

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

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

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // Chunk size for ForwardRefill: long prompts are processed in this-many-token
        // chunks so the per-layer attention-score allocation stays bounded.
        // Override with TS_PREFILL_CHUNK when tuning.
        private static int ResolvePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 2048;
        }

        protected override float[] ForwardRefillCore(int[] tokens)
        {
            if (tokens == null || tokens.Length <= 1)
                return ForwardCore(tokens);

            // The chunked prefill path (PrefillWithoutLogits) uses the non-TP
            // layer loop and non-sharded weights, which are unavailable under
            // tensor parallelism. Route through ForwardCore → ForwardTP instead.
            if (IsTensorParallel)
                return ForwardCore(tokens);

            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;

            // For short prompts stay on the single-pass Forward — the extra
            // PrefillWithoutLogits/Forward split is pure overhead.
            if (tokens.Length <= chunkSize)
                return ForwardCore(tokens);

            for (int pos = 0; pos < lastIdx; pos += chunkSize)
            {
                int chunkLen = Math.Min(chunkSize, lastIdx - pos);
                var chunk = new int[chunkLen];
                Array.Copy(tokens, pos, chunk, 0, chunkLen);
                PrefillWithoutLogits(chunk);
            }
            return ForwardCore(new[] { tokens[lastIdx] });
        }

        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            bool useNativeModelDecode = seqLen == 1 && IsGgmlBackend && _modelDecodeArrays != null;
            bool useNativeDecode = seqLen == 1 && IsGgmlBackend && (_modelDecodeArrays != null || _canUseNativeLayerDecode);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            if (!useNativeDecode)
                EnsureKvCacheHostSynchronized();

            if (useNativeModelDecode)
            {
                long t0 = Stopwatch.GetTimestamp();
                NativeTransformerModelDecode(hidden, startPos);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                _kvCacheHostDirty = true;
            }
            else
            {
                for (int layer = 0; layer < Config.NumLayers; layer++)
                {
                    hidden = TransformerBlock(hidden, layer, seqLen, startPos);
                    if (_backend == BackendType.Mlx && (layer + 1) % MlxEvalEveryNLayers == 0
                        && layer + 1 != Config.NumLayers && hidden != null)
                    {
                        MlxFusedOps.TryAsyncEvaluate(hidden);
                    }
                }
            }

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string[] wn = _layerWeightNames[layer];

            if (seqLen == 1 && IsGgmlBackend && _quantWeights.ContainsKey(wn[1]))
            {
                long t0 = Stopwatch.GetTimestamp();
                NativeTransformerLayerDecode(hidden, layer, wn, startPos);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                _kvCacheHostDirty = true;
                return hidden;
            }

            Tensor normed = RMSNormOp(hidden, wn[0]);
            Tensor attnOut = Attention(normed, layer, wn, seqLen, startPos);
            normed.Dispose();

            // Fused (hidden += attnOut; normed2 = RmsNorm(hidden, ffnNormW)).
            Tensor normed2 = null;
            if (_backend == BackendType.Mlx && _weights.TryGetValue(wn[5], out var ffnNormW))
            {
                normed2 = new Tensor(_allocator, DType.Float32, hidden.Sizes[0], hidden.Sizes[1]);
                if (!MlxFusedOps.TryAddRmsNorm(hidden, attnOut, ffnNormW, Config.Eps, normed2))
                {
                    normed2.Dispose();
                    normed2 = null;
                }
            }
            if (normed2 == null)
            {
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();

                // GGML fused dense SwiGLU FFN in one graph (legacy/per-seq path
                // used by the CLI and the server's per-seq fallback).
                if (TryFusedDenseSwiGLUFFNInto(hidden, wn[5], wn[6], wn[7]))
                    return hidden;

                normed2 = RMSNormOp(hidden, wn[5]);
            }
            else
            {
                attnOut.Dispose();
            }

            Tensor ffnOut = FFN(normed2, wn[6], wn[7], seqLen);
            normed2.Dispose();

            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();

            return hidden;
        }

        private Tensor Attention(Tensor input, int layer, string[] wn, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qDim = numHeads * headDim;
            int kDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            ProjectQkv(input, wn, layer, seqLen, qDim, kDim,
                out Tensor qTensor, out Tensor kTensor, out Tensor vTensor);

            if (_hasQkNorm)
            {
                qTensor = ApplyQKNormInPlace(qTensor, wn[2], numHeads, seqLen);
                kTensor = ApplyQKNormInPlace(kTensor, wn[3], numKVHeads, seqLen);
            }

            if (seqLen == 1)
            {
                ApplyRoPEDecodeInPlace(qTensor, numHeads, headDim, startPos);
                ApplyRoPEDecodeInPlace(kTensor, numKVHeads, headDim, startPos);
            }
            else
            {
                qTensor = ApplyRoPEInPlace(qTensor, numHeads, headDim, seqLen, startPos);
                kTensor = ApplyRoPEInPlace(kTensor, numKVHeads, headDim, seqLen, startPos);
            }

            long t0 = Stopwatch.GetTimestamp();

            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                    numKVHeads, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                var attnResult = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);

                // MLX path: keep K/V on device and run attention via mlx_fast_sdpa.
                // Avoids the per-layer device→host copy of the KV cache that
                // AttentionDecodePureCS triggers via GetHalfPointer/GetFloatPtr.
                bool attnOk = false;
                if (_backend == BackendType.Mlx)
                {
                    attnOk = MlxFusedOps.TryDecodeAttention(
                        attnResult, qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        numHeads, numKVHeads, headDim,
                        0, totalSeqLen, _kvCacheCapacity, false, scale);
                }
                if (!attnOk)
                {
                    AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        attnResult, numHeads, numKVHeads, headDim, totalSeqLen, scale);
                }
                qTensor.Dispose();

                _attnTicks += Stopwatch.GetTimestamp() - t0;

                Tensor decodeOut = LinearForward(attnResult, wn[4]);
                attnResult.Dispose();
                return decodeOut;
            }

            Tensor qHeads = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
            qTensor.Dispose();
            Tensor kHeads = ReshapeToHeads(kTensor, numKVHeads, seqLen, headDim);
            kTensor.Dispose();
            Tensor vHeads = ReshapeToHeads(vTensor, numKVHeads, seqLen, headDim);
            vTensor.Dispose();

            CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
            CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
            kHeads.Dispose();
            vHeads.Dispose();

            int groupSize = numHeads / numKVHeads;
            Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
            Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

            using var kT = kExpanded.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, numHeads, seqLen, totalSeqLen);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            qHeads.Dispose();
            kExpanded.Dispose();

            // Fused causal-mask + softmax on GPU. Replaces AddCausalMask + Softmax
            // (two separate ops) with one Metal kernel.
            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(
                    scores, sinks: null,
                    numHeads: numHeads, seqLen: seqLen, kvLen: totalSeqLen,
                    maskStartPos: startPos, slidingWindow: 0, scale: 1.0f);
            }
            else
            {
                Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                Ops.Softmax(scores, scores);
            }

            var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
            attnOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            Tensor output = LinearForward(flatOutput, wn[4]);
            flatOutput.Dispose();

            return output;
        }

        /// <summary>
        /// Q/K/V projection, from the fused attn_qkv weight when one exists and
        /// from the three separate weights when it does not.
        ///
        /// The unfused path is not dead code: <see cref="FuseQKVWeights"/> can only
        /// concatenate q/k/v when all three share a ggml type, and mixed-precision
        /// GGUFs routinely break that. Qwen2.5-VL-7B Q4_K_M, for instance, stores
        /// attn_v as Q6_K and attn_q/attn_k as Q4_K on half its layers, so only
        /// 14 of 28 layers fuse.
        /// </summary>
        private void ProjectQkv(Tensor input, string[] wn, int layer, int seqLen, int qDim, int kDim,
                                out Tensor qTensor, out Tensor kTensor, out Tensor vTensor)
        {
            Tensor qkvFused = LinearForward(input, wn[1]);
            if (qkvFused == null)
            {
                string p = $"blk.{layer}.";
                qTensor = LinearForward(input, p + "attn_q.weight");
                kTensor = LinearForward(input, p + "attn_k.weight");
                vTensor = LinearForward(input, p + "attn_v.weight");
                if (qTensor == null || kTensor == null || vTensor == null)
                    throw new InvalidOperationException(
                        $"Layer {layer} has neither a fused attn_qkv weight nor separate attn_q/k/v weights.");

                if (_hasQkvBias && _weights.TryGetValue(wn[8], out var splitBias))
                {
                    // The fused bias is laid out Q|K|V, so slice it the same way.
                    using (var qb = splitBias.Narrow(0, 0, qDim)) AddRowBiasInPlace(qTensor, qb, seqLen);
                    using (var kb = splitBias.Narrow(0, qDim, kDim)) AddRowBiasInPlace(kTensor, kb, seqLen);
                    using (var vb = splitBias.Narrow(0, qDim + kDim, kDim)) AddRowBiasInPlace(vTensor, vb, seqLen);
                }
                return;
            }

            if (_hasQkvBias && _weights.TryGetValue(wn[8], out var qkvBias))
                AddRowBiasInPlace(qkvFused, qkvBias, seqLen);

            if (seqLen == 1)
            {
                qTensor = qkvFused.Narrow(1, 0, qDim);
                kTensor = qkvFused.Narrow(1, qDim, kDim);
                vTensor = qkvFused.Narrow(1, qDim + kDim, kDim);
                qkvFused.Dispose();
                return;
            }

            using (var qView = qkvFused.Narrow(1, 0, qDim))
                qTensor = Ops.NewContiguous(qView);
            using (var kView = qkvFused.Narrow(1, qDim, kDim))
                kTensor = Ops.NewContiguous(kView);
            using (var vView = qkvFused.Narrow(1, qDim + kDim, kDim))
                vTensor = Ops.NewContiguous(vView);
            qkvFused.Dispose();
        }

        /// <summary>
        /// Adds a length-<c>cols</c> bias vector to every row of a [rows, cols]
        /// activation, in place. Used for the fused Q|K|V bias on Qwen2-style
        /// architectures, which Qwen3 does not have.
        /// </summary>
        private void AddRowBiasInPlace(Tensor data, Tensor bias, int seqLen)
        {
            int cols = (int)data.Sizes[data.Sizes.Length - 1];
            using var biasRow = bias.View(1, cols);
            if (seqLen == 1)
            {
                Ops.Add(data, data, biasRow);
                return;
            }

            using var expanded = biasRow.Expand(seqLen, cols);
            Ops.Add(data, data, expanded);
        }

        private Tensor ApplyQKNormInPlace(Tensor data, string weightName, int numHeads, int seqLen)
        {
            int headDim = Config.HeadDim;
            var alpha = _weights[weightName];

            if (seqLen == 1)
            {
                RMSNormInPlace(data, alpha, numHeads, headDim, Config.Eps);
                return data;
            }

            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();

            Tensor result = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return result;
        }

        private unsafe void ApplyRoPEDecodeInPlace(Tensor data, int numHeads, int headDim, int position)
        {
            int halfDim = headDim / 2;
            float[] freqs = _ropeFreqs;
            float* ptr = GetFloatPtr(data);

            float* cosTable = stackalloc float[halfDim];
            float* sinTable = stackalloc float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                float theta = position * freqs[i];
                cosTable[i] = MathF.Cos(theta);
                sinTable[i] = MathF.Sin(theta);
            }

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[i];
                    float x1 = head[i + halfDim];
                    head[i] = x0 * cosTable[i] - x1 * sinTable[i];
                    head[i + halfDim] = x0 * sinTable[i] + x1 * cosTable[i];
                }
            }
        }

        private Tensor ApplyRoPEInPlace(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
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
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();

            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        #region Native decode paths

        private unsafe void NativeTransformerLayerDecode(Tensor hidden, int layer, string[] wn, int startPos)
        {
            float* hiddenPtr = GetFloatPtr(hidden);
            int hiddenSize = Config.HiddenSize;

            var attnNormW = _weights[wn[0]];
            var qkvW = _quantWeights[wn[1]];
            var qNormW = _hasQkNorm ? _weights[wn[2]] : null;
            var kNormW = _hasQkNorm ? _weights[wn[3]] : null;
            var oW = _quantWeights[wn[4]];
            var ffnNormW = _weights[wn[5]];
            var guW = _quantWeights[wn[6]];
            var downW = _quantWeights[wn[7]];

            int maxSeqLen = (int)_kvCacheK[layer].Sizes[1];

            GgmlBasicOps.TransformerLayerDecode(
                (IntPtr)hiddenPtr, hiddenSize,
                (IntPtr)GetFloatPtr(attnNormW),
                qkvW.CacheKey, qkvW.GgmlType, qkvW.Ne0, qkvW.Ne1, qkvW.RawBytes,
                _hasQkvBias && _weights.TryGetValue(wn[8], out var qkvBiasW)
                    ? (IntPtr)GetFloatPtr(qkvBiasW) : IntPtr.Zero,
                qNormW != null ? (IntPtr)GetFloatPtr(qNormW) : IntPtr.Zero,
                kNormW != null ? (IntPtr)GetFloatPtr(kNormW) : IntPtr.Zero,
                Config.HeadDim,
                oW.CacheKey, oW.GgmlType, oW.Ne0, oW.Ne1, oW.RawBytes,
                (IntPtr)GetFloatPtr(ffnNormW),
                guW.CacheKey, guW.GgmlType, guW.Ne0, guW.Ne1, guW.RawBytes,
                downW.CacheKey, downW.GgmlType, downW.Ne0, downW.Ne1, downW.RawBytes,
                TensorComputePrimitives.GetStoragePointer(_kvCacheK[layer]),
                TensorComputePrimitives.GetStoragePointer(_kvCacheV[layer]),
                Config.NumHeads, Config.NumKVHeads,
                maxSeqLen, startPos,
                Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                Config.IntermediateSize, 2,
                _kvCacheDtype.GgmlType());
        }

        private class ModelDecodeArrays
        {
            public IntPtr[] AttnNorm, Qkv, QNorm, KNorm, O, FfnNorm, Gu, Down, KCache, VCache;
            // Null when the architecture has no QKV bias (Qwen3); the native
            // kernel treats a null array, or a null entry, as "no bias".
            public IntPtr[] QkvBias;
            // Per-layer split Q/K/V, used for layers where Qkv[l] is Zero because
            // the three weights have different ggml types and cannot be
            // concatenated. SplitType/SplitBytes are flattened [q,k,v] per layer.
            public IntPtr[] Q, K, V;
            public int[] SplitType;
            public long[] SplitBytes;
            // Per-layer type/size for each weight class. A *_K_M quant mixes types
            // ACROSS layers (Qwen2.5-VL-7B Q4_K_M: ffn_down is Q6_K on 14 of 28
            // layers), so one scalar per weight class is not enough.
            public int[] QkvTypes, OTypes, GuTypes, DownTypes;
            public long[] QkvBytesPerLayer, OBytesPerLayer, GuBytesPerLayer, DownBytesPerLayer;
            public int QkvType, OType, GuType, DownType;
            public long QkvNe0, QkvNe1, QkvBytes;
            public long ONe0, ONe1, OBytes;
            public long GuNe0, GuNe1, GuBytes;
            public long DownNe0, DownNe1, DownBytes;
        }

        private unsafe void BuildModelDecodeArrays()
        {
            int numLayers = Config.NumLayers;
            if (!IsGgmlBackend) return;

            // A layer contributes either a fused attn_qkv weight or three separate
            // q/k/v weights; if any layer has neither, the whole-model graph is
            // not buildable and we fall back to the per-layer path.
            for (int l = 0; l < numLayers; l++)
            {
                string[] wnl = _layerWeightNames[l];
                if (_quantWeights.ContainsKey(wnl[1])) continue;
                if (!_quantWeights.ContainsKey($"blk.{l}.attn_q.weight") ||
                    !_quantWeights.ContainsKey($"blk.{l}.attn_k.weight") ||
                    !_quantWeights.ContainsKey($"blk.{l}.attn_v.weight"))
                    return;
            }

            string[] wn0 = _layerWeightNames[0];

            var arr = new ModelDecodeArrays();
            arr.AttnNorm = new IntPtr[numLayers];
            arr.Qkv = new IntPtr[numLayers];
            arr.QNorm = _hasQkNorm ? new IntPtr[numLayers] : null;
            arr.KNorm = _hasQkNorm ? new IntPtr[numLayers] : null;
            arr.QkvBias = _hasQkvBias ? new IntPtr[numLayers] : null;
            arr.QkvTypes = new int[numLayers]; arr.QkvBytesPerLayer = new long[numLayers];
            arr.OTypes = new int[numLayers]; arr.OBytesPerLayer = new long[numLayers];
            arr.GuTypes = new int[numLayers]; arr.GuBytesPerLayer = new long[numLayers];
            arr.DownTypes = new int[numLayers]; arr.DownBytesPerLayer = new long[numLayers];
            arr.O = new IntPtr[numLayers];
            arr.FfnNorm = new IntPtr[numLayers];
            arr.Gu = new IntPtr[numLayers];
            arr.Down = new IntPtr[numLayers];
            arr.KCache = new IntPtr[numLayers];
            arr.VCache = new IntPtr[numLayers];

            // Shapes are uniform across layers; only the quantization TYPE varies,
            // and that is carried per layer below. The fused-QKV shape still has to
            // be read from a layer that actually has a fused weight - layer 0 may
            // well be split.
            QuantizedWeight qkvHeader = null;
            for (int l = 0; l < numLayers && qkvHeader == null; l++)
                _quantWeights.TryGetValue(_layerWeightNames[l][1], out qkvHeader);

            if (qkvHeader != null)
            {
                arr.QkvType = qkvHeader.GgmlType; arr.QkvNe0 = qkvHeader.Ne0;
                arr.QkvNe1 = qkvHeader.Ne1; arr.QkvBytes = qkvHeader.RawBytes;
            }
            else
            {
                // Every layer is split. QkvNe0 (the input dimension) is still read
                // by the native kernel to size the split Q/K/V tensors.
                arr.QkvType = 0;
                arr.QkvNe0 = Config.HiddenSize;
                arr.QkvNe1 = (Config.NumHeads + 2 * Config.NumKVHeads) * (long)Config.HeadDim;
                arr.QkvBytes = 0;
            }
            var o0 = _quantWeights[wn0[4]];
            arr.OType = o0.GgmlType; arr.ONe0 = o0.Ne0; arr.ONe1 = o0.Ne1; arr.OBytes = o0.RawBytes;
            var gu0 = _quantWeights[wn0[6]];
            arr.GuType = gu0.GgmlType; arr.GuNe0 = gu0.Ne0; arr.GuNe1 = gu0.Ne1; arr.GuBytes = gu0.RawBytes;
            var down0 = _quantWeights[wn0[7]];
            arr.DownType = down0.GgmlType; arr.DownNe0 = down0.Ne0; arr.DownNe1 = down0.Ne1; arr.DownBytes = down0.RawBytes;

            for (int l = 0; l < numLayers; l++)
            {
                string[] wn = _layerWeightNames[l];
                arr.AttnNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[0]]);
                if (_quantWeights.TryGetValue(wn[1], out var qkvL))
                {
                    arr.Qkv[l] = qkvL.CacheKey;
                    arr.QkvTypes[l] = qkvL.GgmlType;
                    arr.QkvBytesPerLayer[l] = qkvL.RawBytes;
                }
                else
                {
                    arr.Q ??= new IntPtr[numLayers];
                    arr.K ??= new IntPtr[numLayers];
                    arr.V ??= new IntPtr[numLayers];
                    arr.SplitType ??= new int[3 * numLayers];
                    arr.SplitBytes ??= new long[3 * numLayers];

                    var qW = _quantWeights[$"blk.{l}.attn_q.weight"];
                    var kW = _quantWeights[$"blk.{l}.attn_k.weight"];
                    var vW = _quantWeights[$"blk.{l}.attn_v.weight"];
                    arr.Q[l] = qW.CacheKey; arr.K[l] = kW.CacheKey; arr.V[l] = vW.CacheKey;
                    arr.SplitType[3 * l + 0] = qW.GgmlType;
                    arr.SplitType[3 * l + 1] = kW.GgmlType;
                    arr.SplitType[3 * l + 2] = vW.GgmlType;
                    arr.SplitBytes[3 * l + 0] = qW.RawBytes;
                    arr.SplitBytes[3 * l + 1] = kW.RawBytes;
                    arr.SplitBytes[3 * l + 2] = vW.RawBytes;
                }
                if (arr.QNorm != null)
                {
                    arr.QNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[2]]);
                    arr.KNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[3]]);
                }
                if (arr.QkvBias != null)
                    arr.QkvBias[l] = (IntPtr)GetFloatPtr(_weights[wn[8]]);
                var oL = _quantWeights[wn[4]];
                arr.O[l] = oL.CacheKey;
                arr.OTypes[l] = oL.GgmlType; arr.OBytesPerLayer[l] = oL.RawBytes;
                arr.FfnNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[5]]);
                var guL = _quantWeights[wn[6]];
                arr.Gu[l] = guL.CacheKey;
                arr.GuTypes[l] = guL.GgmlType; arr.GuBytesPerLayer[l] = guL.RawBytes;
                var downL = _quantWeights[wn[7]];
                arr.Down[l] = downL.CacheKey;
                arr.DownTypes[l] = downL.GgmlType; arr.DownBytesPerLayer[l] = downL.RawBytes;
                arr.KCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]);
                arr.VCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]);
            }

            // Escape hatch: TS_QWEN3_MODEL_DECODE=0 runs the per-layer decode path
            // instead of the single whole-model graph, for bisecting a suspected
            // kernel bug against the (slower) per-layer and managed paths.
            if (string.Equals(Environment.GetEnvironmentVariable("TS_QWEN3_MODEL_DECODE"), "0", StringComparison.Ordinal))
                return;

            _modelDecodeArrays = arr;
        }

        private void DetermineNativeLayerDecodeAvailability()
        {
            _canUseNativeLayerDecode = IsGgmlBackend;
            if (!_canUseNativeLayerDecode || _layerWeightNames == null)
                return;

            for (int l = 0; l < Config.NumLayers; l++)
            {
                string[] wn = _layerWeightNames[l];
                if (!_quantWeights.ContainsKey(wn[1]) ||
                    !_quantWeights.ContainsKey(wn[4]) ||
                    !_quantWeights.ContainsKey(wn[6]) ||
                    !_quantWeights.ContainsKey(wn[7]))
                {
                    _canUseNativeLayerDecode = false;
                    return;
                }
            }
        }

        private void EnsureKvCacheHostSynchronized()
        {
            if (!_kvCacheHostDirty || !IsGgmlBackend)
                return;

            var seen = new HashSet<Storage>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvCacheK[l] != null && seen.Add(_kvCacheK[l].Storage))
                    SyncTensorHostCache(_kvCacheK[l]);
                if (_kvCacheV[l] != null && seen.Add(_kvCacheV[l].Storage))
                    SyncTensorHostCache(_kvCacheV[l]);
            }

            _kvCacheHostDirty = false;
        }

        private unsafe void NativeTransformerModelDecode(Tensor hidden, int startPos)
        {
            float* hiddenPtr = GetFloatPtr(hidden);
            int maxSeqLen = (int)_kvCacheK[0].Sizes[1];
            var a = _modelDecodeArrays;

            GgmlBasicOps.TransformerModelDecode(
                (IntPtr)hiddenPtr, Config.HiddenSize, Config.NumLayers,
                a.AttnNorm, a.Qkv, a.QNorm, a.KNorm,
                a.O, a.FfnNorm, a.Gu, a.Down,
                a.KCache, a.VCache,
                a.QkvBias,
                a.Q, a.K, a.V, a.SplitType, a.SplitBytes,
                a.QkvTypes, a.QkvBytesPerLayer,
                a.OTypes, a.OBytesPerLayer,
                a.GuTypes, a.GuBytesPerLayer,
                a.DownTypes, a.DownBytesPerLayer,
                a.QkvType, a.QkvNe0, a.QkvNe1, a.QkvBytes,
                a.OType, a.ONe0, a.ONe1, a.OBytes,
                a.GuType, a.GuNe0, a.GuNe1, a.GuBytes,
                a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                Config.HeadDim, Config.NumHeads, Config.NumKVHeads,
                maxSeqLen, startPos,
                Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                Config.IntermediateSize, 2,
                _kvCacheDtype.GgmlType());
        }

        #endregion

        public override void Dispose()
        {
            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();

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
