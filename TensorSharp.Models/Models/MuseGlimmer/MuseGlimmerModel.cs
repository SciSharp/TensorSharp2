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
    /// <summary>
    /// Muse-Glimmer (arch string "muse-glimmer") - a dense decoder with:
    /// - Interleaved sliding-window attention. The pattern period comes from
    ///   "muse-glimmer.attention.sliding_window_pattern": with period P layer l is a
    ///   sliding-window layer when (l % P) &lt; (P - 1), so every P-th layer is a full
    ///   causal layer. For the 30B (52 layers, P = 4) that is 39 SWA + 13 full.
    /// - RoPE on the sliding-window layers only; the full layers are NoPE.
    ///   RoPE is ggml's NORM (interleaved-pair) flavour, not NeoX.
    /// - Per-head RMSNorm on Q and K. The Q norm weight carries the model's
    ///   qk_scale_factor (folded in at conversion time), the K norm weight is ones.
    /// - An attention output gate: sigmoid(W_gate @ attn_norm(x)) multiplies the
    ///   attention output before o_proj (same shape as afmoe / Qwen3.5).
    /// - Four RMSNorms per layer. The pre-attention and pre-FFN norms use the
    ///   model's f_norm_rms_eps; the post-attention and post-FFN norms use a
    ///   hardcoded 1e-8 (see llama.cpp src/models/muse-glimmer.cpp).
    /// - An unweighted RMSNorm on the input embeddings (no learned scale).
    /// - Dense SwiGLU FFN (no MoE).
    /// - Output multiplier ("muse-glimmer.logit_scale") applied to the logits,
    ///   followed by tanh softcapping at "muse-glimmer.final_logit_softcapping".
    /// - Optional vision tower (mmproj, projector type "muse-glimmer").
    /// </summary>
    public partial class MuseGlimmerModel : ModelBase
    {
        /// <summary>Hardcoded epsilon of the post-attention / post-FFN norms.</summary>
        private const float PostNormEps = 1e-8f;

        private const int DefaultSwaPatternPeriod = 4;
        private const int DefaultSlidingWindow = 2048;
        private const float DefaultFinalLogitSoftcap = 20.0f;
        private const float DefaultLogitScale = 1.0f;

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int _kvCacheCapacity;

        private float[] _ropeFreqs;
        private int _slidingWindow;
        private int _headDim;
        private float _finalLogitScale;
        private float _finalLogitSoftcap;
        private bool _hasTiedOutput;

        /// <summary>is_swa[l]; the complement is a full causal (NoPE) layer.</summary>
        private bool[] _isSwaLayer;

        /// <summary>Cached ones vector for the weightless input-embedding RMSNorm.</summary>
        private Tensor _onesForEmbNorm;

        private string[][] _layerWeightNames;

        private int[] _cachedSWAMaskWidths;
        private int _cachedSWAMaskQueryLen;
        private int _cachedSWAMaskStartPos = -1;

        private MuseGlimmerVisionEncoder _visionEncoder;
        private readonly List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        /// <summary>
        /// On MLX, kick the accumulated lazy graph every N layers so the GPU
        /// starts the front of the stack while the host queues the back.
        /// </summary>
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();

        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS")
                         ?? Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v >= 0)
                return v > 0 ? v : int.MaxValue;   // 0 disables the periodic kick
            return 4;
        }

        /// <param name="draftModelPath">Optional DFlash drafter GGUF (general.architecture
        /// "dflash"). When supplied, speculative decoding is armed after the target
        /// weights are in place.</param>
        public MuseGlimmerModel(string ggufPath, BackendType backend, int tpDegree = 1,
            ITensorParallelGroup tpGroup = null, string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            string arch = _gguf.GetString("general.architecture") ?? "muse-glimmer";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _headDim = Config.HeadDim;
            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", DefaultSlidingWindow);
            Config.SlidingWindow = _slidingWindow;
            _finalLogitSoftcap = _gguf.GetFloat32($"{arch}.final_logit_softcapping", DefaultFinalLogitSoftcap);
            _finalLogitScale = _gguf.GetFloat32($"{arch}.logit_scale", DefaultLogitScale);

            ResolveSwaPattern(arch);

            if (IsTensorParallel)
                throw new NotSupportedException("Muse-Glimmer does not support tensor parallelism yet; run with --tp 1.");

            int swaCount = 0;
            for (int l = 0; l < Config.NumLayers; l++)
                if (_isSwaLayer[l]) swaCount++;

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"HeadDim={_headDim}, FFN={Config.IntermediateSize}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase} scale={Config.RopeScale} (sliding-window layers only; full layers are NoPE)");
            Console.WriteLine($"Sliding window={_slidingWindow}, LogitScale={_finalLogitScale}, Softcap={_finalLogitSoftcap}");
            Console.WriteLine($"Layer types: {swaCount} sliding-window, {Config.NumLayers - swaCount} full causal");

            ParseTokenizer();
            LoadWeights();

            _hasTiedOutput = !_weights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("output.weight");
            if (_hasTiedOutput)
                Console.WriteLine("  Output tied to token_embd.weight");

            FuseGateUpWeights();
            PrepareCudaQuantizedWeightsForInference();

            PrecomputeConstants();

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            InitKVCache(initialCacheLength, maxContextLength);

            // --draft-model, else TS_MUSE_GLIMMER_DFLASH.
            string dflashPath = ResolveDFlashPath(draftModelPath);
            if (!string.IsNullOrWhiteSpace(dflashPath))
                LoadDFlashDraftWeights(dflashPath);
        }

        /// <summary>
        /// "{arch}.attention.sliding_window_pattern" is written either as a scalar
        /// period (llama.cpp's set_swa_pattern) or as a per-layer bool array. Both
        /// spellings are accepted; a missing key falls back to period 4.
        /// </summary>
        private void ResolveSwaPattern(string arch)
        {
            int numLayers = Config.NumLayers;
            _isSwaLayer = new bool[numLayers];

            string key = $"{arch}.attention.sliding_window_pattern";
            bool[] perLayer = _gguf.GetBoolArray(key);
            if (perLayer != null && perLayer.Length >= numLayers)
            {
                Array.Copy(perLayer, _isSwaLayer, numLayers);
                return;
            }

            uint period = _gguf.GetUint32(key, DefaultSwaPatternPeriod);
            if (period == 0)
                period = DefaultSwaPatternPeriod;

            // llama_hparams::set_swa_pattern(period, dense_first = false):
            //   is_swa[l] = (l % period) < (period - 1)
            for (int l = 0; l < numLayers; l++)
                _isSwaLayer[l] = (l % (int)period) < (int)(period - 1);
        }

        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            _layerWeightNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                _layerWeightNames[l] = new[]
                {
                    p + "attn_norm.weight",             // 0
                    p + "attn_q.weight",                // 1
                    p + "attn_k.weight",                // 2
                    p + "attn_v.weight",                // 3
                    p + "attn_gate.weight",             // 4
                    p + "attn_q_norm.weight",           // 5
                    p + "attn_k_norm.weight",           // 6
                    p + "attn_output.weight",           // 7
                    p + "post_attention_norm.weight",   // 8
                    p + "ffn_norm.weight",              // 9
                    p + "ffn_gate_up.weight",           // 10
                    p + "ffn_down.weight",              // 11
                    p + "post_ffw_norm.weight",         // 12
                };
            }

            int halfDim = _headDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / _headDim);
        }

        /// <summary>
        /// Zero a freshly allocated KV tensor UNCONDITIONALLY.
        ///
        /// <see cref="ModelBase.InitializeCacheTensor"/> deliberately skips the fill on
        /// GgmlCuda/MLX because the per-op attention only ever reads rows it wrote.
        /// The fused kernel does not have that property: it reads a window padded up
        /// to a 256-row stride so the decode graph's shape (and therefore its CUDA
        /// capture) stays stable, and the padding rows are masked with -inf. Adding
        /// -inf to an uninitialized NaN still yields NaN, so those rows must be
        /// finite. One fill at allocation is the cheap way to guarantee it.
        /// </summary>
        private void ZeroCacheTensor(Tensor tensor)
        {
            if (tensor != null)
                Ops.Fill(tensor, 0f);
        }

        private void InitKVCache(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            _kvSwaRows = ResolveSwaRows(initialSeqLen);
            int swaLayers = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                bool ring = _kvSwaRows > 0 && _isSwaLayer[l];
                int rows = ring ? _kvSwaRows : initialSeqLen;
                if (ring) swaLayers++;
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, rows, _headDim);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, rows, _headDim);
                ZeroCacheTensor(_kvCacheK[l]);
                ZeroCacheTensor(_kvCacheV[l]);
            }
            if (_kvSwaRows > 0)
            {
                long sized = (long)(Config.NumLayers - swaLayers) * initialSeqLen + (long)swaLayers * _kvSwaRows;
                long uniform = (long)Config.NumLayers * initialSeqLen;
                long bytes = sized * Config.NumKVHeads * _headDim * 2 * kvDtype.Size();
                Console.WriteLine(
                    $"  Muse-Glimmer KV cache: {swaLayers} sliding-window layers ring at {_kvSwaRows} rows, " +
                    $"{Config.NumLayers - swaLayers} full layers at {initialSeqLen} " +
                    $"({bytes / (1024 * 1024)} MB, {sized * 100 / uniform}% of a uniform cache).");
            }
            _cacheSeqLen = 0;
        }

        /// <summary>
        /// Rows allocated to each sliding-window layer, or 0 when every layer is
        /// sized for the full context.
        ///
        /// 39 of the 52 layers never look back further than n_swa (2048), so sizing
        /// them for a 64K context wastes most of the cache: uniform F16 KV at 64K is
        /// 3.5 GB, of which llama.cpp allocates 954 MB by giving its SWA layers their
        /// own small cache (llama-kv-cache-iswa.cpp sizes them
        /// min(n_ctx, n_swa + n_ubatch)). On a 16 GB card that difference is the whole
        /// headroom: a 64K text-only run peaked at 16059 MiB of 16384 and lost 22% of
        /// its decode rate to the resulting memory pressure.
        ///
        /// The rows are a RING indexed by (position % rows). Only the fused kernel
        /// reads that layout - see RequireUniformCache.
        /// </summary>
        private int _kvSwaRows;

        private static readonly bool SwaRingEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_SWA_RING"), "0", StringComparison.Ordinal);

        /// <summary>
        /// Ring size for the SWA layers at a given capacity, or 0 to keep them
        /// full-sized.
        ///
        /// It has to hold the window plus a whole prefill chunk, otherwise a chunk
        /// would overwrite rows its own earlier rows still need. The trailing +1 is
        /// the same margin <see cref="DFlashConfig.RingRows"/> uses (n_swa +
        /// block_size + 1): sized without it, a 4651-token prompt at chunk 2048
        /// diverged from llama.cpp on the first decode step, and 4352 rows
        /// (pad(2048 + 2048 + 1, 256)) is the smallest size that reproduces
        /// llama.cpp exactly.
        /// </summary>
        private int ResolveSwaRows(int capacity)
        {
            if (!SwaRingEnabled || !CanUseFusedForward || _slidingWindow <= 0)
                return 0;
            int need = _slidingWindow + Math.Max(PrefillChunkTokens, 1) + 1;
            string raw = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_SWA_ROWS");
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int forced) && forced > 0)
                need = forced;
            int rows = ((need + 255) / 256) * 256;
            return rows < capacity ? rows : 0;
        }

        /// <summary>Rows the given layer's cache actually has.</summary>
        private int KvRows(int layer)
            => _kvSwaRows > 0 && _isSwaLayer[layer] ? _kvSwaRows : _kvCacheCapacity;

        /// <summary>Cache row holding absolute position <paramref name="pos"/>.</summary>
        private int KvRow(int layer, int pos)
            => _kvSwaRows > 0 && _isSwaLayer[layer] ? pos % _kvSwaRows : pos;

        /// <summary>
        /// The per-op attention path treats a cache row as an absolute position, which
        /// a ring row is not. The ring is only ever armed when the fused kernel is
        /// available, so reaching here means the fused forward declined at run time;
        /// failing loudly beats returning quietly wrong logits.
        /// </summary>
        private void RequireUniformCache(int layer)
        {
            if (_kvSwaRows > 0 && _isSwaLayer[layer])
            {
                throw new InvalidOperationException(
                    "Muse-Glimmer sliding-window layers are using a ring KV cache, which only the fused " +
                    "kernel can read, but the fused forward declined and the per-op path was reached. " +
                    "Set TS_MUSE_GLIMMER_SWA_RING=0 to allocate every layer for the full context instead " +
                    "(costs ~2.5 GB more KV at 64K).");
            }
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            DType kvDtype = _kvCacheDtype.ToDType();
            int oldSwaRows = _kvSwaRows;
            int newSwaRows = ResolveSwaRows(newCapacity);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                bool swa = newSwaRows > 0 && _isSwaLayer[l];
                int rows = swa ? newSwaRows : newCapacity;
                var newK = new Tensor(_allocator, kvDtype, Config.NumKVHeads, rows, _headDim);
                var newV = new Tensor(_allocator, kvDtype, Config.NumKVHeads, rows, _headDim);
                ZeroCacheTensor(newK);
                ZeroCacheTensor(newV);

                if (_cacheSeqLen > 0)
                {
                    int oldRows = oldSwaRows > 0 && _isSwaLayer[l] ? oldSwaRows : _kvCacheCapacity;
                    if (oldRows == rows)
                    {
                        using (var srcK = _kvCacheK[l].Narrow(1, 0, rows))
                        using (var dstK = newK.Narrow(1, 0, rows))
                            Ops.Copy(dstK, srcK);
                        using (var srcV = _kvCacheV[l].Narrow(1, 0, rows))
                        using (var dstV = newV.Narrow(1, 0, rows))
                            Ops.Copy(dstV, srcV);
                    }
                    else if (oldRows > rows || rows > oldRows)
                    {
                        // The layout changed (a layer just became a ring, or the ring
                        // grew), so a row means something different now. Re-place the
                        // positions that are still reachable.
                        int keep = Math.Min(_cacheSeqLen, Math.Min(rows, oldRows));
                        for (int p = _cacheSeqLen - keep; p < _cacheSeqLen; p++)
                        {
                            using (var srcK = _kvCacheK[l].Narrow(1, p % oldRows, 1))
                            using (var dstK = newK.Narrow(1, p % rows, 1))
                                Ops.Copy(dstK, srcK);
                            using (var srcV = _kvCacheV[l].Narrow(1, p % oldRows, 1))
                            using (var dstV = newV.Narrow(1, p % rows, 1))
                                Ops.Copy(dstV, srcV);
                        }
                    }
                }

                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
            _kvSwaRows = newSwaRows;
            // The captured decode graph baked the old KV device addresses and cache
            // size; replaying it against the reallocated buffers would hang.
            ResetFusedDecodeCache();
            _fusedProbed = false;
            if (newSwaRows > 0)
            {
                int swaLayers = 0;
                for (int l = 0; l < Config.NumLayers; l++) if (_isSwaLayer[l]) swaLayers++;
                long sized = (long)(Config.NumLayers - swaLayers) * newCapacity + (long)swaLayers * newSwaRows;
                long uniform = (long)Config.NumLayers * newCapacity;
                Console.WriteLine(
                    $"Expanded Muse-Glimmer attention cache to {newCapacity} tokens " +
                    $"({swaLayers} sliding-window layers ring at {newSwaRows} rows, " +
                    $"{sized * 100 / uniform}% of a uniform cache).");
            }
            else
            {
                Console.WriteLine($"Expanded Muse-Glimmer attention cache to {newCapacity} tokens.");
            }
        }

        protected override void ResetKVCacheCore()
        {
            _cacheSeqLen = 0;
            _cachedSWAMaskStartPos = -1;
            ResetFusedDecodeCache();
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
            base.TruncateKVCacheCore(tokenCount);
            _cachedSWAMaskStartPos = -1;
            ResetFusedDecodeCache();
            if (_kvCacheK == null) return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
        }

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        /// <summary>
        /// The POOLED snapshot path is capped at the sliding-window ring size, for the
        /// same reason Gemma 4 caps at its window: restoring more positions than a ring
        /// layer physically holds would have to reconstruct rows that the wrap already
        /// overwrote. Below the cap a restore is byte-identical to a real prefill (no
        /// row wraps, so every layer addresses linearly).
        ///
        /// Reuse BEYOND the cap still happens, and correctly, through live-cache
        /// continuation (BatchExecutor.ComputeLiveContinuationLcp), which keeps the
        /// model's actual cache between same-session turns instead of rebuilding it.
        /// That path is only armed for models that report a finite cap, so publishing
        /// one here is what turns multi-turn reuse on for Muse-Glimmer.
        /// </summary>
        public override int MaxReusablePrefixTokens
            => _kvSwaRows > 0 ? _kvSwaRows : int.MaxValue;

        public override string KVStateFingerprint =>
            $"muse-glimmer|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}" +
            $"|hd={_headDim}|swa={_slidingWindow}|rows={_kvSwaRows}|dtype={_kvCacheDtype.ToShortString()}";

        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokenCount);

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            // The fused kernel writes K/V on-device; the snapshot walks the caches
            // through host pointers, so the device rows have to come back first or we
            // would capture the host mirror's stale (zero-filled) bytes.
            EnsureFusedKvHostSynchronized();
            return KvBlockTransfer.Extract(
                _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                startToken, tokenCount, destination, _kvSwaRows);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            EnsureCacheCapacity(destToken + tokenCount);
            if (!KvBlockTransfer.Inject(
                    _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                    destToken, tokenCount, source, _kvSwaRows))
            {
                return false;
            }
            // The host copy is now authoritative for [0, destToken+tokenCount); the
            // device copies are dropped below so the next forward re-uploads it.
            _fusedKvDirty = false;
            _cacheSeqLen = destToken + tokenCount;
            _cachedSWAMaskStartPos = -1;
            // Dropping the device copies moves the KV buffers the captured decode
            // graph baked in; replaying that capture would hit freed memory.
            ResetFusedDecodeCache();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            return true;
        }

        public void LoadVisionEncoder(string mmProjPath)
        {
            _visionEncoder = new MuseGlimmerVisionEncoder(mmProjPath, _allocator);
            _visionEncoder.SetHostModel(this);
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public MuseGlimmerVisionEncoder VisionEncoder => _visionEncoder;

        internal bool HasPendingVisionEmbeddings => _pendingVisionEmbeddingsList.Count > 0;

        /// <summary>
        /// Largest number of tokens submitted in one forward. llama.cpp splits a
        /// prompt into <c>n_ubatch</c>-sized chunks for the same reason: the compute
        /// graph's activations scale with the row count, so a single 16K-token graph
        /// needs tens of GB even though 16K tokens of KV cost under a GB. 0 disables
        /// chunking.
        /// </summary>
        private static readonly int PrefillChunkTokens = ResolvePrefillChunk();

        private static int ResolvePrefillChunk()
        {
            string raw = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v >= 0)
                return v;
            return 2048;
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            // Vision rows are injected at absolute prompt offsets and the pending list
            // is drained in one go, so a multimodal prompt has to stay whole.
            if (PrefillChunkTokens > 0 && tokens.Length > PrefillChunkTokens && !HasPendingVisionEmbeddings)
                return ForwardChunked(tokens, PrefillChunkTokens);

            // The SWA ring holds sliding_window + chunk + 1 rows, so a forward wider
            // than (rows - sliding_window) would alias two live positions onto one slot
            // and silently corrupt all 39 sliding-window layers. Chunking keeps text
            // prompts under the limit by construction; a multimodal prompt skips
            // chunking, so it is the one way to get here oversized.
            if (_kvSwaRows > 0 && tokens.Length > _kvSwaRows - _slidingWindow)
            {
                throw new InvalidOperationException(
                    $"Muse-Glimmer forward of {tokens.Length} tokens exceeds what the sliding-window " +
                    $"ring can hold in one pass ({_kvSwaRows - _slidingWindow} rows: ring {_kvSwaRows} " +
                    $"minus window {_slidingWindow}). Raise TS_MUSE_GLIMMER_PREFILL_CHUNK (the ring is " +
                    "sized from it) or set TS_MUSE_GLIMMER_SWA_RING=0.");
            }

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            // Text-only fast path: the fused graph does the embedding gather and the
            // weightless input norm itself, so nothing is materialised on the host.
            if (!HasPendingVisionEmbeddings && FusedInGraphEmbedAvailable)
            {
                float[] embedded = TryFusedForward(null, seqLen, startPos, null, null, tokens);
                if (embedded != null)
                {
                    _logitsBuffer = embedded;
                    _cacheSeqLen += seqLen;
                    _forwardCount++;
                    _forwardSw.Stop();
                    return _logitsBuffer;
                }
            }

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            // Multimodal rows replace the token embeddings BEFORE the input norm:
            // llama.cpp feeds projected image embeddings straight into build_inp_embd
            // and applies the weightless RMSNorm to the merged stream.
            DrainPendingVisionEmbeddings(hidden, startPos);

            ApplyInputEmbeddingNorm(hidden, seqLen);

            // Whole-model fused graph (one GGML submission for all 52 layers, the
            // final norm, the LM head, the logit scale and the softcap).
            float[] fused = TryFusedForward(hidden, seqLen, startPos, null, null);
            if (fused != null)
            {
                hidden.Dispose();
                _logitsBuffer = fused;
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return _logitsBuffer;
            }

            EnsureFusedKvHostSynchronized();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                hidden = TransformerBlock(hidden, l, seqLen, startPos);
                if (_backend == BackendType.Mlx && (l + 1) % MlxEvalEveryNLayers == 0
                    && l + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
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
            Tensor logitsTensor = LinearForward(lastHidden, _hasTiedOutput ? "token_embd.weight" : "output.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
            lastHidden.Dispose();

            ApplyOutputTransform(logitsTensor);

            t0 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

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

            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;
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
            EnsureCacheCapacity(startPos + seqLen);

            if (!HasPendingVisionEmbeddings && FusedInGraphEmbedAvailable
                && TryFusedForward(null, seqLen, startPos, null, null, tokens) != null)
            {
                _cacheSeqLen += seqLen;
                _forwardSw.Stop();
                return;
            }

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            DrainPendingVisionEmbeddings(hidden, startPos);
            ApplyInputEmbeddingNorm(hidden, seqLen);

            // The fused kernel always folds the LM head, so a logits-free prefill
            // chunk simply discards them - still far cheaper than the per-op chain.
            if (TryFusedForward(hidden, seqLen, startPos, null, null) != null)
            {
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardSw.Stop();
                return;
            }

            EnsureFusedKvHostSynchronized();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                hidden = TransformerBlock(hidden, l, seqLen, startPos);
                if (_backend == BackendType.Mlx && (l + 1) % MlxEvalEveryNLayers == 0
                    && l + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        /// <summary>
        /// Feeds a long prompt through in <paramref name="chunk"/>-token pieces.
        /// Each piece advances <c>_cacheSeqLen</c>, so the next one sees the previous
        /// one's keys and values exactly as if the whole prompt had been submitted at
        /// once; only the final piece's logits are wanted, and those are what the
        /// last call returns.
        /// </summary>
        private float[] ForwardChunked(int[] tokens, int chunk)
        {
            float[] last = null;
            for (int off = 0; off < tokens.Length; off += chunk)
            {
                int n = Math.Min(chunk, tokens.Length - off);
                int[] piece = new int[n];
                Array.Copy(tokens, off, piece, 0, n);
                last = ForwardCore(piece);
            }
            return last;
        }

        private void DrainPendingVisionEmbeddings(Tensor hidden, int startPos)
        {
            if (_pendingVisionEmbeddingsList.Count == 0)
                return;
            foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
            {
                InjectVisionEmbeddings(hidden, embeddings, position, startPos);
                embeddings.Dispose();
            }
            _pendingVisionEmbeddingsList.Clear();
        }

        private void InjectVisionEmbeddings(Tensor hidden, Tensor visionEmbeddings, int insertPos, int startPos)
        {
            int numVisionTokens = (int)visionEmbeddings.Sizes[0];
            using var target = hidden.Narrow(0, insertPos, numVisionTokens);
            if (ReferenceEquals(target.Allocator, visionEmbeddings.Allocator))
            {
                Ops.Copy(target, visionEmbeddings);
            }
            else
            {
                float[] host = visionEmbeddings.GetElementsAsFloat((int)visionEmbeddings.ElementCount());
                using var staged = new Tensor(hidden.Allocator, visionEmbeddings.ElementType, visionEmbeddings.Sizes);
                staged.SetElementsAsFloat(host);
                Ops.Copy(target, staged);
            }
            Console.WriteLine($"Injected {numVisionTokens} vision tokens at chunk-offset {insertPos} (absolute position {startPos + insertPos})");
        }

        /// <summary>
        /// Weightless RMSNorm over the input embeddings (llama.cpp:
        /// build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1)).
        /// </summary>
        private void ApplyInputEmbeddingNorm(Tensor hidden, int seqLen)
        {
            int dim = Config.HiddenSize;
            if (_onesForEmbNorm == null || (int)_onesForEmbNorm.Sizes[0] != dim)
            {
                _onesForEmbNorm?.Dispose();
                _onesForEmbNorm = new Tensor(_allocator, DType.Float32, dim);
                Ops.Fill(_onesForEmbNorm, 1f);
            }
            using var reshaped = hidden.View(seqLen, dim);
            Ops.RMSNorm(reshaped, reshaped, _onesForEmbNorm, null, Config.Eps);
        }

        /// <summary>logits *= logit_scale, then tanh softcapping.</summary>
        private void ApplyOutputTransform(Tensor logits)
        {
            if (_finalLogitScale != 1.0f && _finalLogitScale > 0f)
                Ops.Mul(logits, logits, _finalLogitScale);

            if (_finalLogitSoftcap > 0f)
            {
                float cap = _finalLogitSoftcap;
                Ops.Mul(logits, logits, 1f / cap);
                Ops.Tanh(logits, logits);
                Ops.Mul(logits, logits, cap);
            }
        }

        /// <summary>RMSNorm with an explicit epsilon (the post norms use 1e-8, not Config.Eps).</summary>
        private Tensor RMSNormWithEps(Tensor input, string weightName, float eps)
        {
            long t0 = Stopwatch.GetTimestamp();
            var alpha = _weights[weightName];
            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);
            Tensor input2d = input.Sizes.Length != 2 ? input.View(rows, dim) : null;
            Tensor src = input2d ?? input;
            Tensor result = Ops.RMSNorm(null, src, alpha, null, eps);
            input2d?.Dispose();
            _normTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string[] names = _layerWeightNames[layer];

            // Pre-attention norm; the same tensor feeds Q/K/V and the output gate.
            using var attnNormed = RMSNormOp(hidden, names[0]);

            using var attnOut = Attention(attnNormed, layer, names, seqLen, startPos);

            // post_attention_norm: rms_norm(eps = 1e-8) * weight
            using var postAttnNormed = RMSNormWithEps(attnOut, names[8], PostNormEps);

            Ops.Add(postAttnNormed, postAttnNormed, hidden);
            hidden.Dispose();

            // ffn_norm + gate/up + SiLU-mul + down in a single GGML graph when the
            // backend supports it (actType 0 = SwiGLU). The fused rms_norm uses the
            // same weight and Config.Eps as RMSNormOp, so the result is numerically
            // identical to the unfused chain; it just keeps the [tokens, 2*n_ff]
            // intermediate device-resident instead of round-tripping it per op.
            // post_ffw_norm still runs on the small returned tensor.
            //
            // Prefill only. Measured on Muse-Glimmer-30B / ggml_cuda / RTX 3080:
            // prefill 31.5 -> 55.5 tok/s, but single-row decode 4.3 -> 3.9 tok/s -
            // at one row the fused graph's build/bind cost outweighs keeping the
            // [1, 2*n_ff] intermediate resident.
            Tensor ffnOut = seqLen > 1
                ? TryFusedDenseFFNProject(postAttnNormed, names[9], names[10], names[11], actType: 0)
                : null;
            if (ffnOut == null)
            {
                using var ffnNormed = RMSNormOp(postAttnNormed, names[9]);
                ffnOut = FFN(ffnNormed, names[10], names[11], seqLen);
            }

            using var _ffnOut = ffnOut;
            using var postFfnNormed = RMSNormWithEps(ffnOut, names[12], PostNormEps);

            var result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
            Ops.Add(result, postAttnNormed, postFfnNormed);
            return result;
        }

        private Tensor Attention(Tensor input, int layer, string[] names, int seqLen, int startPos)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool useRope = _isSwaLayer[layer];

            Tensor q = LinearForward(input, names[1]);
            Tensor k = LinearForward(input, names[2]);
            Tensor v = LinearForward(input, names[3]);
            Tensor gate = LinearForward(input, names[4]);

            // Per-head QK RMSNorm. The Q norm weight absorbs qk_scale_factor.
            if (seqLen == 1)
            {
                RMSNormInPlace(q, _weights[names[5]], Config.NumHeads, _headDim, Config.Eps);
                RMSNormInPlace(k, _weights[names[6]], Config.NumKVHeads, _headDim, Config.Eps);
            }
            else
            {
                q = ApplyBatchRMSNorm(q, names[5], Config.NumHeads, seqLen, _headDim);
                k = ApplyBatchRMSNorm(k, names[6], Config.NumKVHeads, seqLen, _headDim);
            }

            // RoPE (NORM / interleaved pairs) on the sliding-window layers only.
            if (useRope)
            {
                if (seqLen == 1)
                {
                    ApplyNormRoPEDecode(q, Config.NumHeads, _headDim, startPos, _ropeFreqs);
                    ApplyNormRoPEDecode(k, Config.NumKVHeads, _headDim, startPos, _ropeFreqs);
                }
                else
                {
                    q = ApplyRoPEPrefill(q, Config.NumHeads, _headDim, seqLen, startPos);
                    k = ApplyRoPEPrefill(k, Config.NumKVHeads, _headDim, seqLen, startPos);
                }
            }

            float qScale = 1f / MathF.Sqrt(_headDim);
            Ops.Mul(q, q, qScale);

            int totalSeqLen = startPos + seqLen;
            Tensor attn;

            if (seqLen == 1)
            {
                RequireUniformCache(layer);
                CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v,
                    Config.NumKVHeads, _headDim, startPos);

                int attendLen = _isSwaLayer[layer] ? Math.Min(totalSeqLen, _slidingWindow) : totalSeqLen;
                int attendStart = totalSeqLen - attendLen;

                attn = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * _headDim);
                AttentionDecodeWithWindow(q, _kvCacheK[layer], _kvCacheV[layer], attn,
                    Config.NumHeads, Config.NumKVHeads, _headDim, _headDim,
                    attendStart, totalSeqLen);
            }
            else
            {
                Tensor qHeads = ReshapeToHeads(q, Config.NumHeads, seqLen, _headDim);
                Tensor kHeads = ReshapeToHeads(k, Config.NumKVHeads, seqLen, _headDim);
                Tensor vHeads = ReshapeToHeads(v, Config.NumKVHeads, seqLen, _headDim);

                RequireUniformCache(layer);
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

                ApplyCausalMask(scores, seqLen, totalSeqLen, _isSwaLayer[layer] ? _slidingWindow : 0);
                Ops.Softmax(scores, scores);

                var attnHeads = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, _headDim);
                Ops.AddmmBatch(attnHeads, 0, attnHeads, 1.0f, scores, vExpanded);
                scores.Dispose();
                vExpanded.Dispose();

                attn = ReshapeFromHeads(attnHeads, Config.NumHeads, seqLen, _headDim);
                attnHeads.Dispose();
            }

            q.Dispose();
            k.Dispose();
            v.Dispose();

            // Attention output gate: attn * sigmoid(gate), applied before o_proj.
            Ops.SigmoidMul(attn, attn, gate);
            gate.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            using (attn)
            {
                return LinearForward(attn, names[7]);
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
        /// ggml NORM-flavour RoPE: rotates adjacent pairs (x[2i], x[2i+1]).
        /// Decode fast path (single row) to avoid a RoPEEx dispatch.
        /// </summary>
        private unsafe void ApplyNormRoPEDecode(Tensor data, int numHeads, int headDim, int position, float[] freqs)
        {
            float* ptr = GetFloatPtr(data);
            int pairCount = headDim / 2;

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int i = 0, pair = 0; i < pairCount; i++, pair += 2)
                {
                    float angle = position * freqs[i];
                    float cos = MathF.Cos(angle);
                    float sin = MathF.Sin(angle);
                    float x0 = head[pair];
                    float x1 = head[pair + 1];
                    head[pair] = x0 * cos - x1 * sin;
                    head[pair + 1] = x1 * cos + x0 * sin;
                }
            }
            InvalidateTensorDeviceCache(data);
        }

        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensorOn(data.Storage.Allocator, positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            // mode 0 == GGML_ROPE_TYPE_NORM (interleaved pairs).
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, 0, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        /// <summary>
        /// Single-row decode attention over the half-open key range
        /// [attendStart, totalSeqLen). Full-causal layers pass attendStart = 0;
        /// sliding-window layers pass totalSeqLen - min(totalSeqLen, n_swa), which is
        /// llama.cpp's STANDARD SWA predicate (a key at p0 is visible to the query at
        /// p1 iff p1 - p0 &lt; n_swa).
        ///
        /// <see cref="ApplyModelAlignedKvCacheDefault"/> picks F16 caches for quantized
        /// models, so all three cache dtypes are handled: F32 and F16 walk the cache in
        /// place, block-quantized caches dequantize the live window first (same
        /// deep-fallback shape as <see cref="AttentionDecodePureCS"/>).
        /// </summary>
        private unsafe void AttentionDecodeWithWindow(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int attendStart, int totalSeqLen)
        {
            if (IsBlockQuantCacheDType(kCache.ElementType) && vCache.ElementType == kCache.ElementType)
            {
                // Dequantize [0, totalSeqLen) into compact F32 (no GQA broadcast: the
                // loop below reads per KV head), then re-enter. The compact copy's
                // Sizes[1] == totalSeqLen doubles as its row stride.
                using Tensor kF32 = ExpandKVHeads(kCache, 1, totalSeqLen);
                using Tensor vF32 = ExpandKVHeads(vCache, 1, totalSeqLen);
                AttentionDecodeWithWindow(q, kF32, vF32, result,
                    numHeads, numKVHeads, keyDim, valDim, attendStart, totalSeqLen);
                return;
            }

            bool half = kCache.ElementType == DType.Float16;
            if (half != (vCache.ElementType == DType.Float16))
            {
                throw new NotSupportedException(
                    $"Muse-Glimmer decode attention needs matching K/V cache dtypes, got {kCache.ElementType}/{vCache.ElementType}.");
            }

            float* qPtr = GetFloatPtr(q);
            float* rPtr = GetFloatPtr(result);
            float* kPtrF = half ? null : GetFloatPtr(kCache);
            float* vPtrF = half ? null : GetFloatPtr(vCache);
            ushort* kPtrH = half ? TensorComputePrimitives.GetHalfPointer(kCache) : null;
            ushort* vPtrH = half ? TensorComputePrimitives.GetHalfPointer(vCache) : null;

            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;
            int attendLen = totalSeqLen - attendStart;

            nint qN = (nint)qPtr, rN = (nint)rPtr;
            nint kN = half ? (nint)kPtrH : (nint)kPtrF;
            nint vN = half ? (nint)vPtrH : (nint)vPtrF;

            void Head(int h)
            {
                float* qHead = (float*)qN + h * keyDim;
                int kvHead = h / groupSize;
                long kBase = (long)kvHead * maxSeqLen * keyDim + (long)attendStart * keyDim;
                long vBase = (long)kvHead * maxSeqLen * valDim + (long)attendStart * valDim;

                float* scores = stackalloc float[attendLen];

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < attendLen; t++)
                {
                    float s = half
                        ? TensorComputePrimitives.DotF32F16(qHead, (ushort*)kN + kBase + (long)t * keyDim, keyDim)
                        : VecDot(qHead, (float*)kN + kBase + (long)t * keyDim, keyDim);
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

                float* rHead = (float*)rN + h * valDim;
                VecZero(rHead, valDim);
                for (int t = 0; t < attendLen; t++)
                {
                    if (half)
                        TensorComputePrimitives.ScaleAddF16(rHead, (ushort*)vN + vBase + (long)t * valDim, scores[t], valDim);
                    else
                        VecScaleAdd(rHead, (float*)vN + vBase + (long)t * valDim, scores[t], valDim);
                }
            }

            if (numHeads > 1 && (long)numHeads * attendLen >= 4096)
                System.Threading.Tasks.Parallel.For(0, numHeads, Head);
            else
                for (int h = 0; h < numHeads; h++) Head(h);

            InvalidateTensorDeviceCache(result);
        }

        /// <summary>
        /// Causal mask plus, for sliding-window layers, the llama.cpp STANDARD SWA
        /// predicate: a key at p0 is masked from a query at p1 when p1 - p0 &gt;= n_swa.
        /// </summary>
        private unsafe void ApplyCausalMask(Tensor scores, int queryLen, int totalKVLen, int windowSize)
        {
            int startPos = totalKVLen - queryLen;
            Ops.AddCausalMask(scores, queryLen, startPos, float.NegativeInfinity);

            if (windowSize <= 0)
                return;

            if (_cachedSWAMaskWidths == null ||
                _cachedSWAMaskQueryLen != queryLen ||
                _cachedSWAMaskStartPos != startPos)
            {
                _cachedSWAMaskWidths = new int[queryLen];
                _cachedSWAMaskQueryLen = queryLen;
                _cachedSWAMaskStartPos = startPos;
                for (int qi = 0; qi < queryLen; qi++)
                    _cachedSWAMaskWidths[qi] = Math.Max(0, startPos + qi - windowSize + 1);
            }

            float* sPtr = GetFloatPtr(scores);
            int numHeads = (int)scores.Sizes[0];
            int rowStride = queryLen * totalKVLen;

            for (int h = 0; h < numHeads; h++)
            {
                float* headScores = sPtr + (long)h * rowStride;
                for (int qi = 0; qi < queryLen; qi++)
                {
                    int width = _cachedSWAMaskWidths[qi];
                    if (width > 0)
                        new Span<float>(headScores + (long)qi * totalKVLen, width).Fill(float.NegativeInfinity);
                }
            }
            InvalidateTensorDeviceCache(scores);
        }

        public override void Dispose()
        {
            _visionEncoder?.Dispose();
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings?.Dispose();
            _pendingVisionEmbeddingsList.Clear();
            _onesForEmbNorm?.Dispose();
            DisposeDFlash();
            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            base.Dispose();
        }
    }
}
