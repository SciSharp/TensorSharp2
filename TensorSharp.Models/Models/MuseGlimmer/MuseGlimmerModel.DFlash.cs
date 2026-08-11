// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ---------------------------------------------------------------------------
// DFlash speculative drafting for Muse-Glimmer.
//
// DFlash (llama.cpp src/models/dflash.cpp) is a BLOCK drafter: one forward pass
// proposes the whole speculative window, so it plugs into the shared
// draft/verify/rollback core through IMtpSpeculativeModel.MtpDraftBlock instead
// of the per-token MtpDraftStep, and it reports a WIDE hidden row through
// MtpHiddenSize -- the concatenated per-layer input residuals of the target
// layers its encoder consumes (5 x 6656 = 33280 for Muse-Glimmer 30B).
//
// Three passes, transcribed from llama_model_dflash::graph:
//
//   PASS A -- encoder (graph<true>)
//       feat = concat(target input residual of layers dflash.target_layers)
//       g    = rmsnorm(fc @ feat, enc.output_norm, eps)          [1 row per position]
//
//   PASS B -- KV injection (graph<false>, the ubatch.embd branch)
//       K = rope_neox(headnorm(attn_k @ g, attn_k_norm), target position)
//       V = attn_v @ g                                          (no norm, no rope)
//       ring[pos % ringRows] <- K, V                            per draft layer
//       No Q, no attention, no FFN, no output.
//
//   PASS C -- block draft (graph<false>, the token branch)
//       ids  = [anchor, MASK x (block_size-1)]  at positions p .. p+B-1
//       inpL = target token_embd[ids]                           (no embedding scale)
//       5 x { pre-norm, QK head-norm, NeoX rope, attention over
//             [ring window | this block's own B keys] -- NON-CAUSAL inside the
//             block, SWA-masked against the ring -- , o_proj, residual,
//             ffn-norm, SwiGLU, residual }
//       logits = target output.weight @ rmsnorm(inpL, output_norm)
//   The drafter's logits get NEITHER the target's logit_scale NOR its tanh
//   softcap: llama.cpp's dflash graph ends at build_lora_mm(output, cur). argmax
//   is invariant to both, but the softmax CONFIDENCE is not, so the per-position
//   acceptance probabilities handed to the executor are the softmax of the RAW
//   drafter logits.
//
// Rollback is free: the trunk verify writes correct KV for every row it
// processes into the linear (non-circular) cache and Muse-Glimmer has no
// recurrent state, so partial acceptance only rewinds the position counter
// (MtpVerifyPersistsAcceptedKv).
// ---------------------------------------------------------------------------
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class MuseGlimmerModel : ModelBase, IMtpSpeculativeModel
    {
        // Per-layer weight-name slots (index into _dflashLayerNames[il]).
        private const int DfAttnNorm = 0;
        private const int DfAttnQ = 1;
        private const int DfAttnK = 2;
        private const int DfAttnV = 3;
        private const int DfAttnQNorm = 4;
        private const int DfAttnKNorm = 5;
        private const int DfAttnOutput = 6;
        private const int DfFfnNorm = 7;
        private const int DfFfnGate = 8;
        private const int DfFfnUp = 9;
        private const int DfFfnDown = 10;

        /// <summary>
        /// Prompt-prefill chunk the speculative executor should use. Deliberately
        /// small: one captured row is <see cref="DFlashConfig.FeatureSize"/> = 33280
        /// floats (133 KB), and MtpSpeculativeExecution.PrefillStep keeps TWO
        /// chunk-sized float[] buffers (_chunkH and _chunkHPairs). 128 tokens is
        /// 2 x 17 MB of managed arrays; the model's own 2048-token prefill chunk
        /// would be 2 x 545 MB. Nothing is lost by chunking finer here: the DFlash
        /// catch-up cost is strictly linear in rows (one fc GEMM plus 5 layers of
        /// k/v projections), and the trunk's per-chunk fixed overhead is small
        /// next to a 128-row Muse-Glimmer forward.
        /// </summary>
        private const int DFlashPrefillChunk = 128;

        private DFlashConfig _dflash;
        private bool _hasDFlash;

        /// <summary>target layer index -> its column block in a feature row, or -1.</summary>
        private int[] _dflashCaptureSlot;

        /// <summary>Per draft layer, the 11 weight names of that block.</summary>
        private string[][] _dflashLayerNames;

        /// <summary>The drafter's own KV ring, [numKVHeads, ringRows, headDim] per
        /// draft layer, indexed by (absolute position % ringRows).</summary>
        private Tensor[] _dflashRingK;
        private Tensor[] _dflashRingV;
        private int _dflashRingRows;

        /// <summary>True when a usable DFlash drafter is attached to this model.</summary>
        public bool HasDFlash => _hasDFlash;

        // ====================================================================
        // construction / loading
        // ====================================================================

        /// <summary>Path of the DFlash drafter GGUF, if one was configured
        /// (<c>--draft-model</c> / <c>TS_MUSE_GLIMMER_DFLASH</c>).</summary>
        internal static string ResolveDFlashPath(string explicitPath)
        {
            string path = !string.IsNullOrWhiteSpace(explicitPath)
                ? explicitPath
                : Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_DFLASH");
            return string.IsNullOrWhiteSpace(path) ? null : path;
        }

        /// <summary>
        /// Loads the DFlash drafter GGUF and attaches it to this target model. Its
        /// tensors are merged into the shared weight dictionaries under the
        /// "dflash." prefix so the existing matmul/norm machinery serves them (the
        /// same trick Gemma4Model.LoadMtpDraftTensors uses with "mtp."); the drafter
        /// borrows the TARGET's token_embd.weight and output.weight, which the file
        /// does not carry.
        /// </summary>
        public void LoadDFlashDraftWeights(string ggufPath)
        {
            if (string.IsNullOrEmpty(ggufPath) || !System.IO.File.Exists(ggufPath))
                throw new System.IO.FileNotFoundException("Muse-Glimmer DFlash drafter GGUF not found.", ggufPath);

            using var draft = new GgufFile(ggufPath);
            var cfg = DFlashConfig.FromGguf(draft);

            if (cfg.HiddenSize != Config.HiddenSize)
            {
                throw new InvalidOperationException(
                    $"DFlash embedding_length {cfg.HiddenSize} != target hidden size {Config.HiddenSize}.");
            }
            foreach (int lid in cfg.TargetLayerIds)
            {
                if (lid < 0 || lid >= Config.NumLayers)
                {
                    throw new InvalidOperationException(
                        $"DFlash target layer {lid} is outside the target's {Config.NumLayers} layers.");
                }
            }
            if (cfg.BlockSize > cfg.RingRows)
                throw new InvalidOperationException($"DFlash block_size {cfg.BlockSize} exceeds the ring ({cfg.RingRows} rows).");

            LoadDFlashDraftTensors(draft);

            _dflash = cfg;
            _dflashLayerNames = new string[cfg.NumLayers][];
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string p = $"{DFlashConfig.WeightPrefix}blk.{il}.";
                _dflashLayerNames[il] = new[]
                {
                    p + "attn_norm.weight",     // 0
                    p + "attn_q.weight",        // 1
                    p + "attn_k.weight",        // 2
                    p + "attn_v.weight",        // 3
                    p + "attn_q_norm.weight",   // 4
                    p + "attn_k_norm.weight",   // 5
                    p + "attn_output.weight",   // 6
                    p + "ffn_norm.weight",      // 7
                    p + "ffn_gate.weight",      // 8
                    p + "ffn_up.weight",        // 9
                    p + "ffn_down.weight",      // 10
                };
            }

            if (!VerifyDFlashTensors(out string missing))
            {
                Console.WriteLine($"  DFlash drafter GGUF loaded but '{missing}' is missing; DFlash drafting disabled.");
                _dflash = null;
                _dflashLayerNames = null;
                return;
            }

            _dflashCaptureSlot = new int[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
                _dflashCaptureSlot[l] = -1;
            for (int i = 0; i < cfg.TargetLayerIds.Length; i++)
                _dflashCaptureSlot[cfg.TargetLayerIds[i]] = i;

            _dflashRingRows = cfg.RingRows;
            _dflashRingK = new Tensor[cfg.NumLayers];
            _dflashRingV = new Tensor[cfg.NumLayers];
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                _dflashRingK[il] = new Tensor(_allocator, DType.Float32, cfg.NumKVHeads, _dflashRingRows, cfg.HeadDim);
                _dflashRingV[il] = new Tensor(_allocator, DType.Float32, cfg.NumKVHeads, _dflashRingRows, cfg.HeadDim);
                // Unconditional zero fill (not InitializeCacheTensor, which skips
                // GgmlCuda): the ring is only ever read over positions that have
                // been written, but a finite ring keeps a mis-sized window from
                // silently producing NaNs instead of failing loudly.
                Ops.Fill(_dflashRingK[il], 0f);
                Ops.Fill(_dflashRingV[il], 0f);
            }

            _hasDFlash = true;

            long ringBytes = 2L * cfg.NumLayers * cfg.NumKVHeads * _dflashRingRows * cfg.HeadDim * sizeof(float);
            Console.WriteLine($"  DFlash drafter ready: {cfg}");
            Console.WriteLine($"  DFlash KV ring: {_dflashRingRows} rows x {cfg.NumLayers} layers ({ringBytes / (1024 * 1024)} MB F32)");
        }

        /// <summary>
        /// Merges every tensor of the drafter GGUF into the shared weight
        /// dictionaries under the "dflash." prefix. Byte-for-byte the same shape as
        /// Gemma4Model.LoadMtpDraftTensors (which uses "mtp."), minus the
        /// converter-spelling normalization DFlash does not need: the drafter's
        /// tensor names are already the final ones.
        /// </summary>
        private unsafe void LoadDFlashDraftTensors(GgufFile draft)
        {
            foreach (var kv in draft.Tensors)
            {
                var info = kv.Value;
                string name = DFlashConfig.WeightPrefix + info.Name;
                long byteCount = draft.GetTensorByteCount(info);

                if (IsQuantizedLinearWeight(info))
                {
                    if (IsGgmlBackend)
                        EnsureQuantBackendAvailable();
                    IntPtr ptr = QuantizedWeight.AllocateBuffer(byteCount);
                    draft.ReadTensorDataToNative(info, ptr, byteCount);
                    _quantWeights[name] = new QuantizedWeight(ptr, byteCount, (int)info.Type, (long)info.Shape[0], (long)info.Shape[1]);
                }
                else
                {
                    long numElements = info.NumElements;
                    long[] tsShape = new long[info.Shape.Length];
                    for (int i = 0; i < info.Shape.Length; i++)
                        tsShape[i] = (long)info.Shape[info.Shape.Length - 1 - i];

                    var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                    IntPtr destPtr = TensorComputePrimitives.GetStoragePointer(tensor);
                    if (info.Type == GgmlTensorType.F32)
                    {
                        draft.ReadTensorDataToFloat32Native(info, destPtr, numElements);
                    }
                    else
                    {
                        IntPtr tempPtr = QuantizedWeight.AllocateBuffer(byteCount);
                        try
                        {
                            draft.ReadTensorDataToNative(info, tempPtr, byteCount);
                            NativeDequant.DequantizeToFloat32Native((int)info.Type, tempPtr, destPtr, numElements);
                        }
                        finally
                        {
                            QuantizedWeight.FreeBuffer(tempPtr);
                        }
                    }
                    _weights[name] = tensor;
                }
            }
        }

        /// <summary>True when <paramref name="name"/> resolves to something
        /// <see cref="ModelBase.LinearForward"/> can multiply by.</summary>
        private bool HasDFlashLinear(string name)
            => _quantWeights.ContainsKey(name) || _weights.ContainsKey(name);

        private bool VerifyDFlashTensors(out string missing)
        {
            string[] globals =
            {
                DFlashConfig.WeightPrefix + "fc.weight",
                DFlashConfig.WeightPrefix + "enc.output_norm.weight",
                DFlashConfig.WeightPrefix + "output_norm.weight",
            };
            foreach (string g in globals)
            {
                bool ok = g.EndsWith("norm.weight", StringComparison.Ordinal)
                    ? _weights.ContainsKey(g)
                    : HasDFlashLinear(g);
                if (!ok) { missing = g; return false; }
            }

            for (int il = 0; il < _dflash.NumLayers; il++)
            {
                string[] n = _dflashLayerNames[il];
                foreach (int slot in new[] { DfAttnNorm, DfAttnQNorm, DfAttnKNorm, DfFfnNorm })
                {
                    if (!_weights.ContainsKey(n[slot])) { missing = n[slot]; return false; }
                }
                foreach (int slot in new[] { DfAttnQ, DfAttnK, DfAttnV, DfAttnOutput, DfFfnGate, DfFfnUp, DfFfnDown })
                {
                    if (!HasDFlashLinear(n[slot])) { missing = n[slot]; return false; }
                }
            }

            // The drafter has no LM head of its own: it borrows the target's.
            if (!HasDFlashLinear(TargetOutputWeightName)) { missing = TargetOutputWeightName; return false; }
            if (!HasDFlashLinear("token_embd.weight")) { missing = "token_embd.weight"; return false; }

            missing = null;
            return true;
        }

        /// <summary>The target's LM head, which the drafter borrows.</summary>
        private string TargetOutputWeightName => _hasTiedOutput ? "token_embd.weight" : "output.weight";

        /// <summary>Called from <see cref="Dispose"/> in MuseGlimmerModel.cs. The
        /// drafter's weights live in the shared dictionaries and are released by
        /// <see cref="ModelBase.Dispose"/>; only the rings are ours.</summary>
        private void DisposeDFlash()
        {
            if (_dflashRingK != null)
                foreach (var t in _dflashRingK) t?.Dispose();
            if (_dflashRingV != null)
                foreach (var t in _dflashRingV) t?.Dispose();
            _dflashRingK = null;
            _dflashRingV = null;
            _hasDFlash = false;
        }

        // ====================================================================
        // IMtpSpeculativeModel
        // ====================================================================

        public bool HasMtp => HasDFlash;

        /// <summary>Speculation is profitable exactly when a drafter is loaded: the
        /// verify batch is a single ordinary multi-token Muse-Glimmer forward (the
        /// same batched-attention path a prefill chunk takes), so it amortises the
        /// 30B weight read over the whole window on every backend the model runs
        /// on.</summary>
        public bool MtpSpeculationProfitable => HasDFlash;

        /// <summary>The concatenated target-layer input residuals the encoder
        /// consumes: TargetLayerIds.Length * hidden (5 * 6656 = 33280), NOT the
        /// model's hidden size.</summary>
        public int MtpHiddenSize => _dflash != null ? _dflash.FeatureSize : Config.HiddenSize;

        /// <summary>Number of DRAFTS a block produces, i.e. block_size - 1 (15):
        /// row 0 of the block is the anchor's own prediction and plain DFlash
        /// discards it (only DSpark consumes it).</summary>
        public int MtpDraftBlockSize => _hasDFlash ? _dflash.MaxDraftTokens : 0;

        /// <summary>See <see cref="DFlashPrefillChunk"/>.</summary>
        public int MtpPrefillChunkSize => DFlashPrefillChunk;

        /// <summary>The drafter is fed from the host: every prefill chunk hands its
        /// per-row features back so <see cref="MtpCatchUp"/> can fill the ring.</summary>
        public bool MtpPrefillSelfCatchUp => false;

        /// <summary>The verify batch writes correct KV for every row at its true
        /// position into the linear cache, and Muse-Glimmer holds no recurrent
        /// state, so a partial acceptance only needs a position rewind.</summary>
        public bool MtpVerifyPersistsAcceptedKv => true;

        /// <summary>
        /// Trunk forward that additionally captures, per row, the concatenated
        /// INPUT residuals of the target layers in dflash.target_layers (row stride
        /// <see cref="MtpHiddenSize"/>) and, with <paramref name="allLogitsRows"/>,
        /// LM-head logits for every row instead of only the last. Advances the KV
        /// cache exactly like Forward().
        /// </summary>
        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            RequireDFlash();
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Token batch must not be empty.", nameof(tokens));
            // No lock here: like Gemma4Model.SpecForward, the caller owns
            // model.GpuComputeLock for the whole speculative step.
            SpecForwardCore(tokens, hAllOut, logitsOut, allLogitsRows);
        }

        /// <summary>
        /// ForwardCore() with a feature-capture hook in the layer loop and an
        /// optional all-rows LM head. Kept as a separate copy of the loop (rather
        /// than a hook inside MuseGlimmerModel.ForwardCore) so the non-speculative
        /// decode path stays byte-identical and branch-free.
        /// </summary>
        private unsafe void SpecForwardCore(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int feat = _dflash.FeatureSize;
            EnsureCacheCapacity(startPos + seqLen);

            // Buffer-size contract (IMtpSpeculativeModel.SpecForward): a hidden
            // buffer too small for one row per token means the caller only wants
            // the LAST row, written to row 0.
            bool captureAll = false, captureLast = false;
            if (hAllOut != null && hAllOut.LongLength > 0)
            {
                captureAll = hAllOut.LongLength >= (long)seqLen * feat;
                captureLast = !captureAll;
                if (captureLast && hAllOut.LongLength < feat)
                {
                    throw new ArgumentException(
                        $"DFlash hidden capture buffer holds {hAllOut.LongLength} floats; one feature row needs {feat}.",
                        nameof(hAllOut));
                }
            }

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            DrainPendingVisionEmbeddings(hidden, startPos);
            ApplyInputEmbeddingNorm(hidden, seqLen);

            // Fused whole-model trunk. The kernel emits the same per-layer input
            // residuals the loop below captures by hand, so speculation runs on the
            // fast path instead of forcing the per-op forward just to observe them.
            if (TryFusedSpecForward(hidden, seqLen, startPos, hAllOut, logitsOut, allLogitsRows, captureAll, captureLast))
            {
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return;
            }

            EnsureFusedKvHostSynchronized();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                // res->t_layer_inp[l]: the residual ENTERING layer l (== the output
                // of layer l-1). Captured before the block runs.
                int slot = _dflashCaptureSlot[l];
                if (slot >= 0 && (captureAll || captureLast))
                    DFlashCaptureFeature(hidden, slot, seqLen, hAllOut, captureLast);

                hidden = TransformerBlock(hidden, l, seqLen, startPos);
                if (_backend == BackendType.Mlx && (l + 1) % MlxEvalEveryNLayers == 0
                    && l + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            string outputWeight = TargetOutputWeightName;
            if (allLogitsRows)
            {
                t0 = Stopwatch.GetTimestamp();
                Tensor logitsTensor = LinearForward(normed, outputWeight);
                _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
                normed.Dispose();

                ApplyOutputTransform(logitsTensor);

                t0 = Stopwatch.GetTimestamp();
                long need = (long)seqLen * Config.VocabSize;
                if (logitsOut == null || logitsOut.LongLength < need)
                    throw new ArgumentException($"Per-row logits need {need} floats.", nameof(logitsOut));
                float* src = GetFloatPtr(logitsTensor);
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy(src, dst, logitsOut.LongLength * 4L, need * 4L);
                _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;
                logitsTensor.Dispose();
            }
            else
            {
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
                Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
                _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
                lastHidden.Dispose();

                ApplyOutputTransform(logitsTensor);

                t0 = Stopwatch.GetTimestamp();
                _logitsBuffer = TensorToFloatArray(logitsTensor);
                _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;
                logitsTensor.Dispose();
                if (logitsOut != null)
                    Array.Copy(_logitsBuffer, logitsOut, Config.VocabSize);
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
        }

        /// <summary>
        /// SpecForward on the fused kernel. The kernel writes the captured residuals
        /// BLOCK-major (one [hidden, n_tokens] block per target layer); the
        /// IMtpSpeculativeModel contract wants them ROW-major (per token, the blocks
        /// concatenated into one FeatureSize-wide row), so they are transposed on the
        /// way out. That is 5 x 6656 x n floats of host copying against a whole trunk
        /// forward - negligible, and it keeps the contract unchanged.
        /// </summary>
        private bool TryFusedSpecForward(Tensor hidden, int seqLen, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows, bool captureAll, bool captureLast)
        {
            int nCap = _dflash.TargetLayerIds.Length;
            int hs = Config.HiddenSize;
            int feat = _dflash.FeatureSize;
            bool wantCapture = captureAll || captureLast;

            if (wantCapture)
            {
                long need = (long)nCap * hs * seqLen;
                if (_specCaptureScratch == null || _specCaptureScratch.LongLength < need)
                    _specCaptureScratch = new float[need];
            }

            float[] produced = TryFusedForward(hidden, seqLen, startPos,
                wantCapture ? _specCaptureScratch : null,
                wantCapture ? _dflash.TargetLayerIds : null,
                null, allLogitsRows, allLogitsRows ? logitsOut : null);
            if (produced == null)
                return false;

            if (wantCapture)
            {
                for (int c = 0; c < nCap; c++)
                {
                    int slot = c;                       // capture_layers order == slot order
                    long blockBase = (long)c * hs * seqLen;
                    if (captureLast)
                    {
                        Array.Copy(_specCaptureScratch, blockBase + (long)(seqLen - 1) * hs,
                            hAllOut, (long)slot * hs, hs);
                        continue;
                    }
                    for (int r = 0; r < seqLen; r++)
                    {
                        Array.Copy(_specCaptureScratch, blockBase + (long)r * hs,
                            hAllOut, (long)r * feat + (long)slot * hs, hs);
                    }
                }
            }

            if (!allLogitsRows)
            {
                _logitsBuffer = produced;
                if (logitsOut != null)
                    Array.Copy(produced, logitsOut, Config.VocabSize);
            }
            return true;
        }

        private float[] _specCaptureScratch;

        /// <summary>Copies one target layer's input residual into its column block
        /// of the caller's feature rows.</summary>
        private unsafe void DFlashCaptureFeature(Tensor hidden, int slot, int seqLen, float[] hAllOut, bool lastRowOnly)
        {
            int hs = Config.HiddenSize;
            int feat = _dflash.FeatureSize;
            long rowBytes = (long)hs * sizeof(float);
            float* src = GetFloatPtr(hidden);
            fixed (float* dst0 = hAllOut)
            {
                if (lastRowOnly)
                {
                    Buffer.MemoryCopy(src + (long)(seqLen - 1) * hs, dst0 + (long)slot * hs, rowBytes, rowBytes);
                    return;
                }
                for (int r = 0; r < seqLen; r++)
                    Buffer.MemoryCopy(src + (long)r * hs, dst0 + (long)r * feat + (long)slot * hs, rowBytes, rowBytes);
            }
        }

        /// <summary>
        /// Replays committed trunk positions through the drafter so its KV ring
        /// tracks the real context. Row k of <paramref name="hRows"/> is the feature
        /// row of the token PRECEDING tokens[k] -- i.e. of absolute position
        /// <paramref name="startPos"/> + k - 1, which is exactly the position whose
        /// drafter key it writes (hence the -1, as in DeepSeek4Model.MtpCatchUp).
        /// </summary>
        public void MtpCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            RequireDFlash();
            if (tokens == null || tokens.Length == 0 || hRows == null)
                return;
            DFlashCatchUp(hRows, tokens.Length, startPos - 1);
        }

        /// <summary>Encodes <paramref name="rows"/> feature rows whose first row is
        /// at absolute position <paramref name="firstPos"/> and writes the resulting
        /// keys/values into the ring. Rows before position 0 (the zeroed "hidden
        /// state of the token before the prompt") are skipped, and rows older than
        /// the ring modulus are dropped -- dropping them is what keeps the ring
        /// writes collision-free.</summary>
        private void DFlashCatchUp(float[] hRows, int rows, int firstPos)
        {
            if (rows <= 0)
                return;
            int skip = firstPos < 0 ? Math.Min(-firstPos, rows) : 0;
            rows -= skip;
            firstPos += skip;
            if (rows <= 0)
                return;

            int keep = Math.Min(rows, _dflashRingRows);
            int drop = rows - keep;

            DFlashEncodeAndInject(hRows, skip + drop, keep, firstPos + drop);
        }

        /// <summary>Encode + ring injection, fused into one GGML graph when the backend
        /// supports it, otherwise the per-op pair.</summary>
        private void DFlashEncodeAndInject(float[] hRows, int rowOffset, int n, int startPos)
        {
            if (TryFusedDFlashInject(hRows, rowOffset, n, startPos))
                return;

            EnsureDFlashRingHostSynchronized();
            using Tensor g = DFlashEncode(hRows, rowOffset, n);
            DFlashInjectKv(g, n, startPos);
        }

        /// <summary>
        /// Drafts one block. <paramref name="hPrev"/> holds the target features of
        /// the last FORWARDED position (<paramref name="position"/> - 1) and
        /// <paramref name="lastToken"/> the token at <paramref name="position"/>
        /// (drawn but not yet forwarded). Returns the number of tokens written to
        /// <paramref name="draftOut"/>; <paramref name="confOut"/> receives their
        /// top-1 softmax probabilities.
        /// </summary>
        public int MtpDraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            RequireDFlash();
            if (position <= 0 || draftOut == null || draftOut.Length == 0)
                return 0;

            // The drafter's own key for the last committed position, exactly like
            // the reference's ring[start_pos % win] = kv(main_x).
            DFlashEncodeAndInject(hPrev, 0, 1, position - 1);

            int b = Math.Min(_dflash.BlockSize, draftOut.Length + 1);
            return DFlashDraftBlockCore(lastToken, position, b, draftOut, confOut);
        }

        /// <summary>DFlash drafts whole blocks; the per-token entry point is never
        /// used.</summary>
        public void MtpDraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => throw new NotSupportedException("Muse-Glimmer DFlash drafts whole blocks; use MtpDraftBlock.");

        /// <summary>Pre-grows the trunk KV cache to cover the whole speculative
        /// window. The drafter's ring is fixed-size and needs nothing.</summary>
        public void MtpEnsureCapacity(int requiredSeqLen)
        {
            if (!_hasDFlash)
                return;
            EnsureCacheCapacity(requiredSeqLen);
        }

        /// <summary>No recurrent (GDN/SSM) state in Muse-Glimmer -- drafting and
        /// verifying are stateless given the KV cache.</summary>
        public void MtpSnapshotRecurrentState() { }

        /// <summary>See <see cref="MtpSnapshotRecurrentState"/>.</summary>
        public void MtpRestoreRecurrentState() { }

        /// <summary>
        /// Rewinds the trunk KV position counter after rejected speculative tokens.
        /// Rows past <paramref name="length"/> are overwritten by later writes and
        /// the causal/SWA mask never reads past the live position, so no data moves.
        /// The drafter's ring is untouched: it only ever holds committed positions,
        /// and the next catch-up rewrites the ones a rejected tail would have needed.
        /// </summary>
        public void MtpRewindCache(int length)
        {
            if (length < 0 || length > _cacheSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Rewind length {length} outside [0, {_cacheSeqLen}].");
            }
            _cacheSeqLen = length;
            _cachedSWAMaskStartPos = -1;
        }

        private void RequireDFlash()
        {
            if (!_hasDFlash)
                throw new InvalidOperationException("No DFlash drafter is loaded for this Muse-Glimmer model.");
        }

        // ====================================================================
        // PASS A -- encoder
        // ====================================================================

        /// <summary>
        /// g = rmsnorm(fc @ feat, enc.output_norm). <paramref name="rowOffset"/> is
        /// the first feature row of <paramref name="hRows"/> to consume and
        /// <paramref name="n"/> how many. Returns [n, hidden].
        /// </summary>
        private unsafe Tensor DFlashEncode(float[] hRows, int rowOffset, int n)
        {
            int feat = _dflash.FeatureSize;
            long need = (long)(rowOffset + n) * feat;
            if (hRows == null || hRows.LongLength < need)
            {
                throw new ArgumentException(
                    $"DFlash encoder needs {need} feature floats but got {hRows?.LongLength ?? 0}.", nameof(hRows));
            }

            var featTensor = new Tensor(_allocator, DType.Float32, n, feat);
            long bytes = (long)n * feat * sizeof(float);
            float* dst = GetFloatPtr(featTensor);
            fixed (float* src = &hRows[(long)rowOffset * feat])
                Buffer.MemoryCopy(src, dst, bytes, bytes);
            InvalidateTensorDeviceCache(featTensor);

            Tensor g = LinearForward(featTensor, DFlashConfig.WeightPrefix + "fc.weight");
            featTensor.Dispose();

            Ops.RMSNorm(g, g, _weights[DFlashConfig.WeightPrefix + "enc.output_norm.weight"], null, _dflash.Eps);
            return g;
        }

        // ====================================================================
        // PASS B -- KV injection
        // ====================================================================

        /// <summary>
        /// Writes the drafter's per-position keys/values for <paramref name="n"/>
        /// encoder rows starting at absolute position <paramref name="startPos"/>.
        /// K is head-normed and NeoX-RoPE'd at the TARGET position; V gets neither.
        /// </summary>
        private void DFlashInjectKv(Tensor g, int n, int startPos)
        {
            var cfg = _dflash;
            int kvHeads = cfg.NumKVHeads, hd = cfg.HeadDim;

            int[] positions = new int[n];
            for (int i = 0; i < n; i++)
                positions[i] = startPos + i;
            _dflashRingFilled = Math.Max(_dflashRingFilled, startPos + n);

            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];

                Tensor k = LinearForward(g, names[DfAttnK]);
                Tensor v = LinearForward(g, names[DfAttnV]);

                k = DFlashHeadNorm(k, _weights[names[DfAttnKNorm]], kvHeads, n, hd);
                k = DFlashRoPE(k, kvHeads, n, hd, positions);

                using (Tensor kHeads = ReshapeToHeads(k, kvHeads, n, hd))
                    DFlashRingWrite(_dflashRingK[il], kHeads, startPos, n);
                using (Tensor vHeads = ReshapeToHeads(v, kvHeads, n, hd))
                    DFlashRingWrite(_dflashRingV[il], vHeads, startPos, n);

                k.Dispose();
                v.Dispose();
            }
        }

        /// <summary>Scatters a head-first [kvHeads, n, headDim] tensor into the ring
        /// at (startPos + i) % ringRows, splitting the write at the wrap.</summary>
        private void DFlashRingWrite(Tensor ring, Tensor headFirst, int startPos, int n)
        {
            int rows = _dflashRingRows;
            int keep = Math.Min(n, rows);
            int first = n - keep;                 // older rows would be overwritten anyway
            int done = 0;
            while (done < keep)
            {
                int slot = (startPos + first + done) % rows;
                int len = Math.Min(keep - done, rows - slot);
                using var src = headFirst.Narrow(1, first + done, len);
                CopyToCache(ring, src, slot, len);
                done += len;
            }
        }

        // ====================================================================
        // PASS C -- block draft
        // ====================================================================

        /// <summary>
        /// Runs [anchor, MASK x (b-1)] at positions p..p+b-1 through the drafter and
        /// fills <paramref name="draftOut"/> / <paramref name="confOut"/> from rows
        /// 1..b-1. Returns b-1.
        /// </summary>
        private unsafe int DFlashDraftBlockCore(int anchorToken, int position, int b, int[] draftOut, float[] confOut)
        {
            int fused = TryFusedDFlashDraftBlock(anchorToken, position, b, draftOut, confOut);
            if (fused >= 0)
                return fused;

            EnsureDFlashRingHostSynchronized();
            var cfg = _dflash;
            int heads = cfg.NumHeads, hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads;
            float eps = cfg.Eps;

            int[] ids = new int[b];
            int[] positions = new int[b];
            for (int i = 0; i < b; i++)
            {
                ids[i] = i == 0 ? anchorToken : cfg.MaskTokenId;
                positions[i] = position + i;
            }

            // llama.cpp's dflash graph feeds build_inp_embd straight in: no
            // embedding scale, and (unlike the Muse-Glimmer trunk) no weightless
            // RMSNorm over the embeddings.
            Tensor inpL = Embedding(ids);

            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];

                Tensor h = RMSNormWithEps(inpL, names[DfAttnNorm], eps);
                Tensor q = LinearForward(h, names[DfAttnQ]);
                Tensor k = LinearForward(h, names[DfAttnK]);
                Tensor v = LinearForward(h, names[DfAttnV]);
                h.Dispose();

                q = DFlashHeadNorm(q, _weights[names[DfAttnQNorm]], heads, b, hd);
                k = DFlashHeadNorm(k, _weights[names[DfAttnKNorm]], kvHeads, b, hd);
                q = DFlashRoPE(q, heads, b, hd, positions);
                k = DFlashRoPE(k, kvHeads, b, hd, positions);
                Ops.Mul(q, q, 1f / MathF.Sqrt(hd));

                Tensor attn = DFlashBlockAttention(il, q, k, v, position, b);
                q.Dispose();
                k.Dispose();
                v.Dispose();

                Tensor attnOut = LinearForward(attn, names[DfAttnOutput]);
                attn.Dispose();

                Ops.Add(attnOut, attnOut, inpL);          // ffn_inp = attn + inpL
                inpL.Dispose();

                using (Tensor ffnIn = RMSNormWithEps(attnOut, names[DfFfnNorm], eps))
                using (Tensor ffnOut = DFlashSwiGLU(ffnIn, names[DfFfnGate], names[DfFfnUp], names[DfFfnDown]))
                {
                    Ops.Add(attnOut, attnOut, ffnOut);    // inpL = ffn + ffn_inp
                }
                inpL = attnOut;
            }

            Tensor cur = RMSNormWithEps(inpL, DFlashConfig.WeightPrefix + "output_norm.weight", eps);
            inpL.Dispose();

            // The TARGET's LM head, with NEITHER logit_scale NOR the tanh softcap:
            // llama.cpp's dflash graph ends at build_lora_mm(output, cur).
            Tensor logits = LinearForward(cur, TargetOutputWeightName);
            cur.Dispose();

            // Softmax on the backend, then a max scan per row: argmax is invariant
            // under softmax, and the winning probability IS the confidence the
            // executor multiplies cumulatively (a zero there drafts nothing).
            Ops.Softmax(logits, logits);

            int vocab = Config.VocabSize;
            int n = b - 1;
            float* lp = GetFloatPtr(logits);
            for (int i = 0; i < n; i++)
            {
                // Row 0 is the anchor's own prediction; plain DFlash discards it.
                float* row = lp + (long)(i + 1) * vocab;
                int best = ArgmaxRow(row, vocab, out float prob);
                draftOut[i] = best;
                if (confOut != null && i < confOut.Length)
                    confOut[i] = prob;
            }
            logits.Dispose();
            return n;
        }

        private static unsafe int ArgmaxRow(float* row, int n, out float best)
        {
            int bestIdx = 0;
            float bestVal = row[0];
            for (int i = 1; i < n; i++)
            {
                float v = row[i];
                if (v > bestVal)
                {
                    bestVal = v;
                    bestIdx = i;
                }
            }
            best = bestVal;
            return bestIdx;
        }

        /// <summary>
        /// One draft layer's attention: the b block queries attend
        /// [ring window | this block's own b keys], NON-CAUSALLY inside the block
        /// (llama_set_causal_attn(ctx_dft, false)) and sliding-window masked against
        /// the ring (a cached key at p0 is masked from a query at p1 when
        /// p1 - p0 &gt;= n_swa).
        /// </summary>
        private unsafe Tensor DFlashBlockAttention(int il, Tensor q, Tensor k, Tensor v, int position, int b)
        {
            var cfg = _dflash;
            int kvHeads = cfg.NumKVHeads, hd = cfg.HeadDim, heads = cfg.NumHeads;
            int groupSize = heads / kvHeads;
            int rings = _dflashRingRows;

            // The FIRST query (position p) sees cached keys down to p - (n_swa - 1);
            // later queries in the block see a strict subset, masked below.
            int winStart = Math.Max(0, position - (cfg.SlidingWindow - 1));
            int w = position - winStart;
            int total = w + b;

            var gk = new Tensor(_allocator, DType.Float32, kvHeads, total, hd);
            var gv = new Tensor(_allocator, DType.Float32, kvHeads, total, hd);

            int done = 0;
            while (done < w)
            {
                int slot = (winStart + done) % rings;
                int len = Math.Min(w - done, rings - slot);
                using (var srcK = _dflashRingK[il].Narrow(1, slot, len))
                using (var dstK = gk.Narrow(1, done, len))
                    Ops.Copy(dstK, srcK);
                using (var srcV = _dflashRingV[il].Narrow(1, slot, len))
                using (var dstV = gv.Narrow(1, done, len))
                    Ops.Copy(dstV, srcV);
                done += len;
            }

            using (Tensor kHeads = ReshapeToHeads(k, kvHeads, b, hd))
            using (var dstBlockK = gk.Narrow(1, w, b))
                Ops.Copy(dstBlockK, kHeads);
            using (Tensor vHeads = ReshapeToHeads(v, kvHeads, b, hd))
            using (var dstBlockV = gv.Narrow(1, w, b))
                Ops.Copy(dstBlockV, vHeads);

            // GQA without materializing the expanded K/V: a contiguous head-first
            // [heads, b, hd] query tensor reinterprets exactly as
            // [kvHeads, groupSize*b, hd] (heads 4g..4g+3 are adjacent blocks of
            // b*hd), so one batched GEMM per kv head serves its whole query group.
            // ExpandKVHeads would instead allocate heads*total*hd floats per layer
            // per draft (33 MB here) purely to repeat rows.
            Tensor qHeads = ReshapeToHeads(q, heads, b, hd);
            Tensor scores;
            using (Tensor qGrouped = qHeads.View(kvHeads, (long)groupSize * b, hd))
            using (Tensor kT = gk.Transpose(1, 2))
            {
                scores = new Tensor(_allocator, DType.Float32, kvHeads, (long)groupSize * b, total);
                Ops.AddmmBatch(scores, 0, scores, 1f, qGrouped, kT);
            }
            qHeads.Dispose();
            gk.Dispose();

            DFlashApplyWindowMask(scores, b, groupSize, kvHeads, w, total, position, winStart);
            Ops.Softmax(scores, scores);

            var attnGrouped = new Tensor(_allocator, DType.Float32, kvHeads, (long)groupSize * b, hd);
            Ops.AddmmBatch(attnGrouped, 0, attnGrouped, 1f, scores, gv);
            scores.Dispose();
            gv.Dispose();

            Tensor attn;
            using (Tensor attnHeads = attnGrouped.View(heads, b, hd))
                attn = ReshapeFromHeads(attnHeads, heads, b, hd);
            attnGrouped.Dispose();
            return attn;
        }

        /// <summary>
        /// Masks the cached (ring) columns a query cannot see. Query row j of kv
        /// group g belongs to block slot s = j % b (see the grouping comment in
        /// <see cref="DFlashBlockAttention"/>), i.e. absolute position
        /// <paramref name="position"/> + s; cached column c holds position
        /// <paramref name="winStart"/> + c and is masked when
        /// (position + s) - (winStart + c) &gt;= n_swa. The block's own b columns
        /// are never masked -- attention inside the block is non-causal.
        /// </summary>
        private unsafe void DFlashApplyWindowMask(Tensor scores, int b, int groupSize, int kvHeads,
            int w, int total, int position, int winStart)
        {
            if (w <= 0)
                return;

            int swa = _dflash.SlidingWindow;
            Span<int> widths = stackalloc int[b];
            bool any = false;
            for (int s = 0; s < b; s++)
            {
                int width = position + s - swa - winStart + 1;
                if (width < 0) width = 0;
                if (width > w) width = w;
                widths[s] = width;
                any |= width > 0;
            }
            if (!any)
                return;

            float* sp = GetFloatPtr(scores);
            int rowsPerGroup = groupSize * b;
            for (int g = 0; g < kvHeads; g++)
            {
                float* groupScores = sp + (long)g * rowsPerGroup * total;
                for (int j = 0; j < rowsPerGroup; j++)
                {
                    int width = widths[j % b];
                    if (width > 0)
                        new Span<float>(groupScores + (long)j * total, width).Fill(float.NegativeInfinity);
                }
            }
            InvalidateTensorDeviceCache(scores);
        }

        // ====================================================================
        // small shared pieces
        // ====================================================================

        /// <summary>Per-head RMSNorm over a [rows, numHeads*headDim] tensor, with
        /// the drafter's own epsilon. Consumes <paramref name="data"/>. Same shape
        /// as MuseGlimmerModel.ApplyBatchRMSNorm, but takes the weight tensor and
        /// eps directly (the trunk helper hardcodes Config.Eps).</summary>
        private Tensor DFlashHeadNorm(Tensor data, Tensor alpha, int numHeads, int rows, int headDim)
        {
            using var reshaped = data.View((long)rows * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, _dflash.Eps);
            data.Dispose();
            Tensor flat = normed.View(rows, (long)numHeads * headDim);
            normed.Dispose();
            return flat;
        }

        /// <summary>
        /// NeoX-flavour RoPE (split halves) over a [rows, numHeads*headDim] tensor
        /// with an explicit per-row position. llama.cpp maps LLM_ARCH_DFLASH to
        /// LLAMA_ROPE_TYPE_NEOX, so this is mode 2 -- NOT the interleaved-pair
        /// (mode 0) RoPE MuseGlimmerModel.ApplyRoPEPrefill uses for the trunk.
        /// Consumes <paramref name="data"/>.
        /// </summary>
        private Tensor DFlashRoPE(Tensor data, int numHeads, int rows, int headDim, int[] positions)
        {
            int totalRows = rows * numHeads;
            int[] rowPositions = new int[totalRows];
            for (int s = 0; s < rows; s++)
                for (int h = 0; h < numHeads; h++)
                    rowPositions[s * numHeads + h] = positions[s];
            using var posTensor = CreateIntTensorOn(data.Storage.Allocator, rowPositions, totalRows);

            using var reshaped = data.View(1, rows, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, DFlashConfig.RopeTypeNeoX, 0,
                _dflash.RopeBase, 1.0f,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(rows, (long)numHeads * headDim);
            result.Dispose();
            return flat;
        }

        /// <summary>silu(gate(x)) * up(x) -> down. The drafter ships gate and up as
        /// SEPARATE tensors (ModelBase.FuseGateUpWeights only fuses the target's
        /// "blk.{l}." names), so this cannot use ModelBase.FFN.</summary>
        private Tensor DFlashSwiGLU(Tensor input, string gateName, string upName, string downName)
        {
            Tensor gate = LinearForward(input, gateName);
            Tensor up = LinearForward(input, upName);
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForward(gate, downName);
            gate.Dispose();
            return down;
        }
    }
}
