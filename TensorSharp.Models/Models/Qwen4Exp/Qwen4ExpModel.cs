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
using TensorSharp.Core;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3.8-Flash-Next - GGUF <c>general.architecture = "qwen4exp"</c>, a preview
    /// of the Qwen4 architecture. 125 B total / 6 B active.
    ///
    /// Four things make it different from Qwen 3.5/3.6/3.8, and all four are load
    /// bearing rather than cosmetic:
    ///
    /// 1. <b>Hyper-connections.</b> The residual stream is <c>hc</c> (4) times wide.
    ///    Every block reads it through a learned gated mixer that collapses the four
    ///    streams to one, and writes back through a learned per-stream scatter. There
    ///    is no separate output norm - the final mixer IS the output norm. Same idea
    ///    as DeepSeek V4's, except the gate is produced by a low-rank pair
    ///    (hc_dim -> 320 -> hc_dim) rather than a full-rank matrix.
    ///
    /// 2. <b>PLE n-gram embeddings.</b> One layer (layer 1 in the shipped file) adds a
    ///    lookup into a ~320 M row table addressed by a hash of the token's bigram and
    ///    trigram context. The table is 51 B parameters on its own - most of the file -
    ///    and is read one row per head per token, so it belongs in host memory.
    ///
    /// 3. <b>Qwen Sparse Attention.</b> The full-attention layers (every 4th) score
    ///    blocks of <c>compress_ratio</c> KV cells with a small indexer, then attend
    ///    densely to the best <c>indexer_top_k</c> of them. Below the budget the result
    ///    is exactly dense.
    ///
    /// 4. <b>Gated DeltaNet everywhere else</b>, which is the same recurrence Qwen 3.5
    ///    uses, so it reuses <see cref="GgmlBasicOps.GatedDeltaNetChunked"/>.
    ///
    /// The MoE is conventional apart from its size: 512 experts, top 10, plus a gated
    /// shared expert.
    ///
    /// This is the correctness-first managed path. Text only: the vision tower and
    /// MTP block are not wired up yet, and IMRoPE degenerates to NEOX RoPE when every
    /// position component is equal, which is the case for text.
    /// </summary>
    public partial class Qwen4ExpModel : ModelBase
    {
        public const string ArchitectureId = "qwen4exp";

        // ---- hyper-connections ----
        private int _hc;             // residual stream multiplicity (4)
        private int _hcDim;          // _hc * hidden
        private int _hcLowRank;      // gate bottleneck (320)

        // ---- layer typing ----
        private bool[] _isRecurrent; // GDN rather than full attention
        private bool[] _isPle;
        private int[] _compressRatios;

        // ---- QSA indexer ----
        private int _indexerHeads;
        private int _indexerHeadDim;
        private int _indexerTopK;

        // ---- PLE ----
        private int _pleNgram;
        private int _pleHeadsPerNgram;
        private int _pleConvKernel;
        private int _pleHeads;       // (ngram - 1) * heads_per_ngram
        private int _pleHeadDim;
        private int _pleEosTokenId;
        private int _pleImageTokenId;
        private ulong[] _pleMultipliers;
        private ulong[] _pleHeadOffsets;
        private ulong[] _pleHeadVocabSizes;

        // ---- GDN ----
        private int _convKernel;
        private int _headKDim, _headVDim;
        private int _numKHeads, _numVHeads;

        // ---- MoE ----
        private int _numExperts, _numExpertsUsed;
        private int _expertFf, _sharedFf;

        // ---- RoPE ----
        private int _ropeDimCount;
        private int[] _ropeSections;

        private float _attnScale;

        public Qwen4ExpModel(string ggufPath, BackendType backend, int tpDegree = 1,
            ITensorParallelGroup tpGroup = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            Config = new ModelConfig { Architecture = ArchitectureId };
            ParseBaseConfig();
            ParseQwen4ExpConfig();
            ParseTokenizer();

            Console.WriteLine($"Model: {ArchitectureId}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"  hyper-connections: count={_hc} lowRank={_hcLowRank} (residual is {_hcDim} wide)");
            Console.WriteLine($"  experts={_numExperts} used={_numExpertsUsed} ff={_expertFf} sharedFf={_sharedFf}");
            Console.WriteLine($"  GDN layers={CountTrue(_isRecurrent)}/{Config.NumLayers}, " +
                $"QSA indexer heads={_indexerHeads} dim={_indexerHeadDim} topK={_indexerTopK}");
            if (_pleHeads > 0)
                Console.WriteLine($"  PLE: layers=[{string.Join(",", PleLayerList())}] ngram={_pleNgram} " +
                    $"heads={_pleHeads} headDim={_pleHeadDim}");

            LoadWeights();
            VerifyQwen4ExpTensors();
            PrepareCudaQuantizedWeightsForInference();

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            InitCaches(initialCacheLength, maxContextLength);
        }

        private static int CountTrue(bool[] a)
        {
            int n = 0;
            foreach (bool b in a) if (b) n++;
            return n;
        }

        private List<int> PleLayerList()
        {
            var l = new List<int>();
            for (int i = 0; i < _isPle.Length; i++) if (_isPle[i]) l.Add(i);
            return l;
        }

        private void ParseQwen4ExpConfig()
        {
            const string a = ArchitectureId;
            int nLayer = Config.NumLayers;

            Config.NumKVHeads = (int)_gguf.GetUint32($"{a}.attention.head_count_kv");

            // Hyper-connections. Both keys are required: without the count there is no
            // residual shape and without the low rank there is no gate, and guessing
            // either produces a model that loads and emits noise.
            _hc = (int)_gguf.GetUint32($"{a}.hyper_connection.count");
            _hcLowRank = (int)_gguf.GetUint32($"{a}.hyper_connection.low_rank");
            if (_hc <= 0 || _hcLowRank <= 0)
            {
                throw new NotSupportedException(
                    $"qwen4exp needs {a}.hyper_connection.count and .low_rank; got {_hc} and {_hcLowRank}.");
            }
            _hcDim = _hc * Config.HiddenSize;

            // MoE
            _numExperts = (int)_gguf.GetUint32($"{a}.expert_count");
            _numExpertsUsed = (int)_gguf.GetUint32($"{a}.expert_used_count");
            _expertFf = (int)_gguf.GetUint32($"{a}.expert_feed_forward_length");
            _sharedFf = (int)_gguf.GetUint32($"{a}.expert_shared_feed_forward_length", (uint)_expertFf);

            // Gated DeltaNet, named with the SSM keys it shares with Qwen 3.5.
            _convKernel = (int)_gguf.GetUint32($"{a}.ssm.conv_kernel");
            _headKDim = (int)_gguf.GetUint32($"{a}.ssm.state_size");
            _headVDim = _headKDim;
            _numKHeads = (int)_gguf.GetUint32($"{a}.ssm.group_count");
            _numVHeads = (int)_gguf.GetUint32($"{a}.ssm.time_step_rank");

            // RoPE. dimension_count is the PARTIAL rotary width (64 of a 256-wide head).
            _ropeDimCount = (int)_gguf.GetUint32($"{a}.rope.dimension_count", (uint)Config.HeadDim);
            _ropeSections = _gguf.GetInt32Array($"{a}.rope.dimension_sections") ?? new[] { 0, 0, 0, 0 };

            _attnScale = _gguf.GetFloat32($"{a}.attention.scale", 0f);
            if (_attnScale == 0f)
                _attnScale = 1.0f / MathF.Sqrt(Config.HeadDim);

            // Indexer (QSA). Absent keys mean a dense model, which still runs.
            _indexerHeads = (int)_gguf.GetUint32($"{a}.attention.indexer.head_count");
            _indexerHeadDim = (int)_gguf.GetUint32($"{a}.attention.indexer.key_length");
            _indexerTopK = (int)_gguf.GetUint32($"{a}.attention.indexer.top_k");

            _compressRatios = new int[nLayer];
            int[] ratios = _gguf.GetInt32Array($"{a}.attention.compress_ratios");
            if (ratios != null)
            {
                for (int i = 0; i < nLayer && i < ratios.Length; i++)
                    _compressRatios[i] = ratios[i];
            }

            // Layer typing: linear (GDN) everywhere except every full_attention_interval-th.
            _isRecurrent = new bool[nLayer];
            uint[] recr = _gguf.GetUint32Array($"{a}.attention.recurrent_layers");
            if (recr != null && recr.Length >= nLayer)
            {
                for (int i = 0; i < nLayer; i++) _isRecurrent[i] = recr[i] != 0;
            }
            else
            {
                int interval = (int)_gguf.GetUint32($"{a}.full_attention_interval", 4);
                if (interval <= 0) interval = 4;
                for (int i = 0; i < nLayer; i++) _isRecurrent[i] = ((i + 1) % interval) != 0;
            }

            // PLE. The whole group is optional; when the layer list is absent every
            // field stays zero and the model is a plain hyper-connection stack.
            _isPle = new bool[nLayer];
            int[] pleLayers = _gguf.GetInt32Array($"{a}.ple.layers");
            if (pleLayers != null && pleLayers.Length > 0)
            {
                foreach (int il in pleLayers)
                {
                    if (il < 0 || il >= nLayer)
                        throw new NotSupportedException($"qwen4exp {a}.ple.layers names layer {il}, outside 0..{nLayer - 1}.");
                    _isPle[il] = true;
                }

                _pleNgram = (int)_gguf.GetUint32($"{a}.ple.ngram_size");
                _pleHeadsPerNgram = (int)_gguf.GetUint32($"{a}.ple.heads_per_ngram");
                _pleConvKernel = (int)_gguf.GetUint32($"{a}.ple.conv_kernel");
                _pleEosTokenId = (int)_gguf.GetUint32($"{a}.ple.eos_token_id");
                _pleImageTokenId = (int)_gguf.GetUint32($"{a}.ple.image_token_id");
                _pleHeadDim = (int)_gguf.GetUint32($"{a}.embedding_length_per_layer_input");

                _pleHeads = (_pleNgram - 1) * _pleHeadsPerNgram;
                _pleMultipliers = _gguf.GetUint64Array($"{a}.ple.layer_multipliers");
                _pleHeadOffsets = _gguf.GetUint64Array($"{a}.ple.head_offsets");
                _pleHeadVocabSizes = _gguf.GetUint64Array($"{a}.ple.head_vocab_sizes");

                if (_pleNgram < 2 || _pleHeads <= 0 || _pleHeadDim <= 0
                    || _pleMultipliers == null || _pleMultipliers.Length < _pleNgram
                    || _pleHeadOffsets == null || _pleHeadOffsets.Length < _pleHeads
                    || _pleHeadVocabSizes == null || _pleHeadVocabSizes.Length < _pleHeads)
                {
                    throw new NotSupportedException(
                        "qwen4exp declares PLE layers but its n-gram hash constants are missing or short: "
                        + $"ngram={_pleNgram} heads={_pleHeads} headDim={_pleHeadDim} "
                        + $"multipliers={_pleMultipliers?.Length ?? 0} offsets={_pleHeadOffsets?.Length ?? 0} "
                        + $"vocabSizes={_pleHeadVocabSizes?.Length ?? 0}.");
                }

                // The gather concatenates the heads into one hidden-sized row, so the
                // two have to agree or every PLE projection is fed a wrong-width input.
                if (_pleHeads * _pleHeadDim != Config.HiddenSize)
                {
                    throw new NotSupportedException(
                        $"qwen4exp PLE heads*headDim ({_pleHeads}*{_pleHeadDim}) must equal the hidden size "
                        + $"({Config.HiddenSize}).");
                }
            }
        }

        // Names are checked up front rather than on first use: a missing hyper-connection
        // tensor otherwise surfaces as a null dereference 40 layers into the first
        // forward, long after the useful context is gone.
        private void VerifyQwen4ExpTensors()
        {
            var missing = new List<string>();
            // Stacked expert tensors land in _stackedExpertWeights rather than the
            // plain weight maps, so all three have to be consulted.
            void Need(string n)
            {
                if (!_weights.ContainsKey(n) && !_quantWeights.ContainsKey(n)
                    && !_stackedExpertWeights.ContainsKey(n))
                {
                    missing.Add(n);
                }
            }

            Need("token_embd.weight");
            Need("output_hc_norm.weight");
            Need("output_hc_down.weight");
            Need("output_hc_up.weight");
            if (_pleHeads > 0) Need("per_layer_token_embd.weight");

            for (int il = 0; il < Config.NumLayers; il++)
            {
                Need($"blk.{il}.hc_attn_norm.weight");
                Need($"blk.{il}.hc_attn_down.weight");
                Need($"blk.{il}.hc_attn_up.weight");
                Need($"blk.{il}.hc_attn_inject.weight");
                Need($"blk.{il}.hc_ffn_norm.weight");
                Need($"blk.{il}.hc_ffn_down.weight");
                Need($"blk.{il}.hc_ffn_up.weight");
                Need($"blk.{il}.hc_ffn_inject.weight");

                if (_isRecurrent[il])
                {
                    Need($"blk.{il}.attn_qkv.weight");
                    Need($"blk.{il}.attn_gate.weight");
                    Need($"blk.{il}.ssm_conv1d.weight");
                    Need($"blk.{il}.ssm_norm.weight");
                    Need($"blk.{il}.ssm_out.weight");
                }
                else
                {
                    Need($"blk.{il}.attn_q.weight");
                    Need($"blk.{il}.attn_k.weight");
                    Need($"blk.{il}.attn_v.weight");
                    Need($"blk.{il}.attn_output.weight");
                    Need($"blk.{il}.attn_q_norm.weight");
                    Need($"blk.{il}.attn_k_norm.weight");
                    if (UsesQsa(il))
                    {
                        Need($"blk.{il}.indexer.q_proj.weight");
                        Need($"blk.{il}.indexer.k_proj.weight");
                        Need($"blk.{il}.indexer.q_norm.weight");
                        Need($"blk.{il}.indexer.k_norm.weight");
                    }
                }

                if (_isPle[il])
                {
                    Need($"blk.{il}.ple_key.weight");
                    Need($"blk.{il}.ple_value.weight");
                    Need($"blk.{il}.ple_norm_key.weight");
                    Need($"blk.{il}.ple_norm_query.weight");
                    Need($"blk.{il}.ple_norm_conv.weight");
                    Need($"blk.{il}.ple_conv1d.weight");
                }

                Need($"blk.{il}.ffn_gate_inp.weight");
                Need($"blk.{il}.ffn_down_exps.weight");
                // llama.cpp accepts either a fused ffn_gate_up_exps or the separate
                // pair, so which one a given quant ships is not knowable up front.
                bool hasSplit = _stackedExpertWeights.ContainsKey($"blk.{il}.ffn_gate_exps.weight")
                                && _stackedExpertWeights.ContainsKey($"blk.{il}.ffn_up_exps.weight");
                bool hasFused = _stackedExpertWeights.ContainsKey($"blk.{il}.ffn_gate_up_exps.weight");
                if (!hasSplit && !hasFused)
                    missing.Add($"blk.{il}.ffn_gate_exps.weight (or ffn_gate_up_exps.weight)");
                else if (!hasSplit && il == 0)
                    _fusedGateUpExperts = true;
                Need($"blk.{il}.ffn_gate_inp_shexp.weight");
                Need($"blk.{il}.ffn_gate_shexp.weight");
                Need($"blk.{il}.ffn_up_shexp.weight");
                Need($"blk.{il}.ffn_down_shexp.weight");
            }

            if (missing.Count > 0)
            {
                string head = string.Join(", ", missing.GetRange(0, Math.Min(8, missing.Count)));
                throw new NotSupportedException(
                    $"qwen4exp GGUF is missing {missing.Count} expected tensor(s): {head}"
                    + (missing.Count > 8 ? ", ..." : "") + ".");
            }
        }

        /// <summary>
        /// Keep the PLE n-gram table OUT of VRAM.
        ///
        /// It is the single largest tensor in the file - ~320 M rows, tens of GB, most
        /// of the checkpoint - and the forward never multiplies by it: every use is a
        /// gather of <c>ple_n_heads</c> scattered rows per token, which
        /// <see cref="GatherPleRows"/> serves by dequantizing from the retained host
        /// copy. Preloading it spent the VRAM that the rest of the model and the KV
        /// cache need, and on a 96 GB card that was the difference between running and
        /// failing to allocate 0.4 MiB.
        /// </summary>
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
        {
            if (string.Equals(weightName, "per_layer_token_embd.weight", StringComparison.Ordinal))
                return false;

            // The routed experts reach the device ONCE, as the stacked tensor the
            // batched mul_mat_id kernel binds. ModelBase also exposes each expert as
            // its own 2D view, and preloading those put a second full copy of all 512
            // experts per layer in VRAM - 47 GB of it - which is what made the batched
            // MoE fail to allocate its graph. The host views stay mapped for the
            // portable per-expert path on backends without the stacked kernel.
            if (_stackedExpertMemberNames.Contains(weightName))
                return false;

            return base.ShouldPreloadCudaQuantWeightToDevice(weightName);
        }

        /// <summary>A full-attention layer runs QSA when it has both an indexer and a
        /// block size; either missing means plain dense attention.</summary>
        /// <summary>The GGUF stacks gate and up into one expert tensor rather than
        /// shipping them separately.</summary>
        private bool _fusedGateUpExperts;

        private bool UsesQsa(int il)
            => !_isRecurrent[il] && _indexerHeads > 0 && _indexerHeadDim > 0
               && _compressRatios != null && il < _compressRatios.Length && _compressRatios[il] > 0;
    }
}
