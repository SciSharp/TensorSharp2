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
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Hyper-parameters of a DFlash speculative drafter GGUF
    /// (general.architecture = "dflash"), the block drafter that ships alongside a
    /// Muse-Glimmer target. Parsed from the drafter file only; every tensor the
    /// drafter needs beyond its own blocks (token_embd, output) is borrowed from
    /// the target model.
    ///
    /// DFlash runs three passes (llama.cpp src/models/dflash.cpp):
    ///   A. ENCODE  - the target's per-layer INPUT residuals at
    ///                <see cref="TargetLayerIds"/>, concatenated into one
    ///                <see cref="FeatureSize"/>-wide row per committed position,
    ///                projected by fc.weight and RMS-normed by enc.output_norm.
    ///   B. INJECT  - that row feeds attn_k / attn_v of every draft layer; the
    ///                keys get a per-head RMSNorm + NeoX RoPE at the target
    ///                position and land in the drafter's own KV ring. No Q, no
    ///                attention, no FFN.
    ///   C. DRAFT   - [anchor, MASK x (block_size-1)] runs the 5 draft blocks
    ///                non-causally over [ring window | the block's own keys] and
    ///                the TARGET's LM head turns the result into
    ///                <see cref="BlockSize"/> rows of logits. Row 0 is the
    ///                anchor's own prediction and is discarded.
    /// </summary>
    public sealed class DFlashConfig
    {
        /// <summary>general.architecture of a DFlash drafter GGUF.</summary>
        public const string ArchName = "dflash";

        /// <summary>Name prefix under which the drafter's tensors are merged into
        /// the target model's shared weight dictionaries (mirrors Gemma 4's
        /// "mtp." prefix).</summary>
        public const string WeightPrefix = "dflash.";

        /// <summary>ggml GGML_ROPE_TYPE_NEOX. llama.cpp maps LLM_ARCH_DFLASH to
        /// LLAMA_ROPE_TYPE_NEOX, i.e. the split-halves flavour -- NOT the
        /// interleaved-pair (NORM, mode 0) RoPE the Muse-Glimmer target uses.</summary>
        public const int RopeTypeNeoX = 2;

        /// <summary>dflash.block_count (draft decoder blocks).</summary>
        public int NumLayers { get; private set; }

        /// <summary>dflash.embedding_length; must equal the target's hidden size.</summary>
        public int HiddenSize { get; private set; }

        /// <summary>dflash.feed_forward_length.</summary>
        public int IntermediateSize { get; private set; }

        /// <summary>dflash.attention.head_count (draft query heads).</summary>
        public int NumHeads { get; private set; }

        /// <summary>dflash.attention.head_count_kv. Note this is the DRAFTER's own
        /// GQA ratio (8 kv heads for Muse-Glimmer's dflash) and is unrelated to the
        /// target's (2) -- the drafter keeps a KV cache of its own.</summary>
        public int NumKVHeads { get; private set; }

        /// <summary>dflash.attention.key_length == value_length.</summary>
        public int HeadDim { get; private set; }

        /// <summary>dflash.rope.freq_base.</summary>
        public float RopeBase { get; private set; }

        /// <summary>dflash.attention.layer_norm_rms_epsilon. Every RMSNorm in the
        /// drafter (encoder, per-head Q/K, pre-attention, pre-FFN, final) uses it.</summary>
        public float Eps { get; private set; }

        /// <summary>dflash.block_size: the WIDTH of one draft block, including the
        /// anchor row. The number of usable drafts is <see cref="MaxDraftTokens"/>.</summary>
        public int BlockSize { get; private set; }

        /// <summary>dflash.target_layers: the target layers whose INPUT residual
        /// feeds the encoder, in the order they are concatenated. The converter
        /// already shifted these by +1, so id L means "the hidden state entering
        /// 0-based target layer L" == "the output of target layer L-1".</summary>
        public int[] TargetLayerIds { get; private set; } = Array.Empty<int>();

        /// <summary>dflash.attention.sliding_window (the drafter's own SWA span,
        /// applied against its KV ring).</summary>
        public int SlidingWindow { get; private set; }

        /// <summary>dflash.attention.sliding_window_pattern, one bool per draft
        /// layer (all true for the shipped drafter).</summary>
        public bool[] SwaPattern { get; private set; } = Array.Empty<bool>();

        /// <summary>tokenizer.ggml.mask_token_id: the id filling every draft slot
        /// past the anchor.</summary>
        public int MaskTokenId { get; private set; } = -1;

        /// <summary>Width of one encoder input row = TargetLayerIds.Length * HiddenSize
        /// (33280 for the Muse-Glimmer drafter). This is what the model reports as
        /// ISpeculativeModel.SpecFeatureSize.</summary>
        public int FeatureSize => TargetLayerIds.Length * HiddenSize;

        /// <summary>Tokens a block actually proposes: the block is
        /// [anchor, MASK x (BlockSize-1)] and row 0 (the anchor's own prediction)
        /// is discarded by plain DFlash.</summary>
        public int MaxDraftTokens => BlockSize - 1;

        /// <summary>Rows of the drafter's KV ring: the SWA span plus one whole
        /// block plus the anchor, rounded up to 32 so a wrapped write never aliases
        /// a row the same pass still reads (same sizing rule as the DSpark ring in
        /// Dsv4CudaEngine.Dspark.cs).</summary>
        public int RingRows => Pad(SlidingWindow + BlockSize + 1, 32);

        private static int Pad(int value, int alignment)
            => ((value + alignment - 1) / alignment) * alignment;

        /// <summary>Reads every "dflash.*" key from an opened drafter GGUF.</summary>
        public static DFlashConfig FromGguf(GgufFile gguf)
        {
            ArgumentNullException.ThrowIfNull(gguf);

            string arch = gguf.GetString("general.architecture") ?? string.Empty;
            if (!string.Equals(arch, ArchName, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Expected a '{ArchName}' GGUF for the Muse-Glimmer DFlash drafter but got '{arch}'.");
            }

            int keyLength = (int)gguf.GetUint32($"{ArchName}.attention.key_length");
            int valueLength = (int)gguf.GetUint32($"{ArchName}.attention.value_length", (uint)keyLength);
            if (keyLength != valueLength)
            {
                throw new NotSupportedException(
                    $"DFlash key_length {keyLength} != value_length {valueLength}; only square head dims are supported.");
            }

            var cfg = new DFlashConfig
            {
                NumLayers = (int)gguf.GetUint32($"{ArchName}.block_count"),
                HiddenSize = (int)gguf.GetUint32($"{ArchName}.embedding_length"),
                IntermediateSize = (int)gguf.GetUint32($"{ArchName}.feed_forward_length"),
                NumHeads = (int)gguf.GetUint32($"{ArchName}.attention.head_count"),
                NumKVHeads = (int)gguf.GetUint32($"{ArchName}.attention.head_count_kv"),
                HeadDim = keyLength,
                RopeBase = gguf.GetFloat32($"{ArchName}.rope.freq_base", 10000f),
                Eps = gguf.GetFloat32($"{ArchName}.attention.layer_norm_rms_epsilon", 1e-5f),
                BlockSize = (int)gguf.GetUint32($"{ArchName}.block_size"),
                TargetLayerIds = gguf.GetInt32Array($"{ArchName}.target_layers") ?? Array.Empty<int>(),
                SlidingWindow = (int)gguf.GetUint32($"{ArchName}.attention.sliding_window"),
                SwaPattern = gguf.GetBoolArray($"{ArchName}.attention.sliding_window_pattern") ?? Array.Empty<bool>(),
                // A missing key yields uint.MaxValue, which casts to -1 and is
                // rejected by ValidateSelfConsistent below.
                MaskTokenId = (int)gguf.GetUint32("tokenizer.ggml.mask_token_id", uint.MaxValue),
            };

            cfg.ValidateSelfConsistent();
            return cfg;
        }

        private void ValidateSelfConsistent()
        {
            if (NumLayers <= 0)
                throw new InvalidOperationException($"{ArchName}.block_count is missing or zero.");
            if (HiddenSize <= 0)
                throw new InvalidOperationException($"{ArchName}.embedding_length is missing or zero.");
            if (IntermediateSize <= 0)
                throw new InvalidOperationException($"{ArchName}.feed_forward_length is missing or zero.");
            if (NumHeads <= 0 || NumKVHeads <= 0 || NumHeads % NumKVHeads != 0)
            {
                throw new InvalidOperationException(
                    $"DFlash head counts do not group: head_count={NumHeads}, head_count_kv={NumKVHeads}.");
            }
            if (HeadDim <= 0 || (HeadDim & 1) != 0)
                throw new InvalidOperationException($"DFlash head dim {HeadDim} must be positive and even (NeoX RoPE splits it in half).");
            if (BlockSize < 2)
                throw new InvalidOperationException($"{ArchName}.block_size {BlockSize} must be at least 2 (anchor + one draft).");
            if (TargetLayerIds.Length == 0)
                throw new InvalidOperationException($"{ArchName}.target_layers is missing or empty.");
            if (SlidingWindow <= 0)
                throw new InvalidOperationException($"{ArchName}.attention.sliding_window {SlidingWindow} must be positive.");
            if (MaskTokenId < 0)
                throw new InvalidOperationException("tokenizer.ggml.mask_token_id is missing from the DFlash GGUF; the block draft has no mask id to fill its slots with.");
            if (SwaPattern.Length != NumLayers)
            {
                throw new InvalidOperationException(
                    $"{ArchName}.attention.sliding_window_pattern has {SwaPattern.Length} entries for {NumLayers} layers.");
            }
            for (int il = 0; il < NumLayers; il++)
            {
                if (!SwaPattern[il])
                {
                    // The drafter's history lives in a RingRows-row ring sized for
                    // the SWA span; a full-attention draft layer would have to read
                    // arbitrarily far back, which the ring cannot serve.
                    throw new NotSupportedException(
                        $"DFlash draft layer {il} is not a sliding-window layer; the drafter's KV ring only holds {RingRows} positions.");
                }
            }
        }

        public override string ToString()
            => $"dflash(layers={NumLayers}, hidden={HiddenSize}, ffn={IntermediateSize}, heads={NumHeads}/{NumKVHeads}x{HeadDim}, " +
               $"block={BlockSize}, drafts={MaxDraftTokens}, swa={SlidingWindow}, ring={RingRows}, " +
               $"targets=[{string.Join(",", TargetLayerIds)}], feature={FeatureSize}, mask={MaskTokenId})";
    }
}
