// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 checkpoint tests: architecture detection straight off the real GGUFs.
//
// These matter more than usual because the published H3 GGUFs carry ZERO metadata —
// no general.architecture, no hyperparameters, no tokenizer. Everything the loader
// knows comes from the tensor table, so a detection bug is a silent mis-load rather
// than a clean failure.
//
// Point TS_MINIMAX_H3_DIR at a directory holding the denoiser and text-encoder GGUFs:
//   export TS_MINIMAX_H3_DIR=~/work/models/minimax-h3
// Tests skip visibly when it is unset or the weights are absent.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3CheckpointTests
    {
        private const string DirEnv = "TS_MINIMAX_H3_DIR";
        private readonly ITestOutputHelper _output;

        public MiniMaxH3CheckpointTests(ITestOutputHelper output) { _output = output; }

        // The gate and the loader share this pattern so the skip decision and the load
        // can never disagree about which file is under test.
        private const string DenoiserPattern = "minimax_h3_fl2va|minimax_h3_ref2va";

        private static string DenoiserPath()
        {
            string dir = Environment.GetEnvironmentVariable(DirEnv);
            return string.IsNullOrEmpty(dir) ? null : TestGates.FindGguf(dir, DenoiserPattern);
        }

        [ModelFact(DirEnv, DenoiserPattern)]
        public void DetectsMiniMaxH3FromTheTensorTableAlone()
        {
            string path = DenoiserPath();
            Assert.NotNull(path);
            using var gguf = new GgufFile(path);

            // The premise of the whole detection path: no metadata to read.
            _output.WriteLine($"[h3] {Path.GetFileName(path)}: " +
                              $"{gguf.Tensors.Count} tensors, {gguf.Metadata.Count} metadata keys");
            Assert.Empty(gguf.Metadata);

            Assert.True(MiniMaxH3Config.IsMiniMaxH3(gguf.Tensors));
            var cfg = MiniMaxH3Config.Detect(gguf.Tensors);
            _output.WriteLine("[h3] " + cfg);

            // The published architecture, per MiniMaxAI/MiniMax-H3 transformer/config.json.
            Assert.Equal(5376, cfg.HiddenSize);
            Assert.Equal(50, cfg.NumLayers);
            Assert.Equal(2, cfg.TokenRefinerLayers);
            Assert.Equal(56, cfg.NumAttentionHeads);
            Assert.Equal(128, cfg.AttentionHeadDim);
            Assert.Equal(14336, cfg.FfnHiddenSize);
            Assert.Equal(24, cfg.VideoLatentChannels);
            Assert.Equal(32, cfg.AudioLatentChannels);
            Assert.Equal(5120, cfg.TextDim);
            Assert.Equal(16, cfg.RopeInvFreqLength);

            // Attention is WIDER than the residual stream — 56*128 = 7168 vs 5376. Getting
            // this wrong still produces conformable matmuls in some layouts, so assert it.
            Assert.Equal(7168, cfg.AttentionInnerDim);
            Assert.NotEqual(cfg.HiddenSize, cfg.AttentionInnerDim);

            // Released checkpoints use the learned curve table, not a sinusoidal MLP.
            Assert.Equal(MiniMaxH3TimeEmbedding.CurveTable, cfg.TimeEmbedding);
            Assert.True(cfg.AdaLnCurveGrid > 1, "curve table must have a usable grid");

            // 3 axes x 16 freqs x 2 = 96 of the 128 head dims rotate; the rest pass through.
            Assert.Equal(96, cfg.RopeRotaryDim);
            Assert.True(cfg.RopeRotaryDim < cfg.AttentionHeadDim);
        }

        [ModelFact(DirEnv, DenoiserPattern)]
        public void DetectedShapesAgreeWithEveryWeightInTheCheckpoint()
        {
            string path = DenoiserPath();
            using var gguf = new GgufFile(path);
            var cfg = MiniMaxH3Config.Detect(gguf.Tensors);
            string p = cfg.Prefix;

            // Cross-check the derived config against tensors it was NOT derived from, so a
            // wrong inference cannot agree with itself.
            AssertShape(gguf, p + "video_patch_proj.weight", cfg.VideoPatchDim, cfg.HiddenSize);
            AssertShape(gguf, p + "audio_patch_proj.weight", cfg.AudioLatentChannels, cfg.HiddenSize);
            AssertShape(gguf, p + "condition_proj.weight", cfg.TextDim, cfg.HiddenSize);
            AssertShape(gguf, p + "final_layer.video_out.weight", cfg.HiddenSize, cfg.VideoPatchDim);
            AssertShape(gguf, p + "final_layer.audio_out.weight", cfg.HiddenSize, cfg.AudioLatentChannels);

            // AdaLN: 6 modulation vectors x 3 modalities per block, 2 at the final layer.
            AssertShape(gguf, p + "blocks.0.adaln_proj.linear.weight",
                        cfg.TimeEmbedDim, cfg.AdaLnOutFeatures);
            AssertShape(gguf, p + "final_layer.adaln_proj.linear.weight",
                        cfg.TimeEmbedDim, cfg.FinalAdaLnOutFeatures);

            // Every block must be complete and identically shaped — a truncated download
            // otherwise shows up as a confusing mid-generation failure.
            for (int i = 0; i < cfg.NumLayers; i++)
            {
                AssertShape(gguf, $"{p}blocks.{i}.attn.qkv_proj.weight",
                            cfg.HiddenSize, 3 * cfg.AttentionInnerDim);
                AssertShape(gguf, $"{p}blocks.{i}.attn.out_proj.weight",
                            cfg.AttentionInnerDim, cfg.HiddenSize);
                AssertShape(gguf, $"{p}blocks.{i}.mlp.fc1.weight",
                            cfg.HiddenSize, 2 * cfg.FfnHiddenSize);
                AssertShape(gguf, $"{p}blocks.{i}.mlp.fc2.weight",
                            cfg.FfnHiddenSize, cfg.HiddenSize);
                AssertShape(gguf, $"{p}blocks.{i}.norm1.weight", cfg.HiddenSize);
                AssertShape(gguf, $"{p}blocks.{i}.norm2.weight", cfg.HiddenSize);
                AssertShape(gguf, $"{p}blocks.{i}.attn.q_norm.weight", cfg.AttentionHeadDim);
                AssertShape(gguf, $"{p}blocks.{i}.attn.k_norm.weight", cfg.AttentionHeadDim);
            }
        }

        [ModelFact(DirEnv, DenoiserPattern)]
        public void EveryCheckpointTensorIsAccountedForByTheArchitecture()
        {
            // If the checkpoint holds a tensor the implementation never binds, the
            // implementation is incomplete — this is the check that catches "we forgot
            // a whole submodule" rather than "we got a shape wrong".
            string path = DenoiserPath();
            using var gguf = new GgufFile(path);
            var cfg = MiniMaxH3Config.Detect(gguf.Tensors);

            var expected = new HashSet<string>(ExpectedTensorNames(cfg), StringComparer.Ordinal);
            var actual = new HashSet<string>(gguf.Tensors.Keys, StringComparer.Ordinal);

            var missing = expected.Except(actual).OrderBy(x => x).ToList();
            var unexpected = actual.Except(expected).OrderBy(x => x).ToList();
            foreach (var m in missing.Take(10)) _output.WriteLine("[h3] missing:    " + m);
            foreach (var u in unexpected.Take(10)) _output.WriteLine("[h3] unexpected: " + u);

            Assert.Empty(missing);
            Assert.Empty(unexpected);
        }

        /// <summary>Every tensor a MiniMax-H3 denoiser is expected to contain.</summary>
        private static IEnumerable<string> ExpectedTensorNames(MiniMaxH3Config cfg)
        {
            string p = cfg.Prefix;
            yield return p + "video_patch_proj.weight";
            yield return p + "video_patch_proj.bias";
            yield return p + "audio_patch_proj.weight";
            yield return p + "audio_patch_proj.bias";
            yield return p + "condition_proj.weight";
            yield return p + "condition_proj.bias";
            yield return p + "rope.inv_freq";

            if (cfg.TimeEmbedding == MiniMaxH3TimeEmbedding.CurveTable)
            {
                yield return p + "adaln_t_table";
            }
            else
            {
                yield return p + "time_embedder.proj_in.weight";
                yield return p + "time_embedder.proj_in.bias";
                yield return p + "time_embedder.proj_out.weight";
                yield return p + "time_embedder.proj_out.bias";
            }

            for (int i = 0; i < cfg.TokenRefinerLayers; i++)
                foreach (var n in BlockTensors($"{p}token_refiner.blocks.{i}.", adaLn: false))
                    yield return n;
            yield return p + "token_refiner.final_norm.weight";

            for (int i = 0; i < cfg.NumLayers; i++)
                foreach (var n in BlockTensors($"{p}blocks.{i}.", adaLn: true))
                    yield return n;

            yield return p + "final_layer.norm.weight";
            yield return p + "final_layer.adaln_proj.linear.weight";
            yield return p + "final_layer.adaln_proj.linear.bias";
            yield return p + "final_layer.video_out.weight";
            yield return p + "final_layer.video_out.bias";
            yield return p + "final_layer.audio_out.weight";
            yield return p + "final_layer.audio_out.bias";
        }

        // The token refiner reuses the block layout but is unmodulated, so it has no
        // adaln_proj — that asymmetry is the one thing to get right here.
        private static IEnumerable<string> BlockTensors(string prefix, bool adaLn)
        {
            yield return prefix + "norm1.weight";
            yield return prefix + "norm2.weight";
            yield return prefix + "attn.qkv_proj.weight";
            yield return prefix + "attn.q_norm.weight";
            yield return prefix + "attn.k_norm.weight";
            yield return prefix + "attn.out_proj.weight";
            yield return prefix + "mlp.fc1.weight";
            yield return prefix + "mlp.fc2.weight";
            if (adaLn)
            {
                yield return prefix + "adaln_proj.linear.weight";
                yield return prefix + "adaln_proj.linear.bias";
            }
        }

        private static void AssertShape(GgufFile gguf, string name, params int[] expected)
        {
            Assert.True(gguf.Tensors.TryGetValue(name, out var t), $"missing tensor '{name}'");
            var actual = t.Shape.Select(d => (int)d).ToArray();
            Assert.True(expected.SequenceEqual(actual),
                $"{name}: expected [{string.Join(",", expected)}] but was [{string.Join(",", actual)}]");
        }
    }
}
