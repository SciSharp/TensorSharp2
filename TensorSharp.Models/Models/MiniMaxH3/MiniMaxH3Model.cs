// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3: joint video + 32 kHz stereo audio generation.
//
// Loaded from the denoiser GGUF; the Qwen3-VL text encoder, video VAE and audio VAE
// are resolved beside it or through TS_VIDEO_TEXT_ENCODER / TS_VIDEO_VAE /
// TS_VIDEO_AUDIO_VAE. Not an autoregressive text model — generation goes through
// GenerateVideo().
//
// Note the checkpoints carry NO GGUF metadata, so this model is identified by the
// joint presence of video_patch_proj and audio_patch_proj rather than by an
// architecture string, and the tokenizer has to come from separate vocab files.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>The MiniMax-H3 audio-video generator.</summary>
    public sealed class MiniMaxH3Model : ModelBase, IVideoGenerationModel
    {
        /// <summary>Architecture id this model reports. The GGUFs have no metadata,
        /// so nothing reads this back out of a file — it exists for the API and UI.</summary>
        public const string ArchitectureId = "minimax-h3";

        private readonly string _ditPath;
        private readonly string _tePath;
        private readonly string _vaePath;
        private readonly string _audioVaePath;
        private MiniMaxH3Pipeline _pipeline;

        /// <summary>The backend this model was loaded for; the pipeline needs it to
        /// make the matching GGML backend the active one.</summary>
        internal BackendType Backend => _backend;

        public MiniMaxH3Config DitConfig { get; }
        public MiniMaxH3Partition Partition { get; }

        public MiniMaxH3Model(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            _ditPath = ggufPath;
            string dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath));

            using (var probe = new GgufFile(ggufPath))
            {
                DitConfig = MiniMaxH3Config.Detect(probe.Tensors);
            }
            Partition = MiniMaxH3Config.PartitionFromFileName(ggufPath);
            Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = 0 };

            _tePath = ResolveCompanion("TS_VIDEO_TEXT_ENCODER", "TS_WAN_TE", dir,
                new[] { "qwen3vl_32b_minimax_h3-Q4_K_M.gguf" },
                n => n.Contains("qwen3vl") && n.EndsWith(".gguf"));
            _vaePath = ResolveCompanion("TS_VIDEO_VAE", "TS_WAN_VAE", dir,
                new[] { "minimax_h3_video_vae_fp16.safetensors" },
                n => n.Contains("video_vae") && n.EndsWith(".safetensors"));
            _audioVaePath = ResolveCompanion("TS_VIDEO_AUDIO_VAE", null, dir,
                new[] { "minimax_h3_audio_vae_fp32.safetensors" },
                n => n.Contains("audio_vae") && n.EndsWith(".safetensors"));

            Console.WriteLine($"MiniMax-H3 ({Partition}): DiT={Path.GetFileName(ggufPath)}");
            Console.WriteLine($"  {DitConfig}");
            Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");
            Console.WriteLine($"  video VAE    = {_vaePath ?? "<missing>"}");
            Console.WriteLine($"  audio VAE    = {_audioVaePath ?? "<missing> (video only)"}");

            if (_tePath == null || _vaePath == null)
                throw new FileNotFoundException(
                    "MiniMax-H3 needs companion models beside the denoiser GGUF (or via " +
                    "--video-text-encoder / --video-vae): the Qwen3-VL-32B text encoder GGUF " +
                    "(unsloth/MiniMax-H3-GGUF) and minimax_h3_video_vae_fp16.safetensors " +
                    "(Comfy-Org/MiniMax-H3). The text encoder also needs vocab.json and " +
                    "merges.txt beside it, since its GGUF carries no tokenizer.");
        }

        /// <summary>True when the tensor table looks like a MiniMax-H3 denoiser. Used
        /// to route a GGUF that has no architecture metadata to read.</summary>
        public static bool LooksLikeMiniMaxH3(GgufFile gguf) =>
            gguf != null && MiniMaxH3Config.IsMiniMaxH3(gguf.Tensors);

        private static string ResolveCompanion(string envVar, string legacyEnvVar, string dir,
                                               string[] preferred, Func<string, bool> match)
        {
            foreach (string v in new[] { envVar, legacyEnvVar })
            {
                if (v == null) continue;
                string env = Environment.GetEnvironmentVariable(v);
                if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            }
            string parent = Path.GetDirectoryName(dir);
            foreach (var d in new[] { dir, parent })
            {
                if (d == null) continue;
                foreach (var name in preferred)
                {
                    string p = Path.Combine(d, name);
                    if (File.Exists(p)) return p;
                }
            }
            foreach (var d in new[] { dir, parent })
            {
                if (d == null || !Directory.Exists(d)) continue;
                foreach (var f in Directory.EnumerateFiles(d, "*", SearchOption.AllDirectories)
                                           .OrderBy(f => f.Length))
                    if (match(Path.GetFileName(f).ToLowerInvariant())) return f;
            }
            return null;
        }

        internal MiniMaxH3TextEncoder CreateTextEncoder() => new(_tePath);
        internal MiniMaxH3DiT CreateDiT() => new(_ditPath);
        internal MiniMaxH3VideoVae CreateVideoVae() => new(_vaePath);
        internal MiniMaxH3AudioVae CreateAudioVae() =>
            _audioVaePath != null ? new MiniMaxH3AudioVae(_audioVaePath) : null;
        internal string AudioVaePath => _audioVaePath;

        /// <summary>Generate a clip. MiniMax-H3 is CFG-distilled, so guidance must be
        /// 1.0 and 4-8 steps is the operating point.</summary>
        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams p = null)
        {
            _pipeline ??= new MiniMaxH3Pipeline(this);
            return _pipeline.Generate(prompt, p ?? new VideoGenerationParams());
        }

        // ---- IVideoGenerationModel ----
        string IVideoGenerationModel.VideoModelFamily => ArchitectureId;
        // The joint audio stream is generated either way; only its decode needs the VAE.
        bool IVideoGenerationModel.SupportsAudio => _audioVaePath != null;
        bool IVideoGenerationModel.SupportsImageConditioning => true;
        bool IVideoGenerationModel.SupportsEndImageConditioning =>
            Partition == MiniMaxH3Partition.FirstLastFrame;
        bool IVideoGenerationModel.SupportsReferenceConditioning =>
            Partition == MiniMaxH3Partition.Reference;
        int IVideoGenerationModel.MaxReferenceImages =>
            Partition == MiniMaxH3Partition.Reference ? MiniMaxH3Geometry.MaxReferences : 0;

        // ---- not an autoregressive model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException(
                "MiniMaxH3Model is a video-generation model; use GenerateVideo().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }
    }
}
