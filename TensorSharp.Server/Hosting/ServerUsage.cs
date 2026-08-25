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
using System.IO;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Informational entry points that print and exit before the web host is
    /// built: the full usage page (shown for a bare <c>TensorSharp.Server</c>
    /// invocation or <c>--help</c>) and the Vulkan GPU listing
    /// (<c>--list-gpus</c>). Kept out of <see cref="ServerOptionsBuilder"/> so
    /// the option parser stays pure and testable.
    /// </summary>
    internal static class ServerUsage
    {
        public static bool IsHelpRequested(string[] args)
        {
            if (args == null)
                return false;
            foreach (string a in args)
            {
                if (string.Equals(a, "--help", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "-h", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "-?", StringComparison.Ordinal) ||
                    string.Equals(a, "/?", StringComparison.Ordinal))
                {
                    return true;
                }
            }
            return false;
        }

        public static bool IsListGpusRequested(string[] args)
        {
            if (args == null)
                return false;
            foreach (string a in args)
            {
                if (string.Equals(a, "--list-gpus", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Print the Vulkan devices ggml-vulkan can see (index + adapter name) so the
        /// operator knows what to pass to <c>--gpu-device</c> on multi-GPU hosts.
        /// Enumerating spins up the Vulkan instance but no backend/device state.
        /// Mirrors the CLI's <c>--list-gpus</c>.
        /// </summary>
        public static void PrintVulkanGpus(TextWriter writer)
        {
            int count = TensorSharp.GGML.GgmlBasicOps.GetVulkanDeviceCount();
            if (count <= 0)
            {
                writer.WriteLine("No Vulkan devices found. Ensure the native GGML bridge is built with Vulkan support " +
                    "(TensorSharp.GGML.Native/build-windows.ps1 --vulkan) and a Vulkan driver is installed.");
                return;
            }

            writer.WriteLine($"Vulkan devices ({count}):");
            for (int i = 0; i < count; i++)
            {
                writer.WriteLine($"  {i}: {TensorSharp.GGML.GgmlBasicOps.GetVulkanDeviceDescription(i) ?? "(unknown)"}");
            }
            writer.WriteLine("Select one with: --backend ggml_vulkan --gpu-device <index>");
        }

        /// <summary>One option entry on the usage page.</summary>
        private readonly record struct OptionHelp(string Flag, string Description, string Example);

        // Grouped to match the option passes in Program.cs / ServerOptionsBuilder.
        // Keep flags in sync with ServerOptionsBuilder.ParseArgs and its
        // SuggestFlagCorrection known-flag list.
        private static readonly (string Section, OptionHelp[] Options)[] Sections =
        {
            ("Model", new[]
            {
                new OptionHelp("--model <path>",
                    "GGUF model to host at startup. Required for inference. Other options can start a model-less " +
                    "status process, but /api/models/load cannot select a GGUF that was not supplied at startup.",
                    "--model C:\\models\\gemma-4-E4B-it-Q8_0.gguf"),
                new OptionHelp("--mmproj <path|none>",
                    "Multimodal projector GGUF. A bare filename is resolved next to the model; 'none' disables it. " +
                    "Requires --model. Default: none — pass the matching projector explicitly.",
                    "--mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf"),
            }),
            ("Network", new[]
            {
                new OptionHelp("--port <N>",
                    "TCP port to listen on (1-65535). Default: 5000 (PORT env var overrides). On macOS, port 5000 is " +
                    "taken by the AirPlay Receiver in Control Center, so pick another port or turn that off.",
                    "--port 8080"),
                new OptionHelp("--host <address>",
                    "Interface to bind. Default: 0.0.0.0 — every interface, so the server is reachable from other " +
                    "machines and from outside a container. Use 127.0.0.1 to restrict it to this machine " +
                    "(HOST env var overrides).",
                    "--host 127.0.0.1 --port 8080"),
                new OptionHelp("--urls <urls>",
                    "Full listen URL(s), semicolon-separated, for cases --port/--host cannot express (HTTPS, or " +
                    "binding several endpoints at once). Overridden by --port/--host when both are given; falls back " +
                    "to the ASPNETCORE_URLS env var.",
                    "--urls \"http://0.0.0.0:8080;https://0.0.0.0:8443\""),
                new OptionHelp("--no-webui",
                    "Do not serve the bundled web UI; GET / answers the plain liveness text instead. All HTTP API " +
                    "endpoints (including /uploads) stay up. Default: UI on (TS_NO_WEBUI env var overrides).",
                    "--no-webui"),
            }),
            ("Compute backend", new[]
            {
                new OptionHelp("--backend <type>",
                    "Compute backend: cpu, cuda, mlx, ggml_cpu, ggml_metal, ggml_cuda, or ggml_vulkan. " +
                    "Default: ggml_metal on macOS, ggml_cpu elsewhere (BACKEND env var overrides).",
                    "--backend ggml_vulkan"),
                new OptionHelp("--gpu-device <N>",
                    "Vulkan device index for the ggml_vulkan backend on multi-GPU hosts (e.g. an integrated Intel GPU " +
                    "next to a discrete NVIDIA one). Default: 0 (TS_GGML_VULKAN_DEVICE env var overrides).",
                    "--backend ggml_vulkan --gpu-device 1"),
                new OptionHelp("--list-gpus",
                    "List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit.",
                    "--list-gpus"),
            }),
            ("Tensor parallelism (multi-GPU serving)", new[]
            {
                new OptionHelp("--tp <N>",
                    "Split the model across N GPUs on this machine (tensor parallelism): each GPU holds 1/N of every " +
                    "weight and the shards cooperate on every token. Use it when a model does not fit on one GPU. " +
                    "Range: 1 to the number of local GPUs. Applies to the cuda, ggml_cuda, and ggml_vulkan backends. " +
                    "Default: 1 — no splitting (TENSORSHARP_TP_DEGREE env var overrides).",
                    "--model Qwen3.5-35B-A3B-Q4_K_M.gguf --backend ggml_cuda --tp 2"),
                new OptionHelp("--tp-node-id <N>",
                    "This node's 0-based ID for multi-node (distributed) tensor parallelism over TCP. The server can " +
                    "only be node 0 — the driver that owns sampling and serves HTTP; start every other node as a " +
                    "worker with TensorSharp.Cli using the same model, backend, and --tp-peers list. Requires " +
                    "--tp-peers. Default: none — single-node (TENSORSHARP_TP_NODE_ID env var overrides).",
                    "--tp 2 --tp-node-id 0 --tp-peers 192.168.1.10:9500,192.168.1.11:9500"),
                new OptionHelp("--tp-peers <list>",
                    "Comma-separated host:port list of ALL nodes in the distributed TP cluster, ordered by node ID; " +
                    "every node passes the identical list. Requires --tp-node-id. Default: none " +
                    "(TENSORSHARP_TP_PEERS env var overrides).",
                    "--tp-peers 192.168.1.10:9500,192.168.1.11:9500"),
            }),
            ("Generation defaults (pinned values also override requests — see --sampling-precedence)", new[]
            {
                new OptionHelp("--max-tokens <N>",
                    "Maximum tokens to generate per request: fills in when the request omits a limit, and caps a " +
                    "request that asks for more. Default: 20000, uncapped (MAX_TOKENS env var overrides).",
                    "--max-tokens 4096"),
                new OptionHelp("--temperature <f>",
                    "Sampling temperature; 0 = greedy. Default: 0.8 (TENSORSHARP_TEMPERATURE env var).",
                    "--temperature 0"),
                new OptionHelp("--top-k <N>",
                    "Top-K filtering; 0 disables. Default: 40 (TENSORSHARP_TOP_K env var).",
                    "--top-k 64"),
                new OptionHelp("--top-p <f>",
                    "Nucleus sampling threshold; 1.0 disables. Default: 0.9 (TENSORSHARP_TOP_P env var).",
                    "--top-p 0.95"),
                new OptionHelp("--min-p <f>",
                    "Minimum-probability filtering; 0 disables. Default: 0 (TENSORSHARP_MIN_P env var).",
                    "--min-p 0.05"),
                new OptionHelp("--repeat-penalty <f>",
                    "Repetition penalty; 1.0 = none. Default: 1.1 (TENSORSHARP_REPEAT_PENALTY env var).",
                    "--repeat-penalty 1.0"),
                new OptionHelp("--repeat-last-n <n>",
                    "Recent-token penalty window; 0 disables, -1 uses all. Default: 64 (TENSORSHARP_REPEAT_LAST_N env var).",
                    "--repeat-last-n 128"),
                new OptionHelp("--presence-penalty <f>",
                    "Presence penalty; 0 disables. Default: 0 (TENSORSHARP_PRESENCE_PENALTY env var).",
                    "--presence-penalty 0.2"),
                new OptionHelp("--frequency-penalty <f>",
                    "Frequency penalty; 0 disables. Default: 0 (TENSORSHARP_FREQUENCY_PENALTY env var).",
                    "--frequency-penalty 0.3"),
                new OptionHelp("--seed <N>",
                    "Random seed; -1 = non-deterministic. Default: -1 (TENSORSHARP_SEED env var).",
                    "--seed 42"),
                new OptionHelp("--stop <text>",
                    "Stop sequence; repeat the flag to pin several. Default: none.",
                    "--stop \"</s>\" --stop \"<|eot|>\""),
                new OptionHelp("--sampling-precedence <config|request>",
                    "Who wins when a request also carries a sampling parameter you pinned above. 'config' " +
                    "(default) keeps your values — clients such as VS Code Copilot Chat hardcode temperature/top_p " +
                    "into every request and would otherwise silently override them; parameters you did NOT pin " +
                    "still come from the request. 'request' restores client-always-wins. Pinned stop sequences " +
                    "are merged with the request's rather than replacing them " +
                    "(TENSORSHARP_SAMPLING_PRECEDENCE env var overrides).",
                    "--sampling-precedence request"),
            }),
            ("Mixture-of-Experts CPU offload", new[]
            {
                new OptionHelp("--n-cpu-moe <N> | -ncmoe <N>",
                    "Keep the routed MoE expert weights of the first N layers in system RAM and multiply them on " +
                    "the CPU; attention, norms, the router and the shared expert stay on the accelerator. This is " +
                    "what makes a 35B-A3B MoE fit beside a long-context KV cache on a 12-16 GB card. Pass 'all' " +
                    "for every layer. Default: 0 (everything on the accelerator; TS_N_CPU_MOE env var overrides).",
                    "--n-cpu-moe 32"),
                new OptionHelp("--cpu-moe | -cmoe",
                    "Shorthand for --n-cpu-moe all: every routed expert stays in system RAM. Default: off " +
                    "(TS_CPU_MOE env var overrides).",
                    "--cpu-moe"),
                new OptionHelp("--cpu-moe-threads <N>",
                    "Worker threads for the host-side expert matmul. Default: one less than the CPU parallelism " +
                    "this process can actually use (hardware threads clamped by the affinity mask and the cgroup " +
                    "CPU quota), leaving a core for accelerator submission. Do not set this above the quota: " +
                    "ggml's pool spins at its barriers, so oversubscription collapses throughput rather than " +
                    "degrading it (TS_CPU_MOE_THREADS env var overrides).",
                    "--cpu-moe-threads 12"),
            }),
            ("KV cache", new[]
            {
                new OptionHelp("--kv-cache-dtype <t>",
                    "KV cache precision: f32, f16, q8_0, or q4_0. Quantized caches trade small numerical drift for " +
                    "memory. Default: auto — the backend/model pick (KV_CACHE_DTYPE env var overrides).",
                    "--kv-cache-dtype q8_0"),
            }),
            ("Cross-session paged KV cache", new[]
            {
                new OptionHelp("--paged-kv | --no-paged-kv",
                    "Enable/disable the cross-session paged KV cache (prefix reuse across requests). Default: off.",
                    "--paged-kv"),
                new OptionHelp("--paged-kv-block-size <N>",
                    "Tokens per KV block. Default: 256.",
                    "--paged-kv-block-size 128"),
                new OptionHelp("--paged-kv-ram-mb <N>",
                    "RAM budget for evicted KV blocks, in MB. Default: 1024.",
                    "--paged-kv-ram-mb 2048"),
                new OptionHelp("--paged-kv-ssd-dir <path>",
                    "Directory for the SSD spill tier. Default: disabled.",
                    "--paged-kv-ssd-dir D:\\ts-kv-spill"),
                new OptionHelp("--paged-kv-ssd-mb <N>",
                    "SSD budget for spilled KV blocks, in MB. Default: 16384.",
                    "--paged-kv-ssd-mb 32768"),
                new OptionHelp("--paged-kv-quant-bits <b>",
                    "Quantize spilled KV blocks with the TurboQuant codec: 0 (off), 2, 4, or 8 bits per element. " +
                    "2-bit uses an affine min+scale layout (~4x smaller than the f16 payload). Default: 0.",
                    "--paged-kv-quant-bits 8"),
                new OptionHelp("--paged-kv-redis-url <url>",
                    "Redis connection string for a shared KV cache tier (e.g. localhost:6379). Default: disabled.",
                    "--paged-kv-redis-url localhost:6379"),
                new OptionHelp("--paged-kv-redis-ttl <min>",
                    "TTL in minutes for Redis KV entries (0 = no TTL). Default: 1440.",
                    "--paged-kv-redis-ttl 60"),
                new OptionHelp("--redis-url <url>",
                    "Redis connection string for both the KV cache tier and the Responses API store.",
                    "--redis-url localhost:6379"),
            }),
            ("Scheduling", new[]
            {
                new OptionHelp("--continuous-batching | --no-continuous-batching",
                    "Paged-attention continuous batching across concurrent requests (aliases --paged-batching / " +
                    "--no-paged-batching). Default: on.",
                    "--no-continuous-batching"),
                new OptionHelp("--prefill-chunk-size <N>",
                    "Chunked-prefill granularity under contention; smaller chunks give parallel decodes more frequent " +
                    "turns at the GPU. Default: 1024.",
                    "--prefill-chunk-size 256"),
            }),
            ("Speculative decoding", new[]
            {
                new OptionHelp("--spec | --no-spec",
                    "Enable/disable speculative decoding: a drafter proposes the next few tokens and the trunk " +
                    "verifies them in one batched forward. Every emitted token still comes from a trunk row, so " +
                    "the output is what standard decoding would have produced. Engages for solo (non-concurrent) " +
                    "sequences. Accepted as --mtp-spec / --no-mtp-spec too. Default: off.",
                    "--spec"),
                new OptionHelp("--spec-type <name>",
                    "Speculation algorithm: 'auto' (default) uses whatever drafter the checkpoint carries; " +
                    "'draft-head' and 'block' pin one explicitly; 'ngram' needs no trained weights at all and " +
                    "works on every model, drafting by suffix match over the context (strong when the answer " +
                    "quotes its input: summarizing, editing, structured output, agentic loops).",
                    "--spec --spec-type ngram"),
                new OptionHelp("--spec-draft <N>",
                    "Maximum draft tokens per step (1-64). Accepted as --mtp-draft too. Default: 8.",
                    "--spec-draft 4"),
                new OptionHelp("--spec-pmin <f>",
                    "Draft-confidence gate in (0, 1]; drafting stops below it. What the number means is per " +
                    "algorithm, so each brings its own default — 0.75 for a per-token draft head, 0.35 for a " +
                    "block drafter (where the gate is the CUMULATIVE prefix probability, so the same number " +
                    "means something much stricter), 0 for n-gram. Accepted as --mtp-pmin too.",
                    "--spec-pmin 0.6"),
                new OptionHelp("--spec-draft-model <path>",
                    "Separate draft GGUF for models whose draft head ships as its own file (Gemma 4 assistant). " +
                    "Qwen 3.6 and GLM 5.2 embed their NextN block and need no flag. Accepted as " +
                    "--mtp-draft-model too. Default: none.",
                    "--spec-draft-model gemma-4-E4B-it-assistant.Q8_0.gguf"),
                new OptionHelp("--draft-model <path>",
                    "Block drafter GGUF for architectures whose drafter must be resident before the layer " +
                    "split (DeepSeek V4's DSpark). Naming the file IS the request, so it needs no --spec; " +
                    "engages for solo sequences on the cuda and ggml_cuda backends. Default: none; " +
                    "env TS_DSV4_DSPARK.",
                    "--draft-model DSpark-drafter-Q2K-Q8-0731.gguf"),
            }),
            ("Qwen-Image-Edit companion models (qwen_image DiT GGUFs)", new[]
            {
                new OptionHelp("--qwen-image-vae <path>",
                    "VAE GGUF. Default: same-directory scan next to the DiT model.",
                    "--qwen-image-vae qwen-image-vae.gguf"),
                new OptionHelp("--qwen-image-vl <path>",
                    "Qwen2.5-VL text-encoder GGUF. Default: same-directory scan.",
                    "--qwen-image-vl qwen-image-te-Qwen2.5-VL-7B-Q4_K_M.gguf"),
                new OptionHelp("--qwen-image-mmproj <path>",
                    "Vision projector GGUF for the text encoder. Default: same-directory scan.",
                    "--qwen-image-mmproj Qwen2.5-VL-7B-mmproj-BF16.gguf"),
                new OptionHelp("--qwen-image-lora <path>",
                    "DiT LoRA (e.g. a Lightning step-distillation checkpoint); also switches sampling defaults. " +
                    "Default: none.",
                    "--qwen-image-lora Qwen-Image-Edit-Lightning-8steps.safetensors"),
                new OptionHelp("--offload-cpu",
                    "Stream the DiT weights from RAM instead of holding them resident in VRAM " +
                    "(sd.cpp --offload-to-cpu equivalent): slower per step, but the freed VRAM lets " +
                    "native ~1 MP edits run on small cards. Default: auto (engages only when the " +
                    "target resolution does not fit beside the resident weights).",
                    "--offload-cpu"),
            }),
            ("Video-generation defaults and companion models", new[]
            {
                new OptionHelp("--video-width <px>",
                    "Default output width when a request omits 'width'. THE main quality lever: " +
                    "the Web UI sends no size of its own, so without this every clip is generated " +
                    "at the model's default. 640x384 is a good starting point for MiniMax-H3; " +
                    "--width is accepted as an alias. Rounded up to the model's grid.",
                    "--video-width 640"),
                new OptionHelp("--video-height <px>",
                    "Default output height when a request omits 'height'. Alias: --height. " +
                    "If only one of width/height is given, MiniMax-H3 takes the other from the " +
                    "conditioning image's aspect ratio.",
                    "--video-height 384"),
                new OptionHelp("--video-steps <N>",
                    "Default denoising steps when a request omits 'steps'. The quality/time " +
                    "trade-off after resolution. MiniMax-H3 is step-distilled and CFG-free: 8 is " +
                    "the default, 20 is the default here, 16-24 is visibly cleaner, past ~30 gains little.",
                    "--video-steps 16"),
                new OptionHelp("--video-mode <mode>",
                    "Default conditioning mode when a request omits 'videoMode': t2v (text only), " +
                    "i2v (the image is the first frame and is animated), fl2v (first AND last frame " +
                    "pinned) or ref (the images are identity/appearance references for a new scene). " +
                    "Omit it and the mode is inferred from what each request supplies, which is " +
                    "usually what you want; pin it for a deployment that only offers one.",
                    "--video-mode ref"),
                new OptionHelp("--video-frames <N>",
                    "Default output frame count when a request omits 'frames'. The count is snapped to the " +
                    "model's temporal grid (4k+1 for Wan, 17k+5 for MiniMax-H3). Model default: 33, or 49 for " +
                    "Wan2.2-TI2V. A request value overrides it.",
                    "--video-frames 121"),
                new OptionHelp("--fps <N>",
                    "Default MP4 playback rate when a request omits 'fps'. Model default: 16, or 24 for " +
                    "Wan2.2-TI2V. A request value overrides it; FPS changes playback rate, not generation work. " +
                    "Models trained at a fixed rate (MiniMax-H3, 24 fps) override any other value.",
                    "--fps 24"),
                new OptionHelp("--video-vae <path>",
                    "Video VAE (wan_2.1_vae.safetensors, or Wan2.2_VAE.safetensors for TI2V-5B; " +
                    "minimax_h3_video_vae_fp16.safetensors for MiniMax-H3). Default: same-directory scan next " +
                    "to the DiT model, VAE/ subfolders included (TS_VIDEO_VAE). The former spelling " +
                    "--wan-vae is still accepted.",
                    "--video-vae Wan2.2_VAE.safetensors"),
                new OptionHelp("--video-text-encoder <path>",
                    "Text-encoder GGUF (UMT5-XXL for Wan, Qwen3-VL-32B for MiniMax-H3). Default: " +
                    "same-directory scan (TS_VIDEO_TEXT_ENCODER). Also spelled --video-te; the former " +
                    "spelling --wan-te is still accepted.",
                    "--video-text-encoder umt5-xxl-encoder-Q8_0.gguf"),
                new OptionHelp("--video-dit2 <path>",
                    "Second diffusion expert on dual-expert models (Wan 2.2 A14B's high/low-noise partner of " +
                    "--model). Default: auto-resolved by name from the same or a sibling folder " +
                    "(TS_VIDEO_DIT2); needed only when the pair is not co-located, or to name it in a " +
                    "--config file. The former spelling --wan-dit2 is still accepted.",
                    "--video-dit2 wan2.2_i2v_A14b_low_noise-Q4_K_M.gguf"),
                new OptionHelp("--audio-vae <path>",
                    "Audio VAE for models that generate an audio track jointly with the video " +
                    "(minimax_h3_audio_vae_fp32.safetensors). Without it such a model still runs and produces " +
                    "video, just no audio (TS_VIDEO_AUDIO_VAE).",
                    "--audio-vae minimax_h3_audio_vae_fp32.safetensors"),
            }),
            ("Upload storage (the uploads/ directory next to the server binary)", new[]
            {
                new OptionHelp("--upload-max-mb <N>",
                    "Per-file cap in MB on client-originated writes: multipart /api/upload files and base64 " +
                    "attachments decoded out of chat requests. Default: 500, the request-body limit " +
                    "(TS_UPLOAD_MAX_MB env var overrides).",
                    "--upload-max-mb 25"),
                new OptionHelp("--upload-quota-mb <N>",
                    "Total budget in MB for the upload directory, counting client uploads, decoded attachments, " +
                    "and generated outputs (edited images, videos). Requests that would exceed it are rejected " +
                    "up front — before any model work runs. Default: off (TS_UPLOAD_QUOTA_MB env var overrides).",
                    "--upload-quota-mb 2048"),
                new OptionHelp("--upload-ttl-hours <N>",
                    "Delete upload-directory files older than this many hours (fractions allowed). Default: off, " +
                    "because chat sessions reference attachments by path and may reuse them later; enable it when " +
                    "the server is reachable by untrusted clients (TS_UPLOAD_TTL_HOURS env var overrides).",
                    "--upload-ttl-hours 24"),
            }),
            ("Configuration file", new[]
            {
                new OptionHelp("--config <path>",
                    "Read options from a JSON file whose keys are the same long option names listed here (with or " +
                    "without the leading --). Anything also passed on the command line overrides the file; when the " +
                    "flag is repeated, later files win over earlier ones. String/number values map to '--key value', " +
                    "true maps to the bare '--key' switch, and an array maps to a repeated flag (e.g. \"stop\": [..]). " +
                    "A \"variables\" object lets values share ${name} references; a file option may instead be an " +
                    "object { \"path\": \"...\", \"urls\": [ \"...\" ] } that auto-downloads on first run. See the " +
                    "config/ folder and config/README.md for examples.",
                    "--config server.json --backend ggml_cuda"),
            }),
            ("Help", new[]
            {
                new OptionHelp("--help",
                    "Show this help and exit (also shown when the server is started with no arguments).",
                    "--help"),
            }),
        };

        public static void PrintUsage(TextWriter writer)
        {
            writer.WriteLine("Usage: TensorSharp.Server [options]");
            writer.WriteLine();
            writer.WriteLine("Hosts an OpenAI- and Ollama-compatible inference server (plus a built-in web chat UI)");
            writer.WriteLine("on http://0.0.0.0:5000 by default (change it with --port / --host). Run with no");
            writer.WriteLine("arguments to show this help; pass at least one option to start the server.");

            foreach (var (section, options) in Sections)
            {
                writer.WriteLine();
                writer.WriteLine(section + ":");
                foreach (var option in options)
                {
                    writer.WriteLine($"  {option.Flag}");
                    WriteWrapped(writer, option.Description, indent: "      ");
                    writer.WriteLine($"      Example: {option.Example}");
                }
            }

            writer.WriteLine();
            writer.WriteLine("Examples:");
            writer.WriteLine("  TensorSharp.Server --model C:\\models\\gemma-4-E4B-it-Q8_0.gguf --backend ggml_cpu");
            writer.WriteLine("  TensorSharp.Server --model gemma-4-E4B-it-Q8_0.gguf --mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda");
            writer.WriteLine("  TensorSharp.Server --model Qwen3.5-35B-A3B-Q4_K_M.gguf --backend ggml_cuda --tp 2    (split across 2 GPUs)");
            writer.WriteLine("  TensorSharp.Server --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda --video-frames 121 --fps 24");
            writer.WriteLine("  TensorSharp.Server --backend ggml_cpu    (model-less status process; inference unavailable)");
            writer.WriteLine("  TensorSharp.Server --config server.json    (read options from a file)");
            writer.WriteLine("  TensorSharp.Server --config server.json --backend ggml_cuda    (file, but override the backend)");
            writer.WriteLine();
            writer.WriteLine("Logging env vars: TENSORSHARP_LOG_LEVEL (Information), TENSORSHARP_LOG_DIR (./logs),");
            writer.WriteLine("TENSORSHARP_LOG_FILE=0 disables file logging.");
        }

        private const int WrapColumn = 100;

        private static void WriteWrapped(TextWriter writer, string text, string indent)
        {
            int width = WrapColumn - indent.Length;
            var line = new System.Text.StringBuilder();
            foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (line.Length > 0 && line.Length + 1 + word.Length > width)
                {
                    writer.WriteLine(indent + line);
                    line.Clear();
                }
                if (line.Length > 0)
                    line.Append(' ');
                line.Append(word);
            }
            if (line.Length > 0)
                writer.WriteLine(indent + line);
        }
    }
}
