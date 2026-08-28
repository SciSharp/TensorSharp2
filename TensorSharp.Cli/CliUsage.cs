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

namespace TensorSharp.Cli
{
    /// <summary>
    /// The full usage page, shown for a bare <c>TensorSharp.Cli</c> invocation
    /// or <c>--help</c>. Every operator-facing flag is documented with a
    /// description, its default value, the accepted range where one applies,
    /// and a copy-pasteable example, so a new user can learn the tool from the
    /// help text alone. Mirrors the layout of the server's usage page
    /// (<c>ServerUsage</c> in TensorSharp.Server).
    /// </summary>
    internal static class CliUsage
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

        /// <summary>One option entry on the usage page.</summary>
        private readonly record struct OptionHelp(string Flag, string Description, string Example);

        // Grouped by task. Keep flags in sync with the switch in
        // Program.MainCore and with CliLoggingSetup.ParseFromArgs.
        private static readonly (string Section, OptionHelp[] Options)[] Sections =
        {
            ("Model and prompt input", new[]
            {
                new OptionHelp("--model <path>",
                    "GGUF model to load. Required for every mode except --list-gpus and --test-templates. " +
                    "Quantized (e.g. Q8_0, Q4_K_M) and F16/BF16 GGUFs are supported; the architecture is read " +
                    "from the file. Default: none.",
                    "--model gemma-4-E4B-it-Q8_0.gguf"),
                new OptionHelp("--input <file>",
                    "UTF-8 text file with the prompt. Default: a built-in demo prompt (\"What is 1+1?\"); when an " +
                    "image/audio/video is attached without --input, a matching \"describe this ...\" prompt is used.",
                    "--input prompt.txt"),
                new OptionHelp("--system <text>",
                    "System prompt for the conversation. Default: none (the model's own template default applies).",
                    "--system \"You are a terse assistant.\""),
                new OptionHelp("--system-file <path>",
                    "Read the system prompt from a UTF-8 text file instead of the command line. Default: none.",
                    "--system-file persona.txt"),
                new OptionHelp("--pdf <file>",
                    "PDF document input. A born-digital PDF has its text layer extracted and inlined into the " +
                    "message (no truncation); a scanned/image-only PDF is rendered to page images and needs a " +
                    "vision-capable model (see --mmproj). Combine with --input to ask a question about the " +
                    "document. Default: none.",
                    "--pdf report.pdf --input question.txt"),
                new OptionHelp("--tools <file>",
                    "JSON file with an array of tool/function definitions to expose to the model (function " +
                    "calling). Default: none.",
                    "--tools tools.json"),
                new OptionHelp("--think",
                    "Enable the model's thinking/reasoning mode on architectures that support it (e.g. Qwen3/3.5, " +
                    "Gemma 4). Default: off.",
                    "--think"),
                new OptionHelp("--max-tokens <N>",
                    "Maximum number of tokens to generate. Range: >= 1. Default: 100.",
                    "--max-tokens 1024"),
                new OptionHelp("--output <file>",
                    "Write the generated text to a file instead of stdout (for Qwen-Image-Edit: the output image, " +
                    "default edited.png). Default: print to stdout.",
                    "--output answer.txt"),
            }),
            ("Multimodal input (vision / audio / video models)", new[]
            {
                new OptionHelp("--image <file>",
                    "Image input for a vision model (PNG/JPEG/...). Repeatable: for Qwen-Image-Edit each extra " +
                    "--image adds a source picture the prompt can reference as \"Picture 1\", \"Picture 2\", ... " +
                    "Default: none.",
                    "--image photo.jpg"),
                new OptionHelp("--audio <file>",
                    "Audio input for an audio-capable model (e.g. Gemma 4). Default: none.",
                    "--audio speech.wav"),
                new OptionHelp("--video <file>",
                    "Video input: frames are extracted and fed to the vision encoder. Default: none.",
                    "--video clip.mp4"),
                new OptionHelp("--mmproj <path>",
                    "Multimodal projector (vision/audio encoder) GGUF that pairs with the model. Default: " +
                    "auto-detected next to the model for known architectures (gemma3/gemma4, Qwen3.5, Mistral3, " +
                    "Nemotron); pass it explicitly for anything else.",
                    "--mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf"),
            }),
            ("Compute backend and GPUs", new[]
            {
                new OptionHelp("--backend <type>",
                    "Compute backend: cpu, cuda (direct CUDA), mlx (Apple), ggml_cpu, ggml_metal, ggml_cuda, or " +
                    "ggml_vulkan. Default: ggml_cpu.",
                    "--backend ggml_cuda"),
                new OptionHelp("--gpu-device <N>",
                    "Vulkan device index for the ggml_vulkan backend on multi-GPU hosts (e.g. an integrated Intel " +
                    "GPU next to a discrete NVIDIA one). Range: >= 0 (see --list-gpus). Default: 0 " +
                    "(TS_GGML_VULKAN_DEVICE env var overrides).",
                    "--backend ggml_vulkan --gpu-device 1"),
                new OptionHelp("--list-gpus",
                    "List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit. No model needed.",
                    "--list-gpus"),
            }),
            ("Tensor parallelism (multi-GPU)", new[]
            {
                new OptionHelp("--tp <N>",
                    "Split the model across N GPUs on this machine (tensor parallelism): each GPU holds 1/N of " +
                    "every weight and the shards cooperate on every token. Use it when a model does not fit on one " +
                    "GPU. Range: 1 to the number of local GPUs. Applies to the cuda, ggml_cuda, and ggml_vulkan " +
                    "backends. " +
                    "Multi-GPU is implemented PER ARCHITECTURE, not per backend, and in two forms. Architectures that shard weights run true tensor parallelism. qwen4exp (Qwen3.8-Flash-Next) shards nothing, so --tp N runs it as a LAYER SPLIT instead - each GPU holds a contiguous run of whole layers, which is the same and only multi-GPU mode llama.cpp offers for it. That is a CAPACITY feature: it lets a model, context or resident-weight set that one GPU cannot hold fit across several, and is not expected to raise tok/s. The startup line says which mode actually ran. An architecture that supports neither says so on stderr and runs on one GPU rather than silently leaving the others idle. " +
                    "Default: 1 — no splitting (TENSORSHARP_TP_DEGREE env var overrides).",
                    "--backend ggml_cuda --tp 2"),
                new OptionHelp("--tp-node-id <N>",
                    "This node's 0-based ID for multi-node (distributed) tensor parallelism over TCP. Node 0 is " +
                    "the driver that owns sampling and prints the output; every other node runs a worker loop that " +
                    "mirrors the driver. Requires --tp-peers. Range: 0 to (number of nodes - 1). Default: none — " +
                    "single-node.",
                    "--tp 2 --tp-node-id 0 --tp-peers 192.168.1.10:9500,192.168.1.11:9500"),
                new OptionHelp("--tp-peers <list>",
                    "Comma-separated host:port list of ALL nodes in the distributed TP cluster, ordered by node " +
                    "ID; every node passes the identical list. Requires --tp-node-id. Default: none.",
                    "--tp-peers 192.168.1.10:9500,192.168.1.11:9500"),
            }),
            ("Sampling (batch runs are greedy; interactive chat samples - see below)", new[]
            {
                new OptionHelp("--temperature <f>",
                    "Sampling temperature: 0 = greedy (always the most probable token), higher = more random. " +
                    "Typical range: 0.0-2.0. Default: 0 for one-shot/batch runs, 0.8 for --interactive " +
                    "(overridden by the model's own general.sampling.temp when the GGUF ships one). " +
                    "Greedy chat degenerates into endless repetition on long answers, which is why " +
                    "--interactive does not default to it.",
                    "--temperature 0.7"),
                new OptionHelp("--top-k <N>",
                    "Keep only the K most probable tokens; 0 disables the filter. Typical range: 20-100. " +
                    "Default: 0 for one-shot/batch runs, 40 for --interactive (or the GGUF's general.sampling.top_k).",
                    "--top-k 40"),
                new OptionHelp("--top-p <f>",
                    "Nucleus sampling: keep the smallest token set whose cumulative probability exceeds P; 1.0 " +
                    "disables. Range: 0.0-1.0. Default: 1.0 for one-shot/batch runs, 0.95 for --interactive " +
                    "(or the GGUF's general.sampling.top_p).",
                    "--top-p 0.9"),
                new OptionHelp("--min-p <f>",
                    "Drop tokens whose probability is below min-p x the top token's probability; 0 disables. " +
                    "Range: 0.0-1.0. Default: 0 for one-shot/batch runs, 0.05 for --interactive.",
                    "--min-p 0.05"),
                new OptionHelp("--repeat-penalty <f>",
                    "Multiplicative penalty on recently generated tokens; 1.0 = none, > 1.0 discourages " +
                    "repetition. Typical range: 1.0-1.5. Default: 1.0.",
                    "--repeat-penalty 1.1"),
                new OptionHelp("--penalty-last-n <N>",
                    "How many of the most recent tokens the repeat/presence/frequency penalties consider; " +
                    "0 disables history penalties, -1 uses the whole history. Default: 64.",
                    "--penalty-last-n 128"),
                new OptionHelp("--presence-penalty <f>",
                    "Additive penalty on any token that has already appeared; 0 disables. Typical range: 0.0-2.0. " +
                    "Default: 0.",
                    "--presence-penalty 0.2"),
                new OptionHelp("--frequency-penalty <f>",
                    "Additive penalty proportional to how often a token has appeared; 0 disables. Typical range: " +
                    "0.0-2.0. Default: 0.",
                    "--frequency-penalty 0.3"),
                new OptionHelp("--seed <N>",
                    "Random seed for reproducible sampling; -1 = non-deterministic (time-based). Default: -1.",
                    "--seed 42"),
                new OptionHelp("--stop <text>",
                    "Stop sequence: generation halts when this string is produced (the string itself is not " +
                    "emitted). Repeat the flag for several. Default: none (the model's EOS token always stops).",
                    "--stop \"</s>\" --stop \"<|eot|>\""),
            }),
            ("Speculative decoding", new[]
            {
                new OptionHelp("--spec | --no-spec",
                    "Enable/disable speculative decoding: a drafter proposes the next few tokens and the trunk " +
                    "verifies them in ONE batched forward. Every emitted token still comes from a trunk row, so " +
                    "this is a speed path only - the output is what plain decoding would have produced. Engages " +
                    "on --input, --input-jsonl, --multi-turn-jsonl and --interactive. Must be passed BEFORE the " +
                    "model loads (it is what tells glm-dsa to page its ~3 GiB NextN layer into VRAM, which also " +
                    "leaves less room for the context). Not available under --tp N>1 on a checkpoint whose draft " +
                    "block borrows the trunk's LM head, which includes GLM-5.2. Accepted as --mtp-spec / " +
                    "--no-mtp-spec too. Default: off; env TS_SPEC (or TS_MTP_SPEC; glm-dsa also honours " +
                    "TS_GLM_MTP=1/0, which overrides both).",
                    "--model GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf --backend ggml_cuda --spec --chat"),
                new OptionHelp("--spec-type <name>",
                    "Which speculation ALGORITHM to draft with. 'auto' (default) uses whatever drafter the " +
                    "checkpoint carries: a per-token NextN/MTP head (GLM-5.2, Qwen 3.6, Gemma 4's separate " +
                    "assistant GGUF) or a block drafter (DeepSeek V4 DSpark, DFlash / DFlash2 on Muse-Glimmer and Qwen 3.8). " +
                    "'draft-head' and 'block' pin one of those explicitly. 'ngram' needs NO trained weights at " +
                    "all - it drafts by finding where the last few tokens occurred earlier in the context and " +
                    "proposing what followed, so it works on every model and is strong on summarizing, editing, " +
                    "translating, repetitive structured output and agentic loops, where the answer quotes the " +
                    "prompt. Env: TS_SPEC_TYPE.",
                    "--spec --spec-type ngram --spec-draft 8"),
                new OptionHelp("--spec-draft <N>",
                    "Maximum tokens drafted per speculative step. Also sizes the native graph cache at load, so " +
                    "it belongs on the same command line as --spec. Accepted as --mtp-draft too. Range: 1-64. " +
                    "Default: 8; env TS_SPEC_DRAFT (or TS_MTP_DRAFT).",
                    "--spec --spec-draft 4"),
                new OptionHelp("--spec-pmin <f>",
                    "Confidence gate below which drafting stops. What the number MEANS is the algorithm's " +
                    "business, so each picks its own default rather than sharing one: 0.75 for a per-token head " +
                    "(top-1 probability over its top-10 logits), 0.35 for a block drafter (the CUMULATIVE prefix " +
                    "probability, so the same number is far stricter), 0 for n-gram (where it scales the required " +
                    "match length instead). Accepted as --mtp-pmin too. Range: 0.0-1.0 (exclusive of 0). " +
                    "Env: TS_SPEC_PMIN (or TS_MTP_PMIN).",
                    "--spec --spec-draft 4 --spec-pmin 0.55"),
                new OptionHelp("--spec-draft-model <path>",
                    "Draft-head GGUF for architectures whose speculator weights ship as their own file " +
                    "(Gemma 4's gemma4-assistant). Loaded onto the target at startup so --spec can engage. " +
                    "Qwen 3.6 and GLM-5.2 embed their NextN block in the trunk GGUF and need no such flag. " +
                    "Accepted as --mtp-draft-model too. Default: none; env TS_SPEC_DRAFT_MODEL.",
                    "--spec --spec-draft-model gemma-4-12B-it-Q4_0-MTP.gguf"),
                new OptionHelp("--draft-model <path>",
                    "Block drafter GGUF that has to be resident before the model's layer split runs (DeepSeek " +
                    "V4's DSpark support module, the DFlash / DFlash2 drafters for Muse-Glimmer and Qwen 3.8). "
                    + "The drafter proposes a whole block of " +
                    "tokens per step and the trunk verifies it in one batched forward. Naming the file IS the " +
                    "request - such a drafter needs no --spec. Every emitted token is still drawn from a trunk " +
                    "row - with argmax under a greedy config, with your sampler otherwise - so output is " +
                    "unchanged either way. Default: none; env TS_DSV4_DSPARK.",
                    "--draft-model DSpark-drafter-Q2K-Q8-0731.gguf --temperature 0"),
                new OptionHelp("--spec-draft-n-max <N>",
                    "Older spelling of --spec-draft, kept for block drafters. Cap on tokens drafted per " +
                    "speculative block; a block drafter additionally clamps it to its trained block size " +
                    "(5 for DSpark). Default: the drafter's block size.",
                    "--spec-draft-n-max 3"),
                new OptionHelp("--spec-draft-conf-min <p>",
                    "Older spelling of --spec-pmin, kept for block drafters, where the gate is the CUMULATIVE " +
                    "acceptance probability (the product of the confidence head's per-position estimates). Lower " +
                    "drafts further and rolls back more; higher falls back to plain decode sooner. Range: " +
                    "0.0-1.0. Default: 0.35 for a block drafter, 0.75 for a per-token head.",
                    "--spec-draft-conf-min 0.5"),
            }),
            ("Interactive chat", new[]
            {
                new OptionHelp("-i | --interactive | --chat",
                    "Start an interactive multi-turn chat session in the terminal with KV-cache reuse across " +
                    "turns. In-session commands include /system, /model, /backend, /info, /clear, /exit. Honors " +
                    "--system, --think, --tools, and the sampling flags. Default: off (single-shot generation).",
                    "--model gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --chat"),
            }),
            ("Batch and scripted runs", new[]
            {
                new OptionHelp("--input-jsonl <file>",
                    "Batch mode: one JSON request per line, e.g. {\"id\": \"r1\", \"prompt\": \"...\", " +
                    "\"max_tokens\": 256} or a full {\"messages\": [...]} chat with optional per-request images, " +
                    "audios, sampling fields, and enable_thinking. Results go to stdout or --output. Default: none.",
                    "--input-jsonl requests.jsonl --output results.jsonl"),
                new OptionHelp("--multi-turn-jsonl <file>",
                    "Multi-turn chat simulation with KV-cache reuse (mirrors the web UI's behavior): each line is " +
                    "one user turn, either plain text or {\"content\": \"...\", \"max_tokens\": N, " +
                    "\"force_reset\": true}. Default: none.",
                    "--multi-turn-jsonl turns.jsonl"),
                new OptionHelp("--dump-prompt",
                    "Render the chat template over the input, print the exact prompt text and token IDs, and exit " +
                    "without running inference. Default: off.",
                    "--input prompt.txt --dump-prompt"),
            }),
            ("Mixture-of-Experts CPU offload", new[]
            {
                new OptionHelp("--n-cpu-moe <N> | -ncmoe <N>",
                    "Keep the routed MoE expert weights of the first N layers in system RAM and multiply them on " +
                    "the CPU; attention, norms, the router and the shared expert stay on the accelerator. This is " +
                    "what makes a 35B-A3B MoE fit beside a long-context KV cache on a 12-16 GB card. Pass 'all' " +
                    "for every layer. Default: 0 (everything on the accelerator), except DeepSeek V4 on the GPU " +
                    "backends, which auto-offloads the fewest layers that fit the visible VRAM (TS_N_CPU_MOE env " +
                    "var overrides).",
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
                    "KV cache precision: f32, f16, q8_0, or q4_0. Quantized caches trade small numerical drift " +
                    "for a 2-4x smaller cache. Default: auto — the backend/model pick (KV_CACHE_DTYPE env var " +
                    "overrides).",
                    "--kv-cache-dtype q8_0"),
                new OptionHelp("--paged-kv | --no-paged-kv",
                    "Enable/disable the cross-session paged KV cache (prefix reuse across requests; aliases " +
                    "--paged-kv-cache / --no-paged-kv-cache). Default: off.",
                    "--paged-kv"),
                new OptionHelp("--paged-kv-block-size <N>",
                    "Tokens per paged-KV block. Range: >= 1. Default: 256.",
                    "--paged-kv-block-size 128"),
                new OptionHelp("--paged-kv-ram-mb <N>",
                    "RAM budget for evicted KV blocks, in MB. Range: >= 1. Default: 1024.",
                    "--paged-kv-ram-mb 2048"),
                new OptionHelp("--paged-kv-ssd-dir <path>",
                    "Directory for the SSD spill tier of the paged KV cache. Default: disabled.",
                    "--paged-kv-ssd-dir /var/ts-kv-spill"),
                new OptionHelp("--paged-kv-ssd-mb <N>",
                    "SSD budget for spilled KV blocks, in MB. Range: >= 1. Default: 16384.",
                    "--paged-kv-ssd-mb 32768"),
                new OptionHelp("--paged-kv-quant-bits <b>",
                    "Quantize spilled KV blocks: 0 (off), 2, 4, or 8 bits. Default: 0.",
                    "--paged-kv-quant-bits 8"),
            }),
            ("Scheduling", new[]
            {
                new OptionHelp("--continuous-batching | --no-continuous-batching",
                    "Paged-attention continuous batching across concurrent requests (aliases --paged-batching / " +
                    "--no-paged-batching). Default: on.",
                    "--no-continuous-batching"),
            }),
            ("Benchmarks and diagnostics", new[]
            {
                new OptionHelp("--benchmark",
                    "Measure prefill and decode throughput (tok/s) with synthetic tokens. Decode output reports " +
                    "model-forward and greedy-sampling time separately. Tune with --bench-prefill / " +
                    "--bench-decode / --bench-runs. Default: off.",
                    "--benchmark --bench-prefill 512 --bench-decode 128"),
                new OptionHelp("--bench-prefill <N>",
                    "Prompt length (tokens) for --benchmark. Range: >= 1. Default: 32.",
                    "--bench-prefill 2048"),
                new OptionHelp("--bench-decode <N>",
                    "Number of decode steps for --benchmark. Range: >= 1. Default: 64.",
                    "--bench-decode 256"),
                new OptionHelp("--bench-runs <N>",
                    "Independent benchmark repetitions; the best (minimum-time) run is reported to filter warm-up " +
                    "noise. Range: >= 1. Default: 1.",
                    "--bench-runs 3"),
                new OptionHelp("--bench-chunked",
                    "Use the server-style chunked ForwardRefill path for benchmark prefill. Default: off.",
                    "--benchmark --bench-chunked"),
                new OptionHelp("--bench-fixed-tokens",
                    "Feed a predetermined decode-token stream and time inference without host greedy sampling, " +
                    "for llama-bench-style comparison. An untimed greedy correctness chain is still reported. Default: off.",
                    "--benchmark --bench-fixed-tokens"),
                new OptionHelp("--test",
                    "Run the built-in generation smoke-test prompts against the loaded model. Default: off.",
                    "--test"),
                new OptionHelp("--test-chunked-prefill",
                    "Correctness check: prefill the same prompt single-pass and chunked, then compare the decoded " +
                    "tokens — any divergence is a chunked-prefill bug. Tune with --correct-prefill / " +
                    "--correct-decode. Default: off.",
                    "--test-chunked-prefill"),
                new OptionHelp("--correct-prefill <N>",
                    "Prompt length (tokens) for --test-chunked-prefill. Range: >= 1. Default: 1500.",
                    "--correct-prefill 3000"),
                new OptionHelp("--correct-decode <N>",
                    "Decode steps compared by --test-chunked-prefill. Range: >= 1. Default: 8.",
                    "--correct-decode 16"),
                new OptionHelp("--bench-kvcache",
                    "Benchmark multi-turn chat with KV-cache reuse vs full re-prefill; reports per-turn prefill " +
                    "latency. Tune with --bench-kv-turns. Default: off.",
                    "--bench-kvcache --bench-kv-turns 6"),
                new OptionHelp("--bench-kv-turns <N>",
                    "Number of simulated chat turns for --bench-kvcache. Range: >= 1. Default: 4.",
                    "--bench-kv-turns 8"),
                new OptionHelp("--paged-bench",
                    "Benchmark the cross-session paged KV cache: pay the full prefill once, then measure how much " +
                    "a second identical-prefix request recovers. Tune with --paged-bench-prompt / " +
                    "--paged-bench-trials. Default: off.",
                    "--paged-bench"),
                new OptionHelp("--paged-bench-prompt <N>",
                    "Prompt length (tokens) for --paged-bench. Range: >= 1. Default: 2048.",
                    "--paged-bench-prompt 4096"),
                new OptionHelp("--paged-bench-trials <N>",
                    "Trials for --paged-bench. Range: >= 1. Default: 3.",
                    "--paged-bench-trials 5"),
                new OptionHelp("--warmup-runs <N>",
                    "Run the full inference path N times silently before the real pass so JIT/pipeline compilation " +
                    "and allocator growth don't skew timings. Range: >= 0. Default: 0.",
                    "--warmup-runs 1"),
                new OptionHelp("--test-templates <dir>",
                    "Run chat-template rendering tests against every GGUF in the directory and exit (no inference).",
                    "--test-templates C:\\models"),
            }),
            ("Image generation and editing (DiffusionGemma / Qwen-Image-Edit models)", new[]
            {
                new OptionHelp("--prompt <text>",
                    "Edit instruction for Qwen-Image-Edit (alternative to --input). Reference multiple source " +
                    "images as \"Picture 1\", \"Picture 2\", ... Default: empty.",
                    "--prompt \"Make the sky look like sunset\""),
                new OptionHelp("--cfg <f>",
                    "Classifier-free guidance scale for image editing. Higher follows the prompt more strongly " +
                    "but over-guides (distorts faces) past ~4. Typical range: 1.0-4.0. Default: 2.5 (Lightning " +
                    "LoRA switches it to 1.0).",
                    "--cfg 2.5"),
                new OptionHelp("--diffusion-steps <N>",
                    "Denoising steps. Range: >= 1. Default: 48 for DiffusionGemma; auto for Qwen-Image-Edit " +
                    "(model/LoRA-dependent).",
                    "--diffusion-steps 20"),
                new OptionHelp("--diffusion-seed <N>",
                    "Random seed for the diffusion sampler. Default: 0.",
                    "--diffusion-seed 7"),
                new OptionHelp("--diffusion-blocks <N>",
                    "DiffusionGemma only: number of generation blocks. Default: 0 — derived from --max-tokens and " +
                    "the model's canvas length.",
                    "--diffusion-blocks 4"),
                new OptionHelp("--width <px> / --height <px>",
                    "Fixed output size for Qwen-Image-Edit. Range: > 0, capped at what VRAM allows. Default: 0 — " +
                    "auto (source size, VRAM-clamped).",
                    "--width 1024 --height 768"),
                new OptionHelp("--qwen-image-vae <path>",
                    "Qwen-Image-Edit VAE GGUF. Default: same-directory scan next to the DiT model.",
                    "--qwen-image-vae qwen-image-vae.gguf"),
                new OptionHelp("--qwen-image-vl <path>",
                    "Qwen2.5-VL text-encoder GGUF for Qwen-Image-Edit. Default: same-directory scan.",
                    "--qwen-image-vl qwen-image-te-Qwen2.5-VL-7B-Q4_K_M.gguf"),
                new OptionHelp("--qwen-image-mmproj <path>",
                    "Vision projector GGUF for the Qwen-Image-Edit text encoder. Default: same-directory scan.",
                    "--qwen-image-mmproj Qwen2.5-VL-7B-mmproj-BF16.gguf"),
                new OptionHelp("--qwen-image-lora <path>",
                    "DiT LoRA (e.g. a Lightning step-distillation checkpoint), merged into the weights at load; " +
                    "also switches the sampling defaults (steps, cfg 1.0). Default: none.",
                    "--qwen-image-lora Qwen-Image-Edit-Lightning-8steps.safetensors"),
                new OptionHelp("--offload-cpu",
                    "Stream the DiT weights from RAM instead of holding them resident in VRAM: slower per step, " +
                    "but native ~1 MP edits fit on small cards. Default: auto (engages only when the target " +
                    "resolution does not fit beside the resident weights).",
                    "--offload-cpu"),
                new OptionHelp("--video-frames <N>",
                    "Video generation: number of output frames, snapped to the model's temporal grid " +
                    "(4k+1 for Wan, 17k+5 for MiniMax-H3); 1 = a single still image where the model allows it. " +
                    "Default: the model's own (33; 49 for Wan2.2-TI2V).",
                    "--video-frames 81"),
                new OptionHelp("--fps <N>",
                    "Video generation: playback frame rate of the saved MP4. Default: the model's training " +
                    "rate (16, or 24 for Wan2.2-TI2V). Models trained at a fixed rate — MiniMax-H3 at 24 fps — " +
                    "override any other value.",
                    "--fps 24"),
                new OptionHelp("--flow-shift <F>",
                    "FlowMatch timestep shift. Default: 0 — the model's official recipe (5.0 for Wan 2.2, " +
                    "12.0 for A14B T2V; Wan 2.1: 8.0 for the 1.3B model's video runs, else 3.0 at <= 480p, " +
                    "5.0 above; 12.0 for MiniMax-H3). On models with a joint audio stream this shifts the " +
                    "video stream only.",
                    "--flow-shift 5.0"),
                new OptionHelp("--sampler <name>",
                    "Sampler: unipc (the official Wan sampler; multistep predictor-corrector, better quality " +
                    "at the same step count) or euler. Default: the model's own (unipc for Wan).",
                    "--sampler unipc"),
                new OptionHelp("--negative-prompt <text>",
                    "Video generation: negative prompt for classifier-free guidance. Default: the model's " +
                    "official negative prompt. Unused at --cfg 1.0, where no negative pass runs.",
                    "--negative-prompt \"static, blurry\""),
                new OptionHelp("--cfg-cache-stride <N>",
                    "Guidance cache: run the unconditional CFG pass on one step in N and reuse the cached " +
                    "guidance direction in between (the first three steps and the last always recompute it). " +
                    "At 50 steps, 2 runs 77 of the 100 passes (1.30x faster) and 3 runs 70 (1.43x). This is an " +
                    "approximation — leave it off when matching a reference sample matters, and it has no " +
                    "effect at --cfg 1.0. Default: 0 (off).",
                    "--cfg-cache-stride 2"),
                new OptionHelp("--video-mode <mode>",
                    "How to interpret supplied images, on models with more than one conditioning " +
                    "mode. Default: inferred from what you pass. MiniMax-H3 accepts t2v (text " +
                    "only), i2v (the image IS the first frame and gets animated), fl2v (first and " +
                    "last frame), ref (the images are identity/appearance references for a NEW " +
                    "scene, not the first frame). i2v/fl2v need the fl2va checkpoint and ref needs " +
                    "ref2va — they are separate files. On the ref2va checkpoint a plain --image is " +
                    "taken as a reference, so clients that only attach one image work unchanged.",
                    "--video-mode i2v"),
                new OptionHelp("--end-image <file>",
                    "Last-frame conditioning image, on models that accept one (MiniMax-H3 first/last-frame " +
                    "mode). Combined with --image the clip is steered to start and end on the two frames.",
                    "--end-image last.png"),
                new OptionHelp("--ref-image <file>",
                    "Reference image for reference-conditioned models (MiniMax-H3 Ref2VA): the subject " +
                    "carries over while camera, background and composition come from the prompt. " +
                    "Repeatable up to 9; the prompt refers to them positionally as <Picture 1>, " +
                    "<Picture 2>, ... References are only ever scaled DOWN and keep their own aspect " +
                    "ratio, so the output size still comes from --width/--height.",
                    "--ref-image cat.png"),
                new OptionHelp("--ref-video <path>",
                    "Reference video clip for reference-conditioned models. Repeatable; referred to in the " +
                    "prompt as <Video 1>, <Video 2>, ... Accepts a video FILE or a directory of frames; " +
                    "either way it is resampled onto the model's own 24 fps and its own canvas. Pair a " +
                    "soundtrack with --ref-video-audio.",
                    "--ref-video clip/"),
                new OptionHelp("--ref-video-audio <file>",
                    "Soundtrack for the reference video at the SAME position: the first " +
                    "--ref-video-audio pairs with the first --ref-video, and so on. Separate from " +
                    "--ref-video because a container's audio track is not readable through the " +
                    "frame decoder. Omit it for a silent reference clip. WAV, MP3 or Ogg.",
                    "--ref-video-audio clip.wav"),
                new OptionHelp("--ref-audio <file>",
                    "Reference audio clip for reference-conditioned models. Repeatable; referred to in the " +
                    "prompt as <Audio 1>, <Audio 2>, ... WAV, MP3 or Ogg, resampled to the audio VAE's " +
                    "32 kHz stereo and truncated to the generated clip's duration.",
                    "--ref-audio theme.wav"),
                new OptionHelp("--no-audio",
                    "Skip audio decoding on models that generate an audio track jointly with the video " +
                    "(MiniMax-H3). Saves the audio VAE's time and memory when only the picture is wanted. " +
                    "Ignored by video-only models.",
                    "--no-audio"),
                new OptionHelp("--video-vae <path>",
                    "Video VAE (wan_2.1_vae.safetensors, or Wan2.2_VAE.safetensors for TI2V-5B; " +
                    "minimax_h3_video_vae_fp16.safetensors for MiniMax-H3). Default: same-directory scan next " +
                    "to the DiT model, VAE/ subfolders included (TS_VIDEO_VAE env var also works). " +
                    "The former spelling --wan-vae is still accepted.",
                    "--video-vae Wan2.2_VAE.safetensors"),
                new OptionHelp("--video-text-encoder <path>",
                    "Text-encoder GGUF for video generation (UMT5-XXL for Wan, Qwen3-VL-32B for MiniMax-H3). " +
                    "Default: same-directory scan (TS_VIDEO_TEXT_ENCODER env var also works). Also spelled " +
                    "--video-te; the former spelling --wan-te is still accepted.",
                    "--video-text-encoder umt5-xxl-encoder-Q8_0.gguf"),
                new OptionHelp("--video-dit2 <path>",
                    "Second diffusion expert on dual-expert models (Wan 2.2 A14B's high/low-noise pair). " +
                    "Default: auto-resolved by filename next to the first expert (TS_VIDEO_DIT2 env var also " +
                    "works). The former spelling --wan-dit2 is still accepted.",
                    "--video-dit2 wan2.2_i2v_low_noise.gguf"),
                new OptionHelp("--audio-vae <path>",
                    "Audio VAE for models that generate an audio track jointly with the video " +
                    "(minimax_h3_audio_vae_fp32.safetensors). Without it such a model still runs and produces " +
                    "video, just no audio (TS_VIDEO_AUDIO_VAE env var also works).",
                    "--audio-vae minimax_h3_audio_vae_fp32.safetensors"),
            }),
            ("Configuration file", new[]
            {
                new OptionHelp("--config <path>",
                    "Read options from a JSON file whose keys are the same long option names listed here (with or " +
                    "without the leading --). Anything also passed on the command line overrides the file; when " +
                    "the flag is repeated, later files win over earlier ones. String/number values map to " +
                    "'--key value', true maps to the bare '--key' switch, and an array maps to a repeated flag " +
                    "(e.g. \"stop\": [..]). A \"variables\" object lets values share ${name} references; a file " +
                    "option may instead be an object { \"path\": \"...\", \"urls\": [ \"...\" ] } that " +
                    "auto-downloads on first run. See the config/ folder and config/README.md for examples.",
                    "--config run.json --backend ggml_cuda"),
            }),
            ("Logging", new[]
            {
                new OptionHelp("--log-level <level>",
                    "Minimum log level: trace, debug, info, warning, error, critical, or none. Default: info " +
                    "(TENSORSHARP_LOG_LEVEL env var overrides).",
                    "--log-level debug"),
                new OptionHelp("--log-dir <path>",
                    "Directory for the rolling log file. Default: logs/ next to the executable " +
                    "(TENSORSHARP_LOG_DIR env var overrides).",
                    "--log-dir /tmp/ts-logs"),
                new OptionHelp("--log-file <on|off>",
                    "Enable/disable the log file. Default: on (TENSORSHARP_LOG_FILE=0 disables).",
                    "--log-file off"),
                new OptionHelp("--log-console <on|off>",
                    "Enable/disable console log lines (generated text always prints regardless). Default: on.",
                    "--log-console off"),
            }),
            ("Help", new[]
            {
                new OptionHelp("--help",
                    "Show this help and exit (also shown when the CLI is started with no arguments; aliases -h, " +
                    "-?, /?).",
                    "--help"),
            }),
        };

        public static void PrintUsage(TextWriter writer)
        {
            writer.WriteLine("Usage: TensorSharp.Cli --model <path.gguf> [options]");
            writer.WriteLine();
            writer.WriteLine("Runs local LLM inference from the terminal: single-shot generation, interactive chat,");
            writer.WriteLine("multimodal input (image/audio/video/PDF), JSONL batches, image editing, and benchmarks.");
            writer.WriteLine("Options may also come from a JSON file via --config (command line wins).");

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
            writer.WriteLine("  TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --input prompt.txt --max-tokens 512");
            writer.WriteLine("  TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --chat --think");
            writer.WriteLine("  TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf --image photo.jpg");
            writer.WriteLine("  TensorSharp.Cli --model Qwen3.5-35B-A3B-Q4_K_M.gguf --backend ggml_cuda --tp 2 --chat    (split across 2 GPUs)");
            writer.WriteLine("  TensorSharp.Cli --model DeepSeek-V4-Flash-00001-of-00005.gguf --backend ggml_cuda --draft-model DSpark-drafter.gguf --temperature 0 --chat    (block speculative decoding)");
            writer.WriteLine("  TensorSharp.Cli --model Qwen-Image-Edit-2511-Q4_K_M.gguf --image in.png --prompt \"Turn it into watercolor\" --output out.png");
            writer.WriteLine("  TensorSharp.Cli --model model.gguf --backend ggml_cuda --benchmark --bench-prefill 512 --bench-decode 128");
            writer.WriteLine("  TensorSharp.Cli --config run.json    (read options from a file)");
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
