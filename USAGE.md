# Usage
[English](USAGE.md) | [中文](USAGE_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation. Quick-start commands are in the [README](README.md#quick-start); configuration files are in [config/README.md](config/README.md).

## Compute Backends

| Backend | Flag | Best fit | Description |
|---|---|---|---|
| Direct CUDA/cuBLAS | `--backend cuda` | NVIDIA inference and experimentation | Uses the CUDA Driver API, cuBLAS GEMM, PTX kernels for common float32 ops (fill, unary, binary, ternary, activations, RMSNorm, softmax, RoPE/RoPEEx, SDPA, GQA prefill/decode, causal mask, gather/concat), and native quantized matmul/get-rows for supported GGUF quant types. Unsupported ops route through CPU fallbacks while preserving tensor semantics. |
| MLX Metal | `--backend mlx` | Apple Silicon (alternative to GGML Metal) | GPU-accelerated path built on [mlx-c](https://github.com/ml-explore/mlx-c). Implements quantized ops (Q4_K_M, Q8_0, Q5_K, Q6_K, IQ2_XXS, IQ4_XS, IQ4_NL, MXFP4, etc.) without dequantizing to FP32, fused decode/prefill Metal kernels (fused QKV preprocess, fused gate+up+SiLUMul MoE, fused multi-dim KV write), compiled-graph kernels, async worker dispatch with periodic `async_eval` to overlap GPU/CPU work, batched MoE decode with stacked expert weight slabs, MoE expert offload, GGUF mmap pinned in physical RAM via `mlock(2)`, host-derived allocator caps (`TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB`), and a CPU fallback for ops that aren't yet wired up. Requires `libmlxc` (built locally by `TensorSharp.Backends.MLX/build-native-macos.sh` or located via `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR`). |
| GGML Metal | `--backend ggml_metal` | Apple Silicon (default on macOS) | GPU-accelerated via Apple Metal. Quantized weights are mapped zero-copy from the GGUF file into Metal command buffers via host-pointer buffers, so the resident set stays close to the on-disk model size. |
| GGML CUDA | `--backend ggml_cuda` | NVIDIA inference through ggml | GPU-accelerated via GGML CUDA on Windows or Linux. Quantized weights are uploaded to device memory once at load time and the host copy is released afterwards. |
| GGML Vulkan | `--backend ggml_vulkan` | Vendor-neutral GPU inference through ggml | GPU-accelerated via GGML Vulkan on Windows or Linux — runs on AMD, Intel, and NVIDIA GPUs with a Vulkan 1.3 driver, using cooperative-matrix shaders (KHR coopmat / NV coopmat2) where the driver supports them. Weights are device-resident like GGML CUDA and the same fused whole-model decode/prefill graphs are used. Enabled automatically at native build time when the machine has a Vulkan runtime (loader installed); the build downloads a portable Vulkan toolchain (headers, glslc, SPIRV-Headers, and on Windows a loader import lib) via `eng/fetch-vulkan-toolchain.ps1` / `eng/fetch-vulkan-toolchain.sh` when no Vulkan SDK or distro dev packages are installed. Opt out with `--no-vulkan` (or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`). |
| GGML CPU | `--backend ggml_cpu` | Native CPU kernels | CPU inference using native GGML with optimized kernels. Quantized weights are mapped zero-copy from the GGUF file. |
| Pure C# CPU | `--backend cpu` | Portability and debugging | Portable CPU inference with no native dependencies. |

**DeepSeek V4 Flash is the exception to the table above.** Its 284B compressed-sparse-attention MoE stack runs through one of three dedicated whole-model executors rather than the generic per-op path: a direct-CUDA engine (`--backend cuda`), the native ggml executor (`--backend ggml_cuda` / `ggml_vulkan`), and a 100% pure-C# CPU executor (`--backend cpu`) that serves quantized weights straight from the memory-mapped GGUF shards. All three layer-split the weights across every visible GPU (or, on CPU, stream them from the mapped shards), so a model far larger than one card still runs. See the [DeepSeek V4 card](docs/models/deepseek4.md).

## Configuration file (CLI + Server)

Both `TensorSharp.Cli` and `TensorSharp.Server` can read their options from a JSON
file passed with `--config`, instead of (or in addition to) a long command line:

```bash
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --config config/server-basic.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll       --config config/cli-basic.json
```

**Command-line options always win.** File values are applied first, then anything
you also pass on the command line overrides them — so one file can be reused across
machines while you override just what differs (`--config config/server-basic.json --backend ggml_cpu`).
Repeat `--config` to layer files; later files win over earlier ones.

The keys are the same long option names listed below (with or without the leading
`--`). Comments (`//`, `/* */`) and trailing commas are allowed.

| JSON value | Becomes | Example |
|---|---|---|
| string / number | `--key value` | `"max-tokens": 4096` → `--max-tokens 4096` |
| `true` | the bare switch `--key` | `"continuous-batching": true` → `--continuous-batching` |
| `false` / `null` | nothing (use the negation key, e.g. `"no-continuous-batching": true`) | |
| array | a repeated flag | `"stop": ["</s>", "<\|eot\|>"]` → `--stop </s> --stop <\|eot\|>` |
| object | a downloadable file (see below) | `{ "path": "...", "urls": ["..."] }` |

**Variables.** Define shared values once under `"variables"` and reference them
with `${name}` in any string value. A `${name}` not defined there falls back to an
environment variable of the same name, and variables may reference other variables.
Declare as many roots as you need — models in different folders each get their own.

```json
{
  "variables": { "modelRoot": "C:/models" },
  "backend": "ggml_cuda",
  "model": "${modelRoot}/Qwen3.5-9B-Q8_0.gguf",
  "mmproj": "${modelRoot}/Qwen3.5-mmproj-F16.gguf"
}
```

**Auto-download.** Any file option can be an object with a local `path` and one or
more `urls` instead of a plain string. If `path` is missing it is downloaded from
the first working URL (mirrors are tried in order), saved there, and reused on every
later run; download progress is printed to stderr. An optional `sha256` verifies a
freshly downloaded file.

```json
{
  "backend": "ggml_cuda",
  "model": {
    "path": "C:/models/Qwen3.5-9B-Q8_0.gguf",
    "urls": [ "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf" ]
  }
}
```

Ready-to-use examples live in [`config/`](config/) (`cli-basic.json`,
`server-basic.json`, `variables.json`, `auto-download.json`, `qwen-image-edit.json`)
— each uses real, public, ungated URLs, so it works on a fresh machine. See
[`config/README.md`](config/README.md) for the full reference.

## Console Application

Running `TensorSharp.Cli` with no arguments prints the full parameter reference —
every option with its description, default, range, and an example — and exits
before any logging or model machinery starts; `--help` (also `-h`, `-?`, `/?`)
does the same.

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --help
```

```bash
# Text inference
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Text inference on Windows/Linux + NVIDIA GPU
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# Interactive turn-by-turn chat (REPL) with KV cache reuse and slash commands
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal --interactive
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal -i \
    --system "You are a terse assistant." --temperature 0.7 --top-p 0.9 --think

# Image inference (Gemma 3/4, Qwen 3.5-family)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --image photo.png --backend ggml_metal

# Video inference (Gemma 4)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --video clip.mp4 --backend ggml_metal

# Audio inference (Gemma 4)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --audio speech.wav --backend ggml_metal

# PDF document input: born-digital PDFs are text-extracted and inlined into the
# prompt; scanned PDFs become page images and need a vision model (--mmproj or a
# built-in vision encoder). --input provides the instruction over the document.
echo "Summarize the key findings of this paper." > question.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --pdf paper.pdf --input question.txt \
    --max-tokens 300 --backend ggml_metal

# DiffusionGemma text-diffusion generation
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <diffusion-gemma.gguf> --input prompt.txt --backend ggml_metal \
    --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0

# Qwen-Image-Edit image editing (prompt + input image -> edited image)
# The VAE + Qwen2.5-VL text-encoder companions are resolved next to the DiT GGUF
# (or set --qwen-image-vae / --qwen-image-vl / --qwen-image-mmproj).
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <qwen-image-edit-DiT.gguf> --image input.png \
    --prompt "Make the sky a dramatic sunset." --output edited.png \
    --backend ggml_cuda --diffusion-steps 30 --cfg 2.5 --diffusion-seed 0

# Wan video generation (prompt -> H.264 MP4). The UMT5-XXL text-encoder GGUF and
# video-VAE companions are resolved next to the DiT GGUF (or set --wan-te /
# --wan-vae). Wan 2.1 T2V, Wan 2.2 TI2V-5B, and Wan 2.2 A14B (both experts)
# are auto-detected. See docs/models/wan.md.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "a lovely cat walking through a garden" --output cat.mp4 \
    --width 832 --height 480 --video-frames 49 --backend ggml_cuda \
    --diffusion-seed 7

# Wan 2.2 image-to-video: --image supplies the first frame; the prompt controls
# motion, camera and scene changes (TI2V-5B or I2V-A14B checkpoints).
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "the cat runs toward the camera, cinematic tracking shot" \
    --image first_frame.png --output cat_run.mp4 --backend ggml_cuda

# Thinking / reasoning mode
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# Tool calling
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# With sampling parameters
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# DSpark block speculative decoding (DeepSeek V4): point --model at the first shard
# and --draft-model at the DSpark drafter GGUF. Needs greedy sampling.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <DeepSeek-V4-Flash-...-00001-of-00005.gguf> \
    --backend ggml_cuda --draft-model <DSpark-drafter.gguf> \
    --input prompt.txt --max-tokens 200 --temperature 0

# Batch processing (JSONL)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# Multi-turn chat simulation with KV-cache reuse (mirrors the web UI behavior)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# Throughput benchmark: best-of-N prefill and decode timing
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# KV-cache reuse benchmark: measure prefill speedup across multiple chat turns
# (compares with-cache vs forced-reset prefill latency for an 8-turn conversation)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --bench-kvcache --bench-kv-turns 4 --max-tokens 64

# Inspect the rendered prompt and tokenization without running inference
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --dump-prompt

# Compare hardcoded fallback templates against GGUF Jinja2 templates for every
# *.gguf file in a directory (useful when adding new architectures)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --test-templates ~/models

# Tensor parallelism: split a model across 2 GPUs in one process
# (--backend cuda, ggml_cuda, or ggml_vulkan)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend cuda --tp 2

# Distributed tensor parallelism: 2 nodes × 2 GPUs each (4 GPUs total)
# Node 0:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# Node 1:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
```

**Command-line options:**

| Option | Description |
|---|---|
| `--model <path>` | Path to a GGUF model file (required) |
| `--input <path>` | Text file containing the user prompt |
| `--input-jsonl <path>` | JSONL file with batch requests (one JSON per line) |
| `--multi-turn-jsonl <path>` | JSONL file for multi-turn chat simulation with KV cache reuse |
| `--output <path>` | Write generated text to this file |
| `--image <path>` | Image file for vision inference |
| `--video <path>` | Video file for video inference |
| `--audio <path>` | Audio file (WAV, MP3, OGG) for audio inference |
| `--pdf <path>` | PDF document input (one-shot mode). Born-digital PDFs have their complete text layer extracted and inlined into the prompt (page cap via `TS_PDF_MAX_PAGES`); scanned PDFs are rasterized to page images and require a vision model (`--mmproj` or a built-in vision encoder). `--input` text becomes the instruction over the document. |
| `--mmproj <path>` | Path to the multimodal projector GGUF file |
| `--max-tokens <N>` | Maximum tokens to generate (default: 100) |
| `--backend <type>` | Compute backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan` |
| `--gpu-device <N>` | Vulkan device index for the `ggml_vulkan` backend on multi-GPU hosts (e.g. an integrated Intel GPU next to a discrete NVIDIA one). Defaults to device 0; use `--list-gpus` to see the indices. Also settable via the `TS_GGML_VULKAN_DEVICE` env var. |
| `--list-gpus` | List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit |
| `--n-cpu-moe <N>` / `-ncmoe <N>` | Keep the routed Mixture-of-Experts weights of the first N layers in system RAM and multiply them on the CPU; attention, norms, the router and the always-active shared expert stay on the accelerator (llama.cpp's `--n-cpu-moe` equivalent). This is what lets a 35B-A3B MoE fit beside a long-context KV cache on a 12-16 GB card. Pass `all` for every layer. Default: 0 on every architecture, DeepSeek V4 included — a model that does not fit is refused at load with the number of layers that would make it fit, rather than silently offloaded (env `TS_N_CPU_MOE`). |
| `--cpu-moe` / `-cmoe` | Shorthand for `--n-cpu-moe all`. Default: off (env `TS_CPU_MOE`). |
| `--cpu-moe-threads <N>` | Worker threads for the host-side expert matmul. Default: **half** the CPU parallelism this process can actually use (`hardware_concurrency` clamped by the scheduler affinity mask and the cgroup CPU quota) on hosts with more than 8, all but one below that. The other half is not waste — the accelerator submission threads, and in `TensorSharp.Server` Kestrel and the scheduler, have to be schedulable too, and .NET sizes its own pool from the machine's CPU count rather than the cgroup quota. Sizing this near the quota is a cliff, not a slope: on a 95-CPU quota the hosted 26B MoE measured 20.7 tok/s at 64 threads and 8.2 at 71. Raise it on a dedicated box (env `TS_CPU_MOE_THREADS`). |

**Backend support.** MoE CPU offload is implemented on the GGML backends
(`ggml_cuda`, `ggml_vulkan`, `ggml_metal`, `ggml_cpu`) for every MoE
architecture, and on the pure-C# `cuda` backend for DeepSeek V4 only — that
engine serves experts from its own stacked-expert device buffer everywhere
else. Asking for offload on a combination that does not implement it prints a
`[moe-offload] WARNING` and proceeds without saving VRAM, rather than failing
quietly. Measured on gemma-4-26B-A4B (`--cpu-moe`, peak VRAM): `ggml_cuda`
15244 → 5756 MiB; `cuda` 14261 → 14253 MiB (no-op, warns).
| `--kv-cache-dtype <type>` | KV cache precision: `f32`, `f16`, `q8_0`, or `q4_0` (default: auto — the backend/model pick; env `KV_CACHE_DTYPE`). Half-precision / quantized KV caches reduce memory at the cost of small numerical drift; `q4_0` (~0.56 bytes/elem, ~1/7 of f32) is the most aggressive tier for very long (128K–256K) contexts where the KV cache dominates memory. Block-quantized caches (`q8_0`/`q4_0`) require the native GGML flash path. |
| `--interactive` / `-i` | Start an interactive REPL chat session (turn-by-turn input/output) with KV cache reuse, slash commands, hot-swappable model/backend/projector, file attachments (image, audio, video, text) and live sampling tuning. See the **Interactive REPL commands** section below for the full list. |
| `--system <text>` | System prompt to seed the interactive session (overridden inside the REPL by `/system`) |
| `--system-file <path>` | Read the initial system prompt from a UTF-8 text file (alternative to `--system`) |
| `--think` | Enable thinking/reasoning mode (chain-of-thought) |
| `--tools <path>` | JSON file with tool/function definitions |
| `--draft-model <path>` | Speculative-decoding drafter GGUF for architectures whose drafter ships as its own file — today DeepSeek V4's DSpark support module (see [DeepSeek V4](docs/models/deepseek4.md#dspark-speculative-decoding)). It drafts a whole block per step and the trunk verifies it in one batched forward, so greedy output is unchanged. Engages on every single-sequence path (`--input`, `--input-jsonl`, `--multi-turn-jsonl`, `--interactive`) with `--backend cuda` or `--backend ggml_cuda`, and requires a pure-argmax sampler: any temperature, top-k/p, or repetition/presence/frequency penalty turns it off. Env: `TS_DSV4_DSPARK`. |
| `--spec-draft-n-max <N>` | Cap on tokens drafted per speculative block (default: the drafter's trained block size, 5 for DSpark) |
| `--spec-draft-conf-min <p>` | Minimum cumulative acceptance probability — the product of the confidence head's per-position estimates — for a drafted position to be kept (default: `0.35`). Lower drafts further and rolls back more; higher falls back to plain decode more often. |
| `--temperature <f>` | Sampling temperature (0 = greedy) |
| `--top-k <N>` | Top-K filtering (0 = disabled) |
| `--top-p <f>` | Nucleus sampling threshold (1.0 = disabled) |
| `--min-p <f>` | Minimum probability filtering (0 = disabled) |
| `--repeat-penalty <f>` | Repetition penalty (1.0 = none) |
| `--presence-penalty <f>` | Presence penalty (0 = disabled) |
| `--frequency-penalty <f>` | Frequency penalty (0 = disabled) |
| `--seed <N>` | Random seed (-1 = non-deterministic) |
| `--stop <string>` | Stop sequence (can be repeated) |
| `--dump-prompt` | Render the prompt + tokenization and exit (no generation) |
| `--benchmark` | Run a synthetic prefill/decode throughput benchmark |
| `--bench-prefill <N>` | Synthetic prefill length in tokens (default: 32) |
| `--bench-decode <N>` | Synthetic decode length in tokens (default: 64) |
| `--bench-runs <N>` | Number of benchmark runs; reports best and average (default: 1) |
| `--bench-kvcache` | Run a multi-turn KV-cache reuse benchmark (with-cache vs forced-reset prefill) |
| `--bench-kv-turns <N>` | Number of conversation turns for `--bench-kvcache` (default: 4, max: 8) |
| `--bench-chunked` | Run a chunked-prefill micro-benchmark (Gemma 4) |
| `--warmup-runs <N>` | Number of throw-away forward passes before timing real text / multimodal prompts (default: 0) |
| `--test-chunked-prefill` | Run the chunked-prefill correctness check (compares chunked vs non-chunked logits) |
| `--correct-prefill <N>` | Prompt length used by `--test-chunked-prefill` |
| `--correct-decode <N>` | Decode length used by `--test-chunked-prefill` |
| `--diffusion-steps <N>` | DiffusionGemma denoising steps per block (default: 48). For Qwen-Image-Edit, the FlowMatch-Euler step count — omit for auto (30, or the step count of a loaded Lightning LoRA). |
| `--diffusion-seed <N>` | DiffusionGemma deterministic sampler seed (default: 0) |
| `--diffusion-blocks <N>` | DiffusionGemma block-autoregressive canvas count. `0` derives the count from `--max-tokens` and the model canvas length. |
| `--image <path>` | Input image for Qwen-Image-Edit (also the image input for multimodal chat). Required to trigger image-edit mode on a `qwen_image` DiT GGUF. |
| `--prompt <text>` | Qwen-Image-Edit edit instruction (falls back to `--input` file contents if omitted). |
| `--output <path>` | Qwen-Image-Edit output PNG path (default: `edited.png`). |
| `--cfg <F>` | Qwen-Image-Edit true-CFG guidance scale (`<= 1` disables the negative pass). Omit for auto: 2.5 (the Qwen-Image-Edit-2511 recommendation; 4.0 over-guides and distorts faces), or 1.0 when a Lightning LoRA is loaded. Shares `--diffusion-steps` / `--diffusion-seed` for step count and seed. |
| `--qwen-image-vae <path>` | Override the resolved Qwen-Image VAE companion (`.gguf` or `.safetensors`). |
| `--qwen-image-vl <path>` | Override the resolved Qwen2.5-VL-7B text-encoder GGUF. |
| `--qwen-image-mmproj <path>` | Override the resolved Qwen2.5-VL mmproj (vision grounding) GGUF. |
| `--qwen-image-lora <path>` | Qwen-Image-Edit Lightning distillation LoRA (`.safetensors`), merged into the DiT at load time. Auto-derives the step count (e.g. 4 or 8) and switches CFG to 1.0. Env: `TS_QWEN_IMAGE_LORA`. |
| `--video-frames <N>` | Wan video frame count, snapped to `4k+1` (default: 33; 49 for Wan2.2-TI2V). `1` generates a still image (use `--output out.png`). |
| `--fps <N>` | Wan video playback frame rate of the saved MP4 (default: 16; 24 for Wan2.2-TI2V). |
| `--flow-shift <F>` | Wan FlowMatch timestep shift (default: the model's official recipe — 5.0 for Wan 2.2, 12.0 for A14B T2V, 8.0/3.0/5.0 for Wan 2.1). |
| `--sampler <name>` | Wan sampler: `unipc` (official Wan sampler, default) or `euler`. |
| `--negative-prompt <text>` | Wan negative prompt (default: the official Wan negative prompt). |
| `--wan-vae <path>` | Override the resolved Wan video VAE (`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`). Env: `TS_WAN_VAE`. |
| `--wan-te <path>` | Override the resolved UMT5-XXL text-encoder GGUF. Env: `TS_WAN_TE`. Wan 2.2 A14B additionally resolves the second high/low-noise expert automatically (env: `TS_WAN_DIT2`). |
| `--tp <N>` | Tensor parallelism degree — split the model across N GPUs in a single process (default: `1`). Requires `--backend cuda`, `ggml_cuda`, or `ggml_vulkan`. See [Tensor Parallelism & Distributed Inference](#tensor-parallelism--distributed-inference). |
| `--tp-node-id <N>` | This node's 0-based ID for multi-node (distributed) tensor parallelism. Requires `--tp-peers`. |
| `--tp-peers <list>` | Comma-separated `host:port` list of all nodes in the distributed TP cluster (e.g. `192.168.1.10:9500,192.168.1.11:9500`). Requires `--tp-node-id`. |
| `--test` | Run built-in tokenizer + Qwen3 chat-template + ollama-comparison tests |
| `--test-templates <dir>` | Validate hardcoded chat templates against GGUF Jinja2 templates for every *.gguf in `<dir>` |
| `--config <path>` | Read options from a JSON config file (command-line options override it). Supports `${variables}` and auto-downloading models via `{ "path": ..., "urls": [...] }`. Repeatable. See [Configuration file](#configuration-file-cli--server). |
| `--log-level <lvl>` | Console + file logger level: `trace`, `debug`, `info`, `warning`, `error`, `critical`, `off` |
| `--log-dir <path>` | Directory for the JSON-line file logger (default: `<binDir>/logs`) |
| `--log-file <0\|1>` | Disable (`0`) or enable (`1`) the file logger (default: enabled) |
| `--log-console <0\|1>` | Disable (`0`) or enable (`1`) the console logger (default: enabled) |

The CLI recognizes a small set of legacy projector filenames beside the model, but current repositories often use different names. Pass the downloaded file explicitly with `--mmproj` for reliable multimodal runs. `TensorSharp.Server` never auto-detects the projector.

**JSONL input format:**

Each line is a JSON object with `messages`, optional `prompt`, and optional sampling parameters:

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

**Interactive REPL commands:**

Once the CLI is launched with `--interactive` / `-i`, you can drive the running session with slash commands. Type `/help` (or `/?`) inside the REPL for the same list. Anything that does not start with `/` is treated as a user turn.

The prompt header summarizes the current state on every turn — model, backend, architecture, context length, projector, conversation depth, and any attachments queued for the next turn (e.g. `[turn 3 (2 attachments pending)]> `). Press Ctrl+C while generating to interrupt the current reply; press Ctrl+C at the prompt to exit.

Conversation:

| Command | Description |
|---|---|
| `/help`, `/?` | Show all interactive commands |
| `/exit`, `/quit` | Leave the session |
| `/reset`, `/new` | Clear conversation history and KV cache |
| `/history` | Print the conversation history |
| `/save <file>` | Append the current transcript to a UTF-8 file |
| `/system <text>` | Set the system prompt (empty argument clears it). Resets KV cache. |
| `/think on\|off` | Toggle thinking/reasoning mode for supported models |
| `/multiline on\|off` | Toggle multi-line input (terminate the message with a single `.` on its own line) |

Model and runtime:

| Command | Description |
|---|---|
| `/info`, `/status` | Show the loaded model, backend, architecture, context/vocab size, projector, conversation depth, and pending attachments |
| `/model <path>` | Load a different `.gguf` model on the current backend (resets the session) |
| `/backend <name>` | Reload the current model on a different backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan` |
| `/mmproj <path>` | Load (or replace) the multimodal projector for the current model. Aliases: `/projector` |

Sampling (live, persists across turns):

| Command | Description |
|---|---|
| `/sampling`, `/show` | Print the current sampling configuration |
| `/max <N>` | Maximum reply length in tokens |
| `/temp <float>` | Sampling temperature (0 = greedy) |
| `/topk <int>` | Top-K filtering (0 = disabled) |
| `/topp <float>` | Top-P / nucleus threshold (1.0 = disabled) |
| `/minp <float>` | Min-P filtering (0 = disabled) |
| `/repeat <float>` | Repetition penalty (1.0 = none) |
| `/presence <float>` | Presence penalty |
| `/frequency <float>` | Frequency penalty |
| `/seed <int>` | Random seed (-1 = non-deterministic) |
| `/stop <text>` | Add a stop sequence |
| `/clearstop` | Remove all stop sequences |

Uploads (queued for the next user turn, then auto-cleared after the turn):

| Command | Description |
|---|---|
| `/image <path>`, `/img <path>` | Attach an image (vision-capable models only) |
| `/audio <path>` | Attach an audio file (Gemma 4) |
| `/video <path>`, `/vid <path>` | Attach a video; frames are extracted automatically (Gemma 4) |
| `/text <path>`, `/file <path>`, `/txt <path>` | Inline up to the first 256 KiB of a UTF-8 text/markdown/csv/code file into the next prompt |
| `/clearattach` | Drop any pending image/audio/video/text attachments without sending a turn |

Quoted paths (single or double quotes) are accepted, so drag-and-drop from a file manager works on macOS. Multimodal commands require a multimodal projector to be loaded — pass `--mmproj` at startup or use `/mmproj <path>` from the REPL.

## Web Application

Run these commands from the repository root after building:

```bash
# Start the server with the exact hosted model
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --backend ggml_cuda

# Multimodal models: host an explicit projector too
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda

# Wan video generation: use 121 frames at 24 fps (about five seconds) whenever
# the Web UI or an API request does not supply its own frames / fps value.
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/Wan2.2-TI2V-5B.gguf --backend ggml_cuda \
    --video-frames 121 --fps 24

# Configure server-wide default sampling parameters
# (used whenever a request does not override the value itself)
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42 \
    --stop "</s>" --stop "<|endoftext|>"

# Read all of the above from a reusable JSON file (auto-downloads the model on
# first run). See the Configuration file section and config/ for examples.
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --config config/server-basic.json
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --config config/server-basic.json --backend ggml_cpu
```

Open `http://localhost:5000` in your browser — the root URL serves the chat UI (`GET /health` is the liveness endpoint). The web interface supports:

- Multi-turn chat conversations
- Per-tab chat sessions: each browser tab owns its own tracked conversation history; KV blocks are owned by the inference engine
- A single hosted GGUF selected explicitly with `--model`
- An explicit hosted multimodal projector via `--mmproj` when needed
- Full text and PDF document uploads, plus image, video, and audio uploads for multimodal inference (up to 500 MB)
- Thinking/reasoning mode toggle
- Tool calling with function definitions
- Streaming token generation via Server-Sent Events
- DiffusionGemma denoising previews when a `diffusion-gemma` GGUF is hosted (the UI replaces the whole assistant message on each denoising step, then emits the final answer)
- Backward-compatible queue-status events (the engine itself handles concurrency)
- Message editing and deletion with regeneration from any point in the conversation
- Free scrolling: scroll up to read earlier replies while new tokens stream in; the chat auto-scrolls again as soon as the user scrolls back to the bottom

Use `--model` to choose the hosted GGUF file and `--mmproj` to choose the hosted projector. `TensorSharp.Server` no longer scans a `MODEL_DIR`.

**Server command-line options:**

Running `TensorSharp.Server` with no arguments prints the full parameter reference (description, default, and an example per option) and exits; `--help` does the same. Pass `--model` at startup for inference. Other options can start a model-less status process, but `/api/models/load` cannot select a GGUF that was not supplied at startup.

| Option | Description |
|---|---|
| `--model <path>` | GGUF file to host (required for inference; when other options are passed without it, the server starts but `/api/models/load` will report no hosted model) |
| `--mmproj <path>` | Multimodal projector GGUF (resolved relative to the model directory when only a filename is given; pass `none` to disable). Requires `--model`. |
| `--backend <type>` | Default compute backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan` |
| `--tp <N>` | Tensor parallelism degree — split the hosted model across N local GPUs (default: `1`). Requires `--backend cuda`, `ggml_cuda`, or `ggml_vulkan`. Env: `TENSORSHARP_TP_DEGREE`. See [Tensor Parallelism & Distributed Inference](#tensor-parallelism--distributed-inference). |
| `--tp-node-id <N>` | This node's 0-based ID for multi-node (distributed) tensor parallelism. The server can only be node `0` (the driver that serves HTTP); start worker nodes with `TensorSharp.Cli`. Requires `--tp-peers`. Env: `TENSORSHARP_TP_NODE_ID`. |
| `--tp-peers <list>` | Comma-separated `host:port` list of all nodes in the distributed TP cluster, ordered by node ID (e.g. `192.168.1.10:9500,192.168.1.11:9500`). Requires `--tp-node-id`. Env: `TENSORSHARP_TP_PEERS`. |
| `--gpu-device <N>` | Vulkan device index for the `ggml_vulkan` backend on multi-GPU hosts (e.g. an integrated Intel GPU next to a discrete NVIDIA one). Defaults to device 0; use `--list-gpus` to see the indices. Also settable via the `TS_GGML_VULKAN_DEVICE` env var. |
| `--list-gpus` | List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit |
| `--help` | Print the parameter reference (also shown when the server is started with no arguments) and exit |
| `--max-tokens <N>` | Maximum tokens to generate: fills in when a request omits its own limit, and caps a request that asks for more. Applies to every endpoint (Web UI, `/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/responses`). Default: `20000`, which is a plain default and does not cap. Env: `MAX_TOKENS`. |
| `--video-frames <N>` | Default output frame count for Wan video generation when a Web UI or API request omits `frames`. The VAE snaps it to `4k+1`; `121` frames at 24 fps is about five seconds. Without this flag, the model default is `33` (`49` for Wan 2.2 TI2V). An explicit request `frames` value overrides this default. |
| `--fps <N>` | Default playback frame rate for Wan MP4 output when a Web UI or API request omits `fps`. Without this flag, the model default is `16` fps (`24` for Wan 2.2 TI2V). An explicit request `fps` value overrides this default; changing only FPS changes playback speed rather than the generated frame count. |
| `--temperature <f>` | Sampling temperature (`0` = greedy) |
| `--top-k <N>` | Top-K filtering (`0` = disabled) |
| `--top-p <f>` | Nucleus sampling threshold (`1.0` = disabled) |
| `--min-p <f>` | Min-p filtering (`0` = disabled) |
| `--repeat-penalty <f>` | Repetition penalty (`1.0` = none) |
| `--presence-penalty <f>` | Presence penalty (`0` = disabled) |
| `--frequency-penalty <f>` | Frequency penalty (`0` = disabled) |
| `--seed <N>` | Random seed (`-1` = non-deterministic) |
| `--stop <string>` | Stop sequence (can be repeated). Under `--sampling-precedence config` a per-request `stop`/`stop_sequences` list is merged with these; under `request` it replaces them. |
| `--sampling-precedence <config\|request>` | Who wins when a request also carries a sampling parameter you set above. `config` (default) keeps your value — clients such as VS Code Copilot Chat hardcode `temperature`/`top_p` into every request and would otherwise silently override your configuration; parameters you did **not** set still come from the request. `request` restores client-always-wins. Env: `TENSORSHARP_SAMPLING_PRECEDENCE`. |
| `--n-cpu-moe <N>` / `-ncmoe <N>` | Keep the routed MoE weights of the first N layers in system RAM and run their FFN on the CPU (see **Mixture-of-Experts CPU offload** above). `all` offloads every layer. Default: 0 on every architecture, DeepSeek V4 included; a model that does not fit is refused at load with the number of layers that would make it fit. Env: `TS_N_CPU_MOE`. |
| `--cpu-moe` / `-cmoe` | Shorthand for `--n-cpu-moe all`. Default: off. Env: `TS_CPU_MOE`. |
| `--cpu-moe-threads <N>` | Worker threads for the host-side expert matmul. Default: half the usable CPU parallelism (`hardware_concurrency` clamped by the affinity mask and the cgroup CPU quota) on hosts with more than 8 cores. The server needs the other half for Kestrel, the scheduler and the accelerator submission threads; sizing this near the quota collapses throughput rather than degrading (20.7 tok/s at 64 threads vs 8.2 at 71 on a 95-CPU quota). Env: `TS_CPU_MOE_THREADS`. |
| `--kv-cache-dtype <type>` | KV cache precision for the hosted model: `f32`, `f16`, `q8_0`, or `q4_0` (quantized caches trade small numerical drift for memory; see the CLI table above for the tier trade-offs). Default: auto — the backend/model pick. Env: `KV_CACHE_DTYPE`. |
| `--continuous-batching` / `--no-continuous-batching` | Enable (default) or disable iteration-level paged-batching. When enabled the server admits / preempts sequences mid-batch and packs them into one forward pass on models that implement `IBatchedPagedModel`. `--no-continuous-batching` falls back to per-sequence KV-swap for every model. Alias: `--paged-batching` / `--no-paged-batching`. |
| `--prefill-chunk-size <N>` | Chunked-prefill granularity under contention — the maximum prefill tokens scheduled per step while other requests are running, so parallel decodes get frequent turns at the GPU (default: `1024`). Env: `TS_SCHED_PREFILL_CHUNK`. |
| `--mtp-spec` / `--no-mtp-spec` | Enable NextN/MTP speculative decoding (default off) on models that ship a multi-token-prediction draft head (Qwen 3.6's embedded NextN block, or a Gemma 4 `gemma4-assistant` draft loaded via `--mtp-draft-model`). Engages for solo (non-concurrent) sequences: the draft head proposes up to `--mtp-draft` tokens per step and the trunk verifies them in one batched forward, with the request's own sampler (penalties included) driving both drafting and verification, so output matches standard decode. Engaged automatically only where profitable (ggml backends and the pure-C# `cuda` backend); CPU / MLX serve standard decode. Env: `TS_MTP_SPEC`. |
| `--mtp-draft <N>` | Maximum tokens drafted per speculative step (default `8`). Env: `TS_MTP_DRAFT`. |
| `--mtp-pmin <f>` | Minimum draft confidence in `(0, 1]` for a drafted token to be kept; drafting stops at the first low-confidence token. Default: chosen per drafter kind — `0.75` for a per-token draft head (top-1 probability over its top-10 logits), `0.35` for a block drafter, where the gate is the CUMULATIVE prefix probability and so the same number means something far stricter. Env: `TS_MTP_PMIN`. |
| `--draft-model <path>` | Speculative-decoding draft model for architectures whose drafter ships as its own file. DeepSeek V4's DSpark support GGUF (see [DeepSeek V4](docs/models/deepseek4.md#dspark-speculative-decoding)): it drafts a whole block per step and the trunk verifies it in one batched forward, so greedy output is unchanged. Engages on every single-sequence CLI path — `--input`, `--multi-turn-jsonl` and `--interactive` — with `--backend cuda` or `--backend ggml_cuda`. On the CLI it needs a pure-argmax sampler (any temperature, top-k/p or repetition penalty turns it off, because the standalone decoder verifies with argmax). `TensorSharp.Server` accepts the same flag alongside `--mtp-spec`, where verification runs the request's own sampler and so composes with any sampling settings. Env: `TS_DSV4_DSPARK`. |
| `--spec-draft-n-max <N>` | Cap on tokens drafted per speculative block (default: the drafter's trained block size). |
| `--spec-draft-conf-min <p>` | Minimum cumulative acceptance probability (the product of the confidence head's per-position estimates) for a drafted position to be kept (default `0.35`). |
| `--mtp-draft-model <path>` | Path to a separate MTP draft GGUF for architectures whose draft head ships as its own file (Gemma 4's `gemma4-assistant`). The draft's hidden size must match the target (e.g. pair the 12B target with its 12B draft, not the 26B-A4B draft); a mismatched or incomplete draft fails fast at startup with a remediation hint. Ignored for Qwen 3.6, which embeds its NextN block in the trunk GGUF. Env: `TS_MTP_DRAFT_MODEL`. |
| `--paged-kv` / `--no-paged-kv` | Legacy compatibility flags for the removed per-session paged-KV manager. Current server KV state is engine-owned; use continuous-batching / `TS_SCHED_*` knobs for the engine. Aliases: `--paged-kv-cache` / `--no-paged-kv-cache`. |
| `--paged-kv-block-size <N>` | Legacy standalone paged-KV block size. The current server engine uses `TS_SCHED_BLOCK_SIZE`. |
| `--paged-kv-ram-mb <N>` | Legacy standalone paged-KV RAM-tier cap. |
| `--paged-kv-ssd-dir <dir>` | Legacy standalone paged-KV SSD cold-tier directory. |
| `--paged-kv-ssd-mb <N>` | Legacy standalone paged-KV SSD cap. |
| `--paged-kv-quant-bits <0\|4\|8>` | Legacy standalone paged-KV block quantization accepted by the server (`4`/`8` = symmetric). The runtime env var also accepts `2` for affine min+scale, and the CLI accepts `0\|2\|4\|8`. |
| `--redis-url <url>` | Redis connection string enabling both the shared KV cache tier and the Responses API store (e.g. `localhost:6379`). Sets both `TS_KV_CACHE_REDIS_URL` and `TS_RESPONSES_STORE_REDIS_URL`. |
| `--paged-kv-redis-url <url>` | Redis connection string for the shared KV cache tier only (e.g. `localhost:6379`). Env: `TS_KV_CACHE_REDIS_URL`. |
| `--paged-kv-redis-ttl <min>` | TTL in minutes for Redis KV cache entries; `0` = no TTL (default: `1440`, i.e. 24 hours). Env: `TS_KV_CACHE_REDIS_TTL_MINUTES`. |

Per-request fields in the chat / generate JSON payloads (e.g. `temperature`,
`top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`,
`frequency_penalty`, `seed`, `stop`/`stop_sequences`) fill in every parameter
you did **not** configure above. For a parameter you *did* configure, the
default `--sampling-precedence config` keeps your value and ignores the
request's — many chat clients hardcode `temperature`/`top_p` into every call
with no way for the end user to change them, so an unconfigured server-side
value is the only one a request can move. Pass `--sampling-precedence request`
for the inverse (clients always win), which is what versions before this flag
did unconditionally.

**Runtime environment variables:**

| Variable | Description |
|---|---|
| `BACKEND` | Default compute backend (`cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan`), used when `--backend` is not passed (default: `ggml_metal` on macOS, `ggml_cpu` elsewhere) |
| `MAX_TOKENS` | Maximum generation length when `--max-tokens` is not passed: fills in when a request omits its own limit and caps a request that asks for more (default: `20000`, which is a plain default and does not cap) |
| `VIDEO_SAMPLE_FPS` | Frames sampled per second from an **input video prompt** for multimodal understanding; time-based extraction (default: `1`). This is unrelated to the Wan output setting `--fps`. |
| `VIDEO_MAX_FRAMES` | Optional upper bound on frames extracted from an **input video prompt** (evenly down-sampled); unset/`0` means no cap (default: no cap). This is unrelated to Wan output `--video-frames`. |
| `PORT` / `HOST` | Listen port / bind interface when `--port` / `--host` are not passed (defaults: `5000`, `0.0.0.0`) |
| `ASPNETCORE_URLS` | Full listen URL(s) when none of `--port`, `--host`, `--urls`, `PORT`, or `HOST` is set |
| `TENSORSHARP_TEMPERATURE` | Sampling temperature when `--temperature` is not passed. Counts as operator-configured, so it also outranks the request body under the default `--sampling-precedence config` |
| `TENSORSHARP_TOP_K` | Top-K when `--top-k` is not passed (same precedence rule as `TENSORSHARP_TEMPERATURE`) |
| `TENSORSHARP_TOP_P` | Top-P when `--top-p` is not passed (same precedence rule) |
| `TENSORSHARP_MIN_P` | Min-P when `--min-p` is not passed (same precedence rule) |
| `TENSORSHARP_REPEAT_PENALTY` | Repetition penalty when `--repeat-penalty` is not passed (same precedence rule) |
| `TENSORSHARP_PRESENCE_PENALTY` | Presence penalty when `--presence-penalty` is not passed (same precedence rule) |
| `TENSORSHARP_FREQUENCY_PENALTY` | Frequency penalty when `--frequency-penalty` is not passed (same precedence rule) |
| `TENSORSHARP_SEED` | Random seed when `--seed` is not passed (same precedence rule) |
| `TENSORSHARP_SAMPLING_PRECEDENCE` | `config` (default) or `request`: whether server-configured sampling parameters outrank the ones a client sends. `--sampling-precedence` overrides it |
| `TENSORSHARP_LOG_LEVEL` | Minimum log level for both console and file loggers: `Trace`, `Debug`, `Information`, `Warning`, `Error`, `Critical` (default: `Information`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_DIR` | Directory the JSON-line file logger writes to (default: `<binDir>/logs`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_FILE` | Set to `0` to disable the file logger and keep only the console output (default: enabled). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_TP_DEGREE` | Tensor parallelism degree — number of local GPUs to split the model across (default: `1`). Fallback in `ModelBase.Create` when no `--tp` flag is passed; both `TensorSharp.Cli` and `TensorSharp.Server` expose it as `--tp <N>`. Requires `--backend cuda`, `ggml_cuda`, or `ggml_vulkan`. |
| `TENSORSHARP_TP_DEVICES` | GPU ordinals the TP ranks map to, comma-separated (e.g. `0,2`; default `0..tp-1`). Used by TP on the GGML backends. |
| `TENSORSHARP_TP_NODE_ID` | This node's 0-based ID for multi-node distributed tensor parallelism. Must be set together with `TENSORSHARP_TP_PEERS`. |
| `TENSORSHARP_TP_PEERS` | Comma-separated `host:port` list of all nodes in the distributed TP cluster (e.g. `192.168.1.10:9500,192.168.1.11:9500`). Must be set together with `TENSORSHARP_TP_NODE_ID`. |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | How long each node keeps retrying outbound connections to its peers before giving up (default: `120`). Raise it when nodes are started far apart by hand or by a slow orchestrator. |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | Per-receive timeout for a blocking read from a peer (default: `300`). A stalled peer fails the collective instead of hanging on the OS TCP keepalive (often 2+ hours). |
| `TENSORSHARP_TP_DISABLE_P2P` | Set to `1` to force every cross-GPU transfer through host staging instead of CUDA peer-to-peer DMA. Slower, but reproduces the code path taken by hardware without peer access (A16 vGPU profiles, some consumer cards) and isolates P2P-specific defects. |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | Set to `1` to run the local AllReduce through host memory (device→host, sum, host→device) instead of the device-to-device P2P path. Diagnostic fallback that mirrors the known-good multi-node reduce. |
| `TS_KV_CACHE_REDIS_URL` | Redis connection string for the shared KV cache tier (e.g. `localhost:6379`). When set, KV cache blocks are persisted to Redis for cross-session reuse. CLI: `--redis-url` or `--paged-kv-redis-url`. |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | TTL in minutes for Redis KV cache entries; `0` = no TTL (default: `1440`). CLI: `--paged-kv-redis-ttl`. |
| `TS_RESPONSES_STORE_REDIS_URL` | Redis connection string for the OpenAI Responses API store. When set, `RedisResponsesStore` replaces the in-memory store. CLI: `--redis-url`. |
| `DIFFUSION_STEPS` | Server-side DiffusionGemma denoising steps per block (default: `48`; CLI equivalent is `--diffusion-steps`) |
| `DIFFUSION_MAX_BATCH` | Maximum concurrent DiffusionGemma requests batched by the Web UI diffusion scheduler (default: `2`) |

**Paged KV cache & continuous-batching tunables (read at process / model start)**

These can be set with either the `--paged-kv*` / `--continuous-batching` CLI flags (which translate to the env vars below) or directly via the environment:

| Variable | Description |
|---|---|
| `TS_KV_PAGED_CACHE` | Legacy compatibility switch for the standalone `PagedKvCacheManager`; current `TensorSharp.Server` request KV state is engine-owned. The CLI shortcuts are `--paged-kv` / `--no-paged-kv`. |
| `TS_KV_BLOCK_SIZE` | Legacy standalone paged-KV block size. The engine uses `TS_SCHED_BLOCK_SIZE`. |
| `TS_KV_CACHE_MAX_RAM_MB` | Legacy standalone paged-KV RAM-tier cap. |
| `TS_KV_CACHE_SSD_DIR` | Legacy standalone paged-KV SSD cold-tier directory. |
| `TS_KV_CACHE_MAX_SSD_MB` | Legacy standalone paged-KV SSD cap. |
| `TS_KV_PAGED_QUANT_BITS` | Legacy standalone paged-KV block quantization bits (`0` = passthrough, `2` = affine, `4`, or `8`). |
| `TS_SCHED_DISABLE_BATCHED` | `1` forces the per-sequence KV-swap fallback even when a model implements `IBatchedPagedModel`. The CLI shortcut is `--no-continuous-batching`. |
| `TS_SCHED_MAX_BATCHED_TOKENS` | Scheduler per-step token budget (default: `4096`). |
| `TS_SCHED_MAX_RUNNING_SEQS` | Maximum in-flight sequences (default: `16`). |
| `TS_SCHED_PREFILL_CHUNK` | Maximum prefill tokens per step when requests contend (default: `1024`). |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | Prefill chunk size for the fresh (start_pos = 0) part of a SOLO prompt — one uncontended request gets big fused-prefill chunks (default: `8192`). |
| `TS_SCHED_NUM_BLOCKS` | Physical blocks in the engine block pool (default: `256`). |
| `TS_SCHED_BLOCK_SIZE` | Tokens per block on the engine side (default: `256`). |
| `TS_SCHED_PREFIX_CACHE` | `0` disables block-hash prefix sharing across requests. |
| `TS_SCHED_DECODE_QUANTUM` | Tokens before a sequence-switch is allowed (default: block size). |
| `TS_QWEN35_BATCHED` | Set to `0` to force the Qwen 3.5/3.6 family onto the legacy per-sequence KV-swap path (default: batched/paged). Also implicitly disabled by `--no-continuous-batching`. |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Use the native batched GatedDeltaNet kernel inside Qwen 3.5/3.6 batched path. |
| `TS_GEMMA4_BATCHED` | Set to `0` to force Gemma 4 onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_GPTOSS_BATCHED` | Set to `0` to force GPT OSS onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_GPTOSS_PAGED_ATTN_MANAGED` | Use the managed (C#) paged-attention-with-sinks kernel inside GPT OSS batched path. |
| `TS_NEMOTRON_BATCHED` | Set to `0` to force Nemotron-H onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Use the native Mamba2 batched step kernel inside Nemotron-H batched path. |
| `TS_PAGED_ATTN_KERNEL` | Paged-attention dispatch kernel for `Mistral3Model.BatchedForward`: `native` (default), `tensor` (C# Tensor-based), or `managed` (pure C# scalar). |
| `TS_MLX_PIPELINED_DECODE` | `1` (default) enables pipelined greedy decode on the MLX backend when the request is greedy, has no stop sequences, and the model supports device-side argmax / next-embedding lookup. Set to `0` to disable. CLI only. |
| `TS_MLX_MLOCK_GGUF` | `1` (default) pins the GGUF mmap region in physical RAM via `mlock(2)` so model weights stay resident between forward passes. Set to `0` to skip (use if the process `memlock` rlimit is too low or you want the OS to manage paging). MLX backend only. |
| `TS_MLX_FUSED_KV_WRITE` | `1` (default) uses a single multi-dim `slice_update` to write the per-token KV block. Set to `0` to revert to the per-head loop (A/B testing / regression isolation). |
| `TS_MLX_BATCHED_MOE_DECODE` | `1` (default) collapses K per-expert decode dispatches to one batched dispatch per (gate/up/down) kind for Qwen 3.5/3.6 MoE. Set to `0` on memory-constrained machines (saves ~weight-doubling overhead from the stacked weight slabs). |
| `TS_MLX_MOE_FUSED_GATE_UP_SILU` | `1` (default) fuses gate matmul + up matmul + SiLUMul into one Metal kernel for batched MoE decode. Set to `0` to A/B against the legacy 3-dispatch path. |
| `TS_MLX_DEVICE_ROUTER` | `1` (default) keeps MoE router top-K + softmax on device to skip ~60 host syncs/token on Qwen 3.6-35B-A3B. Set to `0` to disable; the code also falls back automatically when prerequisites are missing. |
| `TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB` | Override the MLX allocator hard cap / unused-buffer cache cap / wired-buffer residency cap (megabytes). Defaults are derived from the host's unified-memory capacity. |
| `TS_MLX_EVAL_EVERY_N_LAYERS` / `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` | Periodic `mlx_async_eval` cadence during decode to overlap GPU work with host queueing. Gemma 4 defaults to every 4 layers via `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS`; Qwen 3 / Qwen 3.5 / Nemotron-H default to every 16 layers via `TS_MLX_EVAL_EVERY_N_LAYERS`. Set to `0` to disable where supported. |
| `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` | Override the search path for `libmlxc` when using `--backend mlx`. |

**MTP / speculative-decoding tunables**

These gate the optional multi-token-prediction speculative decode path (see [MTP / NextN speculative decoding](FEATURES.md#mtp--nextn-speculative-decoding)). `TS_MTP_*` are the shared knobs (also set by the `--mtp-*` CLI flags); `TS_GMTP_*` are Gemma 4 draft-path A/B switches.

| Variable | Description |
|---|---|
| `TS_MTP_SPEC` | `1` enables MTP/NextN speculative decoding for solo sequences (default `0`). CLI: `--mtp-spec` / `--no-mtp-spec`. |
| `TS_MTP_DRAFT` | Maximum tokens drafted per speculative step (default `8`). CLI: `--mtp-draft`. |
| `TS_MTP_PMIN` | Minimum draft confidence in `(0, 1]` to keep a drafted token (default: per drafter kind — `0.75` per-token, `0.35` block). CLI: `--mtp-pmin`. |
| `TS_MTP_DRAFT_MODEL` | Path to the separate Gemma 4 `gemma4-assistant` draft GGUF. CLI: `--mtp-draft-model`. Ignored by Qwen 3.6 (embedded NextN). |
| `TS_GMTP_NO_FUSED` | `1` disables the Gemma 4 fused multi-token-verify / draft-step GGML kernels and falls back to the per-op path (A/B testing on ggml backends). |
| `TS_GMTP_NO_FAST_ROLLBACK` | `1` restores the kept-prefix rollback path instead of the dense exact-match fast rollback used on partial draft acceptance. |
| `TS_GMTP_BATCHED_TRUNK` | `1` opts the Gemma 4 verify trunk back into the batched paged path; the default runs the faster linear trunk for solo speculation. |

**DiffusionGemma-specific tunables**

| Variable | Description |
|---|---|
| `DIFFUSION_NO_SC` | Set to `1` to disable self-conditioning. Enabled by default. |
| `DIFFUSION_SC_TOPK` | Experimental self-conditioning top-K cutoff (default: `32`). |
| `DIFFUSION_NO_PKV` | Set to `1` to disable prompt-KV caching on device-glue backends. Enabled by default where supported. |
| `DIFFUSION_NO_FUSED_DECODE` | Set to `1` to disable the GGML fused model decode path and fall back to per-op / per-layer diffusion decode. |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | Set to `1` to disable the fused output-norm + lm-head + softcap tail. |
| `DIFFUSION_BATCHED_FORWARD` | Set to `1` to use true batched `DecodeCanvasBatched` for active diffusion canvases; default time-slices the faster fused single-canvas path. |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | Memory cap for batched diffusion lm-head logits before falling back to per-sequence lm-head (default: `300`). |

Sampling parameter precedence (highest wins), with the default
`--sampling-precedence config`:

1. Server-wide CLI flags / config-file keys (e.g. `--temperature`, `--top-p`, `--stop`).
2. `TENSORSHARP_*` environment variables listed above.
3. Per-request JSON fields in the API call (e.g. `temperature`, `top_p`, `stop`)
   — for every parameter that steps 1 and 2 did not set.
4. Built-in `SamplingConfig` defaults (`temperature=0.8`, `top_k=40`, `top_p=0.9`, `min_p=0`, `repeat_penalty=1.1`, `repeat_last_n=64`, presence/frequency penalties `0`, `seed=-1`, no stop sequences).

With `--sampling-precedence request`, steps 1–3 swap: the request wins over the
server-wide flags and env vars for parameters it sends, and they still fill in
the rest. Either way `--stop` sequences pinned on the server stay in force under
`config` (merged with the request's) and are replaced by the request under
`request`.

## Mixture-of-Experts CPU offload (`--n-cpu-moe`)

Large MoE models spend almost all of their bytes on routed experts while
activating only a small fraction of them per token — Qwen3.6-35B-A3B is ~11 GB
of weights for ~3B active parameters. `--n-cpu-moe N` keeps the routed experts
of the first N layers in system RAM, leaving attention, the norms, the router
and the always-active shared expert on the accelerator. `--n-cpu-moe` therefore
means "these experts do not live in VRAM" — not "these experts are multiplied on
the CPU". Where they are multiplied depends on the batch:

* **Decode (one token, or any batch under `TS_HOST_MOE_DEVICE_MIN_BATCH`, default
  128).** The host multiplies them, reading the quantized blocks zero-copy out of
  the GGUF mmap. One token only touches `n_expert_used` experts, so this is a few
  MB of RAM traffic per layer against the hundreds of MB an upload would cost.
* **Prefill (a real batch).** The experts are *streamed* to the accelerator for
  that one graph and multiplied there, then the staging memory is reused by the
  next layer — they are never made resident. A 512-token batch amortizes the
  upload over every token in it, and the arithmetic is what the GPU is for.
  llama.cpp's scheduler makes the same call (`ggml_backend_sched` sends an op
  whose weights live in a host buffer back to the GPU above
  `op_offload_min_batch_size`).

Three things make the streamed side fast, and they are why TensorSharp's
offloaded prefill runs several times faster than llama.cpp's on the same files:

1. **The host pages are locked.** A copy out of a pageable mmap cannot DMA — the
   driver stages it through a small pinned buffer. `cudaHostRegister` on the
   expert range costs ~65 ms/GiB once and takes the transfer from 9.3 GB/s to
   55.6 GB/s on PCIe 5.0 x16. Disable with `TS_HOST_MOE_PIN=0`; bound it with
   `TS_HOST_MOE_PIN_MAX_MB` (default: 60% of the cgroup/host memory limit).
2. **Only the experts this batch routes to are sent**, grouped into consecutive
   runs — the same trick llama.cpp's scheduler plays with its used-expert bitset.
   At 512 tokens a large expert pool is only partly covered, and at the small
   batches speculative verification and light serving produce it is a large
   saving. `TS_HOST_MOE_EXPERT_FILTER=0` restores whole-stack uploads.
3. **The copies are issued asynchronously** on the backend's own stream and
   synchronize once, with the graph, instead of three times per layer.

The offloaded layers stay inside the fused whole-model graph: the accelerator
pauses after each offloaded layer's router, the experts are multiplied (on the
host, or on the accelerator from streamed weights), and the result is handed back
before the next segment runs. Everything else in the token — attention, the
shared or dense FFN, the LM head — stays in one graph submission. This holds for
Qwen3.5/3.6, Gemma 4 MoE, GPT-OSS and DiffusionGemma; Gemma 4 MoE segments its
prefill graph the same way, and DiffusionGemma's block decode hands the host all
of the canvas positions at once so its offloaded side is a GEMM rather than a
matvec. Qwen3.5/3.6 segments its prefill graph the same way, and both it and
Gemma 4 MoE keep the fused graph under tensor parallelism too (see the `--tp`
note below). GPT-OSS now has a fused whole-model prefill graph as well, so its
offloaded prefill is segmented rather than dispatched per layer. Architectures
that still lack one (Nemotron-H) reach the streamed path through their per-layer
MoE op, on a single device; under `--tp N` they stay host-side so that N ranks do
not each stream their own copy of the same unsharded experts.

Measured on 2 x Xeon 6952P + RTX PRO 6000 Blackwell (PCIe 5.0 x16), **pp8192 /
tg128**, peak per-process VRAM. The full comparison against llama.cpp — two prompt
lengths, five offload depths per model, including DeepSeek V4 Flash across two
GPUs — is in
[docs/moe_cpu_offload_benchmark.md](docs/moe_cpu_offload_benchmark.md):

| Model | Setting | Peak VRAM | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (UD-IQ4_XS) | default | 16.4 GiB | 11,274 tok/s | 161 tok/s |
| | `--n-cpu-moe 8` | 15.4 GiB | 6,500 tok/s | 80 tok/s |
| | `--n-cpu-moe 16` | 13.8 GiB | 4,888 tok/s | 55 tok/s |
| | `--cpu-moe` (30 layers) | 10.8 GiB | 3,072 tok/s | 40 tok/s |
| Qwen3.5-35B-A3B (UD-IQ4_XS) | default | 19.4 GiB | 9,405 tok/s | 160 tok/s |
| | `--n-cpu-moe 24` | 15.1 GiB | 5,259 tok/s | 52 tok/s |
| | `--cpu-moe` (48 layers) | 11.3 GiB | 3,709 tok/s | 39 tok/s |
| gpt-oss-20b (Q8_0/MXFP4) | default | 12.9 GiB | 12,925 tok/s | 213 tok/s |
| | `--n-cpu-moe 12` | 9.2 GiB | 6,394 tok/s | 52 tok/s |
| | `--cpu-moe` (24 layers) | 4.7 GiB | 3,798 tok/s | 28 tok/s |
| DeepSeek-V4-Flash (UD-Q8_K_XL, 2 GPUs) | default | 165.2 GiB | 4,387 tok/s | 51 tok/s |
| | `--n-cpu-moe 12` | 128.7 GiB | 428 tok/s | 10 tok/s |
| | `--n-cpu-moe 24` | 77.9 GiB | 236 tok/s | 5.3 tok/s |

At every offload depth that is **4.5-10.9x** llama.cpp's prefill and **2.5-4.5x**
its decode on the three seam models. Peak VRAM is higher than llama.cpp's (1.1x
resident, up to 3.5x fully offloaded) — see the report's VRAM note.

Greedy output is **token-identical** at every offload depth on gemma-4-26B-A4B.

A small-machine picture from the same feature on a laptop (RTX 3080 Laptop
16 GB / i7-11800H, 96-token generation) — note how far under the card's 16 GB
the offloaded configurations sit, and that both the GPT-OSS and DiffusionGemma
baselines were over the WDDM spill threshold, which is why offload makes them
*faster* as well as smaller there:

| Model | Setting | Peak VRAM | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (Q4_K_XL) | default | 16.1 GB | 190 tok/s | 39.7 tok/s |
| | `--n-cpu-moe 8` | 13.1 GB | 52 tok/s | 38.6 tok/s |
| | `--n-cpu-moe 16` | 10.2 GB | 24 tok/s | 21.6 tok/s |
| | `--cpu-moe` (30 layers) | 4.8 GB | 15 tok/s | 17.7 tok/s |
| gpt-oss-20b (Q8_0) | default | 16.2 GB | 2.7 tok/s | 0.3 tok/s |
| | `--n-cpu-moe 12` | 14.0 GB | 58 tok/s | 25.4 tok/s |
| | `--cpu-moe` (24 layers) | 2.9 GB | 29 tok/s | 12.1 tok/s |
| diffusiongemma-26B-A4B (Q4_K_M) | default | 16.0 GB | — | 1709 ms/step |
| | `--n-cpu-moe 16` | 9.8 GB | — | 2287 ms/step |
| | `--cpu-moe` (30 layers) | 3.0 GB | — | 4697 ms/step |

(Those laptop prefill numbers predate the streamed-expert path; the same
configurations are several times faster now.)

Notes:

* **Pick the smallest N that fits.** Each offloaded layer costs decode
  throughput, so offload only as many layers as you need to free the VRAM the
  KV cache wants.
* **Decode pays the most, proportionally.** Prompt processing streams the
  experts to the accelerator and stays within a small factor of the resident
  run; decode is bounded by how fast the host can read `n_expert_used` experts
  per layer out of RAM, which is where the throughput goes.
* **The worker count is capped at 64.** The decode-side matmul is one token
  wide — memory-bound work with a barrier after every op — so past a few dozen
  threads each extra worker only adds a barrier participant, and on a 2-socket
  host it adds cross-socket traffic too. Measured on 2x Xeon 6952P
  (192 cores / 384 threads), gemma-4-26B `--cpu-moe`, ms of host matmul per 30
  offloaded layers: 8 threads 45, 16 threads 35, 32 threads 17.6, 64 threads
  17.3, 96 threads 26, 192 threads 120. `--cpu-moe-threads N` overrides.
* **Routing stays on the accelerator**, so expert selection and weights are the
  same ones the fully resident path would pick.
* **The fused whole-model graph is kept.** Qwen3.5/3.6, Gemma 4 MoE, GPT-OSS and
  DiffusionGemma each run a whole token (for DiffusionGemma, a whole canvas
  block) as ONE accelerator graph, and offload does not give that up. Each
  offloaded layer's expert matmuls are cut out of the graph; everything else —
  attention, norms, the router, the shared/dense FFN and the LM head — keeps
  running fused, with the accelerator pausing only to hand the host that layer's
  routed-expert work. Gemma 4 MoE segments its *prefill* graph the same way, so a
  prompt chunk is still one dispatch with the host doing a real GEMM over every
  token in it.
* Nemotron-H has no whole-model fused graph to preserve — its hybrid
  Mamba2/attention/MoE decode is dispatched per layer regardless — so offload
  there simply routes each layer's MoE through the host path.
* **Combines with tensor parallelism.** Under `--tp N` the offloaded layers'
  seams are merged into the same segment schedule the ranks already use for
  their AllReduce points, so the fused multi-rank graph is kept — Qwen3.5/3.6
  and Gemma 4 MoE run both decode and prefill fused. Each offloaded layer is
  evaluated ONCE, on the host, over the unsharded expert stack: the host expert
  backend is a single thread pool, so splitting that matmul per rank would only
  serialize on it while paying an extra download each. The single result is then
  placed so the graph's own reduction lands on the single-GPU value (rank 0 only
  where the layer output feeds a later AllReduce, every rank where it does not).
  Measured on 2x L4 24 GB (short prompt, prefill 512 / decode 64):

  | Model | Setting | Resident weights (GPU0 + GPU1) | Prefill | Decode |
  | --- | --- | --- | --- | --- |
  | Qwen3.5-35B-A3B (UD-IQ4_XS) | `--tp 2` | 9397 + 8032 MB | 1942 tok/s | 76.5 tok/s |
  | | `--tp 2 --cpu-moe` | 2277 + 912 MB | 66 tok/s | 17.8 tok/s |
  | gemma-4-26B-A4B (UD-IQ4_XS) | `--tp 2` | 6835 + 6087 MB | 3062 tok/s | 59.7 tok/s |
  | | `--tp 2 --n-cpu-moe 8` | 5459 + 4711 MB | 217 tok/s | 36.5 tok/s |
  | | `--tp 2 --cpu-moe` | 1588 + 840 MB | 60 tok/s | 13.2 tok/s |

  Greedy output on Gemma 4 is byte-identical to the same `--tp N` run without
  offload on most prompts and, on the rest, agrees for hundreds of characters
  before taking an equivalent alternative ending; Qwen3.5 likewise tracks the
  resident run to a paraphrase point. Both are deterministic — the gap is the
  host expert matmul's activation quantization, the same ~1-5% relative
  difference `TS_HOST_MOE_VERIFY=1` reports on a single GPU. The pure-C# `cuda`
  backend has no host-MoE seam, so `--tp N --cpu-moe` there prints the
  `[moe-offload] WARNING` and keeps the experts resident.
* `TS_HOST_MOE_VERIFY=1` builds the on-GPU expert chain alongside the host one
  and reports their per-layer divergence — a diagnostic for validating the seam
  on a model that also fits in VRAM. `TS_HOST_MOE_DEBUG=1` prints the segment
  plan (the node cuts) and each seam's activation norms. `TS_HOST_MOE_TIMING=1`
  reports the offloaded side's wall clock split into per-call setup and host
  matmul, which is what tells you whether a slow run is the expert GEMM or the
  scaffolding around it.

## Tensor Parallelism & Distributed Inference

TensorSharp supports **tensor parallelism (TP)** — splitting a single model across
multiple GPUs using the Megatron-LM column/row-parallel pattern — and
**distributed (multi-node) tensor parallelism**, where TP spans multiple
machines connected over a TCP peer-to-peer network.

### Local tensor parallelism (single process, multiple GPUs)

Split the model across N GPUs within one process (direct CUDA, or the GGML CUDA / Vulkan backends). Each GPU holds `1/N` of
the sharded weights (column-parallel QKV/gate/up, row-parallel output/down) plus
a full copy of replicated weights (norms, embeddings, LM head). Per-GPU KV
caches are independent; AllReduce (via CUDA P2P copies + elementwise-add kernel)
reconverges the hidden state after each row-parallel projection.

```bash
# CLI: 2-GPU tensor parallelism
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2

# Server: same flag (TENSORSHARP_TP_DEGREE=2 env var also works)
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
    --model <model.gguf> --backend cuda --tp 2

# Config JSON
{ "tp": 2, "backend": "cuda", "model": "<model.gguf>" }
```

### Distributed tensor parallelism (multi-node, peer-to-peer TCP)

Distribute TP across multiple machines. Each node runs its own process with its
own local GPUs; nodes communicate over a TCP mesh using a length-prefixed
framing protocol. AllReduce is hierarchical: local P2P within each node, then
TCP across node representatives, then broadcast back — minimizing network
traffic to `1/tp_local` of the data per collective.

```bash
# 2 nodes × 2 GPUs each (4 GPUs total)
# Node 0:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Node 1:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Server as the cluster front-end: the server must be node 0 (the driver that
# owns sampling and serves HTTP); every other node runs a TensorSharp.Cli worker
# with the same model, backend, and peer list. The TENSORSHARP_TP_* env vars
# work as well.
# Node 0 (server / driver):
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Node 1 (CLI worker):
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Config JSON (per node)
{ "tp": 2, "tp-node-id": 0, "tp-peers": "192.168.1.10:9500,192.168.1.11:9500", "backend": "cuda" }
```

Every node must use the same `--tp-peers` list (or `TENSORSHARP_TP_PEERS` env
var) and a unique `--tp-node-id` (or `TENSORSHARP_TP_NODE_ID`). The port
(`9500` in the examples) is not a default — it must be specified explicitly and
must be reachable between all nodes.

### Supported architectures

| Architecture | TP status | Notes |
|---|---|---|
| Qwen 3 | ✅ | Reference implementation |
| Mistral 3 | ✅ | Fused/separate QKV, YaRN RoPE |
| Gemma 3 | ✅ | Separate Q/K/V, GELU, sliding window |
| Gemma 4 | ✅ | Dense TP + MoE. On GGML the fused whole-model MoE trunk splits *inside* each expert (gate/up column-parallel, down row-parallel) so global expert ids keep working; `TS_GEMMA4_TP_FUSED_MOE=0` falls back to the whole-expert per-op path. Per-expert slicing on direct CUDA |
| Qwen 3.5 / 3.6 family | ✅ | GatedDeltaNet SSM with per-rank V-head ownership; expert-parallel MoE on GGML (whole experts per rank, Megatron-split shared expert), expert slicing on direct CUDA. Runs on both `cuda` and `ggml_cuda` / `ggml_vulkan` — the GGML path uses the packed per-rank GDN kernel (`TSGgml_Qwen35GdnLayerTP`) with device-resident recurrent state |
| GPT OSS | ✅ | Attention sinks, YaRN. Runs on `cuda` and the GGML backends; the GGML path is expert-parallel (whole experts per rank, one batched `ggml_mul_mat_id` dispatch per projection per layer) and falls back to per-expert slicing only when the expert count does not divide the TP degree |
| Nemotron-H | ✅ | Mamba2 replicated on rank 0, MoE expert slicing. Still walks experts per token per rank on GGML (no expert parallelism yet) |
| DiffusionGemma | — | Not applicable (diffusion model) |
| Qwen-Image-Edit | — | Not applicable (image generation) |

### Backend support

TP runs on the **direct CUDA** backend (`--backend cuda`) and on the **GGML
CUDA / Vulkan** backends (`--backend ggml_cuda`, `ggml_vulkan`). MLX is
single-device.

On the GGML backends each rank gets its own ggml backend on its own GPU, with
its own device-resident weight shards and KV cache. Cross-GPU AllReduce uses
ggml-cuda's collective (NCCL when the build finds it, its P2P pipeline
otherwise); small payloads are reduced in host memory instead, which is cheaper
because GGML activations already live there.

```bash
# 2 GPUs on the GGML CUDA backend
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_cuda --tp 2

# Choose which physical GPUs the ranks map to
TENSORSHARP_TP_DEVICES=0,2 dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model <model.gguf> --backend ggml_cuda --tp 2
```

**What to expect.** GGML TP started as a *capacity* feature — it lets a model
that does not fit in one GPU's VRAM run entirely on GPUs — and fused per-rank
execution (Stage 1c) made it a latency win as well. Each TP block now runs as a
per-rank fused native graph (attention, dense FFN, MoE trunk, GatedDeltaNet)
instead of op-at-a-time, so a rank issues a handful of graph launches per layer
rather than hundreds of native calls.

Measured on 2× RTX 2000 Ada (16 GB each, PCIe, no NVLink), prefill 512 /
decode 64, tok/s:

| Model | 1 GPU | `--tp 2` |
|---|---|---|
| Gemma 4 E4B Q8_0 | 2760 / 37.3 | 2488 / **51.7** |
| Gemma 4 26B-A4B IQ4_XS | 1845 / 48.5 | 2537 / **51.2** |
| Qwen 3.5-9B Q8_0 | 1461 / 23.1 | 399 / **24.4** |
| Qwen 3.5-35B-A3B IQ4_XS | does not fit | **184 / 18.1** |

Decode — the memory-bound half TP should help — is 1.39× a single GPU on
Gemma 4 E4B and 1.06× on Qwen 3.5-9B; both Gemma 4 models produce output
byte-identical to their single-GPU runs. Prefill is compute-bound and pays the
collectives, so it lands at or below the single-GPU figure on models that fit on
one card. Qwen 3.5-35B does not fit a 16 GB card at all, so TP is the only way
to run it. See `TENSOR_PARALLELISM_PLAN.md` (Stages 1b and 1c) for the full
measurements and what is left to fuse.

TP composes with MoE CPU offload: `--tp N --n-cpu-moe M` keeps the fused
multi-rank graph and drops the offloaded layers' expert bytes from every rank's
VRAM. See [Mixture-of-Experts CPU offload](#mixture-of-experts-cpu-offload---n-cpu-moe)
for the combined numbers.

| Variable | Effect |
|---|---|
| `TENSORSHARP_TP_DEVICES` | GPU ordinals per rank, e.g. `0,2` (default `0..tp-1`) |
| `TS_GGML_TP_PARALLEL=0` | Drive ranks sequentially instead of concurrently (diagnostic) |
| `TS_GGML_TP_FUSED_MATMUL=1` | Submit both ranks' linears from one thread (off by default; it allocates a device buffer per rank per call, measured 2.3× slower on Qwen 3.5 35B) |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | Element count above which AllReduce uses the device collective (default 262144) |
| `TS_GGML_F32_RESIDENT=0` | Bind F32 linear weights per call instead of keeping them device-resident (diagnostic) |
| `TS_GEMMA4_TP_FUSED_MOE=0` | Gemma 4 only: fall back from the fused whole-model MoE trunk (Megatron split inside each expert) to the whole-expert per-op path. The fused path materializes ~10.5 GB of expert slices at load on the 26B (~36 s) in exchange for a ~10× decode. Layers offloaded by `--n-cpu-moe` are skipped by that materialization — they never run on the accelerator — so `--cpu-moe` also removes the load-time cost |
| `GGML_CUDA_AR_BF16_THRESHOLD` | Payload size above which ggml-cuda's collective converts F32 to BF16 before reducing. TensorSharp raises ggml's default (1 byte — i.e. always) to 1 MB so decode-sized collectives reduce exactly; `0` disables the conversion entirely |
| `TS_QWEN35_LAYER_TRACE=1` | Print a per-layer residual-stream summary for the first forward, from both the single-GPU and TP loops (diagnostic) |
| `GGML_CUDA_ALLREDUCE` | `nccl` / `internal` / `none`, passed through to ggml |

### Constraints

- `numHeads`, `numKVHeads`, and `intermediateSize` must be divisible by the TP degree.
- Quantized row-parallel splits require `ne0` divisible by `tp × blockSize`.
- Batched/continuous-batching forward under TP is implemented for Qwen 3 and Mistral 3; MoE models (Gemma 4, Qwen 3.5/3.6, GPT OSS, Nemotron-H) fall back to per-sequence forward under TP.
- **Muse-Glimmer** caps at `--tp 2`: it has 2 KV heads, and no model here replicates KV heads when `numKVHeads < tp`. Its DFlash drafter and pooled KV-block snapshots stay single-GPU under TP (multi-turn reuse comes from live-cache continuation instead), and it requires the GGML CUDA/Vulkan backends — the fused per-rank plan needs a device collective that ggml-metal does not provide.

### Cluster tuning & diagnostics

Local AllReduce prefers CUDA peer-to-peer DMA. At startup the group enables peer
access for every device pair that reports it, then runs a round-trip self-test:
some topologies (L4 behind certain PCIe switches, IOMMU-enabled hosts, small
BAR1 windows) advertise peer access but silently transfer corrupt data, and any
pair that fails the test is demoted to host staging permanently. Hardware
without peer access at all (A16 vGPU profiles, most consumer cards) simply
stages through host memory from the start.

| Variable | Default | What it does |
|---|---|---|
| `TENSORSHARP_TP_DISABLE_P2P=1` | off | Force every cross-GPU transfer through host memory, exactly as no-peer hardware does. Use it to check whether a multi-GPU defect lives in the P2P DMA path. |
| `TENSORSHARP_TP_HOST_ALLREDUCE=1` | off | Run the local AllReduce as device→host, sum on the CPU, host→device. Slower, but matches the multi-node reduce exactly — useful for isolating P2P correctness issues. |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | `120` | How long a node retries outbound connections to its peers. Nodes are usually launched by hand seconds or minutes apart, so a peer's listener may not be up yet; raise this for slow orchestrators. |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | `300` | Per-receive timeout on a peer socket. Without it a stalled peer would block on the OS TCP keepalive (often 2+ hours) instead of failing the collective. |

Startup logs make the topology explicit: the local group prints
`Tensor parallelism: N GPUs (<device names>)`, P2P demotions print a
`TP: P2P disabled…` / self-test warning, and every distributed node prints
`[TcpCommunicator] Rank r/N connected to all peers.` once the mesh is complete.

### Redis-backed shared state

The server can optionally persist KV cache blocks and the OpenAI Responses API
store to Redis, enabling cross-session KV reuse and durable response storage:

```bash
# Enable Redis for both KV cache and Responses API
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda \
    --redis-url localhost:6379

# KV cache tier only, with a 12-hour TTL
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda \
    --paged-kv-redis-url localhost:6379 --paged-kv-redis-ttl 720
```

## Feature × environment variable matrix

Quick reference for which environment variables (and matching CLI flags) gate each major feature. Variables in **bold** are required to turn the feature on; everything else is a tunable for a feature that's already enabled by default.

#### Continuous batching & paged KV cache

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Continuous-batching engine (`InferenceEngine` + scheduler) | ON in `TensorSharp.Server` | `TS_SCHED_DISABLE_BATCHED=1` to force per-seq fallback | `--no-continuous-batching` / `--continuous-batching` |
| Legacy per-session paged-KV manager | removed from Server request path | `TS_KV_PAGED_CACHE` (`0` / `1`), `TS_KV_BLOCK_SIZE` retained for compatibility / standalone tests | `--paged-kv` / `--no-paged-kv`, `--paged-kv-block-size N` |
| Legacy paged-KV SSD spillover (standalone manager) | OFF | `TS_KV_CACHE_MAX_RAM_MB`, `TS_KV_CACHE_SSD_DIR`, `TS_KV_CACHE_MAX_SSD_MB` | `--paged-kv-ram-mb`, `--paged-kv-ssd-dir`, `--paged-kv-ssd-mb` |
| Legacy paged-KV block quantization (standalone manager) | OFF (`0` = passthrough) | `TS_KV_PAGED_QUANT_BITS` (`0` / `2` / `4` / `8`) | `--paged-kv-quant-bits` |
| Block-hash prefix sharing across requests | ON | `TS_SCHED_PREFIX_CACHE=0` to disable | — |
| Scheduler tunables (per-step token budget, max in-flight seqs, prefill chunks, block pool size, decode quantum) | engine defaults | `TS_SCHED_MAX_BATCHED_TOKENS`, `TS_SCHED_MAX_RUNNING_SEQS`, `TS_SCHED_PREFILL_CHUNK`, `TS_SCHED_SOLO_PREFILL_CHUNK`, `TS_SCHED_NUM_BLOCKS`, `TS_SCHED_BLOCK_SIZE`, `TS_SCHED_DECODE_QUANTUM` | — |

#### Per-model batched / paged forward (`IBatchedPagedModel.ForwardBatch`)

| Model | Default state | Env var to flip default | Native-kernel sub-toggle |
|---|---|---|---|
| Mistral 3 | ON | — | `TS_PAGED_ATTN_KERNEL` = `native` (default) / `tensor` / `managed` |
| Gemma 4 | ON | `TS_GEMMA4_BATCHED=0` to force legacy per-seq | — |
| Qwen 3 | ON (reference port) | — | — |
| Qwen 3.5 / 3.6 family | ON | `TS_QWEN35_BATCHED=0` to force legacy per-seq (or `--no-continuous-batching`) | `TS_QWEN35_BATCHED_GDN_NATIVE=1` enables native batched GDN kernel; `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` overrides fused-attention engage threshold (default 4096) |
| GPT OSS | ON | `TS_GPTOSS_BATCHED=0` to force legacy per-seq | `TS_GPTOSS_PAGED_ATTN_MANAGED=1` forces the managed (C#) sinks softmax instead of the native paged-attention-with-sinks kernel |
| Nemotron-H | ON | `TS_NEMOTRON_BATCHED=0` to force legacy per-seq | `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` enables the native batched Mamba2 step (NEON SIMD + GCD parallelism) |
| Gemma 3 | not implemented (per-seq fallback) | — | — |
| DiffusionGemma | Separate diffusion scheduler in the Web UI path; not an `IBatchedPagedModel` autoregressive path | `DIFFUSION_MAX_BATCH`, `DIFFUSION_STEPS` | `DIFFUSION_BATCHED_FORWARD=1` enables true batched canvas decode; fused GGML decode is on by default unless disabled with `DIFFUSION_NO_FUSED_DECODE=1` |

#### MTP / NextN speculative decoding

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Speculative decode engine (solo sequences) | OFF | **`TS_MTP_SPEC=1`** | `--mtp-spec` / `--no-mtp-spec` |
| Max tokens drafted per step | `8` | `TS_MTP_DRAFT` | `--mtp-draft N` |
| Min draft confidence to keep a token | per drafter kind (`0.75` / `0.35`) | `TS_MTP_PMIN` | `--mtp-pmin X` |
| Gemma 4 separate draft GGUF (`gemma4-assistant`) | none | `TS_MTP_DRAFT_MODEL` | `--mtp-draft-model <path>` |
| Gemma 4 fused verify / draft kernels (ggml) | ON | `TS_GMTP_NO_FUSED=1` falls back to per-op | — |
| Gemma 4 dense fast rollback on partial accept | ON | `TS_GMTP_NO_FAST_ROLLBACK=1` restores kept-prefix rollback | — |
| Gemma 4 verify trunk path | linear (solo) | `TS_GMTP_BATCHED_TRUNK=1` runs the batched paged trunk | — |

#### Tensor parallelism & distributed inference

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Local tensor parallelism (multi-GPU, single process) | OFF (`1` GPU) | **`TENSORSHARP_TP_DEGREE=N`** | `--tp N` (CLI and server) |
| GPU ordinals used by the TP ranks (GGML backends) | `0..tp-1` | `TENSORSHARP_TP_DEVICES=0,2` | — |
| Distributed TP node ID (multi-node) | unset (disabled) | **`TENSORSHARP_TP_NODE_ID=N`** | `--tp-node-id N` (CLI and server; the server must be node `0`) |
| Distributed TP peer endpoints | unset (disabled) | **`TENSORSHARP_TP_PEERS=host1:port1,host2:port2`** | `--tp-peers host1:port1,host2:port2` (CLI and server) |
| Peer connect retry window (multi-node) | `120` s | `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | — |
| Per-receive timeout (multi-node) | `300` s | `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | — |
| Force host-staged cross-GPU copies | OFF (P2P when available) | `TENSORSHARP_TP_DISABLE_P2P=1` | — |
| Force host-staged local AllReduce | OFF (device-to-device) | `TENSORSHARP_TP_HOST_ALLREDUCE=1` | — |

#### Redis shared state (server)

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Redis KV cache tier | OFF | **`TS_KV_CACHE_REDIS_URL`** | `--redis-url` or `--paged-kv-redis-url` |
| Redis KV cache entry TTL | `1440` min (24 h) | `TS_KV_CACHE_REDIS_TTL_MINUTES` (`0` = no TTL) | `--paged-kv-redis-ttl` |
| Redis Responses API store | OFF (in-memory) | **`TS_RESPONSES_STORE_REDIS_URL`** | `--redis-url` |

#### Backends

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Default compute backend | `ggml_metal` (macOS), `ggml_cpu` (Windows/Linux) | `BACKEND` | `--backend` |
| MLX backend library lookup | probe app dir | `TENSORSHARP_MLX_LIBRARY` (full path to `libmlxc`), `TENSORSHARP_MLX_LIBRARY_DIR` (directory) | — |
| MLX pipelined greedy decode (CLI only) | ON when eligible | `TS_MLX_PIPELINED_DECODE=0` disables | — |
| MLX `mlock(2)` of GGUF mmap so weights stay resident | ON | `TS_MLX_MLOCK_GGUF=0` to disable | — |
| MLX fused multi-dim KV write (single `slice_update` per cache block) | ON | `TS_MLX_FUSED_KV_WRITE=0` to revert to per-head loop | — |
| MLX batched MoE decode (Qwen 3.5/3.6 MoE) | ON | `TS_MLX_BATCHED_MOE_DECODE=0` for legacy per-expert path | — |
| MLX fused MoE gate+up+SiLUMul Metal kernel | ON | `TS_MLX_MOE_FUSED_GATE_UP_SILU=0` for legacy 3-dispatch | — |
| MLX on-device MoE router top-K + softmax | ON when prerequisites are met | `TS_MLX_DEVICE_ROUTER=0` disables | — |
| MLX layer-boundary `async_eval` cadence | Gemma 4: every 4 layers; Qwen / Nemotron: every 16 layers | `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS=N` or `TS_MLX_EVAL_EVERY_N_LAYERS=N` (`0` = disabled where supported) | — |
| MLX allocator caps (memory / cache / wired buffer) | host-derived | `TS_MLX_MEMORY_LIMIT_MB`, `TS_MLX_CACHE_LIMIT_MB`, `TS_MLX_WIRED_LIMIT_MB` | — |

#### Sampling defaults (server-only)

These fill in fields the request body omits. CLI flags win over env vars, and
anything set through either outranks the request body unless the server runs
with `--sampling-precedence request` (see [Web Application](#web-application)).

| Sampling field | Env var | CLI equivalent |
|---|---|---|
| `temperature` | `TENSORSHARP_TEMPERATURE` | `--temperature` |
| `top_k` | `TENSORSHARP_TOP_K` | `--top-k` |
| `top_p` | `TENSORSHARP_TOP_P` | `--top-p` |
| `min_p` | `TENSORSHARP_MIN_P` | `--min-p` |
| `repeat_penalty` | `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` |
| `presence_penalty` | `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` |
| `frequency_penalty` | `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` |
| `seed` | `TENSORSHARP_SEED` | `--seed` |
| max tokens | `MAX_TOKENS` | `--max-tokens` |
| stop sequences | — (CLI / per-request only) | `--stop` (repeatable) |
| sampling precedence | `TENSORSHARP_SAMPLING_PRECEDENCE` | `--sampling-precedence` |

#### Hosting & uploads (server-only)

| Feature | Default | Env vars |
|---|---|---|
| ASP.NET Core listener | `http://0.0.0.0:5000` | `--port` / `--host` / `--urls`, then `PORT` / `HOST`, then `ASPNETCORE_URLS` |
| Text and born-digital PDF uploads | Full extracted content; the final rendered prompt must fit the loaded model context | — |
| Video-frame extraction | 1 fps (time-based, no cap) | `VIDEO_SAMPLE_FPS`, `VIDEO_MAX_FRAMES` |
| DiffusionGemma Web UI denoising | 48 steps, max batch 2 | `DIFFUSION_STEPS`, `DIFFUSION_MAX_BATCH` |

#### Logging (server + CLI)

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Console + file log minimum level | `Information` | `TENSORSHARP_LOG_LEVEL` | `--log-level` |
| File logger output directory | `<binDir>/logs` | `TENSORSHARP_LOG_DIR` | `--log-dir` |
| File logger enabled | ON | `TENSORSHARP_LOG_FILE=0` to disable | `--log-file 0\|1` |
| Console logger enabled | ON | — | `--log-console 0\|1` (CLI only) |

#### Native build (compile-time only)

These are read by `build-linux.sh` / `build-windows.ps1` / the auto-build during `dotnet build` for `TensorSharp.GGML.Native`, not at run time.

| Feature | Default | Env vars | Build-script flag |
|---|---|---|---|
| Enable GGML CUDA in the native build | auto-detected from toolchain | `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` | `--cuda` / `--no-cuda` |
| Enable GGML Vulkan in the native build | auto-detected from the installed Vulkan runtime; a portable toolchain (headers, glslc, SPIRV-Headers) is downloaded when no Vulkan SDK / dev packages are installed | `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON/OFF` | `--vulkan` / `--no-vulkan` |
| Narrow `CMAKE_CUDA_ARCHITECTURES` list | auto-detected from visible GPU | `TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES` | `--cuda-arch='86-real;89-real'` |
| Native build parallelism cap | all CPUs, bounded by RAM (~3 GB per `nvcc` job) | `TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL` | — |
| Native build CMake generator (Windows) | Ninja when available, else `Visual Studio NN` | `CMAKE_GENERATOR` | `-G <generator>` |
| Visual Studio installation used by the native build (Windows) | auto-detected, including installs flagged incomplete | `TENSORSHARP_VS_INSTALL_DIR` | — |

## Server Logging

The server emits one structured Information-level entry at the start and end of
every chat / generate turn, so a single grep over the log file provides a compact
request-response audit trail without replaying any traffic.

| Event id | Emitted on | Carries |
|---|---|---|
| `ChatStarted` (1500) | `chat.start`, `generate.start`, plus per-protocol request banners | sampling config, message + attachment counts, `userInput=` (bounded latest-user preview), and `fullInput=` (JSON array of every message with attachment paths, original character counts, and bodies capped at 512 characters). Inlined uploaded documents are replaced by an omission marker while retaining the user's trailing instruction. `/api/generate` likewise logs a bounded prompt preview. |
| `ChatCompleted` (1502) | `chat.complete`, `generate.complete` | token counts, KV cache reuse (`kvReused`, `kvReusePercent`), TTFT, elapsed, throughput, finish reason, full raw assistant output (reasoning + result) |
| `ChatAborted` (1503) | client disconnected mid-stream | partial output, KV reuse fraction at the time of abort |
| `KvCacheReusePlan` (1510) | per-prefix-reuse decision | `Debug`-level fine-grained breakdown (exact match / partial / full reset) |
| `HttpRequestStarted/Completed` (1100/1101) | every HTTP request | method, path, remote IP, status, duration; `/api/queue/status` is demoted to `Debug` so high-frequency UI polling does not drown out the per-turn entries |

The raw assistant output captures `<think>...</think>`, `<|channel|>analysis`,
and any other inline framing the model emits, so the completion entry contains
both reasoning and the user-visible result. Input bodies are deliberately bounded:
losslessly uploaded documents are not duplicated into logs, avoiding large memory/I/O
spikes and accidental document disclosure. The upload manifest and `contentChars`
retain useful audit metadata; capture requests separately when byte-for-byte replay is
required. Set `TENSORSHARP_LOG_LEVEL=Warning` to suppress per-turn Information logs.

Sample `fullInput` payload (formatted for readability; it is emitted as a
single line in the actual log):

```json
[
  {"role":"system","content":"You are a helpful assistant.","contentChars":28},
  {"role":"user","content":"What is the tallest mountain?","contentChars":29},
  {"role":"assistant","content":"Mount Everest.","contentChars":14},
  {"role":"user","content":"How tall is it?","contentChars":15,"images":["/uploads/mountain.jpg"]}
]
```

The same per-turn KV cache reuse stats are surfaced through every API:

- **Web UI SSE** (`POST /api/chat`) - the `done` event carries `promptTokens`, `kvReusedTokens`, and `kvReusePercent`.
- **Ollama NDJSON** (`POST /api/generate`, `POST /api/chat/ollama`) - the final chunk and the non-streaming response carry `prompt_cache_hit_tokens` (int) and `prompt_cache_hit_ratio` (0..1).
- **OpenAI** (`POST /v1/chat/completions`) - the `usage` block carries `prompt_tokens_details.cached_tokens`, matching the OpenAI extension that existing SDKs already understand.

The Web UI footer line under each assistant message also surfaces the cache hit
inline (e.g. `187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)`).

## HTTP APIs

TensorSharp.Server exposes three API styles. See [API_EXAMPLES.md](TensorSharp.Server/API_EXAMPLES.md) for full documentation with curl and Python examples.

**Ollama-compatible API:**

```bash
# List models
curl http://localhost:5000/api/tags

# Generate text
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# Chat
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# Chat with thinking mode
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Solve 17*23"}], "think": true, "stream": false}'

# Chat with tool calling
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "What is the weather?"}], "tools": [{"function": {"name": "get_weather", "description": "Get current weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'
```

**OpenAI-compatible API:**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# Structured outputs (OpenAI response_format)
#
# Enforced by grammar-constrained decoding: the schema is compiled to a grammar
# and any token that would break it is removed from the distribution before
# sampling, so the response is structurally valid by construction rather than
# repaired afterwards. Supported keywords: type, enum, const, properties,
# required, additionalProperties, items, prefixItems, min/maxItems, anyOf, oneOf,
# allOf, $ref/$defs (including recursive), min/maxLength, pattern, the
# date/time/date-time/uuid formats, and integer minimum/maximum.
# Refused up front (a CFG cannot express them): not, if/then/else,
# dependentSchemas, dependentRequired, multipleOf, patternProperties.
# Set TS_JSON_GRAMMAR=0 to fall back to the older prompt-and-repair behaviour.
#
# NOTE: the grammar guarantees what it emits is well-formed, but it cannot
# guarantee the document FITS in max_tokens. Because end-of-sequence stays
# masked until the JSON is complete, too small a budget truncates mid-object.
# Give structured requests enough headroom.
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Extract the city and country from: Paris, France."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "confidence": {"type": ["string", "null"]}
          },
          "required": ["city", "country", "confidence"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q8_0.gguf",
    messages=[{"role": "user", "content": "What is 2+3?"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**Queue status:**

```bash
curl http://localhost:5000/api/queue/status
# {"busy":false,"pending_requests":0,"total_processed":42}
```

