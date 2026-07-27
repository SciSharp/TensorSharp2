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

# Thinking / reasoning mode
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# Tool calling
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# With sampling parameters
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

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

# Tensor parallelism: split a model across 2 CUDA GPUs in one process
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
| `--kv-cache-dtype <type>` | KV cache precision: `f32`, `f16`, `q8_0`, or `q4_0` (default: auto — the backend/model pick; env `KV_CACHE_DTYPE`). Half-precision / quantized KV caches reduce memory at the cost of small numerical drift; `q4_0` (~0.56 bytes/elem, ~1/7 of f32) is the most aggressive tier for very long (128K–256K) contexts where the KV cache dominates memory. Block-quantized caches (`q8_0`/`q4_0`) require the native GGML flash path. |
| `--interactive` / `-i` | Start an interactive REPL chat session (turn-by-turn input/output) with KV cache reuse, slash commands, hot-swappable model/backend/projector, file attachments (image, audio, video, text) and live sampling tuning. See the **Interactive REPL commands** section below for the full list. |
| `--system <text>` | System prompt to seed the interactive session (overridden inside the REPL by `/system`) |
| `--system-file <path>` | Read the initial system prompt from a UTF-8 text file (alternative to `--system`) |
| `--think` | Enable thinking/reasoning mode (chain-of-thought) |
| `--tools <path>` | JSON file with tool/function definitions |
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
| `--tp <N>` | Tensor parallelism degree — split the model across N CUDA GPUs in a single process (default: `1`). Requires `--backend cuda`. See [Tensor Parallelism & Distributed Inference](#tensor-parallelism--distributed-inference). |
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

Open `http://localhost:5000/index.html` in your browser (`GET /` is the liveness endpoint). The web interface supports:

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
| `--gpu-device <N>` | Vulkan device index for the `ggml_vulkan` backend on multi-GPU hosts (e.g. an integrated Intel GPU next to a discrete NVIDIA one). Defaults to device 0; use `--list-gpus` to see the indices. Also settable via the `TS_GGML_VULKAN_DEVICE` env var. |
| `--list-gpus` | List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit |
| `--help` | Print the parameter reference (also shown when the server is started with no arguments) and exit |
| `--max-tokens <N>` | Default maximum tokens to generate when a request omits the limit (default: `20000`) |
| `--temperature <f>` | Default sampling temperature when a request does not provide one (`0` = greedy) |
| `--top-k <N>` | Default top-K filtering when a request does not provide one (`0` = disabled) |
| `--top-p <f>` | Default nucleus sampling threshold when a request does not provide one (`1.0` = disabled) |
| `--min-p <f>` | Default min-p filtering when a request does not provide one (`0` = disabled) |
| `--repeat-penalty <f>` | Default repetition penalty when a request does not provide one (`1.0` = none) |
| `--presence-penalty <f>` | Default presence penalty when a request does not provide one (`0` = disabled) |
| `--frequency-penalty <f>` | Default frequency penalty when a request does not provide one (`0` = disabled) |
| `--seed <N>` | Default random seed when a request does not provide one (`-1` = non-deterministic) |
| `--stop <string>` | Default stop sequence (can be repeated). Per-request `stop`/`stop_sequences` fully replace the default list rather than merge with it. |
| `--kv-cache-dtype <type>` | KV cache precision for the hosted model: `f32`, `f16`, `q8_0`, or `q4_0` (quantized caches trade small numerical drift for memory; see the CLI table above for the tier trade-offs). Default: auto — the backend/model pick. Env: `KV_CACHE_DTYPE`. |
| `--continuous-batching` / `--no-continuous-batching` | Enable (default) or disable iteration-level paged-batching. When enabled the server admits / preempts sequences mid-batch and packs them into one forward pass on models that implement `IBatchedPagedModel`. `--no-continuous-batching` falls back to per-sequence KV-swap for every model. Alias: `--paged-batching` / `--no-paged-batching`. |
| `--prefill-chunk-size <N>` | Chunked-prefill granularity under contention — the maximum prefill tokens scheduled per step while other requests are running, so parallel decodes get frequent turns at the GPU (default: `1024`). Env: `TS_SCHED_PREFILL_CHUNK`. |
| `--mtp-spec` / `--no-mtp-spec` | Enable NextN/MTP speculative decoding (default off) on models that ship a multi-token-prediction draft head (Qwen 3.6's embedded NextN block, or a Gemma 4 `gemma4-assistant` draft loaded via `--mtp-draft-model`). Engages for solo (non-concurrent) sequences: the draft head proposes up to `--mtp-draft` tokens per step and the trunk verifies them in one batched forward, with the request's own sampler (penalties included) driving both drafting and verification, so output matches standard decode. Engaged automatically only where profitable (ggml backends and the pure-C# `cuda` backend); CPU / MLX serve standard decode. Env: `TS_MTP_SPEC`. |
| `--mtp-draft <N>` | Maximum tokens drafted per speculative step (default `8`). Env: `TS_MTP_DRAFT`. |
| `--mtp-pmin <f>` | Minimum draft-head confidence in `(0, 1]` for a drafted token to be kept; drafting stops at the first low-confidence token (default `0.75`). Env: `TS_MTP_PMIN`. |
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
`frequency_penalty`, `seed`, `stop`/`stop_sequences`) always win over these
server-wide defaults; the defaults only fill in fields the client omits.

**Runtime environment variables:**

| Variable | Description |
|---|---|
| `BACKEND` | Default compute backend (`cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan`), used when `--backend` is not passed (default: `ggml_metal` on macOS, `ggml_cpu` elsewhere) |
| `MAX_TOKENS` | Default maximum generation length when neither `--max-tokens` nor a request-level limit is set (default: `20000`) |
| `VIDEO_SAMPLE_FPS` | Frames sampled per second of video for video prompts; time-based extraction (default: `1`) |
| `VIDEO_MAX_FRAMES` | Optional upper bound on extracted video frames (evenly down-sampled); unset/`0` means no cap (default: no cap) |
| `PORT` / `ASPNETCORE_URLS` | Currently overridden by the fixed `http://0.0.0.0:5000` listener in `Program.cs`; Docker Space images rewrite that constant with `APP_PORT` at build time. |
| `TENSORSHARP_TEMPERATURE` | Default sampling temperature when neither `--temperature` nor the request body sets one |
| `TENSORSHARP_TOP_K` | Default top-K when neither `--top-k` nor the request body sets one |
| `TENSORSHARP_TOP_P` | Default top-P when neither `--top-p` nor the request body sets one |
| `TENSORSHARP_MIN_P` | Default min-P when neither `--min-p` nor the request body sets one |
| `TENSORSHARP_REPEAT_PENALTY` | Default repetition penalty when neither `--repeat-penalty` nor the request body sets one |
| `TENSORSHARP_PRESENCE_PENALTY` | Default presence penalty when neither `--presence-penalty` nor the request body sets one |
| `TENSORSHARP_FREQUENCY_PENALTY` | Default frequency penalty when neither `--frequency-penalty` nor the request body sets one |
| `TENSORSHARP_SEED` | Default random seed when neither `--seed` nor the request body sets one |
| `TENSORSHARP_LOG_LEVEL` | Minimum log level for both console and file loggers: `Trace`, `Debug`, `Information`, `Warning`, `Error`, `Critical` (default: `Information`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_DIR` | Directory the JSON-line file logger writes to (default: `<binDir>/logs`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_FILE` | Set to `0` to disable the file logger and keep only the console output (default: enabled). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_TP_DEGREE` | Tensor parallelism degree — number of local CUDA GPUs to split the model across (default: `1`). Used by `TensorSharp.Server` (which has no `--tp` CLI flag) and as a fallback in `ModelBase.Create`. Requires `--backend cuda`. |
| `TENSORSHARP_TP_NODE_ID` | This node's 0-based ID for multi-node distributed tensor parallelism. Must be set together with `TENSORSHARP_TP_PEERS`. |
| `TENSORSHARP_TP_PEERS` | Comma-separated `host:port` list of all nodes in the distributed TP cluster (e.g. `192.168.1.10:9500,192.168.1.11:9500`). Must be set together with `TENSORSHARP_TP_NODE_ID`. |
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
| `TS_MTP_PMIN` | Minimum draft-head confidence in `(0, 1]` to keep a drafted token (default `0.75`). CLI: `--mtp-pmin`. |
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

Sampling parameter precedence (highest wins):

1. Per-request JSON fields in the API call (e.g. `temperature`, `top_p`, `stop`).
2. Server-wide CLI flags (e.g. `--temperature`, `--top-p`, `--stop`).
3. `TENSORSHARP_*` environment variables listed above.
4. Built-in `SamplingConfig` defaults (`temperature=1.0`, `top_k=0`, `top_p=1.0`, `min_p=0`, `repeat_penalty=1.0`, presence/frequency penalties `0`, `seed=-1`, no stop sequences).

## Tensor Parallelism & Distributed Inference

TensorSharp supports **tensor parallelism (TP)** — splitting a single model across
multiple GPUs using the Megatron-LM column/row-parallel pattern — and
**distributed (multi-node) tensor parallelism**, where TP spans multiple
machines connected over a TCP peer-to-peer network.

### Local tensor parallelism (single process, multiple GPUs)

Split the model across N CUDA GPUs within one process. Each GPU holds `1/N` of
the sharded weights (column-parallel QKV/gate/up, row-parallel output/down) plus
a full copy of replicated weights (norms, embeddings, LM head). Per-GPU KV
caches are independent; AllReduce (via CUDA P2P copies + elementwise-add kernel)
reconverges the hidden state after each row-parallel projection.

```bash
# CLI: 2-GPU tensor parallelism
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2

# Server: via environment variable (the server has no --tp CLI flag)
TENSORSHARP_TP_DEGREE=2 dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
    --model <model.gguf> --backend cuda

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

# Server: via environment variables
# Node 0:
TENSORSHARP_TP_DEGREE=2 TENSORSHARP_TP_NODE_ID=0 \
TENSORSHARP_TP_PEERS=192.168.1.10:9500,192.168.1.11:9500 \
    dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda

# Node 1:
TENSORSHARP_TP_DEGREE=2 TENSORSHARP_TP_NODE_ID=1 \
TENSORSHARP_TP_PEERS=192.168.1.10:9500,192.168.1.11:9500 \
    dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda

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
| Gemma 4 | ✅ | Dense + MoE expert slicing, per-layer head dims |
| Qwen 3.5 / 3.6 family | ✅ | GatedDeltaNet SSM with per-rank V-head ownership, MoE expert slicing |
| GPT OSS | ✅ | MoE expert slicing, attention sinks, YaRN |
| Nemotron-H | ✅ | Mamba2 replicated on rank 0, MoE expert slicing |
| DiffusionGemma | — | Not applicable (diffusion model) |
| Qwen-Image-Edit | — | Not applicable (image generation) |

### Constraints

- **CUDA backend only** (`--backend cuda`). GGML, MLX, and Vulkan backends are single-device by design.
- `numHeads`, `numKVHeads`, and `intermediateSize` must be divisible by the TP degree.
- Quantized row-parallel splits require `ne0` divisible by `tp × blockSize`.
- Batched/continuous-batching forward under TP is implemented for Qwen 3 and Mistral 3; MoE models (Gemma 4, Qwen 3.5/3.6, GPT OSS, Nemotron-H) fall back to per-sequence forward under TP.

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
| Min draft-head confidence to keep a token | `0.75` | `TS_MTP_PMIN` | `--mtp-pmin X` |
| Gemma 4 separate draft GGUF (`gemma4-assistant`) | none | `TS_MTP_DRAFT_MODEL` | `--mtp-draft-model <path>` |
| Gemma 4 fused verify / draft kernels (ggml) | ON | `TS_GMTP_NO_FUSED=1` falls back to per-op | — |
| Gemma 4 dense fast rollback on partial accept | ON | `TS_GMTP_NO_FAST_ROLLBACK=1` restores kept-prefix rollback | — |
| Gemma 4 verify trunk path | linear (solo) | `TS_GMTP_BATCHED_TRUNK=1` runs the batched paged trunk | — |

#### Tensor parallelism & distributed inference

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Local tensor parallelism (multi-GPU, single process) | OFF (`1` GPU) | **`TENSORSHARP_TP_DEGREE=N`** | `--tp N` (CLI only) |
| Distributed TP node ID (multi-node) | unset (disabled) | **`TENSORSHARP_TP_NODE_ID=N`** | `--tp-node-id N` (CLI only) |
| Distributed TP peer endpoints | unset (disabled) | **`TENSORSHARP_TP_PEERS=host1:port1,host2:port2`** | `--tp-peers host1:port1,host2:port2` (CLI only) |

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

These fill in fields the request body omits; per-request JSON always wins, CLI flags win over env vars.

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

#### Hosting & uploads (server-only)

| Feature | Default | Env vars |
|---|---|---|
| ASP.NET Core listener | `http://0.0.0.0:5000` | Fixed in `Program.cs`; Docker Space images rewrite it with the `APP_PORT` build arg |
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
| Native build parallelism cap | conservative auto-cap | `TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL` | — |

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

