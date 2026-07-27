# Features
[English](FEATURES.md) | [中文](FEATURES_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation.


- **Multi-architecture support** -- Gemma 4, Gemma 3, DiffusionGemma, Qwen 3, Qwen 3.5/3.6-family, GPT OSS, Nemotron-H, Mistral 3, and Qwen-Image-Edit (image editing)
- **Multimodal inference** -- image, video, and audio inputs (Gemma 4); images for Gemma 3 / Qwen 3.5-family / Mistral 3 / Nemotron-H Omni
- **Thinking / reasoning mode** -- structured chain-of-thought output with `<think>` / `<|channel>thought` / `<|channel>analysis` tags (Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H)
- **Tool calling / function calling** -- models can invoke user-defined tools; multi-turn tool-call conversations supported across all three API styles
- **Quantized model support** -- loads GGUF files with Q4_K_M, Q8_0, F16, MXFP4, and other quantization formats; performs native quantized matmul without dequantizing to FP32, including memory-efficient pure C# CPU loading for large GGUFs
- **GPU-accelerated** -- GGML Metal on macOS, GGML CUDA on Windows/Linux with NVIDIA GPUs, GGML Vulkan on Windows/Linux with AMD/Intel/NVIDIA GPUs, a direct CUDA/cuBLAS backend with PTX kernels, and an MLX backend for Apple Silicon (mlx-c / Metal), all with CPU fallbacks for unsupported ops
- **Optimized pure C# CPU backend** -- managed GEMM fast paths plus fused SIMD kernels for RMSNorm, RoPE, softmax, fused activations, and other inference hot paths
- **Continuous batching & paged KV cache** -- vLLM-style block-paged KV pool with block-hash prefix sharing across requests, iteration-level scheduler that admits / preempts sequences mid-batch, optional SSD-backed tier for very large KV working sets, and a native fused paged-attention kernel (`TSGgml_PagedAttentionForward`) that drives `ggml_flash_attn_ext` on Metal/CUDA/Vulkan. Enabled by default in `TensorSharp.Server`; opt-out with `--no-continuous-batching`. See [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).
- **MTP / NextN speculative decoding** -- multi-token-prediction draft heads accelerate solo (non-concurrent) decode. Qwen 3.6 ships its NextN block fused into the trunk GGUF; Gemma 4 loads a separate EAGLE-style `gemma4-assistant` draft GGUF via `--mtp-draft-model` whose draft layers attend the target's own KV cache. The draft proposes up to `--mtp-draft` tokens per step (kept while draft confidence ≥ `--mtp-pmin`) and the trunk verifies them in a single batched forward; the request's own sampler — penalties included — drives both drafting and verification, so output is identical to standard decode. Opt in with the server's `--mtp-spec` flag (off by default; `TensorSharp.Cli` has no MTP flags — set the `TS_MTP_*` env vars there). On ggml backends fused multi-token-verify / draft-step kernels make it a clear win; the pure-C# `cuda` backend runs a fully GPU-resident per-op verify/draft and is also a win. CPU / MLX stay on standard decode. Env: `TS_MTP_*` (shared) and `TS_GMTP_*` (Gemma 4 tuning).
- **Batched / parallel inference** -- `IBatchedPagedModel.ForwardBatch` implementations for Mistral 3, Gemma 4, GPT OSS, Qwen 3, Qwen 3.5/3.6-family, and Nemotron-H all run by default and pack N sequences into a single forward pass with paged K/V scatter and per-sequence attention via the native kernel. Gemma 4, Qwen 3.5/3.6, GPT OSS, and Nemotron-H expose a per-family `TS_<FAMILY>_BATCHED=0` escape hatch (`TS_GEMMA4_BATCHED=0`, `TS_QWEN35_BATCHED=0`, `TS_GPTOSS_BATCHED=0`, `TS_NEMOTRON_BATCHED=0`) to fall back to the per-sequence KV-swap path for A/B comparison or regression isolation; Qwen 3 and Mistral 3 have no per-family switch — use the global `TS_SCHED_DISABLE_BATCHED=1`.
- **Tensor parallelism & distributed inference** -- split a model across multiple CUDA GPUs (Megatron-LM column/row-parallel pattern) with `--tp N` (CLI) or `TENSORSHARP_TP_DEGREE` (server), and extend across machines with peer-to-peer TCP clustering (`--tp-node-id` / `--tp-peers`). Hierarchical AllReduce minimizes inter-node traffic. Supports all autoregressive architectures (Qwen 3, Mistral 3, Gemma 3/4, Qwen 3.5/3.6-family, GPT OSS, Nemotron-H) with architecture-specific strategies for MoE expert slicing, GatedDeltaNet per-rank V-head ownership, and Mamba2 replication. Optional Redis-backed KV cache and Responses API store for shared state. → [Tensor Parallelism](USAGE.md#tensor-parallelism--distributed-inference)
- **Ollama & OpenAI API compatibility** -- drop-in replacement endpoints for existing tooling
- **Configurable sampling** -- temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, seed, stop sequences
- **Chat templates** -- auto-loaded from GGUF metadata (Jinja2), with hardcoded fallbacks per architecture
- **Inference engine** -- the new `InferenceEngine` (worker-thread scheduler + paged block pool) replaces the legacy single-request FIFO queue inside `TensorSharp.Server`. The old queue object is now a compatibility shim for status/event shapes; the engine itself handles concurrency.
- **Batch processing** -- JSONL input support in the console application, plus a built-in inference benchmark for prefill/decode throughput
- **Streaming** -- token-by-token output via SSE (web) or stdout (console), with abort/stop support for in-flight generations
- **Text-diffusion generation** -- DiffusionGemma uses an iterative EntropyBound denoising sampler instead of autoregressive `Forward()`. The CLI exposes `--diffusion-steps`, `--diffusion-seed`, and `--diffusion-blocks`; the Web UI streams whole-message `replace` events for live denoising previews and batches concurrent diffusion requests through `DiffusionBatchScheduler`.
- **Image editing (Qwen-Image-Edit)** -- a prompt plus an input image produces an edited image. The loaded `qwen_image` GGUF is the MMDiT diffusion transformer; TensorSharp resolves two companion GGUFs alongside it — the Qwen-Image VAE (image ↔ 16-channel latent) and the Qwen2.5-VL-7B text encoder (prompt → 3584-dim conditioning, optional vision grounding via an `mmproj`). The pipeline VAE-encodes the reference, builds text (and optional image) conditioning, runs a FlowMatch-Euler true-CFG denoise loop with reference-latent concatenation, then VAE-decodes back to pixels. The whole 60-block DiT forward is CUDA-graph-captured (`TSGgml_QwenImageForward`), flash-attention is on by default, and the target area is auto-clamped to the device VRAM budget. An optional Lightning distillation LoRA (`--qwen-image-lora` / `TS_QWEN_IMAGE_LORA`, `.safetensors`) merges into the DiT weights at load time, cutting the denoise to the LoRA's step count (e.g. 4 or 8) and switching CFG to 1.0 (no negative pass). Driven from C# via `QwenImageModel.EditImage(prompt, RgbImage, QwenImageParams)`, from the CLI image-edit mode (`--image`, `--prompt`, `--cfg`, `--diffusion-steps`, `--diffusion-seed`), and from the Web UI with live denoising previews. → [Qwen-Image-Edit card](docs/models/qwenimage.md)
- **Hybrid SSM-Transformer** -- Nemotron-H mixes Mamba2 SSM layers, attention-only layers, and MoE FFN layers in a single model. The Mamba2 step has both a per-sequence native kernel and a batched native kernel (`TSGgml_NemotronMamba2BatchedStepF32`, NEON SIMD + GCD parallelism) used by the batched path.
- **Hybrid Attention-Recurrent** -- Qwen 3.5/3.6-family models mix full-attention layers with GatedDeltaNet recurrent layers; the batched path keeps recurrent running state in a per-slot recurrent-state pool
- **Mixture of Experts** -- Gemma 4 MoE variants (e.g. gemma-4-26B-A4B), GPT OSS MoE (e.g. gpt-oss-20b), Qwen 3.5/3.6-family MoE (`qwen35moe` / `qwen3next` variants such as Qwen3.5-35B-A3B), and Nemotron-H MoE FFN layers
- **Batched GPU MoE** -- a single fused GGML graph dispatch handles all selected experts (plus the optional shared expert and residual add) for Qwen 3.5/3.6-family and Nemotron-H decode, eliminating per-expert round-trips
- **KV cache codecs** -- pluggable codec interface (`IKvBlockCodec`) with a built-in TurboQuant (2-bit affine / Q4 / Q8) compressed codec for paged blocks. The CLI accepts all four `--paged-kv-quant-bits 0|2|4|8` values; the server's legacy standalone flag accepts `0|4|8`, while `TS_KV_PAGED_QUANT_BITS=2` selects the 2-bit codec directly. The 2-bit tier reaches ~10x compression on fp32 blocks for very long contexts.
- **Message editing** -- edit or delete previous messages in the web chat UI and regenerate from that point
- **Text/Image/Audio/Video/PDF uploads** -- the web UI accepts file uploads up to 500 MB and preserves text content in full. Born-digital PDFs have their complete text layer extracted and inlined into the prompt (cap pages explicitly with `TS_PDF_MAX_PAGES`); scanned PDFs are rendered to page images for vision-capable models. The final prompt is checked against the model's actual context window instead of an arbitrary upload budget. The CLI accepts a PDF in one-shot mode via `--pdf <file>`
- **Per-turn observability** -- structured logs capture the full user input and the full raw assistant output (both `<think>` reasoning and the final result) plus the KV cache hit ratio. The same cache-hit stats are surfaced through every API: `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio` (Ollama), `usage.prompt_tokens_details.cached_tokens` (OpenAI), and `promptTokens` / `kvReusedTokens` / `kvReusePercent` in the Web UI SSE `done` event


## Thinking / Reasoning Mode

Models that support thinking mode (Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H) can produce structured chain-of-thought reasoning before generating the final answer. The thinking content is separated from the main response and can be displayed or hidden by the client.

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H:** uses `<think>...</think>` tags
- **Gemma 4:** uses `<|channel>thought\n...<channel|>` tags
- **GPT OSS:** uses Harmony format with `<|channel|>analysis` for thinking and `<|channel|>final` for the response

Enable via `--think` (console), `"think": true` (Ollama API), or the thinking toggle in the web UI.

## MTP / NextN Speculative Decoding

Some architectures ship a **multi-token-prediction (MTP / NextN) draft head** that lets `TensorSharp.Server` run lossless speculative decoding for solo (non-concurrent) sequences. The draft proposes several future tokens cheaply, the trunk verifies all of them in one batched forward, and accepted tokens are committed in a single step. Because the request's own sampler — temperature, top-k/p, and repetition/presence/frequency penalties — drives both the draft and the verify, the output is identical to standard decode; speculation only changes how many forward passes it takes to produce it.

Speculative decoding is **off by default**. Enable it on the server with `--mtp-spec` (env `TS_MTP_SPEC=1`):

```bash
# Qwen 3.6 — use the -MTP- repository GGUF so the embedded NextN block is retained
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft 8 --mtp-pmin 0.75

# Gemma 4 — load the separate gemma4-assistant draft GGUF that matches the target
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

**Two draft-head shapes:**

- **Qwen 3.6 (embedded NextN)** — the GGUF carries one extra decoder block past the main stack (`{arch}.nextn_predict_layers`) plus the NextN projection/norm tensors. No separate file is required; `--mtp-draft-model` is ignored. The recurrent trunk state (GatedDeltaNet) is snapshotted so a partially-rejected verify batch can be rolled back.
- **Gemma 4 (separate `gemma4-assistant` GGUF)** — an EAGLE-style recurrent drafter loaded with `--mtp-draft-model`. It holds no K/V of its own: every draft layer queries the **target model's** existing per-layer KV cache (last local + last global layer), so the drafter is stateless given `(token, hidden)`. The draft's hidden size must match the target — pair the 12B target with its 12B draft, not the 26B-A4B draft. A mismatched, missing, or incomplete draft GGUF **fails fast at startup** with a remediation hint instead of silently disabling speculation.

**Where it's profitable** (engaged automatically; otherwise the engine serves standard decode):

| Backend | Qwen 3.6 | Gemma 4 |
|---|---|---|
| GGML CUDA / GGML Metal | ✅ fused multi-token-verify + draft-step kernels | ✅ fused multi-token-verify + draft-step kernels |
| Direct CUDA (`cuda`, pure C#) | ✅ GPU-resident per-op verify/draft | ✅ GPU-resident per-op verify/draft |
| CPU / GGML CPU / MLX | standard decode (verify can't keep up) | standard decode |

Tuning: `--mtp-draft` (default `8`) bounds tokens drafted per step; `--mtp-pmin` (default `0.75`) is the minimum draft-head confidence to keep a token (drafting stops at the first low-confidence token). Gemma 4 draft-path A/B switches are the `TS_GMTP_*` env vars (see the **MTP / speculative-decoding tunables** table under [Web Application](USAGE.md#web-application)). Per-architecture mechanics are in the [Qwen 3.5/3.6 card](docs/models/qwen35.md) and the [Gemma 4 card](docs/models/gemma4.md).

## Tensor Parallelism & Distributed Inference

Tensor parallelism (TP) splits a single model across multiple CUDA GPUs using the
Megatron-LM column/row-parallel pattern. Each transformer block runs
column-parallel projections (QKV, gate/up) that split output heads or
intermediate dimensions across GPUs, independent per-GPU attention or activation
computation, and row-parallel projections (output, down) followed by an AllReduce
that reconverges the hidden state. Norms, embeddings, and the LM head are
replicated.

**Local TP** runs within a single process: one thread issues commands to all GPUs
sequentially, and CUDA streams provide the actual parallelism. Enable with
`--tp N` (CLI) or `TENSORSHARP_TP_DEGREE=N` (server).

**Distributed TP** extends across machines via a peer-to-peer TCP mesh. Each node
runs its own process with its own local GPUs; AllReduce is hierarchical — local
P2P within each node, TCP across node representatives, then broadcast back — so
only `1/tp_local` of the data crosses the network. Enable with `--tp-node-id` and
`--tp-peers` (CLI) or `TENSORSHARP_TP_NODE_ID` and `TENSORSHARP_TP_PEERS`
(server).

Architecture-specific strategies handle heterogeneous layers:

| Architecture | Strategy |
|---|---|
| Dense transformers (Qwen 3, Mistral 3, Gemma 3) | Standard column/row-parallel QKV + FFN |
| MoE (Gemma 4, GPT OSS, Qwen 3.5/3.6, Nemotron-H) | Expert slicing — each GPU holds `1/tp` of every expert's weights; router is replicated |
| GatedDeltaNet SSM (Qwen 3.5/3.6) | Block-cyclic V-head assignment — each rank runs its own GDN kernel on its V-head subset with independent delta/conv state; no cross-rank communication for the recurrent path |
| Mamba2 SSM (Nemotron-H) | Replicated on rank 0, result broadcast to all ranks |

TP requires the `cuda` backend (GGML, MLX, and Vulkan are single-device by
design). Batched/continuous-batching forward under TP is implemented for Qwen 3
and Mistral 3; MoE models fall back to per-sequence forward under TP.

The server also supports optional **Redis-backed shared state**: a shared KV
cache tier (`--redis-url` / `TS_KV_CACHE_REDIS_URL`) for cross-session KV reuse,
and a Redis-backed Responses API store (`TS_RESPONSES_STORE_REDIS_URL`) for
durable response storage.

Full configuration reference and examples: [Usage → Tensor Parallelism & Distributed Inference](USAGE.md#tensor-parallelism--distributed-inference).

## Tool Calling / Function Calling

Models can invoke user-defined tools and participate in multi-turn tool-call conversations. Define tools as JSON and pass them via `--tools` (console) or the `tools` parameter in the API.

Each architecture uses its own wire format for tool calls:

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H:** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Gemma 4:** `<|tool_call>call:function_name{args}<tool_call|>`
- **GPT OSS (Harmony):** tools are declared as a TypeScript namespace in the developer message, and calls are emitted on the commentary channel as `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`

The output parser (`OutputParser.cs`) automatically extracts tool calls from the model's raw output regardless of architecture.

## Multimodal Support

### Gemma 4

Gemma 4 models support image, video, and audio inputs. For the E4B example above, pass the repository's `mmproj-gemma-4-E4B-it-Q8_0.gguf` explicitly with `--mmproj` (use the projector matching other target sizes).

- **Images:** PNG, JPEG, HEIC/HEIF
- **Video:** MP4 (time-based extraction at 1 fps using OpenCV; tune with `VIDEO_SAMPLE_FPS` / `VIDEO_MAX_FRAMES`)
- **Audio:** WAV (16kHz mono), MP3, OGG Vorbis

### Gemma 3

Gemma 3 supports PNG, JPEG, and HEIC/HEIF image inputs. The non-gated example above uses `mmproj-model-f16.gguf`; pass it explicitly with `--mmproj`.

### Qwen 3.5 / 3.6 family

All Qwen 3.5/3.6-family variants (`qwen35`, `qwen35moe`, and `qwen3next`) load through the same `Qwen35Model` implementation. Image inputs are supported via the dynamic-resolution `Qwen35VisionEncoder`; pass the selected repository's projector explicitly (for the 9B and Qwen 3.6 examples, `mmproj-F16.gguf`). The MoE variants (e.g. Qwen3.5-35B-A3B and Qwen3.6-35B-A3B GGUFs that report the same architecture keys) additionally enable a fused `MoEExpertsSwiGLUResidual` GGML kernel during decode that runs all selected experts, the optional shared expert, and the residual add in a single GPU graph dispatch.

### Mistral 3

Mistral 3 supports image inputs via the Pixtral vision encoder. The example repository uses `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf`; pass it explicitly with `--mmproj`.

- **Images:** PNG, JPEG, HEIC/HEIF

### Nemotron-H (Omni distribution)

The Nemotron Omni distribution adds a RADIO / v2_vl ViT image encoder. Pass the matching multimodal projector with `--mmproj` (e.g. `nvidia_Nemotron-H-Omni-mmproj.gguf`); the language-model GGUF stays the same. Image tokens are inserted at `<image>` placeholders and expanded into `<img>` + N tile tokens + `</img>` automatically by the multimodal injector.

- **Images:** PNG, JPEG, HEIC/HEIF
- **Audio:** the chat template emits `<so_embedding>` per uploaded audio file and the CLI runs the Parakeet-style log-mel preprocessor for verification, but actual audio inference requires a Parakeet audio mmproj that the public GGUFs do not currently ship.

