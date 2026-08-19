# Features
[English](FEATURES.md) | [中文](FEATURES_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation.


- **Multi-architecture support** -- DeepSeek V4 Flash, GLM 5.x, Gemma 4, Gemma 3, DiffusionGemma, Qwen 3, Qwen 3.5/3.6-family, GPT OSS, Nemotron-H, Mistral 3, Muse-Glimmer, Qwen-Image-Edit (image editing), and Wan 2.1/2.2 (text- and image-to-video)
- **Multimodal inference** -- image, video, and audio inputs (Gemma 4); images for Gemma 3 / Qwen 3.5/3.6-family / Mistral 3 / Muse-Glimmer / Nemotron-H Omni. Audio input is Gemma 4 only. `--pdf` is architecture-agnostic: a born-digital PDF's text layer is inlined into the prompt for any model, and only scanned PDFs fall back to page images (which then need a vision model)
- **Thinking / reasoning mode** -- structured chain-of-thought output with `<think>` / `<|channel>thought` / `<|channel>analysis` / `to=self` tags (Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H, Muse-Glimmer, DeepSeek V4, GLM 5.x)
- **Tool calling / function calling** -- models can invoke user-defined tools; multi-turn tool-call conversations supported across all three API styles
- **Quantized model support** -- loads GGUF files with Q4_K_M, Q8_0, F16, MXFP4, and other quantization formats; performs native quantized matmul without dequantizing to FP32, including memory-efficient pure C# CPU loading for large GGUFs
- **GPU-accelerated** -- GGML Metal on macOS, GGML CUDA on Windows/Linux with NVIDIA GPUs, GGML Vulkan on Windows/Linux with AMD/Intel/NVIDIA GPUs, a direct CUDA/cuBLAS backend with PTX kernels, and an MLX backend for Apple Silicon (mlx-c / Metal), all with CPU fallbacks for unsupported ops
- **Optimized pure C# CPU backend** -- managed GEMM fast paths plus fused SIMD kernels for RMSNorm, RoPE, softmax, fused activations, and other inference hot paths
- **Continuous batching & paged KV cache** -- vLLM-style block-paged KV pool with block-hash prefix sharing across requests, iteration-level scheduler that admits / preempts sequences mid-batch, optional SSD-backed tier for very large KV working sets, and a native fused paged-attention kernel (`TSGgml_PagedAttentionForward`) that drives `ggml_flash_attn_ext` on Metal/CUDA/Vulkan. Enabled by default in `TensorSharp.Server`; opt-out with `--no-continuous-batching`. See [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md). GLM 5.x is the exception: MLA with weight absorption (one 576-wide cache row per token per layer) and the DSA lightning indexer have no paged layout, so concurrency is served by native per-sequence **slots** instead — each request owns its MLA and indexer caches and its own `n_past`, and binding a request switches the active slot without moving KV bytes.
- **MTP / NextN speculative decoding** -- multi-token-prediction draft heads accelerate solo (non-concurrent) decode. Qwen 3.6 ships its NextN block fused into the trunk GGUF; Gemma 4 loads a separate EAGLE-style `gemma4-assistant` draft GGUF via `--mtp-draft-model` whose draft layers attend the target's own KV cache. The draft proposes up to `--mtp-draft` tokens per step (kept while draft confidence ≥ `--mtp-pmin`) and the trunk verifies them in a single batched forward; the request's own sampler — penalties included — drives both drafting and verification, so output is identical to standard decode. Opt in with the server's `--mtp-spec` flag (off by default; `TensorSharp.Cli` has no MTP flags — set the `TS_MTP_*` env vars there). On ggml backends fused multi-token-verify / draft-step kernels make it a clear win; the direct `cuda` backend runs a fully GPU-resident per-op verify/draft and is also a win. CPU / GGML CPU / MLX stay on standard decode. Env: `TS_MTP_*` (shared) and `TS_GMTP_*` (Gemma 4 tuning).
- **Batched / parallel inference** -- `IBatchedPagedModel.ForwardBatch` implementations for Mistral 3, Gemma 4, GPT OSS, Qwen 3, Qwen 3.5/3.6-family, and Nemotron-H all run by default and pack N sequences into a single forward pass with paged K/V scatter and per-sequence attention via the native kernel. Gemma 4, Qwen 3.5/3.6, GPT OSS, and Nemotron-H expose a per-family `TS_<FAMILY>_BATCHED=0` escape hatch (`TS_GEMMA4_BATCHED=0`, `TS_QWEN35_BATCHED=0`, `TS_GPTOSS_BATCHED=0`, `TS_NEMOTRON_BATCHED=0`) to fall back to the per-sequence KV-swap path for A/B comparison or regression isolation; Qwen 3 and Mistral 3 have no per-family switch — use the global `TS_SCHED_DISABLE_BATCHED=1`. GLM 5.x has no paged `ForwardBatch`; instead an opt-in batched fused decode (`TS_BATCHED_FUSED_DECODE=1`) runs one graph with one token per sequence so the weights are read once — 1.81x aggregate decode at 4 concurrent requests. It stays off by default because batching changes the GEMM shapes and a 2-bit MoE amplifies that into different expert picks.
- **Tensor parallelism & distributed inference** -- split a model across multiple GPUs (Megatron-LM column/row-parallel pattern) with `--tp N` on both `TensorSharp.Cli` and `TensorSharp.Server` (or `TENSORSHARP_TP_DEGREE`), and extend across machines with peer-to-peer TCP clustering (`--tp-node-id` / `--tp-peers`). Hierarchical AllReduce minimizes inter-node traffic. Runs on the direct `cuda` backend and on the GGML CUDA / Vulkan backends, where each rank owns a ggml backend, weight shards, and KV cache on its own GPU. Supports all autoregressive architectures (Qwen 3, Mistral 3, Gemma 3/4, Qwen 3.5/3.6-family, GPT OSS, Nemotron-H, GLM 5.x — GGML backends only, Muse-Glimmer — `--tp 2` max there, 2 KV heads) with architecture-specific strategies for MoE expert parallelism / expert slicing, GatedDeltaNet per-rank V-head ownership, and Mamba2 replication. Fused per-rank graphs make `--tp 2` decode faster than a single GPU (Gemma 4 E4B 51.7 vs 37.3 tok/s) and run models that do not fit one card. Note that TP is not the only way a model reaches several GPUs: DeepSeek V4 and GLM 5.x **layer-split across every visible GPU by default, with no flag** (their whole-model executors bin-pack whole layers against each device's free VRAM), and `--tp` switches GLM 5.x from that to Megatron sharding within each layer, while on DeepSeek V4 it only caps how many GPUs the layer split uses (the same thing `TS_DSV4_NGPU` sets). Every other architecture uses a single GPU unless `--tp` is passed. Optional Redis-backed KV cache and Responses API store for shared state. → [Tensor Parallelism](USAGE.md#tensor-parallelism--distributed-inference)
- **Ollama & OpenAI API compatibility** -- drop-in replacement endpoints for existing tooling
- **Configurable sampling** -- temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, seed, stop sequences
- **Structured outputs** -- the OpenAI `response_format` JSON schema is compiled to a grammar and enforced by grammar-constrained decoding: any token that would break the schema is removed from the distribution before sampling, so the response is structurally valid by construction rather than repaired afterwards. Supported: `type`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`, `prefixItems`, `min/maxItems`, `anyOf`, `oneOf`, `allOf`, `$ref`/`$defs` (recursive included), `min/maxLength`, `pattern`, the date/time/date-time/uuid formats, and integer `minimum`/`maximum`. Keywords a CFG cannot express (`not`, `if`/`then`/`else`, `dependentSchemas`, `dependentRequired`, `multipleOf`, `patternProperties`) are refused up front. `TS_JSON_GRAMMAR=0` falls back to prompt-and-repair.
- **Chat templates** -- auto-loaded from GGUF metadata (Jinja2), with hardcoded fallbacks per architecture
- **Inference engine** -- the new `InferenceEngine` (worker-thread scheduler + paged block pool) replaces the legacy single-request FIFO queue inside `TensorSharp.Server`. The old queue object is now a compatibility shim for status/event shapes; the engine itself handles concurrency.
- **Batch processing** -- JSONL input support in the console application, plus a built-in inference benchmark for prefill/decode throughput
- **Streaming** -- token-by-token output via SSE (web) or stdout (console), with abort/stop support for in-flight generations
- **Text-diffusion generation** -- DiffusionGemma uses an iterative EntropyBound denoising sampler instead of autoregressive `Forward()`. The CLI exposes `--diffusion-steps`, `--diffusion-seed`, and `--diffusion-blocks`; the Web UI streams whole-message `replace` events for live denoising previews and batches concurrent diffusion requests through `DiffusionBatchScheduler`.
- **Image editing (Qwen-Image-Edit)** -- a prompt plus an input image produces an edited image. The loaded `qwen_image` GGUF is the MMDiT diffusion transformer; TensorSharp resolves two companion GGUFs alongside it — the Qwen-Image VAE (image ↔ 16-channel latent) and the Qwen2.5-VL-7B text encoder (prompt → 3584-dim conditioning, optional vision grounding via an `mmproj`). The pipeline VAE-encodes the reference, builds text (and optional image) conditioning, runs a FlowMatch-Euler true-CFG denoise loop with reference-latent concatenation, then VAE-decodes back to pixels. The whole 60-block DiT forward is CUDA-graph-captured (`TSGgml_QwenImageForward`), flash-attention is on by default, and the target area is auto-clamped to the device VRAM budget. An optional Lightning distillation LoRA (`--qwen-image-lora` / `TS_QWEN_IMAGE_LORA`, `.safetensors`) cuts the denoise from the base 30 steps at CFG 2.5 to the LoRA's own step count (e.g. 4 or 8, parsed from its file name) at CFG 1.0 with no negative pass -- 60 DiT forwards become 4-8. It is applied as a runtime F32 side-path next to each targeted projection (`y = W_quant*x + b + (alpha/rank)*up*(down*x)`) with the quantized base weights left untouched, **not** merged into them: the Lightning deltas are ~1e-4 RMS, far below a Q2_K quantization step, and a measured merge changed the velocity by 24% relL2 of pure requantization noise. The side-path costs ~4% extra FLOPs, is CUDA-graph-capture-safe, and requires the whole-model or fused per-block CUDA forward -- on a path that cannot host it the model throws rather than emitting noise. The whole-step denoise caches (`TS_QWEN_DIT_CACHE_MODE`: `easycache` skips 40-55% of steps, `fbc` is First-Block-Cache) stay **off** by default because they measurably soften faces on edit workloads. Measured against stable-diffusion.cpp on the project's CUDA `image_edit` scenario (Q2_K DiT + 4-step Lightning LoRA, 544x1184, identical inputs and seed): 40.44 s vs 48.16 s warm. Driven from C# via `QwenImageModel.EditImage(prompt, RgbImage, QwenImageParams)`, from the CLI image-edit mode (`--image`, `--prompt`, `--cfg`, `--diffusion-steps`, `--diffusion-seed`), and from the Web UI with live denoising previews. → [Qwen-Image-Edit card](docs/models/qwenimage.md)
- **Video generation (Wan 2.1 text-to-video, Wan 2.2 text/image-to-video)** -- a prompt (plus an optional first-frame image on the Wan 2.2 models) produces an H.264 MP4. The loaded `wan` GGUF is the Wan DiT — Wan 2.1 T2V, Wan 2.2 TI2V-5B (48-channel 16×16×4 latent, 24 fps) and Wan 2.2 A14B (two 14B experts switched at a timestep boundary, second GGUF auto-resolved) are auto-detected; TensorSharp resolves the companions alongside it — the UMT5-XXL text encoder GGUF (prompt → 512×4096 conditioning, exact unigram-Viterbi SentencePiece tokenization) and the matching causal 3D video VAE (`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`). The FlowMatch CFG denoise (UniPC or Euler) runs the whole DiT (self-attention with 3D RoPE + flash attention over F16 keys/values, cross-attention, AdaLN time modulation — per-token-timestep for TI2V image-to-video) as ONE resident-weight ggml graph per step, CUDA-graph-captured per shape (`TSGgml_WanDitForward`); the video VAE decodes all temporal chunks in one graph with the causal feature cache carried in-graph (`TSGgml_WanVaeDecode`) -- convs go through MPSGraph on Metal (a 736x544x81f decode: 159 s -> 80 s, 1.99x, numerics unchanged at 93.9 dB PSNR; `TS_WAN_VAE_MPS_CONV=0` restores ggml's im2col+GEMM lowering) and through a banded im2col+GEMM path elsewhere, with the im2col budget and the tiling threshold now sized from free device memory instead of a fixed 16 GB card's budget, so large-memory devices decode a 720p plane whole (565 s vs 655 s banded, peak RSS 4.85 vs 5.37 GB) while small cards still tile; and image-to-video conditioning encodes the first frame through the causal VAE encoder in one graph (`TSGgml_WanVaeEncode`). Each stage releases its VRAM before the next, so TI2V-5B 81-frame 480p image-to-video and both A14B Q4_K_M experts fit a 16 GB GPU. Generation runs on every backend except MLX: the GGML paths (`ggml_cuda`, `ggml_metal`, `ggml_vulkan`, `ggml_cpu`) share the whole-graph kernels, while `--backend cuda` and `--backend cpu` run a ggml-independent direct implementation (`WanDirect*`: resident-quantized linears on TensorSharp's MMQ/dp4a/cuBLAS routing with streaming online-softmax attention kernels on CUDA, parallel SIMD GEMM/attention on CPU, and a channels-last banded-im2col causal video VAE shared by both). **Step-distilled checkpoints are auto-detected from the DiT file name** (`Turbo`, `distill`, `Lightning`, `lightx2v`, `FastWan`, `-dmd`, or an explicit `…-4steps-…`) and are by far the biggest speed lever: the official 50-step x CFG recipe costs 100 DiT passes, a 4-step distilled checkpoint costs 4, and the pipeline switches to that step count with guidance off automatically (`--diffusion-steps` / `--cfg` override). Measured on an M5 Pro at 1088x832x121f = 27 404 tokens, `ggml_metal`, Wan2.2-TI2V-5B Q8_0: the base checkpoint runs 100 passes at 120.2 s for ~3 h 30 m end to end, and the identical request on a Turbo checkpoint runs 4 passes for **17 m 30 s** -- only the `--model` path differs. On base checkpoints `--cfg-cache-stride 2` / `3` reuses the guidance direction between steps for a further 1.30x / 1.43x. Numerics verified against diffusers (DiT cosine > 0.995, VAE encoders > 0.999, decoders 59.9 dB / >35 dB PSNR) and across backends (final-latent cosine ≥ 0.999 on identical seeds); the F16 attention keys/values that make one 27 k-token self-attention 2.02x faster (~1.7x per DiT pass together with the VAE work) score the same 0.999964 DiT cosine as F32. Driven from C# via `WanVideoModel.GenerateVideo(prompt, WanVideoParams)`, the CLI (`--prompt`, `--image`, `--video-frames`, `--fps`, `--flow-shift`, `--negative-prompt`), the server API (`/v1/videos/generations` with base64 `image`, `/api/video-generate[/stream]` with `imagePath`), and the Web UI chat (type a prompt — with an attached image for image-to-video — and get the video with live progress: per-pass timings, a running ETA and a 30 s heartbeat, since one pass over a 5-second 720p latent is minutes of GPU work). → [Wan card](docs/models/wan.md)
- **Hybrid SSM-Transformer** -- Nemotron-H mixes Mamba2 SSM layers, attention-only layers, and MoE FFN layers in a single model. The Mamba2 step has both a per-sequence native kernel and a batched native kernel (`TSGgml_NemotronMamba2BatchedStepF32`, NEON SIMD + GCD parallelism) used by the batched path. On GGML backends the attention layers decode through the device-side flash-attention kernel against the resident KV cache (`TS_NEMOTRON_FLASH_DECODE=0` restores the host path), so decode no longer degrades with context length.
- **Hybrid Attention-Recurrent** -- Qwen 3.5/3.6-family models mix full-attention layers with GatedDeltaNet recurrent layers; the batched path keeps recurrent running state in a per-slot recurrent-state pool
- **Mixture of Experts** -- Gemma 4 MoE variants (e.g. gemma-4-26B-A4B), GPT OSS MoE (e.g. gpt-oss-20b), Qwen 3.5/3.6-family MoE (`qwen35moe` / `qwen3next` variants such as Qwen3.5-35B-A3B), Nemotron-H MoE FFN layers, and GLM 5.2 (744B-A40B: 256 routed experts at top-8 plus one shared expert, sigmoid-gated routing with a selection-only bias and a x2.5 routed scale, after 3 leading dense SwiGLU layers)
- **MoE CPU offload** -- `--n-cpu-moe N` / `--cpu-moe` (llama.cpp's `-ncmoe` / `-cmoe` equivalent) keeps the routed expert weights of the first N layers in system RAM and multiplies them on the host, leaving attention, the norms, the router and the always-active shared expert on the accelerator. The offloaded layers stay inside the fused whole-model graph on every architecture that has one (Qwen 3.5/3.6, Gemma 4 MoE, GPT OSS, DiffusionGemma) — the accelerator pauses after each offloaded layer's router, the host multiplies the selected experts straight out of the GGUF mmap, and the result is handed back before the next segment — so only ~8 KB of activation crosses the bus per layer at decode. Gemma 4 MoE and Qwen 3.5/3.6 segment their prefill graphs the same way, where the host side becomes a real GEMM over the whole prompt chunk. It also composes with tensor parallelism: under `--tp N` the seams merge into the ranks' own AllReduce segment schedule, so the fused multi-rank graph is kept and each offloaded layer is evaluated once on the host over the unsharded expert stack (Qwen3.5-35B-A3B `--tp 2`: 17.4 GB of resident weights across 2 GPUs falls to 3.2 GB; gemma-4-26B-A4B: 12.9 GB falls to 2.4 GB, byte-identical output). Measured on a 16 GB RTX 3080 Laptop: Qwen3.6-35B-A3B 13.4 -> 4.6 GB at `--cpu-moe`; gemma-4-26B-A4B 16.1 -> 4.8 GB (decode 39.7 -> 17.7 tok/s, and 38.6 tok/s at `--n-cpu-moe 8` for 3 GB back); gpt-oss-20b 16.2 -> 2.9 GB, which takes it off the WDDM spill cliff and turns 0.3 tok/s into 25.4 at `--n-cpu-moe 12`. That is what makes these models fit beside a long-context KV cache on a 12-16 GB GPU. DeepSeek V4 Flash uses the same seam on the GPU backends: it is 91% routed-expert bytes, and the loader sizes its layer split against each device's *actual* free VRAM. Offload stays opt-in there too -- a checkpoint that does not fit is refused at load with the exact `--n-cpu-moe N` that would make it fit, rather than silently trading away decode throughput -- and with that flag 3x48 GB RTX A6000 hosts the UD-Q8_K_XL checkpoint at 10 tok/s decode / 126 tok/s prefill. GLM 5.x offloads the same way (92% of that checkpoint is routed-expert bytes) and its host-resident experts are served straight from the GGUF mapping rather than copied. Note that offload is for *fitting*, not speed: on 3x RTX PRO 6000 where GLM-5.2 already fits, `--n-cpu-moe 30` costs pp2048 915.9 -> 94.7 and tg64 43.9 -> 16.4 tok/s. On GLM 5.2 the host-resident experts are multiplied straight out of the GGUF mapping with no private copy, and offload composes with `--tp N` — host-resident layers keep their experts whole and rank 0 evaluates them. It also nearly doubles the context the loader can size there, 342,272 -> 646,400 tokens. -> [MoE CPU offload](USAGE.md#mixture-of-experts-cpu-offload---n-cpu-moe)
- **Batched GPU MoE** -- a single fused GGML graph dispatch handles all selected experts (plus the optional shared expert and residual add) for Qwen 3.5/3.6-family and Nemotron-H decode, eliminating per-expert round-trips
- **Whole-model fused decode graphs** -- Gemma 4 (dense and MoE), Qwen 3.5/3.6 and GPT OSS run an entire decode token — every layer, the MoE router and experts, the final norm and the LM head — as ONE GGML graph dispatch instead of one submission per layer, so the GPU is never left waiting on the host between layers. On CUDA/Vulkan the graph is built once with stable tensor addresses and replayed (`ggml_set_rows` KV write with the row as an I64 input, a stride-padded attention window with an F16 mask input), which is what lets ggml-cuda capture it as a CUDA graph. GPT OSS decode: 24 → 154 tok/s on an A40, and flat in context length (133 tok/s at 16K) where the per-layer path collapsed to 2.3. Disable per model with `TS_GPTOSS_MODEL_DECODE=0` / `TS_GEMMA4_FD_PERSIST=0` / `TS_QWEN35_FD_PERSIST=0`.
- **KV cache codecs** -- pluggable codec interface (`IKvBlockCodec`) with a built-in TurboQuant (2-bit affine / Q4 / Q8) compressed codec for paged blocks. The CLI accepts all four `--paged-kv-quant-bits 0|2|4|8` values; the server's legacy standalone flag accepts `0|4|8`, while `TS_KV_PAGED_QUANT_BITS=2` selects the 2-bit codec directly. The 2-bit tier reaches ~10x compression on fp32 blocks for very long contexts.
- **KV cache precision** -- `--kv-cache-dtype <f32|f16|q8_0|q4_0>` (CLI and server, env `KV_CACHE_DTYPE`; default auto — the backend/model pick) trades a small numerical drift for memory. `q4_0` (~0.56 bytes/element, ~1/7 of f32) is the most aggressive tier and is aimed at the very long (128K–256K) contexts where the KV cache dominates memory; the block-quantized tiers (`q8_0`/`q4_0`) require the native GGML flash path.
- **Message editing** -- edit or delete previous messages in the web chat UI and regenerate from that point
- **Text/Image/Audio/Video/PDF uploads** -- the web UI accepts file uploads up to 500 MB and preserves text content in full. Born-digital PDFs have their complete text layer extracted and inlined into the prompt (cap pages explicitly with `TS_PDF_MAX_PAGES`); scanned PDFs are rendered to page images for vision-capable models. The final prompt is checked against the model's actual context window instead of an arbitrary upload budget. The CLI accepts a PDF in one-shot mode via `--pdf <file>`
- **Per-turn observability** -- structured logs capture the full user input and the full raw assistant output (both `<think>` reasoning and the final result) plus the KV cache hit ratio. The same cache-hit stats are surfaced through every API: `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio` (Ollama), `usage.prompt_tokens_details.cached_tokens` (OpenAI), and `promptTokens` / `kvReusedTokens` / `kvReusePercent` in the Web UI SSE `done` event


## Thinking / Reasoning Mode

Models that support thinking mode (Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H, Muse-Glimmer, DeepSeek V4, GLM 5.x) can produce structured chain-of-thought reasoning before generating the final answer. The thinking content is separated from the main response and can be displayed or hidden by the client.

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H:** uses `<think>...</think>` tags
- **Gemma 4:** uses `<|channel>thought\n...<channel|>` tags
- **GPT OSS:** uses Harmony format with `<|channel|>analysis` for thinking and `<|channel|>final` for the response
- **DeepSeek V4:** uses `<think>...</think>` tags; the chat template closes the block for you unless `--think` is passed, so reasoning is opt-in
- **GLM 5.x:** uses `<think>...</think>` tags, opt-in like the rest -- `--think` adds the `Reasoning Effort: Max` system line and leaves the generation prompt's block open for the model to close; without it the prompt emits an empty `<think></think>` so the model answers directly. Past turns' reasoning is always dropped from the prompt, matching the template's `clear_thinking` default

Enable via `--think` (console), `"think": true` (Ollama API), or the thinking toggle in the web UI.

## DSpark Block Speculative Decoding (DeepSeek V4)

DeepSeek V4 ships **DSpark** ("Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") as a support module in the checkpoint: three DSV4 blocks that read the trunk's hidden states and propose a whole BLOCK of tokens per step instead of one, a Markov head that conditions each block position on the token before it, and a confidence head that predicts each position's acceptance probability. TensorSharp loads it as a separate drafter GGUF (`--draft-model`, built with `eng/dsv4-dspark-to-gguf.py`) and runs it on both GPU engines (`--backend cuda` and `--backend ggml_cuda`) for greedy single-sequence generation — on ggml the drafter is three extra graph layers whose key ring the trunk graph commits itself, so speculation costs no host round trips; the trunk verifies each block in one batched forward and keeps only the prefix it would have produced anyway. Measured **1.3-1.4x decode** on 4xA40 with the drafter's cumulative-confidence gate at its default; the gate matters because each extra verify row pulls a fresh set of MoE experts through VRAM. See the [DeepSeek V4 card](docs/models/deepseek4.md#dspark-speculative-decoding).

## DFlash Block Speculative Decoding (Muse-Glimmer)

Muse-Glimmer has its own block drafter, **DFlash**: a separate 5-layer GGUF (`general.architecture = dflash`) that proposes the whole speculative window in one forward. It borrows the target's token embedding and LM head, keeps its own sliding-window KV ring, and runs three passes per step — *encode* the trunk's per-layer input residuals at `dflash.target_layers` into one wide row, *inject* that row as the K/V of every draft layer, then *draft* `[anchor, MASK x (block-1)]` through the five blocks and score it with the target's LM head. The trunk verifies the block in one batched forward and keeps only the prefix it would have produced anyway, so **the emitted token stream is the plain-greedy stream**.

Both halves are fused native graphs that CUDA-graph-capture and replay, and the draft block finishes with an on-device `argmax`, so the 202048-wide probability block never crosses PCIe. A runtime cost governor measures speculation against plain decoding and parks the drafter while it is measurably slower — speculation can therefore only help, but it needs a few hundred generated tokens to settle. Load it with `--draft-model` (CLI) or `TS_MUSE_GLIMMER_DFLASH`, and pass **no sampler flags**: `--top-k 1` fails `SamplingConfig.IsGreedy` and silently disables the whole speculative path. Measured against llama.cpp's own DFlash on one RTX PRO 6000 Blackwell (Q8_0, greedy, 60-token prompt): 50.9 tok/s vs llama.cpp's 45.5 and 35.0 plain. See the [Muse-Glimmer card](docs/models/muse-glimmer.md#3-dflash-speculative-decoding).

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
| Direct CUDA (`cuda`, driver-API/cuBLAS) | ✅ GPU-resident per-op verify/draft | ✅ GPU-resident per-op verify/draft |
| CPU / GGML CPU / MLX | standard decode (verify can't keep up) | standard decode |

Tuning: `--mtp-draft` (default `8`) bounds tokens drafted per step; `--mtp-pmin` (default `0.75`) is the minimum draft-head confidence to keep a token (drafting stops at the first low-confidence token). Gemma 4 draft-path A/B switches are the `TS_GMTP_*` env vars (see the **MTP / speculative-decoding tunables** table under [Web Application](USAGE.md#web-application)). Per-architecture mechanics are in the [Qwen 3.5/3.6 card](docs/models/qwen35.md) and the [Gemma 4 card](docs/models/gemma4.md).

## Tensor Parallelism & Distributed Inference

Tensor parallelism (TP) splits a single model across multiple GPUs using the
Megatron-LM column/row-parallel pattern. Each transformer block runs
column-parallel projections (QKV, gate/up) that split output heads or
intermediate dimensions across GPUs, independent per-GPU attention or activation
computation, and row-parallel projections (output, down) followed by an AllReduce
that reconverges the hidden state. Norms, embeddings, and the LM head are
replicated.

**Local TP** runs within a single process. On the direct `cuda` backend one
thread issues commands to all GPUs and CUDA streams provide the parallelism; on
the GGML backends a rank worker pool drives the GPUs concurrently, because a
GGML op submits *and* synchronizes in one call. Enable with `--tp N` on either
`TensorSharp.Cli` or `TensorSharp.Server` (or `TENSORSHARP_TP_DEGREE=N`);
`TENSORSHARP_TP_DEVICES=0,2` picks which physical GPUs the ranks map to.

**Distributed TP** extends across machines via a peer-to-peer TCP mesh. Each node
runs its own process with its own local GPUs; AllReduce is hierarchical — local
P2P within each node, TCP across node representatives, then broadcast back — so
only `1/tp_local` of the data crosses the network. Enable with `--tp-node-id` and
`--tp-peers` (or `TENSORSHARP_TP_NODE_ID` / `TENSORSHARP_TP_PEERS`). The server
can join such a cluster as node `0` — the driver that owns sampling and serves
HTTP — with every other node running a `TensorSharp.Cli` worker.

Architecture-specific strategies handle heterogeneous layers:

| Architecture | Strategy |
|---|---|
| Dense transformers (Qwen 3, Mistral 3, Gemma 3) | Standard column/row-parallel QKV + FFN |
| MoE (GPT OSS, Nemotron-H) | Expert slicing — each GPU holds `1/tp` of every expert's weights; router is replicated |
| MoE on GGML (Qwen 3.5/3.6) | Expert parallelism — whole experts partition across GPUs (128 of 256 per rank), so each rank keeps the single batched `ggml_mul_mat_id` dispatch per projection; the shared expert stays Megatron-split |
| MoE on GGML (Gemma 4) | Megatron split *inside* each expert (gate/up column-parallel, down row-parallel) so the fused whole-model MoE trunk kernel keeps working with global expert ids; the expert sum becomes a third row-parallel AllReduce per layer. `TS_GEMMA4_TP_FUSED_MOE=0` falls back to the whole-expert per-op path |
| GatedDeltaNet SSM (Qwen 3.5/3.6) | Block-cyclic V-head assignment — each rank runs its own packed GDN kernel on its V-head subset with independent delta/conv state, resident on its GPU; no cross-rank communication for the recurrent path |
| Mamba2 SSM (Nemotron-H) | Replicated on rank 0, result broadcast to all ranks |
| MLA + sparse-attention MoE on GGML (GLM 5.x) | Attention heads column-parallel (`attn_q_b` / `attn_k_b` / `attn_v_b`) with row-parallel `attn_output`; the 256 routed experts are Megatron-split *inside every expert* (gate/up column-parallel, down row-parallel) rather than partitioned by expert id, because `ggml_mul_mat_id` requires a token's selected expert ids to stay distinct. Router, norms, lightning indexer, shared expert and the 3 dense layers are replicated — two AllReduces per layer. `TS_GLM_TP_SHARD` picks which halves are split (1 heads, 2 experts, 3 both); `TS_GLM_TP_OVERSUBSCRIBE=1` packs several ranks onto one GPU for testing |

TP runs on the `cuda` backend and on the GGML CUDA / Vulkan backends
(`ggml_cuda`, `ggml_vulkan`); MLX is single-device. On the GGML backends each
rank owns a ggml backend on its own GPU with its own weight shards and KV
cache, and cross-GPU AllReduce goes through ggml-cuda's collective (NCCL when
available) or a host reduction for small payloads. CUDA **graph capture stays on
under TP** — a tensor-parallel token is dozens of small per-rank submissions, and
replaying them is worth ~45% of decode throughput (4×A40: Qwen 3.5-9B `--tp 4`
88 → 128.5 tok/s, Qwen 3.5-35B-A3B `--tp 2` 71.3 → 104.1, the latter being the
difference between TP losing and winning against a single GPU). Disable with
`TS_GGML_TP_CUDA_GRAPHS=0`. The collective is chosen by
measurement, not by capability flags: at startup the group verifies that peer
copies between the advertised device pairs actually deliver their bytes and that
a real NCCL AllReduce completes, and it picks the fastest transport that passes.
Hosts that advertise peer access which never arrives (common on virtualized
cloud instances) keep the NCCL collective with peer transport disabled rather
than losing it — which matters past two GPUs, where the pinned-host pipeline
does not apply and the alternative is reducing through host RAM at every layer
boundary (measured on 4×A40: 53.5 → 75.1 tok/s decode on Qwen 3.5-9B Q8_0). GGML TP delivers both
capacity and latency: fused per-rank block graphs (attention, dense FFN, MoE
trunk, GatedDeltaNet) replaced the op-at-a-time forward, so on 2× RTX 2000 Ada
`--tp 2` decodes **1.39×** a single GPU on Gemma 4 E4B Q8_0 (51.7 vs 37.3 tok/s)
and **1.06×** on Qwen 3.5-9B Q8_0, with Gemma 4 output byte-identical to the
single-GPU run. Models that do not fit one card run only under TP:
Qwen 3.5-35B-A3B IQ4_XS (16.6 GB) splits across two 16 GB cards at 184 tok/s
prefill / 18 tok/s decode. Full measurements: `TENSOR_PARALLELISM_PLAN.md`
(Stages 1b and 1c).

How far that generalizes depends on the interconnect and on how much of a layer
can be split at all. On hosts without NVLink the two AllReduces per layer
dominate: GLM-5.2 UD-IQ2_XXS on 3× RTX PRO 6000 (PCIe) runs pp2048 505.6 /
tg64 17.6 tok/s under `--tp 3` against 915.9 / 43.9 on the plain single-node
layer split, and every rank holds a full-length cache, which drops the fitted
context from 342,272 to 91,136 tokens. TP there is a capacity feature
rather than a latency one — and because
it changes the reduction order, a 2-bit MoE reproduces 3 of the 6 recorded
llama.cpp goldens where the layer split reproduces 5 of 6. (Against llama.cpp
running on the same backend, the layer split is 6/6.)

Batched/continuous-batching forward under TP is implemented for Qwen 3
and Mistral 3; MoE models fall back to per-sequence forward under TP.

Local collectives prefer CUDA peer-to-peer DMA, but the group self-tests every
peer-capable device pair at startup and permanently demotes any pair whose
round-trip comes back corrupt (seen on some L4 PCIe topologies), so hosts
without working P2P — A16 vGPU profiles, most consumer cards — fall back to host
staging automatically. Diagnostic overrides: `TENSORSHARP_TP_DISABLE_P2P=1`
(stage every cross-GPU copy through host memory) and
`TENSORSHARP_TP_HOST_ALLREDUCE=1` (run the local AllReduce on the CPU).
Multi-node connect and receive windows are tuned with
`TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` (default 120 s) and
`TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` (default 300 s).

The server also supports optional **Redis-backed shared state**: a shared KV
cache tier (`--redis-url` / `TS_KV_CACHE_REDIS_URL`) for cross-session KV reuse,
and a Redis-backed Responses API store (`TS_RESPONSES_STORE_REDIS_URL`) for
durable response storage.

Full configuration reference and examples: [Usage → Tensor Parallelism & Distributed Inference](USAGE.md#tensor-parallelism--distributed-inference).

## Tool Calling / Function Calling

Models can invoke user-defined tools and participate in multi-turn tool-call conversations. Define tools as JSON and pass them via `--tools` (console) or the `tools` parameter in the API.

Each architecture uses its own wire format for tool calls:

- **Qwen 3 / Nemotron-H:** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Qwen 3.5/3.6-family:** the same `<tool_call>` block, but with an XML body — `<function=NAME><parameter=key>value</parameter></function>` (the JSON form is still accepted)
- **Gemma 4:** `<|tool_call>call:function_name{args}<tool_call|>`
- **Muse-Glimmer:** ATEM XML — `<atem:function_calls><atem:invoke name="NAME"><atem:parameter name="key">value</atem:parameter></atem:invoke></atem:function_calls>`, routed on the assistant's `to=` recipient header
- **GPT OSS (Harmony):** tools are declared as a TypeScript namespace in the developer message, and calls are emitted on the commentary channel as `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`
- **DeepSeek V4:** DSML markup — the system prompt teaches the syntax and carries one JSON schema per function, and the model answers with `<｜DSML｜tool_calls><｜DSML｜invoke name="NAME"><｜DSML｜parameter name="key" string="true|false">value</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`. `string="false"` marks a JSON-typed argument
- **GLM 5.x:** XML with per-argument tags — `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`; the function name follows the opening tag directly as bare text, and every argument is its own `<arg_key>` / `<arg_value>` pair (values the template rendered with `tojson` are parsed back into numbers, arrays and objects)

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

### Muse-Glimmer

Muse-Glimmer-30B supports image inputs through a 50-layer sparse-window ViT with
2D RoPE and a 2x2 pixel shuffle. Pass the companion projector explicitly with
`--mmproj` (e.g. `mmproj-Muse-Glimmer-30B-Q8_0.gguf`). The chat template renders
an image content part as a single `<|patch|>`, which the multimodal injector
expands to `<|image_start|>` + N merged-patch rows + `<|image_end|>` — up to 4096
merged tokens per image, chosen by an aspect-preserving stretch (no tiling, no
padding).

- **Images:** PNG, JPEG, HEIC/HEIF

### Nemotron-H (Omni distribution)

The Nemotron Omni distribution adds a RADIO / v2_vl ViT image encoder. Pass the matching multimodal projector with `--mmproj` (e.g. `nvidia_Nemotron-H-Omni-mmproj.gguf`); the language-model GGUF stays the same. Image tokens are inserted at `<image>` placeholders and expanded into `<img>` + N tile tokens + `</img>` automatically by the multimodal injector.

- **Images:** PNG, JPEG, HEIC/HEIF
- **Audio:** the chat template emits `<so_embedding>` per uploaded audio file and the CLI runs the Parakeet-style log-mel preprocessor for verification, but actual audio inference requires a Parakeet audio mmproj that the public GGUFs do not currently ship.

