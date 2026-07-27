# Environment Variable x Feature Matrix

[English](env_var_feature_matrix.md) | [中文](env_var_feature_matrix_zh-cn.md)

This document is the curated runtime-flag reference used by
[`TensorSharp.TestMatrix`](../TensorSharp.TestMatrix/README.md). It focuses on
environment variables that materially change correctness, throughput, memory
use, or model routing for real inference workloads.

The code source of truth is
[`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs).
The default sweep list is configured in
[`TensorSharp.TestMatrix/Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json).

## How TestMatrix Uses This

- Every applicable `(model, backend, feature)` cell first runs a **baseline**
  case with no forced sweep variable.
- For each selected env var, the runner creates one case per listed value and
  passes only that variable to the `TensorSharp.Cli` subprocess.
- Before each subprocess starts, inherited `TS_*`, `GDN_*`, `QWEN35_*`,
  `FUSED_*`, `KV_CACHE_DTYPE`, `MAX_CONTEXT`, `MAX_TOKENS`,
  `VIDEO_MAX_FRAMES`, and `VIDEO_SAMPLE_FPS` variables are scrubbed so the
  matrix value is authoritative.
- `--env-vars none` disables sweep cases. If a config file has an empty
  `default_env_vars` list and the CLI does not override it, the runner uses all
  registered `EnvVarMatrix.All` entries.

The "Runtime baseline" column below describes the behavior when the variable is
unset. The "Swept by default" column describes the current default config, not
the full set of registered variables.

DiffusionGemma is currently outside the registered TestMatrix feature catalog:
there is no diffusion prompt type, no diffusion-specific env sweep, and inherited
`DIFFUSION_*` variables are not scrubbed by the runner. Use explicit model
configs plus a dedicated feature/env registration before treating diffusion
results as part of the standard matrix.

## Continuous Batching / Batched Forward

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_GPTOSS_BATCHED` | GPT OSS | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_QWEN35_BATCHED` | Qwen 3.5 / 3.6 family, `qwen3next` | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Qwen 3.5 / 3.6 family, `qwen3next` | Native batched GatedDeltaNet kernel | OFF | `0`, `1` | no |
| `TS_NEMOTRON_BATCHED` | Nemotron-H | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_GEMMA4_BATCHED` | Gemma 4 | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Nemotron-H | Native batched Mamba2 step | OFF | `0`, `1` | no |
| `TS_BATCHED_N1_FAST_PATH` | all | Fused N=1 fast-path decode for solo sequences; `0` forces those steps onto the fully-batched path | ON | `0`, `1` | yes |
| `TS_PER_SEQ_FUSED` | fused-capable models (Gemma 4, Qwen 3.5/3.6) | Per-request fused Forward for concurrent (N>=2) sequences; `0` forces the op-by-op batched paged path | ON | `0`, `1` | no |
| `TS_BATCHED_FUSED_DECODE` | fused-capable models | True token-batched fused decode inside the per-seq fused path (one graph for all N) | OFF | `0`, `1` | no |
| `TS_RETAINED_FUSED_CACHE` | fused-capable sliding-window models (Gemma 4) | Retain finished fused KV holders for cross-request prefix reuse | ON | `0`, `1` | no |
| `TS_RETAINED_FUSED_CACHE_MAX` | fused-capable sliding-window models | LRU budget of retained fused holders (VRAM cap) | `4` | n/a | no |
| `TS_SCHED_DISABLE_BATCHED` | all | Global per-sequence KV-swap fallback | OFF | `0`, `1` | yes |

All executor-level switches in this section are read through
`ExecutionOptions.FromEnvironment()` and consumed by `ExecutionPlanner`
(see `docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`, "Execution Planning");
per-model `TS_*_BATCHED` opt-outs surface as the model's declared
`BatchedForwardAvailable` capability.

## KV Cache / Context

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | all | KV cache element type | auto (model-aligned: `f16` when the model's weights are below F32, else `f32`) | `f32`, `f16`, `q8_0` (runtime also accepts `q4_0`, not swept) | yes |
| `TS_KV_PAGED_QUANT_BITS` | all | TurboQuant paged-KV block codec | off (`0`) | `0`, `4`, `8` | yes |
| `MAX_CONTEXT` | long text / uploaded text | Hard context cap | model default | `4096`, `8192`, `16384` | yes |

## Prefill / Decode Tuning

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | swept on GPT OSS, Qwen 3.5 / 3.6 family long-context features; honored at runtime by Gemma 4, Nemotron-H, Mistral 3, and Qwen 3 as well | Chunked prefill block size | architecture default | `256`, `512`, `1024` | yes |
| `GDN_DISABLE_CHUNKED_PREFILL` | `qwen3next` | Disable GDN chunked prefill | OFF | `0`, `1` | no |
| `TS_GGML_ASYNC_COMPUTE` | GGML backends | Async compute submission | ON on `ggml_metal` (`0` disables), OFF on other GGML backends | `0`, `1` | yes |

## Multimodal

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `VIDEO_SAMPLE_FPS` | video features | Time-based frame sampling rate | `1` | `1`, `2` | yes |
| `VIDEO_MAX_FRAMES` | video features | Upper bound on sampled video frames | no cap | `8`, `16` | yes |
| `TS_NEMOTRON_IMAGE_MAX_TILES` | Nemotron-H image features | Maximum image tiles | architecture default | `4`, `8`, `12` | yes |

## MLX-Specific

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MLX_BATCHED_MOE_DECODE` | Qwen 3.5 / 3.6 MoE on MLX | One batched dispatch per gate/up/down instead of per-expert dispatches | ON | `0`, `1` | yes |
| `TS_MLX_DEVICE_ROUTER` | Qwen 3.5 / 3.6 MoE on MLX | Device-side top-K + softmax router when prerequisites are met | ON with automatic fallback | `0`, `1` | yes |
| `TS_MLX_PIPELINED_DECODE` | MLX decode features | Pipelined greedy decode with device-side argmax where supported | ON when eligible | `0`, `1` | yes |
| `TS_MLX_DEVICE_KV_COPY` | MLX | On-device KV scatter | ON | `0`, `1` | no |
| `TS_MLX_QWEN35_GDN_PACKED_KERNELS` | Qwen 3.5 / 3.6 family on MLX | Packed GDN kernels | OFF | `0`, `1` | yes |

## Out-of-Matrix DiffusionGemma Knobs

These variables are real runtime knobs, but they are not registered in
`EnvVarMatrix.All` today and are not swept by the default TestMatrix config.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `DIFFUSION_STEPS` | DiffusionGemma Web UI | Denoising steps per block in the server path | `48` | not registered | no |
| `DIFFUSION_MAX_BATCH` | DiffusionGemma Web UI | Max active requests in `DiffusionBatchScheduler` | `2` | not registered | no |
| `DIFFUSION_BATCHED_FORWARD` | DiffusionGemma | True batched canvas decode vs time-sliced fused single-canvas decode | OFF | not registered | no |
| `DIFFUSION_NO_PKV` | DiffusionGemma | Disable prompt-KV caching on device-glue backends | OFF | not registered | no |
| `DIFFUSION_NO_SC` / `DIFFUSION_SC_TOPK` | DiffusionGemma | Self-conditioning enablement and experimental top-K cutoff | ON / `32` | not registered | no |
| `DIFFUSION_NO_FUSED_DECODE` / `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | DiffusionGemma on GGML backends | Disable fused whole-model diffusion decode or fused lm-head tail | OFF | not registered | no |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | DiffusionGemma | Transient lm-head logits memory cap before per-sequence fallback | `300` | not registered | no |
| `DIFFUSION_VRAM_HEADROOM_MB` | DiffusionGemma on ggml_cuda | VRAM kept free of preloaded weights (compute buffers, device copies) | `2048` | not registered | no |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | DiffusionGemma on ggml_cuda | Device-copy cache cap when the model does not fit VRAM (prompt K/V, masks, activations) | `768` | not registered | no |
| `DIFFUSION_SEGMENTED_DECODE` | DiffusionGemma on ggml_cuda | Force per-layer fused decode on (`1`) / off (`0`); auto-selected when the model does not fit VRAM | auto | not registered | no |
| `DIFFUSION_PIN_STREAMED` | DiffusionGemma on ggml_cuda | Re-home streamed (non-resident) weights into page-locked copies for DMA-speed uploads (costs RAM) | OFF | not registered | no |

## Out-of-Matrix MTP / Speculative-Decoding Knobs

These gate the optional MTP / NextN speculative decode path in `TensorSharp.Server`
(Qwen 3.6 embedded NextN block; Gemma 4 separate `gemma4-assistant` draft GGUF).
Speculation engages only for solo (non-concurrent) sequences and only where it is
profitable (ggml backends and the pure-C# `cuda` backend). They are not registered
in `EnvVarMatrix.All` and are not swept by the default TestMatrix config — the
matrix feature catalog has no speculative-decode feature today, so use explicit
server runs to exercise these. `TS_MTP_*` are also settable via the `--mtp-*`
server CLI flags.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MTP_SPEC` | Qwen 3.6, Gemma 4 (server) | Enable speculative decode for solo sequences | OFF (`0`) | not registered | no |
| `TS_MTP_DRAFT` | Qwen 3.6, Gemma 4 (server) | Max tokens drafted per speculative step | `8` | not registered | no |
| `TS_MTP_PMIN` | Qwen 3.6, Gemma 4 (server) | Min draft-head confidence to keep a token | `0.75` | not registered | no |
| `TS_MTP_DRAFT_MODEL` | Gemma 4 (server) | Path to the separate `gemma4-assistant` draft GGUF | none | not registered | no |
| `TS_GMTP_NO_FUSED` | Gemma 4 on ggml backends | Disable fused multi-token-verify / draft-step kernels (per-op fallback) | OFF | not registered | no |
| `TS_GMTP_NO_FAST_ROLLBACK` | Gemma 4 | Restore kept-prefix rollback instead of dense fast rollback on partial accept | OFF | not registered | no |
| `TS_GMTP_BATCHED_TRUNK` | Gemma 4 | Run the verify trunk through the batched paged path instead of the linear trunk | OFF | not registered | no |

## Out-of-Matrix Tensor Parallelism & Distributed Inference Knobs

These variables configure tensor parallelism (splitting a model across multiple
CUDA GPUs) and distributed multi-node TP over a peer-to-peer TCP mesh. They are
not registered in `EnvVarMatrix.All` and are not swept by the default TestMatrix
config — TP requires the direct `cuda` backend and multiple GPUs, which the
standard single-GPU test harness does not exercise. `TENSORSHARP_TP_DEGREE` is
also settable via the CLI's `--tp` flag; `TENSORSHARP_TP_NODE_ID` and
`TENSORSHARP_TP_PEERS` via `--tp-node-id` and `--tp-peers`.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TENSORSHARP_TP_DEGREE` | all autoregressive models, `cuda` backend | Number of local CUDA GPUs to split the model across (Megatron-LM column/row-parallel) | `1` (single GPU) | not registered | no |
| `TENSORSHARP_TP_NODE_ID` | all autoregressive models, `cuda` backend | This node's 0-based ID for multi-node distributed TP; must be set with `TENSORSHARP_TP_PEERS` | unset (disabled) | not registered | no |
| `TENSORSHARP_TP_PEERS` | all autoregressive models, `cuda` backend | Comma-separated `host:port` list of all nodes in the distributed TP cluster; must be set with `TENSORSHARP_TP_NODE_ID` | unset (disabled) | not registered | no |

## Out-of-Matrix Redis Shared-State Knobs

These variables configure optional Redis-backed shared state in
`TensorSharp.Server`: a shared KV cache tier for cross-session reuse and a
Redis-backed OpenAI Responses API store. They are not registered in
`EnvVarMatrix.All`. `TS_KV_CACHE_REDIS_URL` is also settable via `--redis-url`
or `--paged-kv-redis-url`; `TS_KV_CACHE_REDIS_TTL_MINUTES` via
`--paged-kv-redis-ttl`; `TS_RESPONSES_STORE_REDIS_URL` via `--redis-url`.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_KV_CACHE_REDIS_URL` | `TensorSharp.Server` | Redis connection string for the shared KV cache tier; when set, KV blocks are persisted to Redis for cross-session reuse | unset (disabled) | not registered | no |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | `TensorSharp.Server` | TTL in minutes for Redis KV cache entries; `0` = no TTL | `1440` (24 h) | not registered | no |
| `TS_RESPONSES_STORE_REDIS_URL` | `TensorSharp.Server` | Redis connection string for the OpenAI Responses API store; when set, `RedisResponsesStore` replaces the in-memory store | unset (disabled, in-memory) | not registered | no |

## Out-of-Matrix General Runtime Knobs

These variables are real runtime knobs, but they are not registered in
`EnvVarMatrix.All` today and are not swept by the default TestMatrix config.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_PDF_MAX_PAGES` | PDF document input (CLI `--pdf`, server `/api/upload`) | Cap on the number of PDF pages read for text extraction and page-image rendering | `0` (all pages) | not registered | no |
| `TS_FUSED_QKNORM_ROPE` | Qwen 3.5 / 3.6 text-only prefill on the direct `cuda` backend | Fused QK-Norm + NeoX-RoPE CUDA kernel; `0` falls back to separate norm + RoPE ops (multimodal MRoPE and other backends always use the separate path) | ON | not registered | no |
| `TS_CUDA_QMM_F16GEMM` | direct `cuda` backend, quantized matmuls with ≥ `TS_CUDA_QMM_F16GEMM_MIN_ROWS` activation rows | Dequantize the weight once to F16 and run a tensor-core cuBLAS GEMM (ggml-style prefill route) instead of the block-tile quant kernels; `0` reverts to the quant kernels | ON | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | direct `cuda` backend | Activation-row threshold for the F16 GEMM route | `32` | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | direct `cuda` backend | F16 weight-scratch cap in MB; weights above it (e.g. the LM head) keep the quant kernels | `768` | not registered | no |
| `TS_CUDA_Q80_VEC` | direct `cuda` backend, Q8_0 single-row (decode) matmuls | Warp-per-column dp4a matvec over a q8_1-quantized activation row (like ggml `mul_mat_vec_q`); `0` reverts to the exact FP32 dequant kernel | ON | not registered | no |
| `TS_CUDA_Q80_VEC_MIN_OUT` | direct `cuda` backend | Minimum output width for the Q8_0 dp4a matvec (diagnostic gate) | `0` | not registered | no |
| `TS_CUDA_Q80_MMQ` | direct `cuda` backend, Q8_0 matmuls with 32..`TS_CUDA_Q80_MMQ_MAX_ROWS` activation rows | Direct int8 tensor-core GEMM over raw Q8_0 blocks (mma.m16n8k32, ggml MMQ-style) instead of the dequant+cuBLAS F16 route; `0` reverts to F16 GEMM | ON | not registered | no |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | direct `cuda` backend | Row-count crossover above which the F16 GEMM route wins (MMQ weight sweeps grow as ceil(rows/128)) | `512` | not registered | no |
| `TS_CUDA_Q80_MMQ2` | direct `cuda` backend | cp.async staging variant of the MMQ GEMM (split q8_1 activation scratch + raw weight windows async-copied to shared; taken when inDim % 256 == 0, bit-identical results, ~18% faster prefill); `0` pins the register-prefetch MMQ kernel | ON | not registered | no |
| `TS_CUDA_GDN_PREFILL_SPLIT` | Qwen 3.5 / 3.6 GDN prefill on the direct `cuda` backend | 3-phase sync-free GDN prefill (parallel conv/norm → register-resident row scan → parallel RMS+gate); `0` pins the legacy single-kernel sequential walk | ON (seqLen ≥ 8, headKDim = 128) | not registered | no |
| `TS_CUDA_PREFILL_GRAPH` | Qwen 3.5 / 3.6 text-only multi-token prefill on the direct `cuda` backend | Capture the per-op prefill layer loop as a CUDA graph on the second run of a (seqLen, startPos, cache-identity) shape and replay it in one `cuGraphLaunch` afterwards (bit-identical results; falls back plainly on any capture failure); `0` disables ALL cuda graph capture, including decode graphs | ON | not registered | no |
| `TS_CUDA_DECODE_GRAPH` | Qwen 3.5 / 3.6 text-only decode on the direct `cuda` backend | Capture the per-op decode step (seqLen = 1) as a CUDA graph and replay it every token; position-dependent values (attention length, KV write slot, GDN conv ring index, RoPE position) are re-read from a pinned-host-backed device parameter block, so one graph serves all positions until the KV cache grows (bit-identical results; falls back plainly on any capture failure); `0` disables | ON | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_MAX` | direct `cuda` backend | Cached prefill + decode graphs kept (LRU-evicted; each pins its captured working-set pool blocks) | `4` | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_LOG` | direct `cuda` backend | Log graph capture/replay/abort events (`1`) | OFF | not registered | no |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | direct `cuda` backend | Budget for the global large-block (≥ 2 MB) device-memory cache; keeps prefill-sized activations pooled instead of re-issuing cuMemAlloc/cuMemFree per layer | `1024` | not registered | no |
| `TS_CUDA_PROFILE` | direct `cuda` backend | Print CPU-fallback op and host↔device sync counters at exit (`1`), with call-site attribution (`2`) | OFF | not registered | no |

## Feature Coverage

The matrix feature catalog lives in
[`TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs).
The current feature set is:

| Feature | Driver | Capability gate |
|---|---|---|
| `pp512` | `--benchmark --bench-prefill 512 --bench-decode 0` | all models |
| `pp2048` | `--benchmark --bench-prefill 2048 --bench-decode 0` | all models |
| `tg128` | `--benchmark --bench-prefill 32 --bench-decode 128` | all models |
| `short_text` | `--input prompts/short_text.txt --max-tokens 64` | all models |
| `long_text` | `--input prompts/long_text.txt --max-tokens 64` | all models |
| `uploaded_text` | `--input prompts/upload_text.txt --max-tokens 64` | all models |
| `multi_turn` | `--multi-turn-jsonl multi_turn/three_turn.jsonl` | all models |
| `tools` | `--tools tools/weather_tools.json` | models whose matrix capability says tool calling is supported |
| `thinking` | `--think` | models whose matrix capability says thinking is supported |
| `image` | `--image media/apple.png --mmproj ...` | image-capable models with an mmproj |
| `audio` | `--audio media/sample.mp3 --mmproj ...` | audio-capable models with an mmproj |
| `video` | `--video media/sample.mp4 --mmproj ...` | video-capable models with an mmproj |

Default semantic checks are intentionally weak and catch catastrophic failures:
`blue`, `paged`, `08:01:12`, `alex` + `teal`,
`get_current_weather` + `tokyo`, `10:38`, and `apple` for the relevant text,
multi-turn, tools, thinking, and image features. Audio and video have no default
expected substring because the sample media is runner-provided.

## Filters

The runner filters the combinatorial product before execution:

1. Backend availability: CUDA and Vulkan backends are skipped on macOS (Metal
   is the GPU backend there); MLX requires Apple Silicon; GGML Metal requires
   macOS.
2. Model capability: image/audio/video/tool/thinking features are skipped when
   the discovered or configured model does not advertise that capability.
3. Projector availability: multimodal features require an mmproj path.
4. Env-var applicability: each `EnvVarSpec.AppliesTo` predicate decides whether
   a variable is meaningful for the `(model, backend, feature)` cell.

## Updating The Matrix

To add a new high-impact env var:

1. Register an `EnvVarSpec` in
   [`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs).
2. Add it to `default_env_vars` in
   [`Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json)
   if it should run in the default sweep.
3. Add or update the row in this document and its Chinese counterpart.
4. If the variable changes feature applicability, update
   [`FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)
   or model discovery capability heuristics as needed.
