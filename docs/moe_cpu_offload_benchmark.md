# MoE CPU-offload benchmark — TensorSharp vs llama.cpp

What `--n-cpu-moe N` (llama.cpp `-ncmoe N`) costs and saves on both engines: the same GGUF files, the same host, the same benchmark shape. Three mixture-of-experts models × five offload depths (none → every layer) × two engines = 30 cells, plus a 150.7 GiB DeepSeek V4 Flash across two GPUs at three depths.

**Peak VRAM** is the per-process figure in MiB (lower is better); **pp512** is prefill and **tg128** decode throughput in tokens/second (higher is better).

## Headline

Offload works on both engines and saves the same memory on both — a Gemma 4 26B-A4B that needs 14.3 GiB runs in 4.5 GiB with every expert in system RAM. What differs is what it costs:

| Model | `--n-cpu-moe` | TensorSharp pp512 | llama.cpp pp512 | TensorSharp tg128 | llama.cpp tg128 |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B | 16 of 30 | **3,148** | 1,063 | **62.1** | 21.7 |
| Gemma 4 26B-A4B | all 30 | **1,925** | 522 | **42.7** | 13.3 |
| Qwen 3.5 35B-A3B | 24 of 48 | **1,991** | 598 | **52.2** | 16.2 |
| Qwen 3.5 35B-A3B | all 48 | **1,634** | 496 | **38.3** | 10.2 |
| GPT-OSS 20B | all 24 | **676** | 570 | **34.9** | 10.4 |
| DeepSeek V4 Flash (2 GPUs) | 12 of 43 | **447** | 130 | 10.1 | **12.3** |

Offloaded prefill is **2.0–4.1× faster** than llama.cpp on the Gemma/Qwen models and **2.4–3.4×** on DeepSeek V4; offloaded decode is **2.5–3.9× faster** on the three models that use the host-MoE seam. Without offload TensorSharp is *behind* llama.cpp on this host for the small models (0.6–0.7× prefill, 0.7–0.8× decode) and ahead on DeepSeek V4 (1.25× / 1.04×) — the resident-path gap is separate, pre-existing, and untouched by this work.

## Host and software

| Component | Detail |
|---|---|
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, driver 580.126.20, PCIe 5.0 x16 |
| CPU | 2 × Intel Xeon 6952P (96 cores / 192 threads each; 384 threads total, 6 NUMA nodes), cgroup quota 81.6 CPUs |
| RAM | 1,511 GiB |
| Storage | Models on a MooseFS network mount (page-cache warm for every measured run) |
| OS | Ubuntu 24.04, CUDA 12.8 |
| TensorSharp | branch `feature/support_moe_offload_to_cpu`, .NET 10, backend `ggml_cuda` |
| llama.cpp | `llama-bench` build 4308a4f, CUDA backend, default `-t 192` |
| Date | 2026-08-05 |

## Methodology

- **Identical shape on both engines**: 512 synthetic prompt tokens, 128 decode steps. `llama-bench -p 512 -n 128 -r 3`; TensorSharp `--benchmark --bench-prefill 512 --bench-decode 128 --bench-runs 4 --bench-fixed-tokens`, which keeps sampling out of the timed loop the same way `llama-bench` does. Best-of is reported on both sides.
- **One GPU per cell** (`CUDA_VISIBLE_DEVICES=0`) except DeepSeek V4, which needs both.
- **Peak VRAM** is `nvidia-smi --query-compute-apps=pid,used_memory` sampled every 200 ms over the process's lifetime and summed across the run's PIDs — a real per-process figure, not a whole-device delta.
- **Matched context**: TensorSharp is pinned with `MAX_CONTEXT=1024`; `llama-bench` fixes `n_ctx = n_prompt + n_gen = 640`. The difference is tens of MiB of KV against multi-GiB weight footprints.
- Every cell is a cold process, run one at a time, with a 5 s settle between cells.
- Both engines were left on their own default thread counts (llama.cpp picks 192 here; TensorSharp picks 40 — see [Thread count](#thread-count-is-a-cliff-not-a-slope)).

## Results by model

### Gemma 4 26B-A4B it (UD-IQ4_XS, 30 MoE layers)

| `--n-cpu-moe` | TS VRAM (MiB) | TS pp512 | TS tg128 | llama VRAM (MiB) | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 14,264 | 7,108.6 | 167.1 | 14,246 | 10,733.5 | 206.3 |
| 8 | 11,994 | 3,892.5 | 80.4 | 11,524 | 1,906.2 | 32.8 |
| 16 | 9,312 | 3,147.7 | 62.1 | 8,772 | 1,062.8 | 21.7 |
| 24 | 6,598 | 2,147.9 | 48.7 | 6,018 | 579.2 | 16.0 |
| 30 _(`--cpu-moe`)_ | 4,568 | 1,925.2 | 42.7 | 3,784 | 522.0 | 13.3 |

TensorSharp ÷ llama.cpp (>1.0× = TensorSharp faster; VRAM >1.0× = heavier):

| `--n-cpu-moe` | VRAM | pp512 | tg128 |
|---|---:|---:|---:|
| 0 | 1.00× | 0.66× | 0.81× |
| 8 | 1.04× | **2.04×** | **2.45×** |
| 16 | 1.06× | **2.96×** | **2.86×** |
| 24 | 1.10× | **3.71×** | **3.04×** |
| 30 | 1.21× | **3.69×** | **3.21×** |

### Qwen 3.5 35B-A3B (UD-IQ4_XS, 48 MoE layers)

| `--n-cpu-moe` | TS VRAM (MiB) | TS pp512 | TS tg128 | llama VRAM (MiB) | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 18,698 | 5,832.0 | 165.4 | 17,372 | 8,182.3 | 228.8 |
| 12 | 14,928 | 3,109.0 | 81.2 | 13,132 | 1,010.1 | 27.5 |
| 24 | 10,716 | 1,990.7 | 52.2 | 8,860 | 598.4 | 16.2 |
| 36 | 6,584 | 1,737.7 | 41.1 | 4,588 | 427.5 | 10.5 |
| 48 _(`--cpu-moe`)_ | 5,192 | 1,634.4 | 38.3 | 3,164 | 496.5 | 10.2 |

TensorSharp ÷ llama.cpp:

| `--n-cpu-moe` | VRAM | pp512 | tg128 |
|---|---:|---:|---:|
| 0 | 1.08× | 0.71× | 0.72× |
| 12 | 1.14× | **3.08×** | **2.95×** |
| 24 | 1.21× | **3.33×** | **3.22×** |
| 36 | 1.44× | **4.06×** | **3.91×** |
| 48 | 1.64× | **3.29×** | **3.75×** |

### GPT-OSS 20B (Q8_0 / MXFP4, 24 MoE layers)

| `--n-cpu-moe` | TS VRAM (MiB) | TS pp512 | TS tg128 | llama VRAM (MiB) | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 24,816 | 1,030.5 | 281.7 | 12,018 | 17,141.4 | 343.8 |
| 6 | 19,676 | 840.3 | 91.8 | 9,648 | 1,540.6 | 35.2 |
| 12 | 14,180 | 691.3 | 72.2 | 7,224 | 1,104.9 | 19.0 |
| 18 | 8,658 | 636.5 | 45.1 | 4,798 | 886.7 | 13.4 |
| 24 _(`--cpu-moe`)_ | 3,074 | 676.2 | 34.9 | 2,350 | 570.4 | 10.4 |

TensorSharp ÷ llama.cpp:

| `--n-cpu-moe` | VRAM | pp512 | tg128 |
|---|---:|---:|---:|
| 0 | 2.06× | 0.06× | 0.82× |
| 6 | 2.04× | 0.55× | **2.61×** |
| 12 | 1.96× | 0.63× | **3.79×** |
| 18 | 1.80× | 0.72× | **3.37×** |
| 24 | 1.31× | **1.19×** | **3.36×** |

> **GPT-OSS carries a pre-existing, non-offload defect on `ggml_cuda`.** Its *baseline* prefill is 1,030 t/s against llama.cpp's 17,141 (0.06×) and it holds 24.8 GiB against llama.cpp's 12.0 GiB with nothing offloaded at all. Offload cannot repair what the resident path already loses: the offloaded prefill numbers above inherit that handicap, which is why GPT-OSS is the one model where TensorSharp's offloaded prefill only pulls ahead once *every* layer is offloaded (and the resident part of the model is small enough to stop mattering). Decode — which does not go through the same path — is 2.6–3.8× ahead at every depth. Tracking this separately.

### DeepSeek V4 Flash (UD-Q8_K_XL, 5 shards / 150.7 GiB, 43 layers, both GPUs)

DeepSeek V4 does not use the host-MoE seam: it hands its offloaded layers to `ggml_backend_sched`, which streams the weights back to the GPU for a prefill-sized batch exactly the way llama.cpp does. What it gained here is the **page-locking**, applied to the host-resident expert ranges at load (`[dsv4] page-locked … GiB of host experts`).

| `--n-cpu-moe` | TS VRAM (MiB) | TS pp512 | TS tg128 | llama VRAM (MiB) | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 0 _(baseline, both GPUs)_ | 156,436 | 2,778.4 | 53.8 | 155,518 | 2,222.6 | 51.5 |
| 12 | 117,212 | 447.1 | 10.1 | 117,092 | 129.9 | 12.3 |
| 24 | 77,978 | 216.1 | 5.7 | 78,898 | 88.6 | 6.2 |

| `--n-cpu-moe` | VRAM | pp512 | tg128 | pp512 before pinning | gain |
|---|---:|---:|---:|---:|---:|
| 0 | 1.01× | **1.25×** | **1.04×** | — | — |
| 12 | 1.00× | **3.44×** | 0.82× | 358.6 | **+25%** |
| 24 | 0.99× | **2.44×** | 0.91× | 153.2 | **+41%** |

Greedy output is token-identical at every depth. DeepSeek V4 is the one model where llama.cpp's offloaded *decode* is ahead (0.82–0.91×): both engines run that matmul through the same ggml CPU backend, and DSV4's per-token host read is ~260 MB (6 of 256 experts, `n_ff` 2048, `n_embd` 7168) against ~40 MB for the seam architectures, which puts it in a different part of the thread-scaling curve. That is why DSV4 keeps `available_cpu_parallelism()` workers rather than the seam's capped default.

## What changed in this pass

The starting point was the state described in the previous revision of this document: offload worked, decode was competitive, and prefill had just been moved off the host onto the accelerator with the experts streamed in for one graph. Five things were added, in the order they were found.

### 1. The host pages are locked (the big one)

A `cudaMemcpy` out of a **pageable** mapping cannot DMA — the driver copies it through its own small pinned staging buffer. The GGUF is an mmap, so every streamed expert was paying that. Measured on this host with 1 GiB transfers:

| Source | H2D bandwidth |
|---|---:|
| mmap, pageable | 9.3 GB/s |
| mmap, `cudaHostRegister` | **55.6 GB/s** |
| `cudaHostAlloc` | 55.6 GB/s |

Registering the expert ranges as they are first streamed costs ~65 ms/GiB, once, and buys the full link speed with **no extra host memory** — unlike staging through a pinned mirror, which would need a second copy of every offloaded expert. `ggml` exposes `ggml_backend_cuda_register_host_buffer` for exactly this and llama.cpp never calls it, which is most of why the numbers above look the way they do.

Guard rails, because pinned pages are unswappable: a budget (default 60% of the cgroup/host memory limit, `TS_HOST_MOE_PIN_MAX_MB`), a refusal to lock any single range larger than half of `MemAvailable`, interval bookkeeping so overlapping page-aligned ranges cannot fail registration, and `TS_HOST_MOE_PIN=0` to turn it all off. Pinning is skipped entirely when the accelerator is not CUDA (on Vulkan a `cudaHostRegister` would spin up a CUDA primary context for a device nothing is using).

`TS_HOST_MOE_TIMING=1` confirms it in situ — `pinned=10.2 GiB … experts=53/128 bytes=3095 MiB` for gemma-4-26B at `--cpu-moe`. The rate that line prints (~20 GB/s) is *not* the link rate: its window ends at the first synchronizing call, so it also contains the outer graph's own compute. The 55.6 GB/s figure is the isolated transfer.

### 2. Only the experts the batch actually routes to are streamed

llama.cpp's scheduler builds a used-expert bitset from the ids tensor and copies consecutive runs; TensorSharp now does the same, with the same trailing pad (MMQ reads a tile past the last row it needs, and whatever the reused graph buffer happened to hold there could decode to NaN). On a 128-expert model a 512-token batch typically touches a fraction of the pool, and at the small batches speculative verification and light serving produce the saving is larger still. `TS_HOST_MOE_EXPERT_FILTER=0` restores whole-stack uploads.

### 3. The copies are asynchronous

The uploads now go through `ggml_backend_tensor_set_async` onto the backend's own stream and synchronize once, with the graph, instead of three times per offloaded layer.

Effect of 1–3 together, gemma-4-26B, pp512 (best of 5):

| `--n-cpu-moe` | before | after | gain | vs llama.cpp |
|---|---:|---:|---:|---|
| 8 | 1,110 | 3,616 | **3.3×** | 0.89× → **2.04×** |
| 16 | 713 | 2,906 | **4.1×** | 0.95× → **2.96×** |
| 30 | 315 | 1,970 | **6.3×** | 0.67× → **3.69×** |

Greedy output is **token-identical** across `--n-cpu-moe` 0 / 8 / 16 / 30 (`prefillTopToken=102`, then `103,104,105,108,109,…` in every configuration).

### 4. DeepSeek V4's host experts are locked at load

DSV4 never used the seam — it routes offloaded layers through `ggml_backend_sched`, which streams them back to the GPU for a prefill batch on its own. Registering those ranges once at load (rather than never) is worth **+25% at `-ncmoe 12` and +41% at `-ncmoe 24`** on a 150.7 GiB checkpoint, and costs one line of load-time output.

### 5. Architectures without a fused prefill graph now stream too

GPT-OSS and Nemotron-H have no whole-model *prefill* graph, so their offloaded layers go through the per-layer MoE op rather than the seam. That entry point was hard-wired to the host path because it is reachable through `RunPerRank` under tensor parallelism, where N ranks would each stream their own copy of the same unsharded experts. The guard is now the actual condition (`device_count <= 1`) instead of a blanket refusal:

| GPT-OSS 20B `--n-cpu-moe` | before | after | gain |
|---|---:|---:|---:|
| 6 | 332 | 789 | 2.4× |
| 12 | 197 | 853 | 4.3× |
| 24 (`--cpu-moe`) | 49 | 640 | **13.1×** |

### Thread count is a cliff, not a slope

The host-side matmul at decode is one token wide: a few MB of expert rows read from RAM with a barrier after every ggml op. Past a few dozen workers each extra thread only adds a barrier participant, and on a 2-socket machine it starts adding cross-socket traffic. Measured here (gemma-4-26B `--cpu-moe`, ms of host matmul per 30 offloaded layers):

| threads | 8 | 16 | 32 | 64 | 96 | 192 |
|---|---:|---:|---:|---:|---:|---:|
| ms | 45 | 35 | **17.6** | **17.3** | 26 | 120 |

The old default (`usable_cpus / 2`) picks 192 on a bare-metal host like this one and pays 7×; it only ever looked safe because every previously tested box had a cgroup quota that clamped it into the flat part of the curve. The default is now `min(usable/2, 64)`, overridable with `--cpu-moe-threads`. (This host's own 81.6-CPU quota means the measured runs above used 40 either way.)

## Correctness sweep

Offload is only worth measuring if the answers stay right. Fourteen server configurations (`TensorSharp.Server`, OpenAI-compatible endpoint) × six capabilities, on the three GPU backends, with and without offload:

| Configuration | short text | long text | multi-turn | tools | JSON mode | image |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| gemma4-26B `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_cuda` `-ncmoe 16` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_cuda` `--cpu-moe` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_cuda` `--tp 2 -ncmoe 8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_vulkan` `-ncmoe 16` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| gemma4-E4B `ggml_cuda` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-35B `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| qwen35-35B `ggml_cuda` `-ncmoe 20` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| qwen35-35B `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| qwen35-9B `ggml_cuda` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| qwen35-9B `cuda` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| gptoss-20B `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ❌ | — |
| gptoss-20B `ggml_cuda` `-ncmoe 12` | ✅ | ✅ | ✅ | ✅ | ❌ | — |

77 of 82 checks pass. **Every failure reproduces identically with and without offload**, so none of them is an offload regression:

- **Qwen 3.5 image (5 cells).** The model misreads the test image's text on `ggml_cuda`, `ggml_vulkan` and `cuda` alike. llama.cpp's `llama-mtmd-cli` on the same GGUF + mmproj + image returns the **same wrong answer, token for token** (`areplay` for the 9B) — TensorSharp matches the reference implementation exactly, and the limitation is the model/projector pair on this synthetic input, not the engine.
- **GPT-OSS JSON mode (2 cells).** Structured output is schema-valid but semantically empty (`{"city":"..??..","population":0}`). The other four models answer correctly under the same schema, so this is a GPT-OSS-specific interaction between grammar-constrained decoding and its channel format. Pre-existing, tracked separately.
- **Gemma 4 26B image on the pure-C# `cuda` backend (1 cell).** The reply echoes the prompt instead of describing the image. Text features on that backend are fine, and the same model+image works on both GGML backends. Pre-existing, tracked separately.

Beyond the pass/fail checks, the greedy benchmark chain on gemma-4-26B is **token-identical** across `--n-cpu-moe` 0 / 8 / 16 / 24 / 30.

The same sweep on the **CLI** surface, through `TensorSharp.TestMatrix` (5 models × 3 backends × {short text, image} × `TS_N_CPU_MOE` ∈ {baseline, 0, 16, all}), is **52 of 57 cells green** — the five reds are the same Qwen 3.5 image cells. Every offload cell passes on every backend, which is what the new `TS_N_CPU_MOE` entry in `EnvVarMatrix` now keeps sweeping.

**Speculative decoding** (the only drafter pair on this host, DeepSeek V4 Flash + its DSpark drafter, both GPUs, 128 tokens): 53.0 → **74.5 tok/s (1.41×)**, acceptance 0.658 over a 5-token window, 51 verify steps and 26 rollbacks. Unaffected by this work, checked because the offload path shares the verify batch shapes.

## What did not work

**A device-resident seam.** With the experts already being streamed and the matmul already running on the accelerator, the seam still stages its activations through the host: the whole-model graph's `moe_in` / ids / routing weights are downloaded and the result uploaded, four transfers and four synchronizes per offloaded layer. Rewriting the seam graph to read and write the outer graph's own tensors removed all of that and was worth ~25% more prefill — and produced wrong output. The offloaded layer's `moe_in` **changes value across the seam call** (330.4 → 248.2 at layer 0), i.e. the nested graph's allocator writes over the outer graph's buffers; the corruption disappears with `TS_GGML_REUSE_COMPUTE_BUF=0`, which points at the persistent gallocr rather than at ggml-alloc's view handling. Reverted rather than shipped. The ~135 ms/forward it would save on a 30-layer full offload is the largest single item left on the table.

## Reproducing

```bash
# TensorSharp
MAX_CONTEXT=1024 TensorSharp.Cli --model <model.gguf> --backend ggml_cuda \
    --n-cpu-moe <N> --benchmark --bench-prefill 512 --bench-decode 128 \
    --bench-runs 4 --bench-fixed-tokens

# llama.cpp
llama-bench -m <model.gguf> -p 512 -n 128 -r 3 -ncmoe <N>
```

Diagnostics worth knowing: `TS_HOST_MOE_TIMING=1` splits an offloaded layer's wall clock into setup vs host matmul, and prints streamed bytes, used experts and effective bandwidth for the accelerator path; `=2` adds a per-weight transfer rate (and a synchronize, so it changes what it measures). `TS_HOST_MOE_DEBUG=1` prints the segment plan and each seam's activation norms. `TS_HOST_MOE_VERIFY=1` runs the on-GPU expert chain alongside the host one and reports the divergence.
