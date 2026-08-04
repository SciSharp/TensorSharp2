# MoE CPU-offload benchmark — TensorSharp vs llama.cpp

What `--n-cpu-moe N` (llama.cpp `-ncmoe N`) costs and saves on both engines: the same GGUF files, the same host, the same benchmark shape. Three mixture-of-experts models × five offload depths (none → every layer) × two engines = 30 cells.

**Peak VRAM** is dedicated GPU memory in GiB (lower is better); **pp512** is prefill and **tg128** decode throughput in tokens/second (higher is better).

## Headline

Offload works on both engines and saves the same memory on both — a Gemma 4 26B-A4B that needs 14.4 GiB fits in 2.2 GiB with every expert on the CPU. The cost is throughput, and it is mostly paid by **prefill**, not decode.

The first run of this benchmark found TensorSharp's offloaded prefill running **4–12× slower than llama.cpp's** while decode stayed within 10%. That gap has been diagnosed and fixed in this branch; the tables below are post-fix, with the before/after in [What the gap was](#what-the-gap-was).

## Host and software

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB, driver 566.36 |
| CPU | Intel Core i7-11800H, 8 cores / 16 threads |
| RAM | 32 GiB |
| OS | Windows 11 Pro 26200 (WDDM) |
| TensorSharp | `aba3370` (branch `feature/support_moe_offload_to_cpu`), .NET 10, backend `ggml_cuda` |
| llama.cpp | `llama-bench` build 10005 (`7f575c39d`), CUDA backend |
| Date | 2026-08-04 |

## Methodology

- **Identical shape on both engines**: 512 synthetic prompt tokens, 128 decode steps, one discarded warm-up pass then three measured passes, best reported. Sampling is outside the timed loop on both sides (`llama-bench`; TensorSharp `--benchmark --bench-fixed-tokens`).
- **Best-of rather than mean.** The offloaded experts are read from the GGUF mmap on every pass, so a pass that faults pages back from disk reads ~30% low. Best-of-3 on both engines removes that; it is also why the Gemma 4 column below is not perfectly monotonic in N.
- **Matched context.** `llama-bench` fixes `n_ctx = n_prompt + n_gen = 640`; TensorSharp is pinned with `MAX_CONTEXT=768`, the nearest safe value (see [Bug found](#bug-found-non-multiple-of-256-kv-length-aborts-gemma-4-on-cuda)). The KV caches differ by tens of MiB against multi-GiB weight footprints.
- **Peak VRAM** is `nvidia-smi --query-gpu=memory.used` streamed at 200 ms for each process's lifetime, minus the desktop baseline sampled immediately before launch. Windows WDDM reports no per-process VRAM, so this is a whole-device delta with no other GPU work running; it covers weights, KV cache and compute buffers.
- **Both engines use 8 host threads** for the CPU-side expert matmul — llama.cpp's `-t` default, and what TensorSharp's `avail/2` rule picks on a 16-thread CPU. No thread-count handicap either way.
- Every cell is a cold process; the GPU is polled back to baseline between cells. Only one benchmark runs at a time.

## Summary

Baseline (nothing offloaded) versus every expert on the CPU:

| Model | Engine | Peak VRAM | Prefill | Decode |
|---|---|---|---|---|
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL) | TensorSharp | 14.37 → 2.82 GiB (−80%) | 1,782.4 → 358.8 t/s | 76.7 → 25.8 t/s |
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL) | llama.cpp | 14.08 → 2.15 GiB (−85%) | 2,090.8 → 433.9 t/s | 95.0 → 27.6 t/s |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS) | TensorSharp | 10.45 → 2.55 GiB (−76%) | 1,489.9 → 424.8 t/s | 77.1 → 22.8 t/s |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS) | llama.cpp | 10.49 → 2.08 GiB (−80%) | 1,540.5 → 583.6 t/s | 86.4 → 24.8 t/s |

Reading it: offload buys memory at a steep prefill discount and a mild decode one. Decode holds up because one token's expert matmul is small enough that the CPU keeps pace; prefill collapses on any host-only implementation because a 512-token batch is real arithmetic.

## Results by model

### Gemma 4 26B-A4B it (QAT UD-Q4_K_XL)

`gemma4-26b-a4b`, 30 layers.

| `--n-cpu-moe` | TS VRAM | TS pp512 | TS tg128 | llama VRAM | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 14.37 | 1,782.4 | 76.7 | 14.08 | 2,090.8 | 95.0 |
| 8 | 11.52 | 647.6 | 52.5 | 10.92 | 997.1 | 60.2 |
| 16 | 8.37 | 578.0 | 40.0 | 7.73 | 715.5 | 43.9 |
| 24 | 5.23 | 305.9 | 30.7 | 4.54 | 541.0 | 33.2 |
| 30 _(`--cpu-moe`)_ | 2.82 | 358.8 | 25.8 | 2.15 | 433.9 | 27.6 |

TensorSharp ÷ llama.cpp (>1.0× = TensorSharp faster; VRAM <1.0× = leaner):

| `--n-cpu-moe` | VRAM | pp512 | tg128 |
|---|---:|---:|---:|
| 0 _(baseline)_ | 1.02× | 0.85× | 0.81× |
| 8 | 1.05× | 0.65× | 0.87× |
| 16 | 1.08× | 0.81× | 0.91× |
| 24 | 1.15× | 0.57× | 0.93× |
| 30 _(`--cpu-moe`)_ | 1.31× | 0.83× | 0.94× |

### Qwen 3.6 35B-A3B (UD-IQ2_XXS)

`qwen36-35b-a3b`, 40 layers.

| `--n-cpu-moe` | TS VRAM | TS pp512 | TS tg128 | llama VRAM | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 10.45 | 1,489.9 | 77.1 | 10.49 | 1,540.5 | 86.4 |
| 10 | 8.70 | 917.7 | 52.4 | 8.43 | 1,035.8 | 57.8 |
| 20 | 6.74 | 670.4 | 38.5 | 6.34 | 821.0 | 42.2 |
| 30 | 4.64 | 525.7 | 29.1 | 4.25 | 683.9 | 31.4 |
| 40 _(`--cpu-moe`)_ | 2.55 | 424.8 | 22.8 | 2.08 | 583.6 | 24.8 |

TensorSharp ÷ llama.cpp (>1.0× = TensorSharp faster; VRAM <1.0× = leaner):

| `--n-cpu-moe` | VRAM | pp512 | tg128 |
|---|---:|---:|---:|
| 0 _(baseline)_ | 1.00× | 0.97× | 0.89× |
| 10 | 1.03× | 0.89× | 0.91× |
| 20 | 1.06× | 0.82× | 0.91× |
| 30 | 1.09× | 0.77× | 0.93× |
| 40 _(`--cpu-moe`)_ | 1.23× | 0.73× | 0.92× |

### GPT-OSS 20B (Q8_0/MXFP4)

`gpt-oss-20b`, 24 layers.

> **These cells do not measure MoE offload.** GPT-OSS 20B is slow on TensorSharp's `ggml_cuda` backend *before any offload is requested* (baseline prefill 47.8 t/s against llama.cpp's 2609, decode 3.3 against 97.5), and offload cannot repair what the baseline already loses. The model sits at ~15.8 GiB on a 16 GiB card, past the WDDM spill threshold, and the batched MoE dispatch degrades for the whole run. That is a separate, pre-existing defect; the offload path itself behaves here exactly as it does on the other two models (VRAM falls 15.8 -> 2.2 GiB). It is listed for completeness, and excluded from the summary above.

| `--n-cpu-moe` | TS VRAM | TS pp512 | TS tg128 | llama VRAM | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 15.44 | 47.8 | 3.3 | 11.30 | 2,609.4 | 97.5 |
| 6 | 15.44 | 62.0 | 9.3 | 8.96 | 1,250.7 | 57.5 |
| 12 | 13.00 | 66.7 | 30.0 | 6.59 | 907.8 | 39.4 |
| 18 | 7.60 | 48.3 | 21.2 | 4.23 | 707.6 | 29.1 |
| 24 _(`--cpu-moe`)_ | 2.15 | 35.9 | 15.3 | 1.86 | 606.1 | 22.4 |

TensorSharp ÷ llama.cpp (>1.0× = TensorSharp faster; VRAM <1.0× = leaner):

| `--n-cpu-moe` | VRAM | pp512 | tg128 |
|---|---:|---:|---:|
| 0 _(baseline)_ | 1.37× | 0.02× | 0.03× |
| 6 | 1.72× | 0.05× | 0.16× |
| 12 | 1.97× | 0.07× | 0.76× |
| 18 | 1.80× | 0.07× | 0.73× |
| 24 _(`--cpu-moe`)_ | 1.16× | 0.06× | 0.68× |

## What the gap was

On the first pass TensorSharp's offloaded prefill was 4–12× behind llama.cpp's while decode was within 10% and VRAM was equivalent. A gap that appears only in prefill, only with offload, is not a slow matmul — it is a scheduling decision.

**llama.cpp does not run offloaded experts on the CPU during prefill.** Its scheduler (`ggml_backend_sched_backend_id_from_cur`) offloads any op whose weights live in a host buffer *back to the GPU* once the batch is large enough (`ggml_backend_cuda_device_offload_op`), streaming the weights over PCIe for that one graph. `-ncmoe` therefore means "experts live in RAM", not "experts compute on the CPU": at prefill they are uploaded and multiplied on the GPU, and only at decode does the CPU do the work.

Disabling exactly that one behaviour with `-nopo 1` collapses llama.cpp onto TensorSharp's old number — same model, same `-ncmoe 8`:

| Configuration | pp512 | tg128 |
|---|---:|---:|
| llama.cpp, default (op-offload on) | 926.2 | 58.96 |
| llama.cpp, `-nopo 1` (op-offload off) | **221.0** | 58.01 |
| TensorSharp, before the fix | **219.0** | 54.70 |

TensorSharp's host matmul was never slow — it was within 1% of llama.cpp's own CPU path. It was doing the CPU work llama.cpp had stopped doing.

### The fix

An offloaded layer now picks its executor by batch size instead of always choosing the host (`TensorSharp.GGML.Native/ggml_ops_moe.cpp`):

- **Batch ≥ 128** — bind the stacked experts straight to a deferred upload, compute the existing `mul_mat_id` graph on the accelerator, and let the staging memory be reused by the next layer. The experts are never made *resident*: they land in a reused graph buffer bounded by the largest single layer, not the model.
- **Batch < 128** — unchanged; the host ggml CPU backend reads the experts zero-copy out of the GGUF mmap, exactly as before.

Two things the change had to get right:

1. **Its own graph allocator.** The seam runs *inside* a partially executed whole-model graph, and `ggml_gallocr_alloc_graph` re-plans whatever allocator it is handed. Reusing the shared one relocated the outer graph's tensors mid-pass — a hard `GGML_ASSERT(device >= 0 && device < info.device_count)`, not a wrong number. The streaming graphs get a dedicated persistent allocator.
2. **Tensor parallelism opts out.** Under `--tp N` the offloaded layer is deliberately evaluated once on the host over the *unsharded* expert stack; streaming it would put a whole layer of experts on one rank's device while the others idle. `allow_device_stream=false` there, and on the per-op entry points, which are reachable through `RunPerRank`.

**The threshold is measured, not inherited.** llama.cpp uses 32. On this host that is on the wrong side of its own crossover — the upload is a fixed cost per layer, so it only pays once the batch is big enough to amortize it (gemma-4-26B `--cpu-moe`, prefill ms, lower is better):

| batch | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| host (old behaviour) | **580** | **1081** | 2309 | 4266 |
| streamed to GPU | 1857 | 1234 | **1254** | **1329** |

Streaming loses 3.2× at batch 32 and 1.14× at 64, then wins 1.8× at 128 and 3.2× at 256. TensorSharp defaults to **128**, just past the crossover, which also keeps small block-decode batches (DiffusionGemma's denoising blocks) on the host where they belong. Override with `TS_HOST_MOE_DEVICE_MIN_BATCH`; `0` restores host-only offload.

### Result

Prefill throughput with offload, before → after, and against llama.cpp:

| Model | `--n-cpu-moe` | before | after | gain | vs llama.cpp (before → after) |
|---|---|---:|---:|---:|---|
| Gemma 4 26B-A4B it | 8 | 219.0 | 647.6 | **3.0×** | 0.23× → **0.65×** |
| Gemma 4 26B-A4B it | 16 | 115.1 | 578.0 | **5.0×** | 0.18× → **0.81×** |
| Gemma 4 26B-A4B it | 24 | 72.0 | 305.9 | **4.2×** | 0.16× → **0.57×** |
| Gemma 4 26B-A4B it | 30 _(`--cpu-moe`)_ | 57.3 | 358.8 | **6.3×** | 0.15× → **0.83×** |
| Qwen 3.6 35B-A3B | 10 | 160.9 | 917.7 | **5.7×** | 0.17× → **0.89×** |
| Qwen 3.6 35B-A3B | 20 | 85.9 | 670.4 | **7.8×** | 0.12× → **0.82×** |
| Qwen 3.6 35B-A3B | 30 | 54.1 | 525.7 | **9.7×** | 0.09× → **0.77×** |
| Qwen 3.6 35B-A3B | 40 _(`--cpu-moe`)_ | 41.0 | 424.8 | **10.4×** | 0.08× → **0.73×** |

Decode is unchanged by design (batch 1 stays on the host) and VRAM rises by the streaming buffer — 250–700 MiB, one layer's experts — against the 4–12 GiB that offload saves. Output is unaffected: the greedy correctness chain on gemma-4-26B-A4B is **token-identical** across no offload, `-ncmoe 8` and `--cpu-moe`.

## Bug found: non-multiple-of-256 KV length aborts Gemma 4 on CUDA

Unrelated to offload, hit while matching contexts. Gemma 4 26B-A4B on `ggml_cuda` with `MAX_CONTEXT=640` dies during a 512-token prefill:

```
ExternalProjects/ggml/src/ggml-cuda/fattn.cu:574: fatal error
```

`MAX_CONTEXT` 768, 1024 and 2048 all work; 640 is the only tested value that is not a multiple of 256. The mechanism is exact: Gemma 4's global layers have `key_length = 512`, and ggml-cuda's kernel selector only has a head-dim-512 flash-attention kernel behind `gqa_opt_applies`, which requires `K->ne[1] % FATTN_KQ_STRIDE == 0` with `FATTN_KQ_STRIDE = 256`. A 640-slot KV cache fails it, `ggml_cuda_get_best_fattn_kernel` returns `BEST_FATTN_KERNEL_NONE`, and `ggml_cuda_flash_attn_ext` calls `GGML_ABORT`.

The fix is to round the global-layer KV allocation up to a multiple of 256 (or reject the context length up front) rather than aborting inside a kernel dispatch. Not addressed here — it is a separate defect on a separate code path.

## Reproducing

```
# TensorSharp
MAX_CONTEXT=768 TensorSharp.Cli --model <model.gguf> --backend ggml_cuda \
    --n-cpu-moe <N> --benchmark --bench-prefill 512 --bench-decode 128 \
    --bench-runs 4 --bench-fixed-tokens

# llama.cpp
llama-bench -m <model.gguf> -p 512 -n 128 -r 3 -ncmoe <N>
```
