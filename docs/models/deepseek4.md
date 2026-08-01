# DeepSeek V4 Flash (`deepseek4`)

DeepSeek V4 Flash is a 284B-parameter MoE (256 routed experts, top-6 + 1 shared)
with a novel long-context attention stack: a tiny 128-token raw sliding window
per layer, plus per-layer *compressed* attention over 4:1 (CSA) or 128:1 (HCA)
block-compressed keys, with a lightning indexer selecting the top-512 compressed
rows on CSA layers. Residuals flow through 4-stream hyper-connections with
Sinkhorn-normalized mixing. Advertised context: 1M tokens (YaRN ×16).

## How TensorSharp runs it

DeepSeek V4 has three whole-model executors:

- **`--backend cuda`**: a **direct-CUDA whole-model engine**
  (`TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs`), independent of ggml.
  Quantized weights stream from the GGUF shards straight into per-device
  arenas and are layer-split across every visible GPU, so a model larger than
  one GPU's VRAM is hosted across several.
- **GPU backends** (`--backend ggml_cuda` / `ggml_vulkan`): the native ggml
  executor described below.
- **`--backend cpu`**: a **100% pure C# whole-model executor**
  (`TensorSharp.Models/Models/DeepSeek4/DeepSeek4CpuExecutor.cs`) — no native
  dependencies at all. It serves the quantized weights straight from the
  memory-mapped GGUF shards and runs every op (hyper-connections with Sinkhorn
  mixing, the CSA/HCA block compressors, the lightning indexer, shared-K
  attention with sinks and inverse RoPE, sqrt-softplus MoE routing with hash
  layers) in managed SIMD code.

Both TensorSharp-native executors are built on the **shared tensor stack**
rather than private re-implementations: every buffer is an `IAllocator`-backed
`Tensor` (`CudaAllocator` pool on GPU, `CpuAllocator` on CPU), the linear
layers go through the one quantized-matmul router each backend already has
(`CudaQuantizedOps.AddmmResidentToFloat32` /
`ManagedQuantizedOps.AddmmQuantizedToFloat32`), and generic math uses `Ops`
(`Ops.RMSNorm`, `Ops.SiLUMulClamp`, …). Only genuinely DSV4-specific compute —
hyper-connections, the block compressors, the lightning indexer + top-k, the
sink attention, and the grouped MoE kernels — lives in the DeepSeek V4 files.

The native whole-model executor
(`TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp`):

- Loads the (split) GGUF directly and **layer-splits the weights across every
  visible CUDA GPU** — a model larger than one GPU's VRAM is hosted across all
  of them (the 128 GiB IQ4_XS build needs 2×80GB).
- Owns all DSV4 KV state on-device: raw SWA ring, CSA/HCA compressed-K caches,
  lightning-indexer cache, and the compressor state rings.
- Executes prefill/decode ubatches as single ggml graphs via
  `ggml_backend_sched`, with flash attention (512-dim shared K=V head +
  attention sinks), fused hyper-connection ops, and a shape-signature graph
  cache so steady-state decode replays a captured CUDA graph.

The C# side (`TensorSharp.Models/Models/DeepSeek4/DeepSeek4Model.cs`) handles
GGUF metadata, the `joyai-llm` BPE pre-tokenizer, the DeepSeek V4 chat template
(`<｜User｜>…<｜Assistant｜></think>`, `--think` opens `<think>`), and sampling.

## Usage

```bash
# point --model at the FIRST shard of the split GGUF
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend ggml_cuda --chat
```

```bash
# direct-CUDA engine (no ggml): weights stream into per-GPU arenas, layer-split
# across every visible device
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend cuda --chat
```

Multi-turn chat reuses the KV cache across turns (pure append); prompts that
rewind history re-prefill automatically (compressed caches cannot truncate).

```bash
# pure C# CPU inference (no native libraries; needs enough RAM for the caches,
# weights are served from the memory-mapped shards)
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend cpu --chat
```

## DSpark speculative decoding

DeepSeek ships a **DSpark** support module alongside the model (`mtp.*` in the
checkpoint): three DSV4 blocks that read the trunk's hidden states at layers
40-42 and propose a whole BLOCK of future tokens per step, plus a Markov head
that conditions each block position on the token before it and a confidence
head that predicts each position's acceptance probability. The trunk then
verifies the block in ONE batched forward and keeps the longest prefix its own
sampler would have drawn, so speculation is a speed path, not a quality change.

It is loaded as a separate drafter GGUF with `--draft-model` and engages for
greedy (`--temperature 0`) single-sequence generation on **both GPU engines** —
`--backend cuda` (direct-CUDA) and `--backend ggml_cuda` (the native ggml
executor). `ggml_vulkan` and `cpu` have no speculative path for this
architecture and log a warning if a drafter is configured.

```bash
TensorSharp.Cli --model DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model DeepSeek-V4-Flash-0731-DSpark.gguf \
    --input prompt.txt --max-tokens 200 --temperature 0
```

The two engines implement the same algorithm differently. The **ggml** engine
builds the drafter as three extra layers of the model graph: the trunk graph
captures the target features, projects them and commits the drafter's key ring
in the same pass, and the draft itself is one cached graph whose Markov chain
(`get_rows` -> `mul_mat` -> `argmax`, per block position) runs entirely
on-device. The **direct-CUDA** engine drives the same stages with its own
kernels and feeds the drafter its target features through the host. Nothing in
the C# speculative core is backend-specific.

### Getting a drafter

The drafter is NOT in the target GGUF: every GGUF conversion of DeepSeek V4 drops
the `mtp.*` tensors (`DeepSeek-V4-Flash-0731-UD-Q8_K_XL` has `blk.0`-`blk.42` and
nothing else). Either download a pre-built drafter GGUF (three are listed in
[Model Downloads](../../MODEL_DOWNLOADS.md#dspark-drafters); publishers spell the
tensors and metadata differently and the loader accepts each spelling), or build
one from the upstream **safetensors** checkpoint.

Any DeepSeek V4 release whose `config.json` has `dspark_block_size` carries the
module: `deepseek-ai/DeepSeek-V4-Flash-0731`, `deepseek-ai/DeepSeek-V4-Flash-DSpark`,
`deepseek-ai/DeepSeek-V4-Pro-DSpark`. Only the shards holding `mtp.*` are needed
(the last three, ~11 GB of the ~340 GB repo):

```bash
REPO=deepseek-ai/DeepSeek-V4-Flash-0731
B=https://huggingface.co/$REPO/resolve/main
mkdir -p dspark-src && cd dspark-src
curl -sLO $B/config.json
curl -sLO $B/model.safetensors.index.json

# the shards holding mtp.* (model-000{46,47,48}-of-00048 for the 0731 release)
python - <<'PY' | while read f; do curl -sLO $B/$f; done
import json
wm = json.load(open("model.safetensors.index.json"))["weight_map"]
print("\n".join(sorted({v for k, v in wm.items() if k.startswith("mtp.")})))
PY
```

Then convert (the tokenizer and the other 45 shards are never opened):

```bash
python eng/dsv4-dspark-to-gguf.py --checkpoint dspark-src \
    --out DeepSeek-V4-Flash-0731-DSpark.gguf --expert-type q2_k
```

The routed experts are stored as FP4 with per-32 E8M0 scales in the checkpoint,
which is GGUF's MXFP4 layout up to nibble order, so `mxfp4` repacks them
losslessly (~11 GB); `--expert-type q2_k` roughly halves that.

**Drafter size is a real trade-off.** Its weights are re-read on every
speculative step, so a bigger drafter has to earn its bandwidth back in
acceptance. On 4xA40 the effect was clear on the direct-CUDA engine (5.6 GB
2-bit: 34.0 tok/s / 69% accepted, vs 10.9 GB MXFP4: 30.9 / 65%) and inside
run-to-run noise on ggml (31.1 / 31.5 / 31.8 tok/s for the 5.6 GB, 7 GB and
10.9 GB builds on a 120-token sample, acceptance rising 63% -> 66% -> 68% with
size). Start with a small one; move up only if acceptance is your bottleneck.

| Flag | Default | Meaning |
|---|---|---|
| `--draft-model <path>` | none | DSpark drafter GGUF (env `TS_DSV4_DSPARK`) |
| `--spec-draft-n-max <N>` | block size (5) | Cap on tokens drafted per step |
| `--spec-draft-conf-min <p>` | `0.35` | Minimum CUMULATIVE acceptance probability (the product of the confidence head's per-position estimates) for a drafted position to be kept |

`--spec-draft-conf-min` is the knob that matters: an extra verify row costs
roughly a quarter of a decode step on this sparse-MoE trunk (each row pulls in
its own set of experts), so drafting past a ~0.35 prefix-acceptance estimate is
expected-negative. Lower values draft more and roll back more; higher values
fall back to plain decode more often.

The drafter needs its three target feature layers on the device that owns the
output head; the layer split reserves room for it automatically, and loading
fails with a clear message if the split cannot satisfy that (reduce the GPU
count or drop `--draft-model`).

Measured on 4xA40 46 GB (DeepSeek-V4-Flash-0731 UD-Q8_K_XL, greedy, 200-token
generation, 5.6 GB drafter):

| Metric | `cuda` baseline | `cuda` + DSpark | `ggml_cuda` baseline | `ggml_cuda` + DSpark |
|---|---|---|---|---|
| Decode | 26.0 tok/s | **34.0 tok/s (1.31x)** | 26.4 tok/s | **37.1 tok/s (1.41x)** |
| Prefill (15K prompt) | 962 tok/s | 955 tok/s | 952 tok/s | 954 tok/s |
| Acceptance | — | 69% | — | 69% |

Multi-turn decode benefits most (2.2x on the third turn of the sample
conversation, 93% acceptance): a turn that continues an established context is
exactly where the drafter is confident.

Greedy output was byte-identical to the non-speculative baseline on both the
200-token generation and the 15K-context run. Speculation only re-orders the
arithmetic of the accepted tokens, so a long enough run can still diverge from
sequential decode wherever a batched verify row lands on a near-tie — the same
batched-vs-sequential drift that separates prefill from decode.

Why 1.3x and not more: each extra token in the verify batch pulls its own set of
routed experts through VRAM (the trunk is 6-of-256 sparse), so a verify row costs
~a quarter of a full decode step no matter how cheap the draft was. The
confidence gate is what keeps that trade positive.

| Env | Default | Meaning |
|---|---|---|
| `MAX_CONTEXT` | 65536 | Context window (caches scale with it; metadata allows 1M) |
| `TS_DSV4_UBATCH` | 512 CPU / 1024 GPU | Prefill micro-batch |
| `TS_DSV4_NGPU` | all | Number of GPUs to layer-split across (GPU backends) |
| `TS_DSV4_LOAD_THREADS` | 16 | `--backend cuda`: reader threads for the stream-to-VRAM loader |
| `TS_DSV4_LOAD_STATS` | 0 | `--backend cuda`: 1 = per-stage loader timings |
| `TS_DSV4_STAGED_EXPERTS` | 1 | `--backend cuda`: 0 = per-token expert kernels (A/B) |
| `TS_CUDA_BF16_MATVEC` | 1 | 0 = single-row BF16 projections via cuBLAS instead of the dedicated matvec (`TS_DSV4_BF16_MATVEC` also accepted) |
| `TS_DSV4_FA` | 1 | Flash attention (auto-probed, GPU backends) |
| `TS_DSV4_PERF` | 0 | 1 = tok/s log + DSpark draft phase timings, 2 = per-ubatch stage timing |
| `TS_DSV4_DSPARK` | — | DSpark drafter GGUF path (same as `--draft-model`) |
| `TS_DSV4_DSPARK_CAPTURE` | 1 | 0 = skip the drafter's target-feature capture (A/B knob; drafts then go stale and are rejected) |
| `TS_DSV4_THREADS` | all cores | CPU executor worker threads |
| `TS_DSV4_MMAP` | 1 | CPU executor: 0 = copy all weights into RAM at load (parallel reads; use when the model sits on a network filesystem) |
| `TS_DSV4_BUFFER_SHARDS` | — | CPU executor: comma-separated 1-based shard indexes to copy into RAM (mmap the rest) |
| `TS_DSV4_MLOCK` | 1 | CPU executor: best-effort mlock of mapped shards (needs a memlock rlimit) |
| `TS_DSV4_SPINPOOL` | 0 | CPU executor: 1 = persistent spinning worker pool (helps on dedicated boxes, hurts under shared CPU quotas) |

## Performance (2×A100 80GB, IQ4_XS)

| Metric | TensorSharp | llama.cpp (same box) |
|---|---|---|
| Prefill (3.3K prompt) | ~500 tok/s | 574 (pp512) / 634 (pp4096) |
| Decode @3.3K ctx | ~33 tok/s | 40.3 (tg128) |

Long-context recall (needle tests at 4.6K and 15K tokens) retrieves planted
facts exactly — the compressed-attention paths are exercised beyond the raw
128-token window.
