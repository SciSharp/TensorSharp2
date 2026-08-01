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
| `TS_DSV4_PERF` | 0 | 1 = tok/s log, 2 = per-ubatch stage timing |
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
