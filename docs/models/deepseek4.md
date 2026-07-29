# DeepSeek V4 Flash (`deepseek4`)

DeepSeek V4 Flash is a 284B-parameter MoE (256 routed experts, top-6 + 1 shared)
with a novel long-context attention stack: a tiny 128-token raw sliding window
per layer, plus per-layer *compressed* attention over 4:1 (CSA) or 128:1 (HCA)
block-compressed keys, with a lightning indexer selecting the top-512 compressed
rows on CSA layers. Residuals flow through 4-stream hyper-connections with
Sinkhorn-normalized mixing. Advertised context: 1M tokens (YaRN ×16).

## How TensorSharp runs it

Unlike the other architectures (per-op C# model classes), DeepSeek V4 runs
through a **native whole-model executor**
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

Multi-turn chat reuses the KV cache across turns (pure append); prompts that
rewind history re-prefill automatically (compressed caches cannot truncate).

| Env | Default | Meaning |
|---|---|---|
| `MAX_CONTEXT` | 65536 | Context window (caches scale with it; metadata allows 1M) |
| `TS_DSV4_UBATCH` | 512 | Prefill micro-batch |
| `TS_DSV4_NGPU` | all | Number of GPUs to layer-split across |
| `TS_DSV4_FA` | 1 | Flash attention (auto-probed) |
| `TS_DSV4_PERF` | 0 | 1 = tok/s log, 2 = per-ubatch stage timing |

## Performance (2×A100 80GB, IQ4_XS)

| Metric | TensorSharp | llama.cpp (same box) |
|---|---|---|
| Prefill (3.3K prompt) | ~500 tok/s | 574 (pp512) / 634 (pp4096) |
| Decode @3.3K ctx | ~33 tok/s | 40.3 (tg128) |

Long-context recall (needle tests at 4.6K and 15K tokens) retrieves planted
facts exactly — the compressed-attention paths are exercised beyond the raw
128-token window.
