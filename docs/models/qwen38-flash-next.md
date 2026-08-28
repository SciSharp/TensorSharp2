# Qwen 3.8 Flash Next (`qwen4exp`)

[← back to model index](README.md)

Qwen3.8-Flash-Next is a hybrid MoE: GatedDeltaNet recurrent layers interleaved
with full-attention layers (some behind Qwen Sparse Attention's indexer), a
PLE n-gram embedding block, ×4 hyper-connection streams and a 512-expert MoE.
The GGUF architecture id is `qwen4exp`. Weights:
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
(multi-shard per quant directory; point `--model` at the `-00001-of-` shard;
`mmproj-BF16.gguf` beside the model enables image input).

## How TensorSharp runs it

On the GGML backends the whole token runs as (almost) one graph — embedding,
PLE (in-graph), all 48 layers, the final mixer and the LM head — with a
shape-keyed cache of captured graphs (`TS_Q4E_TOKEN_GRAPH=0` falls back to
per-layer fused kernels, which in turn fall back op-by-op). Vision rides the
Qwen3.5-VL tower with (T,H,W) IMRoPE positions; multi-image and multi-turn
image sessions are supported, with KV reuse across turns (the GDN recurrence
cannot rewind, so a cached prefix is reused only when the new prompt extends
it exactly).

## Continuous batching

Concurrent requests are served through **per-sequence state holders**: each
in-flight request owns its attention KV + QSA indexer caches, its GDN conv +
delta-net state, its PLE conv history and n-gram window, and its pinned kernel
descriptors. The native kernel keys its device-resident recurrent state by the
holder's host seed pointers and its cached graphs by the descriptor addresses,
so switching requests is a reference swap — no state download/upload, no graph
rebuild — and each sequence decodes through its own captured single-graph
fused decode. The engine round-robins sequences per step
(`SupportsPerSequenceFusedForward`); a fused N-way batched decode is a future
optimization.

## Multi-GPU

`--tp N` on `qwen4exp` runs a **layer split**: each GPU holds a contiguous run
of whole layers. It is not tensor parallelism — `qwen4exp` shards no weights —
and it is the same (and only) multi-GPU mode llama.cpp offers this architecture
(`-sm row` refuses to load it). It is a capacity feature, not a speed feature:
it is how you fit the model when one card cannot hold it.

Measured on 2× A100-80GB, Qwen3.8-Flash-Next-UD-Q2_K_XL (73.4 GiB):

- greedy output is **byte-identical** between the 1-GPU and the 2-GPU run
  (same SHA-256).
- VRAM 24.2 GB + 26.2 GB — roughly half the model on each card instead of all
  of it on one.
- throughput unchanged: prefill ~1520–1550 t/s and decode ~56 t/s either way.
  For reference, llama.cpp on the same box: 1 GPU pp1536 1094 / tg128 61.2;
  2 GPUs `-sm layer` 1200 / 61.5 — so llama.cpp also gains ~10% prefill and
  ~0 decode from the second card.

Startup prints which mode ran and the per-GPU layer/byte split.
`TS_Q4E_LAYER_SPLIT=20,28` overrides the automatic balance with explicit layer
counts per GPU (llama.cpp's `--tensor-split` in spirit) and throws rather than
silently ignoring a value it cannot honour — useful because the automatic
balance prices weights and cannot see the vision tower, which loads later and
lands on GPU 0.
