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
