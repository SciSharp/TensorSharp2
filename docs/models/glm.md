# GLM-5.x (`glm-dsa`)

[← back to model index](README.md)

GLM-5.2 is a 744B-parameter MoE (256 routed experts, top-8, plus one shared
expert) built on **DeepSeek Sparse Attention**: Multi-head Latent Attention with
weight absorption, and a "lightning indexer" that decides which cached tokens
each query may attend to. Advertised context: 1M tokens. The GGUF architecture
id is `glm-dsa`.

## The block

| Piece | Shape (GLM-5.2) | Notes |
|---|---|---|
| Attention | MLA, 64 query heads, 1 key head | `q_lora_rank` 2048, `kv_lora_rank` 512, `n_embd_head_k/v_mla` 256, `n_rot` 64 |
| KV cache | **one 576-wide row per token per layer** | `kv_lora_rank + n_rot`; the per-head K/V decompression is folded into the query (`attn_k_b`) and the output (`attn_v_b`) |
| DSA indexer | 32 heads x 128, top-k 2048 | scores every cached token; only the top-k survive into the attention mask |
| Indexer layers | 21 of 78 | layers 0,1,2 then every 4th from 6; the layers in between reuse the last full layer's selection |
| MoE | 256 experts, top-8, `n_ff_exp` 2048 | sigmoid gating, a routing bias for SELECTION only, weight renormalisation, x2.5 routed scale |
| Dense layers | first 3 | plain SwiGLU with `feed_forward_length` 12288 |
| NextN / MTP | 1 trailing block | `block_count` includes it; the trunk graph runs `block_count - nextn_predict_layers` layers. Drives [speculative decoding](#nextn--mtp-speculative-decoding) under `--mtp-spec`; unloaded otherwise |
| RoPE | NORM, base 8e6, no YaRN | applied to the 64-wide rope slice of Q and to the single K rope tail |

Attention softmax scale is `1/sqrt(n_embd_head_k_mla)` = 1/16 — the *decompressed*
head size, not the 576-wide cache row.

## How TensorSharp runs it

Two implementations, both reproducing llama.cpp's `src/models/glm-dsa.cpp`:

- **GGML backends (`--backend ggml_cuda` / `ggml_vulkan` / `ggml_cpu` / `ggml_metal`)**
  run the **native whole-model executor**
  (`TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp`). It loads the split GGUF
  itself, layer-splits the weights across every visible GPU (226 GiB does not
  fit one card), owns the MLA and indexer caches on-device, and submits ONE
  ggml graph per ubatch through `ggml_backend_sched`, with a shape-keyed LRU
  graph cache so steady-state decode replays one allocated (and, on CUDA,
  captured) graph.
- **`--backend cpu` (100% managed) and `--backend cuda`** run the per-op path in
  `TensorSharp.Models/Models/GlmDsa/*`, built on the shared `Ops` /
  `ManagedQuantizedOps` / `CudaQuantizedOps` stack. This is also the reference
  implementation the native executor is checked against; `TS_GLM_NATIVE=0`
  selects it on a GGML backend for an A/B.

Split GGUFs are handled by `GgufFile` itself (`split.count` / `-00001-of-000NN`),
so every model in the repo can now be stored across several files.

### Tensor parallelism

**Without `--tp`, a multi-GPU box already uses every card**: the loader measures
each device's free VRAM and bin-packs the 78 layers across them, so device 0 runs
layers 0..k, hands the hidden state on, and so on. That is the default and there
is no flag for it — a 226 GiB checkpoint fits no other way. `TS_GLM_NGPU` caps how
many devices it uses, and if even the full set is not enough the load is refused
with the exact `--n-cpu-moe N` that would make it fit.

`--tp N` (or `TENSORSHARP_TP_DEGREE`) is the other mode: it runs every layer on
every one of N GPUs and splits the weights *inside* each layer, so decode reads
1/N of the weights per device instead of walking all of them in sequence. The split follows the
Megatron column/row pattern used by the rest of the repo:

| Piece | Split | Collective |
|---|---|---|
| Attention heads | column-parallel `attn_q_b` / `attn_k_b` / `attn_v_b`, row-parallel `attn_output` | one all-reduce per layer |
| Routed experts | column-parallel `ffn_gate_exps` / `ffn_up_exps`, row-parallel `ffn_down_exps` — **every expert is split row-wise, the experts are not divided between ranks** | one all-reduce per layer |
| Router, norms, indexer, shared expert, dense layers | replicated | none |
| MLA + indexer caches | **replicated — every rank keeps its own full-length copy.** The 576-wide MLA row is shared by all heads, so there is nothing head-shaped to shard; this is why `--tp N` multiplies the KV footprint by N and drops the fitted context | none |

Splitting the expert *hidden* dimension rather than the expert *ids* is what
makes this work at all: `ggml_mul_mat_id` needs a token's selected expert ids to
be distinct, so an id-space split would have to invent a duplicate id for every
expert a rank does not own. Row-splitting keeps the router's global top-8 valid
on every rank, gives every rank an equal 1/N of the work no matter how the
routing skews, and cuts each rank's expert FLOPs by N as well as its memory.

A rank's strip of the down-projection cuts each ROW, so it is measured in whole
quantization blocks; a model whose expert hidden size is not N whole blocks
keeps its experts intact and splits only the heads (still exact, just slower).
`TS_GLM_TP_SHARD` selects the halves independently (1 = heads, 2 = experts,
3 = both) and `TS_GLM_TP_OVERSUBSCRIBE=1` lets several ranks share one GPU,
which is how the split is checked for correctness on a single-GPU machine.

### Serving concurrent requests

The native executor keeps every cache on the device, so a request's state is a
native **slot** — a full set of per-layer MLA and indexer caches plus its own
`n_past` — and binding a request to one is an active-slot switch that moves no
KV bytes (`GlmDsaModel.PerSeqCache.cs`, the same contract DeepSeek V4 uses).
Each slot's graphs are cached and captured independently, so concurrent
requests replay their own captured CUDA graphs instead of rebuilding or
replaying another request's baked cache addresses.

The token-batched *paged* path is deliberately not implemented: MLA stores one
compressed row per token and the DSA indexer scores against that same
contiguous history, so there is no paged-KV layout to batch over. What is
implemented instead is a **batched fused decode**: one graph, one token from
each of N sequences, where every projection, the router, the experts and the LM
head run once over the batch and only the cache write, the indexer scoring and
the softmax are built per token — so N concurrent requests read the weights once
between them instead of N times. Measured on the 3-GPU box, four concurrent
200-token completions: **75.2 tok/s aggregate against 41.6 solo** (1.81x), each
stream at 18.8 tok/s.

`TS_BATCHED_FUSED_DECODE=1` turns it on, and it is off by default here for the
same reason it is off everywhere else in the project. Batching changes the shape
of every GEMM, so CUDA picks different kernels and the result differs in the last
bits: the first divergence against the one-at-a-time path shows up at layer 1 at
2e-8 relative. In a dense model that would stay invisible, but 75 of these 78
layers pick 8 of 256 experts by a top-k over near-tied scores, so a last-bit
difference flips a marginal expert, and by the LM head the logits differ by
O(1) — on a 2-bit checkpoint that is a visibly different continuation, not a
rounding wobble. On the CPU backend, whose kernels do not switch on batch size,
batched and serial decode are bitwise identical.

Without the flag, concurrency still works and is exact: the engine interleaves
whole-graph per-sequence forwards, and four concurrent completions come back
byte-identical to running the same four prompts one after another. It just
re-reads the weights once per sequence.

### The sparse-attention path

Below `attention.indexer.top_k` cached tokens the indexer cannot remove
anything — top-k over `n_kv <= k` keeps every cell — so TensorSharp skips the
scoring entirely and attends densely. That is the same function, computed
cheaply, and it is why short prompts pay nothing for DSA. The indexer KEYS are
still cached on those steps, because a later, longer step scores them.

Past top-k the graph builds the full indexer: rope, the Walsh-Hadamard rotation,
`ggml_lightning_indexer` (or an equivalent decomposition where the backend has
no kernel), `ggml_top_k`, and a mask that starts fully masked and is unmasked at
the selected positions before the causal mask is added back.

**The Hadamard rotation is reproduced deliberately.** It is an orthonormal
involution applied to both sides of a dot product, so in exact arithmetic it
cancels — but the indexer key cache is F16, and rotating before rounding spreads
the error evenly across the 128 dimensions. Skipping it changed which tokens the
top-k picked on a 2741-token prompt and broke token-for-token parity with
llama.cpp; reproducing it restored 6/6.

## NextN / MTP speculative decoding

GLM-5.2 ships a **NextN block** in the stock checkpoint — `block_count` is 79
and `nextn_predict_layers` is 1, so `blk.78` is a complete glm-dsa decoder block
(MLA attention, the 256-expert sigmoid-gated MoE with its shared expert) wrapped
in the deepseek-family NextN wiring:

```
h_mtp  = shared_head_norm( block( eh_proj( [ enorm(embed(t)) ; hnorm(h) ] ) ) )
logits = lm_head(h_mtp)
```

where `h` is the trunk's **post-`output_norm`** hidden state of the token before
`t`. The block predicts token *t+1* from token *t*, so chaining it drafts a
window that the trunk then verifies in a single batched forward. Enable it with
`--mtp-spec` on either host; there is nothing to download.

```bash
# CLI — single-shot, chat REPL, or a multi-turn JSONL run
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf \
    --backend ggml_cuda --n-cpu-moe 20 --mtp-spec --chat

# Server
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
    --model models/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf \
    --backend ggml_cuda --n-cpu-moe 20 --mtp-spec
```

`--mtp-draft N` and `--mtp-pmin X` tune the window and the confidence gate on
both hosts (see [Measured](#measured)). Verification draws every emitted token
from a trunk row with whatever sampler the run configured — argmax under
`--temperature 0`, the chat sampler under `--chat` — so speculation never
changes which distribution a token comes from, only how many forward passes it
took to get there.

Three details are worth stating, because getting any of them wrong is silent:

- **The draft block attends densely.** It ships lightning-indexer weights
  (`blk.78.indexer.*`), but llama.cpp's `graph_mtp` builds the plain MLA
  attention input and never reads them — and the block has no indexer key cache
  to score against anyway. Running the indexer there would be a different model,
  not a faster one.
- **It borrows the trunk's embedding table and LM head.** GLM-5.2 ships neither
  `nextn.embed_tokens` nor `nextn.shared_head_head`. Both are optional in
  llama.cpp too, and both are honoured when present. The borrowed head is why
  drafting is refused under `--tp N > 1`: the trunk head is column-parallel
  there, so the draft would read one rank's strip of the vocabulary.
- **The draft block's KV cache never needs a rollback.** A catch-up always
  rewrites from the verified position forward, so rejected speculative rows are
  overwritten before anything reads them. And because glm-dsa has no recurrent
  state, a partially-rejected verify keeps the accepted prefix's KV in the trunk
  and only rewinds the position — no kept-prefix re-forward, which is the
  dominant rollback cost on a long context.

**Opt-in for a reason.** The block is a whole extra decoder layer — ~3 GiB at
IQ2_XXS — competing for the VRAM the loader sizes the context against, so the
native loader only pages it in when `--mtp-spec` (`TS_MTP_SPEC`) was set before
the model loaded. That is why the flag has to be on the command line rather than
toggled later, and why adding it to a command that already just fit can shorten
the context the loader settles on. `TS_GLM_MTP=1` / `0` overrides either way for
an A/B.

### Measured

2× RTX PRO 6000 Blackwell (97 GiB each), GLM-5.2-UD-IQ2_XXS, `--n-cpu-moe 20`,
21-token prompt, 160 tokens generated, greedy
(`InferenceWeb.Tests/GlmDsaMtpModelTests.cs`). Each run takes the plain baseline
as the median of three, because the host-side expert matmul is the noisy half of
the comparison — and the whole benchmark was then repeated five times, because
one round is not enough to tell a 5% tuning effect from that noise.

| Configuration | Decode (5 runs) | vs plain | Draft acceptance | Drafted per verify |
|---|---|---|---|---|
| plain greedy | 17.96 / 18.33 / 20.42 / 20.37 / 18.56 tok/s | 1.00x | — | — |
| `--mtp-spec` (the defaults: k=8, pMin 0.75) | 20.50 / 25.68 / 25.83 / 25.85 / 23.52 tok/s | 1.14 / 1.40 / 1.27 / 1.27 / 1.27x — **median 1.27x** | 93.8% | 1.59 |
| `--mtp-spec --mtp-draft 4 --mtp-pmin 0.55` | 22.35 / 26.99 / 25.86 / 26.81 / 25.89 tok/s | 1.24 / 1.47 / 1.27 / 1.32 / 1.39x — median 1.32x | 75.0% | 2.04 |

**On tuning.** A narrower window with a lower gate was best or tied-best in
every run (and in a `--n-cpu-moe 34` variant), by ~4% on average — but never by
enough, and never consistently enough, to be worth hard-coding as a per-model
default: sweeping the gate alone at k=8 won three runs and lost three. The two
knobs interact, so sweep them together rather than one at a time. The runtime
cost governor in `MtpSpeculativeExecution` measures the model/drafter pair
either way and parks drafting if it stops paying.

Why a narrower window helps at all: a verify amortizes unusually well here — the
trunk reads its routed experts once for the whole window —

| Verify rows | Cost | vs a 1-row decode | Per token |
|---|---|---|---|
| 1 | 95.6 ms | 1.00x | 1.00x |
| 2 | 121.4 ms | 1.27x | 0.64x |
| 3 | 147.5 ms | 1.54x | 0.51x |
| 5 | 190.8 ms | 2.00x | 0.40x |
| 9 | 285.2 ms | 2.98x | 0.33x |

so an extra speculative row costs about a quarter of a decode step and is worth
taking well below 50% expected acceptance. What the 0.75 gate does instead is
cut the chain after one token (1.59 drafted per verify); lowering it lengthens
the chain, and capping the window bounds what a chain that then gets rejected
costs. Neither alone is reliably better than the defaults — together they are.

The break-even also moves with how much of the MoE is offloaded, so it is worth
knowing which side of it your host is on. At `--n-cpu-moe 34` (84.4 GiB of
experts on the host):

| Configuration | Decode | vs plain |
|---|---|---|
| plain greedy | 12.57 tok/s | 1.00x |
| the defaults | 14.57 tok/s | 1.16x |
| `--mtp-draft 4 --mtp-pmin 0.55` | 14.78 tok/s | 1.18x |

Heavier offload slows the 1-row baseline more than it slows a wide verify, so
the curve flattens (a 2-row verify is 1.16x a 1-row decode there, not 1.27x) and
the configurations converge.

### Greedy output and floating point

Every emitted token is drawn from a **trunk** row, so speculation cannot change
which distribution a token comes from — only how many forward passes it took to
get there. It does change the *arithmetic*: a K+1-row verify runs the trunk's
matmuls at a different batch size than a 1-row decode, which selects different
kernels and reduction orders.

On GLM-5.2 that is not invisible. At 2-bit with 256 experts at top-8, a last-bit
difference in a router logit changes *which experts run*, and 78 layers amplify
it. Measured over 140 verify rows against per-token decode: the top token
differs on **2.9%** of rows (max |Δlogit| 2.6), so a long greedy run eventually
takes a different — equally valid — branch. This is the same effect the
[tensor-parallelism section](#tensor-parallelism) describes for `--tp`, and the
tests pin down where it does *not* come from:

- capturing the hidden state is free: an h-capturing forward at one token is
  **bit-identical** to a plain forward (max |Δ| exactly 0);
- driving the whole speculative loop with drafting suppressed — same cache
  bookkeeping, same catch-up calls, same rewinds — reproduces greedy **exactly**.

So the effect is the batch size, not the speculation. If you need a run to match
non-speculative greedy token for token, leave `--mtp-spec` off.

## Numerical parity

Measured on GLM-5.2-UD-IQ2_XXS (226 GiB) against `llama.cpp b200-9731ad3` on the
same machine, feeding the RECORDED prompt token ids so the comparison isolates
the forward pass from tokenization (`.parity/gen_ref_glm.py`,
`InferenceWeb.Tests/GlmDsaParityTests.cs`):

| backend | prompts reproducing llama.cpp token-for-token |
|---|---|
| `ggml_cuda`, 3 GPUs | **6/6** against llama.cpp on the same backend (5 short + one 2741-token prompt through the sparse path) |
| `ggml_cpu` | 3/3 |
| `cpu` (100% managed) | 1/1 |

**On the same backend** is the bar, and it has to be: on the 2741-token prompt
llama.cpp's own CPU and CUDA builds disagree at the fifth generated token
(`1467` vs `8543`, "The ... is a summary" either way), and TensorSharp
reproduces whichever one it is running on — CPU's answer on `ggml_cpu`, CUDA's
on `ggml_cuda`. The recorded golden was captured from a CPU llama-server, so
that one record reads as a mismatch when the check runs on GPU. The DSA
selection is what makes the prompt this sensitive: the indexer key cache is F16,
and at 2741 tokens two candidates in the top-2048 are close enough that a
last-bit difference in the score reorders them.

The tensor-parallel and batched-decode splits are checked the same way but on
the synthetic model, where they can be held to a much harder standard: on the
CPU backend `--tp 1..4` and batched decode are **bitwise identical** to the
single-rank, one-at-a-time path, and on CUDA all seven head/expert split
combinations reproduce the golden continuation token for token
(`--batched` and `TS_GLM_TP_SHARD` in the parity harness).

`InferenceWeb.Tests/GlmDsaTinyModelTests.cs` covers the architecture without the
226 GiB download: it builds a deterministic 1.9 MB `glm-dsa` GGUF with the same
block shape (and `top_k` 8, so a 24-token prompt is already sparse) and checks
the greedy continuation against goldens captured from llama.cpp on that file.

## Performance

3x RTX PRO 6000 Blackwell (97 GiB each), GLM-5.2-UD-IQ2_XXS, layer split, both
engines measured back to back in one session (`llama-bench` for llama.cpp, and
the parity harness's `--bench`, which reports the best of two repetitions the
same way):

| test | llama.cpp | TensorSharp (default `n_ubatch` 1024) | TensorSharp (`TS_GLM_UBATCH=2048`) |
|---|---:|---:|---:|
| pp128 | **276.5 t/s** | 254.8 t/s | 264.4 t/s |
| pp512 | **695.4 t/s** | 666.9 t/s | 659.6 t/s |
| pp2048 | 763.1 t/s | **918.9 t/s** | **1145.8 t/s** |
| pp4096 | 715.8 t/s | **864.7 t/s** | **1048.7 t/s** |
| tg64 | 42.2 t/s | **43.7 t/s** | **43.9 t/s** |

The crossover sits around a thousand prompt tokens, and the micro-batch is why:
with 256 experts at top-8, a 512-token chunk routes only ~16 rows to each
expert, so most of every expert-GEMM tile is padding and a bigger chunk buys
more than anything else on the graph. Below that, a whole prefill is one small
graph and the fixed cost per call — the managed hop, the input uploads, the
154880-wide logits copy back — is a visible fraction of it, which is where
llama.cpp's few percent come from. Decode is memory-bound and lands a few
percent ahead either way. Run-to-run spread on these numbers is about 4%.

Weight load is 218 GiB in ~37 s (5.9 GiB/s) from a warm page cache, using 16
reader threads across the six shards.

### Tensor parallelism, measured

Same 3 GPUs, GLM-5.2-UD-IQ2_XXS, `--tp 3` (heads + expert rows) against the
default layer split:

| test | layer split | `--tp 3` |
|---|---:|---:|
| pp2048 | **896.8 t/s** | 502.8 t/s |
| tg64 | **43.9 t/s** | 16.2 t/s |

Two things to know before reaching for it. The first is exactness: splitting the
attention turns one GEMM into a sum of per-rank partials, so the residual differs
in the last bits, and — exactly as for batched decode above — 75 layers of
top-8-of-256 routing amplify that into a different continuation on a 2-bit
checkpoint. Against the recorded llama.cpp goldens the layer split reproduces
5 of the 6 recorded prompts and `--tp 3` reproduces 3 of 6; the tokens that differ are the
near-tied ones. The second is speed, and the reason is the interconnect. Each
layer needs two all-reduces of the `[6144, n_tokens]` hidden state, and these
cards are PCIe-attached with no
NVLink: a 1024-token prefill chunk moves ~25 MB per crossing, 78 layers x 2
reductions deep, which is more time on the bus than the split saves in
arithmetic. Layer splitting moves the hidden state exactly twice per token, so
on this machine it wins on both counts. On an NVLink/NVSwitch host — where an
all-reduce is roughly an order of magnitude cheaper — the balance reverses; the
split itself is the same either way.

## Running it

```bash
# 3 GPUs, layer split (the default: every visible GPU)
dotnet run --project TensorSharp.Cli -- --model GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf \
    --backend ggml_cuda --prompt "Explain MLA in one paragraph."

# Fewer GPUs, or a specific count
TS_GLM_NGPU=2 dotnet run --project TensorSharp.Cli -- --model ... --backend ggml_cuda

# Tensor parallel across 3 GPUs instead of splitting the layers
dotnet run --project TensorSharp.Cli -- --model ... --backend ggml_cuda --tp 3

# Low VRAM: keep the routed experts (92% of the checkpoint) in system RAM
dotnet run --project TensorSharp.Cli -- --model ... --backend ggml_cuda --cpu-moe
dotnet run --project TensorSharp.Cli -- --model ... --backend ggml_cuda --n-cpu-moe 30
```

Offload is a way to fit a checkpoint that does not fit, not a speed knob — when
the model already fits, moving experts to the host only adds bus round trips.
Same 3 GPUs, same run: default layer split pp2048 **915.9** / tg64 **43.9** tok/s,
`--n-cpu-moe 30` **94.7** / **16.4**, `--tp 3` **505.6** / **17.6**. Reach for
`--n-cpu-moe` when the alternative is not running at all.

Offload composes with tensor parallelism: `--n-cpu-moe 30 --tp 2` loads, and the
host-resident layers keep their experts whole (splitting them would save no host
RAM and no host time, and a strided strip cannot be served in place from the GGUF
mapping — it would turn a mapped file into a 200 GiB private copy), so rank 0
evaluates those layers while the GPU-resident ones stay split. `--n-cpu-moe 30`
on its own reproduces llama.cpp 3/3.

### Context length

The GGUF advertises 1,048,576 tokens. That is not a promise the caches fit: at
78 layers a 576-wide MLA row plus the indexer's is ~93 KiB per token, so a 1M
context is ~93 GiB of KV — a whole card's worth on top of the weights — before the graphs are
counted. So the advertised number is treated as a **ceiling**, not a request:
the loader sizes the context from the VRAM actually left after the weights land
(minus what the DSA masks and the LM head need for one `n_ubatch` graph) and
says what it picked. On the default layer split across the three cards above that
pick is 342,272 tokens, and `--n-cpu-moe 30` raises it to 646,400. The line below
comes from a `--tp 3` run, where every rank holds a full-length cache and the
pick therefore drops much further:

```
[glm] context 91136 tokens (the GGUF advertises 1048576): 18.3 GiB free per rank
      after the weights, and the caches and graphs have to live in it.
```

`MAX_CONTEXT` turns that around: a context you name is a requirement, honoured
if it fits and refused with the numbers if it does not, rather than quietly
shrunk under you.

### Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `TS_GLM_NGPU` | 0 (all) | GPUs to spread the layers over |
| `TS_GLM_UBATCH` | 1024 | prefill micro-batch; 2048 is faster on long prompts if VRAM allows |
| `TS_GLM_THREADS` | min(cores, 32) | CPU-backend threads (the routed-expert matmul overrides this from `--cpu-moe-threads`) |
| `TS_GLM_NATIVE` | 1 | 0 runs the managed per-op path on a GGML backend |
| `TS_GLM_FA` | 1 | 0 disables flash attention (falls back to soft_max) |
| `TS_GLM_FUSED_LID` | 1 | 0 builds the indexer out of primitives instead of `ggml_lightning_indexer` |
| `TS_GLM_OP_OFFLOAD` | auto | scheduler op-offload; off by default once any layer's experts are host-resident |
| `TS_GLM_VRAM_RESERVE_MB` | 3072 | per-device headroom the layer split leaves for compute buffers |
| `TS_GLM_GRAPH_CACHE` | 8 | cached built+allocated graphs |
| `TS_GLM_MOE_MMAP` | 1 | 0 copies host-resident experts instead of mapping the GGUF |
| `TS_GLM_TP_SHARD` | 3 | tensor-parallel split: 1 heads, 2 routed experts, 3 both |
| `TS_GLM_TP_OVERSUBSCRIBE` | 0 | 1 lets tensor-parallel ranks share a GPU (correctness testing only) |
| `TS_GLM_BATCHED_DECODE` | 1 | 0 makes the native side decline every batched decode, forcing the per-sequence path |
| `TS_GLM_TRACE` | — | layer list (or `all`) to dump per-layer activation sums, matching `llama-eval-callback`'s layout |
| `TS_GLM_BD_DEBUG` | 0 | 1 narrates each batched decode step (which slots, graph reused or rebuilt, how far it got) |
| `TS_GLM_TOPK` | 1 | 0 attends densely even past the indexer top-k — an A/B for the DSA selection |
| `TS_GLM_NODES_PER_LAYER` | 256 | graph node budget per layer per rank |
| `TS_GLM_LOAD_THREADS` / `TS_GLM_LOAD_CHUNK_MB` | 16 / 64 | weight-load parallelism and chunk size |

## Chat format

```
[gMASK]<sop>[<|system|>Reasoning Effort: Max][tools]<|user|>...<|assistant|><think>
```

Thinking is opt-in (`--think`, `/think on` in the REPL, `"think": true` in an API
request), as on every other family here. Turning it on adds the
`<|system|>Reasoning Effort: Max` line and leaves the generation prompt's
`<think>` block open for the model to close; left off, the prompt emits
`<think></think>` so the model answers directly. Past turns' reasoning is always
dropped from the prompt, matching the template's `clear_thinking` default. Tool calls come back as
`<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`,
one XML element per argument (values that were rendered with `tojson` are parsed
back into numbers / arrays / objects).
