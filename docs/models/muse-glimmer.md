# Muse-Glimmer

[← back to model index](README.md)

| Property | Value |
|---|---|
| GGUF architecture key | `muse-glimmer` |
| Source class | [`MuseGlimmerModel`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.cs) (legacy per-seq) |
| Speculative drafter | [`MuseGlimmerModel.DFlash.cs`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.DFlash.cs) + [`DFlashConfig`](../../TensorSharp.Models/Models/MuseGlimmer/DFlashConfig.cs) |
| Vision encoder | [`MuseGlimmerVisionEncoder`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerVisionEncoder.cs) |
| Image processor | [`MuseGlimmerImageProcessor`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerImageProcessor.cs) |
| Example models | Muse-Glimmer-30B |
| Modalities | Text, image |
| Thinking mode | Yes (the chat template emits an `assistant to=self` reasoning channel) |
| Tool calling | Yes (ATEM XML markup in the chat template) |
| Batched / paged forward | No (legacy per-seq) |
| Tensor parallelism | Not supported (`--tp 1` only) |

## Quick start

```bash
# text
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --input prompt.txt --backend ggml_cuda --max-tokens 256

# image understanding (needs the mmproj)
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --mmproj models/mmproj-Muse-Glimmer-30B-Q8_0.gguf \
  --image photo.png --input question.txt --backend ggml_cuda --max-tokens 300

# DFlash speculative decoding (lossless; output is identical to plain greedy)
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --draft-model models/dflash-kquant.gguf \
  --spec-draft-n-max 15 --input prompt.txt --backend ggml_cuda
```

`--draft-model` can also be supplied as `TS_MUSE_GLIMMER_DFLASH`.

## 1. Text architecture

52 dense layers, `n_embd` 6656, `n_ff` 19968, 32 query heads / 2 KV heads,
`head_dim` 128, vocab 202048.

* **Interleaved sliding-window attention.** `muse-glimmer.attention.sliding_window_pattern`
  is a scalar period P (4 for the 30B). Layer `l` is a sliding-window layer when
  `l % P < P - 1`, so every P-th layer (l = 3, 7, 11, ... 51) is full causal —
  39 SWA + 13 full for the 30B. This mirrors `llama_hparams::set_swa_pattern(P)`
  with `dense_first = false`. The window predicate is llama.cpp's STANDARD SWA
  rule: a key at `p0` is visible to a query at `p1` iff `p1 - p0 < n_swa` (2048).
* **RoPE only on the sliding-window layers**; the full layers are NoPE. The RoPE
  flavour is ggml's NORM (interleaved adjacent pairs, `mode 0`), not NeoX — the
  converter un-permutes transformers' rotate_half layout at conversion time.
* **Per-head QK RMSNorm.** The Q norm weight is synthesized at conversion to
  carry the model's `qk_scale_factor`; the K norm weight is ones.
* **Attention output gate.** `attn = attn * sigmoid(W_gate @ attn_norm(x))`,
  applied before `o_proj`. The gate is projected from the same post-norm tensor
  that feeds Q/K/V.
* **Four RMSNorms per layer.** `attn_norm` and `ffn_norm` use the model's
  `f_norm_rms_eps` (1e-5); `post_attention_norm` and `post_ffw_norm` use a
  **hardcoded 1e-8** (see `llama.cpp/src/models/muse-glimmer.cpp`). Getting this
  epsilon wrong is a silent numeric divergence.
* **Unweighted RMSNorm on the input embeddings** (no learned scale) — this
  replaces the `sqrt(hidden_size)` scaling other Gemma-like models use.
* **Dense SwiGLU FFN** (no MoE).
* **Output path:** `logits = lm_head(h)`, then `logits *= logit_scale` (0.19612),
  then tanh softcapping at `final_logit_softcapping` (20.0). The scale is applied
  **before** the softcap.

## 2. Vision tower

50-layer ViT, `n_embd` 1536, `n_ff` 8960, 16 heads (`head_dim` 96), patch 14,
LayerNorm with bias, plain 2-linear MLP with exact **erf**-GELU (not the tanh
approximation), learned 32x32 position embedding.

* **Preprocessing** is a pure stretch — no padding, no tiling. The merged-token
  grid is chosen by llama.cpp's `muse_glimmer_grid_size` (aspect-closest of the
  four floor/ceil candidates, ties broken toward more tokens, capped at 4096
  merged tokens), then the image is resized to `grid * 28` px with a
  Pillow-compatible **Lanczos-3** filter and normalized with the mmproj's
  `image_mean` / `image_std` (0.5 / 0.5).
* **Sparse window attention.** Patches are permuted into 32x32 windows (the
  window side is `sqrt(position_embd_rows)`), edge windows clipped. Layers where
  `(il + 1) % 4 == 0` or `il == n_layer - 1` attend globally; the rest attend
  only within their window. Because the permutation makes the mask
  block-diagonal over contiguous row ranges, the encoder runs attention once per
  window instead of materializing an `n_tok x n_tok` mask.
* **2D RoPE** (`theta` 10000): the first half of `head_dim` is roped with the
  1-indexed patch column, the second half with the 1-indexed row, both in the
  interleaved-pair (NORM) flavour.
* **2x2 pixel shuffle with CHANNEL-OUTER packing**: `out[o][c * 4 + s]`, not
  `s`-outer. This is the classic trap and is silently wrong the other way.
* **Adapter**: 6144 -> 4096 -> 4096 -> 6656, erf-GELU between, no biases.

The tower's 2D matmul weights are kept in their GGUF quantization on GGML
backends and fed straight to `AddmmQuant`. Dequantizing this tower to F32 would
cost ~7.4 GB and evict the language model's device-resident weights;
`TS_MUSE_GLIMMER_VENC_F32=1` restores the F32 path for A/B testing.

### Prompt plumbing

The chat template renders an image content part as a single `<|patch|>`.
`ChatTemplate.InjectMultimodalTokens` emits it, and the host (CLI or
`ModelMultimodalInjector`) expands each one into
`<|image_start|>` + N filler rows + `<|image_end|>`, where N is the encoder's
merged-token count. The filler rows are overwritten by the projected embeddings
*before* the input RMSNorm — llama.cpp feeds image embeddings straight into
`build_inp_embd` and norms the merged stream.

## 3. DFlash speculative decoding

DFlash is a **block** drafter: a separate 5-layer GGUF
(`general.architecture = dflash`) that proposes the whole speculative window in
one forward. It borrows the target's `token_embd` and `output` (lm_head) and
keeps its own SWA KV ring. Three passes:

1. **Encode** — the target's per-layer *input* residuals at
   `dflash.target_layers` (`[2, 14, 26, 38, 50]`) are concatenated into one
   33280-wide row per position, projected by `fc.weight` and RMS-normed by
   `enc.output_norm`.
2. **Inject** — that row feeds `attn_k` / `attn_v` of every draft layer; keys get
   a per-head RMSNorm and **NeoX** RoPE (the drafter's rope flavour differs from
   the target's) at the target position, and land in the drafter's ring.
3. **Draft** — `[anchor, MASK x (block_size - 1)]` runs the 5 blocks
   non-causally over `[ring window | the block's own keys]` and the target's
   lm_head turns the result into `block_size` rows of logits. Row 0 is the
   anchor's own prediction and is discarded.

The drafter's logits get **neither** the target's `logit_scale` nor its softcap
(llama.cpp's dflash graph ends at the lm_head matmul). argmax is invariant to
both, but the acceptance confidence is not, so the per-position confidences
handed to the executor are the softmax of the raw drafter logits.

Verification is greedy against the target, so **the emitted token stream is
identical to plain greedy decoding**. Measured acceptance on this model is
73-100% per prompt (mean ~84%) with a mean accepted run of ~2 tokens per verify
step, matching llama.cpp's behaviour on the same GGUFs.

**It is not a speedup on this model yet** (15.5 tok/s against 19.5 for plain
decode). The trunk verify now runs on the fused kernel — the kernel emits the
per-layer input residuals the drafter's encoder needs, and scores every drafted
row in one pass — but the DRAFTER itself is still per-op: a 33280x6656 `fc`
GEMM plus five unfused layers per step costs more than the ~2.5 target forwards
that ~2 accepted tokens per step saves. llama.cpp wins with DFlash (1.3-4.8x on
these prompts) because its drafter is also a single graph. Fusing the drafter is
the obvious next step; until then DFlash is correct, lossless, and off by
default.

## 4. Parity with llama.cpp

`InferenceWeb.Tests/MuseGlimmerParityTests.cs` checks the implementation against
golden outputs captured from a `llama-server` running the same GGUFs
(`.parity/gen_ref.py`, `.parity/gen_ref_vision.py`):

| Check | Result |
|---|---|
| Tokenizer (5 prompts, `add_special`) | Token-identical |
| Greedy continuation (5 prompts, 26-32 tokens each) | Token-identical |
| Long context (4651-token prompt, 2.3x the sliding window) | Token-identical — covers the SWA mask, the padded-window rollover and a KV-cache grow |
| DFlash greedy continuation | Token-identical to both the non-speculative path and llama.cpp |
| Image geometry (`ComputeTargetSize` / `ComputeTokenCount`) | Matches `muse_glimmer_grid_size`; 1024x1024 -> 1036x1036 -> 1369 tokens, 336x336 -> 144 tokens (confirmed against llama-server's `prompt_tokens`) |
| Image description | Near-verbatim match including the OCR'd overlay text |

Regenerate the goldens with:

```bash
llama-server -m Muse-Glimmer-30B-UD-IQ2_XXS.gguf --mmproj mmproj-Muse-Glimmer-30B-Q8_0.gguf -ngl 99 --port 8899
python .parity/gen_ref.py http://127.0.0.1:8899 .parity/ref_text.json
python .parity/gen_ref_vision.py http://127.0.0.1:8899 .parity/ref_vision.json
```

Then `TS_TEST_MODEL_DIR=<model dir> dotnet test --filter MuseGlimmerParityTests`.

## 5. Performance

Measured on an RTX 3080 Laptop (16 GB), `Muse-Glimmer-30B-UD-IQ2_XXS`,
`ggml_cuda`, 16-token prompt, 128 generated tokens, greedy. llama.cpp is
`build-ninja-cuda` at the same commit as the vendored ggml, run with
`llama-cli -st -no-cnv -ngl 99 -c 4096 -n 128 --temp 0`.

| Path | Prefill | Decode |
|---|---|---|
| llama.cpp, plain | 226.5 tok/s | 21.6 tok/s |
| llama.cpp + DFlash (`-md`, `--spec-draft-n-max 15`) | 208.6-216.7 tok/s | 20.0-20.2 tok/s |
| TensorSharp per-op (`TS_MUSE_GLIMMER_FUSED=0`) | 55 tok/s | 4.3 tok/s |
| TensorSharp fused, plain | 357.9-361.3 tok/s | 19.1-19.9 tok/s |
| TensorSharp fused + DFlash (per-op drafter) | ~260 tok/s | 15.8-16.4 tok/s |
| **TensorSharp fused + fused DFlash** | **260-269 tok/s** | **26.0-27.0 tok/s** |

Decode with DFlash is **~1.25x llama.cpp's best** (21.6 plain) and **~1.3x
llama.cpp's own DFlash**. Plain decode is ~90% of llama.cpp's; prefill is ~1.6x.

Note that DFlash is a net *loss* for llama.cpp on this GPU (20.0 vs 21.6): its
drafter is fast in absolute terms but not fast enough to pay for itself when the
target forward is only ~46 ms. It only pays once the drafter itself is fused.

### Long context

Same hardware and model, prompts of 16336 / 32680 / 65373 tokens, 128 generated
tokens, `MAX_CONTEXT` set to fit each prompt. Two reps per point, **alternating
engines** — running one engine's three contexts back to back and then the other's
biases the second engine downward by 10-15% through GPU thermals, which is enough
to invent or hide a gap this size.

| Context | llama.cpp prefill / decode | TensorSharp prefill / decode |
|---|---|---|
| 16K | 589.1, 584.9 / 20.1, 19.9 | 587.4, 580.4 / 18.7, 18.4 |
| 32K | 555.1, 539.7 / 18.0, 16.4 | 551.8, 529.5 / 17.6, 16.5 |
| 64K | 480.9, 498.6 / 16.7, 16.8 | 474.6, 488.1 / 14.8, 17.1 |

Prefill is **98-99.5% of llama.cpp at every context** (it was 77-83% before the
work described below). Decode is 93% at 16K, level at 32K, and 88-102% at 64K —
the 64K decode spread is real run-to-run variance, not a stable number.

Earlier revisions of this table reported 471.5 / 426.0 / 353.3 prefill; that is
the pre-mask-sharing code.

### Why the per-op path was slow

Not arithmetic — dispatch. The per-op forward submits ~600 GGML ops per token
across 52 layers, and each carries a host-visible round trip. Measured: a 1-row
forward cost ~262 ms and a 74-row forward ~1332 ms, i.e. ~262 ms of *fixed*
per-forward cost plus ~14 ms per row. Every model in this repo that reaches
llama.cpp-class decode throughput does it the same way — one whole-model kernel.

### The fused kernel

[`ggml_ops_muse_glimmer.cpp`](../../TensorSharp.GGML.Native/ggml_ops_muse_glimmer.cpp)
exports `TSGgml_MuseGlimmerModelForward`, which builds the entire model — all 52
layers, the final norm, the LM head, the logit scale and the tanh softcap — as
ONE ggml graph. It is numerically the same chain as `TransformerBlock`: same op
order, same epsilons, same NORM-flavour RoPE on sliding-window layers only, same
attention output gate. `TS_MUSE_GLIMMER_FUSED=0` forces the per-op path for A/B.

Design points that mattered:

* **Persistent, capturable graphs.** A graph is built once with stable tensor
  addresses (raw `ggml_init` + `ggml_backend_alloc_ctx_tensors`, not a gallocr,
  whose lifetime packing moves addresses) so ggml-cuda captures it and a token
  becomes one replay. Topology is held byte-identical between steps by writing
  KV with `ggml_set_rows` (the write row is an I64 *input*) and reading a window
  padded to a 256-row stride with an F16 mask input — so a rebuild happens once
  every 256 tokens rather than every token.
* **The graph pool is keyed by `(model, KV holder, n_tokens)`.** A speculative
  run alternates 1-row decode with k-row verify batches whose k varies; keyed
  without `n_tokens` those shapes evict each other from one slot and every step
  pays a full rebuild.
* **Prefill uses the shared gallocr, decode does not.** A prefill chunk's
  intermediates are per-row — the `[2*n_ff, n_tokens]` gate/up alone is 163 MB
  per layer at 1024 rows — so giving every node its own slot asks for ~47 GB.
  `alloc_graph_reuse_gallocr` packs them by lifetime; that is what makes a
  multi-row fused graph fit at all.
* **KV caches are zero-filled at allocation.** `ModelBase.InitializeCacheTensor`
  skips the fill on GgmlCuda because the per-op attention only reads rows it
  wrote. The fused kernel reads the *padded* window, whose extra rows are masked
  with `-inf` — and `-inf + NaN` is still NaN, so those rows must be finite.
* **CUDA/Vulkan only.** GgmlCpu stays on the per-op path: a 1024-row fused graph
  faulted there, and shipping a graceful fallback beats shipping a crash.
* **In-graph embedding is opt-in.** The kernel can do the embedding gather and
  the weightless input norm itself, but binding the 202K x 6656 table pins a
  second ~1.1 GB tensor; on a 16 GB card that evicts layer weights and cost more
  than the two dispatches it saved (18.5 -> 16.1 tok/s). It is therefore enabled
  only when the LM head is tied to the table (so it is resident anyway), or via
  `TS_MUSE_GLIMMER_INGRAPH_EMBED=1`.


### The fused DFlash drafter

The drafter was the reason speculation lost: per speculative step the managed
drafter issued ~150 GPU dispatches for ~2.5 GB of weight reads -- about a quarter
of one target forward -- and cost ~100 ms.
[`ggml_ops_dflash.cpp`](../../TensorSharp.GGML.Native/ggml_ops_dflash.cpp) turns
it into two graphs:

* `TSGgml_DFlashInject` -- `fc` -> RMSNorm -> per draft layer
  {k/v projection, per-head k norm, NeoX RoPE, `ggml_set_rows` into the ring}.
  No Q, no attention, no FFN, no LM head. llama.cpp's `build_dflash` early-returns
  at the same point.
* `TSGgml_DFlashDraftBlock` -- `[anchor, MASK x (b-1)]` through the 5 draft
  blocks, then the *target's* LM head (borrowed, never duplicated) and a softmax.

Both are persistent and capturable. This is where TensorSharp diverges from
llama.cpp, which gets **no** graph reuse on its draft context: `gf_res_prev` is a
single slot shared by encode and decode, and the DFlash cycle
ENCODER -> DECODER(embd) -> DECODER(token) can never satisfy `can_reuse`, so all
three calls rebuild every step. llama.cpp makes that rebuild cheap; fusing avoids
it entirely.

Two details make the capture possible:

* **Attention reads the WHOLE ring**, `kv_len = ring_rows + b`, fixed across
  steps. Attention is permutation-invariant over the KV axis, so the ring's
  circular order does not matter, and a host-built mask keyed on a per-slot
  position map expresses exactly which slots are live.
* **The mask cutoff is the ANCHOR's position, not the query's.** After a
  partially-rejected draft the ring still holds keys the drafter wrote for
  positions past the anchor; those rows are stale and no query in the block may
  see them, not even the later ones whose own position is higher. The per-op path
  gets this for free by only ever gathering `[winStart, anchor)`.

**On-device top-1.** llama.cpp pulls the whole `[202048, 16]` probability block
back to the host every draft step -- 12.9 MB over PCIe plus a 3.2 M-element scan,
and `common_sampler` then materializes a 202048-entry array *per block position*.
The kernel instead finishes with `ggml_argmax` plus a `get_rows` gather of the
winning probability, returning two 16-element tensors. Argmax is invariant under
softmax and the winning probability is exactly the confidence the executor
multiplies cumulatively.

Result on the benchmark above: **15.8-16.4 -> 26.0-27.0 tok/s**, with the draft
statistics (`drafted=96 accepted=75`, 78.1% acceptance) byte-identical to the
per-op drafter -- fusing changed the speed, not a single sampled token.
`TS_DFLASH_FUSED=0` forces the per-op drafter for A/B.


### One mask per attention class, not one per layer

This was the whole long-context prefill gap.

The kernel used to allocate its F16 attention mask **inside** the per-layer loop,
so a graph carried 52 mask tensors — and 52 host-side copies — even though the
contents depend only on `(window, n_tokens, start_pos)` and every sliding-window
layer shares one window while every full layer shares the other. There were only
ever two distinct buffers; the decode replay path proved it by building exactly
two and uploading them into all 52 tensors in a loop.

At 64K with a 2048-row chunk that is 13 x [65536, 2048] + 39 x [4352, 2048] F16 =
**3.90 GB of masks, per chunk**, on a 16 GB card already holding 9.2 GB of
weights. Worse than the residency, all of it was *generated on the host* by a
scalar per-element loop and pushed over PCIe — and on the prefill path through
the synchronous `ggml_backend_tensor_set`, which does a full
`cudaStreamSynchronize` after every copy: 52 stalls per chunk, 1664 across a 64K
prompt. Summed over a 64K prefill the kernel generated and uploaded ~75 GB of
mask.

llama.cpp never had this: `build_attn_inp_kv_iswa` creates exactly two masks
per graph (`llama-graph.cpp:3266,3276`) and the per-layer choice is a pointer
pick (`:3042`).

Sharing them brings 3.90 GB to 273 MB and two uploads per chunk. Measured at 64K:
**prefill 353 -> 486 tok/s, peak VRAM 16059 -> 12671 MiB.** Sharing is safe
because nothing writes a mask — `ggml_flash_attn_ext` takes it as `src[3]` and
`ggml_soft_max_ext` as `src[1]`, and both CUDA kernels bind it `const`.

Three smaller items landed with it:

* **The mask is now generated on the GPU** on the prefill path, straight into its
  device buffer, removing the host fill and the H2D upload entirely. The causal
  kernel already existed (`ggml_ops_mask.cu`, written for the Gemma 4 verify
  kernel); the sliding-window layers needed a new `tsg_cuda_fill_ring_mask_f16`
  because a ring's columns are not in position order. Decode keeps
  `decode_input_set_async`, which is the CUDA-capture-safe path in and is already
  cheap at one row.
* **SwiGLU is one `GGML_OP_GLU` node** instead of view + 2x `cont` + `silu` +
  `mul`. That removes three `[n_ff, n_tokens]` temporaries per layer — ~479 KB of
  traffic per row per layer, and ~156 dispatches per decoded token. Note that
  `ggml.h`'s comment on `ggml_glu` says the gate is the second half, which
  contradicts the kernel; the kernel is authoritative (`swapped=false` applies
  SiLU to the *first* half) and the parity tests confirm it.
* **The two `ggml_cont` calls on the K/V write path are gone.** `ggml_set_rows`
  asserts `ggml_is_contiguous_rows`, not full contiguity, and a `0,2,1,3` permute
  already satisfies that — the copies were pure overhead, 104 kernels per token.

Things that were tried and rejected, so they are not re-tried: **shrinking the
SWA ring by lowering the prefill chunk makes everything worse.** Chunk 2048 (ring
4352) gives 475.6 prefill / 16.1 decode at 64K; chunk 1024 (ring 3328) gives
443.2 / 15.7; chunk 512 (ring 2816) gives 421.9 / 15.1. Decode is not KV-bandwidth
bound here — both engines run at ~35-39% of peak bandwidth because IQ2_XXS matvec
is ALU-bound — so the smaller ring buys nothing and the smaller chunk costs GEMM
efficiency. llama.cpp's ring is smaller (2560 rows) only because its default
`n_ubatch` is 512; that is not an advantage to copy.

### Long prompts: chunking and the SWA ring

Two things had to change before a 16K+ prompt would run at all, let alone fast.

**Prefill is chunked.** `ForwardCore` splits anything longer than
`TS_MUSE_GLIMMER_PREFILL_CHUNK` (default 2048) into chunks, exactly as llama.cpp
splits at `n_ubatch`. Without it a 16336-token prompt is built as one graph whose
activations scale with the row count, and it fails to allocate on both the fused
and the per-op paths -- 16K tokens of KV cost under a GB, but one 16K-row graph
needs tens of them. Multimodal prompts are never chunked: vision rows are injected
at absolute prompt offsets and the pending list is drained in one go.

**Sliding-window layers get their own small ring.** 39 of the 52 layers never look
back past `n_swa` (2048), so sizing them for the full context wastes most of the
cache. Uniform F16 KV at 64K is 3.5 GB; llama.cpp allocates 954 MB because
`llama_kv_cache_iswa` sizes its SWA cache `min(n_ctx, n_swa + n_ubatch)`. On a
16 GB card holding 9.2 GB of weights that difference is the entire headroom -- a
64K text-only run peaked at **16059 MiB of 16384** and lost throughput to the
resulting memory pressure.

TensorSharp now allocates the SWA layers `pad(n_swa + chunk + 1, 256)` rows
(4352 at the default chunk) and indexes them by `position % rows`. At 64K that is
**1049 MB instead of 3.5 GB (29% of a uniform cache)**, and it bought
+56% prefill (226 -> 353 tok/s) and +15% decode (13.4 -> 15.4 tok/s) at 64K.

Details worth knowing:

* **The kernel reads the whole ring, not a sub-span.** A ring's slots are not in
  position order, so there is no contiguous window to narrow to. Reading all
  `rows` keeps the graph shape *fixed*, which is also what keeps the CUDA
  capture; `fill_mg_ring_mask` carries liveness, causality and the window. SWA
  layers therefore get a second I64 write-index input (`kv_index_swa`) holding
  `position % rows`, while full layers keep the plain `position`.
* **The `+1` is load-bearing.** Sized at exactly `n_swa + chunk` (4096 rows), a
  4651-token prompt diverged from llama.cpp on the *first decode step*; 4352 rows
  is the smallest size that reproduces llama.cpp exactly. It is the same margin
  `DFlashConfig.RingRows` already uses (`n_swa + block_size + 1`).
* **The ring only arms when the fused kernel is available**, because the per-op
  attention treats a cache row as an absolute position. If the fused forward ever
  declines while the ring is armed, the per-op path throws rather than returning
  quietly wrong logits. `TS_MUSE_GLIMMER_SWA_RING=0` restores uniform sizing.
* **A forward wider than `rows - n_swa` is rejected.** Chunking keeps text prompts
  under that by construction, but a multimodal prompt deliberately skips chunking
  (vision rows are injected at absolute offsets), so it is the one way to submit an
  oversized batch — which would alias two live positions onto one ring slot and
  silently corrupt all 39 sliding-window layers. `ForwardCore` throws instead.

Also applied on the managed side:

* `ffn_norm` + gate/up + SiLU-mul + down in one GGML graph on the per-op
  **prefill** path (`TryFusedDenseFFNProject`; +53-76% on prefill, -9% on
  single-row decode, hence prefill only). This is the fallback path now.
* The vision tower stays quantized (1.94 GB instead of 7.4 GB).
* The per-op decode attention handles F32, F16 and block-quantized KV caches, so
  `ApplyModelAlignedKvCacheDefault`'s F16 choice is honoured.

## 6. Environment variables

| Variable | Effect |
|---|---|
| `TS_MUSE_GLIMMER_FUSED` | `0` = disable the fused whole-model kernel (per-op A/B) |
| `TS_MUSE_GLIMMER_PERSIST` | `0` = disable the persistent/capturable graph, rebuild every call |
| `TS_MUSE_GLIMMER_INGRAPH_EMBED` | `1` = force the in-graph embedding stage even for an untied LM head |
| `TS_MUSE_GLIMMER_DFLASH` | DFlash drafter GGUF path (same as `--draft-model`) |
| `TS_MUSE_GLIMMER_VENC_F32` | `1` = dequantize the vision tower to F32 (A/B; ~7.4 GB) |
| `TS_MUSE_GLIMMER_GELU_TANH` | `1` = use the tanh GELU approximation in the tower instead of exact erf |
| `TS_MUSE_GLIMMER_VENC_TRACE` | `1` = per-stage checksums of the vision residual stream |
| `TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS` | MLX lazy-graph flush interval (default 4, `0` disables) |
| `TS_PREFILL_CHUNK` | Prompt chunk size for `ForwardRefill` (default 2048) |
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | Tokens per prefill forward (default 2048, `0` disables chunking) |
| `TS_MUSE_GLIMMER_SWA_RING` | `0` = size every layer for the full context instead of ringing the SWA layers |
| `TS_MUSE_GLIMMER_SWA_ROWS` | Override the SWA ring size in rows (diagnostics) |
| `TS_DFLASH_FUSED` | `0` = disable the fused DFlash drafter (per-op A/B) |
| `TS_DFLASH_PERSIST` | `0` = rebuild the DFlash graphs every step instead of replaying |
