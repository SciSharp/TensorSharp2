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
| Tensor parallelism | Yes — GGML CUDA / Vulkan, `--tp 2` max (the 30B has 2 KV heads) |

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

Verification is greedy against the target, so the emitted token stream is the
plain-greedy stream (see [Output agreement](#output-agreement) for what
"lossless" means once floating point is involved).

Both the drafter and the trunk verify run as fused, CUDA-graph-captured native
graphs; that is what turned speculation from a loss into a win on this model.
The current head-to-head against llama.cpp's own DFlash implementation — where
TensorSharp wins, where it loses, and why — is in
[§5 Performance](#5-performance).

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

Re-measured 2026-08-13. This section replaces an earlier table taken on a 16 GB
laptop GPU with a 2-bit quantization; none of those numbers survive here. The
engineering subsections further down keep their original A/B measurements
because they document *why* a change was made — each says which machine it was
taken on.

### Test setup

| | |
|---|---|
| GPU | 1x **NVIDIA RTX PRO 6000 Blackwell Server Edition** (97,887 MiB), driver 580.126.20, PCIe 5.0 x16. The host has two; every row except [Two GPUs](#two-gpus) pins `CUDA_VISIBLE_DEVICES=0`. |
| CPU / RAM | 2x Intel Xeon 6952P (384 threads), 1.5 TiB |
| Model | `Muse-Glimmer-30B-Q8_0.gguf` (27.6 GiB) |
| Drafter | `dflash-kquant.gguf` (1.5 GiB) |
| TensorSharp | commit `5098e3f`, vendored ggml `8846b79` (2026-08-12), `--backend ggml_cuda`, native library built `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real` |
| llama.cpp | master `8e7f22b` (2026-08-13, libggml 0.19.0 — within a day of the vendored ggml), same CUDA arch, `-DGGML_CUDA=ON -DLLAMA_CURL=OFF` |
| Sampling | greedy on both sides (`--temp 0` for llama.cpp; **no** sampler flags for TensorSharp) |
| Generation | 128 tokens |
| Batching | llama.cpp `-b 2048 -ub 2048`, matching TensorSharp's default `TS_MUSE_GLIMMER_PREFILL_CHUNK` of 2048 |
| Reps | 2 per point, **engines alternating within each context** |

**Both engines prefill the same token sequence.** TensorSharp applies the chat
template and llama.cpp's `-no-cnv -f` does not, so handing both the raw question
would compare a 60-token prompt against a 21-token one. The rendered prompt is
dumped once (`TensorSharp.Cli --dump-prompt`) and *that* text is what
`llama-cli` is given; `llama-tokenize` then confirms the count matches what
TensorSharp reports (60 / 501 / 2050 / 16126 / 32274 / 64575 / 123931 tokens).

### Plain text generation

Mean of two reps, tok/s. The ratio column is TensorSharp / llama.cpp, so above
1.00x is TensorSharp ahead.

| Prompt tokens | llama.cpp prefill | TS prefill | ratio | llama.cpp decode | TS decode | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 60 | 362 | **459** | 1.27x | 34.7 | **35.0** | 1.01x |
| 501 | 927 | **1135** | 1.23x | **36.2** | 34.3 | 0.95x |
| 2050 | 1132 | **1317** | 1.16x | **35.0** | 33.5 | 0.96x |
| 16126 | **1325** | 1249 | 0.94x | **32.2** | 30.9 | 0.96x |
| 32274 | **1303** | 1211 | 0.93x | **32.1** | 29.9 | 0.93x |
| 64575 | **1256** | 1150 | 0.92x | **32.4** | 29.1 | 0.90x |
| 123931 | **1166** | 1073 | 0.92x | **30.7** | 26.6 | 0.86x |

The shape is consistent across the ladder: TensorSharp's fused whole-model graph
wins short prompts by 1.16-1.27x, the two engines cross somewhere between 2K and
16K, and llama.cpp keeps a 6-8% prefill edge above that. Decode is a tie at 60
tokens of context and slides to 0.86x at 128K — the gap grows with KV length,
which points at the attention path rather than the FFN.

> **Caveat on the four long rows.** The 60 / 501 / 2050 prompts are byte-identical
> for both engines. The 16126 / 32274 / 64575 / 123931 rows were taken before the
> CRLF normalization described in [Benchmark method notes](#benchmark-method-notes)
> landed, so on those four points TensorSharp prefilled the CRLF form of the same
> document — 1.2% more tokens (16322 / 32666 / 65359 / 125412) for the same text.
> Throughput is a rate, so the effect on tok/s is small, but the generated
> continuations are not strictly comparable on those rows. The corrected re-run
> was queued and the benchmark host went offline before it finished.

Both engines run the full 128K context on one card.

### DFlash speculative decoding

Same runs with `--draft-model dflash-kquant.gguf --spec-draft-n-max 15` against
llama.cpp's `-md … --spec-type draft-dflash --spec-draft-n-max 15 -ngld 99`.
Decode tok/s; parentheses give the two-rep range where it is wide.

| Prompt tokens | llama.cpp | TensorSharp | TS, `--spec-draft-conf-min 0` |
|---|---:|---:|---:|
| 60 | 45.5 | **50.9** | 43.5 |
| 501 | 117.5 | 164.6 (150-179) | **180.3** |
| 2050 | 24.9 | **43.5** (30-57) | 34.7 |
| 16126 | **80.2** | 55.8 (37-75) | 33.2 |
| 32274 | **60.7** (43-79) | 33.8 (31-36) | 29.9 |
| 64575 | **66.1** | 48.7 (34-64) | 49.1 |
| 123931 | **69.0** | 42.3 (30-55) | 59.8 |

Speculation costs *prefill* on both engines, because the drafter's encoder has
to run over the prompt too:

| Prompt tokens | llama.cpp plain → DFlash | TensorSharp plain → DFlash |
|---|---:|---:|
| 60 | 362 → 203 (0.56x) | 459 → 341 (0.74x) |
| 501 | 927 → 495 (0.53x) | 1135 → 700 (0.62x) |
| 2050 | 1132 → 259 (0.23x) | 1317 → 703 (0.53x) |
| 16126 | 1325 → 988 (0.75x) | 1249 → 826 (0.66x) |
| 64575 | 1256 → 985 (0.78x) | 1150 → 780 (0.68x) |
| 123931 | 1166 → 920 (0.79x) | 1073 → 742 (0.69x) |

#### Why the TensorSharp column is a range and llama.cpp's is not

TensorSharp runs an **adaptive cost governor** in front of the drafter
([`MtpSpeculativeExecution`](../../TensorSharp.Runtime/Scheduling/MtpSpeculativeExecution.cs),
`AdaptiveSpeculation`, on by default). Speculation is only ever a speed
optimization, so the executor measures ms/token with drafting and without it,
and if drafting loses it **parks** the drafter for `ParkedProbeInterval = 64`
steps before probing again. llama.cpp has no such governor — it drafts every
step, unconditionally.

On a 128-token run one bad probe therefore costs half the measurement. The two
16K reps below are the same prompt, the same binary and the same weights; they
differ only in the governor's verdict:

| 16K rep | drafted / accepted | verify steps | parked steps | decode |
|---|---|---:|---:|---:|
| 1 | 64 / 48 (75%) | 13 | **67** | 36.7 tok/s |
| 2 | 132 / 103 (78%) | 22 | 3 | **74.8 tok/s** |

Rep 1's *post-park* re-probe measured speculation at **14.0 ms/token against
37.8 ms/token plain** — 2.7x faster — so the verdict that parked it was wrong.
The first speculative steps after a prefill pay the one-off graph build for the
verify batch shape, and that is exactly what the probe samples. The parked reps
at 32K, 64K and 128K show the same fingerprint (`drafted` stuck at 64-84: one
probe window, then nothing).

Reading the unparked rep as the steady state, TensorSharp's fused DFlash reaches
**94% of llama.cpp's at 16K (74.8 vs 79.8) and 96% at 64K (63.6 vs 66.5)** — not
the 1.6-2.4x deficit an earlier revision of this document reported on other
hardware. Warming the verify shape before the probe, or dropping the first
speculative step from the sample, is the obvious fix and is not done yet.

#### The confidence floor

TensorSharp defaults to `confMin = 0.35`; llama.cpp's `p_min` defaults to 0 on
this path, i.e. it always drafts its full window. `--spec-draft-conf-min 0`
makes the two policies comparable and is **not** uniformly better: it wins at
501 and 128K tokens (180 vs 165, 60 vs 42) and loses badly at 16K and 32K
(33 vs 56, 30 vs 34), where acceptance collapses from ~75% to 24-42% and every
rejected row still occupies a verify slot. The floor wants to be adaptive, not a
constant — the same conclusion the earlier laptop run reached.

#### The 2K anomaly belongs to llama.cpp

At a 2050-token prompt llama.cpp's DFlash decode (24.9 tok/s, both reps) falls
*below* its own plain decode (35.0), and its DFlash prefill collapses to 259
tok/s from 1132 — a 4.4x penalty against the 0.75-0.79x it pays at 16K and up.
TensorSharp pays 0.53x on the same point and decodes at 43.5. Nothing about the
prompt is unusual except that it stops mid-document, so the continuation is less
predictable than the question-and-answer prompts at the other sizes.

### Peak VRAM

Whole-process peak on one GPU, sampled every 2 s (MiB):

| Prompt tokens | llama.cpp plain | TS plain | llama.cpp DFlash | TS DFlash |
|---|---:|---:|---:|---:|
| 501 | 28329 | 29655 | 31881 | 29887 |
| 16126 | 28585 | 30567 | 32191 | 34089 |
| 64575 | 29401 | 32003 | 33007 | 35809 |
| 123931 | 30471 | 33787 | 34641 | 37769 |

TensorSharp carries 1.3-3.3 GB more than llama.cpp on the plain path and about
3 GB more with the drafter resident. Both fit 128K on a 96 GB card easily; on a
40 GB card the 128K DFlash configuration is the first one to run out.

### Output agreement

Greedy verification makes DFlash lossless *in exact arithmetic* — the verify
batch re-scores every drafted row against the target and keeps only the prefix
the target itself would have emitted. In floating point the verify GEMM has a
different shape from the 1-row decode GEMM, the logits differ in the last bits,
and a near-tie can flip. Measured across these runs:

* TensorSharp is **deterministic**: every configuration produced a byte-identical
  continuation to its own repeat.
* TensorSharp plain vs TensorSharp DFlash: identical at 60 / 501 / 2050 / 16126
  prompt tokens, diverged at 32274.
* llama.cpp plain vs llama.cpp DFlash: identical everywhere except at 2050.

Both engines show the same behaviour on the same corpus, so this is the tie
flip, not a verification bug. Across engines the two continuations agree for the
first 127-636 characters and then split — expected from different kernels and
different reduction orders over the same weights.

### Benchmark method notes

Five things here moved a number by more than the effect being measured, so they
are recorded rather than rediscovered:

* **Alternate the engines.** Running one engine's whole ladder and then the
  other's biases the second one down. Every context runs
  llama.cpp → TensorSharp → llama.cpp DFlash → TensorSharp DFlash.
* **Give llama.cpp the templated prompt** (above). At the 60-token point the
  difference is 3x in prompt length.
* **Normalize line endings first.** The long prompt files were CRLF. Handing
  llama.cpp an LF copy of the same document made TensorSharp prefill 16322
  tokens where llama.cpp prefilled 16126 — the same text, 1.2% more tokens, a
  different continuation. Compare the two engines' reported prompt-token counts
  before believing any long-context row.
* **Never pass `--top-k 1` to `TensorSharp.Cli`.** `SamplingConfig.IsGreedy`
  requires `TopK <= 0`, so `--top-k 1` fails
  `InteractiveSession.IsArgmaxSampling` and the speculative path silently never
  arms; the run then reports plain decode under a "dflash" label. The CLI already
  starts from `SamplingConfig.Greedy`, which is what `--temp 0` means on the
  llama.cpp side, so pass no sampler flags at all — and assert
  `cli.inference speculative:` appears in the log.
* **128 generated tokens is too few for a speculative comparison.** It is
  shorter than two park intervals, so one governor misfire moves the number by
  2x (the 16K reps above).

The GPU reports `HW Power Brake Slowdown: Active` for the whole session at
2280-2347 MHz, 180-270 W of a 450 W cap, 28-42 C — a host-level power brake, not
thermal throttling, and it applies to both engines equally. Roughly one run in
twenty comes out ~40% slow on both prefill and decode with no clock or
temperature signature (llama.cpp's 32K DFlash rep 2 is the clearest: 603 / 42.8
against 1017 / 78.6 for a byte-identical run). Treat a single-rep delta on this
host with suspicion.

### The corpus matters more than either engine

The long prompts come from `.parity/gen_long_prompts.py`, which emits a
synthetic, highly repetitive document ("Chapter *n*. The *n*th study …") and
asks a question whose answer quotes one chapter almost verbatim. Drafts on that
text are near-perfect — the 501-token point reaches **100% acceptance on both
engines**, which is why its DFlash numbers (117-180 tok/s) are 3-5x the plain
decode and should not be read as a general speculative speedup. Acceptance on
natural prose is closer to the 55-78% the 16K-128K rows show. Compare engines on
these rows; do not quote the absolute DFlash figures as what a chat workload
will see.

### Why the per-op path was slow

Not arithmetic — dispatch. The per-op forward submits ~600 GGML ops per token
across 52 layers, and each carries a host-visible round trip. Measured on the
RTX 3080 Laptop: a 1-row forward cost ~262 ms and a 74-row forward ~1332 ms,
i.e. ~262 ms of *fixed* per-forward cost plus ~14 ms per row. Every model in
this repo that reaches llama.cpp-class decode throughput does it the same way —
one whole-model kernel.

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
  second ~1.1 GB tensor; on a 16 GB card that evicted layer weights and cost more
  than the two dispatches it saved (18.5 -> 16.1 tok/s). It is therefore enabled
  only when the LM head is tied to the table (so it is resident anyway), or via
  `TS_MUSE_GLIMMER_INGRAPH_EMBED=1`.

### The fused DFlash drafter

The drafter was the reason speculation used to lose outright: per speculative
step the managed drafter issued ~150 GPU dispatches for ~2.5 GB of weight reads
— about a quarter of one target forward — and cost ~100 ms.
[`ggml_ops_dflash.cpp`](../../TensorSharp.GGML.Native/ggml_ops_dflash.cpp) turns
it into two graphs:

* `TSGgml_DFlashInject` — `fc` -> RMSNorm -> per draft layer
  {k/v projection, per-head k norm, NeoX RoPE, `ggml_set_rows` into the ring}.
  No Q, no attention, no FFN, no LM head. llama.cpp's `build_dflash` early-returns
  at the same point.
* `TSGgml_DFlashDraftBlock` — `[anchor, MASK x (b-1)]` through the 5 draft
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
back to the host every draft step — 12.9 MB over PCIe plus a 3.2 M-element scan,
and `common_sampler` then materializes a 202048-entry array *per block position*.
The kernel instead finishes with `ggml_argmax` plus a `get_rows` gather of the
winning probability, returning two 16-element tensors. Argmax is invariant under
softmax and the winning probability is exactly the confidence the executor
multiplies cumulatively.

On the RTX 3080 Laptop where this work was done, fusing the drafter moved decode
from 15.8-16.4 to 26.0-27.0 tok/s with the draft statistics
(`drafted=96 accepted=75`, 78.1% acceptance) byte-identical to the per-op
drafter — fusing changed the speed, not a single sampled token.
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

Sharing them brings 3.90 GB to 273 MB and two uploads per chunk. Measured at 64K
on the RTX 3080 Laptop: **prefill 353 -> 486 tok/s, peak VRAM 16059 -> 12671
MiB.** Sharing is safe because nothing writes a mask — `ggml_flash_attn_ext`
takes it as `src[3]` and `ggml_soft_max_ext` as `src[1]`, and both CUDA kernels
bind it `const`.

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
443.2 / 15.7; chunk 512 (ring 2816) gives 421.9 / 15.1 (all RTX 3080 Laptop,
IQ2_XXS). Decode is not KV-bandwidth bound here — both engines run at ~35-39% of
peak bandwidth because IQ2_XXS matvec is ALU-bound — so the smaller ring buys
nothing and the smaller chunk costs GEMM efficiency. llama.cpp's ring is smaller
(2560 rows) only because its default `n_ubatch` is 512; that is not an advantage
to copy.

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

TensorSharp allocates the SWA layers `pad(n_swa + chunk + 1, 256)` rows
(4352 at the default chunk) and indexes them by `position % rows`. At 64K that is
**1049 MB instead of 3.5 GB (29% of a uniform cache)**, and on the laptop card it
bought +56% prefill (226 -> 353 tok/s) and +15% decode (13.4 -> 15.4 tok/s) at 64K.

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

### The padded KV window has to be materialized on CUDA

ggml-cuda's flash-attention **vec** kernel — the one `ggml_cuda_get_best_fattn_kernel`
selects for a single query row, i.e. every decode step — returns wrong results when
K/V is a view whose `ne[1]` is a *truncated prefix* of a longer axis. A padded
attention window over a flat cache is exactly that shape: a full-attention layer
reads `[0, window_full)` rows out of a `cache_size`-row tensor, so the KV-head
stride jumps over the unread tail.

The failure is not subtle once you look at the tensor: on the first full-attention
layer (layer 3), **all 16 query heads sharing a KV head came back with the same
output vector**, wrong by up to 4.6 absolute. Recomputing the attention on the host
from the very `q`/`k`/`v`/`mask` tensors the kernel was handed reproduces the
contiguous result to 2e-6 — the inputs were right and the kernel was not. The error
then compounded through the remaining 49 layers into a different token stream: the
model stopped echoing the prompt and started stuttering (`请详细介绍详细介绍最终最终幻想幻想7`).

Three things kept it hidden:

* the **sliding-window layers read the whole ring** (`window == rows`), so their view
  is already contiguous — only the 13 full layers were affected;
* **prefill is fine**, because more than one query row selects the MMA kernel, which
  does honour the strides. The prefill logits matched the per-op path to 4e-4 and
  picked the same top-1 token, so the divergence only ever showed up from the first
  decode step onward;
* **Metal and the per-op path were both correct**, which made the fused CUDA path
  look like the odd one out for reasons that had nothing to do with this kernel.

`ggml_cont` on the window fixes it. The predicate must be "the window is a
sub-range of the cache", **not** `ggml_is_contiguous(k_full)`: `ggml_is_contiguous_n`
skips dimensions whose `ne` is 1, so with a single KV head — which is what every rank
holds under `--tp 2` — a truncated window reports itself contiguous and the guard
lets the bad shape straight through. `--tp 2` reproduced the identical wrong token
stream until the predicate stopped consulting `ggml_is_contiguous`.

Cost is not measurable: decode 40.5 → 40.5 tok/s and prefill 465 → 465 tok/s on a
50-token prompt, and 1159 → 1197 prefill / 38.2 → 37.7 decode on a 12371-token one
(2× RTX PRO 4000 Blackwell). The copy is only taken when the window is smaller than
the cache, which is also when it is cheap.

## 6. Tensor parallelism

`--tp 2` splits the model across two GPUs on the GGML CUDA / Vulkan backends.
Two is the ceiling for the 30B: it has **2 KV heads**, and no model in this repo
replicates KV heads when `num_kv_heads < tp`.

| Weight | Split |
|---|---|
| `attn_q` / `attn_k` / `attn_v` / `attn_gate` | column-parallel (by head) |
| `ffn_gate_up` | column-parallel, **per segment** — each half of the fused `[gate\|up]` is split independently |
| `attn_output` / `ffn_down` | row-parallel → AllReduce |
| all four per-layer norms, `attn_q_norm`, `attn_k_norm` | replicated (the QK norms are per-head `[headDim]` vectors; the Q norm also carries the folded `qk_scale_factor`) |
| `output_norm`, `token_embd`, `output` | replicated; the tail runs on rank 0 |

The attention output gate is column-parallel alongside Q and is applied **inside**
the per-rank region, before the row-parallel `o_proj`. Both AllReduce points land
on the raw matmul output, **before** the 1e-8 post-norms — RMSNorm is non-linear,
so reducing after it produces coherent-looking but wrong output. That is 104
reductions per step across 52 layers.

`TSGgml_MuseGlimmerModelForward` takes a `tp_degree` / `tp_plan_out` pair: in TP
mode it builds each rank's graph and returns a `TpRankPlan` instead of executing,
so the driver runs all ranks with the collectives at the segment boundaries. The
per-op TP path exists as a fallback but is far slower (the plan document measures
op-at-a-time TP at 0.01–0.04× single-GPU).

DFlash speculative decoding and pooled KV block snapshots follow the single-GPU
path only; multi-turn reuse under `--tp` comes from live-cache continuation.

### Two GPUs

Measured on **2× RTX PRO 4000 Blackwell 24 GB (PCIe)** — a different machine from
the single-GPU tables above, and not re-measured in the 2026-08-13 run. Prefill
512 / decode 64, best of 3:

| Model | | prefill tok/s | decode tok/s | GPU 0 | GPU 1 |
|---|---|---|---|---|---|
| 30B-UD-IQ2_XXS (10.2 GB) | `--tp 1` | 1171 | 40.2 | 9178 MB | — |
| 30B-UD-IQ2_XXS | `--tp 2` | **1569** (1.34×) | **63.2** (1.57×) | 5115 MB | 4063 MB |
| 30B-Q8_0 (28.2 GB) | `--tp 2` | 1691 | 34.3 | 15474 MB | 12748 MB |

The Q8_0 has no single-GPU row on that machine: 28.2 GB of weights does not fit
on one 24 GB card, so `--tp 2` is the only way to run it at all.

**Correctness.** `--tp 2` is byte-identical across repeat runs (which rules out a
race in the rank worker pool), and tracks the `--tp 1` greedy continuation for
468 of 500 characters before diverging at a paraphrase point with both
continuations coherent — the expected consequence of summing the row-parallel
partials in a different order.

## 7. Environment variables

| Variable | Effect |
|---|---|
| `TS_MUSE_GLIMMER_FUSED` | `0` = disable the fused whole-model kernel (per-op A/B) |
| `TS_MUSE_GLIMMER_PERSIST` | `0` = disable the persistent/capturable graph, rebuild every call |
| `TS_MUSE_GLIMMER_INGRAPH_EMBED` | `1` = force the in-graph embedding stage even for an untied LM head |
| `TS_MUSE_GLIMMER_DFLASH` | DFlash drafter GGUF path (same as `--draft-model`) |
| `TS_MUSE_GLIMMER_VENC_F32` | `1` = dequantize the vision tower to F32 (A/B; ~7.4 GB) |
| `TS_MUSE_GLIMMER_VENC_FUSED` | `0` = disable the CUDA fused vision-block/flash-attention path (diagnostic fallback) |
| `TS_MUSE_GLIMMER_GELU_TANH` | `1` = use the tanh GELU approximation in the tower instead of exact erf |
| `TS_MUSE_GLIMMER_VENC_TRACE` | `1` = per-stage checksums of the vision residual stream |
| `TS_MUSE_GLIMMER_LAYER_TRACE` | `1` = print a checksum of the residual *entering* every layer. Both the fused kernel and the per-op loop emit it, so diffing a fused run against a `TS_MUSE_GLIMMER_FUSED=0` run localizes a divergence to a layer |
| `TS_MUSE_GLIMMER_LAYER_TRACE_POS` | Trace the first forward at or past this absolute position (skips the startup warmup, whose prefill and decode run at unrelated positions) |
| `TS_MUSE_GLIMMER_LAYER_TRACE_N` | How many consecutive forwards to trace from there (default 1) |
| `TS_MUSE_GLIMMER_LAYER_TRACE_DIR` | Also dump each traced residual as raw F32 to `<dir>/{fused,perop}_S<step>_L<layer>.bin`, for an element-wise diff — the checksums alone hide a divergence that is small in aggregate |
| `TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS` | MLX lazy-graph flush interval (default 4, `0` disables) |
| `TS_PREFILL_CHUNK` | Prompt chunk size for `ForwardRefill` (default 2048) |
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | Tokens per prefill forward (default 2048, `0` disables chunking) |
| `TS_MUSE_GLIMMER_SWA_RING` | `0` = size every layer for the full context instead of ringing the SWA layers |
| `TS_MUSE_GLIMMER_SWA_ROWS` | Override the SWA ring size in rows (diagnostics) |
| `TS_DFLASH_FUSED` | `0` = disable the fused DFlash drafter (per-op A/B) |
| `TS_DFLASH_PERSIST` | `0` = rebuild the DFlash graphs every step instead of replaying |
