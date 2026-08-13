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

Two measurement passes are recorded here, and they answer different questions.

* **[Blackwell / Q8_0](#test-setup)** is the head-to-head against llama.cpp at
  the model's real quantization, on a card big enough that nothing is memory
  constrained. It is measured at commit `5098e3f`, i.e. **before** the
  optimization pass below, and it is what identified the three gaps that pass
  went after. The host went offline afterwards, so it has not been re-run.
* **[RTX 3080 Laptop / IQ2_XXS](#the-2026-08-13-optimization-pass)** is the
  before/after A/B for those optimizations, plus its own llama.cpp head-to-head.
  Different card, different quantization, so read it as a delta, not as a
  replacement for the table above it.

The engineering subsections further down keep their original A/B measurements
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

#### Why the TensorSharp column was a range and llama.cpp's is not

TensorSharp runs an **adaptive cost governor** in front of the drafter
([`MtpSpeculativeExecution`](../../TensorSharp.Runtime/Scheduling/MtpSpeculativeExecution.cs),
`AdaptiveSpeculation`, on by default). Speculation is only ever a speed
optimization, so the executor measures ms/token with drafting and without it,
and if drafting loses it **parks** the drafter before probing again. llama.cpp
has no such governor — it drafts every step, unconditionally.

On a 128-token run one bad probe therefore costs half the measurement. The two
16K reps below are the same prompt, the same binary and the same weights; they
differ only in the governor's verdict:

| 16K rep | drafted / accepted | verify steps | parked steps | decode |
|---|---|---:|---:|---:|
| 1 | 64 / 48 (75%) | 13 | **67** | 36.7 tok/s |
| 2 | 132 / 103 (78%) | 22 | 3 | **74.8 tok/s** |

Rep 1's *post-park* re-probe measured speculation at **14.0 ms/token against
37.8 ms/token plain** — 2.7x faster — so the verdict that parked it was wrong.

That was not bad luck; the estimator was **biased against speculation, one-sidedly
and by construction**. `GovernorRecord` averaged per-step *rates*
(`elapsedTicks / tokensEmitted`) with equal weight. A speculative step emits
between 1 and 16 tokens, so a fully rejected step contributed its whole cost as
one sample weighted exactly like a step that emitted 16; plain steps always emit
one token, so their mean of rates already equalled their aggregate and they were
unbiased. On a realistic 8-step probe — four steps emitting 16 tokens in 60 ms
and four emitting 1 token in 55 ms — the true figure is 6.8 ms/token and the old
estimator reported 29.4.

Three things compounded it: the round starts on the first decode step after a
prefill, so the sample is the least predictable tokens of the answer *and* pays
four one-off graph builds (the trunk verify shape, the drafter's `n_rows = 1`
inject, the draft block, and a possible KV-capacity grow that drops every
persistent graph); `SpecWinMargin` was 1.02, a 2% tolerance on an estimator whose
bias measured 2.7x; and one wrong verdict cost a flat 64 steps.

Fixed 2026-08-13, all in `MtpSpeculativeExecution`:

* the estimator is a **ratio of sums** (`sum(ticks) / sum(tokens)`) instead of a
  mean of rates;
* the **first sample of each side is discarded** every round — that removes all
  four cold-build sites at once without having to know which one fired, and it
  has to apply to the plain side too, because after a run of verify batches the
  1-row trunk graph is cold for the same reason;
* the **worst speculative sample of the round is trimmed**, which absorbs a
  mid-probe rebuild (`k` varies per step, so a probe can present several verify
  shapes, and 8 speculative steps can also carry the padded KV window across a
  256-row boundary);
* `SpecWinMargin` 1.02 → **1.15**, and the park **backs off** (16 steps for the
  first losing verdict, doubling to the old 64) so the least trustworthy verdict
  the governor ever makes — the one right after a prefill — is also the cheapest
  to get wrong;
* `Reset()` clears the verdict, so a park decided on one turn no longer leaks
  into the next request through a cached decoder;
* `Stats.ParkedSteps` counts only genuine parks (it used to include the three
  calibration plain steps, which is why a healthy run reported "parked 3").

The misfire reproduces on any machine. On the RTX 3080 Laptop below, the
**pre-fix** build parked 137 of 163 steps at a 4776-token prompt and 67 of 118 at
32K — in both cases while its own telemetry said speculation was faster
(55.8 vs 61.6 and 60.7 vs 125.7 ms/token). Post-fix the same points park 0.

#### The confidence floor

TensorSharp defaults to `confMin = 0.35`; llama.cpp's `p_min` defaults to 0 on
this path, i.e. it always drafts its full window. `--spec-draft-conf-min 0`
makes the two policies comparable and is **not** uniformly better: it wins at
501 and 128K tokens (180 vs 165, 60 vs 42) and loses badly at 16K and 32K
(33 vs 56, 30 vs 34), where acceptance collapses from ~75% to 24-42% and every
rejected row still occupies a verify slot. The floor wants to be adaptive, not a
constant — the same conclusion the earlier laptop run reached.

The 2026-08-13 pass re-ran that A/B on the RTX 3080 Laptop and got the same
split — the floor wins at a 4776-token prompt (32.2 vs 27.1 tok/s, acceptance
82.8% vs 23.2%) and loses at 16K (26.9 vs 31.2, acceptance 72.7% vs 27.4%) — but
the run was set up to answer a second question and did: `conf-min 0` always drafts
the full 15-token window, so the verify batch is `n_tokens = 16` on **every** step
and the persistent graph pool sees exactly one verify shape instead of up to
fifteen. If graph-shape thrash were what makes a speculative step expensive,
pinning the shape would have been faster at both points. It was not, so the
per-step cost is real work, not rebuilds — which is what
[what is left](#what-is-left) is about.

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

Almost all of the *growth* in that delta was the materialized KV window
(see [the section on it](#but-only-for-the-shapes-ggml-actually-hands-to-the-vec-kernel)):
the 26 copy destinations are 208 / 416 / 832 / 1593 MiB at 16K / 32K / 64K / 128K,
against +218 / +576 / +838 / +1552 MiB of measured growth over the 2K point.
Removing them leaves a roughly context-independent ~1.7 GB offset, which is a
separate question and not yet chased.

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

### The 2026-08-13 optimization pass

The Blackwell table above identified four gaps. Three of them turned out to be
bugs rather than costs, and the fourth is now the largest thing left.

| Gap in the Blackwell table | What it actually was | Where |
|---|---|---|
| decode 1.01x → 0.86x, monotone in KV length | 26 `ggml_cont` copies of the K/V window per decode step, guarding a ggml-cuda kernel that is **not selected** at those shapes | [narrowing the copy](#but-only-for-the-shapes-ggml-actually-hands-to-the-vec-kernel) |
| peak VRAM +1.3 → +3.3 GB | the same 26 copies, resident for the life of the persistent decode graph | same |
| DFlash decode bimodal (36.7 vs 74.8 tok/s on the same prompt) | the cost governor's estimator was biased against speculation and parked a drafter its own telemetry said was faster | [the governor](#why-the-tensorsharp-column-was-a-range-and-llamacpps-is-not) |
| DFlash prefill 0.66-0.69x against llama.cpp's 0.75-0.79x | the drafter shrank the **trunk** prefill chunk to 128, so a 124K prompt ran 980 trunk forwards where plain prefill runs 61 | [the prefill chunk](#the-dflash-prefill-chunk-drove-the-trunk-and-it-was-128) |
| prefill 0.92-0.94x above 16K | still open — see [what is left](#what-is-left) | |

A smaller item landed with them: the decode attention masks were regenerated on
the host and re-uploaded whole on every token
([details](#one-mask-per-attention-class-not-one-per-layer)).

#### Measurement setup for the before/after table

The Blackwell host that produced §5's table went offline, so the A/B was taken on
a different machine and a different quantization. Read it as a **delta**, not as a
replacement for the Blackwell numbers.

| | |
|---|---|
| GPU | 1x **NVIDIA GeForce RTX 3080 Laptop GPU**, 16,383 MiB, compute 8.6, driver 566.36 / CUDA 12.7. It throttles hard: ~990 MHz sustained SM clock under load |
| CPU / RAM | 16 cores / 32 GiB |
| Model | `Muse-Glimmer-30B-UD-IQ2_XXS.gguf` (10.0 GiB; 9178 MB of it device-resident) |
| Drafter | `dflash-kquant.gguf` (1.5 GiB) |
| TensorSharp | *before* = `5128d58`; *after* = the same tree plus this pass. Vendored ggml `8846b79`, `--backend ggml_cuda`, native library built `-DCMAKE_CUDA_ARCHITECTURES=86-real` |
| llama.cpp | master `9558fa44c` (2026-08-12), rebuilt the same day, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF` |
| Sampling | greedy on both sides |
| Generation | 128 tokens on the plain path, 256 on the DFlash path |
| Harness | [`.parity/bench_mg_win.sh`](../../.parity/bench_mg_win.sh), engines alternating within each context, VRAM sampled every 2 s |

**Read the DFlash rows with the card in mind.** Every TensorSharp DFlash
configuration peaks at 16.07-16.16 GiB of 16.38 — the card is full — while
llama.cpp's peak at 14.4-15.9 GiB. The plain rows have 1.4-3.3 GiB of headroom
and are the cleaner comparison.

#### Plain text generation

Ratios, because the card drifts as it heats: llama.cpp's own 64K decode moved
14.3 → 15.0 tok/s between the two passes on byte-identical input. The two engines
alternate inside each context, so a within-pass ratio shares a thermal state; the
absolute columns across passes do not. Peak VRAM is deterministic and is the
cleanest evidence.

| Prompt tokens | prefill TS/llama.cpp | decode TS/llama.cpp | peak VRAM, TS − llama.cpp (MiB) |
|---|---:|---:|---:|
| 60 | 0.86 → 1.06 | 0.92 → 0.92 | 2114 → 2110 |
| 4776 | 0.97 → 0.99 | 0.94 → 0.91 | 559 → 504 |
| 16322 | 0.96 → 0.93 | 0.88 → **0.91** | 762 → **544** |
| 32666 | 0.92 → 0.98 | 0.89 → **0.92** | 923 → **524** |
| 65359 | 0.94 → 0.93 | 1.08 → 0.95 | 1343 → **509** |

The VRAM column is the result: **−218 / −399 / −834 MiB** at 16K / 32K / 64K
against the 208 / 416 / 832 MiB of copy destinations the arithmetic predicts.

Decode follows at 16K and 32K (0.88 → 0.91 and 0.89 → 0.92, against ~2% and ~4%
predicted from the traffic removed). The 65359 row is not evidence in either
direction — llama.cpp's own decode there moved ±5% between passes on identical
input, and both engines sit within 1.4-1.9 GiB of filling the card.

Prefill is unchanged **by design**: the copy only ever fired on a single-row
decode. The wobble in that column (0.86 → 1.06 at 60 tokens, 0.96 → 0.93 at 16K)
is the thermal drift, not the change.

The 60-token row's ~2.1 GB VRAM delta is not the run: TensorSharp warms its
kernels with a 1024-token prefill at startup, and at a 60-token prompt that
warmup, not the generation, sets the process peak.

##### A same-binary A/B, and what it can and cannot settle

`TS_KV_FATTN_COPY=force` restores the unconditional copy inside the shipped
build, so both settings can be run from one binary in one session:

| Prompt tokens | prefill, forced → skipped | decode, forced → skipped | peak VRAM, forced → skipped |
|---|---:|---:|---:|
| 16322 | 578 → 543 | 17.7 → 15.2 | 11984 → 11790 (**−194 MiB**) |
| 32666 | 516 → 490 | 15.5 → 14.6 | 12473 → 12245 (**−228 MiB**) |
| 65359 | 456 → 454 | 14.5 → **14.9** | 13423 → 12610 (**−813 MiB**) |

The VRAM column agrees with the arithmetic (208 / 416 / 832 MiB predicted) and
with the cross-pass table above, at all three lengths.

**Read the throughput columns with the prefill column, which is the drift meter**
— the copy cannot touch prefill, because it only ever fires on a single-row
decode, so any movement there is the card. The two settings were run as *blocks*
rather than interleaved (a design mistake), and the drift is not uniform:

* at 16322 and 32666 prefill fell 6.0% and 4.9% between the blocks, which swamps
  the effect being measured — the decode deltas there (−14%, −6%) carry no
  information;
* at 65359 prefill moved **0.3%**, i.e. those two runs share a thermal state, and
  the decode delta is **+2.8% in favour of skipping the copy**.

+2.8% at 64K is the same order as the arithmetic predicts (1.72 GB of removed
traffic against a ~67 ms token). The larger effect the Blackwell host should see
at 128K — 3.3 GB per token — is not measurable here.

#### DFlash speculative decoding

Two changes land here and they have to be read separately, because raising the
trunk prefill chunk changes the prefill GEMM tiling, which flips a greedy tie and
produces a **different continuation**. Acceptance — and therefore DFlash
throughput — is a property of the text being generated, so the before and after
*decode* columns at the new default are not scoring the same run.
`TS_DFLASH_PREFILL_CHUNK=128` pins the old numerics back, which is how the
governor is isolated further down.

**Prefill is the clean comparison**: same prompt, same token count, and the chunk
change is exactly what it measures. Each cell is that engine's own plain prefill →
its DFlash prefill, tok/s, in the format §5's Blackwell table uses.

| Prompt tokens | llama.cpp | TensorSharp before | TensorSharp after |
|---|---:|---:|---:|
| 60 | 221 → 178 (0.81x) | 261 → 187 (0.72x) | 234 → 171 (0.73x) |
| 4776 | 600 → 520 (0.87x) | 603 → 493 (0.82x) | 593 → 554 (**0.93x**) |
| 16322 | 595 → 531 (0.89x) | 585 → 476 (0.81x) | 552 → 512 (**0.93x**) |
| 32666 | 510 → 481 (0.94x) | 536 → 422 (0.79x) | 499 → 478 (**0.96x**) |
| 65359 | 476 → 443 (0.93x) | 471 → 387 (0.82x) | 441 → 442 (**1.00x**) |

TensorSharp now pays a *smaller* prefill penalty for speculation than llama.cpp
does at 4776 and 16322 tokens, and the same at 32666 — where the Blackwell run
had it paying 0.66-0.69x against llama.cpp's 0.75-0.79x.

**Decode**, with the governor's own park counter alongside. Where the old
governor parked, un-parking is worth 1.3-1.8x; where it did not park (16322), the
number moves with the continuation rather than with the engine.

| Prompt tokens | llama.cpp | TS before | TS after | parked steps before → after |
|---|---:|---:|---:|---|
| 60 | 28.1 / 26.5 | 17.9 | 15.4 | 69 → 0 |
| 4776 | 44.3 / 43.1 | 20.3 | **36.4** | 137 → 0 |
| 16322 | 54.1 / 51.2 | 34.4 | 27.2 | 3 → 0 |
| 32666 | 35.7 / 33.5 | 17.9 | **27.7** | 67 → 0 |
| 65359 | 44.1 / 43.4 | 18.0 | **23.8** | 70 → 0 |

(llama.cpp is given as *before pass / after pass* to show the run-to-run spread on
this card; it is unaffected by any of these changes.)

**The governor in isolation.** Same binary, `TS_DFLASH_PREFILL_CHUNK=128` so the
prefill numerics — and therefore the generated tokens — match the pre-fix run
exactly. The only variable is the estimator:

| Prompt tokens | parked steps | drafted / accepted | decode tok/s |
|---|---|---|---:|
| 60 | 69 → **0** | 142/105 → 185/138 | 17.9 → **21.8** (+22%) |
| 4776 | 137 → **0** | 129/93 → 224/177 | 20.3 → **29.9** (+47%) |
| 16322 | 3 → 0 | 218/197 → 217/197 | 34.4 → 34.2 (unchanged) |
| 32666 | 67 → **0** | 161/138 → 244/193 | 17.9 → **29.9** (+67%) |
| 65359 | 70 → **0** | 187/140 → 278/194 | 18.0 → **24.1** (+34%) |

Every point the old estimator parked, it parked wrongly, and every one of them
gains 22-67%. The 16322 row is the control: it is the one context the old
governor did *not* park, and there the new estimator changes nothing — same
drafts, same acceptances, same throughput. Which also identifies the apparent
16322 regression in the decode table above as what it is: the trunk chunk moving
128 → 1024 flipped a tie and the run is generating different text, not decoding
the same text more slowly.

The prefill column confirms the two runs are the same configuration
(187 → 187, 493 → 472, 476 → 483, 422 → 419, 387 → 379 tok/s; the residual is
drift).

**Single-rep DFlash decode on this card is worth ±40%, so weigh the draft
statistics over the tok/s.** The 60-token point makes it plain: the `after` and
`after128` runs did *byte-identical* work — same continuation, `drafted=185
accepted=138` in both — and came out at 15.4 and 21.8 tok/s. The card is
thermally saturated and every DFlash configuration is inside 300 MiB of filling
it. Park counts and draft/accept counts are stable; a single decode figure is
not. (The Blackwell host has the same disease at a lower rate — "one run in
twenty ~40% slow" in [the method notes](#benchmark-method-notes).)

**Short prompts remain the weakest point for TensorSharp here.** Even the good
60-token run (21.8) is below llama.cpp's 26.5, and the 15.4 run is below
TensorSharp's own plain decode of 19.1. The governor is neither the cause nor the
cure: its choice is between "draft" and "don't draft *but still catch the
drafter's ring up*", and it now picks correctly inside that choice (49.0 against
94.0 ms/token). What is left over is the catch-up plus the verify overhead — the
[per-step cost](#what-is-left).

#### How the pass was verified

* **The copy narrowing is numerics-neutral.** A 16322-token prompt generating 128
  tokens produced a **byte-identical** continuation with `TS_KV_FATTN_COPY=force`
  and with the copy skipped, and so did a 60-token prompt at 96 tokens. Same
  binary, same weights, only the predicate differs.
* **The mask change rides on the same check** — it is on the decode replay path,
  so those two runs exercise it 128 and 96 times each.
* **The plain-versus-DFlash divergence on this model is pre-existing.** At 60
  prompt tokens the DFlash continuation splits from the plain one at character
  285 ("mention *different variants*" versus "mention *types of gradient
  descent*") — **at the identical position, with the identical tokens, on both
  the pre-fix and the post-fix build**. It is the tie flip §5 already describes
  (the verify batch's GEMM has a different shape from the 1-row decode GEMM), it
  shows up earlier on IQ2_XXS than on Q8_0, and nothing in this pass touches
  verification, rollback or acceptance. The governor change does make it *more
  visible*, because a drafter that is no longer parked runs more verify batches.
* **The golden parity suite could not be run on this box.** Every
  `MuseGlimmerParityTests` case that loads a model fails in the test host with
  "Failed to initialize ggml-cuda. A different GGML backend was already
  initialized in this process" — 170 ms in, before any of this code runs, and
  including when the class is run entirely alone. That is the pre-existing,
  order-dependent harness problem recorded in the repo notes, not a regression;
  the 44 MuseGlimmer tests that do not construct a GGML context all pass. The
  suite still needs to be run somewhere it works before this pass is considered
  fully validated.

#### What is left

* **Prefill above 16K is still 0.92-0.96x of llama.cpp**, and none of this pass
  touched it — the copy it removed fires only on a single-row decode. Two
  candidates, neither yet isolated: TensorSharp rebuilds the whole 52-layer graph
  per prefill chunk (~1700 nodes, a gallocr re-plan and ~730 weight bindings; at
  61 chunks over a 124K prompt that is 1-3% if a rebuild costs 20-50 ms), and its
  SWA ring is 4352 rows where llama.cpp's iSWA cache is `n_swa + n_ubatch` = 4096
  at the same `-ub 2048`, i.e. 6% more attention work in 39 of the 52 layers.
* **A speculative step still costs ~2.7x a plain step** where the extra rows alone
  predict ~1.1x on a dense model (both read the same 10.2 GB of weights). The
  `conf-min 0` probe rules out graph-shape thrash. What is left to check: the
  12.9 MB `[vocab, 16]` verify-logits download per step, the 808 KB
  `new float[vocab]` allocated per step (over the LOH threshold), and
  `MtpCatchUp`'s 130 KB-per-row host round trip on unpinned managed arrays.
* **The confidence floor still wants to be adaptive.** The governor already owns
  the right objective (ms per emitted token) and the timing plumbing; making
  `MinDraftProb` a 4-arm bandit over `{0, 0.15, 0.35, 0.6}` re-explored on context
  growth is the shape that fits.
* **A context-independent slice of the VRAM delta is unexplained** once the copy
  buffers are subtracted — ~1.7 GB on the Blackwell Q8_0 run, ~0.5 GB on the
  RTX 3080 IQ2_XXS run.
* **The permanent version of the copy fix is a KV relayout.** llama.cpp stores K/V
  token-major with the heads interleaved inside a row (`[n_embd_kv_gqa, kv_size]`),
  so `nb[1]` and `nb[2]` of the attention view are constants and truncating the KV
  axis changes only `ne[1]` — no stride ever encodes the allocation size, and no
  copy is needed on any kernel at any length. TensorSharp is head-outer
  (`[head_dim, cache_rows, kv_heads]`), which is what makes a partial window a
  holed view in the first place. Changing it would let
  `kv_window_needs_cuda_flash_attn_copy` be deleted outright across all seven
  fused model kernels, at the cost of touching the per-op path, the KV snapshot,
  the TP sharding and the DFlash ring together.

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

#### The DFlash prefill chunk drove the TRUNK, and it was 128

`MtpPrefillChunkSize` sets the width of the *trunk* forward the speculative
executor uses for prompt prefill, not just the drafter's. It was a hardcoded 128,
chosen to bound the host-side capture buffer — one captured row is
`DFlashConfig.FeatureSize` = 33280 floats = 130 KB, and `PrefillStep` kept two
chunk-sized `float[]`s, so a 2048-row chunk would have been 2 × 545 MB. The
comment justifying it said "the trunk's per-chunk fixed overhead is small next to
a 128-row Muse-Glimmer forward".

It is not. Differencing the plain and DFlash prefill times in
`bench_mg_blackwell_q8.csv` gives the cost of one *extra* chunk directly, and it
is a flat **~60 ms from 2K to 128K of context** — a full 52-layer graph rebuild
(a 128-row forward is over `kMgMaxPersistTokens = 64`, so it never persists) plus
a DFlash host round trip, against roughly 13 ms of useful work inside a 128-row
chunk. At a 124K prompt that is 980 trunk forwards where plain prefill does 61:
918 extra forwards × ~64 ms = 58 s piled onto a 112 s prefill, which is the whole
of the 0.69x DFlash prefill ratio. llama.cpp never shrinks its target batch for
DFlash — it runs the drafter as a hook over the same `-ub 2048` batch the target
just decoded — which is the entire reason its ratio plateaus at 0.75-0.79.

Two changes, both 2026-08-13:

* `PrefillStep` builds the (token *k*, hidden of *k-1*) pairing **in place** —
  save the last row, slide the block down one, prepend the carry-in — so only one
  chunk-sized buffer exists instead of two.
* the default chunk is **1024** (`TS_DFLASH_PREFILL_CHUNK`), which is one 130 MB
  host buffer and removes 87% of the extra chunks. It is clamped to what the two
  rings can absorb in a single forward: the drafter's is
  `DFlashConfig.RingRows` = 2080 and the trunk's SWA ring refuses anything wider
  than `rows - n_swa` = 2304.

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
  `decode_input_set_async`, which is the CUDA-capture-safe path in.
  *Follow-up, 2026-08-13:* "already cheap at one row" was only true at short
  context. The decode replay regenerated **both** masks from scratch every token —
  at 128K that is a 250 KB `std::vector` allocated, filled by a scalar loop over
  125 K entries and re-uploaded, per token, on the critical path. The two staging
  buffers now live in the `MgDecodeCache` entry (allocated once per graph), and the
  full-causal one is **extended, not regenerated**: for a 1-row decode its content
  is a prefix of zeros that grows by exactly one entry per token, so the step
  writes one `fp16` and uploads two bytes at an offset. The SWA ring mask is still
  regenerated whole — 4352 entries, and its liveness is not monotone in position.
  Any other shape (a verify batch, a rollback that moves `start_pos` backwards, a
  rebuilt window) falls back to the full path.
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

Cost is not measurable *at the length it was measured at*: decode 40.5 → 40.5 tok/s
and prefill 465 → 465 tok/s on a 50-token prompt, and 1159 → 1197 prefill /
38.2 → 37.7 decode on a 12371-token one (2× RTX PRO 4000 Blackwell).

#### …but only for the shapes ggml actually hands to the vec kernel

That conclusion did not survive long context, and the reason is that the copy was
being taken in a regime where the kernel it guards against is **never selected**.
`ggml_cuda_get_best_fattn_kernel` reaches its vec branch for an unquantized K/V
only when

```c
cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1
    && !(gqa_ratio > 4 && K->ne[1] >= 8192)          // fattn.cu
```

and the `!gqa_opt_applies` escape one line below it cannot fire for this model
(GQA 16, a mask, `max_bias == 0`, a 256-aligned window, every `nb` 16-aligned).
Muse-Glimmer's `gqa_ratio` is 32/2 = 16, so **every window of 8192 rows or more
goes to `BEST_FATTN_KERNEL_MMA_F16`** — the same stride-honouring kernel prefill
has always used — and on Turing/Ampere the vec branch is unreachable at *any*
length. The fault that motivated the copy was measured at a 256-row window, where
vec really is selected and the copy costs 6.5 MiB.

Taken unconditionally, the copy is 26 `ggml_cont` nodes (13 full layers × K and V)
on every decode step, each the size of the whole window:

| context | window | copied per token | extra HBM traffic | resident in the decode graph |
|---:|---:|---:|---:|---:|
| 16126 | 16128 | 215 MB | 0.43 GB | 208 MiB |
| 32274 | 32512 | 433 MB | 0.87 GB | 416 MiB |
| 64575 | 64768 | 862 MB | 1.72 GB | 832 MiB |
| 123931 | 124160 | 1653 MB | 3.31 GB | 1593 MiB |

The right-hand column is the part that hurt twice: the persistent decode graph is
allocated with `ggml_backend_alloc_ctx_tensors`, one slot per tensor and no
lifetime packing (that is what keeps the addresses stable for the CUDA capture),
so all 26 destinations are resident for the life of the graph. It matches the
measured TensorSharp-minus-llama.cpp peak-VRAM *growth* almost exactly: +218 MiB
of measured growth from 2K to 16K against 191 MiB of copy buffers, +838 against
815 at 64K, +1552 against 1576 at 128K.

`kv_window_needs_cuda_flash_attn_copy` therefore now takes the flash-attention op's
`gqa_ratio` and mirrors ggml's own selection: with `gqa_ratio >= 2` it skips the
copy on Turing/Ampere outright, and on Ada and newer for any window ≥ 8192 rows.
Callers that pass no ratio (every other model's kernel, for now) keep the old
unconditional behaviour. `TS_KV_FATTN_COPY=force` pins the copy back on if an
upstream ggml bump ever moves that heuristic, and `TS_KV_FATTN_COPY=0` still
disables it entirely for the A/B that reproduces the original fault.

The clause being mirrored is a ggml tuning heuristic, so it is worth re-checking on
an `ExternalProjects/ggml` bump. The A/B is cheap: a 16322-token prompt generating
128 tokens produced a **byte-identical** continuation with the copy forced and with
it skipped (RTX 3080 Laptop, IQ2_XXS).

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
| `TS_DFLASH_PREFILL_CHUNK` | Tokens per **trunk** forward while a DFlash prefill catches the drafter up (default 1024, was a hardcoded 128). Clamped to what the drafter's ring (2080 rows) and the trunk's SWA ring (`rows - n_swa` = 2304) can absorb in one forward |
| `TS_KV_FATTN_COPY` | `0` = never materialize a padded KV window (reproduces the ggml-cuda flash-attention **vec** fault; diagnostic only). `force` = always materialize it, i.e. the pre-2026-08-13 unconditional behaviour, for use if an `ExternalProjects/ggml` bump moves the kernel-selection heuristic this now mirrors |
