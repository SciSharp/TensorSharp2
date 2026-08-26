# Speculative Decoding in TensorSharp

Speculative decoding is a **speed** optimization and nothing else. A drafter
guesses the next few tokens, the trunk verifies them all in one batched forward,
and every emitted token is still drawn from a trunk row — so the output is what
plain decoding would have produced. A wrong guess costs a rollback, never a
wrong token.

This document describes how that is *built* in TensorSharp, and what you have to
write to add a new model or a new algorithm.

## The three layers

The design rests on one distinction:

```
   Model architecture      !=      Speculation algorithm      !=      Speculator weights
   ISpeculativeTarget              ISpeculator                        IDraftHead implementation
```

Conflating these is what makes speculative decoding hard to extend. They have
genuinely different lifetimes:

| Layer | What it is | Transfers between models? |
| --- | --- | --- |
| **Algorithm** | MTP / EAGLE / DFlash / DSpark / n-gram, as *code* | Yes — write once |
| **Runtime** | draft → verify → accept → rollback → commit | Yes — write once |
| **Weights** | the trained drafter in a checkpoint | **No** — bound to one target model |

A learned drafter reads the target's hidden states, borrows its tokenizer,
embedding and LM head, and is trained against its representation space. Even
when two models happen to share a hidden size, `h_Qwen != h_Gemma`. So the
weights are a model-specific artifact behind a thin adapter, exactly like a LoRA
checkpoint — while the framework and the runtime around them are not.

## The pieces

Everything lives in `TensorSharp.Runtime/Speculative/`.

### Layer 1 — the target model adapter (`ISpeculativeTarget`)

What the shared loop needs from the trunk, and nothing about drafting:

```csharp
void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows);
void SpecEnsureCapacity(int requiredSeqLen);
void SpecSnapshotRecurrentState();
void SpecRestoreRecurrentState();
void SpecRewindCache(int length);
int  CacheSeqLen { get; }
int  MaxContextLength { get; }
```

Two capabilities together make verification possible: a multi-row forward with
per-row logits (so a whole draft window is checked for roughly the cost of one
decode step, optionally tapping each row's hidden state), and a way to undo the
rejected tail.

`IBatchedSpeculativeTarget` adds the same thing over the batched paged path, so
the speculative trunk can run on the same kernels as the non-speculative batched
baseline and compose with prefix caching.

`ISpecTrunk` is the small seam that lets one loop serve both KV regimes:
`LinearSpecTrunk` (the model's live linear cache) and `BatchedSpecTrunk` (paged
KV + per-slot recurrent state, in `BatchExecutor`).

### Layer 2 — the algorithm (`ISpeculator`)

```csharp
int  Propose(in DraftContext ctx, List<int> draftOut);
void Commit(int[] tokens, float[] hRows, int startPos);
void Reset();
```

`Propose` guesses; `Commit` tells the speculator what actually landed in the
trunk, with the exact hidden states, so a learned drafter's KV cache and a
lookup drafter's corpus both track reality. `DraftContext` is one struct rather
than a parameter list precisely so a future algorithm that needs a new signal
gets a new field instead of a new overload everywhere.

Shipped implementations:

| Name | Class | Weights | Notes |
| --- | --- | --- | --- |
| `draft-head` | `DraftHeadSpeculator` | required | One token per pass, chaining its own hidden output: NextN/MTP (Qwen 3.6, GLM 5.2, Gemma 4's separate assistant GGUF). EAGLE-shaped heads fit here unchanged. |
| `block` | `BlockDraftSpeculator` | required | A whole block per pass with a confidence head: DeepSeek V4 DSpark, DFlash and DFlash2 (Muse-Glimmer, Qwen 3.8). |
| `ngram` | `NGramSpeculator` | **none** | Suffix matching over the sequence's own tokens (prompt-lookup decoding). Works on every model. |
| `auto` | — | — | Default: use whatever drafter the checkpoint carries. |

### Layer 3 — the weights (`IDraftHead`)

The model-specific adapter over a trained drafter:

```csharp
DraftHeadKind DraftHeadKind { get; }              // None | PerToken | Block
int  DraftBlockSize { get; }
void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut);
int  DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut);
void DraftCatchUp(int[] tokens, float[] hRows, int startPos);
```

A model with no drafter reports `DraftHeadKind.None` and still gets every
weight-free algorithm.

### The shared runtime

`SpeculativeExecution` is the one implementation of the protocol, for every
algorithm and every model:

1. **Draft** — `ISpeculator.Propose`. The loop does not know or care how.
2. **Verify** — the trunk forwards `[lastToken, d1..dK]` as ONE batch with
   per-row logits; the caller's sampler draws each row and drafts are accepted
   while the drawn token matches. Row *m*'s drawn token is the corrected (or
   bonus) token for free.
3. **Rollback** — on partial acceptance, restore the recurrent snapshot and
   re-advance over the kept prefix. Trunks whose verify already persisted usable
   KV (`SpecVerifyPersistsAcceptedKv`) skip the re-forward and just rewind the
   position — the dominant rollback cost on long contexts.
4. **Commit** — kept tokens go back to the speculator with their exact hidden
   states.

`SpeculationCostGovernor` sits alongside it, not inside it: speculation must
never make decoding slower, so the governor measures speculative against plain
steps at runtime and parks drafting while it is losing, re-probing with backoff.
It is a measurement *policy*, kept separate because it is the piece most likely
to be tuned or switched off per deployment.

`SpeculatorRegistry` maps a name to a factory. It is the only place that knows
which algorithms exist.

## Adding a new speculation algorithm

Write the class and register it. No model, executor or scheduler code changes.

```csharp
public sealed class MedusaSpeculator : ISpeculator
{
    public string Name => "medusa";
    public int MaxDraftTokens { get; }
    public float MinDraftProb { get; set; }
    public float DefaultMinDraftProb => 0.6f;
    public bool NeedsHiddenState => true;
    public bool HandlesOwnPrefill => false;

    public int Propose(in DraftContext ctx, List<int> draftOut) { /* ... */ }
    public void Commit(int[] tokens, float[] hRows, int startPos) { /* ... */ }
    public void Reset() { }
    public void Dispose() { }
}

SpeculatorRegistry.Register("medusa",
    (target, options) => target is IDraftHead h && h.DraftHeadKind == DraftHeadKind.PerToken
        ? new MedusaSpeculator(h, target.Config.VocabSize, target.SpecFeatureSize, options.MaxDraftTokens)
        : null,
    requiresDraftHead: true);
```

Return `null` from the factory when the algorithm cannot serve that model; the
registry turns it into an operator-facing decline reason. `requiresDraftHead`
lets the execution planner explain "no draft head" up front instead of routing a
request onto a path that will bail.

Correctness is **not** the algorithm's responsibility. Whatever it proposes,
verification emits only tokens drawn from a trunk row. A speculator is free to
be wrong; it must not be slow.

## Adding a new model

Implement `ISpeculativeTarget` on the model — the multi-row forward plus the
rollback trio — and it can immediately be sped up by every weight-free
algorithm. If the checkpoint also ships a drafter, implement `IDraftHead` on the
same class (there is an `ISpeculativeModel` alias for the pair) and report the
matching `DraftHeadKind`.

One caveat worth stating: some models' `SpecForward` is not drafter-independent
— Gemma 4, Muse-Glimmer and DeepSeek V4 share the fused verify kernel with their
drafter and refuse to run without it. Those report `SpeculationProfitable` as
false when no drafter is loaded, so weight-free speculation is declined rather
than crashed. Qwen 3.5/3.6 and GLM 5.2 have drafter-independent trunks and
accept `--spec-type ngram` on any checkpoint.

## Operator surface

```
--spec | --no-spec              enable/disable speculative decoding
--spec-type <name>              auto (default) | draft-head | block | ngram
--spec-draft <N>                max tokens drafted per step (1-64, default 8)
--spec-pmin <f>                 confidence gate (default: per algorithm)
--spec-draft-model <path>       separate draft-head GGUF (Gemma 4)
--draft-model <path>            block drafter GGUF resident before the layer split
```

The historical `--mtp-spec`, `--mtp-draft`, `--mtp-pmin` and `--mtp-draft-model`
spellings are accepted unchanged. Environment variables are published under both
`TS_SPEC_*` and `TS_MTP_*`, and that is not merely for compatibility: the glm-dsa
**native** loader reads `TS_MTP_SPEC` and `TS_MTP_DRAFT` from C++ while the model
is loading — it decides whether to page a whole extra 256-expert decoder layer
into VRAM, and sizes its graph cache — so those names are a cross-language
contract.

`--spec-pmin` means something different per algorithm, which is why each brings
its own default rather than sharing one: `0.75` for a per-token head (top-1
probability over its top-10 logits), `0.35` for a block drafter (the CUMULATIVE
prefix probability, so the same number is far stricter), `0` for n-gram (where it
scales the required match length instead).

## DFlash and DFlash2

A **DFlash** drafter is a small block-diffusion model that ships as its own GGUF
(`general.architecture = dflash`) and is bound to one target. It reads the
target's own residuals rather than only its tokens, and it proposes the whole
speculative window in ONE forward pass instead of one token at a time:

```
PASS A  encoder      feat = concat(target residual entering dflash.target_layers)
                     g    = rmsnorm(fc @ feat, enc.output_norm)
PASS B  KV inject    K = rope_neox(headnorm(attn_k @ g)) ; V = attn_v @ g
                     ring[pos % ringRows] <- K, V        (no Q, no attention, no FFN)
PASS C  block draft  ids = [anchor, MASK x (block_size-1)]
                     -> draft blocks -> the TARGET's LM head -> block_size-1 drafts
```

The drafter owns a small sliding-window KV ring of its own, sized from
`dflash.attention.sliding_window`; the target's KV cache is untouched. Everything
the drafter needs beyond its own blocks - the token embedding and the LM head -
is borrowed from the target, so the file is ~1-3 GB against a 27-30B trunk.

**DFlash2** is the same backbone with two additions, both keyed off the GGUF, so
one code path serves both generations:

* **A grouped dynamic depthwise convolution** around every attention and every
  FFN sublayer (`dflash.conv_kernel_size`, `dflash.conv_group_size`). One
  projection of the sublayer's INPUT produces both the filter applied to that
  input and the filter applied to the sublayer's OUTPUT. Tap *t* of channel *c*
  at block position *r* is `base[t][c] + delta[r][t][c / group_size]` - static
  per channel, dynamic per group - multiplying `x[r-t][c]`, and masked to zero
  for `r < t` so the filter never reaches across a block boundary. It is what
  gives a block-diffusion draft a local left-to-right signal without a second
  forward pass.

* **A candidate selector** (`dflash.selector_rank`, `dflash.selector_top_k`).
  Plain DFlash takes each block position's argmax over the vocabulary
  INDEPENDENTLY - exactly the weakness of block diffusion, since position *i+1*
  is chosen without knowing what *i* chose. The selector keeps the top-K
  candidates per position and scores every (predecessor, candidate) pair through
  two low-rank `[vocab, r]` codebooks:

  ```
  score[e][p][c] = unary[e][c] + < A[pred[e][p]] * (P h_e) , B[cand[e][c]] >
  ```

  `A`/`B` are `selector_predecessor`/`selector_successor`, `P` is
  `selector_hidden`, `pred[0]` is the verified anchor token and `pred[e]` is
  `cand[e-1]`. The block is then read off as a greedy walk through that lattice:
  one small matmul per position, no extra draft forward.

  `unary` is the target LM head's logit for that candidate **after the target's
  own logit transform** (`dflash.logit_scale`, `dflash.final_logit_softcapping`),
  which is why those keys exist on a DFlash2 file at all. Plain DFlash takes an
  argmax and is invariant to both; the lattice ADDS the unary term to a
  transition score, so an untransformed unary is simply the wrong size and
  swamps the transition it is meant to compete with. Skipping it on the
  Muse-Glimmer drafter (scale 0.196, softcap 20) cost more than half the
  acceptance rate.

Both extensions are no-ops when their keys are absent, so a first-generation
DFlash file runs through the same code unchanged. `TS_DFLASH_SELECTOR=0` and
`TS_DFLASH_CONV=0` switch one off for attribution; neither is a supported way to
run a model, since the weights were trained with both.

### Where it runs

Both passes are one fused GGML graph each (`ggml_ops_dflash.cpp`,
`TSGgml_DFlashInject` / `TSGgml_DFlashDraftBlock`) on CUDA, Vulkan and Metal,
with a persistent graph that ggml-cuda can capture and replay; the per-op
managed drafter is the fallback and the reference the fused path is checked
against. `TS_DFLASH_FUSED=0` forces it.

The selector's lattice comes back to the host as `k + k*k*(gamma-1)` floats
(~7 KB) rather than the `[vocab, block]` block a naive readback would move
(12.9 MB), and the walk itself - inherently sequential, tiny - runs on the host.

### Attaching one

`--draft-model <path>` (or `--spec-draft-model`, or `TS_QWEN35_DFLASH` /
`TS_MUSE_GLIMMER_DFLASH`). The file's `general.architecture` decides what it is,
not its name. A target that already carries a NextN/MTP block (Qwen 3.8 does)
uses the DFlash drafter instead when one is attached: they consume different
hidden rows and drive different speculators, and the operator named the file
explicitly.

### What the target has to provide

Only the residual tap. A DFlash target implements `SpecForward` so that, per
row, it also writes the concatenated residuals ENTERING each layer in
`dflash.target_layers` - `SpecFeatureSize` wide instead of one hidden. Both
shipped targets do it inside their fused whole-model kernel (a `ggml_cpy` per
tapped layer), so speculation does not force the op-by-op loop.

### What to expect

Measured on one RTX 3080 Laptop (16 GB), greedy, best of two runs. Two prompts,
because acceptance - and therefore everything - depends entirely on how
predictable the continuation is: a free-form "explain how a GPU does a matmul"
(prose) and a "list the first 20 primes" (factual).

| target | drafter | prose tok/s | factual tok/s |
| --- | --- | ---: | ---: |
| Muse-Glimmer 30B IQ2_XXS | none | 18.7 | - |
| Muse-Glimmer 30B IQ2_XXS | DFlash (1.6 GB) | 25.4 (1.36x) | - |
| Muse-Glimmer 30B IQ2_XXS | DFlash2 Q4_K_M (1.6 GB) | 23.0 (1.23x) | - |
| Muse-Glimmer 30B IQ2_XXS | DFlash2 Q8_0 (3.0 GB) | 14.1 (0.75x) | - |
| Qwen 3.8 27B IQ3_XXS | none | 17.9 | 17.1 |
| Qwen 3.8 27B IQ3_XXS | NextN/MTP | 20.4 (1.14x) | 30.1 (1.76x) |
| Qwen 3.8 27B IQ3_XXS | DFlash2 Q4_K_M | 15.3 (0.85x) | 25.8 (1.51x) |
| Qwen 3.8 27B IQ3_XXS | DFlash2 Q4_K_M, `--spec-draft 7` | 9.8 (0.55x) | 23.5 (1.37x) |

The four Qwen rows are one uninterrupted sweep, so they are comparable to each
other; the Muse-Glimmer rows are from a separate one and are not comparable to
them in absolute terms.

Treat the absolute numbers as indicative, not exact. On this laptop card a plain
decode - which does identical work per token whatever the prompt - measured 18.7
and 16.5 tok/s in two back-to-back runs of the same binary. Anything under about
15% apart on a single run is noise here; the comparisons below that matter were
all made as paired runs, alternating the configurations inside one batch.

That caveat is not theoretical: an earlier revision of this page reported DFlash2
on the prose prompt at 20.9 tok/s (1.14x), and it does not reproduce. Repeated
paired runs put it at 0.85-0.96x - break-even at best on free-form prose - while
the plain baseline measured beside them barely moved. The factual rows and the
MTP rows did reproduce. Believe the ratios, re-measure before believing a
single-run figure, and do not compare a number here against one taken on another
day.

### Against llama.cpp

llama.cpp b10630 on the same files and card: Muse-Glimmer plain 19.7 / DFlash
22.0. It cannot load a DFlash2 drafter at all - it rejects the file with "wrong
number of tensors; expected 81, got 58", the 23 convolution and selector tensors
it has no code for - so on DFlash2 there is nothing to compare against.

On MTP there is, and getting it right took two corrections. A first pass ran the
two engines on the same prompt without noticing that llama.cpp turns thinking
mode ON by default for this checkpoint and TensorSharp does not, so they were
answering with different continuations; since acceptance is a property of the
continuation, that measured the text rather than the engine. The numbers below
are a true like-for-like: same prompt, thinking disabled on both
(`chat_template_kwargs: {"enable_thinking": false}`), greedy, 256 tokens, draft
window 3.

| | tokens/accept call | acceptance | tok/s | ms/step |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp `draft-mtp` | 3.63 | 0.885 | 39.4 | 92.1 |
| TensorSharp `--spec` | 3.67 | 0.932 | 33.9 | 97.4 |

**Drafting is at parity or better** - TensorSharp gets slightly more tokens per
verify call than llama.cpp does. The gap is entirely per-step cost, and
TensorSharp's own phase counters locate it: a verify is 77 ms (llama.cpp's works
out to about the same), and the draft calls are 20 ms against roughly 13.

One caution about llama.cpp as a yardstick: its eval time reproduces to within
0.04% run to run on this card (2886.84 ms and 2885.74 ms on two identical
requests), where TensorSharp swings by several percent. The variance is
TensorSharp's, not the machine's.

#### What was eliminated

llama.cpp runs its MTP block ONCE over `n_accepted + 1` rows, folding the
catch-up over the accepted tokens and the first draft step into a single call.
TensorSharp ran a catch-up and then a separate first `DraftStep`, and on a head
whose per-call cost is mostly fixed that extra call was the largest single
difference. It now folds too (`SupportsFusedCatchUpStep` /
`DraftCatchUpAndStep`, `TS_MTP_FOLD_CATCHUP=0` to revert): `catchUpMs` 191 -> 0,
worth +4.0% at 256 tokens and +5.3% on prose, with byte-identical output and
unchanged acceptance.

#### What is left, measured

Three things were checked and are NOT the problem, which is worth recording
because each looks like an obvious suspect:

- **CUDA-graph capture of the verify.** `TS_GGML_LOG_DEBUG=1` surfaces ggml's
  "CUDA graph warmup complete"/"reset" lines. Capture does churn (the persist
  cache evicts across draft shapes), but raising
  `TS_Q35_VERIFY_CACHE_BUDGET_MB` from 1536 to 3072 halves the resets and
  changes throughput not at all.
- **The MTP draft graph not persisting.** `TS_Q35_MTP_DRAFT_PERSIST=1` moves
  `draftMs` by less than the run-to-run noise.
- **The confidence gate.** llama.cpp does not gate at all; dropping `--spec-pmin`
  to 0.05 is a wash on both prompts, because the steps it declines genuinely
  would have drafted badly.

What IS left is the per-call overhead of the MTP block. Instrumenting the two
halves over a 256-token run: the C# input projection (`MtpProjectInput` -
embedding, two RMS norms, a concat and `eh_proj`) costs 462 ms against the fused
block kernel's 804 ms, over 208 calls. That is 2.2 ms of every 6.1 ms draft
call, and **6.4% of the whole run**, spent on about six separate device op
launches. Caching its scratch tensors changes nothing (the allocator already
pools), so the cost is the launches themselves: on CUDA every op synchronises,
because the lazy-sync path (`TS_GGML_ASYNC_COMPUTE`) is Metal-only - it relies
on Metal's zero-copy host mapping. Folding the projection into the fused MTP
graph, so the whole draft step is one graph, is the next concrete step.

Three things in that table are worth reading carefully.

**The drafter's SIZE is a first-order performance variable on a card with no
headroom.** The same DFlash2 drafter at Q8_0 (3.0 GB) instead of Q4_K_M (1.6 GB)
turns a 1.23x win into a 0.75x loss - not because it drafts worse (its
acceptance is identical) but because the extra 1.4 GB pushes the trunk into
WDDM paging and the trunk's own verify slows from 78 ms to 128 ms. Match the
drafter quant to the headroom, not to the best available fidelity.

**The window is a workload choice, and the default is the conservative one.**
Qwen 3.8 defaults to 3 (see `SpecPreferredDraftWindow`). Widening it to 7 costs
9% on the factual prompt and 36% on prose, because a wider window buys verify
rows that get rejected AND makes the recurrent-state snapshots below
proportionally larger - ~150 MB per slot here, which on a card with no headroom
is its own second penalty. `--spec-draft N` overrides it. (An earlier revision
claimed a window of 7 WON by 10% on the factual prompt; that was measured before
the state stopped round-tripping, when a wider window amortised a fixed per-step
transfer that no longer exists.)

**What the drafter proposes is only half the story on a recurrent trunk** - the
other half is what a REJECTION costs, which is the next section.

## Rejection on a recurrent trunk

Qwen 3.5/3.6/3.8 are hybrids: 48 of Qwen 3.8's 64 layers are GatedDeltaNet, and
GDN carries a recurrent state that a KV cache's "drop the rejected tail" does not
apply to. Rolling a partially-rejected verify back used to mean restoring a
pre-verify copy of that state and re-forwarding the accepted prefix through the
entire trunk - a second whole-model forward - because the state after row *m*
simply did not exist anywhere. On top of that the state (151 MB for this model)
crossed PCIe twice per step: uploaded into the verify graph, downloaded again
after it. Speculation therefore cost MORE than the plain decode it was meant to
beat: 15.5 tok/s against 18.3 for DFlash2, 15.7 against 18.3 for MTP.

Three changes, all in the fused verify kernel and its Qwen 3.5 caller, removed
that (`ggml_ops_qwen35_verify.cpp`, `Qwen35Model.GatedDeltaNet.cs`):

1. **The verify keeps one recurrent-state snapshot per row.**
   `ggml_gated_delta_net` already takes a snapshot count and emits the last K
   per-token states; the conv state after row *m* is a window of a tensor the
   graph already builds. The state a rollback wants is therefore never
   recomputed - it is slot `N-1-accepted`.

2. **A snapshot is committed into the live state on the DEVICE.** Every cached
   verify graph binds its `*_state_in` from one shared device buffer, so writing
   a slot into it is visible to the next verify whatever shape it runs at. The
   state stops round-tripping: the next verify skips its upload, this one skips
   its download.

3. **The pre-verify snapshot becomes free.** A verify only READS the live
   slices - it writes its results to `*_state_out` and the snapshot slots - so
   the slices ARE the pre-verify state until a commit overwrites them, and a
   commit only happens after the rollback decision. `SpecSnapshotRecurrentState`
   copies nothing.

4. **The single-row steps stop round-tripping too.** A speculative session is
   not all verifies: when the drafter declines to propose, the step falls
   through to an ordinary one-row forward, and those ran the old download.
   Each one broke the device-state chain - 151 MB down, and the *next* verify
   had to upload it again - which on an MTP run (46 such steps out of 125) was
   most of what was left. A one-row step's post-window state is simply the
   `*_state_out` slices and nothing decides anything about it later, so the
   kernel now defers it as well and the caller commits slot -1 immediately:
   one device-to-device copy instead of 302 MB across PCIe.

The measured effect on Qwen3.8-27B, DFlash2, 256 prose tokens: `rollbackMs`
3604 -> 0, `snapshotMs` 919 -> 69, and 15.5 -> 20.9 tok/s. On the factual prompt
24.3 -> 31.7. Deferring the one-row steps (4) is worth a further 5-20% on top,
paired-run: DFlash2 factual 22.0 -> 27.1, MTP prose 19.1 -> 21.4.

Output is unchanged - in fact it is *more* exactly unchanged than before.
Committing on the device is a raw copy of the tensor the graph produced, where
the host round trip went through the state's unpack-and-repack; on the factual
prompt the device path reproduces plain decoding byte for byte while the host
path drifted in the last few tokens.

`TS_Q35_VERIFY_SNAPSHOTS=0` restores the old path entirely, and
`TS_Q35_VERIFY_DEFER_STATE=0` keeps the snapshots but restores the download, so
the two halves can be measured apart. Either is also what a shape the kernel
will not persist falls back to, automatically. The cost of the snapshots is
VRAM: the GDN op's output grows by one state per slot, ~150 MB per slot for this
model across all 48 recurrent layers, which is the other reason the default
window is 3 rather than 8.

## Where n-gram pays

`--spec-type ngram` needs no trained weights, so it works on every checkpoint,
including those that ship no speculator at all. It drafts by finding where the
last few tokens occurred earlier in the context and proposing what followed, so
it is strong exactly where the answer quotes its input: summarizing, editing,
translating or answering about a document, repetitive structured output, code
with repeated identifiers, agentic tool loops. On free-form prose it finds
nothing, every step degrades to a plain decode, and the cost governor keeps that
cheap.

Lookup is O(1) per step (an incrementally maintained hash index per n-gram
order), and every hit is verified token by token, so a hash collision can only
cost a rejected draft.
