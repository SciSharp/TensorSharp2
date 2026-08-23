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
| `block` | `BlockDraftSpeculator` | required | A whole block per pass with a confidence head: DeepSeek V4 DSpark, Muse-Glimmer DFlash. |
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
