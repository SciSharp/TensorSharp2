// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Model contract for NextN/MTP (multi-token prediction) speculative
    /// decoding. Implemented by models that ship a one-block draft head past
    /// the trunk (Qwen3.6's <c>nextn_predict_layers</c>; llama.cpp's
    /// <c>--spec-type draft-mtp</c>, vLLM's <c>qwen3_5_mtp</c> speculator).
    /// Lives in Runtime (not Models) so <see cref="BatchExecutor"/> can drive
    /// speculation without a Models reference.
    /// </summary>
    public interface IMtpSpeculativeModel : IModelArchitecture
    {
        /// <summary>True when the loaded weights contain a usable NextN/MTP draft block.</summary>
        bool HasMtp { get; }

        /// <summary>
        /// True when speculation is expected to be PROFITABLE on the current
        /// backend — i.e. the model can drive its accelerated MTP path (the fused
        /// multi-token verify + draft kernels). When false the verify (seqLen=K+1)
        /// and draft fall to the per-op fallback, which does not amortize the trunk
        /// over the speculative window and makes speculation net-negative; the
        /// executor then serves the sequence with the normal (fast) decode path
        /// instead of arming speculation. Default true (backends/models that always
        /// have their accelerated path, or where the per-op path still wins).
        /// </summary>
        bool MtpSpeculationProfitable => true;

        /// <summary>
        /// Width of one row of the hidden state passed between the trunk and the
        /// draft head (<see cref="SpecForward"/>'s hAllOut, <see cref="MtpDraftStep"/>'s
        /// hPrev). Defaults to the model's hidden size; block drafters that consume
        /// several concatenated layers (DeepSeek V4's DSpark) report a wider row.
        /// </summary>
        int MtpHiddenSize => Config.HiddenSize;

        /// <summary>
        /// Tokens drafted per BLOCK by a semi-autoregressive draft head (DSpark),
        /// or 0 for classic one-token-at-a-time MTP drafting. When positive the
        /// executor calls <see cref="MtpDraftBlock"/> once per step instead of
        /// looping <see cref="MtpDraftStep"/>.
        /// </summary>
        int MtpDraftBlockSize => 0;

        /// <summary>Prompt-prefill chunk the model prefers for speculative
        /// prefill (0 = let the caller choose). Models whose trunk re-reads
        /// per-expert weights once per micro-batch want whole micro-batches.</summary>
        int MtpPrefillChunkSize => 0;

        /// <summary>
        /// True when <see cref="SpecForward"/> replays the draft head's own state
        /// for every row it processes, so prompt prefill needs neither per-row
        /// hidden states nor a <see cref="MtpCatchUp"/> call. Prefill then runs
        /// as ONE call and keeps the trunk's micro-batch pipelining instead of
        /// stalling on a readback per chunk.
        ///
        /// Implementations must honour the buffer-size contract this implies:
        /// when <c>hAllOut</c> is too small to hold one row per token,
        /// <see cref="SpecForward"/> fills only its first row, with the hidden
        /// state of the LAST token it processed.
        /// </summary>
        bool MtpPrefillSelfCatchUp => false;

        /// <summary>
        /// Drafts one block at <paramref name="position"/> (the position of
        /// <paramref name="lastToken"/>, which is not yet in the trunk cache),
        /// given the trunk hidden state of the position before it. Writes the
        /// drafted tokens to <paramref name="draftOut"/> and their predicted
        /// acceptance probabilities to <paramref name="confOut"/>, and returns
        /// how many it produced.
        /// </summary>
        int MtpDraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut) => 0;

        /// <summary>Trunk tokens currently committed to the model's live KV cache.</summary>
        int CacheSeqLen { get; }

        /// <summary>Maximum trunk context length.</summary>
        int MaxContextLength { get; }

        /// <summary>
        /// Trunk forward identical to <see cref="IModelArchitecture.Forward"/>
        /// but additionally captures the post-final-norm hidden state of every
        /// row into <paramref name="hAllOut"/> (n*hidden floats) and, when
        /// <paramref name="allLogitsRows"/> is set, LM-head logits for every
        /// row into <paramref name="logitsOut"/> (n*vocab floats) instead of
        /// only the last row. Advances the KV caches like Forward().
        /// </summary>
        void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows);

        /// <summary>One MTP draft step at <paramref name="pos"/>: consume
        /// (token, previous hidden), fill next-token logits and the chained
        /// MTP hidden state.</summary>
        void MtpDraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut);

        /// <summary>Replay verified trunk tokens through the MTP block so its
        /// KV cache tracks exact trunk hidden states (llama.cpp's draft-mtp
        /// process()). Row k of <paramref name="hRows"/> is the hidden state of
        /// the token PRECEDING tokens[k].</summary>
        void MtpCatchUp(int[] tokens, float[] hRows, int startPos);

        /// <summary>Pre-grow the KV caches to cover a full speculative window
        /// (growing mid-draft would drop MTP rows written past the trunk position).</summary>
        void MtpEnsureCapacity(int requiredSeqLen);

        /// <summary>Snapshot the recurrent (GDN/SSM) state before a verify batch.</summary>
        void MtpSnapshotRecurrentState();

        /// <summary>Restore the recurrent state captured by <see cref="MtpSnapshotRecurrentState"/>.</summary>
        void MtpRestoreRecurrentState();

        /// <summary>Rewind the attention KV position counter after rejected
        /// speculative tokens (no data movement needed).</summary>
        void MtpRewindCache(int length);

        /// <summary>
        /// True when a verify batch has already written reusable attention KV for
        /// EVERY token it processed (so the accepted prefix's KV is correct in the
        /// live cache), and the model has no recurrent state that a re-forward would
        /// need to advance. When true the executor skips the redundant kept-prefix
        /// re-forward on partial acceptance and simply rewinds the cache position to
        /// keep the verify's writes — the dominant rollback cost on long contexts.
        /// Default false (the re-forward path) so recurrent models (e.g. Qwen 3.5
        /// GatedDeltaNet) are unaffected.
        /// </summary>
        bool MtpVerifyPersistsAcceptedKv => false;
    }

    /// <summary>
    /// Model contract for serving the speculative TRUNK passes through the
    /// batched paged path (paged KV via slot mapping, per-slot recurrent
    /// state) instead of the single live linear cache. This is the
    /// high-throughput option: the trunk runs on the same kernels as the
    /// non-speculative batched baseline, composes with prefix caching, and
    /// transitions gracefully to/from concurrent batches (the sequence's K/V
    /// always lives in paged storage). The MTP draft head itself still runs
    /// on the linear cache at the draft layer — it is one decoder block whose
    /// state is private to the speculative context.
    /// </summary>
    public interface IMtpBatchedSpeculativeModel : IMtpSpeculativeModel
    {
        /// <summary>True when the loaded model/backend can serve speculative
        /// trunk passes through its batched paged path.</summary>
        bool SupportsBatchedSpecTrunk { get; }

        /// <summary>
        /// Trunk forward of <paramref name="tokens"/> for ONE sequence through
        /// the batched paged path. <paramref name="startPos"/> must equal
        /// <c>seq.NumComputedTokens</c> (the caller advances the sequence only
        /// after the step completes); the sequence's block table must already
        /// cover <c>startPos + tokens.Length</c> positions. Captures per-row
        /// post-final-norm hidden states into <paramref name="hAllOut"/>
        /// (n*hidden floats; may be null) and logits into
        /// <paramref name="logitsOut"/> (n*vocab floats when
        /// <paramref name="allLogitsRows"/>, else vocab floats for the last row).
        /// </summary>
        void SpecForwardBatched(SequenceState seq, int[] tokens, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows);

        /// <summary>Snapshot the per-slot recurrent (GDN/SSM) state of
        /// <paramref name="seq"/> before a verify batch.</summary>
        void MtpSnapshotRecurrentStateSlots(SequenceState seq);

        /// <summary>Restore the per-slot recurrent state captured by
        /// <see cref="MtpSnapshotRecurrentStateSlots"/>. Paged attention KV
        /// needs no rewind: reads are bounded by the per-pass sequence length
        /// and rejected slots are overwritten by the kept-prefix re-forward
        /// and subsequent steps.</summary>
        void MtpRestoreRecurrentStateSlots(SequenceState seq);
    }

    /// <summary>
    /// The trunk backend a speculative execution drives: prompt/verify/plain
    /// forwards plus recurrent-state snapshot/rollback. Two implementations:
    /// <see cref="LinearMtpTrunk"/> (the model's live linear cache — the
    /// standalone decoder and the per-sequence engine fallback) and the
    /// executor's batched trunk (paged KV + per-slot state via
    /// <see cref="IMtpBatchedSpeculativeModel"/>).
    /// </summary>
    public interface IMtpSpecTrunk
    {
        /// <summary>Forward <paramref name="tokens"/> at the trunk's current
        /// position, capturing per-row hidden states and logits like
        /// <see cref="IMtpSpeculativeModel.SpecForward"/>. Advances the trunk
        /// by <c>tokens.Length</c>.</summary>
        void Forward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows);

        /// <summary>Snapshot recurrent state before a verify batch.</summary>
        void SnapshotRecurrentState();

        /// <summary>Roll the trunk back to <paramref name="position"/>
        /// committed tokens: restore the recurrent snapshot and rewind any
        /// attention-KV bookkeeping.</summary>
        void Rollback(int position);

        /// <summary>
        /// Fast partial-acceptance commit: when the trunk's verify already wrote
        /// reusable KV for the accepted prefix (no recurrent state to replay), keep
        /// those writes and just set the live position to <paramref name="newPosition"/>
        /// (= committed + accepted + 1), skipping the redundant kept-prefix
        /// re-forward. Returns false when the trunk cannot do this (caller falls back
        /// to <see cref="Rollback"/> + re-forward). Default: not supported.
        /// </summary>
        bool TryCommitVerifiedPrefix(int newPosition) => false;
    }

    /// <summary>Linear-cache trunk: forwards through
    /// <see cref="IMtpSpeculativeModel.SpecForward"/> on the model's single
    /// live KV cache.</summary>
    public sealed class LinearMtpTrunk : IMtpSpecTrunk
    {
        private readonly IMtpSpeculativeModel _model;

        public LinearMtpTrunk(IMtpSpeculativeModel model)
            => _model = model ?? throw new ArgumentNullException(nameof(model));

        public void Forward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
            => _model.SpecForward(tokens, hAllOut, logitsOut, allLogitsRows);

        public void SnapshotRecurrentState() => _model.MtpSnapshotRecurrentState();

        public void Rollback(int position)
        {
            _model.MtpRestoreRecurrentState();
            _model.MtpRewindCache(position);
        }

        public bool TryCommitVerifiedPrefix(int newPosition)
        {
            if (!_model.MtpVerifyPersistsAcceptedKv)
                return false;
            // The verify wrote correct KV for all batch tokens; the accepted prefix's
            // KV is already live. Just drop the rejected tail by rewinding the position
            // (rejected slots are overwritten by later writes and never read past the
            // live position). No recurrent state to restore for such models.
            _model.MtpRewindCache(newPosition);
            return true;
        }
    }

    /// <summary>Cumulative speculative-decoding counters for one execution
    /// (one request on the engine path; one GenerateGreedy call on the
    /// standalone path). The engine logs these when a request finishes.</summary>
    public sealed class MtpSpecStats
    {
        public long TokensDrafted { get; internal set; }
        public long TokensAccepted { get; internal set; }
        public long VerifySteps { get; internal set; }
        public long PlainSteps { get; internal set; }
        public long RollbackSteps { get; internal set; }
        public double AcceptanceRate => TokensDrafted > 0 ? (double)TokensAccepted / TokensDrafted : 0;

        // Wall-clock phase breakdown (Stopwatch ticks) so a slow speculative
        // request can be attributed: drafting (sequential MTP head steps),
        // trunk verify forwards, recurrent-state snapshots, rollback
        // re-forwards, MTP catch-up replays, and plain (non-drafting) steps.
        internal long DraftTicks;
        internal long VerifyTicks;
        internal long SnapshotTicks;
        internal long RollbackTicks;
        internal long CatchUpTicks;
        internal long PlainTicks;

        public double DraftMs => TicksToMs(DraftTicks);
        public double VerifyMs => TicksToMs(VerifyTicks);
        public double SnapshotMs => TicksToMs(SnapshotTicks);
        public double RollbackMs => TicksToMs(RollbackTicks);
        public double CatchUpMs => TicksToMs(CatchUpTicks);
        public double PlainMs => TicksToMs(PlainTicks);

        private static double TicksToMs(long ticks)
            => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

        public void Reset()
        {
            TokensDrafted = TokensAccepted = VerifySteps = PlainSteps = RollbackSteps = 0;
            DraftTicks = VerifyTicks = SnapshotTicks = RollbackTicks = CatchUpTicks = PlainTicks = 0;
        }
    }

    /// <summary>Result of one <see cref="MtpSpeculativeExecution.DecodeStep"/>.</summary>
    public readonly struct MtpDecodeOutcome
    {
        /// <summary>Drafted tokens accepted by verification (each already
        /// reported through the <c>onDraftAccepted</c> callback, in order).</summary>
        public int AcceptedCount { get; init; }

        /// <summary>The token DRAWN from the first mismatching verify row (or
        /// the bonus row after full acceptance). It is the sequence's next
        /// output token and must be emitted as-is on the caller's next step —
        /// re-drawing from <see cref="NextLogits"/> later would bias the stream
        /// toward the drafts (the classic speculative-sampling pitfall).
        /// -1 when the step degraded to a plain decode (no confident drafts):
        /// the caller samples from <see cref="NextLogits"/> as usual.</summary>
        public int NextToken { get; init; }

        /// <summary>Caller-owned copy of the logits at the sequence's new last
        /// position (verify row m, or the plain step's logits).</summary>
        public float[] NextLogits { get; init; }

        /// <summary>False when the step ran as a plain single-token decode.</summary>
        public bool UsedSpeculation { get; init; }
    }

    /// <summary>
    /// Shared core of NextN/MTP speculative decoding, driven either by the
    /// engine's <see cref="BatchExecutor"/> (sampler-verified, one step per
    /// scheduler iteration) or by the standalone greedy decoder in
    /// TensorSharp.Models (argmax-verified loop). Per step:
    ///   1. DRAFT: the 1-block MTP head autoregressively proposes up to
    ///      <see cref="MaxDraftTokens"/> high-confidence tokens, chaining its own
    ///      hidden output. The optional <c>adjustDraftLogits</c> hook lets the
    ///      caller apply its sampler's repetition/presence/frequency penalties
    ///      to the draft logits so drafting argmaxes the SAME distribution
    ///      verification draws from — without it, penalized configs (the chat
    ///      default repPen 1.1, including --temperature 0) diverge ever more
    ///      often as the output history grows and acceptance decays to ~0.
    ///   2. VERIFY: the trunk forwards [lastToken, d1..dK] as ONE batch with
    ///      per-row logits; the caller's <c>drawNext</c> draws each row with the
    ///      request's own sampler and drafts are accepted while the drawn token
    ///      matches; row m's drawn token is the corrected/bonus token for free.
    ///   3. ROLLBACK: on partial acceptance the recurrent (GDN) state is
    ///      restored from a pre-verify snapshot and re-advanced over the kept
    ///      prefix; attention KV only needs a position rewind.
    ///   4. CATCH-UP: kept tokens are replayed through the MTP block with exact
    ///      trunk hidden states so its KV cache tracks the real context.
    ///
    /// Single-sequence; the caller owns the model's KV cache lifecycle and the
    /// position bookkeeping (a step at <c>position</c> advances the trunk to
    /// <c>position + AcceptedCount + 1</c>).
    /// </summary>
    public sealed class MtpSpeculativeExecution
    {
        private readonly IMtpSpeculativeModel _model;
        private readonly IMtpSpecTrunk _trunk;
        private readonly int _hidden;
        private readonly int _vocab;

        // h_nextn of the token immediately BEFORE the next pending token
        // (llama.cpp's pending_h). Zeros before the first prompt token.
        private readonly float[] _pendingH;

        // Reusable buffers (speculative windows are small; prefill chunk
        // buffers grow to the largest chunk seen).
        private readonly float[] _draftLogits;
        private readonly float[] _draftHA;
        private readonly float[] _draftHB;
        private readonly float[] _verifyLogits;  // [(K+1) * vocab]
        private readonly float[] _verifyH;       // [(K+1) * hidden]
        private readonly float[] _catchUpH;      // [(K+1) * hidden]
        private readonly float[] _stepLogits;    // [vocab] for plain/re-advance steps
        private readonly float[] _rowLogits;     // [vocab] scratch row handed to drawNext
        private readonly List<int> _draftTokens = new();
        private float[] _chunkH;                 // [chunk * hidden] prefill h capture
        private float[] _chunkHPairs;            // [chunk * hidden] (token k, h of token k-1) pairs

        /// <summary>Maximum tokens drafted per speculative step (llama.cpp n_max).</summary>
        public int MaxDraftTokens { get; }

        /// <summary>
        /// Default gate for a per-token MTP head: the top-1 probability over the
        /// head's top-10 logits, thresholded per drafted token (llama.cpp's
        /// top-k(10) draft sampler with p_min).
        /// </summary>
        public const float DefaultTokenMinDraftProb = 0.75f;

        /// <summary>
        /// Default gate for a BLOCK drafter (DSpark). Much lower than
        /// <see cref="DefaultTokenMinDraftProb"/> because it thresholds a
        /// different quantity: the CUMULATIVE prefix probability (the product of
        /// the confidence head's per-position estimates), which decays with every
        /// position. 0.35 is the break-even point where an extra verify row stops
        /// paying for itself on a sparse-MoE trunk; a per-token-sized 0.75 here
        /// truncates almost every block to nothing.
        /// </summary>
        public const float DefaultBlockMinDraftProb = 0.35f;

        /// <summary>
        /// Minimum draft confidence for a drafted token to be kept; drafting
        /// stops at the first token below it. For MTP heads this is the top-1
        /// probability over the head's top-10 logits (llama.cpp's top-k(10)
        /// draft sampler); for a DSpark block it is the CUMULATIVE product of the
        /// confidence head's predicted acceptance probabilities. Defaults to the
        /// constant matching the loaded drafter's kind.
        /// </summary>
        public float MinDraftProb { get; set; } = DefaultTokenMinDraftProb;

        private readonly int _blockDraft;
        private readonly int[] _blockTokens;
        private readonly float[] _blockConf;

        public MtpSpecStats Stats { get; } = new();

        public MtpSpeculativeExecution(IMtpSpeculativeModel model, int maxDraftTokens = 8, IMtpSpecTrunk trunk = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            if (!model.HasMtp)
                throw new ArgumentException("Model has no NextN/MTP draft block.", nameof(model));
            if (maxDraftTokens < 1)
                throw new ArgumentOutOfRangeException(nameof(maxDraftTokens));
            _trunk = trunk ?? new LinearMtpTrunk(model);

            _blockDraft = model.MtpDraftBlockSize;
            if (_blockDraft > 0)
            {
                MaxDraftTokens = maxDraftTokens = Math.Min(maxDraftTokens, _blockDraft);
                MinDraftProb = DefaultBlockMinDraftProb;
            }
            else
            {
                MaxDraftTokens = maxDraftTokens;
            }
            _hidden = model.MtpHiddenSize;
            _vocab = model.Config.VocabSize;
            _blockTokens = new int[Math.Max(_blockDraft, 1)];
            _blockConf = new float[Math.Max(_blockDraft, 1)];

            _pendingH = new float[_hidden];
            _draftLogits = new float[_vocab];
            _draftHA = new float[_hidden];
            _draftHB = new float[_hidden];
            _verifyLogits = new float[(maxDraftTokens + 1) * (long)_vocab];
            _verifyH = new float[(maxDraftTokens + 1) * _hidden];
            _catchUpH = new float[(maxDraftTokens + 1) * _hidden];
            _stepLogits = new float[_vocab];
            _rowLogits = new float[_vocab];
        }

        /// <summary>Reset speculative state and statistics. Does NOT touch the model's KV cache.</summary>
        public void Reset()
        {
            Array.Clear(_pendingH);
            Stats.Reset();
        }

        /// <summary>
        /// Forward one prompt chunk through the trunk (capturing h_nextn) and
        /// replay it through the MTP block so its KV cache covers the chunk.
        /// <paramref name="startPos"/> is the chunk's first trunk position
        /// (chunks are contiguous from position 0); the trunk must be
        /// positioned there. Returns a caller-owned copy of the last-position
        /// logits.
        /// </summary>
        public float[] PrefillStep(int[] chunk, int startPos)
        {
            if (chunk == null || chunk.Length == 0)
                throw new ArgumentException("Chunk must not be empty.", nameof(chunk));

            int n = chunk.Length;

            if (_model.MtpPrefillSelfCatchUp)
            {
                // The trunk keeps the draft head in sync itself and only hands
                // back the last row's hidden state.
                _trunk.Forward(chunk, _pendingH, _stepLogits, allLogitsRows: false);
            }
            else
            {
                EnsureChunkBuffers(n);

                _trunk.Forward(chunk, _chunkH, _stepLogits, allLogitsRows: false);

                // Pair token k with the hidden state of the token before it.
                Array.Copy(_pendingH, 0, _chunkHPairs, 0, _hidden);
                if (n > 1)
                    Array.Copy(_chunkH, 0, _chunkHPairs, _hidden, (long)(n - 1) * _hidden);
                _model.MtpCatchUp(chunk, _chunkHPairs, startPos);

                Array.Copy(_chunkH, (long)(n - 1) * _hidden, _pendingH, 0, _hidden);
            }

            float[] logits = new float[_vocab];
            Array.Copy(_stepLogits, logits, _vocab);
            return logits;
        }

        /// <summary>
        /// One speculative decode step for <paramref name="lastToken"/> at trunk
        /// position <paramref name="position"/> (== tokens already in the cache).
        /// <paramref name="kMax"/> additionally caps this step's draft window
        /// (token budget, KV block capacity); the effective window is
        /// min(kMax, <see cref="MaxDraftTokens"/>, context headroom) and a
        /// non-positive window degrades to a plain decode.
        /// <paramref name="drawNext"/> draws a token from a verify-row logits
        /// copy with the caller's sampler; <paramref name="onDraftAccepted"/>
        /// fires for each accepted draft BEFORE the next row is drawn so the
        /// caller can keep its penalty history exact;
        /// <paramref name="adjustDraftLogits"/> (optional) mutates draft logits
        /// in place given the drafts pending in this window.
        /// The trunk advances to <c>position + AcceptedCount + 1</c>.
        /// </summary>
        public MtpDecodeOutcome DecodeStep(
            int lastToken,
            int position,
            int kMax,
            Func<float[], int> drawNext,
            Action<float[], IReadOnlyList<int>> adjustDraftLogits = null,
            Action<int> onDraftAccepted = null)
        {
            ArgumentNullException.ThrowIfNull(drawNext);

            kMax = Math.Min(kMax, MaxDraftTokens);
            // Verify needs position + K + 1 trunk slots; drafting writes MTP
            // rows up to position + K.
            kMax = Math.Min(kMax, _model.MaxContextLength - position - 2);

            _draftTokens.Clear();
            if (kMax > 0)
            {
                long tDraft0 = Stopwatch.GetTimestamp();
                _model.MtpEnsureCapacity(position + kMax + 1);

                if (_blockDraft > 0)
                {
                    // Semi-autoregressive block draft (DSpark): one pass proposes
                    // the whole block, and the confidence head truncates it.
                    //
                    // The gate is CUMULATIVE: draft position i only pays off when
                    // the whole prefix before it is also accepted, so the product
                    // of the per-position acceptance probabilities is the expected
                    // value of adding it, and MinDraftProb is the point where that
                    // value stops covering the extra verify row. (Per-position
                    // gating -- what the reference runtime does -- keeps positions
                    // whose prefix has already gone unlikely.)
                    int n = _model.MtpDraftBlock(lastToken, _pendingH, position, _blockTokens, _blockConf);
                    double cum = 1.0;
                    for (int i = 0; i < Math.Min(n, kMax); i++)
                    {
                        cum *= _blockConf[i];
                        if (cum < MinDraftProb)
                            break;
                        _draftTokens.Add(_blockTokens[i]);
                    }
                }
                else
                {
                    float[] hIn = _pendingH;
                    float[] hOut = _draftHA;
                    int tokIn = lastToken;
                    for (int i = 0; i < kMax; i++)
                    {
                        _model.MtpDraftStep(tokIn, hIn, position + i, _draftLogits, hOut);
                        adjustDraftLogits?.Invoke(_draftLogits, _draftTokens);
                        int d = ArgmaxWithTopKConfidence(_draftLogits, _vocab, out float p);
                        if (p < MinDraftProb)
                            break;
                        _draftTokens.Add(d);
                        tokIn = d;
                        // Chain the MTP hidden output into the next draft step.
                        float[] next = ReferenceEquals(hOut, _draftHA) ? _draftHB : _draftHA;
                        hIn = hOut;
                        hOut = next;
                    }
                }
                Stats.DraftTicks += Stopwatch.GetTimestamp() - tDraft0;
            }

            if (_draftTokens.Count == 0)
            {
                // Plain decode step (still captures h + keeps the MTP cache in sync).
                Stats.PlainSteps++;
                long tPlain0 = Stopwatch.GetTimestamp();
                _trunk.Forward(new[] { lastToken }, _verifyH, _stepLogits, allLogitsRows: false);
                Array.Copy(_pendingH, 0, _catchUpH, 0, _hidden);
                _model.MtpCatchUp(new[] { lastToken }, _catchUpH, position);
                Array.Copy(_verifyH, 0, _pendingH, 0, _hidden);
                Stats.PlainTicks += Stopwatch.GetTimestamp() - tPlain0;

                float[] plainLogits = new float[_vocab];
                Array.Copy(_stepLogits, plainLogits, _vocab);
                return new MtpDecodeOutcome
                {
                    AcceptedCount = 0,
                    NextToken = -1,
                    NextLogits = plainLogits,
                    UsedSpeculation = false,
                };
            }

            // VERIFY: one batched trunk forward over [lastToken, d1..dK].
            Stats.VerifySteps++;
            int k = _draftTokens.Count;
            int[] batch = new int[k + 1];
            batch[0] = lastToken;
            for (int i = 0; i < k; i++)
                batch[i + 1] = _draftTokens[i];

            long tSnap0 = Stopwatch.GetTimestamp();
            _trunk.SnapshotRecurrentState();
            long tVerify0 = Stopwatch.GetTimestamp();
            Stats.SnapshotTicks += tVerify0 - tSnap0;
            _trunk.Forward(batch, _verifyH, _verifyLogits, allLogitsRows: true);
            Stats.VerifyTicks += Stopwatch.GetTimestamp() - tVerify0;

            int m = 0;
            int nextToken;
            while (true)
            {
                Array.Copy(_verifyLogits, (long)m * _vocab, _rowLogits, 0, _vocab);
                int drawn = drawNext(_rowLogits);
                if (m < k && drawn == _draftTokens[m])
                {
                    onDraftAccepted?.Invoke(drawn);
                    m++;
                    continue;
                }
                nextToken = drawn;
                break;
            }

            Stats.TokensDrafted += k;
            Stats.TokensAccepted += m;

            if (m < k)
            {
                // Partial acceptance. Fast path: if the verify already persisted
                // reusable KV for the accepted prefix (no recurrent state), just keep
                // those writes and advance the position — the kept-prefix re-forward
                // is redundant (it would recompute byte-identical KV). This is the
                // dominant rollback cost on long contexts. Otherwise roll back to the
                // pre-verify checkpoint and re-advance over the kept prefix.
                Stats.RollbackSteps++;
                long tRoll0 = Stopwatch.GetTimestamp();
                if (!_trunk.TryCommitVerifiedPrefix(position + m + 1))
                {
                    _trunk.Rollback(position);
                    int[] keep = new int[m + 1];
                    Array.Copy(batch, keep, m + 1);
                    _trunk.Forward(keep, null, _stepLogits, allLogitsRows: false);
                }
                Stats.RollbackTicks += Stopwatch.GetTimestamp() - tRoll0;
            }

            // MTP catch-up over the kept tokens with exact trunk hidden states.
            {
                long tCatch0 = Stopwatch.GetTimestamp();
                int[] keep = new int[m + 1];
                Array.Copy(batch, keep, m + 1);
                Array.Copy(_pendingH, 0, _catchUpH, 0, _hidden);
                if (m > 0)
                    Array.Copy(_verifyH, 0, _catchUpH, _hidden, (long)m * _hidden);
                _model.MtpCatchUp(keep, _catchUpH, position);
                Stats.CatchUpTicks += Stopwatch.GetTimestamp() - tCatch0;
            }

            Array.Copy(_verifyH, (long)m * _hidden, _pendingH, 0, _hidden);

            float[] nextLogits = new float[_vocab];
            Array.Copy(_verifyLogits, (long)m * _vocab, nextLogits, 0, _vocab);
            return new MtpDecodeOutcome
            {
                AcceptedCount = m,
                NextToken = nextToken,
                NextLogits = nextLogits,
                UsedSpeculation = true,
            };
        }

        private void EnsureChunkBuffers(int chunkLen)
        {
            long need = (long)chunkLen * _hidden;
            if (_chunkH == null || _chunkH.Length < need)
            {
                _chunkH = new float[need];
                _chunkHPairs = new float[need];
            }
        }

        /// <summary>
        /// Argmax plus the top-1 probability computed over the top-10 logits
        /// (softmax restricted to the 10 best candidates — the same confidence
        /// measure llama.cpp's draft-mtp top-k(10) sampler thresholds with p_min).
        /// </summary>
        private static int ArgmaxWithTopKConfidence(float[] logits, int vocab, out float prob)
        {
            const int K = 10;
            Span<float> topV = stackalloc float[K];
            topV.Fill(float.NegativeInfinity);
            int best = 0;
            for (int i = 0; i < vocab; i++)
            {
                float v = logits[i];
                if (v <= topV[K - 1])
                    continue;
                int j = K - 1;
                while (j > 0 && topV[j - 1] < v)
                {
                    topV[j] = topV[j - 1];
                    j--;
                }
                topV[j] = v;
                if (j == 0)
                    best = i;
            }

            double denom = 0;
            for (int j = 0; j < K; j++)
            {
                if (float.IsNegativeInfinity(topV[j]))
                    break;
                denom += Math.Exp(topV[j] - topV[0]);
            }
            prob = denom > 0 ? (float)(1.0 / denom) : 0f;
            return best;
        }
    }
}
