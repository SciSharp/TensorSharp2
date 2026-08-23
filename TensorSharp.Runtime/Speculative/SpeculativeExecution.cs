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

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// The shared draft/verify core - the ONE place speculative decoding is
    /// implemented, for every algorithm and every model. Driven either by the
    /// engine's <c>BatchExecutor</c> (sampler-verified, one step per scheduler
    /// iteration) or by the standalone <c>SpeculativeDecoder</c> in
    /// TensorSharp.Models (which owns its own generate loop).
    ///
    /// Per step:
    ///   1. DRAFT: <see cref="ISpeculator.Propose"/> proposes up to
    ///      <see cref="MaxDraftTokens"/> tokens. This class does not know or
    ///      care how - a chained NextN head, a one-pass block drafter, an
    ///      n-gram lookup over the context, or something not written yet.
    ///   2. VERIFY: the trunk forwards [lastToken, d1..dK] as ONE batch with
    ///      per-row logits; the caller's <c>drawNext</c> draws each row with the
    ///      request's own sampler and drafts are accepted while the drawn token
    ///      matches; row m's drawn token is the corrected/bonus token for free.
    ///   3. ROLLBACK: on partial acceptance the recurrent (GDN) state is
    ///      restored from a pre-verify snapshot and re-advanced over the kept
    ///      prefix; attention KV only needs a position rewind. Trunks whose
    ///      verify already persisted usable KV skip the re-forward entirely.
    ///   4. COMMIT: kept tokens are handed back to the speculator with their
    ///      exact trunk hidden states, so a learned drafter's KV cache tracks
    ///      the real context and a lookup drafter's corpus tracks the real
    ///      output.
    ///
    /// WHY THE OUTPUT CANNOT CHANGE: every emitted token is drawn by the
    /// CALLER's sampler from a TRUNK row. The draft only decides whether that
    /// row had already been computed. A wrong draft costs a rollback, never a
    /// wrong token, which is why <see cref="ISpeculator"/> carries no
    /// correctness obligations at all.
    ///
    /// Single-sequence; the caller owns the model's KV cache lifecycle and the
    /// position bookkeeping (a step at <c>position</c> advances the trunk to
    /// <c>position + AcceptedCount + 1</c>).
    /// </summary>
    public sealed class SpeculativeExecution : IDisposable
    {
        private readonly ISpeculativeTarget _model;
        private readonly ISpeculator _speculator;
        private readonly ISpecTrunk _trunk;
        private readonly bool _needsHidden;
        private readonly int _hidden;
        private readonly int _vocab;

        // Trunk hidden state of the token immediately BEFORE the next pending
        // token (llama.cpp's pending_h). Zeros before the first prompt token.
        // Null when the speculator needs no hidden state.
        private readonly float[] _pendingH;

        // Reusable buffers (speculative windows are small; prefill chunk
        // buffers grow to the largest chunk seen).
        private readonly float[] _verifyLogits;  // [(K+1) * vocab]
        private readonly float[] _verifyH;       // [(K+1) * hidden]
        private readonly float[] _catchUpH;      // [(K+1) * hidden]
        private readonly float[] _stepLogits;    // [vocab] for plain/re-advance steps
        private readonly float[] _rowLogits;     // [vocab] scratch row handed to drawNext
        private readonly int[] _oneToken = new int[1];
        private readonly List<int> _draftTokens = new();
        private float[] _chunkH;                 // [chunk * hidden] prefill h capture, shifted in place into (token k, h of k-1) pairs
        private float[] _lastRowH;               // [hidden] the row the in-place shift would otherwise overwrite

        /// <summary>The algorithm doing the drafting. Exposed for logs and for
        /// callers that want to inspect or retune it mid-run.</summary>
        public ISpeculator Speculator => _speculator;

        /// <summary>The cost governor guarding this execution. Set
        /// <c>Enabled = false</c> to force drafting on for A/B measurement.</summary>
        public SpeculationCostGovernor Governor { get; } = new();

        /// <summary>Maximum tokens drafted per speculative step (llama.cpp n_max).</summary>
        public int MaxDraftTokens => _speculator.MaxDraftTokens;

        /// <summary>The drafter's confidence gate; see
        /// <see cref="ISpeculator.MinDraftProb"/> for what the number means for
        /// the algorithm in use.</summary>
        public float MinDraftProb
        {
            get => _speculator.MinDraftProb;
            set => _speculator.MinDraftProb = value;
        }

        /// <summary>Measure speculation against plain decoding at runtime and
        /// skip drafting while it is measurably slower. On by default.</summary>
        public bool AdaptiveSpeculation
        {
            get => Governor.Enabled;
            set => Governor.Enabled = value;
        }

        /// <summary>True when the trunk keeps the speculator in sync itself, so
        /// prompt prefill needs neither per-row hidden states nor a commit.</summary>
        public bool PrefillSelfCatchUp => _speculator.HandlesOwnPrefill;

        public SpeculationStats Stats { get; } = new();

        public SpeculativeExecution(ISpeculativeTarget model, ISpeculator speculator, ISpecTrunk trunk = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _speculator = speculator ?? throw new ArgumentNullException(nameof(speculator));
            _trunk = trunk ?? new LinearSpecTrunk(model);

            _needsHidden = speculator.NeedsHiddenState;
            _hidden = model.SpecFeatureSize;
            _vocab = model.Config.VocabSize;

            int k = speculator.MaxDraftTokens;
            _verifyLogits = new float[(k + 1) * (long)_vocab];
            _stepLogits = new float[_vocab];
            _rowLogits = new float[_vocab];
            if (_needsHidden)
            {
                _pendingH = new float[_hidden];
                _verifyH = new float[(k + 1) * _hidden];
                _catchUpH = new float[(k + 1) * _hidden];
            }
        }

        /// <summary>
        /// Start the statistics and the cost governor over WITHOUT touching the
        /// carry-in hidden state, for a caller that reports per turn while keeping
        /// the drafter aligned with a KV cache it is extending. Resetting the
        /// stats alone would leave the governor's round accumulators running behind
        /// zeroed counters - a per-turn log that reads 0.0 ms/token - and would
        /// carry a park decided on one turn's context into the next.
        /// </summary>
        public void ResetStatsAndGovernor()
        {
            Stats.Reset();
            Governor.Reset();
        }

        /// <summary>Reset speculative state and statistics. Does NOT touch the model's KV cache.</summary>
        public void Reset()
        {
            if (_pendingH != null)
                Array.Clear(_pendingH);
            _speculator.Reset();
            Stats.Reset();
            Governor.Reset();
        }

        /// <summary>
        /// Forward one prompt chunk through the trunk (capturing hidden states
        /// where the speculator needs them) and hand the chunk to the
        /// speculator so its own state covers it.
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

            if (!_needsHidden)
            {
                _trunk.Forward(chunk, null, _stepLogits, allLogitsRows: false);
                _speculator.Commit(chunk, null, startPos);
            }
            else if (_speculator.HandlesOwnPrefill)
            {
                // The trunk keeps the drafter in sync itself and only hands
                // back the last row's hidden state.
                _trunk.Forward(chunk, _pendingH, _stepLogits, allLogitsRows: false);
            }
            else
            {
                EnsureChunkBuffers(n);

                _trunk.Forward(chunk, _chunkH, _stepLogits, allLogitsRows: false);

                // Pair token k with the hidden state of the token before it. Done
                // IN PLACE: the only row still needed after the shift is the last
                // one (it becomes the next chunk's carry-in), so save that, slide
                // the block down a row and prepend the carried-in row. A second
                // chunk-sized buffer used to hold the result, which for a DFlash
                // feature row (33280 floats, 130 KB) is 130 MB at a 1024-row chunk
                // - and doubling the prefill chunk is worth far more than the
                // buffer it costs. Array.Copy is memmove-safe on overlap.
                Array.Copy(_chunkH, (long)(n - 1) * _hidden, _lastRowH, 0, _hidden);
                if (n > 1)
                    Array.Copy(_chunkH, 0, _chunkH, _hidden, (long)(n - 1) * _hidden);
                Array.Copy(_pendingH, 0, _chunkH, 0, _hidden);
                _speculator.Commit(chunk, _chunkH, startPos);

                Array.Copy(_lastRowH, 0, _pendingH, 0, _hidden);
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
        /// in place given the drafts pending in this window;
        /// <paramref name="history"/> (optional) is the caller's emitted-token
        /// history, for algorithms that mine it.
        /// The trunk advances to <c>position + AcceptedCount + 1</c>.
        /// </summary>
        public SpeculativeStepOutcome DecodeStep(
            int lastToken,
            int position,
            int kMax,
            Func<float[], int> drawNext,
            Action<float[], IReadOnlyList<int>> adjustDraftLogits = null,
            Action<int> onDraftAccepted = null,
            IReadOnlyList<int> history = null)
        {
            ArgumentNullException.ThrowIfNull(drawNext);

            long tStep0 = Stopwatch.GetTimestamp();
            // Governor: a step it declines becomes a plain decode, which is the
            // same path an all-low-confidence draft window already takes (and so
            // keeps the drafter's own state in sync via the commit below).
            if (!Governor.AllowsSpeculation())
            {
                kMax = 0;
                // Only a genuine park counts. The calibration plain steps that a
                // round takes to establish its baseline also come through here, and
                // counting them made a perfectly healthy run report "parked 3",
                // which is indistinguishable from a short real park in a log.
                if (Governor.IsParked)
                    Stats.ParkedSteps++;
            }

            kMax = Math.Min(kMax, MaxDraftTokens);
            // Verify needs position + K + 1 trunk slots; drafting writes drafter
            // rows up to position + K.
            kMax = Math.Min(kMax, _model.MaxContextLength - position - 2);

            _draftTokens.Clear();
            if (kMax > 0)
            {
                long tDraft0 = Stopwatch.GetTimestamp();
                _model.SpecEnsureCapacity(position + kMax + 1);
                _speculator.Propose(
                    new DraftContext
                    {
                        LastToken = lastToken,
                        Position = position,
                        MaxTokens = kMax,
                        CarryHidden = _pendingH,
                        History = history,
                        AdjustLogits = adjustDraftLogits,
                    },
                    _draftTokens);
                Stats.DraftTicks += Stopwatch.GetTimestamp() - tDraft0;
            }

            if (_draftTokens.Count == 0)
            {
                // Plain decode step (still captures h + keeps the drafter in sync).
                Stats.PlainSteps++;
                long tPlain0 = Stopwatch.GetTimestamp();
                _oneToken[0] = lastToken;
                _trunk.Forward(_oneToken, _verifyH, _stepLogits, allLogitsRows: false);
                if (_needsHidden)
                    Array.Copy(_pendingH, 0, _catchUpH, 0, _hidden);
                _speculator.Commit(_oneToken, _catchUpH, position);
                if (_needsHidden)
                    Array.Copy(_verifyH, 0, _pendingH, 0, _hidden);
                Stats.PlainTicks += Stopwatch.GetTimestamp() - tPlain0;

                float[] plainLogits = new float[_vocab];
                Array.Copy(_stepLogits, plainLogits, _vocab);
                RecordStep(speculated: false, tokensEmitted: 1, tStep0);
                return new SpeculativeStepOutcome
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

            // The tokens this step commits to the trunk: the verify batch's
            // accepted prefix plus the token it started from.
            int[] keep = new int[m + 1];
            Array.Copy(batch, keep, m + 1);

            if (m < k)
            {
                // Partial acceptance. Fast path: if the verify already persisted
                // reusable KV for the accepted prefix (no recurrent state), just keep
                // those writes and advance the position - the kept-prefix re-forward
                // is redundant (it would recompute byte-identical KV). This is the
                // dominant rollback cost on long contexts. Otherwise roll back to the
                // pre-verify checkpoint and re-advance over the kept prefix.
                Stats.RollbackSteps++;
                long tRoll0 = Stopwatch.GetTimestamp();
                if (!_trunk.TryCommitVerifiedPrefix(position + m + 1))
                {
                    _trunk.Rollback(position);
                    _trunk.Forward(keep, null, _stepLogits, allLogitsRows: false);
                }
                Stats.RollbackTicks += Stopwatch.GetTimestamp() - tRoll0;
            }

            // Hand the kept tokens to the speculator with their exact trunk
            // hidden states.
            {
                long tCatch0 = Stopwatch.GetTimestamp();
                if (_needsHidden)
                {
                    Array.Copy(_pendingH, 0, _catchUpH, 0, _hidden);
                    if (m > 0)
                        Array.Copy(_verifyH, 0, _catchUpH, _hidden, (long)m * _hidden);
                }
                _speculator.Commit(keep, _catchUpH, position);
                Stats.CatchUpTicks += Stopwatch.GetTimestamp() - tCatch0;
            }

            if (_needsHidden)
                Array.Copy(_verifyH, (long)m * _hidden, _pendingH, 0, _hidden);

            float[] nextLogits = new float[_vocab];
            Array.Copy(_verifyLogits, (long)m * _vocab, nextLogits, 0, _vocab);
            // m accepted drafts plus the corrected/bonus token.
            RecordStep(speculated: true, tokensEmitted: m + 1, tStep0);
            return new SpeculativeStepOutcome
            {
                AcceptedCount = m,
                NextToken = nextToken,
                NextLogits = nextLogits,
                UsedSpeculation = true,
            };
        }

        public void Dispose() => _speculator.Dispose();

        private void RecordStep(bool speculated, int tokensEmitted, long tStep0)
        {
            Governor.Record(speculated, tokensEmitted, Stopwatch.GetTimestamp() - tStep0);
            Stats.PlainMsPerToken = Governor.PlainMsPerToken;
            Stats.SpecMsPerToken = Governor.SpecMsPerToken;
        }

        private void EnsureChunkBuffers(int chunkLen)
        {
            long need = (long)chunkLen * _hidden;
            if (_chunkH == null || _chunkH.Length < need)
                _chunkH = new float[need];
            _lastRowH ??= new float[_hidden];
        }
    }
}
