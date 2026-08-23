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
using System.Diagnostics;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Decides, at run time and from measurement alone, whether speculation is
    /// currently worth doing - and parks it when it is not.
    ///
    /// Speculation is only ever a SPEED optimization (the emitted stream is
    /// identical either way), so it must never make decoding slower. Whether it
    /// pays off is a property of the (model, drafter, backend) TRIPLE, not of
    /// the code: on Muse-Glimmer-30B (a sparse MoE where a decode step only
    /// touches the active experts) the 1.5 GB dense DFlash drafter costs roughly
    /// half a trunk step per drafted token, and measured end to end at every
    /// window size from 1 to 15 it ran 18-31% SLOWER than plain greedy despite a
    /// 70% acceptance rate. Acceptance rate alone cannot see that: it says
    /// nothing about what a draft COSTS relative to the trunk.
    ///
    /// So measure both and let the numbers decide. A round measures speculation
    /// first (so the very first decode step still speculates), then takes a
    /// short plain baseline, then judges. A win holds for
    /// <see cref="WinRecheckInterval"/> steps; a loss parks drafting for a
    /// backed-off interval and re-probes, so a pair that becomes profitable
    /// later (longer context, different prompt) is picked back up. Overhead when
    /// speculation is a loss is bounded by one probe per park interval.
    ///
    /// Kept apart from <see cref="SpeculativeExecution"/> on purpose: this is a
    /// measurement POLICY, and it is the piece most likely to be tuned,
    /// replaced or switched off per deployment without touching the
    /// draft/verify protocol.
    /// </summary>
    public sealed class SpeculationCostGovernor
    {
        /// <summary>Speculative steps measured per round before judging speculation.</summary>
        private const int ProbeSpecSteps = 8;

        /// <summary>Plain steps measured per round, for the baseline to compare against.</summary>
        private const int CalibrationPlainSteps = 3;

        /// <summary>Longest a losing verdict may park drafting for.</summary>
        private const int ParkedProbeInterval = 64;

        /// <summary>
        /// Park length after the FIRST losing verdict. A verdict formed right after
        /// a prefill is the least trustworthy one the governor ever makes, so the
        /// penalty for getting it wrong is capped low and only backs off toward
        /// <see cref="ParkedProbeInterval"/> if speculation keeps losing.
        /// </summary>
        private const int FirstParkInterval = 16;

        /// <summary>Steps a winning verdict holds for before the next measuring round.</summary>
        private const int WinRecheckInterval = 512;

        /// <summary>
        /// Speculation must not be more than this much slower to stay on. This is a
        /// tolerance on a NOISY estimator, not a precision knob: the sample is 8
        /// steps taken at the least predictable point of a response, so a few
        /// percent either way carries no information. It was 1.02, which let
        /// ordinary sampling noise park a drafter that was measurably 2.7x faster
        /// on the very next probe.
        /// </summary>
        private const double SpecWinMargin = 1.15;

        // Ratio of SUMS, not a mean of per-step rates. A speculative step emits
        // between 1 and MaxDraftTokens+1 tokens; normalising each step to ms/token
        // BEFORE averaging weights a fully-rejected step (1 token, full step cost)
        // exactly as heavily as a fully-accepted one (16 tokens, barely more cost),
        // which over-states speculation's cost and only speculation's - plain steps
        // always emit one token, so their mean of rates already equals their
        // aggregate. On a realistic 8-step probe (4 steps x 16 tokens, 4 steps x 1)
        // the old estimator read 29.4 ms/token where the true figure was 6.8.
        private long _plainTicks, _specTicks;
        private long _plainTokens, _specTokens;
        private int _plainSamples, _specSamples;

        // The single most expensive speculative sample of the round (by ms/token),
        // kept as its raw (ticks, tokens) so the verdict can drop the WHOLE step.
        // The verify graph is rebuilt whenever the batch shape or the padded KV
        // window changes, and both happen DURING a probe (k varies per step, and 8
        // speculative steps can advance the position across a window boundary), so
        // one sample in a round routinely carries a graph build that never recurs.
        private long _specWorstTicks;
        private int _specWorstTokens;

        // Whether the first sample of each side has already been discarded this
        // round. The first speculative step after a prefill pays four one-off graph
        // builds (trunk verify shape, drafter inject at n_rows=1, the draft block,
        // and possibly a KV-capacity grow that drops every persistent graph); the
        // first plain step after a run of verify batches finds the 1-row trunk
        // graph cold for the same reason.
        private bool _specFirstSkipped, _plainFirstSkipped;

        private int _parkedRemaining;
        private int _recheckRemaining;
        private int _consecutiveLosses;
        private bool _specWins = true;

        /// <summary>
        /// Measure speculation against plain decoding at runtime and skip drafting
        /// while it is measurably slower. On by default; set false to force the
        /// drafter on for A/B measurement.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>True while a losing verdict is actively suppressing drafting
        /// (as opposed to a round merely taking its plain baseline). Only these
        /// steps are reported as "parked".</summary>
        public bool IsParked => _parkedRemaining > 0;

        /// <summary>Measured ms per emitted token for the current round, plain
        /// side. 0 until the round has a sample.</summary>
        public double PlainMsPerToken { get; private set; }

        /// <summary>Measured ms per emitted token for the current round,
        /// speculative side. 0 until the round has a sample.</summary>
        public double SpecMsPerToken { get; private set; }

        /// <summary>Whether this step may draft.</summary>
        public bool AllowsSpeculation()
        {
            if (!Enabled)
                return true;
            if (_parkedRemaining > 0)
                return false;
            if (_recheckRemaining > 0)
                return _specWins;
            if (_specSamples < ProbeSpecSteps)
                return true;                        // measuring speculation
            if (_plainSamples < CalibrationPlainSteps)
                return false;                       // measuring the plain baseline
            return _specWins;
        }

        /// <summary>Record one completed decode step.</summary>
        public void Record(bool speculated, int tokensEmitted, long elapsedTicks)
        {
            if (!Enabled || tokensEmitted <= 0)
                return;

            if (_parkedRemaining > 0)
            {
                _parkedRemaining--;
                return;                             // parked steps are not samples
            }
            if (_recheckRemaining > 0)
            {
                _recheckRemaining--;
                return;
            }

            if (speculated)
            {
                if (!_specFirstSkipped) { _specFirstSkipped = true; return; }
                _specTicks += elapsedTicks;
                _specTokens += tokensEmitted;
                _specSamples++;
                if (_specWorstTokens == 0 ||
                    MsPerToken(elapsedTicks, tokensEmitted) > MsPerToken(_specWorstTicks, _specWorstTokens))
                {
                    _specWorstTicks = elapsedTicks;
                    _specWorstTokens = tokensEmitted;
                }
                SpecMsPerToken = MsPerToken(_specTicks, _specTokens);
            }
            else
            {
                if (!_plainFirstSkipped) { _plainFirstSkipped = true; return; }
                _plainTicks += elapsedTicks;
                _plainTokens += tokensEmitted;
                _plainSamples++;
                PlainMsPerToken = MsPerToken(_plainTicks, _plainTokens);
            }

            if (_specSamples < ProbeSpecSteps || _plainSamples < CalibrationPlainSteps)
                return;

            // Trim the worst speculative sample (a graph rebuild that will not
            // recur) before comparing. Only when it leaves something to compare.
            double specMs = MsPerToken(_specTicks, _specTokens);
            if (_specSamples > 1)
            {
                long keptTicks = _specTicks - _specWorstTicks;
                long keptTokens = _specTokens - _specWorstTokens;
                if (keptTicks > 0 && keptTokens > 0)
                    specMs = Math.Min(specMs, MsPerToken(keptTicks, keptTokens));
            }

            _specWins = specMs <= MsPerToken(_plainTicks, _plainTokens) * SpecWinMargin;
            _consecutiveLosses = _specWins ? 0 : _consecutiveLosses + 1;
            _parkedRemaining = _specWins
                ? 0
                : Math.Min(ParkedProbeInterval, FirstParkInterval << Math.Min(_consecutiveLosses - 1, 4));
            _recheckRemaining = _specWins ? WinRecheckInterval : 0;
            ResetRound();
        }

        /// <summary>
        /// Drop every verdict and start measuring from scratch. A park decided on
        /// one turn describes that turn's context length and prompt, and carrying
        /// it into the next request means a fresh generation can start with
        /// drafting disabled for no reason.
        /// </summary>
        public void Reset()
        {
            ResetRound();
            _parkedRemaining = _recheckRemaining = 0;
            _consecutiveLosses = 0;
            _specWins = true;
            // A fresh governor has measured nothing; leaving the last run's
            // rates behind would make a per-turn log report numbers from the
            // previous turn's context length.
            PlainMsPerToken = SpecMsPerToken = 0;
        }

        private void ResetRound()
        {
            _specSamples = _plainSamples = 0;
            _specTicks = _plainTicks = 0;
            _specTokens = _plainTokens = 0;
            _specWorstTicks = 0;
            _specWorstTokens = 0;
            _specFirstSkipped = _plainFirstSkipped = false;
        }

        private static double MsPerToken(long ticks, long tokens) =>
            tokens > 0 ? ticks * 1000.0 / Stopwatch.Frequency / tokens : 0.0;
    }
}
