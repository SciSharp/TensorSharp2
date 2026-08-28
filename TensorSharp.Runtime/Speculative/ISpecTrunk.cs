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

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// The trunk backend a speculative execution drives: prompt/verify/plain
    /// forwards plus recurrent-state snapshot/rollback. Two implementations:
    /// <see cref="LinearSpecTrunk"/> (the model's live linear cache - the
    /// standalone decoder and the per-sequence engine fallback) and the
    /// executor's batched trunk (paged KV + per-slot state via
    /// <see cref="IBatchedSpeculativeTarget"/>).
    ///
    /// Separating this from <see cref="ISpeculativeTarget"/> is what lets one
    /// draft/verify loop serve both KV regimes: the loop only ever asks for
    /// "forward these tokens" and "undo back to here".
    /// </summary>
    public interface ISpecTrunk
    {
        /// <summary>Forward <paramref name="tokens"/> at the trunk's current
        /// position, capturing per-row hidden states and logits like
        /// <see cref="ISpeculativeTarget.SpecForward"/>. Advances the trunk
        /// by <c>tokens.Length</c>.</summary>
        void Forward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows);

        /// <summary>Snapshot recurrent state before a verify batch.</summary>
        void SnapshotRecurrentState();

        /// <summary>
        /// The accept count for the verify that just ran, handed to the trunk
        /// BEFORE any rollback decision and on EVERY step - full acceptance
        /// included. A trunk that defers part of its post-verify bookkeeping until
        /// it knows how much was accepted settles it here; Qwen 3.5/3.8 uses it to
        /// pick the recurrent-state snapshot for the accepted prefix, which is what
        /// lets <see cref="TryCommitVerifiedPrefix"/> then succeed on a recurrent
        /// model at all. Default: nothing to do.
        /// </summary>
        void OnVerifyAccepted(int acceptedRows, int verifyRows) { }

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
    /// <see cref="ISpeculativeTarget.SpecForward"/> on the model's single
    /// live KV cache.</summary>
    public sealed class LinearSpecTrunk : ISpecTrunk
    {
        private readonly ISpeculativeTarget _model;

        public LinearSpecTrunk(ISpeculativeTarget model)
            => _model = model ?? throw new ArgumentNullException(nameof(model));

        public void Forward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
            => _model.SpecForward(tokens, hAllOut, logitsOut, allLogitsRows);

        public void SnapshotRecurrentState() => _model.SpecSnapshotRecurrentState();

        public void OnVerifyAccepted(int acceptedRows, int verifyRows)
            => _model.SpecOnVerifyAccepted(acceptedRows, verifyRows);

        public void Rollback(int position)
        {
            _model.SpecRestoreRecurrentState();
            _model.SpecRewindCache(position);
        }

        public bool TryCommitVerifiedPrefix(int newPosition)
        {
            if (!_model.SpecVerifyPersistsAcceptedKv)
                return false;
            // The verify wrote correct KV for all batch tokens; the accepted prefix's
            // KV is already live. Just drop the rejected tail by rewinding the position
            // (rejected slots are overwritten by later writes and never read past the
            // live position). No recurrent state to restore for such models.
            _model.SpecRewindCache(newPosition);
            return true;
        }
    }
}
