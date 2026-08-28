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

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Per-token speculation through a learned draft head that chains its own
    /// hidden output: a NextN/MTP block (Qwen 3.6, GLM-5.2, Gemma 4's separate
    /// assistant GGUF - llama.cpp's <c>--spec-type draft-mtp</c>, vLLM's
    /// <c>qwen3_5_mtp</c> speculator) and, unchanged, an EAGLE-style head,
    /// which differs only in what the weights behind
    /// <see cref="IDraftHead.DraftStep"/> compute.
    ///
    /// One pass per drafted token: feed (token, hidden of the token before it),
    /// take the argmax, feed that token and the head's own hidden output back
    /// in. Drafting stops at the first token whose confidence - the top-1
    /// probability over the head's top-10 logits, which is what llama.cpp's
    /// top-k(10) draft sampler thresholds with p_min - falls below
    /// <see cref="MinDraftProb"/>, so a longer window only ever extends a
    /// confident streak.
    /// </summary>
    public sealed class DraftHeadSpeculator : ISpeculator
    {
        /// <summary>
        /// Default gate for a per-token head: the top-1 probability over the
        /// head's top-10 logits, thresholded per drafted token (llama.cpp's
        /// top-k(10) draft sampler with p_min).
        /// </summary>
        public const float DefaultGate = 0.75f;

        private readonly IDraftHead _head;
        private readonly int _vocab;
        private readonly int _featureSize;

        // Reusable buffers: one logits row and the two hidden rows the chain
        // ping-pongs between. Speculative windows are small, so these are
        // allocated once per sequence and never resized.
        private readonly float[] _logits;
        private readonly float[] _hA;
        private readonly float[] _hB;

        // Catch-up folding (llama.cpp's draft-mtp, which runs its block over
        // n_accepted + 1 rows instead of a catch-up pass plus a first draft step).
        // Commit stashes the verified run instead of replaying it; the next Propose
        // appends the token it starts from and does both in ONE head call. Worth a
        // whole head call per speculative step, which on Qwen 3.8 is 6.4 ms of 100.
        private readonly bool _fold;
        private int[] _pendTokens;
        private float[] _pendH;
        private int _pendCount;
        private int _pendStart;
        private bool _hasPend;
        // The folded call's inputs: the stashed run plus one row.
        private int[] _foldTokens;
        private float[] _foldH;

        public DraftHeadSpeculator(IDraftHead head, int vocabSize, int featureSize, int maxDraftTokens)
        {
            _head = head ?? throw new ArgumentNullException(nameof(head));
            if (maxDraftTokens < 1)
                throw new ArgumentOutOfRangeException(nameof(maxDraftTokens));
            _vocab = vocabSize;
            _featureSize = featureSize;
            MaxDraftTokens = maxDraftTokens;
            _fold = head.SupportsFusedCatchUpStep;
            _logits = new float[vocabSize];
            _hA = new float[featureSize];
            _hB = new float[featureSize];
        }

        public string Name => SpeculatorRegistry.DraftHead;

        public string Describe() => "per-token";

        public int MaxDraftTokens { get; }

        public float MinDraftProb { get; set; } = DefaultGate;

        public float DefaultMinDraftProb => DefaultGate;

        public bool NeedsHiddenState => true;

        public bool HandlesOwnPrefill => _head.DraftSelfCatchUp;

        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            float[] hIn = ctx.CarryHidden;
            float[] hOut = _hA;
            int tokIn = ctx.LastToken;
            int first = 0;

            // A stashed catch-up whose rows run right up to this step's position
            // folds into draft step 0: one head call replays the verified run AND
            // produces the first draft. Any other position means some path put a
            // step in between, so replay it on its own and draft normally.
            if (_hasPend)
            {
                if (_pendStart + _pendCount != ctx.Position)
                {
                    FlushPending();
                }
                else
                {
                    int n = _pendCount + 1;
                    // EXACTLY n: the head takes its row count from tokens.Length, so a
                    // buffer left long by an earlier, longer step would replay stale
                    // trailing tokens as if they were verified rows.
                    if (_foldTokens == null || _foldTokens.Length != n)
                        _foldTokens = new int[n];
                    if (_foldH == null || _foldH.Length < (long)n * _featureSize)
                        _foldH = new float[(long)n * _featureSize];
                    Array.Copy(_pendTokens, _foldTokens, _pendCount);
                    _foldTokens[_pendCount] = ctx.LastToken;
                    Array.Copy(_pendH, _foldH, (long)_pendCount * _featureSize);
                    Array.Copy(ctx.CarryHidden, 0, _foldH, (long)_pendCount * _featureSize, _featureSize);
                    _hasPend = false;

                    _head.DraftCatchUpAndStep(_foldTokens, _foldH, _pendStart, _logits, hOut);
                    if (ctx.MaxTokens < 1)
                        return 0;
                    ctx.AdjustLogits?.Invoke(_logits, draftOut);
                    int d0 = ArgmaxWithTopKConfidence(_logits, _vocab, out float p0);
                    if (p0 < MinDraftProb)
                        return 0;
                    draftOut.Add(d0);
                    tokIn = d0;
                    float[] nx = ReferenceEquals(hOut, _hA) ? _hB : _hA;
                    hIn = hOut;
                    hOut = nx;
                    first = 1;
                }
            }

            for (int i = first; i < ctx.MaxTokens; i++)
            {
                _head.DraftStep(tokIn, hIn, ctx.Position + i, _logits, hOut);
                // Penalty-aligned drafting: argmax the SAME distribution
                // verification will draw from, or acceptance decays toward zero
                // as the output history grows.
                ctx.AdjustLogits?.Invoke(_logits, draftOut);

                int d = ArgmaxWithTopKConfidence(_logits, _vocab, out float p);
                if (p < MinDraftProb)
                    break;

                draftOut.Add(d);
                tokIn = d;
                // Chain the head's hidden output into the next draft step,
                // ping-ponging so neither buffer is read and written at once.
                float[] next = ReferenceEquals(hOut, _hA) ? _hB : _hA;
                hIn = hOut;
                hOut = next;
            }
            return draftOut.Count;
        }

        public void Commit(int[] tokens, float[] hRows, int startPos)
        {
            if (!_fold || hRows == null)
            {
                _head.DraftCatchUp(tokens, hRows, startPos);
                return;
            }
            // Two commits with no Propose between them (a governor-declined step
            // straight after a speculative one) must not lose the first.
            FlushPending();
            int n = tokens.Length;
            if (_pendTokens == null || _pendTokens.Length < n)
                _pendTokens = new int[n];
            if (_pendH == null || _pendH.Length < (long)n * _featureSize)
                _pendH = new float[(long)n * _featureSize];
            Array.Copy(tokens, _pendTokens, n);
            Array.Copy(hRows, _pendH, (long)n * _featureSize);
            _pendCount = n;
            _pendStart = startPos;
            _hasPend = true;
        }

        /// <summary>Replay a stashed catch-up on its own, for when it cannot be
        /// folded into the next draft (or there is no next draft).</summary>
        private void FlushPending()
        {
            if (!_hasPend)
                return;
            _hasPend = false;
            var toks = new int[_pendCount];
            Array.Copy(_pendTokens, toks, _pendCount);
            _head.DraftCatchUp(toks, _pendH, _pendStart);
        }

        public void Reset() => FlushPending();

        public void Dispose() => _hasPend = false;

        /// <summary>
        /// Argmax plus the top-1 probability computed over the top-10 logits
        /// (softmax restricted to the 10 best candidates - the same confidence
        /// measure llama.cpp's draft-mtp top-k(10) sampler thresholds with
        /// p_min).
        /// </summary>
        internal static int ArgmaxWithTopKConfidence(float[] logits, int vocab, out float prob)
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
