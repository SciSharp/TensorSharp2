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
    /// Weight-free speculation by suffix matching over the sequence's own
    /// tokens - "prompt lookup" / n-gram decoding (llama.cpp's lookup decoding,
    /// vLLM's ngram proposer).
    ///
    /// This is the algorithm that proves the layering: it needs no draft head,
    /// no trained weights and no hidden states, so it works on EVERY model that
    /// implements <see cref="ISpeculativeTarget"/>, including every checkpoint
    /// that ships without a speculator. Learned drafters (MTP, DFlash, DSpark,
    /// EAGLE) are bound to one target model and cannot transfer; this one is
    /// bound to nothing.
    ///
    /// How it works: take the last n tokens ending at the current one, find
    /// where that same n-gram last occurred earlier in the context, and propose
    /// whatever followed it. On the workloads where that is true - summarizing,
    /// editing, translating or answering about a document, repetitive
    /// structured output, code with repeated identifiers, agentic tool loops -
    /// long verbatim spans get accepted for the price of one trunk pass. On
    /// free-form prose it simply finds nothing and every step degrades to a
    /// plain decode, which the cost governor then keeps cheap.
    ///
    /// Lookup is O(1) per step: an incrementally maintained hash index per
    /// n-gram order maps a rolling hash to the position after its most recent
    /// occurrence, and every hit is verified token-by-token so a hash collision
    /// can only cost a rejected draft, never a wrong output.
    ///
    /// Memory: one dictionary entry per (n-gram order, position), so roughly
    /// <c>(MaxNgram - MinNgram + 1)</c> entries per committed token - about 30 MB
    /// at a 100k-token context with the defaults. That is small next to the model
    /// but not free, which is one more reason speculation is opt-in.
    /// </summary>
    public sealed class NGramSpeculator : ISpeculator
    {
        /// <summary>
        /// Shortest suffix that may match. Two tokens is where matches stop
        /// being coincidences often enough to be worth a verify row.
        /// </summary>
        public const int DefaultMinNgram = 2;

        /// <summary>Longest suffix tried first; a longer match is a stronger
        /// signal, so the search runs from here downwards.</summary>
        public const int DefaultMaxNgram = 8;

        /// <summary>
        /// Default gate. 0 means "accept a match at <see cref="MinNgram"/>",
        /// which is the right default because n-gram drafts cost only a verify
        /// row that would otherwise have been a plain decode step anyway. See
        /// <see cref="MinDraftProb"/> for how a higher value is interpreted.
        /// </summary>
        public const float DefaultGate = 0f;

        private readonly int _minNgram;
        private readonly int _maxNgram;

        // Committed tokens (prompt + accepted output), in order. This is the
        // corpus the suffix search runs against.
        private readonly List<int> _tokens = new();

        // One index per n-gram order in [MinNgram, MaxNgram]: rolling hash of
        // the n tokens ending at position j -> j+1, i.e. the position of the
        // token that FOLLOWED that n-gram. Most recent occurrence wins, which
        // is what makes the drafter track the local structure of the text
        // rather than something from the top of the document.
        private readonly Dictionary<ulong, int>[] _index;

        // Scratch for the pattern being looked up (last n-1 committed tokens
        // plus the not-yet-committed last token).
        private readonly int[] _pattern;

        public NGramSpeculator(int maxDraftTokens,
            int minNgram = DefaultMinNgram, int maxNgram = DefaultMaxNgram)
        {
            if (maxDraftTokens < 1)
                throw new ArgumentOutOfRangeException(nameof(maxDraftTokens));
            if (minNgram < 1)
                throw new ArgumentOutOfRangeException(nameof(minNgram));
            if (maxNgram < minNgram)
                throw new ArgumentOutOfRangeException(nameof(maxNgram));

            MaxDraftTokens = maxDraftTokens;
            _minNgram = minNgram;
            _maxNgram = maxNgram;
            _index = new Dictionary<ulong, int>[maxNgram + 1];
            for (int n = minNgram; n <= maxNgram; n++)
                _index[n] = new Dictionary<ulong, int>();
            _pattern = new int[maxNgram];
        }

        public string Name => SpeculatorRegistry.NGram;

        public string Describe() => $"ngram(n={_minNgram}..{_maxNgram})";

        /// <summary>Shortest suffix that may match.</summary>
        public int MinNgram => _minNgram;

        /// <summary>Longest suffix tried first.</summary>
        public int MaxNgram => _maxNgram;

        public int MaxDraftTokens { get; }

        /// <summary>
        /// Interpreted as a required MATCH STRENGTH: a hit only counts when the
        /// suffix that matched is at least
        /// <c>ceil(MinDraftProb * MaxNgram)</c> tokens long (never below
        /// <see cref="MinNgram"/>). 0 - the default - accepts any match at
        /// <see cref="MinNgram"/>; 1 demands a full <see cref="MaxNgram"/>
        /// match. There is no probability to threshold here, so reusing the
        /// same knob as a confidence gate would be a false precision; this
        /// mapping at least makes the knob monotonic in the same direction
        /// (higher = draft less, reject less).
        /// </summary>
        public float MinDraftProb { get; set; } = DefaultGate;

        public float DefaultMinDraftProb => DefaultGate;

        public bool NeedsHiddenState => false;

        public bool HandlesOwnPrefill => false;

        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            int committed = _tokens.Count;
            // The loop drives Commit for every token it puts in the trunk, so a
            // disagreement here means the speculator was attached mid-sequence;
            // draft nothing rather than mine a corpus that is not this
            // sequence's.
            if (committed != ctx.Position)
                return 0;

            int required = RequiredMatchLength();
            int longest = Math.Min(_maxNgram, committed + 1);

            for (int n = longest; n >= required; n--)
            {
                // Pattern = the n tokens ending at LastToken. LastToken is not
                // in _tokens yet (it is the token the verify batch starts with),
                // so it is appended here.
                for (int i = 0; i < n - 1; i++)
                    _pattern[i] = _tokens[committed - (n - 1) + i];
                _pattern[n - 1] = ctx.LastToken;

                if (!_index[n].TryGetValue(Hash(_pattern, n), out int after))
                    continue;
                // Verify the match token-by-token: the index is a hash map, so a
                // collision must cost a rejected draft, never a wrong proposal.
                if (!MatchesAt(after - n, n))
                    continue;

                int take = Math.Min(ctx.MaxTokens, committed - after);
                for (int i = 0; i < take; i++)
                    draftOut.Add(_tokens[after + i]);
                if (draftOut.Count > 0)
                    return draftOut.Count;
            }
            return 0;
        }

        public void Commit(int[] tokens, float[] hRows, int startPos)
        {
            if (tokens == null || tokens.Length == 0)
                return;
            // Commit is the loop's single source of truth for "these tokens are
            // now in the trunk", so a rolled-back window never reaches the
            // corpus and the index needs no undo.
            if (startPos != _tokens.Count)
            {
                if (startPos < _tokens.Count)
                    RebuildTo(startPos);
                else
                    return;     // a gap we cannot account for: stay out of the way
            }

            foreach (int t in tokens)
            {
                _tokens.Add(t);
                IndexLast();
            }
        }

        public void Reset()
        {
            _tokens.Clear();
            for (int n = _minNgram; n <= _maxNgram; n++)
                _index[n].Clear();
        }

        public void Dispose() { }

        private int RequiredMatchLength()
        {
            if (MinDraftProb <= 0f)
                return _minNgram;
            int scaled = (int)Math.Ceiling(Math.Min(1f, MinDraftProb) * _maxNgram);
            return Math.Clamp(scaled, _minNgram, _maxNgram);
        }

        /// <summary>Index every n-gram ENDING at the last committed token.</summary>
        private void IndexLast() => IndexNgramsEndingAt(_tokens.Count);

        /// <summary>Shrink the corpus back to <paramref name="length"/> tokens
        /// and re-index. Only reachable if a caller rewinds the trunk without
        /// resetting the speculator; a full re-index is linear and this is not a
        /// hot path.</summary>
        private void RebuildTo(int length)
        {
            _tokens.RemoveRange(length, _tokens.Count - length);
            for (int n = _minNgram; n <= _maxNgram; n++)
                _index[n].Clear();
            int total = _tokens.Count;
            for (int end = 1; end <= total; end++)
                IndexNgramsEndingAt(end);
        }

        /// <summary>Index every n-gram whose last token is at
        /// <paramref name="end"/> - 1.</summary>
        private void IndexNgramsEndingAt(int end)
        {
            for (int n = _minNgram; n <= _maxNgram; n++)
            {
                int start = end - n;
                if (start < 0)
                    break;
                for (int i = 0; i < n; i++)
                    _pattern[i] = _tokens[start + i];
                _index[n][Hash(_pattern, n)] = end;
            }
        }

        private bool MatchesAt(int start, int n)
        {
            if (start < 0 || start + n > _tokens.Count)
                return false;
            for (int i = 0; i < n; i++)
            {
                if (_tokens[start + i] != _pattern[i])
                    return false;
            }
            return true;
        }

        /// <summary>FNV-1a over the token ids. Cheap, and collisions are
        /// verified away by <see cref="MatchesAt"/>.</summary>
        private static ulong Hash(int[] tokens, int count)
        {
            ulong h = 1469598103934665603UL;
            for (int i = 0; i < count; i++)
            {
                uint v = unchecked((uint)tokens[i]);
                for (int b = 0; b < 4; b++)
                {
                    h ^= (byte)(v >> (b * 8));
                    h *= 1099511628211UL;
                }
            }
            return h;
        }
    }
}
