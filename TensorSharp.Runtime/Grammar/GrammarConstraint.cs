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

namespace TensorSharp.Runtime.Grammar
{
    /// <summary>
    /// Per-(grammar, vocabulary) cache of everything that does not depend on a
    /// particular request: interned parser states, character transitions between
    /// them, and the token bitmask each state admits. Thread-safe and shared by
    /// every request using the same grammar.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Three layers of reuse, in increasing order of value:
    /// </para>
    /// <list type="number">
    /// <item><b>State interning</b> — equal states become the same object, so the
    /// two caches below can key on them cheaply and the mask cache actually
    /// hits.</item>
    /// <item><b>Transition memo</b> — <c>(state, code point) → state</c>. Without
    /// it the trie walk would re-derive the same character transition once per
    /// trie node; inside a JSON string that is hundreds of thousands of
    /// redundant grammar advances for a few hundred distinct characters.</item>
    /// <item><b>Mask cache</b> — <c>state → bitmask</c>. JSON generation cycles
    /// through a handful of states ("expecting a key", "inside a string", "after
    /// a comma"), so in steady state producing a mask is a dictionary lookup.
    /// This is the adaptive-cache idea from xgrammar.</item>
    /// </list>
    /// </remarks>
    public sealed class GrammarMaskCache
    {
        private readonly GrammarMatcher _matcher;
        private readonly GrammarTokenVocabulary _vocab;

        private readonly Dictionary<GrammarState, GrammarState> _interned = new();
        private readonly Dictionary<TransitionKey, GrammarState> _transitions = new();
        private readonly Dictionary<GrammarState, ulong[]> _masks = new();
        private readonly object _lock = new();

        /// <summary>
        /// Direct-mapped transitions for ASCII, which is nearly every edge in the
        /// trie: an array index instead of hashing a (state, code point) tuple.
        /// Null = not yet computed. Measured worth only ~2% of a cold mask on a
        /// 592k-node trie — the walk itself dominates, not the lookup — but it is
        /// a few KB per state and removes the tuple hashing from the inner loop.
        /// </summary>
        private readonly Dictionary<GrammarState, GrammarState?[]> _asciiTransitions = new();
        private const int AsciiLimit = 128;

        /// <summary>Cap on retained masks; each costs vocabSize/8 bytes.</summary>
        private const int MaxCachedMasks = 4096;

        public GrammarMaskCache(Grammar grammar, GrammarTokenVocabulary vocab)
        {
            if (grammar == null) throw new ArgumentNullException(nameof(grammar));
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            _matcher = new GrammarMatcher(grammar);
            InitialState = Intern(_matcher.InitialState);
        }

        public GrammarState InitialState { get; }
        public GrammarTokenVocabulary Vocabulary => _vocab;
        public GrammarMatcher Matcher => _matcher;

        private readonly struct TransitionKey : IEquatable<TransitionKey>
        {
            public readonly GrammarState State;
            public readonly uint CodePoint;
            public TransitionKey(GrammarState s, uint c) { State = s; CodePoint = c; }
            public bool Equals(TransitionKey o) =>
                ReferenceEquals(State, o.State) && CodePoint == o.CodePoint;
            public override bool Equals(object? o) => o is TransitionKey k && Equals(k);
            public override int GetHashCode() =>
                System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(State) * 397 ^ (int)CodePoint;
        }

        private GrammarState Intern(GrammarState state)
        {
            if (_interned.TryGetValue(state, out GrammarState? existing)) return existing;
            _interned[state] = state;
            return state;
        }

        /// <summary>Advance one code point, memoized.</summary>
        public GrammarState Advance(GrammarState state, uint codePoint)
        {
            lock (_lock)
            {
                return AdvanceLocked(state, codePoint);
            }
        }

        private GrammarState AdvanceLocked(GrammarState state, uint codePoint)
        {
            if (codePoint < AsciiLimit)
            {
                if (!_asciiTransitions.TryGetValue(state, out GrammarState?[]? row))
                {
                    row = new GrammarState?[AsciiLimit];
                    _asciiTransitions[state] = row;
                }
                GrammarState? hit = row[codePoint];
                if (hit != null) return hit;
                GrammarState computed = Intern(_matcher.AcceptCodePoint(state, codePoint));
                row[codePoint] = computed;
                return computed;
            }

            var key = new TransitionKey(state, codePoint);
            if (_transitions.TryGetValue(key, out GrammarState? next)) return next;
            next = Intern(_matcher.AcceptCodePoint(state, codePoint));
            _transitions[key] = next;
            return next;
        }

        /// <summary>
        /// Bitmask of the tokens admitted in <paramref name="state"/>. Bit
        /// <c>i</c> of word <c>i&gt;&gt;6</c> set means token <c>i</c> is legal.
        /// The returned array is shared and must not be modified.
        /// </summary>
        public ulong[] GetMask(GrammarState state)
        {
            lock (_lock)
            {
                if (_masks.TryGetValue(state, out ulong[]? cached)) return cached;

                var mask = new ulong[_vocab.MaskWords];
                // Walk the shared-prefix trie once, carrying the grammar state
                // down each branch and abandoning a subtree the moment its prefix
                // stops being derivable.
                Descend(0, state, PartialUtf8.Empty, mask);

                if (_masks.Count >= MaxCachedMasks) _masks.Clear();
                _masks[state] = mask;
                return mask;
            }
        }

        private void Descend(int node, GrammarState state, PartialUtf8 partial, ulong[] mask)
        {
            int end = _vocab.ChildEnd(node);
            for (int edge = _vocab.ChildStart(node); edge < end; edge++)
            {
                byte b = _vocab.EdgeByte(edge);

                PartialUtf8 nextPartial = partial;
                if (!GrammarMatcher.TryFeedByte(ref nextPartial, b, out uint codePoint, out bool complete))
                    continue;                       // malformed UTF-8: no token below this can be text

                GrammarState nextState;
                if (complete)
                {
                    nextState = AdvanceLocked(state, codePoint);
                    if (nextState.IsDead) continue; // prunes the entire subtree
                }
                else
                {
                    // Mid-character: keep the branch only if some completion of
                    // this partial code point could still satisfy the grammar.
                    if (!_matcher.AcceptsPartial(state, nextPartial)) continue;
                    nextState = state;
                }

                int child = _vocab.EdgeNode(edge);
                int tokenId = _vocab.TokenAt(child);
                // A token is legal only if it ends on a complete character; one
                // that stops mid-character leaves the matcher unable to commit.
                if (tokenId >= 0 && nextPartial.Remaining == 0)
                    mask[tokenId >> 6] |= 1UL << (tokenId & 63);

                Descend(child, nextState, nextPartial, mask);
            }
        }
    }

    /// <summary>
    /// Per-request grammar constraint: the live parser position plus the shared
    /// caches. One instance per sequence; not thread-safe on its own, which is
    /// fine because a sequence decodes serially.
    /// </summary>
    public sealed class GrammarConstraint
    {
        private readonly GrammarMaskCache _cache;
        private GrammarState _state;
        private PartialUtf8 _partial;
        private readonly List<byte> _tokenBytes = new(64);
        private readonly ITokenizer _tokenizer;

        public GrammarConstraint(Grammar grammar, ITokenizer tokenizer)
            : this(new GrammarMaskCache(grammar, GrammarTokenVocabulary.ForTokenizer(tokenizer)), tokenizer)
        {
        }

        public GrammarConstraint(GrammarMaskCache cache, ITokenizer tokenizer)
        {
            _cache = cache ?? throw new ArgumentNullException(nameof(cache));
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _state = cache.InitialState;
            _partial = PartialUtf8.Empty;
        }

        /// <summary>The grammar can legally end here, so EOS is permitted.</summary>
        public bool IsComplete => _partial.Remaining == 0 && _state.CanTerminate;

        /// <summary>
        /// No continuation exists. Should not happen while the sampler honours
        /// the mask; it is the signal that something bypassed the constraint.
        /// </summary>
        public bool IsDead => _state.IsDead;

        /// <summary>Current mask; bit <c>i</c> set means token <c>i</c> is legal.</summary>
        public ulong[] CurrentMask() => _cache.GetMask(_state);

        /// <summary>Opaque snapshot for rollback (speculative decoding).</summary>
        public (GrammarState State, PartialUtf8 Partial) Snapshot() => (_state, _partial);

        /// <summary>Restore a snapshot taken earlier.</summary>
        public void Restore((GrammarState State, PartialUtf8 Partial) snapshot)
        {
            _state = snapshot.State;
            _partial = snapshot.Partial;
        }

        /// <summary>
        /// Advance the parser over raw UTF-8 bytes. Use when text has already
        /// been produced outside the sampler — a forced prefix, a prefilled
        /// partial response being resumed, or a tool-call envelope the server
        /// emitted itself — so the constraint stays aligned with what the client
        /// will actually see.
        /// </summary>
        public void AcceptBytes(ReadOnlySpan<byte> bytes)
        {
            foreach (byte b in bytes)
            {
                if (!GrammarMatcher.TryFeedByte(ref _partial, b, out uint cp, out bool complete))
                {
                    _state = new GrammarState(Array.Empty<int[]>());
                    return;
                }
                if (!complete) continue;
                _state = _cache.Advance(_state, cp);
                if (_state.IsDead) return;
            }
        }

        /// <summary>
        /// Commit a token the sampler chose, advancing the parser. EOS and other
        /// special ids carry no text and leave the state untouched.
        /// </summary>
        public void Accept(int tokenId)
        {
            if (_tokenizer.IsEos(tokenId)) return;

            _tokenBytes.Clear();
            try
            {
                _tokenizer.AppendTokenBytes(tokenId, _tokenBytes);
            }
            catch
            {
                return;
            }
            if (_tokenBytes.Count == 0) return;

            for (int i = 0; i < _tokenBytes.Count; i++)
            {
                if (!GrammarMatcher.TryFeedByte(ref _partial, _tokenBytes[i], out uint cp, out bool complete))
                {
                    _state = new GrammarState(Array.Empty<int[]>());
                    return;
                }
                if (!complete) continue;
                _state = _cache.Advance(_state, cp);
                if (_state.IsDead) return;
            }
        }

        /// <summary>
        /// Set every token the grammar forbids to negative infinity, so any
        /// downstream sampler — greedy, top-k, top-p — can only pick a legal one.
        /// </summary>
        /// <param name="logits">Logit buffer, modified in place.</param>
        /// <param name="allowEos">
        /// Whether EOS ids may survive; callers pass <see cref="IsComplete"/>.
        /// </param>
        public void ApplyMask(float[] logits, bool allowEos)
        {
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            ulong[] mask = CurrentMask();
            int vocab = Math.Min(logits.Length, _cache.Vocabulary.VocabSize);

            // EOS ids are deliberately absent from the trie, so the sweep below
            // would bury them. Save the model's own scores first and put them
            // back afterwards: whether generation should stop is a judgement the
            // grammar can only *permit*, not make. Overwriting the logit with a
            // constant would either force an early stop or suppress a wanted one.
            int[] eosIds = _tokenizer.EosTokenIds;
            Span<float> savedEos = eosIds.Length <= 8 ? stackalloc float[eosIds.Length] : new float[eosIds.Length];
            if (allowEos)
            {
                for (int i = 0; i < eosIds.Length; i++)
                    savedEos[i] = (uint)eosIds[i] < (uint)logits.Length
                        ? logits[eosIds[i]]
                        : float.NegativeInfinity;
            }

            for (int word = 0; word < mask.Length; word++)
            {
                ulong bits = mask[word];
                int baseId = word << 6;
                int limit = Math.Min(64, vocab - baseId);
                if (limit <= 0) break;
                if (bits == ulong.MaxValue && limit == 64) continue;   // every id legal
                for (int b = 0; b < limit; b++)
                {
                    if ((bits & (1UL << b)) == 0)
                        logits[baseId + b] = float.NegativeInfinity;
                }
            }

            if (allowEos)
            {
                for (int i = 0; i < eosIds.Length; i++)
                    if ((uint)eosIds[i] < (uint)logits.Length)
                        logits[eosIds[i]] = savedEos[i];
            }
        }
    }
}
