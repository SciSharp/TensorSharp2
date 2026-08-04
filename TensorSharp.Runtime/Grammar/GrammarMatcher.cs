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
    /// Partially decoded UTF-8 sequence: the bits accumulated so far and how
    /// many continuation bytes are still outstanding.
    /// </summary>
    /// <remarks>
    /// Needed because a token's bytes may end mid-code-point. Byte-fallback
    /// vocabularies emit single bytes (<c>&lt;0xE4&gt;</c>), so a multi-byte
    /// character routinely spans several tokens; the grammar, which matches
    /// whole code points, must be able to say "this prefix is still viable"
    /// without having seen the full character yet.
    /// </remarks>
    public readonly struct PartialUtf8 : IEquatable<PartialUtf8>
    {
        /// <summary>Bits accumulated so far (unshifted).</summary>
        public readonly uint Value;

        /// <summary>Continuation bytes still expected; 0 = none, -1 = invalid.</summary>
        public readonly int Remaining;

        public PartialUtf8(uint value, int remaining) { Value = value; Remaining = remaining; }

        public static PartialUtf8 Empty => new PartialUtf8(0, 0);
        public bool IsInvalid => Remaining < 0;

        public bool Equals(PartialUtf8 other) => Value == other.Value && Remaining == other.Remaining;
        public override bool Equals(object? obj) => obj is PartialUtf8 p && Equals(p);
        public override int GetHashCode() => (int)(Value * 31) ^ Remaining;
    }

    /// <summary>
    /// The set of parser stacks a grammar can currently be in — the matcher's
    /// entire state, and the key its token masks are cached under.
    /// </summary>
    /// <remarks>
    /// A grammar is ambiguous in general, so the matcher tracks every viable
    /// parse simultaneously (a pushdown automaton with a set of stacks) rather
    /// than backtracking. Stacks are <c>int[]</c> of flattened element
    /// positions, kept sorted and deduplicated so that two states reached by
    /// different routes compare and hash equal — which is what lets the mask
    /// cache hit. An <b>empty</b> stack in the set means the grammar may
    /// legally end here.
    /// </remarks>
    public sealed class GrammarState : IEquatable<GrammarState>
    {
        internal readonly int[][] Stacks;
        private readonly int _hash;

        internal GrammarState(int[][] stacks)
        {
            Stacks = stacks;
            unchecked
            {
                int h = 17;
                foreach (int[] s in stacks)
                {
                    h = h * 31 + s.Length;
                    foreach (int p in s) h = h * 31 + p;
                }
                _hash = h;
            }
        }

        /// <summary>No viable parse remains: nothing can be accepted from here.</summary>
        public bool IsDead => Stacks.Length == 0;

        /// <summary>True when the grammar may terminate in this state.</summary>
        public bool CanTerminate
        {
            get
            {
                foreach (int[] s in Stacks)
                    if (s.Length == 0) return true;
                return false;
            }
        }

        public bool Equals(GrammarState? other)
        {
            if (other is null || other._hash != _hash || other.Stacks.Length != Stacks.Length)
                return false;
            for (int i = 0; i < Stacks.Length; i++)
            {
                int[] a = Stacks[i], b = other.Stacks[i];
                if (a.Length != b.Length) return false;
                for (int j = 0; j < a.Length; j++)
                    if (a[j] != b[j]) return false;
            }
            return true;
        }

        public override bool Equals(object? obj) => Equals(obj as GrammarState);
        public override int GetHashCode() => _hash;
    }

    /// <summary>
    /// Advances a <see cref="Grammar"/> over code points and computes, for any
    /// state, which characters may come next. Pure and stateless with respect to
    /// a request: every method takes and returns a <see cref="GrammarState"/>,
    /// so states can be cached, shared and rolled back (needed for speculative
    /// decoding, where drafted tokens are provisionally accepted and may be
    /// rejected).
    /// </summary>
    public sealed class GrammarMatcher
    {
        private readonly Grammar _grammar;

        public GrammarMatcher(Grammar grammar)
        {
            _grammar = grammar ?? throw new ArgumentNullException(nameof(grammar));
            InitialState = BuildInitialState();
        }

        public Grammar Grammar => _grammar;

        /// <summary>State before any character has been matched.</summary>
        public GrammarState InitialState { get; }

        private GrammarState BuildInitialState()
        {
            var stacks = new List<int[]>();
            int start = _grammar.RuleStart(_grammar.RootRuleId);
            // Seed with the root rule's first alternate; AdvanceStack expands the
            // remaining alternates and every reachable non-terminal.
            var seed = _grammar.IsEndOfSequence(start)
                ? Array.Empty<int>()
                : new[] { start };
            AdvanceStack(seed, stacks);
            return Canonicalize(stacks);
        }

        /// <summary>
        /// Expand a stack until its top is a terminal (or it is empty),
        /// pushing every resulting stack into <paramref name="output"/>.
        /// Mirrors llama.cpp's <c>llama_grammar_advance_stack</c>.
        /// </summary>
        private void AdvanceStack(int[] stack, List<int[]> output)
        {
            var todo = new Stack<int[]>();
            todo.Push(stack);
            var seen = new HashSet<int[]>(IntArrayComparer.Instance);

            while (todo.Count > 0)
            {
                int[] cur = todo.Pop();
                if (!seen.Add(cur)) continue;

                if (cur.Length == 0)
                {
                    if (!ContainsStack(output, cur)) output.Add(cur);
                    continue;
                }

                int pos = cur[cur.Length - 1];
                GrammarElement e = _grammar[pos];

                switch (e.Type)
                {
                    case GrammarElementType.RuleRef:
                    {
                        int subpos = _grammar.RuleStart((int)e.Value);
                        while (true)
                        {
                            // Drop the non-terminal, keep what follows it, then
                            // descend into this alternate.
                            int extra = (!_grammar.IsEndOfSequence(pos + 1) ? 1 : 0) +
                                        (!_grammar.IsEndOfSequence(subpos) ? 1 : 0);
                            var next = new int[cur.Length - 1 + extra];
                            Array.Copy(cur, next, cur.Length - 1);
                            int k = cur.Length - 1;
                            if (!_grammar.IsEndOfSequence(pos + 1)) next[k++] = pos + 1;
                            if (!_grammar.IsEndOfSequence(subpos)) next[k] = subpos;
                            todo.Push(next);

                            while (!_grammar.IsEndOfSequence(subpos)) subpos++;
                            if (_grammar[subpos].Type == GrammarElementType.Alt) subpos++;
                            else break;
                        }
                        break;
                    }
                    case GrammarElementType.Char:
                    case GrammarElementType.CharNot:
                    case GrammarElementType.CharAny:
                        if (!ContainsStack(output, cur)) output.Add(cur);
                        break;
                    default:
                        throw new InvalidOperationException(
                            $"grammar stack left on element type {e.Type} at {pos}");
                }
            }
        }

        private static bool ContainsStack(List<int[]> list, int[] candidate)
        {
            foreach (int[] s in list)
                if (IntArrayComparer.Instance.Equals(s, candidate)) return true;
            return false;
        }

        private static GrammarState Canonicalize(List<int[]> stacks)
        {
            // Sorting makes states reached by different expansion orders compare
            // equal, which is what the mask cache depends on.
            stacks.Sort(CompareStacks);
            var unique = new List<int[]>(stacks.Count);
            for (int i = 0; i < stacks.Count; i++)
                if (i == 0 || CompareStacks(stacks[i - 1], stacks[i]) != 0)
                    unique.Add(stacks[i]);
            return new GrammarState(unique.ToArray());
        }

        private static int CompareStacks(int[] a, int[] b)
        {
            int n = Math.Min(a.Length, b.Length);
            for (int i = 0; i < n; i++)
                if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
            return a.Length.CompareTo(b.Length);
        }

        /// <summary>
        /// Match <paramref name="codePoint"/> against the char element at
        /// <paramref name="pos"/>, walking the whole class
        /// (<c>[a-z0-9_]</c> is one element run). Returns whether it matched and
        /// the position just past the class.
        /// </summary>
        private (bool Matched, int Next) MatchChar(int pos, uint codePoint)
        {
            GrammarElement e = _grammar[pos];
            bool isPositive = e.Type == GrammarElementType.Char ||
                              e.Type == GrammarElementType.CharAny;
            bool found = false;
            int p = pos;
            do
            {
                if (_grammar[p + 1].Type == GrammarElementType.CharRngUpper)
                {
                    found = found || (_grammar[p].Value <= codePoint && codePoint <= _grammar[p + 1].Value);
                    p += 2;
                }
                else if (_grammar[p].Type == GrammarElementType.CharAny)
                {
                    found = true;
                    p += 1;
                }
                else
                {
                    found = found || _grammar[p].Value == codePoint;
                    p += 1;
                }
            }
            while (_grammar[p].Type == GrammarElementType.CharAlt);

            return (found == isPositive, p);
        }

        /// <summary>
        /// Could <b>some</b> completion of a partially decoded code point satisfy
        /// the class at <paramref name="pos"/>? Used to keep a token alive whose
        /// bytes end mid-character.
        /// </summary>
        private bool MatchPartialChar(int pos, PartialUtf8 partial)
        {
            int remaining = partial.Remaining;
            if (remaining < 1 || remaining > 3) return false;

            // The value seen so far bounds the final code point to a range.
            uint low = partial.Value << (remaining * 6);
            uint high = low | ((1u << (remaining * 6)) - 1);
            if (low == 0)
            {
                low = remaining switch { 1 => 0x80, 2 => 0x800, 3 => 0x10000, _ => 0 };
            }

            GrammarElement e = _grammar[pos];
            bool isPositive = e.Type == GrammarElementType.Char ||
                              e.Type == GrammarElementType.CharAny;
            int p = pos;
            do
            {
                if (_grammar[p + 1].Type == GrammarElementType.CharRngUpper)
                {
                    if (Overlaps(_grammar[p].Value, _grammar[p + 1].Value, low, high) == isPositive)
                        return true;
                    p += 2;
                }
                else if (_grammar[p].Type == GrammarElementType.CharAny)
                {
                    return true;
                }
                else
                {
                    if (Overlaps(_grammar[p].Value, _grammar[p].Value, low, high) == isPositive)
                        return true;
                    p += 1;
                }
            }
            while (_grammar[p].Type == GrammarElementType.CharAlt);

            // For a negated class, any code point outside every listed range is
            // acceptable, so the prefix survives unless the ranges cover it all.
            return !isPositive;
        }

        private static bool Overlaps(uint lo1, uint hi1, uint lo2, uint hi2) =>
            lo1 <= hi2 && lo2 <= hi1;

        /// <summary>Advance the state by one complete code point.</summary>
        public GrammarState AcceptCodePoint(GrammarState state, uint codePoint)
        {
            var next = new List<int[]>();
            foreach (int[] stack in state.Stacks)
            {
                if (stack.Length == 0) continue;
                int pos = stack[stack.Length - 1];
                var (matched, after) = MatchChar(pos, codePoint);
                if (!matched) continue;

                bool keep = !_grammar.IsEndOfSequence(after);
                var reduced = new int[stack.Length - (keep ? 0 : 1)];
                Array.Copy(stack, reduced, stack.Length - 1);
                if (keep) reduced[stack.Length - 1] = after;
                AdvanceStack(reduced, next);
            }
            return Canonicalize(next);
        }

        /// <summary>
        /// True if the state could still be viable with a code point that is
        /// only partially decoded.
        /// </summary>
        public bool AcceptsPartial(GrammarState state, PartialUtf8 partial)
        {
            if (partial.IsInvalid) return false;
            if (partial.Remaining == 0) return !state.IsDead;
            foreach (int[] stack in state.Stacks)
            {
                if (stack.Length == 0) continue;
                if (MatchPartialChar(stack[stack.Length - 1], partial)) return true;
            }
            return false;
        }

        /// <summary>
        /// Feed raw UTF-8 bytes, carrying any incomplete trailing code point out
        /// through <paramref name="partial"/>. Returns the dead state as soon as
        /// the input stops being derivable.
        /// </summary>
        public GrammarState AcceptBytes(GrammarState state, ReadOnlySpan<byte> bytes, ref PartialUtf8 partial)
        {
            foreach (byte b in bytes)
            {
                if (!TryFeedByte(ref partial, b, out uint codePoint, out bool complete))
                    return new GrammarState(Array.Empty<int[]>());
                if (!complete) continue;
                state = AcceptCodePoint(state, codePoint);
                if (state.IsDead) return state;
            }
            return state;
        }

        /// <summary>
        /// Push one byte into a UTF-8 decoder. Returns false on a malformed
        /// sequence; on success <paramref name="complete"/> says whether a code
        /// point finished.
        /// </summary>
        public static bool TryFeedByte(ref PartialUtf8 partial, byte b, out uint codePoint, out bool complete)
        {
            codePoint = 0;
            complete = false;
            if (partial.IsInvalid) return false;

            if (partial.Remaining == 0)
            {
                if (b < 0x80) { codePoint = b; complete = true; return true; }
                if ((b & 0xE0) == 0xC0) { partial = new PartialUtf8((uint)(b & 0x1F), 1); return true; }
                if ((b & 0xF0) == 0xE0) { partial = new PartialUtf8((uint)(b & 0x0F), 2); return true; }
                if ((b & 0xF8) == 0xF0) { partial = new PartialUtf8((uint)(b & 0x07), 3); return true; }
                return false;   // continuation byte or 0xF8..0xFF with nothing open
            }

            if ((b & 0xC0) != 0x80) return false;   // expected a continuation byte
            uint value = (partial.Value << 6) | (uint)(b & 0x3F);
            int remaining = partial.Remaining - 1;
            if (remaining == 0)
            {
                partial = PartialUtf8.Empty;
                codePoint = value;
                complete = true;
                return true;
            }
            partial = new PartialUtf8(value, remaining);
            return true;
        }
    }

    internal sealed class IntArrayComparer : IEqualityComparer<int[]>
    {
        public static readonly IntArrayComparer Instance = new();

        public bool Equals(int[]? a, int[]? b)
        {
            if (ReferenceEquals(a, b)) return true;
            if (a is null || b is null || a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) return false;
            return true;
        }

        public int GetHashCode(int[] a)
        {
            unchecked
            {
                int h = 17;
                foreach (int v in a) h = h * 31 + v;
                return h;
            }
        }
    }
}
