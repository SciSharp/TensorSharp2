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
    /// Element kinds of a compiled grammar, matching llama.cpp's GBNF element
    /// model (<c>llama_gretype</c>) so grammars are portable between the two
    /// engines byte for byte.
    /// </summary>
    public enum GrammarElementType : byte
    {
        /// <summary>End of a rule's alternate.</summary>
        End = 0,

        /// <summary>Separator between a rule's alternates (<c>|</c>).</summary>
        Alt = 1,

        /// <summary>Non-terminal: expand rule <see cref="GrammarElement.Value"/>.</summary>
        RuleRef = 2,

        /// <summary>Terminal: match this Unicode code point.</summary>
        Char = 3,

        /// <summary>Terminal: match anything *but* the following char set (<c>[^…]</c>).</summary>
        CharNot = 4,

        /// <summary>Upgrades the preceding <see cref="Char"/>/<see cref="CharAlt"/> to an inclusive range (<c>[a-z]</c>).</summary>
        CharRngUpper = 5,

        /// <summary>Adds another alternative to the preceding char set (<c>[ab]</c>).</summary>
        CharAlt = 6,

        /// <summary>Terminal: match any single code point (<c>.</c>).</summary>
        CharAny = 7,
    }

    /// <summary>One element of a compiled grammar rule.</summary>
    public readonly struct GrammarElement : IEquatable<GrammarElement>
    {
        public readonly GrammarElementType Type;

        /// <summary>Code point, rule id, or range bound depending on <see cref="Type"/>.</summary>
        public readonly uint Value;

        public GrammarElement(GrammarElementType type, uint value)
        {
            Type = type;
            Value = value;
        }

        public bool Equals(GrammarElement other) => Type == other.Type && Value == other.Value;
        public override bool Equals(object? obj) => obj is GrammarElement e && Equals(e);
        public override int GetHashCode() => ((int)Type * 397) ^ (int)Value;
    }

    /// <summary>
    /// A parsed, immutable grammar: the rule set plus the root rule to start
    /// matching from. Compile once (it is independent of any request) and share;
    /// per-request mutable state lives in <see cref="GrammarMatcher"/>.
    /// </summary>
    /// <remarks>
    /// Rules are stored <b>flattened</b> into a single element array with an
    /// index of where each rule starts, rather than as a jagged array. A parse
    /// position is then a single <c>int</c> instead of a (rule, offset) pair or
    /// a pointer, which is what makes a matcher stack a plain <c>int[]</c> —
    /// cheap to copy, compare and hash, and therefore usable as a token-mask
    /// cache key (see <see cref="GrammarMatcher"/>). Walking off the end of a
    /// rule is impossible because every alternate is terminated by an
    /// <see cref="GrammarElementType.End"/> element that the matcher never steps
    /// past.
    /// </remarks>
    public sealed class Grammar
    {
        private readonly GrammarElement[] _elements;
        private readonly int[] _ruleStart;

        internal Grammar(GrammarElement[] elements, int[] ruleStart, int rootRuleId)
        {
            _elements = elements ?? throw new ArgumentNullException(nameof(elements));
            _ruleStart = ruleStart ?? throw new ArgumentNullException(nameof(ruleStart));
            if (rootRuleId < 0 || rootRuleId >= ruleStart.Length)
                throw new ArgumentOutOfRangeException(nameof(rootRuleId));
            RootRuleId = rootRuleId;
        }

        /// <summary>Rule the matcher starts in (GBNF's <c>root</c>).</summary>
        public int RootRuleId { get; }

        /// <summary>Number of rules.</summary>
        public int RuleCount => _ruleStart.Length;

        /// <summary>Total element count across all rules.</summary>
        public int ElementCount => _elements.Length;

        internal GrammarElement this[int position] => _elements[position];

        /// <summary>First element position of <paramref name="ruleId"/>.</summary>
        internal int RuleStart(int ruleId) => _ruleStart[ruleId];

        /// <summary>
        /// True when <paramref name="position"/> terminates an alternate, i.e. the
        /// matcher must not advance past it within this rule.
        /// </summary>
        internal bool IsEndOfSequence(int position)
        {
            GrammarElementType t = _elements[position].Type;
            return t == GrammarElementType.End || t == GrammarElementType.Alt;
        }

        /// <summary>
        /// Parse GBNF source into a grammar. The source must define a
        /// <c>root</c> rule.
        /// </summary>
        public static Grammar Parse(string gbnf) => GbnfParser.Parse(gbnf);

        /// <summary>
        /// Compile a JSON Schema (or <c>null</c>/empty for "any JSON value") into
        /// a grammar that admits exactly the documents the schema accepts.
        /// </summary>
        public static Grammar FromJsonSchema(string? schemaJson) =>
            Parse(JsonSchemaGrammarCompiler.Compile(schemaJson));

        /// <summary>
        /// Grammar accepting any single well-formed JSON <b>object</b> — what
        /// OpenAI's <c>response_format: {"type": "json_object"}</c> promises.
        /// </summary>
        public static Grammar JsonObject() => Parse(JsonSchemaGrammarCompiler.JsonObjectGrammar);

        /// <summary>Render the grammar back to GBNF (diagnostics and tests).</summary>
        public string ToGbnf()
        {
            var sb = new System.Text.StringBuilder();
            for (int rule = 0; rule < _ruleStart.Length; rule++)
            {
                sb.Append(rule == RootRuleId ? "root" : $"rule{rule}").Append(" ::= ");
                int pos = _ruleStart[rule];
                bool first = true;
                while (true)
                {
                    GrammarElement e = _elements[pos];
                    if (e.Type == GrammarElementType.End)
                        break;
                    if (e.Type == GrammarElementType.Alt)
                    {
                        sb.Append(" | ");
                        first = true;
                        pos++;
                        continue;
                    }
                    if (!first) sb.Append(' ');
                    first = false;
                    switch (e.Type)
                    {
                        case GrammarElementType.RuleRef:
                            sb.Append(e.Value == (uint)RootRuleId ? "root" : $"rule{e.Value}");
                            break;
                        case GrammarElementType.CharAny:
                            sb.Append('.');
                            break;
                        case GrammarElementType.Char:
                        case GrammarElementType.CharNot:
                            sb.Append('[');
                            if (e.Type == GrammarElementType.CharNot) sb.Append('^');
                            AppendCharSet(sb, ref pos);
                            sb.Append(']');
                            continue;
                    }
                    pos++;
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }

        private void AppendCharSet(System.Text.StringBuilder sb, ref int pos)
        {
            AppendChar(sb, _elements[pos].Value);
            pos++;
            while (true)
            {
                GrammarElement e = _elements[pos];
                if (e.Type == GrammarElementType.CharRngUpper)
                {
                    sb.Append('-');
                    AppendChar(sb, e.Value);
                    pos++;
                }
                else if (e.Type == GrammarElementType.CharAlt)
                {
                    AppendChar(sb, e.Value);
                    pos++;
                }
                else
                {
                    break;
                }
            }
        }

        private static void AppendChar(System.Text.StringBuilder sb, uint c)
        {
            switch (c)
            {
                case '\n': sb.Append("\\n"); return;
                case '\r': sb.Append("\\r"); return;
                case '\t': sb.Append("\\t"); return;
                case '\\': sb.Append("\\\\"); return;
                case '[': sb.Append("\\["); return;
                case ']': sb.Append("\\]"); return;
                case '"': sb.Append("\\\""); return;
            }
            if (c >= 0x20 && c <= 0x7e)
                sb.Append((char)c);
            else
                sb.Append("\\u").Append(c.ToString("X4"));
        }

        /// <summary>Builder used by the parser and the schema compiler.</summary>
        internal sealed class Builder
        {
            private readonly List<List<GrammarElement>> _rules = new();

            public int AddRule(List<GrammarElement> elements)
            {
                _rules.Add(elements);
                return _rules.Count - 1;
            }

            public int ReserveRule()
            {
                _rules.Add(new List<GrammarElement>());
                return _rules.Count - 1;
            }

            public void SetRule(int id, List<GrammarElement> elements) => _rules[id] = elements;

            public int RuleCount => _rules.Count;

            public Grammar Build(int rootRuleId)
            {
                var flat = new List<GrammarElement>();
                var starts = new int[_rules.Count];
                for (int i = 0; i < _rules.Count; i++)
                {
                    starts[i] = flat.Count;
                    flat.AddRange(_rules[i]);
                    // Every rule is terminated so a position can never walk out
                    // of its rule; the parser also appends End per alternate set.
                    if (flat.Count == starts[i] ||
                        flat[flat.Count - 1].Type != GrammarElementType.End)
                    {
                        flat.Add(new GrammarElement(GrammarElementType.End, 0));
                    }
                }
                return new Grammar(flat.ToArray(), starts, rootRuleId);
            }
        }
    }
}
