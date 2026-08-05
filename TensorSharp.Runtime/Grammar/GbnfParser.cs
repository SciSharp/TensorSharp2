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
using System.Globalization;

namespace TensorSharp.Runtime.Grammar
{
    /// <summary>Thrown when GBNF source cannot be parsed.</summary>
    public sealed class GrammarParseException : Exception
    {
        public GrammarParseException(string message) : base(message) { }
    }

    /// <summary>
    /// Parser for GBNF, the grammar syntax llama.cpp uses. Supported: rule
    /// definitions (<c>name ::= …</c>), alternation (<c>|</c>), grouping
    /// (<c>(…)</c>), literals (<c>"…"</c>), character classes
    /// (<c>[a-z]</c>, <c>[^…]</c>), any-char (<c>.</c>), repetition
    /// (<c>*</c>, <c>+</c>, <c>?</c>, <c>{m}</c>, <c>{m,}</c>, <c>{m,n}</c>),
    /// and <c>#</c> line comments.
    /// </summary>
    /// <remarks>
    /// Repetition is desugared into generated rules rather than represented
    /// directly, because the matcher's element model has no repeat operator:
    /// <c>X*</c> becomes <c>S ::= X S |</c>, <c>X+</c> becomes <c>S ::= X S | X</c>,
    /// and <c>X{m,n}</c> is unrolled. Right recursion (not left) is required —
    /// the matcher expands non-terminals eagerly, so a left-recursive rule would
    /// not terminate. <see cref="DetectLeftRecursion"/> rejects those up front
    /// with a message naming the rule instead of hanging.
    /// </remarks>
    public static class GbnfParser
    {
        public static Grammar Parse(string source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));

            var state = new ParserState(source);
            state.ParseAll();

            if (!state.SymbolIds.TryGetValue("root", out int rootId))
                throw new GrammarParseException("grammar must define a 'root' rule");

            Grammar grammar = state.Builder.Build(rootId);
            DetectLeftRecursion(grammar);
            return grammar;
        }

        /// <summary>
        /// Reject grammars whose rules can reach themselves without consuming a
        /// character. The matcher expands non-terminals eagerly, so such a rule
        /// is an infinite loop rather than a parse error at match time.
        /// </summary>
        private static void DetectLeftRecursion(Grammar grammar)
        {
            int n = grammar.RuleCount;
            var visited = new bool[n];
            var inProgress = new bool[n];
            var mayBeEmpty = new bool[n];
            for (int i = 0; i < n; i++)
            {
                if (!visited[i] && Recurse(grammar, i, visited, inProgress, mayBeEmpty))
                    throw new GrammarParseException(
                        $"grammar is left-recursive through rule {i}; rewrite it with right recursion");
            }
        }

        private static bool Recurse(Grammar g, int ruleId, bool[] visited, bool[] inProgress, bool[] mayBeEmpty)
        {
            if (inProgress[ruleId]) return true;
            if (visited[ruleId]) return false;
            visited[ruleId] = true;
            inProgress[ruleId] = true;

            bool ruleMayBeEmpty = false;
            int pos = g.RuleStart(ruleId);
            while (true)
            {
                // Walk one alternate; it may be empty only if every element in it may be.
                bool altMayBeEmpty = true;
                while (!g.IsEndOfSequence(pos))
                {
                    GrammarElement e = g[pos];
                    if (e.Type == GrammarElementType.RuleRef)
                    {
                        if (Recurse(g, (int)e.Value, visited, inProgress, mayBeEmpty))
                        {
                            inProgress[ruleId] = false;
                            return true;
                        }
                        if (!mayBeEmpty[(int)e.Value])
                        {
                            altMayBeEmpty = false;
                            break;
                        }
                        pos++;
                    }
                    else
                    {
                        // Any char element consumes input, so this alternate is
                        // productive and cannot loop back without progress.
                        altMayBeEmpty = false;
                        break;
                    }
                }
                if (altMayBeEmpty) ruleMayBeEmpty = true;

                while (!g.IsEndOfSequence(pos)) pos++;
                if (g[pos].Type == GrammarElementType.Alt) { pos++; continue; }
                break;
            }

            inProgress[ruleId] = false;
            mayBeEmpty[ruleId] = ruleMayBeEmpty;
            return false;
        }

        private sealed class ParserState
        {
            private readonly string _src;
            private int _i;

            public readonly Grammar.Builder Builder = new();
            public readonly Dictionary<string, int> SymbolIds = new(StringComparer.Ordinal);
            private int _generatedCounter;

            public ParserState(string src) { _src = src; _i = 0; }

            private char Cur => _i < _src.Length ? _src[_i] : '\0';
            private char At(int off) => _i + off < _src.Length ? _src[_i + off] : '\0';
            private bool Eof => _i >= _src.Length;

            public void ParseAll()
            {
                SkipTrivia();
                while (!Eof)
                {
                    ParseRule();
                    SkipTrivia();
                }
            }

            private void SkipTrivia()
            {
                while (!Eof)
                {
                    char c = Cur;
                    if (c == '#')
                    {
                        while (!Eof && Cur != '\n') _i++;
                    }
                    else if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
                    {
                        _i++;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            /// <summary>Whitespace/comments that may not cross a newline (inside a rule body).</summary>
            private void SkipInlineTrivia()
            {
                while (!Eof)
                {
                    char c = Cur;
                    if (c == '#')
                    {
                        while (!Eof && Cur != '\n') _i++;
                    }
                    else if (c == ' ' || c == '\t' || c == '\r')
                    {
                        _i++;
                    }
                    else if (c == '\n')
                    {
                        // A newline continues the rule only when the next
                        // non-trivial line starts with an alternation bar.
                        int save = _i;
                        _i++;
                        SkipTrivia();
                        if (Cur == '|') return;
                        _i = save;
                        return;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            private int GetSymbolId(string name)
            {
                if (SymbolIds.TryGetValue(name, out int id)) return id;
                id = Builder.ReserveRule();
                SymbolIds[name] = id;
                return id;
            }

            private int GenerateSymbolId(string baseName)
            {
                int id = Builder.ReserveRule();
                SymbolIds[$"{baseName}_{_generatedCounter++}"] = id;
                return id;
            }

            private void ParseRule()
            {
                string name = ParseName();
                SkipInlineTrivia();
                if (!(Cur == ':' && At(1) == ':' && At(2) == '='))
                    throw new GrammarParseException($"expecting '::=' at offset {_i}");
                _i += 3;
                SkipInlineTrivia();

                int ruleId = GetSymbolId(name);
                var elements = ParseAlternates(name, ruleId);
                elements.Add(new GrammarElement(GrammarElementType.End, 0));
                Builder.SetRule(ruleId, elements);

                // A rule body ends at the newline; trailing inline trivia only.
                while (!Eof && (Cur == ' ' || Cur == '\t' || Cur == '\r')) _i++;
                if (!Eof && Cur == '#') { while (!Eof && Cur != '\n') _i++; }
                if (!Eof && Cur == '\n') _i++;
            }

            private string ParseName()
            {
                int start = _i;
                while (!Eof && (char.IsLetterOrDigit(Cur) || Cur == '-' || Cur == '_')) _i++;
                if (_i == start)
                    throw new GrammarParseException($"expecting rule name at offset {_i}");
                return _src.Substring(start, _i - start);
            }

            private List<GrammarElement> ParseAlternates(string ruleName, int ruleId)
            {
                var result = ParseSequence(ruleName, ruleId, isNested: false);
                SkipInlineTrivia();
                while (Cur == '|')
                {
                    _i++;
                    SkipInlineTrivia();
                    result.Add(new GrammarElement(GrammarElementType.Alt, 0));
                    result.AddRange(ParseSequence(ruleName, ruleId, isNested: false));
                    SkipInlineTrivia();
                }
                return result;
            }

            private List<GrammarElement> ParseSequence(string ruleName, int ruleId, bool isNested)
            {
                var seq = new List<GrammarElement>();

                while (true)
                {
                    SkipInlineTrivia();
                    if (Eof) break;
                    char c = Cur;
                    if (c == '|' || c == ')' || c == '\n') break;

                    // Index in `seq` where the element just parsed begins, so a
                    // following repetition operator knows what it repeats.
                    int lastStart = seq.Count;

                    if (c == '"')
                    {
                        _i++;
                        while (Cur != '"')
                        {
                            if (Eof) throw new GrammarParseException("unterminated string literal");
                            uint cp = ParseChar();
                            seq.Add(new GrammarElement(GrammarElementType.Char, cp));
                        }
                        _i++;
                    }
                    else if (c == '[')
                    {
                        _i++;
                        var type = GrammarElementType.Char;
                        if (Cur == '^') { _i++; type = GrammarElementType.CharNot; }
                        bool first = true;
                        while (Cur != ']')
                        {
                            if (Eof) throw new GrammarParseException("unterminated character class");
                            uint cp = ParseChar();
                            seq.Add(new GrammarElement(
                                first ? type : GrammarElementType.CharAlt, cp));
                            first = false;
                            if (Cur == '-' && At(1) != ']')
                            {
                                _i++;
                                uint hi = ParseChar();
                                seq.Add(new GrammarElement(GrammarElementType.CharRngUpper, hi));
                            }
                        }
                        _i++;
                    }
                    else if (c == '.')
                    {
                        _i++;
                        seq.Add(new GrammarElement(GrammarElementType.CharAny, 0));
                    }
                    else if (c == '(')
                    {
                        _i++;
                        SkipTrivia();
                        int sub = GenerateSymbolId(ruleName);
                        var inner = ParseAlternates(ruleName, sub);
                        inner.Add(new GrammarElement(GrammarElementType.End, 0));
                        Builder.SetRule(sub, inner);
                        SkipTrivia();
                        if (Cur != ')')
                            throw new GrammarParseException($"expecting ')' at offset {_i}");
                        _i++;
                        seq.Add(new GrammarElement(GrammarElementType.RuleRef, (uint)sub));
                    }
                    else if (char.IsLetter(c) || c == '_')
                    {
                        string refName = ParseName();
                        seq.Add(new GrammarElement(GrammarElementType.RuleRef, (uint)GetSymbolId(refName)));
                    }
                    else
                    {
                        throw new GrammarParseException($"unexpected character '{c}' at offset {_i}");
                    }

                    // Repetition operators bind to the element just parsed.
                    while (true)
                    {
                        if (Cur == '*') { _i++; ApplyRepetition(seq, lastStart, ruleName, 0, -1); }
                        else if (Cur == '+') { _i++; ApplyRepetition(seq, lastStart, ruleName, 1, -1); }
                        else if (Cur == '?') { _i++; ApplyRepetition(seq, lastStart, ruleName, 0, 1); }
                        else if (Cur == '{')
                        {
                            _i++;
                            int min = ParseInt();
                            int max;
                            if (Cur == ',')
                            {
                                _i++;
                                max = (Cur == '}') ? -1 : ParseInt();
                            }
                            else
                            {
                                max = min;
                            }
                            if (Cur != '}')
                                throw new GrammarParseException($"expecting '}}' at offset {_i}");
                            _i++;
                            ApplyRepetition(seq, lastStart, ruleName, min, max);
                        }
                        else
                        {
                            break;
                        }
                    }
                }

                return seq;
            }

            /// <summary>
            /// Replace <c>seq[from..]</c> (the element(s) a repetition operator
            /// applies to) with a reference to a generated rule implementing the
            /// repetition. Uses right recursion so the matcher, which expands
            /// non-terminals eagerly, always makes progress.
            /// </summary>
            private void ApplyRepetition(List<GrammarElement> seq, int from, string ruleName, int min, int max)
            {
                if (min < 0 || (max >= 0 && max < min))
                    throw new GrammarParseException($"invalid repetition bounds {{{min},{max}}}");

                var body = seq.GetRange(from, seq.Count - from);
                seq.RemoveRange(from, seq.Count - from);
                if (body.Count == 0)
                    throw new GrammarParseException("repetition operator with nothing to repeat");

                // Wrap a multi-element body in its own rule so it can be
                // referenced as a unit.
                List<GrammarElement> BodyRef()
                {
                    if (body.Count == 1 && body[0].Type == GrammarElementType.RuleRef)
                        return new List<GrammarElement>(body);
                    int id = GenerateSymbolId(ruleName);
                    var r = new List<GrammarElement>(body) { new GrammarElement(GrammarElementType.End, 0) };
                    Builder.SetRule(id, r);
                    return new List<GrammarElement> { new GrammarElement(GrammarElementType.RuleRef, (uint)id) };
                }

                var unit = BodyRef();

                if (max < 0)
                {
                    // S ::= unit S |          (for *)
                    // S ::= unit S | unit     (for +, expressed as min copies then *)
                    int star = GenerateSymbolId(ruleName);
                    var starRule = new List<GrammarElement>();
                    starRule.AddRange(unit);
                    starRule.Add(new GrammarElement(GrammarElementType.RuleRef, (uint)star));
                    starRule.Add(new GrammarElement(GrammarElementType.Alt, 0));
                    starRule.Add(new GrammarElement(GrammarElementType.End, 0));
                    Builder.SetRule(star, starRule);

                    for (int k = 0; k < min; k++) seq.AddRange(unit);
                    seq.Add(new GrammarElement(GrammarElementType.RuleRef, (uint)star));
                    return;
                }

                // Bounded: min mandatory copies, then (max - min) optional ones
                // nested so that "abc{2,4}" cannot stop mid-way through a copy.
                for (int k = 0; k < min; k++) seq.AddRange(unit);

                int optional = max - min;
                if (optional == 0) return;

                int prev = -1;
                for (int k = 0; k < optional; k++)
                {
                    int id = GenerateSymbolId(ruleName);
                    var r = new List<GrammarElement>();
                    r.AddRange(unit);
                    if (prev >= 0)
                        r.Add(new GrammarElement(GrammarElementType.RuleRef, (uint)prev));
                    r.Add(new GrammarElement(GrammarElementType.Alt, 0));
                    r.Add(new GrammarElement(GrammarElementType.End, 0));
                    Builder.SetRule(id, r);
                    prev = id;
                }
                seq.Add(new GrammarElement(GrammarElementType.RuleRef, (uint)prev));
            }

            private int ParseInt()
            {
                int start = _i;
                while (!Eof && char.IsDigit(Cur)) _i++;
                if (_i == start)
                    throw new GrammarParseException($"expecting integer at offset {_i}");
                return int.Parse(_src.AsSpan(start, _i - start), CultureInfo.InvariantCulture);
            }

            /// <summary>Parse one (possibly escaped) character, returning its code point.</summary>
            private uint ParseChar()
            {
                if (Cur == '\\')
                {
                    char e = At(1);
                    switch (e)
                    {
                        case 'x': _i += 2; return ParseHex(2);
                        case 'u': _i += 2; return ParseHex(4);
                        case 'U': _i += 2; return ParseHex(8);
                        case 't': _i += 2; return '\t';
                        case 'r': _i += 2; return '\r';
                        case 'n': _i += 2; return '\n';
                        case '\\':
                        case '"':
                        case '[':
                        case ']':
                            _i += 2;
                            return e;
                        default:
                            throw new GrammarParseException($"unknown escape '\\{e}' at offset {_i}");
                    }
                }
                if (Eof) throw new GrammarParseException("unexpected end of input");

                // Decode one code point from the C# UTF-16 source, so grammars
                // may contain non-BMP literals.
                if (char.IsHighSurrogate(Cur) && char.IsLowSurrogate(At(1)))
                {
                    uint cp = (uint)char.ConvertToUtf32(Cur, At(1));
                    _i += 2;
                    return cp;
                }
                return _src[_i++];
            }

            private uint ParseHex(int digits)
            {
                uint v = 0;
                for (int k = 0; k < digits; k++)
                {
                    if (Eof) throw new GrammarParseException("truncated hex escape");
                    uint d = (uint)Convert.ToInt32(Cur.ToString(), 16);
                    char c = Cur;
                    if (!Uri.IsHexDigit(c))
                        throw new GrammarParseException($"invalid hex digit '{c}' at offset {_i}");
                    v = v * 16 + d;
                    _i++;
                }
                return v;
            }
        }
    }
}
