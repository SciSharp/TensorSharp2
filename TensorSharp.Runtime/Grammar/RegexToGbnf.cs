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
using System.Text;

namespace TensorSharp.Runtime.Grammar
{
    /// <summary>
    /// Translates the regular-expression subset that appears in JSON Schema
    /// <c>pattern</c> keywords into a GBNF fragment.
    /// </summary>
    /// <remarks>
    /// Supported: literals, <c>.</c>, character classes (including negation,
    /// ranges and the <c>\d \w \s</c> shorthands), groups, alternation, and the
    /// <c>* + ? {m} {m,} {m,n}</c> quantifiers. Anchors <c>^</c>/<c>$</c> are
    /// accepted and dropped, since a schema pattern is implicitly anchored to
    /// the whole string here.
    /// <para>
    /// Anything else — backreferences, lookaround, non-greedy quantifiers, named
    /// groups, unicode property escapes — makes <see cref="TryConvert"/> return
    /// false, and the caller falls back to an unconstrained string. Producing an
    /// approximate grammar would risk rejecting valid output, which is the one
    /// failure mode constrained decoding must not have.
    /// </para>
    /// </remarks>
    internal static class RegexToGbnf
    {
        public static bool TryConvert(string pattern, out string gbnf)
        {
            gbnf = string.Empty;
            if (string.IsNullOrEmpty(pattern)) return false;
            try
            {
                var p = new Parser(pattern);
                string body = p.ParseAlternation();
                if (!p.AtEnd) return false;
                gbnf = body;
                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        private sealed class Parser
        {
            private readonly string _s;
            private int _i;

            public Parser(string s)
            {
                _s = s;
                _i = 0;
                // A JSON Schema pattern is a search by definition, but for
                // generation we treat it as matching the entire string; explicit
                // anchors are therefore redundant rather than meaningful.
                if (_i < _s.Length && _s[_i] == '^') _i++;
            }

            public bool AtEnd
            {
                get
                {
                    if (_i < _s.Length && _s[_i] == '$' && _i + 1 == _s.Length) _i++;
                    return _i >= _s.Length;
                }
            }

            private char Cur => _i < _s.Length ? _s[_i] : '\0';

            public string ParseAlternation()
            {
                var sb = new StringBuilder();
                sb.Append(ParseSequence());
                while (Cur == '|')
                {
                    _i++;
                    sb.Append(" | ").Append(ParseSequence());
                }
                return sb.ToString();
            }

            private string ParseSequence()
            {
                var sb = new StringBuilder();
                bool first = true;
                while (_i < _s.Length && Cur != '|' && Cur != ')')
                {
                    if (Cur == '$' && _i + 1 == _s.Length) break;
                    string atom = ParseAtom();
                    atom = ParseQuantifier(atom);
                    if (!first) sb.Append(' ');
                    sb.Append(atom);
                    first = false;
                }
                return sb.Length == 0 ? "\"\"" : sb.ToString();
            }

            private string ParseAtom()
            {
                char c = Cur;
                if (c == '(')
                {
                    _i++;
                    // Only plain and non-capturing groups; anything else is a
                    // construct we do not model.
                    if (Cur == '?')
                    {
                        if (_i + 1 < _s.Length && _s[_i + 1] == ':') _i += 2;
                        else throw new NotSupportedException("lookaround or named group");
                    }
                    string inner = ParseAlternation();
                    if (Cur != ')') throw new NotSupportedException("unbalanced group");
                    _i++;
                    return "(" + inner + ")";
                }
                if (c == '[') return ParseCharClass();
                if (c == '.') { _i++; return "[^\"\\\\]"; }
                if (c == '\\') return ParseEscape(inClass: false);
                if (c == '*' || c == '+' || c == '?' || c == '{')
                    throw new NotSupportedException("quantifier with no atom");
                _i++;
                return Literal(c);
            }

            private string ParseQuantifier(string atom)
            {
                if (_i >= _s.Length) return atom;
                char c = Cur;
                string suffix;
                if (c == '*') { _i++; suffix = "*"; }
                else if (c == '+') { _i++; suffix = "+"; }
                else if (c == '?') { _i++; suffix = "?"; }
                else if (c == '{')
                {
                    int save = _i;
                    _i++;
                    int start = _i;
                    while (_i < _s.Length && char.IsDigit(_s[_i])) _i++;
                    if (_i == start) { _i = save; return atom; }
                    string min = _s.Substring(start, _i - start);
                    string max = min;
                    if (Cur == ',')
                    {
                        _i++;
                        int ms = _i;
                        while (_i < _s.Length && char.IsDigit(_s[_i])) _i++;
                        max = _i > ms ? _s.Substring(ms, _i - ms) : string.Empty;
                    }
                    if (Cur != '}') { _i = save; return atom; }
                    _i++;
                    suffix = max.Length == 0 ? "{" + min + ",}" : "{" + min + "," + max + "}";
                }
                else
                {
                    return atom;
                }

                if (Cur == '?')
                    throw new NotSupportedException("non-greedy quantifier");

                // A quantifier binds to a single element in GBNF too, so a
                // multi-element atom must be parenthesised.
                bool needsGroup = atom.Length > 1 &&
                                  !(atom[0] == '(' && atom[atom.Length - 1] == ')') &&
                                  !(atom[0] == '[' && atom[atom.Length - 1] == ']') &&
                                  !(atom[0] == '"' && atom[atom.Length - 1] == '"' && atom.Length <= 4);
                return (needsGroup ? "(" + atom + ")" : atom) + suffix;
            }

            private string ParseCharClass()
            {
                _i++;   // '['
                var sb = new StringBuilder("[");
                if (Cur == '^') { sb.Append('^'); _i++; }
                bool any = false;
                while (_i < _s.Length && Cur != ']')
                {
                    if (Cur == '\\')
                    {
                        sb.Append(ParseEscape(inClass: true));
                        any = true;
                        continue;
                    }
                    char c = _s[_i++];
                    sb.Append(ClassChar(c));
                    any = true;
                    if (Cur == '-' && _i + 1 < _s.Length && _s[_i + 1] != ']')
                    {
                        _i++;
                        char hi = Cur == '\\' ? throw new NotSupportedException("escaped range bound") : _s[_i++];
                        sb.Append('-').Append(ClassChar(hi));
                    }
                }
                if (Cur != ']') throw new NotSupportedException("unterminated character class");
                _i++;
                if (!any) throw new NotSupportedException("empty character class");
                sb.Append(']');
                return sb.ToString();
            }

            private string ParseEscape(bool inClass)
            {
                _i++;   // backslash
                if (_i >= _s.Length) throw new NotSupportedException("trailing backslash");
                char c = _s[_i++];
                switch (c)
                {
                    case 'd': return inClass ? "0-9" : "[0-9]";
                    case 'D': return inClass ? throw new NotSupportedException("\\D in class") : "[^0-9]";
                    case 'w': return inClass ? "a-zA-Z0-9_" : "[a-zA-Z0-9_]";
                    case 'W': return inClass ? throw new NotSupportedException("\\W in class") : "[^a-zA-Z0-9_]";
                    case 's': return inClass ? " \\t\\n\\r" : "[ \\t\\n\\r]";
                    case 'S': return inClass ? throw new NotSupportedException("\\S in class") : "[^ \\t\\n\\r]";
                    case 'n': return inClass ? "\\n" : "\"\\n\"";
                    case 'r': return inClass ? "\\r" : "\"\\r\"";
                    case 't': return inClass ? "\\t" : "\"\\t\"";
                    case 'u':
                    {
                        if (_i + 4 > _s.Length) throw new NotSupportedException("truncated \\u escape");
                        string hex = _s.Substring(_i, 4);
                        _i += 4;
                        if (!int.TryParse(hex, System.Globalization.NumberStyles.HexNumber,
                                          System.Globalization.CultureInfo.InvariantCulture, out int cp))
                            throw new NotSupportedException("bad \\u escape");
                        return inClass ? "\\u" + hex : Literal((char)cp);
                    }
                    default:
                        if (char.IsLetterOrDigit(c))
                            throw new NotSupportedException($"unsupported escape \\{c}");
                        return inClass ? ClassChar(c) : Literal(c);
                }
            }

            /// <summary>Escape a character for use inside a GBNF character class.</summary>
            private static string ClassChar(char c) => c switch
            {
                ']' => "\\]",
                '[' => "\\[",
                '\\' => "\\\\",
                '-' => "\\-",
                '^' => "\\^",
                '"' => "\\\"",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                _ => c < 0x20 || c > 0x7e ? "\\u" + ((int)c).ToString("X4") : c.ToString(),
            };

            /// <summary>Escape a character as a standalone GBNF string literal.</summary>
            private static string Literal(char c) => c switch
            {
                '"' => "\"\\\"\"",
                '\\' => "\"\\\\\"",
                '\n' => "\"\\n\"",
                '\r' => "\"\\r\"",
                '\t' => "\"\\t\"",
                _ => c < 0x20 || c > 0x7e
                    ? "[\\u" + ((int)c).ToString("X4") + "]"
                    : "\"" + c + "\"",
            };
        }
    }
}
