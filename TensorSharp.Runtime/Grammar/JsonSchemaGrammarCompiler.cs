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
using System.Linq;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime.Grammar
{
    /// <summary>
    /// Compiles a JSON Schema into GBNF, following the same strategy as
    /// llama.cpp's <c>json-schema-to-grammar</c> so the two engines constrain
    /// identically.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The governing rule is <b>never reject a valid document</b>. A grammar
    /// that is too tight makes generation impossible — the sampler runs out of
    /// legal tokens mid-string and the request fails — whereas one that is too
    /// loose merely lets through something a post-hoc validator can still catch.
    /// So every keyword this compiler does not model is <i>relaxed</i> to the
    /// unconstrained form of its type rather than approximated. What is enforced:
    /// </para>
    /// <list type="bullet">
    /// <item><c>type</c> (single or union), <c>enum</c>, <c>const</c></item>
    /// <item><c>properties</c>, <c>required</c>, <c>additionalProperties</c></item>
    /// <item><c>items</c>, <c>prefixItems</c>, <c>minItems</c>, <c>maxItems</c></item>
    /// <item><c>anyOf</c>, <c>oneOf</c>, shallow <c>allOf</c> merge</item>
    /// <item><c>$ref</c> into <c>$defs</c>/<c>definitions</c> (including recursive schemas)</item>
    /// <item><c>minLength</c>/<c>maxLength</c>, <c>pattern</c> (regex subset), and the
    ///       <c>date</c>/<c>time</c>/<c>date-time</c>/<c>uuid</c> formats</item>
    /// <item><c>minimum</c>/<c>maximum</c> on integers, and sign-only bounds on numbers</item>
    /// </list>
    /// <para>
    /// Not modelled (deliberately relaxed): <c>patternProperties</c>,
    /// <c>uniqueItems</c>, <c>multipleOf</c>, <c>not</c>, <c>if</c>/<c>then</c>/<c>else</c>,
    /// <c>dependentSchemas</c>, and exclusive numeric bounds on non-integers.
    /// </para>
    /// </remarks>
    public static class JsonSchemaGrammarCompiler
    {
        /// <summary>Grammar for a single well-formed JSON object of any shape.</summary>
        public const string JsonObjectGrammar = @"
root   ::= object
value  ::= object | array | string | number | boolean | null
object ::= ""{"" ws ( string "":"" ws value ("","" ws string "":"" ws value)* )? ""}"" ws
array  ::= ""["" ws ( value ("","" ws value)* )? ""]"" ws
string ::= ""\"""" char* ""\"""" ws
char   ::= [^""\\] | ""\\"" ([""\\bfnrt/] | ""u"" [0-9a-fA-F]{4})
number ::= ""-""? ([0-9] | [1-9] [0-9]{0,15}) (""."" [0-9]{1,16})? ([eE] [-+]? [0-9]{1,3})? ws
boolean::= (""true"" | ""false"") ws
null   ::= ""null"" ws
ws     ::= [ \t\n]{0,20}
";

        /// <summary>Grammar for any JSON value (object, array or scalar).</summary>
        public const string JsonValueGrammar = @"
root   ::= value
value  ::= object | array | string | number | boolean | null
object ::= ""{"" ws ( string "":"" ws value ("","" ws string "":"" ws value)* )? ""}"" ws
array  ::= ""["" ws ( value ("","" ws value)* )? ""]"" ws
string ::= ""\"""" char* ""\"""" ws
char   ::= [^""\\] | ""\\"" ([""\\bfnrt/] | ""u"" [0-9a-fA-F]{4})
number ::= ""-""? ([0-9] | [1-9] [0-9]{0,15}) (""."" [0-9]{1,16})? ([eE] [-+]? [0-9]{1,3})? ws
boolean::= (""true"" | ""false"") ws
null   ::= ""null"" ws
ws     ::= [ \t\n]{0,20}
";

        /// <summary>
        /// Compile <paramref name="schemaJson"/> to GBNF. A null or empty schema,
        /// or one with no constraints, yields the any-JSON-value grammar.
        /// </summary>
        public static string Compile(string? schemaJson)
        {
            if (string.IsNullOrWhiteSpace(schemaJson))
                return JsonValueGrammar;

            JsonDocument doc;
            try
            {
                doc = JsonDocument.Parse(schemaJson);
            }
            catch (JsonException ex)
            {
                throw new GrammarParseException($"response_format schema is not valid JSON: {ex.Message}");
            }

            using (doc)
            {
                var ctx = new CompileContext(doc.RootElement);
                string rootRef = ctx.Visit(doc.RootElement, "root");
                return ctx.Emit(rootRef);
            }
        }

        private sealed class CompileContext
        {
            private readonly JsonElement _documentRoot;
            private readonly Dictionary<string, string> _rules = new(StringComparer.Ordinal);
            private readonly Dictionary<string, string> _refCache = new(StringComparer.Ordinal);
            private readonly HashSet<string> _emitted = new(StringComparer.Ordinal);
            private int _counter;

            public CompileContext(JsonElement documentRoot) => _documentRoot = documentRoot;

            public string Emit(string rootRule)
            {
                var sb = new StringBuilder();
                // Emit `root ::= <rule>` as an alias rather than inlining the
                // rule's body under the name "root". Inlining drops the original
                // name, and a recursive schema ($defs/node whose child is a node)
                // still references it -- leaving a dangling non-terminal that
                // parses fine but can never be entered.
                if (rootRule != "root")
                    sb.Append("root ::= ").AppendLine(rootRule);
                foreach (var kv in _rules)
                    sb.Append(kv.Key).Append(" ::= ").AppendLine(kv.Value);
                return sb.ToString();
            }

            private string AddRule(string name, string body)
            {
                string unique = name;
                while (_rules.ContainsKey(unique) && _rules[unique] != body)
                    unique = $"{name}-{_counter++}";
                _rules[unique] = body;
                return unique;
            }

            private string Sanitize(string name)
            {
                var sb = new StringBuilder(name.Length);
                foreach (char c in name)
                    sb.Append(char.IsLetterOrDigit(c) || c == '-' ? c : '-');
                string s = sb.ToString().Trim('-');
                if (s.Length == 0) s = "x";
                if (char.IsDigit(s[0])) s = "r" + s;
                return s.ToLowerInvariant();
            }

            /// <summary>Add the shared JSON primitives on first use.</summary>
            private string Primitive(string name)
            {
                if (_emitted.Add(name))
                {
                    switch (name)
                    {
                        case "ws":
                            _rules["ws"] = "[ \\t\\n]{0,20}";
                            break;
                        case "string":
                            Primitive("ws");
                            Primitive("char");
                            _rules["string"] = "\"\\\"\" char* \"\\\"\" ws";
                            break;
                        case "char":
                            _rules["char"] = "[^\"\\\\] | \"\\\\\" ([\"\\\\bfnrt/] | \"u\" [0-9a-fA-F]{4})";
                            break;
                        case "integer":
                            Primitive("ws");
                            _rules["integer"] = "\"-\"? ([0-9] | [1-9] [0-9]{0,15}) ws";
                            break;
                        case "number":
                            Primitive("ws");
                            _rules["number"] =
                                "\"-\"? ([0-9] | [1-9] [0-9]{0,15}) (\".\" [0-9]{1,16})? ([eE] [-+]? [0-9]{1,3})? ws";
                            break;
                        case "boolean":
                            Primitive("ws");
                            _rules["boolean"] = "(\"true\" | \"false\") ws";
                            break;
                        case "null":
                            Primitive("ws");
                            _rules["null"] = "\"null\" ws";
                            break;
                        case "value":
                            Primitive("object-any");
                            Primitive("array-any");
                            Primitive("string");
                            Primitive("number");
                            Primitive("boolean");
                            Primitive("null");
                            _rules["value"] = "object-any | array-any | string | number | boolean | null";
                            break;
                        case "object-any":
                            Primitive("ws");
                            Primitive("string");
                            Primitive("value");
                            _rules["object-any"] =
                                "\"{\" ws ( string \":\" ws value (\",\" ws string \":\" ws value)* )? \"}\" ws";
                            break;
                        case "array-any":
                            Primitive("ws");
                            Primitive("value");
                            _rules["array-any"] = "\"[\" ws ( value (\",\" ws value)* )? \"]\" ws";
                            break;
                    }
                }
                return name;
            }

            /// <summary>Compile <paramref name="schema"/>, returning the rule name that matches it.</summary>
            public string Visit(JsonElement schema, string name)
            {
                if (schema.ValueKind == JsonValueKind.True || schema.ValueKind == JsonValueKind.Undefined)
                    return Primitive("value");
                if (schema.ValueKind == JsonValueKind.False)
                {
                    // Matches nothing. Represent as an unsatisfiable rule so the
                    // caller fails fast rather than silently accepting anything.
                    return AddRule(Sanitize(name), "\"\\u0000\"");
                }
                if (schema.ValueKind != JsonValueKind.Object)
                    return Primitive("value");

                if (schema.TryGetProperty("$ref", out JsonElement refEl) && refEl.ValueKind == JsonValueKind.String)
                    return VisitRef(refEl.GetString()!, name);

                if (schema.TryGetProperty("const", out JsonElement constEl))
                    return AddRule(Sanitize(name), LiteralOf(constEl) + " " + Primitive("ws"));

                if (schema.TryGetProperty("enum", out JsonElement enumEl) &&
                    enumEl.ValueKind == JsonValueKind.Array && enumEl.GetArrayLength() > 0)
                {
                    var alts = enumEl.EnumerateArray().Select(LiteralOf);
                    return AddRule(Sanitize(name), "(" + string.Join(" | ", alts) + ") " + Primitive("ws"));
                }

                if (TryVisitCombinator(schema, name, out string combined))
                    return combined;

                string? type = ReadType(schema, out string[]? unionTypes);
                if (unionTypes != null)
                {
                    var alts = new List<string>();
                    foreach (string t in unionTypes)
                        alts.Add(VisitTyped(schema, t, $"{name}-{t}"));
                    return AddRule(Sanitize(name), string.Join(" | ", alts));
                }

                if (type == null)
                {
                    // No type keyword: infer from the shape-bearing keywords, else
                    // accept any value.
                    if (schema.TryGetProperty("properties", out _) ||
                        schema.TryGetProperty("required", out _) ||
                        schema.TryGetProperty("additionalProperties", out _))
                        type = "object";
                    else if (schema.TryGetProperty("items", out _) ||
                             schema.TryGetProperty("prefixItems", out _))
                        type = "array";
                    else
                        return Primitive("value");
                }

                return VisitTyped(schema, type, name);
            }

            private bool TryVisitCombinator(JsonElement schema, string name, out string rule)
            {
                rule = string.Empty;
                foreach (string key in new[] { "anyOf", "oneOf" })
                {
                    if (schema.TryGetProperty(key, out JsonElement arr) &&
                        arr.ValueKind == JsonValueKind.Array && arr.GetArrayLength() > 0)
                    {
                        var alts = new List<string>();
                        int i = 0;
                        foreach (JsonElement sub in arr.EnumerateArray())
                            alts.Add(Visit(sub, $"{name}-{i++}"));
                        // oneOf is compiled as anyOf: enforcing "exactly one"
                        // needs negation, which a CFG cannot express. Being
                        // permissive here is the safe direction.
                        rule = AddRule(Sanitize(name), string.Join(" | ", alts));
                        return true;
                    }
                }

                if (schema.TryGetProperty("allOf", out JsonElement all) &&
                    all.ValueKind == JsonValueKind.Array && all.GetArrayLength() > 0)
                {
                    // Shallow merge: union the properties and required lists of
                    // the object-shaped members. Anything else falls back to the
                    // first member, which is a superset of the intersection.
                    var merged = MergeAllOf(all);
                    if (merged != null)
                    {
                        rule = VisitMergedObject(merged, name);
                        return true;
                    }
                    rule = Visit(all[0], name);
                    return true;
                }

                return false;
            }

            private sealed class MergedObject
            {
                public readonly List<(string Name, JsonElement Schema)> Properties = new();
                public readonly HashSet<string> Required = new(StringComparer.Ordinal);
                public bool AllowAdditional = true;
            }

            private MergedObject? MergeAllOf(JsonElement all)
            {
                var m = new MergedObject();
                foreach (JsonElement sub in all.EnumerateArray())
                {
                    JsonElement s = sub;
                    if (s.ValueKind == JsonValueKind.Object &&
                        s.TryGetProperty("$ref", out JsonElement r) && r.ValueKind == JsonValueKind.String)
                    {
                        if (!TryResolveRef(r.GetString()!, out s)) return null;
                    }
                    if (s.ValueKind != JsonValueKind.Object) return null;
                    if (s.TryGetProperty("properties", out JsonElement props) &&
                        props.ValueKind == JsonValueKind.Object)
                    {
                        foreach (JsonProperty p in props.EnumerateObject())
                            m.Properties.Add((p.Name, p.Value));
                    }
                    if (s.TryGetProperty("required", out JsonElement req) &&
                        req.ValueKind == JsonValueKind.Array)
                    {
                        foreach (JsonElement e in req.EnumerateArray())
                            if (e.ValueKind == JsonValueKind.String) m.Required.Add(e.GetString()!);
                    }
                    if (s.TryGetProperty("additionalProperties", out JsonElement ap) &&
                        ap.ValueKind == JsonValueKind.False)
                        m.AllowAdditional = false;
                }
                return m.Properties.Count > 0 ? m : null;
            }

            private string VisitMergedObject(MergedObject m, string name)
            {
                var props = m.Properties
                    .Select(p => (p.Name, p.Schema, Required: m.Required.Contains(p.Name)))
                    .ToList();
                return BuildObjectRule(props, m.AllowAdditional, name);
            }

            private string VisitRef(string reference, string name)
            {
                if (_refCache.TryGetValue(reference, out string? cached))
                    return cached;

                if (!TryResolveRef(reference, out JsonElement target))
                    return Primitive("value");

                // Reserve the rule name BEFORE recursing so a self-referential
                // schema ($defs/node with a child of the same type) terminates:
                // the nested $ref hits _refCache and emits this name instead of
                // descending forever.
                string ruleName = Sanitize(reference.Substring(reference.LastIndexOf('/') + 1));
                if (ruleName == "root" || _rules.ContainsKey(ruleName))
                    ruleName = $"{ruleName}-{_counter++}";
                _refCache[reference] = ruleName;
                _rules[ruleName] = Primitive("value");     // placeholder, replaced below

                // The definition compiles to its own rule and this one aliases
                // it, so the name stays a stable target for recursive references.
                string body = Visit(target, ruleName + "-def");
                _rules[ruleName] = body == ruleName ? Primitive("value") : body;
                return ruleName;
            }

            private bool TryResolveRef(string reference, out JsonElement target)
            {
                target = default;
                if (!reference.StartsWith("#", StringComparison.Ordinal)) return false;
                JsonElement cur = _documentRoot;
                foreach (string rawPart in reference.TrimStart('#').Split('/', StringSplitOptions.RemoveEmptyEntries))
                {
                    string part = rawPart.Replace("~1", "/").Replace("~0", "~");
                    if (cur.ValueKind != JsonValueKind.Object || !cur.TryGetProperty(part, out cur))
                        return false;
                }
                target = cur;
                return true;
            }

            private static string? ReadType(JsonElement schema, out string[]? union)
            {
                union = null;
                if (!schema.TryGetProperty("type", out JsonElement t)) return null;
                if (t.ValueKind == JsonValueKind.String) return t.GetString();
                if (t.ValueKind == JsonValueKind.Array)
                {
                    var list = t.EnumerateArray()
                        .Where(e => e.ValueKind == JsonValueKind.String)
                        .Select(e => e.GetString()!)
                        .ToArray();
                    if (list.Length == 1) return list[0];
                    if (list.Length > 1) { union = list; return null; }
                }
                return null;
            }

            private string VisitTyped(JsonElement schema, string type, string name)
            {
                switch (type)
                {
                    case "object": return VisitObject(schema, name);
                    case "array": return VisitArray(schema, name);
                    case "string": return VisitString(schema, name);
                    case "integer": return VisitNumeric(schema, name, integer: true);
                    case "number": return VisitNumeric(schema, name, integer: false);
                    case "boolean": return Primitive("boolean");
                    case "null": return Primitive("null");
                    default: return Primitive("value");
                }
            }

            private string VisitObject(JsonElement schema, string name)
            {
                var props = new List<(string Name, JsonElement Schema, bool Required)>();
                var required = new HashSet<string>(StringComparer.Ordinal);
                if (schema.TryGetProperty("required", out JsonElement req) &&
                    req.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement e in req.EnumerateArray())
                        if (e.ValueKind == JsonValueKind.String) required.Add(e.GetString()!);
                }
                if (schema.TryGetProperty("properties", out JsonElement propsEl) &&
                    propsEl.ValueKind == JsonValueKind.Object)
                {
                    foreach (JsonProperty p in propsEl.EnumerateObject())
                        props.Add((p.Name, p.Value, required.Contains(p.Name)));
                }

                bool allowAdditional = true;
                if (schema.TryGetProperty("additionalProperties", out JsonElement ap))
                    allowAdditional = ap.ValueKind != JsonValueKind.False;

                if (props.Count == 0)
                    return allowAdditional ? Primitive("object-any") : AddRule(Sanitize(name), "\"{\" ws \"}\" ws");

                return BuildObjectRule(props, allowAdditional, name);
            }

            /// <summary>
            /// Emit an object rule. Properties are generated in schema order;
            /// optional ones become a nested optional tail so that any prefix of
            /// the declared order is accepted but the order itself is fixed.
            /// Fixing the order is what makes the grammar a clean CFG — permuting
            /// n optional keys would need n! alternates.
            /// </summary>
            private string BuildObjectRule(
                List<(string Name, JsonElement Schema, bool Required)> props,
                bool allowAdditional,
                string name)
            {
                Primitive("ws");
                string baseName = Sanitize(name);

                var valueRules = new List<string>(props.Count);
                foreach (var p in props)
                    valueRules.Add(Visit(p.Schema, $"{baseName}-{Sanitize(p.Name)}"));

                var required = new List<int>();
                var optional = new List<int>();
                for (int i = 0; i < props.Count; i++)
                    (props[i].Required ? required : optional).Add(i);

                // The key terminal must include its JSON quotes, so serialize the
                // name to JSON first and *then* quote it as a GBNF literal.
                string KeyValue(int i) =>
                    $"{GbnfPropertyName(props[i].Name)} ws \":\" ws {valueRules[i]}";

                // Optional tail: opt_k ::= ("," ws kv_k opt_{k+1})?  — nested so a
                // trailing comma can never be emitted without a following pair.
                string BuildOptionalTail(int from, bool needsLeadingComma)
                {
                    if (from >= optional.Count) return string.Empty;
                    string inner = BuildOptionalTail(from + 1, true);
                    string comma = needsLeadingComma ? "\",\" ws " : string.Empty;
                    string body = $"{comma}{KeyValue(optional[from])}{(inner.Length > 0 ? " " + inner : string.Empty)}";
                    return $"({body})?";
                }

                var sb = new StringBuilder();
                sb.Append("\"{\" ws ");
                if (required.Count > 0)
                {
                    for (int i = 0; i < required.Count; i++)
                    {
                        if (i > 0) sb.Append(" \",\" ws ");
                        sb.Append(KeyValue(required[i]));
                    }
                    string tail = BuildOptionalTail(0, true);
                    if (tail.Length > 0) sb.Append(' ').Append(tail);
                }
                else
                {
                    // Everything optional: the first emitted pair must not be
                    // preceded by a comma, so the head is its own alternation.
                    string tail = BuildOptionalTail(0, false);
                    if (tail.Length > 0) sb.Append(tail);
                }
                sb.Append(" \"}\" ws");

                _ = allowAdditional;    // additional keys are simply not generated
                return AddRule(baseName, sb.ToString());
            }

            private string VisitArray(JsonElement schema, string name)
            {
                Primitive("ws");
                string baseName = Sanitize(name);

                if (schema.TryGetProperty("prefixItems", out JsonElement prefix) &&
                    prefix.ValueKind == JsonValueKind.Array && prefix.GetArrayLength() > 0)
                {
                    var parts = new List<string>();
                    int i = 0;
                    foreach (JsonElement sub in prefix.EnumerateArray())
                        parts.Add(Visit(sub, $"{baseName}-{i++}"));
                    string tuple = string.Join(" \",\" ws ", parts);
                    return AddRule(baseName, $"\"[\" ws {tuple} \"]\" ws");
                }

                string itemRule = schema.TryGetProperty("items", out JsonElement items)
                    ? Visit(items, baseName + "-item")
                    : Primitive("value");

                int min = ReadInt(schema, "minItems", 0);
                int max = ReadInt(schema, "maxItems", -1);

                string body;
                if (min <= 0 && max < 0)
                    body = $"\"[\" ws ({itemRule} (\",\" ws {itemRule})*)? \"]\" ws";
                else if (max < 0)
                    body = $"\"[\" ws {itemRule} (\",\" ws {itemRule}){{{Math.Max(0, min - 1)},}} \"]\" ws";
                else if (max == 0)
                    body = "\"[\" ws \"]\" ws";
                else
                {
                    int lo = Math.Max(0, min - 1);
                    int hi = max - 1;
                    string rest = $"(\",\" ws {itemRule}){{{lo},{hi}}}";
                    body = min <= 0
                        ? $"\"[\" ws ({itemRule} {rest})? \"]\" ws"
                        : $"\"[\" ws {itemRule} {rest} \"]\" ws";
                }
                return AddRule(baseName, body);
            }

            private string VisitString(JsonElement schema, string name)
            {
                Primitive("ws");
                Primitive("char");
                string baseName = Sanitize(name);

                if (schema.TryGetProperty("format", out JsonElement fmt) &&
                    fmt.ValueKind == JsonValueKind.String)
                {
                    string? pattern = fmt.GetString() switch
                    {
                        "date" => "[0-9]{4} \"-\" [0-9]{2} \"-\" [0-9]{2}",
                        "time" => "[0-9]{2} \":\" [0-9]{2} \":\" [0-9]{2} (\".\" [0-9]{1,6})? " +
                                  "(\"Z\" | [+\\-] [0-9]{2} \":\" [0-9]{2})",
                        "date-time" => "[0-9]{4} \"-\" [0-9]{2} \"-\" [0-9]{2} \"T\" " +
                                       "[0-9]{2} \":\" [0-9]{2} \":\" [0-9]{2} (\".\" [0-9]{1,6})? " +
                                       "(\"Z\" | [+\\-] [0-9]{2} \":\" [0-9]{2})",
                        "uuid" => "[0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" " +
                                  "[0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12}",
                        _ => null,
                    };
                    if (pattern != null)
                        return AddRule(baseName, $"\"\\\"\" {pattern} \"\\\"\" ws");
                }

                if (schema.TryGetProperty("pattern", out JsonElement pat) &&
                    pat.ValueKind == JsonValueKind.String &&
                    RegexToGbnf.TryConvert(pat.GetString()!, out string regexBody))
                {
                    return AddRule(baseName, $"\"\\\"\" {regexBody} \"\\\"\" ws");
                }

                int min = ReadInt(schema, "minLength", 0);
                int max = ReadInt(schema, "maxLength", -1);
                if (min <= 0 && max < 0)
                    return Primitive("string");

                string rep = max < 0 ? $"{{{min},}}" : $"{{{min},{max}}}";
                return AddRule(baseName, $"\"\\\"\" char{rep} \"\\\"\" ws");
            }

            private string VisitNumeric(JsonElement schema, string name, bool integer)
            {
                Primitive("ws");
                string baseName = Sanitize(name);

                double? min = ReadDouble(schema, "minimum");
                double? max = ReadDouble(schema, "maximum");
                if (min == null && ReadDouble(schema, "exclusiveMinimum") is double exMin)
                    min = integer ? exMin + 1 : (double?)null;
                if (max == null && ReadDouble(schema, "exclusiveMaximum") is double exMax)
                    max = integer ? exMax - 1 : (double?)null;

                if (integer && min.HasValue && max.HasValue &&
                    min.Value >= long.MinValue && max.Value <= long.MaxValue &&
                    min.Value <= max.Value && max.Value - min.Value <= 1_000_000)
                {
                    string digits = IntegerRangeGrammar.Build((long)min.Value, (long)max.Value);
                    return AddRule(baseName, $"{digits} ws");
                }

                // Only the sign is expressible without a full range grammar.
                bool nonNegative = min.HasValue && min.Value >= 0;
                if (!nonNegative)
                    return Primitive(integer ? "integer" : "number");

                return integer
                    ? AddRule(baseName, "([0-9] | [1-9] [0-9]{0,15}) ws")
                    : AddRule(baseName,
                        "([0-9] | [1-9] [0-9]{0,15}) (\".\" [0-9]{1,16})? ([eE] [-+]? [0-9]{1,3})? ws");
            }

            private static int ReadInt(JsonElement schema, string key, int fallback) =>
                schema.TryGetProperty(key, out JsonElement e) &&
                e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out int v) ? v : fallback;

            private static double? ReadDouble(JsonElement schema, string key) =>
                schema.TryGetProperty(key, out JsonElement e) &&
                e.ValueKind == JsonValueKind.Number && e.TryGetDouble(out double v) ? v : (double?)null;

            /// <summary>Render a JSON value as a GBNF literal matching exactly it.</summary>
            private string LiteralOf(JsonElement value)
            {
                switch (value.ValueKind)
                {
                    case JsonValueKind.String:
                        return GbnfString(JsonSerializer.Serialize(value.GetString()));
                    case JsonValueKind.Number:
                        return GbnfString(value.GetRawText());
                    case JsonValueKind.True: return GbnfString("true");
                    case JsonValueKind.False: return GbnfString("false");
                    case JsonValueKind.Null: return GbnfString("null");
                    default:
                        // Structured const/enum members are compared literally
                        // against their compact JSON encoding.
                        return GbnfString(JsonSerializer.Serialize(value));
                }
            }

            /// <summary>Quote a raw string as a GBNF terminal, escaping as GBNF requires.</summary>
            private static string GbnfString(string raw)
            {
                var sb = new StringBuilder(raw.Length + 8);
                sb.Append('"');
                foreach (char c in raw)
                {
                    switch (c)
                    {
                        case '"': sb.Append("\\\""); break;
                        case '\\': sb.Append("\\\\"); break;
                        case '\n': sb.Append("\\n"); break;
                        case '\r': sb.Append("\\r"); break;
                        case '\t': sb.Append("\\t"); break;
                        default:
                            if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("X4"));
                            else sb.Append(c);
                            break;
                    }
                }
                sb.Append('"');
                return sb.ToString();
            }

            /// <summary>Quote a property name as a JSON string terminal.</summary>
            private static string GbnfPropertyName(string propertyName) =>
                GbnfString(JsonSerializer.Serialize(propertyName));
        }

    }

    /// <summary>
    /// Emits a GBNF alternation matching exactly the integers in
    /// <c>[min, max]</c>. Used for bounded <c>integer</c> schemas, where a plain
    /// digit rule would let the model emit an out-of-range value.
    /// </summary>
    internal static class IntegerRangeGrammar
    {
        public static string Build(long min, long max)
        {
            // Small ranges are cheapest and clearest as a literal alternation;
            // this is also the only form that is exactly right at the boundaries.
            const int MaxEnumerated = 512;
            if (max - min + 1 <= MaxEnumerated)
            {
                var parts = new List<string>();
                for (long v = min; v <= max; v++)
                    parts.Add("\"" + v.ToString(CultureInfo.InvariantCulture) + "\"");
                return "(" + string.Join(" | ", parts) + ")";
            }

            // Wider ranges: constrain the digit count and sign, which bounds the
            // magnitude without enumerating. Deliberately a superset at the
            // extremes rather than a subset -- see the compiler's contract.
            long absMax = Math.Max(Math.Abs(min), Math.Abs(max));
            int digits = absMax.ToString(CultureInfo.InvariantCulture).Length;
            string sign = min < 0 ? "\"-\"? " : string.Empty;
            return digits <= 1
                ? $"({sign}[0-9])"
                : $"({sign}([0-9] | [1-9] [0-9]{{0,{digits - 1}}}))";
        }
    }
}
