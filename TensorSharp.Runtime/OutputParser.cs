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
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Represents a tool function definition provided to the model.
    /// </summary>
    public class ToolFunction
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public Dictionary<string, ToolParameter> Parameters { get; set; } = new();
        public List<string> Required { get; set; } = new();

        /// <summary>
        /// Parse a list of tool definitions from JSON, accepting every shape a
        /// caller plausibly writes:
        ///
        /// <list type="bullet">
        /// <item>this type's own flat shape —
        ///   <c>{"name", "description", "parameters": {"city": {...}}, "required": [...]}</c></item>
        /// <item>the JSON Schema shape the OpenAI API uses, where
        ///   <c>parameters</c> is a schema object —
        ///   <c>{"name", "parameters": {"type": "object", "properties": {...}, "required": [...]}}</c></item>
        /// <item>either of those inside the OpenAI tools wrapper —
        ///   <c>{"type": "function", "function": {...}}</c></item>
        /// </list>
        ///
        /// The second is what anyone copying a tool definition out of an API
        /// request writes, and the server has always accepted it; the CLI's
        /// <c>--tools</c> flag used to deserialize straight into this type and
        /// die with an unhandled <c>JsonException</c> ("The JSON value could not
        /// be converted to ToolParameter") on the schema's own <c>"type":
        /// "object"</c>.
        /// </summary>
        /// <exception cref="JsonException">
        /// The document is not valid JSON, or is not an array/object of tool
        /// definitions. The message names what was expected.
        /// </exception>
        public static List<ToolFunction> ParseList(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                return new List<ToolFunction>();

            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;

            // Tolerate a single object, and the OpenAI request shape where the
            // array hangs off a "tools" property.
            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("tools", out JsonElement toolsProp))
                root = toolsProp;

            var result = new List<ToolFunction>();
            if (root.ValueKind == JsonValueKind.Object)
            {
                result.Add(ParseOne(root));
                return result;
            }
            if (root.ValueKind != JsonValueKind.Array)
                throw new JsonException(
                    "Tool definitions must be a JSON array of objects (or a single object), " +
                    $"but the document root is {root.ValueKind}.");

            foreach (JsonElement entry in root.EnumerateArray())
            {
                if (entry.ValueKind != JsonValueKind.Object)
                    throw new JsonException(
                        $"Each tool definition must be a JSON object, but found {entry.ValueKind}.");
                result.Add(ParseOne(entry));
            }
            return result;
        }

        private static ToolFunction ParseOne(JsonElement entry)
        {
            // OpenAI wrapper: {"type": "function", "function": {...}}
            if (entry.TryGetProperty("function", out JsonElement inner) && inner.ValueKind == JsonValueKind.Object)
                entry = inner;

            var fn = new ToolFunction
            {
                Name = GetString(entry, "name") ?? string.Empty,
                Description = GetString(entry, "description") ?? string.Empty,
            };

            if (!entry.TryGetProperty("parameters", out JsonElement parameters)
                || parameters.ValueKind != JsonValueKind.Object)
            {
                CollectRequired(entry, fn.Required);
                return fn;
            }

            // JSON Schema shape: the properties live one level down and the
            // required list belongs to the schema, not the function.
            JsonElement propertyBag = parameters;
            if (parameters.TryGetProperty("properties", out JsonElement properties)
                && properties.ValueKind == JsonValueKind.Object)
            {
                propertyBag = properties;
                CollectRequired(parameters, fn.Required);
            }

            foreach (JsonProperty prop in propertyBag.EnumerateObject())
            {
                if (prop.Value.ValueKind != JsonValueKind.Object)
                    continue;   // a schema keyword sitting next to "properties" ("type", "$schema", ...)
                var param = new ToolParameter
                {
                    Type = GetString(prop.Value, "type") ?? string.Empty,
                    Description = GetString(prop.Value, "description") ?? string.Empty,
                };
                if (prop.Value.TryGetProperty("enum", out JsonElement enumValues)
                    && enumValues.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement v in enumValues.EnumerateArray())
                        param.Enum.Add(v.ValueKind == JsonValueKind.String ? v.GetString() : v.ToString());
                }
                fn.Parameters[prop.Name] = param;
            }

            // A flat definition carries "required" on the function itself.
            if (fn.Required.Count == 0)
                CollectRequired(entry, fn.Required);
            return fn;
        }

        private static string GetString(JsonElement obj, string name)
            => obj.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String
                ? v.GetString()
                : null;

        private static void CollectRequired(JsonElement obj, List<string> into)
        {
            if (!obj.TryGetProperty("required", out JsonElement req) || req.ValueKind != JsonValueKind.Array)
                return;
            foreach (JsonElement v in req.EnumerateArray())
                if (v.ValueKind == JsonValueKind.String)
                    into.Add(v.GetString());
        }
    }

    public class ToolParameter
    {
        public string Type { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public List<string> Enum { get; set; } = new();
    }

    /// <summary>
    /// Represents a tool call extracted from model output.
    /// </summary>
    public class ToolCall
    {
        public string Name { get; set; } = string.Empty;
        public Dictionary<string, object> Arguments { get; set; } = new();
        public int Index { get; set; }

        public override string ToString()
        {
            string args = Arguments != null ? JsonSerializer.Serialize(Arguments) : "{}";
            return $"{Name}({args})";
        }
    }

    /// <summary>
    /// Parsed output from a model generation step.
    /// </summary>
    public class ParsedOutput
    {
        public string Content { get; set; } = "";
        public string Thinking { get; set; } = "";
        public List<ToolCall>? ToolCalls { get; set; }
    }

    /// <summary>
    /// Streaming parser that extracts thinking content, regular content, and tool calls
    /// from model output. Handles model-specific tag formats.
    /// </summary>
    public interface IOutputParser : IOutputProtocolParser
    {
    }

    // ========================================================================
    // Qwen3 Parser: <think>...</think> for thinking, <tool_call>...</tool_call>
    // ========================================================================

    public class Qwen3OutputParser : IOutputParser
    {
        private enum State { CollectingThinking, ThinkingDone, CollectingContent, CollectingTool }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _stripLeadingThinkTag;
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => false;

        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _callIndex = 0;
            if (enableThinking)
            {
                _state = State.CollectingThinking;
                _stripLeadingThinkTag = true;
            }
            else
            {
                _state = State.CollectingContent;
                _stripLeadingThinkTag = false;
            }
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var thinkingSb = new StringBuilder();
            var contentSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();

                switch (_state)
                {
                    case State.CollectingThinking:
                        if (_stripLeadingThinkTag)
                        {
                            string trimmed = buf.TrimStart();
                            if (trimmed.StartsWith("<think>"))
                            {
                                buf = trimmed.Substring(7).TrimStart();
                                _buffer.Clear();
                                _buffer.Append(buf);
                                _stripLeadingThinkTag = false;
                                keepParsing = buf.Length > 0;
                                break;
                            }
                            if ("<think>".StartsWith(trimmed) && !done)
                                break;
                            _stripLeadingThinkTag = false;
                        }

                        int closeIdx = buf.IndexOf("</think>", StringComparison.Ordinal);
                        int toolIdx = buf.IndexOf("<tool_call>", StringComparison.Ordinal);

                        if (toolIdx >= 0 && (closeIdx < 0 || toolIdx < closeIdx))
                        {
                            string before = buf.Substring(0, toolIdx).TrimEnd();
                            string after = buf.Substring(toolIdx + 11).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) thinkingSb.Append(before);
                            _state = State.CollectingTool;
                            keepParsing = true;
                        }
                        else if (closeIdx >= 0)
                        {
                            string thinking = buf.Substring(0, closeIdx).TrimEnd();
                            string after = buf.Substring(closeIdx + 8).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (thinking.Length > 0) thinkingSb.Append(thinking);
                            _state = after.Length > 0 ? State.CollectingContent : State.ThinkingDone;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            if (buf.Length > 0) thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, "</think>", "<tool_call>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) thinkingSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                thinkingSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        break;

                    case State.ThinkingDone:
                        string td = buf.TrimStart();
                        _buffer.Clear();
                        if (td.Length > 0)
                        {
                            _buffer.Append(td);
                            _state = State.CollectingContent;
                            keepParsing = true;
                        }
                        break;

                    case State.CollectingContent:
                        int tcIdx = buf.IndexOf("<tool_call>", StringComparison.Ordinal);
                        if (tcIdx >= 0)
                        {
                            string before = buf.Substring(0, tcIdx).TrimEnd();
                            string after = buf.Substring(tcIdx + 11).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingTool;
                            keepParsing = true;
                        }
                        else if (done)
                        {
                            if (buf.Length > 0) contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, "<tool_call>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) contentSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                contentSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        break;

                    case State.CollectingTool:
                        int endIdx = buf.IndexOf("</tool_call>", StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            string raw = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + 12).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            var tc = ParseQwen3ToolCall(raw);
                            if (tc != null) toolCalls.Add(tc);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            var tc = ParseQwen3ToolCall(buf);
                            if (tc != null) toolCalls.Add(tc);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        private ToolCall? ParseQwen3ToolCall(string raw)
        {
            raw = raw.Trim();
            if (raw.Length == 0) return null;
            try
            {
                using var doc = JsonDocument.Parse(raw);
                var root = doc.RootElement;
                string? name = root.GetProperty("name").GetString();
                if (string.IsNullOrEmpty(name)) return null;

                var args = new Dictionary<string, object>();
                if (root.TryGetProperty("arguments", out var argsEl) && argsEl.ValueKind == JsonValueKind.Object)
                {
                    foreach (var prop in argsEl.EnumerateObject())
                        args[prop.Name] = JsonElementToObject(prop.Value);
                }
                return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
            }
            catch (JsonException)
            {
                // Qwen 3.5 emits the XML-ish call body instead of a JSON object:
                //   <function=get_weather>
                //   <parameter=city>\nParis\n</parameter>
                //   </function>
                // Dropping it silently loses the whole turn (the text was already
                // consumed as a tool call), so fall back to that form here.
                return ParseXmlToolCall(raw);
            }
        }

        /// <summary>
        /// Parse the `&lt;function=NAME&gt;&lt;parameter=KEY&gt;VALUE&lt;/parameter&gt;&lt;/function&gt;`
        /// tool-call body. Each parameter value is trimmed of the surrounding
        /// newlines the template emits, and parsed as JSON when it is a scalar /
        /// object / array so numbers and booleans do not arrive quoted.
        /// </summary>
        private ToolCall? ParseXmlToolCall(string raw)
        {
            const string fnOpen = "<function=";
            int fnIdx = raw.IndexOf(fnOpen, StringComparison.Ordinal);
            if (fnIdx < 0) return null;
            int nameEnd = raw.IndexOf('>', fnIdx + fnOpen.Length);
            if (nameEnd < 0) return null;

            string name = raw.Substring(fnIdx + fnOpen.Length, nameEnd - fnIdx - fnOpen.Length).Trim();
            if (name.Length == 0) return null;

            var args = new Dictionary<string, object>();
            const string paramOpen = "<parameter=";
            const string paramClose = "</parameter>";
            int pos = nameEnd + 1;
            while (true)
            {
                int pIdx = raw.IndexOf(paramOpen, pos, StringComparison.Ordinal);
                if (pIdx < 0) break;
                int keyEnd = raw.IndexOf('>', pIdx + paramOpen.Length);
                if (keyEnd < 0) break;
                string key = raw.Substring(pIdx + paramOpen.Length, keyEnd - pIdx - paramOpen.Length).Trim();

                int valEnd = raw.IndexOf(paramClose, keyEnd + 1, StringComparison.Ordinal);
                string value = valEnd < 0
                    ? raw.Substring(keyEnd + 1)
                    : raw.Substring(keyEnd + 1, valEnd - keyEnd - 1);
                if (key.Length > 0)
                    args[key] = ParseScalarOrText(value.Trim());

                if (valEnd < 0) break;
                pos = valEnd + paramClose.Length;
            }

            return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
        }

        private static object ParseScalarOrText(string value)
        {
            if (value.Length == 0) return value;
            char c = value[0];
            bool looksJson = c == '{' || c == '[' || c == '-' || char.IsDigit(c) ||
                             value == "true" || value == "false" || value == "null";
            if (looksJson)
            {
                try
                {
                    using var doc = JsonDocument.Parse(value);
                    return JsonElementToObject(doc.RootElement);
                }
                catch (JsonException)
                {
                    // Not JSON after all (e.g. a date like 2026-08-01): keep the text.
                }
            }
            return value;
        }

        private static int HoldBackForPartialTag(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }

        internal static object JsonElementToObject(JsonElement el)
        {
            return el.ValueKind switch
            {
                JsonValueKind.String => el.GetString() ?? string.Empty,
                JsonValueKind.Number => el.TryGetInt64(out long l) ? (object)l : el.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null!,
                JsonValueKind.Object => JsonElementToDict(el),
                JsonValueKind.Array => JsonElementToList(el),
                _ => el.GetRawText()
            };
        }

        private static Dictionary<string, object> JsonElementToDict(JsonElement el)
        {
            var d = new Dictionary<string, object>();
            foreach (var p in el.EnumerateObject())
                d[p.Name] = JsonElementToObject(p.Value);
            return d;
        }

        private static List<object> JsonElementToList(JsonElement el)
        {
            var list = new List<object>();
            foreach (var item in el.EnumerateArray())
                list.Add(JsonElementToObject(item));
            return list;
        }
    }

    // ========================================================================
    // Qwen3.5 Parser: same tags as Qwen3, always starts in thinking mode
    // ========================================================================

    public class Qwen35OutputParser : Qwen3OutputParser
    {
    }

    // ========================================================================
    // Gemma4 Parser: <|channel>thought\n...<channel|> for thinking,
    //                <|tool_call>call:NAME{args}<tool_call|> for tool calls
    // ========================================================================

    public class Gemma4OutputParser : IOutputParser
    {
        private enum State { CollectingContent, CollectingThinking, CollectingToolCall }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _thinkingEnabled;
        private bool _needsChannelNameStrip;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _needsChannelNameStrip = false;
            _state = State.CollectingContent;
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var thinkingSb = new StringBuilder();
            var contentSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0) break;

                switch (_state)
                {
                    case State.CollectingContent:
                        int chIdx = buf.IndexOf("<|channel>", StringComparison.Ordinal);
                        int tcIdx = buf.IndexOf("<|tool_call>", StringComparison.Ordinal);
                        // A CLOSING <channel|> with no opener in the generated text:
                        // the block was opened by the prompt's channel primer, so the
                        // model is closing a thinking block we never saw start. Gemma 4
                        // does this even with thinking disabled (the primer is a
                        // complete empty block, but smaller checkpoints still reason
                        // first), and treating it as content is what surfaced the whole
                        // chain of thought — plus the raw marker — as the answer.
                        int strayCloseIdx = buf.IndexOf("<channel|>", StringComparison.Ordinal);

                        if (strayCloseIdx >= 0
                            && (chIdx < 0 || strayCloseIdx < chIdx)
                            && (tcIdx < 0 || strayCloseIdx < tcIdx))
                        {
                            string thought = buf.Substring(0, strayCloseIdx);
                            // The model may re-emit the channel name it is closing.
                            if (thought.StartsWith("thought\n", StringComparison.Ordinal))
                                thought = thought.Substring(8);
                            string after = buf.Substring(strayCloseIdx + 10).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            // Only text still buffered can be reclassified: a streaming
                            // consumer already received whatever was flushed before the
                            // marker arrived, and there is no bounded lookahead that
                            // would let us hold an arbitrarily long thought block back.
                            // Batch callers (Add(full, done: true)) buffer the whole
                            // output, so they get the split exactly right.
                            thought = thought.Trim();
                            if (thought.Length > 0 && _thinkingEnabled) thinkingSb.Append(thought);
                            keepParsing = after.Length > 0;
                        }
                        else if (chIdx >= 0 && (tcIdx < 0 || chIdx < tcIdx))
                        {
                            string before = buf.Substring(0, chIdx).TrimEnd();
                            string after = buf.Substring(chIdx + 10);
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingThinking;
                            _needsChannelNameStrip = true;
                            keepParsing = true;
                        }
                        else if (tcIdx >= 0)
                        {
                            string before = buf.Substring(0, tcIdx).TrimEnd();
                            string after = buf.Substring(tcIdx + 12);
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingToolCall;
                            keepParsing = true;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, "<|channel>", "<|tool_call>", "<channel|>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) contentSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                contentSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0) contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        break;

                    case State.CollectingThinking:
                        if (_needsChannelNameStrip)
                        {
                            if (buf.StartsWith("thought\n"))
                            {
                                buf = buf.Substring(8);
                                _buffer.Clear();
                                _buffer.Append(buf);
                                _needsChannelNameStrip = false;
                                keepParsing = buf.Length > 0;
                                break;
                            }
                            if (!done && ("thought\n".StartsWith(buf) || buf.StartsWith("thought")))
                                break;
                            _needsChannelNameStrip = false;
                        }

                        int closeIdx = buf.IndexOf("<channel|>", StringComparison.Ordinal);
                        if (closeIdx >= 0)
                        {
                            string thinking = buf.Substring(0, closeIdx).TrimEnd();
                            string after = buf.Substring(closeIdx + 10).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (thinking.Length > 0 && _thinkingEnabled) thinkingSb.Append(thinking);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, "<channel|>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0 && _thinkingEnabled) thinkingSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                if (_thinkingEnabled) thinkingSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0 && _thinkingEnabled) thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        break;

                    case State.CollectingToolCall:
                        int endIdx = buf.IndexOf("<tool_call|>", StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            string raw = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + 12).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            var tc = ParseGemma4ToolCall(raw);
                            if (tc != null) toolCalls.Add(tc);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            var tc = ParseGemma4ToolCall(buf);
                            if (tc != null) toolCalls.Add(tc);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        private static readonly Regex GemmaQuotedStringRe = new(@"<\|""\|>(.*?)<\|""\|>", RegexOptions.Singleline);
        private static readonly Regex GemmaBareKeyRe = new(@"([,{])(\w+):");

        private static ToolCall? ParseGemma4ToolCall(string content)
        {
            content = content.Trim();
            if (!content.StartsWith("call:")) return null;
            content = content.Substring(5);

            int braceIdx = content.IndexOf('{');
            if (braceIdx < 0) return null;

            string name = content.Substring(0, braceIdx).Trim();
            string argsStr = content.Substring(braceIdx);

            string json = Gemma4ArgsToJson(argsStr);
            try
            {
                using var doc = JsonDocument.Parse(json);
                var args = new Dictionary<string, object>();
                foreach (var prop in doc.RootElement.EnumerateObject())
                    args[prop.Name] = Qwen3OutputParser.JsonElementToObject(prop.Value);
                return new ToolCall { Name = name, Arguments = args };
            }
            catch
            {
                return null;
            }
        }

        internal static string Gemma4ArgsToJson(string s)
        {
            var quotedStrings = new List<string>();
            string text = GemmaQuotedStringRe.Replace(s, m =>
            {
                quotedStrings.Add(m.Groups[1].Value);
                return "\x00" + (char)(quotedStrings.Count - 1) + "\x00";
            });

            text = GemmaBareKeyRe.Replace(text, "$1\"$2\":");

            for (int i = 0; i < quotedStrings.Count; i++)
            {
                string escaped = JsonSerializer.Serialize(quotedStrings[i]);
                text = text.Replace("\x00" + (char)i + "\x00", escaped);
            }

            return text;
        }

        private static int HoldBack(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // GPT OSS / Harmony Parser
    // Uses <|start|>...<|end|> message framing with <|message|> header end,
    // <|channel|>analysis for thinking, <|channel|>final for content
    // ========================================================================

    public class HarmonyOutputParser : IOutputParser
    {
        private enum HState { LookingForStart, ParsingHeader, ParsingContent }

        private HState _state;
        private readonly StringBuilder _buffer = new();
        private readonly StringBuilder _toolArgs = new();
        private string? _currentChannel;
        private string? _currentRecipient;
        private int _callIndex;

        private const string MsgStartTag = "<|start|>";
        private const string MsgEndTag = "<|end|>";
        private const string CallTag = "<|call|>";
        private const string ReturnTag = "<|return|>";
        private const string HeaderEndTag = "<|message|>";
        private const string ChannelTag = "<|channel|>";
        private const string FunctionPrefix = "functions.";

        // Tags that terminate a content message during generation.
        private static readonly string[] EndTags = { MsgEndTag, CallTag, ReturnTag };
        // Tags whose partial suffixes must be held back while streaming content.
        private static readonly string[] HoldTags = { MsgEndTag, CallTag, ReturnTag, MsgStartTag };

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _toolArgs.Clear();
            _state = HState.LookingForStart;
            _currentChannel = null;
            _currentRecipient = null;
            _callIndex = 0;

            // The prompt's generation marker is "<|start|>assistant", so the
            // model's first emitted token is "<|channel|>". Prime the buffer so
            // the parser is already past the start tag.
            _buffer.Append("<|start|>assistant");
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0)
                {
                    // A generation that stops at EOS emits no closing
                    // <|end|>/<|call|>/<|return|> tag, and its last content chunk
                    // may already have been drained into `_toolArgs`. Finalizing
                    // here is what keeps that trailing message — in particular a
                    // commentary tool call, the whole answer for a function-call
                    // turn — from being dropped on the floor.
                    if (done && _state == HState.ParsingContent)
                    {
                        FinalizeMessage(toolCalls);
                        _state = HState.LookingForStart;
                    }
                    break;
                }

                switch (_state)
                {
                    case HState.LookingForStart:
                        int startIdx = buf.IndexOf(MsgStartTag, StringComparison.Ordinal);
                        if (startIdx >= 0)
                        {
                            string after = buf.Substring(startIdx + MsgStartTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = HState.ParsingHeader;
                            keepParsing = true;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, MsgStartTag);
                            if (hold > 0)
                            {
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;

                    case HState.ParsingHeader:
                        int headerEnd = buf.IndexOf(HeaderEndTag, StringComparison.Ordinal);
                        if (headerEnd >= 0)
                        {
                            string header = buf.Substring(0, headerEnd);
                            string after = buf.Substring(headerEnd + HeaderEndTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);

                            ParseHeader(header);

                            _state = HState.ParsingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, HeaderEndTag);
                            if (hold > 0 && hold < buf.Length)
                            {
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;

                    case HState.ParsingContent:
                        int endIdx = FindEarliestEndTag(buf, out int tagLen);
                        if (endIdx >= 0)
                        {
                            string content = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + tagLen);
                            _buffer.Clear();
                            _buffer.Append(after);

                            EmitContent(content, contentSb, thinkingSb);
                            FinalizeMessage(toolCalls);
                            _state = HState.LookingForStart;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, HoldTags);
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) EmitContent(emit, contentSb, thinkingSb);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                EmitContent(buf, contentSb, thinkingSb);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0) EmitContent(buf, contentSb, thinkingSb);
                            FinalizeMessage(toolCalls);
                            _buffer.Clear();
                            _state = HState.LookingForStart;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            if (toolCalls.Count > 0)
                result.ToolCalls = toolCalls;
            return result;
        }

        /// <summary>
        /// Parse a message header (the text between &lt;|start|&gt; and &lt;|message|&gt;)
        /// to extract the channel and, for tool calls, the "to=functions.NAME" recipient.
        /// Handles both header orderings (recipient before or after the channel tag).
        /// </summary>
        private void ParseHeader(string header)
        {
            int chIdx = header.IndexOf(ChannelTag, StringComparison.Ordinal);
            if (chIdx >= 0)
            {
                string channelPart = header.Substring(chIdx + ChannelTag.Length);
                int spaceIdx = channelPart.IndexOfAny(new[] { ' ', '\t', '\n', '\r' });
                _currentChannel = spaceIdx >= 0 ? channelPart.Substring(0, spaceIdx) : channelPart;
            }
            else
            {
                _currentChannel = "final";
            }

            _currentRecipient = null;
            int toIdx = header.IndexOf("to=", StringComparison.Ordinal);
            if (toIdx >= 0)
            {
                string rest = header.Substring(toIdx + 3);
                int end = 0;
                while (end < rest.Length && !char.IsWhiteSpace(rest[end]) && rest[end] != '<')
                    end++;
                if (end > 0)
                    _currentRecipient = rest.Substring(0, end);
            }
        }

        private void EmitContent(string content, StringBuilder contentSb, StringBuilder thinkingSb)
        {
            if (content.Length == 0) return;
            if (IsToolCall())
                _toolArgs.Append(content);
            else if (_currentChannel == "analysis")
                thinkingSb.Append(content);
            else
                contentSb.Append(content);
        }

        /// <summary>Finalize the current message: emit a tool call if it targeted functions.*.</summary>
        private void FinalizeMessage(List<ToolCall> toolCalls)
        {
            if (IsToolCall())
            {
                var tc = BuildToolCall();
                if (tc != null)
                    toolCalls.Add(tc);
            }
            _toolArgs.Clear();
            _currentRecipient = null;
        }

        private bool IsToolCall() =>
            _currentRecipient != null && _currentRecipient.StartsWith(FunctionPrefix, StringComparison.Ordinal);

        private ToolCall? BuildToolCall()
        {
            string name = _currentRecipient!.Substring(FunctionPrefix.Length);
            if (string.IsNullOrEmpty(name)) return null;

            var args = new Dictionary<string, object>();
            string raw = _toolArgs.ToString().Trim();
            if (raw.Length > 0)
            {
                try
                {
                    using var doc = JsonDocument.Parse(raw);
                    if (doc.RootElement.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var prop in doc.RootElement.EnumerateObject())
                            args[prop.Name] = Qwen3OutputParser.JsonElementToObject(prop.Value);
                    }
                }
                catch
                {
                    // Malformed JSON: surface the call with no parsed arguments
                    // rather than dropping it entirely.
                }
            }
            return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
        }

        /// <summary>Find the earliest message-terminating tag in the buffer.</summary>
        private static int FindEarliestEndTag(string buf, out int tagLen)
        {
            int best = -1;
            tagLen = 0;
            foreach (var tag in EndTags)
            {
                int idx = buf.IndexOf(tag, StringComparison.Ordinal);
                if (idx >= 0 && (best < 0 || idx < best))
                {
                    best = idx;
                    tagLen = tag.Length;
                }
            }
            return best;
        }

        private static int HoldBack(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // Passthrough parser (no thinking/tool parsing)
    // ========================================================================

    public class PassthroughOutputParser : IOutputParser
    {
        public bool HasThinkingSupport => false;
        public bool HasToolSupport => false;
        public bool AlwaysRequired => false;

        public void Init(bool enableThinking, List<ToolFunction> tools) { }

        public ParsedOutput Add(string text, bool done)
        {
            return new ParsedOutput { Content = text };
        }
    }

    // ========================================================================
    // DeepSeek V4 Parser: <think>...</think> for reasoning, and DSML markup for
    // tool calls:
    //     <｜DSML｜tool_calls>
    //     <｜DSML｜invoke name="get_weather">
    //     <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
    //     </｜DSML｜invoke>
    //     </｜DSML｜tool_calls>
    // `string="true"` means the value is the raw text between the tags; anything
    // else is JSON. Multiple <invoke> blocks in one call block are parallel calls.
    // ========================================================================

    public class DeepSeek4OutputParser : IOutputParser
    {
        private enum State { Content, Thinking, ToolCalls }

        private const string ThinkOpen = "<think>";
        private const string ThinkClose = "</think>";
        private const string Dsml = "｜DSML｜";
        private const string CallsOpen = "<" + Dsml + "tool_calls>";
        private const string CallsClose = "</" + Dsml + "tool_calls>";

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _thinkingEnabled;
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _callIndex = 0;
            // The generation prompt already emitted `<think>` (thinking) or
            // `</think>` (not), so the model's own output starts inside the
            // reasoning block or straight in content.
            _state = enableThinking ? State.Thinking : State.Content;
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0)
                    break;

                switch (_state)
                {
                    case State.Thinking:
                    {
                        int closeIdx = buf.IndexOf(ThinkClose, StringComparison.Ordinal);
                        if (closeIdx >= 0)
                        {
                            thinkingSb.Append(buf, 0, closeIdx);
                            string after = buf.Substring(closeIdx + ThinkClose.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Content;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, ThinkClose);
                            if (hold < buf.Length)
                            {
                                thinkingSb.Append(buf, 0, buf.Length - hold);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;
                    }

                    case State.Content:
                    {
                        int callIdx = buf.IndexOf(CallsOpen, StringComparison.Ordinal);
                        if (callIdx >= 0)
                        {
                            contentSb.Append(buf, 0, callIdx);
                            string after = buf.Substring(callIdx + CallsOpen.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.ToolCalls;
                            keepParsing = true;
                            break;
                        }
                        // A late <think> can still open (the model may reason
                        // before answering even when the prompt closed the block).
                        int thinkIdx = _thinkingEnabled ? buf.IndexOf(ThinkOpen, StringComparison.Ordinal) : -1;
                        if (thinkIdx >= 0)
                        {
                            contentSb.Append(buf, 0, thinkIdx);
                            string after = buf.Substring(thinkIdx + ThinkOpen.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Thinking;
                            keepParsing = after.Length > 0;
                            break;
                        }
                        if (done)
                        {
                            contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, CallsOpen, ThinkOpen);
                            if (hold < buf.Length)
                            {
                                contentSb.Append(buf, 0, buf.Length - hold);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;
                    }

                    case State.ToolCalls:
                    {
                        int endIdx = buf.IndexOf(CallsClose, StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            ParseInvokes(buf.Substring(0, endIdx), toolCalls);
                            string after = buf.Substring(endIdx + CallsClose.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Content;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            // Generation stopped inside the block (hit the token
                            // budget, or EOS right after the last </invoke>):
                            // surface whatever invokes completed.
                            ParseInvokes(buf, toolCalls);
                            _buffer.Clear();
                            _state = State.Content;
                        }
                        break;
                    }
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        /// <summary>Parse every complete `&lt;invoke&gt;` block in the body.</summary>
        private void ParseInvokes(string body, List<ToolCall> toolCalls)
        {
            const string invokeOpen = "<" + Dsml + "invoke name=\"";
            const string invokeClose = "</" + Dsml + "invoke>";
            const string paramOpen = "<" + Dsml + "parameter name=\"";
            const string paramClose = "</" + Dsml + "parameter>";

            int pos = 0;
            while (true)
            {
                int start = body.IndexOf(invokeOpen, pos, StringComparison.Ordinal);
                if (start < 0)
                    break;
                int nameEnd = body.IndexOf('"', start + invokeOpen.Length);
                if (nameEnd < 0)
                    break;
                string name = body.Substring(start + invokeOpen.Length, nameEnd - start - invokeOpen.Length);

                int end = body.IndexOf(invokeClose, nameEnd, StringComparison.Ordinal);
                string inner = end < 0 ? body.Substring(nameEnd) : body.Substring(nameEnd, end - nameEnd);

                var args = new Dictionary<string, object>();
                int p = 0;
                while (true)
                {
                    int pStart = inner.IndexOf(paramOpen, p, StringComparison.Ordinal);
                    if (pStart < 0)
                        break;
                    int keyEnd = inner.IndexOf('"', pStart + paramOpen.Length);
                    if (keyEnd < 0)
                        break;
                    string key = inner.Substring(pStart + paramOpen.Length, keyEnd - pStart - paramOpen.Length);

                    // string="true|false" decides whether the value is raw text
                    // or JSON; a missing attribute is treated as text.
                    int tagEnd = inner.IndexOf('>', keyEnd);
                    if (tagEnd < 0)
                        break;
                    string attrs = inner.Substring(keyEnd, tagEnd - keyEnd);
                    bool isString = !attrs.Contains("string=\"false\"", StringComparison.Ordinal);

                    int valEnd = inner.IndexOf(paramClose, tagEnd + 1, StringComparison.Ordinal);
                    string raw = valEnd < 0
                        ? inner.Substring(tagEnd + 1)
                        : inner.Substring(tagEnd + 1, valEnd - tagEnd - 1);

                    if (key.Length > 0)
                        args[key] = isString ? raw.Trim() : ParseJsonValue(raw.Trim());

                    if (valEnd < 0)
                        break;
                    p = valEnd + paramClose.Length;
                }

                if (name.Length > 0)
                    toolCalls.Add(new ToolCall { Name = name, Arguments = args, Index = _callIndex++ });

                if (end < 0)
                    break;
                pos = end + invokeClose.Length;
            }
        }

        private static object ParseJsonValue(string value)
        {
            if (value.Length == 0)
                return value;
            try
            {
                using var doc = JsonDocument.Parse(value);
                return Qwen3OutputParser.JsonElementToObject(doc.RootElement);
            }
            catch (JsonException)
            {
                // The model labelled it non-string but did not write JSON; the
                // text is still better than dropping the argument.
                return value;
            }
        }

        private static int HoldBackForPartialTag(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // Factory
    // ========================================================================

    public static class OutputParserFactory
    {
        public static IOutputParser Create(string architecture)
        {
            return architecture switch
            {
                "gemma4" => new Gemma4OutputParser(),
                "qwen3" => new Qwen3OutputParser(),
                "qwen35" or "qwen35moe" or "qwen3next" or "qwen3vl" or "qwen3vlmoe" => new Qwen35OutputParser(),
                "gptoss" or "gpt-oss" => new HarmonyOutputParser(),
                "deepseek4" => new DeepSeek4OutputParser(),
                "nemotron_h" or "nemotron_h_moe" => new Qwen3OutputParser(),
                _ => new PassthroughOutputParser()
            };
        }

        /// <summary>
        /// Text after which a structured-output grammar may start enforcing, or
        /// null when the model's very first token is already part of the answer.
        ///
        /// GPT-OSS opens every reply with a harmony channel header and reasons in
        /// the <c>analysis</c> channel before answering in <c>final</c>. A grammar
        /// armed from token 0 forbids that header, so the model is pushed straight
        /// into a JSON object having done no reasoning and fills the schema with
        /// placeholders. Arming on the final channel's header instead lets it
        /// think and constrains only the answer. See
        /// <c>GrammarConstraint.ActivateAfter</c>.
        /// </summary>
        public static string? GrammarActivationTrigger(string architecture)
            => architecture is "gptoss" or "gpt-oss" ? "final<|message|>" : null;

        public static bool IsAlwaysRequired(string architecture)
        {
            // DeepSeek V4 joins this set because its reasoning block and its DSML
            // tool calls both arrive as plain text: without the parser the
            // </think> marker and the whole <｜DSML｜tool_calls> block would be
            // streamed to the client as if they were the answer.
            return architecture is "gptoss" or "gpt-oss" or "gemma4" or "deepseek4";
        }
    }
}

