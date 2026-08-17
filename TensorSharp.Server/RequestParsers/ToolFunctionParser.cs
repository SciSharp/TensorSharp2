// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Models;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Parsers for the <c>tools</c> field across all three protocols. Returns
    /// null when no tools are declared so downstream code can short-circuit
    /// (the parsing path is identical, only the wrapping differs: Ollama nests
    /// the function under <c>"function"</c>; OpenAI additionally requires
    /// <c>"type": "function"</c>).
    /// <para>
    /// Every read goes through the kind-checked helpers at the bottom of this
    /// file rather than <see cref="JsonElement"/>'s typed accessors, because
    /// those accessors throw <see cref="System.InvalidOperationException"/> when
    /// the element is of another kind — and a tool's <c>parameters</c> is a JSON
    /// Schema document, in which several fields legally hold non-strings:
    /// </para>
    /// <list type="bullet">
    /// <item><c>enum</c> takes values of any type — <c>"enum": [1, 2, 3]</c> on an
    ///   integer parameter is as valid as a list of strings</item>
    /// <item><c>type</c> is a type name <i>or an array of them</i> —
    ///   <c>"type": ["string", "null"]</c> is what most schema generators emit for
    ///   a nullable field</item>
    /// <item><c>parameters</c> is frequently <c>null</c> on a tool that takes no
    ///   arguments, and a property schema may itself be a boolean</item>
    /// </list>
    /// <para>
    /// Each of those used to escape as an unhandled exception and fail the whole
    /// chat request rather than the one field it came from, so a harness sending
    /// a perfectly valid OpenAI tool spec got a 500 instead of a completion
    /// (https://github.com/zhongkaifu/TensorSharp/issues/142). Nothing here
    /// throws: a field of an unexpected kind is skipped, not fatal.
    /// </para>
    /// </summary>
    internal static class ToolFunctionParser
    {
        public static List<ToolFunction> ParseOllama(JsonElement body)
        {
            if (!TryGetArray(body, "tools", out var toolsEl))
                return null;

            var tools = new List<ToolFunction>();
            foreach (var toolEl in toolsEl.EnumerateArray())
            {
                if (!TryGetObject(toolEl, "function", out var fnEl))
                    continue;
                tools.Add(ParseFunction(fnEl));
            }
            return tools.Count > 0 ? tools : null;
        }

        public static List<ToolFunction> ParseOpenAI(JsonElement body)
        {
            if (!TryGetArray(body, "tools", out var toolsEl))
                return null;

            var tools = new List<ToolFunction>();
            foreach (var toolEl in toolsEl.EnumerateArray())
            {
                if (!IsFunctionTool(toolEl)) continue;
                if (!TryGetObject(toolEl, "function", out var fnEl)) continue;

                tools.Add(ParseFunction(fnEl));
            }
            return tools.Count > 0 ? tools : null;
        }

        /// <summary>
        /// Parse the Responses API's flat <c>tools</c> shape: each entry is
        /// <c>{type:"function", name, description, parameters}</c> directly
        /// (no nested <c>"function"</c> wrapper like Chat Completions uses).
        /// </summary>
        public static List<ToolFunction> ParseOpenAIResponses(JsonElement body)
        {
            if (!TryGetArray(body, "tools", out var toolsEl))
                return null;

            var tools = new List<ToolFunction>();
            foreach (var toolEl in toolsEl.EnumerateArray())
            {
                if (!IsFunctionTool(toolEl)) continue;

                tools.Add(ParseFunction(toolEl));
            }
            return tools.Count > 0 ? tools : null;
        }

        /// <summary>
        /// An OpenAI tools entry names its kind in <c>type</c>; anything else —
        /// the Responses API also carries server-side built-ins such as
        /// <c>web_search</c> — is not a function we can offer the model. A
        /// missing or null <c>type</c> means "function", which is what every
        /// Chat Completions client intends, while a <c>type</c> that is not a
        /// string at all is not a function declaration.
        /// </summary>
        private static bool IsFunctionTool(JsonElement toolEl)
        {
            if (toolEl.ValueKind != JsonValueKind.Object)
                return false;
            if (!toolEl.TryGetProperty("type", out var typeEl))
                return true;

            return typeEl.ValueKind switch
            {
                JsonValueKind.Null => true,
                JsonValueKind.String => typeEl.GetString() == "function",
                _ => false,
            };
        }

        /// <summary>
        /// Parse one function declaration — <c>{name, description, parameters}</c>,
        /// where <c>parameters</c> is a JSON Schema object. Never returns null.
        /// </summary>
        private static ToolFunction ParseFunction(JsonElement fnEl)
        {
            var tf = new ToolFunction
            {
                Name = ReadString(fnEl, "name") ?? string.Empty,
                Description = ReadString(fnEl, "description") ?? string.Empty
            };

            // "parameters": null on an argument-less tool is common, and so is
            // omitting it entirely; both leave the defaults from ToolFunction.
            if (!TryGetObject(fnEl, "parameters", out var paramsEl))
                return tf;

            if (TryGetObject(paramsEl, "properties", out var propsEl))
            {
                tf.Parameters = new Dictionary<string, ToolParameter>();
                foreach (var prop in propsEl.EnumerateObject())
                    tf.Parameters[prop.Name] = ParseParameter(prop.Value);
            }

            if (TryGetArray(paramsEl, "required", out var reqEl))
            {
                // "required" lists property names, so a non-string entry names
                // nothing and is dropped rather than carried through as a null.
                var required = new List<string>();
                foreach (var item in reqEl.EnumerateArray())
                {
                    if (item.ValueKind == JsonValueKind.String)
                        required.Add(item.GetString());
                }
                tf.Required = required;
            }

            return tf;
        }

        /// <summary>
        /// Parse one property schema. An absent <c>type</c> keeps the historical
        /// <c>"string"</c> default; a schema that is not even an object (JSON
        /// Schema allows a bare <c>true</c>/<c>false</c> in a property slot)
        /// yields that same untyped default.
        /// </summary>
        private static ToolParameter ParseParameter(JsonElement schema)
        {
            var tp = new ToolParameter
            {
                Type = ReadSchemaType(schema) ?? "string",
                Description = ReadString(schema, "description") ?? string.Empty
            };

            if (TryGetArray(schema, "enum", out var enumEl))
            {
                // ToolParameter.Enum is a list of strings, but the schema's values
                // need not be. Render a non-string member as its raw JSON text so
                // it round-trips exactly when the prompt renderers re-serialize
                // the schema: 1 stays 1 and true stays true. JsonElement.ToString()
                // is the wrong tool here — it renders booleans in .NET's casing
                // ("True") and null as an empty string.
                var values = new List<string>();
                foreach (var item in enumEl.EnumerateArray())
                    values.Add(item.ValueKind == JsonValueKind.String ? item.GetString() : item.GetRawText());
                tp.Enum = values;
            }

            return tp;
        }

        /// <summary>
        /// Read a property schema's <c>type</c>. JSON Schema allows a union —
        /// <c>"type": ["string", "null"]</c> is the standard spelling of a
        /// nullable field — while <see cref="ToolParameter.Type"/> holds a single
        /// name, because it is re-emitted into the prompt as a JSON Schema string
        /// and switched on when rendering Harmony's TypeScript tool namespace. So
        /// keep the first real type and drop the <c>"null"</c> member, whose
        /// meaning the <c>required</c> list already carries.
        /// </summary>
        private static string ReadSchemaType(JsonElement schema)
        {
            if (schema.ValueKind != JsonValueKind.Object || !schema.TryGetProperty("type", out var typeEl))
                return null;

            if (typeEl.ValueKind == JsonValueKind.String)
                return typeEl.GetString();

            if (typeEl.ValueKind == JsonValueKind.Array)
            {
                string first = null;
                foreach (var item in typeEl.EnumerateArray())
                {
                    if (item.ValueKind != JsonValueKind.String)
                        continue;
                    string name = item.GetString();
                    first ??= name;
                    if (name != "null")
                        return name;
                }
                return first;
            }

            return null;
        }

        /// <summary>Read <paramref name="name"/> as a string, or null unless it is one.</summary>
        private static string ReadString(JsonElement obj, string name)
            => obj.ValueKind == JsonValueKind.Object
               && obj.TryGetProperty(name, out var value)
               && value.ValueKind == JsonValueKind.String
                ? value.GetString()
                : null;

        private static bool TryGetObject(JsonElement obj, string name, out JsonElement value)
            => TryGetOfKind(obj, name, JsonValueKind.Object, out value);

        private static bool TryGetArray(JsonElement obj, string name, out JsonElement value)
            => TryGetOfKind(obj, name, JsonValueKind.Array, out value);

        /// <summary>
        /// <see cref="JsonElement.TryGetProperty(string, out JsonElement)"/> is
        /// itself only safe on an object — it throws on any other kind — so the
        /// container is checked before the member.
        /// </summary>
        private static bool TryGetOfKind(JsonElement obj, string name, JsonValueKind kind, out JsonElement value)
        {
            value = default;
            if (obj.ValueKind != JsonValueKind.Object
                || !obj.TryGetProperty(name, out var found)
                || found.ValueKind != kind)
            {
                return false;
            }

            value = found;
            return true;
        }
    }
}
