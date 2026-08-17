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
using System.Text.Json;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="ToolFunctionParser"/> reads a tool's <c>parameters</c>, which is a
/// JSON Schema document — and JSON Schema puts non-strings in places a naive
/// reader assumes are strings. Every accessor used to be an unguarded
/// <c>GetString()</c>, so an <c>"enum": [1, 2, 3]</c> or a nullable field's
/// <c>"type": ["string", "null"]</c> threw <see cref="InvalidOperationException"/>
/// out of the parser and failed the entire chat request with a 500, even though
/// the spec was valid (https://github.com/zhongkaifu/TensorSharp/issues/142).
/// These tests pin every kind that is legal in a tool spec, on all three
/// protocol entry points.
/// <para>
/// The sibling <see cref="ToolFunctionParseTests"/> covers the unrelated public
/// <c>TensorSharp.Runtime.ToolFunction.ParseList</c>, which is the CLI's
/// <c>--tools</c> path.
/// </para>
/// </summary>
public class ToolFunctionParserTests
{
    /// <summary>
    /// The payload from issue #142, reduced: an OpenAI tool spec whose schema is
    /// entirely legal but uses an integer enum, a boolean enum, a union type and
    /// a no-argument tool. Every one of these threw before the fix.
    /// </summary>
    private const string HarnessToolsBody = """
    {
      "model": "qwen35",
      "messages": [ { "role": "user", "content": "hi" } ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "set_verbosity",
            "description": "Set the log verbosity.",
            "parameters": {
              "type": "object",
              "properties": {
                "level":   { "type": "integer", "description": "0-3.", "enum": [0, 1, 2, 3] },
                "scope":   { "type": ["string", "null"], "description": "Subsystem, or null for all." },
                "persist": { "type": "boolean", "enum": [true, false] }
              },
              "required": ["level"]
            }
          }
        },
        {
          "type": "function",
          "function": { "name": "now", "description": "Current time.", "parameters": null }
        }
      ]
    }
    """;

    private static JsonElement Body(string json) => JsonDocument.Parse(json).RootElement;

    // ---- issue #142 regression --------------------------------------------

    [Fact]
    public void ParseOpenAI_HarnessToolSpec_ParsesInsteadOfThrowing()
    {
        var tools = ToolFunctionParser.ParseOpenAI(Body(HarnessToolsBody));

        Assert.Equal(2, tools.Count);

        var verbosity = tools[0];
        Assert.Equal("set_verbosity", verbosity.Name);
        Assert.Equal("Set the log verbosity.", verbosity.Description);
        Assert.Equal(["level"], verbosity.Required);

        // An integer enum keeps its JSON spelling, so the schema handed to the
        // model still says 0-3 rather than "0"-"3" or nothing at all.
        Assert.Equal("integer", verbosity.Parameters["level"].Type);
        Assert.Equal(["0", "1", "2", "3"], verbosity.Parameters["level"].Enum);

        // "type": ["string", "null"] collapses to the real type; the union's
        // nullability is already expressed by "level" being the only required one.
        Assert.Equal("string", verbosity.Parameters["scope"].Type);
        Assert.Equal("Subsystem, or null for all.", verbosity.Parameters["scope"].Description);

        // JSON casing, not .NET's — JsonElement.ToString() would render "True".
        Assert.Equal("boolean", verbosity.Parameters["persist"].Type);
        Assert.Equal(["true", "false"], verbosity.Parameters["persist"].Enum);

        // "parameters": null is how a no-argument tool is usually written.
        Assert.Equal("now", tools[1].Name);
        Assert.Empty(tools[1].Parameters);
        Assert.Empty(tools[1].Required);
    }

    [Theory]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string","enum":[1,"b",true,null,{"k":1},[2]]}}}}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":["null","integer"]}}}}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":true}}}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":{"const":"x"}}}}}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","required":["a",7,null]}}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"description":42}}}}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":7,"description":[1]}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":[]}}]}""")]
    [InlineData("""{"tools":[{"type":"function","function":{"name":"f","parameters":{"properties":"nope","required":"nope"}}}]}""")]
    [InlineData("""{"tools":["a string",42,null,[],{"type":"function"}]}""")]
    [InlineData("""{"tools":[{"type":7,"function":{"name":"f"}}]}""")]
    [InlineData("""{"tools":"not an array"}""")]
    [InlineData("""[1,2,3]""")]
    public void EveryProtocol_TolerantOfAnyJsonKindInAToolSpec(string json)
    {
        // No shape of syntactically valid JSON may escape as an exception: the
        // parsers run before the request has produced any output, so a throw
        // here is a 500 on a request the server could otherwise have served.
        Assert.Null(Record.Exception(() => ToolFunctionParser.ParseOpenAI(Body(json))));
        Assert.Null(Record.Exception(() => ToolFunctionParser.ParseOpenAIResponses(Body(json))));
        Assert.Null(Record.Exception(() => ToolFunctionParser.ParseOllama(Body(json))));
    }

    // ---- enum value rendering ----------------------------------------------

    [Fact]
    public void ParseOpenAI_NonStringEnumValues_KeepTheirRawJsonText()
    {
        const string body = """
        {
          "tools": [{
            "type": "function",
            "function": {
              "name": "f",
              "parameters": {
                "type": "object",
                "properties": { "a": { "type": "string", "enum": ["s", 1, 2.5, true, false, null, {"k":1}, [2]] } }
              }
            }
          }]
        }
        """;

        var values = ToolFunctionParser.ParseOpenAI(Body(body))[0].Parameters["a"].Enum;

        // A string member is unquoted so it re-serializes as itself; everything
        // else is its literal JSON text, which is what the prompt renderers emit
        // back into the schema they show the model.
        Assert.Equal(["s", "1", "2.5", "true", "false", "null", """{"k":1}""", "[2]"], values);
    }

    // ---- shapes that already worked keep working ---------------------------

    [Fact]
    public void ParseOpenAI_PlainStringSchema_IsUnchanged()
    {
        const string body = """
        {
          "tools": [{
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "description": "Get the current weather in a given city.",
              "parameters": {
                "type": "object",
                "properties": {
                  "city": { "type": "string", "description": "The city name." },
                  "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                },
                "required": ["city"]
              }
            }
          }]
        }
        """;

        var tool = Assert.Single(ToolFunctionParser.ParseOpenAI(Body(body)));
        Assert.Equal("get_current_weather", tool.Name);
        Assert.Equal("Get the current weather in a given city.", tool.Description);
        Assert.Equal(2, tool.Parameters.Count);
        Assert.Equal("string", tool.Parameters["city"].Type);
        Assert.Equal("The city name.", tool.Parameters["city"].Description);
        Assert.Equal(["celsius", "fahrenheit"], tool.Parameters["unit"].Enum);
        Assert.Equal(["city"], tool.Required);
    }

    [Fact]
    public void ParseOllama_NestsUnderFunctionWithoutRequiringAType()
    {
        const string body = """
        {
          "tools": [{
            "function": {
              "name": "get_current_weather",
              "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" }, "days": { "type": "integer", "enum": [1, 7] } },
                "required": ["city"]
              }
            }
          }]
        }
        """;

        var tool = Assert.Single(ToolFunctionParser.ParseOllama(Body(body)));
        Assert.Equal("get_current_weather", tool.Name);
        Assert.Equal(["1", "7"], tool.Parameters["days"].Enum);
        Assert.Equal(["city"], tool.Required);
    }

    [Fact]
    public void ParseOpenAIResponses_FlatShapeStillParses()
    {
        const string body = """
        {
          "tools": [
            { "type": "web_search" },
            {
              "type": "function",
              "name": "get_weather",
              "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
              }
            }
          ]
        }
        """;

        // The built-in web_search tool is not ours to run and must not become a
        // nameless function declaration in the prompt.
        var tool = Assert.Single(ToolFunctionParser.ParseOpenAIResponses(Body(body)));
        Assert.Equal("get_weather", tool.Name);
        Assert.Equal("string", tool.Parameters["city"].Type);
    }

    // ---- nothing to parse ---------------------------------------------------

    [Theory]
    [InlineData("{}")]
    [InlineData("""{"tools":[]}""")]
    [InlineData("""{"tools":null}""")]
    [InlineData("""{"tools":[{"type":"web_search"}]}""")]
    public void NoUsableTools_ReturnsNullSoCallersShortCircuit(string json)
    {
        Assert.Null(ToolFunctionParser.ParseOpenAI(Body(json)));
        Assert.Null(ToolFunctionParser.ParseOpenAIResponses(Body(json)));
    }

    [Fact]
    public void ParseOpenAI_NonFunctionEntriesAreSkippedButFunctionsSurvive()
    {
        const string body = """
        {
          "tools": [
            { "type": "code_interpreter" },
            { "type": "function", "function": { "name": "kept" } },
            { "function": { "name": "also_kept" } }
          ]
        }
        """;

        var tools = ToolFunctionParser.ParseOpenAI(Body(body));
        Assert.Equal(["kept", "also_kept"], tools.ConvertAll(t => t.Name));
    }

    [Fact]
    public void ParseFunction_MissingStringFieldsBecomeEmptyNotNull()
    {
        // ToolFunction/ToolParameter declare these non-nullable, and the prompt
        // renderers call string.IsNullOrEmpty on them; a null would be a latent
        // NullReferenceException one refactor away.
        const string body = """
        {"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{}}}}}]}
        """;

        var tool = Assert.Single(ToolFunctionParser.ParseOpenAI(Body(body)));
        Assert.Equal(string.Empty, tool.Description);
        Assert.Equal("string", tool.Parameters["a"].Type);
        Assert.Equal(string.Empty, tool.Parameters["a"].Description);
        Assert.Empty(tool.Parameters["a"].Enum);
    }

    [Fact]
    public void ParseFunction_RequiredDropsEntriesThatCannotNameAProperty()
    {
        const string body = """
        {"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","required":["a",7,null,"b"]}}}]}
        """;

        Assert.Equal(["a", "b"], Assert.Single(ToolFunctionParser.ParseOpenAI(Body(body))).Required);
    }
}
