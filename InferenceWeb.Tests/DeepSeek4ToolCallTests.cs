// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Unit tests for DeepSeek V4 function calling. V4 does not use a JSON tool-call
// block: the prompt teaches a "DSML" markup and the model answers in it. Both
// halves are checked here against the wire format in the model's own chat
// template — the prompt must carry the tool schemas at all (it used to drop them
// entirely, so the model answered "I don't have access to a weather tool"), and
// the parser must turn the markup back into OpenAI tool calls.

using System.Collections.Generic;

namespace InferenceWeb.Tests;

public class DeepSeek4ToolCallTests
{
    private const string Dsml = "｜DSML｜";

    private static List<ToolFunction> WeatherTool() => new()
    {
        new ToolFunction
        {
            Name = "get_weather",
            Description = "Get the current weather for a city.",
            Parameters = new Dictionary<string, ToolParameter>
            {
                ["city"] = new ToolParameter { Type = "string", Description = "City name" },
                ["days"] = new ToolParameter { Type = "integer", Description = "Forecast days" },
            },
            Required = new List<string> { "city" },
        },
    };

    private static IOutputParser NewParser(bool thinking = false)
    {
        var parser = OutputParserFactory.Create("deepseek4");
        parser.Init(thinking, WeatherTool());
        return parser;
    }

    // ---------------- prompt side ----------------

    [Fact]
    public void PromptCarriesToolSchemas()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Weather in Paris?" } };
        string prompt = ChatTemplate.RenderDeepSeek4(messages, addGenerationPrompt: true,
                                                     enableThinking: false, tools: WeatherTool());

        Assert.Contains("## Tools", prompt);
        Assert.Contains("<" + Dsml + "tool_calls>", prompt);           // syntax the model must copy
        Assert.Contains("\"name\":\"get_weather\"", prompt);            // the schema itself
        Assert.Contains("\"city\"", prompt);
        Assert.Contains("You MUST strictly follow the above defined tool name", prompt);
        Assert.EndsWith("<｜Assistant｜></think>", prompt);
    }

    [Fact]
    public void PromptWithoutToolsIsUnchanged()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Hi" } };
        string withNull = ChatTemplate.RenderDeepSeek4(messages, true, false, null);
        string withEmpty = ChatTemplate.RenderDeepSeek4(messages, true, false, new List<ToolFunction>());

        Assert.Equal(withNull, withEmpty);
        Assert.DoesNotContain("## Tools", withNull);
    }

    [Fact]
    public void ToolsAttachAfterAnExistingSystemPrompt()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are terse." },
            new() { Role = "user", Content = "Weather in Paris?" },
        };
        string prompt = ChatTemplate.RenderDeepSeek4(messages, true, false, WeatherTool());

        int sys = prompt.IndexOf("You are terse.", System.StringComparison.Ordinal);
        int tools = prompt.IndexOf("## Tools", System.StringComparison.Ordinal);
        Assert.True(sys >= 0 && tools > sys, "tools must follow the system prompt");
        Assert.Contains("You are terse.\n\n## Tools", prompt);
    }

    [Fact]
    public void AssistantToolCallsAreRenderedBackAsDsml()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Weather in Paris?" },
            new()
            {
                Role = "assistant",
                Content = "",
                ToolCalls = new List<ToolCall>
                {
                    new()
                    {
                        Name = "get_weather",
                        Arguments = new Dictionary<string, object> { ["city"] = "Paris", ["days"] = 3L },
                    },
                },
            },
            new() { Role = "tool", Content = "18C, sunny" },
        };
        string prompt = ChatTemplate.RenderDeepSeek4(messages, true, false, WeatherTool());

        Assert.Contains("<" + Dsml + "invoke name=\"get_weather\">", prompt);
        Assert.Contains("<" + Dsml + "parameter name=\"city\" string=\"true\">Paris</" + Dsml + "parameter>", prompt);
        Assert.Contains("<" + Dsml + "parameter name=\"days\" string=\"false\">3</" + Dsml + "parameter>", prompt);
        Assert.Contains("<tool_result>18C, sunny</tool_result>", prompt);
    }

    // ---------------- parse side ----------------

    private static ParsedOutput ParseAll(IOutputParser parser, string text)
    {
        var first = parser.Add(text, false);
        var final = parser.Add("", true);
        return new ParsedOutput
        {
            Content = (first.Content ?? "") + (final.Content ?? ""),
            Thinking = (first.Thinking ?? "") + (final.Thinking ?? ""),
            ToolCalls = final.ToolCalls ?? first.ToolCalls,
        };
    }

    [Fact]
    public void ParsesDsmlToolCall()
    {
        var parsed = ParseAll(NewParser(),
            "<" + Dsml + "tool_calls>\n" +
            "<" + Dsml + "invoke name=\"get_weather\">\n" +
            "<" + Dsml + "parameter name=\"city\" string=\"true\">Paris</" + Dsml + "parameter>\n" +
            "<" + Dsml + "parameter name=\"days\" string=\"false\">3</" + Dsml + "parameter>\n" +
            "</" + Dsml + "invoke>\n" +
            "</" + Dsml + "tool_calls>");

        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal(3L, call.Arguments["days"]);
        Assert.Equal("", parsed.Content);
    }

    [Fact]
    public void ParsesParallelInvokes()
    {
        var parsed = ParseAll(NewParser(),
            "<" + Dsml + "tool_calls>\n" +
            "<" + Dsml + "invoke name=\"get_weather\">\n" +
            "<" + Dsml + "parameter name=\"city\" string=\"true\">Paris</" + Dsml + "parameter>\n" +
            "</" + Dsml + "invoke>\n" +
            "<" + Dsml + "invoke name=\"get_weather\">\n" +
            "<" + Dsml + "parameter name=\"city\" string=\"true\">Berlin</" + Dsml + "parameter>\n" +
            "</" + Dsml + "invoke>\n" +
            "</" + Dsml + "tool_calls>");

        Assert.Equal(2, parsed.ToolCalls!.Count);
        Assert.Equal("Paris", parsed.ToolCalls[0].Arguments["city"]);
        Assert.Equal("Berlin", parsed.ToolCalls[1].Arguments["city"]);
        Assert.Equal(0, parsed.ToolCalls[0].Index);
        Assert.Equal(1, parsed.ToolCalls[1].Index);
    }

    [Fact]
    public void KeepsTextBeforeTheCallAsContent()
    {
        var parsed = ParseAll(NewParser(),
            "Let me look that up.\n\n<" + Dsml + "tool_calls>\n" +
            "<" + Dsml + "invoke name=\"get_weather\">\n" +
            "<" + Dsml + "parameter name=\"city\" string=\"true\">Paris</" + Dsml + "parameter>\n" +
            "</" + Dsml + "invoke>\n</" + Dsml + "tool_calls>");

        Assert.Contains("Let me look that up.", parsed.Content);
        Assert.DoesNotContain(Dsml, parsed.Content);
        Assert.Single(parsed.ToolCalls!);
    }

    [Fact]
    public void SeparatesReasoningFromContent()
    {
        var parser = NewParser(thinking: true);
        var parsed = ParseAll(parser, "the user wants weather</think>It is sunny in Paris.");

        Assert.Equal("the user wants weather", parsed.Thinking);
        Assert.Equal("It is sunny in Paris.", parsed.Content);
        Assert.Null(parsed.ToolCalls);
    }

    [Fact]
    public void ParsesToolCallStreamedInFragments()
    {
        var parser = NewParser();
        string[] chunks =
        {
            "<" + Dsml.Substring(0, 3), Dsml.Substring(3) + "tool_", "calls>\n<" + Dsml + "invoke name=\"get_",
            "weather\">\n<" + Dsml + "parameter name=\"city\" string=\"true\">Par", "is</" + Dsml + "parameter>\n",
            "</" + Dsml + "invoke>\n</" + Dsml + "tool_calls>",
        };
        var calls = new List<ToolCall>();
        var content = new System.Text.StringBuilder();
        foreach (var chunk in chunks)
        {
            var p = parser.Add(chunk, false);
            if (p.ToolCalls != null) calls.AddRange(p.ToolCalls);
            content.Append(p.Content);
        }
        var last = parser.Add("", true);
        if (last.ToolCalls != null) calls.AddRange(last.ToolCalls);
        content.Append(last.Content);

        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        // No fragment of the markup may leak into the visible answer.
        Assert.DoesNotContain("DSML", content.ToString());
    }

    [Fact]
    public void EmitsCallWhenGenerationStopsInsideTheBlock()
    {
        // EOS right after </invoke>, before the closing </｜DSML｜tool_calls>.
        var parsed = ParseAll(NewParser(),
            "<" + Dsml + "tool_calls>\n" +
            "<" + Dsml + "invoke name=\"get_weather\">\n" +
            "<" + Dsml + "parameter name=\"city\" string=\"true\">Paris</" + Dsml + "parameter>\n" +
            "</" + Dsml + "invoke>");

        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("Paris", call.Arguments["city"]);
    }
}
