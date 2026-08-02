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
// Unit tests for Qwen 3 / 3.5 tool-call parsing. Qwen 3 emits a JSON object
// inside <tool_call>...</tool_call>; Qwen 3.5 emits the XML-ish
// <function=NAME><parameter=KEY>VALUE</parameter></function> body instead.
// Both must reach the OpenAI `tool_calls` surface — a body the parser cannot
// read is silently dropped along with the whole turn (the text has already been
// consumed as a tool call), which is what made every Qwen 3.5 function-call
// request come back empty with finish_reason "stop".

using System.Collections.Generic;

namespace InferenceWeb.Tests;

public class Qwen35ToolCallTests
{
    private static List<ToolFunction> WeatherTool() => new()
    {
        new ToolFunction
        {
            Name = "get_weather",
            Description = "Get the current weather for a city",
            Parameters = new Dictionary<string, ToolParameter>
            {
                ["city"] = new ToolParameter { Type = "string", Description = "City name" },
                ["units"] = new ToolParameter { Type = "string", Description = "c or f" },
            },
            Required = new List<string> { "city" },
        },
    };

    private static IOutputParser NewParser()
    {
        var parser = OutputParserFactory.Create("qwen35moe");
        parser.Init(enableThinking: false, tools: WeatherTool());
        return parser;
    }

    /// <summary>Feed the text in one go and return the accumulated parse.</summary>
    private static ParsedOutput ParseAll(string text)
    {
        var parser = NewParser();
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
    public void ParsesXmlStyleToolCall()
    {
        var parsed = ParseAll(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n" +
            "<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>");

        Assert.NotNull(parsed.ToolCalls);
        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal("c", call.Arguments["units"]);
    }

    [Fact]
    public void ParsesJsonStyleToolCall()
    {
        var parsed = ParseAll(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"units\": \"c\"}}\n</tool_call>");

        Assert.NotNull(parsed.ToolCalls);
        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
    }

    [Fact]
    public void XmlParameterValuesKeepTheirJsonTypes()
    {
        var parsed = ParseAll(
            "<tool_call>\n<function=book_flight>\n<parameter=passengers>\n3\n</parameter>\n" +
            "<parameter=refundable>\ntrue\n</parameter>\n<parameter=seats>\n[\"12A\",\"12B\"]\n</parameter>\n" +
            "<parameter=depart>\n2026-08-01\n</parameter>\n</function>\n</tool_call>");

        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("book_flight", call.Name);
        Assert.Equal(3L, call.Arguments["passengers"]);
        Assert.Equal(true, call.Arguments["refundable"]);
        Assert.IsAssignableFrom<List<object>>(call.Arguments["seats"]);
        // A date is not JSON — it must survive as text rather than becoming 2026 - 8 - 1.
        Assert.Equal("2026-08-01", call.Arguments["depart"]);
    }

    [Fact]
    public void XmlToolCallArrivesWhenStreamedInFragments()
    {
        var parser = NewParser();
        string[] chunks =
        {
            "<tool_", "call>\n<function=get_", "weather>\n<parameter=city>\nPar", "is\n</parameter>\n",
            "<parameter=units>\nc\n</parameter>\n</func", "tion>\n</tool_call>",
        };
        var calls = new List<ToolCall>();
        foreach (var chunk in chunks)
        {
            var p = parser.Add(chunk, false);
            if (p.ToolCalls != null) calls.AddRange(p.ToolCalls);
        }
        var last = parser.Add("", true);
        if (last.ToolCalls != null) calls.AddRange(last.ToolCalls);

        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal("c", call.Arguments["units"]);
    }

    [Fact]
    public void TextAroundTheToolCallIsStillContent()
    {
        var parsed = ParseAll(
            "Let me check that.<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n" +
            "</parameter>\n</function>\n</tool_call>");

        Assert.Contains("Let me check that.", parsed.Content);
        Assert.Single(parsed.ToolCalls!);
    }
}
