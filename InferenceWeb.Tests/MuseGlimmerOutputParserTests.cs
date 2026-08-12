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
// Unit tests for the Muse-Glimmer streaming output parser. Muse-Glimmer frames
// every assistant message as
//   <|start|>assistant[ to=RECIPIENT]<|message|>BODY<|eom|>   (more this turn)
//   <|start|>assistant<|message|>BODY<|eot|>                  (turn over)
// with the reasoning channel addressed "to=self" and tool calls addressed to the
// tool, carrying an ATEM XML body. Before this parser existed the architecture
// fell through to PassthroughOutputParser, so every reply began with the literal
// text " to=self<|message|>" and streamed the whole chain of thought as content.
// These run without a model.

using System.Collections.Generic;
using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class MuseGlimmerOutputParserTests
{
    private static IOutputParser NewParser()
    {
        var p = OutputParserFactory.Create("muse-glimmer");
        p.Init(enableThinking: true, tools: new List<ToolFunction>());
        return p;
    }

    /// <summary>Feed the text one character at a time to exercise the streaming holdback.</summary>
    private static (string content, string thinking, List<ToolCall> calls) StreamCharByChar(string text)
    {
        var p = NewParser();
        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var calls = new List<ToolCall>();
        for (int i = 0; i < text.Length; i++)
        {
            var r = p.Add(text[i].ToString(), done: i == text.Length - 1);
            content.Append(r.Content);
            thinking.Append(r.Thinking);
            if (r.ToolCalls != null) calls.AddRange(r.ToolCalls);
        }
        return (content.ToString(), thinking.ToString(), calls);
    }

    private static (string content, string thinking, List<ToolCall> calls) ParseWhole(string text)
    {
        var p = NewParser();
        var r = p.Add(text, done: true);
        return (r.Content, r.Thinking, r.ToolCalls ?? new List<ToolCall>());
    }

    [Fact]
    public void FactoryReturnsMuseGlimmerParser()
    {
        Assert.IsType<MuseGlimmerOutputParser>(OutputParserFactory.Create("muse-glimmer"));
        Assert.True(OutputParserFactory.IsAlwaysRequired("muse-glimmer"));
    }

    [Fact]
    public void SelfChannelBecomesThinking_AnswerBecomesContent()
    {
        // The prompt ends at "<|start|>assistant", so generation resumes mid-header.
        const string gen =
            " to=self<|message|>Rayleigh scattering. Keep it to four sentences.<|eom|>" +
            "<|start|>assistant<|message|>The sky is blue because air scatters short wavelengths.<|eot|>";

        var (content, thinking, calls) = ParseWhole(gen);
        Assert.Equal("The sky is blue because air scatters short wavelengths.", content);
        Assert.Equal("Rayleigh scattering. Keep it to four sentences.", thinking);
        Assert.Empty(calls);
        // No framing leaks into either stream.
        Assert.DoesNotContain("<|", content);
        Assert.DoesNotContain("to=self", content);
    }

    [Fact]
    public void StreamingCharByChar_MatchesWholeParse()
    {
        const string gen =
            " to=self<|message|>think a<|eom|><|start|>assistant<|message|>answer b<|eot|>";
        var whole = ParseWhole(gen);
        var streamed = StreamCharByChar(gen);
        Assert.Equal(whole.content, streamed.content);
        Assert.Equal(whole.thinking, streamed.thinking);
        Assert.Equal("answer b", streamed.content);
        Assert.Equal("think a", streamed.thinking);
    }

    [Fact]
    public void NoReasoningChannel_AnswerStillParsed()
    {
        var (content, thinking, _) = ParseWhole("<|message|>Direct answer.<|eot|>");
        Assert.Equal("Direct answer.", content);
        Assert.Equal("", thinking);
    }

    [Fact]
    public void TruncatedGeneration_FlushesPartialContent()
    {
        // max_tokens hit mid-answer: no closing <|eot|> is ever emitted.
        var (content, thinking, _) = ParseWhole(
            " to=self<|message|>reasoning<|eom|><|start|>assistant<|message|>partial ans");
        Assert.Equal("partial ans", content);
        Assert.Equal("reasoning", thinking);
    }

    [Fact]
    public void AtemToolCall_ParsedWithTypedArguments()
    {
        const string gen =
            " to=self<|message|>Need the weather.<|eom|>" +
            "<|start|>assistant to=weather.get<|message|><atem:function_calls>\n" +
            "<atem:invoke name=\"weather.get\">\n" +
            "<atem:parameter name=\"location\">Paris</atem:parameter>\n" +
            "<atem:parameter name=\"days\">3</atem:parameter>\n" +
            "<atem:parameter name=\"metric\">true</atem:parameter>\n" +
            "<atem:parameter name=\"tags\">[\"a\", \"b\"]</atem:parameter>\n" +
            "</atem:invoke>\n</atem:function_calls><|eom|>";

        var (content, thinking, calls) = ParseWhole(gen);
        Assert.Equal("Need the weather.", thinking);
        // The ATEM block is a tool call, not prose: it must not reach the user.
        Assert.Equal("", content);
        var call = Assert.Single(calls);
        Assert.Equal("weather.get", call.Name);
        Assert.Equal("Paris", call.Arguments["location"]);
        // Integers arrive as long, matching every other parser's JSON decoding.
        Assert.Equal(3L, call.Arguments["days"]);
        Assert.Equal(true, call.Arguments["metric"]);
        Assert.Equal(new List<object> { "a", "b" }, call.Arguments["tags"]);
    }

    [Fact]
    public void ToUserRecipient_IsContentNotToolCall()
    {
        var (content, _, calls) = ParseWhole("<|start|>assistant to=user<|message|>hello<|eot|>");
        Assert.Equal("hello", content);
        Assert.Empty(calls);
    }
}
