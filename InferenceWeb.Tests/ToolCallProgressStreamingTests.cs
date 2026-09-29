// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the incremental tool-call progress signal. While a model writes a
/// <c>shell</c> call it produces nothing a UI can show — the whole command, heredoc
/// and all, is tool markup, deliberately withheld from the content stream — so the parser now
/// surfaces the call's body text as it arrives (<see cref="ParsedOutput.ToolCallText"/>)
/// and the tool's name once known (<see cref="ParsedOutput.ToolCallName"/>), without
/// changing what <see cref="ParsedOutput.ToolCalls"/> delivers at completion.
/// </summary>
public class ToolCallProgressStreamingTests
{
    private static Gemma4OutputParser NewParser()
    {
        var parser = new Gemma4OutputParser();
        parser.Init(enableThinking: false, null);
        return parser;
    }

    private const string Call =
        "<|tool_call>call:shell{command:<|\"|>python3 solve.py<|\"|>}<tool_call|>";

    [Fact]
    public void TheCallBody_StreamsAsProgressText_WhileTheCallIsOpen()
    {
        var parser = NewParser();

        // The opening tag plus the start of the body, streamed in small pieces the
        // way decoding actually delivers them.
        var progress = new System.Text.StringBuilder();
        string name = null;
        foreach (string piece in new[] { "<|tool_", "call>call:sh", "ell{command:<|\"|>pyt", "hon3 sol" })
        {
            ParsedOutput delta = parser.Add(piece, false);
            progress.Append(delta.ToolCallText);
            name ??= delta.ToolCallName;
            Assert.Empty(delta.Content);            // the body must never leak as content
            Assert.Null(delta.ToolCalls);           // and the call is not complete yet
        }

        Assert.Equal("shell", name);
        Assert.StartsWith("call:shell{command:", progress.ToString(), StringComparison.Ordinal);
        Assert.Contains("python3 sol", progress.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void TheCompletedCall_StillParses_AndTheProgressTextIsTheWholeBody()
    {
        var parser = NewParser();
        var progress = new System.Text.StringBuilder();
        List<ToolCall> calls = null;

        // Two-character pieces: every hold-back boundary gets exercised.
        for (int i = 0; i < Call.Length; i += 2)
        {
            ParsedOutput delta = parser.Add(Call.Substring(i, Math.Min(2, Call.Length - i)), false);
            progress.Append(delta.ToolCallText);
            if (delta.ToolCalls != null)
                calls = delta.ToolCalls;
        }
        ParsedOutput last = parser.Add(string.Empty, true);
        progress.Append(last.ToolCallText);
        calls ??= last.ToolCalls;

        ToolCall call = Assert.Single(calls);
        Assert.Equal("shell", call.Name);
        Assert.Equal("python3 solve.py", call.Arguments["command"]?.ToString());

        // The streamed progress adds up to exactly the call body, no more, no less.
        Assert.Equal(
            "call:shell{command:<|\"|>python3 solve.py<|\"|>}",
            progress.ToString());
    }

    [Fact]
    public void ASingleAddWithTheWholeCall_ReportsBodyAndCallTogether()
    {
        var parser = NewParser();

        ParsedOutput delta = parser.Add("before " + Call, true);

        Assert.Equal("before", delta.Content.Trim());
        Assert.Single(delta.ToolCalls);
        // Batch consumers get the body too; they are free to ignore it.
        Assert.Contains("python3 solve.py", delta.ToolCallText, StringComparison.Ordinal);
    }

    [Fact]
    public void OrdinaryContent_CarriesNoProgressSignal()
    {
        var parser = NewParser();

        ParsedOutput delta = parser.Add("Just a plain answer.", false);

        Assert.Empty(delta.ToolCallText);
        Assert.Null(delta.ToolCallName);
    }

    // ---- ChatML-family parser ----------------------------------------------

    private static Qwen35OutputParser NewQwenParser(bool thinking = false)
    {
        var parser = new Qwen35OutputParser();
        parser.Init(thinking, null);
        return parser;
    }

    private const string QwenJsonCall =
        "<tool_call>{\"name\": \"shell\", \"arguments\": {\"command\": \"python3 solve.py\"}}</tool_call>";

    [Fact]
    public void QwenJsonCallBody_StreamsAsProgress_AndStillParses()
    {
        var parser = NewQwenParser();
        var progress = new System.Text.StringBuilder();
        string name = null;
        List<ToolCall> calls = null;

        for (int i = 0; i < QwenJsonCall.Length; i += 3)
        {
            ParsedOutput delta = parser.Add(QwenJsonCall.Substring(i, Math.Min(3, QwenJsonCall.Length - i)), false);
            progress.Append(delta.ToolCallText);
            name ??= delta.ToolCallName;
            Assert.Empty(delta.Content);
            if (delta.ToolCalls != null) calls = delta.ToolCalls;
        }
        ParsedOutput last = parser.Add(string.Empty, true);
        progress.Append(last.ToolCallText);
        calls ??= last.ToolCalls;

        Assert.Equal("shell", name);
        ToolCall call = Assert.Single(calls);
        Assert.Equal("shell", call.Name);
        Assert.Equal("python3 solve.py", call.Arguments["command"]?.ToString());
        Assert.Equal(
            "{\"name\": \"shell\", \"arguments\": {\"command\": \"python3 solve.py\"}}",
            progress.ToString());
    }

    [Fact]
    public void QwenXmlCallBody_AlsoStreams_AndNamesTheTool()
    {
        // Qwen 3.5+ sometimes emits the XML-ish body instead of JSON.
        const string xml =
            "<tool_call><function=shell>\n<parameter=command>\npython3 solve.py\n</parameter>\n" +
            "<parameter=timeout_ms>\n60000\n</parameter>\n</function></tool_call>";

        var parser = NewQwenParser();
        var progress = new System.Text.StringBuilder();
        string name = null;
        List<ToolCall> calls = null;

        for (int i = 0; i < xml.Length; i += 5)
        {
            ParsedOutput delta = parser.Add(xml.Substring(i, Math.Min(5, xml.Length - i)), false);
            progress.Append(delta.ToolCallText);
            name ??= delta.ToolCallName;
            if (delta.ToolCalls != null) calls = delta.ToolCalls;
        }
        calls ??= parser.Add(string.Empty, true).ToolCalls;

        Assert.Equal("shell", name);
        ToolCall call = Assert.Single(calls);
        Assert.Equal("60000", call.Arguments["timeout_ms"]?.ToString());
        Assert.Contains("<function=shell>", progress.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void QwenThinkingThenCall_ProgressStartsOnlyInsideTheCall()
    {
        var parser = NewQwenParser(thinking: true);
        string raw = "<think>let me compute</think>Sure.\n" + QwenJsonCall;

        var progress = new System.Text.StringBuilder();
        var thinking = new System.Text.StringBuilder();
        var content = new System.Text.StringBuilder();
        for (int i = 0; i < raw.Length; i += 4)
        {
            ParsedOutput delta = parser.Add(raw.Substring(i, Math.Min(4, raw.Length - i)), false);
            progress.Append(delta.ToolCallText);
            thinking.Append(delta.Thinking);
            content.Append(delta.Content);
        }
        parser.Add(string.Empty, true);

        Assert.Contains("let me compute", thinking.ToString(), StringComparison.Ordinal);
        Assert.Contains("Sure.", content.ToString(), StringComparison.Ordinal);
        Assert.StartsWith("{\"name\"", progress.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain("Sure.", progress.ToString(), StringComparison.Ordinal);
    }

    // ---- Harmony (gpt-oss) --------------------------------------------------

    [Fact]
    public void HarmonyToolCallBody_StreamsAsProgress_WithTheToolName()
    {
        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: true, null);

        const string raw =
            "<|channel|>commentary to=functions.shell<|message|>" +
            "{\"command\": \"python3 solve.py\"}<|call|>";

        var progress = new System.Text.StringBuilder();
        string name = null;
        List<ToolCall> calls = null;
        for (int i = 0; i < raw.Length; i += 4)
        {
            ParsedOutput delta = parser.Add(raw.Substring(i, Math.Min(4, raw.Length - i)), false);
            progress.Append(delta.ToolCallText);
            name ??= delta.ToolCallName;
            if (delta.ToolCalls != null) calls = delta.ToolCalls;
        }
        calls ??= parser.Add(string.Empty, true).ToolCalls;

        Assert.Equal("shell", name);
        ToolCall call = Assert.Single(calls);
        Assert.Equal("shell", call.Name);
        Assert.Equal("python3 solve.py", call.Arguments["command"]?.ToString());
        Assert.Contains("\"command\"", progress.ToString(), StringComparison.Ordinal);
    }

    // ---- Muse Glimmer -------------------------------------------------------

    [Fact]
    public void MuseGlimmerToolCallBody_StreamsAsProgress()
    {
        var parser = new MuseGlimmerOutputParser();
        parser.Init(enableThinking: true, null);

        const string raw =
            "<|start|>assistant to=shell<|message|>" +
            "<atem:function_calls><atem:invoke name=\"shell\">" +
            "<atem:parameter name=\"command\">python3 solve.py</atem:parameter>" +
            "</atem:invoke></atem:function_calls><|eom|>";

        var progress = new System.Text.StringBuilder();
        string name = null;
        List<ToolCall> calls = null;
        for (int i = 0; i < raw.Length; i += 6)
        {
            ParsedOutput delta = parser.Add(raw.Substring(i, Math.Min(6, raw.Length - i)), false);
            progress.Append(delta.ToolCallText);
            name ??= delta.ToolCallName;
            if (delta.ToolCalls != null) calls = delta.ToolCalls;
        }
        calls ??= parser.Add(string.Empty, true).ToolCalls;

        Assert.Equal("shell", name);
        Assert.NotNull(calls);
        Assert.Contains("atem:invoke", progress.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void TwoCallsInOneRound_DoNotBleedProgressIntoEachOther()
    {
        var parser = NewParser();
        var progress = new System.Text.StringBuilder();
        var calls = new List<ToolCall>();

        string two = Call + Call;
        for (int i = 0; i < two.Length; i += 7)
        {
            ParsedOutput delta = parser.Add(two.Substring(i, Math.Min(7, two.Length - i)), false);
            progress.Append(delta.ToolCallText);
            if (delta.ToolCalls != null) calls.AddRange(delta.ToolCalls);
        }
        ParsedOutput last = parser.Add(string.Empty, true);
        progress.Append(last.ToolCallText);
        if (last.ToolCalls != null) calls.AddRange(last.ToolCalls);

        Assert.Equal(2, calls.Count);
        string body = "call:shell{command:<|\"|>python3 solve.py<|\"|>}";
        Assert.Equal(body + body, progress.ToString());
    }
}
