// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public sealed class Glm5NextReasoningEffortTests
{
    // The effort preamble and generation boundary from the published glm5next
    // template, with a minimal user turn so this exercises actual Jinja rendering.
    private const string Template = "[gMASK]<sop>"
        + "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
        + "<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}"
        + "{%- for m in messages -%}<|user|>{{ m.content }}{%- endfor -%}"
        + "{%- if add_generation_prompt -%}<|assistant|>{{- '<think>' -}}{%- endif -%}";

    [Theory]
    [InlineData(null, "Max")]
    [InlineData("low", "Low")]
    [InlineData(" HIGH ", "High")]
    [InlineData("medium", "Max")]
    [InlineData("invalid", "Max")]
    public void RequestedEffort_ReachesBothRenderers_WithoutClosingThinking(string? effort, string expected)
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Hi" } };
        string exact = $"[gMASK]<sop><|system|>Reasoning Effort: {expected}<|user|>Hi<|assistant|><think>";
        foreach (bool thinking in new[] { false, true })
        {
            Assert.Equal(exact, ChatTemplate.RenderFromGgufTemplate(Template, messages,
                architecture: "glm5next", enableThinking: thinking, reasoningEffort: effort));
            Assert.Equal(exact, ChatTemplate.RenderFromGgufTemplate(null, messages,
                architecture: "glm5next", enableThinking: thinking, reasoningEffort: effort));
        }
    }

    [Fact]
    public void ParsedLowEffort_IsRendered_AndParticipatesInSharedPrefixIdentity()
    {
        using var document = JsonDocument.Parse("{\"reasoning_effort\":\"low\"}");
        Assert.True(ReasoningEffortParser.TryParse(document.RootElement, out string? effort, out string? error), error);
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "2+2?" } };
        Assert.Contains("Reasoning Effort: Low", ChatTemplate.RenderFromGgufTemplate(
            Template, messages, architecture: "glm5next", reasoningEffort: effort));
        Assert.True(ChatProtocolRegistry.For("glm5next")!.RendersReasoningEffort);
    }

    [Theory]
    [InlineData("", "")]
    [InlineData("hIGH", "High")]
    [InlineData(" low", " low")]
    public void PublishedTemplateCapitalizeFilter_PreservesWhitespace(string value, string expected)
    {
        Assert.Equal(expected, new Jinja2Template("{{ value | capitalize }}")
            .Render(new Dictionary<string, object> { ["value"] = value }));
    }
}
