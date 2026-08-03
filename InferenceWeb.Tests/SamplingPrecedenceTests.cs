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
using System.IO;
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Covers github.com/zhongkaifu/TensorSharp/issues/113: an operator's config
/// file pinned temperature / top-k / top-p / min-p / repeat-penalty /
/// presence-penalty, but the values the server actually sampled with came from
/// the client. VS Code Copilot Chat hardcodes <c>temperature</c> and
/// <c>top_p</c> into every <c>/v1/chat/completions</c> body, so under the old
/// request-always-wins rule the operator's configuration could never apply. The
/// same request also showed <c>maxTokens=200</c> — a hard-coded adapter default
/// that ignored <c>--max-tokens</c> entirely.
///
/// The two rules under test:
///   - a parameter the operator PINNED wins over the request (default
///     <c>--sampling-precedence config</c>); a parameter they left alone still
///     comes from the request,
///   - <c>--max-tokens</c> is the fallback for every protocol and caps requests
///     that ask for more, while a smaller request limit is honoured.
/// </summary>
public class SamplingPrecedenceTests : IDisposable
{
    private readonly string _baseDir;
    private readonly EnvScope _env = new();

    public SamplingPrecedenceTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-sampling-precedence-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
        // The builder reads these as pinning fallbacks; clear anything the host
        // environment might have set so each case starts from a known state.
        foreach (string name in new[]
        {
            "TENSORSHARP_TEMPERATURE", "TENSORSHARP_TOP_K", "TENSORSHARP_TOP_P", "TENSORSHARP_MIN_P",
            "TENSORSHARP_REPEAT_PENALTY", "TENSORSHARP_REPEAT_LAST_N", "TENSORSHARP_PRESENCE_PENALTY",
            "TENSORSHARP_FREQUENCY_PENALTY", "TENSORSHARP_SEED", "TENSORSHARP_SAMPLING_PRECEDENCE",
            "MAX_TOKENS",
        })
        {
            _env.Set(name, null);
        }
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>The exact config file from the issue, as CLI tokens.</summary>
    private static string[] IssueConfigArgs => new[]
    {
        "--max-tokens", "2048",
        "--temperature", "0.6",
        "--top-k", "40",
        "--top-p", "0.95",
        "--min-p", "0.05",
        "--repeat-penalty", "1.15",
        "--presence-penalty", "0.3",
    };

    /// <summary>The body VS Code Copilot Chat sends (temperature + top_p hardcoded).</summary>
    private const string CopilotBody = """{"temperature": 0.1, "top_p": 1, "presence_penalty": 0}""";

    [Fact]
    public void Issue113_CopilotBody_KeepsEveryPinnedConfigValue()
    {
        var options = ServerOptionsBuilder.Build(IssueConfigArgs, _baseDir);
        using var doc = JsonDocument.Parse(CopilotBody);

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(0.6f, effective.Temperature);
        Assert.Equal(40, effective.TopK);
        Assert.Equal(0.95f, effective.TopP);
        Assert.Equal(0.05f, effective.MinP);
        Assert.Equal(1.15f, effective.RepetitionPenalty);
        Assert.Equal(0.3f, effective.PresencePenalty);
    }

    [Fact]
    public void Issue113_MaxTokensComesFromConfigNotTheHardCoded200()
    {
        var options = ServerOptionsBuilder.Build(IssueConfigArgs, _baseDir);

        // Copilot's request carried no max_tokens; the old adapters fell back to 200.
        Assert.Equal(2048, options.ResolveMaxTokens(null));
    }

    [Fact]
    public void UnpinnedParameter_StillTakesTheRequestValue()
    {
        // Only temperature is pinned; top_p is not, so the client keeps control of it.
        var options = ServerOptionsBuilder.Build(new[] { "--temperature", "0.6" }, _baseDir);
        using var doc = JsonDocument.Parse("""{"temperature": 0.1, "top_p": 0.5, "frequency_penalty": 0.7}""");

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(0.6f, effective.Temperature);
        Assert.Equal(0.5f, effective.TopP);
        Assert.Equal(0.7f, effective.FrequencyPenalty);
    }

    [Fact]
    public void NoServerFlags_RequestKeepsFullControl()
    {
        // Regression guard: a server started without sampling flags must behave
        // exactly as before this change.
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);
        using var doc = JsonDocument.Parse("""{"temperature": 0.1, "top_p": 1}""");

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(0.1f, effective.Temperature);
        Assert.Equal(1f, effective.TopP);
        Assert.False(options.SamplingDefaults.EnforcesAnyField);
    }

    [Fact]
    public void SamplingPrecedenceRequest_RestoresClientWins()
    {
        var args = new[] { "--temperature", "0.6", "--sampling-precedence", "request" };
        var options = ServerOptionsBuilder.Build(args, _baseDir);
        using var doc = JsonDocument.Parse(CopilotBody);

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(SamplingPrecedence.Request, options.SamplingDefaults.Precedence);
        Assert.Equal(0.1f, effective.Temperature);
    }

    [Fact]
    public void SamplingPrecedence_EnvVarIsHonoured_AndCliWinsOverIt()
    {
        _env.Set("TENSORSHARP_SAMPLING_PRECEDENCE", "request");
        Assert.Equal(
            SamplingPrecedence.Request,
            ServerOptionsBuilder.Build(new[] { "--temperature", "0.6" }, _baseDir).SamplingDefaults.Precedence);

        Assert.Equal(
            SamplingPrecedence.Config,
            ServerOptionsBuilder.Build(
                new[] { "--temperature", "0.6", "--sampling-precedence", "config" }, _baseDir).SamplingDefaults.Precedence);
    }

    [Fact]
    public void SamplingPrecedence_InvalidValue_ThrowsWithGuidance()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => ServerOptionsBuilder.Build(new[] { "--sampling-precedence", "whatever" }, _baseDir));
        Assert.Contains("--sampling-precedence", ex.Message);
        Assert.Contains("request", ex.Message);
    }

    [Fact]
    public void EnvVarPinnedValue_AlsoOverridesTheRequest()
    {
        // A TENSORSHARP_* value is just as explicit an operator decision as a flag.
        _env.Set("TENSORSHARP_TEMPERATURE", "0.65");
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);
        using var doc = JsonDocument.Parse("""{"temperature": 0.1}""");

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(0.65f, effective.Temperature);
    }

    [Fact]
    public void PinnedStopSequences_MergeWithTheRequestsInsteadOfReplacingThem()
    {
        // A client's stop strings terminate its own protocol, so they survive;
        // the operator's are added on top.
        var options = ServerOptionsBuilder.Build(new[] { "--stop", "<|eot|>" }, _baseDir);
        using var doc = JsonDocument.Parse("""{"stop": ["</tool_call>"]}""");

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(new[] { "</tool_call>", "<|eot|>" }, effective.StopSequences);
    }

    [Fact]
    public void PinnedValues_ApplyToWebUiAndOllamaBodiesToo()
    {
        var options = ServerOptionsBuilder.Build(new[] { "--temperature", "0.6" }, _baseDir);

        using var webUi = JsonDocument.Parse("""{"temperature": 0.1, "topK": 7}""");
        var fromWebUi = SamplingConfigParser.ParseWebUi(webUi.RootElement, options.SamplingDefaults);
        Assert.Equal(0.6f, fromWebUi.Temperature);
        Assert.Equal(7, fromWebUi.TopK);

        using var ollama = JsonDocument.Parse("""{"options": {"temperature": 0.1, "top_k": 7}}""");
        var fromOllama = SamplingConfigParser.ParseOllama(ollama.RootElement, options.SamplingDefaults);
        Assert.Equal(0.6f, fromOllama.Temperature);
        Assert.Equal(7, fromOllama.TopK);
    }

    [Fact]
    public void ParsingDoesNotMutateTheSharedDefaults()
    {
        var options = ServerOptionsBuilder.Build(new[] { "--temperature", "0.6", "--stop", "<|eot|>" }, _baseDir);
        using var doc = JsonDocument.Parse("""{"temperature": 0.1, "stop": ["</tool_call>"]}""");

        SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);
        SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(0.6f, options.DefaultSamplingConfig.Temperature);
        Assert.Equal(new[] { "<|eot|>" }, options.DefaultSamplingConfig.StopSequences);
    }

    [Theory]
    [InlineData("""{"temperature": null, "top_p": null, "max_tokens": null}""")]
    [InlineData("""{"temperature": "0.9", "top_k": true}""")]
    public void MalformedOrNullSamplingFields_AreIgnoredRatherThanThrowing(string body)
    {
        // Several OpenAI clients serialise unset options as explicit nulls; a
        // 500 on the whole request is a poor answer to that.
        var options = ServerOptionsBuilder.Build(new[] { "--temperature", "0.6" }, _baseDir);
        using var doc = JsonDocument.Parse(body);

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(0.6f, effective.Temperature);
        Assert.Equal(2048, ServerOptionsBuilder.Build(
            new[] { "--max-tokens", "2048" }, _baseDir).ResolveMaxTokens(
                SamplingConfigParser.ReadRequestedMaxTokens(doc.RootElement, "max_tokens")));
    }

    [Fact]
    public void MaxTokens_PinnedValueFillsInAndCapsButNeverLengthensARequest()
    {
        var options = ServerOptionsBuilder.Build(new[] { "--max-tokens", "2048" }, _baseDir);

        Assert.Equal(2048, options.ResolveMaxTokens(null));       // client said nothing
        Assert.Equal(2048, options.ResolveMaxTokens(-1));         // Ollama's "unbounded"
        Assert.Equal(2048, options.ResolveMaxTokens(100000));     // clamped to the operator's cap
        Assert.Equal(64, options.ResolveMaxTokens(64));           // a shorter request stays short
    }

    [Fact]
    public void MaxTokens_UnpinnedDefaultDoesNotCapRequests()
    {
        // Without --max-tokens / MAX_TOKENS the 20000 fallback is only a default,
        // so a client asking for more is still served (unchanged behaviour).
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);

        Assert.False(options.MaxTokensPinned);
        Assert.Equal(20000, options.ResolveMaxTokens(null));
        Assert.Equal(50000, options.ResolveMaxTokens(50000));
    }

    [Fact]
    public void MaxTokens_MaxTokensEnvVarPinsToo()
    {
        _env.Set("MAX_TOKENS", "512");
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);

        Assert.True(options.MaxTokensPinned);
        Assert.Equal(512, options.ResolveMaxTokens(null));
        Assert.Equal(512, options.ResolveMaxTokens(4096));
    }

    [Fact]
    public void ReadRequestedMaxTokens_AcceptsOpenAisNewerMaxCompletionTokens()
    {
        using var legacy = JsonDocument.Parse("""{"max_tokens": 128}""");
        using var modern = JsonDocument.Parse("""{"max_completion_tokens": 256}""");
        using var neither = JsonDocument.Parse("""{"model": "m"}""");

        Assert.Equal(128, SamplingConfigParser.ReadRequestedMaxTokens(legacy.RootElement, "max_tokens", "max_completion_tokens"));
        Assert.Equal(256, SamplingConfigParser.ReadRequestedMaxTokens(modern.RootElement, "max_tokens", "max_completion_tokens"));
        Assert.Null(SamplingConfigParser.ReadRequestedMaxTokens(neither.RootElement, "max_tokens", "max_completion_tokens"));
    }

    [Fact]
    public void OpenAIBody_AcceptsTheLlamaCppStyleSamplerExtensions()
    {
        var options = ServerOptionsBuilder.Build(Array.Empty<string>(), _baseDir);
        using var doc = JsonDocument.Parse("""{"top_k": 64, "min_p": 0.05, "repeat_penalty": 1.2}""");

        var effective = SamplingConfigParser.ParseOpenAI(doc.RootElement, options.SamplingDefaults);

        Assert.Equal(64, effective.TopK);
        Assert.Equal(0.05f, effective.MinP);
        Assert.Equal(1.2f, effective.RepetitionPenalty);
    }
}
