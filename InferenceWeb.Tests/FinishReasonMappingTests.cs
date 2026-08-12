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
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the translation from the generation pipeline's finish reason to each
/// protocol's own vocabulary — the thing that used to be a hard-coded
/// <c>"stop"</c> in every adapter, so a request that burned its whole
/// <c>max_tokens</c> budget and returned a half-sentence still claimed to have
/// finished normally. Clients continue truncated answers by testing
/// <c>finish_reason == "length"</c>, so these values are load-bearing.
///
/// <para>Pure mapping, no model required. The live end-to-end behaviour is
/// covered by running the server against a real GGUF.</para>
/// </summary>
public class FinishReasonMappingTests
{
    /// <summary>Every finish reason the pipeline can produce except the one that
    /// means "cut off". <c>max_tokens</c> is deliberately absent: it is the sole
    /// reason that must survive translation intact on every protocol.</summary>
    private static readonly string[] NonTruncatingReasons =
    {
        "eos",           // model emitted end-of-sequence: it finished by itself
        "stop_sequence", // a caller-supplied stop string matched
        "cancelled",     // client hung up mid-generation
        "aborted",       // engine terminated the sequence
        "error",         // engine failed the sequence
        "stop",          // diffusion sampler finished its planned blocks
        null,            // reason unavailable (e.g. an adapter path that never saw a done update)
        "something_new", // an unrecognised reason must not be mistaken for truncation
    };

    public static TheoryData<string> NonTruncatingPipelineReasons()
    {
        var data = new TheoryData<string>();
        foreach (string reason in NonTruncatingReasons)
            data.Add(reason);
        return data;
    }

    // ---- OpenAI /v1/chat/completions ---------------------------------------

    [Fact]
    public void OpenAIChat_MaxTokens_IsLength()
    {
        Assert.Equal("length", FinishReasonMapper.ToOpenAIChat("max_tokens", hasToolCalls: false));
    }

    [Theory]
    [MemberData(nameof(NonTruncatingPipelineReasons))]
    public void OpenAIChat_EverythingElse_IsStop(string pipelineReason)
    {
        Assert.Equal("stop", FinishReasonMapper.ToOpenAIChat(pipelineReason, hasToolCalls: false));
    }

    [Theory]
    [MemberData(nameof(NonTruncatingPipelineReasons))]
    public void OpenAIChat_ToolCalls_WinOverNaturalFinishes(string pipelineReason)
    {
        Assert.Equal("tool_calls", FinishReasonMapper.ToOpenAIChat(pipelineReason, hasToolCalls: true));
    }

    [Fact]
    public void OpenAIChat_Truncation_OutranksToolCalls()
    {
        // A run that hit the budget may have cut a tool call in half. Reporting
        // "tool_calls" would send the client off to dispatch the fragment instead
        // of noticing the answer never finished.
        Assert.Equal("length", FinishReasonMapper.ToOpenAIChat("max_tokens", hasToolCalls: true));
    }

    [Fact]
    public void OpenAIChat_OnlyEmitsValuesTheSchemaDefines()
    {
        var allowed = new HashSet<string> { "stop", "length", "tool_calls", "content_filter", "function_call" };
        foreach (string reason in AllPipelineReasons())
        {
            Assert.Contains(FinishReasonMapper.ToOpenAIChat(reason, hasToolCalls: false), allowed);
            Assert.Contains(FinishReasonMapper.ToOpenAIChat(reason, hasToolCalls: true), allowed);
        }
    }

    // ---- Ollama /api/chat, /api/generate -----------------------------------

    [Fact]
    public void Ollama_MaxTokens_IsLength()
    {
        Assert.Equal("length", FinishReasonMapper.ToOllamaDoneReason("max_tokens", hasToolCalls: false));
    }

    [Theory]
    [MemberData(nameof(NonTruncatingPipelineReasons))]
    public void Ollama_EverythingElse_IsStop(string pipelineReason)
    {
        Assert.Equal("stop", FinishReasonMapper.ToOllamaDoneReason(pipelineReason, hasToolCalls: false));
    }

    [Fact]
    public void Ollama_ToolCalls_PreserveTheirOwnDoneReason()
    {
        Assert.Equal("tool_calls", FinishReasonMapper.ToOllamaDoneReason("eos", hasToolCalls: true));
        Assert.Equal("length", FinishReasonMapper.ToOllamaDoneReason("max_tokens", hasToolCalls: true));
    }

    // ---- OpenAI /v1/responses ----------------------------------------------

    [Fact]
    public void Responses_MaxTokens_IsIncompleteWithMaxOutputTokens()
    {
        var (status, incompleteReason) = FinishReasonMapper.ToResponsesStatus("max_tokens");

        Assert.Equal("incomplete", status);
        Assert.Equal("max_output_tokens", incompleteReason);
    }

    [Theory]
    [MemberData(nameof(NonTruncatingPipelineReasons))]
    public void Responses_EverythingElse_IsCompletedWithNoDetails(string pipelineReason)
    {
        var (status, incompleteReason) = FinishReasonMapper.ToResponsesStatus(pipelineReason);

        Assert.Equal("completed", status);
        Assert.Null(incompleteReason);
    }

    // ---- IsTruncated -------------------------------------------------------

    [Fact]
    public void IsTruncated_OnlyMaxTokens()
    {
        Assert.True(FinishReasonMapper.IsTruncated("max_tokens"));
        foreach (string reason in NonTruncatingReasons)
            Assert.False(FinishReasonMapper.IsTruncated(reason));
    }

    [Fact]
    public void IsTruncated_IsCaseSensitive_SoNearMissesFailLoudlyRatherThanSilently()
    {
        // Guards against an adapter inventing its own spelling and quietly
        // getting the "finished normally" branch.
        Assert.False(FinishReasonMapper.IsTruncated("MAX_TOKENS"));
        Assert.False(FinishReasonMapper.IsTruncated("max-tokens"));
        Assert.False(FinishReasonMapper.IsTruncated("length"));
    }

    // ---- Wire shape --------------------------------------------------------

    [Fact]
    public void ResponsesFactory_IncompleteResponse_CarriesIncompleteDetailsAndUsage()
    {
        var (status, incompleteReason) = FinishReasonMapper.ToResponsesStatus("max_tokens");
        var output = new List<object>
        {
            OpenAIResponsesFactory.OutputMessageItem("msg_1", "a half-finished thought", "incomplete"),
        };

        var response = OpenAIResponsesFactory.Response(
            "resp_abc", "test-model", status, instructions: null, maxOutputTokens: 200, output,
            store: false, samplingConfig: null, promptTokens: 12, evalTokens: 200, kvCacheReusedTokens: 0,
            incompleteReason: incompleteReason);

        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(response));
        var root = doc.RootElement;

        Assert.Equal("incomplete", root.GetProperty("status").GetString());
        Assert.Equal("max_output_tokens",
            root.GetProperty("incomplete_details").GetProperty("reason").GetString());
        // A truncated response still generated tokens, and output_tokens is exactly
        // what proves the budget is what stopped it.
        Assert.Equal(200, root.GetProperty("usage").GetProperty("output_tokens").GetInt32());
        Assert.Equal("incomplete", root.GetProperty("output")[0].GetProperty("status").GetString());
    }

    [Fact]
    public void ResponsesFactory_CompletedResponse_HasNullIncompleteDetails()
    {
        var (status, incompleteReason) = FinishReasonMapper.ToResponsesStatus("eos");

        var response = OpenAIResponsesFactory.Response(
            "resp_abc", "test-model", status, instructions: null, maxOutputTokens: 600,
            new List<object>(), store: false, samplingConfig: null,
            promptTokens: 12, evalTokens: 295, kvCacheReusedTokens: 0,
            incompleteReason: incompleteReason);

        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(response));
        var root = doc.RootElement;

        Assert.Equal("completed", root.GetProperty("status").GetString());
        Assert.Equal(JsonValueKind.Null, root.GetProperty("incomplete_details").ValueKind);
    }

    [Fact]
    public void OllamaFactory_FinalChunks_EmitTheMappedDoneReason()
    {
        string truncated = FinishReasonMapper.ToOllamaDoneReason("max_tokens", hasToolCalls: false);

        Assert.Equal("length", ReadDoneReason(
            OllamaResponseFactory.ChatRawFinalChunk("m", truncated, 10, 200, 0, 1, 1, 1)));
        Assert.Equal("length", ReadDoneReason(
            OllamaResponseFactory.GenerateFinalChunk("m", truncated, 10, 200, 0, 1, 1, 1)));
        Assert.Equal("length", ReadDoneReason(
            OllamaResponseFactory.GenerateNonStreamingResponse("m", "cut off", truncated, 10, 200, 0, 1, 1, 1)));
        Assert.Equal("length", ReadDoneReason(
            OllamaResponseFactory.ChatParsedFinalChunk("m", truncated, null, 10, 200, 0, 1, 1, 1)));

        string finished = FinishReasonMapper.ToOllamaDoneReason("eos", hasToolCalls: false);
        Assert.Equal("stop", ReadDoneReason(
            OllamaResponseFactory.ChatRawFinalChunk("m", finished, 10, 42, 0, 1, 1, 1)));
    }

    [Fact]
    public void OpenAIFactory_EndChunkAndCompletion_CarryTheMappedFinishReason()
    {
        string truncated = FinishReasonMapper.ToOpenAIChat("max_tokens", hasToolCalls: false);

        using var chunkDoc = JsonDocument.Parse(JsonSerializer.Serialize(
            OpenAIResponseFactory.EndChunk("chatcmpl-1", "m", truncated, 10, 200, 0)));
        Assert.Equal("length",
            chunkDoc.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal(200,
            chunkDoc.RootElement.GetProperty("usage").GetProperty("completion_tokens").GetInt32());

        using var completionDoc = JsonDocument.Parse(JsonSerializer.Serialize(
            OpenAIResponseFactory.Completion("chatcmpl-1", "m",
                OpenAIResponseFactory.PlainAssistantMessage("cut off"), truncated, 10, 200, 0)));
        Assert.Equal("length",
            completionDoc.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
    }

    [Fact]
    public void WebUiDoneEvent_FlagsTruncationSeparatelyFromAbort()
    {
        // `aborted` means the user stopped it; `truncated` means the budget ran
        // out. Conflating them would tell a user to retry when they should raise
        // max tokens.
        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(
            WebUiSseEvents.Done(200, 14.0, 14.3, aborted: false, error: null, sessionId: "s1",
                promptTokens: 10, kvCacheReusedTokens: 0,
                truncated: FinishReasonMapper.IsTruncated("max_tokens"))));

        Assert.True(doc.RootElement.GetProperty("truncated").GetBoolean());
        Assert.False(doc.RootElement.GetProperty("aborted").GetBoolean());
    }

    // ---- Helpers -----------------------------------------------------------

    private static string ReadDoneReason(object chunk)
    {
        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(chunk));
        return doc.RootElement.GetProperty("done_reason").GetString();
    }

    private static IEnumerable<string> AllPipelineReasons()
    {
        yield return "max_tokens";
        foreach (string reason in NonTruncatingReasons)
            yield return reason;
    }
}
