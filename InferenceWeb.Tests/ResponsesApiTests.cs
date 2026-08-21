using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

public class ResponsesApiTests
{
    // ---- ChatMessageParser.ParseResponsesInput ----------------------------

    [Fact]
    public void ParseResponsesInput_StringInput_BecomesSingleUserMessage()
    {
        using var doc = JsonDocument.Parse("\"hello there\"");
        var messages = ChatMessageParser.ParseResponsesInput(doc.RootElement, null, new UploadStoragePolicy(Path.GetTempPath()));

        var msg = Assert.Single(messages);
        Assert.Equal("user", msg.Role);
        Assert.Equal("hello there", msg.Content);
    }

    [Fact]
    public void ParseResponsesInput_Instructions_PrependsSystemMessage()
    {
        using var doc = JsonDocument.Parse("\"hi\"");
        var messages = ChatMessageParser.ParseResponsesInput(doc.RootElement, "be nice", new UploadStoragePolicy(Path.GetTempPath()));

        Assert.Equal(2, messages.Count);
        Assert.Equal("system", messages[0].Role);
        Assert.Equal("be nice", messages[0].Content);
        Assert.Equal("user", messages[1].Role);
    }

    [Fact]
    public void ParseResponsesInput_ArrayOfMessageItems_ParsesRoleAndStringContent()
    {
        const string body = """
        [
          {"role": "user", "content": "first turn"},
          {"role": "assistant", "content": "reply"},
          {"role": "user", "content": "second turn"}
        ]
        """;
        using var doc = JsonDocument.Parse(body);
        var messages = ChatMessageParser.ParseResponsesInput(doc.RootElement, null, new UploadStoragePolicy(Path.GetTempPath()));

        Assert.Equal(3, messages.Count);
        Assert.Equal(["user", "assistant", "user"], messages.ConvertAll(m => m.Role));
        Assert.Equal("second turn", messages[2].Content);
    }

    [Fact]
    public void ParseResponsesInput_ContentPartsArray_JoinsTextPartsAndDecodesImage()
    {
        byte[] pngBytes = [137, 80, 78, 71];
        string b64 = Convert.ToBase64String(pngBytes);
        string tempDir = Path.Combine(Path.GetTempPath(), "responses-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            string body = $$"""
            [
              {"role": "user", "content": [
                {"type": "input_text", "text": "what is this"},
                {"type": "input_image", "image_url": "data:image/png;base64,{{b64}}"}
              ]}
            ]
            """;
            using var doc = JsonDocument.Parse(body);
            var messages = ChatMessageParser.ParseResponsesInput(doc.RootElement, null, new UploadStoragePolicy(tempDir));

            var msg = Assert.Single(messages);
            Assert.Equal("what is this", msg.Content);
            var imagePath = Assert.Single(msg.ImagePaths);
            Assert.True(File.Exists(imagePath));
            Assert.Equal(pngBytes, File.ReadAllBytes(imagePath));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void ParseResponsesInput_IgnoresNonMessageItemTypes()
    {
        const string body = """
        [
          {"type": "function_call_output", "call_id": "call_1", "output": "42"},
          {"role": "user", "content": "hi"}
        ]
        """;
        using var doc = JsonDocument.Parse(body);
        var messages = ChatMessageParser.ParseResponsesInput(doc.RootElement, null, new UploadStoragePolicy(Path.GetTempPath()));

        var msg = Assert.Single(messages);
        Assert.Equal("user", msg.Role);
    }

    // ---- ToolFunctionParser.ParseOpenAIResponses ----------------------------

    [Fact]
    public void ParseOpenAIResponses_FlatToolShape_ParsesNameParametersAndRequired()
    {
        const string body = """
        {
          "tools": [
            {
              "type": "function",
              "name": "get_weather",
              "description": "Get the weather",
              "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string", "description": "City name" } },
                "required": ["city"]
              }
            }
          ]
        }
        """;
        using var doc = JsonDocument.Parse(body);
        var tools = ToolFunctionParser.ParseOpenAIResponses(doc.RootElement);

        var tool = Assert.Single(tools);
        Assert.Equal("get_weather", tool.Name);
        Assert.Equal("Get the weather", tool.Description);
        Assert.Equal("string", tool.Parameters["city"].Type);
        Assert.Equal(["city"], tool.Required);
    }

    [Fact]
    public void ParseOpenAIResponses_NoToolsField_ReturnsNull()
    {
        using var doc = JsonDocument.Parse("{}");
        Assert.Null(ToolFunctionParser.ParseOpenAIResponses(doc.RootElement));
    }

    // ---- OpenAIResponseFormatParser.TryParseResponsesText -------------------

    [Fact]
    public void TryParseResponsesText_NoTextField_SucceedsWithNullFormat()
    {
        using var doc = JsonDocument.Parse("{}");
        bool ok = OpenAIResponseFormatParser.TryParseResponsesText(doc.RootElement, out var format, out var error);

        Assert.True(ok);
        Assert.Null(format);
        Assert.Null(error);
    }

    [Fact]
    public void TryParseResponsesText_JsonSchema_ParsesFlatShape()
    {
        const string body = """
        {
          "text": {
            "format": {
              "type": "json_schema",
              "name": "planet",
              "schema": { "type": "object", "properties": { "name": { "type": "string" } } },
              "strict": true
            }
          }
        }
        """;
        using var doc = JsonDocument.Parse(body);
        bool ok = OpenAIResponseFormatParser.TryParseResponsesText(doc.RootElement, out var format, out var error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal(StructuredOutputKind.JsonSchema, format.Kind);
    }

    [Fact]
    public void TryParseResponsesText_JsonObject_ReturnsJsonObjectFormat()
    {
        const string body = """{"text": {"format": {"type": "json_object"}}}""";
        using var doc = JsonDocument.Parse(body);
        bool ok = OpenAIResponseFormatParser.TryParseResponsesText(doc.RootElement, out var format, out var error);

        Assert.True(ok);
        Assert.Equal(StructuredOutputKind.JsonObject, format.Kind);
    }

    [Fact]
    public void TryParseResponsesText_UnsupportedType_Fails()
    {
        const string body = """{"text": {"format": {"type": "bogus"}}}""";
        using var doc = JsonDocument.Parse(body);
        bool ok = OpenAIResponseFormatParser.TryParseResponsesText(doc.RootElement, out var format, out var error);

        Assert.False(ok);
        Assert.Null(format);
        Assert.Contains("text.format.type", error);
    }

    [Fact]
    public void TryParseResponsesText_JsonSchemaMissingName_ErrorUsesTextFormatPrefixNotResponseFormat()
    {
        const string body = """
        {
          "text": {
            "format": { "type": "json_schema", "schema": { "type": "object" } }
          }
        }
        """;
        using var doc = JsonDocument.Parse(body);
        bool ok = OpenAIResponseFormatParser.TryParseResponsesText(doc.RootElement, out var format, out var error);

        Assert.False(ok);
        Assert.Null(format);
        Assert.StartsWith("text.format.json_schema.name", error);
        Assert.DoesNotContain("response_format", error);
    }

    [Fact]
    public void TryParse_ChatCompletions_JsonSchemaMissingName_StillUsesResponseFormatPrefix()
    {
        const string body = """
        {
          "response_format": {
            "type": "json_schema",
            "json_schema": { "schema": { "type": "object" } }
          }
        }
        """;
        using var doc = JsonDocument.Parse(body);
        bool ok = OpenAIResponseFormatParser.TryParse(doc.RootElement, out var format, out var error);

        Assert.False(ok);
        Assert.Null(format);
        Assert.StartsWith("response_format.json_schema.name", error);
    }

    // ---- OpenAIResponsesFactory ---------------------------------------------

    [Fact]
    public void ResponsesFactory_Response_CompletedShapeIncludesUsageAndOutput()
    {
        var samplingConfig = new SamplingConfig { Temperature = 0.7f, TopP = 0.9f };
        var output = new List<object> { OpenAIResponsesFactory.OutputMessageItem("msg_1", "hi there") };

        var response = OpenAIResponsesFactory.Response(
            "resp_abc", "test-model", "completed", "be nice", 200, output,
            store: true, samplingConfig, promptTokens: 10, evalTokens: 5, kvCacheReusedTokens: 2);

        string json = JsonSerializer.Serialize(response);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        Assert.Equal("resp_abc", root.GetProperty("id").GetString());
        Assert.Equal("response", root.GetProperty("object").GetString());
        Assert.Equal("completed", root.GetProperty("status").GetString());
        Assert.Equal("test-model", root.GetProperty("model").GetString());
        Assert.Equal(10, root.GetProperty("usage").GetProperty("input_tokens").GetInt32());
        Assert.Equal(5, root.GetProperty("usage").GetProperty("output_tokens").GetInt32());
        Assert.Equal(2, root.GetProperty("usage").GetProperty("input_tokens_details").GetProperty("cached_tokens").GetInt32());
        Assert.Equal(1, root.GetProperty("output").GetArrayLength());
    }

    [Fact]
    public void ResponsesFactory_FunctionCallItem_SerializesArgumentsAsJsonString()
    {
        var call = new ToolCall { Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } };
        var item = OpenAIResponsesFactory.FunctionCallItem("fc_1", "call_1", call);

        string json = JsonSerializer.Serialize(item);
        using var doc = JsonDocument.Parse(json);
        Assert.Equal("function_call", doc.RootElement.GetProperty("type").GetString());
        Assert.Equal("get_weather", doc.RootElement.GetProperty("name").GetString());

        using var argsDoc = JsonDocument.Parse(doc.RootElement.GetProperty("arguments").GetString()!);
        Assert.Equal("Paris", argsDoc.RootElement.GetProperty("city").GetString());
    }

    // ---- InMemoryResponsesStore ----------------------------------------------

    [Fact]
    public void InMemoryResponsesStore_StoreThenGet_RoundTrips()
    {
        var store = new InMemoryResponsesStore();
        store.Store(new StoredResponse { Id = "resp_1", Json = """{"id":"resp_1"}""" });

        Assert.True(store.TryGet("resp_1", out var found));
        Assert.Equal("""{"id":"resp_1"}""", found.Json);
    }

    [Fact]
    public void InMemoryResponsesStore_UnknownId_ReturnsFalse()
    {
        var store = new InMemoryResponsesStore();
        Assert.False(store.TryGet("does_not_exist", out _));
    }

    [Fact]
    public void InMemoryResponsesStore_EntryOlderThanTtl_ExpiresAndIsNotReturned()
    {
        using var store = new InMemoryResponsesStore(maxEntries: 100, ttl: TimeSpan.FromMilliseconds(1));
        store.Store(new StoredResponse { Id = "resp_1", Json = "{}" });

        // MemoryCache expires entries lazily (on the next access after the clock
        // passes the absolute expiration), so poll briefly rather than assuming
        // a fixed delay is always enough on a loaded CI box.
        bool expired = SpinWaitUntil(() => !store.TryGet("resp_1", out _), TimeSpan.FromSeconds(2));
        Assert.True(expired, "expected the entry to expire once its TTL elapsed");
    }

    [Fact]
    public void InMemoryResponsesStore_OverMaxEntries_EvictsUnderPressure()
    {
        using var store = new InMemoryResponsesStore(maxEntries: 2, ttl: TimeSpan.FromMinutes(30));
        store.Store(new StoredResponse { Id = "resp_1", Json = "{}" });
        store.Store(new StoredResponse { Id = "resp_2", Json = "{}" });
        store.Store(new StoredResponse { Id = "resp_3", Json = "{}" });

        // MemoryCache's size-limit eviction runs on a compaction pass rather than
        // synchronously on the Set() call that exceeds the limit, so poll instead
        // of asserting immediately after the third Store().
        bool evicted = SpinWaitUntil(() =>
            !(store.TryGet("resp_1", out _) && store.TryGet("resp_2", out _) && store.TryGet("resp_3", out _)),
            TimeSpan.FromSeconds(2));
        Assert.True(evicted, "expected at least one entry to be evicted once the store exceeded its max size");
    }

    [Fact]
    public void InMemoryResponsesStore_MaxEntriesEnvVar_IsTrimmedBeforeParsing()
    {
        WithEnvVar("TS_RESPONSES_STORE_MAX_ENTRIES", " 2 ", () =>
        {
            using var store = new InMemoryResponsesStore();
            store.Store(new StoredResponse { Id = "resp_1", Json = "{}" });
            store.Store(new StoredResponse { Id = "resp_2", Json = "{}" });
            store.Store(new StoredResponse { Id = "resp_3", Json = "{}" });

            // If the padded value fell back to the 1000-entry default (i.e. trimming
            // was missing), nothing would ever be evicted here.
            bool evicted = SpinWaitUntil(() =>
                !(store.TryGet("resp_1", out _) && store.TryGet("resp_2", out _) && store.TryGet("resp_3", out _)),
                TimeSpan.FromSeconds(2));
            Assert.True(evicted, "expected the trimmed max-entries env var to be honoured");
        });
    }

    [Fact]
    public void InMemoryResponsesStore_TtlEnvVar_ParsedWithInvariantCultureRegardlessOfCurrentCulture()
    {
        var previousCulture = System.Threading.Thread.CurrentThread.CurrentCulture;
        System.Threading.Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo("de-DE");
        try
        {
            // "0.02" (a period) is only valid under InvariantCulture; under de-DE,
            // '.' is a group separator, so a culture-sensitive parse would silently
            // fall back to the 60-minute default instead of expiring almost immediately.
            WithEnvVar("TS_RESPONSES_STORE_TTL_MINUTES", "  0.02  ", () =>
            {
                using var store = new InMemoryResponsesStore();
                store.Store(new StoredResponse { Id = "resp_1", Json = "{}" });

                bool expired = SpinWaitUntil(() => !store.TryGet("resp_1", out _), TimeSpan.FromSeconds(3));
                Assert.True(expired, "expected the invariant-culture-parsed TTL (~1.2s) to be honoured");
            });
        }
        finally
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = previousCulture;
        }
    }

    private static void WithEnvVar(string name, string value, Action action)
    {
        string previous = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, value);
        try
        {
            action();
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, previous);
        }
    }

    private static bool SpinWaitUntil(Func<bool> condition, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (condition())
                return true;
            System.Threading.Thread.Sleep(10);
        }
        return condition();
    }
}
