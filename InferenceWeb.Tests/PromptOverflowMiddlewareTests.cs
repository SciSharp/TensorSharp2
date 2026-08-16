using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using TensorSharp.Server;
using TensorSharp.Server.Logging;

namespace InferenceWeb.Tests;

/// <summary>
/// A prompt that does not fit the context window must surface as an HTTP 400
/// with a protocol-appropriate JSON body. The overflow is thrown during prompt
/// preparation, before any response body is written, so the middleware can
/// still set the status.
/// </summary>
public class PromptOverflowMiddlewareTests
{
    private static DefaultHttpContext ContextFor(string path)
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Path = path;
        ctx.Response.Body = new MemoryStream();
        return ctx;
    }

    private static async Task<string> BodyOf(HttpContext ctx)
    {
        ctx.Response.Body.Seek(0, SeekOrigin.Begin);
        using var reader = new StreamReader(ctx.Response.Body, Encoding.UTF8);
        return await reader.ReadToEndAsync();
    }

    [Fact]
    public async Task Overflow_OnOpenAiRoute_Becomes400WithOpenAiErrorShape()
    {
        var ctx = ContextFor("/v1/chat/completions");

        await PromptOverflowMiddleware.InvokeAsync(ctx,
            () => throw new PromptContextOverflowException("Prompt (9000 tokens) exceeds the model's context limit (4096 tokens)."));

        Assert.Equal(400, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Contains("context limit", error.GetProperty("message").GetString());
    }

    [Fact]
    public async Task Overflow_OnOllamaRoute_Becomes400WithFlatErrorShape()
    {
        var ctx = ContextFor("/api/generate");

        await PromptOverflowMiddleware.InvokeAsync(ctx,
            () => throw new PromptContextOverflowException("Prompt exceeds the model's context limit."));

        Assert.Equal(400, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("error").ValueKind);
    }

    [Fact]
    public async Task NonOverflowException_IsNotSwallowed()
    {
        var ctx = ContextFor("/v1/chat/completions");

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            PromptOverflowMiddleware.InvokeAsync(ctx,
                () => throw new InvalidOperationException("something else")));
    }

    [Fact]
    public async Task Overflow_AfterResponseStarted_IsNotConvertedTo400()
    {
        // A stream already in flight: the body is committed, so the status
        // can't be rewritten and the exception must propagate.
        var ctx = ContextFor("/v1/chat/completions");
        ctx.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpResponseFeature>(new StartedResponseFeature());

        await Assert.ThrowsAsync<PromptContextOverflowException>(() =>
            PromptOverflowMiddleware.InvokeAsync(ctx,
                () => throw new PromptContextOverflowException("too long")));
    }

    private sealed class StartedResponseFeature : Microsoft.AspNetCore.Http.Features.IHttpResponseFeature
    {
        public int StatusCode { get; set; } = 200;
        public string ReasonPhrase { get; set; }
        public IHeaderDictionary Headers { get; set; } = new HeaderDictionary();
        public Stream Body { get; set; } = new MemoryStream();
        public bool HasStarted => true;
        public void OnStarting(System.Func<object, Task> callback, object state) { }
        public void OnCompleted(System.Func<object, Task> callback, object state) { }
    }

    [Fact]
    public async Task NoException_PassesThroughUntouched()
    {
        var ctx = ContextFor("/v1/chat/completions");
        ctx.Response.StatusCode = 200;

        await PromptOverflowMiddleware.InvokeAsync(ctx, () => Task.CompletedTask);

        Assert.Equal(200, ctx.Response.StatusCode);
    }
}
