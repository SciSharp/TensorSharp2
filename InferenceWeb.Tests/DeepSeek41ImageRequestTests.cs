// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41ImageRequestTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "dsv41-image-" + Guid.NewGuid().ToString("N"));

    public DeepSeek41ImageRequestTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }

    [Theory]
    [InlineData("http://example.invalid/image.png")]
    [InlineData("https://example.invalid/image.png")]
    [InlineData("HTTPS://example.invalid/image.png")]
    [InlineData("file:///tmp/image.png")]
    [InlineData("")]
    [InlineData("data:image/png;base64")]
    [InlineData("data:image/png;base64,?")]
    [InlineData("data:image/png;base64,AQ=")]
    [InlineData("data:image/png;base64,")]
    public void InvalidLaterImageRejectsWholeHistoryBeforeAnyUpload(string url)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var request = JsonDocument.Parse(JsonSerializer.Serialize(new object[]
        {
            new { role = "user", content = new[] { Image("data:image/png;base64,AQID") } },
            new { role = "assistant", content = "An earlier image." },
            new { role = "user", content = new[] { Image(url) } },
        }));

        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(request.RootElement, uploads, architecture: "deepseek41"));
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.Equal(0, uploads.UsedBytes);
    }

    [Theory]
    [InlineData("{\"type\":\"image_url\"}")]
    [InlineData("{\"type\":\"image_url\",\"image_url\":\"https://example.invalid/image.png\"}")]
    [InlineData("{\"type\":\"image_url\",\"image_url\":{\"url\":null}}")]
    public void MalformedImageUrlShapeIsARequestError(string part)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var request = JsonDocument.Parse("[{\"role\":\"user\",\"content\":[" + part + "]}]");
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(request.RootElement, uploads, architecture: "deepseek41"));
        Assert.Empty(Directory.EnumerateFiles(_directory));
    }

    [Fact]
    public void ValidDataUrisPreserveAllImagesAcrossTurnsAndBase64Whitespace()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var request = JsonDocument.Parse(JsonSerializer.Serialize(new object[]
        {
            new { role = "user", content = new[] { Image("data:image/png;base64,AQID") } },
            new { role = "assistant", content = "Describe the next two images." },
            new { role = "user", content = new[]
            {
                Image("data:image/jpeg;charset=utf-8;base64, BAU=\r\n"),
                Image("data:image/webp;base64,BgcI"),
            } },
        }));
        var messages = ChatMessageParser.ParseOpenAI(request.RootElement, uploads, architecture: "deepseek41");
        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(Assert.Single(messages[0].ImagePaths)));
        Assert.Equal(2, messages[2].ImagePaths.Count);
        Assert.Equal(new byte[] { 4, 5 }, File.ReadAllBytes(messages[2].ImagePaths[0]));
        Assert.Equal(new byte[] { 6, 7, 8 }, File.ReadAllBytes(messages[2].ImagePaths[1]));
        Assert.Equal(8, uploads.UsedBytes);
    }

    [Theory]
    [InlineData("deepseek4")]
    [InlineData("qwen35")]
    public void OtherArchitecturesKeepExistingRemoteImagePolicy(string architecture)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var request = JsonDocument.Parse(JsonSerializer.Serialize(new[]
        {
            new { role = "user", content = new[] { Image("https://example.invalid/image.png") } },
        }));
        var message = Assert.Single(ChatMessageParser.ParseOpenAI(request.RootElement, uploads, architecture: architecture));
        Assert.Null(message.ImagePaths);
        Assert.Empty(Directory.EnumerateFiles(_directory));
    }

    private static object Image(string url) => new { type = "image_url", image_url = new { url } };

    [Theory]
    [InlineData("https://example.invalid/image.png")]
    [InlineData("http://example.invalid/image.png")]
    [InlineData("data:image/png;base64,?")]
    [InlineData("data:image/png;base64,")]
    [InlineData("")]
    [InlineData(null)]
    public void ResponsesInvalidLaterImageRejectsWholeInputBeforeAnyUpload(string? url)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var request = JsonDocument.Parse(JsonSerializer.Serialize(new[]
        {
            new { role = "user", content = new[] { new { type = "input_image", image_url = "data:image/png;base64,AQID" } } },
            new { role = "user", content = new[] { new { type = "input_image", image_url = url } } },
        }));
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseResponsesInput(request.RootElement, null, uploads, architecture: "deepseek41"));
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.Equal(0, uploads.UsedBytes);
    }

    [Fact]
    public void ResponsesDataUrisPreserveImagesAndOtherModelsKeepExistingPolicy()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var valid = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"input_image","image_url":"data:image/jpeg;base64,AQID"}]},
             {"role":"assistant","content":"First image."},
             {"role":"user","content":[{"type":"input_image","image_url":"data:image/webp;base64, BAU=\r\n"}]}]
            """);
        var messages = ChatMessageParser.ParseResponsesInput(valid.RootElement, null, uploads, architecture: "deepseek41");
        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(Assert.Single(messages[0].ImagePaths)));
        Assert.Equal(new byte[] { 4, 5 }, File.ReadAllBytes(Assert.Single(messages[2].ImagePaths)));
        Assert.Equal(5, uploads.UsedBytes);

        using var remote = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"input_image","image_url":"https://example.invalid/image.png"}]}]
            """);
        var unchanged = Assert.Single(ChatMessageParser.ParseResponsesInput(remote.RootElement, null, uploads, architecture: "qwen35"));
        Assert.Null(unchanged.ImagePaths);
        Assert.Equal(5, uploads.UsedBytes);
    }
}
