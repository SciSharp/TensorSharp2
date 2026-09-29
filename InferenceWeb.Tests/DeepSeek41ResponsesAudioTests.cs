// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41ResponsesAudioTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "dsv41-responses-audio-" + Guid.NewGuid().ToString("N"));

    public DeepSeek41ResponsesAudioTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, true);

    [Theory]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"AQID\"}}")]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"?\"}}")]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AQID\"}}")]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\"}}")]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":null}")]
    [InlineData("{\"type\":\"input_audio\",\"input_audio\":\"wrong-shape\"}")]
    [InlineData("{\"type\":\"input_audio\"}")]
    [InlineData("{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"https://example.invalid/audio.wav\"}}")]
    [InlineData("{\"type\":\"audio_url\",\"audio_url\":null}")]
    public void AudioInLaterMessageRejectsWholeInputBeforeWritingAnyMedia(string audioPart)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var input = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AQID"}]},
             {"role":"assistant","content":"An earlier image."},
             {"type":"message","role":"user","content":[
            """ + audioPart + "]}]");

        JsonException error = Assert.Throws<JsonException>(() => ChatMessageParser.ParseResponsesInput(
            input.RootElement, null, uploads, architecture: "deepseek41"));

        Assert.Equal(ChatGenerationPipeline.DeepSeek41AudioInputError, error.Message);
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.Equal(0, uploads.UsedBytes);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("gemma4")]
    [InlineData("qwen35")]
    public void OtherArchitecturesAndDefaultCallerKeepExistingAudioParsing(string? architecture)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var input = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"input_audio","input_audio":{"format":"wav","data":"AQID"}}]},
             {"role":"user","content":[{"type":"input_audio"}]}]
            """);

        var messages = ChatMessageParser.ParseResponsesInput(input.RootElement, null, uploads, architecture: architecture);

        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(Assert.Single(messages[0].AudioPaths)));
        Assert.Null(messages[1].AudioPaths);
        Assert.Equal(3, uploads.UsedBytes);
    }
}
