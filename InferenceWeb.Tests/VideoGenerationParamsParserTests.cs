// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Verifies the precedence between server startup defaults and fields supplied
/// to any of the three video-generation endpoints, plus the model-agnostic fields
/// added for joint audio-video and reference-conditioned models.
/// </summary>
public class VideoGenerationParamsParserTests : IDisposable
{
    private readonly string _baseDir;

    public VideoGenerationParamsParserTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-wan-video-params-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Parse_OmittedFields_UsesServerStartupDefaults()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("{}");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(81, parsed.Frames);
        Assert.Equal(16, parsed.Fps);
    }

    [Fact]
    public void Parse_RequestFields_OverrideServerStartupDefaults()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"frames":121,"fps":24}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(121, parsed.Frames);
        Assert.Equal(24, parsed.Fps);
    }

    [Fact]
    public void Parse_PartialRequestOverride_RetainsOtherStartupDefault()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"frames":121}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(121, parsed.Frames);
        Assert.Equal(16, parsed.Fps);
    }

    [Fact]
    public void Parse_ExplicitZero_RestoresWanModelDefaultsForThatRequest()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"frames":0,"fps":0}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(0, parsed.Frames);
        Assert.Equal(0, parsed.Fps);
    }

    [Fact]
    public void Parse_ImagePathAsBareFileName_ResolvesUnderUploadDirectory()
    {
        var options = BuildOptions(81, 16);
        File.WriteAllBytes(Path.Combine(options.UploadDirectory, "cond.png"), new byte[] { 1, 2, 3 });
        using var doc = JsonDocument.Parse("""{"imagePath":"cond.png"}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(new byte[] { 1, 2, 3 }, parsed.ImageBytes);
    }

    [Fact]
    public void Parse_ImagePathOutsideUploadDirectory_IsRejected()
    {
        var options = BuildOptions(81, 16);
        string outside = Path.Combine(_baseDir, "outside.png");
        File.WriteAllBytes(outside, new byte[] { 1 });
        using var doc = JsonDocument.Parse($$"""{"imagePath":{{JsonSerializer.Serialize(outside)}}}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.NotNull(error);
        Assert.Null(parsed.ImageBytes);
    }

    // ---- model-agnostic fields added alongside the second video model ----------
    // Every one is accepted in camelCase (web UI) and snake_case (OpenAI-shaped route).

    [Fact]
    public void Parse_GenerateAudio_DefaultsToTrueAndAcceptsBothSpellings()
    {
        var options = BuildOptions(81, 16);

        using var none = JsonDocument.Parse("{}");
        Assert.True(VideoGenerationParamsParser.Parse(none.RootElement, options, out _).GenerateAudio);

        using var camel = JsonDocument.Parse("""{"generateAudio":false}""");
        Assert.False(VideoGenerationParamsParser.Parse(camel.RootElement, options, out _).GenerateAudio);

        using var snake = JsonDocument.Parse("""{"generate_audio":false}""");
        Assert.False(VideoGenerationParamsParser.Parse(snake.RootElement, options, out _).GenerateAudio);
    }

    [Theory]
    [InlineData("endImage")]
    [InlineData("end_image")]
    public void Parse_EndImage_ResolvesAnUploadedFile(string field)
    {
        var options = BuildOptions(81, 16);
        string name = WriteUpload(options, "last.png");
        using var doc = JsonDocument.Parse($$"""{"{{field}}":"{{name}}"}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.NotNull(parsed.EndImagePath);
        Assert.True(File.Exists(parsed.EndImagePath));
    }

    [Theory]
    [InlineData("referenceImages")]
    [InlineData("reference_images")]
    public void Parse_ReferenceImages_ResolvesEveryEntry(string field)
    {
        var options = BuildOptions(81, 16);
        string a = WriteUpload(options, "ref-a.png");
        string b = WriteUpload(options, "ref-b.png");
        using var doc = JsonDocument.Parse($$"""{"{{field}}":["{{a}}","{{b}}"]}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(2, parsed.ReferenceImagePaths.Count);
        Assert.All(parsed.ReferenceImagePaths, p => Assert.True(File.Exists(p)));
    }

    [Fact]
    public void Parse_ReferenceVideosAndAudios_AreParsedIndependently()
    {
        var options = BuildOptions(81, 16);
        string vid = WriteUpload(options, "clip.mp4");
        string aud = WriteUpload(options, "theme.wav");
        using var doc = JsonDocument.Parse(
            $$"""{"referenceVideos":["{{vid}}"],"reference_audios":["{{aud}}"]}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Single(parsed.ReferenceVideoPaths);
        Assert.Single(parsed.ReferenceAudioPaths);
        Assert.Null(parsed.ReferenceImagePaths);
    }

    [Fact]
    public void Parse_OmittedReferenceLists_StayNull()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("{}");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Null(parsed.EndImagePath);
        Assert.Null(parsed.ReferenceImagePaths);
        Assert.Null(parsed.ReferenceVideoPaths);
        Assert.Null(parsed.ReferenceAudioPaths);
    }

    // ---- confinement: these are server-side paths, so they get imagePath's treatment ----

    [Theory]
    [InlineData("""{"endImage":"../../../etc/passwd"}""")]
    [InlineData("""{"referenceImages":["../../../etc/passwd"]}""")]
    [InlineData("""{"referenceVideos":["../../../etc/passwd"]}""")]
    [InlineData("""{"referenceAudios":["../../../etc/passwd"]}""")]
    public void Parse_TraversalOutsideTheUploadRoot_IsRejected(string json)
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse(json);

        VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.NotNull(error);
    }

    [Fact]
    public void Parse_ReferenceEntryThatWasNeverUploaded_IsRejected()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"referenceImages":["never-uploaded.png"]}""");

        var parsed = VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.NotNull(error);
        Assert.Null(parsed.ReferenceImagePaths);
    }

    [Fact]
    public void Parse_NonStringReferenceEntry_IsRejected()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"referenceImages":[42]}""");

        VideoGenerationParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.NotNull(error);
    }

    [Fact]
    public void Parse_VideoMode_TakesTheStartupDefaultAndTheRequestOverridesIt()
    {
        var options = ServerOptionsBuilder.Build(new[] { "--video-mode", "ref" }, _baseDir);

        using var none = JsonDocument.Parse("{}");
        Assert.Equal("ref", VideoGenerationParamsParser.Parse(none.RootElement, options, out _).Mode);

        using var camel = JsonDocument.Parse("""{"videoMode":"i2v"}""");
        Assert.Equal("i2v", VideoGenerationParamsParser.Parse(camel.RootElement, options, out _).Mode);

        using var snake = JsonDocument.Parse("""{"video_mode":"fl2v"}""");
        Assert.Equal("fl2v", VideoGenerationParamsParser.Parse(snake.RootElement, options, out _).Mode);
    }

    [Fact]
    public void Parse_VideoMode_StaysUnsetWithoutAStartupDefault()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("{}");

        // Unset means "infer it from what the request supplies", which is what every
        // deployment offering more than one mode wants.
        Assert.Null(VideoGenerationParamsParser.Parse(doc.RootElement, options, out _).Mode);
    }

    // Drop a file into the server's upload directory and return the bare name a
    // request would reference it by.
    private static string WriteUpload(ServerHostingOptions options, string name)
    {
        Directory.CreateDirectory(options.UploadDirectory);
        File.WriteAllBytes(Path.Combine(options.UploadDirectory, name), new byte[] { 1, 2, 3, 4 });
        return name;
    }

    private ServerHostingOptions BuildOptions(int frames, int fps)
    {
        return ServerOptionsBuilder.Build(
            new[] { "--video-frames", frames.ToString(), "--fps", fps.ToString() },
            _baseDir);
    }
}
