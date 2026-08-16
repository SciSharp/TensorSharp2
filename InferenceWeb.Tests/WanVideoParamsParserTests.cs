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
/// to any of the three Wan video-generation endpoints.
/// </summary>
public class WanVideoParamsParserTests : IDisposable
{
    private readonly string _baseDir;

    public WanVideoParamsParserTests()
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

        var parsed = WanVideoParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(81, parsed.Frames);
        Assert.Equal(16, parsed.Fps);
    }

    [Fact]
    public void Parse_RequestFields_OverrideServerStartupDefaults()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"frames":121,"fps":24}""");

        var parsed = WanVideoParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(121, parsed.Frames);
        Assert.Equal(24, parsed.Fps);
    }

    [Fact]
    public void Parse_PartialRequestOverride_RetainsOtherStartupDefault()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"frames":121}""");

        var parsed = WanVideoParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.Null(error);
        Assert.Equal(121, parsed.Frames);
        Assert.Equal(16, parsed.Fps);
    }

    [Fact]
    public void Parse_ExplicitZero_RestoresWanModelDefaultsForThatRequest()
    {
        var options = BuildOptions(81, 16);
        using var doc = JsonDocument.Parse("""{"frames":0,"fps":0}""");

        var parsed = WanVideoParamsParser.Parse(doc.RootElement, options, out string error);

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

        var parsed = WanVideoParamsParser.Parse(doc.RootElement, options, out string error);

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

        var parsed = WanVideoParamsParser.Parse(doc.RootElement, options, out string error);

        Assert.NotNull(error);
        Assert.Null(parsed.ImageBytes);
    }

    private ServerHostingOptions BuildOptions(int frames, int fps)
    {
        return ServerOptionsBuilder.Build(
            new[] { "--video-frames", frames.ToString(), "--fps", fps.ToString() },
            _baseDir);
    }
}
