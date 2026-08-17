using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

/// <summary>
/// The /api/models response is readable by any client, so it must identify the
/// hosted model and projector by filename only — absolute host paths reveal the
/// server's filesystem layout. /api/models/load resolves the filename against
/// the startup path server-side (HostedModelGuard), so no client needs the
/// full path.
/// </summary>
public class ApiModelsResponseTests : IDisposable
{
    private readonly string _baseDir;

    public ApiModelsResponseTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-models-response-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void GetModels_ExposesFileNamesButNoHostPaths()
    {
        string modelDir = Path.Combine(_baseDir, "secret-model-location");
        var adapter = new WebUiAdapter(
            new ModelService(),
            new InferenceQueue(),
            new SessionManager(),
            new ServerHostingOptions(
                startupModelPath: Path.Combine(modelDir, "foo.gguf"),
                startupMmProjPath: Path.Combine(modelDir, "mmproj-foo.gguf"),
                defaultBackend: "ggml_cpu",
                supportedBackends: null,
                defaultMaxTokens: 100,
                maxTokensPinned: false,
                defaultWanVideoFrames: 0,
                defaultWanVideoFps: 0,
                uploadDirectory: _baseDir,
                logDirectory: Path.Combine(_baseDir, "logs"),
                fileLoggingEnabled: false,
                samplingDefaults: null),
            NullLoggerFactory.Instance);

        var result = adapter.GetModels();
        object value = result.GetType().GetProperty("Value")?.GetValue(result);
        Assert.NotNull(value);
        string json = JsonSerializer.Serialize(value);

        Assert.Contains("foo.gguf", json);
        Assert.Contains("mmproj-foo.gguf", json);
        Assert.DoesNotContain("secret-model-location", json);
        Assert.DoesNotContain(modelDir, json);
    }
}
