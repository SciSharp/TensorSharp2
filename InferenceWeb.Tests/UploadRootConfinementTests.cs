using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// The image-edit (<c>imagePath</c>/<c>imagePaths</c>) and Wan video
/// (<c>imagePath</c>) endpoints read client-supplied paths that must resolve
/// inside the upload directory. These check the confinement at both call sites
/// and in the shared <see cref="UploadPathGuard"/>: paths outside the root —
/// including a sibling directory that merely shares the root's name — are
/// rejected.
/// </summary>
public class UploadRootConfinementTests : IDisposable
{
    private readonly string _baseDir;
    private readonly string _uploadRoot;

    public UploadRootConfinementTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-upload-confine-" + Guid.NewGuid().ToString("N"));
        _uploadRoot = Path.Combine(_baseDir, "uploads");
        Directory.CreateDirectory(_uploadRoot);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    // ---- UploadPathGuard ---------------------------------------------------

    [Fact]
    public void Guard_PathInsideRoot_IsAccepted()
    {
        Assert.True(UploadPathGuard.IsInsideDirectory(_uploadRoot, Path.Combine(_uploadRoot, "img.png")));
        Assert.True(UploadPathGuard.IsInsideDirectory(_uploadRoot, Path.Combine(_uploadRoot, "nested", "img.png")));
    }

    [Fact]
    public void Guard_SiblingDirectorySharingPrefix_IsRejected()
    {
        Assert.False(UploadPathGuard.IsInsideDirectory(_uploadRoot, Path.Combine(_baseDir, "uploads-evil", "img.png")));
    }

    [Fact]
    public void Guard_PathOutsideRoot_IsRejected()
    {
        Assert.False(UploadPathGuard.IsInsideDirectory(_uploadRoot, Path.Combine(_baseDir, "secret.png")));
        Assert.False(UploadPathGuard.IsInsideDirectory(_uploadRoot, _baseDir));
    }

    [Fact]
    public void Guard_TraversalOutOfRoot_IsRejected()
    {
        string traversal = Path.GetFullPath(Path.Combine(_uploadRoot, "..", "secret.png"));
        Assert.False(UploadPathGuard.IsInsideDirectory(_uploadRoot, traversal));
    }

    // ---- Wan video imagePath ----------------------------------------------

    [Fact]
    public void WanParser_ImagePathInsideRoot_IsRead()
    {
        string inside = Path.Combine(_uploadRoot, "cond.png");
        File.WriteAllBytes(inside, new byte[] { 1, 2, 3 });

        var p = WanVideoParamsParser.Parse(VideoRequest(inside), Options(), out string error);

        Assert.Null(error);
        Assert.Equal(new byte[] { 1, 2, 3 }, p.ImageBytes);
    }

    [Fact]
    public void WanParser_ImagePathInSiblingDirectory_IsRejected()
    {
        string sibling = Path.Combine(_baseDir, "uploads-evil");
        Directory.CreateDirectory(sibling);
        string outside = Path.Combine(sibling, "cond.png");
        File.WriteAllBytes(outside, new byte[] { 1, 2, 3 });

        var p = WanVideoParamsParser.Parse(VideoRequest(outside), Options(), out string error);

        Assert.NotNull(error);
        Assert.Null(p.ImageBytes);
    }

    [Fact]
    public void WanParser_ImagePathOutsideRoot_IsRejected()
    {
        string outside = Path.Combine(_baseDir, "secret.png");
        File.WriteAllBytes(outside, new byte[] { 1, 2, 3 });

        WanVideoParamsParser.Parse(VideoRequest(outside), Options(), out string error);

        Assert.NotNull(error);
    }

    // ---- Image-edit imagePath(s) -------------------------------------------

    [Fact]
    public async Task ImageEdit_PathInsideRoot_IsRead()
    {
        string inside = Path.Combine(_uploadRoot, "edit.png");
        File.WriteAllBytes(inside, new byte[] { 4, 5, 6 });
        var images = new List<byte[]>();

        string error = await Adapter().ReadUploadedImagesAsync(
            EditRequest(inside), images, CancellationToken.None);

        Assert.Null(error);
        Assert.Single(images);
        Assert.Equal(new byte[] { 4, 5, 6 }, images[0]);
    }

    [Fact]
    public async Task ImageEdit_PathInSiblingDirectory_IsRejected()
    {
        string sibling = Path.Combine(_baseDir, "uploads-evil");
        Directory.CreateDirectory(sibling);
        string outside = Path.Combine(sibling, "edit.png");
        File.WriteAllBytes(outside, new byte[] { 4, 5, 6 });
        var images = new List<byte[]>();

        string error = await Adapter().ReadUploadedImagesAsync(
            EditRequest(outside), images, CancellationToken.None);

        Assert.NotNull(error);
        Assert.Empty(images);
    }

    [Fact]
    public async Task ImageEdit_PathOutsideRootInImagePathsArray_IsRejected()
    {
        string inside = Path.Combine(_uploadRoot, "ok.png");
        File.WriteAllBytes(inside, new byte[] { 7 });
        string outside = Path.Combine(_baseDir, "secret.png");
        File.WriteAllBytes(outside, new byte[] { 8 });
        var images = new List<byte[]>();

        var request = JsonSerializer.Deserialize<JsonElement>(
            JsonSerializer.Serialize(new { imagePaths = new[] { inside, outside } }));
        string error = await Adapter().ReadUploadedImagesAsync(request, images, CancellationToken.None);

        Assert.NotNull(error);
    }

    // ---- helpers -----------------------------------------------------------

    private ServerHostingOptions Options() => new(
        startupModelPath: null,
        startupMmProjPath: null,
        defaultBackend: "ggml_cpu",
        supportedBackends: null,
        defaultMaxTokens: 100,
        maxTokensPinned: false,
        defaultWanVideoFrames: 0,
        defaultWanVideoFps: 0,
        uploadDirectory: _uploadRoot,
        logDirectory: Path.Combine(_baseDir, "logs"),
        fileLoggingEnabled: false,
        samplingDefaults: null);

    private WebUiAdapter Adapter() => new(
        new ModelService(),
        new InferenceQueue(),
        new SessionManager(),
        Options(),
        new UploadStoragePolicy(_uploadRoot),
        NullLoggerFactory.Instance);

    private static JsonElement VideoRequest(string imagePath) =>
        JsonSerializer.Deserialize<JsonElement>(
            JsonSerializer.Serialize(new { prompt = "a cat", imagePath }));

    private static JsonElement EditRequest(string imagePath) =>
        JsonSerializer.Deserialize<JsonElement>(
            JsonSerializer.Serialize(new { imagePath }));
}
