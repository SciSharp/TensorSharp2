using System.Security.Cryptography;
using TensorAgent.Core.Catalog;

namespace TensorAgent.Tests;

/// <summary>
/// The sideload-only card path: an exact, hash-pinned artifact without a publisher URL is
/// imported from a user-selected local file and verified before the engine can see it.
/// No built-in entry needs it today, so these tests use a synthetic card.
/// </summary>
public sealed class SideloadImportTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-sideload-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best-effort test cleanup */ }
    }

    [Fact]
    public void NoBuiltInEntryIsSideloadOnly()
        => Assert.DoesNotContain(ModelCatalog.BuiltIn, model => model.SideloadOnly);

    [Fact]
    public async Task ImportStagesAndHashChecksBeforePublishingTheWeights()
    {
        byte[] expected = Enumerable.Range(0, 4096).Select(i => (byte)(i * 37)).ToArray();
        CatalogModel model = SideloadCard(expected);
        var store = new ModelStore(_root);
        string? notified = null;
        store.OnFileCreated = path => notified = path;

        await using var source = new MemoryStream(expected, writable: false);
        await store.ImportAsync(model, source);

        string destination = store.PathFor(model, model.Weights);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
        Assert.Equal(destination, store.WeightsPath(model));
        Assert.Equal(expected, await File.ReadAllBytesAsync(destination));
        Assert.Equal(destination, notified);
        Assert.Empty(Directory.EnumerateFiles(store.DirectoryFor(model), "*.import-*"));
    }

    [Fact]
    public async Task WrongLocalFileCannotReplaceAPreviouslyVerifiedImport()
    {
        byte[] expected = Enumerable.Range(0, 4096).Select(i => (byte)(i * 17)).ToArray();
        byte[] wrong = Enumerable.Range(0, expected.Length).Select(i => (byte)(255 - i)).ToArray();
        CatalogModel model = SideloadCard(expected);
        var store = new ModelStore(_root);
        Directory.CreateDirectory(store.DirectoryFor(model));
        string destination = store.PathFor(model, model.Weights);
        await File.WriteAllBytesAsync(destination, expected);

        await using var source = new MemoryStream(wrong, writable: false);
        InvalidDataException error = await Assert.ThrowsAsync<InvalidDataException>(
            () => store.ImportAsync(model, source));

        Assert.Contains("SHA-256", error.Message, StringComparison.Ordinal);
        Assert.Equal(expected, await File.ReadAllBytesAsync(destination));
        Assert.Empty(Directory.EnumerateFiles(store.DirectoryFor(model), "*.import-*"));
    }

    [Fact]
    public async Task SideloadCardCannotFallThroughToAnEmptyDownloadUrl()
    {
        CatalogModel model = SideloadCard(new byte[4096]);
        var store = new ModelStore(_root);

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => store.DownloadAsync(model, progress: null, CancellationToken.None));

        Assert.Contains("Import", error.Message, StringComparison.Ordinal);
        Assert.False(Directory.Exists(store.DirectoryFor(model)));
    }

    private static CatalogModel SideloadCard(byte[] expected) => new()
    {
        Id = "sideload-test",
        DisplayName = "Sideload test fixture",
        Family = CatalogFamily.Qwen35,
        Kind = CatalogArchitectureKind.Dense,
        Parameters = "test",
        Quantization = "test",
        Files = new[]
        {
            new CatalogFile(
                CatalogFileRole.Weights,
                "sideload-test.gguf",
                string.Empty,
                expected.LongLength,
                Convert.ToHexString(SHA256.HashData(expected)).ToLowerInvariant()),
        },
        Modalities = CatalogModalities.Text,
        MinDeviceMemoryGB = 12,
        ContextLength = 4096,
        KvCacheDtype = "q8_0",
        Sampling = new CatalogSampling(0.5f, 20, 0.85f, 0.0f),
        SupportsThinking = true,
        SideloadOnly = true,
        License = "test",
    };
}
