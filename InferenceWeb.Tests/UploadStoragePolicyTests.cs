using System;
using System.IO;
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for <see cref="UploadStoragePolicy"/> (per-file cap, quota
/// reservation, TTL cleanup) and for its enforcement inside the base64
/// materialisation path of <see cref="ChatMessageParser"/>.
/// </summary>
public class UploadStoragePolicyTests : IDisposable
{
    private readonly string _dir;

    public UploadStoragePolicyTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-upload-policy-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private string WriteFile(string name, int bytes)
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, new byte[bytes]);
        return path;
    }

    [Fact]
    public void Constructor_ScansExistingFilesIntoUsedBytes()
    {
        WriteFile("a.png", 100);
        WriteFile("b.mp4", 250);

        var policy = new UploadStoragePolicy(_dir);

        Assert.Equal(350, policy.UsedBytes);
    }

    [Fact]
    public void Defaults_ArePermissive()
    {
        var policy = new UploadStoragePolicy(_dir);

        Assert.Equal(500L * 1024 * 1024, policy.MaxFileBytes);
        Assert.False(policy.QuotaEnabled);
        Assert.Null(policy.Ttl);
        Assert.True(policy.TryReserveClientWrite(100L * 1024 * 1024, out _, out _));
        Assert.True(policy.HasQuotaHeadroom(out _));
    }

    [Fact]
    public void TryReserveClientWrite_FileOverCap_Rejects413()
    {
        var policy = new UploadStoragePolicy(_dir, maxFileBytes: 1024);

        Assert.False(policy.TryReserveClientWrite(2048, out string error, out int status));
        Assert.Equal(413, status);
        Assert.Contains("too large", error);
        // A rejected file must not consume quota.
        Assert.Equal(0, policy.UsedBytes);
    }

    [Fact]
    public void TryReserveClientWrite_QuotaExhausted_Rejects507()
    {
        var policy = new UploadStoragePolicy(_dir, quotaBytes: 1000);

        Assert.True(policy.TryReserveClientWrite(800, out _, out _));
        Assert.False(policy.TryReserveClientWrite(300, out string error, out int status));
        Assert.Equal(507, status);
        Assert.Contains("quota", error);
        // The failed attempt must not have reserved anything.
        Assert.Equal(800, policy.UsedBytes);
        // A write that still fits is admitted.
        Assert.True(policy.TryReserveClientWrite(200, out _, out _));
    }

    [Fact]
    public void Release_ReturnsReservedBytes()
    {
        var policy = new UploadStoragePolicy(_dir, quotaBytes: 1000);

        Assert.True(policy.TryReserveClientWrite(900, out _, out _));
        policy.Release(900);

        Assert.Equal(0, policy.UsedBytes);
        Assert.True(policy.TryReserveClientWrite(900, out _, out _));
    }

    [Fact]
    public void HasQuotaHeadroom_FalseOnceQuotaReached()
    {
        WriteFile("existing.bin", 1000);
        var policy = new UploadStoragePolicy(_dir, quotaBytes: 1000);

        Assert.False(policy.HasQuotaHeadroom(out string error));
        Assert.Contains("quota", error);
    }

    [Fact]
    public void RecordFile_AddsActualSize_IgnoresMissing()
    {
        var policy = new UploadStoragePolicy(_dir);
        string path = WriteFile("generated.png", 640);

        policy.RecordFile(path);
        policy.RecordFile(Path.Combine(_dir, "missing.png"));

        Assert.Equal(640, policy.UsedBytes);
    }

    [Fact]
    public void CleanupExpired_DeletesOnlyFilesOlderThanTtl_AndUpdatesUsedBytes()
    {
        string oldFile = WriteFile("old.png", 300);
        string newFile = WriteFile("new.png", 200);
        File.SetLastWriteTimeUtc(oldFile, DateTime.UtcNow.AddHours(-3));

        var policy = new UploadStoragePolicy(_dir, ttl: TimeSpan.FromHours(1));
        int deleted = policy.CleanupExpired(out long freed);

        Assert.Equal(1, deleted);
        Assert.Equal(300, freed);
        Assert.False(File.Exists(oldFile));
        Assert.True(File.Exists(newFile));
        Assert.Equal(200, policy.UsedBytes);
    }

    [Fact]
    public void CleanupExpired_NoTtl_IsNoOp()
    {
        string oldFile = WriteFile("old.png", 300);
        File.SetLastWriteTimeUtc(oldFile, DateTime.UtcNow.AddYears(-1));

        var policy = new UploadStoragePolicy(_dir);

        Assert.Equal(0, policy.CleanupExpired(out long freed));
        Assert.Equal(0, freed);
        Assert.True(File.Exists(oldFile));
    }

    // ---- Enforcement inside the chat parsers ------------------------------

    private static JsonDocument OllamaImagesBody(int imageBytes)
    {
        string b64 = Convert.ToBase64String(new byte[imageBytes]);
        return JsonDocument.Parse($$"""{"images": ["{{b64}}"]}""");
    }

    [Fact]
    public void DecodeBase64Images_OverPerFileCap_ThrowsWith413()
    {
        var policy = new UploadStoragePolicy(_dir, maxFileBytes: 16);
        using var doc = OllamaImagesBody(imageBytes: 64);

        var ex = Assert.Throws<UploadLimitExceededException>(
            () => ChatMessageParser.DecodeBase64Images(doc.RootElement, policy));
        Assert.Equal(413, ex.StatusCode);
        Assert.Empty(Directory.GetFiles(_dir));
    }

    [Fact]
    public void DecodeBase64Images_QuotaExhausted_ThrowsWith507()
    {
        WriteFile("existing.bin", 100);
        var policy = new UploadStoragePolicy(_dir, quotaBytes: 110);
        using var doc = OllamaImagesBody(imageBytes: 64);

        var ex = Assert.Throws<UploadLimitExceededException>(
            () => ChatMessageParser.DecodeBase64Images(doc.RootElement, policy));
        Assert.Equal(507, ex.StatusCode);
    }

    [Fact]
    public void DecodeBase64Images_WithinLimits_WritesAndCountsBytes()
    {
        var policy = new UploadStoragePolicy(_dir, maxFileBytes: 1024, quotaBytes: 1024);
        using var doc = OllamaImagesBody(imageBytes: 64);

        var paths = ChatMessageParser.DecodeBase64Images(doc.RootElement, policy);

        string path = Assert.Single(paths);
        Assert.True(File.Exists(path));
        Assert.Equal(64, policy.UsedBytes);
    }
}
