using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for <see cref="UploadFileReference.TryResolve"/> — the shared
/// resolver behind chat attachments, image-edit imagePath(s), and the Wan
/// image-to-video conditioning image.
/// </summary>
public class UploadFileReferenceTests : IDisposable
{
    private readonly string _baseDir;
    private readonly string _uploadRoot;

    public UploadFileReferenceTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-upload-ref-tests-" + Guid.NewGuid().ToString("N"));
        _uploadRoot = Path.Combine(_baseDir, "uploads");
        Directory.CreateDirectory(_uploadRoot);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void BareFileName_ResolvesUnderRoot()
    {
        Assert.True(UploadFileReference.TryResolve(_uploadRoot, "a1b2.png", out string full));
        Assert.Equal(Path.Combine(Path.GetFullPath(_uploadRoot), "a1b2.png"), full);
    }

    [Fact]
    public void AbsolutePathInsideRoot_IsAccepted()
    {
        string inside = Path.Combine(_uploadRoot, "a1b2.png");
        Assert.True(UploadFileReference.TryResolve(_uploadRoot, inside, out string full));
        Assert.Equal(Path.GetFullPath(inside), full);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(".")]
    [InlineData("..")]
    public void EmptyOrDotReferences_AreRejected(string reference)
    {
        Assert.False(UploadFileReference.TryResolve(_uploadRoot, reference, out _));
    }

    [Fact]
    public void RelativeTraversal_IsRejected()
    {
        Assert.False(UploadFileReference.TryResolve(_uploadRoot, Path.Combine("..", "secret.png"), out _));
        Assert.False(UploadFileReference.TryResolve(_uploadRoot, Path.Combine(_uploadRoot, "..", "secret.png"), out _));
    }

    [Fact]
    public void AbsolutePathOutsideRoot_IsRejected()
    {
        Assert.False(UploadFileReference.TryResolve(_uploadRoot, Path.Combine(_baseDir, "secret.png"), out _));
    }

    [Fact]
    public void SiblingDirectorySharingPrefix_IsRejected()
    {
        string sibling = Path.Combine(_baseDir, "uploads-evil");
        Directory.CreateDirectory(sibling);
        Assert.False(UploadFileReference.TryResolve(_uploadRoot, Path.Combine(sibling, "img.png"), out _));
    }

    [Fact]
    public void NameWithSeparator_IsTreatedAsPathNotFileName()
    {
        // "sub/x.png" is not a bare filename; it resolves against the current
        // directory, which is outside the temp upload root, so it is rejected
        // rather than silently joined under the root.
        Assert.False(UploadFileReference.TryResolve(_uploadRoot, "sub/x.png", out _));
    }
}
