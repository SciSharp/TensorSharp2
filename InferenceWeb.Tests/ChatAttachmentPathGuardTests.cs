using TensorSharp.Runtime;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Web UI /api/chat attachment paths (imagePaths/audioPaths/textFilePaths)
/// arrive as client-supplied absolute paths that the multimodal injector later
/// opens, so every one must resolve inside the upload directory. These check
/// that paths outside it — including a sibling directory that shares the root's
/// name — are rejected.
/// </summary>
public class ChatAttachmentPathGuardTests : IDisposable
{
    private readonly string _baseDir;
    private readonly string _uploadRoot;

    public ChatAttachmentPathGuardTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-chat-path-guard-" + Guid.NewGuid().ToString("N"));
        _uploadRoot = Path.Combine(_baseDir, "uploads");
        Directory.CreateDirectory(_uploadRoot);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    private static List<ChatMessage> MessageWithImage(string path) =>
        new() { new ChatMessage { Role = "user", Content = "hi", ImagePaths = new List<string> { path } } };

    [Fact]
    public void PathInsideUploadRoot_IsAccepted()
    {
        string inside = Path.Combine(_uploadRoot, "img.png");
        Assert.Null(ChatMessageParser.ValidateAttachmentPaths(MessageWithImage(inside), _uploadRoot));
    }

    [Fact]
    public void AbsolutePathOutsideUploadRoot_IsRejected()
    {
        string secret = Path.Combine(_baseDir, "secret.png");
        File.WriteAllText(secret, "top secret");

        Assert.NotNull(ChatMessageParser.ValidateAttachmentPaths(MessageWithImage(secret), _uploadRoot));
    }

    [Fact]
    public void TraversalOutOfUploadRoot_IsRejected()
    {
        string traversal = Path.Combine(_uploadRoot, "..", "secret.png");
        Assert.NotNull(ChatMessageParser.ValidateAttachmentPaths(MessageWithImage(traversal), _uploadRoot));
    }

    [Fact]
    public void SiblingDirectorySharingPrefix_IsRejected()
    {
        string sibling = Path.Combine(_baseDir, "uploads-evil");
        Directory.CreateDirectory(sibling);
        string path = Path.Combine(sibling, "img.png");

        Assert.NotNull(ChatMessageParser.ValidateAttachmentPaths(MessageWithImage(path), _uploadRoot));
    }

    [Fact]
    public void AudioAndTextFilePaths_AreValidatedToo()
    {
        string outside = Path.Combine(_baseDir, "outside.wav");
        var audio = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "hi", AudioPaths = new List<string> { outside } }
        };
        var text = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "hi", TextFilePaths = new List<string> { outside } }
        };

        Assert.NotNull(ChatMessageParser.ValidateAttachmentPaths(audio, _uploadRoot));
        Assert.NotNull(ChatMessageParser.ValidateAttachmentPaths(text, _uploadRoot));
    }

    [Fact]
    public void NullOrEmptyPathEntry_IsRejected()
    {
        Assert.NotNull(ChatMessageParser.ValidateAttachmentPaths(MessageWithImage(null), _uploadRoot));
        Assert.NotNull(ChatMessageParser.ValidateAttachmentPaths(MessageWithImage("  "), _uploadRoot));
    }

    [Fact]
    public void MessagesWithoutAttachments_AreAccepted()
    {
        var plain = new List<ChatMessage> { new ChatMessage { Role = "user", Content = "hi" } };
        Assert.Null(ChatMessageParser.ValidateAttachmentPaths(plain, _uploadRoot));
        Assert.Null(ChatMessageParser.ValidateAttachmentPaths(null, _uploadRoot));
    }
}
