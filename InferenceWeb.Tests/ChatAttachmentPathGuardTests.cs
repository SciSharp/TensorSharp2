using TensorSharp.Runtime;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Web UI /api/chat attachment references (imagePaths/audioPaths/textFilePaths)
/// are resolved by <see cref="ChatMessageParser.ResolveAttachmentPaths"/>: the
/// canonical form is the bare server filename from /api/upload, while legacy
/// absolute paths are accepted only when they resolve inside the upload
/// directory. These check both forms, and that paths outside the root —
/// including a sibling directory sharing the root's name — are rejected.
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
    public void BareFileName_ResolvesUnderUploadRoot()
    {
        var messages = MessageWithImage("img.png");

        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(messages, _uploadRoot));
        Assert.Equal(Path.Combine(Path.GetFullPath(_uploadRoot), "img.png"), messages[0].ImagePaths[0]);
    }

    [Fact]
    public void PathInsideUploadRoot_IsAccepted()
    {
        string inside = Path.Combine(_uploadRoot, "img.png");
        var messages = MessageWithImage(inside);

        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(messages, _uploadRoot));
        Assert.Equal(Path.GetFullPath(inside), messages[0].ImagePaths[0]);
    }

    [Fact]
    public void AbsolutePathOutsideUploadRoot_IsRejected()
    {
        string secret = Path.Combine(_baseDir, "secret.png");
        File.WriteAllText(secret, "top secret");

        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(MessageWithImage(secret), _uploadRoot));
    }

    [Fact]
    public void TraversalOutOfUploadRoot_IsRejected()
    {
        string traversal = Path.Combine(_uploadRoot, "..", "secret.png");
        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(MessageWithImage(traversal), _uploadRoot));
    }

    [Fact]
    public void DotDotFileName_IsRejected()
    {
        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(MessageWithImage(".."), _uploadRoot));
        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(MessageWithImage("."), _uploadRoot));
    }

    [Fact]
    public void SiblingDirectorySharingPrefix_IsRejected()
    {
        string sibling = Path.Combine(_baseDir, "uploads-evil");
        Directory.CreateDirectory(sibling);
        string path = Path.Combine(sibling, "img.png");

        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(MessageWithImage(path), _uploadRoot));
    }

    [Fact]
    public void AudioAndTextFileReferences_AreResolvedToo()
    {
        string outside = Path.Combine(_baseDir, "outside.wav");
        var audio = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "hi", AudioPaths = new List<string> { outside } }
        };
        var text = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "hi", TextFilePaths = new List<string> { "doc.txt" } }
        };

        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(audio, _uploadRoot));
        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(text, _uploadRoot));
        Assert.Equal(Path.Combine(Path.GetFullPath(_uploadRoot), "doc.txt"), text[0].TextFilePaths[0]);
    }

    [Fact]
    public void NullOrEmptyPathEntry_IsRejected()
    {
        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(MessageWithImage(null), _uploadRoot));
        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(MessageWithImage("  "), _uploadRoot));
    }

    [Fact]
    public void MessagesWithoutAttachments_AreAccepted()
    {
        var plain = new List<ChatMessage> { new ChatMessage { Role = "user", Content = "hi" } };
        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(plain, _uploadRoot));
        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(null, _uploadRoot));
    }
}
