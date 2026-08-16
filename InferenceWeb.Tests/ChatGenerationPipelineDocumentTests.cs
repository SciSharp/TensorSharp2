namespace InferenceWeb.Tests;

public class ChatGenerationPipelineDocumentTests
{
    [Fact]
    public void RejectAttachedDocumentOverflow_AllowsCompleteDocumentThatFits()
    {
        ChatGenerationPipeline.RejectAttachedDocumentOverflow(
            promptTokens: 68_847,
            maxTokens: 4_096,
            modelContextLimit: 131_072,
            preserveAllInput: true);
    }

    [Fact]
    public void RejectAttachedDocumentOverflow_RejectsInsteadOfSilentlyTruncating()
    {
        var ex = Assert.Throws<PromptContextOverflowException>(() =>
            ChatGenerationPipeline.RejectAttachedDocumentOverflow(
                promptTokens: 130_000,
                maxTokens: 4_096,
                modelContextLimit: 131_072,
                preserveAllInput: true));

        Assert.Contains("No document content was truncated", ex.Message);
        Assert.Contains("130000 prompt tokens", ex.Message);
        Assert.Contains("131072 context tokens", ex.Message);
    }

    [Fact]
    public void HasTextFileAttachments_DetectsUploadedTextOrPdfPath()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "analyze this" },
            new()
            {
                Role = "user",
                Content = "[File: book.pdf]...",
                TextFilePaths = new List<string> { "uploads/book.pdf" },
            },
        };

        Assert.True(ChatGenerationPipeline.HasTextFileAttachments(history));
        Assert.True(ChatGenerationPipeline.HasTextFileAttachments(
            new List<ChatMessage>
            {
                new()
                {
                    Role = "user",
                    Content = "[File: book.pdf]\ncomplete text\n[End of file]\nSummarize it",
                },
            }));
        Assert.False(ChatGenerationPipeline.HasTextFileAttachments(
            new List<ChatMessage> { new() { Role = "user", Content = "plain chat" } }));
    }
}
