using TensorSharp.Server;

namespace InferenceWeb.Tests;

/// <summary>
/// The generation reserve (max new tokens) is clamped to the context room the
/// prompt leaves, so a large default reserve on a small-context model still
/// admits a short prompt while a genuine overflow is left to the caller.
/// </summary>
public class ClampGenerationReserveTests
{
    [Fact]
    public void OversizedReserve_ShrinksToRoomLeftByPrompt()
    {
        // 100-token prompt on a 4096 context with the 20000 default keeps a
        // generous reserve: the remaining 3996 tokens.
        Assert.Equal(3996, ChatGenerationPipeline.ClampGenerationReserve(20000, 100, 4096));
    }

    [Fact]
    public void ReserveThatAlreadyFits_IsUnchanged()
    {
        Assert.Equal(200, ChatGenerationPipeline.ClampGenerationReserve(200, 100, 4096));
    }

    [Fact]
    public void ReserveExactlyFillingContext_IsUnchanged()
    {
        Assert.Equal(3996, ChatGenerationPipeline.ClampGenerationReserve(3996, 100, 4096));
    }

    [Fact]
    public void PromptAloneOverflowsContext_ReserveFloorsAtOne()
    {
        Assert.Equal(1, ChatGenerationPipeline.ClampGenerationReserve(20000, 5000, 4096));
    }

    [Fact]
    public void PromptEqualToContext_ReserveFloorsAtOne()
    {
        Assert.Equal(1, ChatGenerationPipeline.ClampGenerationReserve(20000, 4096, 4096));
    }

    [Fact]
    public void UnknownContext_LeavesReserveUntouched()
    {
        Assert.Equal(20000, ChatGenerationPipeline.ClampGenerationReserve(20000, 100, 0));
    }

    [Fact]
    public void LargeContext_LeavesGenerousReserveUntouched()
    {
        // 32k context, 20000 default, short prompt: nothing to clamp.
        Assert.Equal(20000, ChatGenerationPipeline.ClampGenerationReserve(20000, 500, 32768));
    }
}
