// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// End-to-end contracts of DiffusionGemma's pure-C# (cpu backend) forward against a real GGUF.
// Opt-in via TS_TEST_MODEL_DIR (the 16 GB checkpoint), always on BackendType.Cpu regardless of
// TS_TEST_BACKEND: prompt-KV caching on this backend is only correct if the canvas decode
// reproduces the unified [prompt|canvas] forward BIT FOR BIT - Q8 activation quantization and top-8
// routing turn any last-bit difference into a visibly different answer - so these assert equality.
namespace InferenceWeb.Tests;

public sealed class DiffusionGemmaCpuForwardTests(Xunit.Abstractions.ITestOutputHelper output)
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string GgufPattern = "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion";

    private static DiffusionGemmaModel Load()
        => (DiffusionGemmaModel)ModelBase.Create(
            TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvModelDir), GgufPattern), BackendType.Cpu);

    // Runs the unified [prompt|canvas] forward, then restores whatever prompt-KV setting the model had.
    private static T WithoutPromptKv<T>(DiffusionGemmaModel model, Func<T> run)
    {
        bool saved = model.SupportsPromptKvCache;
        model.SupportsPromptKvCache = false;
        try { return run(); }
        finally { model.SupportsPromptKvCache = saved; }
    }

    private static int[] Render(DiffusionGemmaModel model, string text)
        => model.Tokenizer.Encode(new GgufPromptRenderer().Render(model.Config.ChatTemplate,
            [new ChatMessage { Role = "user", Content = text }], addGenerationPrompt: true,
            architecture: model.Config.Architecture), addSpecial: true).ToArray();

    [ModelFact(EnvModelDir, GgufPattern)]
    public void StructuredRead_PromptKv_IsBitwiseTheUnifiedForward()
    {
        using var model = Load();
        Assert.True(model.SupportsPromptKvCache, "prompt-KV caching should be on by default on the cpu backend");
        int[] prompt = Render(model, "Classify: I love this product and would buy it again. Answer A for positive, B for negative.");
        int[] labels = new[] { "A", "B", "C" }.Select(l => model.Tokenizer.Encode(l, false).Single()).ToArray();
        foreach (int width in new[] { 7, 16, 33 })
        {
            int[] canvas = Enumerable.Range(0, width).Select(i => i % 3 == 0 ? model.MaskTokenId : labels[i % 3]).ToArray();
            int[] positions = [1, width - 1];
            int[][] rows = [labels, [labels[1], labels[0]]];
            model.ClearStructuredCache();
            float[][] cached = model.ReadStructured(prompt, canvas, positions, rows);
            float[][] cachedAgain = model.ReadStructured(prompt, canvas, positions, rows);   // reuses the prompt K/V
            float[][] unified = WithoutPromptKv(model, () => model.ReadStructured(prompt, canvas, positions, rows));
            output.WriteLine($"width={width}: cached=[{string.Join(",", cached[0])}] unified=[{string.Join(",", unified[0])}]");
            for (int r = 0; r < rows.Length; r++)
            {
                Assert.Equal(unified[r], cached[r]);
                Assert.Equal(cached[r], cachedAgain[r]);
            }
        }
    }

    // A prompt longer than the sliding window: the prefill keeps only the last (swa-1) prompt rows of
    // each local layer and the canvas decode addresses them from row 0 (UniformLo), while global
    // layers keep all P rows. An off-by-one there shifts which prompt keys the canvas sees without
    // failing anything else, so the cached read must still be the unified forward bit for bit.
    // ~1.1k prompt tokens and a 7-token canvas keep it to a few minutes on the cpu backend.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void StructuredRead_PromptKv_LongPromptPastSlidingWindow_IsBitwiseTheUnifiedForward()
    {
        using var model = Load();
        Assert.True(model.SupportsPromptKvCache, "prompt-KV caching should be on by default on the cpu backend");
        const int swa = 1024;   // diffusiongemma-26B-A4B's sliding window
        const string question = "\nClassify the final note: I love this product. Answer A for positive, B for negative.";
        var filler = new System.Text.StringBuilder();
        int[] prompt = Render(model, question);
        for (int i = 0; prompt.Length < swa + 64; i++)   // just past the window: the cost is linear in P
        {
            for (int j = 0; j < 8; j++) filler.Append($"note {i * 8 + j}: the shipment was logged. ");
            prompt = Render(model, filler + question);
        }
        output.WriteLine($"prompt tokens: {prompt.Length}");
        Assert.InRange(prompt.Length, swa + 64, 2 * swa);
        int[] labels = new[] { "A", "B", "C" }.Select(l => model.Tokenizer.Encode(l, false).Single()).ToArray();
        int[] canvas = [labels[0], model.MaskTokenId, labels[1], model.MaskTokenId, labels[2], model.MaskTokenId, labels[0]];
        int[] positions = [1, 5];
        int[][] rows = [labels, [labels[1], labels[0]]];
        model.ClearStructuredCache();
        float[][] cached = model.ReadStructured(prompt, canvas, positions, rows);
        float[][] unified = WithoutPromptKv(model, () => model.ReadStructured(prompt, canvas, positions, rows));
        output.WriteLine($"cached=[{string.Join(",", cached[0])}] unified=[{string.Join(",", unified[0])}]");
        for (int r = 0; r < rows.Length; r++) Assert.Equal(unified[r], cached[r]);
    }

    // DecodeCanvasBatched on the cpu backend runs every sequence's canvas rows through one forward;
    // each sequence's logits must equal its own single-sequence decode exactly.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void BatchedCanvasDecode_IsBitwiseThePerSequenceDecode()
    {
        using var model = Load();
        model.SelfConditioningEnabled = false;
        DiffusionSeqState a = model.CreateSeqState(), b = model.CreateSeqState();
        try
        {
            model.PrefillSeq(a, Render(model, "List three primary colors."));
            model.PrefillSeq(b, Render(model, "Explain in one sentence why the sky is blue."));
            int[] canvasA = Enumerable.Repeat(model.MaskTokenId, 6).ToArray();
            int[] canvasB = Enumerable.Range(0, 9).Select(i => i + 1000).ToArray();
            float[] soloA = (float[])model.DecodeCanvasSeq(a, canvasA, null, 0f, 1f).Clone();
            float[] soloB = (float[])model.DecodeCanvasSeq(b, canvasB, null, 0f, 1f).Clone();
            float[][] batched = model.DecodeCanvasBatched([a, b], [canvasA, canvasB], new float[2][], [0f, 0f], [1f, 1f]);
            Assert.Equal(soloA, batched[0]);
            Assert.Equal(soloB, batched[1]);
        }
        finally
        {
            model.DisposeSeqState(a);
            model.DisposeSeqState(b);
        }
    }

    // The sampler with prompt-KV caching (prefill once, canvas-only steps) against the unified forward
    // every step: same seed, same token stream. Two fixed steps keep it affordable on the cpu backend.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void Generate_PromptKv_MatchesUnifiedForward()
    {
        using var model = Load();
        int[] prompt = Render(model, "What is the capital of France? Answer in one short sentence.");
        var sampler = new DiffusionGemmaSampler(model);
        var p = new DiffusionEbParams
        {
            MaxDenoisingSteps = 2, Seed = 3, MaxBlocks = 1,
            ConfidenceThreshold = -1f, StabilityThreshold = int.MaxValue,
        };
        var stepsCached = new List<int[]>();
        var stepsUnified = new List<int[]>();
        List<int> cached = sampler.Generate(prompt, p, (_, _, _, canvas) => stepsCached.Add(canvas));
        List<int> unified = WithoutPromptKv(model, () => sampler.Generate(prompt, p, (_, _, _, canvas) => stepsUnified.Add(canvas)));
        Assert.Equal(stepsUnified.Count, stepsCached.Count);
        for (int s = 0; s < stepsCached.Count; s++) Assert.Equal(stepsUnified[s], stepsCached[s]);
        Assert.Equal(unified, cached);
    }
}
