// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The diffusion scheduler now lets Jev reads (and image encodes) run between the forwards of a
// denoising block instead of after it. That is only acceptable if it is invisible: the block's tokens
// and the reads' answers must be exactly what each produces alone. The model tests check that on the
// cpu backend and on the pinned GGML backend against the real GGUF (opt-in via TS_TEST_MODEL_DIR);
// the span-scope tests pin the no-op scope that keeps a text-only unified forward from invalidating
// the span-keyed caches.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public sealed class DiffusionGemmaInterleaveTests(Xunit.Abstractions.ITestOutputHelper output)
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string GgufPattern = "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion";

    private static int[] Render(DiffusionGemmaModel model, string text)
        => model.Tokenizer.Encode(new GgufPromptRenderer().Render(model.Config.ChatTemplate,
            [new ChatMessage { Role = "user", Content = text }], addGenerationPrompt: true,
            architecture: model.Config.Architecture), addSpecial: true).ToArray();

    private static (List<int> Tokens, List<int[]> Previews) RunBlock(DiffusionGemmaModel model, int[] prompt,
        DiffusionEbParams p, Action beforeForward)
    {
        var previews = new List<int[]>();
        DiffusionSeqState state = model.CreateSeqState();
        try
        {
            var run = new DiffusionSeqRun(prompt, p, state, CancellationToken.None,
                (_, _, _, preview) => previews.Add(preview));
            new DiffusionGemmaSampler(model).RunBlockBatched([run], default, beforeForward);
            return (run.Response, previews);
        }
        finally { model.DisposeSeqState(state); }
    }

    // Each forward of a chat block is preceded by what a waiting job would do there: a Jev structured
    // read, plus a canvas forward of another prompt that overwrites the pooled logits buffer (the worst
    // a job could do to state shared with the block). Checked with prompt-KV caching (the cpu default)
    // and with the unified forward (DIFFUSION_NO_PKV / ggml_cpu's path, and the span scope).
    [ModelFact(EnvModelDir, GgufPattern)]
    public void JobsBetweenForwards_ChangeNeitherTheBlockNorTheReads()
        => CheckJobsBetweenForwards(BackendType.Cpu);

    // The same on the GGML backend this process pins (TS_TEST_GGML_BACKEND, ggml_cpu by default):
    // the unified forward on ggml_cpu, the device-glue prompt-KV decode on the GPU backends.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void JobsBetweenForwards_OnThePinnedGgmlBackend_ChangeNothing()
        => CheckJobsBetweenForwards(TestGates.PinnedGgmlBackend);

    private void CheckJobsBetweenForwards(BackendType backend)
    {
        using var model = (DiffusionGemmaModel)ModelBase.Create(
            TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvModelDir), GgufPattern), backend);
        bool defaultPkv = model.SupportsPromptKvCache;
        if (backend == BackendType.Cpu)
            Assert.True(defaultPkv, "prompt-KV caching should be on by default on the cpu backend");
        int[] chat = Render(model, "Name the three primary colors and one use of each.");
        int[] jev = Render(model, "Classify: I love this product and would buy it again. Answer A for positive, B for negative.");
        int[] other = Render(model, "Write one sentence about the sea.");
        int[] labels = new[] { "A", "B", "C" }.Select(l => model.Tokenizer.Encode(l, false).Single()).ToArray();
        int[] readCanvas = Enumerable.Range(0, 16).Select(i => i % 3 == 0 ? model.MaskTokenId : labels[i % 3]).ToArray();
        int[] positions = [1, 15];
        int[][] rows = [labels, [labels[1], labels[0]]];
        int[] otherCanvas = Enumerable.Range(0, model.CanvasLength).Select(i => 1000 + i).ToArray();
        var p = new DiffusionEbParams
        {
            MaxDenoisingSteps = 3, Seed = 11, MaxBlocks = 1,
            ConfidenceThreshold = -1f, StabilityThreshold = int.MaxValue,
        };

        List<int>? previousTokens = null;
        foreach (bool pkv in defaultPkv ? new[] { true, false } : new[] { false })
        {
            model.SupportsPromptKvCache = pkv;
            model.ClearStructuredCache();
            DiffusionSeqState otherState = model.CreateSeqState();
            try
            {
                if (pkv) model.PrefillSeq(otherState, other);
                model.ReadStructured(jev, readCanvas, positions, rows);   // prefills the read's prompt cache
                float[][] idle = model.ReadStructured(jev, readCanvas, positions, rows);
                var alone = RunBlock(model, chat, p, null);

                var reads = new List<float[][]>();
                var interleaved = RunBlock(model, chat, p, () =>
                {
                    reads.Add(model.ReadStructured(jev, readCanvas, positions, rows));
                    if (pkv) model.DecodeCanvasSeq(otherState, otherCanvas, null, 0f, 1f);
                    else model.ForwardCanvas([.. other, .. otherCanvas], other.Length);
                });

                output.WriteLine($"{backend} pkv={pkv}: {reads.Count} interleaved jobs, {alone.Previews.Count} steps, " +
                    $"tokens [{string.Join(",", alone.Tokens.Take(12))}...], p(A)={idle[0][0]:R}");
                Assert.Equal(pkv ? 1 + p.MaxDenoisingSteps : p.MaxDenoisingSteps, reads.Count);
                Assert.Equal(alone.Previews.Count, interleaved.Previews.Count);
                for (int s = 0; s < alone.Previews.Count; s++) Assert.Equal(alone.Previews[s], interleaved.Previews[s]);
                Assert.Equal(alone.Tokens, interleaved.Tokens);
                foreach (float[][] read in reads)
                    for (int r = 0; r < rows.Length; r++) Assert.Equal(idle[r], read[r]);

                // On cpu the cached canvas decode is the unified forward bit for bit, so both passes agree.
                if (backend == BackendType.Cpu && previousTokens != null) Assert.Equal(previousTokens, alone.Tokens);
                previousTokens = alone.Tokens;
            }
            finally
            {
                model.DisposeSeqState(otherState);
                model.SupportsPromptKvCache = defaultPkv;
            }
        }
    }

    // ---- the span scope (no weights) ---------------------------------------------------------

    private const int Hidden = 8;

    private static DiffusionGemmaModel SpanOnlyModel()
    {
        var model = (DiffusionGemmaModel)RuntimeHelpers.GetUninitializedObject(typeof(DiffusionGemmaModel));
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new ModelConfig { HiddenSize = Hidden });
        Field("_ownedVisionEmbeddingsList").SetValue(model, new List<(Tensor Embeddings, int Position)>());
        Field("_visionSpans").SetValue(model, Array.Empty<(int Start, int Length)>());
        return model;
    }

    private static FieldInfo Field(string name)
        => typeof(DiffusionGemmaModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static int Version(DiffusionGemmaModel model) => (int)Field("_visionSpanVersion").GetValue(model)!;
    private static (int Start, int Length)[] ActiveSpans(DiffusionGemmaModel model)
        => ((int Start, int Length)[])Field("_visionSpans").GetValue(model)!;

    private static Tensor Rows(int rows)
    {
        var t = new Tensor(new CpuAllocator(BlasEnum.DotNet), DType.Float32, rows, Hidden);
        t.SetElementsAsFloat(new float[rows * Hidden]);
        return t;
    }

    [Fact]
    public void TextOnlySequence_OnAModelWithoutSpans_LeavesTheSpanVersionAlone()
    {
        DiffusionGemmaModel model = SpanOnlyModel();
        var seq = model.CreateSeqState();
        int before = Version(model);
        using (model.UseSequenceVision(seq))
        {
            Assert.Equal(before, Version(model));
            Assert.False(model.HasPendingVisionEmbeddings);
            Assert.Empty(ActiveSpans(model));
        }
        Assert.Equal(before, Version(model));
    }

    [Fact]
    public void SequenceWithSpans_IsScopedAndInvalidatesTheMaskKey()
    {
        DiffusionGemmaModel model = SpanOnlyModel();
        var seq = model.CreateSeqState();
        model.SetSequenceVisionEmbeddings(seq, Rows(4), insertPosition: 3);
        int before = Version(model);
        using (model.UseSequenceVision(seq))
        {
            Assert.True(Version(model) > before);
            Assert.True(model.HasPendingVisionEmbeddings);
            Assert.Equal(new[] { (3, 4) }, ActiveSpans(model));
        }
        Assert.True(Version(model) > before);
        Assert.False(model.HasPendingVisionEmbeddings);
        Assert.Empty(ActiveSpans(model));
        model.DisposeSeqState(seq);
    }

    // Model-level spans belong to a single-request reader (a Jev read with images). A text-only
    // sequence forwarded meanwhile must not see them, so its scope still has to swap them out.
    [Fact]
    public void TextOnlySequence_OnAModelWithSpans_HidesThem()
    {
        DiffusionGemmaModel model = SpanOnlyModel();
        model.SetVisionEmbeddings(Rows(5), insertPosition: 2);
        var seq = model.CreateSeqState();
        int before = Version(model);
        using (model.UseSequenceVision(seq))
        {
            Assert.True(Version(model) > before);
            Assert.False(model.HasPendingVisionEmbeddings);
            Assert.Empty(ActiveSpans(model));
        }
        Assert.True(model.HasPendingVisionEmbeddings);
        Assert.Equal(new[] { (2, 5) }, ActiveSpans(model));
        model.ClearVisionEmbeddings();
    }
}
