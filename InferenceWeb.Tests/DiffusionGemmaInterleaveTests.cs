// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The diffusion scheduler now lets Jev reads (and image encodes) run between the forwards of a
// denoising block instead of after it. That is only acceptable if it is invisible: the block's tokens
// and the reads' answers must be exactly what each produces alone. The model tests check that on the
// cpu backend and on the pinned GGML backend against the real GGUF (opt-in via TS_TEST_MODEL_DIR);
// the span-scope tests pin the no-op scope that keeps a text-only unified forward from invalidating
// the span-keyed caches, and the latch tests pin how a fallback switch a job turns off mid-block
// waits for the block to end.
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

    private static DiffusionGemmaModel Load(BackendType backend)
        => (DiffusionGemmaModel)ModelBase.Create(
            TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvModelDir), GgufPattern), backend);

    private static (List<int> Tokens, List<int[]> Previews) RunBlock(DiffusionGemmaModel model, int[] prompt,
        DiffusionEbParams p, Action beforeForward, Action<int>? afterStep = null)
    {
        var previews = new List<int[]>();
        DiffusionSeqState state = model.CreateSeqState();
        try
        {
            var run = new DiffusionSeqRun(prompt, p, state, CancellationToken.None,
                (_, step, _, preview) => { previews.Add(preview); afterStep?.Invoke(step); });
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
        using var model = Load(backend);
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

    private static DiffusionEbParams FourSteps() => new()
    {
        MaxDenoisingSteps = 4, Seed = 11, MaxBlocks = 1,
        ConfidenceThreshold = -1f, StabilityThreshold = int.MaxValue,
    };

    // A job that latches fallback switches between two forwards (simulated: the fused decode, the fused
    // lm_head tail and device sampling turned off, as their kernels do when they reject a layout) must
    // not move the rest of the block off the path it started on. The block's steps and tokens stay those
    // of the block alone, later jobs see the switches off, and the model has them off once the block ends.
    // The switches steer the GGML device-glue paths, so ggml_cuda is where this bites; on ggml_cpu it pins
    // the bookkeeping. For contrast, the lm_head tail and device sampling turned off by the block itself
    // (from its preview callback, which is no handoff) apply at once; the output line reports whether that
    // moved the block. (The contrast leaves the fused decode on: the per-op layers are not worth the time.)
    [ModelFact(EnvModelDir, GgufPattern)]
    public void JobLatches_WaitForTheBlockToEnd_OnThePinnedGgmlBackend()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var model = Load(backend);
        int[] chat = Render(model, "Name the three primary colors and one use of each.");
        DiffusionEbParams p = FourSteps();
        DiffusionFallbackLatches initial = model.FallbackLatches;
        DiffusionFallbackLatches failed = initial with { FusedDecodeOk = false, FusedLmHeadTailOk = false, DeviceSampleOk = false };
        bool deviceSampling = model.SupportsDeviceSampling;
        try
        {
            var alone = RunBlock(model, chat, p, null);
            Assert.Equal(initial, model.FallbackLatches);

            var seenByJobs = new List<DiffusionFallbackLatches>();
            var interleaved = RunBlock(model, chat, p, () =>
            {
                seenByJobs.Add(model.FallbackLatches);
                if (seenByJobs.Count == 2) model.FallbackLatches = failed;
            });
            DiffusionFallbackLatches afterBlock = model.FallbackLatches;

            model.FallbackLatches = initial;
            var selfLatched = RunBlock(model, chat, p, null, step =>
            {
                if (step == 0) model.FallbackLatches = initial with { FusedLmHeadTailOk = false, DeviceSampleOk = false };
            });
            bool selfMoved = !selfLatched.Previews.Zip(alone.Previews).All(pair => pair.First.SequenceEqual(pair.Second));

            output.WriteLine($"{backend} device sampling={deviceSampling}: {seenByJobs.Count} handoffs, " +
                $"tokens [{string.Join(",", alone.Tokens.Take(12))}...]; the contrast latch applied inside the " +
                $"block {(selfMoved ? "changed" : "did not change")} its steps");
            Assert.Equal((model.SupportsPromptKvCache ? 1 : 0) + p.MaxDenoisingSteps, seenByJobs.Count);
            Assert.Equal(alone.Previews.Count, interleaved.Previews.Count);
            for (int s = 0; s < alone.Previews.Count; s++) Assert.Equal(alone.Previews[s], interleaved.Previews[s]);
            Assert.Equal(alone.Tokens, interleaved.Tokens);
            Assert.Equal(new[] { initial, initial }, seenByJobs.Take(2));
            Assert.All(seenByJobs.Skip(2), seen => Assert.Equal(failed, seen));
            Assert.Equal(failed, afterBlock);
        }
        finally { model.FallbackLatches = initial; }
    }

    // A batched block that loses on-device sampling mid-way must continue like the single-request
    // DenoiseBlock: host sampling with self-conditioning resumed from the next step. It used to leave
    // self-conditioning off for the rest of the block, because a block that starts on the device path
    // allocates no host self-conditioning buffer. Needs device sampling (ggml_cuda with a model that fits
    // in VRAM, or DIFFUSION_DEVICE_SAMPLE_FORCE=1); elsewhere there is nothing to fall back from.
    [ModelFact(EnvModelDir, GgufPattern)]
    public void DeviceSampleFallback_MidBlock_MatchesTheSingleRequestPath()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var model = Load(backend);
        if (!model.SupportsDeviceSampling)
        {
            output.WriteLine($"{backend}: no on-device sampling on this host; nothing to fall back from.");
            return;
        }
        int[] prompt = Render(model, "Name the three primary colors and one use of each.");
        DiffusionEbParams p = FourSteps();
        DiffusionFallbackLatches initial = model.FallbackLatches;
        var sampler = new DiffusionGemmaSampler(model);
        try
        {
            List<int[]> Single(Action<int>? afterStep)
            {
                var steps = new List<int[]>();
                sampler.DenoiseBlock(prompt, p, (step, _, argmax) => { steps.Add(argmax); afterStep?.Invoke(step); });
                return steps;
            }
            void RejectFromStep1(int step)
            {
                if (step == 0) model.FallbackLatches = initial with { DeviceSampleOk = false };
            }

            List<int[]> single = Single(null);
            var batched = RunBlock(model, prompt, p, null);
            Assert.Equal(single, batched.Previews);   // the two paths agree without a fall back

            model.FallbackLatches = initial;
            List<int[]> singleFellBack = Single(RejectFromStep1);
            Assert.False(model.SupportsDeviceSampling);
            model.FallbackLatches = initial;
            var batchedFellBack = RunBlock(model, prompt, p, null, RejectFromStep1);

            bool moved = !singleFellBack.Zip(single).All(pair => pair.First.SequenceEqual(pair.Second));
            output.WriteLine($"{backend}: {single.Count} steps; the fall back {(moved ? "changed" : "did not change")} the single-request steps");
            Assert.Equal(singleFellBack.Count, batchedFellBack.Previews.Count);
            for (int s = 0; s < singleFellBack.Count; s++) Assert.Equal(singleFellBack[s], batchedFellBack.Previews[s]);
        }
        finally { model.FallbackLatches = initial; }
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

    // ---- fallback switches a job latches mid-block (no weights) --------------------------------

    private static readonly DiffusionFallbackLatches Fresh = new(
        FusedDecodeOk: true, FusedLmHeadTailOk: true, DeviceSampleOk: true,
        MlxFusedDeviceMoeOk: true, MlxFusedDeviceMoeChecked: false,
        MlxGatherQmmMoeOk: true, MlxGatherQmmMoeChecked: false);

    private static DiffusionGemmaModel BareModel()
    {
        var model = (DiffusionGemmaModel)RuntimeHelpers.GetUninitializedObject(typeof(DiffusionGemmaModel));
        model.FallbackLatches = Fresh;
        return model;
    }

    [Fact]
    public void ASwitchAJobLatches_WaitsForTheBlockToEnd_ButLaterJobsSeeIt()
    {
        DiffusionGemmaModel model = BareModel();
        var seen = new List<DiffusionFallbackLatches>();
        var handoff = new DiffusionBlockHandoff(model, () =>
        {
            seen.Add(model.FallbackLatches);
            if (seen.Count == 2) model.FallbackLatches = model.FallbackLatches with { FusedDecodeOk = false };
        });

        handoff.BeforeForward();
        handoff.BeforeForward();                        // this job's fused decode is rejected
        Assert.Equal(Fresh, model.FallbackLatches);     // the block keeps its path
        model.FallbackLatches = model.FallbackLatches with { FusedLmHeadTailOk = false };   // the block's own rejection
        handoff.BeforeForward();
        Assert.Equal(Fresh with { FusedLmHeadTailOk = false }, model.FallbackLatches);   // applies to it at once
        handoff.EndBlock();

        DiffusionFallbackLatches both = Fresh with { FusedDecodeOk = false, FusedLmHeadTailOk = false };
        Assert.Equal(new[] { Fresh, Fresh, both }, seen);
        Assert.Equal(both, model.FallbackLatches);
    }

    // A job that runs an opt-in MLX MoE path's one-time self-check before the block does must not mark
    // the check done for the block: the block runs its own, and both verdicts count after it.
    [Fact]
    public void AJobsSelfCheck_DoesNotStandInForTheBlocks()
    {
        DiffusionGemmaModel model = BareModel();
        var handoff = new DiffusionBlockHandoff(model, () =>
        {
            if (!model.FallbackLatches.MlxGatherQmmMoeChecked)
                model.FallbackLatches = model.FallbackLatches with { MlxGatherQmmMoeChecked = true, MlxGatherQmmMoeOk = false };
        });

        handoff.BeforeForward();
        Assert.Equal(Fresh, model.FallbackLatches);
        model.FallbackLatches = model.FallbackLatches with { MlxGatherQmmMoeChecked = true };   // the block's check passes
        handoff.BeforeForward();
        Assert.Equal(Fresh with { MlxGatherQmmMoeChecked = true }, model.FallbackLatches);
        handoff.EndBlock();

        Assert.Equal(Fresh with { MlxGatherQmmMoeChecked = true, MlxGatherQmmMoeOk = false }, model.FallbackLatches);
    }

    [Fact]
    public void AJobThatThrows_StillLeavesTheBlockItsPath()
    {
        DiffusionGemmaModel model = BareModel();
        var handoff = new DiffusionBlockHandoff(model, () =>
        {
            model.FallbackLatches = model.FallbackLatches with { DeviceSampleOk = false };
            throw new InvalidOperationException("job failed");
        });

        Assert.Throws<InvalidOperationException>(handoff.BeforeForward);
        Assert.Equal(Fresh, model.FallbackLatches);
        handoff.EndBlock();
        Assert.Equal(Fresh with { DeviceSampleOk = false }, model.FallbackLatches);
    }

    // The device sampling tail reads the fused layer graph's output. Advertising it without the fused
    // decode made each block start on the device path and fall back at its first step.
    [Fact]
    public void DeviceSampling_NeedsTheFusedDecode()
    {
        DiffusionGemmaModel model = BareModel();
        typeof(ModelBase).GetField("_backend", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, BackendType.GgmlCuda);
        Field("_deviceSampleEnabled").SetValue(model, true);
        Field("_fusedDecodeEnabled").SetValue(model, true);
        Assert.True(model.SupportsDeviceSampling);

        model.FallbackLatches = Fresh with { FusedDecodeOk = false };
        Assert.False(model.SupportsDeviceSampling);

        model.FallbackLatches = Fresh;
        Field("_fusedDecodeEnabled").SetValue(model, false);   // DIFFUSION_NO_FUSED_DECODE=1
        Assert.False(model.SupportsDeviceSampling);
    }
}
