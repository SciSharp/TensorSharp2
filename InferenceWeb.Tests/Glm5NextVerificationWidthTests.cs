// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Teacher-forced, full-vocabulary verification parity on mixed KDA/MLA and
/// routed-expert fixtures. Adjacent positions straddling the 256-key padding
/// boundary must not reuse a graph with the wrong per-row key extents. Wide
/// captured windows also test the native decode-width cap and accepted replay.
/// This is numerical regression coverage, not a model-quality benchmark.
/// </summary>
public sealed class Glm5NextVerificationWidthTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-glm-width-" + Guid.NewGuid().ToString("N"));
    private const double Tolerance = 2e-5;

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void NativeCpu_MixedAttention_AllVerifyRowsMatchSequential(bool quantized, bool sparse)
        => Check(BackendType.GgmlCpu, tensorParallel: false, degree: 1, quantized, sparse);

    [GlmNativeCudaFact(2)]
    public void LayerSplit2_QuantizedDense_AllVerifyRowsMatchSequential()
        => Check(BackendType.GgmlCuda, tensorParallel: false, degree: 2, quantized: true, sparse: false);

    [GlmNativeCudaFact(2)]
    public void LayerSplit2_QuantizedSparse_AllVerifyRowsMatchSequential()
        => Check(BackendType.GgmlCuda, tensorParallel: false, degree: 2, quantized: true, sparse: true);

    [GlmNativeCudaFact(2)]
    public void TensorParallel2_QuantizedDense_AllVerifyRowsMatchSequential()
        => Check(BackendType.GgmlCuda, tensorParallel: true, degree: 2, quantized: true, sparse: false);

    [GlmNativeCudaFact(2)]
    public void TensorParallel2_QuantizedSparse_AllVerifyRowsMatchSequential()
        => Check(BackendType.GgmlCuda, tensorParallel: true, degree: 2, quantized: true, sparse: true);

    private void Check(BackendType backend, bool tensorParallel, int degree, bool quantized, bool sparse)
    {
        Directory.CreateDirectory(_directory);
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_directory, "mixed.gguf"), numHeads: 4, quantizeAttentionOutput: quantized,
            numLayers: 2, mixedAttention: true, routedExperts: true, quantizeExperts: quantized,
            contextLength: 512, indexerTopK: sparse ? 4 : 1024);
        using var env = new NativeEnvironment();
        foreach (string name in new[]
        {
            "TS_GLM_NATIVE", "TS_GLM_NGPU", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE",
            "TS_GLM_NODES_PER_LAYER", "TS_GLM_MOE_MMAP", "TS_N_CPU_MOE", "TS_CPU_MOE",
            "TS_GLM_FA", "TS_GLM_HC_NATIVE", "TS_GLM_TP_FUSED", "GGML_CUDA_DISABLE_FUSION",
            "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE",
            "TS_SPEC", "TS_MTP", "TS_SPEC_DRAFT_MODEL", "TS_MTP_DRAFT_MODEL",
        }) env.Set(name, null);
        env.Set("MAX_CONTEXT", "512");
        env.Set("TS_GLM_UBATCH", "64");
        env.Set("TS_GLM_TOPK", "1");
        env.Set("TS_GLM_MTP", "0");
        env.Set("TS_GLM_THREADS", "2");
        env.Set("TS_CPU_MOE_THREADS", "2");

        using ModelBase model = ModelBase.Create(path, backend,
            tpDegree: tensorParallel ? degree : 1, layerSplitDegree: tensorParallel ? 1 : degree);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        Assert.True(target.SpeculationProfitable);
        int vocab = model.Config.VocabSize;
        double worst = 0;
        // Same final padded key extent (512), different first-row extents. The
        // same model and its native graph cache serve both positions and widths.
        foreach (int width in new[] { 2, 3, 4, 5, 6, 7, 8, 9, 65 })
        foreach (int prefixLength in new[] { 255, 257 })
        {
            int[] prompt = Enumerable.Range(0, prefixLength).Select(i => 3 + (i * 17 + i / 7) % 120).ToArray();
            int[] window = Enumerable.Range(0, width).Select(i => 2 + (i * 29 + width) % 120).ToArray();
            int keep = width - 1;
            float[][] expected = new float[width][];

            void Refill()
            {
                model.ResetKVCache();
                model.ForwardRefill(prompt);
                Assert.Equal(prefixLength, target.CacheSeqLen);
            }

            Refill();
            for (int row = 0; row < width; row++)
                expected[row] = (float[])model.Forward(new[] { window[row] }).Clone();
            float[] expectedAfterWindow = (float[])model.Forward(new[] { 91 }).Clone();
            Refill();
            for (int row = 0; row < keep; row++) model.Forward(new[] { window[row] });
            float[] expectedAfterReplay = (float[])model.Forward(new[] { 93 }).Clone();

            Refill();
            target.SpecEnsureCapacity(prefixLength + width + 1);
            target.SpecSnapshotRecurrentState();
            var actual = new float[width * vocab];
            Array.Fill(actual, float.NaN);
            target.SpecForward(window, null, actual, allLogitsRows: true);
            Assert.Equal(prefixLength + width, target.CacheSeqLen);
            string label = $"{backend} degree={degree} tp={tensorParallel} quant={quantized} sparse={sparse} prefix={prefixLength} width={width}";
            for (int row = 0; row < width; row++)
                Compare(expected[row], actual.AsSpan(row * vocab, vocab), label + $" row={row}", ref worst);
            Compare(expectedAfterWindow, model.Forward(new[] { 91 }), label + " committed state", ref worst);

            target.SpecRestoreRecurrentState();
            target.SpecRewindCache(prefixLength);
            var replay = new float[vocab];
            target.SpecForward(window[..keep], null, replay, allLogitsRows: false);
            Assert.Equal(prefixLength + keep, target.CacheSeqLen);
            Compare(expected[keep - 1], replay, label + " replay row", ref worst);
            Compare(expectedAfterReplay, model.Forward(new[] { 93 }), label + " replay state", ref worst);
            output.WriteLine($"{label}: all {width * vocab} verify logits and both continuations passed; worstAbs={worst:G9}");
        }
    }

    private static void Compare(float[] expected, ReadOnlySpan<float> actual, string label, ref double worst)
    {
        Assert.Equal(expected.Length, actual.Length);
        int expectedTop = 0, actualTop = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]), $"{label}: non-finite logit {i}");
            double error = Math.Abs((double)expected[i] - actual[i]);
            worst = Math.Max(worst, error);
            Assert.True(error <= Tolerance * (1 + Math.Abs(expected[i])),
                $"{label}: logit {i}, expected={expected[i]:R}, actual={actual[i]:R}, error={error:G9}, tolerance={Tolerance:G}");
            if (expected[i] > expected[expectedTop]) expectedTop = i;
            if (actual[i] > actual[actualTop]) actualTop = i;
        }
        Assert.Equal(expectedTop, actualTop);
    }

    // Keep libc getenv and managed readers consistent when the CUDA test runner
    // was launched with tuning variables for a production benchmark.
    private sealed class NativeEnvironment : IDisposable
    {
        private readonly Dictionary<string, string?> _original = new();
        [DllImport("libc", EntryPoint = "setenv")] private static extern int SetUnix(string name, string value, int overwrite);
        [DllImport("libc", EntryPoint = "unsetenv")] private static extern int ClearUnix(string name);
        [DllImport("ucrtbase", EntryPoint = "_putenv_s")] private static extern int SetWindows(string name, string value);
        public void Set(string name, string? value)
        {
            _original.TryAdd(name, Environment.GetEnvironmentVariable(name));
            Apply(name, value);
        }
        private static void Apply(string name, string? value)
        {
            Environment.SetEnvironmentVariable(name, value);
            int result = OperatingSystem.IsWindows() ? SetWindows(name, value ?? "")
                : value == null ? ClearUnix(name) : SetUnix(name, value, 1);
            Assert.Equal(0, result);
        }
        public void Dispose() { foreach (var pair in _original) Apply(pair.Key, pair.Value); }
    }
}
