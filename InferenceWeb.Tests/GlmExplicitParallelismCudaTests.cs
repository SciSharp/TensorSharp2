// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Exercises the public model factory, native loading, sparse prefill, decode and
/// reset in both explicit modes. Synthetic weights test numerical correctness;
/// the timing output is fixture overhead, not a production-model benchmark.
/// </summary>
public sealed class GlmExplicitParallelismCudaTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-glm-explicit-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [GlmNativeCudaFact(2)]
    public void LayerSplitFactory_MatchesSingleGpuPrefillDecodeAndReset() => CheckMode(tensorParallel: false);

    [GlmNativeCudaFact(2)]
    public void TensorParallelFactory_MatchesSingleGpuPrefillDecodeAndReset() => CheckMode(tensorParallel: true);

    private void CheckMode(bool tensorParallel)
    {
        Directory.CreateDirectory(_dir);
        string path = GlmDsaSyntheticModelBuilder.Write(Path.Combine(_dir, "tiny-glm-dsa.gguf"));
        using var env = new EnvScope();
        env.ClearSpeculationVars();
        foreach (string name in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TS_GLM_NATIVE" })
            env.Set(name, null);
        env.Set("MAX_CONTEXT", "256");
        env.Set("TS_GLM_UBATCH", "16");
        env.Set("TS_GLM_THREADS", "2");
        env.Set("TS_GLM_NGPU", "1");

        int[] prompt = Enumerable.Range(0, 37).Select(i => 3 + (i * 11) % 90).ToArray();
        const int steps = 8;
        var reference = new float[steps][];
        var tokens = new int[steps];
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
        {
            reference[0] = (float[])model.ForwardRefill(prompt).Clone();
            for (int step = 0; step < steps; step++)
            {
                tokens[step] = ArgMax(reference[step]);
                if (step + 1 < steps) reference[step + 1] = (float[])model.Forward(new[] { tokens[step] }).Clone();
            }
        }

        env.Set("TS_GLM_NGPU", null);
        using ModelBase parallel = ModelBase.Create(path, BackendType.GgmlCuda,
            tpDegree: tensorParallel ? 2 : 1, layerSplitDegree: tensorParallel ? 1 : 2);
        double worst = 0;
        var timer = Stopwatch.StartNew();
        for (int repeat = 0; repeat < 2; repeat++)
        {
            parallel.ResetKVCache();
            float[] actual = parallel.ForwardRefill(prompt);
            for (int step = 0; step < steps; step++)
            {
                Assert.Equal(reference[step].Length, actual.Length);
                for (int v = 0; v < actual.Length; v++)
                {
                    Assert.True(float.IsFinite(actual[v]), $"Non-finite logit at repeat {repeat}, step {step}, vocab {v}.");
                    worst = Math.Max(worst, Math.Abs(actual[v] - reference[step][v]));
                }
                Assert.True(worst < 2e-3, $"Max logit error {worst:G6} at repeat {repeat}, step {step}.");
                Assert.Equal(tokens[step], ArgMax(actual));
                if (step + 1 < steps) actual = parallel.Forward(new[] { tokens[step] });
            }
        }
        output.WriteLine($"mode={(tensorParallel ? "tensor-parallel" : "layer-split")}, GPUs=2, max_logit_error={worst:G6}, " +
            $"two_prefill_decode_reset_runs_ms={timer.Elapsed.TotalMilliseconds:F3}; synthetic fixture, not production throughput.");
    }

    private static int ArgMax(float[] values)
    {
        int result = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[result]) result = i;
        return result;
    }
}
