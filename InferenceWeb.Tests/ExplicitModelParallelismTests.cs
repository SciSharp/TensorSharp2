// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

public sealed class ExplicitModelParallelismTests : IDisposable
{
    private readonly EnvScope _env = new();

    public ExplicitModelParallelismTests()
    {
        foreach (string name in new[] { "TS_DSV41_TP", "TS_DSV4_NGPU", "TS_GLM_NGPU", "TS_GLM_NATIVE",
            "TS_GLM_TP_SHARD", "TENSORSHARP_TP_DEVICES", "TENSORSHARP_LAYER_SPLIT_DEVICES" })
            _env.Set(name, null);
    }

    public void Dispose() => _env.Dispose();

    private static int Resolve(string name, BackendType backend, int tp, int split, out int resolvedSplit)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(name, out var descriptor));
        ITensorParallelGroup group = null;
        return ModelBase.ResolveTensorParallelSupport(descriptor, backend, tp, ref group, out resolvedSplit, split);
    }

    [Theory]
    [InlineData("qwen4exp")]
    [InlineData("deepseek4")]
    [InlineData("deepseek41")]
    public void LayerOnlyArchitecturesNeverInterpretTpAsPlacement(string architecture)
    {
        var error = Assert.Throws<NotSupportedException>(() => Resolve(architecture, BackendType.GgmlCuda, 2, 1, out _));
        Assert.Contains("--layer-split", error.Message);
        Assert.Equal(1, Resolve(architecture, BackendType.GgmlCuda, 1, 2, out int split));
        Assert.Equal(2, split);
    }

    [Theory]
    [InlineData("glm-dsa")]
    [InlineData("glm5next")]
    public void GlmSelectsTheRequestedMode(string architecture)
    {
        Assert.Equal(2, Resolve(architecture, BackendType.GgmlCuda, 2, 1, out int split));
        Assert.Equal(1, split);
        Assert.Equal(1, Resolve(architecture, BackendType.GgmlCuda, 1, 2, out split));
        Assert.Equal(2, split);
    }

    [Fact]
    public void ConflictingModesAndUnsupportedLayerPlacementAreRefused()
    {
        Assert.Throws<ArgumentException>(() => Resolve("glm-dsa", BackendType.GgmlCuda, 2, 2, out _));
        Assert.Throws<NotSupportedException>(() => Resolve("qwen3", BackendType.GgmlCuda, 1, 2, out _));
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlMetal)]
    [InlineData(BackendType.Mlx)]
    public void SingleDeviceBackendsDoNotSilentlyAcceptTp(BackendType backend)
        => Assert.Throws<NotSupportedException>(() => Resolve("qwen3", backend, 2, 1, out _));

    [Fact]
    public void NativeDeepSeekExpertTpRetainsExplicitTpDegree()
    {
        _env.Set("TS_DSV41_TP", "2");
        Assert.Equal(2, Resolve("deepseek41", BackendType.GgmlCuda, 2, 1, out int split));
        Assert.Equal(1, split);
        Assert.Throws<ArgumentException>(() => new DeepSeek41Model("not-opened.gguf", BackendType.GgmlCuda, layerSplitDegree: 2));
    }

    [Fact]
    public void NativeGpuOverridesCannotChangeExplicitDeviceCount()
    {
        Assert.Equal(1, GlmDsaModel.ResolveNativeGpuCount(1, 1));
        Assert.Equal(1, DeepSeek4Model.ResolveNativeGpuCount(1));
        _env.Set("TS_GLM_NGPU", "1");
        Assert.Throws<ArgumentException>(() => GlmDsaModel.ResolveNativeGpuCount(1, 2));
        Assert.Throws<ArgumentException>(() => GlmDsaModel.ResolveNativeGpuCount(2, 1));
        _env.Set("TS_DSV4_NGPU", "1");
        Assert.Throws<ArgumentException>(() => DeepSeek4Model.ResolveNativeGpuCount(2));
        _env.Set("TS_GLM_NGPU", "2");
        Assert.Equal(2, GlmDsaModel.ResolveNativeGpuCount(1, 2));
        Assert.Equal(2, GlmDsaModel.ResolveNativeGpuCount(2, 1));
    }

    [Fact]
    public void NativeDeviceLimitsCannotSilentlyClampExplicitDegrees()
    {
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ResolveNativeGpuCount(1, 9));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ResolveNativeGpuCount(9, 1));
        Assert.Throws<NotSupportedException>(() => DeepSeek4Model.ResolveNativeGpuCount(9));
        _env.Set("TS_GLM_NGPU", "9");
        _env.Set("TS_DSV4_NGPU", "9");
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ResolveNativeGpuCount(1, 1));
        Assert.Throws<NotSupportedException>(() => DeepSeek4Model.ResolveNativeGpuCount(1));
    }

    [Theory]
    [InlineData(BackendType.Cpu, null)]
    [InlineData(BackendType.Cuda, null)]
    [InlineData(BackendType.GgmlCpu, "0")]
    [InlineData(BackendType.GgmlCuda, "0")]
    public void NativeGpuLimitsAndOverridesDoNotConstrainManagedGlmPaths(BackendType backend, string? native)
    {
        _env.Set("TS_GLM_NATIVE", native);
        _env.Set("TS_GLM_NGPU", "invalid-native-only-setting");
        var validate = typeof(GlmDsaModel).GetMethod("ValidateParallelism",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(validate);
        // The native executor caps ranks at eight. The managed path must reach
        // its own topology validation and ignore settings belonging to native.
        Assert.Equal(backend, validate.Invoke(null, new object[] { backend, 9, null, 1 }));
    }

    [Theory]
    [InlineData("invalid-native-only-setting", typeof(ArgumentException))]
    [InlineData("9", typeof(NotSupportedException))]
    public void NativeGpuOverridesDoNotConstrainManagedDeepSeekCpu(string setting, Type nativeError)
    {
        _env.Set("TS_DSV4_NGPU", setting);
        var validate = typeof(DeepSeek4Model).GetMethod("ValidateParallelism",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(validate);
        // Exercise the constructor's pre-load validation without allocating a
        // backend: only the pure C# executor ignores this native placement knob.
        Assert.Equal(BackendType.Cpu,
            validate.Invoke(null, new object[] { BackendType.Cpu, 1, null, 1 }));
        foreach (var backend in new[] { BackendType.Cuda, BackendType.GgmlCuda,
            BackendType.GgmlVulkan, BackendType.GgmlCpu })
        {
            var error = Assert.Throws<System.Reflection.TargetInvocationException>(() =>
                validate.Invoke(null, new object[] { backend, 1, null, 1 }));
            Assert.IsType(nativeError, error.InnerException);
        }
    }

    [Fact]
    public void NativeLayerSplitCannotFallBackToTheManagedGlmExecutor()
    {
        _env.Set("TS_GLM_NATIVE", "0");
        var error = Assert.Throws<NotSupportedException>(() =>
            new GlmDsaModel("not-opened.gguf", BackendType.GgmlCuda, layerSplitDegree: 2));
        Assert.Contains("TS_GLM_NATIVE", error.Message);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("4")]
    [InlineData("none")]
    public void NativeTpCannotDisableEveryWeightShard(string mask)
    {
        _env.Set("TS_GLM_TP_SHARD", mask);
        Assert.Throws<ArgumentException>(() => new GlmDsaModel("not-opened.gguf", BackendType.GgmlCuda, tpDegree: 2));
    }

    [Fact]
    public void DeviceSelectionIsIndependentAndRejectsDuplicatePhysicalDevices()
    {
        _env.Set("TENSORSHARP_TP_DEVICES", "1,3");
        _env.Set("TENSORSHARP_LAYER_SPLIT_DEVICES", "0,2");
        Assert.Equal(new[] { 1, 3 }, ModelBase.ParseDeviceIds(2, "TENSORSHARP_TP_DEVICES"));
        Assert.Equal(new[] { 0, 2 }, ModelBase.ParseDeviceIds(2, "TENSORSHARP_LAYER_SPLIT_DEVICES"));
        _env.Set("TENSORSHARP_LAYER_SPLIT_DEVICES", "0,0");
        Assert.Throws<ArgumentException>(() => ModelBase.ParseDeviceIds(2, "TENSORSHARP_LAYER_SPLIT_DEVICES"));
    }

    [Theory]
    [InlineData("glm-dsa")]
    [InlineData("glm5next")]
    [InlineData("deepseek4")]
    [InlineData("deepseek41")]
    [InlineData("qwen4exp")]
    [InlineData("qwen-image")]
    [InlineData("diffusion-gemma")]
    public void DistributedPreflightRejectsExecutorsWithoutCrossNodeCollectives(string architecture)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(architecture, out var descriptor));
        var error = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ValidateDistributedTensorParallelism(descriptor, BackendType.GgmlCuda));
        Assert.Contains("--tp-node-id/--tp-peers", error.Message);
    }

    [Fact]
    public void DistributedPreflightUsesMetadataWithoutLoadingWeightsOrNativeBackend()
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-glm-preflight-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            GlmDsaSyntheticModelBuilder.Write(path);
            Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedTensorParallelism(path, BackendType.GgmlCuda));
        }
        finally { File.Delete(path); }
        Assert.True(ModelArchitectureRegistry.TryGet("qwen3", out var descriptor));
        ModelBase.ValidateDistributedTensorParallelism(descriptor, BackendType.GgmlCuda);
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedTensorParallelism(descriptor, BackendType.Cpu));
    }
}
