// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.GGML;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The CLI tears the GGML backend down at exit (the ggml-metal residency assertion needs
/// it), and that teardown is a P/Invoke. It used to run for every backend, so a
/// <c>--backend cpu</c> run loaded GgmlOps at exit only to free nothing. It now runs for
/// the GGML backends exactly as before, and for the others only when the process had
/// already bound the native library.
/// </summary>
public sealed class CliGgmlShutdownTests
{
    [Theory]
    [InlineData(null)]              // no --backend: the CLI default is ggml_cpu
    [InlineData("ggml_cpu")]
    [InlineData("ggml_metal")]
    [InlineData("ggml_cuda")]
    [InlineData("ggml-cuda")]
    [InlineData("ggml_vulkan")]
    [InlineData("ggml-vulkan")]
    public void GgmlBackendsAlwaysShutDown(string? backend)
    {
        Assert.True(TensorSharp.Cli.Program.ShouldShutdownGgml(backend, nativeLibraryLoaded: false));
        Assert.True(TensorSharp.Cli.Program.ShouldShutdownGgml(backend, nativeLibraryLoaded: true));
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("cuda")]
    [InlineData("direct_cuda")]
    [InlineData("mlx")]
    public void OtherBackendsShutDownOnlyWhenGgmlWasLoaded(string backend)
    {
        Assert.False(TensorSharp.Cli.Program.ShouldShutdownGgml(backend, nativeLibraryLoaded: false));
        Assert.True(TensorSharp.Cli.Program.ShouldShutdownGgml(backend, nativeLibraryLoaded: true));
    }

    [Fact]
    public void LoadedFlagIsSetOnceTheNativeLibraryIsBound()
    {
        // Any P/Invoke binds the library through GgmlNative's import resolver. The CPU backend
        // is compiled into every GgmlOps build, so this also proves the library is present.
        Assert.True(GgmlBasicOps.CanInitializeBackend(GgmlBackendType.Cpu), "GgmlOps could not be loaded.");
        Assert.True(GgmlBasicOps.IsNativeLibraryLoaded);
    }
}
