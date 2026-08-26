// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public class Qwen35TokenDecodeContractTests
{
    [Theory]
    [InlineData(BackendType.GgmlMetal, false, "1", true)]
    [InlineData(BackendType.GgmlMetal, false, "0", false)]
    [InlineData(BackendType.GgmlMetal, false, null, true)]
    [InlineData(BackendType.GgmlMetal, false, "", true)]
    [InlineData(BackendType.GgmlMetal, true, "1", false)]
    [InlineData(BackendType.GgmlMetal, true, null, false)]
    [InlineData(BackendType.GgmlCuda, false, "1", false)]
    [InlineData(BackendType.GgmlCuda, false, null, false)]
    [InlineData(BackendType.GgmlVulkan, false, "1", false)]
    public void MetalGdnInplaceStateGate_DefaultsOnOnlyForSingleDeviceMetal(
        BackendType backend,
        bool isTensorParallel,
        string? environmentValue,
        bool expected)
    {
        Assert.Equal(
            expected,
            Qwen35Model.ShouldUseMetalGdnInplaceState(
                backend,
                isTensorParallel,
                environmentValue));
    }

    [Theory]
    [InlineData(BackendType.GgmlMetal, true, -1, 4, false)]
    [InlineData(BackendType.GgmlMetal, true, 4, 4, false)]
    [InlineData(BackendType.GgmlCuda, true, -1, 4, true)]
    [InlineData(BackendType.GgmlCuda, true, 4, 4, true)]
    [InlineData(BackendType.GgmlCuda, true, 1, 4, false)]
    [InlineData(BackendType.GgmlCuda, false, -1, 4, false)]
    public void VerifyResidentState_NeverRetainsMetalStateView(
        BackendType backend,
        bool residentEnabled,
        int nLogitRows,
        int seqLen,
        bool expected)
    {
        Assert.Equal(
            expected,
            Qwen35Model.ShouldUseVerifyResidentState(
                backend,
                residentEnabled,
                nLogitRows,
                seqLen));
    }

    [Fact]
    public void MetalGdnInplaceStateLayout_UsesOneAttentionRowAsBackingPrefix()
    {
        const int numVHeads = 3;
        const int headVDim = 5;
        const int headKDim = 7;
        const long attentionElements = numVHeads * headVDim;
        const long stateElements = attentionElements * headKDim;
        var allocator = new CpuAllocator(BlasEnum.DotNet);

        using Tensor ordinary = Qwen35Model.AllocateGdnDeltaStateTensor(
            allocator,
            useMetalInplaceLayout: false,
            numVHeads,
            headVDim,
            headKDim);
        Assert.Equal(0, ordinary.StorageOffset);
        Assert.Equal(stateElements, ordinary.Storage.ElementCount);

        using Tensor inplace = Qwen35Model.AllocateGdnDeltaStateTensor(
            allocator,
            useMetalInplaceLayout: true,
            numVHeads,
            headVDim,
            headKDim);
        Assert.Equal(attentionElements, inplace.StorageOffset);
        Assert.Equal(attentionElements + stateElements, inplace.Storage.ElementCount);
        Assert.Equal(stateElements * sizeof(float), Qwen35Model.GdnDeltaStateBytes(inplace));
        Assert.Equal(
            attentionElements * sizeof(float),
            TensorComputePrimitives.GetStoragePointer(inplace).ToInt64() -
                TensorComputePrimitives.GetStorageBasePointer(inplace).ToInt64());
    }

    [Fact]
    public void DirectTokenDecodeRoute_AcceptsOrdinarySingleTokenMetalDecode()
    {
        Assert.True(Qwen35Model.CanTryDirectTokenDecode(
            BackendType.GgmlMetal,
            isTensorParallel: false,
            sequenceLength: 1,
            hasVisionEmbeddings: false,
            hasMRoPEPositions: false,
            tokenId: 42,
            vocabSize: 128));
    }

    [Theory]
    [InlineData(BackendType.GgmlCuda, false, 1, false, false, 42, 128)]
    [InlineData(BackendType.GgmlVulkan, false, 1, false, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, true, 1, false, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 2, false, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, true, false, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, true, 42, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, false, -1, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, false, 128, 128)]
    [InlineData(BackendType.GgmlMetal, false, 1, false, false, 0, 0)]
    public void DirectTokenDecodeRoute_RejectsUnsupportedOrInvalidInputs(
        BackendType backend,
        bool isTensorParallel,
        int sequenceLength,
        bool hasVisionEmbeddings,
        bool hasMRoPEPositions,
        int tokenId,
        int vocabSize)
    {
        Assert.False(Qwen35Model.CanTryDirectTokenDecode(
            backend,
            isTensorParallel,
            sequenceLength,
            hasVisionEmbeddings,
            hasMRoPEPositions,
            tokenId,
            vocabSize));
    }

    [Theory]
    [InlineData(nameof(GgmlBasicOps.Qwen35ModelDecode), "TSGgml_Qwen35ModelDecode")]
    [InlineData(nameof(GgmlBasicOps.Qwen35ModelDecodeToken), "TSGgml_Qwen35ModelDecodeToken")]
    public void DecodePInvoke_MatchesPublicManagedParameterContract(
        string publicMethodName,
        string nativeMethodName)
    {
        MethodInfo publicMethod = typeof(GgmlBasicOps).GetMethod(
            publicMethodName,
            BindingFlags.Public | BindingFlags.Static)!;

        Type nativeType = typeof(GgmlBasicOps).Assembly.GetType(
            "TensorSharp.GGML.GgmlNative",
            throwOnError: true)!;
        MethodInfo nativeMethod = nativeType.GetMethod(
            nativeMethodName,
            BindingFlags.NonPublic | BindingFlags.Static)!;

        Assert.NotNull(publicMethod);
        Assert.NotNull(nativeMethod);
        Assert.Equal(typeof(bool), publicMethod.ReturnType);
        Assert.Equal(typeof(int), nativeMethod.ReturnType);
        Assert.Equal(
            publicMethod.GetParameters().Select(parameter => parameter.ParameterType),
            nativeMethod.GetParameters().Select(parameter => parameter.ParameterType));

        LibraryImportAttribute import = nativeMethod.GetCustomAttribute<LibraryImportAttribute>()!;
        Assert.NotNull(import);
        Assert.Equal("GgmlOps", import.LibraryName);
        UnmanagedCallConvAttribute callConv = nativeMethod.GetCustomAttribute<UnmanagedCallConvAttribute>()!;
        Assert.NotNull(callConv);
        Assert.Contains(typeof(System.Runtime.CompilerServices.CallConvCdecl), callConv.CallConvs!);
        Assert.Equal(nativeMethodName, nativeMethod.Name);

        ParameterInfo reseedParameter = nativeMethod.GetParameters()[2];
        Assert.Equal(typeof(bool), reseedParameter.ParameterType);
        MarshalAsAttribute marshalAs = reseedParameter.GetCustomAttribute<MarshalAsAttribute>()!;
        Assert.NotNull(marshalAs);
        Assert.Equal(UnmanagedType.Bool, marshalAs.Value);
    }

    [Fact]
    public void Qwen35LayerDescriptor_HasExpectedBlittableLayout()
    {
        FieldInfo[] fields = typeof(Qwen35LayerDecodeArgs).GetFields(
            BindingFlags.Public | BindingFlags.Instance);

        // Every addition is appended at the END of its own run, so the struct stays
        // a pointers / int64 / int32 sequence and the native TSGgmlQwen35LayerDesc
        // keeps the same offsets for every field before it. The two most recent:
        //   CpuMoe                    per-layer MoE CPU offload (--n-cpu-moe)
        //   FfnGateW / FfnUpW (+ their shapes and types)
        //                             the dense FFN of a mixed-quant "UD" layer whose
        //                             ffn_gate and ffn_up have different GGML types
        //                             and so cannot be fused into one tensor
        Assert.Equal(35, fields.Count(field => field.FieldType == typeof(IntPtr)));
        Assert.Equal(54, fields.Count(field => field.FieldType == typeof(long)));
        Assert.Equal(26, fields.Count(field => field.FieldType == typeof(int)));
        Assert.All(fields, field => Assert.True(
            field.FieldType == typeof(IntPtr) ||
            field.FieldType == typeof(long) ||
            field.FieldType == typeof(int),
            $"Unexpected non-blittable field {field.Name}: {field.FieldType}."));

        static long Align(long value, int alignment)
            => (value + alignment - 1) / alignment * alignment;

        long int64Start = Align(35L * IntPtr.Size, sizeof(long));
        long int32Start = int64Start + 54L * sizeof(long);
        long expectedSize = Align(
            int32Start + 26L * sizeof(int),
            Math.Max(IntPtr.Size, sizeof(long)));

        Assert.Equal(0, Marshal.OffsetOf<Qwen35LayerDecodeArgs>(
            nameof(Qwen35LayerDecodeArgs.AttnNormW)).ToInt64());
        Assert.Equal(int64Start, Marshal.OffsetOf<Qwen35LayerDecodeArgs>(
            nameof(Qwen35LayerDecodeArgs.QkvNe0)).ToInt64());
        Assert.Equal(int32Start, Marshal.OffsetOf<Qwen35LayerDecodeArgs>(
            nameof(Qwen35LayerDecodeArgs.StructBytes)).ToInt64());
        Assert.Equal(expectedSize, Marshal.SizeOf<Qwen35LayerDecodeArgs>());

        // The int32 run's ORDER is what the struct_bytes handshake cannot check -
        // it only catches a size change - so pin the tail explicitly. CpuMoe kept
        // its slot when the split-gate/up pair was appended after it.
        Assert.Equal(
            int32Start + 23L * sizeof(int),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.CpuMoe)).ToInt64());
        Assert.Equal(
            int32Start + 24L * sizeof(int),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnGateType)).ToInt64());
        Assert.Equal(
            int32Start + 25L * sizeof(int),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnUpType)).ToInt64());
        // Same for the pointer and int64 runs.
        Assert.Equal(
            34L * IntPtr.Size,
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnUpW)).ToInt64());
        Assert.Equal(
            int64Start + 48L * sizeof(long),
            Marshal.OffsetOf<Qwen35LayerDecodeArgs>(nameof(Qwen35LayerDecodeArgs.FfnGateNe0)).ToInt64());
    }
}
