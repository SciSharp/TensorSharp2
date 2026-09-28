// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// The instruction sets the hand-written pure-C# CPU kernels may use, decided in one place:
    /// the Core SGEMM and elementwise kernels, the quantized GEMM and per-row dots, the packed
    /// GEMM of the Qwen-Image VAE/vision/text encoder, the Qwen-Image DiT kernels and the
    /// DiffusionGemma attention all derive their kernel choice from these flags, so one host
    /// never runs AVX-512 in one of them and AVX2 in another.
    ///
    /// AVX-512 counts as present only when the CPU has F/BW/DQ AND the JIT accelerates
    /// Vector512: .NET reports Vector512 as not accelerated where it prefers 256-bit vectors
    /// (some Skylake-X/Cascade Lake parts) and under DOTNET_EnableAVX512=0, and every kernel
    /// follows that. TS_CPU_DISABLE_AVX512=1 turns the AVX-512 kernels off on a host that has
    /// them (the AVX2 path a CPU without AVX-512 takes, testable on an AVX-512 machine); it does
    /// not narrow TensorPrimitives or plain copies, which pick their own width. DOTNET_EnableAVX2=0
    /// (or DOTNET_EnableHWIntrinsic=0) leaves the portable Vector128 / scalar paths, the same ones
    /// ARM64 runs.
    /// </summary>
    internal static class CpuIsa
    {
        /// <summary>TS_CPU_DISABLE_AVX512=1: keep the kernels on AVX2 even where AVX-512 exists.</summary>
        internal static readonly bool Avx512DisabledByEnv =
            Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") == "1";

        /// <summary>The hardware and runtime can run the AVX-512 kernels (F/BW/DQ, Vector512
        /// accelerated), whatever TS_CPU_DISABLE_AVX512 says. Tests and benchmarks use it to pick
        /// every kernel this machine can execute.</summary>
        internal static bool HasAvx512 =>
            Avx512F.IsSupported && Avx512BW.IsSupported && Avx512DQ.IsSupported && Vector512.IsHardwareAccelerated;

        /// <summary>The hardware and runtime can run the AVX2+FMA kernels (256-bit vectors accelerated).</summary>
        internal static bool HasAvx2Fma =>
            Avx2.IsSupported && Fma.IsSupported && Vector256.IsHardwareAccelerated;

        /// <summary>Default kernel choice: AVX-512 when present and not switched off.</summary>
        internal static readonly bool Avx512 = HasAvx512 && !Avx512DisabledByEnv;

        /// <summary>Default kernel choice: AVX2+FMA (true on AVX-512 hosts too).</summary>
        internal static readonly bool Avx2Fma = HasAvx2Fma;
    }
}
