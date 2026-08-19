// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the native GLM (glm-dsa) whole-model executor
// (TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp). The native side loads the
// (split) GGUF itself, places the layers across every visible GPU, owns the
// MLA and lightning-indexer KV caches, and runs prefill / decode ubatches
// through ggml_backend_sched.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static class GgmlGlmNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Conv = CallingConvention.Cdecl;

        /// <summary>No routed-expert CPU offload (the default).</summary>
        public const int CpuMoeNone = 0;
        /// <summary>Offload only as many leading layers as the VRAM actually needs.</summary>
        public const int CpuMoeAuto = -1;

        static GgmlGlmNative()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        // Paths cross as UTF-8: ggml_fopen decodes them as UTF-8 on Windows,
        // while CharSet.Ansi would marshal the active code page.
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern IntPtr TSGgml_GlmLoadModel([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName, int tp);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmVocabSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmCtxSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmNPast(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmForward(IntPtr handle, int* tokens, int nTokens, float* logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmReset(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmRewind(IntPtr handle, int nPast);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmForwardBatchedDecode(IntPtr handle, int n, int[] slotIds,
            int[] tokens, int[] positions, float[] logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotAlloc(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSetActiveSlot(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotFree(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmFree(IntPtr handle);

        /// <param name="nGpu">GPUs to spread the layers over; 0 = every visible device.</param>
        /// <param name="nCpuMoe">Leading layers whose routed experts stay in system RAM;
        /// <see cref="CpuMoeAuto"/> offloads the fewest that make the model fit.</param>
        /// <param name="backendName">ggml backend registry name ("CUDA" / "Vulkan" / "Metal");
        /// null or empty accepts any GPU backend.</param>
        /// <param name="tp">Tensor-parallel ranks. 1 (the default) layer-splits the
        /// model across the visible GPUs; &gt;1 runs EVERY layer on every rank with
        /// the attention heads and the routed experts sharded, so all the GPUs work
        /// on each token instead of one at a time.</param>
        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads,
            int nCpuMoe = CpuMoeNone, string backendName = null, int tp = 1)
            => TSGgml_GlmLoadModel(ggufPath, nGpu, nCtx, nUbatch, nThreads, nCpuMoe, backendName ?? string.Empty, tp);

        public static int VocabSize(IntPtr handle) => TSGgml_GlmVocabSize(handle);
        public static int CtxSize(IntPtr handle) => TSGgml_GlmCtxSize(handle);
        public static int NPast(IntPtr handle) => TSGgml_GlmNPast(handle);

        /// <summary>Evaluate <paramref name="tokens"/> at the active slot's current
        /// position. <paramref name="logitsOut"/>, when non-null, receives the last
        /// token's logits.</summary>
        public static unsafe bool Forward(IntPtr handle, int[] tokens, float[] logitsOut)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
            {
                return TSGgml_GlmForward(handle, t, tokens.Length, l) != 0;
            }
        }

        public static void Reset(IntPtr handle) => TSGgml_GlmReset(handle);

        /// <summary>Drop the KV of the tokens after <paramref name="nPast"/>.</summary>
        public static bool Rewind(IntPtr handle, int nPast) => TSGgml_GlmRewind(handle, nPast) != 0;

        /// <summary>One decode step for several sequences at once — one token per
        /// sequence, sharing a single read of the weights. Returns false when the
        /// native side declines the batch (tensor parallelism, a position that
        /// disagrees with its slot, a repeated slot), in which case the caller
        /// stays on the per-sequence path.</summary>
        public static bool ForwardBatchedDecode(IntPtr handle, int[] slotIds, int[] tokens,
                                                int[] positions, float[] logitsOut)
            => TSGgml_GlmForwardBatchedDecode(handle, slotIds.Length, slotIds, tokens, positions, logitsOut) != 0;

        /// <summary>Allocate a sequence slot (own KV caches, shared weights);
        /// -1 on failure.</summary>
        public static int SlotAlloc(IntPtr handle) => TSGgml_GlmSlotAlloc(handle);

        /// <summary>Select the slot Forward / Reset / NPast act on.</summary>
        public static bool SetActiveSlot(IntPtr handle, int slotId) => TSGgml_GlmSetActiveSlot(handle, slotId) != 0;

        /// <summary>Free a slot's caches and the graphs built against them.</summary>
        public static bool SlotFree(IntPtr handle, int slotId) => TSGgml_GlmSlotFree(handle, slotId) != 0;

        public static void Free(IntPtr handle) => TSGgml_GlmFree(handle);
    }
}
