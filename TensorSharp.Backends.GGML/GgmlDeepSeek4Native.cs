// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the native DeepSeek V4 (Flash) whole-model executor
// (TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp). The native side loads the
// (split) GGUF itself, places the layers across every visible GPU, owns the
// DSV4 KV caches, and runs prefill/decode ubatches through ggml_backend_sched.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static class GgmlDeepSeek4Native
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Conv = CallingConvention.Cdecl;

        static GgmlDeepSeek4Native()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        [DllImport(DllName, CallingConvention = Conv, CharSet = CharSet.Ansi)]
        private static extern IntPtr TSGgml_Dsv4LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_Dsv4VocabSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_Dsv4CtxSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_Dsv4NPast(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_Dsv4Forward(IntPtr handle, int* tokens, int nTokens, float* logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_Dsv4Reset(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_Dsv4Free(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_Dsv4SlotAlloc(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_Dsv4SetActiveSlot(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_Dsv4SlotFree(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_Dsv4ForwardBatchedDecode(
            IntPtr handle, int n, int* slotIds, int* tokens, int* positions, float* logitsOut);

        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads)
            => TSGgml_Dsv4LoadModel(ggufPath, nGpu, nCtx, nUbatch, nThreads);

        public static int VocabSize(IntPtr handle) => TSGgml_Dsv4VocabSize(handle);
        public static int CtxSize(IntPtr handle) => TSGgml_Dsv4CtxSize(handle);
        public static int NPast(IntPtr handle) => TSGgml_Dsv4NPast(handle);

        public static unsafe bool Forward(IntPtr handle, int[] tokens, float[] logitsOut)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
            {
                return TSGgml_Dsv4Forward(handle, t, tokens.Length, l) == 0;
            }
        }

        public static void Reset(IntPtr handle) => TSGgml_Dsv4Reset(handle);
        public static void Free(IntPtr handle) => TSGgml_Dsv4Free(handle);

        /// <summary>Allocate a new sequence slot (own KV caches, shared
        /// weights). Returns the slot id, or -1 on failure (e.g. VRAM).</summary>
        public static int SlotAlloc(IntPtr handle) => TSGgml_Dsv4SlotAlloc(handle);

        /// <summary>Select the slot Forward/Reset/NPast act on.</summary>
        public static bool SetActiveSlot(IntPtr handle, int slotId)
            => TSGgml_Dsv4SetActiveSlot(handle, slotId) == 0;

        /// <summary>Free a slot's caches (and its cached graphs). The active
        /// slot cannot be freed.</summary>
        public static bool SlotFree(IntPtr handle, int slotId)
            => TSGgml_Dsv4SlotFree(handle, slotId) == 0;

        /// <summary>True token-batched decode: one token per slot in a single
        /// fused graph. logitsOut receives n rows of vocab-size floats; every
        /// slot advances one position on success. Returns false when the step
        /// can't be batched (caller falls back to per-slot forwards).</summary>
        public static unsafe bool ForwardBatchedDecode(
            IntPtr handle, int[] slotIds, int[] tokens, int[] positions, float[] logitsOut)
        {
            fixed (int* s = slotIds)
            fixed (int* t = tokens)
            fixed (int* p = positions)
            fixed (float* l = logitsOut)
            {
                return TSGgml_Dsv4ForwardBatchedDecode(handle, slotIds.Length, s, t, p, l) == 0;
            }
        }
    }
}
