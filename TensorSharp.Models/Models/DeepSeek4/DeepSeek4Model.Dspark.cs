// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DSpark speculative decoding for DeepSeek V4: the model-side contract over the
// direct-CUDA engine's drafter (see Dsv4CudaEngine.Dspark.cs for the algorithm).
//
// DSpark is a BLOCK drafter -- one forward pass proposes the whole speculative
// window -- so it plugs into the shared draft/verify/rollback core through
// IMtpSpeculativeModel.MtpDraftBlock instead of the per-token MtpDraftStep, and
// reports the wider hidden row (the concatenated target-layer features its
// main_proj consumes) through MtpHiddenSize.
//
// Rollback is free here: the verify pass wrote correct KV for every token it
// processed, and the engine's rings are sized so a rejected tail can never
// alias a row a later pass still reads, so partial acceptance only rewinds the
// position counter (MtpVerifyPersistsAcceptedKv).
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : IMtpSpeculativeModel
    {
        /// <summary>Path of the DSpark drafter GGUF, if one was configured
        /// (<c>--draft-model</c> / <c>TS_DSV4_DSPARK</c>).</summary>
        internal static string ResolveDsparkPath(string explicitPath)
        {
            string path = !string.IsNullOrWhiteSpace(explicitPath)
                ? explicitPath
                : Environment.GetEnvironmentVariable("TS_DSV4_DSPARK");
            return string.IsNullOrWhiteSpace(path) ? null : path;
        }

        /// <summary>True when a DSpark drafter is loaded and usable, on either
        /// executor: the direct-CUDA engine or the native ggml one.</summary>
        public bool HasMtp => (_cudaExec != null && _cudaExec.HasDspark) || _nativeDsparkBlock > 0;

        /// <summary>Both DSpark implementations run the batched verify on their
        /// own accelerated path.</summary>
        public bool MtpSpeculationProfitable => HasMtp;

        /// <summary>
        /// Width of the hidden row handed between the trunk and the drafter. The
        /// direct-CUDA engine feeds the drafter its target features through the
        /// host, so it reports their real width; the ggml engine commits the
        /// drafter key ring inside the trunk graph and needs nothing from the
        /// host, so its rows are a single (unused) float.
        /// </summary>
        public int MtpHiddenSize => _cudaExec != null && _cudaExec.HasDspark ? _cudaExec.DsparkFeatureSize : 1;

        public int MtpDraftBlockSize => _cudaExec != null ? _cudaExec.DsparkBlockSize : _nativeDsparkBlock;

        /// <summary>Prefill in whole engine micro-batches: the MoE trunk reads
        /// every touched expert's weights once per micro-batch, so halving the
        /// chunk nearly doubles prefill weight traffic per token.</summary>
        public int MtpPrefillChunkSize => _cudaExec != null ? _cudaExec.UBatch : _nativeUBatch;

        /// <summary>Both engines replay the drafter key ring from their own
        /// on-device features, so prefill needs no per-row readback.</summary>
        public bool MtpPrefillSelfCatchUp => true;

        /// <summary>The verify batch writes reusable KV for every row, and the
        /// engine's rings tolerate a rejected tail, so partial acceptance needs
        /// no re-forward.</summary>
        public bool MtpVerifyPersistsAcceptedKv => true;

        /// <summary>Trunk position tracked by the executor (the base class's
        /// counter is not used by the DSV4 executors).</summary>
        public new int CacheSeqLen => _cudaExec != null
            ? _cudaExec.NPast
            : (_handle != IntPtr.Zero ? GgmlDeepSeek4Native.NPast(_handle) : base.CacheSeqLen);

        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            RequireDspark();
            lock (_sync)
            {
                if (_cudaExec != null)
                {
                    _cudaExec.ForwardSpec(tokens, hAllOut, logitsOut, allLogitsRows);
                    return;
                }

                // The native executor emits per-row logits only for the verify
                // batch; a prefill chunk takes the ordinary forward, which also
                // commits the drafter key ring, and reports the last row.
                bool ok = allLogitsRows
                    ? GgmlDeepSeek4Native.ForwardSpec(_handle, tokens, logitsOut)
                    : GgmlDeepSeek4Native.Forward(_handle, tokens, logitsOut);
                if (!ok)
                    throw new InvalidOperationException("DeepSeek V4 native speculative forward failed (see stderr).");
                if (hAllOut != null && hAllOut.Length > 0)
                    Array.Clear(hAllOut, 0, hAllOut.Length);   // the drafter reads its features on-device
            }
        }

        public int MtpDraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            RequireDspark();
            lock (_sync)
            {
                if (_cudaExec != null)
                    return _cudaExec.DsparkDraft(lastToken, hPrev, position, draftOut, confOut);
                return GgmlDeepSeek4Native.DsparkDraft(_handle, lastToken, draftOut, confOut);
            }
        }

        /// <summary>Row k of <paramref name="hRows"/> is the hidden state of the
        /// token BEFORE tokens[k], i.e. the features of position startPos+k-1 --
        /// which is exactly the position whose drafter key it writes.</summary>
        public void MtpCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            RequireDspark();
            lock (_sync)
            {
                // Only the direct-CUDA drafter is fed from the host; the native
                // one commits its ring inside the trunk graph.
                if (_cudaExec != null)
                    _cudaExec.DsparkCatchUp(hRows, tokens.Length, startPos - 1);
            }
        }

        public void MtpRewindCache(int length)
        {
            RequireDspark();
            lock (_sync)
            {
                if (_cudaExec != null)
                    _cudaExec.Rewind(length);
                else
                    GgmlDeepSeek4Native.Rewind(_handle, length);
            }
        }

        /// <summary>DSpark drafts a whole block per step; the per-token draft
        /// entry point is never used.</summary>
        public void MtpDraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => throw new NotSupportedException("DeepSeek V4 drafts whole blocks; use MtpDraftBlock.");

        // The DSV4 caches are preallocated for the whole context and hold no
        // recurrent state that a rejected draft could corrupt.
        public void MtpEnsureCapacity(int requiredSeqLen) { }

        public void MtpSnapshotRecurrentState() { }

        public void MtpRestoreRecurrentState() { }

        private void RequireDspark()
        {
            if (!HasMtp)
                throw new InvalidOperationException("No DSpark drafter is loaded for this DeepSeek V4 model.");
        }
    }
}
