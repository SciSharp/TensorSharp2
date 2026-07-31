// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the DeepSeek V4 direct-CUDA kernels
// (native/kernels/tensorsharp_dsv4_kernels.cu, compiled to
// tensorsharp_dsv4_kernels.ptx). One instance per device; the module is
// loaded against whichever context is current at construction time.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class Dsv4Kernels : IDisposable
    {
        private const int BlockSize = 256;

        private readonly CudaModule module;

        private readonly IntPtr embed;
        private readonly IntPtr hcRms;
        private readonly IntPtr hcGatesComb;
        private readonly IntPtr hcCollapse;
        private readonly IntPtr hcPost;
        private readonly IntPtr hcHead;
        private readonly IntPtr rmsNorm;
        private readonly IntPtr attnPrep;
        private readonly IntPtr apeAdd;
        private readonly IntPtr compress;
        private readonly IntPtr persist;
        private readonly IntPtr idxPrep;
        private readonly IntPtr idxScores;
        private readonly IntPtr topk;
        private readonly IntPtr attention;
        private readonly IntPtr attnFinish;
        private readonly IntPtr regroup;
        private readonly IntPtr moeSelect;
        private readonly IntPtr moeCount;
        private readonly IntPtr moeScan;
        private readonly IntPtr moeScatter;
        private readonly IntPtr moeGateUp;
        private readonly IntPtr moeDown;
        private readonly IntPtr moeGateUpStaged;
        private readonly IntPtr moeDownStaged;
        private readonly IntPtr moeGateUpDecode;
        private readonly IntPtr moeDownDecode;
        private readonly IntPtr swigluClamp;
        private readonly IntPtr moeScatterAdd;

        private Dsv4Kernels(CudaModule module)
        {
            this.module = module;
            embed = module.GetFunction("ts_dsv4_embed_f32");
            hcRms = module.GetFunction("ts_dsv4_hc_rms_f32");
            hcGatesComb = module.GetFunction("ts_dsv4_hc_gates_comb_f32");
            hcCollapse = module.GetFunction("ts_dsv4_hc_collapse_f32");
            hcPost = module.GetFunction("ts_dsv4_hc_post_f32");
            hcHead = module.GetFunction("ts_dsv4_hc_head_f32");
            rmsNorm = module.GetFunction("ts_dsv4_rmsnorm_f32");
            attnPrep = module.GetFunction("ts_dsv4_attn_prep_f32");
            apeAdd = module.GetFunction("ts_dsv4_ape_add_f32");
            compress = module.GetFunction("ts_dsv4_compress_f32");
            persist = module.GetFunction("ts_dsv4_persist_f32");
            idxPrep = module.GetFunction("ts_dsv4_idx_prep_f32");
            idxScores = module.GetFunction("ts_dsv4_idx_scores_f32");
            topk = module.GetFunction("ts_dsv4_topk_f32");
            attention = module.GetFunction("ts_dsv4_attention_f32");
            attnFinish = module.GetFunction("ts_dsv4_attn_finish_f32");
            regroup = module.GetFunction("ts_dsv4_regroup_f32");
            moeSelect = module.GetFunction("ts_dsv4_moe_select_f32");
            moeCount = module.GetFunction("ts_dsv4_moe_count_i32");
            moeScan = module.GetFunction("ts_dsv4_moe_scan_i32");
            moeScatter = module.GetFunction("ts_dsv4_moe_scatter_i32");
            moeGateUp = module.GetFunction("ts_dsv4_moe_gateup_f32");
            moeDown = module.GetFunction("ts_dsv4_moe_down_f32");
            moeGateUpStaged = module.GetFunction("ts_dsv4_moe_gateup_staged_f32");
            moeDownStaged = module.GetFunction("ts_dsv4_moe_down_staged_f32");
            moeGateUpDecode = module.GetFunction("ts_dsv4_moe_gateup_decode_f32");
            moeDownDecode = module.GetFunction("ts_dsv4_moe_down_decode_f32");
            swigluClamp = module.GetFunction("ts_dsv4_swiglu_clamp_f32");
            moeScatterAdd = module.GetFunction("ts_dsv4_moe_scatter_add_f32");
        }

        public static Dsv4Kernels Create()
        {
            string path = CudaKernels.LocatePtxPath("tensorsharp_dsv4_kernels.ptx");
            if (path == null)
            {
                throw new InvalidOperationException(
                    "tensorsharp_dsv4_kernels.ptx could not be located next to the application or under " +
                    "a 'cuda_kernels' folder. Build TensorSharp.Backends.Cuda with nvcc on the PATH.");
            }
            return new Dsv4Kernels(CudaModule.LoadFromFile(path));
        }

        private static void Launch(IntPtr fn, uint gx, uint gy, uint gz, int block, uint sharedBytes, IntPtr stream, void** args)
        {
            CudaDriverApi.cuLaunchKernel(fn, gx, gy, gz, (uint)block, 1, 1, sharedBytes, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        private static uint CeilDiv(long n, int d) => (uint)Math.Max(1, (n + d - 1) / d);

        public void Embed(IntPtr w, IntPtr tokens, IntPtr xs, int wtype, long rowBytes, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = w, a1 = tokens, a2 = xs;
            int a3 = wtype; long a4 = rowBytes; int a5 = nt, a6 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(embed, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcRms(IntPtr xs, IntPtr inv, int nt, int flatDim, float eps, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = inv;
            int a2 = flatDim; float a3 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(hcRms, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        public void HcGatesComb(IntPtr mixes, IntPtr inv, IntPtr scale, IntPtr baseW, IntPtr pre, IntPtr post, IntPtr comb,
            int nt, int iters, float eps, IntPtr stream)
        {
            IntPtr a0 = mixes, a1 = inv, a2 = scale, a3 = baseW, a4 = pre, a5 = post, a6 = comb;
            int a7 = nt, a8 = iters; float a9 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(hcGatesComb, CeilDiv(nt, 128), 1, 1, 128, 0, stream, args);
        }

        public void HcCollapse(IntPtr xs, IntPtr pre, IntPtr cur, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = pre, a2 = cur;
            int a3 = nt, a4 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(hcCollapse, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcPost(IntPtr xs, IntPtr blockOut, IntPtr post, IntPtr comb, IntPtr xsOut, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = xs, a1 = blockOut, a2 = post, a3 = comb, a4 = xsOut;
            int a5 = nt, a6 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(hcPost, CeilDiv(4L * e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcHead(IntPtr x, IntPtr fn, IntPtr scale, IntPtr baseW, IntPtr cur, int e, int scaleCount, int baseCount, float eps, IntPtr stream)
        {
            IntPtr a0 = x, a1 = fn, a2 = scale, a3 = baseW, a4 = cur;
            int a5 = e, a6 = scaleCount, a7 = baseCount; float a8 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(hcHead, 1, 1, 1, BlockSize, 0, stream, args);
        }

        public void RmsNorm(IntPtr data, IntPtr weight, int rows, int n, float eps, IntPtr stream)
        {
            IntPtr a0 = data, a1 = weight;
            int a2 = rows, a3 = n; float a4 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(rmsNorm, (uint)rows, 1, 1, BlockSize, 0, stream, args);
        }

        public void AttnPrep(IntPtr q, IntPtr kvRaw, IntPtr kvNormW, IntPtr ropeTab, IntPtr ring,
            int p0, int ringRows, int nh, int hd, int nRot, float eps, int nt, IntPtr stream)
        {
            IntPtr a0 = q, a1 = kvRaw, a2 = kvNormW, a3 = ropeTab, a4 = ring;
            int a5 = p0, a6 = ringRows, a7 = nh, a8 = hd, a9 = nRot; float a10 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            Launch(attnPrep, (uint)(nh + 1), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void ApeAdd(IntPtr stScore, IntPtr ape, int p0, int ratio, int nt, int cw, IntPtr stream)
        {
            IntPtr a0 = stScore, a1 = ape;
            int a2 = p0, a3 = ratio, a4 = nt, a5 = cw;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(apeAdd, CeilDiv(cw, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void Compress(IntPtr stKv, IntPtr stScore, IntPtr histKv, IntPtr histScore, IntPtr normW, IntPtr ropeTab, IntPtr cache,
            long firstBoundary, int nBlocks, int p0, int ratio, int coff, int head, int stateSize, int nRot, float eps, IntPtr stream)
        {
            IntPtr a0 = stKv, a1 = stScore, a2 = histKv, a3 = histScore, a4 = normW, a5 = ropeTab, a6 = cache;
            long a7 = firstBoundary;
            int a8 = nBlocks, a9 = p0, a10 = ratio, a11 = coff, a12 = head, a13 = stateSize, a14 = nRot;
            float a15 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
            Launch(compress, (uint)nBlocks, 1, 1, BlockSize, 0, stream, args);
        }

        public void Persist(IntPtr stKv, IntPtr stScore, IntPtr histKv, IntPtr histScore,
            int p0, int nt, int stateSize, int cw, IntPtr stream)
        {
            IntPtr a0 = stKv, a1 = stScore, a2 = histKv, a3 = histScore;
            int a4 = p0, a5 = nt, a6 = stateSize, a7 = cw;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            long np = Math.Min(nt, stateSize);
            Launch(persist, CeilDiv(np * cw, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void IdxPrep(IntPtr iq, IntPtr iw, IntPtr ropeTab, int p0, int ih, int id, int nRot, float iwScale, int nt, IntPtr stream)
        {
            IntPtr a0 = iq, a1 = iw, a2 = ropeTab;
            int a3 = p0, a4 = ih, a5 = id, a6 = nRot; float a7 = iwScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(idxPrep, (uint)ih, (uint)nt, 1, 128, 0, stream, args);
        }

        public void IdxScores(IntPtr iq, IntPtr iw, IntPtr lidK, IntPtr scores,
            int p0, int ratio, int ih, int id, int nt, int scoreStride, int rowsMax, IntPtr stream)
        {
            IntPtr a0 = iq, a1 = iw, a2 = lidK, a3 = scores;
            int a4 = p0, a5 = ratio, a6 = ih, a7 = id, a8 = nt, a9 = scoreStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            uint shared = (uint)((ih * id + ih) * sizeof(float));
            Launch(idxScores, CeilDiv(rowsMax, 8), (uint)nt, 1, BlockSize, shared, stream, args);
        }

        public void TopK(IntPtr scores, IntPtr topkIdx, IntPtr topkCnt, int p0, int ratio, int k, int scoreStride, int nt, IntPtr stream)
        {
            IntPtr a0 = scores, a1 = topkIdx, a2 = topkCnt;
            int a3 = p0, a4 = ratio, a5 = k, a6 = scoreStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(topk, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        public void Attention(IntPtr q, IntPtr ring, IntPtr comp, IntPtr topkIdx, IntPtr topkCnt, IntPtr sinks, IntPtr attnO,
            int p0, int nSwa, int ringRows, int nh, int hd, int mode, int ratio, int k, float kqScale, int nt, IntPtr stream)
        {
            IntPtr a0 = q, a1 = ring, a2 = comp, a3 = topkIdx, a4 = topkCnt, a5 = sinks, a6 = attnO;
            int a7 = p0, a8 = nSwa, a9 = ringRows, a10 = nh, a11 = hd, a12 = mode, a13 = ratio, a14 = k;
            float a15 = kqScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
            Launch(attention, (uint)nh, (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void AttnFinish(IntPtr attnO, IntPtr ropeTab, IntPtr dst, int p0, int nh, int hd, int nRot, int headsPerGroup, int nt, IntPtr stream)
        {
            IntPtr a0 = attnO, a1 = ropeTab, a2 = dst;
            int a3 = p0, a4 = nh, a5 = hd, a6 = nRot, a7 = headsPerGroup, a8 = nt;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(attnFinish, (uint)nh, (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void Regroup(IntPtr oGtmp, IntPtr oG, int g, int nt, int r, IntPtr stream)
        {
            IntPtr a0 = oGtmp, a1 = oG;
            int a2 = g, a3 = nt, a4 = r;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(regroup, CeilDiv((long)g * nt * r, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeSelect(IntPtr logits, IntPtr bias, IntPtr tid2eid, IntPtr tokens, IntPtr sel, IntPtr wOut,
            int nExpert, int nUsed, int norm, float wScale, int nt, IntPtr stream)
        {
            IntPtr a0 = logits, a1 = bias, a2 = tid2eid, a3 = tokens, a4 = sel, a5 = wOut;
            int a6 = nExpert, a7 = nUsed, a8 = norm; float a9 = wScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(moeSelect, (uint)nt, 1, 1, BlockSize, (uint)(nExpert * sizeof(float)), stream, args);
        }

        public void MoeCount(IntPtr sel, IntPtr counts, int s, IntPtr stream)
        {
            IntPtr a0 = sel, a1 = counts;
            int a2 = s;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(moeCount, CeilDiv(s, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeScan(IntPtr counts, IntPtr offsets, IntPtr cursors, int nExpert, IntPtr stream)
        {
            IntPtr a0 = counts, a1 = offsets, a2 = cursors;
            int a3 = nExpert;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(moeScan, 1, 1, 1, 32, 0, stream, args);
        }

        public void MoeScatter(IntPtr sel, IntPtr cursors, IntPtr rowOfSlot, IntPtr slotToken, int s, int nUsed, IntPtr stream)
        {
            IntPtr a0 = sel, a1 = cursors, a2 = rowOfSlot, a3 = slotToken;
            int a4 = s, a5 = nUsed;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(moeScatter, CeilDiv(s, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeGateUp(IntPtr gateW, IntPtr upW, IntPtr act, IntPtr counts, IntPtr offsets, IntPtr slotToken,
            IntPtr gateOut, IntPtr upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = act, a3 = counts, a4 = offsets, a5 = slotToken, a6 = gateOut, a7 = upOut;
            int a8 = wtype, a9 = ff, a10 = inDim; long a11 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            Launch(moeGateUp, (uint)nExpert, CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        public void MoeDown(IntPtr downW, IntPtr act, IntPtr counts, IntPtr offsets, IntPtr downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = act, a2 = counts, a3 = offsets, a4 = downOut;
            int a5 = wtype, a6 = e, a7 = ff; long a8 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(moeDown, (uint)nExpert, CeilDiv(e, 32), 1, BlockSize, 0, stream, args);
        }

        // Staged-weight variants (prefill): each weight row is decoded ONCE into
        // registers and reused across the expert's member tokens, reading the
        // activations from the dense split-q8_1 scratch.
        private const int StageWarps = 8;
        // must match TS_DSV4_STAGE_ROWS in tensorsharp_dsv4_kernels.cu
        private const int StageRows = 2;

        /// <summary>The register-staged kernels hold inDim/1024 sub-blocks per
        /// lane, so they cover input widths up to 32 * 32 * MAX_SUBS.</summary>
        public static bool StagedSupports(int inDim) => inDim % 1024 == 0 && inDim / 1024 <= 4;

        public void MoeGateUpStaged(IntPtr gateW, IntPtr upW, IntPtr actQs, IntPtr actD,
            IntPtr counts, IntPtr offsets, IntPtr slotToken,
            IntPtr gateOut, IntPtr upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = actQs, a3 = actD, a4 = counts, a5 = offsets, a6 = slotToken,
                a7 = gateOut, a8 = upOut;
            int a9 = wtype, a10 = ff, a11 = inDim; long a12 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            // grid.x = row tiles, grid.y = experts (blockIdx.x varies fastest,
            // so one expert's blocks stay co-resident and share its activations)
            uint rowTiles = CeilDiv(ff, StageWarps * StageRows);
            Launch(moeGateUpStaged, rowTiles, (uint)nExpert, 2, BlockSize, 0, stream, args);
        }

        public void MoeDownStaged(IntPtr downW, IntPtr actQs, IntPtr actD, IntPtr counts, IntPtr offsets, IntPtr downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = actQs, a2 = actD, a3 = counts, a4 = offsets, a5 = downOut;
            int a6 = wtype, a7 = e, a8 = ff; long a9 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            uint rowTiles = CeilDiv(e, StageWarps * StageRows);
            Launch(moeDownStaged, rowTiles, (uint)nExpert, 1, BlockSize, 0, stream, args);
        }

        public void MoeGateUpDecode(IntPtr gateW, IntPtr upW, IntPtr act, IntPtr sel,
            IntPtr gateOut, IntPtr upOut, int wtype, int ff, int inDim, long rowBytes, int nUsed, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = act, a3 = sel, a4 = gateOut, a5 = upOut;
            int a6 = wtype, a7 = ff, a8 = inDim; long a9 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(moeGateUpDecode, (uint)nUsed, CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        public void MoeDownDecode(IntPtr downW, IntPtr act, IntPtr sel, IntPtr downOut,
            int wtype, int e, int ff, long rowBytes, int nUsed, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = act, a2 = sel, a3 = downOut;
            int a4 = wtype, a5 = e, a6 = ff; long a7 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(moeDownDecode, (uint)nUsed, CeilDiv(e, 32), 1, BlockSize, 0, stream, args);
        }

        public void SwigluClamp(IntPtr gate, IntPtr up, long n, float limit, IntPtr stream)
        {
            IntPtr a0 = gate, a1 = up;
            long a2 = n; float a3 = limit; int a4 = limit > 1e-6f ? 1 : 0;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(swigluClamp, CeilDiv(n, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeScatterAdd(IntPtr downOut, IntPtr rowOfSlot, IntPtr wSel, IntPtr shexp, IntPtr ffnOut,
            int nt, int nUsed, int e, IntPtr stream)
        {
            IntPtr a0 = downOut, a1 = rowOfSlot, a2 = wSel, a3 = shexp, a4 = ffnOut;
            int a5 = nt, a6 = nUsed, a7 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(moeScatterAdd, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void Dispose()
        {
            module.Dispose();
        }
    }
}
