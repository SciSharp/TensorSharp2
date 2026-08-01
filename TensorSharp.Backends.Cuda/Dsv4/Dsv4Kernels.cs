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
        private const int HC = 4; // hyper-connection streams (Dsv4CudaEngine.HC)

        private readonly CudaModule module;

        private readonly IntPtr embed;
        private readonly IntPtr hcRms;
        private readonly IntPtr hcGatesComb;
        private readonly IntPtr hcCollapse;
        private readonly IntPtr hcPost;
        private readonly IntPtr hcHead;
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
        private readonly IntPtr moeScatterAdd;
        private readonly IntPtr hcMean;
        private readonly IntPtr dsparkPrep;
        private readonly IntPtr dsparkGather;
        private readonly IntPtr dsparkArgmax;
        private readonly IntPtr dsparkConf;

        private Dsv4Kernels(CudaModule module)
        {
            this.module = module;
            embed = module.GetFunction("ts_dsv4_embed_f32");
            hcRms = module.GetFunction("ts_dsv4_hc_rms_f32");
            hcGatesComb = module.GetFunction("ts_dsv4_hc_gates_comb_f32");
            hcCollapse = module.GetFunction("ts_dsv4_hc_collapse_f32");
            hcPost = module.GetFunction("ts_dsv4_hc_post_f32");
            hcHead = module.GetFunction("ts_dsv4_hc_head_f32");
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
            moeScatterAdd = module.GetFunction("ts_dsv4_moe_scatter_add_f32");
            hcMean = module.GetFunction("ts_dsv4_hc_mean_f32");
            dsparkPrep = module.GetFunction("ts_dsv4_dspark_prep_f32");
            dsparkGather = module.GetFunction("ts_dsv4_dspark_gather_f32");
            dsparkArgmax = module.GetFunction("ts_dsv4_dspark_argmax_f32");
            dsparkConf = module.GetFunction("ts_dsv4_dspark_conf_f32");
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

        /// <summary>Device pointer behind an engine scratch tensor (null =>
        /// nullptr, which several kernels accept as "argument absent").</summary>
        private static IntPtr P(Tensor t) => Dsv4CudaEngine.Ptr(t);

        public void Embed(IntPtr w, Tensor tokens, Tensor xs, int wtype, long rowBytes, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = w, a1 = P(tokens), a2 = P(xs);
            int a3 = wtype; long a4 = rowBytes; int a5 = nt, a6 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(embed, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcRms(Tensor xs, Tensor inv, int nt, int flatDim, float eps, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(inv);
            int a2 = flatDim; float a3 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(hcRms, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        public void HcGatesComb(Tensor mixes, Tensor inv, IntPtr scale, IntPtr baseW, Tensor pre, Tensor post, Tensor comb,
            int nt, int iters, float eps, IntPtr stream)
        {
            IntPtr a0 = P(mixes), a1 = P(inv), a2 = scale, a3 = baseW, a4 = P(pre), a5 = P(post), a6 = P(comb);
            int a7 = nt, a8 = iters; float a9 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(hcGatesComb, CeilDiv(nt, 128), 1, 1, 128, 0, stream, args);
        }

        public void HcCollapse(Tensor xs, Tensor pre, Tensor cur, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(pre), a2 = P(cur);
            int a3 = nt, a4 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(hcCollapse, CeilDiv(e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcPost(Tensor xs, Tensor blockOut, Tensor post, Tensor comb, Tensor xsOut, int nt, int e, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(blockOut), a2 = P(post), a3 = P(comb), a4 = P(xsOut);
            int a5 = nt, a6 = e;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(hcPost, CeilDiv(4L * e, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void HcHead(Tensor x, IntPtr fn, IntPtr scale, IntPtr baseW, Tensor cur, int e, int scaleCount, int baseCount, float eps, IntPtr stream, int rows = 1)
        {
            IntPtr a0 = P(x), a1 = fn, a2 = scale, a3 = baseW, a4 = P(cur);
            int a5 = e, a6 = scaleCount, a7 = baseCount; float a8 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(hcHead, (uint)Math.Max(rows, 1), 1, 1, BlockSize, 0, stream, args);
        }

        public void AttnPrep(Tensor q, Tensor kvRaw, IntPtr kvNormW, IntPtr ropeTab, IntPtr ring,
            int p0, int ringRows, int nh, int hd, int nRot, float eps, int nt, IntPtr stream)
        {
            IntPtr a0 = P(q), a1 = P(kvRaw), a2 = kvNormW, a3 = ropeTab, a4 = ring;
            int a5 = p0, a6 = ringRows, a7 = nh, a8 = hd, a9 = nRot; float a10 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            Launch(attnPrep, (uint)(nh + 1), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void ApeAdd(Tensor stScore, IntPtr ape, int p0, int ratio, int nt, int cw, IntPtr stream)
        {
            IntPtr a0 = P(stScore), a1 = ape;
            int a2 = p0, a3 = ratio, a4 = nt, a5 = cw;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(apeAdd, CeilDiv(cw, BlockSize), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void Compress(Tensor stKv, Tensor stScore, IntPtr histKv, IntPtr histScore, IntPtr normW, IntPtr ropeTab, IntPtr cache,
            long firstBoundary, int nBlocks, int p0, int ratio, int coff, int head, int stateSize, int nRot, float eps, IntPtr stream)
        {
            IntPtr a0 = P(stKv), a1 = P(stScore), a2 = histKv, a3 = histScore, a4 = normW, a5 = ropeTab, a6 = cache;
            long a7 = firstBoundary;
            int a8 = nBlocks, a9 = p0, a10 = ratio, a11 = coff, a12 = head, a13 = stateSize, a14 = nRot;
            float a15 = eps;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
            Launch(compress, (uint)nBlocks, 1, 1, BlockSize, 0, stream, args);
        }

        public void Persist(Tensor stKv, Tensor stScore, IntPtr histKv, IntPtr histScore,
            int p0, int nt, int stateSize, int cw, IntPtr stream)
        {
            IntPtr a0 = P(stKv), a1 = P(stScore), a2 = histKv, a3 = histScore;
            int a4 = p0, a5 = nt, a6 = stateSize, a7 = cw;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            long np = Math.Min(nt, stateSize);
            Launch(persist, CeilDiv(np * cw, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void IdxPrep(Tensor iq, Tensor iw, IntPtr ropeTab, int p0, int ih, int id, int nRot, float iwScale, int nt, IntPtr stream)
        {
            IntPtr a0 = P(iq), a1 = P(iw), a2 = ropeTab;
            int a3 = p0, a4 = ih, a5 = id, a6 = nRot; float a7 = iwScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(idxPrep, (uint)ih, (uint)nt, 1, 128, 0, stream, args);
        }

        public void IdxScores(Tensor iq, Tensor iw, IntPtr lidK, Tensor scores,
            int p0, int ratio, int ih, int id, int nt, int scoreStride, int rowsMax, IntPtr stream)
        {
            IntPtr a0 = P(iq), a1 = P(iw), a2 = lidK, a3 = P(scores);
            int a4 = p0, a5 = ratio, a6 = ih, a7 = id, a8 = nt, a9 = scoreStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            uint shared = (uint)((ih * id + ih) * sizeof(float));
            Launch(idxScores, CeilDiv(rowsMax, 8), (uint)nt, 1, BlockSize, shared, stream, args);
        }

        public void TopK(Tensor scores, Tensor topkIdx, Tensor topkCnt, int p0, int ratio, int k, int scoreStride, int nt, IntPtr stream)
        {
            IntPtr a0 = P(scores), a1 = P(topkIdx), a2 = P(topkCnt);
            int a3 = p0, a4 = ratio, a5 = k, a6 = scoreStride;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(topk, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        public void Attention(Tensor q, IntPtr ring, IntPtr comp, Tensor topkIdx, Tensor topkCnt, IntPtr sinks, Tensor attnO,
            int p0, int nSwa, int ringRows, int nh, int hd, int mode, int ratio, int k, float kqScale, int nt, IntPtr stream)
        {
            IntPtr a0 = P(q), a1 = ring, a2 = comp, a3 = P(topkIdx), a4 = P(topkCnt), a5 = sinks, a6 = P(attnO);
            int a7 = p0, a8 = nSwa, a9 = ringRows, a10 = nh, a11 = hd, a12 = mode, a13 = ratio, a14 = k;
            float a15 = kqScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
            Launch(attention, (uint)nh, (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void AttnFinish(Tensor attnO, IntPtr ropeTab, Tensor dst, int p0, int nh, int hd, int nRot, int headsPerGroup, int nt, IntPtr stream)
        {
            IntPtr a0 = P(attnO), a1 = ropeTab, a2 = P(dst);
            int a3 = p0, a4 = nh, a5 = hd, a6 = nRot, a7 = headsPerGroup, a8 = nt;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(attnFinish, (uint)nh, (uint)nt, 1, BlockSize, 0, stream, args);
        }

        public void Regroup(Tensor oGtmp, Tensor oG, int g, int nt, int r, IntPtr stream)
        {
            IntPtr a0 = P(oGtmp), a1 = P(oG);
            int a2 = g, a3 = nt, a4 = r;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(regroup, CeilDiv((long)g * nt * r, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        // ---- DSpark drafter ----

        /// <summary>Mean over the hyper-connection streams of every row, written
        /// into column block <paramref name="dstOff"/> of a [nt, dstStride] buffer
        /// (the drafter's concatenated target-layer features).</summary>
        public void HcMean(Tensor xs, Tensor dst, int nt, int e, int dstStride, int dstOff, IntPtr stream)
        {
            IntPtr a0 = P(xs), a1 = P(dst);
            int a2 = nt, a3 = e, a4 = HC, a5 = dstStride, a6 = dstOff;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(hcMean, CeilDiv((long)nt * e, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Drafter RMS-norm + RoPE with the rope position
        /// (<paramref name="pos0"/>) and the destination slot
        /// (<paramref name="slot0"/> mod <paramref name="slotMod"/>) decoupled.
        /// <paramref name="kvOnly"/> skips the query heads (catch-up pass).</summary>
        public void DsparkPrep(Tensor q, Tensor kvRaw, IntPtr kvNormW, IntPtr ropeTab, IntPtr kvOut,
            int pos0, int slot0, int slotMod, int nh, int hd, int nRot, float eps, int nt, bool kvOnly, IntPtr stream)
        {
            IntPtr a0 = P(q), a1 = P(kvRaw), a2 = kvNormW, a3 = ropeTab, a4 = kvOut;
            int a5 = pos0, a6 = slot0, a7 = slotMod, a8 = nh, a9 = hd, a10 = nRot;
            float a11 = eps;
            int a12 = kvOnly ? 1 : 0;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            Launch(dsparkPrep, kvOnly ? 1u : (uint)(nh + 1), (uint)nt, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Copy the Markov embedding row of the previous token
        /// (<paramref name="tokSlot"/> &lt; 0 selects <paramref name="anchorTok"/>).</summary>
        public void DsparkGather(IntPtr w1, Tensor toks, int tokSlot, int anchorTok, Tensor dst, int rank, IntPtr stream)
        {
            IntPtr a0 = w1, a1 = P(toks);
            int a2 = tokSlot, a3 = anchorTok;
            IntPtr a4 = P(dst);
            int a5 = rank;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(dsparkGather, 1, 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>toks[slot] = argmax(logits + bias) over the vocabulary.</summary>
        public void DsparkArgmax(Tensor logits, Tensor bias, Tensor toks, int slot, int vocab, IntPtr stream)
        {
            IntPtr a0 = P(logits), a1 = P(bias), a2 = P(toks);
            int a3 = slot, a4 = vocab;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(dsparkArgmax, 1, 1, 1, BlockSize, 0, stream, args);
        }

        /// <summary>Per-block-position acceptance probability from the
        /// confidence head.</summary>
        public void DsparkConf(Tensor x, Tensor w1Rows, IntPtr proj, Tensor conf, int e, int rank, int nt, IntPtr stream)
        {
            IntPtr a0 = P(x), a1 = P(w1Rows), a2 = proj, a3 = P(conf);
            int a4 = e, a5 = rank;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(dsparkConf, (uint)nt, 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeSelect(Tensor logits, IntPtr bias, IntPtr tid2eid, Tensor tokens, Tensor sel, Tensor wOut,
            int nExpert, int nUsed, int norm, float wScale, int nt, IntPtr stream)
        {
            IntPtr a0 = P(logits), a1 = bias, a2 = tid2eid, a3 = P(tokens), a4 = P(sel), a5 = P(wOut);
            int a6 = nExpert, a7 = nUsed, a8 = norm; float a9 = wScale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(moeSelect, (uint)nt, 1, 1, BlockSize, (uint)(nExpert * sizeof(float)), stream, args);
        }

        public void MoeCount(Tensor sel, Tensor counts, int s, IntPtr stream)
        {
            IntPtr a0 = P(sel), a1 = P(counts);
            int a2 = s;
            void** args = stackalloc void*[] { &a0, &a1, &a2 };
            Launch(moeCount, CeilDiv(s, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeScan(Tensor counts, Tensor offsets, Tensor cursors, int nExpert, IntPtr stream)
        {
            IntPtr a0 = P(counts), a1 = P(offsets), a2 = P(cursors);
            int a3 = nExpert;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
            Launch(moeScan, 1, 1, 1, 32, 0, stream, args);
        }

        public void MoeScatter(Tensor sel, Tensor cursors, Tensor rowOfSlot, Tensor slotToken, int s, int nUsed, IntPtr stream)
        {
            IntPtr a0 = P(sel), a1 = P(cursors), a2 = P(rowOfSlot), a3 = P(slotToken);
            int a4 = s, a5 = nUsed;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(moeScatter, CeilDiv(s, BlockSize), 1, 1, BlockSize, 0, stream, args);
        }

        public void MoeGateUp(IntPtr gateW, IntPtr upW, Tensor act, Tensor counts, Tensor offsets, Tensor slotToken,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(act), a3 = P(counts), a4 = P(offsets), a5 = P(slotToken), a6 = P(gateOut), a7 = P(upOut);
            int a8 = wtype, a9 = ff, a10 = inDim; long a11 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            Launch(moeGateUp, (uint)nExpert, CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        public void MoeDown(IntPtr downW, Tensor act, Tensor counts, Tensor offsets, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(counts), a3 = P(offsets), a4 = P(downOut);
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

        public void MoeGateUpStaged(IntPtr gateW, IntPtr upW, Tensor actQs, Tensor actD,
            Tensor counts, Tensor offsets, Tensor slotToken,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(actQs), a3 = P(actD), a4 = P(counts), a5 = P(offsets), a6 = P(slotToken),
                a7 = P(gateOut), a8 = P(upOut);
            int a9 = wtype, a10 = ff, a11 = inDim; long a12 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12 };
            // grid.x = row tiles, grid.y = experts (blockIdx.x varies fastest,
            // so one expert's blocks stay co-resident and share its activations)
            uint rowTiles = CeilDiv(ff, StageWarps * StageRows);
            Launch(moeGateUpStaged, rowTiles, (uint)nExpert, 2, BlockSize, 0, stream, args);
        }

        public void MoeDownStaged(IntPtr downW, Tensor actQs, Tensor actD, Tensor counts, Tensor offsets, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int nExpert, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = P(actQs), a2 = P(actD), a3 = P(counts), a4 = P(offsets), a5 = P(downOut);
            int a6 = wtype, a7 = e, a8 = ff; long a9 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            uint rowTiles = CeilDiv(e, StageWarps * StageRows);
            Launch(moeDownStaged, rowTiles, (uint)nExpert, 1, BlockSize, 0, stream, args);
        }

        public void MoeGateUpDecode(IntPtr gateW, IntPtr upW, Tensor act, Tensor sel,
            Tensor gateOut, Tensor upOut, int wtype, int ff, int inDim, long rowBytes, int nUsed, IntPtr stream)
        {
            IntPtr a0 = gateW, a1 = upW, a2 = P(act), a3 = P(sel), a4 = P(gateOut), a5 = P(upOut);
            int a6 = wtype, a7 = ff, a8 = inDim; long a9 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9 };
            Launch(moeGateUpDecode, (uint)nUsed, CeilDiv(ff, 32), 2, BlockSize, 0, stream, args);
        }

        public void MoeDownDecode(IntPtr downW, Tensor act, Tensor sel, Tensor downOut,
            int wtype, int e, int ff, long rowBytes, int nUsed, IntPtr stream)
        {
            IntPtr a0 = downW, a1 = P(act), a2 = P(sel), a3 = P(downOut);
            int a4 = wtype, a5 = e, a6 = ff; long a7 = rowBytes;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
            Launch(moeDownDecode, (uint)nUsed, CeilDiv(e, 32), 1, BlockSize, 0, stream, args);
        }

        public void MoeScatterAdd(Tensor downOut, Tensor rowOfSlot, Tensor wSel, Tensor shexp, Tensor ffnOut,
            int nt, int nUsed, int e, IntPtr stream)
        {
            IntPtr a0 = P(downOut), a1 = P(rowOfSlot), a2 = P(wSel), a3 = P(shexp), a4 = P(ffnOut);
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
