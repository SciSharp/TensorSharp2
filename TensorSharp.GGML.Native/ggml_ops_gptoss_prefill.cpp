// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#include "ggml_ops_internal.h"
#include "ggml_ops_transformer_common.h"
#include "ggml_ops_gptoss_kv.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace tsg;

// ============================================================================
// GPT-OSS MODEL-WIDE prefill: N tokens through every layer (attention + MoE
// FFN) plus the folded final norm and LM head as ONE GGML graph.
//
// This is the sibling of TSGgml_GptOssModelDecode and of the Gemma 4 / Qwen 3.5
// whole-model verify kernels, and it exists because GPT-OSS was the one MoE
// architecture still prefilling layer by layer. That path cost, per layer:
//
//   fused attention graph -> post-attn RMSNorm -> router mul_mat -> DOWNLOAD the
//   router scores -> host top-k -> fill(0) -> fused MoE graph (rebuilt, with the
//   stacked expert biases re-uploaded every call) -> DOWNLOAD the MoE output ->
//   residual add
//
// i.e. ~6 dispatches and two host round trips per layer, 24 layers deep, twice
// over because the caller chunked anything past 256 tokens. Measured on an RTX
// PRO 6000 that was 512 tokens in 519 ms (987 tok/s) against llama.cpp's 30 ms.
// Everything in the list above is a graph node here instead: the routing top-k
// runs on the accelerator (ggml_top_k over the router logits, exactly the
// SOFTMAX_WEIGHT gate llama.cpp's build_moe_ffn uses for this architecture) and
// nothing crosses the bus between the hidden states going in and the logits
// coming out.
//
// Shapes follow the decode kernel with the token axis restored: hidden is
// [H, N], the RoPE positions are an I32[N], the KV write is one ggml_cpy per
// layer into a [start_pos, start_pos+N) row view, and the attention mask is F16
// [window, N] filled causally (plus the sliding-window floor on GPT-OSS's
// even layers). Only the LAST token's row reaches the MoE of the final layer
// and the LM head — the caller wants one logits vector — which is the same
// narrowing the per-layer path did.
//
// KV residency is shared with the other two GPT-OSS kernels through
// tsg_gptoss::kv_acquire, so prefill, per-layer prefill and decode all attach
// to one device copy per cache. Like decode, this kernel does NOT mirror the
// rows it writes back to host memory; the C# model marks its cache host-dirty
// and any host reader syncs through TSGgml_GptOssSyncKvCacheToHost first.
// ============================================================================
namespace
{
    constexpr const char* kGptOssPrefillKernel = "GPT-OSS model prefill";

    // Causal mask over [0, window) keys for the N queries at absolute positions
    // [start_pos, start_pos+N). A sliding-window layer additionally floors each
    // query at position-sliding_window+1, which is what makes GPT-OSS's even
    // layers cheap.
    void gptoss_fill_prefill_mask(std::vector<ggml_fp16_t>& mask, int wstart, int wlen,
                                  int start_pos, int n, bool swa, int sliding_window)
    {
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
        mask.assign(static_cast<std::size_t>(wlen) * static_cast<std::size_t>(n), neg_inf);
        for (int qi = 0; qi < n; qi++)
        {
            const int position = start_pos + qi;
            const int lo = swa ? std::max(0, position - sliding_window + 1) : 0;
            ggml_fp16_t* row = &mask[static_cast<std::size_t>(qi) * static_cast<std::size_t>(wlen)];
            for (int ki = 0; ki < wlen; ki++)
            {
                const int abs_row = wstart + ki;
                if (abs_row >= lo && abs_row <= position)
                    row[ki] = zero_val;
            }
        }
    }
}

TSG_EXPORT int TSGgml_GptOssModelPrefill(
    const TSGgmlGptOssLayerDesc* layers, int num_layers,
    // [num_tokens, hidden_size] F32 embeddings in. Not written back: the caller
    // only wants the logits, and the hidden states never leave the device.
    const void* hidden_data, int hidden_size, int num_tokens, int start_pos,
    // Folded final-norm + lm_head over the LAST token only. All of logits_data /
    // lm_head_data / final_norm_data must be supplied with vocab_size > 0.
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data)
{
    tsg::PhaseTimer pt(kGptOssPrefillKernel);
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr || num_tokens <= 0)
        {
            set_last_error("GPT-OSS model prefill: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGptOssLayerDesc)))
        {
            set_last_error("GPT-OSS model prefill: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }
        if (logits_data == nullptr || lm_head_data == nullptr || final_norm_data == nullptr || vocab_size <= 0)
        {
            set_last_error("GPT-OSS model prefill: the folded LM head is required.");
            return 0;
        }

        const int H = hidden_size;
        const int N = num_tokens;
        const int totalSeqLen = start_pos + N;
        const int kvType = layers[0].kv_cache_type;
        if (kvType != GGML_TYPE_F32 && kvType != GGML_TYPE_F16)
        {
            set_last_error("GPT-OSS model prefill: only F32/F16 KV caches are supported.");
            return 0;
        }

        // ---- device KV windows (shared with decode and the per-layer kernel) ----
        std::unique_lock<std::mutex> kv_lock(tsg_gptoss::kv_mutex());
        std::vector<tsg_gptoss::KvWindow*> k_wins(num_layers, nullptr);
        std::vector<tsg_gptoss::KvWindow*> v_wins(num_layers, nullptr);
        for (int l = 0; l < num_layers; l++)
        {
            if (!tsg_gptoss::kv_acquire_pair(layers[l], totalSeqLen, k_wins[l], v_wins[l]))
            {
                set_last_error("GPT-OSS model prefill: failed to acquire a device KV window.");
                return 0;
            }
        }

        // Rows the device copy is missing below start_pos (a fresh window, a
        // rewind, or a jump into another sequence's prefix) come from the host
        // mirror — the same contract decode uses.
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            const std::int64_t kvalid = std::min<std::int64_t>(k_wins[l]->rows_valid, start_pos);
            const std::int64_t vvalid = std::min<std::int64_t>(v_wins[l]->rows_valid, start_pos);
            tsg_gptoss::kv_upload(k_wins[l], d.k_cache, d.cache_size, kvalid, start_pos);
            tsg_gptoss::kv_upload(v_wins[l], d.v_cache, d.cache_size, vvalid, start_pos);
        }

        // ---- per-layer attention window geometry ----
        // A query at position p reads keys [p - sliding_window + 1, p], so the
        // whole chunk needs [start_pos - sliding_window + 1, totalSeqLen).
        std::vector<int> wstart(num_layers, 0);
        std::vector<int> wlen(num_layers, 0);
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            const int capacity = static_cast<int>(std::min(k_wins[l]->capacity, v_wins[l]->capacity));
            const bool swa = d.is_swa != 0 && d.sliding_window > 0;
            wstart[l] = swa ? std::max(0, start_pos - d.sliding_window + 1) : 0;
            wlen[l] = std::min(capacity - wstart[l], totalSeqLen - wstart[l]);
            if (wlen[l] <= 0 || wstart[l] + wlen[l] > capacity)
            {
                set_last_error("GPT-OSS model prefill: invalid attention window geometry.");
                return 0;
            }
        }

        // ---- build ----
        // The shared context pool hands out 32 MiB blocks (ggml_pool), which is
        // ~20x what this graph's tensor metadata needs.
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("GPT-OSS model prefill: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        struct LayerTensors
        {
            ggml_tensor* attn_norm_w = nullptr;
            ggml_tensor* qkv_w = nullptr;
            ggml_tensor* qkv_b = nullptr;
            ggml_tensor* k_w = nullptr;
            ggml_tensor* k_b = nullptr;
            ggml_tensor* v_w = nullptr;
            ggml_tensor* v_b = nullptr;
            ggml_tensor* o_w = nullptr;
            ggml_tensor* o_b = nullptr;
            ggml_tensor* sinks = nullptr;
            ggml_tensor* k_cache = nullptr;
            ggml_tensor* v_cache = nullptr;
            ggml_tensor* post_attn_norm_w = nullptr;
            ggml_tensor* gate_inp_w = nullptr;
            ggml_tensor* gate_inp_b = nullptr;
            ggml_tensor* gate_exps = nullptr;
            ggml_tensor* gate_exps_b = nullptr;
            ggml_tensor* up_exps = nullptr;
            ggml_tensor* up_exps_b = nullptr;
            ggml_tensor* down_exps = nullptr;
            ggml_tensor* down_exps_b = nullptr;
            ggml_tensor* k_cpy = nullptr;
            ggml_tensor* v_cpy = nullptr;
            ggml_tensor* attn_mask = nullptr;   // shared with same-geometry layers
            int mask_slot = -1;                 // index into mask_store
        };
        std::vector<LayerTensors> lt(num_layers);
        std::vector<tsg::HostMoeSegment> host_moe;
        struct MaskKey
        {
            int wstart, wlen; bool swa; int sliding_window;
            bool operator==(const MaskKey& o) const
            { return wstart == o.wstart && wlen == o.wlen && swa == o.swa && sliding_window == o.sliding_window; }
        };
        struct MaskEntry { MaskKey key; ggml_tensor* tensor; int slot; };
        std::vector<MaskEntry> mask_cache;
        std::vector<std::vector<ggml_fp16_t>> mask_store;

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int qDim = d.num_heads * hd;
            const int kDim = kvH * hd;
            const int nExp = d.num_experts;

            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            const int qkvDim = (d.separate_qkv != 0) ? qDim : (qDim + 2 * kDim);
            if (d.qkv_b != nullptr) t.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkvDim);
            if (d.separate_qkv != 0)
            {
                t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                if (d.k_b != nullptr) t.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
                if (d.v_b != nullptr) t.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
            }
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            if (d.o_b != nullptr) t.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.sinks != nullptr) t.sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.num_heads);
            t.k_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, k_wins[l]->capacity, kvH);
            t.v_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, v_wins[l]->capacity, kvH);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            if (d.gate_inp_b != nullptr) t.gate_inp_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp);
            // MoE CPU offload: an offloaded layer's routed experts stay in system
            // RAM, so their tensors are never created and never bound. That
            // omission IS the VRAM saving.
            if (d.cpu_moe == 0 || host_moe_verify_enabled())
            {
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ge_type), d.ge_ne0, d.ge_ne1, nExp);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ue_type), d.ue_ne0, d.ue_ne1, nExp);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
                if (d.gate_exps_b != nullptr) t.gate_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ge_ne1, nExp);
                if (d.up_exps_b != nullptr) t.up_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ue_ne1, nExp);
                if (d.down_exps_b != nullptr) t.down_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.de_ne1, nExp);
            }
            // Masks are shared by geometry rather than built per layer. GPT-OSS
            // has exactly two: the sliding-window one on even layers and the full
            // causal one on odd layers. Building 24 of them cost, at an 8192-token
            // chunk, ~200M fp16 host writes and ~380 MB of uploads per forward for
            // masks that are bit-identical in pairs.
            {
                const MaskKey key{wstart[l], wlen[l], d.is_swa != 0, d.sliding_window};
                bool found = false;
                for (const auto& m : mask_cache)
                {
                    if (m.key == key) { t.attn_mask = m.tensor; t.mask_slot = m.slot; found = true; break; }
                }
                if (!found)
                {
                    t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, wlen[l], N, 1, 1);
                    t.mask_slot = static_cast<int>(mask_store.size());
                    mask_store.emplace_back();
                    gptoss_fill_prefill_mask(mask_store.back(), wstart[l], wlen[l], start_pos, N,
                                             d.is_swa != 0, d.sliding_window);
                    mask_cache.push_back({key, t.attn_mask, t.mask_slot});
                }
            }
        }

        ggml_tensor* hidden = hidden_t;      // [H, N]
        bool fa_unsupported = false;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int nH = d.num_heads;
            const int kvH = d.num_kv_heads;
            const int qDim = nH * hd;
            const int kDim = kvH * hd;
            const int nExp = d.num_experts;
            const int nUsed = d.num_experts_used;
            const std::int64_t nFf = d.ge_ne1;
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
            const bool last_layer = (l == num_layers - 1);

            // ===== Attention =====
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, d.eps), t.attn_norm_w); // [H, N]

            // RoPE and the KV permute both take strided sources, so the fused
            // QKV is sliced with view_3d and never copied out — the same shape
            // llama.cpp's openai-moe-iswa graph uses.
            ggml_tensor* q_3d;
            ggml_tensor* k_3d;
            ggml_tensor* v_3d;
            if (d.separate_qkv != 0)
            {
                ggml_tensor* q_proj = ggml_mul_mat(ctx, t.qkv_w, normed);
                if (t.qkv_b != nullptr) q_proj = ggml_add(ctx, q_proj, t.qkv_b);
                ggml_tensor* k_proj = ggml_mul_mat(ctx, t.k_w, normed);
                if (t.k_b != nullptr) k_proj = ggml_add(ctx, k_proj, t.k_b);
                ggml_tensor* v_proj = ggml_mul_mat(ctx, t.v_w, normed);
                if (t.v_b != nullptr) v_proj = ggml_add(ctx, v_proj, t.v_b);
                q_3d = ggml_reshape_3d(ctx, q_proj, hd, nH, N);
                k_3d = ggml_reshape_3d(ctx, k_proj, hd, kvH, N);
                v_3d = ggml_reshape_3d(ctx, v_proj, hd, kvH, N);
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);          // [qDim+2*kDim, N]
                if (t.qkv_b != nullptr) qkv = ggml_add(ctx, qkv, t.qkv_b);
                const std::size_t hd_bytes = static_cast<std::size_t>(hd) * sizeof(float);
                q_3d = ggml_view_3d(ctx, qkv, hd, nH, N, hd_bytes, qkv->nb[1], 0);
                k_3d = ggml_view_3d(ctx, qkv, hd, kvH, N, hd_bytes, qkv->nb[1],
                                    static_cast<std::size_t>(qDim) * sizeof(float));
                v_3d = ggml_view_3d(ctx, qkv, hd, kvH, N, hd_bytes, qkv->nb[1],
                                    static_cast<std::size_t>(qDim + kDim) * sizeof(float));
            }

            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
                d.rope_n_dims, /*mode=*/2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
                /*ext_factor=*/1.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
                d.rope_n_dims, 2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
                1.0f, 1.0f, 32.0f, 1.0f);

            // Cache layout is [head_dim, rows, kv_heads]; permute the token axis
            // into the row slot and write the chunk's N rows in one copy.
            ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));  // [hd, N, kvH]
            ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_3d, 0, 2, 1, 3));

            const std::size_t kv_offset = static_cast<std::size_t>(start_pos) * t.k_cache->nb[1];
            ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cache, hd, N, kvH,
                t.k_cache->nb[1], t.k_cache->nb[2], kv_offset);
            ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cache, hd, N, kvH,
                t.v_cache->nb[1], t.v_cache->nb[2], kv_offset);
            t.k_cpy = ggml_cpy(ctx, k_write, k_dst);
            t.v_cpy = ggml_cpy(ctx, v_write, v_dst);

            ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache, hd,
                static_cast<int>(k_wins[l]->capacity), kvH, wstart[l], wlen[l], kvType);
            ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache, hd,
                static_cast<int>(v_wins[l]->capacity), kvH, wstart[l], wlen[l], kvType);
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("GPT-OSS model prefill: failed to build KV cache views.");
                return 0;
            }

            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);   // [hd, N, nH]
            ggml_tensor* attn_flat = nullptr;
            ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, t.attn_mask, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            if (t.sinks != nullptr)
                ggml_flash_attn_ext_add_sinks(fa, t.sinks);
            if (l == 0)
                fa_unsupported = !backend_supports_op(fa);
            if (!fa_unsupported)
            {
                attn_flat = ggml_reshape_2d(ctx, fa, qDim, N);
            }
            else
            {
                ggml_tensor* q_cont = ggml_cont(ctx, q_attn);
                ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q_cont);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, t.attn_mask, scale, 0.0f);
                if (t.sinks != nullptr)
                    ggml_soft_max_add_sinks(probs, t.sinks);
                ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
                ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
                ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
                attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, N);
            }

            ggml_tensor* o_mm = ggml_mul_mat(ctx, t.o_w, attn_flat);       // [H, N]
            if (t.o_b != nullptr) o_mm = ggml_add(ctx, o_mm, t.o_b);
            ggml_tensor* ffn_inp = ggml_add(ctx, hidden, o_mm);            // [H, N]

            // ===== MoE FFN =====
            // Only the last token's logits are wanted, so the final layer's MoE
            // (the most expensive single block in the graph) runs on one row.
            // Same narrowing the per-layer path did.
            int M = N;
            ggml_tensor* moe_src = ffn_inp;
            if (last_layer && N > 1)
            {
                moe_src = ggml_cont(ctx, ggml_view_2d(ctx, ffn_inp, H, 1, ffn_inp->nb[1],
                    static_cast<std::size_t>(N - 1) * ffn_inp->nb[1]));
                M = 1;
            }

            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, moe_src, d.eps), t.post_attn_norm_w); // [H, M]

            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, moe_in);   // [nExp, M]
            if (t.gate_inp_b != nullptr)
                router_logits = ggml_add(ctx, router_logits, t.gate_inp_b);
            // GPT-OSS gates with SOFTMAX_WEIGHT: top-k over the RAW logits, then a
            // softmax over just the selected ones (no renormalisation afterwards).
            ggml_tensor* sel = ggml_top_k(ctx, router_logits, nUsed);               // [nUsed, M] I32
            ggml_tensor* logits_r = ggml_reshape_3d(ctx, router_logits, 1, nExp, M);
            ggml_tensor* w = ggml_get_rows(ctx, logits_r, sel);                     // [1, nUsed, M]
            ggml_tensor* w_soft = ggml_soft_max(ctx, ggml_reshape_2d(ctx, w, nUsed, M));
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_soft, 1, nUsed, M);

            ggml_tensor* moe_out_2d = nullptr;
            if (d.cpu_moe != 0)
            {
                // ---- MoE CPU offload seam (see tsg::HostMoeSegment) ----
                // Prefill hands the host all M tokens at once, so the offloaded
                // side is a real GEMM over the chunk rather than M matvecs.
                // Attention, the router and the LM head stay in this graph.
                tsg::HostMoeSegment hm;
                hm.layer = l;
                hm.moe_in = ggml_cont(ctx, moe_in);
                hm.sel_ids = ggml_cont(ctx, ggml_reshape_1d(ctx, sel, static_cast<std::int64_t>(nUsed) * M));
                hm.weights = ggml_cont(ctx, ggml_reshape_1d(ctx, w_final, static_cast<std::int64_t>(nUsed) * M));
                ggml_set_output(hm.moe_in);
                ggml_set_output(hm.sel_ids);
                ggml_set_output(hm.weights);

                // Written by the host between segments; flagged BOTH input and
                // output so ggml-alloc pre-allocates it and never recycles the
                // block behind our back.
                ggml_tensor* moe_host_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, M);
                ggml_set_input(moe_host_out);
                ggml_set_output(moe_host_out);

                hm.moe_out = moe_host_out;
                hm.gate_data = d.gate_exps; hm.gate_type = d.ge_type;
                hm.gate_ne0 = d.ge_ne0;     hm.gate_ne1 = d.ge_ne1;  hm.gate_bytes = d.ge_bytes;
                hm.up_data = d.up_exps;     hm.up_type = d.ue_type;
                hm.up_ne0 = d.ue_ne0;       hm.up_ne1 = d.ue_ne1;    hm.up_bytes = d.ue_bytes;
                hm.down_data = d.down_exps; hm.down_type = d.de_type;
                hm.down_ne0 = d.de_ne0;     hm.down_ne1 = d.de_ne1;  hm.down_bytes = d.de_bytes;
                hm.gate_bias = static_cast<const float*>(d.gate_exps_b);
                hm.up_bias = static_cast<const float*>(d.up_exps_b);
                hm.down_bias = static_cast<const float*>(d.down_exps_b);
                hm.activation = 1;          // gpt-oss clamped SwiGLU
                hm.oai_alpha = d.oai_alpha;
                hm.oai_limit = d.oai_limit;
                hm.num_experts = nExp;
                hm.n_used = nUsed;
                hm.n_ff = nFf;
                hm.seq_len = M;
                hm.hidden = H;

                if (host_moe_verify_enabled())
                {
                    ggml_tensor* vin = ggml_reshape_3d(ctx, moe_in, H, 1, M);
                    ggml_tensor* vg = ggml_mul_mat_id(ctx, t.gate_exps, vin, sel);
                    if (t.gate_exps_b != nullptr) vg = ggml_add_id(ctx, vg, t.gate_exps_b, sel);
                    ggml_tensor* vu = ggml_mul_mat_id(ctx, t.up_exps, vin, sel);
                    if (t.up_exps_b != nullptr) vu = ggml_add_id(ctx, vu, t.up_exps_b, sel);
                    ggml_tensor* vd = ggml_mul_mat_id(ctx, t.down_exps,
                        ggml_swiglu_oai(ctx, vg, vu, d.oai_alpha, d.oai_limit), sel);
                    if (t.down_exps_b != nullptr) vd = ggml_add_id(ctx, vd, t.down_exps_b, sel);
                    ggml_tensor* vw = ggml_mul(ctx, vd, w_final);
                    ggml_tensor* vsum = ggml_view_2d(ctx, vw, H, M, vw->nb[2], 0);
                    for (int u = 1; u < nUsed; ++u)
                    {
                        ggml_tensor* vv = ggml_view_2d(ctx, vw, H, M, vw->nb[2],
                            static_cast<std::size_t>(u) * vw->nb[1]);
                        vsum = ggml_add(ctx, vsum, vv);
                    }
                    hm.verify_gpu = ggml_cont(ctx, vsum);
                    ggml_set_output(hm.verify_gpu);
                }

                host_moe.push_back(hm);
                moe_out_2d = moe_host_out;
            }
            else
            {
                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, M);
                ggml_tensor* gate = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);   // [nFf, nUsed, M]
                if (t.gate_exps_b != nullptr) gate = ggml_add_id(ctx, gate, t.gate_exps_b, sel);
                ggml_tensor* up = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                if (t.up_exps_b != nullptr) up = ggml_add_id(ctx, up, t.up_exps_b, sel);
                ggml_tensor* act = ggml_swiglu_oai(ctx, gate, up, d.oai_alpha, d.oai_limit);
                ggml_tensor* down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);         // [H, nUsed, M]
                if (t.down_exps_b != nullptr) down = ggml_add_id(ctx, down, t.down_exps_b, sel);
                ggml_tensor* weighted = ggml_mul(ctx, down, w_final);

                // Sum the nUsed expert results per token. weighted is
                // [H, nUsed, M], so each expert slice is a [H, M] view with the
                // token stride in nb[2] and the expert offset in nb[1].
                ggml_tensor* acc = ggml_view_2d(ctx, weighted, H, M, weighted->nb[2], 0);
                for (int u = 1; u < nUsed; ++u)
                {
                    ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, M, weighted->nb[2],
                        static_cast<std::size_t>(u) * weighted->nb[1]);
                    acc = ggml_add(ctx, acc, view_u);
                }
                moe_out_2d = acc;
            }
            hidden = ggml_add(ctx, moe_src, moe_out_2d);   // [H, M]
        }

        // ---- folded final norm + LM head over the last row ----
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* last_hidden = (hidden->ne[1] == 1)
            ? hidden
            : ggml_cont(ctx, ggml_view_2d(ctx, hidden, H, 1, hidden->nb[1],
                  static_cast<std::size_t>(hidden->ne[1] - 1) * hidden->nb[1]));
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, last_hidden, layers[0].eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);                 // [vocab, 1]
        ggml_tensor* logits_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
        ggml_tensor* out_cpy = ggml_cpy(ctx, ggml_reshape_1d(ctx, logits, vocab_size), logits_out);
        ggml_set_output(out_cpy);

        pt.mark("build");
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 192 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // KV writes first so they are ordered before the reads, and each
        // offloaded layer's seam boundary right after its own layer, so the cut
        // lands exactly where the accelerator has to pause for the host.
        std::size_t next_host_moe = 0;
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, lt[l].k_cpy);
            ggml_build_forward_expand(graph, lt[l].v_cpy);
            if (next_host_moe < host_moe.size() && host_moe[next_host_moe].layer == l)
            {
                const tsg::HostMoeSegment& hm = host_moe[next_host_moe];
                ggml_build_forward_expand(graph, hm.moe_in);
                ggml_build_forward_expand(graph, hm.sel_ids);
                ggml_build_forward_expand(graph, hm.weights);
                if (hm.verify_gpu != nullptr)
                    ggml_build_forward_expand(graph, hm.verify_gpu);
                ++next_host_moe;
            }
        }
        ggml_build_forward_expand(graph, out_cpy);

        std::vector<int> host_moe_seg_end;
        if (!host_moe_build_segment_ends(graph, host_moe, host_moe_seg_end, kGptOssPrefillKernel))
            return 0;

        // ---- bind ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; const void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* tgt, const void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            void* raw = const_cast<void*>(data);
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, raw, bytes, needs_upload, usage))
                {
                    if (needs_upload) upload_list.push_back({tgt, data, bytes});
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, raw, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tgt, raw) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        bool bind_failed = false;
        for (int l = 0; l < num_layers && !bind_failed; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int nExp = d.num_experts;
            const int qDim = d.num_heads * hd;
            const int kDim = d.num_kv_heads * hd;
            const int qkvDim = (d.separate_qkv != 0) ? qDim : (qDim + 2 * kDim);

            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
            bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
            bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.ge_bytes), true);
            bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.ue_bytes), true);
            bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);
            bind_or_mark(t.gate_inp_b, d.gate_inp_b, static_cast<std::size_t>(nExp) * sizeof(float), true);
            bind_or_mark(t.qkv_b, d.qkv_b, static_cast<std::size_t>(qkvDim) * sizeof(float), true);
            bind_or_mark(t.k_b, d.k_b, static_cast<std::size_t>(kDim) * sizeof(float), true);
            bind_or_mark(t.v_b, d.v_b, static_cast<std::size_t>(kDim) * sizeof(float), true);
            bind_or_mark(t.o_b, d.o_b, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.sinks, d.sinks, static_cast<std::size_t>(d.num_heads) * sizeof(float), true);
            bind_or_mark(t.gate_exps_b, d.gate_exps_b, static_cast<std::size_t>(d.ge_ne1) * nExp * sizeof(float), true);
            bind_or_mark(t.up_exps_b, d.up_exps_b, static_cast<std::size_t>(d.ue_ne1) * nExp * sizeof(float), true);
            bind_or_mark(t.down_exps_b, d.down_exps_b, static_cast<std::size_t>(d.de_ne1) * nExp * sizeof(float), true);

            // The K/V caches are the shared device windows: point this graph's
            // tensors straight at them (no host binding, no upload).
            if (ggml_backend_tensor_alloc(k_wins[l]->buffer, t.k_cache, k_wins[l]->tensor->data) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(v_wins[l]->buffer, t.v_cache, v_wins[l]->tensor->data) != GGML_STATUS_SUCCESS)
            {
                bind_failed = true;
            }
        }
        if (bind_failed)
        {
            set_last_error("GPT-OSS model prefill: failed to bind a KV window.");
            return 0;
        }
        bind_or_mark(lm_head_t, lm_head_data, static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, final_norm_data, static_cast<std::size_t>(H) * sizeof(float), true);
        // One bind per DISTINCT mask, not per layer.
        for (const auto& m : mask_cache)
            bind_or_mark(m.tensor, mask_store[static_cast<std::size_t>(m.slot)].data(),
                         mask_store[static_cast<std::size_t>(m.slot)].size() * sizeof(ggml_fp16_t), false);

        pt.mark("bind");
        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("GPT-OSS model prefill: failed to allocate backend buffer.");
                return 0;
            }
        }

        pt.mark("alloc");
        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0,
                                static_cast<std::size_t>(H) * static_cast<std::size_t>(N) * sizeof(float));
        std::vector<std::int32_t> pos_vals(static_cast<std::size_t>(N));
        for (int i = 0; i < N; i++)
            pos_vals[static_cast<std::size_t>(i)] = start_pos + i;
        ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, pos_vals.size() * sizeof(std::int32_t));

        ggml_status status = GGML_STATUS_SUCCESS;
        if (!host_moe.empty())
        {
            if (!host_moe_execute_segments(graph, host_moe, host_moe_seg_end, kGptOssPrefillKernel))
                status = GGML_STATUS_FAILED;
        }
        else
        {
            status = tsg::graph_compute_profiled(g_backend, graph, kGptOssPrefillKernel);
        }
        if (status != GGML_STATUS_SUCCESS)
        {
            if (host_moe.empty())
                set_last_error("GPT-OSS model prefill: graph execution failed.");
            return 0;
        }

        pt.mark("compute");
        finalize_compute_with_download(logits_out, logits_data,
                                       static_cast<std::size_t>(vocab_size) * sizeof(float));
        host_read_barrier();
        pt.mark("download");

        for (int l = 0; l < num_layers; l++)
        {
            k_wins[l]->rows_valid = std::max<std::int64_t>(k_wins[l]->rows_valid, totalSeqLen);
            v_wins[l]->rows_valid = std::max<std::int64_t>(v_wins[l]->rows_valid, totalSeqLen);
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in GPT-OSS model prefill.");
        return 0;
    }
}
