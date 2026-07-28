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
#include <chrono>
#include <cstdio>

using namespace tsg;

// PERSIST / CUDA-graph capture pool for the token-batched dense decode
// (TSGgml_Gemma4ModelDecodeBatched). Analogue of the single-stream g_g4dc_pool,
// but each entry's identity is the SET of N per-request KV caches (sig_kc, in the
// C#-canonicalised request-id order) + N + per-layer padded window. A recurring
// concurrent request-set replays its captured graph (set_rows KV write + fixed
// padded window + F16 mask inputs => identical topology + stable addresses) so
// ggml-cuda's CUDA-graph capture engages. Dropped on prefill / KV reset
// (TSGgml_Gemma4ResetBatchedDecodeCache) since those move device addresses.
namespace
{
    constexpr int kG4BatchPersistKvStride = 256;

    struct G4BatchedDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_gallocr_t galloc = nullptr;      // dedicated packed allocator (MoE batched: VRAM-frugal capture)
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* logits_out = nullptr;
        std::vector<ggml_tensor*> kv_index;   // [num_layers * n_seqs] I64 set_rows write rows
        std::vector<ggml_tensor*> attn_mask;  // [num_layers] F16 [win,1,1,n_seqs]
        std::vector<int> layer_window;        // [num_layers]
        const void* sig_disc = nullptr;
        std::vector<const void*> sig_kc;      // [n_seqs] layer-0 K cache ptrs (canonical order)
        int num_layers = 0, hidden_size = 0, n_seqs = 0, vocab = 0;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = pos_tensor = logits_out = nullptr;
            kv_index.clear(); attn_mask.clear(); layer_window.clear();
            sig_disc = nullptr; sig_kc.clear();
            num_layers = hidden_size = n_seqs = vocab = 0;
        }
    };

    constexpr int kG4BatchedMaxCaches = 4;
    struct G4BatchedDecodeCachePool
    {
        G4BatchedDecodeCache entries[kG4BatchedMaxCaches];
        std::uint64_t used[kG4BatchedMaxCaches] = {};
        std::uint64_t clock = 0;

        G4BatchedDecodeCache* find(const void* sig, const std::vector<const void*>& kc, int n)
        {
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].n_seqs == n && entries[i].sig_kc == kc)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        G4BatchedDecodeCache& claim(const void* sig, const std::vector<const void*>& kc, int n)
        {
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].n_seqs == n && entries[i].sig_kc == kc)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kG4BatchedMaxCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    G4BatchedDecodeCachePool g_g4batched_pool;
}

// ============================================================================
// TRUE TOKEN-BATCHED dense decode (N concurrent sequences, one token each, in
// ONE ggml graph + ONE compute buffer). This is the llama-parity concurrency
// path: where TSGgml_Gemma4ModelDecode (ggml_ops_gemma4_decode.cpp) decodes ONE token and the engine
// round-robins N serial calls for N concurrent requests (N weight loads ->
// aggregate ~= single-stream), this kernel processes all N decode tokens
// together so every weight is loaded ONCE and applied to N tokens. Decode is
// memory-bandwidth bound, so that amortisation is the win (and one compute
// buffer instead of N fixes the per-request-buffer VRAM blowup).
//
// Each sequence owns its OWN KV cache (the per-request holders the C# engine
// manages); k_cache_arr / v_cache_arr are sized num_layers * n_seqs, indexed
// [layer * n_seqs + seq]. positions[seq] is each sequence's current length.
// Hidden in/out and logits are packed column-major [.., n_seqs].
//
// v1 scope (correctness-first): DENSE only (no MoE), no PLE (ple_dim==0), no
// KV-donor/shared layers, requires the folded lm_head, and the NO-WRAP regime
// (every sequence's total length <= every layer's cache size, so each query
// attends [0, total) with a simple per-sequence padding mask). The C# caller
// (Gemma4Model.TryForwardBatchedFusedDecode) enforces these and otherwise falls
// back to the round-robin per-sequence path. Non-persist (no CUDA-graph capture)
// in v1; capture is a follow-up once correctness is proven.
//
// Attention is one ggml_flash_attn_ext with batch dim ne3 = n_seqs: Q is
// reshaped to [head_dim, 1, num_heads, n_seqs]; each sequence's KV window is
// cont()'d and concat()'d along ne3 into [head_dim, win, kv_heads, n_seqs]; the
// mask is [win, 1, 1, n_seqs] with column s zeroing [0, total_s) and -inf'ing
// the padding. Everything else (norms, projections, FFN, lm_head) operates on
// [hidden, n_seqs] for free.
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4ModelDecodeBatched(
    float* hidden_data, int hidden_size, int num_layers, int n_seqs,
    void** attn_norm_arr,
    void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    // Per-(layer,seq) KV caches: k_cache_arr[layer * n_seqs + seq].
    void** k_cache_arr, void** v_cache_arr,
    int* head_dim_arr,
    int* kv_heads_arr,
    int* cache_size_arr,
    int* is_local_arr,
    float* rope_base_arr,
    float* layer_scalar_arr,
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    int num_heads,
    const int* positions,           // [n_seqs]
    float eps, int sliding_window,
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    int kv_cache_type,
    // Separate K/V projection weights for mixed-quant layers (null => fused qkv).
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    // Folded final-norm + lm_head (required for this kernel).
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (hidden_data == nullptr || n_seqs <= 0 || num_layers <= 0)
        {
            set_last_error("Gemma4 batched decode: invalid arguments.");
            return 0;
        }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold)
        {
            set_last_error("Gemma4 batched decode: folded lm_head required.");
            return 0;
        }

        struct LayerInfo { int hd; int kvHeads; int qDim; int kDim; int cacheSize; bool isLocal; int win; };
        std::vector<LayerInfo> li(num_layers);

        int maxTotal = 0;
        for (int s = 0; s < n_seqs; s++)
            maxTotal = std::max(maxTotal, positions[s] + 1);

        // Persist (CUDA-graph capture) gate: identical to the single-stream
        // g_g4dc. Capturable when on CUDA and TS_GEMMA4_FD_PERSIST != 0.
        static const bool g4b_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        bool can_persist = g4b_persist && g_backend_type == BACKEND_TYPE_CUDA;

        auto roundup_stride = [](int v){ return ((v + kG4BatchPersistKvStride - 1) / kG4BatchPersistKvStride) * kG4BatchPersistKvStride; };
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            info.cacheSize = cache_size_arr[l];
            info.isLocal = is_local_arr[l] != 0;
            // v1 NO-WRAP gate: every sequence must fit the cache window so each
            // query attends a contiguous [0, total) prefix.
            if (info.cacheSize <= 0 || maxTotal > info.cacheSize)
            {
                set_last_error("Gemma4 batched decode: sequence exceeds cache window (wrap not supported in v1).");
                return 0;
            }
            // Persist: pad the window to a 256-stride (bounded by the cache) so the
            // graph topology is identical token-to-token (only changes every 256
            // tokens), which is what lets ggml-cuda capture engage. Non-persist
            // uses the tight flash_attn length.
            info.win = can_persist
                ? std::min(info.cacheSize, std::max(roundup_stride(maxTotal), flash_attn_kv_length(maxTotal, info.cacheSize, info.hd)))
                : flash_attn_kv_length(maxTotal, info.cacheSize, info.hd);
        }

        // Per-seq attention-mask fill: column s zeroes [0, pos_s+1) (clamped to
        // win), -inf elsewhere. Shared by the reuse fast-path and the build path.
        auto fill_batched_mask = [&](std::vector<ggml_fp16_t>& md, int win)
        {
            md.assign(static_cast<std::size_t>(win) * n_seqs, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, win);
                for (int k = 0; k < valid; k++)
                    md[static_cast<std::size_t>(s) * win + k] = static_cast<ggml_fp16_t>(0);
            }
        };

        // Pool identity: model instance + the canonical-order set of N layer-0 K
        // caches + N + per-layer window. (C# canonicalises the request order.)
        const void* sig_disc = attn_norm_arr[0];
        std::vector<const void*> sig_kc(n_seqs);
        for (int s = 0; s < n_seqs; s++) sig_kc[s] = k_cache_arr[s];   // layer 0, seq s
        std::vector<int> winvec(num_layers);
        for (int l = 0; l < num_layers; l++) winvec[l] = li[l].win;

        // ---- reuse fast-path: replay this request-set's captured graph ----
        G4BatchedDecodeCache* dc = can_persist ? g_g4batched_pool.find(sig_disc, sig_kc, n_seqs) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == hidden_size &&
            dc->vocab == vocab_size && dc->layer_window == winvec)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_in, hidden_data, 0, static_cast<std::size_t>(hidden_size) * n_seqs * sizeof(float));
            ggml_backend_tensor_set(dc->pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
            for (int l = 0; l < num_layers; l++)
            {
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = positions[s];
                    ggml_backend_tensor_set(dc->kv_index[l * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
                std::vector<ggml_fp16_t> md;
                fill_batched_mask(md, li[l].win);
                ggml_backend_tensor_set(dc->attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
            }
            if (ggml_backend_graph_compute(g_backend, dc->graph) != GGML_STATUS_SUCCESS)
            {
                set_last_error("Gemma4 batched decode: replay graph compute failed.");
                dc->reset();
                return 0;
            }
            finalize_compute_with_download(dc->logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        // ---- build context: persist = raw no_alloc ctx kept alive in the pool
        // (stable addresses for capture); non-persist = pooled (<=32MB). ----
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 batched decode: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Gemma4 batched decode: failed to create ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_seqs);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        if (can_persist) { ggml_set_input(current); ggml_set_input(pos_tensor); }

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // Per-(layer,seq) I64 set_rows write-row inputs (persist only); kept for
        // the pool entry so the reuse path can refresh them each replay.
        std::vector<ggml_tensor*> kv_index_all(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* post_ffn_norm_w;
            std::vector<ggml_tensor*> k_cached;   // per seq
            std::vector<ggml_tensor*> v_cached;   // per seq
            std::vector<ggml_tensor*> k_cpy;      // per seq (KV write op)
            std::vector<ggml_tensor*> v_cpy;
            ggml_tensor* attn_mask;
            std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type_arr[l]), qkv_ne0_arr[l], qkv_ne1_arr[l]);
            const bool separate_qkv = (k_arr != nullptr && k_arr[l] != nullptr);
            if (separate_qkv)
            {
                lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
                lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            }
            else { lt.k_w = nullptr; lt.v_w = nullptr; }
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]), gu_ne0_arr[l], gu_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            lt.k_cached.resize(n_seqs);
            lt.v_cached.resize(n_seqs);
            lt.k_cpy.resize(n_seqs, nullptr);
            lt.v_cpy.resize(n_seqs, nullptr);
            for (int s = 0; s < n_seqs; s++)
            {
                lt.k_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                lt.v_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
            }
            lt.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, info.win, 1, 1, n_seqs);
            if (can_persist) ggml_set_input(lt.attn_mask);
        }

        // lm_head + final norm
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // ---- build graph ----
        ggml_tensor* hidden = current;   // [hidden_size, n_seqs]
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];
            int rope_dims = rope_n_dims_arr[l];
            ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;

            // 1. attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);   // [H, N]

            // 2. QKV projection -> [qDim, N] / [kDim, N]
            ggml_tensor* q_raw; ggml_tensor* k_raw; ggml_tensor* v_raw;
            if (lt.k_w != nullptr)
            {
                q_raw = ggml_mul_mat(ctx, lt.qkv_w, normed);   // [qDim, N]
                k_raw = ggml_mul_mat(ctx, lt.k_w, normed);     // [kDim, N]
                v_raw = ggml_mul_mat(ctx, lt.v_w, normed);     // [kDim, N]
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, lt.qkv_w, normed);   // [qDim+2kDim, N]
                q_raw = ggml_view_2d(ctx, qkv, info.qDim, n_seqs, qkv->nb[1], 0);
                k_raw = ggml_view_2d(ctx, qkv, info.kDim, n_seqs, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim) * sizeof(float));
                v_raw = ggml_view_2d(ctx, qkv, info.kDim, n_seqs, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));
            }

            // 3. per-head Q/K norm + V norm -> [hd, heads, N]
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), info.hd, num_heads, n_seqs);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), info.hd, info.kvHeads, n_seqs);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), info.hd, info.kvHeads, n_seqs);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), lt.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), lt.k_norm_w);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_3d, eps);

            // 4. RoPE (pos[seq] applied per ne2 slice)
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, pos_tensor, rope_ff,
                rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);   // [hd, num_heads, N]
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_normed, pos_tensor, rope_ff,
                rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);   // [hd, kvHeads, N]

            // 5. per-seq KV write + windowed read; concat windows along ne3
            ggml_tensor* k_cat = nullptr;
            ggml_tensor* v_cat = nullptr;
            for (int s = 0; s < n_seqs; s++)
            {
                const int cachePos = positions[s];   // no-wrap: index == position
                // slice this seq's new K/V: [hd, kvHeads, 1] -> permute -> [hd, 1, kvHeads]
                ggml_tensor* k_s = ggml_view_3d(ctx, k_rope, info.hd, info.kvHeads, 1,
                    k_rope->nb[1], k_rope->nb[2], static_cast<std::size_t>(s) * k_rope->nb[2]);
                ggml_tensor* v_s = ggml_view_3d(ctx, v_normed, info.hd, info.kvHeads, 1,
                    v_normed->nb[1], v_normed->nb[2], static_cast<std::size_t>(s) * v_normed->nb[2]);
                ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_s, 0, 2, 1, 3));   // [hd, 1, kvHeads]
                ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_s, 0, 2, 1, 3));   // [hd, 1, kvHeads]
                // k_src/v_src is the tensor the window read sees. For persist we
                // read from the set_rows RESULT (full-cache tensor), which gives a
                // real graph edge write->read so the topological sort can never
                // place the window read before this seq's KV write (without the
                // edge, ggml relied on insertion order, which reordered the last
                // sequence's read ahead of its write at N>=4 -> stale K/V).
                ggml_tensor* k_src; ggml_tensor* v_src;
                if (can_persist)
                {
                    // set_rows write (row = an I64 INPUT) keeps the graph topology
                    // identical token-to-token so CUDA-graph capture engages.
                    ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                    ggml_set_input(kv_idx);
                    kv_index_all[static_cast<std::size_t>(l) * n_seqs + s] = kv_idx;
                    lt.k_cpy[s] = ggml_set_rows(ctx, lt.k_cached[s], k_write, kv_idx);
                    lt.v_cpy[s] = ggml_set_rows(ctx, lt.v_cached[s], v_write, kv_idx);
                    k_src = lt.k_cpy[s];   // full-cache result of the write
                    v_src = lt.v_cpy[s];
                }
                else
                {
                    ggml_tensor* k_dst = ggml_view_3d(ctx, lt.k_cached[s], info.hd, 1, info.kvHeads,
                        lt.k_cached[s]->nb[1], lt.k_cached[s]->nb[2],
                        static_cast<std::size_t>(cachePos) * lt.k_cached[s]->nb[1]);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, lt.v_cached[s], info.hd, 1, info.kvHeads,
                        lt.v_cached[s]->nb[1], lt.v_cached[s]->nb[2],
                        static_cast<std::size_t>(cachePos) * lt.v_cached[s]->nb[1]);
                    lt.k_cpy[s] = ggml_cpy(ctx, k_write, k_dst);
                    lt.v_cpy[s] = ggml_cpy(ctx, v_write, v_dst);
                    k_src = lt.k_cached[s];
                    v_src = lt.v_cached[s];
                }

                // windowed read [0, win). Pass the STRIDED view straight to concat
                // (its non-contiguous kernel reads via strides) — the previous
                // ggml_cont here copied the whole padded window per seq/layer EVERY
                // step (only 1 KV row actually changed), a big redundant memcpy.
                // ggml_concat already produces a contiguous output for flash.
                ggml_tensor* k_win = view_kv_cache_window(ctx, k_src, info.hd, info.cacheSize, info.kvHeads, 0, info.win, kv_cache_type);
                ggml_tensor* v_win = view_kv_cache_window(ctx, v_src, info.hd, info.cacheSize, info.kvHeads, 0, info.win, kv_cache_type);
                k_cat = (k_cat == nullptr) ? k_win : ggml_concat(ctx, k_cat, k_win, 3);
                v_cat = (v_cat == nullptr) ? v_win : ggml_concat(ctx, v_cat, v_win, 3);
            }

            // 6. flash attention, batched over ne3 = n_seqs
            ggml_tensor* q4d = ggml_reshape_4d(ctx, ggml_cont(ctx, q_rope), info.hd, 1, num_heads, n_seqs);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q4d, k_cat, v_cat, lt.attn_mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);

            // 7. O projection -> [H, N]
            ggml_tensor* attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), info.qDim, n_seqs);
            ggml_tensor* o_out = ggml_mul_mat(ctx, lt.o_w, attn_2d);   // [H, N]

            // 8. post-attn norm + residual
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn);

            // 9. FFN
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            std::int64_t inter = gu_ne1_arr[l] / 2;
            ggml_tensor* gu = ggml_mul_mat(ctx, lt.gu_w, ffn_normed);   // [2*inter, N]
            ggml_tensor* gate = ggml_view_2d(ctx, gu, inter, n_seqs, gu->nb[1], 0);
            ggml_tensor* up = ggml_view_2d(ctx, gu, inter, n_seqs, gu->nb[1],
                static_cast<std::size_t>(inter) * sizeof(float));
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_gelu(ctx, ggml_cont(ctx, gate)), ggml_cont(ctx, up));   // [inter, N]
            ggml_tensor* down_out = ggml_mul_mat(ctx, lt.down_w, ffn_hidden);   // [H, N]

            // 10. post-FFN norm + residual
            ggml_tensor* post_ffn = ggml_mul(ctx, ggml_rms_norm(ctx, down_out, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, residual1, post_ffn);

            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;
        }

        // final norm + lm_head -> logits [vocab, N]
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);   // [vocab, N]
        if (logit_softcap > 0.0f)
        {
            logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, logit_softcap);
        }
        ggml_tensor* logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_seqs);
        ggml_tensor* out_op = ggml_cpy(ctx, logits, logits_out);
        ggml_set_output(out_op);

        // build graph: KV writes first, then output
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (256 + 16 * n_seqs) + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
            for (int s = 0; s < n_seqs; s++)
            {
                if (layers[l].k_cpy[s] != nullptr) ggml_build_forward_expand(graph, layers[l].k_cpy[s]);
                if (layers[l].v_cpy[s] != nullptr) ggml_build_forward_expand(graph, layers[l].v_cpy[s]);
            }
        ggml_build_forward_expand(graph, out_op);

        // ---- bind weights ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({t, data, bytes});
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            bind_or_mark(lt.qkv_w, qkv_arr[l], static_cast<std::size_t>(qkv_bytes_arr[l]), true);
            if (lt.k_w != nullptr)
            {
                bind_or_mark(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
                bind_or_mark(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            }
            bind_or_mark(lt.o_w, o_arr[l], static_cast<std::size_t>(o_bytes_arr[l]), true);
            bind_or_mark(lt.gu_w, gu_arr[l], static_cast<std::size_t>(gu_bytes_arr[l]), true);
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);
            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w, ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_ffn_norm_w, post_ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w, q_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            for (int s = 0; s < n_seqs; s++)
            {
                bind_or_mark(lt.k_cached[s], k_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(lt.v_cached[s], v_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            // per-seq attention mask (host scratch, not cacheable)
            lt.attn_mask_data.assign(static_cast<std::size_t>(info.win) * n_seqs,
                ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, info.win);
                for (int k = 0; k < valid; k++)
                    lt.attn_mask_data[static_cast<std::size_t>(s) * info.win + k] = static_cast<ggml_fp16_t>(0);
            }
            bind_or_mark(lt.attn_mask, lt.attn_mask_data.data(), lt.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(hidden_size) * sizeof(float), true);

        // Persist: every tensor gets its own slot (stable addresses for capture),
        // kept alive in the pool. Non-persist: reuse the pooled compute buffer.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Gemma4 batched decode: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Gemma4 batched decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * n_seqs * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0, static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));
        if (can_persist)
        {
            for (int l = 0; l < num_layers; l++)
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = positions[s];
                    ggml_backend_tensor_set(kv_index_all[static_cast<std::size_t>(l) * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Gemma4 batched decode: graph compute failed.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        finalize_compute_with_download(logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));

        if (can_persist)
        {
            host_read_barrier();
            G4BatchedDecodeCache& e = g_g4batched_pool.claim(sig_disc, sig_kc, n_seqs);
            e.ctx = ctx; e.buffer = persist_buf; e.graph = graph;
            e.hidden_in = current; e.pos_tensor = pos_tensor; e.logits_out = logits_out;
            e.kv_index = kv_index_all;
            e.attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) e.attn_mask[l] = layers[l].attn_mask;
            e.layer_window = winvec;
            e.sig_disc = sig_disc; e.sig_kc = sig_kc;
            e.num_layers = num_layers; e.hidden_size = hidden_size; e.n_seqs = n_seqs; e.vocab = vocab_size;
            e.valid = true;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Gemma4 batched decode."); return 0; }
}

// Drop all captured token-batched decode graphs. The captured graphs pin
// ggml-cuda's compute-pool scratch + the per-request KV buffers; a prefill (grows
// the pool) or a KV reset/grow can move those, so the C# caller drops the cache
// before any prefill and on ResetKVCache. No-op when persist is off.
TSG_EXPORT void TSGgml_Gemma4ResetBatchedDecodeCache()
{
    g_g4batched_pool.reset_all();
}

namespace { G4BatchedDecodeCachePool g_g4moebatched_pool; }

// ============================================================================
// TRUE TOKEN-BATCHED MoE decode (N concurrent sequences, one token each, one
// ggml graph + one compute buffer). The MoE sibling of
// TSGgml_Gemma4ModelDecodeBatched: identical batched attention (per-request KV,
// flash_attn_ext over ne3=N, set_rows write + fixed padded window for capture)
// but the FFN is the Gemma-4 MoE block (dense shared FFN + in-graph router +
// stacked experts via ggml_mul_mat_id over N tokens — token-parallel, copied
// from the verify kernel). Weights come from the existing per-layer
// TSGgmlGemma4MoELayerDesc array (its k_cache/v_cache/position fields are
// IGNORED — overridden by k_cache_arr[layer*n_seqs+seq] / positions[seq]).
// v1 scope mirrors the dense path: all-MoE, no PLE, no KV-donor, folded lm_head,
// no-wrap regime; CUDA-graph capture via g_g4moebatched_pool.
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4MoEModelDecodeBatched(
    const TSGgmlGemma4MoELayerDesc* layers, int num_layers, int n_seqs,
    void* hidden_data,
    void** k_cache_arr, void** v_cache_arr,   // [layer * n_seqs + seq]
    const int* positions,                     // [n_seqs]
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (layers == nullptr || hidden_data == nullptr || n_seqs <= 0 || num_layers <= 0)
        { set_last_error("Gemma4 MoE batched decode: invalid arguments."); return 0; }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        { set_last_error("Gemma4 MoE batched decode: descriptor size mismatch."); return 0; }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold) { set_last_error("Gemma4 MoE batched decode: folded lm_head required."); return 0; }

        const int H = layers[0].hidden_size;
        const int num_heads = layers[0].num_heads;
        const float eps = layers[0].eps;
        const int kvType = layers[0].kv_cache_type;

        int maxTotal = 0;
        for (int s = 0; s < n_seqs; s++) maxTotal = std::max(maxTotal, positions[s] + 1);

        static const bool g4mb_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        bool can_persist = g4mb_persist && g_backend_type == BACKEND_TYPE_CUDA;
        auto roundup_stride = [](int v){ return ((v + kG4BatchPersistKvStride - 1) / kG4BatchPersistKvStride) * kG4BatchPersistKvStride; };

        struct LInfo { int hd, kvH, qDim, cacheSize, win; bool isLocal; };
        std::vector<LInfo> li(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l];
            if (d.is_shared != 0) { set_last_error("Gemma4 MoE batched decode: KV-donor layers unsupported."); return 0; }
            auto& info = li[l];
            info.hd = d.head_dim; info.kvH = d.num_kv_heads; info.qDim = num_heads * d.head_dim;
            info.cacheSize = d.cache_size; info.isLocal = d.is_local != 0;
            if (info.cacheSize <= 0 || maxTotal > info.cacheSize)
            { set_last_error("Gemma4 MoE batched decode: sequence exceeds cache window (wrap unsupported)."); return 0; }
            info.win = can_persist
                ? std::min(info.cacheSize, std::max(roundup_stride(maxTotal), flash_attn_kv_length(maxTotal, info.cacheSize, info.hd)))
                : flash_attn_kv_length(maxTotal, info.cacheSize, info.hd);
        }

        auto fill_batched_mask = [&](std::vector<ggml_fp16_t>& md, int win)
        {
            md.assign(static_cast<std::size_t>(win) * n_seqs, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, win);
                for (int k = 0; k < valid; k++) md[static_cast<std::size_t>(s) * win + k] = static_cast<ggml_fp16_t>(0);
            }
        };

        const void* sig_disc = layers[0].attn_norm_w;
        std::vector<const void*> sig_kc(n_seqs);
        for (int s = 0; s < n_seqs; s++) sig_kc[s] = k_cache_arr[s];
        std::vector<int> winvec(num_layers);
        for (int l = 0; l < num_layers; l++) winvec[l] = li[l].win;

        // ---- reuse fast-path ----
        G4BatchedDecodeCache* dc = can_persist ? g_g4moebatched_pool.find(sig_disc, sig_kc, n_seqs) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->vocab == vocab_size && dc->layer_window == winvec)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_in, hidden_data, 0, static_cast<std::size_t>(H) * n_seqs * sizeof(float));
            ggml_backend_tensor_set(dc->pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
            for (int l = 0; l < num_layers; l++)
            {
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = positions[s];
                    ggml_backend_tensor_set(dc->kv_index[l * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
                std::vector<ggml_fp16_t> md; fill_batched_mask(md, li[l].win);
                ggml_backend_tensor_set(dc->attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
            }
            if (ggml_backend_graph_compute(g_backend, dc->graph) != GGML_STATUS_SUCCESS)
            { set_last_error("Gemma4 MoE batched decode: replay failed."); dc->reset(); return 0; }
            finalize_compute_with_download(dc->logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        // ---- build ----
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 MoE batched decode: ctx init failed."); return 0; }
        }
        else { if (!context.init(ctx_size)) { set_last_error("Gemma4 MoE batched decode: ctx pool failed."); return 0; } ctx = context.value; }

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_seqs);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        if (can_persist) { ggml_set_input(current); ggml_set_input(pos_tensor); }
        std::vector<ggml_tensor*> kv_index_all(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);

        struct LT {
            ggml_tensor *attn_norm_w, *qkv_w, *k_w, *v_w, *q_norm_w, *k_norm_w, *o_w, *post_attn_norm_w;
            ggml_tensor *ffn_norm_w, *gu_w, *down_w, *post_ffw_norm_1_w;
            ggml_tensor *gate_inp_w, *gate_inp_scale_t, *pre_ffw_norm_2_w, *gate_up_exps_t, *down_exps_t, *down_exps_scale_t, *post_ffw_norm_2_w, *post_ffw_norm_w;
            ggml_tensor *freq_factors_t;
            std::vector<ggml_tensor*> k_cached, v_cached, k_cpy, v_cpy;
            ggml_tensor* attn_mask; std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<LT> lt(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            const int nExp = d.num_experts;
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (d.separate_qkv != 0) { t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1); t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1); }
            else { t.k_w = nullptr; t.v_w = nullptr; }
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            t.gate_inp_scale_t = (d.gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
            t.pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gue_type), d.gue_ne0, d.gue_ne1, nExp);
            t.down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            t.down_exps_scale_t = (d.down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
            t.post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.freq_factors_t = (!info.isLocal && d.freq_factors != nullptr && d.freq_factors_len > 0)
                ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.freq_factors_len) : nullptr;
            t.k_cached.resize(n_seqs); t.v_cached.resize(n_seqs); t.k_cpy.resize(n_seqs, nullptr); t.v_cpy.resize(n_seqs, nullptr);
            for (int s = 0; s < n_seqs; s++)
            {
                t.k_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), info.hd, info.cacheSize, info.kvH);
                t.v_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), info.hd, info.cacheSize, info.kvH);
            }
            t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, info.win, 1, 1, n_seqs);
            if (can_persist) ggml_set_input(t.attn_mask);
        }

        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);


        ggml_tensor* hidden = current;
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            const int nExp = d.num_experts, nUsed = d.num_experts_used;
            const std::int64_t ffDense = d.gu_ne1 / 2, ffMoe = d.gue_ne1 / 2;
            ggml_tensor* rope_ff = t.freq_factors_t;

            // ---- attention (batched, per-seq KV) ----
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* q_raw; ggml_tensor* k_raw; ggml_tensor* v_raw;
            if (t.k_w != nullptr) { q_raw = ggml_mul_mat(ctx, t.qkv_w, normed); k_raw = ggml_mul_mat(ctx, t.k_w, normed); v_raw = ggml_mul_mat(ctx, t.v_w, normed); }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);
                q_raw = ggml_view_2d(ctx, qkv, info.qDim, n_seqs, qkv->nb[1], 0);
                k_raw = ggml_view_2d(ctx, qkv, info.kvH * info.hd, n_seqs, qkv->nb[1], static_cast<std::size_t>(info.qDim) * sizeof(float));
                v_raw = ggml_view_2d(ctx, qkv, info.kvH * info.hd, n_seqs, qkv->nb[1], static_cast<std::size_t>(info.qDim + info.kvH * info.hd) * sizeof(float));
            }
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), info.hd, num_heads, n_seqs);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), info.hd, info.kvH, n_seqs);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), info.hd, info.kvH, n_seqs);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), t.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), t.k_norm_w);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_3d, eps);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, pos_tensor, rope_ff, d.rope_n_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_normed, pos_tensor, rope_ff, d.rope_n_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);

            ggml_tensor* k_cat = nullptr; ggml_tensor* v_cat = nullptr;
            for (int s = 0; s < n_seqs; s++)
            {
                const int cachePos = positions[s];
                ggml_tensor* k_s = ggml_view_3d(ctx, k_rope, info.hd, info.kvH, 1, k_rope->nb[1], k_rope->nb[2], static_cast<std::size_t>(s) * k_rope->nb[2]);
                ggml_tensor* v_s = ggml_view_3d(ctx, v_normed, info.hd, info.kvH, 1, v_normed->nb[1], v_normed->nb[2], static_cast<std::size_t>(s) * v_normed->nb[2]);
                ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_s, 0, 2, 1, 3));
                ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_s, 0, 2, 1, 3));
                ggml_tensor* k_src; ggml_tensor* v_src;
                if (can_persist)
                {
                    ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                    ggml_set_input(kv_idx);
                    kv_index_all[static_cast<std::size_t>(l) * n_seqs + s] = kv_idx;
                    t.k_cpy[s] = ggml_set_rows(ctx, t.k_cached[s], k_write, kv_idx);
                    t.v_cpy[s] = ggml_set_rows(ctx, t.v_cached[s], v_write, kv_idx);
                    k_src = t.k_cpy[s]; v_src = t.v_cpy[s];
                }
                else
                {
                    ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cached[s], info.hd, 1, info.kvH, t.k_cached[s]->nb[1], t.k_cached[s]->nb[2], static_cast<std::size_t>(cachePos) * t.k_cached[s]->nb[1]);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cached[s], info.hd, 1, info.kvH, t.v_cached[s]->nb[1], t.v_cached[s]->nb[2], static_cast<std::size_t>(cachePos) * t.v_cached[s]->nb[1]);
                    t.k_cpy[s] = ggml_cpy(ctx, k_write, k_dst);
                    t.v_cpy[s] = ggml_cpy(ctx, v_write, v_dst);
                    k_src = t.k_cached[s]; v_src = t.v_cached[s];
                }
                ggml_tensor* k_win = view_kv_cache_window(ctx, k_src, info.hd, info.cacheSize, info.kvH, 0, info.win, kvType);
                ggml_tensor* v_win = view_kv_cache_window(ctx, v_src, info.hd, info.cacheSize, info.kvH, 0, info.win, kvType);
                k_cat = (k_cat == nullptr) ? k_win : ggml_concat(ctx, k_cat, k_win, 3);
                v_cat = (v_cat == nullptr) ? v_win : ggml_concat(ctx, v_cat, v_win, 3);
            }
            ggml_tensor* q4d = ggml_reshape_4d(ctx, ggml_cont(ctx, q_rope), info.hd, 1, num_heads, n_seqs);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q4d, k_cat, v_cat, t.attn_mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
            ggml_tensor* attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), info.qDim, n_seqs);
            ggml_tensor* o_out = ggml_mul_mat(ctx, t.o_w, attn_2d);
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), t.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn);   // [H, N]

            // ---- dense shared FFN (N tokens) ----
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
            ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed);
            ggml_tensor* dense_gate = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, n_seqs, gu->nb[1], 0));
            ggml_tensor* dense_up = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, n_seqs, gu->nb[1], static_cast<std::size_t>(ffDense) * sizeof(float)));
            ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, dense_gate), dense_up);
            ggml_tensor* dense_down = ggml_mul_mat(ctx, t.down_w, dense_h);
            ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), t.post_ffw_norm_1_w);

            // ---- MoE router (N tokens) ----
            ggml_tensor* route_n = ggml_rms_norm(ctx, residual1, eps);
            route_n = ggml_scale(ctx, route_n, d.inv_sqrt_hidden);
            if (t.gate_inp_scale_t != nullptr) route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, route_n);
            ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
            ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);
            ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, n_seqs);
            ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);
            ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, n_seqs);
            ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
            w_2d = ggml_div(ctx, w_2d, w_sum);
            if (t.down_exps_scale_t != nullptr)
            {
                ggml_tensor* scale_b = ggml_repeat(ctx, ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1), probs_r);
                ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_b, sel);
                w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, n_seqs));
            }
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, n_seqs);

            // ---- MoE experts (N tokens) ----
            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.pre_ffw_norm_2_w);
            ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, n_seqs);
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in_3d, sel);
            ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
            ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);
            ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, moe_act, sel);
            ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
            ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2], 0);
            for (int u = 1; u < nUsed; ++u)
            {
                ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                moe_out = ggml_add(ctx, moe_out, view_u);
            }
            ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out, eps), t.post_ffw_norm_2_w);
            mlp = ggml_add(ctx, mlp, moe_normed);

            ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w);
            ggml_tensor* result = ggml_add(ctx, residual1, mlp_normed);
            if (std::fabs(d.layer_output_scale - 1.0f) > 1e-9f) result = ggml_scale(ctx, result, d.layer_output_scale);
            hidden = result;
        }

        // fold final-norm + lm_head -> logits [vocab, N]
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);
        if (logit_softcap > 0.0f)
        {
            logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, logit_softcap);
        }
        ggml_tensor* logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_seqs);
        ggml_tensor* out_op = ggml_cpy(ctx, logits, logits_out);
        ggml_set_output(out_op);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (384 + 24 * n_seqs) + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
            for (int s = 0; s < n_seqs; s++)
            {
                if (lt[l].k_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].k_cpy[s]);
                if (lt[l].v_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].v_cpy[s]);
            }
        ggml_build_forward_expand(graph, out_op);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) { if (needs_upload) upload_list.push_back({t, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf); if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            if (t.k_w != nullptr) { bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true); bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true); }
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, static_cast<std::size_t>(d.gue_bytes), true);
            bind_or_mark(t.down_exps_t, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * d.num_experts * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr) bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr) bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, static_cast<std::size_t>(d.num_experts) * sizeof(float), true);
            if (t.freq_factors_t != nullptr) bind_or_mark(t.freq_factors_t, d.freq_factors, static_cast<std::size_t>(d.freq_factors_len) * sizeof(float), true);
            for (int s = 0; s < n_seqs; s++)
            {
                bind_or_mark(t.k_cached[s], k_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvH, info.cacheSize, info.hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cached[s], v_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvH, info.cacheSize, info.hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            fill_batched_mask(t.attn_mask_data, info.win);
            bind_or_mark(t.attn_mask, t.attn_mask_data.data(), t.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);

        // NOTE: a dedicated gallocr (packed, VRAM-frugal) was tried here to fit the
        // 26B on 16GB — it ran FAST (N=2 ~99 t/s) but produced garbage under this
        // fork's CUDA-graph capture (gallocr slot-REUSE is incompatible with the
        // captured graph; own-slot works precisely because it never reuses). So we
        // keep the capture-safe own-slot alloc; the 26B batched is VRAM-bound on 16GB
        // and stays gated off (see TS_BATCHED_FUSED_MOE).
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr) { set_last_error("Gemma4 MoE batched decode: persist alloc failed."); ggml_free(ctx); return 0; }
        }
        else if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("Gemma4 MoE batched decode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : upload_list) ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(H) * n_seqs * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
        if (can_persist)
            for (int l = 0; l < num_layers; l++)
                for (int s = 0; s < n_seqs; s++)
                { std::int64_t row = positions[s]; ggml_backend_tensor_set(kv_index_all[static_cast<std::size_t>(l) * n_seqs + s], &row, 0, sizeof(std::int64_t)); }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        { set_last_error("Gemma4 MoE batched decode: graph compute failed."); if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); } return 0; }

        finalize_compute_with_download(logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));

        if (can_persist)
        {
            host_read_barrier();
            G4BatchedDecodeCache& e = g_g4moebatched_pool.claim(sig_disc, sig_kc, n_seqs);
            e.ctx = ctx; e.buffer = persist_buf; e.graph = graph;
            e.hidden_in = current; e.pos_tensor = pos_tensor; e.logits_out = logits_out;
            e.kv_index = kv_index_all;
            e.attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) e.attn_mask[l] = lt[l].attn_mask;
            e.layer_window = winvec; e.sig_disc = sig_disc; e.sig_kc = sig_kc;
            e.num_layers = num_layers; e.hidden_size = H; e.n_seqs = n_seqs; e.vocab = vocab_size;
            e.valid = true;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Gemma4 MoE batched decode."); return 0; }
}

TSG_EXPORT void TSGgml_Gemma4ResetMoEBatchedDecodeCache()
{
    g_g4moebatched_pool.reset_all();
}

