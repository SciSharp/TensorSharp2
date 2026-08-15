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
#include <cmath>
#include <cstdio>

using namespace tsg;

// ============================================================================
// DFlash drafter, fused.
//
// The trunk already runs as one graph (ggml_ops_muse_glimmer.cpp); the drafter
// did not, and that is what kept speculation a net loss. Per speculative step the
// managed drafter issued ~150 separate GPU dispatches, while its actual work is
// only ~2.5 GB of weight reads (5 blocks of 467M params plus the borrowed
// 202K-vocab LM head) - about a quarter of one target forward. Dispatch overhead,
// not arithmetic, was costing ~100 ms/step.
//
// Two entry points, one graph each:
//
//   TSGgml_DFlashInject - PASS A+B. fc(feat) -> RMSNorm -> per draft layer
//       {k_proj, v_proj, per-head k RMSNorm, NeoX RoPE, set_rows into the ring}.
//       No Q, no attention, no FFN.
//
//   TSGgml_DFlashDraftBlock - PASS C. [anchor, MASK x (b-1)] through the draft
//       blocks, then the TARGET's LM head and a softmax, returning [vocab, b]
//       probabilities. Attention reads the WHOLE ring plus this block's own keys.
//
// Reading the whole ring (rather than the live window) is deliberate: attention is
// permutation-invariant over the KV axis, so the ring's circular order does not
// matter, and a FIXED [ring_rows + b] key length keeps the graph shape constant
// across steps - which is what lets ggml-cuda capture and replay it. The host
// supplies a per-slot position map so the mask can express exactly which slots are
// live and visible.
//
// Mask semantics mirror the (llama.cpp-verified) managed implementation: a ring
// slot holding position p is masked from a block query at position qp when
// qp - p >= n_swa or p > qp; the block's own b columns are never masked, because
// DFlash drafts with llama_set_causal_attn(ctx_dft, false).
// ============================================================================

namespace
{
    struct DfLayer
    {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* q_w = nullptr;
        ggml_tensor* k_w = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* q_norm_w = nullptr;
        ggml_tensor* k_norm_w = nullptr;
        ggml_tensor* o_w = nullptr;
        ggml_tensor* ffn_norm_w = nullptr;
        ggml_tensor* gate_w = nullptr;
        ggml_tensor* up_w = nullptr;
        ggml_tensor* down_w = nullptr;
        ggml_tensor* ring_k = nullptr;
        ggml_tensor* ring_v = nullptr;
        ggml_tensor* k_cpy = nullptr;
        ggml_tensor* v_cpy = nullptr;
    };

    // Persistent graph cache. Both entry points key on
    // (model instance, ring buffer, row count) - see the pool in
    // ggml_ops_muse_glimmer.cpp for why n_tokens has to be part of the identity.
    struct DfCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* in_main = nullptr;      // feature rows (inject) or token ids (draft)
        ggml_tensor* pos = nullptr;
        ggml_tensor* kv_index = nullptr;     // inject only
        ggml_tensor* mask = nullptr;         // draft only (shared across layers)
        ggml_tensor* out = nullptr;
        ggml_tensor* out_conf = nullptr;   // draft only
        const void* sig = nullptr;
        const void* sig_ring = nullptr;
        int n_rows = 0;
        int out_count = 0;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            in_main = pos = kv_index = mask = out = out_conf = nullptr;
            sig = sig_ring = nullptr;
            n_rows = out_count = 0;
        }
    };

    constexpr int kDfMaxCaches = 8;
    struct DfCachePool
    {
        DfCache entries[kDfMaxCaches];
        std::uint64_t used[kDfMaxCaches] = {};
        std::uint64_t clock = 0;

        DfCache* find(const void* sig, const void* ring, int n_rows)
        {
            for (int i = 0; i < kDfMaxCaches; i++)
                if (entries[i].valid && entries[i].sig == sig && entries[i].sig_ring == ring
                    && entries[i].n_rows == n_rows)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        DfCache& claim(const void* sig, const void* ring, int n_rows)
        {
            for (int i = 0; i < kDfMaxCaches; i++)
                if (entries[i].valid && entries[i].sig == sig && entries[i].sig_ring == ring
                    && entries[i].n_rows == n_rows)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kDfMaxCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kDfMaxCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };

    DfCachePool g_df_inject_pools[tsg::TSG_MAX_DEVICES];
    DfCachePool g_df_draft_pools[tsg::TSG_MAX_DEVICES];
    inline DfCachePool& df_inject_pool() { return g_df_inject_pools[tsg::g_active_rank]; }
    inline DfCachePool& df_draft_pool()  { return g_df_draft_pools[tsg::g_active_rank]; }

    bool df_persist_enabled()
    {
        static const bool on = [] {
            const char* e = std::getenv("TS_DFLASH_PERSIST");
            return e == nullptr || e[0] != '0';
        }();
        return on;
    }

    struct DfBinder
    {
        struct Upload { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<Upload> uploads;
        std::vector<BufferHandle> ephemeral;
        ggml_backend_dev_t dev = nullptr;

        void bind(ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                  enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, t, data, bytes, needs_upload, usage))
                {
                    if (needs_upload) uploads.push_back({ t, data, bytes });
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            uploads.push_back({ t, data, bytes });
        }

        void flush()
        {
            for (auto& u : uploads)
                ggml_backend_tensor_set(u.t, resolve_upload_source(u.data), 0, u.bytes);
        }
    };
}

// ---------------------------------------------------------------------------
// PASS A + B: encode the target's per-layer residual features and inject the
// resulting K/V into the drafter's ring.
// ---------------------------------------------------------------------------
TSG_EXPORT int TSGgml_DFlashInject(
    const float* feat_rows, int feature_size, int n_rows,
    const std::int64_t* ring_rows_idx,          // ring slot per row (position % ring_rows)
    const int* positions,                        // absolute position per row (for RoPE)
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr,
    void** ring_k_arr, void** ring_v_arr,
    int ring_dtype)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (n_rows <= 0 || num_layers <= 0) { set_last_error("DFlash inject: invalid arguments."); return 0; }

        const void* sig = k_arr[0];
        const void* sig_ring = ring_k_arr[0];
        // Metal replays the persistent graph without a capture (every submit
        // re-encodes) but still skips the per-step metadata rebuild, re-binds
        // and gallocr re-plan - same rationale as the trunk kernel.
        const bool can_persist = df_persist_enabled() &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN ||
             g_backend_type == BACKEND_TYPE_METAL);

        DfCache* dc = can_persist ? df_inject_pool().find(sig, sig_ring, n_rows) : nullptr;
        if (dc != nullptr && dc->graph != nullptr)
        {
            host_read_barrier();
            decode_input_set_async(dc->in_main, feat_rows,
                static_cast<std::size_t>(feature_size) * n_rows * sizeof(float));
            decode_input_set_async(dc->pos, positions, static_cast<std::size_t>(n_rows) * sizeof(std::int32_t));
            decode_input_set_async(dc->kv_index, ring_rows_idx, static_cast<std::size_t>(n_rows) * sizeof(std::int64_t));
            if (tsg::graph_compute_profiled(g_backend, dc->graph, "dflash inject") != GGML_STATUS_SUCCESS)
            {
                set_last_error("DFlash inject: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        DfCache* slot = can_persist ? &df_inject_pool().claim(sig, sig_ring, n_rows) : nullptr;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle pooled;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("DFlash inject: failed to init context."); return 0; }
        }
        else
        {
            if (!pooled.init(ctx_size)) { set_last_error("DFlash inject: failed to create context."); return 0; }
            ctx = pooled.value;
        }

        ggml_tensor* feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feature_size, n_rows);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
        ggml_tensor* idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_rows);
        ggml_set_input(feat); ggml_set_input(pos_t); ggml_set_input(idx_t);

        ggml_tensor* fc_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(fc_type), fc_ne0, fc_ne1);
        ggml_tensor* enc_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // g = rmsnorm(fc @ feat) * enc_output_norm
        ggml_tensor* g = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_mul_mat(ctx, fc_t, feat), eps), enc_norm_t);

        std::vector<DfLayer> layers(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
            lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.ring_k = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
            lt.ring_v = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);

            ggml_tensor* k = ggml_mul_mat(ctx, lt.k_w, g);                 // [kv_dim, n]
            ggml_tensor* v = ggml_mul_mat(ctx, lt.v_w, g);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, k, head_dim, num_kv_heads, n_rows);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, n_rows);
            k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, eps), lt.k_norm_w);
            // NeoX rope: llama.cpp maps LLM_ARCH_DFLASH to LLAMA_ROPE_TYPE_NEOX,
            // which differs from the Muse-Glimmer trunk's NORM flavour. V is neither
            // normed nor roped.
            k3 = ggml_rope_ext(ctx, k3, pos_t, nullptr, head_dim, /*mode=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor* kw = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));   // [hd, n, kv_heads]
            ggml_tensor* vw = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
            lt.k_cpy = ggml_set_rows(ctx, lt.ring_k, kw, idx_t);
            lt.v_cpy = ggml_set_rows(ctx, lt.ring_v, vw, idx_t);
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(num_layers) * 64 + 256, false);
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, layers[l].k_cpy);
            ggml_build_forward_expand(graph, layers[l].v_cpy);
        }

        DfBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        binder.bind(fc_t, const_cast<void*>(fc_data), static_cast<std::size_t>(fc_bytes), true);
        binder.bind(enc_norm_t, const_cast<void*>(enc_norm_data),
                    static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        const std::size_t ring_bytes = kv_cache_bytes(num_kv_heads, ring_rows, head_dim, ring_dtype);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            binder.bind(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
            binder.bind(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            binder.bind(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(head_dim) * sizeof(float), true);
            binder.bind(lt.ring_k, ring_k_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            binder.bind(lt.ring_v, ring_v_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }

        // Metal: reorder for encoder concurrency before allocation (no-op on
        // other backends) - same as the trunk kernel.
        optimize_graph_for_metal(graph);

        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("DFlash inject: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else
        {
            for (auto& u : binder.uploads) ggml_set_input(u.t);
            if (!alloc_graph_reuse_gallocr(graph))
            {
                set_last_error("DFlash inject: failed to allocate graph.");
                return 0;
            }
        }

        host_read_barrier();
        binder.flush();
        ggml_backend_tensor_set(feat, feat_rows, 0, static_cast<std::size_t>(feature_size) * n_rows * sizeof(float));
        ggml_backend_tensor_set(pos_t, positions, 0, static_cast<std::size_t>(n_rows) * sizeof(std::int32_t));
        ggml_backend_tensor_set(idx_t, ring_rows_idx, 0, static_cast<std::size_t>(n_rows) * sizeof(std::int64_t));

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("DFlash inject: graph execution failed.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }
        host_read_barrier();

        if (can_persist && slot != nullptr)
        {
            slot->ctx = ctx; slot->buffer = persist_buf; slot->graph = graph;
            slot->in_main = feat; slot->pos = pos_t; slot->kv_index = idx_t;
            slot->sig = sig; slot->sig_ring = sig_ring; slot->n_rows = n_rows;
            slot->valid = true;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in DFlash inject."); return 0; }
}

// ---------------------------------------------------------------------------
// PASS C: the block draft.
// ---------------------------------------------------------------------------
TSG_EXPORT int TSGgml_DFlashDraftBlock(
    const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    // ring slot -> absolute position, or -1 when the slot is not live
    const int* ring_slot_pos, int sliding_window,
    void** attn_norm_arr,
    void** q_arr, int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    void** ffn_norm_arr,
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,
    void** down_arr, int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const void* out_norm_data,
    const void* tok_embd_data, int tok_embd_type, std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    int vocab_size, int* ids_out, float* conf_out)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (block_len <= 0 || num_layers <= 0 || ids_out == nullptr || conf_out == nullptr)
        { set_last_error("DFlash draft: invalid arguments."); return 0; }

        const int b = block_len;
        const int kv_len = ring_rows + b;      // FIXED across steps -> capturable
        const int q_dim = num_heads * head_dim;
        const int out_count = b;

        const void* sig = attn_norm_arr[0];
        const void* sig_ring = ring_k_arr[0];
        // Metal is included for the same reason as the inject graph above.
        const bool can_persist = df_persist_enabled() &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN ||
             g_backend_type == BACKEND_TYPE_METAL);

        // Mask [kv_len, b]: ring slot s holds ring_slot_pos[s] (or -1 = dead);
        // it is visible to block query i (position positions[i]) when the slot is
        // live, not in the future, and inside the sliding window. The block's own
        // b columns are always visible - DFlash drafts non-causally within a block.
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kv_len) * b,
                                           ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        const ggml_fp16_t vis = static_cast<ggml_fp16_t>(0);
        const int anchor_pos = positions[0];
        for (int i = 0; i < b; i++)
        {
            const int qp = positions[i];
            ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(i) * kv_len];
            for (int s = 0; s < ring_rows; s++)
            {
                const int p = ring_slot_pos[s];
                // Cutoff is the ANCHOR's position, not this query's. After a
                // partially-rejected draft the ring still holds keys the drafter
                // wrote for positions past the anchor; those rows are stale and no
                // query in the block may see them, even the later ones whose own
                // position is higher. The per-op path gets this for free by only
                // ever gathering [winStart, anchor).
                if (p < 0 || p >= anchor_pos) continue;
                if (sliding_window > 0 && qp - p >= sliding_window) continue;
                row[s] = vis;
            }
            for (int j = 0; j < b; j++)
                row[ring_rows + j] = vis;
        }

        DfCache* dc = can_persist ? df_draft_pool().find(sig, sig_ring, b) : nullptr;
        if (dc != nullptr && dc->graph != nullptr && dc->out_count == out_count)
        {
            host_read_barrier();
            decode_input_set_async(dc->in_main, block_ids, static_cast<std::size_t>(b) * sizeof(std::int32_t));
            decode_input_set_async(dc->pos, positions, static_cast<std::size_t>(b) * sizeof(std::int32_t));
            decode_input_set_async(dc->mask, mask_data.data(), mask_data.size() * sizeof(ggml_fp16_t));
            if (tsg::graph_compute_profiled(g_backend, dc->graph, "dflash draft") != GGML_STATUS_SUCCESS)
            {
                set_last_error("DFlash draft: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            // Metal's graph_compute is async and a shared-buffer tensor_get is a
            // raw memcpy, so the draft ids must not be read until the queue
            // drains - otherwise the executor verifies stale drafts (correct
            // output, silently collapsed acceptance).
            if (g_backend_type == BACKEND_TYPE_METAL)
                ggml_backend_synchronize(g_backend);
            ggml_backend_tensor_get(dc->out, ids_out, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
            finalize_compute_with_download(dc->out_conf, conf_out, static_cast<std::size_t>(b) * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        DfCache* slot = can_persist ? &df_draft_pool().claim(sig, sig_ring, b) : nullptr;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle pooled;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("DFlash draft: failed to init context."); return 0; }
        }
        else
        {
            if (!pooled.init(ctx_size)) { set_last_error("DFlash draft: failed to create context."); return 0; }
            ctx = pooled.value;
        }

        ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_len, b, 1, 1);
        ggml_set_input(ids_t); ggml_set_input(pos_t); ggml_set_input(mask_t);

        ggml_tensor* tok_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(tok_embd_type), tok_embd_ne0, tok_embd_ne1);
        ggml_tensor* out_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);

        // llama.cpp's dflash graph feeds build_inp_embd straight in: no embedding
        // scale and (unlike the Muse-Glimmer trunk) no weightless input RMSNorm.
        ggml_tensor* inpL = ggml_get_rows(ctx, tok_t, ids_t);              // [hidden, b]

        std::vector<DfLayer> layers(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.q_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.k_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.q_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(q_type_arr[l]),    q_ne0_arr[l],    q_ne1_arr[l]);
            lt.k_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]),    k_ne0_arr[l],    k_ne1_arr[l]);
            lt.v_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]),    v_ne0_arr[l],    v_ne1_arr[l]);
            lt.o_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]),    o_ne0_arr[l],    o_ne1_arr[l]);
            lt.gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gate_type_arr[l]), gate_ne0_arr[l], gate_ne1_arr[l]);
            lt.up_w   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(up_type_arr[l]),   up_ne0_arr[l],   up_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.ring_k = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
            lt.ring_v = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
        }

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];

            ggml_tensor* h = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, eps), lt.attn_norm_w);
            ggml_tensor* q = ggml_mul_mat(ctx, lt.q_w, h);
            ggml_tensor* k = ggml_mul_mat(ctx, lt.k_w, h);
            ggml_tensor* v = ggml_mul_mat(ctx, lt.v_w, h);

            ggml_tensor* q3 = ggml_reshape_3d(ctx, q, head_dim, num_heads, b);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, k, head_dim, num_kv_heads, b);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, b);
            q3 = ggml_mul(ctx, ggml_rms_norm(ctx, q3, eps), lt.q_norm_w);
            k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, eps), lt.k_norm_w);
            q3 = ggml_rope_ext(ctx, q3, pos_t, nullptr, head_dim, /*NeoX=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_ext(ctx, k3, pos_t, nullptr, head_dim, /*NeoX=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

            // [ring | this block's own keys]; the ring is read whole and the mask
            // decides which slots are live (attention is order-invariant over KV).
            ggml_tensor* kb = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));   // [hd, b, kv_heads]
            ggml_tensor* vb = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
            ggml_tensor* kcat = ggml_concat(ctx, lt.ring_k, kb, 1);
            ggml_tensor* vcat = ggml_concat(ctx, lt.ring_v, vb, 1);

            ggml_tensor* qa = ggml_permute(ctx, q3, 0, 2, 1, 3);                   // [hd, b, heads]
            ggml_tensor* attn_flat;
            ggml_tensor* fa = ggml_flash_attn_ext(ctx, qa, kcat, vcat, mask_t, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            if (backend_supports_op(fa))
            {
                attn_flat = ggml_reshape_2d(ctx, fa, q_dim, b);
            }
            else
            {
                ggml_tensor* qc = ggml_cont(ctx, qa);
                ggml_tensor* scores = ggml_mul_mat(ctx, kcat, qc);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask_t, kq_scale, 0.0f);
                ggml_tensor* vperm = ggml_cont(ctx, ggml_permute(ctx, vcat, 1, 0, 2, 3));
                ggml_tensor* o = ggml_mul_mat(ctx, vperm, probs);
                attn_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), q_dim, b);
            }

            ggml_tensor* attn_out = ggml_mul_mat(ctx, lt.o_w, attn_flat);
            ggml_tensor* ffn_inp = ggml_add(ctx, attn_out, inpL);

            ggml_tensor* fh = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, eps), lt.ffn_norm_w);
            ggml_tensor* gate = ggml_mul_mat(ctx, lt.gate_w, fh);
            ggml_tensor* up   = ggml_mul_mat(ctx, lt.up_w, fh);
            ggml_tensor* act  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            ggml_tensor* down = ggml_mul_mat(ctx, lt.down_w, act);
            inpL = ggml_add(ctx, down, ffn_inp);
        }

        ggml_tensor* cur = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, eps), out_norm_t);
        // The TARGET's LM head, with NEITHER logit_scale NOR the tanh softcap:
        // llama.cpp's dflash graph ends at build_lora_mm(output, cur).
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, cur);        // [vocab, b]
        // Softmax on device: argmax is invariant under it, and the winning
        // probability IS the confidence the executor multiplies cumulatively.
        ggml_tensor* probs = ggml_soft_max(ctx, logits);

        // Reduce to (argmax id, winning probability) ON DEVICE. llama.cpp pulls the
        // whole [vocab, b] block back every draft step -- 202048*16*4 = 12.9 MB of
        // PCIe traffic plus a 3.2 M-element host scan -- and that readback is a
        // large share of its per-step cost. Two b-element tensors carry everything
        // the caller needs: argmax is invariant under softmax, and the winning
        // probability IS the confidence the executor multiplies cumulatively.
        ggml_tensor* am = ggml_argmax(ctx, probs);                            // [b] I32
        ggml_tensor* am2 = ggml_reshape_2d(ctx, am, 1, b);                    // [1, b]
        ggml_tensor* pr3 = ggml_reshape_3d(ctx, probs, 1, vocab_size, b);     // [1, vocab, b]
        ggml_tensor* mp = ggml_get_rows(ctx, pr3, am2);                       // [1, 1, b] F32
        ggml_tensor* out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
        ggml_tensor* out_conf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b);
        ggml_tensor* out_node = ggml_cpy(ctx, am, out);
        ggml_tensor* conf_node = ggml_cpy(ctx, ggml_reshape_1d(ctx, mp, b), out_conf);
        ggml_set_output(out_node);
        ggml_set_output(conf_node);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(num_layers) * 128 + 512, false);
        ggml_build_forward_expand(graph, out_node);
        ggml_build_forward_expand(graph, conf_node);

        DfBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        const std::size_t norm_bytes = static_cast<std::size_t>(hidden_size) * sizeof(float);
        const std::size_t head_norm_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
        const std::size_t ring_bytes = kv_cache_bytes(num_kv_heads, ring_rows, head_dim, ring_dtype);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            binder.bind(lt.q_w,    q_arr[l],    static_cast<std::size_t>(q_bytes_arr[l]), true);
            binder.bind(lt.k_w,    k_arr[l],    static_cast<std::size_t>(k_bytes_arr[l]), true);
            binder.bind(lt.v_w,    v_arr[l],    static_cast<std::size_t>(v_bytes_arr[l]), true);
            binder.bind(lt.o_w,    o_arr[l],    static_cast<std::size_t>(o_bytes_arr[l]), true);
            binder.bind(lt.gate_w, gate_arr[l], static_cast<std::size_t>(gate_bytes_arr[l]), true);
            binder.bind(lt.up_w,   up_arr[l],   static_cast<std::size_t>(up_bytes_arr[l]), true);
            binder.bind(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);
            binder.bind(lt.attn_norm_w, attn_norm_arr[l], norm_bytes, true);
            binder.bind(lt.ffn_norm_w,  ffn_norm_arr[l],  norm_bytes, true);
            binder.bind(lt.q_norm_w,    q_norm_arr[l],    head_norm_bytes, true);
            binder.bind(lt.k_norm_w,    k_norm_arr[l],    head_norm_bytes, true);
            binder.bind(lt.ring_k, ring_k_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            binder.bind(lt.ring_v, ring_v_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        binder.bind(out_norm_t, const_cast<void*>(out_norm_data), norm_bytes, true);
        binder.bind(tok_t, const_cast<void*>(tok_embd_data), static_cast<std::size_t>(tok_embd_bytes), true);
        binder.bind(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);

        // Metal: reorder for encoder concurrency before allocation (no-op on
        // other backends).
        optimize_graph_for_metal(graph);

        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("DFlash draft: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else
        {
            for (auto& u : binder.uploads) ggml_set_input(u.t);
            if (!alloc_graph_reuse_gallocr(graph))
            {
                set_last_error("DFlash draft: failed to allocate graph.");
                return 0;
            }
        }

        host_read_barrier();
        binder.flush();
        ggml_backend_tensor_set(ids_t, block_ids, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
        ggml_backend_tensor_set(pos_t, positions, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("DFlash draft: graph execution failed.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }
        // See the replay path above: Metal must drain before reading the ids.
        if (g_backend_type == BACKEND_TYPE_METAL)
            ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(out, ids_out, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
        finalize_compute_with_download(out_conf, conf_out, static_cast<std::size_t>(b) * sizeof(float));
        // Unconditional: conf_out is the caller's host confidence array and on
        // Metal async mode the download above is only QUEUED.
        host_read_barrier();

        if (can_persist && slot != nullptr)
        {
            slot->ctx = ctx; slot->buffer = persist_buf; slot->graph = graph;
            slot->in_main = ids_t; slot->pos = pos_t; slot->mask = mask_t;
            slot->out = out; slot->out_conf = out_conf;
            slot->sig = sig; slot->sig_ring = sig_ring; slot->n_rows = b; slot->out_count = out_count;
            slot->valid = true;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in DFlash draft."); return 0; }
}

TSG_EXPORT void TSGgml_DFlashResetCaches()
{
    for (auto& p : g_df_inject_pools) p.reset_all();
    for (auto& p : g_df_draft_pools)  p.reset_all();
}
