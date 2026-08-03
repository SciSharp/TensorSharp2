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
// GPT-OSS MODEL-WIDE decode: the whole transformer (all layers, MoE included,
// plus the folded final norm + LM head) as ONE GGML graph, dispatched and
// synchronised once per token.
//
// Before this kernel a decoded token cost ~48 graph submissions (one fused
// attention kernel + one fused MoE kernel per layer), each with its own ggml
// context, buffer allocation and full device synchronise. The GPU sat at
// 40-62% utilisation waiting on the host between layers — the same shape of
// bottleneck the Gemma 4 / Qwen 3.5 whole-model decode graphs were built to
// fix, and this kernel is their GPT-OSS sibling.
//
// Per layer the graph is exactly what TSGgml_GptOssAttentionLayerPrefill builds
// at seqLen == 1 (attn RMSNorm -> fused QKV + bias -> NeoX yarn RoPE -> KV
// append -> causal/SWA-masked flash attention with sinks -> O-proj + bias ->
// residual) followed by the MoE FFN the fused MoE kernel builds (post-attn
// RMSNorm -> router logits + bias -> top-k -> softmax over the selected ->
// mul_mat_id gate/up + add_id bias -> swiglu_oai(alpha, limit) -> mul_mat_id
// down + add_id bias -> routing-weighted sum -> residual), chained layer to
// layer so layer L's residual feeds L+1.
//
// PERSIST / CUDA-graph capture (g_gptoss_dc): as in the Gemma 4 kernels, the
// graph is built ONCE with stable tensor addresses (raw ggml ctx +
// alloc_ctx_tensors). The KV write uses ggml_set_rows with the write row as an
// I64 INPUT and attention reads a FIXED padded window with an F16 mask INPUT,
// so the topology is byte-identical token to token and ggml-cuda's graph
// capture engages. Windows are padded to a 256-row stride so they only change
// (forcing a rebuild) every 256 tokens; SWA layers additionally slide their
// window start on the same stride so a 128-token window never reads more than
// 512 rows of cache. Metal falls back to the non-persist ggml_cpy path
// (ggml_set_rows segfaults there), which still collapses the token into one
// dispatch.
//
// KV residency: the K/V device copies are the SAME windows the per-layer
// prefill kernel uses (tsg_gptoss::kv_acquire), so prefill and decode share one
// device copy per cache and stay coherent. Decode does NOT download the row it
// writes — that is the traffic this kernel exists to remove — so the host cache
// goes stale and the C# model marks it dirty; host readers call
// TSGgml_GptOssSyncKvCacheToHost first.
// ============================================================================
namespace
{
    constexpr int kGptOssPersistKvStride = 256;
    constexpr int kGptOssMaxDecodeCaches = 8;
    constexpr const char* kGptOssDecodeKernel = "GPT-OSS model decode";

    struct GptOssDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        std::vector<ggml_tensor*> kv_index;     // per-layer I64 set_rows write row
        std::vector<ggml_tensor*> attn_mask;    // per-layer F16 window mask
        std::vector<int> layer_wstart;          // per-layer window start row
        std::vector<int> layer_wlen;            // per-layer window length
        std::vector<const void*> k_win_addr;    // device window addresses (detect realloc)
        std::vector<const void*> v_win_addr;
        const void* sig_disc = nullptr;         // model-instance discriminator (attn_norm[0])
        const void* sig_kcache0 = nullptr;      // first host KV pointer (detects request/realloc)
        int num_layers = 0;
        int hidden_size = 0;
        bool folded = false;
        int out_count = 0;
        // MoE CPU offload: the layers whose routed experts the host multiplies,
        // and the node index each accelerator segment stops at. The REPLAY path
        // needs these too - a cached graph run end-to-end would read whatever
        // stale bytes moe_out still held from the previous token.
        std::vector<tsg::HostMoeSegment> host_moe;
        std::vector<int> host_moe_seg_end;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = hidden_out = pos_tensor = nullptr;
            kv_index.clear(); attn_mask.clear();
            layer_wstart.clear(); layer_wlen.clear();
            k_win_addr.clear(); v_win_addr.clear();
            host_moe.clear(); host_moe_seg_end.clear();
            sig_disc = sig_kcache0 = nullptr;
            num_layers = hidden_size = 0;
            folded = false; out_count = 0;
        }
    };

    // Per-request pool: concurrent requests decode through their own KV holder,
    // so a single shared entry would rebuild on every switch and collapse N>=2
    // throughput. Keyed by (model instance, first KV cache pointer).
    struct GptOssDecodeCachePool
    {
        GptOssDecodeCache entries[kGptOssMaxDecodeCaches];
        std::uint64_t used[kGptOssMaxDecodeCaches] = {};
        std::uint64_t clock = 0;

        GptOssDecodeCache* find(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kGptOssMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        GptOssDecodeCache& claim(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kGptOssMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kGptOssMaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kGptOssMaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };

    GptOssDecodeCachePool g_gptoss_dc_pools[tsg::TSG_MAX_DEVICES];
    GptOssDecodeCachePool& gptoss_pool() { return g_gptoss_dc_pools[tsg::g_active_rank]; }

    // Rows of a device window copied back into its host mirror. The window and
    // the host cache share the [head_dim, rows, kv_heads] layout but not the row
    // capacity, so the copy is per head.
    void gptoss_kv_download(tsg_gptoss::KvWindow* w, void* host_cache, int cache_rows, std::int64_t rows)
    {
        if (w == nullptr || w->tensor == nullptr || host_cache == nullptr || rows <= 0)
            return;
        rows = std::min<std::int64_t>(rows, std::min<std::int64_t>(w->capacity, cache_rows));
        const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(w->type), w->head_dim);
        const std::size_t host_stride = static_cast<std::size_t>(cache_rows) * row_bytes;
        const std::size_t win_stride = static_cast<std::size_t>(w->capacity) * row_bytes;
        const std::size_t bytes = static_cast<std::size_t>(rows) * row_bytes;
        char* host = static_cast<char*>(host_cache);
        for (int h = 0; h < w->kv_heads; h++)
            ggml_backend_tensor_get(w->tensor, host + h * host_stride, h * win_stride, bytes);
    }

    void gptoss_kv_upload(tsg_gptoss::KvWindow* w, const void* host_cache, int cache_rows,
                          std::int64_t from_row, std::int64_t to_row)
    {
        if (w == nullptr || w->tensor == nullptr || host_cache == nullptr || to_row <= from_row)
            return;
        const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(w->type), w->head_dim);
        const std::size_t host_stride = static_cast<std::size_t>(cache_rows) * row_bytes;
        const std::size_t win_stride = static_cast<std::size_t>(w->capacity) * row_bytes;
        const std::size_t offset = static_cast<std::size_t>(from_row) * row_bytes;
        const std::size_t bytes = static_cast<std::size_t>(to_row - from_row) * row_bytes;
        const char* host = static_cast<const char*>(host_cache);
        for (int h = 0; h < w->kv_heads; h++)
            ggml_backend_tensor_set(w->tensor, host + h * host_stride + offset,
                                    h * win_stride + offset, bytes);
    }

    // Acquire the K/V window pair for a layer, preserving whatever the device
    // already holds when the window has to grow. Growth reallocates, which would
    // otherwise drop rows that only exist on the device (decode never writes them
    // back), so the old contents are flushed to the host mirror first and the
    // freshly allocated window re-uploads them from there.
    bool gptoss_acquire_pair(const TSGgmlGptOssLayerDesc& d, std::int64_t needed_rows,
                             tsg_gptoss::KvWindow*& k_win, tsg_gptoss::KvWindow*& v_win)
    {
        const ggml_type kvType = static_cast<ggml_type>(d.kv_cache_type);
        tsg_gptoss::KvWindow* existing_k = tsg_gptoss::kv_find(d.k_cache);
        tsg_gptoss::KvWindow* existing_v = tsg_gptoss::kv_find(d.v_cache);
        if (existing_k != nullptr && existing_k->capacity < needed_rows)
            gptoss_kv_download(existing_k, d.k_cache, d.cache_size, existing_k->rows_valid);
        if (existing_v != nullptr && existing_v->capacity < needed_rows)
            gptoss_kv_download(existing_v, d.v_cache, d.cache_size, existing_v->rows_valid);

        k_win = tsg_gptoss::kv_acquire(d.k_cache, d.head_dim, d.num_kv_heads, kvType, needed_rows, d.cache_size);
        v_win = (k_win == nullptr)
            ? nullptr
            : tsg_gptoss::kv_acquire(d.v_cache, d.head_dim, d.num_kv_heads, kvType, needed_rows, d.cache_size);
        if (k_win == nullptr || v_win == nullptr)
        {
            // Half a pair is worse than none: drop both so the per-layer path
            // rebuilds from the host mirror rather than trusting a stale
            // rows_valid on the survivor.
            tsg_gptoss::kv_drop_pair_locked(d.k_cache, d.v_cache);
            return false;
        }
        return true;
    }

    // Window mask for a single decode query at absolute `position`: unmasked iff
    // the cache row holds a real position that the query may attend to.
    void gptoss_fill_window_mask(std::vector<ggml_fp16_t>& mask, int wstart, int wlen,
                                 int position, bool swa, int sliding_window)
    {
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
        mask.assign(static_cast<std::size_t>(wlen), neg_inf);
        const int lo = swa ? std::max(0, position - sliding_window + 1) : 0;
        for (int i = 0; i < wlen; i++)
        {
            const int abs_row = wstart + i;
            if (abs_row >= lo && abs_row <= position)
                mask[static_cast<std::size_t>(i)] = zero_val;
        }
    }

    int gptoss_round_up(int v, int stride)
    {
        return ((v + stride - 1) / stride) * stride;
    }
}

TSG_EXPORT int TSGgml_GptOssModelDecode(
    const TSGgmlGptOssLayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int position,
    // Folded final-norm + lm_head (nullable). When all of logits_data /
    // lm_head_data / final_norm_data are supplied and vocab_size > 0 the graph
    // appends output_norm + the LM head, so the 201K-vocab projection is part of
    // the same captured replay and the C# caller skips its own tail.
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("GPT-OSS model decode: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGptOssLayerDesc)))
        {
            set_last_error("GPT-OSS model decode: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }

        const int H = hidden_size;
        const int totalSeqLen = position + 1;
        const int kvType = layers[0].kv_cache_type;
        if (kvType != GGML_TYPE_F32 && kvType != GGML_TYPE_F16)
        {
            set_last_error("GPT-OSS model decode: only F32/F16 KV caches are supported.");
            return 0;
        }

        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        const int out_count = fold ? vocab_size : H;

        static const bool gptoss_persist = []{
            const char* e = std::getenv("TS_GPTOSS_FD_PERSIST");
            return e == nullptr || e[0] != '0';
        }();
        bool can_persist = gptoss_persist &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN);

        // ---- device KV windows (shared with the per-layer prefill kernel) ----
        std::unique_lock<std::mutex> kv_lock(tsg_gptoss::kv_mutex());
        std::vector<tsg_gptoss::KvWindow*> k_wins(num_layers, nullptr);
        std::vector<tsg_gptoss::KvWindow*> v_wins(num_layers, nullptr);
        for (int l = 0; l < num_layers; l++)
        {
            if (!gptoss_acquire_pair(layers[l], totalSeqLen, k_wins[l], v_wins[l]))
            {
                set_last_error("GPT-OSS model decode: failed to acquire a device KV window.");
                return 0;
            }
        }

        // ---- per-layer attention window geometry ----
        std::vector<int> wstart(num_layers, 0);
        std::vector<int> wlen(num_layers, 0);
        // K and V are acquired as a pair with the same needed_rows so their
        // capacities normally match; take the smaller so a window that survived
        // from an earlier, larger request can never over-run its sibling.
        std::vector<int> wcap(num_layers, 0);
        for (int l = 0; l < num_layers; l++)
            wcap[l] = static_cast<int>(std::min(k_wins[l]->capacity, v_wins[l]->capacity));

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            const int capacity = wcap[l];
            const bool swa = d.is_swa != 0 && d.sliding_window > 0;
            const int lo = swa ? std::max(0, totalSeqLen - d.sliding_window) : 0;
            if (can_persist)
            {
                // Snap start and length to the persist stride so the topology is
                // stable for 256 tokens at a time.
                wstart[l] = (lo / kGptOssPersistKvStride) * kGptOssPersistKvStride;
                wlen[l] = std::min(capacity - wstart[l],
                                   gptoss_round_up(totalSeqLen - wstart[l], kGptOssPersistKvStride));
            }
            else
            {
                wstart[l] = lo;
                wlen[l] = totalSeqLen - lo;
            }
            if (wlen[l] <= 0 || wstart[l] + wlen[l] > capacity)
            {
                set_last_error("GPT-OSS model decode: invalid attention window geometry.");
                return 0;
            }
        }

        std::vector<const void*> k_addr(num_layers, nullptr);
        std::vector<const void*> v_addr(num_layers, nullptr);
        for (int l = 0; l < num_layers; l++)
        {
            k_addr[l] = k_wins[l]->tensor->data;
            v_addr[l] = v_wins[l]->tensor->data;
        }

        // Rows the device copy is missing (a fresh window, a rewind, or a jump
        // into another sequence's prefix) come from the host mirror.
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGptOssLayerDesc& d = layers[l];
            const std::int64_t kvalid = std::min<std::int64_t>(k_wins[l]->rows_valid, position);
            const std::int64_t vvalid = std::min<std::int64_t>(v_wins[l]->rows_valid, position);
            gptoss_kv_upload(k_wins[l], d.k_cache, d.cache_size, kvalid, position);
            gptoss_kv_upload(v_wins[l], d.v_cache, d.cache_size, vvalid, position);
        }

        const void* sig_disc = layers[0].attn_norm_w;
        const void* sig_kc0 = layers[0].k_cache;

        auto mark_rows_written = [&]() {
            for (int l = 0; l < num_layers; l++)
            {
                k_wins[l]->rows_valid = std::max<std::int64_t>(k_wins[l]->rows_valid, totalSeqLen);
                v_wins[l]->rows_valid = std::max<std::int64_t>(v_wins[l]->rows_valid, totalSeqLen);
            }
        };

        // ---- reuse fast-path: replay this request's captured graph ----
        GptOssDecodeCache* dc = can_persist ? gptoss_pool().find(sig_disc, sig_kc0) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->layer_wstart == wstart && dc->layer_wlen == wlen &&
            dc->k_win_addr == k_addr && dc->v_win_addr == v_addr &&
            dc->folded == fold && dc->out_count == out_count)
        {
            host_read_barrier();
            decode_input_set_async(dc->hidden_in, hidden_data, static_cast<std::size_t>(H) * sizeof(float));
            std::int32_t pos_val = position;
            decode_input_set_async(dc->pos_tensor, &pos_val, sizeof(std::int32_t));
            std::vector<ggml_fp16_t> md;
            for (int l = 0; l < num_layers; l++)
            {
                std::int64_t row = position;
                if (dc->kv_index[l] != nullptr)
                    decode_input_set_async(dc->kv_index[l], &row, sizeof(std::int64_t));
                if (dc->attn_mask[l] != nullptr)
                {
                    gptoss_fill_window_mask(md, wstart[l], wlen[l], position,
                                            layers[l].is_swa != 0, layers[l].sliding_window);
                    decode_input_set_async(dc->attn_mask[l], md.data(), md.size() * sizeof(ggml_fp16_t));
                }
            }

            // MoE CPU offload replays with the same segment cuts the build pass
            // recorded; everything else still goes out as one submission.
            ggml_status st = GGML_STATUS_SUCCESS;
            if (!dc->host_moe.empty())
            {
                if (!host_moe_execute_segments(dc->graph, dc->host_moe, dc->host_moe_seg_end, kGptOssDecodeKernel))
                    st = GGML_STATUS_FAILED;
            }
            else
            {
                st = ggml_backend_graph_compute(g_backend, dc->graph);
            }
            if (st != GGML_STATUS_SUCCESS)
            {
                if (dc->host_moe.empty())
                    set_last_error("GPT-OSS model decode: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            void* reuse_out = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, reuse_out,
                                           static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            mark_rows_written();
            clear_last_error();
            return 1;
        }

        GptOssDecodeCache* dcache = can_persist ? &gptoss_pool().claim(sig_disc, sig_kc0) : nullptr;

        // ---- build ----
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr)
            {
                set_last_error("GPT-OSS model decode: failed to init persist ggml context.");
                return 0;
            }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("GPT-OSS model decode: failed to acquire ggml context.");
                return 0;
            }
            ctx = context.value;
        }

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
            ggml_tensor* kv_index = nullptr;
            ggml_tensor* attn_mask = nullptr;
            std::vector<ggml_fp16_t> mask_data;
        };
        std::vector<LayerTensors> lt(num_layers);
        // MoE CPU offload: filled per offloaded layer while the graph is built,
        // then turned into segment boundaries once the node order is fixed.
        std::vector<tsg::HostMoeSegment> host_moe;

        ggml_tensor* hidden_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        if (can_persist)
        {
            ggml_set_input(hidden_t);
            ggml_set_input(pos_tensor);
        }

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
            // ne1 must be each window's OWN row capacity: it is the real device
            // stride between heads.
            t.k_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, k_wins[l]->capacity, kvH);
            t.v_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, v_wins[l]->capacity, kvH);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            if (d.gate_inp_b != nullptr) t.gate_inp_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp);
            // MoE CPU offload: leave the routed-expert tensors (and their
            // per-expert biases) null so the bind pass below never asks for a
            // device copy. That omission IS the VRAM saving - the host reads the
            // same bytes straight out of the GGUF mmap.
            if (d.cpu_moe == 0 || host_moe_verify_enabled())
            {
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ge_type), d.ge_ne0, d.ge_ne1, nExp);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.ue_type), d.ue_ne0, d.ue_ne1, nExp);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
                if (d.gate_exps_b != nullptr) t.gate_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ge_ne1, nExp);
                if (d.up_exps_b != nullptr) t.up_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ue_ne1, nExp);
                if (d.down_exps_b != nullptr) t.down_exps_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.de_ne1, nExp);
            }
        }

        ggml_tensor* hidden = hidden_t;
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

            // ===== Attention =====
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, d.eps), t.attn_norm_w);
            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, H, 1);

            ggml_tensor* q_raw;
            ggml_tensor* k_raw;
            ggml_tensor* v_raw;
            if (d.separate_qkv != 0)
            {
                ggml_tensor* q_proj = ggml_mul_mat(ctx, t.qkv_w, normed_2d);
                if (t.qkv_b != nullptr) q_proj = ggml_add(ctx, q_proj, t.qkv_b);
                ggml_tensor* k_proj = ggml_mul_mat(ctx, t.k_w, normed_2d);
                if (t.k_b != nullptr) k_proj = ggml_add(ctx, k_proj, t.k_b);
                ggml_tensor* v_proj = ggml_mul_mat(ctx, t.v_w, normed_2d);
                if (t.v_b != nullptr) v_proj = ggml_add(ctx, v_proj, t.v_b);
                q_raw = ggml_reshape_1d(ctx, q_proj, qDim);
                k_raw = ggml_reshape_1d(ctx, k_proj, kDim);
                v_raw = ggml_reshape_1d(ctx, v_proj, kDim);
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed_2d);
                if (t.qkv_b != nullptr) qkv = ggml_add(ctx, qkv, t.qkv_b);
                ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, qkv, qDim + 2 * kDim);
                q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
                k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim) * sizeof(float));
                v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));
            }

            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_raw, hd, nH, 1);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_raw, hd, kvH, 1);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
                d.rope_n_dims, /*mode=*/2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
                /*ext_factor=*/1.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
                d.rope_n_dims, 2, d.orig_ctx_len, d.rope_base, d.rope_freq_scale,
                1.0f, 1.0f, 32.0f, 1.0f);

            ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, hd, kvH, 1);
            ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_3d, 0, 2, 1, 3));

            if (can_persist)
            {
                ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                ggml_set_input(kv_idx);
                t.kv_index = kv_idx;
                t.k_cpy = ggml_set_rows(ctx, t.k_cache, k_write, kv_idx);
                t.v_cpy = ggml_set_rows(ctx, t.v_cache, v_write, kv_idx);
                t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, wlen[l], 1, 1, 1);
                ggml_set_input(t.attn_mask);
            }
            else
            {
                const std::size_t kv_offset = static_cast<std::size_t>(position) * t.k_cache->nb[1];
                ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cache, hd, 1, kvH,
                    t.k_cache->nb[1], t.k_cache->nb[2], kv_offset);
                ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cache, hd, 1, kvH,
                    t.v_cache->nb[1], t.v_cache->nb[2], kv_offset);
                t.k_cpy = ggml_cpy(ctx, k_write, k_dst);
                t.v_cpy = ggml_cpy(ctx, v_write, v_dst);
                t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, wlen[l], 1, 1, 1);
                gptoss_fill_window_mask(t.mask_data, wstart[l], wlen[l], position,
                                        d.is_swa != 0, d.sliding_window);
            }

            ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache, hd,
                static_cast<int>(k_wins[l]->capacity), kvH, wstart[l], wlen[l], kvType);
            ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache, hd,
                static_cast<int>(v_wins[l]->capacity), kvH, wstart[l], wlen[l], kvType);
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("GPT-OSS model decode: failed to build KV cache views.");
                if (can_persist) ggml_free(ctx);
                return 0;
            }

            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* attn_flat = nullptr;
            ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, t.attn_mask, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            if (t.sinks != nullptr)
                ggml_flash_attn_ext_add_sinks(fa, t.sinks);
            if (l == 0)
                fa_unsupported = !backend_supports_op(fa);
            if (!fa_unsupported)
            {
                attn_flat = ggml_reshape_2d(ctx, fa, qDim, 1);
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
                attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, 1);
            }

            ggml_tensor* o_mm = ggml_mul_mat(ctx, t.o_w, attn_flat);
            if (t.o_b != nullptr) o_mm = ggml_add(ctx, o_mm, t.o_b);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx, o_mm, H);
            ggml_tensor* ffn_inp = ggml_add(ctx, hidden, o_flat);

            // ===== MoE FFN =====
            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, d.eps), t.post_attn_norm_w);
            ggml_tensor* moe_in_2d = ggml_reshape_2d(ctx, moe_in, H, 1);

            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, moe_in_2d);   // [nExp, 1]
            if (t.gate_inp_b != nullptr)
                router_logits = ggml_add(ctx, router_logits, t.gate_inp_b);
            // GPT-OSS gates with SOFTMAX_WEIGHT: top-k over the RAW logits, then a
            // softmax over just the selected ones (no renormalisation afterwards).
            ggml_tensor* sel = ggml_top_k(ctx, router_logits, nUsed);
            ggml_tensor* logits_r = ggml_reshape_3d(ctx, router_logits, 1, nExp, 1);
            ggml_tensor* w = ggml_get_rows(ctx, logits_r, sel);                        // [1, nUsed, 1]
            ggml_tensor* w_soft = ggml_soft_max(ctx, ggml_reshape_2d(ctx, w, nUsed, 1));
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_soft, 1, nUsed, 1);

            ggml_tensor* moe_out_1d = nullptr;
            if (d.cpu_moe != 0)
            {
                // ---- MoE CPU offload seam (see tsg::HostMoeSegment) ----
                // Hand the host everything it needs to reproduce exactly what the
                // mul_mat_id chain below would have computed: the MoE input norm
                // and the router's own top-k ids and softmaxed weights. GPT-OSS
                // gates with SOFTMAX_WEIGHT, so w_final is already final - the
                // host applies it unchanged, and the per-expert bias tensors ride
                // along as plain host pointers.
                tsg::HostMoeSegment hm;
                hm.layer = l;
                hm.moe_in = ggml_cont(ctx, ggml_reshape_1d(ctx, moe_in, H));
                hm.sel_ids = ggml_cont(ctx, ggml_reshape_1d(ctx, sel, nUsed));
                hm.weights = ggml_cont(ctx, ggml_reshape_1d(ctx, w_final, nUsed));
                ggml_set_output(hm.moe_in);
                ggml_set_output(hm.sel_ids);
                ggml_set_output(hm.weights);

                // The host writes here between segments. Flagged BOTH input and
                // output: ggml-alloc pre-allocates inputs but still frees them
                // after their last consumer, and this one is written from outside
                // the graph, so it has to stay pinned for the whole pass.
                moe_out_1d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
                ggml_set_input(moe_out_1d);
                ggml_set_output(moe_out_1d);

                hm.moe_out = moe_out_1d;
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
                hm.seq_len = 1;
                hm.hidden = H;

                if (host_moe_verify_enabled())
                {
                    ggml_tensor* vin = ggml_reshape_3d(ctx, moe_in, H, 1, 1);
                    ggml_tensor* vg = ggml_mul_mat_id(ctx, t.gate_exps, vin, sel);
                    if (t.gate_exps_b != nullptr) vg = ggml_add_id(ctx, vg, t.gate_exps_b, sel);
                    ggml_tensor* vu = ggml_mul_mat_id(ctx, t.up_exps, vin, sel);
                    if (t.up_exps_b != nullptr) vu = ggml_add_id(ctx, vu, t.up_exps_b, sel);
                    ggml_tensor* vd = ggml_mul_mat_id(ctx, t.down_exps,
                        ggml_swiglu_oai(ctx, vg, vu, d.oai_alpha, d.oai_limit), sel);
                    if (t.down_exps_b != nullptr) vd = ggml_add_id(ctx, vd, t.down_exps_b, sel);
                    ggml_tensor* vw = ggml_mul(ctx, vd, w_final);
                    ggml_tensor* vsum = ggml_view_2d(ctx, vw, H, 1, vw->nb[2], 0);
                    for (int u = 1; u < nUsed; ++u)
                    {
                        ggml_tensor* vv = ggml_view_2d(ctx, vw, H, 1, vw->nb[2], static_cast<std::size_t>(u) * vw->nb[1]);
                        vsum = ggml_add(ctx, vsum, vv);
                    }
                    hm.verify_gpu = ggml_cont(ctx, ggml_reshape_1d(ctx, vsum, H));
                    ggml_set_output(hm.verify_gpu);
                }

                host_moe.push_back(hm);
            }
            else
            {
                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, 1);
                ggml_tensor* gate = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);     // [nFf, nUsed, 1]
                if (t.gate_exps_b != nullptr) gate = ggml_add_id(ctx, gate, t.gate_exps_b, sel);
                ggml_tensor* up = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                if (t.up_exps_b != nullptr) up = ggml_add_id(ctx, up, t.up_exps_b, sel);
                ggml_tensor* act = ggml_swiglu_oai(ctx, gate, up, d.oai_alpha, d.oai_limit);
                ggml_tensor* down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);           // [H, nUsed, 1]
                if (t.down_exps_b != nullptr) down = ggml_add_id(ctx, down, t.down_exps_b, sel);
                ggml_tensor* weighted = ggml_mul(ctx, down, w_final);

                ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], 0);
                for (int u = 1; u < nUsed; ++u)
                {
                    ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2],
                        static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, view_u);
                }
                moe_out_1d = ggml_reshape_1d(ctx, moe_out, H);
            }
            hidden = ggml_add(ctx, ffn_inp, moe_out_1d);
            (void)nFf;
        }

        ggml_tensor* lm_head_t = nullptr;
        ggml_tensor* final_norm_t = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* out_cpy = nullptr;
        if (fold)
        {
            lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, layers[0].eps), final_norm_t);
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, ggml_reshape_2d(ctx, fn, H, 1));
            logits = ggml_reshape_1d(ctx, logits, vocab_size);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
            out_cpy = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        }
        ggml_set_output(out_cpy);

        // KV writes first so they are ordered before the reads.
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 160 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // Expand layer by layer. ggml_build_forward_expand appends each root's
        // not-yet-emitted dependencies in topological order, so walking the
        // layers in order - KV writes first, then the layer's host-MoE boundary
        // if it has one - puts the cut exactly where the accelerator has to pause
        // for the host. The trailing expand of out_cpy picks up the rest.
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

        // Turn the recorded boundaries into node cut points (see
        // host_moe_build_segment_ends). It fails when the builder and the
        // expander disagree, which would silently feed the host stale values.
        std::vector<int> host_moe_seg_end;
        if (!host_moe_build_segment_ends(graph, host_moe, host_moe_seg_end, kGptOssDecodeKernel))
        {
            if (can_persist) ggml_free(ctx);
            return 0;
        }

        // ---- bind ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({tgt, data, bytes});
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
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
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
            if (!can_persist && t.attn_mask != nullptr && !t.mask_data.empty())
                bind_or_mark(t.attn_mask, t.mask_data.data(), t.mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        if (bind_failed)
        {
            set_last_error("GPT-OSS model decode: failed to bind a KV window.");
            if (can_persist) ggml_free(ctx);
            return 0;
        }
        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);
        }

        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("GPT-OSS model decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("GPT-OSS model decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * sizeof(float));
        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));
        if (can_persist)
        {
            std::int64_t row = position;
            std::vector<ggml_fp16_t> md;
            for (int l = 0; l < num_layers; l++)
            {
                if (lt[l].kv_index != nullptr)
                    ggml_backend_tensor_set(lt[l].kv_index, &row, 0, sizeof(std::int64_t));
                if (lt[l].attn_mask != nullptr)
                {
                    gptoss_fill_window_mask(md, wstart[l], wlen[l], position,
                                            layers[l].is_swa != 0, layers[l].sliding_window);
                    ggml_backend_tensor_set(lt[l].attn_mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
                }
            }
        }

        // MoE CPU offload runs the graph in segments, pausing at each offloaded
        // layer for the host expert matmul. Everything else still goes out as one
        // graph submission.
        ggml_status status = GGML_STATUS_SUCCESS;
        if (!host_moe.empty())
        {
            if (!host_moe_execute_segments(graph, host_moe, host_moe_seg_end, kGptOssDecodeKernel))
                status = GGML_STATUS_FAILED;
        }
        else
        {
            status = ggml_backend_graph_compute(g_backend, graph);
        }
        if (status != GGML_STATUS_SUCCESS)
        {
            // The segmented path already set a specific error.
            if (host_moe.empty())
                set_last_error("GPT-OSS model decode: graph execution failed.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        void* out_data = fold ? logits_data : hidden_data;
        finalize_compute_with_download(hidden_out, out_data, static_cast<std::size_t>(out_count) * sizeof(float));
        if (buffer.value != nullptr || can_persist) host_read_barrier();

        if (can_persist && dcache != nullptr)
        {
            dcache->ctx = ctx;
            dcache->buffer = persist_buf;
            dcache->graph = graph;
            dcache->hidden_in = hidden_t;
            dcache->hidden_out = hidden_out;
            dcache->pos_tensor = pos_tensor;
            dcache->kv_index.resize(num_layers);
            dcache->attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++)
            {
                dcache->kv_index[l] = lt[l].kv_index;
                dcache->attn_mask[l] = lt[l].attn_mask;
            }
            dcache->layer_wstart = wstart;
            dcache->layer_wlen = wlen;
            // MoE CPU offload: the replay path re-runs these segments, so the
            // plan has to live in the cache slot. Without it a cached graph would
            // run end to end and read whatever stale bytes moe_out still held
            // from the previous token.
            dcache->host_moe = host_moe;
            dcache->host_moe_seg_end = host_moe_seg_end;
            dcache->k_win_addr = k_addr;
            dcache->v_win_addr = v_addr;
            dcache->sig_disc = sig_disc;
            dcache->sig_kcache0 = sig_kc0;
            dcache->num_layers = num_layers;
            dcache->hidden_size = H;
            dcache->folded = fold;
            dcache->out_count = out_count;
            dcache->valid = true;
        }

        mark_rows_written();
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
        set_last_error("Unknown error in GPT-OSS model decode.");
        return 0;
    }
}

// Drop every persistent (CUDA-graph-captured) GPT-OSS decode graph. The captured
// graph pins ggml-cuda's compute-pool scratch addresses and the KV windows; a
// prefill (which grows the pool) or a KV reset/grow can move those, so the C#
// caller drops the cache before any prefill and on ResetKVCache. No-op when the
// persist path is off.
TSG_EXPORT void TSGgml_GptOssResetDecodeCache()
{
    for (auto& pool : g_gptoss_dc_pools)
        pool.reset_all();
}

// Copy the device-resident KV rows [0, rows) back into the host mirror. The
// whole-model decode graph deliberately never does this per token; anything that
// reads the host cache (KV snapshot/extract, cache growth, the legacy per-op
// attention path) calls this first. Returns 1 when a window existed, 0 when
// there was nothing to sync (host is already authoritative).
TSG_EXPORT int TSGgml_GptOssSyncKvCacheToHost(
    void* k_cache, void* v_cache, int cache_size, int rows)
{
    try
    {
        if (!ensure_backend())
            return 0;
        std::lock_guard<std::mutex> lock(tsg_gptoss::kv_mutex());
        tsg_gptoss::KvWindow* kw = tsg_gptoss::kv_find(k_cache);
        tsg_gptoss::KvWindow* vw = tsg_gptoss::kv_find(v_cache);
        if (kw == nullptr && vw == nullptr)
            return 0;
        ggml_backend_synchronize(g_backend);
        if (kw != nullptr)
            gptoss_kv_download(kw, k_cache, cache_size, std::min<std::int64_t>(rows, kw->rows_valid));
        if (vw != nullptr)
            gptoss_kv_download(vw, v_cache, cache_size, std::min<std::int64_t>(rows, vw->rows_valid));
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
        set_last_error("Unknown error in GPT-OSS KV host sync.");
        return 0;
    }
}
