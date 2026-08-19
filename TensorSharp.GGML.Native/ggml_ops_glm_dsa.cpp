// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// GLM (glm-dsa) whole-model executor for the GGML backends.
//
// Like ggml_ops_deepseek4.cpp — and for the same reason — this is a
// self-contained model runtime rather than a per-op fused path: GLM-5.2 is
// 744B parameters (226 GiB at IQ2_XXS), so its weights have to be placed
// across every visible GPU and every token has to be one graph submission.
// A per-op managed forward would spend more time in host round trips than in
// arithmetic.
//
// The graph is a port of llama.cpp's `src/models/glm-dsa.cpp`:
//   * MLA with the absorption optimisation. The cache holds ONE compressed row
//     per token (kv_lora_rank + n_rot = 576 halfs), and the per-head K/V
//     decompression is folded into the query (wk_b) and the output (wv_b), so
//     64 query heads attend to a single key head.
//   * A DeepSeek-style "lightning indexer": a 32-head / 128-dim scorer over the
//     whole cache whose top-k (2048) selects which cached tokens the real
//     attention may see. GLM computes it on ~1 layer in 4 and shares the
//     selection with the layers in between (glm-dsa.attention.indexer.types).
//   * Sigmoid-gated MoE with a selection bias, weight renormalisation, a x2.5
//     routed scale, and one always-on shared expert. The leading
//     `leading_dense_block_count` layers are dense SwiGLU instead.
//
// Placement, offload and caching follow the DSV4 executor:
//   * layer split across GPUs, sized against each device's free VRAM;
//   * `--n-cpu-moe` moves the leading layers' routed experts to system RAM,
//     served zero-copy from the GGUF mmap (the routed experts are 92% of this
//     checkpoint, so it is the only knob with enough range to matter);
//   * an LRU cache of built+allocated graphs keyed by shape, so decode reuses
//     one graph (and one CUDA-graph capture) token after token.
//
// The Hadamard rotation llama.cpp applies to the indexer keys IS reproduced.
// It is an orthonormal involution applied to both sides of a dot product, so in
// exact arithmetic it cancels — but the indexer key cache is F16, and rotating
// before rounding is what keeps the error even across the 128 dimensions.
// Skipping it changed which tokens the top-k picked at long context and broke
// token-for-token parity with llama.cpp; reproducing it restores it.
// ---------------------------------------------------------------------------

#include "ggml_ops_internal.h"

#include "gguf.h"

#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tsg_glm
{

/// TS_GLM_BD_DEBUG=1 narrates the batched-decode path — which sequences are in
/// the batch, whether their graph was reused, and how far the step got. Batched
/// decode is the one place where a single step touches N sequence slots at once,
/// so when something goes wrong the first question is always which of them.
static bool bd_debug()
{
    static const bool on = []() { const char * e = getenv("TS_GLM_BD_DEBUG"); return e && atoi(e) != 0; }();
    return on;
}

static void bd_log(const char * fmt, ...)
{
    if (!bd_debug()) return;
    fprintf(stderr, "[glm-bd] ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
}

static constexpr int MAX_GPUS = 8;
/// Sequences one batched decode step will serve. Past this the shared weight
/// read stops paying for the extra per-token attention subgraphs.
static constexpr int MAX_BATCHED_DECODE = 16;

static int64_t pad_to(int64_t v, int64_t p) { return ((v + p - 1) / p) * p; }

// ---------------------------------------------------------------------------
// hyper-parameters
// ---------------------------------------------------------------------------
struct glm_hparams
{
    int32_t n_layer_all = 0;     // block_count, including the NextN/MTP block(s)
    int32_t n_layer = 0;         // trunk blocks the main graph runs
    int32_t n_layer_nextn = 0;
    int32_t n_dense_lead = 0;
    int32_t n_embd = 0;
    int32_t n_head = 0;
    int32_t n_rot = 0;
    int32_t n_embd_head_k = 0;   // n_embd_head_k_mla
    int32_t n_embd_head_v = 0;   // n_embd_head_v_mla
    int32_t n_nope = 0;          // n_embd_head_k - n_rot
    int32_t q_lora_rank = 0;
    int32_t kv_lora_rank = 0;
    int32_t n_kv_row = 0;        // kv_lora_rank + n_rot: one cache row
    int32_t n_ff = 0;            // dense feed_forward_length
    int32_t n_ff_exp = 0;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_expert_shared = 0;
    float   expert_weights_scale = 1.0f;
    bool    expert_weights_norm = false;
    int32_t expert_gating_func = 2;   // 1 softmax, 2 sigmoid
    int32_t indexer_n_head = 0;
    int32_t indexer_head_size = 0;
    int32_t indexer_top_k = 0;
    std::vector<uint8_t> indexer_full;   // per trunk layer
    float   rms_eps = 1e-5f;
    float   rope_freq_base = 10000.0f;
    int32_t n_ctx_train = 0;
    int32_t n_vocab = 0;

    float kq_scale() const { return 1.0f / sqrtf((float) n_embd_head_k); }
};

// One rank's view of a layer. In layer-split mode only rank 0 exists and holds
// the whole layer; under tensor parallelism every rank has its own set, with the
// head-sharded and expert-sharded matrices sliced and everything else replicated.
struct glm_layer_weights
{
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq_a = nullptr;
    ggml_tensor * q_a_norm = nullptr;
    ggml_tensor * wq_b = nullptr;        // column-parallel: this rank's heads
    ggml_tensor * wkv_a_mqa = nullptr;
    ggml_tensor * kv_a_norm = nullptr;
    ggml_tensor * wk_b = nullptr;        // this rank's heads
    ggml_tensor * wv_b = nullptr;        // this rank's heads
    ggml_tensor * wo = nullptr;          // row-parallel: this rank's head columns

    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;

    ggml_tensor * ffn_gate_inp = nullptr;
    ggml_tensor * exp_probs_b = nullptr;
    ggml_tensor * ffn_gate_exps = nullptr;   // this rank's experts
    ggml_tensor * ffn_up_exps = nullptr;
    ggml_tensor * ffn_down_exps = nullptr;
    ggml_tensor * ffn_gate_shexp = nullptr;
    ggml_tensor * ffn_up_shexp = nullptr;
    ggml_tensor * ffn_down_shexp = nullptr;

    ggml_tensor * idx_attn_q_b = nullptr;
    ggml_tensor * idx_attn_k = nullptr;
    ggml_tensor * idx_k_norm_w = nullptr;
    ggml_tensor * idx_k_norm_b = nullptr;
    ggml_tensor * idx_proj = nullptr;
};

struct glm_layer
{
    glm_layer_weights w[MAX_GPUS];

    int  device = 0;          // layer-split: the device that hosts this layer
    bool cpu_moe = false;     // routed experts live in host RAM
    bool indexer_full = false;
    bool is_moe = false;

    // Shard boundaries under tensor parallelism (rank r owns [first[r], first[r+1])).
    int head_first[MAX_GPUS + 1] = {};
    /// Where each rank's slice of the routed experts' hidden dimension starts.
    /// The experts are split row-wise (every rank holds a strip of every
    /// expert), not by expert id: that keeps the router's global top-k ids valid
    /// on every rank, which ggml_mul_mat_id requires — it cannot handle the
    /// duplicate ids an id-space split would have to invent for the experts a
    /// rank does not own.
    int moe_ff_first[MAX_GPUS + 1] = {};
};

/// First index owned by rank `r` when `total` items are spread over `n` ranks;
/// the remainder goes to the leading ranks, and `shard_first(total, n, n)` is
/// `total`, so a count is always `first[r+1] - first[r]`.
static int shard_first(int total, int n, int r)
{
    const int base = total / n;
    const int rem = total % n;
    return r * base + std::min(r, rem);
}

/// Same contiguous tiling, but the remainder is handed out starting at rank
/// `rot` instead of always at rank 0. The routed experts are split in whole
/// quantization blocks, and GLM-5.2 has only eight of them per expert row: at
/// tp=3 a fixed split gives 3/3/2, so one rank carries a third less weight and
/// does a third less work in every one of the 78 layers. Rotating by layer
/// index evens the totals out without breaking contiguity.
static int shard_first_rot(int total, int n, int r, int rot)
{
    const int base = total / n;
    const int rem = total % n;
    int first = base * r;
    for (int k = 0; k < r; k++)
        if (((k - rot) % n + n) % n < rem) first++;
    return first;
}

// Per-sequence caches. One row per token per layer; the indexer cache exists
// only on layers that compute a fresh top-k.
struct glm_slot
{
    int id = 0;
    int64_t n_past = 0;
    // [rank][layer]. Under tensor parallelism every rank keeps its own copy of
    // the (rank-independent) MLA and indexer rows, so attention never reads
    // across the split and no collective is needed to share them.
    std::vector<ggml_tensor *> kv_k[MAX_GPUS];    // [n_kv_row, n_ctx] F16
    std::vector<ggml_tensor *> idx_k[MAX_GPUS];   // [indexer_head_size, n_ctx] F16 (or null)
};

struct tensor_source
{
    int shard = -1;
    size_t offset = 0;
    size_t size = 0;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
};

struct shard_files
{
    std::vector<std::string> paths;
};

struct graph_inputs
{
    ggml_tensor * tokens[MAX_GPUS + 1] = {};   // I32 [nt]
    ggml_tensor * pos[MAX_GPUS + 1] = {};      // I32 [nt]
    ggml_tensor * kq_mask[MAX_GPUS + 1] = {};  // F16/F32 [n_kv, nt]
    ggml_tensor * lid_mask[MAX_GPUS + 1] = {}; // F16 [n_kv, nt] (sparse layers only)
    ggml_tensor * kv_idxs[MAX_GPUS + 1] = {};  // I64 [nt] destination cache rows
    ggml_tensor * out_ids = nullptr;           // I32 [n_out]
};

/// Named intermediates kept as graph outputs when TS_GLM_TRACE names layers,
/// so a cross-implementation (or cross-configuration) divergence can be walked
/// back to the first tensor that differs.
struct trace_entry
{
    std::string name;
    ggml_tensor * t;
};

/// One token of a batched decode step: which sequence slot it belongs to, where
/// in that slot's cache it lands, and the per-token inputs whose shape depends
/// on that slot's length. Everything else in the step — every projection, the
/// router and the experts — is shared across the batch, which is the point:
/// N concurrent requests read the weights once between them instead of N times.
struct bd_token
{
    int slot_id = 0;
    int64_t p = 0;            // position of this token in its sequence
    int64_t n_kv = 0;         // cache rows the token may attend to (padded)
    bool sparse = false;      // past the indexer top-k, so the selection bites

    ggml_tensor * kv_idx[MAX_GPUS + 1] = {};    // I64 [1]  destination cache row
    ggml_tensor * kq_mask[MAX_GPUS + 1] = {};   // F16 [n_kv, 1]
    ggml_tensor * lid_mask[MAX_GPUS + 1] = {};  // F16 [n_kv, 1]
};

struct graph_build_result
{
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    graph_inputs inp;
    ggml_tensor * logits = nullptr;
    int64_t nt = 0;
    int64_t n_kv = 0;
    int64_t n_out = 0;
    bool sparse = false;
    bool want_logits = true;
    int slot_id = 0;
    /// Non-empty when this is a batched-decode graph; one entry per token.
    std::vector<bd_token> bd;
    std::vector<trace_entry> traces;

    ~graph_build_result()
    {
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
    }
};

struct glm_model
{
    glm_hparams hp;

    ggml_backend_t backends[MAX_GPUS + 1] = {};
    int n_gpu = 0;
    int n_backends = 0;
    ggml_threadpool_t cpu_threadpool = nullptr;

    ggml_backend_t sched_backends[MAX_GPUS + 1] = {};
    ggml_backend_buffer_type_t sched_bufts[MAX_GPUS + 1] = {};
    int n_sched_backends = 0;

    ggml_context * w_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t w_buf[MAX_GPUS + 1] = {};
    ggml_context * c_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t c_buf[MAX_GPUS + 1] = {};

    std::vector<void *> mmap_addrs;
    std::vector<size_t> mmap_sizes;
    std::vector<ggml_backend_buffer_t> mmap_bufs;
    size_t mmap_weight_bytes = 0;

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output = nullptr;

    std::vector<glm_layer> layers;

    int32_t n_ctx = 0;
    int32_t n_ubatch = 0;

    /// 1 = layer split (each layer on one device). >1 = tensor parallel: every
    /// rank runs every layer with its own slice of the head- and expert-sharded
    /// matrices, and the two row-parallel outputs per layer are summed across
    /// ranks.
    int tp = 1;

    /// Which tensors the tensor-parallel mode actually shards, as a bitmask
    /// (1 = attention heads, 2 = routed experts). Both by default;
    /// TS_GLM_TP_SHARD exists so either half can be turned off to attribute a
    /// regression or measure what each contributes.
    int tp_shard = 3;
    bool tp_heads() const { return (tp_shard & 1) != 0; }
    /// Cleared when the routed experts' rows do not split on a quantization
    /// block boundary; the experts then stay whole on every rank.
    bool tp_moe_rows = true;
    bool tp_experts() const { return (tp_shard & 2) != 0 && tp_moe_rows; }

    /// GPU that hosts rank `r`. One rank per GPU in every real configuration;
    /// the wrap only happens under TS_GLM_TP_OVERSUBSCRIBE.
    int rank_device(int r) const { return n_gpu > 0 ? r % n_gpu : 0; }

    // Per-slot cache contexts/buffers, tagged with the slot they belong to so
    // freeing a sequence releases exactly its own VRAM.
    struct slot_ctx { int slot_id; ggml_context * ctx; ggml_backend_buffer_t buf; };
    std::vector<slot_ctx> slot_ctxs;

    std::map<int, std::unique_ptr<glm_slot>> slots;
    glm_slot * active_slot = nullptr;
    int next_slot_id = 0;

    bool flash_attn = false;
    bool fused_lid = false;      // ggml_lightning_indexer has a kernel here

    // ggml_backend_sched's "a higher-priority backend wants this op" heuristic.
    // It is a win when a host-resident tensor is small enough to stream, and a
    // hard failure when it is not: with the routed experts offloaded, a
    // prefill-sized batch makes the scheduler try to copy 2.9 GiB of experts PER
    // LAYER onto the accelerator, and it sizes one buffer for all of them at
    // once (28 GiB here) — which simply does not fit next to the weights that
    // are already resident. So it is off whenever any layer's experts live on
    // the host; TS_GLM_OP_OFFLOAD=1 forces it back on.
    bool op_offload = true;

    // Orthonormal Walsh-Hadamard rotation applied to the indexer keys and
    // queries. It cancels out of their dot product in exact arithmetic, but the
    // indexer key CACHE is F16: rotating first spreads the outliers so the
    // rounding is even across the 128 dimensions, and llama.cpp does the same,
    // so reproducing it is what keeps long-context top-k selections identical.
    std::vector<float> hadamard_host;
    ggml_tensor * hadamard[MAX_GPUS + 1] = {};


    std::list<std::unique_ptr<graph_build_result>> graph_cache;
    int graph_cache_cap = 8;

    std::vector<float> logits;

    // host scratch, refilled per ubatch
    std::vector<int32_t> h_tokens;
    std::vector<int32_t> h_pos;
    std::vector<int64_t> h_kv_idxs;
    std::vector<uint16_t> h_mask_f16;
    std::vector<float> h_mask_f32;
    std::vector<int32_t> h_out_ids;

    ~glm_model()
    {
        graph_cache.clear();
        slots.clear();
        for (auto & sc : slot_ctxs)
        {
            if (sc.buf) ggml_backend_buffer_free(sc.buf);
            if (sc.ctx) ggml_free(sc.ctx);
        }
        slot_ctxs.clear();
        for (int i = 0; i <= MAX_GPUS; i++)
        {
            if (c_buf[i]) ggml_backend_buffer_free(c_buf[i]);
            if (c_ctx[i]) ggml_free(c_ctx[i]);
            if (w_buf[i]) ggml_backend_buffer_free(w_buf[i]);
            if (w_ctx[i]) ggml_free(w_ctx[i]);
        }
        if (!mmap_addrs.empty())
        {
            for (ggml_backend_buffer_t b : mmap_bufs)
                if (b) ggml_backend_buffer_free(b);
#if !defined(_WIN32)
            for (size_t i = 0; i < mmap_addrs.size(); i++)
                if (mmap_addrs[i]) munmap(mmap_addrs[i], mmap_sizes[i]);
#endif
        }
        for (int i = 0; i < n_backends; i++)
            if (backends[i]) ggml_backend_free(backends[i]);
        if (cpu_threadpool) ggml_threadpool_free(cpu_threadpool);
    }
};

// ---------------------------------------------------------------------------
// GGUF helpers
// ---------------------------------------------------------------------------

static bool kv_u32(gguf_context * g, const char * key, int32_t * out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    switch (gguf_get_kv_type(g, id))
    {
        case GGUF_TYPE_UINT32: *out = (int32_t) gguf_get_val_u32(g, id); return true;
        case GGUF_TYPE_INT32:  *out = gguf_get_val_i32(g, id); return true;
        case GGUF_TYPE_UINT16: *out = (int32_t) gguf_get_val_u16(g, id); return true;
        case GGUF_TYPE_UINT64: *out = (int32_t) gguf_get_val_u64(g, id); return true;
        default: return false;
    }
}

static bool kv_f32(gguf_context * g, const char * key, float * out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return false;
    *out = gguf_get_val_f32(g, id);
    return true;
}

static bool kv_bool(gguf_context * g, const char * key, bool * out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) return false;
    *out = gguf_get_val_bool(g, id);
    return true;
}

static bool kv_arr_u8(gguf_context * g, const char * key, std::vector<uint8_t> & out)
{
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return false;
    const gguf_type et = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    out.resize(n);
    if (et == GGUF_TYPE_UINT32 || et == GGUF_TYPE_INT32)
    {
        const int32_t * d = (const int32_t *) gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; i++) out[i] = d[i] != 0 ? 1 : 0;
        return true;
    }
    if (et == GGUF_TYPE_BOOL)
    {
        const int8_t * d = (const int8_t *) gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; i++) out[i] = d[i] != 0 ? 1 : 0;
        return true;
    }
    return false;
}

static shard_files resolve_shards(const std::string & first, int split_count)
{
    shard_files res;
    if (split_count <= 1) { res.paths.push_back(first); return res; }

    const std::string marker = "-00001-of-";
    size_t pos = first.find(marker);
    if (pos == std::string::npos) { res.paths.push_back(first); return res; }

    for (int i = 1; i <= split_count; i++)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "-%05d-of-", i);
        std::string p = first;
        p.replace(pos, marker.size(), buf);
        res.paths.push_back(p);
    }
    return res;
}

/// One weight tensor and where its bytes come from. Sliced tensors (tensor
/// parallelism) read a sub-range of the file, which is contiguous for ne1/ne2
/// splits and strided for an ne0 split, so the loader is told both.
struct pending_weight
{
    ggml_tensor * t = nullptr;
    int shard = -1;
    size_t file_off = 0;    // absolute offset of the first byte to read
    size_t bytes = 0;       // total bytes of the (possibly sliced) tensor
    size_t src_row = 0;     // 0 = contiguous; otherwise the file stride per row
    size_t row_len = 0;     // bytes per row when src_row != 0
};

struct weight_loader
{
    glm_model & m;
    std::map<std::string, tensor_source> & sources;
    std::vector<pending_weight> pending;

    weight_loader(glm_model & model, std::map<std::string, tensor_source> & src) : m(model), sources(src) {}

    const tensor_source * find(const char * name, bool required)
    {
        auto it = sources.find(name);
        if (it == sources.end())
        {
            if (required) fprintf(stderr, "[glm] missing tensor: %s\n", name);
            return nullptr;
        }
        return &it->second;
    }

    /// The whole tensor, as stored.
    ggml_tensor * full(int device, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        ggml_set_name(t, name);
        pending.push_back({ t, src->shard, src->offset, src->size, 0, 0 });
        return t;
    }

    /// Rows [first, first+count) of dimension `dim` (1 or 2). Both are contiguous
    /// runs in the file, so this is a plain sub-range.
    ggml_tensor * slice_hi(int device, int dim, int64_t first, int64_t count, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        const size_t row = ggml_row_size(src->type, src->ne[0]);
        int64_t ne[4] = { src->ne[0], src->ne[1], src->ne[2], src->ne[3] };
        size_t unit = row;                       // bytes per unit of `dim`
        if (dim == 2) unit = row * src->ne[1];
        ne[dim] = count;

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, ne[0], ne[1], ne[2], ne[3]);
        ggml_format_name(t, "%s.r%d", name, device);
        pending.push_back({ t, src->shard, src->offset + (size_t) first * unit, (size_t) count * unit, 0, 0 });
        return t;
    }

    /// Rows [first, first+count) of dimension 1, for every matrix along dim 2.
    /// Each expert's rows are contiguous but the experts are not adjacent, so
    /// this reads one strided run per expert.
    ggml_tensor * slice_mid(int device, int64_t first, int64_t count, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        const size_t row = ggml_row_size(src->type, src->ne[0]);
        const size_t src_stride = row * (size_t) src->ne[1];    // one whole expert
        const size_t part = row * (size_t) count;               // this rank's strip of it

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, src->ne[0], count, src->ne[2], src->ne[3]);
        ggml_format_name(t, "%s.r%d", name, device);
        pending.push_back({ t, src->shard, src->offset + (size_t) first * row,
                            part * (size_t) (src->ne[2] * src->ne[3]), src_stride, part });
        return t;
    }

    /// Columns [first, first+count) of dimension 0 — the row-parallel split. The
    /// bounds must be block-aligned for the quantization type; they always are
    /// here because the split falls on attention-head boundaries and a head is a
    /// whole number of blocks.
    ggml_tensor * slice_lo(int device, int64_t first, int64_t count, bool required, const char * fmt, ...)
    {
        char name[256];
        va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
        const tensor_source * src = find(name, required);
        if (!src) return nullptr;

        const int64_t blck = ggml_blck_size(src->type);
        if (first % blck != 0 || count % blck != 0)
        {
            fprintf(stderr, "[glm] %s: a row-parallel split at [%" PRId64 ", %" PRId64 ") is not a multiple of the "
                    "%" PRId64 "-element block of %s\n", name, first, first + count, blck, ggml_type_name(src->type));
            return nullptr;
        }

        const size_t full_row = ggml_row_size(src->type, src->ne[0]);
        const size_t part_row = ggml_row_size(src->type, count);
        const size_t off_row = ggml_row_size(src->type, first);
        const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];

        ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src->type, count, src->ne[1], src->ne[2], src->ne[3]);
        ggml_format_name(t, "%s.r%d", name, device);
        pending.push_back({ t, src->shard, src->offset + off_row, (size_t) rows * part_row, full_row, part_row });
        return t;
    }
};

static size_t file_size_of(const char * path)
{
    FILE * f = fopen(path, "rb");
    if (!f) return (size_t) -1;
#if defined(_WIN32)
    _fseeki64(f, 0, SEEK_END);
    long long sz = _ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    off_t sz = ftello(f);
#endif
    fclose(f);
    return sz < 0 ? (size_t) -1 : (size_t) sz;
}

static bool check_shard_complete(const char * path, gguf_context * g, ggml_context * meta)
{
    const size_t fsz = file_size_of(path);
    if (fsz == (size_t) -1) return true;

    const size_t data_off = gguf_get_data_offset(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    size_t needed = data_off;
    const char * last = nullptr;
    for (int64_t ti = 0; ti < n_tensors; ti++)
    {
        const char * name = gguf_get_tensor_name(g, ti);
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt) continue;
        const size_t end = data_off + gguf_get_tensor_offset(g, ti) + ggml_nbytes(mt);
        if (end > needed) { needed = end; last = name; }
    }
    if (needed <= fsz) return true;

    fprintf(stderr, "[glm] %s is incomplete: the file is %zu bytes but its %" PRId64 " tensors need %zu "
                    "(%.2f GiB missing; %s is the last one). Re-download this file.\n",
            path, fsz, n_tensors, needed, (needed - fsz) / (1024.0 * 1024.0 * 1024.0), last ? last : "?");
    return false;
}

static ggml_backend_buffer_t mmap_shard(glm_model & m, const shard_files & shards, int si)
{
#if defined(_WIN32)
    (void) m; (void) shards; (void) si;
    return nullptr;
#else
    if (si < 0 || si >= (int) shards.paths.size()) return nullptr;
    if (m.mmap_bufs.size() < shards.paths.size())
    {
        m.mmap_addrs.resize(shards.paths.size(), nullptr);
        m.mmap_sizes.resize(shards.paths.size(), 0);
        m.mmap_bufs.resize(shards.paths.size(), nullptr);
    }
    if (m.mmap_bufs[si]) return m.mmap_bufs[si];

    const int fd = open(shards.paths[si].c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return nullptr; }
    void * addr = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return nullptr;

    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(addr, (size_t) st.st_size);
    if (!buf) { munmap(addr, (size_t) st.st_size); return nullptr; }
    // WEIGHTS, not the default ANY: ggml_backend_sched only pins a node to the
    // backend that owns an input when that input's buffer is marked as weights
    // (ggml-backend.cpp, "assign nodes that use weights to the backend of the
    // weights"). Left at ANY, an offloaded layer's mul_mat_id is assigned to the
    // accelerator instead, and the scheduler copies the whole 2.9 GiB expert
    // block over the bus once per token — measured 33 s/token against 0.19 s
    // with the flag set.
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.mmap_addrs[si] = addr;
    m.mmap_sizes[si] = (size_t) st.st_size;
    m.mmap_bufs[si] = buf;
    return buf;
#endif
}

struct load_job
{
    ggml_tensor * t;
    int shard;
    size_t file_off;     // where the first byte of this chunk lives in the file
    size_t tensor_off;   // where it lands inside the tensor
    size_t len;          // bytes to move
    size_t src_row = 0;  // 0 = one contiguous read; otherwise the file stride per row
    size_t row_len = 0;  // bytes per row when src_row != 0
};

static bool upload_parallel(const shard_files & shards, std::vector<load_job> & jobs, int n_threads)
{
    std::sort(jobs.begin(), jobs.end(), [](const load_job & a, const load_job & b)
    {
        if (a.shard != b.shard) return a.shard < b.shard;
        return a.file_off < b.file_off;
    });

    if (n_threads > (int) jobs.size()) n_threads = (int) jobs.size();
    if (n_threads < 1) n_threads = 1;

    std::atomic<size_t> cursor(0);
    std::atomic<bool> failed(false);

    auto worker = [&]()
    {
        std::vector<FILE *> files(shards.paths.size(), nullptr);
        std::vector<uint8_t> staging;
        while (!failed.load(std::memory_order_relaxed))
        {
            size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) break;
            const load_job & j = jobs[i];

            FILE *& f = files[j.shard];
            if (!f)
            {
                f = fopen(shards.paths[j.shard].c_str(), "rb");
                if (!f)
                {
                    fprintf(stderr, "[glm] cannot open %s\n", shards.paths[j.shard].c_str());
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            if (staging.size() < j.len) staging.resize(j.len);

            bool ok = true;
            if (j.src_row == 0)
            {
#if defined(_WIN32)
                _fseeki64(f, (long long) j.file_off, SEEK_SET);
#else
                fseeko(f, (off_t) j.file_off, SEEK_SET);
#endif
                ok = fread(staging.data(), 1, j.len, f) == j.len;
            }
            else
            {
                // Row-parallel slice: gather one strip out of every source row.
                const size_t rows = j.len / j.row_len;
                for (size_t r = 0; r < rows && ok; r++)
                {
                    const size_t off = j.file_off + r * j.src_row;
#if defined(_WIN32)
                    _fseeki64(f, (long long) off, SEEK_SET);
#else
                    fseeko(f, (off_t) off, SEEK_SET);
#endif
                    ok = fread(staging.data() + r * j.row_len, 1, j.row_len, f) == j.row_len;
                }
            }
            if (!ok)
            {
                fprintf(stderr, "[glm] short read for %s: %zu bytes at offset %zu of %s\n",
                        j.t->name, j.len, j.file_off, shards.paths[j.shard].c_str());
                failed.store(true, std::memory_order_relaxed);
                break;
            }
            ggml_backend_tensor_set(j.t, staging.data(), j.tensor_off, j.len);
        }
        for (FILE * f : files) if (f) fclose(f);
    };

    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (int i = 0; i < n_threads; i++) pool.emplace_back(worker);
    for (auto & th : pool) th.join();
    return !failed.load();
}

static bool ieq(const char * a, const char * b)
{
    if (!a || !b) return false;
    for (; *a && *b; ++a, ++b)
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) return false;
    return *a == *b;
}

// Backend registry names are compared case-insensitively: ggml spells them
// "CUDA" / "Metal" / "Vulkan", but which capitalisation a given build uses is
// not something callers should have to know.
static bool backend_matches(const char * reg_name, const char * want)
{
    if (!want || !*want) return true;
    if (!reg_name) return false;
    if (ieq(want, "CUDA"))
        return ieq(reg_name, "CUDA") || ieq(reg_name, "ROCm") || ieq(reg_name, "MUSA");
    // ggml's Metal registry calls itself "MTL"; accept the name users type too.
    if (ieq(want, "Metal"))
        return ieq(reg_name, "Metal") || ieq(reg_name, "MTL");
    return ieq(reg_name, want);
}

// Memory this process may actually keep resident (cgroup limit under a
// container, MemTotal otherwise). 0 = unknown.
static size_t host_mem_allowance()
{
    size_t limit = 0;
#ifdef __linux__
    if (FILE * f = fopen("/sys/fs/cgroup/memory.max", "r"))
    {
        char v[64] = { 0 };
        if (fscanf(f, "%63s", v) == 1 && strcmp(v, "max") != 0) limit = (size_t) atoll(v);
        fclose(f);
    }
    if (limit == 0)
    {
        if (FILE * f = fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r"))
        {
            long long v = 0;
            if (fscanf(f, "%lld", &v) == 1 && v > 0 && v < (1ll << 62)) limit = (size_t) v;
            fclose(f);
        }
    }
    if (limit == 0)
    {
        if (FILE * f = fopen("/proc/meminfo", "r"))
        {
            char line[256];
            while (fgets(line, sizeof(line), f))
            {
                long long kb = 0;
                if (sscanf(line, "MemTotal: %lld kB", &kb) == 1) { limit = (size_t) kb * 1024; break; }
            }
            fclose(f);
        }
    }
#endif
    return limit;
}

// llama.cpp's published GLM-5.2 indexer pattern: layers 0,1,2 are full, then
// every fourth layer from 6 on. GLM-5.0/5.1 (shorter training context) made
// every layer full. Overridden by the metadata array when the file has one.
static void resolve_indexer_types(glm_hparams & hp, gguf_context * g)
{
    hp.indexer_full.assign((size_t) hp.n_layer, 0);
    const bool pre_5_2 = hp.n_ctx_train > 0 && hp.n_ctx_train < 1048576;
    for (int il = 0; il < hp.n_layer; il++)
        hp.indexer_full[il] = pre_5_2 ? 1 : (il < 3 || ((il - 2) % 4 == 0)) ? 1 : 0;

    std::vector<uint8_t> explicit_types;
    if (kv_arr_u8(g, "glm-dsa.attention.indexer.types", explicit_types))
        for (int il = 0; il < hp.n_layer && il < (int) explicit_types.size(); il++)
            hp.indexer_full[il] = explicit_types[il];
}

// Orthonormal Walsh-Hadamard matrix (H == H^T, H*H == I), the Sylvester
// construction ggml_gen_hadamard uses.
static void gen_hadamard(std::vector<float> & out, int n)
{
    out.assign((size_t) n * n, 0.0f);
    out[0] = 1.0f / sqrtf((float) n);
    for (int s = 1; s < n; s *= 2)
    {
        for (int i = 0; i < s; i++)
        {
            for (int j = 0; j < s; j++)
            {
                const float v = out[(size_t) i * n + j];
                out[(size_t) (i + s) * n + (j    )] =  v;
                out[(size_t) (i    ) * n + (j + s)] =  v;
                out[(size_t) (i + s) * n + (j + s)] = -v;
            }
        }
    }
}

static bool is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

static glm_slot * slot_alloc(glm_model & m)
{
    auto slot = std::unique_ptr<glm_slot>(new glm_slot());
    slot->id = m.next_slot_id++;
    for (int r = 0; r < m.tp; r++)
    {
        slot->kv_k[r].assign((size_t) m.hp.n_layer, nullptr);
        slot->idx_k[r].assign((size_t) m.hp.n_layer, nullptr);
    }

    // Cache tensors live on the device that reads them, so attention never
    // crosses the split.
    std::vector<ggml_context *> ctxs((size_t) m.n_gpu + 1, nullptr);
    for (int d = 0; d <= m.n_gpu; d++)
    {
        ggml_init_params p = { (size_t) (4 * m.tp * m.hp.n_layer + 16) * ggml_tensor_overhead(), nullptr, true };
        ctxs[d] = ggml_init(p);
    }

    for (int il = 0; il < m.hp.n_layer; il++)
    {
        for (int r = 0; r < m.tp; r++)
        {
            const int d = m.tp > 1 ? m.rank_device(r) : m.layers[il].device;
            slot->kv_k[r][il] = ggml_new_tensor_2d(ctxs[d], GGML_TYPE_F16, m.hp.n_kv_row, m.n_ctx);
            ggml_format_name(slot->kv_k[r][il], "slot%d_kv.%d.%d", slot->id, r, il);
            if (m.layers[il].indexer_full)
            {
                slot->idx_k[r][il] = ggml_new_tensor_2d(ctxs[d], GGML_TYPE_F16, m.hp.indexer_head_size, m.n_ctx);
                ggml_format_name(slot->idx_k[r][il], "slot%d_idx.%d.%d", slot->id, r, il);
            }
        }
    }

    // One buffer per device for the whole slot; freed with the slot.
    std::vector<ggml_backend_buffer_t> bufs((size_t) m.n_gpu + 1, nullptr);
    bool ok = true;
    for (int d = 0; d <= m.n_gpu && ok; d++)
    {
        if (ggml_get_first_tensor(ctxs[d]) == nullptr) continue;
        bufs[d] = ggml_backend_alloc_ctx_tensors(ctxs[d], m.backends[d]);
        if (!bufs[d]) ok = false;
        else ggml_backend_buffer_clear(bufs[d], 0);
    }
    if (!ok)
    {
        for (auto b : bufs) if (b) ggml_backend_buffer_free(b);
        for (auto c : ctxs) if (c) ggml_free(c);
        fprintf(stderr, "[glm] KV cache allocation failed for a new sequence slot\n");
        return nullptr;
    }

    // Keep the contexts and buffers alive for the slot's lifetime by parking
    // them in the model's cache contexts list.
    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (!bufs[d]) { if (ctxs[d]) ggml_free(ctxs[d]); continue; }
        // chain onto the model-level trackers
        m.slot_ctxs.push_back({ slot->id, ctxs[d], bufs[d] });
    }

    glm_slot * raw = slot.get();
    m.slots[slot->id] = std::move(slot);
    return raw;
}

static void slot_free(glm_model & m, int slot_id)
{
    // Graphs bake this slot's cache addresses into their nodes; drop them first.
    // A batched-decode graph belongs to no single slot but reads N of them, and
    // slot ids are reused, so a stale one could be matched by a later batch and
    // run against caches that no longer exist.
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); )
    {
        bool refs = (*it)->slot_id == slot_id;
        for (const bd_token & B : (*it)->bd)
            if (B.slot_id == slot_id) refs = true;
        it = refs ? m.graph_cache.erase(it) : std::next(it);
    }

    m.slots.erase(slot_id);
    for (auto it = m.slot_ctxs.begin(); it != m.slot_ctxs.end(); )
    {
        if (it->slot_id != slot_id) { ++it; continue; }
        if (it->buf) ggml_backend_buffer_free(it->buf);
        if (it->ctx) ggml_free(it->ctx);
        it = m.slot_ctxs.erase(it);
    }
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

/// `ctx_is_hard_limit` distinguishes "the user asked for this context" from
/// "this is what the GGUF advertises". The first is honoured or refused; the
/// second is a ceiling the loader may cap to whatever the VRAM actually holds,
/// because a 1M-token advertisement is not a promise that 1M tokens of KV fit
/// beside the weights.
static glm_model * glm_load(const char * gguf_path, int n_gpu_req, int n_ctx, int n_ubatch,
                            int n_threads, int n_cpu_moe_req, const char * backend_name, int tp_req,
                            bool ctx_is_hard_limit)
{
    auto t_start = std::chrono::steady_clock::now();
    std::unique_ptr<glm_model> m(new glm_model());

    // --- backends ---------------------------------------------------------
    // "CPU" is an explicit request for a host-only run (--backend ggml_cpu):
    // without it the device scan would happily pick up Metal or CUDA, which is
    // never what that flag means.
    const bool cpu_only = backend_name && strcmp(backend_name, "CPU") == 0;
    int n_gpu = 0;
    for (int pass = 0; pass < 2 && n_gpu == 0 && !cpu_only; pass++)
    {
        const char * want = pass == 0 ? backend_name : nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count() && n_gpu < MAX_GPUS; i++)
        {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (!backend_matches(reg ? ggml_backend_reg_name(reg) : nullptr, want)) continue;
            if (n_gpu_req > 0 && n_gpu >= n_gpu_req) break;
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (!be) continue;
            m->backends[n_gpu++] = be;
        }
        if (pass == 0 && n_gpu == 0 && backend_name && *backend_name)
            fprintf(stderr, "[glm] no %s GPU devices found; falling back to any available GPU backend\n", backend_name);
    }
    m->n_gpu = n_gpu;

    // Tensor parallelism uses exactly `tp` devices, each running every layer on
    // its own slice of the head- and expert-sharded matrices. Anything else is
    // the layer split, where each layer lives on one device.
    if (tp_req > 1)
    {
        if (tp_req > n_gpu)
        {
            // More ranks than GPUs is not a production configuration — it buys
            // no parallelism — but it is how the sharding math gets tested on a
            // single-GPU machine, so allow it explicitly rather than only on a
            // multi-GPU host.
            if (getenv("TS_GLM_TP_OVERSUBSCRIBE") == nullptr)
            {
                fprintf(stderr, "[glm] --tp %d needs %d GPUs; only %d are visible "
                        "(TS_GLM_TP_OVERSUBSCRIBE=1 packs several ranks onto one GPU for testing)\n",
                        tp_req, tp_req, n_gpu);
                return nullptr;
            }
            fprintf(stderr, "[glm] --tp %d on %d GPU(s): ranks share devices (testing only, no speedup)\n",
                    tp_req, n_gpu);
        }
        m->tp = tp_req;
        if (const char * e = getenv("TS_GLM_TP_SHARD")) m->tp_shard = atoi(e) & 3;
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    {
        // With --n-cpu-moe (or a GPU-less run) the routed-expert matmuls land
        // here on every token; that is memory-bandwidth bound and wants every
        // core the process is actually allowed to use.
        int cpu_threads = n_threads > 0 ? n_threads : 16;
        if (n_cpu_moe_req != 0 || n_gpu == 0) cpu_threads = tsg::available_cpu_parallelism();
        if (const char * e = getenv("TS_CPU_MOE_THREADS")) { int v = atoi(e); if (v > 0) cpu_threads = v; }
        ggml_backend_cpu_set_n_threads(cpu, cpu_threads);

        // A persistent pool: one token is one CPU graph split per offloaded
        // layer, and spawning/joining a disposable pool per split costs more
        // than the matmuls it parallelises.
        ggml_threadpool_params tpp = ggml_threadpool_params_default(cpu_threads);
        m->cpu_threadpool = ggml_threadpool_new(&tpp);
        if (m->cpu_threadpool) ggml_backend_cpu_set_threadpool(cpu, m->cpu_threadpool);
    }
    m->backends[n_gpu] = cpu;
    m->n_backends = n_gpu + 1;

    {
        int nb = 0;
        for (int d = 0; d < n_gpu; d++) m->sched_backends[nb++] = m->backends[d];
        m->sched_backends[nb++] = cpu;
        for (int i = 0; i < nb; i++)
            m->sched_bufts[i] = ggml_backend_get_default_buffer_type(m->sched_backends[i]);
        m->n_sched_backends = nb;

        if (const char * e = getenv("TS_GLM_GRAPH_CACHE")) { int v = atoi(e); if (v > 0) m->graph_cache_cap = v; }
    }

    // --- metadata ---------------------------------------------------------
    ggml_context * meta0 = nullptr;
    gguf_init_params ip = { true, &meta0 };
    gguf_context * g0 = gguf_init_from_file(gguf_path, ip);
    if (!g0) { fprintf(stderr, "[glm] failed to open %s\n", gguf_path); return nullptr; }

    int32_t split_count = 1;
    kv_u32(g0, "split.count", &split_count);
    shard_files shards = resolve_shards(gguf_path, split_count);

    glm_hparams & hp = m->hp;
    bool ok = true;
    ok &= kv_u32(g0, "glm-dsa.block_count", &hp.n_layer_all);
    ok &= kv_u32(g0, "glm-dsa.embedding_length", &hp.n_embd);
    ok &= kv_u32(g0, "glm-dsa.attention.head_count", &hp.n_head);
    ok &= kv_u32(g0, "glm-dsa.rope.dimension_count", &hp.n_rot);
    ok &= kv_u32(g0, "glm-dsa.attention.q_lora_rank", &hp.q_lora_rank);
    ok &= kv_u32(g0, "glm-dsa.attention.kv_lora_rank", &hp.kv_lora_rank);
    ok &= kv_f32(g0, "glm-dsa.attention.layer_norm_rms_epsilon", &hp.rms_eps);
    ok &= kv_u32(g0, "glm-dsa.expert_count", &hp.n_expert);
    ok &= kv_u32(g0, "glm-dsa.expert_used_count", &hp.n_expert_used);
    ok &= kv_u32(g0, "glm-dsa.expert_feed_forward_length", &hp.n_ff_exp);
    ok &= kv_u32(g0, "glm-dsa.attention.indexer.head_count", &hp.indexer_n_head);
    ok &= kv_u32(g0, "glm-dsa.attention.indexer.key_length", &hp.indexer_head_size);
    ok &= kv_u32(g0, "glm-dsa.attention.indexer.top_k", &hp.indexer_top_k);

    kv_u32(g0, "glm-dsa.feed_forward_length", &hp.n_ff);
    kv_u32(g0, "glm-dsa.leading_dense_block_count", &hp.n_dense_lead);
    kv_u32(g0, "glm-dsa.nextn_predict_layers", &hp.n_layer_nextn);
    kv_u32(g0, "glm-dsa.expert_shared_count", &hp.n_expert_shared);
    kv_f32(g0, "glm-dsa.expert_weights_scale", &hp.expert_weights_scale);
    kv_bool(g0, "glm-dsa.expert_weights_norm", &hp.expert_weights_norm);
    kv_u32(g0, "glm-dsa.expert_gating_func", &hp.expert_gating_func);
    kv_f32(g0, "glm-dsa.rope.freq_base", &hp.rope_freq_base);
    kv_u32(g0, "glm-dsa.context_length", &hp.n_ctx_train);

    int32_t key_len = 0, val_len = 0;
    kv_u32(g0, "glm-dsa.attention.key_length", &key_len);
    kv_u32(g0, "glm-dsa.attention.value_length", &val_len);
    hp.n_embd_head_k = key_len;
    hp.n_embd_head_v = val_len;
    kv_u32(g0, "glm-dsa.attention.key_length_mla", &hp.n_embd_head_k);
    kv_u32(g0, "glm-dsa.attention.value_length_mla", &hp.n_embd_head_v);

    int32_t expert_groups = 1, expert_groups_used = 1;
    kv_u32(g0, "glm-dsa.expert_group_count", &expert_groups);
    kv_u32(g0, "glm-dsa.expert_group_used_count", &expert_groups_used);

    hp.n_layer = hp.n_layer_all - hp.n_layer_nextn;
    hp.n_nope = hp.n_embd_head_k - hp.n_rot;
    hp.n_kv_row = hp.kv_lora_rank + hp.n_rot;
    if (hp.expert_gating_func == 0) hp.expert_gating_func = 2;

    resolve_indexer_types(hp, g0);
    gguf_free(g0);
    if (meta0) { ggml_free(meta0); meta0 = nullptr; }

    if (!ok || hp.n_layer <= 0 || hp.n_nope <= 0 || hp.n_head <= 0)
    {
        fprintf(stderr, "[glm] missing or invalid glm-dsa metadata\n");
        return nullptr;
    }
    if (expert_groups > 1 && expert_groups_used != expert_groups)
    {
        fprintf(stderr, "[glm] %d expert groups (top-%d) are not supported\n", expert_groups, expert_groups_used);
        return nullptr;
    }
    if (hp.indexer_full.empty() || hp.indexer_full[0] == 0)
    {
        fprintf(stderr, "[glm] layer 0 must carry a full DSA indexer\n");
        return nullptr;
    }

    // --- tensor sources across shards -------------------------------------
    std::map<std::string, tensor_source> sources;
    std::vector<size_t> layer_bytes((size_t) hp.n_layer, 0);
    std::vector<size_t> layer_exps_bytes((size_t) hp.n_layer, 0);
    size_t root_bytes = 0;

    for (size_t si = 0; si < shards.paths.size(); si++)
    {
        ggml_context * meta = nullptr;
        gguf_init_params sp = { true, &meta };
        gguf_context * g = gguf_init_from_file(shards.paths[si].c_str(), sp);
        if (!g) { fprintf(stderr, "[glm] failed to open shard %s\n", shards.paths[si].c_str()); return nullptr; }
        if (!check_shard_complete(shards.paths[si].c_str(), g, meta)) { gguf_free(g); ggml_free(meta); return nullptr; }

        const size_t data_off = gguf_get_data_offset(g);
        const int64_t n_tensors = gguf_get_n_tensors(g);
        for (int64_t ti = 0; ti < n_tensors; ti++)
        {
            const char * name = gguf_get_tensor_name(g, ti);
            ggml_tensor * mt = ggml_get_tensor(meta, name);
            if (!mt) continue;
            tensor_source src;
            src.shard = (int) si;
            src.offset = data_off + gguf_get_tensor_offset(g, ti);
            src.size = ggml_nbytes(mt);
            src.type = mt->type;
            for (int d = 0; d < 4; d++) src.ne[d] = mt->ne[d];
            sources[name] = src;

            int bid = -1;
            if (sscanf(name, "blk.%d.", &bid) == 1 && bid >= 0 && bid < hp.n_layer)
            {
                layer_bytes[bid] += src.size;
                if (strstr(name, "_exps.") != nullptr) layer_exps_bytes[bid] += src.size;
            }
            else if (bid < 0)
            {
                root_bytes += src.size;   // token_embd / output / output_norm
            }
            // NextN block tensors are not loaded by the trunk graph.
        }
        gguf_free(g);
        ggml_free(meta);
    }

    size_t embd_bytes = 0;
    {
        auto it = sources.find("token_embd.weight");
        if (it == sources.end()) { fprintf(stderr, "[glm] token_embd.weight missing\n"); return nullptr; }
        hp.n_vocab = (int32_t) it->second.ne[1];
        embd_bytes = it->second.size;
    }
    const size_t head_bytes = root_bytes > embd_bytes ? root_bytes - embd_bytes : 0;

    m->n_ctx = n_ctx > 0 ? n_ctx : 8192;
    // What the caller asked for, kept so a later measurement can hand slack back
    // without ever exceeding the request.
    const int ctx_requested = m->n_ctx;
    m->n_ubatch = n_ubatch > 0 ? n_ubatch : 512;
    m->layers.resize((size_t) hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++)
    {
        m->layers[il].indexer_full = hp.indexer_full[il] != 0;
        m->layers[il].is_moe = il >= hp.n_dense_lead;
    }

    // What one layer's caches cost the device that hosts it. Priced into the
    // split so a device packed to the last byte of weights does not fail at
    // slot allocation instead of at load.
    auto layer_cache_bytes = [&](int il) -> size_t
    {
        size_t b = (size_t) hp.n_kv_row * m->n_ctx * 2;
        if (hp.indexer_full[il]) b += (size_t) hp.indexer_head_size * m->n_ctx * 2;
        return b + 4 * 256;
    };

    // --- per-device budget + layer split ----------------------------------
    std::vector<size_t> dev_budget((size_t) std::max(n_gpu, 1), 0);
    std::vector<size_t> dev_free((size_t) std::max(n_gpu, 1), 0);
    {
        size_t reserve_mb = 3072;
        if (const char * e = getenv("TS_GLM_VRAM_RESERVE_MB")) { long v = atol(e); if (v >= 0) reserve_mb = (size_t) v; }
        const size_t reserve = reserve_mb * 1024 * 1024;
        for (int d = 0; d < n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            dev_free[d] = free_b;
            dev_budget[d] = free_b > reserve ? free_b - reserve : 0;
        }
    }

    size_t total_bytes = root_bytes;
    for (auto b : layer_bytes) total_bytes += b;

    // KV bytes one token costs, summed over every layer. GLM-5.2 advertises a
    // 1M context, which at 78 layers is ~93 GiB of cache — a whole card's worth
    // on top of the weights — so the advertised number is a ceiling to fit
    // under, not a promise.
    size_t kv_bytes_per_token = 0;
    for (int il = 0; il < hp.n_layer; il++)
    {
        kv_bytes_per_token += (size_t) hp.n_kv_row * 2;
        if (hp.indexer_full[il]) kv_bytes_per_token += (size_t) hp.indexer_head_size * 2;
    }

    // What a graph needs on top of the caches. Two things scale with the
    // context: the DSA top-k masks, which are [n_kv, n_ubatch] F16 and live for
    // the whole graph on every full-indexer layer, and nothing else. The rest is
    // fixed per ubatch and dominated by the LM head's [n_vocab, n_ubatch] output.
    // Counted twice over because the graph cache keeps several built graphs
    // alive, each holding its own allocation.
    const int n_full_indexer = (int) std::count(hp.indexer_full.begin(), hp.indexer_full.end(), (uint8_t) 1);
    const size_t graph_bytes_per_token = 2 * (size_t) n_full_indexer * (size_t) m->n_ubatch * 2;
    const size_t graph_bytes_fixed =
        2 * (size_t) m->n_ubatch * ((size_t) hp.n_vocab + 16 * (size_t) hp.n_embd) * 4;

    /// Largest context whose caches AND graphs still fit in `avail`, in whole
    /// 256-token steps because that is the granularity the graphs are keyed on.
    auto ctx_that_fits = [&](size_t avail) -> int
    {
        const size_t per_token = kv_bytes_per_token + graph_bytes_per_token;
        if (per_token == 0 || avail <= graph_bytes_fixed) return 0;
        size_t tokens = (avail - graph_bytes_fixed) / per_token;
        tokens -= tokens % 256;
        if (tokens > (size_t) ctx_requested) tokens = (size_t) ctx_requested;
        return (int) tokens;
    };

    /// Everything a rank must hold at `n_ctx` besides its weights.
    auto runtime_bytes = [&](int n_ctx_want) -> size_t
    {
        return (size_t) n_ctx_want * (kv_bytes_per_token + graph_bytes_per_token) + graph_bytes_fixed;
    };

    // The pre-load estimate only has to be conservative enough to refuse the
    // hopeless cases; when the devices can be asked directly after the weights
    // land, that measurement decides the context and the estimate stays quiet.
    const bool remeasure_ctx = !ctx_is_hard_limit && n_gpu > 0;

    int n_cpu_moe = 0;
    if (m->tp > 1)
    {
        // Every rank runs every layer, so the budget is per rank: the routed
        // experts are the only weights that shrink with `tp`, and the MLA and
        // indexer caches do not shrink at all — they are rank-independent, so
        // every rank keeps its own full-length copy. Without this the load only
        // discovered it did not fit when the first sequence slot failed to
        // allocate, which is minutes of weight reads after the point of no
        // return.
        auto rank_weight_bytes = [&](int n_cpu) -> size_t
        {
            size_t b = embd_bytes + head_bytes;              // replicated on rank 0's device
            for (int il = 0; il < hp.n_layer; il++)
            {
                const size_t e = layer_exps_bytes[il];
                b += layer_bytes[il] - e;                    // dense half: replicated (upper bound)
                if (il >= n_cpu) b += e / (size_t) m->tp;    // this rank's expert rows
            }
            return b;
        };
        // Only meaningful with one rank per GPU: a host-only run has no device
        // budget to measure, and TS_GLM_TP_OVERSUBSCRIBE deliberately stacks
        // several ranks on one card, where a per-rank budget says nothing.
        const bool budgeted = n_gpu > 0 && m->tp <= n_gpu;

        const size_t kv_bytes = runtime_bytes(m->n_ctx);

        size_t budget = 0;
        if (budgeted)
        {
            budget = SIZE_MAX;
            for (int r = 0; r < m->tp; r++) budget = std::min(budget, dev_budget[(size_t) m->rank_device(r)]);
        }

        int need_cpu_moe = 0;
        if (budgeted)
            while (need_cpu_moe <= hp.n_layer && rank_weight_bytes(need_cpu_moe) + kv_bytes > budget) need_cpu_moe++;

        n_cpu_moe = n_cpu_moe_req < 0 ? std::min(need_cpu_moe, hp.n_layer)
                                      : std::min(std::max(n_cpu_moe_req, 0), hp.n_layer);

        // Experts can only free so much; past that the context is what does not
        // fit, and shrinking it is the only remedy that keeps the model resident.
        const size_t w = rank_weight_bytes(n_cpu_moe);
        if (budgeted && w + kv_bytes > budget)
        {
            const int fit = w < budget ? ctx_that_fits(budget - w) : 0;
            if (fit < 256 || ctx_is_hard_limit)
            {
                fprintf(stderr,
                        "[glm] not enough VRAM for --tp %d: %.1f GiB per rank of weights plus %.1f GiB of KV and "
                        "graphs for an %d-token context, against %.1f GiB usable on the smallest rank. Lower "
                        "MAX_CONTEXT (%d tokens would fit) or add --n-cpu-moe N.\n",
                        m->tp, w / 1073741824.0, kv_bytes / 1073741824.0, m->n_ctx,
                        budget / 1073741824.0, fit);
                return nullptr;
            }
            if (!remeasure_ctx)
                fprintf(stderr, "[glm] context capped to %d tokens (the GGUF advertises %d): %.1f GiB per rank of "
                        "weights leaves %.1f GiB for the caches and graphs. Set MAX_CONTEXT to ask for a "
                        "different one.\n",
                        fit, m->n_ctx, w / 1073741824.0, (budget - w) / 1073741824.0);
            m->n_ctx = fit;
        }

        for (int il = 0; il < n_cpu_moe && il < hp.n_layer; il++) m->layers[il].cpu_moe = true;
        if (n_cpu_moe > 0)
            fprintf(stderr, "[glm] MoE CPU offload: this rank's experts for layers 0..%d stay in system RAM%s\n",
                    n_cpu_moe - 1, n_cpu_moe_req < 0 ? " (auto: they do not fit alongside the caches otherwise)" : "");
    }
    else if (n_gpu == 0)
    {
        // Pure CPU run: everything on the host backend.
        for (int il = 0; il < hp.n_layer; il++) { m->layers[il].device = 0; m->layers[il].cpu_moe = true; }
        n_cpu_moe = hp.n_layer;
    }
    else
    {
        std::vector<size_t> fixed_bytes((size_t) n_gpu, 0);
        fixed_bytes[0] += embd_bytes;
        fixed_bytes[(size_t) n_gpu - 1] += head_bytes;

        auto layer_cost = [&](int il, int n_cpu) -> size_t
        {
            size_t w = layer_bytes[il];
            if (il < n_cpu) w -= layer_exps_bytes[il];
            return w + layer_cache_bytes(il);
        };

        // Contiguous runs, in pipeline order: fill each device to `frac` of its
        // budget and report whether every layer landed.
        auto pack = [&](double frac, int n_cpu, std::vector<int> * out) -> bool
        {
            int dev = 0;
            size_t used = fixed_bytes[0];
            for (int il = 0; il < hp.n_layer; il++)
            {
                const size_t cost = layer_cost(il, n_cpu);
                while (used + cost > (size_t) (dev_budget[dev] * frac))
                {
                    if (dev + 1 >= n_gpu) return false;
                    used = fixed_bytes[++dev];
                }
                used += cost;
                if (out) (*out)[il] = dev;
            }
            return true;
        };

        // An advertised context that does not fit is capped, not refused — the
        // same ceiling rule the tensor-parallel path uses. Only when MAX_CONTEXT
        // named the number is it treated as a requirement.
        if (!ctx_is_hard_limit)
        {
            const int want_cpu = n_cpu_moe_req < 0 ? 0 : std::min(n_cpu_moe_req, hp.n_layer);
            if (!pack(1.0, want_cpu, nullptr))
            {
                size_t weights = root_bytes;
                for (int il = 0; il < hp.n_layer; il++)
                {
                    weights += layer_bytes[il];
                    if (il < want_cpu) weights -= layer_exps_bytes[il];
                }
                size_t budget_total = 0;
                for (int d = 0; d < n_gpu; d++) budget_total += dev_budget[d];
                const int fit = weights < budget_total ? ctx_that_fits(budget_total - weights) : 0;
                if (fit >= 256 && fit < m->n_ctx)
                {
                    if (!remeasure_ctx)
                        fprintf(stderr, "[glm] context capped to %d tokens (the GGUF advertises %d) so the weights "
                                "and their caches fit the %d visible GPU(s). Set MAX_CONTEXT to ask for a "
                                "different one.\n", fit, m->n_ctx, n_gpu);
                    m->n_ctx = fit;
                }
            }
        }

        int need_cpu_moe = 0;
        while (need_cpu_moe <= hp.n_layer && !pack(1.0, need_cpu_moe, nullptr)) need_cpu_moe++;

        if (n_cpu_moe_req < 0)
        {
            n_cpu_moe = std::min(need_cpu_moe, hp.n_layer);
        }
        else
        {
            n_cpu_moe = std::min(n_cpu_moe_req, hp.n_layer);
            if (n_cpu_moe < need_cpu_moe && need_cpu_moe <= hp.n_layer)
            {
                size_t free_total = 0;
                for (int d = 0; d < n_gpu; d++) free_total += dev_free[d];
                size_t would_free = 0;
                for (int il = 0; il < need_cpu_moe && il < hp.n_layer; il++) would_free += layer_exps_bytes[il];
                fprintf(stderr,
                        "[glm] not enough VRAM: %.1f GiB of weights plus this context's KV caches against %.1f GiB "
                        "free across %d device(s). Re-run with --n-cpu-moe %d (moves the routed experts of the "
                        "first %d layer(s), %.1f GiB, to system RAM) or --cpu-moe to offload every layer.\n",
                        total_bytes / 1073741824.0, free_total / 1073741824.0, n_gpu,
                        need_cpu_moe, need_cpu_moe, would_free / 1073741824.0);
                return nullptr;
            }
        }

        // Balance the largest fraction-of-budget used, then fall back to a
        // proportional split if even a perfect pack cannot fit.
        std::vector<int> assign((size_t) hp.n_layer, 0);
        double lo = 0.0, hi = 1.0;
        if (pack(1.0, n_cpu_moe, &assign))
        {
            for (int it = 0; it < 40; it++)
            {
                const double mid = 0.5 * (lo + hi);
                std::vector<int> tmp((size_t) hp.n_layer, 0);
                if (pack(mid, n_cpu_moe, &tmp)) { hi = mid; assign = tmp; }
                else lo = mid;
            }
        }
        else
        {
            size_t acc = 0;
            for (int il = 0; il < hp.n_layer; il++)
            {
                int dev = (int) ((acc * n_gpu) / (total_bytes + 1));
                if (dev >= n_gpu) dev = n_gpu - 1;
                assign[il] = dev;
                acc += layer_bytes[il];
            }
        }
        for (int il = 0; il < hp.n_layer; il++) m->layers[il].device = assign[il];
        for (int il = 0; il < n_cpu_moe && il < hp.n_layer; il++) m->layers[il].cpu_moe = true;

        if (n_cpu_moe > 0)
        {
            size_t host_bytes = 0;
            for (int il = 0; il < n_cpu_moe; il++) host_bytes += layer_exps_bytes[il];
            fprintf(stderr, "[glm] MoE CPU offload: routed experts of layers 0..%d (%.1f GiB) stay in system RAM "
                    "and run on the host%s\n", n_cpu_moe - 1, host_bytes / 1073741824.0,
                    n_cpu_moe_req < 0 ? " (auto: the model does not fit the visible VRAM otherwise)" : "");
        }
    }

    // --- weight tensors ---------------------------------------------------
    for (int d = 0; d <= n_gpu; d++)
    {
        // ~28 tensors per layer per rank that lands on this device. Sized for the
        // worst case (every rank on one device, which TS_GLM_TP_OVERSUBSCRIBE
        // allows) rather than the expected one-rank-per-GPU.
        const size_t per_dev = (size_t) (32 * hp.n_layer * std::max(1, m->tp) + 64);
        ggml_init_params wp = { per_dev * ggml_tensor_overhead(), nullptr, true };
        m->w_ctx[d] = ggml_init(wp);
        ggml_init_params cp = { (size_t) 64 * ggml_tensor_overhead(), nullptr, true };
        m->c_ctx[d] = ggml_init(cp);
    }

    // Under tensor parallelism the embedding table and the LM head are not
    // sharded; they live on rank 0 and the scheduler copies the (tiny) rows.
    const int dev_first = m->tp > 1 ? 0 : m->layers[0].device;
    const int dev_last = m->tp > 1 ? 0 : m->layers[(size_t) hp.n_layer - 1].device;

    weight_loader WL(*m, sources);

    m->tok_embd = WL.full(dev_first, true, "token_embd.weight");
    m->output_norm = WL.full(dev_last, true, "output_norm.weight");
    m->output = WL.full(dev_last, false, "output.weight");
    if (!m->output)   // tied embeddings
        m->output = WL.full(dev_last, true, "token_embd.weight");
    if (!m->tok_embd || !m->output_norm || !m->output) return nullptr;

    const int n_rank = m->tp;
    // Splitting the routed experts row-wise cuts every down-projection ROW, so a
    // rank's strip has to start and end on a quantization block boundary: the
    // split is counted in blocks, not elements. A model whose expert hidden size
    // is not a whole number of blocks, or one with fewer blocks than ranks, keeps
    // its experts whole and splits only the attention heads — slower, still exact.
    int moe_ff_blck = 1;
    {
        bool & ok = m->tp_moe_rows;
        for (int il = 0; il < hp.n_layer && n_rank > 1 && m->tp_experts(); il++)
        {
            char nm[256];
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
            auto it = sources.find(nm);
            if (it == sources.end()) continue;
            const int b = (int) ggml_blck_size(it->second.type);
            moe_ff_blck = std::max(moe_ff_blck, b);
            if (hp.n_ff_exp % moe_ff_blck != 0 || hp.n_ff_exp / moe_ff_blck < n_rank)
            {
                ok = false;
                fprintf(stderr, "[glm] %d expert rows are not %d whole %d-element %s blocks: keeping the experts "
                        "whole and splitting only the attention heads\n",
                        hp.n_ff_exp, n_rank, moe_ff_blck, ggml_type_name(it->second.type));
            }
        }
    }

    for (int il = 0; il < hp.n_layer; il++)
    {
        glm_layer & L = m->layers[il];

        for (int r = 0; r <= n_rank; r++)
        {
            const bool sh = n_rank > 1 && m->tp_heads();
            const bool se = n_rank > 1 && m->tp_experts();
            L.head_first[r] = sh ? shard_first(hp.n_head, n_rank, r) : (r == 0 ? 0 : hp.n_head);
            L.moe_ff_first[r] = se ? shard_first_rot(hp.n_ff_exp / moe_ff_blck, n_rank, r, il) * moe_ff_blck
                                   : (r == 0 ? 0 : hp.n_ff_exp);
        }

        for (int r = 0; r < n_rank; r++)
        {
            glm_layer_weights & W = L.w[r];
            const int d = n_rank > 1 ? m->rank_device(r) : L.device;

            // --- replicated on every rank -------------------------------------
            // Norms, the query/KV down-projections, the router and the indexer
            // are a few percent of the layer, and replicating them keeps the KV
            // rows and the top-k selection identical on every rank without a
            // collective.
            W.attn_norm = WL.full(d, true, "blk.%d.attn_norm.weight", il);
            W.wq_a      = WL.full(d, true, "blk.%d.attn_q_a.weight", il);
            W.q_a_norm  = WL.full(d, true, "blk.%d.attn_q_a_norm.weight", il);
            W.wkv_a_mqa = WL.full(d, true, "blk.%d.attn_kv_a_mqa.weight", il);
            W.kv_a_norm = WL.full(d, true, "blk.%d.attn_kv_a_norm.weight", il);
            W.ffn_norm  = WL.full(d, true, "blk.%d.ffn_norm.weight", il);

            if (L.indexer_full)
            {
                W.idx_attn_q_b = WL.full(d, true, "blk.%d.indexer.attn_q_b.weight", il);
                W.idx_attn_k   = WL.full(d, true, "blk.%d.indexer.attn_k.weight", il);
                W.idx_k_norm_w = WL.full(d, true, "blk.%d.indexer.k_norm.weight", il);
                W.idx_k_norm_b = WL.full(d, true, "blk.%d.indexer.k_norm.bias", il);
                W.idx_proj     = WL.full(d, true, "blk.%d.indexer.proj.weight", il);
            }

            // --- attention, sharded by head ------------------------------------
            const int64_t h0 = L.head_first[r];
            const int64_t hn = L.head_first[r + 1] - h0;
            if (n_rank > 1 && m->tp_heads())
            {
                W.wq_b = WL.slice_hi(d, 1, h0 * hp.n_embd_head_k, hn * hp.n_embd_head_k, true,
                                     "blk.%d.attn_q_b.weight", il);
                W.wk_b = WL.slice_hi(d, 2, h0, hn, true, "blk.%d.attn_k_b.weight", il);
                W.wv_b = WL.slice_hi(d, 2, h0, hn, true, "blk.%d.attn_v_b.weight", il);
                // Row-parallel: this rank's heads are a contiguous column range
                // of the output projection, and a head is n_embd_head_v elements,
                // so the split is always block-aligned.
                W.wo   = WL.slice_lo(d, h0 * hp.n_embd_head_v, hn * hp.n_embd_head_v, true,
                                     "blk.%d.attn_output.weight", il);
            }
            else
            {
                W.wq_b = WL.full(d, true, "blk.%d.attn_q_b.weight", il);
                W.wk_b = WL.full(d, true, "blk.%d.attn_k_b.weight", il);
                W.wv_b = WL.full(d, true, "blk.%d.attn_v_b.weight", il);
                W.wo   = WL.full(d, true, "blk.%d.attn_output.weight", il);
            }

            // --- FFN ------------------------------------------------------------
            if (!L.is_moe)
            {
                // Three dense layers in the whole model; replicating them costs
                // less than the collective a split would add.
                W.ffn_gate = WL.full(d, true, "blk.%d.ffn_gate.weight", il);
                W.ffn_up   = WL.full(d, true, "blk.%d.ffn_up.weight", il);
                W.ffn_down = WL.full(d, true, "blk.%d.ffn_down.weight", il);
            }
            else
            {
                W.ffn_gate_inp = WL.full(d, true, "blk.%d.ffn_gate_inp.weight", il);
                W.exp_probs_b  = WL.full(d, false, "blk.%d.exp_probs_b.bias", il);

                // Only the routed experts move to the host under --n-cpu-moe: the
                // router, the norms and the always-active shared expert are small
                // and every token needs them, so keeping them on the accelerator
                // saves a second host round trip for nothing.
                const int de = L.cpu_moe ? n_gpu : d;
                const int64_t f0 = L.moe_ff_first[r];
                const int64_t fn = L.moe_ff_first[r + 1] - f0;
                // Host-resident experts are left whole even under tensor
                // parallelism. Splitting them would save no host RAM and no host
                // time — the CPU backend already threads across every core — but
                // a strided strip cannot be served in place from the GGUF
                // mapping, so it would turn a mapped file into a private copy of
                // 200+ GiB of anonymous memory. Rank 0 evaluates these layers.
                if (n_rank > 1 && m->tp_experts() && !L.cpu_moe)
                {
                    // Column-parallel gate/up (a strip of the hidden dimension)
                    // feeding a row-parallel down, exactly as for a dense MLP —
                    // only here every one of the n_expert matrices is split the
                    // same way, so the ranks' outputs sum to the dense result.
                    W.ffn_gate_exps = WL.slice_mid(de, f0, fn, true, "blk.%d.ffn_gate_exps.weight", il);
                    W.ffn_up_exps   = WL.slice_mid(de, f0, fn, true, "blk.%d.ffn_up_exps.weight", il);
                    W.ffn_down_exps = WL.slice_lo(de, f0, fn, true, "blk.%d.ffn_down_exps.weight", il);
                }
                else
                {
                    W.ffn_gate_exps = WL.full(de, true, "blk.%d.ffn_gate_exps.weight", il);
                    W.ffn_up_exps   = WL.full(de, true, "blk.%d.ffn_up_exps.weight", il);
                    W.ffn_down_exps = WL.full(de, true, "blk.%d.ffn_down_exps.weight", il);
                }

                if (hp.n_expert_shared > 0)
                {
                    W.ffn_gate_shexp = WL.full(d, true, "blk.%d.ffn_gate_shexp.weight", il);
                    W.ffn_up_shexp   = WL.full(d, true, "blk.%d.ffn_up_shexp.weight", il);
                    W.ffn_down_shexp = WL.full(d, true, "blk.%d.ffn_down_shexp.weight", il);
                }
            }

            if (!W.attn_norm || !W.wq_a || !W.wq_b || !W.wkv_a_mqa || !W.wk_b || !W.wv_b || !W.wo || !W.ffn_norm)
            {
                fprintf(stderr, "[glm] layer %d rank %d is incomplete\n", il, r);
                return nullptr;
            }
        }
    }

    // Where each weight's bytes live, keyed by tensor pointer: sliced tensors
    // (tensor parallelism) read a sub-range that the GGUF tensor table does not
    // describe, so the loader carries its own list rather than re-deriving it.
    std::map<const ggml_tensor *, const pending_weight *> pending_by_tensor;
    for (const pending_weight & pw : WL.pending) pending_by_tensor[pw.t] = &pw;

    // --- host-resident experts straight from the GGUF mapping -------------
    // A private copy of 200+ GiB of experts is anonymous memory the kernel
    // cannot reclaim: under a container quota that is a silent OOM kill rather
    // than a slow load. File pages are evictable, so the same model loads
    // everywhere and degrades only to page-cache speed.
    if (ggml_get_first_tensor(m->w_ctx[n_gpu]) != nullptr)
    {
        const char * e = getenv("TS_GLM_MOE_MMAP");
        if (!(e && atoi(e) == 0))
        {
            ggml_context * hctx = m->w_ctx[n_gpu];
            for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
            {
                auto it = pending_by_tensor.find(t);
                if (it == pending_by_tensor.end()) continue;
                const pending_weight & pw = *it->second;
                // Only a contiguous range can be served in place.
                if (pw.src_row != 0 || ggml_nbytes(t) != pw.bytes) continue;
                ggml_backend_buffer_t buf = mmap_shard(*m, shards, pw.shard);
                if (!buf) continue;
                char * base = (char *) m->mmap_addrs[pw.shard];
                if (ggml_backend_tensor_alloc(buf, t, base + pw.file_off) != GGML_STATUS_SUCCESS) continue;
                m->mmap_weight_bytes += ggml_nbytes(t);
            }
            if (m->mmap_weight_bytes > 0)
                fprintf(stderr, "[glm] host experts served from the GGUF mapping (%.1f GiB, no private copy)\n",
                        m->mmap_weight_bytes / 1073741824.0);
        }
    }

    for (int d = 0; d <= n_gpu; d++)
    {
        if (ggml_get_first_tensor(m->w_ctx[d]) != nullptr)
        {
            bool unallocated = false;
            for (ggml_tensor * t = ggml_get_first_tensor(m->w_ctx[d]); t; t = ggml_get_next_tensor(m->w_ctx[d], t))
                if (t->data == nullptr) { unallocated = true; break; }
            if (unallocated)
            {
                m->w_buf[d] = ggml_backend_alloc_ctx_tensors(m->w_ctx[d], m->backends[d]);
                if (!m->w_buf[d]) { fprintf(stderr, "[glm] weight allocation failed on device %d\n", d); return nullptr; }
                // Weights, and ggml_backend_sched has to be told so: only a
                // WEIGHTS buffer makes it place an op on the device that holds
                // the matrix. With the default usage it picks by its own
                // heuristics, and under tensor parallelism that means a rank's
                // mul_mat_id can be scheduled onto another rank's device and
                // dereference a pointer that device does not own.
                ggml_backend_buffer_set_usage(m->w_buf[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
    }

    // --- upload -----------------------------------------------------------
    size_t uploaded = 0;
    {
        size_t chunk = (size_t) 64 * 1024 * 1024;
        if (const char * e = getenv("TS_GLM_LOAD_CHUNK_MB")) { long v = atol(e); if (v > 0) chunk = (size_t) v * 1024 * 1024; }
        int load_threads = 16;
        if (const char * e = getenv("TS_GLM_LOAD_THREADS")) { int v = atoi(e); if (v > 0) load_threads = v; }
        unsigned hw = std::thread::hardware_concurrency();
        if (hw > 0 && load_threads > (int) hw) load_threads = (int) hw;

        std::vector<load_job> jobs;
        bool sizes_ok = true;
        for (const pending_weight & pw : WL.pending)
        {
            ggml_tensor * t = pw.t;
            if (t->buffer != nullptr &&
                std::find(m->mmap_bufs.begin(), m->mmap_bufs.end(), t->buffer) != m->mmap_bufs.end())
                continue;   // served from the GGUF mapping

            const size_t total = ggml_nbytes(t);
            if (total != pw.bytes)
            {
                fprintf(stderr, "[glm] size mismatch for %s: tensor %zu vs source %zu\n", t->name, total, pw.bytes);
                sizes_ok = false;
                continue;
            }

            if (pw.src_row == 0)
            {
                for (size_t off = 0; off < total; off += chunk)
                    jobs.push_back({ t, pw.shard, pw.file_off + off, off, std::min(chunk, total - off), 0, 0 });
            }
            else
            {
                // Strided (row-parallel) slice: chunk by whole source rows.
                const size_t rows_total = total / pw.row_len;
                const size_t rows_per_chunk = std::max<size_t>(1, chunk / pw.row_len);
                for (size_t r0 = 0; r0 < rows_total; r0 += rows_per_chunk)
                {
                    const size_t rows = std::min(rows_per_chunk, rows_total - r0);
                    jobs.push_back({ t, pw.shard, pw.file_off + r0 * pw.src_row, r0 * pw.row_len,
                                     rows * pw.row_len, pw.src_row, pw.row_len });
                }
            }
            uploaded += total;
        }
        if (!sizes_ok) return nullptr;
        if (!upload_parallel(shards, jobs, load_threads)) return nullptr;

        // Warm the mapped experts with the same parallelism the copy path had —
        // lazy faulting would charge the whole read to the first prompt at
        // single-stream storage speed. Only when they actually fit: warming
        // more than the allowance just evicts itself.
        if (m->mmap_weight_bytes > 0)
        {
            const size_t allow = host_mem_allowance();
            const size_t headroom = (size_t) 8 * 1024 * 1024 * 1024;
            if (allow == 0 || m->mmap_weight_bytes + headroom <= allow)
            {
                const auto t_warm = std::chrono::steady_clock::now();
                std::vector<std::pair<const volatile char *, size_t>> ranges;
                ggml_context * hctx = m->w_ctx[n_gpu];
                for (ggml_tensor * t = ggml_get_first_tensor(hctx); t; t = ggml_get_next_tensor(hctx, t))
                {
                    if (t->buffer == nullptr ||
                        std::find(m->mmap_bufs.begin(), m->mmap_bufs.end(), t->buffer) == m->mmap_bufs.end())
                        continue;
                    ranges.emplace_back((const volatile char *) t->data, ggml_nbytes(t));
                }
                std::atomic<size_t> cursor(0);
                auto warm = [&]()
                {
                    for (;;)
                    {
                        const size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
                        if (i >= ranges.size()) break;
                        const volatile char * p = ranges[i].first;
                        for (size_t off = 0; off < ranges[i].second; off += 4096) (void) p[off];
                    }
                };
                std::vector<std::thread> pool;
                for (int i = 0; i < load_threads; i++) pool.emplace_back(warm);
                for (auto & th : pool) th.join();
                fprintf(stderr, "[glm] prefaulted %.1f GiB of mmapped host experts in %.1fs\n",
                        m->mmap_weight_bytes / 1073741824.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warm).count());
            }
        }
    }

    // --- tensor-parallel expert lookup tables ------------------------------

    // --- indexer rotation constant ----------------------------------------
    if (is_pow2(hp.indexer_head_size))
    {
        gen_hadamard(m->hadamard_host, hp.indexer_head_size);
        std::vector<uint8_t> dev_needs((size_t) n_gpu + 1, 0);
        for (int il = 0; il < hp.n_layer; il++)
        {
            if (!hp.indexer_full[il]) continue;
            if (m->tp > 1) { for (int r = 0; r < m->tp; r++) dev_needs[(size_t) m->rank_device(r)] = 1; }
            else dev_needs[(size_t) m->layers[il].device] = 1;
        }

        for (int d = 0; d <= n_gpu; d++)
        {
            if (!dev_needs[(size_t) d]) continue;
            m->hadamard[d] = ggml_new_tensor_2d(m->c_ctx[d], GGML_TYPE_F32,
                                                hp.indexer_head_size, hp.indexer_head_size);
            ggml_format_name(m->hadamard[d], "indexer_rot.%d", d);
        }
    }
    else
    {
        fprintf(stderr, "[glm] indexer key length %d is not a power of two; the Hadamard rotation is skipped "
                "(scores are mathematically the same, but F16 rounding will differ from llama.cpp)\n",
                hp.indexer_head_size);
    }

    // One allocation pass for every constant created above.
    for (int d = 0; d <= n_gpu; d++)
    {
        if (ggml_get_first_tensor(m->c_ctx[d]) == nullptr) continue;
        m->c_buf[d] = ggml_backend_alloc_ctx_tensors(m->c_ctx[d], m->backends[d]);
        if (!m->c_buf[d]) { fprintf(stderr, "[glm] constant allocation failed on device %d\n", d); return nullptr; }
        // These are per-device read-only constants — the Hadamard basis and, under
        // expert parallelism, the rank's expert mask. Marking the buffer as weights
        // is what makes ggml_backend_sched pin their consumers to this device; with
        // the default usage the scheduler is free to run the consumer elsewhere and
        // read the pointer from the wrong device.
        ggml_backend_buffer_set_usage(m->c_buf[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (m->hadamard[d])
            ggml_backend_tensor_set(m->hadamard[d], m->hadamard_host.data(), 0,
                                    m->hadamard_host.size() * sizeof(float));
    }

    // --- flash attention / fused indexer probes ---------------------------
    if (n_gpu > 0)
    {
        const char * fa_env = getenv("TS_GLM_FA");
        if (!(fa_env && atoi(fa_env) == 0))
        {
            ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
            ggml_context * pctx = ggml_init(pp);
            ggml_tensor * q = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.n_kv_row, 1, hp.n_head, 1);
            ggml_tensor * k = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, hp.n_kv_row, 256, 1, 1);
            ggml_tensor * v = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, hp.kv_lora_rank, 256, 1, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, 256, 1, 1, 1);
            ggml_tensor * fa = ggml_flash_attn_ext(pctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            m->flash_attn = ggml_backend_supports_op(m->backends[0], fa);
            ggml_free(pctx);
        }

        const char * lid_env = getenv("TS_GLM_FUSED_LID");
        if (!(lid_env && atoi(lid_env) == 0))
        {
            ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
            ggml_context * pctx = ggml_init(pp);
            ggml_tensor * q = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.indexer_head_size, hp.indexer_n_head, 1, 1);
            ggml_tensor * k = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, hp.indexer_head_size, 1, 256, 1);
            ggml_tensor * w = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.indexer_n_head, 1, 1, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, 256, 1, 1, 1);
            ggml_tensor * li = ggml_lightning_indexer(pctx, q, k, w, mask);
            m->fused_lid = ggml_backend_supports_op(m->backends[0], li);
            ggml_free(pctx);
        }
    }

    m->op_offload = (n_cpu_moe == 0);
    if (const char * e = getenv("TS_GLM_OP_OFFLOAD")) m->op_offload = atoi(e) != 0;

    // The weights are resident now, so stop estimating: ask the devices how much
    // they actually have left and size the caches to that. The pre-load estimate
    // above only has to be conservative enough to refuse the hopeless cases; this
    // is what decides the context the session really gets.
    if (!ctx_is_hard_limit && n_gpu > 0 && (m->tp <= 1 || m->tp <= n_gpu))
    {
        size_t free_min = SIZE_MAX;
        const int n_rank = m->tp > 1 ? m->tp : n_gpu;
        for (int r = 0; r < n_rank; r++)
        {
            const int d = m->tp > 1 ? m->rank_device(r) : r;
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            // Under a layer split the caches are spread over the devices, so each
            // one only carries its own layers' share; with tensor parallelism
            // every rank carries the lot.
            const size_t share = m->tp > 1 ? free_b : free_b * (size_t) n_gpu;
            free_min = std::min(free_min, share);
        }
        const size_t reserve = (size_t) 1024 * 1024 * 1024;   // leave the driver room to breathe
        const int fit = free_min > reserve ? ctx_that_fits(free_min - reserve) : 0;
        if (fit >= 256 && fit != m->n_ctx)
        {
            if (fit < ctx_requested)
                fprintf(stderr, "[glm] context %d tokens (the GGUF advertises %d): %.1f GiB free per rank after the "
                        "weights, and the caches and graphs have to live in it. Set MAX_CONTEXT to ask for a "
                        "different one.\n", fit, ctx_requested, free_min / 1073741824.0);
            m->n_ctx = fit;
        }
    }

    m->active_slot = slot_alloc(*m);
    if (!m->active_slot) { fprintf(stderr, "[glm] primary slot allocation failed\n"); return nullptr; }

    m->logits.resize((size_t) hp.n_vocab);

    {
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "[glm] glm-dsa: %d trunk layers (+%d MTP), n_embd=%d, %d heads, MLA(q_lora=%d, kv_lora=%d, "
                "head_k=%d, head_v=%d, rope=%d), %d experts top-%d (ff=%d), dense_lead=%d, indexer %dx%d top-%d on "
                "%d/%d layers, vocab=%d\n",
                hp.n_layer, hp.n_layer_nextn, hp.n_embd, hp.n_head, hp.q_lora_rank, hp.kv_lora_rank,
                hp.n_embd_head_k, hp.n_embd_head_v, hp.n_rot, hp.n_expert, hp.n_expert_used, hp.n_ff_exp,
                hp.n_dense_lead, hp.indexer_n_head, hp.indexer_head_size, hp.indexer_top_k,
                (int) std::count(hp.indexer_full.begin(), hp.indexer_full.end(), (uint8_t) 1), hp.n_layer, hp.n_vocab);
        fprintf(stderr, "[glm] loaded %.1f GiB across %d GPU(s) in %.1fs (%.2f GiB/s%s); n_ctx=%d, n_ubatch=%d, "
                "flash_attn=%s, fused_indexer=%s\n",
                uploaded / 1073741824.0, n_gpu, secs, secs > 0 ? uploaded / 1073741824.0 / secs : 0.0,
                m->mmap_weight_bytes > 0 ? ", host experts mmapped" : "",
                m->n_ctx, m->n_ubatch, m->flash_attn ? "on" : "off", m->fused_lid ? "on" : "off");
        if (m->tp > 1)
        {
            // The expert rows are split in whole quantization blocks and the
            // remainder rotates by layer, so quote the per-rank average rather
            // than one layer's slice.
            int rows0 = 0;
            for (int il = 0; il < hp.n_layer; il++)
                if (m->layers[il].is_moe) rows0 += m->layers[il].moe_ff_first[1];
            const int moe_layers = std::max(1, hp.n_layer - hp.n_dense_lead);
            fprintf(stderr, "[glm] tensor parallel across %d rank(s): every layer on every rank, "
                    "heads %d/%d and ~%d/%d expert rows per rank, 2 reductions per layer\n",
                    m->tp, m->layers[0].head_first[1], hp.n_head, rows0 / moe_layers, hp.n_ff_exp);
        }
        for (int d = 0; d < n_gpu; d++)
        {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->backends[d]), &free_b, &total_b);
            if (m->tp > 1)
            {
                if (d < m->tp)
                    fprintf(stderr, "[glm]   rank %d: all %d layers, %.1f GiB free after load\n",
                            d, hp.n_layer, free_b / 1073741824.0);
                continue;
            }
            int first = -1, last = -1, count = 0;
            for (int il = 0; il < hp.n_layer; il++)
                if (m->layers[il].device == d) { if (first < 0) first = il; last = il; count++; }
            if (count > 0)
                fprintf(stderr, "[glm]   device %d: layers %d..%d (%d), %.1f GiB free after load\n",
                        d, first, last, count, free_b / 1073741824.0);
            else
                fprintf(stderr, "[glm]   device %d: no layers, %.1f GiB free after load\n", d, free_b / 1073741824.0);
        }
    }

    return m.release();
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

struct graph_builder
{
    glm_model & m;
    const glm_hparams & hp;
    ggml_context * ctx;
    ggml_cgraph * gf;
    graph_build_result & res;
    glm_slot & slot;
    int64_t nt;      // tokens in this ubatch
    int64_t p0;      // position of the first token
    int64_t n_kv;    // cache columns the attention may read
    bool sparse;     // the indexer can actually drop something at this length

    graph_builder(glm_model & model, graph_build_result & r, glm_slot & s,
                  int64_t nt_, int64_t p0_, int64_t n_kv_, bool sparse_)
        : m(model), hp(model.hp), ctx(r.ctx), gf(r.gf), res(r), slot(s),
          nt(nt_), p0(p0_), n_kv(n_kv_), sparse(sparse_) {}

    /// Device that runs layer `il` for rank `r`: the rank's device under tensor
    /// parallelism, the layer's home device under a layer split.
    int device_of(int il, int r) const { return m.tp > 1 ? m.rank_device(r) : m.layers[il].device; }

    /// Sum of one partial per rank. With a single rank this is the identity, so
    /// the layer-split path pays nothing for it.
    ggml_tensor * reduce_ranks(ggml_tensor ** parts, int n)
    {
        ggml_tensor * acc = parts[0];
        for (int r = 1; r < n; r++) acc = ggml_add(ctx, acc, parts[r]);
        return acc;
    }

    // ---- batched decode ---------------------------------------------------
    // One graph, N tokens, N different sequences. Everything that does not touch
    // a cache is shared across the batch; only the cache write, the indexer
    // scoring and the attention itself are built per token, because each token
    // sees a different slot's history of a different length.

    /// The top-k mask for a single token, over that token's own n_kv.
    ggml_tensor * build_topk_mask_1(int dev, ggml_tensor * kq_mask, ggml_tensor * top_k, int64_t nkv)
    {
        const int64_t n_top_k = top_k->ne[0];

        ggml_tensor * base = ggml_fill(ctx, kq_mask, -INFINITY);                // [nkv, 1]
        ggml_set_output(base);
        ggml_tensor * all = ggml_view_3d(ctx, base, 1, nkv, 1, base->nb[0], base->nb[1], 0);
        ggml_tensor * idx = ggml_view_3d(ctx, top_k, n_top_k, 1, 1, top_k->nb[1], top_k->nb[1], 0);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top_k, 1), 0.0f);

        ggml_tensor * unmasked = ggml_set_rows(ctx, all, zeros, idx);
        ggml_tensor * masked = ggml_view_2d(ctx, unmasked, nkv, 1, unmasked->nb[2], 0);
        ggml_tensor * out = ggml_add(ctx, masked, kq_mask);
        pin(base, dev);
        pin(zeros, dev);
        pin(unmasked, dev);
        pin(out, dev);
        return out;
    }

    /// Indexer for a batch of single tokens. The projections run once over the
    /// whole batch; the cache append and the scoring run per token.
    void build_indexer_bd(int il, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                          bool score_now, std::vector<ggml_tensor *> & top_k)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[0];
        const int dev = device_of(il, 0);
        const int64_t D = hp.indexer_head_size;
        const int64_t H = hp.indexer_n_head;
        const int64_t N = nt;

        ggml_tensor * k = ggml_mul_mat(ctx, LW.idx_attn_k, cur);           // [D, N]
        k = ggml_norm(ctx, k, 0.0f);
        k = ggml_mul(ctx, k, LW.idx_k_norm_w);
        k = ggml_add(ctx, k, LW.idx_k_norm_b);
        k = ggml_reshape_3d(ctx, k, D, 1, N);
        k = rope(k, pos);
        if (m.hadamard[dev]) k = ggml_mul_mat(ctx, m.hadamard[dev], k);
        k = ggml_cont(ctx, k);

        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            ggml_tensor * cache = m.slots.at(B.slot_id)->idx_k[0][il];
            ggml_tensor * ki = ggml_cont(ctx, ggml_view_2d(ctx, k, D, 1, k->nb[1], (size_t) i * k->nb[2]));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, ki, B.kv_idx[dev]));
        }

        if (!score_now) return;

        ggml_tensor * q = ggml_mul_mat(ctx, LW.idx_attn_q_b, qr);          // [D*H, N]
        q = ggml_reshape_3d(ctx, q, D, H, N);
        q = rope(q, pos);
        if (m.hadamard[dev]) q = ggml_mul_mat(ctx, m.hadamard[dev], q);
        q = ggml_cont(ctx, q);

        ggml_tensor * w = ggml_mul_mat(ctx, LW.idx_proj, cur);             // [H, N]
        w = ggml_scale(ctx, w, 1.0f / sqrtf((float) (D * H)));

        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            if (!B.sparse) { top_k[(size_t) i] = nullptr; continue; }

            ggml_tensor * cache = m.slots.at(B.slot_id)->idx_k[0][il];
            ggml_tensor * k_all = ggml_view_3d(ctx, cache, D, 1, B.n_kv, cache->nb[1], cache->nb[1], 0);
            // Materialised rather than passed as offset views: a fused kernel is
            // free to assume its small operands start at their buffer.
            ggml_tensor * qi = ggml_cont(ctx, ggml_view_3d(ctx, q, D, H, 1, q->nb[1], q->nb[2], (size_t) i * q->nb[2]));
            ggml_tensor * wi = ggml_cont(ctx, ggml_view_2d(ctx, w, H, 1, w->nb[1], (size_t) i * w->nb[1]));

            ggml_tensor * score = nullptr;
            if (m.fused_lid)
            {
                score = ggml_lightning_indexer(ctx, qi, k_all, wi, B.lid_mask[dev]);
            }
            else
            {
                ggml_tensor * qp = ggml_permute(ctx, qi, 0, 2, 1, 3);      // [D, 1, H]
                ggml_tensor * kp = ggml_permute(ctx, k_all, 0, 2, 1, 3);   // [D, n_kv, 1]
                ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);              // [n_kv, 1, H]
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));    // [H, 1, n_kv]
                ggml_tensor * sc = ggml_relu(ctx, kq);
                sc = ggml_mul(ctx, sc, wi);
                sc = ggml_sum_rows(ctx, sc);
                sc = ggml_cont(ctx, ggml_permute(ctx, sc, 2, 1, 0, 3));    // [n_kv, 1, 1]
                score = ggml_add(ctx, sc, ggml_cast(ctx, B.lid_mask[dev], GGML_TYPE_F32));
            }
            const int n_top_k = (int) std::min<int64_t>(B.n_kv, hp.indexer_top_k);
            top_k[(size_t) i] = ggml_cont(ctx, ggml_top_k(ctx, score, n_top_k));
        }
    }

    /// Attention for a batch of single tokens: shared projections, per-token
    /// cache write and softmax, then a shared V decompression and output
    /// projection over the re-assembled batch.
    ggml_tensor * build_attention_bd(int il, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                                     const std::vector<ggml_tensor *> & masks)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[0];
        const int dev = device_of(il, 0);
        const int64_t n_head = hp.n_head;
        const int64_t N = nt;

        ggml_tensor * q = ggml_mul_mat(ctx, LW.wq_b, qr);
        ggml_tensor * q_nope = ggml_view_3d(ctx, q, hp.n_nope, n_head, N,
                                            ggml_row_size(q->type, hp.n_embd_head_k),
                                            ggml_row_size(q->type, hp.n_embd_head_k) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx, q, hp.n_rot, n_head, N,
                                          ggml_row_size(q->type, hp.n_embd_head_k),
                                          ggml_row_size(q->type, hp.n_embd_head_k) * n_head,
                                          ggml_row_size(q->type, hp.n_nope));

        ggml_tensor * kv_pe = ggml_mul_mat(ctx, LW.wkv_a_mqa, cur);
        const size_t row = ggml_row_size(kv_pe->type, hp.n_kv_row);
        ggml_tensor * kv_cmpr = ggml_view_2d(ctx, kv_pe, hp.kv_lora_rank, N, row, 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx, kv_pe, hp.n_rot, 1, N, row, row,
                                          ggml_row_size(kv_pe->type, hp.kv_lora_rank));

        q_pe = rope(q_pe, pos);
        k_pe = rope(k_pe, pos);
        kv_cmpr = rms(kv_cmpr, LW.kv_a_norm);

        ggml_tensor * q_nope_p = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_abs = ggml_mul_mat(ctx, LW.wk_b, q_nope_p);              // [kv_lora, N, n_head]
        q_abs = ggml_permute(ctx, q_abs, 0, 2, 1, 3);                           // [kv_lora, n_head, N]

        ggml_tensor * Qcur = ggml_cont(ctx, ggml_concat(ctx, q_abs, q_pe, 0));  // [n_kv_row, n_head, N]
        ggml_tensor * Kcur = ggml_cont(ctx, ggml_concat(ctx,
                ggml_reshape_3d(ctx, kv_cmpr, hp.kv_lora_rank, 1, N), k_pe, 0)); // [n_kv_row, 1, N]

        ggml_tensor * cat = nullptr;
        for (int64_t i = 0; i < N; i++)
        {
            bd_token & B = res.bd[(size_t) i];
            ggml_tensor * kv = m.slots.at(B.slot_id)->kv_k[0][il];

            ggml_tensor * ki = ggml_cont(ctx, ggml_view_2d(ctx, Kcur, hp.n_kv_row, 1, Kcur->nb[1], (size_t) i * Kcur->nb[2]));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv, ki, B.kv_idx[dev]));

            ggml_tensor * K = ggml_view_3d(ctx, kv, hp.n_kv_row, B.n_kv, 1, kv->nb[1], kv->nb[1] * B.n_kv, 0);
            ggml_tensor * V = ggml_view_3d(ctx, kv, hp.kv_lora_rank, B.n_kv, 1, kv->nb[1], kv->nb[1] * B.n_kv, 0);
            ggml_tensor * Qi = ggml_cont(ctx, ggml_view_3d(ctx, Qcur, hp.n_kv_row, n_head, 1,
                                                           Qcur->nb[1], Qcur->nb[2], (size_t) i * Qcur->nb[2]));

            ggml_tensor * out = nullptr;                                        // [kv_lora, 1, n_head]
            if (m.flash_attn)
            {
                ggml_tensor * qf = ggml_permute(ctx, Qi, 0, 2, 1, 3);           // [n_kv_row, 1, n_head]
                ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, K, V, masks[(size_t) i], hp.kq_scale(), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                out = ggml_permute(ctx, fa, 0, 2, 1, 3);                        // [kv_lora, 1, n_head]
            }
            else
            {
                ggml_tensor * qp = ggml_permute(ctx, Qi, 0, 2, 1, 3);
                ggml_tensor * kq = ggml_mul_mat(ctx, K, qp);                    // [n_kv, 1, n_head]
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                kq = ggml_soft_max_ext(ctx, kq, masks[(size_t) i], hp.kq_scale(), 0.0f);
                ggml_tensor * vp = ggml_cont(ctx, ggml_transpose(ctx, V));
                out = ggml_mul_mat(ctx, vp, kq);                                // [kv_lora, 1, n_head]
                ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            }
            out = ggml_cont(ctx, out);
            cat = cat ? ggml_concat(ctx, cat, out, 1) : out;                    // grow along the token axis
        }

        // One V decompression and one output projection for the whole batch.
        ggml_tensor * kqv = ggml_mul_mat(ctx, LW.wv_b, cat);                    // [head_v, N, n_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                               // [head_v, n_head, N]
        ggml_tensor * flat = ggml_cont_2d(ctx, kqv, (int64_t) hp.n_embd_head_v * n_head, N);
        return ggml_mul_mat(ctx, LW.wo, flat);
    }

    /// Whole graph for one batched decode step.
    void build_batched()
    {

        graph_inputs & inp = res.inp;
        const int64_t N = nt;

        bool dev_used[MAX_GPUS + 1] = {};
        for (int il = 0; il < hp.n_layer; il++) dev_used[m.layers[il].device] = true;

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!dev_used[d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", d);
            inp.tokens[d] = new_input(GGML_TYPE_I32, N, 0, nb, d);
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input(GGML_TYPE_I32, N, 0, nb, d);

            for (int64_t i = 0; i < N; i++)
            {
                bd_token & B = res.bd[(size_t) i];
                snprintf(nb, sizeof(nb), "bd%lld_kv_idx.%d", (long long) i, d);
                B.kv_idx[d] = new_input(GGML_TYPE_I64, 1, 0, nb, d);
                snprintf(nb, sizeof(nb), "bd%lld_kq_mask.%d", (long long) i, d);
                B.kq_mask[d] = new_input(m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, B.n_kv, 1, nb, d);
                if (B.sparse)
                {
                    snprintf(nb, sizeof(nb), "bd%lld_lid_mask.%d", (long long) i, d);
                    B.lid_mask[d] = new_input(GGML_TYPE_F16, B.n_kv, 1, nb, d);
                }
            }
        }

        ggml_tensor * inpL = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.layers[0].device]);

        std::vector<ggml_tensor *> top_k((size_t) N, nullptr);
        std::vector<ggml_tensor *> masks((size_t) N, nullptr);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const glm_layer & L = m.layers[il];
            const int dev = L.device;

            if (il > 0 && L.device != m.layers[il - 1].device)
                pin(inpL, L.device);

            ggml_tensor * residual = inpL;
            ggml_tensor * cur = rms(inpL, L.w[0].attn_norm);
            ggml_tensor * qr = rms(ggml_mul_mat(ctx, L.w[0].wq_a, cur), L.w[0].q_a_norm);

            if (L.indexer_full)
            {
                build_indexer_bd(il, cur, qr, inp.pos[dev], true, top_k);
                for (int64_t i = 0; i < N; i++)
                {
                    bd_token & B = res.bd[(size_t) i];
                    masks[(size_t) i] = (B.sparse && top_k[(size_t) i])
                        ? build_topk_mask_1(dev, B.kq_mask[dev], top_k[(size_t) i], B.n_kv)
                        : B.kq_mask[dev];
                }
            }
            else
            {
                // A layer without its own indexer has no key cache either — it
                // attends through the previous full layer's selection.
                for (int64_t i = 0; i < N; i++)
                    if (!masks[(size_t) i]) masks[(size_t) i] = res.bd[(size_t) i].kq_mask[dev];
            }

            trace("attn_norm", il, 0, cur);
            trace("qr", il, 0, qr);
            inpL = ggml_add(ctx, residual, build_attention_bd(il, cur, qr, inp.pos[dev], masks));
            trace("ffn_inp", il, 0, inpL);

            residual = inpL;
            cur = rms(inpL, L.w[0].ffn_norm);
            if (!L.is_moe)
            {
                inpL = ggml_add(ctx, residual, build_dense_ffn(il, 0, cur));
                trace("l_out", il, 0, inpL);
                continue;
            }
            ggml_tensor * ffn = build_moe(il, 0, cur);
            ggml_tensor * shexp = build_shexp(il, 0, cur);
            if (shexp) ffn = ggml_add(ctx, ffn, shexp);
            inpL = ggml_add(ctx, residual, ffn);
            trace("l_out", il, 0, inpL);
        }

        ggml_tensor * cur = rms(inpL, m.output_norm);
        cur = ggml_mul_mat(ctx, m.output, cur);                                 // [vocab, N]
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;
        ggml_build_forward_expand(gf, cur);
        bd_log("built, %d nodes\n", ggml_graph_n_nodes(gf));
    }

    /// Layers TS_GLM_TRACE selects (comma-separated, or "all").
    static bool trace_layer(int il)
    {
        static const std::string spec = []() { const char * e = getenv("TS_GLM_TRACE"); return e ? std::string(e) : std::string(); }();
        if (spec.empty()) return false;
        if (spec == "all") return true;
        const std::string needle = std::to_string(il);
        size_t pos = 0;
        while ((pos = spec.find(needle, pos)) != std::string::npos)
        {
            const bool lb = pos == 0 || spec[pos - 1] == ',';
            const size_t end = pos + needle.size();
            const bool rb = end == spec.size() || spec[end] == ',';
            if (lb && rb) return true;
            pos = end;
        }
        return false;
    }

    int trace_il = 0;
    int trace_rank = 0;

    void trace(const char * tag, int il, int rank, ggml_tensor * t)
    {
        if (!t || !trace_layer(il)) return;
        char name[128];
        snprintf(name, sizeof(name), "%s-%d.r%d", tag, il, rank);
        ggml_set_output(t);
        res.traces.push_back({ name, t });
        ggml_build_forward_expand(gf, t);
    }

    ggml_tensor * rms(ggml_tensor * x, ggml_tensor * w)
    {
        x = ggml_rms_norm(ctx, x, hp.rms_eps);
        if (w) x = ggml_mul(ctx, x, w);
        return x;
    }

    // A pin is honoured by ggml_backend_sched unconditionally, so it is a
    // promise that the target backend can run the node. Leaf inputs have
    // nothing to execute and are always safe to pin.
    void pin(ggml_tensor * t, int dev)
    {
        if (dev < 0 || dev >= m.n_gpu) return;
        if (t->op != GGML_OP_NONE && !ggml_backend_supports_op(m.backends[dev], t)) return;
        ggml_backend_sched_set_tensor_backend(res.sched, t, m.backends[dev]);
    }

    ggml_tensor * new_input_3d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    ggml_tensor * new_input(ggml_type type, int64_t ne0, int64_t ne1, const char * name, int dev)
    {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1)
                                  : ggml_new_tensor_1d(ctx, type, ne0);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * pos)
    {
        // glm-dsa is LLAMA_ROPE_TYPE_NORM and carries no YaRN scaling.
        return ggml_rope_ext(ctx, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NORMAL, 0,
                             hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    // ---- DSA lightning indexer -------------------------------------------
    // score[j, t] = sum_h relu(q[t,h] . k[j]) * w[t,h], masked causally; the
    // top-k of that is what the real attention is allowed to see.
    //
    // The keys are cached on EVERY full-indexer layer, including while the
    // context is still shorter than top_k and no selection is needed: a later
    // ubatch scores these tokens, and a chunk that skipped its own keys would
    // score whatever zeros the cache was cleared to. Returns null when only the
    // keys were wanted.
    ggml_tensor * build_indexer(int il, int rank, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                                ggml_tensor * lid_mask, ggml_tensor * kv_idxs, bool score_now)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        const int dev = device_of(il, rank);
        const int64_t D = hp.indexer_head_size;
        const int64_t H = hp.indexer_n_head;

        ggml_tensor * k = ggml_mul_mat(ctx, LW.idx_attn_k, cur);           // [D, nt]
        k = ggml_norm(ctx, k, 0.0f);                                       // f_norm_eps is 0 for glm-dsa
        k = ggml_mul(ctx, k, LW.idx_k_norm_w);
        k = ggml_add(ctx, k, LW.idx_k_norm_b);
        k = ggml_reshape_3d(ctx, k, D, 1, nt);
        k = rope(k, pos);
        if (m.hadamard[dev]) k = ggml_mul_mat(ctx, m.hadamard[dev], k);

        // Append this ubatch's indexer keys. The destination rows arrive as a
        // graph INPUT rather than a baked view offset, so one cached graph
        // serves every position (which is the whole point of the cache).
        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, slot.idx_k[rank][il], ggml_reshape_2d(ctx, k, D, nt), kv_idxs));

        if (!score_now)
            return nullptr;

        ggml_tensor * q = ggml_mul_mat(ctx, LW.idx_attn_q_b, qr);          // [D*H, nt]
        q = ggml_reshape_3d(ctx, q, D, H, nt);
        q = rope(q, pos);
        if (m.hadamard[dev]) q = ggml_mul_mat(ctx, m.hadamard[dev], q);

        ggml_tensor * k_all = ggml_view_3d(ctx, slot.idx_k[rank][il], D, 1, n_kv,
                                           slot.idx_k[rank][il]->nb[1], slot.idx_k[rank][il]->nb[1], 0);

        ggml_tensor * w = ggml_mul_mat(ctx, LW.idx_proj, cur);             // [H, nt]
        // Pre-scaled so the scale never touches the big [n_kv, nt] score tensor.
        w = ggml_scale(ctx, w, 1.0f / sqrtf((float) (D * H)));

        ggml_tensor * score = nullptr;
        if (m.fused_lid)
        {
            score = ggml_lightning_indexer(ctx, q, k_all, w, lid_mask);    // [n_kv, nt]
        }
        else
        {
            ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);           // [D, nt, H]
            ggml_tensor * kp = ggml_permute(ctx, k_all, 0, 2, 1, 3);       // [D, n_kv, 1]
            ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                  // [n_kv, nt, H]
            // F32 accumulation: the cache is F16, and without this ggml would
            // convert the query to F16 and dot in half precision. The fused
            // kernel converts the KEY up to F32 instead, and matching it is what
            // keeps the top-k selection identical at long context.
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));        // [H, nt, n_kv]
            ggml_tensor * s = ggml_relu(ctx, kq);
            s = ggml_mul(ctx, s, w);                                       // broadcast over n_kv
            s = ggml_sum_rows(ctx, s);                                     // [1, nt, n_kv]
            s = ggml_cont(ctx, ggml_permute(ctx, s, 2, 1, 0, 3));          // [n_kv, nt, 1]
            score = ggml_add(ctx, s, ggml_cast(ctx, lid_mask, GGML_TYPE_F32));
        }
        const int n_top_k = (int) std::min<int64_t>(n_kv, hp.indexer_top_k);
        return ggml_cont(ctx, ggml_top_k(ctx, score, n_top_k));            // I32 [n_top_k, nt]
    }

    /// Causal mask with the DSA selection folded in: start fully masked, unmask
    /// the cells the indexer picked, then add the plain causal mask back so a
    /// picked-but-future cell stays masked.
    ///
    /// <para>Built once per FULL indexer layer and reused by the "shared" layers
    /// that follow it — they attend through the same selection, so rebuilding an
    /// identical [n_kv, n_tokens] mask three more times is pure cost.</para>
    ///
    /// <para>The -inf canvas is marked as a graph output on purpose. ggml_set_rows
    /// writes THROUGH a view into that tensor's buffer, and left as an ordinary
    /// intermediate the graph allocator is free to hand the same bytes to another
    /// node while the view is still live — which showed up as an all-masked row
    /// and a NaN softmax as soon as several tensor-parallel ranks put enough
    /// nodes between the write and the read. Marking it an output takes it out of
    /// the reuse pool; hoisting keeps the number of such buffers at one per full
    /// indexer layer.</para>
    ggml_tensor * build_topk_mask(int dev, ggml_tensor * kq_mask, ggml_tensor * top_k)
    {
        const int64_t n_top_k = top_k->ne[0];
        trace("topk", trace_il, trace_rank, top_k);

        ggml_tensor * base = ggml_fill(ctx, kq_mask, -INFINITY);               // [n_kv, nt]
        ggml_set_output(base);
        ggml_tensor * all = ggml_view_3d(ctx, base, 1, n_kv, nt, base->nb[0], base->nb[1], 0);   // [1, n_kv, nt]

        ggml_tensor * idx = ggml_view_3d(ctx, top_k, n_top_k, nt, 1,
                                         top_k->nb[1], top_k->nb[2], 0);        // [n_top_k, nt, 1]

        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top_k, nt), 0.0f);

        ggml_tensor * unmasked = ggml_set_rows(ctx, all, zeros, idx);           // [1, n_kv, nt]
        ggml_tensor * masked = ggml_view_2d(ctx, unmasked, n_kv, nt, unmasked->nb[2], 0);   // [n_kv, nt]
        ggml_tensor * out = ggml_add(ctx, masked, kq_mask);

        // ggml_set_rows writes THROUGH a view into `base`, so the write and the
        // buffer have to belong to the same backend. Left to the scheduler's own
        // heuristics the fill and the write can land on different devices, and
        // then the store goes to a buffer the writing backend does not own —
        // which corrupts whatever else lives there. Pinning the whole chain to
        // the device that consumes the mask keeps the aliasing local.
        pin(base, dev);
        pin(zeros, dev);
        pin(unmasked, dev);
        pin(out, dev);
        return out;
    }

    // ---- MLA attention ----------------------------------------------------
    ggml_tensor * build_attention(int il, int rank, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * pos,
                                  ggml_tensor * mask, ggml_tensor * kv_idxs)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        // Under tensor parallelism this rank owns a contiguous run of heads; the
        // key/value cache row is head-independent, so it is computed (identically)
        // and stored by every rank instead of being shared through a collective.
        const int64_t n_head = (m.tp > 1 && m.tp_heads()) ? (L.head_first[rank + 1] - L.head_first[rank])
                                                          : hp.n_head;

        ggml_tensor * q = ggml_mul_mat(ctx, LW.wq_b, qr);                       // [n_head*head_k, nt]

        ggml_tensor * q_nope = ggml_view_3d(ctx, q, hp.n_nope, n_head, nt,
                                            ggml_row_size(q->type, hp.n_embd_head_k),
                                            ggml_row_size(q->type, hp.n_embd_head_k) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx, q, hp.n_rot, n_head, nt,
                                          ggml_row_size(q->type, hp.n_embd_head_k),
                                          ggml_row_size(q->type, hp.n_embd_head_k) * n_head,
                                          ggml_row_size(q->type, hp.n_nope));

        ggml_tensor * kv_pe = ggml_mul_mat(ctx, LW.wkv_a_mqa, cur);              // [kv_lora + rope, nt]
        const size_t row = ggml_row_size(kv_pe->type, hp.n_kv_row);
        ggml_tensor * kv_cmpr = ggml_view_2d(ctx, kv_pe, hp.kv_lora_rank, nt, row, 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx, kv_pe, hp.n_rot, 1, nt, row, row,
                                          ggml_row_size(kv_pe->type, hp.kv_lora_rank));

        q_pe = rope(q_pe, pos);
        k_pe = rope(k_pe, pos);
        kv_cmpr = rms(kv_cmpr, LW.kv_a_norm);

        // q_absorbed[h] = wk_b[h]^T q_nope[h] : the per-head K decompression,
        // folded into the query so 64 heads share one 576-wide key head.
        ggml_tensor * q_nope_p = ggml_permute(ctx, q_nope, 0, 2, 1, 3);         // [nope, nt, n_head]
        ggml_tensor * q_abs = ggml_mul_mat(ctx, LW.wk_b, q_nope_p);              // [kv_lora, nt, n_head]
        q_abs = ggml_permute(ctx, q_abs, 0, 2, 1, 3);                           // [kv_lora, n_head, nt]

        // rope goes last so an in-place context shift stays possible
        ggml_tensor * Qcur = ggml_concat(ctx, q_abs, q_pe, 0);                  // [n_kv_row, n_head, nt]

        ggml_tensor * Kcur = ggml_concat(ctx, ggml_reshape_3d(ctx, kv_cmpr, hp.kv_lora_rank, 1, nt),
                                         k_pe, 0);                              // [n_kv_row, 1, nt]

        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, slot.kv_k[rank][il], ggml_reshape_2d(ctx, Kcur, hp.n_kv_row, nt), kv_idxs));

        ggml_tensor * kv = slot.kv_k[rank][il];
        ggml_tensor * K = ggml_view_3d(ctx, kv, hp.n_kv_row, n_kv, 1, kv->nb[1], kv->nb[1] * n_kv, 0);
        // V is the compressed half of the same rows; wv_b decompresses the result.
        ggml_tensor * V = ggml_view_3d(ctx, kv, hp.kv_lora_rank, n_kv, 1, kv->nb[1], kv->nb[1] * n_kv, 0);

        ggml_tensor * cur_attn = nullptr;
        if (m.flash_attn)
        {
            ggml_tensor * qf = ggml_permute(ctx, Qcur, 0, 2, 1, 3);             // [n_kv_row, nt, n_head]
            ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, K, V, mask, hp.kq_scale(), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            // [kv_lora, n_head, nt] -> [kv_lora, nt, n_head] so wv_b's per-head
            // matmul runs as a matrix-matrix product with nt in dimension 1.
            fa = ggml_permute(ctx, fa, 0, 2, 1, 3);
            fa = ggml_mul_mat(ctx, LW.wv_b, fa);                                 // [head_v, nt, n_head]
            fa = ggml_cont(ctx, ggml_permute(ctx, fa, 0, 2, 1, 3));             // [head_v, n_head, nt]
            cur_attn = ggml_reshape_2d(ctx, fa, (int64_t) hp.n_embd_head_v * n_head, nt);
        }
        else
        {
            // K and V already come out of the cache in the layout mul_mat wants
            // ([n_embd, n_kv, n_head_kv]); only the query needs permuting.
            ggml_tensor * qp = ggml_permute(ctx, Qcur, 0, 2, 1, 3);             // [n_kv_row, nt, n_head]
            ggml_tensor * kq = ggml_mul_mat(ctx, K, qp);                        // [n_kv, nt, n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, mask, hp.kq_scale(), 0.0f);

            ggml_tensor * vp = ggml_cont(ctx, ggml_transpose(ctx, V));          // [n_kv, kv_lora, 1]
            ggml_tensor * kqv = ggml_mul_mat(ctx, vp, kq);                      // [kv_lora, nt, n_head]
            // V is F16; without this the F32 softmax weights would be rounded to
            // half before the context product.
            ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
            kqv = ggml_mul_mat(ctx, LW.wv_b, kqv);                               // [head_v, nt, n_head]
            kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                           // [head_v, n_head, nt]
            cur_attn = ggml_cont_2d(ctx, kqv, (int64_t) hp.n_embd_head_v * n_head, nt);
        }

        return ggml_mul_mat(ctx, LW.wo, cur_attn);
    }

    // ---- FFN --------------------------------------------------------------
    ggml_tensor * build_dense_ffn(int il, int rank, ggml_tensor * cur)
    {
        const glm_layer_weights & LW = m.layers[il].w[rank];
        ggml_tensor * gate = ggml_mul_mat(ctx, LW.ffn_gate, cur);
        ggml_tensor * up = ggml_mul_mat(ctx, LW.ffn_up, cur);
        ggml_tensor * h = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        return ggml_mul_mat(ctx, LW.ffn_down, h);
    }

    ggml_tensor * build_shexp(int il, int rank, ggml_tensor * cur)
    {
        const glm_layer_weights & LW = m.layers[il].w[rank];
        if (!LW.ffn_gate_shexp) return nullptr;
        ggml_tensor * gate = ggml_mul_mat(ctx, LW.ffn_gate_shexp, cur);
        ggml_tensor * up = ggml_mul_mat(ctx, LW.ffn_up_shexp, cur);
        ggml_tensor * h = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        return ggml_mul_mat(ctx, LW.ffn_down_shexp, h);
    }

    /// Routed-expert FFN for one rank. Returns this rank's PARTIAL sum; the
    /// caller reduces across ranks and adds the (replicated) shared expert.
    ggml_tensor * build_moe(int il, int rank, ggml_tensor * cur)
    {
        const glm_layer & L = m.layers[il];
        const glm_layer_weights & LW = L.w[rank];
        const int64_t n_expert = hp.n_expert;
        const int64_t n_used = hp.n_expert_used;

        // Routing is replicated: every rank sees the same logits and picks the
        // same global top-k, so no collective is needed to agree on it.
        ggml_tensor * logits = ggml_mul_mat(ctx, LW.ffn_gate_inp, cur);    // [n_expert, nt]
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);

        ggml_tensor * probs = hp.expert_gating_func == 2
            ? ggml_sigmoid(ctx, logits)
            : ggml_soft_max(ctx, logits);

        // The routing bias steers SELECTION only; the weights come from the
        // unbiased probabilities.
        ggml_tensor * selection = LW.exp_probs_b ? ggml_add(ctx, probs, LW.exp_probs_b) : probs;
        ggml_tensor * selected = ggml_argsort_top_k(ctx, selection, (int) n_used);   // I32 [n_used, nt]

        // The router runs identically on every rank — same logits, same global
        // top-k, same weights — because the split is inside each expert, not
        // across them. Backends fuse this whole chain (sigmoid, bias, argsort,
        // gather, normalise) into one kernel, so nothing here may reach in for
        // an intermediate: a fused kernel never materialises them.
        ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, n_expert, nt);
        ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);      // [1, n_used, nt]

        if (hp.expert_weights_norm)
        {
            weights = ggml_reshape_2d(ctx, weights, n_used, nt);
            ggml_tensor * sum = ggml_sum_rows(ctx, weights);
            sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, sum);
            weights = ggml_reshape_3d(ctx, weights, 1, n_used, nt);
        }
        if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
            weights = ggml_scale(ctx, weights, hp.expert_weights_scale);

        ggml_build_forward_expand(gf, weights);

        ggml_tensor * cur3 = ggml_reshape_3d(ctx, cur, hp.n_embd, 1, nt);

        ggml_tensor * up = ggml_mul_mat_id(ctx, LW.ffn_up_exps, cur3, selected);    // [n_ff_exp, n_used, nt]
        ggml_tensor * gate = ggml_mul_mat_id(ctx, LW.ffn_gate_exps, cur3, selected);
        ggml_tensor * h = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, LW.ffn_down_exps, h, selected); // [n_embd, n_used, nt]

        trace("moe_sel", il, rank, selected);
        trace("moe_probs", il, rank, probs);
        trace("moe_w", il, rank, weights);
        trace("moe_up", il, rank, up);
        trace("moe_gate", il, rank, gate);
        trace("moe_down", il, rank, experts);

        experts = ggml_mul(ctx, experts, weights);

        ggml_tensor * moe_out = nullptr;
        for (int64_t e = 0; e < n_used; e++)
        {
            ggml_tensor * v = ggml_view_2d(ctx, experts, hp.n_embd, nt, experts->nb[2], e * experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx, moe_out, v) : v;
        }
        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);
        return moe_out;
    }

    void build()
    {
        graph_inputs & inp = res.inp;

        // Per-token inputs are duplicated per participating device so the
        // scheduler never inserts a synchronized cross-backend input copy
        // inside the per-token compute.
        std::vector<uint8_t> used_dev((size_t) m.n_gpu + 1, 0);
        if (m.tp > 1)
            for (int r = 0; r < m.tp; r++) used_dev[(size_t) m.rank_device(r)] = 1;
        else
            for (int il = 0; il < hp.n_layer; il++) used_dev[(size_t) m.layers[il].device] = 1;

        // Only the token ids are single-device (the embedding lookup); the
        // positions, masks and cache indices are consumed by every layer, so
        // each device gets its own copy and the scheduler never has to insert a
        // synchronized cross-backend input copy inside the per-token compute.
        //
        // An input no node reads is never allocated by the graph allocator, so
        // creating one for a device that turns out not to need it would leave a
        // buffer-less tensor for the input fill to write into. Only devices that
        // actually host a layer get inputs, and the "has a full indexer" test
        // gates the indexer mask the same way.
        const ggml_type mask_type = m.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        std::vector<uint8_t> dev_has_indexer((size_t) m.n_gpu + 1, 0);
        for (int il = 0; il < hp.n_layer; il++)
        {
            if (!m.layers[il].indexer_full) continue;
            if (m.tp > 1) { for (int r = 0; r < m.tp; r++) dev_has_indexer[(size_t) m.rank_device(r)] = 1; }
            else dev_has_indexer[(size_t) m.layers[il].device] = 1;
        }

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!used_dev[(size_t) d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input(GGML_TYPE_I32, nt, 0, nb, d);
            snprintf(nb, sizeof(nb), "kq_mask.%d", d);
            inp.kq_mask[d] = new_input(mask_type, n_kv, nt, nb, d);
            snprintf(nb, sizeof(nb), "kv_idxs.%d", d);
            inp.kv_idxs[d] = new_input(GGML_TYPE_I64, nt, 0, nb, d);
            if (sparse && dev_has_indexer[(size_t) d])
            {
                snprintf(nb, sizeof(nb), "lid_mask.%d", d);
                inp.lid_mask[d] = new_input(GGML_TYPE_F16, n_kv, nt, nb, d);
            }
        }
        {
            const int dev_embd = m.tp > 1 ? 0 : m.layers[0].device;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", dev_embd);
            inp.tokens[dev_embd] = new_input(GGML_TYPE_I32, nt, 0, nb, dev_embd);
        }
        const int dev_last = m.tp > 1 ? 0 : m.layers[(size_t) hp.n_layer - 1].device;
        // Only when the LM head runs: an input no node reads is never allocated,
        // and writing to it would fault.
        if (res.want_logits)
            inp.out_ids = new_input(GGML_TYPE_I32, res.n_out, 0, "inp_out_ids", dev_last);

        ggml_tensor * inpL = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.tp > 1 ? 0 : m.layers[0].device]);

        // `top_k` deliberately survives a device (or rank) change. A "shared"
        // indexer layer must see the selection of the last FULL layer wherever
        // that landed, exactly as llama.cpp does — dropping it would silently run
        // the layers after a boundary with dense attention, which is a different
        // model.
        ggml_tensor * top_k[MAX_GPUS] = {};
        // The attention mask that goes with the current selection, per rank.
        ggml_tensor * topk_mask[MAX_GPUS] = {};
        ggml_tensor * part[MAX_GPUS] = {};
        ggml_tensor * shexp[MAX_GPUS] = {};
        // A half that is not sharded is computed once, on rank 0: every rank
        // holds the same weights there, so summing N identical partials would
        // multiply the result by N.
        const int n_attn = (m.tp > 1 && m.tp_heads()) ? m.tp : 1;
        const int n_moe = (m.tp > 1 && m.tp_experts()) ? m.tp : 1;

        for (int il = 0; il < hp.n_layer; il++)
        {
            const glm_layer & L = m.layers[il];

            if (m.tp == 1 && il > 0 && L.device != m.layers[il - 1].device)
                pin(inpL, L.device);

            // ---- attention: every rank runs its own heads ----------------------
            ggml_tensor * residual = inpL;
            for (int r = 0; r < n_attn; r++)
            {
                const int dev = device_of(il, r);
                ggml_tensor * cur = rms(inpL, L.w[r].attn_norm);
                ggml_tensor * qr = rms(ggml_mul_mat(ctx, L.w[r].wq_a, cur), L.w[r].q_a_norm);

                trace("attn_norm", il, r, cur);
                trace("qr", il, r, qr);

                if (L.indexer_full)
                {
                    top_k[r] = build_indexer(il, r, cur, qr, inp.pos[dev], inp.lid_mask[dev],
                                             inp.kv_idxs[dev], sparse);
                    trace_il = il; trace_rank = r;
                    topk_mask[r] = (sparse && top_k[r]) ? build_topk_mask(dev, inp.kq_mask[dev], top_k[r]) : nullptr;
                }

                ggml_tensor * mask = (sparse && topk_mask[r]) ? topk_mask[r] : inp.kq_mask[dev];
                part[r] = build_attention(il, r, cur, qr, inp.pos[dev], mask, inp.kv_idxs[dev]);
                trace("attn_out", il, r, part[r]);
            }
            // Row-parallel output projection: the ranks' partials sum to the
            // dense result.
            inpL = ggml_add(ctx, residual, reduce_ranks(part, n_attn));
            trace("ffn_inp", il, 0, inpL);

            // ---- FFN ------------------------------------------------------------
            residual = inpL;
            if (!L.is_moe)
            {
                // Dense layers are replicated, so rank 0's result IS the result.
                ggml_tensor * cur = rms(inpL, L.w[0].ffn_norm);
                inpL = ggml_add(ctx, residual, build_dense_ffn(il, 0, cur));
                continue;
            }

            // A layer whose experts stayed on the host also stayed whole, so
            // rank 0's result is the whole result.
            const int n_moe_l = L.cpu_moe ? 1 : n_moe;
            for (int r = 0; r < n_moe_l; r++)
            {
                ggml_tensor * cur = rms(inpL, L.w[r].ffn_norm);
                part[r] = build_moe(il, r, cur);
                shexp[r] = r == 0 ? build_shexp(il, 0, cur) : nullptr;
                trace("ffn_norm", il, r, cur);
                trace("moe_out", il, r, part[r]);
            }
            // The shared expert is replicated, so it is added ONCE, after the
            // routed partials have been reduced.
            ggml_tensor * ffn = reduce_ranks(part, n_moe_l);
            if (shexp[0]) ffn = ggml_add(ctx, ffn, shexp[0]);
            inpL = ggml_add(ctx, residual, ffn);
            trace("l_out", il, 0, inpL);
        }

        if (!res.want_logits)
        {
            // A non-final prefill chunk only has to leave its KV behind. Running
            // the 154880-row LM head anyway would re-read the whole output matrix
            // once per chunk for a result nobody reads.
            ggml_build_forward_expand(gf, inpL);
            return;
        }

        ggml_tensor * cur = ggml_get_rows(ctx, inpL, inp.out_ids);
        cur = rms(cur, m.output_norm);
        cur = ggml_mul_mat(ctx, m.output, cur);
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;
        ggml_build_forward_expand(gf, cur);
    }
};

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

// Cache columns the attention reads. Padded so decode reuses one graph shape
// for a whole 256-token stretch instead of rebuilding every token.
static int64_t plan_n_kv(const glm_model & m, int64_t p_end)
{
    return std::min<int64_t>(pad_to(p_end, 256), m.n_ctx);
}

/// An input the built graph turned out not to need. The per-device inputs are
/// created for every device a layer maps to, but a device can end up hosting
/// only part of a layer — expert-parallel-only sharding gives a rank the MoE
/// and no attention — and then the scheduler never allocates the inputs that
/// part would have consumed. Writing to one of those asserts inside ggml, so
/// every host-side fill goes through this check.
static inline bool live(const ggml_tensor * t)
{
    return t != nullptr && t->buffer != nullptr;
}

static void set_input_i32(ggml_tensor * t, const int32_t * v, size_t n)
{
    if (live(t)) ggml_backend_tensor_set(t, v, 0, n * sizeof(int32_t));
}

/// Print what TS_GLM_TRACE selected, once the graph has run.
static void print_traces(const graph_build_result * gr)
{
    if (gr->traces.empty()) return;
    {
        std::vector<float> buf;
        for (const trace_entry & e : gr->traces)
        {
            const int64_t n = ggml_nelements(e.t);
            buf.resize((size_t) n);
            if (e.t->type == GGML_TYPE_I32)
            {
                std::vector<int32_t> ib((size_t) n);
                ggml_backend_tensor_get(e.t, ib.data(), 0, (size_t) n * sizeof(int32_t));
                int32_t lo = ib.empty() ? 0 : ib[0], hi = lo;
                long long isum = 0;
                for (int64_t i = 0; i < n; i++) { lo = std::min(lo, ib[(size_t) i]); hi = std::max(hi, ib[(size_t) i]); isum += ib[(size_t) i]; }
                fprintf(stderr, "[glm-trace] %-20s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] i32 min=%d max=%d sum=%lld\n",
                        e.name.c_str(), e.t->ne[0], e.t->ne[1], e.t->ne[2], lo, hi, isum);
                continue;
            }
            ggml_backend_tensor_get(e.t, buf.data(), 0, (size_t) n * sizeof(float));
            double sum = 0, asum = 0;
            bool nan = false;
            for (int64_t i = 0; i < n; i++)
            {
                sum += buf[(size_t) i];
                asum += fabs((double) buf[(size_t) i]);
                if (std::isnan(buf[(size_t) i])) nan = true;
            }
            fprintf(stderr, "[glm-trace] %-20s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] asum=%.6f sum=%.6f%s\n",
                    e.name.c_str(), e.t->ne[0], e.t->ne[1], e.t->ne[2], asum, sum, nan ? "  <-- NaN" : "");
        }
    }
}

/// A batched-decode graph is keyed on the exact batch shape: how many tokens,
/// which slot each belongs to, how much of that slot's cache it sees (padded, so
/// the key is stable for 256 steps), and whether it is past the indexer top-k.
static graph_build_result * acquire_batched_graph(glm_model & m, int n, const int32_t * slot_ids,
                                                  const int32_t * positions, bool * out_reused)
{
    static const bool topk_enabled = []() { const char * e = getenv("TS_GLM_TOPK"); return !(e && atoi(e) == 0); }();

    std::vector<bd_token> want((size_t) n);
    for (int i = 0; i < n; i++)
    {
        bd_token & B = want[(size_t) i];
        B.slot_id = slot_ids[i];
        B.p = positions[i];
        B.n_kv = plan_n_kv(m, B.p + 1);
        B.sparse = topk_enabled && (B.p + 1) > m.hp.indexer_top_k;
        if (B.p + 1 > m.n_ctx) return nullptr;
    }

    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        graph_build_result * e = it->get();
        if ((int) e->bd.size() != n || e->nt != n) continue;
        bool same = true;
        for (int i = 0; i < n && same; i++)
            same = e->bd[(size_t) i].slot_id == want[(size_t) i].slot_id
                && e->bd[(size_t) i].n_kv == want[(size_t) i].n_kv
                && e->bd[(size_t) i].sparse == want[(size_t) i].sparse;
        if (!same) continue;
        auto entry = std::move(*it);
        m.graph_cache.erase(it);
        graph_build_result * raw = entry.get();
        m.graph_cache.push_front(std::move(entry));
        for (int i = 0; i < n; i++) raw->bd[(size_t) i].p = want[(size_t) i].p;
        if (out_reused) *out_reused = true;
        return raw;
    }

    auto entry = std::unique_ptr<graph_build_result>(new graph_build_result());
    entry->nt = n;
    entry->n_kv = 0;
    entry->n_out = n;
    entry->want_logits = true;
    entry->slot_id = -1;
    entry->bd = want;

    size_t nodes_per_layer = 256;
    if (const char * e = getenv("TS_GLM_NODES_PER_LAYER")) { int v = atoi(e); if (v > 0) nodes_per_layer = (size_t) v; }
    // The per-token part of a layer (cache write, scoring, softmax) is built n
    // times over, so the budget grows with the batch.
    const size_t n_nodes = (size_t) m.hp.n_layer * nodes_per_layer * (size_t) std::max(1, n) + 1024;
    ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false),
                            nullptr, true };
    entry->ctx = ggml_init(gp);
    if (!entry->ctx) return nullptr;
    entry->gf = ggml_new_graph_custom(entry->ctx, n_nodes, false);
    entry->sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends,
                                          n_nodes, false, m.op_offload);
    if (!entry->sched) return nullptr;

    glm_slot & any = *m.slots.at(slot_ids[0]);
    graph_builder gb(m, *entry, any, n, 0, 0, false);
    gb.build_batched();

    if (!ggml_backend_sched_alloc_graph(entry->sched, entry->gf))
    {
        fprintf(stderr, "[glm] failed to allocate a batched graph for n=%d\n", n);
        return nullptr;
    }

    graph_build_result * raw = entry.get();
    m.graph_cache.push_front(std::move(entry));
    while ((int) m.graph_cache.size() > m.graph_cache_cap) m.graph_cache.pop_back();
    if (out_reused) *out_reused = false;
    return raw;
}

/// One decode step for n sequences at once. Every sequence contributes exactly
/// one token; the weights are read once for the whole batch, which is the only
/// reason to do this rather than n separate forwards.
static bool forward_batched_decode(glm_model & m, int n, const int32_t * slot_ids, const int32_t * tokens,
                                   const int32_t * positions, float * logits_out)
{
    if (bd_debug())
    {
        fprintf(stderr, "[glm-bd] enter n=%d", n);
        for (int i = 0; i < n; i++) fprintf(stderr, " (slot %d tok %d pos %d)", slot_ids[i], tokens[i], positions[i]);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    for (int i = 0; i < n; i++)
    {
        auto it = m.slots.find(slot_ids[i]);
        if (it == m.slots.end() || it->second->n_past != positions[i]) return false;
        for (int j = 0; j < i; j++) if (slot_ids[j] == slot_ids[i]) return false;
    }

    bd_log("checks passed\n");
    bool reused = false;
    graph_build_result * gr = acquire_batched_graph(m, n, slot_ids, positions, &reused);
    if (!gr) { bd_log("no graph\n"); return false; }
    bd_log("graph %s\n", reused ? "reused" : "built");

    const bool f16_mask = m.flash_attn;
    std::vector<ggml_fp16_t> m16;
    std::vector<float> m32;
    std::vector<int64_t> idx(1);
    std::vector<int32_t> v32((size_t) n);

    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (!live(gr->inp.tokens[d])) continue;
        for (int i = 0; i < n; i++) v32[(size_t) i] = tokens[i];
        set_input_i32(gr->inp.tokens[d], v32.data(), (size_t) n);
        for (int i = 0; i < n; i++) v32[(size_t) i] = positions[i];
        set_input_i32(gr->inp.pos[d], v32.data(), (size_t) n);
    }

    for (int i = 0; i < n; i++)
    {
        bd_token & B = gr->bd[(size_t) i];
        const int64_t p = positions[i];
        idx[0] = p;
        // Everything at or before this token's position is visible; the padding
        // past it is masked out.
        if (f16_mask)
        {
            m16.assign((size_t) B.n_kv, (ggml_fp16_t) 0xFC00);
            for (int64_t j = 0; j <= p && j < B.n_kv; j++) m16[(size_t) j] = 0;
        }
        else
        {
            m32.assign((size_t) B.n_kv, -INFINITY);
            for (int64_t j = 0; j <= p && j < B.n_kv; j++) m32[(size_t) j] = 0.0f;
        }
        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (live(B.kv_idx[d]))
                ggml_backend_tensor_set(B.kv_idx[d], idx.data(), 0, sizeof(int64_t));
            if (live(B.kq_mask[d]))
            {
                if (f16_mask) ggml_backend_tensor_set(B.kq_mask[d], m16.data(), 0, m16.size() * 2);
                else          ggml_backend_tensor_set(B.kq_mask[d], m32.data(), 0, m32.size() * 4);
            }
            if (live(B.lid_mask[d]))
            {
                // The indexer scores in F16 and never sees a padded row.
                if (!f16_mask)
                {
                    m16.assign((size_t) B.n_kv, (ggml_fp16_t) 0xFC00);
                    for (int64_t j = 0; j <= p && j < B.n_kv; j++) m16[(size_t) j] = 0;
                }
                ggml_backend_tensor_set(B.lid_mask[d], m16.data(), 0, (size_t) B.n_kv * 2);
            }
        }
    }

    bd_log("inputs set\n");
    if (ggml_backend_sched_graph_compute_async(gr->sched, gr->gf) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_sched_synchronize(gr->sched);
    bd_log("computed\n");
    print_traces(gr);

    const int64_t vocab = m.hp.n_vocab;
    for (int i = 0; i < n; i++)
    {
        ggml_backend_tensor_get(gr->logits, logits_out + (size_t) i * vocab,
                                (size_t) i * vocab * sizeof(float), (size_t) vocab * sizeof(float));
        m.slots.at(slot_ids[i])->n_past = positions[i] + 1;
    }
    bd_log("done\n");
    return true;
}

static graph_build_result * acquire_graph(glm_model & m, glm_slot & slot, int64_t nt, int64_t p0,
                                          int64_t n_out, bool want_logits, bool * out_reused)
{
    const int64_t n_kv = plan_n_kv(m, p0 + nt);
    static const bool topk_enabled = []() { const char * e = getenv("TS_GLM_TOPK"); return !(e && atoi(e) == 0); }();
    const bool sparse = topk_enabled && (p0 + nt) > m.hp.indexer_top_k;

    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        graph_build_result * e = it->get();
        if (e->nt == nt && e->n_kv == n_kv && e->n_out == n_out && e->want_logits == want_logits &&
            e->sparse == sparse && e->slot_id == slot.id)
        {
            auto entry = std::move(*it);
            m.graph_cache.erase(it);
            graph_build_result * raw = entry.get();
            m.graph_cache.push_front(std::move(entry));
            if (out_reused) *out_reused = true;
            return raw;
        }
    }

    auto entry = std::unique_ptr<graph_build_result>(new graph_build_result());
    entry->nt = nt;
    entry->n_kv = n_kv;
    entry->n_out = n_out;
    entry->sparse = sparse;
    entry->want_logits = want_logits;
    entry->slot_id = slot.id;

    // Node budget. A GLM layer costs ~110 nodes per rank (MoE alone is ~40, the
    // DSA indexer another ~20 on a full layer), so 256 per layer per rank leaves
    // room for the scheduler's own copies without making the hash set
    // pointlessly large.
    size_t nodes_per_layer = 256;
    if (const char * e = getenv("TS_GLM_NODES_PER_LAYER")) { int v = atoi(e); if (v > 0) nodes_per_layer = (size_t) v; }
    const size_t n_nodes = (size_t) m.hp.n_layer * nodes_per_layer * (size_t) std::max(1, m.tp) + 1024;
    ggml_init_params gp = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false),
                            nullptr, true };
    entry->ctx = ggml_init(gp);
    if (!entry->ctx) return nullptr;
    entry->gf = ggml_new_graph_custom(entry->ctx, n_nodes, false);

    entry->sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends,
                                          n_nodes, false, m.op_offload);
    if (!entry->sched) return nullptr;

    graph_builder gb(m, *entry, slot, nt, p0, n_kv, sparse);
    gb.build();

    // Allocate once, here, and never reset: the allocation (and, on CUDA, the
    // captured graph) is what makes a cached entry worth keeping. Note this
    // must not be preceded by ggml_backend_sched_reserve, which resets the
    // scheduler and would drop the per-device pins build() just set.
    if (!ggml_backend_sched_alloc_graph(entry->sched, entry->gf))
    {
        fprintf(stderr, "[glm] failed to allocate a graph for nt=%" PRId64 " n_kv=%" PRId64 "\n", nt, n_kv);
        return nullptr;
    }

    graph_build_result * raw = entry.get();
    m.graph_cache.push_front(std::move(entry));
    while ((int) m.graph_cache.size() > m.graph_cache_cap) m.graph_cache.pop_back();
    if (out_reused) *out_reused = false;
    return raw;
}

static bool forward_ubatch(glm_model & m, const int32_t * tokens, int64_t nt, bool want_logits, float * logits_out)
{
    glm_slot & slot = *m.active_slot;
    const int64_t p0 = slot.n_past;
    if (p0 + nt > m.n_ctx)
    {
        fprintf(stderr, "[glm] context overflow: %" PRId64 " + %" PRId64 " > %d\n", p0, nt, m.n_ctx);
        return false;
    }

    const int64_t n_out = 1;   // only the last row's logits are ever needed
    bool reused = false;
    graph_build_result * gr = acquire_graph(m, slot, nt, p0, n_out, want_logits, &reused);
    if (!gr) return false;

    const int64_t n_kv = gr->n_kv;

    m.h_tokens.assign(tokens, tokens + nt);
    m.h_pos.resize((size_t) nt);
    m.h_kv_idxs.resize((size_t) nt);
    for (int64_t i = 0; i < nt; i++)
    {
        m.h_pos[(size_t) i] = (int32_t) (p0 + i);
        m.h_kv_idxs[(size_t) i] = p0 + i;
    }
    m.h_out_ids.assign(1, (int32_t) (nt - 1));

    // Causal mask over the padded cache window.
    const bool f16_mask = m.flash_attn;
    if (f16_mask) m.h_mask_f16.assign((size_t) (n_kv * nt), 0);
    else          m.h_mask_f32.assign((size_t) (n_kv * nt), 0.0f);
    for (int64_t t = 0; t < nt; t++)
    {
        const int64_t last = p0 + t;
        for (int64_t j = 0; j < n_kv; j++)
        {
            const bool visible = j <= last;
            if (f16_mask) m.h_mask_f16[(size_t) (t * n_kv + j)] = visible ? 0 : 0xFC00 /* -inf */;
            else          m.h_mask_f32[(size_t) (t * n_kv + j)] = visible ? 0.0f : -INFINITY;
        }
    }

    for (int d = 0; d <= m.n_gpu; d++)
    {
        set_input_i32(gr->inp.tokens[d], m.h_tokens.data(), (size_t) nt);
        set_input_i32(gr->inp.pos[d], m.h_pos.data(), (size_t) nt);
        if (gr->inp.kv_idxs[d])
            if (live(gr->inp.kv_idxs[d]))
                ggml_backend_tensor_set(gr->inp.kv_idxs[d], m.h_kv_idxs.data(), 0, (size_t) nt * sizeof(int64_t));
        if (gr->inp.kq_mask[d])
        {
            if (live(gr->inp.kq_mask[d]))
            {
                if (f16_mask) ggml_backend_tensor_set(gr->inp.kq_mask[d], m.h_mask_f16.data(), 0, m.h_mask_f16.size() * 2);
                else          ggml_backend_tensor_set(gr->inp.kq_mask[d], m.h_mask_f32.data(), 0, m.h_mask_f32.size() * 4);
            }
        }
        if (gr->inp.lid_mask[d])
        {
            // The indexer always wants an F16 mask, whatever the attention uses.
            if (!f16_mask)
            {
                m.h_mask_f16.resize((size_t) (n_kv * nt));
                for (size_t i = 0; i < m.h_mask_f16.size(); i++)
                    m.h_mask_f16[i] = m.h_mask_f32[i] == 0.0f ? 0 : 0xFC00;
            }
            if (live(gr->inp.lid_mask[d]))
                ggml_backend_tensor_set(gr->inp.lid_mask[d], m.h_mask_f16.data(), 0, m.h_mask_f16.size() * 2);
        }
    }
    set_input_i32(gr->inp.out_ids, m.h_out_ids.data(), 1);


    // _async, not the synchronous entry point: the graph is already allocated
    // above (which the synchronous one asserts against), and the explicit
    // synchronize below is where the logits read waits.
    if (ggml_backend_sched_graph_compute_async(gr->sched, gr->gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[glm] graph compute failed\n");
        return false;
    }
    ggml_backend_sched_synchronize(gr->sched);

    print_traces(gr);

    slot.n_past += nt;

    if (want_logits && logits_out && gr->logits)
        ggml_backend_tensor_get(gr->logits, logits_out, 0, (size_t) m.hp.n_vocab * sizeof(float));
    return true;
}

} // namespace tsg_glm

// ---------------------------------------------------------------------------
// exported API
// ---------------------------------------------------------------------------

using namespace tsg_glm;

TSG_EXPORT void * TSGgml_GlmLoadModel(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch,
                                      int n_threads, int n_cpu_moe, const char * backend_name, int tp,
                                      int ctx_is_hard_limit)
{
    try
    {
        return glm_load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads, n_cpu_moe, backend_name, tp,
                        ctx_is_hard_limit != 0);
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[glm] load failed: %s\n", e.what());
        return nullptr;
    }
}

TSG_EXPORT int TSGgml_GlmVocabSize(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return m ? m->hp.n_vocab : 0;
}

TSG_EXPORT int TSGgml_GlmCtxSize(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return m ? m->n_ctx : 0;
}

TSG_EXPORT int TSGgml_GlmNPast(void * handle)
{
    glm_model * m = (glm_model *) handle;
    return (m && m->active_slot) ? (int) m->active_slot->n_past : 0;
}

// Evaluate n_tokens at the active slot's current position; logits_out receives
// the last token's row when it is non-null.
TSG_EXPORT int TSGgml_GlmForward(void * handle, const int32_t * tokens, int n_tokens, float * logits_out)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot || n_tokens <= 0) return 0;

    const int ub = m->n_ubatch > 0 ? m->n_ubatch : 512;
    int done = 0;
    while (done < n_tokens)
    {
        const int take = std::min(ub, n_tokens - done);
        const bool last = (done + take) == n_tokens;
        if (!forward_ubatch(*m, tokens + done, take, last && logits_out != nullptr, logits_out))
            return 0;
        done += take;
    }
    return 1;
}

/// One decode step for `n` sequences at once (one token each). Declines — by
/// returning 0 without touching any state — whenever it cannot serve the batch,
/// leaving the caller on the per-sequence path.
TSG_EXPORT int TSGgml_GlmForwardBatchedDecode(void * handle, int n, const int32_t * slot_ids,
                                              const int32_t * tokens, const int32_t * positions,
                                              float * logits_out)
{
    glm_model * m = (glm_model *) handle;
    if (!m || n < 2 || !slot_ids || !tokens || !positions || !logits_out) return 0;
    if (n > MAX_BATCHED_DECODE) return 0;
    // Tensor parallelism splits the batch's work across ranks a second time; the
    // per-sequence path already handles that case correctly, so the batched
    // graph stays single-rank rather than duplicating the reduction plumbing.
    if (m->tp > 1) return 0;
    static const bool enabled = []() { const char * e = getenv("TS_GLM_BATCHED_DECODE"); return !(e && atoi(e) == 0); }();
    if (!enabled) return 0;

    try
    {
        return forward_batched_decode(*m, n, slot_ids, tokens, positions, logits_out) ? 1 : 0;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[glm] batched decode failed: %s\n", e.what());
        return 0;
    }
}

TSG_EXPORT void TSGgml_GlmReset(void * handle)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot) return;
    m->active_slot->n_past = 0;
}

TSG_EXPORT int TSGgml_GlmRewind(void * handle, int n_past)
{
    glm_model * m = (glm_model *) handle;
    if (!m || !m->active_slot || n_past < 0 || n_past > m->active_slot->n_past) return 0;
    m->active_slot->n_past = n_past;
    return 1;
}

TSG_EXPORT int TSGgml_GlmSlotAlloc(void * handle)
{
    glm_model * m = (glm_model *) handle;
    if (!m) return -1;
    glm_slot * s = slot_alloc(*m);
    return s ? s->id : -1;
}

TSG_EXPORT int TSGgml_GlmSetActiveSlot(void * handle, int slot_id)
{
    glm_model * m = (glm_model *) handle;
    if (!m) return 0;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return 0;
    m->active_slot = it->second.get();
    return 1;
}

TSG_EXPORT int TSGgml_GlmSlotFree(void * handle, int slot_id)
{
    glm_model * m = (glm_model *) handle;
    if (!m) return 0;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return 0;
    if (m->active_slot == it->second.get())
        m->active_slot = nullptr;
    slot_free(*m, slot_id);
    if (!m->active_slot && !m->slots.empty())
        m->active_slot = m->slots.begin()->second.get();
    return 1;
}

TSG_EXPORT void TSGgml_GlmFree(void * handle)
{
    delete (glm_model *) handle;
}
