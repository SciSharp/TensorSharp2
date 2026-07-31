// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) whole-model executor for the GGML backend.
//
// Unlike the per-op fused paths used by the other architectures, this file is
// a self-contained model runtime: it loads the (possibly split) GGUF itself,
// places the per-layer weights across every visible GPU (layer split, so a
// model larger than one GPU's VRAM is hosted across all of them), owns the
// DSV4 KV caches (raw sliding-window ring + CSA/HCA compressed K caches +
// compressor state rings + lightning-indexer cache), and executes prefill /
// decode ubatches through ggml_backend_sched.
//
// The graph is a port of llama.cpp's `src/models/deepseek4.cpp` restricted to
// a single sequence / single stream:
//   * 4-stream hyper-connections (fused ggml_dsv4_hc_pre/comb/post ops).
//   * LoRA-factored Q, single shared 512-dim K(=V) head, per-head q RMS norm,
//     attention sinks, RoPE -> attention -> inverse RoPE on the rope slice,
//     grouped LoRA output projection.
//   * Per-layer compress ratio 0 (raw SWA-128 attention only), 4 (CSA:
//     overlap-compressed rows + lightning-indexer top-k selection), or 128
//     (HCA: block-compressed rows, all visible).
//   * MoE: sqrt(softplus) routing in F32, bias for selection, first
//     `hash_layer_count` layers route by token id through a tid2eid LUT,
//     swiglu clamp, weight normalization, x1.5 routed scale + shared expert.
//
// The Hadamard rotation llama.cpp applies to quantized KV caches is skipped
// on purpose: caches here are F16/F32 and the rotation is an orthonormal
// involution applied consistently to q/k/v and inverted on output, so
// skipping it everywhere is mathematically identical.
// ---------------------------------------------------------------------------

#include "ggml_ops_internal.h"
#include "ggml_ops_dsv4_fused.h"

#include "gguf.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <thread>

namespace tsg_dsv4
{

static constexpr int64_t CSA_RATIO = 4;
static constexpr int64_t HCA_RATIO = 128;
static constexpr int     MAX_GPUS  = 8;

static int64_t pad64(int64_t v, int64_t p) { return ((v + p - 1) / p) * p; }

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------
struct dsv4_hparams
{
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_head = 0;
    int32_t n_vocab = 0;
    int32_t n_embd_head = 0;      // attention.key_length (512)
    int32_t n_rot = 0;            // rope.dimension_count (64)
    int32_t q_lora_rank = 0;
    int32_t o_groups = 0;
    int32_t o_lora_rank = 0;
    int32_t n_swa = 0;            // sliding window (raw attention)
    float   rms_eps = 1e-6f;

    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_expert_shared = 0;
    int32_t n_ff_exp = 0;
    float   expert_weights_scale = 1.0f;
    bool    expert_weights_norm = false;
    int32_t hash_layer_count = 0;
    std::vector<float> swiglu_clamp_exp;
    std::vector<float> swiglu_clamp_shexp;

    int32_t indexer_n_head = 0;
    int32_t indexer_head_size = 0;
    int32_t indexer_top_k = 0;

    std::vector<int32_t> compress_ratios;
    float compress_rope_base = 10000.0f;

    int32_t hc_mult = 0;
    int32_t hc_sinkhorn_iters = 0;
    float   hc_eps = 1e-6f;

    float rope_freq_base = 10000.0f;
    float yarn_freq_scale = 1.0f;   // 1/factor
    float yarn_ext_factor = 0.0f;
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;
    int32_t n_ctx_orig = 0;
};

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor)
{
    if (ext_factor == 0.0f) return 1.0f;
    return 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------
struct dsv4_layer
{
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * attn_sinks = nullptr;
    ggml_tensor * wq_a = nullptr;
    ggml_tensor * attn_q_a_norm = nullptr;
    ggml_tensor * wq_b = nullptr;
    ggml_tensor * wkv = nullptr;
    ggml_tensor * attn_kv_norm = nullptr;
    ggml_tensor * wo_a = nullptr;
    ggml_tensor * wo_b = nullptr;

    ggml_tensor * hc_attn_fn = nullptr;
    ggml_tensor * hc_attn_base = nullptr;
    ggml_tensor * hc_attn_scale = nullptr;
    ggml_tensor * hc_ffn_fn = nullptr;
    ggml_tensor * hc_ffn_base = nullptr;
    ggml_tensor * hc_ffn_scale = nullptr;

    ggml_tensor * attn_comp_wkv = nullptr;
    ggml_tensor * attn_comp_wgate = nullptr;
    ggml_tensor * attn_comp_ape = nullptr;
    ggml_tensor * attn_comp_norm = nullptr;

    ggml_tensor * indexer_proj = nullptr;
    ggml_tensor * indexer_attn_q_b = nullptr;
    ggml_tensor * indexer_comp_wkv = nullptr;
    ggml_tensor * indexer_comp_wgate = nullptr;
    ggml_tensor * indexer_comp_ape = nullptr;
    ggml_tensor * indexer_comp_norm = nullptr;

    ggml_tensor * ffn_gate_inp = nullptr;
    ggml_tensor * ffn_gate_tid2eid = nullptr;
    ggml_tensor * ffn_exp_probs_b = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate_exps = nullptr;
    ggml_tensor * ffn_down_exps = nullptr;
    ggml_tensor * ffn_up_exps = nullptr;
    ggml_tensor * ffn_gate_shexp = nullptr;
    ggml_tensor * ffn_down_shexp = nullptr;
    ggml_tensor * ffn_up_shexp = nullptr;

    int device = 0;
};

// Per-layer KV caches / compressor state for ONE sequence slot
// (device-resident, persist across ubatches).
struct dsv4_slot_layer
{
    ggml_tensor * raw_k = nullptr;        // F16 [n_embd_head, ring_raw]
    ggml_tensor * csa_k = nullptr;        // F16 [n_embd_head, n_csa_rows]   (ratio 4)
    ggml_tensor * lid_k = nullptr;        // F16 [idx_head_size, n_csa_rows] (ratio 4)
    ggml_tensor * hca_k = nullptr;        // F16 [n_embd_head, n_hca_rows]   (ratio 128)
    ggml_tensor * comp_state_kv = nullptr;    // F32 [coff*head, state_size]
    ggml_tensor * comp_state_score = nullptr; // F32 [coff*head, state_size]
    ggml_tensor * lid_state_kv = nullptr;     // F32 [2*idx_head, 8]
    ggml_tensor * lid_state_score = nullptr;  // F32 [2*idx_head, 8]
};

// One independent sequence context: its own caches + position, sharing the
// model weights. The server's continuous-batching engine binds one slot per
// in-flight request; the CLI / single-stream path only ever uses slot 0.
struct dsv4_slot
{
    int id = 0;
    int32_t n_past = 0;
    std::vector<dsv4_slot_layer> layers;
    ggml_context * ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t buf[MAX_GPUS + 1] = {};

    ~dsv4_slot()
    {
        for (int i = 0; i <= MAX_GPUS; i++)
        {
            if (buf[i]) ggml_backend_buffer_free(buf[i]);
            if (ctx[i]) ggml_free(ctx[i]);
        }
    }
};

// gguf tensor location within the (possibly split) model
struct tensor_source
{
    int shard = -1;
    size_t offset = 0; // absolute file offset of the tensor data
    size_t size = 0;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
};

// ---------------------------------------------------------------------------
// Compression plans (single sequence port of dsv4_build_comp_plan)
// ---------------------------------------------------------------------------
struct comp_plan
{
    std::vector<int32_t> state_pos;
    std::vector<int32_t> state_persist_src_idxs;
    std::vector<int64_t> state_persist_dst_idxs;
    std::vector<int32_t> state_read_idxs;
    std::vector<int64_t> state_write_idxs;
    std::vector<int32_t> state_write_pos;
    std::vector<int32_t> n_visible;
    int64_t n_kv = 0;
};

struct plan_inputs
{
    ggml_tensor * state_pos = nullptr;      // I32 [nt]
    ggml_tensor * read_idxs = nullptr;      // I32 [...]
    ggml_tensor * write_idxs = nullptr;     // I64 [...]  (unfused)
    ggml_tensor * write_pos = nullptr;      // I32 [...]  (unfused)
    ggml_tensor * persist_src = nullptr;    // I32 [...]  (unfused)
    ggml_tensor * persist_dst = nullptr;    // I64 [...]  (unfused)
    // fused: single I32 input [read_idxs | (cache_row,rope_pos)*n_blocks | (src_col,dst_row)*np]
    ggml_tensor * comp_meta = nullptr;
    ggml_tensor * kq_mask = nullptr;        // F16 [n_kv, nt]
    int64_t n_kv = 0;
};

// Per-token inputs are duplicated per participating device (indexed by device
// id) so no scheduler-driven cross-backend input copies — each a synchronized
// host round trip — happen inside the per-token compute.
struct graph_inputs
{
    ggml_tensor * tokens[MAX_GPUS + 1] = {};    // I32 [nt]
    ggml_tensor * pos[MAX_GPUS + 1] = {};       // I32 [nt]
    ggml_tensor * raw_idxs[MAX_GPUS + 1] = {};  // I64 [nt]
    ggml_tensor * raw_mask[MAX_GPUS + 1] = {};  // F16 [ring, nt]
    // decode index-gather mode: F16 [ring + top_k, 1] (top-k tail all zeros)
    ggml_tensor * gather_mask[MAX_GPUS + 1] = {};
    ggml_tensor * out_ids = nullptr;            // I32 [1] (last device)
    plan_inputs csa[MAX_GPUS + 1];
    plan_inputs hca[MAX_GPUS + 1];
    plan_inputs lid[MAX_GPUS + 1];
};

struct graph_build_result
{
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;   // owned; keeps this graph's allocation alive
    graph_inputs inp;
    ggml_tensor * logits = nullptr;
    comp_plan plan_csa;
    comp_plan plan_hca;
    comp_plan plan_lid;
    int64_t nt = 0;
    int64_t p0 = 0;
    // sequence slot whose cache tensors this graph was built against; entries
    // are purged when their slot is freed (the baked addresses die with it)
    int slot_id = 0;
    // shape signature this graph was built for (plan array sizes + n_kv pads)
    uint64_t sig = 0;
    // descriptors referenced (by pointer) from the fused GGML_OP_CUSTOM nodes
    std::deque<tsg_dsv4_fused_desc> descs;
    // per-device completion events of this entry's last pipelined use; input
    // refills wait on these instead of a full pipeline drain
    ggml_backend_event_t use_events[MAX_GPUS] = {};

    ~graph_build_result()
    {
        for (auto & ev : use_events)
            if (ev) ggml_backend_event_free(ev);
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
    }
};

struct dsv4_model
{
    dsv4_hparams hp;

    // backends: [0..n_gpu) GPU, [n_gpu] CPU
    ggml_backend_t backends[MAX_GPUS + 1] = {};
    int n_gpu = 0;
    int n_backends = 0;

    // fused-op backends (one per GPU, launching on the CUDA backend's stream)
    ggml_backend_t ts_backends[MAX_GPUS] = {};
    bool fused = false;

    // scheduler backend list: [cuda 0..n_gpu) , (fused ts 0..n_gpu) , cpu]
    ggml_backend_t sched_backends[2 * MAX_GPUS + 1] = {};
    ggml_backend_buffer_type_t sched_bufts[2 * MAX_GPUS + 1] = {};
    int n_sched_backends = 0;

    // weight + cache contexts/buffers per device (+1 = cpu, unused normally)
    ggml_context * w_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t w_buf[MAX_GPUS + 1] = {};
    ggml_context * c_ctx[MAX_GPUS + 1] = {};
    ggml_backend_buffer_t c_buf[MAX_GPUS + 1] = {};

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * hc_head_fn = nullptr;
    ggml_tensor * hc_head_base = nullptr;
    ggml_tensor * hc_head_scale = nullptr;

    std::vector<dsv4_layer> layers;

    // geometry
    int32_t n_ctx = 0;
    int32_t n_ubatch = 0;
    int64_t ring_raw = 0;
    int64_t n_csa_rows = 0;
    int64_t n_hca_rows = 0;

    // sequence slots (per-request caches); slot 0 is the primary/single-stream
    // slot created at load. All Forward/Reset/NPast calls act on active_slot.
    std::map<int, std::unique_ptr<dsv4_slot>> slots;
    dsv4_slot * active_slot = nullptr;
    int next_slot_id = 0;

    // final position of the current TSGgml_Dsv4Forward call; keeps every
    // chunk of one prefill on one graph shape (bucketing + indexer skip)
    int64_t pos_end_hint = 0;

    // flash attention (F16 masks, fused kernel). Probed at load; TS_DSV4_FA=0 forces off.
    bool flash_attn = false;

    // RoPE cos/sin tables for the fused table-driven kernels, generated by
    // running ggml's own rope over (1,0) unit pairs so YaRN semantics and the
    // attn_factor scaling are baked in exactly. [0]=raw layers, [1]=compress
    // layers. Device copies live in the cache contexts, which Reset() clears —
    // the kept host copies are re-uploaded afterwards.
    std::vector<float> rope_tab_host[2];
    ggml_tensor * rope_tab_dev[2][MAX_GPUS + 1] = {};

    // LRU cache of built+allocated graphs keyed by shape signature. Each entry
    // owns its scheduler (and therefore its allocation), so alternating shapes
    // (decode phases, prefill chunk parity) reuse their graphs without
    // re-building, re-allocating, or thrashing ggml-cuda's captured graphs.
    std::list<std::unique_ptr<graph_build_result>> graph_cache;
    int graph_cache_cap = 12;

    // scratch for logits
    std::vector<float> logits;

    bool gather_env = true; // TS_DSV4_GATHER=0 disables decode index-gather

    ~dsv4_model()
    {
        graph_cache.clear();
        slots.clear();   // slot buffers must go before the backends below
        for (int i = 0; i <= MAX_GPUS; i++)
        {
            if (c_buf[i]) ggml_backend_buffer_free(c_buf[i]);
            if (c_ctx[i]) ggml_free(c_ctx[i]);
            if (w_buf[i]) ggml_backend_buffer_free(w_buf[i]);
            if (w_ctx[i]) ggml_free(w_ctx[i]);
        }
        for (int i = 0; i < MAX_GPUS; i++)
            if (ts_backends[i]) ggml_backend_free(ts_backends[i]);
        for (int i = 0; i < n_backends; i++)
            if (backends[i]) ggml_backend_free(backends[i]);
    }
};

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

static bool gguf_get_f32_key(gguf_context * ctx, const char * key, float * out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return false;
    *out = gguf_get_val_f32(ctx, id);
    return true;
}

static bool gguf_get_u32_key(gguf_context * ctx, const char * key, int32_t * out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return false;
    switch (gguf_get_kv_type(ctx, id))
    {
        case GGUF_TYPE_UINT32: *out = (int32_t) gguf_get_val_u32(ctx, id); return true;
        case GGUF_TYPE_INT32:  *out = gguf_get_val_i32(ctx, id); return true;
        case GGUF_TYPE_UINT64: *out = (int32_t) gguf_get_val_u64(ctx, id); return true;
        case GGUF_TYPE_INT64:  *out = (int32_t) gguf_get_val_i64(ctx, id); return true;
        case GGUF_TYPE_UINT16: *out = (int32_t) gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_INT16:  *out = (int32_t) gguf_get_val_i16(ctx, id); return true;
        case GGUF_TYPE_UINT8:  *out = (int32_t) gguf_get_val_u8(ctx, id); return true;
        case GGUF_TYPE_INT8:   *out = (int32_t) gguf_get_val_i8(ctx, id); return true;
        default: return false;
    }
}

static bool gguf_get_bool_key(gguf_context * ctx, const char * key, bool * out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return false;
    *out = gguf_get_val_bool(ctx, id);
    return true;
}

static bool gguf_get_arr_i32(gguf_context * ctx, const char * key, std::vector<int32_t> & out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return false;
    size_t n = gguf_get_arr_n(ctx, id);
    out.resize(n);
    const void * data = gguf_get_arr_data(ctx, id);
    switch (gguf_get_arr_type(ctx, id))
    {
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_UINT32:
            memcpy(out.data(), data, n * sizeof(int32_t));
            return true;
        default:
            return false;
    }
}

static bool gguf_get_arr_f32(gguf_context * ctx, const char * key, std::vector<float> & out)
{
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return false;
    size_t n = gguf_get_arr_n(ctx, id);
    out.resize(n);
    const void * data = gguf_get_arr_data(ctx, id);
    if (gguf_get_arr_type(ctx, id) != GGUF_TYPE_FLOAT32) return false;
    memcpy(out.data(), data, n * sizeof(float));
    return true;
}

struct shard_files
{
    std::vector<std::string> paths;
};

// Derive split shard paths from the first shard: "...-00001-of-000NN.gguf"
static shard_files resolve_shards(const std::string & first, int split_count)
{
    shard_files res;
    if (split_count <= 1)
    {
        res.paths.push_back(first);
        return res;
    }

    const std::string marker = "-00001-of-";
    size_t pos = first.find(marker);
    if (pos == std::string::npos)
    {
        res.paths.push_back(first);
        return res;
    }

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

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static ggml_tensor * dsv4_create_weight(
    dsv4_model & m,
    std::map<std::string, tensor_source> & sources,
    int device,
    const char * fmt, ...)
{
    char name[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(name, sizeof(name), fmt, args);
    va_end(args);

    auto it = sources.find(name);
    if (it == sources.end())
    {
        fprintf(stderr, "[dsv4] missing tensor: %s\n", name);
        return nullptr;
    }
    const tensor_source & src = it->second;
    ggml_tensor * t = ggml_new_tensor_4d(m.w_ctx[device], src.type, src.ne[0], src.ne[1], src.ne[2], src.ne[3]);
    ggml_set_name(t, name);
    return t;
}

// One chunk of one weight tensor: read [file_off, file_off+len) from its
// shard and land it at tensor_off inside the (already allocated) tensor.
struct load_job
{
    ggml_tensor * t;
    int shard;
    size_t file_off;
    size_t tensor_off;
    size_t len;
};

// Stream every job through a pool of reader threads. Model files commonly sit
// on slow or network-backed storage where one reader stream is the whole load
// time (a single thread also serializes disk reads against H2D copies).
// Each worker owns its FILE* per shard and its staging buffer, and
// ggml_backend_tensor_set copies on cudaStreamPerThread, so reads and uploads
// to all GPUs proceed concurrently. Jobs are handed out in file order per
// shard to keep the concurrent streams roughly sequential for readahead.
static bool dsv4_upload_parallel(const shard_files & shards, std::vector<load_job> & jobs, int n_threads)
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
                    fprintf(stderr, "[dsv4] cannot open %s\n", shards.paths[j.shard].c_str());
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            if (staging.size() < j.len) staging.resize(j.len);

#if defined(_WIN32)
            _fseeki64(f, (long long) j.file_off, SEEK_SET);
#else
            fseeko(f, (off_t) j.file_off, SEEK_SET);
#endif
            if (fread(staging.data(), 1, j.len, f) != j.len)
            {
                fprintf(stderr, "[dsv4] short read for %s\n", j.t->name);
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

// The overlap compressors read a synthetic "before the first block" source
// from the state rings' extra row: kv stays zero (buffer clear), the score
// row must be -inf so the per-element softmax gives those slots zero weight.
static void dsv4_init_slot_state_rows(dsv4_slot & slot)
{
    for (auto & C : slot.layers)
    {
        auto fill_neg_inf_row = [&](ggml_tensor * t)
        {
            if (!t) return;
            std::vector<float> row((size_t) t->ne[0], -INFINITY);
            ggml_backend_tensor_set(t, row.data(), (size_t) (t->ne[1] - 1) * t->nb[1], row.size() * sizeof(float));
        };
        fill_neg_inf_row(C.comp_state_score);
        fill_neg_inf_row(C.lid_state_score);
    }
}

// Clear a slot's caches back to the fresh state (position 0).
static void dsv4_reset_slot(dsv4_slot & slot)
{
    slot.n_past = 0;
    for (int d = 0; d <= MAX_GPUS; d++)
        if (slot.buf[d]) ggml_backend_buffer_clear(slot.buf[d], 0);
    dsv4_init_slot_state_rows(slot);
}

// Allocate a new sequence slot: per-layer caches + compressor state rings on
// each layer's device. Weights and rope tables are shared across slots.
// Returns nullptr on allocation failure (e.g. VRAM exhausted).
static dsv4_slot * dsv4_slot_alloc(dsv4_model & m)
{
    const dsv4_hparams & hp = m.hp;
    auto slot = std::make_unique<dsv4_slot>();
    slot->id = m.next_slot_id++;
    slot->layers.resize(hp.n_layer);

    for (int d = 0; d <= m.n_gpu; d++)
    {
        ggml_init_params cp = { (size_t) (hp.n_layer * 10 + 16) * ggml_tensor_overhead(), nullptr, true };
        slot->ctx[d] = ggml_init(cp);
        if (!slot->ctx[d]) return nullptr;
    }

    for (int il = 0; il < hp.n_layer; il++)
    {
        const int d = m.layers[il].device;
        const int ratio = hp.compress_ratios[il];
        dsv4_slot_layer & C = slot->layers[il];
        ggml_context * ctx = slot->ctx[d];

        const int64_t head = hp.n_embd_head;
        C.raw_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, m.ring_raw);
        ggml_format_name(C.raw_k, "cache_raw_k.%d.%d", slot->id, il);
        // State rings carry one extra row (index state_size) that stays
        // zero (kv) / -inf (score): the overlap compressor's synthetic
        // "before the first block" source. It is never written by persists
        // (dst = pos % state_size < state_size), so no per-eval zero-row
        // append is needed in the graph.
        if (ratio == CSA_RATIO)
        {
            C.csa_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, m.n_csa_rows);
            ggml_format_name(C.csa_k, "cache_csa_k.%d.%d", slot->id, il);
            C.lid_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hp.indexer_head_size, m.n_csa_rows);
            ggml_format_name(C.lid_k, "cache_lid_k.%d.%d", slot->id, il);
            C.comp_state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * head, 2 * CSA_RATIO + 1);
            C.comp_state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * head, 2 * CSA_RATIO + 1);
            C.lid_state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * hp.indexer_head_size, 2 * CSA_RATIO + 1);
            C.lid_state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * hp.indexer_head_size, 2 * CSA_RATIO + 1);
            ggml_format_name(C.comp_state_kv, "state_csa_kv.%d.%d", slot->id, il);
            ggml_format_name(C.comp_state_score, "state_csa_score.%d.%d", slot->id, il);
            ggml_format_name(C.lid_state_kv, "state_lid_kv.%d.%d", slot->id, il);
            ggml_format_name(C.lid_state_score, "state_lid_score.%d.%d", slot->id, il);
        }
        else if (ratio == HCA_RATIO)
        {
            C.hca_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head, m.n_hca_rows);
            ggml_format_name(C.hca_k, "cache_hca_k.%d.%d", slot->id, il);
            C.comp_state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head, HCA_RATIO + 1);
            C.comp_state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head, HCA_RATIO + 1);
            ggml_format_name(C.comp_state_kv, "state_hca_kv.%d.%d", slot->id, il);
            ggml_format_name(C.comp_state_score, "state_hca_score.%d.%d", slot->id, il);
        }
    }

    for (int d = 0; d <= m.n_gpu; d++)
    {
        if (ggml_get_first_tensor(slot->ctx[d]) == nullptr) continue;
        slot->buf[d] = ggml_backend_alloc_ctx_tensors(slot->ctx[d], m.backends[d]);
        if (!slot->buf[d])
        {
            fprintf(stderr, "[dsv4] slot %d cache alloc failed on device %d\n", slot->id, d);
            return nullptr;
        }
        ggml_backend_buffer_clear(slot->buf[d], 0);
    }
    dsv4_init_slot_state_rows(*slot);

    dsv4_slot * raw = slot.get();
    m.slots[slot->id] = std::move(slot);
    return raw;
}

// Build the [n_rot, n_ctx] cos/sin table by running ggml's own rope over
// (1,0) unit pairs on the CPU backend: out[pos, 2i+0] = cos_i(pos),
// out[pos, 2i+1] = sin_i(pos), with YaRN correction and attn_factor scaling
// exactly as the graph-level ggml_rope_ext computes them.
static bool dsv4_build_rope_table_host(dsv4_model & m, bool comp, std::vector<float> & out)
{
    const dsv4_hparams & hp = m.hp;
    const int64_t n_rot = hp.n_rot;
    const int64_t n_pos = m.n_ctx;

    const float freq_base   = comp ? hp.compress_rope_base : hp.rope_freq_base;
    const float freq_scale  = comp ? hp.yarn_freq_scale : 1.0f;
    const float ext_factor  = comp ? hp.yarn_ext_factor : 0.0f;
    const float attn_factor = dsv4_rope_attn_factor(freq_scale, ext_factor);
    const float beta_fast   = comp ? hp.yarn_beta_fast : 0.0f;
    const float beta_slow   = comp ? hp.yarn_beta_slow : 0.0f;
    const int   n_ctx_orig  = comp ? hp.n_ctx_orig : 0;

    const size_t need = (size_t) (n_rot * n_pos) * sizeof(float) * 2
        + (size_t) n_pos * sizeof(int32_t)
        + 16 * ggml_tensor_overhead() + ggml_graph_overhead() + (1u << 20);
    ggml_init_params ip = { need, nullptr, false };
    ggml_context * c = ggml_init(ip);
    if (!c) return false;

    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_rot, 1, n_pos);
    float * xd = (float *) x->data;
    for (int64_t p = 0; p < n_pos; p++)
        for (int64_t i = 0; i < n_rot; i += 2)
        {
            xd[p * n_rot + i] = 1.0f;
            xd[p * n_rot + i + 1] = 0.0f;
        }

    ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_pos);
    int32_t * pd = (int32_t *) pos->data;
    for (int64_t p = 0; p < n_pos; p++) pd[p] = (int32_t) p;

    ggml_tensor * r = ggml_rope_ext(c, x, pos, nullptr, (int) n_rot, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_cgraph * g = ggml_new_graph(c);
    ggml_build_forward_expand(g, r);
    if (ggml_backend_graph_compute(m.backends[m.n_gpu], g) != GGML_STATUS_SUCCESS)
    {
        ggml_free(c);
        return false;
    }

    const float * rd = (const float *) r->data;
    out.assign(rd, rd + (size_t) (n_rot * n_pos));
    ggml_free(c);
    return true;
}

static void dsv4_upload_rope_tables(dsv4_model & m)
{
    for (int k = 0; k < 2; k++)
    {
        if (m.rope_tab_host[k].empty()) continue;
        for (int d = 0; d < m.n_gpu; d++)
            if (m.rope_tab_dev[k][d])
                ggml_backend_tensor_set(m.rope_tab_dev[k][d], m.rope_tab_host[k].data(), 0,
                                        m.rope_tab_host[k].size() * sizeof(float));
    }
}

static dsv4_model * dsv4_load(const char * gguf_path, int n_gpu_req, int n_ctx, int n_ubatch, int n_threads)
{
    auto t_start = std::chrono::steady_clock::now();

    std::unique_ptr<dsv4_model> m(new dsv4_model());

    // --- backends ---
    int n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count() && n_gpu < MAX_GPUS; i++)
    {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        if (n_gpu_req > 0 && n_gpu >= n_gpu_req) break;
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) continue;
        m->backends[n_gpu++] = be;
    }
    if (n_gpu == 0)
    {
        fprintf(stderr, "[dsv4] no GPU backend available; refusing CPU-only run for a %s\n", "250B model");
        return nullptr;
    }
    m->n_gpu = n_gpu;
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_cpu_set_n_threads(cpu, n_threads > 0 ? n_threads : 16);
    m->backends[n_gpu] = cpu;
    m->n_backends = n_gpu + 1;

    // --- fused-op backends (kernel-count is the decode wall; the fused ops
    // collapse the small-op chains, injected via GGML_OP_CUSTOM nodes) ---
    {
        const char * fe = getenv("TS_DSV4_FUSED");
        bool want_fused = !(fe && atoi(fe) == 0);
#ifdef TSG_GGML_USE_CUDA
        if (want_fused)
        {
            bool all_ok = true;
            for (int d = 0; d < n_gpu; d++)
            {
                m->ts_backends[d] = tsg_dsv4_fused_backend_init(m->backends[d]);
                if (!m->ts_backends[d]) { all_ok = false; break; }
            }
            m->fused = all_ok;
        }
#else
        (void) want_fused;
#endif
        int nb = 0;
        for (int d = 0; d < n_gpu; d++) m->sched_backends[nb++] = m->backends[d];
        if (m->fused)
            for (int d = 0; d < n_gpu; d++) m->sched_backends[nb++] = m->ts_backends[d];
        m->sched_backends[nb++] = cpu;
        for (int i = 0; i < nb; i++)
            m->sched_bufts[i] = ggml_backend_get_default_buffer_type(m->sched_backends[i]);
        m->n_sched_backends = nb;

        const char * gc = getenv("TS_DSV4_GRAPH_CACHE");
        if (gc && atoi(gc) > 0) m->graph_cache_cap = atoi(gc);

        const char * ge = getenv("TS_DSV4_GATHER");
        m->gather_env = !(ge && atoi(ge) == 0);
    }

    // --- read gguf metadata (all shards) ---
    ggml_context * meta_ctx0 = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, &meta_ctx0 };
    gguf_context * g0 = gguf_init_from_file(gguf_path, ip);
    if (!g0)
    {
        fprintf(stderr, "[dsv4] failed to open %s\n", gguf_path);
        return nullptr;
    }

    int32_t split_count = 1;
    gguf_get_u32_key(g0, "split.count", &split_count);
    shard_files shards = resolve_shards(gguf_path, split_count);

    dsv4_hparams & hp = m->hp;
    bool ok = true;
    ok &= gguf_get_u32_key(g0, "deepseek4.block_count", &hp.n_layer);
    ok &= gguf_get_u32_key(g0, "deepseek4.embedding_length", &hp.n_embd);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.head_count", &hp.n_head);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.key_length", &hp.n_embd_head);
    ok &= gguf_get_u32_key(g0, "deepseek4.rope.dimension_count", &hp.n_rot);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.q_lora_rank", &hp.q_lora_rank);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.output_group_count", &hp.o_groups);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.output_lora_rank", &hp.o_lora_rank);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.sliding_window", &hp.n_swa);
    ok &= gguf_get_f32_key(g0, "deepseek4.attention.layer_norm_rms_epsilon", &hp.rms_eps);
    ok &= gguf_get_u32_key(g0, "deepseek4.expert_count", &hp.n_expert);
    ok &= gguf_get_u32_key(g0, "deepseek4.expert_used_count", &hp.n_expert_used);
    ok &= gguf_get_u32_key(g0, "deepseek4.expert_shared_count", &hp.n_expert_shared);
    ok &= gguf_get_u32_key(g0, "deepseek4.expert_feed_forward_length", &hp.n_ff_exp);
    ok &= gguf_get_f32_key(g0, "deepseek4.expert_weights_scale", &hp.expert_weights_scale);
    ok &= gguf_get_bool_key(g0, "deepseek4.expert_weights_norm", &hp.expert_weights_norm);
    ok &= gguf_get_u32_key(g0, "deepseek4.hash_layer_count", &hp.hash_layer_count);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.indexer.head_count", &hp.indexer_n_head);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.indexer.key_length", &hp.indexer_head_size);
    ok &= gguf_get_u32_key(g0, "deepseek4.attention.indexer.top_k", &hp.indexer_top_k);
    ok &= gguf_get_arr_i32(g0, "deepseek4.attention.compress_ratios", hp.compress_ratios);
    ok &= gguf_get_f32_key(g0, "deepseek4.attention.compress_rope_freq_base", &hp.compress_rope_base);
    ok &= gguf_get_u32_key(g0, "deepseek4.hyper_connection.count", &hp.hc_mult);
    ok &= gguf_get_u32_key(g0, "deepseek4.hyper_connection.sinkhorn_iterations", &hp.hc_sinkhorn_iters);
    ok &= gguf_get_f32_key(g0, "deepseek4.hyper_connection.epsilon", &hp.hc_eps);
    gguf_get_arr_f32(g0, "deepseek4.swiglu_clamp_exp", hp.swiglu_clamp_exp);
    if (!gguf_get_arr_f32(g0, "deepseek4.swiglu_clamp_shexp", hp.swiglu_clamp_shexp))
        hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp;
    gguf_get_f32_key(g0, "deepseek4.rope.freq_base", &hp.rope_freq_base);

    float yarn_factor = 0.0f;
    if (gguf_get_f32_key(g0, "deepseek4.rope.scaling.factor", &yarn_factor) && yarn_factor > 0.0f)
    {
        hp.yarn_freq_scale = 1.0f / yarn_factor;
        hp.yarn_ext_factor = 1.0f;
    }
    gguf_get_u32_key(g0, "deepseek4.rope.scaling.original_context_length", &hp.n_ctx_orig);
    gguf_get_f32_key(g0, "deepseek4.rope.scaling.yarn_beta_fast", &hp.yarn_beta_fast);
    gguf_get_f32_key(g0, "deepseek4.rope.scaling.yarn_beta_slow", &hp.yarn_beta_slow);

    if (!ok || hp.n_layer <= 0 || (int) hp.compress_ratios.size() < hp.n_layer)
    {
        fprintf(stderr, "[dsv4] missing/invalid deepseek4 metadata\n");
        gguf_free(g0);
        if (meta_ctx0) ggml_free(meta_ctx0);
        return nullptr;
    }
    gguf_free(g0);
    if (meta_ctx0) { ggml_free(meta_ctx0); meta_ctx0 = nullptr; }

    // --- gather tensor sources across shards ---
    std::map<std::string, tensor_source> sources;
    std::vector<size_t> layer_bytes(hp.n_layer, 0);
    size_t root_bytes = 0;

    for (size_t si = 0; si < shards.paths.size(); si++)
    {
        ggml_context * meta = nullptr;
        gguf_init_params sp = { true, &meta };
        gguf_context * g = gguf_init_from_file(shards.paths[si].c_str(), sp);
        if (!g)
        {
            fprintf(stderr, "[dsv4] failed to open shard %s\n", shards.paths[si].c_str());
            return nullptr;
        }
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
                layer_bytes[bid] += src.size;
            else
                root_bytes += src.size;
        }
        gguf_free(g);
        ggml_free(meta);
    }

    // --- vocab size from tok_embd ---
    {
        auto it = sources.find("token_embd.weight");
        if (it == sources.end()) { fprintf(stderr, "[dsv4] token_embd missing\n"); return nullptr; }
        hp.n_vocab = (int32_t) it->second.ne[1];
    }

    // --- geometry ---
    m->n_ctx = n_ctx > 0 ? n_ctx : 16384;
    m->n_ubatch = n_ubatch > 0 ? n_ubatch : 512;
    m->ring_raw = pad64(hp.n_swa + m->n_ubatch, 256);
    // +1 so the masked scratch row (last row) used by non-boundary CSA/LID
    // decode steps never collides with a real compressed row.
    m->n_csa_rows = pad64(m->n_ctx / CSA_RATIO + 1, 256);
    m->n_hca_rows = pad64(m->n_ctx / HCA_RATIO + 1, 256);

    // --- layer -> device split by cumulative weight bytes ---
    size_t total_bytes = root_bytes;
    for (auto b : layer_bytes) total_bytes += b;
    m->layers.resize(hp.n_layer);
    {
        size_t acc = 0;
        for (int il = 0; il < hp.n_layer; il++)
        {
            int dev = (int) ((acc * n_gpu) / (total_bytes + 1));
            if (dev >= n_gpu) dev = n_gpu - 1;
            m->layers[il].device = dev;
            acc += layer_bytes[il];
        }
    }

    // --- create weight contexts ---
    for (int d = 0; d <= n_gpu; d++)
    {
        ggml_init_params wp = { (size_t) 4096 * ggml_tensor_overhead(), nullptr, true };
        m->w_ctx[d] = ggml_init(wp);
        ggml_init_params cp = { (size_t) 4096 * ggml_tensor_overhead(), nullptr, true };
        m->c_ctx[d] = ggml_init(cp);
    }

    const int dev_first = m->layers[0].device;
    const int dev_last = m->layers[hp.n_layer - 1].device;

    auto W = [&](int device, const char * fmt, auto... args) -> ggml_tensor *
    {
        return dsv4_create_weight(*m, sources, device, fmt, args...);
    };

    m->tok_embd = W(dev_first, "token_embd.weight");
    m->output_norm = W(dev_last, "output_norm.weight");
    m->output = W(dev_last, "output.weight");
    m->hc_head_fn = W(dev_last, "output_hc_fn.weight");
    m->hc_head_base = W(dev_last, "output_hc_base.weight");
    m->hc_head_scale = W(dev_last, "output_hc_scale.weight");
    if (!m->tok_embd || !m->output_norm || !m->output || !m->hc_head_fn || !m->hc_head_base || !m->hc_head_scale)
        return nullptr;

    for (int il = 0; il < hp.n_layer; il++)
    {
        dsv4_layer & L = m->layers[il];
        const int d = L.device;
        const int ratio = hp.compress_ratios[il];

        L.attn_norm = W(d, "blk.%d.attn_norm.weight", il);
        L.attn_sinks = W(d, "blk.%d.attn_sinks.weight", il);
        L.wq_a = W(d, "blk.%d.attn_q_a.weight", il);
        L.attn_q_a_norm = W(d, "blk.%d.attn_q_a_norm.weight", il);
        L.wq_b = W(d, "blk.%d.attn_q_b.weight", il);
        L.wkv = W(d, "blk.%d.attn_kv.weight", il);
        L.attn_kv_norm = W(d, "blk.%d.attn_kv_a_norm.weight", il);
        L.wo_a = W(d, "blk.%d.attn_output_a.weight", il);
        L.wo_b = W(d, "blk.%d.attn_output_b.weight", il);

        L.hc_attn_fn = W(d, "blk.%d.hc_attn_fn.weight", il);
        L.hc_attn_base = W(d, "blk.%d.hc_attn_base.weight", il);
        L.hc_attn_scale = W(d, "blk.%d.hc_attn_scale.weight", il);
        L.hc_ffn_fn = W(d, "blk.%d.hc_ffn_fn.weight", il);
        L.hc_ffn_base = W(d, "blk.%d.hc_ffn_base.weight", il);
        L.hc_ffn_scale = W(d, "blk.%d.hc_ffn_scale.weight", il);

        if (ratio != 0)
        {
            L.attn_comp_wkv = W(d, "blk.%d.attn_compressor_kv.weight", il);
            L.attn_comp_wgate = W(d, "blk.%d.attn_compressor_gate.weight", il);
            L.attn_comp_ape = W(d, "blk.%d.attn_compressor_ape.weight", il);
            L.attn_comp_norm = W(d, "blk.%d.attn_compressor_norm.weight", il);
            if (ratio == CSA_RATIO)
            {
                L.indexer_proj = W(d, "blk.%d.indexer.proj.weight", il);
                L.indexer_attn_q_b = W(d, "blk.%d.indexer.attn_q_b.weight", il);
                L.indexer_comp_wkv = W(d, "blk.%d.indexer_compressor_kv.weight", il);
                L.indexer_comp_wgate = W(d, "blk.%d.indexer_compressor_gate.weight", il);
                L.indexer_comp_ape = W(d, "blk.%d.indexer_compressor_ape.weight", il);
                L.indexer_comp_norm = W(d, "blk.%d.indexer_compressor_norm.weight", il);
            }
        }

        L.ffn_gate_inp = W(d, "blk.%d.ffn_gate_inp.weight", il);
        if (il < hp.hash_layer_count)
            L.ffn_gate_tid2eid = W(d, "blk.%d.ffn_gate_tid2eid.weight", il);
        else
            L.ffn_exp_probs_b = W(d, "blk.%d.exp_probs_b.bias", il);
        L.ffn_norm = W(d, "blk.%d.ffn_norm.weight", il);
        L.ffn_gate_exps = W(d, "blk.%d.ffn_gate_exps.weight", il);
        L.ffn_down_exps = W(d, "blk.%d.ffn_down_exps.weight", il);
        L.ffn_up_exps = W(d, "blk.%d.ffn_up_exps.weight", il);
        L.ffn_gate_shexp = W(d, "blk.%d.ffn_gate_shexp.weight", il);
        L.ffn_down_shexp = W(d, "blk.%d.ffn_down_shexp.weight", il);
        L.ffn_up_shexp = W(d, "blk.%d.ffn_up_shexp.weight", il);

        if (!L.attn_norm || !L.attn_sinks || !L.wq_a || !L.wq_b || !L.wkv || !L.wo_a || !L.wo_b ||
            !L.ffn_gate_inp || !L.ffn_norm || !L.ffn_gate_exps || !L.ffn_down_exps || !L.ffn_up_exps)
        {
            fprintf(stderr, "[dsv4] layer %d incomplete\n", il);
            return nullptr;
        }
    }

    // rope cos/sin tables for the fused table-driven kernels (per device)
    if (m->fused)
    {
        for (int d = 0; d < n_gpu; d++)
        {
            for (int k = 0; k < 2; k++)
            {
                m->rope_tab_dev[k][d] = ggml_new_tensor_2d(m->c_ctx[d], GGML_TYPE_F32, hp.n_rot, m->n_ctx);
                ggml_format_name(m->rope_tab_dev[k][d], "rope_tab_%s.%d", k == 0 ? "raw" : "comp", d);
            }
        }
    }

    // --- allocate + upload ---
    for (int d = 0; d <= n_gpu; d++)
    {
        if (ggml_get_first_tensor(m->w_ctx[d]) != nullptr)
        {
            m->w_buf[d] = ggml_backend_alloc_ctx_tensors(m->w_ctx[d], m->backends[d]);
            if (!m->w_buf[d]) { fprintf(stderr, "[dsv4] weight alloc failed on device %d\n", d); return nullptr; }
        }
        if (ggml_get_first_tensor(m->c_ctx[d]) != nullptr)
        {
            m->c_buf[d] = ggml_backend_alloc_ctx_tensors(m->c_ctx[d], m->backends[d]);
            if (!m->c_buf[d]) { fprintf(stderr, "[dsv4] cache alloc failed on device %d\n", d); return nullptr; }
            ggml_backend_buffer_clear(m->c_buf[d], 0);
        }
    }

    {
        // Chunk size balances two pressures: big reads amortize network-FS
        // round trips, small chunks keep all reader threads busy at the tail.
        size_t chunk = (size_t) 64 * 1024 * 1024;
        if (const char * e = getenv("TS_DSV4_LOAD_CHUNK_MB")) { long v = atol(e); if (v > 0) chunk = (size_t) v * 1024 * 1024; }

        int load_threads = 16;
        if (const char * e = getenv("TS_DSV4_LOAD_THREADS")) { int v = atoi(e); if (v > 0) load_threads = v; }
        unsigned hw = std::thread::hardware_concurrency();
        if (hw > 0 && load_threads > (int) hw) load_threads = (int) hw;

        std::vector<load_job> jobs;
        size_t uploaded = 0;
        bool sizes_ok = true;
        for (int d = 0; d <= n_gpu; d++)
        {
            ggml_context * ctx = m->w_ctx[d];
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
            {
                auto it = sources.find(t->name);
                if (it == sources.end()) continue; // cache tensors etc.
                const tensor_source & src = it->second;
                const size_t total = ggml_nbytes(t);
                if (total != src.size)
                {
                    fprintf(stderr, "[dsv4] size mismatch for %s: tensor %zu vs file %zu\n", t->name, total, src.size);
                    sizes_ok = false;
                    continue;
                }
                for (size_t off = 0; off < total; off += chunk)
                    jobs.push_back({ t, src.shard, src.offset + off, off, std::min(chunk, total - off) });
                uploaded += total;
            }
        }
        if (!sizes_ok) return nullptr;
        if (!dsv4_upload_parallel(shards, jobs, load_threads)) return nullptr;

        auto t_end = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(stderr, "[dsv4] loaded %.1f GiB across %d GPU(s) in %.1fs (%.2f GiB/s, %d load threads; n_ctx=%d, ring=%" PRId64 ", csa_rows=%" PRId64 ", hca_rows=%" PRId64 ")\n",
                uploaded / (1024.0 * 1024.0 * 1024.0), n_gpu, secs,
                secs > 0 ? uploaded / (1024.0 * 1024.0 * 1024.0) / secs : 0.0,
                load_threads, m->n_ctx, m->ring_raw, m->n_csa_rows, m->n_hca_rows);
    }

    // primary sequence slot (slot 0) — the CLI / single-stream cache
    m->active_slot = dsv4_slot_alloc(*m);
    if (!m->active_slot)
    {
        fprintf(stderr, "[dsv4] primary slot allocation failed\n");
        return nullptr;
    }

    if (m->fused)
    {
        if (!dsv4_build_rope_table_host(*m, false, m->rope_tab_host[0]) ||
            !dsv4_build_rope_table_host(*m, true, m->rope_tab_host[1]))
        {
            fprintf(stderr, "[dsv4] rope table build failed; disabling fused ops\n");
            m->fused = false;
        }
        else
        {
            dsv4_upload_rope_tables(*m);
        }
        fprintf(stderr, "[dsv4] fused ops: %s\n", m->fused ? "on" : "off");
    }

    // --- flash attention probe ---
    {
        const char * fa_env = getenv("TS_DSV4_FA");
        const bool want_fa = !(fa_env && atoi(fa_env) == 0);
        if (want_fa)
        {
            ggml_init_params pp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
            ggml_context * pctx = ggml_init(pp);
            ggml_tensor * q = ggml_new_tensor_4d(pctx, GGML_TYPE_F32, hp.n_embd_head, 1, hp.n_head, 1);
            ggml_tensor * k = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, hp.n_embd_head, 256, 1, 1);
            ggml_tensor * mask = ggml_new_tensor_4d(pctx, GGML_TYPE_F16, 256, 1, 1, 1);
            ggml_tensor * fa = ggml_flash_attn_ext(pctx, q, k, k, mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            m->flash_attn = ggml_backend_supports_op(m->backends[0], fa);
            ggml_free(pctx);
        }
        fprintf(stderr, "[dsv4] flash attention: %s\n", m->flash_attn ? "on" : "off");
    }

    m->logits.resize(hp.n_vocab);
    return m.release();
}

// Decode-time index-gather sparse attention: past the indexer-skip horizon
// (pos_end > top_k * ratio) the decode token sees n_visible >= top_k
// compressed rows, so every top-k index is a valid row and CSA attention can
// read a compact [raw ring | gathered top-k] K with an all-visible tail mask
// instead of masked-dense over every compressed row. Attention cost then
// stops growing with context; only the indexer scores/top-k stay O(n).
static bool dsv4_use_gather(const dsv4_model & m, int64_t nt, int64_t pos_end)
{
    return m.fused && m.gather_env && nt == 1 &&
           pos_end > (int64_t) m.hp.indexer_top_k * CSA_RATIO;
}

static comp_plan build_comp_plan(
    int64_t p0, int64_t nt,
    int64_t ratio, bool overlap,
    int64_t state_size, int64_t kv_rows,
    int64_t hint_rows = 0)
{
    comp_plan plan;
    plan.n_visible.resize(nt);

    struct persist_row { int64_t dst; int32_t src; int64_t pos; };
    std::vector<persist_row> persist;

    std::vector<int32_t> prev_reads, cur_reads;

    // Source row space: [0, state_size) persistent ring, state_size = the
    // ring's built-in zero/-inf row, [state_size+1, ...) current-ubatch scratch.
    auto src_idx = [&](int64_t pos) -> int32_t
    {
        if (pos < 0) return (int32_t) state_size;                       // built-in zero/-inf row
        if (pos >= p0) return (int32_t) (state_size + 1 + (pos - p0));  // current-ubatch scratch
        return (int32_t) (pos % state_size);                            // persistent ring
    };

    for (int64_t i = 0; i < nt; i++)
    {
        const int64_t pos = p0 + i;
        plan.state_pos.push_back((int32_t) (pos % ratio));
        const int64_t n_visible = (pos + 1) / ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        const int64_t state_idx = pos % state_size;
        auto it = std::find_if(persist.begin(), persist.end(), [&](const persist_row & r) { return r.dst == state_idx; });
        if (it == persist.end()) persist.push_back({ state_idx, (int32_t) i, pos });
        else if (pos > it->pos) { it->src = (int32_t) i; it->pos = pos; }

        if ((pos + 1) % ratio != 0) continue;

        const int64_t source_start = pos + 1 - ratio;
        plan.state_write_idxs.push_back(pos / ratio);
        plan.state_write_pos.push_back((int32_t) source_start);

        if (overlap)
        {
            const int64_t prev_start = source_start - ratio;
            for (int64_t j = 0; j < ratio; j++) prev_reads.push_back(src_idx(prev_start + j));
            for (int64_t j = 0; j < ratio; j++) cur_reads.push_back(src_idx(source_start + j));
        }
        else
        {
            for (int64_t j = 0; j < ratio; j++) plan.state_read_idxs.push_back(src_idx(source_start + j));
        }
    }

    // CSA-rate plans keep the graph shape stable on non-boundary steps by
    // compressing into a masked scratch row (last cache row).
    if (ratio == CSA_RATIO && plan.state_write_idxs.empty() && !plan.state_pos.empty())
    {
        const int32_t source_idx = src_idx(p0);
        plan.state_write_idxs.push_back(kv_rows - 1);
        plan.state_write_pos.push_back(0);
        if (overlap)
        {
            for (int64_t j = 0; j < ratio; j++) { prev_reads.push_back(source_idx); cur_reads.push_back(source_idx); }
        }
        else
        {
            for (int64_t j = 0; j < ratio; j++) plan.state_read_idxs.push_back(source_idx);
        }
    }

    if (overlap)
    {
        plan.state_read_idxs.insert(plan.state_read_idxs.end(), prev_reads.begin(), prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(), cur_reads.begin(), cur_reads.end());
    }

    // Power-of-two n_kv buckets (min 256): the graph shape then only changes
    // at bucket doublings, so the graph cache and the pipelined prefill reuse
    // entries across long position stretches instead of re-building (and
    // device-synchronizing on fresh arena allocations) every 1024 positions.
    // hint_rows (the whole forward call's final row count, capped by the
    // caller) makes every chunk of one prefill share a single bucket.
    {
        int64_t bucket = 256;
        const int64_t needed = std::max<int64_t>(std::max(plan.n_kv, hint_rows), 1);
        while (bucket < needed) bucket <<= 1;
        plan.n_kv = std::min(bucket, kv_rows);
    }

    std::sort(persist.begin(), persist.end(), [](const persist_row & a, const persist_row & b) { return a.dst < b.dst; });
    for (const auto & r : persist)
    {
        plan.state_persist_src_idxs.push_back(r.src);
        plan.state_persist_dst_idxs.push_back(r.dst);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Graph building
// ---------------------------------------------------------------------------

struct graph_builder
{
    dsv4_model & m;
    const dsv4_hparams & hp;
    ggml_context * ctx;
    ggml_cgraph * gf;
    graph_build_result & res;
    int64_t nt;
    int64_t p0;

    graph_builder(dsv4_model & model, graph_build_result & r, int64_t nt_, int64_t p0_)
        : m(model), hp(model.hp), ctx(r.ctx), gf(r.gf), res(r), nt(nt_), p0(p0_) {}

    ggml_tensor * rms(ggml_tensor * x, ggml_tensor * w)
    {
        x = ggml_rms_norm(ctx, x, hp.rms_eps);
        if (w) x = ggml_mul(ctx, x, w);
        return x;
    }

    // Inputs are pinned to the device that consumes them so the scheduler
    // never inserts per-token synchronized cross-backend input copies.
    void pin(ggml_tensor * t, int dev)
    {
        ggml_backend_sched_set_tensor_backend(res.sched, t, m.backends[dev]);
    }

    ggml_tensor * new_input_i32(int64_t n, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    ggml_tensor * new_input_i64(int64_t n, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    // Exact [n_kv, nt] — ggml_lightning_indexer asserts mask ne1 == nt, and the
    // non-flash soft_max path accepts unpadded masks.
    ggml_tensor * new_input_mask(int64_t n_kv, ggml_type type, const char * name, int dev)
    {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n_kv, nt);
        ggml_set_input(t);
        ggml_set_name(t, name);
        pin(t, dev);
        return t;
    }

    // Emit a fused GGML_OP_CUSTOM node handled by the TensorSharp fused-op
    // backend of `dev` (CUDA) or by the CPU reference fallback.
    ggml_tensor * fused_node(int kind, ggml_type type,
                             int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                             std::initializer_list<ggml_tensor *> args, int dev,
                             int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0,
                             float f0 = 0.0f, float f1 = 0.0f)
    {
        res.descs.emplace_back();
        tsg_dsv4_fused_desc & d = res.descs.back();
        d.kind = kind;
        d.i0 = i0; d.i1 = i1; d.i2 = i2; d.i3 = i3;
        d.f0 = f0; d.f1 = f1;

        ggml_tensor * a[GGML_MAX_SRC];
        int n = 0;
        for (ggml_tensor * t : args) a[n++] = t;

        ggml_tensor * out = ggml_custom_4d(ctx, type, ne0, ne1, ne2, ne3, a, n, tsg_dsv4_fused_cpu, 1, &d);
        if (m.ts_backends[dev])
            ggml_backend_sched_set_tensor_backend(res.sched, out, m.ts_backends[dev]);
        return out;
    }

    // rope parameters for a layer
    void rope_params(int il, float & freq_base, float & freq_scale, float & ext_factor,
                     float & attn_factor, float & beta_fast, float & beta_slow, int & n_ctx_orig) const
    {
        const bool comp = hp.compress_ratios[il] != 0;
        freq_base = comp ? hp.compress_rope_base : hp.rope_freq_base;
        freq_scale = comp ? hp.yarn_freq_scale : 1.0f;
        ext_factor = comp ? hp.yarn_ext_factor : 0.0f;
        attn_factor = dsv4_rope_attn_factor(freq_scale, ext_factor);
        beta_fast = comp ? hp.yarn_beta_fast : 0.0f;
        beta_slow = comp ? hp.yarn_beta_slow : 0.0f;
        n_ctx_orig = comp ? hp.n_ctx_orig : 0;
    }

    ggml_tensor * rope_compress(ggml_tensor * x, ggml_tensor * pos)
    {
        return ggml_rope_ext(ctx, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NORMAL, hp.n_ctx_orig,
                             hp.compress_rope_base, hp.yarn_freq_scale, hp.yarn_ext_factor,
                             dsv4_rope_attn_factor(hp.yarn_freq_scale, hp.yarn_ext_factor),
                             hp.yarn_beta_fast, hp.yarn_beta_slow);
    }

    // ---- hyper connections ----
    ggml_tensor * hc_affine(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base)
    {
        return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
    }

    ggml_tensor * view_row_1d(ggml_tensor * t, int64_t ne0, int64_t i0)
    {
        return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
    }

    ggml_tensor * view_row_2d(ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0)
    {
        return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
    }

    ggml_tensor * build_hc_pre(ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
                               ggml_tensor ** post, ggml_tensor ** comb, int dev)
    {
        const int64_t hc = hp.hc_mult;
        const int64_t hc_dim = hc * hp.n_embd;
        const int64_t n = x->ne[2];

        ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc_dim, n);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, hp.rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_norm);

        ggml_tensor * pre;
        if (m.fused)
        {
            ggml_tensor * gates = fused_node(TSG_DSV4_FUSED_HC_GATES, GGML_TYPE_F32, 2 * hc, n, 1, 1,
                    { mixes, hc_scale, hc_base }, dev, 0, 0, 0, 0, hp.hc_eps);
            pre = view_row_2d(gates, hc, n, 0);
            *post = view_row_2d(gates, hc, n, hc);
        }
        else
        {
            ggml_tensor * scale_pre = view_row_1d(hc_scale, 1, 0);
            ggml_tensor * scale_post = view_row_1d(hc_scale, 1, 1);
            ggml_tensor * base_pre = view_row_1d(hc_base, hc, 0);
            ggml_tensor * base_post = view_row_1d(hc_base, hc, hc);

            pre = view_row_2d(mixes, hc, n, 0);
            pre = hc_affine(pre, scale_pre, base_pre);
            pre = ggml_sigmoid(ctx, pre);
            pre = ggml_scale_bias(ctx, pre, 1.0f, hp.hc_eps);

            *post = view_row_2d(mixes, hc, n, hc);
            *post = hc_affine(*post, scale_post, base_post);
            *post = ggml_sigmoid(ctx, *post);
            *post = ggml_scale(ctx, *post, 2.0f);
        }

        *comb = ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, hp.hc_eps, hp.hc_sinkhorn_iters);

        return ggml_dsv4_hc_pre(ctx, x, pre);
    }

    ggml_tensor * build_hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb)
    {
        return ggml_dsv4_hc_post(ctx, x, residual, post, comb);
    }

    ggml_tensor * build_hc_head(ggml_tensor * x)
    {
        const int64_t hc = hp.hc_mult;
        const int64_t hc_dim = hc * hp.n_embd;
        const int64_t n = x->ne[2];

        ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc_dim, n);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, hp.rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(ctx, m.hc_head_fn, flat_norm);

        ggml_tensor * pre = hc_affine(mixes, m.hc_head_scale, m.hc_head_base);
        pre = ggml_sigmoid(ctx, pre);
        pre = ggml_scale_bias(ctx, pre, 1.0f, hp.hc_eps);

        return ggml_dsv4_hc_pre(ctx, x, pre);
    }

    // ---- compression ----

    // Overlap (ratio 4) compression from [persistent ring | scratch | zero] rows.
    ggml_tensor * build_overlap_compressed(
        ggml_tensor * kv_state,      // [2*head, state+nt]
        ggml_tensor * score_state,   // [2*head, state+nt]
        ggml_tensor * read_idxs,     // I32 [2*ratio*n_blocks]
        ggml_tensor * comp_pos,      // I32 [n_blocks]
        ggml_tensor * norm_w,
        int64_t ratio, int64_t head)
    {
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;
        const int64_t n_blocks = comp_pos->ne[0];
        const int64_t n_read = ratio * n_blocks;

        // The zero/-inf "before the first block" source row is baked into the
        // persistent state ring (row state_size), so no per-eval append here.
        ggml_tensor * kv_rows = ggml_get_rows(ctx, kv_state, read_idxs);
        ggml_tensor * score_rows = ggml_get_rows(ctx, score_state, read_idxs);

        ggml_tensor * kv_prev = ggml_cont(ctx, ggml_view_2d(ctx, kv_rows, head, n_read, kv_rows->nb[1], 0));
        kv_prev = ggml_reshape_3d(ctx, kv_prev, head, ratio, n_blocks);
        ggml_tensor * score_prev = ggml_cont(ctx, ggml_view_2d(ctx, score_rows, head, n_read, score_rows->nb[1], 0));
        score_prev = ggml_reshape_3d(ctx, score_prev, head, ratio, n_blocks);

        ggml_tensor * kv_cur = ggml_cont(ctx, ggml_view_2d(ctx, kv_rows, head, n_read, kv_rows->nb[1],
                n_read * kv_rows->nb[1] + ggml_row_size(kv_rows->type, head)));
        kv_cur = ggml_reshape_3d(ctx, kv_cur, head, ratio, n_blocks);
        ggml_tensor * score_cur = ggml_cont(ctx, ggml_view_2d(ctx, score_rows, head, n_read, score_rows->nb[1],
                n_read * score_rows->nb[1] + ggml_row_size(score_rows->type, head)));
        score_cur = ggml_reshape_3d(ctx, score_cur, head, ratio, n_blocks);

        ggml_tensor * values = ggml_concat(ctx, kv_prev, kv_cur, 1);
        ggml_tensor * scores = ggml_concat(ctx, score_prev, score_cur, 1);

        values = ggml_cont(ctx, ggml_permute(ctx, values, 1, 0, 2, 3));
        scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx, scores);
        ggml_tensor * comp = ggml_mul(ctx, values, weights);
        comp = ggml_sum_rows(ctx, comp);
        comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));

        comp = rms(comp, norm_w);

        ggml_tensor * comp_nope = ggml_view_3d(ctx, comp, n_nope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head), 0);
        ggml_tensor * comp_pe = ggml_view_3d(ctx, comp, n_rope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head),
                ggml_row_size(comp->type, n_nope));

        comp_pe = rope_compress(comp_pe, comp_pos);
        return ggml_concat(ctx, comp_nope, comp_pe, 0);
    }

    // Non-overlap (ratio 128) compression.
    ggml_tensor * build_hca_compressed(
        ggml_tensor * kv_state, ggml_tensor * score_state,
        ggml_tensor * read_idxs, ggml_tensor * comp_pos,
        ggml_tensor * norm_w, int64_t head)
    {
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;
        const int64_t n_blocks = comp_pos->ne[0];

        ggml_tensor * kv = ggml_get_rows(ctx, kv_state, read_idxs);
        kv = ggml_reshape_3d(ctx, kv, head, HCA_RATIO, n_blocks);
        ggml_tensor * score = ggml_get_rows(ctx, score_state, read_idxs);
        score = ggml_reshape_3d(ctx, score, head, HCA_RATIO, n_blocks);

        ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
        ggml_tensor * scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx, scores);
        ggml_tensor * comp = ggml_mul(ctx, values, weights);
        comp = ggml_sum_rows(ctx, comp);
        comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));

        comp = rms(comp, norm_w);

        ggml_tensor * comp_nope = ggml_view_3d(ctx, comp, n_nope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head), 0);
        ggml_tensor * comp_pe = ggml_view_3d(ctx, comp, n_rope, 1, n_blocks,
                ggml_row_size(comp->type, head), ggml_row_size(comp->type, head),
                ggml_row_size(comp->type, n_nope));

        comp_pe = rope_compress(comp_pe, comp_pos);
        return ggml_concat(ctx, comp_nope, comp_pe, 0);
    }

    // Update a compressor state ring + write completed compressed rows into `cache`.
    void run_compressor(
        int il,
        ggml_tensor * cur,               // [n_embd, nt]
        ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape, ggml_tensor * norm_w,
        ggml_tensor * state_kv, ggml_tensor * state_score,
        ggml_tensor * cache, int64_t head, int64_t ratio, bool overlap,
        const plan_inputs & pi, const comp_plan & plan, int dev)
    {
        ggml_tensor * st_kv = ggml_mul_mat(ctx, wkv, cur);        // [coff*head, nt]
        ggml_tensor * st_score = ggml_mul_mat(ctx, wgate, cur);
        ggml_tensor * ape_rows = ggml_get_rows(ctx, ape, pi.state_pos);
        st_score = ggml_add(ctx, st_score, ape_rows);

        if (m.fused)
        {
            // one fused node: window softmax-compress + RMS + table RoPE + F16
            // cache commit, then state-ring persist (mutates state/cache in
            // place; the marker result orders it before the attention reads)
            const int n_blocks = (int) plan.state_write_idxs.size();
            const int np = (int) plan.state_persist_src_idxs.size();
            ggml_tensor * marker = fused_node(TSG_DSV4_FUSED_COMPRESS, GGML_TYPE_F32, 1, 1, 1, 1,
                    { st_kv, st_score, state_kv, state_score, norm_w, m.rope_tab_dev[1][dev],
                      pi.comp_meta, cache },
                    dev,
                    n_blocks, (int) ratio, overlap ? 2 : 1, hp.n_rot,
                    hp.rms_eps, (float) np);
            ggml_build_forward_expand(gf, marker);
            return;
        }

        ggml_tensor * source_kv = ggml_concat(ctx, state_kv, st_kv, 1);
        ggml_tensor * source_score = ggml_concat(ctx, state_score, st_score, 1);

        ggml_tensor * comp = nullptr;
        if (pi.write_idxs != nullptr)
        {
            if (overlap)
                comp = build_overlap_compressed(source_kv, source_score, pi.read_idxs, pi.write_pos, norm_w, ratio, head);
            else
                comp = build_hca_compressed(source_kv, source_score, pi.read_idxs, pi.write_pos, norm_w, head);

            // write compressed rows into the cache (F16 set_rows casts)
            ggml_tensor * comp2d = ggml_reshape_2d(ctx, comp, head, comp->ne[2]);
            ggml_tensor * wr = ggml_set_rows(ctx, cache, comp2d, pi.write_idxs);
            ggml_build_forward_expand(gf, wr);
        }

        // persist ring update — expanded after the compression nodes, and each
        // device's split executes its nodes strictly in graph order on one
        // stream, so the ring writes land after the compression reads without
        // needing llama.cpp's zero-dependency ordering trick
        GGML_UNUSED(comp);
        ggml_tensor * persist_kv = ggml_get_rows(ctx, st_kv, pi.persist_src);
        ggml_tensor * persist_score = ggml_get_rows(ctx, st_score, pi.persist_src);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state_kv, persist_kv, pi.persist_dst));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state_score, persist_score, pi.persist_dst));
    }

    // ---- attention ----
    ggml_tensor * attn_mha(ggml_tensor * q, ggml_tensor * k, ggml_tensor * kq_mask, ggml_tensor * sinks, float kq_scale)
    {
        // q [head, n_head, nt], k [head, 1, n_kv] contiguous F16. K doubles as V.
        const int64_t head = k->ne[0];
        const int64_t n_kv = k->ne[2];

        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);   // [head, nt, n_head]
        ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);   // [head, n_kv, 1]

        ggml_tensor * cur;
        if (m.flash_attn)
        {
            cur = ggml_flash_attn_ext(ctx, qp, kp, kp, kq_mask, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(cur, sinks);
            ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        }
        else
        {
            ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);   // [n_kv, nt, n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);
            ggml_soft_max_add_sinks(kq, sinks);

            // v^T [n_kv, head] from the contiguous pre-permute K
            ggml_tensor * v = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, k, head, n_kv)));
            ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);  // [head, nt, n_head]

            ggml_tensor * curp = ggml_permute(ctx, kqv, 0, 2, 1, 3);   // [head, n_head, nt]
            cur = ggml_cont_2d(ctx, curp, curp->ne[0] * curp->ne[1], curp->ne[2]);
        }
        ggml_build_forward_expand(gf, cur);
        return cur;
    }

    ggml_tensor * build_lid_top_k(int il, ggml_tensor * qr, ggml_tensor * cur, ggml_tensor * inp_pos, int dev)
    {
        const dsv4_layer & L = m.layers[il];
        const dsv4_slot_layer & C = m.active_slot->layers[il];
        const int64_t idx_head = hp.indexer_head_size;
        const int64_t n_idx_head = hp.indexer_n_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = idx_head - n_rope;

        ggml_tensor * iq = ggml_mul_mat(ctx, L.indexer_attn_q_b, qr);
        iq = ggml_reshape_3d(ctx, iq, idx_head, n_idx_head, nt);

        ggml_tensor * iq_nope = ggml_view_3d(ctx, iq, n_nope, n_idx_head, nt,
                ggml_row_size(iq->type, idx_head), ggml_row_size(iq->type, idx_head) * n_idx_head, 0);
        ggml_tensor * iq_pe = ggml_view_3d(ctx, iq, n_rope, n_idx_head, nt,
                ggml_row_size(iq->type, idx_head), ggml_row_size(iq->type, idx_head) * n_idx_head,
                ggml_row_size(iq->type, n_nope));
        iq_pe = rope_compress(iq_pe, inp_pos);
        iq = ggml_concat(ctx, iq_nope, iq_pe, 0);   // [idx_head, n_idx_head, nt]

        ggml_tensor * iw = ggml_mul_mat(ctx, L.indexer_proj, cur);   // [n_idx_head, nt]
        iw = ggml_scale(ctx, iw, 1.0f / sqrtf((float) (idx_head * n_idx_head)));

        const int64_t n_lid = res.plan_lid.n_kv;
        ggml_tensor * ik = ggml_view_2d(ctx, C.lid_k, idx_head, n_lid, C.lid_k->nb[1], 0);
        ik = ggml_reshape_3d(ctx, ik, idx_head, 1, n_lid);

        // fused lightning indexer: q [idx_head, n_idx_head, nt], k [idx_head, 1, n_lid],
        // weights [n_idx_head, nt], mask f16 [n_lid, nt] -> [n_lid, nt]
        ggml_tensor * score = ggml_lightning_indexer(ctx, iq, ik, iw, res.inp.lid[dev].kq_mask);

        const int64_t k = std::min<int64_t>(hp.indexer_top_k, score->ne[0]);
        ggml_tensor * top_k = ggml_cont(ctx, ggml_top_k(ctx, score, (int) k));
        return top_k;
    }

    ggml_tensor * build_top_k_mask(ggml_tensor * kq_mask, ggml_tensor * top_k)
    {
        // port of llama.cpp build_top_k_mask, n_stream == 1
        ggml_tensor * mask_all = ggml_fill(ctx, kq_mask, -INFINITY);
        mask_all = ggml_view_4d(ctx, mask_all, 1, mask_all->ne[0], mask_all->ne[1], 1,
                mask_all->nb[0], mask_all->nb[1], mask_all->nb[2], 0);

        ggml_tensor * top_k_3d = ggml_view_4d(ctx, top_k, top_k->ne[0], top_k->ne[1], 1, 1,
                top_k->nb[1], top_k->nb[2], top_k->nb[3], 0);

        ggml_tensor * zeros = ggml_new_tensor_4d(ctx, kq_mask->type, 1, top_k_3d->ne[0], top_k_3d->ne[1], 1);
        zeros = ggml_fill(ctx, zeros, 0.0f);

        ggml_tensor * masked = ggml_set_rows(ctx, mask_all, zeros, top_k_3d);
        masked = ggml_view_4d(ctx, masked, masked->ne[1], masked->ne[2], 1, 1,
                masked->nb[2], masked->nb[3], masked->nb[3], 0);

        return ggml_add(ctx, masked, kq_mask);
    }

    ggml_tensor * build_attention(int il, ggml_tensor * cur, ggml_tensor * inp_pos)
    {
        const dsv4_layer & L = m.layers[il];
        const dsv4_slot_layer & C = m.active_slot->layers[il];
        const int dev = L.device;
        const int64_t head = hp.n_embd_head;
        const int64_t n_head = hp.n_head;
        const int64_t n_rope = hp.n_rot;
        const int64_t n_nope = head - n_rope;
        const int64_t n_groups = hp.o_groups;
        const int64_t o_lora = hp.o_lora_rank;
        const int64_t o_group_dim = (n_head / n_groups) * head;
        const int32_t ratio = hp.compress_ratios[il];
        const bool comp_layer = ratio != 0;
        ggml_tensor * rope_tab = m.rope_tab_dev[comp_layer ? 1 : 0][dev];

        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        int n_ctx_orig;
        rope_params(il, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_ctx_orig);

        auto rope_l = [&](ggml_tensor * x, ggml_tensor * pos)
        {
            return ggml_rope_ext(ctx, x, pos, nullptr, (int) n_rope, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        };

        ggml_tensor * qr = ggml_mul_mat(ctx, L.wq_a, cur);
        qr = rms(qr, L.attn_q_a_norm);

        ggml_tensor * q = nullptr;
        if (m.fused)
        {
            // Emit the compressor scratch matmuls first so attn_prep and the
            // compress nodes form one contiguous fused-backend split (fewer
            // scheduler splits, better per-segment CUDA-graph reuse).
            auto comp_scratch = [&](ggml_tensor * wkv, ggml_tensor * wgate, ggml_tensor * ape,
                                    const plan_inputs & pi, ggml_tensor ** st_kv, ggml_tensor ** st_score)
            {
                *st_kv = ggml_mul_mat(ctx, wkv, cur);        // [coff*head, nt]
                ggml_tensor * sc = ggml_mul_mat(ctx, wgate, cur);
                ggml_tensor * ape_rows = ggml_get_rows(ctx, ape, pi.state_pos);
                *st_score = ggml_add(ctx, sc, ape_rows);
            };
            auto emit_compress = [&](ggml_tensor * st_kv, ggml_tensor * st_score,
                                     ggml_tensor * state_kv, ggml_tensor * state_score,
                                     ggml_tensor * norm_w, ggml_tensor * cache,
                                     int64_t cratio, bool overlap,
                                     const plan_inputs & pi, const comp_plan & plan)
            {
                const int n_blocks = (int) plan.state_write_idxs.size();
                const int np = (int) plan.state_persist_src_idxs.size();
                ggml_tensor * marker = fused_node(TSG_DSV4_FUSED_COMPRESS, GGML_TYPE_F32, 1, 1, 1, 1,
                        { st_kv, st_score, state_kv, state_score, norm_w, m.rope_tab_dev[1][dev],
                          pi.comp_meta, cache },
                        dev,
                        n_blocks, (int) cratio, overlap ? 2 : 1, hp.n_rot,
                        hp.rms_eps, (float) np);
                ggml_build_forward_expand(gf, marker);
            };

            ggml_tensor * q_raw = ggml_mul_mat(ctx, L.wq_b, qr);    // [head*n_head, nt]
            ggml_tensor * kv_raw = ggml_mul_mat(ctx, L.wkv, cur);   // [head, nt]

            ggml_tensor * cs_kv = nullptr, * cs_score = nullptr;
            ggml_tensor * ls_kv = nullptr, * ls_score = nullptr;
            if (ratio == CSA_RATIO)
            {
                comp_scratch(L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, res.inp.csa[dev], &cs_kv, &cs_score);
                comp_scratch(L.indexer_comp_wkv, L.indexer_comp_wgate, L.indexer_comp_ape, res.inp.lid[dev], &ls_kv, &ls_score);
            }
            else if (ratio == HCA_RATIO)
            {
                comp_scratch(L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, res.inp.hca[dev], &cs_kv, &cs_score);
            }

            // per-head q RMS + RoPE, kv RMS(w) + RoPE + F16 SWA-ring commit
            q = fused_node(TSG_DSV4_FUSED_ATTN_PREP, GGML_TYPE_F32, head, n_head, nt, 1,
                    { q_raw, kv_raw, L.attn_kv_norm, rope_tab, inp_pos, C.raw_k, res.inp.raw_idxs[dev] },
                    dev, (int) n_rope, 0, 0, 0, hp.rms_eps);
            ggml_build_forward_expand(gf, q);

            if (ratio == CSA_RATIO)
            {
                emit_compress(cs_kv, cs_score, C.comp_state_kv, C.comp_state_score, L.attn_comp_norm,
                              C.csa_k, CSA_RATIO, true, res.inp.csa[dev], res.plan_csa);
                emit_compress(ls_kv, ls_score, C.lid_state_kv, C.lid_state_score, L.indexer_comp_norm,
                              C.lid_k, CSA_RATIO, true, res.inp.lid[dev], res.plan_lid);
            }
            else if (ratio == HCA_RATIO)
            {
                emit_compress(cs_kv, cs_score, C.comp_state_kv, C.comp_state_score, L.attn_comp_norm,
                              C.hca_k, HCA_RATIO, false, res.inp.hca[dev], res.plan_hca);
            }
        }
        else
        {
            q = ggml_mul_mat(ctx, L.wq_b, qr);
            q = ggml_reshape_3d(ctx, q, head, n_head, nt);
            q = ggml_rms_norm(ctx, q, hp.rms_eps);

            ggml_tensor * q_nope = ggml_view_3d(ctx, q, n_nope, n_head, nt,
                    ggml_row_size(q->type, head), ggml_row_size(q->type, head) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx, q, n_rope, n_head, nt,
                    ggml_row_size(q->type, head), ggml_row_size(q->type, head) * n_head,
                    ggml_row_size(q->type, n_nope));
            q_pe = rope_l(q_pe, inp_pos);
            q = ggml_concat(ctx, q_nope, q_pe, 0);   // [head, n_head, nt]

            ggml_tensor * kv = ggml_mul_mat(ctx, L.wkv, cur);
            kv = rms(kv, L.attn_kv_norm);
            kv = ggml_reshape_3d(ctx, kv, head, 1, nt);

            ggml_tensor * kv_nope = ggml_view_3d(ctx, kv, n_nope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx, kv, n_rope, 1, nt,
                    ggml_row_size(kv->type, head), ggml_row_size(kv->type, head),
                    ggml_row_size(kv->type, n_nope));
            kv_pe = rope_l(kv_pe, inp_pos);
            kv = ggml_concat(ctx, kv_nope, kv_pe, 0);   // [head, 1, nt]

            // write raw K into the SWA ring
            ggml_tensor * kv2d = ggml_reshape_2d(ctx, kv, head, nt);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.raw_k, kv2d, res.inp.raw_idxs[dev]));
            ggml_build_forward_expand(gf, q);

            // compressor state updates + compressed row commits
            if (ratio == CSA_RATIO)
            {
                run_compressor(il, cur, L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, L.attn_comp_norm,
                               C.comp_state_kv, C.comp_state_score, C.csa_k, head, CSA_RATIO, true,
                               res.inp.csa[dev], res.plan_csa, dev);
                run_compressor(il, cur, L.indexer_comp_wkv, L.indexer_comp_wgate, L.indexer_comp_ape, L.indexer_comp_norm,
                               C.lid_state_kv, C.lid_state_score, C.lid_k, hp.indexer_head_size, CSA_RATIO, true,
                               res.inp.lid[dev], res.plan_lid, dev);
            }
            else if (ratio == HCA_RATIO)
            {
                run_compressor(il, cur, L.attn_comp_wkv, L.attn_comp_wgate, L.attn_comp_ape, L.attn_comp_norm,
                               C.comp_state_kv, C.comp_state_score, C.hca_k, head, HCA_RATIO, false,
                               res.inp.hca[dev], res.plan_hca, dev);
            }
        }

        // attention source: raw ring (+ compressed rows)
        ggml_tensor * raw_k = ggml_view_2d(ctx, C.raw_k, head, m.ring_raw, C.raw_k->nb[1], 0);
        raw_k = ggml_reshape_3d(ctx, raw_k, head, 1, m.ring_raw);

        ggml_tensor * out = nullptr;
        const float kq_scale = 1.0f / sqrtf((float) head);

        if (ratio == CSA_RATIO && res.inp.gather_mask[dev])
        {
            // Decode index-gather: compact [raw ring | top-k rows] K. Every
            // top-k index is a valid compressed row here (see
            // dsv4_use_gather), so the gathered tail needs no masking and the
            // softmax over the gathered subset equals the masked-dense
            // result. The gather runs after this layer's compressor commit
            // (same stream, graph order), so a just-completed block is
            // selectable exactly like in the masked-dense path.
            ggml_tensor * top_k = build_lid_top_k(il, qr, cur, inp_pos, dev);
            ggml_tensor * k_sel = fused_node(TSG_DSV4_FUSED_KGATHER, GGML_TYPE_F16,
                    head, 1, m.ring_raw + top_k->ne[0], 1,
                    { C.raw_k, C.csa_k, top_k }, dev, (int) m.ring_raw);
            out = attn_mha(q, k_sel, res.inp.gather_mask[dev], L.attn_sinks, kq_scale);
        }
        else if (ratio == CSA_RATIO)
        {
            ggml_tensor * csa_k = ggml_view_2d(ctx, C.csa_k, head, res.plan_csa.n_kv, C.csa_k->nb[1], 0);
            csa_k = ggml_reshape_3d(ctx, csa_k, head, 1, res.plan_csa.n_kv);

            ggml_tensor * k_all = ggml_concat(ctx, raw_k, csa_k, 2);

            // While every visible compressed row is within the top-k budget
            // (pos_end/ratio <= top_k), the indexer cannot exclude anything —
            // the plain visibility mask is exact and the whole indexer query
            // chain (projections, RoPE, lightning indexer, argsort) is dead
            // weight. The LID cache is still maintained by its compressor.
            // Decided on the whole forward call's final position (must match
            // dsv4_acquire_graph's signature bit).
            const bool skip_topk = m.fused &&
                std::max(m.pos_end_hint, p0 + nt) <= (int64_t) hp.indexer_top_k * CSA_RATIO;

            ggml_tensor * kq_mask;
            if (skip_topk)
            {
                kq_mask = ggml_concat(ctx, res.inp.raw_mask[dev], res.inp.csa[dev].kq_mask, 0);
            }
            else if (m.fused)
            {
                ggml_tensor * top_k = build_lid_top_k(il, qr, cur, inp_pos, dev);
                ggml_tensor * base = ggml_concat(ctx, res.inp.raw_mask[dev], res.inp.csa[dev].kq_mask, 0);
                kq_mask = fused_node(TSG_DSV4_FUSED_TOPK_MASK, GGML_TYPE_F16,
                        m.ring_raw + res.plan_csa.n_kv, nt, 1, 1,
                        { base, top_k }, dev, (int) m.ring_raw);
            }
            else
            {
                ggml_tensor * top_k = build_lid_top_k(il, qr, cur, inp_pos, dev);
                ggml_tensor * csa_mask = build_top_k_mask(res.inp.csa[dev].kq_mask, top_k);
                kq_mask = ggml_concat(ctx, res.inp.raw_mask[dev], csa_mask, 0);
            }

            out = attn_mha(q, k_all, kq_mask, L.attn_sinks, kq_scale);
        }
        else if (ratio == HCA_RATIO)
        {
            ggml_tensor * hca_k = ggml_view_2d(ctx, C.hca_k, head, res.plan_hca.n_kv, C.hca_k->nb[1], 0);
            hca_k = ggml_reshape_3d(ctx, hca_k, head, 1, res.plan_hca.n_kv);

            ggml_tensor * k_all = ggml_concat(ctx, raw_k, hca_k, 2);
            ggml_tensor * kq_mask = ggml_concat(ctx, res.inp.raw_mask[dev], res.inp.hca[dev].kq_mask, 0);

            out = attn_mha(q, k_all, kq_mask, L.attn_sinks, kq_scale);
        }
        else
        {
            out = attn_mha(q, raw_k, res.inp.raw_mask[dev], L.attn_sinks, kq_scale);
        }

        ggml_tensor * grouped = nullptr;
        if (m.fused)
        {
            // inverse table-RoPE on the attention output + regroup to the
            // grouped-LoRA layout [o_group_dim, nt, n_groups]
            grouped = fused_node(TSG_DSV4_FUSED_ATTN_FINISH, GGML_TYPE_F32, o_group_dim, nt, n_groups, 1,
                    { out, rope_tab, inp_pos }, dev, (int) n_rope, (int) n_groups, (int) head);
        }
        else
        {
            // inverse RoPE on the rope slice of the attention output
            out = ggml_reshape_3d(ctx, out, head, n_head, nt);
            ggml_tensor * out_nope = ggml_view_3d(ctx, out, n_nope, n_head, nt,
                    ggml_row_size(out->type, head), ggml_row_size(out->type, head) * n_head, 0);
            ggml_tensor * out_pe = ggml_view_3d(ctx, out, n_rope, n_head, nt,
                    ggml_row_size(out->type, head), ggml_row_size(out->type, head) * n_head,
                    ggml_row_size(out->type, n_nope));
            out_pe = ggml_rope_ext_back(ctx, out_pe, inp_pos, nullptr, (int) n_rope, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            out = ggml_concat(ctx, out_nope, out_pe, 0);

            out = ggml_reshape_3d(ctx, out, o_group_dim, n_groups, nt);
            grouped = ggml_permute(ctx, out, 0, 2, 1, 3);   // [o_group_dim, nt, n_groups]
        }

        // grouped LoRA output projection
        ggml_tensor * oa = ggml_mul_mat(ctx,
                ggml_reshape_3d(ctx, L.wo_a, L.wo_a->ne[0], o_lora, n_groups), grouped);   // [o_lora, nt, n_groups]
        oa = ggml_permute(ctx, oa, 0, 2, 1, 3);     // [o_lora, n_groups, nt]
        oa = ggml_cont_2d(ctx, oa, o_lora * n_groups, nt);

        return ggml_mul_mat(ctx, L.wo_b, oa);
    }

    // ---- MoE ----

    ggml_tensor * swiglu_clamped(ggml_tensor * gate, ggml_tensor * up, float clamp_limit, int dev)
    {
        if (m.fused)
        {
            const float limit = clamp_limit > 1e-6f ? clamp_limit : INFINITY;
            return fused_node(TSG_DSV4_FUSED_SWIGLU_CLAMP, GGML_TYPE_F32,
                    gate->ne[0], gate->ne[1], gate->ne[2], gate->ne[3],
                    { gate, up }, dev, 0, 0, 0, 0, limit);
        }
        if (clamp_limit > 1e-6f)
        {
            up = ggml_clamp(ctx, up, -clamp_limit, clamp_limit);
            gate = ggml_clamp(ctx, gate, -INFINITY, clamp_limit);
        }
        return ggml_swiglu_split(ctx, gate, up);
    }

    // shexp: the shared expert's FFN output (added to the routed experts)
    ggml_tensor * build_moe(int il, ggml_tensor * cur, ggml_tensor * inp_tokens, ggml_tensor * shexp)
    {
        const dsv4_layer & L = m.layers[il];
        const int dev = L.device;
        const int64_t n_expert = hp.n_expert;
        const int64_t n_used = hp.n_expert_used;
        const float clamp_limit = hp.swiglu_clamp_exp.empty() ? 0.0f : hp.swiglu_clamp_exp[il];

        ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);   // [n_expert, nt]
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);

        ggml_tensor * selected = nullptr;
        ggml_tensor * weights = nullptr;

        if (m.fused)
        {
            if (il < hp.hash_layer_count)
                selected = ggml_get_rows(ctx, L.ffn_gate_tid2eid, inp_tokens);   // I32 [n_used, nt]
            else
                selected = fused_node(TSG_DSV4_FUSED_MOE_TOPK, GGML_TYPE_I32, n_used, nt, 1, 1,
                        { logits, L.ffn_exp_probs_b }, dev);

            const float scale = (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
                ? hp.expert_weights_scale : 1.0f;
            weights = fused_node(TSG_DSV4_FUSED_MOE_WEIGHTS, GGML_TYPE_F32, 1, n_used, nt, 1,
                    { logits, selected }, dev, hp.expert_weights_norm ? 1 : 0, 0, 0, 0, scale);
        }
        else
        {
            ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));

            if (il < hp.hash_layer_count)
            {
                selected = ggml_get_rows(ctx, L.ffn_gate_tid2eid, inp_tokens);   // I32 [n_used, nt]
            }
            else
            {
                ggml_tensor * selection = ggml_add(ctx, probs, L.ffn_exp_probs_b);
                selected = ggml_argsort_top_k(ctx, selection, (int) n_used);
            }

            ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, n_expert, nt);
            weights = ggml_get_rows(ctx, probs3, selected);       // [1, n_used, nt]

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
        }

        ggml_build_forward_expand(gf, weights);

        ggml_tensor * cur3 = ggml_reshape_3d(ctx, cur, hp.n_embd, 1, nt);

        ggml_tensor * up = ggml_mul_mat_id(ctx, L.ffn_up_exps, cur3, selected);     // [n_ff, n_used, nt]
        ggml_tensor * gate = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur3, selected);

        ggml_tensor * h = swiglu_clamped(gate, up, clamp_limit, dev);

        ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, h, selected); // [n_embd, n_used, nt]

        if (m.fused)
        {
            // weighted expert sum + shared-expert add in one launch
            return fused_node(TSG_DSV4_FUSED_EXPERT_REDUCE, GGML_TYPE_F32, hp.n_embd, nt, 1, 1,
                    { experts, weights, shexp }, dev);
        }

        experts = ggml_mul(ctx, experts, weights);

        // sum over the used experts
        ggml_tensor * moe_out = nullptr;
        for (int64_t e = 0; e < n_used; e++)
        {
            ggml_tensor * v = ggml_view_2d(ctx, experts, hp.n_embd, nt, experts->nb[2], e * experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx, moe_out, v) : v;
        }
        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);
        return ggml_add(ctx, moe_out, shexp);
    }

    ggml_tensor * build_shexp(int il, ggml_tensor * cur)
    {
        const dsv4_layer & L = m.layers[il];
        const float clamp_limit = hp.swiglu_clamp_shexp.empty() ? 0.0f : hp.swiglu_clamp_shexp[il];

        ggml_tensor * up = ggml_mul_mat(ctx, L.ffn_up_shexp, cur);
        ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
        ggml_tensor * h = swiglu_clamped(gate, up, clamp_limit, L.device);
        return ggml_mul_mat(ctx, L.ffn_down_shexp, h);
    }

    // ---- whole graph ----
    void build()
    {
        graph_inputs & inp = res.inp;

        // devices that host at least one layer
        bool dev_used[MAX_GPUS + 1] = {};
        for (int il = 0; il < hp.n_layer; il++) dev_used[m.layers[il].device] = true;
        const int dev_last = m.layers[hp.n_layer - 1].device;

        auto make_plan_inputs = [&](plan_inputs & pi, const comp_plan & plan, ggml_type mask_type, const char * tag, int d)
        {
            char nb[128];
            pi.n_kv = plan.n_kv;
            snprintf(nb, sizeof(nb), "inp_%s_state_pos.%d", tag, d);
            pi.state_pos = new_input_i32(plan.state_pos.size(), nb, d);
            if (m.fused)
            {
                const size_t n_meta = plan.state_read_idxs.size()
                    + 2 * plan.state_write_idxs.size()
                    + 2 * plan.state_persist_src_idxs.size();
                snprintf(nb, sizeof(nb), "inp_%s_comp_meta.%d", tag, d);
                pi.comp_meta = new_input_i32(std::max<size_t>(n_meta, 1), nb, d);
            }
            else
            {
                snprintf(nb, sizeof(nb), "inp_%s_persist_src.%d", tag, d);
                pi.persist_src = new_input_i32(plan.state_persist_src_idxs.size(), nb, d);
                snprintf(nb, sizeof(nb), "inp_%s_persist_dst.%d", tag, d);
                pi.persist_dst = new_input_i64(plan.state_persist_dst_idxs.size(), nb, d);
                if (!plan.state_write_idxs.empty())
                {
                    snprintf(nb, sizeof(nb), "inp_%s_read_idxs.%d", tag, d);
                    pi.read_idxs = new_input_i32(plan.state_read_idxs.size(), nb, d);
                    snprintf(nb, sizeof(nb), "inp_%s_write_idxs.%d", tag, d);
                    pi.write_idxs = new_input_i64(plan.state_write_idxs.size(), nb, d);
                    snprintf(nb, sizeof(nb), "inp_%s_write_pos.%d", tag, d);
                    pi.write_pos = new_input_i32(plan.state_write_pos.size(), nb, d);
                }
            }
            snprintf(nb, sizeof(nb), "inp_%s_mask.%d", tag, d);
            pi.kq_mask = new_input_mask(plan.n_kv, mask_type, nb, d);
        };

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!dev_used[d]) continue;
            char nb[64];
            snprintf(nb, sizeof(nb), "inp_tokens.%d", d);
            inp.tokens[d] = new_input_i32(nt, nb, d);
            snprintf(nb, sizeof(nb), "inp_pos.%d", d);
            inp.pos[d] = new_input_i32(nt, nb, d);
            snprintf(nb, sizeof(nb), "inp_raw_idxs.%d", d);
            inp.raw_idxs[d] = new_input_i64(nt, nb, d);
            snprintf(nb, sizeof(nb), "inp_raw_mask.%d", d);
            inp.raw_mask[d] = new_input_mask(m.ring_raw, GGML_TYPE_F16, nb, d);
            if (dsv4_use_gather(m, nt, std::max(m.pos_end_hint, p0 + nt)))
            {
                snprintf(nb, sizeof(nb), "inp_gather_mask.%d", d);
                inp.gather_mask[d] = new_input_mask(m.ring_raw + hp.indexer_top_k, GGML_TYPE_F16, nb, d);
            }
            make_plan_inputs(inp.csa[d], res.plan_csa, GGML_TYPE_F16, "csa", d);
            make_plan_inputs(inp.hca[d], res.plan_hca, GGML_TYPE_F16, "hca", d);
            make_plan_inputs(inp.lid[d], res.plan_lid, GGML_TYPE_F16, "lid", d);
        }
        inp.out_ids = new_input_i32(1, "inp_out_ids", dev_last);

        const int64_t hc = hp.hc_mult;

        ggml_tensor * emb = ggml_get_rows(ctx, m.tok_embd, inp.tokens[m.layers[0].device]);   // [n_embd, nt]
        ggml_tensor * inpL = ggml_reshape_3d(ctx, emb, hp.n_embd, 1, nt);
        inpL = ggml_repeat_4d(ctx, inpL, hp.n_embd, hc, nt, 1);

        for (int il = 0; il < hp.n_layer; il++)
        {
            const dsv4_layer & L = m.layers[il];
            const int dev = L.device;

            // At a device boundary, pin the incoming activations to the new
            // device. Without this the scheduler ping-pongs the boundary
            // layer's elementwise ops between GPUs (x-derived sources vote for
            // the old device, weight views for the new one), costing ~6 extra
            // graph splits per eval.
            if (il > 0 && L.device != m.layers[il - 1].device)
                ggml_backend_sched_set_tensor_backend(res.sched, inpL, m.backends[L.device]);

            ggml_tensor * residual = inpL;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;

            ggml_tensor * cur = build_hc_pre(inpL, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, &post, &comb, dev);
            cur = rms(cur, L.attn_norm);
            cur = build_attention(il, cur, inp.pos[dev]);
            inpL = build_hc_post(cur, residual, post, comb);

            residual = inpL;
            cur = build_hc_pre(inpL, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, &post, &comb, dev);
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, comb);

            cur = rms(cur, L.ffn_norm);

            ggml_tensor * shexp = build_shexp(il, cur);
            cur = build_moe(il, cur, inp.tokens[dev], shexp);

            inpL = build_hc_post(cur, residual, post, comb);
        }

        // gather output row(s)
        ggml_tensor * flat = ggml_reshape_2d(ctx, inpL, hp.n_embd * hc, nt);
        flat = ggml_get_rows(ctx, flat, inp.out_ids);
        inpL = ggml_reshape_3d(ctx, flat, hp.n_embd, hc, 1);

        ggml_tensor * cur = build_hc_head(inpL);
        cur = rms(cur, m.output_norm);
        cur = ggml_mul_mat(ctx, m.output, cur);
        ggml_set_output(cur);
        ggml_set_name(cur, "logits");
        res.logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// Unreferenced per-device input tensors (e.g. tokens on a device with no hash
// MoE layer) never enter the graph and stay unallocated — skip those.
static void set_i32(ggml_tensor * t, const std::vector<int32_t> & v)
{
    if (!t || !t->buffer || v.empty()) return;
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int32_t));
}

static void set_i64(ggml_tensor * t, const std::vector<int64_t> & v)
{
    if (!t || !t->buffer || v.empty()) return;
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int64_t));
}

static uint64_t dsv4_plan_sig(const comp_plan & p)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix((uint64_t) p.state_pos.size());
    mix((uint64_t) p.state_persist_src_idxs.size());
    mix((uint64_t) p.state_read_idxs.size());
    mix((uint64_t) p.state_write_idxs.size());
    mix((uint64_t) p.n_kv);
    return h;
}

// Build (or fetch from the LRU cache) the graph entry for a (nt, p0,
// pipeline-parity) shape. Each entry owns its scheduler/allocation, so
// alternating shapes never rebuild, realloc, or thrash the per-split CUDA
// graphs captured by ggml-cuda. Fresh entries device-synchronize on their
// arena allocation — TSGgml_Dsv4Forward pre-acquires the first two prefill
// chunks so this happens while the devices are idle.
static graph_build_result * dsv4_acquire_graph(dsv4_model & m, int64_t nt, int64_t p0, bool pipeline, bool * out_reuse)
{
    const dsv4_hparams & hp = m.hp;

    // Uniform shape across a forward call: bucket by the call's final
    // position (capped so very long prefills don't over-pad early attention)
    // and decide the indexer skip by the final position, so every chunk of a
    // prefill shares one graph-shape pair and the pipeline never stalls on a
    // mid-flight allocation.
    const int64_t pos_end = std::max(m.pos_end_hint, p0 + nt);
    const int64_t hint = std::min<int64_t>(pos_end, 8192);

    comp_plan plan_csa = build_comp_plan(p0, nt, CSA_RATIO, true, 2 * CSA_RATIO, m.n_csa_rows, hint / CSA_RATIO);
    comp_plan plan_hca = build_comp_plan(p0, nt, HCA_RATIO, false, HCA_RATIO, m.n_hca_rows, hint / HCA_RATIO);
    comp_plan plan_lid = build_comp_plan(p0, nt, CSA_RATIO, true, 2 * CSA_RATIO, m.n_csa_rows, hint / CSA_RATIO);

    uint64_t sig = 14695981039346656037ull ^ (uint64_t) nt;
    sig = sig * 1099511628211ull ^ dsv4_plan_sig(plan_csa);
    sig = sig * 1099511628211ull ^ dsv4_plan_sig(plan_hca);
    sig = sig * 1099511628211ull ^ dsv4_plan_sig(plan_lid);
    // graphs bake the active slot's cache tensor addresses
    sig = sig * 1099511628211ull ^ (uint64_t) (m.active_slot->id + 1);
    // the short-context indexer skip changes the graph shape
    if (m.fused && pos_end <= (int64_t) hp.indexer_top_k * CSA_RATIO)
        sig ^= 0x9e3779b97f4a7c15ull;
    // pipelined chunks alternate between two entries so in-flight inputs stay private
    if (pipeline && ((p0 / std::max<int64_t>(nt, 1)) & 1))
        sig ^= 0x517cc1b727220a95ull;

    graph_build_result * res_p = nullptr;
    bool reuse = false;
    for (auto it = m.graph_cache.begin(); it != m.graph_cache.end(); ++it)
    {
        if ((*it)->sig == sig)
        {
            if (it != m.graph_cache.begin())
                m.graph_cache.splice(m.graph_cache.begin(), m.graph_cache, it);
            res_p = m.graph_cache.front().get();
            reuse = true;
            break;
        }
    }

    if (!reuse)
    {
        m.graph_cache.emplace_front(new graph_build_result());
        graph_build_result & r = *m.graph_cache.front();
        const size_t meta_size = (size_t) 96 * 1024 * 1024;
        ggml_init_params gp = { meta_size, nullptr, true };
        r.ctx = ggml_init(gp);
        r.gf = ggml_new_graph_custom(r.ctx, 32768, false);
        r.nt = nt;
        r.sig = sig;
        r.slot_id = m.active_slot->id;
        r.plan_csa = plan_csa;
        r.plan_hca = plan_hca;
        r.plan_lid = plan_lid;
        r.sched = ggml_backend_sched_new(m.sched_backends, m.sched_bufts, m.n_sched_backends, 32768, false, true);
        if (!r.sched)
        {
            fprintf(stderr, "[dsv4] sched creation failed\n");
            m.graph_cache.pop_front();
            return nullptr;
        }

        graph_builder gb(m, r, nt, p0);
        gb.build();

        if (!ggml_backend_sched_alloc_graph(r.sched, r.gf))
        {
            fprintf(stderr, "[dsv4] sched_alloc_graph failed (nt=%" PRId64 ")\n", nt);
            m.graph_cache.pop_front();
            return nullptr;
        }
        res_p = &r;

        while ((int) m.graph_cache.size() > m.graph_cache_cap)
            m.graph_cache.pop_back();
    }

    res_p->p0 = p0;
    res_p->plan_csa = std::move(plan_csa);
    res_p->plan_hca = std::move(plan_hca);
    res_p->plan_lid = std::move(plan_lid);
    if (out_reuse) *out_reuse = reuse;
    return res_p;
}

static bool dsv4_forward_ubatch(dsv4_model & m, const int32_t * tokens, int64_t nt, int64_t p0, bool want_logits, float * logits_out)
{
    const dsv4_hparams & hp = m.hp;

    static const int perf = []() { const char * e = getenv("TS_DSV4_PERF"); return e ? atoi(e) : 0; }();
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto t0 = now();

    // Non-final prefill chunks are submitted asynchronously so the layer-split
    // devices pipeline: chunk i+1's first-device layers overlap chunk i's
    // second-device layers. Adjacent chunks are forced onto alternating cache
    // entries (chunk-parity in the signature) so each entry's input tensors
    // are private to in-flight work; reuse waits on the entry's use_events.
    const bool pipeline = !want_logits && m.n_gpu > 1;

    bool reuse = false;
    graph_build_result * res_p = dsv4_acquire_graph(m, nt, p0, pipeline, &reuse);
    if (!res_p) return false;
    graph_build_result & res = *res_p;

    auto t_build = now();
    auto t_alloc = now();

    // Before refilling a reused entry's inputs, wait until its previous
    // pipelined use has finished reading them (near-free at steady state:
    // the entry was last used two chunks ago).
    if (reuse)
    {
        for (int d = 0; d < m.n_gpu; d++)
            if (res.use_events[d])
                ggml_backend_event_synchronize(res.use_events[d]);
    }

    // ---- fill inputs (per participating device) ----
    {
        std::vector<int32_t> v32(nt);
        std::vector<int64_t> ridx(nt);
        std::vector<int32_t> out_ids(1, (int32_t) (nt - 1));
        std::vector<int32_t> meta;

        const ggml_fp16_t NEG_INF16 = ggml_fp32_to_fp16(-INFINITY);
        const ggml_fp16_t ZERO16 = ggml_fp32_to_fp16(0.0f);

        // raw SWA mask [ring, nt] (F16)
        const int64_t p_last = p0 + nt - 1;
        std::vector<ggml_fp16_t> mask((size_t) m.ring_raw * nt, NEG_INF16);
        for (int64_t i = 0; i < nt; i++)
        {
            const int64_t p = p0 + i;
            for (int64_t s = 0; s < m.ring_raw; s++)
            {
                // token currently held by slot s (all ubatch tokens are written before attention)
                int64_t t = p_last - ((p_last - s) % m.ring_raw + m.ring_raw) % m.ring_raw;
                if (t < 0) continue;
                if (t <= p && t > p - hp.n_swa)
                    mask[(size_t) i * m.ring_raw + s] = ZERO16;
            }
        }

        auto plan_mask = [&](const comp_plan & plan, std::vector<ggml_fp16_t> & mk)
        {
            const int64_t n_kv = plan.n_kv;
            mk.assign((size_t) n_kv * nt, NEG_INF16);
            for (int64_t i = 0; i < nt; i++)
                for (int64_t r = 0; r < std::min<int64_t>(plan.n_visible[i], n_kv); r++)
                    mk[(size_t) i * n_kv + r] = ZERO16;
        };
        std::vector<ggml_fp16_t> mk_csa, mk_hca, mk_lid;
        plan_mask(res.plan_csa, mk_csa);
        plan_mask(res.plan_hca, mk_hca);
        plan_mask(res.plan_lid, mk_lid);

        // gather-mode attention mask: [raw ring mask | top-k rows, all visible]
        std::vector<ggml_fp16_t> gmask;
        bool any_gather = false;
        for (int d = 0; d <= m.n_gpu && !any_gather; d++) any_gather = res.inp.gather_mask[d] != nullptr;
        if (any_gather)
        {
            gmask.assign(mask.begin(), mask.end());
            gmask.resize((size_t) (m.ring_raw + hp.indexer_top_k) * nt, ZERO16);
        }

        auto set_f16 = [](ggml_tensor * t, const std::vector<ggml_fp16_t> & v)
        {
            if (!t || !t->buffer || v.empty()) return;
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
        };

        auto fill_plan = [&](plan_inputs & pi, const comp_plan & plan, const std::vector<ggml_fp16_t> & mk)
        {
            set_i32(pi.state_pos, plan.state_pos);
            if (m.fused)
            {
                meta.clear();
                meta.insert(meta.end(), plan.state_read_idxs.begin(), plan.state_read_idxs.end());
                for (size_t i = 0; i < plan.state_write_idxs.size(); i++)
                {
                    meta.push_back((int32_t) plan.state_write_idxs[i]);
                    meta.push_back(plan.state_write_pos[i]);
                }
                for (size_t i = 0; i < plan.state_persist_src_idxs.size(); i++)
                {
                    meta.push_back(plan.state_persist_src_idxs[i]);
                    meta.push_back((int32_t) plan.state_persist_dst_idxs[i]);
                }
                set_i32(pi.comp_meta, meta);
            }
            else
            {
                set_i32(pi.persist_src, plan.state_persist_src_idxs);
                set_i64(pi.persist_dst, plan.state_persist_dst_idxs);
                if (pi.write_idxs)
                {
                    set_i32(pi.read_idxs, plan.state_read_idxs);
                    set_i64(pi.write_idxs, plan.state_write_idxs);
                    set_i32(pi.write_pos, plan.state_write_pos);
                }
            }
            set_f16(pi.kq_mask, mk);
        };

        for (int d = 0; d <= m.n_gpu; d++)
        {
            if (!res.inp.pos[d]) continue;
            for (int64_t i = 0; i < nt; i++) v32[i] = tokens[i];
            set_i32(res.inp.tokens[d], v32);
            for (int64_t i = 0; i < nt; i++) v32[i] = (int32_t) (p0 + i);
            set_i32(res.inp.pos[d], v32);
            for (int64_t i = 0; i < nt; i++) ridx[i] = (p0 + i) % m.ring_raw;
            set_i64(res.inp.raw_idxs[d], ridx);
            set_f16(res.inp.raw_mask[d], mask);
            set_f16(res.inp.gather_mask[d], gmask);
            fill_plan(res.inp.csa[d], res.plan_csa, mk_csa);
            fill_plan(res.inp.hca[d], res.plan_hca, mk_hca);
            fill_plan(res.inp.lid[d], res.plan_lid, mk_lid);
        }
        set_i32(res.inp.out_ids, out_ids);
    }

    auto t_inputs = now();

    if (pipeline)
    {
        if (ggml_backend_sched_graph_compute_async(res.sched, res.gf) != GGML_STATUS_SUCCESS)
        {
            fprintf(stderr, "[dsv4] graph compute failed\n");
            return false;
        }
        for (int d = 0; d < m.n_gpu; d++)
        {
            if (!res.use_events[d])
                res.use_events[d] = ggml_backend_event_new(ggml_backend_get_device(m.backends[d]));
            if (res.use_events[d])
                ggml_backend_event_record(res.use_events[d], m.backends[d]);
        }
    }
    else if (ggml_backend_sched_graph_compute(res.sched, res.gf) != GGML_STATUS_SUCCESS)
    {
        fprintf(stderr, "[dsv4] graph compute failed\n");
        return false;
    }

    auto t_compute = now();

    if (want_logits && logits_out)
        ggml_backend_tensor_get(res.logits, logits_out, 0, (size_t) hp.n_vocab * sizeof(float));

    if (perf >= 2)
    {
        fprintf(stderr, "[dsv4] ubatch nt=%" PRId64 " p0=%" PRId64 ": build %.2fms alloc %.2fms inputs %.2fms compute %.2fms (nodes %d, splits %d%s)\n",
                nt, p0, ms(t0, t_build), ms(t_build, t_alloc), ms(t_alloc, t_inputs), ms(t_inputs, t_compute),
                ggml_graph_n_nodes(res.gf), ggml_backend_sched_get_n_splits(res.sched), reuse ? ", reused" : "");
    }

    if (perf >= 3)
    {
        // report backend transitions in execution order (split boundaries)
        ggml_backend_t prev = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(res.gf); i++)
        {
            ggml_tensor * node = ggml_graph_node(res.gf, i);
            ggml_backend_t be = ggml_backend_sched_get_tensor_backend(res.sched, node);
            if (be != prev)
            {
                fprintf(stderr, "[dsv4]   node %5d %-16s op=%-14s -> %s\n",
                        i, node->name, ggml_op_name(node->op), be ? ggml_backend_name(be) : "?");
                prev = be;
            }
        }
    }

    return true;
}

} // namespace tsg_dsv4

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

TSG_EXPORT void * TSGgml_Dsv4LoadModel(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch, int n_threads)
{
    try
    {
        return tsg_dsv4::dsv4_load(gguf_path, n_gpu, n_ctx, n_ubatch, n_threads);
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[dsv4] load failed: %s\n", e.what());
        return nullptr;
    }
}

TSG_EXPORT int TSGgml_Dsv4VocabSize(void * handle)
{
    if (!handle) return 0;
    return ((tsg_dsv4::dsv4_model *) handle)->hp.n_vocab;
}

TSG_EXPORT int TSGgml_Dsv4CtxSize(void * handle)
{
    if (!handle) return 0;
    return ((tsg_dsv4::dsv4_model *) handle)->n_ctx;
}

TSG_EXPORT int TSGgml_Dsv4NPast(void * handle)
{
    if (!handle) return 0;
    return ((tsg_dsv4::dsv4_model *) handle)->active_slot->n_past;
}

// Feeds `n_tokens` tokens at positions [n_past, n_past + n_tokens) of the
// ACTIVE slot and writes the last token's logits into logits_out (size
// n_vocab). Returns 0 on success, negative on failure.
TSG_EXPORT int TSGgml_Dsv4Forward(void * handle, const int32_t * tokens, int n_tokens, float * logits_out)
{
    if (!handle || !tokens || n_tokens <= 0) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    tsg_dsv4::dsv4_slot * slot = m->active_slot;

    if (slot->n_past + n_tokens > m->n_ctx)
    {
        fprintf(stderr, "[dsv4] context overflow: n_past=%d + %d > n_ctx=%d\n", slot->n_past, n_tokens, m->n_ctx);
        return -2;
    }

    const bool perf = []() { const char * e = getenv("TS_DSV4_PERF"); return e && atoi(e) > 0; }();
    auto t0 = std::chrono::steady_clock::now();

    m->pos_end_hint = (int64_t) slot->n_past + n_tokens;

    // Pre-acquire the first two chunk graphs (the pipelined pair) and the
    // final chunk's graph so the pipelined prefill never stalls on a
    // device-synchronizing arena allocation mid-flight.
    if (n_tokens > m->n_ubatch)
    {
        tsg_dsv4::dsv4_acquire_graph(*m, m->n_ubatch, slot->n_past, m->n_gpu > 1, nullptr);
        const int nt2 = std::min(m->n_ubatch, n_tokens - m->n_ubatch);
        const bool last2 = (2 * m->n_ubatch >= n_tokens);
        tsg_dsv4::dsv4_acquire_graph(*m, nt2, slot->n_past + m->n_ubatch, !last2 && m->n_gpu > 1, nullptr);
        if (!last2)
        {
            const int last_nt = n_tokens - m->n_ubatch * ((n_tokens - 1) / m->n_ubatch);
            tsg_dsv4::dsv4_acquire_graph(*m, last_nt, (int64_t) slot->n_past + (n_tokens - last_nt), false, nullptr);
        }
    }

    int done = 0;
    while (done < n_tokens)
    {
        const int nt = std::min(m->n_ubatch, n_tokens - done);
        const bool last = (done + nt == n_tokens);
        if (!tsg_dsv4::dsv4_forward_ubatch(*m, tokens + done, nt, slot->n_past, last, last ? logits_out : nullptr))
            return -3;
        slot->n_past += nt;
        done += nt;
    }

    if (perf)
    {
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "[dsv4] forward %d tokens in %.3fs (%.1f tok/s)\n", n_tokens, s, n_tokens / s);
    }
    return 0;
}

// Resets the ACTIVE slot's caches to position 0. Weights, rope tables and
// other slots are untouched.
TSG_EXPORT void TSGgml_Dsv4Reset(void * handle)
{
    if (!handle) return;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    tsg_dsv4::dsv4_reset_slot(*m->active_slot);
}

// Allocate a new sequence slot (own caches, shared weights). Returns the slot
// id, or -1 on allocation failure. The new slot is NOT made active.
TSG_EXPORT int TSGgml_Dsv4SlotAlloc(void * handle)
{
    if (!handle) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    try
    {
        tsg_dsv4::dsv4_slot * slot = tsg_dsv4::dsv4_slot_alloc(*m);
        return slot ? slot->id : -1;
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "[dsv4] slot alloc failed: %s\n", e.what());
        return -1;
    }
}

// Select which slot Forward/Reset/NPast operate on. Returns 0 on success,
// -1 for an unknown slot id.
TSG_EXPORT int TSGgml_Dsv4SetActiveSlot(void * handle, int slot_id)
{
    if (!handle) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return -1;
    m->active_slot = it->second.get();
    return 0;
}

// Free a slot's caches and every cached graph built against them (captured
// graphs bake the slot's device addresses). The active slot cannot be freed;
// callers switch to another slot first.
TSG_EXPORT int TSGgml_Dsv4SlotFree(void * handle, int slot_id)
{
    if (!handle) return -1;
    auto * m = (tsg_dsv4::dsv4_model *) handle;
    auto it = m->slots.find(slot_id);
    if (it == m->slots.end()) return -1;
    if (it->second.get() == m->active_slot)
    {
        fprintf(stderr, "[dsv4] refusing to free the active slot %d\n", slot_id);
        return -1;
    }
    for (auto gi = m->graph_cache.begin(); gi != m->graph_cache.end(); )
    {
        if ((*gi)->slot_id == slot_id) gi = m->graph_cache.erase(gi);
        else ++gi;
    }
    m->slots.erase(it);
    return 0;
}

TSG_EXPORT void TSGgml_Dsv4Free(void * handle)
{
    if (!handle) return;
    delete (tsg_dsv4::dsv4_model *) handle;
}
