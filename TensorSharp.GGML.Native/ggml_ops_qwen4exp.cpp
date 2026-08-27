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
#include "ggml-impl.h"
#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

using namespace tsg;

// ============================================================================
// Qwen3.8-Flash-Next (qwen4exp) fused layers.
//
// Three per-layer half-kernels (FFN, GDN, attention), and above them ONE graph
// for a whole run of layers - which in practice means one graph per token, cut
// only where the PLE layer has to read the residual on the host.
//
// Why. A qwen4exp decode step is dispatch bound, not arithmetic bound: the
// op-by-op path issued roughly 850 GGML submissions per token; the per-layer
// fused kernels cut that to 96, but 96 is still 96 graph launches, 96 stream
// drains and 96 host round trips of the residual where llama.cpp has exactly
// one of each. The token-span kernel is that same shape: one graph, the
// residual crossing the PCIe bus twice per token instead of 192 times, and one
// CUDA graph for ggml-cuda to capture instead of 96 alternating ones.
//
// The per-layer entry points remain as the fallback - QSA over budget, foreign
// shapes, debugging - and share the node builders with the span, so there is a
// single source of truth for the graph each half builds.
// ============================================================================
namespace
{
    constexpr const char* kQwen4ExpFfnKernel = "qwen4exp fused FFN block";
    constexpr const char* kQwen4ExpGdnKernel = "qwen4exp fused GDN block";
    constexpr const char* kQwen4ExpAttnKernel = "qwen4exp fused attention block";
    constexpr const char* kQwen4ExpSpanKernel = "qwen4exp fused token span";
    constexpr int kQwen4ExpMaxSlots = 128;
    constexpr int kQwen4ExpSpanSlots = 32;
    // A 48-layer span builds ~150 nodes a layer; 16384 leaves headroom.
    constexpr int kQwen4ExpSpanGraphSize = 16384;

    // A weight binding resolved through the resident cache, remembered so a
    // REPLAY can re-resolve it. The cache can move or re-create a device copy
    // (large allocations elsewhere churn it), and a persisted graph would keep
    // reading the old address forever - the graph-uid stamp even tells ggml-cuda
    // to skip the staleness walk that might have noticed.
    struct Q4eCachedBind
    {
        ggml_tensor* tensor;
        void* data;
        std::size_t bytes;
        ggml_backend_buffer_usage usage;
    };

    // One built graph per layer (or per span), kept across tokens.
    //
    // Building the graph is most of a decode step's cost here: the arithmetic for
    // one token is trivial next to a ggml context, a gallocr plan and a fresh
    // topology 48 times a token. Holding the graph makes a decode step an upload of
    // the residual, a replay and a download - and keeps the topology byte-identical
    // token to token, which is what lets ggml-cuda's graph capture engage.
    struct Qwen4ExpFfnCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_tensor* res_in = nullptr;
        ggml_tensor* res_out = nullptr;
        int n_tokens = 0;
        int hc_dim = 0;
        const void* sig = nullptr;      // the descriptor this graph was built from
        // Span graphs are keyed on all three descriptor arrays and the layer range.
        const void* sig2 = nullptr;
        const void* sig3 = nullptr;
        int layer_begin = -1;
        int layer_end = -1;
        int kv_capacity = -1;
        // Span only: the first layer contributes only its FFN half (its attention
        // half already ran through the per-layer kernel).
        int first_ffn_only = 0;
        // Recurrent state, read through *_in and written through *_out; the caller
        // copies out -> in after every compute. The span writes the state in place
        // instead and only uses the buffer.
        ggml_tensor* conv_in = nullptr;
        ggml_tensor* conv_out = nullptr;
        ggml_tensor* ssm_in = nullptr;
        ggml_tensor* ssm_out = nullptr;
        ggml_backend_buffer_t state_buf = nullptr;
        bool state_ready = false;
        // Whether res_in is bound to the shared device buffer. A graph built one way
        // cannot be replayed the other: a non-resident graph owns a private res_in, so
        // replaying it in resident mode chains nothing and the layers stop composing.
        int res_resident = -1;
        // Span only: state write-backs issued host-side after each compute when
        // TS_Q4E_SPAN_STATE=host - (src, dst) pairs, src a graph output, dst the
        // persistent state tensor.
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> span_copies;
        std::vector<Q4eCachedBind> rebinds;
        // Span attention inputs, one set per in-span attention layer. Private per
        // layer, exactly as the per-layer kernels have them.
        std::vector<ggml_tensor*> span_masks;
        std::vector<ggml_tensor*> span_pos;
        std::vector<ggml_tensor*> span_kvidx;
        std::vector<ggml_tensor*> gdn_probe;
        // Attention only: the mask is an input and the graph shape follows n_kv.
        ggml_tensor* mask = nullptr;
        ggml_tensor* pos = nullptr;
        ggml_tensor* kv_idx = nullptr;
        int n_kv = -1;

        // Drop the graph but KEEP the recurrent state: a shape change does this.
        void reset_graph()
        {
            if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
            if (ctx) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; res_in = nullptr; res_out = nullptr;
            conv_in = conv_out = ssm_in = ssm_out = nullptr;
            valid = false; n_tokens = 0; hc_dim = 0; sig = nullptr; res_resident = -1;
            sig2 = nullptr; sig3 = nullptr;
            layer_begin = -1; layer_end = -1; kv_capacity = -1; first_ffn_only = 0;
            mask = nullptr; pos = nullptr; kv_idx = nullptr; n_kv = -1;
            span_copies.clear(); rebinds.clear(); gdn_probe.clear();
            span_masks.clear(); span_pos.clear(); span_kvidx.clear();
        }

        // Drop everything including the state: a KV reset does this.
        void reset()
        {
            reset_graph();
            if (state_buf) { ggml_backend_buffer_free(state_buf); state_buf = nullptr; }
            state_ready = false;
        }
    };

    Qwen4ExpFfnCache g_q4e_ffn[kQwen4ExpMaxSlots];
    Qwen4ExpFfnCache g_q4e_gdn[kQwen4ExpMaxSlots];
    Qwen4ExpFfnCache g_q4e_attn[kQwen4ExpMaxSlots];
    Qwen4ExpFfnCache g_q4e_span[kQwen4ExpSpanSlots];

    // ggml-cuda's flash attention takes F16 K/V and one of a fixed set of head sizes;
    // for head_dim 256 the only other condition is V->ne[0] == K->ne[0], which holds
    // here. TS_Q4E_FLASH_ATTN=0 falls back to the soft_max path.
    bool q4e_flash_attn_ok(int kv_type, int head_dim)
    {
        static const bool enabled = []{
            const char* e = std::getenv("TS_Q4E_FLASH_ATTN");
            return !(e != nullptr && e[0] == '0');
        }();
        if (!enabled || kv_type != GGML_TYPE_F16) return false;
        switch (head_dim)
        {
            case 64: case 80: case 96: case 112: case 128: case 256: return true;
            default: return false;
        }
    }

    // ggml-cuda picks its GQA-optimised flash-attention kernel only when
    // K->ne[1] % FATTN_KQ_STRIDE == 0, so the window is padded to that and the pad
    // masked off. This is what llama.cpp's get_n_kv rounds to, for the same reason.
    // The padding is only worth it under flash attention: the soft_max path pays for
    // every padded column instead of skipping the block.
    constexpr int kQwen4ExpKvStride = 256;

    // The span carries the GDN state through separate output tensors copied back
    // after each compute - the per-layer dataflow, and the DEFAULT. Writing the
    // state in place inside the graph (TS_Q4E_SPAN_STATE=graph) is faster on paper
    // and measurably WRONG in practice: the historical in-place failure reproduced
    // exactly, with node order controlled, CUDA graph capture off and buffers
    // zeroed. It stays available only for investigation.
    // TS_Q4E_SPAN_REBUILD=1 disables the span replay path entirely - every call
    // rebuilds the graph. Diagnosis only: separates a wrong-graph bug from a
    // wrong-replay one.
    bool q4e_span_force_rebuild()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_REBUILD");
            return e != nullptr && e[0] == '1';
        }();
        return v;
    }

    // TS_Q4E_SPAN_FA_MAX=N: only the first N attention layers in a span use flash
    // attention; the rest run the soft_max path over the SAME padded window.
    // Diagnosis only - N=0 separates "the pad poisons the output" from "the flash
    // attention node does".
    int q4e_span_fa_max()
    {
        static const int v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_FA_MAX");
            return (e != nullptr && *e != 0) ? std::atoi(e) : 1 << 30;
        }();
        return v;
    }

    // TS_Q4E_SPAN_TRACE=1 prints the residual L2 norm after every layer of every
    // span call. Diagnosis only: diffing a good run against a bad one names the
    // first layer whose output moves.
    bool q4e_span_trace()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_TRACE");
            return e != nullptr && e[0] == '1';
        }();
        return v;
    }

    bool q4e_span_state_in_graph()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_SPAN_STATE");
            return e != nullptr && e[0] == 'g';
        }();
        return v;
    }

    // Re-resolve a persisted graph's cache-bound weights before a replay. If the
    // resident cache moved (or re-created) any device copy, the graph's captured
    // pointers are stale: report it so the caller rebuilds. Content refreshes
    // (needs_upload with an unmoved pointer) are handled in place.
    bool q4e_refresh_bindings(Qwen4ExpFfnCache* slot, ggml_backend_dev_t dev)
    {
        for (const Q4eCachedBind& cb : slot->rebinds)
        {
            void* before = cb.tensor->data;
            bool needs_upload = false;
            if (!try_bind_cached_tensor(g_backend, dev, cb.tensor, cb.data, cb.bytes,
                                        needs_upload, cb.usage))
                return false;
            if (cb.tensor->data != before)
            {
                fprintf(stderr, "[q4e] cached weight moved (%p -> %p, %zu bytes); rebuilding the graph%c",
                        before, cb.tensor->data, cb.bytes, 10);
                return false;
            }
            if (needs_upload)
                ggml_backend_tensor_set(cb.tensor, cb.data, 0, cb.bytes);
        }
        return true;
    }

    // Print the L2 of the probed GDN nodes. Diagnosis only.
    void q4e_trace_probe(Qwen4ExpFfnCache* slot, const char* tag, int position)
    {
        if (slot->gdn_probe.empty()) return;
        static const char* names[] = { "mixed", "qkv", "convout", "gdnout", "proj" };
        fprintf(stderr, "[q4e-gdn0] %s pos=%d:", tag, position);
        std::vector<float> buf;
        for (std::size_t i = 0; i < slot->gdn_probe.size() && i < 5; ++i)
        {
            ggml_tensor* t = slot->gdn_probe[i];
            buf.resize((std::size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
            double n2 = 0.0;
            for (float f : buf) n2 += (double)f * f;
            fprintf(stderr, " %s=%.9e", names[i], std::sqrt(n2));
        }
        fprintf(stderr, "%c", 10);
    }

    // Print the L2 of every state tensor a span carries. Diagnosis only.
    void q4e_trace_state(Qwen4ExpFfnCache* slot, const char* tag, int position)
    {
        if (!q4e_span_trace() || slot->span_copies.empty()) return;
        fprintf(stderr, "[q4e-state] %s pos=%d:", tag, position);
        std::vector<float> buf;
        for (std::size_t i = 0; i < slot->span_copies.size(); ++i)
        {
            ggml_tensor* st = slot->span_copies[i].second;
            buf.resize((std::size_t)ggml_nelements(st));
            ggml_backend_tensor_get(st, buf.data(), 0, ggml_nbytes(st));
            double n2 = 0.0;
            for (float f : buf) n2 += (double)f * f;
            fprintf(stderr, " %.9e", std::sqrt(n2));
        }
        fprintf(stderr, "%c", 10);
    }

    int q4e_pad_kv(int n_kv, int kv_capacity, bool use_flash)
    {
        const int stride = use_flash ? kQwen4ExpKvStride : 1;
        int n_kv_pad = ((n_kv + stride - 1) / stride) * stride;
        if (n_kv_pad > kv_capacity) n_kv_pad = kv_capacity;
        return n_kv_pad;
    }

    // Fill the two index inputs: RoPE positions and the KV rows this step writes.
    // Values change per token, shapes do not - which is what lets the graph persist.
    void q4e_set_attn_indices(ggml_tensor* pos, ggml_tensor* kv_idx, int T, int position)
    {
        std::vector<int32_t> p((std::size_t)T);
        std::vector<int64_t> k((std::size_t)T);
        for (int i = 0; i < T; ++i) { p[i] = position + i; k[i] = position + i; }
        ggml_backend_tensor_set(pos, p.data(), 0, (std::size_t)T * sizeof(int32_t));
        ggml_backend_tensor_set(kv_idx, k.data(), 0, (std::size_t)T * sizeof(int64_t));
    }

    // TS_Q4E_LOG=1 reports how often each kernel rebuilt its graph rather than
    // replaying it.
    struct Q4eStat { long builds = 0; long replays = 0; };
    Q4eStat g_q4e_stat[4];   // 0 = FFN, 1 = GDN, 2 = ATTN, 3 = SPAN
    bool q4e_log_enabled()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_LOG");
            return e != nullptr && e[0] == '1';
        }();
        return v;
    }
    void q4e_note(int k, bool build)
    {
        if (!q4e_log_enabled()) return;
        if (build) g_q4e_stat[k].builds++; else g_q4e_stat[k].replays++;
        long total = 0;
        for (const Q4eStat& st : g_q4e_stat) total += st.builds + st.replays;
        if (total % 2000 == 0 || (k == 3 && (g_q4e_stat[3].builds + g_q4e_stat[3].replays) % 200 == 0))
            fprintf(stderr, "[q4e] ffn b=%ld r=%ld | gdn b=%ld r=%ld | attn b=%ld r=%ld | span b=%ld r=%ld\n",
                    g_q4e_stat[0].builds, g_q4e_stat[0].replays,
                    g_q4e_stat[1].builds, g_q4e_stat[1].replays,
                    g_q4e_stat[2].builds, g_q4e_stat[2].replays,
                    g_q4e_stat[3].builds, g_q4e_stat[3].replays);
    }

    // Stamp every persisted graph with a stable non-zero id.
    //
    // ggml_new_graph leaves uid at 0, and ggml-cuda treats 0 as "unknown", so on every
    // replay it re-walks the nodes comparing a copy of each tensor struct and its
    // sources to decide whether the captured CUDA graph is still valid. With a stable
    // id it recognises the graph and skips that walk.
    // TS_Q4E_GRAPH_UID=0 leaves uid at 0 so ggml-cuda re-checks node properties on
    // every replay instead of trusting the id.
    bool q4e_graph_uid_enabled()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_Q4E_GRAPH_UID");
            return !(e != nullptr && e[0] == '0');
        }();
        return v;
    }

    uint64_t q4e_next_graph_uid()
    {
        static uint64_t next = 1;
        return next++;
    }

    // The 4-wide residual, held on the DEVICE for the whole forward.
    //
    // Used by the per-layer fallback's residency experiment; the token span does not
    // need it - inside one graph the residual never exists on the host at all.
    ggml_backend_buffer_t g_q4e_res_buf = nullptr;
    std::size_t g_q4e_res_capacity = 0;
    ggml_context* g_q4e_res_ctx = nullptr;
    ggml_tensor* g_q4e_res = nullptr;

    // Ensure the shared residual tensor exists and is at least `bytes` big.
    bool q4e_res_ensure(std::size_t bytes)
    {
        if (g_q4e_res != nullptr && g_q4e_res_capacity >= bytes)
            return true;
        // Every persisted graph binds its res_in to this buffer's base, so a realloc
        // here strands every one of them. Loud on purpose.
        if (g_q4e_res != nullptr)
            fprintf(stderr, "[q4e] residual buffer REALLOC %zu -> %zu bytes\n",
                    g_q4e_res_capacity, bytes);
        if (g_q4e_res_ctx) { ggml_free(g_q4e_res_ctx); g_q4e_res_ctx = nullptr; }
        if (g_q4e_res_buf) { ggml_backend_buffer_free(g_q4e_res_buf); g_q4e_res_buf = nullptr; }
        g_q4e_res = nullptr;

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 4;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        g_q4e_res_ctx = ggml_init(ip);
        if (g_q4e_res_ctx == nullptr) return false;

        g_q4e_res = ggml_new_tensor_1d(g_q4e_res_ctx, GGML_TYPE_F32, (int64_t)(bytes / sizeof(float)));
        g_q4e_res_buf = ggml_backend_buft_alloc_buffer(
                ggml_backend_get_default_buffer_type(g_backend), bytes);
        if (g_q4e_res_buf == nullptr) return false;
        if (ggml_backend_tensor_alloc(g_q4e_res_buf, g_q4e_res,
                ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            return false;
        g_q4e_res_capacity = bytes;
        return true;
    }

    // One place for the weight-binding policy the three block builders shared as a
    // copy-pasted lambda each. Collecting the uploads here rather than binding
    // immediately is what lets several layers build into one graph: everything is
    // bound before a single ggml_gallocr_alloc_graph runs over the lot.
    struct Q4eHostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };

    struct Q4eBinder
    {
        ggml_backend_dev_t dev = nullptr;
        std::vector<Q4eHostBinding> upload_list;
        std::vector<Q4eCachedBind> cached;

        void add(ggml_tensor* tgt, void* data, std::size_t bytes,
                 ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        {
            if (tgt == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload, usage))
                {
                    cached.push_back({tgt, data, bytes, usage});
                    if (needs_upload) upload_list.push_back({tgt, data, bytes});
                    return;
                }
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, true, buf))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
                // A weight this size normally cache-binds; falling through here means
                // the resident cache could not take it (VRAM pressure). It will live
                // in the graph's own allocation instead - flagged below - and that is
                // worth being able to see.
                fprintf(stderr, "[q4e] weight (%zu bytes) fell out of the resident cache into the graph allocation%c",
                        bytes, 10);
            }
            // The tensor becomes a gallocr-owned leaf, uploaded once at build time.
            // It MUST be flagged as an input: gallocr treats an unflagged leaf like
            // any intermediate and reuses its memory after the last consumer, so the
            // FIRST compute overwrites the weight in place and every REPLAY of the
            // graph reads garbage. A build works exactly once; that is precisely the
            // difference between a rebuilt-per-token graph that stays correct and a
            // persisted one that decays.
            ggml_set_input(tgt);
            upload_list.push_back({tgt, data, bytes});
        }

        void flush()
        {
            for (const Q4eHostBinding& hb : upload_list)
                ggml_backend_tensor_set(hb.tensor, hb.data, 0, hb.bytes);
            upload_list.clear();
        }
    };
}

extern "C"
{

// Per-layer weights. Pointers first, then int64, then int32 - the layout the
// C# side mirrors; append within a run rather than reordering.
struct TSGgmlQwen4ExpFfnArgs
{
    // hyper-connection mixer
    void* hc_norm;          // f32 [hc_dim], gamma folded to (1 + w)
    void* hc_down;          // [hc_dim, hc_low_rank]
    void* hc_up;            // [hc_low_rank, hc_dim]
    void* hc_inject;        // [hc_dim, hc]
    // MoE
    void* router;           // [n_embd, n_expert]
    void* gate_exps;        // [n_embd, n_ff, n_expert]
    void* up_exps;          // [n_embd, n_ff, n_expert]
    void* down_exps;        // [n_ff, n_embd, n_expert]
    // shared expert
    void* sh_gate_inp;      // f32 [n_embd] - one sigmoid scalar per token
    void* sh_gate;          // [n_embd, n_ff_sh]
    void* sh_up;            // [n_embd, n_ff_sh]
    void* sh_down;          // [n_ff_sh, n_embd]

    long long hc_down_bytes, hc_up_bytes, hc_inject_bytes;
    long long router_bytes, gate_exps_bytes, up_exps_bytes, down_exps_bytes;
    long long sh_gate_bytes, sh_up_bytes, sh_down_bytes;

    int hc_down_type, hc_up_type, hc_inject_type;
    int router_type, gate_exps_type, up_exps_type, down_exps_type;
    int sh_gate_type, sh_up_type, sh_down_type;
};

// Per-layer weights for the recurrent (Gated DeltaNet) half of a layer.
struct TSGgmlQwen4ExpGdnArgs
{
    // hyper-connection mixer
    void* hc_norm;
    void* hc_down;
    void* hc_up;
    void* hc_inject;
    // delta net
    void* qkv;              // [n_embd, conv_dim]
    void* gate;             // [n_embd, value_dim]
    void* beta;             // [n_embd, n_v_heads]
    void* alpha;            // [n_embd, n_v_heads]
    void* conv1d;           // f32 [d_conv, conv_dim]
    void* ssm_dt;           // f32 [n_v_heads]
    void* ssm_a;            // f32 [n_v_heads], pre-negated
    void* ssm_norm;         // f32 [head_v_dim]
    void* out_proj;         // [value_dim, n_embd]
    // state, updated in place
    void* conv_state;       // f32 [d_conv-1, conv_dim]
    void* ssm_state;        // f32 [head_v_dim, head_v_dim, n_v_heads]

    long long hc_down_bytes, hc_up_bytes, hc_inject_bytes;
    long long qkv_bytes, gate_bytes, beta_bytes, alpha_bytes, out_proj_bytes;

    int hc_down_type, hc_up_type, hc_inject_type;
    int qkv_type, gate_type, beta_type, alpha_type, out_proj_type;
};

// Per-layer weights for the full-attention half of a layer.
struct TSGgmlQwen4ExpAttnArgs
{
    void* hc_norm;
    void* hc_down;
    void* hc_up;
    void* hc_inject;
    void* wq;               // [n_embd, head_dim * n_head * 2]  (query|gate interleaved)
    void* wk;               // [n_embd, head_dim * n_head_kv]
    void* wv;
    void* wo;               // [head_dim * n_head, n_embd]
    void* q_norm;           // f32 [head_dim]
    void* k_norm;           // f32 [head_dim]
    void* k_cache;          // f16/f32 [head_dim, capacity, n_head_kv]
    void* v_cache;

    long long hc_down_bytes, hc_up_bytes, hc_inject_bytes;
    long long wq_bytes, wk_bytes, wv_bytes, wo_bytes;
    long long kv_bytes;     // per cache

    int hc_down_type, hc_up_type, hc_inject_type;
    int wq_type, wk_type, wv_type, wo_type;
    int kv_type;
};

// ============================================================================
// Node builders. Each appends one half-layer to ctx and returns the new
// residual; the entry points and the token span share them, so the graph a
// half builds has a single source of truth.
// ============================================================================

// FFN half: hyper-connection mixer -> routed experts + gated shared expert ->
// hyper-connection scatter. No side effects; expands nothing.
static ggml_tensor* q4e_nodes_ffn(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpFfnArgs* a, ggml_tensor* res_in,
    int n_embd, int hc, int hc_low_rank, int T,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh, float eps)
{
    const int hc_dim = hc * n_embd;
    ggml_tensor* w_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor* w_down = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_down_type, hc_dim, hc_low_rank);
    ggml_tensor* w_up = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_up_type, hc_low_rank, hc_dim);
    ggml_tensor* w_inject = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_inject_type, hc_dim, hc);
    ggml_tensor* w_router = ggml_new_tensor_2d(ctx, (ggml_type)a->router_type, n_embd, n_expert);
    ggml_tensor* w_gate_e = ggml_new_tensor_3d(ctx, (ggml_type)a->gate_exps_type, n_embd, n_ff, n_expert);
    ggml_tensor* w_up_e = ggml_new_tensor_3d(ctx, (ggml_type)a->up_exps_type, n_embd, n_ff, n_expert);
    ggml_tensor* w_down_e = ggml_new_tensor_3d(ctx, (ggml_type)a->down_exps_type, n_ff, n_embd, n_expert);
    ggml_tensor* w_sh_gi = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor* w_sh_g = ggml_new_tensor_2d(ctx, (ggml_type)a->sh_gate_type, n_embd, n_ff_sh);
    ggml_tensor* w_sh_u = ggml_new_tensor_2d(ctx, (ggml_type)a->sh_up_type, n_embd, n_ff_sh);
    ggml_tensor* w_sh_d = ggml_new_tensor_2d(ctx, (ggml_type)a->sh_down_type, n_ff_sh, n_embd);

    // ---- hyper-connection mixer ------------------------------------------
    // Grouped RMS norm: normalise over ONE residual stream, then scale the
    // whole hc-wide row by the gamma vector.
    ggml_tensor* res3 = ggml_reshape_3d(ctx, res_in, n_embd, hc, T);
    ggml_tensor* xn = ggml_rms_norm(ctx, res3, eps);
    xn = ggml_reshape_2d(ctx, xn, hc_dim, T);
    xn = ggml_mul(ctx, xn, w_norm);

    ggml_tensor* lo = ggml_mul_mat(ctx, w_down, xn);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float)hc));
    ggml_tensor* gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));

    ggml_tensor* gated = ggml_mul(ctx, xn, gate);
    gated = ggml_reshape_3d(ctx, gated, n_embd, hc, T);

    // Collapse the streams by their mean.
    ggml_tensor* mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, T,
            ggml_row_size(gated->type, n_embd) * hc, 0));
    for (int c = 1; c < hc; ++c)
    {
        ggml_tensor* s = ggml_view_2d(ctx, gated, n_embd, T,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx, mixed, s);
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float)hc);

    ggml_tensor* inject = ggml_mul_mat(ctx, w_inject, xn);   // [hc, T]

    // ---- routed experts ---------------------------------------------------
    // Softmax over every expert, top-k, then renormalise the selected
    // weights - llama.cpp's build_moe_ffn with norm_w.
    ggml_tensor* logits = ggml_mul_mat(ctx, w_router, mixed);      // [n_expert, T]
    ggml_tensor* probs = ggml_soft_max(ctx, logits);
    // ggml_argsort_top_k, not ggml_top_k: this is the exact node shape llama.cpp's
    // build_moe_ffn emits, and ggml-cuda's topk_moe fusion matches on the node
    // sequence rather than on intent.
    ggml_tensor* sel = ggml_argsort_top_k(ctx, probs, n_expert_used);  // [n_used, T] i32

    ggml_tensor* w_sel = ggml_get_rows(ctx,
            ggml_reshape_3d(ctx, probs, 1, n_expert, T), sel);     // [1, n_used, T]
    w_sel = ggml_reshape_2d(ctx, w_sel, n_expert_used, T);
    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_sel);                // [1, T]
    w_sel = ggml_div(ctx, w_sel, w_sum);
    w_sel = ggml_reshape_3d(ctx, w_sel, 1, n_expert_used, T);

    ggml_tensor* moe_in = ggml_reshape_3d(ctx, mixed, n_embd, 1, T);
    ggml_tensor* e_up = ggml_mul_mat_id(ctx, w_up_e, moe_in, sel);      // [n_ff, n_used, T]
    ggml_tensor* e_gate = ggml_mul_mat_id(ctx, w_gate_e, moe_in, sel);
    ggml_tensor* par = ggml_mul(ctx, ggml_silu(ctx, e_gate), e_up);
    ggml_tensor* experts = ggml_mul_mat_id(ctx, w_down_e, par, sel);    // [n_embd, n_used, T]
    experts = ggml_mul(ctx, experts, w_sel);

    ggml_tensor* moe_out = ggml_view_2d(ctx, experts, n_embd, T,
            experts->nb[2], 0);
    for (int k = 1; k < n_expert_used; ++k)
    {
        ggml_tensor* s = ggml_view_2d(ctx, experts, n_embd, T,
                experts->nb[2], (std::size_t)k * experts->nb[1]);
        moe_out = ggml_add(ctx, moe_out, s);
    }

    // ---- shared expert, behind its own sigmoid scalar ---------------------
    ggml_tensor* sg = ggml_mul_mat(ctx, w_sh_g, mixed);
    ggml_tensor* su = ggml_mul_mat(ctx, w_sh_u, mixed);
    ggml_tensor* sh = ggml_mul_mat(ctx, w_sh_d, ggml_mul(ctx, ggml_silu(ctx, sg), su));
    ggml_tensor* s_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_sh_gi, mixed)); // [1, T]
    ggml_tensor* ffn_out = ggml_add(ctx, moe_out, ggml_mul(ctx, sh, s_gate));

    // ---- hyper-connection scatter ----------------------------------------
    // 2*sigmoid centres the weights on 1, so an untrained injection matrix
    // reproduces a plain residual add.
    ggml_tensor* wsc = ggml_scale(ctx, ggml_sigmoid(ctx,
            ggml_scale(ctx, inject, 1.0f / (float)hc)), 2.0f);
    wsc = ggml_reshape_3d(ctx, wsc, 1, hc, T);

    ggml_tensor* b = ggml_reshape_3d(ctx, ffn_out, n_embd, 1, T);
    b = ggml_repeat_4d(ctx, b, n_embd, hc, T, 1);

    ggml_tensor* res_out = ggml_add(ctx, res3, ggml_mul(ctx, b, wsc));
    res_out = ggml_reshape_2d(ctx, res_out, hc_dim, T);

    bnd.add(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
    bnd.add(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
    bnd.add(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
    bnd.add(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
    bnd.add(w_router, a->router, (std::size_t)a->router_bytes);
    bnd.add(w_gate_e, a->gate_exps, (std::size_t)a->gate_exps_bytes);
    bnd.add(w_up_e, a->up_exps, (std::size_t)a->up_exps_bytes);
    bnd.add(w_down_e, a->down_exps, (std::size_t)a->down_exps_bytes);
    bnd.add(w_sh_gi, a->sh_gate_inp, (std::size_t)n_embd * sizeof(float));
    bnd.add(w_sh_g, a->sh_gate, (std::size_t)a->sh_gate_bytes);
    bnd.add(w_sh_u, a->sh_up, (std::size_t)a->sh_up_bytes);
    bnd.add(w_sh_d, a->sh_down, (std::size_t)a->sh_down_bytes);

    return res_out;
}

// GDN half: hyper-connection mixer -> projections -> causal conv ->
// gated delta net -> sigmoid-gated norm -> out proj -> scatter.
//
// The caller owns conv_state / ssm_state (created in ctx, allocated into the
// layer's persistent state buffer) and decides how the write-back reaches
// them: the per-layer entry copies out -> in after the compute exactly as it
// always has; the span expands cpy(tail -> conv_state) into the graph AFTER
// the nodes that read the state, so node order sequences the write behind the
// read. The write-back sources come out through `wb`.
struct Q4eGdnWriteback { ggml_tensor* tail; ggml_tensor* new_state; };

static ggml_tensor* q4e_nodes_gdn(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpGdnArgs* a, ggml_tensor* res_in,
    ggml_tensor* conv_state, ggml_tensor* ssm_state,
    int n_embd, int hc, int hc_low_rank, int T,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    float eps, Q4eGdnWriteback* wb,
    std::vector<ggml_tensor*>* probe = nullptr)
{
    const int hc_dim = hc * n_embd;
    const int key_dim = head_k_dim * n_k_heads;
    const int value_dim = head_v_dim * n_v_heads;
    const int conv_dim = key_dim * 2 + value_dim;
    const int hist = d_conv - 1;

    ggml_tensor* w_norm    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor* w_down    = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_down_type, hc_dim, hc_low_rank);
    ggml_tensor* w_up      = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_up_type, hc_low_rank, hc_dim);
    ggml_tensor* w_inject  = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_inject_type, hc_dim, hc);
    ggml_tensor* w_qkv     = ggml_new_tensor_2d(ctx, (ggml_type)a->qkv_type, n_embd, conv_dim);
    ggml_tensor* w_gate    = ggml_new_tensor_2d(ctx, (ggml_type)a->gate_type, n_embd, value_dim);
    ggml_tensor* w_beta    = ggml_new_tensor_2d(ctx, (ggml_type)a->beta_type, n_embd, n_v_heads);
    ggml_tensor* w_alpha   = ggml_new_tensor_2d(ctx, (ggml_type)a->alpha_type, n_embd, n_v_heads);
    ggml_tensor* w_conv    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, conv_dim);
    ggml_tensor* w_dt      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_v_heads);
    ggml_tensor* w_a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_v_heads);
    ggml_tensor* w_ssmnorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
    ggml_tensor* w_out     = ggml_new_tensor_2d(ctx, (ggml_type)a->out_proj_type, value_dim, n_embd);

    // ---- hyper-connection mixer ----
    ggml_tensor* res3 = ggml_reshape_3d(ctx, res_in, n_embd, hc, T);
    ggml_tensor* xn = ggml_rms_norm(ctx, res3, eps);
    xn = ggml_reshape_2d(ctx, xn, hc_dim, T);
    xn = ggml_mul(ctx, xn, w_norm);

    ggml_tensor* lo = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, w_down, xn), 1.0f / (float)hc));
    ggml_tensor* gt = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));
    ggml_tensor* gated = ggml_reshape_3d(ctx, ggml_mul(ctx, xn, gt), n_embd, hc, T);

    ggml_tensor* mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, T,
            ggml_row_size(gated->type, n_embd) * hc, 0));
    for (int c = 1; c < hc; ++c)
    {
        mixed = ggml_add(ctx, mixed, ggml_view_2d(ctx, gated, n_embd, T,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c));
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float)hc);
    ggml_tensor* inject = ggml_mul_mat(ctx, w_inject, xn);

    // ---- projections ----
    ggml_tensor* qkv = ggml_mul_mat(ctx, w_qkv, mixed);        // [conv_dim, T]
    ggml_tensor* z = ggml_mul_mat(ctx, w_gate, mixed);         // [value_dim, T]
    ggml_tensor* beta_raw = ggml_mul_mat(ctx, w_beta, mixed);  // [n_v_heads, T]
    ggml_tensor* alpha_raw = ggml_mul_mat(ctx, w_alpha, mixed);

    // ---- causal depthwise conv over the ring history ----
    // conv_state is [hist, conv_dim]; qkv transposed is [T, conv_dim].
    ggml_tensor* qkv_t = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, qkv)),
            T, conv_dim, 1);
    ggml_tensor* conv_in = ggml_concat(ctx, conv_state, qkv_t, 0); // [hist + T, conv_dim, 1]
    ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, w_conv)); // [conv_dim, T, 1]

    // keep the last `hist` columns for the next token
    ggml_tensor* tail = ggml_cont(ctx, ggml_view_3d(ctx, conv_in, hist, conv_dim, 1,
            conv_in->nb[1], conv_in->nb[2], ggml_row_size(conv_in->type, T)));

    // ---- delta net ----
    ggml_tensor* q = ggml_view_3d(ctx, conv_out, head_k_dim, n_k_heads, T,
            ggml_row_size(conv_out->type, head_k_dim), conv_out->nb[1], 0);
    ggml_tensor* k = ggml_view_3d(ctx, conv_out, head_k_dim, n_k_heads, T,
            ggml_row_size(conv_out->type, head_k_dim), conv_out->nb[1],
            ggml_row_size(conv_out->type, key_dim));
    ggml_tensor* v = ggml_view_3d(ctx, conv_out, head_v_dim, n_v_heads, T,
            ggml_row_size(conv_out->type, head_v_dim), conv_out->nb[1],
            ggml_row_size(conv_out->type, 2 * key_dim));

    q = ggml_l2_norm(ctx, ggml_cont(ctx, q), eps);
    k = ggml_l2_norm(ctx, ggml_cont(ctx, k), eps);

    // Repeat q/k up to the value-head count. ggml_repeat TILES (head h reads
    // h % n_k_heads), which is the convention Qwen 3.5's kernel and llama.cpp's
    // non-fused path both use; leaving it to the op's own broadcast produced
    // fluent-looking noise.
    if (n_k_heads != n_v_heads)
    {
        q = ggml_repeat_4d(ctx, q, head_k_dim, n_v_heads, T, 1);
        k = ggml_repeat_4d(ctx, k, head_k_dim, n_v_heads, T, 1);
    }
    q = ggml_reshape_4d(ctx, ggml_cont(ctx, q), head_k_dim, n_v_heads, T, 1);
    k = ggml_reshape_4d(ctx, ggml_cont(ctx, k), head_k_dim, n_v_heads, T, 1);
    v = ggml_reshape_4d(ctx, ggml_cont(ctx, v), head_v_dim, n_v_heads, T, 1);

    // The op scales q internally (llama.cpp passes it unscaled), and a uniform
    // scale here would be absorbed by the RMS norm below in any case.

    ggml_tensor* b4 = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, beta_raw), 1, n_v_heads, T, 1);
    ggml_tensor* g4 = ggml_reshape_4d(ctx,
            ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha_raw, w_dt)), w_a),
            1, n_v_heads, T, 1);
    ggml_tensor* s4 = ggml_reshape_4d(ctx, ssm_state, head_v_dim, head_v_dim, n_v_heads, 1);

    ggml_tensor* gdn_out = ggml_gated_delta_net(ctx, q, k, v, g4, b4, s4, 1);

    const int64_t attn_elems = (int64_t)head_v_dim * n_v_heads * T;
    ggml_tensor* core = ggml_view_3d(ctx, gdn_out, head_v_dim, n_v_heads, T,
            ggml_row_size(gdn_out->type, head_v_dim),
            ggml_row_size(gdn_out->type, head_v_dim * n_v_heads), 0);
    ggml_tensor* new_state = ggml_view_3d(ctx, gdn_out, head_v_dim, head_v_dim, n_v_heads,
            ggml_row_size(gdn_out->type, head_v_dim),
            ggml_row_size(gdn_out->type, head_v_dim * head_v_dim),
            ggml_row_size(gdn_out->type, attn_elems));

    // qwen4exp closes with a SIGMOID gate, where Qwen 3.5 uses SiLU.
    ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, core), eps), w_ssmnorm);
    ggml_tensor* zg = ggml_sigmoid(ctx, ggml_reshape_3d(ctx, z, head_v_dim, n_v_heads, T));
    ggml_tensor* out2 = ggml_reshape_2d(ctx, ggml_mul(ctx, normed, zg), value_dim, T);
    ggml_tensor* proj = ggml_mul_mat(ctx, w_out, out2);        // [n_embd, T]

    // ---- hyper-connection scatter ----
    ggml_tensor* wsc = ggml_reshape_3d(ctx, ggml_scale(ctx,
            ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float)hc)), 2.0f), 1, hc, T);
    ggml_tensor* bexp = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, proj, n_embd, 1, T),
            n_embd, hc, T, 1);
    ggml_tensor* res_out = ggml_reshape_2d(ctx,
            ggml_add(ctx, res3, ggml_mul(ctx, bexp, wsc)), hc_dim, T);

    bnd.add(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
    bnd.add(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
    bnd.add(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
    bnd.add(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
    bnd.add(w_qkv, a->qkv, (std::size_t)a->qkv_bytes);
    bnd.add(w_gate, a->gate, (std::size_t)a->gate_bytes);
    bnd.add(w_beta, a->beta, (std::size_t)a->beta_bytes);
    bnd.add(w_alpha, a->alpha, (std::size_t)a->alpha_bytes);
    bnd.add(w_conv, a->conv1d, (std::size_t)d_conv * conv_dim * sizeof(float));
    bnd.add(w_dt, a->ssm_dt, (std::size_t)n_v_heads * sizeof(float));
    bnd.add(w_a, a->ssm_a, (std::size_t)n_v_heads * sizeof(float));
    bnd.add(w_ssmnorm, a->ssm_norm, (std::size_t)head_v_dim * sizeof(float));
    bnd.add(w_out, a->out_proj, (std::size_t)a->out_proj_bytes);

    if (probe != nullptr)
    {
        ggml_tensor* nodes[] = { mixed, qkv, conv_out, gdn_out, proj };
        for (ggml_tensor* t : nodes) { ggml_set_output(t); probe->push_back(t); }
    }

    wb->tail = tail;
    wb->new_state = new_state;
    return res_out;
}

// Attention half: hyper-connection mixer -> joint query|gate -> Q/K norm ->
// partial rotary -> KV append -> (flash) attention -> sigmoid gate -> out
// proj -> scatter.
//
// The KV write is expanded into `graph` HERE, before the caller expands the
// returned residual: nothing in the residual's tree depends on the write -
// k_full is a plain view of the cache and ggml does not treat view aliasing as
// an edge - so node order is the only thing sequencing the write against the
// read, and this token has to be able to attend to itself.
static ggml_tensor* q4e_nodes_attn(
    ggml_context* ctx, ggml_cgraph* graph, Q4eBinder& bnd,
    const TSGgmlQwen4ExpAttnArgs* a, ggml_tensor* res_in,
    ggml_tensor* mask, ggml_tensor* pos, ggml_tensor* kv_idx,
    int n_embd, int hc, int hc_low_rank, int T,
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv_pad,
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale,
    float eps, bool use_flash,
    std::vector<ggml_tensor*>* kv_out = nullptr,
    std::vector<ggml_tensor*>* probe = nullptr)
{
    const int hc_dim = hc * n_embd;
    const int q_dim = head_dim * n_head;

    ggml_tensor* w_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor* w_down   = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_down_type, hc_dim, hc_low_rank);
    ggml_tensor* w_up     = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_up_type, hc_low_rank, hc_dim);
    ggml_tensor* w_inject = ggml_new_tensor_2d(ctx, (ggml_type)a->hc_inject_type, hc_dim, hc);
    ggml_tensor* wq       = ggml_new_tensor_2d(ctx, (ggml_type)a->wq_type, n_embd, q_dim * 2);
    ggml_tensor* wk       = ggml_new_tensor_2d(ctx, (ggml_type)a->wk_type, n_embd, head_dim * n_head_kv);
    ggml_tensor* wv       = ggml_new_tensor_2d(ctx, (ggml_type)a->wv_type, n_embd, head_dim * n_head_kv);
    ggml_tensor* wo       = ggml_new_tensor_2d(ctx, (ggml_type)a->wo_type, q_dim, n_embd);
    ggml_tensor* q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    ggml_tensor* k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    ggml_tensor* k_cache  = ggml_new_tensor_3d(ctx, (ggml_type)a->kv_type, head_dim, kv_capacity, n_head_kv);
    ggml_tensor* v_cache  = ggml_new_tensor_3d(ctx, (ggml_type)a->kv_type, head_dim, kv_capacity, n_head_kv);

    // ---- hyper-connection mixer ----
    ggml_tensor* res3 = ggml_reshape_3d(ctx, res_in, n_embd, hc, T);
    ggml_tensor* xn = ggml_mul(ctx,
            ggml_reshape_2d(ctx, ggml_rms_norm(ctx, res3, eps), hc_dim, T), w_norm);
    ggml_tensor* lo = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, w_down, xn), 1.0f / (float)hc));
    ggml_tensor* gt = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));
    ggml_tensor* gated = ggml_reshape_3d(ctx, ggml_mul(ctx, xn, gt), n_embd, hc, T);
    ggml_tensor* mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, T,
            ggml_row_size(gated->type, n_embd) * hc, 0));
    for (int c = 1; c < hc; ++c)
        mixed = ggml_add(ctx, mixed, ggml_view_2d(ctx, gated, n_embd, T,
                ggml_row_size(gated->type, n_embd) * hc, ggml_row_size(gated->type, n_embd) * c));
    mixed = ggml_scale(ctx, mixed, 1.0f / (float)hc);
    ggml_tensor* inject = ggml_mul_mat(ctx, w_inject, xn);

    // ---- q | gate, interleaved per head ----
    ggml_tensor* qg = ggml_mul_mat(ctx, wq, mixed);                 // [q_dim*2, T]
    const std::size_t esz = ggml_element_size(qg);
    ggml_tensor* q = ggml_view_3d(ctx, qg, head_dim, n_head, T,
            esz * head_dim * 2, esz * head_dim * 2 * n_head, 0);
    ggml_tensor* gate = ggml_cont(ctx, ggml_view_3d(ctx, qg, head_dim, n_head, T,
            esz * head_dim * 2, esz * head_dim * 2 * n_head, esz * head_dim));

    q = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, q), eps), q_norm_w);
    ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wk, mixed), head_dim, n_head_kv, T);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), k_norm_w);
    ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wv, mixed), head_dim, n_head_kv, T);

    // Partial rotary over the first n_rot dims. IMRoPE reduces to NEOX when every
    // position component is equal, which it is for text.
    q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot, 2, 0, rope_base, rope_freq_scale,
                      0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, n_rot, 2, 0, rope_base, rope_freq_scale,
                      0.0f, 1.0f, 0.0f, 0.0f);

    // ---- append to the cache ----
    ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, T, kvH]
    ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor* k_full = ggml_view_3d(ctx, k_cache, head_dim, n_kv_pad, n_head_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
    ggml_tensor* v_full = ggml_view_3d(ctx, v_cache, head_dim, n_kv_pad, n_head_kv,
            v_cache->nb[1], v_cache->nb[2], 0);

    // The KV write goes into the graph FIRST - see the function comment.
    ggml_build_forward_expand(graph, ggml_set_rows(ctx, k_cache, k_write, kv_idx));
    ggml_build_forward_expand(graph, ggml_set_rows(ctx, v_cache, v_write, kv_idx));

    // ---- attention ----
    ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [hd, T, nH]
    ggml_tensor* attn = nullptr;
    if (use_flash)
    {
        // One fused kernel in place of mul_mat -> soft_max -> cont(permute(V)) ->
        // mul_mat -> cont(permute). It never materialises the [n_kv, T, n_head]
        // scores and never copies the whole V window. This is also what llama.cpp
        // runs here. Result lands as [hd, nH, T] - already the layout the gate and
        // the output projection want.
        attn = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, mask,
                                   attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    }
    else
    {
        ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q_attn);             // [n_kv, T, nH]
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask, attn_scale, 0.0f);
        ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
        attn = ggml_mul_mat(ctx, v_perm, probs);                             // [hd, T, nH]
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));          // [hd, nH, T]
        if (probe != nullptr)
        {
            ggml_set_output(scores); ggml_set_output(probs); ggml_set_output(attn);
            probe->push_back(scores); probe->push_back(probs); probe->push_back(attn);
        }
    }

    // qwen4exp gates the attention output before the output projection.
    attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
    ggml_tensor* proj = ggml_mul_mat(ctx, wo, ggml_reshape_2d(ctx, attn, q_dim, T));

    // ---- hyper-connection scatter ----
    ggml_tensor* wsc = ggml_reshape_3d(ctx, ggml_scale(ctx,
            ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float)hc)), 2.0f), 1, hc, T);
    ggml_tensor* bexp = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, proj, n_embd, 1, T),
            n_embd, hc, T, 1);
    ggml_tensor* res_out = ggml_reshape_2d(ctx,
            ggml_add(ctx, res3, ggml_mul(ctx, bexp, wsc)), hc_dim, T);

    bnd.add(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
    bnd.add(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
    bnd.add(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
    bnd.add(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
    bnd.add(wq, a->wq, (std::size_t)a->wq_bytes);
    bnd.add(wk, a->wk, (std::size_t)a->wk_bytes);
    bnd.add(wv, a->wv, (std::size_t)a->wv_bytes);
    bnd.add(wo, a->wo, (std::size_t)a->wo_bytes);
    bnd.add(q_norm_w, a->q_norm, (std::size_t)head_dim * sizeof(float));
    bnd.add(k_norm_w, a->k_norm, (std::size_t)head_dim * sizeof(float));
    // The caches are read AND written, so they need a device buffer that outlives
    // the graph rather than a weights binding.
    bnd.add(k_cache, a->k_cache, (std::size_t)a->kv_bytes, GGML_BACKEND_BUFFER_USAGE_ANY);
    bnd.add(v_cache, a->v_cache, (std::size_t)a->kv_bytes, GGML_BACKEND_BUFFER_USAGE_ANY);
    if (kv_out != nullptr) { kv_out->push_back(k_cache); kv_out->push_back(v_cache); }

    return res_out;
}

// ============================================================================
// Per-layer entry points (the fallback path).
// ============================================================================

// res_data is the 4-wide residual, [hc * n_embd, n_tokens] row-major on the
// host, read and written in place.
TSG_EXPORT int TSGgml_Qwen4ExpFfnBlock(
    const TSGgmlQwen4ExpFfnArgs* a,
    void* res_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh,
    float eps, int cache_slot, int res_resident)
{
    try
    {
        if (a == nullptr || res_data == nullptr)
        {
            set_last_error("qwen4exp FFN block: null args.");
            return 0;
        }
        if (!ensure_backend())
            return 0;

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);

        // Replay: same layer, same shape, same weights.
        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpMaxSlots)
            ? &g_q4e_ffn[cache_slot] : nullptr;
        if (slot != nullptr && slot->valid && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)a && slot->res_resident == res_resident
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend)))
        {
            if (!res_resident) ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            q4e_note(0, false);
            if (graph_compute_profiled(g_backend, slot->graph, kQwen4ExpFfnKernel) != GGML_STATUS_SUCCESS)
            {
                slot->reset();
                set_last_error("qwen4exp FFN block: replay failed.");
                return 0;
            }
            if (res_resident)
            {
                // Chain on the device: the next layer reads what this one wrote.
                ggml_backend_tensor_copy(slot->res_out, slot->res_in);
            }
            else
            {
                ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            }
            return 1;
        }
        if (slot != nullptr) slot->reset();

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr)
        {
            set_last_error("qwen4exp FFN block: ggml_init failed.");
            return 0;
        }

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);
        if (res_resident)
        {
            if (!q4e_res_ensure(res_bytes) ||
                ggml_backend_tensor_alloc(g_q4e_res_buf, res_in,
                        ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp FFN block: failed to bind the residual buffer.");
                return 0;
            }
        }

        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        ggml_tensor* res_out = q4e_nodes_ffn(ctx, binder, a, res_in,
                n_embd, hc, hc_low_rank, T, n_expert, n_expert_used, n_ff, n_ff_sh, eps);
        if (res_out == nullptr)
        {
            ggml_free(ctx);
            set_last_error("qwen4exp FFN block: failed to build the graph.");
            return 0;
        }
        ggml_set_output(res_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();
        ggml_build_forward_expand(graph, res_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp FFN block: failed to allocate graph tensors.");
            return 0;
        }

        binder.flush();
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);

        q4e_note(0, true);
        if (graph_compute_profiled(g_backend, graph, kQwen4ExpFfnKernel) != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp FFN block: graph compute failed.");
            return 0;
        }

        if (res_resident)
        {
            ggml_backend_tensor_copy(res_out, res_in);
        }
        else
        {
            ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);
        }

        if (slot != nullptr)
        {
            // Hand the graph to the cache. The weight bindings and the gallocr plan
            // stay valid as long as the descriptor and the shape do, which the
            // replay check above enforces.
            slot->ctx = ctx;
            slot->graph = graph;
            slot->alloc = alloc;
            slot->res_in = res_in;
            slot->res_out = res_out;
            slot->n_tokens = T;
            slot->hc_dim = hc_dim;
            slot->sig = (const void*)a;
            slot->res_resident = res_resident;
            slot->rebinds = std::move(binder.cached);
            slot->valid = true;
            return 1;
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    {
        set_last_error(std::string("qwen4exp FFN block: ") + e.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("qwen4exp FFN block: unknown error.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_Qwen4ExpGdnBlock(
    const TSGgmlQwen4ExpGdnArgs* a,
    void* res_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    float eps, int cache_slot, int res_resident)
{
    try
    {
        if (a == nullptr || res_data == nullptr) { set_last_error("qwen4exp GDN block: null args."); return 0; }
        if (!ensure_backend()) return 0;

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const int key_dim = head_k_dim * n_k_heads;
        const int value_dim = head_v_dim * n_v_heads;
        const int conv_dim = key_dim * 2 + value_dim;
        const int hist = d_conv - 1;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);

        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpMaxSlots)
            ? &g_q4e_gdn[cache_slot] : nullptr;
        if (slot != nullptr && slot->valid && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)a
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend)))
        {
            ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            q4e_note(1, false);
            if (graph_compute_profiled(g_backend, slot->graph, kQwen4ExpGdnKernel) != GGML_STATUS_SUCCESS)
            { slot->reset(); set_last_error("qwen4exp GDN block: replay failed."); return 0; }
            ggml_backend_synchronize(g_backend);
            if (slot->conv_out != nullptr)
            {
                ggml_backend_tensor_copy(slot->conv_out, slot->conv_in);
                ggml_backend_tensor_copy(slot->ssm_out, slot->ssm_in);
            }
            if (res_resident) ggml_backend_tensor_copy(slot->res_out, slot->res_in);
            else ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            return 1;
        }
        // Rebuild the graph; the state buffer below is untouched by that.
        if (slot != nullptr) slot->reset_graph();

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("qwen4exp GDN block: ggml_init failed."); return 0; }

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);
        if (res_resident)
        {
            if (!q4e_res_ensure(res_bytes) ||
                ggml_backend_tensor_alloc(g_q4e_res_buf, res_in,
                        ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp GDN block: failed to bind the residual buffer.");
                return 0;
            }
        }

        // The state is read through *_in and written through a SEPARATE *_out, copied
        // back device-to-device after the graph runs - the per-layer path's proven
        // dataflow, kept as is. (The token span writes the state in place instead,
        // with node order sequencing the write behind the read.)
        ggml_tensor* conv_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim, 1);
        ggml_tensor* ssm_state  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, n_v_heads);
        ggml_tensor* conv_state_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim, 1);
        ggml_tensor* ssm_state_out  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, n_v_heads);
        ggml_set_input(conv_state);      ggml_set_input(ssm_state);
        ggml_set_output(conv_state_out); ggml_set_output(ssm_state_out);

        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        Q4eGdnWriteback wb{};
        ggml_tensor* res_out = q4e_nodes_gdn(ctx, binder, a, res_in,
                conv_state, ssm_state,
                n_embd, hc, hc_low_rank, T,
                head_k_dim, head_v_dim, n_k_heads, n_v_heads, d_conv, eps, &wb);
        ggml_set_output(res_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();
        ggml_build_forward_expand(graph, res_out);
        // state write-back rides the same graph
        ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.tail, conv_state_out));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.new_state, ssm_state_out));

        // Give the two *_in state tensors their OWN device buffer so gallocr never
        // owns them and a graph rebuild cannot disturb them. Carrying the state across
        // a rebuild by copying was the alternative and it did not survive contact:
        // one fused layer was enough to derail the model.
        const std::size_t conv_bytes = (std::size_t)hist * conv_dim * sizeof(float);
        const std::size_t ssm_bytes = (std::size_t)head_v_dim * head_v_dim * n_v_heads * sizeof(float);
        const std::size_t ssm_off = (conv_bytes + 255) & ~(std::size_t)255;
        bool zero_state = true;
        if (slot != nullptr)
        {
            if (slot->state_buf == nullptr)
            {
                slot->state_buf = ggml_backend_buft_alloc_buffer(
                        ggml_backend_get_default_buffer_type(g_backend), ssm_off + ssm_bytes);
                if (slot->state_buf == nullptr)
                {
                    ggml_free(ctx);
                    set_last_error("qwen4exp GDN block: failed to allocate the state buffer.");
                    return 0;
                }
                slot->state_ready = false;
            }
            zero_state = !slot->state_ready;
            std::uint8_t* base = (std::uint8_t*)ggml_backend_buffer_get_base(slot->state_buf);
            if (ggml_backend_tensor_alloc(slot->state_buf, conv_state, base) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(slot->state_buf, ssm_state, base + ssm_off) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp GDN block: failed to bind the state buffer.");
                return 0;
            }
        }

        // Seed the state only when the buffer is new; a rebuild keeps what is there.
        if (zero_state)
        {
            binder.upload_list.push_back({conv_state, a->conv_state, conv_bytes});
            binder.upload_list.push_back({ssm_state, a->ssm_state, ssm_bytes});
        }

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp GDN block: failed to allocate graph tensors.");
            return 0;
        }

        binder.flush();
        if (slot != nullptr) slot->state_ready = true;
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);

        q4e_note(1, true);
        if (graph_compute_profiled(g_backend, graph, kQwen4ExpGdnKernel) != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(alloc); ggml_free(ctx);
            set_last_error("qwen4exp GDN block: graph compute failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_copy(conv_state_out, conv_state);
        ggml_backend_tensor_copy(ssm_state_out, ssm_state);
        if (res_resident) ggml_backend_tensor_copy(res_out, res_in);
        else ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);

        if (slot != nullptr)
        {
            slot->ctx = ctx; slot->graph = graph; slot->alloc = alloc;
            slot->res_in = res_in; slot->res_out = res_out;
            slot->conv_in = conv_state; slot->conv_out = conv_state_out;
            slot->ssm_in = ssm_state; slot->ssm_out = ssm_state_out;
            slot->n_tokens = T; slot->hc_dim = hc_dim; slot->sig = (const void*)a;
            slot->res_resident = res_resident;
            slot->rebinds = std::move(binder.cached);
            slot->valid = true;
            return 1;
        }
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    { set_last_error(std::string("qwen4exp GDN block: ") + e.what()); return 0; }
    catch (...)
    { set_last_error("qwen4exp GDN block: unknown error."); return 0; }
}

// mask_data is [n_kv_pad, T] F16, built host-side: 0 where token t may attend to
// cell j, -inf otherwise (the C# side pads with the same predicate and stride).
TSG_EXPORT int TSGgml_Qwen4ExpAttnBlock(
    const TSGgmlQwen4ExpAttnArgs* a,
    void* res_data,
    const void* mask_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv, int position,
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale,
    float eps, int cache_slot, int res_resident)
{
    try
    {
        if (a == nullptr || res_data == nullptr) { set_last_error("qwen4exp attn block: null args."); return 0; }
        if (!ensure_backend()) return 0;

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);
        const bool use_flash = q4e_flash_attn_ok(a->kv_type, head_dim);
        const int n_kv_pad = q4e_pad_kv(n_kv, kv_capacity, use_flash);
        const std::size_t mask_bytes = (std::size_t)n_kv_pad * T * sizeof(uint16_t);

        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpMaxSlots)
            ? &g_q4e_attn[cache_slot] : nullptr;

        // Keyed on the PADDED width, so the topology only moves once every stride
        // tokens instead of every token. Position and the KV write row reach the graph
        // as inputs, so their values change without the shape moving.
        if (slot != nullptr && slot->valid && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)a && slot->res_resident == res_resident
            && slot->n_kv == n_kv_pad
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend)))
        {
            if (!res_resident) ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            ggml_backend_tensor_set(slot->mask, mask_data, 0, mask_bytes);
            q4e_set_attn_indices(slot->pos, slot->kv_idx, T, position);
            q4e_note(2, false);
            if (graph_compute_profiled(g_backend, slot->graph, kQwen4ExpAttnKernel) != GGML_STATUS_SUCCESS)
            { slot->reset_graph(); set_last_error("qwen4exp attn block: replay failed."); return 0; }
            if (res_resident) ggml_backend_tensor_copy(slot->res_out, slot->res_in);
            else ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            return 1;
        }
        if (slot != nullptr) slot->reset_graph();

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("qwen4exp attn block: ggml_init failed."); return 0; }

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);
        if (res_resident)
        {
            if (!q4e_res_ensure(res_bytes) ||
                ggml_backend_tensor_alloc(g_q4e_res_buf, res_in,
                        ggml_backend_buffer_get_base(g_q4e_res_buf)) != GGML_STATUS_SUCCESS)
            {
                ggml_free(ctx);
                set_last_error("qwen4exp attn block: failed to bind the residual buffer.");
                return 0;
            }
        }

        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
        ggml_set_input(mask);
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(pos);
        // The rows this step writes, as an INPUT rather than a baked view offset.
        ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
        ggml_set_input(kv_idx);

        // The builder expands the KV write into the graph before we expand the
        // residual below, so the graph exists first.
        ggml_cgraph* graph = ggml_new_graph(ctx);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();

        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        std::vector<ggml_tensor*> kv_tensors;
        ggml_tensor* res_out = q4e_nodes_attn(ctx, graph, binder, a, res_in,
                mask, pos, kv_idx,
                n_embd, hc, hc_low_rank, T,
                head_dim, n_head, n_head_kv, kv_capacity, n_kv_pad,
                n_rot, rope_base, rope_freq_scale, attn_scale, eps, use_flash,
                &kv_tensors);
        ggml_set_output(res_out);
        ggml_build_forward_expand(graph, res_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp attn block: failed to allocate graph tensors.");
            return 0;
        }

        binder.flush();
        // Same pad-reads-uninitialised-memory hazard as the span; see the comment
        // there. Zero the device K/V at the start of the sequence.
        if (position == 0)
            for (ggml_tensor* t : kv_tensors)
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);
        ggml_backend_tensor_set(mask, mask_data, 0, mask_bytes);
        q4e_set_attn_indices(pos, kv_idx, T, position);

        q4e_note(2, true);
        if (graph_compute_profiled(g_backend, graph, kQwen4ExpAttnKernel) != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(alloc); ggml_free(ctx);
            set_last_error("qwen4exp attn block: graph compute failed.");
            return 0;
        }

        if (res_resident) ggml_backend_tensor_copy(res_out, res_in);
        else ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);

        if (slot != nullptr)
        {
            slot->ctx = ctx; slot->graph = graph; slot->alloc = alloc;
            slot->res_in = res_in; slot->res_out = res_out; slot->mask = mask;
            slot->pos = pos; slot->kv_idx = kv_idx;
            slot->n_tokens = T; slot->hc_dim = hc_dim; slot->sig = (const void*)a;
            slot->res_resident = res_resident; slot->n_kv = n_kv_pad;
            slot->rebinds = std::move(binder.cached);
            slot->valid = true;
            return 1;
        }
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return 1;
    }
    catch (const std::exception& e)
    { set_last_error(std::string("qwen4exp attn block: ") + e.what()); return 0; }
    catch (...)
    { set_last_error("qwen4exp attn block: unknown error."); return 0; }
}

// ============================================================================
// The token span: layers [layer_begin, layer_end) - both halves each - as ONE
// graph. With the PLE layer the only host interruption, a decode token is two
// of these calls instead of 96 per-layer ones: one residual upload, one graph
// launch, one download per span, and a single stable topology for ggml-cuda's
// CUDA graph capture instead of 96 alternating ones.
//
// ffn/gdn/attn are the BASE pointers of the full per-layer descriptor arrays
// (pinned on the C# side), indexed here by absolute layer id. kinds[il] != 0
// marks a recurrent (GDN) layer.
//
// GDN state lives in the per-layer slots' state buffers (g_q4e_gdn[il]), so
// the span and the per-layer fallback read and write the SAME state and a
// fallback mid-sequence stays coherent. The span writes the state in place -
// cpy(tail -> conv_state) expanded after the nodes that read conv_state, so
// node order sequences the write behind the read.
// ============================================================================
TSG_EXPORT int TSGgml_Qwen4ExpTokenSpan(
    const TSGgmlQwen4ExpFfnArgs* ffn,
    const TSGgmlQwen4ExpGdnArgs* gdn,
    const TSGgmlQwen4ExpAttnArgs* attn,
    const unsigned char* kinds,
    int layer_begin, int layer_end,
    void* res_data,
    const void* mask_data,
    int n_embd, int hc, int hc_low_rank, int n_tokens,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv, int position,
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh,
    float eps, int cache_slot, int first_ffn_only)
{
    try
    {
        if (ffn == nullptr || gdn == nullptr || attn == nullptr || kinds == nullptr
            || res_data == nullptr || layer_begin < 0 || layer_end <= layer_begin
            || layer_end > kQwen4ExpMaxSlots)
        {
            set_last_error("qwen4exp token span: bad args.");
            return 0;
        }
        if (!ensure_backend()) return 0;

        const int hc_dim = hc * n_embd;
        const int T = n_tokens;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);

        bool has_attn = false;
        for (int il = layer_begin; il < layer_end; ++il)
        {
            if (il == layer_begin && first_ffn_only != 0) continue;
            if (kinds[il] == 0) { has_attn = true; break; }
        }
        if (has_attn && mask_data == nullptr)
        {
            set_last_error("qwen4exp token span: an attention layer needs the mask.");
            return 0;
        }

        // One padded window for every attention layer in the span; they share one
        // mask, one position tensor and one write-row tensor.
        bool use_flash = false;
        int n_kv_pad = 0;
        if (has_attn)
        {
            int first_attn = (first_ffn_only != 0) ? layer_begin + 1 : layer_begin;
            while (kinds[first_attn] != 0) ++first_attn;
            use_flash = q4e_flash_attn_ok(attn[first_attn].kv_type, head_dim);
            n_kv_pad = q4e_pad_kv(n_kv, kv_capacity, use_flash);
        }
        const std::size_t mask_bytes = (std::size_t)n_kv_pad * T * sizeof(uint16_t);

        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpSpanSlots)
            ? &g_q4e_span[cache_slot] : nullptr;
        if (slot == nullptr)
        {
            set_last_error("qwen4exp token span: bad cache slot.");
            return 0;
        }

        // Replay: same span, same shape, same descriptors, same padded window - and
        // every cache-bound weight still where the graph believes it is.
        if (!q4e_span_force_rebuild()
            && slot->valid
            && q4e_refresh_bindings(slot, ggml_backend_get_device(g_backend))
            && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)ffn && slot->sig2 == (const void*)gdn
            && slot->sig3 == (const void*)attn
            && slot->layer_begin == layer_begin && slot->layer_end == layer_end
            && slot->kv_capacity == kv_capacity && slot->n_kv == n_kv_pad
            && slot->first_ffn_only == first_ffn_only)
        {
            ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            for (ggml_tensor* m : slot->span_masks)
                ggml_backend_tensor_set(m, mask_data, 0, mask_bytes);
            for (std::size_t i = 0; i < slot->span_pos.size(); ++i)
                q4e_set_attn_indices(slot->span_pos[i], slot->span_kvidx[i], T, position);
            q4e_note(3, false);
            if (q4e_span_trace() && !slot->span_copies.empty() && T == 1)
            {
                // What the graph will actually read as its recurrent state, sampled
                // immediately before the compute.
                std::vector<float> pb;
                fprintf(stderr, "[q4e-prestate] slot%d pos=%d:", cache_slot, position);
                for (std::size_t i = 0; i < slot->span_copies.size(); ++i)
                {
                    ggml_tensor* st = slot->span_copies[i].second;
                    pb.resize((std::size_t)ggml_nelements(st));
                    ggml_backend_tensor_get(st, pb.data(), 0, ggml_nbytes(st));
                    double n2 = 0.0;
                    for (float f : pb) n2 += (double)f * f;
                    fprintf(stderr, " %.9e", std::sqrt(n2));
                }
                fprintf(stderr, "%c", 10);
            }
            if (graph_compute_profiled(g_backend, slot->graph, kQwen4ExpSpanKernel) != GGML_STATUS_SUCCESS)
            {
                slot->reset_graph();
                set_last_error("qwen4exp token span: replay failed.");
                return 0;
            }
            // The synchronize is LOAD-BEARING: ggml_backend_tensor_copy issues a
            // legacy-stream memcpy, and ggml-cuda's compute stream is non-blocking,
            // so without the drain the copy can read conv_out/ssm_out while the
            // graph is still writing them. That race is timing-dependent - short
            // contexts kept the GPU caught up and hid it; long contexts queue
            // deeper and the copy wins, which corrupted the recurrent state from
            // the first replay after a long prefill.
            if (!slot->span_copies.empty())
                ggml_backend_synchronize(g_backend);
            for (const auto& c : slot->span_copies)
                ggml_backend_tensor_copy(c.first, c.second);
            ggml_backend_tensor_get(slot->res_out, res_data, 0, res_bytes);
            q4e_trace_probe(slot, "replay", position);
            q4e_trace_state(slot, "replay", position);
            return 1;
        }
        slot->reset_graph();

        const int n_layers = layer_end - layer_begin;
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * ((std::size_t)n_layers * 256 + 1024)
                    + ggml_graph_overhead_custom(kQwen4ExpSpanGraphSize, false);
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("qwen4exp token span: ggml_init failed."); return 0; }

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, T);
        ggml_set_input(res_in);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, kQwen4ExpSpanGraphSize, false);
        if (q4e_graph_uid_enabled()) graph->uid = q4e_next_graph_uid();

        Q4eBinder binder{ggml_backend_get_device(g_backend)};

        const std::size_t conv_dim = (std::size_t)(head_k_dim * n_k_heads) * 2
                                   + (std::size_t)(head_v_dim * n_v_heads);
        const std::size_t conv_bytes = (std::size_t)(d_conv - 1) * conv_dim * sizeof(float);
        const std::size_t ssm_bytes = (std::size_t)head_v_dim * head_v_dim * n_v_heads * sizeof(float);
        const std::size_t ssm_off = (conv_bytes + 255) & ~(std::size_t)255;

        std::vector<Qwen4ExpFfnCache*> seeded;
        std::vector<ggml_tensor*> kv_tensors;
        std::vector<ggml_tensor*> trace_res;
        std::vector<ggml_tensor*> probe_nodes;
        int attn_seen = 0;
        ggml_tensor* res = res_in;
        bool failed = false;
        const char* fail_what = nullptr;

        for (int il = layer_begin; il < layer_end && !failed; ++il)
        {
            if (il == layer_begin && first_ffn_only != 0)
            {
                // The attention half of this layer already ran per-layer; the span
                // picks up from its FFN half below.
            }
            else if (kinds[il] != 0)
            {
                // ---- recurrent half ----
                Qwen4ExpFfnCache* gslot = &g_q4e_gdn[il];
                if (gslot->state_buf == nullptr)
                {
                    gslot->state_buf = ggml_backend_buft_alloc_buffer(
                            ggml_backend_get_default_buffer_type(g_backend), ssm_off + ssm_bytes);
                    if (gslot->state_buf == nullptr)
                    { failed = true; fail_what = "state buffer alloc"; break; }
                    gslot->state_ready = false;
                }
                ggml_tensor* conv_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                        d_conv - 1, (int64_t)conv_dim, 1);
                ggml_tensor* ssm_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                        head_v_dim, head_v_dim, n_v_heads);
                ggml_set_input(conv_state);
                ggml_set_input(ssm_state);
                std::uint8_t* base = (std::uint8_t*)ggml_backend_buffer_get_base(gslot->state_buf);
                if (ggml_backend_tensor_alloc(gslot->state_buf, conv_state, base) != GGML_STATUS_SUCCESS ||
                    ggml_backend_tensor_alloc(gslot->state_buf, ssm_state, base + ssm_off) != GGML_STATUS_SUCCESS)
                { failed = true; fail_what = "state buffer bind"; break; }
                if (!gslot->state_ready)
                {
                    binder.upload_list.push_back({conv_state, gdn[il].conv_state, conv_bytes});
                    binder.upload_list.push_back({ssm_state, gdn[il].ssm_state, ssm_bytes});
                    seeded.push_back(gslot);
                }

                Q4eGdnWriteback wb{};
                res = q4e_nodes_gdn(ctx, binder, &gdn[il], res,
                        conv_state, ssm_state,
                        n_embd, hc, hc_low_rank, T,
                        head_k_dim, head_v_dim, n_k_heads, n_v_heads, d_conv, eps, &wb,
                        (q4e_span_trace() && il == layer_begin && cache_slot == 0 && T == 1)
                            ? &slot->gdn_probe : nullptr);
                // The residual first: its tree holds every node that READS the state
                // (the concat, the delta-net input). Then the write-back, so node
                // order puts the write strictly after the read.
                ggml_build_forward_expand(graph, res);
                if (q4e_span_state_in_graph())
                {
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.tail, conv_state));
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.new_state, ssm_state));
                }
                else
                {
                    ggml_tensor* conv_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                            d_conv - 1, (int64_t)conv_dim, 1);
                    ggml_tensor* ssm_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                            head_v_dim, head_v_dim, n_v_heads);
                    ggml_set_output(conv_out);
                    ggml_set_output(ssm_out);
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.tail, conv_out));
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, wb.new_state, ssm_out));
                    slot->span_copies.push_back({conv_out, conv_state});
                    slot->span_copies.push_back({ssm_out, ssm_state});
                }
            }
            else
            {
                // ---- attention half (expands its own KV write first) ----
                bool fa_here = use_flash && (attn_seen < q4e_span_fa_max());
                ++attn_seen;
                // Private inputs per attention layer, exactly as the per-layer kernel
                // has them.
                ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
                ggml_set_input(mask);
                ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
                ggml_set_input(pos);
                ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
                ggml_set_input(kv_idx);
                slot->span_masks.push_back(mask);
                slot->span_pos.push_back(pos);
                slot->span_kvidx.push_back(kv_idx);
                res = q4e_nodes_attn(ctx, graph, binder, &attn[il], res,
                        mask, pos, kv_idx,
                        n_embd, hc, hc_low_rank, T,
                        head_dim, n_head, n_head_kv, kv_capacity, n_kv_pad,
                        n_rot, rope_base, rope_freq_scale, attn_scale, eps, fa_here,
                        &kv_tensors,
                        (q4e_span_trace() && attn_seen == 1 && T == 1) ? &probe_nodes : nullptr);
                ggml_build_forward_expand(graph, res);
            }
            if (q4e_span_trace()) { ggml_set_output(res); trace_res.push_back(res); }

            // ---- FFN half ----
            res = q4e_nodes_ffn(ctx, binder, &ffn[il], res,
                    n_embd, hc, hc_low_rank, T,
                    n_expert, n_expert_used, n_ff, n_ff_sh, eps);
            ggml_build_forward_expand(graph, res);
            if (q4e_span_trace()) { ggml_set_output(res); trace_res.push_back(res); }
        }

        if (failed)
        {
            ggml_free(ctx);
            set_last_error(std::string("qwen4exp token span: ") + (fail_what ? fail_what : "build") + " failed.");
            return 0;
        }

        ggml_tensor* res_out = res;
        ggml_set_output(res_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp token span: failed to allocate graph tensors.");
            return 0;
        }

        binder.flush();
        for (Qwen4ExpFfnCache* g : seeded) g->state_ready = true;

        // A fresh sequence starts with a KV cache whose device copy holds whatever
        // was already in that memory - the host buffer it uploads from is a pooled
        // allocation with no zero guarantee. The window this graph reads is padded
        // past the rows any token has written, and a masked-off column contributes
        // nothing only if it is FINITE: an Inf in a never-written K row makes its
        // score Inf, Inf plus the -inf mask is NaN, and one NaN takes the whole
        // softmax row (flash attention alike). Zero the device copies at the start
        // of the sequence so the pad always reads zeros.
        if (position == 0)
            for (ggml_tensor* t : kv_tensors)
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));

        ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);
        for (ggml_tensor* m : slot->span_masks)
            ggml_backend_tensor_set(m, mask_data, 0, mask_bytes);
        for (std::size_t i = 0; i < slot->span_pos.size(); ++i)
            q4e_set_attn_indices(slot->span_pos[i], slot->span_kvidx[i], T, position);

        q4e_note(3, true);
        if (graph_compute_profiled(g_backend, graph, kQwen4ExpSpanKernel) != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(alloc); ggml_free(ctx);
            set_last_error("qwen4exp token span: graph compute failed.");
            return 0;
        }

        // See the replay path: the drain before the copies is load-bearing.
        if (!slot->span_copies.empty())
            ggml_backend_synchronize(g_backend);
        for (const auto& c : slot->span_copies)
            ggml_backend_tensor_copy(c.first, c.second);
        ggml_backend_tensor_get(res_out, res_data, 0, res_bytes);
        q4e_trace_probe(slot, "build", position);
        q4e_trace_state(slot, "build", position);

        if (!probe_nodes.empty())
        {
            static const char* names[] = { "scores", "probs", "attn" };
            for (std::size_t i = 0; i < probe_nodes.size() && i < 3; ++i)
            {
                ggml_tensor* t = probe_nodes[i];
                std::vector<float> pb((std::size_t)ggml_nelements(t));
                ggml_backend_tensor_get(t, pb.data(), 0, ggml_nbytes(t));
                double n2 = 0.0; float amax = 0.0f; std::size_t iamax = 0; bool bad = false;
                for (std::size_t j = 0; j < pb.size(); ++j)
                {
                    float f = pb[j];
                    n2 += (double)f * f;
                    if (std::fabs(f) > amax) { amax = std::fabs(f); iamax = j; }
                    if (!std::isfinite(f)) bad = true;
                }
                fprintf(stderr, "[q4e-probe] %s ne=[%d,%d,%d] l2=%.9e amax=%.6e@%zu head=%.6e %.6e %.6e%s%c",
                        names[i], (int)t->ne[0], (int)t->ne[1], (int)t->ne[2],
                        std::sqrt(n2), amax, iamax,
                        pb.size() > 0 ? pb[0] : 0.0f, pb.size() > 1 ? pb[1] : 0.0f,
                        pb.size() > 52 ? pb[52] : 0.0f, bad ? " NAN" : "", 10);
            }
        }
        if (q4e_span_trace() && has_attn && T == 1)
        {
            // Read back the K/V pad rows this graph could see. A masked-off pad
            // column only contributes nothing if its K row is exactly zero.
            std::vector<uint16_t> kv_buf;
            for (std::size_t t = 0; t < kv_tensors.size(); ++t)
            {
                ggml_tensor* kc = kv_tensors[t];
                const std::size_t total = ggml_nbytes(kc);
                kv_buf.resize(total / 2);
                ggml_backend_tensor_get(kc, kv_buf.data(), 0, total);
                // [head_dim, capacity, kvH]; pad rows are n_kv..n_kv_pad of dim 1.
                int max_bits = 0; long nonzero = 0;
                for (int64_t h = 0; h < kc->ne[2]; ++h)
                    for (int64_t r = n_kv; r < n_kv_pad; ++r)
                    {
                        const uint16_t* row = kv_buf.data()
                            + (h * kc->ne[1] + r) * kc->ne[0];
                        for (int64_t c = 0; c < kc->ne[0]; ++c)
                        {
                            int m = row[c] & 0x7FFF;   // f16 magnitude bits
                            if (m != 0) { ++nonzero; if (m > max_bits) max_bits = m; }
                        }
                    }
                if (nonzero > 0)
                    fprintf(stderr, "[q4e-kvpad] tensor %zu: %ld nonzero pad values, max f16 bits 0x%04x (rows %d..%d)%c",
                            t, nonzero, (unsigned)max_bits, n_kv, n_kv_pad, 10);
            }
            fprintf(stderr, "[q4e-kvpad] scan done (%zu tensors)%c", kv_tensors.size(), 10);
        }
        if (q4e_span_trace())
        {
            std::vector<float> buf((std::size_t)hc_dim * T);
            fprintf(stderr, "[q4e-trace] span %d..%d T=%d pos=%d:", layer_begin, layer_end, T, position);
            for (std::size_t i = 0; i < trace_res.size(); ++i)
            {
                ggml_backend_tensor_get(trace_res[i], buf.data(), 0, buf.size() * sizeof(float));
                double n2 = 0.0; bool bad = false;
                for (float f : buf) { n2 += (double)f * f; if (!std::isfinite(f)) bad = true; }
                fprintf(stderr, " %zu:%.4e%s", i, std::sqrt(n2), bad ? "!NAN" : "");
            }
            fprintf(stderr, "\n");
        }

        slot->ctx = ctx; slot->graph = graph; slot->alloc = alloc;
        slot->res_in = res_in; slot->res_out = res_out;
        slot->n_tokens = T; slot->hc_dim = hc_dim;
        slot->sig = (const void*)ffn; slot->sig2 = (const void*)gdn; slot->sig3 = (const void*)attn;
        slot->layer_begin = layer_begin; slot->layer_end = layer_end;
        slot->kv_capacity = kv_capacity; slot->n_kv = n_kv_pad;
        slot->first_ffn_only = first_ffn_only;
        slot->rebinds = std::move(binder.cached);
        slot->res_resident = 0;
        slot->valid = true;
        return 1;
    }
    catch (const std::exception& e)
    { set_last_error(std::string("qwen4exp token span: ") + e.what()); return 0; }
    catch (...)
    { set_last_error("qwen4exp token span: unknown error."); return 0; }
}

// Copy the residual to / from the device-resident buffer. The op-by-op attention
// half still works on the host, so it brackets itself with these.
TSG_EXPORT int TSGgml_Qwen4ExpResUpload(const void* data, long long bytes)
{
    try
    {
        if (!ensure_backend() || data == nullptr || bytes <= 0) return 0;
        if (!q4e_res_ensure((std::size_t)bytes)) return 0;
        ggml_backend_tensor_set(g_q4e_res, data, 0, (std::size_t)bytes);
        return 1;
    }
    catch (...) { set_last_error("qwen4exp residual upload failed."); return 0; }
}

TSG_EXPORT int TSGgml_Qwen4ExpResDownload(void* data, long long bytes)
{
    try
    {
        if (!ensure_backend() || data == nullptr || bytes <= 0) return 0;
        if (g_q4e_res == nullptr || g_q4e_res_capacity < (std::size_t)bytes) return 0;
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(g_q4e_res, data, 0, (std::size_t)bytes);
        return 1;
    }
    catch (...) { set_last_error("qwen4exp residual download failed."); return 0; }
}

TSG_EXPORT void TSGgml_Qwen4ExpResetFfnCache()
{
    for (int i = 0; i < kQwen4ExpMaxSlots; ++i)
    {
        g_q4e_ffn[i].reset();
        g_q4e_gdn[i].reset();
        g_q4e_attn[i].reset();
    }
    for (int i = 0; i < kQwen4ExpSpanSlots; ++i)
        g_q4e_span[i].reset();
    if (g_q4e_res_ctx) { ggml_free(g_q4e_res_ctx); g_q4e_res_ctx = nullptr; }
    if (g_q4e_res_buf) { ggml_backend_buffer_free(g_q4e_res_buf); g_q4e_res_buf = nullptr; }
    g_q4e_res = nullptr; g_q4e_res_capacity = 0;
}

} // extern "C"
