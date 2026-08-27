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
#include <cstdlib>   // ggml_cgraph::uid

#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

using namespace tsg;

// ============================================================================
// Qwen3.8-Flash-Next (qwen4exp) fused FFN half-layer.
//
// One GGML graph for: hyper-connection mixer -> 512-expert MoE (routed via
// mul_mat_id) + gated shared expert -> hyper-connection scatter back into the
// 4-wide residual.
//
// Why this exists. A qwen4exp decode step is dispatch bound, not arithmetic
// bound: the op-by-op path issues roughly 850 GGML submissions per token, each
// its own context, allocation and device synchronise, and the GPU idles between
// them. ggml_cpu and ggml_cuda measured within 4% of each other on this model,
// which is the signature of that bottleneck rather than of slow kernels. The
// FFN half alone was 8 of the ~18 submissions per layer (3 for the mixer, 5 for
// the experts and the shared expert); this collapses those 8 into 1, 48 times a
// token.
//
// The mixer and the scatter come along because they bracket the MoE and would
// otherwise force the residual back to the host on either side of it.
// ============================================================================
namespace
{
    constexpr const char* kQwen4ExpFfnKernel = "qwen4exp fused FFN block";
    constexpr int kQwen4ExpMaxSlots = 128;

    // One built graph per layer, kept across tokens.
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
        // Recurrent state, read through *_in and written through *_out; the caller
        // copies out -> in after every compute.
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
            mask = nullptr; pos = nullptr; kv_idx = nullptr; n_kv = -1;
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
    constexpr const char* kQwen4ExpGdnKernel = "qwen4exp fused GDN block";
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

    constexpr const char* kQwen4ExpAttnKernel = "qwen4exp fused attention block";
    Qwen4ExpFfnCache g_q4e_attn[kQwen4ExpMaxSlots];

    // Stamp every persisted graph with a stable non-zero id.
    //
    // ggml_new_graph leaves uid at 0, and ggml-cuda treats 0 as "unknown", so on every
    // replay it re-walks all ~30 nodes comparing a copy of each tensor struct and its
    // sources to decide whether the captured CUDA graph is still valid. With a stable
    // id it recognises the graph and skips that walk. At 96 replays a decode token,
    // that bookkeeping is not a rounding error.
    uint64_t q4e_next_graph_uid()
    {
        static uint64_t next = 1;
        return next++;
    }

    // The 4-wide residual, held on the DEVICE for the whole forward.
    //
    // Once the recurrence and the experts were fused, each kernel call was an upload
    // of the residual, a replay, a synchronise and a download - 96 of those per token,
    // ~0.125 ms each, and that round trip rather than the arithmetic was most of a
    // decode step. Keeping the residual resident turns a layer into a bare graph
    // replay; the host only sees it at the ends of the forward and around the layers
    // that are still op-by-op.
    ggml_backend_buffer_t g_q4e_res_buf = nullptr;
    std::size_t g_q4e_res_capacity = 0;
    ggml_context* g_q4e_res_ctx = nullptr;
    ggml_tensor* g_q4e_res = nullptr;

    // Ensure the shared residual tensor exists and is at least `bytes` big.
    bool q4e_res_ensure(std::size_t bytes)
    {
        if (g_q4e_res != nullptr && g_q4e_res_capacity >= bytes)
            return true;
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
            && slot->sig == (const void*)a && slot->res_resident == res_resident)
        {
            if (!res_resident) ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
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

        // ---- inputs / weights -------------------------------------------------
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
        ggml_set_output(res_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        graph->uid = q4e_next_graph_uid();
        ggml_build_forward_expand(graph, res_out);

        // ---- bind -------------------------------------------------------------
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind = [&](ggml_tensor* tgt, void* data, std::size_t bytes) {
            if (tgt == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload,
                                           GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
                {
                    if (needs_upload) upload_list.push_back({tgt, data, bytes});
                    return;
                }
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, true, buf))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        bind(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
        bind(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
        bind(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
        bind(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
        bind(w_router, a->router, (std::size_t)a->router_bytes);
        bind(w_gate_e, a->gate_exps, (std::size_t)a->gate_exps_bytes);
        bind(w_up_e, a->up_exps, (std::size_t)a->up_exps_bytes);
        bind(w_down_e, a->down_exps, (std::size_t)a->down_exps_bytes);
        bind(w_sh_gi, a->sh_gate_inp, (std::size_t)n_embd * sizeof(float));
        bind(w_sh_g, a->sh_gate, (std::size_t)a->sh_gate_bytes);
        bind(w_sh_u, a->sh_up, (std::size_t)a->sh_up_bytes);
        bind(w_sh_d, a->sh_down, (std::size_t)a->sh_down_bytes);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp FFN block: failed to allocate graph tensors.");
            return 0;
        }

        for (const HostBinding& hb : upload_list)
            ggml_backend_tensor_set(hb.tensor, hb.data, 0, hb.bytes);
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);

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
            && slot->sig == (const void*)a)
        {
            ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
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

        // ggml_ssm_conv asserts a 3-D input, so the history carries the sequence axis.
        //
        // The state is read through *_in and written through a SEPARATE *_out, copied
        // back device-to-device after the graph runs. Writing back into the tensor the
        // same graph reads did not take effect: a single call matched the reference to
        // 0.3%, but two consecutive calls diverged by 100% because the second one was
        // still starting from the initial state.
        ggml_tensor* conv_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim, 1);
        ggml_tensor* ssm_state  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, n_v_heads);
        ggml_tensor* conv_state_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hist, conv_dim, 1);
        ggml_tensor* ssm_state_out  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v_dim, head_v_dim, n_v_heads);
        ggml_set_input(conv_state);      ggml_set_input(ssm_state);
        ggml_set_output(conv_state_out); ggml_set_output(ssm_state_out);

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
        ggml_set_output(res_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        graph->uid = q4e_next_graph_uid();
        ggml_build_forward_expand(graph, res_out);
        // state write-back rides the same graph
        ggml_build_forward_expand(graph, ggml_cpy(ctx, tail, conv_state_out));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, new_state, ssm_state_out));

        // ---- bind ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;

        auto bind = [&](ggml_tensor* tgt, void* data, std::size_t bytes) {
            if (tgt == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload,
                                           GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
                { if (needs_upload) upload_list.push_back({tgt, data, bytes}); return; }
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, true, buf))
                { if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        bind(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
        bind(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
        bind(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
        bind(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
        bind(w_qkv, a->qkv, (std::size_t)a->qkv_bytes);
        bind(w_gate, a->gate, (std::size_t)a->gate_bytes);
        bind(w_beta, a->beta, (std::size_t)a->beta_bytes);
        bind(w_alpha, a->alpha, (std::size_t)a->alpha_bytes);
        bind(w_conv, a->conv1d, (std::size_t)d_conv * conv_dim * sizeof(float));
        bind(w_dt, a->ssm_dt, (std::size_t)n_v_heads * sizeof(float));
        bind(w_a, a->ssm_a, (std::size_t)n_v_heads * sizeof(float));
        bind(w_ssmnorm, a->ssm_norm, (std::size_t)head_v_dim * sizeof(float));
        bind(w_out, a->out_proj, (std::size_t)a->out_proj_bytes);
        // State is read AND written, so it must be a real device buffer this graph
        // owns rather than a weight binding.
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
            upload_list.push_back({conv_state, a->conv_state, conv_bytes});
            upload_list.push_back({ssm_state, a->ssm_state, ssm_bytes});
        }

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp GDN block: failed to allocate graph tensors.");
            return 0;
        }

        for (const HostBinding& hb : upload_list)
            ggml_backend_tensor_set(hb.tensor, hb.data, 0, hb.bytes);
        if (slot != nullptr) slot->state_ready = true;
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);

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

// mask_data is [n_kv, T] F16, built host-side: 0 where token t may attend to cell j,
// -inf otherwise. Passing it in keeps the causal/window policy on the host where it
// is cheap to express and keeps this graph shape-stable.
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
        const int q_dim = head_dim * n_head;
        const std::size_t res_bytes = (std::size_t)hc_dim * T * sizeof(float);
        const bool use_flash = q4e_flash_attn_ok(a->kv_type, head_dim);
        const int kv_stride = use_flash ? kQwen4ExpKvStride : 1;
        int n_kv_pad = ((n_kv + kv_stride - 1) / kv_stride) * kv_stride;
        if (n_kv_pad > kv_capacity) n_kv_pad = kv_capacity;
        const std::size_t mask_bytes = (std::size_t)n_kv_pad * T * sizeof(uint16_t);

        Qwen4ExpFfnCache* slot = (cache_slot >= 0 && cache_slot < kQwen4ExpMaxSlots)
            ? &g_q4e_attn[cache_slot] : nullptr;

        // Keyed on the PADDED width, so the topology only moves once every stride
        // tokens instead of every token. Position and the KV write row reach the graph
        // as inputs, so their values change without the shape moving. Without this the
        // 12 attention graphs were torn down and rebuilt - new context, new bindings,
        // fresh gallocr plan - 12 times a token, and a graph that gets rebuilt is
        // never a graph ggml-cuda has captured.
        if (slot != nullptr && slot->valid && slot->n_tokens == T && slot->hc_dim == hc_dim
            && slot->sig == (const void*)a && slot->res_resident == res_resident
            && slot->n_kv == n_kv_pad)
        {
            if (!res_resident) ggml_backend_tensor_set(slot->res_in, res_data, 0, res_bytes);
            ggml_backend_tensor_set(slot->mask, mask_data, 0, mask_bytes);
            q4e_set_attn_indices(slot->pos, slot->kv_idx, T, position);
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

        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
        ggml_set_input(mask);
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(pos);
        // The rows this step writes, as an INPUT rather than a baked view offset.
        ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
        ggml_set_input(kv_idx);

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

        // ---- attention ----
        ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [hd, T, nH]
        ggml_tensor* attn = nullptr;
        if (q4e_flash_attn_ok(a->kv_type, head_dim))
        {
            // One fused kernel in place of mul_mat -> soft_max -> cont(permute(V)) ->
            // mul_mat -> cont(permute). It never materialises the [n_kv, T, n_head]
            // scores and never copies the whole V window, which at batch 1 is most of
            // what this half was doing: that cont alone moved n_kv*head_dim*n_head_kv
            // elements per layer per token. This is also what llama.cpp runs here.
            // Result lands as [hd, nH, T] - already the layout the gate and the output
            // projection want, so the trailing permute goes too.
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
        ggml_set_output(res_out);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        graph->uid = q4e_next_graph_uid();
        // The KV write goes in FIRST. Nothing in res_out's tree depends on it - k_full
        // is a plain view of the cache and ggml does not treat view aliasing as an
        // edge - so node order is the only thing sequencing the write against the
        // read, and this token has to be able to attend to itself.
        ggml_build_forward_expand(graph, ggml_set_rows(ctx, k_cache, k_write, kv_idx));
        ggml_build_forward_expand(graph, ggml_set_rows(ctx, v_cache, v_write, kv_idx));
        ggml_build_forward_expand(graph, res_out);

        // ---- bind ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        auto bind = [&](ggml_tensor* tgt, void* data, std::size_t bytes,
                        enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, tgt, data, bytes, needs_upload, usage))
                { if (needs_upload) upload_list.push_back({tgt, data, bytes}); return; }
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, true, buf))
                { if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        bind(w_norm, a->hc_norm, (std::size_t)hc_dim * sizeof(float));
        bind(w_down, a->hc_down, (std::size_t)a->hc_down_bytes);
        bind(w_up, a->hc_up, (std::size_t)a->hc_up_bytes);
        bind(w_inject, a->hc_inject, (std::size_t)a->hc_inject_bytes);
        bind(wq, a->wq, (std::size_t)a->wq_bytes);
        bind(wk, a->wk, (std::size_t)a->wk_bytes);
        bind(wv, a->wv, (std::size_t)a->wv_bytes);
        bind(wo, a->wo, (std::size_t)a->wo_bytes);
        bind(q_norm_w, a->q_norm, (std::size_t)head_dim * sizeof(float));
        bind(k_norm_w, a->k_norm, (std::size_t)head_dim * sizeof(float));
        // The caches are read AND written, so they need a device buffer that outlives
        // the graph rather than a weights binding.
        bind(k_cache, a->k_cache, (std::size_t)a->kv_bytes, GGML_BACKEND_BUFFER_USAGE_ANY);
        bind(v_cache, a->v_cache, (std::size_t)a->kv_bytes, GGML_BACKEND_BUFFER_USAGE_ANY);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph))
        {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            set_last_error("qwen4exp attn block: failed to allocate graph tensors.");
            return 0;
        }

        for (const HostBinding& hb : upload_list)
            ggml_backend_tensor_set(hb.tensor, hb.data, 0, hb.bytes);
        if (!res_resident) ggml_backend_tensor_set(res_in, res_data, 0, res_bytes);
        ggml_backend_tensor_set(mask, mask_data, 0, mask_bytes);
        q4e_set_attn_indices(pos, kv_idx, T, position);

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
    if (g_q4e_res_ctx) { ggml_free(g_q4e_res_ctx); g_q4e_res_ctx = nullptr; }
    if (g_q4e_res_buf) { ggml_backend_buffer_free(g_q4e_res_buf); g_q4e_res_buf = nullptr; }
    g_q4e_res = nullptr; g_q4e_res_capacity = 0;
}

} // extern "C"
