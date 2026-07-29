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

// ============================================================================
// Qwen3.5 attention layer decode kernel (single token, single layer).
//
// Performs the full Qwen3.5 FullAttention block in a single GGML graph:
//   1. RMSNorm(hidden) * attn_norm_w
//   2. fused QKV matmul -> [Q_with_gate_interleaved (2*qDim), K (kvDim), V (kvDim)]
//   3. deinterleave Q and gate (each [num_heads, head_dim])
//   4. RMSNorm(Q) * q_norm_w  per head
//      RMSNorm(K) * k_norm_w  per head
//   5. RoPE on Q and K at `position`
//   6. append K, V into the persistent KV cache at `position`
//   7. flash attention against the populated KV cache window -> attn_out
//   8. attn_out *= sigmoid(gate)
//   9. residual += matmul(attn_out_flat, output_w)
//
// Replaces:
//   - 1 FusedRmsNormMatMulQuant call (norm + qkv)
//   - ~6 small CPU ops between (QK norm, RoPE, sigmoid gate, KV cache write)
//   - 1 FusedMatMulQuantAdd call (output + residual)
// with a single graph dispatch. Eliminates ~2 Metal command buffer dispatches
// + several CPU/GPU sync points per attention layer per decode token.
//
// All weights and the KV cache are bound zero-copy via host-pointer buffers
// when supported (Apple Silicon Metal, GGML CPU backend, integrated GPUs).
// ============================================================================
namespace
{
    int qwen35_attn_layer_decode_impl(
        float* residual_data, int hidden_size,
        float* attn_norm_data,
        void* qkv_data, int qkv_type,
        std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
        float* q_norm_data, float* k_norm_data, int head_dim,
        void* o_data, int o_type,
        std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
        void* k_cache_data, void* v_cache_data,
        int num_heads, int num_kv_heads,
        int max_seq_len, int position,
        float eps, float rope_base, float rope_freq_scale,
        int rope_mode,
        int kv_cache_type = GGML_TYPE_F32)
    {
        if (!ensure_backend())
            return 0;

        if (residual_data == nullptr || attn_norm_data == nullptr ||
            qkv_data == nullptr || q_norm_data == nullptr || k_norm_data == nullptr ||
            o_data == nullptr || k_cache_data == nullptr || v_cache_data == nullptr)
        {
            set_last_error("Null pointer passed to Qwen3.5 attention layer decode kernel.");
            return 0;
        }
        if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0 || position < 0)
        {
            set_last_error("Invalid dimensions passed to Qwen3.5 attention layer decode kernel.");
            return 0;
        }

        const int qDim = num_heads * head_dim;          // post-deinterleave Q dim
        const int qFullDim = qDim * 2;                  // pre-deinterleave Q+gate dim
        const int kDim = num_kv_heads * head_dim;
        const int totalSeqLen = position + 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int attnKvLen = flash_attn_kv_length(totalSeqLen, max_seq_len, head_dim);
        std::vector<ggml_fp16_t> attn_mask_data;

        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Qwen3.5 attention layer decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // Inputs / outputs
        ggml_tensor* residual_in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* attn_norm_w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* q_norm_w      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* k_norm_w      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* qkv_w         = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type), qkv_ne0, qkv_ne1);
        ggml_tensor* o_w           = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type), o_ne0, o_ne1);
        ggml_tensor* pos_tensor    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* k_cache_base  = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        ggml_tensor* v_cache_base  = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        ggml_tensor* residual_out  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* attn_mask = nullptr;
        if (flash_attn_requires_masked_padding(head_dim))
        {
            attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
            fill_flash_attn_mask(attn_mask_data, attnKvLen, totalSeqLen);
        }

        if (residual_in == nullptr || attn_norm_w == nullptr || q_norm_w == nullptr ||
            k_norm_w == nullptr || qkv_w == nullptr || o_w == nullptr || pos_tensor == nullptr ||
            k_cache_base == nullptr || v_cache_base == nullptr || residual_out == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for Qwen3.5 attention layer decode.");
            return 0;
        }

        // === Build computation graph ===

        // 1. Attention norm: RMSNorm + element-wise scale
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual_in, eps), attn_norm_w);

        // 2. Fused QKV projection: [hidden] -> [qFullDim + 2*kvDim]
        ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
        ggml_tensor* qkv_flat  = ggml_reshape_1d(
            ctx,
            ggml_mul_mat(ctx, qkv_w, normed_2d),
            qFullDim + 2 * kDim);

        // 3. Slice fused QKV into Q+gate, K, V
        //    The Q part has layout [head0_Q, head0_gate, head1_Q, head1_gate, ...] in memory:
        //    interpreted as a 3D tensor [head_dim, 2, num_heads] with row-major (C) layout
        //    where the innermost stride is sizeof(float).
        ggml_tensor* qg_part = ggml_view_1d(ctx, qkv_flat, qFullDim, 0);
        ggml_tensor* k_raw   = ggml_view_1d(ctx, qkv_flat, kDim,
            static_cast<std::size_t>(qFullDim) * sizeof(float));
        ggml_tensor* v_raw   = ggml_view_1d(ctx, qkv_flat, kDim,
            static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));

        ggml_tensor* qg_3d = ggml_reshape_3d(ctx, qg_part, head_dim, 2, num_heads);

        // Q view: [head_dim, num_heads] strided (skip the gate half)
        ggml_tensor* q_view = ggml_view_2d(
            ctx, qg_3d, head_dim, num_heads,
            qg_3d->nb[2], 0);
        ggml_tensor* gate_view = ggml_view_2d(
            ctx, qg_3d, head_dim, num_heads,
            qg_3d->nb[2], qg_3d->nb[1]);

        // We need contiguous Q for the per-head RMSNorm + RoPE that follow.
        ggml_tensor* q_2d_raw = ggml_cont(ctx, q_view);
        ggml_tensor* k_2d_raw = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);

        // 4. Per-head QK norm
        ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d_raw, eps), q_norm_w);
        ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d_raw, eps), k_norm_w);

        // 5. RoPE (NeoX style for Qwen3.5)
        ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
        ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);

        ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
        ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

        // 6. Append K, V into the persistent cache at `position`
        // q_rope: [head_dim, num_heads, 1] -> q_attn: [head_dim, 1, num_heads]
        ggml_tensor* q_attn       = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        ggml_tensor* k_rope_perm  = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
        ggml_tensor* v_3d         = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
        ggml_tensor* v_perm       = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
        ggml_tensor* k_write      = ggml_cont(ctx, k_rope_perm);
        ggml_tensor* v_write      = ggml_cont(ctx, v_perm);
        const std::size_t kv_byte_offset =
            static_cast<std::size_t>(position) * k_cache_base->nb[1];
        ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_base,
            head_dim, 1, num_kv_heads,
            k_cache_base->nb[1], k_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_base,
            head_dim, 1, num_kv_heads,
            v_cache_base->nb[1], v_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* k_cache_cpy = ggml_cpy(ctx, k_write, k_dst);
        ggml_tensor* v_cache_cpy = ggml_cpy(ctx, v_write, v_dst);

        ggml_tensor* k_full = view_kv_cache_window(ctx, k_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        ggml_tensor* v_full = view_kv_cache_window(ctx, v_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        if (k_full == nullptr || v_full == nullptr)
        {
            set_last_error("Failed to create KV cache views for Qwen3.5 attention layer decode.");
            return 0;
        }

        // 7. Flash attention (handles GQA broadcasting)
        ggml_tensor* attn_out_4d = ggml_flash_attn_ext(ctx,
            q_attn, k_full, v_full, attn_mask, scale, 0.0f, 0.0f);

        // attn_out_4d: [head_dim, num_heads, 1] -> reshape to [head_dim, num_heads]
        ggml_tensor* attn_out_2d = ggml_reshape_2d(ctx, attn_out_4d, head_dim, num_heads);

        // 8. Sigmoid-gated mix: attn_out *= sigmoid(gate)
        // gate_view is the strided view into the QKV output; need it contiguous for elementwise mul.
        ggml_tensor* gate_2d = ggml_cont(ctx, gate_view);
        ggml_tensor* gate_sig = ggml_sigmoid(ctx, gate_2d);
        ggml_tensor* attn_gated = ggml_mul(ctx, attn_out_2d, gate_sig);

        // 9. Output projection + residual: residual += matmul(attn_gated_flat, o_w)
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_gated, qDim, 1);
        ggml_tensor* o_flat    = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, o_w, attn_flat), hidden_size);
        ggml_tensor* result    = ggml_add(ctx, residual_in, o_flat);

        ggml_tensor* out_residual = ggml_cpy(ctx, result, residual_out);
        ggml_set_output(out_residual);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, k_cache_cpy);
        ggml_build_forward_expand(graph, v_cache_cpy);
        ggml_build_forward_expand(graph, out_residual);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr)
                return;

            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (st == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                            upload_list.push_back({t, data, bytes});
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
                    if (!cacheable)
                        ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        bind_or_mark(qkv_w,        qkv_data,        static_cast<std::size_t>(qkv_bytes), true);
        bind_or_mark(o_w,          o_data,          static_cast<std::size_t>(o_bytes),   true);
        bind_or_mark(attn_norm_w,  attn_norm_data,  static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        bind_or_mark(q_norm_w,     q_norm_data,     static_cast<std::size_t>(head_dim)    * sizeof(float), true);
        bind_or_mark(k_norm_w,     k_norm_data,     static_cast<std::size_t>(head_dim)    * sizeof(float), true);
        bind_or_mark(k_cache_base, k_cache_data,    kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind_or_mark(v_cache_base, v_cache_data,    kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (attn_mask != nullptr && !attn_mask_data.empty())
            bind_or_mark(attn_mask, attn_mask_data.data(), attn_mask_data.size() * sizeof(ggml_fp16_t), false);

        // Bind the input residual buffer directly so that the output write goes
        // back into the caller's memory without an explicit download. Falls back
        // to upload+download when the host pointer is not cacheable.
        ggml_backend_buffer_t res_in_buf = nullptr;
        bool residual_zero_copy = try_get_host_ptr_buffer(g_backend, dev, residual_data,
            static_cast<std::size_t>(hidden_size) * sizeof(float), false, res_in_buf);
        if (residual_zero_copy)
        {
            ephemeral_bufs.emplace_back(res_in_buf);
            ggml_status st = ggml_backend_tensor_alloc(res_in_buf, residual_in, residual_data);
            if (st != GGML_STATUS_SUCCESS)
                residual_zero_copy = false;
        }

        ggml_backend_buffer_t res_out_buf = nullptr;
        bool residual_out_zero_copy = try_get_host_ptr_buffer(g_backend, dev, residual_data,
            static_cast<std::size_t>(hidden_size) * sizeof(float), false, res_out_buf);
        if (residual_out_zero_copy)
        {
            ephemeral_bufs.emplace_back(res_out_buf);
            ggml_status st = ggml_backend_tensor_alloc(res_out_buf, residual_out, residual_data);
            if (st != GGML_STATUS_SUCCESS)
                residual_out_zero_copy = false;
        }

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for Qwen3.5 attention layer decode.");
            return 0;
        }

        // Drain pending async work before CPU memcpys from C# tensor buffers.
        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        if (!residual_zero_copy)
            ggml_backend_tensor_set(residual_in, residual_data,
                0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Qwen3.5 attention layer decode.");
            return 0;
        }

        finalize_compute(residual_out_zero_copy, residual_out, residual_data,
            static_cast<std::size_t>(hidden_size) * sizeof(float));

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_Qwen35AttentionLayerDecode(
    float* residual_data, int hidden_size,
    float* attn_norm_data,
    void* qkv_data, int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
    float* q_norm_data, float* k_norm_data, int head_dim,
    void* o_data, int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
    void* k_cache_data, void* v_cache_data,
    int num_heads, int num_kv_heads,
    int max_seq_len, int position,
    float eps, float rope_base, float rope_freq_scale,
    int rope_mode,
    int kv_cache_type)
{
    try
    {
        return qwen35_attn_layer_decode_impl(
            residual_data, hidden_size,
            attn_norm_data,
            qkv_data, qkv_type, qkv_ne0, qkv_ne1, qkv_bytes,
            q_norm_data, k_norm_data, head_dim,
            o_data, o_type, o_ne0, o_ne1, o_bytes,
            k_cache_data, v_cache_data,
            num_heads, num_kv_heads,
            max_seq_len, position,
            eps, rope_base, rope_freq_scale, rope_mode,
            kv_cache_type);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in Qwen3.5 attention layer decode.");
        return 0;
    }
}

// ============================================================================
// Qwen3.5/3.6 FULL-MODEL decode: the whole hybrid transformer (full-attention +
// GatedDeltaNet recurrent layers, with a per-layer dense or MoE FFN) executed as
// ONE GGML graph per decode token. This collapses the ~120-400 per-op kernel
// dispatches/token (each a WDDM submit + host sync, the dominant decode cost on
// this architecture) down to a single graph_compute, mirroring llama.cpp's
// single-graph decode (src/models/qwen35moe.cpp / delta-net-base.cpp).
//
// Attention layers: built exactly like TSGgml_Qwen35AttentionLayerDecode (fused
// QKV with interleaved Q+gate, per-head q/k RMSNorm, NeoX RoPE, device-resident
// circular KV cache append, flash attention, sigmoid-gated output, o-proj).
// Recurrent (GDN) layers: built like llama.cpp's build_layer_attn_linear +
// build_delta_net_fused — qkv/z/beta/alpha projections, ssm_conv over a host
// conv-state ring, SiLU, q/k L2-norm + head tiling, the fused ggml_gated_delta_net
// op (K=1), gated RMSNorm with z, and the ssm output projection.
//
// GDN recurrent state (conv ring + delta state) is passed per-token via host
// in/out pointers (no device residency / latch), which keeps this path trivially
// correct alongside the per-op fallback and MTP (the host buffers are always the
// single source of truth). Attention KV cache stays device-resident (cacheable
// bind, in-place append) exactly like the per-op attention kernel.
//
// Output is the final pre-output-norm hidden state [hidden_size]; the C# caller
// owns output_norm + the LM head. Returns 0 on anything it cannot handle so the
// caller falls back to the per-op decode.
// ============================================================================
namespace
{
    // Persistent decode-graph cache (single entry). When TS_QWEN35_FD_PERSIST is
    // set, the whole-model decode graph is built ONCE with stable tensor addresses
    // (raw ggml ctx + ggml_backend_alloc_ctx_tensors) and reused across tokens, so
    // ggml-cuda's CUDA-graph capture engages (key = cgraph->nodes[0]; replays one
    // captured graph instead of re-launching ~2000 kernels/token, which on WDDM the
    // GPU is starved waiting for — measured ~35% util / 21 W without it). The KV
    // write uses ggml_set_rows (position is an I64 input, not a baked view offset)
    // and attention uses a padded window + F16 mask input, so the graph topology is
    // identical token-to-token within a 256-token stride. Dropped + rebuilt when the
    // window grows a stride or the GDN state re-seeds (TSGgml_Qwen35ResetDecodeCache).
    struct Q35DecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* kv_index = nullptr;   // I64 [1] = write position (shared, all attn layers)
        ggml_tensor* attn_mask = nullptr;  // F16 [window] causal padding mask (shared)
        const void* sig_disc = nullptr;    // model-instance discriminator
        const void* sig_kcache0 = nullptr; // first attention layer's KV ptr (per-holder identity)
        int num_layers = 0;
        int hidden_size = 0;
        int window = 0;
        bool folded = false;               // hidden_out holds logits (final norm + lm_head folded in)
        int out_count = 0;                 // element count of hidden_out (vocab when folded, else hidden)
        // Tensor-parallel plan over this cache's graph: one entry per rank-local
        // build (each rank keeps its own pool slot, keyed by its KV pointer).
        tsg::TpRankPlan tp_plan;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_t = hidden_out = pos_tensor = kv_index = attn_mask = nullptr;
            sig_disc = sig_kcache0 = nullptr; num_layers = hidden_size = window = 0;
            folded = false; out_count = 0;
            tp_plan.clear();
        }
    };

    // Concurrent (N>=2) Qwen3.5 requests each decode through their OWN per-request
    // KV + GDN state holder (Qwen35Model.BindSequenceCache swaps _kvCacheK /
    // _deltaStateTensor / _fdConvScratch), so a single shared decode-graph entry —
    // whose captured graph bakes those device addresses — would be busted on EVERY
    // request switch and rebuilt from scratch, collapsing aggregate throughput AND
    // (worse) replaying the previous request's baked addresses if the cheap reuse
    // key didn't include the holder identity (wrong output). Keep a small pool keyed
    // by (sig_disc, sig_kcache0 = first attention layer's KV ptr) so each in-flight
    // request retains its own persistent, CUDA-graph-captured decode graph. Exact
    // analogue of the dense Gemma4 g_g4dc_pool (see its comment). ResetDecodeCache
    // drops them all.
    constexpr int kQ35MaxDecodeCaches = 8;
    struct Q35DecodeCachePool
    {
        Q35DecodeCache entries[kQ35MaxDecodeCaches];
        std::uint64_t used[kQ35MaxDecodeCaches] = {};   // LRU clock per slot
        std::uint64_t clock = 0;

        Q35DecodeCache* find(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        Q35DecodeCache& claim(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kQ35MaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void drop(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                    entries[i].reset();
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    Q35DecodeCachePool g_q35dc_pool;

    int qwen35_model_decode_impl(
        const TSGgmlQwen35LayerDesc* layers, int num_layers,
        void* hidden_data, int hidden_size, int position,
        int num_heads, int num_kv_heads, int head_dim, int cache_size,
        int rope_n_dims, int rope_mode, int kv_cache_type,
        int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
        float eps, float rope_base, float rope_freq_scale,
        int num_experts, int num_experts_used, int expert_ff, int shared_ff,
        int norm_topk, float expert_weights_scale,
        void* logits_data, int vocab_size,
        const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
        const void* final_norm_data,
        int tp_degree = 1, void** tp_plan_out = nullptr)
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("Qwen3.5 model decode: invalid arguments.");
            return 0;
        }
        // Tensor parallelism. With tp_degree > 1 the caller drives one rank at a
        // time (SetActiveDevice): this builds the active rank's graph and hands
        // back a plan (via tp_plan_out) instead of running it; the driver then
        // executes every rank's plan segment-by-segment, reducing the partials
        // at the row-parallel cut points. Per-layer dims (heads, GDN heads,
        // stacked experts) arrive already sharded for this rank; the MoE router
        // stays global — every rank computes the same top-k from the replicated
        // hidden state, and a per-rank id LUT + weight mask (see below) confine
        // its rank's mul_mat_id to the experts it owns.
        const bool tp_mode = tp_degree > 1 && tp_plan_out != nullptr;
        if (tp_mode)
            *tp_plan_out = nullptr;
        const int tp_rank = tp_mode ? g_active_rank : 0;
        const int stacked_experts = tp_mode && num_experts > 0 ? num_experts / tp_degree : num_experts;
        if (tp_mode && num_experts > 0 &&
            (num_experts % tp_degree != 0 || stacked_experts < num_experts_used))
        {
            set_last_error("Qwen3.5 model decode: expert count is not shardable across the tensor-parallel ranks.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 model decode: descriptor size mismatch (C# " +
                std::to_string(layers[0].struct_bytes) + " vs native " +
                std::to_string(sizeof(TSGgmlQwen35LayerDesc)) + ").");
            return 0;
        }

        const int H = hidden_size;
        const int totalSeqLen = position + 1;
        const int qDim = num_heads * head_dim;
        const int qFullDim = qDim * 2;            // Q + gate interleaved per head
        const int kDim = num_kv_heads * head_dim;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;

        // Persistent decode graph: default ON; TS_QWEN35_FD_PERSIST=0 disables.
        // Persist mode uses ggml_set_rows (KV write) + a fixed-topology graph that is
        // built once and REPLAYED each token (upload 4 dynamic inputs + graph_compute,
        // no per-token graph rebuild / backend-buffer alloc+free / weight re-upload).
        //   - CUDA: the static graph additionally lets ggml-cuda capture+replay a CUDA
        //     graph (cuts per-node launch latency).
        //   - Vulkan: no graph capture, but the replay still skips the ~120ms/token
        //     non-persist rebuild churn (fresh vkAllocateMemory + re-record of 4200+
        //     nodes + 176 norm-weight re-uploads every token). set_rows has a Vulkan
        //     kernel in ggml v0.15.3 (unlike Metal, where ggml_metal_op_set_rows
        //     SEGFAULTs — Metal stays on the non-persist cpy path).
        // Persist mode pads the attention window to a fixed stride so the graph is
        // identical token-to-token (CUDA-graph capture); the F16 mask zeroes valid
        // positions and -inf's the padding. Non-persist keeps the exact window.
        static const bool persist_cfg = []{ const char* e = std::getenv("TS_QWEN35_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        const bool persist = persist_cfg &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN);
        constexpr int kPersistKvStride = 256;
        const int attnKvLen = persist
            ? std::min(cache_size, ((totalSeqLen + kPersistKvStride - 1) / kPersistKvStride) * kPersistKvStride)
            : flash_attn_kv_length(totalSeqLen, cache_size, head_dim);
        const bool use_persist_mask = persist;
        // VULKAN CORRECTNESS: the persist path pads the flash-attn KV window to the
        // 256-stride so the graph topology stays constant for replay. On ggml-vulkan a
        // padded window (KV a multiple of the flash block width) selects the "aligned"
        // flash-attn shader, which computes INCORRECT attention for this model's
        // head_dim=256 GQA over the masked/padded window — output stays coherent for
        // ~10-20 tokens then degenerates into a repetition loop (proven by A/B: forcing
        // the padded window + F16 mask into the otherwise-correct non-persist path
        // reproduces it, and it survives zeroing the padded KV rows, so it is the
        // aligned-shader path itself, not stale KV). CUDA's flash handles the padded
        // window correctly. FIX: on Vulkan persist, compute attention WITHOUT flash —
        // explicit mul_mat + soft_max_ext(mask) + mul_mat (see the attention block).
        // That applies the -inf mask through soft_max (so padded positions contribute
        // nothing) using only core, validated Vulkan ops, and keeps the stable padded
        // topology persist needs — restoring the persist perf win (~16 vs ~5.7 tok/s)
        // with correct output. Non-persist and CUDA keep flash. TS_QWEN35_VULKAN_FLASH=1
        // forces the (incorrect) flash path on Vulkan persist for A/B debugging only.
        static const bool vulkan_flash_forced = []{ const char* e = std::getenv("TS_QWEN35_VULKAN_FLASH"); return e != nullptr && e[0] == '1'; }();
        const bool use_non_flash_attn = persist && g_backend_type == BACKEND_TYPE_VULKAN && !vulkan_flash_forced;
        const void* sig_disc = layers[0].attn_norm_w;
        // Per-holder identity: first attention layer's KV cache device ptr. With
        // per-request fused-decode holders (Qwen35Model.BindSequenceCache) this is
        // distinct per concurrent request, so each retains its own captured graph
        // in g_q35dc_pool instead of busting/rebuilding (or, worse, replaying the
        // other request's baked addresses) on every switch.
        const void* sig_kcache0 = nullptr;
        for (int l = 0; l < num_layers; l++)
            if (!layers[l].is_recurrent && layers[l].k_cache != nullptr) { sig_kcache0 = layers[l].k_cache; break; }
        // Fold final-norm + lm_head into the graph so the whole token (incl. the
        // 248K-vocab logits) is one captured replay -> no separate lm_head submit.
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;

        // The tensor-parallel driver executes the graph *after* this call
        // returns, so the graph, its context and its backend buffer must
        // outlive the call — which is exactly what persist mode provides.
        if (tp_mode && !persist)
        {
            set_last_error("Qwen3.5 model decode: tensor-parallel mode requires the persistent decode graph.");
            return 0;
        }

        // ===== Persist reuse fast-path: replay THIS request's captured graph =====
        // find() already matched (sig_disc, sig_kcache0); check the finer shape here.
        Q35DecodeCache* dc = persist ? g_q35dc_pool.find(sig_disc, sig_kcache0) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->window == attnKvLen)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * sizeof(float));
            std::int32_t pos_val = position;
            ggml_backend_tensor_set(dc->pos_tensor, &pos_val, 0, sizeof(std::int32_t));
            std::int64_t kv_idx = position;
            ggml_backend_tensor_set(dc->kv_index, &kv_idx, 0, sizeof(std::int64_t));
            std::vector<ggml_fp16_t> mask_data;
            fill_flash_attn_mask(mask_data, attnKvLen, totalSeqLen);
            ggml_backend_tensor_set(dc->attn_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
            if (tp_mode)
            {
                // Inputs are staged; the driver runs the segments across ranks.
                if (!dc->tp_plan.valid())
                {
                    set_last_error("Qwen3.5 model decode: cached graph has no tensor-parallel plan.");
                    dc->reset();
                    return 0;
                }
                dc->tp_plan.out_tensor = dc->hidden_out;
                dc->tp_plan.out_host = dc->folded ? logits_data : hidden_data;
                dc->tp_plan.out_bytes = static_cast<std::size_t>(dc->out_count) * sizeof(float);
                *tp_plan_out = &dc->tp_plan;
                clear_last_error();
                return 1;
            }
            ggml_status st = ggml_backend_graph_compute(g_backend, dc->graph);
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Qwen3.5 model decode: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            void* out_data = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, out_data, static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            return 1;
        }
        // Miss -> (re)build into this request's slot (reset in place / evict LRU).
        Q35DecodeCache* dcb = persist ? &g_q35dc_pool.claim(sig_disc, sig_kcache0) : nullptr;

        // no_alloc ctx: tensor metadata only. Non-persist uses the pooled 32 MB
        // block; persist uses a raw ctx kept alive in g_q35dc for graph reuse.
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr)
            {
                set_last_error("Qwen3.5 model decode: failed to init persist ggml context.");
                return 0;
            }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Qwen3.5 model decode: failed to acquire ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        ggml_tensor* hidden_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* lm_head_t = fold ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1) : nullptr;
        ggml_tensor* final_norm_t = fold ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
        // Shared per-token inputs for the static (capturable) graph.
        ggml_tensor* shared_kv_index = use_persist_mask ? ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1) : nullptr;
        ggml_tensor* shared_attn_mask = use_persist_mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1) : nullptr;
        if (use_persist_mask)
        {
            ggml_set_input(hidden_t);
            ggml_set_input(pos_tensor);
            ggml_set_input(shared_kv_index);
            ggml_set_input(shared_attn_mask);
        }

        struct LayerTensors {
            // attention
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;
            ggml_tensor* v_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* k_cache_base;
            ggml_tensor* v_cache_base;
            ggml_tensor* attn_mask;
            ggml_tensor* k_cpy;
            ggml_tensor* v_cpy;
            std::vector<ggml_fp16_t> attn_mask_data;
            // gdn
            ggml_tensor* gdn_qkv_w;
            ggml_tensor* gdn_gate_w;
            ggml_tensor* ssm_beta_w;
            ggml_tensor* ssm_alpha_w;
            ggml_tensor* conv1d_w;
            ggml_tensor* ssm_dt_w;
            ggml_tensor* ssm_a_w;
            ggml_tensor* ssm_norm_w;
            ggml_tensor* ssm_out_w;
            ggml_tensor* conv_state_in;
            ggml_tensor* delta_state_in;
            ggml_tensor* conv_state_out;
            ggml_tensor* delta_state_out;
            // ffn (dense)
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            // ffn (MoE)
            ggml_tensor* gate_inp_w;
            ggml_tensor* gate_exps;
            ggml_tensor* up_exps;
            ggml_tensor* down_exps;
            ggml_tensor* shexp_gate_w;
            ggml_tensor* shexp_up_w;
            ggml_tensor* shexp_down_w;
            ggml_tensor* shexp_gate_inp_w;
        };
        std::vector<LayerTensors> lt(num_layers);

        // --- create per-layer tensors ---
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.is_recurrent == 0)
            {
                t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                if (d.separate_qkv != 0)
                {
                    t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                    t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                }
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                t.k_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, cache_size, num_kv_heads);
                t.v_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, cache_size, num_kv_heads);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                // Packed in-projection (the TP shards): one [hidden, Q|K|V|Z|beta|alpha]
                // weight instead of four separate ones; gdn_gate_w == null marks
                // it, and the z/beta/alpha weights are neither created nor bound.
                if (d.gdn_gate_w != nullptr)
                {
                    t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                    t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                    t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                }
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                t.conv_state_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                t.delta_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
            }
            // FFN
            if (d.is_moe == 0)
            {
                t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            }
            else
            {
                t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                // Under TP the stacked expert tensors hold only this rank's
                // whole-expert slice; the router dims stay global.
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), hidden_size, expert_ff, stacked_experts);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), hidden_size, expert_ff, stacked_experts);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, hidden_size, stacked_experts);
                t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // Expert-parallel routing constants (TP MoE only). Whole experts
        // partition across ranks, but the top-k runs on the GLOBAL router
        // probabilities (identical on every rank, since the hidden stream is
        // replicated bit-for-bit after each reduction), so the selected ids
        // must be confined to this rank's slice without integer arithmetic —
        // which ggml lacks on I32 — before feeding mul_mat_id:
        //   ep_lut  I32 [1, num_experts]: global id -> local id (foreign -> 0)
        //   ep_mask F32 [1, num_experts]: 1 for owned experts, else 0
        // get_rows over the selected ids yields the local ids and a weight
        // mask; a zero weight nullifies the (locally computed, wrong-expert)
        // contribution of every foreign route. Both are uploaded once at graph
        // build and live in the persist buffer.
        ggml_tensor* ep_lut = nullptr;
        ggml_tensor* ep_mask = nullptr;
        std::vector<std::int32_t> ep_lut_data;
        std::vector<float> ep_mask_data;
        if (tp_mode && num_experts > 0)
        {
            bool any_moe = false;
            for (int l = 0; l < num_layers; l++)
                if (layers[l].is_moe != 0) { any_moe = true; break; }
            if (any_moe)
            {
                ep_lut = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, num_experts);
                ep_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, num_experts);
                const int first = tp_rank * stacked_experts;
                const int last = first + stacked_experts;
                ep_lut_data.resize(static_cast<std::size_t>(num_experts));
                ep_mask_data.resize(static_cast<std::size_t>(num_experts));
                for (int e = 0; e < num_experts; e++)
                {
                    const bool own = e >= first && e < last;
                    ep_lut_data[static_cast<std::size_t>(e)] = own ? e - first : 0;
                    ep_mask_data[static_cast<std::size_t>(e)] = own ? 1.0f : 0.0f;
                }
            }
        }

        // Tensor-parallel cut points. `tp_partial` is the row-parallel matmul
        // whose per-rank outputs the collective reduces; `tp_boundary` is the
        // last graph node that may run before that reduction.
        std::vector<ggml_tensor*> tp_partial;
        std::vector<ggml_tensor*> tp_boundary;
        if (tp_mode)
        {
            tp_partial.reserve(static_cast<std::size_t>(num_layers) * 2);
            tp_boundary.reserve(static_cast<std::size_t>(num_layers) * 2);
        }

        // --- build the chained graph ---
        ggml_tensor* hidden = hidden_t;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, H, 1);
            ggml_tensor* block_out; // the attention / gdn output added to residual

            if (d.is_recurrent == 0)
            {
                // ===== Full attention =====
                ggml_tensor* qg_part;
                ggml_tensor* k_raw;
                ggml_tensor* v_raw;
                if (d.separate_qkv != 0)
                {
                    qg_part = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qFullDim);
                    k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.k_w, normed_2d), kDim);
                    v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.v_w, normed_2d), kDim);
                }
                else
                {
                    ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qFullDim + 2 * kDim);
                    qg_part = ggml_view_1d(ctx, qkv_flat, qFullDim, 0);
                    k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qFullDim) * sizeof(float));
                    v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));
                }

                ggml_tensor* qg_3d = ggml_reshape_3d(ctx, qg_part, head_dim, 2, num_heads);
                ggml_tensor* q_view = ggml_view_2d(ctx, qg_3d, head_dim, num_heads, qg_3d->nb[2], 0);
                ggml_tensor* gate_view = ggml_view_2d(ctx, qg_3d, head_dim, num_heads, qg_3d->nb[2], qg_3d->nb[1]);

                ggml_tensor* q_2d_raw = ggml_cont(ctx, q_view);
                ggml_tensor* k_2d_raw = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d_raw, eps), t.q_norm_w);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d_raw, eps), t.k_norm_w);

                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);
                ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

                ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
                ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
                ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
                ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
                ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
                ggml_tensor* v_write = ggml_cont(ctx, v_perm);
                ggml_tensor* mask_for_attn;
                if (persist)
                {
                    // KV write via set_rows: the write position is an I64 INPUT
                    // (shared_kv_index), not a baked view offset, so the graph
                    // topology is identical token-to-token (CUDA-graph capturable).
                    t.k_cpy = ggml_set_rows(ctx, t.k_cache_base, k_write, shared_kv_index);
                    t.v_cpy = ggml_set_rows(ctx, t.v_cache_base, v_write, shared_kv_index);
                    mask_for_attn = shared_attn_mask;
                }
                else
                {
                    const std::size_t kv_byte_offset = static_cast<std::size_t>(position) * t.k_cache_base->nb[1];
                    ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cache_base, head_dim, 1, num_kv_heads, t.k_cache_base->nb[1], t.k_cache_base->nb[2], kv_byte_offset);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cache_base, head_dim, 1, num_kv_heads, t.v_cache_base->nb[1], t.v_cache_base->nb[2], kv_byte_offset);
                    t.k_cpy = ggml_cpy(ctx, k_write, k_dst);
                    t.v_cpy = ggml_cpy(ctx, v_write, v_dst);
                    if (flash_attn_requires_masked_padding(head_dim))
                    {
                        t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                        fill_flash_attn_mask(t.attn_mask_data, attnKvLen, totalSeqLen);
                    }
                    mask_for_attn = t.attn_mask;
                }
                ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache_base, head_dim, cache_size, num_kv_heads, 0, attnKvLen, kv_cache_type);
                ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache_base, head_dim, cache_size, num_kv_heads, 0, attnKvLen, kv_cache_type);
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Qwen3.5 model decode: failed to build KV cache views.");
                    if (persist) ggml_free(ctx);
                    return 0;
                }
                ggml_tensor* attn_out_2d;
                if (use_non_flash_attn)
                {
                    // Non-flash attention (avoids ggml-vulkan's incorrect aligned
                    // flash-attn shader on the padded persist window; see the
                    // use_non_flash_attn comment above). Single decode query:
                    //   q_attn : [head_dim, 1, num_heads]
                    //   k_full/v_full : [head_dim, KV, num_kv_heads]
                    // Materialize K/V as CONTIGUOUS F32 so the GQA-broadcast matmuls
                    // work for any KV cache dtype (F16/quant) and any window stride
                    // (view_kv_cache_window returns a view strided by cache_size), which
                    // is what flash_attn_ext handles internally.
                    ggml_tensor* k_f32 = ggml_cpy(ctx, k_full, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, attnKvLen, num_kv_heads));
                    ggml_tensor* v_f32 = ggml_cpy(ctx, v_full, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, attnKvLen, num_kv_heads));
                    // scores = softmax( (Q·Kᵀ)*scale + mask ), broadcasting num_kv_heads→num_heads.
                    // q_attn is a permuted (non-contiguous) view; ggml_mul_mat needs a
                    // contiguous src1, so cont it (flash_attn_ext handled the permute
                    // internally). Matches the verify-path non-flash fallback.
                    ggml_tensor* q_attn_cont = ggml_cont(ctx, q_attn);
                    ggml_tensor* kq = ggml_mul_mat(ctx, k_f32, q_attn_cont);            // [KV, 1, num_heads]
                    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                    ggml_tensor* kq_soft = ggml_soft_max_ext(ctx, kq, mask_for_attn, attn_scale, 0.0f);
                    // out = scores · V  (transpose V to [KV, head_dim, num_kv_heads])
                    ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v_f32, 1, 0, 2, 3));
                    ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq_soft);                  // [head_dim, 1, num_heads]
                    attn_out_2d = ggml_reshape_2d(ctx, kqv, head_dim, num_heads);
                }
                else
                {
                    ggml_tensor* attn_out_4d = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, mask_for_attn, attn_scale, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(attn_out_4d, GGML_PREC_F32);
                    attn_out_2d = ggml_reshape_2d(ctx, attn_out_4d, head_dim, num_heads);
                }
                ggml_tensor* gate_2d = ggml_cont(ctx, gate_view);
                ggml_tensor* attn_gated = ggml_mul(ctx, attn_out_2d, ggml_sigmoid(ctx, gate_2d));
                ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_gated, qDim, 1);
                ggml_tensor* o_mm = ggml_mul_mat(ctx, t.o_w, attn_flat);
                block_out = ggml_reshape_1d(ctx, o_mm, H);
                if (tp_mode) { tp_partial.push_back(o_mm); tp_boundary.push_back(block_out); }
            }
            else
            {
                // ===== Gated Delta Net (linear attention) =====
                ggml_tensor* qkv_mixed;
                ggml_tensor* z;
                ggml_tensor* beta_raw;
                ggml_tensor* alpha_raw;
                if (t.gdn_gate_w == nullptr)
                {
                    // Packed in-projection: one matmul, sliced [Q|K|V | Z | beta | alpha].
                    const std::int64_t packed_dim = d.gdn_qkv_ne1;
                    ggml_tensor* packed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed_2d);          // [packed_dim, 1]
                    ggml_tensor* packed_flat = ggml_reshape_1d(ctx, packed, packed_dim);
                    qkv_mixed = ggml_reshape_2d(ctx,
                        ggml_view_1d(ctx, packed_flat, conv_dim, 0), conv_dim, 1);
                    z = ggml_reshape_2d(ctx,
                        ggml_view_1d(ctx, packed_flat, value_dim,
                            static_cast<std::size_t>(conv_dim) * sizeof(float)), value_dim, 1);
                    beta_raw = ggml_reshape_2d(ctx,
                        ggml_view_1d(ctx, packed_flat, num_v_heads,
                            static_cast<std::size_t>(conv_dim + value_dim) * sizeof(float)), num_v_heads, 1);
                    alpha_raw = ggml_reshape_2d(ctx,
                        ggml_view_1d(ctx, packed_flat, num_v_heads,
                            static_cast<std::size_t>(conv_dim + value_dim + num_v_heads) * sizeof(float)), num_v_heads, 1);
                }
                else
                {
                    qkv_mixed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed_2d);          // [conv_dim, 1]
                    z = ggml_mul_mat(ctx, t.gdn_gate_w, normed_2d);                 // [value_dim, 1]
                    beta_raw = ggml_mul_mat(ctx, t.ssm_beta_w, normed_2d);          // [num_v_heads, 1]
                    alpha_raw = ggml_mul_mat(ctx, t.ssm_alpha_w, normed_2d);        // [num_v_heads, 1]
                }

                ggml_tensor* beta = ggml_sigmoid(ctx, beta_raw);
                beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, 1, 1);

                ggml_tensor* alpha_1d = ggml_reshape_1d(ctx, alpha_raw, num_v_heads);
                ggml_tensor* g = ggml_softplus(ctx, ggml_add(ctx, alpha_1d, t.ssm_dt_w));    // softplus(alpha + dt)
                g = ggml_mul(ctx, g, t.ssm_a_w);                                             // * (-exp(A_log))
                g = ggml_reshape_4d(ctx, g, 1, num_v_heads, 1, 1);

                // conv over the host ring state + the new mixed input. Concat straight
                // from the non-contiguous transpose view (ggml_concat writes a contiguous
                // result for ssm_conv) — drops a redundant transpose-cont copy.
                ggml_tensor* conv_input = ggml_concat(ctx, t.conv_state_in, ggml_transpose(ctx, qkv_mixed), 0); // [convDim+1, conv_dim]
                ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, t.conv1d_w);           // [conv_dim, 1]
                conv_out = ggml_silu(ctx, conv_out);
                ggml_tensor* conv_out_1d = ggml_reshape_1d(ctx, conv_out, conv_dim);

                // new conv state = the most recent convDim time-steps of conv_input,
                // written in-place back to the device-resident conv-state buffer.
                ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, conv_dim,
                    conv_input->nb[1], static_cast<std::size_t>(1) * conv_input->nb[0]));
                t.conv_state_out = ggml_cpy(ctx, new_conv, t.conv_state_in);

                // split q/k/v
                ggml_tensor* q_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads,
                    static_cast<std::size_t>(head_k_dim) * sizeof(float), 0));
                ggml_tensor* k_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads,
                    static_cast<std::size_t>(head_k_dim) * sizeof(float), static_cast<std::size_t>(key_dim) * sizeof(float)));
                ggml_tensor* v_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_v_dim, num_v_heads,
                    static_cast<std::size_t>(head_v_dim) * sizeof(float), static_cast<std::size_t>(2 * key_dim) * sizeof(float)));

                q_c = ggml_l2_norm(ctx, q_c, eps);
                k_c = ggml_l2_norm(ctx, k_c, eps);

                // q/k keep num_k_heads heads: the fused gated_delta_net kernel broadcasts
                // each v-head h to k-head (h % num_k_heads) internally, so the explicit
                // concat-tiling to num_v_heads is redundant (matches the verify/prefill path
                // and llama.cpp's fused GDN). Fewer op dispatches (helps Vulkan decode).
                ggml_tensor* q4 = ggml_reshape_4d(ctx, q_c, head_k_dim, num_k_heads, 1, 1);
                ggml_tensor* k4 = ggml_reshape_4d(ctx, k_c, head_k_dim, num_k_heads, 1, 1);
                ggml_tensor* v4 = ggml_reshape_4d(ctx, v_c, head_v_dim, num_v_heads, 1, 1);
                ggml_tensor* state4 = ggml_reshape_4d(ctx, t.delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1);

                ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g, beta, state4, 1);
                ggml_tensor* gdn_out = ggml_view_4d(ctx, gdn, head_v_dim, num_v_heads, 1, 1,
                    ggml_row_size(gdn->type, head_v_dim),
                    ggml_row_size(gdn->type, head_v_dim * num_v_heads),
                    ggml_row_size(gdn->type, head_v_dim * num_v_heads), 0);
                ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                    ggml_row_size(gdn->type, head_k_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                    ggml_row_size(gdn->type, head_v_dim * num_v_heads));
                // write the new recurrent state in-place back to the device-resident
                // delta-state buffer (state4 aliases delta_state_in).
                t.delta_state_out = ggml_cpy(ctx, new_state, state4);

                // gated RMSNorm: rms_norm(core, ssm_norm) * silu(z)
                ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads);
                ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), t.ssm_norm_w);
                ggml_tensor* z_2d = ggml_reshape_2d(ctx, z, head_v_dim, num_v_heads);
                ggml_tensor* gated = ggml_mul(ctx, out_n, ggml_silu(ctx, z_2d));
                ggml_tensor* gated_flat = ggml_reshape_2d(ctx, gated, value_dim, 1);
                ggml_tensor* ssm_mm = ggml_mul_mat(ctx, t.ssm_out_w, gated_flat);
                block_out = ggml_reshape_1d(ctx, ssm_mm, H);
                if (tp_mode) { tp_partial.push_back(ssm_mm); tp_boundary.push_back(block_out); }
            }

            ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out);

            // ===== FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w);
            ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, H, 1);
            ggml_tensor* ffn_down;
            if (d.is_moe == 0)
            {
                // dense SwiGLU: fused silu(gate)*up over the packed [gate|up] matmul
                // output (gate-first, matching ggml_swiglu swapped=false). One gated
                // kernel instead of a cont + standalone silu + separate mul — fewer op
                // dispatches, which matters most on backends without graph capture (Vulkan).
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed_2d); // [2*ffDense, 1]
                ggml_tensor* act_2d = ggml_swiglu(ctx, gu);                 // [ffDense, 1]
                ggml_tensor* down_mm = ggml_mul_mat(ctx, t.down_w, act_2d);
                ffn_down = ggml_reshape_1d(ctx, down_mm, H);
                if (tp_mode) { tp_partial.push_back(down_mm); tp_boundary.push_back(ffn_down); }
            }
            else
            {
                // ----- MoE: router -> top-k -> renorm -> stacked experts + gated shared expert -----
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed_2d); // [num_experts, 1]
                ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
                ggml_tensor* sel = ggml_top_k(ctx, probs, num_experts_used);                 // [num_used, 1]
                ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, 1);
                ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                            // [1, num_used, 1]
                ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, num_experts_used, 1);
                if (norm_topk != 0)
                {
                    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
                    w_2d = ggml_div(ctx, w_2d, w_sum);
                }
                if (expert_weights_scale != 1.0f)
                    w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, 1);

                // Expert-parallel TP: confine the (global) top-k to this rank's
                // expert slice — local ids via the I32 LUT, foreign contributions
                // nullified through the weight mask. The renormalization above
                // ran on the global weights first, so every rank's masked
                // weights sum to the single-GPU values across the group.
                ggml_tensor* sel_ids = sel;
                if (ep_lut != nullptr)
                {
                    ggml_tensor* lut_r = ggml_reshape_3d(ctx, ep_lut, 1, num_experts, 1);
                    ggml_tensor* local_ids = ggml_get_rows(ctx, lut_r, sel);                  // [1, num_used, 1] I32
                    sel_ids = ggml_reshape_2d(ctx, local_ids, num_experts_used, 1);
                    ggml_tensor* mask_r = ggml_reshape_3d(ctx, ep_mask, 1, num_experts, 1);
                    ggml_tensor* own_mask = ggml_get_rows(ctx, mask_r, sel);                  // [1, num_used, 1] F32
                    w_final = ggml_mul(ctx, w_final, own_mask);
                }

                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, 1);
                ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel_ids);   // [expert_ff, num_used, 1]
                ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel_ids);
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);               // [expert_ff, num_used, 1]
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel_ids);      // [hidden, num_used, 1]
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);

                ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], 0);
                for (int u = 1; u < num_experts_used; ++u)
                {
                    ggml_tensor* vu = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, vu);
                }
                ggml_tensor* moe_out_1d = ggml_reshape_1d(ctx, moe_out, H);

                // gated shared expert
                ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed_2d);         // [shared_ff, 1]
                ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed_2d);
                ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                ggml_tensor* sh_down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.shexp_down_w, sh_act), H);
                ggml_tensor* sh_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed_2d)); // [1,1]
                ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);

                // Both the routed sum (this rank's experts only) and the
                // Megatron-split shared expert are partials, so their sum is
                // the layer's single reduction point.
                ffn_down = ggml_add(ctx, moe_out_1d, sh_out);
                if (tp_mode) { tp_partial.push_back(ffn_down); tp_boundary.push_back(ffn_down); }
            }

            hidden = ggml_add(ctx, residual1, ffn_down);
        }

        ggml_tensor* hidden_out;
        ggml_tensor* out_cpy;
        if (fold)
        {
            // Final RMSNorm * weight, then lm_head -> logits, folded into the graph
            // so the lm_head matmul + the 248K-vocab output are part of the captured
            // replay (no separate per-token lm_head graph_compute / submit).
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* fn_2d = ggml_reshape_2d(ctx, fn, H, 1);
            ggml_tensor* logits = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lm_head_t, fn_2d), vocab_size);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
            out_cpy = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        }
        ggml_set_output(out_cpy);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 160 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent == 0)
            {
                ggml_build_forward_expand(graph, lt[l].k_cpy);
                ggml_build_forward_expand(graph, lt[l].v_cpy);
            }
            else
            {
                ggml_set_output(lt[l].conv_state_out);
                ggml_set_output(lt[l].delta_state_out);
                ggml_build_forward_expand(graph, lt[l].conv_state_out);
                ggml_build_forward_expand(graph, lt[l].delta_state_out);
            }
        }
        ggml_build_forward_expand(graph, out_cpy);

        // --- bind tensors ---
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

        const std::size_t convStateBytes = static_cast<std::size_t>(convDim) * conv_dim * sizeof(float);
        const std::size_t deltaStateBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * sizeof(float);
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            if (d.is_moe == 0)
            {
                bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
                bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            }
            else
            {
                bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes), true);
                bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes), true);
                bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes), true);
                bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes), true);
                bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes), true);
                bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes), true);
                bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes), true);
                bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float), true);
            }
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_cache_base, d.k_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cache_base, d.v_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (t.attn_mask != nullptr && !t.attn_mask_data.empty())
                    bind_or_mark(t.attn_mask, t.attn_mask_data.data(), t.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes), true);
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes), true);
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes), true);
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes), true);
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float), true);
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float), true);
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes), true);
                // GDN recurrent state is device-resident across decode tokens
                // (cacheable COMPUTE buffer, updated in-place each token); the C#
                // seed path invalidates these on (re)seed so the host state is
                // re-uploaded. Mirrors the KV-cache binding.
                bind_or_mark(t.conv_state_in, d.conv_state_in, convStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.delta_state_in, d.delta_state_in, deltaStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
        }
        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);
        }

        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (persist)
        {
            // STABLE addresses for CUDA-graph capture: give every tensor its own
            // slot (no gallocr lifetime packing, whose plan can move addresses).
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Qwen3.5 model decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
            if (vram_log_enabled())
                vram_log("q35-decode-persist", static_cast<std::int64_t>(ggml_backend_buffer_get_size(persist_buf)));
        }
        else
        {
            // Non-persist (Metal): the whole-model GDN decode graph must NOT use the
            // shared reuse gallocr. Its lifetime-packing mis-aliases this graph's
            // intermediates across the gated_delta_net + in-place recurrent-state
            // ggml_cpy chain, so the packed scratch retains the PREVIOUS token's
            // activations and they leak into the current token — producing coherent
            // but interleaved-garbage output (two tokens' continuations merged word
            // by word). A fresh per-call backend buffer fixes it; the decode graph's
            // scratch is small so the alloc costs only ~4 ms/token (still ~5x faster
            // than the op-by-op path). The dense Gemma4 decode is unaffected (it has
            // no in-place recurrent state) and keeps the reuse gallocr.
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Qwen3.5 model decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        // Zero the padded KV rows [totalSeqLen, attnKvLen) once per graph build for the
        // Vulkan non-flash persist path. That path reads the FULL padded window and
        // masks the padding to -inf via soft_max_ext. But the padded rows are
        // UNINITIALIZED device memory whose F16/quant garbage can decode to +/-inf;
        // `inf + (-inf) = NaN` inside soft_max survives the mask and corrupts attention
        // (coherent for the first 256-window, then a repetition loop once the window
        // grows to 512+ and the padding region is large). Zeroing makes the masked-out
        // rows finite (q·0 = 0 -> exp(0 + -inf) = 0), matching llama.cpp's
        // zero-initialized KV cache. Only positions < totalSeqLen are read as valid;
        // the growing decode writes into this zeroed tail as it advances within the
        // stride, so one build-time zero covers the whole stride. Cheap: a few memsets
        // per attention layer, once per 256-token window grow (not per token).
        if (use_non_flash_attn && attnKvLen > totalSeqLen)
        {
            const std::size_t kvRowBytes = ggml_row_size(static_cast<ggml_type>(kv_cache_type), head_dim);
            const std::size_t padBytes = static_cast<std::size_t>(attnKvLen - totalSeqLen) * kvRowBytes;
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent != 0)
                    continue;
                LayerTensors& t = lt[l];
                if (t.k_cache_base == nullptr || t.v_cache_base == nullptr)
                    continue;
                for (int h = 0; h < num_kv_heads; h++)
                {
                    const std::size_t off =
                        (static_cast<std::size_t>(h) * cache_size + static_cast<std::size_t>(totalSeqLen)) * kvRowBytes;
                    ggml_backend_tensor_memset(t.k_cache_base, 0, off, padBytes);
                    ggml_backend_tensor_memset(t.v_cache_base, 0, off, padBytes);
                }
            }
        }

        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * sizeof(float));
        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));
        if (persist)
        {
            std::int64_t kv_idx = position;
            ggml_backend_tensor_set(shared_kv_index, &kv_idx, 0, sizeof(std::int64_t));
            std::vector<ggml_fp16_t> mask_data;
            fill_flash_attn_mask(mask_data, attnKvLen, totalSeqLen);
            ggml_backend_tensor_set(shared_attn_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        }
        if (ep_lut != nullptr)
        {
            ggml_backend_tensor_set(ep_lut, ep_lut_data.data(), 0, ep_lut_data.size() * sizeof(std::int32_t));
            ggml_backend_tensor_set(ep_mask, ep_mask_data.data(), 0, ep_mask_data.size() * sizeof(float));
        }

        if (tp_mode)
        {
            // Hand the built graph back as a rank plan instead of running it:
            // the driver executes every rank's plan segment-by-segment with the
            // partials reduced at the recorded cut points. The persist cache
            // keeps the ctx/graph/buffer alive across tokens; the replay path
            // above refreshes the per-token inputs and re-returns this plan.
            Q35DecodeCache* slot = dcb;
            slot->tp_plan.clear();
            slot->tp_plan.graph = graph;
            slot->tp_plan.ar_tensor = tp_partial;
            if (!tp_plan_segments(slot->tp_plan, tp_boundary))
            {
                slot->tp_plan.clear();
                ggml_backend_buffer_free(persist_buf);
                ggml_free(ctx);
                slot->ctx = nullptr; slot->buffer = nullptr; slot->graph = nullptr; slot->valid = false;
                return 0;
            }
            const int out_count_tp = fold ? vocab_size : H;
            slot->tp_plan.out_tensor = hidden_out;
            slot->tp_plan.out_host = fold ? logits_data : hidden_data;
            slot->tp_plan.out_bytes = static_cast<std::size_t>(out_count_tp) * sizeof(float);
            slot->ctx = ctx;
            slot->buffer = persist_buf;
            slot->graph = graph;
            slot->hidden_t = hidden_t;
            slot->hidden_out = hidden_out;
            slot->pos_tensor = pos_tensor;
            slot->kv_index = shared_kv_index;
            slot->attn_mask = shared_attn_mask;
            slot->sig_disc = sig_disc;
            slot->sig_kcache0 = sig_kcache0;
            slot->num_layers = num_layers;
            slot->hidden_size = H;
            slot->window = attnKvLen;
            slot->folded = fold;
            slot->out_count = out_count_tp;
            slot->valid = true;
            *tp_plan_out = &slot->tp_plan;
            clear_last_error();
            return 1;
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 model decode: graph execution failed.");
            if (persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        // GDN state is device-resident (updated in-place). Download either the
        // folded logits or the bare hidden state.
        const int out_count = fold ? vocab_size : H;
        void* out_data = fold ? logits_data : hidden_data;
        finalize_compute_with_download(hidden_out, out_data, static_cast<std::size_t>(out_count) * sizeof(float));
        if (persist || buffer.value != nullptr) host_read_barrier();

        if (persist && dcb != nullptr)
        {
            // Keep the ctx/graph/buffer alive for capture+replay on later tokens.
            dcb->ctx = ctx;
            dcb->buffer = persist_buf;
            dcb->graph = graph;
            dcb->hidden_t = hidden_t;
            dcb->hidden_out = hidden_out;
            dcb->pos_tensor = pos_tensor;
            dcb->kv_index = shared_kv_index;
            dcb->attn_mask = shared_attn_mask;
            dcb->sig_disc = sig_disc;
            dcb->sig_kcache0 = sig_kcache0;
            dcb->num_layers = num_layers;
            dcb->hidden_size = H;
            dcb->window = attnKvLen;
            dcb->folded = fold;
            dcb->out_count = out_count;
            dcb->valid = true;
        }
        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_Qwen35ModelDecode(
    const TSGgmlQwen35LayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int position,
    int num_heads, int num_kv_heads, int head_dim, int cache_size,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    int tp_degree, void** tp_plan_out)
{
    try
    {
        int r = qwen35_model_decode_impl(
            layers, num_layers, hidden_data, hidden_size, position,
            num_heads, num_kv_heads, head_dim, cache_size,
            rope_n_dims, rope_mode, kv_cache_type,
            conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            eps, rope_base, rope_freq_scale,
            num_experts, num_experts_used, expert_ff, shared_ff,
            norm_topk, expert_weights_scale,
            logits_data, vocab_size,
            lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
            final_norm_data,
            tp_degree, tp_plan_out);
        return r;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in Qwen3.5 model decode.");
        return 0;
    }
}

// Drop the persistent decode-graph cache. Called from C# whenever the GDN
// recurrent state is re-seeded / the per-op path runs (the cached graph pins the
// conv/delta device-buffer addresses, which move on re-seed), so the next fused
// decode rebuilds against the fresh state.
TSG_EXPORT void TSGgml_Qwen35ResetDecodeCache()
{
    g_q35dc_pool.reset_all();
}

// ============================================================================
// TSGgml_Qwen35ModelDecodeBatched
//
// TRUE token-batched fused decode (vLLM-style continuous batching): processes
// N sequences' decode tokens (one token per sequence) through the WHOLE hybrid
// transformer in ONE ggml graph. The heavy per-layer matmuls (QKV/GDN input
// projections, dense/MoE FFN, lm-head) run BATCHED over all N tokens, so the
// quantized weights are read from VRAM ONCE per step and amortized across the
// batch — the core throughput win over running N separate single-sequence
// decodes. The cheap per-token recurrent/attention ops (ssm_conv +
// gated_delta_net for GDN layers, flash-attn for attention layers) are emitted
// per-sequence inside the same graph (N small), reusing the exact validated
// single-sequence shapes.
//
// KV cache is PAGED: each attention layer owns a device-resident pool
// [head_dim*num_kv_heads, total_slots]; token t writes its K/V to slot_mapping[t]
// via ggml_set_rows, and each sequence gathers its own history from the pool via
// ggml_get_rows over its block-table-derived slot list (gather_idx/gather_off).
//
// GDN recurrent state is host-passed per batch ([.., n_seqs] conv ring + delta
// state), updated and downloaded each step (the C# layer scatters it back to the
// per-sequence slots), keeping this path trivially correct alongside the per-op
// fallback (host buffers are the single source of truth).
//
// Returns the final pre-output-norm hidden state [hidden, n_tokens]; the C#
// caller owns output_norm + the per-sequence LM head. Returns 0 on anything it
// cannot handle so the caller falls back to the op-by-op batched path.
// ============================================================================
namespace
{
    // Persistent batched-decode graph cache (single entry) for CUDA-graph capture.
    // Built ONCE with stable tensor addresses (raw ctx + alloc_ctx_tensors) and
    // reused across decode steps so ggml-cuda's CUDA-graph capture engages
    // (key = cgraph->nodes[0]); replays one captured graph instead of relaunching
    // the whole ~Nlayers*Nseqs-node graph per step (the WDDM per-node launch tax).
    // Per-step inputs (hidden, positions, slot_mapping, padded gather idx, per-seq
    // mask, GDN conv/delta state) are uploaded to stable addresses each step; the
    // KV pools + weights stay in cached device buffers. Dropped + rebuilt when the
    // shape signature (n_seqs / pad_kv / model) changes or the pools move
    // (TSGgml_Qwen35ResetBatchedDecodeCache from C#).
    struct Q35BatchedDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_t = nullptr;
        ggml_tensor* slot_t = nullptr;
        std::vector<ggml_tensor*> gidx;        // [n_seqs] padded gather idx
        std::vector<ggml_tensor*> mask;        // [n_seqs] F16 attn mask
        std::vector<ggml_tensor*> conv_state;  // per recurrent layer
        std::vector<ggml_tensor*> delta_state; // per recurrent layer
        std::vector<int> gdn_layer;            // layer index for each gdn state entry
        const void* sig = nullptr;
        int num_layers = 0, hidden_size = 0, n_seqs = 0, pad_kv = 0;
        std::size_t conv_bytes = 0, delta_bytes = 0;
        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_t = hidden_out = pos_t = slot_t = nullptr;
            gidx.clear(); mask.clear(); conv_state.clear(); delta_state.clear(); gdn_layer.clear();
            sig = nullptr; num_layers = hidden_size = n_seqs = pad_kv = 0; conv_bytes = delta_bytes = 0;
        }
    };
    Q35BatchedDecodeCache g_q35bdc;

    inline void bfd_upload_mask(ggml_tensor* mask_t, int pad_kv, int seq_len)
    {
        std::vector<ggml_fp16_t> m;
        fill_flash_attn_mask(m, pad_kv, seq_len);
        ggml_backend_tensor_set(mask_t, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
    }

    int qwen35_model_decode_batched_impl(
        const TSGgmlQwen35LayerDesc* layers, int num_layers,
        void* hidden_data, int hidden_size, int n_tokens, int n_seqs,
        const int* positions,            // [n_tokens] I32 absolute positions
        const std::int64_t* slot_mapping,// [n_tokens] I64 global KV slot per token
        const int* gather_idx,           // [n_seqs * pad_kv] I32 padded per-seq slot lists
        const int* seq_lens,             // [n_seqs] I32 valid length per seq (drives attn mask)
        int pad_kv, int total_slots,
        int num_heads, int num_kv_heads, int head_dim,
        int rope_n_dims, int rope_mode, int kv_cache_type,
        int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
        float eps, float rope_base, float rope_freq_scale,
        int num_experts, int num_experts_used, int expert_ff, int shared_ff,
        int norm_topk, float expert_weights_scale)
    {
        if (!ensure_backend()) return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr || n_tokens <= 0 || n_seqs <= 0)
        {
            set_last_error("Qwen3.5 batched decode: invalid arguments.");
            return 0;
        }
        if (n_tokens != n_seqs)
        {
            set_last_error("Qwen3.5 batched decode: V1 requires one token per sequence (n_tokens==n_seqs).");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 batched decode: descriptor size mismatch.");
            return 0;
        }

        const int H = hidden_size;
        const int T = n_tokens;
        const int qDim = num_heads * head_dim;
        const int qFullDim = qDim * 2;            // Q + gate interleaved per head
        const int kDim = num_kv_heads * head_dim;
        const int kvFlat = num_kv_heads * head_dim; // pool row size
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const int head_tile = (num_k_heads > 0) ? (num_v_heads / num_k_heads) : 1;

        const void* sig = layers[0].attn_norm_w;   // model-instance discriminator
        // Persistent capturable graph: opt-in (TS_QWEN35_BFD_PERSIST=1). On WDDM
        // (Windows) the captured replay regresses for this batched graph (the
        // dynamic paged gather + per-step GDN-state up/downloads force per-step
        // re-instantiation); on Linux/WSL ggml_cuda (no WDDM per-node tax) the
        // non-captured graph already runs near the kernel floor, so capture is
        // left as a knob for experimentation rather than the default.
        static const bool persist = []{ const char* e = std::getenv("TS_QWEN35_BFD_PERSIST"); return e != nullptr && e[0] == '1'; }();
        const std::size_t convStateBytes = static_cast<std::size_t>(convDim) * conv_dim * n_seqs * sizeof(float);
        const std::size_t deltaStateBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * n_seqs * sizeof(float);

        // ===== Persist reuse fast-path: replay the captured graph =====
        if (persist && g_q35bdc.valid && g_q35bdc.graph != nullptr &&
            g_q35bdc.sig == sig && g_q35bdc.num_layers == num_layers &&
            g_q35bdc.hidden_size == H && g_q35bdc.n_seqs == n_seqs && g_q35bdc.pad_kv == pad_kv)
        {
            host_read_barrier();
            ggml_backend_tensor_set(g_q35bdc.hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * T * sizeof(float));
            ggml_backend_tensor_set(g_q35bdc.pos_t, positions, 0, static_cast<std::size_t>(T) * sizeof(std::int32_t));
            ggml_backend_tensor_set(g_q35bdc.slot_t, slot_mapping, 0, static_cast<std::size_t>(T) * sizeof(std::int64_t));
            for (int s = 0; s < n_seqs; s++)
            {
                ggml_backend_tensor_set(g_q35bdc.gidx[s], gather_idx + static_cast<std::size_t>(s) * pad_kv, 0, static_cast<std::size_t>(pad_kv) * sizeof(std::int32_t));
                bfd_upload_mask(g_q35bdc.mask[s], pad_kv, seq_lens[s]);
            }
            for (std::size_t gi = 0; gi < g_q35bdc.gdn_layer.size(); ++gi)
            {
                int l = g_q35bdc.gdn_layer[gi];
                ggml_backend_tensor_set(g_q35bdc.conv_state[gi], layers[l].conv_state_in, 0, convStateBytes);
                ggml_backend_tensor_set(g_q35bdc.delta_state[gi], layers[l].delta_state_in, 0, deltaStateBytes);
            }
            ggml_status st = ggml_backend_graph_compute(g_backend, g_q35bdc.graph);
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Qwen3.5 batched decode: cached graph execution failed.");
                g_q35bdc.reset();
                return 0;
            }
            finalize_compute_with_download(g_q35bdc.hidden_out, hidden_data, static_cast<std::size_t>(H) * T * sizeof(float));
            for (std::size_t gi = 0; gi < g_q35bdc.gdn_layer.size(); ++gi)
            {
                int l = g_q35bdc.gdn_layer[gi];
                finalize_compute_with_download(g_q35bdc.conv_state[gi], layers[l].conv_state_out, convStateBytes);
                finalize_compute_with_download(g_q35bdc.delta_state[gi], layers[l].delta_state_out, deltaStateBytes);
            }
            host_read_barrier();
            return 1;
        }
        if (persist) g_q35bdc.reset();

        // 32 MB matches the single-seq decode + the pool's max slot size; the
        // metadata-only (no_alloc) ctx needs only ~1-2 MB even for a 40-layer
        // N-seq graph, so this is ample. Persist uses a raw ctx kept alive in the
        // cache for capture/replay; non-persist uses the pooled block.
        const std::size_t ctx_size = static_cast<std::size_t>(32) * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Qwen3.5 batched decode: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Qwen3.5 batched decode: failed to acquire ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        // --- per-token inputs ---
        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_tensor* slot_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
        // Per-seq PADDED gather index + attention mask (fixed size pad_kv so the
        // graph topology is identical token-to-token = CUDA-graph capturable).
        std::vector<ggml_tensor*> gidx(n_seqs, nullptr);
        std::vector<ggml_tensor*> mask(n_seqs, nullptr);
        for (int s = 0; s < n_seqs; s++)
        {
            gidx[s] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pad_kv);
            mask[s] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, pad_kv, 1, 1, 1);
            if (persist) { ggml_set_input(gidx[s]); ggml_set_input(mask[s]); }
        }
        if (persist)
        {
            ggml_set_input(hidden_t);
            ggml_set_input(pos_t);
            ggml_set_input(slot_t);
        }

        struct LayerTensors {
            ggml_tensor* attn_norm_w; ggml_tensor* post_attn_norm_w;
            ggml_tensor* qkv_w; ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w; ggml_tensor* o_w;
            ggml_tensor* k_pool; ggml_tensor* v_pool;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;
            ggml_tensor* gdn_qkv_w; ggml_tensor* gdn_gate_w; ggml_tensor* ssm_beta_w; ggml_tensor* ssm_alpha_w;
            ggml_tensor* conv1d_w; ggml_tensor* ssm_dt_w; ggml_tensor* ssm_a_w; ggml_tensor* ssm_norm_w; ggml_tensor* ssm_out_w;
            ggml_tensor* conv_state_in; ggml_tensor* delta_state_in;
            ggml_tensor* conv_state_out; ggml_tensor* delta_state_out;
            ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* gate_inp_w; ggml_tensor* gate_exps; ggml_tensor* up_exps; ggml_tensor* down_exps;
            ggml_tensor* shexp_gate_w; ggml_tensor* shexp_up_w; ggml_tensor* shexp_down_w; ggml_tensor* shexp_gate_inp_w;
        };
        std::vector<LayerTensors> lt(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            t = {};
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.is_recurrent == 0)
            {
                t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                if (d.separate_qkv != 0)
                {
                    t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                    t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                }
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                t.k_pool = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(kv_cache_type), kvFlat, total_slots);
                t.v_pool = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(kv_cache_type), kvFlat, total_slots);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                t.conv_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, convDim, conv_dim, n_seqs);
                t.delta_state_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads, n_seqs);
                if (persist) { ggml_set_input(t.conv_state_in); ggml_set_input(t.delta_state_in); }
            }
            if (d.is_moe == 0)
            {
                t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            }
            else
            {
                t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), hidden_size, expert_ff, num_experts);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), hidden_size, expert_ff, num_experts);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, hidden_size, num_experts);
                t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // --- build the chained graph ---
        std::vector<ggml_tensor*> gdn_state_writes; // in-place conv/delta state writes (graph outputs)
        ggml_tensor* hidden = hidden_t;   // [H, T]
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w); // [H, T]
            ggml_tensor* block_out; // [H, T]

            if (d.is_recurrent == 0)
            {
                // ===== Full attention (batched proj + per-seq flash-attn) =====
                ggml_tensor* qg_part; ggml_tensor* k_raw; ggml_tensor* v_raw;
                if (d.separate_qkv != 0)
                {
                    qg_part = ggml_mul_mat(ctx, t.qkv_w, normed);   // [qFullDim, T]
                    k_raw = ggml_mul_mat(ctx, t.k_w, normed);       // [kDim, T]
                    v_raw = ggml_mul_mat(ctx, t.v_w, normed);       // [kDim, T]
                }
                else
                {
                    ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed); // [qFullDim+2kDim, T]
                    qg_part = ggml_cont(ctx, ggml_view_2d(ctx, qkv, qFullDim, T, qkv->nb[1], 0));
                    k_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, T, qkv->nb[1], static_cast<std::size_t>(qFullDim) * sizeof(float)));
                    v_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, T, qkv->nb[1], static_cast<std::size_t>(qFullDim + kDim) * sizeof(float)));
                }

                // Deinterleave Q from gate: qg_part is [head_dim*2*num_heads, T].
                ggml_tensor* qg_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, qg_part), head_dim, 2, num_heads, T);
                ggml_tensor* q_view = ggml_view_4d(ctx, qg_4d, head_dim, 1, num_heads, T, qg_4d->nb[1], qg_4d->nb[2], qg_4d->nb[3], 0);
                ggml_tensor* gate_view = ggml_view_4d(ctx, qg_4d, head_dim, 1, num_heads, T, qg_4d->nb[1], qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);
                ggml_tensor* q_hd = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, q_view), head_dim, num_heads, T));   // [head_dim, num_heads, T]
                ggml_tensor* gate_hd = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, gate_view), head_dim, num_heads, T));
                ggml_tensor* k_hd = ggml_reshape_3d(ctx, k_raw, head_dim, num_kv_heads, T);
                ggml_tensor* v_hd = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, T);

                // per-head q/k RMSNorm (over head_dim = ne0).
                ggml_tensor* q_n = ggml_mul(ctx, ggml_rms_norm(ctx, q_hd, eps), t.q_norm_w);
                ggml_tensor* k_n = ggml_mul(ctx, ggml_rms_norm(ctx, k_hd, eps), t.k_norm_w);

                // RoPE per token (pos_t [T]).
                ggml_tensor* q_rope = ggml_rope_ext(ctx, q_n, pos_t, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0); // [head_dim, num_heads, T]
                ggml_tensor* k_rope = ggml_rope_ext(ctx, k_n, pos_t, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0); // [head_dim, num_kv_heads, T]

                // Write K/V to the paged pool: flatten heads, set_rows by slot.
                ggml_tensor* k_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, k_rope), kvFlat, T);
                ggml_tensor* v_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, v_hd), kvFlat, T);
                t.k_cpy = ggml_set_rows(ctx, t.k_pool, k_flat, slot_t);
                t.v_cpy = ggml_set_rows(ctx, t.v_pool, v_flat, slot_t);

                // Per-seq attention: gather this seq's KV history, flash-attn the
                // seq's single query token. Depends on the set_rows writes above
                // (so the seq's own freshly-written token is visible).
                std::vector<ggml_tensor*> attn_per_seq(n_seqs);
                for (int s = 0; s < n_seqs; s++)
                {
                    ggml_tensor* kf = ggml_get_rows(ctx, t.k_cpy, gidx[s]); // [kvFlat, pad_kv]
                    ggml_tensor* vf = ggml_get_rows(ctx, t.v_cpy, gidx[s]);
                    ggml_tensor* k3 = ggml_reshape_3d(ctx, kf, head_dim, num_kv_heads, pad_kv);
                    ggml_tensor* v3 = ggml_reshape_3d(ctx, vf, head_dim, num_kv_heads, pad_kv);
                    ggml_tensor* kperm = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3)); // [head_dim, pad_kv, num_kv_heads]
                    ggml_tensor* vperm = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
                    // q for seq s: [head_dim, num_heads, 1] -> permute [head_dim, 1, num_heads]
                    ggml_tensor* qs = ggml_view_3d(ctx, q_rope, head_dim, num_heads, 1, q_rope->nb[1], q_rope->nb[2], static_cast<std::size_t>(s) * q_rope->nb[2]);
                    ggml_tensor* qperm = ggml_cont(ctx, ggml_permute(ctx, qs, 0, 2, 1, 3)); // [head_dim, 1, num_heads]
                    // Padded gather: positions [seq_len, pad_kv) point at slot 0 and
                    // are masked out by mask[s] (0 valid, -inf padding).
                    ggml_tensor* o4 = ggml_flash_attn_ext(ctx, qperm, kperm, vperm, mask[s], attn_scale, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(o4, GGML_PREC_F32);
                    // o4: [head_dim, num_heads, 1, 1] -> [head_dim*num_heads, 1]
                    attn_per_seq[s] = ggml_reshape_2d(ctx, o4, qDim, 1);
                }
                ggml_tensor* attn_cat = attn_per_seq[0];
                for (int s = 1; s < n_seqs; s++)
                    attn_cat = ggml_concat(ctx, attn_cat, attn_per_seq[s], 1); // [qDim, T]

                // Sigmoid-gated output: attn * sigmoid(gate).
                ggml_tensor* gate_flat = ggml_reshape_2d(ctx, gate_hd, qDim, T);
                ggml_tensor* attn_gated = ggml_mul(ctx, attn_cat, ggml_sigmoid(ctx, gate_flat));
                block_out = ggml_mul_mat(ctx, t.o_w, attn_gated); // [H, T]
            }
            else
            {
                // ===== Gated Delta Net (batched proj + per-seq recurrence) =====
                ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed);   // [conv_dim, T]
                ggml_tensor* z_all = ggml_mul_mat(ctx, t.gdn_gate_w, normed);      // [value_dim, T]
                ggml_tensor* beta_all = ggml_sigmoid(ctx, ggml_mul_mat(ctx, t.ssm_beta_w, normed));   // [num_v_heads, T]
                ggml_tensor* alpha_all = ggml_mul_mat(ctx, t.ssm_alpha_w, normed); // [num_v_heads, T]
                // g = softplus(alpha + dt) * a  (per head, broadcast over T)
                ggml_tensor* g_all = ggml_softplus(ctx, ggml_add(ctx, alpha_all, t.ssm_dt_w));
                g_all = ggml_mul(ctx, g_all, t.ssm_a_w);                            // [num_v_heads, T]

                std::vector<ggml_tensor*> gdn_per_seq(n_seqs);
                for (int s = 0; s < n_seqs; s++)
                {
                    ggml_tensor* qkv_s = ggml_cont(ctx, ggml_view_2d(ctx, qkv_mixed, conv_dim, 1, qkv_mixed->nb[1], static_cast<std::size_t>(s) * qkv_mixed->nb[1])); // [conv_dim, 1]
                    ggml_tensor* conv_state_s = ggml_view_2d(ctx, t.conv_state_in, convDim, conv_dim, t.conv_state_in->nb[1], static_cast<std::size_t>(s) * t.conv_state_in->nb[2]); // [convDim, conv_dim]
                    ggml_tensor* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, qkv_s));   // [1, conv_dim]
                    ggml_tensor* conv_input = ggml_concat(ctx, conv_state_s, qkv_T, 0); // [convDim+1, conv_dim]
                    ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, t.conv1d_w)); // [conv_dim, 1]
                    ggml_tensor* conv_out_1d = ggml_reshape_1d(ctx, conv_out, conv_dim);
                    ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, conv_dim, conv_input->nb[1], static_cast<std::size_t>(1) * conv_input->nb[0]));
                    ggml_tensor* conv_save = ggml_cpy(ctx, new_conv, conv_state_s);

                    ggml_tensor* q_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads, static_cast<std::size_t>(head_k_dim) * sizeof(float), 0));
                    ggml_tensor* k_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads, static_cast<std::size_t>(head_k_dim) * sizeof(float), static_cast<std::size_t>(key_dim) * sizeof(float)));
                    ggml_tensor* v_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_v_dim, num_v_heads, static_cast<std::size_t>(head_v_dim) * sizeof(float), static_cast<std::size_t>(2 * key_dim) * sizeof(float)));
                    q_c = ggml_l2_norm(ctx, q_c, eps);
                    k_c = ggml_l2_norm(ctx, k_c, eps);
                    ggml_tensor* q_tl = q_c; ggml_tensor* k_tl = k_c;
                    for (int r = 1; r < head_tile; r++) { q_tl = ggml_concat(ctx, q_tl, q_c, 1); k_tl = ggml_concat(ctx, k_tl, k_c, 1); }
                    ggml_tensor* q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q_tl), head_k_dim, num_v_heads, 1, 1);
                    ggml_tensor* k4 = ggml_reshape_4d(ctx, ggml_cont(ctx, k_tl), head_k_dim, num_v_heads, 1, 1);
                    ggml_tensor* v4 = ggml_reshape_4d(ctx, v_c, head_v_dim, num_v_heads, 1, 1);
                    ggml_tensor* beta_s = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, beta_all, num_v_heads, 1, beta_all->nb[1], static_cast<std::size_t>(s) * beta_all->nb[1])), 1, num_v_heads, 1, 1);
                    ggml_tensor* g_s = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, g_all, num_v_heads, 1, g_all->nb[1], static_cast<std::size_t>(s) * g_all->nb[1])), 1, num_v_heads, 1, 1);
                    ggml_tensor* state_s = ggml_view_4d(ctx, t.delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1, t.delta_state_in->nb[1], t.delta_state_in->nb[2], t.delta_state_in->nb[3], static_cast<std::size_t>(s) * t.delta_state_in->nb[3]);
                    ggml_tensor* state4 = ggml_cont(ctx, state_s);

                    ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g_s, beta_s, state4, 1);
                    ggml_tensor* gdn_out = ggml_view_4d(ctx, gdn, head_v_dim, num_v_heads, 1, 1,
                        ggml_row_size(gdn->type, head_v_dim), ggml_row_size(gdn->type, head_v_dim * num_v_heads), ggml_row_size(gdn->type, head_v_dim * num_v_heads), 0);
                    ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                        ggml_row_size(gdn->type, head_k_dim), ggml_row_size(gdn->type, head_k_dim * head_v_dim), ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                        ggml_row_size(gdn->type, head_v_dim * num_v_heads));
                    ggml_tensor* state_save = ggml_cpy(ctx, new_state, state_s);

                    // gated RMSNorm with z, then collect [value_dim, 1].
                    ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads);
                    ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), t.ssm_norm_w);
                    ggml_tensor* z_s = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z_all, value_dim, 1, z_all->nb[1], static_cast<std::size_t>(s) * z_all->nb[1])), head_v_dim, num_v_heads);
                    ggml_tensor* gated = ggml_mul(ctx, out_n, ggml_silu(ctx, z_s));
                    gdn_per_seq[s] = ggml_reshape_2d(ctx, gated, value_dim, 1);
                    // The in-place conv/delta state writes are graph outputs (the
                    // updated state is downloaded after compute); collect them so
                    // ggml_build_forward_expand includes them.
                    gdn_state_writes.push_back(conv_save);
                    gdn_state_writes.push_back(state_save);
                }
                ggml_tensor* gdn_cat = gdn_per_seq[0];
                for (int s = 1; s < n_seqs; s++)
                    gdn_cat = ggml_concat(ctx, gdn_cat, gdn_per_seq[s], 1); // [value_dim, T]
                block_out = ggml_mul_mat(ctx, t.ssm_out_w, gdn_cat); // [H, T]
            }

            ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out); // [H, T]

            // ===== FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w); // [H, T]
            ggml_tensor* ffn_out;
            if (d.is_moe == 0)
            {
                const std::int64_t ffDense = d.ff_dense;
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed); // [2*ffDense, T]
                ggml_tensor* g_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, T, gu->nb[1], 0));
                ggml_tensor* u_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, T, gu->nb[1], static_cast<std::size_t>(ffDense) * sizeof(float)));
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_part), u_part); // [ffDense, T]
                ffn_out = ggml_mul_mat(ctx, t.down_w, act); // [H, T]
            }
            else
            {
                // MoE: router -> top-k -> renorm -> stacked experts (mul_mat_id over T tokens) + gated shared expert.
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed); // [num_experts, T]
                ggml_tensor* probs = ggml_soft_max(ctx, router_logits);                   // [num_experts, T]
                ggml_tensor* sel = ggml_top_k(ctx, probs, num_experts_used);              // [num_used, T] I32
                ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, T);
                ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                         // [1, num_used, T]
                ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, num_experts_used, T);
                if (norm_topk != 0)
                {
                    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);                          // [1, T]
                    w_2d = ggml_div(ctx, w_2d, w_sum);
                }
                if (expert_weights_scale != 1.0f)
                    w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, T);

                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, T);
                ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);     // [expert_ff, num_used, T]
                ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);        // [H, num_used, T]
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);                   // [H, num_used, T]
                // sum over num_used (ne1)
                ggml_tensor* moe_out = ggml_cont(ctx, ggml_view_3d(ctx, weighted, H, 1, T, weighted->nb[1], weighted->nb[2], 0));
                for (int u = 1; u < num_experts_used; ++u)
                {
                    ggml_tensor* vu = ggml_view_3d(ctx, weighted, H, 1, T, weighted->nb[1], weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, vu);
                }
                ggml_tensor* moe_out_2d = ggml_reshape_2d(ctx, moe_out, H, T);

                // gated shared expert
                ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed); // [shared_ff, T]
                ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed);
                ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                ggml_tensor* sh_down = ggml_mul_mat(ctx, t.shexp_down_w, sh_act); // [H, T]
                ggml_tensor* sh_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed)); // [1, T]
                ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);
                ffn_out = ggml_add(ctx, moe_out_2d, sh_out);
            }

            hidden = ggml_add(ctx, residual1, ffn_out); // [H, T]
        }

        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_tensor* out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_cpy);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (160 + 32 * n_seqs) + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent == 0)
            {
                ggml_build_forward_expand(graph, lt[l].k_cpy);
                ggml_build_forward_expand(graph, lt[l].v_cpy);
            }
        }
        for (ggml_tensor* w : gdn_state_writes)
        {
            ggml_set_output(w);
            ggml_build_forward_expand(graph, w);
        }
        ggml_build_forward_expand(graph, out_cpy);

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    { if (needs_upload) upload_list.push_back({tgt, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf);
                  if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        const std::size_t poolBytes = kv_cache_bytes(num_kv_heads, total_slots, head_dim, kv_cache_type);
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            if (d.is_moe == 0)
            {
                bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
                bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            }
            else
            {
                bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes), true);
                bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes), true);
                bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes), true);
                bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes), true);
                bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes), true);
                bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes), true);
                bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes), true);
                bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float), true);
            }
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_pool, d.k_cache, poolBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_pool, d.v_cache, poolBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes), true);
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes), true);
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes), true);
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes), true);
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float), true);
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float), true);
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes), true);
                bind_or_mark(t.conv_state_in, d.conv_state_in, convStateBytes, false, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.delta_state_in, d.delta_state_in, deltaStateBytes, false, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
        }

        // Allocate the graph tensors. Persist uses alloc_ctx_tensors (each tensor
        // its own slot = STABLE addresses, required for CUDA-graph capture); non-
        // persist tries gallocr lifetime-packing first.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Qwen3.5 batched decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
            if (vram_log_enabled())
                vram_log("q35-batched-decode-persist", static_cast<std::int64_t>(ggml_backend_buffer_get_size(persist_buf)));
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Qwen3.5 batched decode: failed to allocate backend buffer.");
                return 0;
            }
            if (vram_log_enabled())
                vram_log("q35-batched-decode-ctx", static_cast<std::int64_t>(ggml_backend_buffer_get_size(buffer.value)));
        }

        host_read_barrier();
        for (auto& u : upload_list) ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * T * sizeof(float));
        ggml_backend_tensor_set(pos_t, positions, 0, static_cast<std::size_t>(T) * sizeof(std::int32_t));
        ggml_backend_tensor_set(slot_t, slot_mapping, 0, static_cast<std::size_t>(T) * sizeof(std::int64_t));
        for (int s = 0; s < n_seqs; s++)
        {
            ggml_backend_tensor_set(gidx[s], gather_idx + static_cast<std::size_t>(s) * pad_kv, 0, static_cast<std::size_t>(pad_kv) * sizeof(std::int32_t));
            bfd_upload_mask(mask[s], pad_kv, seq_lens[s]);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 batched decode: graph execution failed.");
            if (persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(H) * T * sizeof(float));
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent != 0)
            {
                finalize_compute_with_download(lt[l].conv_state_in, layers[l].conv_state_out, convStateBytes);
                finalize_compute_with_download(lt[l].delta_state_in, layers[l].delta_state_out, deltaStateBytes);
            }
        }
        host_read_barrier();

        if (persist)
        {
            g_q35bdc.ctx = ctx;
            g_q35bdc.buffer = persist_buf;
            g_q35bdc.graph = graph;
            g_q35bdc.hidden_t = hidden_t;
            g_q35bdc.hidden_out = hidden_out;
            g_q35bdc.pos_t = pos_t;
            g_q35bdc.slot_t = slot_t;
            g_q35bdc.gidx = gidx;
            g_q35bdc.mask = mask;
            g_q35bdc.conv_state.clear(); g_q35bdc.delta_state.clear(); g_q35bdc.gdn_layer.clear();
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent != 0)
                {
                    g_q35bdc.conv_state.push_back(lt[l].conv_state_in);
                    g_q35bdc.delta_state.push_back(lt[l].delta_state_in);
                    g_q35bdc.gdn_layer.push_back(l);
                }
            }
            g_q35bdc.sig = sig;
            g_q35bdc.num_layers = num_layers;
            g_q35bdc.hidden_size = H;
            g_q35bdc.n_seqs = n_seqs;
            g_q35bdc.pad_kv = pad_kv;
            g_q35bdc.conv_bytes = convStateBytes;
            g_q35bdc.delta_bytes = deltaStateBytes;
            g_q35bdc.valid = true;
        }
        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_Qwen35ModelDecodeBatched(
    const TSGgmlQwen35LayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int n_tokens, int n_seqs,
    const int* positions, const std::int64_t* slot_mapping,
    const int* gather_idx, const int* seq_lens, int pad_kv, int total_slots,
    int num_heads, int num_kv_heads, int head_dim,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale)
{
    try
    {
        int r = qwen35_model_decode_batched_impl(
            layers, num_layers, hidden_data, hidden_size, n_tokens, n_seqs,
            positions, slot_mapping, gather_idx, seq_lens, pad_kv, total_slots,
            num_heads, num_kv_heads, head_dim, rope_n_dims, rope_mode, kv_cache_type,
            conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            eps, rope_base, rope_freq_scale,
            num_experts, num_experts_used, expert_ff, shared_ff,
            norm_topk, expert_weights_scale);
        return r;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 batched decode."); return 0; }
}

// Drop the persistent batched-decode graph cache (C# calls this when the device
// KV pools are reallocated or the model state is reset, since the cached graph
// pins those device addresses).
TSG_EXPORT void TSGgml_Qwen35ResetBatchedDecodeCache()
{
    g_q35bdc.reset();
}

