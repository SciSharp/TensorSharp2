// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#pragma once

#include "ggml_ops_internal.h"

#ifdef TSG_GGML_USE_CUDA
// On-device causal-mask fill (ggml_ops_mask.cu): generate the verify kernel's
// [kvLen, N] F16 causal(+windowed) masks straight into their device buffers,
// eliminating the host fill + H2D upload. Bit-identical to the host path.
extern "C" bool tsg_cuda_fill_causal_mask_f16(
    void* mask_dev, int kvLen, int N, int nPast, int window, int validLen);
extern "C" bool tsg_cuda_sync_stream0(void);
#endif

// ============================================================================
// Shared pieces of the fused transformer kernels, used by the per-model
// translation units (ggml_ops_transformer.cpp, ggml_ops_qwen35_*.cpp,
// ggml_ops_gemma4_*.cpp, ggml_ops_transformer_prefill.cpp): KV-cache sizing
// and circular-window views, flash-attention padding/mask helpers, async
// input upload, and the C# layer-descriptor structs. Header-only (inline)
// so every TU compiles the exact definitions it had when these lived in a
// single file.
// ============================================================================
namespace tsg
{
    // KV-cache element size in bytes for the given GGML tensor type.
    // F32 = 4, F16 = 2. For block-quantized types (Q8_0) the *element size* is
    // fractional (1.0625 bytes for Q8_0) so callers that need a per-element byte
    // count should NOT use this helper - they should use ggml's per-row stride
    // (`tensor->nb[1]`) directly, which already accounts for block padding.
    // We still expose this helper for the linear types because some byte-offset
    // arithmetic happens before the cache tensor is materialised.
    inline std::size_t kv_cache_elem_size(int kv_cache_type)
    {
        switch (static_cast<ggml_type>(kv_cache_type))
        {
            case GGML_TYPE_F32:  return 4;
            case GGML_TYPE_F16:  return 2;
            // Block-quantized: callers should use nb[1] / row_size instead.
            // Returning 0 here makes any accidental `head_dim * elem_size` use
            // visibly wrong rather than silently miscounting.
            case GGML_TYPE_Q8_0: return 0;
            case GGML_TYPE_Q4_0: return 0;
            default:             return 4;
        }
    }

    inline bool kv_cache_is_block_quantized(int kv_cache_type)
    {
        const ggml_type t = static_cast<ggml_type>(kv_cache_type);
        return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
    }

    // Bytes occupied by a [kv_heads, cache_size, head_dim] cache tensor of the
    // given GGML type. Uses ggml_row_size so block-quantized layouts (Q8_0) are
    // accounted for correctly: a Q8_0 row of 256 elements occupies 8 blocks * 34
    // bytes = 272 bytes (vs. 256 raw bytes if we used a fractional 1.0625 value).
    inline std::size_t kv_cache_bytes(int kv_heads, int cache_size, int head_dim, int kv_cache_type = GGML_TYPE_F32)
    {
        const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(kv_cache_type), head_dim);
        return static_cast<std::size_t>(kv_heads) *
               static_cast<std::size_t>(cache_size) *
               row_bytes;
    }

    inline constexpr int kFlashAttnKvStride = 256;

    inline bool flash_attn_requires_masked_padding(int head_dim)
    {
        // The custom CUDA kernels added for 512/576-dim attention only support
        // the grouped-query path, which expects a non-null mask and a KV length
        // aligned to FATTN_KQ_STRIDE.
        return head_dim == 512 || head_dim == 576;
    }

    inline int flash_attn_kv_length(int valid_len, int cache_size, int head_dim)
    {
        if (!flash_attn_requires_masked_padding(head_dim))
            return valid_len;

        const int padded = ((valid_len + kFlashAttnKvStride - 1) / kFlashAttnKvStride) * kFlashAttnKvStride;
        return std::min(cache_size, std::max(valid_len, padded));
    }

    inline void fill_flash_attn_mask(std::vector<ggml_fp16_t>& mask, int padded_len, int valid_len)
    {
        mask.assign(static_cast<std::size_t>(padded_len), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        const int unclamped_valid = std::max(valid_len, 0);
        const int clamped_valid = std::min(unclamped_valid, padded_len);
        std::fill_n(mask.begin(), clamped_valid, static_cast<ggml_fp16_t>(0));
    }

    // Upload one captured-decode INPUT tensor without a per-copy stream sync.
    // On CUDA the copy is queued on the backend stream (ordered ahead of the graph
    // replay that reads it; the post-compute download syncs the stream), so we drop
    // the redundant cudaStreamSynchronize that the synchronous setter issues per
    // call — meaningful when a decode token refreshes ~2*num_layers small inputs.
    // CUDA pageable host->device async is host-synchronous w.r.t. the source, so a
    // caller's transient buffer (e.g. a per-layer mask vector) is safe to free right
    // after this returns. Non-CUDA backends fall back to the synchronous setter.
    inline void decode_input_set_async(ggml_tensor* tensor, const void* data, std::size_t bytes)
    {
        if (tensor == nullptr || data == nullptr || bytes == 0)
            return;
        if (g_backend_type == BACKEND_TYPE_CUDA)
            ggml_backend_tensor_set_async(g_backend, tensor, data, 0, bytes);
        else
            ggml_backend_tensor_set(tensor, data, 0, bytes);
    }

    inline ggml_tensor* view_kv_cache_window(
        ggml_context* ctx,
        ggml_tensor* cache,
        int head_dim,
        int cache_size,
        int kv_heads,
        int start_idx,
        int length,
        int kv_cache_type = GGML_TYPE_F32)
    {
        if (ctx == nullptr || cache == nullptr || head_dim <= 0 || cache_size <= 0 || kv_heads <= 0 || length <= 0)
            return nullptr;

        start_idx %= cache_size;
        if (start_idx < 0)
            start_idx += cache_size;

        // ggml_row_size handles block-quantized types (Q8_0) correctly: a row of
        // 256 Q8_0 elements is 8 blocks * 34 bytes = 272 bytes, not 256/1.0625.
        // For linear types it reduces to head_dim * sizeof(elem).
        const std::size_t nb1 = ggml_row_size(static_cast<ggml_type>(kv_cache_type), head_dim);
        const std::size_t nb2 = static_cast<std::size_t>(cache_size) * nb1;

        if (start_idx + length <= cache_size)
        {
            return ggml_view_3d(
                ctx,
                cache,
                head_dim,
                length,
                kv_heads,
                nb1,
                nb2,
                static_cast<std::size_t>(start_idx) * nb1);
        }

        const int tail_length = cache_size - start_idx;
        const int head_length = length - tail_length;
        ggml_tensor* tail = ggml_view_3d(
            ctx,
            cache,
            head_dim,
            tail_length,
            kv_heads,
            nb1,
            nb2,
            static_cast<std::size_t>(start_idx) * nb1);
        ggml_tensor* head = ggml_view_3d(ctx, cache, head_dim, head_length, kv_heads, nb1, nb2, 0);
        if (tail == nullptr || head == nullptr)
            return nullptr;

        // GPU concat kernels only implement F32 inputs. Wrapped circular windows may
        // come from F16/Q8_0 KV caches, so materialize both slices as F32 first.
        if (static_cast<ggml_type>(kv_cache_type) != GGML_TYPE_F32)
        {
            ggml_tensor* tail_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, tail_length, kv_heads);
            ggml_tensor* head_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, head_length, kv_heads);
            if (tail_f32 == nullptr || head_f32 == nullptr)
                return nullptr;

            tail = ggml_cpy(ctx, tail, tail_f32);
            head = ggml_cpy(ctx, head, head_f32);
        }

        return ggml_concat(ctx, tail, head, 1);
    }
}

// Per-layer descriptor for the Qwen3.5/3.6 full-model kernels (decode,
// verify, batched decode). Passed by pointer from C#. Layout MUST match the
// mirror struct in GgmlNative.cs: pointers first, then int64 shapes, then
// int32 scalars, so natural alignment is identical on both sides.
struct TSGgmlQwen35LayerDesc
{
    // --- pointers (host memory); 8-byte each, FIRST per interop convention ---
    void* attn_norm_w;       // [hidden] F32 (input norm, both layer kinds)
    void* post_attn_norm_w;  // [hidden] F32 (FFN input norm)
    // attention layer
    void* qkv_w;             // attn_qkv [hidden, 2*qDim + 2*kDim]
    void* q_norm_w;          // [head_dim] F32
    void* k_norm_w;          // [head_dim] F32
    void* o_w;               // attn_output [qDim, hidden]
    void* k_cache;           // device-resident [kv_heads, cache, head_dim]
    void* v_cache;
    // gated-delta-net layer
    void* gdn_qkv_w;         // attn_qkv (recurrent) [hidden, conv_dim]
    void* gdn_gate_w;        // attn_gate / z [hidden, value_dim]
    void* ssm_beta_w;        // [hidden, num_v_heads]
    void* ssm_alpha_w;       // [hidden, num_v_heads]
    void* conv1d_w;          // [conv_kernel, conv_dim] F32
    void* ssm_dt_w;          // [num_v_heads] F32 (dt bias)
    void* ssm_a_w;           // [num_v_heads] F32 (-exp(A_log))
    void* ssm_norm_w;        // [head_v_dim] F32
    void* ssm_out_w;         // [value_dim, hidden]
    void* conv_state_in;     // host [conv_kernel-1, conv_dim] ggml layout (ne0=time)
    void* delta_state_in;    // host [head_k_dim, head_v_dim, num_v_heads]
    void* conv_state_out;    // host, same layout as conv_state_in
    void* delta_state_out;   // host, same layout as delta_state_in
    // dense FFN
    void* gu_w;              // ffn_gate_up [hidden, 2*ff_dense]
    void* down_w;            // ffn_down [ff_dense, hidden]
    // separate attention K/V (when separate_qkv: qkv_w holds Q+gate [hidden, 2*qDim])
    void* k_w;
    void* v_w;
    // MoE FFN (used when is_moe != 0)
    void* gate_inp_w;        // router [hidden, num_experts]
    void* gate_exps;         // stacked [hidden, expert_ff, num_experts]
    void* up_exps;           // stacked [hidden, expert_ff, num_experts]
    void* down_exps;         // stacked [expert_ff, hidden, num_experts]
    void* shexp_gate_w;      // [hidden, shared_ff]
    void* shexp_up_w;        // [hidden, shared_ff]
    void* shexp_down_w;      // [shared_ff, hidden]
    void* shexp_gate_inp_w;  // [hidden] F32 (shared-expert sigmoid gate)

    // --- int64 weight shapes/bytes ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t gdn_qkv_ne0, gdn_qkv_ne1, gdn_qkv_bytes;
    std::int64_t gdn_gate_ne0, gdn_gate_ne1, gdn_gate_bytes;
    std::int64_t ssm_beta_ne0, ssm_beta_ne1, ssm_beta_bytes;
    std::int64_t ssm_alpha_ne0, ssm_alpha_ne1, ssm_alpha_bytes;
    std::int64_t ssm_out_ne0, ssm_out_ne1, ssm_out_bytes;
    std::int64_t gu_ne0, gu_ne1, gu_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;
    std::int64_t gate_inp_ne0, gate_inp_ne1, gate_inp_bytes;
    std::int64_t gate_exps_bytes, up_exps_bytes, down_exps_bytes;
    std::int64_t shexp_gate_ne0, shexp_gate_ne1, shexp_gate_bytes;
    std::int64_t shexp_up_ne0, shexp_up_ne1, shexp_up_bytes;
    std::int64_t shexp_down_ne0, shexp_down_ne1, shexp_down_bytes;

    // --- int32 scalars ---
    std::int32_t struct_bytes;
    std::int32_t is_recurrent;
    std::int32_t is_moe;
    std::int32_t qkv_type, o_type;
    std::int32_t gdn_qkv_type, gdn_gate_type, ssm_beta_type, ssm_alpha_type, ssm_out_type;
    std::int32_t gu_type, down_type;
    std::int32_t ff_dense;
    std::int32_t separate_qkv, k_type, v_type;
    std::int32_t gate_inp_type, gate_exps_type, up_exps_type, down_exps_type;
    std::int32_t shexp_gate_type, shexp_up_type, shexp_down_type;
    // MoE CPU offload (--n-cpu-moe): non-zero keeps this layer's routed experts
    // in system RAM. The whole-model decode graph then omits their mul_mat_id
    // chain, pauses after the router, and lets the host multiply them.
    std::int32_t cpu_moe;
};

// Per-layer descriptor for the GPT-OSS whole-model decode kernel
// (TSGgml_GptOssModelDecode). Passed by pointer from C#; layout MUST match
// GptOssLayerDecodeArgs in GgmlNative.cs — pointers first, then int64, then
// int32, then float, so natural alignment is identical on both sides.
struct TSGgmlGptOssLayerDesc
{
    // --- pointers (host memory) ---
    void* attn_norm_w;       // [hidden] F32
    void* qkv_w;             // fused QKV [hidden, qDim+2*kDim], or Q-only when separate_qkv
    void* qkv_b;             // F32 [qDim+2*kDim] (or [qDim]); may be null
    void* k_w;               // separate K weight (null unless separate_qkv)
    void* k_b;               // F32 [kDim]; may be null
    void* v_w;
    void* v_b;
    void* o_w;               // attn_output [qDim, hidden]
    void* o_b;               // F32 [hidden]; may be null
    void* k_cache;           // HOST cache [kv_heads, cache_size, head_dim] — identifies the device window
    void* v_cache;
    void* sinks;             // F32 [num_heads] attention sinks; may be null
    void* post_attn_norm_w;  // [hidden] F32 (MoE input norm)
    void* gate_inp_w;        // router [hidden, num_experts] F32
    void* gate_inp_b;        // [num_experts] F32; may be null
    void* gate_exps;         // stacked [hidden, expert_ff, num_experts]
    void* gate_exps_b;       // [expert_ff, num_experts] F32; may be null
    void* up_exps;           // stacked [hidden, expert_ff, num_experts]
    void* up_exps_b;         // [expert_ff, num_experts] F32; may be null
    void* down_exps;         // stacked [expert_ff, hidden, num_experts]
    void* down_exps_b;       // [hidden, num_experts] F32; may be null

    // --- int64 weight shapes (per-expert ne0/ne1 + TOTAL bytes for stacked) ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t ge_ne0, ge_ne1, ge_bytes;
    std::int64_t ue_ne0, ue_ne1, ue_bytes;
    std::int64_t de_ne0, de_ne1, de_bytes;

    // --- int32 scalars ---
    std::int32_t struct_bytes;       // sizeof sanity check
    std::int32_t hidden_size;
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t cache_size;         // rows in the HOST cache
    std::int32_t is_swa;             // non-zero: sliding-window layer
    std::int32_t sliding_window;
    std::int32_t rope_n_dims;
    std::int32_t orig_ctx_len;       // RoPE yarn original context length
    std::int32_t kv_cache_type;      // GGML_TYPE_F32 / GGML_TYPE_F16
    std::int32_t num_experts;
    std::int32_t num_experts_used;
    std::int32_t separate_qkv;
    std::int32_t qkv_type, k_type, v_type, o_type;
    std::int32_t ge_type, ue_type, de_type;
    // MoE CPU offload (--n-cpu-moe): non-zero keeps this layer's routed experts
    // in system RAM. The whole-model decode graph then omits their mul_mat_id
    // chain, pauses after the router, and lets the host multiply them.
    std::int32_t cpu_moe;

    // --- float scalars ---
    float eps;
    float rope_base;
    float rope_freq_scale;
    float oai_alpha;
    float oai_limit;
};

// MoE layer descriptor for the Gemma 4 MoE kernels (layer/model decode,
// verify, batched decode).
// Descriptor passed by pointer from C#. Layout MUST match
// Gemma4MoELayerDecodeDesc in GgmlNative.cs. 8-byte fields (pointers + int64)
// are grouped first, then 4-byte (int32 + float), so natural alignment is
// identical on both sides with no implicit padding surprises.
struct TSGgmlGemma4MoELayerDesc
{
    // --- pointers (host memory) ---
    void* hidden;            // [hidden_size] F32, in/out (residual stream)
    void* attn_norm_w;       // [hidden_size] F32
    void* qkv_w;             // fused QKV, or Q-only when separate_qkv
    void* k_w;               // separate K weight (null unless separate_qkv)
    void* v_w;               // separate V weight (null unless separate_qkv)
    void* q_norm_w;          // [head_dim] F32
    void* k_norm_w;          // [head_dim] F32 (null for shared layers)
    void* o_w;               // attn_output weight
    void* post_attn_norm_w;  // [hidden_size] F32
    void* k_cache;           // [kv_heads, cache_size, head_dim] (donor's for shared)
    void* v_cache;
    void* freq_factors;      // [freq_factors_len] F32 (null for local/no-scaling)
    void* ffn_norm_w;        // [hidden_size] F32
    void* gu_w;              // dense fused gate_up weight [hidden, 2*ff_dense]
    void* down_w;            // dense down weight [ff_dense, hidden]
    void* post_ffw_norm_1_w; // [hidden_size] F32
    void* gate_inp_w;        // router [hidden, num_experts] F32
    void* gate_inp_scale;    // [hidden] F32 (null if absent)
    void* pre_ffw_norm_2_w;  // [hidden_size] F32 (expert input norm)
    void* gate_up_exps;      // stacked experts [hidden, 2*ff_moe, num_experts]
    void* down_exps;         // stacked experts [ff_moe, hidden, num_experts]
    void* down_exps_scale;   // [num_experts] F32 (null if absent)
    void* post_ffw_norm_2_w; // [hidden_size] F32
    void* post_ffw_norm_w;   // [hidden_size] F32

    // --- int64 weight shapes ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t gu_ne0, gu_ne1, gu_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;
    std::int64_t gue_ne0, gue_ne1, gue_bytes; // per-expert ne0/ne1 + TOTAL bytes
    std::int64_t de_ne0, de_ne1, de_bytes;

    // --- int32 scalars / shapes ---
    std::int32_t struct_bytes;       // sizeof sanity check
    std::int32_t hidden_size;
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t cache_size;
    std::int32_t is_local;
    std::int32_t is_shared;
    std::int32_t sliding_window;
    std::int32_t position;
    std::int32_t rope_n_dims;
    std::int32_t kv_cache_type;
    std::int32_t num_experts;
    std::int32_t num_experts_used;
    std::int32_t freq_factors_len;
    std::int32_t qkv_type;
    std::int32_t k_type;
    std::int32_t v_type;
    std::int32_t o_type;
    std::int32_t gu_type;
    std::int32_t down_type;
    std::int32_t gue_type;
    std::int32_t de_type;
    std::int32_t separate_qkv;
    // MoE CPU offload (--n-cpu-moe): non-zero keeps this layer's routed experts
    // in system RAM. The whole-model decode graph then omits their mul_mat_id
    // chain, pauses after the router, and lets the host multiply them.
    std::int32_t cpu_moe;

    // --- float scalars ---
    float eps;
    float rope_base;
    float inv_sqrt_hidden;     // 1/sqrt(hidden_size) for the router
    float layer_output_scale;
};
