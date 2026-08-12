// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#include "ggml_ops_internal.h"

using namespace tsg;

// Muse-Glimmer's vision tower is intentionally separate from the language-model
// graph in ggml_ops_muse_glimmer.cpp.  A max-size image contains 16,224 ViT
// patches.  Running the block as individual public tensor operations makes every
// projection, RoPE and activation synchronize through the host; the 8,960-wide
// FFN alone then evaluates more than seven billion scalar erf approximations on
// the CPU over 50 blocks.  This entry point keeps one complete block on-device.

namespace
{
    struct MuseGlimmerVisionBlockDesc
    {
        TensorView2DDesc hidden;

        const float* ln1_w;
        const float* ln1_b;
        void* q_w;
        const float* q_b;
        void* k_w;
        const float* k_b;
        void* v_w;
        const float* v_b;
        void* out_w;
        const float* out_b;

        const float* ln2_w;
        const float* ln2_b;
        void* up_w;
        const float* up_b;
        void* down_w;
        const float* down_b;

        const std::int32_t* pos_w;
        const std::int32_t* pos_h;
        const std::int32_t* window_offsets;

        std::int64_t q_ne0;
        std::int64_t q_ne1;
        std::int64_t q_bytes;
        std::int64_t k_ne0;
        std::int64_t k_ne1;
        std::int64_t k_bytes;
        std::int64_t v_ne0;
        std::int64_t v_ne1;
        std::int64_t v_bytes;
        std::int64_t out_ne0;
        std::int64_t out_ne1;
        std::int64_t out_bytes;
        std::int64_t up_ne0;
        std::int64_t up_ne1;
        std::int64_t up_bytes;
        std::int64_t down_ne0;
        std::int64_t down_ne1;
        std::int64_t down_bytes;

        int struct_bytes;
        int hidden_size;
        int intermediate_size;
        int num_tokens;
        int num_heads;
        int head_dim;
        int window_count;
        int is_global;
        int q_type;
        int k_type;
        int v_type;
        int out_type;
        int up_type;
        int down_type;

        float eps;
        float rope_theta;
    };

    struct Upload
    {
        ggml_tensor* tensor;
        const void* data;
        std::size_t bytes;
    };

    bool validate_quant(
        const char* name,
        const void* data,
        int type,
        std::int64_t ne0,
        std::int64_t ne1,
        std::int64_t bytes,
        std::int64_t expected_ne0,
        std::int64_t expected_ne1)
    {
        if (data == nullptr || ne0 != expected_ne0 || ne1 != expected_ne1 || bytes <= 0)
        {
            set_last_error(std::string("Muse-Glimmer vision block: invalid ") + name + " weight descriptor.");
            return false;
        }
        const ggml_type gg_type = static_cast<ggml_type>(type);
        if (gg_type < 0 || gg_type >= GGML_TYPE_COUNT || ggml_type_size(gg_type) == 0)
        {
            set_last_error(std::string("Muse-Glimmer vision block: invalid ") + name + " weight type.");
            return false;
        }
        return true;
    }

    ggml_tensor* repeat_bias(ggml_context* ctx, ggml_tensor* value, ggml_tensor* bias, int dim)
    {
        return ggml_add(ctx, value,
            ggml_repeat(ctx, ggml_reshape_2d(ctx, bias, dim, 1), value));
    }

    ggml_tensor* rope_2d_half(
        ggml_context* ctx,
        ggml_tensor* value,
        ggml_tensor* pos_w,
        ggml_tensor* pos_h,
        int num_tokens,
        int num_heads,
        int head_dim,
        float rope_theta)
    {
        // Identical to llama.cpp clip_graph::build_rope_2d for
        // interleave_freq=false: each 48-channel half is independently rotated
        // in NORMAL (adjacent-pair) order.  Positions are already in the sparse
        // window permutation and are deliberately 1-indexed.
        const int half_dim = head_dim / 2;
        ggml_tensor* value_3d = ggml_reshape_3d(ctx, value, head_dim, num_heads, num_tokens);
        if (value_3d == nullptr)
            return nullptr;

        const std::size_t head_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
        const std::size_t token_bytes = head_bytes * num_heads;
        const std::size_t half_bytes = static_cast<std::size_t>(half_dim) * sizeof(float);

        ggml_tensor* first = ggml_view_3d(ctx, value_3d,
            half_dim, num_heads, num_tokens, head_bytes, token_bytes, 0);
        ggml_tensor* second = ggml_view_3d(ctx, value_3d,
            half_dim, num_heads, num_tokens, head_bytes, token_bytes, half_bytes);
        if (first == nullptr || second == nullptr)
            return nullptr;

        first = ggml_rope_ext(ctx, first, pos_w, nullptr,
            half_dim, GGML_ROPE_TYPE_NORMAL, 0, rope_theta,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        second = ggml_rope_ext(ctx, second, pos_h, nullptr,
            half_dim, GGML_ROPE_TYPE_NORMAL, 0, rope_theta,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return (first == nullptr || second == nullptr)
            ? nullptr
            : ggml_concat(ctx, first, second, 0);
    }

    ggml_tensor* segment_view(
        ggml_context* ctx,
        ggml_tensor* value,
        int head_dim,
        int length,
        int num_heads,
        int offset)
    {
        if (offset == 0 && length == value->ne[1])
            return value;
        return ggml_view_3d(ctx, value,
            head_dim, length, num_heads,
            value->nb[1], value->nb[2],
            static_cast<std::size_t>(offset) * value->nb[1]);
    }

    int muse_glimmer_vision_block_impl(const MuseGlimmerVisionBlockDesc& d)
    {
        if (!ensure_backend())
            return 0;
        if (d.struct_bytes != static_cast<int>(sizeof(MuseGlimmerVisionBlockDesc)))
        {
            set_last_error("Muse-Glimmer vision block: descriptor size mismatch.");
            return 0;
        }
        if (!validate_desc(d.hidden, "hidden"))
            return 0;
        if (d.hidden.dim0 != d.num_tokens || d.hidden.dim1 != d.hidden_size ||
            d.num_tokens <= 0 || d.hidden_size <= 0 || d.intermediate_size <= 0 ||
            d.num_heads <= 0 || d.head_dim <= 0 ||
            d.num_heads * d.head_dim != d.hidden_size || (d.head_dim % 4) != 0 ||
            d.eps <= 0.0f || d.rope_theta <= 0.0f)
        {
            set_last_error("Muse-Glimmer vision block: invalid dimensions or hyperparameters.");
            return 0;
        }
        if (d.ln1_w == nullptr || d.ln1_b == nullptr || d.q_b == nullptr ||
            d.k_b == nullptr || d.v_b == nullptr || d.out_b == nullptr ||
            d.ln2_w == nullptr || d.ln2_b == nullptr || d.up_b == nullptr ||
            d.down_b == nullptr || d.pos_w == nullptr || d.pos_h == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: a required F32 input is null.");
            return 0;
        }
        if (!d.is_global)
        {
            if (d.window_offsets == nullptr || d.window_count <= 0 ||
                d.window_offsets[0] != 0 || d.window_offsets[d.window_count] != d.num_tokens)
            {
                set_last_error("Muse-Glimmer vision block: invalid sparse-window offsets.");
                return 0;
            }
            for (int i = 0; i < d.window_count; ++i)
            {
                if (d.window_offsets[i + 1] <= d.window_offsets[i])
                {
                    set_last_error("Muse-Glimmer vision block: window offsets are not increasing.");
                    return 0;
                }
            }
        }

        const int hidden = d.hidden_size;
        const int dff = d.intermediate_size;
        if (!validate_quant("Q", d.q_w, d.q_type, d.q_ne0, d.q_ne1, d.q_bytes, hidden, hidden) ||
            !validate_quant("K", d.k_w, d.k_type, d.k_ne0, d.k_ne1, d.k_bytes, hidden, hidden) ||
            !validate_quant("V", d.v_w, d.v_type, d.v_ne0, d.v_ne1, d.v_bytes, hidden, hidden) ||
            !validate_quant("output", d.out_w, d.out_type, d.out_ne0, d.out_ne1, d.out_bytes, hidden, hidden) ||
            !validate_quant("FFN up", d.up_w, d.up_type, d.up_ne0, d.up_ne1, d.up_bytes, hidden, dff) ||
            !validate_quant("FFN down", d.down_w, d.down_type, d.down_ne0, d.down_ne1, d.down_bytes, dff, hidden))
        {
            return 0;
        }

        const std::size_t context_bytes = 8 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(context_bytes))
        {
            set_last_error("Muse-Glimmer vision block: failed to create ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;
        std::vector<Upload> uploads;
        uploads.reserve(20);

        auto input_1d = [&](ggml_type type, std::int64_t ne0, const void* data, std::size_t bytes) {
            ggml_tensor* t = ggml_new_tensor_1d(ctx, type, ne0);
            if (t != nullptr)
            {
                ggml_set_input(t);
                uploads.push_back({ t, data, bytes });
            }
            return t;
        };
        auto quant_2d = [&](int type, std::int64_t ne0, std::int64_t ne1,
                            const void* data, std::size_t bytes) {
            ggml_tensor* t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(type), ne0, ne1);
            if (t != nullptr)
            {
                ggml_set_input(t);
                uploads.push_back({ t, data, bytes });
            }
            return t;
        };

        ggml_tensor* hidden_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, d.num_tokens);
        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, d.num_tokens);
        if (hidden_in == nullptr || hidden_out == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: failed to create residual tensors.");
            return 0;
        }
        ggml_set_input(hidden_in);

        const std::size_t hidden_vector_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
        const std::size_t dff_vector_bytes = static_cast<std::size_t>(dff) * sizeof(float);
        ggml_tensor* ln1_w = input_1d(GGML_TYPE_F32, hidden, d.ln1_w, hidden_vector_bytes);
        ggml_tensor* ln1_b = input_1d(GGML_TYPE_F32, hidden, d.ln1_b, hidden_vector_bytes);
        ggml_tensor* q_b = input_1d(GGML_TYPE_F32, hidden, d.q_b, hidden_vector_bytes);
        ggml_tensor* k_b = input_1d(GGML_TYPE_F32, hidden, d.k_b, hidden_vector_bytes);
        ggml_tensor* v_b = input_1d(GGML_TYPE_F32, hidden, d.v_b, hidden_vector_bytes);
        ggml_tensor* out_b = input_1d(GGML_TYPE_F32, hidden, d.out_b, hidden_vector_bytes);
        ggml_tensor* ln2_w = input_1d(GGML_TYPE_F32, hidden, d.ln2_w, hidden_vector_bytes);
        ggml_tensor* ln2_b = input_1d(GGML_TYPE_F32, hidden, d.ln2_b, hidden_vector_bytes);
        ggml_tensor* up_b = input_1d(GGML_TYPE_F32, dff, d.up_b, dff_vector_bytes);
        ggml_tensor* down_b = input_1d(GGML_TYPE_F32, hidden, d.down_b, hidden_vector_bytes);
        ggml_tensor* pos_w = input_1d(GGML_TYPE_I32, d.num_tokens, d.pos_w,
            static_cast<std::size_t>(d.num_tokens) * sizeof(std::int32_t));
        ggml_tensor* pos_h = input_1d(GGML_TYPE_I32, d.num_tokens, d.pos_h,
            static_cast<std::size_t>(d.num_tokens) * sizeof(std::int32_t));

        ggml_tensor* q_w = quant_2d(d.q_type, d.q_ne0, d.q_ne1, d.q_w, static_cast<std::size_t>(d.q_bytes));
        ggml_tensor* k_w = quant_2d(d.k_type, d.k_ne0, d.k_ne1, d.k_w, static_cast<std::size_t>(d.k_bytes));
        ggml_tensor* v_w = quant_2d(d.v_type, d.v_ne0, d.v_ne1, d.v_w, static_cast<std::size_t>(d.v_bytes));
        ggml_tensor* out_w = quant_2d(d.out_type, d.out_ne0, d.out_ne1, d.out_w, static_cast<std::size_t>(d.out_bytes));
        ggml_tensor* up_w = quant_2d(d.up_type, d.up_ne0, d.up_ne1, d.up_w, static_cast<std::size_t>(d.up_bytes));
        ggml_tensor* down_w = quant_2d(d.down_type, d.down_ne0, d.down_ne1, d.down_w, static_cast<std::size_t>(d.down_bytes));

        if (ln1_w == nullptr || ln1_b == nullptr || q_b == nullptr || k_b == nullptr ||
            v_b == nullptr || out_b == nullptr || ln2_w == nullptr || ln2_b == nullptr ||
            up_b == nullptr || down_b == nullptr || pos_w == nullptr || pos_h == nullptr ||
            q_w == nullptr || k_w == nullptr || v_w == nullptr || out_w == nullptr ||
            up_w == nullptr || down_w == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: failed to create input tensors.");
            return 0;
        }

        // Pre-LN attention with separate quantized Q/K/V projections.
        ggml_tensor* ln1 = ggml_norm(ctx, hidden_in, d.eps);
        ln1 = ggml_mul(ctx, ln1, ln1_w);
        ln1 = ggml_add(ctx, ln1, ln1_b);
        ggml_tensor* q = repeat_bias(ctx, ggml_mul_mat(ctx, q_w, ln1), q_b, hidden);
        ggml_tensor* k = repeat_bias(ctx, ggml_mul_mat(ctx, k_w, ln1), k_b, hidden);
        ggml_tensor* v = repeat_bias(ctx, ggml_mul_mat(ctx, v_w, ln1), v_b, hidden);
        if (q == nullptr || k == nullptr || v == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: failed to build Q/K/V projections.");
            return 0;
        }

        q = rope_2d_half(ctx, q, pos_w, pos_h, d.num_tokens, d.num_heads, d.head_dim, d.rope_theta);
        k = rope_2d_half(ctx, k, pos_w, pos_h, d.num_tokens, d.num_heads, d.head_dim, d.rope_theta);
        if (q == nullptr || k == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: failed to build 2D RoPE.");
            return 0;
        }

        // Keep the caller's true packed windows.  No dense block-diagonal mask is
        // constructed: a sparse layer is a collection of exact independent flash
        // attention calls within this one graph.  K/V are cast once to the layout
        // ggml-cuda's optimized head-dim-96 kernel expects.
        ggml_tensor* q_all = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor* k_all = ggml_cast(ctx,
            ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)), GGML_TYPE_F16);
        ggml_tensor* v_3d = ggml_reshape_3d(ctx, v, d.head_dim, d.num_heads, d.num_tokens);
        ggml_tensor* v_all = ggml_cast(ctx,
            ggml_cont(ctx, ggml_permute(ctx, v_3d, 0, 2, 1, 3)), GGML_TYPE_F16);
        if (q_all == nullptr || k_all == nullptr || v_all == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: failed to build attention layouts.");
            return 0;
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(d.head_dim));
        ggml_tensor* attention = nullptr;
        const int segment_count = d.is_global ? 1 : d.window_count;
        for (int segment = 0; segment < segment_count; ++segment)
        {
            const int start = d.is_global ? 0 : d.window_offsets[segment];
            const int end = d.is_global ? d.num_tokens : d.window_offsets[segment + 1];
            const int length = end - start;
            ggml_tensor* q_segment = segment_view(ctx, q_all, d.head_dim, length, d.num_heads, start);
            ggml_tensor* k_segment = segment_view(ctx, k_all, d.head_dim, length, d.num_heads, start);
            ggml_tensor* v_segment = segment_view(ctx, v_all, d.head_dim, length, d.num_heads, start);
            if (q_segment == nullptr || k_segment == nullptr || v_segment == nullptr)
            {
                set_last_error("Muse-Glimmer vision block: failed to create a window view.");
                return 0;
            }

            ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_segment, k_segment, v_segment,
                nullptr, scale, 0.0f, 0.0f);
            if (fa == nullptr)
            {
                set_last_error("Muse-Glimmer vision block: failed to create flash attention.");
                return 0;
            }
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            if (!backend_supports_op(fa))
            {
                // Do not silently build the old O(N^2) fallback here.  Returning
                // false lets the managed bounded-query fallback take over on an
                // unsupported backend without ever attempting a 32 GiB buffer.
                set_last_error("Muse-Glimmer vision block: backend does not support flash attention for this geometry.");
                return 0;
            }
            attention = attention == nullptr ? fa : ggml_concat(ctx, attention, fa, 2);
        }

        ggml_tensor* attention_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, attention), hidden, d.num_tokens);
        ggml_tensor* projected = repeat_bias(ctx, ggml_mul_mat(ctx, out_w, attention_flat), out_b, hidden);
        ggml_tensor* residual1 = ggml_add(ctx, hidden_in, projected);

        // Exact reference MLP.  erf-GELU is deliberately in-place so the 581 MiB
        // [tokens, 8960] activation does not require a second equally large slot.
        ggml_tensor* ln2 = ggml_norm(ctx, residual1, d.eps);
        ln2 = ggml_mul(ctx, ln2, ln2_w);
        ln2 = ggml_add(ctx, ln2, ln2_b);
        ggml_tensor* fc1 = repeat_bias(ctx, ggml_mul_mat(ctx, up_w, ln2), up_b, dff);
        ggml_tensor* activated = ggml_gelu_erf_inplace(ctx, fc1);
        ggml_tensor* fc2 = repeat_bias(ctx, ggml_mul_mat(ctx, down_w, activated), down_b, hidden);
        ggml_tensor* residual2 = ggml_add(ctx, residual1, fc2);
        ggml_tensor* output = ggml_cpy(ctx, residual2, hidden_out);
        if (output == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: failed to build output node.");
            return 0;
        }
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx,
            static_cast<std::size_t>(1024 + segment_count * 16), false);
        if (graph == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: failed to create graph.");
            return 0;
        }
        ggml_build_forward_expand(graph, output);

        // A lifetime-aware allocator is essential here.  Static context
        // allocation would add every intermediate—including the large FFN
        // activation—although their lifetimes do not overlap.  The allocator is
        // local so its scratch buffer is returned before language-model decode.
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, graph))
        {
            if (galloc != nullptr)
                ggml_gallocr_free(galloc);
            set_last_error("Muse-Glimmer vision block: failed to allocate bounded graph workspace.");
            return 0;
        }

        host_read_barrier();
        ggml_backend_tensor_set(hidden_in, d.hidden.data, 0, logical_bytes(d.hidden));
        for (const Upload& upload : uploads)
            ggml_backend_tensor_set(upload.tensor, resolve_upload_source(upload.data), 0, upload.bytes);

        const ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            ggml_gallocr_free(galloc);
            set_last_error("Muse-Glimmer vision block: graph execution failed.");
            return 0;
        }
        finalize_compute_with_download(hidden_out, d.hidden.data, logical_bytes(d.hidden));
        host_read_barrier();
        ggml_gallocr_free(galloc);

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_MuseGlimmerVisionBlockQuantF32(
    const MuseGlimmerVisionBlockDesc* desc)
{
    try
    {
        if (desc == nullptr)
        {
            set_last_error("Muse-Glimmer vision block: descriptor is null.");
            return 0;
        }
        return muse_glimmer_vision_block_impl(*desc);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown Muse-Glimmer vision block failure.");
        return 0;
    }
}
