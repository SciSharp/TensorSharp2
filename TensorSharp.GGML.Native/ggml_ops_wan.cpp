// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Wan 2.1 text-to-video native kernels. Three whole-graph entry points drive
// the WanVideo pipeline (TensorSharp.Models/Models/WanVideo):
//
//   TSGgml_WanT5Encode   – the UMT5-XXL text encoder (24 layers, per-layer
//                          relative-attention bias) as ONE graph: token ids in,
//                          final hidden states out. Weights resident-cached by
//                          their GGUF mmap pointer.
//   TSGgml_WanDitForward – one denoising-step velocity prediction of the Wan
//                          DiT (patch embedding, time/text conditioning, N
//                          single-stream blocks with self-attn + cross-attn,
//                          head) as ONE resident-weight graph. On CUDA the
//                          graph is kept persistent per shape so ggml-cuda's
//                          CUDA-graph capture engages (same design as
//                          TSGgml_QwenImageForward).
//   TSGgml_WanVaeDecode  – the Wan causal 3D video VAE decoder: latent
//                          [w,h,t,zc] -> pixels as one graph that iterates the
//                          temporal chunks internally with the causal feature
//                          cache carried between chunks (port of
//                          stable-diffusion.cpp wan_vae.hpp). version 2 adds
//                          the Wan 2.2 TI2V decoder (48-channel latent, DupUp3D
//                          residual shortcuts, 2x2 pixel unpatchify).
//   TSGgml_WanVaeEncode  – the Wan causal 3D video VAE encoder (both the
//                          Wan 2.1 16-channel and Wan 2.2 48-channel variants):
//                          pixels in, posterior mean out, chunked 1+4k frames
//                          with the causal feature caches inside one graph.
//                          Image-to-video conditioning is built on this.
//
// The graph topology mirrors stable-diffusion.cpp's verified Wan implementation
// (src/model/diffusion/wan.hpp, src/model/vae/wan_vae.hpp, src/model/te/t5.hpp);
// the managed side supplies pre-flattened weights where that removes graph work
// (patch embedding as a matmul, causal conv3d as per-temporal-tap 2D kernels).
// ============================================================================
#include "ggml_ops_internal.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tsg;

extern "C" {

// One (possibly quantized) weight matrix + optional F32 bias for the Wan kernels.
// MUST match managed WanW (TensorSharp.Backends.GGML/GgmlNative.cs).
struct TSGWanW
{
    void* w;                    // weight data (GGUF mmap or stable dequant buffer)
    std::int32_t type;          // ggml_type
    std::int32_t reserved;
    std::int64_t ne0, ne1;      // ggml dims: ne0 = input dim, ne1 = output dim
    std::int64_t bytes;
    void* b;                    // [ne1] F32 bias or null
};

// ---- UMT5-XXL encoder ------------------------------------------------------

struct TSGWanT5LayerW
{
    TSGWanW q, k, v, o;         // attention projections (no bias)
    TSGWanW gate, up, down;     // wi_0 / wi_1 / wo (gated GELU FFN)
    void* attn_norm;            // [dim] F32 RMS gain
    void* ffn_norm;             // [dim] F32 RMS gain
    void* rel_b;                // [heads, 32] F32 relative-attention bias table
};

struct TSGgmlWanT5Desc
{
    const std::int32_t* tokens;       // [n_tokens]
    const std::int32_t* rel_bucket;   // [n_tokens * n_tokens], bucket[q*n + k]
    const float* attn_mask;           // [n_tokens] additive (0 / -inf) or null
    TSGWanW tok_embd;                 // [dim, vocab]
    const TSGWanT5LayerW* layers;
    void* final_norm;                 // [dim] F32 RMS gain
    float* out;                       // [dim * n_tokens] written
    std::int32_t n_tokens, num_layers, dim, ff, heads, head_dim;
    float eps;
    std::int32_t struct_bytes;
};

// ---- Wan DiT ---------------------------------------------------------------

struct TSGWanDitBlockW
{
    void* modulation;           // [6*dim] F32 (per-block learned AdaLN bias)
    TSGWanW sq, sk, sv, so;     // self-attention (+F32 bias each)
    void* s_norm_q;             // [dim] F32 full-dim RMS gain
    void* s_norm_k;             // [dim] F32
    void* norm3_w;              // [dim] F32 (cross-attn LayerNorm affine)
    void* norm3_b;              // [dim] F32
    TSGWanW xq, xk, xv, xo;     // cross-attention (+bias)
    void* x_norm_q;             // [dim] F32
    void* x_norm_k;             // [dim] F32
    TSGWanW ffn0, ffn2;         // FFN (+bias)
};

struct TSGgmlWanDitDesc
{
    float* x;                   // [in_dim*pt*ph*pw, seq] patchified latent tokens (in)
    float* out;                 // [out_dim*pt*ph*pw, seq] velocity tokens (out)
    float* context;             // [text_dim, ctx_len] UMT5 states (zero-padded)
    float* tsin;                // [freq_dim] sinusoidal timestep embedding
    float* tsin0;               // [freq_dim] sinusoid for tokens [0, seq0) or null
    float* cosf;                // [head_dim, seq] RoPE cos (pair-duplicated)
    float* sinf;                // [head_dim, seq] RoPE sin
    TSGWanW patch;              // [in_dim*pt*ph*pw, dim] F32 (pre-flattened conv) + bias
    TSGWanW text0, text2;       // text_embedding.0 / .2 (+bias)
    TSGWanW time0, time2;       // time_embedding.0 / .2 (+bias)
    TSGWanW tproj;              // time_projection.1 (+bias) -> [6*dim]
    TSGWanW head;               // head.head (+bias) -> [out_dim*pt*ph*pw]
    void* head_mod;             // [2*dim] F32 head modulation
    const TSGWanDitBlockW* blocks;
    // seq0 > 0 splits the AdaLN modulation into two token segments: tokens
    // [0, seq0) are modulated with tsin0's timestep, the rest with tsin's
    // (Wan 2.2 TI2V image-to-video: the first latent frame is the clean image
    // conditioned at timestep 0). 0 = uniform timestep over all tokens.
    std::int32_t num_layers, dim, ff, heads, head_dim, seq, ctx_len, freq_dim, text_dim, seq0;
    float eps;
    std::int32_t struct_bytes;
};

// ---- Wan VAE decoder -------------------------------------------------------

// One causal conv3d, decomposed by the managed side into per-temporal-tap 2D
// kernels: tap[j] is the [k, k, ic, oc] F32 kernel for temporal offset j
// (0 = oldest frame). kd == 1 uses tap[0] only.
struct TSGWanVaeConv
{
    void* tap0; void* tap1; void* tap2;
    void* bias;                 // [oc] F32 or null
    std::int32_t kd, k, ic, oc;
};

struct TSGWanVaeNorm { void* gamma; std::int32_t c; std::int32_t pad; };

struct TSGWanVaeResBlockW
{
    TSGWanVaeNorm n0, n3;
    TSGWanVaeConv c2, c6;       // 3x3x3 causal convs
    TSGWanVaeConv shortcut;     // 1x1x1 (tap0 == null when in == out: identity)
};

struct TSGWanVaeAttnW
{
    TSGWanVaeNorm norm;
    void* qkv_w;                // [c, 3c] F32 matmul weight (1x1 conv)
    void* qkv_b;                // [3c] F32
    void* proj_w;               // [c, c] F32
    void* proj_b;               // [c] F32
    std::int32_t c, pad;
};

struct TSGWanVaeUpsampleW
{
    TSGWanVaeConv time_conv;    // (3,1,1) dim -> 2*dim; tap0 == null => spatial-only
    TSGWanVaeConv sconv;        // 3x3 2D conv after nearest x2 (kd == 1)
};

struct TSGgmlWanVaeDecodeDesc
{
    float* z;                   // [w, h, t, zc] latent (already de-normalized)
    float* out;                 // [w*8, h*8, 1+(t-1)*4, 3] written
    std::int64_t out_len;       // expected element count of out
    TSGWanVaeConv conv2;        // post-quant 1x1x1 (zc -> zc)
    TSGWanVaeConv conv1;        // 3x3x3 (zc -> 384 / 1024)
    TSGWanVaeResBlockW mid0, mid2;
    TSGWanVaeAttnW mid1;
    TSGWanVaeResBlockW res[12]; // 4 scales x 3 residual blocks
    TSGWanVaeUpsampleW up[3];   // after scales 0,1 (temporal+spatial) and 2 (spatial)
    TSGWanVaeNorm head_norm;
    TSGWanVaeConv head_conv;    // 3x3x3 (96 -> 3, or 256 -> 12 for wan2.2)
    std::int32_t zw, zh, zt, zc;
    std::int32_t version;       // 1 (or 0) = wan2.1; 2 = wan2.2 TI2V (DupUp shortcuts)
    std::int32_t patch;         // pixel unpatchify factor (2 for wan2.2, else 1)
    std::int32_t struct_bytes;
};

// One encoder Resample stage: right/bottom-padded 3x3 stride-2 spatial conv,
// plus (downsample3d only) a stride-2 temporal conv with a 1-frame causal cache.
struct TSGWanVaeDownW
{
    TSGWanVaeConv sconv;        // 3x3 stride-2 2D conv (kd == 1)
    TSGWanVaeConv tconv;        // (3,1,1) stride-2 temporal conv; tap0 == null => downsample2d
};

struct TSGgmlWanVaeEncodeDesc
{
    float* x;                   // [px_w, px_h, px_c, px_t] pixels in [-1,1] (pre-patchified for wan2.2)
    float* out;                 // [lw, lh, z_dim, lt] posterior mean, written
    std::int64_t out_len;       // expected element count of out
    TSGWanVaeConv stem;         // encoder conv1: px_c -> dim, 3x3x3 causal
    TSGWanVaeResBlockW res[8];  // 4 scales x 2 residual blocks
    TSGWanVaeDownW down[3];     // after scales 0 (spatial), 1 and 2 (temporal+spatial)
    TSGWanVaeResBlockW mid0, mid2;
    TSGWanVaeAttnW mid1;
    TSGWanVaeNorm head_norm;
    TSGWanVaeConv head_conv;    // 3x3x3 -> 2*z_dim
    TSGWanVaeConv quant;        // quant conv 1x1x1 (2z -> 2z)
    std::int32_t px_w, px_h, px_c, px_t;
    std::int32_t z_dim;
    std::int32_t version;       // 1 (or 0) = wan2.1; 2 = wan2.2 TI2V (AvgDown shortcuts)
    std::int32_t struct_bytes;
    std::int32_t reserved;
};

} // extern "C" (struct declarations)

namespace {

// ---------------------------------------------------------------------------
// Shared graph helpers
// ---------------------------------------------------------------------------

struct WanUpload { ggml_tensor* t; void* d; std::size_t b; };

// Collects the weight-leaf bindings for one graph build: tensors that hit the
// resident cache upload once (on a cache miss); everything else becomes a
// gallocr input slot uploaded per call.
struct WanBind
{
    ggml_context* ctx = nullptr;
    ggml_backend_dev_t dev = nullptr;
    std::vector<WanUpload> uploads;

    void bind(ggml_tensor* tt, void* dd, std::size_t bytes)
    {
        if (tt == nullptr || dd == nullptr) return;
        static const bool noResident = []{
            const char* e = std::getenv("TS_WAN_NO_RESIDENT"); return e != nullptr && e[0] == '1'; }();
        if (bytes >= 4096 && !noResident)
        {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, tt, dd, bytes, buf, addr, needs)
                && ggml_backend_tensor_alloc(buf, tt, addr) == GGML_STATUS_SUCCESS)
            {
                if (needs) uploads.push_back({tt, dd, bytes});
                return;
            }
            invalidate_cached_buffer(dd);
        }
        ggml_set_input(tt);
        uploads.push_back({tt, dd, bytes});
    }

    // Declare + bind a weight matrix (+ optional F32 bias) from a TSGWanW.
    ggml_tensor* w2d(const TSGWanW& s, ggml_tensor** bias_out = nullptr)
    {
        ggml_tensor* wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
        bind(wt, s.w, static_cast<std::size_t>(s.bytes));
        if (bias_out != nullptr)
        {
            ggml_tensor* bt = nullptr;
            if (s.b != nullptr)
            {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * sizeof(float));
            }
            *bias_out = bt;
        }
        return wt;
    }

    // Declare + bind a small F32 vector (norm gains, modulation, biases).
    ggml_tensor* f32v(void* data, std::int64_t n)
    {
        if (data == nullptr) return nullptr;
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
        return t;
    }
};

// Overflow-safe matmul for quantized weights (see qi_mm in ggml_ops_qwen_image.cpp:
// ggml quantizes the activation to q8_1 whose per-block FP16 sum overflows for
// large activations; scaling is exact for the scale-invariant q8 formats).
constexpr float WAN_MM_SCALE = 1024.0f;

ggml_tensor* wan_mm(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, bool prescale)
{
    if (!prescale) return ggml_mul_mat(ctx, w, x);
    ggml_tensor* xs = ggml_scale(ctx, x, 1.0f / WAN_MM_SCALE);
    return ggml_scale(ctx, ggml_mul_mat(ctx, w, xs), WAN_MM_SCALE);
}

ggml_tensor* wan_lin(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b, bool prescale = false)
{
    ggml_tensor* o = wan_mm(ctx, w, x, prescale);
    return b != nullptr ? ggml_add(ctx, o, b) : o;
}

// RMS norm over ne0 then * gain.
ggml_tensor* wan_rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps)
{
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Interleaved RoPE with pair-duplicated cos/sin tables [head_dim, seq]; the
// output uses the half-split channel layout, which is dot-product-invariant when
// applied to both q and k (see qi_rope for the launch-count rationale).
ggml_tensor* wan_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cosf, ggml_tensor* sinf,
                      int head_dim, int heads, int seq)
{
    const int half = head_dim / 2;
    ggml_tensor* x4 = ggml_reshape_4d(ctx, x, 2, half, heads, seq);
    ggml_tensor* even = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], 0);
    ggml_tensor* odd  = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[0]);
    ggml_tensor* cosh = ggml_view_4d(ctx, cosf, 1, half, 1, seq, 2 * cosf->nb[0], cosf->nb[1], cosf->nb[1], 0);
    ggml_tensor* sinh = ggml_view_4d(ctx, sinf, 1, half, 1, seq, 2 * sinf->nb[0], sinf->nb[1], sinf->nb[1], 0);
    ggml_tensor* ep = ggml_sub(ctx, ggml_mul(ctx, even, cosh), ggml_mul(ctx, odd, sinh));
    ggml_tensor* op = ggml_add(ctx, ggml_mul(ctx, odd, cosh), ggml_mul(ctx, even, sinh));
    ggml_tensor* ep3 = ggml_reshape_3d(ctx, ep, half, heads, seq);
    ggml_tensor* op3 = ggml_reshape_3d(ctx, op, half, heads, seq);
    return ggml_concat(ctx, ep3, op3, 0);
}

inline bool wan_flash_enabled()
{
    static const bool on = []{ const char* e = std::getenv("TS_WAN_DIT_FLASH"); return e == nullptr || e[0] != '0'; }();
    return on;
}

constexpr int kWanKvStride = 256;   // ggml-cuda FATTN_KQ_STRIDE

// Bidirectional flash mask for `n` real tokens padded to the KV stride: zeros for
// real keys, -inf for padded keys. Cached per n (stable pointer -> resident bind).
const ggml_fp16_t* wan_attn_mask_host(int n, int& n_pad)
{
    n_pad = ((n + kWanKvStride - 1) / kWanKvStride) * kWanKvStride;
    static std::unordered_map<int, std::vector<ggml_fp16_t>> cache;
    auto it = cache.find(n);
    if (it != cache.end()) return it->second.data();
    std::vector<ggml_fp16_t>& m = cache[n];
    m.assign(static_cast<std::size_t>(n_pad) * n_pad, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (int q = 0; q < n_pad; q++)
        for (int kv = n; kv < n_pad; kv++)
            m[static_cast<std::size_t>(q) * n_pad + kv] = ninf;
    return m.data();
}

// Attention over q [hd, n_q, heads], k/v [hd, n_kv, heads]. When `mask` is given
// (F16 [kv_pad, q_pad]) and flash is enabled+supported, k/v must already be padded
// to kv_pad rows. Falls back to materialized scores+softmax otherwise.
// Returns [hd*heads, n_q].
ggml_tensor* wan_attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                           ggml_tensor* mask, int dim, int n_q, float scale)
{
    if (wan_flash_enabled())
    {
        ggml_tensor* fa = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        if (backend_supports_op(fa))
            return ggml_reshape_2d(ctx, fa, dim, n_q);
    }
    // Materialized reference path (backends without flash support; O(n_kv * n_q)
    // scores). The caller's k/v are already padded when a mask is given, and
    // soft_max_ext folds the scale and the (F16) additive mask in one op.
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                       // [n_kv, n_q, heads]
    ggml_tensor* m = mask != nullptr
        ? ggml_view_2d(ctx, mask, k->ne[1], n_q, mask->nb[1], 0)
        : nullptr;
    kq = ggml_soft_max_ext(ctx, kq, m, scale, 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [n_kv, hd, heads]
    ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);                        // [hd, n_q, heads]
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [hd, heads, n_q]
    return ggml_reshape_2d(ctx, merged, dim, n_q);
}

// [hd, heads, seq] -> [hd, seq, heads] (flash-attn layout)
ggml_tensor* wan_heads_seq(ggml_context* ctx, ggml_tensor* x)
{
    return ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
}

// ---------------------------------------------------------------------------
// UMT5-XXL encoder graph
// ---------------------------------------------------------------------------

bool wan_t5_build_and_run(const TSGgmlWanT5Desc* d)
{
    const int n = d->n_tokens, dim = d->dim, heads = d->heads, hd = d->head_dim, nl = d->num_layers;

    const std::size_t nodes = static_cast<std::size_t>(nl) * 48 + 512;
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 512)
                             + ggml_graph_overhead_custom(nodes, false) + (4u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ContextHandle context(ggml_init(ip));
    if (context.value == nullptr) { set_last_error("WanT5Encode: ctx alloc failed."); return false; }
    ggml_context* ctx = context.value;

    WanBind wb; wb.ctx = ctx; wb.dev = ggml_backend_get_device(g_backend);

    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_tensor* bucket = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<std::int64_t>(n) * n);
    ggml_tensor* mask = d->attn_mask != nullptr ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n) : nullptr;
    ggml_tensor* outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n);

    ggml_tensor* embd = wb.w2d(d->tok_embd);
    ggml_tensor* x = ggml_get_rows(ctx, embd, tokens);               // [dim, n]

    ggml_tensor* mask3 = mask != nullptr ? ggml_reshape_3d(ctx, mask, n, 1, 1) : nullptr;

    for (int l = 0; l < nl; l++)
    {
        const TSGWanT5LayerW& s = d->layers[l];
        ggml_tensor* qw = wb.w2d(s.q); ggml_tensor* kw = wb.w2d(s.k);
        ggml_tensor* vw = wb.w2d(s.v); ggml_tensor* ow = wb.w2d(s.o);
        ggml_tensor* gw = wb.w2d(s.gate); ggml_tensor* uw = wb.w2d(s.up); ggml_tensor* dw = wb.w2d(s.down);
        ggml_tensor* an = wb.f32v(s.attn_norm, dim);
        ggml_tensor* fn = wb.f32v(s.ffn_norm, dim);
        ggml_tensor* rb = wb.f32v(s.rel_b, static_cast<std::int64_t>(heads) * 32);

        // relative position bias: gather per (q,k) bucket -> [n_k, n_q, heads]
        ggml_tensor* rb2 = ggml_reshape_2d(ctx, rb, heads, 32);
        ggml_tensor* pos = ggml_get_rows(ctx, rb2, bucket);          // [heads, n*n]
        pos = ggml_reshape_3d(ctx, pos, heads, n, n);                // [heads, k, q]
        pos = ggml_cont(ctx, ggml_permute(ctx, pos, 2, 0, 1, 3));    // [k, q, heads]
        ggml_tensor* bias = mask3 != nullptr ? ggml_add(ctx, pos, mask3) : pos;

        // self-attention (T5: no dot-product scaling, bias added to logits)
        ggml_tensor* h = wan_rms(ctx, x, an, d->eps);
        ggml_tensor* q = wan_mm(ctx, qw, h, true);
        ggml_tensor* k = wan_mm(ctx, kw, h, true);
        ggml_tensor* v = wan_mm(ctx, vw, h, true);
        ggml_tensor* q3 = wan_heads_seq(ctx, ggml_reshape_3d(ctx, q, hd, heads, n));  // [hd, n, heads]
        ggml_tensor* k3 = wan_heads_seq(ctx, ggml_reshape_3d(ctx, k, hd, heads, n));
        ggml_tensor* kq = ggml_mul_mat(ctx, k3, q3);                 // [n_k, n_q, heads]
        kq = ggml_add(ctx, kq, bias);
        kq = ggml_soft_max(ctx, kq);
        ggml_tensor* v3 = ggml_reshape_3d(ctx, v, hd, heads, n);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v3, 1, 2, 0, 3));  // [n, hd, heads]
        ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);                // [hd, n_q, heads]
        ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        merged = ggml_reshape_2d(ctx, merged, dim, n);
        x = ggml_add(ctx, x, wan_mm(ctx, ow, merged, true));

        // gated-GELU FFN (T5DenseGatedActDense)
        ggml_tensor* h2 = wan_rms(ctx, x, fn, d->eps);
        ggml_tensor* hg = ggml_gelu(ctx, wan_mm(ctx, gw, h2, true));
        ggml_tensor* hu = wan_mm(ctx, uw, h2, true);
        x = ggml_add(ctx, x, wan_mm(ctx, dw, ggml_mul(ctx, hg, hu), true));
    }

    ggml_tensor* fnorm = wb.f32v(d->final_norm, dim);
    x = wan_rms(ctx, x, fnorm, d->eps);
    ggml_tensor* outc = ggml_cpy(ctx, x, outT);
    ggml_set_output(outc);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(graph, outc);

    ggml_set_input(tokens);
    ggml_set_input(bucket);
    if (mask != nullptr) ggml_set_input(mask);

    BufferHandle buffer(nullptr);
    if (!alloc_graph_reuse_gallocr(graph))
    {
        buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (buffer.value == nullptr) { set_last_error("WanT5Encode: buffer alloc failed."); return false; }
    }

    host_read_barrier();
    for (auto& u : wb.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    ggml_backend_tensor_set(tokens, d->tokens, 0, static_cast<std::size_t>(n) * sizeof(std::int32_t));
    ggml_backend_tensor_set(bucket, d->rel_bucket, 0, static_cast<std::size_t>(n) * n * sizeof(std::int32_t));
    if (mask != nullptr)
        ggml_backend_tensor_set(mask, d->attn_mask, 0, static_cast<std::size_t>(n) * sizeof(float));

    if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
    { set_last_error("WanT5Encode: graph compute failed."); return false; }
    ggml_backend_synchronize(g_backend);
    ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(dim) * n * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// Wan DiT graph
// ---------------------------------------------------------------------------

struct WanDitGraph
{
    ggml_cgraph* graph = nullptr;
    ggml_tensor *xIn = nullptr, *ctxIn = nullptr, *tsinIn = nullptr, *tsin0In = nullptr;
    ggml_tensor *cosIn = nullptr, *sinIn = nullptr, *mask = nullptr, *outT = nullptr;
    const ggml_fp16_t* maskHost = nullptr; int seq_pad = 0;
    std::vector<WanUpload> uploads;
    // TS_WAN_DIT_TRACE: named intermediates flagged as graph outputs (so gallocr
    // cannot alias their buffers) whose stats are printed after compute.
    std::vector<std::pair<std::string, ggml_tensor*>> taps;
};

// Per-block device-side weight leaves.
struct WanDitBlockT
{
    ggml_tensor *mod;
    ggml_tensor *sqw, *sqb, *skw, *skb, *svw, *svb, *sow, *sob;
    ggml_tensor *snq, *snk;
    ggml_tensor *n3w, *n3b;
    ggml_tensor *xqw, *xqb, *xkw, *xkb, *xvw, *xvb, *xow, *xob;
    ggml_tensor *xnq, *xnk;
    ggml_tensor *f0w, *f0b, *f2w, *f2b;
};

bool wan_dit_build_graph(ggml_context* ctx, const TSGgmlWanDitDesc* d, WanDitGraph& g)
{
    // TS_WAN_DIT_MAX_LAYERS=k: build only the first k blocks (debug bisect aid).
    static const int maxLayers = []() {
        const char* e = std::getenv("TS_WAN_DIT_MAX_LAYERS");
        return e != nullptr ? std::atoi(e) : -1;
    }();
    const int dim = d->dim, heads = d->heads, hd = d->head_dim, seq = d->seq, cl = d->ctx_len;
    const int nl = maxLayers >= 0 && maxLayers < d->num_layers ? maxLayers : d->num_layers;
    const float eps = d->eps;
    const int in_tok = static_cast<int>(d->patch.ne0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    WanBind wb; wb.ctx = ctx; wb.dev = ggml_backend_get_device(g_backend);
    const bool traceOn = std::getenv("TS_WAN_DIT_TRACE") != nullptr;
    auto tap = [&](const char* name, ggml_tensor* t) {
        if (!traceOn || t == nullptr) return;
        ggml_set_output(t);
        g.taps.emplace_back(name, t);
    };

    // seq0 > 0: the first seq0 tokens (the conditioning image's latent frame in
    // Wan 2.2 TI2V i2v) are AdaLN-modulated with tsin0's timestep instead of
    // tsin's. Attention and every matmul still run over the full joint sequence.
    const int s0 = (d->seq0 > 0 && d->tsin0 != nullptr && d->seq0 < seq) ? d->seq0 : 0;

    g.xIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_tok, seq);
    g.ctxIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->text_dim, cl);
    g.tsinIn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->freq_dim);
    g.tsin0In = s0 > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->freq_dim) : nullptr;
    g.cosIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
    g.sinIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
    g.outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<int>(d->head.ne1), seq);

    if (wan_flash_enabled())
    {
        g.maskHost = wan_attn_mask_host(seq, g.seq_pad);
        g.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.seq_pad, g.seq_pad);
    }

    // ---- prelude weights ----
    ggml_tensor *patchB, *t0B, *t2B, *ti0B, *ti2B, *tpB, *headB;
    ggml_tensor* patchW = wb.w2d(d->patch, &patchB);
    ggml_tensor* text0W = wb.w2d(d->text0, &t0B);
    ggml_tensor* text2W = wb.w2d(d->text2, &t2B);
    ggml_tensor* time0W = wb.w2d(d->time0, &ti0B);
    ggml_tensor* time2W = wb.w2d(d->time2, &ti2B);
    ggml_tensor* tprojW = wb.w2d(d->tproj, &tpB);
    ggml_tensor* headW = wb.w2d(d->head, &headB);
    ggml_tensor* headMod = wb.f32v(d->head_mod, 2LL * dim);

    // ---- per-block weights ----
    std::vector<WanDitBlockT> bt(nl);
    for (int l = 0; l < nl; l++)
    {
        const TSGWanDitBlockW& s = d->blocks[l];
        WanDitBlockT& b = bt[l];
        b.mod = wb.f32v(s.modulation, 6LL * dim);
        b.sqw = wb.w2d(s.sq, &b.sqb); b.skw = wb.w2d(s.sk, &b.skb);
        b.svw = wb.w2d(s.sv, &b.svb); b.sow = wb.w2d(s.so, &b.sob);
        b.snq = wb.f32v(s.s_norm_q, dim); b.snk = wb.f32v(s.s_norm_k, dim);
        b.n3w = wb.f32v(s.norm3_w, dim); b.n3b = wb.f32v(s.norm3_b, dim);
        b.xqw = wb.w2d(s.xq, &b.xqb); b.xkw = wb.w2d(s.xk, &b.xkb);
        b.xvw = wb.w2d(s.xv, &b.xvb); b.xow = wb.w2d(s.xo, &b.xob);
        b.xnq = wb.f32v(s.x_norm_q, dim); b.xnk = wb.f32v(s.x_norm_k, dim);
        b.f0w = wb.w2d(s.ffn0, &b.f0b); b.f2w = wb.w2d(s.ffn2, &b.f2b);
    }

    // ---- prelude ----
    // patch embedding (pre-flattened conv3d): [in_tok, seq] -> [dim, seq]
    ggml_tensor* x = wan_lin(ctx, patchW, g.xIn, patchB);
    tap("patch_embed", x);

    // time embedding: sinusoid [freq_dim] -> [dim]; e0 = time_projection(silu(e)) -> [6*dim]
    auto timeEmbed = [&](ggml_tensor* tsin, ggml_tensor** e2dOut) -> ggml_tensor*
    {
        ggml_tensor* e = wan_lin(ctx, time0W, tsin, ti0B);
        e = ggml_silu(ctx, e);
        e = wan_lin(ctx, time2W, e, ti2B);                            // [dim]
        ggml_tensor* e2d = ggml_reshape_2d(ctx, e, dim, 1);
        *e2dOut = e2d;
        return wan_lin(ctx, tprojW, ggml_silu(ctx, e2d), tpB);        // [6*dim, 1]
    };
    ggml_tensor* e2d = nullptr;
    ggml_tensor* e0 = timeEmbed(g.tsinIn, &e2d);
    tap("time_e", e2d); tap("time_e0", e0);
    ggml_tensor* e2dB = nullptr;
    ggml_tensor* e0B = s0 > 0 ? timeEmbed(g.tsin0In, &e2dB) : nullptr;

    // Apply f per token segment: rows [0, s0) with the B (tsin0) parameters, rows
    // [s0, seq) with the A (tsin) parameters; concat restores [dim', seq]. The row
    // ranges of a contiguous [dim', seq] tensor are themselves contiguous views.
    auto segRows = [&](ggml_tensor* x, std::int64_t from, std::int64_t count) {
        return ggml_view_2d(ctx, x, x->ne[0], count, x->nb[1],
                            static_cast<std::size_t>(from) * x->nb[1]);
    };
    auto segModulate = [&](ggml_tensor* n, ggml_tensor* shiftA, ggml_tensor* scaleA,
                           ggml_tensor* shiftB, ggml_tensor* scaleB) -> ggml_tensor*
    {
        if (s0 == 0)
            return ggml_add(ctx, ggml_add(ctx, n, ggml_mul(ctx, n, scaleA)), shiftA);
        ggml_tensor* nB = segRows(n, 0, s0);
        ggml_tensor* nA = segRows(n, s0, seq - s0);
        ggml_tensor* yB = ggml_add(ctx, ggml_add(ctx, nB, ggml_mul(ctx, nB, scaleB)), shiftB);
        ggml_tensor* yA = ggml_add(ctx, ggml_add(ctx, nA, ggml_mul(ctx, nA, scaleA)), shiftA);
        return ggml_concat(ctx, yB, yA, 1);
    };
    auto segGate = [&](ggml_tensor* v, ggml_tensor* gateA, ggml_tensor* gateB) -> ggml_tensor*
    {
        if (s0 == 0)
            return ggml_mul(ctx, v, gateA);
        ggml_tensor* vB = segRows(v, 0, s0);
        ggml_tensor* vA = segRows(v, s0, seq - s0);
        return ggml_concat(ctx, ggml_mul(ctx, vB, gateB), ggml_mul(ctx, vA, gateA), 1);
    };

    // text embedding: [text_dim, cl] -> [dim, cl] (GELU-tanh between the linears)
    ggml_tensor* txt = wan_lin(ctx, text0W, g.ctxIn, t0B);
    txt = ggml_gelu(ctx, txt);
    txt = wan_lin(ctx, text2W, txt, t2B);                             // [dim, cl]
    tap("text_embed", txt);

    // ---- blocks ----
    for (int l = 0; l < nl; l++)
    {
        const WanDitBlockT& b = bt[l];

        // e_b = e0 + per-block modulation -> six [dim, 1] chunks (per timestep segment)
        ggml_tensor* mod2d = ggml_reshape_2d(ctx, b.mod, 6LL * dim, 1);
        ggml_tensor* eb = ggml_add(ctx, e0, mod2d);
        ggml_tensor* ebB = s0 > 0 ? ggml_add(ctx, e0B, mod2d) : nullptr;
        auto chunk = [&](ggml_tensor* e6, int k) -> ggml_tensor* {
            if (e6 == nullptr) return nullptr;
            return ggml_view_2d(ctx, e6, dim, 1, e6->nb[1], static_cast<std::size_t>(k) * dim * sizeof(float));
        };
        ggml_tensor* eShiftA = chunk(eb, 0); ggml_tensor* eScaleA = chunk(eb, 1); ggml_tensor* eGateA = chunk(eb, 2);
        ggml_tensor* eShiftM = chunk(eb, 3); ggml_tensor* eScaleM = chunk(eb, 4); ggml_tensor* eGateM = chunk(eb, 5);
        ggml_tensor* eShiftAB = chunk(ebB, 0); ggml_tensor* eScaleAB = chunk(ebB, 1); ggml_tensor* eGateAB = chunk(ebB, 2);
        ggml_tensor* eShiftMB = chunk(ebB, 3); ggml_tensor* eScaleMB = chunk(ebB, 4); ggml_tensor* eGateMB = chunk(ebB, 5);

        // --- self-attention sub-layer ---
        ggml_tensor* n1 = ggml_norm(ctx, x, eps);                    // LayerNorm, no affine
        ggml_tensor* y = segModulate(n1, eShiftA, eScaleA, eShiftAB, eScaleAB);

        ggml_tensor* q = wan_rms(ctx, wan_lin(ctx, b.sqw, y, b.sqb), b.snq, eps);
        ggml_tensor* k = wan_rms(ctx, wan_lin(ctx, b.skw, y, b.skb), b.snk, eps);
        ggml_tensor* v = wan_lin(ctx, b.svw, y, b.svb);

        ggml_tensor* q3 = ggml_reshape_3d(ctx, q, hd, heads, seq);
        ggml_tensor* k3 = ggml_reshape_3d(ctx, k, hd, heads, seq);
        q3 = wan_rope(ctx, q3, g.cosIn, g.sinIn, hd, heads, seq);
        k3 = wan_rope(ctx, k3, g.cosIn, g.sinIn, hd, heads, seq);

        ggml_tensor* qa = wan_heads_seq(ctx, q3);                    // [hd, seq, heads]
        ggml_tensor* ka = wan_heads_seq(ctx, k3);
        ggml_tensor* va = wan_heads_seq(ctx, ggml_reshape_3d(ctx, v, hd, heads, seq));
        if (g.mask != nullptr && g.seq_pad > seq)
        {
            ka = ggml_pad(ctx, ka, 0, g.seq_pad - seq, 0, 0);
            va = ggml_pad(ctx, va, 0, g.seq_pad - seq, 0, 0);
        }
        ggml_tensor* attn = wan_attention(ctx, qa, ka, va, g.mask, dim, seq, scale);
        attn = wan_lin(ctx, b.sow, attn, b.sob);
        x = ggml_add(ctx, x, segGate(attn, eGateA, eGateAB));

        // --- cross-attention sub-layer (affine LayerNorm, no gating) ---
        ggml_tensor* n3 = ggml_norm(ctx, x, eps);
        n3 = ggml_add(ctx, ggml_mul(ctx, n3, b.n3w), b.n3b);
        ggml_tensor* xq = wan_rms(ctx, wan_lin(ctx, b.xqw, n3, b.xqb), b.xnq, eps);
        ggml_tensor* xk = wan_rms(ctx, wan_lin(ctx, b.xkw, txt, b.xkb), b.xnk, eps);
        ggml_tensor* xv = wan_lin(ctx, b.xvw, txt, b.xvb);
        ggml_tensor* xqa = wan_heads_seq(ctx, ggml_reshape_3d(ctx, xq, hd, heads, seq));
        ggml_tensor* xka = wan_heads_seq(ctx, ggml_reshape_3d(ctx, xk, hd, heads, cl));
        ggml_tensor* xva = wan_heads_seq(ctx, ggml_reshape_3d(ctx, xv, hd, heads, cl));
        // ctx_len is a multiple of the KV stride (512), so flash needs no mask here.
        ggml_tensor* xattn = wan_attention(ctx, xqa, xka, xva, nullptr, dim, seq, scale);
        x = ggml_add(ctx, x, wan_lin(ctx, b.xow, xattn, b.xob));

        // --- FFN sub-layer ---
        ggml_tensor* n2 = ggml_norm(ctx, x, eps);
        ggml_tensor* ym = segModulate(n2, eShiftM, eScaleM, eShiftMB, eScaleMB);
        ggml_tensor* hM = ggml_gelu(ctx, wan_lin(ctx, b.f0w, ym, b.f0b));
        ggml_tensor* mlp = wan_lin(ctx, b.f2w, hM, b.f2b);
        x = ggml_add(ctx, x, segGate(mlp, eGateM, eGateMB));
        if (l == 0 || l == nl - 1)
        {
            tap((std::string("block") + std::to_string(l) + "_out").c_str(), x);
        }
    }

    // ---- head: modulated norm + projection ----
    // es = head_mod([2, dim]) + e; x = norm(x) * (1 + es[1]) + es[0]
    ggml_tensor* hm = ggml_reshape_2d(ctx, headMod, dim, 2);
    auto headChunks = [&](ggml_tensor* eSeg, ggml_tensor** shift, ggml_tensor** scaleT)
    {
        ggml_tensor* he = ggml_add(ctx, hm, eSeg);                   // broadcast e over the 2 rows
        *shift = ggml_view_2d(ctx, he, dim, 1, he->nb[1], 0);
        *scaleT = ggml_view_2d(ctx, he, dim, 1, he->nb[1], he->nb[1]);
    };
    ggml_tensor *hShift, *hScale, *hShiftB = nullptr, *hScaleB = nullptr;
    headChunks(e2d, &hShift, &hScale);
    if (s0 > 0) headChunks(e2dB, &hShiftB, &hScaleB);
    ggml_tensor* hn = ggml_norm(ctx, x, eps);
    hn = segModulate(hn, hShift, hScale, hShiftB, hScaleB);
    tap("head_norm", hn);
    ggml_tensor* outv = wan_lin(ctx, headW, hn, headB);              // [out_tok, seq]

    ggml_tensor* outc = ggml_cpy(ctx, outv, g.outT);
    ggml_set_output(outc);

    const std::size_t nodes = static_cast<std::size_t>(nl) * 128 + 1024;
    g.graph = ggml_new_graph_custom(ctx, nodes, false);
    if (g.graph == nullptr) return false;
    ggml_build_forward_expand(g.graph, outc);
    for (auto& [tname, tt] : g.taps) ggml_build_forward_expand(g.graph, tt);

    ggml_set_input(g.xIn); ggml_set_input(g.ctxIn); ggml_set_input(g.tsinIn);
    if (g.tsin0In != nullptr) ggml_set_input(g.tsin0In);
    ggml_set_input(g.cosIn); ggml_set_input(g.sinIn);

    // Bind the flash mask resident (stable cached host pointer, uploaded once).
    bool maskResident = false;
    if (g.mask != nullptr && g.maskHost != nullptr)
    {
        const std::size_t mbytes = static_cast<std::size_t>(g.seq_pad) * g.seq_pad * sizeof(ggml_fp16_t);
        void* mhost = const_cast<ggml_fp16_t*>(g.maskHost);
        ggml_backend_buffer_t mbuf = nullptr; void* maddr = nullptr; bool mneeds = false;
        if (try_get_cacheable_tensor_buffer(g_backend, wb.dev, g.mask, mhost, mbytes, mbuf, maddr, mneeds)
            && ggml_backend_tensor_alloc(mbuf, g.mask, maddr) == GGML_STATUS_SUCCESS)
        {
            maskResident = true;
            if (mneeds) wb.uploads.push_back({g.mask, mhost, mbytes});
        }
        else invalidate_cached_buffer(mhost);
    }
    if (g.mask != nullptr && !maskResident)
    {
        ggml_set_input(g.mask);
        wb.uploads.push_back({g.mask, const_cast<ggml_fp16_t*>(g.maskHost),
                              static_cast<std::size_t>(g.seq_pad) * g.seq_pad * sizeof(ggml_fp16_t)});
    }

    g.uploads = std::move(wb.uploads);
    return true;
}

// Persistent per-shape entry (CUDA-graph capture; same pattern as QiForwardPersist).
struct WanDitPersist
{
    bool valid = false;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    WanDitGraph g{};
    int seq = 0, cl = 0, nl = 0;
    const void* wkey = nullptr;

    int seq0 = 0;

    bool matches(const TSGgmlWanDitDesc* d) const
    {
        return valid && seq == d->seq && cl == d->ctx_len && nl == d->num_layers &&
               seq0 == d->seq0 && wkey == d->blocks[0].sq.w;
    }
    void reset()
    {
        if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        g = WanDitGraph{}; valid = false;
        seq = cl = nl = seq0 = 0; wkey = nullptr;
    }
};

constexpr int kWanDitCacheMax = 2;
WanDitPersist g_wanDit[kWanDitCacheMax];
int g_wanDitRR = 0;

bool wan_dit_capture_enabled()
{
    static const int s = []() {
        const char* e = std::getenv("TS_WAN_DIT_CAPTURE");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (s == 0 || g_backend == nullptr) return false;
    const char* name = ggml_backend_name(g_backend);
    return name != nullptr && std::strncmp(name, "CUDA", 4) == 0;
}

void wan_dit_upload_inputs(const WanDitGraph& g, const TSGgmlWanDitDesc* d)
{
    ggml_backend_tensor_set(g.xIn, d->x, 0, static_cast<std::size_t>(d->patch.ne0) * d->seq * sizeof(float));
    ggml_backend_tensor_set(g.ctxIn, d->context, 0, static_cast<std::size_t>(d->text_dim) * d->ctx_len * sizeof(float));
    ggml_backend_tensor_set(g.tsinIn, d->tsin, 0, static_cast<std::size_t>(d->freq_dim) * sizeof(float));
    if (g.tsin0In != nullptr && d->tsin0 != nullptr)
        ggml_backend_tensor_set(g.tsin0In, d->tsin0, 0, static_cast<std::size_t>(d->freq_dim) * sizeof(float));
    ggml_backend_tensor_set(g.cosIn, d->cosf, 0, static_cast<std::size_t>(d->head_dim) * d->seq * sizeof(float));
    ggml_backend_tensor_set(g.sinIn, d->sinf, 0, static_cast<std::size_t>(d->head_dim) * d->seq * sizeof(float));
}

// TS_WAN_DIT_TRACE=1: after compute, scan the graph nodes for the first one whose
// (possibly reused) buffer holds NaN and print its neighborhood. Debug aid only —
// gallocr reuses buffers, so downstream reads can alias, but the NaN onset node
// in execution order is reliable because NaN propagates forward.
void wan_dit_trace_taps(const WanDitGraph& g)
{
    const char* path = std::getenv("TS_WAN_DIT_TRACE");
    if (path == nullptr || g.taps.empty()) return;
    std::FILE* out = std::fopen(path, "a");
    if (out == nullptr) return;
    for (const auto& [name, t] : g.taps)
    {
        if (t->buffer == nullptr) { std::fprintf(out, "[wan-trace] %s: NO BUFFER\n", name.c_str()); continue; }
        const std::int64_t nel = ggml_nelements(t);
        std::vector<float> host(static_cast<std::size_t>(nel));
        ggml_backend_tensor_get(t, host.data(), 0, static_cast<std::size_t>(nel) * sizeof(float));
        double mn = 1e300, mx = -1e300, sum = 0; std::int64_t nan = 0;
        for (std::int64_t j = 0; j < nel; j++)
        {
            float v = host[static_cast<std::size_t>(j)];
            if (std::isnan(v) || std::isinf(v)) { nan++; continue; }
            mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); sum += v;
        }
        std::fprintf(out, "[wan-trace] %-14s ne=[%lld,%lld] nan=%lld/%lld min=%g max=%g mean=%g\n",
                     name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                     (long long)nan, (long long)nel, mn, mx, sum / std::max<std::int64_t>(1, nel - nan));
    }
    std::fclose(out);
}

int wan_dit_run_persist(WanDitPersist* e, const TSGgmlWanDitDesc* d)
{
    WanDitGraph& g = e->g;
    if (e->galloc != nullptr && !ggml_gallocr_alloc_graph(e->galloc, g.graph))
    {
        e->reset();
        set_last_error("WanDitForward: gallocr realloc failed.");
        return 0;
    }
    // Re-upload the input-slot leaves living in the gallocr buffer (see qi_fwd_run:
    // the buffer is not cleared here because this graph never reads an intermediate
    // before writing it, but input-slot weights must be refreshed after a re-plan).
    host_read_barrier();
    if (g.outT != nullptr && g.outT->buffer != nullptr)
    {
        ggml_backend_buffer_t gb = g.outT->buffer;
        for (auto& u : g.uploads)
            if (u.t->buffer == gb) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    }
    wan_dit_upload_inputs(g, d);
    if (ggml_backend_graph_compute(g_backend, g.graph) != GGML_STATUS_SUCCESS)
    { set_last_error("WanDitForward: graph compute failed."); return 0; }
    ggml_backend_synchronize(g_backend);
    ggml_backend_tensor_get(g.outT, d->out, 0, static_cast<std::size_t>(d->head.ne1) * d->seq * sizeof(float));
    clear_last_error();
    return 1;
}

WanDitPersist* wan_dit_build_persist(const TSGgmlWanDitDesc* d)
{
    const int nl = d->num_layers;
    const std::size_t nodes = static_cast<std::size_t>(nl) * 128 + 1024;
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 1024)
                             + ggml_graph_overhead_custom(nodes, false) + (8u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) return nullptr;

    WanDitGraph g;
    if (!wan_dit_build_graph(ctx, d, g)) { ggml_free(ctx); return nullptr; }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr) { ggml_free(ctx); return nullptr; }
    if (!ggml_gallocr_alloc_graph(galloc, g.graph)) { ggml_gallocr_free(galloc); ggml_free(ctx); return nullptr; }

    // VRAM spill guard (see qi_fwd_build_persist).
    {
        ggml_backend_dev_t mdev = ggml_backend_get_device(g_backend);
        std::size_t freeb = 0, totalb = 0;
        if (mdev) ggml_backend_dev_memory(mdev, &freeb, &totalb);
        if (totalb > 0 && freeb < static_cast<std::size_t>(384) * 1024 * 1024)
        {
            std::fprintf(stderr, "[wan] dit capture: free VRAM %zu MiB after alloc -> non-persistent fallback\n", freeb >> 20);
            ggml_gallocr_free(galloc); ggml_free(ctx); return nullptr;
        }
    }

    host_read_barrier();
    for (auto& u : g.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);

    WanDitPersist* e = nullptr;
    for (auto& c : g_wanDit) if (!c.valid) { e = &c; break; }
    if (e == nullptr) { e = &g_wanDit[g_wanDitRR]; g_wanDitRR = (g_wanDitRR + 1) % kWanDitCacheMax; e->reset(); }
    e->ctx = ctx; e->galloc = galloc; e->g = g;
    e->seq = d->seq; e->cl = d->ctx_len; e->nl = nl; e->seq0 = d->seq0; e->wkey = d->blocks[0].sq.w;
    e->valid = true;
    return e;
}

// ---------------------------------------------------------------------------
// Wan VAE decoder graph
// ---------------------------------------------------------------------------

// Feature layout throughout the decoder: [W, H, C, T] (frame-contiguous slabs so
// temporal windows are plain ne3 views and 2D convs batch over frames).
struct WanVaeBuild
{
    ggml_context* ctx = nullptr;
    WanBind* wb = nullptr;
    // Causal feature caches (CACHE_T = 2 trailing input frames per conv), carried
    // across the temporal chunks inside the one graph.
    std::vector<ggml_tensor*> cache;
    int cursor = 0;
    int chunk = 0;
    long long gemmMax = 384LL << 20;   // im2col scratch budget (see wan_vae_gemm_budget)
};

// im2col scratch budget for wan_vae_conv2d. TS_WAN_VAE_GEMM_MAX_MB pins it
// (0 = direct conv); otherwise it adapts to the device's free memory. Both
// extremes were measured slow at 720p on a 16 GB card: the fixed 384 MB
// default splits every full-res conv into ~11 bands whose strided CONT +
// CONCAT-chain copies through ggml's generic kernels were 72% of decode time,
// while unbanded (4.1 GB im2col) oversubscribed WDDM residency and ran
// IM2COL/MUL_MAT at PCIe speed (~6 GB/s). free/8 keeps the scratch a small
// slice of what must stay resident (activation planes, cross-chunk caches,
// the growing output) — a few bands on 16 GB, unbanded on large cards — and
// the 384 MB floor preserves the old behavior under memory pressure.
static long long wan_vae_gemm_budget()
{
    const char* e = std::getenv("TS_WAN_VAE_GEMM_MAX_MB");
    if (e != nullptr) return std::strtoll(e, nullptr, 10) * 1024 * 1024;
    // Non-CUDA device backends (Vulkan) stay at the banded floor: their
    // drivers reject the multi-GB single gallocr arena that an unbanded
    // full-plane im2col produces (Vulkan maxMemoryAllocationSize is commonly
    // 4 GB or less), and the CPU backend gains nothing from bigger scratch.
    const char* name = g_backend != nullptr ? ggml_backend_name(g_backend) : nullptr;
    if (name == nullptr || std::strncmp(name, "CUDA", 4) != 0)
        return 384LL << 20;
    std::size_t freeB = 0, totalB = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
    if (dev != nullptr) ggml_backend_dev_memory(dev, &freeB, &totalB);
    long long budget = static_cast<long long>(freeB / 8);
    const long long lo = 384LL << 20, hi = 8LL << 30;
    if (budget < lo) budget = lo;
    if (budget > hi) budget = hi;
    return budget;
}

ggml_tensor* wan_vae_conv_w(WanVaeBuild& b, void* tap, const TSGWanVaeConv& c)
{
    ggml_tensor* t = ggml_new_tensor_4d(b.ctx, GGML_TYPE_F32, c.k, c.k, c.ic, c.oc);
    b.wb->bind(t, tap, static_cast<std::size_t>(c.k) * c.k * c.ic * c.oc * sizeof(float));
    return t;
}

// One 2D convolution of the decoder. ggml_conv_2d_direct needs no scratch but its
// CUDA kernel is a naive direct convolution — the whole-video decode measured
// ~50x slower than the im2col+GEMM path at 480p. So convs run as im2col (F16) +
// mul_mat, with the LARGE convs split into horizontal bands so the materialized
// im2col stays bounded (TS_WAN_VAE_GEMM_MAX_MB, default 384): the full-resolution
// 3x3 convs would otherwise materialize ~2.8 GB each, and together with the
// cross-chunk feature caches that filled a 16 GB card into WDDM paging (decode
// 51 s -> 180 s depending on what else held VRAM). Banding is exact — each band
// gets its vertical context rows from a single pre-padded copy of the input.
ggml_tensor* wan_vae_conv2d(WanVaeBuild& b, ggml_tensor* wt, ggml_tensor* x, int pad, int stride = 1)
{
    const long long gemmMax = b.gemmMax;
    ggml_context* ctx = b.ctx;
    if (gemmMax <= 0)
        return ggml_conv_2d_direct(ctx, wt, x, stride, stride, pad, pad, 1, 1);

    const long long KW = wt->ne[0], KH = wt->ne[1], IC = wt->ne[2];
    const long long OW = (x->ne[0] + 2 * pad - KW) / stride + 1;
    const long long OH = (x->ne[1] + 2 * pad - KH) / stride + 1;
    const long long T = x->ne[3];
    const long long rowBytes = KW * KH * IC * OW * T * 2;   // im2col bytes per output row
    ggml_tensor* wf16 = wt->type == GGML_TYPE_F16 ? wt : ggml_cast(ctx, wt, GGML_TYPE_F16);

    if (rowBytes * OH <= gemmMax)
        return ggml_conv_2d(ctx, wf16, x, stride, stride, pad, pad, 1, 1);

    // Banded path: pre-pad vertically once, then convolve horizontal bands whose
    // views carry their own context rows (horizontal padding stays in the conv).
    long long bandRows = std::max<long long>(1, gemmMax / std::max<long long>(1, rowBytes));
    ggml_tensor* xp = pad > 0 ? ggml_pad_ext(ctx, x, 0, 0, pad, pad, 0, 0, 0, 0) : x;
    ggml_tensor* out = nullptr;
    for (long long y = 0; y < OH; y += bandRows)
    {
        const long long rows = std::min(bandRows, OH - y);
        const long long inRows = (rows - 1) * stride + KH;
        ggml_tensor* band = ggml_view_4d(ctx, xp, xp->ne[0], inRows, xp->ne[2], xp->ne[3],
                                         xp->nb[1], xp->nb[2], xp->nb[3], y * stride * xp->nb[1]);
        // CUDA CONV_2D requires contiguous input; the row-band view is strided.
        band = ggml_cont(ctx, band);
        ggml_tensor* yb = ggml_conv_2d(ctx, wf16, band, stride, stride, pad, 0, 1, 1);
        out = out == nullptr ? yb : ggml_concat(ctx, out, yb, 1);
    }
    return out;
}

// Causal conv3d over x [W,H,IC,T] -> [W,H,OC,T]. Temporal context comes from the
// per-conv cache (previous chunk's trailing frames) or zero front-padding on the
// first chunk; the cache is updated to this chunk's trailing input frames.
ggml_tensor* wan_vae_causal_conv(WanVaeBuild& b, const TSGWanVaeConv& c, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t T = x->ne[3];
    const int pad = c.k == 3 ? 1 : 0;

    ggml_tensor* xin = x;
    if (c.kd == 3)
    {
        const int idx = b.cursor++;
        // Caches live across every remaining chunk of the graph, so they dominate
        // the cross-chunk VRAM liveness; they are stored F16 (visually lossless
        // temporal context, half the resident bytes) and cast back on use.
        ggml_tensor* prev = b.cache[idx] != nullptr ? ggml_cast(ctx, b.cache[idx], GGML_TYPE_F32) : nullptr;
        if (prev == nullptr)
        {
            // first chunk: 2 zero frames in front (causal zero padding)
            xin = ggml_pad_ext(ctx, x, 0, 0, 0, 0, 0, 0, 2, 0);
        }
        else
        {
            if (prev->ne[3] < 2)
            {
                // previous chunk had a single frame: zero-pad its front once
                prev = ggml_pad_ext(ctx, prev, 0, 0, 0, 0, 0, 0, 2 - static_cast<int>(prev->ne[3]), 0);
            }
            xin = ggml_concat(ctx, prev, x, 3);
        }
        // new cache = trailing (up to 2) input frames of x, extended with the old
        // cache's last frame when x is a single frame (sd.cpp CACHE_T logic).
        ggml_tensor* nc;
        if (T >= 2)
            nc = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], 2,
                              x->nb[1], x->nb[2], x->nb[3], (T - 2) * x->nb[3]);
        else if (prev != nullptr)
        {
            ggml_tensor* lastPrev = ggml_view_4d(ctx, prev,
                prev->ne[0], prev->ne[1], prev->ne[2], 1,
                prev->nb[1], prev->nb[2], prev->nb[3],
                (prev->ne[3] - 1) * prev->nb[3]);
            nc = ggml_concat(ctx, lastPrev, x, 3);
        }
        else
            nc = x;
        b.cache[idx] = ggml_cast(ctx, nc, GGML_TYPE_F16);
    }

    // Sum of per-temporal-tap 2D convolutions; output frame t reads input frames
    // t .. t+kd-1 of the assembled (front-padded) input.
    void* taps[3] = { c.tap0, c.tap1, c.tap2 };
    ggml_tensor* acc = nullptr;
    for (int j = 0; j < c.kd; j++)
    {
        ggml_tensor* wt = wan_vae_conv_w(b, taps[j], c);
        ggml_tensor* win = xin;
        if (c.kd > 1)
            win = ggml_view_4d(b.ctx, xin, xin->ne[0], xin->ne[1], xin->ne[2], T,
                               xin->nb[1], xin->nb[2], xin->nb[3], j * xin->nb[3]);
        ggml_tensor* y = wan_vae_conv2d(b, wt, win, pad);
        acc = acc == nullptr ? y : ggml_add(ctx, acc, y);
    }
    if (c.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(c.bias, c.oc);
        acc = ggml_add(ctx, acc, ggml_reshape_4d(ctx, bt, 1, 1, c.oc, 1));
    }
    return acc;
}

// Channel RMS norm (RMS_norm in wan_vae.hpp): normalize over C at every (w,h,t).
ggml_tensor* wan_vae_norm(WanVaeBuild& b, const TSGWanVaeNorm& n, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    ggml_tensor* g = b.wb->f32v(n.gamma, n.c);
    ggml_tensor* h = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));   // [C, W, H, T]
    h = ggml_mul(ctx, ggml_rms_norm(ctx, h, 1e-12f), g);
    return ggml_cont(ctx, ggml_permute(ctx, h, 2, 0, 1, 3));             // [W, H, C, T]
}

ggml_tensor* wan_vae_res_block(WanVaeBuild& b, const TSGWanVaeResBlockW& r, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    ggml_tensor* h = x;
    if (r.shortcut.tap0 != nullptr)
        h = wan_vae_causal_conv(b, r.shortcut, x);
    ggml_tensor* y = wan_vae_norm(b, r.n0, x);
    y = ggml_silu(ctx, y);
    y = wan_vae_causal_conv(b, r.c2, y);
    y = wan_vae_norm(b, r.n3, y);
    y = ggml_silu(ctx, y);
    y = wan_vae_causal_conv(b, r.c6, y);
    return ggml_add(ctx, y, h);
}

// Spatial single-head self-attention over each frame (mid block; T == 1 for the
// wan2.1 decoder chunks).
ggml_tensor* wan_vae_attn(WanVaeBuild& b, const TSGWanVaeAttnW& a, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
    const std::int64_t WH = W * H;

    ggml_tensor* qkvW = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 3 * C);
    b.wb->bind(qkvW, a.qkv_w, static_cast<std::size_t>(C) * 3 * C * sizeof(float));
    ggml_tensor* qkvB = b.wb->f32v(a.qkv_b, 3 * C);
    ggml_tensor* projW = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    b.wb->bind(projW, a.proj_w, static_cast<std::size_t>(C) * C * sizeof(float));
    ggml_tensor* projB = b.wb->f32v(a.proj_b, C);

    // [W,H,C,T] -> [C, WH, T]
    ggml_tensor* xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [C, W, H, T]
    xt = ggml_reshape_3d(ctx, xt, C, WH, T);
    ggml_tensor* identity = xt;

    ggml_tensor* n = ggml_mul(ctx, ggml_rms_norm(ctx, xt, 1e-12f),
                              b.wb->f32v(a.norm.gamma, C));
    ggml_tensor* qkv = ggml_add(ctx, ggml_mul_mat(ctx, qkvW, n),
                                ggml_reshape_3d(ctx, qkvB, 3 * C, 1, 1)); // [3C, WH, T]

    // q/k/v are ne0 slices of the fused [3C, WH, T] projection.
    ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, C, WH, T, qkv->nb[1], qkv->nb[2], 0));
    ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, C, WH, T, qkv->nb[1], qkv->nb[2], C * sizeof(float)));
    ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, C, WH, T, qkv->nb[1], qkv->nb[2], 2 * C * sizeof(float)));

    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                            // [WH_k, WH_q, T]
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(C)));
    kq = ggml_soft_max(ctx, kq);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));   // [WH, C, T]
    ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);                         // [C, WH_q, T]
    ggml_tensor* o = ggml_add(ctx, ggml_mul_mat(ctx, projW, kqv),
                              ggml_reshape_3d(ctx, projB, C, 1, 1));
    o = ggml_add(ctx, o, identity);

    // back to [W,H,C,T]
    o = ggml_reshape_4d(ctx, o, C, W, H, T);
    return ggml_cont(ctx, ggml_permute(ctx, o, 2, 0, 1, 3));
}

// Resample: optional causal temporal doubling (time_conv, chunks >= 1) then
// nearest x2 spatial upsample + 3x3 conv.
ggml_tensor* wan_vae_upsample(WanVaeBuild& b, const TSGWanVaeUpsampleW& u, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;

    if (u.time_conv.tap0 != nullptr)
    {
        const int idx = b.cursor++;
        if (b.chunk == 0)
        {
            // first chunk passes through untouched; the cache stays empty and the
            // NEXT chunk zero-pads (sd.cpp Resample upsample3d, chunk_idx == 0).
        }
        else
        {
            const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
            ggml_tensor* prev = b.cache[idx] != nullptr ? ggml_cast(ctx, b.cache[idx], GGML_TYPE_F32) : nullptr;
            ggml_tensor* xin;
            if (prev == nullptr)
                xin = ggml_pad_ext(ctx, x, 0, 0, 0, 0, 0, 0, 2, 0);
            else
                xin = ggml_concat(ctx, prev, x, 3);
            // cache = trailing 2 frames of x (zero-front-padded when T == 1 on the
            // first cached chunk; extended with the previous cache otherwise)
            ggml_tensor* nc;
            if (T >= 2)
                nc = ggml_view_4d(ctx, x, W, H, C, 2, x->nb[1], x->nb[2], x->nb[3], (T - 2) * x->nb[3]);
            else if (prev != nullptr)
            {
                ggml_tensor* lastPrev = ggml_view_4d(ctx, prev, W, H, C, 1,
                    prev->nb[1], prev->nb[2], prev->nb[3], (prev->ne[3] - 1) * prev->nb[3]);
                nc = ggml_concat(ctx, lastPrev, x, 3);
            }
            else
                nc = ggml_pad_ext(ctx, x, 0, 0, 0, 0, 0, 0, 1, 0);
            b.cache[idx] = ggml_cast(ctx, nc, GGML_TYPE_F16);

            // temporal conv (kd=3, 1x1 spatial): per-tap 1x1 conv over the window
            void* taps[3] = { u.time_conv.tap0, u.time_conv.tap1, u.time_conv.tap2 };
            ggml_tensor* acc = nullptr;
            for (int j = 0; j < 3; j++)
            {
                ggml_tensor* wt = wan_vae_conv_w(b, taps[j], u.time_conv);
                ggml_tensor* win = ggml_view_4d(ctx, xin, xin->ne[0], xin->ne[1], xin->ne[2], T,
                                                xin->nb[1], xin->nb[2], xin->nb[3], j * xin->nb[3]);
                ggml_tensor* y = wan_vae_conv2d(b, wt, win, 0);
                acc = acc == nullptr ? y : ggml_add(ctx, acc, y);
            }
            if (u.time_conv.bias != nullptr)
            {
                ggml_tensor* bt = b.wb->f32v(u.time_conv.bias, u.time_conv.oc);
                acc = ggml_add(ctx, acc, ggml_reshape_4d(ctx, bt, 1, 1, u.time_conv.oc, 1));
            }
            // [W,H,2C,T] -> interleave the channel halves as frame pairs -> [W,H,C,2T].
            // The conv output channel index is half*C + c (torch view(2, c, ...)), so a
            // contiguous reinterpret already yields frame index half + 2*t.
            x = ggml_reshape_4d(ctx, acc, W, H, C, 2 * T);
        }
        // Advance chunk-0 state: leave cache[idx] == nullptr so chunk 1 zero-pads.
    }

    // nearest x2 spatial upsample + 3x3 conv (pad 1)
    x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
    ggml_tensor* wt = wan_vae_conv_w(b, u.sconv.tap0, u.sconv);
    ggml_tensor* y = wan_vae_conv2d(b, wt, x, 1);
    if (u.sconv.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(u.sconv.bias, u.sconv.oc);
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, bt, 1, 1, u.sconv.oc, 1));
    }
    return y;
}

// DupUp3D (wan2.2 Up_ResidualBlock shortcut): channel-grouped nearest upsample.
// diffusers semantics: out[co, t*ft+ift, y*fs+ys, x*fs+xs] = in[c2 / repeats]
// with c2 = co*(ft*fs*fs) + ift*fs*fs + ys*fs + xs. The wan2.2 decoder uses only
// (ft=2, fs=2, in==out) — exact nearest x2 in T/H/W — and (ft=1, fs=2, in==2*out)
// — output row parity selects the input channel pair half, columns duplicate.
ggml_tensor* wan_vae_dup_up(WanVaeBuild& b, ggml_tensor* x, int ft, int fs, bool first_chunk)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
    (void)fs;   // both decoder configurations have fs == 2

    if (ft == 2)
    {
        // temporal duplicate [W,H,C,T] -> [W,H,C,2T] (frame t -> t*2, t*2+1) ...
        ggml_tensor* r = ggml_reshape_3d(ctx, x, W * H * C, 1, T);
        r = ggml_concat(ctx, r, r, 1);                            // [WHC, 2, T]
        x = ggml_reshape_4d(ctx, r, W, H, C, 2 * T);
        // ... then spatial nearest x2
        x = ggml_interpolate(ctx, x, W * 2, H * 2, C, 2 * T, GGML_SCALE_MODE_NEAREST);
        if (first_chunk)
        {
            // causal first chunk keeps only the last of the ft duplicated frames
            x = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], 2 * T - 1,
                             x->nb[1], x->nb[2], x->nb[3], x->nb[3]);
        }
        return x;
    }

    // ft == 1, in == 2*out: out[co, t, 2y+ys, 2x+xs] = in[2*co + ys, t, y, x]
    const std::int64_t Co = C / 2;
    ggml_tensor* r = ggml_reshape_4d(ctx, x, W, H, 2, Co * T);       // isolate ys (channel low bit)
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));            // [W, 2, H, Co*T]
    r = ggml_reshape_4d(ctx, r, W, 2 * H, Co, T);                    // rows interleaved by parity
    return ggml_interpolate(ctx, r, W * 2, 2 * H, Co, T, GGML_SCALE_MODE_NEAREST);
}

// wan2.2 pixel unpatchify: [W, H, 4c, T] -> [2W, 2H, c, T] with channel index
// c*4 + xoff*2 + yoff (diffusers unpatchify order).
ggml_tensor* wan_vae_unpatchify2(WanVaeBuild& b, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t W = x->ne[0], H = x->ne[1], C4 = x->ne[2], T = x->ne[3];
    const std::int64_t C = C4 / 4;
    ggml_tensor* t1 = ggml_reshape_4d(ctx, x, W, H, 2, 2 * C * T);   // isolate yoff
    t1 = ggml_cont(ctx, ggml_permute(ctx, t1, 0, 2, 1, 3));          // [W, 2, H, 2CT]
    t1 = ggml_reshape_4d(ctx, t1, W, 2 * H, 2, C * T);               // rows = h*2+yoff; isolate xoff
    t1 = ggml_cont(ctx, ggml_permute(ctx, t1, 1, 2, 0, 3));          // [2, W, 2H, CT]
    return ggml_reshape_4d(ctx, t1, 2 * W, 2 * H, C, T);             // cols = w*2+xoff
}

// One decoder pass over a single latent frame (chunk). Mirrors Decoder3d::forward:
// conv1, middle (res/attn/res), 4 scales x 3 res blocks with upsamples after
// scales 0/1 (temporal+spatial) and 2 (spatial), head. Wan 2.2 (version 2) adds
// the DupUp3D residual shortcut around each upsampled scale group and a final
// 2x2 pixel unpatchify (12 -> 3 channels).
ggml_tensor* wan_vae_decode_chunk(WanVaeBuild& b, const TSGgmlWanVaeDecodeDesc* d, ggml_tensor* z1)
{
    ggml_context* ctx = b.ctx;
    b.cursor = 0;
    const bool w22 = d->version == 2;

    ggml_tensor* x = wan_vae_causal_conv(b, d->conv1, z1);
    x = wan_vae_res_block(b, d->mid0, x);
    x = wan_vae_attn(b, d->mid1, x);
    x = wan_vae_res_block(b, d->mid2, x);

    for (int scale = 0; scale < 4; scale++)
    {
        ggml_tensor* xin = x;
        for (int r = 0; r < 3; r++)
            x = wan_vae_res_block(b, d->res[scale * 3 + r], x);
        if (scale < 3)
        {
            x = wan_vae_upsample(b, d->up[scale], x);
            if (w22)
            {
                // temperal_upsample = [true, true, false]
                const int ft = scale < 2 ? 2 : 1;
                ggml_tensor* sc = wan_vae_dup_up(b, xin, ft, 2, b.chunk == 0);
                x = ggml_add(ctx, x, sc);
            }
        }
    }

    x = wan_vae_norm(b, d->head_norm, x);
    x = ggml_silu(ctx, x);
    x = wan_vae_causal_conv(b, d->head_conv, x);
    if (w22 && d->patch == 2)
        x = wan_vae_unpatchify2(b, x);
    return x;   // [W*8*patch, H*8*patch, 3, t_out]
}

// Encoder Resample: right/bottom zero-pad + 3x3 stride-2 spatial conv, then
// (downsample3d) a stride-2 temporal conv whose 1-frame cache carries the causal
// context between chunks (diffusers WanResample downsample2d/3d).
ggml_tensor* wan_vae_enc_resample(WanVaeBuild& b, const TSGWanVaeDownW& u, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;

    ggml_tensor* xp = ggml_pad_ext(ctx, x, 0, 1, 0, 1, 0, 0, 0, 0);
    ggml_tensor* wt = wan_vae_conv_w(b, u.sconv.tap0, u.sconv);
    ggml_tensor* y = wan_vae_conv2d(b, wt, xp, 0, 2);
    if (u.sconv.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(u.sconv.bias, u.sconv.oc);
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, bt, 1, 1, u.sconv.oc, 1));
    }

    if (u.tconv.tap0 == nullptr)
        return y;                                                   // downsample2d

    const int idx = b.cursor++;
    const std::int64_t T = y->ne[3];
    if (b.cache[idx] == nullptr)
    {
        // first chunk: store the (single) frame, temporal conv starts next chunk
        b.cache[idx] = ggml_cast(ctx, y, GGML_TYPE_F16);
        return y;
    }

    ggml_tensor* prev = ggml_cast(ctx, b.cache[idx], GGML_TYPE_F32);
    ggml_tensor* lastPrev = ggml_view_4d(ctx, prev, prev->ne[0], prev->ne[1], prev->ne[2], 1,
                                         prev->nb[1], prev->nb[2], prev->nb[3],
                                         (prev->ne[3] - 1) * prev->nb[3]);
    ggml_tensor* xin = ggml_concat(ctx, lastPrev, y, 3);            // [.., T + 1]
    b.cache[idx] = ggml_cast(ctx,
        ggml_view_4d(ctx, y, y->ne[0], y->ne[1], y->ne[2], 1,
                     y->nb[1], y->nb[2], y->nb[3], (T - 1) * y->nb[3]),
        GGML_TYPE_F16);

    // stride-2 temporal conv over the assembled window: out[j] = sum_tap w_tap * xin[2j + tap]
    const std::int64_t T2 = (xin->ne[3] - 3) / 2 + 1;
    void* taps[3] = { u.tconv.tap0, u.tconv.tap1, u.tconv.tap2 };
    ggml_tensor* acc = nullptr;
    for (int j = 0; j < 3; j++)
    {
        ggml_tensor* tw = wan_vae_conv_w(b, taps[j], u.tconv);
        ggml_tensor* win = ggml_view_4d(ctx, xin, xin->ne[0], xin->ne[1], xin->ne[2], T2,
                                        xin->nb[1], xin->nb[2], 2 * xin->nb[3], j * xin->nb[3]);
        win = ggml_cont(ctx, win);
        ggml_tensor* yj = wan_vae_conv2d(b, tw, win, 0);
        acc = acc == nullptr ? yj : ggml_add(ctx, acc, yj);
    }
    if (u.tconv.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(u.tconv.bias, u.tconv.oc);
        acc = ggml_add(ctx, acc, ggml_reshape_4d(ctx, bt, 1, 1, u.tconv.oc, 1));
    }
    return acc;
}

// AvgDown3D (wan2.2 Down_ResidualBlock shortcut). With the wan2.2 encoder's
// group sizes this reduces to a spatial 2x2 average pool plus (ft == 2) a
// temporal-pair channel regroup: out channel c*2 + ift holds frame parity ift
// (front zero-padded to even length, matching diffusers' F.pad).
ggml_tensor* wan_vae_avg_down(WanVaeBuild& b, ggml_tensor* x, int ft, int fs)
{
    ggml_context* ctx = b.ctx;
    if (fs == 2)
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
    if (ft == 2)
    {
        if (x->ne[3] % 2 == 1)
            x = ggml_pad_ext(ctx, x, 0, 0, 0, 0, 0, 0, 1, 0);       // front zero frame
        const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
        ggml_tensor* r = ggml_reshape_4d(ctx, x, W * H, C, 2, T / 2);
        r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));       // [WH, 2, C, T/2]
        x = ggml_reshape_4d(ctx, r, W, H, 2 * C, T / 2);
    }
    return x;
}

// One encoder pass over one pixel chunk (1 frame, then 4-frame chunks). Mirrors
// WanEncoder3d: stem conv, 4 scales x 2 res blocks with downsamples after scales
// 0 (spatial) / 1,2 (temporal+spatial), middle (res/attn/res), head, quant conv;
// returns the posterior mean (first z_dim channels). Wan 2.2 adds the AvgDown3D
// residual shortcut around every scale group.
ggml_tensor* wan_vae_encode_chunk(WanVaeBuild& b, const TSGgmlWanVaeEncodeDesc* d, ggml_tensor* xc)
{
    ggml_context* ctx = b.ctx;
    b.cursor = 0;
    const bool w22 = d->version == 2;

    ggml_tensor* x = wan_vae_causal_conv(b, d->stem, xc);

    for (int scale = 0; scale < 4; scale++)
    {
        ggml_tensor* xin = x;
        for (int r = 0; r < 2; r++)
            x = wan_vae_res_block(b, d->res[scale * 2 + r], x);
        if (scale < 3)
            x = wan_vae_enc_resample(b, d->down[scale], x);
        if (w22)
        {
            // temperal_downsample = [false, true, true]; spatial down at scales 0..2
            const int ft = (scale == 1 || scale == 2) ? 2 : 1;
            const int fs = scale < 3 ? 2 : 1;
            ggml_tensor* sc = (ft == 1 && fs == 1) ? xin : wan_vae_avg_down(b, xin, ft, fs);
            x = ggml_add(ctx, x, sc);
        }
    }

    x = wan_vae_res_block(b, d->mid0, x);
    x = wan_vae_attn(b, d->mid1, x);
    x = wan_vae_res_block(b, d->mid2, x);

    x = wan_vae_norm(b, d->head_norm, x);
    x = ggml_silu(ctx, x);
    x = wan_vae_causal_conv(b, d->head_conv, x);                    // [lw, lh, 2z, t]
    x = wan_vae_causal_conv(b, d->quant, x);

    // posterior mean = first z_dim channels
    return ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], x->ne[1], d->z_dim, x->ne[3],
                                       x->nb[1], x->nb[2], x->nb[3], 0));
}

// Per-chunk graph-node budget for the VAE graphs. The 4096 base covers the
// conv/norm chains at the 832x480 recipe, but the banded im2col conv splits
// (wan_vae_conv2d) add nodes proportional to the full-resolution plane area —
// and inversely to TS_WAN_VAE_GEMM_MAX_MB — so the budget must scale with both
// (a 768x1152 decode overflowed the fixed budget AFTER the full denoise).
// Meta is transient host memory, so overestimating is cheap; underestimating
// is a GGML abort.
static std::size_t wan_vae_chunk_nodes(long long pixels)
{
    long long scale = (pixels + 832LL * 480 - 1) / (832LL * 480);
    const char* e = std::getenv("TS_WAN_VAE_GEMM_MAX_MB");
    const long long mb = e != nullptr ? std::strtoll(e, nullptr, 10) : 384;
    if (mb > 0 && mb < 384)
        scale *= (384 + mb - 1) / mb;
    if (scale < 1) scale = 1;
    if (scale > 64) scale = 64;   // RAM guard: 64x is already ~2 GB of graph meta
    return static_cast<std::size_t>(4096) * static_cast<std::size_t>(scale);
}

bool wan_vae_encode(const TSGgmlWanVaeEncodeDesc* d)
{
    const int pw = d->px_w, ph = d->px_h, pc = d->px_c, pt = d->px_t;
    const int chunks = 1 + (pt - 1 + 3) / 4;

    // pw/ph are the (pre-patchified for wan2.2) pixel grid: x4 restores pixels.
    const std::size_t nodes = static_cast<std::size_t>(chunks) *
        wan_vae_chunk_nodes(static_cast<long long>(pw) * ph * (d->version == 2 ? 4 : 1)) + 4096;
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 4096)
                             + ggml_graph_overhead_custom(nodes, false) + (16u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ContextHandle context(ggml_init(ip));
    if (context.value == nullptr) { set_last_error("WanVaeEncode: ctx alloc failed."); return false; }
    ggml_context* ctx = context.value;

    WanBind wbind; wbind.ctx = ctx; wbind.dev = ggml_backend_get_device(g_backend);
    WanVaeBuild b; b.ctx = ctx; b.wb = &wbind;
    b.cache.assign(64, nullptr);
    b.gemmMax = wan_vae_gemm_budget();

    ggml_tensor* xIn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, pw, ph, pc, pt);
    ggml_set_input(xIn);

    ggml_tensor* out = nullptr;
    for (int i = 0; i < chunks; i++)
    {
        b.chunk = i;
        const int from = i == 0 ? 0 : 1 + 4 * (i - 1);
        const int count = std::min(pt, 1 + 4 * i) - from;
        ggml_tensor* xc = ggml_view_4d(ctx, xIn, pw, ph, pc, count,
                                       xIn->nb[1], xIn->nb[2], xIn->nb[3],
                                       static_cast<std::size_t>(from) * xIn->nb[3]);
        ggml_tensor* mu = wan_vae_encode_chunk(b, d, xc);
        out = out == nullptr ? mu : ggml_concat(ctx, out, mu, 3);
    }

    ggml_tensor* outT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
    ggml_tensor* outc = ggml_cpy(ctx, out, outT);
    ggml_set_output(outc);

    const std::int64_t outElems = out->ne[0] * out->ne[1] * out->ne[2] * out->ne[3];
    if (outElems != d->out_len)
    {
        std::fprintf(stderr, "[wan] vae encode: got [%lld, %lld, %lld, %lld] = %lld, expected %lld\n",
                     (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
                     (long long)outElems, (long long)d->out_len);
        set_last_error("WanVaeEncode: output shape mismatch.");
        return false;
    }

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(graph, outc);
    for (ggml_tensor* c : b.cache)
        if (c != nullptr) ggml_build_forward_expand(graph, c);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr) { set_last_error("WanVaeEncode: gallocr alloc failed."); return false; }
    struct GallocGuard { ggml_gallocr_t g; ~GallocGuard() { if (g) ggml_gallocr_free(g); } } guard{galloc};
    if (!ggml_gallocr_alloc_graph(galloc, graph))
    { set_last_error("WanVaeEncode: graph alloc failed (out of device memory?)."); return false; }

    host_read_barrier();
    for (auto& u : wbind.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    ggml_backend_tensor_set(xIn, d->x, 0, static_cast<std::size_t>(pw) * ph * pc * pt * sizeof(float));

    if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
    { set_last_error("WanVaeEncode: graph compute failed."); return false; }
    ggml_backend_synchronize(g_backend);
    ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(d->out_len) * sizeof(float));
    return true;
}

bool wan_vae_decode(const TSGgmlWanVaeDecodeDesc* d)
{
    const int zw = d->zw, zh = d->zh, zt = d->zt, zc = d->zc;

    // 16x (wan2.2) / 8x (wan2.1) spatial VAE: latent -> pixel area for the budget.
    const std::size_t nodes = static_cast<std::size_t>(zt) *
        wan_vae_chunk_nodes(static_cast<long long>(zw) * zh * (d->version == 2 ? 256 : 64)) + 4096;
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 4096)
                             + ggml_graph_overhead_custom(nodes, false) + (16u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ContextHandle context(ggml_init(ip));
    if (context.value == nullptr) { set_last_error("WanVaeDecode: ctx alloc failed."); return false; }
    ggml_context* ctx = context.value;

    WanBind wbind; wbind.ctx = ctx; wbind.dev = ggml_backend_get_device(g_backend);
    WanVaeBuild b; b.ctx = ctx; b.wb = &wbind;
    b.cache.assign(64, nullptr);
    b.gemmMax = wan_vae_gemm_budget();

    ggml_tensor* zIn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, zw, zh, zc, zt);
    ggml_set_input(zIn);

    // post-quant conv (1x1x1) over the whole latent
    ggml_tensor* z2 = wan_vae_causal_conv(b, d->conv2, zIn);

    ggml_tensor* out = nullptr;
    for (int i = 0; i < zt; i++)
    {
        b.chunk = i;
        ggml_tensor* z1 = ggml_view_4d(ctx, z2, zw, zh, zc, 1, z2->nb[1], z2->nb[2], z2->nb[3], i * z2->nb[3]);
        ggml_tensor* frames = wan_vae_decode_chunk(b, d, z1);
        out = out == nullptr ? frames : ggml_concat(ctx, out, frames, 3);
    }

    ggml_tensor* outT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
    ggml_tensor* outc = ggml_cpy(ctx, out, outT);
    ggml_set_output(outc);

    const std::int64_t outElems = out->ne[0] * out->ne[1] * out->ne[2] * out->ne[3];
    if (outElems != d->out_len)
    { set_last_error("WanVaeDecode: output shape mismatch."); return false; }

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(graph, outc);
    // Keep every cross-chunk cache tensor alive to the end of the graph (its last
    // reader may otherwise free the buffer slot a later chunk still reads).
    for (ggml_tensor* c : b.cache)
        if (c != nullptr) ggml_build_forward_expand(graph, c);

    // The VAE working set is huge (hundreds of MB per tensor at full resolution);
    // keep it in a dedicated gallocr freed at the end of the call rather than
    // growing the shared reuse gallocr to VAE size permanently.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr) { set_last_error("WanVaeDecode: gallocr alloc failed."); return false; }
    struct GallocGuard { ggml_gallocr_t g; ~GallocGuard() { if (g) ggml_gallocr_free(g); } } guard{galloc};
    if (!ggml_gallocr_alloc_graph(galloc, graph))
    { set_last_error("WanVaeDecode: graph alloc failed (out of device memory?)."); return false; }

    host_read_barrier();
    for (auto& u : wbind.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    ggml_backend_tensor_set(zIn, d->z, 0, static_cast<std::size_t>(zw) * zh * zc * zt * sizeof(float));

    if (tsg::graph_compute_profiled(g_backend, graph, "wan vae decode") != GGML_STATUS_SUCCESS)
    { set_last_error("WanVaeDecode: graph compute failed."); return false; }
    ggml_backend_synchronize(g_backend);
    ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(d->out_len) * sizeof(float));
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

extern "C" {

TSG_EXPORT void TSGgml_WanResetForwardCache()
{
    for (auto& e : g_wanDit) e.reset();
    g_wanDitRR = 0;
}

TSG_EXPORT int TSGgml_WanT5Encode(const TSGgmlWanT5Desc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanT5Desc)))
        { set_last_error("WanT5Encode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;
        if (!wan_t5_build_and_run(d)) return 0;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanT5Encode: unknown error."); return 0; }
}

TSG_EXPORT int TSGgml_WanDitForward(const TSGgmlWanDitDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanDitDesc)))
        { set_last_error("WanDitForward: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        if (wan_dit_capture_enabled())
        {
            WanDitPersist* e = nullptr;
            for (auto& c : g_wanDit) if (c.matches(d)) { e = &c; break; }
            if (e == nullptr) e = wan_dit_build_persist(d);
            if (e != nullptr) return wan_dit_run_persist(e, d);
            // build failed (VRAM?): fall through to the rebuild-per-forward path
        }

        const std::size_t nodes = static_cast<std::size_t>(d->num_layers) * 128 + 1024;
        const std::size_t meta = ggml_tensor_overhead() * (nodes + 1024)
                                 + ggml_graph_overhead_custom(nodes, false) + (8u << 20);
        ggml_init_params ip{ meta, nullptr, true };
        ContextHandle context(ggml_init(ip));
        if (context.value == nullptr) { set_last_error("WanDitForward: ctx alloc failed."); return 0; }

        WanDitGraph g;
        if (!wan_dit_build_graph(context.value, d, g))
        { set_last_error("WanDitForward: graph build failed."); return 0; }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(g.graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(context.value, g_backend);
            if (buffer.value == nullptr) { set_last_error("WanDitForward: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : g.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        wan_dit_upload_inputs(g, d);

        if (ggml_backend_graph_compute(g_backend, g.graph) != GGML_STATUS_SUCCESS)
        { set_last_error("WanDitForward: graph compute failed."); return 0; }
        ggml_backend_synchronize(g_backend);
        wan_dit_trace_taps(g);
        ggml_backend_tensor_get(g.outT, d->out, 0, static_cast<std::size_t>(d->head.ne1) * d->seq * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanDitForward: unknown error."); return 0; }
}

TSG_EXPORT int TSGgml_WanVaeDecode(const TSGgmlWanVaeDecodeDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanVaeDecodeDesc)))
        { set_last_error("WanVaeDecode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;
        if (!wan_vae_decode(d)) return 0;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanVaeDecode: unknown error."); return 0; }
}

TSG_EXPORT int TSGgml_WanVaeEncode(const TSGgmlWanVaeEncodeDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanVaeEncodeDesc)))
        { set_last_error("WanVaeEncode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;
        if (!wan_vae_encode(d)) return 0;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanVaeEncode: unknown error."); return 0; }
}

} // extern "C"
