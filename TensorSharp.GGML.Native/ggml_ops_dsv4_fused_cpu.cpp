// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Scalar CPU reference implementations of the fused DeepSeek V4 (Flash) ops.
// These run when a fused GGML_OP_CUSTOM node is scheduled onto the CPU
// backend (ggml_custom_op_t entry point), and serve as the semantics
// reference for the CUDA kernels in ggml_ops_dsv4_fused.cu.
// ---------------------------------------------------------------------------

#include "ggml_ops_dsv4_fused.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

static inline float tsg_dsv4_softplus_f32(float x)
{
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static void tsg_dsv4_cpu_compress(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * st_kv        = dst->src[0];
    const ggml_tensor * st_score     = dst->src[1];
    const ggml_tensor * state_kv     = dst->src[2];
    const ggml_tensor * state_score  = dst->src[3];
    const ggml_tensor * norm_w       = dst->src[4];
    const ggml_tensor * rope_tab     = dst->src[5];
    const ggml_tensor * comp_meta    = dst->src[6];  // I32 [read | write_meta | persist_meta]
    const ggml_tensor * cache        = dst->src[7];

    const int32_t n_blocks = d->i0;
    const int32_t ratio    = d->i1;
    const int32_t coff     = d->i2;
    const int32_t n_rope   = d->i3;
    const float   eps      = d->f0;
    const int64_t np       = (int64_t) d->f1;

    const int64_t head = cache->ne[0];
    const int64_t cw   = (int64_t) coff * head;
    const int64_t ss   = state_kv->ne[1] - 1;

    const float * kv_scr  = (const float *) st_kv->data;
    const float * sc_scr  = (const float *) st_score->data;
    float * ring_kv       = (float *) state_kv->data;
    float * ring_sc       = (float *) state_score->data;
    const float * nw      = (const float *) norm_w->data;
    const float * rt      = (const float *) rope_tab->data;
    const int32_t * meta  = (const int32_t *) comp_meta->data;
    const int32_t * ridx  = meta;
    const int32_t * wmeta = meta + (int64_t) coff * ratio * n_blocks;
    ggml_fp16_t * out     = (ggml_fp16_t *) cache->data;

    std::vector<float> row(head);

    const int W = coff * ratio;
    for (int b = 0; b < n_blocks; ++b)
    {
        const int64_t cache_row = wmeta[2*b + 0];
        const int64_t pos       = wmeta[2*b + 1];

        float ss2 = 0.0f;
        for (int64_t dd = 0; dd < head; ++dd)
        {
            float m = -INFINITY, se = 0.0f, sv = 0.0f;
            for (int w = 0; w < W; ++w)
            {
                const bool prev = coff == 2 && w < ratio;
                const int32_t idx = prev ? ridx[(int64_t) b*ratio + w]
                    : coff == 2 ? ridx[(int64_t) ratio*n_blocks + (int64_t) b*ratio + (w - ratio)]
                                : ridx[(int64_t) b*ratio + w];
                const int64_t off = (coff == 2 && !prev) ? head : 0;
                const bool from_ring = idx <= ss;
                const int64_t so = from_ring ? (int64_t) idx * cw + off + dd
                                             : (int64_t) (idx - ss - 1) * cw + off + dd;
                const float sc = from_ring ? ring_sc[so] : sc_scr[so];
                const float kv = from_ring ? ring_kv[so] : kv_scr[so];
                if (sc > m)
                {
                    const float r = expf(m - sc);
                    se *= r; sv *= r; m = sc;
                }
                const float e = sc == -INFINITY ? 0.0f : expf(sc - m);
                se += e; sv += e * kv;
            }
            const float v = se > 0.0f ? sv/se : 0.0f;
            row[dd] = v;
            ss2 += v*v;
        }
        const float scale = 1.0f / sqrtf(ss2/head + eps);
        for (int64_t dd = 0; dd < head; ++dd)
            row[dd] *= scale * nw[dd];

        const int64_t rbase = head - n_rope;
        for (int64_t dd = 0; dd < head; ++dd)
        {
            float v = row[dd];
            if (dd >= rbase)
            {
                const int64_t i = (dd - rbase) >> 1;
                const float c = rt[pos*n_rope + 2*i + 0];
                const float s = rt[pos*n_rope + 2*i + 1];
                const float x0 = row[rbase + 2*i + 0];
                const float x1 = row[rbase + 2*i + 1];
                v = ((dd - rbase) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
            }
            out[cache_row*head + dd] = ggml_fp32_to_fp16(v);
        }
    }

    const int32_t * pmeta = wmeta + 2 * n_blocks;
    for (int64_t p = 0; p < np; ++p)
    {
        const int64_t src  = pmeta[2*p + 0];
        const int64_t drow = pmeta[2*p + 1];
        memcpy(ring_kv + drow*cw, kv_scr + src*cw, cw*sizeof(float));
        memcpy(ring_sc + drow*cw, sc_scr + src*cw, cw*sizeof(float));
    }
}

static void tsg_dsv4_cpu_attn_prep(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * q_raw     = dst->src[0];
    const ggml_tensor * kv_raw    = dst->src[1];
    const ggml_tensor * kv_norm_w = dst->src[2];
    const ggml_tensor * rope_tab  = dst->src[3];
    const ggml_tensor * pos       = dst->src[4];
    const ggml_tensor * ring      = dst->src[5];
    const ggml_tensor * raw_idxs  = dst->src[6];

    const int32_t n_rope = d->i0;
    const float   eps    = d->f0;

    const int64_t head   = dst->ne[0];
    const int64_t n_head = dst->ne[1];
    const int64_t nt     = dst->ne[2];
    const int64_t rbase  = head - n_rope;

    const float * rt = (const float *) rope_tab->data;
    const int32_t * ps = (const int32_t *) pos->data;
    const int64_t * ridx = (const int64_t *) raw_idxs->data;

    std::vector<float> row(head);

    auto process = [&](const float * src, float * fout, ggml_fp16_t * hout, const float * w, int64_t p)
    {
        float ss2 = 0.0f;
        for (int64_t dd = 0; dd < head; ++dd) ss2 += src[dd]*src[dd];
        const float scale = 1.0f / sqrtf(ss2/head + eps);
        for (int64_t dd = 0; dd < head; ++dd) row[dd] = src[dd] * scale * (w ? w[dd] : 1.0f);
        for (int64_t dd = 0; dd < head; ++dd)
        {
            float v = row[dd];
            if (dd >= rbase)
            {
                const int64_t i = (dd - rbase) >> 1;
                const float c = rt[p*n_rope + 2*i + 0];
                const float s = rt[p*n_rope + 2*i + 1];
                const float x0 = row[rbase + 2*i + 0];
                const float x1 = row[rbase + 2*i + 1];
                v = ((dd - rbase) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
            }
            if (fout) fout[dd] = v;
            else hout[dd] = ggml_fp32_to_fp16(v);
        }
    };

    for (int64_t t = 0; t < nt; ++t)
    {
        const int64_t p = ps[t];
        for (int64_t h = 0; h < n_head; ++h)
        {
            process((const float *) q_raw->data + (t*n_head + h)*head,
                    (float *) dst->data + (t*n_head + h)*head, nullptr, nullptr, p);
        }
        process((const float *) kv_raw->data + t*head, nullptr,
                (ggml_fp16_t *) ring->data + ridx[t]*head,
                (const float *) kv_norm_w->data, p);
    }
}

static void tsg_dsv4_cpu_attn_finish(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * attn     = dst->src[0];
    const ggml_tensor * rope_tab = dst->src[1];
    const ggml_tensor * pos      = dst->src[2];

    const int32_t n_rope   = d->i0;
    const int32_t n_groups = d->i1;
    const int32_t head     = d->i2;

    const int64_t n_head = attn->ne[0] / head;
    const int64_t nt     = attn->ne[1];
    const int64_t hpg    = n_head / n_groups;
    const int64_t gdim   = hpg * head;
    const int64_t rbase  = head - n_rope;

    const float * rt = (const float *) rope_tab->data;
    const int32_t * ps = (const int32_t *) pos->data;

    for (int64_t t = 0; t < nt; ++t)
    {
        const int64_t p = ps[t];
        for (int64_t h = 0; h < n_head; ++h)
        {
            const float * src = (const float *) attn->data + (t*n_head + h)*head;
            float * out = (float *) dst->data + (h/hpg)*gdim*nt + t*gdim + (h%hpg)*head;
            for (int64_t dd = 0; dd < head; ++dd)
            {
                float v = src[dd];
                if (dd >= rbase)
                {
                    const int64_t i = (dd - rbase) >> 1;
                    const float c = rt[p*n_rope + 2*i + 0];
                    const float s = rt[p*n_rope + 2*i + 1];
                    const float x0 = src[rbase + 2*i + 0];
                    const float x1 = src[rbase + 2*i + 1];
                    v = ((dd - rbase) & 1) == 0 ? x0*c + x1*s : -x0*s + x1*c;
                }
                out[dd] = v;
            }
        }
    }
}

static void tsg_dsv4_cpu_moe_topk(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    GGML_UNUSED(d);
    const ggml_tensor * logits = dst->src[0];
    const ggml_tensor * bias   = dst->src[1];

    const int64_t n_expert = logits->ne[0];
    const int64_t n_used   = dst->ne[0];
    const int64_t nt       = dst->ne[1];

    std::vector<float> sel(n_expert);
    const float * bs = (const float *) bias->data;

    for (int64_t t = 0; t < nt; ++t)
    {
        const float * lg = (const float *) logits->data + t*n_expert;
        for (int64_t e = 0; e < n_expert; ++e)
            sel[e] = sqrtf(tsg_dsv4_softplus_f32(lg[e])) + bs[e];
        int32_t * out = (int32_t *) dst->data + t*n_used;
        for (int64_t k = 0; k < n_used; ++k)
        {
            int64_t best = 0;
            for (int64_t e = 1; e < n_expert; ++e)
                if (sel[e] > sel[best]) best = e;
            out[k] = (int32_t) best;
            sel[best] = -INFINITY;
        }
    }
}

static void tsg_dsv4_cpu_moe_weights(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * logits = dst->src[0];
    const ggml_tensor * ids    = dst->src[1];

    const int32_t norm = d->i0;
    const float scale = d->f0;

    const int64_t n_expert = logits->ne[0];
    const int64_t n_used   = dst->ne[1];
    const int64_t nt       = dst->ne[2];

    for (int64_t t = 0; t < nt; ++t)
    {
        const float * lg = (const float *) logits->data + t*n_expert;
        const int32_t * id = (const int32_t *) ids->data + t*n_used;
        float * out = (float *) dst->data + t*n_used;
        float sum = 0.0f;
        for (int64_t e = 0; e < n_used; ++e)
        {
            out[e] = sqrtf(tsg_dsv4_softplus_f32(lg[id[e]]));
            sum += out[e];
        }
        if (norm)
        {
            sum = std::max(sum, 6.103515625e-5f);
            for (int64_t e = 0; e < n_used; ++e) out[e] /= sum;
        }
        for (int64_t e = 0; e < n_used; ++e) out[e] *= scale;
    }
}

static void tsg_dsv4_cpu_expert_reduce(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    GGML_UNUSED(d);
    const ggml_tensor * experts = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * shexp   = dst->src[2];

    const int64_t n_embd = dst->ne[0];
    const int64_t n_used = experts->ne[1];
    const int64_t nt     = dst->ne[1];

    for (int64_t t = 0; t < nt; ++t)
    {
        const float * w = (const float *) weights->data + t*n_used;
        const float * sx = (const float *) shexp->data + t*n_embd;
        float * out = (float *) dst->data + t*n_embd;
        for (int64_t dd = 0; dd < n_embd; ++dd)
        {
            float acc = 0.0f;
            for (int64_t e = 0; e < n_used; ++e)
                acc += ((const float *) experts->data)[t*n_used*n_embd + e*n_embd + dd] * w[e];
            out[dd] = acc + sx[dd];
        }
    }
}

static void tsg_dsv4_cpu_swiglu_clamp(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];

    const float limit = d->f0;
    const int64_t n = ggml_nelements(dst);
    const float * g = (const float *) gate->data;
    const float * u = (const float *) up->data;
    float * out = (float *) dst->data;

    for (int64_t i = 0; i < n; ++i)
    {
        const float gv = std::min(g[i], limit);
        const float uv = std::min(std::max(u[i], -limit), limit);
        out[i] = uv * (gv / (1.0f + expf(-gv)));
    }
}

static void tsg_dsv4_cpu_hc_gates(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    const float eps = d->f0;

    const int64_t hc2 = dst->ne[0];
    const int64_t hc  = hc2 / 2;
    const int64_t nt  = dst->ne[1];

    const int64_t sm1 = mixes->nb[1] / sizeof(float);
    const int64_t ss0 = scale->nb[0] / sizeof(float);
    const int64_t sb0 = base->nb[0] / sizeof(float);

    const float * mx = (const float *) mixes->data;
    const float * sc = (const float *) scale->data;
    const float * bs = (const float *) base->data;
    float * out = (float *) dst->data;

    for (int64_t t = 0; t < nt; ++t)
    {
        for (int64_t r = 0; r < hc2; ++r)
        {
            const bool pre = r < hc;
            const float s = sc[(pre ? 0 : 1)*ss0];
            const float v = 1.0f / (1.0f + expf(-(mx[t*sm1 + r]*s + bs[r*sb0])));
            out[t*hc2 + r] = pre ? v + eps : 2.0f*v;
        }
    }
}

static void tsg_dsv4_cpu_topk_mask(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * base = dst->src[0];
    const ggml_tensor * topk = dst->src[1];

    const int32_t offset = d->i0;

    const int64_t W  = dst->ne[0];
    const int64_t nt = dst->ne[1];
    const int64_t k  = topk->ne[0];

    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

    for (int64_t t = 0; t < nt; ++t)
    {
        const ggml_fp16_t * bs = (const ggml_fp16_t *) base->data + t*W;
        ggml_fp16_t * out = (ggml_fp16_t *) dst->data + t*W;
        for (int64_t r = 0; r < W; ++r)
            out[r] = r < offset ? bs[r] : neg_inf;
        const int32_t * tk = (const int32_t *) topk->data + t*k;
        for (int64_t i = 0; i < k; ++i)
        {
            const int64_t r = offset + tk[i];
            if (r >= offset && r < W)
                out[r] = bs[r];
        }
    }
}

static void tsg_dsv4_cpu_kgather(ggml_tensor * dst, const tsg_dsv4_fused_desc * d)
{
    const ggml_tensor * ring = dst->src[0];
    const ggml_tensor * comp = dst->src[1];
    const ggml_tensor * topk = dst->src[2];

    const int64_t head      = dst->ne[0];
    const int64_t n_rows    = dst->ne[2];
    const int64_t ring_rows = d->i0;

    const int32_t * tk = (const int32_t *) topk->data;

    for (int64_t r = 0; r < n_rows; ++r)
    {
        const ggml_fp16_t * src = r < ring_rows
            ? (const ggml_fp16_t *) ring->data + r * head
            : (const ggml_fp16_t *) comp->data + (int64_t) tk[r - ring_rows] * head;
        memcpy((ggml_fp16_t *) dst->data + r * head, src, head * sizeof(ggml_fp16_t));
    }
}

void tsg_dsv4_fused_cpu(ggml_tensor * dst, int ith, int nth, void * userdata)
{
    GGML_UNUSED(nth);
    if (ith != 0) return;

    const auto * d = (const tsg_dsv4_fused_desc *) userdata;
    GGML_ASSERT(d && d->magic == TSG_DSV4_FUSED_MAGIC);

    switch (d->kind)
    {
        case TSG_DSV4_FUSED_COMPRESS:      tsg_dsv4_cpu_compress(dst, d); break;
        case TSG_DSV4_FUSED_ATTN_PREP:     tsg_dsv4_cpu_attn_prep(dst, d); break;
        case TSG_DSV4_FUSED_ATTN_FINISH:   tsg_dsv4_cpu_attn_finish(dst, d); break;
        case TSG_DSV4_FUSED_MOE_TOPK:      tsg_dsv4_cpu_moe_topk(dst, d); break;
        case TSG_DSV4_FUSED_MOE_WEIGHTS:   tsg_dsv4_cpu_moe_weights(dst, d); break;
        case TSG_DSV4_FUSED_EXPERT_REDUCE: tsg_dsv4_cpu_expert_reduce(dst, d); break;
        case TSG_DSV4_FUSED_SWIGLU_CLAMP:  tsg_dsv4_cpu_swiglu_clamp(dst, d); break;
        case TSG_DSV4_FUSED_HC_GATES:      tsg_dsv4_cpu_hc_gates(dst, d); break;
        case TSG_DSV4_FUSED_TOPK_MASK:     tsg_dsv4_cpu_topk_mask(dst, d); break;
        case TSG_DSV4_FUSED_KGATHER:       tsg_dsv4_cpu_kgather(dst, d); break;
        default: GGML_ABORT("tsg_dsv4_fused_cpu: unknown kind %d", d->kind);
    }
}
