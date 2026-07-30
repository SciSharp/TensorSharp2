// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Fused DeepSeek V4 (Flash) CUDA kernels + the TensorSharp fused-op backend.
//
// The DSV4 decode graph is launch-bound: ~7.8K ggml nodes per token, most of
// them tiny elementwise/view chains. These kernels collapse the hot chains
// (compressors, attention prologue/epilogue, MoE routing/reduction, clamped
// SwiGLU, hyper-connection gates, top-k visibility masks) into single
// launches.
//
// Injection works without touching the vendored ggml: the graph builder emits
// GGML_OP_CUSTOM nodes (public ggml_custom_4d API) whose userdata points at a
// tsg_dsv4_fused_desc, and a minimal ggml-backend implemented here claims
// those nodes in ggml_backend_sched. The backend reports the paired CUDA
// device's default buffer type as its own, so the scheduler interleaves
// CUDA-backend and fused-backend splits with no tensor copies and no
// synchronization; kernels are launched on the ggml-cuda backend's own stream
// (obtained from its context), so plain stream order guarantees correctness
// and ggml-cuda's per-split CUDA-graph capture keeps working around us.
// ---------------------------------------------------------------------------

#include "ggml_ops_dsv4_fused.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"   // ggml_backend_cuda_context (stream access)

#include <cstring>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float tsg_dsv4_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float tsg_dsv4_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// per-element softmax-weighted window compression from [state ring | scratch]
// sources, RMS norm, table-driven RoPE, F16 cache commit — one thread block
// per compressed row — followed by the state-ring persist update.
static __global__ void tsg_dsv4_compress_f32(
        const float * __restrict__ kv_scr,
        const float * __restrict__ sc_scr,
        const float * __restrict__ ring_kv,
        const float * __restrict__ ring_sc,
        const float * __restrict__ norm_w,
        const float * __restrict__ rope_tab,
        const int32_t * __restrict__ read_idxs,
        const int32_t * __restrict__ write_meta,
        half * __restrict__ cache,
        const int n_blocks,
        const int ratio,
        const int coff,
        const int head,
        const int n_rope,
        const float eps,
        const int ss) {
    const int b = blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    const int64_t cw = (int64_t) coff * head;
    const int W = coff * ratio;

    const int64_t cache_row = write_meta[2*b + 0];
    const int64_t pos       = write_meta[2*b + 1];

    __shared__ float sh_out[512];
    __shared__ float sh_red[256];

    float acc2 = 0.0f;

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        float m  = -INFINITY;
        float se = 0.0f;
        float sv = 0.0f;

        for (int w = 0; w < W; ++w) {
            const bool prev = coff == 2 && w < ratio;
            const int32_t idx = prev ? read_idxs[(int64_t) b*ratio + w]
                : coff == 2 ? read_idxs[(int64_t) ratio*n_blocks + (int64_t) b*ratio + (w - ratio)]
                            : read_idxs[(int64_t) b*ratio + w];
            const int64_t off = (coff == 2 && !prev) ? head : 0;

            const bool from_ring = idx <= ss;
            const int64_t src_off = from_ring ? (int64_t) idx * cw + off + d
                                              : (int64_t) (idx - ss - 1) * cw + off + d;
            const float sc = from_ring ? ring_sc[src_off] : sc_scr[src_off];
            const float kv = from_ring ? ring_kv[src_off] : kv_scr[src_off];

            if (sc > m) {
                const float r = expf(m - sc); // 0 when m == -inf
                se *= r;
                sv *= r;
                m = sc;
            }
            // guard the -inf boundary rows: expf(-inf - -inf) would be NaN
            const float e = sc == -INFINITY ? 0.0f : expf(sc - m);
            se += e;
            sv += e * kv;
        }

        const float out = se > 0.0f ? sv/se : 0.0f;
        sh_out[d] = out;
        acc2 += out*out;
    }

    // block-wide sum of squares for the RMS norm
    sh_red[threadIdx.x] = acc2;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_red[threadIdx.x] += sh_red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(sh_red[0]/head + eps);

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        sh_out[d] *= scale * norm_w[d];
    }
    __syncthreads();

    // RoPE (normal/interleaved pairs) on the last n_rope dims + F16 store
    const int base = head - n_rope;
    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        float v = sh_out[d];
        if (d >= base) {
            const int i = (d - base) >> 1;
            const float c = rope_tab[pos*n_rope + 2*i + 0];
            const float s = rope_tab[pos*n_rope + 2*i + 1];
            const float x0 = sh_out[base + 2*i + 0];
            const float x1 = sh_out[base + 2*i + 1];
            v = ((d - base) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
        }
        cache[cache_row*head + d] = __float2half(v);
    }
}

static __global__ void tsg_dsv4_persist_f32(
        const float * __restrict__ kv_scr,
        const float * __restrict__ sc_scr,
        float * __restrict__ ring_kv,
        float * __restrict__ ring_sc,
        const int32_t * __restrict__ persist_meta,
        const int np,
        const int64_t cw) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t) np * cw) {
        return;
    }

    const int p = (int) (t / cw);
    const int64_t d = t % cw;

    const int64_t src  = persist_meta[2*p + 0];
    const int64_t drow = persist_meta[2*p + 1];

    ring_kv[drow*cw + d] = kv_scr[src*cw + d];
    ring_sc[drow*cw + d] = sc_scr[src*cw + d];
}

// blockIdx.x in [0, n_head]: heads 0..n_head-1 normalize+rope q into dst,
// block n_head normalizes+ropes kv and commits it (F16) into the SWA ring.
static __global__ void tsg_dsv4_attn_prep_f32(
        const float * __restrict__ q_raw,     // [head*n_head, nt]
        const float * __restrict__ kv_raw,    // [head, nt]
        const float * __restrict__ kv_norm_w, // [head]
        const float * __restrict__ rope_tab,  // [n_rope, n_ctx]
        const int32_t * __restrict__ pos,     // [nt]
        half * __restrict__ ring,             // [head, ring_rows]
        const int64_t * __restrict__ raw_idxs,// [nt]
        float * __restrict__ dst,             // [head, n_head, nt]
        const int n_head,
        const int head,
        const int n_rope,
        const float eps) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[512];
    __shared__ float sh_red[256];

    const bool is_kv = h == n_head;
    const float * src = is_kv ? kv_raw + (int64_t) t * head
                              : q_raw + (int64_t) t * head * n_head + (int64_t) h * head;

    float acc2 = 0.0f;
    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        const float v = src[d];
        sh[d] = v;
        acc2 += v*v;
    }
    sh_red[threadIdx.x] = acc2;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_red[threadIdx.x] += sh_red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(sh_red[0]/head + eps);

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        sh[d] *= is_kv ? scale * kv_norm_w[d] : scale;
    }
    __syncthreads();

    const int rbase = head - n_rope;
    const int64_t p = pos[t];

    if (is_kv) {
        half * out = ring + raw_idxs[t] * head;
        for (int d = threadIdx.x; d < head; d += blockDim.x) {
            float v = sh[d];
            if (d >= rbase) {
                const int i = (d - rbase) >> 1;
                const float c = rope_tab[p*n_rope + 2*i + 0];
                const float s = rope_tab[p*n_rope + 2*i + 1];
                const float x0 = sh[rbase + 2*i + 0];
                const float x1 = sh[rbase + 2*i + 1];
                v = ((d - rbase) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
            }
            out[d] = __float2half(v);
        }
    } else {
        float * out = dst + (int64_t) t * head * n_head + (int64_t) h * head;
        for (int d = threadIdx.x; d < head; d += blockDim.x) {
            float v = sh[d];
            if (d >= rbase) {
                const int i = (d - rbase) >> 1;
                const float c = rope_tab[p*n_rope + 2*i + 0];
                const float s = rope_tab[p*n_rope + 2*i + 1];
                const float x0 = sh[rbase + 2*i + 0];
                const float x1 = sh[rbase + 2*i + 1];
                v = ((d - rbase) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
            }
            out[d] = v;
        }
    }
}

// inverse RoPE on the tail dims of each attention output head + store in the
// grouped output layout [group_dim, nt, n_groups].
static __global__ void tsg_dsv4_attn_finish_f32(
        const float * __restrict__ attn,     // [head*n_head, nt]
        const float * __restrict__ rope_tab, // [n_rope, n_ctx]
        const int32_t * __restrict__ pos,    // [nt]
        float * __restrict__ dst,            // [group_dim, nt, n_groups]
        const int n_head,
        const int head,
        const int n_rope,
        const int heads_per_group,
        const int nt) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[512];

    const float * src = attn + (int64_t) t * head * n_head + (int64_t) h * head;
    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        sh[d] = src[d];
    }
    __syncthreads();

    const int rbase = head - n_rope;
    const int64_t p = pos[t];
    const int g  = h / heads_per_group;
    const int hg = h % heads_per_group;
    const int64_t group_dim = (int64_t) heads_per_group * head;

    float * out = dst + (int64_t) g * group_dim * nt + (int64_t) t * group_dim + (int64_t) hg * head;

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        float v = sh[d];
        if (d >= rbase) {
            // inverse (transpose) rotation of the forward RoPE
            const int i = (d - rbase) >> 1;
            const float c = rope_tab[p*n_rope + 2*i + 0];
            const float s = rope_tab[p*n_rope + 2*i + 1];
            const float x0 = sh[rbase + 2*i + 0];
            const float x1 = sh[rbase + 2*i + 1];
            v = ((d - rbase) & 1) == 0 ? x0*c + x1*s : -x0*s + x1*c;
        }
        out[d] = v;
    }
}

// sel = sqrt(softplus(logits)) + bias; ids = top-k (descending, ties to the
// lower index). One block of 256 threads per token, n_expert <= 1024.
static __global__ void tsg_dsv4_moe_topk_f32(
        const float * __restrict__ logits, // [n_expert, nt]
        const float * __restrict__ bias,   // [n_expert]
        int32_t * __restrict__ dst,        // [n_used, nt]
        const int n_expert,
        const int n_used) {
    const int t = blockIdx.x;

    extern __shared__ float sh_sel[]; // [n_expert]
    __shared__ float sh_val[256];
    __shared__ int   sh_idx[256];

    const float * lg = logits + (int64_t) t * n_expert;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        sh_sel[e] = sqrtf(tsg_dsv4_softplus(lg[e])) + bias[e];
    }
    __syncthreads();

    for (int k = 0; k < n_used; ++k) {
        float best = -INFINITY;
        int bidx = -1;
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
            const float v = sh_sel[e];
            if (v > best || (v == best && e < bidx)) {
                best = v;
                bidx = e;
            }
        }
        sh_val[threadIdx.x] = best;
        sh_idx[threadIdx.x] = bidx;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                const float ov = sh_val[threadIdx.x + s];
                const int   oi = sh_idx[threadIdx.x + s];
                if (ov > sh_val[threadIdx.x] || (ov == sh_val[threadIdx.x] && oi >= 0 &&
                        (sh_idx[threadIdx.x] < 0 || oi < sh_idx[threadIdx.x]))) {
                    sh_val[threadIdx.x] = ov;
                    sh_idx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int sel = sh_idx[0] < 0 ? 0 : sh_idx[0];
            dst[(int64_t) t * n_used + k] = sel;
            sh_sel[sel] = -INFINITY;
        }
        __syncthreads();
    }
}

// w_e = sqrt(softplus(logits[ids_e])); optional sum-normalization; scale.
static __global__ void tsg_dsv4_moe_weights_f32(
        const float * __restrict__ logits, // [n_expert, nt]
        const int32_t * __restrict__ ids,  // [n_used, nt]
        float * __restrict__ dst,          // [1, n_used, nt]
        const int n_expert,
        const int n_used,
        const int norm,
        const float scale) {
    const int t = blockIdx.x;
    const int e = threadIdx.x;

    float w = 0.0f;
    if (e < n_used) {
        const int32_t id = ids[(int64_t) t * n_used + e];
        w = sqrtf(tsg_dsv4_softplus(logits[(int64_t) t * n_expert + id]));
    }

    if (norm) {
        float sum = w;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off, 32);
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        w /= sum;
    }

    if (e < n_used) {
        dst[(int64_t) t * n_used + e] = w * scale;
    }
}

// dst = sum_e experts[:,e,t]*w[e,t] + shexp[:,t]
static __global__ void tsg_dsv4_expert_reduce_f32(
        const float * __restrict__ experts, // [n_embd, n_used, nt]
        const float * __restrict__ weights, // [1, n_used, nt]
        const float * __restrict__ shexp,   // [n_embd, nt]
        float * __restrict__ dst,           // [n_embd, nt]
        const int n_embd,
        const int n_used) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) {
        return;
    }

    const float * ex = experts + (int64_t) t * n_embd * n_used + d;
    const float * w  = weights + (int64_t) t * n_used;

    float acc = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        acc += ex[(int64_t) e * n_embd] * w[e];
    }
    acc += shexp[(int64_t) t * n_embd + d];
    dst[(int64_t) t * n_embd + d] = acc;
}

static __global__ void tsg_dsv4_swiglu_clamp_f32(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ dst,
        const int64_t n,
        const float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    dst[i] = u * (g / (1.0f + expf(-g)));
}

// dst rows [0,hc): sigmoid(mixes_pre*scale0 + base_pre) + eps
// dst rows [hc,2hc): 2*sigmoid(mixes_post*scale1 + base_post)
static __global__ void tsg_dsv4_hc_gates_f32(
        const float * __restrict__ mixes, // [(2+hc)*hc, nt]
        const float * __restrict__ scale, // [3]
        const float * __restrict__ base,  // [(2+hc)*hc]
        float * __restrict__ dst,         // [2*hc, nt]
        const int64_t n,
        const int hc,
        const int64_t sm1,
        const int64_t ss0,
        const int64_t sb0,
        const float eps) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t t = i / (2*hc);
    const int     r = (int) (i % (2*hc));
    const bool  pre = r < hc;

    const float sc = scale[(pre ? 0 : 1) * ss0];
    const float b  = base[(int64_t) r * sb0];
    const float v  = tsg_dsv4_sigmoid(mixes[t*sm1 + r] * sc + b);

    dst[t*2*hc + r] = pre ? v + eps : 2.0f*v;
}

// dst[r,t] = base[r,t] for r < offset (raw window region);
//            base[r,t] where (r-offset) in top_k[:,t], else -inf.
static __global__ void tsg_dsv4_topk_mask_f16(
        const half * __restrict__ base,   // [W, nt]
        const int32_t * __restrict__ topk,// [k, nt]
        half * __restrict__ dst,          // [W, nt]
        const int W,
        const int k,
        const int offset) {
    const int t = blockIdx.x;

    const half * bs = base + (int64_t) t * W;
    half * out      = dst + (int64_t) t * W;

    const half neg_inf = __float2half(-INFINITY);
    for (int r = threadIdx.x; r < W; r += blockDim.x) {
        out[r] = r < offset ? bs[r] : neg_inf;
    }
    __syncthreads();

    const int32_t * tk = topk + (int64_t) t * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int r = offset + tk[i];
        if (r >= offset && r < W) {
            out[r] = bs[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

static void tsg_dsv4_fused_launch(const tsg_dsv4_fused_desc * d, ggml_tensor * dst, cudaStream_t stream)
{
    switch (d->kind)
    {
        case TSG_DSV4_FUSED_COMPRESS:
        {
            const ggml_tensor * st_kv        = dst->src[0];
            const ggml_tensor * st_score     = dst->src[1];
            const ggml_tensor * state_kv     = dst->src[2];
            const ggml_tensor * state_score  = dst->src[3];
            const ggml_tensor * norm_w       = dst->src[4];
            const ggml_tensor * rope_tab     = dst->src[5];
            const ggml_tensor * comp_meta    = dst->src[6];  // I32 [read | write_meta | persist_meta]
            const ggml_tensor * cache        = dst->src[7];

            const int n_blocks = d->i0;
            const int ratio    = d->i1;
            const int coff     = d->i2;
            const int n_rope   = d->i3;
            const float eps    = d->f0;
            const int np       = (int) (d->f1);   // persist count

            const int64_t head = cache->ne[0];
            const int64_t cw   = (int64_t) coff * head;
            const int     ss   = (int) (state_kv->ne[1] - 1);

            const int32_t * meta = (const int32_t *) comp_meta->data;
            const int32_t * read_idxs    = meta;
            const int32_t * write_meta   = meta + (int64_t) coff * ratio * n_blocks;
            const int32_t * persist_meta = write_meta + 2 * n_blocks;

            if (n_blocks > 0) {
                // the compression must read the ring before the persist below updates it
                tsg_dsv4_compress_f32<<<n_blocks, 256, 0, stream>>>(
                    (const float *) st_kv->data, (const float *) st_score->data,
                    (const float *) state_kv->data, (const float *) state_score->data,
                    (const float *) norm_w->data, (const float *) rope_tab->data,
                    read_idxs, write_meta,
                    (half *) cache->data,
                    n_blocks, ratio, coff, (int) head, n_rope, eps, ss);
            }
            if (np > 0) {
                const int64_t total = (int64_t) np * cw;
                const int nblk = (int) ((total + 255) / 256);
                tsg_dsv4_persist_f32<<<nblk, 256, 0, stream>>>(
                    (const float *) st_kv->data, (const float *) st_score->data,
                    (float *) state_kv->data, (float *) state_score->data,
                    persist_meta, np, cw);
            }
        } break;

        case TSG_DSV4_FUSED_ATTN_PREP:
        {
            const ggml_tensor * q_raw     = dst->src[0];
            const ggml_tensor * kv_raw    = dst->src[1];
            const ggml_tensor * kv_norm_w = dst->src[2];
            const ggml_tensor * rope_tab  = dst->src[3];
            const ggml_tensor * pos       = dst->src[4];
            const ggml_tensor * ring      = dst->src[5];
            const ggml_tensor * raw_idxs  = dst->src[6];

            const int head   = (int) dst->ne[0];
            const int n_head = (int) dst->ne[1];
            const int nt     = (int) dst->ne[2];

            const dim3 grid(n_head + 1, nt, 1);
            tsg_dsv4_attn_prep_f32<<<grid, 256, 0, stream>>>(
                (const float *) q_raw->data, (const float *) kv_raw->data,
                (const float *) kv_norm_w->data, (const float *) rope_tab->data,
                (const int32_t *) pos->data,
                (half *) ring->data, (const int64_t *) raw_idxs->data,
                (float *) dst->data,
                n_head, head, d->i0 /*n_rope*/, d->f0 /*eps*/);
        } break;

        case TSG_DSV4_FUSED_ATTN_FINISH:
        {
            const ggml_tensor * attn     = dst->src[0];
            const ggml_tensor * rope_tab = dst->src[1];
            const ggml_tensor * pos      = dst->src[2];

            const int n_rope   = d->i0;
            const int n_groups = d->i1;
            const int head     = d->i2;

            const int nt    = (int) dst->ne[1];
            const int heads = (int) (attn->ne[0] / head);
            const int heads_per_group = heads / n_groups;

            const dim3 grid(heads, nt, 1);
            tsg_dsv4_attn_finish_f32<<<grid, 256, 0, stream>>>(
                (const float *) attn->data, (const float *) rope_tab->data, (const int32_t *) pos->data,
                (float *) dst->data,
                heads, head, n_rope, heads_per_group, nt);
        } break;

        case TSG_DSV4_FUSED_MOE_TOPK:
        {
            const ggml_tensor * logits = dst->src[0];
            const ggml_tensor * bias   = dst->src[1];

            const int n_expert = (int) logits->ne[0];
            const int n_used   = (int) dst->ne[0];
            const int nt       = (int) dst->ne[1];

            tsg_dsv4_moe_topk_f32<<<nt, 256, n_expert*sizeof(float), stream>>>(
                (const float *) logits->data, (const float *) bias->data,
                (int32_t *) dst->data, n_expert, n_used);
        } break;

        case TSG_DSV4_FUSED_MOE_WEIGHTS:
        {
            const ggml_tensor * logits = dst->src[0];
            const ggml_tensor * ids    = dst->src[1];

            const int n_expert = (int) logits->ne[0];
            const int n_used   = (int) dst->ne[1];
            const int nt       = (int) dst->ne[2];

            tsg_dsv4_moe_weights_f32<<<nt, 32, 0, stream>>>(
                (const float *) logits->data, (const int32_t *) ids->data,
                (float *) dst->data, n_expert, n_used, d->i0 /*norm*/, d->f0 /*scale*/);
        } break;

        case TSG_DSV4_FUSED_EXPERT_REDUCE:
        {
            const ggml_tensor * experts = dst->src[0];
            const ggml_tensor * weights = dst->src[1];
            const ggml_tensor * shexp   = dst->src[2];

            const int n_embd = (int) dst->ne[0];
            const int n_used = (int) experts->ne[1];
            const int nt     = (int) dst->ne[1];

            const dim3 grid((n_embd + 255) / 256, nt, 1);
            tsg_dsv4_expert_reduce_f32<<<grid, 256, 0, stream>>>(
                (const float *) experts->data, (const float *) weights->data,
                (const float *) shexp->data, (float *) dst->data, n_embd, n_used);
        } break;

        case TSG_DSV4_FUSED_SWIGLU_CLAMP:
        {
            const ggml_tensor * gate = dst->src[0];
            const ggml_tensor * up   = dst->src[1];

            const int64_t n = ggml_nelements(dst);
            const int nblk = (int) ((n + 255) / 256);
            tsg_dsv4_swiglu_clamp_f32<<<nblk, 256, 0, stream>>>(
                (const float *) gate->data, (const float *) up->data, (float *) dst->data, n, d->f0 /*limit*/);
        } break;

        case TSG_DSV4_FUSED_HC_GATES:
        {
            const ggml_tensor * mixes = dst->src[0];
            const ggml_tensor * scale = dst->src[1];
            const ggml_tensor * base  = dst->src[2];

            const int hc = (int) (dst->ne[0] / 2);
            const int64_t nt = dst->ne[1];
            const int64_t n = 2*hc*nt;

            tsg_dsv4_hc_gates_f32<<<(int) ((n + 255)/256), 256, 0, stream>>>(
                (const float *) mixes->data, (const float *) scale->data, (const float *) base->data,
                (float *) dst->data, n, hc,
                (int64_t) (mixes->nb[1]/sizeof(float)),
                (int64_t) (scale->nb[0]/sizeof(float)),
                (int64_t) (base->nb[0]/sizeof(float)),
                d->f0 /*eps*/);
        } break;

        case TSG_DSV4_FUSED_TOPK_MASK:
        {
            const ggml_tensor * base = dst->src[0];
            const ggml_tensor * topk = dst->src[1];

            const int W  = (int) dst->ne[0];
            const int nt = (int) dst->ne[1];
            const int k  = (int) topk->ne[0];

            tsg_dsv4_topk_mask_f16<<<nt, 256, 0, stream>>>(
                (const half *) base->data, (const int32_t *) topk->data,
                (half *) dst->data, W, k, d->i0 /*offset*/);
        } break;

        default:
            GGML_ABORT("tsg_dsv4_fused: unknown kind %d", d->kind);
    }
}

// ---------------------------------------------------------------------------
// Fused-op backend
// ---------------------------------------------------------------------------

struct tsg_dsv4_backend_ctx
{
    int device = 0;
    ggml_backend_t cuda_backend = nullptr;           // paired instance (not owned)
    ggml_backend_buffer_type_t cuda_buft = nullptr;
    char name[32] = {};
    char desc[64] = {};
    ggml_backend_dev_t cuda_dev = nullptr;
};

static const tsg_dsv4_fused_desc * tsg_dsv4_node_desc(const ggml_tensor * node)
{
    if (node->op != GGML_OP_CUSTOM) return nullptr;
    ggml_custom_op_params p;
    memcpy(&p, node->op_params, sizeof(p));
    const tsg_dsv4_fused_desc * d = (const tsg_dsv4_fused_desc *) p.userdata;
    if (!d || d->magic != TSG_DSV4_FUSED_MAGIC) return nullptr;
    return d;
}

static cudaStream_t tsg_dsv4_backend_stream(tsg_dsv4_backend_ctx * c)
{
    auto * cc = (ggml_backend_cuda_context *) c->cuda_backend->context;
    return cc->stream(c->device, 0);
}

// ---- backend iface ----

static const char * tsg_dsv4_backend_get_name(ggml_backend_t backend)
{
    return ((tsg_dsv4_backend_ctx *) backend->context)->name;
}

static void tsg_dsv4_backend_free(ggml_backend_t backend)
{
    delete (tsg_dsv4_backend_ctx *) backend->context;
    delete backend;
}

static void tsg_dsv4_backend_synchronize(ggml_backend_t backend)
{
    auto * c = (tsg_dsv4_backend_ctx *) backend->context;
    cudaSetDevice(c->device);
    cudaStreamSynchronize(tsg_dsv4_backend_stream(c));
}

static enum ggml_status tsg_dsv4_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)
{
    auto * c = (tsg_dsv4_backend_ctx *) backend->context;
    cudaSetDevice(c->device);
    cudaStream_t stream = tsg_dsv4_backend_stream(c);

    for (int i = 0; i < cgraph->n_nodes; i++)
    {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op)
        {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            default:
                break;
        }
        const tsg_dsv4_fused_desc * d = tsg_dsv4_node_desc(node);
        if (!d)
        {
            GGML_LOG_ERROR("%s: unexpected op %s in fused split\n", __func__, ggml_op_desc(node));
            return GGML_STATUS_FAILED;
        }
        tsg_dsv4_fused_launch(d, node, stream);
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i tsg_dsv4_backend_iface = {
    /* .get_name            = */ tsg_dsv4_backend_get_name,
    /* .free                = */ tsg_dsv4_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ tsg_dsv4_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ tsg_dsv4_backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// ---- device iface ----

static const char * tsg_dsv4_dev_get_name(ggml_backend_dev_t dev)
{
    return ((tsg_dsv4_backend_ctx *) dev->context)->name;
}

static const char * tsg_dsv4_dev_get_description(ggml_backend_dev_t dev)
{
    return ((tsg_dsv4_backend_ctx *) dev->context)->desc;
}

static void tsg_dsv4_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    ggml_backend_dev_memory(c->cuda_dev, free, total);
}

static enum ggml_backend_dev_type tsg_dsv4_dev_get_type(ggml_backend_dev_t dev)
{
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void tsg_dsv4_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props)
{
    memset(props, 0, sizeof(*props));
    props->name = tsg_dsv4_dev_get_name(dev);
    props->description = tsg_dsv4_dev_get_description(dev);
    props->type = tsg_dsv4_dev_get_type(dev);
    tsg_dsv4_dev_get_memory(dev, &props->memory_free, &props->memory_total);
}

static ggml_backend_t tsg_dsv4_dev_init_backend(ggml_backend_dev_t dev, const char * params)
{
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return nullptr; // instances are created via tsg_dsv4_fused_backend_init
}

static ggml_backend_buffer_type_t tsg_dsv4_dev_get_buffer_type(ggml_backend_dev_t dev)
{
    return ((tsg_dsv4_backend_ctx *) dev->context)->cuda_buft;
}

static bool tsg_dsv4_dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op)
{
    GGML_UNUSED(dev);
    return tsg_dsv4_node_desc(op) != nullptr;
}

static bool tsg_dsv4_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft)
{
    return buft == ((tsg_dsv4_backend_ctx *) dev->context)->cuda_buft;
}

static const ggml_backend_device_i tsg_dsv4_device_iface = {
    /* .get_name             = */ tsg_dsv4_dev_get_name,
    /* .get_description      = */ tsg_dsv4_dev_get_description,
    /* .get_memory           = */ tsg_dsv4_dev_get_memory,
    /* .get_type             = */ tsg_dsv4_dev_get_type,
    /* .get_props            = */ tsg_dsv4_dev_get_props,
    /* .init_backend         = */ tsg_dsv4_dev_init_backend,
    /* .get_buffer_type      = */ tsg_dsv4_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ tsg_dsv4_dev_supports_op,
    /* .supports_buft        = */ tsg_dsv4_dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_guid_t tsg_dsv4_backend_guid()
{
    static ggml_guid guid = { 0x7d, 0x54, 0x53, 0x44, 0x53, 0x56, 0x34, 0x46, 0x55, 0x53, 0x45, 0x44, 0x42, 0x4b, 0x4e, 0x44 };
    return &guid;
}

ggml_backend_t tsg_dsv4_fused_backend_init(ggml_backend_t cuda_backend)
{
    if (!cuda_backend || !ggml_backend_is_cuda(cuda_backend))
    {
        return nullptr;
    }

    auto * cc = (ggml_backend_cuda_context *) cuda_backend->context;

    auto * ctx = new tsg_dsv4_backend_ctx();
    ctx->device = cc->device;
    ctx->cuda_backend = cuda_backend;
    ctx->cuda_dev = ggml_backend_get_device(cuda_backend);
    ctx->cuda_buft = ggml_backend_get_default_buffer_type(cuda_backend);
    snprintf(ctx->name, sizeof(ctx->name), "TSDSV4-%d", ctx->device);
    snprintf(ctx->desc, sizeof(ctx->desc), "TensorSharp DSV4 fused ops (CUDA%d)", ctx->device);

    // one static device record per CUDA device index
    static ggml_backend_device devices[GGML_CUDA_MAX_DEVICES];
    ggml_backend_device * dev = &devices[ctx->device];
    dev->iface = tsg_dsv4_device_iface;
    dev->reg = nullptr;
    dev->context = ctx;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ tsg_dsv4_backend_guid(),
        /* .iface   = */ tsg_dsv4_backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}
