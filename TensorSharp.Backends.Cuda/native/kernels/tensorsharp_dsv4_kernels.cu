// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) kernels for the direct-CUDA backend.
//
// These implement the model-specific stages of the DSV4 forward pass
// (4-stream hyper-connections, shared single-KV-head attention with sinks and
// inverse RoPE, CSA/HCA block compressors, the lightning indexer + top-k row
// selection, and the sqrt-softplus MoE with IQ3_S / MXFP4 expert weights) as
// standalone driver-API kernels, so the DeepSeek4 CUDA executor runs the
// whole model without ggml. Dense Q8_0 / Q6_K projections reuse the generic
// quantized-matmul kernels in tensorsharp_kernels.cu (a separate module
// loaded on the same stream); everything DSV4-specific lives here.
//
// The imperative executor knows p0/nt for every launch, so unlike the fused
// ggml-graph implementation none of these kernels needs metadata tensors:
// ring/scratch source resolution is plain index arithmetic (the CPU executor
// in DeepSeek4CpuExecutor.cs is the reference for the exact semantics).
//
// Caches are F16 (matching the native executor and llama.cpp); activations
// and compressor state rings are F32. RoPE uses precomputed per-position
// tables [n_ctx, n_rot] of interleaved (cos, sin) with the YaRN attention
// factor folded in, built host-side with the same math as the CPU executor.
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <stdint.h>

#include "tensorsharp_dsv4_tables.cuh"

#define TS_DSV4_QK8_1 32

// Matches ts_block_q8_1 in tensorsharp_kernels.cu (and the layout produced by
// ts_quantize_q8_1_rows_warp_f32): half scale, half scale*sum, 32 int8.
struct ts_dsv4_block_q8_1
{
    half d;
    half s;
    int8_t qs[TS_DSV4_QK8_1];
};

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int ts_dsv4_dp4a(int a, int b, int c)
{
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t* a8 = (const int8_t*)&a;
    const int8_t* b8 = (const int8_t*)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

__device__ __forceinline__ float ts_dsv4_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

__device__ __forceinline__ float ts_dsv4_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float ts_dsv4_softplus(float x)
{
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// int load from a 2-byte-aligned byte pointer (block interiors of K-quants).
__device__ __forceinline__ int ts_dsv4_get_int_b2(const void* x, int i)
{
    const uint16_t* x16 = (const uint16_t*)x;
    int v = x16[2*i + 0];
    v |= ((int)x16[2*i + 1]) << 16;
    return v;
}

// int assembled from an unaligned byte pointer (MXFP4 qs starts at offset 1).
__device__ __forceinline__ int ts_dsv4_get_int_b1(const uint8_t* x, int i)
{
    return (int)x[4*i + 0]
        | ((int)x[4*i + 1] << 8)
        | ((int)x[4*i + 2] << 16)
        | ((int)x[4*i + 3] << 24);
}

// Map 8 packed nibbles onto table values: .x = low nibbles, .y = high nibbles
// (ggml get_int_from_table_16).
__device__ __forceinline__ int2 ts_dsv4_int_from_table16(int q4, const int8_t* table)
{
    const int q0_32 = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t* q0_8 = (const int8_t*)&q0_32;
    char4 v0 = make_char4(table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int q1_32 = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t* q1_8 = (const int8_t*)&q1_32;
    char4 v1 = make_char4(table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    int2 r;
    r.x = *(const int*)&v0;
    r.y = *(const int*)&v1;
    return r;
}

// ---------------------------------------------------------------------------
// quantized dot helpers (weight superblock/block vs q8_1 activation blocks)
// ---------------------------------------------------------------------------

#define TS_DSV4_IQ3S_BLOCK_BYTES 110  // half d + qs[64] + qh[8] + signs[32] + scales[4], 256 values
#define TS_DSV4_MXFP4_BLOCK_BYTES 17  // uint8 e + qs[16], 32 values
#define TS_DSV4_Q8_0_BLOCK_BYTES 34   // half d + qs[32], 32 values

// Half of one IQ3_S superblock: iqs in [iqsBegin, iqsBegin+8), i.e. 128 of the
// 256 values against 4 consecutive q8_1 blocks. Splitting the superblock lets a
// 32-lane warp cover a 4096-wide row (only 16 superblocks) with every lane
// busy instead of half of them idle.
__device__ __forceinline__ float ts_dsv4_dot_iq3s_half(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8, int iqsBegin)
{
    const half d16 = *(const half*)blk;
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 2 + 64;
    const uint8_t* signs = blk + 2 + 64 + 8;
    const uint8_t* scales = blk + 2 + 64 + 8 + 32;

    const float d = __half2float(d16);
    float sum = 0.0f;

#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int iqs = iqsBegin + 2 * j;
        int qs_pack0 = ts_dsv4_get_int_b2(qs, iqs + 0);
        int qs_pack1 = ts_dsv4_get_int_b2(qs, iqs + 1);
        const uint8_t* qsp0 = (const uint8_t*)&qs_pack0;
        const uint8_t* qsp1 = (const uint8_t*)&qs_pack1;
        uint8_t qsb[8];
        qsb[0] = qsp0[0]; qsb[1] = qsp0[1]; qsb[2] = qsp0[2]; qsb[3] = qsp0[3];
        qsb[4] = qsp1[0]; qsb[5] = qsp1[1]; qsb[6] = qsp1[2]; qsb[7] = qsp1[3];

        const int qhv = qh[iqs / 2];
        const int signs_packed_32 = ts_dsv4_get_int_b2(signs, iqs / 2);
        const uint8_t* sp8 = (const uint8_t*)&signs_packed_32;

        const ts_dsv4_block_q8_1* q8b = q8 + (iqs - iqsBegin) / 2;
        const int* q8i = (const int*)q8b->qs;

        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
        {
            const uint32_t g0 = ts_iq3s_grid[qsb[l0 + 0] | ((qhv << (8 - l0)) & 0x100)];
            const uint32_t g1 = ts_iq3s_grid[qsb[l0 + 1] | ((qhv << (7 - l0)) & 0x100)];

            const int signs0 = __vcmpne4(((sp8[l0/2] & 0x03) << 7) | ((sp8[l0/2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((sp8[l0/2] & 0x30) << 3) | ((sp8[l0/2] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4((int)g0 ^ signs0, signs0);
            const int grid_h = __vsub4((int)g1 ^ signs1, signs1);

            sumi = ts_dsv4_dp4a(grid_l, q8i[l0 + 0], sumi);
            sumi = ts_dsv4_dp4a(grid_h, q8i[l0 + 1], sumi);
        }

        sumi *= 1 + 2 * ((scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
        sum += d * __half2float(q8b->d) * (float)sumi;
    }
    return sum;
}

// One IQ3_S superblock (256 values) against 8 consecutive q8_1 blocks.
// Port of ggml-cuda vec_dot_iq3_s_q8_1 (loop over iqs = 0,2,..,14).
__device__ __forceinline__ float ts_dsv4_dot_iq3s_superblock(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8)
{
    const half d16 = *(const half*)blk;
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 2 + 64;
    const uint8_t* signs = blk + 2 + 64 + 8;
    const uint8_t* scales = blk + 2 + 64 + 8 + 32;

    const float d = __half2float(d16);
    float sum = 0.0f;

#pragma unroll
    for (int iqs = 0; iqs < 16; iqs += 2)
    {
        int qs_pack0 = ts_dsv4_get_int_b2(qs, iqs + 0);
        int qs_pack1 = ts_dsv4_get_int_b2(qs, iqs + 1);
        const uint8_t* qsp0 = (const uint8_t*)&qs_pack0;
        const uint8_t* qsp1 = (const uint8_t*)&qs_pack1;
        // byte j of the packed pair in ggml's flat view
        uint8_t qsb[8];
        qsb[0] = qsp0[0]; qsb[1] = qsp0[1]; qsb[2] = qsp0[2]; qsb[3] = qsp0[3];
        qsb[4] = qsp1[0]; qsb[5] = qsp1[1]; qsb[6] = qsp1[2]; qsb[7] = qsp1[3];

        const int qhv = qh[iqs / 2];
        const int signs_packed_32 = ts_dsv4_get_int_b2(signs, iqs / 2);
        const uint8_t* sp8 = (const uint8_t*)&signs_packed_32;

        const ts_dsv4_block_q8_1* q8b = q8 + iqs / 2;
        const int* q8i = (const int*)q8b->qs;

        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
        {
            const uint32_t g0 = ts_iq3s_grid[qsb[l0 + 0] | ((qhv << (8 - l0)) & 0x100)];
            const uint32_t g1 = ts_iq3s_grid[qsb[l0 + 1] | ((qhv << (7 - l0)) & 0x100)];

            const int signs0 = __vcmpne4(((sp8[l0/2] & 0x03) << 7) | ((sp8[l0/2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((sp8[l0/2] & 0x30) << 3) | ((sp8[l0/2] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4((int)g0 ^ signs0, signs0);
            const int grid_h = __vsub4((int)g1 ^ signs1, signs1);

            sumi = ts_dsv4_dp4a(grid_l, q8i[l0 + 0], sumi);
            sumi = ts_dsv4_dp4a(grid_h, q8i[l0 + 1], sumi);
        }

        sumi *= 1 + 2 * ((scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
        sum += d * __half2float(q8b->d) * (float)sumi;
    }
    return sum;
}

// One MXFP4 block (32 values) against one q8_1 block.
// Port of ggml-cuda vec_dot_mxfp4_q8_1.
__device__ __forceinline__ float ts_dsv4_dot_mxfp4_block(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8)
{
    const uint8_t e = blk[0];
    const uint8_t* qs = blk + 1;
    const int* q8i = (const int*)q8->qs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int aux = ts_dsv4_get_int_b1(qs, j);
        const int2 v = ts_dsv4_int_from_table16(aux, ts_kvalues_mxfp4);
        sumi = ts_dsv4_dp4a(v.x, q8i[j + 0], sumi);
        sumi = ts_dsv4_dp4a(v.y, q8i[j + 4], sumi);
    }

    const uint32_t ebits = ((uint32_t)e) << 23;
    const float escale = __uint_as_float(ebits);   // 2^(e-127)
    return escale * 0.5f * __half2float(q8->d) * (float)sumi;
}

// One Q8_0 block (32 values) against one q8_1 block (robustness for DSV4
// variants whose expert tensors are Q8_0).
__device__ __forceinline__ float ts_dsv4_dot_q8_0_block(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8)
{
    const half d16 = *(const half*)blk;
    const int8_t* qs = (const int8_t*)(blk + 2);
    const int* q8i = (const int*)q8->qs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int w = ts_dsv4_get_int_b2(qs, j);
        sumi = ts_dsv4_dp4a(w, q8i[j], sumi);
    }
    return __half2float(d16) * __half2float(q8->d) * (float)sumi;
}

// One Q6_K superblock (256 values, 210 bytes) against 8 consecutive q8_1
// blocks. Same math as ts_quant_matmul_q6k_dp4a_f32 in tensorsharp_kernels.cu
// (all four 8-value lane groups run serially here since one lane owns the
// whole superblock). 210-byte blocks are only 2-byte aligned, so the packed
// words are assembled bytewise.
__device__ __forceinline__ float ts_dsv4_dot_q6k_superblock(
    const uint8_t* sblock, const ts_dsv4_block_q8_1* q8)
{
    const uint8_t* ql = sblock;
    const uint8_t* qh = sblock + 128;
    const int8_t* scales = (const int8_t*)(sblock + 192);
    const float dSb = __half2float(*(const half*)(sblock + 208));

    float acc = 0.0f;
#pragma unroll
    for (int ls = 0; ls < 8; ++ls)
    {
        const ts_dsv4_block_q8_1* ablk = q8 + ls;
        const int* q8i = (const int*)ablk->qs;
        const float dq = __half2float(ablk->d);

        const int halfIdx = ls >> 2;
        const int group = ls & 3;
        const uint8_t* qlGroup = ql + halfIdx * 64 + ((group & 1) ? 32 : 0);
        const uint8_t* qhGroup = qh + halfIdx * 32;
        const int qlShift = group >= 2 ? 4 : 0;
        const int qhShift = group * 2;

#pragma unroll
        for (int lb = 0; lb < 4; ++lb)
        {
            const int g = lb * 2;
            const int raw0 = (((unsigned)ts_dsv4_get_int_b1(qlGroup, g) >> qlShift) & 0x0F0F0F0F)
                | ((((unsigned)ts_dsv4_get_int_b1(qhGroup, g) >> qhShift) & 0x03030303) << 4);
            const int raw1 = (((unsigned)ts_dsv4_get_int_b1(qlGroup, g + 1) >> qlShift) & 0x0F0F0F0F)
                | ((((unsigned)ts_dsv4_get_int_b1(qhGroup, g + 1) >> qhShift) & 0x03030303) << 4);
            const int w0 = __vsubss4(raw0, 0x20202020);
            const int w1 = __vsubss4(raw1, 0x20202020);
            int sumi = ts_dsv4_dp4a(w0, q8i[g], 0);
            sumi = ts_dsv4_dp4a(w1, q8i[g + 1], sumi);
            const int sc = scales[halfIdx * 8 + group * 2 + (lb >= 2 ? 1 : 0)];
            acc += dSb * (float)sc * dq * (float)sumi;
        }
    }
    return acc;
}

#define TS_DSV4_WTYPE_Q8_0 8
#define TS_DSV4_WTYPE_Q6_K 14
#define TS_DSV4_WTYPE_IQ3S 21
#define TS_DSV4_WTYPE_MXFP4 39
#define TS_DSV4_Q6_K_BLOCK_BYTES 210

// Full-row dot: one warp computes dot(weight row, activation row).
// inDim % 256 == 0 for every DSV4 projection, so each lane handles whole
// superblocks/blocks with no tail handling.
__device__ __forceinline__ float ts_dsv4_dot_row_warp(
    const uint8_t* wRow, int wtype, const ts_dsv4_block_q8_1* act, int inDim, int lane)
{
    float sum = 0.0f;
    if (wtype == TS_DSV4_WTYPE_IQ3S)
    {
        // Two half-superblocks per lane-pair: a 4096-wide row is only 16
        // superblocks, so whole-superblock striding would idle half the warp.
        const int nHalves = inDim / 128;
        for (int h = lane; h < nHalves; h += 32)
        {
            const int sb = h >> 1;
            const int iqsBegin = (h & 1) ? 8 : 0;
            sum += ts_dsv4_dot_iq3s_half(
                wRow + (size_t)sb * TS_DSV4_IQ3S_BLOCK_BYTES, act + sb * 8 + (iqsBegin / 2), iqsBegin);
        }
    }
    else if (wtype == TS_DSV4_WTYPE_MXFP4)
    {
        const int nBlk = inDim / 32;
        for (int b = lane; b < nBlk; b += 32)
            sum += ts_dsv4_dot_mxfp4_block(wRow + (size_t)b * TS_DSV4_MXFP4_BLOCK_BYTES, act + b);
    }
    else if (wtype == TS_DSV4_WTYPE_Q6_K)
    {
        const int nSuper = inDim / 256;
        for (int sb = lane; sb < nSuper; sb += 32)
            sum += ts_dsv4_dot_q6k_superblock(wRow + (size_t)sb * TS_DSV4_Q6_K_BLOCK_BYTES, act + sb * 8);
    }
    else // TS_DSV4_WTYPE_Q8_0
    {
        const int nBlk = inDim / 32;
        for (int b = lane; b < nBlk; b += 32)
            sum += ts_dsv4_dot_q8_0_block(wRow + (size_t)b * TS_DSV4_Q8_0_BLOCK_BYTES, act + b);
    }
    return ts_dsv4_warp_sum(sum);
}

// ---------------------------------------------------------------------------
// token embedding: dequantize token rows and replicate over the 4 HC streams
// ---------------------------------------------------------------------------

extern "C" __global__ void ts_dsv4_embed_f32(
    const uint8_t* __restrict__ w,   // token_embd rows
    const int32_t* __restrict__ tokens,
    float* __restrict__ xs,          // [nt, 4, E]
    const int wtype,                 // Q8_0 / F16 / F32
    const long long rowBytes,
    const int nt,
    const int E)
{
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || e >= E)
        return;

    const uint8_t* row = w + (size_t)tokens[t] * rowBytes;
    float v;
    if (wtype == TS_DSV4_WTYPE_Q8_0)
    {
        const uint8_t* blk = row + (e / 32) * TS_DSV4_Q8_0_BLOCK_BYTES;
        const half d = *(const half*)blk;
        const int8_t q = ((const int8_t*)(blk + 2))[e % 32];
        v = __half2float(d) * (float)q;
    }
    else if (wtype == 1) // F16
    {
        v = __half2float(((const half*)row)[e]);
    }
    else if (wtype == 30) // BF16
    {
        v = __uint_as_float((unsigned int)((const unsigned short*)row)[e] << 16);
    }
    else // F32
    {
        v = ((const float*)row)[e];
    }

    float* dst = xs + ((size_t)t * 4) * E + e;
#pragma unroll
    for (int s = 0; s < 4; ++s)
        dst[(size_t)s * E] = v;
}

// ---------------------------------------------------------------------------
// hyper-connections
// ---------------------------------------------------------------------------

// inv[t] = rsqrt(mean(flat^2) + eps) over the flattened 4*E streams.
extern "C" __global__ void ts_dsv4_hc_rms_f32(
    const float* __restrict__ xs,
    float* __restrict__ inv,
    const int flatDim,
    const float eps)
{
    const int t = blockIdx.x;
    const float* x = xs + (size_t)t * flatDim;

    __shared__ float red[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < flatDim; i += blockDim.x)
    {
        const float v = x[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        inv[t] = rsqrtf(red[0] / flatDim + eps);
}

// pre/post sigmoid gates + Sinkhorn-normalized 4x4 comb matrix, one thread per
// token (the whole computation is ~500 flops on 24 inputs).
extern "C" __global__ void ts_dsv4_hc_gates_comb_f32(
    const float* __restrict__ mixesRaw,  // [nt, 24] (un-scaled hc_fn output)
    const float* __restrict__ inv,       // [nt]
    const float* __restrict__ scale,     // [3]
    const float* __restrict__ baseW,     // [24]
    float* __restrict__ pre,             // [nt, 4]
    float* __restrict__ post,            // [nt, 4]
    float* __restrict__ comb,            // [nt, 16] (idst + 4*isrc)
    const int nt,
    const int iters,
    const float eps)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt)
        return;

    const float invT = inv[t];
    float mixes[24];
#pragma unroll
    for (int i = 0; i < 24; ++i)
        mixes[i] = mixesRaw[(size_t)t * 24 + i] * invT;

#pragma unroll
    for (int s = 0; s < 4; ++s)
    {
        pre[(size_t)t * 4 + s] = ts_dsv4_sigmoid(mixes[s] * scale[0] + baseW[s]) + eps;
        post[(size_t)t * 4 + s] = 2.0f * ts_dsv4_sigmoid(mixes[4 + s] * scale[1] + baseW[4 + s]);
    }

    float c[16];
#pragma unroll
    for (int isrc = 0; isrc < 4; ++isrc)
    {
        float m = -INFINITY;
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
        {
            const int idx = idst + 4 * isrc;
            const float v = mixes[8 + idx] * scale[2] + baseW[8 + idx];
            c[idx] = v;
            if (v > m) m = v;
        }
        float sum = 0.0f;
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
        {
            const int idx = idst + 4 * isrc;
            const float v = expf(c[idx] - m);
            c[idx] = v;
            sum += v;
        }
        const float invSum = 1.0f / sum;
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
            c[idst + 4 * isrc] = c[idst + 4 * isrc] * invSum + eps;
    }

    // Sinkhorn: col-norm, then (iters-1) x (row-norm, col-norm)
    for (int it = 0; it < iters; ++it)
    {
        if (it > 0)
        {
#pragma unroll
            for (int isrc = 0; isrc < 4; ++isrc)
            {
                float sum = eps;
#pragma unroll
                for (int idst = 0; idst < 4; ++idst) sum += c[idst + 4 * isrc];
                const float invSum = 1.0f / sum;
#pragma unroll
                for (int idst = 0; idst < 4; ++idst) c[idst + 4 * isrc] *= invSum;
            }
        }
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
        {
            float sum = eps;
#pragma unroll
            for (int isrc = 0; isrc < 4; ++isrc) sum += c[idst + 4 * isrc];
            const float invSum = 1.0f / sum;
#pragma unroll
            for (int isrc = 0; isrc < 4; ++isrc) c[idst + 4 * isrc] *= invSum;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
        comb[(size_t)t * 16 + i] = c[i];
}

// cur[t,e] = sum_s xs[t,s,e] * pre[t,s]
extern "C" __global__ void ts_dsv4_hc_collapse_f32(
    const float* __restrict__ xs,
    const float* __restrict__ pre,
    float* __restrict__ cur,
    const int nt,
    const int E)
{
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || e >= E)
        return;

    const float* x = xs + (size_t)t * 4 * E + e;
    const float* p = pre + (size_t)t * 4;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s)
        acc += x[(size_t)s * E] * p[s];
    cur[(size_t)t * E + e] = acc;
}

// xsOut[t,idst,e] = o[t,e]*post[t,idst] + sum_isrc xs[t,isrc,e]*comb[t, idst+4*isrc]
extern "C" __global__ void ts_dsv4_hc_post_f32(
    const float* __restrict__ xs,
    const float* __restrict__ blockOut,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ xsOut,
    const int nt,
    const int E)
{
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || i >= 4 * E)
        return;

    const int idst = i / E;
    const int e = i % E;

    const float* x = xs + (size_t)t * 4 * E + e;
    const float* cb = comb + (size_t)t * 16;
    float acc = blockOut[(size_t)t * E + e] * post[(size_t)t * 4 + idst];
#pragma unroll
    for (int isrc = 0; isrc < 4; ++isrc)
        acc += x[(size_t)isrc * E] * cb[idst + 4 * isrc];
    xsOut[(size_t)t * 4 * E + (size_t)idst * E + e] = acc;
}

// hc_head for the final token: collapse streams with sigmoid gates from
// output_hc_fn (F32 [4, flatDim]), written into cur [E].
extern "C" __global__ void ts_dsv4_hc_head_f32(
    const float* __restrict__ x,      // [4*E] (the selected token's streams)
    const float* __restrict__ fn,     // [4, 4*E] F32
    const float* __restrict__ scale,  // [>=1]
    const float* __restrict__ baseW,  // [>=1]
    float* __restrict__ cur,          // [E]
    const int E,
    const int scaleCount,
    const int baseCount,
    const float eps)
{
    const int flatDim = 4 * E;
    __shared__ float red[256];
    __shared__ float gates[4];

    float acc = 0.0f;
    for (int i = threadIdx.x; i < flatDim; i += blockDim.x)
    {
        const float v = x[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / flatDim + eps);
    __syncthreads();

    for (int s = 0; s < 4; ++s)
    {
        const float* fr = fn + (size_t)s * flatDim;
        float dot = 0.0f;
        for (int i = threadIdx.x; i < flatDim; i += blockDim.x)
            dot += fr[i] * x[i];
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int r = blockDim.x / 2; r > 0; r >>= 1)
        {
            if (threadIdx.x < r)
                red[threadIdx.x] += red[threadIdx.x + r];
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            const float sc = scaleCount >= 4 ? scale[s] : scale[0];
            const float b = baseCount >= 4 ? baseW[s] : baseW[0];
            gates[s] = ts_dsv4_sigmoid(red[0] * inv * sc + b) + eps;
        }
        __syncthreads();
    }

    for (int e = threadIdx.x; e < E; e += blockDim.x)
    {
        float v = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s)
            v += x[(size_t)s * E + e] * gates[s];
        cur[e] = v;
    }
}

// ---------------------------------------------------------------------------
// norms / rope / attention prologue
// ---------------------------------------------------------------------------

// In-place RMS norm over rows; weight may be null.
extern "C" __global__ void ts_dsv4_rmsnorm_f32(
    float* __restrict__ data,
    const float* __restrict__ weight,
    const int rows,
    const int n,
    const float eps)
{
    const int r = blockIdx.x;
    if (r >= rows)
        return;
    float* row = data + (size_t)r * n;

    __shared__ float red[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = row[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / n + eps);

    if (weight != nullptr)
    {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            row[i] = row[i] * inv * weight[i];
    }
    else
    {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            row[i] *= inv;
    }
}

// blockIdx.x in [0, NH]: heads 0..NH-1 per-head-RMS + RoPE q in place; block
// NH RMS-norms (with weight) + RoPEs kv and commits it (F16) to the SWA ring.
extern "C" __global__ void ts_dsv4_attn_prep_f32(
    float* __restrict__ q,               // [nt, NH, HD] in-place
    const float* __restrict__ kvRaw,     // [nt, HD]
    const float* __restrict__ kvNormW,   // [HD]
    const float* __restrict__ ropeTab,   // [nCtx, nRot]
    half* __restrict__ ring,             // [ringRows, HD]
    const int p0,
    const int ringRows,
    const int NH,
    const int HD,
    const int nRot,
    const float eps)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const bool isKv = h == NH;

    __shared__ float sh[512];
    __shared__ float red[256];

    float* qDst = q + ((size_t)t * NH + h) * HD;
    const float* src = isKv ? kvRaw + (size_t)t * HD : qDst;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        const float v = src[d];
        sh[d] = v;
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / HD + eps);

    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        sh[d] *= isKv ? inv * kvNormW[d] : inv;
    __syncthreads();

    const long long p = (long long)p0 + t;
    const float* tab = ropeTab + p * nRot;
    const int rbase = HD - nRot;

    if (isKv)
    {
        half* out = ring + (size_t)(p % ringRows) * HD;
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
        {
            float v = sh[d];
            if (d >= rbase)
            {
                const int i = (d - rbase) >> 1;
                const float c = tab[2 * i + 0];
                const float s = tab[2 * i + 1];
                const float x0 = sh[rbase + 2 * i + 0];
                const float x1 = sh[rbase + 2 * i + 1];
                v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            out[d] = __float2half(v);
        }
    }
    else
    {
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
        {
            float v = sh[d];
            if (d >= rbase)
            {
                const int i = (d - rbase) >> 1;
                const float c = tab[2 * i + 0];
                const float s = tab[2 * i + 1];
                const float x0 = sh[rbase + 2 * i + 0];
                const float x1 = sh[rbase + 2 * i + 1];
                v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            qDst[d] = v;
        }
    }
}

// stScore[t] += ape[(p0+t) % ratio]
extern "C" __global__ void ts_dsv4_ape_add_f32(
    float* __restrict__ stScore,
    const float* __restrict__ ape,
    const int p0,
    const int ratio,
    const int nt,
    const int cw)
{
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || i >= cw)
        return;
    const int r = (int)(((long long)p0 + t) % ratio);
    stScore[(size_t)t * cw + i] += ape[(size_t)r * cw + i];
}

// ---------------------------------------------------------------------------
// block compressor (CSA coff=2 / HCA coff=1 / LID)
// ---------------------------------------------------------------------------

// One block per boundary in the ubatch. Sources are resolved by index math:
// window token tw reads the scratch projections when tw >= p0 and the state
// ring otherwise; tw < 0 contributes nothing (kv 0 / score -inf).
extern "C" __global__ void ts_dsv4_compress_f32(
    const float* __restrict__ stKv,      // [nt, cw]
    const float* __restrict__ stScore,   // [nt, cw]
    const float* __restrict__ histKv,    // [stateSize, cw]
    const float* __restrict__ histScore, // [stateSize, cw]
    const float* __restrict__ normW,     // [head]
    const float* __restrict__ ropeTab,   // [nCtx, nRot]
    half* __restrict__ cache,            // [rows, head]
    const long long firstBoundary,       // absolute position of first boundary
    const int nBlocks,
    const int p0,
    const int ratio,
    const int coff,
    const int head,
    const int stateSize,
    const int nRot,
    const float eps)
{
    const int b = blockIdx.x;
    if (b >= nBlocks)
        return;

    const int cw = coff * head;
    const int W = coff * ratio;
    const long long p = firstBoundary + (long long)b * ratio;
    const long long cacheRow = p / ratio;
    const long long blockStart = p + 1 - ratio;

    __shared__ float shOut[512];
    __shared__ float red[256];

    float acc2 = 0.0f;
    for (int d = threadIdx.x; d < head; d += blockDim.x)
    {
        float m = -INFINITY;
        float se = 0.0f;
        float sv = 0.0f;
        for (int w = 0; w < W; ++w)
        {
            const long long tw = p + 1 - W + w;
            if (tw < 0)
                continue;
            const int half_ = (coff == 2 && w >= ratio) ? head : 0;
            const float* kvSrc;
            const float* scSrc;
            if (tw >= p0)
            {
                const size_t off = (size_t)(tw - p0) * cw + half_ + d;
                kvSrc = stKv + off;
                scSrc = stScore + off;
            }
            else
            {
                const size_t off = (size_t)(tw % stateSize) * cw + half_ + d;
                kvSrc = histKv + off;
                scSrc = histScore + off;
            }
            const float sc = *scSrc;
            if (sc > m)
            {
                const float r = expf(m - sc); // 0 when m was -inf
                se *= r;
                sv *= r;
                m = sc;
            }
            const float e = sc == -INFINITY ? 0.0f : expf(sc - m);
            se += e;
            sv += e * (*kvSrc);
        }
        const float out = se > 0.0f ? sv / se : 0.0f;
        shOut[d] = out;
        acc2 += out * out;
    }

    red[threadIdx.x] = acc2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(red[0] / head + eps);

    for (int d = threadIdx.x; d < head; d += blockDim.x)
        shOut[d] *= scale * normW[d];
    __syncthreads();

    const float* tab = ropeTab + blockStart * nRot;
    const int rbase = head - nRot;
    for (int d = threadIdx.x; d < head; d += blockDim.x)
    {
        float v = shOut[d];
        if (d >= rbase)
        {
            const int i = (d - rbase) >> 1;
            const float c = tab[2 * i + 0];
            const float s = tab[2 * i + 1];
            const float x0 = shOut[rbase + 2 * i + 0];
            const float x1 = shOut[rbase + 2 * i + 1];
            v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        cache[cacheRow * head + d] = __float2half(v);
    }
}

// Persist the tail of the ubatch's projections into the state ring.
extern "C" __global__ void ts_dsv4_persist_f32(
    const float* __restrict__ stKv,
    const float* __restrict__ stScore,
    float* __restrict__ histKv,
    float* __restrict__ histScore,
    const int p0,
    const int nt,
    const int stateSize,
    const int cw)
{
    const int np = nt < stateSize ? nt : stateSize;
    const long long total = (long long)np * cw;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int j = (int)(i / cw);
    const int d = (int)(i % cw);
    const int t = nt - np + j;
    const long long slot = ((long long)p0 + t) % stateSize;
    histKv[slot * cw + d] = stKv[(size_t)t * cw + d];
    histScore[slot * cw + d] = stScore[(size_t)t * cw + d];
}

// ---------------------------------------------------------------------------
// lightning indexer
// ---------------------------------------------------------------------------

// RoPE the tail dims of each indexer q head; scale the per-head weights.
extern "C" __global__ void ts_dsv4_idx_prep_f32(
    float* __restrict__ iq,            // [nt, IH, ID] in-place
    float* __restrict__ iw,            // [nt, IH]
    const float* __restrict__ ropeTab, // comp table
    const int p0,
    const int IH,
    const int ID,
    const int nRot,
    const float iwScale)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[256];

    float* row = iq + ((size_t)t * IH + h) * ID;
    for (int d = threadIdx.x; d < ID; d += blockDim.x)
        sh[d] = row[d];
    __syncthreads();

    const long long p = (long long)p0 + t;
    const float* tab = ropeTab + p * nRot;
    const int rbase = ID - nRot;
    for (int d = threadIdx.x; d < ID; d += blockDim.x)
    {
        float v = sh[d];
        if (d >= rbase)
        {
            const int i = (d - rbase) >> 1;
            const float c = tab[2 * i + 0];
            const float s = tab[2 * i + 1];
            const float x0 = sh[rbase + 2 * i + 0];
            const float x1 = sh[rbase + 2 * i + 1];
            v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        row[d] = v;
    }

    if (h == 0 && threadIdx.x < IH)
        iw[(size_t)t * IH + threadIdx.x] *= iwScale;
}

// scores[t, r] = sum_h relu(q_th . k_r) * w_th for r < nVis(t).
// One warp per (row, token); q rows staged in shared per block.
#define TS_DSV4_IDX_ROWS_PER_BLOCK 8

extern "C" __global__ void ts_dsv4_idx_scores_f32(
    const float* __restrict__ iq,     // [nt, IH, ID]
    const float* __restrict__ iw,     // [nt, IH]
    const half* __restrict__ lidK,    // [rows, ID]
    float* __restrict__ scores,       // [nt, scoreStride]
    const int p0,
    const int ratio,
    const int IH,
    const int ID,
    const int nt,
    const int scoreStride)
{
    const int t = blockIdx.y;
    const int nVis = (int)(((long long)p0 + t + 1) / ratio);
    const int r0 = blockIdx.x * TS_DSV4_IDX_ROWS_PER_BLOCK;
    if (r0 >= nVis)
        return;

    // stage this token's q heads and weights (IH*ID <= 64*128 floats = 32KB)
    extern __shared__ float shQ[]; // [IH*ID + IH]
    float* shW = shQ + IH * ID;
    for (int i = threadIdx.x; i < IH * ID; i += blockDim.x)
        shQ[i] = iq[(size_t)t * IH * ID + i];
    for (int i = threadIdx.x; i < IH; i += blockDim.x)
        shW[i] = iw[(size_t)t * IH + i];
    __syncthreads();

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int r = r0 + warpId;
    if (r >= nVis)
        return;

    // each lane holds 4 consecutive k dims (ID = 128)
    const half* kRow = lidK + (size_t)r * ID;
    float k0 = __half2float(kRow[lane * 4 + 0]);
    float k1 = __half2float(kRow[lane * 4 + 1]);
    float k2 = __half2float(kRow[lane * 4 + 2]);
    float k3 = __half2float(kRow[lane * 4 + 3]);

    float score = 0.0f;
    for (int h = 0; h < IH; ++h)
    {
        const float* qh = shQ + (size_t)h * ID + lane * 4;
        float part = qh[0] * k0 + qh[1] * k1 + qh[2] * k2 + qh[3] * k3;
        part = ts_dsv4_warp_sum(part);
        if (lane == 0 && part > 0.0f)
            score += part * shW[h];
    }
    if (lane == 0)
        scores[(size_t)t * scoreStride + r] = score;
}

// Per-token top-K selection over scores[0, nVis) via 4-pass MSB radix select
// on flipped float keys. Emits an (unordered) index list; attention treats
// the selection as a set, so order does not matter.
extern "C" __global__ void ts_dsv4_topk_f32(
    const float* __restrict__ scores, // [nt, scoreStride]
    int32_t* __restrict__ topkIdx,    // [nt, K]
    int32_t* __restrict__ topkCnt,    // [nt]
    const int p0,
    const int ratio,
    const int K,
    const int scoreStride)
{
    const int t = blockIdx.x;
    const int nVis = (int)(((long long)p0 + t + 1) / ratio);
    const float* sc = scores + (size_t)t * scoreStride;
    int32_t* outIdx = topkIdx + (size_t)t * K;

    const int k = nVis < K ? nVis : K;
    if (threadIdx.x == 0)
        topkCnt[t] = k;
    if (k <= 0)
        return;

    if (nVis <= K)
    {
        for (int r = threadIdx.x; r < nVis; r += blockDim.x)
            outIdx[r] = r;
        return;
    }

    __shared__ unsigned int hist[256];
    __shared__ unsigned int shPrefix;
    __shared__ int shNeed;
    __shared__ unsigned int shEmit;
    __shared__ unsigned int shTieEmit;

    if (threadIdx.x == 0)
    {
        shPrefix = 0;
        shNeed = k;
    }
    __syncthreads();

    // find the k-th largest key byte by byte from the MSB
    for (int pass = 0; pass < 4; ++pass)
    {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256)
            hist[threadIdx.x] = 0;
        __syncthreads();

        const unsigned int prefix = shPrefix;
        for (int r = threadIdx.x; r < nVis; r += blockDim.x)
        {
            const float v = sc[r];
            unsigned int key = __float_as_uint(v);
            key ^= (key & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
            if (pass > 0 && (key >> (shift + 8)) != prefix)
                continue;
            atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int need = shNeed;
            unsigned int bin = 255;
            for (int b2 = 255; b2 >= 0; --b2)
            {
                const int c = (int)hist[b2];
                if (c >= need)
                {
                    bin = (unsigned int)b2;
                    break;
                }
                need -= c;
            }
            shNeed = need;
            shPrefix = (shPrefix << 8) | bin;
        }
        __syncthreads();
    }

    // shPrefix is now the k-th largest key; emit keys > threshold, then pad
    // with ties (== threshold) up to k.
    const unsigned int threshold = shPrefix;
    if (threadIdx.x == 0)
    {
        shEmit = 0;
        shTieEmit = 0;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < nVis; r += blockDim.x)
    {
        const float v = sc[r];
        unsigned int key = __float_as_uint(v);
        key ^= (key & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        if (key > threshold)
        {
            const unsigned int slot = atomicAdd(&shEmit, 1u);
            outIdx[slot] = r;
        }
    }
    __syncthreads();
    const unsigned int strictCount = shEmit;
    for (int r = threadIdx.x; r < nVis; r += blockDim.x)
    {
        const float v = sc[r];
        unsigned int key = __float_as_uint(v);
        key ^= (key & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        if (key == threshold)
        {
            const unsigned int tie = atomicAdd(&shTieEmit, 1u);
            const unsigned int slot = strictCount + tie;
            if (slot < (unsigned int)k)
                outIdx[slot] = r;
        }
    }
}

// ---------------------------------------------------------------------------
// attention core
// ---------------------------------------------------------------------------

// Online-softmax attention with sinks over [raw SWA ring rows | compressed
// rows]. K doubles as V. mode: 0 = raw rows only, 1 = compressed rows from
// the top-k list, 2 = all visible compressed rows ((p+1)/ratio).
// grid (NH, nt), 256 threads = 8 warps; warps split the rows, each lane owns
// HD/32 = 16 dims of its warp's accumulator, partials merged through shared.
extern "C" __global__ void ts_dsv4_attention_f32(
    const float* __restrict__ q,        // [nt, NH, HD]
    const half* __restrict__ ring,      // [ringRows, HD]
    const half* __restrict__ comp,      // [rows, HD] (null for mode 0)
    const int32_t* __restrict__ topkIdx,// [nt, K] (mode 1)
    const int32_t* __restrict__ topkCnt,// [nt] (mode 1)
    const float* __restrict__ sinks,    // [NH]
    float* __restrict__ attnO,          // [nt, NH, HD]
    const int p0,
    const int nSwa,
    const int ringRows,
    const int NH,
    const int HD,
    const int mode,
    const int ratio,
    const int K,
    const float kqScale)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const long long p = (long long)p0 + t;

    const long long rawStart = p - nSwa + 1 > 0 ? p - nSwa + 1 : 0;
    const int rawCnt = (int)(p - rawStart + 1);
    int compCnt = 0;
    const int32_t* sel = nullptr;
    if (mode == 1)
    {
        compCnt = topkCnt[t];
        sel = topkIdx + (size_t)t * K;
    }
    else if (mode == 2)
    {
        compCnt = (int)((p + 1) / ratio);
    }
    const int n = rawCnt + compCnt;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int dimsPerLane = HD / 32; // 16

    const float* qRow = q + ((size_t)t * NH + h) * HD;
    float qReg[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        qReg[i] = qRow[lane * dimsPerLane + i];

    float m = -INFINITY;
    float sum = 0.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] = 0.0f;

    for (int i = warpId; i < n; i += 8)
    {
        const half* kRow;
        if (i < rawCnt)
        {
            kRow = ring + (size_t)((rawStart + i) % ringRows) * HD;
        }
        else
        {
            const int cr = mode == 1 ? sel[i - rawCnt] : (i - rawCnt);
            kRow = comp + (size_t)cr * HD;
        }

        // 16 halves per lane = two 16-byte vector loads. Reading them as 16
        // scalar __half loads issues 8x the load instructions for the same
        // bytes, and K traffic is what this kernel is bound by.
        const half* kLane = kRow + lane * dimsPerLane;
        float kReg[16];
        {
            const int4 v0 = *(const int4*)kLane;
            const int4 v1 = *(const int4*)(kLane + 8);
            const half2* h0 = (const half2*)&v0;
            const half2* h1 = (const half2*)&v1;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const float2 f0 = __half22float2(h0[j]);
                const float2 f1 = __half22float2(h1[j]);
                kReg[2*j + 0] = f0.x;
                kReg[2*j + 1] = f0.y;
                kReg[8 + 2*j + 0] = f1.x;
                kReg[8 + 2*j + 1] = f1.y;
            }
        }

        float part = 0.0f;
#pragma unroll
        for (int d2 = 0; d2 < 16; ++d2)
            part += qReg[d2] * kReg[d2];
        const float s = ts_dsv4_warp_sum(part) * kqScale;

        if (s > m)
        {
            const float r = expf(m - s); // 0 when m was -inf
            sum *= r;
#pragma unroll
            for (int d2 = 0; d2 < 16; ++d2)
                acc[d2] *= r;
            m = s;
        }
        const float w = expf(s - m);
        sum += w;
#pragma unroll
        for (int d2 = 0; d2 < 16; ++d2)
            acc[d2] += w * kReg[d2];
    }

    // merge the 8 warp partials
    __shared__ float shAcc[8][512];
    __shared__ float shM[8];
    __shared__ float shSum[8];

#pragma unroll
    for (int d2 = 0; d2 < 16; ++d2)
        shAcc[warpId][lane * dimsPerLane + d2] = acc[d2];
    if (lane == 0)
    {
        shM[warpId] = m;
        shSum[warpId] = sum;
    }
    __syncthreads();

    __shared__ float shTotM;
    __shared__ float shTotSum;
    if (threadIdx.x == 0)
    {
        float M = sinks[h];
        for (int w = 0; w < 8; ++w)
            if (shM[w] > M) M = shM[w];
        float S = expf(sinks[h] - M);
        for (int w = 0; w < 8; ++w)
            S += shSum[w] * expf(shM[w] - M);
        shTotM = M;
        shTotSum = S;
    }
    __syncthreads();

    const float M = shTotM;
    const float invSum = 1.0f / shTotSum;
    float* out = attnO + ((size_t)t * NH + h) * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w)
            v += shAcc[w][d] * expf(shM[w] - M);
        out[d] = v * invSum;
    }
}

// Inverse RoPE on the tail dims of each attention output head, stored in the
// grouped layout [G, nt, headsPerGroup*HD] for the grouped LoRA out-proj.
extern "C" __global__ void ts_dsv4_attn_finish_f32(
    const float* __restrict__ attnO,   // [nt, NH, HD]
    const float* __restrict__ ropeTab,
    float* __restrict__ dst,           // [G, nt, headsPerGroup*HD]
    const int p0,
    const int NH,
    const int HD,
    const int nRot,
    const int headsPerGroup,
    const int nt)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[512];
    const float* src = attnO + ((size_t)t * NH + h) * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        sh[d] = src[d];
    __syncthreads();

    const long long p = (long long)p0 + t;
    const float* tab = ropeTab + p * nRot;
    const int rbase = HD - nRot;
    const int g = h / headsPerGroup;
    const int hg = h % headsPerGroup;
    const size_t groupDim = (size_t)headsPerGroup * HD;
    float* out = dst + (size_t)g * nt * groupDim + (size_t)t * groupDim + (size_t)hg * HD;

    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        float v = sh[d];
        if (d >= rbase)
        {
            const int i = (d - rbase) >> 1;
            const float c = tab[2 * i + 0];
            const float s = tab[2 * i + 1];
            const float x0 = sh[rbase + 2 * i + 0];
            const float x1 = sh[rbase + 2 * i + 1];
            // transpose (inverse) of the forward rotation
            v = ((d - rbase) & 1) == 0 ? x0 * c + x1 * s : -x0 * s + x1 * c;
        }
        out[d] = v;
    }
}

// oG[t, g*R + r] = oGtmp[g, t, r] (regroup the per-group LoRA-A outputs into
// the concatenated wo_b input layout).
extern "C" __global__ void ts_dsv4_regroup_f32(
    const float* __restrict__ oGtmp, // [G, nt, R]
    float* __restrict__ oG,          // [nt, G*R]
    const int G,
    const int nt,
    const int R)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)G * nt * R;
    if (i >= total)
        return;
    const int g = (int)(i / ((long long)nt * R));
    const int rem = (int)(i % ((long long)nt * R));
    const int t = rem / R;
    const int r = rem % R;
    oG[(size_t)t * G * R + (size_t)g * R + r] = oGtmp[i];
}

// ---------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------

// Selection + weights. Hash layers route through the tid2eid LUT; the rest
// take the top-k of sqrt(softplus(logits)) + bias. Weights are the unbiased
// probs of the selected experts, optionally sum-normalized, then scaled.
extern "C" __global__ void ts_dsv4_moe_select_f32(
    const float* __restrict__ logits,   // [nt, nExpert]
    const float* __restrict__ bias,     // [nExpert] or null
    const int32_t* __restrict__ tid2eid,// [nVocab, nUsed] or null
    const int32_t* __restrict__ tokens, // [nt] (hash layers)
    int32_t* __restrict__ sel,          // [nt, nUsed]
    float* __restrict__ wOut,           // [nt, nUsed]
    const int nExpert,
    const int nUsed,
    const int norm,
    const float wScale)
{
    const int t = blockIdx.x;
    const float* lg = logits + (size_t)t * nExpert;
    int32_t* selT = sel + (size_t)t * nUsed;
    float* wT = wOut + (size_t)t * nUsed;

    if (tid2eid != nullptr)
    {
        if (threadIdx.x < (unsigned)nUsed)
            selT[threadIdx.x] = tid2eid[(size_t)tokens[t] * nUsed + threadIdx.x];
    }
    else
    {
        extern __shared__ float shSel[]; // [nExpert]
        for (int e = threadIdx.x; e < nExpert; e += blockDim.x)
            shSel[e] = sqrtf(ts_dsv4_softplus(lg[e])) + bias[e];
        __syncthreads();

        __shared__ float shVal[256];
        __shared__ int shIdx[256];
        for (int k = 0; k < nUsed; ++k)
        {
            float best = -INFINITY;
            int bidx = -1;
            for (int e = threadIdx.x; e < nExpert; e += blockDim.x)
            {
                const float v = shSel[e];
                if (v > best || (v == best && e < bidx))
                {
                    best = v;
                    bidx = e;
                }
            }
            shVal[threadIdx.x] = best;
            shIdx[threadIdx.x] = bidx;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (threadIdx.x < s)
                {
                    const float ov = shVal[threadIdx.x + s];
                    const int oi = shIdx[threadIdx.x + s];
                    if (ov > shVal[threadIdx.x] || (ov == shVal[threadIdx.x] && oi >= 0 &&
                            (shIdx[threadIdx.x] < 0 || oi < shIdx[threadIdx.x])))
                    {
                        shVal[threadIdx.x] = ov;
                        shIdx[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                const int chosen = shIdx[0] < 0 ? 0 : shIdx[0];
                selT[k] = chosen;
                shSel[chosen] = -INFINITY;
            }
            __syncthreads();
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float w[16];
        float sum = 0.0f;
        for (int j = 0; j < nUsed; ++j)
        {
            w[j] = sqrtf(ts_dsv4_softplus(lg[selT[j]]));
            sum += w[j];
        }
        if (norm)
        {
            if (sum < 6.103515625e-5f)
                sum = 6.103515625e-5f;
            const float inv = 1.0f / sum;
            for (int j = 0; j < nUsed; ++j)
                w[j] *= inv;
        }
        for (int j = 0; j < nUsed; ++j)
            wT[j] = w[j] * wScale;
    }
}

// grouping plan: histogram / exclusive scan / slot scatter
extern "C" __global__ void ts_dsv4_moe_count_i32(
    const int32_t* __restrict__ sel, int32_t* __restrict__ counts, const int S)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S)
        return;
    atomicAdd(&counts[sel[i]], 1);
}

extern "C" __global__ void ts_dsv4_moe_scan_i32(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ cursors,
    const int nExpert)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    int off = 0;
    for (int e = 0; e < nExpert; ++e)
    {
        offsets[e] = off;
        cursors[e] = off;
        off += counts[e];
    }
}

extern "C" __global__ void ts_dsv4_moe_scatter_i32(
    const int32_t* __restrict__ sel,
    int32_t* __restrict__ cursors,
    int32_t* __restrict__ rowOfSlot,   // [S]
    int32_t* __restrict__ slotToken,   // [S]
    const int S,
    const int nUsed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S)
        return;
    const int row = atomicAdd(&cursors[sel[i]], 1);
    rowOfSlot[i] = row;
    slotToken[row] = i / nUsed;
}

// ---------------------------------------------------------------------------
// Staged-weight expert projections (prefill).
//
// The per-token kernels below re-decode a quantized weight row for every
// member token: for IQ3_S that is ~32 codebook-grid lookups per 128 values,
// repeated m times, which dominates once an expert has more than a couple of
// routed tokens. These variants decode each row ONCE into shared memory as
// plain int8 plus a per-32-value float scale, then dot the staged row against
// every member token with nothing but dp4a. Weight-decode cost becomes O(1) in
// m instead of O(m).
//
// The staged row lives in REGISTERS, not shared memory: lane l owns the 32-value
// sub-blocks b = l, l+32, l+64, ... (four of them for a 4096-wide row), holding
// each as 8 packed ints plus its scale. That removes shared-memory bank
// conflicts entirely, needs no __syncthreads, and -- because a 36KB shared
// staging buffer capped occupancy at one block per SM -- lets many blocks run
// per SM.
//
// Scales fold the weight's block scale (and, for IQ3_S, the sub-block
// (1 + 2*ls) factor); the activation's own q8_1 scale is applied at use.
// Codebook values fit int8 for every supported type (IQ3_S grid |v| <= 15 after
// sign application, MXFP4 table |v| <= 12, Q6_K raw-32 is -32..31, Q8_0 is
// already int8). Q6_K's two half-group scales are kept separately.
//
// Activations come from the DENSE split q8_1 layout (ts_quantize_q8_1_split_rows_f32):
// a contiguous int8 row plus a separate per-block float scale. The interleaved
// 36-byte ts_block_q8_1 is only 4-byte aligned, so reading it costs eight
// scattered 4-byte loads per sub-block; the dense form is read as two int4s.
// ---------------------------------------------------------------------------

#define TS_DSV4_MAX_SUBS_PER_LANE 4   // 32 lanes x 4 sub-blocks x 32 values = 4096
#define TS_DSV4_STAGE_ROWS 2          // weight rows staged per warp

// Decode one 32-value sub-block of a quantized weight row into 8 packed ints
// plus its scale(s). scHi is only written for Q6_K (whose 32-value group holds
// two 16-value scales).
__device__ __forceinline__ void ts_dsv4_decode_sub(
    const uint8_t* __restrict__ wRow, int wtype, int sub, int* __restrict__ w8,
    float* __restrict__ scLo, float* __restrict__ scHi)
{
    if (wtype == TS_DSV4_WTYPE_IQ3S)
    {
        const int sb = sub >> 3;
        const int iqs = (sub & 7) * 2;
        const uint8_t* blk = wRow + (size_t)sb * TS_DSV4_IQ3S_BLOCK_BYTES;
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 2 + 64;
        const uint8_t* signs = blk + 2 + 64 + 8;
        const uint8_t* scales = blk + 2 + 64 + 8 + 32;

        int qs_pack0 = ts_dsv4_get_int_b2(qs, iqs + 0);
        int qs_pack1 = ts_dsv4_get_int_b2(qs, iqs + 1);
        const uint8_t* qsp0 = (const uint8_t*)&qs_pack0;
        const uint8_t* qsp1 = (const uint8_t*)&qs_pack1;
        uint8_t qsb[8];
        qsb[0] = qsp0[0]; qsb[1] = qsp0[1]; qsb[2] = qsp0[2]; qsb[3] = qsp0[3];
        qsb[4] = qsp1[0]; qsb[5] = qsp1[1]; qsb[6] = qsp1[2]; qsb[7] = qsp1[3];

        const int qhv = qh[iqs / 2];
        const int signs_packed_32 = ts_dsv4_get_int_b2(signs, iqs / 2);
        const uint8_t* sp8 = (const uint8_t*)&signs_packed_32;

#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
        {
            const uint32_t g0 = ts_iq3s_grid[qsb[l0 + 0] | ((qhv << (8 - l0)) & 0x100)];
            const uint32_t g1 = ts_iq3s_grid[qsb[l0 + 1] | ((qhv << (7 - l0)) & 0x100)];
            const int s0 = __vcmpne4(((sp8[l0/2] & 0x03) << 7) | ((sp8[l0/2] & 0x0C) << 21), 0x00000000);
            const int s1 = __vcmpne4(((sp8[l0/2] & 0x30) << 3) | ((sp8[l0/2] & 0xC0) << 17), 0x00000000);
            w8[l0 + 0] = __vsub4((int)g0 ^ s0, s0);
            w8[l0 + 1] = __vsub4((int)g1 ^ s1, s1);
        }
        *scLo = __half2float(*(const half*)blk)
              * (float)(1 + 2 * ((scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F));
        *scHi = *scLo;
    }
    else if (wtype == TS_DSV4_WTYPE_MXFP4)
    {
        const uint8_t* blk = wRow + (size_t)sub * TS_DSV4_MXFP4_BLOCK_BYTES;
        const uint8_t* qs = blk + 1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int2 v = ts_dsv4_int_from_table16(ts_dsv4_get_int_b1(qs, j), ts_kvalues_mxfp4);
            w8[j + 0] = v.x;
            w8[j + 4] = v.y;
        }
        *scLo = __uint_as_float(((uint32_t)blk[0]) << 23) * 0.5f;
        *scHi = *scLo;
    }
    else if (wtype == TS_DSV4_WTYPE_Q6_K)
    {
        const int sb = sub >> 3;
        const int ls = sub & 7;
        const uint8_t* sblock = wRow + (size_t)sb * TS_DSV4_Q6_K_BLOCK_BYTES;
        const uint8_t* ql = sblock;
        const uint8_t* qh = sblock + 128;
        const int8_t* scales = (const int8_t*)(sblock + 192);
        const float dSb = __half2float(*(const half*)(sblock + 208));

        const int halfIdx = ls >> 2;
        const int group = ls & 3;
        const uint8_t* qlGroup = ql + halfIdx * 64 + ((group & 1) ? 32 : 0);
        const uint8_t* qhGroup = qh + halfIdx * 32;
        const int qlShift = group >= 2 ? 4 : 0;
        const int qhShift = group * 2;

#pragma unroll
        for (int lb = 0; lb < 4; ++lb)
        {
            const int g = lb * 2;
            const int raw0 = (((unsigned)ts_dsv4_get_int_b1(qlGroup, g) >> qlShift) & 0x0F0F0F0F)
                | ((((unsigned)ts_dsv4_get_int_b1(qhGroup, g) >> qhShift) & 0x03030303) << 4);
            const int raw1 = (((unsigned)ts_dsv4_get_int_b1(qlGroup, g + 1) >> qlShift) & 0x0F0F0F0F)
                | ((((unsigned)ts_dsv4_get_int_b1(qhGroup, g + 1) >> qhShift) & 0x03030303) << 4);
            w8[g + 0] = __vsubss4(raw0, 0x20202020);
            w8[g + 1] = __vsubss4(raw1, 0x20202020);
        }
        *scLo = dSb * (float)scales[halfIdx * 8 + group * 2 + 0];
        *scHi = dSb * (float)scales[halfIdx * 8 + group * 2 + 1];
    }
    else // Q8_0
    {
        const uint8_t* blk = wRow + (size_t)sub * TS_DSV4_Q8_0_BLOCK_BYTES;
        const int8_t* src = (const int8_t*)(blk + 2);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w8[j] = ts_dsv4_get_int_b2(src, j);
        *scLo = __half2float(*(const half*)blk);
        *scHi = *scLo;
    }
}

// One warp: decode row `wRow` into registers, then dot it against `m` member
// tokens' dense split-q8_1 activations, writing one output per token.
// NK = sub-blocks per lane = inDim / 1024, a compile-time constant so the
// staged weights stay in registers.
// NR = weight rows staged per warp. Each activation load then feeds NR rows,
// which is what keeps this off the L2 bandwidth ceiling: with one row per warp
// every (row, token) pair pulls the whole activation row again.
template <int NK, int NR>
__device__ __forceinline__ void ts_dsv4_expert_row_tokens(
    const uint8_t* __restrict__ wBase, long long rowBytes, int wtype, int lane,
    const int8_t* __restrict__ actQs, const float* __restrict__ actD,
    const int32_t* __restrict__ slotToken, int off, int m,
    int inDim, int rowLimit,
    float* __restrict__ out, long long outStride, int outRow0)
{
    int w[NR][NK][8];
    float scLo[NR][NK], scHi[NR][NK];
    bool live[NR];
#pragma unroll
    for (int rr = 0; rr < NR; ++rr)
    {
        live[rr] = (outRow0 + rr) < rowLimit;
        const uint8_t* wRow = wBase + (size_t)(outRow0 + rr) * rowBytes;
#pragma unroll
        for (int k = 0; k < NK; ++k)
        {
            if (live[rr])
                ts_dsv4_decode_sub(wRow, wtype, lane + 32 * k, w[rr][k], &scLo[rr][k], &scHi[rr][k]);
        }
    }

    const int nSub = inDim / 32;
    for (int i = 0; i < m; ++i)
    {
        // slotToken == nullptr: activations are already packed in slot order
        const int arow = slotToken != nullptr ? slotToken[off + i] : (off + i);
        const int8_t* aQs = actQs + (size_t)arow * inDim;
        const float* aD = actD + (size_t)arow * nSub;

        float sum[NR];
#pragma unroll
        for (int rr = 0; rr < NR; ++rr)
            sum[rr] = 0.0f;

#pragma unroll
        for (int k = 0; k < NK; ++k)
        {
            const int b = lane + 32 * k;
            const int4 a0 = *(const int4*)(aQs + (size_t)b * 32);
            const int4 a1 = *(const int4*)(aQs + (size_t)b * 32 + 16);
            const float dAct = aD[b];
#pragma unroll
            for (int rr = 0; rr < NR; ++rr)
            {
                int lo = 0, hi = 0;
                lo = ts_dsv4_dp4a(w[rr][k][0], a0.x, lo);
                lo = ts_dsv4_dp4a(w[rr][k][1], a0.y, lo);
                lo = ts_dsv4_dp4a(w[rr][k][2], a0.z, lo);
                lo = ts_dsv4_dp4a(w[rr][k][3], a0.w, lo);
                hi = ts_dsv4_dp4a(w[rr][k][4], a1.x, hi);
                hi = ts_dsv4_dp4a(w[rr][k][5], a1.y, hi);
                hi = ts_dsv4_dp4a(w[rr][k][6], a1.z, hi);
                hi = ts_dsv4_dp4a(w[rr][k][7], a1.w, hi);
                sum[rr] += dAct * (scLo[rr][k] * (float)lo + scHi[rr][k] * (float)hi);
            }
        }

#pragma unroll
        for (int rr = 0; rr < NR; ++rr)
        {
            const float s = ts_dsv4_warp_sum(sum[rr]);
            if (lane == 0 && live[rr])
                out[(size_t)(off + i) * outStride + outRow0 + rr] = s;
        }
    }
}
// ---------------------------------------------------------------------------

#define TS_DSV4_STAGE_WARPS 8

// Decode one weight row into (qs32, scale) staging buffers. Lane-parallel over
// 32-value sub-blocks.
__device__ __forceinline__ void ts_dsv4_stage_row(
    const uint8_t* __restrict__ wRow, int wtype, int inDim, int lane,
    int* __restrict__ qs32, float* __restrict__ scale)
{
    const int nSub = inDim / 32;

    if (wtype == TS_DSV4_WTYPE_IQ3S)
    {
        // one lane per 32-value sub-block; 8 sub-blocks share a superblock
        for (int sub = lane; sub < nSub; sub += 32)
        {
            const int sb = sub >> 3;
            const int iqs = (sub & 7) * 2;
            const uint8_t* blk = wRow + (size_t)sb * TS_DSV4_IQ3S_BLOCK_BYTES;
            const uint8_t* qs = blk + 2;
            const uint8_t* qh = blk + 2 + 64;
            const uint8_t* signs = blk + 2 + 64 + 8;
            const uint8_t* scales = blk + 2 + 64 + 8 + 32;
            const float d = __half2float(*(const half*)blk);

            int qs_pack0 = ts_dsv4_get_int_b2(qs, iqs + 0);
            int qs_pack1 = ts_dsv4_get_int_b2(qs, iqs + 1);
            const uint8_t* qsp0 = (const uint8_t*)&qs_pack0;
            const uint8_t* qsp1 = (const uint8_t*)&qs_pack1;
            uint8_t qsb[8];
            qsb[0] = qsp0[0]; qsb[1] = qsp0[1]; qsb[2] = qsp0[2]; qsb[3] = qsp0[3];
            qsb[4] = qsp1[0]; qsb[5] = qsp1[1]; qsb[6] = qsp1[2]; qsb[7] = qsp1[3];

            const int qhv = qh[iqs / 2];
            const int signs_packed_32 = ts_dsv4_get_int_b2(signs, iqs / 2);
            const uint8_t* sp8 = (const uint8_t*)&signs_packed_32;

#pragma unroll
            for (int l0 = 0; l0 < 8; l0 += 2)
            {
                const uint32_t g0 = ts_iq3s_grid[qsb[l0 + 0] | ((qhv << (8 - l0)) & 0x100)];
                const uint32_t g1 = ts_iq3s_grid[qsb[l0 + 1] | ((qhv << (7 - l0)) & 0x100)];
                const int signs0 = __vcmpne4(((sp8[l0/2] & 0x03) << 7) | ((sp8[l0/2] & 0x0C) << 21), 0x00000000);
                const int signs1 = __vcmpne4(((sp8[l0/2] & 0x30) << 3) | ((sp8[l0/2] & 0xC0) << 17), 0x00000000);
                qs32[(l0 + 0) * nSub + sub] = __vsub4((int)g0 ^ signs0, signs0);
                qs32[(l0 + 1) * nSub + sub] = __vsub4((int)g1 ^ signs1, signs1);
            }
            scale[sub] = d * (float)(1 + 2 * ((scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F));
        }
    }
    else if (wtype == TS_DSV4_WTYPE_MXFP4)
    {
        for (int b = lane; b < nSub; b += 32)
        {
            const uint8_t* blk = wRow + (size_t)b * TS_DSV4_MXFP4_BLOCK_BYTES;
            const uint8_t* qs = blk + 1;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int2 v = ts_dsv4_int_from_table16(ts_dsv4_get_int_b1(qs, j), ts_kvalues_mxfp4);
                qs32[(j + 0) * nSub + b] = v.x;
                qs32[(j + 4) * nSub + b] = v.y;
            }
            scale[b] = __uint_as_float(((uint32_t)blk[0]) << 23) * 0.5f;
        }
    }
    else if (wtype == TS_DSV4_WTYPE_Q6_K)
    {
        for (int sub = lane; sub < nSub; sub += 32)
        {
            const int sb = sub >> 3;
            const int ls = sub & 7;
            const uint8_t* sblock = wRow + (size_t)sb * TS_DSV4_Q6_K_BLOCK_BYTES;
            const uint8_t* ql = sblock;
            const uint8_t* qh = sblock + 128;
            const int8_t* scales = (const int8_t*)(sblock + 192);
            const float dSb = __half2float(*(const half*)(sblock + 208));

            const int halfIdx = ls >> 2;
            const int group = ls & 3;
            const uint8_t* qlGroup = ql + halfIdx * 64 + ((group & 1) ? 32 : 0);
            const uint8_t* qhGroup = qh + halfIdx * 32;
            const int qlShift = group >= 2 ? 4 : 0;
            const int qhShift = group * 2;

            // A Q6_K 32-value group carries TWO scales (values 0-15 and 16-31),
            // so the staged values stay raw (range -32..31) and both scales are
            // kept: scale[sub] for the low half, scale[nSub + sub] for the high.
#pragma unroll
            for (int lb = 0; lb < 4; ++lb)
            {
                const int g = lb * 2;
                const int raw0 = (((unsigned)ts_dsv4_get_int_b1(qlGroup, g) >> qlShift) & 0x0F0F0F0F)
                    | ((((unsigned)ts_dsv4_get_int_b1(qhGroup, g) >> qhShift) & 0x03030303) << 4);
                const int raw1 = (((unsigned)ts_dsv4_get_int_b1(qlGroup, g + 1) >> qlShift) & 0x0F0F0F0F)
                    | ((((unsigned)ts_dsv4_get_int_b1(qhGroup, g + 1) >> qhShift) & 0x03030303) << 4);
                const int sc = scales[halfIdx * 8 + group * 2 + (lb >= 2 ? 1 : 0)];
                qs32[(g + 0) * nSub + sub] = __vsubss4(raw0, 0x20202020);
                qs32[(g + 1) * nSub + sub] = __vsubss4(raw1, 0x20202020);
                if (lb == 0)
                    scale[sub] = dSb * (float)sc;
                if (lb == 2)
                    scale[nSub + sub] = dSb * (float)sc;
            }
        }
    }
    else // Q8_0
    {
        for (int b = lane; b < nSub; b += 32)
        {
            const uint8_t* blk = wRow + (size_t)b * TS_DSV4_Q8_0_BLOCK_BYTES;
            const int8_t* src = (const int8_t*)(blk + 2);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                qs32[j * nSub + b] = ts_dsv4_get_int_b2(src, j);
            scale[b] = __half2float(*(const half*)blk);
        }
    }
}

// Dot a staged row against one q8_1-quantized activation row.
__device__ __forceinline__ float ts_dsv4_dot_staged(
    const int* __restrict__ qs32, const float* __restrict__ scale,
    const ts_dsv4_block_q8_1* __restrict__ act, int inDim, int wtype, int lane)
{
    const int nSub = inDim / 32;
    float sum = 0.0f;
    for (int b = lane; b < nSub; b += 32)
    {
        const int* a = (const int*)act[b].qs;
        const float dAct = __half2float(act[b].d);

        if (wtype == TS_DSV4_WTYPE_Q6_K)
        {
            // two 16-value halves carry different sub-block scales
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                s0 = ts_dsv4_dp4a(qs32[j * nSub + b], a[j], s0);
#pragma unroll
            for (int j = 4; j < 8; ++j)
                s1 = ts_dsv4_dp4a(qs32[j * nSub + b], a[j], s1);
            sum += dAct * (scale[b] * (float)s0 + scale[nSub + b] * (float)s1);
        }
        else
        {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sumi = ts_dsv4_dp4a(qs32[j * nSub + b], a[j], sumi);
            sum += scale[b] * dAct * (float)sumi;
        }
    }
    return ts_dsv4_warp_sum(sum);
}

// Staged gate/up. grid.x = row tiles, grid.y = expert, grid.z = 0 gate / 1 up.
// grid.y carries the expert because blockIdx.x varies fastest at launch, so a
// single expert's blocks stay co-resident and share its activations in cache.
extern "C" __global__ void ts_dsv4_moe_gateup_staged_f32(
    const uint8_t* __restrict__ gateW,
    const uint8_t* __restrict__ upW,
    const int8_t* __restrict__ actQs,
    const float* __restrict__ actD,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ slotToken,
    float* __restrict__ gateOut,
    float* __restrict__ upOut,
    const int wtype,
    const int ff,
    const int inDim,
    const long long rowBytes)
{
    const int e = blockIdx.y;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int r = (blockIdx.x * TS_DSV4_STAGE_WARPS + warpId) * TS_DSV4_STAGE_ROWS;
    if (r >= ff)
        return;

    const bool isUp = blockIdx.z == 1;
    const uint8_t* wBase = (isUp ? upW : gateW) + (size_t)e * ff * rowBytes;
    float* outBase = isUp ? upOut : gateOut;
    const int off = offsets[e];

    switch (inDim / 1024)
    {
        case 4:
            ts_dsv4_expert_row_tokens<4, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
        case 2:
            ts_dsv4_expert_row_tokens<2, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
        default:
            ts_dsv4_expert_row_tokens<1, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
    }
}

// Staged down projection over packed hidden rows (already in slot order, so no
// slotToken indirection).
extern "C" __global__ void ts_dsv4_moe_down_staged_f32(
    const uint8_t* __restrict__ downW,
    const int8_t* __restrict__ actQs,
    const float* __restrict__ actD,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    float* __restrict__ downOut,
    const int wtype,
    const int E,
    const int ff,
    const long long rowBytes)
{
    const int e = blockIdx.y;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int r = (blockIdx.x * TS_DSV4_STAGE_WARPS + warpId) * TS_DSV4_STAGE_ROWS;
    if (r >= E)
        return;

    const uint8_t* wBase = downW + (size_t)e * E * rowBytes;
    const int off = offsets[e];

    switch (ff / 1024)
    {
        case 4:
            ts_dsv4_expert_row_tokens<4, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
        case 2:
            ts_dsv4_expert_row_tokens<2, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
        default:
            ts_dsv4_expert_row_tokens<1, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
    }
}

// Expert gate/up projections over grouped tokens. grid (nExpert, ffTiles, 2);
// z = 0 -> gate, z = 1 -> up. Each warp sweeps a weight row and dots it
// against every member token's q8_1 activation, so weight bytes are read once
// per chunk while activations come from L2.
#define TS_DSV4_MOE_TILE 32

extern "C" __global__ void ts_dsv4_moe_gateup_f32(
    const uint8_t* __restrict__ gateW,   // [nExpert, ff, inDim]
    const uint8_t* __restrict__ upW,
    const ts_dsv4_block_q8_1* __restrict__ act, // [nt] rows
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ slotToken,
    float* __restrict__ gateOut,         // [S, ff]
    float* __restrict__ upOut,           // [S, ff]
    const int wtype,
    const int ff,
    const int inDim,
    const long long rowBytes)
{
    const int e = blockIdx.x;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const bool isUp = blockIdx.z == 1;
    const uint8_t* wBase = (isUp ? upW : gateW) + (size_t)e * ff * rowBytes;
    float* outBase = isUp ? upOut : gateOut;
    const int off = offsets[e];
    const int actBlocks = inDim / TS_DSV4_QK8_1;

    for (int rr = rBegin; rr < rEnd && rr < ff; ++rr)
    {
        const uint8_t* wRow = wBase + (size_t)rr * rowBytes;
        for (int i = 0; i < m; ++i)
        {
            const int tok = slotToken[off + i];
            const float dot = ts_dsv4_dot_row_warp(wRow, wtype, act + (size_t)tok * actBlocks, inDim, lane);
            if (lane == 0)
                outBase[(size_t)(off + i) * ff + rr] = dot;
        }
    }
}

// Expert down projection over grouped (already packed) hidden rows.
extern "C" __global__ void ts_dsv4_moe_down_f32(
    const uint8_t* __restrict__ downW,   // [nExpert, E, ff]
    const ts_dsv4_block_q8_1* __restrict__ act, // [S] packed rows
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    float* __restrict__ downOut,         // [S, E]
    const int wtype,
    const int E,
    const int ff,
    const long long rowBytes)
{
    const int e = blockIdx.x;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const uint8_t* wBase = downW + (size_t)e * E * rowBytes;
    const int off = offsets[e];
    const int actBlocks = ff / TS_DSV4_QK8_1;

    for (int rr = rBegin; rr < rEnd && rr < E; ++rr)
    {
        const uint8_t* wRow = wBase + (size_t)rr * rowBytes;
        for (int i = 0; i < m; ++i)
        {
            const float dot = ts_dsv4_dot_row_warp(wRow, wtype, act + (size_t)(off + i) * actBlocks, ff, lane);
            if (lane == 0)
                downOut[(size_t)(off + i) * E + rr] = dot;
        }
    }
}

// Decode-specialized expert projections: grid maps directly over the selected
// experts (no grouping plan). z = 0 -> gate, z = 1 -> up.
extern "C" __global__ void ts_dsv4_moe_gateup_decode_f32(
    const uint8_t* __restrict__ gateW,
    const uint8_t* __restrict__ upW,
    const ts_dsv4_block_q8_1* __restrict__ act, // single row
    const int32_t* __restrict__ sel,     // [nUsed]
    float* __restrict__ gateOut,         // [nUsed, ff]
    float* __restrict__ upOut,           // [nUsed, ff]
    const int wtype,
    const int ff,
    const int inDim,
    const long long rowBytes)
{
    const int slot = blockIdx.x;
    const int e = sel[slot];

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const bool isUp = blockIdx.z == 1;
    const uint8_t* wBase = (isUp ? upW : gateW) + (size_t)e * ff * rowBytes;
    float* outBase = isUp ? upOut : gateOut;

    for (int rr = rBegin; rr < rEnd && rr < ff; ++rr)
    {
        const float dot = ts_dsv4_dot_row_warp(wBase + (size_t)rr * rowBytes, wtype, act, inDim, lane);
        if (lane == 0)
            outBase[(size_t)slot * ff + rr] = dot;
    }
}

extern "C" __global__ void ts_dsv4_moe_down_decode_f32(
    const uint8_t* __restrict__ downW,
    const ts_dsv4_block_q8_1* __restrict__ act, // [nUsed] rows
    const int32_t* __restrict__ sel,
    float* __restrict__ downOut,         // [nUsed, E]
    const int wtype,
    const int E,
    const int ff,
    const long long rowBytes)
{
    const int slot = blockIdx.x;
    const int e = sel[slot];

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const uint8_t* wBase = downW + (size_t)e * E * rowBytes;
    const int actBlocks = ff / TS_DSV4_QK8_1;

    for (int rr = rBegin; rr < rEnd && rr < E; ++rr)
    {
        const float dot = ts_dsv4_dot_row_warp(wBase + (size_t)rr * rowBytes, wtype, act + (size_t)slot * actBlocks, ff, lane);
        if (lane == 0)
            downOut[(size_t)slot * E + rr] = dot;
    }
}

// gate = swiglu_clamp(gate, up): up clamped to [-limit, limit], gate to
// (-inf, limit], result = silu(gate) * up.
extern "C" __global__ void ts_dsv4_swiglu_clamp_f32(
    float* __restrict__ gate,
    const float* __restrict__ up,
    const long long n,
    const float limit,
    const int doClamp)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float g = gate[i];
    float u = up[i];
    if (doClamp)
    {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    gate[i] = u * (g / (1.0f + expf(-g)));
}

// ffnOut[t] = sum_j w[t,j] * downOut[row(t,j)] (+ shexp[t]); rowOfSlot null
// means identity (decode: slot j -> row j).
extern "C" __global__ void ts_dsv4_moe_scatter_add_f32(
    const float* __restrict__ downOut,
    const int32_t* __restrict__ rowOfSlot, // [nt*nUsed] or null
    const float* __restrict__ wSel,        // [nt, nUsed]
    const float* __restrict__ shexp,       // [nt, E] or null
    float* __restrict__ ffnOut,            // [nt, E]
    const int nt,
    const int nUsed,
    const int E)
{
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || e >= E)
        return;

    float acc = shexp != nullptr ? shexp[(size_t)t * E + e] : 0.0f;
    for (int j = 0; j < nUsed; ++j)
    {
        const int slot = t * nUsed + j;
        const int row = rowOfSlot != nullptr ? rowOfSlot[slot] : slot;
        acc += downOut[(size_t)row * E + e] * wSel[slot];
    }
    ffnOut[(size_t)t * E + e] = acc;
}
