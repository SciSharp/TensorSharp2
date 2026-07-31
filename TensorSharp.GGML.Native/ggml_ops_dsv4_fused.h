// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Fused DeepSeek V4 (Flash) ops, injected into the ggml graph as
// GGML_OP_CUSTOM nodes and executed by a small TensorSharp-owned ggml-backend
// that launches CUDA kernels on the paired ggml-cuda backend's stream. This
// keeps all custom-kernel work inside the TensorSharp native project — the
// vendored ggml tree is used strictly through its public backend API.
//
// Each fused node's op_params carry a ggml_custom_op_params whose userdata
// points at a tsg_dsv4_fused_desc (owned by the graph build result). The
// descs double as the dispatch table for both the CUDA launcher and the CPU
// reference fallback.
// ---------------------------------------------------------------------------

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

enum tsg_dsv4_fused_kind : int32_t
{
    TSG_DSV4_FUSED_COMPRESS      = 1,
    TSG_DSV4_FUSED_ATTN_PREP     = 2,
    TSG_DSV4_FUSED_ATTN_FINISH   = 3,
    TSG_DSV4_FUSED_MOE_TOPK      = 4,
    TSG_DSV4_FUSED_MOE_WEIGHTS   = 5,
    TSG_DSV4_FUSED_EXPERT_REDUCE = 6,
    TSG_DSV4_FUSED_SWIGLU_CLAMP  = 7,
    TSG_DSV4_FUSED_HC_GATES      = 8,
    TSG_DSV4_FUSED_TOPK_MASK     = 9,
    TSG_DSV4_FUSED_TOPK_SELECT   = 10,
    TSG_DSV4_FUSED_KGATHER       = 11,
};

#define TSG_DSV4_FUSED_MAGIC 0x5453445356344655ull  // "TSDSV4FU"

struct tsg_dsv4_fused_desc
{
    uint64_t magic = TSG_DSV4_FUSED_MAGIC;
    int32_t  kind = 0;
    int32_t  i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    float    f0 = 0.0f, f1 = 0.0f;
};

// CPU reference implementation (ggml_custom_op_t signature); used when a
// fused node is scheduled onto the CPU backend.
void tsg_dsv4_fused_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata);

#ifdef TSG_GGML_USE_CUDA
// Create a fused-op backend bound to `cuda_backend`'s device and stream.
// The returned backend claims support for GGML_OP_CUSTOM nodes carrying a
// tsg_dsv4_fused_desc and for the CUDA device's default buffer type, so
// ggml_backend_sched interleaves it with the CUDA backend with no copies and
// no synchronization (everything runs in order on one stream).
// The cuda_backend must outlive the returned backend.
ggml_backend_t tsg_dsv4_fused_backend_init(ggml_backend_t cuda_backend);
#endif
