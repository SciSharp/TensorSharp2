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

#include <cstdint>
#include <mutex>

// ============================================================================
// Device-resident KV windows shared by the GPT-OSS kernels.
//
// One window per host KV cache buffer (K and V are separate host allocations,
// so the host pointer identifies the window). Both the per-layer attention
// kernel (TSGgml_GptOssAttentionLayerPrefill, used for prefill) and the
// whole-model decode graph (TSGgml_GptOssModelDecode) attach to the SAME
// window, so there is exactly one device copy of each cache and the two paths
// stay coherent without copying through host memory between them.
//
// `rows_valid` records how many leading rows of the device copy are known good.
// Prefill uploads [rows_valid, start_pos) when a caller rewinds or jumps, then
// downloads the rows it just wrote so the host stays a valid mirror. Decode
// deliberately skips that download (it is the whole point of the fused graph),
// so it leaves the host stale and the model marks its cache host-dirty; a host
// reader syncs via TSGgml_GptOssSyncKvCacheToHost first.
//
// Definitions live in ggml_ops_transformer_prefill.cpp.
// ============================================================================
namespace tsg_gptoss
{
    struct KvWindow
    {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* tensor = nullptr;      // [head_dim, capacity, kv_heads]
        ggml_backend_t backend = nullptr;
        std::int64_t capacity = 0;
        std::int64_t rows_valid = 0;
        int head_dim = 0;
        int kv_heads = 0;
        int type = -1;
    };

    // Registry lock. Every entry point below expects it to be held.
    std::mutex& kv_mutex();

    // Window for `host_cache` sized to hold at least `needed_rows`, allocating
    // (or growing) it when required. Returns null when the allocation fails.
    KvWindow* kv_acquire(const void* host_cache, int head_dim, int kv_heads,
                         ggml_type type, std::int64_t needed_rows, std::int64_t cache_rows);

    // Existing window for `host_cache`, or null. Never allocates.
    KvWindow* kv_find(const void* host_cache);

    // Release the window(s) for the given host cache pointer(s).
    void kv_drop_locked(const void* host_cache);
    void kv_drop_pair_locked(const void* k_cache, const void* v_cache);
    void kv_drop_all_locked();
}
