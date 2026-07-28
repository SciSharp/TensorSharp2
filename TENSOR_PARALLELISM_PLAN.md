# Tensor Parallelism Plan

## Overview

This document describes the plan for adding tensor parallelism (TP) to TensorSharp,
progressing from local multi-GPU parallelism through network-distributed inference
with shared state, to RDMA-class memory access if the performance profile warrants it.

---

## Stage 1 — Local Tensor Parallelism (Implemented)

**Goal:** Split a single model across multiple CUDA GPUs within one process, using
the Megatron-LM column/row-parallel pattern.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Process                                                │
│                                                         │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐        │
│  │ GPU 0    │  │ GPU 1    │  ...  │ GPU N-1  │        │
│  │ Allocator│  │ Allocator│       │ Allocator│        │
│  │ Stream   │  │ Stream   │       │ Stream   │        │
│  │ Weights  │  │ Weights  │       │ Weights  │        │
│  │ (shard)  │  │ (shard)  │       │ (shard)  │        │
│  │ KV Cache │  │ KV Cache │       │ KV Cache │        │
│  └────┬─────┘  └────┬─────┘       └────┬─────┘        │
│       │              │                  │              │
│       └──────────────┼──────────────────┘              │
│                      │                                 │
│              P2P AllReduce                             │
│         (cuMemcpyPeerAsync + add kernel)               │
└─────────────────────────────────────────────────────────┘
```

### Transformer Block TP Pattern

```
Replicated hidden state (all GPUs hold identical copy)
        │
        ▼
  ┌─ RMSNorm (replicated, each GPU independently) ─┐
  │                                                 │
  ▼                                                 │
  Column-Parallel QKV (split output heads)          │
  │  GPU 0: heads [0..H/tp)                        │
  │  GPU 1: heads [H/tp..2H/tp)                    │
  │  ...                                           │
  ▼                                                 │
  Per-GPU Attention (independent head subsets)      │
  │  Each GPU: QK norm, RoPE, KV cache, SDPA       │
  ▼                                                 │
  Row-Parallel Output Proj + AllReduce ─────────────┘
  │
  ▼
  Replicated hidden state (restored by AllReduce)
        │
        ▼
  ┌─ RMSNorm (replicated) ─────────────────────────┐
  │                                                 │
  ▼                                                 │
  Column-Parallel Gate/Up (split intermediate dim)  │
  │                                                 │
  ▼                                                 │
  Per-GPU SiLU·mul (independent)                    │
  │                                                 │
  ▼                                                 │
  Row-Parallel Down + AllReduce ────────────────────┘
  │
  ▼
  Replicated hidden state
```

### Files Created / Modified

| File | Change |
|------|--------|
| `TensorSharp.Backends.Cuda/Interop/CudaDriverApi.cs` | Added P2P bindings: `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`, `cuMemcpyPeerAsync`, `cuEventSynchronize` |
| `TensorSharp.Backends.Cuda/CudaEvent.cs` | **New.** CUDA event wrapper for cross-stream/device synchronization |
| `TensorSharp.Backends.Cuda/CudaP2PCommunicator.cs` | **New.** AllReduce via P2P copies + elementwise-add kernel. Reduce-to-zero + broadcast algorithm. Host-memory fallback when P2P is unavailable |
| `TensorSharp.Backends.Cuda/TensorParallelGroup.cs` | **New.** Multi-GPU coordinator: owns N `CudaAllocator`s + P2P communicator. Public API: `AllReduce(Tensor[])`, `GetAllocator(rank)`, `Synchronize()` |
| `TensorSharp.Models/ModelBase.cs` | Added TP fields, `tpDegree` constructor param, `ShardWeightsForTensorParallelism()`, `TpColumnParallelLinear()`, `TpRowParallelLinear()`, `TpRMSNorm()`, `TpResidualAdd()`, `BroadcastTensorToAllRanks()`, `PrepareCudaQuantizedWeightsForInferenceTP()`, TP cleanup in `Dispose()`, `Create()` accepts `tpDegree` |
| `TensorSharp.Models/Models/Qwen3/Qwen3Model.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active, TP KV cache init/dispose |
| `TensorSharp.Models/Models/Qwen3/Qwen3Model.TensorParallel.cs` | **New.** TP forward pass: `ForwardTP`, `TransformerBlockTP`, `AttentionTP`, per-GPU KV cache management |
| `TensorSharp.Models/Models/Mistral3/Mistral3Model.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Mistral3/Mistral3Model.TensorParallel.cs` | **New.** TP forward pass with fused/separate QKV, YaRN RoPE |
| `TensorSharp.Models/Models/Gemma3/Gemma3Model.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Gemma3/Gemma3Model.TensorParallel.cs` | **New.** TP forward pass with separate Q/K/V, GELU, sliding window, extra norms |
| `TensorSharp.Models/Models/Gemma4/Gemma4Model.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Gemma4/Gemma4Model.TensorParallel.cs` | **New.** TP forward pass with dense GeGLU + MoE dual-path FFN, per-layer head dims, shared KV layers |
| `TensorSharp.Models/Models/Qwen35/Qwen35Model.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Qwen35/Qwen35Model.TensorParallel.cs` | **New.** TP forward pass with GatedDeltaNet SSM + full attention + MoE, block-cyclic V-head assignment, CUDA-native GDN kernels |
| `TensorSharp.Models/Models/GptOss/GptOssModel.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/GptOss/GptOssModel.TensorParallel.cs` | **New.** TP forward pass with biased QKV, attention sinks, YaRN, clamped SiLU GLU MoE |
| `TensorSharp.Models/Models/Nemotron/NemotronModel.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Nemotron/NemotronModel.TensorParallel.cs` | **New.** TP forward pass with Mamba2 (replicated), attention (no RoPE), dense + MoE FFN |
| `TensorSharp.Cli/Program.cs` | Added `--tp <N>` CLI argument |
| `TensorSharp.Models/Models/Qwen3/Qwen3Model.BatchedForwardTP.cs` | **New.** TP batched paged-attention forward: per-rank paged KV buffers, TP column/row-parallel linears, per-rank ManagedPagedAttention |
| `TensorSharp.Models/Models/Mistral3/Mistral3Model.BatchedForwardTP.cs` | **New.** TP batched forward with YaRN RoPE, fused/separate QKV, position-dependent Q scaling |
| `TensorSharp.Models/Models/Qwen3/Qwen3Model.BatchedForward.cs` | Added TP dispatch to `ForwardBatchTP` when `IsTensorParallel` |
| `TensorSharp.Models/Models/Mistral3/Mistral3Model.BatchedForward.cs` | Added TP dispatch to `ForwardBatchTP` when `IsTensorParallel` |
| `TensorSharp.Models/Models/Gemma4/Gemma4Model.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |
| `TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |
| `TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |
| `TensorSharp.Models/Models/Nemotron/NemotronModel.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |

### Weight Sharding

| Weight Pattern | Parallel Type | Split Dimension | Notes |
|---------------|---------------|-----------------|-------|
| `attn_qkv.weight` | Column | ne1 (output) | Consecutive rows → zero-copy view |
| `ffn_gate_up.weight` | Column | ne1 (output) | Consecutive rows → zero-copy view |
| `attn_output.weight` | Row | ne0 (input) | Block-aligned column extraction |
| `ffn_down.weight` | Row | ne0 (input) | Block-aligned column extraction |
| `*_norm.weight` | Replicated | — | Full copy on every GPU |
| `token_embd.weight` | Replicated | — | Embedding lookup on GPU 0 |
| `output.weight` | Replicated | — | LM head on GPU 0 (post-AllReduce) |

Quantized weights (Q4_0, Q8_0, etc.) are split at block boundaries (32 elements).
Column-parallel splits are zero-copy views into the original mmap'd data.
Row-parallel splits copy the relevant blocks per row into new aligned buffers.

### Usage

```bash
# CLI: 2-GPU tensor parallelism
TensorSharp.Cli --model qwen3-8b.gguf --backend cuda --tp 2

# Server: via environment variable
TENSORSHARP_TP_DEGREE=2 dotnet run --project TensorSharp.Server

# Config JSON (auto-expanded to CLI args)
{ "tp": 2, "backend": "cuda", "model": "qwen3-8b.gguf" }
```

### Constraints

- Direct CUDA and the GGML CUDA/Vulkan backends (see Stage 1b); MLX is single-device
- `numHeads` and `numKVHeads` must be divisible by TP degree
- `intermediateSize` must be divisible by TP degree
- Quantized row-parallel splits require `ne0` divisible by `tp × blockSize`
- Single-process: one thread issues commands to all GPUs sequentially; CUDA
  streams provide the actual parallelism; AllReduce is the synchronization point

### MoE and SSM TP Strategies

MoE and SSM architectures required non-standard TP approaches beyond the
column/row-parallel pattern:

| Architecture | Strategy | Details |
|-------------|----------|---------|
| **MoE (Gemma4, Qwen3.5, GptOss, Nemotron)** | Expert slicing | Each GPU holds 1/tp slice of every expert's weights (column-parallel gate/up, row-parallel down). Router is replicated. AllReduce after weighted expert sum. |
| **GatedDeltaNet SSM (Qwen3.5)** | Per-rank V-head ownership | Block-cyclic V-head assignment. Each rank runs its own GDN kernel on its V-head subset with independent delta/conv state — no cross-rank communication needed for the recurrent path. Requires CUDA-native GDN kernels (`ts_qwen35_gdn_*`). |
| **Mamba2 SSM (Nemotron)** | Replicated on rank 0 | Mamba2 layers run on rank 0 only, result broadcast to all ranks. SSM state lives in host arrays with a managed per-token loop; sharding would require a device-resident per-rank kernel. Attention + FFN/MoE layers hold the bulk of weights, so TP still delivers most memory savings. |

### Known Limitations / Future Work (Stage 1)

- [x] Extend TP to other model architectures (Gemma4, Qwen3.5, Mistral3, etc.)
- [x] TP-aware batched forward (Qwen3, Mistral3 implemented; MoE models fall back to per-seq)
- [ ] TP-aware batched forward for MoE models (Gemma4, Qwen3.5, GptOss, Nemotron)
- [ ] TP-aware CUDA graph capture (current graphs are single-device)
- [ ] Overlap AllReduce with next layer's norm (pipeline communication)
- [ ] NCCL backend for Linux (higher throughput than P2P for N > 2)
- [ ] Column-parallel LM head with AllGather (currently replicated on GPU 0)
- [ ] GptOss: expert down-projection bias is skipped in TP mode (small correction, documented as follow-up)
- [ ] Nemotron: Mamba2 layers are replicated on rank 0 rather than sharded (requires device-resident SSM kernel)

---

## Stage 1b — Tensor Parallelism on the GGML Backends (Implemented)

**Goal:** Let the GGML CUDA / Vulkan backends span several GPUs (and, via
Stage 2's TCP layer, several nodes) so a model larger than one GPU's VRAM can
run without falling back to CPU offload.

### Why this needed its own stage

The GGML backend has a different execution model from direct CUDA. Tensors live
in **host** memory (`GgmlStorage` allocates from an mmap/VirtualAlloc pool); ops
upload to the device, run a ggml graph, and copy the result back. There was also
exactly one ggml backend per process (`g_backend`), so "which GPU" was not a
concept the bridge had.

Three things had to change:

1. **One ggml backend per GPU.** `tsg::g_device_states[]` holds a backend plus
   its own buffer caches per rank; the active rank is a `thread_local`, so a
   worker pool can drive several GPUs at once. `g_backend` and the cache globals
   became macros onto the active slot, which kept ~500 existing use sites
   working unchanged.
2. **Per-rank device residency.** The host-pointer-keyed buffer caches are now
   per rank. This matters for *replicated* weights: one host pointer, but a
   distinct device copy per GPU. Without it rank 1 would be handed a buffer
   living on rank 0's card.
3. **A collective.** `ggml_backend_reg_get_proc_address("ggml_backend_comm_*")`
   exposes ggml-cuda's AllReduce — NCCL when available, its P2P pipeline
   otherwise, butterfly as a last resort. This is the same collective
   llama.cpp's `--split-mode tensor` uses. A multi-threaded host reduction is
   the fallback for backends without one, and is the *faster* choice for small
   payloads: TensorSharp's activations are already in host RAM, so summing them
   there costs nothing extra, while the device path pays two PCIe crossings.
   `TS_GGML_TP_DEVICE_AR_THRESHOLD` (default 256 Ki elements) picks between them.

### Files Created / Modified

| File | Change |
|------|--------|
| `TensorSharp.GGML.Native/ggml_ops_internal.h` | `DeviceState` table, `TSG_MAX_DEVICES`, `thread_local g_active_rank`, `ScopedRank`, compatibility macros for the old globals |
| `TensorSharp.GGML.Native/ggml_ops_core.cpp` | Per-rank backend creation (`create_backend_instance_on_device`, `gpu_device_count`), per-rank reuse buffers/gallocr, rank-wide teardown and budget config |
| `TensorSharp.GGML.Native/ggml_ops_tensor_parallel.cpp` | **New.** TP init, collective resolution, device + host AllReduce, and the fused multi-rank matmul |
| `TensorSharp.GGML.Native/CMakeLists.txt` | Auto-detect NCCL instead of forcing `GGML_CUDA_NCCL=OFF` |
| `TensorSharp.Backends.GGML/GgmlContext.cs` | Accepts N device ids; exposes `Degree`, `DeviceIds`, `HasDeviceAllReduce` |
| `TensorSharp.Backends.GGML/GgmlTensorParallel.cs` | **New.** P/Invoke surface: device enumeration, rank selection, AllReduce, multi-rank matmul |
| `TensorSharp.Backends.GGML/GgmlTensorParallelGroup.cs` | **New.** `ITensorParallelGroup` over N ggml backends: per-rank allocators, rank worker pool, op dispatch hook |
| `TensorSharp.Core/ITensorParallelGroup.cs` | Moved out of the CUDA project; `GetAllocator` returns `IAllocator`; added `RunPerRank`; added `INestedTensorParallelGroup` |
| `TensorSharp.Core/OpRegistry.cs` | `PreInvokeHook` so a multi-device backend can route an op to its result's device |
| `TensorSharp.Distributed/DistributedTensorParallelGroup.cs` | Takes any local group, so multi-node works over GGML too |
| `TensorSharp.Models/ModelBase.cs` | GGML TP context/group construction, `TENSORSHARP_TP_DEVICES`, per-rank weight preload, GGML branch in `AddmmQuantManaged`, fused TP linear path |

### How an op finds its GPU

Two mechanisms, because rank is a property of a *block of work*, not of a
single op:

* `ITensorParallelGroup.RunPerRank(body)` pins each rank's worker thread to its
  GPU for the duration of `body`. This is the reliable base, and it is what
  makes the ranks actually run concurrently — GGML ops submit *and* synchronize
  in one call, so a sequential rank loop runs the GPUs strictly one after
  another and TP ends up slower than a single device.
* `OpRegistry.PreInvokeHook` selects the device from an op's result tensor.
  TensorSharp ops take the destination first, so the result's allocator names
  the rank. This catches ops issued outside a `RunPerRank` scope without every
  model file having to say which GPU it means.

### Two hazards the concurrent-rank design had to solve

**ggml-cuda's memory pool is not shareable between threads.**
`ggml_cuda_pool_vmm` is a lock-free bump allocator on the backend context that
asserts every free is the exact reverse of its alloc
(`GGML_ASSERT(ptr == pool_addr + pool_used)`). Rank fan-out is safe only because
each thread owns a *different* backend — so the per-op dispatch hook must never
move a worker onto another rank's backend. `RunPerRank` therefore **pins** the
thread's rank for the duration of the body, and the hook yields to an active pin
(`SetActiveRankIfUnpinned`). Without the pin, an op whose result happened to live
on another rank dragged the worker onto a backend its owner was still using and
tripped the assert mid-prefill.

**Preloaded weights are addressed by an opaque cache key.** Once C# pins a
quantized weight it hands the bridge a GCHandle, not a host pointer, so the
device copy survives the host buffer being released. Every lookup is keyed on it
— but the *miss* path uploaded from that same pointer, dereferencing a GCHandle
as if it were weight bytes and segfaulting inside `cudaMemcpyAsync`. The bridge
now records `cache key -> host data` at preload (`register_cache_key`) and
resolves it in `upload_binding`, so a miss is merely a re-upload. Set
`TS_GGML_LOG_VRAM=1` to see when that fires — each occurrence is a whole weight
re-uploaded, so a steady stream of them is a performance signal.

### CUDA graph capture

ggml-cuda records graphs with `cudaStreamBeginCapture`. Capture is process-wide:
while one rank's thread is capturing, an unsafe CUDA call from another rank's
thread poisons it (`operation failed due to a previous error during capture`).
Concurrent ranks are the point of TP here, so `TSGgml_TensorParallelInit` sets
`GGML_CUDA_DISABLE_GRAPHS=1` for multi-GPU runs with concurrent dispatch. It is
set natively rather than from C# because .NET's `Environment.SetEnvironmentVariable`
does not reach the native `getenv`.

### Usage

```bash
# 2 GPUs on the GGML CUDA backend
TensorSharp.Cli --model qwen3-8b.gguf --backend ggml_cuda --tp 2

# Pick specific GPUs
TENSORSHARP_TP_DEVICES=0,2 TensorSharp.Cli --model m.gguf --backend ggml_cuda --tp 2

# Multi-node (2 nodes x 2 local GPUs)
TensorSharp.Cli --model m.gguf --backend ggml_cuda --tp 2 \
  --tp-node-id 0 --tp-peers "10.0.0.1:9500,10.0.0.2:9500"
```

| Variable | Effect |
|---|---|
| `TENSORSHARP_TP_DEVICES` | Explicit GPU ordinals per rank (default `0..tp-1`) |
| `TS_GGML_TP_PARALLEL=0` | Sequential rank dispatch (diagnostic) |
| `TS_GGML_TP_FUSED_MATMUL=0` | Use the generic per-rank linear instead of the fused multi-rank one (diagnostic) |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | Element count above which AllReduce uses the device collective |
| `GGML_CUDA_ALLREDUCE` | `nccl` / `internal` / `none` — passed through to ggml |

### Measured results

2× RTX 2000 Ada (16 GB each, PCIe, no NVLink), CUDA 12.8, NCCL 2.25.

**Memory — the capacity win, verified:**

| Model | 1 GPU | TP=2 rank 0 | TP=2 rank 1 |
|---|---|---|---|
| Qwen3-4B Q8_0 | 4075 MB | 2234 MB | 1840 MB |
| Gemma-4-26B-A4B IQ4_XS | 12908 MB | 6835 MB | 6087 MB |

**Correctness:** on a fixed prompt with greedy decode, TP=2 and single-GPU
produce byte-identical text for short generations and diverge only after ~30
tokens, at a natural branch point, with both continuations coherent — the
expected consequence of summing the row-parallel partials in a different order.
Parallel and sequential rank dispatch are bit-identical to each other, which is
what rules out a race in the worker pool.

**Throughput (Qwen3-4B Q8_0, tokens/s):**

| | prefill 512 | prefill 2048 | decode |
|---|---|---|---|
| 1 GPU | 67.8 | 42.1 | 25.8 |
| TP=2 | 42.3 | 33.8 | 11.6 |

TP is currently **slower than a single GPU for models that fit on one**. This is
not an AllReduce cost — it is dispatch count. The single-GPU GGML path runs a
whole 36-layer decode as *one* fused native graph
(`TSGgml_TransformerModelDecode`); the TP path runs the generic op-at-a-time
forward, which is ~1000 native calls per token, doubled across two ranks. The
per-op overhead is fixed, which is why prefill closes the gap as the prompt
grows (0.62× at 512 tokens, 0.80× at 2048) while decode does not.

### Known Limitations / Future Work (Stage 1b)

- [ ] **Fused TP decode.** The decisive fix: a per-rank whole-model graph split
      at the two AllReduce points per layer, submitted asynchronously — the
      structure ggml's own meta backend (`ggml-backend-meta.cpp`, already
      vendored) implements for llama.cpp's `--split-mode tensor`. Wiring
      TensorSharp's fused decode onto the meta device would replace ~1000
      dispatches per token with ~2·L graph launches.
- [ ] **Qwen3.5 on GGML.** Its per-rank GatedDeltaNet is a single fused CUDA
      kernel (`ts_qwen35_gdn_*`); the GGML bridge exposes only the unpacked
      chunked and batched-step forms. Validation rejects the combination with an
      explicit message. The attention and MoE halves are already backend-agnostic.
- [ ] **MoE TP throughput.** Gemma4/GptOss walk experts per token per rank, so a
      512-token prefill issues hundreds of thousands of tiny matmuls. Pre-existing
      (it affects direct CUDA too); the ranks now at least run concurrently, and
      `WarmUpKernels` uses the short warmup for MoE-under-TP so startup no longer
      looks hung. Measured on Gemma-4-26B-A4B with `--tp 2`: ~2.2 tok/s decode.
      The fix is to give each rank a *stacked* shard of the expert weights so the
      existing `ggml_mul_mat_id` batched kernel (`MoEFFNPrefillSwiGLU`, already
      used on the single-GPU path) can run per rank — one call per layer instead
      of `tokens x experts_used x 3`.
- [ ] **Fused multi-rank linear at rank-count 1.** `TSGgml_TensorParallelMatmul`
      is correct for a genuine multi-GPU group (single-node TP=2 output is
      byte-identical to single-GPU) but produced wrong results in the multi-node
      layout, where each node contributes one rank. It is now gated on
      `TpDegree >= 2`, which costs nothing — with one local rank there is no
      concurrency to gain — but the underlying discrepancy is unexplained and
      worth root-causing before the gate is relaxed.
- [ ] Qwen3's QKV fusion requires all three of Q/K/V to share a quant type, so
      mixed-quant GGUFs (`Q4_K_M`) fuse only some layers and then fault. Pre-existing,
      hit while sourcing a test model.

### Multi-node

Verified on the GGML CUDA backend with two processes (one GPU each) over TCP
loopback: the driver/worker lockstep, hierarchical AllReduce, and shutdown all
work, and the generated text matches both the single-GPU and the direct-CUDA
multi-node runs.

```bash
# node 0                                         # node 1
TENSORSHARP_TP_DEVICES=0 ... --tp 1 \            TENSORSHARP_TP_DEVICES=1 ... --tp 1 \
  --tp-node-id 0 --tp-peers "h0:9500,h1:9500"      --tp-node-id 1 --tp-peers "h0:9500,h1:9500"
```

---

## Stage 2 — Network Parallelism with Shared State

**Goal:** Distribute TP across multiple machines connected by a network, with
reconvergent operations and shared response/KV-cache state via a coordination
layer (Redis initially).

### Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Node 0          │         │  Node 1          │
│  ┌─────┐┌─────┐ │  TCP/   │ ┌─────┐┌─────┐  │
│  │GPU 0││GPU 1│ │  RDMA   │ │GPU 2││GPU 3│  │
│  └──┬──┘└──┬──┘ │  network│ └──┬──┘└──┬──┘  │
│     └──┬───┘    │         │    └──┬───┘     │
│  Local AllReduce│         │  Local AllReduce │
│     └──┬───┘    │         │    └──┬───┘     │
│        │        │         │       │         │
│   Rank 0-1      │         │   Rank 2-3      │
└────────┼────────┘         └───────┼─────────┘
         │                          │
         └──────────┬───────────────┘
                    │
         ┌──────────▼──────────┐
         │  Redis / Valkey     │
         │  ┌───────────────┐  │
         │  │ KV Cache Pool │  │
         │  │ Response Queue│  │
         │  │ Rank Barrier  │  │
         │  │ Weight Registry│ │
         │  └───────────────┘  │
         └─────────────────────┘
```

### Components

#### 2.1 Network Communicator (`INetworkCommunicator`)

```csharp
public interface INetworkCommunicator
{
    int Rank { get; }
    int WorldSize { get; }

    // Collective operations (blocking, all ranks must call).
    void AllReduce(Span<float> buffer);
    void AllGather(Span<float> send, Span<float> recv);
    void Barrier();

    // Point-to-point.
    void Send(int destRank, ReadOnlySpan<byte> data);
    void Recv(int srcRank, Span<byte> buffer);
}
```

Initial implementation: TCP sockets with a simple framing protocol.
Each node runs a `NetworkCommunicatorServer` thread that handles
incoming connections from peer ranks.

#### 2.2 Hierarchical AllReduce

For multi-node TP, AllReduce is decomposed into two phases:

1. **Intra-node**: Local P2P AllReduce (Stage 1 code) reduces within each node
2. **Inter-node**: Network AllReduce across node representatives (rank 0 of each node)
3. **Intra-node broadcast**: Result propagated to local peers

This minimizes network traffic: only `1/tp_local` of the data crosses the network.

#### 2.3 Shared KV Cache (Redis)

The KV cache is stored in Redis as binary blobs keyed by
`kv:{session_id}:{layer}:{rank}`. This enables:

- **Session migration**: a request can be served by any node that
  fetches the KV cache from Redis
- **Prefix caching**: shared prefixes are stored once and referenced
  by multiple sessions
- **Crash recovery**: KV state survives node restarts

```
Redis key layout:
  kv:{session}:meta          → JSON { layers, heads, headDim, seqLen, dtype }
  kv:{session}:L{l}:R{r}:K  → binary blob (numKVHeads/tp × seqLen × headDim × dtypeSize)
  kv:{session}:L{l}:R{r}:V  → binary blob
```

Optimization: use Redis `MEMORY` commands and pipelining to batch
layer transfers. For large caches, use Redis Streams or a dedicated
binary protocol (Stage 3).

#### 2.4 Shared Response Queue

Generated tokens are published to a Redis Stream
`response:{session_id}` so that:

- Any API server node can stream tokens to the client regardless of
  which compute node produced them
- Multiple consumers (logging, metrics) can subscribe independently

#### 2.5 Rank Coordination

A Redis-based barrier and rank registry:

```
tp:group:{group_id}:ranks  → SET of "node:rank" members
tp:group:{group_id}:barrier → INCR/DECR counter for sync
tp:group:{group_id}:config → JSON { worldSize, localTp, nodeCount }
```

### Changes Required (Stage 2 — Implemented)

| Component | Change |
|-----------|--------|
| `TensorSharp.Backends.Cuda/ITensorParallelGroup.cs` | **New.** Interface abstracting local and distributed TP groups |
| `TensorSharp.Backends.Cuda/TensorParallelGroup.cs` | Implements `ITensorParallelGroup`; adds `GlobalDegree`, `GlobalRankOffset`, `NodeCount` |
| New project: `TensorSharp.Distributed` | TCP communicator, distributed TP group, config parsing |
| `TensorSharp.Distributed/TcpCommunicator.cs` | **New.** TCP mesh with length-prefixed framing; AllReduce, Broadcast, Barrier |
| `TensorSharp.Distributed/DistributedTensorParallelGroup.cs` | **New.** Hierarchical AllReduce: local P2P → TCP → local broadcast |
| `TensorSharp.Distributed/DistributedTpConfig.cs` | **New.** Peer endpoint parsing, env-var configuration |
| `TensorSharp.Models/ModelBase.cs` | `ITensorParallelGroup` field, `GlobalTpDegree`/`TpRankOffset` properties, multi-node weight sharding |
| `TensorSharp.Cli/Program.cs` | `--tp-node-id`, `--tp-peers` arguments |
| `TensorSharp.Server/ModelLifecycleService.cs` | `TENSORSHARP_TP_NODE_ID`, `TENSORSHARP_TP_PEERS` env-var support |

### Configuration

```bash
# CLI: 2-node tensor parallelism (each node has 2 GPUs)
# Node 0:
TensorSharp.Cli --model qwen3-8b.gguf --backend cuda --tp 2 \
  --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Node 1:
TensorSharp.Cli --model qwen3-8b.gguf --backend cuda --tp 2 \
  --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Server: via environment variables
# Node 0:
TENSORSHARP_TP_DEGREE=2 TENSORSHARP_TP_NODE_ID=0 \
TENSORSHARP_TP_PEERS=192.168.1.10:9500,192.168.1.11:9500 \
  dotnet run --project TensorSharp.Server

# Config JSON (auto-expanded to CLI args)
{ "tp": 2, "tp-node-id": 0, "tp-peers": "192.168.1.10:9500,192.168.1.11:9500", "backend": "cuda" }
```

---

## Stage 3 — RDMA Memory Access

**Goal:** Replace TCP-based inter-node communication with RDMA
(Remote Direct Memory Access) for microsecond-latency collective
operations, if profiling shows network latency is the bottleneck.

### When RDMA Helps

RDMA is beneficial when:
- Inter-node AllReduce latency dominates compute time (small batch decode)
- Network bandwidth is the bottleneck for KV cache transfers
- Tail latency matters (real-time serving with strict SLAs)

RDMA is **not** beneficial when:
- Compute dominates (large-batch prefill)
- Network is already fast enough (100Gbps+ TCP with kernel bypass)
- Hardware doesn't support it (consumer GPUs, no InfiniBand/RoCE NICs)

### Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Node 0          │         │  Node 1          │
│  GPU 0 ←─ GPUDirect RDMA ─→ GPU 2            │
│  GPU 1 ←─ GPUDirect RDMA ─→ GPU 3            │
│                  │         │                  │
│  (NIC registers GPU VRAM   │                  │
│   for zero-copy transfers) │                  │
└──────────────────┘         └──────────────────┘
```

### Components

#### 3.1 RDMA Transport

Two options depending on hardware:

| Transport | Hardware | API | Latency |
|-----------|----------|-----|---------|
| InfiniBand | Mellanox ConnectX + IB switch | libibverbs | ~1-2 µs |
| RoCE v2 | Any RDMA-capable NIC + Ethernet | libibverbs over UDP | ~2-5 µs |
| iWARP | Intel/Chelsio NICs + Ethernet | librdmacm | ~5-10 µs |

On Windows, use the WinRDMA API (`ndis.sys` NDK) or fall back to
`NetworkDirect` (Intel/Chelsio). On Linux, use `libibverbs` directly.

#### 3.2 GPUDirect RDMA

NVIDIA GPUDirect RDMA allows the NIC to read/write GPU VRAM directly,
bypassing the CPU and system memory:

```
GPU VRAM → PCIe → NIC → Network → NIC → PCIe → GPU VRAM
```

Requires:
- NVIDIA GPU with PCIe BAR1 mapping (all datacenter GPUs, some consumer)
- NIC on the same PCIe switch/root complex as the GPU
- `nvidia-peermem` kernel module (Linux) or NDK (Windows)

#### 3.3 NCCL Integration (Linux)

On Linux, the simplest path is NCCL, which handles RDMA transparently:

```csharp
// P/Invoke to libnccl.so
[DllImport("nccl")]
static extern int ncclAllReduce(IntPtr sendbuff, IntPtr recvbuff,
    long count, ncclDataType_t datatype, ncclRedOp_t op,
    IntPtr comm, IntPtr stream);
```

NCCL auto-detects the best transport (NVLink > PCIe P2P > IB/RoCE > TCP)
and handles GPUDirect RDMA setup.

#### 3.4 Custom RDMA (Windows / Fine-grained Control)

For Windows or when NCCL isn't suitable:

1. **Memory registration**: Register GPU buffers with the NIC via
   `ndkRegisterBuffer` (Windows NDK) or `ibvRegMr` (Linux verbs)
2. **Queue pairs**: Create RDMA queue pairs between ranks
3. **RDMA Write/Read**: One-sided operations for AllReduce:
   - Each rank writes its partial to a remote rank's registered buffer
   - Remote rank sums in-place after a completion notification
4. **Completion queue**: Poll for operation completion

### Changes Required

| Component | Change |
|-----------|--------|
| `TensorSharp.Distributed` | Add `RdmaCommunicator` implementing `INetworkCommunicator` |
| `TensorSharp.Backends.Cuda` | Add NCCL P/Invoke bindings (Linux), GPUDirect buffer registration |
| New: `TensorSharp.Rdma` | Low-level RDMA bindings (libibverbs / WinNDK) |
| `TensorSharp.Models/ModelBase.cs` | Transport selection: auto-detect best available |

### Decision Criteria

Before implementing Stage 3, profile Stage 2 with TCP:

| Metric | TCP Sufficient | RDMA Needed |
|--------|---------------|-------------|
| AllReduce latency (per layer) | < 100 µs | > 500 µs |
| KV cache transfer (1K tokens) | < 1 ms | > 5 ms |
| Decode throughput degradation | < 10% vs local TP | > 30% |
| Tail latency (P99) | < 2× median | > 5× median |

If TCP meets the latency targets, Stage 3 adds complexity without
meaningful user-facing improvement.

---

## Implementation Priority

```
Stage 1 (Local TP)          ████████████████████ NEARLY COMPLETE
  ├── Qwen3 reference impl  ████████████████████ DONE
  ├── Mistral3               ████████████████████ DONE
  ├── Gemma3                 ████████████████████ DONE
  ├── Gemma4 (dense+MoE)     ████████████████████ DONE
  ├── Qwen3.5 (SSM+MoE)     ████████████████████ DONE
  ├── GptOss (MoE)           ██████████████████░░ DONE (down-bias gap)
  ├── Nemotron (SSM+MoE)     ████████████████████ DONE (Mamba2 replicated)
  ├── DiffusionGemma         ──────────────────── N/A (diffusion model)
  ├── QwenImage              ──────────────────── N/A (image generation)
  ├── Batched forward TP     ████████████████████ DONE (Qwen3, Mistral3; MoE models fall back to per-seq)
  └── NCCL (Linux)           ░░░░░░░░░░░░░░░░░░░░ TODO

Stage 2 (Network TP)        ██████████░░░░░░░░░░ IN PROGRESS
  ├── TCP communicator       ████████████████████ DONE
  ├── Hierarchical AllReduce ████████████████████ DONE
  ├── ITensorParallelGroup   ████████████████████ DONE
  ├── ModelBase multi-node   ████████████████████ DONE
  ├── CLI --tp-node-id/peers ████████████████████ DONE
  ├── Server env-var config  ████████████████████ DONE
  ├── Model-specific sharding████████████████░░░░ IN PROGRESS
  ├── Redis KV cache         ░░░░░░░░░░░░░░░░░░░░ DEFERRED (direct TCP instead)
  ├── Redis response queue   ░░░░░░░░░░░░░░░░░░░░ DEFERRED (direct TCP instead)
  └── Multi-node server      ████████████████████ DONE

Stage 3 (RDMA)              ░░░░░░░░░░░░░░░░░░░░ CONDITIONAL
  ├── Profile Stage 2 first  ░░░░░░░░░░░░░░░░░░░░
  ├── NCCL integration       ░░░░░░░░░░░░░░░░░░░░
  └── Custom RDMA transport  ░░░░░░░░░░░░░░░░░░░░
```

### Model TP Feasibility

| Model | Architecture | TP Status | Notes |
|-------|-------------|-----------|-------|
| Qwen3 | Dense transformer | ✅ Done | Reference implementation |
| Mistral3 | Dense transformer | ✅ Done | Fused/separate QKV, YaRN RoPE |
| Gemma3 | Dense transformer | ✅ Done | Separate Q/K/V, GELU, sliding window, extra norms |
| Gemma4 | Dense + MoE | ✅ Done | Expert slicing, dual dense+MoE FFN, per-layer head dims, shared KV layers |
| Qwen3.5 | SSM + MoE | ✅ Done | GatedDeltaNet SSM with per-rank V-head ownership + CUDA kernels, MoE expert slicing, shared experts |
| GptOss | MoE | ✅ Done | Expert slicing with biased projections, attention sinks, YaRN; expert down-bias skipped in TP |
| Nemotron | SSM + MoE | ✅ Done | Mamba2 replicated on rank 0, attention (no RoPE), MoE expert slicing |
| DiffusionGemma | Diffusion | ❌ N/A | Not autoregressive text generation |
| QwenImage | Image gen | ❌ N/A | Not autoregressive text generation |
