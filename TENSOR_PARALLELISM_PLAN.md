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

- CUDA backend only (GGML, MLX backends are single-device by design)
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
