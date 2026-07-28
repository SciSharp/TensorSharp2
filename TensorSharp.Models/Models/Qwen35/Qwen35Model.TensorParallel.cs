// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ============================================================================
// Qwen35Model.TensorParallel.cs
//
// Tensor-parallel forward pass for the Qwen3.5 hybrid model. Splits:
//   * FullAttention layers: Q/K/V heads + output projection (Megatron pattern)
//   * GatedDeltaNet layers: V heads (block-cyclic on K mapping) + ssm_out
//   * Dense FFN layers: gate_up (column) + down (row)
//   * MoE layers: tensor-parallel experts (1/tp slice of every expert)
//
// The GDN recurrent state (delta state + conv state) is per-rank and never
// communicated — each rank owns the state for its own V heads. The CUDA-native
// GDN kernel is the only supported execution path under TP.
// ============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // ====================================================================
        // Per-rank KV caches for full-attention layers: [layer][rank]
        // ====================================================================
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        // ====================================================================
        // Per-rank GDN recurrent state: [layer][rank]
        // ====================================================================
        private Tensor[][] _tpDeltaState;    // [layer][rank]: [nV/tp, headVDim, headKDim]
        private Tensor[][] _tpConvState;     // [layer][rank]: [convKernel-1, qkvDim/tp]
        private int[] _tpConvWriteIdx;       // [layer] — identical on all ranks

        // ====================================================================
        // Block-cyclic head mapping
        // ====================================================================

        /// <summary>
        /// Compute the block-cyclic V-head assignment for a given rank.
        /// Rank r owns K heads [r*nK/tp, (r+1)*nK/tp) and the V heads that
        /// map to them: { h : (h % nK) ∈ that range }.
        /// Returns the sorted list of global V-head indices owned by this rank.
        /// </summary>
        internal static int[] ComputeBlockCyclicVHeads(int rank, int tp, int numVHeads, int numKHeads)
        {
            int kBlockWidth = numKHeads / tp;
            int kStart = rank * kBlockWidth;
            int kEnd = kStart + kBlockWidth;

            var heads = new List<int>();
            for (int h = 0; h < numVHeads; h++)
            {
                int kHead = h % numKHeads;
                if (kHead >= kStart && kHead < kEnd)
                    heads.Add(h);
            }
            return heads.ToArray();
        }

        /// <summary>
        /// Compute the K-head indices owned by a given rank (contiguous block).
        /// </summary>
        internal static int[] ComputeKHeadsForRank(int rank, int tp, int numKHeads)
        {
            int blockWidth = numKHeads / tp;
            int start = rank * blockWidth;
            var heads = new int[blockWidth];
            for (int i = 0; i < blockWidth; i++)
                heads[i] = start + i;
            return heads;
        }

        /// <summary>
        /// Build the permutation that maps local V-head order (sorted by global
        /// head index) to the block-cyclic order expected by the ssm_out weight.
        /// The ssm_out weight's input dimension is indexed by (v_head, v_dim),
        /// so its column ordering must match the V-head shard.
        /// </summary>
        internal static int[] BuildVHeadPermutation(int rank, int tp, int numVHeads, int numKHeads, int headVDim)
        {
            int[] globalHeads = ComputeBlockCyclicVHeads(rank, tp, numVHeads, numKHeads);
            int localVHeads = globalHeads.Length;
            var perm = new int[localVHeads * headVDim];
            for (int lh = 0; lh < localVHeads; lh++)
            {
                int gh = globalHeads[lh];
                for (int d = 0; d < headVDim; d++)
                    perm[lh * headVDim + d] = gh * headVDim + d;
            }
            return perm;
        }

        // ====================================================================
        // TP constraint validation
        // ====================================================================

        private void ValidateTpConstraints()
        {
            int tp = GlobalTpDegree;
            var errors = new List<string>();

            if (_numKHeads % tp != 0)
                errors.Add($"GDN K heads ({_numKHeads}) not divisible by TP degree ({tp})");
            if (_numVHeads % tp != 0)
                errors.Add($"GDN V heads ({_numVHeads}) not divisible by TP degree ({tp})");
            if (Config.NumHeads % tp != 0)
                errors.Add($"Attention heads ({Config.NumHeads}) not divisible by TP degree ({tp})");
            if (Config.NumKVHeads % tp != 0)
                errors.Add($"Attention KV heads ({Config.NumKVHeads}) not divisible by TP degree ({tp})");
            if (Config.IntermediateSize > 0 && Config.IntermediateSize % tp != 0)
                errors.Add($"Intermediate size ({Config.IntermediateSize}) not divisible by TP degree ({tp})");
            if (_numExperts > 0 && _expertFfnLength % tp != 0)
                errors.Add($"Expert FFN length ({_expertFfnLength}) not divisible by TP degree ({tp})");
            if (_numExperts > 0 && _sharedExpertFfnLength > 0 && _sharedExpertFfnLength % tp != 0)
                errors.Add($"Shared expert FFN length ({_sharedExpertFfnLength}) not divisible by TP degree ({tp})");
            if (_numVHeads % _numKHeads != 0)
                errors.Add($"Model invariant violated: V heads ({_numVHeads}) not divisible by K heads ({_numKHeads})");

            // Verify CUDA-native GDN path is available (the only supported path under TP)
            // Qwen3.5 is the one architecture still pinned to direct CUDA under
            // TP: its per-rank GatedDeltaNet runs as a single fused CUDA kernel
            // (ts_qwen35_gdn_*) that keeps the conv ring buffer and the delta
            // state on-device, and the GGML bridge exposes no packed-GDN
            // equivalent — only the unpacked chunked and batched-step forms. The
            // attention and MoE halves of the TP forward are backend-agnostic, so
            // this lifts once a packed GDN lands in ggml_ops_gated_delta_net.cpp.
            if (_backend != BackendType.Cuda)
                errors.Add($"Qwen3.5/3.6 tensor parallelism requires the direct CUDA backend, got {_backend}. " +
                           "Its per-rank GatedDeltaNet runs as one fused CUDA kernel and the GGML bridge has no " +
                           "packed-GDN equivalent yet. Either run multi-GPU on direct CUDA (--backend cuda --tp N), " +
                           "or run single-GPU on GGML (--backend ggml_cuda, no --tp).");

            if (errors.Count > 0)
                throw new InvalidOperationException(
                    $"Qwen3.5 TP validation failed:\n  " + string.Join("\n  ", errors));

            Console.WriteLine($"  TP constraints validated: tp={tp} (local={TpDegree}), " +
                $"GDN heads V={_numVHeads}/K={_numKHeads}, " +
                $"Attn heads Q={Config.NumHeads}/KV={Config.NumKVHeads}");
        }

        // ====================================================================
        // Weight sharding
        // ====================================================================

        private void ShardQwen35WeightsForTP()
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;

            // --- Full-attention layers: column/row parallel ---
            // attn_output.weight is row-parallel (split input dim).
            // attn_qkv.weight is column-parallel, but the fused weight is a plain
            // [Q(+gate) | K | V] concatenation along the output dim. A contiguous
            // split would hand rank 0 mostly Q rows and no K/V, so it is sharded
            // separately with head-aware regrouping (see ShardFusedQkvForTP).
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: Array.Empty<string>(),
                rowParallelPatterns: new[] { "attn_output.weight" });

            for (int layer = 0; layer < TotalLayerCount; layer++)
            {
                if (_isRecurrent[layer])
                    continue; // recurrent layers pack Q/K into ssm_in_proj instead
                ShardFusedQkvForTP($"blk.{layer}.attn_qkv.weight");
            }

            // --- GDN layers: segmented sharding ---
            // Run BEFORE the dense FFN sharding so the large F32 recurrent
            // input packs (dequantized during fusion for mixed-quant models)
            // are sharded and their full-size sources freed early, reducing
            // peak host memory during the subsequent gate_up sharding.
            ShardGdnWeightsForTP();

            // --- Dense FFN layers: column/row parallel ---
            // ffn_down.weight is row-parallel. ffn_gate_up.weight is a plain
            // [gate | up] concatenation, so it needs the same segment-aware
            // regrouping as QKV: a contiguous split would give rank 0 all of
            // gate and rank 1 all of up.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: Array.Empty<string>(),
                rowParallelPatterns: new[] { "ffn_down.weight" });

            for (int layer = 0; layer < TotalLayerCount; layer++)
            {
                if (layer == 0 || (layer + 1) % 16 == 0 || layer + 1 == TotalLayerCount)
                    Console.WriteLine($"    Gate+Up sharding: layer {layer} ({layer + 1}/{TotalLayerCount})");
                ShardFusedGateUpColumnParallel($"blk.{layer}.ffn_gate_up.weight");
            }

            // --- MoE layers: tensor-parallel experts ---
            if (_numExperts > 0)
                ShardMoeWeightsForTP();

            Console.WriteLine($"  Qwen3.5 TP weight sharding complete ({globalTp} GPUs, {tp} local).");
        }

        /// <summary>
        /// Shard the fused full-attention QKV weight for TP with head-aware
        /// regrouping. The fused weight is a plain output-dim concatenation
        ///   [ Q+gate (2*numHeads*headDim) | K (numKVHeads*headDim) | V (numKVHeads*headDim) ]
        /// so a generic contiguous split would give rank 0 mostly Q rows and no
        /// K/V. See <see cref="ShardConcatenatedColumnParallel"/>.
        /// When the fused weight was never created (mixed quant types prevented
        /// fusion), falls back to <see cref="ShardSeparateColumnParallel"/> which
        /// reads the individual Q/K/V weights directly, avoiding a full F32
        /// intermediate that can OOM on memory-constrained hosts.
        /// </summary>
        private void ShardFusedQkvForTP(string weightName)
        {
            int headDim = Config.HeadDim;
            int qDim = 2 * Config.NumHeads * headDim;
            int kDim = Config.NumKVHeads * headDim;
            int vDim = Config.NumKVHeads * headDim;

            if (_quantWeights.ContainsKey(weightName) || _weights.ContainsKey(weightName))
            {
                ShardConcatenatedColumnParallel(weightName, qDim, kDim, vDim);
            }
            else
            {
                // Derive the layer prefix from the fused name (blk.N.attn_qkv.weight → blk.N.).
                string prefix = weightName.Substring(0, weightName.IndexOf("attn_qkv", StringComparison.Ordinal));
                ShardSeparateColumnParallel(weightName,
                    new[] { prefix + "attn_q.weight", prefix + "attn_k.weight", prefix + "attn_v.weight" },
                    new[] { qDim, kDim, vDim });
            }
        }

        /// <summary>
        /// Shard the GDN-specific weights using block-cyclic head assignment.
        /// The packed ssm_in_proj has layout: [Q | K | V | Z | beta | alpha]
        /// where Q/K are contiguous by K-head and V/Z/beta/alpha are strided by V-head.
        /// </summary>
        private void ShardGdnWeightsForTP()
        {
            int tp = TpDegree;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            int qkvDim = 2 * qkDim + vDim;
            int packedDim = qkvDim + vDim + 2 * _numVHeads; // Q+K+V + Z + beta + alpha

            int recurrentTotal = 0, recurrentDone = 0;
            for (int l = 0; l < Config.NumLayers; l++)
                if (_isRecurrent[l]) recurrentTotal++;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!_isRecurrent[layer])
                    continue;

                recurrentDone++;
                if (recurrentDone == 1 || recurrentDone % 12 == 0 || recurrentDone == recurrentTotal)
                    Console.WriteLine($"    GDN sharding: layer {layer} ({recurrentDone}/{recurrentTotal})");

                string prefix = $"blk.{layer}.";

                // --- ssm_in_proj.weight: segmented column-parallel ---
                ShardPackedSsmInProj(prefix + "ssm_in_proj.weight",
                    qkDim, vDim, qkvDim, packedDim);

                // --- ssm_conv1d.weight: shard dim 0 with Q|K|V segmentation ---
                ShardConv1dWeight(prefix + "ssm_conv1d.weight", qkDim, vDim, qkvDim);

                // --- ssm_dt_bias (ssm_dt.bias): block-cyclic per V head ---
                ShardPerVHeadWeight(prefix + "ssm_dt.bias");

                // --- ssm_a: block-cyclic per V head ---
                ShardPerVHeadWeight(prefix + "ssm_a");

                // --- ssm_norm.weight: replicated (shared across heads) ---
                // Already in _weights, no sharding needed.

                // --- ssm_out.weight: row-parallel with V-head permutation ---
                ShardSsmOutWeight(prefix + "ssm_out.weight");
            }
        }

        /// <summary>
        /// Shard the packed ssm_in_proj weight. Layout:
        /// [Q(nK*dK) | K(nK*dK) | V(nV*dV) | Z(nV*dV) | beta(nV) | alpha(nV)]
        /// Q/K: contiguous split by K-head blocks.
        /// V/Z/beta/alpha: block-cyclic gather by V-head.
        /// </summary>
        private void ShardPackedSsmInProj(string weightName, int qkDim, int vDim, int qkvDim, int packedDim)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;
            int hiddenSize = Config.HiddenSize;
            int kBlockWidth = _numKHeads / globalTp;
            int localKHeads = kBlockWidth;
            int localQkDim = _headKDim * localKHeads;
            int localVHeads = _numVHeads / globalTp;
            int localVDim = _headVDim * localVHeads;
            int localQkvDim = 2 * localQkDim + localVDim;
            int localPackedDim = localQkvDim + localVDim + 2 * localVHeads;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                // Quantized path: extract per-rank shards
                var shards = new QuantizedWeight[tp];
                var type = (GgmlTensorType)qw.GgmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                long typeSize = GgufFile.GetTypeSize(type);
                long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);

                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    int kStart = globalRank * kBlockWidth;

                    // Build the row indices for this rank's shard
                    var rowIndices = BuildSsmInProjRowIndices(globalRank, globalTp, qkDim, vDim, qkvDim, packedDim);
                    long dstNe0 = qw.Ne0; // input dim unchanged
                    long dstNe1 = rowIndices.Length;
                    long dstRowBytes = NativeDequant.RowSize(qw.GgmlType, dstNe0);
                    long totalBytes = dstNe1 * dstRowBytes;

                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        for (long row = 0; row < dstNe1; row++)
                        {
                            long srcRow = rowIndices[row];
                            Buffer.MemoryCopy(
                                src + srcRow * srcRowBytes,
                                dst + row * dstRowBytes,
                                dstRowBytes, dstRowBytes);
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytes,
                        qw.GgmlType, dstNe0, dstNe1);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                // F32 path
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] rowIndices = BuildSsmInProjRowIndices(globalRank, globalTp, qkDim, vDim, qkvDim, packedDim);
                    var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rowIndices.Length, hiddenSize);
                    unsafe
                    {
                        float* srcPtr = GetFloatPtr(w);
                        float* dstPtr = GetFloatPtr(shard);
                        for (int row = 0; row < rowIndices.Length; row++)
                        {
                            Buffer.MemoryCopy(
                                srcPtr + (long)rowIndices[row] * hiddenSize,
                                dstPtr + (long)row * hiddenSize,
                                hiddenSize * 4, hiddenSize * 4);
                        }
                    }
                    shards[r] = shard;
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        /// <summary>
        /// Build the output-row indices for rank r's ssm_in_proj shard.
        /// The packed layout is: [Q | K | V | Z | beta | alpha]
        /// Q rows: kStart*dK .. (kStart+kBlockWidth)*dK - 1
        /// K rows: qkDim + kStart*dK .. qkDim + (kStart+kBlockWidth)*dK - 1
        /// V rows: 2*qkDim + vHead*dV .. 2*qkDim + (vHead+1)*dV - 1 for each owned V head
        /// Z rows: qkvDim + vHead*dV .. qkvDim + (vHead+1)*dV - 1
        /// beta rows: qkvDim + vDim + vHead
        /// alpha rows: qkvDim + vDim + nV + vHead
        /// </summary>
        private int[] BuildSsmInProjRowIndices(int rank, int tp, int qkDim, int vDim, int qkvDim, int packedDim)
        {
            int kBlockWidth = _numKHeads / tp;
            int kStart = rank * kBlockWidth;
            int[] vHeads = ComputeBlockCyclicVHeads(rank, tp, _numVHeads, _numKHeads);

            var indices = new List<int>();

            // Q rows: contiguous block of kBlockWidth K-heads
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(kh * _headKDim + d);

            // K rows: same block, offset by qkDim
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(qkDim + kh * _headKDim + d);

            // V rows: block-cyclic V heads, offset by 2*qkDim
            foreach (int vh in vHeads)
                for (int d = 0; d < _headVDim; d++)
                    indices.Add(2 * qkDim + vh * _headVDim + d);

            // Z rows: same V heads, offset by qkvDim
            foreach (int vh in vHeads)
                for (int d = 0; d < _headVDim; d++)
                    indices.Add(qkvDim + vh * _headVDim + d);

            // beta rows: one per V head
            foreach (int vh in vHeads)
                indices.Add(qkvDim + vDim + vh);

            // alpha rows: one per V head
            foreach (int vh in vHeads)
                indices.Add(qkvDim + vDim + _numVHeads + vh);

            return indices.ToArray();
        }

        /// <summary>
        /// Shard the depthwise conv1d weight [qkvDim, convKernel] along dim 0
        /// using the same Q|K|V segmentation as ssm_in_proj.
        /// </summary>
        private void ShardConv1dWeight(string weightName, int qkDim, int vDim, int qkvDim)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (!_weights.TryGetValue(weightName, out var w))
                return;

            int convKernel = (int)w.Sizes[1];
            var shards = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                int[] rowIndices = BuildConv1dRowIndices(globalRank, globalTp, qkDim, vDim, qkvDim);
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rowIndices.Length, convKernel);
                unsafe
                {
                    float* srcPtr = GetFloatPtr(w);
                    float* dstPtr = GetFloatPtr(shard);
                    for (int row = 0; row < rowIndices.Length; row++)
                    {
                        Buffer.MemoryCopy(
                            srcPtr + (long)rowIndices[row] * convKernel,
                            dstPtr + (long)row * convKernel,
                            convKernel * 4, convKernel * 4);
                    }
                }
                shards[r] = shard;
            }

            _tpWeights[weightName] = shards;
            _weights.Remove(weightName);
            w.Dispose();
        }

        private int[] BuildConv1dRowIndices(int rank, int tp, int qkDim, int vDim, int qkvDim)
        {
            int kBlockWidth = _numKHeads / tp;
            int kStart = rank * kBlockWidth;
            int[] vHeads = ComputeBlockCyclicVHeads(rank, tp, _numVHeads, _numKHeads);

            var indices = new List<int>();

            // Q channels
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(kh * _headKDim + d);

            // K channels
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(qkDim + kh * _headKDim + d);

            // V channels
            foreach (int vh in vHeads)
                for (int d = 0; d < _headVDim; d++)
                    indices.Add(2 * qkDim + vh * _headVDim + d);

            return indices.ToArray();
        }

        /// <summary>
        /// Shard a 1D weight indexed per V-head (dt_bias, a) using block-cyclic assignment.
        /// </summary>
        private void ShardPerVHeadWeight(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (!_weights.TryGetValue(weightName, out var w))
                return;

            var shards = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, vHeads.Length);
                unsafe
                {
                    float* srcPtr = GetFloatPtr(w);
                    float* dstPtr = GetFloatPtr(shard);
                    for (int i = 0; i < vHeads.Length; i++)
                        dstPtr[i] = srcPtr[vHeads[i]];
                }
                shards[r] = shard;
            }

            _tpWeights[weightName] = shards;
            _weights.Remove(weightName);
            w.Dispose();
        }

        /// <summary>
        /// Shard ssm_out.weight [hidden, v_dim] as row-parallel.
        /// The input columns are gathered in block-cyclic V-head order so
        /// each rank's shard aligns with the GDN kernel output (which emits
        /// V heads in the order returned by <see cref="ComputeBlockCyclicVHeads"/>).
        /// A plain contiguous split would pair the wrong V-head weights with
        /// the wrong GDN outputs and corrupt every recurrent layer.
        /// </summary>
        private void ShardSsmOutWeight(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                // Quantized row-parallel: gather block-aligned columns per V head.
                var type = (GgmlTensorType)qw.GgmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                long typeSize = GgufFile.GetTypeSize(type);
                long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                int blocksPerVHead = _headVDim / (int)blockSize;
                long vHeadBytes = (long)blocksPerVHead * typeSize;
                int localVHeads = _numVHeads / globalTp;
                long ne0PerShard = (long)localVHeads * _headVDim;
                long dstRowBytes = (long)localVHeads * blocksPerVHead * typeSize;
                long totalBytesPerShard = qw.Ne1 * dstRowBytes;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        for (long row = 0; row < qw.Ne1; row++)
                        {
                            long dstOffset = 0;
                            for (int vhIdx = 0; vhIdx < vHeads.Length; vhIdx++)
                            {
                                long srcVhOffset = (long)vHeads[vhIdx] * blocksPerVHead * typeSize;
                                Buffer.MemoryCopy(
                                    src + row * srcRowBytes + srcVhOffset,
                                    dst + row * dstRowBytes + dstOffset,
                                    vHeadBytes, vHeadBytes);
                                dstOffset += vHeadBytes;
                            }
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                        qw.GgmlType, ne0PerShard, qw.Ne1);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                // F32 row-parallel: gather columns per V head in block-cyclic order.
                int totalVDim = (int)w.Sizes[1];
                int localVHeads = _numVHeads / globalTp;
                int vDimPerShard = localVHeads * _headVDim;
                int hiddenDim = (int)w.Sizes[0];
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, hiddenDim, vDimPerShard);
                    unsafe
                    {
                        float* srcPtr = GetFloatPtr(w);
                        float* dstPtr = GetFloatPtr(shard);
                        for (int row = 0; row < hiddenDim; row++)
                        {
                            long dstColOffset = 0;
                            for (int vhIdx = 0; vhIdx < vHeads.Length; vhIdx++)
                            {
                                long srcCol = (long)vHeads[vhIdx] * _headVDim;
                                Buffer.MemoryCopy(
                                    srcPtr + (long)row * totalVDim + srcCol,
                                    dstPtr + (long)row * vDimPerShard + dstColOffset,
                                    (long)_headVDim * 4, (long)_headVDim * 4);
                                dstColOffset += _headVDim;
                            }
                        }
                    }
                    shards[r] = shard;
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        /// <summary>
        /// Shard MoE expert weights for tensor-parallel experts.
        /// Each rank holds 1/tp of every expert's FFN width.
        /// Router weights (ffn_gate_inp) are replicated.
        /// </summary>
        private void ShardMoeWeightsForTP()
        {
            int tp = TpDegree;

            for (int layer = 0; layer < TotalLayerCount; layer++)
            {
                if (_isMoeLayer == null || !_isMoeLayer[layer])
                    continue;

                string prefix = $"blk.{layer}.";

                // Router weight: replicated (no sharding needed, stays in _weights)

                // Expert gate/up weights: column-parallel (split expertFfnLength)
                for (int e = 0; e < _numExperts; e++)
                {
                    string gateKey = prefix + $"ffn_gate_exps.{e}.weight";
                    string upKey = prefix + $"ffn_up_exps.{e}.weight";
                    string downKey = prefix + $"ffn_down_exps.{e}.weight";

                    ShardExpertColumnParallel(gateKey);
                    ShardExpertColumnParallel(upKey);
                    ShardExpertRowParallel(downKey);
                }

                // Shared expert weights: same column/row split
                if (_hasSharedExperts != null && _hasSharedExperts[layer])
                {
                    ShardExpertColumnParallel(prefix + "ffn_gate_shexp.weight");
                    ShardExpertColumnParallel(prefix + "ffn_up_shexp.weight");
                    ShardExpertRowParallel(prefix + "ffn_down_shexp.weight");
                    // ffn_gate_inp_shexp: replicated (stays in _weights)
                }
            }
        }

        private void ShardExpertColumnParallel(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                long rowsPerShard = qw.Ne1 / globalTp;
                long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                long bytesPerShard = rowsPerShard * rowBytes;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    IntPtr shardPtr = IntPtr.Add(qw.Data, (int)(globalRank * bytesPerShard));
                    shards[r] = QuantizedWeight.CreateExternalView(
                        shardPtr, bytesPerShard, qw.GgmlType, qw.Ne0, rowsPerShard, qw);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                long shardSize = w.Sizes[0] / globalTp;
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    var view = w.Narrow(0, globalRank * shardSize, shardSize);
                    shards[r] = Ops.NewContiguous(view);
                    view.Dispose();
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        private void ShardExpertRowParallel(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                var type = (GgmlTensorType)qw.GgmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                long typeSize = GgufFile.GetTypeSize(type);
                long blocksPerRow = qw.Ne0 / blockSize;
                long blocksPerShard = blocksPerRow / globalTp;
                long ne0PerShard = blocksPerShard * blockSize;
                // A row-parallel split can only cut on quant-block boundaries. If the
                // blocks per row do not divide evenly the shards would silently drop
                // the remainder (and at tp > blocksPerRow they would be empty),
                // producing a model that loads fine and generates garbage.
                if (blocksPerShard <= 0 || blocksPerRow % globalTp != 0)
                {
                    throw new InvalidOperationException(
                        $"TP degree {globalTp} cannot row-split '{weightName}': {blocksPerRow} " +
                        $"{(GgmlTensorType)qw.GgmlType} blocks per row (block size {blockSize}, ne0 {qw.Ne0}) " +
                        $"is not divisible by {globalTp}. Use a TP degree that divides the block count.");
                }
                long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                long dstRowBytes = (ne0PerShard / blockSize) * typeSize;
                long totalBytesPerShard = qw.Ne1 * dstRowBytes;
                long blockBytesPerShard = blocksPerShard * typeSize;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        long srcBlockOffset = globalRank * blocksPerShard * typeSize;
                        for (long row = 0; row < qw.Ne1; row++)
                        {
                            Buffer.MemoryCopy(
                                src + row * srcRowBytes + srcBlockOffset,
                                dst + row * dstRowBytes,
                                dstRowBytes, blockBytesPerShard);
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                        qw.GgmlType, ne0PerShard, qw.Ne1);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                long shardSize = w.Sizes[1] / globalTp;
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    var view = w.Narrow(1, globalRank * shardSize, shardSize);
                    shards[r] = Ops.NewContiguous(view);
                    view.Dispose();
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        // ====================================================================
        // TP cache initialization
        // ====================================================================

        private void InitTpCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _initialKvCacheCapacity = initialSeqLen;
            int tp = TpDegree;

            // --- Full-attention KV caches ---
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();

            _tpKvCacheCapacity = initialSeqLen;
            _tpKvCacheK = new Tensor[TotalLayerCount][];
            _tpKvCacheV = new Tensor[TotalLayerCount][];

            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (_isRecurrent[l])
                {
                    _tpKvCacheK[l] = null;
                    _tpKvCacheV[l] = null;
                    continue;
                }

                _tpKvCacheK[l] = new Tensor[tp];
                _tpKvCacheV[l] = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, headDim);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, headDim);
                    InitializeCacheTensor(_tpKvCacheK[l][r]);
                    InitializeCacheTensor(_tpKvCacheV[l][r]);
                }
            }

            // --- GDN recurrent state (sharded by the GLOBAL degree) ---
            int convDim = _convKernel - 1;
            int localVHeads = _numVHeads / GlobalTpDegree;
            int localKHeads = _numKHeads / GlobalTpDegree;
            int localQkvDim = 2 * (_headKDim * localKHeads) + (_headVDim * localVHeads);

            _tpDeltaState = new Tensor[Config.NumLayers][];
            _tpConvState = new Tensor[Config.NumLayers][];
            _tpConvWriteIdx = new int[Config.NumLayers];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    _tpDeltaState[l] = null;
                    _tpConvState[l] = null;
                    continue;
                }

                _tpDeltaState[l] = new Tensor[tp];
                _tpConvState[l] = new Tensor[tp];
                _tpConvWriteIdx[l] = 0;

                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpDeltaState[l][r] = new Tensor(alloc, DType.Float32, localVHeads, _headVDim, _headKDim);
                    Ops.Fill(_tpDeltaState[l][r], 0);

                    if (convDim > 0)
                    {
                        _tpConvState[l][r] = new Tensor(alloc, DType.Float32, convDim, localQkvDim);
                        Ops.Fill(_tpConvState[l][r], 0);
                    }
                }
            }

            Console.WriteLine($"  TP caches initialized: {tp} GPUs, " +
                $"KV heads/GPU={numKVHeadsPerGpu}, GDN V heads/GPU={localVHeads}");
        }

        private void EnsureTpCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _tpKvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_tpKvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            int tp = TpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();

            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (_isRecurrent[l] || _tpKvCacheK[l] == null)
                    continue;

                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    var newK = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, headDim);
                    var newV = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, headDim);
                    InitializeCacheTensor(newK);
                    InitializeCacheTensor(newV);

                    if (_cacheSeqLen > 0)
                    {
                        using var srcK = _tpKvCacheK[l][r].Narrow(1, 0, _cacheSeqLen);
                        using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
                        Ops.Copy(dstK, srcK);

                        using var srcV = _tpKvCacheV[l][r].Narrow(1, 0, _cacheSeqLen);
                        using var dstV = newV.Narrow(1, 0, _cacheSeqLen);
                        Ops.Copy(dstV, srcV);
                    }

                    _tpKvCacheK[l][r].Dispose();
                    _tpKvCacheV[l][r].Dispose();
                    _tpKvCacheK[l][r] = newK;
                    _tpKvCacheV[l][r] = newV;
                }
            }

            _tpKvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded Qwen3.5 TP attention cache to {newCapacity} tokens ({tp} GPUs).");
        }

        // ====================================================================
        // TP forward pass
        // ====================================================================

        private float[] ForwardTP(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int tp = TpDegree;
            EnsureTpCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden0 = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            // Inject any queued vision embeddings on rank 0 before broadcasting
            // (mirrors the non-TP ForwardCore). The matching MRoPE positions are
            // staged via SetMRoPEPositions and consumed by the attention block.
            if (_visionEmbeddingsList.Count > 0)
                InjectVisionEmbeddings(hidden0, seqLen);

            // Broadcast embedding to all GPUs.
            Tensor[] hidden = BroadcastTensorToAllRanks(hidden0);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    hidden = RecurrentBlockTP(hidden, layer, seqLen, startPos);
                else
                    hidden = AttentionBlockTP(hidden, layer, seqLen, startPos);
            }

            // Final norm + LM head on GPU 0 only (hidden is replicated after AllReduce).
            Tensor normed = RMSNormOp(hidden[0], "output_norm.weight");
            for (int r = 0; r < tp; r++)
                hidden[r].Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            // Drop the MRoPE positions staged for this (multimodal) forward so the
            // next call defaults to scalar positions, matching the non-TP path.
            _pendingMRoPEPositions = null;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // ====================================================================
        // Full-attention block under TP
        // ====================================================================

        private Tensor[] AttentionBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            string prefix = $"blk.{layer}.";

            // 1. Attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, _attnNormKey[layer]);

            // 2. Column-parallel QKV projection.
            // Qwen3.5 QKV output: [Q+gate (2*numHeads*headDim) | K (numKVHeads*headDim) | V (numKVHeads*headDim)]
            Tensor[] qkvFused = TpColumnParallelLinear(normed[0], _attnQkvKey[layer]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU attention.
            Tensor[] attnOut = FullAttentionTP(qkvFused, layer, seqLen, startPos);

            // 4. Row-parallel output projection + AllReduce. AllReduce leaves the
            // sum on every rank, so the residual can be added in place without a
            // second broadcast from rank 0.
            Tensor[] attnReduced = TpRowParallelLinearAllRanks(attnOut, _attnOutputKey[layer]);
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            // 5. Residual add (hidden stays replicated: identical inputs + identical addend).
            TpResidualAdd(hidden, attnReduced);
            for (int r = 0; r < tp; r++)
                attnReduced[r].Dispose();

            // 6. FFN (dense or MoE).
            hidden = FFNBlockTP(hidden, layer, seqLen);

            return hidden;
        }

        private Tensor[] FullAttentionTP(Tensor[] qkvFused, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            // Qwen3.5: Q output includes gate (2x), so Q dim per GPU = 2 * numHeadsPerGpu * headDim
            int qFullDimPerGpu = 2 * numHeadsPerGpu * headDim;
            int kDimPerGpu = numKVHeadsPerGpu * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                // Split Q+gate, K, V from the fused QKV output.
                Tensor qFull, kTensor, vTensor;
                if (seqLen == 1)
                {
                    qFull = qkvFused[r].Narrow(1, 0, qFullDimPerGpu);
                    kTensor = qkvFused[r].Narrow(1, qFullDimPerGpu, kDimPerGpu);
                    vTensor = qkvFused[r].Narrow(1, qFullDimPerGpu + kDimPerGpu, kDimPerGpu);
                    qkvFused[r].Dispose();
                }
                else
                {
                    using (var qView = qkvFused[r].Narrow(1, 0, qFullDimPerGpu))
                        qFull = Ops.NewContiguous(qView);
                    using (var kView = qkvFused[r].Narrow(1, qFullDimPerGpu, kDimPerGpu))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused[r].Narrow(1, qFullDimPerGpu + kDimPerGpu, kDimPerGpu))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused[r].Dispose();
                }

                // Deinterleave Q and gate: Q is [numHeadsPerGpu, headDim], gate is [numHeadsPerGpu, headDim]
                // interleaved per head: [Q0, gate0, Q1, gate1, ...]
                int qDimPerGpu = numHeadsPerGpu * headDim;
                Tensor qTensor, gateTensor;
                DeinterleaveQGate(qFull, out qTensor, out gateTensor, numHeadsPerGpu, headDim, seqLen, alloc);
                qFull.Dispose();

                // QK norm (per-GPU). The RMSNorm weight is replicated on GPU 0;
                // use a rank-local copy so the kernel doesn't read it cross-GPU.
                qTensor = ApplyQKNormCached(qTensor, ReplicaOnRank(_attnQNormW[layer], r), numHeadsPerGpu, seqLen);
                kTensor = ApplyQKNormCached(kTensor, ReplicaOnRank(_attnKNormW[layer], r), numKVHeadsPerGpu, seqLen);

                // RoPE: per-axis MRoPE when multimodal positions are staged for
                // this forward, otherwise the scalar position RoPE. MRoPE positions
                // are per-token, so they apply identically to this rank's head slice.
                bool useMRoPE = _pendingMRoPEPositions != null && _pendingMRoPEPositions.Length >= 3 * seqLen;
                if (useMRoPE)
                {
                    qTensor = ApplyMRoPEPrefill(qTensor, numHeadsPerGpu, seqLen, _pendingMRoPEPositions);
                    kTensor = ApplyMRoPEPrefill(kTensor, numKVHeadsPerGpu, seqLen, _pendingMRoPEPositions);
                }
                else
                {
                    qTensor = ApplyRoPEPrefill(qTensor, numHeadsPerGpu, seqLen, startPos);
                    kTensor = ApplyRoPEPrefill(kTensor, numKVHeadsPerGpu, seqLen, startPos);
                }

                if (seqLen == 1)
                {
                    // Decode: copy K/V to per-GPU cache, run attention.
                    CopyToCacheDecode(_tpKvCacheK[layer][r], kTensor, _tpKvCacheV[layer][r], vTensor,
                        numKVHeadsPerGpu, headDim, startPos);
                    kTensor.Dispose();
                    vTensor.Dispose();

                    var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * headDim);
                    AttentionDecodePureCS(qTensor, _tpKvCacheK[layer][r], _tpKvCacheV[layer][r],
                        attnResult, numHeadsPerGpu, numKVHeadsPerGpu, headDim, totalSeqLen, scale);
                    qTensor.Dispose();

                    // Apply sigmoid gate: output = attn * sigmoid(gate)
                    Ops.SigmoidMul(attnResult, attnResult, gateTensor);
                    gateTensor.Dispose();

                    results[r] = attnResult;
                }
                else
                {
                    // Prefill path.
                    Tensor qHeads = ReshapeToHeads(qTensor, numHeadsPerGpu, seqLen, headDim);
                    qTensor.Dispose();
                    Tensor kHeads = ReshapeToHeads(kTensor, numKVHeadsPerGpu, seqLen, headDim);
                    kTensor.Dispose();
                    Tensor vHeads = ReshapeToHeads(vTensor, numKVHeadsPerGpu, seqLen, headDim);
                    vTensor.Dispose();

                    CopyToCache(_tpKvCacheK[layer][r], kHeads, startPos, seqLen);
                    CopyToCache(_tpKvCacheV[layer][r], vHeads, startPos, seqLen);
                    kHeads.Dispose();
                    vHeads.Dispose();

                    int groupSize = numHeadsPerGpu / numKVHeadsPerGpu;
                    Tensor kExpanded = ExpandKVHeads(_tpKvCacheK[layer][r], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_tpKvCacheV[layer][r], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, totalSeqLen);
                    Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                    Ops.Softmax(scores, scores);

                    var attnOut = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, headDim);
                    Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    Tensor flatOutput = ReshapeFromHeads(attnOut, numHeadsPerGpu, seqLen, headDim);
                    attnOut.Dispose();

                    // Apply sigmoid gate.
                    Ops.SigmoidMul(flatOutput, flatOutput, gateTensor);
                    gateTensor.Dispose();

                    results[r] = flatOutput;
                }
            }

            return results;
        }

        /// <summary>
        /// Deinterleave Q and gate from the fused Q+gate tensor.
        /// Input layout: [Q0, gate0, Q1, gate1, ...] per head.
        /// Output: separate Q [seqLen, numHeads*headDim] and gate [seqLen, numHeads*headDim].
        /// </summary>
        private void DeinterleaveQGate(Tensor qFull, out Tensor q, out Tensor gate,
            int numHeads, int headDim, int seqLen, IAllocator alloc)
        {
            int totalDim = numHeads * headDim;
            q = new Tensor(alloc, DType.Float32, seqLen, totalDim);
            gate = new Tensor(alloc, DType.Float32, seqLen, totalDim);

            unsafe
            {
                float* srcPtr = GetFloatPtr(qFull);
                float* qPtr = GetFloatPtr(q);
                float* gPtr = GetFloatPtr(gate);

                for (int s = 0; s < seqLen; s++)
                {
                    float* srcRow = srcPtr + s * 2 * totalDim;
                    float* qRow = qPtr + s * totalDim;
                    float* gRow = gPtr + s * totalDim;

                    for (int h = 0; h < numHeads; h++)
                    {
                        int srcBase = h * 2 * headDim;
                        int dstBase = h * headDim;
                        Buffer.MemoryCopy(srcRow + srcBase, qRow + dstBase, headDim * 4, headDim * 4);
                        Buffer.MemoryCopy(srcRow + srcBase + headDim, gRow + dstBase, headDim * 4, headDim * 4);
                    }
                }
            }
        }

        // ====================================================================
        // GDN recurrent block under TP
        // ====================================================================

        private Tensor[] RecurrentBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            string prefix = $"blk.{layer}.";

            // 1. Input norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, _attnNormKey[layer]);

            // 2. Column-parallel packed input projection (segmented).
            Tensor[] packedInput = TpColumnParallelLinear(normed[0], _ssmInProjKey[layer]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-rank GDN: conv1d → L2norm → delta-rule scan → gated RMSNorm.
            Tensor[] gatedOut = GatedDeltaNetTP(packedInput, layer, seqLen);
            for (int r = 0; r < tp; r++)
                packedInput[r].Dispose();

            // 4. Row-parallel ssm_out + AllReduce (result already on every rank).
            Tensor[] gdnReduced = TpRowParallelLinearAllRanks(gatedOut, _ssmOutKey[layer]);
            for (int r = 0; r < tp; r++)
                gatedOut[r].Dispose();

            // 5. Residual add.
            TpResidualAdd(hidden, gdnReduced);
            for (int r = 0; r < tp; r++)
                gdnReduced[r].Dispose();

            // 6. FFN (dense or MoE).
            hidden = FFNBlockTP(hidden, layer, seqLen);

            return hidden;
        }

        /// <summary>
        /// Run the GDN mixer on each rank's shard. Uses the CUDA-native kernel
        /// with per-rank dimensions (localVHeads, localKHeads).
        /// </summary>
        private Tensor[] GatedDeltaNetTP(Tensor[] packedInput, int layer, int seqLen)
        {
            int tp = TpDegree;
            int localKHeads = _numKHeads / GlobalTpDegree;
            int localVHeads = _numVHeads / GlobalTpDegree;
            int localQkDim = _headKDim * localKHeads;
            int localVDim = _headVDim * localVHeads;
            int localQkvDim = 2 * localQkDim + localVDim;
            int localPackedDim = localQkvDim + localVDim + 2 * localVHeads;

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);
                Tensor gated = new Tensor(alloc, DType.Float32, seqLen, localVDim);

                // The packed input is already in the correct per-rank layout
                // (Q|K|V|Z|beta|alpha for this rank's heads).
                // Run the CUDA-native GDN kernel with local dimensions.
                bool ok = CudaFusedOps.TryQwen35GatedDeltaNetPacked(
                    gated,
                    packedInput[r],
                    _tpConvState[layer][r],
                    _tpDeltaState[layer][r],
                    GetTpShardTensor(_ssmConv1dKey[layer], r),
                    GetTpShardTensor(_ssmDtBiasKey[layer], r),
                    GetTpShardTensor(_ssmAKey[layer], r),
                    ReplicaOnRank(_ssmNormW[layer], r),  // replicated, but resident on rank r's GPU
                    seqLen,
                    localPackedDim,
                    localQkvDim,
                    localQkDim,
                    localVDim,
                    localKHeads,
                    localVHeads,
                    _headKDim,
                    _headVDim,
                    _convKernel,
                    _tpConvWriteIdx[layer],
                    Config.Eps);

                if (!ok)
                    throw new InvalidOperationException(
                        $"CUDA-native GDN kernel failed under TP (layer {layer}, rank {r}). " +
                        "TP requires the CUDA-native GDN path (TS_CUDA_QWEN35_GDN_NATIVE must not be 0).");

                results[r] = gated;
            }

            // Advance conv write index (identical on all ranks).
            int convDim = _convKernel - 1;
            if (convDim > 0)
                _tpConvWriteIdx[layer] = (_tpConvWriteIdx[layer] + seqLen) % convDim;

            return results;
        }

        /// <summary>
        /// Get a TP-sharded F32 tensor for a given weight name and rank.
        /// </summary>
        private Tensor GetTpShardTensor(string weightName, int rank)
        {
            if (_tpWeights.TryGetValue(weightName, out var shards))
                return shards[rank];
            // Fall back to a replicated weight (e.g. ssm_norm). It lives on rank 0,
            // so hand back a copy resident on THIS rank's GPU — the caller feeds it
            // straight to a rank-r kernel.
            if (_weights.TryGetValue(weightName, out var w))
                return ReplicaOnRank(w, rank);
            throw new KeyNotFoundException($"TP weight '{weightName}' not found.");
        }

        /// <summary>
        /// A replicated weight resident on rank r's GPU, so a per-rank kernel
        /// that reads it directly never crosses GPUs. Rank 0 aliases the
        /// original; other ranks get a lazily-cached copy. Cheap for the small
        /// norm weights this is used for.
        /// </summary>
        private Tensor ReplicaOnRank(Tensor weight, int rank) => TpReplicatedWeight(weight, rank);

        // ====================================================================
        // FFN block under TP (dense or MoE)
        // ====================================================================

        private Tensor[] FFNBlockTP(Tensor[] hidden, int layer, int seqLen)
        {
            bool isMoe = _isMoeLayer != null && _isMoeLayer[layer];

            if (isMoe)
                return MoEBlockTP(hidden, layer, seqLen);

            // Dense FFN: column-parallel gate_up → SiLU·mul → row-parallel down + AllReduce.
            int tp = TpDegree;

            // 1. Post-attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, _postAttnNormKey[layer]);

            // 2. Column-parallel gate/up.
            Tensor[] gateUp = TpColumnParallelLinear(normed[0], _ffnGateUpKey[layer]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU SiLU·mul.
            int halfDim = (int)(gateUp[0].Sizes[1] / 2);
            Tensor[] gateResults = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                Tensor gate, up;
                if (seqLen == 1)
                {
                    gate = gateUp[r].Narrow(1, 0, halfDim);
                    up = gateUp[r].Narrow(1, halfDim, halfDim);
                }
                else
                {
                    using var gView = gateUp[r].Narrow(1, 0, halfDim);
                    gate = Ops.NewContiguous(gView);
                    using var uView = gateUp[r].Narrow(1, halfDim, halfDim);
                    up = Ops.NewContiguous(uView);
                }
                gateUp[r].Dispose();

                Ops.SiLUMul(gate, gate, up);
                up.Dispose();
                gateResults[r] = gate;
            }

            // 4. Row-parallel down + AllReduce (result already on every rank).
            Tensor[] ffnReduced = TpRowParallelLinearAllRanks(gateResults, _ffnDownKey[layer]);
            for (int r = 0; r < tp; r++)
                gateResults[r].Dispose();

            // 5. Residual add.
            TpResidualAdd(hidden, ffnReduced);
            for (int r = 0; r < tp; r++)
                ffnReduced[r].Dispose();

            return hidden;
        }

        // ====================================================================
        // MoE block under TP (tensor-parallel experts)
        // ====================================================================

        private Tensor[] MoEBlockTP(Tensor[] hidden, int layer, int seqLen)
        {
            int tp = TpDegree;
            int hiddenSize = Config.HiddenSize;
            string prefix = $"blk.{layer}.";

            // 1. Post-attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, _postAttnNormKey[layer]);

            // 2. Router. The router weight is NOT sharded — it lives on rank 0 —
            // so the logits are computed ONCE there and replicated. Running
            // LinearForward per rank would mix a rank-r input with the rank-0
            // weight and a rank-0 output in a single kernel launch: a cross-device
            // dereference that faults the context (CUDA 719) or silently returns
            // garbage. Computing once is also the only way to guarantee every rank
            // routes each token to the same experts.
            var results = new Tensor[tp];

            Tensor rank0Logits = LinearForward(normed[0], _ffnGateInpKey[layer]);
            Tensor[] routerLogitsPerRank = BroadcastTensorToAllRanks(rank0Logits);
            rank0Logits.Dispose();

            // 2a. Fully on-device fused MoE per rank (device-resident expert shards,
            // no host round-trip). Each rank returns its partial contribution.
            bool allOnDevice = true;
            for (int r = 0; r < tp; r++)
            {
                Tensor devOut = TryMoEForwardTpOnDevice(normed[r], routerLogitsPerRank[r], layer, r, seqLen);
                if (devOut == null)
                {
                    allOnDevice = false;
                    break;
                }
                // TryMoEForwardTpOnDevice consumed (disposed) routerLogitsPerRank[r].
                routerLogitsPerRank[r] = null;
                results[r] = devOut;
            }

            if (allOnDevice)
            {
                for (int r = 0; r < tp; r++)
                    normed[r].Dispose();
                return FinishMoEBlockTP(hidden, results, tp);
            }

            // Fused path unavailable — undo the partial attempt and fall back to the
            // host loop below.
            for (int r = 0; r < tp; r++)
            {
                results[r]?.Dispose();
                results[r] = null;
            }

            // Host fallback: the routing decision is derived once from the rank-0
            // logits and shared by every rank, so all ranks stay in lockstep.
            // If the fused path already consumed rank 0's copy before bailing out on a
            // later rank, recompute it here — and own it, so it is disposed below
            // rather than leaked.
            Tensor routerSource = routerLogitsPerRank[0];
            if (routerSource == null)
            {
                routerSource = BroadcastRouterLogitsForFallback(normed[0], layer);
                routerLogitsPerRank[0] = routerSource;
            }
            Tensor routerData = _normTopKProb ? routerSource : Ops.Softmax(null, routerSource);
            float[] routePtr = TensorToFloatArray(routerData);
            if (!ReferenceEquals(routerData, routerSource)) routerData.Dispose();
            for (int r = 0; r < tp; r++)
                routerLogitsPerRank[r]?.Dispose();

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);
                var localInput = normed[r];

                // Accumulate expert outputs.
                var output = new Tensor(alloc, DType.Float32, seqLen, hiddenSize);
                Ops.Fill(output, 0f);

                // Per-token routing: each token selects its own top-K experts.
                for (int t = 0; t < seqLen; t++)
                {
                    // Extract this token's router logits (numExperts elements).
                    float[] tokenLogits = new float[_numExperts];
                    Array.Copy(routePtr, t * _numExperts, tokenLogits, 0, _numExperts);

                    var (topExperts, routeWeights) = SelectTopKExperts(tokenLogits, _numExpertsUsed);

                    // Extract this token's input row [1, hiddenSize].
                    using var tokenInput = localInput.Narrow(0, t, 1);

                    for (int k = 0; k < _numExpertsUsed; k++)
                    {
                        int expertIdx = topExperts[k];
                        float weight = routeWeights[k];

                        string gateKey = prefix + $"ffn_gate_exps.{expertIdx}.weight";
                        string upKey = prefix + $"ffn_up_exps.{expertIdx}.weight";
                        string downKey = prefix + $"ffn_down_exps.{expertIdx}.weight";

                        // Column-parallel gate/up (per-rank shard).
                        Tensor gateOut = TpExpertLinear(tokenInput, gateKey, r, 1);
                        Tensor upOut = TpExpertLinear(tokenInput, upKey, r, 1);

                        // SiLU·mul.
                        Ops.SiLUMul(gateOut, gateOut, upOut);
                        upOut.Dispose();

                        // Row-parallel down (per-rank shard, partial result).
                        Tensor downOut = TpExpertLinear(gateOut, downKey, r, 1);
                        gateOut.Dispose();

                        // Weighted accumulate into the token's output row.
                        Ops.Mul(downOut, downOut, weight);
                        using var outputRow = output.Narrow(0, t, 1);
                        Ops.Add(outputRow, outputRow, downOut);
                        downOut.Dispose();
                    }
                }

                // Shared experts (if present) — apply to ALL tokens.
                if (_hasSharedExperts != null && _hasSharedExperts[layer])
                {
                    Tensor sharedGate = TpExpertLinear(localInput, prefix + "ffn_gate_shexp.weight", r, seqLen);
                    Tensor sharedUp = TpExpertLinear(localInput, prefix + "ffn_up_shexp.weight", r, seqLen);
                    Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                    sharedUp.Dispose();
                    Tensor sharedDown = TpExpertLinear(sharedGate, prefix + "ffn_down_shexp.weight", r, seqLen);
                    sharedGate.Dispose();

                    // Shared expert gate (sigmoid scalar). The gate vector is
                    // replicated on rank 0, so the dot product is evaluated against
                    // the rank-0 input — identical on every rank — and the resulting
                    // scalar scales this rank's partial. Reading it with a rank-r
                    // input would pair a rank-r pointer with a rank-0 pointer.
                    if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                    {
                        var gateVec = _ffnGateInpShexpVec?[layer];
                        if (gateVec != null)
                        {
                            float gateVal = ComputeSharedExpertGate(normed[0], gateVec);
                            Ops.Mul(sharedDown, sharedDown, gateVal);
                        }
                        Ops.Add(output, output, sharedDown);
                    }
                    else
                    {
                        Ops.Add(output, output, sharedDown);
                    }
                    sharedDown.Dispose();
                }

                results[r] = output;
            }

            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            return FinishMoEBlockTP(hidden, results, tp);
        }

        /// <summary>
        /// Shared tail of the TP MoE block: AllReduce the per-rank partial expert
        /// outputs, add the residual and re-replicate the hidden state for the next
        /// layer.
        /// </summary>
        private Tensor[] FinishMoEBlockTP(Tensor[] hidden, Tensor[] results, int tp)
        {
            // AllReduce across ranks (sum partial expert results). This leaves the
            // identical sum on every rank.
            _tpGroup.AllReduce(results);

            // Residual add. `hidden` entered replicated and every rank adds the same
            // reduced value, so it stays replicated — the block used to follow this
            // with a full BroadcastTensorToAllRanks(hidden[0]), re-sending a value
            // each rank already held.
            TpResidualAdd(hidden, results);
            for (int r = 0; r < tp; r++)
                results[r].Dispose();

            return hidden;
        }

        /// <summary>
        /// Router logits on rank 0 for the host fallback, used only when the fused
        /// device path consumed the pre-broadcast copies before bailing out.
        /// </summary>
        private Tensor BroadcastRouterLogitsForFallback(Tensor normed0, int layer)
            => LinearForward(normed0, _ffnGateInpKey[layer]);

        /// <summary>
        /// Linear forward using a TP-sharded expert weight for a specific rank.
        /// </summary>
        private Tensor TpExpertLinear(Tensor input, string weightName, int rank, int seqLen)
        {
            var alloc = _tpGroup.GetAllocator(rank);

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                var qw = qShards[rank];
                int outDim = (int)qw.Ne1;
                var result = new Tensor(alloc, DType.Float32, seqLen, outDim);
                var localInput = ReplicateTensorToRank(input, rank);
                AddmmQuantManaged(result, localInput, qw);
                if (!ReferenceEquals(localInput, input)) localInput.Dispose();
                return result;
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                var w = wShards[rank];
                int outDim = (int)w.Sizes[0];
                var result = new Tensor(alloc, DType.Float32, seqLen, outDim);
                using var wT = w.Transpose();
                var localInput = ReplicateTensorToRank(input, rank);
                Ops.Addmm(result, 0, result, 1.0f, localInput, wT);
                if (!ReferenceEquals(localInput, input)) localInput.Dispose();
                return result;
            }

            throw new KeyNotFoundException($"TP expert weight '{weightName}' not found.");
        }

        private (int[] experts, float[] weights) SelectTopKExperts(float[] routerLogits, int topK)
        {
            int numExperts = routerLogits.Length;
            var indices = new int[numExperts];
            for (int i = 0; i < numExperts; i++) indices[i] = i;
            Array.Sort(indices, (a, b) => routerLogits[b].CompareTo(routerLogits[a]));

            var topExperts = new int[topK];
            var topWeights = new float[topK];
            for (int k = 0; k < topK; k++)
                topExperts[k] = indices[k];

            if (_normTopKProb)
            {
                // routerLogits are RAW logits: softmax over the selected top-K
                // experts (matches the non-TP SelectTopKRouteWeights). The previous
                // raw-logit / raw-sum renormalization produced wrong (even negative)
                // expert weights and corrupted every MoE layer.
                float maxLogit = float.NegativeInfinity;
                for (int k = 0; k < topK; k++)
                {
                    float v = routerLogits[topExperts[k]];
                    if (v > maxLogit) maxLogit = v;
                }
                float sum = 0f;
                for (int k = 0; k < topK; k++)
                {
                    float w = MathF.Exp(routerLogits[topExperts[k]] - maxLogit);
                    topWeights[k] = w;
                    sum += w;
                }
                if (sum > 0f)
                {
                    float inv = 1.0f / sum;
                    for (int k = 0; k < topK; k++)
                        topWeights[k] *= inv;
                }
            }
            else
            {
                // routerLogits are already full-softmax probabilities (the caller
                // pre-softmaxed): use the selected probabilities directly, with no
                // further renormalization.
                for (int k = 0; k < topK; k++)
                    topWeights[k] = routerLogits[topExperts[k]];
            }

            return (topExperts, topWeights);
        }

        private float ComputeSharedExpertGate(Tensor input, Tensor gateVec)
        {
            // dot(input, gateVec) → sigmoid
            unsafe
            {
                float* inputPtr = GetFloatPtr(input);
                float* gatePtr = GetFloatPtr(gateVec);
                int dim = (int)gateVec.ElementCount();
                float dot = 0;
                for (int i = 0; i < dim; i++)
                    dot += inputPtr[i] * gatePtr[i];
                return 1.0f / (1.0f + MathF.Exp(-dot));
            }
        }

        // ====================================================================
        // TP-aware ResetKVCache
        // ====================================================================

        private void ResetTpKVCache()
        {
            int tp = TpDegree;

            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (!_isRecurrent[l] && _tpKvCacheK[l] != null)
                {
                    for (int r = 0; r < tp; r++)
                    {
                        ResetCacheTensor(_tpKvCacheK[l][r]);
                        ResetCacheTensor(_tpKvCacheV[l][r]);
                    }
                }
                else if (_isRecurrent[l] && _tpDeltaState[l] != null)
                {
                    for (int r = 0; r < tp; r++)
                    {
                        Ops.Fill(_tpDeltaState[l][r], 0);
                        if (_tpConvState[l][r] != null)
                            Ops.Fill(_tpConvState[l][r], 0);
                    }
                    _tpConvWriteIdx[l] = 0;
                }
            }
        }

        // ====================================================================
        // TP-aware Dispose
        // ====================================================================

        private void DisposeTpState()
        {
            if (_tpKvCacheK != null)
            {
                for (int l = 0; l < _tpKvCacheK.Length; l++)
                {
                    if (_tpKvCacheK[l] == null) continue;
                    for (int r = 0; r < _tpKvCacheK[l].Length; r++)
                    {
                        _tpKvCacheK[l][r]?.Dispose();
                        _tpKvCacheV[l][r]?.Dispose();
                    }
                }
            }

            if (_tpDeltaState != null)
            {
                for (int l = 0; l < _tpDeltaState.Length; l++)
                {
                    if (_tpDeltaState[l] == null) continue;
                    for (int r = 0; r < _tpDeltaState[l].Length; r++)
                    {
                        _tpDeltaState[l][r]?.Dispose();
                        _tpConvState[l]?[r]?.Dispose();
                    }
                }
            }

            DisposeTpWeightReplicaCache();
            FreeQwenTpMoETables();
        }
    }
}
