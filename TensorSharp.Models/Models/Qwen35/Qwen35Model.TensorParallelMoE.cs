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
// Qwen35Model.TensorParallelMoE.cs
//
// On-device (direct-CUDA) MoE for the tensor-parallel Qwen3.5/3.6 path.
//
// The original TP MoE block ran a host-orchestrated triple loop
// (rank x token x top-k expert), issuing three separate quantized matmuls per
// (token, expert, rank) against expert shards that were deliberately kept in
// HOST memory. On Qwen3.5-35B-A3B that is ~15.6 GB of weights never reaching
// VRAM and ~61k CPU dequant matmuls for a 32-token prefill.
//
// This file gives each rank the same fused device kernels the single-GPU path
// uses (see Qwen35Model.CudaMoE.cs): router top-k, per-expert gate/up matvec,
// SwiGLU, routing-weighted down accumulate and the gated shared expert all run
// on the rank's own GPU with no host round-trip. Under TP the experts are
// Megatron-split (gate/up column-parallel, down row-parallel), so each rank
// produces a PARTIAL [seqLen, hidden] output and the caller AllReduces:
//
//   down(h) = sum_r  h_r @ down_r^T,   h_r = silu(gate_r(x)) * up_r(x)
//
// which is exactly the full product. The routing weight and the shared-expert
// sigmoid gate are scalars applied per rank, so they distribute over the sum.
// ============================================================================
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // Per-layer, per-rank device pointer tables addressing that rank's shard
        // of every expert's gate / up / down weight, indexed by expert id.
        private IntPtr[][] _tpMoEGatePtrTable;   // [layer][rank]
        private IntPtr[][] _tpMoEUpPtrTable;     // [layer][rank]
        private IntPtr[][] _tpMoEDownPtrTable;   // [layer][rank]
        private int[] _tpMoEGateUpType;          // [layer]
        private int[] _tpMoEDownType;            // [layer]
        private bool _tpCudaMoETablesReady;
        private bool _tpCudaMoEUsable;

        // Per-rank expert FFN width after the column-parallel split.
        private int _tpMoENFfLocal;

        private static readonly bool s_tpMoePrefillOnDevice =
            Environment.GetEnvironmentVariable("TS_TP_MOE_PREFILL_ONDEVICE") != "0";

        /// <summary>
        /// Build the per-rank expert pointer tables from the TP shards resident on
        /// each rank's GPU. Any gap (a shard that never made it to the device, a
        /// gate/up type mismatch) disables the on-device path for the whole model
        /// and leaves the host fallback in charge.
        /// </summary>
        private void EnsureQwenTpMoETablesBuilt()
        {
            if (_tpCudaMoETablesReady)
                return;
            _tpCudaMoETablesReady = true;
            _tpCudaMoEUsable = false;

            if (!IsTensorParallel || _backend != BackendType.Cuda
                || !s_qwenCudaMoeOnDeviceEnabled || _numExperts <= 0)
                return;

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            if (_expertFfnLength % globalTp != 0)
                return;
            _tpMoENFfLocal = _expertFfnLength / globalTp;

            int numLayers = Config.NumLayers;
            _tpMoEGatePtrTable = new IntPtr[numLayers][];
            _tpMoEUpPtrTable = new IntPtr[numLayers][];
            _tpMoEDownPtrTable = new IntPtr[numLayers][];
            _tpMoEGateUpType = new int[numLayers];
            _tpMoEDownType = new int[numLayers];

            for (int l = 0; l < numLayers; l++)
            {
                if (_isMoeLayer == null || l >= _isMoeLayer.Length || !_isMoeLayer[l])
                    continue;

                string prefix = $"blk.{l}.";
                var gateTables = new IntPtr[tp];
                var upTables = new IntPtr[tp];
                var downTables = new IntPtr[tp];
                int gateUpType = -1;
                int downType = -1;
                bool ok = true;

                for (int r = 0; r < tp && ok; r++)
                {
                    if (_tpGroup.GetAllocator(r) is not CudaAllocator alloc)
                    {
                        ok = false;
                        break;
                    }

                    var gPtrs = new IntPtr[_numExperts];
                    var uPtrs = new IntPtr[_numExperts];
                    var dPtrs = new IntPtr[_numExperts];

                    for (int e = 0; e < _numExperts; e++)
                    {
                        // The kernels are told nFf == _tpMoENFfLocal, so a shard whose
                        // real geometry differs (an uneven block split) must not be
                        // reachable through the pointer table.
                        if (!TpShardHasExpectedShape(prefix + $"ffn_gate_exps.{e}.weight", r, Config.HiddenSize, _tpMoENFfLocal)
                            || !TpShardHasExpectedShape(prefix + $"ffn_up_exps.{e}.weight", r, Config.HiddenSize, _tpMoENFfLocal)
                            || !TpShardHasExpectedShape(prefix + $"ffn_down_exps.{e}.weight", r, _tpMoENFfLocal, Config.HiddenSize))
                        {
                            ok = false;
                            break;
                        }

                        if (!TryGetTpShardDevicePtr(alloc, prefix + $"ffn_gate_exps.{e}.weight", r, out IntPtr gDev, out int gType)
                            || !TryGetTpShardDevicePtr(alloc, prefix + $"ffn_up_exps.{e}.weight", r, out IntPtr uDev, out int uType)
                            || !TryGetTpShardDevicePtr(alloc, prefix + $"ffn_down_exps.{e}.weight", r, out IntPtr dDev, out int dType)
                            || gType != uType)
                        {
                            // gate and up must share a type: they run the same
                            // projection kernel against one activation buffer.
                            ok = false;
                            break;
                        }

                        if (gateUpType < 0) { gateUpType = gType; downType = dType; }
                        else if (gateUpType != gType || downType != dType) { ok = false; break; }

                        gPtrs[e] = gDev;
                        uPtrs[e] = uDev;
                        dPtrs[e] = dDev;
                    }

                    if (!ok)
                        break;

                    gateTables[r] = CudaQuantizedOps.CreateDevicePointerTable(alloc, gPtrs);
                    upTables[r] = CudaQuantizedOps.CreateDevicePointerTable(alloc, uPtrs);
                    downTables[r] = CudaQuantizedOps.CreateDevicePointerTable(alloc, dPtrs);
                }

                if (!ok || gateUpType < 0)
                {
                    FreeTpMoePtrTableRow(gateTables);
                    FreeTpMoePtrTableRow(upTables);
                    FreeTpMoePtrTableRow(downTables);
                    FreeQwenTpMoETables();
                    return; // any gap -> host path for the whole model
                }

                _tpMoEGatePtrTable[l] = gateTables;
                _tpMoEUpPtrTable[l] = upTables;
                _tpMoEDownPtrTable[l] = downTables;
                _tpMoEGateUpType[l] = gateUpType;
                _tpMoEDownType[l] = downType;
            }

            _tpCudaMoEUsable = true;
        }

        /// <summary>
        /// Whether the rank's shard of <paramref name="weightName"/> really has the
        /// geometry the fused kernels will assume. Guards against an uneven
        /// quant-block split silently handing the kernel a shorter row.
        /// </summary>
        private bool TpShardHasExpectedShape(string weightName, int rank, long expectedNe0, long expectedNe1)
        {
            if (!_tpQuantWeights.TryGetValue(weightName, out var shards)
                || shards == null || rank >= shards.Length || shards[rank] == null)
                return false;
            var qw = shards[rank];
            return qw.Ne0 == expectedNe0 && qw.Ne1 == expectedNe1;
        }

        private bool TryGetTpShardDevicePtr(CudaAllocator alloc, string weightName, int rank,
            out IntPtr devicePtr, out int ggmlType)
        {
            devicePtr = IntPtr.Zero;
            ggmlType = 0;
            if (!_tpQuantWeights.TryGetValue(weightName, out var shards)
                || shards == null || rank >= shards.Length || shards[rank] == null)
                return false;

            return CudaQuantizedOps.TryGetResidentDevicePtr(
                alloc, shards[rank].EnsureDeviceCacheKey(), out devicePtr, out ggmlType);
        }

        private bool CanUseTpCudaMoE(int layer)
        {
            EnsureQwenTpMoETablesBuilt();
            return _tpCudaMoEUsable
                && _numExpertsUsed > 0
                && _numExpertsUsed <= _numExperts
                && _numExpertsUsed <= CudaFusedOps.MaxMoEExpertsUsed
                && _tpMoEGatePtrTable != null
                && layer >= 0
                && layer < _tpMoEGatePtrTable.Length
                && _tpMoEGatePtrTable[layer] != null;
        }

        /// <summary>
        /// Fused on-device MoE for one rank. <paramref name="input"/> is that rank's
        /// RMSNorm'd MoE input and <paramref name="routerLogits"/> the rank-local copy
        /// of the (identical on every rank) router logits. Returns this rank's PARTIAL
        /// [seqLen, hidden] contribution — the caller must AllReduce — or null to fall
        /// back to the host path (leaving <paramref name="routerLogits"/> alive).
        /// </summary>
        private Tensor TryMoEForwardTpOnDevice(Tensor input, Tensor routerLogits, int layer, int rank, int seqLen)
        {
            // The fused router kernel implements the normalized top-k softmax
            // (ts_moe_router_f32); any other routing must use the host path.
            if (!_normTopKProb || !CanUseTpCudaMoE(layer))
                return null;
            // Unlike single-GPU — where prefill falls back to a fast expert-grouped
            // batched matmul and the on-device kernels are opt-in — the TP fallback
            // is the naive per-(token, expert) loop, so on-device wins for prefill
            // too. TS_TP_MOE_PREFILL_ONDEVICE=0 forces the host path.
            if (seqLen > 1 && (seqLen > CudaMaxGridDim || !s_tpMoePrefillOnDevice))
                return null;
            if (_tpGroup.GetAllocator(rank) is not CudaAllocator alloc)
                return null;

            int hiddenDim = Config.HiddenSize;
            int nFf = _tpMoENFfLocal;
            int nUsed = _numExpertsUsed;
            int gateUpType = _tpMoEGateUpType[layer];
            int downType = _tpMoEDownType[layer];

            long t0 = Stopwatch.GetTimestamp();

            // Shared expert on this rank's shards: gate/up are column-parallel and
            // down is row-parallel, so sharedDown is itself a partial that the same
            // AllReduce completes.
            Tensor sharedDown = null;
            IntPtr sharedGateVecPtr = IntPtr.Zero;
            string prefix = $"blk.{layer}.";
            if (_hasSharedExperts != null && layer < _hasSharedExperts.Length && _hasSharedExperts[layer])
            {
                Tensor sg = TryTpExpertLinearLocal(input, prefix + "ffn_gate_shexp.weight", rank, seqLen);
                Tensor su = TryTpExpertLinearLocal(input, prefix + "ffn_up_shexp.weight", rank, seqLen);
                if (sg != null && su != null)
                {
                    Ops.SiLUMul(sg, sg, su);
                    sharedDown = TryTpExpertLinearLocal(sg, prefix + "ffn_down_shexp.weight", rank, seqLen);
                }
                su?.Dispose();
                sg?.Dispose();

                if (sharedDown == null)
                    return null;

                bool sharedGateRequired = _hasSharedExpertGate != null
                    && layer < _hasSharedExpertGate.Length
                    && _hasSharedExpertGate[layer];
                if (sharedGateRequired)
                {
                    Tensor sharedGateVec = _ffnGateInpShexpVec != null
                        && layer < _ffnGateInpShexpVec.Length
                        ? _ffnGateInpShexpVec[layer]
                        : null;
                    // The gate vector is replicated (not sharded). It must live on
                    // THIS rank's GPU or the kernel would dereference a pointer
                    // belonging to another device.
                    if (sharedGateVec != null && sharedGateVec.ElementCount() >= hiddenDim)
                    {
                        sharedGateVecPtr = CudaFusedOps.GetDeviceResidentPtr(ReplicaOnRank(sharedGateVec, rank));
                        if (sharedGateVecPtr == IntPtr.Zero)
                        {
                            sharedDown.Dispose();
                            return null;
                        }
                    }
                    else
                    {
                        sharedDown.Dispose();
                        return null;
                    }
                }
            }

            var output = new Tensor(alloc, DType.Float32, seqLen, hiddenDim);
            var selected = new Tensor(alloc, DType.Int32, (long)seqLen * nUsed);
            var routeW = new Tensor(alloc, DType.Float32, (long)seqLen * nUsed);
            var gateOut = new Tensor(alloc, DType.Float32, (long)seqLen * nUsed, nFf);
            var upOut = new Tensor(alloc, DType.Float32, (long)seqLen * nUsed, nFf);

            bool gateUpDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(gateUpType)
                && IsMoeDp4aDimensionSupported(gateUpType, hiddenDim);
            bool downDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(downType)
                && IsMoeDp4aDimensionSupported(downType, nFf);
            Tensor moeInputQ8 = null, gateOutQ8 = null;
            if (gateUpDp4a)
                moeInputQ8 = new Tensor(alloc, DType.UInt8, (long)seqLen * (hiddenDim / 32) * CudaFusedOps.Q81BlockBytes);
            if (downDp4a)
                gateOutQ8 = new Tensor(alloc, DType.UInt8, (long)seqLen * nUsed * (nFf / 32) * CudaFusedOps.Q81BlockBytes);

            bool ok = seqLen == 1
                ? CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU(
                    routerLogits, input, output, selected, routeW, gateOut, upOut,
                    IntPtr.Zero,
                    _tpMoEGatePtrTable[layer][rank], _tpMoEUpPtrTable[layer][rank], _tpMoEDownPtrTable[layer][rank],
                    gateUpType, downType, _numExperts, nUsed, hiddenDim, nFf,
                    sharedDown, sharedGateVecPtr,
                    moeInputQ8, gateOutQ8, gateUpDp4a, downDp4a)
                : CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(
                    routerLogits, input, output, selected, routeW, gateOut, upOut,
                    IntPtr.Zero,
                    _tpMoEGatePtrTable[layer][rank], _tpMoEUpPtrTable[layer][rank], _tpMoEDownPtrTable[layer][rank],
                    gateUpType, downType, _numExperts, nUsed, hiddenDim, nFf, seqLen,
                    sharedDown, sharedGateVecPtr,
                    moeInputQ8, gateOutQ8, gateUpDp4a, downDp4a);

            gateOutQ8?.Dispose();
            moeInputQ8?.Dispose();
            upOut.Dispose();
            gateOut.Dispose();
            routeW.Dispose();
            selected.Dispose();
            sharedDown?.Dispose();

            if (!ok)
            {
                output.Dispose();
                return null; // routerLogits still alive for the host fallback
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            if (seqLen > 1)
                InvalidateTensorDeviceCache(output);
            return output;
        }

        /// <summary>
        /// <see cref="TpExpertLinear"/> for an input that is ALREADY on the target
        /// rank's GPU: no cross-GPU replication, and returns null (instead of
        /// throwing) when the weight is not sharded, so callers can fall back.
        /// </summary>
        private Tensor TryTpExpertLinearLocal(Tensor input, string weightName, int rank, int seqLen)
        {
            var alloc = _tpGroup.GetAllocator(rank);

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                var qw = qShards[rank];
                var result = new Tensor(alloc, DType.Float32, seqLen, (int)qw.Ne1);
                AddmmQuantManaged(result, input, qw);
                return result;
            }
            if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                var w = wShards[rank];
                var result = new Tensor(alloc, DType.Float32, seqLen, (int)w.Sizes[0]);
                using var wT = w.Transpose();
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
                return result;
            }
            return null;
        }

        private void FreeQwenTpMoETables()
        {
            FreeTpMoePtrTables(_tpMoEGatePtrTable);
            FreeTpMoePtrTables(_tpMoEUpPtrTable);
            FreeTpMoePtrTables(_tpMoEDownPtrTable);
            _tpMoEGatePtrTable = null;
            _tpMoEUpPtrTable = null;
            _tpMoEDownPtrTable = null;
            _tpCudaMoEUsable = false;
        }

        private void FreeTpMoePtrTables(IntPtr[][] tables)
        {
            if (tables == null)
                return;
            for (int l = 0; l < tables.Length; l++)
                FreeTpMoePtrTableRow(tables[l]);
        }

        private void FreeTpMoePtrTableRow(IntPtr[] perRank)
        {
            if (perRank == null)
                return;
            for (int r = 0; r < perRank.Length; r++)
            {
                if (perRank[r] != IntPtr.Zero && _tpGroup?.GetAllocator(r) is CudaAllocator alloc)
                    CudaQuantizedOps.FreeDeviceBuffer(alloc, perRank[r]);
                perRank[r] = IntPtr.Zero;
            }
        }
    }
}
