// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Routed-expert CPU offload for the direct-CUDA DeepSeek V4 engine
// (llama.cpp's --n-cpu-moe, and what the ggml executor does in
// ggml_ops_deepseek4.cpp).
//
// 91% of a V4 Flash checkpoint is routed-expert weights -- 137 of 151 GiB at
// UD-Q8_K_XL -- so on any box where the model outweighs the cards, they are
// the only thing worth moving. For a layer selected by the split, the three
// stacked expert tensors stay in system RAM and their FFN runs on the host
// through ManagedQuantizedOps; the router, the norms, the attention stack and
// the always-active shared expert stay on the GPU. Per offloaded layer the bus
// carries only the [n_tokens, n_embd] activation down and the
// [n_tokens*n_used, n_embd] expert output back.
//
// The host output is written in SLOT order (row = token*n_used + k), which is
// exactly what ts_dsv4_moe_scatter_add_f32 reads when rowOfSlot is null, so
// the device-side epilogue is shared with the resident path and the two are
// interchangeable layer by layer.
using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Dsv4CudaEngine
    {
        /// <summary>
        /// One layer's routed experts held in host memory: the stacked
        /// [ne0, ne1, n_expert] quantized block exactly as it sits in the GGUF,
        /// so expert i's matrix starts at <c>i * Ne1 * RowBytes</c>.
        /// </summary>
        private struct HostQW
        {
            public IntPtr Ptr;
            public int Type;
            public int Ne0;      // input dim (row length)
            public int Ne1;      // output dim (rows per expert)
            public long RowBytes;
            public long ExpertBytes => (long)Ne1 * RowBytes;
            public bool IsValid => Ptr != IntPtr.Zero;
        }

        private sealed class HostMoeLayer
        {
            public HostQW Gate, Up, Down;
            public float ClampExp;
        }

        // Indexed by layer; null for layers whose experts stayed on the GPU.
        private HostMoeLayer[] _hostMoe;
        private int _nCpuMoe;
        private IDsv4HostMatMul _hostMatMul;

        /// <summary>Host buffers holding the offloaded expert weights; freed in Dispose.</summary>
        private readonly System.Collections.Generic.List<IntPtr> _hostMoeBuffers = new System.Collections.Generic.List<IntPtr>();

        /// <summary>True when at least one layer's routed experts live on the host.</summary>
        private bool HostMoeActive => _nCpuMoe > 0;

        /// <summary>
        /// Read one layer's three stacked expert tensors into system RAM. Called
        /// from the (parallel) weight-load pass, so it must not touch device
        /// state; the read goes through the same shard source the VRAM uploads
        /// use, which is documented thread-safe for positional reads.
        /// </summary>
        private HostMoeLayer LoadHostExperts(LayerDesc src)
        {
            var hm = new HostMoeLayer { ClampExp = src.ClampExp };
            hm.Gate = StageHost(src.GateExps);
            hm.Up = StageHost(src.UpExps);
            hm.Down = StageHost(src.DownExps);
            // The gate and up projections are batched into one dispatch, which
            // carries a single quantized type.
            if (hm.Gate.Type != hm.Up.Type)
                throw new NotSupportedException("[dsv4-cuda] host MoE offload needs matching gate/up expert quant types");
            return hm;
        }

        private HostQW StageHost(in QuantWeightDesc d)
        {
            if (!d.IsValid)
                throw new InvalidOperationException("[dsv4-cuda] cannot offload an expert weight that has no source");

            long bytes = d.TotalBytes;
            IntPtr host;
            if (d.HostPtr != IntPtr.Zero)
            {
                // Already resident (mmap'd or pre-staged): borrow it, no copy and
                // no second 3 GiB of RAM per layer.
                host = d.HostPtr;
            }
            else
            {
                host = Marshal.AllocHGlobal((nint)bytes);
                lock (_hostMoeBuffers)
                    _hostMoeBuffers.Add(host);
                d.Source.Read(d.SourceOffset, host, bytes);
            }

            return new HostQW
            {
                Ptr = host,
                Type = d.GgmlType,
                Ne0 = d.Ne0,
                Ne1 = d.Ne1,
                RowBytes = d.RowBytes,
            };
        }

        private void FreeHostMoeBuffers()
        {
            foreach (IntPtr p in _hostMoeBuffers)
                Marshal.FreeHGlobal(p);
            _hostMoeBuffers.Clear();
        }

        /// <summary>
        /// Routed-expert FFN for an offloaded layer. Replaces everything the
        /// resident path does between the router and
        /// <c>MoeScatterAdd</c>: it fills <c>dev.ExpDown</c> in slot order and
        /// leaves the weighted sum + shared-expert add to the same kernel.
        /// </summary>
        private void MoeFfnHost(Dev dev, HostMoeLayer hm, int il, int nt)
        {
            var m = _m;
            int e = m.NEmbd, ff = m.NFfExp, nUsed = m.NExpertUsed;
            int slots = nt * nUsed;

            // The host reads what the GPU just produced, so the stream has to be
            // drained first; everything queued after this call is ordered behind
            // the upload at the end.
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();

            float[] cur = ArrayPool<float>.Shared.Rent(nt * e);
            int[] sel = ArrayPool<int>.Shared.Rent(slots);
            float[] outRows = ArrayPool<float>.Shared.Rent(slots * e);
            float[] gate = ArrayPool<float>.Shared.Rent(slots * ff);
            float[] up = ArrayPool<float>.Shared.Rent(slots * ff);
            try
            {
                fixed (float* curP = cur)
                fixed (int* selP = sel)
                fixed (float* outP = outRows)
                fixed (float* gateP = gate)
                fixed (float* upP = up)
                {
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)curP, Ptr(dev.Cur), new UIntPtr((ulong)nt * (ulong)e * sizeof(float)))
                        .ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)selP, Ptr(dev.Sel), new UIntPtr((ulong)slots * sizeof(int)))
                        .ThrowOnError();

                    HostExpertFfn(hm, curP, selP, gateP, upP, outP, nt, nUsed, e, ff);

                    CudaDriverApi.cuMemcpyHtoD(Ptr(dev.ExpDown), (IntPtr)outP,
                        new UIntPtr((ulong)slots * (ulong)e * sizeof(float))).ThrowOnError();
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(up);
                ArrayPool<float>.Shared.Return(gate);
                ArrayPool<float>.Shared.Return(outRows);
                ArrayPool<int>.Shared.Return(sel);
                ArrayPool<float>.Shared.Return(cur);
            }
        }

        /// <summary>
        /// gate/up/down over the selected experts, host side.
        ///
        /// Tokens are grouped by expert so each expert's weight block is read
        /// once per call rather than once per token: at prefill widths that is
        /// the difference between streaming ~3 GiB of expert blocks per layer
        /// and streaming it n_tokens times. Within an expert the three
        /// projections go through the shared managed quantized matmul, which
        /// parallelizes over output rows — the outer loop stays serial so the
        /// two levels do not oversubscribe each other.
        /// </summary>
        private void HostExpertFfn(HostMoeLayer hm, float* cur, int* sel,
            float* gate, float* up, float* outRows, int nt, int nUsed, int e, int ff)
        {
            int slots = nt * nUsed;

            // slot -> expert, bucketed. n_expert is 256 and slots is at most
            // n_ubatch*n_used, so a count+prefix pass is cheaper than a dictionary.
            int nExpert = _m.NExpert;
            int[] counts = ArrayPool<int>.Shared.Rent(nExpert + 1);
            int[] order = ArrayPool<int>.Shared.Rent(slots);
            float[] rows = ArrayPool<float>.Shared.Rent(slots * e);
            try
            {
                Array.Clear(counts, 0, nExpert + 1);
                for (int i = 0; i < slots; i++)
                {
                    int x = sel[i];
                    if ((uint)x >= (uint)nExpert)
                        throw new InvalidOperationException($"[dsv4-cuda] host MoE: expert id {x} out of range");
                    counts[x + 1]++;
                }
                for (int x = 0; x < nExpert; x++)
                    counts[x + 1] += counts[x];

                int[] cursor = ArrayPool<int>.Shared.Rent(nExpert);
                Array.Copy(counts, cursor, nExpert);
                for (int i = 0; i < slots; i++)
                    order[cursor[sel[i]]++] = i;
                ArrayPool<int>.Shared.Return(cursor);

                fixed (float* rowsP = rows)
                {
                    // Gather every slot's activation row into grouped order up
                    // front, so all of an expert's rows are contiguous and the
                    // three projections below can be batched across experts.
                    for (int g = 0; g < slots; g++)
                    {
                        int slot = order[g];
                        Buffer.MemoryCopy(cur + (long)(slot / nUsed) * e, rowsP + (long)g * e,
                            (long)e * sizeof(float), (long)e * sizeof(float));
                    }

                    // Two dispatches per layer instead of 3 per active expert:
                    // gate+up together (they share the activation rows), then
                    // down after the activation.
                    int nActive = 0;
                    for (int x = 0; x < nExpert; x++)
                        if (counts[x + 1] > counts[x]) nActive++;

                    var gu = new IDsv4HostMatMul.Job[2 * nActive];
                    var dn = new IDsv4HostMatMul.Job[nActive];
                    int gi = 0, di = 0;
                    for (int x = 0; x < nExpert; x++)
                    {
                        int begin = counts[x], k = counts[x + 1] - begin;
                        if (k == 0)
                            continue;
                        IntPtr inRows = (IntPtr)(rowsP + (long)begin * e);
                        gu[gi++] = new IDsv4HostMatMul.Job(
                            (IntPtr)((byte*)hm.Gate.Ptr + (long)x * hm.Gate.ExpertBytes),
                            inRows, (IntPtr)(gate + (long)begin * ff), ff, k, ff);
                        gu[gi++] = new IDsv4HostMatMul.Job(
                            (IntPtr)((byte*)hm.Up.Ptr + (long)x * hm.Up.ExpertBytes),
                            inRows, (IntPtr)(up + (long)begin * ff), ff, k, ff);
                        dn[di++] = new IDsv4HostMatMul.Job(
                            (IntPtr)((byte*)hm.Down.Ptr + (long)x * hm.Down.ExpertBytes),
                            (IntPtr)(gate + (long)begin * ff), (IntPtr)(outRows + (long)begin * e), e, k, e);
                    }

                    Batch(hm.Gate.Type, e, e, gu);
                    SiLUMulClampHost(gate, up, (long)slots * ff, hm.ClampExp);
                    Batch(hm.Down.Type, ff, ff, dn);
                }

                // grouped order -> slot order (what moe_scatter_add reads with a
                // null rowOfSlot)
                float[] tmp = ArrayPool<float>.Shared.Rent(slots * e);
                try
                {
                    fixed (float* tmpP = tmp)
                    {
                        Buffer.MemoryCopy(outRows, tmpP, (long)slots * e * sizeof(float), (long)slots * e * sizeof(float));
                        for (int j = 0; j < slots; j++)
                            Buffer.MemoryCopy(tmpP + (long)j * e, outRows + (long)order[j] * e,
                                (long)e * sizeof(float), (long)e * sizeof(float));
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(tmp);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rows);
                ArrayPool<int>.Shared.Return(order);
                ArrayPool<int>.Shared.Return(counts);
            }
        }

        private void Batch(int ggmlType, int inDim, int inputRowStride, IDsv4HostMatMul.Job[] jobs)
        {
            if (!_hostMatMul.TryMatMulBatch(ggmlType, inDim, inputRowStride, jobs))
            {
                throw new NotSupportedException(
                    $"[dsv4-cuda] host MoE offload does not support quantized type {ggmlType}; " +
                    "run on a machine with enough VRAM for every expert, or use --backend ggml_cuda.");
            }
        }

        /// <summary>
        /// <c>gate = clamp(up, ±limit) * SiLU(min(gate, limit))</c>, in place on
        /// <paramref name="gate"/> — the same arithmetic, in the same order, as
        /// <c>TensorApplyCPU.SiLUMulClamp</c>, which is what
        /// <c>Ops.SiLUMulClamp</c> runs for the resident path. Chunked so the
        /// two scratch spans stay in cache: a prefill layer activates
        /// n_tokens*n_used*n_ff floats and a scalar loop over that was worth
        /// several percent of the whole host FFN.
        /// </summary>
        private static void SiLUMulClampHost(float* gate, float* up, long n, float limit)
        {
            const int Chunk = 16384;
            int rentLen = (int)Math.Min(n, Chunk);
            float[] gRent = ArrayPool<float>.Shared.Rent(rentLen);
            float[] uRent = ArrayPool<float>.Shared.Rent(rentLen);
            try
            {
                for (long off = 0; off < n; off += Chunk)
                {
                    int len = (int)Math.Min(Chunk, n - off);
                    var gateSpan = new Span<float>(gate + off, len);
                    var upSpan = new ReadOnlySpan<float>(up + off, len);
                    Span<float> g = gRent.AsSpan(0, len);
                    Span<float> u = uRent.AsSpan(0, len);

                    if (limit > 0.0f)
                    {
                        TensorPrimitives.Min(gateSpan, limit, g);
                        TensorPrimitives.Max(upSpan, -limit, u);
                        TensorPrimitives.Min(u, limit, u);
                    }
                    else
                    {
                        gateSpan.CopyTo(g);
                        upSpan.CopyTo(u);
                    }

                    TensorPrimitives.Sigmoid(g, gateSpan);        // sigmoid(g)
                    TensorPrimitives.Multiply(g, gateSpan, gateSpan);  // SiLU(g)
                    TensorPrimitives.Multiply(gateSpan, u, gateSpan);  // * clamp(up)
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(uRent);
                ArrayPool<float>.Shared.Return(gRent);
            }
        }
    }
}
