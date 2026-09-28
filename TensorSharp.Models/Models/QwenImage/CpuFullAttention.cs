// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Non-causal multi-head attention for the pure-C# (BackendType.Cpu) vision tower, straight from
// the row-major [n, heads*dim] q / k / v projections: out[i, h] = sum_j softmax_j(scale * q_ih .
// k_jh) v_jh. The generic CPU path (head-first copies, a [heads, n, n] score tensor through
// Ops.AddmmBatch and Ops.Softmax, a transpose back) ran at ~100 GF/s and was ~70% of a 1024x1024
// reference image's 29 s vision encode (4096 patches, 27 blocks x 77 GFLOP of attention). This
// path runs a layer in ~150 ms (~500 GF/s on the i7-11800H), and the encode takes 12-13 s.
//
// Here every (head, block of 64 queries) is one pool task running two serial packed GEMMs
// (CpuPackedGemm) on a thread-local slab, so a layer is one dispatch, not hundreds:
//
//     S^T [n, R]   = K_h [n, dim] . Q_blk^T [dim, R]      (K_h packed once per head per layer)
//     E            = exp(scale * (S^T - colmax))           (column softmax, vectorized over queries)
//     O^T [dim, R] = V_h^T [dim, n] . E [n, R]            (V_h^T packed once per head per layer)
//     out[i, h]    = O^T[:, i] / colsum_i
//
// Scores are kept transposed so both GEMMs have a clean shape: dim = 72 is exactly 6 (12) row
// panels of the AVX-512 (AVX2) kernel as the M of the PV product, and E is read row-major as
// its B operand. The slab is n x 64 floats (1 MB at 4096 patches): it never leaves L2/L3, and
// no n x n tensor exists. Normalizing after the PV product (not before) is the usual flash-style
// order; it differs from softmax-then-matmul by float rounding only.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Per-head packed K and V^T, reused across layers and calls (same shapes every
    /// block of a tower, so after the first layer no pack allocates).</summary>
    internal sealed class CpuAttentionWorkspace
    {
        internal PackedPanels[] K = Array.Empty<PackedPanels>();
        internal PackedPanels[] Vt = Array.Empty<PackedPanels>();
    }

    internal static unsafe class CpuFullAttention
    {
        // Queries per task: the slab is n x QueryBlock floats, and 64 is a whole number of B
        // panels for every kernel (32 / 16 / 8 columns).
        internal const int QueryBlock = 64;

        [ThreadStatic] private static float[] t_slabRaw;
        [ThreadStatic] private static nint t_slab;
        [ThreadStatic] private static int t_slabFloats;

        /// <summary>
        /// out[i, h*dim + d] = sum_j softmax_j(scale * q[i, h] . k[j, h]) v[j, h*dim + d] for n
        /// tokens; q, k, v and out are row-major [n, heads*dim].
        /// </summary>
        internal static void Run(float* q, float* k, float* v, float* output, int n, int heads, int dim, float scale,
            CpuAttentionWorkspace workspace, CpuWorkerPool pool = null)
        {
            if (n <= 0) return;
            pool ??= CpuWorkerPool.Shared;
            CpuGemmIsa isa = CpuPackedGemm.Isa;
            long stride = (long)heads * dim;
            if (workspace.K.Length != heads)
            {
                workspace.K = new PackedPanels[heads];
                workspace.Vt = new PackedPanels[heads];
            }
            PackedPanels[] kPacks = workspace.K, vtPacks = workspace.Vt;
            nint qL = (nint)q, kL = (nint)k, vL = (nint)v, oL = (nint)output;

            // K_h as the A operand of the score GEMM ([n, dim], rows = keys), V_h^T as the A
            // operand of the PV GEMM ([dim, n]); one serial pack per task, 2 x heads tasks.
            pool.For(2 * heads, t =>
            {
                int h = t >> 1;
                if ((t & 1) == 0)
                    kPacks[h] = CpuPackedGemm.PackA((float*)kL + (long)h * dim, n, dim, stride, 1, isa,
                        reuse: kPacks[h], serial: true);
                else
                    vtPacks[h] = CpuPackedGemm.PackA((float*)vL + (long)h * dim, dim, n, 1, stride, isa,
                        reuse: vtPacks[h], serial: true);
            });

            int blocks = (n + QueryBlock - 1) / QueryBlock;
            pool.For(heads * blocks, t =>
            {
                int h = t / blocks, i0 = (t - h * blocks) * QueryBlock;
                int rows = Math.Min(QueryBlock, n - i0);
                float* slab = Slab((long)n * rows + (long)dim * rows);
                float* outT = slab + (long)n * rows;
                float* colMax = stackalloc float[QueryBlock];
                float* colSum = stackalloc float[QueryBlock];

                // S^T[j, i] = k_j . q_(i0+i) (B = Q_blk^T: element (d, i) at q[(i0+i)*stride + h*dim + d]).
                CpuPackedGemm.Gemm(kPacks[h], new StridedPanelSource((float*)qL + i0 * stride + (long)h * dim, 1, stride, rows),
                    rows, slab, rows, serial: true);
                ColumnSoftmaxNumerators(slab, n, rows, scale, colMax, colSum);
                // O^T[d, i] = sum_j v_j[d] E[j, i] (B = E, row-major [n, rows]).
                CpuPackedGemm.Gemm(vtPacks[h], new StridedPanelSource(slab, rows, 1, rows), rows, outT, rows, serial: true);

                float* o = (float*)oL + i0 * stride + (long)h * dim;
                for (int i = 0; i < rows; i++)
                {
                    float inv = 1f / colSum[i];
                    float* oi = o + i * stride;
                    for (int d = 0; d < dim; d++) oi[d] = outT[(long)d * rows + i] * inv;
                }
            });
        }

        // In place over the [n, rows] slab: E[j, i] = exp(scale * (S[j, i] - max_j S[j, i])), and
        // sum[i] = sum_j E[j, i] (float, in j order). Walks rows, vectorized across the queries:
        // 16 per zmm when the GEMM runs AVX-512 (TS_CPU_DISABLE_AVX512 covers this too), else 8.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void ColumnSoftmaxNumerators(float* s, int n, int rows, float scale, float* colMax, float* colSum)
        {
            for (int i = 0; i < rows; i++) { colMax[i] = float.NegativeInfinity; colSum[i] = 0f; }
            if (CpuPackedGemm.Isa == CpuGemmIsa.Avx512 && rows % 16 == 0)
            {
                for (int j = 0; j < n; j++)
                {
                    float* r = s + (long)j * rows;
                    for (int i = 0; i < rows; i += 16)
                        Vector512.Store(Vector512.MaxNative(Vector512.Load(colMax + i), Vector512.Load(r + i)), colMax + i);
                }
                var vScale512 = Vector512.Create(scale);
                for (int j = 0; j < n; j++)
                {
                    float* r = s + (long)j * rows;
                    for (int i = 0; i < rows; i += 16)
                    {
                        var e = ExpNonPositive((Vector512.Load(r + i) - Vector512.Load(colMax + i)) * vScale512);
                        Vector512.Store(e, r + i);
                        Vector512.Store(Vector512.Load(colSum + i) + e, colSum + i);
                    }
                }
                return;
            }
            int vec = rows & ~7;
            for (int j = 0; j < n; j++)
            {
                float* r = s + (long)j * rows;
                int i = 0;
                for (; i < vec; i += 8) Vector256.Store(Vector256.MaxNative(Vector256.Load(colMax + i), Vector256.Load(r + i)), colMax + i);
                for (; i < rows; i++) colMax[i] = MathF.Max(colMax[i], r[i]);
            }
            var vScale = Vector256.Create(scale);
            for (int j = 0; j < n; j++)
            {
                float* r = s + (long)j * rows;
                int i = 0;
                for (; i < vec; i += 8)
                {
                    var e = ExpNonPositive((Vector256.Load(r + i) - Vector256.Load(colMax + i)) * vScale);
                    Vector256.Store(e, r + i);
                    Vector256.Store(Vector256.Load(colSum + i) + e, colSum + i);
                }
                for (; i < rows; i++)
                {
                    float e = MathF.Exp((r[i] - colMax[i]) * scale);
                    r[i] = e;
                    colSum[i] += e;
                }
            }
        }

        // exp(x) for the softmax's x <= 0, within ~2 ulp of MathF.Exp: x = n*ln2 + r with |r| <=
        // ln2/2 (ln2 split in two so n*LN2_HI is exact), a degree-6 Cephes polynomial for e^r, and
        // 2^n put straight into the exponent bits. The library Vector256.Exp evaluates float exp
        // in double and cost as much as a fifth of the attention at 4096 patches; this is ~12
        // float ops. Below -87 it returns ~1e-38 instead of a denormal or 0: negligible next to
        // the column's own max term, e^0 = 1, in the softmax sum.
        private const float Log2E = 1.44269504088896341f, Ln2Hi = 0.693359375f, Ln2Lo = -2.12194440e-4f;
        private const float P0 = 1.9875691500E-4f, P1 = 1.3981999507E-3f, P2 = 8.3334519073E-3f,
            P3 = 4.1665795894E-2f, P4 = 1.6666665459E-1f, P5 = 5.0000001201E-1f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<float> ExpNonPositive(Vector256<float> x)
        {
            x = Vector256.MaxNative(Vector256.Create(-87f), x);   // NaN operand second: it propagates
            var fx = Vector256.Round(x * Vector256.Create(Log2E));
            var r = Vector256.MultiplyAddEstimate(fx, Vector256.Create(-Ln2Hi), x);
            r = Vector256.MultiplyAddEstimate(fx, Vector256.Create(-Ln2Lo), r);
            var p = Vector256.MultiplyAddEstimate(Vector256.Create(P0), r, Vector256.Create(P1));
            p = Vector256.MultiplyAddEstimate(p, r, Vector256.Create(P2));
            p = Vector256.MultiplyAddEstimate(p, r, Vector256.Create(P3));
            p = Vector256.MultiplyAddEstimate(p, r, Vector256.Create(P4));
            p = Vector256.MultiplyAddEstimate(p, r, Vector256.Create(P5));
            var y = Vector256.MultiplyAddEstimate(p, r * r, r + Vector256<float>.One);
            var pow2n = Vector256.ShiftLeft(Vector256.ConvertToInt32Native(fx) + Vector256.Create(127), 23).AsSingle();
            return y * pow2n;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector512<float> ExpNonPositive(Vector512<float> x)
        {
            x = Vector512.MaxNative(Vector512.Create(-87f), x);
            var fx = Vector512.Round(x * Vector512.Create(Log2E));
            var r = Vector512.MultiplyAddEstimate(fx, Vector512.Create(-Ln2Hi), x);
            r = Vector512.MultiplyAddEstimate(fx, Vector512.Create(-Ln2Lo), r);
            var p = Vector512.MultiplyAddEstimate(Vector512.Create(P0), r, Vector512.Create(P1));
            p = Vector512.MultiplyAddEstimate(p, r, Vector512.Create(P2));
            p = Vector512.MultiplyAddEstimate(p, r, Vector512.Create(P3));
            p = Vector512.MultiplyAddEstimate(p, r, Vector512.Create(P4));
            p = Vector512.MultiplyAddEstimate(p, r, Vector512.Create(P5));
            var y = Vector512.MultiplyAddEstimate(p, r * r, r + Vector512<float>.One);
            var pow2n = Vector512.ShiftLeft(Vector512.ConvertToInt32Native(fx) + Vector512.Create(127), 23).AsSingle();
            return y * pow2n;
        }

        private static float* Slab(long floats)
        {
            if (t_slabFloats < floats)
            {
                t_slabRaw = GC.AllocateUninitializedArray<float>(checked((int)floats + 16), pinned: true);
                t_slab = ((nint)Unsafe.AsPointer(ref t_slabRaw[0]) + 63) & ~(nint)63;
                t_slabFloats = (int)floats;
            }
            return (float*)t_slab;
        }
    }
}
