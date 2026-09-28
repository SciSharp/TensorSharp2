// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Models
{
    /// <summary>
    /// One query block of a region-aware DiffusionGemma attention on the pure-C# CPU backend.
    ///
    /// The keys a query may see are addressed in VIRTUAL coordinates <c>[0, LenA + LenB)</c>: index
    /// <c>j &lt; LenA</c> is row <c>j</c> of segment A (the prompt K/V, cached or fresh), index
    /// <c>j &gt;= LenA</c> is row <c>j - LenA</c> of segment B (the canvas K/V). That is exactly the
    /// key order of the unified <c>[prompt | canvas]</c> forward, so the same kernel serves the
    /// unified forward (one segment), the prompt prefill (one segment) and the prompt-KV canvas
    /// decode (cached prompt + fresh canvas) without concatenating K/V. Every query admits one
    /// contiguous interval <c>[Klo, Khi)</c> (see <c>AllowedRange</c>).
    /// </summary>
    internal sealed class DiffusionCpuAttnGroup
    {
        public int QStart, QCount;           // query rows [QStart, QStart+QCount) of the q/o buffers
        public nint KA, VA; public int LenA; // segment A rows, row stride = kvHeads*headDim floats
        public nint KB, VB; public int LenB; // segment B rows (0 = none)
        public int[] Klo, Khi;               // per query (length QCount); null => UniformLo/UniformHi
        public int UniformLo, UniformHi;
    }

    /// <summary>Arithmetic of the CPU attention micro-kernels.</summary>
    internal enum DiffusionAttnKernel
    {
        /// <summary>Bit for bit the legacy per-pair kernel (TensorComputePrimitives.Dot, scalar
        /// MathF.Exp, in-order sum, multiply-then-add value accumulation) - only blocked and parallel.</summary>
        Exact,
        /// <summary><see cref="Vector{T}"/> FMA tiles (AVX2 / AdvSimd) and a vectorized softmax.</summary>
        Fma,
        /// <summary>Vector512 FMA tiles (AVX-512) and a vectorized softmax.</summary>
        Fma512,
    }

    /// <summary>
    /// SIMD kernels for the DiffusionGemma pure-C# CPU forward: fused per-head RMSNorm+NeoX RoPE,
    /// region-aware attention, GELU(gate)*up, the MoE weighted combine and the router dot.
    ///
    /// This model's output is extremely sensitive to last-bit differences: activations are
    /// quantized to Q8 before every matmul (a one-ulp change can move an element across a rounding
    /// step) and each layer picks a top-8 of 128 experts. Measured on diffusiongemma-26B-A4B, a
    /// structured read's label probability moved from 0.43 to 0.30-0.74 when only the attention's
    /// float grouping changed. So everything here reproduces the arithmetic of the Ops chain it
    /// replaced as that chain computed before the Core CPU rewrite (TS_CPU_SIMD_ELEMENTWISE /
    /// TS_CPU_SGEMM / TS_CPU_QGEMM, whose defaults now round differently), which is what makes the new
    /// CPU forward bitwise-checkable against the old one - with the whole rollback recipe
    /// (DiffusionGemmaModel.Cpu.cs), not DIFFUSION_CPU_LEGACY=1 alone. The
    /// attention is a fraction of a percent of a forward at Jev/chat prompt lengths, so the exact
    /// kernel is the default; <c>DIFFUSION_CPU_ATTN_FAST=1</c> selects FMA tiles (Vector512 when the
    /// hardware accelerates it; <c>TS_CPU_DISABLE_AVX512=1</c> keeps them at Vector&lt;T&gt;) and a
    /// vectorized softmax for long prompts, where attention grows quadratically.
    /// </summary>
    internal static unsafe class DiffusionGemmaCpuKernels
    {
        internal static readonly DiffusionAttnKernel DefaultAttention =
            Environment.GetEnvironmentVariable("DIFFUSION_CPU_ATTN_FAST") != "1" ? DiffusionAttnKernel.Exact
            : TensorSharp.Cpu.CpuIsa.Avx512 ? DiffusionAttnKernel.Fma512
            : DiffusionAttnKernel.Fma;

        private const int MicroQ = 4;   // queries per attention micro-block (share each K/V row load)

        [ThreadStatic] private static float[] t_scores;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> Fmadd(Vector<float> a, Vector<float> b, Vector<float> c)
            => Fma.IsSupported || AdvSimd.IsSupported ? Vector.FusedMultiplyAdd(a, b, c) : a * b + c;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> Ld(float* p) => Unsafe.ReadUnaligned<Vector<float>>(p);

        // ------------------------------------------------------------------------------------
        //  Per-head RMSNorm (+ optional NeoX RoPE), one row.
        //  Bitwise identical to the pre-SIMD Ops.RMSNorm - the one TS_CPU_SIMD_ELEMENTWISE=0 still
        //  runs: a single-accumulator Vector<float> sum of squares and the (x*invRms)*gamma product
        //  order (the default CpuKernels.RmsNormRow sums in two wider accumulators, so its bits
        //  differ) - followed by ApplyNeoXRoPERaw's (x0*c - x1*s, x0*s + x1*c), with no FMA
        //  contraction, so one pass replaces three.
        // ------------------------------------------------------------------------------------
        internal static void HeadNormRopeRow(float* src, float* dst, int heads, int hd, float* weight, float eps,
            float* cos, float* sin)
        {
            int vLen = Vector<float>.Count;
            float colsAsFloat = hd;
            int half = hd / 2;
            for (int h = 0; h < heads; h++)
            {
                float* x = src + (long)h * hd;
                float* y = dst + (long)h * hd;

                Vector<float> acc = Vector<float>.Zero;
                int i = 0;
                for (; i <= hd - vLen; i += vLen)
                {
                    Vector<float> vx = Ld(x + i);
                    acc += vx * vx;
                }
                float sqSum = Vector.Sum(acc);
                for (; i < hd; i++) sqSum += x[i] * x[i];
                float invRms = 1.0f / MathF.Sqrt(sqSum / colsAsFloat + eps);
                var vInv = new Vector<float>(invRms);

                if (cos == null)
                {
                    i = 0;
                    if (weight == null)
                    {
                        for (; i <= hd - vLen; i += vLen)
                            Unsafe.WriteUnaligned(y + i, Ld(x + i) * vInv);
                        for (; i < hd; i++) y[i] = x[i] * invRms;
                    }
                    else
                    {
                        for (; i <= hd - vLen; i += vLen)
                            Unsafe.WriteUnaligned(y + i, Ld(x + i) * vInv * Ld(weight + i));
                        for (; i < hd; i++) y[i] = weight[i] * (x[i] * invRms);
                    }
                    continue;
                }

                // norm + rotate the (j, j+half) pairs in registers; x and y may alias (each
                // iteration reads its two chunks before writing them).
                int j = 0;
                for (; j <= half - vLen; j += vLen)
                {
                    Vector<float> x0 = Ld(x + j) * vInv;
                    Vector<float> x1 = Ld(x + j + half) * vInv;
                    if (weight != null)
                    {
                        x0 *= Ld(weight + j);
                        x1 *= Ld(weight + j + half);
                    }
                    Vector<float> c = Ld(cos + j);
                    Vector<float> s = Ld(sin + j);
                    Unsafe.WriteUnaligned(y + j, x0 * c - x1 * s);
                    Unsafe.WriteUnaligned(y + j + half, x0 * s + x1 * c);
                }
                for (; j < half; j++)
                {
                    float x0 = weight != null ? weight[j] * (x[j] * invRms) : x[j] * invRms;
                    float x1 = weight != null ? weight[j + half] * (x[j + half] * invRms) : x[j + half] * invRms;
                    float c = cos[j], s = sin[j];
                    y[j] = x0 * c - x1 * s;
                    y[j + half] = x0 * s + x1 * c;
                }
            }
        }

        // ------------------------------------------------------------------------------------
        //  GELU(gate) * up, bit for bit what Ops.GELUMul computed on the CPU backend before the SIMD
        //  rewrite and still computes under TS_CPU_SIMD_ELEMENTWISE=0 (TensorApplyCPU: the tanh
        //  approximation with a double-precision Math.Tanh); the default CpuKernels.GeluMul uses a
        //  vectorized sigmoid form and differs by a few ulp. Kept scalar on purpose: a vectorized
        //  float tanh is a few ulp off (see the class remarks). Run over the pool it costs well
        //  under 1% of a layer; the reference ran on one thread.
        // ------------------------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float Gelu(float x)
            => 0.5f * x * (1.0f + (float)Math.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));

        internal static void GeluMulRow(float* gate, float* up, float* dst, int n)
        {
            for (int i = 0; i < n; i++)
                dst[i] = Gelu(gate[i]) * up[i];
        }

        // ------------------------------------------------------------------------------------
        //  MoE weighted combine for one token: dst = sum_k w_k * (s_k * y_k) in the caller's order,
        //  starting from zero - the exact accumulation of the per-expert reference loop (which
        //  first scaled each expert's output by its ffn_down_exps.scale, then did dst += w*src in
        //  ascending expert order). Pass scales == null when every scale is 1.
        // ------------------------------------------------------------------------------------
        internal static void WeightedRowSum(float* dst, int n, float** rows, float* weights, float* scales, int count)
        {
            int vLen = Vector<float>.Count;
            int i = 0;
            for (; i <= n - vLen; i += vLen)
            {
                Vector<float> acc = Vector<float>.Zero;
                for (int k = 0; k < count; k++)
                {
                    Vector<float> y = Ld(rows[k] + i);
                    if (scales != null) y *= new Vector<float>(scales[k]);
                    acc += new Vector<float>(weights[k]) * y;
                }
                Unsafe.WriteUnaligned(dst + i, acc);
            }
            for (; i < n; i++)
            {
                float acc = 0f;
                for (int k = 0; k < count; k++)
                {
                    float y = rows[k][i];
                    if (scales != null) y *= scales[k];
                    acc += weights[k] * y;
                }
                dst[i] = acc;
            }
        }

        /// <summary>Router dot with the arithmetic of the managed F32 GEMM's 4x4 row-column kernel
        /// (TensorSharp.Cpu DotContiguousFourByFour: one Vector&lt;float&gt; accumulator of
        /// multiply-then-add, lane sum, scalar tail), which scores every token of a 4-row block in the
        /// legacy linear (the matmul TS_CPU_SGEMM=0 restores; the default packed SGEMM sums in another
        /// order). Unlike that GEMM it does not switch kernels for a trailing partial block, so
        /// a token's scores never depend on how many rows share the call - the prompt-KV decode needs
        /// that to route each canvas token exactly as the unified forward does.</summary>
        internal static float RouterDot(float* a, float* b, int n)
        {
            int vLen = Vector<float>.Count;
            Vector<float> acc = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                acc += Ld(a + i) * Ld(b + i);
            float sum = Vector.Sum(acc);
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }

        // ------------------------------------------------------------------------------------
        //  Region-aware attention (scale 1.0: the learnable Q/K norms absorb 1/sqrt(d)).
        //
        //  Work item = (group, query head, block of QueryBlock queries), spread over the persistent
        //  CPU pool; the legacy kernel ran one ThreadPool task per head (16 tasks). Inside an item
        //  the queries go MicroQ=4 at a time over the UNION of their key intervals (adjacent causal
        //  queries differ by a key or two, canvas queries share one interval), so every K and V row
        //  read feeds four queries. Keys outside a query's own interval get a weight of exactly 0,
        //  which adds nothing to its outputs, so a query's result does not depend on which queries
        //  share its micro-block or on where its keys are stored - the property the prompt-KV decode
        //  relies on.
        // ------------------------------------------------------------------------------------
        internal const int QueryBlock = 16;

        /// <summary>Number of pool work items <see cref="AttendItem"/> expects for these groups.</summary>
        internal static int CountAttendItems(DiffusionCpuAttnGroup[] groups, int qHeads, out int[] itemStart)
        {
            itemStart = new int[groups.Length + 1];
            int total = 0;
            for (int g = 0; g < groups.Length; g++)
            {
                itemStart[g] = total;
                total += qHeads * ((groups[g].QCount + QueryBlock - 1) / QueryBlock);
            }
            itemStart[groups.Length] = total;
            return total;
        }

        internal static void AttendItem(int item, DiffusionCpuAttnGroup[] groups, int[] itemStart,
            float* q, float* o, int qHeads, int kvHeads, int hd, DiffusionAttnKernel kernel)
        {
            int g = 0;
            while (itemStart[g + 1] <= item) g++;
            DiffusionCpuAttnGroup grp = groups[g];
            int local = item - itemStart[g];
            int blocks = (grp.QCount + QueryBlock - 1) / QueryBlock;
            int h = local / blocks;
            int qb = local - h * blocks;

            int groupSize = qHeads / kvHeads;
            int kvHead = h / groupSize;
            long qStride = (long)qHeads * hd;
            long kvStride = (long)kvHeads * hd;
            float* kA = (float*)grp.KA + (long)kvHead * hd;
            float* vA = (float*)grp.VA + (long)kvHead * hd;
            float* kB = (float*)grp.KB + (long)kvHead * hd;
            float* vB = (float*)grp.VB + (long)kvHead * hd;
            int lenA = grp.LenA;

            int t0 = qb * QueryBlock;
            int t1 = Math.Min(grp.QCount, t0 + QueryBlock);
            var qp = stackalloc float*[MicroQ];
            var op = stackalloc float*[MicroQ];
            var lo4 = stackalloc int[MicroQ];
            var hi4 = stackalloc int[MicroQ];
            for (int m = t0; m < t1; m += MicroQ)
            {
                int n = Math.Min(MicroQ, t1 - m);
                int lo = int.MaxValue, hi = int.MinValue;
                for (int i = 0; i < MicroQ; i++)
                {
                    int li = m + Math.Min(i, n - 1);   // pad a short block by repeating its last query
                    long row = grp.QStart + li;
                    qp[i] = q + row * qStride + (long)h * hd;
                    op[i] = o + row * qStride + (long)h * hd;
                    int klo = grp.Klo != null ? grp.Klo[li] : grp.UniformLo;
                    int khi = grp.Khi != null ? grp.Khi[li] : grp.UniformHi;
                    if (khi <= klo) khi = klo + 1;   // AllowedRange guard, as the reference kernel
                    lo4[i] = klo; hi4[i] = khi;
                    if (klo < lo) lo = klo;
                    if (khi > hi) hi = khi;
                }
                int L = hi - lo;
                float[] scratch = t_scores;
                if (scratch == null || scratch.Length < MicroQ * L)
                    t_scores = scratch = new float[Math.Max(MicroQ * L, 4096)];
                fixed (float* s = scratch)
                {
                    Scores(qp, s, L, lo, lenA, kA, kB, kvStride, hd, kernel);
                    for (int i = 0; i < MicroQ; i++)
                    {
                        float* row = s + (long)i * L;
                        if (i >= n) { new Span<float>(row, L).Clear(); continue; }
                        SoftmaxInterval(row, L, lo4[i] - lo, hi4[i] - lo, kernel != DiffusionAttnKernel.Exact);
                    }
                    WeightedValues(s, L, lo, lenA, vA, vB, kvStride, hd, op, n, kernel);
                }
            }
        }

        private static float* RowPtr(int j, int lenA, float* a, float* b, long stride)
            => j < lenA ? a + (long)j * stride : b + (long)(j - lenA) * stride;

        private static void Scores(float** qp, float* s, int L, int lo, int lenA, float* kA, float* kB, long kvStride, int hd,
            DiffusionAttnKernel kernel)
        {
            var r = stackalloc float[16];
            if (kernel == DiffusionAttnKernel.Exact)
            {
                for (int j = 0; j < L; j++)
                {
                    Dot4x1Exact(qp, RowPtr(lo + j, lenA, kA, kB, kvStride), hd, r);
                    s[j] = r[0]; s[L + j] = r[1]; s[2 * L + j] = r[2]; s[3 * L + j] = r[3];
                }
                return;
            }
            bool v512 = kernel == DiffusionAttnKernel.Fma512;
            int jj = 0;
            var kp = stackalloc float*[4];
            for (; jj + 4 <= L; jj += 4)
            {
                for (int x = 0; x < 4; x++) kp[x] = RowPtr(lo + jj + x, lenA, kA, kB, kvStride);
                Dot4x4(qp, kp, hd, r, v512);
                for (int i = 0; i < MicroQ; i++)
                {
                    float* dst = s + (long)i * L + jj;
                    dst[0] = r[i * 4]; dst[1] = r[i * 4 + 1]; dst[2] = r[i * 4 + 2]; dst[3] = r[i * 4 + 3];
                }
            }
            for (; jj < L; jj++)
            {
                float* k = RowPtr(lo + jj, lenA, kA, kB, kvStride);
                kp[0] = kp[1] = kp[2] = kp[3] = k;
                Dot4x4(qp, kp, hd, r, v512);
                for (int i = 0; i < MicroQ; i++) s[(long)i * L + jj] = r[i * 4];
            }
        }

        /// <summary>r[i] = dot(q_i, k) for four queries against one key, each with exactly
        /// TensorComputePrimitives.Dot's arithmetic (two interleaved Vector&lt;float&gt; accumulators of
        /// multiply-then-add, their sum, one more vector step, lane sum, scalar tail) - the legacy
        /// kernel's VecDot - while loading every K chunk once for all four.</summary>
        internal static void Dot4x1Exact(float** qp, float* k, int hd, float* r)
        {
            int vLen = Vector<float>.Count;
            float* q0 = qp[0], q1 = qp[1], q2 = qp[2], q3 = qp[3];
            Vector<float> a0 = Vector<float>.Zero, b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
            int i = 0;
            for (; i <= hd - vLen * 2; i += vLen * 2)
            {
                Vector<float> k0 = Ld(k + i), k1 = Ld(k + i + vLen);
                a0 += Ld(q0 + i) * k0; b0 += Ld(q0 + i + vLen) * k1;
                a1 += Ld(q1 + i) * k0; b1 += Ld(q1 + i + vLen) * k1;
                a2 += Ld(q2 + i) * k0; b2 += Ld(q2 + i + vLen) * k1;
                a3 += Ld(q3 + i) * k0; b3 += Ld(q3 + i + vLen) * k1;
            }
            Vector<float> s0 = a0 + b0, s1 = a1 + b1, s2 = a2 + b2, s3 = a3 + b3;
            for (; i <= hd - vLen; i += vLen)
            {
                Vector<float> kk = Ld(k + i);
                s0 += Ld(q0 + i) * kk; s1 += Ld(q1 + i) * kk; s2 += Ld(q2 + i) * kk; s3 += Ld(q3 + i) * kk;
            }
            float r0 = Vector.Sum(s0), r1 = Vector.Sum(s1), r2 = Vector.Sum(s2), r3 = Vector.Sum(s3);
            for (; i < hd; i++)
            {
                float kv = k[i];
                r0 += q0[i] * kv; r1 += q1[i] * kv; r2 += q2[i] * kv; r3 += q3[i] * kv;
            }
            r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
        }

        /// <summary>FMA tile: r[i*4+x] = dot(q_i, k_x) for 4 queries x 4 keys.</summary>
        internal static void Dot4x4(float** qp, float** kp, int hd, float* r, bool v512)
        {
            int i = 0;
            if (v512 && hd >= 16)
            {
                var a00 = Vector512<float>.Zero; var a01 = a00; var a02 = a00; var a03 = a00;
                var a10 = a00; var a11 = a00; var a12 = a00; var a13 = a00;
                var a20 = a00; var a21 = a00; var a22 = a00; var a23 = a00;
                var a30 = a00; var a31 = a00; var a32 = a00; var a33 = a00;
                float* q0 = qp[0], q1 = qp[1], q2 = qp[2], q3 = qp[3];
                float* k0 = kp[0], k1 = kp[1], k2 = kp[2], k3 = kp[3];
                for (; i <= hd - 16; i += 16)
                {
                    var vk0 = Vector512.Load(k0 + i); var vk1 = Vector512.Load(k1 + i);
                    var vk2 = Vector512.Load(k2 + i); var vk3 = Vector512.Load(k3 + i);
                    var vq = Vector512.Load(q0 + i);
                    a00 = Vector512.FusedMultiplyAdd(vq, vk0, a00); a01 = Vector512.FusedMultiplyAdd(vq, vk1, a01);
                    a02 = Vector512.FusedMultiplyAdd(vq, vk2, a02); a03 = Vector512.FusedMultiplyAdd(vq, vk3, a03);
                    vq = Vector512.Load(q1 + i);
                    a10 = Vector512.FusedMultiplyAdd(vq, vk0, a10); a11 = Vector512.FusedMultiplyAdd(vq, vk1, a11);
                    a12 = Vector512.FusedMultiplyAdd(vq, vk2, a12); a13 = Vector512.FusedMultiplyAdd(vq, vk3, a13);
                    vq = Vector512.Load(q2 + i);
                    a20 = Vector512.FusedMultiplyAdd(vq, vk0, a20); a21 = Vector512.FusedMultiplyAdd(vq, vk1, a21);
                    a22 = Vector512.FusedMultiplyAdd(vq, vk2, a22); a23 = Vector512.FusedMultiplyAdd(vq, vk3, a23);
                    vq = Vector512.Load(q3 + i);
                    a30 = Vector512.FusedMultiplyAdd(vq, vk0, a30); a31 = Vector512.FusedMultiplyAdd(vq, vk1, a31);
                    a32 = Vector512.FusedMultiplyAdd(vq, vk2, a32); a33 = Vector512.FusedMultiplyAdd(vq, vk3, a33);
                }
                r[0] = Vector512.Sum(a00); r[1] = Vector512.Sum(a01); r[2] = Vector512.Sum(a02); r[3] = Vector512.Sum(a03);
                r[4] = Vector512.Sum(a10); r[5] = Vector512.Sum(a11); r[6] = Vector512.Sum(a12); r[7] = Vector512.Sum(a13);
                r[8] = Vector512.Sum(a20); r[9] = Vector512.Sum(a21); r[10] = Vector512.Sum(a22); r[11] = Vector512.Sum(a23);
                r[12] = Vector512.Sum(a30); r[13] = Vector512.Sum(a31); r[14] = Vector512.Sum(a32); r[15] = Vector512.Sum(a33);
            }
            else
            {
                // Vector<float>: two 4x2 halves keep the tile inside 16 ymm registers.
                int vLen = Vector<float>.Count;
                for (int half = 0; half < 4; half += 2)
                {
                    var a0 = Vector<float>.Zero; var b0 = a0; var a1 = a0; var b1 = a0;
                    var a2 = a0; var b2 = a0; var a3 = a0; var b3 = a0;
                    float* k0 = kp[half], k1 = kp[half + 1];
                    int ii = 0;
                    for (; ii <= hd - vLen; ii += vLen)
                    {
                        var vk0 = Ld(k0 + ii);
                        var vk1 = Ld(k1 + ii);
                        var vq = Ld(qp[0] + ii);
                        a0 = Fmadd(vq, vk0, a0); b0 = Fmadd(vq, vk1, b0);
                        vq = Ld(qp[1] + ii);
                        a1 = Fmadd(vq, vk0, a1); b1 = Fmadd(vq, vk1, b1);
                        vq = Ld(qp[2] + ii);
                        a2 = Fmadd(vq, vk0, a2); b2 = Fmadd(vq, vk1, b2);
                        vq = Ld(qp[3] + ii);
                        a3 = Fmadd(vq, vk0, a3); b3 = Fmadd(vq, vk1, b3);
                    }
                    r[0 * 4 + half] = Vector.Sum(a0); r[0 * 4 + half + 1] = Vector.Sum(b0);
                    r[1 * 4 + half] = Vector.Sum(a1); r[1 * 4 + half + 1] = Vector.Sum(b1);
                    r[2 * 4 + half] = Vector.Sum(a2); r[2 * 4 + half + 1] = Vector.Sum(b2);
                    r[3 * 4 + half] = Vector.Sum(a3); r[3 * 4 + half + 1] = Vector.Sum(b3);
                    i = ii;
                }
            }
            for (; i < hd; i++)
                for (int qi = 0; qi < 4; qi++)
                    for (int x = 0; x < 4; x++)
                        r[qi * 4 + x] += qp[qi][i] * kp[x][i];
        }

        /// <summary>In place: row[j] = softmax over [a, b) and exactly 0 elsewhere. The exact kernel
        /// keeps the legacy max, scalar MathF.Exp(s - max), in-order sum and (e * 1/sum); the FMA
        /// kernels use the vectorized TensorPrimitives max/exp/sum (a few ulp and a re-associated sum),
        /// since at long prompts the ~P^2/2 scalar exps per head become the attention's largest cost.</summary>
        private static void SoftmaxInterval(float* row, int L, int a, int b, bool vectorized)
        {
            for (int j = 0; j < a; j++) row[j] = 0f;
            for (int j = b; j < L; j++) row[j] = 0f;
            if (vectorized)
            {
                // Max, the subtraction, exp and the scaling are element-wise (their results depend
                // only on the values and the interval length). The sum is not: TensorPrimitives.Sum
                // groups its loads by buffer alignment, which would make a query's weights depend
                // on where its interval sits in the micro-block's scratch row, i.e. on which queries
                // share the block - and the prompt-KV decode must reproduce the unified forward.
                var span = new Span<float>(row + a, b - a);
                float vmax = TensorPrimitives.Max(span);
                TensorPrimitives.Subtract(span, vmax, span);
                TensorPrimitives.Exp(span, span);
                float vsum = SumFromStart(row + a, b - a);
                TensorPrimitives.Multiply(span, vsum > 0f ? 1f / vsum : 0f, span);
                return;
            }
            float max = float.NegativeInfinity;
            for (int j = a; j < b; j++) if (row[j] > max) max = row[j];
            float sum = 0f;
            for (int j = a; j < b; j++)
            {
                float e = MathF.Exp(row[j] - max);
                row[j] = e;
                sum += e;
            }
            float inv = sum > 0f ? 1f / sum : 0f;
            for (int j = a; j < b; j++) row[j] *= inv;
        }

        /// <summary>Vector sum in a fixed order from <paramref name="p"/> itself (one accumulator per
        /// lane, lane sum, scalar tail), whatever the address.</summary>
        private static float SumFromStart(float* p, int n)
        {
            int vLen = Vector<float>.Count;
            Vector<float> acc = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen; i += vLen) acc += Ld(p + i);
            float sum = Vector.Sum(acc);
            for (; i < n; i++) sum += p[i];
            return sum;
        }

        /// <summary>o_i = sum_j p[i][j] * V_j for the (up to) four queries of a micro-block, keys in order.
        /// The exact kernel accumulates v * w with a separate multiply and add, as the legacy
        /// VecScaleAdd did; the fast ones fuse it.</summary>
        private static void WeightedValues(float* p, int L, int lo, int lenA, float* vA, float* vB, long kvStride,
            int hd, float** op, int n, DiffusionAttnKernel kernel)
        {
            // Split the key sweep at the segment boundary so each piece walks one contiguous stride.
            int splitA = Math.Clamp(lenA - lo, 0, L);
            int c = 0;
            if (kernel == DiffusionAttnKernel.Fma512)
            {
                for (; c <= hd - 64; c += 64)
                {
                    var o00 = Vector512<float>.Zero; var o01 = o00; var o02 = o00; var o03 = o00;
                    var o10 = o00; var o11 = o00; var o12 = o00; var o13 = o00;
                    var o20 = o00; var o21 = o00; var o22 = o00; var o23 = o00;
                    var o30 = o00; var o31 = o00; var o32 = o00; var o33 = o00;
                    for (int piece = 0; piece < 2; piece++)
                    {
                        int j0 = piece == 0 ? 0 : splitA;
                        int j1 = piece == 0 ? splitA : L;
                        float* vbase = piece == 0 ? vA + (long)lo * kvStride : vB + (long)(lo + splitA - lenA) * kvStride;
                        for (int j = j0; j < j1; j++)
                        {
                            float* v = vbase + (long)(j - j0) * kvStride + c;
                            var v0 = Vector512.Load(v); var v1 = Vector512.Load(v + 16);
                            var v2 = Vector512.Load(v + 32); var v3 = Vector512.Load(v + 48);
                            var w = Vector512.Create(p[j]);
                            o00 = Vector512.FusedMultiplyAdd(w, v0, o00); o01 = Vector512.FusedMultiplyAdd(w, v1, o01);
                            o02 = Vector512.FusedMultiplyAdd(w, v2, o02); o03 = Vector512.FusedMultiplyAdd(w, v3, o03);
                            w = Vector512.Create(p[L + j]);
                            o10 = Vector512.FusedMultiplyAdd(w, v0, o10); o11 = Vector512.FusedMultiplyAdd(w, v1, o11);
                            o12 = Vector512.FusedMultiplyAdd(w, v2, o12); o13 = Vector512.FusedMultiplyAdd(w, v3, o13);
                            w = Vector512.Create(p[2 * L + j]);
                            o20 = Vector512.FusedMultiplyAdd(w, v0, o20); o21 = Vector512.FusedMultiplyAdd(w, v1, o21);
                            o22 = Vector512.FusedMultiplyAdd(w, v2, o22); o23 = Vector512.FusedMultiplyAdd(w, v3, o23);
                            w = Vector512.Create(p[3 * L + j]);
                            o30 = Vector512.FusedMultiplyAdd(w, v0, o30); o31 = Vector512.FusedMultiplyAdd(w, v1, o31);
                            o32 = Vector512.FusedMultiplyAdd(w, v2, o32); o33 = Vector512.FusedMultiplyAdd(w, v3, o33);
                        }
                    }
                    Store4(op[0] + c, o00, o01, o02, o03);
                    if (n > 1) Store4(op[1] + c, o10, o11, o12, o13);
                    if (n > 2) Store4(op[2] + c, o20, o21, o22, o23);
                    if (n > 3) Store4(op[3] + c, o30, o31, o32, o33);
                }
            }
            else
            {
                bool fused = kernel == DiffusionAttnKernel.Fma;
                int vLen = Vector<float>.Count;
                int chunk = 2 * vLen;
                for (; c <= hd - chunk; c += chunk)
                {
                    var o00 = Vector<float>.Zero; var o01 = o00; var o10 = o00; var o11 = o00;
                    var o20 = o00; var o21 = o00; var o30 = o00; var o31 = o00;
                    for (int piece = 0; piece < 2; piece++)
                    {
                        int j0 = piece == 0 ? 0 : splitA;
                        int j1 = piece == 0 ? splitA : L;
                        float* vbase = piece == 0 ? vA + (long)lo * kvStride : vB + (long)(lo + splitA - lenA) * kvStride;
                        if (fused)
                        {
                            for (int j = j0; j < j1; j++)
                            {
                                float* v = vbase + (long)(j - j0) * kvStride + c;
                                Vector<float> v0 = Ld(v), v1 = Ld(v + vLen);
                                var w = new Vector<float>(p[j]);
                                o00 = Fmadd(w, v0, o00); o01 = Fmadd(w, v1, o01);
                                w = new Vector<float>(p[L + j]);
                                o10 = Fmadd(w, v0, o10); o11 = Fmadd(w, v1, o11);
                                w = new Vector<float>(p[2 * L + j]);
                                o20 = Fmadd(w, v0, o20); o21 = Fmadd(w, v1, o21);
                                w = new Vector<float>(p[3 * L + j]);
                                o30 = Fmadd(w, v0, o30); o31 = Fmadd(w, v1, o31);
                            }
                        }
                        else
                        {
                            for (int j = j0; j < j1; j++)
                            {
                                float* v = vbase + (long)(j - j0) * kvStride + c;
                                Vector<float> v0 = Ld(v), v1 = Ld(v + vLen);
                                var w = new Vector<float>(p[j]);
                                o00 += v0 * w; o01 += v1 * w;
                                w = new Vector<float>(p[L + j]);
                                o10 += v0 * w; o11 += v1 * w;
                                w = new Vector<float>(p[2 * L + j]);
                                o20 += v0 * w; o21 += v1 * w;
                                w = new Vector<float>(p[3 * L + j]);
                                o30 += v0 * w; o31 += v1 * w;
                            }
                        }
                    }
                    Unsafe.WriteUnaligned(op[0] + c, o00); Unsafe.WriteUnaligned(op[0] + c + vLen, o01);
                    if (n > 1) { Unsafe.WriteUnaligned(op[1] + c, o10); Unsafe.WriteUnaligned(op[1] + c + vLen, o11); }
                    if (n > 2) { Unsafe.WriteUnaligned(op[2] + c, o20); Unsafe.WriteUnaligned(op[2] + c + vLen, o21); }
                    if (n > 3) { Unsafe.WriteUnaligned(op[3] + c, o30); Unsafe.WriteUnaligned(op[3] + c + vLen, o31); }
                }
                // A single-vector step keeps the exact kernel on the legacy VecScaleAdd's vector lanes
                // for head dims that are an odd number of vectors.
                for (; c <= hd - vLen; c += vLen)
                {
                    for (int i = 0; i < n; i++)
                    {
                        Vector<float> acc = Vector<float>.Zero;
                        for (int j = 0; j < L; j++)
                        {
                            Vector<float> v = Ld(RowPtr(lo + j, lenA, vA, vB, kvStride) + c);
                            var w = new Vector<float>(p[(long)i * L + j]);
                            acc = fused ? Fmadd(w, v, acc) : acc + v * w;
                        }
                        Unsafe.WriteUnaligned(op[i] + c, acc);
                    }
                }
            }
            // scalar tail (head dims that are not a multiple of the vector width)
            for (; c < hd; c++)
            {
                for (int i = 0; i < n; i++)
                {
                    float acc = 0f;
                    for (int j = 0; j < L; j++)
                        acc += p[(long)i * L + j] * RowPtr(lo + j, lenA, vA, vB, kvStride)[c];
                    op[i][c] = acc;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store4(float* d, Vector512<float> a, Vector512<float> b, Vector512<float> c, Vector512<float> e)
        {
            a.Store(d); b.Store(d + 16); c.Store(d + 32); e.Store(d + 48);
        }
    }
}
