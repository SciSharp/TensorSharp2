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
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Pointer-level SIMD kernels behind the contiguous fast paths of the CPU elementwise,
    /// normalization and softmax ops (TensorApplyCPU), usable directly by model code.
    ///
    /// Every kernel is Vector512 where <see cref="CpuIsa.Avx512"/> allows it (TS_CPU_DISABLE_AVX512=1
    /// pins Vector256), Vector256 otherwise, with a scalar tail; exp-based activations use the BCL's
    /// vectorized Vector512/256/128.Exp and tanh uses TensorPrimitives (a few ULP from MathF).
    /// Work is split on <see cref="CpuParallel"/> by element count, never by thread count: a
    /// [70, 2816] residual add stays on the calling thread, a [256, 262144] logit softcap is
    /// spread over every worker. Formulas and operation order match the scalar code they
    /// replace, so apart from the transcendental approximations results are bit-identical.
    ///
    /// TS_CPU_SIMD_ELEMENTWISE=0 keeps TensorApplyCPU on its previous loops for A/B runs.
    /// </summary>
    internal static unsafe class CpuKernels
    {
        /// <summary>False when TS_CPU_SIMD_ELEMENTWISE=0.</summary>
        public static bool Enabled { get; } = Environment.GetEnvironmentVariable("TS_CPU_SIMD_ELEMENTWISE") != "0";

        /// <summary>Use 512-bit vectors (tests pin it off to cover the 256-bit path).</summary>
        internal static bool Use512 { get; set; } = CpuIsa.Avx512;

        // Elements per parallel block. Streaming ops move ~12 bytes/element, so 64K elements
        // is ~0.75 MB of traffic (tens of microseconds) against a few microseconds to fork;
        // transcendental ops cost ~10x more per element and split earlier.
        private const long StreamChunk = 64 * 1024;
        private const long TranscendentalChunk = 8 * 1024;
        private const long RowChunkElements = 16 * 1024;

        // ------------------------------------------------------------------------------------
        //  Binary ops
        // ------------------------------------------------------------------------------------

        public enum BinaryOp { Add, Sub, Mul, Div }

        internal interface IBinary
        {
            static abstract float Apply(float x, float y);
            static abstract Vector128<float> Apply(Vector128<float> x, Vector128<float> y);
            static abstract Vector256<float> Apply(Vector256<float> x, Vector256<float> y);
            static abstract Vector512<float> Apply(Vector512<float> x, Vector512<float> y);
        }

        internal struct AddOp : IBinary
        {
            public static float Apply(float x, float y) => x + y;
            public static Vector128<float> Apply(Vector128<float> x, Vector128<float> y) => x + y;
            public static Vector256<float> Apply(Vector256<float> x, Vector256<float> y) => x + y;
            public static Vector512<float> Apply(Vector512<float> x, Vector512<float> y) => x + y;
        }

        internal struct SubOp : IBinary
        {
            public static float Apply(float x, float y) => x - y;
            public static Vector128<float> Apply(Vector128<float> x, Vector128<float> y) => x - y;
            public static Vector256<float> Apply(Vector256<float> x, Vector256<float> y) => x - y;
            public static Vector512<float> Apply(Vector512<float> x, Vector512<float> y) => x - y;
        }

        internal struct MulOp : IBinary
        {
            public static float Apply(float x, float y) => x * y;
            public static Vector128<float> Apply(Vector128<float> x, Vector128<float> y) => x * y;
            public static Vector256<float> Apply(Vector256<float> x, Vector256<float> y) => x * y;
            public static Vector512<float> Apply(Vector512<float> x, Vector512<float> y) => x * y;
        }

        internal struct DivOp : IBinary
        {
            public static float Apply(float x, float y) => x / y;
            public static Vector128<float> Apply(Vector128<float> x, Vector128<float> y) => x / y;
            public static Vector256<float> Apply(Vector256<float> x, Vector256<float> y) => x / y;
            public static Vector512<float> Apply(Vector512<float> x, Vector512<float> y) => x / y;
        }

        /// <summary>r[i] = x[i] op y[i]. r may alias x or y.</summary>
        public static void Binary(BinaryOp op, float* r, float* x, float* y, long n)
        {
            switch (op)
            {
                case BinaryOp.Add: Split(n, StreamChunk, (s, e) => BinaryLoop<AddOp>(r + s, x + s, y + s, e - s)); break;
                case BinaryOp.Sub: Split(n, StreamChunk, (s, e) => BinaryLoop<SubOp>(r + s, x + s, y + s, e - s)); break;
                case BinaryOp.Mul: Split(n, StreamChunk, (s, e) => BinaryLoop<MulOp>(r + s, x + s, y + s, e - s)); break;
                default: Split(n, StreamChunk, (s, e) => BinaryLoop<DivOp>(r + s, x + s, y + s, e - s)); break;
            }
        }

        /// <summary>r[i] = x[i] op s (scalarOnLeft: r[i] = s op x[i]). r may alias x.</summary>
        public static void BinaryScalar(BinaryOp op, float* r, float* x, float s, long n, bool scalarOnLeft = false)
        {
            switch (op)
            {
                case BinaryOp.Add: Split(n, StreamChunk, (a, e) => ScalarLoop<AddOp>(r + a, x + a, s, e - a, scalarOnLeft)); break;
                case BinaryOp.Sub: Split(n, StreamChunk, (a, e) => ScalarLoop<SubOp>(r + a, x + a, s, e - a, scalarOnLeft)); break;
                case BinaryOp.Mul: Split(n, StreamChunk, (a, e) => ScalarLoop<MulOp>(r + a, x + a, s, e - a, scalarOnLeft)); break;
                default: Split(n, StreamChunk, (a, e) => ScalarLoop<DivOp>(r + a, x + a, s, e - a, scalarOnLeft)); break;
            }
        }

        /// <summary>r[i, c] = x[i, c] op row[c] over contiguous [rows, cols]. r may alias x.</summary>
        public static void BinaryRowBroadcast(BinaryOp op, float* r, float* x, float* row, long rows, int cols)
        {
            long minRows = Math.Max(1, StreamChunk / Math.Max(1, cols));
            switch (op)
            {
                case BinaryOp.Add: CpuParallel.ForRange(rows, minRows, (s, e) => RowLoop<AddOp>(r, x, row, s, e, cols)); break;
                case BinaryOp.Sub: CpuParallel.ForRange(rows, minRows, (s, e) => RowLoop<SubOp>(r, x, row, s, e, cols)); break;
                case BinaryOp.Mul: CpuParallel.ForRange(rows, minRows, (s, e) => RowLoop<MulOp>(r, x, row, s, e, cols)); break;
                default: CpuParallel.ForRange(rows, minRows, (s, e) => RowLoop<DivOp>(r, x, row, s, e, cols)); break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void RowLoop<TOp>(float* r, float* x, float* row, long start, long end, int cols) where TOp : struct, IBinary
        {
            for (long i = start; i < end; i++)
                BinaryLoop<TOp>(r + i * cols, x + i * cols, row, cols);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void BinaryLoop<TOp>(float* r, float* x, float* y, long n) where TOp : struct, IBinary
        {
            long i = 0;
            if (Use512)
            {
                for (; i + 32 <= n; i += 32)
                {
                    Vector512<float> a0 = TOp.Apply(Vector512.Load(x + i), Vector512.Load(y + i));
                    Vector512<float> a1 = TOp.Apply(Vector512.Load(x + i + 16), Vector512.Load(y + i + 16));
                    Vector512.Store(a0, r + i);
                    Vector512.Store(a1, r + i + 16);
                }
                for (; i + 16 <= n; i += 16)
                    Vector512.Store(TOp.Apply(Vector512.Load(x + i), Vector512.Load(y + i)), r + i);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 8 <= n; i += 8)
                    Vector256.Store(TOp.Apply(Vector256.Load(x + i), Vector256.Load(y + i)), r + i);
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                for (; i + 4 <= n; i += 4)
                    Vector128.Store(TOp.Apply(Vector128.Load(x + i), Vector128.Load(y + i)), r + i);
            }
            for (; i < n; i++)
                r[i] = TOp.Apply(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void ScalarLoop<TOp>(float* r, float* x, float s, long n, bool scalarOnLeft) where TOp : struct, IBinary
        {
            long i = 0;
            if (Use512)
            {
                Vector512<float> vs = Vector512.Create(s);
                if (scalarOnLeft)
                {
                    for (; i + 16 <= n; i += 16) Vector512.Store(TOp.Apply(vs, Vector512.Load(x + i)), r + i);
                }
                else
                {
                    for (; i + 16 <= n; i += 16) Vector512.Store(TOp.Apply(Vector512.Load(x + i), vs), r + i);
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vs = Vector256.Create(s);
                if (scalarOnLeft)
                {
                    for (; i + 8 <= n; i += 8) Vector256.Store(TOp.Apply(vs, Vector256.Load(x + i)), r + i);
                }
                else
                {
                    for (; i + 8 <= n; i += 8) Vector256.Store(TOp.Apply(Vector256.Load(x + i), vs), r + i);
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vs = Vector128.Create(s);
                if (scalarOnLeft)
                {
                    for (; i + 4 <= n; i += 4) Vector128.Store(TOp.Apply(vs, Vector128.Load(x + i)), r + i);
                }
                else
                {
                    for (; i + 4 <= n; i += 4) Vector128.Store(TOp.Apply(Vector128.Load(x + i), vs), r + i);
                }
            }
            if (scalarOnLeft)
            {
                for (; i < n; i++) r[i] = TOp.Apply(s, x[i]);
            }
            else
            {
                for (; i < n; i++) r[i] = TOp.Apply(x[i], s);
            }
        }

        // ------------------------------------------------------------------------------------
        //  Unary / gated activations
        //
        //  Evaluated here with Vector512/256/128.Exp inside AggressiveOptimization loops rather
        //  than through TensorPrimitives: that assembly starts at Tier0 and a per-chunk call is
        //  too short for OSR, so chunked TensorPrimitives calls ran ~10x slow until call counting
        //  tiered them up. Tails go through the same vector code on a padded copy, so an
        //  element's result never depends on where a chunk boundary fell (thread count).
        // ------------------------------------------------------------------------------------

        internal interface IUnary
        {
            static abstract Vector512<float> Apply(Vector512<float> x);
            static abstract Vector256<float> Apply(Vector256<float> x);
            static abstract Vector128<float> Apply(Vector128<float> x);
            static abstract float Apply(float x);
        }

        internal interface IGated
        {
            static abstract Vector512<float> Apply(Vector512<float> x, Vector512<float> y);
            static abstract Vector256<float> Apply(Vector256<float> x, Vector256<float> y);
            static abstract Vector128<float> Apply(Vector128<float> x, Vector128<float> y);
            static abstract float Apply(float x, float y);
        }

        // sigmoid(x) = 1 / (1 + exp(-x)), the TensorPrimitives.Sigmoid formula.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<float> Sig(Vector512<float> x) => Vector512<float>.One / (Vector512<float>.One + Vector512.Exp(-x));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> Sig(Vector256<float> x) => Vector256<float>.One / (Vector256<float>.One + Vector256.Exp(-x));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Sig(Vector128<float> x) => Vector128<float>.One / (Vector128<float>.One + Vector128.Exp(-x));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sig(float x) => 1.0f / (1.0f + MathF.Exp(-x));

        // tanh-approximation GELU with TensorApplyCPU.GELU's constants and inner polynomial
        // u = 0.7978845608 * (x + 0.044715 * x^3). 0.5 * x * (1 + tanh(u)) is evaluated through
        // the exact identity x * sigmoid(2u), which avoids the cancellation of 1 + tanh(u) where
        // tanh(u) -> -1 (the float result of tanh there carries a whole ULP of 1).
        private const float GeluC0 = 0.7978845608f;
        private const float GeluC1 = 0.044715f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<float> GeluV(Vector512<float> v)
        {
            Vector512<float> u = Vector512.Create(GeluC0) * (v + Vector512.Create(GeluC1) * v * v * v);
            return v * Sig(u + u);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> GeluV(Vector256<float> v)
        {
            Vector256<float> u = Vector256.Create(GeluC0) * (v + Vector256.Create(GeluC1) * v * v * v);
            return v * Sig(u + u);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GeluV(Vector128<float> v)
        {
            Vector128<float> u = Vector128.Create(GeluC0) * (v + Vector128.Create(GeluC1) * v * v * v);
            return v * Sig(u + u);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float GeluS(float v)
        {
            float u = GeluC0 * (v + GeluC1 * v * v * v);
            return v * Sig(u + u);
        }

        internal struct ExpOp : IUnary
        {
            public static Vector512<float> Apply(Vector512<float> x) => Vector512.Exp(x);
            public static Vector256<float> Apply(Vector256<float> x) => Vector256.Exp(x);
            public static Vector128<float> Apply(Vector128<float> x) => Vector128.Exp(x);
            public static float Apply(float x) => MathF.Exp(x);
        }

        internal struct SigmoidOp : IUnary
        {
            public static Vector512<float> Apply(Vector512<float> x) => Sig(x);
            public static Vector256<float> Apply(Vector256<float> x) => Sig(x);
            public static Vector128<float> Apply(Vector128<float> x) => Sig(x);
            public static float Apply(float x) => Sig(x);
        }

        // SiLU = x * sigmoid(x) (the previous path: Sigmoid, then Multiply(input, sigmoid)).
        internal struct SiLUOp : IUnary
        {
            public static Vector512<float> Apply(Vector512<float> x) => x * Sig(x);
            public static Vector256<float> Apply(Vector256<float> x) => x * Sig(x);
            public static Vector128<float> Apply(Vector128<float> x) => x * Sig(x);
            public static float Apply(float x) => x * Sig(x);
        }

        internal struct GeluOp : IUnary
        {
            public static Vector512<float> Apply(Vector512<float> x) => GeluV(x);
            public static Vector256<float> Apply(Vector256<float> x) => GeluV(x);
            public static Vector128<float> Apply(Vector128<float> x) => GeluV(x);
            public static float Apply(float x) => GeluS(x);
        }

        // (gate * sigmoid(gate)) * up, the previous MultiplySiLUGateUp order.
        internal struct SiLUMulOp : IGated
        {
            public static Vector512<float> Apply(Vector512<float> g, Vector512<float> u) => g * Sig(g) * u;
            public static Vector256<float> Apply(Vector256<float> g, Vector256<float> u) => g * Sig(g) * u;
            public static Vector128<float> Apply(Vector128<float> g, Vector128<float> u) => g * Sig(g) * u;
            public static float Apply(float g, float u) => g * Sig(g) * u;
        }

        // x * sigmoid(gate).
        internal struct SigmoidMulOp : IGated
        {
            public static Vector512<float> Apply(Vector512<float> x, Vector512<float> g) => x * Sig(g);
            public static Vector256<float> Apply(Vector256<float> x, Vector256<float> g) => x * Sig(g);
            public static Vector128<float> Apply(Vector128<float> x, Vector128<float> g) => x * Sig(g);
            public static float Apply(float x, float g) => x * Sig(g);
        }

        internal struct GeluMulOp : IGated
        {
            public static Vector512<float> Apply(Vector512<float> g, Vector512<float> u) => GeluV(g) * u;
            public static Vector256<float> Apply(Vector256<float> g, Vector256<float> u) => GeluV(g) * u;
            public static Vector128<float> Apply(Vector128<float> g, Vector128<float> u) => GeluV(g) * u;
            public static float Apply(float g, float u) => GeluS(g) * u;
        }

        /// <summary>r = tanh(x) (TensorPrimitives' tanh: keeps full relative accuracy near 0).</summary>
        public static void Tanh(float* r, float* x, long n)
            => Split(n, TranscendentalChunk, (s, e) => ForSpans(s, e, (o, len) =>
                TensorPrimitives.Tanh(new ReadOnlySpan<float>(x + o, len), new Span<float>(r + o, len))));

        /// <summary>r = exp(x).</summary>
        public static void Exp(float* r, float* x, long n)
            => Split(n, TranscendentalChunk, (s, e) => UnaryLoop<ExpOp>(r + s, x + s, e - s));

        /// <summary>r = 1 / (1 + exp(-x)).</summary>
        public static void Sigmoid(float* r, float* x, long n)
            => Split(n, TranscendentalChunk, (s, e) => UnaryLoop<SigmoidOp>(r + s, x + s, e - s));

        /// <summary>r = x * sigmoid(x).</summary>
        public static void SiLU(float* r, float* x, long n)
            => Split(n, TranscendentalChunk, (s, e) => UnaryLoop<SiLUOp>(r + s, x + s, e - s));

        /// <summary>r = (gate * sigmoid(gate)) * up.</summary>
        public static void SiLUMul(float* r, float* gate, float* up, long n)
            => Split(n, TranscendentalChunk, (s, e) => GatedLoop<SiLUMulOp>(r + s, gate + s, up + s, e - s));

        /// <summary>r = (g * sigmoid(g)) * clamp(up, -limit, limit) with g = min(gate, limit).</summary>
        public static void SiLUMulClamp(float* r, float* gate, float* up, long n, float limit)
            => Split(n, TranscendentalChunk, (s, e) => SiLUMulClampLoop(r + s, gate + s, up + s, e - s, limit));

        /// <summary>r = x * sigmoid(gate).</summary>
        public static void SigmoidMul(float* r, float* x, float* gate, long n)
            => Split(n, TranscendentalChunk, (s, e) => GatedLoop<SigmoidMulOp>(r + s, x + s, gate + s, e - s));

        /// <summary>tanh-approximation GELU: 0.5*x*(1 + tanh(0.7978845608*(x + 0.044715*x^3))).</summary>
        public static void Gelu(float* r, float* x, long n)
            => Split(n, TranscendentalChunk, (s, e) => UnaryLoop<GeluOp>(r + s, x + s, e - s));

        /// <summary>r = GELU(gate) * up.</summary>
        public static void GeluMul(float* r, float* gate, float* up, long n)
            => Split(n, TranscendentalChunk, (s, e) => GatedLoop<GeluMulOp>(r + s, gate + s, up + s, e - s));

        /// <summary>r[i] = op(x[i]); r may alias x.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void UnaryLoop<TOp>(float* r, float* x, long n) where TOp : struct, IUnary
        {
            long i = 0;
            if (Use512)
            {
                for (; i + 16 <= n; i += 16) Vector512.Store(TOp.Apply(Vector512.Load(x + i)), r + i);
                if (i < n)
                {
                    float* t = stackalloc float[16];
                    int rem = (int)(n - i);
                    PadTail(t, x + i, rem, 16);
                    Vector512.Store(TOp.Apply(Vector512.Load(t)), t);
                    for (int j = 0; j < rem; j++) r[i + j] = t[j];
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 8 <= n; i += 8) Vector256.Store(TOp.Apply(Vector256.Load(x + i)), r + i);
                if (i < n)
                {
                    float* t = stackalloc float[8];
                    int rem = (int)(n - i);
                    PadTail(t, x + i, rem, 8);
                    Vector256.Store(TOp.Apply(Vector256.Load(t)), t);
                    for (int j = 0; j < rem; j++) r[i + j] = t[j];
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                for (; i + 4 <= n; i += 4) Vector128.Store(TOp.Apply(Vector128.Load(x + i)), r + i);
                if (i < n)
                {
                    float* t = stackalloc float[4];
                    int rem = (int)(n - i);
                    PadTail(t, x + i, rem, 4);
                    Vector128.Store(TOp.Apply(Vector128.Load(t)), t);
                    for (int j = 0; j < rem; j++) r[i + j] = t[j];
                }
            }
            else
            {
                for (; i < n; i++) r[i] = TOp.Apply(x[i]);
            }
        }

        /// <summary>r[i] = op(x[i], y[i]); r may alias x or y.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void GatedLoop<TOp>(float* r, float* x, float* y, long n) where TOp : struct, IGated
        {
            long i = 0;
            if (Use512)
            {
                for (; i + 16 <= n; i += 16) Vector512.Store(TOp.Apply(Vector512.Load(x + i), Vector512.Load(y + i)), r + i);
                if (i < n)
                {
                    float* t = stackalloc float[32];
                    int rem = (int)(n - i);
                    PadTail(t, x + i, rem, 16);
                    PadTail(t + 16, y + i, rem, 16);
                    Vector512.Store(TOp.Apply(Vector512.Load(t), Vector512.Load(t + 16)), t);
                    for (int j = 0; j < rem; j++) r[i + j] = t[j];
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 8 <= n; i += 8) Vector256.Store(TOp.Apply(Vector256.Load(x + i), Vector256.Load(y + i)), r + i);
                if (i < n)
                {
                    float* t = stackalloc float[16];
                    int rem = (int)(n - i);
                    PadTail(t, x + i, rem, 8);
                    PadTail(t + 8, y + i, rem, 8);
                    Vector256.Store(TOp.Apply(Vector256.Load(t), Vector256.Load(t + 8)), t);
                    for (int j = 0; j < rem; j++) r[i + j] = t[j];
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                for (; i + 4 <= n; i += 4) Vector128.Store(TOp.Apply(Vector128.Load(x + i), Vector128.Load(y + i)), r + i);
                if (i < n)
                {
                    float* t = stackalloc float[8];
                    int rem = (int)(n - i);
                    PadTail(t, x + i, rem, 4);
                    PadTail(t + 4, y + i, rem, 4);
                    Vector128.Store(TOp.Apply(Vector128.Load(t), Vector128.Load(t + 4)), t);
                    for (int j = 0; j < rem; j++) r[i + j] = t[j];
                }
            }
            else
            {
                for (; i < n; i++) r[i] = TOp.Apply(x[i], y[i]);
            }
        }

        /// <summary>SiLUMul on g = min(gate, limit), u = min(max(up, -limit), limit) - the IEEE
        /// minimum/maximum TensorPrimitives.Min/Max implement (a NaN operand propagates).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void SiLUMulClampLoop(float* r, float* gate, float* up, long n, float limit)
        {
            const int Block = 256;
            float* g = stackalloc float[Block];
            float* u = stackalloc float[Block];
            for (long o = 0; o < n; o += Block)
            {
                int len = (int)Math.Min(Block, n - o);
                ClampTo(g, gate + o, float.NegativeInfinity, limit, len);
                ClampTo(u, up + o, -limit, limit, len);
                GatedLoop<SiLUMulOp>(r + o, g, u, len);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void ClampTo(float* dst, float* src, float lo, float hi, int n)
        {
            int i = 0;
            if (Use512)
            {
                Vector512<float> vlo = Vector512.Create(lo), vhi = Vector512.Create(hi);
                for (; i + 16 <= n; i += 16) Vector512.Store(Vector512.Min(Vector512.Max(Vector512.Load(src + i), vlo), vhi), dst + i);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vlo = Vector256.Create(lo), vhi = Vector256.Create(hi);
                for (; i + 8 <= n; i += 8) Vector256.Store(Vector256.Min(Vector256.Max(Vector256.Load(src + i), vlo), vhi), dst + i);
            }
            for (; i < n; i++) dst[i] = float.Min(float.Max(src[i], lo), hi);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void PadTail(float* dst, float* src, int count, int width)
        {
            int j = 0;
            for (; j < count; j++) dst[j] = src[j];
            for (; j < width; j++) dst[j] = 0f;
        }

        // ------------------------------------------------------------------------------------
        //  Row ops over contiguous [rows, cols]
        // ------------------------------------------------------------------------------------

        /// <summary>
        /// y = x * (1 / sqrt(mean(x^2) + eps)) [* gamma] [+ beta], per row. gamma/beta may be null.
        /// y may alias x.
        /// </summary>
        public static void RmsNormRows(float* y, float* x, float* gamma, float* beta, long rows, int cols, float eps)
        {
            CpuParallel.ForRange(rows, RowsPerBlock(cols), (s, e) =>
            {
                for (long i = s; i < e; i++)
                    RmsNormRow(y + i * cols, x + i * cols, gamma, beta, cols, eps);
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void RmsNormRow(float* y, float* x, float* gamma, float* beta, int cols, float eps)
        {
            float sqSum = SumSquares(x, cols);
            float invRms = 1.0f / MathF.Sqrt(sqSum / cols + eps);
            int i = 0;
            if (Use512)
            {
                Vector512<float> vi = Vector512.Create(invRms);
                for (; i + 16 <= cols; i += 16)
                {
                    Vector512<float> v = Vector512.Load(x + i) * vi;
                    if (gamma != null) v *= Vector512.Load(gamma + i);
                    if (beta != null) v += Vector512.Load(beta + i);
                    Vector512.Store(v, y + i);
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vi = Vector256.Create(invRms);
                for (; i + 8 <= cols; i += 8)
                {
                    Vector256<float> v = Vector256.Load(x + i) * vi;
                    if (gamma != null) v *= Vector256.Load(gamma + i);
                    if (beta != null) v += Vector256.Load(beta + i);
                    Vector256.Store(v, y + i);
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vi = Vector128.Create(invRms);
                for (; i + 4 <= cols; i += 4)
                {
                    Vector128<float> v = Vector128.Load(x + i) * vi;
                    if (gamma != null) v *= Vector128.Load(gamma + i);
                    if (beta != null) v += Vector128.Load(beta + i);
                    Vector128.Store(v, y + i);
                }
            }
            for (; i < cols; i++)
            {
                float v = x[i] * invRms;
                if (gamma != null) v *= gamma[i];
                if (beta != null) v += beta[i];
                y[i] = v;
            }
        }

        /// <summary>
        /// Two-pass LayerNorm per row: y = gamma * ((x - mean) / sqrt(eps + var)) [+ beta].
        /// gamma null means unit gain. y may alias x.
        /// </summary>
        public static void LayerNormRows(float* y, float* x, float* gamma, float* beta, long rows, int cols, float eps)
        {
            CpuParallel.ForRange(rows, RowsPerBlock(cols), (s, e) =>
            {
                for (long i = s; i < e; i++)
                    LayerNormRow(y + i * cols, x + i * cols, gamma, beta, cols, eps);
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void LayerNormRow(float* y, float* x, float* gamma, float* beta, int cols, float eps)
        {
            float mean = Sum(x, cols) / cols;
            float sqSum = SumSquaredDeviation(x, mean, cols);
            float sigma = (float)Math.Sqrt(eps + sqSum / cols);
            int i = 0;
            if (Use512)
            {
                Vector512<float> vm = Vector512.Create(mean), vs = Vector512.Create(sigma);
                for (; i + 16 <= cols; i += 16)
                {
                    Vector512<float> v = (Vector512.Load(x + i) - vm) / vs;
                    if (gamma != null) v = Vector512.Load(gamma + i) * v;
                    if (beta != null) v += Vector512.Load(beta + i);
                    Vector512.Store(v, y + i);
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vm = Vector256.Create(mean), vs = Vector256.Create(sigma);
                for (; i + 8 <= cols; i += 8)
                {
                    Vector256<float> v = (Vector256.Load(x + i) - vm) / vs;
                    if (gamma != null) v = Vector256.Load(gamma + i) * v;
                    if (beta != null) v += Vector256.Load(beta + i);
                    Vector256.Store(v, y + i);
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vm = Vector128.Create(mean), vs = Vector128.Create(sigma);
                for (; i + 4 <= cols; i += 4)
                {
                    Vector128<float> v = (Vector128.Load(x + i) - vm) / vs;
                    if (gamma != null) v = Vector128.Load(gamma + i) * v;
                    if (beta != null) v += Vector128.Load(beta + i);
                    Vector128.Store(v, y + i);
                }
            }
            for (; i < cols; i++)
            {
                float v = (x[i] - mean) / sigma;
                if (gamma != null) v = gamma[i] * v;
                if (beta != null) v += beta[i];
                y[i] = v;
            }
        }

        /// <summary>Row-wise softmax (max-subtracted, max starts at -inf like ggml_soft_max). y may alias x.</summary>
        public static void SoftmaxRows(float* y, float* x, long rows, int cols)
        {
            // exp costs ~10x a streaming op per element, so rows split earlier.
            CpuParallel.ForRange(rows, Math.Max(1, TranscendentalChunk / Math.Max(1, cols)), (s, e) =>
            {
                for (long i = s; i < e; i++)
                    SoftmaxRow(y + i * cols, x + i * cols, cols);
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void SoftmaxRow(float* y, float* x, int cols)
        {
            float max = Max(x, cols);
            float sum = 0f;
            int i = 0;
            if (Use512)
            {
                Vector512<float> vm = Vector512.Create(max), acc = Vector512<float>.Zero;
                for (; i + 16 <= cols; i += 16)
                {
                    Vector512<float> ex = Vector512.Exp(Vector512.Load(x + i) - vm);
                    Vector512.Store(ex, y + i);
                    acc += ex;
                }
                if (i < cols)
                {
                    // Tail through the same vector exp; -inf padding contributes exp(-inf) = 0.
                    float* t = stackalloc float[16];
                    int rem = cols - i;
                    for (int j = 0; j < 16; j++) t[j] = j < rem ? x[i + j] : float.NegativeInfinity;
                    Vector512<float> ex = Vector512.Exp(Vector512.Load(t) - vm);
                    Vector512.Store(ex, t);
                    acc += ex;
                    for (int j = 0; j < rem; j++) y[i + j] = t[j];
                    i = cols;
                }
                sum = Vector512.Sum(acc);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vm = Vector256.Create(max), acc = Vector256<float>.Zero;
                for (; i + 8 <= cols; i += 8)
                {
                    Vector256<float> ex = Vector256.Exp(Vector256.Load(x + i) - vm);
                    Vector256.Store(ex, y + i);
                    acc += ex;
                }
                if (i < cols)
                {
                    float* t = stackalloc float[8];
                    int rem = cols - i;
                    for (int j = 0; j < 8; j++) t[j] = j < rem ? x[i + j] : float.NegativeInfinity;
                    Vector256<float> ex = Vector256.Exp(Vector256.Load(t) - vm);
                    Vector256.Store(ex, t);
                    acc += ex;
                    for (int j = 0; j < rem; j++) y[i + j] = t[j];
                    i = cols;
                }
                sum = Vector256.Sum(acc);
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vm = Vector128.Create(max), acc = Vector128<float>.Zero;
                for (; i + 4 <= cols; i += 4)
                {
                    Vector128<float> ex = Vector128.Exp(Vector128.Load(x + i) - vm);
                    Vector128.Store(ex, y + i);
                    acc += ex;
                }
                if (i < cols)
                {
                    float* t = stackalloc float[4];
                    int rem = cols - i;
                    for (int j = 0; j < 4; j++) t[j] = j < rem ? x[i + j] : float.NegativeInfinity;
                    Vector128<float> ex = Vector128.Exp(Vector128.Load(t) - vm);
                    Vector128.Store(ex, t);
                    acc += ex;
                    for (int j = 0; j < rem; j++) y[i + j] = t[j];
                    i = cols;
                }
                sum = Vector128.Sum(acc);
            }
            for (; i < cols; i++)
            {
                float ex = MathF.Exp(x[i] - max);
                y[i] = ex;
                sum += ex;
            }

            float invSum = 1.0f / sum;
            ScaleInPlace(y, invSum, cols);
        }

        private static long RowsPerBlock(int cols) => Math.Max(1, RowChunkElements / Math.Max(1, cols));

        // ------------------------------------------------------------------------------------
        //  Reductions / small helpers
        // ------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void ScaleInPlace(float* y, float s, int n)
        {
            int i = 0;
            if (Use512)
            {
                Vector512<float> vs = Vector512.Create(s);
                for (; i + 16 <= n; i += 16) Vector512.Store(Vector512.Load(y + i) * vs, y + i);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vs = Vector256.Create(s);
                for (; i + 8 <= n; i += 8) Vector256.Store(Vector256.Load(y + i) * vs, y + i);
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vs = Vector128.Create(s);
                for (; i + 4 <= n; i += 4) Vector128.Store(Vector128.Load(y + i) * vs, y + i);
            }
            for (; i < n; i++) y[i] *= s;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static float Max(float* x, int n)
        {
            float max = float.NegativeInfinity;
            int i = 0;
            if (Use512 && n >= 16)
            {
                Vector512<float> vm = Vector512.Create(float.NegativeInfinity);
                for (; i + 16 <= n; i += 16) vm = Vector512.Max(vm, Vector512.Load(x + i));
                for (int lane = 0; lane < 16; lane++) max = MathF.Max(max, vm[lane]);
            }
            else if (Vector256.IsHardwareAccelerated && n >= 8)
            {
                Vector256<float> vm = Vector256.Create(float.NegativeInfinity);
                for (; i + 8 <= n; i += 8) vm = Vector256.Max(vm, Vector256.Load(x + i));
                for (int lane = 0; lane < 8; lane++) max = MathF.Max(max, vm[lane]);
            }
            else if (Vector128.IsHardwareAccelerated && n >= 4)
            {
                Vector128<float> vm = Vector128.Create(float.NegativeInfinity);
                for (; i + 4 <= n; i += 4) vm = Vector128.Max(vm, Vector128.Load(x + i));
                for (int lane = 0; lane < 4; lane++) max = MathF.Max(max, vm[lane]);
            }
            for (; i < n; i++) max = MathF.Max(max, x[i]);
            return max;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static float Sum(float* x, int n)
        {
            int i = 0;
            float sum = 0f;
            if (Use512)
            {
                Vector512<float> a0 = Vector512<float>.Zero, a1 = Vector512<float>.Zero;
                for (; i + 32 <= n; i += 32)
                {
                    a0 += Vector512.Load(x + i);
                    a1 += Vector512.Load(x + i + 16);
                }
                for (; i + 16 <= n; i += 16) a0 += Vector512.Load(x + i);
                sum = Vector512.Sum(a0 + a1);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
                for (; i + 16 <= n; i += 16)
                {
                    a0 += Vector256.Load(x + i);
                    a1 += Vector256.Load(x + i + 8);
                }
                for (; i + 8 <= n; i += 8) a0 += Vector256.Load(x + i);
                sum = Vector256.Sum(a0 + a1);
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> a0 = Vector128<float>.Zero;
                for (; i + 4 <= n; i += 4) a0 += Vector128.Load(x + i);
                sum = Vector128.Sum(a0);
            }
            for (; i < n; i++) sum += x[i];
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static float SumSquares(float* x, int n)
        {
            int i = 0;
            float sum = 0f;
            if (Use512)
            {
                Vector512<float> a0 = Vector512<float>.Zero, a1 = Vector512<float>.Zero;
                for (; i + 32 <= n; i += 32)
                {
                    Vector512<float> v0 = Vector512.Load(x + i), v1 = Vector512.Load(x + i + 16);
                    a0 += v0 * v0;
                    a1 += v1 * v1;
                }
                for (; i + 16 <= n; i += 16)
                {
                    Vector512<float> v = Vector512.Load(x + i);
                    a0 += v * v;
                }
                sum = Vector512.Sum(a0 + a1);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
                for (; i + 16 <= n; i += 16)
                {
                    Vector256<float> v0 = Vector256.Load(x + i), v1 = Vector256.Load(x + i + 8);
                    a0 += v0 * v0;
                    a1 += v1 * v1;
                }
                for (; i + 8 <= n; i += 8)
                {
                    Vector256<float> v = Vector256.Load(x + i);
                    a0 += v * v;
                }
                sum = Vector256.Sum(a0 + a1);
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> a0 = Vector128<float>.Zero;
                for (; i + 4 <= n; i += 4)
                {
                    Vector128<float> v = Vector128.Load(x + i);
                    a0 += v * v;
                }
                sum = Vector128.Sum(a0);
            }
            for (; i < n; i++) sum += x[i] * x[i];
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static float SumSquaredDeviation(float* x, float mean, int n)
        {
            int i = 0;
            float sum = 0f;
            if (Use512)
            {
                Vector512<float> vm = Vector512.Create(mean), a0 = Vector512<float>.Zero;
                for (; i + 16 <= n; i += 16)
                {
                    Vector512<float> d = Vector512.Load(x + i) - vm;
                    a0 += d * d;
                }
                sum = Vector512.Sum(a0);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vm = Vector256.Create(mean), a0 = Vector256<float>.Zero;
                for (; i + 8 <= n; i += 8)
                {
                    Vector256<float> d = Vector256.Load(x + i) - vm;
                    a0 += d * d;
                }
                sum = Vector256.Sum(a0);
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vm = Vector128.Create(mean), a0 = Vector128<float>.Zero;
                for (; i + 4 <= n; i += 4)
                {
                    Vector128<float> d = Vector128.Load(x + i) - vm;
                    a0 += d * d;
                }
                sum = Vector128.Sum(a0);
            }
            for (; i < n; i++)
            {
                float d = x[i] - mean;
                sum += d * d;
            }
            return sum;
        }

        // ------------------------------------------------------------------------------------
        //  Work splitting
        // ------------------------------------------------------------------------------------

        private static void Split(long n, long minChunk, Action<long, long> body) => CpuParallel.ForRange(n, minChunk, body);

        /// <summary>TensorPrimitives takes int-length spans: walk [start, end) in span-sized pieces.</summary>
        private static void ForSpans(long start, long end, Action<long, int> body)
        {
            const int MaxSpan = 1 << 28;
            for (long o = start; o < end; o += MaxSpan)
                body(o, (int)Math.Min(MaxSpan, end - o));
        }
    }
}
