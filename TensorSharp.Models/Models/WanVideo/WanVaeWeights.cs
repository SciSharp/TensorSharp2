// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Managed weight tables for the Wan causal 3D video VAE, parsed once from the
// original safetensors checkpoint (wan_2.1_vae.safetensors / Wan2.2_VAE.safetensors)
// and shared by BOTH execution paths: the GGML whole-graph kernels (WanVae
// registers the arrays as stable unmanaged F32 pointers) and the direct
// Cuda/Cpu backends (WanDirectVae uploads them as device tensors). Every causal
// 3D conv is pre-sliced into its temporal taps; each tap is a [oc, ic*kh*kw]
// row-major matrix — simultaneously the ggml [kw, kh, ic, oc] conv kernel and
// the im2col GEMM weight, which are the same bytes.
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using TensorSharp.Runtime;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanVaeConvWeights
    {
        public int Kd, K, Ic, Oc;
        public float[][] Taps;    // Kd arrays of [Oc * Ic * K * K]
        public float[] Bias;      // [Oc] or null
    }

    internal sealed class WanVaeNormWeights
    {
        public float[] Gamma;     // [C]
        public int C;
    }

    internal sealed class WanVaeResBlockWeights
    {
        public WanVaeNormWeights N0, N3;
        public WanVaeConvWeights C2, C6;
        public WanVaeConvWeights Shortcut;   // null when in == out
    }

    internal sealed class WanVaeAttnWeights
    {
        public WanVaeNormWeights Norm;
        public float[] QkvW;      // [3C, C] row-major
        public float[] QkvB;      // [3C]
        public float[] ProjW;     // [C, C]
        public float[] ProjB;     // [C]
        public int C;
    }

    internal sealed class WanVaeUpsampleWeights
    {
        public WanVaeConvWeights TimeConv;   // (3,1,1) causal temporal conv or null
        public WanVaeConvWeights SConv;      // 3x3 2D conv after nearest x2
    }

    internal sealed class WanVaeDownWeights
    {
        public WanVaeConvWeights SConv;      // 3x3 stride-2 2D conv
        public WanVaeConvWeights TConv;      // (3,1,1) stride-2 temporal conv or null
    }

    /// <summary>
    /// The full parsed VAE checkpoint: decoder (+ encoder when present) weight
    /// tables in the layout both execution paths consume. See WanVae for the
    /// architecture notes (Wan 2.1 vs the Wan 2.2 TI2V variant).
    /// </summary>
    internal sealed class WanVaeWeights
    {
        /// <summary>1 = Wan 2.1, 2 = Wan 2.2 TI2V.</summary>
        public int Version;
        public int ZDim;
        public int PatchSize;
        public int SpatialScale;
        public bool HasEncoder;

        // ---- decoder ----
        public WanVaeConvWeights Conv2, Conv1;
        public WanVaeResBlockWeights Mid0, Mid2;
        public WanVaeAttnWeights Mid1;
        public WanVaeResBlockWeights[] Res;   // 12
        public WanVaeUpsampleWeights[] Up;    // 3
        public WanVaeNormWeights HeadNorm;
        public WanVaeConvWeights HeadConv;

        // ---- encoder ----
        public WanVaeConvWeights EncStem;
        public WanVaeResBlockWeights[] EncRes;   // 8
        public WanVaeDownWeights[] EncDown;      // 3
        public WanVaeResBlockWeights EncMid0, EncMid2;
        public WanVaeAttnWeights EncMid1;
        public WanVaeNormWeights EncHeadNorm;
        public WanVaeConvWeights EncHeadConv;
        public WanVaeConvWeights EncQuant;

        public static WanVaeWeights Load(string vaePath)
        {
            if (!vaePath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"Wan VAE must be an original safetensors checkpoint (wan_2.1_vae.safetensors / Wan2.2_VAE.safetensors); got '{vaePath}'.");
            using var st = new SafetensorsFile(vaePath);
            var w = new WanVaeWeights();

            // Wan 2.2 nests each scale's blocks inside Up_/Down_ResidualBlock groups.
            w.Version = st.HasTensor("decoder.upsamples.0.upsamples.0.residual.0.gamma") ? 2 : 1;
            w.ZDim = w.Version == 2 ? 48 : 16;
            w.PatchSize = w.Version == 2 ? 2 : 1;
            w.SpatialScale = 8 * w.PatchSize;

            WanVaeConvWeights Conv(string prefix, int kd, int k, int ic, int oc)
            {
                float[] wts = st.ReadFloat32(prefix + ".weight");   // torch [oc, ic, kd, kh, kw] (or [oc, ic, kh, kw])
                float[] b = st.HasTensor(prefix + ".bias") ? st.ReadFloat32(prefix + ".bias") : null;
                var c = new WanVaeConvWeights { Kd = kd, K = k, Ic = ic, Oc = oc, Bias = b, Taps = new float[kd][] };
                int khw = k * k;
                int kdhw = kd * khw;
                for (int tap = 0; tap < kd; tap++)
                {
                    // torch element (oc, ic, kd, kh, kw) -> per-tap [oc, ic, kh, kw]
                    // (the ggml [kw, kh, ic, oc] kernel and the [oc, ic*k*k] GEMM
                    // weight are this same memory).
                    var slice = new float[(long)oc * ic * khw];
                    int tapLocal = tap;
                    Parallel.For(0, oc, o =>
                    {
                        for (int i = 0; i < ic; i++)
                        {
                            long src = ((long)o * ic + i) * kdhw + (long)tapLocal * khw;
                            long dst = ((long)o * ic + i) * khw;
                            Array.Copy(wts, src, slice, dst, khw);
                        }
                    });
                    c.Taps[tap] = slice;
                }
                return c;
            }
            WanVaeNormWeights Norm(string name, int ch) => new WanVaeNormWeights { Gamma = st.ReadFloat32(name), C = ch };
            WanVaeResBlockWeights Res(string prefix, int inDim, int outDim)
            {
                var r = new WanVaeResBlockWeights
                {
                    N0 = Norm(prefix + ".residual.0.gamma", inDim),
                    C2 = Conv(prefix + ".residual.2", 3, 3, inDim, outDim),
                    N3 = Norm(prefix + ".residual.3.gamma", outDim),
                    C6 = Conv(prefix + ".residual.6", 3, 3, outDim, outDim),
                };
                if (inDim != outDim)
                    r.Shortcut = Conv(prefix + ".shortcut", 1, 1, inDim, outDim);
                return r;
            }
            WanVaeAttnWeights Attn(string prefix, int dim) => new WanVaeAttnWeights
            {
                Norm = Norm(prefix + ".norm.gamma", dim),
                QkvW = st.ReadFloat32(prefix + ".to_qkv.weight"),
                QkvB = st.ReadFloat32(prefix + ".to_qkv.bias"),
                ProjW = st.ReadFloat32(prefix + ".proj.weight"),
                ProjB = st.ReadFloat32(prefix + ".proj.bias"),
                C = dim,
            };

            // ---- decoder ----
            if (w.Version == 1)
            {
                w.Conv2 = Conv("conv2", 1, 1, w.ZDim, w.ZDim);
                w.Conv1 = Conv("decoder.conv1", 3, 3, w.ZDim, 384);
                w.Mid0 = Res("decoder.middle.0", 384, 384);
                w.Mid2 = Res("decoder.middle.2", 384, 384);
                w.Mid1 = Attn("decoder.middle.1", 384);

                // decoder.upsamples.N layout (wan2.1): per scale i in 0..3, three residual
                // blocks then (i < 3) a Resample; the first block of scales 1..3 changes
                // the channel count (the preceding Resample halved it).
                int[] resIn = { 384, 384, 384, 192, 384, 384, 192, 192, 192, 96, 96, 96 };
                int[] resOut = { 384, 384, 384, 384, 384, 384, 192, 192, 192, 96, 96, 96 };
                w.Res = new WanVaeResBlockWeights[12];
                w.Up = new WanVaeUpsampleWeights[3];
                int idx = 0, ri = 0;
                for (int scale = 0; scale < 4; scale++)
                {
                    for (int rblk = 0; rblk < 3; rblk++)
                    {
                        w.Res[ri] = Res($"decoder.upsamples.{idx++}", resIn[ri], resOut[ri]);
                        ri++;
                    }
                    if (scale < 3)
                    {
                        string p = $"decoder.upsamples.{idx++}";
                        int dim = resOut[ri - 1];
                        var up = new WanVaeUpsampleWeights
                        {
                            SConv = Conv(p + ".resample.1", 1, 3, dim, dim / 2),
                        };
                        if (st.HasTensor(p + ".time_conv.weight"))
                            up.TimeConv = Conv(p + ".time_conv", 3, 1, dim, dim * 2);
                        w.Up[scale] = up;
                    }
                }

                w.HeadNorm = Norm("decoder.head.0.gamma", 96);
                w.HeadConv = Conv("decoder.head.2", 3, 3, 96, 3);
            }
            else
            {
                w.Conv2 = Conv("conv2", 1, 1, w.ZDim, w.ZDim);
                w.Conv1 = Conv("decoder.conv1", 3, 3, w.ZDim, 1024);
                w.Mid0 = Res("decoder.middle.0", 1024, 1024);
                w.Mid2 = Res("decoder.middle.2", 1024, 1024);
                w.Mid1 = Attn("decoder.middle.1", 1024);

                // wan2.2 decoder: 4 Up_ResidualBlock groups of 3 residual blocks, with a
                // channel-preserving Resample (upsamples.3) after groups 0..2; the DupUp3D
                // shortcuts around each group are weight-free.
                int[] resIn = { 1024, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256 };
                int[] resOut = { 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256, 256 };
                w.Res = new WanVaeResBlockWeights[12];
                w.Up = new WanVaeUpsampleWeights[3];
                for (int scale = 0; scale < 4; scale++)
                {
                    for (int rblk = 0; rblk < 3; rblk++)
                    {
                        int ri = scale * 3 + rblk;
                        w.Res[ri] = Res($"decoder.upsamples.{scale}.upsamples.{rblk}", resIn[ri], resOut[ri]);
                    }
                    if (scale < 3)
                    {
                        string p = $"decoder.upsamples.{scale}.upsamples.3";
                        int dim = resOut[scale * 3 + 2];
                        var up = new WanVaeUpsampleWeights
                        {
                            SConv = Conv(p + ".resample.1", 1, 3, dim, dim),
                        };
                        if (st.HasTensor(p + ".time_conv.weight"))
                            up.TimeConv = Conv(p + ".time_conv", 3, 1, dim, dim * 2);
                        w.Up[scale] = up;
                    }
                }

                w.HeadNorm = Norm("decoder.head.0.gamma", 256);
                w.HeadConv = Conv("decoder.head.2", 3, 3, 256, 12);
            }

            // ---- encoder (image/video -> latent; used for I2V conditioning) ----
            w.HasEncoder = st.HasTensor("encoder.conv1.weight");
            if (w.HasEncoder)
            {
                w.EncRes = new WanVaeResBlockWeights[8];
                w.EncDown = new WanVaeDownWeights[3];
                int[] resIn, resOut;
                if (w.Version == 1)
                {
                    w.EncStem = Conv("encoder.conv1", 3, 3, 3, 96);
                    resIn = new[] { 96, 96, 96, 192, 192, 384, 384, 384 };
                    resOut = new[] { 96, 96, 192, 192, 384, 384, 384, 384 };
                }
                else
                {
                    w.EncStem = Conv("encoder.conv1", 3, 3, 12, 160);
                    resIn = new[] { 160, 160, 160, 320, 320, 640, 640, 640 };
                    resOut = new[] { 160, 160, 320, 320, 640, 640, 640, 640 };
                }
                int idx = 0;
                for (int scale = 0; scale < 4; scale++)
                {
                    for (int rblk = 0; rblk < 2; rblk++)
                    {
                        int ri = scale * 2 + rblk;
                        string prefix = w.Version == 1
                            ? $"encoder.downsamples.{idx++}"
                            : $"encoder.downsamples.{scale}.downsamples.{rblk}";
                        w.EncRes[ri] = Res(prefix, resIn[ri], resOut[ri]);
                    }
                    if (scale < 3)
                    {
                        string p = w.Version == 1
                            ? $"encoder.downsamples.{idx++}"
                            : $"encoder.downsamples.{scale}.downsamples.2";
                        int dim = resOut[scale * 2 + 1];
                        var dn = new WanVaeDownWeights { SConv = Conv(p + ".resample.1", 1, 3, dim, dim) };
                        if (st.HasTensor(p + ".time_conv.weight"))
                            dn.TConv = Conv(p + ".time_conv", 3, 1, dim, dim);
                        w.EncDown[scale] = dn;
                    }
                }
                int mid = w.Version == 1 ? 384 : 640;
                w.EncMid0 = Res("encoder.middle.0", mid, mid);
                w.EncMid2 = Res("encoder.middle.2", mid, mid);
                w.EncMid1 = Attn("encoder.middle.1", mid);
                w.EncHeadNorm = Norm("encoder.head.0.gamma", mid);
                w.EncHeadConv = Conv("encoder.head.2", 3, 3, mid, 2 * w.ZDim);
                w.EncQuant = Conv("conv1", 1, 1, 2 * w.ZDim, 2 * w.ZDim);
            }
            return w;
        }
    }
}
