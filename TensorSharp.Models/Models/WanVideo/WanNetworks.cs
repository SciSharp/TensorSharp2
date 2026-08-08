// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Backend-selection seams for the three Wan networks: each has a GGML
// whole-graph implementation (WanTextEncoder / WanDiT / WanVae) and a direct
// Cuda/Cpu implementation (WanDirectT5 / WanDirectDiT / WanDirectVae). The
// pipeline talks to these interfaces only; the VAE seam is the WanVaeBase
// class (it carries shared pixel-pipeline logic, not just a contract).
using System;

namespace TensorSharp.Models.WanVideo
{
    internal interface IWanTextEncoder : IDisposable
    {
        /// <summary>Encode a prompt to the Wan conditioning: [512, 4096] hidden
        /// states, zero at padded positions.</summary>
        float[] Encode(string text);
    }

    internal interface IWanDit : IDisposable
    {
        /// <summary>Transformer width (1536 for the 1.3B model — used by the
        /// pipeline's flow-shift recipe).</summary>
        int Dim { get; }
        /// <summary>DiT input channels (16 T2V / 36 A14B I2V / 48 TI2V).</summary>
        int InDim { get; }

        /// <summary>One denoising-step forward: patchified latent tokens in,
        /// velocity tokens out (see WanDiT.Predict).</summary>
        float[] Predict(float[] xTokens, float[] context, float timestep, WanRope rope, int seq,
                        int seq0 = 0, float timestep0 = 0f);
    }
}
