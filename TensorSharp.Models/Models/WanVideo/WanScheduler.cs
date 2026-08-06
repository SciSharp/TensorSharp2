// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.WanVideo
{
    /// <summary>
    /// FlowMatch Euler scheduler with the constant timestep shift Wan uses
    /// (diffusers <c>FlowMatchEulerDiscreteScheduler</c> with <c>shift</c>, no
    /// resolution-dependent mu): sigma' = shift*s / (1 + (shift-1)*s) over
    /// s = linspace(1, 1/N, N), ending at 0.
    /// </summary>
    internal sealed class WanScheduler
    {
        public float[] Sigmas { get; }      // length steps+1, ending at 0
        public float[] Timesteps { get; }   // length steps, = sigma*1000

        public int Steps => Timesteps.Length;

        public WanScheduler(int numSteps, float shift)
        {
            var sigmas = new float[numSteps + 1];
            for (int i = 0; i < numSteps; i++)
            {
                double s = numSteps == 1 ? 1.0 : 1.0 + (1.0 / numSteps - 1.0) * i / (numSteps - 1);
                sigmas[i] = (float)(shift * s / (1.0 + (shift - 1.0) * s));
            }
            sigmas[numSteps] = 0f;
            Sigmas = sigmas;
            Timesteps = new float[numSteps];
            for (int i = 0; i < numSteps; i++) Timesteps[i] = sigmas[i] * 1000f;
        }

        /// <summary>Euler step in place: <c>x += (sigma_next - sigma) * velocity</c>.</summary>
        public void Step(float[] latents, float[] velocity, int stepIndex)
        {
            float dt = Sigmas[stepIndex + 1] - Sigmas[stepIndex];
            for (long i = 0; i < latents.Length; i++)
                latents[i] += dt * velocity[i];
        }
    }
}
