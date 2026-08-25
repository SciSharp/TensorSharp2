// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 flow-matching schedule.
//
// H3 denoises video and audio jointly, but the two streams want DIFFERENT
// timestep shifts: 12 for video, 3 for audio. Running two samplers is not an
// option -- both streams live in one packed tensor and are stepped together.
//
// The trick H3 uses is to keep the sampler entirely on the video schedule and
// make the model do the conversion:
//
//   * the audio stream is conditioned on the timestep a shift-3 schedule would
//     have had at the same underlying uniform time -- MapSigma() below unshifts
//     by 12 and reshifts by 3; and
//   * the audio velocity the model emits is pre-multiplied by dSigmaAudio/dSigmaVideo
//     (ShiftSlope) so that the shared Euler update x += v * dSigmaVideo integrates
//     the audio along its own trajectory.
//
// Both schedules start at sigma = 1 and end at sigma = 0, so the endpoints agree
// exactly and one sampler lands both streams on the data manifold together.
using System;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Discrete flow-matching schedule for MiniMax-H3, with the dual video/audio
    /// timestep shift folded into a single sampler pass.</summary>
    public static class MiniMaxH3Scheduler
    {
        /// <summary>Default timestep shift for the video stream.</summary>
        public const float VideoShift = 12.0f;

        /// <summary>Timestep shift for the audio stream. Fixed by the checkpoint; it is
        /// NOT affected by a user-supplied flow shift, which only moves the video stream.</summary>
        public const float AudioShift = 3.0f;

        /// <summary>Timestep at which conditioning (keyframe / reference) latents are presented:
        /// essentially clean, but not exactly, matching the tiny noise blend applied to them.</summary>
        public const float VisualConditionTimestep = 0.999f;

        /// <summary>Default denoising steps. H3 is step-distilled and CFG-free; 4-8 steps
        /// are the usual operating point, and the reference CLI default is 20.</summary>
        public const int DefaultSteps = 20;

        /// <summary>The flow-matching SNR shift: sigma' = a*t / (1 + (a-1)*t).</summary>
        public static double TimeSnrShift(double shift, double t) => shift * t / (1.0 + (shift - 1.0) * t);

        /// <summary>Inverse of <see cref="TimeSnrShift"/>: recovers the uniform time that
        /// produced <paramref name="sigma"/> under <paramref name="shift"/>.</summary>
        public static double InverseTimeSnrShift(double shift, double sigma) =>
            sigma / (shift + sigma * (1.0 - shift));

        /// <summary>Build the discrete sigma schedule for the video stream. Returns
        /// <paramref name="steps"/>+1 values running from 1.0 down to exactly 0.0.</summary>
        public static float[] BuildSigmas(int steps, float shift = VideoShift)
        {
            if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
            var sigmas = new float[steps + 1];
            for (int i = 0; i < steps; i++)
            {
                // The reference walks the discrete timestep axis 999 -> 0 uniformly,
                // then maps each through the shift. t+1 (not t) is what gets divided
                // by 1000, so the last step lands just above zero rather than on it.
                double t = steps == 1 ? 999.0 : 999.0 - i * (999.0 / (steps - 1));
                sigmas[i] = (float)TimeSnrShift(shift, (t + 1.0) / 1000.0);
            }
            sigmas[steps] = 0f;
            return sigmas;
        }

        /// <summary>Re-express a sigma from one shift's schedule on another's:
        /// unshift by <paramref name="fromShift"/>, reshift by <paramref name="toShift"/>.
        /// This is how the audio stream's timestep is derived from the video sigma.</summary>
        public static float MapSigma(float sigma, float fromShift = VideoShift, float toShift = AudioShift)
        {
            double b = InverseTimeSnrShift(fromShift, sigma);
            return (float)TimeSnrShift(toShift, b);
        }

        /// <summary>d(sigmaTo)/d(sigmaFrom) -- the chain-rule factor the audio velocity is
        /// scaled by so a single Euler step on the video schedule advances the audio stream
        /// along its own schedule.</summary>
        public static float ShiftSlope(float sigma, float fromShift = VideoShift, float toShift = AudioShift)
        {
            double b = InverseTimeSnrShift(fromShift, sigma);
            double a = 1.0 + (fromShift - 1.0) * b;
            double c = 1.0 + (toShift - 1.0) * b;
            return (float)(toShift * a * a / (fromShift * c * c));
        }

        /// <summary>Conditioning timestep fed to the video stream's AdaLN for a given sigma.</summary>
        public static float VideoTimestep(float sigma) => 1f - sigma;

        /// <summary>Conditioning timestep fed to the audio stream's AdaLN for a given
        /// VIDEO sigma (the sampler only ever knows the video schedule).</summary>
        public static float AudioTimestep(float sigma, float videoShift = VideoShift, float audioShift = AudioShift)
            => 1f - MapSigma(sigma, videoShift, audioShift);

        /// <summary>Clamp a video sigma into the domain the timestep mapping is defined on.</summary>
        public static float ClampSigma(float sigma) => Math.Clamp(sigma, 1e-6f, 1f);

        /// <summary>One Euler update: <c>x += velocity * (sigmaNext - sigma)</c>.
        /// The model already emits the correctly-signed and correctly-scaled velocity for
        /// both streams, so the sampler itself stays completely stream-agnostic.</summary>
        public static void EulerStep(float[] x, float[] velocity, float sigma, float sigmaNext)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (velocity is null) throw new ArgumentNullException(nameof(velocity));
            if (x.Length != velocity.Length)
                throw new ArgumentException("velocity length must match the latent length", nameof(velocity));
            float dt = sigmaNext - sigma;
            for (int i = 0; i < x.Length; i++) x[i] += velocity[i] * dt;
        }
    }
}
