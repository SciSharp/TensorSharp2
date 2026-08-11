// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// UniPC multistep scheduler for flow matching — the sampler the official Wan
// pipeline uses (diffusers UniPCMultistepScheduler with use_flow_sigmas=True,
// prediction_type="flow_prediction", solver_order=2, solver_type="bh2",
// predict_x0=True, lower_order_final=True). A faithful port of
// diffusers/schedulers/scheduling_unipc_multistep.py restricted to that
// configuration: predictor-corrector, one model evaluation per step, effective
// 3rd-order accuracy — visibly better structure and detail than Euler at the
// same step count.
using System;
using System.Threading.Tasks;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanUniPCScheduler
    {
        private const int Order = 2;
        private const int NumTrainTimesteps = 1000;

        public float[] Sigmas { get; }      // length steps+1, ending at 0
        public float[] Timesteps { get; }   // length steps; floor(sigma*1000) (diffusers casts to int64)

        public int Steps => Timesteps.Length;

        private readonly float[][] _modelOutputs = new float[Order][];   // x0 predictions
        private float[] _lastSample;
        private int _lowerOrderNums;
        private int _thisOrder;
        private int _stepIndex;

        public WanUniPCScheduler(int numSteps, float shift)
        {
            // sigmas = linspace(1, 1/1000, N+1)[:-1], then the constant flow shift.
            var sigmas = new double[numSteps + 1];
            for (int i = 0; i < numSteps; i++)
            {
                double s = 1.0 - i * (1.0 - 1.0 / NumTrainTimesteps) / numSteps;
                sigmas[i] = shift * s / (1.0 + (shift - 1.0) * s);
            }
            // Guard: sigma_0 == 1 makes alpha = 1-sigma = 0 and log(alpha) = -inf in
            // the first multistep updates (diffusers applies the same epsilon).
            if (Math.Abs(sigmas[0] - 1.0) < 1e-6) sigmas[0] -= 1e-6;
            sigmas[numSteps] = 0.0;

            Sigmas = new float[numSteps + 1];
            Timesteps = new float[numSteps];
            for (int i = 0; i <= numSteps; i++) Sigmas[i] = (float)sigmas[i];
            for (int i = 0; i < numSteps; i++) Timesteps[i] = (long)(sigmas[i] * NumTrainTimesteps);
        }

        private static double Lambda(double sigma)
        {
            double alpha = 1.0 - sigma;
            return Math.Log(alpha) - Math.Log(sigma);
        }

        // exp(x) - 1 with small-x accuracy (.NET has no Math.Expm1).
        private static double Expm1(double x)
        {
            if (Math.Abs(x) < 1e-5) return x + 0.5 * x * x;
            return Math.Exp(x) - 1.0;
        }

        /// <summary>
        /// One UniPC step: corrects the current sample with this step's model output
        /// (UniC), then predicts the next sample (UniP). Returns the next sample;
        /// <paramref name="sample"/> and <paramref name="velocity"/> are not modified.
        /// </summary>
        public float[] Step(float[] sample, float[] velocity)
        {
            int i = _stepIndex;
            long n = sample.LongLength;
            float sigma = Sigmas[i];

            // flow prediction, predict_x0: x0 = sample - sigma * v
            var x0 = new float[n];
            Parallel.For(0, (int)(n / 4096) + 1, blk =>
            {
                long start = (long)blk * 4096, end = Math.Min(n, start + 4096);
                for (long j = start; j < end; j++) x0[j] = sample[j] - sigma * velocity[j];
            });

            float[] s = sample;
            if (i > 0 && _lastSample != null)
                s = UniC(x0, _lastSample, sample, _thisOrder);

            // roll the model-output history
            for (int k = 0; k < Order - 1; k++) _modelOutputs[k] = _modelOutputs[k + 1];
            _modelOutputs[Order - 1] = x0;

            int thisOrder = Math.Min(Order, Steps - i);          // lower_order_final
            _thisOrder = Math.Min(thisOrder, _lowerOrderNums + 1);   // multistep warmup
            _lastSample = s;

            float[] prev = UniP(s, _thisOrder);
            if (_lowerOrderNums < Order) _lowerOrderNums++;
            _stepIndex++;
            return prev;
        }

        // Predictor (UniP, B(h) with bh2): sample at sigma[i] -> sample at sigma[i+1].
        private float[] UniP(float[] x, int order)
        {
            int i = _stepIndex;
            long n = x.LongLength;
            double sigmaT = Sigmas[i + 1], sigmaS0 = Sigmas[i];
            double alphaT = 1.0 - sigmaT;
            float[] m0 = _modelOutputs[Order - 1];
            var outp = new float[n];

            if (sigmaT <= 0.0)
            {
                // Limit h -> inf: h_phi_1 = B_h = -1, sigma_t/sigma_s0 = 0 — the update
                // collapses to the x0 prediction (order is 1 here via lower_order_final).
                Array.Copy(m0, outp, n);
                return outp;
            }

            double h = Lambda(sigmaT) - Lambda(sigmaS0);
            double hh = -h;
            double hPhi1 = Expm1(hh);
            double ratio = sigmaT / sigmaS0;

            if (order >= 2)
            {
                // rk from the older model output; rhos_p = 0.5 for order 2 (diffusers'
                // simplified path).
                double lambdaS1 = Lambda(Sigmas[i - 1]);
                double rk = (lambdaS1 - Lambda(sigmaS0)) / h;
                double bH = Expm1(hh);   // bh2
                float[] m1 = _modelOutputs[Order - 2];
                double c0 = -alphaT * hPhi1;
                double c1 = -alphaT * bH * 0.5 / rk;
                Parallel.For(0, (int)(n / 4096) + 1, blk =>
                {
                    long start = (long)blk * 4096, end = Math.Min(n, start + 4096);
                    for (long j = start; j < end; j++)
                    {
                        double d1 = m1[j] - m0[j];
                        outp[j] = (float)(ratio * x[j] + c0 * m0[j] + c1 * d1);
                    }
                });
            }
            else
            {
                double c0 = -alphaT * hPhi1;
                Parallel.For(0, (int)(n / 4096) + 1, blk =>
                {
                    long start = (long)blk * 4096, end = Math.Min(n, start + 4096);
                    for (long j = start; j < end; j++)
                        outp[j] = (float)(ratio * x[j] + c0 * m0[j]);
                });
            }
            return outp;
        }

        // Corrector (UniC, B(h) with bh2): re-derives the CURRENT sample from the last
        // one using this step's fresh model output. thisX0 is the x0 prediction at the
        // current sample; lastSample is the sample before the previous predictor.
        private float[] UniC(float[] thisX0, float[] lastSample, float[] thisSample, int order)
        {
            int i = _stepIndex;
            long n = thisSample.LongLength;
            double sigmaT = Sigmas[i], sigmaS0 = Sigmas[i - 1];
            double alphaT = 1.0 - sigmaT;
            double h = Lambda(sigmaT) - Lambda(sigmaS0);
            double hh = -h;
            double hPhi1 = Expm1(hh);
            double bH = Expm1(hh);   // bh2
            double ratio = sigmaT / sigmaS0;
            float[] m0 = _modelOutputs[Order - 1];
            var outp = new float[n];

            if (order >= 2 && _modelOutputs[Order - 2] != null)
            {
                double lambdaS1 = Lambda(Sigmas[i - 2]);
                double rk = (lambdaS1 - Lambda(sigmaS0)) / h;

                // rhos_c = solve(R, b) with R = [[1, 1], [rk, 1]] (rows rks^0, rks^1):
                //   b1 = h_phi_1_over := (h_phi_1/hh - 1) / B_h
                //   b2 = ((h_phi_1/hh - 1)/hh - 1/2) * 2 / B_h
                double hPhiK = hPhi1 / hh - 1.0;
                double b1 = hPhiK / bH;
                hPhiK = hPhiK / hh - 1.0 / 2.0;
                double b2 = hPhiK * 2.0 / bH;
                double det = 1.0 - rk;
                double rho0 = (b1 - b2) / det;
                double rho1 = (b2 - rk * b1) / det;

                float[] m1 = _modelOutputs[Order - 2];
                double c0 = -alphaT * hPhi1;
                double cD1 = -alphaT * bH * rho0 / rk;
                double cDt = -alphaT * bH * rho1;
                Parallel.For(0, (int)(n / 4096) + 1, blk =>
                {
                    long start = (long)blk * 4096, end = Math.Min(n, start + 4096);
                    for (long j = start; j < end; j++)
                    {
                        double d1 = m1[j] - m0[j];
                        double d1t = thisX0[j] - m0[j];
                        outp[j] = (float)(ratio * lastSample[j] + c0 * m0[j] + cD1 * d1 + cDt * d1t);
                    }
                });
            }
            else
            {
                // order 1: rhos_c = [0.5]
                double c0 = -alphaT * hPhi1;
                double cDt = -alphaT * bH * 0.5;
                Parallel.For(0, (int)(n / 4096) + 1, blk =>
                {
                    long start = (long)blk * 4096, end = Math.Min(n, start + 4096);
                    for (long j = start; j < end; j++)
                    {
                        double d1t = thisX0[j] - m0[j];
                        outp[j] = (float)(ratio * lastSample[j] + c0 * m0[j] + cDt * d1t);
                    }
                });
            }
            return outp;
        }
    }
}
