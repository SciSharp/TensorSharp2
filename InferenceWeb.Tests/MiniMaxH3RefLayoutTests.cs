// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Ref2VA packed-sequence layout, against a dump from stable-diffusion.cpp.
//
// The layout is where a reference's place on the shared timeline is decided, and
// getting it wrong produces a clip that is merely worse rather than one that
// fails — so it is checked against the reference implementation number for
// number, not just for self-consistency. This test needs no weights.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3RefLayoutTests
    {
        private const string DirEnv = "TS_MINIMAX_H3_DIR";
        private readonly ITestOutputHelper _output;

        public MiniMaxH3RefLayoutTests(ITestOutputHelper output) { _output = output; }

        private static string FixtureDir
        {
            get
            {
                string root = Environment.GetEnvironmentVariable(DirEnv);
                return root == null ? null : Path.Combine(root, "fixtures");
            }
        }

        [ModelFact(DirEnv)]
        public void ReferenceBlockLayoutMatchesTheUpstreamReference()
        {
            string meta = Path.Combine(FixtureDir ?? ".", "h3_ref_layout.txt");
            string posPath = Path.Combine(FixtureDir ?? ".", "h3_ref_layout_positions.bin");
            if (!File.Exists(meta) || !File.Exists(posPath))
            {
                _output.WriteLine("[h3-layout] fixtures absent; skipping");
                return;
            }

            var fields = File.ReadAllLines(meta)
                .Where(l => l.Contains('='))
                .ToDictionary(l => l.Split('=')[0], l => l.Split('=', 2)[1]);
            int expectedTokens = int.Parse(fields["tokens"], CultureInfo.InvariantCulture);
            float[] expectedTimesteps = fields["timesteps"].Split(',')
                .Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            var expectedSegments = fields["segments"].Split(';')
                .Select(seg => seg.Split(',').Select(v => int.Parse(v, CultureInfo.InvariantCulture)).ToArray())
                .ToArray();

            byte[] posBytes = File.ReadAllBytes(posPath);
            var expectedPositions = new float[posBytes.Length / 4];
            Buffer.BlockCopy(posBytes, 0, expectedPositions, 0, posBytes.Length);

            // The dumped run: a 288x224 reference photo, a 256x256 22-frame clip, and
            // a 74-token prompt whose tokens 6..70 stand in for the reference image.
            var shape = MiniMaxH3Geometry.Resolve(256, 256, 22, MiniMaxH3Geometry.Fps);
            const int textLength = 74, imageIndex = 7, imageTokens = 63;
            var modalities = new List<int>();
            for (int i = 0; i < textLength; i++)
                modalities.Add(i >= imageIndex - 1 && i < imageIndex + imageTokens + 1
                    ? MiniMaxH3Modality.Visual : MiniMaxH3Modality.Text);

            var layout = MiniMaxH3Layout.Build(
                textLength, shape, sigma: 1f,
                textTokenModalities: modalities,
                references: new[]
                {
                    new MiniMaxH3ReferenceBlock
                    {
                        Kind = MiniMaxH3ReferenceKind.Image,
                        VideoIndex = 0,
                        LatentFrames = 1,
                        LatentHeight = 224 / MiniMaxH3Geometry.VaeSpatialRatio,
                        LatentWidth = 288 / MiniMaxH3Geometry.VaeSpatialRatio,
                    },
                });

            _output.WriteLine($"[h3-layout] tokens={layout.TokenCount} (expected {expectedTokens})");
            Assert.Equal(expectedTokens, layout.TokenCount);
            Assert.Equal(expectedTimesteps.Length, layout.Timesteps.Length);
            for (int i = 0; i < expectedTimesteps.Length; i++)
                Assert.Equal(expectedTimesteps[i], layout.Timesteps[i], 5);

            var actualSegments = layout.ModulationSpans
                .Select(s => new[] { s.Start, s.End, s.Row }).ToArray();
            _output.WriteLine("[h3-layout] spans = " +
                string.Join("; ", actualSegments.Select(s => $"{s[0]},{s[1]},{s[2]}")));
            Assert.Equal(expectedSegments.Length, actualSegments.Length);
            for (int i = 0; i < expectedSegments.Length; i++)
                Assert.Equal(expectedSegments[i], actualSegments[i]);

            Assert.Equal(expectedPositions.Length, layout.Positions.Length);
            double maxDiff = 0;
            for (int i = 0; i < expectedPositions.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(expectedPositions[i] - layout.Positions[i]));
            _output.WriteLine($"[h3-layout] max position difference = {maxDiff:E3}");
            Assert.True(maxDiff < 1e-4, $"positions differ by up to {maxDiff:E3}");
        }
    }
}
