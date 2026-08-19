// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Activation tracing for cross-implementation debugging.
//
// Set TS_GLM_DEBUG=1 to print, for a chosen set of layers, the shape / sum /
// leading values of every named intermediate. The tags line up with llama.cpp's
// graph callback names (attn_norm, qr, q, kv_cmpr, k_pe, kqv_out, ffn_inp,
// ffn_norm, ffn_out, l_out, result_norm, result_output), so
// `llama-eval-callback` output and this trace can be diffed tag by tag to find
// the first layer where two implementations disagree.
//
//   TS_GLM_DEBUG=1                    trace layer 0 only
//   TS_GLM_DEBUG_LAYERS=0,1,77        trace these layers
//   TS_GLM_DEBUG_LAYERS=all           trace every layer
using System;
using System.Collections.Generic;
using System.Globalization;
using TensorSharp;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        private static readonly bool DebugTrace =
            Environment.GetEnvironmentVariable("TS_GLM_DEBUG") == "1" ||
            !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_GLM_DEBUG_LAYERS"));

        private static readonly HashSet<int> DebugLayers = ResolveDebugLayers();
        private static readonly bool DebugAllLayers =
            string.Equals(Environment.GetEnvironmentVariable("TS_GLM_DEBUG_LAYERS"), "all", StringComparison.OrdinalIgnoreCase);

        private static HashSet<int> ResolveDebugLayers()
        {
            var set = new HashSet<int>();
            string env = Environment.GetEnvironmentVariable("TS_GLM_DEBUG_LAYERS");
            if (string.IsNullOrEmpty(env))
            {
                set.Add(0);
                return set;
            }
            foreach (var part in env.Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                if (int.TryParse(part.Trim(), out int v))
                    set.Add(v);
            }
            return set;
        }

        private bool DebugWants(int layer) => DebugTrace && (layer < 0 || DebugAllLayers || DebugLayers.Contains(layer));

        private unsafe void Trace(string tag, int layer, Tensor t, int count = 3)
        {
            if (!DebugWants(layer) || t == null)
                return;

            InvalidateTensorDeviceCache(t);
            long n = t.ElementCount();
            float* p = GetFloatPtr(t);

            double sum = 0;
            for (long i = 0; i < n; i++) sum += p[i];

            string shape = string.Join(", ", t.Sizes.ToArray());
            string name = layer >= 0 ? $"{tag}-{layer}" : tag;
            Console.WriteLine($"[glm-trace] {name,-24} ne = [{shape}]  sum = {sum:F6}");

            // Row-wise heads/tails, matching llama-eval-callback's layout so the two
            // traces line up element by element and not just on the whole-tensor sum.
            long rowLen = t.Sizes[t.Sizes.Length - 1];
            long rows = n / Math.Max(1, rowLen);
            for (long r = 0; r < Math.Min(rows, 5); r++)
            {
                float* row = p + r * rowLen;
                var parts = new List<string>();
                for (int i = 0; i < Math.Min(count, rowLen); i++)
                    parts.Add(row[i].ToString("F6", CultureInfo.InvariantCulture));
                parts.Add("...");
                for (long i = Math.Max(0, rowLen - 3); i < rowLen; i++)
                    parts.Add(row[i].ToString("F6", CultureInfo.InvariantCulture));
                Console.WriteLine($"[glm-trace]     row {r}: [{string.Join(", ", parts)}]");
            }
        }
    }
}
