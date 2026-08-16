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
using System.Collections.Generic;
using System.Globalization;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// The sampling parameters a model author shipped inside the GGUF under the
    /// <c>general.sampling.*</c> keys. Every field is nullable: only the keys the
    /// file actually carries are populated, so applying this over a config leaves
    /// everything else alone.
    ///
    /// llama.cpp registers the same key set (src/llama-arch.cpp) and layers it over
    /// its built-in defaults for every field the operator did not pin
    /// (common/common.cpp). TensorSharp read none of them, so a GGUF that ships
    /// "use top_k=20, top_p=0.95" was silently sampled with whatever the entry
    /// point happened to default to.
    /// </summary>
    public sealed class RecommendedSampling
    {
        public float? Temperature { get; init; }
        public int? TopK { get; init; }
        public float? TopP { get; init; }
        public float? MinP { get; init; }
        public int? PenaltyLastN { get; init; }
        public float? RepetitionPenalty { get; init; }

        public bool HasAny =>
            Temperature.HasValue || TopK.HasValue || TopP.HasValue ||
            MinP.HasValue || PenaltyLastN.HasValue || RepetitionPenalty.HasValue;

        /// <summary>
        /// Reads the <c>general.sampling.*</c> keys out of GGUF metadata.
        /// Returns null when the file carries none of them.
        /// </summary>
        public static RecommendedSampling FromGgufMetadata(IReadOnlyDictionary<string, object> metadata)
        {
            if (metadata == null || metadata.Count == 0)
                return null;

            var result = new RecommendedSampling
            {
                Temperature = ReadFloat(metadata, "general.sampling.temp"),
                TopK = ReadInt(metadata, "general.sampling.top_k"),
                TopP = ReadFloat(metadata, "general.sampling.top_p"),
                MinP = ReadFloat(metadata, "general.sampling.min_p"),
                PenaltyLastN = ReadInt(metadata, "general.sampling.penalty_last_n"),
                RepetitionPenalty = ReadFloat(metadata, "general.sampling.penalty_repeat"),
            };
            return result.HasAny ? result : null;
        }

        /// <summary>
        /// Overlays these recommendations onto <paramref name="config"/>, skipping
        /// any field named in <paramref name="pinned"/> (fields the operator set
        /// explicitly). Returns a human-readable summary of what was applied, or
        /// an empty string when nothing changed.
        /// </summary>
        public string ApplyTo(SamplingConfig config, SamplingFields pinned)
        {
            if (config == null) return string.Empty;
            var applied = new List<string>(6);

            if (Temperature.HasValue && !pinned.HasFlag(SamplingFields.Temperature))
            { config.Temperature = Temperature.Value; applied.Add($"temp={Fmt(Temperature.Value)}"); }
            if (TopK.HasValue && !pinned.HasFlag(SamplingFields.TopK))
            { config.TopK = TopK.Value; applied.Add($"top_k={TopK.Value}"); }
            if (TopP.HasValue && !pinned.HasFlag(SamplingFields.TopP))
            { config.TopP = TopP.Value; applied.Add($"top_p={Fmt(TopP.Value)}"); }
            if (MinP.HasValue && !pinned.HasFlag(SamplingFields.MinP))
            { config.MinP = MinP.Value; applied.Add($"min_p={Fmt(MinP.Value)}"); }
            if (PenaltyLastN.HasValue && !pinned.HasFlag(SamplingFields.PenaltyLastN))
            { config.PenaltyLastN = PenaltyLastN.Value; applied.Add($"penalty_last_n={PenaltyLastN.Value}"); }
            if (RepetitionPenalty.HasValue && !pinned.HasFlag(SamplingFields.RepetitionPenalty))
            { config.RepetitionPenalty = RepetitionPenalty.Value; applied.Add($"repeat_penalty={Fmt(RepetitionPenalty.Value)}"); }

            return applied.Count == 0 ? string.Empty : string.Join(" ", applied);
        }

        private static string Fmt(float v) => v.ToString("0.###", CultureInfo.InvariantCulture);

        private static float? ReadFloat(IReadOnlyDictionary<string, object> md, string key)
        {
            if (!md.TryGetValue(key, out var v) || v == null) return null;
            try
            {
                if (v is float[] fa) return fa.Length > 0 ? fa[0] : (float?)null;
                if (v is double[] da) return da.Length > 0 ? (float)da[0] : (float?)null;
                return Convert.ToSingle(v, CultureInfo.InvariantCulture);
            }
            catch (Exception e) when (e is InvalidCastException or FormatException or OverflowException)
            {
                return null;
            }
        }

        private static int? ReadInt(IReadOnlyDictionary<string, object> md, string key)
        {
            if (!md.TryGetValue(key, out var v) || v == null) return null;
            try
            {
                if (v is int[] ia) return ia.Length > 0 ? ia[0] : (int?)null;
                if (v is uint[] ua) return ua.Length > 0 ? (int)ua[0] : (int?)null;
                return Convert.ToInt32(v, CultureInfo.InvariantCulture);
            }
            catch (Exception e) when (e is InvalidCastException or FormatException or OverflowException)
            {
                return null;
            }
        }
    }

    /// <summary>
    /// Which sampling fields the operator pinned explicitly (on the command line
    /// or in a request), so model- and mode-derived defaults leave them alone.
    /// </summary>
    [Flags]
    public enum SamplingFields
    {
        None = 0,
        Temperature = 1 << 0,
        TopK = 1 << 1,
        TopP = 1 << 2,
        MinP = 1 << 3,
        RepetitionPenalty = 1 << 4,
        PenaltyLastN = 1 << 5,
        PresencePenalty = 1 << 6,
        FrequencyPenalty = 1 << 7,
    }
}
