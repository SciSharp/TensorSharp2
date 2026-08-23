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
using System.Globalization;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// The operator's speculative-decoding request, resolved once and passed
    /// down: whether to speculate, with which algorithm, how wide a window and
    /// what confidence gate. Hosts (CLI, server) build one of these; nothing
    /// below reads the environment again.
    /// </summary>
    public sealed record SpeculationOptions
    {
        /// <summary>Speculative decoding requested. Default OFF - a per-token
        /// head is resident in every checkpoint that ships one, so engaging by
        /// its mere presence would silently change what a plain run does.
        /// CLI: <c>--spec</c> / <c>--mtp-spec</c>; env: <c>TS_SPEC</c> /
        /// <c>TS_MTP_SPEC</c>.</summary>
        public bool Enabled { get; init; }

        /// <summary>Which algorithm to use; see <see cref="SpeculatorRegistry"/>.
        /// Default <see cref="SpeculatorRegistry.Auto"/> = "whatever drafter the
        /// checkpoint carries". CLI: <c>--spec-type</c>; env:
        /// <c>TS_SPEC_TYPE</c>.</summary>
        public string SpeculatorName { get; init; } = SpeculatorRegistry.Auto;

        /// <summary>Maximum tokens drafted per speculative step (llama.cpp
        /// n_max). CLI: <c>--spec-draft</c> / <c>--mtp-draft</c>; env:
        /// <c>TS_SPEC_DRAFT</c> / <c>TS_MTP_DRAFT</c>.</summary>
        public int MaxDraftTokens { get; init; } = DefaultMaxDraftTokens;

        /// <summary>
        /// Confidence gate, or null to let the algorithm pick its own
        /// (<see cref="ISpeculator.DefaultMinDraftProb"/>). The gates threshold
        /// DIFFERENT quantities per algorithm, so one shared default cannot
        /// serve them all - leave this unset unless the operator asked for a
        /// specific value. CLI: <c>--spec-pmin</c> / <c>--mtp-pmin</c>; env:
        /// <c>TS_SPEC_PMIN</c> / <c>TS_MTP_PMIN</c>.
        /// </summary>
        public float? MinDraftProb { get; init; }

        /// <summary>Default draft window: 8. A confidence gate stops drafting at
        /// the first low-confidence token, so a longer window only extends
        /// confident streaks - measured 1.21x vs 1.08x (window 4) on
        /// Qwen3.6-35B-A3B ggml_cpu at unchanged 86% acceptance; neutral on 27B
        /// ggml_cuda.</summary>
        public const int DefaultMaxDraftTokens = 8;

        /// <summary>
        /// Largest accepted draft window. Bounded because the window is not
        /// free on either side of the boundary: the draft/verify buffers are
        /// <c>(N+1) x vocab</c> floats (on a 155k-token vocabulary, 40 MB at 64),
        /// and the glm-dsa native loader ignores anything above this when it
        /// sizes its graph cache from the same variable - so a larger window
        /// would decode through a cache too small for the graph shapes it
        /// produces, rebuilding one every step.
        /// </summary>
        public const int MaxAllowedDraftTokens = 64;

        /// <summary>Speculation off.</summary>
        public static SpeculationOptions Disabled => new();

        /// <summary>
        /// Read the <c>TS_SPEC_*</c> environment, falling back to the older
        /// <c>TS_MTP_*</c> spellings. Both are supported for good reason and
        /// not merely for compatibility: the glm-dsa NATIVE loader reads
        /// <c>TS_MTP_SPEC</c> and <c>TS_MTP_DRAFT</c> from C++ while the model
        /// is loading (it decides whether to page a whole extra 256-expert
        /// decoder layer into VRAM, and sizes its graph cache), so those names
        /// are a cross-language contract that cannot simply be renamed. Hosts
        /// write BOTH spellings; readers accept either.
        /// </summary>
        public static SpeculationOptions FromEnvironment()
        {
            return new SpeculationOptions
            {
                Enabled = ReadBool(SpeculationEnvVars.Enabled, SpeculationEnvVars.LegacyEnabled, false),
                SpeculatorName = ReadString(SpeculationEnvVars.Type, null) ?? SpeculatorRegistry.Auto,
                MaxDraftTokens = ReadPositiveInt(SpeculationEnvVars.Draft, SpeculationEnvVars.LegacyDraft,
                    DefaultMaxDraftTokens),
                MinDraftProb = ReadFloatOrNull(SpeculationEnvVars.PMin, SpeculationEnvVars.LegacyPMin),
            };
        }

        private static string ReadString(string name, string fallbackName)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw) && fallbackName != null)
                raw = Environment.GetEnvironmentVariable(fallbackName);
            return string.IsNullOrWhiteSpace(raw) ? null : raw.Trim();
        }

        private static bool ReadBool(string name, string fallbackName, bool fallback)
        {
            string raw = ReadString(name, fallbackName);
            if (raw == null)
                return fallback;
            return raw is "1" or "true" or "TRUE" or "True" or "yes" or "on";
        }

        private static int ReadPositiveInt(string name, string fallbackName, int fallback)
        {
            string raw = ReadString(name, fallbackName);
            return raw != null
                   && int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
                   && v > 0
                ? v
                : fallback;
        }

        private static float? ReadFloatOrNull(string name, string fallbackName)
        {
            string raw = ReadString(name, fallbackName);
            return raw != null
                   && float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float v)
                   && v > 0f && v <= 1f
                ? v
                : null;
        }
    }

    /// <summary>
    /// The environment-variable contract. Named constants rather than string
    /// literals because these are read from THREE languages/layers - managed
    /// hosts, the model loaders, and the glm-dsa native C++ loader - and a typo
    /// in one of them shows up only as speculation silently not engaging.
    /// </summary>
    public static class SpeculationEnvVars
    {
        /// <summary>Preferred: <c>1</c>/<c>0</c>, set by <c>--spec</c> / <c>--no-spec</c>.</summary>
        public const string Enabled = "TS_SPEC";

        /// <summary>Algorithm name (see <see cref="SpeculatorRegistry"/>).</summary>
        public const string Type = "TS_SPEC_TYPE";

        /// <summary>Maximum tokens drafted per speculative step.</summary>
        public const string Draft = "TS_SPEC_DRAFT";

        /// <summary>Draft-confidence gate.</summary>
        public const string PMin = "TS_SPEC_PMIN";

        /// <summary>Separate draft-head GGUF for architectures that ship one.</summary>
        public const string DraftModel = "TS_SPEC_DRAFT_MODEL";

        /// <summary>Legacy spelling, ALSO read by the glm-dsa native loader.</summary>
        public const string LegacyEnabled = "TS_MTP_SPEC";

        /// <summary>Legacy spelling, ALSO read by the glm-dsa native loader
        /// (it sizes its graph cache from this).</summary>
        public const string LegacyDraft = "TS_MTP_DRAFT";

        /// <summary>Legacy spelling.</summary>
        public const string LegacyPMin = "TS_MTP_PMIN";

        /// <summary>Legacy spelling.</summary>
        public const string LegacyDraftModel = "TS_MTP_DRAFT_MODEL";
    }
}
