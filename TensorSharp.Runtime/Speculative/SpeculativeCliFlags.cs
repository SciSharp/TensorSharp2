// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Globalization;
using System.IO;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Translation of the speculative-decoding command-line flags into the
    /// environment variables <see cref="SpeculationOptions.FromEnvironment"/>
    /// and the model loaders read.
    ///
    /// The env var - not a parsed value object - is the contract here because
    /// the request has to reach places the flags cannot: the glm-dsa NATIVE
    /// loader decides from <c>TS_MTP_SPEC</c> whether to page the NextN block
    /// into VRAM at all (it is a whole extra 256-expert decoder layer), and
    /// sizes its graph cache from <c>TS_MTP_DRAFT</c> - both from C++, while the
    /// model is loading, long before any decoder object exists. Hosts must
    /// therefore apply these BEFORE they construct the model, and every flag is
    /// published under BOTH the current <c>TS_SPEC_*</c> spelling and the
    /// legacy <c>TS_MTP_*</c> one the native loader reads.
    ///
    /// Flag spellings: the algorithm-neutral <c>--spec*</c> names are current;
    /// the <c>--mtp-*</c> names are accepted unchanged so existing scripts and
    /// deployments keep working.
    ///
    /// Shared by <c>TensorSharp.Server</c> and <c>TensorSharp.Cli</c> so the two
    /// hosts cannot drift on flag names, validation or defaults.
    /// </summary>
    public static class SpeculativeCliFlags
    {
        /// <summary>Set to <c>1</c>/<c>0</c> by <c>--spec</c>/<c>--no-spec</c>
        /// (aliases <c>--mtp-spec</c>/<c>--no-mtp-spec</c>).</summary>
        public const string SpecEnvVar = SpeculationEnvVars.LegacyEnabled;

        /// <summary>Maximum tokens drafted per speculative step
        /// (<c>--spec-draft</c>, alias <c>--mtp-draft</c>).</summary>
        public const string DraftEnvVar = SpeculationEnvVars.LegacyDraft;

        /// <summary>Minimum draft confidence to keep a drafted token
        /// (<c>--spec-pmin</c>, alias <c>--mtp-pmin</c>).</summary>
        public const string PMinEnvVar = SpeculationEnvVars.LegacyPMin;

        /// <summary>Separate draft-head GGUF for architectures that ship one
        /// (<c>--spec-draft-model</c>, alias <c>--mtp-draft-model</c>).</summary>
        public const string DraftModelEnvVar = SpeculationEnvVars.LegacyDraftModel;

        /// <summary>Speculation algorithm (<c>--spec-type</c>).</summary>
        public const string TypeEnvVar = SpeculationEnvVars.Type;

        /// <summary>Largest accepted draft window; see
        /// <see cref="SpeculationOptions.MaxAllowedDraftTokens"/>.</summary>
        public const int MaxDraftTokens = SpeculationOptions.MaxAllowedDraftTokens;

        /// <summary>
        /// Apply the speculative-decoding flags from <paramref name="args"/> to
        /// the process environment. Both <c>--opt V</c> and <c>--opt=V</c>
        /// spellings are accepted, and each flag has an algorithm-neutral name
        /// plus its historical <c>--mtp-*</c> alias:
        ///
        /// <code>
        ///   --spec | --no-spec              (--mtp-spec | --no-mtp-spec)
        ///   --spec-type NAME                (new: auto | draft-head | block | ngram)
        ///   --spec-draft N                  (--mtp-draft N)
        ///   --spec-pmin X                   (--mtp-pmin X)
        ///   --spec-draft-model PATH         (--mtp-draft-model PATH)
        /// </code>
        ///
        /// Returns true when at least one flag was applied, so the caller can
        /// emit a startup log line.
        /// </summary>
        /// <exception cref="ArgumentException">A flag carried a missing or unusable value.</exception>
        public static bool Apply(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            bool changed = false;
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                if (IsFlag(a, "--spec", "--mtp-spec"))
                {
                    SetBoth(SpeculationEnvVars.Enabled, SpeculationEnvVars.LegacyEnabled, "1");
                    changed = true;
                    continue;
                }
                if (IsFlag(a, "--no-spec", "--no-mtp-spec"))
                {
                    SetBoth(SpeculationEnvVars.Enabled, SpeculationEnvVars.LegacyEnabled, "0");
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--spec-type", "--mtp-type", out string typeOpt, out string typeFlag))
                {
                    if (!SpeculatorRegistry.IsKnown(typeOpt))
                    {
                        throw new ArgumentException(
                            $"Invalid value for {typeFlag}: '{typeOpt}'. Expected one of: "
                            + $"{SpeculatorRegistry.Auto}, {string.Join(", ", SpeculatorRegistry.Names)}.");
                    }
                    Environment.SetEnvironmentVariable(SpeculationEnvVars.Type, typeOpt.Trim());
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--spec-draft", "--mtp-draft", out string draftOpt, out string draftFlag))
                {
                    if (!int.TryParse(draftOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int draft)
                        || draft < 1 || draft > MaxDraftTokens)
                    {
                        throw new ArgumentException(
                            $"Invalid value for {draftFlag}: '{draftOpt}'. Expected an integer in [1, {MaxDraftTokens}].");
                    }
                    SetBoth(SpeculationEnvVars.Draft, SpeculationEnvVars.LegacyDraft,
                        draft.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--spec-pmin", "--mtp-pmin", out string pminOpt, out string pminFlag))
                {
                    if (!float.TryParse(pminOpt, NumberStyles.Float, CultureInfo.InvariantCulture, out float pmin)
                        || pmin <= 0f || pmin > 1f)
                    {
                        throw new ArgumentException(
                            $"Invalid value for {pminFlag}: '{pminOpt}'. Expected a probability in (0, 1].");
                    }
                    SetBoth(SpeculationEnvVars.PMin, SpeculationEnvVars.LegacyPMin,
                        pmin.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                // Path to a SEPARATE draft GGUF for models whose draft head ships
                // as its own file (Gemma 4's "gemma4-assistant"). Qwen3.6 and
                // GLM-5.2 embed their NextN block in the trunk GGUF and need no
                // such flag.
                if (TryReadOption(args, ref i, "--spec-draft-model", "--mtp-draft-model",
                        out string draftModelOpt, out string draftModelFlag))
                {
                    if (string.IsNullOrWhiteSpace(draftModelOpt) || !File.Exists(draftModelOpt))
                        throw new ArgumentException($"{draftModelFlag} file not found: '{draftModelOpt}'.");
                    SetBoth(SpeculationEnvVars.DraftModel, SpeculationEnvVars.LegacyDraftModel, draftModelOpt);
                    changed = true;
                    continue;
                }
            }
            return changed;
        }

        /// <summary>
        /// Reads <c>--opt VALUE</c> or <c>--opt=VALUE</c> at <paramref name="index"/>,
        /// advancing past a consumed value token.
        /// </summary>
        public static bool TryReadOption(string[] args, ref int index, string option, out string value)
            => TryReadOption(args, ref index, option, null, out value);

        /// <summary>As <see cref="TryReadOption(string[], ref int, string, out string)"/>,
        /// also accepting a historical <paramref name="alias"/> spelling.</summary>
        public static bool TryReadOption(string[] args, ref int index, string option, string alias, out string value)
            => TryReadOption(args, ref index, option, alias, out value, out _);

        /// <summary>
        /// As above, additionally reporting WHICH spelling matched. Diagnostics
        /// quote the flag the operator actually typed: being told
        /// "--spec-draft is invalid" after typing <c>--mtp-draft</c> sends them
        /// looking for a flag they never used.
        /// </summary>
        public static bool TryReadOption(string[] args, ref int index, string option, string alias,
            out string value, out string matchedOption)
        {
            if (TryReadOne(args, ref index, option, out value))
            {
                matchedOption = option;
                return true;
            }
            if (alias != null && TryReadOne(args, ref index, alias, out value))
            {
                matchedOption = alias;
                return true;
            }
            matchedOption = option;
            return false;
        }

        private static bool TryReadOne(string[] args, ref int index, string option, out string value)
        {
            string arg = args[index];
            if (string.Equals(arg, option, StringComparison.OrdinalIgnoreCase))
            {
                if (index + 1 >= args.Length)
                    throw new ArgumentException($"Missing value for option '{option}'.");

                value = args[++index];
                return true;
            }

            string prefix = option + "=";
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                value = arg.Substring(prefix.Length);
                return true;
            }

            value = null;
            return false;
        }

        private static bool IsFlag(string arg, string option, string alias)
            => string.Equals(arg, option, StringComparison.OrdinalIgnoreCase)
               || string.Equals(arg, alias, StringComparison.OrdinalIgnoreCase);

        /// <summary>Publish under both spellings: managed readers prefer
        /// <c>TS_SPEC_*</c>, the glm-dsa native loader only knows
        /// <c>TS_MTP_*</c>.</summary>
        private static void SetBoth(string name, string legacyName, string value)
        {
            Environment.SetEnvironmentVariable(name, value);
            Environment.SetEnvironmentVariable(legacyName, value);
        }
    }
}
