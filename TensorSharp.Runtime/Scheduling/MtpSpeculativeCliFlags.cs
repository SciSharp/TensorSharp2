// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Globalization;
using System.IO;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Translation of the <c>--mtp-*</c> command-line flags into the
    /// <c>TS_MTP_*</c> environment variables that <see cref="SchedulerConfig.FromEnvironment"/>
    /// and the model loaders read.
    ///
    /// The env var is the contract rather than a parsed value object because the
    /// request has to reach places the flags cannot: the glm-dsa native loader
    /// decides from <c>TS_MTP_SPEC</c> whether to page the NextN block into VRAM
    /// at all (it is a whole extra 256-expert decoder layer), and sizes its graph
    /// cache from <c>TS_MTP_DRAFT</c> — both while the model is loading, long
    /// before any decoder object exists. Hosts must therefore apply these BEFORE
    /// they construct the model.
    ///
    /// Shared by <c>TensorSharp.Server</c> and <c>TensorSharp.Cli</c> so the two
    /// hosts cannot drift on flag names, validation or defaults.
    /// </summary>
    public static class MtpSpeculativeCliFlags
    {
        /// <summary>Set to <c>1</c>/<c>0</c> by <c>--mtp-spec</c>/<c>--no-mtp-spec</c>.</summary>
        public const string SpecEnvVar = "TS_MTP_SPEC";

        /// <summary>Maximum tokens drafted per speculative step (<c>--mtp-draft</c>).</summary>
        public const string DraftEnvVar = "TS_MTP_DRAFT";

        /// <summary>Minimum draft confidence to keep a drafted token (<c>--mtp-pmin</c>).</summary>
        public const string PMinEnvVar = "TS_MTP_PMIN";

        /// <summary>Separate MTP draft GGUF for architectures that ship one (<c>--mtp-draft-model</c>).</summary>
        public const string DraftModelEnvVar = "TS_MTP_DRAFT_MODEL";

        /// <summary>
        /// Largest accepted <c>--mtp-draft</c>. Bounded because the window is not
        /// free on either side of the boundary: the draft/verify buffers are
        /// <c>(N+1) x vocab</c> floats (on a 155k-token vocabulary, 40 MB at 64),
        /// and the glm-dsa native loader ignores anything above this when it sizes
        /// its graph cache from the same variable — so a larger window would decode
        /// through a cache too small for the graph shapes it produces, rebuilding
        /// one every step.
        /// </summary>
        public const int MaxDraftTokens = 64;

        /// <summary>
        /// Apply <c>--mtp-spec</c> / <c>--no-mtp-spec</c> / <c>--mtp-draft N</c> /
        /// <c>--mtp-pmin X</c> / <c>--mtp-draft-model PATH</c> from
        /// <paramref name="args"/> to the process environment. Both <c>--opt V</c>
        /// and <c>--opt=V</c> spellings are accepted. Returns true when at least
        /// one flag was applied, so the caller can emit a startup log line.
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
                if (string.Equals(a, "--mtp-spec", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable(SpecEnvVar, "1");
                    changed = true;
                    continue;
                }
                if (string.Equals(a, "--no-mtp-spec", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable(SpecEnvVar, "0");
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--mtp-draft", out string draftOpt))
                {
                    if (!int.TryParse(draftOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int draft)
                        || draft < 1 || draft > MaxDraftTokens)
                    {
                        throw new ArgumentException(
                            $"Invalid value for --mtp-draft: '{draftOpt}'. Expected an integer in [1, {MaxDraftTokens}].");
                    }
                    Environment.SetEnvironmentVariable(DraftEnvVar, draft.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--mtp-pmin", out string pminOpt))
                {
                    if (!float.TryParse(pminOpt, NumberStyles.Float, CultureInfo.InvariantCulture, out float pmin)
                        || pmin <= 0f || pmin > 1f)
                    {
                        throw new ArgumentException($"Invalid value for --mtp-pmin: '{pminOpt}'. Expected a probability in (0, 1].");
                    }
                    Environment.SetEnvironmentVariable(PMinEnvVar, pmin.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                // Path to a SEPARATE draft GGUF for models whose MTP draft head
                // ships as its own file (Gemma 4's "gemma4-assistant"). Qwen3.6
                // and GLM-5.2 embed their NextN block in the trunk GGUF and need
                // no such flag.
                if (TryReadOption(args, ref i, "--mtp-draft-model", out string draftModelOpt))
                {
                    if (string.IsNullOrWhiteSpace(draftModelOpt) || !File.Exists(draftModelOpt))
                        throw new ArgumentException($"--mtp-draft-model file not found: '{draftModelOpt}'.");
                    Environment.SetEnvironmentVariable(DraftModelEnvVar, draftModelOpt);
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
    }
}
