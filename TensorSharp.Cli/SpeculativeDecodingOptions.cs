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
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Cli
{
    /// <summary>
    /// The CLI's speculative-decoding policy: one place where every generation
    /// path (single-shot, JSONL batch, multi-turn, interactive) decides whether a
    /// turn decodes through a draft head, with what window, and — when it cannot —
    /// why not.
    ///
    /// Two kinds of drafter reach this, and they are opted into differently:
    ///
    ///   * a BLOCK drafter that ships as its own GGUF (DeepSeek V4's DSpark,
    ///     Muse-Glimmer's DFlash) is requested by naming the file on
    ///     <c>--draft-model</c>, so its presence in the weights IS the request;
    ///   * a PER-TOKEN NextN/MTP head embedded in the trunk checkpoint (GLM-5.2,
    ///     Qwen 3.6) is always there once the checkpoint is loaded, so it needs an
    ///     explicit <c>--mtp-spec</c> — matching the server's opt-in default, and
    ///     for GLM also matching what the loader was told: the native loader only
    ///     pages the ~3 GiB draft layer into VRAM when <c>TS_MTP_SPEC</c> was
    ///     already set, so a request that arrives after the model is loaded cannot
    ///     be honoured at all.
    /// </summary>
    internal static class SpeculativeDecodingOptions
    {
        /// <summary>Resolved knobs for a run, from the CLI flags and the TS_MTP_* environment.</summary>
        internal readonly struct Settings
        {
            /// <summary>True when <c>--mtp-spec</c> (or <c>TS_MTP_SPEC=1</c>) asked for
            /// speculation on a drafter that is not itself an explicit request.</summary>
            public bool Requested { get; init; }

            /// <summary>Cap on tokens drafted per step. Always positive.</summary>
            public int MaxDraftTokens { get; init; }

            /// <summary>Draft-confidence gate, or null to let the drafter apply its own
            /// default — 0.75 for a per-token head, 0.35 for a block drafter. The two
            /// threshold different quantities, so there is no shared default to fall
            /// back on and "unset" has to survive all the way to the executor.</summary>
            public float? MinDraftProb { get; init; }

            /// <summary>Whether anything was explicitly configured, for the startup log.</summary>
            public bool AnyExplicit { get; init; }
        }

        /// <summary>
        /// Publish the effective draft window to <c>TS_MTP_DRAFT</c> before the model
        /// is created, so the older <c>--spec-draft-n-max</c> spelling reaches the
        /// places only the environment can reach.
        ///
        /// <see cref="MtpSpeculativeCliFlags.Apply"/> already did this for
        /// <c>--mtp-draft</c>, at the top of Main; <c>--spec-draft-n-max</c> is parsed
        /// later, in the main argument switch, and would otherwise resize the decoder
        /// without resizing what the LOADER sizes from the same number — the glm-dsa
        /// native graph cache, which holds one entry per live graph shape and gets
        /// <c>8 + 2*(N+1)</c> of them. Left at the default 8, a
        /// <c>--spec-draft-n-max 16</c> run rebuilds and re-allocates a graph it had a
        /// moment ago on every step, and measures slower than plain decoding for a
        /// reason nothing in the log explains.
        /// </summary>
        internal static void PublishDraftWindow(int specDraftMax)
        {
            if (specDraftMax <= 0)
                return;
            if (specDraftMax > MtpSpeculativeCliFlags.MaxDraftTokens)
            {
                throw new ArgumentException(
                    $"Invalid value for --spec-draft-n-max: '{specDraftMax}'. "
                    + $"Expected an integer in [1, {MtpSpeculativeCliFlags.MaxDraftTokens}].");
            }
            Environment.SetEnvironmentVariable(
                MtpSpeculativeCliFlags.DraftEnvVar,
                specDraftMax.ToString(CultureInfo.InvariantCulture));
        }

        /// <summary>
        /// Combine the TS_MTP_* environment (already carrying whatever
        /// <see cref="MtpSpeculativeCliFlags.Apply"/> translated from the command
        /// line) with the CLI's own <c>--spec-draft-*</c> flags, which win when set
        /// because they are the more specific spelling.
        /// </summary>
        internal static Settings Resolve(int specDraftMax, float specDraftConfMin)
        {
            var cfg = SchedulerConfig.FromEnvironment();
            return new Settings
            {
                Requested = cfg.MtpSpeculativeEnabled,
                MaxDraftTokens = specDraftMax > 0 ? specDraftMax : Math.Max(1, cfg.MtpMaxDraftTokens),
                MinDraftProb = specDraftConfMin >= 0f ? specDraftConfMin : cfg.MtpMinDraftProb,
                AnyExplicit = cfg.MtpSpeculativeEnabled || specDraftMax > 0 || specDraftConfMin >= 0f
                              || cfg.MtpMinDraftProb.HasValue,
            };
        }

        /// <summary>
        /// The decoder to serve a turn with, or null to decode one token per forward.
        /// <paramref name="declineReason"/> is set (for an operator-facing warning)
        /// only when speculation was ASKED for and could not be given; a model that
        /// simply has no drafter declines silently.
        ///
        /// The run's sampler is deliberately not a factor: verification draws every
        /// emitted token from a trunk row with whatever sampler the caller then
        /// passes to <see cref="MtpSpeculativeDecoder.GenerateSampled"/>, so a
        /// temperature or a penalty changes how the tokens are drawn, not whether
        /// speculation is sound.
        /// </summary>
        /// <param name="existing">A decoder already built for this model, reused
        /// rather than rebuilt. It carries the hidden state pairing the trunk with
        /// the drafter across turns, and its buffers are sized by the vocabulary —
        /// on a 155k-token vocabulary a rebuilt one costs several MB per turn for
        /// nothing.</param>
        internal static MtpSpeculativeDecoder TryCreate(
            ModelBase model, in Settings settings,
            bool hasMediaAttachments, out string declineReason,
            MtpSpeculativeDecoder existing = null)
        {
            declineReason = null;

            if (model is not IMtpSpeculativeModel spec)
                return null;

            bool blockDrafter = spec.MtpDraftBlockSize > 0;
            // A per-token head that nobody asked for stays off: it is resident in
            // every checkpoint that ships one, so engaging by its mere presence
            // would silently change what `--input` on a Qwen 3.6 or GLM model does.
            if (!blockDrafter && !settings.Requested)
                return null;

            if (!spec.HasMtp)
            {
                // The loader is the authority here, and it has already explained
                // itself on stderr (a trunk-only requantization, a column-parallel
                // borrowed head under --tp, no room on the device). Name the
                // symptom and point at the log rather than guessing the cause.
                declineReason = "the loaded checkpoint exposes no NextN/MTP draft block "
                              + "(see the loader's message above; a trunk-only requantization and "
                              + "--tp with a borrowed LM head both land here)";
                return null;
            }

            // Backends whose accelerated verify/draft kernels are missing run the
            // per-op fallback, which does not amortize the trunk over the window.
            if (!spec.MtpSpeculationProfitable)
            {
                declineReason = "the draft head has no accelerated path on this backend, "
                              + "where speculation costs more than it saves";
                return null;
            }

            // The speculative prefill has no place to queue per-chunk vision/audio
            // embeddings, which are injected by ModelBase.Forward's hook.
            if (hasMediaAttachments)
            {
                declineReason = "the turn carries image/audio/video attachments, "
                              + "whose embeddings only the plain prefill can inject";
                return null;
            }

            if (existing != null)
                return existing;

            // A block drafter's window can never exceed the block it was trained to
            // emit; a per-token head chains one step at a time and is bounded only
            // by --mtp-draft.
            int window = blockDrafter
                ? Math.Min(settings.MaxDraftTokens, spec.MtpDraftBlockSize)
                : settings.MaxDraftTokens;
            if (window < 1)
                window = 1;

            var decoder = new MtpSpeculativeDecoder(spec, window)
            {
                PrefillChunkSize = spec.MtpPrefillChunkSize > 0 ? spec.MtpPrefillChunkSize : 512,
            };
            if (settings.MinDraftProb.HasValue)
                decoder.MinDraftProb = settings.MinDraftProb.Value;
            return decoder;
        }

        /// <summary>How the drafter proposes tokens, for logs.</summary>
        internal static string DescribeDrafter(IMtpSpeculativeModel spec)
            => spec.MtpDraftBlockSize > 0 ? $"block({spec.MtpDraftBlockSize})" : "per-token";

        /// <summary>
        /// How a turn verifies: argmax keeps the greedy stream exactly, anything
        /// else draws each emitted token with the run's own sampler.
        /// </summary>
        internal static string DescribeVerification(SamplingConfig sampling)
            => InteractiveSession.IsArgmaxSampling(sampling) ? "argmax" : "sampled";
    }
}
