// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>Everything a chat renderer is given for one prompt.</summary>
    public sealed record ChatRenderRequest(
        List<ChatMessage> Messages,
        bool AddGenerationPrompt,
        string? Architecture,
        List<ToolFunction>? Tools,
        bool EnableThinking);

    /// <summary>
    /// One model family's CHAT PROTOCOL: how its prompts are framed, what media
    /// placeholders its template expects, and how its replies are parsed back apart.
    ///
    /// These four questions used to be answered by four separate chains of
    /// architecture-name comparisons spread across <see cref="ChatTemplate"/> and
    /// <see cref="OutputParserFactory"/> - about two dozen sites in all - so adding a
    /// family meant finding every one of them, and forgetting one failed quietly
    /// (an unparsed reply streams its own reasoning tags to the client as if they
    /// were the answer; a missing media placeholder silently discards the image).
    /// One entry per family answers all four together.
    ///
    /// Deliberately separate from <c>ModelArchitectureDescriptor</c>: a chat protocol
    /// is a text format, not a model. Several architectures share one (GLM-DSA and
    /// GLM-5.3-Flash), the same protocol serves names that have no loader at all
    /// (<c>qwen3vl</c>), and this assembly must not depend on TensorSharp.Models.
    /// </summary>
    public sealed class ChatProtocol
    {
        /// <summary>Canonical protocol id, for diagnostics.</summary>
        public required string Id { get; init; }

        /// <summary>Every <c>general.architecture</c> string this protocol serves.</summary>
        public required IReadOnlyList<string> Architectures { get; init; }

        /// <summary>
        /// Purpose-built renderer for this family. Null means the generic Qwen3-style
        /// ChatML renderer, which is the fallback for anything unrecognised.
        /// </summary>
        public Func<ChatRenderRequest, string>? Render { get; init; }

        /// <summary>
        /// True when the GGUF-embedded Jinja template must be bypassed in favour of
        /// <see cref="Render"/>. Several families ship templates built on Jinja
        /// features the lightweight engine renders inconsistently (recursive macros,
        /// namespaces, <c>tojson</c>, dict walkers), and their formats are simple
        /// enough to render directly. A predicate rather than a flag because Qwen 3.5
        /// bypasses only when thinking is disabled.
        /// </summary>
        public Func<ChatRenderRequest, bool>? PreferOwnRenderer { get; init; }

        /// <summary>
        /// Append the media placeholder tokens this family's template expects for one
        /// message, before its text content. Null for text-only protocols.
        ///
        /// Getting this wrong is silent: the encoder runs, and the embeddings are
        /// dropped because no placeholder marks where they go.
        /// </summary>
        public Action<ChatMessage, StringBuilder>? AppendMediaPlaceholders { get; init; }

        /// <summary>Parser that turns this family's raw stream back into content,
        /// reasoning and tool calls. Null means the passthrough parser.</summary>
        public Func<IOutputParser>? CreateOutputParser { get; init; }

        /// <summary>
        /// The reply is unreadable without the parser, so it must run even when the
        /// caller did not ask for reasoning or tool calls - the framing tokens and the
        /// whole chain of thought would otherwise be streamed as the answer.
        /// </summary>
        public bool OutputParserAlwaysRequired { get; init; }

        /// <summary>
        /// Text after which a structured-output grammar may start enforcing, or null
        /// when the model's very first token is already part of the answer.
        ///
        /// GPT-OSS opens every reply with a harmony channel header and reasons in the
        /// <c>analysis</c> channel before answering in <c>final</c>. A grammar armed
        /// from token 0 forbids that header, so the model is pushed straight into a
        /// JSON object having done no reasoning and fills the schema with
        /// placeholders. Arming on the final channel's header instead lets it think
        /// and constrains only the answer. See <c>GrammarConstraint.ActivateAfter</c>.
        /// </summary>
        public string? GrammarActivationTrigger { get; init; }

        /// <summary>
        /// Text the GENERATION PROMPT appends after the assistant role marker that
        /// re-rendering the same turn as HISTORY does not reproduce - given the
        /// thinking flag the turn ran under. Returns null/empty when the family's
        /// template frames past and current assistant turns identically.
        ///
        /// This is a KV-cache prefix-reuse fact, and getting it wrong is silent and
        /// expensive: the re-rendered prefix diverges at the first assistant boundary,
        /// every block hash misses, and each multi-turn request re-prefills the whole
        /// conversation at full cost while still answering correctly.
        /// </summary>
        public Func<bool, string?>? AssistantGenerationSuffix { get; init; }

        /// <summary>
        /// True when this family's template emits an EMPTY
        /// <c>&lt;think&gt;&lt;/think&gt;</c> block ahead of a past assistant turn to
        /// say that turn's reasoning was dropped. The KV cache holds no such block -
        /// the original turn forwarded a real <c>&lt;think&gt;</c> plus reasoning
        /// tokens - so it must be removed before the suffix above is injected, or four
        /// spurious tokens land right at the first assistant boundary.
        /// </summary>
        public Func<bool, bool>? EmitsEmptyThinkBlockForPastTurns { get; init; }

        /// <summary>
        /// Marker after which the template emits an assistant HEADER that the raw
        /// generated tokens already carry themselves, or null when the template's
        /// assistant framing matches what the model actually produced.
        ///
        /// Muse-Glimmer is the case that needs this. Its generation prompt is bare -
        /// <c>&lt;|start|&gt;assistant</c> and nothing else - so the model itself emits
        /// the routing header and channel framing that follow. Rendering the SAME turn
        /// again as history takes the template's past-assistant branch, which emits
        /// <c>&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c> before the content;
        /// splicing raw tokens after that header prepends three tokens the cache never
        /// saw, and prefix reuse was 0% for every multi-turn Muse-Glimmer conversation.
        /// </summary>
        public string? TemplateAssistantHeaderAnchor { get; init; }

        /// <summary>
        /// True when this family turns a video into N evenly spaced FRAME images that
        /// each cost a full image's worth of tokens, so a long clip has to be
        /// downsampled to the configured frame cap before it reaches the model.
        /// Families that render a video as a single placeholder token do not.
        /// </summary>
        public bool CapsVideoFrames { get; init; }

        internal void Validate()
        {
            if (string.IsNullOrWhiteSpace(Id))
                throw new InvalidOperationException("Chat protocol has no Id.");
            if (Architectures == null || Architectures.Count == 0)
                throw new InvalidOperationException($"Chat protocol '{Id}' serves no architectures.");
            if (OutputParserAlwaysRequired && CreateOutputParser == null)
            {
                throw new InvalidOperationException(
                    $"Chat protocol '{Id}' says its parser is always required but supplies none.");
            }
        }
    }
}
