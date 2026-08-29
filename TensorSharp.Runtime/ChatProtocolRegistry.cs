// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// The chat protocols this process knows, keyed by architecture name.
    ///
    /// THIS TABLE IS THE ONLY PLACE A NEW MODEL FAMILY'S TEXT FORMAT IS DECLARED.
    /// Prompt framing, GGUF-template bypass, media placeholders, reply parsing and
    /// grammar arming all read from the same entry, so a family cannot be half-added
    /// the way it could when those five things lived in five separate name chains.
    ///
    /// <see cref="Register"/> is public so a host can add a protocol without forking.
    /// </summary>
    public static class ChatProtocolRegistry
    {
        private static readonly object Gate = new();
        private static readonly Dictionary<string, ChatProtocol> ByArchitecture =
            new(StringComparer.OrdinalIgnoreCase);
        private static readonly List<ChatProtocol> Ordered = new();

        static ChatProtocolRegistry() => RegisterBuiltIns();

        public static void Register(ChatProtocol protocol)
        {
            ArgumentNullException.ThrowIfNull(protocol);
            protocol.Validate();

            lock (Gate)
            {
                foreach (string arch in protocol.Architectures)
                {
                    if (ByArchitecture.TryGetValue(arch, out var existing) && !ReferenceEquals(existing, protocol))
                    {
                        throw new InvalidOperationException(
                            $"Architecture '{arch}' already uses chat protocol '{existing.Id}'; " +
                            $"'{protocol.Id}' cannot claim it too.");
                    }
                }

                if (Ordered.Contains(protocol))
                    return;

                foreach (string arch in protocol.Architectures)
                    ByArchitecture[arch] = protocol;
                Ordered.Add(protocol);
            }
        }

        /// <summary>All registered protocols, in registration order.</summary>
        public static IReadOnlyList<ChatProtocol> All
        {
            get { lock (Gate) return Ordered.ToArray(); }
        }

        /// <summary>The protocol for an architecture, or null when it has none and the
        /// generic ChatML path applies.</summary>
        public static ChatProtocol? For(string? architecture)
        {
            if (string.IsNullOrEmpty(architecture))
                return null;
            lock (Gate)
                return ByArchitecture.TryGetValue(architecture, out var protocol) ? protocol : null;
        }

        private static void RegisterBuiltIns()
        {
            // ---- Gemma ------------------------------------------------------
            Register(new ChatProtocol
            {
                Id = "gemma3",
                Architectures = new[] { "gemma3" },
                Render = r => ChatTemplate.RenderGemma3(r.Messages, r.AddGenerationPrompt),
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("<start_of_image>");
                },
            });

            Register(new ChatProtocol
            {
                Id = "gemma4",
                Architectures = new[] { "gemma4" },
                Render = r => ChatTemplate.RenderGemma4(r.Messages, r.AddGenerationPrompt, r.Tools, r.EnableThinking),
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    if (msg.IsVideo && msg.ImagePaths != null)
                        sb.Append("<|video>");
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("<|image>");
                    if (msg.AudioPaths != null)
                        foreach (var _ in msg.AudioPaths) sb.Append("<|audio>");
                },
                CapsVideoFrames = true,
                CreateOutputParser = () => new Gemma4OutputParser(),
                OutputParserAlwaysRequired = true,
                // Thinking-disabled adds an empty <|channel>thought<channel|> block to
                // the generation prompt so the model skips reasoning; the template does
                // not re-emit it for past assistant messages, but the cache holds it.
                AssistantGenerationSuffix = thinking => thinking ? null : "<|channel>thought\n<channel|>",
            });

            // ---- Qwen -------------------------------------------------------
            Register(new ChatProtocol
            {
                Id = "qwen3",
                Architectures = new[] { "qwen3" },
                CreateOutputParser = () => new Qwen3OutputParser(),
            });

            Register(new ChatProtocol
            {
                Id = "qwen35",
                Architectures = new[] { "qwen35", "qwen35moe", "qwen3next", "qwen3vl", "qwen3vlmoe" },
                Render = r => ChatTemplate.RenderQwen35(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                // With thinking ON the GGUF template is used as shipped; with it OFF the
                // purpose-built renderer is the only one that suppresses the block
                // correctly.
                PreferOwnRenderer = r => !r.EnableThinking,
                AppendMediaPlaceholders = AppendQwenVisionPads,
                CreateOutputParser = () => new Qwen35OutputParser(),
                // With thinking ENABLED the Jinja template emits `<think>\n` after the
                // assistant role marker as part of the generation prompt, and does NOT
                // re-emit `<think>...</think>` framing for PAST assistant messages.
                // Without this the cache's `<think>` token has no counterpart in the
                // next turn's render and every multi-turn request resets the cache.
                // With thinking DISABLED the purpose-built renderer already emits
                // `<think>\n\n</think>\n\n` for past turns, so nothing is needed.
                AssistantGenerationSuffix = thinking => thinking ? "<think>\n" : null,
                EmitsEmptyThinkBlockForPastTurns = thinking => thinking,
            });

            // Qwen3.8-Flash-Next frames chat exactly like Qwen 3 (the generic ChatML
            // renderer) but carries the Qwen-VL vision placeholders.
            Register(new ChatProtocol
            {
                Id = "qwen4exp",
                Architectures = new[] { "qwen4exp" },
                // qwen4exp appends `<think>` to the generation prompt UNCONDITIONALLY -
                // the model always reasons - so the cache always holds it, whatever the
                // thinking flag says.
                AssistantGenerationSuffix = _ => "<think>\n",
                EmitsEmptyThinkBlockForPastTurns = _ => true,
                AppendMediaPlaceholders = AppendQwenVisionPads,
            });

            // ---- GPT-OSS / Harmony -----------------------------------------
            Register(new ChatProtocol
            {
                Id = "harmony",
                Architectures = new[] { "gptoss", "gpt-oss" },
                Render = r => ChatTemplate.RenderHarmony(r.Messages, r.AddGenerationPrompt, r.Tools, r.EnableThinking),
                // The embedded template relies on recursive macros, namespace(),
                // strftime_now and list slicing - especially on the tool-rendering path
                // - which the lightweight Jinja engine does not fully support.
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new HarmonyOutputParser(),
                OutputParserAlwaysRequired = true,
                GrammarActivationTrigger = "final<|message|>",
            });

            // ---- Others -----------------------------------------------------
            Register(new ChatProtocol
            {
                Id = "muse-glimmer",
                Architectures = new[] { "muse-glimmer", "muse_glimmer" },
                TemplateAssistantHeaderAnchor = "<|start|>assistant",
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    // The GGUF Jinja template renders an image content part as a single
                    // <|patch|> and a video part as <|video|>. The host later expands
                    // each <|patch|> into <|image_start|> + N filler rows +
                    // <|image_end|>, matching llama.cpp's mtmd chunking for
                    // PROJECTOR_TYPE_MUSE_GLIMMER.
                    if (msg.IsVideo && msg.ImagePaths != null)
                        sb.Append("<|video|>");
                    else if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("<|patch|>");
                },
                CreateOutputParser = () => new MuseGlimmerOutputParser(),
                // Every assistant message is wrapped in <|start|>...<|message|>...
                // <|eom|>/<|eot|> framing and its reasoning arrives on the "to=self"
                // channel, so an unparsed stream shows the raw tags and the whole chain
                // of thought as if it were the answer.
                OutputParserAlwaysRequired = true,
            });

            Register(new ChatProtocol
            {
                Id = "deepseek4",
                Architectures = new[] { "deepseek4" },
                Render = r => ChatTemplate.RenderDeepSeek4(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                // The GGUF-embedded (Unsloth) template leans on Jinja features the
                // lightweight engine handles inconsistently (nested namespaces,
                // from_json, dict.items()); the format itself is simple.
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new DeepSeek4OutputParser(),
                // Its reasoning block and its DSML tool calls both arrive as plain text:
                // without the parser the </think> marker and the whole
                // <｜DSML｜tool_calls> block would be streamed to the client as if they
                // were the answer.
                OutputParserAlwaysRequired = true,
            });

            Register(new ChatProtocol
            {
                Id = "glm-dsa",
                Architectures = new[] { "glm-dsa", "glm_dsa" },
                Render = r => ChatTemplate.RenderGlmDsa(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                // The shipped template is built out of macros, namespaces, tojson and a
                // visible_text walker over structured content - the exact feature set
                // the lightweight Jinja engine renders inconsistently.
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new GlmDsaOutputParser(),
                OutputParserAlwaysRequired = true,
            });

            Register(new ChatProtocol
            {
                Id = "glm5next",
                Architectures = new[] { "glm5next" },
                // GLM-5.3-Flash ALWAYS opens a <think> block in the generation prompt
                // (its template has no thinking-off shape), with no newline after it.
                // Re-rendered history goes through the template's empty-<think></think>
                // branch; stripping that restores the half the cache actually holds.
                AssistantGenerationSuffix = _ => "<think>",
                EmitsEmptyThinkBlockForPastTurns = _ => true,
                Render = r => ChatTemplate.RenderGlm5Next(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    // The template's emit_image() macro. The host later expands the
                    // single <|image|> into N placeholder tokens matching the merged
                    // patch count.
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths)
                            sb.Append("<|begin_of_image|><|image|><|end_of_image|>");
                },
                CreateOutputParser = () => new GlmDsaOutputParser(),
                OutputParserAlwaysRequired = true,
            });

            Register(new ChatProtocol
            {
                Id = "nemotron_h",
                Architectures = new[] { "nemotron_h", "nemotron_h_moe", "nemotron_h_omni" },
                Render = r => ChatTemplate.RenderNemotron(r.Messages, r.AddGenerationPrompt, r.Tools, r.EnableThinking),
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new Qwen3OutputParser(),
            });

            Register(new ChatProtocol
            {
                Id = "mistral3",
                Architectures = new[] { "mistral3" },
                Render = r => ChatTemplate.RenderMistral3(r.Messages, r.AddGenerationPrompt),
                PreferOwnRenderer = _ => true,
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("[IMG]");
                },
            });
        }

        private static void AppendQwenVisionPads(ChatMessage msg, System.Text.StringBuilder sb)
        {
            if (msg.ImagePaths != null)
                foreach (var _ in msg.ImagePaths)
                    sb.Append("<|vision_start|><|image_pad|><|vision_end|>");
        }
    }
}
