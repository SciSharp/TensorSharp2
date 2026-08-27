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

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Renders a chat history to a token sequence in a way that is COMPATIBLE with
    /// per-turn KV cache reuse.
    ///
    /// Key invariant: when an assistant message has <see cref="ChatMessage.RawOutputTokens"/>
    /// set (i.e. the model previously generated this turn), those raw tokens are spliced
    /// directly into the rendered token sequence INSTEAD OF re-tokenizing the assistant's
    /// content text.
    ///
    /// Why this matters: assistant content is typically lossy with respect to raw
    /// generation. Thinking-style models emit <c>&lt;think&gt;...&lt;/think&gt;</c> tokens
    /// that the output parser strips out of <see cref="ChatMessage.Content"/>. Harmony /
    /// channel-based models emit channel framing tokens that the parser collapses into
    /// natural-language output. Even for "plain" models the BPE tokenizer can pick a
    /// different encoding for the same text when context changes (whitespace handling,
    /// re-merged tokens, etc.). All of these effects make naive re-rendering produce
    /// tokens that DIVERGE from what's in the KV cache after a few positions, which kills
    /// the cache hit rate.
    ///
    /// By splicing raw tokens we guarantee that the prefix of the new token sequence
    /// EXACTLY matches the cached tokens for as many tokens as the model has already
    /// produced - which is precisely what we want.
    ///
    /// Implementation strategy:
    ///   1. Replace each cached assistant message's content with a unique placeholder
    ///      string (a Private-Use-Area Unicode character + counter).
    ///   2. Render the chat template normally (text-level).
    ///   3. Walk the rendered text segment by segment, splitting on placeholders.
    ///   4. Tokenize each text segment using the model's tokenizer.
    ///   5. For each placeholder boundary, splice the corresponding raw tokens.
    ///
    /// Because placeholders are surrounded by structural tokens (e.g. an
    /// <c>&lt;|im_start|&gt;assistant\n...&lt;|im_end|&gt;</c> framing in Qwen, a
    /// <c>&lt;|turn&gt;model\n...&lt;turn|&gt;</c> framing in Gemma, etc.), each text
    /// segment is independently tokenizable: the BPE pretokenizer always splits at
    /// whitespace / newline / punctuation, so the encoder gives the same token sequence
    /// whether the segment is encoded alone or as part of the whole prompt.
    ///
    /// Only the first segment encodes BOS; subsequent segments use addSpecial=false so we
    /// never accidentally inject extra special tokens between turns.
    /// </summary>
    public sealed class KVCachePromptRenderer
    {
        // U+E000 is in the Private Use Area: real text never contains it. Each cached
        // assistant message gets a numbered placeholder so we can locate raw tokens in
        // order even if the chat template duplicates content (it doesn't today, but
        // numbering is cheap and defensive).
        internal const char PlaceholderSentinel = '\uE000';

        private readonly IPromptRenderer _innerRenderer;

        public KVCachePromptRenderer(IPromptRenderer innerRenderer)
        {
            _innerRenderer = innerRenderer ?? throw new ArgumentNullException(nameof(innerRenderer));
        }

        /// <summary>
        /// Returns the text suffix that the chat template appends AFTER the assistant
        /// role marker but BEFORE the model's generated content, for the given architecture
        /// and thinking mode.
        ///
        /// During the first turn this suffix is part of the rendered prompt (it ends the
        /// "generation prompt"). On subsequent turns we therefore need the renderer to
        /// re-emit this text BEFORE the spliced raw tokens of cached assistant messages,
        /// otherwise the re-rendered token sequence will diverge from what's in the KV
        /// cache (cache has the suffix, naive re-render does not).
        ///
        /// Returns an empty string for architectures whose chat templates already emit
        /// the suffix as part of the standard assistant-message framing.
        /// </summary>
        internal static string GetAssistantGenerationSuffix(string architecture, bool enableThinking)
        {
            if (string.IsNullOrEmpty(architecture))
                return string.Empty;

            // Gemma 4 thinking-disabled adds an empty <|channel>thought<channel|> block
            // to the generation prompt so the model skips reasoning. The standard chat
            // template doesn't re-emit this for past assistant messages, but we need it
            // to match what's in the KV cache.
            if (architecture == "gemma4" && !enableThinking)
                return "<|channel>thought\n<channel|>";

            // Qwen 3.5 family (including Qwen 3.6 which reports as "qwen35moe") with
            // thinking ENABLED uses a Jinja template that emits `<think>\n` after the
            // assistant role marker as part of the generation prompt. The same template
            // does NOT re-emit `<think>...</think>` framing for PAST assistant messages
            // (it only does so for the most-recent query's assistant turn). Without an
            // injection here, the cache's `<think>` token (forwarded in turn N as part
            // of the generation prompt) has no counterpart in turn N+1's render of the
            // same assistant turn, causing every multi-turn request from the WebUI to
            // reset the cache.
            if (IsQwen35FamilyArch(architecture) && enableThinking)
                return "<think>\n";

            // qwen4exp appends `<think>` to the generation prompt UNCONDITIONALLY -
            // the model always reasons - so the cache always holds it, whatever the
            // thinking flag says. (The trailing newline is trimmed away again at
            // interior boundaries when the renderer's own TrimEnd behaviour calls
            // for it, mirroring the generation prompt.)
            if (architecture == "qwen4exp")
                return "<think>\n";

            // Qwen 3.5 family with thinking DISABLED is rendered through the hardcoded
            // RenderQwen35 path, which DOES emit `<think>\n\n</think>\n\n` for past
            // assistant messages already - so no injection is needed.

            // All other supported architectures (Qwen3, GptOss / Harmony, Gemma3,
            // Mistral3, Nemotron, ...) emit consistent framing for past and current-turn
            // assistant messages and need no injection.
            return string.Empty;
        }

        /// <summary>
        /// Marker after which the chat template emits an assistant HEADER that the raw
        /// generated tokens already carry themselves, or null when the template's
        /// assistant framing matches what the model actually produced.
        ///
        /// Muse-Glimmer's GGUF template is the case that needs this. Its generation
        /// prompt is bare — <c>&lt;|start|&gt;assistant</c> and nothing else — so the model
        /// itself emits the routing header and channel framing that follow
        /// (<c> to=self&lt;|message|&gt;</c> for its reasoning turn, then
        /// <c>&lt;|eom|&gt;&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c> for the answer).
        /// Rendering the SAME turn again as history, however, takes the template's
        /// past-assistant branch, which emits <c>&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c>
        /// before the content. Splicing the raw tokens after that header prepends three
        /// tokens the KV cache never saw (` to`, `=user`, `&lt;|message|&gt;`), so the very
        /// first follow-up turn diverges at the first assistant boundary and EVERY block
        /// hash misses — prefix reuse was reported as 0% for every multi-turn Muse-Glimmer
        /// conversation. Dropping the template's header restores the exact cached stream:
        /// <c>&lt;|start|&gt;assistant</c> + raw tokens.
        /// </summary>
        internal static string GetTemplateAssistantHeaderAnchor(string architecture)
        {
            if (architecture is "muse-glimmer" or "muse_glimmer")
                return "<|start|>assistant";
            return null;
        }

        /// <summary>
        /// Delete the text the template emitted between <paramref name="anchor"/> and each
        /// spliced-raw-token placeholder. The scan stops at the nearest anchor before the
        /// placeholder and refuses to cross a turn boundary (another <c>&lt;|start|&gt;</c>),
        /// so a template that legitimately emits no header is left untouched.
        /// </summary>
        private static string StripTemplateAssistantHeaders(string text, string anchor)
        {
            var sb = new System.Text.StringBuilder(text.Length);
            int searchPos = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }

                int copyEnd = sentinel;
                int anchorStart = text.LastIndexOf(anchor, sentinel - 1 < 0 ? 0 : sentinel - 1, StringComparison.Ordinal);
                if (anchorStart >= searchPos)
                {
                    int headerStart = anchorStart + anchor.Length;
                    // Refuse to swallow a whole turn: the header may only be the short
                    // routing/channel run the template adds right before the content.
                    if (text.IndexOf("<|start|>", headerStart, sentinel - headerStart, StringComparison.Ordinal) < 0)
                        copyEnd = headerStart;
                }
                sb.Append(text, searchPos, copyEnd - searchPos);

                int sentinelEnd = text.IndexOf(PlaceholderSentinel, sentinel + 1);
                if (sentinelEnd < 0)
                    throw new InvalidOperationException(
                        "Malformed KV-cache placeholder: opening sentinel without matching close.");
                sb.Append(text, sentinel, sentinelEnd - sentinel + 1);
                searchPos = sentinelEnd + 1;
            }
            return sb.ToString();
        }

        private static bool IsQwen35FamilyArch(string architecture)
        {
            return architecture == "qwen35"
                || architecture == "qwen35moe"
                || architecture == "qwen3next"
                || architecture == "qwen3vl"
                || architecture == "qwen3vlmoe"
                || architecture == "qwen4exp";
        }

        /// <summary>
        /// Render <paramref name="messages"/> through the configured chat template into a
        /// token sequence, splicing raw assistant output tokens where available.
        /// </summary>
        /// <param name="tokenizer">Tokenizer to use for encoding text segments.</param>
        /// <param name="chatTemplate">The model's GGUF-embedded Jinja2 template (may be null).</param>
        /// <param name="messages">Chat history (may include assistant messages with <see cref="ChatMessage.RawOutputTokens"/>).</param>
        /// <param name="architecture">Architecture name from <see cref="ModelConfig.Architecture"/>.</param>
        /// <param name="addGenerationPrompt">Whether to append a generation-prompt suffix (e.g. <c>&lt;|im_start|&gt;assistant</c>).</param>
        /// <param name="tools">Optional tool list for tool-calling templates.</param>
        /// <param name="enableThinking">Whether to enable the model's thinking / reasoning channel.</param>
        public List<int> RenderToTokens(
            ITokenizer tokenizer,
            string chatTemplate,
            List<ChatMessage> messages,
            string architecture,
            bool addGenerationPrompt,
            List<ToolFunction>? tools = null,
            bool enableThinking = false)
        {
            if (tokenizer == null)
                throw new ArgumentNullException(nameof(tokenizer));
            if (messages == null)
                throw new ArgumentNullException(nameof(messages));

            // Build a parallel list where each cached assistant message is replaced with a
            // placeholder ChatMessage. Track the raw tokens in render order so we can splice
            // them back in.
            List<ChatMessage>? renderedMessages = null;
            List<List<int>>? rawTokensByPlaceholderIndex = null;
            int placeholderCount = 0;

            for (int i = 0; i < messages.Count; i++)
            {
                ChatMessage msg = messages[i];
                bool hasRawTokens = msg != null
                    && msg.Role == "assistant"
                    && msg.RawOutputTokens != null
                    && msg.RawOutputTokens.Count > 0;

                if (!hasRawTokens)
                {
                    if (renderedMessages != null)
                        renderedMessages.Add(msg!);
                    continue;
                }

                if (renderedMessages == null)
                {
                    renderedMessages = new List<ChatMessage>(messages.Count);
                    for (int j = 0; j < i; j++)
                        renderedMessages.Add(messages[j]);
                    rawTokensByPlaceholderIndex = new List<List<int>>();
                }

                renderedMessages.Add(new ChatMessage
                {
                    Role = msg!.Role,
                    Content = MakePlaceholder(placeholderCount),
                    // Don't carry Thinking through the template - the raw tokens already contain it.
                    Thinking = null,
                    ToolCalls = null,
                    ImagePaths = msg.ImagePaths,
                    AudioPaths = msg.AudioPaths,
                    IsVideo = msg.IsVideo,
                });

                rawTokensByPlaceholderIndex!.Add(msg.RawOutputTokens!);
                placeholderCount++;
            }

            List<ChatMessage> messagesForRender = renderedMessages ?? messages;

            string text = _innerRenderer.Render(
                chatTemplate,
                messagesForRender,
                addGenerationPrompt: addGenerationPrompt,
                architecture: architecture,
                tools: tools,
                enableThinking: enableThinking);

            // Fast path: no placeholders -> just tokenize the whole rendered string.
            if (placeholderCount == 0)
                return tokenizer.Encode(text, addSpecial: true);

            // Some chat templates (notably Gemma 4) call a strip_thinking filter on
            // assistant content, which would silently delete a prefix injected via the
            // Content field. To work around this AND to keep the renderer template-agnostic,
            // we inject the architecture-specific generation suffix as POST-render text
            // patching: walk the rendered text and prepend the suffix before each placeholder.
            // Templates that carry a thinking channel emit an EMPTY `<think></think>`
            // block ahead of a past assistant turn, to tell the model that turn's
            // reasoning was dropped. The KV cache has no such block: turn N forwarded
            // `<think>` + the model's real reasoning tokens. Left in place, that empty
            // block inserts four tokens (`<think>`, `\n\n`, `</think>`, `\n\n`) right
            // at the first assistant boundary, so the re-rendered prefix diverges there
            // and every multi-turn request re-prefills the whole conversation. Drop it
            // before injecting the suffix that reproduces what the cache actually saw.
            if (enableThinking || architecture == "qwen4exp")
                text = StripEmptyThinkBlockBeforePlaceholders(text);

            string suffix = GetAssistantGenerationSuffix(architecture, enableThinking);
            if (!string.IsNullOrEmpty(suffix))
                text = InjectSuffixBeforePlaceholders(text, suffix);

            // The mirror image of the suffix injection: some templates emit MORE assistant
            // framing for a past turn than the generation prompt did, and the raw tokens
            // already contain their own. See GetTemplateAssistantHeaderAnchor.
            string headerAnchor = GetTemplateAssistantHeaderAnchor(architecture);
            if (!string.IsNullOrEmpty(headerAnchor))
                text = StripTemplateAssistantHeaders(text, headerAnchor);

            // Some renderers (those that go through ChatTemplate.RenderFromGgufTemplate's
            // jinja path) apply a final TrimEnd to the whole rendered text. That stripped
            // trailing whitespace from the GENERATION PROMPT in the previous turn, so the
            // KV cache contains tokens WITHOUT that trailing whitespace at the boundary
            // between the assistant prompt and the model's first generated token.
            //
            // For our re-render to produce a token sequence whose prefix matches the cache,
            // we need to mimic the same trim at every interior placeholder boundary.
            // We detect "renderer applied TrimEnd" simply by checking whether the FINAL
            // character of the rendered text is whitespace - if it is, the renderer didn't
            // trim and we shouldn't either; if it isn't, the renderer trimmed and we mirror
            // that trimming at each interior boundary.
            bool rendererStrippedTrailingWhitespace =
                text.Length > 0 && !char.IsWhiteSpace(text[text.Length - 1]);
            if (rendererStrippedTrailingWhitespace)
                text = TrimWhitespaceBeforeEachPlaceholder(text);

            return TokenizeAndReplacePlaceholderSpans(tokenizer, text, rawTokensByPlaceholderIndex!);
        }

        /// <summary>
        /// Remove an EMPTY thinking block (<c>&lt;think&gt;</c>, only whitespace,
        /// <c>&lt;/think&gt;</c>, then whitespace) that the chat template emitted
        /// immediately before a spliced assistant turn.
        ///
        /// Only an empty block is removed. A block with real reasoning text in it was
        /// not produced by this mechanism and is left alone, so a client that replays
        /// prior reasoning verbatim still renders it.
        /// </summary>
        internal static string StripEmptyThinkBlockBeforePlaceholders(string text)
        {
            const string open = "<think>";
            const string close = "</think>";
            if (text.IndexOf(PlaceholderSentinel) < 0 || text.IndexOf(open, StringComparison.Ordinal) < 0)
                return text;

            var sb = new System.Text.StringBuilder(text.Length);
            int searchPos = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }

                // Walk back over trailing whitespace, then require "</think>".
                int cursor = sentinel;
                while (cursor > searchPos && char.IsWhiteSpace(text[cursor - 1]))
                    cursor--;
                if (cursor - searchPos >= close.Length
                    && string.CompareOrdinal(text, cursor - close.Length, close, 0, close.Length) == 0)
                {
                    cursor -= close.Length;
                    // Walking back over whitespace has to land exactly on the end of
                    // "<think>" - anything else means the block held real reasoning.
                    while (cursor > searchPos && char.IsWhiteSpace(text[cursor - 1]))
                        cursor--;
                    if (cursor - searchPos >= open.Length
                        && string.CompareOrdinal(text, cursor - open.Length, open, 0, open.Length) == 0)
                    {
                        // Copy up to the "<think>" and drop the whole empty block,
                        // INCLUDING the whitespace the template put after "</think>".
                        // Keeping that whitespace would leave `assistant\n` + `\n\n`
                        // where the cache has `assistant\n` + `<think>\n`, which just
                        // moves the divergence rather than removing it.
                        sb.Append(text, searchPos, (cursor - open.Length) - searchPos);
                        sb.Append(text[sentinel]);
                        searchPos = sentinel + 1;
                        continue;
                    }
                }

                sb.Append(text, searchPos, sentinel - searchPos + 1);
                searchPos = sentinel + 1;
            }
            return sb.ToString();
        }

        private static string TrimWhitespaceBeforeEachPlaceholder(string text)
        {
            var sb = new System.Text.StringBuilder(text.Length);
            int searchPos = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }

                int copyEnd = sentinel;
                while (copyEnd > searchPos && char.IsWhiteSpace(text[copyEnd - 1]))
                    copyEnd--;
                sb.Append(text, searchPos, copyEnd - searchPos);

                int sentinelEnd = text.IndexOf(PlaceholderSentinel, sentinel + 1);
                if (sentinelEnd < 0)
                    throw new InvalidOperationException(
                        "Malformed KV-cache placeholder: opening sentinel without matching close.");
                sb.Append(text, sentinel, sentinelEnd - sentinel + 1);
                searchPos = sentinelEnd + 1;
            }
            return sb.ToString();
        }

        private static string InjectSuffixBeforePlaceholders(string text, string suffix)
        {
            var sb = new System.Text.StringBuilder(text.Length + suffix.Length * 4);
            int searchPos = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }
                sb.Append(text, searchPos, sentinel - searchPos);
                sb.Append(suffix);
                int sentinelEnd = text.IndexOf(PlaceholderSentinel, sentinel + 1);
                if (sentinelEnd < 0)
                    throw new InvalidOperationException(
                        "Malformed KV-cache placeholder: opening sentinel without matching close.");
                sb.Append(text, sentinel, sentinelEnd - sentinel + 1);
                searchPos = sentinelEnd + 1;
            }
            return sb.ToString();
        }

        internal static string MakePlaceholder(int index)
        {
            // Encoded as PUA-sentinel + DigitsBase32 + PUA-sentinel.
            // Using two sentinels makes the split unambiguous in the tokenizer regex's eyes,
            // and the digits guarantee that two adjacent placeholders never get merged.
            return $"{PlaceholderSentinel}R{index:D4}{PlaceholderSentinel}";
        }

        /// <summary>
        /// Tokenize <paramref name="text"/> as a SINGLE string (so the BPE/SentencePiece
        /// merging decisions at segment boundaries match exactly what the renderer would
        /// have produced for an entire turn-1-style prompt), then replace each occurrence
        /// of the placeholder marker's tokens with the corresponding raw output tokens.
        ///
        /// Tokenizing the whole text in one shot is what makes this approach
        /// renderer-agnostic: it doesn't matter whether the chat template applies a
        /// final TrimEnd, whether it appends additional suffixes, or whether the BPE
        /// tokenizer would have merged the boundary differently between an interior
        /// segment and a trailing one. The placeholder text is built from PUA codepoints
        /// (Unicode <see cref="PlaceholderSentinel"/>) plus ASCII digits/letters so its
        /// tokenization is locally-stable: the BPE pretokenizer regex always isolates
        /// these characters into their own chunks regardless of surrounding context.
        /// </summary>
        private static List<int> TokenizeAndReplacePlaceholderSpans(
            ITokenizer tokenizer,
            string text,
            List<List<int>> rawTokensByPlaceholderIndex)
        {
            // Step 1: tokenize the rendered text as a whole.
            List<int> tokens = tokenizer.Encode(text, addSpecial: true);

            // Step 2: for each placeholder, find its token span and replace.
            // Working backwards (highest-numbered placeholder first) keeps earlier
            // positions stable as we splice (which can lengthen or shorten the list).
            int placeholderCount = rawTokensByPlaceholderIndex.Count;
            for (int i = placeholderCount - 1; i >= 0; i--)
            {
                string placeholder = MakePlaceholder(i);
                List<int> placeholderTokens = tokenizer.Encode(placeholder, addSpecial: false);

                int spanStart = FindSubsequence(tokens, placeholderTokens);
                if (spanStart < 0)
                    throw new InvalidOperationException(
                        $"Could not locate placeholder #{i} ({placeholder.Length} chars, {placeholderTokens.Count} tokens) in tokenized output. " +
                        "This usually means the tokenizer is treating the placeholder differently in context vs in isolation; " +
                        "consider switching to a placeholder character that survives BPE pretokenization.");

                tokens.RemoveRange(spanStart, placeholderTokens.Count);
                tokens.InsertRange(spanStart, rawTokensByPlaceholderIndex[i]);
            }

            return tokens;
        }

        private static int FindSubsequence(List<int> haystack, List<int> needle)
        {
            if (needle.Count == 0 || haystack.Count < needle.Count)
                return -1;

            int last = haystack.Count - needle.Count;
            for (int i = 0; i <= last; i++)
            {
                bool match = true;
                for (int j = 0; j < needle.Count; j++)
                {
                    if (haystack[i + j] != needle[j])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                    return i;
            }
            return -1;
        }
    }
}
