// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// Regression tests for Muse-Glimmer multi-turn KV prefix reuse.
///
/// The bug: Muse-Glimmer's GGUF chat template uses a BARE generation prompt
/// (<c>&lt;|start|&gt;assistant</c> and nothing more) — the model itself emits the
/// routing header that follows (<c> to=self&lt;|message|&gt;</c> for its reasoning turn,
/// then <c>&lt;|eom|&gt;&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c> for the answer).
/// But when the SAME turn is re-rendered as history, the template takes its
/// past-assistant branch and emits <c>&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c>
/// before the content. Splicing the raw generated tokens after that header prepends
/// three tokens the KV cache never saw, so turn 2 diverges from the cache at the very
/// first assistant boundary and EVERY block hash misses.
///
/// Observed end-to-end before the fix: a 2-turn conversation on
/// Muse-Glimmer-30B / ggml_metal reported kvReused=0 (0.0%) with a 761-token
/// turn-2 prompt; token dumps showed the divergence at index 51, where turn 2 had
/// the extra <c>[" to", "=user", "&lt;|message|&gt;"]</c> run. After the fix the same
/// conversation reports 256/358 tokens reused (71.5%).
/// </summary>
public class MuseGlimmerKvReuseRenderTests
{
    /// <summary>
    /// Reproduces the shape of Muse-Glimmer's GGUF template that matters here: a bare
    /// <c>&lt;|start|&gt;assistant</c> generation prompt, but a full
    /// <c>&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c> header for a past assistant turn.
    /// </summary>
    private sealed class MuseGlimmerLikeRenderer : IPromptRenderer
    {
        public string Render(string? template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string? architecture = null,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var m in messages)
            {
                if (m.Role == "assistant")
                {
                    sb.Append("<|start|>assistant to=user<|message|>");
                    sb.Append(m.Content ?? "");
                    sb.Append("<|eot|>");
                }
                else
                {
                    sb.Append("<|start|>").Append(m.Role).Append("<|message|>");
                    sb.Append(m.Content ?? "");
                    sb.Append("<|eot|>");
                }
            }
            if (addGenerationPrompt)
                sb.Append("<|start|>assistant");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Tokenizer that treats every <c>&lt;|...|&gt;</c> marker as one token and every other
    /// character as one token, which is enough to see the extra header tokens the bug
    /// inserted while keeping the placeholder locally stable.
    /// </summary>
    private sealed class MarkerTokenizer : ITokenizer
    {
        private readonly Dictionary<string, int> _ids = new();
        private readonly List<string> _vocab = new();
        private const int Bos = 0;
        private const int Eos = 1;

        public MarkerTokenizer()
        {
            _vocab.Add("<bos>");
            _vocab.Add("<eos>");
        }

        public string[] Vocab => _vocab.ToArray();
        public int BosTokenId => Bos;
        public int[] EosTokenIds => new[] { Eos };
        public int VocabSize => _vocab.Count;

        private int IdOf(string piece)
        {
            if (!_ids.TryGetValue(piece, out int id))
            {
                id = _vocab.Count;
                _ids[piece] = id;
                _vocab.Add(piece);
            }
            return id;
        }

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var result = new List<int>();
            if (addSpecial)
                result.Add(Bos);
            if (text == null)
                return result;

            int i = 0;
            while (i < text.Length)
            {
                if (text[i] == '<' && i + 1 < text.Length && text[i + 1] == '|')
                {
                    int close = text.IndexOf("|>", i + 2, StringComparison.Ordinal);
                    if (close >= 0)
                    {
                        result.Add(IdOf(text.Substring(i, close + 2 - i)));
                        i = close + 2;
                        continue;
                    }
                }
                result.Add(IdOf(text[i].ToString()));
                i++;
            }
            return result;
        }

        public string Decode(List<int> ids)
        {
            var sb = new System.Text.StringBuilder();
            if (ids != null)
                foreach (var id in ids)
                    if (id != Bos && id != Eos && id < _vocab.Count)
                        sb.Append(_vocab[id]);
            return sb.ToString();
        }

        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            if (tokenId == Bos || tokenId == Eos) return;
            if (tokenId < _vocab.Count)
                buffer.AddRange(System.Text.Encoding.UTF8.GetBytes(_vocab[tokenId]));
        }

        public bool IsEos(int tokenId) => tokenId == Eos;

        public int LookupToken(string tokenStr) => _ids.TryGetValue(tokenStr, out var id) ? id : -1;
    }

    private static int FindSubsequence(List<int> haystack, List<int> needle)
    {
        if (needle.Count == 0 || haystack.Count < needle.Count) return -1;
        for (int i = 0; i <= haystack.Count - needle.Count; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Count; j++)
                if (haystack[i + j] != needle[j]) { match = false; break; }
            if (match) return i;
        }
        return -1;
    }

    /// <summary>
    /// The invariant that makes prefix reuse possible: turn 2's rendered tokens must
    /// START WITH turn 1's rendered tokens followed immediately by turn 1's raw output
    /// tokens. Any header the template inserts between the two breaks every block hash.
    /// </summary>
    [Fact]
    public void MuseGlimmer_SecondTurnPrompt_ExtendsFirstTurnCacheExactly()
    {
        var renderer = new KVCachePromptRenderer(new MuseGlimmerLikeRenderer());
        var tokenizer = new MarkerTokenizer();

        var turn1 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "hello" },
        };
        List<int> turn1Tokens = renderer.RenderToTokens(
            tokenizer, chatTemplate: null, turn1, architecture: "muse-glimmer", addGenerationPrompt: true);

        // What the model actually generated, INCLUDING its own routing/channel header —
        // this is exactly what lands in the KV cache after the generation prompt.
        List<int> rawOutput = tokenizer.Encode(
            " to=self<|message|>thinking<|eom|><|start|>assistant to=user<|message|>answer",
            addSpecial: false);

        var turn2 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "hello" },
            new() { Role = "assistant", Content = "answer", RawOutputTokens = rawOutput },
            new() { Role = "user", Content = "continue" },
        };
        List<int> turn2Tokens = renderer.RenderToTokens(
            tokenizer, chatTemplate: null, turn2, architecture: "muse-glimmer", addGenerationPrompt: true);

        var cached = new List<int>(turn1Tokens);
        cached.AddRange(rawOutput);

        Assert.True(turn2Tokens.Count > cached.Count,
            "turn 2 must extend the cached prefix, not shorten it");
        for (int i = 0; i < cached.Count; i++)
        {
            Assert.True(cached[i] == turn2Tokens[i],
                $"turn 2 diverges from the cached prefix at token {i}: " +
                $"cached='{tokenizer.Decode(new List<int> { cached[i] })}' " +
                $"rendered='{tokenizer.Decode(new List<int> { turn2Tokens[i] })}'");
        }
    }

    /// <summary>
    /// The specific symptom: the template's <c> to=user&lt;|message|&gt;</c> header must not
    /// survive in front of the spliced raw tokens, because the raw tokens carry their own.
    /// </summary>
    [Fact]
    public void MuseGlimmer_TemplateAssistantHeader_IsStrippedBeforeSplicedRawTokens()
    {
        var renderer = new KVCachePromptRenderer(new MuseGlimmerLikeRenderer());
        var tokenizer = new MarkerTokenizer();

        List<int> rawOutput = tokenizer.Encode(" to=self<|message|>hi", addSpecial: false);
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "u" },
            new() { Role = "assistant", Content = "hi", RawOutputTokens = rawOutput },
            new() { Role = "user", Content = "v" },
        };

        List<int> tokens = renderer.RenderToTokens(
            tokenizer, chatTemplate: null, messages, architecture: "muse-glimmer", addGenerationPrompt: true);

        // The raw run must start immediately after the BARE "<|start|>assistant" the
        // generation prompt ended with - no ` to=user<|message|>` in between.
        int rawAt = FindSubsequence(tokens, rawOutput);
        Assert.True(rawAt > 0, "spliced raw tokens should appear in the rendered sequence");
        string beforeRaw = tokenizer.Decode(tokens.GetRange(0, rawAt));
        Assert.EndsWith("<|start|>assistant", beforeRaw, StringComparison.Ordinal);

        Assert.DoesNotContain("assistant to=user<|message|> to=self", tokenizer.Decode(tokens));
    }

    /// <summary>
    /// The strip is architecture-scoped: an unrelated architecture must keep whatever
    /// framing its template emits, or this fix would silently break every other model.
    /// </summary>
    [Fact]
    public void OtherArchitectures_KeepTheirTemplateAssistantHeader()
    {
        var renderer = new KVCachePromptRenderer(new MuseGlimmerLikeRenderer());
        var tokenizer = new MarkerTokenizer();

        List<int> rawOutput = tokenizer.Encode("hi", addSpecial: false);
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "u" },
            new() { Role = "assistant", Content = "hi", RawOutputTokens = rawOutput },
            new() { Role = "user", Content = "v" },
        };

        List<int> tokens = renderer.RenderToTokens(
            tokenizer, chatTemplate: null, messages, architecture: "qwen3", addGenerationPrompt: true);

        Assert.Contains("assistant to=user<|message|>", tokenizer.Decode(tokens));
    }
}
