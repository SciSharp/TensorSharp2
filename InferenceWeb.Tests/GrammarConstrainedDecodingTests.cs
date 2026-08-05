using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

/// <summary>
/// Covers grammar-constrained decoding end to end: the GBNF parser, the
/// pushdown matcher, the JSON Schema compiler, and the token-mask layer that
/// makes it fast.
/// </summary>
public class GrammarConstrainedDecodingTests
{
    // ---- helpers ---------------------------------------------------------

    /// <summary>
    /// A tokenizer whose vocabulary is a fixed set of text pieces, so mask
    /// expectations can be written by hand. Includes multi-character pieces and
    /// a multi-byte character, because those are where a naive implementation
    /// breaks.
    /// </summary>
    private sealed class PieceTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        private readonly string[] _pieces;
        public PieceTokenizer(params string[] pieces)
        {
            // Index 0 is reserved as EOS so masking has something to re-open.
            _pieces = new[] { "<eos>" }.Concat(pieces).ToArray();
        }
        public string[] Vocab => _pieces;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 0 };
        public int VocabSize => _pieces.Length;

        /// <summary>
        /// EOS plus anything spelled like a chat-control marker, mirroring what
        /// a real tokenizer reports from its GGUF token types.
        /// </summary>
        public IReadOnlyCollection<int> SpecialTokenIds =>
            Enumerable.Range(0, _pieces.Length)
                      .Where(i => i == 0 || _pieces[i].StartsWith("<|", StringComparison.Ordinal))
                      .ToArray();
        public List<int> Encode(string text, bool addSpecial = true) => new();
        public string Decode(List<int> ids) => string.Concat(ids.Select(i => _pieces[i]));
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            if (tokenId == 0) return;               // EOS carries no text
            buffer.AddRange(Encoding.UTF8.GetBytes(_pieces[tokenId]));
        }
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string tokenStr) => Array.IndexOf(_pieces, tokenStr);
    }

    /// <summary>Feed text through a grammar; report acceptance and completeness.</summary>
    private static (bool Accepted, bool Complete) Feed(Grammar g, string text)
    {
        var m = new GrammarMatcher(g);
        var state = m.InitialState;
        var partial = PartialUtf8.Empty;
        state = m.AcceptBytes(state, Encoding.UTF8.GetBytes(text), ref partial);
        bool accepted = !state.IsDead && partial.Remaining == 0;
        return (accepted, accepted && state.CanTerminate);
    }

    private static bool Allows(GrammarConstraint c, int tokenId)
    {
        ulong[] mask = c.CurrentMask();
        return (mask[tokenId >> 6] & (1UL << (tokenId & 63))) != 0;
    }

    // ---- GBNF parser -----------------------------------------------------

    [Fact]
    public void ParsesLiteralsClassesAndRanges()
    {
        var g = Grammar.Parse("root ::= \"a\" [b-d] [xyz]");
        Assert.True(Feed(g, "acy").Complete);
        Assert.False(Feed(g, "aey").Accepted);
        Assert.False(Feed(g, "acw").Accepted);
    }

    [Fact]
    public void ParsesNegatedCharacterClass()
    {
        var g = Grammar.Parse("root ::= [^abc]");
        Assert.True(Feed(g, "z").Complete);
        Assert.False(Feed(g, "b").Accepted);
    }

    [Theory]
    [InlineData("root ::= \"x\"*", "", true)]
    [InlineData("root ::= \"x\"*", "xxx", true)]
    [InlineData("root ::= \"x\"+", "", false)]
    [InlineData("root ::= \"x\"+", "x", true)]
    [InlineData("root ::= \"x\"?", "", true)]
    [InlineData("root ::= \"x\"?", "xx", false)]
    [InlineData("root ::= \"x\"{2,3}", "x", false)]
    [InlineData("root ::= \"x\"{2,3}", "xx", true)]
    [InlineData("root ::= \"x\"{2,3}", "xxx", true)]
    [InlineData("root ::= \"x\"{2,3}", "xxxx", false)]
    public void HandlesRepetitionOperators(string gbnf, string input, bool complete)
    {
        Assert.Equal(complete, Feed(Grammar.Parse(gbnf), input).Complete);
    }

    [Fact]
    public void RejectsLeftRecursionInsteadOfHanging()
    {
        // Left recursion would make the eager non-terminal expansion loop
        // forever, so it must be refused at parse time with a clear message.
        var ex = Assert.Throws<GrammarParseException>(() =>
            Grammar.Parse("root ::= root \"a\" | \"b\""));
        Assert.Contains("left-recursive", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RequiresARootRule()
    {
        Assert.Throws<GrammarParseException>(() => Grammar.Parse("other ::= \"a\""));
    }

    // ---- JSON grammar ----------------------------------------------------

    [Theory]
    [InlineData("{}", true)]
    [InlineData("{\"a\": 1}", true)]
    [InlineData("{\"a\": [1, 2, {\"b\": null}]}", true)]
    [InlineData("{\"a\": \"x\\ny\"}", true)]
    [InlineData("{\"a\": -1.5e10}", true)]
    [InlineData("{\"a\": 1,}", false)]
    [InlineData("{a: 1}", false)]
    [InlineData("{'a': 1}", false)]
    [InlineData("{\"a\": 01}", false)]
    public void JsonObjectGrammarMatchesJsonSyntax(string text, bool valid)
    {
        Assert.Equal(valid, Feed(Grammar.JsonObject(), text).Complete);
    }

    [Fact]
    public void JsonGrammarHandlesMultiByteCharacters()
    {
        // The matcher works in code points but tokens arrive as bytes; a
        // multi-byte character must survive being split.
        Assert.True(Feed(Grammar.JsonObject(), "{\"k\": \"héllo 世界 🎉\"}").Complete);
    }

    [Fact]
    public void IncompleteJsonIsAcceptedButNotComplete()
    {
        var (accepted, complete) = Feed(Grammar.JsonObject(), "{\"a\": ");
        Assert.True(accepted);      // still a viable prefix
        Assert.False(complete);     // but generation may not stop here
    }

    // ---- JSON Schema compiler -------------------------------------------

    private const string PersonSchema = """
        {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "age":  { "type": "integer" },
            "tags": { "type": "array", "items": { "type": "string" } }
          },
          "required": ["name", "age"]
        }
        """;

    [Theory]
    [InlineData("{\"name\": \"a\", \"age\": 3}", true)]
    [InlineData("{\"name\": \"a\", \"age\": 3, \"tags\": [\"x\"]}", true)]
    [InlineData("{\"name\": \"a\"}", false)]                       // missing required
    [InlineData("{\"age\": 3, \"name\": \"a\"}", false)]           // declared order is fixed
    [InlineData("{\"name\": \"a\", \"age\": \"3\"}", false)]       // wrong type
    [InlineData("{\"name\": \"a\", \"age\": 3, \"x\": 1}", false)] // undeclared property
    public void SchemaCompilerEnforcesShape(string json, bool valid)
    {
        Assert.Equal(valid, Feed(Grammar.FromJsonSchema(PersonSchema), json).Complete);
    }

    [Fact]
    public void SchemaCompilerHandlesEnumAndConst()
    {
        var g = Grammar.FromJsonSchema(
            """{"type":"object","properties":{"c":{"enum":["red","green"]},"k":{"const":7}},"required":["c","k"]}""");
        Assert.True(Feed(g, "{\"c\": \"red\", \"k\": 7}").Complete);
        Assert.False(Feed(g, "{\"c\": \"blue\", \"k\": 7}").Accepted);
        Assert.False(Feed(g, "{\"c\": \"red\", \"k\": 8}").Accepted);
    }

    [Fact]
    public void SchemaCompilerHandlesRecursiveRefs()
    {
        var g = Grammar.FromJsonSchema("""
            {
              "$defs": { "node": { "type": "object",
                  "properties": { "v": {"type":"integer"}, "next": {"$ref":"#/$defs/node"} },
                  "required": ["v"] } },
              "$ref": "#/$defs/node"
            }
            """);
        Assert.True(Feed(g, "{\"v\":1}").Complete);
        Assert.True(Feed(g, "{\"v\":1,\"next\":{\"v\":2,\"next\":{\"v\":3}}}").Complete);
    }

    [Fact]
    public void SchemaCompilerHandlesAnyOf()
    {
        var g = Grammar.FromJsonSchema(
            """{"type":"object","properties":{"v":{"anyOf":[{"type":"integer"},{"type":"string"}]}},"required":["v"]}""");
        Assert.True(Feed(g, "{\"v\": 5}").Complete);
        Assert.True(Feed(g, "{\"v\": \"five\"}").Complete);
        Assert.False(Feed(g, "{\"v\": true}").Accepted);
    }

    [Fact]
    public void SchemaCompilerBoundsIntegerRanges()
    {
        var g = Grammar.FromJsonSchema(
            """{"type":"object","properties":{"n":{"type":"integer","minimum":1,"maximum":5}},"required":["n"]}""");
        Assert.True(Feed(g, "{\"n\": 1}").Complete);
        Assert.True(Feed(g, "{\"n\": 5}").Complete);
        Assert.False(Feed(g, "{\"n\": 6}").Accepted);
        Assert.False(Feed(g, "{\"n\": 0}").Accepted);
    }

    [Fact]
    public void SchemaCompilerAppliesStringPatterns()
    {
        var g = Grammar.FromJsonSchema(
            """{"type":"object","properties":{"id":{"type":"string","pattern":"^[A-Z]{2}-[0-9]+$"}},"required":["id"]}""");
        Assert.True(Feed(g, "{\"id\": \"AB-9\"}").Complete);
        Assert.True(Feed(g, "{\"id\": \"AB-1234\"}").Complete);
        Assert.False(Feed(g, "{\"id\": \"ab-9\"}").Accepted);
        Assert.False(Feed(g, "{\"id\": \"ABC-9\"}").Accepted);
    }

    [Fact]
    public void UnsupportedPatternRelaxesRatherThanRejecting()
    {
        // A pattern we cannot model must widen to an ordinary string. Emitting
        // an approximate grammar could make valid output unreachable, which is
        // the one failure constrained decoding must never have.
        var g = Grammar.FromJsonSchema(
            """{"type":"object","properties":{"s":{"type":"string","pattern":"(?<=a)b"}},"required":["s"]}""");
        Assert.True(Feed(g, "{\"s\": \"anything at all\"}").Complete);
    }

    [Fact]
    public void EmptySchemaAcceptsAnyJsonValue()
    {
        var g = Grammar.FromJsonSchema(null);
        Assert.True(Feed(g, "{\"a\":1}").Complete);
        Assert.True(Feed(g, "[1,2]").Complete);
        Assert.True(Feed(g, "\"text\"").Complete);
        Assert.True(Feed(g, "42").Complete);
    }

    [Fact]
    public void MalformedSchemaIsReportedNotSwallowed()
    {
        Assert.Throws<GrammarParseException>(() => Grammar.FromJsonSchema("{ not json"));
    }

    // ---- token masking ---------------------------------------------------

    [Fact]
    public void MaskAllowsOnlyTokensTheGrammarCanAccept()
    {
        var tok = new PieceTokenizer("{", "}", "\"", "a", "b", ":", ",", " ", "1", "[");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);

        // At the start only an object may open.
        Assert.True(Allows(c, tok.LookupToken("{")));
        Assert.False(Allows(c, tok.LookupToken("}")));
        Assert.False(Allows(c, tok.LookupToken("a")));
        Assert.False(Allows(c, tok.LookupToken("[")));

        c.Accept(tok.LookupToken("{"));
        // After '{' either a key opens or the object closes.
        Assert.True(Allows(c, tok.LookupToken("\"")));
        Assert.True(Allows(c, tok.LookupToken("}")));
        Assert.False(Allows(c, tok.LookupToken(",")));
        Assert.False(Allows(c, tok.LookupToken("1")));
    }

    [Fact]
    public void MaskMatchesBruteForcePerTokenEvaluation()
    {
        // The trie walk is an optimisation of "test every token"; it must agree
        // with that reference exactly, including for multi-character pieces.
        var tok = new PieceTokenizer(
            "{", "}", "[", "]", "\"", ":", ",", " ", "\n",
            "a", "ab", "abc", "1", "12", "true", "false", "null",
            "\"a\"", "{\"", "\":", "é", "世界");

        var grammar = Grammar.JsonObject();
        var matcher = new GrammarMatcher(grammar);

        foreach (string prefix in new[] { "", "{", "{\"", "{\"a", "{\"a\":", "{\"a\": 1", "{\"a\": [" })
        {
            var c = new GrammarConstraint(grammar, tok);
            c.AcceptBytes(Encoding.UTF8.GetBytes(prefix));
            ulong[] fast = c.CurrentMask();

            for (int id = 1; id < tok.VocabSize; id++)
            {
                var bytes = new List<byte>();
                tok.AppendTokenBytes(id, bytes);
                bool expected = Feed(grammar, prefix + Encoding.UTF8.GetString(bytes.ToArray())).Accepted;
                bool actual = (fast[id >> 6] & (1UL << (id & 63))) != 0;
                Assert.True(expected == actual,
                    $"prefix '{prefix}' token '{tok.Vocab[id]}': brute force says {expected}, mask says {actual}");
            }
        }
    }

    [Fact]
    public void SpecialTokensAreNeverUnmaskedAsText()
    {
        // A control token has a printable spelling; a permissive grammar must
        // not accept it as if the model had typed those characters.
        var tok = new PieceTokenizer("{", "}", "\"", "a", "<|im_start|>");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);
        c.AcceptBytes(Encoding.UTF8.GetBytes("{\""));
        Assert.False(Allows(c, tok.LookupToken("<|im_start|>")));
        Assert.True(Allows(c, tok.LookupToken("a")));
    }

    [Fact]
    public void ApplyMaskRemovesIllegalTokensFromLogits()
    {
        var tok = new PieceTokenizer("{", "}", "\"", "a", ":", "1");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);

        var logits = new float[tok.VocabSize];
        for (int i = 0; i < logits.Length; i++) logits[i] = 1.0f;
        c.ApplyMask(logits, allowEos: false);

        Assert.True(float.IsNegativeInfinity(logits[tok.LookupToken("}")]));
        Assert.True(float.IsNegativeInfinity(logits[tok.LookupToken("a")]));
        Assert.False(float.IsNegativeInfinity(logits[tok.LookupToken("{")]));
    }

    [Fact]
    public void EosIsMaskedUntilTheGrammarMayTerminate()
    {
        var tok = new PieceTokenizer("{", "}", "\"", "a");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);

        var logits = new float[tok.VocabSize];
        Array.Fill(logits, 1.0f);
        c.ApplyMask(logits, allowEos: c.IsComplete);
        Assert.False(c.IsComplete);
        Assert.True(float.IsNegativeInfinity(logits[0]));   // EOS suppressed mid-object

        c.Accept(tok.LookupToken("{"));
        c.Accept(tok.LookupToken("}"));
        Assert.True(c.IsComplete);

        Array.Fill(logits, 1.0f);
        c.ApplyMask(logits, allowEos: c.IsComplete);
        Assert.Equal(1.0f, logits[0]);                      // and restored, unmodified
    }

    [Fact]
    public void ApplyMaskPreservesTheModelsOwnEosScore()
    {
        // Whether to stop is the model's call once the grammar permits it; the
        // mask must not overwrite that logit with a constant.
        var tok = new PieceTokenizer("{", "}", "\"", "a");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);
        c.Accept(tok.LookupToken("{"));
        c.Accept(tok.LookupToken("}"));

        var logits = new float[tok.VocabSize];
        Array.Fill(logits, 0.5f);
        logits[0] = -3.25f;
        c.ApplyMask(logits, allowEos: true);
        Assert.Equal(-3.25f, logits[0]);
    }

    [Fact]
    public void SnapshotAndRestoreRewindTheParser()
    {
        // Speculative decoding accepts drafts provisionally and may reject them.
        var tok = new PieceTokenizer("{", "}", "\"", "a", ":");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);
        c.Accept(tok.LookupToken("{"));

        // ':' discriminates the two states: illegal right after '{', but legal
        // once inside a key, where it is just another string character. ('}'
        // would not discriminate — it is legal in both, closing the object in
        // one and appearing literally in the string in the other.)
        var snapshot = c.Snapshot();
        Assert.False(Allows(c, tok.LookupToken(":")));

        c.Accept(tok.LookupToken("\""));
        Assert.True(Allows(c, tok.LookupToken(":")));    // inside a key now

        c.Restore(snapshot);
        Assert.False(Allows(c, tok.LookupToken(":")));   // back to just after '{'
    }

    // ---- sampler integration --------------------------------------------

    [Fact]
    public void SamplerCannotSelectAnIllegalTokenEvenWhenItScoresHighest()
    {
        var tok = new PieceTokenizer("{", "}", "\"", "a", ":", "1");
        var config = SamplingConfig.Greedy;
        config.Grammar = new GrammarConstraint(Grammar.JsonObject(), tok);

        var sampler = new TokenSampler(config);
        var logits = new float[tok.VocabSize];
        Array.Fill(logits, 0f);
        logits[tok.LookupToken("a")] = 100f;   // model desperately wants 'a'

        int chosen = sampler.Sample(logits, new List<int>());
        Assert.Equal(tok.LookupToken("{"), chosen);
    }

    [Fact]
    public void ConstrainedGreedyDecodeProducesParseableJson()
    {
        // Drive a full decode with logits that always favour an illegal token,
        // and confirm the grammar alone still yields valid JSON.
        var tok = new PieceTokenizer("{", "}", "[", "]", "\"", ":", ",", " ", "a", "b", "1", "2");
        var config = SamplingConfig.Greedy;
        var constraint = new GrammarConstraint(Grammar.JsonObject(), tok);
        config.Grammar = constraint;
        var sampler = new TokenSampler(config);

        var produced = new List<int>();
        var rng = new Random(7);
        for (int step = 0; step < 200 && !constraint.IsComplete; step++)
        {
            var logits = new float[tok.VocabSize];
            for (int i = 0; i < logits.Length; i++) logits[i] = (float)rng.NextDouble();
            int id = sampler.Sample(logits, produced);
            Assert.NotEqual(0, id);                       // EOS is masked until complete
            produced.Add(id);
            constraint.Accept(id);
            Assert.False(constraint.IsDead);
        }

        Assert.True(constraint.IsComplete, "decode did not reach a complete JSON value");
        string text = tok.Decode(produced);
        using var doc = JsonDocument.Parse(text);          // must parse
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
    }

    [Fact]
    public void ConstrainedDecodeSatisfiesTheSchemaItWasGiven()
    {
        // Same idea, but against a schema: the emitted document must validate.
        var tok = new PieceTokenizer(
            "{", "}", "\"", ":", ",", " ", "name", "age", "1", "2", "3", "x", "y");
        var grammar = Grammar.FromJsonSchema(
            """{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}""");

        var constraint = new GrammarConstraint(grammar, tok);
        var config = SamplingConfig.Greedy;
        config.Grammar = constraint;
        var sampler = new TokenSampler(config);

        var produced = new List<int>();
        var rng = new Random(11);
        for (int step = 0; step < 400 && !constraint.IsComplete; step++)
        {
            var logits = new float[tok.VocabSize];
            for (int i = 0; i < logits.Length; i++) logits[i] = (float)rng.NextDouble();
            int id = sampler.Sample(logits, produced);
            produced.Add(id);
            constraint.Accept(id);
        }

        Assert.True(constraint.IsComplete);
        using var doc = JsonDocument.Parse(tok.Decode(produced));
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("name").ValueKind);
        Assert.Equal(JsonValueKind.Number, doc.RootElement.GetProperty("age").ValueKind);
    }

    [Fact]
    public void TopKSelectionStaysFastWhenMostLogitsAreEqual()
    {
        // Regression: top-k used a last-element-pivot quickselect partitioned
        // with '>=', which degrades to O(n^2) when scores tie -- every element
        // lands on one side and the range shrinks by one per pass. Masking makes
        // ~all of a large vocabulary -inf, which is exactly that input, and a
        // single token took minutes. Must stay linear-ish.
        const int Vocab = 200_000;
        var logits = new float[Vocab];
        Array.Fill(logits, float.NegativeInfinity);
        logits[12345] = 3.0f;
        logits[54321] = 2.0f;
        logits[99999] = 1.0f;

        var config = new SamplingConfig { Temperature = 1.0f, TopK = 40, TopP = 1.0f, MinP = 0f, RepetitionPenalty = 1.0f, Seed = 3 };
        var sampler = new TokenSampler(config);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        int chosen = sampler.Sample(logits, new List<int>());
        sw.Stop();

        Assert.True(sw.ElapsedMilliseconds < 2000,
            $"top-k over {Vocab} tied logits took {sw.ElapsedMilliseconds} ms");
        Assert.Contains(chosen, new[] { 12345, 54321, 99999 });
    }

    [Fact]
    public void TopKReturnsTheHighestScoringTokensInOrder()
    {
        // The heap replaced a quickselect; it must still select the true top-k,
        // ordered by score descending.
        var rng = new Random(5);
        var logits = new float[5000];
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)rng.NextDouble() * 10f;

        var expected = Enumerable.Range(0, logits.Length)
            .OrderByDescending(i => logits[i]).ThenBy(i => i)
            .Take(40).ToArray();

        var config = new SamplingConfig { Temperature = 1.0f, TopK = 40, TopP = 1.0f, MinP = 0f, RepetitionPenalty = 1.0f, Seed = 1 };
        // Sample many times; every draw must come from the true top-k set.
        var sampler = new TokenSampler(config);
        var allowed = new HashSet<int>(expected);
        for (int trial = 0; trial < 200; trial++)
        {
            var copy = (float[])logits.Clone();
            Assert.Contains(sampler.Sample(copy, new List<int>()), allowed);
        }
    }

    // ---- caching ---------------------------------------------------------

    [Fact]
    public void GrammarLibraryReusesCompiledGrammarsPerTokenizer()
    {
        var tok = new PieceTokenizer("{", "}", "\"", "a");
        var a = GrammarLibrary.ForJsonObject(tok);
        var b = GrammarLibrary.ForJsonObject(tok);
        Assert.Same(a, b);

        // Warming the masks is the expensive part; two requests must share it.
        var c1 = GrammarLibrary.NewConstraint(a, tok);
        var c2 = GrammarLibrary.NewConstraint(b, tok);
        Assert.Same(c1.CurrentMask(), c2.CurrentMask());
        Assert.NotSame(c1, c2);
    }

    [Fact]
    public void SeparateConstraintsDoNotShareParseState()
    {
        var tok = new PieceTokenizer("{", "}", "\"", "a");
        var cache = GrammarLibrary.ForJsonObject(tok);
        var c1 = GrammarLibrary.NewConstraint(cache, tok);
        var c2 = GrammarLibrary.NewConstraint(cache, tok);

        c1.Accept(tok.LookupToken("{"));
        Assert.True(Allows(c1, tok.LookupToken("}")));    // c1 is inside the object
        Assert.False(Allows(c2, tok.LookupToken("}")));   // c2 has not started
    }

    [Fact]
    public void VocabularyTrieIsBuiltOncePerTokenizer()
    {
        var tok = new PieceTokenizer("{", "}", "a");
        Assert.Same(GrammarTokenVocabulary.ForTokenizer(tok), GrammarTokenVocabulary.ForTokenizer(tok));
    }

    // ---- lazy activation (reasoning-channel models) ----------------------

    /// <summary>
    /// GPT-OSS opens every reply with a harmony channel header and reasons in
    /// the analysis channel first. A grammar armed from token 0 forbids that
    /// header, and the model — pushed into JSON with no reasoning behind it —
    /// fills the schema with placeholders. The constraint therefore stays
    /// dormant until the final channel's header has been generated.
    /// </summary>
    [Fact]
    public void LazyConstraintMasksNothingBeforeItsTrigger()
    {
        var tok = new PieceTokenizer("<|channel|>", "analysis", "final", "<|message|>", "{", "}", "\"", "a", "The");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);
        c.ActivateAfter("final<|message|>");

        Assert.False(c.IsActive);
        // The channel header and prose are not JSON, and must not be blocked.
        var logits = new float[tok.VocabSize];
        c.ApplyMask(logits, allowEos: false);
        Assert.All(logits, v => Assert.False(float.IsNegativeInfinity(v)));

        c.Accept(tok.LookupToken("<|channel|>"));
        c.Accept(tok.LookupToken("analysis"));
        c.Accept(tok.LookupToken("<|message|>"));
        c.Accept(tok.LookupToken("The"));
        Assert.False(c.IsActive);
    }

    [Fact]
    public void LazyConstraintEnforcesAfterItsTrigger()
    {
        var tok = new PieceTokenizer("<|channel|>", "analysis", "final", "<|message|>", "{", "}", "\"", "a", "The");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);
        c.ActivateAfter("final<|message|>");

        c.Accept(tok.LookupToken("<|channel|>"));
        c.Accept(tok.LookupToken("analysis"));
        c.Accept(tok.LookupToken("<|message|>"));
        c.Accept(tok.LookupToken("The"));
        c.Accept(tok.LookupToken("<|channel|>"));
        c.Accept(tok.LookupToken("final"));
        c.Accept(tok.LookupToken("<|message|>"));

        Assert.True(c.IsActive);
        // The prelude was not fed to the grammar: the parser is still at the
        // start of the object, so only "{" is legal.
        Assert.True(Allows(c, tok.LookupToken("{")));
        Assert.False(Allows(c, tok.LookupToken("The")));

        var logits = new float[tok.VocabSize];
        for (int i = 0; i < logits.Length; i++) logits[i] = 1.0f;
        c.ApplyMask(logits, allowEos: false);
        Assert.True(float.IsNegativeInfinity(logits[tok.LookupToken("The")]));
        Assert.False(float.IsNegativeInfinity(logits[tok.LookupToken("{")]));
    }

    [Fact]
    public void ConstraintWithoutATriggerEnforcesImmediately()
    {
        var tok = new PieceTokenizer("{", "}", "a");
        var c = new GrammarConstraint(Grammar.JsonObject(), tok);
        Assert.True(c.IsActive);
        Assert.False(Allows(c, tok.LookupToken("a")));
    }
}
