// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41ToolGrammarTests
{
    private const string Open = "<｜DSML｜ calls>";
    private const string End = "</｜DSML｜ calls>";
    private const string InvokeEnd = "</｜DSML｜ invoke>";
    private const string WeatherSchema = """
        {"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city","units"],"additionalProperties":false}
        """;

    [Fact]
    public void CanonicalCallPreservesRawUnicodeNewlinesAndLiteralXml()
    {
        string city = "  Paris\n\"café\" <b>東京🦊</b>  ";
        string text = Call("get_weather", Parameter("city", city) + Parameter("units", "celsius"));
        var plan = DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required);
        Assert.True(Accepts(plan, text));
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { Weather() });
        var call = Assert.Single(parser.Add(text, true).ToolCalls!);
        Assert.Equal(city, call.Arguments["city"]);
        Assert.Equal("celsius", call.Arguments["units"]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ObservedThinkingToolFailuresMaskMistypedClosersButPermitTheCanonicalCall(bool namedWeather)
    {
        string function = namedWeather ? "get_weather" : "read_invoice";
        string parameter = namedWeather ? "city" : "invoice_id";
        string value = namedWeather ? "Paris" : "INV-472";
        var tool = namedWeather ? Weather() : new ToolFunction
        {
            Name = function,
            ParametersSchemaJson = """
                {"type":"object","properties":{"invoice_id":{"type":"string"}},"required":["invoice_id"]}
                """,
        };
        string malformed = namedWeather
            ? "</parameter_name>\n<param name=\"units\">celsius</parameter_name>\n</invoke>\n</function_call>"
            : "</parameter_view>\n</invoke>\n<invoke name=\"calculate_total\">\n<invoke name=\"read_invoice\">";
        string correct = "</｜DSML｜ parameter>\n" + (namedWeather ? Parameter("units", "celsius") : "") + InvokeEnd + "\n" + End;
        var tokenizer = new ByteTokenizer(new[] { malformed, correct });
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool },
            namedWeather ? DeepSeek41ToolChoice.Named : DeepSeek41ToolChoice.Auto,
            namedWeather ? function : null);
        var grammar = plan.NewConstraint(tokenizer, thinking: true);
        string prefix = "I will call the declared tool.</think>\n\n" + Open + "\n<｜DSML｜ invoke name=\"" + function +
            "\">\n<｜DSML｜ parameter name=\"" + parameter + "\" string=\"true\">" + value;
        grammar.AcceptBytes(Encoding.UTF8.GetBytes(prefix));
        Assert.True(grammar.IsActive && !grammar.IsDead && !grammar.IsComplete);
        Assert.False(Allows(grammar, tokenizer.LookupToken(malformed)));
        Assert.True(Allows(grammar, tokenizer.LookupToken(correct)));

        // A byte-fragmented wrong closing tag must be stopped at the reserved
        // family prefix, before it can become unrelated-looking argument text.
        var fragmented = grammar.Fork();
        foreach (byte b in Encoding.UTF8.GetBytes("</para"))
        {
            Assert.True(Allows(fragmented, b));
            fragmented.Accept(b);
        }
        Assert.False(Allows(fragmented, (byte)'m'));
        var logits = new float[tokenizer.VocabSize];
        fragmented.ApplyMask(logits, false);
        Assert.True(float.IsNegativeInfinity(logits['m']));

        grammar.Accept(tokenizer.LookupToken(correct));
        Assert.True(grammar.IsComplete && !grammar.IsDead);
        var parser = new DeepSeek41OutputParser();
        parser.Init(true, new() { tool });
        var call = Assert.Single(parser.Add(prefix + correct, true).ToolCalls!);
        Assert.Equal(value, call.Arguments[parameter]);
        if (namedWeather) Assert.Equal("celsius", call.Arguments["units"]);
    }

    [Theory]
    [InlineData("<param")]
    [InlineData("</param")]
    [InlineData("<invoke")]
    [InlineData("</invoke")]
    public void RawParameterAndInvokeTagFamiliesCannotBeCommitted(string family)
    {
        var tokenizer = new ByteTokenizer(new[] { family });
        var grammar = DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required)
            .NewConstraint(tokenizer, false);
        grammar.AcceptBytes(Encoding.UTF8.GetBytes(Open + "<｜DSML｜ invoke name=\"get_weather\">" +
            "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris"));
        Assert.False(Allows(grammar, tokenizer.LookupToken(family)));
        foreach (char ch in family[..^1])
        {
            Assert.True(Allows(grammar, ch));
            grammar.Accept(ch);
        }
        Assert.False(Allows(grammar, family[^1]));
    }

    [Theory]
    [InlineData("<parameter_view>", false)]
    [InlineData("</parameter_name>", false)]
    [InlineData("<invoke name=\"example\">", false)]
    [InlineData("</invoke>", false)]
    [InlineData("<parameter_view>", true)]
    [InlineData("</parameter_name>", true)]
    [InlineData("<invoke name=\"example\">", true)]
    [InlineData("</invoke>", true)]
    public void ReservedTagFamilyValuesRemainLosslessThroughJsonAndToolHistory(string marker, bool nested)
    {
        string value = "  " + marker + " café🦊\n  ";
        object expected = nested ? new Dictionary<string, object> { [value] = new[] { value } } : value;
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson =
            "{\"type\":\"object\",\"properties\":{\"payload\":{\"type\":\"" + (nested ? "object" : "string") + "\"}},\"required\":[\"payload\"]}" };
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required);
        string original = Call("record", Parameter("payload", JsonSerializer.Serialize(expected), false));
        Assert.True(Accepts(plan, original));
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { tool });
        var calls = parser.Add(original, true).ToolCalls!;
        var call = Assert.Single(calls);
        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(call.Arguments["payload"]));

        string rendered = ChatTemplate.RenderDeepSeek41(new() { new ChatMessage { Role = "assistant", ToolCalls = calls } },
            addGenerationPrompt: false);
        int start = rendered.IndexOf(Open, StringComparison.Ordinal);
        int end = rendered.LastIndexOf(End, StringComparison.Ordinal) + End.Length;
        string replay = rendered[start..end];
        Assert.Contains("name=\"payload\" string=\"false\">", replay);
        Assert.DoesNotContain(marker, replay);
        Assert.True(Accepts(plan, replay));
        var replayParser = new DeepSeek41OutputParser();
        replayParser.Init(false, new() { tool });
        var replayed = Assert.Single(replayParser.Add(replay, true).ToolCalls!);
        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(replayed.Arguments["payload"]));
    }

    [Fact]
    public void RawArgumentsStillPermitOrdinaryXmlAndComparisonSigns()
    {
        const string value = "<x>東京</x> 1 < 2 and 3 > 2; </par> <inversion>";
        string text = Call("get_weather", Parameter("city", value) + Parameter("units", "celsius"));
        Assert.True(Accepts(DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required), text));
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { Weather() });
        var call = Assert.Single(parser.Add(text, true).ToolCalls!);
        Assert.Equal(value, call.Arguments["city"]);
    }

    [Theory]
    [InlineData("plain-invoke-close")]
    [InlineData("missing-required")]
    [InlineData("duplicate-parameter")]
    [InlineData("bad-enum")]
    [InlineData("unknown-tool")]
    [InlineData("incomplete-calls")]
    public void MalformedCheckpointOutputCannotFinishAsAConstrainedCall(string mutation)
    {
        string arguments = Parameter("city", "Paris") + Parameter("units", "celsius");
        string text = mutation switch
        {
            "plain-invoke-close" => Call("get_weather", arguments).Replace(InvokeEnd, "</invoke>\n</｜DSML｜ parameter>\n</invoke>"),
            "missing-required" => Call("get_weather", Parameter("city", "Paris")),
            "duplicate-parameter" => Call("get_weather", arguments + Parameter("city", "Rome")),
            "bad-enum" => Call("get_weather", Parameter("city", "Paris") + Parameter("units", "kelvin")),
            "unknown-tool" => Call("send_email", arguments),
            _ => Call("get_weather", arguments)[..^End.Length],
        };
        Assert.False(Accepts(DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required), text));
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void ParallelToolCallPolicyBoundsCompleteInvocations(bool parallel, bool acceptsTwo)
    {
        string invoke = Invoke("get_weather", Parameter("city", "Paris") + Parameter("units", "celsius"));
        var plan = DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required, parallelToolCalls: parallel);
        Assert.True(Accepts(plan, Open + invoke + End));
        Assert.Equal(acceptsTwo, Accepts(plan, Open + invoke + invoke + End));
    }

    [Fact]
    public void NamedChoiceAllowsOnlyTheRequestedDeclaredFunction()
    {
        var plan = DeepSeek41ToolGrammar.Compile(new[] { Weather(), Weather("forecast") }, DeepSeek41ToolChoice.Named, "forecast");
        string args = Parameter("city", "Paris") + Parameter("units", "celsius");
        Assert.True(Accepts(plan, Call("forecast", args)));
        Assert.False(Accepts(plan, Call("get_weather", args)));
        Assert.Throws<NotSupportedException>(() => DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Named, "undeclared"));
    }

    [Fact]
    public void NestedJsonParametersKeepTheirTypesAndRequiredProperties()
    {
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson = """
            {"type":"object","properties":{"data":{"type":"object","properties":{"optional":{"type":"boolean"},"items":{"type":"array","items":{"type":"integer"}}},"required":["items"],"additionalProperties":false}},"required":["data"]}
            """ };
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required);
        Assert.True(Accepts(plan, Call("record", Parameter("data", "{\"items\":[1,-2,300]}", false))));
        Assert.True(Accepts(plan, Call("record", Parameter("data", "{\"optional\":true,\"items\":[]}", false))));
        Assert.False(Accepts(plan, Call("record", Parameter("data", "{\"items\":[1.5]}", false))));
        Assert.False(Accepts(plan, Call("record", Parameter("data", "{}", false))));
        Assert.False(Accepts(plan, Call("record", Parameter("data", "{\"extra\":1,\"items\":[]}", false))));
    }

    [Theory]
    [InlineData("{\"type\":\"object\"}", true)]
    [InlineData("{\"type\":\"object\",\"additionalProperties\":true}", true)]
    [InlineData("{\"type\":\"object\",\"properties\":{}}", true)]
    [InlineData("{\"type\":\"object\",\"additionalProperties\":false}", false)]
    public void NestedOpenMapPreservesArbitraryKeysAndValues(string nestedSchema, bool acceptsMembers)
    {
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson =
            "{\"type\":\"object\",\"properties\":{\"data\":" + nestedSchema + "},\"required\":[\"data\"]}" };
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required);
        var expected = new Dictionary<string, object>
        {
            ["count"] = 42,
            ["ready"] = true,
            ["nested"] = new object[] { 1.25, new Dictionary<string, object> { ["city"] = "東京" } },
            ["</｜DSML｜ calls>"] = "</parameter> café🦊",
        };
        string text = Call("record", Parameter("data", JsonSerializer.Serialize(expected), false));
        Assert.Equal(acceptsMembers, Accepts(plan, text));
        Assert.True(Accepts(plan, Call("record", Parameter("data", "{}", false))));
        if (!acceptsMembers) return;
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { tool });
        var call = Assert.Single(parser.Add(text, true).ToolCalls!);
        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(call.Arguments["data"]));
    }

    [Fact]
    public void DeclaredObjectPropertiesUseDocumentedSubsetAndTypedExtraMapsFailExplicitly()
    {
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson = """
            {"type":"object","properties":{"data":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":true}},"required":["data"]}
            """ };
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required);
        Assert.True(Accepts(plan, Call("record", Parameter("data", "{\"name\":\"Paris\"}", false))));
        Assert.False(Accepts(plan, Call("record", Parameter("data", "{\"name\":\"Paris\",\"extra\":1}", false))));
        tool.ParametersSchemaJson = """
            {"type":"object","properties":{"data":{"type":"object","additionalProperties":{"type":"string"}}}}
            """;
        Assert.Throws<NotSupportedException>(() => DeepSeek41ToolGrammar.Compile(new[] { tool }));
    }

    [Theory]
    [InlineData("</parameter>")]
    [InlineData("</｜DSML｜ parameter>")]
    [InlineData("</｜DSML｜ invoke>")]
    [InlineData("</｜DSML｜ calls>")]
    [InlineData("<parameter")]
    public void JsonStringsEscapeEnvelopeDelimitersAndRoundTripLosslessly(string marker)
    {
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson = """
            {"type":"object","properties":{"data":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},"required":["data"]}
            """ };
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required);
        string json = JsonSerializer.Serialize(new { text = marker });
        string text = Call("record", Parameter("data", json, false));
        Assert.True(Accepts(plan, text));
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { tool });
        var call = Assert.Single(parser.Add(text, true).ToolCalls!);
        var data = Assert.IsType<Dictionary<string, object>>(call.Arguments["data"]);
        Assert.Equal(marker, data["text"]);
        Assert.False(Accepts(plan, Call("record", Parameter("data", "{\"text\":\"" + marker + "\"}", false))));
    }

    [Theory]
    [InlineData("</｜DSML｜ invoke>")]
    [InlineData("</｜DSML｜ calls>")]
    public void ReservedRawClosingMarkersCannotPrematurelyCompleteTheOuterEnvelope(string marker)
    {
        var plan = DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required);
        Assert.False(Accepts(plan, Call("get_weather", Parameter("city", marker) + Parameter("units", "celsius"))));
        var tool = Weather();
        tool.ParametersSchemaJson = "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\",\"const\":" + JsonSerializer.Serialize(marker) + "}}}";
        var constant = DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required);
        Assert.True(Accepts(constant, Call("get_weather", Parameter("x", JsonSerializer.Serialize(marker), false))));
    }

    [Fact]
    public void StringParameterContainingEveryReservedMarkerUsesLosslessJsonStringAlternative()
    {
        const string value = "  </parameter> </｜DSML｜ parameter> </｜DSML｜ invoke> </｜DSML｜ calls> <parameter <｜DSML｜ parameter <｜DSML｜ invoke café🦊\n  ";
        var plan = DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required);
        string text = Call("get_weather", Parameter("city", JsonSerializer.Serialize(value), false) + Parameter("units", "celsius"));
        Assert.True(Accepts(plan, text));
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { Weather() });
        var call = Assert.Single(parser.Add(text, true).ToolCalls!);
        Assert.Equal(value, call.Arguments["city"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ToolHistoryReplayPreservesDelimiterBearingStringAndNestedJsonArguments(bool nested)
    {
        const string value = "</｜DSML｜ calls> </｜DSML｜ invoke> </｜DSML｜ parameter> </parameter> café🦊";
        object expected = nested ? new Dictionary<string, object> { [value] = new[] { value } } : value;
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson =
            "{\"type\":\"object\",\"properties\":{\"payload\":{\"type\":\"" + (nested ? "object" : "string") + "\"}},\"required\":[\"payload\"]}" };
        string original = Call("record", Parameter("payload", JsonSerializer.Serialize(expected), false));
        Assert.True(Accepts(DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required), original));
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { tool });
        var calls = parser.Add(original, true).ToolCalls!;
        var call = Assert.Single(calls);
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Record the literal payload." },
            new() { Role = "assistant", ToolCalls = calls },
            new() { Role = "tool", ToolCallId = call.Id, Content = "Recorded." },
        };
        string rendered = ChatTemplate.RenderDeepSeek41(history, addGenerationPrompt: false);
        int start = rendered.IndexOf(Open, StringComparison.Ordinal);
        int end = rendered.LastIndexOf(End, StringComparison.Ordinal) + End.Length;
        string replay = rendered[start..end];
        var replayParser = new DeepSeek41OutputParser();
        replayParser.Init(false, new() { tool });
        var replayed = Assert.Single(replayParser.Add(replay, true).ToolCalls!);
        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(replayed.Arguments["payload"]));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void ToolHistoryReplayUsesExistingNewlineSeparator(int trailingNewlines)
    {
        string content = "Completed." + new string('\n', trailingNewlines);
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Run the check." },
            new() { Role = "assistant", Content = content, ToolCalls = new()
            {
                new() { Name = "check", Arguments = new() { ["path"] = "line1\nline2" } },
            } },
        };

        string rendered = ChatTemplate.RenderDeepSeek41(history, addGenerationPrompt: false);
        string expected = content + new string('\n', Math.Max(0, 2 - trailingNewlines)) + Open;
        Assert.Contains(expected, rendered);
        Assert.Contains("line1\nline2", rendered);
    }

    [Fact]
    public void OrdinaryToolHistoryRetainsRawStringsAndUnicodeJsonSpelling()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Record a snippet." },
            new() { Role = "assistant", ToolCalls = new()
            {
                new() { Name = "record", Arguments = new()
                {
                    ["text"] = "  <b>café</b>  ",
                    ["data"] = new Dictionary<string, object> { ["snippet"] = "<b>東京</b>" },
                } },
            } },
        };
        string rendered = ChatTemplate.RenderDeepSeek41(history, addGenerationPrompt: false);
        Assert.Contains("name=\"text\" string=\"true\">  <b>café</b>  </｜DSML｜ parameter>", rendered);
        Assert.Contains("name=\"data\" string=\"false\">{\"snippet\": \"<b>東京</b>\"}</｜DSML｜ parameter>", rendered);
    }

    [Fact]
    public void OptionalParametersCanBeOmittedAndArgumentlessFunctionsStayValid()
    {
        var tool = new ToolFunction { Name = "optional", ParametersSchemaJson = """
            {"type":"object","properties":{"first":{"type":"string"},"second":{"type":"number"},"third":{"type":"boolean"}}}
            """ };
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool, new ToolFunction { Name = "ping" } }, DeepSeek41ToolChoice.Required);
        Assert.True(Accepts(plan, Call("optional", "")));
        Assert.True(Accepts(plan, Call("optional", Parameter("second", "1.25e-2", false))));
        Assert.True(Accepts(plan, Call("ping", "")));
    }

    [Theory]
    [InlineData("{\"type\":\"string\",\"minLength\":3}")]
    [InlineData("{\"type\":\"number\",\"minimum\":0}")]
    [InlineData("{\"type\":[\"string\",\"null\"]}")]
    [InlineData("{\"oneOf\":[{\"type\":\"string\"}]}")]
    [InlineData("{\"type\":\"string\",\"pattern\":\"x.*\"}")]
    [InlineData("{\"type\":\"string\",\"enum\":[\"x\"],\"const\":\"y\"}")]
    public void UnsupportedSchemaAssertionsAreRejectedExplicitly(string parameterSchema)
    {
        var tool = new ToolFunction { Name = "strict", ParametersSchemaJson =
            "{\"type\":\"object\",\"properties\":{\"x\":" + parameterSchema + "}}" };
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41ToolGrammar.Compile(new[] { tool }));
        Assert.Contains("Unsupported DeepSeek V4.1", error.Message);
    }

    [Fact]
    public void ExtremeIntegerConstantsFailExplicitly()
    {
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson = """
            {"type":"object","properties":{"value":{"type":"integer","const":999999999999999999999999999999999999}},"required":["value"]}
            """ };
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41ToolGrammar.Compile(new[] { tool }));
        Assert.Contains("enum/const", error.Message);
    }

    [Theory]
    [InlineData("const", "-9223372036854775808")]
    [InlineData("const", "9223372036854775807")]
    [InlineData("enum", "-9223372036854775808")]
    [InlineData("enum", "9223372036854775807")]
    public void IntegerConstantBoundariesRoundTripExactlyAsInt64(string assertion, string integer)
    {
        string value = assertion == "enum" ? "[" + integer + "]" : integer;
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson =
            "{\"type\":\"object\",\"properties\":{\"value\":{\"type\":\"integer\",\"" + assertion + "\":" + value + "}},\"required\":[\"value\"]}" };
        var plan = DeepSeek41ToolGrammar.Compile(new[] { tool }, DeepSeek41ToolChoice.Required);
        string text = Call("record", Parameter("value", integer, false));
        Assert.True(Accepts(plan, text));
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, new() { tool });
        var call = Assert.Single(parser.Add(text, true).ToolCalls!);
        Assert.Equal(long.Parse(integer, System.Globalization.CultureInfo.InvariantCulture), Assert.IsType<long>(call.Arguments["value"]));
    }

    [Theory]
    [InlineData("const", "-9223372036854775809")]
    [InlineData("const", "9223372036854775808")]
    [InlineData("enum", "-9223372036854775809")]
    [InlineData("enum", "9223372036854775808")]
    [InlineData("const", "1.0000000000000000000000000000001")]
    public void IntegerConstantsOutsideSupportedExactEncodingAreRejected(string assertion, string integer)
    {
        string value = assertion == "enum" ? "[" + integer + "]" : integer;
        var tool = new ToolFunction { Name = "record", ParametersSchemaJson =
            "{\"type\":\"object\",\"properties\":{\"value\":{\"type\":\"integer\",\"" + assertion + "\":" + value + "}}}" };
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41ToolGrammar.Compile(new[] { tool }));
        Assert.Contains("Int64", error.Message);
    }

    [Fact]
    public void NoneAllowsOrdinaryAnswersButNeverCompleteToolMarkup()
    {
        var plan = DeepSeek41ToolGrammar.Compile(Array.Empty<ToolFunction>(), DeepSeek41ToolChoice.None);
        Assert.True(Accepts(plan, "Paris is sunny. café 東京🦊 <b>19°C</b>"));
        Assert.False(Accepts(plan, Call("get_weather", Parameter("city", "Paris"))));
    }

    [Fact]
    public void DsmlAllowlistIsScopedAndNeverAdmitsOtherControlTokens()
    {
        var tokenizer = new ByteTokenizer();
        var plan = DeepSeek41ToolGrammar.Compile(new[] { Weather() }, DeepSeek41ToolChoice.Required);
        var tool = plan.NewConstraint(tokenizer, false);
        tool.AcceptBytes(Encoding.UTF8.GetBytes("<"));
        var logits = new float[tokenizer.VocabSize];
        tool.ApplyMask(logits, false);
        Assert.True(float.IsFinite(logits[ByteTokenizer.Dsml]));
        Assert.True(float.IsNegativeInfinity(logits[ByteTokenizer.OtherControl]));

        var json = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        json.AcceptBytes(Encoding.UTF8.GetBytes("{\"x\":\""));
        Array.Fill(logits, 0);
        json.ApplyMask(logits, false);
        Assert.True(float.IsNegativeInfinity(logits[ByteTokenizer.Dsml]));
        Assert.Throws<ArgumentException>(() => GrammarTokenVocabulary.ForTokenizer(tokenizer, new[] { ByteTokenizer.Eos }));
    }

    [Fact]
    public void AutoPreservesPlainAnswersAndIgnoresQuotedCallsDuringThinking()
    {
        string answer = "Paris is sunny.";
        string call = Call("get_weather", Parameter("city", "Paris") + Parameter("units", "celsius"));
        var tokenizer = new ByteTokenizer(new[] { answer, "I could write " + Open + " in reasoning.", "</think>" + call });
        var plan = DeepSeek41ToolGrammar.Compile(new[] { Weather() });
        var ordinary = plan.NewConstraint(tokenizer, false);
        ordinary.Accept(tokenizer.LookupToken(answer));
        Assert.False(ordinary.IsActive);
        Assert.True(ordinary.IsComplete);
        var thinking = plan.NewConstraint(tokenizer, true);
        thinking.Accept(tokenizer.LookupToken("I could write " + Open + " in reasoning."));
        Assert.False(thinking.IsActive);
        thinking.Accept(tokenizer.LookupToken("</think>" + call));
        Assert.True(thinking.IsActive && thinking.IsComplete && !thinking.IsDead);
    }

    [Fact]
    public void SkillRoundsForkAnUnconsumedConstraintEvenWhenCodingDefaultsCloneSettings()
    {
        var tokenizer = new ByteTokenizer(new[] { Open });
        var original = SamplingConfig.Greedy;
        original.Grammar = DeepSeek41ToolGrammar.Compile(new[] { Weather() }).NewConstraint(tokenizer, false);
        var codingDefaults = original.Clone();
        var first = ModelService.SamplingForDeepSeek41SkillRound("deepseek41", codingDefaults, original);
        var second = ModelService.SamplingForDeepSeek41SkillRound("deepseek41", codingDefaults, original);
        first.Grammar!.Accept(tokenizer.LookupToken(Open));
        Assert.True(first.Grammar.IsActive);
        Assert.False(second.Grammar!.IsActive || original.Grammar.IsActive);
        Assert.Same(codingDefaults, ModelService.SamplingForDeepSeek41SkillRound("qwen35", codingDefaults, original));
    }

    [Theory]
    [InlineData("\"auto\"", DeepSeek41ToolChoice.Auto)]
    [InlineData("\"none\"", DeepSeek41ToolChoice.None)]
    [InlineData("\"required\"", DeepSeek41ToolChoice.Required)]
    [InlineData("{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}", DeepSeek41ToolChoice.Named)]
    public void OpenAiPoliciesCompileBeforeGeneration(string choice, DeepSeek41ToolChoice expected)
    {
        using var body = JsonDocument.Parse("{\"tool_choice\":" + choice + "}");
        var tools = new List<ToolFunction> { Weather() };
        var plan = OpenAIChatAdapter.PrepareDeepSeek41ToolGrammar(body.RootElement, tools, tools, null);
        Assert.Equal(expected, plan.Choice);
    }

    [Theory]
    [InlineData("{\"tool_choice\":\"bogus\"}")]
    [InlineData("{\"tool_choice\":false}")]
    [InlineData("{\"tool_choice\":{\"function\":{\"name\":\"get_weather\"}}}")]
    [InlineData("{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"undeclared\"}}}")]
    [InlineData("{\"parallel_tool_calls\":\"false\"}")]
    [InlineData("{\"tools\":[{\"type\":\"web_search\"}]}")]
    public void UnsupportedOpenAiPoliciesNeverSilentlyBecomeAuto(string request)
    {
        using var body = JsonDocument.Parse(request);
        var tools = new List<ToolFunction> { Weather() };
        var error = Record.Exception(() => OpenAIChatAdapter.PrepareDeepSeek41ToolGrammar(body.RootElement, tools, tools, null));
        Assert.True(error is NotSupportedException or JsonException, error?.ToString());
    }

    [Fact]
    public void OriginalNestedSchemaIsPreservedForV41PromptAndCompiler()
    {
        const string schema = "{\"type\":\"object\",\"properties\":{\"codes\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}},\"required\":[\"codes\"],\"additionalProperties\":false}";
        using var body = JsonDocument.Parse("{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"record\",\"parameters\":" + schema + "}}]}");
        var tool = Assert.Single(ToolFunctionParser.ParseOpenAI(body.RootElement));
        Assert.Equal(schema, tool.ParametersSchemaJson);
        string prompt = ChatTemplate.RenderDeepSeek41(new() { new ChatMessage { Role = "user", Content = "Record codes" } }, tools: new() { tool });
        Assert.Contains("\"items\": {\"type\": \"integer\"}", prompt);
        Assert.Contains("\"additionalProperties\": false", prompt);
    }

    private static ToolFunction Weather(string name = "get_weather") => new() { Name = name, ParametersSchemaJson = WeatherSchema };
    private static string Parameter(string name, string value, bool isString = true)
        => "<｜DSML｜ parameter name=\"" + name + "\" string=\"" + (isString ? "true" : "false") + "\">" + value + "</｜DSML｜ parameter>\n";
    private static string Invoke(string name, string arguments) => "<｜DSML｜ invoke name=\"" + name + "\">\n" + arguments + InvokeEnd + "\n";
    private static string Call(string name, string arguments) => Open + "\n" + Invoke(name, arguments) + End;
    private static bool Accepts(DeepSeek41ToolGrammar plan, string text)
    {
        var constraint = new GrammarConstraint(Grammar.Parse(plan.Source), new ByteTokenizer());
        constraint.AcceptBytes(Encoding.UTF8.GetBytes(text));
        return !constraint.IsDead && constraint.IsComplete;
    }

    private static bool Allows(GrammarConstraint grammar, int token)
        => (grammar.CurrentMask()[token >> 6] & (1UL << (token & 63))) != 0;

    internal sealed class ByteTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        public const int Dsml = 256, ThinkClose = 257, Eos = 258, OtherControl = 259;
        public ByteTokenizer(IEnumerable<string>? pieces = null)
            => Vocab = Enumerable.Range(0, 256).Select(i => ((char)i).ToString())
                .Concat(new[] { "｜DSML｜", "</think>", "<eos>", "<other-control>" }).Concat(pieces ?? []).ToArray();
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { Eos };
        public int VocabSize => Vocab.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => new[] { Dsml, ThinkClose, Eos, OtherControl };
        public List<int> Encode(string text, bool addSpecial = true) => Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList();
        public string Decode(List<int> ids)
        {
            var bytes = new List<byte>();
            foreach (int id in ids) AppendTokenBytes(id, bytes);
            return Encoding.UTF8.GetString(bytes.ToArray());
        }
        public void AppendTokenBytes(int id, List<byte> bytes)
        {
            if (id < 256) bytes.Add((byte)id);
            else bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[id]));
        }
        public bool IsEos(int id) => id == Eos;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }
}
