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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Micro-benchmark for <see cref="ToolFunctionParser"/>. Implemented as XUnit
/// Facts so it runs under <c>dotnet test</c> alongside the correctness suite;
/// numbers are printed to stdout (look for the <c>[ToolFunctionParser.*]</c>
/// banner). Needs no model, no GPU and no native init.
/// <para>
/// This sits on the synchronous prologue of every chat request that declares
/// tools — <c>OpenAIChatAdapter.ChatCompletionsAsync</c> parses the spec before
/// it takes a queue ticket — so an agent harness that re-sends its whole tool
/// catalogue on every turn pays this cost per request, ahead of any inference.
/// The corpus below is sized for that case: a catalogue the size of a real MCP
/// server's, not a single toy tool.
/// </para>
/// <para>
/// <see cref="Parse_KindTolerantVersusLegacyStringOnlyParser"/> A/Bs the current
/// kind-checked parser against <see cref="LegacyParseOpenAI"/>, a verbatim copy
/// of the pre-#142 implementation. That implementation threw on half the specs
/// in this file, so the A/B runs on the all-strings corpus it could still handle
/// — the only input on which the two are comparable — to show that reading
/// <see cref="JsonValueKind"/> before each access costs nothing.
/// </para>
/// </summary>
public class ToolFunctionParserBenchmark
{
    // A tool catalogue the size of a working agent harness: 12 tools x 6
    // parameters, each parameter carrying a description and half of them an
    // enum. ~14 KB of JSON, comparable to what a filesystem+shell+search MCP
    // server advertises.
    private const int ToolCount = 12;
    private const int ParamsPerTool = 6;

    private static readonly string StringOnlySpec = BuildSpec(mixedKinds: false);
    private static readonly string MixedKindSpec = BuildSpec(mixedKinds: true);

    // Both candidates are warmed before either is timed, and the rounds
    // alternate between them, so neither is charged for tiered-JIT promotion or
    // for a cold ArrayPool — measuring them one after the other made whichever
    // ran first look ~3x slower than it is.
    private const int Warmup = 500;
    private const int Rounds = 6;
    private const int ItersPerRound = 1000;

    /// <summary>
    /// One process-wide warm-up, run once before any measurement. .NET promotes
    /// a method to its optimized tier on call count <i>and</i> elapsed time, so a
    /// per-benchmark warm-up loop is not enough on its own: whichever benchmark
    /// happened to run first still measured tier-0 code and read ~3.5x slower
    /// than the very same parser did one call later.
    /// </summary>
    private static readonly bool Warmed = WarmUp();

    private static bool WarmUp()
    {
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < 750)
        {
            Run(StringOnlySpec, 20, ToolFunctionParser.ParseOpenAI);
            Run(StringOnlySpec, 20, LegacyParseOpenAI);
            Run(MixedKindSpec, 20, ToolFunctionParser.ParseOpenAI);
        }
        return true;
    }

    [Fact]
    public void Parse_KindTolerantVersusLegacyStringOnlyParser()
    {
        Assert.True(Warmed);
        Console.WriteLine($"[ToolFunctionParser] corpus: {ToolCount} tools x {ParamsPerTool} params, " +
                          $"{Encoding.UTF8.GetByteCount(StringOnlySpec) / 1024.0:F1} KB JSON, " +
                          $"{Rounds} rounds x {ItersPerRound} parses, best round reported");

        var candidates = new (string Label, Func<JsonElement, List<ToolFunction>> Parse)[]
        {
            ("legacy (pre-#142, string-only)", LegacyParseOpenAI),
            ("current (kind-checked)", ToolFunctionParser.ParseOpenAI),
        };
        double[] best = BenchAlternating(StringOnlySpec, candidates);

        double legacy = best[0], current = best[1];
        Console.WriteLine($"[ToolFunctionParser] current / legacy = {current / legacy:F3}x " +
                          $"({(legacy - current) / legacy * 100:+0.0;-0.0}% faster)");

        // The A/B is only meaningful if both parsers saw the same catalogue.
        using var doc = JsonDocument.Parse(StringOnlySpec);
        Assert.Equal(ToolCount, ToolFunctionParser.ParseOpenAI(doc.RootElement).Count);
        Assert.Equal(ToolCount, LegacyParseOpenAI(doc.RootElement).Count);
    }

    [Fact]
    public void Parse_MixedKindSpec_IsNotAPathologicalCase()
    {
        Assert.True(Warmed);
        Console.WriteLine($"[ToolFunctionParser] mixed-kind corpus: " +
                          $"{Encoding.UTF8.GetByteCount(MixedKindSpec) / 1024.0:F1} KB JSON " +
                          $"(integer/boolean enums, union types — the shapes that used to throw)");

        // The same parser over both corpora: the kinds the legacy parser could
        // not read must not cost materially more than the ones it could.
        var candidates = new (string, Func<JsonElement, List<ToolFunction>>)[]
        {
            ("current (all strings)", ToolFunctionParser.ParseOpenAI),
        };
        double strings = BenchAlternating(StringOnlySpec, candidates)[0];
        double mixed = BenchAlternating(MixedKindSpec,
            [("current (mixed kinds)", ToolFunctionParser.ParseOpenAI)])[0];

        Console.WriteLine($"[ToolFunctionParser] mixed / strings = {mixed / strings:F3}x");

        using var doc = JsonDocument.Parse(MixedKindSpec);
        var tools = ToolFunctionParser.ParseOpenAI(doc.RootElement);
        Assert.Equal(ToolCount, tools.Count);
        Assert.All(tools, t => Assert.Equal(ParamsPerTool, t.Parameters.Count));
    }

    /// <summary>
    /// Time every candidate over the same corpus, reporting each one's fastest
    /// round in microseconds per spec. Returns the results in candidate order.
    /// </summary>
    private static double[] BenchAlternating(
        string json, (string Label, Func<JsonElement, List<ToolFunction>> Parse)[] candidates)
    {
        var best = new double[candidates.Length];
        Array.Fill(best, double.MaxValue);
        int checksum = 0;

        foreach (var candidate in candidates)
            checksum += Run(json, Warmup, candidate.Parse);

        for (int round = 0; round < Rounds; round++)
        {
            for (int c = 0; c < candidates.Length; c++)
            {
                var sw = Stopwatch.StartNew();
                checksum += Run(json, ItersPerRound, candidates[c].Parse);
                sw.Stop();
                best[c] = Math.Min(best[c], sw.Elapsed.TotalMilliseconds * 1000.0 / ItersPerRound);
            }
        }

        for (int c = 0; c < candidates.Length; c++)
        {
            Console.WriteLine($"[ToolFunctionParser] {candidates[c].Label,-32} {best[c],8:F2} us/spec  " +
                              $"{1_000_000.0 / best[c],10:N0} spec/s");
        }
        Console.WriteLine($"[ToolFunctionParser] (checksum {checksum})");
        return best;
    }

    private static int Run(string json, int iters, Func<JsonElement, List<ToolFunction>> parse)
    {
        // Each iteration re-parses the document too, because that is what the
        // request path actually does — timing the parser against a pre-parsed
        // JsonDocument would measure a call that never happens in production.
        int tools = 0;
        for (int i = 0; i < iters; i++)
        {
            using var doc = JsonDocument.Parse(json);
            tools += parse(doc.RootElement).Count;
        }
        return tools;
    }

    /// <summary>
    /// The parser exactly as it stood before the issue #142 fix
    /// (<c>ToolFunctionParser.ParseOpenAI</c>/<c>ParseFunction</c> at commit
    /// b353392), kept only as the benchmark's baseline. It throws
    /// <see cref="InvalidOperationException"/> on any tool spec that puts a
    /// non-string where it expected one, so it can only be run on
    /// <see cref="StringOnlySpec"/>.
    /// </summary>
    private static List<ToolFunction> LegacyParseOpenAI(JsonElement body)
    {
        if (!body.TryGetProperty("tools", out var toolsEl) || toolsEl.ValueKind != JsonValueKind.Array)
            return null;

        var tools = new List<ToolFunction>();
        foreach (var toolEl in toolsEl.EnumerateArray())
        {
            string type = toolEl.TryGetProperty("type", out var t) ? t.GetString() : "function";
            if (type != "function") continue;
            if (!toolEl.TryGetProperty("function", out var fnEl)) continue;

            var tf = new ToolFunction
            {
                Name = fnEl.TryGetProperty("name", out var n) ? n.GetString() : "",
                Description = fnEl.TryGetProperty("description", out var d) ? d.GetString() : ""
            };

            if (fnEl.TryGetProperty("parameters", out var paramsEl))
            {
                if (paramsEl.TryGetProperty("properties", out var propsEl) &&
                    propsEl.ValueKind == JsonValueKind.Object)
                {
                    tf.Parameters = new Dictionary<string, ToolParameter>();
                    foreach (var prop in propsEl.EnumerateObject())
                    {
                        var tp = new ToolParameter
                        {
                            Type = prop.Value.TryGetProperty("type", out var pt) ? pt.GetString() : "string",
                            Description = prop.Value.TryGetProperty("description", out var pd) ? pd.GetString() : null
                        };
                        if (prop.Value.TryGetProperty("enum", out var enumEl) && enumEl.ValueKind == JsonValueKind.Array)
                            tp.Enum = enumEl.EnumerateArray().Select(e => e.GetString()).ToList();
                        tf.Parameters[prop.Name] = tp;
                    }
                }
                if (paramsEl.TryGetProperty("required", out var reqEl) && reqEl.ValueKind == JsonValueKind.Array)
                    tf.Required = reqEl.EnumerateArray().Select(e => e.GetString()).ToList();
            }

            tools.Add(tf);
        }
        return tools.Count > 0 ? tools : null;
    }

    /// <summary>
    /// Build an OpenAI Chat Completions <c>tools</c> catalogue. With
    /// <paramref name="mixedKinds"/> the schemas use the JSON Schema spellings
    /// that the legacy parser could not read — integer and boolean enums, and
    /// <c>"type": ["…", "null"]</c> for nullable fields — while keeping the same
    /// tool and parameter counts so the two corpora stay comparable.
    /// </summary>
    private static string BuildSpec(bool mixedKinds)
    {
        var tools = new List<object>(ToolCount);
        for (int t = 0; t < ToolCount; t++)
        {
            var properties = new Dictionary<string, object>(ParamsPerTool);
            for (int p = 0; p < ParamsPerTool; p++)
            {
                var schema = new Dictionary<string, object>
                {
                    // p % 3 spreads the three type spellings over the parameters;
                    // the union form is the one a nullable field gets.
                    ["type"] = !mixedKinds ? "string"
                             : p % 3 == 0 ? new[] { "string", "null" }
                             : p % 3 == 1 ? "integer"
                             : (object)"string",
                    ["description"] = $"Parameter {p} of tool {t}, described about as fully as a real one is."
                };

                if (p % 2 == 0)
                {
                    schema["enum"] = !mixedKinds ? new object[] { "alpha", "beta", "gamma" }
                                   : p % 3 == 1 ? new object[] { 0, 1, 2, 3 }
                                   : p % 3 == 2 ? new object[] { true, false }
                                   : new object[] { "alpha", "beta", "gamma" };
                }

                properties[$"param_{p}"] = schema;
            }

            tools.Add(new
            {
                type = "function",
                function = new
                {
                    name = $"tool_{t}",
                    description = $"Benchmark tool number {t}, with a description of the length a real tool carries.",
                    parameters = new Dictionary<string, object>
                    {
                        ["type"] = "object",
                        ["properties"] = properties,
                        ["required"] = new[] { "param_0", "param_1" }
                    }
                }
            });
        }

        return JsonSerializer.Serialize(new
        {
            model = "bench",
            messages = new[] { new { role = "user", content = "go" } },
            tools
        });
    }
}
