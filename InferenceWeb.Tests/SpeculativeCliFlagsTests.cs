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
using System.IO;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The speculative-decoding flags translated to their environment variables,
/// under both the current <c>--spec*</c> / <c>TS_SPEC_*</c> spelling and the
/// legacy <c>--mtp-*</c> / <c>TS_MTP_*</c> one. Shared by
/// TensorSharp.Cli and TensorSharp.Server. This is the only seam either host's
/// MTP command line can be tested through: the CLI's own parser is a switch
/// inside a private <c>MainCore</c> with no return value.
///
/// The env var is the contract rather than a parsed value object because the
/// request has to reach the model LOADER — glm-dsa decides from
/// <c>TS_MTP_SPEC</c> whether to page a whole extra 256-expert decoder layer
/// into VRAM, and sizes its graph cache from <c>TS_MTP_DRAFT</c>, both while the
/// model is loading.
/// </summary>
public sealed class SpeculativeCliFlagsTests : IDisposable
{
    private readonly EnvScope _env = new();

    public SpeculativeCliFlagsTests()
    {
        // Every test starts from "operator configured nothing".
        _env.ClearSpeculationVars();
    }

    public void Dispose() => _env.Dispose();

    [Fact]
    public void Apply_SpecSpec_TurnsSpeculationOnForTheScheduler()
    {
        bool applied = SpeculativeCliFlags.Apply(new[] { "--mtp-spec" });

        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void Apply_NoMtpSpec_TurnsSpeculationOffOverAnExportedEnvVar()
    {
        _env.Set("TS_MTP_SPEC", "1");

        Assert.True(SpeculativeCliFlags.Apply(new[] { "--no-mtp-spec" }));

        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.False(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    [Fact]
    public void Apply_NoFlags_LeavesTheEnvironmentAlone()
    {
        _env.Set("TS_MTP_SPEC", "1");

        Assert.False(SpeculativeCliFlags.Apply(new[] { "--model", "x.gguf", "--backend", "ggml_cuda" }));

        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
    }

    [Theory]
    [InlineData("--mtp-draft", "4")]
    [InlineData("--mtp-draft=4", null)]
    public void Apply_SpecDraft_AcceptsBothSpellings(string first, string second)
    {
        string[] args = second == null ? new[] { first } : new[] { first, second };

        Assert.True(SpeculativeCliFlags.Apply(args));

        Assert.Equal("4", Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
        Assert.Equal(4, SchedulerConfig.FromEnvironment().Speculation.MaxDraftTokens);
    }

    [Fact]
    public void Apply_SpecPmin_ReachesTheSchedulerAsANullableProbability()
    {
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--mtp-pmin", "0.55" }));

        Assert.Equal(0.55f, SchedulerConfig.FromEnvironment().Speculation.MinDraftProb);
    }

    [Fact]
    public void Apply_WithoutMtpPmin_LeavesTheGateUnsetForTheDrafterToChoose()
    {
        // The per-token gate (top-1 probability over the head's top-10 logits,
        // default 0.75) and the block gate (cumulative prefix-acceptance product,
        // default 0.35) threshold different quantities, so "unset" has to survive
        // all the way to SpeculativeExecution rather than collapsing to one
        // shared number that badly mis-gates the other kind.
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--mtp-spec" }));

        Assert.Null(SchedulerConfig.FromEnvironment().Speculation.MinDraftProb);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("abc")]
    // Above the bound the glm-dsa native loader silently ignores when it sizes its
    // graph cache from the same variable, so a larger window would decode through a
    // cache too small for the graph shapes it produces.
    [InlineData("65")]
    [InlineData("1000")]
    public void Apply_SpecDraftWithAnUnusableValue_FailsFastNamingTheFlag(string value)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--mtp-draft", value }));

        Assert.Contains("--mtp-draft", ex.Message);
        Assert.Null(Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("1.5")]
    [InlineData("nope")]
    public void Apply_SpecPminOutsideTheUnitInterval_FailsFastNamingTheFlag(string value)
    {
        // The (0, 1] bound exists ONLY here: SchedulerConfig reads the variable
        // back with a plain float parse, so a value of 5 would be accepted there
        // and would reject every draft while speculation still logged as armed.
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--mtp-pmin", value }));

        Assert.Contains("--mtp-pmin", ex.Message);
        Assert.Null(Environment.GetEnvironmentVariable("TS_MTP_PMIN"));
    }

    [Fact]
    public void Apply_ValueFlagWithNoValue_SaysSoInsteadOfIndexingPastTheEnd()
    {
        // Every other CLI flag does args[++i] and surfaces as an unhandled
        // IndexOutOfRangeException; these say which option is missing a value.
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--model", "x.gguf", "--mtp-draft" }));

        Assert.Contains("--mtp-draft", ex.Message);
    }

    [Fact]
    public void Apply_SpecDraftModel_DoesNotSwallowMtpDraft()
    {
        // "--mtp-draft" is a strict prefix of "--mtp-draft-model". A parser that
        // matched on a prefix would route the GGUF path into TS_MTP_DRAFT, where
        // an int parse discards it back to the default while the draft model
        // silently never loads.
        string gguf = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            Assert.True(SpeculativeCliFlags.Apply(new[]
            {
                "--mtp-spec", "--mtp-draft", "6", "--mtp-draft-model", gguf,
            }));

            Assert.Equal("6", Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
            Assert.Equal(gguf, Environment.GetEnvironmentVariable("TS_MTP_DRAFT_MODEL"));
        }
        finally
        {
            File.Delete(gguf);
        }
    }

    [Fact]
    public void Apply_SpecDraftModelThatDoesNotExist_FailsFast()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--mtp-draft-model", "/no/such/draft.gguf" }));

        Assert.Contains("--mtp-draft-model", ex.Message);
    }

    [Fact]
    public void Apply_SpecDraftAtTheBound_IsAccepted()
    {
        Assert.True(SpeculativeCliFlags.Apply(
            new[] { "--mtp-draft", SpeculativeCliFlags.MaxDraftTokens.ToString() }));

        Assert.Equal(SpeculativeCliFlags.MaxDraftTokens,
            SchedulerConfig.FromEnvironment().Speculation.MaxDraftTokens);
    }

    [Fact]
    public void Apply_LastOccurrenceWins()
    {
        // Config-file expansion (ConfigFileArgs.Expand) splices file-derived
        // tokens ahead of the real command line, so the operator's own flag has
        // to be the one that survives.
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--mtp-spec", "--no-mtp-spec" }));
        Assert.False(SchedulerConfig.FromEnvironment().Speculation.Enabled);

        Assert.True(SpeculativeCliFlags.Apply(new[] { "--no-mtp-spec", "--mtp-spec" }));
        Assert.True(SchedulerConfig.FromEnvironment().Speculation.Enabled);
    }

    // ----- the dual TS_SPEC_* / TS_MTP_* spelling -----

    [Fact]
    public void Apply_PublishesBothSpellings_SoTheNativeLoaderStillSeesTheRequest()
    {
        // The glm-dsa NATIVE loader reads TS_MTP_SPEC / TS_MTP_DRAFT from C++ while
        // the model is loading (it decides whether to page a whole extra 256-expert
        // decoder layer into VRAM, and sizes its graph cache). Publishing only the
        // current spelling would leave that loader blind, and speculation would go
        // quiet with nothing in the log to explain it.
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec", "--spec-draft", "5", "--spec-pmin", "0.6" }));

        Assert.Equal("1", Environment.GetEnvironmentVariable(SpeculationEnvVars.Enabled));
        Assert.Equal("1", Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyEnabled));
        Assert.Equal("5", Environment.GetEnvironmentVariable(SpeculationEnvVars.Draft));
        Assert.Equal("5", Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyDraft));
        Assert.Equal("0.6", Environment.GetEnvironmentVariable(SpeculationEnvVars.PMin));
        Assert.Equal("0.6", Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyPMin));
    }

    [Fact]
    public void Apply_LegacyMtpSpellings_AreStillHonoured()
    {
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--mtp-spec", "--mtp-draft", "3", "--mtp-pmin", "0.9" }));

        var cfg = SchedulerConfig.FromEnvironment().Speculation;
        Assert.True(cfg.Enabled);
        Assert.Equal(3, cfg.MaxDraftTokens);
        Assert.Equal(0.9f, cfg.MinDraftProb);
    }

    [Fact]
    public void FromEnvironment_ReadsLegacyWhenOnlyLegacyIsSet()
    {
        // A deployment that exports TS_MTP_* directly (no CLI flag) must keep
        // working unchanged.
        _env.Set(SpeculationEnvVars.LegacyEnabled, "1");
        _env.Set(SpeculationEnvVars.LegacyDraft, "12");

        var cfg = SpeculationOptions.FromEnvironment();
        Assert.True(cfg.Enabled);
        Assert.Equal(12, cfg.MaxDraftTokens);
    }

    [Fact]
    public void SpecType_UnknownAlgorithm_FailsFastListingTheKnownOnes()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--spec-type", "eagle-9000" }));

        Assert.Contains("--spec-type", ex.Message);
        Assert.Contains("eagle-9000", ex.Message);
        Assert.Contains(SpeculatorRegistry.NGram, ex.Message);
    }

    [Fact]
    public void SpecType_KnownAlgorithm_ReachesTheSchedulerConfig()
    {
        Assert.True(SpeculativeCliFlags.Apply(new[] { "--spec", "--spec-type", SpeculatorRegistry.NGram }));

        Assert.Equal(SpeculatorRegistry.NGram, SchedulerConfig.FromEnvironment().Speculation.SpeculatorName);
    }

    [Fact]
    public void SpecDraftModel_DoesNotCollideWithSpecDraft()
    {
        // --spec-draft is a prefix of --spec-draft-model; the parser must route
        // each to its own variable rather than mis-reading the longer flag.
        string gguf = Path.Combine(Path.GetTempPath(), $"spec-draft-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            Assert.True(SpeculativeCliFlags.Apply(
                new[] { "--spec-draft", "6", "--spec-draft-model", gguf }));

            Assert.Equal("6", Environment.GetEnvironmentVariable(SpeculationEnvVars.Draft));
            Assert.Equal(gguf, Environment.GetEnvironmentVariable(SpeculationEnvVars.DraftModel));
            Assert.Equal(gguf, Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyDraftModel));
        }
        finally
        {
            File.Delete(gguf);
        }
    }

    [Fact]
    public void Apply_InvalidValue_NamesTheSpellingTheOperatorActuallyTyped()
    {
        // Being told "--spec-draft is invalid" after typing --mtp-draft sends the
        // operator looking for a flag they never used.
        var legacy = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--mtp-draft", "999" }));
        Assert.Contains("--mtp-draft", legacy.Message);

        var current = Assert.Throws<ArgumentException>(() =>
            SpeculativeCliFlags.Apply(new[] { "--spec-draft", "999" }));
        Assert.Contains("--spec-draft", current.Message);
    }
}
