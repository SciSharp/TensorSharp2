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
/// The <c>--mtp-*</c> flags translated to <c>TS_MTP_*</c>, shared by
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
public sealed class MtpSpeculativeCliFlagsTests : IDisposable
{
    private readonly EnvScope _env = new();

    public MtpSpeculativeCliFlagsTests()
    {
        // Every test starts from "operator configured nothing".
        _env.Set("TS_MTP_SPEC", null);
        _env.Set("TS_MTP_DRAFT", null);
        _env.Set("TS_MTP_PMIN", null);
        _env.Set("TS_MTP_DRAFT_MODEL", null);
    }

    public void Dispose() => _env.Dispose();

    [Fact]
    public void Apply_MtpSpec_TurnsSpeculationOnForTheScheduler()
    {
        bool applied = MtpSpeculativeCliFlags.Apply(new[] { "--mtp-spec" });

        Assert.True(applied);
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.True(SchedulerConfig.FromEnvironment().MtpSpeculativeEnabled);
    }

    [Fact]
    public void Apply_NoMtpSpec_TurnsSpeculationOffOverAnExportedEnvVar()
    {
        _env.Set("TS_MTP_SPEC", "1");

        Assert.True(MtpSpeculativeCliFlags.Apply(new[] { "--no-mtp-spec" }));

        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
        Assert.False(SchedulerConfig.FromEnvironment().MtpSpeculativeEnabled);
    }

    [Fact]
    public void Apply_NoFlags_LeavesTheEnvironmentAlone()
    {
        _env.Set("TS_MTP_SPEC", "1");

        Assert.False(MtpSpeculativeCliFlags.Apply(new[] { "--model", "x.gguf", "--backend", "ggml_cuda" }));

        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MTP_SPEC"));
    }

    [Theory]
    [InlineData("--mtp-draft", "4")]
    [InlineData("--mtp-draft=4", null)]
    public void Apply_MtpDraft_AcceptsBothSpellings(string first, string second)
    {
        string[] args = second == null ? new[] { first } : new[] { first, second };

        Assert.True(MtpSpeculativeCliFlags.Apply(args));

        Assert.Equal("4", Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
        Assert.Equal(4, SchedulerConfig.FromEnvironment().MtpMaxDraftTokens);
    }

    [Fact]
    public void Apply_MtpPmin_ReachesTheSchedulerAsANullableProbability()
    {
        Assert.True(MtpSpeculativeCliFlags.Apply(new[] { "--mtp-pmin", "0.55" }));

        Assert.Equal(0.55f, SchedulerConfig.FromEnvironment().MtpMinDraftProb);
    }

    [Fact]
    public void Apply_WithoutMtpPmin_LeavesTheGateUnsetForTheDrafterToChoose()
    {
        // The per-token gate (top-1 probability over the head's top-10 logits,
        // default 0.75) and the block gate (cumulative prefix-acceptance product,
        // default 0.35) threshold different quantities, so "unset" has to survive
        // all the way to MtpSpeculativeExecution rather than collapsing to one
        // shared number that badly mis-gates the other kind.
        Assert.True(MtpSpeculativeCliFlags.Apply(new[] { "--mtp-spec" }));

        Assert.Null(SchedulerConfig.FromEnvironment().MtpMinDraftProb);
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
    public void Apply_MtpDraftWithAnUnusableValue_FailsFastNamingTheFlag(string value)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            MtpSpeculativeCliFlags.Apply(new[] { "--mtp-draft", value }));

        Assert.Contains("--mtp-draft", ex.Message);
        Assert.Null(Environment.GetEnvironmentVariable("TS_MTP_DRAFT"));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("1.5")]
    [InlineData("nope")]
    public void Apply_MtpPminOutsideTheUnitInterval_FailsFastNamingTheFlag(string value)
    {
        // The (0, 1] bound exists ONLY here: SchedulerConfig reads the variable
        // back with a plain float parse, so a value of 5 would be accepted there
        // and would reject every draft while speculation still logged as armed.
        var ex = Assert.Throws<ArgumentException>(() =>
            MtpSpeculativeCliFlags.Apply(new[] { "--mtp-pmin", value }));

        Assert.Contains("--mtp-pmin", ex.Message);
        Assert.Null(Environment.GetEnvironmentVariable("TS_MTP_PMIN"));
    }

    [Fact]
    public void Apply_ValueFlagWithNoValue_SaysSoInsteadOfIndexingPastTheEnd()
    {
        // Every other CLI flag does args[++i] and surfaces as an unhandled
        // IndexOutOfRangeException; these say which option is missing a value.
        var ex = Assert.Throws<ArgumentException>(() =>
            MtpSpeculativeCliFlags.Apply(new[] { "--model", "x.gguf", "--mtp-draft" }));

        Assert.Contains("--mtp-draft", ex.Message);
    }

    [Fact]
    public void Apply_MtpDraftModel_DoesNotSwallowMtpDraft()
    {
        // "--mtp-draft" is a strict prefix of "--mtp-draft-model". A parser that
        // matched on a prefix would route the GGUF path into TS_MTP_DRAFT, where
        // an int parse discards it back to the default while the draft model
        // silently never loads.
        string gguf = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".gguf");
        File.WriteAllBytes(gguf, new byte[] { 1, 2, 3 });
        try
        {
            Assert.True(MtpSpeculativeCliFlags.Apply(new[]
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
    public void Apply_MtpDraftModelThatDoesNotExist_FailsFast()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            MtpSpeculativeCliFlags.Apply(new[] { "--mtp-draft-model", "/no/such/draft.gguf" }));

        Assert.Contains("--mtp-draft-model", ex.Message);
    }

    [Fact]
    public void Apply_MtpDraftAtTheBound_IsAccepted()
    {
        Assert.True(MtpSpeculativeCliFlags.Apply(
            new[] { "--mtp-draft", MtpSpeculativeCliFlags.MaxDraftTokens.ToString() }));

        Assert.Equal(MtpSpeculativeCliFlags.MaxDraftTokens,
            SchedulerConfig.FromEnvironment().MtpMaxDraftTokens);
    }

    [Fact]
    public void Apply_LastOccurrenceWins()
    {
        // Config-file expansion (ConfigFileArgs.Expand) splices file-derived
        // tokens ahead of the real command line, so the operator's own flag has
        // to be the one that survives.
        Assert.True(MtpSpeculativeCliFlags.Apply(new[] { "--mtp-spec", "--no-mtp-spec" }));
        Assert.False(SchedulerConfig.FromEnvironment().MtpSpeculativeEnabled);

        Assert.True(MtpSpeculativeCliFlags.Apply(new[] { "--no-mtp-spec", "--mtp-spec" }));
        Assert.True(SchedulerConfig.FromEnvironment().MtpSpeculativeEnabled);
    }
}
