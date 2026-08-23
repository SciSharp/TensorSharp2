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

namespace InferenceWeb.Tests;

/// <summary>
/// Disposable helper that snapshots and restores environment variables
/// touched during a test. Without this, the env vars set by one test could
/// leak into another test that runs in the same process.
/// </summary>
internal sealed class EnvScope : IDisposable
{
    private readonly Dictionary<string, string?> _originals = new();

    public void Set(string name, string value)
    {
        if (!_originals.ContainsKey(name))
            _originals[name] = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, value);
    }

    /// <summary>
    /// Clear every speculative-decoding variable, in BOTH spellings. A flag
    /// applied through <c>SpeculativeCliFlags</c> is published twice - under
    /// <c>TS_SPEC_*</c> for managed readers and <c>TS_MTP_*</c> for the glm-dsa
    /// native loader - so a test that clears only one spelling still reads the
    /// other one's leftovers.
    /// </summary>
    public void ClearSpeculationVars()
    {
        foreach (string name in new[]
                 {
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.Enabled,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.Type,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.Draft,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.PMin,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.DraftModel,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.LegacyEnabled,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.LegacyDraft,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.LegacyPMin,
                     TensorSharp.Runtime.Speculative.SpeculationEnvVars.LegacyDraftModel,
                 })
        {
            Set(name, null);
        }
    }

    public void Dispose()
    {
        foreach (var kv in _originals)
            Environment.SetEnvironmentVariable(kv.Key, kv.Value);
        _originals.Clear();
    }
}
