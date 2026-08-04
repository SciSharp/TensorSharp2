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

    public void Dispose()
    {
        foreach (var kv in _originals)
            Environment.SetEnvironmentVariable(kv.Key, kv.Value);
        _originals.Clear();
    }
}
