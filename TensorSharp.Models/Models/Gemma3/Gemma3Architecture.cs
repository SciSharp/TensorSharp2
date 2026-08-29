// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Gemma 3 architecture plug-in.</summary>
    internal static class Gemma3Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "gemma3",
            DisplayName = "Gemma 3",
            Aliases = new[] { "gemma3" },
            Factory = c => new Gemma3Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            ProjectorFileHints = new[] { "mmproj-gemma3-4b-f16.gguf" },
        };
    }
}
