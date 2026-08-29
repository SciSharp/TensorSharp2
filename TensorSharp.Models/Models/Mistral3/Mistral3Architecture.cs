// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Mistral 3 architecture plug-in.</summary>
    internal static class Mistral3Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "mistral3",
            DisplayName = "Mistral 3",
            Aliases = new[] { "mistral3" },
            Factory = c => new Mistral3Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            ProjectorFileHints = new[] { "mistral3-mmproj.gguf" },
        };
    }
}
