// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Diffusion Gemma (diffusion LM) architecture plug-in.</summary>
    internal static class DiffusionGemmaArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "diffusion-gemma",
            DisplayName = "Diffusion Gemma (diffusion LM)",
            Aliases = new[] { "diffusion-gemma", "diffusion_gemma" },
            Factory = c => new DiffusionGemmaModel(c.GgufPath, c.Backend),
        };
    }
}
