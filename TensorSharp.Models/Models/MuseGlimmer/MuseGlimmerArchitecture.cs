// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Muse-Glimmer architecture plug-in.</summary>
    internal static class MuseGlimmerArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "muse-glimmer",
            DisplayName = "Muse-Glimmer",
            Aliases = new[] { "muse-glimmer", "muse_glimmer" },
            Factory = c => new MuseGlimmerModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath),
            ProjectorFileHints = new[] { "*mmproj*Muse*Glimmer*.gguf" },
        };
    }
}
