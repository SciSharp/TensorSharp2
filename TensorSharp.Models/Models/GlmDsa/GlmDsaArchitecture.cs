// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>GLM-5.x (DeepSeek Sparse Attention) and GLM-5.3-Flash architecture plug-in.</summary>
    internal static class GlmDsaArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            // glm-dsa: MLA + lightning indexer + sigmoid MoE.
            // glm5next (GLM-5.3-Flash): hybrid KDA linear attention + nope-only MLA with a
            // pooled DSA indexer, Sinkhorn hyper-connections, 288-expert MoE. Same class.
            Id = "glm-dsa",
            DisplayName = "GLM-5.x (DeepSeek Sparse Attention) and GLM-5.3-Flash",
            Aliases = new[] { "glm-dsa", "glm_dsa", "glm5next" },
            Factory = c => new GlmDsaModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            ProjectorFileHints = new[] { "*mmproj*.gguf" },
        };
    }
}
