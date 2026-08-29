// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>DeepSeek V4 (Flash) architecture plug-in.</summary>
    internal static class DeepSeek4Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            // Deliberately TensorParallel as far as the shared gate is concerned:
            // DeepSeek V4 drives several GPUs through its OWN executor, sized by
            // TS_DSV4_NGPU, and the gate must not interfere with that.
            Id = "deepseek4",
            DisplayName = "DeepSeek V4 (Flash)",
            Aliases = new[] { "deepseek4" },
            Factory = c => new DeepSeek4Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath),
        };
    }
}
