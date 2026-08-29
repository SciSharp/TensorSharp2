// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Qwen 3 / Qwen 2 / Qwen 2.5-VL architecture plug-in.</summary>
    internal static class Qwen3Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            // qwen2vl is Qwen2/Qwen2.5-VL. Its language model is Qwen3's block with a
            // QKV bias and no QK norm, both of which Qwen3Model detects from the weights.
            // Text-only chat works without the vision tower: M-RoPE degenerates to standard
            // RoPE when the t/h/w position components are equal, which they are for text.
            Id = "qwen3",
            DisplayName = "Qwen 3 / Qwen 2 / Qwen 2.5-VL",
            Aliases = new[] { "qwen3", "qwen2", "qwen2vl", "qwen2_vl" },
            Factory = c => new Qwen3Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
        };
    }
}
