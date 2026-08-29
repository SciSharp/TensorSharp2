// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Qwen-Image / Qwen-Image-Edit architecture plug-in.</summary>
    internal static class QwenImageArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "qwen_image",
            DisplayName = "Qwen-Image",
            Aliases = new[] { "qwen_image", "qwen-image" },
            Factory = c => new QwenImageModel(c.GgufPath, c.Backend),
        };
    }
}
