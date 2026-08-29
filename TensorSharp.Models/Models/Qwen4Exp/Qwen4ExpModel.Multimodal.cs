// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.8-Flash-Next reuses the Qwen 3.5 vision tower and its M-RoPE span layout.
using System.Collections.Generic;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel : IVisionCapableModel, IMRoPEPositionSink, IMultimodalPromptExpander
    {
        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessQwenVLHistory(VisionEncoder, history, inputTokens);
    }
}
