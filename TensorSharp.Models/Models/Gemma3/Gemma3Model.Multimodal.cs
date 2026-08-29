// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Gemma 3's multimodal contract: SigLIP images from the mmproj GGUF.
using System.Collections.Generic;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class Gemma3Model : IVisionCapableModel, IMultimodalPromptExpander
    {
        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessGemma3History(this, history, inputTokens);
    }
}
