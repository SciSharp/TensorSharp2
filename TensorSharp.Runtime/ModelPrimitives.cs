// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
namespace TensorSharp.Runtime
{
    public enum BackendType
    {
        Cpu,
        Cuda,
        Mlx,
        GgmlCpu,
        GgmlMetal,
        GgmlCuda,
        GgmlVulkan,
    }

    public class ModelConfig
    {
        public string Architecture { get; set; } = string.Empty;
        public int HiddenSize { get; set; }
        public int NumHeads { get; set; }
        public int NumKVHeads { get; set; }
        public int KeyLength { get; set; }
        public int ValueLength { get; set; }
        public float Eps { get; set; }
        public float RopeBase { get; set; }
        public float RopeScale { get; set; } = 1f;
        public int NumLayers { get; set; }
        public int VocabSize { get; set; }
        public int IntermediateSize { get; set; }
        public string ChatTemplate { get; set; } = string.Empty;

        public int NumExperts { get; set; }
        public int NumExpertsUsed { get; set; }
        public int SlidingWindow { get; set; }
        public bool UsesCircularKvCache { get; set; }
        public int OriginalContextLength { get; set; }

        /// <summary>
        /// Sampling parameters the model author shipped inside the GGUF under the
        /// <c>general.sampling.*</c> keys, or null when the file carries none.
        /// Only the fields actually present are set; the rest stay at their
        /// <see cref="SamplingConfig"/> defaults. llama.cpp applies these over its
        /// own defaults for every field the operator did not pin
        /// (common/common.cpp), and Qwen3.x GGUFs ship them — for example
        /// Qwen3.8-27B carries top_k=20, top_p=0.95, temp=1.0.
        /// </summary>
        public RecommendedSampling RecommendedSampling { get; set; }

        public int HeadDim => KeyLength > 0 ? KeyLength : (ValueLength > 0 ? ValueLength : HiddenSize / NumHeads);
    }
}

