# 模型架构卡片

[English](README.md) | [中文](README_zh-cn.md)

本目录是 TensorSharp 支持的每一种模型架构的权威分卡参考。每张卡片都是一份独立的简报：从「我从未听说过这个模型」走到「我能解释它的前向计算图，并能在 TensorSharp 中复现它的推理路径」。如果你只需要顶层指引，请使用下方的总览表；否则请直接阅读对应的卡片。

## 每张卡片的内容

为了便于横向对比，所有卡片都采用相同的章节结构：

1. **来源与目标** —— 模型作者、GGUF 架构标识，以及它支持的能力（多模态、思维链、工具调用）。
2. **模型架构** —— 顶层模块图、层数与每层异构性。
3. **前向计算图** —— 单 token（decode）、多 token（prefill）或 diffusion 去噪步骤依次经过的算子序列，包含 residual 与各处 normalization。
4. **组件细节** —— 每一个子模块（attention、FFN/SSM、routing、normalization、RoPE 变体、视觉/音频编码器）的数学描述。
5. **参数与配置** —— GGUF 元数据 key、权重张量命名约定、dtype 要求。
6. **TensorSharp 实现** —— C# 源码定位、初始化顺序、缓存布局、模型如何接入 `ModelBase` / `Ops` / 原生 GGML kernel。
7. **Prefill 优化** —— 分块、融合 per-layer kernel、并行化、跨层缓存。
8. **Decode 优化** —— 融合单调用 kernel、预解析权重指针、批量 MoE、in-place kernel、缓存复用。
9. **内存与 KV cache 策略** —— 环形缓存 vs 线性缓存、mmap 权重、预分配 decode 缓冲。
10. **多模态管线** —— 图像 / 音频 / 视频如何被处理、编码并注入到语言模型。
11. **输出 / 聊天模板** —— 协议解析器、停止 token、思维链 / 工具调用格式。
12. **优化机会** —— 已知尚未实现但能进一步释放性能或能力的工作。

## 已验证的起步路径

已验证的原生 GGML 家族 / 路径层级是 Gemma 4 E4B Q8_0；推荐的公开文件来源是
[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)。
后端选择 `ggml_cuda`、`ggml_metal` 或 `ggml_vulkan`；这条路径会实际执行
融合原生内核。详见
[Gemma 4 卡片](gemma4_zh-cn.md#已验证的-gemma-4-e4b-原生-ggml-快速路径)。
匹配的 `mmproj` 对纯文本可选，对图像、视频或音频输入则是必需的。

如果希望沿着一条连贯路线，从张量基础一直学到完整的多模态推理引擎，请阅读
Zhongkai Fu 的 [《From Tensors to Tokens》书籍指南](../BOOK_zh-cn.md)，或
[在 Amazon 查看平装本](https://www.amazon.com/dp/B0H9P44QZZ)。

## 实现矩阵

| 架构 | 卡片 | 已验证下载（HF） | 模型类 | GGUF keys | 模态 | 思维链 | 工具调用 | 批处理 / 分页前向 | 主要加速路径 |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V4 Flash | [deepseek4_zh-cn.md](deepseek4_zh-cn.md) | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)（每个量化档一个子目录、均为多分片；`--model` 指向 `-00001-of-` 那一片）。DSpark 草稿器见 [MODEL_DOWNLOADS_zh-cn.md](../../MODEL_DOWNLOADS_zh-cn.md) | `DeepSeek4Model`（+ `DeepSeek4CudaExecutor`、`DeepSeek4CpuExecutor`） | `deepseek4` | 文本 | 是 | 是（DSML 标记） | 使用原生 per-sequence slot（`DeepSeek4Model.PerSeqCache.cs`）而非 `IBatchedPagedModel`——仍可通过同一引擎以连续批处理对外服务 | 三套整模型执行器（Direct CUDA、原生 ggml、纯 C#）、按层自动切分到所有可见 GPU、设备端压缩 KV 状态（SWA 环 + CSA/HCA + lightning indexer）、按形状签名的计算图缓存以重放已捕获的 CUDA 图、对 `[ring \| top-512]` K 的融合 decode index-gather，以及 DSpark 块级投机解码（decode 提速 1.3–1.4×） |
| Gemma 3 | [gemma3_zh-cn.md](gemma3_zh-cn.md) | [ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) | `Gemma3Model` | `gemma3` | 文本、图像 | 否 | 否 | 否（仅旧单序列路径） | SWA / 全局注意力交替、GeGLU FFN、QK-norm、V-norm |
| Gemma 4 | [gemma4_zh-cn.md](gemma4_zh-cn.md) | E4B Q8_0 是已验证的原生 GGML 家族 / 路径层级；[ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) 是推荐的公开文件来源 | `Gemma4Model` | `gemma4`（`gemma4-assistant` / `gemma4_assistant` 仅作为 MTP 草稿加载） | 文本、图像、视频、音频 | 是 | 是 | **默认启用**（可用 `TS_GEMMA4_BATCHED=0` 关闭） | 整模型融合 decode（一次 GGML 调度）、带内核内 PLE + 共享 KV 处理的融合整模型 prefill/verify、分块 prefill、SWA 环形缓存与 MoE 变体。批处理路径与旧路径 logits 在 FP 噪声内一致（`Gemma4BatchedForwardTests`）；batch=8 短 prompt 达 ~1.5×，4×800-token prompt 达 ~1.6×。 |
| DiffusionGemma | [diffusiongemma_zh-cn.md](diffusiongemma_zh-cn.md) | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | `DiffusionGemmaModel` + `DiffusionGemmaSampler` | `diffusion-gemma`、`diffusion_gemma` | 文本 | 否 | 否 | 独立的 Web UI `DiffusionBatchScheduler`；不是自回归 `IBatchedPagedModel` 路径 | `[prompt \| canvas]` 上的 EntropyBound 分块去噪、GPU prompt-KV 缓存、self-conditioning、融合 GGML 整模型 diffusion decode 与融合 lm-head tail |
| Qwen-Image-Edit | [qwenimage_zh-cn.md](qwenimage_zh-cn.md) | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)（DiT；VAE / 文本编码器配套文件见卡片） | `QwenImageModel`（+ `QwenImagePipeline`） | `qwen_image`、`qwen-image` | 图像编辑（图像+文本 → 图像） | 否 | 否 | 无——`Forward()` 抛异常；编辑通过 `EditImage()` 并串行执行 | 60 块 MMDiT 扩散（FlowMatch-Euler、true-CFG、参考潜变量拼接）、CUDA 图捕获的整 DiT 前向、默认 flash 注意力、CFG-batching、First-Block-Cache、融合 Qwen2.5-VL 视觉编码器、按 VRAM 钳制面积 |
| Qwen 3 | [qwen3_zh-cn.md](qwen3_zh-cn.md) | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | `Qwen3Model` | `qwen3` | 文本 | 是 | 是 | 参考实现（`Qwen3Model.BatchedForward.cs`）—— 当提供基础 Qwen3 GGUF 时由 `Qwen3BatchedForwardTests` 验证 | 整模型原生 decode，权重指针在加载时预解析 |
| Qwen 3.5 / 3.6 family | [qwen35_zh-cn.md](qwen35_zh-cn.md) | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)；NextN MTP：[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)（基础仓库的 Qwen3.6 GGUF 去掉了 NextN 块，会静默回退到标准 decode） | `Qwen35Model` | `qwen35`、`qwen35moe`、`qwen3next` | 文本、图像 | 是 | 是 | **默认启用**（`TS_QWEN35_BATCHED=0` 或 `--no-continuous-batching` 可关闭）。带每槽位的递归状态池，可选原生 GatedDeltaNet 内核（`TS_QWEN35_BATCHED_GDN_NATIVE=1`）。 | 全注意力 + GatedDeltaNet 递归混合、融合 attention 层 decode、融合 prefill attention、融合输出投影 + FFN、融合输出投影 + norm + router、批量 MoE（routed + shared + residual 一次完成）、融合视觉编码器 |
| GPT OSS | [gptoss_zh-cn.md](gptoss_zh-cn.md) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `GptOssModel` | `gptoss`、`gpt-oss` | 文本 | 是（始终启用） | 是 | **默认启用**（`TS_GPTOSS_BATCHED=0` 可关闭）。通过 `TSGgml_PagedAttentionForwardWithSinks` 处理每头 attention sinks（或 `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 使用 C# fallback）。在 `GptOssBatchedCorrectnessTests` 中与旧路径 100% 贪心一致。 | Stacked MoE prefill kernel（mul_mat_id + add_id + swiglu_oai）、attention sinks、MXFP4 专家权重 |
| Nemotron-H | [nemotron_zh-cn.md](nemotron_zh-cn.md) | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF)；Omni：[unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)（图像输入需另配 `mmproj-BF16.gguf`） | `NemotronModel` | `nemotron_h`、`nemotron_h_moe` | 文本、图像（Omni 版） | 是 | 是 | **默认启用**（`TS_NEMOTRON_BATCHED=0` 可关闭）。带每槽位 Mamba2 conv + SSM 状态池，可选原生批处理 Mamba2 步（`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`）。与旧路径 100% 贪心一致；Apple M4 Pro 上 batch=3 最高可达 3.95× tps。 | Mamba2 + 注意力 + MoE FFN 混合堆栈、批量 GPU MoE、RADIO/v2_vl 图像编码器、Parakeet 音频预处理器（音频推理需要 GGUF 发行版未附带的 Parakeet mmproj） |
| Mistral 3 | [mistral3_zh-cn.md](mistral3_zh-cn.md) | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | `Mistral3Model` | `mistral3` | 文本、图像 | 否 | 否 | **默认启用** —— `IBatchedPagedModel` 的参考实现。在 Ministral-3-14B 上完成端到端验证；原生分页注意力内核在长上下文下比旧的单序列路径快 ~21%。 | YaRN 校正 RoPE 与位置相关 Q 缩放、融合 QKV / gate_up、Pixtral 视觉编码器 |
| Muse-Glimmer | [muse-glimmer_zh-cn.md](muse-glimmer_zh-cn.md) | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)（`Muse-Glimmer-30B-*.gguf` + `mmproj-*.gguf`） | `MuseGlimmerModel` | `muse-glimmer` | 文本、图像 | 是 | 是 | 否（传统单序列） | 交错滑动窗口 + NoPE 全注意力层、注意力输出门控、每层 4 个 RMSNorm（post-norm eps 1e-8）、logit 缩放 + tanh 软上限、稀疏窗口 2D-RoPE ViT（2x2 像素重排）、可选 DFlash 块级草稿（`--draft-model`，无损）、**张量并行**（GGML CUDA/Vulkan 上 `--tp 2`——2 个 KV 头将并行度上限定为 2） |
| Wan 视频 | [wan_zh-cn.md](wan_zh-cn.md) | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)、[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)、[city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)（另需 UMT5-XXL 编码器与视频 VAE，见卡片） | `WanVideoModel`（+ `WanVideoPipeline`） | `wan` | 视频输出（文本 → 视频、图像 → 视频） | 否 | 否 | 无——`Forward()` 抛异常；生成走 `GenerateVideo()` 且进程内串行 | 每个去噪步一张常驻权重的 ggml 图（CUDA 图捕获、flash attention，TI2V 图生视频带 per-token 时间步调制）、因果 3D 视频 VAE 编/解码各一张图、A14B 的两个 14B 专家在时间步边界热切换、分阶段显存交接（TE → DiT → VAE）、720p 分块解码 |

## 后端说明

模型代码尽量保持后端无关。`ModelBase` 通过 `BackendType` 与执行计划选择张量存储，再把算子分发给已注册的后端实现：

| 后端类型 | 包 | 说明 |
|---|---|---|
| `Cpu` | `TensorSharp.Core` | 纯托管张量，附带 SIMD / 托管量化快路径（RMSNorm、RoPE、softmax、融合激活、GEMM、dequant）。 |
| `Cuda` | `TensorSharp.Backends.Cuda` | Direct CUDA Driver-API 分配器与存储、cuBLAS GEMM、热点算子的 PTX 内核（RMSNorm、softmax、RoPE/RoPEEx、SDPA、GQA prefill/decode、causal mask、gather/concat、融合激活）、受支持量化类型的原生 quant matmul / get_rows，未实现的算子回退到 CPU。 |
| `Mlx` | `TensorSharp.Backends.MLX` | Apple Silicon `mlx-c` 桥接，含量化 / 融合 / 编译内核、异步 worker 派发、MoE 专家 offload，以及 CPU 回退层。依赖 `libmlxc`。 |
| `GgmlCpu` / `GgmlMetal` / `GgmlCuda` | `TensorSharp.Backends.GGML` + `TensorSharp.GGML.Native` | 原生 ggml 桥接，包括量化计算图调度与平台后端；mmap 量化权重通过 host 指针缓冲零拷贝绑定。还包含驱动批处理 / 分页执行路径的分页注意力内核（`TSGgml_PagedAttentionForward`，含 GPT OSS sinks 变体）。 |

凡是卡片中提到融合 GGML kernel（例如 `Qwen35AttentionLayerDecode`、`Gemma4LayerPrefill`、`MoEExpertsSwiGLUResidual`），其源码都在 `TensorSharp.GGML.Native/ggml_ops_*.cpp`，并通过 `TensorSharp.Backends.GGML/GgmlBasicOps.cs` 暴露给托管侧。如果某个融合路径只在 GGML CPU / Metal / CUDA 上启用而在纯托管 CPU 或 direct CUDA 上没有启用，请到原生桥侧查看。

## 连续批处理 & 分页 KV 缓存

上表所列的自回归架构都会经过共享的 `InferenceEngine` + `ContinuousBatchScheduler` + `BatchExecutor` 栈，详情见 [`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)。实现了 `IBatchedPagedModel.ForwardBatch` 的模型会在每个调度步骤中执行一次批处理前向（使用基于 `slotMapping` 的 K/V 写入与共享分页缓冲，并通过原生分页内核做按序列注意力）；其余模型则在同一引擎内沿用按序列 KV 交换。DiffusionGemma 不支持自回归 `Forward()`，因此改用 `DiffusionGemmaSampler` 与服务端 `DiffusionBatchScheduler`。Qwen-Image-Edit 同样非自回归：`Forward()` 抛异常，编辑通过 `QwenImageModel.EditImage()` 在 FlowMatch-Euler 扩散循环上进行，且并发编辑被串行化（扩散网络非线程安全）。各模型的启用方式见上方实现矩阵以及项目根 README。

对于自带多 token 预测草稿头的架构——Qwen 3.6（内嵌 NextN 块）与 Gemma 4（独立 `gemma4-assistant` 草稿 GGUF）——单序列（无并发）请求还可以通过同一引擎运行无损的 MTP 投机解码（`--mtp-spec` —— 这是 `TensorSharp.Server` 的标志；CLI 没有 MTP 标志，只有 `TS_MTP_*` 环境变量）。共享的起草 / 验证 / 回滚核心是 `MtpSpeculativeExecution`；各架构具体机制见 Qwen 3.5/3.6（§12）与 Gemma 4（§12）卡片。

DeepSeek V4 把一个**块级**草稿器接入了同一套核心：它的 DSpark 支持模块作为独立 GGUF 通过 `--draft-model` 加载（CLI 与服务端都支持），每步提议一整块 token 而不是逐个。由于草稿器的权重必须计入层切分，它在加载阶段就传给 `ModelBase.Create()`，而不是事后附加。详见 [DeepSeek V4 卡片](deepseek4_zh-cn.md#dspark-投机解码)。

## 架构对比

| 特性 | DeepSeek V4 | Gemma 3 | Gemma 4 | DiffusionGemma | Qwen 3 | Qwen 3.5 / 3.6 family | GPT OSS | Nemotron-H | Mistral 3 |
|---|---|---|---|---|---|---|---|---|---|
| 层类型 | MoE（256 个路由专家，top-6 + 1 共享） | 密集 | 密集 / MoE | Gemma-4 派生 MoE encoder/decoder | 密集 | 混合（注意力 + 递归）± MoE | MoE | 混合（Mamba2 + 注意力 + FFN，密集或 MoE） | 密集 |
| 注意力 | 原始 SWA-128 + 压缩注意力 CSA 4:1 / HCA 128:1（CSA 层用 lightning indexer 选 top-512） | SWA + 全局 | SWA + 全局 | 区分 prompt/canvas 的区域感知注意力 | 全 GQA | 全 GQA + Sigmoid Gate | 全 + Sinks | 全 GQA（无 RoPE） | 全 GQA |
| FFN 激活 | SwiGLU（每层带 clamp） | GeGLU | GeGLU | Dense GeGLU + top-8 MoE | SwiGLU | SwiGLU | SiLUAlphaLimit（带 clamp 的 GLU） | ReLU² | SwiGLU |
| RoPE 类型 | 交错成对 + YaRN；raw 与 compress 两套 base，注意力后再做逆旋转 | NeoX（双 base） | NeoX + 比例 / 部分 | NeoX，local/global base | NeoX | NeoX / MRoPE | NeoX + YaRN | 无 | GPT-J + YaRN |
| QK-norm | 仅 Q（每头 RMS） | 是 | 是 | 是 | 是 | 是 | 否 | 否 | 否 |
| V-norm | 否 | 否 | 是（无权重） | 是（无权重） | 否 | 否 | 否 | 否 | 否 |
| 投影偏置 | 无（仅路由选择偏置） | 无 | 无 | 无 | 无 | 无 | 全部都有 | 无 | 无 |
| 每层缩放 | 否（改为每层 swiglu clamp 与压缩比） | 否 | 是 | encoder / decoder 标量 | 否 | 否 | 否 | 否 | 否 |
| Per-Layer Embedding (PLE) | 否 | 否 | 是 | 否 | 否 | 否 | 否 | 否 | 否 |
| KV 共享 | 是（所有 query 共用一个 512 维 K=V 头） | 否 | 是（尾部若干层） | 去噪多步间复用 prompt-KV | 否 | 否 | 否 | 否 | 否 |
| Attention sinks | 是 | 否 | 否 | 否 | 否 | 否 | 是 | 否 | 否 |
| 环形 KV cache | 是（原始 SWA-128 环） | 否 | 是（SWA 层） | 无自回归 KV | 否 | 否 | 否 | 否 | 否 |
| SSM / 递归层 | 否（用 4 路 hyper-connection 取代普通残差） | 否 | 否 | 否 | 否 | 是（GatedDeltaNet） | 否 | 是（Mamba2） | 否 |
| 共享专家 | 是 | 否 | 否 | 否 | 否 | 是（qwen35moe / qwen3next） | 否 | 是（可选） | 否 |
| Latent bottleneck FFN | 否（改为 LoRA 分解的 Q / 输出投影） | 否 | 否 | 否 | 否 | 否 | 否 | 是（可选） | 否 |
| 位置相关 Q 缩放 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 是（与 YaRN 配合） |
| 视觉 | 否 | 是 | 是 | 否 | 否 | 是 | 否 | 是（Omni） | 是（Pixtral） |
| 音频 | 否 | 否 | 是 | 否 | 否 | 否 | 否 | 否 —— Omni 仅图像（Parakeet log-mel 预处理已实现，但推理需要未随发行版提供的音频 mmproj） | 否 |
| 视频 | 否 | 否 | 是 | 否 | 否 | 否 | 否 | 否 | 否 |
| 思维链 | 是 | 否 | 是 | 否 | 是 | 是 | 是（始终启用） | 是 | 否 |
| 工具调用 | 是（DSML 标记） | 否 | 是 | 否 | 是 | 是 | 是 | 是 | 否 |
| MTP / NextN 投机解码 | DSpark 块级草稿器（独立 GGUF，`--draft-model`） | 否 | 是（独立 `gemma4-assistant` 草稿 GGUF） | 否 | 否 | Qwen 3.6 支持（内嵌 NextN 块） | 否 | 否 | 否 |
| 融合 QKV | n/a（LoRA 分解的 Q，单个共享 K=V 头） | 否 | 是 | 是 | 是 | 混合（attention 层拆开，递归层融合 5 路） | 是 | 是 | 是 |
| 融合单调用 decode | 是（整模型执行器，每个 ubatch 一张图，重放 CUDA 图） | 否 | 是（Gemma4ModelDecode） | 是（DiffusionModelDecode + lm-head tail） | 是（TransformerModelDecode，原生循环） | per-layer 融合（Qwen35AttentionLayerDecode、FusedOutProjFFN、FusedOutProjNormRouter） | per-layer | per-layer / 批量 MoE | 否 |
| 融合单调用 prefill | 是（同一整模型执行器，分块 ubatch） | 否 | 是（整模型 NativeGemma4ModelVerify + 逐层 Gemma4LayerPrefill 回退） | prompt-KV prefill cache | 否 | 是（FusedPrefillAttention、FusedOutProjFFN、MoE prefill） | 是（MoE prefill via mul_mat_id） | 否 | 否 |
| 批量 GPU MoE | 是（分组专家内核） | n/a | 全 MoE 变体已支持（融合整模型 MoE decode/verify）；混合 dense+MoE 待实现 | 融合单 canvas MoE；并发请求由 diffusion scheduler 批处理 | n/a | 是（routed + shared + residual 融合） | 是（stacked weight slabs） | 是 | n/a |
| 融合视觉编码器 | n/a | n/a | 标准 | n/a | n/a | 是（FusedVisionAttention + FusedVisionMLP） | n/a | 标准（RADIO ViT） | 标准（Pixtral） |
| 输出解析器 | `DeepSeek4OutputParser` | `PassthroughOutputParser` | `Gemma4OutputParser` | `PassthroughOutputParser` | `Qwen3OutputParser` | `Qwen35OutputParser` | `HarmonyOutputParser`（始终启用） | `Qwen3OutputParser` | `PassthroughOutputParser` |

## 新增模型架构

要在 TensorSharp 中加入新的模型架构：

1. 在 `TensorSharp.Models/Models/<Name>/<Name>Model.cs` 创建类并继承 `ModelBase`。
2. 构造函数中：通过 `_gguf.GetXxx()` 读取 GGUF 元数据，调用 `ParseBaseConfig()` 与 `ParseTokenizer()`，调用 `LoadWeights()`，融合权重，然后初始化缓存。
3. 自回归模型实现 `Forward(int[] tokens) → float[]`：embedding → 可选多模态注入 → transformer 层 → final norm → LM head → logits 拷贝。扩散模型需要明确记录替代 sampler 入口，并让不支持的自回归路径显式失败。
4. 实现 `ResetKVCache()` 与 `Dispose()`；如支持 KV 缓存复用则实现 `TruncateKVCache()`。
5. 在 `TensorSharp.Models/ModelBase.cs` 的 `ModelBase.Create()` switch 中注册。
6. 如果模型有非标准的输出格式，在 `TensorSharp.Runtime/OutputParser.cs` 中实现 `IOutputParser`，并在 `OutputParserFactory.Create()` 注册。
7. 如果模型使用了新的聊天模板，在 `TensorSharp.Runtime/ChatTemplate.cs` / `Jinja2Template.cs` 中加入支持。
8. 在 `docs/models/<name>.md`（与 `<name>_zh-cn.md` 如果你打算双语覆盖）下新增卡片，更新本 README 的实现矩阵，并从项目根 README 链入卡片。
9. 如果模型涉及新的模态、思维链或工具能力，更新 `TensorSharp.Server/testdata/` 的能力门控。
