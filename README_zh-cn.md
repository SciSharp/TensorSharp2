# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

**面向 GGUF 模型的原生 .NET LLM 推理引擎** —— 覆盖自回归 LLM *与* DiffusionGemma 风格的文本扩散模型，以及 Qwen-Image-Edit 图像编辑、Wan 2.1/2.2 文本与图像生视频。提供控制台应用、浏览器聊天界面，以及兼容 Ollama/OpenAI 的 HTTP API。一个纯 .NET 引擎，在相同 GGUF 文件与相同 GPU 上与手工优化的 C++ `llama.cpp` 互有胜负。

## 《From Tensors to Tokens》—— TensorSharp 实战书籍

<p align="center">
  <a href="https://www.amazon.com/dp/B0H9P44QZZ">
    <img src="website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="220">
  </a>
</p>

Zhongkai Fu 所著的 **[From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B](https://www.amazon.com/dp/B0H9P44QZZ)** 将本仓库串成一条端到端的学习路径。全书以 Gemma 4 E4B 为示例，连接张量基础、模型执行、多模态输入，以及一个可运行 LLM 推理引擎的应用接口。

**[查看书籍介绍与仓库伴读路线](docs/BOOK_zh-cn.md)** · **[在 Amazon 购买平装本](https://www.amazon.com/dp/B0H9P44QZZ)**

## 亮点功能

- **⚡ 与 llama.cpp 互有胜负——用纯 .NET 做到。** 在相同 GGUF 文件、相同 GPU 上，TensorSharp 在关键负载上追平乃至超越 `llama.cpp`：Gemma 4 E4B 与 2-bit 量化的 Qwen 3.6 35B-A3B MoE 在 CUDA 上 prefill 快 **1.28×**、首 token 早 **1.27×**（多轮最高 **1.49×**）；Gemma 4 12B 在 Vulkan 上 decode 快 **1.21×**（长上下文最高 **1.32×**）。→ [性能数据](#性能数据)
- **🚀 连续批处理 & 分页 KV 缓存。** vLLM 风格的分页 KV 池，支持基于内容哈希的前缀共享与迭代级调度器，服务端默认启用。→ [深入文档](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)
- **🧬 DeepSeek V4 Flash（284B MoE），三套整模型执行器。** 这套压缩稀疏注意力、1M 上下文的架构可运行在 Direct CUDA 引擎（`--backend cuda`）、原生 ggml 执行器（`--backend ggml_cuda` / `ggml_vulkan`），以及 **100% 纯 C# 的 CPU 执行器**（`--backend cpu`，零原生依赖）上。权重会自动按层切分到所有可见 GPU，因此远大于单卡显存的模型依然跑得起来；服务端以原生 per-sequence slot + 连续批处理托管它。→ [DeepSeek V4 卡片](docs/models/deepseek4.md)
- **🧠 GLM-5.2（744B-A40B MoE），支持张量并行与 CPU MoE offload。** Multi-head Latent Attention，外加一个 DeepSeek 稀疏注意力的 "lightning indexer"，由它挑出每个 query 可以看的那 2048 个已缓存 token。`--tp N` 让每一层都跑在每张 GPU 上（head 按列/行并行，每个专家按行切开），`--cpu-moe` 则把路由专家——占 checkpoint 的 92%——留在系统内存里。默认的按层切分与 `--cpu-moe` 在同后端下与 llama.cpp 逐 token 一致；`--tp` 是把各 rank 的局部和相加，在 2 bit MoE 上这点最后一位的差别会传到 top-8 路由，几乎并列的 token 可能不同。3× RTX PRO 6000 上的正面对比：**pp2048 918.9，llama.cpp 为 763.1 tok/s**；tg64 43.7 对 42.2。自报的 1M 上下文（约 93 GiB 的 KV）是上限而非承诺——权重落盘之后，加载器会按实际空闲的显存来定上下文并打印它的选择（按层切分 342,272 token，`--n-cpu-moe 30` 为 646,400）；设 `MAX_CONTEXT` 则把某个长度变成硬性要求。→ [GLM 卡片](docs/models/glm_zh-cn.md)
- **🔮 投机解码——MTP / NextN 与 DSpark。** 多 token 预测草稿头加速单序列 decode：Qwen 3.6（NextN 块内嵌于主干——需使用保留该块的 GGUF，例如 [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)；基础仓库的同名文件已剥离该块）、**GLM 5.2**（NextN 块已随官方 checkpoint 一同分发，无需额外下载；2× RTX PRO 6000 + `--n-cpu-moe 20` 实测 decode **约 1.3×**，草稿接受率 94%）与 Gemma 4（独立 `gemma4-assistant` 草稿 GGUF，`--mtp-draft-model`）；DeepSeek V4 则新增 **DSpark** 块级起草（`--draft-model`），每步提议一整块 token，decode 提速 **1.3–1.4×**（多轮对话最高 2.0×）。三者都是草稿提议、主干一次批量前向验证，输出与标准 decode 一致。→ [投机解码](FEATURES_zh-cn.md#mtp--nextn-投机解码)
- **🔗 张量并行与分布式集群。** 用 `--tp N` 把一个模型切分到多张 GPU 上——Direct `cuda` 后端**以及** GGML CUDA / Vulkan 后端均支持——再用点对点 TCP 集群（`--tp-node-id` / `--tp-peers`）扩展到多台机器。采用 Megatron-LM 列/行并行范式与分层 AllReduce；GGML 上提供 MoE 专家并行与按 rank 的 GatedDeltaNet 融合内核。融合式按 rank 执行让 Gemma 4 E4B 上 `--tp 2` 的 decode 达到单卡的 **1.39×**、Muse-Glimmer 30B 达到 **1.57×**（其 prefill 同时提升 **1.34×**——是这里唯一在两个阶段都超过单卡的模型），也让单卡装不下的模型（Qwen 3.5-35B-A3B；24 GB 卡上 28.2 GB 的 Muse-Glimmer 30B Q8_0）得以运行。可选 Redis 支撑的 KV 缓存与 Responses API 存储。→ [张量并行](USAGE_zh-cn.md#张量并行与分布式推理)
- **🎨 Qwen-Image-Edit 图像编辑。** 提示词 + 输入图像 → 编辑后的图像，驱动 60 块 MMDiT，配以 Qwen-Image VAE 与 Qwen2.5-VL-7B 文本编码器。CUDA 图捕获的整 DiT、FlowMatch-Euler true-CFG 去噪、Web UI 实时预览，以及 [Lightning 蒸馏 LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) 快速路径（`--qwen-image-lora`，以运行期旁路的形式挂在原封不动的量化权重旁），把默认的 30 步 × CFG——即 60 次 DiT 前向——降到 **4** 次。热态 4 步编辑比 `stable-diffusion.cpp` 快 **1.19×**。→ [Qwen-Image-Edit 卡片](docs/models/qwenimage_zh-cn.md)
- **🎬 Wan 2.1 / 2.2 视频生成（文本→视频、图像→视频）。** 提示词 → H.264 MP4；Wan 2.2（TI2V-5B、I2V-A14B）上上传的图像作为首帧，提示词驱动运动、镜头与场景变化。每个去噪步一张常驻权重的 ggml 图（CUDA 图捕获、flash attention），因果 3D 视频 VAE 编/解码各一张图，A14B 的两个 14B 专家在时间步边界热切换，分阶段显存交接——TI2V-5B 在 16 GB GPU 上 8 分钟内生成 81 帧 480p 图生视频，Wan 2.1 端到端比 `stable-diffusion.cpp` 快 **6.0×**。**步数蒸馏检查点会按 DiT 文件名自动识别**（`Turbo` / `distill` / `Lightning` / `lightx2v` / `FastWan` / `…-4steps-…`），并切换到该步数、关闭引导——4 次 DiT 前向而不是官方配方的 100 次，同一个 1088×832×121 帧的图生视频请求因此从 **3 小时 30 分降到 17 分 30 秒**（M5 Pro）。这是本仓库中最大的单项提速手段，不需要任何参数，只是换一个 `--model` 文件。数值已对照 diffusers 验证（DiT 余弦 > 0.995，VAE 编码器余弦 > 0.999，解码 59.9 dB / >35 dB PSNR）。支持 CLI（`--image`）、`/v1/videos/generations` 与可上传图片的 Web UI 聊天。→ [Wan 卡片](docs/models/wan_zh-cn.md)
- **🌫️ DiffusionGemma 文本扩散。** 基于 Gemma-4 派生 MoE backbone 的分块 EntropyBound 去噪，提供 CLI 参数与 Web UI 实时去噪预览。→ [DiffusionGemma 卡片](docs/models/diffusiongemma_zh-cn.md)
- **🖼️ 多模态。** 图像 / 视频 / 音频（Gemma 4）；图像输入（Gemma 3、Qwen 3.5-family、Mistral 3、Nemotron-H Omni、Muse-Glimmer）；CLI 与 Web UI 支持 PDF。→ [多模态](FEATURES_zh-cn.md#多模态支持)
- **🛠️ 工具调用与思维链。** Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H、Muse-Glimmer（ATEM 标记）、DeepSeek V4（DSML 标记）均支持多轮工具调用与结构化思维链。→ [功能特性](FEATURES_zh-cn.md)
- **🔌 兼容 Ollama 与 OpenAI 的 API**，外加浏览器聊天 UI——现有工具可直接接入。→ [HTTP API](USAGE_zh-cn.md#http-api)
- **📄 配置文件 + 自动下载。** 把 CLI/Server 参数写进可复用的 JSON 文件，支持 `${变量}` 与首次运行自动下载模型的 `{ "path", "urls" }` 条目。→ [config/README.md](config/README.md)
- **🧮 原生量化计算。** Q4_K_M / Q8_0 / MXFP4 / IQ2_XXS / IQ2_S / IQ3_S / IQ3_XXS / IQ4_XS 等直接参与 matmul，无需反量化为 FP32。可运行于 GGML Metal / CUDA / Vulkan、Direct CUDA/cuBLAS、MLX（Apple Silicon）与纯 C# CPU 路径，均带 CPU 回退。MLX 上的 IQ 解码内核把码本与 scale 的读取摊销到整个子块（而非每权重重读一次），使混合 IQ 量化的 30B 在 M5 Pro 上从 3.6 提升到 **14.3 tok/s**（达融合 ggml-metal 图的 67%，输出逐字节一致）。→ [后端](USAGE_zh-cn.md#计算后端)

## 快速开始

TensorSharp 面向 .NET 10。全新机器需要安装完整的 **.NET 10 SDK**；只安装 .NET Runtime 无法构建 TensorSharp：

| 平台 | 安装 SDK |
|---|---|
| **Windows** | 在 PowerShell 中运行 `winget install Microsoft.DotNet.SDK.10`，或参阅 Microsoft 的 [Windows 安装说明](https://learn.microsoft.com/zh-cn/dotnet/core/install/windows)。 |
| **macOS** | 使用 [.NET 10 SDK 安装程序](https://dotnet.microsoft.com/zh-cn/download/dotnet/10.0)：Apple 芯片选择 **Arm64**，Intel Mac 选择 **x64**。另见 Microsoft 的 [macOS 安装说明](https://learn.microsoft.com/zh-cn/dotnet/core/install/macos)。 |
| **Linux** | 按照 Microsoft 的 [Linux 发行版指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux)为当前发行版配置正确的软件源，并安装其 .NET 10 SDK 包（通常名为 `dotnet-sdk-10.0`）。 |

安装后打开新终端，确认列表中包含 `10.0.x` SDK：

```bash
dotnet --list-sdks
```

更多细节见 [.NET 跨平台安装概览](https://learn.microsoft.com/zh-cn/dotnet/core/install/)或[开发 → 前置要求](DEVELOPMENT_zh-cn.md#前置要求)。

然后即可在已验证的原生 GGML 快速路径（Gemma 4 E4B）上约 30 秒跑起来。其他前置包括 `git`、`curl`，以及所选 GPU 后端的工具链（见 [开发 → 前置要求](DEVELOPMENT_zh-cn.md#前置要求)）。推荐的公开文件是 [`gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf)（7.48 GiB）；纯文本推理无需投影器。

**Windows + NVIDIA（PowerShell）**

```powershell
git clone https://github.com/zhongkaifu/TensorSharp.git; Set-Location TensorSharp
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models\gemma-4-E4B-it-Q8_0.gguf
'用一句话回答：TensorSharp 是什么？' | Set-Content prompt.txt
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models\gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 128 --backend ggml_cuda
```

**macOS（Apple Silicon）** —— 去掉 CUDA 环境变量，使用 `--backend ggml_metal`。
**Linux + NVIDIA** —— 在 `dotnet run` 前加 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`，使用 `--backend ggml_cuda`。
**AMD / Intel / NVIDIA Vulkan** —— 设置 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON`，使用 `--backend ggml_vulkan`。

**Linux（Ubuntu）+ 多张 NVIDIA GPU —— 张量并行**

张量并行把一个模型切分到 N 张 GPU 上，可运行在 Direct `cuda` 后端以及 GGML CUDA /
Vulkan 后端（`--backend ggml_cuda`、`ggml_vulkan`）。请先安装 CUDA 工具包，然后：

```bash
# 在 RunPod 的 Ubuntu 24.04 镜像上，需要先让动态链接器找到 CUDA 兼容库：
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH
# 较旧的 Ubuntu 版本需要从 backports PPA 安装 .NET 10 SDK：
add-apt-repository ppa:dotnet/backports

apt update && apt install dotnet-sdk-10.0
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
mkdir models
wget "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -O models/gemma-4-E4B-it-Q8_0.gguf
bash TensorSharp.GGML.Native/build-linux.sh
dotnet build -c Release

# 单进程内使用 2 张 GPU
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend cuda --interactive --max-tokens 20000 --tp 2

# 同样的用法也适用于 GGML CUDA 后端（可加 TENSORSHARP_TP_DEVICES=0,2 指定 GPU）
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_cuda --interactive --max-tokens 20000 --tp 2
```

只需再加上节点 ID 与共享的 peer 列表，同一个模型就能跨机器扩展 —— 2 节点 × 2 GPU 即全局 TP 度为 4：

```bash
# 节点 0
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# 节点 1（peer 列表相同，节点 ID 不同）
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
```

`TensorSharp.Server` 支持同样的 `--tp`、`--tp-node-id`、`--tp-peers` 参数（也可用
`TENSORSHARP_TP_*` 环境变量）；在多节点集群中，服务端必须是节点 `0`（对外提供 HTTP
的 driver），其余节点各运行一个 `TensorSharp.Cli` worker。完整参考：**[张量并行与分布式推理](USAGE_zh-cn.md#张量并行与分布式推理)**。

将同一模型作为服务托管（浏览器 UI 在 <http://localhost:5000>，另有 Ollama/OpenAI API）：

```bash
dotnet run --project TensorSharp.Server -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> 服务端默认绑定 `0.0.0.0:5000`（可用 `--port` / `--host` 或 `PORT` / `HOST` 环境变量修改；macOS 上 5000 端口已被 AirPlay 接收器占用），无内置鉴权或 TLS——请置于防火墙之后，或使用带鉴权的 HTTPS 反向代理。图像/视频/音频需追加伴随文件 [`mmproj-gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf)，用 `--mmproj` 指定。

两个可执行程序在不带参数或使用 `--help` 启动时，都会打印完整的参数参考——逐项列出说明、默认值、取值范围与示例：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --help
dotnet run --project TensorSharp.Server -c Release -- --help
```

完整命令参考：**[CLI](USAGE_zh-cn.md#控制台应用)** · **[Server](USAGE_zh-cn.md#web-应用)** · 更多可下载模型：**[模型下载](MODEL_DOWNLOADS_zh-cn.md)** · 想用配置文件？**[config/](config/README.md)**。

## 选择后端

每个后端对尚未实现的算子都会回退到 CPU，因此所有后端的输出都正确。

| 你的硬件 | 推荐后端 | 标志 | 说明 |
|---|---|---|---|
| **Apple Silicon（Mac）** | GGML Metal | `--backend ggml_metal` | macOS 默认。`--backend mlx` 是另一条 Apple Silicon GPU 路径。 |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | 测试最充分的 NVIDIA 路径。`--backend cuda` 是用于实验的 Direct PTX/cuBLAS 后端。 |
| **Windows / Linux + AMD / Intel / NVIDIA GPU** | GGML Vulkan | `--backend ggml_vulkan` | 与厂商无关的 GPU 路径（ggml-vulkan）。机器有 Vulkan 运行时即自动构建；用 `--no-vulkan` 退出。 |
| **无 GPU / 可移植 / 调试** | 纯 C# CPU | `--backend cpu` | 无原生依赖。需要更快的 CPU 推理可用 `--backend ggml_cpu`（原生算子）。 |

每个后端的完整说明见 [使用方法 → 计算后端](USAGE_zh-cn.md#计算后端)。

## 已验证模型

以下架构均已实现，并由测试 / 基准矩阵覆盖。请选择适配你硬件的量化（低内存用 Q4_K_M、更高质量用 Q8_0）。更多尺寸与投影器文件见 [模型下载](MODEL_DOWNLOADS_zh-cn.md)。

| 家族 | 示例模型（GGUF） | 图像 / 视频 / 音频 | 思维链 | 工具 | 卡片 |
|---|---|---|---|---|---|
| DeepSeek V4 Flash | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)（284B MoE，分片 GGUF） | — / — / — | ✅ | ✅ | [deepseek4](docs/models/deepseek4_zh-cn.md) |
| GLM 5.x | [GLM-5.2](https://huggingface.co/unsloth/GLM-5.2-GGUF)（744B-A40B MoE，分片 GGUF） | — / — / — | ✅ | ✅ | [glm](docs/models/glm_zh-cn.md) |
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)（另有 31B、26B-A4B MoE） | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4](docs/models/gemma4_zh-cn.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)（另有 35B-A3B MoE） | ✅ / — / — | ✅ | ✅ | [qwen35](docs/models/qwen35_zh-cn.md) |
| Qwen 3 | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | — / — / — | ✅ | ✅ | [qwen3](docs/models/qwen3_zh-cn.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)（MoE） | — / — / — | ✅ | ✅ | [gptoss](docs/models/gptoss_zh-cn.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF)（另有 47B、Omni） | ✅（Omni） / — / — | ✅ | ✅ | [nemotron](docs/models/nemotron_zh-cn.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3](docs/models/mistral3_zh-cn.md) |
| Gemma 3 | [gemma-3-4b-it](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) | ✅ / — / — | — | — | [gemma3](docs/models/gemma3_zh-cn.md) |
| DiffusionGemma | [diffusiongemma-26B-A4B-it](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | — / — / — | — | — | [diffusiongemma](docs/models/diffusiongemma_zh-cn.md) |
| Muse-Glimmer | [Muse-Glimmer-30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)（+ mmproj） | ✅ / — / — | ✅ | ✅ | [muse-glimmer](docs/models/muse-glimmer_zh-cn.md) |
| Qwen-Image-Edit | [Qwen-Image-Edit-2511](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)（MMDiT + VAE + Qwen2.5-VL）· 快速路径：[Lightning 4 步 LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | 🖼️ 图像→图像 | — | — | [qwenimage](docs/models/qwenimage_zh-cn.md) |
| Wan 2.1 / 2.2 视频 | [Wan2.2-TI2V-5B](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)（另有 [T2V-A14B](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF)、[I2V-A14B](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)、[Wan2.1-T2V-14B](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)）+ UMT5-XXL + 视频 VAE · 快速路径：[TI2V-5B-Turbo](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)（4 步，DiT 前向次数减少 25×） | 🎬 文本→视频、图像→视频 | — | — | [wan](docs/models/wan_zh-cn.md) |

## 让它跑得更快

有几个家族存在一条**快速路径**——换一个可下载的权重文件，或加一个参数——就能把一次运行的开销降低一个数量级。在做其他调优之前，先看这张表。

| 家族 | 快速路径的权重或参数 | 实测效果 |
|---|---|---|
| **Wan 2.1 / 2.2 视频** | **步数蒸馏的 DiT GGUF**——TI2V-5B 用 [hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)（`Wan2_2-TI2V-5B-Turbo-Q8_0.gguf`），A14B 用 [jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)。无需任何参数——按文件名自动识别。 | 100 次 DiT 前向 → **4** 次，且关闭引导。同一个 1088×832×121 帧的图生视频请求：**3 小时 30 分 → 17 分 30 秒**（M5 Pro，`ggml_metal`）。 |
| Wan，仅基础权重 | `--cfg-cache-stride 2` / `3` | 50 步下 1.30× / 1.43×（近似方法；蒸馏权重本身已无 guidance，用它没有意义）。 |
| Wan，任意权重 | 在训练分辨率上生成后再下采样——用 736×544 代替 1088×832 | 121 帧、Turbo 权重：**6 分 19 秒**，而非 17 分 30 秒。但低于约 0.3 MP 反而会掉质量。 |
| **Qwen-Image-Edit** | `--qwen-image-lora` 加载 [Lightning 4 步 LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)（`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`） | 采样默认值从 30 步 / CFG 2.5（60 次 DiT 前向）切换为 **4** 步 / CFG 1.0。热态 4 步编辑比 `stable-diffusion.cpp` 快 **1.19×**。 |
| Qwen-Image-Edit | `TS_QWEN_DIT_CACHE_MODE=easycache`（默认关闭——质量优先） | 可跳过 40–55% 的去噪步；但会让编辑结果的细节（如人脸）明显变软，因此需显式开启。 |
| **DeepSeek V4 Flash** | `--draft-model` 加载 [DSpark 草稿 GGUF](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF)（服务端再加 `--mtp-spec`）；仅 `cuda` / `ggml_cuda` | 4×A40 上 decode 从 26.4 提升到 **37.1 tok/s**（1.41×），接受率 69%；多轮对话最高 **2.0×**。输出不变——主干会逐块验证。 |
| **Muse-Glimmer** | `--draft-model` 加载 DFlash 草稿模型（`dflash-kquant.gguf`，位于 [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)）；**不要**传任何采样参数 | 在其开发所用的 CUDA 主机上 decode 提升 1.3–5×——单张 RTX PRO 6000、60 token 提示、贪心解码下由 35.0 提升到 **50.9 tok/s**。Apple Silicon 上目前普通 decode 仍更快。 |
| **Qwen 3.6** | 保留 MTP 块的 GGUF——[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)，而非基础仓库——再加服务端的 `--mtp-spec` | 启用单序列上的 NextN 投机解码。基础仓库文件名相同但已剥离该块，会静默回退到普通 decode。 |
| **Gemma 4** | `--mtp-draft-model` 加载配套的 [`gemma4-assistant` 草稿 GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF)，并加 `--mtp-spec`（服务端） | 在 GGML 各后端与 Direct `cuda` 后端上启用投机解码。草稿与目标的 hidden size 必须一致，否则启动即失败。 |
| 显存装不下的 MoE | `--n-cpu-moe N` / `--cpu-moe` | 16 GB 笔记本显卡上 gpt-oss-20b 显存从 16.2 GB 降到 2.9 GB，把 WDDM 换页造成的 0.3 tok/s 变成 `--n-cpu-moe 12` 下的 **25.4 tok/s**。 |
| 多 GPU | `--tp N` | Gemma 4 E4B decode 达单卡的 **1.39×**，Muse-Glimmer 30B decode **1.57×** / prefill **1.34×**——并且能跑单卡装不下的模型。 |
| 所有家族 | 选对后端：NVIDIA 用 `ggml_cuda`，Apple Silicon 用 `ggml_metal`，无 GPU 用 `ggml_cpu`（而非 `cpu`） | Gemma 4 26B-A4B 在 `ggml_cuda` 上 decode 78.7 tok/s，Direct `cuda` 后端仅 35.3；Apple Silicon 上 Muse-Glimmer 30B 的 prefill 在 `ggml_metal` 上为 413.6 tok/s，MLX 上为 29.0。 |

每一行背后的完整数据与逐家族细节：[Wan](docs/models/wan_zh-cn.md) · [Qwen-Image-Edit](docs/models/qwenimage_zh-cn.md) · [DeepSeek V4](docs/models/deepseek4_zh-cn.md) · [Muse-Glimmer](docs/models/muse-glimmer_zh-cn.md) · [功能特性](FEATURES_zh-cn.md)。

## 支持的模型架构

| 架构 | GGUF 架构标识 | 示例模型 | 多模态 | 思维链 | 工具调用 | MTP 投机 | 卡片 |
|---|---|---|---|---|---|---|---|
| DeepSeek V4 Flash | `deepseek4` | DeepSeek-V4-Flash（284B MoE，256 专家，压缩稀疏注意力，1M 上下文） | 仅文本 | 支持 | 支持（DSML） | 支持（DSpark 块级草稿，独立 GGUF） | [deepseek4](docs/models/deepseek4_zh-cn.md) |
| GLM 5.x | `glm-dsa` | GLM-5.2（744B-A40B MoE，256 专家，MLA + DeepSeek 稀疏注意力，1M 上下文） | 仅文本 | 支持 | 支持（XML 工具调用） | 支持（内嵌 NextN 块） | [glm](docs/models/glm_zh-cn.md) |
| Gemma 4 | `gemma4` | gemma-4-E4B、gemma-4-31B、gemma-4-26B-A4B（MoE） | 图像、视频、音频 | 支持 | 支持 | 支持（独立草稿 GGUF） | [gemma4](docs/models/gemma4_zh-cn.md) |
| Gemma 3 | `gemma3` | gemma-3-4b | 图像 | 不支持 | 不支持 | — | [gemma3](docs/models/gemma3_zh-cn.md) |
| Qwen 3 | `qwen3`、`qwen2`、`qwen2vl`、`qwen2_vl` | Qwen3-4B（Qwen2 / Qwen2.5-VL 的 GGUF 也能加载，按纯文本对话运行） | 仅文本 | 支持 | 支持 | — | [qwen3](docs/models/qwen3_zh-cn.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B（混合 Attn+递归）、Qwen3.5/3.6-35B-A3B（MoE） | 图像 | 支持 | 支持 | Qwen 3.6 支持（内嵌 NextN） | [qwen35](docs/models/qwen35_zh-cn.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b（MoE） | 仅文本 | 支持（始终） | 支持 | — | [gptoss](docs/models/gptoss_zh-cn.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B/47B（混合 SSM-Transformer，MoE）、Nemotron 3 Nano Omni | 图像（Omni） | 支持 | 支持 | — | [nemotron](docs/models/nemotron_zh-cn.md) |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | 图像 | 不支持 | 不支持 | — | [mistral3](docs/models/mistral3_zh-cn.md) |
| Muse-Glimmer | `muse-glimmer`、`muse_glimmer` | Muse-Glimmer-30B（交错滑动窗口 + NoPE 全注意力层，注意力输出门控） | 图像 | 支持 | 支持（ATEM） | 支持（DFlash 块级草稿，独立 GGUF） | [muse-glimmer](docs/models/muse-glimmer_zh-cn.md) |
| DiffusionGemma | `diffusion-gemma`、`diffusion_gemma` | diffusion-gemma 文本扩散 GGUF | 仅文本 | 不支持 | 不支持 | — | [diffusiongemma](docs/models/diffusiongemma_zh-cn.md) |
| Qwen-Image-Edit | `qwen_image`、`qwen-image` | qwen-image-edit MMDiT GGUF（+ VAE 与 Qwen2.5-VL） | 图像编辑（图像+文本 → 图像） | 不支持 | 不支持 | — | [qwenimage](docs/models/qwenimage_zh-cn.md) |
| Wan 视频 | `wan`、`wan2.1`、`wan2.2` | Wan 2.1 T2V 1.3B/14B、Wan 2.2 TI2V-5B、Wan 2.2 A14B T2V/I2V（双专家） | 视频输出（文本→视频、图像→视频） | 不支持 | 不支持 | — | [wan](docs/models/wan_zh-cn.md) |

各架构的端到端文档（前向图、组件、参数、prefill/decode 优化）见[按模型架构卡片](docs/models/README_zh-cn.md)。

## 性能数据

### 对比 llama.cpp 的同台评测（引擎对比）

纯 .NET 引擎与手工优化的 C++ `llama.cpp` 正面较量：**相同的 GGUF 文件、相同的 NVIDIA RTX 3080 Laptop GPU（16 GB）、统一的 OpenAI `/v1/chat/completions` 接口**，**两个引擎均分别在 GGML CUDA 与 Vulkan 构建上测量**。下表为 **在相同后端上，TensorSharp 相对 llama.cpp 的几何平均加速比**（单流、贪心采样、关闭 MTP）；**> 1.0× 表示 TensorSharp 更快 / 延迟更低**。完整表格见 [`docs/engine_comparison_report.md`](docs/engine_comparison_report.md)。

| 模型 | 后端 | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it（Q8_0，dense 多模态） | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it（Q8_0，dense 多模态） | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | Vulkan | 1.02× | 0.85× | 0.84× |

TensorSharp 在 CUDA 的 prefill / 首 token 延迟上明显领先（多轮 prefill **每个模型**都获胜，最高 **1.49×**），CUDA decode 保持持平或更快，Vulkan 上 dense 12B 的 decode 明显胜出（长上下文最高 **1.32×**）——即便在 2-bit IQ2_XXS 量化下亦然。剩余低于 1.0× 的项仍是正在优化的目标。该框架还提供工具调用、结构化输出、图像编辑（对比 `stable-diffusion.cpp`）、MTP 开/关与并发场景，可通过 [`benchmarks/engine_comparison`](benchmarks/engine_comparison) 在你自己的硬件上运行。完整报告见 [此处](docs/engine_comparison_report.md)。

放不进这台 16 GB 机器的模型，会在各自的卡片里给出同样方式测得的正面对比（两个引擎、同一份 GGUF、同一台机器、背靠背）：[GLM-5.2 744B-A40B，3× RTX PRO 6000](docs/models/glm_zh-cn.md#性能) —— 从约 1k prompt token 起 TensorSharp 的 prefill 领先（pp2048 **1.20×**、pp4096 **1.21×**），decode 领先 1.04×，短 prefill 上则是 llama.cpp 快几个百分点。

## 文档

初次使用？上面几节足以让你跑起来。其余均为详细参考：

| 文档 | 内容 |
|---|---|
| [书籍指南：《From Tensors to Tokens》](docs/BOOK_zh-cn.md) | 从张量基础走向 Gemma 4 E4B 多模态推理引擎的连贯路线，含出版信息与配套仓库阅读指引 |
| [模型下载](MODEL_DOWNLOADS_zh-cn.md) | 各模型 `huggingface-cli` 下载 + 运行速查（量化档位、投影器、伴随文件） |
| [使用方法](USAGE_zh-cn.md) | 完整 CLI 参考（选项、交互式 REPL、JSONL 批处理）、服务端托管、日志、HTTP API 示例、后端与环境变量矩阵 |
| [功能特性](FEATURES_zh-cn.md) | 连续批处理、MTP 投机解码、工具调用、思维链、多模态、MoE、KV 编解码等深入说明 |
| [配置文件](config/README.md) | 把参数写进可复用的 JSON 文件，支持 `${变量}` 与模型自动下载 |
| [开发](DEVELOPMENT_zh-cn.md) | 前置要求、构建原生 GGML/MLX 库、仓库结构、包分层、内部架构与测试工具 |
| [按模型架构卡片](docs/models/README_zh-cn.md) | 各架构端到端文档（前向图、组件、参数、prefill/decode 优化） |
| [分页注意力 & 连续批处理](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md) | vLLM 风格的分页 KV 缓存、前缀共享与迭代级调度器 |
| [环境变量功能矩阵](docs/env_var_feature_matrix_zh-cn.md) | 哪些高影响运行时开关影响哪些模型、后端与提示类型 |
| [引擎对比报告](docs/engine_comparison_report.md) | TensorSharp 对比 llama.cpp / stable-diffusion.cpp 的完整逐场景表格 |
| [测试 / 基准矩阵运行器](TensorSharp.TestMatrix/README_zh-cn.md) | 扫描 model × backend × feature × env-var 组合并生成回归报告 |
| [服务端 API 示例](TensorSharp.Server/API_EXAMPLES_zh-cn.md) | 完整的 curl 与 Python 示例 |

## 当前状态

| 范围 | 状态 |
|---|---|
| 模型家族 | DeepSeek V4 Flash（`deepseek4`）、GLM 5.x（`glm-dsa`）、Gemma 3/4、DiffusionGemma、Qwen 3、Qwen 3.5/3.6-family（`qwen35`、`qwen35moe`、`qwen3next`）、GPT OSS、Nemotron-H（含 Nemotron 3 Nano Omni）、Mistral 3、Muse-Glimmer（`muse-glimmer`、`muse_glimmer`）。图像编辑通过 Qwen-Image-Edit（`qwen_image`、`qwen-image` MMDiT）；视频生成通过 Wan 2.1 / 2.2（`wan`、`wan2.1`、`wan2.2`）。 |
| 推理宿主 | CLI、交互式 REPL、ASP.NET Core Web UI、Ollama 风格 API、OpenAI Chat Completions 风格 API。 |
| 后端 | 纯 C# CPU、Direct CUDA/cuBLAS（`cuda`）、MLX Metal（`mlx`）、GGML CPU、GGML Metal、GGML CUDA、GGML Vulkan。DeepSeek V4 另有三套专属的整模型执行器——Direct CUDA、原生 ggml 与纯 C# CPU——都会把权重按层切分到所有可见 GPU（`--tp N` / `TS_DSV4_NGPU` 限定卡数）。Wan 是唯一对后端有限制的家族：它可运行于各 GGML 后端以及 Direct `cuda` / 纯 C# `cpu` 后端，但不支持 MLX。 |
| 多模态 | Gemma 4 图像/视频/音频；Gemma 3、Qwen 3.5-family、Mistral 3、Nemotron-H Omni、Muse-Glimmer 图像输入；PDF（CLI `--pdf` + Web UI）。媒体*输出*：Qwen-Image-Edit（图像）与 Wan 2.1 / 2.2（H.264 MP4 视频，文本→视频与图像→视频）。 |
| 连续批处理 | vLLM 风格分页 KV 缓存、基于内容哈希的前缀共享、迭代级调度器（默认启用，`--no-continuous-batching` 关闭）。DeepSeek V4 与 GLM 5.x 在同一引擎上通过各自原生的 per-sequence slot 提供服务——压缩后的 MLA 每 token 只有一行缓存，没有可分页的布局——GLM 还提供可选的批处理融合解码（`TS_BATCHED_FUSED_DECODE=1`，4 路并发下总吞吐 1.81 倍）。 |
| 投机解码 | Qwen 3.6（内嵌）与 Gemma 4（独立草稿 GGUF）的 MTP / NextN 草稿头；DeepSeek V4 的 DSpark 块级起草（仅 `cuda` / `ggml_cuda`）与 Muse-Glimmer 的 DFlash 块级起草，两者都通过 `--draft-model` 加载独立的草稿 GGUF。验证以贪心方式对齐主干，因此输出与普通贪心 decode 一致。默认关闭；服务端用 `--mtp-spec` 启用，CLI 直接传 `--draft-model` 即可。 |
| 张量并行 | Direct `cuda` 后端与 GGML CUDA / Vulkan 后端上的 Megatron-LM 列/行并行 TP（`--tp N` / `TENSORSHARP_TP_DEGREE`，CLI 与服务端均支持）；通过点对点 TCP 的多节点分布式 TP（`--tp-node-id` / `--tp-peers`），采用分层 AllReduce，CUDA P2P 不可用时自动回退到主机中转。覆盖全部自回归架构；GGML 上 Gemma 4 与 Qwen 3.5/3.6 使用 MoE 专家并行与融合的按 rank decode/prefill 计算图。可选 Redis 支撑的 KV 缓存与 Responses API 存储。 |
| 服务端模型范围 | 通过 `--model` 显式托管单个 GGUF；可通过 `--mmproj` 显式指定投影器；不扫描目录。 |
| 可观测性 | 结构化每轮日志、队列状态，以及 Web UI / Ollama / OpenAI 中的 KV 缓存复用指标。 |

## 作者

Zhongkai Fu

## 许可证

详见 [LICENSE](LICENSE)。
