# 模型下载（GGUF）
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。另见[各模型架构卡片](docs/models/README_zh-cn.md)。


TensorSharp 使用 GGUF 格式模型文件。以下是各架构对应的已核对 Hugging Face 下载入口与伴随文件。请根据硬件条件选择合适的量化版本（Q4_K_M / UD-Q4_K_XL 适合低内存，Q8_0 适合更高质量等）。

| 架构 | 模型 | GGUF 下载 |
|---|---|---|
| Gemma 4 已验证原生规格 | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)；推荐公开文件为 `gemma-4-E4B-it-Q8_0.gguf`，另有低内存 Q4_K_M；同仓库投影器为 `mmproj-gemma-4-E4B-it-Q8_0.gguf` |
| Gemma 4 | 12B / 26B-A4B QAT | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) / [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)；同仓库含 `mmproj-BF16.gguf` 与匹配的 MTP draft |
| Gemma 4 | 31B / 26B-A4B | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) / [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF)；同仓库含 mmproj |
| Gemma 4 | E4B / 26B-A4B MTP draft | [AtomicChat E4B assistant](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) / [AtomicChat 26B assistant](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF)；仅与匹配尺寸的目标配对 |
| Gemma 3 | gemma-3-4b-it | 非 gated 的 [ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)，投影器 `mmproj-model-f16.gguf`；官方 [Google QAT 仓库](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf)需要登录并接受许可证 |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF)，如 `Qwen3-4B-Q4_K_M.gguf` |
| Qwen 3.5 | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)，投影器 `mmproj-F16.gguf` |
| Qwen 3.5 | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF)，投影器 `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` |
| Qwen 3.6 | Qwen3.6-35B-A3B（保留 NextN） | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)，投影器 `mmproj-F16.gguf`；基础仓库会剥离 NextN 块 |
| GPT OSS | gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)，文件 `gpt-oss-20b-mxfp4.gguf` |
| Nemotron-H | Nemotron-H-8B / 47B Reasoning | [8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) / [47B](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)，图像输入需 `mmproj-BF16.gguf`；仓库未附真实音频推理需要的 Parakeet mmproj |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF)，Pixtral 投影器 `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` |
| DeepSeek V4 | DeepSeek-V4-Flash-0731（284B MoE） | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)；每种量化一个子目录（`UD-Q8_K_XL/`、`UD-IQ4_XS/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。仅文本 |
| DeepSeek V4 | DSpark 推测解码 draft | 见下方 [DSpark draft 模型](#dspark-draft-模型)，用 `--draft-model` 加载，解码约 1.3-1.4 倍 |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)，如 `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` |
| Qwen-Image-Edit | MMDiT DiT（必需） | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)，如 `qwen-image-edit-2511-Q4_K_M.gguf` |
| Qwen-Image-Edit | VAE + Qwen2.5-VL（必需） | [QuantStack VAE](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) 中的 `VAE/Qwen_Image-VAE.safetensors` + [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) |
| Qwen-Image-Edit | Lightning LoRA（可选） | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)，如 4-step `.safetensors`；通过 `--qwen-image-lora` 加载 |

### DSpark draft 模型

[DSpark](docs/models/deepseek4.md#dspark-speculative-decoding) 是 DeepSeek 的分块推测解码
draft 模型。TensorSharp 目前在 **DeepSeek V4** 上支持它，两个 GPU 引擎均可用
（`--backend cuda` 与 `--backend ggml_cuda`）：draft 是独立的 GGUF，用 `--draft-model`
加载；由主干逐块验证，贪心输出不变。

以下三个任选其一，均可直接加载（加载器兼容各发布者的张量/元数据命名）。draft 读取主干的
隐藏状态，因此与模型**同一 checkpoint 版本**的 draft 接受率更高：

| Draft | 大小 | 适配 | 说明 |
|---|---|---|---|
| [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF) | 7.0 GB | **0731** 版本请用 `DSpark-drafter-Q2K-Q8-0731.gguf`（同仓库另有非 0731 版本） | Q2_K 专家 + Q8_0 稠密层；实测接受率 71% |
| [sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF) | 5.6 GB | 0731 之前的 `DeepSeek-V4-Flash` | 体积最小；即使搭配 0731 主干也最快（接受率 69%）——draft 的带宽开销比精度更重要 |
| [alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF) | 10.9 GB | **0731** 版本 | MXFP4 专家（对 checkpoint FP4 的无损重排）；精度最高、显存占用最大 |

也可以从任何带该模块的 DeepSeek V4 checkpoint 自行转换（只需下载其三个 `mtp.*` 分片，约
11 GB）：见 [Getting a drafter](docs/models/deepseek4.md#getting-a-drafter) 与
`eng/dsv4-dspark-to-gguf.py`。

**其他架构的 DSpark draft 暂不支持。** DeepSeek 也发布了 Qwen 3 与 Gemma 4 的 DSpark
draft，社区亦有 GGUF 转换，但它们是另一种 draft 结构：5 层 Transformer + 对五个目标层做
`fc` 融合（`general.architecture` 为 `dspark` 或 `dflash`，block_size 7），而非 DeepSeek V4
的三个超连接块（`mtp.*`）。TensorSharp 会明确报错而不会错误加载。这里列出以便了解上游现状：

| 主干 | 官方 checkpoint（safetensors） | 社区 GGUF |
|---|---|---|
| Qwen3-4B | [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) | — |
| Qwen3-8B | [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | [ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF](https://huggingface.co/ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF)（1.5 GB） |
| Qwen3-14B | [deepseek-ai/dspark_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7) | — |
| Gemma-4-12B | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | [ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF](https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF)（1.9 GB）、[williamliao/dspark_gemma4_12b-GGUF](https://huggingface.co/williamliao/dspark_gemma4_12b-GGUF) |
| Gemma-4-26B-A4B | — | [williamliao/dspark_gemma4_26b-a4b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_26b-a4b-it-GGUF) |
| Gemma-4-31B | — | [williamliao/dspark_gemma4_31b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_31b-it-GGUF) |

Gemma 4 目前已有可用的推测解码路径：上表中的 `gemma4-assistant` MTP draft（
`--mtp-spec --mtp-draft-model`）；Qwen 3.6 则内置 NextN 块。它们与 DSpark 是不同的 draft。

### 按模型下载并运行

以下命令从仓库根目录运行；请先按平台安装完整的 [.NET 10 SDK](DEVELOPMENT_zh-cn.md#安装-net-10-sdk)，再执行 `dotnet build TensorSharp.slnx -c Release`。仅安装 Runtime 无法构建下方使用的二进制文件。`hf` 来自 Hugging Face CLI（`pip install -U huggingface_hub`）。单次文本提示词必须通过 `--input` 文件传入，`--prompt` 仅用于 Qwen-Image-Edit。按硬件把示例的 `ggml_cuda` 换成 `ggml_metal`、`ggml_vulkan` 或 `ggml_cpu`。

```bash
echo "列出三条关于月球的事实。" > prompt.txt
```

**Gemma 4**（文本 + 图像/视频/音频、思维链、工具、可选 MTP）：

```bash
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download AtomicChat/gemma-4-E4B-it-assistant-GGUF gemma-4-E4B-it-assistant.Q8_0.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --mtp-spec --mtp-draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

第三个下载与两项 MTP 参数可省略。

**Gemma 3**（文本 + 图像；下方非 gated 仓库）：

```bash
hf download ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir models
hf download ggml-org/gemma-3-4b-it-GGUF mmproj-model-f16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-3-4b-it-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-3-4b-it-Q4_K_M.gguf --mmproj models/mmproj-model-f16.gguf --backend ggml_cuda
```

**Qwen 3**（文本、思维链、工具）：

```bash
hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3-4B-Q4_K_M.gguf --input prompt.txt --think --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3-4B-Q4_K_M.gguf --backend ggml_cuda
```

**Qwen 3.5 / 3.6**（图像、思维链、工具；3.6 可用 NextN）：

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --backend ggml_cuda

# 3.6 必须从保留 NextN 块的 -MTP- 仓库下载
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --mtp-spec
```

**GPT OSS**（文本、始终思考、工具）：

```bash
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-mxfp4.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-mxfp4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gpt-oss-20b-mxfp4.gguf --backend ggml_cuda
```

**Nemotron-H**（文本、思维链、工具）：

```bash
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --backend ggml_cuda
```

Omni 图像版本需另下同仓库 `mmproj-BF16.gguf`；当前发行版没有真实音频推理所需的 Parakeet audio mmproj。

**Mistral 3**（文本 + 图像）：

```bash
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --local-dir models
hf download bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --mmproj models/mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf --backend ggml_cuda
```

**DiffusionGemma**（块文本扩散）：

```bash
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --max-tokens 256 --diffusion-steps 48 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

**Qwen-Image-Edit**（DiT + VAE + 文本编码器；Lightning LoRA 可选）：

```bash
hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q4_K_M.gguf --local-dir models
hf download QuantStack/Qwen-Image-Edit-GGUF VAE/Qwen_Image-VAE.safetensors --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --local-dir models
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --image input.png --prompt "把天空改成壮丽的日落。" --output edited.png --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
```

