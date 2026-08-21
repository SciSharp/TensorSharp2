# 模型下载（GGUF）
[English](MODEL_DOWNLOADS.md) | [中文](MODEL_DOWNLOADS_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。另见[各模型架构卡片](docs/models/README_zh-cn.md)。


TensorSharp 使用 GGUF 格式模型文件。以下是各架构对应的已核对 Hugging Face 下载入口与伴随文件。请根据硬件条件选择合适的量化版本（Q4_K_M / UD-Q4_K_XL 适合低内存，Q8_0 适合更高质量等）。标注“可选”的条目是提速用的产物——步数蒸馏 checkpoint、蒸馏 LoRA、推测解码 draft 模型。不下载也能跑通，但它们往往就是“几分钟”和“几小时”的差别，动手前请先扫一眼。

| 架构 | 模型 | GGUF 下载 |
|---|---|---|
| Gemma 4 已验证原生规格 | gemma-4-E4B-it Q8_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)；推荐公开文件为 `gemma-4-E4B-it-Q8_0.gguf`，另有低内存 Q4_K_M；同仓库投影器为 `mmproj-gemma-4-E4B-it-Q8_0.gguf` |
| Gemma 4 | 12B / 26B-A4B QAT | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) / [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)；同仓库含 `mmproj-BF16.gguf`，以及匹配的 MTP draft（`mtp-gemma-4-12B-it.gguf` / `mtp-gemma-4-26B-A4B-it.gguf`，可选，仅用于推测解码） |
| Gemma 4 | 31B / 26B-A4B | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) / [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF)；同仓库含 mmproj |
| Gemma 4 | E4B / 26B-A4B MTP draft（可选，仅用于推测解码） | [AtomicChat E4B assistant](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) / [AtomicChat 26B assistant](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF)；仅与匹配尺寸的目标配对 |
| Gemma 3 | gemma-3-4b-it | 非 gated 的 [ggml-org/gemma-3-4b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)，投影器 `mmproj-model-f16.gguf`；官方 [Google QAT 仓库](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf)是 **gated** 仓库（需 HF 登录并接受 Google 的 Gemma 许可证），也是本文唯一一个 gated 仓库 |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF)，如 `Qwen3-4B-Q4_K_M.gguf` |
| Qwen 3.5 | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)，投影器 `mmproj-F16.gguf` |
| Qwen 3.5 | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF)，投影器 `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` |
| Qwen 3.6 | Qwen3.6-35B-A3B（保留 NextN） | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)，投影器 `mmproj-F16.gguf`。**注意不要下载基础仓库** [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)：它的文件名完全相同，但剥离了 NextN 块，`--mtp-spec` 会静默回落到普通解码 |
| GPT OSS | gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)，文件 `gpt-oss-20b-MXFP4.gguf`（注意 `MXFP4` 为大写）；纯文本，无伴随文件 |
| Nemotron-H | Nemotron-H-8B / 47B Reasoning | [8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) / [47B](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron 3 Nano Omni 30B-A3B | [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)，图像输入需 `mmproj-BF16.gguf`；仓库未附真实音频推理需要的 Parakeet mmproj |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF)，Pixtral 投影器 `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf` |
| Muse-Glimmer | Muse-Glimmer-30B（稠密，支持图像） | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)，如 `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` 或 `Muse-Glimmer-30B-Q8_0.gguf`；`general.architecture` 为 `muse-glimmer` / `muse_glimmer`。图像输入需同仓库的 `mmproj-Muse-Glimmer-30B-Q8_0.gguf`，且必须**显式**用 `--mmproj` 指定——这是唯一没有 mmproj 自动探测的系列。可选提速产物：同仓库的 DFlash 分块 draft `dflash-kquant.gguf`，用 `--draft-model` 加载即可无损推测解码——不要传任何采样参数，它只在纯贪心下生效 |
| DeepSeek V4 | DeepSeek-V4-Flash-0731（284B MoE） | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)；每种量化一个子目录（`UD-Q8_K_XL/`、`UD-IQ4_XS/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。仅文本 |
| GLM 5.x | GLM-5.2（744B-A40B MoE，内嵌 NextN MTP） | [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)；每种量化一个子目录（`UD-Q4_K_XL/`、`UD-IQ2_XXS/` 等），均为多分片，`--model` 指向 `-00001-of-` 分片。仅文本。这些 GGUF 已带有服务端 `--mtp-spec` 所需的 NextN 块——与 Qwen 3.6 不同，不存在需要挑选的独立 MTP 仓库 |
| DeepSeek V4 | DSpark 推测解码 draft（可选，仅提速） | 见下方 [DSpark draft 模型](#dspark-draft-模型)，用 `--draft-model` 加载，解码约 1.3-1.4 倍 |
| DiffusionGemma | diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)，如 `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` |
| Qwen-Image-Edit | MMDiT DiT（必需） | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)，如 `qwen-image-edit-2511-Q4_K_M.gguf` |
| Qwen-Image-Edit | VAE + Qwen2.5-VL（必需） | [QuantStack VAE](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) 中的 `VAE/Qwen_Image-VAE.safetensors` + [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) |
| Qwen-Image-Edit | 视觉 mmproj（可选） | [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) 中的 `mmproj-BF16.gguf`，用 `--qwen-image-mmproj` / `TS_QWEN_IMAGE_MMPROJ` 加载，可让编辑指令参考源图内容 |
| Qwen-Image-Edit | Lightning LoRA（可选，4/8 步） | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)，文件 `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`（0.850 GB）；用 `--qwen-image-lora` / `TS_QWEN_IMAGE_LORA` 加载，会自动按文件名里的步数把采样默认值切到该步数 + CFG 1.0（基础默认为 30 步、CFG 2.5） |
| Wan 视频生成 | **步数蒸馏 DiT（首选）** | **这是最大的提速手段——除非要复现参考样例，都应该用它。**蒸馏 checkpoint 生成同一段视频只跑 4 次去噪，而官方配方要跑 100 次：在 M5 Pro / `ggml_metal` 上以 1088×832×121 帧实测，端到端 **17 分 30 秒**，而基础 checkpoint 是 **3 小时 30 分**——同一个请求，其他参数一律不变。TI2V-5B：[hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)，文件 `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf`（5.40 GB），另有 Q6_K（4.22 GB）、Q5_K_M（3.82 GB）、Q4_K_M（3.44 GB），最小到 Q2_K（1.86 GB）。**注意文件名里是 `Wan2_2` 下划线**，照抄基础仓库的 `Wan2.2` 写法会 404。I2V-A14B：[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)，Lightning 已合并进两个专家；需同时下载 `high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf` **和** `low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf`（各 9.66 GB；Q8_0 15.42 GB，Q2_K 5.31 GB），放在同一个 `--local-dir` 下，`--model` 指向任意一个即可，另一个专家会自动找到。备选：[Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF](https://huggingface.co/Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF)（`FastWan2.2-TI2V-5B-q8_0.gguf`，5.41 GB）。**无需任何参数**：TensorSharp 读取 DiT 文件名，命中 `turbo` / `distill` / `lightning` / `lightx2v` / `fastwan` / `-dmd` 或显式的 `<N>steps`（1-16）即切换到该步数并关闭 guidance，加载时打印 `step-distilled checkpoint detected -> N steps, guidance off`；`--diffusion-steps` / `--cfg` 可覆盖。Turbo 与 A14B 蒸馏仓库都不含 VAE 和文本编码器，请从下面两行获取 |
| Wan 视频生成 | 基础 DiT（`--model` GGUF） | 完整官方配方（50 步 × 2 次 CFG = 100 次 DiT 前向）——需要对齐参考样例时才用，否则优先用上一行的蒸馏版本。Wan 2.2 文/图生视频：[QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)（`Wan2.2-TI2V-5B-Q8_0.gguf` 5.40 GB 或 `Wan2.2-TI2V-5B-Q4_K_M.gguf` 3.43 GB，仓库自带 `VAE/Wan2.2_VAE.safetensors`）、[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF) 或 [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF)（`HighNoise/` 与 `LowNoise/` 两个专家缺一不可，两个仓库都自带 `VAE/Wan2.1_VAE.safetensors`）；Wan 2.1 文生视频：[samuelchristlie/Wan2.1-T2V-1.3B-GGUF](https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF)（`Wan2.1-T2V-1.3B-Q8_0.gguf` / `-F16.gguf`）或 [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)（文件名为小写，如 `wan2.1-t2v-14b-Q8_0.gguf`）——这两个 2.1 仓库都不含 VAE 和编码器。`general.architecture` 为 `wan` / `wan2.1` / `wan2.2`。参见 [docs/models/wan.md](docs/models/wan.md) |
| Wan 视频生成 | UMT5-XXL 文本编码器（必需，所有 Wan checkpoint 都要） | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf)：`umt5-xxl-encoder-Q8_0.gguf`（6.04 GB），内存紧张可用 `umt5-xxl-encoder-Q5_K_M.gguf`（4.15 GB）/ `umt5-xxl-encoder-Q4_K_M.gguf`（3.66 GB）。负责把提示词编码成条件向量，去噪开始前即从显存释放。放在 DiT 旁或用 `--wan-te` / `TS_WAN_TE` 指定 |
| Wan 视频生成 | 视频 VAE（必需） | 把 latent 解码成画面——**用哪个由 DiT 自己决定**，不是由你选：TI2V-5B 需要 [`Wan2.2_VAE.safetensors`](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE)（TI2V-5B 仓库自带），Wan 2.1 与 A14B 需要 `Wan2.1_VAE.safetensors`——两个 QuantStack A14B 仓库里就有 `VAE/Wan2.1_VAE.safetensors`，也可单独下载 [`wan_2.1_vae.safetensors`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors)。上面的蒸馏仓库都不含 VAE，请从这里配一个对应的。放在 DiT 旁（`VAE/` 子目录亦可）或用 `--wan-vae` / `TS_WAN_VAE` 指定 |

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

> 这套 5 层 `fc` 融合结构**已经**在 Muse-Glimmer 上实现——见
> [DFlash 投机解码](docs/models/muse-glimmer_zh-cn.md#3-dflash-投机解码)。
> 下表这些 draft 没有接入，是因为它们的编码器需要目标模型暴露逐层输入残差，
> 目前只有 `MuseGlimmerModel` 做到了这一点。

| 主干 | 官方 checkpoint（safetensors） | 社区 GGUF |
|---|---|---|
| Qwen3-4B | [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) | — |
| Qwen3-8B | [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | [ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF](https://huggingface.co/ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF)（1.5 GB） |
| Qwen3-14B | [deepseek-ai/dspark_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7) | — |
| Gemma-4-12B | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | [ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF](https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF)（1.9 GB）、[williamliao/dspark_gemma4_12b-GGUF](https://huggingface.co/williamliao/dspark_gemma4_12b-GGUF) |
| Gemma-4-26B-A4B | — | [williamliao/dspark_gemma4_26b-a4b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_26b-a4b-it-GGUF) |
| Gemma-4-31B | — | [williamliao/dspark_gemma4_31b-it-GGUF](https://huggingface.co/williamliao/dspark_gemma4_31b-it-GGUF) |

Gemma 4 目前已有可用的推测解码路径：上表中的 `gemma4-assistant` MTP draft（
`--mtp-spec --mtp-draft-model`）；Qwen 3.6 与 GLM 5.2 则内置 NextN 块。它们与 DSpark 是不同的 draft。

### 按模型下载并运行

以下命令从仓库根目录运行；请先按平台安装完整的 [.NET 10 SDK](DEVELOPMENT_zh-cn.md#安装-net-10-sdk)，再执行 `dotnet build TensorSharp.slnx -c Release`。仅安装 Runtime 无法构建下方使用的二进制文件。`hf` 来自 Hugging Face CLI（`pip install -U huggingface_hub`），所有文件都会下载到 `./models`。通用提示：单次文本提示词通过 `--input` 文件传入（`--prompt` 用于 Qwen-Image-Edit 的编辑指令和 Wan 的视频提示词）；CLI 默认贪心采样，且不加 `--max-tokens` 时只生成 100 个 token；服务端固定监听 **http://localhost:5000**。按硬件把示例中的 `ggml_cuda` 换成 `ggml_metal`、`ggml_vulkan` 或 `ggml_cpu`（见 [选择后端](README_zh-cn.md#选择后端)）。

```bash
echo "列出三条关于月球的事实。" > prompt.txt
```

**DeepSeek V4 Flash**（284B MoE，纯文本，支持 DSpark 推测解码）：

```bash
# 约 160 GB 权重：需要多张 GPU（自动按层切分），draft 另需约 7 GB
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "UD-Q8_K_XL/*" --local-dir models
hf download bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF DSpark-drafter-Q2K-Q8-0731.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model models/DSpark-drafter-Q2K-Q8-0731.gguf \
    --input prompt.txt --max-tokens 200 --temperature 0
```

去掉 `--draft-model` 即为普通解码。CLI 上的推测解码要求纯贪心采样（`--temperature 0`）；
`--spec-draft-conf-min` 控制每个块草拟到多深。

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
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gpt-oss-20b-MXFP4.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gpt-oss-20b-MXFP4.gguf --backend ggml_cuda
```

**Nemotron-H**（文本、思维链、工具）：

```bash
hf download bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --input prompt.txt --max-tokens 300 --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf --backend ggml_cuda
```

图像输入请改用 Omni 发行版：从 [unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF) 下载 `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` 与 `mmproj-BF16.gguf`；当前发行版没有真实音频推理所需的 Parakeet audio mmproj。

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

（Web UI 会实时流式展示 DiffusionGemma 的去噪过程；兼容 API 只返回最终文本。）

**Qwen-Image-Edit**（DiT + VAE + 文本编码器；Lightning LoRA 可选）：

```bash
hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q4_K_M.gguf --local-dir models
hf download QuantStack/Qwen-Image-Edit-GGUF VAE/Qwen_Image-VAE.safetensors --local-dir models
hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --local-dir models
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir models
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --image input.png --prompt "把天空改成壮丽的日落。" --output edited.png --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/qwen-image-edit-2511-Q4_K_M.gguf --qwen-image-vae models/VAE/Qwen_Image-VAE.safetensors --qwen-image-vl models/Qwen2.5-VL-7B-Instruct-UD-IQ2_XXS.gguf --qwen-image-lora models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --backend ggml_cuda
```

（在 Web UI 里上传图片并输入编辑指令即可。Lightning LoRA 的下载与 `--qwen-image-lora` 参数是可选的——加上后去噪降到 4 步、CFG 1.0。）

**Wan 视频生成**（提示词 + 可选首帧图片 → H.264 MP4；需要 DiT + 视频 VAE + UMT5-XXL 文本编码器）：

```bash
# 步数蒸馏的 Turbo DiT：只跑 4 次去噪而不是 100 次，由文件名自动识别。
# 注意 Turbo 文件名里的 Wan2_2 下划线；VAE 和文本编码器仍需从基础仓库获取。
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_cuda \
    --wan-vae models/VAE/Wan2.2_VAE.safetensors --wan-te models/umt5-xxl-encoder-Q8_0.gguf \
    --prompt "a cute fluffy orange cat walking through a sunny garden with flowers" \
    --output cat.mp4 --width 832 --height 480 --video-frames 81
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
    --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_cuda \
    --wan-vae models/VAE/Wan2.2_VAE.safetensors --wan-te models/umt5-xxl-encoder-Q8_0.gguf \
    --video-frames 121 --fps 24
```

加载时控制台会打印 `step-distilled checkpoint detected -> 4 steps, guidance off`——看到这行就说明走的是快路径。
只把 `--model` 换成基础的 `Wan2.2-TI2V-5B-Q8_0.gguf`，就会按官方 50 步 + CFG 配方运行：同一个
1088×832×121 帧的请求实测为 3 小时 30 分，而这里是 17 分 30 秒（M5 Pro，`ggml_metal`）。
加 `--image first_frame.png` 即为图生视频，Web UI 里上传图片也一样（该图作为首帧）；服务端的
`--video-frames` / `--fps` 只是默认值，单个请求可以覆盖。Wan 是唯一不支持 `--backend mlx` 的系列，
请使用 `ggml_cuda`、`ggml_metal`、`ggml_vulkan`、`ggml_cpu`、`cuda` 或 `cpu`。

三个文件放在同一个目录下时（`VAE/` 子目录也算），`--wan-vae` / `--wan-te` 可以省略，会自动解析。
双专家的 A14B 模型需要**同时**下载两个专家到同一个 `--local-dir`，`--model` 指向其中任意一个：

```bash
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download QuantStack/Wan2.2-I2V-A14B-GGUF VAE/Wan2.1_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model models/high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf \
    --backend ggml_cuda --wan-vae models/VAE/Wan2.1_VAE.safetensors \
    --wan-te models/umt5-xxl-encoder-Q8_0.gguf \
    --prompt "the ship sails into the storm, waves crashing" --image ship.jpg --output ship.mp4
```

