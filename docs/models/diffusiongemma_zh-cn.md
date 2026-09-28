# DiffusionGemma

[← 返回模型索引](README_zh-cn.md) | [English](diffusiongemma.md)

## 状态快照

| 字段 | 状态 |
|---|---|
| GGUF 架构标识 | `diffusion-gemma`、`diffusion_gemma` |
| 模型类 | [`DiffusionGemmaModel`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaModel.cs) |
| 采样器 | [`DiffusionGemmaSampler`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaSampler.cs) |
| 模态 | 原生文本 + **图像**；[Jev](jev_zh-cn.md) 另支持文档提取、抽样视频帧，以及已配置 ASR 配套服务的语音转录 |
| 思维链 / 工具调用 | 不提示思考：提示词始终以关闭思考的方式渲染。模型自行写出的思维块会被解析剥离，仅在 `"think": true` 时作为推理返回；若 canvas 只有思维块，这段文本会作为答案返回（见 §6）。tools/tool_choice 以 HTTP 400 拒绝 |
| 生成方式 | 分块文本扩散，不是自回归 token decode |
| CLI 支持 | `TensorSharp.Cli` 检测到 `DiffusionGemmaModel` 后进入 diffusion 运行模式 |
| 服务端支持 | Web UI chat stream 带实时去噪预览；Ollama/OpenAI 兼容端点使用 append-oriented 响应形状，只返回最终文本（没有去噪预览） |
| 连续批处理 | 独立的 [`DiffusionBatchScheduler`](../../TensorSharp.Chat/DiffusionBatchScheduler.cs)，在 block 边界接纳请求 |

## 下载

已验证的 GGUF 下载指引：

| 模型 | HF 仓库 | 推荐文件 | 备注 |
|---|---|---|---|
| diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`（16.807 GB）；另有 `Q5_K_M`、`Q6_K`、`Q8_0`、`BF16` | GGUF `general.architecture` = `diffusion-gemma`。官方上游权重：[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |

`Q4_K_M` 是 unsloth 仓库中最小的量化版本。显存更紧张时，
`config/diffusiongemma-26b-a4b-q3.json` 固定使用 DevQuasar 的 `Q3_K_M`（约 13.3 GB；
unsloth 没有发布 Q3_K_M）。它的元数据与 unsloth 文件一致，只多了 TensorSharp 不读取的
`diffusion.eb_*` 采样提示。

**图像输入还需要视觉塔，而任何 GGUF 里都没有它。** 该检查点已发布的 GGUF 全部
是纯文本的 —— 转换过程丢弃了视觉塔，官方也从未发布过 mmproj。上游权重里确实有
这座塔：它的全部 356 个张量都在 11 分片 BF16 检查点的某一个 2.8 GB 分片中，
TensorSharp 直接加载该分片（无需转换）：

| 文件 | HF 仓库 | 大小 |
|---|---|---|
| `model-00011-of-00011.safetensors` | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 2.84 GB |

```bash
hf download google/diffusiongemma-26B-A4B-it model-00011-of-00011.safetensors --local-dir models
```

也可以让配置在首次使用时自动获取 —— `config/diffusiongemma-26b-a4b-q4.json`、
`config/diffusiongemma-26b-a4b-q3.json` 和 `config/jev-diffusiongemma-q4.json` 都在
`mmproj` 下声明了它并带 SHA-256，只下载一次，之后复用。若要只跑纯文本且不下载该分片，
在命令行上追加 `--mmproj none`：命令行上的 `--mmproj` 会在解析之前丢弃配置中的条目，因此
该分片永远不会被下载，而 `none` 表示不加载投影器——服务端与 CLI 都是如此（纯文本请求正常，
图像请求被拒绝）。

如果更想要一个普通的 mmproj GGUF，
[`eng/diffusiongemma-mmproj.py`](../../eng/diffusiongemma-mmproj.py) 只依赖 numpy（不需要 torch
或 gguf 包）就能把该分片转换成单文件的 `gemma4v` 投影器，其二维矩阵乘权重为 F16（`--dtype f32`
则保留 F32）：

```bash
python3 eng/diffusiongemma-mmproj.py --src models/model-00011-of-00011.safetensors --out models/mmproj-diffusiongemma-26B-A4B-it-F16.gguf
```

像其他投影器一样把结果传给 `--mmproj` 即可。
`Gemma4VisionOracleTests.MmprojTower_AgreesWithSafetensorsTower` 会检查它与分片的编码结果一致；
该测试需要本地 fixture 目录（`TS_DIFFUSIONGEMMA_VISION_DIR`），没有时会跳过。脚本同样面向
llama.cpp 的 clip 加载器编写，但没有记录过用 llama.cpp 运行其输出的结果。

原生音频**不受支持**，任何 projector 也补不上：上游 config 没有 `audio_config`，
权重里也没有音频塔，因此 tokenizer 从 Gemma 4 继承来的 `<|audio|>` token 背后
什么都没有。

类型化判定（`noul`、`choice`、`score`）请使用原生 [`/v1/systemone` Jev 端点](jev_zh-cn.md)：
它在一次去噪步中从带种子的 canvas 读取标签概率，使用稀疏输出投影，不解析生成的 JSON。
其状态支持上传文本/文档、通过视觉塔读取的图像与抽样视频帧，以及由配置的 ASR 配套服务提供的音频转录。
这些预处理通路详见 Jev 指南。可从
[`jev-diffusiongemma-q4.json`](../../config/jev-diffusiongemma-q4.json) 开始。

命令行下载（每个文件一行；需要先 `pip install -U huggingface_hub`）：

```bash
python -m pip install -U huggingface_hub
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
```

CLI diffusion 模式（根据模型架构自动分发 —— 无需模式标志；提示词通过
`--input` 从文件读取）：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt \
  --backend ggml_cuda --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

若要针对图片提问，追加 `--mmproj models/model-00011-of-00011.safetensors` 与
`--image photo.png`（多张图片可重复 `--image`，按顺序传入）。CLI 不会在模型旁边查找
这个分片；没有加载视觉塔时，它会拒绝图像，而不是在看不到图像的情况下作答。

服务端（`http://localhost:5000/index.html` 的 Web UI 会流式展示实时去噪预览 ——
每一步通过 `replace` SSE 帧重绘整条消息；Ollama/OpenAI 兼容端点只返回最终文本）：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

追加 `--mmproj models/model-00011-of-00011.safetensors` 即可在聊天请求中接受图像；
服务端从不自动探测投影器。

## 1. 来源与目标

DiffusionGemma 是基于 Gemma-4 风格 Mixture-of-Experts backbone 的分块文本扩散语言模型。
它和自回归 `gemma4` 的运行时契约不同：

- `Forward(int[] tokens)` 会主动抛错；生成必须通过 `DiffusionGemmaSampler`。
- 每个去噪步骤在拼接后的 `[prompt | canvas]` 序列上运行。
- prompt 区域是因果的，并且不会看见 canvas。
- canvas 区域对 prompt + canvas 做双向注意力。
- 每个 block 输出当前确定性的 argmax canvas，并在多步去噪中逐步收敛。

GGUF 必须报告 `general.architecture=diffusion-gemma` 或 `diffusion_gemma`；
`ModelBase.Create()` 会把这两个 key 路由到 `DiffusionGemmaModel`。

## 2. 前向计算图

模型暴露两种执行路径。

统一正确性路径是 `ForwardCanvas(tokens, promptLen)`：

```text
[prompt tokens | canvas tokens]
  -> 区域感知 embedding scale
  -> prompt/canvas attention mask
  -> N 个 Gemma 风格 transformer 层
       - local/global QK-norm attention
       - dense gated-GELU MLP
       - top-k MoE experts
       - prompt encoder scale / canvas decoder scale
  -> output norm
  -> tied lm-head
  -> final logit softcap
  -> canvas logits
```

GPU 优化路径会把每个 block 拆成 prompt prefill 与多次 canvas decode：

1. `PrefillPrompt(promptTokens)` 只计算一次 prompt K/V。
2. `DecodeCanvas(canvasTokens, scBuffer, scUse, prevTempInv)` 在每个去噪步复用 prompt K/V。
3. 采样器接受低熵位置，重新噪声化其余位置，然后继续迭代。

Prompt-KV 缓存在 device-glue 后端（`ggml_metal`、`ggml_cuda`、`mlx`、`cuda`）以及纯 C# 的 `cpu`
后端上启用，`cpu` 把 prompt K/V 保存在主机内存中（见第 5 节）；在 `ggml_cpu` 与 `ggml_vulkan` 上，每一步
都改走统一的 `[prefix|canvas]` forward。

## 3. 采样器契约

`DiffusionEbParams` 控制生成：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `MaxDenoisingSteps` | 48 | 每个 canvas block 最大去噪步数 |
| `TMin` / `TMax` | 0.4 / 0.8 | 从后期到前期的温度调度 |
| `EntropyBound` | 0.1 | 被接受位置的累计互信息上界 |
| `StabilityThreshold` | 1 | 早停前 argmax canvas 需要稳定的步数 |
| `ConfidenceThreshold` | 0.005 | 早停使用的平均熵阈值 |
| `Seed` | 0 | 确定性采样种子 |
| `MaxBlocks` | 1 | block-autoregressive canvas 数量 |

CLI 对应：

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --backend ggml_metal \
  --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

当 `--diffusion-blocks` 为 `0` 时，CLI 会根据 `--max-tokens` 与
`diffusion.canvas_length` 推导 block 数。

## 4. 架构细节

DiffusionGemma 复用了许多 Gemma-4 backbone 设计：

- NeoX RoPE，并区分 local/global 维度。
- 五个 local sliding-window 层加一个 global 层的循环模式。
- 每头 Q/K RMSNorm 与无权重 V RMSNorm。
- Global 层可以没有 `attn_v.weight`，此时 V 使用原始 K 投影。
- Dense gated-GELU MLP 加 128-expert top-8 MoE。
- 共享 embedding / lm-head，并带 final logit softcapping。

Diffusion 专属元数据：

| Key | 含义 |
|---|---|
| `diffusion.canvas_length` | 每个 block 去噪的 canvas 位置数，默认 256 |
| `tokenizer.ggml.mask_token_id` | warmup 与回退路径使用的 mask token id |
| `<arch>.attention.sliding_window_pattern` | local/global 层模式 |
| `<arch>.attention.head_count_kv` | 每层 KV head 数 |
| `<arch>.expert_count` / `<arch>.expert_used_count` | MoE expert 总数与 active top-k |

## 5. 加速状态

当前优化路径包括：

- `ggml_metal`、`ggml_cuda`、`mlx`、`cuda` 与 `cpu` 上的 prompt-KV 缓存（`ggml_cpu` 与 `ggml_vulkan` 上没有）。
- 默认启用 self-conditioning；可用 `DIFFUSION_NO_SC=1` 关闭。
- GGML 融合 decode layer、融合整模型 decode、融合 lm-head tail。
- CUDA VRAM 常驻规划：当模型大于 VRAM 时，按优先级把权重预加载到设备
  （lm_head/embedding、逐层 attention/dense、再到 MoE 专家堆叠），上限为
  可用 VRAM 减去余量；对 device-copy 缓存设置上限；并把 decode 切换到
  逐层分段融合路径，使未常驻的部分通过一个有界的复用 staging 缓冲流式
  上传，而不是超订 VRAM（超订会让 Windows WDDM 在每次提交时换页，实测
  比流式上传慢约 4 倍）。
- 与去噪步无关的 decode mask 在 host 端缓存并以 cacheable 方式绑定
  （每个 block 几何只上传一次，而不是每层每步重建并上传）。
- SIMD 向量化 host 路径（`TensorPrimitives`）：每位置
  argmax/熵/多项式采样以及 final-logit softcap；融合 lm-head 的 logits
  写入一个池化的 pinned 缓冲，而不是每步新分配 268 MB。
- 针对 DiffusionGemma 多行 canvas 工作负载的 MLX K-quant affine repack。
- `TensorSharp.Server` 中通过 `DiffusionBatchScheduler` 做 block 边界连续批处理。
- 带有自己的 prompt-KV 缓存与批处理内核的纯 C# `cpu` 路径，见下文。

### 纯 C# CPU 后端（`--backend cpu`）

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --backend cpu \
  --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

服务端与 [Jev 端点](jev_zh-cn.md)接受同样的 `--backend cpu`。在这个后端上
（[`DiffusionGemmaModel.Cpu.cs`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaModel.Cpu.cs)）：

- **Prompt K/V 缓存在主机上。** `PrefillPrompt` 把每层经过 norm 与 RoPE 之后的 K/V 按 token 顺序保存；
  滑动窗口层只保留 canvas 查询能看到的最后若干行。此后每个去噪步只计算 canvas；对已经 prefill 过的提示，
  Jev 读取会复用保存的 K/V。调度器的多请求 decode 在主机上把活跃的 canvas 批在一起。
- **同一输入的多个投影共用一次调度**（Q/K/V，以及 gate/up）；MoE 把所有活跃专家的 gate/up 与 down 投影
  作为两次批量调度，交给多行量化 GEMM，每批 `DIFFUSION_CPU_MOE_CHUNK` 个 token。
- **注意力**是分块内核，默认算术精确复现旧内核（`DIFFUSION_CPU_ATTN_FAST=1` 改用 FMA 分块）。这个模型对
  最后一个比特的变化异常敏感：每次 matmul 前都会量化激活，每层又要从 128 个专家里选 8 个，路由上的平局
  可能翻转。在数值上等价的不同内核之间，单字段 Jev 概率的变化可达 ±0.2，因此下文按标签判定来评估质量。
- `DIFFUSION_NO_PKV=1` 关闭缓存；`DIFFUSION_CPU_LEGACY=1` 恢复这个模型这一侧的旧 CPU 路径，按阶段的开关各恢复一个
  阶段（见下表）。它们调用的共享 CPU 内核也已重写，要与旧版本逐位一致还需
  `TS_CPU_SIMD_ELEMENTWISE=0 TS_CPU_SGEMM=0 TS_CPU_QGEMM=0`。

在 i7-11800H（8 核 16 线程、AVX-512）、32 GB、Windows 上，用 `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` 与
`eng/JevProbe` 实测（对一个 54 token 的提示做结构化读取；"旧构建"指本次 `cpu` 改动之前的构建）：

| Jev 结构化读取 | `cpu` | `cpu`，旧构建 | `ggml_cpu` |
|---|---:|---:|---:|
| 宽度 16，新提示的首次读取 | 0.92 s | 9.5 s | 1.57 s |
| 宽度 16，同一提示的后续读取 | 0.22–0.25 s | 9.5 s | 1.57 s |
| 宽度 64，新提示的首次读取 | 1.42 s | 未测 | 2.68 s |
| 宽度 64，同一提示的后续读取 | 0.63–0.73 s | 未测 | 2.68 s |

`ggml_cpu` 没有 prompt-KV 缓存，所以那里每次读取的代价都一样。在 `--quality` 集合（9 个双问题提示，
每个分别以宽度 16 与 64 读取）上，`cpu` 与 `ggml_cpu` 都选中了全部 36 个预期标签（平均概率 0.9906 与 0.9886），两者的
首选标签 36 个全部一致，最大概率差为 0.049。`cpu` 上的聊天生成使用同样的内核与缓存，但其速度没有测量。

重要开关：

| 变量 | 作用 |
|---|---|
| `DIFFUSION_STEPS` | 服务端每个 block 的去噪步数，默认 48 |
| `DIFFUSION_MAX_BATCH` | 服务端 diffusion scheduler 最大活跃请求数，默认 2 |
| `DIFFUSION_NO_PKV=1` | 关闭 device-glue 后端与 `cpu` 上的 prompt-KV 缓存 |
| `DIFFUSION_CPU_LEGACY=1` | `cpu`：恢复这个模型的旧 CPU 路径（没有 prompt-KV 缓存，投影、注意力、路由与 MoE 都用旧实现）；只有同时设置 `TS_CPU_SIMD_ELEMENTWISE=0 TS_CPU_SGEMM=0 TS_CPU_QGEMM=0` 才与旧版本逐位一致 |
| `DIFFUSION_CPU_LEGACY_MOE` / `_PROJ` / `_ATTN` / `_ROUTER` `=1` | `cpu`：恢复单个阶段。`_ATTN` 只作用于统一前向，因此注意力的 A/B 还需要 `DIFFUSION_NO_PKV=1` |
| `DIFFUSION_CPU_ATTN_FAST=1` | `cpu`：用 FMA 注意力分块与向量化 softmax 代替精确的默认内核 |
| `DIFFUSION_CPU_MOE_CHUNK` | `cpu`：每次批量 MoE 处理的 token 数，默认 512 |
| `DIFFUSION_NO_SC=1` | 关闭 self-conditioning |
| `DIFFUSION_SC_TOPK` | 实验用 self-conditioning top-K 截断，默认 32 |
| `DIFFUSION_BATCHED_FORWARD=1` | 使用真正的批处理 canvas decode，而不是按时间片执行融合单 canvas decode |
| `DIFFUSION_NO_FUSED_DECODE=1` | 关闭 GGML 融合整模型 diffusion decode |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL=1` | 关闭融合 output-norm + lm-head + softcap tail |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | 临时批处理 lm-head logits 的内存上限，默认 300 MB |
| `DIFFUSION_VRAM_HEADROOM_MB` | ggml_cuda：预加载权重之外保留的 VRAM 余量，默认 2048 |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | ggml_cuda：模型放不进 VRAM 时 device-copy 缓存上限，默认 768 |
| `DIFFUSION_SEGMENTED_DECODE` | ggml_cuda：强制开启/关闭逐层融合 decode（`1`/`0`，放不进 VRAM 时自动启用） |
| `DIFFUSION_PIN_STREAMED=1` | ggml_cuda：把流式权重复制到页锁定内存以 DMA 速度上传（消耗 RAM） |
| `DIFFUSION_FUSED_PREFILL_ATTN` | 融合的 GGML prompt 注意力：`ggml_cuda` 上默认开启（`0` 恢复逐算子参考路径），其他 GGML 后端可用 `1` 选择开启，但这不代表已经过验证；图像提示始终关闭，因为它需要图像 span 的双向 mask |
| `DIFFUSION_NO_DEVICE_SAMPLE=1` | ggml_cuda：关闭设备端 argmax / 熵 / 采样 / self-conditioning top-K（模型放得进 VRAM 时默认开启） |
| `DIFFUSION_DEVICE_SAMPLE_FORCE=1` | ggml_cuda：即使模型放不进 VRAM（分段 decode，此时通常使用 host 采样器）也保持设备端采样；仅用于实验 |
| `DIFFUSION_IMAGE_BIDIRECTIONAL=0` | 让图像软 token span 变为因果注意力；默认在滑动窗口层上对 span 内部做双向注意力 |
| `DIFFUSION_ASYNC_COMPUTE=1` | ggml_metal：保持 Metal 的 lazy-sync 异步计算开启（GGML 只在 Metal 上启用它）。模型在加载时会关闭它，因为其逐算子路径会在 CPU 上写入张量（MoE 专家输入、embedding、mask、self-conditioning、每步重新加噪的 canvas），而 Metal 没有 host 写屏障，整轮缓存的 prompt K/V 可能因此被破坏，回答流畅但答非所问；不安全，仅用于测量同步的开销 |

## 6. 服务端行为

当 Web UI 承载 DiffusionGemma GGUF 时：

- `/api/chat` 会进入 diffusion 路径。
- 流式输出发送 `replace` 事件而不是 token append，因为每一步都会重新修正整个 canvas。
- 在 `done` 事件前会先发送最终定稿 replacement。
- 并发请求共享一个后台 diffusion scheduler，并在 block 之间被接纳。
- Jev 结构化读取或新一轮对话的图像编码不必等聊天的整个 block：scheduler 会在下一次前向之前把模型交出，
  所以只需等正在进行的那一次前向。block 的输出不变。
- 在没有 prompt-KV 缓存的后端（`ggml_cpu`、`ggml_vulkan`，以及设置了 `DIFFUSION_NO_PKV=1` 或
  `DIFFUSION_CPU_LEGACY=1` 的 `cpu`）上，scheduler 会让每个序列的每一步走统一的 `[prefix|canvas]`
  前向，而不是 prefill + canvas decode；行为与输出完全一致。
- 在 `cpu` 与 `ggml_cpu` 上，服务端对该模型跳过启动时的共享提示预热。DiffusionGemma 不在请求之间保留
  任何状态，而这次预热是在打开端口之前，按每种思考模式各对 256 token 的 canvas 做一次完整的 48 步去噪
  （在上文的 `cpu` 改动之前，`cpu` 上第一次就用了 252.8 s）。GPU 后端保留预热：它会在第一个请求之前构建 prefill、融合 decode 与
  lm-head 计算图。
- 图像回合需要用 `--mmproj` 加载视觉塔。每张图片在上下文检查之前展开为其软 token
  span；编码后的 span 归 scheduler 中的该序列所有，每次 prefill 该序列的提示时都会
  重新应用（每个 block 一次；在没有 prompt-KV 缓存的后端上则是每一步），因此图像请求
  与文本请求可以一起批处理。普通聊天拒绝音频，没有原生视频通路：OpenAI 的 `video_url` 会被拒绝；
  在 Web UI 中上传的视频只以抽取出的帧送入模型，每帧按普通 `<|image>` 渲染（没有
  `<|video>` 标记，也没有帧时间戳）。独立的 [Jev 端点](jev_zh-cn.md#文件文档视频与音频)支持
  有界视频抽帧，以及 ASR 配套服务的语音转录。

Ollama 与 OpenAI 兼容适配器仍通过 `ChatStreamWithMetricsAsync` 使用 append-oriented
响应形状。它们可以返回 DiffusionGemma 的最终文本，但实时去噪预览与 `replace`
帧只在 Web UI 中提供。

模型使用 Gemma 4 的 channel 语法：一张 canvas 可能以 `<|channel>thought\n` 开头，或
用一个孤立的 `<channel|>` 关闭提示里打开的思维块，然后才是答案。该架构注册为
`diffusion-gemma` 聊天协议（`Gemma4OutputParser`，始终必需，提示仍由 GGUF 模板渲染），
每一帧预览和最终文本都经过该解析器：除非请求开启推理（`"think": true` 时以
`reasoning_content` 返回），思维块会被丢弃，channel 标记永远不会到达客户端。此前
原始 canvas 被原样返回，OpenAI 的回答以字面的 `<|channel>thought` 标记开头。

该检查点经常在思维块内部作答，而且从不写出结束的 `<channel|>`。已完成的 canvas 不是被截断的
思考，因此当解析结果只有思维块、没有答案时，这段文本会作为答案返回（不会再作为推理重复返回）。
如果不这样处理，Q4_K_M 文件上 6 个一句话问题中有 3 个返回空内容。同样，若 canvas 上只有模型
自行写出的工具调用，它会以文本形式显示，而不是返回空答案；若同时还有其他答案文本，该调用会被丢弃。

工具调用会被直接拒绝：加载 DiffusionGemma 模型时，`/v1/chat/completions` 对任何带
`tools` 或 `tool_choice`（`"none"` 除外）的请求返回 HTTP 400（`{"error": ...}`，
`invalid_request_error`），因为分块扩散的一轮没有可以回填结果的工具循环。
`/v1/responses` 与 Ollama 端点 `/api/chat/ollama` 以同样方式拒绝 `tools`。内置的
skills / 代码执行工具以及子智能体协调工具也永远不会提供给该系列（协议条目声明
`RendersToolDeclarations = false`），因此 `--code-exec`、skills 发现与子智能体委派
都不会改变扩散请求的任何行为。

## 7. 测试覆盖

[`DiffusionGemmaTests`](../../InferenceWeb.Tests/DiffusionGemmaTests.cs) 通过
`TS_TEST_MODEL_DIR` 指向真实 GGUF 后按需运行，覆盖：

- `ForwardCanvas` 有限 logits 正确性。
- 端到端 EntropyBound 生成。
- Prompt-KV 等价性与速度探针。
- 重复 token 输出与 device memory 留存的回归保护。
- 与服务端调度风格一致的批处理 decode 等价性和双请求生成。

`cpu` 路径另有三组测试。
[`DiffusionGemmaCpuForwardTests`](../../InferenceWeb.Tests/DiffusionGemmaCpuForwardTests.cs)
（同样由 `TS_TEST_MODEL_DIR` 启用）在真实 GGUF 上检查：prompt-KV 读取与统一前向逐比特一致（包括超过
滑动窗口的长提示），批处理 canvas decode 与逐序列 decode 逐比特一致，带缓存的生成与统一前向一致。
[`DiffusionGemmaCpuKernelTests`](../../InferenceWeb.Tests/DiffusionGemmaCpuKernelTests.cs)
与 [`DiffusionGemmaQuantContractTests`](../../InferenceWeb.Tests/DiffusionGemmaQuantContractTests.cs)
不需要权重：它们把融合的 norm/RoPE、GELU、路由与注意力内核逐比特固定到被替换的算术上，并固定"某一行的量化
matmul 结果与同批有多少行、在哪个批里运行无关"。

[`DiffusionGemmaProtocolTests`](../../InferenceWeb.Tests/DiffusionGemmaProtocolTests.cs)
不需要权重，固定了 channel 解析行为：未请求时丢弃思维块，未闭合的思维块作为答案，模型自行
写出的工具调用以文本呈现。

图像输入另有两组测试。
[`DiffusionGemmaVisionConcurrencyTests`](../../InferenceWeb.Tests/DiffusionGemmaVisionConcurrencyTests.cs)
不需要权重，它固定了“图像 span 属于序列”这一性质，因此两个同时在途的图像请求不会互相覆盖。
[`Gemma4VisionOracleTests`](../../InferenceWeb.Tests/Gemma4VisionOracleTests.cs)
（通过 `TS_DIFFUSIONGEMMA_VISION_DIR` 按需启用）把从分片加载的视觉塔与 Hugging Face
参考实现的 NumPy 转写（`eng/diffusiongemma-vision-oracle.py`）进行比较。

## 8. 后续工作

- 等 Ollama/OpenAI 适配层有 diffusion-aware 兼容面后，再补充专门 API 示例。
- 只有在目标 GPU 上胜过当前路径时，才把真正批处理 canvas decode 提升为默认；目前单个 canvas 已经能吃满 GPU 时，融合单 canvas 路径可能更快。
- 如运维需要更细粒度可见性，可把更多 diffusion scheduler 指标接入 `/api/queue/status`。
