# 使用方法
[English](USAGE.md) | [中文](USAGE_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。快速开始命令见 [README](README_zh-cn.md#快速开始)；配置文件见 [config/README.md](config/README.md)。

## 计算后端

| 后端 | 参数 | 适合场景 | 说明 |
|---|---|---|---|
| Direct CUDA/cuBLAS | `--backend cuda` | NVIDIA 推理与实验 | 通过 CUDA Driver API、cuBLAS GEMM、常用 Float32 PTX 内核（fill、unary、binary、ternary、activations、RMSNorm、softmax、RoPE/RoPEEx、SDPA、GQA prefill/decode、causal mask、gather/concat），以及受支持 GGUF 量化类型的原生量化 matmul/get-rows 加速推理；未实现的算子会回退到 CPU，同时保持张量语义。 |
| MLX Metal | `--backend mlx` | Apple Silicon（GGML Metal 之外的另一选择） | 基于 [mlx-c](https://github.com/ml-explore/mlx-c) 的 GPU 加速路径。实现了原生量化算子（Q4_K_M、Q8_0、Q5_K、Q6_K、IQ2_XXS、IQ4_XS、IQ4_NL、MXFP4 等无需反量化到 FP32）、融合 decode / prefill Metal kernel（融合 QKV 预处理、融合 gate+up+SiLUMul MoE、融合多维 KV 写入）、编译图 kernel、定期 `async_eval` 让 GPU/CPU 工作重叠的异步 worker 派发、用堆叠权重 slab 的批处理 MoE 解码、MoE 专家 offload、通过 `mlock(2)` 把 GGUF mmap 钉在物理内存、按宿主机派生的分配器上限（`TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB`），并对未实现的算子提供 CPU 回退。依赖 `libmlxc`（可通过 `TensorSharp.Backends.MLX/build-native-macos.sh` 在本地编译，或用 `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` 指定路径）。 |
| GGML Metal | `--backend ggml_metal` | Apple Silicon（macOS 默认） | 通过 Apple Metal 进行 GPU 加速。量化权重通过 host 指针缓冲区从 GGUF 文件零拷贝映射到 Metal command buffer，常驻内存接近模型在磁盘上的大小。 |
| GGML CUDA | `--backend ggml_cuda` | 通过 ggml 使用 NVIDIA 推理 | 通过 GGML CUDA 在 Windows 或 Linux + NVIDIA GPU 上进行加速。量化权重在加载时一次性上传到设备显存，之后释放主机端拷贝。 |
| GGML Vulkan | `--backend ggml_vulkan` | 通过 ggml 的厂商无关 GPU 推理 | 通过 GGML Vulkan 在 Windows 或 Linux 上加速——支持带 Vulkan 1.3 驱动的 AMD、Intel 与 NVIDIA GPU，驱动支持时使用 cooperative-matrix（KHR coopmat / NV coopmat2）着色器。权重与 GGML CUDA 一样常驻显存，并复用同样的融合整模型 decode/prefill 图。机器有 Vulkan 运行时（已安装 loader）时原生构建会自动启用；未安装 Vulkan SDK 或发行版开发包时，构建会通过 `eng/fetch-vulkan-toolchain.ps1` / `eng/fetch-vulkan-toolchain.sh` 自动下载便携工具链（headers、glslc、SPIRV-Headers，Windows 上还有 loader 导入库）。用 `--no-vulkan`（或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`）退出。 |
| GGML CPU | `--backend ggml_cpu` | 原生 CPU 内核 | 使用原生 GGML 与优化内核进行 CPU 推理。量化权重以零拷贝方式从 GGUF 文件映射。 |
| 纯 C# CPU | `--backend cpu` | 可移植性与调试 | 无原生依赖的可移植 CPU 推理。 |



## 配置文件（CLI + Server）

`TensorSharp.Cli` 与 `TensorSharp.Server` 都可以通过 `--config` 从一个 JSON 文件读取参数，
以替代（或补充）冗长的命令行：

```bash
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --config config/server-basic.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll       --config config/cli-basic.json
```

**命令行参数始终优先。** 先应用文件中的值，命令行上再次给出的参数会覆盖它们——因此可以在多台机器上
复用同一个文件，只覆盖需要变化的部分（`--config config/server-basic.json --backend ggml_cpu`）。
`--config` 可重复出现以叠加多个文件；后出现的文件优先。

键名与下文列出的长选项名相同（可带或不带前缀 `--`）。允许注释（`//`、`/* */`）与尾随逗号。

| JSON 值 | 展开为 | 示例 |
|---|---|---|
| 字符串 / 数字 | `--key value` | `"max-tokens": 4096` → `--max-tokens 4096` |
| `true` | 裸开关 `--key` | `"continuous-batching": true` → `--continuous-batching` |
| `false` / `null` | 什么都不加（要关闭请用取反键，如 `"no-continuous-batching": true`） | |
| 数组 | 重复的标志 | `"stop": ["</s>", "<\|eot\|>"]` → `--stop </s> --stop <\|eot\|>` |
| 对象 | 可下载文件（见下） | `{ "path": "...", "urls": ["..."] }` |

**变量。** 在 `"variables"` 中定义一次共享值，用 `${name}` 在任意字符串值中引用。未定义的 `${name}`
会回退到同名环境变量，变量之间也可以互相引用。可以定义任意多个根路径——位于不同目录的模型各用一个。

```json
{
  "variables": { "modelRoot": "C:/models" },
  "backend": "ggml_cuda",
  "model": "${modelRoot}/Qwen3.5-9B-Q8_0.gguf",
  "mmproj": "${modelRoot}/Qwen3.5-mmproj-F16.gguf"
}
```

**自动下载。** 任何文件参数都可以写成带本地 `path` 与一个或多个 `urls` 的对象，而非普通字符串。
若 `path` 不存在，则从第一个可用 URL 下载（镜像按顺序尝试），保存到该路径，之后每次运行复用；
下载进度打印到 stderr。可选的 `sha256` 用于校验刚下载的文件。

```json
{
  "backend": "ggml_cuda",
  "model": {
    "path": "C:/models/Qwen3.5-9B-Q8_0.gguf",
    "urls": [ "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf" ]
  }
}
```

开箱即用的示例见 [`config/`](config/)（`cli-basic.json`、`server-basic.json`、`variables.json`、
`auto-download.json`、`qwen-image-edit.json`）——每个都使用真实、公开、无需授权的 URL，
因此在全新机器上也能直接运行。完整说明见 [`config/README.md`](config/README.md)。

## 控制台应用

```bash
# 文本推理
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Windows/Linux + NVIDIA GPU 文本推理
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# 交互式逐轮对话（REPL），支持 KV 缓存复用与斜杠命令
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal --interactive
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal -i \
    --system "你是一名简洁的助手。" --temperature 0.7 --top-p 0.9 --think

# 图像推理（Gemma 3/4，Qwen 3.5-family）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --image photo.png --backend ggml_metal

# 视频推理（Gemma 4）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --video clip.mp4 --backend ggml_metal

# 音频推理（Gemma 4）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --audio speech.wav --backend ggml_metal

# DiffusionGemma 文本扩散生成
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <diffusion-gemma.gguf> --input prompt.txt --backend ggml_metal \
    --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0

# Qwen-Image-Edit 图像编辑（提示词 + 输入图像 -> 编辑后的图像）
# VAE + Qwen2.5-VL 文本编码器伴随文件会在 DiT GGUF 旁解析
# （或用 --qwen-image-vae / --qwen-image-vl / --qwen-image-mmproj 指定）。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <qwen-image-edit-DiT.gguf> --image input.png \
    --prompt "Make the sky a dramatic sunset." --output edited.png \
    --backend ggml_cuda --diffusion-steps 30 --cfg 2.5 --diffusion-seed 0

# 思维链 / 推理模式
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# 工具调用
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# 使用采样参数
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# 批处理（JSONL）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# 多轮对话模拟（含 KV 缓存复用，模拟 Web UI 行为）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# 吞吐基准测试：N 次最优运行的 prefill 和 decode 计时
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# KV 缓存复用基准：在多轮对话中比较启用与禁用缓存的 prefill 时延
# （以一个 8 轮的对话为例，对比有缓存与强制重置的 prefill 延迟差异）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --bench-kvcache --bench-kv-turns 4 --max-tokens 64

# 仅查看渲染后的 prompt 和分词结果（不运行推理）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --dump-prompt

# 对目录下每个 *.gguf 文件，对比硬编码回退模板与 GGUF 内置 Jinja2 模板
# （在适配新架构时尤其有用）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --test-templates ~/models
```

**命令行参数：**

| 参数 | 说明 |
|---|---|
| `--model <path>` | GGUF 模型文件路径（必填） |
| `--input <path>` | 包含用户提示词的文本文件 |
| `--input-jsonl <path>` | JSONL 批量请求文件（每行一个 JSON） |
| `--multi-turn-jsonl <path>` | 用于多轮对话模拟（含 KV 缓存复用）的 JSONL 文件 |
| `--output <path>` | 将生成文本写入该文件 |
| `--image <path>` | 用于视觉推理的图像文件 |
| `--video <path>` | 用于视频推理的视频文件 |
| `--audio <path>` | 音频文件（WAV、MP3、OGG）用于音频推理 |
| `--pdf <path>` | PDF 文档输入（单次推理模式）。文字型 PDF 会提取并内联完整文本层（页数上限由 `TS_PDF_MAX_PAGES` 控制）；扫描型 PDF 会栅格化为页面图像，并需要视觉模型（`--mmproj` 或内置视觉编码器）。`--input` 文本作为针对文档的指令。 |
| `--mmproj <path>` | 多模态投影器 GGUF 文件路径 |
| `--max-tokens <N>` | 最大生成 token 数（默认：100） |
| `--backend <type>` | 计算后端：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan` |
| `--gpu-device <N>` | `ggml_vulkan` 后端使用的 Vulkan 设备索引，用于多 GPU 主机（例如同时装有 Intel 集成显卡和 NVIDIA 独立显卡的机器）。默认使用设备 0；可用 `--list-gpus` 查看索引。也可通过环境变量 `TS_GGML_VULKAN_DEVICE` 设置。 |
| `--list-gpus` | 列出 ggml-vulkan 可见的 Vulkan 设备（索引 + 显卡名称）后退出 |
| `--kv-cache-dtype <type>` | KV 缓存精度：`f32`（默认）、`f16`、`q8_0` 或 `q4_0`。量化 / 半精度 KV 缓存以微小数值漂移换取内存节省；`q4_0`（约 0.56 字节/元素，约为 f32 的 1/7）是最激进的档位，面向 KV 缓存占主导内存的超长（128K–256K）上下文。块量化缓存（`q8_0`/`q4_0`）需要原生 GGML flash 路径。 |
| `--tp <N>` | 张量并行度 —— 在单个进程内把模型切分到 N 张 CUDA GPU 上（默认：`1`）。需要 `--backend cuda`。详见[张量并行与分布式推理](#张量并行与分布式推理)。 |
| `--tp-node-id <N>` | 多节点分布式张量并行中本节点的 0 起始编号。必须与 `--tp-peers` 一起使用。 |
| `--tp-peers <list>` | 集群中所有节点的 `host:port` 列表（逗号分隔，例如 `192.168.1.10:9500,192.168.1.11:9500`）。所有节点必须使用完全相同的列表。必须与 `--tp-node-id` 一起使用。 |
| `--interactive` / `-i` | 进入交互式 REPL 聊天会话（逐轮输入/输出），支持 KV 缓存复用、斜杠命令、运行时热切换 模型/后端/投影器、文件附件（图像、音频、视频、文本）以及实时调整采样参数。完整命令列表见下文「**交互式 REPL 命令**」一节 |
| `--system <text>` | 用于初始化交互式会话的系统提示词（在 REPL 中可用 `/system` 覆盖） |
| `--system-file <path>` | 从 UTF-8 文本文件读取初始系统提示词（`--system` 的替代写法） |
| `--think` | 启用思维链/推理模式 |
| `--tools <path>` | 包含工具/函数定义的 JSON 文件 |
| `--temperature <f>` | 采样温度（0 = 贪心） |
| `--top-k <N>` | Top-K 过滤（0 = 关闭） |
| `--top-p <f>` | Nucleus 采样阈值（1.0 = 关闭） |
| `--min-p <f>` | 最小概率过滤（0 = 关闭） |
| `--repeat-penalty <f>` | 重复惩罚（1.0 = 无） |
| `--presence-penalty <f>` | 存在惩罚（0 = 关闭） |
| `--frequency-penalty <f>` | 频率惩罚（0 = 关闭） |
| `--seed <N>` | 随机种子（-1 = 非确定性） |
| `--stop <string>` | 停止序列（可重复指定） |
| `--dump-prompt` | 仅渲染 prompt 与分词后退出（不进行推理） |
| `--benchmark` | 运行合成的 prefill / decode 吞吐基准 |
| `--bench-prefill <N>` | 合成 prefill 的 token 长度（默认：32） |
| `--bench-decode <N>` | 合成 decode 的 token 长度（默认：64） |
| `--bench-runs <N>` | 基准运行次数；输出最佳与平均结果（默认：1） |
| `--bench-kvcache` | 运行多轮 KV 缓存复用基准（对比启用缓存与强制重置时的 prefill 延迟） |
| `--bench-kv-turns <N>` | `--bench-kvcache` 使用的对话轮数（默认：4，最多 8） |
| `--bench-chunked` | 运行分块 prefill 微基准（Gemma 4） |
| `--warmup-runs <N>` | 在对真实文本 / 多模态 prompt 计时前丢弃的前向次数（默认：0） |
| `--test-chunked-prefill` | 运行分块 prefill 正确性检查（对比分块与非分块 logits） |
| `--correct-prefill <N>` | `--test-chunked-prefill` 使用的 prompt 长度 |
| `--correct-decode <N>` | `--test-chunked-prefill` 使用的 decode 长度 |
| `--diffusion-steps <N>` | DiffusionGemma 每个 block 的去噪步数（默认：48）。对 Qwen-Image-Edit 则是 FlowMatch-Euler 步数——省略时自动选择（30，或已加载 Lightning LoRA 的步数）。 |
| `--diffusion-seed <N>` | DiffusionGemma 确定性采样种子（默认：0） |
| `--diffusion-blocks <N>` | DiffusionGemma block-autoregressive canvas 数量。`0` 表示根据 `--max-tokens` 与模型 canvas 长度推导。 |
| `--image <path>` | Qwen-Image-Edit 的输入图像（也是多模态聊天的图像输入）。在 `qwen_image` DiT GGUF 上触发图像编辑模式所必需。 |
| `--prompt <text>` | Qwen-Image-Edit 编辑指令（省略时回退到 `--input` 文件内容）。 |
| `--output <path>` | Qwen-Image-Edit 输出 PNG 路径（默认：`edited.png`）。 |
| `--cfg <F>` | Qwen-Image-Edit true-CFG 引导尺度（`<= 1` 关闭负向分支）。省略时自动选择：2.5（Qwen-Image-Edit-2511 的推荐值；4.0 会过度引导并扭曲人脸），加载 Lightning LoRA 时为 1.0。步数与种子复用 `--diffusion-steps` / `--diffusion-seed`。 |
| `--qwen-image-vae <path>` | 覆盖解析到的 Qwen-Image VAE 伴随文件（`.gguf` 或 `.safetensors`）。 |
| `--qwen-image-vl <path>` | 覆盖解析到的 Qwen2.5-VL-7B 文本编码器 GGUF。 |
| `--qwen-image-mmproj <path>` | 覆盖解析到的 Qwen2.5-VL mmproj（视觉接地）GGUF。 |
| `--qwen-image-lora <path>` | Qwen-Image-Edit 的 Lightning 蒸馏 LoRA（`.safetensors`），在加载时合并进 DiT。自动推导步数（例如 4 或 8）并把 CFG 切换为 1.0。环境变量：`TS_QWEN_IMAGE_LORA`。 |
| `--test` | 运行内置的分词器、Qwen3 聊天模板与 ollama 对比测试 |
| `--test-templates <dir>` | 对 `<dir>` 下的每个 *.gguf 校验硬编码模板与 GGUF Jinja2 模板的一致性 |
| `--config <path>` | 从 JSON 配置文件读取参数（命令行参数会覆盖它）。支持 `${变量}` 与通过 `{ "path": ..., "urls": [...] }` 自动下载模型。可重复。见[配置文件](#配置文件cli--server)。 |
| `--log-level <lvl>` | 控制台与文件日志级别：`trace`、`debug`、`info`、`warning`、`error`、`critical`、`off` |
| `--log-dir <path>` | JSON-line 文件日志的写入目录（默认：`<binDir>/logs`） |
| `--log-file <0\|1>` | 关闭（`0`）或开启（`1`）文件日志（默认：开启） |
| `--log-console <0\|1>` | 关闭（`0`）或开启（`1`）控制台日志（默认：开启） |

CLI 只会自动识别少数旧式投影器文件名，而当前模型仓库经常使用不同名称。多模态运行请用 `--mmproj` 显式传入已下载文件；`TensorSharp.Server` 从不自动检测投影器。

**JSONL 输入格式：**

每行是一个 JSON 对象，包含 `messages`、可选 `prompt` 和可选采样参数：

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

**交互式 REPL 命令：**

通过 `--interactive` / `-i` 启动后，可使用斜杠命令驱动当前会话。在 REPL 中输入 `/help`（或 `/?`）可查看相同的命令列表。任何不以 `/` 开头的输入都会被视为一轮用户消息。

每轮提示符前的状态行会汇总当前状态——模型、后端、架构、上下文长度、投影器、对话深度，以及为下一轮排队的附件数量（例如 `[turn 3 (2 attachments pending)]> `）。生成过程中按 Ctrl+C 可中断当前回复；在提示符处按 Ctrl+C 可退出。

会话控制：

| 命令 | 说明 |
|---|---|
| `/help`、`/?` | 显示全部交互命令 |
| `/exit`、`/quit` | 退出当前会话 |
| `/reset`、`/new` | 清空对话历史与 KV 缓存 |
| `/history` | 打印对话历史 |
| `/save <文件>` | 将当前对话追加写入 UTF-8 文件 |
| `/system <文本>` | 设置系统提示词（参数为空表示清空），并重置 KV 缓存 |
| `/think on\|off` | 切换思维链/推理模式（仅对支持的模型生效） |
| `/multiline on\|off` | 切换多行输入（在单独一行输入 `.` 结束消息） |

模型与运行时：

| 命令 | 说明 |
|---|---|
| `/info`、`/status` | 显示当前加载的模型、后端、架构、上下文/词表大小、投影器、对话深度与待发送附件 |
| `/model <路径>` | 在当前后端上加载另一个 `.gguf` 模型（会重置会话） |
| `/backend <名称>` | 用其他后端重新加载当前模型：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan` |
| `/mmproj <路径>` | 为当前模型加载（或替换）多模态投影器。别名：`/projector` |

采样（实时生效，跨多轮持久化）：

| 命令 | 说明 |
|---|---|
| `/sampling`、`/show` | 打印当前采样配置 |
| `/max <N>` | 单次回复最大 token 数 |
| `/temp <float>` | 采样温度（0 = 贪心） |
| `/topk <int>` | Top-K 过滤（0 = 关闭） |
| `/topp <float>` | Top-P / Nucleus 阈值（1.0 = 关闭） |
| `/minp <float>` | Min-P 过滤（0 = 关闭） |
| `/repeat <float>` | 重复惩罚（1.0 = 关闭） |
| `/presence <float>` | 存在惩罚 |
| `/frequency <float>` | 频率惩罚 |
| `/seed <int>` | 随机种子（-1 = 非确定性） |
| `/stop <文本>` | 追加一条停止序列 |
| `/clearstop` | 清空所有停止序列 |

附件上传（排队到下一轮，发送后自动清空）：

| 命令 | 说明 |
|---|---|
| `/image <路径>`、`/img <路径>` | 附加一张图像（仅对视觉模型有效） |
| `/audio <路径>` | 附加一个音频文件（Gemma 4） |
| `/video <路径>`、`/vid <路径>` | 附加视频，自动抽取关键帧（Gemma 4） |
| `/text <路径>`、`/file <路径>`、`/txt <路径>` | 将 UTF-8 文本/Markdown/CSV/代码文件的前 256 KiB 内联到下一轮提示词中 |
| `/clearattach` | 清空尚未发送的图像/音频/视频/文本附件 |

路径支持单引号或双引号，因此可以直接在 macOS 上从 Finder 拖拽文件到终端。多模态命令需要先加载多模态投影器——在启动时通过 `--mmproj` 指定，或在 REPL 中用 `/mmproj <路径>` 加载。

## Web 应用

在构建完成后，从仓库根目录运行：

```bash
# 通过 --model 指定要托管的模型
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --backend ggml_cuda

# 多模态模型：同时显式指定投影器
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda

# 配置服务端默认采样参数（仅在请求未自行覆盖时生效）
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/model.gguf --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42 \
    --stop "</s>" --stop "<|endoftext|>"

# 用可复用的 JSON 文件读取以上全部参数（首次运行会自动下载模型）。
# 示例见“配置文件”一节与 config/ 目录。
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --config config/server-basic.json
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --config config/server-basic.json --backend ggml_cpu
```

在浏览器中打开 `http://localhost:5000/index.html`（`GET /` 是存活检查接口）。Web 界面支持：

- 多轮聊天
- 每个浏览器 Tab 独立的会话：每个 Tab 拥有自己的对话历史；KV block 由推理引擎统一管理
- 通过 `--model` 显式托管单个 GGUF 模型
- 在需要时通过 `--mmproj` 显式托管多模态投影器
- 完整上传文本和 PDF 文档，并可上传图像、视频和音频进行多模态推理（最大 500 MB）
- 思维链/推理模式切换
- 带函数定义的工具调用
- 通过 Server-Sent Events 进行流式 token 生成
- 承载 `diffusion-gemma` GGUF 时展示 DiffusionGemma 去噪预览（每一步替换整条 assistant 消息，最终再发出定稿）
- 向后兼容的队列状态事件（实际并发由推理引擎处理）
- 消息编辑和删除，支持从对话中任意位置重新生成
- 自由滚动：在生成过程中可向上滚动查看历史消息；只要重新滚回底部，新内容会继续自动跟随

使用 `--model` 选择要托管的 GGUF 文件，使用 `--mmproj` 选择要托管的投影器文件。`TensorSharp.Server` 不再扫描 `MODEL_DIR`。

**服务命令行参数：**

不带任何参数运行 `TensorSharp.Server` 会打印完整的参数说明（每个参数的描述、默认值和示例）后退出；`--help` 效果相同。推理必须在启动时传入 `--model`。其他参数可以启动无模型的状态进程，但 `/api/models/load` 不能选择启动时未提供的 GGUF。

| 参数 | 说明 |
|---|---|
| `--model <path>` | 需要托管的 GGUF 文件（推理时必填；如传入了其他参数但未指定该项，服务仍可启动，但 `/api/models/load` 会报告未加载模型） |
| `--mmproj <path>` | 多模态投影器 GGUF（仅给文件名时按模型目录解析；传 `none` 可显式禁用）。需要先指定 `--model`。 |
| `--backend <type>` | 默认计算后端：`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan` |
| `--gpu-device <N>` | `ggml_vulkan` 后端使用的 Vulkan 设备索引，用于多 GPU 主机（例如同时装有 Intel 集成显卡和 NVIDIA 独立显卡的机器）。默认使用设备 0；可用 `--list-gpus` 查看索引。也可通过环境变量 `TS_GGML_VULKAN_DEVICE` 设置。 |
| `--list-gpus` | 列出 ggml-vulkan 可见的 Vulkan 设备（索引 + 显卡名称）后退出 |
| `--help` | 打印参数说明后退出（不带任何参数启动服务时也会显示） |
| `--max-tokens <N>` | 当请求未携带 max-tokens 时使用的默认上限（默认：`20000`） |
| `--temperature <f>` | 当请求未提供时使用的默认采样温度（`0` = 贪心） |
| `--top-k <N>` | 当请求未提供时使用的默认 Top-K 过滤（`0` = 关闭） |
| `--top-p <f>` | 当请求未提供时使用的默认 Nucleus 采样阈值（`1.0` = 关闭） |
| `--min-p <f>` | 当请求未提供时使用的默认 min-p 过滤（`0` = 关闭） |
| `--repeat-penalty <f>` | 当请求未提供时使用的默认重复惩罚（`1.0` = 无） |
| `--presence-penalty <f>` | 当请求未提供时使用的默认存在惩罚（`0` = 关闭） |
| `--frequency-penalty <f>` | 当请求未提供时使用的默认频率惩罚（`0` = 关闭） |
| `--seed <N>` | 当请求未提供时使用的默认随机种子（`-1` = 非确定性） |
| `--stop <string>` | 默认停止序列（可重复指定）。请求体里的 `stop`/`stop_sequences` 会**完全替换**默认列表，而不是与之合并。 |
| `--kv-cache-dtype <type>` | 托管模型的 KV 缓存精度：`f32`、`f16`、`q8_0` 或 `q4_0`（量化缓存以微小数值漂移换取内存节省；各档位的取舍见上文 CLI 参数表）。默认：自动 —— 由后端 / 模型决定。环境变量：`KV_CACHE_DTYPE`。 |
| `--continuous-batching` / `--no-continuous-batching` | 启用（默认）或关闭迭代级分页批处理。启用时服务会在批内动态加入 / 抢占序列，并在实现了 `IBatchedPagedModel` 的模型上将多个序列打包到一次前向中执行。`--no-continuous-batching` 会让所有模型回退到按序列 KV 交换。别名：`--paged-batching` / `--no-paged-batching`。 |
| `--prefill-chunk-size <N>` | 存在竞争时的分块 prefill 粒度 —— 有其他请求同时运行时，每个调度步最多处理的 prefill token 数；块越小，并行 decode 请求越容易频繁轮到 GPU（默认：`1024`）。环境变量：`TS_SCHED_PREFILL_CHUNK`。 |
| `--mtp-spec` / `--no-mtp-spec` | 在带有多 token 预测草稿头的模型上启用 NextN/MTP 投机解码（默认关闭）。草稿头可以是 Qwen 3.6 内嵌的 NextN 块，或通过 `--mtp-draft-model` 加载的 Gemma 4 `gemma4-assistant` 草稿。仅对单序列（无并发）请求生效：草稿头每步最多提议 `--mtp-draft` 个 token，主干网络用一次批量前向完成验证；起草与验证均由该请求自己的采样器（含惩罚项）驱动，输出与标准 decode 一致。仅在有收益处自动启用（ggml 后端与纯 C# `cuda` 后端）；CPU / MLX 走标准 decode。环境变量：`TS_MTP_SPEC`。 |
| `--mtp-draft <N>` | 每个投机步最多起草的 token 数（默认 `8`）。环境变量：`TS_MTP_DRAFT`。 |
| `--mtp-pmin <f>` | 草稿 token 被保留所需的最低置信度，取值 `(0, 1]`；遇到第一个低置信 token 即停止起草（默认 `0.75`）。环境变量：`TS_MTP_PMIN`。 |
| `--mtp-draft-model <path>` | 对于草稿头作为独立文件发布的架构（Gemma 4 的 `gemma4-assistant`），指定其草稿 GGUF 路径。草稿的隐藏维度必须与目标一致（例如 12B 目标配 12B 草稿，而非 26B-A4B 草稿）；草稿不匹配或不完整会在启动时立即失败并给出修复提示。Qwen 3.6 将 NextN 块内嵌在主干 GGUF 中，此参数对其无效。环境变量：`TS_MTP_DRAFT_MODEL`。 |
| `--paged-kv` / `--no-paged-kv` | 已移除的按会话分页 KV 管理器的兼容参数。当前服务端 KV 状态由引擎持有；请使用连续批处理 / `TS_SCHED_*` 开关调节引擎。别名：`--paged-kv-cache` / `--no-paged-kv-cache`。 |
| `--paged-kv-block-size <N>` | 旧的独立分页 KV 块大小。当前引擎使用 `TS_SCHED_BLOCK_SIZE`。 |
| `--paged-kv-ram-mb <N>` | 旧的独立分页 KV RAM 层上限。 |
| `--paged-kv-ssd-dir <dir>` | 旧的独立分页 KV SSD 冷层目录。 |
| `--paged-kv-ssd-mb <N>` | 旧的独立分页 KV SSD 上限。 |
| `--paged-kv-quant-bits <0\|4\|8>` | 服务端接受的旧式独立分页 KV 块量化（`4`/`8` = 对称）。运行时环境变量还接受仿射 min+scale 的 `2`，CLI 则接受 `0\|2\|4\|8`。 |

请求 JSON 中的字段（如 `temperature`、`top_p`、`top_k`、`min_p`、
`repeat_penalty`、`presence_penalty`、`frequency_penalty`、`seed`、
`stop`/`stop_sequences`）始终优先于上述服务端默认值；这些默认值仅
用于填充客户端未指定的字段。

**运行时环境变量：**

| 变量 | 说明 |
|---|---|
| `BACKEND` | 未传 `--backend` 时使用的默认计算后端（`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan`；默认：macOS 为 `ggml_metal`，其他平台为 `ggml_cpu`） |
| `MAX_TOKENS` | 当 `--max-tokens` 与请求级上限均未指定时使用的默认生成长度（默认：`20000`） |
| `VIDEO_SAMPLE_FPS` | 视频提示词每秒抽取的帧数；基于时间的抽帧（默认：`1`） |
| `VIDEO_MAX_FRAMES` | 抽取视频帧数量的可选上限（超出时均匀降采样）；未设置或为 `0` 表示不限制（默认：不限制） |
| `PORT` / `ASPNETCORE_URLS` | 当前会被 `Program.cs` 中固定的 `http://0.0.0.0:5000` 监听地址覆盖；Docker Space 镜像会在构建时用 `APP_PORT` 改写该常量。 |
| `TENSORSHARP_TEMPERATURE` | `--temperature` 与请求体均未指定时的默认采样温度 |
| `TENSORSHARP_TOP_K` | `--top-k` 与请求体均未指定时的默认 Top-K |
| `TENSORSHARP_TOP_P` | `--top-p` 与请求体均未指定时的默认 Top-P |
| `TENSORSHARP_MIN_P` | `--min-p` 与请求体均未指定时的默认 min-P |
| `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` 与请求体均未指定时的默认重复惩罚 |
| `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` 与请求体均未指定时的默认存在惩罚 |
| `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` 与请求体均未指定时的默认频率惩罚 |
| `TENSORSHARP_SEED` | `--seed` 与请求体均未指定时的默认随机种子 |
| `TENSORSHARP_LOG_LEVEL` | 控制台与文件日志的最低输出级别：`Trace`、`Debug`、`Information`、`Warning`、`Error`、`Critical`（默认：`Information`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_DIR` | JSON-line 文件日志的写入目录（默认：`<binDir>/logs`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_FILE` | 设为 `0` 可关闭文件日志，仅保留控制台输出（默认：开启）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_TP_DEGREE` | 张量并行度 —— 把模型切分到本机多少张 CUDA GPU 上（默认：`1`）。`TensorSharp.Server` 使用该变量（服务端没有 `--tp` 命令行参数），同时也是 `ModelBase.Create` 的兜底来源。需要 `--backend cuda`。 |
| `TENSORSHARP_TP_NODE_ID` | 多节点分布式张量并行中本节点的 0 起始编号。必须与 `TENSORSHARP_TP_PEERS` 一起设置。 |
| `TENSORSHARP_TP_PEERS` | 分布式 TP 集群中所有节点的 `host:port` 列表（逗号分隔，例如 `192.168.1.10:9500,192.168.1.11:9500`）。必须与 `TENSORSHARP_TP_NODE_ID` 一起设置。 |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | 各节点向 peer 发起连接的重试窗口（默认：`120` 秒）。若节点由人工或较慢的编排系统间隔较久启动，可调大该值。 |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | 从 peer 阻塞读取的单次接收超时（默认：`300` 秒）。有了它，卡住的 peer 会让集合通信直接失败，而不是挂在操作系统的 TCP keepalive（往往 2 小时以上）上。 |
| `TENSORSHARP_TP_DISABLE_P2P` | 设为 `1` 后，所有跨 GPU 传输都经主机内存中转，不使用 CUDA 点对点 DMA。速度更慢，但可复现无 P2P 硬件（A16 vGPU 配置、部分消费级显卡）的代码路径，便于定位 P2P 相关缺陷。 |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | 设为 `1` 后，本地 AllReduce 走主机内存（设备→主机、求和、主机→设备），而非设备到设备路径。这是与已知可靠的多节点归约完全一致的诊断兜底路径。 |
| `TS_KV_CACHE_REDIS_URL` | 共享 KV 缓存层的 Redis 连接串（例如 `localhost:6379`）。设置后，KV 缓存块会持久化到 Redis 以便跨会话复用。CLI：`--redis-url` 或 `--paged-kv-redis-url`。 |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | Redis KV 缓存条目的 TTL（分钟）；`0` = 不过期（默认：`1440`）。CLI：`--paged-kv-redis-ttl`。 |
| `TS_RESPONSES_STORE_REDIS_URL` | OpenAI Responses API 存储的 Redis 连接串。设置后由 `RedisResponsesStore` 取代内存存储。CLI：`--redis-url`。 |
| `DIFFUSION_STEPS` | 服务端 DiffusionGemma 每个 block 的去噪步数（默认：`48`；CLI 对应 `--diffusion-steps`） |
| `DIFFUSION_MAX_BATCH` | Web UI 扩散调度器可批处理的最大并发 DiffusionGemma 请求数（默认：`2`） |

**分页 KV 缓存 & 连续批处理可调参数（进程 / 模型启动时读取）**

下述变量也可以通过 `--paged-kv*` / `--continuous-batching` CLI 参数设置（它们会被翻译为对应的环境变量）：

| 变量 | 说明 |
|---|---|
| `TS_KV_PAGED_CACHE` | 旧的独立 `PagedKvCacheManager` 兼容开关；当前 `TensorSharp.Server` 的请求 KV 状态由引擎持有。CLI 快捷方式是 `--paged-kv` / `--no-paged-kv`。 |
| `TS_KV_BLOCK_SIZE` | 旧的独立分页 KV 块大小。当前引擎使用 `TS_SCHED_BLOCK_SIZE`。 |
| `TS_KV_CACHE_MAX_RAM_MB` | 旧的独立分页 KV RAM 层上限。 |
| `TS_KV_CACHE_SSD_DIR` | 旧的独立分页 KV SSD 冷层目录。 |
| `TS_KV_CACHE_MAX_SSD_MB` | 旧的独立分页 KV SSD 上限。 |
| `TS_KV_PAGED_QUANT_BITS` | 旧的独立分页 KV 块量化位数（`0` = 透传，`2` = 仿射，`4`，或 `8`）。 |
| `TS_SCHED_DISABLE_BATCHED` | `1` 会即使模型实现了 `IBatchedPagedModel`，也强制回退到按序列 KV 交换。CLI 快捷方式是 `--no-continuous-batching`。 |
| `TS_SCHED_MAX_BATCHED_TOKENS` | 调度器每步 token 预算（默认：`4096`）。 |
| `TS_SCHED_MAX_RUNNING_SEQS` | 同时在执行的最大序列数（默认：`16`）。 |
| `TS_SCHED_PREFILL_CHUNK` | 多个请求争用时每步最大 prefill token 数（默认：`1024`）。 |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | SOLO（无争用）prompt 全新部分（start_pos = 0）的 prefill 分块大小——单个无争用请求会以大分块走融合 prefill 路径（默认：`8192`）。 |
| `TS_SCHED_NUM_BLOCKS` | 引擎块池的物理块数（默认：`256`）。 |
| `TS_SCHED_BLOCK_SIZE` | 引擎侧每块的 token 数（默认：`256`）。 |
| `TS_SCHED_PREFIX_CACHE` | `0` 关闭跨请求的块级哈希前缀共享。 |
| `TS_SCHED_DECODE_QUANTUM` | 在允许切换序列前的 token 数（默认与 block size 相同）。 |
| `TS_QWEN35_BATCHED` | 设为 `0` 强制 Qwen 3.5/3.6 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。`--no-continuous-batching` 也会隐式关闭。 |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | 在 Qwen 3.5/3.6 批处理路径中使用原生批处理 GatedDeltaNet 内核。 |
| `TS_GEMMA4_BATCHED` | 设为 `0` 可强制 Gemma 4 走旧的单序列 KV 交换路径（默认走批处理 / 分页）。 |
| `TS_GPTOSS_BATCHED` | 设为 `0` 强制 GPT OSS 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。 |
| `TS_GPTOSS_PAGED_ATTN_MANAGED` | 在 GPT OSS 批处理路径中使用托管 (C#) 的带 sinks 分页注意力内核。 |
| `TS_NEMOTRON_BATCHED` | 设为 `0` 强制 Nemotron-H 走旧的按序列 KV-swap 路径（默认走批处理 / 分页）。 |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | 在 Nemotron-H 批处理路径中使用原生 Mamba2 批处理步骤内核。 |
| `TS_PAGED_ATTN_KERNEL` | `Mistral3Model.BatchedForward` 选择的分页注意力派发内核：`native`（默认）、`tensor`（基于 C# Tensor）或 `managed`（纯 C# 标量）。 |
| `TS_MLX_PIPELINED_DECODE` | 默认 `1`，当请求为贪心采样、没有 stop 序列且模型支持 device-side argmax / 下一 token embedding 查找时，在 MLX 后端启用流水化贪心 decode。设为 `0` 可关闭。仅 CLI。 |
| `TS_MLX_MLOCK_GGUF` | 默认 `1`，通过 `mlock(2)` 把 GGUF mmap 区域钉在物理内存，避免前向之间被换出。设为 `0` 关闭（适用于进程 `memlock` rlimit 太低、或希望让 OS 自行管理分页的情况）。仅 MLX 后端。 |
| `TS_MLX_FUSED_KV_WRITE` | 默认 `1`，使用单次多维 `slice_update` 写入每个 token 的 KV block。设为 `0` 回退到按 head 的循环（A/B 测试 / 隔离回归用）。 |
| `TS_MLX_BATCHED_MOE_DECODE` | 默认 `1`，将 Qwen 3.5/3.6 MoE 解码时每专家的 K 次 dispatch 合并为每种（gate / up / down）一次批处理 dispatch。在显存紧张的机器上可设为 `0` 关闭（可节省堆叠权重 slab 带来的近一倍权重显存占用）。 |
| `TS_MLX_MOE_FUSED_GATE_UP_SILU` | 默认 `1`，把批处理 MoE 解码的 gate matmul + up matmul + SiLUMul 融合到一个 Metal kernel。设为 `0` 用于和旧的 3-dispatch 路径做 A/B 对比。 |
| `TS_MLX_DEVICE_ROUTER` | 默认 `1`，让 MoE router 的 top-K + softmax 留在 device 上，避免每个 MoE 层一次主机同步（在 Qwen3.6-35B-A3B 上约能节省每 token ~60 次同步）。设为 `0` 可关闭；不满足前置条件时会自动回退到 host routing。 |
| `TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB` | 覆盖 MLX 分配器硬上限 / 空闲缓冲池上限 / wired 缓冲上限（兆字节）。默认值会根据宿主机统一内存大小派生。 |
| `TS_MLX_EVAL_EVERY_N_LAYERS` / `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` | 解码时定期触发 `mlx_async_eval` 的层间隔，用于让 GPU 计算和宿主端排队重叠。Gemma 4 通过 `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` 默认每 4 层一次；Qwen 3 / Qwen 3.5 / Nemotron-H 通过 `TS_MLX_EVAL_EVERY_N_LAYERS` 默认每 16 层一次。支持处可设为 `0` 关闭。 |
| `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` | 覆盖 `--backend mlx` 时 `libmlxc` 的搜索路径。 |

**MTP / 投机解码调优变量**

这些变量控制可选的多 token 预测投机解码路径（见 [MTP / NextN 投机解码](FEATURES_zh-cn.md#mtp--nextn-投机解码)）。`TS_MTP_*` 为通用开关（也可由 `--mtp-*` CLI 参数设置）；`TS_GMTP_*` 为 Gemma 4 草稿路径 A/B 开关。

| 变量 | 说明 |
|---|---|
| `TS_MTP_SPEC` | `1` 为单序列启用 MTP/NextN 投机解码（默认 `0`）。CLI：`--mtp-spec` / `--no-mtp-spec`。 |
| `TS_MTP_DRAFT` | 每个投机步最多起草的 token 数（默认 `8`）。CLI：`--mtp-draft`。 |
| `TS_MTP_PMIN` | 草稿 token 被保留所需的最低置信度，取值 `(0, 1]`（默认 `0.75`）。CLI：`--mtp-pmin`。 |
| `TS_MTP_DRAFT_MODEL` | Gemma 4 独立 `gemma4-assistant` 草稿 GGUF 路径。CLI：`--mtp-draft-model`。Qwen 3.6（内嵌 NextN）忽略此项。 |
| `TS_GMTP_NO_FUSED` | `1` 关闭 Gemma 4 融合多 token 验证 / 草稿步 GGML 内核，回退到逐算子路径（ggml 后端上的 A/B 测试）。 |
| `TS_GMTP_NO_FAST_ROLLBACK` | `1` 恢复保留前缀的回滚路径，而非部分接受时使用的稠密精确匹配快速回滚。 |
| `TS_GMTP_BATCHED_TRUNK` | `1` 让 Gemma 4 验证主干走批量分页路径；默认对单序列投机使用更快的线性主干。 |

**DiffusionGemma 专属调优变量**

| 变量 | 说明 |
|---|---|
| `DIFFUSION_NO_SC` | 设为 `1` 关闭 self-conditioning。默认开启。 |
| `DIFFUSION_SC_TOPK` | 实验用 self-conditioning top-K 截断（默认：`32`）。 |
| `DIFFUSION_NO_PKV` | 设为 `1` 关闭 device-glue 后端上的 prompt-KV 缓存。支持处默认开启。 |
| `DIFFUSION_NO_FUSED_DECODE` | 设为 `1` 关闭 GGML 融合整模型 diffusion decode，回退到逐算子 / 逐层 diffusion decode。 |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | 设为 `1` 关闭融合 output-norm + lm-head + softcap 尾部。 |
| `DIFFUSION_BATCHED_FORWARD` | 设为 `1` 后，对活跃 diffusion canvas 使用真正的 `DecodeCanvasBatched`；默认按请求时间片执行更快的融合单 canvas 路径。 |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | diffusion lm-head logits 批处理内存上限，超过后回退到按序列 lm-head（默认：`300`）。 |

采样参数的优先级（从高到低）：

1. API 请求 JSON 中的字段（如 `temperature`、`top_p`、`stop`）。
2. 服务端命令行参数（如 `--temperature`、`--top-p`、`--stop`）。
3. 上面列出的 `TENSORSHARP_*` 环境变量。
4. `SamplingConfig` 内置默认值（`temperature=1.0`、`top_k=0`、`top_p=1.0`、`min_p=0`、`repeat_penalty=1.0`、存在/频率惩罚均为 `0`、`seed=-1`、无停止序列）。

## 张量并行与分布式推理

TensorSharp 支持**张量并行（TP）**——按 Megatron-LM 列/行并行范式把单个模型切分
到多张 GPU 上——以及**分布式（多节点）张量并行**，让 TP 跨越多台通过 TCP 点对点
网络互联的机器。

### 本地张量并行（单进程，多 GPU）

在一个进程内把模型切分到 N 张 CUDA GPU 上。每张 GPU 持有 `1/N` 的切分权重（列
并行的 QKV / gate / up，行并行的 output / down），外加一份完整的复制权重（归一
化层、词嵌入、LM head）。各 GPU 的 KV 缓存彼此独立；每次行并行投影之后由
AllReduce（CUDA P2P 拷贝 + 逐元素加法内核）把隐藏状态重新汇聚。

```bash
# CLI：2 GPU 张量并行
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2

# 服务端：通过环境变量（服务端没有 --tp 命令行参数）
TENSORSHARP_TP_DEGREE=2 dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
    --model <model.gguf> --backend cuda

# 配置文件 JSON
{ "tp": 2, "backend": "cuda", "model": "<model.gguf>" }
```

### 分布式张量并行（多节点，点对点 TCP）

把 TP 扩展到多台机器。每个节点运行自己的进程、管理自己的本地 GPU；节点之间通过
带长度前缀的分帧协议在 TCP 网格上通信。AllReduce 是分层的：先在节点内做本地
P2P 归约，再由各节点代表之间走 TCP，最后广播回来——每次集合通信只有 `1/tp_local`
的数据需要穿过网络。

```bash
# 2 节点 × 每节点 2 GPU（共 4 GPU）
# 节点 0：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# 节点 1：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# 服务端：通过环境变量
# 节点 0：
TENSORSHARP_TP_DEGREE=2 TENSORSHARP_TP_NODE_ID=0 \
TENSORSHARP_TP_PEERS=192.168.1.10:9500,192.168.1.11:9500 \
    dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda

# 节点 1：
TENSORSHARP_TP_DEGREE=2 TENSORSHARP_TP_NODE_ID=1 \
TENSORSHARP_TP_PEERS=192.168.1.10:9500,192.168.1.11:9500 \
    dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda

# 配置文件 JSON（每个节点一份）
{ "tp": 2, "tp-node-id": 0, "tp-peers": "192.168.1.10:9500,192.168.1.11:9500", "backend": "cuda" }
```

所有节点必须使用完全相同的 `--tp-peers` 列表（或 `TENSORSHARP_TP_PEERS` 环境变
量），并各自使用唯一的 `--tp-node-id`（或 `TENSORSHARP_TP_NODE_ID`）。示例中的端
口（`9500`）不是默认值——必须显式指定，且在所有节点之间可达。

### 支持的架构

| 架构 | TP 状态 | 说明 |
|---|---|---|
| Qwen 3 | ✅ | 参考实现 |
| Mistral 3 | ✅ | 融合 / 分离 QKV，YaRN RoPE |
| Gemma 3 | ✅ | 分离 Q/K/V，GELU，滑动窗口 |
| Gemma 4 | ✅ | 稠密 + MoE 专家切分，逐层 head 维度 |
| Qwen 3.5 / 3.6 family | ✅ | GatedDeltaNet SSM 按 rank 划分 V-head 归属；直连 CUDA 与 GGML 均有打包 GDN 内核，GGML 上另有专家并行 MoE 与列并行 LM head |
| GPT OSS | ✅ | MoE 专家切分，attention sink，YaRN |
| Nemotron-H | ✅ | Mamba2 在 rank 0 上复制计算，MoE 专家切分 |
| DiffusionGemma | — | 不适用（扩散模型） |
| Qwen-Image-Edit | — | 不适用（图像生成） |

### 约束

- **仅支持 CUDA 后端**（`--backend cuda`）。GGML、MLX、Vulkan 在设计上都是单设备。
- `numHeads`、`numKVHeads` 与 `intermediateSize` 必须能被 TP 度整除。
- 量化权重的行并行切分要求 `ne0` 能被 `tp × blockSize` 整除。
- TP 下的批处理 / 连续批处理前向目前实现于 Qwen 3 与 Mistral 3；MoE 模型（Gemma 4、Qwen 3.5/3.6、GPT OSS、Nemotron-H）在 TP 下回退到按序列前向。

### 集群调优与诊断

本地 AllReduce 优先使用 CUDA 点对点（P2P）DMA。启动时，并行组会为每一对报告支持
P2P 的设备启用 peer access，随后做一次往返自检：部分拓扑（挂在某些 PCIe 交换机
后的 L4、开启 IOMMU 的主机、BAR1 窗口过小）虽然声称支持 P2P，却会静默传输损坏数
据——任何未通过自检的设备对都会被永久降级为主机中转。完全不支持 peer access 的
硬件（A16 vGPU 配置、大多数消费级显卡）则一开始就走主机内存中转。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `TENSORSHARP_TP_DISABLE_P2P=1` | 关闭 | 所有跨 GPU 传输一律经主机内存中转，与无 P2P 硬件的路径完全一致。用于判断某个多 GPU 缺陷是否出在 P2P DMA 路径上。 |
| `TENSORSHARP_TP_HOST_ALLREDUCE=1` | 关闭 | 本地 AllReduce 改为设备→主机、CPU 求和、主机→设备。更慢，但与多节点归约路径完全一致，便于隔离 P2P 正确性问题。 |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | `120` | 节点向 peer 重试连接的时长。节点通常由人工间隔数秒到数分钟启动，peer 的监听端可能尚未就绪；编排系统较慢时可调大。 |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | `300` | peer 套接字的单次接收超时。否则卡住的 peer 会一直阻塞到操作系统 TCP keepalive 超时（往往 2 小时以上），而不是让集合通信失败。 |

启动日志会明确给出拓扑：本地并行组打印
`Tensor parallelism: N GPUs (<设备名>)`，P2P 降级会打印 `TP: P2P disabled…` 或自
检告警，每个分布式节点在网格建立完成后打印
`[TcpCommunicator] Rank r/N connected to all peers.`。

### Redis 支撑的共享状态

服务端可以选择把 KV 缓存块与 OpenAI Responses API 存储持久化到 Redis，从而实现跨
会话的 KV 复用与持久化的响应存储：

```bash
# 同时为 KV 缓存与 Responses API 启用 Redis
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda \
    --redis-url localhost:6379

# 仅启用 KV 缓存层，TTL 设为 12 小时
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda \
    --paged-kv-redis-url localhost:6379 --paged-kv-redis-ttl 720
```

## 功能 × 环境变量矩阵

每个主要功能由哪些环境变量（以及对应的 CLI 参数）控制的速查矩阵。**加粗**的变量是该功能的开关；其余的是该功能默认启用后的调优参数。

#### 连续批处理 & 分页 KV 缓存

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 连续批处理引擎（`InferenceEngine` + 调度器） | 在 `TensorSharp.Server` 中默认启用 | `TS_SCHED_DISABLE_BATCHED=1` 强制按序列回退 | `--no-continuous-batching` / `--continuous-batching` |
| 旧的按会话分页 KV 管理器 | 已从服务端请求路径移除 | `TS_KV_PAGED_CACHE`（`0` / `1`）、`TS_KV_BLOCK_SIZE` 仅为兼容 / 独立测试保留 | `--paged-kv` / `--no-paged-kv`、`--paged-kv-block-size N` |
| 旧的分页 KV SSD 冷层溢出 | 关闭 | `TS_KV_CACHE_MAX_RAM_MB`、`TS_KV_CACHE_SSD_DIR`、`TS_KV_CACHE_MAX_SSD_MB` | `--paged-kv-ram-mb`、`--paged-kv-ssd-dir`、`--paged-kv-ssd-mb` |
| 旧的分页 KV 块量化（TurboQuantKvCodec） | 关闭（`0` = 透传） | `TS_KV_PAGED_QUANT_BITS`（`0` / `2` / `4` / `8`） | `--paged-kv-quant-bits` |
| 跨请求的块级哈希前缀共享 | 启用 | `TS_SCHED_PREFIX_CACHE=0` 关闭 | — |
| 调度器调优（每步 token 预算、最大同时序列数、prefill 分块、块池大小、decode quantum） | 引擎默认 | `TS_SCHED_MAX_BATCHED_TOKENS`、`TS_SCHED_MAX_RUNNING_SEQS`、`TS_SCHED_PREFILL_CHUNK`、`TS_SCHED_SOLO_PREFILL_CHUNK`、`TS_SCHED_NUM_BLOCKS`、`TS_SCHED_BLOCK_SIZE`、`TS_SCHED_DECODE_QUANTUM` | — |

#### 按模型的批处理 / 分页前向（`IBatchedPagedModel.ForwardBatch`）

| 模型 | 默认状态 | 切换默认的环境变量 | 原生内核子开关 |
|---|---|---|---|
| Mistral 3 | 启用 | — | `TS_PAGED_ATTN_KERNEL` = `native`（默认）/ `tensor` / `managed` |
| Gemma 4 | 启用 | `TS_GEMMA4_BATCHED=0` 强制走旧的按序列路径 | — |
| Qwen 3 | 启用（参考移植） | — | — |
| Qwen 3.5 / 3.6 系列 | 启用 | `TS_QWEN35_BATCHED=0` 强制走旧的按序列路径（或 `--no-continuous-batching`） | `TS_QWEN35_BATCHED_GDN_NATIVE=1` 启用原生批处理 GDN 内核；`FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖融合注意力启用阈值（默认 4096） |
| GPT OSS | 启用 | `TS_GPTOSS_BATCHED=0` 强制走旧的按序列路径 | `TS_GPTOSS_PAGED_ATTN_MANAGED=1` 强制使用托管 (C#) sinks softmax，而非原生带 sinks 的分页注意力内核 |
| Nemotron-H | 启用 | `TS_NEMOTRON_BATCHED=0` 强制走旧的按序列路径 | `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` 启用原生批处理 Mamba2 步（NEON SIMD + GCD 并行） |
| Gemma 3 | 未实现（走按序列回退） | — | — |
| DiffusionGemma | Web UI 路径使用独立 diffusion 调度器；不是 `IBatchedPagedModel` 自回归路径 | `DIFFUSION_MAX_BATCH`、`DIFFUSION_STEPS` | `DIFFUSION_BATCHED_FORWARD=1` 启用真正的批处理 canvas decode；GGML 融合 decode 默认开启，可用 `DIFFUSION_NO_FUSED_DECODE=1` 关闭 |

#### MTP / NextN 投机解码

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 投机解码引擎（单序列） | 关闭 | **`TS_MTP_SPEC=1`** | `--mtp-spec` / `--no-mtp-spec` |
| 每步最多起草 token 数 | `8` | `TS_MTP_DRAFT` | `--mtp-draft N` |
| 草稿 token 被保留所需最低置信度 | `0.75` | `TS_MTP_PMIN` | `--mtp-pmin X` |
| Gemma 4 独立草稿 GGUF（`gemma4-assistant`） | 无 | `TS_MTP_DRAFT_MODEL` | `--mtp-draft-model <path>` |
| Gemma 4 融合验证 / 草稿内核（ggml） | 开启 | `TS_GMTP_NO_FUSED=1` 回退到逐算子 | — |
| Gemma 4 部分接受时的稠密快速回滚 | 开启 | `TS_GMTP_NO_FAST_ROLLBACK=1` 恢复保留前缀回滚 | — |
| Gemma 4 验证主干路径 | 线性（单序列） | `TS_GMTP_BATCHED_TRUNK=1` 走批量分页主干 | — |

#### 张量并行与分布式推理

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 本地张量并行（单进程，多 GPU） | 关闭（`1` 张 GPU） | **`TENSORSHARP_TP_DEGREE=N`** | `--tp N`（仅 CLI） |
| 分布式 TP 节点编号（多节点） | 未设置（关闭） | **`TENSORSHARP_TP_NODE_ID=N`** | `--tp-node-id N`（仅 CLI） |
| 分布式 TP peer 端点 | 未设置（关闭） | **`TENSORSHARP_TP_PEERS=host1:port1,host2:port2`** | `--tp-peers host1:port1,host2:port2`（仅 CLI） |
| peer 连接重试窗口（多节点） | `120` 秒 | `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | — |
| 单次接收超时（多节点） | `300` 秒 | `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | — |
| 强制跨 GPU 拷贝走主机中转 | 关闭（可用时使用 P2P） | `TENSORSHARP_TP_DISABLE_P2P=1` | — |
| 强制本地 AllReduce 走主机中转 | 关闭（设备到设备） | `TENSORSHARP_TP_HOST_ALLREDUCE=1` | — |

#### Redis 共享状态（仅服务端）

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| Redis KV 缓存层 | 关闭 | **`TS_KV_CACHE_REDIS_URL`** | `--redis-url` 或 `--paged-kv-redis-url` |
| Redis KV 缓存条目 TTL | `1440` 分钟（24 小时） | `TS_KV_CACHE_REDIS_TTL_MINUTES`（`0` = 不过期） | `--paged-kv-redis-ttl` |
| Redis Responses API 存储 | 关闭（内存） | **`TS_RESPONSES_STORE_REDIS_URL`** | `--redis-url` |

#### 后端

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 默认计算后端 | `ggml_metal`（macOS）、`ggml_cpu`（Windows/Linux） | `BACKEND` | `--backend` |
| MLX 后端库查找 | 优先探测应用目录 | `TENSORSHARP_MLX_LIBRARY`（`libmlxc` 完整路径）、`TENSORSHARP_MLX_LIBRARY_DIR`（目录） | — |
| MLX 流水化贪心 decode（仅 CLI） | 满足条件时启用 | `TS_MLX_PIPELINED_DECODE=0` 关闭 | — |
| 使用 `mlock(2)` 钉住 GGUF mmap，使权重常驻 | 启用 | `TS_MLX_MLOCK_GGUF=0` 关闭 | — |
| MLX 融合多维 KV 写入（每个 cache block 单次 `slice_update`） | 启用 | `TS_MLX_FUSED_KV_WRITE=0` 回退到按 head 循环 | — |
| MLX 批处理 MoE 解码（Qwen 3.5/3.6 MoE） | 启用 | `TS_MLX_BATCHED_MOE_DECODE=0` 走旧的按专家路径 | — |
| MLX MoE gate+up+SiLUMul 融合 Metal kernel | 启用 | `TS_MLX_MOE_FUSED_GATE_UP_SILU=0` 走旧的 3-dispatch | — |
| MLX 设备端 MoE router top-K + softmax | 满足条件时启用 | `TS_MLX_DEVICE_ROUTER=0` 关闭 | — |
| MLX 解码层边界 `async_eval` 间隔 | Gemma 4：每 4 层；Qwen / Nemotron：每 16 层 | `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS=N` 或 `TS_MLX_EVAL_EVERY_N_LAYERS=N`（支持处 `0` = 关闭） | — |
| MLX 分配器上限（内存 / 缓存 / wired buffer） | 按宿主机派生 | `TS_MLX_MEMORY_LIMIT_MB`、`TS_MLX_CACHE_LIMIT_MB`、`TS_MLX_WIRED_LIMIT_MB` | — |

#### 采样默认值（仅服务端）

这些变量用于填充请求体未提供的字段；请求 JSON 字段始终优先于 CLI 参数，CLI 参数优先于环境变量。

| 采样字段 | 环境变量 | CLI 等价参数 |
|---|---|---|
| `temperature` | `TENSORSHARP_TEMPERATURE` | `--temperature` |
| `top_k` | `TENSORSHARP_TOP_K` | `--top-k` |
| `top_p` | `TENSORSHARP_TOP_P` | `--top-p` |
| `min_p` | `TENSORSHARP_MIN_P` | `--min-p` |
| `repeat_penalty` | `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` |
| `presence_penalty` | `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` |
| `frequency_penalty` | `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` |
| `seed` | `TENSORSHARP_SEED` | `--seed` |
| 最大 token 数 | `MAX_TOKENS` | `--max-tokens` |
| 停止序列 | —（仅 CLI / 请求体支持） | `--stop`（可重复） |

#### 服务托管与上传（仅服务端）

| 功能 | 默认 | 环境变量 |
|---|---|---|
| ASP.NET Core 监听 | `http://0.0.0.0:5000` | 固定在 `Program.cs`；Docker Space 镜像用 `APP_PORT` 构建参数改写 |
| 文本及原生数字 PDF 上传 | 保留全部提取内容；最终渲染的提示词必须能放入已加载模型的上下文 | — |
| 视频帧抽取 | 1 fps（基于时间，不限制） | `VIDEO_SAMPLE_FPS`、`VIDEO_MAX_FRAMES` |
| DiffusionGemma Web UI 去噪 | 48 步，最大 batch 2 | `DIFFUSION_STEPS`、`DIFFUSION_MAX_BATCH` |

#### 日志（服务端 + CLI）

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 控制台 + 文件日志最低级别 | `Information` | `TENSORSHARP_LOG_LEVEL` | `--log-level` |
| 文件日志输出目录 | `<binDir>/logs` | `TENSORSHARP_LOG_DIR` | `--log-dir` |
| 文件日志开关 | 启用 | `TENSORSHARP_LOG_FILE=0` 关闭 | `--log-file 0\|1` |
| 控制台日志开关 | 启用 | — | `--log-console 0\|1`（仅 CLI） |

#### 原生构建（仅编译期）

下列变量由 `build-linux.sh` / `build-windows.ps1` / `dotnet build` 时自动构建 `TensorSharp.GGML.Native` 的脚本读取，不在运行时生效。

| 功能 | 默认 | 环境变量 | 构建脚本参数 |
|---|---|---|---|
| 在原生构建中启用 GGML CUDA | 根据工具链自动检测 | `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` | `--cuda` / `--no-cuda` |
| 在原生构建中启用 GGML Vulkan | 根据已安装的 Vulkan 运行时自动检测；未安装 Vulkan SDK / 开发包时自动下载便携工具链（headers、glslc、SPIRV-Headers） | `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON/OFF` | `--vulkan` / `--no-vulkan` |
| 精简 `CMAKE_CUDA_ARCHITECTURES` 列表 | 根据可见 GPU 自动检测 | `TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES` | `--cuda-arch='86-real;89-real'` |
| 原生构建并行度上限 | 使用全部 CPU，并按内存容量限制（`nvcc` 每任务约 3 GB） | `TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL` | — |
| 原生构建使用的 CMake 生成器（Windows） | 有 Ninja 时优先使用，否则用 `Visual Studio NN` | `CMAKE_GENERATOR` | `-G <生成器>` |
| 原生构建使用的 Visual Studio 安装（Windows） | 自动检测，包含被标记为"不完整"的安装 | `TENSORSHARP_VS_INSTALL_DIR` | — |

## 服务端日志

每一轮 chat / generate 请求开始与结束时，服务都会打出一条结构化的
Information 级别日志。只需对日志文件做一次 grep，即可获得紧凑的请求—响应审计
轨迹，无需重放任何流量。

| 事件 ID | 触发位置 | 字段 |
|---|---|---|
| `ChatStarted`（1500） | `chat.start`、`generate.start` 以及各协议的请求横幅 | 采样配置、消息与附件计数、`userInput=`（最近一条用户消息的有界预览），以及 `fullInput=`（本轮全部消息的 JSON 数组，包含附件路径、原始字符数，正文最多 512 个字符）。内联上传文档会替换为省略标记，但保留末尾用户指令；`/api/generate` 同样仅记录有界 prompt 预览。 |
| `ChatCompleted`（1502） | `chat.complete`、`generate.complete` | token 数、KV 缓存复用（`kvReused`、`kvReusePercent`）、TTFT、耗时、吞吐、终止原因，以及完整的模型原始输出（思维链 + 结果） |
| `ChatAborted`（1503） | 客户端中途断开 | 已生成的部分输出、当时的 KV 复用占比 |
| `KvCacheReusePlan`（1510） | 每次前缀复用判断 | `Debug` 级的细粒度分支信息（精确匹配 / 部分复用 / 完整重置） |
| `HttpRequestStarted/Completed`（1100/1101） | 每个 HTTP 请求 | method、path、远端 IP、状态码、耗时；`/api/queue/status` 被降级到 `Debug`，避免 UI 高频轮询淹没每轮日志 |

模型原始输出会保留 `<think>...</think>`、`<|channel|>analysis` 等内联标记，因
此完成日志可以同时看到推理过程与最终用户可见结果。输入正文会刻意限制长度：
无损上传的文档不会再被完整复制到日志中，从而避免额外内存/IO 峰值和意外泄露
文档内容。上传清单与 `contentChars` 仍保留可用的审计元数据；如需逐字节重放，
请另行捕获请求。设置 `TENSORSHARP_LOG_LEVEL=Warning` 可关闭逐轮信息日志。

`fullInput` 字段示例（为便于阅读做了缩进，实际日志为单行）：

```json
[
  {"role":"system","content":"你是一个有帮助的助手。","contentChars":11},
  {"role":"user","content":"世界上最高的山是哪座？","contentChars":11},
  {"role":"assistant","content":"珠穆朗玛峰。","contentChars":6},
  {"role":"user","content":"它有多高？","contentChars":5,"images":["/uploads/mountain.jpg"]}
]
```

同样的 KV 缓存复用统计会通过所有 API 透出：

- **Web UI SSE**（`POST /api/chat`） —— `done` 事件携带 `promptTokens`、`kvReusedTokens`、`kvReusePercent`。
- **Ollama NDJSON**（`POST /api/generate`、`POST /api/chat/ollama`） —— 流式末尾 chunk 与非流式响应均携带 `prompt_cache_hit_tokens`（int）和 `prompt_cache_hit_ratio`（0..1）。
- **OpenAI**（`POST /v1/chat/completions`） —— `usage` 块携带 `prompt_tokens_details.cached_tokens`，与 OpenAI 标准扩展一致，现有 SDK 可直接读取。

Web UI 中每条助手消息下方的统计行也会展示命中率（例如 `187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)`）。

## HTTP API

TensorSharp.Server 暴露三种 API 风格。完整文档及 curl/Python 示例见 [API_EXAMPLES.md](TensorSharp.Server/API_EXAMPLES.md)。

**兼容 Ollama 的 API：**

```bash
# 列出模型
curl http://localhost:5000/api/tags

# 文本生成
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# 聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# 启用思维链模式的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "计算 17*23"}], "think": true, "stream": false}'

# 带工具调用的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "天气怎么样？"}], "tools": [{"function": {"name": "get_weather", "description": "获取当前天气", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'
```

**兼容 OpenAI 的 API：**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# 结构化输出（OpenAI response_format）
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "从“Paris, France”中提取城市与国家。"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "confidence": {"type": ["string", "null"]}
          },
          "required": ["city", "country", "confidence"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**OpenAI Python SDK：**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q8_0.gguf",
    messages=[{"role": "user", "content": "What is 2+3?"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**队列状态：**

```bash
curl http://localhost:5000/api/queue/status
# {"busy":false,"pending_requests":0,"total_processed":42}
```

