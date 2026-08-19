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

**DeepSeek V4 Flash 是上表的例外。** 它那套 284B 的压缩稀疏注意力 MoE 结构不走通用的逐算子路径，而是使用三套专属的整模型执行器之一：Direct CUDA 引擎（`--backend cuda`）、原生 ggml 执行器（`--backend ggml_cuda` / `ggml_vulkan`），以及 100% 纯 C# 的 CPU 执行器（`--backend cpu`，直接从内存映射的 GGUF 分片提供量化权重）。三者都会把权重按层切分到所有可见 GPU（CPU 路径则从映射分片流式读取），因此远大于单卡显存的模型依然跑得起来。详见 [DeepSeek V4 卡片](docs/models/deepseek4_zh-cn.md)。



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

不带任何参数运行 `TensorSharp.Cli` 会打印完整的参数参考——逐项列出说明、默认值、
取值范围与示例——并在日志与模型机制启动之前退出；`--help`（以及 `-h`、`-?`、`/?`）
效果相同。

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --help
```

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

# PDF 文档输入：文字型 PDF 会提取文本层并内联进提示词；扫描型 PDF 会转成页面
# 图像，需要视觉模型（--mmproj 或内置视觉编码器）。--input 提供针对文档的指令。
echo "Summarize the key findings of this paper." > question.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --pdf paper.pdf --input question.txt \
    --max-tokens 300 --backend ggml_metal

# DiffusionGemma 文本扩散生成
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <diffusion-gemma.gguf> --input prompt.txt --backend ggml_metal \
    --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0

# Qwen-Image-Edit 图像编辑（提示词 + 输入图像 -> 编辑后的图像）
# VAE + Qwen2.5-VL 文本编码器伴随文件会在 DiT GGUF 旁解析
# （或用 --qwen-image-vae / --qwen-image-vl / --qwen-image-mmproj 指定）。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <qwen-image-edit-DiT.gguf> --image input.png \
    --prompt "Make the sky a dramatic sunset." --output edited.png \
    --backend ggml_cuda --diffusion-steps 30 --cfg 2.5 --diffusion-seed 0

# Wan 视频生成（提示词 -> H.264 MP4）。UMT5-XXL 文本编码器 GGUF 与视频 VAE
# 伴随文件会在 DiT GGUF 旁解析（或用 --wan-te / --wan-vae 指定）。
# Wan 2.1 T2V、Wan 2.2 TI2V-5B 与 Wan 2.2 A14B（两个专家）都会自动识别；
# 步数蒸馏（Turbo / Lightning / FastWan）检查点同样自动识别——同一段视频只需
# 4 次 DiT 前向而不是 100 次。详见下文“视频生成（Wan）”与 docs/models/wan_zh-cn.md。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "a lovely cat walking through a garden" --output cat.mp4 \
    --width 832 --height 480 --video-frames 49 --backend ggml_cuda \
    --diffusion-seed 7
# Wan 2.2 图生视频：--image 提供首帧，提示词控制运动、镜头与场景变化
# （TI2V-5B 或 I2V-A14B 检查点）。
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "the cat runs toward the camera, cinematic tracking shot" \
    --image first_frame.png --output cat_run.mp4 --backend ggml_cuda

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

# 张量并行：在单个进程内把模型切分到 2 张 GPU 上
# （--backend cuda、ggml_cuda 或 ggml_vulkan）
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend cuda --tp 2

# 分布式张量并行：2 节点 × 每节点 2 GPU（共 4 GPU）
# 节点 0：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# 节点 1：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
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
| `--n-cpu-moe <N>` / `-ncmoe <N>` | 把前 N 层的路由 MoE 权重留在系统内存里并在 CPU 上做乘法；注意力、norm、路由器与始终活跃的共享专家仍留在加速器上（等价于 llama.cpp 的 `--n-cpu-moe`）。正是它让 35B-A3B 这类 MoE 能与长上下文 KV 缓存一起塞进 12–16 GB 的显卡。传 `all` 表示所有层。默认：所有架构（含 DeepSeek V4）都是 0 —— 装不下的模型会在加载时被拒绝并告知需要卸载多少层，而不会被悄悄卸载（环境变量 `TS_N_CPU_MOE`）。详见[混合专家 CPU 卸载](#混合专家-cpu-卸载--n-cpu-moe)。 |
| `--cpu-moe` / `-cmoe` | `--n-cpu-moe all` 的简写。默认：关闭（环境变量 `TS_CPU_MOE`）。 |
| `--cpu-moe-threads <N>` | 主机侧专家矩阵乘的工作线程数。默认：在核数多于 8 的主机上取本进程实际可用 CPU 并行度（`hardware_concurrency`，再受调度亲和性掩码与 cgroup CPU 配额约束）的**一半**，低于 8 时取「全部减一」。另一半并非浪费——加速器提交线程，以及 `TensorSharp.Server` 里的 Kestrel 与调度器，同样需要可被调度，而 .NET 自己的线程池是按机器 CPU 数而不是 cgroup 配额来定的。把它设到接近配额是悬崖而不是缓坡：在 95 CPU 配额下，托管的 26B MoE 在 64 线程时实测 20.7 tok/s，71 线程时只剩 8.2。独占机器上可以调高（环境变量 `TS_CPU_MOE_THREADS`）。 |
| `--kv-cache-dtype <type>` | KV 缓存精度：`f32`（默认）、`f16`、`q8_0` 或 `q4_0`。量化 / 半精度 KV 缓存以微小数值漂移换取内存节省；`q4_0`（约 0.56 字节/元素，约为 f32 的 1/7）是最激进的档位，面向 KV 缓存占主导内存的超长（128K–256K）上下文。块量化缓存（`q8_0`/`q4_0`）需要原生 GGML flash 路径。 |
| `--tp <N>` | 张量并行度 —— 在单个进程内把模型切分到 N 张 GPU 上（默认：`1`）。需要 `--backend cuda`、`ggml_cuda` 或 `ggml_vulkan`。详见[张量并行与分布式推理](#张量并行与分布式推理)。 |
| `--tp-node-id <N>` | 多节点分布式张量并行中本节点的 0 起始编号。必须与 `--tp-peers` 一起使用。 |
| `--tp-peers <list>` | 集群中所有节点的 `host:port` 列表（逗号分隔，例如 `192.168.1.10:9500,192.168.1.11:9500`）。所有节点必须使用完全相同的列表。必须与 `--tp-node-id` 一起使用。 |
| `--interactive` / `-i` | 进入交互式 REPL 聊天会话（逐轮输入/输出），支持 KV 缓存复用、斜杠命令、运行时热切换 模型/后端/投影器、文件附件（图像、音频、视频、文本）以及实时调整采样参数。完整命令列表见下文「**交互式 REPL 命令**」一节 |
| `--system <text>` | 用于初始化交互式会话的系统提示词（在 REPL 中可用 `/system` 覆盖） |
| `--system-file <path>` | 从 UTF-8 文本文件读取初始系统提示词（`--system` 的替代写法） |
| `--think` | 启用思维链/推理模式 |
| `--tools <path>` | 包含工具/函数定义的 JSON 文件 |
| `--draft-model <path>` | 投机解码草稿 GGUF，适用于草稿器以独立文件发布的架构——DeepSeek V4 的 DSpark 支持模块（见 [DeepSeek V4](docs/models/deepseek4_zh-cn.md#dspark-投机解码)）与 Muse-Glimmer 的 DFlash 块级草稿器（见 [Muse-Glimmer](docs/models/muse-glimmer_zh-cn.md#3-dflash-投机解码)，环境变量 `TS_MUSE_GLIMMER_DFLASH`）。它每步起草一整块 token，主干用一次批量前向验证，因此贪心输出保持不变。在所有单序列路径（`--input`、`--input-jsonl`、`--multi-turn-jsonl`、`--interactive`）上生效，需要 `--backend cuda` 或 `--backend ggml_cuda`，并且必须是纯 argmax 采样：任何 temperature、top-k/p 或重复/存在/频率惩罚都会将其关闭。环境变量：`TS_DSV4_DSPARK`。 |
| `--spec-draft-n-max <N>` | 每个投机块最多起草的 token 数（默认：草稿器训练时的块大小；DSpark 为 5，Muse-Glimmer 的 DFlash 为 15） |
| `--spec-draft-conf-min <p>` | 保留某个起草位置所需的最小**累积**接受概率（置信度头各位置估计值的乘积，默认 `0.35`）。调低会起草更远、回滚更多；调高则更早退回普通 decode。 |
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
| `--qwen-image-lora <path>` | Qwen-Image-Edit 的 Lightning 蒸馏 LoRA（`.safetensors`）。它以运行期 F32 旁路的形式接在每个目标投影旁（`y = W_quant·x + b + (alpha/rank)·up·(down·x)`），量化基权重原样保留——**不会**被合并进权重。步数从文件名自动推导（例如 4 或 8），并把 CFG 切换为 1.0、时间步 shift 固定为 3，于是默认的 30 步 × 2 次 CFG 前向（60 次 DiT 前向）变成 4–8 次。它需要整模型或融合逐块的 CUDA 前向路径；在没有该旁路的路径上会直接报错而不是输出噪声。环境变量：`TS_QWEN_IMAGE_LORA`。 |
| `--width <px>` / `--height <px>` | Qwen-Image-Edit 与 Wan 视频的输出尺寸。默认 `0` —— 自动（Qwen-Image-Edit：源图尺寸，按 VRAM 钳制；Wan：按输入图的宽高比取模型原生面积，TI2V-5B 为 1280×704，其余为 832×480）。 |
| `--video-frames <N>` | Wan 视频帧数，会对齐到 `4k+1`（默认：33；Wan2.2-TI2V 为 49）。`1` 生成一张静态图（配合 `--output out.png`）。 |
| `--fps <N>` | 保存的 Wan MP4 的播放帧率（默认：16；Wan2.2-TI2V 为 24）。 |
| `--flow-shift <F>` | Wan FlowMatch 时间步 shift（默认：模型官方配方 —— Wan 2.2 为 5.0，A14B T2V 为 12.0，Wan 2.1 为 8.0/3.0/5.0）。 |
| `--sampler <name>` | Wan 采样器：`unipc`（官方采样器，默认）或 `euler`。 |
| `--negative-prompt <text>` | Wan 负向提示词（默认：官方 Wan 负向提示词）。 |
| _（步数蒸馏检查点）_ | 按 DiT 文件名自动识别（`Turbo`、`distill`、`Lightning`、`lightx2v`、`FastWan`、`-dmd`，或显式的 `…-4steps-…` / `…8step…`，N 取 1–16）：管线切换到该步数并关闭引导，把官方 50 步 × CFG 配方的 100 次 DiT 前向变成 4 次。这是 Wan 最大的提速手段——详见下文 **[视频生成（Wan）](#视频生成wan)**。`--diffusion-steps` / `--cfg` 可覆盖它。 |
| `--cfg-cache-stride <N>` | Wan 引导缓存：每 `N` 步只跑一次无条件 CFG 前向，其余步复用缓存的引导方向（默认关闭——每步都跑两次前向）。`2` 约快 1.30×，`3` 约快 1.43×；属于近似，需要严格对齐参考样本时请关闭。 |
| `--wan-vae <path>` | 覆盖解析到的 Wan 视频 VAE（`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`）。环境变量：`TS_WAN_VAE`。 |
| `--wan-te <path>` | 覆盖解析到的 UMT5-XXL 文本编码器 GGUF。环境变量：`TS_WAN_TE`。Wan 2.2 A14B 还会自动解析第二个 high/low-noise 专家（环境变量 `TS_WAN_DIT2`）。 |
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

# Wan 视频生成：当 Web UI 或 API 请求未提供自己的 frames / fps 时，
# 默认以 24 fps 生成 121 帧（约五秒）。
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ./models/Wan2.2-TI2V-5B.gguf --backend ggml_cuda \
    --video-frames 121 --fps 24
# TI2V-5B 原生面积下的 121 帧是 27k 个 DiT token，而自注意力对 token 数是平方级，
# 因此在**基础**检查点上跑完 50 步在笔记本级 GPU 上需要数小时（实测 M5 Pro 约 3 小时
# 30 分）。把 --model 换成步数蒸馏检查点后，同一个请求只需 17 分 30 秒——其他参数
# 一律不变。详见下文“视频生成（Wan）”。
# 服务端会打印每次前向的耗时与滚动 ETA，并每 30 秒发一次心跳，Web UI 两者都会显示；
# 完整成本表见 docs/models/wan_zh-cn.md。

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

在浏览器中打开 `http://localhost:5000` —— 根地址即为聊天界面（`GET /health` 是存活检查接口）。Web 界面支持：

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
| `--tp <N>` | 张量并行度 —— 把托管的模型切分到本机 N 张 GPU 上（默认：`1`）。需要 `--backend cuda`、`ggml_cuda` 或 `ggml_vulkan`。环境变量：`TENSORSHARP_TP_DEGREE`。详见[张量并行与分布式推理](#张量并行与分布式推理)。 |
| `--tp-node-id <N>` | 多节点（分布式）张量并行中本节点的 0 起始编号。服务端只能是节点 `0`（对外提供 HTTP 的 driver）；其余节点请用 `TensorSharp.Cli` 启动。必须与 `--tp-peers` 一起使用。环境变量：`TENSORSHARP_TP_NODE_ID`。 |
| `--tp-peers <list>` | 分布式 TP 集群中所有节点的 `host:port` 列表（逗号分隔，按节点 ID 排序，例如 `192.168.1.10:9500,192.168.1.11:9500`）。必须与 `--tp-node-id` 一起使用。环境变量：`TENSORSHARP_TP_PEERS`。 |
| `--gpu-device <N>` | `ggml_vulkan` 后端使用的 Vulkan 设备索引，用于多 GPU 主机（例如同时装有 Intel 集成显卡和 NVIDIA 独立显卡的机器）。默认使用设备 0；可用 `--list-gpus` 查看索引。也可通过环境变量 `TS_GGML_VULKAN_DEVICE` 设置。 |
| `--list-gpus` | 列出 ggml-vulkan 可见的 Vulkan 设备（索引 + 显卡名称）后退出 |
| `--n-cpu-moe <N>` / `-ncmoe <N>` | 把前 N 层的路由 MoE 权重留在系统内存里、在 CPU 上运行其 FFN（见上文**混合专家 CPU 卸载**）。`all` 表示卸载所有层。默认：所有架构（含 DeepSeek V4）都是 0；装不下的模型会在加载时被拒绝并告知需要卸载多少层。环境变量：`TS_N_CPU_MOE`。 |
| `--cpu-moe` / `-cmoe` | `--n-cpu-moe all` 的简写。默认：关闭。环境变量：`TS_CPU_MOE`。 |
| `--cpu-moe-threads <N>` | 主机侧专家矩阵乘的工作线程数。默认：在核数多于 8 的主机上取可用 CPU 并行度（`hardware_concurrency`，再受亲和性掩码与 cgroup CPU 配额约束）的一半。服务端还需要另一半来跑 Kestrel、调度器与加速器提交线程；把它设到接近配额会让吞吐直接崩塌而不是缓慢下降（95 CPU 配额下 64 线程 20.7 tok/s，71 线程只剩 8.2）。环境变量：`TS_CPU_MOE_THREADS`。 |
| `--help` | 打印参数说明后退出（不带任何参数启动服务时也会显示） |
| `--max-tokens <N>` | 最大生成 token 数：请求未携带上限时用它填充，请求要求更多时按它截断。对所有端点生效（Web UI、`/api/chat`、`/api/generate`、`/v1/chat/completions`、`/v1/responses`）。默认：`20000`，此默认值只用于填充、不做截断。环境变量：`MAX_TOKENS`。 |
| `--video-frames <N>` | Web UI 或 API 请求未提供 `frames` 时，Wan 视频生成使用的默认输出帧数。VAE 会将其对齐到 `4k+1`；以 24 fps 生成 `121` 帧约为五秒。未指定此参数时沿用模型回退值：通常为 `33`，Wan 2.2 TI2V-5B 为 `49`。请求中显式提供的 `frames` 会覆盖此默认值。 |
| `--fps <N>` | Web UI 或 API 请求未提供 `fps` 时，Wan MP4 输出使用的默认播放帧率。未指定此参数时沿用模型回退值：通常为 `16` fps，Wan 2.2 TI2V-5B 为 `24` fps。请求中显式提供的 `fps` 会覆盖此默认值；仅改变 FPS 会改变播放速度，而不会改变生成的帧数。 |
| `--temperature <f>` | 采样温度（`0` = 贪心） |
| `--top-k <N>` | Top-K 过滤（`0` = 关闭） |
| `--top-p <f>` | Nucleus 采样阈值（`1.0` = 关闭） |
| `--min-p <f>` | min-p 过滤（`0` = 关闭） |
| `--repeat-penalty <f>` | 重复惩罚（`1.0` = 无） |
| `--presence-penalty <f>` | 存在惩罚（`0` = 关闭） |
| `--frequency-penalty <f>` | 频率惩罚（`0` = 关闭） |
| `--seed <N>` | 随机种子（`-1` = 非确定性） |
| `--stop <string>` | 停止序列（可重复指定）。在默认的 `--sampling-precedence config` 下，请求体里的 `stop`/`stop_sequences` 会与这里的列表**合并**；在 `request` 下则完全替换。 |
| `--sampling-precedence <config\|request>` | 当请求同时携带了你在上面配置过的采样参数时，以谁为准。`config`（默认）保留你的配置值 —— VS Code Copilot Chat 等客户端会把 `temperature`/`top_p` 硬编码进每一次请求，否则就会静默覆盖你的配置；你**没有**配置过的参数仍然取请求中的值。`request` 恢复“客户端优先”。环境变量：`TENSORSHARP_SAMPLING_PRECEDENCE`。 |
| `--kv-cache-dtype <type>` | 托管模型的 KV 缓存精度：`f32`、`f16`、`q8_0` 或 `q4_0`（量化缓存以微小数值漂移换取内存节省；各档位的取舍见上文 CLI 参数表）。默认：自动 —— 由后端 / 模型决定。环境变量：`KV_CACHE_DTYPE`。 |
| `--continuous-batching` / `--no-continuous-batching` | 启用（默认）或关闭迭代级分页批处理。启用时服务会在批内动态加入 / 抢占序列，并在实现了 `IBatchedPagedModel` 的模型上将多个序列打包到一次前向中执行。`--no-continuous-batching` 会让所有模型回退到按序列 KV 交换。别名：`--paged-batching` / `--no-paged-batching`。 |
| `--prefill-chunk-size <N>` | 存在竞争时的分块 prefill 粒度 —— 有其他请求同时运行时，每个调度步最多处理的 prefill token 数；块越小，并行 decode 请求越容易频繁轮到 GPU（默认：`1024`）。环境变量：`TS_SCHED_PREFILL_CHUNK`。 |
| `--mtp-spec` / `--no-mtp-spec` | 在带有多 token 预测草稿头的模型上启用 NextN/MTP 投机解码（默认关闭）。草稿头可以是 Qwen 3.6 内嵌的 NextN 块，或通过 `--mtp-draft-model` 加载的 Gemma 4 `gemma4-assistant` 草稿。仅对单序列（无并发）请求生效：草稿头每步最多提议 `--mtp-draft` 个 token，主干网络用一次批量前向完成验证；起草与验证均由该请求自己的采样器（含惩罚项）驱动，输出与标准 decode 一致。仅在有收益处自动启用：Qwen 3.6 的内嵌 NextN 块在所有后端上都被认为有收益，而 Gemma 4 的独立草稿头只在各 ggml 后端与 Direct `cuda` 后端上启用；CPU / GGML CPU / MLX 走标准 decode。环境变量：`TS_MTP_SPEC`。 |
| `--mtp-draft <N>` | 每个投机步最多起草的 token 数（默认 `8`）。环境变量：`TS_MTP_DRAFT`。 |
| `--mtp-pmin <f>` | 草稿 token 被保留所需的最低置信度，取值 `(0, 1]`；遇到第一个低置信 token 即停止起草。默认值按草稿器类型选择：逐 token 草稿头为 `0.75`（其 top-10 logits 上的 top-1 概率），块级草稿器为 `0.35`——后者的门限是**累积**前缀概率，因此同一个数字要严格得多。环境变量：`TS_MTP_PMIN`。 |
| `--draft-model <path>` | 草稿器以独立文件发布的架构所用的投机解码草稿模型：DeepSeek V4 的 DSpark 支持 GGUF（见 [DeepSeek V4](docs/models/deepseek4_zh-cn.md#dspark-投机解码)）与 Muse-Glimmer 的 DFlash 草稿器（见 [Muse-Glimmer](docs/models/muse-glimmer_zh-cn.md)，环境变量 `TS_MUSE_GLIMMER_DFLASH`）。两者都每步起草一整块 token，主干用一次批量前向验证，因此贪心输出保持不变。需要与 `--mtp-spec` 一起使用，在 `cuda` 与 `ggml_cuda` 后端上对单序列请求生效。与 CLI 不同，服务端的每一行验证都用该请求自己的采样器，因此可与任意采样设置组合。环境变量：`TS_DSV4_DSPARK`。 |
| `--spec-draft-n-max <N>` | 每个投机块最多起草的 token 数（默认：草稿器训练时的块大小）。 |
| `--spec-draft-conf-min <p>` | 保留某个起草位置所需的最小累积接受概率（置信度头各位置估计值的乘积，默认 `0.35`）。 |
| `--mtp-draft-model <path>` | 对于草稿头作为独立文件发布的架构（Gemma 4 的 `gemma4-assistant`），指定其草稿 GGUF 路径。草稿的隐藏维度必须与目标一致（例如 12B 目标配 12B 草稿，而非 26B-A4B 草稿）；草稿不匹配或不完整会在启动时立即失败并给出修复提示。Qwen 3.6 将 NextN 块内嵌在主干 GGUF 中，此参数对其无效。环境变量：`TS_MTP_DRAFT_MODEL`。 |
| `--paged-kv` / `--no-paged-kv` | 已移除的按会话分页 KV 管理器的兼容参数。当前服务端 KV 状态由引擎持有；请使用连续批处理 / `TS_SCHED_*` 开关调节引擎。别名：`--paged-kv-cache` / `--no-paged-kv-cache`。 |
| `--paged-kv-block-size <N>` | 旧的独立分页 KV 块大小。当前引擎使用 `TS_SCHED_BLOCK_SIZE`。 |
| `--paged-kv-ram-mb <N>` | 旧的独立分页 KV RAM 层上限。 |
| `--paged-kv-ssd-dir <dir>` | 旧的独立分页 KV SSD 冷层目录。 |
| `--paged-kv-ssd-mb <N>` | 旧的独立分页 KV SSD 上限。 |
| `--paged-kv-quant-bits <0\|4\|8>` | 服务端接受的旧式独立分页 KV 块量化（`4`/`8` = 对称）。运行时环境变量还接受仿射 min+scale 的 `2`，CLI 则接受 `0\|2\|4\|8`。 |

请求 JSON 中的字段（如 `temperature`、`top_p`、`top_k`、`min_p`、
`repeat_penalty`、`presence_penalty`、`frequency_penalty`、`seed`、
`stop`/`stop_sequences`）会填充所有你**没有**在上面配置过的参数。对于你
**已经**配置过的参数，默认的 `--sampling-precedence config` 保留你的取值
并忽略请求中的值 —— 很多聊天客户端会把 `temperature`/`top_p` 硬编码进每一
次调用，终端用户无从修改，因此只有服务端未配置的参数才应由请求决定。若需
要相反的行为（客户端始终优先，也就是本参数出现之前的固定行为），请使用
`--sampling-precedence request`。

**运行时环境变量：**

| 变量 | 说明 |
|---|---|
| `BACKEND` | 未传 `--backend` 时使用的默认计算后端（`cpu`、`cuda`、`mlx`、`ggml_cpu`、`ggml_metal`、`ggml_cuda` 或 `ggml_vulkan`；默认：macOS 为 `ggml_metal`，其他平台为 `ggml_cpu`） |
| `MAX_TOKENS` | 未传 `--max-tokens` 时的最大生成长度：请求未携带上限时用它填充，请求要求更多时按它截断（默认：`20000`，该默认值只填充、不截断） |
| `VIDEO_SAMPLE_FPS` | 输入视频作为多模态提示词时每秒抽取的帧数；基于时间的抽帧（默认：`1`）。它与 Wan 生成视频的输出 `--fps` 无关 |
| `VIDEO_MAX_FRAMES` | 输入视频作为多模态提示词时抽取帧数的可选上限（超出时均匀降采样）；未设置或为 `0` 表示不限制（默认：不限制）。它与 Wan 生成视频的输出 `--video-frames` 无关 |
| `PORT` / `HOST` | 未传 `--port` / `--host` 时的监听端口与绑定网卡（默认：`5000`、`0.0.0.0`） |
| `ASPNETCORE_URLS` | 当 `--port`、`--host`、`--urls`、`PORT`、`HOST` 均未设置时使用的完整监听 URL |
| `TENSORSHARP_TEMPERATURE` | 未传 `--temperature` 时的采样温度。它同样算作“运维方已配置”，因此在默认的 `--sampling-precedence config` 下也优先于请求体 |
| `TENSORSHARP_TOP_K` | 未传 `--top-k` 时的 Top-K（优先级规则同 `TENSORSHARP_TEMPERATURE`） |
| `TENSORSHARP_TOP_P` | 未传 `--top-p` 时的 Top-P（优先级规则同上） |
| `TENSORSHARP_MIN_P` | 未传 `--min-p` 时的 min-P（优先级规则同上） |
| `TENSORSHARP_REPEAT_PENALTY` | 未传 `--repeat-penalty` 时的重复惩罚（优先级规则同上） |
| `TENSORSHARP_PRESENCE_PENALTY` | 未传 `--presence-penalty` 时的存在惩罚（优先级规则同上） |
| `TENSORSHARP_FREQUENCY_PENALTY` | 未传 `--frequency-penalty` 时的频率惩罚（优先级规则同上） |
| `TENSORSHARP_SEED` | 未传 `--seed` 时的随机种子（优先级规则同上） |
| `TENSORSHARP_SAMPLING_PRECEDENCE` | `config`（默认）或 `request`：服务端配置的采样参数是否优先于客户端发来的值。`--sampling-precedence` 可覆盖它 |
| `TENSORSHARP_LOG_LEVEL` | 控制台与文件日志的最低输出级别：`Trace`、`Debug`、`Information`、`Warning`、`Error`、`Critical`（默认：`Information`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_DIR` | JSON-line 文件日志的写入目录（默认：`<binDir>/logs`）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_LOG_FILE` | 设为 `0` 可关闭文件日志，仅保留控制台输出（默认：开启）。`TensorSharp.Cli` 同样识别该变量。 |
| `TENSORSHARP_TP_DEGREE` | 张量并行度 —— 把模型切分到本机多少张 GPU 上（默认：`1`）。当未传 `--tp` 参数时作为 `ModelBase.Create` 的兜底来源；`TensorSharp.Cli` 与 `TensorSharp.Server` 都提供了 `--tp <N>` 参数。需要 `--backend cuda`、`ggml_cuda` 或 `ggml_vulkan`。 |
| `TENSORSHARP_TP_DEVICES` | 各 rank 使用的 GPU 序号（逗号分隔，例如 `0,2`；默认 `0..tp-1`）。用于 GGML 后端上的 TP。 |
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
| `TS_MTP_PMIN` | 草稿 token 被保留所需的最低置信度，取值 `(0, 1]`（默认按草稿器类型：逐 token 为 `0.75`，块级为 `0.35`）。CLI：`--mtp-pmin`。 |
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

采样参数的优先级（从高到低，对应默认的 `--sampling-precedence config`）：

1. 服务端命令行参数 / 配置文件键（如 `--temperature`、`--top-p`、`--stop`）。
2. 上面列出的 `TENSORSHARP_*` 环境变量。
3. API 请求 JSON 中的字段（如 `temperature`、`top_p`、`stop`）—— 仅对第 1、2 步未设置的参数生效。
4. `SamplingConfig` 内置默认值（`temperature=0.8`、`top_k=40`、`top_p=0.9`、`min_p=0`、`repeat_penalty=1.1`、`repeat_last_n=64`、存在/频率惩罚均为 `0`、`seed=-1`、无停止序列）。

使用 `--sampling-precedence request` 时，第 1~3 步互换：请求中出现的参数优先
于服务端参数与环境变量，其余参数仍由服务端填充。无论哪种模式，服务端 `--stop`
在 `config` 下始终生效（与请求的列表合并），在 `request` 下则被请求替换。

## 视频生成（Wan）

一个 `wan` GGUF 可以把提示词——Wan 2.2 模型还可再加一张首帧图片——变成 H.264 MP4，
`TensorSharp.Cli`、服务端的三个视频端点以及 Web UI 聊天都能驱动。完整的架构细节见
[Wan 卡片](docs/models/wan_zh-cn.md)；本节是运维视角：该下载哪个检查点，以及哪些开关
真正影响墙钟时间。

### 选哪个检查点

| 家族 | 潜空间 | 模式 | 说明 |
|---|---|---|---|
| Wan 2.2 TI2V-5B | 48 通道、16×16×4（`Wan2.2_VAE.safetensors`） | 文生视频 + 图生视频 | 稠密 5B、24 fps、原生 720p；同分辨率下 DiT token 数约为 Wan 2.1 的 1/2.7，因此在消费级 GPU 上既最快也质量最好 |
| Wan 2.2 A14B（T2V / I2V） | 16 通道（I2V 输入 36 通道），`wan_2.1_vae.safetensors` | 文生视频 + 图生视频 | 两个 14B 专家在时间步边界切换；**两个**专家 GGUF 都必须就位（同一目录，或 `HighNoise/` + `LowNoise/`） |
| Wan 2.1 T2V（1.3B / 14B） | 16 通道，`wan_2.1_vae.safetensors` | 文生视频 | 单个 DiT |

每个家族都还需要 UMT5-XXL 文本编码器（`umt5-xxl-encoder-Q8_0.gguf`）和匹配的视频 VAE。
这三个伴随文件都会从 DiT 自身所在目录解析，包括 `VAE/`、`HighNoise/`、`LowNoise/` 这类
子目录，因此一个 `--local-dir` 就够了；`--wan-vae` / `--wan-te`（以及第二个 A14B 专家的
`TS_WAN_DIT2`）可以覆盖搜索结果。

Wan 是唯一会直接拒绝某个后端的家族：它可以运行在 `ggml_cuda`、`ggml_vulkan`、
`ggml_metal`、`ggml_cpu`、`cuda` 与 `cpu` 上，**不支持** `--backend mlx`。`ggml_cuda`
最快（RTX 2000 Ada、Wan2.1-1.3B F16、832×480×33f、30 步：`ggml_cuda` 12.0 秒/步，
`ggml_vulkan` 17.2 秒/步，Direct `cuda` 19.3 秒/步）；`cpu` / `ggml_cpu` 仅供功能性使用。

### 提速主路径：步数蒸馏检查点

**这是 Wan 最大的提速手段，代价只是换一个下载。** Wan2.2-TI2V-5B 的官方配方是
50 步 × 2 次无分类器引导前向 = **100 次 DiT 前向**。步数蒸馏检查点（Turbo / Lightning /
FastWan / DMD）本身就是按无引导、少步数训练的，因此同一段视频只需 **4 次 DiT 前向**，
去噪工作量降到 1/25。

TensorSharp 从 DiT 的**文件名**识别它：不区分大小写地匹配 `turbo`、`distill`、
`lightning`、`lightx2v`、`fastwan`、`-dmd`，或显式的 `<N>steps` / `<N>_steps`（1 ≤ N ≤ 16，
显式步数优先于标记）。只有标记而没有步数时默认为 4 步。加载时控制台会打印

```
step-distilled checkpoint detected -> 4 steps, guidance off (--diffusion-steps / --cfg override)
```

因此可以从日志确认识别是否生效。这里没有任何参数——蒸馏 GGUF 就当作普通的 `--model` 传入。

在 M5 Pro（20 核 GPU、48 GB 统一内存）、`ggml_metal`、Wan2.2-TI2V-5B Q8_0、
1088×832×121f = 27 404 token、图生视频（即完整的五秒 720p 级请求）上实测：

| | 基础检查点 | **Turbo 检查点** |
|---|---|---|
| DiT 前向次数 | 100（50 步 × CFG） | **4**（4 步，无引导） |
| 每次前向 | 120.2 秒 | 120.2 秒 |
| 去噪合计 | 12 020 秒 | **481 秒** |
| VAE 解码 121 帧 | 563 秒 | 563 秒 |
| **端到端** | **约 3 小时 30 分** | **17 分 30 秒** |

这两列之间只有 `--model` 路径不同。一旦用上蒸馏检查点，瓶颈就变成 VAE 解码（约占整段
运行的 55%），而不再是 DiT。

下载方式（TI2V-5B —— 注意文件名里的版本号用的是**下划线** `Wan2_2`，与基础仓库不同）：

```bash
pip install -U huggingface_hub
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
# Turbo 仓库不含 VAE 与文本编码器——从基础仓库取
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf \
    --backend ggml_metal --image first_frame.png --output out.mp4 \
    --prompt "the cat runs toward the camera, cinematic tracking shot" \
    --video-frames 121 --fps 24

dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf \
    --backend ggml_metal --video-frames 121 --fps 24
```

Wan 2.2 I2V-A14B 的即插即用蒸馏 GGUF 是
[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)，
它已经把蒸馏合并进两个专家，分别放在 `high_noise/` 与 `low_noise/` 下；两个都下载，
`--model` 指向其中任意一个即可。该仓库同样不含 VAE 与文本编码器，因此
`VAE/Wan2.1_VAE.safetensors` 取自
[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)，
UMT5-XXL 编码器同上。

> [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
> 只发布 LoRA `.safetensors`，而 TensorSharp 没有任何 Wan LoRA 参数——请选择已经把
> 蒸馏烘焙进 GGUF 的仓库。

### `--cfg-cache-stride`（仅基础检查点）

一次带引导的去噪步是 `v = v_cond + (cfg-1)·d`，其中 `d = v_cond - v_uncond`。引导方向 `d`
在整个调度上的变化远比 `v` 缓慢，因此 `--cfg-cache-stride N` 让无条件前向每 `N` 步只跑
一次，中间各步复用缓存的 `d`。在 50 步下，`2` 只跑 100 次前向中的 77 次（**1.30×**），
`3` 跑 70 次（**1.43×**）。前三步与最后一步始终重新计算 `d`。服务端 JSON 字段：
`"cfgCacheStride": 2`。

这是一种近似——需要严格对齐参考样本时请关闭；在步数蒸馏检查点上也没有意义，因为它们本来
就无引导运行（cfg ≤ 1.0 时缓存会被自动禁用）。

### 让一个大请求变便宜，按收益排序

1. **换用步数蒸馏检查点。** 100 次 DiT 前向变成 4 次，收益远超其他所有手段。
2. **减少帧数。** 121 → 61 大约把注意力工作量降到 1/4（token 数是
   `latent_frames × (h/2) × (w/2)`，自注意力是 `O(tokens²)`），VAE 解码减半。
3. **减小画面面积**——但不要低于 Wan 的训练分辨率。低于约 0.3 MP 后模型进入分布外，
   视频会变*差*而不只是变便宜，管线也会给出警告。Wan 在 480p（832×480）与
   720p（1280×704）上训练，所以请在受支持的尺寸上生成，再事后缩小。
4. **减少步数**，仅限基础检查点——30 步相对官方 50 步在观感上很接近，成本降到 1/1.7。
5. **`--cfg-cache-stride 2` 或 `3`**——1.30× / 1.43×，仅限基础检查点。

同一台 M5 Pro、同一个 Turbo 检查点与输入图，分辨率与墙钟时间的关系：

| 输出 | Token 数 | 去噪 | VAE 解码 | **合计** |
|---|---|---|---|---|
| 736×544 × 81f（3.4 秒，480p 级） | 8 211 | 84 秒 | 159 秒 | **4 分 09 秒** |
| 736×544 × 121f（5 秒，480p 级） | 12 121 | 137 秒 | 237 秒 | **6 分 19 秒** |
| 1088×832 × 121f（5 秒，720p 级） | 27 404 | 481 秒 | 563 秒 | **17 分 30 秒** |

480p（约 0.4 MP）是 Wan *训练过*的分辨率，因此前两行属于分布内，而不是降级模式——当你
只能接受几分钟时，就该选它。

### 服务端默认值

`--video-frames N` 与 `--fps N` 为 Web UI 和三个视频端点设置服务端级**默认值**，不是上限；
请求中提供的 `frames` 或 `fps` 会各自独立覆盖。两者都不提供时使用模型配方：Wan2.2-TI2V
为 49 帧 / 24 fps，其余为 33 帧 / 16 fps。帧数会对齐到 VAE 的 `4k+1` 时序网格。改变时长
时请保持模型原生 FPS 而调整帧数——只改 FPS 只会改变播放速度。

### 其他 Wan 开关

这些开关用于 A/B 与调试，除注明外都会让速度变慢。`TS_WAN_DIT_KV_F16=0` 恢复 F32 的注意力
键值（F16 是默认值，单次 27k token 自注意力快 2.02×，且没有可测的精度损失）；
`TS_WAN_VAE_MPS_CONV=0` 在 Metal 上恢复 ggml 的 im2col+GEMM 卷积下降路径（MPSGraph 是默认
值，把 736×544×81f 的 VAE 解码从 159 秒降到 80 秒）；`TS_WAN_VAE_GEMM_MAX_MB` 设置 im2col
预算，`TS_WAN_VAE_TILE=0` 关闭分块；`TS_WAN_DIT_CAPTURE=0` 关闭常驻的 CUDA 图捕获 DiT 图；
`TS_WAN_DIT_FLASH=0` 强制走物化注意力；`TS_WAN_HEARTBEAT_S` 设置进度心跳间隔（默认 30 秒，
`0` 静默）；`TS_FFMPEG` 指定用于近无损 CRF 17 H.264 导出的 `ffmpeg`。

## 混合专家 CPU 卸载（`--n-cpu-moe`）

大型 MoE 模型几乎所有权重都花在路由专家上，而每个 token 只激活其中很小一部分
——Qwen3.6-35B-A3B 有约 11 GB 权重，活跃参数却只有约 3B。`--n-cpu-moe N` 把前 N
层的路由专家留在系统内存里，注意力、各个 norm、路由器以及始终活跃的共享专家仍留
在加速器上。因此 `--n-cpu-moe` 的含义是「这些专家不驻留显存」，而不是「这些专家在
CPU 上做矩阵乘」。它们究竟在哪里被相乘，取决于批大小：

* **Decode（单 token，或任何小于 `TS_HOST_MOE_DEVICE_MIN_BATCH` 的批，默认 128）。**
  由主机做乘法，直接从 GGUF 的 mmap 零拷贝读取量化块。一个 token 只会碰到
  `n_expert_used` 个专家，因此每层只有几 MB 的内存流量，而一次上传要花掉几百 MB。
* **Prefill（真正成批时）。** 专家会为这一张图**流式**送到加速器上并在那里相乘，
  随后暂存内存被下一层复用——它们始终不会常驻。512 token 的批会把这次上传摊薄到
  批内每个 token 上，而算术正是 GPU 的用武之地。llama.cpp 的调度器也是同样的判断
  （`ggml_backend_sched` 会把权重位于主机缓冲区的算子在超过
  `op_offload_min_batch_size` 时送回 GPU）。

有三点让流式这一侧变得很快，这也是 TensorSharp 的卸载 prefill 在相同文件上比
llama.cpp 快数倍的原因：

1. **主机页被锁定。** 从可分页的 mmap 里做拷贝无法 DMA——驱动会通过一小块固定内存
   中转。对专家区间调用 `cudaHostRegister` 一次性花费约 65 ms/GiB，却把 PCIe 5.0
   x16 上的传输从 9.3 GB/s 提升到 55.6 GB/s。用 `TS_HOST_MOE_PIN=0` 关闭，用
   `TS_HOST_MOE_PIN_MAX_MB` 设上限（默认为 cgroup/主机内存上限的 60%）。
2. **只发送这一批实际路由到的专家**，并按连续区段分组——和 llama.cpp 调度器用
   已用专家位图玩的是同一个把戏。512 token 时较大的专家池只会被部分覆盖，而在投机
   验证与轻负载服务产生的小批下，这项节省相当可观。`TS_HOST_MOE_EXPERT_FILTER=0`
   恢复整栈上传。
3. **拷贝是异步下发的**，走后端自己的 stream，并且随图一次同步，而不是每层同步三次。

被卸载的层仍然留在整模型融合图里：加速器在每个卸载层的路由器之后暂停，专家被相乘
（在主机上，或用流式送达的权重在加速器上），结果再交回，之后下一段才继续运行。token
里的其他一切——注意力、共享或稠密 FFN、LM head——仍然是一次图提交。这对
Qwen3.5/3.6、Gemma 4 MoE、GPT-OSS 与 DiffusionGemma 都成立；Gemma 4 MoE 的 prefill
图也按同样方式分段，而 DiffusionGemma 的块级 decode 会一次性把整块画布位置交给主机，
因此它卸载出去的是 GEMM 而不是 matvec。Qwen3.5/3.6 的 prefill 图同样分段，并且它和
Gemma 4 MoE 在张量并行下也保留融合图（见下文 `--tp` 说明）。GPT-OSS 现在同样有整模型
融合 prefill 图，因此它的卸载 prefill 也是分段而非逐层派发。仍然没有融合图的架构
（Nemotron-H）通过各自的逐层 MoE 算子进入流式路径，且只在单设备上；在 `--tp N` 下
它们保持在主机侧，以免 N 个 rank 各自流式传输同一份未切分的专家。

在 2 × Xeon 6952P + RTX PRO 6000 Blackwell（PCIe 5.0 x16）上实测，**pp8192 /
tg128**，显存为每进程峰值。与 llama.cpp 的完整对比——两种提示长度、每个模型五个卸载
深度，含跨两张 GPU 的 DeepSeek V4 Flash——见
[docs/moe_cpu_offload_benchmark.md](docs/moe_cpu_offload_benchmark.md)：

| 模型 | 设置 | 峰值显存 | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (UD-IQ4_XS) | 默认 | 16.4 GiB | 11,274 tok/s | 161 tok/s |
| | `--n-cpu-moe 8` | 15.4 GiB | 6,500 tok/s | 80 tok/s |
| | `--n-cpu-moe 16` | 13.8 GiB | 4,888 tok/s | 55 tok/s |
| | `--cpu-moe`（30 层） | 10.8 GiB | 3,072 tok/s | 40 tok/s |
| Qwen3.5-35B-A3B (UD-IQ4_XS) | 默认 | 19.4 GiB | 9,405 tok/s | 160 tok/s |
| | `--n-cpu-moe 24` | 15.1 GiB | 5,259 tok/s | 52 tok/s |
| | `--cpu-moe`（48 层） | 11.3 GiB | 3,709 tok/s | 39 tok/s |
| gpt-oss-20b (Q8_0/MXFP4) | 默认 | 12.9 GiB | 12,925 tok/s | 213 tok/s |
| | `--n-cpu-moe 12` | 9.2 GiB | 6,394 tok/s | 52 tok/s |
| | `--cpu-moe`（24 层） | 4.7 GiB | 3,798 tok/s | 28 tok/s |
| DeepSeek-V4-Flash (UD-Q8_K_XL，2 张 GPU) | 默认 | 165.2 GiB | 4,387 tok/s | 51 tok/s |
| | `--n-cpu-moe 12` | 128.7 GiB | 428 tok/s | 10 tok/s |
| | `--n-cpu-moe 24` | 77.9 GiB | 236 tok/s | 5.3 tok/s |

在三个对照模型上，每一个卸载深度的 prefill 都是 llama.cpp 的 **4.5–10.9×**，decode
是 **2.5–4.5×**。峰值显存高于 llama.cpp（常驻时 1.1×，完全卸载时最高 3.5×）——详见
报告中的显存说明。

在 gemma-4-26B-A4B 上，任何卸载深度下的贪心输出都是 **token 一致**的。

同一功能在笔记本上的小机器画面（RTX 3080 Laptop 16 GB / i7-11800H，生成 96 个
token）——注意卸载配置离这张卡的 16 GB 有多远，以及 GPT-OSS 与 DiffusionGemma 的基线
本来就越过了 WDDM 的溢出阈值，这正是卸载在那里既能减小占用又能**提速**的原因：

| 模型 | 设置 | 峰值显存 | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (Q4_K_XL) | 默认 | 16.1 GB | 190 tok/s | 39.7 tok/s |
| | `--n-cpu-moe 8` | 13.1 GB | 52 tok/s | 38.6 tok/s |
| | `--n-cpu-moe 16` | 10.2 GB | 24 tok/s | 21.6 tok/s |
| | `--cpu-moe`（30 层） | 4.8 GB | 15 tok/s | 17.7 tok/s |
| gpt-oss-20b (Q8_0) | 默认 | 16.2 GB | 2.7 tok/s | 0.3 tok/s |
| | `--n-cpu-moe 12` | 14.0 GB | 58 tok/s | 25.4 tok/s |
| | `--cpu-moe`（24 层） | 2.9 GB | 29 tok/s | 12.1 tok/s |
| diffusiongemma-26B-A4B (Q4_K_M) | 默认 | 16.0 GB | — | 1709 ms/步 |
| | `--n-cpu-moe 16` | 9.8 GB | — | 2287 ms/步 |
| | `--cpu-moe`（30 层） | 3.0 GB | — | 4697 ms/步 |

（上面这些笔记本 prefill 数字早于流式专家路径；同样的配置现在要快好几倍。）

注意事项：

* **选能装下的最小 N。** 每卸载一层都要付出 decode 吞吐的代价，因此只卸载刚好够
  腾出 KV 缓存所需显存的层数。
* **按比例算，decode 付出的代价最大。** 提示处理会把专家流式送到加速器上，与常驻
  运行相差不大；decode 则受限于主机每层从内存里读取 `n_expert_used` 个专家的速度，
  吞吐就消耗在这里。
* **工作线程数上限是 64。** decode 侧的矩阵乘只有一个 token 宽——是内存受限、每个
  算子后都有一次栅栏的工作——因此超过几十个线程之后，每多一个 worker 只是多一个栅栏
  参与者，在双路主机上还会增加跨 socket 流量。在 2× Xeon 6952P（192 核 / 384 线程）
  上用 gemma-4-26B `--cpu-moe` 实测，30 个卸载层的主机矩阵乘耗时（毫秒）：8 线程 45、
  16 线程 35、32 线程 17.6、64 线程 17.3、96 线程 26、192 线程 120。可用
  `--cpu-moe-threads N` 覆盖。
* **路由仍在加速器上**，因此专家的选择与权重与完全常驻路径挑出来的完全一致。
* **整模型融合图会被保留。** Qwen3.5/3.6、Gemma 4 MoE、GPT-OSS 与 DiffusionGemma
  各自把一整个 token（DiffusionGemma 是一整块画布）作为**单张**加速器图运行，卸载不会
  放弃这一点。每个卸载层的专家矩阵乘被从图里切出去；其余的一切——注意力、norm、路由器、
  共享/稠密 FFN 与 LM head——依然融合运行，加速器只在把该层的路由专家工作交给主机时暂停。
  Gemma 4 MoE 的 **prefill** 图也按同样方式分段，因此一个提示分块仍然是一次派发，
  而主机对块内每个 token 做的是真正的 GEMM。
* Nemotron-H 没有可保留的整模型融合图——它混合 Mamba2/注意力/MoE 的 decode 本来就是
  逐层派发——因此在它上面卸载只是把每层 MoE 走主机路径。
* **可与张量并行组合。** 在 `--tp N` 下，卸载层的接缝被并入各 rank 已有的 AllReduce
  分段调度中，因此多 rank 融合图得以保留——Qwen3.5/3.6 与 Gemma 4 MoE 的 decode 与
  prefill 都保持融合。每个卸载层只在主机上、对未切分的专家栈**求值一次**：主机专家后端
  是单一线程池，把那次矩阵乘按 rank 拆开只会在它上面串行，还要各自多付一次下载。随后
  这唯一的结果会被放置到使图自身的归约落在单 GPU 数值上的位置（该层输出会喂给后续
  AllReduce 时只放 rank 0，否则每个 rank 都放）。在 2× L4 24 GB 上实测（短提示，
  prefill 512 / decode 64）：

  | 模型 | 设置 | 常驻权重（GPU0 + GPU1） | Prefill | Decode |
  | --- | --- | --- | --- | --- |
  | Qwen3.5-35B-A3B (UD-IQ4_XS) | `--tp 2` | 9397 + 8032 MB | 1942 tok/s | 76.5 tok/s |
  | | `--tp 2 --cpu-moe` | 2277 + 912 MB | 66 tok/s | 17.8 tok/s |
  | gemma-4-26B-A4B (UD-IQ4_XS) | `--tp 2` | 6835 + 6087 MB | 3062 tok/s | 59.7 tok/s |
  | | `--tp 2 --n-cpu-moe 8` | 5459 + 4711 MB | 217 tok/s | 36.5 tok/s |
  | | `--tp 2 --cpu-moe` | 1588 + 840 MB | 60 tok/s | 13.2 tok/s |

  在 Gemma 4 上，贪心输出在大多数提示下与不带卸载的同一次 `--tp N` 运行逐字节一致，
  其余提示下也能一致数百个字符，之后走向一个等价的结尾；Qwen3.5 同样能跟随常驻运行
  到一个改写点。两者都是确定性的——差异来自主机专家矩阵乘的激活量化，也就是
  `TS_HOST_MOE_VERIFY=1` 在单 GPU 上报告的那约 1–5% 的相对差异。纯 C# 的 `cuda`
  后端没有主机 MoE 接缝，因此在那里使用 `--tp N --cpu-moe` 会打印
  `[moe-offload] WARNING` 并让专家保持常驻。
* `TS_HOST_MOE_VERIFY=1` 会在主机链路旁同时构建 GPU 上的专家链路，并报告二者的逐层
  偏差——用于在同样能塞进显存的模型上验证接缝的诊断手段。`TS_HOST_MOE_DEBUG=1` 打印
  分段计划（节点切点）与每个接缝的激活范数。`TS_HOST_MOE_TIMING=1` 报告卸载侧的墙钟
  时间，拆分为每次调用的准备开销与主机矩阵乘——这能告诉你一次慢运行到底慢在专家 GEMM
  还是它周围的脚手架上。

## 张量并行与分布式推理

TensorSharp 支持**张量并行（TP）**——按 Megatron-LM 列/行并行范式把单个模型切分
到多张 GPU 上——以及**分布式（多节点）张量并行**，让 TP 跨越多台通过 TCP 点对点
网络互联的机器。

### 本地张量并行（单进程，多 GPU）

在一个进程内把模型切分到 N 张 GPU 上（Direct CUDA，或 GGML CUDA / Vulkan 后端）。
每张 GPU 持有 `1/N` 的切分权重（列并行的 QKV / gate / up，行并行的 output /
down），外加一份完整的复制权重（归一化层、词嵌入、LM head）。各 GPU 的 KV 缓存
彼此独立；每次行并行投影之后由 AllReduce（CUDA P2P 拷贝 + 逐元素加法内核）把隐
藏状态重新汇聚。

```bash
# CLI：2 GPU 张量并行
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2

# 服务端：同样的参数（也可用 TENSORSHARP_TP_DEGREE=2 环境变量）
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
    --model <model.gguf> --backend cuda --tp 2

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

# 以服务端作为集群前端：服务端必须是节点 0（负责采样并对外提供 HTTP 的 driver），
# 其余每个节点都运行一个 TensorSharp.Cli worker，使用相同的模型、后端与 peer 列表。
# TENSORSHARP_TP_* 环境变量同样可用。
# 节点 0（服务端 / driver）：
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# 节点 1（CLI worker）：
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

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
| Gemma 4 | ✅ | 稠密 TP + MoE。GGML 上融合的整模 MoE 主干在**每个专家内部**切分（gate/up 列并行、down 行并行），从而保留全局专家 id；`TS_GEMMA4_TP_FUSED_MOE=0` 可回退到逐算子的整专家路径。Direct CUDA 上为逐专家切分 |
| Qwen 3.5 / 3.6 family | ✅ | GatedDeltaNet SSM 按 rank 划分 V-head 归属；GGML 上为专家并行 MoE（每个 rank 持有整个专家，shared expert 仍按 Megatron 切分）与列并行 LM head，Direct CUDA 上为专家切分。`cuda` 与 `ggml_cuda` / `ggml_vulkan` 均可运行——GGML 路径使用打包的按 rank GDN 内核（`TSGgml_Qwen35GdnLayerTP`）并把循环状态常驻设备 |
| GPT OSS | ✅ | MoE 专家切分，attention sink，YaRN。`cuda` 与 GGML 后端均可运行；GGML 路径目前仍按 token 逐个遍历专家（尚未使用专家并行） |
| Nemotron-H | ✅ | Mamba2 在 rank 0 上复制计算，MoE 专家切分。GGML 上的限制与 GPT OSS 相同 |
| DiffusionGemma | — | 不适用（扩散模型） |
| Qwen-Image-Edit | — | 不适用（图像生成） |

### 后端支持

TP 可运行在 **Direct CUDA** 后端（`--backend cuda`）以及 **GGML CUDA / Vulkan**
后端（`--backend ggml_cuda`、`ggml_vulkan`）上；MLX 为单设备。

在 GGML 后端上，每个 rank 在自己的 GPU 上拥有独立的 ggml 后端、常驻设备的权重分
片与 KV 缓存。跨 GPU AllReduce 使用 ggml-cuda 的集合通信（构建时能找到 NCCL 就用
NCCL，否则用其 P2P 流水线）；小载荷则改在主机内存中归约——因为 GGML 的激活本来就
在主机内存里，这样反而更省。

```bash
# GGML CUDA 后端上的 2 GPU
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_cuda --tp 2

# 指定各 rank 对应的物理 GPU
TENSORSHARP_TP_DEVICES=0,2 dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model <model.gguf> --backend ggml_cuda --tp 2
```

**预期效果。** GGML 上的 TP 最初是**容量**特性——让单卡显存装不下的模型完整跑在
GPU 上——而融合的按 rank 执行（Stage 1c）让它同时成为延迟收益。现在每个 TP block
都以按 rank 融合的原生计算图运行（注意力、稠密 FFN、MoE 主干、GatedDeltaNet），
而不再逐算子下发，因此每层每个 rank 只需几次图调度，而不是数百次原生调用。

在 2× RTX 2000 Ada（各 16 GB，PCIe，无 NVLink）上测得，prefill 512 / decode 64，
单位 tok/s：

| 模型 | 单 GPU | `--tp 2` |
|---|---|---|
| Gemma 4 E4B Q8_0 | 2760 / 37.3 | 2488 / **51.7** |
| Gemma 4 26B-A4B IQ4_XS | 1845 / 48.5 | 2537 / **51.2** |
| Qwen 3.5-9B Q8_0 | 1461 / 23.1 | 399 / **24.4** |
| Qwen 3.5-35B-A3B IQ4_XS | 装不下 | **184 / 18.1** |

Decode——TP 本该受益的访存瓶颈部分——在 Gemma 4 E4B 上达到单卡的 1.39×，在
Qwen 3.5-9B 上为 1.06×；两个 Gemma 4 模型的输出与单卡逐字节一致。Prefill 受计算
约束且要承担集合通信开销，因此在单卡装得下的模型上持平或略低于单卡。
Qwen 3.5-35B 在 16 GB 卡上根本装不下，只能靠 TP 运行。完整测量数据与尚待融合的
部分见 `TENSOR_PARALLELISM_PLAN.md`（Stage 1b 与 1c）。

| 变量 | 作用 |
|---|---|
| `TENSORSHARP_TP_DEVICES` | 各 rank 使用的 GPU 序号，例如 `0,2`（默认 `0..tp-1`） |
| `TS_GGML_TP_PARALLEL=0` | 顺序而非并发地驱动各 rank（诊断用） |
| `TS_GGML_TP_FUSED_MATMUL=1` | 由单个线程提交两个 rank 的线性层（默认关闭；它每次调用都要为每个 rank 分配设备缓冲，在 Qwen 3.5 35B 上实测慢 2.3×） |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | 超过该元素数量时 AllReduce 走设备集合通信（默认 262144） |
| `TS_GGML_F32_RESIDENT=0` | 每次调用重新绑定 F32 线性层权重，而不是常驻设备（诊断用） |
| `TS_GEMMA4_TP_FUSED_MOE=0` | 仅 Gemma 4：从融合的整模 MoE 主干（专家内部 Megatron 切分）回退到逐算子的整专家路径。融合路径在 26B 上加载时会物化约 10.5 GB 的专家分片（约 36 秒），换来约 10× 的 decode |
| `GGML_CUDA_AR_BF16_THRESHOLD` | ggml-cuda 集合通信在多大载荷以上把 F32 转成 BF16 再归约。TensorSharp 把 ggml 的默认值（1 字节，即总是转换）提高到 1 MB，使 decode 规模的集合通信精确归约；设为 `0` 则完全禁用转换 |
| `TS_QWEN35_LAYER_TRACE=1` | 打印首次前向的逐层残差流摘要，单卡与 TP 两条路径都会输出（诊断用） |
| `GGML_CUDA_ALLREDUCE` | `nccl` / `internal` / `none`，直接透传给 ggml |

### 约束

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
| 草稿 token 被保留所需最低置信度 | 按草稿器类型（`0.75` / `0.35`） | `TS_MTP_PMIN` | `--mtp-pmin X` |
| Gemma 4 独立草稿 GGUF（`gemma4-assistant`） | 无 | `TS_MTP_DRAFT_MODEL` | `--mtp-draft-model <path>` |
| Gemma 4 融合验证 / 草稿内核（ggml） | 开启 | `TS_GMTP_NO_FUSED=1` 回退到逐算子 | — |
| Gemma 4 部分接受时的稠密快速回滚 | 开启 | `TS_GMTP_NO_FAST_ROLLBACK=1` 恢复保留前缀回滚 | — |
| Gemma 4 验证主干路径 | 线性（单序列） | `TS_GMTP_BATCHED_TRUNK=1` 走批量分页主干 | — |

#### 张量并行与分布式推理

| 功能 | 默认 | 环境变量 | CLI 等价参数 |
|---|---|---|---|
| 本地张量并行（单进程，多 GPU） | 关闭（`1` 张 GPU） | **`TENSORSHARP_TP_DEGREE=N`** | `--tp N`（CLI 与服务端） |
| TP 各 rank 使用的 GPU 序号（GGML 后端） | `0..tp-1` | `TENSORSHARP_TP_DEVICES=0,2` | — |
| 分布式 TP 节点编号（多节点） | 未设置（关闭） | **`TENSORSHARP_TP_NODE_ID=N`** | `--tp-node-id N`（CLI 与服务端；服务端必须是节点 `0`） |
| 分布式 TP peer 端点 | 未设置（关闭） | **`TENSORSHARP_TP_PEERS=host1:port1,host2:port2`** | `--tp-peers host1:port1,host2:port2`（CLI 与服务端） |
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

这些变量用于填充请求体未提供的字段。CLI 参数优先于环境变量；除非服务以
`--sampling-precedence request` 启动，否则通过二者之一配置的参数还会优先于请求体。

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
| 采样优先级 | `TENSORSHARP_SAMPLING_PRECEDENCE` | `--sampling-precedence` |

#### 服务托管与上传（仅服务端）

| 功能 | 默认 | 环境变量 |
|---|---|---|
| ASP.NET Core 监听 | `http://0.0.0.0:5000` | `--port` / `--host` / `--urls`，其次 `PORT` / `HOST`，再次 `ASPNETCORE_URLS` |
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

