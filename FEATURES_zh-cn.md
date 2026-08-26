# 功能特性
[English](FEATURES.md) | [中文](FEATURES_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分。


- **多架构支持** —— DeepSeek V4 Flash、GLM 5.x、Gemma 4、Gemma 3、DiffusionGemma、Qwen 3、Qwen 3.5/3.6-family、GPT OSS、Nemotron-H、Mistral 3、Muse-Glimmer、Qwen-Image-Edit（图像编辑）、MiniMax-H3（视频 + 原生 32 kHz 立体声音频），以及 Wan 2.1/2.2（仅视频）
- **多模态推理** —— 图像、视频和音频输入（Gemma 4）；图像输入（Gemma 3 / Qwen 3.5/3.6-family / Mistral 3 / Muse-Glimmer / Nemotron-H Omni）。音频输入仅 Gemma 4 支持。`--pdf` 与架构无关：原生数字 PDF 的文本层会被内联进任意模型的提示词，只有扫描件才回退为页面图像（此时需要视觉模型）。生成的媒体是另一条轴：Qwen-Image-Edit 输出图像，Wan 2.1/2.2 输出 H.264 MP4，而 MiniMax-H3 是唯一**连音频一起输出**的家族——32 kHz 立体声音轨与画面联合去噪，并作为旁挂 `.wav` 写在 MP4 旁边
- **思维链 / 推理模式** —— 通过 `<think>` / `<|channel>thought` / `<|channel>analysis` 标签输出结构化的思维链推理（Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H、Muse-Glimmer、DeepSeek V4、GLM 5.x）
- **工具调用 / 函数调用** —— 模型可调用用户定义的工具；所有三种 API 风格均支持多轮工具调用对话
- **量化模型支持** —— 加载 Q4_K_M、Q8_0、F16、MXFP4 等量化格式的 GGUF 文件；执行原生量化矩阵乘法（matmul），无需反量化到 FP32，并且纯 C# CPU 后端在加载大型 GGUF 时也会保持量化权重压缩状态
- **GPU 加速** —— 通过 GGML 支持 Apple Metal（macOS）、GGML CUDA（Windows/Linux + NVIDIA）和 GGML Vulkan（Windows/Linux + AMD/Intel/NVIDIA），并提供 Direct CUDA/cuBLAS 后端（含 PTX 内核与未覆盖算子的 CPU 回退），以及面向 Apple Silicon 的 MLX 后端（mlx-c / Metal）
- **优化后的纯 C# CPU 后端** —— 为 GEMM、RMSNorm、RoPE、softmax、融合激活等推理热点路径提供托管快速路径和 SIMD 内核
- **连续批处理 & 分页 KV 缓存** —— vLLM 风格的分页 KV 块池，跨请求的块级哈希前缀共享，迭代级调度器（可在批内动态加入/抢占序列），可选的 SSD 冷层用于超大 KV 工作集，原生融合分页注意力内核（`TSGgml_PagedAttentionForward`，在 Metal/CUDA/Vulkan 上驱动 `ggml_flash_attn_ext`）。`TensorSharp.Server` 默认启用，可用 `--no-continuous-batching` 关闭。详见 [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。GLM 5.x 是个例外：带权重吸收的 MLA（每层每 token 只占一行 576 宽的缓存）与 DSA lightning indexer 没有分页布局，因此那里的并发靠原生的按序列**槽位**来承载——每个请求拥有自己的 MLA 与索引器缓存以及自己的 `n_past`，绑定请求只是切换活跃槽位，不搬运任何 KV 字节。
- **投机解码** —— 在共享的"起草—验证—回滚"运行时之上，架了一层可插拔的算法（`--spec-type`：`auto` / `draft-head` / `block` / `ngram`）；无权重的 `ngram` 投机器对**所有**模型都可用，训练出来的草稿头则加速单序列（无并发）decode。Qwen 3.6 与 GLM 5.2 将 NextN 块内嵌在主干 GGUF 中；Gemma 4 通过 `--spec-draft-model` 加载独立的 EAGLE 风格 `gemma4-assistant` 草稿 GGUF，其草稿层读取目标模型自身的 KV 缓存。草稿每步最多提议 `--spec-draft` 个 token（草稿置信度 ≥ `--spec-pmin` 时保留），主干用一次批量前向完成验证；起草与验证均由该请求自己的采样器（含惩罚项）驱动，因此输出与标准 decode 完全一致。CLI 与服务端均通过 `--spec` 启用（默认关闭）。在 `TensorSharp.Cli` 上，它在所有单序列路径上生效——`--input`、`--multi-turn-jsonl` 与 `--interactive`。ggml 后端有融合的多 token 验证 / 草稿步内核，是明确收益；Direct `cuda` 后端运行完全驻留 GPU 的逐算子验证 / 草稿，同样有收益；CPU / GGML CPU / MLX 保持标准 decode。环境变量：`TS_SPEC_*`（通用；旧的 `TS_MTP_*` 拼法仍然有效）与 `TS_GMTP_*`（Gemma 4 调优）。
- **张量并行与分布式推理** —— 用 `--tp N`（`TensorSharp.Cli` 与 `TensorSharp.Server` 均支持，也可用 `TENSORSHARP_TP_DEGREE`）把一个模型按 Megatron-LM 列/行并行范式切分到多张 GPU 上，再用点对点 TCP 集群（`--tp-node-id` / `--tp-peers`）扩展到多台机器。分层 AllReduce 把跨网络流量降到最低。可运行在 Direct `cuda` 后端以及 GGML CUDA / Vulkan 后端上——后者每个 rank 在自己的 GPU 上拥有独立的 ggml 后端、权重分片与 KV 缓存。支持全部自回归架构（Qwen 3、Mistral 3、Gemma 3/4、Qwen 3.5/3.6-family、GPT OSS、Nemotron-H、GLM 5.x、Muse-Glimmer——因为只有 2 个 KV 头，并行度上限为 `--tp 2`），并针对 MoE 专家并行 / 专家切分、GatedDeltaNet 按 rank V-head 归属、Mamba2 复制等异构层提供各自的策略。融合的按 rank 计算图使 `--tp 2` 的 decode 快于单卡（Gemma 4 E4B 51.7 对 37.3 tok/s），也让单卡装不下的模型得以运行。注意 TP 并不是模型用上多张 GPU 的唯一途径：DeepSeek V4 与 GLM 5.x **不加任何开关就会按层切分到所有可见 GPU**（它们的整模型执行器会按每张卡的空闲显存对整层做装箱），`--tp` 在 GLM 5.x 上会把这种切分换成层内部的 Megatron 切分，在 DeepSeek V4 上则只是限制按层切分使用几张卡（等同 `TS_DSV4_NGPU`）；其他所有架构在不加 `--tp` 时只用一张 GPU。服务端还可选用 Redis 支撑的共享 KV 缓存与 Responses API 存储。→ [张量并行](USAGE_zh-cn.md#张量并行与分布式推理)
- **批处理 / 并行推理** —— 已为 Mistral 3、Gemma 4、GPT OSS、Qwen 3、Qwen 3.5/3.6-family、Nemotron-H 默认启用 `IBatchedPagedModel.ForwardBatch`，能在一次前向传播中打包 N 个序列，使用 `slotMapping` 进行分页 K/V 写入，并通过原生内核做按序列注意力。Gemma 4、Qwen 3.5/3.6、GPT OSS 与 Nemotron-H 提供各自的 `TS_<FAMILY>_BATCHED=0` 兜底开关；Qwen 3 与 Mistral 3 没有家族专属开关，请用全局 `TS_SCHED_DISABLE_BATCHED=1` 强制回到按序列 KV-swap 路径。GLM 5.x 没有分页版 `ForwardBatch`；取而代之的是一条可选的批处理融合解码（`TS_BATCHED_FUSED_DECODE=1`）：一张图、每个序列一个 token，权重只读一次——4 路并发下总解码吞吐 1.81 倍。默认关闭，因为批处理会改变 GEMM 形状，而 2-bit MoE 会把这点差异放大成不同的专家选择。
- **兼容 Ollama 与 OpenAI API** —— 可作为现有工具链的即插即用替代端点
- **可配置采样** —— temperature、top-k、top-p、min-p、重复/存在/频率惩罚、seed、停止序列
- **结构化输出** —— OpenAI `response_format` 中的 JSON schema 会被编译成语法，并通过语法约束解码强制执行：任何会破坏 schema 的 token 在采样前就被从分布中剔除，因此返回值天然结构合法，而不是事后修补。支持 `type`、`enum`、`const`、`properties`、`required`、`additionalProperties`、`items`、`prefixItems`、`min/maxItems`、`anyOf`、`oneOf`、`allOf`、`$ref`/`$defs`（含递归）、`min/maxLength`、`pattern`，以及 date/time/date-time/uuid 格式与整数 `minimum`/`maximum`。CFG 无法表达的关键字（`not`、`if`/`then`/`else`、`dependentSchemas`、`dependentRequired`、`multipleOf`、`patternProperties`）会在请求阶段直接拒绝。`TS_JSON_GRAMMAR=0` 回退到旧的提示 + 修补行为。
- **聊天模板** —— 从 GGUF 元数据自动加载（Jinja2），并为不同架构提供硬编码回退模板
- **推理引擎** —— `TensorSharp.Server` 中的新 `InferenceEngine`（工作线程调度器 + 分页块池）取代了旧的单请求 FIFO 队列。旧队列对象现在只是状态 / 事件形状的兼容 shim；引擎本身已经处理并发。
- **批处理** —— 控制台应用支持 JSONL 输入，并内置用于测量 prefill / decode 吞吐的推理基准
- **流式输出** —— 按 token 输出（Web 通过 SSE，控制台通过 stdout），并支持中断/停止正在生成的请求
- **文本扩散生成** —— DiffusionGemma 使用 EntropyBound 迭代去噪采样器，而不是自回归 `Forward()`。CLI 提供 `--diffusion-steps`、`--diffusion-seed` 与 `--diffusion-blocks`；Web UI 使用整条消息 `replace` 事件展示实时去噪预览，并通过 `DiffusionBatchScheduler` 批处理并发扩散请求。
- **图像编辑（Qwen-Image-Edit）** —— 提示词加输入图像生成编辑后的图像。所加载的 `qwen_image` GGUF 是 MMDiT 扩散 Transformer；TensorSharp 在其旁解析两个伴随 GGUF——Qwen-Image VAE（图像 ↔ 16 通道潜变量）与 Qwen2.5-VL-7B 文本编码器（提示词 → 3584 维条件，可选通过 `mmproj` 做视觉接地）。流水线对参考图做 VAE 编码、构建文本（及可选图像）条件、运行带参考潜变量拼接的 FlowMatch-Euler true-CFG 去噪循环，再 VAE 解码回像素。整个 60 块 DiT 前向被 CUDA 图捕获（`TSGgml_QwenImageForward`），flash 注意力默认开启，目标面积按设备 VRAM 预算自动钳制。可选的 Lightning 蒸馏 LoRA（`--qwen-image-lora` / `TS_QWEN_IMAGE_LORA`，`.safetensors`）把默认的 30 步 / CFG 2.5（共 60 次 DiT 前向）降为该 LoRA 自带的步数（例如 4 或 8，从文件名解析）且 CFG 1.0、无负向分支。它以运行期 F32 旁路的形式接在每个目标投影旁（`y = W_quant*x + b + (alpha/rank)*up*(down*x)`），量化基权重原样保留，**不会**被合并：Lightning 的增量 RMS 约 1e-4，远低于一个 Q2_K 量化步长，实测合并会让速度场产生 24% 的 relL2 变化，而那全是重量化噪声。该旁路额外开销约 4% FLOPs，可安全参与 CUDA 图捕获，并且要求整模型或融合分块的 CUDA 前向路径——若落到无法承载旁路的路径上，模型会直接报错而不是输出噪声。整步去噪缓存（`TS_QWEN_DIT_CACHE_MODE`：`easycache` 可跳过 40–55% 的步骤，`fbc` 为 First-Block-Cache）默认**关闭**，因为在编辑类任务上会明显柔化人脸细节。在本项目 CUDA `image_edit` 场景下与 stable-diffusion.cpp 对比（Q2_K DiT + 4 步 Lightning LoRA、544x1184、相同输入与种子）：热启动 40.44 秒 对 48.16 秒。可从 C# 通过 `QwenImageModel.EditImage(prompt, RgbImage, QwenImageParams)` 驱动，从 CLI 图像编辑模式（`--image`、`--prompt`、`--cfg`、`--diffusion-steps`、`--diffusion-seed`）驱动，以及从带实时去噪预览的 Web UI 驱动。→ [Qwen-Image-Edit 卡片](docs/models/qwenimage_zh-cn.md)
- **音视频联合生成（MiniMax-H3）** —— 提示词生成一段 H.264 MP4，**并同时生成原生 32 kHz 立体声音轨**：同一个扩散 Transformer 在单条 token 序列里对打包好的视频+音频潜变量一起去噪，因此音轨是模型输出的一部分，而不是事后配上去的。最长 15 秒、24 fps。TensorSharp 把它跑成原生的整网络 ggml 计算图——每个网络一张图，权重直接从 GGUF / safetensors 的 mmap 常驻绑定：Qwen3-VL-32B 文本编码器（`TSGgml_MiniMaxH3TextEncode`）、它的视觉塔（`TSGgml_MiniMaxH3VisionEncode`）、DiT 单步（`TSGgml_MiniMaxH3DitForward`），以及视频 / 音频 VAE 的编码与解码。DiT 有 50 个块、约 193 亿参数，单流且**没有交叉注意力**——文本、条件帧、目标音频与目标视频是**同一条**序列，在全双向注意力下一起计算（hidden 5376，56 头 × 128 = 7168 内维，patch (t, h, w) = (1, 2, 2)）；AdaLN 不跑时间步 MLP，而是在一张学出来的 `[8, 1025]` 曲线表上插值；3 轴 RoPE 使用**连续浮点**位置，把两条流放在同一条以音频潜变量单位（1/40 秒）计量的时间轴上。两个 checkpoint 是两个文件而不是两个开关：`minimax_h3_fl2va_pruned-Q4_K.gguf` 接受文本与关键帧（`--video-mode t2v`、`i2v`——图片**就是**首帧并被动画化——或 `fl2v`，同时钉住首帧与尾帧），`minimax_h3_ref2va_pruned-Q4_K.gguf` 接受用于**全新场景**的身份 / 外观参考（`--video-mode ref`：最多九张 `--ref-image`，另有 `--ref-video` / `--ref-video-audio` / `--ref-audio`，在提示词里按位置以 `<Picture 1>`、`<Video 1>`、`<Audio 1>` 引用）；向其中一个 checkpoint 请求另一个的模式会直接报错并点名该加载哪个文件，而不是默默丢掉输入。它出厂就是 **CFG 蒸馏**的，因此必须 `--cfg 1.0`（更高的值 TensorSharp 会拒绝，`--negative-prompt` 也因此无效——根本不跑无条件分支），`--diffusion-steps` 的快速工作点是 4–8 步，而默认是 20 步。`--video-frames` 会向上对齐到 `17k+5` 网格（5、22、39、56、73、90……），`--width`/`--height` 向上对齐到 32 的倍数，fps 无论请求什么都被钉在 24。视频 VAE 每次解码 5 个潜变量帧、带 2 帧前瞻并对接缝做交叉淡化，并按 256 px 分块——这两件事都是**正确性**要求而非优化，因为它的解码器是一个纯 36 层 Transformer，其 RoPE 坐标是按交给它的那段范围做长度归一化的。长片段还需要一处修复：ggml 的 flash-attention 内核把 softmax 分子保持在 FP16 且只留三位余量，而 H3 是对整段片段做双向注意力（640×384 下 22 帧是 2364 个打包 token，107 帧则是 8646 个），因此 `h3_attend` 会按 key 数量取一个 2 的幂预先缩放 V、再在输出上还原——因为注意力对 V 是线性的，这一步是精确的——这正是把一段全黑的 107 帧片段变回正常的原因；采样器还会拒绝保存已经发散的样本，在潜变量变成非有限值的那一步直接让请求失败，而不是写出一个黑文件（`TS_H3_TRACE=1` 打印逐步的幅值）。在 M5 Pro 上实测（`ggml_metal`、22 帧、8 步、相同种子），对比 stable-diffusion.cpp 的最佳配置：256×256 为 **20.9 秒对 49.3 秒（快 2.4×）**，640×384 为 **63.1 秒对 108.5 秒（快 1.7×）**；而在 16 GB 的 RTX 3080 Laptop 上（`ggml_cuda`，相同工作负载，stable-diffusion.cpp 使用 `--auto-fit --stream-layers --diffusion-fa --rng cpu`，因为它默认的 `--offload-to-cpu` 路径在这台机器上跑不动该模型），端到端结果反过来——256×256 为 43.6 秒对 37.8 秒，640×384 为 63.7 秒对 59.8 秒，stable-diffusion.cpp 领先 1.15× / 1.07×——但*逐去噪步*的开销仍然是 TensorSharp 更低（按 8 步与 16 步的斜率推算为 3.325 秒对 3.338 秒）：差距来自固定的启动开销，这台机器只有 16 GB 显存与 31.7 GB 内存，却要面对 33.5 GB 的模型集合；而 640×384 那 3.9 秒差距里约有 3 秒是 H.264 编码（sd.cpp 写的是 MJPEG+PCM 的 AVI）加上 .NET 进程启动（对手是原生可执行文件），并不是推理。在这种显存规模上，流水线会主动管理驻留而不是假定权重放得下：当去噪器与视频 VAE 无法同时容纳时，会先交还已用完的去噪器的设备副本再加载 VAE（解码期间显存峰值 16 041 → 约 5 600 MiB，在 640×384 上值 22 秒——在 Windows/WDDM 上超额分配并不会失败，而是被悄悄地用主机内存兜底，于是整段解码以 PCIe 速度运行）；同时在文本主干产出隐藏状态的那一刻就开始顺序预读去噪器 GGUF，并与它自己的上传流水化而不是先等它读完——因为权重是以指针形式绑定在 mmap 上的，否则首次上传会在主机到设备的拷贝*内部*逐页缺页换入，只有 0.91 GB/s，而页面常驻后是 5.97 GB/s。预读开与关的输出逐字节一致；首个去噪步从 14.87 秒降到约 10.2 秒，整轮从 89.0 秒降到 **63.7 秒**（640×384），256×256 从 67.2 秒降到 **43.6 秒**。`TS_H3_PREFAULT` 选择模式（`0` 关闭，`1` 串行，`2` 与文本条件计算重叠，`3` 与上传流水化——默认值；模式 `2` 更差，因为文本编码器会把自己的 17 GB 也冲过同一份页缓存，正好把刚放进去的页面挤出来），`TS_H3_PREFAULT_THREADS` 选择读取流数（默认 `1`；4 流与 16 流实测都更慢，因为这次预读是和它要预热的拆解与上传并发进行的）。`TS_H3_PHASE=1` 打印分阶段耗时——文本编码器打开 / 主干 / 拆解、预读、每一个去噪步、VAE 打开 / 解码；`TS_H3_TE_GROUP=<n>` 把 50 层文本编码器主干按每组 `n` 层执行并逐组释放设备副本，默认**关闭**：它确实消除了编码器自身的溢出（峰值 16 041 → 12 981 MiB）且逐位一致，却仍然慢 3 秒——十来个 token 的一次性 prefill 对每个权重只读一次，溢出的约 1.3 GB 只多付一次 PCIe 传输，而分组仍然要搬完整的 17 GB，还额外增加了分配 / 失效的开销。每个网络都是对着参考实现校验的，而不是自己跟自己比——文本编码器 cos 0.999999，DiT 单步视频 cos 0.9983 / 音频 cos 0.9998，视频 VAE 编码与解码 cos 1.000000，音频 VAE 解码 cos 0.999995。可从 C# 通过 `MiniMaxH3Model.GenerateVideo(prompt, VideoGenerationParams)` 驱动，从 CLI（`--prompt`、`--image`、`--end-image`、`--ref-image`、`--video-mode`、`--width`/`--height`、`--video-frames`、`--diffusion-steps`、`--cfg`、`--audio-vae`、`--no-audio`）驱动，也可从服务器 API 驱动——`/api/video-generate[/stream]` 与 `/v1/videos/generations` 接受 `videoMode`、`endImage`、`referenceImages` / `referenceVideos` / `referenceAudios` / `referenceVideoAudios` 与 `generateAudio`，并在 MP4 之外返回 `audioUrl` / `audio_url`；`GET /api/models` 则报告当前 checkpoint 能接受什么（`video.family` = `minimax-h3`、`supportsAudio`、`supportsEndImageConditioning`、`supportsReferenceConditioning`、`maxReferenceImages`），Web UI 据此只放出真正可用的附件控件。音轨作为**旁挂 `.wav`** 写在 MP4 旁边而不是混流进去，因为混流需要一个不能假定存在的编码器——`ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4`。→ [MiniMax-H3 卡片](docs/models/minimax-h3_zh-cn.md)
- **视频生成，仅视频（Wan 2.1 文生视频，Wan 2.2 文/图生视频）** —— 提示词（Wan 2.2 模型可再加一张首帧图片）生成 H.264 MP4 视频（无音轨）。所加载的 `wan` GGUF 是 Wan DiT —— 自动识别 Wan 2.1 T2V、Wan 2.2 TI2V-5B（48 通道 16×16×4 潜空间、24 fps）与 Wan 2.2 A14B（两个 14B 专家按时间步边界切换，第二个 GGUF 自动配对）；TensorSharp 在其旁解析伴随模型——UMT5-XXL 文本编码器 GGUF（提示词 → 512×4096 条件，精确的 unigram-Viterbi SentencePiece 分词）与对应的因果 3D 视频 VAE（`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`）。FlowMatch CFG 去噪（UniPC 或 Euler）每步将整个 DiT（带 3D RoPE + flash 注意力的自注意力、交叉注意力、AdaLN 时间调制——TI2V 图生视频为逐 token 时间步）作为单个常驻权重 ggml 图运行，按形状 CUDA 图捕获（`TSGgml_WanDitForward`）；视频 VAE 在单个图内解码全部时序块（`TSGgml_WanVaeDecode`）——Metal 上卷积走 MPSGraph（736x544x81f 的一次解码从 159 秒降到 80 秒，1.99×，数值不变，PSNR 93.9 dB；`TS_WAN_VAE_MPS_CONV=0` 可恢复 ggml 的 im2col+GEMM 下降路径），其他后端走带状 im2col+GEMM；im2col 预算与分块阈值现在按设备可用显存推导，而不再固定按 16 GB 显卡的预算，因此大显存设备可整幅解码 720p 平面（565 秒 / 峰值 RSS 4.85 GB，对比分成两带的 655 秒 / 5.37 GB），小显存设备仍然分块。图生视频的首帧经因果 VAE 编码器单图编码（`TSGgml_WanVaeEncode`）。各阶段在进入下一阶段前释放各自 VRAM，因此 TI2V-5B 81 帧 480p 图生视频与两个 A14B Q4_K_M 专家均可在 16 GB GPU 上运行。**步数蒸馏检查点会按 DiT 文件名自动识别**（`Turbo`、`distill`、`Lightning`、`lightx2v`、`FastWan`、`-dmd`，或显式的 `…-4steps-…`），这是最大的提速手段：官方 50 步 × CFG 配方需要 100 次 DiT 前向，而 4 步蒸馏检查点只需 4 次，管线会自动切换到该步数并关闭引导（`--diffusion-steps` / `--cfg` 可覆盖）。在 M5 Pro、`ggml_metal`、Wan2.2-TI2V-5B Q8_0、1088×832×121f = 27 404 token 上实测：基础检查点 100 次前向、每次 120.2 秒，端到端约 3 小时 30 分；同一请求换成 Turbo 检查点只需 4 次前向，端到端 **17 分 30 秒**——只有 `--model` 路径不同。基础检查点上还可用 `--cfg-cache-stride 2` / `3` 复用引导方向，再快 1.30× / 1.43×。数值已对照 diffusers 验证（DiT 余弦 > 0.995，VAE 编码器 > 0.999，解码器 59.9 dB / >35 dB PSNR）；让单次 27k token 自注意力快 2.02×（连同 VAE 的改动，每次 DiT 前向约 1.7×）的 F16 注意力键值，其 DiT 余弦与 F32 同为 0.999964。可从 C# 通过 `WanVideoModel.GenerateVideo(prompt, WanVideoParams)`、CLI（`--prompt`、`--image`、`--video-frames`、`--fps`、`--flow-shift`、`--negative-prompt`）、服务器 API（`/v1/videos/generations` 支持 base64 `image`，`/api/video-generate[/stream]` 支持 `imagePath`）以及 Web UI 聊天（输入提示词——附图即为图生视频——获得带实时进度的视频）驱动。→ [Wan 卡片](docs/models/wan_zh-cn.md)
- **混合 SSM-Transformer** —— Nemotron-H 在单个模型中混合 Mamba2 SSM 层、纯注意力层和 MoE FFN 层；Mamba2 步现在同时提供单序列原生内核与批处理原生内核（`TSGgml_NemotronMamba2BatchedStepF32`，NEON SIMD + GCD 并行）。在 GGML 后端上，注意力层直接用设备侧 flash-attention 内核对常驻 KV 缓存做 decode（`TS_NEMOTRON_FLASH_DECODE=0` 恢复主机路径），decode 速度不再随上下文长度衰减。
- **混合注意力-递归网络** —— Qwen 3.5/3.6-family 在同一模型中混合全注意力层与 GatedDeltaNet 递归层；批处理路径下递归运行状态保存在每槽位的递归状态池中
- **专家混合（MoE）** —— 支持 Gemma 4 MoE 变体（例如 gemma-4-26B-A4B）、GPT OSS MoE（例如 gpt-oss-20b）、Qwen 3.5/3.6-family MoE（`qwen35moe` / `qwen3next` 变体，例如 Qwen3.5-35B-A3B）、Nemotron-H MoE FFN 层，以及 GLM 5.2（744B-A40B：256 个路由专家 top-8 加 1 个共享专家，sigmoid 门控路由带一个只影响选择的 bias 与 x2.5 的路由缩放，前面还有 3 个稠密 SwiGLU 层）
- **MoE 专家 CPU 卸载** —— `--n-cpu-moe N` / `--cpu-moe`（对应 llama.cpp 的 `-ncmoe` / `-cmoe`，环境变量 `TS_N_CPU_MOE`）把前 N 层的路由专家权重留在系统内存并在主机侧相乘，注意力、各处 norm、router 与常驻共享专家仍留在加速器上。在所有具备整模型融合图的架构（Qwen 3.5/3.6、Gemma 4 MoE、GPT OSS、DiffusionGemma）上，被卸载的层仍留在同一张融合图内——加速器在每个被卸载层的 router 之后暂停，主机直接从 GGUF mmap 中取出被选中的专家做乘法，再把结果交回下一段，因此 decode 时每层只有约 8 KB 激活跨总线。它同样能与张量并行组合：`--tp N` 下这些接缝会并入各 rank 的 AllReduce 分段计划（Qwen3.5-35B-A3B `--tp 2`：两卡上 17.4 GB 常驻权重降到 3.2 GB；gemma-4-26B-A4B：12.9 GB 降到 2.4 GB，输出逐字节一致）。在 16 GB 的 RTX 3080 Laptop 上实测：Qwen3.6-35B-A3B `--cpu-moe` 后显存 13.4 → 4.6 GB；gemma-4-26B-A4B 16.1 → 4.8 GB（decode 39.7 → 17.7 tok/s，若只用 `--n-cpu-moe 8` 则为 38.6 tok/s 并让出 3 GB）；gpt-oss-20b 16.2 → 2.9 GB，从而避开 WDDM 溢出悬崖，`--n-cpu-moe 12` 时把 0.3 tok/s 变成 25.4。所有架构（含 DeepSeek V4 Flash）默认都是 0：装不下的模型会在加载时直接拒绝并给出所需的 `--n-cpu-moe N`，而不是悄悄牺牲 decode 吞吐。GLM 5.x 走同一条路径（该 checkpoint 92% 的字节是路由专家），并且主机常驻的专家直接由 GGUF 映射提供，不做私有拷贝。要记住 offload 是为了"装得下"而不是为了快：在 GLM-5.2 本来就放得下的 3× RTX PRO 6000 上，`--n-cpu-moe 30` 会把 pp2048 从 915.9 拉到 94.7、tg64 从 43.9 拉到 16.4 tok/s。→ [MoE CPU 卸载（英文）](USAGE.md#mixture-of-experts-cpu-offload---n-cpu-moe) GLM 5.2 在它的原生 `glm-dsa` 执行器上使用同一条接缝：`--cpu-moe` / `--n-cpu-moe N` 把路由专家留在系统内存里、直接从 GGUF 映射中取用做乘法（不做私有拷贝），并且能与 `--tp N` 组合——驻留主机的层保留完整专家，由 rank 0 求值。在 3x RTX PRO 6000 上，`--n-cpu-moe 30` 用吞吐换空间（pp2048 94.7 / tg64 16.4，对比全部常驻时的 915.9 / 43.9），把加载器能定下的上下文从 342,272 抬到 646,400 token，几乎翻倍。
- **批量 GPU MoE** —— Qwen 3.5/3.6-family 与 Nemotron-H 在 decode 时通过单次融合的 GGML 计算图调度处理所有被选中的专家（Qwen 3.5-family 还包括可选的 shared expert 与残差加法），消除每个专家的 CPU-GPU 往返
- **整模型融合 decode 计算图** —— Gemma 4（dense 与 MoE）、Qwen 3.5/3.6 与 GPT OSS 把一个 decode token 的全部计算——每一层、MoE 路由与专家、最终 norm 与 LM head——作为**一次** GGML 计算图调度提交，而不是每层提交一次，GPU 因此不会在层与层之间空等主机。在 CUDA/Vulkan 上该图只构建一次、张量地址保持稳定后反复重放（KV 写入用 `ggml_set_rows`、行号作为 I64 输入，注意力窗口按 stride 补齐、掩码作为 F16 输入），这正是 ggml-cuda 能把它捕获成 CUDA 图的前提。GPT OSS decode 在 A40 上从 24 → 154 tok/s，且随上下文长度基本持平（16K 时仍有 133 tok/s，而逐层路径已跌到 2.3）。可按模型用 `TS_GPTOSS_MODEL_DECODE=0` / `TS_GEMMA4_FD_PERSIST=0` / `TS_QWEN35_FD_PERSIST=0` 关闭。
- **KV 缓存编解码器** —— 通过 `IKvBlockCodec` 接口插件化；内置 TurboQuant（2-bit 仿射 / Q4 / Q8）分页块压缩。CLI 的 `--paged-kv-quant-bits` 接受 `0|2|4|8`；服务端旧式独立分页参数接受 `0|4|8`，也可直接用 `TS_KV_PAGED_QUANT_BITS=2` 选择 2-bit 编解码器。2-bit 档位在 fp32 块上可达约 10 倍压缩，面向超长上下文。
- **KV 缓存精度** —— `--kv-cache-dtype <f32|f16|q8_0|q4_0>`（CLI 与服务端，环境变量 `KV_CACHE_DTYPE`；默认 auto，由后端 / 模型决定）用很小的数值漂移换内存。`q4_0`（约 0.56 字节/元素，约为 f32 的 1/7）是最激进的档位，面向 KV 缓存主导内存的超长上下文（128K–256K）；块量化档位（`q8_0`/`q4_0`）需要原生 GGML flash 路径。
- **消息编辑** —— 在 Web 聊天界面中编辑或删除历史消息，并从该位置重新生成回复
- **文本/图像/音频/视频/PDF 上传** —— Web 界面支持最大 500 MB 的文件上传并完整保留文本内容；原生数字 PDF 会完整提取文本层（可通过 `TS_PDF_MAX_PAGES` 显式限制页数）。最终提示词按模型的实际上下文窗口检查，而不是使用任意的上传预算
- **每轮可观测性** —— 结构化日志会完整保留用户输入与模型原始输出（包括 `<think>` 思维链和最终结果），并记录 KV 缓存命中率。同样的命中率指标通过所有 API 透出：Ollama 的 `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio`、OpenAI 的 `usage.prompt_tokens_details.cached_tokens`，以及 Web UI SSE `done` 事件中的 `promptTokens` / `kvReusedTokens` / `kvReusePercent`


## 思维链 / 推理模式

支持思维链模式的模型（Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H、DeepSeek V4、GLM 5.x）可以在生成最终答案之前产出结构化的思维链推理内容。思维内容与主要回复分开，客户端可选择显示或隐藏。

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H：** 使用 `<think>...</think>` 标签
- **Gemma 4：** 使用 `<|channel>thought\n...<channel|>` 标签
- **GPT OSS：** 使用 Harmony 格式，以 `<|channel|>analysis` 标记思维过程，以 `<|channel|>final` 标记最终回复
- **DeepSeek V4：** 使用 `<think>...</think>` 标签；不传 `--think` 时聊天模板会直接闭合该块，因此推理是显式开启的
- **GLM 5.x：** 同样是 `<think>...</think>`，也与其他系列一样按需开启——加 `--think` 会补上 `Reasoning Effort: Max` 系统行，并在生成提示里留下一个未闭合的 `<think>` 由模型自己收尾；不加时提示里写的是空的 `<think></think>`，模型于是直接作答。历史轮次的思考内容始终不会带进提示，与模板 `clear_thinking` 的默认行为一致

通过 `--think`（控制台）、`"think": true`（Ollama API）或 Web 界面中的思维链开关启用。

## DSpark 块级投机解码（DeepSeek V4）

DeepSeek V4 的 checkpoint 中随模型附带一个 **DSpark** 支持模块（"Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"）：三个 DSV4 块读取主干的 hidden states，每步提议**一整块** token 而不是一个；一个 Markov 头让块内每个位置以其前一个 token 为条件；一个置信度头预测每个位置被接受的概率。TensorSharp 把它作为独立的草稿 GGUF 加载（`--draft-model`，可用 `eng/dsv4-dspark-to-gguf.py` 自行转换），并在两个 GPU 引擎（`--backend cuda` 与 `--backend ggml_cuda`）上为贪心单序列生成启用——在 ggml 上草稿器就是计算图里额外的三层，其 key ring 由主干图自己提交，因此投机不产生任何主机往返；主干用一次批量前向验证整块，只保留它本来也会产生的前缀。在 4×A40 上以默认累积置信度门限测得 **decode 提速 1.3–1.4×**；这个门限很关键，因为验证批中每多一行都要把一整套 MoE 专家重新拉过显存。详见 [DeepSeek V4 卡片](docs/models/deepseek4_zh-cn.md#dspark-投机解码)。

## DFlash 块级投机解码（Muse-Glimmer）

Muse-Glimmer 有自己的块级草稿模型 **DFlash**：一个独立的 5 层 GGUF（`general.architecture = dflash`），一次前向即提出整个投机窗口。它复用主干的 token embedding 与 LM head，维护自己的滑动窗口 KV 环，每步跑三遍——把主干在 `dflash.target_layers` 处的逐层输入残差*编码*成一行宽向量，将该行*注入*为每个草稿层的 K/V，再把 `[anchor, MASK x (block-1)]` 送过 5 个块*起草*，并用主干的 LM head 打分。主干用一次批量前向验证该块，只保留它自己本来也会产生的前缀，因此**输出的 token 流与普通贪心 decode 完全一致**。

两部分都是可被 CUDA 图捕获并重放的融合原生图，草稿块以设备端 `argmax` 收尾，因此 202048 宽的概率块无需经 PCIe 回传。运行期的成本调控器会把投机与普通解码逐 token 计时对比，在投机确实更慢时暂停起草——所以投机只会更快，但需要生成数百个 token 才能稳定。用 `--draft-model`（CLI）或 `TS_MUSE_GLIMMER_DFLASH` 加载。采样是可组合的——验证会用本次运行自己的采样器从主干的某一行抽取每个 token——但要注意块级草稿器一次提出整块草稿，惩罚项并不会施加到该提案上，接受率会随带惩罚的历史增长而下降。在单张 RTX PRO 6000 Blackwell 上与 llama.cpp 自带的 DFlash 对比（Q8_0、贪心、60 token 提示）：50.9 tok/s，对比 llama.cpp 的 45.5 与普通解码的 35.0。详见 [Muse-Glimmer 卡片](docs/models/muse-glimmer_zh-cn.md)。

## 投机解码

草稿器廉价地提议若干未来 token，主干用一次批量前向验证全部 token，被接受的 token 一步提交。由于起草与验证都由该请求自己的采样器（temperature、top-k/p、重复/存在/频率惩罚）驱动，输出与标准 decode 完全一致——投机只改变产生这些 token 所需的前向次数。只对单序列（无并发）请求生效。

**整套设计分为三个互不耦合的层次**——*模型架构* ≠ *投机算法* ≠ *草稿器权重*——因此新模型可以复用全部算法，新算法也可以复用全部模型。模型实现 `ISpeculativeTarget`（多行前向 + 逐行 logits，外加 KV 回滚三件套）；算法实现 `ISpeculator`（`Propose` / `Commit`）并按名字注册；checkpoint 自带的训练好的草稿器则藏在 `IDraftHead` 这个模型专属的薄适配层之后，因为那些权重绑定在某一个目标模型上、无法迁移。要加 EAGLE、Medusa、PARD 或树状起草，只需一个新类加一次 `SpeculatorRegistry.Register` 调用——模型、执行器与调度器的代码都不用动。详见 [TensorSharp 中的投机解码（英文）](docs/speculative_decoding.md)。

目前随包提供四种算法，用 `--spec-type` 选择：

| `--spec-type` | 起草方式 | 需要训练好的权重？ |
|---|---|---|
| `auto` *（默认）* | checkpoint 自带哪种草稿器就用哪种 | — |
| `draft-head` | 每次前向出一个 token，走 NextN/MTP 头并把自己的 hidden state 串下去（Qwen 3.6、GLM 5.2、Gemma 4 的独立 assistant GGUF） | 是，且与目标模型一一对应 |
| `block` | 每次前向出一整块，配一个置信度头（DeepSeek V4 DSpark、Muse-Glimmer DFlash） | 是，且与目标模型一一对应 |
| `ngram` | 在序列自己的 token 上做后缀匹配——这几个 token 之前出现在哪里、后面跟的是什么？ | **否** |

`ngram` 是与模型无关的那一个：它对**每一个** checkpoint 都可用，包括完全不带草稿器的模型；在答案会引用输入的场景下最强——摘要、编辑、翻译、就文档作答、重复性的结构化输出、含重复标识符的代码、智能体的工具循环。在不带草稿头的 Qwen3.5-9B 上实测（Q8_0、ggml_metal、M5 Pro）：一条"复现这份配置"的提示词跑出 **45.2 tok/s，对比普通 decode 的 31.4（1.44×）**，输出逐字节一致。在自由散文上它找不到可用后缀，每一步都退化成普通 decode，而运行期的成本调控器会让这件事保持廉价。

投机解码**默认关闭**。在服务端或 `TensorSharp.Cli` 上用 `--spec`（环境变量 `TS_SPEC=1`）启用。历史拼法 `--mtp-spec` / `--mtp-draft` / `--mtp-pmin` / `--mtp-draft-model` 原样保留，环境变量也同时以 `TS_SPEC_*` 与 `TS_MTP_*` 两套名字发布——glm-dsa 的原生加载器会在加载模型时从 C++ 侧读取 `TS_MTP_SPEC` / `TS_MTP_DRAFT`，所以那套名字是一份跨语言契约：

```bash
# Qwen 3.6 —— 使用 -MTP- 仓库 GGUF，确保主干保留内嵌 NextN 块
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft 8 --mtp-pmin 0.75

# Gemma 4 —— 加载与目标匹配的独立 gemma4-assistant 草稿 GGUF
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

**三种草稿头形态：**

- **Qwen 3.6（内嵌 NextN）** —— GGUF 在主干栈之后带有一个额外解码块（`{arch}.nextn_predict_layers`）以及 NextN 投影 / 归一化张量。无需独立文件，`--mtp-draft-model` 被忽略。主干的递归状态（GatedDeltaNet）会被快照，以便部分被拒的验证批次可以回滚。
- **GLM 5.2（内嵌 NextN）** —— 形态相同，且官方 [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF) 已经带有该块（`blk.78.nextn.*` 加上一个完整的 MLA + 256 专家解码块），无需额外下载，`--mtp-spec` 就是全部配置——在 CLI（`--input`、`--multi-turn-jsonl`、`--interactive`）上与在服务端上都是如此。该块只在传入该参数时才会加载：它是一整个解码层（IQ2_XXS 下约 3 GiB），会与 KV 缓存争抢 loader 用来确定上下文长度的同一块显存。glm-dsa 没有递归状态，因此部分被拒的验证批次会保留已接受前缀的 KV，只回退位置计数，不需要重跑。详见 [GLM 卡片](docs/models/glm_zh-cn.md#nextn--mtp-投机解码)。
- **Gemma 4（独立 `gemma4-assistant` GGUF）** —— 通过 `--mtp-draft-model` 加载的 EAGLE 风格递归草稿器。它自身不保存任何 K/V：每个草稿层都查询**目标模型**已有的逐层 KV 缓存（最后一个 local 层 + 最后一个 global 层），因此在给定 `(token, hidden)` 时草稿器是无状态的。草稿的隐藏维度必须与目标一致——12B 目标配 12B 草稿，而非 26B-A4B 草稿。草稿 GGUF 不匹配、缺失或不完整会在启动时**立即失败**并给出修复提示，而非静默关闭投机。

**何处有收益**（自动启用；否则引擎走标准 decode）：

| 后端 | Qwen 3.6 | GLM 5.2 | Gemma 4 |
|---|---|---|---|
| GGML CUDA / GGML Metal | ✅ 融合多 token 验证 + 草稿步内核 | ✅ 原生整模型执行器中每个验证窗口一张图 | ✅ 融合多 token 验证 + 草稿步内核 |
| Direct CUDA（`cuda`，Driver API / cuBLAS） | ✅ 完全驻留 GPU 的逐算子验证 / 草稿 | —（GLM 的逐算子路径只在 `cpu` 上运行） | ✅ 完全驻留 GPU 的逐算子验证 / 草稿 |
| CPU / GGML CPU / MLX | 标准 decode（验证跟不上） | 逐算子参考实现（正确，但不快） | 标准 decode |

调优：`--spec-draft`（默认 `8`，别名 `--mtp-draft`）限制每步起草的 token 数；`--spec-pmin`（别名 `--mtp-pmin`）是置信度门限，遇到第一个低于该值的 token 即停止起草。这个数字*意味着什么*由算法自己决定，因此各算法带各自的默认值而不是共用一个：逐 token 草稿头是 `0.75`（在其 top-10 logits 上的 top-1 概率），块级草稿器是累计 `0.35`，n-gram 是 `0`（在那里它转而缩放所需的匹配长度）。这两个参数是相互作用的——窗口开得宽时偶尔会形成一条最终大部分被拒的长链，而那些验证行的开销照付不误——所以在新的模型 / 机器组合上值得把它们一起扫描，而不是分别调。在 GLM 5.2 上，`--spec-draft 4 --spec-pmin 0.55` 在每一轮实测中都是最好或并列最好，比默认值高约 4%。Gemma 4 草稿路径 A/B 开关为 `TS_GMTP_*` 环境变量（见 [Web 应用](USAGE_zh-cn.md#web-应用) 下的 **MTP / 投机解码调优变量** 表）。各架构具体机制见 [Qwen 3.5/3.6 卡片](docs/models/qwen35_zh-cn.md)、[GLM 卡片](docs/models/glm_zh-cn.md#nextn--mtp-投机解码) 与 [Gemma 4 卡片](docs/models/gemma4_zh-cn.md)。

**贪心输出与浮点。** 每个输出 token 都取自**主干**的某一行，因此投机不会改变 token 来自哪个分布，只改变得到它需要几次前向。但它确实改变了**算术**：K+1 行的验证让主干的矩阵乘运行在与 1 行 decode 不同的 batch 尺寸上，从而选中不同的 kernel 与归约顺序。在稠密模型上这不可见；在 GLM-5.2 上——2 bit 权重、256 专家 top-8——路由 logit 的最后一位差异会改变**实际激活哪些专家**，78 层会把它放大。对 140 个验证行与逐 token decode 的实测：**2.9%** 的行 top-1 token 不同，因此长贪心生成最终会走向另一条（同样合理的）分支。关闭起草后跑同一条投机代码路径可与贪心逐 token 一致，这正说明该效应来自 batch 尺寸而非投机本身。

## 张量并行与分布式推理

TensorSharp 支持**张量并行（TP）**——按 Megatron-LM 列/行并行范式把单个模型切
分到多张 GPU 上——以及**分布式（多节点）张量并行**，让 TP 跨越多台通过 TCP 点
对点网络互联的机器。

每个 transformer block 依次执行：列并行投影（QKV、gate/up，按输出头或中间维度
切分到各 GPU）→ 各 GPU 独立完成注意力或激活计算 → 行并行投影（output、down），
随后由一次 AllReduce 把隐藏状态重新汇聚。归一化层、词嵌入与 LM head 在各 rank
上复制。

**本地 TP** 在单个进程内运行。在 Direct `cuda` 后端上，由一个线程向所有 GPU 下发
命令，真正的并行由 CUDA stream 提供；在 GGML 后端上则由一个 rank 工作线程池并发
驱动各张 GPU——因为 GGML 的一次算子调用同时完成提交与同步。用 `--tp N`
（`TensorSharp.Cli` 与 `TensorSharp.Server` 均支持，或 `TENSORSHARP_TP_DEGREE=N`）
启用；`TENSORSHARP_TP_DEVICES=0,2` 可指定各 rank 对应的物理 GPU。

**分布式 TP** 通过点对点 TCP 网格跨机器扩展。每个节点运行自己的进程、管理自己
的本地 GPU；AllReduce 是分层的——先在节点内做本地 P2P 归约，再由各节点代表之间
走 TCP，最后广播回来——因此只有 `1/tp_local` 的数据需要穿过网络。用
`--tp-node-id` 与 `--tp-peers`（或 `TENSORSHARP_TP_NODE_ID` 与
`TENSORSHARP_TP_PEERS`）启用。服务端可以作为节点 `0` 加入这样的集群——即负责采样
并对外提供 HTTP 的 driver——其余节点各运行一个 `TensorSharp.Cli` worker。

针对异构层的架构专属策略：

| 架构 | 策略 |
|---|---|
| 稠密 transformer（Qwen 3、Mistral 3、Gemma 3） | 标准列/行并行 QKV + FFN |
| MoE（GPT OSS、Nemotron-H） | 专家切分——每张 GPU 持有每个专家权重的 `1/tp`；router 复制 |
| GGML 上的 MoE（Qwen 3.5/3.6） | 专家并行——整个专家在各 GPU 之间划分（每个 rank 拿 256 选 128），因此每个 rank 每个投影仍是一次批量 `ggml_mul_mat_id` 调度；shared expert 仍按 Megatron 方式切分 |
| GGML 上的 MoE（Gemma 4） | 在**每个专家内部**按 Megatron 方式切分（gate/up 列并行、down 行并行），使融合的整模 MoE 主干内核仍能使用全局专家 id；专家求和成为该层的第三个行并行 AllReduce 点。`TS_GEMMA4_TP_FUSED_MOE=0` 可回退到逐算子的整专家路径 |
| GatedDeltaNet SSM（Qwen 3.5/3.6） | 块循环 V-head 分配——各 rank 在自己的 V-head 子集上运行常驻本卡的打包 GDN 内核，delta/conv 状态相互独立；循环路径无需跨 rank 通信 |
| Mamba2 SSM（Nemotron-H） | 在 rank 0 上复制计算，结果广播给所有 rank |
| GGML 上的 MLA + 稀疏注意力 MoE（GLM 5.x） | 注意力头列并行（`attn_q_b` / `attn_k_b` / `attn_v_b`）配行并行 `attn_output`；256 个路由专家不是按专家 id 划分，而是在**每个专家内部**按 Megatron 方式切分（gate/up 列并行、down 行并行），因为 `ggml_mul_mat_id` 要求同一 token 选中的专家 id 互不相同。router、各处 norm、lightning indexer、共享专家与 3 个稠密层均为复制；每层两次 AllReduce。`TS_GLM_TP_SHARD` 选择切哪一半（1 注意力头、2 专家、3 两者都切），`TS_GLM_TP_OVERSUBSCRIBE=1` 可把多个 rank 挤在同一张卡上做测试 |

TP 可运行在 `cuda` 后端以及 GGML CUDA / Vulkan 后端（`ggml_cuda`、`ggml_vulkan`）上；MLX 为单设备。在 GGML 后端上，每个 rank 拥有自己 GPU 上的 ggml 后端、权重分片与 KV 缓存，跨 GPU AllReduce 走 ggml-cuda 的集合通信（可用时用 NCCL），小载荷则在主机内存中归约。**TP 下 CUDA 图捕获保持开启**——一个张量并行 token 是几十次按 rank 的小提交，重放它们值约 45% 的 decode 吞吐（4×A40：Qwen 3.5-9B `--tp 4` 从 88 → 128.5 tok/s，Qwen 3.5-35B-A3B `--tp 2` 从 71.3 → 104.1，后者正是 TP 输给还是赢过单卡的分界线）。用 `TS_GGML_TP_CUDA_GRAPHS=0` 关闭。集合通信的选择靠实测而非能力标志位：启动时该组会验证所宣称的设备对之间的 peer copy 是否真的把数据送到，以及一次真实的 NCCL AllReduce 能否完成，然后选出通过检验的最快传输。有些主机（常见于虚拟化云实例）宣称支持 peer access 却从不兑现，此时会保留 NCCL 集合通信但禁用 peer 传输，而不是干脆放弃它——这在超过两张卡时尤为重要，因为那里用不上 pinned-host 流水线，替代方案是每个层边界都经主机内存归约（4×A40 实测：Qwen 3.5-9B Q8_0 decode 53.5 → 75.1 tok/s）。GGML 上的 TP 同时带来**容量**与**延迟**收益：融合的按 rank block 计算图（注意力、稠密 FFN、MoE 主干、GatedDeltaNet）取代了逐算子前向，在 2× RTX 2000 Ada 上 `--tp 2` 的 decode 达到单卡的 **1.39×**（Gemma 4 E4B Q8_0，51.7 对 37.3 tok/s）与 **1.06×**（Qwen 3.5-9B Q8_0），且 Gemma 4 的输出与单卡逐字节一致；单卡装不下的模型则只能靠 TP 运行（Qwen 3.5-35B-A3B IQ4_XS 共 16.6 GB，拆到两张 16 GB 卡上，prefill 184 tok/s、decode 18 tok/s）。完整测量数据见 `TENSOR_PARALLELISM_PLAN.md`（Stage 1b 与 1c）。这能推广多远，取决于互连带宽以及一层里究竟有多少能拆：在没有 NVLink 的主机上，每层两次 AllReduce 会成为瓶颈——GLM-5.2 UD-IQ2_XXS 在 3× RTX PRO 6000（PCIe）上 `--tp 3` 为 pp2048 505.6 / tg64 17.6 tok/s，而单机按层切分是 915.9 / 43.9，并且每个 rank 都要各自持有一份全长缓存，能装下的上下文从 342,272 掉到 91,136 token。那里的 TP 是**容量**特性而非延迟特性；它还改变了归约顺序，因此在 2-bit MoE 上，对着录制的 llama.cpp 金标准，按层切分复现 5/6 条提示，而 `--tp 3` 只复现 3/6。TP 下的批处理 /
连续批处理前向目前实现于 Qwen 3 与 Mistral 3；MoE 模型在 TP 下回退到按序列前向。

本地集合通信优先使用 CUDA 点对点（P2P）DMA，但启动时会对每一对支持 P2P 的设备
做一次往返自检，任何回读数据损坏的设备对（在部分 L4 PCIe 拓扑上出现过）都会被
永久降级；因此在 P2P 不可用的主机（A16 vGPU 配置、大多数消费级显卡）上会自动改
走主机内存中转。诊断开关：`TENSORSHARP_TP_DISABLE_P2P=1`（所有跨 GPU 拷贝一律走
主机中转）与 `TENSORSHARP_TP_HOST_ALLREDUCE=1`（本地 AllReduce 在 CPU 上完成）。
多节点的连接与接收窗口分别由 `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS`（默认
120 秒）与 `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS`（默认 300 秒）控制。

服务端还支持可选的 **Redis 共享状态**：共享 KV 缓存层
（`--redis-url` / `TS_KV_CACHE_REDIS_URL`）用于跨会话复用 KV，以及 Redis 支撑的
Responses API 存储（`TS_RESPONSES_STORE_REDIS_URL`）用于持久化响应。

完整配置参考与示例：[用法 → 张量并行与分布式推理](USAGE_zh-cn.md#张量并行与分布式推理)。

## 工具调用 / 函数调用

模型可以调用用户定义的工具并参与多轮工具调用对话。将工具定义为 JSON 格式，通过 `--tools`（控制台）或 API 中的 `tools` 参数传入。

各架构使用各自的工具调用格式：

- **Qwen 3 / Nemotron-H：** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Qwen 3.5/3.6-family：** 同样是 `<tool_call>` 块，但内容为 XML —— `<function=NAME><parameter=key>value</parameter></function>`（JSON 形式仍然被接受）
- **Gemma 4：** `<|tool_call>call:function_name{args}<tool_call|>`
- **GPT OSS（Harmony）：** 工具以 TypeScript namespace 形式声明在 developer 消息中，调用通过 commentary channel 输出：`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`
- **DeepSeek V4：** DSML 标记 —— 系统提示词负责讲解语法并携带每个函数的 JSON schema，模型则以 `<｜DSML｜tool_calls><｜DSML｜invoke name="NAME"><｜DSML｜parameter name="key" string="true|false">value</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>` 作答。`string="false"` 表示该参数是 JSON 类型
- **GLM 5.x：** 逐参数标签的 XML —— `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`，函数名以裸文本紧跟在开标签之后，每个参数是自成一组的 `<arg_key>` / `<arg_value>`（模板里用 `tojson` 渲染过的值会被解析回数字、数组与对象）

输出解析器（`OutputParser.cs`）会自动从模型原始输出中提取工具调用，与架构无关。

## 多模态支持

### Gemma 4

Gemma 4 模型支持图像、视频和音频输入。上文 E4B 示例使用同仓库的 `mmproj-gemma-4-E4B-it-Q8_0.gguf`；请通过 `--mmproj` 显式传入（其他目标尺寸使用各自匹配的投影器）。

- **图像：** PNG、JPEG、HEIC/HEIF
- **视频：** MP4（使用 OpenCV 以 1 fps 基于时间抽帧；可通过 `VIDEO_SAMPLE_FPS` / `VIDEO_MAX_FRAMES` 调整）
- **音频：** WAV（16kHz 单声道）、MP3、OGG Vorbis

### Gemma 3

Gemma 3 支持 PNG、JPEG 与 HEIC/HEIF 图像输入。上文非 gated 示例使用 `mmproj-model-f16.gguf`；请通过 `--mmproj` 显式传入。

### Qwen 3.5 / 3.6 family

所有 Qwen 3.5/3.6-family 变体（`qwen35`、`qwen35moe` 与 `qwen3next`）共用同一个 `Qwen35Model` 实现。图像输入通过支持动态分辨率的 `Qwen35VisionEncoder` 处理；请显式传入所选仓库的投影器（上文 9B 与 Qwen 3.6 示例均为 `mmproj-F16.gguf`）。MoE 变体（例如 Qwen3.5-35B-A3B，以及使用同一架构标识的 Qwen3.6-35B-A3B GGUF）在 decode 时还会启用融合的 `MoEExpertsSwiGLUResidual` GGML 内核，将所有被选中的专家、可选的 shared expert 与残差加法合并到一次 GPU 计算图调度中执行。

### Mistral 3

Mistral 3 通过 Pixtral 视觉编码器支持图像输入。示例仓库使用 `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf`；请通过 `--mmproj` 显式传入。

- **图像：** PNG、JPEG、HEIC/HEIF

### Nemotron-H（Omni 发行版）

Nemotron Omni 发行版加入了 RADIO / v2_vl ViT 图像编码器。通过 `--mmproj` 传入对应的多模态投影器（例如 `nvidia_Nemotron-H-Omni-mmproj.gguf`）即可启用；语言模型 GGUF 不变。图像 token 在 `<image>` 占位符处插入，并由多模态注入器自动展开为 `<img>` + N 个 tile token + `</img>`。

- **图像：** PNG、JPEG、HEIC/HEIF
- **音频：** 聊天模板会为每个上传的音频文件发出一个 `<so_embedding>` token，CLI 仍会运行 Parakeet 风格 log-mel 预处理器以验证管线，但真正的音频推理需要尚未在公开 GGUF 中发布的 Parakeet 音频 mmproj。

