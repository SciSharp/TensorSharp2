# 环境变量 x 功能矩阵

[English](env_var_feature_matrix.md) | [中文](env_var_feature_matrix_zh-cn.md)

本文是 [`TensorSharp.TestMatrix`](../TensorSharp.TestMatrix/README_zh-cn.md)
使用的运行时开关参考。它只覆盖会真实影响推理正确性、吞吐、内存占用或模型路由
的高影响环境变量。

代码侧的事实来源是
[`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs)。
默认 sweep 列表配置在
[`TensorSharp.TestMatrix/Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json)。

## TestMatrix 如何使用本文

- 每个适用的 `(model, backend, feature)` cell 都会先运行一个**baseline**：
  不强制设置任何 sweep 变量。
- 对每个被选中的环境变量，运行器会为每个列出的值创建一个 case，并只把该变量
  传给 `TensorSharp.Cli` 子进程。
- 每个子进程启动前，会清理继承来的 `TS_*`、`GDN_*`、`QWEN35_*`、
  `FUSED_*`、`KV_CACHE_DTYPE`、`MAX_CONTEXT`、`MAX_TOKENS`、
  `VIDEO_MAX_FRAMES`、`VIDEO_SAMPLE_FPS`，确保矩阵值是权威输入。
- `--env-vars none` 会关闭 sweep case。如果配置文件中的 `default_env_vars`
  为空，且 CLI 没有覆盖它，运行器会使用全部已注册的 `EnvVarMatrix.All` 项。

下表中的“运行时 baseline”表示变量未设置时的行为。“默认 sweep”表示当前默认
配置是否会扫描该变量，而不是所有已注册变量。

DiffusionGemma 当前不属于已注册的 TestMatrix 功能目录：还没有 diffusion prompt
类型，没有 diffusion 专属 env sweep，运行器也不会清理继承来的 `DIFFUSION_*`
变量。要把 diffusion 结果纳入标准矩阵，请先显式配置模型，并新增对应 feature
与 env-var 注册。

## 连续批处理 / 批处理前向

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_GPTOSS_BATCHED` | GPT OSS | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_QWEN35_BATCHED` | Qwen 3.5 / 3.6 family、`qwen3next` | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Qwen 3.5 / 3.6 family、`qwen3next` | 原生批处理 GatedDeltaNet 内核 | 关闭 | `0`, `1` | 否 |
| `TS_NEMOTRON_BATCHED` | Nemotron-H | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_GEMMA4_BATCHED` | Gemma 4 | 批处理分页前向 vs 按序列回退 | 启用 | `0`, `1` | 是 |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Nemotron-H | 原生批处理 Mamba2 step | 关闭 | `0`, `1` | 否 |
| `TS_BATCHED_N1_FAST_PATH` | 全部 | solo 序列走融合 N=1 快速路径 decode；`0` 强制这些步骤走完全批处理路径 | 启用 | `0`, `1` | 是 |
| `TS_PER_SEQ_FUSED` | fused 能力模型（Gemma 4、Qwen 3.5/3.6） | 并发（N>=2）序列走 per-request 融合 Forward；`0` 强制逐算子批处理分页路径 | 启用 | `0`, `1` | 否 |
| `TS_BATCHED_FUSED_DECODE` | fused 能力模型 | per-seq fused 路径内的真正 token 批量融合 decode（一张图跑全部 N 个序列） | 关闭 | `0`, `1` | 否 |
| `TS_RETAINED_FUSED_CACHE` | fused 能力的滑窗模型（Gemma 4） | 保留已完成 fused KV holder 用于跨请求前缀复用 | 启用 | `0`, `1` | 否 |
| `TS_RETAINED_FUSED_CACHE_MAX` | fused 能力的滑窗模型 | 保留 fused holder 的 LRU 预算（限 VRAM） | `4` | 不适用 | 否 |
| `TS_SCHED_DISABLE_BATCHED` | 全部 | 全局按序列 KV-swap 回退 | 关闭 | `0`, `1` | 是 |

本节所有 executor 级开关都通过 `ExecutionOptions.FromEnvironment()` 统一读取，
由 `ExecutionPlanner` 消费（见 `docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md`
的"执行规划"一节）；按模型的 `TS_*_BATCHED` opt-out 则体现为模型声明的
`BatchedForwardAvailable` 能力。

## KV Cache / 上下文

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | 全部 | KV cache 元素类型 | 自动（随模型对齐：模型权重低于 F32 时为 `f16`，否则为 `f32`） | `f32`, `f16`, `q8_0`（运行时还接受 `q4_0`，不参与 sweep） | 是 |
| `TS_KV_PAGED_QUANT_BITS` | 全部 | TurboQuant 分页 KV 块编解码器 | 关闭（`0`） | `0`, `4`, `8` | 是 |
| `MAX_CONTEXT` | 长文本 / 上传文本 | 硬上下文上限 | 模型默认值 | `4096`, `8192`, `16384` | 是 |

## Prefill / Decode 调优

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | 在 GPT OSS、Qwen 3.5 / 3.6 family 长上下文功能上 sweep；运行时 Gemma 4、Nemotron-H、Mistral 3、Qwen 3 也会读取 | 分块 prefill 大小 | 架构默认值 | `256`, `512`, `1024` | 是 |
| `GDN_DISABLE_CHUNKED_PREFILL` | `qwen3next` | 关闭 GDN 分块 prefill | 关闭 | `0`, `1` | 否 |
| `TS_GGML_ASYNC_COMPUTE` | GGML 后端 | 异步 compute 提交 | `ggml_metal` 上启用（`0` 关闭），其他 GGML 后端关闭 | `0`, `1` | 是 |
| `TS_QWEN35_FD_PERSIST` | GGML GPU 后端上的 Qwen 3.5 / 3.6 family | 保留并重放整模型单 token decode 图 | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_GDN_INPLACE_STATE` | 单设备 `ggml_metal` 上的 Qwen 3.5 / 3.6 family | 让 K=1 GatedDeltaNet 输出与递归状态共享存储，消除逐层状态复制 | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_TOKEN_INPUT` | `ggml_metal` 上的 Qwen 3.5 / 3.6 family | 在 decode 图内直接从量化表读取 token embedding | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_KV_CPY` | `ggml_metal` 上的 Qwen 3.5 / 3.6 family | 通过可移动 `CPY` view 追加 K/V，而不是索引 scatter | 启用 | `0`, `1` | 否 |
| `TS_QWEN35_METAL_ASYNC_SUBMIT` | `ggml_metal` 上的 Qwen 3.5 / 3.6 family | decode 和 logits 回读提交后只同步一次 | 启用 | `0`, `1` | 否 |

## 多模态

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `VIDEO_SAMPLE_FPS` | 视频功能 | 按时间抽帧的每秒帧数 | `1` | `1`, `2` | 是 |
| `VIDEO_MAX_FRAMES` | 视频功能 | 抽取视频帧上限 | 不限制 | `8`, `16` | 是 |
| `TS_NEMOTRON_IMAGE_MAX_TILES` | Nemotron-H 图像功能 | 最大图像 tile 数 | 架构默认值 | `4`, `8`, `12` | 是 |

## MLX 专属

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_MLX_BATCHED_MOE_DECODE` | MLX 上的 Qwen 3.5 / 3.6 MoE | 每种 gate/up/down 一次批处理 dispatch，而不是按 expert dispatch | 启用 | `0`, `1` | 是 |
| `TS_MLX_DEVICE_ROUTER` | MLX 上的 Qwen 3.5 / 3.6 MoE | 满足前置条件时在 device 上执行 top-K + softmax router | 启用，且会自动回退 | `0`, `1` | 是 |
| `TS_MLX_PIPELINED_DECODE` | MLX decode 功能 | 模型支持时使用 device-side argmax 的流水化贪心 decode | 满足条件时启用 | `0`, `1` | 是 |
| `TS_MLX_DEVICE_KV_COPY` | MLX | Device 侧 KV scatter | 启用 | `0`, `1` | 否 |
| `TS_MLX_QWEN35_GDN_PACKED_KERNELS` | MLX 上的 Qwen 3.5 / 3.6 family | Packed GDN kernel | 关闭 | `0`, `1` | 是 |

## 矩阵外的 DiffusionGemma 变量

这些变量是真实运行时开关，但目前未注册到 `EnvVarMatrix.All`，也不在默认
TestMatrix 配置中 sweep。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `DIFFUSION_STEPS` | DiffusionGemma Web UI | 服务端路径每个 block 的去噪步数 | `48` | 未注册 | 否 |
| `DIFFUSION_MAX_BATCH` | DiffusionGemma Web UI | `DiffusionBatchScheduler` 的最大活跃请求数 | `2` | 未注册 | 否 |
| `DIFFUSION_BATCHED_FORWARD` | DiffusionGemma | 真正批处理 canvas decode vs 按时间片执行融合单 canvas decode | 关闭 | 未注册 | 否 |
| `DIFFUSION_NO_PKV` | DiffusionGemma | 关闭 device-glue 后端上的 prompt-KV 缓存 | 关闭 | 未注册 | 否 |
| `DIFFUSION_NO_SC` / `DIFFUSION_SC_TOPK` | DiffusionGemma | self-conditioning 开关与实验 top-K 截断 | 启用 / `32` | 未注册 | 否 |
| `DIFFUSION_NO_FUSED_DECODE` / `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | GGML 后端上的 DiffusionGemma | 关闭融合整模型 diffusion decode 或融合 lm-head tail | 关闭 | 未注册 | 否 |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | DiffusionGemma | 回退到按序列 lm-head 前的临时 logits 内存上限 | `300` | 未注册 | 否 |
| `DIFFUSION_VRAM_HEADROOM_MB` | ggml_cuda 上的 DiffusionGemma | 预加载权重之外保留的 VRAM 余量（计算缓冲、device copy） | `2048` | 未注册 | 否 |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | ggml_cuda 上的 DiffusionGemma | 模型放不进 VRAM 时 device-copy 缓存的上限（prompt K/V、mask、激活） | `768` | 未注册 | 否 |
| `DIFFUSION_SEGMENTED_DECODE` | ggml_cuda 上的 DiffusionGemma | 强制开启（`1`）/关闭（`0`）逐层融合 decode；模型放不进 VRAM 时自动启用 | 自动 | 未注册 | 否 |
| `DIFFUSION_PIN_STREAMED` | ggml_cuda 上的 DiffusionGemma | 把流式（非常驻）权重复制到页锁定内存以 DMA 速度上传（消耗 RAM） | 关闭 | 未注册 | 否 |

## 矩阵外的 MTP / 投机解码变量

这些变量控制 `TensorSharp.Server` 中可选的 MTP / NextN 投机解码路径（Qwen 3.6 内嵌
NextN 块；Gemma 4 独立 `gemma4-assistant` 草稿 GGUF）。投机仅对单序列（无并发）请求
生效，且只在有收益处（ggml 后端与纯 C# `cuda` 后端）启用。它们未注册在
`EnvVarMatrix.All` 中，也不在默认 TestMatrix 配置里扫描——矩阵特性目录目前没有投机解码
特性，请用显式服务端运行来验证这些变量。`TS_MTP_*` 也可由 `--mtp-*` 服务端 CLI 参数设置。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_MTP_SPEC` | Qwen 3.6、Gemma 4（服务端） | 为单序列启用投机解码 | 关闭（`0`） | 未注册 | 否 |
| `TS_MTP_DRAFT` | Qwen 3.6、Gemma 4（服务端） | 每个投机步最多起草的 token 数 | `8` | 未注册 | 否 |
| `TS_MTP_PMIN` | Qwen 3.6、Gemma 4（服务端） | 保留草稿 token 所需最低置信度 | `0.75` | 未注册 | 否 |
| `TS_MTP_DRAFT_MODEL` | Gemma 4（服务端） | 独立 `gemma4-assistant` 草稿 GGUF 路径 | 无 | 未注册 | 否 |
| `TS_GMTP_NO_FUSED` | ggml 后端上的 Gemma 4 | 关闭融合多 token 验证 / 草稿步内核（逐算子回退） | 关闭 | 未注册 | 否 |
| `TS_GMTP_NO_FAST_ROLLBACK` | Gemma 4 | 部分接受时恢复保留前缀回滚，而非稠密快速回滚 | 关闭 | 未注册 | 否 |
| `TS_GMTP_BATCHED_TRUNK` | Gemma 4 | 验证主干走批量分页路径，而非线性主干 | 关闭 | 未注册 | 否 |

## 矩阵外的张量并行 / 分布式推理变量

这些变量配置张量并行（把单个模型切分到多张 GPU）以及基于点对点 TCP 网格的多节点
分布式 TP。它们未注册在 `EnvVarMatrix.All` 中，也不在默认 TestMatrix 配置里扫描
——TP 需要多张 GPU，而标准的单 GPU 测试环境无法覆盖。TP 可运行在直连 `cuda` 后端
以及 GGML CUDA / Vulkan 后端（`ggml_cuda`、`ggml_vulkan`）上。
`TENSORSHARP_TP_DEGREE`、`TENSORSHARP_TP_NODE_ID` 与 `TENSORSHARP_TP_PEERS` 也可
通过 `TensorSharp.Cli` 与 `TensorSharp.Server` 的 `--tp`、`--tp-node-id`、
`--tp-peers` 参数设置。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TENSORSHARP_TP_DEGREE` | 全部自回归模型；`cuda`、`ggml_cuda`、`ggml_vulkan` 后端 | 把模型切分到本机多少张 GPU（Megatron-LM 列/行并行） | `1`（单 GPU） | 未注册 | 否 |
| `TENSORSHARP_TP_DEVICES` | GGML 后端上的本地 TP | 各 rank 使用的 GPU 序号（逗号分隔，例如 `0,2`） | `0..tp-1` | 未注册 | 否 |
| `TENSORSHARP_TP_NODE_ID` | 全部自回归模型；`cuda`、`ggml_cuda`、`ggml_vulkan` 后端 | 多节点分布式 TP 中本节点的 0 起始编号；必须与 `TENSORSHARP_TP_PEERS` 一起设置 | 未设置（关闭） | 未注册 | 否 |
| `TENSORSHARP_TP_PEERS` | 全部自回归模型；`cuda`、`ggml_cuda`、`ggml_vulkan` 后端 | 分布式 TP 集群中所有节点的 `host:port` 列表（逗号分隔）；必须与 `TENSORSHARP_TP_NODE_ID` 一起设置 | 未设置（关闭） | 未注册 | 否 |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | 仅分布式 TP | 各节点向 peer 重试连接多久后放弃 | `120` 秒 | 未注册 | 否 |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | 仅分布式 TP | peer 套接字的单次接收超时；卡住的 peer 会让集合通信失败而不是一直挂起 | `300` 秒 | 未注册 | 否 |
| `TENSORSHARP_TP_DISABLE_P2P` | 本地 TP，`cuda` 后端 | `1` 表示所有跨 GPU 传输一律经主机中转，不使用 CUDA 点对点 DMA（与 A16 vGPU 等无 P2P 硬件一致） | 关闭（通过 DMA 自检的设备对使用 P2P） | 未注册 | 否 |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | 本地 TP，`cuda` 后端 | `1` 表示本地 AllReduce 走主机内存（设备→主机、求和、主机→设备）而非设备到设备路径——诊断兜底 | 关闭（设备到设备） | 未注册 | 否 |
| `TS_GGML_TP_PARALLEL` | 本地 TP，GGML 后端 | `0` 表示顺序而非并发地驱动各 rank（诊断用） | 开启（并发 rank 工作线程） | 未注册 | 否 |
| `TS_GGML_TP_FUSED_MATMUL` | 本地 TP，GGML 后端 | `1` 表示由单个线程提交两个 rank 的线性层；每次调用都要为每个 rank 分配设备缓冲，在 Qwen 3.5 35B 上实测慢 2.3× | 关闭（通用按 rank 路径） | 未注册 | 否 |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | 本地 TP，GGML 后端 | 超过该元素数量时 AllReduce 走设备集合通信，否则在主机内存中归约 | `262144` | 未注册 | 否 |
| `TS_GGML_F32_RESIDENT` | GGML 后端 | `0` 表示每次调用重新绑定 F32 线性层权重，而不是常驻设备（诊断用） | 开启（常驻设备） | 未注册 | 否 |
| `TS_GEMMA4_TP_FUSED_MOE` | GGML 上 TP 下的 Gemma 4 MoE | `0` 表示从融合的整模 MoE 主干（专家内部 Megatron 切分）回退到逐算子的整专家路径 | 开启（融合主干） | 未注册 | 否 |
| `GGML_CUDA_ALLREDUCE` | 本地 TP，`ggml_cuda` | `nccl` / `internal` / `none` —— 直接透传给 ggml 的集合通信选择 | 自动（构建时能找到 NCCL 就用 NCCL） | 未注册 | 否 |
| `GGML_CUDA_AR_BF16_THRESHOLD` | 本地 TP，`ggml_cuda` | ggml 在多大载荷以上把 F32 集合通信转成 BF16；TensorSharp 把 ggml 的默认值提高到 1 MB，使 decode 规模的归约保持精确 | `1 MB`（由 `TSGgml_TensorParallelInit` 设置） | 未注册 | 否 |
| `TS_QWEN35_LAYER_TRACE` | Qwen 3.5/3.6 | `1` 打印首次前向的逐层残差流摘要，单卡与 TP 两条路径都会输出（诊断用） | 关闭 | 未注册 | 否 |

## 矩阵外的 Redis 共享状态变量

这些变量为 `TensorSharp.Server` 配置可选的 Redis 共享状态：用于跨会话复用的共享 KV
缓存层，以及持久化的 OpenAI Responses API 存储。它们未注册在 `EnvVarMatrix.All`
中，也不在默认 TestMatrix 配置里扫描。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_KV_CACHE_REDIS_URL` | 仅服务端 | 共享 KV 缓存层的 Redis 连接串；设置后 KV 块持久化到 Redis 以便跨会话复用 | 未设置（关闭） | 未注册 | 否 |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | 仅服务端 | Redis KV 缓存条目的 TTL（分钟）；`0` = 不过期 | `1440`（24 小时） | 未注册 | 否 |
| `TS_RESPONSES_STORE_REDIS_URL` | 仅服务端 | Responses API 存储的 Redis 连接串；设置后取代内存存储 | 未设置（关闭） | 未注册 | 否 |

## 矩阵外的通用运行时开关

这些变量是真实运行时开关，但目前未注册到 `EnvVarMatrix.All`，也不在默认
TestMatrix 配置中 sweep。

| 环境变量 | 适用范围 | 功能影响 | 运行时 baseline | Sweep 值 | 默认 sweep |
|---|---|---|---|---|---|
| `TS_PDF_MAX_PAGES` | PDF 文档输入（CLI `--pdf`、服务端 `/api/upload`） | 文本提取与页面图像渲染读取的 PDF 页数上限 | `0`（全部页面） | 未注册 | 否 |
| `TS_FUSED_QKNORM_ROPE` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 纯文本 prefill | 融合 QK-Norm + NeoX-RoPE CUDA 内核；`0` 回退到分离的 norm + RoPE 算子（多模态 MRoPE 与其他后端始终走分离路径） | 启用 | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM` | 直连 `cuda` 后端，激活行数 ≥ `TS_CUDA_QMM_F16GEMM_MIN_ROWS` 的量化矩阵乘 | 将权重一次性反量化为 F16 并走张量核心 cuBLAS GEMM（ggml 风格的 prefill 路线），替代分块量化内核；`0` 回退到量化内核 | 启用 | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | 直连 `cuda` 后端 | F16 GEMM 路线的激活行数阈值 | `32` | 未注册 | 否 |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | 直连 `cuda` 后端 | F16 权重暂存区上限（MB）；超过上限的权重（如 LM head）继续使用量化内核 | `768` | 未注册 | 否 |
| `TS_CUDA_Q80_VEC` | 直连 `cuda` 后端，Q8_0 单行（decode）矩阵乘 | 对 q8_1 量化后的激活行执行每 warp 一列的 dp4a 矩阵-向量乘（类似 ggml `mul_mat_vec_q`）；`0` 回退到精确 FP32 反量化内核 | 启用 | 未注册 | 否 |
| `TS_CUDA_Q80_VEC_MIN_OUT` | 直连 `cuda` 后端 | Q8_0 dp4a 矩阵-向量乘的最小输出宽度（诊断开关） | `0` | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ` | 直连 `cuda` 后端，激活行数在 32..`TS_CUDA_Q80_MMQ_MAX_ROWS` 的 Q8_0 矩阵乘 | 直接在原始 Q8_0 块上执行 int8 张量核心 GEMM（mma.m16n8k32，ggml MMQ 风格），替代反量化+cuBLAS F16 路线；`0` 回退到 F16 GEMM | 启用 | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | 直连 `cuda` 后端 | 行数超过该阈值后 F16 GEMM 路线更优（MMQ 的权重扫描次数随 ceil(rows/128) 增长） | `512` | 未注册 | 否 |
| `TS_CUDA_Q80_MMQ2` | 直连 `cuda` 后端 | MMQ GEMM 的 cp.async 暂存变体（拆分的 q8_1 激活暂存 + 原始权重窗口以 cp.async 异步拷贝到共享内存；在 inDim % 256 == 0 时启用，结果逐位一致，prefill 约快 18%）；`0` 固定使用寄存器预取的 MMQ 内核 | 启用 | 未注册 | 否 |
| `TS_CUDA_GDN_PREFILL_SPLIT` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 GDN prefill | 三阶段无同步 GDN prefill（并行卷积/归一化 → 寄存器驻留行扫描 → 并行 RMS+门控）；`0` 固定使用旧的单内核逐 token 路径 | 启用（seqLen ≥ 8 且 headKDim = 128） | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 纯文本多 token prefill | 在同一 (seqLen, startPos, 缓存标识) 形状第二次运行时把逐算子 prefill 层循环捕获为 CUDA graph，之后以单次 `cuGraphLaunch` 重放（结果逐位一致；捕获失败时自动回退普通路径）；`0` 关闭全部 CUDA graph 捕获（包括 decode graph） | 启用 | 未注册 | 否 |
| `TS_CUDA_DECODE_GRAPH` | 直连 `cuda` 后端上的 Qwen 3.5 / 3.6 纯文本 decode | 把逐算子 decode 步骤（seqLen = 1）捕获为 CUDA graph 并逐 token 重放；位置相关的值（注意力长度、KV 写入槽位、GDN 卷积环索引、RoPE 位置）由内核从一块以锁页主机内存刷新的设备参数块中读取，因此在 KV 缓存扩容之前一个 graph 可服务所有位置（结果逐位一致；捕获失败时自动回退普通路径）；`0` 关闭 | 启用 | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_MAX` | 直连 `cuda` 后端 | 缓存的 prefill + decode graph 数量（LRU 淘汰；每个 graph 固定持有其捕获时使用的内存池块） | `4` | 未注册 | 否 |
| `TS_CUDA_PREFILL_GRAPH_LOG` | 直连 `cuda` 后端 | 打印 graph 捕获/重放/中止事件（`1`） | 关闭 | 未注册 | 否 |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | 直连 `cuda` 后端 | 全局大块（≥ 2 MB）显存缓存预算；让 prefill 级激活保持池化，避免每层重复 cuMemAlloc/cuMemFree | `1024` | 未注册 | 否 |
| `TS_CUDA_PROFILE` | 直连 `cuda` 后端 | 退出时打印 CPU 回退算子与主机↔设备同步计数（`1`），含调用点归因（`2`） | 关闭 | 未注册 | 否 |

## 功能覆盖

功能目录位于
[`TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)。
当前功能集合如下：

| 功能 | 驱动方式 | 能力门控 |
|---|---|---|
| `pp512` | `--benchmark --bench-prefill 512 --bench-decode 0` | 所有模型 |
| `pp2048` | `--benchmark --bench-prefill 2048 --bench-decode 0` | 所有模型 |
| `tg128` | `--benchmark --bench-prefill 32 --bench-decode 128` | 所有模型 |
| `short_text` | `--input prompts/short_text.txt --max-tokens 64` | 所有模型 |
| `long_text` | `--input prompts/long_text.txt --max-tokens 64` | 所有模型 |
| `uploaded_text` | `--input prompts/upload_text.txt --max-tokens 64` | 所有模型 |
| `multi_turn` | `--multi-turn-jsonl multi_turn/three_turn.jsonl` | 所有模型 |
| `tools` | `--tools tools/weather_tools.json` | 矩阵能力标记为支持工具调用的模型 |
| `thinking` | `--think` | 矩阵能力标记为支持思维链的模型 |
| `image` | `--image media/apple.png --mmproj ...` | 图像模型且有 mmproj |
| `audio` | `--audio media/sample.mp3 --mmproj ...` | 音频模型且有 mmproj |
| `video` | `--video media/sample.mp4 --mmproj ...` | 视频模型且有 mmproj |

默认语义检查刻意保持较弱，用于捕获灾难性回归。相关功能会检查 `blue`、
`paged`、`08:01:12`、`alex` + `teal`、`get_current_weather` + `tokyo`、
`10:38` 与 `apple`。音频与视频没有默认期望子串，因为样例媒体由运行环境提供。

## 过滤规则

运行前会过滤组合爆炸：

1. 后端可用性：CUDA 与 Vulkan 后端在 macOS 跳过（macOS 上的 GPU 后端是 Metal）；MLX 需要 Apple Silicon；GGML Metal 需要 macOS。
2. 模型能力：当发现或配置的模型不支持图像 / 音频 / 视频 / 工具 / 思维链时，对应功能跳过。
3. 投影器可用性：多模态功能需要 mmproj 路径。
4. 环境变量适用性：每个 `EnvVarSpec.AppliesTo` 决定该变量是否对当前 `(model, backend, feature)` cell 有意义。

## 更新矩阵

新增高影响环境变量时：

1. 在 [`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs) 注册一个 `EnvVarSpec`。
2. 如果它应进入默认 sweep，把它加入
   [`Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json)
   的 `default_env_vars`。
3. 更新本文和英文版本中的对应行。
4. 如果该变量改变功能适用性，同步更新
   [`FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)
   或模型发现的能力推断。
