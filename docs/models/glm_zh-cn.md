# GLM-5.x（`glm-dsa`）

[← 返回模型索引](README_zh-cn.md) | [English](glm.md)

GLM-5.2 是一个 744B 参数的 MoE 模型（256 个路由专家，top-8，外加 1 个共享专家），
构建在 **DeepSeek 稀疏注意力**之上：带权重吸收的 Multi-head Latent Attention，
再加一个 "lightning indexer"，由它决定每个 query 可以看见哪些已缓存的 token。
官方宣称上下文：1M token。GGUF 架构 id 是 `glm-dsa`。

## 这一层长什么样

| 部件 | 形状（GLM-5.2） | 说明 |
|---|---|---|
| 注意力 | MLA，64 个 query head，1 个 key head | `q_lora_rank` 2048、`kv_lora_rank` 512、`n_embd_head_k/v_mla` 256、`n_rot` 64 |
| KV 缓存 | **每层每 token 只有一行 576 宽** | `kv_lora_rank + n_rot`；逐 head 的 K/V 解压被折进了 query（`attn_k_b`）和输出（`attn_v_b`） |
| DSA indexer | 32 head × 128，top-k 2048 | 对每个已缓存 token 打分；只有 top-k 能进入注意力掩码 |
| indexer 层 | 78 层里的 21 层 | 第 0、1、2 层，之后从第 6 层起每隔 4 层；中间的层复用上一个完整层的选择 |
| MoE | 256 专家，top-8，`n_ff_exp` 2048 | sigmoid 门控，路由 bias **只影响选择**，权重重归一化，路由分支 ×2.5 |
| Dense 层 | 前 3 层 | 普通 SwiGLU，`feed_forward_length` 12288 |
| NextN / MTP | 末尾 1 个块 | `block_count` 把它算在内；主干图跑的是 `block_count - nextn_predict_layers` 层 |
| RoPE | NORM，base 8e6，无 YaRN | 作用在 Q 的 64 宽 rope 切片和单个 K 的 rope 尾部 |

注意力 softmax 的缩放是 `1/sqrt(n_embd_head_k_mla)` = 1/16 —— 用的是**解压后**的
head 尺寸，而不是 576 宽的缓存行。

## TensorSharp 如何运行它

两套实现，都复现 llama.cpp 的 `src/models/glm-dsa.cpp`：

- **GGML 后端**（`--backend ggml_cuda` / `ggml_vulkan` / `ggml_cpu` / `ggml_metal`）
  走**原生整模型执行器**（`TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp`）。它自己
  加载分片 GGUF，把权重按层切分到每一张可见 GPU（226 GiB 放不进单卡），在设备上
  持有 MLA 与 indexer 缓存，并通过 `ggml_backend_sched` 每个 ubatch 只提交**一张**
  ggml 图；再配一个按形状索引的 LRU 图缓存，于是稳定态 decode 只是重放一张已分配
  好（在 CUDA 上还是已捕获）的图。
- **`--backend cpu`（100% 托管）和 `--backend cuda`** 走
  `TensorSharp.Models/Models/GlmDsa/*` 里的逐算子路径，建立在共享的 `Ops` /
  `ManagedQuantizedOps` / `CudaQuantizedOps` 之上。它同时也是原生执行器的参考
  实现；在 GGML 后端上用 `TS_GLM_NATIVE=0` 可以切到它做 A/B 对比。

分片 GGUF 由 `GgufFile` 自己处理（`split.count` / `-00001-of-000NN`），因此仓库里
任何模型现在都可以跨多个文件存放。

### 张量并行

`--tp N`（或 `TENSORSHARP_TP_DEGREE`）让**每一层都跑在每一张 GPU 上**，而不是把层
分给不同的卡，于是 decode 时每张卡只需要读 1/N 的权重，而不是依次走完全部。切分方式
沿用仓库其它模型一致的 Megatron column/row 模式：

| 部件 | 切法 | 集合通信 |
|---|---|---|
| 注意力 head | `attn_q_b` / `attn_k_b` / `attn_v_b` 按列并行，`attn_output` 按行并行 | 每层一次 all-reduce |
| 路由专家 | `ffn_gate_exps` / `ffn_up_exps` 按列并行，`ffn_down_exps` 按行并行 —— **每个专家都按行切开，专家本身不在 rank 之间分配** | 每层一次 all-reduce |
| 路由器、norm、indexer、共享专家、dense 层 | 复制 | 无 |
| MLA 与 indexer 缓存 | 每个 rank 只存自己那部分 head | 无 |

切专家的**隐藏维**而不是切专家的 **id**，是这件事能成立的关键：`ggml_mul_mat_id`
要求同一个 token 选中的专家 id 互不相同，而按 id 切分就必须给"本 rank 没有的专家"
编造出重复的 id。按行切分让路由器的全局 top-8 在每个 rank 上都仍然有效，不论路由
如何倾斜每个 rank 都恰好承担 1/N 的工作量，而且省下的不只是显存，还有 1/N 的算力。

一个 rank 的 down 投影切片是横着切每一**行**的，所以它以完整的量化块为单位；如果某个
模型的专家隐藏维不是 N 个整块，那就保持专家完整、只切 head（结果依然精确，只是慢一些）。
`TS_GLM_TP_SHARD` 可以单独选择两半（1 = head，2 = 专家，3 = 两者都切），
`TS_GLM_TP_OVERSUBSCRIBE=1` 允许多个 rank 共用一张 GPU —— 单卡机器上就是这样验证
切分正确性的。

### 并发请求的服务方式

原生执行器把所有缓存都留在设备上，所以一个请求的状态就是一个原生 **slot** ——
每层完整的一套 MLA 与 indexer 缓存，外加它自己的 `n_past`；把请求绑定到 slot 只是
切换活动 slot，不搬运任何 KV 字节（`GlmDsaModel.PerSeqCache.cs`，与 DeepSeek V4
使用同一套契约）。每个 slot 的计算图独立缓存与捕获，因此并发请求各自重放自己那张
已捕获的 CUDA 图，而不会重建、也不会重放别的请求里写死的缓存地址。

按 token 批处理的**分页**路径是刻意不实现的：MLA 每个 token 只存一行压缩表示，
DSA indexer 又要对同一段连续历史打分，根本没有分页 KV 布局可批。取而代之实现的是
**批量融合 decode**：一张图、来自 N 个序列各一个 token，所有投影、路由器、专家与
LM head 在整批上只跑一次，只有写缓存、indexer 打分和 softmax 按 token 展开——于是
N 个并发请求之间只读一遍权重，而不是各读一遍。在这台 3 卡机器上实测：四个并发的
200 token 补全，**合计 75.2 tok/s，而单流为 41.6**（1.81×），每条流 18.8 tok/s。

`TS_BATCHED_FUSED_DECODE=1` 打开它；默认关闭的原因与项目其它地方一致：批处理改变
了每个 GEMM 的形状，CUDA 因此选择不同 kernel，结果的最后几位会不同——与逐条路径的
首个偏差出现在第 1 层，相对量级 2e-8。稠密模型里这点差别看不见，但这 78 层里有 75
层要在几乎并列的分数上做 256 选 8 的 top-k，最后一位的差别会翻转边缘专家，到 LM
head 时 logits 已经差到 O(1)——在 2 bit 权重上这就是肉眼可见的另一段续写，而不是
舍入抖动。在 CPU 后端（kernel 不随批大小切换）上，批量与逐条 decode 逐位相同。

不开这个开关时并发依然可用而且精确：引擎交替执行每个序列的整图前向，四个并发补全
的结果与依次执行同样四个提示逐字节相同，只是权重要按序列各读一遍。

### 稀疏注意力这条路

当已缓存 token 数小于 `attention.indexer.top_k` 时，indexer 其实什么也去不掉——
对 `n_kv <= k` 求 top-k 会保留每一个 cell——所以 TensorSharp 干脆跳过打分，直接做
稠密注意力。这是同一个函数的廉价算法，也正因如此短 prompt 不必为 DSA 付出任何代价。
那些步骤里 indexer 的 **key 仍然会写进缓存**，因为后面更长的一步要给它们打分。

超过 top-k 之后，图会构建完整的 indexer：rope、Walsh-Hadamard 旋转、
`ggml_lightning_indexer`（后端没有该 kernel 时用等价分解），`ggml_top_k`，以及一张
先全部掩掉、再在选中位置解掩、最后加回因果掩码的注意力掩码。

**Hadamard 旋转是刻意复现的。** 它是作用在点积两侧的正交对合变换，在精确算术下会
相互抵消——但 indexer 的 key 缓存是 F16，先旋转再舍入能把误差均匀摊到 128 个维度上。
去掉它会改变一个 2741 token 的 prompt 上 top-k 选中的 token，从而破坏与 llama.cpp
的逐 token 一致；复现它则恢复了 6/6。

## 数值一致性

在同一台机器上，用 GLM-5.2-UD-IQ2_XXS（222 GiB）对比 `llama.cpp b200-9731ad3`，
输入**记录下来的** prompt token id，从而把前向过程与分词过程隔离开
（`.parity/gen_ref_glm.py`、`InferenceWeb.Tests/GlmDsaParityTests.cs`）：

| 后端 | 与 llama.cpp 逐 token 一致的 prompt 数 |
|---|---|
| `ggml_cuda`，3 张 GPU | 与同后端的 llama.cpp **6/6**（5 个短 prompt + 1 个走稀疏路径的 2741 token prompt） |
| `ggml_cpu` | 3/3 |
| `cpu`（100% 托管） | 1/1 |

**"同后端"这个限定是必要的**：在那个 2741 token 的 prompt 上，llama.cpp 自己的 CPU
与 CUDA 构建在第 5 个生成 token 上就不一致（`1467` 对 `8543`，两边读起来都是
"The … is a summary"），而 TensorSharp 会精确复现它当前所跑后端的那一个——在
`ggml_cpu` 上给出 CPU 的答案，在 `ggml_cuda` 上给出 CUDA 的答案。金标准是从一台 CPU
llama-server 上抓的，所以在 GPU 上跑校验时那一条会显示为不匹配。让这个 prompt 如此
敏感的正是 DSA 选择：indexer 的 key 缓存是 F16，在 2741 token 处 top-2048 里有两个
候选足够接近，分数上最后一位的差别就会把它们的顺序换过来。

张量并行与批量 decode 的切分用同样的方式验证，只是放在合成模型上，可以卡到严格得多
的标准：在 CPU 后端上 `--tp 1..4` 与批量 decode 与单 rank、逐条执行的路径**逐位相同**，
在 CUDA 上七种 head/专家切分组合全部逐 token 复现 golden 续写（校验工具里的
`--batched` 与 `TS_GLM_TP_SHARD`）。

`InferenceWeb.Tests/GlmDsaTinyModelTests.cs` 不需要下载 226 GiB 也能覆盖这套架构：
它会构造一个确定性的 1.9 MB `glm-dsa` GGUF，块结构与真模型相同（`top_k` 取 8，
所以 24 token 的 prompt 已经走稀疏路径），再用在该文件上从 llama.cpp 抓下来的
golden 校验贪心续写。

### 张量并行的实测数字

同样 3 张 GPU、GLM-5.2-UD-IQ2_XXS，`--tp 3`（head + 专家行）对比默认的按层切分：

| 测试 | 按层切分 | `--tp 3` |
|---|---:|---:|
| pp2048 | **896.8 t/s** | 502.8 t/s |
| tg64 | **43.9 t/s** | 16.2 t/s |

动手之前有两点要清楚。第一是精确性：切开注意力会把一次 GEMM 变成各 rank 局部和的
加总，残差因此在最后几位上不同——和上面的批量 decode 完全一样，75 层 256 选 8 的路由
会把这点差别放大成 2 bit 权重上另一段续写。对着记录下来的 llama.cpp 金标准，按层切分
复现 5/6 个短 prompt，`--tp 3` 复现 3/6；不同的那些 token 都是几乎并列的那种。第二
是速度，原因在互连：每层需要两次
对 `[6144, n_tokens]` 隐状态的 all-reduce，而这些卡是 PCIe 直连、没有 NVLink——一个
1024 token 的 prefill 分块每次跨越要搬约 25 MB，78 层 × 2 次归约下来，花在总线上的
时间超过切分省下的算术时间。按层切分每个 token 只搬两次隐状态，所以在这台机器上它
两项都赢。换到 NVLink/NVSwitch 主机上——all-reduce 便宜大约一个数量级——结论会反过来；
切分本身两边是一样的。

## 性能

3× RTX PRO 6000 Blackwell（每张 97 GiB），GLM-5.2-UD-IQ2_XXS，按层切分，两侧在同一
轮里背靠背测量（llama.cpp 用 `llama-bench`，TensorSharp 用校验工具的 `--bench`，同样
取两次重复中的最好值）：

| 测试 | llama.cpp | TensorSharp（默认 `n_ubatch` 1024） | TensorSharp（`TS_GLM_UBATCH=2048`） |
|---|---:|---:|---:|
| pp128 | **276.5 t/s** | 254.8 t/s | 264.4 t/s |
| pp512 | **695.4 t/s** | 666.9 t/s | 659.6 t/s |
| pp2048 | 763.1 t/s | **918.9 t/s** | **1145.8 t/s** |
| pp4096 | 715.8 t/s | **864.7 t/s** | **1048.7 t/s** |
| tg64 | 42.2 t/s | **43.7 t/s** | **43.9 t/s** |

交叉点大约在一千个 prompt token，原因在 micro-batch：256 个专家、top-8 的情况下，
512 token 的分块平均只给每个专家路由约 16 行，专家 GEMM 的 tile 大部分是填充，
把分块加大比图上任何别的改动都划算。再短一些时，整个 prefill 就是一张小图，每次调用
的固定开销——托管层跳转、输入上传、154880 宽的 logits 回传——占比就显出来了，
llama.cpp 那几个百分点就来自这里。decode 受显存带宽限制，两边都是 TensorSharp 略快
几个百分点。这些数字的重复测量波动约为 4%。

权重加载：热页缓存下 218 GiB 用时约 37 秒（5.9 GiB/s），16 个读线程跨 6 个分片并行。

## 怎么跑

```bash
# 3 张 GPU，按层切分（默认行为：用上每一张可见 GPU）
dotnet run --project TensorSharp.Cli -- --model GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf \
    --backend ggml_cuda --prompt "用一段话解释 MLA。"

# 指定 GPU 数量
TS_GLM_NGPU=2 dotnet run --project TensorSharp.Cli -- --model ... --backend ggml_cuda

# 显存不足：把路由专家（占 checkpoint 的 92%）留在系统内存里
dotnet run --project TensorSharp.Cli -- --model ... --backend ggml_cuda --cpu-moe
dotnet run --project TensorSharp.Cli -- --model ... --backend ggml_cuda --n-cpu-moe 30
```

offload 是让放不下的 checkpoint 能跑起来的手段，不是提速开关——模型本来就放得下时，
把专家挪到主机只会多出总线往返。同样 3 张卡、同一轮：默认按层切分 pp2048 **915.9** /
tg64 **43.9** tok/s，`--n-cpu-moe 30` 为 **94.7** / **16.4**，`--tp 3` 为 **505.6** /
**17.6**。只有在"otherwise 根本跑不起来"时才该用 `--n-cpu-moe`。

offload 与张量并行可以叠加：`--n-cpu-moe 30 --tp 2` 能正常加载，主机常驻的那些层会
保持专家完整（切开它们既省不了主机内存也省不了主机时间，而且跨步切片没法直接由 GGUF
映射就地提供——那会把一个映射文件变成 200 GiB 的私有副本），于是这些层由 rank 0 计算，
而留在 GPU 上的层照常切分。单独用 `--n-cpu-moe 30` 时与 llama.cpp 3/3 一致。

### 上下文长度

GGUF 宣称 1,048,576 token，但这并不意味着缓存放得下：78 层里每个 token 的 576 宽 MLA
行加上 indexer 的那一行约 93 KiB，1M 上下文就是约 98 GiB 的 KV——比权重还多，而且还没
算计算图。所以宣称的数字被当作**上限**而不是请求：加载器会用权重落盘后设备上真正剩下
的显存（再扣掉一张 `n_ubatch` 计算图里 DSA 掩码与 LM head 所需的部分）来定上下文，并把
选中的数字打出来。

```
[glm] context 91136 tokens (the GGUF advertises 1048576): 18.3 GiB free per rank
      after the weights, and the caches and graphs have to live in it.
```

`MAX_CONTEXT` 则反过来：你指定的上下文是硬性要求，放得下就照办，放不下就带着数字拒绝，
而不会在你背后悄悄缩小。

### 环境变量

| 变量 | 默认值 | 含义 |
|---|---|---|
| `TS_GLM_NGPU` | 0（全部） | 分摊层的 GPU 数 |
| `TS_GLM_UBATCH` | 1024 | prefill micro-batch；显存允许时 2048 在长 prompt 上更快 |
| `TS_GLM_THREADS` | min(核数, 32) | CPU 后端线程数（路由专家矩阵乘由 `--cpu-moe-threads` 覆盖） |
| `TS_GLM_NATIVE` | 1 | 置 0 则在 GGML 后端上改走托管逐算子路径 |
| `TS_GLM_FA` | 1 | 置 0 关闭 flash attention（回落到 soft_max） |
| `TS_GLM_FUSED_LID` | 1 | 置 0 用基础算子拼出 indexer，而不用 `ggml_lightning_indexer` |
| `TS_GLM_OP_OFFLOAD` | 自动 | 调度器 op-offload；一旦有层的专家常驻主机内存就默认关闭 |
| `TS_GLM_VRAM_RESERVE_MB` | 3072 | 层切分为计算缓冲区在每张卡上预留的余量 |
| `TS_GLM_GRAPH_CACHE` | 8 | 缓存的已构建+已分配计算图数量 |
| `TS_GLM_MOE_MMAP` | 1 | 置 0 则复制主机端专家，而不是映射 GGUF |
| `TS_GLM_TP_SHARD` | 3 | 张量并行切法：1 head，2 路由专家，3 两者 |
| `TS_GLM_TP_OVERSUBSCRIBE` | 0 | 置 1 允许多个张量并行 rank 共用一张 GPU（仅用于正确性测试） |
| `TS_GLM_BATCHED_DECODE` | 1 | 置 0 让原生侧拒绝所有批量 decode，强制走逐序列路径 |
| `TS_GLM_TRACE` | — | 层号列表（或 `all`），按 `llama-eval-callback` 的排版打印逐层激活和 |
| `TS_GLM_BD_DEBUG` | 0 | 置 1 打印每一步批量 decode 的过程（涉及哪些 slot、图是复用还是重建、走到哪一步） |
| `TS_GLM_TOPK` | 1 | 置 0 即使超过 indexer top-k 也做稠密注意力——用于 DSA 选择的 A/B |
| `TS_GLM_NODES_PER_LAYER` | 256 | 每层每 rank 的计算图节点预算 |
| `TS_GLM_LOAD_THREADS` / `TS_GLM_LOAD_CHUNK_MB` | 16 / 64 | 权重加载的并行度与分块大小 |

## 对话格式

```
[gMASK]<sop>[<|system|>Reasoning Effort: Max][tools]<|user|>...<|assistant|><think>
```

这一系列默认**开启**思考：正是那行 reasoning-effort 系统提示打开了它，而生成提示
会先写下一个 `<think>`，由模型自己闭合。关闭思考时提示里写的是 `<think></think>`，
模型于是直接作答。工具调用回来的形式是
`<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`，
每个参数一个 XML 元素（用 `tojson` 渲染的值会被解析回数字 / 数组 / 对象）。
