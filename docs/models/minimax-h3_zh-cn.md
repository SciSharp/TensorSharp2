# MiniMax-H3

MiniMax-H3 **同时生成视频与原生立体声音频**，最长 15 秒、24 fps、32 kHz 立体声。
它不是"视频模型外挂音频"：同一个扩散 Transformer 在**一条 token 序列**里对打包
在一起的"视频 + 音频"潜变量做去噪，因此声音是模型输出的一部分，而不是事后配上去的。

TensorSharp 用原生 ggml 整图运行它——每个网络一张图，权重直接从 GGUF / safetensors
的 mmap 常驻绑定。

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

这会写出 `fox.mp4`，以及带生成音轨的 `fox.wav`。

## 支持状态

| 组件 | 状态 |
|---|---|
| 文生视频（`t2v`） | 可用 |
| 图生视频（`i2v`）——让照片动起来 | 可用 |
| 首尾帧（`fl2v`） | 可用 |
| 参考生视频（`ref`）——参考图、参考视频、参考音频 | 可用 |
| 联合立体声音频 | 可用 |
| 视频 VAE 编码 + 解码 | 可用（分块） |
| 超过 22 帧的片段 | 可用（VAE 每次解码 5 个潜变量帧） |

## 性能

M5 Pro / Metal，22 帧、8 步、相同随机种子，对比
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 的最佳配置：

| 分辨率 | stable-diffusion.cpp | TensorSharp | |
|---|---|---|---|
| 256×256 | 49.3 秒 | **20.9 秒** | 快 2.4 倍 |
| 640×384 | 108.5 秒 | **63.1 秒** | 快 1.7 倍 |

## 需要的文件

H3 需要四个文件。两个去噪器是**两个独立的检查点，而不是开关**——你加载哪个，
决定了它接受哪种条件输入。

| 文件 | 来源 |
|---|---|
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)：文本 + 关键帧 |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 同一仓库：文本 + 参考 |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 同一仓库：文本编码器，两者共用 |
| `minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimax_h3_audio_vae_fp32.safetensors` | 同上；不给则输出无声视频 |

伴随文件会在去噪器旁自动解析，也可用 `--video-vae`、`--video-text-encoder`、
`--audio-vae` 指定。

> **文本编码器的 GGUF 不带分词器。** 请把
> [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor)
> 里的 `vocab.json` 和 `merges.txt` 放到它旁边，或用 `TS_VIDEO_TOKENIZER` 指向它们。
> 两个已发布的 GGUF **完全没有元数据**，因此 TensorSharp 靠张量而不是架构字符串来识别 H3。

> **必须用 `--cfg 1.0`。** H3 是 CFG 蒸馏模型，超过 1.0 质量会明显劣化，
> TensorSharp 会直接拒绝。4–8 步是常用工作点，所以迭代很快。

## 条件模式

一张图对 H3 来说可以有两种**完全不同**的含义，而且它们用**不同的检查点**。
`--video-mode` 用来显式指定；不指定时会根据你传入的内容自动推断。

| 你想要什么 | 模式 | 检查点 |
|---|---|---|
| "让这张照片动起来" | `i2v` | FL2VA |
| "从照片 A 变到照片 B" | `fl2v` | FL2VA |
| "用这个人物，生成全新场景" | `ref` | Ref2VA |
| "参考这个产品/人物，但换机位、背景和构图" | `ref` | Ref2VA |
| 只有文本 | `t2v` | 都可以 |

### 文生视频

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

### 图生视频——让照片动起来

关键帧会被**两次**用作条件，两者缺一不可：VAE 编码出的潜变量把它钉在所锚定的那一帧上，
同一张图还会经 Qwen3-VL 视觉塔进入提示词，为整个片段描述画面内容。只有潜变量的话条件会
衰减——片段开头贴合图片，一两秒后就跑偏了；这在 22 帧时看不出来，到 124 帧就很明显。

图片会成为**第一帧**，提示词决定后续发生什么。

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image portrait.jpg \
  --prompt "the person turns toward the camera and smiles, subtle handheld motion" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output animated.mp4
```

加 `--video-mode i2v` 可显式声明。图片会被缩放到生成画布，所以它的宽高比最好与
`--width`/`--height` 一致，否则会被拉伸。

### 首尾帧

两端都被钉住，模型补出中间的运动。

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image start.png --end-image end.png \
  --prompt "a slow cinematic push-in" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output morph.mp4
```

只给 `--end-image` 也可以：片段会结束在那一帧。

### 参考生视频——保留主体，其余全部重来

Ref2VA 把图片当作身份与外观的**参考**，而不是画面帧。第一帧完全不必与它相似：
人物、产品或物体的特征会保留下来，而机位、背景和构图完全由提示词决定。

这需要 **Ref2VA** 检查点——`i2v`/`fl2v` 与 `ref` 是两个不同的文件，不是一个开关。

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg \
  --prompt "the same woman sits at a table in a sunlit cafe by a window, drinking coffee, wide shot" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output cafe.mp4
```

多次传入 `--ref-image` 可以给出多张参考（最多九张），例如一个人物加上她手中的产品：

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg --ref-image bottle.png \
  --prompt "she holds the bottle up to the light on a rooftop at golden hour, slow orbit" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output rooftop.mp4
```

注意事项：

* 参考图只会被**缩小**到生成画面的面积，并保持自身的宽高比，不会被拉伸；
  输出画布也**不会**取自参考图，请用 `--width`/`--height` 明确指定。
* 常见尺寸下每张参考图大约多占 63 个序列 token，所以九张会明显比一张慢。
* 提示词里要写清画面中有谁。参考图提供身份，镜头内容仍然要靠提示词描述，
  否则会得到一个画得很好、但主体缺席的场景。
* 参考也可以是**视频片段**或**音频**，不限于静态图——见下文。

> **在 Ref2VA 检查点上，普通的 `--image` 会被当作参考图。** 因此只会"上传一张图"的
> 客户端（包括 Web UI）无需任何额外字段即可使用 Ref2VA。想写明确可以加 `--video-mode ref`。

### 参考视频与参考音频

`--ref-video` 接受视频**文件**或一个帧目录，`--ref-audio` 接受 WAV/MP3/Ogg。
视频自带的声音要用 `--ref-video-audio` 单独给出，按位置与 `--ref-video` 配对——
容器里的音轨无法通过帧解码器读取，所以两者是分开的输入。

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-video walk.mp4 --ref-video-audio walk.wav \
  --prompt "the same woman walks along a beach at sunset, wide shot, waves behind her" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output beach.mp4
```

不带画面的独立音频同样可以作为参考：

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image singer.jpg --ref-audio song.wav \
  --prompt "she performs on a small club stage under a single spotlight" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output stage.mp4
```

参考视频会经过这些处理：

* 重采样到 H3 自己的 **24 fps**，并向下对齐到 17k+5 帧网格，且不超过本次生成的帧数。
  24 fps 下不足 5 帧的片段会被拒绝。
* 画布由它自身的宽高比在 768 px 附近推出，再按面积封顶并对齐到 32。比这更**小**的
  源保持自身尺寸：放大并不会产生可供参考的细节，代价却很大——448x320 的片段若放大到
  1088x768，会带来 5712 个条件 token，而被生成的片段本身只需要 1680 个。
* 语言模型以 **2 fps** 看它，每次看两帧，并标注该对所处的时间；去噪器则以潜变量的
  形式看到全部内容。
* 音频会重采样到音频 VAE 的 32 kHz 立体声，并截断到生成片段的时长。

> **参考视频是 H3 最昂贵的输入。** 一个 22 帧 448x320 的参考会在输出本身所需的
> 1680 个 token 之外再加 980 个条件 token，而且 VAE 需要先把这 22 帧全部编码。
> 在 M5 Pro 上实测：编码参考约 14 秒，之后每个去噪步约 9 秒（不带参考时约 5 秒）。

## 如何获得高质量

### 先把主体放进画面——这一点比其余所有设置加起来都重要

**主体在画面中占多大比例，比下面任何参数都更决定成败。** 模型只能按给定的分辨率
渲染它拿到的东西；如果条件图里一张脸只有 40 像素宽，再多的步数、再大的输出尺寸
也补不回来。

用一张拍摄公园通行证的照片实测（证件上印着两个很小的人像），提示词、种子、20 步
全部相同：

| 输入 | 输出中的人脸 |
|---|---|
| 原样照片，640x384 | 约 40 像素宽——糊成一团，认不出 |
| 裁到两个人物，768x480 | 约 250 像素宽——头发、皮肤纹理、领针都清晰可辨 |

除了裁剪之外什么都没变。如果你在意的是人脸，请先把源图裁到人脸再生成。

> **条件图永远不会被拉伸。** 当画布宽高比与图片不一致时，图片会被居中裁剪以适配，
> 并在运行日志中说明。否则一张 4:3 的照片被强行放进 640x384（5:3）会被横向压缩 25%，
> 而关键帧**就是**第一帧，这个变形会被后面每一帧继承。不指定 `--width`/`--height`
> 可以直接采用图片自身的宽高比；想控制取景就自己裁剪源图。


有两个设置起决定性作用，其余的相比之下都是零头。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--width` / `--height` | 640×384；有条件图时按该面积取图片的宽高比 | **影响最大的一项。** 人脸需要像素：256×256 下一张脸只有几十像素宽，无论其他参数怎么调都会糊、都会变形。 |
| `--diffusion-steps` | 20 | 消除运动主体周围的彩色边缘。8 步是快速档，约 20 步干净，超过 30 提升有限。耗时大致线性增长。 |
| `--diffusion-seed` | 随机 | 有些种子构图就是更好，是最省事的重试手段。 |
| `--flow-shift` | 12 | 改变时间表把步数花在哪里，通常不需要动。 |
| 量化等级 | — | 去噪器用 `-Q8_0` 优于 `-Q4_K`，文本编码器用 `Q4_K_M` 优于 `Q2_K_M`，显存够就用更高的。 |

`--cfg` **不是**质量参数：H3 是 CFG 蒸馏模型，只接受 1.0。同理 `--negative-prompt`
也不起作用——根本没有无条件分支可以用来做反向引导。

同一个图生视频请求下实测（M5 Pro、`ggml_metal`、seed 42、22 帧）：

| 设置 | 耗时 | 结果 |
|---|---|---|
| 256×256、8 步 | 26 秒 | 模糊，人脸无法辨认 |
| 640×384、8 步 | 79 秒 | 清晰，但运动的手周围有彩色边缘 |
| 640×384、24 步 | 212 秒 | 干净 |
| 576×448（图片宽高比）、20 步 | 187 秒 | 最好——没有任何拉伸 |

> **在服务端，尺寸是启动参数。** Web UI 只发送提示词和图片，请求会继承服务端默认值。
> 启动时加上 `--video-width 640 --video-height 384 --video-steps 20`
> （或者干脆不加，让每个请求按自己的图片取宽高比）。`--width` / `--height` 是别名。

> **强行指定与图片不匹配的尺寸会拉伸画面。** 4:3 的照片被塞进 640×384 会被水平压缩约 25%，
> 而这正是让人脸"看起来不对"的那种形变。做图生视频时不指定宽高，就会按图片的宽高比来。

> **本次修订改变了音轨输出。** 去噪器输出的是白化后的音频潜变量，而解码器需要 VAE
> 自身的尺度；此前缺少了反白化这一步，因此每一条生成的音轨都是从一个约小 1.9 倍的
> 潜变量解码出来的。它只表现为音量偏低和音色略有偏差，不会是静音或噪声。已对照参考
> 实现对同一潜变量的解码验证：余弦 0.99998、增益 1.0000 倍（此前为 0.81 倍）。

## 尺寸与长度

* 宽高向**上**取整到 32 的倍数。
* 帧数向上对齐到 **17k+5** 网格——5、22、39、56、90……这来自视频 VAE 的时间分块，
  不是随意规定的。
* fps 固定为 24，传别的值会被覆盖。
* TensorSharp 目前只生成一个 VAE 时间块，因此**实际可用长度是 22 帧**。

> **长片段会保留条件图，但不一定保留取景。** 关键帧钉住的是片段的**开头**。在 124 帧
> （5.2 秒）里，一个描述了不同镜头的提示词会占上风：把"两个人争吵的中景、手持镜头"
> 用在一张**拍摄证件的照片**上，镜头会一路推进到证件里，大约第 60 帧就变成了那个中景。
> 换成不要求运镜的提示词，同样 124 帧的片段全程与源照片保持 0.94–0.97 的相关度。
> 想让照片一直在画面里，就在提示词里说明，或者生成更短的片段。

## HTTP API

```sh
curl -s localhost:5001/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "a red fox trotting through falling snow, cinematic",
  "width": 640, "height": 384, "frames": 22, "steps": 8, "cfg": 1.0,
  "imagePath": "card.jpeg", "videoMode": "i2v"
}'
```

参考条件走同一个路由，只是换成 Ref2VA 检查点：

```sh
curl -s localhost:5001/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "the same woman sits at a table in a sunlit cafe by a window, wide shot",
  "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
  "referenceImages": ["person.jpg", "bottle.png"], "videoMode": "ref"
}'
```

返回 `{ ok, url, audioUrl, width, height, frames, fps, seed, codec, elapsedSeconds }`。
`imagePath` / `endImage` / `referenceImages` 引用的是之前通过 `/api/upload` 上传的文件；
上传目录之外的路径会被拒绝。OpenAI 风格的 `/v1/videos/generations` 接受同样的字段
（snake_case：`reference_images`、`video_mode`），返回 `audio_url`。

音频写成独立的 `.wav` 而不是封装进 MP4，因为封装需要一个未必安装的编码器。合并：

```sh
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

## 架构要点

这些都记在这里，是因为检查点本身没有任何元数据，无从推断。以下内容与
stable-diffusion.cpp、上游参考 PyTorch 实现、以及 GGUF 张量表三方交叉核对过。

### 扩散 Transformer

50 层、约 193 亿参数，单流：文本、条件帧、目标音频、目标视频在**同一条序列**里做
全双向注意力，**没有 cross-attention**。

| | |
|---|---|
| hidden | 5376 |
| 注意力 | 56 头 × 128 = **内部 7168**，比 5376 的残差流更宽 |
| MLP | SwiGLU，gate 在前 |
| patch | (t, h, w) = (1, 2, 2)；视频 24 通道 → 96 维 patch |
| 音频 | 32 通道，每个潜变量帧、每个立体声通道各一个 token |

一个 token 内部的 96 个视频数值是**通道优先、patch 其次**（`c*4 + ky*2 + kx`）。
反过来排列统计特征完全一样，却会悄悄打乱每一个 token。

**AdaLN** 用的是学习到的曲线表，而不是时间步 MLP：`adaln_t_table` 形状 `[8, 1025]`，
在时间步处插值，每层一个 `Linear(8 → 96768)` 为 **3 个模态**（0=视觉、1=文本、2=音频）
各产生 6 个调制向量。一段 token 取 `timestep*3 + modality` 这一行。

**RoPE** 是三轴的，`inv_freq` 是学习到的 16 项，因此 128 个头维中有 96 维参与旋转。
位置是**连续浮点数**：时间轴以音频潜变量为单位（1/40 秒），使两条流共享同一条时间线；
空间轴按画面的几何平均尺度归一化。

### 视频 / 音频双 shift

视频要 shift 12，音频要 shift 3。H3 没有跑两个采样器，而是让采样器留在视频时间表上、
由模型做换算：音频流按"同一底层时间下 shift-3 时间表会有的 sigma"来加条件，
其速度再乘以 `d(sigma_a)/d(sigma_v)`，于是共享的 Euler 步会沿音频自己的轨迹积分。
两条时间表都从 1 开始、到 0 结束，所以两条流同时落地。`--flow-shift` 只影响视频流。

### 文本编码器

Qwen3-VL-32B 截断到 50 层并**去掉最后的 norm**——DiT 取的是未归一化的第 50 层隐状态，
其中数量级上万的离群激活是模型的一部分，不是 bug。**没有 chat template**：
提示词就是原始文本，套模板会让每个隐状态都发生偏移。

### 视频 VAE

空间 16 倍、时间 4 倍。解码器是**纯 36 层 Transformer**，没有任何反卷积。
每个潜变量体素是一个 token，最后一个 `Linear(2048 → 3072)` 加 depth-to-space
一步完成全部上采样。

解码**按 256 px 分块，而且这是正确性要求而非优化**：解码器的 RoPE 坐标会按拿到的
范围做长度归一化，只有在它训练时的那个范围内才成立。把 640×384 一次性解码，
即使每个分块单独都完全正确，整体也会出现明显的错切。

编码器是因果 3D CNN；但图像条件只需要单帧，而因果 padding 补的是两帧**零**，
于是每个卷积核只有最后一个时间切片起作用，整个网络**精确地**退化成 2D 卷积。

### 音频 VAE

DAC/BigVGAN，32 潜变量通道 → 32 kHz 立体声。上采样倍率 {5,5,2,2,2,2,2} 相乘为 800，
而 32000/800 正好是 40 Hz 的潜变量帧率。使用无混叠 snake 激活：每个非线性都被夹在
"2 倍上采样 / 激活 / 2 倍下采样"之间，避免把高频折回带内。

与视频潜变量不同，音频潜变量**原样**送进解码器——**不**应用 `latents_mean` /
`latents_std`。应用它们会让幅度小约 15 倍，得到一条频谱看似合理、实际几乎听不见的音轨。

## 数值验证

每个网络都是对着**参考实现**校验的，而不是自己跟自己比。测试基准来自上游 PyTorch
以及 stable-diffusion.cpp 自己的中间张量。

| 组件 | 结果 |
|---|---|
| 文本编码器（全部 50 层） | 对比 sd.cpp 的条件输出 cos 0.999999 |
| DiT 单步（全部 50 层） | 视频 cos 0.9983、音频 cos 0.9998 |
| 视频 VAE 解码 | cos 1.000000 |
| 视频 VAE 编码 | cos 1.000000 |
| 音频 VAE 解码 | cos 0.999995 |

英文版与完整命令见 [minimax-h3.md](minimax-h3.md)。
