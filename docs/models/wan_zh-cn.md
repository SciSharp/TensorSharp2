# Wan 视频生成（2.1 文生视频 · 2.2 文/图生视频）

[English](wan.md) | [中文](wan_zh-cn.md)

TensorSharp 原生运行 [Wan 2.1](https://github.com/Wan-Video/Wan2.1) 与
[Wan 2.2](https://github.com/Wan-Video/Wan2.2) 视频扩散模型：输入提示词
（Wan 2.2 模型可再加一张首帧图片），输出 H.264 MP4，`TensorSharp.Cli` 与
`TensorSharp.Server`（OpenAI 风格 API + 自带 Web UI 聊天）均可驱动。

支持的模型家族（按 DiT GGUF 自动识别）：

| 家族 | 潜空间 | VAE | 模式 | 说明 |
|---|---|---|---|---|
| Wan 2.1 T2V（1.3B / 14B） | 16 通道，8×8×4 | wan_2.1_vae | 文生视频 | 单 DiT |
| Wan 2.2 TI2V-5B | 48 通道，16×16×4 | Wan2.2_VAE | 文生 + **图生视频** | 稠密 5B，24 fps，可达 720p |
| Wan 2.2 A14B（T2V / I2V） | 16 通道（I2V 输入 36 通道） | wan_2.1_vae | 文生 + **图生视频** | 两个 14B 专家按时间步边界切换 |

图生视频时，上传的图片成为视频的**首帧**，文本提示词控制动作、镜头运动、
氛围与场景变化。

各网络放在同一目录即可自动解析（`VAE/`、`HighNoise/`、`LowNoise/` 等子目录亦可）：

| 组件 | 文件 | 来源 |
|---|---|---|
| DiT（`--model` GGUF，`general.architecture = wan`） | 如 `Wan2.2-TI2V-5B-Q8_0.gguf` | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)、[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)、[city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf) |
| A14B 第二专家（仅 A14B） | 对应的 `…HighNoise…`/`…LowNoise…` GGUF | 同一仓库（按文件名自动配对；`TS_WAN_DIT2` 可覆盖） |
| UMT5-XXL 文本编码器 | `umt5-xxl-encoder-Q8_0.gguf` | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf)（`--wan-te` / `TS_WAN_TE`） |
| Wan 2.1 视频 VAE（2.1 + A14B） | `wan_2.1_vae.safetensors` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors)（`--wan-vae` / `TS_WAN_VAE`） |
| Wan 2.2 视频 VAE（TI2V-5B） | `Wan2.2_VAE.safetensors` | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE) 内附 |

文本编码器在去噪开始前、（图生视频时）VAE 编码器在 DiT 加载前、DiT 在 VAE 解码前
分别释放 VRAM，因此峰值显存约为 `max(TE, DiT + 注意力, VAE)` —— TI2V-5B Q8_0
可在 16 GB GPU 上不到 8 分钟生成 81 帧 480p 图生视频；两个 A14B Q4_K_M 专家
可在同一张卡上顺序运行。

## CLI

文生视频（任意家族）：

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "一只可爱的毛茸茸的橘猫走过开满鲜花的阳光花园" \
  --output cat.mp4 --width 832 --height 480 --video-frames 49 --diffusion-seed 7
```

图生视频（Wan 2.2 模型，加 `--image`）：

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "猫向镜头跑来，动感追踪镜头，黄金时刻光线" \
  --image first_frame.png --output cat_run.mp4 --video-frames 81
```

```bash
# A14B：--model 指向任一专家即可，另一专家自动配对
TensorSharp.Cli --model HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf \
  --backend ggml_cuda --prompt "帆船驶入风暴，巨浪拍打" --image ship.jpg --output ship.mp4
```

- `--image <文件>` —— 首帧条件图（PNG/JPEG/WebP/…），自动应用 EXIF 旋转。
  未显式指定 `--width/--height` 时，输出分辨率按图片纵横比适配模型原生面积
  （TI2V-5B：1280×704，即官方 720p 配方；其余：832×480）。
- `--video-frames N` —— 帧数，对齐到 `4k+1`；`1` 生成静态图（配合 `--output out.png`）。
- `--sampler unipc|euler` —— UniPC（默认）为官方采样器，数值上对照 diffusers 验证。
- `--cfg` / `--flow-shift` / `--diffusion-steps` 默认为各模型官方配方：TI2V-5B
  cfg 5.0、shift 5.0、50 步、24 fps；A14B I2V cfg 3.5（双专家）、shift 5.0、40 步；
  A14B T2V cfg 4.0/3.0、shift 12.0；Wan 2.1 cfg 6.0、shift 8.0（1.3B 视频）或
  3.0/5.0、30 步、16 fps。
- MP4 输出优先使用 `PATH` 上的 `ffmpeg`（或 `TS_FFMPEG=<path>`，或可执行文件旁的
  `ffmpeg` 目录）—— 近无损 CRF 17 H.264。没有 ffmpeg 时回退到系统编码器
  （OpenCV），默认码率画质明显更差，CLI 会给出警告。

### 获得好画质

- **分辨率是最大的杠杆。** Wan 的训练分辨率为 480p（832×480 面积）与
  720p（1280×704）；TI2V-5B 原生是 720p 模型。远低于 480p 面积时 DiT 处于
  分布外，无论多少步都会输出模糊、不稳定、偏色的视频——低于约 0.3 MP 时
  管线会打印警告。请在受支持的分辨率下生成后再缩小（例如用
  `--width 480 --height 704` 而不是 320×480）。
- **把镜头描述完整。** Wan 对详细提示词（主体、动作、运镜、光线、风格）的
  跟随远好于三四个词的短语；官方演示会先用 LLM 做"提示词扩写"。过短的
  提示词让模型缺乏约束、容易漂移。
- 超过官方配方（50）继续加 `--diffusion-steps` 收益很小；优先解决分辨率
  与提示词。

## 服务器

```bash
TensorSharp.Server --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --video-frames 121 --fps 24
```

`--video-frames` 与 `--fps` 为 Web UI 和下述三个视频端点设置服务启动默认值。
Web UI 不会发送这两个字段，因此上例会以 24 fps 生成 121 帧（播放时长约 5.04 秒）。
API 请求显式提供的 `frames` 或 `fps` 会分别覆盖对应的启动默认值；这些值是默认值，
不是上限。如果启动参数和请求字段均省略，则采用模型配方：TI2V-5B 为 49 帧 / 24 fps，
其余模型为 33 帧 / 16 fps。帧数会对齐到 `4k+1`。要调整时长，建议保持模型原生
帧率并修改帧数；仅改变 FPS 会改变播放速度。

- **Web UI**（`http://localhost:5000`）：在聊天框输入提示词 —— 附带一张图片即为
  图生视频（图片成为首帧）；回复即是带实时去噪进度的生成视频。
- **API**：

```bash
curl http://localhost:5000/v1/videos/generations -H "Content-Type: application/json" -d '{
  "prompt": "猫向镜头跑来，电影感",
  "image": "data:image/png;base64,....",
  "size": "832x480", "frames": 81, "steps": 50, "seed": 7
}'
```

  `image` 接受 base64（可带 `data:` 前缀），省略即为文生视频。另有
  `POST /api/video-generate`（同样的 JSON，支持 `imagePath` 引用已上传文件）与
  `POST /api/video-generate/stream`（SSE 进度，Web UI 使用）。A14B 各端点均支持
  `cfg2`（低噪专家引导系数）。

## 性能

RTX 3080 Laptop 16 GB（Windows/WDDM），ggml_cuda：

| 模型 / 工作负载 | 文本编码 | 图像编码 | 去噪 | VAE 解码 | 总计 |
|---|---|---|---|---|---|
| **Wan2.2-TI2V-5B Q8_0** 图生视频 832×480×81 帧、30 步 | 3.5 s | 8.5 s | 11.5 s/步（CFG ×2，8190 token） | 103 s | 464 s |
| Wan2.2-TI2V-5B Q8_0 文生视频 640×384×25 帧、20 步 | 3.7 s | — | 1.5 s/步 | 23 s | 65 s |
| Wan2.2-I2V-A14B Q4_K_M 480×480×9 帧、6 步（冒烟） | 3.4 s | 5 s | 7–8 s/步 + 两次专家加载 | 7.4 s | 89 s |
| Wan2.1-T2V-1.3B F16 832×480×33 帧、30 步 | 3.1 s | — | 9.4 s/步（CFG ×2） | 51 s | 337 s |

TI2V-5B 的 16×16 空间压缩使同分辨率下 DiT token 数约为 Wan 2.1 的 1/2.7 ——
它是消费级 GPU 上速度最快、质量最高的选择，也是唯一的 720p-24fps 模型。
数值验证、实现细节与环境变量见[英文卡片](wan.md)。
