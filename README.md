# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

**Native .NET LLM inference engine for GGUF models** — autoregressive LLMs *and* DiffusionGemma-style text-diffusion, plus Qwen-Image-Edit image editing and Wan 2.1/2.2 text- and image-to-video. Ships a console app, a browser chat UI, and Ollama/OpenAI-compatible HTTP APIs. A pure-.NET engine that trades wins with the hand-tuned C++ `llama.cpp` on identical GGUF files and the same GPU.

## From Tensors to Tokens — the TensorSharp book

<p align="center">
  <a href="https://www.amazon.com/dp/B0H9P44QZZ">
    <img src="website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="220">
  </a>
</p>

**[From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B](https://www.amazon.com/dp/B0H9P44QZZ)** by Zhongkai Fu turns this repository into a guided, end-to-end learning journey. It uses Gemma 4 E4B to connect tensor fundamentals, model execution, multimodal inputs, and the application surfaces of a working LLM inference engine.

**[Explore the book and its repository reading path](docs/BOOK.md)** · **[Buy the paperback on Amazon](https://www.amazon.com/dp/B0H9P44QZZ)**

## Highlights

- **⚡ Trades wins with llama.cpp — from pure .NET.** On identical GGUF files and the same GPU, TensorSharp matches or beats `llama.cpp` on the workloads that matter: Gemma 4 E4B and 2-bit Qwen 3.6 35B-A3B MoE prefill **1.28×** faster on CUDA with first tokens **1.27×** sooner (multi-turn up to **1.49×**); Gemma 4 12B decodes **1.21×** faster on Vulkan (up to **1.32×** on long context). → [Benchmarks](#benchmarks)
- **🚀 Continuous batching & paged KV cache.** vLLM-style paged KV pool with block-hash prefix sharing and an iteration-level scheduler, on by default in the server. → [deep dive](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)
- **🧬 DeepSeek V4 Flash (284B MoE) with three whole-model executors.** The compressed-sparse-attention 1M-context architecture runs on a direct-CUDA engine (`--backend cuda`), the native ggml executor (`--backend ggml_cuda` / `ggml_vulkan`), *and* a 100% pure-C# CPU executor (`--backend cpu`, no native dependencies). Weights layer-split automatically across every visible GPU, so a model far larger than one card still runs; the server hosts it with per-sequence slots and continuous batching. → [DeepSeek V4 card](docs/models/deepseek4.md)
- **🔮 Speculative decoding — MTP / NextN *and* DSpark.** Multi-token-prediction draft heads accelerate solo decode on Qwen 3.6 (NextN block embedded in the trunk — use an MTP-retaining GGUF such as [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF); the base repo ships the same file names with the block stripped) and Gemma 4 (separate `gemma4-assistant` draft GGUF, `--mtp-draft-model`); DeepSeek V4 adds **DSpark** block drafting (`--draft-model`), which proposes a whole block of tokens per step for **1.3–1.4× decode** (up to 2.0× on multi-turn chat). In every case the draft proposes, the trunk verifies in one batched forward, and the output matches standard decode. → [Speculative decoding](FEATURES.md#mtp--nextn-speculative-decoding)
- **🔗 Tensor parallelism & distributed clustering.** Split a model across multiple GPUs with `--tp N` — on the direct `cuda` backend **and** on GGML CUDA / Vulkan — and extend across machines with peer-to-peer TCP clustering (`--tp-node-id` / `--tp-peers`). Megatron-LM column/row-parallel pattern with hierarchical AllReduce; MoE expert parallelism and per-rank GatedDeltaNet kernels on GGML. Fused per-rank execution makes `--tp 2` decode **1.39×** a single GPU on Gemma 4 E4B and **1.57×** on Muse-Glimmer 30B (which also gains **1.34×** prefill — the one model that beats a single GPU on both phases), and runs models that do not fit one card at all (Qwen 3.5-35B-A3B; Muse-Glimmer 30B Q8_0 at 28.2 GB on 24 GB cards). Optional Redis-backed KV cache and Responses API store. → [Tensor Parallelism](USAGE.md#tensor-parallelism--distributed-inference)
- **🎨 Qwen-Image-Edit image editing.** Prompt + input image → edited image, driving a 60-block MMDiT with a Qwen-Image VAE and Qwen2.5-VL-7B text encoder. CUDA-graph-captured DiT, FlowMatch-Euler true-CFG denoise, live Web UI previews, and a [Lightning distillation LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) fast path (`--qwen-image-lora`, applied as a runtime side-path over the untouched quantized weights) that takes the default 30 steps × CFG — 60 DiT forwards — down to **4**. Beat `stable-diffusion.cpp` **1.19×** on a warm 4-step edit. → [Qwen-Image-Edit card](docs/models/qwenimage.md)
- **🎬 Wan 2.1 / 2.2 video generation (text → video and image → video).** Prompt → H.264 MP4; on the Wan 2.2 models (TI2V-5B, I2V-A14B) an uploaded image becomes the video's first frame while the prompt drives motion, camera and scene changes. One resident-weight ggml graph per denoise step (CUDA-graph-captured, flash attention, per-token-timestep modulation for TI2V i2v), causal 3D video VAE encode+decode each as a single graph, A14B's two 14B experts hot-swapped at the timestep boundary, stagewise VRAM handoff — TI2V-5B generates 81-frame 480p image-to-video on a 16 GB GPU in under 8 min, and Wan 2.1 runs **6.0×** faster end-to-end than `stable-diffusion.cpp` on the identical workload. **Step-distilled checkpoints are auto-detected from the DiT file name** (`Turbo` / `distill` / `Lightning` / `lightx2v` / `FastWan` / `…-4steps-…`) and switch to that step count with guidance off — 4 DiT passes instead of the official recipe's 100, which took the same 1088×832×121-frame image-to-video from **3 h 30 m to 17 m 30 s** on an M5 Pro. It is the single biggest speed lever in the repository and needs no flag, only a different `--model` file. Numerics verified against diffusers (DiT cos > 0.995, VAE encoders cos > 0.999, decode 59.9 dB / >35 dB PSNR). CLI (`--image`), `/v1/videos/generations`, and Web UI chat with image upload. → [Wan card](docs/models/wan.md)
- **🌫️ DiffusionGemma text diffusion.** Block-wise EntropyBound denoising over a Gemma-4-derived MoE backbone, with CLI flags and a Web UI denoising preview stream. → [DiffusionGemma card](docs/models/diffusiongemma.md)
- **🖼️ Multimodal.** Image / video / audio (Gemma 4); image input for Gemma 3, Qwen 3.5-family, Mistral 3, Nemotron-H Omni, and Muse-Glimmer; PDF documents via CLI and Web UI. → [Multimodal](FEATURES.md#multimodal-support)
- **🛠️ Tool calling & thinking mode.** Multi-turn tool calls and structured chain-of-thought across Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H, Muse-Glimmer (ATEM markup), and DeepSeek V4 (DSML markup). → [Features](FEATURES.md)
- **🔌 Ollama- & OpenAI-compatible APIs** plus a browser chat UI — drop-in for existing tooling. → [HTTP APIs](USAGE.md#http-apis)
- **📄 Config files with auto-download.** Put CLI/Server options in a reusable JSON file with `${variables}` and `{ "path", "urls" }` entries that fetch the model on first run. → [config/README.md](config/README.md)
- **🧮 Native quantized compute.** Q4_K_M / Q8_0 / MXFP4 / IQ2_XXS / IQ2_S / IQ3_S / IQ3_XXS / IQ4_XS and more run in matmul without dequantizing to FP32. Runs on GGML Metal / CUDA / Vulkan, a direct CUDA/cuBLAS backend, MLX (Apple Silicon), and a pure-C# CPU path — all with CPU fallbacks. On MLX the IQ decode kernels amortize the codebook and scale loads across a whole sub-block instead of re-reading them per weight, which took a mixed-IQ 30B from 3.6 to **14.3 tok/s** on an M5 Pro (67% of the fused ggml-metal graph, output byte-identical). → [Backends](USAGE.md#compute-backends)

## Quick Start

TensorSharp targets .NET 10. On a new machine, install the full **.NET 10 SDK**—the .NET Runtime alone cannot build TensorSharp:

| Platform | Install the SDK |
|---|---|
| **Windows** | In PowerShell, run `winget install Microsoft.DotNet.SDK.10`, or use Microsoft's [.NET installation guide for Windows](https://learn.microsoft.com/en-us/dotnet/core/install/windows). |
| **macOS** | Use the [.NET 10 SDK installer](https://dotnet.microsoft.com/en-us/download/dotnet/10.0): choose **Arm64** for Apple silicon or **x64** for an Intel Mac. See Microsoft's [macOS instructions](https://learn.microsoft.com/en-us/dotnet/core/install/macos). |
| **Linux** | Follow Microsoft's [Linux distribution guide](https://learn.microsoft.com/en-us/dotnet/core/install/linux) to configure the correct package source for your distro and install its .NET 10 SDK package (commonly `dotnet-sdk-10.0`). |

Open a new terminal and verify that a `10.0.x` SDK is listed:

```bash
dotnet --list-sdks
```

See the [cross-platform .NET install overview](https://learn.microsoft.com/en-us/dotnet/core/install/) or [Development → Prerequisites](DEVELOPMENT.md#prerequisites) for more detail.

Then get running in ~30 seconds on the verified native GGML fast path — Gemma 4 E4B. The other prerequisites are `git`, `curl`, and the toolchain for your GPU backend (see [Development → Prerequisites](DEVELOPMENT.md#prerequisites)). The recommended public file is [`gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf) (7.48 GiB); text-only inference needs no projector.

**Windows + NVIDIA (PowerShell)**

```powershell
git clone https://github.com/zhongkaifu/TensorSharp.git; Set-Location TensorSharp
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models\gemma-4-E4B-it-Q8_0.gguf
'Answer in one short sentence: what is TensorSharp?' | Set-Content prompt.txt
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models\gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 128 --backend ggml_cuda
```

**macOS (Apple Silicon)** — drop the CUDA env var and use `--backend ggml_metal`.
**Linux + NVIDIA** — prefix the `dotnet run` with `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` and use `--backend ggml_cuda`.
**AMD / Intel / NVIDIA Vulkan** — set `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON` and use `--backend ggml_vulkan`.

**Linux (Ubuntu) + multiple NVIDIA GPUs — tensor parallelism**

Tensor parallelism splits one model across N GPUs. It runs on the direct
`cuda` backend and on the GGML CUDA / Vulkan backends (`--backend ggml_cuda`,
`ggml_vulkan`). Install the CUDA toolkit first, then:

```bash
# On RunPod's Ubuntu 24.04 images, point the loader at the CUDA compat libraries first:
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH
# On older Ubuntu releases the .NET 10 SDK comes from the backports PPA:
add-apt-repository ppa:dotnet/backports

apt update && apt install dotnet-sdk-10.0
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
mkdir models
wget "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -O models/gemma-4-E4B-it-Q8_0.gguf
bash TensorSharp.GGML.Native/build-linux.sh
dotnet build -c Release

# 2 GPUs in one process
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend cuda --interactive --max-tokens 20000 --tp 2

# Same thing on the GGML CUDA backend (add TENSORSHARP_TP_DEVICES=0,2 to pick GPUs)
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_cuda --interactive --max-tokens 20000 --tp 2
```

Scale the same model across machines by adding a node ID and the shared peer
list — 2 nodes × 2 GPUs gives a global TP degree of 4:

```bash
# Node 0
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# Node 1 (same peer list, different node ID)
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
```

`TensorSharp.Server` takes the same `--tp`, `--tp-node-id`, and `--tp-peers`
flags (or the `TENSORSHARP_TP_*` environment variables); in a multi-node
cluster the server is node `0` — the driver that serves HTTP — and every other
node runs a `TensorSharp.Cli` worker. Full reference:
**[Tensor Parallelism & Distributed Inference](USAGE.md#tensor-parallelism--distributed-inference)**.


Host the same model as a server (browser UI at <http://localhost:5000>, plus Ollama/OpenAI APIs):

```bash
dotnet run --project TensorSharp.Server -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> The server binds `0.0.0.0:5000` by default (change it with `--port` / `--host`, or the `PORT` / `HOST` environment variables; on macOS port 5000 is taken by the AirPlay Receiver) with no built-in auth or TLS — keep it behind a firewall or an authenticated HTTPS reverse proxy. For image/video/audio add the companion [`mmproj-gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf) with `--mmproj`.

Both executables print their full option reference — description, default, range, and an example per flag — when started with no arguments or with `--help`:

```bash
dotnet run --project TensorSharp.Cli -c Release -- --help
dotnet run --project TensorSharp.Server -c Release -- --help
```

Full command reference: **[CLI](USAGE.md#console-application)** · **[Server](USAGE.md#web-application)** · more models to download: **[Model Downloads](MODEL_DOWNLOADS.md)** · prefer a config file? **[config/](config/README.md)**.

## Pick a Backend

Every backend falls back to CPU for any op it does not implement, so output stays correct on all of them.

| Your hardware | Recommended backend | Flag | Notes |
|---|---|---|---|
| **Apple Silicon (Mac)** | GGML Metal | `--backend ggml_metal` | Default on macOS. `--backend mlx` is an alternative Apple-Silicon GPU path. |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | Most-tested NVIDIA path. `--backend cuda` is the direct PTX/cuBLAS backend for experimentation. |
| **Windows / Linux + AMD / Intel / NVIDIA GPU** | GGML Vulkan | `--backend ggml_vulkan` | Vendor-neutral GPU path via ggml-vulkan. Built automatically when a Vulkan runtime is present; `--no-vulkan` opts out. |
| **No GPU / portability / debugging** | Pure C# CPU | `--backend cpu` | No native dependencies. For faster CPU inference use `--backend ggml_cpu` (native kernels). |

Full per-backend description: [Usage → Compute Backends](USAGE.md#compute-backends).

## Verified Models

Implemented and exercised by the test/benchmark matrix. Pick a quantization that fits your hardware (Q4_K_M for low memory, Q8_0 for higher quality). More sizes and projector files: [Model Downloads](MODEL_DOWNLOADS.md).

| Family | Example model (GGUF) | Image / Video / Audio | Thinking | Tools | Card |
|---|---|---|---|---|---|
| DeepSeek V4 Flash | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) (284B MoE, split GGUF) | — / — / — | ✅ | ✅ | [deepseek4.md](docs/models/deepseek4.md) |
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) (also 31B, 26B-A4B MoE) | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4.md](docs/models/gemma4.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) (also 35B-A3B MoE) | ✅ / — / — | ✅ | ✅ | [qwen35.md](docs/models/qwen35.md) |
| Qwen 3 | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | — / — / — | ✅ | ✅ | [qwen3.md](docs/models/qwen3.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) (MoE) | — / — / — | ✅ | ✅ | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) (also 47B, Omni) | ✅ (Omni) / — / — | ✅ | ✅ | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3.md](docs/models/mistral3.md) |
| Muse-Glimmer | [Muse-Glimmer-30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) (+ mmproj) | ✅ / — / — | ✅ | ✅ | [muse-glimmer.md](docs/models/muse-glimmer.md) |
| Gemma 3 | [gemma-3-4b-it](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) | ✅ / — / — | — | — | [gemma3.md](docs/models/gemma3.md) |
| DiffusionGemma | [diffusiongemma-26B-A4B-it](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | — / — / — | — | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |
| Qwen-Image-Edit | [Qwen-Image-Edit-2511](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) (MMDiT + VAE + Qwen2.5-VL) · fast lane: [Lightning 4-step LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | 🖼️ image→image | — | — | [qwenimage.md](docs/models/qwenimage.md) |
| Wan 2.1 / 2.2 video | [Wan2.2-TI2V-5B](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF) (also [T2V-A14B](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF), [I2V-A14B](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF), [Wan2.1-T2V-14B](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)) + UMT5-XXL + video VAE · fast lane: [TI2V-5B-Turbo](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF) (4-step, 25× fewer DiT passes) | 🎬 text→video, image→video | — | — | [wan.md](docs/models/wan.md) |

## Make It Fast

Several families have a *fast lane* — a different artifact to download, or one flag — that changes the cost of a run by an order of magnitude. Reach for these before tuning anything else.

| Family | Fast artifact or flag | Measured effect |
|---|---|---|
| **Wan 2.1 / 2.2 video** | A **step-distilled DiT GGUF** — e.g. [hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF) (`Wan2_2-TI2V-5B-Turbo-Q8_0.gguf`) for TI2V-5B, or [jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF) for A14B. No flag — detected from the file name. | 100 DiT passes → **4**, guidance off. The same 1088×832×121f image-to-video: **3 h 30 m → 17 m 30 s** (M5 Pro, `ggml_metal`). |
| Wan, base checkpoints only | `--cfg-cache-stride 2` / `3` | 1.30× / 1.43× at 50 steps (approximate; pointless on a distilled checkpoint, which is already guidance-free). |
| Wan, any checkpoint | Generate at a trained resolution and downscale — 736×544 instead of 1088×832 | 121 frames, Turbo checkpoint: **6 m 19 s** instead of 17 m 30 s. Below ~0.3 MP quality falls off instead. |
| **Qwen-Image-Edit** | `--qwen-image-lora` with the [Lightning 4-step LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) (`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`) | Sampling defaults switch from 30 steps at CFG 2.5 (60 DiT forwards) to **4** at CFG 1.0. A warm 4-step edit beat `stable-diffusion.cpp` **1.19×**. |
| Qwen-Image-Edit | `TS_QWEN_DIT_CACHE_MODE=easycache` (off by default — quality first) | Skips 40–55% of denoise steps; measurably softens fine detail on edits, which is why it is opt-in. |
| **DeepSeek V4 Flash** | `--draft-model` with a [DSpark drafter GGUF](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF) (server: add `--mtp-spec`); `cuda` / `ggml_cuda` only | Decode 26.4 → **37.1 tok/s** (1.41×) on 4×A40, 69% acceptance; up to **2.0×** on multi-turn chat. Output is unchanged — the trunk verifies every block. |
| **Muse-Glimmer** | `--draft-model` with the DFlash drafter (`dflash-kquant.gguf`, in [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)); pass **no** sampler flags | 1.3–5× decode on the CUDA hosts it was built on — 35.0 → **50.9 tok/s** greedy at a 60-token prompt on one RTX PRO 6000. On Apple Silicon plain decode is still faster today. |
| **Qwen 3.6** | An MTP-retaining GGUF — [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), not the base repo — plus the server's `--mtp-spec` | Enables NextN speculative decode on solo sequences. The base repo ships the same file names with the block stripped and silently falls back. |
| **Gemma 4** | `--mtp-draft-model` with the matching [`gemma4-assistant` draft GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) plus `--mtp-spec` (server) | Speculative decode on GGML backends and the direct `cuda` backend. Draft and target hidden sizes must match, or startup fails. |
| Any MoE that does not fit the card | `--n-cpu-moe N` / `--cpu-moe` | gpt-oss-20b 16.2 → 2.9 GB VRAM on a 16 GB laptop card, turning the WDDM spill cliff's 0.3 tok/s into **25.4** at `--n-cpu-moe 12`. |
| Multi-GPU | `--tp N` | Gemma 4 E4B decode **1.39×** a single GPU, Muse-Glimmer 30B **1.57×** decode / **1.34×** prefill — and it runs models that fit on no single card. |
| Every family | Pick the right backend: `ggml_cuda` on NVIDIA, `ggml_metal` on Apple Silicon, `ggml_cpu` (not `cpu`) without a GPU | Gemma 4 26B-A4B decodes 78.7 tok/s on `ggml_cuda` vs 35.3 on the direct `cuda` backend; on Apple Silicon Muse-Glimmer 30B prefills 413.6 tok/s on `ggml_metal` vs 29.0 on MLX. |

Per-family detail, including the numbers behind every row: [Wan](docs/models/wan.md) · [Qwen-Image-Edit](docs/models/qwenimage.md) · [DeepSeek V4](docs/models/deepseek4.md) · [Muse-Glimmer](docs/models/muse-glimmer.md) · [Features](FEATURES.md).

## Supported Model Architectures

| Architecture | GGUF arch keys | Example Models | Multimodal | Thinking | Tools | MTP spec | Card |
|---|---|---|---|---|---|---|---|
| DeepSeek V4 Flash | `deepseek4` | DeepSeek-V4-Flash (284B MoE, 256 experts, compressed sparse attention, 1M context) | Text only | Yes | Yes (DSML) | Yes (DSpark block drafter, separate GGUF) | [deepseek4.md](docs/models/deepseek4.md) |
| Gemma 4 | `gemma4` | gemma-4-E4B, gemma-4-31B, gemma-4-26B-A4B (MoE) | Image, Video, Audio | Yes | Yes | Yes (separate draft GGUF) | [gemma4.md](docs/models/gemma4.md) |
| Gemma 3 | `gemma3` | gemma-3-4b | Image | No | No | — | [gemma3.md](docs/models/gemma3.md) |
| Qwen 3 | `qwen3`, `qwen2`, `qwen2vl`, `qwen2_vl` | Qwen3-4B (Qwen2 / Qwen2.5-VL GGUFs also load, as text-only chat) | Text only | Yes | Yes | — | [qwen3.md](docs/models/qwen3.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B (hybrid Attn+Recurrent), Qwen3.5/3.6-35B-A3B (MoE) | Image | Yes | Yes | Yes on Qwen 3.6 (embedded NextN) | [qwen35.md](docs/models/qwen35.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b (MoE) | Text only | Yes (always) | Yes | — | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B/47B (Hybrid SSM-Transformer, MoE), Nemotron 3 Nano Omni | Image (Omni) | Yes | Yes | — | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | Image | No | No | — | [mistral3.md](docs/models/mistral3.md) |
| Muse-Glimmer | `muse-glimmer`, `muse_glimmer` | Muse-Glimmer-30B (interleaved SWA + NoPE full layers, attention output gate) | Image | Yes | Yes (ATEM) | Yes (DFlash block drafter, separate GGUF) | [muse-glimmer.md](docs/models/muse-glimmer.md) |
| DiffusionGemma | `diffusion-gemma`, `diffusion_gemma` | diffusion-gemma text-diffusion GGUFs | Text only | No | No | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |
| Qwen-Image-Edit | `qwen_image`, `qwen-image` | qwen-image-edit MMDiT GGUFs (+ VAE & Qwen2.5-VL) | Image edit (image+text → image) | No | No | — | [qwenimage.md](docs/models/qwenimage.md) |
| Wan video | `wan`, `wan2.1`, `wan2.2` | Wan 2.1 T2V 1.3B/14B, Wan 2.2 TI2V-5B, Wan 2.2 A14B T2V/I2V (two experts) | Video out (text→video, image→video) | No | No | — | [wan.md](docs/models/wan.md) |

End-to-end per-model documentation (origin, forward graph, components, parameters, prefill/decode optimizations): [architecture cards](docs/models/README.md).

## Benchmarks

### Head-to-head vs llama.cpp (engine comparison)

A pure-.NET engine going toe-to-toe with the hand-tuned C++ `llama.cpp` on **identical GGUF files, the same NVIDIA RTX 3080 Laptop GPU (16 GB), and one uniform OpenAI `/v1/chat/completions` surface** — with **both engines measured on their GGML CUDA and Vulkan builds**. Numbers are the **geomean speedup of TensorSharp over llama.cpp on the same backend** (single-stream, greedy, MTP off); **> 1.0× means TensorSharp is faster / lower-latency**. Full per-scenario tables: [`docs/engine_comparison_report.md`](docs/engine_comparison_report.md).

| Model | Backend | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it (Q8_0, dense multimodal) | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it (Q8_0, dense multimodal) | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | Vulkan | 1.02× | 0.85× | 0.84× |

TensorSharp pulls clearly ahead on CUDA prefill / first-token latency (multi-turn prefill wins on **every** model, up to **1.49×**), holds decode parity-or-better on CUDA, and wins Vulkan decode on the dense 12B (up to **1.32×** on long context) — even at 2-bit IQ2_XXS quantization. The remaining sub-1.0× cells are active optimization targets. The harness also covers tool-calling, structured-output, image-edit (vs `stable-diffusion.cpp`), MTP on/off, and parallel-request scenarios you can run yourself via [`benchmarks/engine_comparison`](benchmarks/engine_comparison). Every cell is in the [full report](docs/engine_comparison_report.md).

## Documentation

New here? The sections above are all you need to get running. Everything else is detailed reference:

| Doc | What's inside |
|---|---|
| [Book guide: From Tensors to Tokens](docs/BOOK.md) | A guided path from tensor fundamentals to a multimodal Gemma 4 E4B inference engine, with publication details and links into the companion repository |
| [Model Downloads](MODEL_DOWNLOADS.md) | Per-model `huggingface-cli` download + run quick reference (quant tiers, projectors, companions) |
| [Usage](USAGE.md) | Full CLI reference (options, interactive REPL, JSONL batch), server hosting, logging, HTTP API examples, backends, and the env-var matrix |
| [Features](FEATURES.md) | Deep dives on continuous batching, MTP speculative decoding, tool calling, thinking mode, multimodal, MoE, KV codecs, and more |
| [Configuration files](config/README.md) | Put options in a reusable JSON file with `${variables}` and auto-downloading models |
| [Development](DEVELOPMENT.md) | Prerequisites, building the native GGML/MLX libraries, repository layout, package boundaries, internal architecture, and the test harness |
| [Per-model architecture cards](docs/models/README.md) | End-to-end docs of each architecture (forward graph, components, parameters, prefill/decode optimizations) |
| [Paged attention & continuous batching](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | The vLLM-style paged KV cache, prefix sharing, and iteration-level scheduler |
| [Environment variable feature matrix](docs/env_var_feature_matrix.md) | Which high-impact runtime flags affect which models, backends, and prompt types |
| [Engine comparison report](docs/engine_comparison_report.md) | Full per-scenario TensorSharp vs llama.cpp / stable-diffusion.cpp tables |
| [Test/benchmark matrix runner](TensorSharp.TestMatrix/README.md) | Sweep model × backend × feature × env-var cells and generate regression reports |
| [Server API examples](TensorSharp.Server/API_EXAMPLES.md) | Complete curl and Python examples for the server surface |

## Current Status

| Area | Status |
|---|---|
| Model families | DeepSeek V4 Flash (`deepseek4`), Gemma 3/4, DiffusionGemma, Qwen 3, Qwen 3.5/3.6-family (`qwen35`, `qwen35moe`, `qwen3next`), GPT OSS, Nemotron-H (incl. Nemotron 3 Nano Omni), Mistral 3, Muse-Glimmer (`muse-glimmer`, `muse_glimmer`). Image editing via Qwen-Image-Edit (`qwen_image`, `qwen-image` MMDiT); video generation via Wan 2.1 / 2.2 (`wan`, `wan2.1`, `wan2.2`). |
| Inference hosts | CLI, interactive REPL, ASP.NET Core web UI, Ollama-style API, OpenAI Chat Completions-style API. |
| Backends | Pure C# CPU, direct CUDA/cuBLAS (`cuda`), MLX Metal (`mlx`), GGML CPU, GGML Metal, GGML CUDA, GGML Vulkan. DeepSeek V4 additionally has three whole-model executors of its own — direct-CUDA, native ggml, and a pure-C# CPU one — each layer-splitting the weights across every visible GPU (`--tp N` / `TS_DSV4_NGPU` caps the count). Wan is the one family that restricts its backends: it runs on the GGML backends and on the direct `cuda` / pure-C# `cpu` ones, but not on MLX. |
| Multimodal | Gemma 4 image/video/audio; Gemma 3, Qwen 3.5-family, Mistral 3, Nemotron-H Omni, Muse-Glimmer image input; PDF documents (CLI `--pdf` + Web UI). Media *out*: Qwen-Image-Edit (image) and Wan 2.1 / 2.2 (H.264 MP4 video, text→video and image→video). |
| Continuous batching | vLLM-style paged KV cache, block-hash prefix sharing, iteration-level scheduler (default on; opt-out `--no-continuous-batching`). DeepSeek V4 serves through its own native per-sequence slots on the same engine. |
| Speculative decoding | MTP / NextN draft heads on Qwen 3.6 (embedded) and Gemma 4 (separate draft GGUF); DSpark block drafting on DeepSeek V4 (`cuda` / `ggml_cuda` only) and DFlash block drafting on Muse-Glimmer, both loading a separate drafter GGUF via `--draft-model`. Verification is greedy against the target, so the emitted stream is the plain-greedy stream. Off by default; opt-in via the server's `--mtp-spec`, or by passing `--draft-model` on the CLI. |
| Tensor parallelism | Megatron-LM column/row-parallel TP on the direct `cuda` backend and on GGML CUDA / Vulkan (`--tp N` / `TENSORSHARP_TP_DEGREE`, CLI and server); distributed multi-node TP via peer-to-peer TCP (`--tp-node-id` / `--tp-peers`), with hierarchical AllReduce and automatic host-staging fallback when CUDA P2P is unavailable. All autoregressive architectures; MoE expert parallelism and fused per-rank decode/prefill graphs for Gemma 4 and Qwen 3.5/3.6 on GGML. Optional Redis-backed KV cache and Responses API store. |
| Server model scope | One explicitly hosted GGUF via `--model`; optional explicit projector via `--mmproj`; no directory scanning. |
| Observability | Structured per-turn logs, queue status, and KV-cache reuse metrics across Web UI, Ollama, and OpenAI shapes. |

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
