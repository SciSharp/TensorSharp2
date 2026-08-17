# Wan Video Generation (2.1 Text-to-Video · 2.2 Text/Image-to-Video)

TensorSharp runs the [Wan 2.1](https://github.com/Wan-Video/Wan2.1) and
[Wan 2.2](https://github.com/Wan-Video/Wan2.2) video diffusion models natively:
prompt (plus an optional first-frame image on the Wan 2.2 models) in, H.264 MP4
out, from both `TensorSharp.Cli` and `TensorSharp.Server` (OpenAI-style API +
the bundled Web UI chat).

Supported checkpoint families (auto-detected from the DiT GGUF):

| Family | Latent | VAE | Modes | Notes |
|---|---|---|---|---|
| Wan 2.1 T2V (1.3B / 14B) | 16 ch, 8×8×4 | wan_2.1_vae | T2V | single DiT |
| Wan 2.2 TI2V-5B | 48 ch, 16×16×4 | Wan2.2_VAE | T2V + **I2V** | dense 5B, 24 fps, 720p-capable |
| Wan 2.2 A14B (T2V / I2V) | 16 ch (36 ch I2V input) | wan_2.1_vae | T2V + **I2V** | two 14B experts switched at a timestep boundary |

For image-to-video the uploaded image becomes the **first frame** of the video
and the text prompt drives motion, camera work, atmosphere and scene changes.

The networks cooperate per generation (all resolved automatically when placed
in one folder — subfolders like `VAE/`, `HighNoise/`, `LowNoise/` included):

| Piece | File | Source |
|---|---|---|
| DiT (the `--model` GGUF, `general.architecture = wan`) | e.g. `Wan2.2-TI2V-5B-Q8_0.gguf` | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF), [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF), [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf) |
| second A14B expert (A14B only) | the matching `…HighNoise…`/`…LowNoise…` GGUF | same repo (`TS_WAN_DIT2` overrides; auto-found by name in the same/sibling folder) |
| UMT5-XXL text encoder | `umt5-xxl-encoder-Q8_0.gguf` | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf) (`--wan-te` / `TS_WAN_TE`) |
| Wan 2.1 video VAE (2.1 + A14B) | `wan_2.1_vae.safetensors` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors) (`--wan-vae` / `TS_WAN_VAE`) |
| Wan 2.2 video VAE (TI2V-5B) | `Wan2.2_VAE.safetensors` | bundled in [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE) |

The text encoder is released from VRAM before the denoise starts, the VAE
encoder (I2V) before the DiT loads, and the DiT before the VAE decode, so peak
VRAM is roughly `max(TE, DiT + attention, VAE)` — TI2V-5B Q8_0 generates
81-frame 480p image-to-video on a 16 GB GPU in under 8 minutes, and both A14B
Q4_K_M experts run sequentially on the same card.

## Backends

Video generation runs on every TensorSharp backend:

| `--backend` | Path | Notes |
|---|---|---|
| `ggml_cuda` | GGML whole-graph kernels | fastest; persistent DiT graph + CUDA-graph capture |
| `ggml_vulkan` | GGML whole-graph kernels | AMD/Intel/NVIDIA via Vulkan; the VAE stays on the banded conv path (Vulkan drivers reject multi-GB arenas) |
| `ggml_cpu` | GGML whole-graph kernels | CPU-only machines; slow but exact |
| `cuda` | direct CUDA (`WanDirect*`) | ggml-independent: quantized weights resident via TensorSharp's own MMQ/dp4a/cuBLAS routing, streaming online-softmax attention kernels, channels-last im2col VAE |
| `cpu` | direct pure-C# (`WanDirect*`) | no native dependencies beyond the managed runtime; parallel SIMD GEMM/attention kernels |

All backends produce numerically equivalent videos (verified against each other
and against diffusers' reference implementation on identical seeds/embeddings —
final-latent cosine ≥ 0.999 between backends, ≈ 0.994 vs the bf16 diffusers
reference with quantized checkpoints). The `cpu` / `ggml_cpu` backends are for
functional use — a 480p multi-second video takes tens of minutes on a
high-core-count machine.

Reference timings (RTX 2000 Ada 16 GB, Wan2.1-1.3B F16, 832×480, 33 frames,
30 UniPC steps — the official 480p recipe): `ggml_cuda` 12.0 s/step (445 s
total), `ggml_vulkan` 17.2 s/step (625 s), direct `cuda` 19.3 s/step (700 s).
On shorter sequences the direct backend matches `ggml_cuda` (TI2V-5B
480×480×9f: 1.1 s/step on both); at the 14k-token recipe its F32 attention
path trails ggml's F16 flash attention. Long-sequence attention routes through
chunked cuBLAS GEMM+softmax automatically (`TS_WAN_DIRECT_FUSED_ATTN=0` forces
it everywhere).

## CLI

Text-to-video (any family):

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "a cute fluffy orange cat walking through a sunny garden with flowers" \
  --output cat.mp4 --width 832 --height 480 --video-frames 49 --diffusion-seed 7
```

Image-to-video (Wan 2.2 models — pass `--image`):

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "the cat runs toward the camera, dynamic tracking shot, golden hour" \
  --image first_frame.png --output cat_run.mp4 --video-frames 81
```

```bash
# A14B: point --model at either expert; the sibling is found automatically
TensorSharp.Cli --model HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf \
  --backend ggml_cuda --prompt "the ship sails into the storm, waves crashing" \
  --image ship.jpg --output ship.mp4
```

- `--image <file>` — first-frame conditioning (PNG/JPEG/WebP/…). EXIF
  orientation is honored. When no explicit `--width/--height` is given, the
  output resolution follows the image's aspect ratio at the model's native
  area (TI2V-5B: 1280×704 = the official 720p recipe; others: 832×480).
- `--video-frames N` — frames, snapped to `4k+1`; `--video-frames 1` produces a
  still image (use `--output out.png`).
- `--sampler unipc|euler` — UniPC (default) is the official Wan sampler,
  verified numerically against diffusers' `UniPCMultistepScheduler`.
- `--cfg` / `--flow-shift` / `--diffusion-steps` default to each model's
  official recipe: TI2V-5B cfg 5.0, shift 5.0, 50 steps, 24 fps; A14B I2V
  cfg 3.5 (both experts), shift 5.0, 40 steps; A14B T2V cfg 4.0/3.0, shift
  12.0; Wan 2.1 cfg 6.0, shift 8.0 (1.3B video) or 3.0/5.0, 30 steps, 16 fps.
- `--negative-prompt` defaults to the official Wan negative prompt.
- `--cfg-cache-stride N` (default off) — approximate speedup. A guided step
  is `v = v_cond + (cfg-1)·d` with `d = v_cond - v_uncond`; the guidance
  direction `d` changes much more slowly across the schedule than `v` does, so
  the unconditional pass can run on one step in `N` and the cached `d` cover the
  rest. At 50 steps, `2` runs 77 of the 100 passes (1.30× faster) and `3` runs
  70 (1.43×). The first three steps and the last always recompute `d`. Leave it
  off when matching a reference sample matters. Server: `"cfgCacheStride": 2`.
- MP4 writing prefers `ffmpeg` on `PATH` (or `TS_FFMPEG=<path>`, or an
  `ffmpeg` folder next to the executable) — near-lossless CRF 17 H.264.
  Without it the OS codec via OpenCV is used at its default bitrate, which
  is visibly worse; the CLI warns when that happens.

### Getting good quality

- **Resolution is the biggest lever.** Wan is trained at 480p (832×480 area)
  and 720p (1280×704); TI2V-5B is natively a 720p model. Well below the 480p
  area the DiT is out of distribution and produces soft, unstable, color-drifting
  video no matter how many steps you spend — the pipeline warns below ~0.3 MP.
  Generate at a supported resolution and downscale afterwards instead of
  generating small (e.g. use `--width 480 --height 704` rather than 320×480).
- **Describe the shot.** Wan's prompt adherence rewards detailed prompts —
  subject, action, camera, lighting, style — over 3-4 word fragments; the
  official demos run prompts through an LLM "prompt extend" step first. Short
  prompts leave the model underconstrained and drift-prone.
- More `--diffusion-steps` beyond the recipe (50) buys little; prefer fixing
  resolution and prompt first.

## Server

```bash
TensorSharp.Server --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --video-frames 121 --fps 24
```

`--video-frames` and `--fps` set server-wide defaults for the Web UI and all
three video endpoints below. The Web UI does not send either field, so the
example produces 121 frames at 24 fps (about 5.04 seconds of playback). An API
request that supplies `frames` or `fps` overrides the corresponding startup
default independently; these are defaults, not caps. If the flags and request
fields are both omitted, the model recipes apply: 49 frames at 24 fps for
TI2V-5B, and 33 frames at 16 fps otherwise. Frame counts are snapped to `4k+1`.
Keep the model's native FPS and adjust the frame count when changing duration;
changing only FPS changes playback speed.

- **Web UI** (`http://localhost:5000`): type a prompt in the chat — attach an
  image to get image-to-video (the image becomes the first frame); the reply is
  the generated video with live denoising progress.
- **API**:

```bash
curl http://localhost:5000/v1/videos/generations -H "Content-Type: application/json" -d '{
  "prompt": "the cat runs toward the camera, cinematic",
  "image": "data:image/png;base64,....",
  "size": "832x480", "frames": 81, "steps": 50, "seed": 7
}'
# -> { "created": ..., "data": [ { "url": "/uploads/video-....mp4" } ], "frames": 81, ... }
```

  `image` accepts base64 (with or without a `data:` URL prefix); omit it for
  text-to-video. Also `POST /api/video-generate` (same JSON, `{ ok, url, ... }`
  envelope, plus `imagePath` referencing a previously uploaded file) and
  `POST /api/video-generate/stream` (SSE progress, used by the Web UI). A14B
  accepts `cfg2` (low-noise-expert guidance) on every endpoint.

## Implementation notes

The networks each run as ONE resident-weight ggml graph per invocation
(`TensorSharp.GGML.Native/ggml_ops_wan.cpp`):

- `TSGgml_WanT5Encode` — 24-layer UMT5 encoder with per-layer relative attention
  bias; tokens in, hidden states out (conditioning is zero-padded to 512 tokens,
  Wan semantics).
- `TSGgml_WanDitForward` — one denoising-step velocity prediction; persistent
  per shape on CUDA so ggml's CUDA-graph capture removes the launch overhead;
  flash attention over the 3D-RoPE self-attention. For TI2V-5B image-to-video
  the graph modulates two token segments at different timesteps (the
  conditioning frame's tokens at t=0, diffusers `expand_timesteps` semantics)
  — attention still runs over the full joint sequence.
- `TSGgml_WanVaeDecode` — the causal 3D video VAE decoder (both generations),
  iterating temporal chunks in-graph with the causal feature cache carried
  between chunks; convs run as banded im2col+GEMM under a scratch budget so
  peak VRAM stays bounded (`TS_WAN_VAE_GEMM_MAX_MB`; the default adapts to
  free device memory, floor 384 MB; `0` forces direct convolution), with the
  cross-chunk feature caches stored F16. The Wan 2.2 decoder adds the
  weight-free DupUp3D residual shortcuts and the final 2×2 pixel unpatchify.
  Above ~0.5 MP the decode is additionally tiled into full-width horizontal
  bands blended over an 8-latent-row overlap (the diffusers `enable_tiling`
  approach, ~59 dB vs the untiled decode): a 720p plane's activations plus
  causal caches are otherwise ~12 GB device-resident, which pushes 16 GB
  WDDM cards into shared-memory paging (33 min → ~3 min for 69-frame 720p).
  `TS_WAN_VAE_TILE=0` disables the tiling.
- `TSGgml_WanVaeEncode` — the causal 3D VAE encoder (both generations) behind
  image-to-video conditioning: 1+4k pixel-frame chunks, stride-2 spatial and
  temporal downsampling with the causal caches, AvgDown3D shortcuts and 2×2
  pixel patchify for Wan 2.2.

I2V conditioning follows the reference pipelines exactly: TI2V-5B replaces the
first latent frame with the VAE-encoded image every step and modulates its
tokens at timestep 0; A14B concatenates a 4-channel first-frame mask plus the
VAE-encoded `[image, zeros…]` clip behind the 16 noise channels (36-channel DiT
input) and switches from the high-noise to the low-noise expert at
`t < boundary·1000` (0.9 I2V / 0.875 T2V), releasing the first expert's VRAM at
the switch.

Numerics are verified against the reference implementations by guarded tests
(`InferenceWeb.Tests/WanVideoOracleTests.cs`, fixtures generated by
`InferenceWeb.Tests/tools/wan22_oracle.py`): the TI2V-5B DiT (Q8_0) matches
diffusers' `WanTransformer3DModel`
at cosine > 0.995 in both uniform and per-token-timestep modes, both VAE
encoders match `AutoencoderKLWan` at cosine > 0.999, the Wan 2.2 VAE decode at
> 35 dB PSNR (Wan 2.1 decode: 59.9 dB), and the tokenizer reproduces the
HuggingFace T5 ids exactly on English and CJK prompts.

Environment knobs: `TS_WAN_DIT_CAPTURE=0` (disable the persistent captured DiT
graph), `TS_WAN_DIT_FLASH=0` (materialized attention), `TS_WAN_DIT_KV_F16=0`
(F32 attention keys/values — the old, ~2x slower default), `TS_WAN_HEARTBEAT_S`
(progress tick interval, default 30; `0` silences the ticks),
`TS_WAN_VAE_GEMM_MAX_MB` (im2col budget),
`TS_WAN_VAE`/`TS_WAN_TE`/`TS_WAN_DIT2` (companion paths), `TS_FFMPEG` (ffmpeg
path for MP4 export), `TS_WAN_DIT_TRACE=<file>` (per-stage activation stats for
debugging).

## Performance

RTX 3080 Laptop 16 GB (Windows/WDDM), ggml_cuda:

| Model / workload | Text enc | Image enc | Denoise | VAE decode | Total |
|---|---|---|---|---|---|
| **Wan2.2-TI2V-5B Q8_0** I2V 832×480×81f, 30 steps | 3.5 s | 8.5 s | 11.5 s/step (CFG ×2, 8190 tokens) | 103 s | 464 s |
| Wan2.2-TI2V-5B Q8_0 T2V 640×384×25f, 20 steps | 3.7 s | — | 1.5 s/step | 23 s | 65 s |
| Wan2.2-I2V-A14B Q4_K_M 480×480×9f, 6 steps (smoke) | 3.4 s | 5 s | 7–8 s/step + 2 expert loads | 7.4 s | 89 s |
| Wan2.1-T2V-1.3B F16 832×480×33f, 30 steps | 3.1 s | — | 9.4 s/step (CFG ×2) | 51 s | 337 s |

The TI2V-5B model's 16×16 spatial compression gives it ~2.7× fewer DiT tokens
than Wan 2.1 at the same resolution — it is both the fastest and the
highest-quality option for consumer GPUs, and the only 720p-24fps one.

### Long sequences: cost is quadratic in the token count

Token count is `latent_frames × (h/2) × (w/2)`, and DiT self-attention costs
`O(tokens²)`. Past a few thousand tokens attention — not the weight matmuls —
is where the time goes, so the frame count and the frame area both matter far
more than the step count:

| Request (TI2V-5B) | Tokens | Attention work |
|---|---|---|
| 640×384×25f | 1 200 | 1× |
| 832×480×81f | 8 190 | 47× |
| 1088×832×121f (5 s at 24 fps, 720p class) | 27 404 | 520× |

The full 5-second 720p recipe is a genuinely large job: 50 steps × 2 CFG passes
= 100 DiT passes over 27 k tokens. The pipeline prints the token count, a
per-pass timing and a running ETA, and heartbeats every 30 s while a pass is in
flight, so a long run is visibly progressing rather than apparently hung:

```
Wan 2.2-TI2V I2V: 1088x832x121f (31x26x34 = 27404 tokens), 50 steps, unipc, cfg 5, shift 5, seed ...
  [wan] large request: 100 DiT passes over 27404 tokens. Self-attention costs O(tokens^2) ...
  step 1/50 t=999.0 (249.0s: cond 124.6s + uncond 124.4s) — ~3h 24m left
  [wan] …denoise step 2/50 (cond pass) 60s in this pass, 315s total, ~3h 24m left
```

To make such a request cheaper, in order of effect:

1. **Fewer frames.** 121 → 61 frames roughly quarters the attention work; the
   video is 2.5 s instead of 5 s at 24 fps.
2. **Smaller frame area**, but not below Wan's training resolutions — under
   ~0.3 MP the model is out of distribution and the result gets *worse*, not
   just cheaper. `--width 480 --height 704` is a good portrait target.
3. **Fewer steps.** 50 is the official TI2V recipe; 30 is visibly close and
   1.7× cheaper.
4. **`--cfg-cache-stride 2` or `3`** — 1.30× / 1.43× by reusing the guidance
   direction between steps (approximate; see the CLI section).

M5 Pro (20-core GPU, 48 GB unified), `ggml_metal`, Wan2.2-TI2V-5B Q8_0,
1088×832×121f = 27 404 tokens, image-to-video, the official 50-step recipe:

| Stage | Before | Now |
|---|---|---|
| text encode (UMT5-XXL, both prompts) | 1.9 s | 1.9 s |
| image encode (VAE) | 5.2 s | 5.2 s |
| denoise, per step (2 CFG passes) | 412.3 s | **249.0 s** |
| VAE decode, 121 frames | 863 s (3 bands × 24 latent rows) | **734 s** (2 × 30) |
| **50-step total** | ≈ 5 h 58 m | **≈ 3 h 40 m** (1.63×) |
| 50-step total, `--cfg-cache-stride 3` | — | ≈ 2 h 38 m (2.27×) |

Two changes account for that.

**F16 attention keys and values.** Every backend's flash-attention kernel is
built around an F16 KV cache: ggml-metal instantiates the F32 variant with
`simdgroup_float8x8` accumulators where the F16 one uses `simdgroup_half8x8`,
and the F32 tiles also cost twice the bandwidth in a kernel that re-streams K
and V once per 8-query threadgroup. The permute into flash-attention layout and
the narrowing happen in one `ggml_cpy`, so the F16 path also writes half the
bytes the old `ggml_cont` did. `TS_WAN_DIT_KV_F16=0` restores F32.

**VAE decode band layout.** Tiling picks the band *count* from the memory budget
and then splits the plane evenly, instead of walking a fixed band height and
letting the last band land wherever the stride puts it. At 52 latent rows that
was three 24-row bands — 72 rows decoded to produce 52 — and is now two 30-row
bands (60 rows, one seam instead of two) within the same per-band budget:
863 s → 734 s, the ratio the row counts predict.

#### Alternatives measured and rejected

One self-attention at this shape (seq 27 404, 24 heads, head dim 128), same run:

| KV type | time | vs F32 |
|---|---|---|
| F32 (before) | 4993 ms | 1.00× |
| **F16 (now)** | **2467 ms** | **2.02×** |
| Q8_0 | 2652 ms | 1.88× |
| Q4_0 | 2395 ms | 2.08× |

Q8_0 keys/values are *slower* than F16 — the dequantization outweighs the
bandwidth saved — and Q4_0 buys 3% for a real precision cost, so F16 is the
sweet spot rather than a compromise. Materialized chunked attention
(`mul_mat` + `soft_max_ext`, which unlike flash attention *can* use the Metal 4
tensor API) came in at 1.08×: at this size attention is bound by score traffic,
not by the GEMM rate. Dequantizing the DiT weights from Q8_0 to F16 moved the
matmuls by <5% (25.2 → 26.4 ms for an attention projection). The Metal 4 tensor
API stays off for TI2V-5B: it makes the weight matmuls 2.9× faster (~22 s per
pass) but forces the VAE onto the direct-conv path, which at 121 frames costs
hours — see `ApplyArchitectureNativeTunables`.

#### Quality

F16 keys and values change the DiT's output by less than the sampler's own
sensitivity to floating-point reassociation:

- One DiT forward against the diffusers `WanTransformer3DModel` reference
  (`WanVideoOracleTests`): cosine **0.999964** with F16 K/V and **0.999964**
  with F32 K/V — the per-pass accuracy is unchanged.
- Full 25-frame generations from an identical seed: F16 vs F32 K/V differ by
  39.08 dB mean PSNR. The control — F32 K/V with flash attention *off*, i.e. the
  same arithmetic in a different summation order — differs by 39.40 dB. The two
  are the same magnitude, so the pixel difference is an 8-step diffusion
  trajectory diverging from a rounding difference, not a loss of accuracy.

The speedup only appears where attention dominates: at 2 310 tokens
(480×704×25f) F16 and F32 K/V both run 5.0 s per pass.

Wan 2.1 vs stable-diffusion.cpp (master-769, identical GGUFs/settings,
33-frame 480p): sampling is near parity (281 s vs 258 s) but sd.cpp's VAE
decode materializes ≈8 GB of 3D im2col and oversubscribes a 16 GB card into
WDDM paging (1762 s vs **51 s**) — TensorSharp is 6.0× faster end to end.
