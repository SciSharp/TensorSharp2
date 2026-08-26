# MiniMax-H3

MiniMax-H3 generates video **and native stereo audio together**, up to 15 s at
24 fps with 32 kHz stereo. It is not a video model with audio bolted on: one
diffusion transformer denoises a packed video+audio latent in a single token
sequence, so the soundtrack is part of the model output rather than something
added afterwards.

TensorSharp runs it as **seven** native whole-network ggml graphs — text encode,
vision encode, the DiT forward, and encode + decode for each of the two VAEs —
with weights bound resident straight from the GGUF/safetensors mmap. The public
entry point is `MiniMaxH3Model.GenerateVideo(prompt, VideoGenerationParams)`,
behind the shared `IVideoGenerationModel` seam, so the CLI and the server drive
H3 and Wan down one path instead of type-testing each concrete model.

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

That writes `fox.mp4` plus `fox.wav` with the generated soundtrack.

## Status

| Component | State |
|---|---|
| Text-to-video (`t2v`) | working |
| Image-to-video (`i2v`) — animate a photo | working |
| First/last frame (`fl2v`) | working |
| Reference-to-video (`ref`) — images, clips and soundtracks | working |
| Joint stereo audio | working |
| Video VAE encode + decode | working, tiled |
| Clips past 22 frames | working (the VAE decodes 5 latent frames at a time) |

### Long clips and the FP16 attention ceiling

H3 attends bidirectionally over ONE packed sequence, so its key count is the whole
clip rather than a window: a 640x384 request is 2364 packed tokens at 22 frames but
8646 at 107. ggml's flash-attention kernels keep the softmax numerator in FP16 and
give it three bits of headroom, so that key count is a numeric budget, and H3 is the
model that spends it - `v_norm` does not exist in the checkpoint, so the value stream
carries the same unbounded magnitudes that `h3_mm` guards against one op later. Left
alone, a 107-frame clip overflowed to NaN on the first denoise step and came back
with every pixel black and every audio sample clamped, while 73 frames of the same
request rendered correctly.

`h3_attend` now pre-scales V by a power of two derived from the key count and undoes
it on the output - exact, because attention is linear in V - so the accumulator sees
a bounded number of effective keys at any clip length. The sampler also refuses to
write a diverged clip: a non-finite velocity fails the request naming the step it
appeared at, rather than saving a black file. Set `TS_H3_TRACE=1` to print the latent
and velocity magnitudes for every step.

## Performance

M5 Pro / Metal, 22 frames, 8 steps, identical seed, versus
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) at its
best-performing configuration:

| Resolution | stable-diffusion.cpp | TensorSharp | |
|---|---|---|---|
| 256×256 | 49.3 s | **20.9 s** | 2.4× faster |
| 640×384 | 108.5 s | **63.1 s** | 1.7× faster |

## Files

Four networks cooperate, so a run needs four files. The two denoisers are
**separate checkpoints, not settings** — which one you load decides what
conditioning it accepts.

| File | Size | Source |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) — text + keyframes |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB | same repo — text + references |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB | same repo — the text encoder, shared (`-Q2_K_M.gguf`, 12.20 GiB, pairs with a smaller denoiser) |
| `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (also mirrored under `vae/` in the unsloth repo) |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | same repo — omit for silent video |

About 33.5 GB for one set. Each network is loaded and **released** in turn, so
peak VRAM is `max(...)`, not the sum; adding the second denoiser later costs only
its own ~10.6 GiB, because the encoder and both VAEs are shared.

Companions resolve automatically next to the denoiser, or with `--video-vae`,
`--video-text-encoder`, `--audio-vae`.

Two shipped configs download the lot on first run and only differ in their `model`
entry, so the encoder and the VAEs are fetched once:

```sh
TensorSharp.Server --config config/minimax-h3-fl2va.json      # keyframes
TensorSharp.Cli    --config config/minimax-h3-ref2va.json \
  --ref-image person.png --prompt "…" --output out.mp4        # references
```

Both name `"backend": "ggml_cuda"` (a `--backend` on the command line wins) and set
width 640, height 384, 22 frames, 24 fps. Neither sets steps or guidance: a shipped
config has to parse on both hosts, and the two spell steps differently
(`--video-steps N` on the server, `--diffusion-steps N` on the CLI), while the
server takes no `--cfg` at all — so the model's own defaults apply.

> **The text-encoder GGUF carries no tokenizer**, and that is the one thing a config
> cannot fetch for you: auto-download fills in options that are **flags**, and the
> tokenizer is not one. Put `vocab.json` and `merges.txt`
> from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor)
> beside it, or point `TS_VIDEO_TOKENIZER` at them. Neither published GGUF has any
> metadata at all, so TensorSharp identifies H3 by its tensors rather than by an
> architecture string (arch keys `minimax-h3` / `minimax_h3` are accepted when a
> file does declare one).

> **`--cfg 1.0` is required.** H3 is CFG-distilled; above 1.0 it degrades badly and
> TensorSharp refuses. 4–8 steps is the operating point, so iteration is fast.

## Conditioning modes

An image can mean two completely different things to H3, and they use different
checkpoints. `--video-mode` makes the choice explicit; without it the mode is
inferred from what you pass.

| What you want | Mode | Checkpoint |
|---|---|---|
| "animate this photo" | `i2v` | FL2VA |
| "go from photo A to photo B" | `fl2v` | FL2VA |
| "use this person, brand-new scene" | `ref` | Ref2VA |
| "reference this product, new angle and background" | `ref` | Ref2VA |
| text only | `t2v` | either |

Which partition is active is read off the **file name** — `ref2va` anywhere in it,
case-insensitively — so keep that substring if you rename or requantize. Asking one
checkpoint for the other's mode does not silently drop the input: the request fails
with a message naming the file to load instead. Keyframes and named references in
the same request are refused outright, on either checkpoint.

### Text to video

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

### Image to video — animate a photo

The image becomes the **first frame** and the prompt drives what happens next.

A keyframe is conditioned **twice**, and both halves matter: the VAE-encoded latent
pins the frame it is anchored to, and the same picture goes through the Qwen3-VL
vision tower into the prompt, where it describes the scene for the whole clip. The
latent alone fades — the clip starts on the image and wanders off within a second
or two, which is invisible at 22 frames and obvious at 124.

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image portrait.jpg \
  --prompt "the person turns toward the camera and smiles, subtle handheld motion" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output animated.mp4
```

`--video-mode i2v` states it explicitly. The image is resized to the generation
canvas, so its aspect ratio should match `--width`/`--height` or it will be
stretched.

### First and last frame

Both ends are pinned and the model fills in the motion between them.

```sh
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image start.png --end-image end.png \
  --prompt "a slow cinematic push-in" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output morph.mp4
```

`--end-image` alone also works: the clip ends on that frame.

### Reference to video — keep the subject, change everything else

Ref2VA treats an image as an identity and appearance **reference**, not as a frame.
The first frame need not resemble it at all: the person, product or object carries
over while the camera, background and composition come entirely from the prompt.

This needs the **Ref2VA** checkpoint — `i2v`/`fl2v` and `ref` are separate files,
not settings.

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg \
  --prompt "the same woman sits at a table in a sunlit cafe by a window, drinking coffee, wide shot" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output cafe.mp4
```

Pass `--ref-image` more than once for several references (up to nine), for example
a person and the product they are holding:

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg --ref-image bottle.png \
  --prompt "she holds the bottle up to the light on a rooftop at golden hour, slow orbit" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output rooftop.mp4
```

Notes:

* References are only ever scaled **down**, to the generated clip's area, and keep
  their own aspect ratio. Nothing is stretched to the output canvas, and the output
  canvas is *not* taken from the reference — say what you want with
  `--width`/`--height`.
* **Each reference is extra tokens in the same packed sequence**, so the cost is
  linear and easy to predict: a 640x384 reference is 240 tokens, one at 576x448 is
  252. Measured on an RTX 3080 Laptop against a 640x384 22-frame clip, a denoise step
  goes from 4.37 s with no references to 9.38 s with eight — about 626 ms per
  reference per step, flat from one to eight.
* **Past about four references the Qwen3-VL pass dominates, not the denoiser.** Every
  reference adds roughly 250 vision placeholder tokens to the prompt, and that prompt
  is prefilled through all 50 layers: two references make a 548-token prompt and eight
  make 2086, so text conditioning grows from ~65 s to ~447 s while the whole 8-step
  denoise costs ~75 s. Reach for fewer, better references before reaching for fewer
  steps.
* TensorSharp caps reference images at nine. The reference implementation has no cap;
  this one exists because the packed sequence is attended over unmasked and its length
  is a numeric budget as well as a time one.
* References are **labelled in the order you pass them** before the language model
  sees them — `<Picture 1>`, `<Picture 2>` … for stills, `<Video 1>` … for clips,
  `<Audio 1>` … for soundtracks — so a prompt can name one: "the jacket from
  `<Picture 2>`". A clip that arrived with its own soundtrack is presented as
  *two* items and takes an `<Audio n>` label as well as a `<Video n>` one.
* Say who is in frame. The reference supplies identity; the prompt still has to
  describe the shot, or you get a well-rendered scene with the subject missing.
* A reference can also be a **clip** or a **soundtrack**, not just a still — see
  below.

> **A plain `--image` on the Ref2VA checkpoint is treated as a reference.** Clients
> that only know how to attach "an image" — the Web UI among them — therefore work
> against Ref2VA without any extra field. Pass `--video-mode ref` to be explicit.

### Reference clips and soundtracks

`--ref-video` takes a video **file** or a directory of frames, and `--ref-audio` a
WAV/MP3/Ogg. A clip's own soundtrack goes in separately with `--ref-video-audio`,
paired by position with `--ref-video` — a container's audio track is not readable
through the frame decoder, so the two arrive as separate inputs.

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-video walk.mp4 --ref-video-audio walk.wav \
  --prompt "the same woman walks along a beach at sunset, wide shot, waves behind her" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output beach.mp4
```

A standalone soundtrack, with no picture attached, is a reference too:

```sh
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image singer.jpg --ref-audio song.wav \
  --prompt "she performs on a small club stage under a single spotlight" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output stage.mp4
```

What happens to a reference clip:

* It is resampled onto H3's own **24 fps** and pulled down to the 17k+5 frame grid,
  capped at the number of frames being generated. A clip shorter than 5 frames at
  24 fps is rejected.
* Its canvas comes from its own aspect ratio around a nominal 768 px, capped by
  area and snapped to 32. A source **smaller** than that keeps its own size:
  upscaling invents no detail to reference and costs a great deal — a 448x320 clip
  blown up to 1088x768 would be 5712 conditioning tokens against 1680 for the clip
  being generated.
* The language model sees it at **2 fps**, two frames at a time, each pair labelled
  with the time it sits at. The denoiser sees all of it, as latents.
* A soundtrack is resampled to the audio VAE's 32 kHz stereo and truncated to the
  generated clip's duration.

> **A reference clip is the most expensive input H3 takes.** A 22-frame 448x320
> reference adds 980 conditioning tokens on top of the 1680 the output itself
> needs, and the VAE has to encode all 22 frames first. Measured on an M5 Pro:
> 14 s to encode the reference, then about 9 s per denoising step against 5 s
> without it.

## Getting good quality

Two settings dominate; everything else is a rounding error next to them.

| Lever | Default | What it does |
|---|---|---|
| `--width` / `--height` | 640×384, or the conditioning image's aspect at that area | **The single biggest factor.** Faces need pixels: at 256×256 a face is a few dozen pixels across and comes back blurry and malformed no matter what else you set. |
| `--diffusion-steps` | 20 | Removes chromatic fringing around moving subjects. 8 is the fast lane, ~20 is clean, past ~30 gains little. Time scales roughly linearly. |
| `--diffusion-seed` | random | Some seeds simply compose better. Cheapest thing to retry. |
| `--flow-shift` | 12 | Moves where the schedule spends its steps. Rarely needs touching. |
| model quant | — | `-Q8_0` over `-Q4_K` for the denoiser, and `qwen3vl_32b_minimax_h3-Q4_K_M` over `-Q2_K_M`, both help if you have the memory. |

### Frame the subject — this dominates everything else

**How much of the frame your subject occupies matters more than every setting
below combined.** The model renders what it is given at the resolution it is
given; if a face is 40 pixels across in the conditioning image, no number of
steps and no output size will recover it.

Measured on one photo of a park pass with two small printed portraits, same
prompt, same seed, 20 steps:

| Input | Faces in the output |
|---|---|
| the photo as shot, 640x384 | ~40 px across — mushy, unrecognizable |
| cropped to the two figures, 768x480 | ~250 px across — hair, skin texture, lapel pin all legible |

Nothing changed but the crop. If the thing you care about is a face, crop the
source photo to it before generating.

> **A conditioning image is never stretched.** When the canvas aspect differs
> from the image's, the image is centre-cropped to fit and the run says so. A 4:3
> photo forced into 640x384 (5:3) would otherwise be squeezed 25% horizontally,
> and since the keyframe *is* the first frame, that distortion is baked into
> every frame after it. Leave `--width`/`--height` unset to take the canvas from
> the image instead, and crop the source yourself when you want to control what
> is in shot.

`--cfg` is **not** a quality lever here — H3 is CFG-distilled and only accepts 1.0.
`--negative-prompt` does nothing for the same reason: there is no unconditional
pass to steer away from, and `--cfg-cache-stride`, which caches that pass, has
nothing to cache. `--sampler` is a Wan-family knob; H3 runs its own flow-match
schedule and ignores it.

Measured on the same image-to-video request (M5 Pro, `ggml_metal`, seed 42, 22 frames):

| Setting | Time | Result |
|---|---|---|
| 256×256, 8 steps | 26 s | blurry, faces unrecognizable |
| 640×384, 8 steps | 79 s | sharp, but coloured fringing around moving hands |
| 640×384, 24 steps | 212 s | clean |
| 576×448 (image aspect), 20 steps | 187 s | best — nothing stretched |

> **On the server, size is a startup flag.** The Web UI sends only the prompt and
> the image, so requests inherit the server's defaults. Start it with
> `--video-width 640 --video-height 384 --video-steps 20 --video-frames 22` (or omit
> width and height and let each request pick the aspect from its image; give only
> one of the two and H3 takes the other from the conditioning image). `--width` /
> `--height` are accepted as aliases, and `--video-mode` pins the conditioning mode
> for a deployment that only offers one. The server has **no `--cfg`** — there is
> nothing to set, since H3 enforces 1.0 itself.

> **Forcing a size that does not match your image stretches it.** A 4:3 photo forced
> into 640×384 is squeezed ~25% horizontally, which is exactly the kind of distortion
> that makes faces look wrong. Leave width/height unset for image-to-video and the
> aspect is taken from the image.

> **The soundtrack changed in this revision.** The denoiser emits a whitened audio
> latent and the decoder wants the VAE's own scale; the un-whitening step was
> missing, so every generated track was decoded from a latent about 1.9x too small.
> It was audible only as a wrong level and a slightly wrong timbre, never as
> silence or noise. Verified against the reference implementation's own decode of
> the same latent: cosine 0.99998 at 1.0000x gain, from 0.81x before.

## Sizes and lengths

* Width and height are rounded **up** to a multiple of 32. Default 640×384, or
  that area at the conditioning image's aspect ratio.
* Frame count is rounded up onto the **17k+5** grid — 5, 22, 39, 56, 73, 90, 107,
  124 … — which comes from the video VAE's temporal chunking, not from an
  arbitrary choice: each 5-latent-frame chunk yields 17 pixel frames, on top of a
  5-frame lead-in. Default 22.
* fps is pinned to 24; any other value is overridden.
* Clips of any grid length decode correctly: the VAE runs 5 latent frames at a
  time with a 2-frame look-ahead and cross-fades the seams, exactly as the
  reference does. Decoding a long clip in one call instead washes detail out
  progressively — measured against the conditioning photo, frame 0 fell from 0.97
  correlation at 22 frames to 0.86 at 90 — so the chunking is a correctness
  requirement, not an optimization.

## HTTP API

Three routes share one parser: `POST /api/video-generate`,
`POST /api/video-generate/stream` (same body, SSE progress ticks) and the
OpenAI-shaped `POST /v1/videos/generations`.

```sh
curl -s localhost:5001/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "a red fox trotting through falling snow, cinematic",
  "width": 640, "height": 384, "frames": 22, "steps": 8, "cfg": 1.0,
  "imagePath": "card.jpeg", "videoMode": "i2v"
}'
```

Reference conditioning uses the same route against the Ref2VA checkpoint:

```sh
curl -s localhost:5001/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "the same woman sits at a table in a sunlit cafe by a window, wide shot",
  "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
  "referenceImages": ["person.jpg", "bottle.png"], "videoMode": "ref"
}'
```

Returns `{ ok, url, audioUrl, width, height, frames, fps, seed, codec, elapsedSeconds }`.
`audioUrl` is null when the model produced no track. The full field set is `prompt`,
`width`, `height`, `frames`, `steps`, `cfg`, `fps`, `imagePath`, `videoMode`,
`generateAudio`, `endImage`, `referenceImages`, `referenceVideos`,
`referenceAudios`, `referenceVideoAudios`. Most of them are **camelCase only** —
`videoMode`, `generateAudio`, `endImage` and the four `reference*` lists are the
only ones that also accept a snake_case spelling (`video_mode`, `generate_audio`,
`end_image`, `reference_images`, …), and camelCase wins when both are present.
Sending `image_path` or `flow_shift` is silently ignored rather than rejected, so
prefer camelCase everywhere. `referenceVideoAudios` pairs **by index** with
`referenceVideos`, the same positional rule `--ref-video-audio` follows. `imagePath` / `endImage` /
`referenceImages` name files previously uploaded through `/api/upload`; paths
outside the upload directory are rejected. `/v1/videos/generations` shares the same
parser, so the same seven fields accept snake_case there too (`video_mode`,
`reference_images`, …); it returns `audio_url` rather than `audioUrl`.

A model rejection — the wrong checkpoint for the mode, or keyframes together with
references — comes back as a **400 carrying the model's own message**, not a generic
500, so the file to load instead is in the response body.

`GET /api/models` answers with a `video` object (null for every non-video model)
carrying `family` (`minimax-h3`), `supportsAudio`, `supportsImageConditioning`,
`supportsEndImageConditioning`, `supportsReferenceConditioning` and
`maxReferenceImages`. That is how the Web UI decides whether to offer a first frame,
a last frame or up to nine references, instead of pattern-matching an architecture
string.

Audio is written as a sidecar WAV rather than muxed into the MP4, because muxing
needs an encoder that may not be installed. To combine them:

```sh
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

`--no-audio` (`"generateAudio": false`) skips the audio decode entirely, saving the
audio VAE's time and memory; video-only models ignore it.

## Architecture

Recorded here because none of it is inferable from the checkpoints, which carry no
metadata. Cross-checked against stable-diffusion.cpp, the upstream reference
PyTorch, and the GGUF tensor tables.

### Diffusion transformer

50 blocks, ~19.3 B parameters, single-stream: text, conditioning frames, target
audio and target video are ONE sequence under full bidirectional attention. There
is **no cross-attention**.

| | |
|---|---|
| hidden | 5376 |
| attention | 56 heads × 128 = **7168 inner**, wider than the 5376 residual stream |
| MLP | SwiGLU, gate first |
| patch | (t, h, w) = (1, 2, 2); video 24 ch → 96-element patch |
| audio | 32 ch, one token per latent frame per stereo channel |

Within a token the 96 video values are **channel-major, patch-minor**
(`c*4 + ky*2 + kx`). The transposed order has identical statistics and silently
scrambles every token.

**AdaLN** uses a learned curve table, not a timestep MLP: `adaln_t_table` is
`[8, 1025]`, interpolated at the timestep, and one `Linear(8 → 96768)` per block
emits 6 modulation vectors for each of 3 modalities (0 = visual, 1 = text,
2 = audio). A token run picks `timestep*3 + modality`.

**RoPE** is 3-axis with a learned 16-entry `inv_freq`, so 96 of 128 head dims
rotate. Positions are **continuous floats**: time is measured in audio-latent units
(1/40 s) so both streams share one timeline, and the spatial axes are normalized by
the frame's geometric mean extent.

### The dual video/audio shift

Video wants timestep shift 12, audio wants 3. Rather than run two samplers, the
sampler stays on the video schedule and the model converts: the audio stream is
conditioned on the sigma a shift-3 schedule would have had at the same underlying
time, and its velocity is pre-multiplied by `d(sigma_a)/d(sigma_v)` so the shared
Euler step integrates it along its own trajectory. Both schedules start at 1 and
end at 0, so the streams land together. `--flow-shift` moves the video stream only.

### Text encoder

Qwen3-VL-32B truncated to 50 language layers **with the final norm removed** — the
DiT consumes the raw layer-50 hidden state, whose massive-activation outliers
(absmax ~15 000) are load-bearing rather than a bug. **There is no chat template**:
the prompt is the raw text. Applying one shifts every hidden state.

### Video VAE

16× spatial, 4× temporal. The decoder is a **pure 36-layer transformer** — no
deconvolutions. Each latent voxel is one token, and a final `Linear(2048 → 3072)`
plus depth-to-space does the whole upsample in one step.

Decoding is **tiled at 256 px, and that is a correctness requirement, not an
optimization**: the decoder's RoPE coordinates are length-normalized over whatever
extent it is handed, and it only works over the extent it was trained on. Decoding
a 640×384 frame in one call produces visibly sheared output even though every
individual tile is exact.

The encoder is a causal 3-D CNN, but for the single frame image conditioning needs,
the causal padding is two leading *zero* frames — so only the last temporal slice
of each kernel contributes and the network reduces **exactly** to 2-D convolutions.

### Audio VAE

DAC/BigVGAN, 32 latent channels → stereo 32 kHz. Upsample rates {5,5,2,2,2,2,2}
multiply to 800, and 32000/800 = the 40 Hz latent rate. Uses alias-free snake
activations: every nonlinearity is wrapped in a 2× upsample / activate / 2×
downsample sandwich so it cannot fold high frequencies back into the band.

Unlike the video latent, the audio latent is decoded **as-is** — `latents_mean` /
`latents_std` are *not* applied. Applying them costs about 15× amplitude and yields
a track that is spectrally plausible but inaudible.

## Verification

Every network is checked against the reference implementation, not just against
itself. Fixtures come from the upstream PyTorch and from stable-diffusion.cpp's own
intermediate tensors.

| Component | Result |
|---|---|
| Text encoder, all 50 layers | cos 0.999999 vs sd.cpp's conditioning |
| DiT step, all 50 blocks | video cos 0.9983, audio cos 0.9998 |
| Video VAE decode | cos 1.000000 |
| Video VAE encode | cos 1.000000 |
| Audio VAE decode | cos 0.999995 |

Regenerate the fixtures with:

```sh
python InferenceWeb.Tests/tools/minimax_h3_oracle.py --ref-dir <reference py> \
    --weights-dir ~/work/models/minimax-h3 --out-dir ~/work/models/minimax-h3/fixtures
python InferenceWeb.Tests/tools/minimax_h3_dit_oracle.py --gguf <denoiser> --out-dir <fixtures>
python InferenceWeb.Tests/tools/minimax_h3_te_oracle.py --gguf <text encoder> --out-dir <fixtures>
```

Then run the gated tests:

```sh
export TS_MINIMAX_H3_DIR=~/work/models/minimax-h3
export TS_TEST_GGML_BACKEND=metal
dotnet test InferenceWeb.Tests -c Release --filter "FullyQualifiedName~MiniMaxH3"
```
