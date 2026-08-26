# Configuration files

Both `TensorSharp.Cli` and `TensorSharp.Server` can read their startup options
from a JSON file passed with `--config`, in addition to the command line:

```bash
TensorSharp.Server --config config/server-basic.json
TensorSharp.Cli    --config config/cli-basic.json
```

**Command-line options always win.** File values are applied first, then anything
you also pass on the command line overrides them — so one config file can be
reused across machines while you override just what differs:

```bash
# Use the file, but force the CPU backend for this run:
TensorSharp.Server --config config/server-basic.json --backend ggml_cpu
```

You can pass `--config` more than once; later files override earlier ones (and
the command line still wins over all of them).

## File format

The file is a JSON object whose keys are the same long option names each host
already accepts (listed in `--help`), with or without the leading `--`. Comments
(`//`, `/* */`) and trailing commas are allowed.

| JSON value          | Becomes                          | Example |
|---------------------|----------------------------------|---------|
| string / number     | `--key value`                    | `"max-tokens": 4096` → `--max-tokens 4096` |
| `true`              | the bare switch `--key`          | `"continuous-batching": true` → `--continuous-batching` |
| `false` / `null`    | nothing (omit the option)        | to turn something off, use its negation key, e.g. `"no-continuous-batching": true` |
| array               | a repeated flag                  | `"stop": ["</s>", "<\|eot\|>"]` → `--stop </s> --stop <\|eot\|>` |
| object              | a downloadable file (see below)  | `{ "path": "...", "urls": ["..."] }` |

## Variables

Define shared values once under `"variables"` and reference them with `${name}`
in any string value. A `${name}` that is not defined there falls back to an
environment variable of the same name, and variables may reference other
variables. `${name:-fallback}` supplies a value for when the name is defined
nowhere. See [`variables.json`](variables.json).

```json
{
  "variables": { "modelRoot": "${TENSORSHARP_MODELS:-../models}" },
  "model": "${modelRoot}/gemma-4-E4B-it-Q8_0.gguf",
  "mmproj": "${modelRoot}/mmproj-gemma-4-E4B-it-Q8_0.gguf"
}
```

### Where the models go (and why no config here has a drive letter)

Every config in this folder resolves its model root the same way:

```json
"modelRoot": "${TENSORSHARP_MODELS:-../models}"
```

- Set **`TENSORSHARP_MODELS`** to keep your models wherever you like — a Windows
  path, `/mnt/data/models`, `~/models`. One variable covers every config here.
- Leave it unset and the models land in a `models/` folder next to the repository
  (a relative `path` is resolved against the **config file**, not the working
  directory).

This is not a style preference. A hard-coded `"C:/models/x.gguf"` is **not an
absolute path on Linux or macOS** — .NET's `Path.IsPathRooted` knows nothing about
drive letters there — so it would be treated as *relative* and silently glued onto
the config's own directory. Written the way above, one file works unmodified on
Windows, Linux and macOS.

You can declare **as many variables (and root paths) as you need** — models that
live in different folders each get their own root:

```json
{
  "variables": {
    "ditRoot": "${TENSORSHARP_MODELS:-../models}/qwen-image-edit",
    "companionRoot": "${TENSORSHARP_MODELS:-../models}"
  },
  "model": { "path": "${ditRoot}/qwen-image-edit-2511-Q4_K_M.gguf", "urls": ["..."] },
  "qwen-image-vae": "${companionRoot}/Qwen_Image-VAE.safetensors"
}
```

## Auto-download

Any file option can be an object with a local `path` and one or more `urls`
instead of a plain string. On startup:

- if `path` already exists, it is used as-is (no network access);
- otherwise the file is downloaded from the first working URL, saved to `path`,
  and reused on every later run.

List several mirrors under `urls` for automatic fallback — if the first URL
fails, the next is tried. A relative `path` resolves next to the config file. An
optional `sha256` (lowercase hex) verifies a freshly downloaded file. Download
progress is printed to standard error so you can see what is being fetched and
how far along it is. See [`auto-download.json`](auto-download.json).

```json
{
  "model": {
    "path": "C:/models/Qwen3.5-9B-Q8_0.gguf",
    "urls": [
      "https://primary-mirror.example.com/Qwen3.5-9B-Q8_0.gguf",
      "https://backup-mirror.example.com/Qwen3.5-9B-Q8_0.gguf"
    ],
    "sha256": "0123...optional..."
  }
}
```

`url` (singular) is accepted as a shorthand for a single-entry `urls`.

## Examples in this folder

Every example uses **real, public, ungated** GGUF URLs, so each one works on a
fresh machine: the files auto-download to their local `path` on the first run and
are reused afterward. If you already have a file at that `path`, it is used as-is
(no download).

| File | Model(s) | Shows |
|------|----------|-------|
| [`cli-basic.json`](cli-basic.json) | Qwen3.5-9B (~8.9 GB) | Minimal CLI config, auto-download |
| [`server-basic.json`](server-basic.json) | Gemma-4 E4B model + vision projector | Multimodal server, sampling defaults, auto-download |
| [`variables.json`](variables.json) | Gemma-4 26B-A4B: model + mmproj + MTP draft | One shared root/repo reused across three related files |
| [`auto-download.json`](auto-download.json) | Qwen3-0.6B (~640 MB) | Small, fast download demo with mirror fallback |
| [`qwen-image-edit.json`](qwen-image-edit.json) | Qwen-Image-Edit 2511: DiT + VAE + text encoder + mmproj + Lightning LoRA | Multi-file image pipeline, all auto-downloaded |
| [`minimax-h3-fl2va.json`](minimax-h3-fl2va.json) | MiniMax-H3 FL2VA: DiT + Qwen3-VL-32B + video VAE + audio VAE (~33.5 GB) | **Video and 32 kHz stereo audio in one packed latent**; text-to-video, image-to-video, first/last frame |
| [`minimax-h3-ref2va.json`](minimax-h3-ref2va.json) | MiniMax-H3 Ref2VA: DiT + Qwen3-VL-32B + video VAE + audio VAE (~33.4 GB) | The same four networks, reference checkpoint: up to nine stills, clips and soundtracks |
| [`wan-video-ti2v-5b-turbo.json`](wan-video-ti2v-5b-turbo.json) | Wan 2.2 TI2V-5B Turbo: DiT + video VAE + UMT5 (~9.5 GB) | Video only, 4-step distilled, text- **and** image-to-video |
| [`wan-video-ti2v-5b.json`](wan-video-ti2v-5b.json) | Wan 2.2 TI2V-5B: DiT + video VAE + UMT5 (~11.4 GB) | Video only, undistilled 50-step reference recipe |
| [`wan-video-i2v-a14b.json`](wan-video-i2v-a14b.json) | Wan 2.2 I2V-A14B: **two** expert DiTs + Wan 2.1 VAE + UMT5 (~24 GB) | Image-to-video, two-expert schedule |

`server-basic.json` uses the standard `gemma-4-E4B-it` build — point its `path` at
your own file to host a different variant.

## Ready-made configs, one per runnable model

One config per runnable model, with its companions (vision projector, image-edit
VAE / text encoder, MTP draft head, LoRA) already wired in. Each points at the
existing local file, so no download happens — just run it. Every file works with
**both** hosts (only host-recognized keys are used):

```bash
TensorSharp.Cli    --config config/qwen3.5-9b-q8.json --input prompt.txt
TensorSharp.Server --config config/gemma-4-26b-a4b.json
```

| File | Model | Kind |
|------|-------|------|
| [`qwen3.5-9b-q8.json`](qwen3.5-9b-q8.json) | Qwen3.5-9B (Q8_0) | Text LLM |
| [`qwen3.5-9b-iq4_xs.json`](qwen3.5-9b-iq4_xs.json) | Qwen3.5-9B (IQ4_XS) | Text LLM, smaller quant |
| [`qwen3.5-9b-uncensored-q8.json`](qwen3.5-9b-uncensored-q8.json) | Qwen3.5-9B Uncensored (Q8_0) | Text LLM |
| [`qwen3.6-27b.json`](qwen3.6-27b.json) | Qwen3.6-27B + vision | Multimodal LLM |
| [`qwen3.6-35b-a3b.json`](qwen3.6-35b-a3b.json) | Qwen3.6-35B-A3B (MoE) + vision | Multimodal MoE LLM |
| [`gemma-4-e4b.json`](gemma-4-e4b.json) | Gemma-4 E4B + vision + MTP draft | Multimodal LLM |
| [`gemma-4-12b.json`](gemma-4-12b.json) | Gemma-4 12B (QAT) + vision + MTP draft | Multimodal LLM |
| [`gemma-4-26b-a4b.json`](gemma-4-26b-a4b.json) | Gemma-4 26B-A4B (MoE) + vision + MTP draft | Multimodal MoE LLM |
| [`gpt-oss-20b.json`](gpt-oss-20b.json) | gpt-oss-20b (Q8_0) | Text reasoning LLM |
| [`diffusiongemma-26b-a4b-q4.json`](diffusiongemma-26b-a4b-q4.json) | DiffusionGemma 26B-A4B (Q4_K_M) | Text diffusion (CLI) |
| [`diffusiongemma-26b-a4b-q3.json`](diffusiongemma-26b-a4b-q3.json) | DiffusionGemma 26B-A4B (Q3_K_M) | Text diffusion (CLI), smaller |
| [`qwen-image-edit-2511.json`](qwen-image-edit-2511.json) | Qwen-Image-Edit 2511 + VAE/TE/mmproj + Lightning LoRA | Image edit |
| [`qwen-image-rapid-nsfw.json`](qwen-image-rapid-nsfw.json) | Qwen-Rapid v9.0 DiT + VAE/TE/mmproj | Image edit (few-step) |

Notes:

- **Multimodal** configs load a vision projector, so add `--image photo.png` to ask
  about a picture.
- **Speculative decoding** is lossless and works on **both** hosts — a key here
  becomes the matching flag, and `TensorSharp.Cli` honours every one of them. The
  current spellings are `"spec": true` and `"spec-type"`; `"mtp-spec"` and
  `"mtp-draft-model"` are accepted aliases, which is why the shipped Gemma configs
  still use them. Gemma's draft head is a separate GGUF, named by
  `"spec-draft-model"`; Qwen3.6 and GLM 5.2 embed theirs in the trunk, so
  `"spec": true` is all they need. `"spec-type": "ngram"` needs no drafter at all,
  so it works with any config in this folder.
- **Image-edit** configs run the DiT pipeline: `--image in.png --prompt "…" --output
  out.png`. Per-edit `--diffusion-steps` / `--cfg` / `--diffusion-seed` are CLI flags.
- **DiffusionGemma** uses the CLI's iterative denoising path; tune it with
  `--diffusion-steps` / `--diffusion-seed` on the command line.
- To make any of these auto-download on another machine, turn a `"model": "…path…"`
  string into an object: `{ "path": "…", "urls": ["https://…"] }` (see the examples
  above).

## Video generation with sound (MiniMax-H3)

MiniMax-H3 denoises video **and 32 kHz stereo audio in one packed latent**, so a run
writes an `.mp4` and a matching `.wav`. Four networks cooperate — denoiser, text
encoder, video VAE, audio VAE — which makes a config file the practical way to run it:

```bash
TensorSharp.Server --config config/minimax-h3-fl2va.json

TensorSharp.Cli --config config/minimax-h3-fl2va.json \
  --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4

# Image-to-video, and first-and-last-frame:
TensorSharp.Cli --config config/minimax-h3-fl2va.json \
  --image start.png --end-image end.png \
  --prompt "a slow cinematic push-in" --output morph.mp4

# References, on the OTHER checkpoint: up to nine, in any mix of stills, clips
# and soundtracks. The clip does not reproduce them; it borrows the identity and
# appearance they carry into a brand-new scene.
TensorSharp.Cli --config config/minimax-h3-ref2va.json \
  --ref-image person.png --ref-image jacket.png --ref-image street.png \
  --prompt "she walks through a night market, neon reflections" --output out.mp4
```

| Config | Denoiser | VAEs | Text encoder | Modes |
|---|---|---|---|---|
| [`minimax-h3-fl2va.json`](minimax-h3-fl2va.json) | FL2VA pruned (Q4_K) | H3 video **+ H3 audio** | Qwen3-VL-32B | T2V + I2V + first/last frame |
| [`minimax-h3-ref2va.json`](minimax-h3-ref2va.json) | Ref2VA pruned (Q4_K) | H3 video **+ H3 audio** | Qwen3-VL-32B | T2V + references (stills, clips, soundtracks) |

Notes:

- **The first run downloads ~33.5 GB** — denoiser 10.64 GiB, text encoder 16.97 GiB,
  video VAE 5.21 GB, audio VAE 0.61 GB. All four come from unsloth, with Comfy-Org
  listed as a second source for the two VAEs. Each network is loaded and released in
  turn, so peak VRAM is `max(...)` and not the sum.
- **Drop the `audio-vae` entry and you still get video**, just silent — that entry is
  what decodes the audio half of the latent.
- **The text encoder ships no tokenizer**, and auto-download cannot fill that gap: it
  only resolves options that are flags, and the tokenizer is not one. Put `vocab.json`
  and `merges.txt` from
  [MiniMaxAI/MiniMax-H3/processor](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor)
  next to the encoder GGUF in `${modelRoot}`, or point `TS_VIDEO_TOKENIZER` at the
  folder holding them. Without them the run stops when the encoder loads.
- **Steps and guidance are host-specific, so no shipped config sets them.** The server
  takes `--video-steps N` and has no `--cfg` at all; the CLI takes `--diffusion-steps N`
  and `--cfg`. A flag the server does not know is not ignored — it refuses to start —
  so a config carrying a CLI-only key would break `TensorSharp.Server --config`. H3 is
  CFG-distilled and enforces cfg 1.0 itself; its default is 20 steps, and 4-8 is the
  fast operating point.
- **`video-frames` snaps to the `17k+5` grid** (5, 22, 39, 56, 73, 90 …), not Wan's
  `4k+1`, and fps is pinned to 24.
- **References need the other checkpoint**, which now has its own file. FL2VA and Ref2VA
  are separate checkpoints rather than a setting: `minimax-h3-fl2va.json` refuses
  `--ref-image` and `minimax-h3-ref2va.json` refuses `--image` as a keyframe, and each
  error names the other file. Only the `model` entry differs between the two configs, so
  the text encoder and both VAEs are shared and will not download twice.
- **Ref2VA takes up to nine references in any mix** — `--ref-image` (a still),
  `--ref-video` (a clip, or a folder of frames), `--ref-audio` (a soundtrack), and
  `--ref-video-audio` to give the i-th `--ref-video` its own sound. Each takes its own
  stretch of the shared timeline before the generated clip and keeps its own aspect
  ratio, so every reference you add lengthens the packed sequence the DiT attends over.

## Video generation, video-only (Wan)

Wan generates **video alone**, from three cooperating networks instead of H3's four,
and a config file is still the easiest way to run it — it names them all and
downloads whatever is missing:

```bash
# Text-to-video and image-to-video, 4-step distilled. Start here.
TensorSharp.Server --config config/wan-video-ti2v-5b-turbo.json

TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --prompt "a red fox trotting through falling snow" --output fox.mp4

# Image-to-video: the image becomes frame 0 and the prompt drives the motion.
TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --image first_frame.png --prompt "she turns and smiles" --output clip.mp4
```

| Config | DiT | VAE | Text encoder | Modes |
|---|---|---|---|---|
| [`wan-video-ti2v-5b-turbo.json`](wan-video-ti2v-5b-turbo.json) | TI2V-5B Turbo (4-step) | Wan 2.2 | UMT5-XXL | T2V + I2V |
| [`wan-video-ti2v-5b.json`](wan-video-ti2v-5b.json) | TI2V-5B (50-step) | Wan 2.2 | UMT5-XXL | T2V + I2V |
| [`wan-video-i2v-a14b.json`](wan-video-i2v-a14b.json) | A14B high **+** low noise | Wan 2.1 | UMT5-XXL | I2V only |

Notes:

- **The VAEs are not interchangeable.** TI2V-5B uses the Wan **2.2** VAE (48-channel
  latent, 16×16×4 compression); A14B and the Wan 2.1 models use the Wan **2.1** VAE
  (16-channel, 8×8×4). Each config names the right one.
- **UMT5-XXL is shared** by every Wan model, so the ~6 GB encoder downloads once and
  the other Wan configs reuse it from `${modelRoot}`.
- **A14B needs both experts.** They are auto-paired by their `high_noise`/`low_noise`
  names when they sit in the same folder; `video-dit2` in the config is what lets the
  second one auto-download.
- **Turbo/Lightning/distilled checkpoints are detected from the file name** and
  switch to 4 steps with guidance off — around 25× fewer DiT passes. The startup log
  prints `step-distilled checkpoint detected`; `--diffusion-steps` / `--cfg` override
  it.
- Each network is loaded and released in turn, so peak VRAM is
  `max(text encoder, DiT, VAE)` rather than their sum.
- `video-frames` is snapped to `4k+1`. Halving it is the cheapest quality-neutral
  saving: self-attention is O(tokens²), so 121 → 61 frames is roughly 4× less
  attention work and half the VAE decode.
