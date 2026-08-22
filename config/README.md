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
| [`wan-video-ti2v-5b-turbo.json`](wan-video-ti2v-5b-turbo.json) | Wan 2.2 TI2V-5B Turbo: DiT + video VAE + UMT5 (~9.5 GB) | **Video generation**, 4-step distilled, text- **and** image-to-video |
| [`wan-video-ti2v-5b.json`](wan-video-ti2v-5b.json) | Wan 2.2 TI2V-5B: DiT + video VAE + UMT5 (~11.4 GB) | Video generation, undistilled 50-step reference recipe |
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
- **MTP speculative decoding** (Gemma configs) is a **server** feature — the draft
  head is wired via `mtp-draft-model`; the CLI ignores it. It is lossless. For
  Qwen3.6 the draft head is embedded in the trunk, so just add `"mtp-spec": true`.
- **Image-edit** configs run the DiT pipeline: `--image in.png --prompt "…" --output
  out.png`. Per-edit `--diffusion-steps` / `--cfg` / `--diffusion-seed` are CLI flags.
- **DiffusionGemma** uses the CLI's iterative denoising path; tune it with
  `--diffusion-steps` / `--diffusion-seed` on the command line.
- To make any of these auto-download on another machine, turn a `"model": "…path…"`
  string into an object: `{ "path": "…", "urls": ["https://…"] }` (see the examples
  above).

## Video generation (Wan)

Video is the one pipeline that needs **several cooperating networks**, so a config
file is the easiest way to run it — it names them all and downloads whatever is
missing:

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
  names when they sit in the same folder; `wan-dit2` in the config is what lets the
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
