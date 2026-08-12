# Muse-Glimmer parity goldens

Golden outputs captured from a `llama-server` running the same GGUFs, consumed by
`InferenceWeb.Tests/MuseGlimmerParityTests.cs`. The tests feed the RECORDED prompt
token ids into the model, so the forward pass is compared in isolation from any
tokenizer or chat-template difference; a separate test covers the tokenizer.

Regenerate:

```bash
llama-server -m Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
             --mmproj mmproj-Muse-Glimmer-30B-Q8_0.gguf -ngl 99 -c 4096 --port 8899
python gen_ref.py        http://127.0.0.1:8899 ref_text.json
python gen_ref_vision.py http://127.0.0.1:8899 ref_vision.json

# and, with the DFlash drafter attached, to confirm llama.cpp's own speculative
# path reproduces its baseline token-for-token:
llama-server -m Muse-Glimmer-30B-UD-IQ2_XXS.gguf -md dflash-kquant.gguf \
             --spec-type draft-dflash --spec-draft-n-max 15 -ngl 99 -ngld 99 --port 8898
python gen_ref.py http://127.0.0.1:8898 ref_text_dflash.json
```

A long-context golden exercises the sliding-window mask, which the short prompts
never reach (the window is 2048):

```bash
llama-server -m Muse-Glimmer-30B-UD-IQ2_XXS.gguf -ngl 99 -c 8192 --port 8899
python gen_ref_long.py http://127.0.0.1:8899 ref_text_long.json
```

| File | Contents |
|---|---|
| `ref_text.json` | 5 prompts: prompt token ids, greedy continuation ids/pieces, first-token top-10 logprobs, timings |
| `ref_text_long.json` | one ~4650-token prompt (2.3x the sliding window) + its greedy continuation |
| `ref_text_dflash.json` | same 5 prompts through llama.cpp's DFlash path (identical tokens, 1.33-4.77x) |
| `ref_vision.json` | image descriptions for `img_small.png` (336x336) and a 1024x1024 photo |
| `img_small.png` | 336x336 test image (maps to a clean 12x12 token grid) |
| `q_text.txt`, `q_image.txt` | CLI prompts used for the benchmarks |
