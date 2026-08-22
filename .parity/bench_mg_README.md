# Muse-Glimmer vs llama.cpp benchmark harness (Linux / RTX PRO 6000)

Files here reproduce the numbers in [`docs/models/muse-glimmer.md` §5](../docs/models/muse-glimmer.md#5-performance),
measured 2026-08-13 on one RTX PRO 6000 Blackwell with `Muse-Glimmer-30B-Q8_0`.

| File | What it is |
|---|---|
| `bench_mg_linux.sh` | The ladder runner: one context, five variants (llama.cpp plain / TS plain / llama.cpp DFlash / TS DFlash / TS DFlash with the confidence floor removed), engines alternating, VRAM sampled |
| `bench_mg_linux2.sh` | Same plus `TS_ENV=` (for `TS_MUSE_GLIMMER_FUSED=0` / `TS_DFLASH_FUSED=0` A/B) and a `nat16k` natural-prose context |
| `bench_mg_driver.sh` | Drives the corrected long-context re-run and the supplementary passes in order |
| `bench_mg_stats.sh` | llama.cpp with `-v`, purely to capture `draft acceptance = …` / `graphs reused = …` (llama-cli hides the per-slot summary otherwise). Not used for timings |
| `bench_mg_analyze.py` | CSV -> per-point mean/range tables plus a TS/llama.cpp ratio table |
| `bench_mg_blackwell_q8.csv` | The raw run behind the published table |
| `bench_mg_win.sh` | Git-Bash-on-Windows sibling of `bench_mg_linux.sh` (taskkill instead of pkill, and it takes the prompt token count from TensorSharp's own `--dump-prompt` rather than `llama-tokenize`). Used for the 2026-08-13 RTX 3080 Laptop before/after table |
| `bench_mg_win_analyze.py` | Tables for `bench_mg_win.sh`, including a TensorSharp before/after column when the CSV holds more than one `TAG` |

## Setup the scripts assume

```bash
# on the benchmark box
MODEL=/workspace/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf
DRAFT=/workspace/models/muse-glimmer/dflash-kquant.gguf
CLI=/root/TensorSharp/TensorSharp.Cli/bin/TensorSharp.Cli
LLAMA=/root/llama.cpp/build/bin/llama-cli   # -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
# prompts in ./parity/ next to the script (q_text, q_512, q_2k, q_16k, q_32k, q_64k, q_128k + tpl_* copies)
```

## Four things that will silently ruin a run

1. **The two engines must prefill the same tokens.** TensorSharp applies the chat
   template; `llama-cli -no-cnv -f` does not. Dump the rendered prompt once with
   `TensorSharp.Cli --dump-prompt` and feed *that* to llama.cpp — the `tpl_*.txt`
   files here are exactly that. Verify with
   `llama-tokenize -m … -f tpl_x.txt --show-count --ids` against the
   `prefill complete: tokens=` line TensorSharp logs.
2. **Normalize line endings first.** The prompt files were authored on Windows.
   An LF copy for llama.cpp against a CRLF original for TensorSharp is 1.2% more
   tokens on the TensorSharp side and a different continuation.
3. **Pass no sampler flags to `TensorSharp.Cli`.** `SamplingConfig.IsGreedy`
   requires `TopK <= 0`, so `--top-k 1` disarms the speculative path and the run
   reports plain decode under a "dflash" label. The scripts assert
   `cli.inference speculative:` is in the log and mark the row otherwise.
4. **128 generated tokens is too few for a speculative comparison.** It is
   shorter than two of TensorSharp's `ParkedProbeInterval` windows, so a single
   cost-governor misfire moves the number by 2x. Use `NGEN=512` for a
   steady-state figure.

## Reproduce

```bash
TAG=m1 REPS=2 bash bench_mg_linux.sh                    # the full ladder
python3 bench_mg_analyze.py bench_results.csv           # tables
```
