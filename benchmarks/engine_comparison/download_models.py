#!/usr/bin/env python3
"""
Pre-fetch the benchmark models declared in a config file.

`run_matrix.py` already downloads whatever is missing when it starts, so this
script is only for provisioning a host ahead of time (a CI cache step, a fresh
GPU box) or for re-fetching a corrupted file with `--force`.

For every selected model it downloads each file the config resolves to a local
path — `gguf` (all shards of a split GGUF), `mmproj`, `mtp_draft` and any
`components` — from the model's declared source into exactly that path, so a
subsequent `run_matrix.py --config <same config>` finds them locally.

Sources are declared per model as `"source": "<hf-repo-id>"` (or
`{"hf_repo": ..., "revision": ...}`, or a direct base URL), with optional
per-file `{"path": ..., "url": ...}` overrides; the legacy `_hf` field is
accepted as a source too. Files already present are skipped, so this is cheap
to re-run.

Usage:
    python download_models.py [--config benchmark_config_ci.json] \
        [--models gemma4-12b,qwen36-35b-a3b] [--force]

Respects the same env overrides as the rest of the harness (BENCH_MODEL_ROOT,
BENCH_CONFIG, ...). Set HF_TOKEN for gated Hugging Face repos.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _peek_config_arg(argv) -> str | None:
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return None


HERE = Path(__file__).resolve().parent
_cfg_arg = _peek_config_arg(sys.argv[1:]) or os.environ.get("BENCH_CONFIG") \
    or str(HERE / "benchmark_config_ci.json")
os.environ["BENCH_CONFIG"] = _cfg_arg

import config      # noqa: E402  (must come after BENCH_CONFIG is set)
import downloads   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="benchmark config file (default: benchmark_config_ci.json)")
    ap.add_argument("--models", default=None,
                    help="comma list of model ids to fetch (default: the config's "
                         "defaults.models)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even when the file is already present")
    args = ap.parse_args()

    wanted = [m.strip() for m in (args.models or "").split(",") if m.strip()] \
        or list(config.DEFAULT_MODELS)
    mode = "force" if args.force else "auto"

    failed = []
    for model_id in wanted:
        model = config.MODELS.get(model_id)
        if model is None:
            print(f"[{model_id}] unknown model id (known: {', '.join(config.MODELS)})")
            failed.append(model_id)
            continue
        for role, path, url in model.files():
            if path.exists() and mode != "force":
                print(f"[{model_id}] present: {path}")
            elif not url:
                print(f"[{model_id}] {role}: no source URL in {config.CONFIG_PATH.name}; "
                      f"expecting {path} to already exist locally")
        ok, detail = downloads.ensure_model(model, mode=mode)
        if not ok:
            print(f"[{model_id}] {detail}")
            failed.append(model_id)
    if failed:
        raise SystemExit(f"model download(s) failed: {', '.join(failed)}")
    print("all requested model files are in place")


if __name__ == "__main__":
    main()
