#!/usr/bin/env python3
"""
Model fetcher: make a config's model files exist locally.

Every model in a benchmark config may declare where its files come from, so a
run on a fresh host is self-provisioning: any file the config resolves to a
local path that is not there yet is downloaded from that source into exactly
that path, and every later run finds it already present.

Config surface (see `config._source_base` / `ModelSpec.files`):

    "some-model": {
      "source": "unsloth/Qwen3.5-9B-GGUF",            # Hugging Face repo id
      "gguf":   "${model_root}/Qwen3.5-9B-Q8_0.gguf", # -> <repo>/resolve/main/<name>
      "mmproj": {"path": "${model_root}/mm.gguf", "url": "mmproj-F16.gguf"}
    }

`source` may also be `{"hf_repo": ..., "revision": ...}` or a direct base URL
(`"https://host/dir/"`); a file's `url` may be a plain name resolved against
that base, or an absolute URL of its own. Split GGUFs (`-00001-of-00005.gguf`)
expand to every shard automatically.

Downloads stream to `<target>.part` and resume with an HTTP Range request when
interrupted, so a partial file never masquerades as a complete model. Set
`HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) for gated Hugging Face repos.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import requests

import config

CHUNK = 8 * 1024 * 1024
_PROGRESS_EVERY_S = 15.0


def _auth_headers(url: str) -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token and "huggingface.co" in url:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def download(url: str, target: Path, *, log=print, retries: int = 3) -> None:
    """Download `url` to `target` (atomically, resumable). Raises on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = _auth_headers(url)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=(30, 600), allow_redirects=True) as r:
                if have and r.status_code == 200:
                    have = 0                      # server ignored the range: restart
                elif have and r.status_code == 416:
                    r.close()                     # already complete
                    part.replace(target)
                    return
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0) + have
                mode = "ab" if have else "wb"
                t_last = time.monotonic()
                done = have
                with open(part, mode) as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - t_last >= _PROGRESS_EVERY_S:
                            pct = f" ({100.0 * done / total:.0f}%)" if total else ""
                            log(f"      {_human(done)}"
                                f"{'/' + _human(total) if total else ''}{pct}")
                            t_last = now
            if total and part.stat().st_size < total:
                raise IOError(f"truncated: {part.stat().st_size} of {total} bytes")
            part.replace(target)
            return
        except requests.HTTPError as ex:          # a bad URL will not fix itself
            status = ex.response.status_code if ex.response is not None else 0
            if 400 <= status < 500 and status not in (408, 429):
                raise RuntimeError(f"download failed: {url} -> {target}: {ex}") from None
            last_err = ex
            if attempt < retries:
                log(f"      retry {attempt}/{retries - 1} after error: {ex}")
                time.sleep(3.0 * attempt)
        except Exception as ex:                   # network hiccup: resume
            last_err = ex
            if attempt < retries:
                log(f"      retry {attempt}/{retries - 1} after error: {ex}")
                time.sleep(3.0 * attempt)
    raise RuntimeError(f"download failed: {url} -> {target}: {last_err}")


def ensure_model(model: config.ModelSpec, *, mode: str = "auto",
                 log=print) -> tuple[bool, str]:
    """Make every file of `model` present locally.

    mode: "auto" downloads what is missing, "never" only reports, "force"
    re-downloads everything. Returns (ok, detail); `detail` names the first
    blocking problem when ok is False.
    """
    wanted = model.files()
    if mode == "force":
        todo = list(wanted)
        for _, p, _ in todo:                      # start clean, do not resume
            for f in (p, p.with_suffix(p.suffix + ".part")):
                try:
                    f.unlink()
                except OSError:
                    pass
    else:
        todo = [(role, p, url) for role, p, url in wanted if not p.exists()]
    if not todo:
        return True, ""
    if mode == "never":
        role, p, _ = todo[0]
        return False, f"model file not found: {p} (downloads disabled)"

    for role, path, url in todo:
        if not url:
            return False, (f"model file not found: {path} — and no source URL for "
                           f"'{role}' in {config.CONFIG_PATH.name} "
                           f"(add a model-level \"source\" or a per-file \"url\")")
        log(f"    fetching {role}: {url}")
        log(f"      -> {path}")
        t0 = time.monotonic()
        try:
            download(url, path, log=log)
        except Exception as ex:
            return False, f"download failed for '{role}': {ex}"
        dt = time.monotonic() - t0
        size = path.stat().st_size if path.exists() else 0
        rate = f" ({_human(size / dt)}/s)" if dt > 0 and size else ""
        log(f"      done: {_human(size)} in {dt:.0f}s{rate}")
    return True, ""


def ensure_models(model_ids, *, mode: str = "auto", log=print) -> dict:
    """Fetch several models up front; returns {model_id: (ok, detail)}."""
    out = {}
    for mid in model_ids:
        model = config.MODELS.get(mid)
        if model is None:
            out[mid] = (False, f"unknown model id '{mid}'")
            continue
        missing = model.missing_files()
        if missing and mode != "never":
            log(f"[{mid}] {len(missing)} file(s) missing; fetching")
        out[mid] = ensure_model(model, mode=mode, log=log)
    return out
