#!/usr/bin/env python3
"""Stage missing model suffixes on another host; never qualify a full checkpoint.

Offsets name immutable shard basenames and their existing prefix lengths. The
full original SHA256 must pass after prefix/tail assembly before model use.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
from urllib.parse import unquote, urlsplit


SPEC = importlib.util.spec_from_file_location("download_validated_model", Path(__file__).with_name("download-validated-model.py"))
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def bind_source_identity(path, part, identity):
    identity_path = path.with_suffix(path.suffix + ".source.json")
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing tail belongs to a different pinned source/offset: " + path.name)
    elif part.exists() or path.exists():
        raise ValueError("Existing tail has no source identity; refusing unsafe resume: " + path.name)
    else:
        with identity_path.open("x") as output:
            output.write(json.dumps(identity, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("offsets", type=Path)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error("workers must be in [1, 8]")
    offsets = json.loads(a.offsets.read_text())
    manifest = json.loads(a.manifest.read_text())
    jobs = []
    for item in manifest["shards"]:
        name = unquote(urlsplit(item["url"]).path.rsplit("/", 1)[-1])
        if not name or Path(name).name != name:
            raise ValueError("Invalid shard basename")
        if name not in offsets:
            continue
        offset = offsets[name]
        if not isinstance(offset, int) or not 0 < offset < item["bytes"]:
            raise ValueError("Tail source offset must be inside its immutable shard")
        jobs.append((item, name, offset))
    if len(jobs) != len(offsets) or not jobs:
        raise ValueError("Every supplied offset must identify exactly one manifest shard")
    a.directory.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    required = sum(item["bytes"] - offset for item, _, offset in jobs)
    if shutil.disk_usage(a.directory).free < required + 1024**3:
        raise ValueError("Insufficient free space for the requested missing suffixes")
    report = {"status": "running", "model_verified": False, "manifest_sha256": sha(a.manifest),
              "offsets_sha256": sha(a.offsets), "started_unix": time.time(), "total_tail_bytes": required, "tails": []}

    def stage(job):
        item, name, offset = job
        path = a.directory / (name + ".tail")
        part = path.with_suffix(path.suffix + ".part")
        expected = item["bytes"] - offset
        identity = {"url": item["url"], "bytes": item["bytes"], "sha256": item["sha256"], "source_offset": offset}
        bind_source_identity(path, part, identity)
        if not path.exists():
            with (a.directory / (name + ".log")).open("a") as log:
                DOWNLOADER.download_buffered(item["url"], part, expected, log, source_offset=offset)
            if part.stat().st_size != expected:
                raise ValueError("Staged tail size mismatch")
            part.rename(path)
        if path.stat().st_size != expected:
            raise ValueError("Existing staged tail size mismatch")
        row = {"source": item, "source_offset": offset, "bytes": expected, "path": str(path),
               "tail_sha256": sha(path), "model_verified": False}
        print(json.dumps(row), flush=True)
        return row

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(stage, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                report["tails"].append(future.result())
                a.report.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "staged-tails-only"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["wall_seconds"] = time.time() - report["started_unix"]
        a.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
