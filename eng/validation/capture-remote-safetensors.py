#!/usr/bin/env python3
"""Capture pinned safetensors directories selected by tensor prefix, without weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Metadata exceeds 32 MiB bound")
    return data


def byte_range(url, first, stop, total):
    request = urllib.request.Request(url + f"?capture_range={first}-{stop}", headers={
        "Range": f"bytes={first}-{stop-1}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{stop-1}/{total}":
            raise ValueError("Server did not honor bounded byte range")
        data = response.read(stop-first+1)
    if len(data) != stop-first:
        raise ValueError("Byte range length differs")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repository")
    p.add_argument("--revision", required=True)
    p.add_argument("--tensor-prefix", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        p.error("Revision must be an immutable commit SHA")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output / "metadata"
    headers = args.output / "headers"
    metadata.mkdir(exist_ok=True)
    headers.mkdir(exist_ok=True)
    base = f"https://huggingface.co/{args.repository}/resolve/{args.revision}/"
    api = json.loads(fetch(f"https://huggingface.co/api/models/{args.repository}/revision/{args.revision}?blobs=true"))
    save(metadata / "repository-api.json", api)
    files = {row["rfilename"]: row for row in api["siblings"]}
    parsed = {}
    for name in ("config.json", "model.safetensors.index.json"):
        data = fetch(base+name)
        (metadata / name).write_bytes(data)
        parsed[name] = json.loads(data)
    index = parsed["model.safetensors.index.json"]["weight_map"]
    selected = {file for name, file in index.items() if name.startswith(args.tensor_prefix)}
    if not selected:
        raise ValueError("Tensor prefix matched no checkpoint tensors")
    report = dict(repository=args.repository, revision=args.revision, status="captured-header-only", shards=[])
    for name in sorted(selected):
        file = files[name]
        size = file["size"]
        prefix = byte_range(base+name, 0, 8, size)
        length, = struct.unpack("<Q", prefix)
        if not 0 < length <= 32 * 1024 * 1024:
            raise ValueError("Invalid safetensors directory length")
        prefix += byte_range(base+name, 8, 8+length, size)
        directory = json.loads(prefix[8:])
        tensors = {key: value for key, value in directory.items() if key != "__metadata__"}
        if any(not key.startswith(args.tensor_prefix) for key in tensors):
            raise ValueError("Selected shard also contains unrelated tensors: " + name)
        if set(tensors) != {key for key, file in index.items() if file == name}:
            raise ValueError("Safetensors directory differs from checkpoint index")
        (headers / (name + ".header-prefix")).write_bytes(prefix)
        row = dict(file=name, file_bytes=size, publisher_sha256=file["lfs"]["sha256"], origin_url=base+name,
                   header_prefix_bytes=len(prefix), header_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                   tensor_metadata=tensors)
        report["shards"].append(row)
        print(name, size, len(tensors), flush=True)
    save(headers / "capture.json", report)


if __name__ == "__main__":
    main()
