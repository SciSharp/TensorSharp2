#!/usr/bin/env python3
"""Compare bounded full and resumable HTTP responses for a pinned model shard.

Payload samples are discarded. This measures transport only, never verifies a
checkpoint. urllib uses HTTP/1.1; results must not be presented as HTTP/2 tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--index", type=int, default=0, help="Zero-based shard index")
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--sample-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    item = manifest["shards"][args.index]
    if not 0 < args.sample_bytes <= 512 * 1024**2 or not 0 <= args.offset < item["bytes"] - args.sample_bytes:
        parser.error("Use a sample in (0, 512 MiB] and an offset with enough remaining bytes")
    report = {"source": item, "http_version": "HTTP/1.1", "sample_bytes": args.sample_bytes,
              "range_offset": args.offset, "checkpoint_verified": False, "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in ("full", "range", "range-fresh", "full-fresh"):
        url = item["url"]
        if mode.endswith("fresh"):
            url += ("&" if "?" in url else "?") + "transport_probe=" + str(time.time_ns())
        ranged = mode.startswith("range")
        headers = {"Accept-Encoding": "identity"}
        if ranged:
            headers["Range"] = f"bytes={args.offset}-"
        started = time.perf_counter()
        row = {"mode": mode, "started_unix": time.time()}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                row.update(status=response.status, headers_seconds=time.perf_counter() - started,
                           content_range=response.headers.get("Content-Range"),
                           final_host=urllib.parse.urlsplit(response.url).hostname)
                expected_status = 206 if ranged else 200
                expected_range = f"bytes {args.offset}-{item['bytes'] - 1}/{item['bytes']}" if ranged else None
                if response.status != expected_status or row["content_range"] != expected_range:
                    raise ValueError("Response does not match the requested full/open-ended range")
                count = 0
                digest = hashlib.sha256()
                while count < args.sample_bytes:
                    block = response.read(min(1024**2, args.sample_bytes - count))
                    if not block:
                        raise ValueError("Response ended before the bounded sample was complete")
                    count += len(block)
                    digest.update(block)
            elapsed = time.perf_counter() - started
            row.update(bytes=count, wall_seconds=elapsed, bytes_per_second=count / elapsed,
                       payload_bytes_per_second=count / (elapsed - row["headers_seconds"]),
                       sample_sha256=digest.hexdigest())
        except Exception as error:
            row.update(error=repr(error), wall_seconds=time.perf_counter() - started)
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
