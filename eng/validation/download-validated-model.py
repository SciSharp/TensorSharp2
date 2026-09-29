#!/usr/bin/env python3
"""Download public model shards from a pinned manifest and verify size and SHA256."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, unquote


def download_buffered(url, part, expected_bytes, log, retries=5, source_offset=0):
    """Resume through large positioned writes, avoiding tiny O_APPEND writes on FUSE."""
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_bytes:
            raise ValueError("Partial file exceeds expected size: " + part.name)
        if offset == expected_bytes:
            return
        if source_offset < 0:
            raise ValueError("Source offset must not be negative")
        remote_offset = source_offset + offset
        source_bytes = source_offset + expected_bytes
        headers = {"Accept-Encoding": "identity"}
        if remote_offset:
            headers["Range"] = f"bytes={remote_offset}-"
        print(f"attempt={attempt + 1} offset={offset} expected_bytes={expected_bytes}", file=log, flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                expected_status = 206 if remote_offset else 200
                expected_range = f"bytes {remote_offset}-{source_bytes - 1}/{source_bytes}" if remote_offset else None
                if response.status != expected_status or response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Server did not honor the exact resumable byte range")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected_bytes - offset:
                    raise ValueError("Response Content-Length differs from the expected remaining bytes")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected encoded model response")
                # r+b + seek preserves the existing prefix without O_APPEND's
                # per-write serialization on some network filesystems.
                with part.open("r+b" if part.exists() else "wb") as output:
                    output.seek(offset)
                    while offset < expected_bytes:
                        block = response.read(min(8 * 1024**2, expected_bytes - offset))
                        if not block:
                            raise OSError("Response ended before the expected file size")
                        output.write(block)
                        offset += len(block)
                    if response.read(1):
                        raise ValueError("Response exceeds the expected file size")
            return
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            print(f"retryable_error={error!r}", file=log, flush=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--transport", choices=("curl", "buffered"), default="curl",
                   help="buffered uses HTTP/1.1 and large positioned writes for network filesystems")
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.directory.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    def download(item):
        name = unquote(urlsplit(item["url"]).path.rsplit("/", 1)[-1])
        if not name or Path(name).name != name:
            raise ValueError("Invalid model basename")
        path = a.directory / name
        part = path.with_suffix(path.suffix + ".part")
        if not path.exists():
            if part.exists() and part.stat().st_size > item["bytes"]:
                raise ValueError("Partial file exceeds expected size: " + name)
            # Hash a complete partial after an interrupted verification. Asking
            # the server for an EOF byte range can otherwise refuse a valid file.
            if not part.exists() or part.stat().st_size < item["bytes"]:
                with (a.report.parent / (name + ".download.log")).open("a") as log:
                    if a.transport == "buffered":
                        download_buffered(item["url"], part, item["bytes"], log)
                    else:
                        subprocess.run(["curl", "-fL", "--retry", "5", "--retry-all-errors", "--retry-delay", "2", "--continue-at", "-",
                                        "--output", str(part), item["url"]], stdout=log, stderr=subprocess.STDOUT, check=True)
            candidate = part
        else:
            candidate = path
        if candidate.stat().st_size != item["bytes"]:
            raise ValueError("Size differs: " + name)
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != item["sha256"]:
            raise ValueError("SHA256 differs: " + name)
        if candidate != path:
            candidate.rename(path)
        row = dict(item, path=str(path), verified=True, actual_sha256=actual)
        print(name, "verified", flush=True)
        return row
    report = {"status": "running", "manifest": str(a.manifest), "transport": a.transport,
              "started_unix": started, "shards": []}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for row in pool.map(download, manifest["shards"]):
                report["shards"].append(row)
                a.report.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "verified"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["wall_seconds"] = time.time() - started
        a.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
