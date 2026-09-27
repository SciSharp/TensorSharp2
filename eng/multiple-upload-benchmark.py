#!/usr/bin/env python3
"""Compare serial and batched /api/upload transport, without model inference.

python3 eng/multiple-upload-benchmark.py --endpoint http://127.0.0.1:5188

Paired rounds alternate serial-first and batch-first. Defaults: 1, 2 and 8 CSV
files, each 64 KiB or 1 MiB; two warmup pairs and ten measured pairs per case.
Every stored URL is downloaded and checked byte-for-byte by SHA-256 after the
uploads. These downloads, fixture reads and multipart assembly are excluded from
upload latency. Server storage/processing and HTTP response reads are included.
Synthetic CSV transport measurements do not establish model or media quality.
Uploads remain on the server and count toward its configured storage quota/TTL.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jev_attachments", ROOT / "eng/jev-attachments-benchmark.py")
UPLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPLOAD)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_list(value):
    values = [int(item) for item in value.split(",")]
    require(values and min(values) > 0 and len(values) == len(set(values)), "Expected distinct positive comma-separated integers")
    return values


def make_files(directory, count, size, same_basename=False):
    require(size >= 64, "Each fixture must contain at least 64 bytes")
    paths = []
    for index in range(count):
        folder = directory / str(index)
        folder.mkdir(parents=True)
        path = folder / ("input.csv" if same_basename else f"input-{index:02d}.csv")
        header = f"file_id,payload\n{index},".encode()
        pattern = f"payload-{index:02d}-".encode()
        remaining = size - len(header) - 1
        path.write_bytes(header + (pattern * (remaining // len(pattern) + 1))[:remaining] + b"\n")
        paths.append(path)
    return paths


def check_response(result, paths):
    require(result["status"] == 200, f"Upload HTTP {result['status']}: {result.get('error') or result.get('body')}")
    body = result["body"]
    require(isinstance(body, dict) and body.get("ok") is True, "Upload response missing ok=true")
    items = body.get("files", [body])
    require(isinstance(items, list) and len(items) == len(paths), "Upload response dropped or added files")
    for index, (item, path) in enumerate(zip(items, paths)):
        require(isinstance(item, dict) and item.get("ok") is True, f"File {index} missing ok=true")
        require(item.get("fileName") == path.name, f"File {index} name/order differs from submitted file")
        reference = item.get("file")
        require(isinstance(reference, str) and reference and not any(c in reference for c in "/\\")
                and reference not in (".", ".."), f"File {index} has an invalid stored reference")
        require(isinstance(item.get("url"), str) and item["url"], f"File {index} has no download URL")
    return items


class NoDownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        # Upload URLs must already point at their stored bytes. Do not forward a
        # caller's Authorization header to an origin chosen by a redirect.
        return None


def check_download(args, item, path):
    url = urllib.parse.urljoin(args.upload_url, item["url"])
    expected_origin = urllib.parse.urlsplit(args.upload_url)
    actual = urllib.parse.urlsplit(url)
    require((actual.scheme, actual.netloc) == (expected_origin.scheme, expected_origin.netloc),
            "Refusing an upload download URL outside the benchmark server")
    require(urllib.parse.unquote(actual.path).rsplit("/", 1)[-1] == item["file"],
            "Download URL does not match its stored reference")
    headers = {"Authorization": "Bearer " + args.api_key} if args.api_key else {}
    expected = path.read_bytes()
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.build_opener(NoDownloadRedirect()).open(request, timeout=args.timeout) as response:
        require(response.status == 200, f"Download HTTP {response.status}")
        actual_bytes = response.read(len(expected) + 1)
    expected_hash = hashlib.sha256(expected).hexdigest()
    actual_hash = hashlib.sha256(actual_bytes).hexdigest()
    require(len(actual_bytes) == len(expected) and actual_hash == expected_hash,
            f"Downloaded content differs for {path.name}: expected {expected_hash}, received {actual_hash}")
    return {"fileName": path.name, "file": item["file"], "bytes": len(expected), "sha256": actual_hash}


def run_once(args, paths, mode):
    row = {"mode": mode, "status": "failed", "payload_bytes": sum(path.stat().st_size for path in paths),
           "file_count": len(paths), "upload_ms": 0.0, "upload_roundtrips": 0,
           "download_roundtrips": 0, "uploads": [], "verified": [], "errors": []}
    groups = [[path] for path in paths] if mode == "serial" else [paths]
    stored = []
    for group in groups:
        result = (UPLOAD.upload(args.upload_url, group[0], args.timeout, args.api_key) if mode == "serial"
                  else UPLOAD.upload_many(args.upload_url, group, args.timeout, args.api_key))
        row["uploads"].append(result)
        row["upload_ms"] += result["elapsed_ms"]
        row["upload_roundtrips"] += 1
        try:
            stored.extend(zip(check_response(result, group), group))
        except (KeyError, TypeError, ValueError) as error:
            row["errors"].append(str(error))
    references = [item["file"] for item, _ in stored]
    if len(set(references)) != len(references):
        row["errors"].append("Different submitted files reused the same stored reference")
    # Verify only after all measured uploads so GET timing never enters upload_ms.
    for item, path in stored:
        row["download_roundtrips"] += 1
        try:
            row["verified"].append(check_download(args, item, path))
        except (OSError, ValueError, TimeoutError) as error:
            row["errors"].append(f"{path.name}: {type(error).__name__}: {error}")
    if not row["errors"] and len(row["verified"]) == len(paths):
        row["status"] = "passed"
    return row


def summarize(rows):
    passed = [row for row in rows if row["status"] == "passed"]
    durations = [row["upload_ms"] for row in passed]
    total_ms = sum(durations)
    return {"samples": len(rows), "passed": len(passed), "failed": len(rows) - len(passed),
            "upload_latency_ms": {"p50": UPLOAD.BENCH.percentile(durations, .5),
                                  "p95": UPLOAD.BENCH.percentile(durations, .95)},
            "successful_payload_bytes": sum(row["payload_bytes"] for row in passed),
            "payload_mib_per_upload_second": (sum(row["payload_bytes"] for row in passed) / (1024 ** 2)
                                               / (total_ms / 1000)) if total_ms else None,
            "upload_roundtrips": sum(row["upload_roundtrips"] for row in rows),
            "download_roundtrips_excluded_from_timing": sum(row["download_roundtrips"] for row in rows)}


def paired_summary(rows):
    ratios = []
    for repetition in sorted({row["repetition"] for row in rows}):
        pair = {row["mode"]: row for row in rows if row["repetition"] == repetition}
        if all(pair.get(mode, {}).get("status") == "passed" for mode in ("serial", "batch")):
            if pair["batch"]["upload_ms"] > 0:
                ratios.append(pair["serial"]["upload_ms"] / pair["batch"]["upload_ms"])
    return {"validated_pairs": len(ratios), "serial_over_batch_latency_ratio": {
        "p50": UPLOAD.BENCH.percentile(ratios, .5), "p95": UPLOAD.BENCH.percentile(ratios, .95)}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--counts", type=positive_list, default=[1, 2, 8])
    parser.add_argument("--sizes", type=positive_list, default=[65536, 1048576], help="Per-file bytes")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--same-basename", action="store_true", help="Each distinct file is named input.csv")
    parser.add_argument("--api-key-env")
    parser.add_argument("--description", default="")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multiple-upload-benchmark"
                        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args(argv)
    require(args.repeats > 0 and args.warmup >= 0 and math.isfinite(args.timeout) and args.timeout > 0,
            "repeats and timeout must be positive; warmup must be nonnegative")
    require(min(args.sizes) >= 64 and max(args.counts) <= 32, "Sizes must be >=64 bytes; file counts must be <=32")
    endpoint = urllib.parse.urlsplit(args.endpoint)
    require(endpoint.scheme in ("http", "https") and endpoint.netloc and not endpoint.username
            and not endpoint.password and not endpoint.query and not endpoint.fragment,
            "Endpoint must be an HTTP(S) server URL without credentials, query, or fragment")
    args.upload_url = args.endpoint.rstrip("/").removesuffix("/api/upload") + "/api/upload"
    args.api_key = os.environ[args.api_key_env] if args.api_key_env else None
    return args


def main(argv=None):
    args = parse_args(argv)
    output = UPLOAD.output_directory(args.output)
    require(not (output / "requests.jsonl").exists() and not (output / "summary.json").exists(),
            "Choose a new output directory; existing evidence will not be overwritten")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "endpoint": args.upload_url,
              "description": args.description, "platform": platform.platform(), "scope": "upload transport only; no model inference",
              "configuration": {"counts": args.counts, "file_sizes_bytes": args.sizes, "repeats": args.repeats,
                                "warmup_pairs": args.warmup, "same_basename": args.same_basename},
              "limitations": ["Latency sums upload HTTP timings; fixture reads, multipart assembly and verification GETs are excluded.",
                              "Server upload processing and response reads are included; throughput counts file payload bytes only.",
                              "Successful samples only contribute latency/throughput; all failures, including warmups, fail this run.",
                              "Paired alternating order reduces drift but does not simulate concurrent users or establish model quality.",
                              "Fresh references reuse synthetic CSV content and warm caches; uploads remain under server quota/TTL."],
              "groups": [], "failures": []}
    with tempfile.TemporaryDirectory(prefix="tensorsharp-upload-bench-") as temporary, (output / "requests.jsonl").open("w", encoding="utf-8") as evidence:
        for size in args.sizes:
            for count in args.counts:
                paths = make_files(Path(temporary) / f"{size}-{count}", count, size, args.same_basename)
                measured = []
                for phase, repeats in (("warmup", args.warmup), ("measured", args.repeats)):
                    for repetition in range(repeats):
                        modes = ("serial", "batch") if repetition % 2 == 0 else ("batch", "serial")
                        for order, mode in enumerate(modes):
                            row = {"phase": phase, "repetition": repetition, "pair_order": order, "file_size_bytes": size,
                                   **run_once(args, paths, mode)}
                            evidence.write(json.dumps(row, allow_nan=False) + "\n")
                            evidence.flush()
                            if row["status"] != "passed":
                                report["failures"].append(row)
                            if phase == "measured":
                                measured.append(row)
                            print(f"{phase} {count}x{size} {mode} pair={repetition}: {row['status']} {row['upload_ms']:.2f} ms", flush=True)
                report["groups"].append({"file_count": count, "file_size_bytes": size, "payload_bytes_per_iteration": count * size,
                    "modes": {mode: summarize([row for row in measured if row["mode"] == mode]) for mode in ("serial", "batch")},
                    **paired_summary(measured)})
    report["status"] = "failed" if report["failures"] else "passed"
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {output / 'summary.json'}", flush=True)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
