#!/usr/bin/env python3
"""Submit local attachments to Jev, inline or through /api/upload (stdlib only)."""

import argparse
import base64
import json
import mimetypes
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post(url, data, content_type, timeout):
    request = Request(url, data=data, headers={"Content-Type": content_type})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def attachments(paths, args):
    sizes = [path.stat().st_size for path in paths]
    if any(size <= 0 or size > 32 * 1024 * 1024 for size in sizes):
        raise ValueError("Jev accepts 1 byte to 32 MiB per attachment")
    if sum(sizes) > 64 * 1024 * 1024:
        raise ValueError("Jev accepts at most 64 MiB of attachments per request")
    if not args.upload:
        return [{"name": path.name, "data": base64.b64encode(path.read_bytes()).decode("ascii")} for path in paths]

    # Send the entire selection in one multipart request. The returned list is
    # in the same order even when two local files have the same basename.
    boundary = "jev-" + uuid.uuid4().hex
    parts = []
    for path in paths:
        if any(c in path.name for c in '\r\n"\\'):
            raise ValueError(f"{path}: rename the file before uploading it")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        parts.extend([header, path.read_bytes(), b"\r\n"])
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    body = b"".join(parts)
    uploaded = post(args.endpoint + "/api/upload", body,
                    f"multipart/form-data; boundary={boundary}", args.timeout)
    files = uploaded.get("files", [uploaded])
    if uploaded.get("ok") is not True or not isinstance(files, list) or len(files) != len(paths):
        raise ValueError("Upload did not return every requested file")
    if any(not isinstance(item, dict) or item.get("ok") is not True or not isinstance(item.get("file"), str)
           or not item["file"] for item in files):
        raise ValueError("Upload returned an invalid file reference")
    return [{"file": item["file"], "name": path.name} for path, item in zip(paths, files)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", metavar="FILE", nargs="+", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5000")
    parser.add_argument("--upload", action="store_true",
                        help="upload all files together, then submit bare server file references")
    parser.add_argument("--field", choices=("files", "documents", "videos", "audios"),
                        default="files", help="files detects the kind from the extension")
    parser.add_argument("--state", default="Review the attached evidence.")
    parser.add_argument("--question", default="Does the evidence report an active service outage?")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.endpoint = args.endpoint.rstrip("/")
    if len(args.paths) > 8:
        parser.error("Jev accepts at most 8 attachments across the attachment arrays")
    try:
        body = {
            "model": "jev-latest",
            "state": args.state,
            args.field: attachments(args.paths, args),
            "questions": {"match": {"type": "noul", "instructions": args.question}},
            "samples": 1,
            "seed": 42,
        }
        response = post(args.endpoint + "/v1/systemone", json.dumps(body).encode("utf-8"),
                        "application/json", args.timeout)
        print(json.dumps(response, indent=2, ensure_ascii=False))
    except HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except (OSError, URLError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
