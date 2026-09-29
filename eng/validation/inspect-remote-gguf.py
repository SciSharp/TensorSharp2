#!/usr/bin/env python3
"""Inspect pinned Hugging Face GGUF headers without downloading tensor data.

Records immutable download URLs, expected LFS SHA-256 values and stored tensor
byte ranges. Weight sizes include alignment padding; they exclude runtime,
cache, graph, staging and device-context allocations.
"""
import argparse
import collections
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import struct
import time
import urllib.parse
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def list_files(repository, revision):
    url = f"https://huggingface.co/api/models/{repository}/tree/{revision}?recursive=true&expand=false"
    result = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            result.extend(entry for entry in json.load(response) if entry["type"] == "file")
            links = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', links)
        url = match.group(1) if match else None
    return result


def read_header(url, size, limit):
    stop = min(size, limit)
    error = None
    for attempt in range(4):
        request = urllib.request.Request(url + f"?header_bytes={stop}&attempt={time.time_ns()}",
            headers={"Range": f"bytes=0-{stop - 1}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-{stop - 1}/{size}":
                    raise ValueError("Server did not honor the bounded header range")
                data = response.read(stop + 1)
            if len(data) != stop:
                raise ValueError("Incomplete header range")
            return data
        except Exception as ex:
            error = ex
            if attempt < 3:
                time.sleep(1 + attempt)
    raise error


class Reader:
    def __init__(self, data):
        self.data, self.position = data, 0

    def read(self, fmt):
        value = struct.unpack_from("<" + fmt, self.data, self.position)
        self.position += struct.calcsize("<" + fmt)
        return value[0] if len(value) == 1 else value

    def string(self):
        count = self.read("Q")
        if self.position + count > len(self.data):
            raise EOFError("Header exceeds fetched range")
        value = self.data[self.position:self.position + count].decode("utf-8")
        self.position += count
        return value

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            start = self.position
            subtype, count = self.read("I"), self.read("Q")
            values = [self.value(subtype) for _ in range(count)]
            return values if count <= 256 else {"array_type": subtype, "count": count, "first_eight": values[:8],
                "encoded_sha256": hashlib.sha256(self.data[start:self.position]).hexdigest()}
        return self.read({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                          7: "?", 10: "Q", 11: "q", 12: "d"}[kind])


def inspect(data, file_size):
    reader = Reader(data)
    if data[:4] != b"GGUF":
        raise ValueError("Not a GGUF file")
    reader.position = 4
    version, tensor_count, metadata_count = reader.read("I"), reader.read("Q"), reader.read("Q")
    metadata = {}
    for _ in range(metadata_count):
        name = reader.string()
        metadata[name] = reader.value(reader.read("I"))
    tensors = []
    for _ in range(tensor_count):
        name = reader.string()
        dims = [reader.read("Q") for _ in range(reader.read("I"))]
        tensors.append({"name": name, "dimensions": dims, "type": reader.read("I"), "offset": reader.read("Q")})
    alignment = metadata.get("general.alignment", 32)
    start = (reader.position + alignment - 1) // alignment * alignment
    tensors.sort(key=lambda tensor: tensor["offset"])
    for index, tensor in enumerate(tensors):
        end = tensors[index + 1]["offset"] if index + 1 < len(tensors) else file_size - start
        tensor["stored_bytes"] = end - tensor["offset"]
        tensor["file_offset"] = start + tensor["offset"]
    return {"version": version, "header_bytes": start, "metadata": metadata, "tensors": tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--include", required=True, help="File glob, e.g. '*Q2_K-*.gguf'")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit SHA")
    entries = [entry for entry in list_files(args.repository, args.revision)
               if fnmatch.fnmatch(entry["path"], args.include)]
    if not entries:
        raise ValueError("No matching files")
    report = {"repository": args.repository, "revision": args.revision,
              "limitations": "Stored byte ranges include alignment padding; excludes runtime/cache/graph/staging allocations.",
              "shards": [], "layers": {}, "type_bytes": {}, "root_bytes": 0, "total_file_bytes": 0}
    layers = collections.defaultdict(collections.Counter)
    types = collections.Counter()
    for entry in sorted(entries, key=lambda item: item["path"]):
        url = f"https://huggingface.co/{args.repository}/resolve/{args.revision}/" + urllib.parse.quote(entry["path"])
        limit = 8 * 1024**2
        while True:
            try:
                header = inspect(read_header(url, entry["size"], limit), entry["size"])
                break
            except (EOFError, struct.error):
                if limit >= min(entry["size"], 64 * 1024**2):
                    raise
                limit *= 2
        shard = {"path": entry["path"], "url": url, "bytes": entry["size"],
                 "sha256": entry.get("lfs", {}).get("oid"), **header}
        report["shards"].append(shard)
        report["total_file_bytes"] += entry["size"]
        for tensor in header["tensors"]:
            name, count = tensor["name"], tensor["stored_bytes"]
            types[tensor["type"]] += count
            match = re.match(r"blk\.(\d+)\.", name)
            if match:
                category = "engram" if ".engram_embd." in name else "experts" if "_exps." in name else "other"
                layers[int(match.group(1))][category] += count
            else:
                report["root_bytes"] += count
        print(entry["path"], entry["size"], len(header["tensors"]), flush=True)
        report["layers"] = dict(sorted(layers.items()))
        report["type_bytes"] = dict(types)
        report["total_tensor_bytes"] = sum(types.values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
