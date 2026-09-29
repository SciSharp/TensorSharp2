#!/usr/bin/env python3
"""Receive one SSH forced-command file or staged suffix, then verify the full file.

The caller provisions and promptly removes a temporary file-specific SSH key.
This receiver never changes authorized_keys and never runs a remote command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def receive(stream, part, prefix_bytes, source_bytes, expected_source_sha256):
    """Keep an existing prefix and resume our own partial suffix without append writes."""
    current = part.stat().st_size if part.exists() else 0
    if not prefix_bytes <= current <= prefix_bytes + source_bytes:
        raise ValueError("Existing partial size does not match the planned prefix/suffix")
    already_received = current - prefix_bytes
    digest = hashlib.sha256()
    received = 0
    with part.open("r+b" if part.exists() else "wb") as output:
        output.seek(current)
        while received < source_bytes:
            block = stream.read(min(8 * 1024**2, source_bytes - received))
            if not block:
                raise OSError("Source stream ended before the expected suffix size")
            digest.update(block)
            skip = min(len(block), max(0, already_received - received))
            output.write(block[skip:])
            received += len(block)
        if stream.read(1):
            raise ValueError("Source stream exceeds the expected suffix size")
    actual = digest.hexdigest()
    if actual != expected_source_sha256:
        raise ValueError("Transferred source SHA256 differs from the staging report")
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    output = Path(plan["output"])
    part = output.with_suffix(output.suffix + ".part")
    prefix = plan.get("prefix_bytes", 0)
    source_bytes = plan["source_bytes"]
    if prefix < 0 or source_bytes <= 0 or prefix + source_bytes != plan["full_bytes"]:
        raise ValueError("Invalid prefix/source/full byte counts")
    if output.exists():
        raise ValueError("Final output already exists; refusing to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
               "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + plan["known_hosts"],
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
               "-i", plan["temporary_key"], "-p", str(plan["port"]), plan["host"]]
    report = {"status": "running", "plan": plan, "started_unix": time.time(), "file_verified": False}
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    process = timer = None
    try:
        with a.report.with_suffix(".ssh.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log)
            timer = threading.Timer(a.timeout, process.kill)
            timer.start()
            report["source_sha256"] = receive(process.stdout, part, prefix, source_bytes, plan["source_sha256"])
            if process.wait(timeout=10) != 0:
                raise RuntimeError("SSH source did not exit successfully")
            timer.cancel()
        report["transfer_seconds"] = time.time() - report["started_unix"]
        if part.stat().st_size != plan["full_bytes"]:
            raise ValueError("Assembled full file has the wrong size")
        report["actual_full_sha256"] = sha(part)
        if report["actual_full_sha256"] != plan["full_sha256"]:
            raise ValueError("Full file SHA256 differs from its immutable source")
        part.rename(output)
        report.update(status="verified", file_verified=True)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        report["wall_seconds"] = time.time() - report["started_unix"]
        a.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
