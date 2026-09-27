#!/usr/bin/env python3
"""Summarize sampled memory and page faults beside retained AgentTurnBench rows.

The source CSV files remain authoritative. Fault deltas use samples inside each
row's recorded interval, so they can omit work between the interval boundary
and its nearest sample. GPU sizes retain nvidia-smi's MiB unit.
"""
import argparse
import csv
import json
from pathlib import Path
import re


def csv_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source, skipinitialspace=True))


def summarize(directory):
    processes = csv_rows(directory / "process.csv")
    memory = csv_rows(directory / "memory.csv")
    gpu_peak = {}
    for row in csv_rows(directory / "gpu.csv"):
        index = row["index"].strip()
        value = re.fullmatch(r"(\d+)\s+MiB", row["memory.used [MiB]"].strip())
        if value:
            gpu_peak[index] = max(gpu_peak.get(index, 0), int(value[1]))
    report = {
        "peak_gpu_mib": gpu_peak,
        "max_cgroup_bytes": max(int(row["cgroup_current_bytes"]) for row in memory),
        "max_rss_kib": max(int(row["rss_kib"]) for row in processes),
        "final_major_faults": int(processes[-1]["major_faults"]),
        "scope": "Sampled process-level counters; interval deltas exclude unsampled boundary work.",
        "rows": [],
    }
    for row in json.loads((directory / "rows.json").read_text()):
        start = row["StartedUnixMilliseconds"] / 1000
        end = start + row["TotalMs"] / 1000
        samples = [sample for sample in processes if start <= float(sample["unix_time"]) <= end]
        item = {
            "scenario": row["Scenario"], "label": row["Label"],
            "sample_count": len(samples), "tokens": row["OutTokens"],
            "decode_tps": row["DecodeTps"], "ttft_ms": row["TtftMs"], "finish": row["Finish"],
        }
        if samples:
            first, last = (int(samples[index]["major_faults"]) for index in (0, -1))
            item.update(major_faults_first=first, major_faults_last=last,
                        major_faults_during_row=last - first)
        report["rows"].append(item)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        report = summarize(directory)
        target = directory / "telemetry-summary.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(target)


if __name__ == "__main__":
    main()
