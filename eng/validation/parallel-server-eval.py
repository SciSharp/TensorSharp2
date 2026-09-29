#!/usr/bin/env python3
"""Measure an existing TensorSharp server and retain every response as evidence.

Use the same model, seed, prompt suite and options for every placement. These
small deterministic probes detect output regressions; they do not qualify a
model's general quality. SSH tunnels and tiny models are not scaling benchmarks.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.error
import urllib.request


CASES = [
    ("capital", "Reply with only the capital of France.", "paris"),
    ("arithmetic", "What is 17 + 25? Reply with only the number.", "42"),
    ("opposite", "What is the opposite of hot? Reply with one word.", "cold"),
    ("extraction", "The secret code is ORCHID-742. Reply with only the secret code.", "orchid-742"),
    ("sorting", "Sort these numbers from smallest to largest: 9, 2, 5. Reply with only the sorted numbers.", "2"),
    ("language", "Translate the English word 'hello' into Spanish. Reply with one word.", "hola"),
    ("reasoning", "Anna is older than Ben. Ben is older than Clara. Who is the youngest? Reply with only the name.", "clara"),
    ("retrieval", "Remember this fact: the blue box contains a silver key. " + "The other boxes are empty. " * 24 + "What is inside the blue box? Reply briefly.", "silver key"),
]


def simple_check(name, content, expected):
    if name == "sorting":
        return [int(x) for x in re.findall(r"-?\d+", content)] == [2, 5, 9]
    if name == "arithmetic":
        return [int(x) for x in re.findall(r"-?\d+", content)] == [42]
    if name == "retrieval":
        return re.search(r"\bsilver\s+key\b", content.casefold()) is not None
    return content.casefold().strip(" \t\r\n.!\"'`*") == expected


def completed_answer(payload):
    return (payload.get("done") is True and payload.get("done_reason") == "stop"
            and bool(payload.get("message", {}).get("content", "").strip())
            and payload.get("eval_count", 0) > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:5100")
    p.add_argument("--model", default="default")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    p.add_argument("--cases", help="Comma-separated subset of the named deterministic probes")
    a = p.parse_args()
    if min(a.repeats, a.concurrency, a.max_tokens) < 1:
        p.error("repeats, concurrency and max-tokens must be positive")
    selected = a.cases.split(',') if a.cases else [case[0] for case in CASES]
    if len(selected) != len(set(selected)) or set(selected) - {case[0] for case in CASES}:
        p.error("cases must contain unique known probe names")
    cases = [case for case in CASES if case[0] in selected]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def request(case, repeat):
        name, prompt, expected = case
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "think": False, "multi_agent": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": a.max_tokens}}
        if a.reasoning_effort:
            body["reasoning_effort"] = a.reasoning_effort
        req = urllib.request.Request(a.url.rstrip("/") + "/api/chat/ollama",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error
        elapsed = time.perf_counter() - start
        content = payload.get("message", {}).get("content", "")
        count = payload.get("eval_count", 0)
        decode_ns = payload.get("eval_duration", 0)
        prefill_ns = payload.get("prompt_eval_duration", 0)
        return {"case": name, "repeat": repeat, "request": body, "response": payload,
                "wall_seconds": elapsed, "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "completed_answer": completed_answer(payload),
                "semantic_check_passed": simple_check(name, content, expected),
                "simple_check_passed": completed_answer(payload) and simple_check(name, content, expected),
                "decode_tokens_per_second": count * 1e9 / decode_ns if decode_ns else None,
                "prefill_tokens_per_second": payload.get("prompt_eval_count", 0) * 1e9 / prefill_ns if prefill_ns else None}

    report = {"status": "running", "configuration": vars(a) | {"output": str(a.output), "baseline": str(a.baseline) if a.baseline else None},
              "selected_cases": [case[0] for case in cases],
              "limitations": ["Selected simple deterministic probes, not a general model quality evaluation.",
                              "Server timings include its actual scheduler/cache behavior; record placement and transport separately.",
                              ("First repetition is warmup and excluded from steady-state median." if a.repeats > 1
                               else "One repetition only; no warmup sample was excluded.")], "rows": rows}

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for repeat in range(a.repeats):
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                for row in pool.map(lambda case: request(case, repeat), cases):
                    rows.append(row)
                    save()
                    print(row["case"], repeat, row["simple_check_passed"], round(row["wall_seconds"], 3), flush=True)
        steady = [r for r in rows if r["repeat"] > 0] or rows
        report["summary"] = {"requests": len(rows), "simple_checks_passed": sum(r["simple_check_passed"] for r in rows),
                             "median_wall_seconds": statistics.median(r["wall_seconds"] for r in steady)}
        for field in ("decode_tokens_per_second", "prefill_tokens_per_second"):
            values = [r[field] for r in steady if r[field] is not None]
            report["summary"]["median_" + field] = statistics.median(values) if values else None
        if a.baseline:
            previous = json.loads(a.baseline.read_text())
            lookup = {(r["case"], r["repeat"]): r for r in previous["rows"]}
            compared = [r for r in rows if (r["case"], r["repeat"]) in lookup]
            report["baseline_comparison"] = {"compared": len(compared),
                "identical_content": sum(r["content_sha256"] == lookup[r["case"], r["repeat"]]["content_sha256"] for r in compared),
                "differences": [{"case": r["case"], "repeat": r["repeat"], "actual": r["content"],
                                 "baseline": lookup[r["case"], r["repeat"]]["content"]} for r in compared
                                if r["content_sha256"] != lookup[r["case"], r["repeat"]]["content_sha256"]]}
        report["status"] = "completed" if all(r["simple_check_passed"] for r in rows) else "failed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
