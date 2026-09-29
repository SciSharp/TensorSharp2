#!/usr/bin/env python3
"""Compare TensorSharp TS_DUMP_LOGITS F32 vectors without requiring numpy."""
import argparse
from array import array
import json
import math
from pathlib import Path
import sys


def read(path):
    value = array("f")
    value.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        value.byteswap()
    if not value or not all(math.isfinite(x) for x in value):
        raise ValueError(f"Empty or nonfinite logits: {path}")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    reference, candidate = read(a.reference), read(a.candidate)
    if len(reference) != len(candidate):
        raise ValueError("Logit vector sizes differ")
    delta = [x - y for x, y in zip(reference, candidate)]
    dot = sum(x * y for x, y in zip(reference, candidate))
    norm = math.sqrt(sum(x * x for x in reference) * sum(y * y for y in candidate))
    ref_max = max(range(len(reference)), key=reference.__getitem__)
    cand_max = max(range(len(candidate)), key=candidate.__getitem__)
    report = {"reference": str(a.reference), "candidate": str(a.candidate), "count": len(reference),
              "all_finite": True, "max_absolute_error": max(abs(x) for x in delta),
              "rmse": math.sqrt(sum(x * x for x in delta) / len(delta)), "cosine_similarity": dot / norm,
              "reference_argmax": ref_max, "candidate_argmax": cand_max, "same_argmax": ref_max == cand_max,
              "limitations": "One prefill vector on fixed input, not a full model quality qualification."}
    encoded = json.dumps(report, indent=2) + "\n"
    if a.output:
        a.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
