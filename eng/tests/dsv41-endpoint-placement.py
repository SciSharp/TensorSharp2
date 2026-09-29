#!/usr/bin/env python3
"""Exercise real two-GPU endpoint-only DeepSeek V4.1 placement.

Requires deterministic eng/dsv41-fixture.py --f32 --cuda-index weights, its
DSpark fixture, and native test hooks. The budget cap uses the real packer.
Checks numerical parity with one GPU plus independent PyTorch text logits.
Synthetic weights establish graph correctness, not trained quality or speed.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--vision", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.fixture_dir / "deepseek41.config.json").read_text())
    if not metadata.get("fixture"):
        raise ValueError("Only deterministic synthetic weights are permitted")
    config = metadata["config"]["text_config"]
    vocab, dim, layers = config["vocab_size"], config["hidden_size"], config["num_hidden_layers"]
    lib = ct.CDLL(str(args.library.resolve()))
    ptr, integer = ct.c_void_p, ct.c_int
    signatures = {
        "Dsv4LoadModel": ([ct.c_char_p] + [integer] * 5 + [ct.c_char_p], ptr),
        "Dsv4LoadModelDspark": ([ct.c_char_p] + [integer] * 4 + [ct.c_char_p, integer, ct.c_char_p], ptr),
        "Dsv4Free": ([ptr], None), "Dsv4ResetChecked": ([ptr], integer),
        "Dsv4Forward": ([ptr, ptr, integer, ptr], integer),
        "Dsv4ForwardSpec": ([ptr, ptr, integer, ptr], integer),
        "Dsv4DsparkDraft": ([ptr, integer, ptr, ptr], integer),
        "Dsv4Rewind": ([ptr, integer], integer),
        "Dsv4NPast": ([ptr], integer), "Dsv4SlotAlloc": ([ptr], integer),
        "Dsv4SetActiveSlot": ([ptr, integer], integer),
        "Dsv4ForwardBatchedDecode": ([ptr, integer, ptr, ptr, ptr, ptr], integer),
        "Dsv4TestLayerDevice": ([ptr, integer], integer),
        "Dsv4TestEndpointPlacement": ([ptr, ptr, integer], integer),
        "Dsv41VisionLoad": ([ct.c_char_p, ct.c_char_p, integer, integer], ptr),
        "Dsv41VisionFree": ([ptr], None), "Dsv41AttachVision": ([ptr, ptr], integer),
        "Dsv41ForwardVision": ([ptr, ptr, ptr, ptr, integer, integer, ptr], integer),
    }
    api = {}
    for name, (parameters, result) in signatures.items():
        fn = getattr(lib, "TSGgml_" + name)
        fn.argtypes, fn.restype = parameters, result
        api[name] = fn
    settings = {"TS_DSV4_FA": "0", "TS_DSV4_GATHER": "0", "TS_DSV41_ENGRAM_WARM": "0",
                "TS_DSV41_ENGRAM_THREADS": "2", "TS_DSV41_REWIND_CHECKPOINT": "1",
                "TS_DSV41_RETAINED_CACHE": "0", "TS_DSV41_TP": "0"}
    original = {key: os.environ.get(key) for key in (*settings, "TS_DSV4_TEST_ENDPOINT_LAYOUT")}
    checks, runs = [], []
    report = dict(scope="Synthetic endpoint graph correctness; no trained performance claim",
                  image_injection_excluded=args.vision is None, vision_encoder_excluded=True, v4_graph_excluded=True, checks=checks, runs=runs,
                  native_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    def check(name, ok, **details):
        checks.append(dict(name=name, passed=bool(ok), **details))
        save()
        if not ok:
            raise AssertionError(name)
    def compare(name, actual, expected):
        exact = np.issubdtype(actual.dtype, np.integer)
        check(name, np.array_equal(actual, expected) if exact else np.allclose(actual, expected, atol=2e-5, rtol=2e-5),
              exact_integer_comparison=exact, max_abs=float(np.max(np.abs(actual - expected))))
    def forward(handle, ids, spec=False):
        ids = np.ascontiguousarray(ids, np.int32)
        result = np.empty((len(ids) if spec else 1, vocab), np.float32)
        status = api["Dsv4ForwardSpec" if spec else "Dsv4Forward"](handle, ids.ctypes.data, len(ids), result.ctypes.data)
        if status != (1 if spec else 0):
            raise AssertionError(f"forward status={status}")
        return result
    def draft(handle):
        tokens, confidence = np.empty(5, np.int32), np.empty(5, np.float32)
        position = api["Dsv4NPast"](handle)
        if api["Dsv4DsparkDraft"](handle, 37, tokens.ctypes.data, confidence.ctypes.data) != 5:
            raise AssertionError("follow-up draft did not fill its block")
        if api["Dsv4NPast"](handle) != position:
            raise AssertionError("follow-up draft mutated trunk position")
        return tokens, confidence
    ids = np.array([0, 15, 32, 64, 128], np.int32)
    spec = importlib.util.spec_from_file_location("reference", Path(__file__).parents[1] / "dsv41-reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    weights = reference.GgufWeights(args.fixture_dir / "deepseek41-fixture.gguf")
    oracle = reference.Reference(weights, metadata["config"], reference.load_engram(weights), "model")
    expected = oracle.forward(ids.tolist()).cpu().numpy()[-1:]
    baseline = {}
    try:
        os.environ.update(settings)
        # Plain and DSpark first establish one-GPU numerical baselines, then
        # execute the identical operations with either endpoint lacking trunk.
        for head in (False, True):
            for layout, tp, offload in ((None, 0, 0), ("empty-first", 0, 0), ("empty-last", 0, 0),
                                        ("empty-first", 2, 0), ("empty-last", 2, 0), ("empty-last", 2, 1)):
                if layout is None:
                    os.environ.pop("TS_DSV4_TEST_ENDPOINT_LAYOUT", None)
                else:
                    os.environ["TS_DSV4_TEST_ENDPOINT_LAYOUT"] = layout
                os.environ["TS_DSV41_TP"] = str(tp)
                gpus = 1 if layout is None else 2
                label = f"{'dspark' if head else 'plain'}_{layout or 'single'}_tp{tp}_cpu{offload}"
                path = str(args.fixture_dir / "deepseek41-fixture.gguf").encode()
                handle = (api["Dsv4LoadModelDspark"](path, gpus, 256, 32, 2, str(args.head).encode(), offload, b"CUDA")
                          if head else api["Dsv4LoadModel"](path, gpus, 256, 32, 2, offload, b"CUDA"))
                check(label + "_load", bool(handle))
                vision = None
                run = dict(name=label)
                runs.append(run)
                outputs = {}
                try:
                    devices = [api["Dsv4TestLayerDevice"](handle, i) for i in range(layers)]
                    run["trunk_devices"] = devices
                    if layout:
                        check(label + "_empty_endpoint", devices == [1 if layout == "empty-first" else 0] * layers)
                    def placement(suffix):
                        description = ct.create_string_buffer(160)
                        result = api["Dsv4TestEndpointPlacement"](handle, description, len(description))
                        check(label + "_placement_" + suffix, result == 1, detail=description.value.decode())
                    outputs["prefill"] = forward(handle, ids)
                    compare(label + "_independent_oracle", outputs["prefill"], expected)
                    placement("prefill")
                    outputs["decode"] = forward(handle, [13])
                    placement("decode")
                    if head:
                        tokens, confidence = np.empty(5, np.int32), np.empty(5, np.float32)
                        check(label + "_draft", api["Dsv4DsparkDraft"](handle, 17, tokens.ctypes.data, confidence.ctypes.data) == 5)
                        outputs["draft_tokens"], outputs["draft_confidence"] = tokens, confidence
                        outputs["verify"] = forward(handle, [17, 19, 23], spec=True)
                        check(label + "_verify_position", api["Dsv4NPast"](handle) == 9)
                        placement("verify")
                        check(label + "_rewind", api["Dsv4Rewind"](handle, 7) == 1)
                        outputs["rejected_tail_continuation"] = forward(handle, [19, 31])
                        tail_tokens, tail_confidence = draft(handle)
                        check(label + "_reset_cold", api["Dsv4ResetChecked"](handle) == 1)
                        forward(handle, ids)
                        forward(handle, [13])
                        forward(handle, [17])
                        compare(label + "_rollback_cold_parity", outputs["rejected_tail_continuation"], forward(handle, [19, 31]))
                        cold_tokens, cold_confidence = draft(handle)
                        compare(label + "_rollback_draft_tokens", tail_tokens, cold_tokens)
                        compare(label + "_rollback_draft_confidence", tail_confidence, cold_confidence)
                    else:
                        second = api["Dsv4SlotAlloc"](handle)
                        check(label + "_slot", second > 0)
                        check(label + "_select_second", api["Dsv4SetActiveSlot"](handle, second) == 0)
                        forward(handle, ids)
                        forward(handle, [13])
                        slots = np.array([0, second], np.int32)
                        tokens, positions = np.array([17, 19], np.int32), np.array([6, 6], np.int32)
                        logits = np.empty((2, vocab), np.float32)
                        status = api["Dsv4ForwardBatchedDecode"](handle, 2, slots.ctypes.data, tokens.ctypes.data,
                                                               positions.ctypes.data, logits.ctypes.data)
                        check(label + "_batched", status == 0)
                        outputs["batched"] = logits
                        placement("batched")
                    if args.vision:
                        check(label + "_reset_image", api["Dsv4ResetChecked"](handle) == 1)
                        vision = api["Dsv41VisionLoad"](str(args.vision).encode(), b"CUDA", 0, 2)
                        check(label + "_vision_load", bool(vision))
                        check(label + "_vision_attach", api["Dsv41AttachVision"](handle, vision) == 0)
                        image_ids = np.array([0, 250, 250, 13], np.int32)
                        image_mask = np.array([0, 1, 1, 0], np.uint8)
                        rows = np.random.default_rng(4109).normal(0, .1, (2, dim)).astype(np.float32)
                        logits = np.empty((1, vocab), np.float32)
                        check(label + "_image", api["Dsv41ForwardVision"](handle, image_ids.ctypes.data,
                            image_mask.ctypes.data, rows.ctypes.data, len(image_ids), len(rows), logits.ctypes.data) == 0)
                        outputs["image"] = logits
                        placement("image")
                        check(label + "_reset_text_control", api["Dsv4ResetChecked"](handle) == 1)
                        text_control = forward(handle, image_ids)
                        check(label + "_image_changes_logits", not np.allclose(outputs["image"], text_control,
                            atol=2e-5, rtol=2e-5), max_abs=float(np.max(np.abs(outputs["image"] - text_control))))
                        check(label + "_reset_image_continuation", api["Dsv4ResetChecked"](handle) == 1)
                        check(label + "_image_replay", api["Dsv41ForwardVision"](handle, image_ids.ctypes.data,
                            image_mask.ctypes.data, rows.ctypes.data, len(image_ids), len(rows), logits.ctypes.data) == 0)
                        outputs["image_decode"] = forward(handle, [17])
                        if head:
                            tokens, confidence = np.empty(5, np.int32), np.empty(5, np.float32)
                            check(label + "_image_draft", api["Dsv4DsparkDraft"](handle, 19,
                                tokens.ctypes.data, confidence.ctypes.data) == 5)
                            outputs["image_draft_tokens"], outputs["image_draft_confidence"] = tokens, confidence
                            outputs["image_verify"] = forward(handle, [19, 23, 29], spec=True)
                            check(label + "_image_rewind", api["Dsv4Rewind"](handle, 6) == 1)
                            outputs["image_rejected_tail"] = forward(handle, [31])
                            image_tail_tokens, image_tail_confidence = draft(handle)
                            check(label + "_image_reset_cold", api["Dsv4ResetChecked"](handle) == 1)
                            check(label + "_image_cold", api["Dsv41ForwardVision"](handle, image_ids.ctypes.data,
                                image_mask.ctypes.data, rows.ctypes.data, len(image_ids), len(rows), logits.ctypes.data) == 0)
                            forward(handle, [17])
                            forward(handle, [19])
                            compare(label + "_image_rollback_cold_parity", outputs["image_rejected_tail"], forward(handle, [31]))
                            cold_tokens, cold_confidence = draft(handle)
                            compare(label + "_image_rollback_draft_tokens", image_tail_tokens, cold_tokens)
                            compare(label + "_image_rollback_draft_confidence", image_tail_confidence, cold_confidence)
                    if layout is None:
                        baseline[head] = outputs
                    else:
                        for name, values in outputs.items():
                            compare(label + "_single_parity_" + name, values, baseline[head][name])
                finally:
                    api["Dsv4Free"](handle)
                    if vision:
                        api["Dsv41VisionFree"](vision)
        report["status"] = "passed"
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        save()
    print(f"Passed {len(checks)} endpoint placement/numerical checks")


if __name__ == "__main__":
    main()
