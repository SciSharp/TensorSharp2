#!/usr/bin/env python3
"""Deterministic image/video HTTP checks for the prepared V4.1 companion.

Prepare with Pillow and ffmpeg, then validate against a running server. Video
uses ordered sampled images and time labels; this does not test an audio model.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import engines
import scenarios as text_scenarios
from validate_inference import SAMPLING, digest, exact_json


CARDS = (("4821", "red"), ("9364", "blue"))


def prepare(directory):
    from PIL import Image, ImageDraw, ImageFont
    directory.mkdir(parents=True, exist_ok=True)
    fonts = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    font = next((ImageFont.truetype(path, 90) for path in fonts if Path(path).exists()),
                ImageFont.load_default(size=90))
    def card(path, code, color):
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        draw.text((320, 125), code, font=font, fill="black", anchor="mm")
        draw.rectangle((100, 245, 540, 405), fill=color)
        image.save(path)
    for index, (code, color) in enumerate(CARDS):
        card(directory / f"card-{index}.png", code, color)
    for index, code in enumerate(("17", "42", "86")):
        card(directory / f"frame-{index}.png", code, ("red", "green", "blue")[index])
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", "1",
                    "-i", str(directory / "frame-%d.png"), "-frames:v", "3", "-c:v", "libx264",
                    "-threads", "1", "-pix_fmt", "yuv420p", "-r", "1",
                    str(directory / "ordered-cards.mp4")], check=True)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in directory.iterdir() if path.suffix in (".png", ".mp4")}
    (directory / "manifest.json").write_text(json.dumps({"sha256": files,
        "cards": CARDS, "video_codes": ["17", "42", "86"], "video_seconds": [0, 1, 2]}, indent=2) + "\n")


def attachment(path, video=False):
    mime = "video/mp4" if video else "image/png"
    url = "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()
    if video:
        return {"type": "video_url", "video_url": {"url": url, "fps": 1, "max_frames": 3}}
    return {"type": "image_url", "image_url": {"url": url}}


def run_case(args, scenario, tag, index):
    selected = index % len(CARDS)
    code, color = CARDS[selected]
    media = [attachment(args.fixtures / f"card-{selected}.png")]
    expected = {"code": code, "color": color}
    prompt = 'Read the four-digit code and the color of the solid rectangle. Return JSON with exactly "code" (a string) and "color" (a lowercase English color).'
    if scenario == "image_long_context":
        prompt = (text_scenarios._sliced_corpus(8192) +
                  "\nThe preceding document is background. Answer using the attached image.\n" + prompt)
    elif scenario == "multi_image":
        other = (selected + 1) % len(CARDS)
        media.append(attachment(args.fixtures / f"card-{other}.png"))
        prompt = 'Read both images in attachment order. Return only JSON with "codes": an array of the two code strings in that order.'
        expected = {"codes": [code, CARDS[other][0]]}
    elif scenario == "image_follow_up":
        prompt = 'Read the four-digit code. Return only JSON with exactly "code", as a string.'
        expected = {"code": code}
    elif scenario.startswith("video_"):
        media = [attachment(args.fixtures / "ordered-cards.mp4", video=True)]
        if scenario == "video_order":
            prompt = 'Read the number on each sampled video frame in time order. Return only JSON with "codes": an array of number strings, one per frame.'
            expected = {"codes": ["17", "42", "86"]}
        else:
            prompt = 'At what time in seconds does the sampled video frame display 42? Use the frame time labels. Return only JSON with "second": the numeric time.'
            expected = {"second": 1}
    messages = [{"role": "user", "content": [{"type": "text", "text": f"[validation {tag}]\n{prompt}"}, *media]}]
    result = {"scenario": scenario, "tag": tag, "status": "fail", "turns": [],
              "expected": expected, "input_sha256": digest(messages)}
    started = time.monotonic()
    try:
        for step in range(2 if scenario == "image_follow_up" else 1):
            request = {"messages": messages, "response_format": {"type": "json_object"},
                       "extra_body": {**SAMPLING, **engines.thinking_body("tensorsharp", False)},
                       "max_tokens": args.max_tokens, "stream": not args.blocking}
            if args.reasoning_effort:
                request["extra_body"]["reasoning_effort"] = args.reasoning_effort
            metrics = engines.run_openai_chat(args.url, args.model, timeout_s=1200, **request)
            result["turns"].append({"request_sha256": digest(request), "metrics": metrics,
                                    "expected": expected})
            if not metrics.get("usage_present") or metrics.get("finish_reason") != "stop":
                raise ValueError("response must include usage and a completed final answer")
            if not exact_json(metrics["assistant_message"].get("content"), expected):
                raise ValueError("image/video answer failed the exact content and JSON check")
            if "second" in expected and isinstance(json.loads(metrics["assistant_message"]["content"])["second"], bool):
                raise ValueError("video timestamp must be a number, not a JSON boolean")
            if scenario == "image_follow_up" and step == 0:
                messages = [*messages, metrics["assistant_message"], {"role": "user", "content":
                    'Now inspect that image again: what color is the rectangle? Return only JSON with "color", as a lowercase English color.'}]
                expected = {"color": color}
        result["status"] = "ok"
    except Exception as error:
        result["detail"] = f"{type(error).__name__}: {error}"
        response = getattr(error, "response", None)
        if response is not None:
            result["error_body"] = response.text[:16384]
    result["wall_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--model")
    parser.add_argument("--weights-id")
    parser.add_argument("--companion-sha256")
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--scenarios", default="image_ocr,multi_image,image_follow_up,video_order,video_timestamp")
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if args.prepare:
        prepare(args.fixtures)
        return 0
    if not all((args.model, args.weights_id, args.companion_sha256, args.profile, args.output)):
        parser.error("validation requires model, weights-id, companion-sha256, profile and output")
    degrees = [int(x) for x in args.concurrency.split(",")]
    if any(n < 1 for n in degrees):
        parser.error("concurrency must be positive")
    scenarios = args.scenarios.split(",")
    if set(scenarios) - {"image_ocr", "image_long_context", "multi_image", "image_follow_up", "video_order", "video_timestamp"}:
        parser.error("unknown media scenario")
    fixture_manifest = json.loads((args.fixtures / "manifest.json").read_text())
    for name, expected_hash in fixture_manifest["sha256"].items():
        if hashlib.sha256((args.fixtures / name).read_bytes()).hexdigest() != expected_hash:
            parser.error(f"fixture SHA256 mismatch: {name}")
    report = {"weights_id": args.weights_id, "companion_sha256": args.companion_sha256,
              "harness_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in (__file__, engines.__file__, text_scenarios.__file__,
                                              Path(__file__).with_name("validate_inference.py"))},
              "profile": args.profile, "model": args.model, "stream": not args.blocking,
              "sampling": SAMPLING, "fixtures": fixture_manifest,
              "reasoning_effort": args.reasoning_effort,
              "execution_plan": {"scenarios": scenarios, "concurrency": degrees,
                                 "expected_cases": len(scenarios) * sum(degrees)},
              "run_complete": False,
              "scope": "Synthetic OCR, color, optional long text/image context, attachment order, image history and sampled video time labels; no audio validation or broad visual-quality parity.",
              "cases": [], "waves": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        for concurrency in degrees:
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                cases = list(pool.map(lambda i: run_case(args, scenario, f"{scenario}-c{concurrency}-i{i}", i), range(concurrency)))
            for case in cases:
                case["concurrency"] = concurrency
            report["cases"].extend(cases)
            passed = sum(case["status"] == "ok" for case in cases)
            report["waves"].append({"scenario": scenario, "concurrency": concurrency,
                                    "passed": passed, "wall_seconds": time.monotonic() - start})
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(f"{scenario} c{concurrency}: {passed}/{concurrency} passed", flush=True)
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return int(any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
