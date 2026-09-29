#!/usr/bin/env python3
"""Exercise the shipped chat page with controlled subagent SSE in a real browser.

Requires the Python playwright package and an installed Chromium browser. Example:
  python eng/validation/validate-webui-agent-activity.py --browser "PATH/TO/chrome"

The fixture serves the unmodified Web UI asset on loopback, mocks model/session
metadata, and controls event delivery so clicks and terminal states are checked
while a request remains open. No model or external service is used. This validates
rendering and SSE handling, not model delegation or real-device performance.
Generated screenshots and reports go under ignored artifacts/ by default.
"""

import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "TensorSharp.Server.Host/wwwroot/index.html"


class Stream:
    def __init__(self):
        self.frames = queue.Queue()

    def send(self, frame):
        self.frames.put(("data: " + json.dumps(frame) + "\n\n").encode())

    def close(self):
        self.frames.put(None)


@contextmanager
def fixture():
    streams = queue.Queue()
    active = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.respond(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path in ("/images/assistant_logo.png", "/images/banner_1.png"):
                self.respond((PAGE.parent / self.path.lstrip("/")).read_bytes(), "image/png")
            elif self.path == "/api/models":
                self.respond(json.dumps({"loaded": "Browser regression fixture", "architecture": "qwen35"}).encode())
            elif self.path == "/api/queue/status":
                self.respond(b'{"processing":0,"pending_requests":0}')
            elif self.path == "/api/skills":
                self.respond(b'{"skills":[]}')
            else:
                self.send_error(404)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/api/sessions":
                self.respond(b'{"sessionId":"browser-regression"}')
                return
            if self.path != "/api/chat":
                self.send_error(404)
                return
            stream = Stream()
            active.append(stream)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            streams.put(stream)
            try:
                while True:
                    frame = stream.frames.get(timeout=30)
                    if frame is None:
                        break
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, queue.Empty):
                pass

        def do_DELETE(self):
            self.respond(b'{}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", streams
    finally:
        for stream in active:
            stream.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def progress(agents=None, seconds=1, tool="wait_agent", phase="running"):
    frame = {"tool_progress": phase, "tool": tool, "seconds": seconds,
             "text": "", "detail": tool}
    if agents is not None:
        frame["agents"] = agents
    return frame


def agent(name, status="running", **values):
    return {"agent_id": "/root/" + name, "parent_id": "/root", "agent_type": "reviewer",
            "task": "Review " + name + " and report supporting evidence.", "status": status,
            "tool": None, "tool_status": None, "detail": None, "result": None,
            "error": None, **values}


def begin(page, streams, name):
    expect(page.locator("#btn-send")).to_have_class("send-btn")
    page.locator("#message-input").fill("Exercise " + name)
    page.locator("#btn-send").click()
    return streams.get(timeout=10)


def finish(stream, error=None):
    if error is None:
        stream.send({"token": "Fixture response completed."})
    stream.send({"done": True, "tokenCount": 4, "elapsed": 0.1, "tokPerSec": 40,
                 "error": error})


def exercise(page, streams, output, checks):
    details = page.locator(".current-agent-details")
    summary = page.locator(".current-agent-summary")
    cards = page.locator(".current-agent-card")
    activity = page.locator(".current-activity")
    stream = begin(page, streams, "subagent progress")
    stream.send(progress([]))
    expect(details).to_be_visible()
    expect(details).not_to_have_attribute("open", "")
    summary.click()
    expect(details).to_have_attribute("open", "")
    checks.append("Empty snapshot remains expandable")

    hostile = '<img src=x onerror="window.agentMarkupExecuted=true"> <script>bad()</script>'
    agents = [
        agent("source_review", task="Inspect source and tests. " + "Evidence needed. " * 40,
              tool="read_file", tool_status="running", detail="Inspecting source line 42. " + hostile),
        agent("completed_review", "completed", result="Confirmed expected behavior. " + hostile),
        agent("failed_review", "failed", error="Synthetic tool failure. " + hostile),
        agent("cancelled_review", "cancelled", error="Cancelled by parent."),
        agent("budget_review", "limit_reached", error="Generation budget exhausted."),
        agent("timed_review", "timed_out", error="Deadline expired."),
    ]
    stream.send(progress(agents, 2))
    expect(cards).to_have_count(len(agents))
    expect(cards.first).to_contain_text("Inspect source and tests.")
    for name in ("source_review", "completed_review", "failed_review", "cancelled_review", "budget_review", "timed_review"):
        expect(cards.filter(has_text="/root/" + name)).to_be_visible()
    for index, state in enumerate(("running", "completed", "failed", "cancelled", "limit reached", "timed out")):
        expect(cards.nth(index).locator(".current-agent-state")).to_contain_text(state)
    expect(cards).to_have_count(len(agents))
    expect(cards.nth(1)).to_contain_text("Confirmed expected behavior.")
    expect(cards.nth(2)).to_contain_text("Synthetic tool failure.")
    expect(details.locator("img, script")).to_have_count(0)
    assert page.evaluate("window.agentMarkupExecuted === undefined")
    expect(details).to_contain_text(hostile)
    checks.append("Concurrent tasks, statuses, tool activity, results and errors render as plain text")

    # Click the left edge, where the disclosure marker is drawn, then its heading.
    summary.click(position={"x": 12, "y": 14})
    expect(details).not_to_have_attribute("open", "")
    expect(cards.first).not_to_be_visible()
    stream.send(progress(agents, 3))
    expect(activity).to_contain_text("3s")
    expect(details).not_to_have_attribute("open", "")
    summary.click()
    expect(cards.first).to_be_visible()
    summary.focus()
    page.keyboard.press("Enter")
    expect(details).not_to_have_attribute("open", "")
    page.keyboard.press("Space")
    expect(details).to_have_attribute("open", "")
    checks.append("Marker and heading clicks, Enter and Space expand/collapse; collapsed state survives updates")

    listing = page.locator(".current-agent-list")
    scroll_top = listing.evaluate("el => { el.scrollTop = 75; return el.scrollTop; }")
    assert scroll_top > 0, "Agent details must provide bounded scrolling"
    updated = [{**row} for row in agents]
    updated[0]["detail"] = "Second source activity update. " + hostile
    stream.send(progress(updated, 4))
    expect(cards.first).to_contain_text("Second source activity update.")
    expect(details).to_have_attribute("open", "")
    expect(summary).to_be_focused()
    assert listing.evaluate("el => el.scrollTop") == scroll_top, "Heartbeat reset the user's scroll position"
    checks.append("Expanded state, keyboard focus and inner scroll survive progress updates")
    listing.evaluate("el => el.scrollTop = 0")
    page.screenshot(path=str(output / "subagents-expanded-desktop.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    summary.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Narrow screen has horizontal page overflow"
    page.screenshot(path=str(output / "subagents-expanded-mobile.png"), full_page=True)
    summary.click()
    expect(details).not_to_have_attribute("open", "")
    summary.click()
    expect(details).to_have_attribute("open", "")
    checks.append("Narrow viewport disclosure remains usable without page overflow")
    page.set_viewport_size({"width": 1100, "height": 850})

    stream.send(progress(phase="finished"))
    expect(activity).to_have_count(0)
    stream.send(progress([agent("followup_review")], 1))
    expect(details).to_be_visible()
    expect(details).to_have_attribute("open", "")
    expect(cards).to_have_count(1)
    expect(cards.first).to_contain_text("followup_review")
    checks.append("Consecutive wait operations preserve disclosure choice and discard stale agents")
    stream.send(progress(phase="finished"))
    expect(activity).to_have_count(0)
    finish(stream)
    stream.close()
    expect(page.locator("#btn-send")).to_have_class("send-btn")
    checks.append("Finished operation removes temporary agent details")

    # Exercise the real request loop's independent terminal paths, keeping each
    # stream alive until the panel is visible so early cleanup cannot fake a pass.
    for ending in ("done", "error", "cancel", "early_close"):
        stream = begin(page, streams, ending)
        stream.send(progress([agent("terminal_review")]))
        expect(details).to_be_visible()
        summary.click()
        expect(cards.first).to_be_visible()
        if ending == "cancel":
            page.locator("#btn-send").click()
            expect(page.locator(".stats").last).to_contain_text("Stopped by user")
        elif ending == "early_close":
            stream.close()
        else:
            finish(stream, "Synthetic inference failure" if ending == "error" else None)
            expect(activity).to_have_count(0)
            if ending == "error":
                expect(page.locator(".message.assistant .bubble-text").last).to_contain_text("Synthetic inference failure")
            stream.close()
        expect(activity).to_have_count(0)
        stream.close()
        expect(page.locator("#btn-send")).to_have_class("send-btn")
        checks.append(ending + " removes temporary agent details through the request loop")

    stream = begin(page, streams, "server without agent snapshots")
    stream.send(progress())
    expect(details).to_be_visible()
    summary.click()
    expect(details).to_contain_text("Sub-agent details are not available from this server.")
    finish(stream)
    stream.close()
    expect(page.locator("#btn-send")).to_have_class("send-btn")
    checks.append("Missing snapshot shows an explicit compatibility message")

    stream = begin(page, streams, "ordinary shell progress")
    stream.send({**progress(tool="shell"), "text": "Ordinary process output"})
    expect(activity).to_contain_text("Ordinary process output")
    expect(details).not_to_be_visible()
    stream.send(progress(tool="shell", phase="finished"))
    expect(activity).to_have_count(0)
    finish(stream)
    stream.close()
    expect(page.locator("#btn-send")).to_have_class("send-btn")
    checks.append("Ordinary tool progress remains visible and transient")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, help="Installed Chromium executable; defaults to Playwright's installed Chromium")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/webui-agent-activity")
    args = parser.parse_args()
    output = args.out.resolve()
    allowed = (ROOT / "artifacts", ROOT / "docs/validation")
    if not any(output.is_relative_to(path.resolve()) for path in allowed):
        parser.error("--out must be inside ignored artifacts/ or docs/validation/")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "asset_sha256": hashlib.sha256(PAGE.read_bytes()).hexdigest(),
              "checks": [], "page_errors": [],
              "limitations": "Controlled SSE and synthetic agents in installed Chromium only; no loaded model, actual delegation, GPU, or other browser engine validation."}
    try:
        with fixture() as (url, streams), sync_playwright() as playwright:
            options = {"headless": True}
            if args.browser:
                options["executable_path"] = str(args.browser.resolve())
            browser = playwright.chromium.launch(**options)
            report["browser_version"] = browser.version
            context = browser.new_context(viewport={"width": 1100, "height": 850})
            # Keep the fixture isolated even if the shipped page adds remote assets.
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url + "/") else route.abort())
            page = context.new_page()
            page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
            page.set_default_timeout(10000)
            try:
                page.goto(url)
                expect(page.locator("#status-badge")).to_contain_text("Browser regression fixture")
                exercise(page, streams, output, report["checks"])
                assert not report["page_errors"], report["page_errors"]
            except Exception:
                page.screenshot(path=str(output / "failure.png"), full_page=True)
                raise
            finally:
                context.close()
                browser.close()
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "report": str(output / "report.json")}))


if __name__ == "__main__":
    main()
