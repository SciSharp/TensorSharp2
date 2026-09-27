#!/usr/bin/env python3
"""Transport harness tests against an in-process HTTP fixture, never a model server."""
import contextlib
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("multiple_upload_benchmark", Path(__file__).parents[1] / "multiple-upload-benchmark.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class FixtureServer(BaseHTTPRequestHandler):
    stored = {}
    uploads = []
    reverse = False

    def log_message(self, *_):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        message = BytesParser(policy=default).parsebytes(
            ("Content-Type: " + self.headers["Content-Type"] + "\r\n\r\n").encode() + raw)
        items = []
        for part in message.iter_parts():
            reference = f"stored-{len(self.stored)}.csv"
            self.stored[reference] = part.get_payload(decode=True)
            items.append({"ok": True, "file": reference, "fileName": part.get_filename(), "url": "/uploads/" + reference})
        self.uploads.append(items)
        if self.reverse:
            items.reverse()
        body = items[0] if len(items) == 1 else {"ok": True, "files": items}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = self.stored[self.path.rsplit("/", 1)[-1]]
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class MultipleUploadBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureServer)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def setUp(self):
        FixtureServer.stored = {}
        FixtureServer.uploads = []
        FixtureServer.reverse = False
        self.args = SimpleNamespace(upload_url=f"http://127.0.0.1:{self.server.server_port}/api/upload", api_key=None, timeout=5)

    def test_serial_and_batch_have_correct_bytes_order_and_roundtrip_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = BENCH.make_files(Path(directory), 3, 65536, same_basename=True)
            self.assertEqual(3, len({path.read_bytes() for path in paths}))
            for mode, trips in (("serial", 3), ("batch", 1)):
                with self.subTest(mode=mode):
                    row = BENCH.run_once(self.args, paths, mode)
                    self.assertEqual("passed", row["status"], row["errors"])
                    self.assertEqual(trips, row["upload_roundtrips"])
                    self.assertEqual(3, row["download_roundtrips"])
                    self.assertEqual(3 * 65536, row["payload_bytes"])
                    self.assertEqual(3, len(row["verified"]))
                    self.assertEqual(sum(result["elapsed_ms"] for result in row["uploads"]), row["upload_ms"])

    def test_same_basename_swapped_content_is_rejected_by_download_hashes(self):
        FixtureServer.reverse = True
        with tempfile.TemporaryDirectory() as directory:
            paths = BENCH.make_files(Path(directory), 2, 128, same_basename=True)
            row = BENCH.run_once(self.args, paths, "batch")
        self.assertEqual("failed", row["status"])
        self.assertEqual(2, len(row["errors"]))
        self.assertTrue(all("Downloaded content differs" in error for error in row["errors"]))

    def test_partial_response_is_failure(self):
        with self.assertRaisesRegex(ValueError, "dropped or added"):
            BENCH.check_response({"status": 200, "body": {"ok": True, "files": []}}, [Path("a.csv")])

    def test_stored_urls_cannot_send_authentication_to_another_server(self):
        with self.assertRaisesRegex(ValueError, "outside the benchmark server"):
            BENCH.check_download(self.args, {"url": "http://outside.invalid/a.csv", "file": "a.csv"}, Path("a.csv"))

    def test_summaries_exclude_failed_latency_and_report_actual_roundtrips(self):
        base = {"payload_bytes": 1024 ** 2, "upload_roundtrips": 2, "download_roundtrips": 2}
        rows = [{**base, "status": "passed", "upload_ms": 1000},
                {**base, "status": "failed", "upload_ms": 1000000}]
        summary = BENCH.summarize(rows)
        self.assertEqual(1, summary["failed"])
        self.assertEqual(1000, summary["upload_latency_ms"]["p95"])
        self.assertEqual(1, summary["payload_mib_per_upload_second"])
        self.assertEqual(4, summary["upload_roundtrips"])

    def test_paired_order_warmups_and_failures_are_recorded(self):
        order = []

        def run_once(_args, paths, mode):
            order.append(mode)
            return {"mode": mode, "status": "failed" if len(order) == 1 else "passed", "payload_bytes": 128,
                    "file_count": 1, "upload_ms": 2 if mode == "serial" else 1,
                    "upload_roundtrips": 1, "download_roundtrips": 1, "errors": ["warmup failed"] if len(order) == 1 else []}

        artifact_root = BENCH.ROOT / "artifacts/multiple-files"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory, patch.object(BENCH, "run_once", run_once), contextlib.redirect_stdout(io.StringIO()):
            status = BENCH.main(["--endpoint", "http://fixture.invalid", "--counts", "1", "--sizes", "128",
                                 "--warmup", "2", "--repeats", "2", "--output", directory])
            report = json.loads((Path(directory) / "summary.json").read_text())
            rows = [json.loads(line) for line in (Path(directory) / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(["serial", "batch", "batch", "serial"] * 2, order)
        self.assertEqual(1, status)
        self.assertEqual("warmup", report["failures"][0]["phase"])
        self.assertEqual(2, report["groups"][0]["modes"]["serial"]["samples"])
        self.assertEqual(2, report["groups"][0]["validated_pairs"])
        self.assertEqual(8, len(rows))

    def test_defaults_and_output_confinement(self):
        args = BENCH.parse_args(["--endpoint", "http://localhost:5188"])
        self.assertEqual([1, 2, 8], args.counts)
        self.assertEqual([65536, 1048576], args.sizes)
        self.assertEqual((10, 2), (args.repeats, args.warmup))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "ignored artifacts"):
            BENCH.UPLOAD.output_directory(directory)


if __name__ == "__main__":
    unittest.main()
