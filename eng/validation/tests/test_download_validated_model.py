#!/usr/bin/env python3
"""Local recovery tests for completed and invalid partial model downloads."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "download-validated-model.py"
SPEC = importlib.util.spec_from_file_location("download_validated_model", SCRIPT)
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


class PartialDownloadRecoveryTests(unittest.TestCase):
    def run_partial(self, payload, expected, error=None, transport="curl"):
        with tempfile.TemporaryDirectory(prefix="tensorsharp-download-test-") as temporary:
            directory = Path(temporary).resolve()
            self.assertEqual(directory.parent, Path(tempfile.gettempdir()).resolve())
            partial = directory / "weights.gguf.part"
            final = directory / "weights.gguf"
            partial.write_bytes(payload)
            manifest = directory / "manifest.json"
            report = directory / "report.json"
            expected_hash = hashlib.sha256(expected).hexdigest()
            manifest.write_text(json.dumps({"shards": [{
                "url": "https://example.invalid/weights.gguf",
                "bytes": len(expected), "sha256": expected_hash,
            }]}), encoding="utf-8")
            argv = [str(SCRIPT), str(manifest), "--directory", str(directory),
                    "--report", str(report), "--transport", transport]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(DOWNLOADER.subprocess, "run") as curl, \
                    mock.patch.object(DOWNLOADER.urllib.request, "urlopen") as urlopen, \
                    contextlib.redirect_stdout(io.StringIO()):
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        DOWNLOADER.main()
                else:
                    DOWNLOADER.main()
                curl.assert_not_called()
                urlopen.assert_not_called()
            result = json.loads(report.read_text(encoding="utf-8"))
            if error:
                self.assertEqual(result["status"], "failed")
                self.assertFalse(final.exists(), "An unverified file must not be published")
                self.assertEqual(partial.read_bytes(), payload)
                self.assertEqual(result["shards"], [])
            else:
                self.assertEqual(result["status"], "verified")
                self.assertEqual(final.read_bytes(), expected)
                self.assertFalse(partial.exists())
                self.assertTrue(result["shards"][0]["verified"])
                self.assertEqual(result["shards"][0]["actual_sha256"], expected_hash)

    def test_completed_verified_partial_bypasses_curl_and_publishes(self):
        for transport in ("curl", "buffered"):
            with self.subTest(transport=transport):
                self.run_partial(b"complete validated payload", b"complete validated payload", transport=transport)

    def test_completed_bad_hash_stays_unpublished(self):
        for transport in ("curl", "buffered"):
            with self.subTest(transport=transport):
                self.run_partial(b"bad data", b"gooddata", "SHA256 differs", transport=transport)

    def test_oversized_partial_refuses_without_network(self):
        for transport in ("curl", "buffered"):
            with self.subTest(transport=transport):
                self.run_partial(b"oversized payload", b"small", "Partial file exceeds expected size", transport=transport)


class BufferedTransportTests(unittest.TestCase):
    def response(self, status, headers, reads):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = status
        response.headers = headers
        response.read.side_effect = reads
        return response

    def test_rejected_full_response_preserves_existing_prefix(self):
        response = self.response(200, {"Content-Length": "8"}, [])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            part.write_bytes(b"abc")
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response), \
                    self.assertRaisesRegex(ValueError, "exact resumable byte range"):
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO())
            self.assertEqual(part.read_bytes(), b"abc")
            response.read.assert_not_called()

    def test_mismatched_range_preserves_existing_prefix(self):
        for content_range in ("bytes 2-7/8", "bytes 3-7/9", "bytes 3-6/8"):
            with self.subTest(content_range=content_range), tempfile.TemporaryDirectory() as temporary:
                response = self.response(206, {"Content-Range": content_range}, [])
                part = Path(temporary) / "weights.part"
                part.write_bytes(b"abc")
                with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response), \
                        self.assertRaisesRegex(ValueError, "exact resumable byte range"):
                    DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO())
                self.assertEqual(part.read_bytes(), b"abc")
                response.read.assert_not_called()

    def test_interrupted_response_resumes_without_duplicate_bytes(self):
        first = self.response(200, {"Content-Length": "8"}, [b"abc", OSError("interrupted")])
        second = self.response(206, {"Content-Range": "bytes 3-7/8", "Content-Length": "5"}, [b"defgh", b""])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", side_effect=[first, second]) as urlopen, \
                    mock.patch.object(DOWNLOADER.time, "sleep"):
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO())
            self.assertEqual(part.read_bytes(), b"abcdefgh")
            requests = [call.args[0] for call in urlopen.call_args_list]
            self.assertIsNone(requests[0].get_header("Range"))
            self.assertEqual(requests[1].get_header("Range"), "bytes=3-")

    def test_retry_exhaustion_preserves_received_prefix(self):
        response = self.response(200, {"Content-Length": "8"}, [b"abc", OSError("interrupted")])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response) as urlopen, \
                    self.assertRaisesRegex(OSError, "interrupted"):
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO(), retries=0)
            self.assertEqual(part.read_bytes(), b"abc")
            self.assertEqual(urlopen.call_count, 1)

    def test_staged_tail_resume_uses_absolute_source_offsets(self):
        response = self.response(206, {"Content-Range": "bytes 6-7/8", "Content-Length": "2"}, [b"gh", b""])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.tail.part"
            part.write_bytes(b"def")
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response) as urlopen:
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 5, io.StringIO(), source_offset=3)
            self.assertEqual(part.read_bytes(), b"defgh")
            self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=6-")

    def test_buffered_download_is_hashed_before_publish(self):
        for payload, expected_status in ((b"abcdefgh", "verified"), (b"bad-data", "failed")):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest, report = directory / "manifest.json", directory / "report.json"
                manifest.write_text(json.dumps({"shards": [{"url": "https://example.invalid/weights.gguf",
                    "bytes": 8, "sha256": hashlib.sha256(b"abcdefgh").hexdigest()}]}))
                response = self.response(200, {"Content-Length": "8"}, [payload, b""])
                argv = [str(SCRIPT), str(manifest), "--directory", str(directory), "--report", str(report),
                        "--transport", "buffered"]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if expected_status == "failed":
                        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                            DOWNLOADER.main()
                    else:
                        DOWNLOADER.main()
                self.assertEqual(json.loads(report.read_text())["status"], expected_status)
                self.assertEqual((directory / "weights.gguf").exists(), expected_status == "verified")


if __name__ == "__main__":
    unittest.main()
