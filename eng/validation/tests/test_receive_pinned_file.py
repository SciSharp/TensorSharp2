import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "receive-pinned-file.py"
SPEC = importlib.util.spec_from_file_location("receive_pinned_file", SCRIPT)
RECEIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIVER)


class PinnedFileReceiveTests(unittest.TestCase):
    def test_appends_only_missing_suffix_after_interrupted_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            part.write_bytes(b"abcde")
            RECEIVER.receive(io.BytesIO(b"defgh"), part, 3, 5, hashlib.sha256(b"defgh").hexdigest())
            self.assertEqual(part.read_bytes(), b"abcdefgh")

    def test_truncated_source_keeps_prefix_and_received_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            part.write_bytes(b"abc")
            with self.assertRaisesRegex(OSError, "ended before"):
                RECEIVER.receive(io.BytesIO(b"de"), part, 3, 5, hashlib.sha256(b"defgh").hexdigest())
            self.assertEqual(part.read_bytes(), b"abcde")

    def test_wrong_source_hash_refuses_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            part.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                RECEIVER.receive(io.BytesIO(b"wrong"), part, 3, 5, hashlib.sha256(b"defgh").hexdigest())
            self.assertEqual(part.read_bytes()[:3], b"abc")

    def test_unexpected_prefix_size_refuses_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            part.write_bytes(b"ab")
            source = io.BytesIO(b"defgh")
            with self.assertRaisesRegex(ValueError, "partial size"):
                RECEIVER.receive(source, part, 3, 5, hashlib.sha256(b"defgh").hexdigest())
            self.assertEqual(source.tell(), 0)
            self.assertEqual(part.read_bytes(), b"ab")

    def test_valid_tail_cannot_publish_a_corrupt_full_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "weights.gguf"
            part = output.with_suffix(".gguf.part")
            part.write_bytes(b"xyz")
            plan, report = directory / "plan.json", directory / "report.json"
            plan.write_text(json.dumps({"output": str(output), "prefix_bytes": 3, "source_bytes": 5,
                "source_sha256": hashlib.sha256(b"defgh").hexdigest(), "full_bytes": 8,
                "full_sha256": hashlib.sha256(b"abcdefgh").hexdigest(), "known_hosts": "unused",
                "temporary_key": "unused", "port": 22, "host": "example.invalid"}))
            process = mock.Mock(stdout=io.BytesIO(b"defgh"))
            process.wait.return_value = process.poll.return_value = 0
            with mock.patch.object(sys, "argv", [str(SCRIPT), str(plan), "--report", str(report)]), \
                    mock.patch.object(RECEIVER.subprocess, "Popen", return_value=process), \
                    mock.patch.object(RECEIVER.threading, "Timer"), \
                    self.assertRaisesRegex(ValueError, "Full file SHA256 differs"):
                RECEIVER.main()
            self.assertFalse(output.exists())
            self.assertEqual(part.read_bytes(), b"xyzdefgh")
            self.assertFalse(json.loads(report.read_text())["file_verified"])


if __name__ == "__main__":
    unittest.main()
