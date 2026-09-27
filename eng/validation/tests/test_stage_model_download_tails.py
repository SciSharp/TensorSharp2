import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "stage-model-download-tails.py"
SPEC = importlib.util.spec_from_file_location("stage_model_download_tails", SCRIPT)
STAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGER)


class TailSourceIdentityTests(unittest.TestCase):
    def test_same_identity_can_resume_without_modifying_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.tail"
            part = path.with_suffix(".tail.part")
            identity = {"url": "https://example.invalid/commit/weights", "bytes": 8, "sha256": "pinned", "source_offset": 3}
            STAGER.bind_source_identity(path, part, identity)
            part.write_bytes(b"def")
            STAGER.bind_source_identity(path, part, identity)
            self.assertEqual(part.read_bytes(), b"def")

    def test_changed_offset_or_source_refuses_existing_prefix(self):
        for field, value in (("source_offset", 4), ("sha256", "different"), ("url", "https://example.invalid/other/weights"), ("bytes", 9)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "weights.tail"
                part = path.with_suffix(".tail.part")
                identity = {"url": "https://example.invalid/commit/weights", "bytes": 8, "sha256": "pinned", "source_offset": 3}
                STAGER.bind_source_identity(path, part, identity)
                part.write_bytes(b"def")
                with self.assertRaisesRegex(ValueError, "different pinned source/offset"):
                    STAGER.bind_source_identity(path, part, identity | {field: value})
                self.assertEqual(part.read_bytes(), b"def")
                self.assertEqual(json.loads(path.with_suffix(".tail.source.json").read_text()), identity)

    def test_unbound_existing_tail_refuses_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.tail"
            part = path.with_suffix(".tail.part")
            part.write_bytes(b"def")
            with self.assertRaisesRegex(ValueError, "no source identity"):
                STAGER.bind_source_identity(path, part, {"source_offset": 3})
            self.assertEqual(part.read_bytes(), b"def")


if __name__ == "__main__":
    unittest.main()
