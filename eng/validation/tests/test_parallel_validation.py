"""Regressions for failures that successful response rows used to conceal."""
import importlib.util
from pathlib import Path
import signal
import unittest


spec = importlib.util.spec_from_file_location("parallel_case", Path(__file__).parents[1] / "run-parallel-case.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
eval_spec = importlib.util.spec_from_file_location("parallel_eval", Path(__file__).parents[1] / "parallel-server-eval.py")
evaluator = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(evaluator)


class ShutdownEvidenceTests(unittest.TestCase):
    def test_completed_responses_do_not_hide_native_disposal_abort(self):
        shutdown = dict(started=True, requested=True, forced=False, exit_code=-signal.SIGABRT)
        failures = runner.shutdown_failures(shutdown,
            "All requests completed\nggml-backend.cpp:426: GGML_ASSERT(backend) failed\n")
        self.assertEqual(2, len(failures))
        self.assertIn("GGML_ASSERT", failures[1])

    def test_native_assertion_is_preserved_even_if_a_parent_masks_the_exit(self):
        shutdown = dict(started=True, requested=True, forced=False, exit_code=0)
        self.assertTrue(runner.shutdown_failures(shutdown, "GGML_ASSERT(backend) failed"))

    def test_server_must_finish_disposal_but_worker_can_be_terminated(self):
        shutdown = dict(started=True, requested=True, forced=False, exit_code=-signal.SIGTERM)
        self.assertTrue(runner.shutdown_failures(shutdown, ""))
        self.assertEqual([], runner.shutdown_failures(shutdown, "", allow_sigterm=True))

    def test_timeout_and_unrequested_crash_are_failures(self):
        for shutdown in (
            dict(started=True, requested=True, forced=True, exit_code=-9),
            dict(started=True, requested=False, forced=False, exit_code=-signal.SIGTERM),
        ):
            with self.subTest(shutdown=shutdown):
                self.assertTrue(runner.shutdown_failures(shutdown, "", allow_sigterm=True))

    def test_successful_shutdown_and_unstarted_process_are_distinct_valid_states(self):
        for shutdown in (dict(started=False),
                         dict(started=True, requested=True, forced=False, exit_code=0)):
            self.assertEqual([], runner.shutdown_failures(shutdown, "clean shutdown"))


class CompletedAnswerTests(unittest.TestCase):
    def test_correct_but_truncated_answer_is_not_a_quality_pass(self):
        payload = dict(done=True, done_reason="length", eval_count=1, message={"content": "42"})
        self.assertTrue(evaluator.simple_check("arithmetic", "42", "42"))
        self.assertFalse(evaluator.completed_answer(payload))
        payload["done_reason"] = "stop"
        self.assertTrue(evaluator.completed_answer(payload))

    def test_empty_or_unreported_completion_cannot_pass(self):
        for payload in ({}, dict(done=True, done_reason="stop", eval_count=2, message={"content": ""}),
                        dict(done=True, done_reason="stop", message={"content": "42"})):
            with self.subTest(payload=payload):
                self.assertFalse(evaluator.completed_answer(payload))


if __name__ == "__main__":
    unittest.main()
