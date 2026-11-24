"""Regression tests for the grader: python3 -m unittest experiments/lowbit/test_eval_tasks.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_tasks import math_answer, run_code  # noqa: E402

FENCE = "```"


class MathAnswer(unittest.TestCase):
    def test_values(self):
        cases = {"Answer: 42": 42, "Answer: **$1,200**": 1200, "Answer: 18/5": 3.6, "Answer: 18/-5": -3.6,
                 "Answer: 18/1,000": 0.018, "Answer: 3.5.": 3.5, "Answer: 7, which": 7,
                 "x Answer: 7\nAnswer: 2": 2}
        for text, want in cases.items():
            self.assertAlmostEqual(math_answer(text), want, msg=text)

    def test_rejected(self):
        for text in ["Answer: 18/", "Answer: 18/5/2", "Answer: 2\nAnswer: unknown", "Answer:\n2", "the result is 5"]:
            self.assertIsNone(math_answer(text), msg=text)


class RunCode(unittest.TestCase):
    TESTS = "assert add(2, 3) == 5"

    def run_(self, body):
        return run_code(f"{FENCE}python\n{body}\n{FENCE}", self.TESTS)[0]

    def test_pass_and_fail(self):
        self.assertTrue(self.run_("def add(a, b): return a + b"))
        self.assertFalse(self.run_("def add(a, b): return a - b"))

    def test_early_exit_cannot_pass(self):
        self.assertFalse(self.run_("raise SystemExit(0)"))
        self.assertFalse(self.run_('print("__ALL_TESTS_COMPLETED_7f3a__")\nraise SystemExit(0)'))
        self.assertFalse(self.run_('import os\nprint("__TESTS_DONE_" + "0" * 32 + "__")\nos._exit(0)'))


if __name__ == "__main__":
    unittest.main()
