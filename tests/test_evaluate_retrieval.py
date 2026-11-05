import re
import unittest

from evaluate_retrieval import (
    DEFAULT_EVAL_SET,
    DEFAULT_EXTENDED_EVAL_SET,
    evaluate,
    load_eval_set,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_expanded_set_preserves_baseline_and_covers_requested_query_types(self):
        baseline = load_eval_set(DEFAULT_EVAL_SET)
        additional = load_eval_set(DEFAULT_EXTENDED_EVAL_SET, expected_count=60)
        expanded = baseline + additional

        self.assertEqual(len(baseline), 40)
        self.assertEqual(len(expanded), 100)
        self.assertTrue(any(re.search(r"[\u0600-\u06FF]", case["question"]) for case in additional))
        self.assertTrue(any("kahan waqay" in case["question"].casefold() for case in additional))
        self.assertTrue(any(not case["answerable"] for case in additional))
        self.assertTrue(any(case["location"] for case in additional))

    def test_evaluation_reports_elapsed_time(self):
        class EmptyCorpus:
            documents = []
            retrieval_mode = "bm25"

            def search(self, _question, limit):
                return []

            def has_evidence(self, _question, _documents):
                return False

        result = evaluate(
            EmptyCorpus(),
            [
                {
                    "question": "Who won today's football match?",
                    "expected_urls": [],
                    "answerable": False,
                    "location": False,
                }
            ],
            [],
        )

        self.assertEqual(result["cases"], 1)
        self.assertGreaterEqual(result["elapsed_ms"], 0)


if __name__ == "__main__":
    unittest.main()
