import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.benchmark_two_two_candidates import (
    cosine, load_heldout_groups, summarize_model,
)


class TwoTwoCandidateBenchmarkTests(unittest.TestCase):
    def test_cosine_handles_zero_vectors_and_dimension_mismatch(self):
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])

    def test_loads_only_heldout_two_two_groups(self):
        rows = [
            {
                "id": "dev", "split": "calibration",
                "reference_status": "automatic_silver_reference",
                "source": ["A.", "B."], "target": ["甲。", "乙。"],
                "proposed_groups": [{"source": [0, 1], "target": [0, 1]}],
            },
            {
                "id": "test", "split": "test",
                "reference_status": "automatic_silver_reference",
                "source": ["A.", "B.", "C."], "target": ["甲。", "乙。", "丙。"],
                "source_sentence_origins": [
                    {"sentence_index": 10}, {"sentence_index": 11}, {"sentence_index": 12},
                ],
                "target_sentence_origins": [
                    {"sentence_index": 20}, {"sentence_index": 21}, {"sentence_index": 22},
                ],
                "proposed_groups": [
                    {"source": [0, 1], "target": [0, 1]},
                    {"source": [2], "target": [2]},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silver.jsonl"
            path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
            groups, counts = load_heldout_groups(path)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["document_id"], "test")
        self.assertEqual(groups[0]["source_indices"], [10, 11])
        self.assertEqual(groups[0]["target_indices"], [20, 21])
        self.assertEqual(counts["two_two_groups"], 1)
        self.assertEqual(counts["test_documents"], 1)

    @patch("tools.benchmark_two_two_candidates.clear_model_cache")
    @patch("tools.benchmark_two_two_candidates.encode_sentences")
    def test_reports_ordered_candidates_without_claiming_accuracy(self, encode, _clear):
        # source A aligns strongly to target A; source B aligns strongly to target B.
        encode.return_value = [
            [1.0, 0.0], [0.0, 1.0],
            [0.9, 0.1], [0.1, 0.9],
        ]
        groups = [
            {
                "document_id": "movie-1",
                "source": ["A.", "B."],
                "target": ["甲。", "乙。"],
                "source_indices": [0, 1],
                "target_indices": [0, 1],
                "source_boundary_ids": ["line-a", "line-b"],
                "target_boundary_ids": ["line-c", "line-d"],
            },
            {
                "document_id": "movie-2",
                "source": ["C.", "D."],
                "target": ["丙。", "丁。"],
                "source_indices": [0, 1],
                "target_indices": [0, 1],
            },
        ]
        result = summarize_model(groups, "fake-model", 0.35, True)
        self.assertEqual(result["candidate_support_counts"]["both_ordered_pairs_meet_threshold"], 2)
        self.assertEqual(result["ordered_vs_cross_margin"]["positive"], 2)
        self.assertIn("not proof", result["interpretation"])
        self.assertEqual(len(result["all_group_diagnostics"]), 2)
        self.assertEqual(result["all_group_diagnostics"][0]["source_boundary_ids"], ["line-a", "line-b"])
        self.assertEqual(result["all_group_diagnostics"][0]["target_boundary_ids"], ["line-c", "line-d"])
        self.assertEqual(encode.call_count, 2)
        _clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
