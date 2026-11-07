import unittest

from tools.diagnose_boundary_decomposition import (
    classify_boundary_group, build_report,
)


class BoundaryDecompositionTests(unittest.TestCase):
    def test_exact_2x2_boundary_is_distinguished(self):
        status, fragments = classify_boundary_group(
            [4, 5], [7, 8],
            [{"source": [4, 5], "target": [7, 8]}],
        )
        self.assertEqual(status, "exact_boundary_group")
        self.assertEqual(fragments, [])

    def test_2x2_boundary_split_into_ordered_1x1_is_explicit(self):
        status, fragments = classify_boundary_group(
            [4, 5], [7, 8],
            [{"source": [4], "target": [7]}, {"source": [5], "target": [8]}],
        )
        self.assertEqual(status, "split_into_two_ordered_1_to_1")
        self.assertEqual(len(fragments), 2)

    def test_crossed_1x1_is_not_mislabeled_as_ordered(self):
        status, _ = classify_boundary_group(
            [4, 5], [7, 8],
            [{"source": [4], "target": [8]}, {"source": [5], "target": [7]}],
        )
        self.assertEqual(status, "split_into_two_crossed_1_to_1")

    def test_partial_coverage_is_not_reported_as_complete(self):
        status, _ = classify_boundary_group(
            [4, 5], [7, 8], [{"source": [4], "target": [7]}],
        )
        self.assertEqual(status, "partial_or_missed")

    def test_report_joins_candidate_evidence_without_claiming_accuracy(self):
        comparison = {
            "reference_type": "automatic silver",
            "languages": {"source": "English", "target": "Simplified Chinese (zh_CN)"},
            "split_unit": "movie",
            "model_comparisons": [{
                "model": "test-model",
                "heldout_document_errors": [{
                    "id": "doc-1",
                    "predicted": [
                        {"source": [0], "target": [0]},
                        {"source": [1], "target": [1]},
                    ],
                    "original_boundary_groups": [{
                        "boundary_id": "line-1", "source": [0, 1],
                        "target": [0, 1], "type": "2:2",
                    }],
                }],
            }],
        }
        candidates = {"model_diagnostics": [{
            "model": "test-model",
            "all_group_diagnostics": [{
                "document_id": "doc-1", "source_indices": [100, 101],
                "target_indices": [200, 201],
                "source_boundary_ids": ["line-1"],
                "target_boundary_ids": ["line-1"],
                "ordered_minus_cross_margin": 0.4,
                "pairs_over_threshold": 2,
            }],
        }]}
        report = build_report(comparison, candidates)
        self.assertIn("NOT_SEMANTIC_ACCURACY", report["status"])
        model = report["models"][0]
        self.assertEqual(
            model["by_group_type"]["2:2"]["classifications"]["split_into_two_ordered_1_to_1"],
            1,
        )
        evidence = model["two_two_embedding_evidence_by_dp_classification"]
        self.assertEqual(evidence["split_into_two_ordered_1_to_1"]["positive_ordered_minus_cross_margin"], 1)
        self.assertEqual(evidence["split_into_two_ordered_1_to_1"]["both_ordered_pairs_over_threshold"], 1)


if __name__ == "__main__":
    unittest.main()
