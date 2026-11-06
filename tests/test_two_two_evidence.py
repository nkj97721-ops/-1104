import unittest

from tools.align_document import build_two_two_pair_evidence


class TwoTwoPairEvidenceTests(unittest.TestCase):
    def test_keeps_2x2_boundary_and_exposes_ordered_1x1_candidates(self):
        source = [[0.0, 1.0] for _ in range(5)]
        source[3], source[4] = [1.0, 0.0], [0.0, 1.0]
        target = [[0.0, 1.0] for _ in range(9)]
        target[7], target[8] = [0.99, 0.01], [0.02, 0.98]
        match = {"source_indices": [3, 4], "target_indices": [7, 8], "similarity": 0.71}
        result = build_two_two_pair_evidence(
            match,
            ["unused", "unused", "unused", "First English.", "Second English."],
            ["unused"] * 7 + ["第一句。", "第二句。"],
            source, target, threshold=0.8,
        )
        self.assertEqual(result["boundary_group_type"], "2:2")
        self.assertEqual(result["boundary_source_indices"], [3, 4])
        self.assertEqual(result["boundary_target_indices"], [7, 8])
        self.assertEqual(
            [(p["source_index"], p["target_index"]) for p in result["ordered_pair_candidates"]],
            [(3, 7), (4, 8)],
        )
        self.assertEqual(result["candidate_status"], "two_ordered_pair_candidates")
        self.assertTrue(all(p["status"] == "candidate_only" for p in result["ordered_pair_candidates"]))
        self.assertEqual(match["source_indices"], [3, 4])
        self.assertEqual(match["target_indices"], [7, 8])

    def test_does_not_promote_weak_internal_pairs(self):
        result = build_two_two_pair_evidence(
            {"source_indices": [0, 1], "target_indices": [0, 1], "similarity": 0.6},
            ["A.", "B."], ["甲。", "乙。"],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]], threshold=0.3,
        )
        self.assertEqual(result["candidate_status"], "insufficient_individual_pair_evidence")
        self.assertFalse(any(p["meets_threshold"] for p in result["ordered_pair_candidates"]))

    def test_non_2x2_group_has_no_internal_pair_evidence(self):
        result = build_two_two_pair_evidence(
            {"source_indices": [0], "target_indices": [0], "similarity": 0.9},
            ["A."], ["甲。"], [[1.0, 0.0]], [[1.0, 0.0]], threshold=0.3,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
