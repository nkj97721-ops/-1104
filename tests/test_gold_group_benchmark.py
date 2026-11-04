import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_gold_groups import aggregate, load_documents, prf


class GoldGroupFormatTests(unittest.TestCase):
    def test_loads_1_to_2_and_2_to_1_and_infers_unmatched(self):
        payload = {
            "id": "synthetic-1",
            "split": "calibration",
            "annotation_status": "human_verified",
            "annotator": "test-annotator",
            "reviewed_at": "2026-11-04T20:00:00+08:00",
            "source": ["A", "B", "C"],
            "target": ["甲", "乙", "丙", "丁"],
            "gold": [
                {"source": [0], "target": [0, 1]},
                {"source": [1, 2], "target": [2]},
            ],
        }
        # A second independent test document is required by the loader.
        test_payload = {
            "id": "synthetic-2",
            "split": "test",
            "annotation_status": "human_verified",
            "annotator": "test-annotator",
            "reviewed_at": "2026-11-04T20:00:00+08:00",
            "source": ["D"],
            "target": ["戊", "己"],
            "gold": [{"source": [0], "target": [0]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.jsonl"
            path.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n" +
                json.dumps(test_payload, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            docs = load_documents(path)
        self.assertEqual(docs[0]["_gold_groups"], [((0,), (0, 1)), ((1, 2), (2,))])
        self.assertEqual(docs[0]["_gold_unmatched_source"], [])
        self.assertEqual(docs[0]["_gold_unmatched_target"], [3])
        self.assertEqual(docs[1]["_gold_unmatched_target"], [1])

    def test_rejects_unreviewed_candidate_labels(self):
        payload = {
            "id": "pending",
            "split": "calibration",
            "source": ["A"],
            "target": ["甲"],
            "gold": [{"source": [0], "target": [0]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending.jsonl"
            test_payload = dict(payload, id="pending-test", split="test")
            path.write_text(
                json.dumps(payload) + "\n" + json.dumps(test_payload) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "human_verified"):
                load_documents(path)

    def test_rejects_movie_leakage_across_splits(self):
        common = {
            "annotation_status": "human_verified",
            "annotator": "reviewer",
            "reviewed_at": "2026-11-04T20:00:00+08:00",
            "movie_id": "same-movie",
            "source": ["A"],
            "target": ["甲"],
            "gold": [{"source": [0], "target": [0]}],
        }
        calibration = dict(common, id="cal", split="calibration")
        test = dict(common, id="test", split="test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leak.jsonl"
            path.write_text(
                json.dumps(calibration) + "\n" + json.dumps(test) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "leaks across"):
                load_documents(path)

    def test_aggregate_reports_per_group_type_and_marks_unmatched_not_estimable(self):
        row = {
            "exact_groups": 2, "predicted_groups": 3, "gold_groups": 2,
            "correct_links": 3, "predicted_links": 4, "gold_links": 3,
            "correct_unmatched": 0, "predicted_unmatched": 1, "gold_unmatched": 0,
            "group_type_counts": {
                "1:1": {"correct": 1, "predicted": 2, "reference": 1},
                "2:1": {"correct": 1, "predicted": 1, "reference": 1},
            },
        }
        result = aggregate([row])
        self.assertAlmostEqual(result["group_type"]["1:1"]["precision"], 0.5)
        self.assertAlmostEqual(result["group_type"]["1:1"]["recall"], 1.0)
        self.assertEqual(result["unmatched"]["status"], "NOT_ESTIMABLE_NO_REFERENCE_UNMATCHED")
        self.assertIsNone(result["unmatched"]["f1"])

    def test_prf_zero_denominators_are_safe(self):
        self.assertEqual(prf(0, 0, 0), {"precision": 0.0, "recall": 0.0, "f1": 0.0})


if __name__ == "__main__":
    unittest.main()
