import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_gold_groups import load_documents, prf


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
                json.dumps(payload) + "\\n" + json.dumps(test_payload) + "\\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "human_verified"):
                load_documents(path)

    def test_prf_zero_denominators_are_safe(self):
        self.assertEqual(prf(0, 0, 0), {"precision": 0.0, "recall": 0.0, "f1": 0.0})


if __name__ == "__main__":
    unittest.main()
