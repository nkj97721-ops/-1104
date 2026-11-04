import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_silver_groups import load_silver


class SilverGroupFilteringTests(unittest.TestCase):
    def test_keeps_supported_groups_and_excludes_only_oversize_regions(self):
        calibration = {
            "id": "cal-1",
            "movie_id": "movie-cal",
            "split": "calibration",
            "annotation_status": "pending_human_review",
            "source": ["A", "B", "C", "D", "E"],
            "target": ["甲", "乙", "丙", "丁"],
            "proposed_groups": [
                {"source": [0], "target": [0, 1]},
                {"source": [1, 2, 3], "target": [2]},
            ],
            "proposed_unmatched_source": [4],
            "proposed_unmatched_target": [3],
        }
        test = {
            "id": "test-1",
            "movie_id": "movie-test",
            "split": "test",
            "annotation_status": "pending_human_review",
            "source": ["X"],
            "target": ["戊"],
            "proposed_groups": [{"source": [0], "target": [0]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.jsonl"
            path.write_text(
                json.dumps(calibration, ensure_ascii=False) + "\n" +
                json.dumps(test, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            docs, counts = load_silver(path, 10)
        first = docs[0]
        self.assertEqual(first["source"], ["A", "E"])
        self.assertEqual(first["target"], ["甲", "乙", "丁"])
        self.assertEqual(first["_gold_groups"], [((0,), (0, 1))])
        self.assertEqual(first["_gold_unmatched_source"], [1])
        self.assertEqual(first["_gold_unmatched_target"], [2])
        self.assertEqual(counts["excluded_oversize_groups"], 1)


if __name__ == "__main__":
    unittest.main()
