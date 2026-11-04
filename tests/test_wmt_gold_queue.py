import unittest

from tools.prepare_wmt_gold_queue import make_record


class WMTGoldQueueTests(unittest.TestCase):
    def test_independent_segmentation_proposes_many_to_many_groups(self):
        records = make_record(
            "wmt17-doc-1",
            "wmt17-document-1",
            "test",
            [
                ("First sentence. Second sentence.", "第一句。"),
                ("Wait?", "等等？再见。"),
            ],
            10,
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["source"], ["First sentence.", "Second sentence.", "Wait?"])
        self.assertEqual(record["target"], ["第一句。", "等等？", "再见。"])
        self.assertEqual(
            [(x["source"], x["target"]) for x in record["proposed_groups"]],
            [([0, 1], [0]), ([2], [1, 2])],
        )
        self.assertEqual(record["reference_status"], "automatic_silver_reference")


if __name__ == "__main__":
    unittest.main()
