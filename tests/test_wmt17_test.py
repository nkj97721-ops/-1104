import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from download_wmt17_test import join_segment_pairs, parse_sgm_segments


class WMT17SGMTests(unittest.TestCase):
    def test_parse_segments_with_document_and_segment_ids(self):
        sgm = """<srcset setid="newstest2017" srclang="en">
<doc docid="news-a">
<seg id="1">Hello &amp; welcome.</seg>
<seg id="2">Second sentence.</seg>
</doc>
<doc docid="news-b">
<seg id="1">Another document.</seg>
</doc>
</srcset>"""
        parsed = parse_sgm_segments(sgm)
        self.assertEqual(parsed[("news-a", "1")], "Hello & welcome.")
        self.assertEqual(parsed[("news-b", "1")], "Another document.")

    def test_pairs_are_joined_by_ids_not_dictionary_order(self):
        source = {
            ("doc-a", "1"): "First",
            ("doc-a", "2"): "Second",
        }
        target = {
            ("doc-a", "2"): "第二",
            ("doc-a", "1"): "第一",
        }
        pairs = join_segment_pairs(source, target)
        self.assertEqual(
            pairs,
            [
                (("doc-a", "1"), "First", "第一"),
                (("doc-a", "2"), "Second", "第二"),
            ],
        )

    def test_mismatched_segment_ids_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "segment IDs do not match"):
            join_segment_pairs({("doc", "1"): "A"}, {("doc", "2"): "甲"})

    def test_duplicate_segment_ids_are_rejected(self):
        sgm = """<srcset><doc docid="doc">
<seg id="1">First</seg><seg id="1">Duplicate</seg>
</doc></srcset>"""
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_sgm_segments(sgm)


if __name__ == "__main__":
    unittest.main()
