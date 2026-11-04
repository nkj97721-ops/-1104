"""Regression tests for preserving original line/segment provenance."""

from __future__ import annotations

import unittest

from tools.prepare_opensubtitles2024_gold import split_aligned_lines
from tools.prepare_wmt_gold_queue import make_record


class BoundaryProvenanceTests(unittest.TestCase):
    def test_opensubtitles_line_ids_survive_sentence_splitting(self):
        result = split_aligned_lines(
            ["First sentence. Second sentence.", "Third sentence."],
            ["第一句。", "第二句。第三句。"],
            boundary_ids=["movie-7:line:10", "movie-7:line:11"],
        )
        source, target, groups, _, _, _, source_origins, target_origins = result

        self.assertEqual(len(source), len(source_origins))
        self.assertEqual(len(target), len(target_origins))
        self.assertEqual(
            [x["boundary_id"] for x in source_origins],
            ["movie-7:line:10", "movie-7:line:10", "movie-7:line:11"],
        )
        self.assertEqual(
            [x["boundary_id"] for x in target_origins],
            ["movie-7:line:10", "movie-7:line:11", "movie-7:line:11"],
        )
        self.assertEqual(
            [x["suggested_type"] for x in groups],
            ["2:1", "1:2"],
        )
        self.assertEqual(groups[0]["source_boundary_ids"], ["movie-7:line:10"])
        self.assertEqual(groups[0]["target_boundary_ids"], ["movie-7:line:10"])

    def test_boundary_groups_are_reconstructed_separately(self):
        from tools.boundary_groups import reconstruct_boundary_groups

        _, _, _, _, _, _, source_origins, target_origins = split_aligned_lines(
            ["First. Second.", "Third."],
            ["第一句。", "第二句。第三句。"],
            boundary_ids=["movie:line:1", "movie:line:2"],
        )
        result = reconstruct_boundary_groups(source_origins, target_origins)
        self.assertEqual(
            [(x["type"], x["source"], x["target"]) for x in result["groups"]],
            [("2:1", [0, 1], [0]), ("1:2", [2], [1, 2])],
        )
        self.assertEqual(result["unmatched_source"], [])
        self.assertEqual(result["unmatched_target"], [])

    def test_duplicate_boundary_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique ID"):
            split_aligned_lines(["One.", "Two."], ["一。", "二。"],
                                boundary_ids=["same", "same"])

    def test_wmt_segment_keys_survive_record_chunking(self):
        records = make_record(
            "wmt17-doc-1",
            "wmt17-document-1",
            "test",
            [("One. Two.", "一。"), ("Three.", "二。三。")],
            10,
            ["wmt17:doc-1:segment:4", "wmt17:doc-1:segment:9"],
        )
        record = records[0]
        self.assertEqual(
            [x["boundary_id"] for x in record["source_sentence_origins"]],
            ["wmt17:doc-1:segment:4", "wmt17:doc-1:segment:4", "wmt17:doc-1:segment:9"],
        )
        self.assertEqual(
            [x["boundary_id"] for x in record["target_sentence_origins"]],
            ["wmt17:doc-1:segment:4", "wmt17:doc-1:segment:9", "wmt17:doc-1:segment:9"],
        )
        self.assertEqual(record["boundary_metadata_status"],
                         "preserved_original_parallel_segment_ids")


if __name__ == "__main__":
    unittest.main()
