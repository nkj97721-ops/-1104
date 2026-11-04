import unittest

from tools.prepare_opensubtitles2024_gold import identify_text_member, split_aligned_lines
from tools.review_gold_queue import validate_groups


class OpenSubtitlesPreparationTests(unittest.TestCase):
    def test_identifies_only_english_and_simplified_chinese_members(self):
        self.assertEqual(
            identify_text_member("set/movie_123.en"),
            ("movie_123", "en"),
        )
        self.assertEqual(
            identify_text_member("set/movie_123.zh_CN"),
            ("movie_123", "zh_CN"),
        )
        self.assertIsNone(identify_text_member("set/movie_123.zh_TW"))
        self.assertIsNone(identify_text_member("set/movie_123.yue"))

    def test_splits_multisentence_lines_independently_and_preserves_groups(self):
        source, target, groups, unmatched_s, unmatched_t, oversize = split_aligned_lines(
            ["Hello. Goodbye.", "Wait!"],
            ["你好。", "等等！再见。"],
        )
        self.assertEqual(source, ["Hello.", "Goodbye.", "Wait!"])
        self.assertEqual(target, ["你好。", "等等！", "再见。"])
        self.assertEqual(
            [(g["source"], g["target"]) for g in groups],
            [([0, 1], [0]), ([2], [1, 2])],
        )
        self.assertEqual(unmatched_s, [])
        self.assertEqual(unmatched_t, [])
        self.assertEqual(oversize, 0)

    def test_rejects_line_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "line counts"):
            split_aligned_lines(["one", "two"], ["一"])

    def test_human_group_validator_rejects_empty_manual_resegmentation(self):
        with self.assertRaisesRegex(ValueError, "non-empty sentence strings"):
            validate_groups([""], ["甲"], [])

    def test_human_group_validator_accepts_2_to_2_and_rejects_reuse(self):
        validate_groups(
            ["A", "B"],
            ["甲", "乙"],
            [{"source": [0, 1], "target": [0, 1]}],
        )
        with self.assertRaisesRegex(ValueError, "reused|strictly monotonic"):
            validate_groups(
                ["A", "B"],
                ["甲", "乙"],
                [
                    {"source": [0], "target": [0]},
                    {"source": [0], "target": [1]},
                ],
            )


if __name__ == "__main__":
    unittest.main()
