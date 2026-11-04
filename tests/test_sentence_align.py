import unittest
from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "sentence_align.py"
SPEC = importlib.util.spec_from_file_location("twinline_sentence_align", MODULE_PATH)
ALIGN = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ALIGN)


class SentenceAlignmentTests(unittest.TestCase):
    def test_split_sentences_keeps_chinese_and_english_punctuation(self):
        self.assertEqual(ALIGN.split_sentences("Hello world. 你好世界！Next?"),
                         ["Hello world.", "你好世界！", "Next?"])

    def test_does_not_split_common_abbreviations_or_decimals(self):
        self.assertEqual(
            ALIGN.split_sentences("Dr. Smith paid 3.14 dollars. Next sentence."),
            ["Dr. Smith paid 3.14 dollars.", "Next sentence."],
        )

    def test_aligns_monotonic_one_to_one_pairs(self):
        result = ALIGN.align_sentence_embeddings([[1, 0], [0, 1]],
                                                  [[0.99, 0.01], [0.02, 0.98]],
                                                  threshold=0.3)
        self.assertEqual([(m["source_indices"], m["target_indices"]) for m in result["matches"]],
                         [([0], [0]), ([1], [1])])
        self.assertFalse(result["unmatched_source"])
        self.assertFalse(result["unmatched_target"])

    def test_supports_one_to_two_sentence_merge(self):
        result = ALIGN.align_sentence_embeddings([[1, 0]], [[0.9, 0.1], [0.8, 0.2]],
                                                  threshold=0.3, merge_penalty=0.0, gap_penalty=0.2)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["source_indices"], [0])
        self.assertEqual(result["matches"][0]["target_indices"], [0, 1])

    def test_marks_unrelated_sentences_unmatched(self):
        result = ALIGN.align_sentence_embeddings([[1, 0]], [[-1, 0]], threshold=0.4, gap_penalty=0.1)
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["unmatched_source"], [0])
        self.assertEqual(result["unmatched_target"], [0])

    def test_rejects_mismatched_vector_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            ALIGN.align_sentence_embeddings([[1, 0]], [[1, 0, 0]])

    def test_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            ALIGN.align_sentence_embeddings([[1, 0]], [[1, 0]], threshold=1.5)

    def test_rejects_wrong_similarity_table_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            ALIGN.align_sentence_embeddings(
                [[1, 0]], [[1, 0]], similarity_tables=([[1, 0]], [[1]], [[1]])
            )


if __name__ == "__main__":
    unittest.main()
