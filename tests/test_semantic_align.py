import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "semantic_align.py"
SPEC = importlib.util.spec_from_file_location("twinline_semantic_align", MODULE_PATH)
ALIGN = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ALIGN)


class SemanticAlignmentTests(unittest.TestCase):
    def test_english_tokenizer_preserves_contractions_and_punctuation(self):
        self.assertEqual(
            ALIGN.tokenize_english("Don't stop, 2026!"),
            ["Don't", "stop", ",", "2026", "!"],
        )

    def test_chinese_tokenizer_uses_lexical_words(self):
        tokens = ALIGN.tokenize_chinese("我喜欢自然语言处理。")
        self.assertTrue(tokens)
        self.assertEqual("".join(tokens), "我喜欢自然语言处理。")
        self.assertIn("。", tokens)

    def test_extracts_consistent_phrase_candidates(self):
        source = ["take", "a", "closer", "look"]
        target = ["仔细", "看一看"]
        links = [
            {"source_index": 0, "target_index": 0, "probability": 0.91},
            {"source_index": 1, "target_index": 0, "probability": 0.88},
            {"source_index": 2, "target_index": 1, "probability": 0.90},
            {"source_index": 3, "target_index": 1, "probability": 0.93},
        ]
        candidates = ALIGN.extract_phrase_candidates(source, target, links)
        pairs = {(item["source"], item["target"]) for item in candidates}
        self.assertIn(("take a", "仔细"), pairs)
        self.assertIn(("closer look", "看一看"), pairs)
        self.assertTrue(all(0 <= item["score"] <= 1 for item in candidates))

    def test_rejects_phrase_with_external_alignment_inside_target_span(self):
        source = ["red", "car", "fast"]
        target = ["红色", "汽车", "快速"]
        links = [
            {"source_index": 0, "target_index": 0, "probability": 0.9},
            {"source_index": 1, "target_index": 1, "probability": 0.9},
            {"source_index": 2, "target_index": 1, "probability": 0.8},
        ]
        candidates = ALIGN.extract_phrase_candidates(source, target, links)
        pairs = {(item["source"], item["target"]) for item in candidates}
        self.assertNotIn(("red", "红色 汽车"), pairs)


if __name__ == "__main__":
    unittest.main()
