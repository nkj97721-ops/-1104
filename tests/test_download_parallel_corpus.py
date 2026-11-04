import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "download_parallel_corpus.py"
SPEC = importlib.util.spec_from_file_location("twinline_download_corpus", MODULE_PATH)
CORPUS = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(CORPUS)


class FakeOpenCC:
    def convert(self, text):
        return text.replace("繁", "简")


class CorpusFilteringTests(unittest.TestCase):
    def test_accepts_english_and_simplified_chinese_pair(self):
        self.assertEqual(
            CORPUS.normalize_pair({"source": "Hello world", "target": "你好世界"}, FakeOpenCC()),
            ("Hello world", "你好世界", None),
        )

    def test_rejects_traditional_chinese(self):
        self.assertEqual(
            CORPUS.normalize_pair({"source": "Hello", "target": "繁體中文"}, FakeOpenCC())[2],
            "traditional",
        )

    def test_rejects_likely_cantonese_written_markers(self):
        self.assertEqual(
            CORPUS.normalize_pair({"source": "How are you", "target": "你係邊個"}, FakeOpenCC())[2],
            "likely_cantonese",
        )

    def test_rejects_non_english_source_and_non_chinese_target(self):
        self.assertEqual(
            CORPUS.normalize_pair({"source": "你好", "target": "世界"}, FakeOpenCC())[2],
            "not_english",
        )
        self.assertEqual(
            CORPUS.normalize_pair({"source": "Hello", "target": "hello"}, FakeOpenCC())[2],
            "not_chinese",
        )

    def test_collapses_internal_whitespace(self):
        self.assertEqual(
            CORPUS.normalize_pair(
                {"source": "Hello\n world", "target": "你 好\n世界"}, FakeOpenCC()
            ),
            ("Hello world", "你 好 世界", None),
        )

    def test_dataset_page_must_be_complete_and_ordered(self):
        page = {
            "rows": [
                {"row_idx": 20, "row": {"source": "Hello", "target": "你好"}},
                {"row_idx": 21, "row": {"source": "World", "target": "世界"}},
            ]
        }
        self.assertEqual(
            list(CORPUS._extract_page_rows(page, 20, 2)),
            [{"source": "Hello", "target": "你好"}, {"source": "World", "target": "世界"}],
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            list(CORPUS._extract_page_rows({"rows": page["rows"][:1]}, 20, 2))
        with self.assertRaisesRegex(RuntimeError, "order mismatch"):
            list(CORPUS._extract_page_rows({"rows": [page["rows"][1]]}, 20, 1))

    def test_dataset_page_requires_paired_fields(self):
        with self.assertRaisesRegex(RuntimeError, "source/target"):
            list(CORPUS._extract_page_rows(
                {"rows": [{"row_idx": 0, "row": {"source": "Hello"}}]}, 0, 1
            ))


if __name__ == "__main__":
    unittest.main()
