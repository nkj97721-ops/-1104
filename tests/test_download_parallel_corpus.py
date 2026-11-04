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

    def test_parses_one_paired_tsv_record(self):
        self.assertEqual(
            CORPUS.parse_parallel_line("Hello world" + chr(9) + "你好世界" + chr(10)),
            {"source": "Hello world", "target": "你好世界"},
        )

    def test_tsv_parser_splits_only_at_first_tab(self):
        self.assertEqual(
            CORPUS.parse_parallel_line("Hello" + chr(9) + "你好" + chr(9) + "附注"),
            {"source": "Hello", "target": "你好" + chr(9) + "附注"},
        )

    def test_malformed_nonblank_tsv_line_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "expected EN"):
            CORPUS.parse_parallel_line("no separator here")

    def test_blank_physical_line_is_ignored(self):
        self.assertEqual(
            CORPUS.parse_parallel_line("\\n"),
            {"source": "", "target": ""},
        )

    def test_empty_language_column_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "field is empty"):
            CORPUS.parse_parallel_line("English only\\t")


if __name__ == "__main__":
    unittest.main()
