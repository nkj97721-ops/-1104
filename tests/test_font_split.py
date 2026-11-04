import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("twinline_build_font", ROOT / "build_font.py")
FONT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(FONT)


class FontSplitTests(unittest.TestCase):
    def setUp(self):
        self.cmap = {ord(char): char for char in "alpha beta gamma delta"}
        self.metrics = {char: (1, 0) for char in self.cmap.values()}

    def test_split_prefers_word_boundaries(self):
        original = "alpha beta gamma delta"
        left, right = FONT.split_line_in_two(original, self.cmap, self.metrics, 1.0)
        self.assertEqual(left + right, original)
        self.assertTrue(left.rstrip().endswith("alpha") or left.rstrip().endswith("beta"))
        self.assertTrue(right.lstrip().startswith("beta") or right.lstrip().startswith("gamma"))
        self.assertNotRegex(left.rstrip(), r"alph$")
        self.assertNotRegex(right.lstrip(), r"^amma")

    def test_split_preserves_text_when_no_whitespace_exists(self):
        original = "中文字符"
        cmap = {ord(char): char for char in original}
        metrics = {char: (1, 0) for char in original}
        parts = FONT.split_line_in_two(original, cmap, metrics, 1.0)
        self.assertIsNotNone(parts)
        self.assertEqual("".join(parts), original)


if __name__ == "__main__":
    unittest.main()
