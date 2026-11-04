import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class SpecimenTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.html = (root / "specimen" / "index.html").read_text(encoding="utf-8")
        self.renderer = (root / "specimen" / "renderer.js").read_text(encoding="utf-8")

    def test_uses_synchronized_pair_renderer(self):
        self.assertIn('src="renderer.js"', self.html)
        self.assertIn("wrapPair", self.renderer)
        self.assertIn("splitOneLine", self.renderer)
        self.assertIn("paragraphBlocks", self.renderer)
        self.assertIn("splitSentences", self.renderer)
        self.assertIn("preparePairs", self.renderer)
        self.assertIn("数量不一致", self.renderer)
        self.assertIn("按句子配对", self.html)
        self.assertIn("已暂停输出", self.html)
        self.assertIn("new ResizeObserver", self.html)
        self.assertIn("TwinLine-layout.html", self.html)

    def test_inline_javascript_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is unavailable")
        scripts = re.findall(r"<script>(.*?)</script>", self.html, flags=re.S)
        self.assertEqual(len(scripts), 1)
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as handle:
            handle.write(scripts[0])
            handle.flush()
            result = subprocess.run([node, "--check", handle.name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_renderer_javascript_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is unavailable")
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as handle:
            handle.write(self.renderer)
            handle.flush()
            result = subprocess.run([node, "--check", handle.name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
