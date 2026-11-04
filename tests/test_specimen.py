import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class SpecimenTests(unittest.TestCase):
    def setUp(self):
        self.html = (Path(__file__).resolve().parents[1] / "specimen" / "index.html").read_text(encoding="utf-8")

    def test_supports_paragraph_pairing_and_responsive_wrap(self):
        self.assertIn('value="paragraphs"', self.html)
        self.assertIn(r"raw.split(/\\n\\s*\\n+/)", self.html)
        self.assertIn("overflow-wrap:anywhere", self.html)
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


if __name__ == "__main__":
    unittest.main()
