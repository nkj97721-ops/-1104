import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import benchmark_parallel as BENCHMARK


class BenchmarkDataTests(unittest.TestCase):
    def test_load_pairs_deduplicates_exact_pairs(self):
        expected = [(f"Hello {i}", f"你好{i}") for i in range(100)]
        lines = [en + chr(9) + zh for en, zh in expected]
        lines.append(lines[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.tsv"
            path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
            self.assertEqual(BENCHMARK.load_pairs(path), expected)

    def test_make_blocks_uses_only_complete_blocks(self):
        pairs = [(f"en-{i}", f"中-{i}") for i in range(45)]
        blocks = BENCHMARK.make_blocks(pairs, 20)
        self.assertEqual([len(block) for block in blocks], [20, 20])
        self.assertEqual(blocks[0][0], pairs[0])
        self.assertEqual(blocks[1][-1], pairs[39])


if __name__ == "__main__":
    unittest.main()
