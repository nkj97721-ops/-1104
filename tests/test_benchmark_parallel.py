import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import benchmark_parallel as BENCHMARK


class BenchmarkDataTests(unittest.TestCase):
    def test_load_pairs_deduplicates_exact_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.tsv"
            path.write_text("Hello\\t你好\\nHello\\t你好\\nWorld\\t世界\\n", encoding="utf-8")
            self.assertEqual(
                BENCHMARK.load_pairs(path),
                [("Hello", "你好"), ("World", "世界")],
            )

    def test_make_blocks_uses_only_complete_blocks(self):
        pairs = [(f"en-{i}", f"中-{i}") for i in range(45)]
        blocks = BENCHMARK.make_blocks(pairs, 20)
        self.assertEqual([len(block) for block in blocks], [20, 20])
        self.assertEqual(blocks[0][0], pairs[0])
        self.assertEqual(blocks[1][-1], pairs[39])


if __name__ == "__main__":
    unittest.main()
