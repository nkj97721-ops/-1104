import unittest

from tools.sentence_align import align_sentence_embeddings


class SentenceAlignmentTests(unittest.TestCase):
    def test_exact_parallel_pairs_are_aligned(self):
        result = align_sentence_embeddings(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            threshold=0.3,
        )
        self.assertEqual(result["matches"][0]["source_indices"], [0])
        self.assertEqual(result["matches"][0]["target_indices"], [0])
        self.assertEqual(result["matches"][1]["source_indices"], [1])
        self.assertEqual(result["matches"][1]["target_indices"], [1])

    def test_unrelated_pair_can_be_left_unmatched(self):
        result = align_sentence_embeddings(
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            threshold=0.8,
        )
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["unmatched_source"], [0])
        self.assertEqual(result["unmatched_target"], [0])

    def test_merge_transition_is_supported(self):
        result = align_sentence_embeddings(
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0]],
            threshold=0.3,
            gap_penalty=0.5,
        )
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["source_count"], 2)
        self.assertEqual(result["matches"][0]["target_count"], 1)


if __name__ == "__main__":
    unittest.main()
