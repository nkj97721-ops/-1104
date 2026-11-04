import unittest

from build_font import MAX_PAIRS, PUA_START, validate_pairs


class ValidatePairsTests(unittest.TestCase):
    def test_accepts_multiple_aligned_pairs(self):
        validate_pairs([
            {"en": "Hello", "zh": "你好"},
            {"en": "Good night", "zh": "晚安"},
        ])

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            validate_pairs([])

    def test_rejects_missing_translation(self):
        with self.assertRaises(ValueError):
            validate_pairs([{"en": "Hello", "zh": ""}])

    def test_rejects_embedded_newline(self):
        with self.assertRaises(ValueError):
            validate_pairs([{"en": "Hello\\nWorld", "zh": "你好"}])

    def test_rejects_too_many_pairs_before_build(self):
        with self.assertRaises(ValueError):
            validate_pairs([{"en": "A", "zh": "甲"}] * (MAX_PAIRS + 1))

    def test_private_use_range_has_expected_capacity(self):
        self.assertEqual(PUA_START, 0xE000)
        self.assertEqual(MAX_PAIRS, 6400)
        self.assertEqual(PUA_START + MAX_PAIRS - 1, 0xF8FF)


if __name__ == "__main__":
    unittest.main()
