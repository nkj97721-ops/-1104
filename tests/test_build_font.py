import unittest

from build_font import (MAX_GLYPH_ADVANCE, MAX_PAIRS, PUA_START, composite_advance_width,
                        expand_pairs, split_text_to_width, validate_pairs)


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
            validate_pairs([{"en": "Hello\nWorld", "zh": "你好"}])

    def test_rejects_too_many_pairs_before_build(self):
        with self.assertRaises(ValueError):
            validate_pairs([{"en": "A", "zh": "甲"}] * (MAX_PAIRS + 1))

    def test_true_type_advance_width_limit_is_16_bit(self):
        self.assertEqual(MAX_GLYPH_ADVANCE, 65535)
        self.assertEqual(composite_advance_width(1200, 1400, 1), 2048)

    def test_rejects_glyph_that_cannot_fit_true_type_advance_field(self):
        with self.assertRaisesRegex(ValueError, "too wide for a TrueType glyph"):
            composite_advance_width(70000, 1000, 3)

    def test_wraps_english_at_word_boundaries(self):
        cmap = {ord(c): c for c in "hello world"}
        metrics = {c: (1000, 0) for c in set(cmap.values())}
        wrapped = split_text_to_width("hello world", cmap, metrics, 0.48, 2700)
        self.assertEqual(wrapped, ["hello", "world"])

    def test_wraps_cjk_without_dropping_characters(self):
        text = "中文自动换行"
        cmap = {ord(c): c for c in text}
        metrics = {c: (1000, 0) for c in set(cmap.values())}
        wrapped = split_text_to_width(text, cmap, metrics, 0.48, 1000)
        self.assertEqual("".join(wrapped), text)
        self.assertTrue(all(len(line) <= 2 for line in wrapped))

    def test_expands_pair_and_keeps_source_order(self):
        en = "abcdefgh"
        zh = "中文测试文字"
        en_cmap = {ord(c): c for c in en}
        zh_cmap = {ord(c): c for c in zh}
        en_metrics = {c: (1000, 0) for c in set(en_cmap.values())}
        zh_metrics = {c: (1000, 0) for c in set(zh_cmap.values())}
        rows = expand_pairs([{"en": en, "zh": zh}], en_cmap, en_metrics,
                            zh_cmap, zh_metrics, 2000)
        self.assertGreater(len(rows), 1)
        self.assertEqual("".join(row["en"] for row in rows), en)
        self.assertEqual("".join(row["zh"] for row in rows), zh)
        self.assertTrue(all(row["source_pair"] == 1 for row in rows))

    def test_private_use_range_has_expected_capacity(self):
        self.assertEqual(PUA_START, 0xE000)
        self.assertEqual(MAX_PAIRS, 6400)
        self.assertEqual(PUA_START + MAX_PAIRS - 1, 0xF8FF)


if __name__ == "__main__":
    unittest.main()
