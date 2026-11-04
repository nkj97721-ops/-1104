#!/usr/bin/env python3
"""Build paired English-over-Chinese composite glyphs as a TrueType font."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from fontTools.ttLib import TTFont, TTCollection
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.cu2quPen import Cu2QuPen

UPM, PUA_START, PUA_END = 2048, 0xE000, 0xF8FF
MAX_GLYPH_ADVANCE = 65535
MAX_PAIRS = PUA_END - PUA_START + 1

def composite_advance_width(english_width, chinese_width, pair_index):
    width = max(UPM, int(max(english_width, chinese_width) + 100))
    if width > MAX_GLYPH_ADVANCE:
        raise ValueError(
            f"Pair {pair_index} is too wide for a TrueType glyph ({width} units). "
            "Split this English/Chinese pair into shorter corresponding lines; the font will not be written."
        )
    return width



def validate_pairs(pairs):
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Input must be a non-empty JSON array of {en, zh} pairs")
    if len(pairs) > MAX_PAIRS:
        raise ValueError(f"Too many pairs: maximum is {MAX_PAIRS} (BMP Private Use Area)")
    for index, pair in enumerate(pairs, 1):
        if not isinstance(pair, dict):
            raise ValueError(f"Pair {index} must be an object")
        for key in ("en", "zh"):
            value = pair.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Pair {index}: '{key}' must be a non-empty string")
            if "\n" in value or "\r" in value:
                raise ValueError(f"Pair {index}: '{key}' must be one line; split paragraphs into paired lines")


def fc_match(pattern):
    path = subprocess.run(
        ["fc-match", "-f", "%{file}", pattern],
        check=True, text=True, capture_output=True
    ).stdout.strip()
    if not path or not os.path.isfile(path):
        raise RuntimeError("Font not found: " + pattern)
    return path


def load_cjk(path):
    if path.lower().endswith((".ttc", ".otc")):
        coll = TTCollection(path)
        for font in coll.fonts:
            names = [n.toUnicode() for n in font["name"].names if n.nameID == 1]
            if any("CJK SC" in name for name in names):
                return font
        raise RuntimeError("Simplified Chinese face missing")
    return TTFont(path)


def source(font):
    cmap = {}
    for table in font["cmap"].tables:
        if table.isUnicode():
            cmap.update(table.cmap)
    return font.getGlyphSet(), cmap, font["hmtx"].metrics


def draw_text(text, glyph_set, cmap, metrics, pen, scale, baseline, missing):
    x = 0
    for char in text:
        name = cmap.get(ord(char))
        if name is None:
            missing.add(char)
            name = cmap.get(0x25A1) or cmap.get(ord("?"))
        if name is None:
            x += 600 * scale
            continue
        advance, _ = metrics[name]
        target = TransformPen(pen, (scale, 0, 0, scale, x, baseline))
        glyph_set[name].draw(Cu2QuPen(target, max_err=1.0))
        x += advance * scale
    return x


def build(pairs, output, family, encoded_output=None):
    validate_pairs(pairs)
    output = Path(output)
    encoded_output = Path(encoded_output) if encoded_output else output.with_suffix(".txt")

    latin_path = fc_match("Lato:style=Medium")
    cjk_path = fc_match("Noto Sans CJK SC:style=Regular")
    latin_gs, latin_cmap, latin_metrics = source(TTFont(latin_path))
    cjk_gs, cjk_cmap, cjk_metrics = source(load_cjk(cjk_path))

    order = [".notdef"]
    cmap = {}
    glyf = {".notdef": TTGlyphPen(None).glyph()}
    hmtx = {".notdef": (UPM, 0)}
    missing = set()
    mapping = []
    encoded_chars = []

    for index, pair in enumerate(pairs):
        glyph_name = f"pair{index:04d}"
        pen = TTGlyphPen(None)
        english_width = draw_text(
            pair["en"], latin_gs, latin_cmap, latin_metrics, pen, 0.48, 1120, missing
        )
        chinese_width = draw_text(
            pair["zh"], cjk_gs, cjk_cmap, cjk_metrics, pen, 0.48, 110, missing
        )
        advance_width = composite_advance_width(english_width, chinese_width, index + 1)
        glyf[glyph_name] = pen.glyph()
        hmtx[glyph_name] = (advance_width, 0)
        order.append(glyph_name)

        codepoint = PUA_START + index
        character = chr(codepoint)
        cmap[codepoint] = glyph_name
        encoded_chars.append(character)
        mapping.append({
            "index": index + 1,
            "codepoint": f"U+{codepoint:04X}",
            "character": character,
            "en": pair["en"],
            "zh": pair["zh"],
        })

    if missing:
        raise RuntimeError("Missing source glyphs: " + repr(sorted(missing)))

    builder = FontBuilder(UPM, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap(cmap)
    builder.setupGlyf(glyf)
    builder.setupHorizontalMetrics(hmtx)
    builder.setupHorizontalHeader(ascent=2048, descent=-180)
    builder.setupNameTable({
        "familyName": family,
        "styleName": "Regular",
        "uniqueFontIdentifier": family + " Regular 1.0",
        "fullName": family + " Regular",
        "psName": family.replace(" ", "") + "-Regular",
        "version": "Version 1.000",
    })
    builder.setupOS2(
        sTypoAscender=2048, sTypoDescender=-180,
        usWinAscent=2048, usWinDescent=180,
        sxHeight=1050, sCapHeight=1400,
        usWeightClass=500, usWidthClass=5,
    )
    builder.setupPost()
    builder.setupMaxp()
    font = builder.font
    font["head"].created = font["head"].modified = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    font.save(output)
    mapping_path = output.with_suffix(".mapping.json")
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    encoded_output.parent.mkdir(parents=True, exist_ok=True)
    encoded_output.write_text("\n".join(encoded_chars) + "\n", encoding="utf-8")

    check = TTFont(output)
    glyph_set = check.getGlyphSet()
    bad = []
    for glyph_name in order[1:]:
        bounds_pen = BoundsPen(glyph_set)
        glyph_set[glyph_name].draw(bounds_pen)
        if bounds_pen.bounds and (bounds_pen.bounds[1] < -30 or bounds_pen.bounds[3] > 2048):
            bad.append((glyph_name, bounds_pen.bounds))
    if bad:
        raise RuntimeError("Vertical overflow: " + repr(bad[:5]))
    if len(check.getBestCmap()) != len(pairs):
        raise RuntimeError("cmap count mismatch")
    if len(encoded_chars) != len(mapping):
        raise RuntimeError("Encoded text and mapping count mismatch")

    print(json.dumps({
        "font": str(output),
        "bytes": output.stat().st_size,
        "pairs": len(pairs),
        "vertical_bounds": "PASS",
        "missing_glyphs": [],
        "mapping": str(mapping_path),
        "encoded_text": str(encoded_output),
        "encoded_characters": len(encoded_chars),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="samples/pairs.json")
    parser.add_argument("--output", default="dist/TwinLine-V2-Sample.ttf")
    parser.add_argument("--family", default="TwinLine V2 Sample")
    parser.add_argument("--encoded-output", help="Output PUA text file; defaults to output name with .txt")
    args = parser.parse_args()

    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    validate_pairs(pairs)
    build(pairs, args.output, args.family, args.encoded_output)


if __name__ == "__main__":
    main()
