#!/usr/bin/env python3
"""Stream a verified English–Simplified-Chinese OpenSubtitles corpus.

The Hugging Face mirror is derived from OPUS OpenSubtitles v2024 and exposes
explicit source/target columns. Streaming avoids loading a multi-gigabyte
archive into RAM and keeps each translation pair in one dataset row.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

DATASET_ID = "FradSer/OpenSubtitles-en-zh-cn-20m"

# These are strong written-Cantonese markers, not a general dialect detector.
# A single marker is enough to exclude a likely Cantonese target from this
# Mandarin-oriented benchmark; the count is recorded in the manifest.
CANTONESE_MARKERS = re.compile(r"[嘅咗喺冇唔咩噉嗰喎啱嚟乜]")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")


def normalize_pair(row: dict[str, Any], traditional_to_simplified) -> tuple[str, str, str | None]:
    """Return normalized (English, Simplified Chinese, rejection reason)."""
    en = str(row.get("source", "") or "").strip()
    zh = str(row.get("target", "") or "").strip()
    if not en or not zh:
        return "", "", "empty"
    if not LATIN.search(en):
        return "", "", "not_english"
    if not CJK.search(zh):
        return "", "", "not_chinese"
    simplified = traditional_to_simplified.convert(zh)
    if simplified != zh:
        return "", "", "traditional"
    if CANTONESE_MARKERS.search(zh):
        return "", "", "likely_cantonese"
    # One TSV row is one dataset record: no independent line-wise pairing.
    en = re.sub(r"\s+", " ", en).replace("\t", " ")
    zh = re.sub(r"\s+", " ", zh).replace("\t", " ")
    return en, zh, None


def iter_dataset_rows() -> Iterator[dict[str, Any]]:
    """Yield paired records from the streaming dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("缺少 datasets 依赖：请执行 pip install 'datasets>=3,<5'") from exc

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    columns = set(getattr(dataset, "column_names", []) or [])
    if columns and not {"source", "target"}.issubset(columns):
        raise RuntimeError(
            f"语料字段不符合预期：需要 source/target，实际字段为 {sorted(columns)}"
        )
    yield from dataset


def download_pairs(output: Path, limit: int = 0, rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    if limit < 0:
        raise ValueError("--limit 不能为负数。")

    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise SystemExit(
            "缺少简繁检测依赖：请执行 pip install opencc-python-reimplemented"
        ) from exc

    converter = OpenCC("t2s")
    counters = {
        "seen": 0, "written": 0, "empty": 0, "not_english": 0,
        "not_chinese": 0, "traditional": 0, "likely_cantonese": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    source_rows = rows if rows is not None else iter_dataset_rows()

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in source_rows:
                counters["seen"] += 1
                en, zh, reason = normalize_pair(row, converter)
                if reason:
                    counters[reason] += 1
                    continue
                stream.write(en + "\t" + zh + "\n")
                counters["written"] += 1
                if limit and counters["written"] >= limit:
                    break

        if counters["written"] < 100:
            raise RuntimeError(
                f"有效英中简体句对只有 {counters['written']} 条；"
                "拒绝生成不足以评测的语料。"
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "dataset": DATASET_ID,
        "source": "OPUS OpenSubtitles v2024 via Hugging Face dataset mirror",
        "language_pair": "en-zh-CN",
        "script_policy": "Simplified Chinese only; rows changed by OpenCC t2s are rejected",
        "dialect_policy": "heuristic written-Cantonese marker filter; not a complete dialect classifier",
        **counters,
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/opensubtitles-en-zh-cn.tsv")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum accepted pairs; 0 streams the full dataset")
    args = parser.parse_args()
    report = download_pairs(Path(args.output), args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
