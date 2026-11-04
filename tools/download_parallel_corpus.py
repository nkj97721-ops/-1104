#!/usr/bin/env python3
"""Download paired English–Chinese sentences from WMT News Commentary v16."""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.request import Request, urlopen

DATASET_NAME = "WMT News Commentary v16"
DATASET_URL = (
    "https://data.statmt.org/news-commentary/v16/training/"
    "news-commentary-v16.en-zh.tsv.gz"
)
CANTONESE_MARKERS = re.compile(r"[嘅咗喺冇唔咩噉嗰喎啱嚟乜係佢]")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")


def parse_parallel_line(line: str) -> dict[str, str]:
    """Parse one source-preserving TSV record; never zip independent files."""
    line = line.rstrip("\r\n")
    if "\t" not in line:
        return {"source": "", "target": ""}
    source, target = line.split("\t", 1)
    return {"source": source.lstrip("\ufeff"), "target": target}


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
    if traditional_to_simplified.convert(zh) != zh:
        return "", "", "traditional"
    if CANTONESE_MARKERS.search(zh):
        return "", "", "likely_cantonese"
    en = re.sub(r"\s+", " ", en).replace("\t", " ")
    zh = re.sub(r"\s+", " ", zh).replace("\t", " ")
    return en, zh, None


def iter_dataset_rows() -> Iterator[dict[str, str]]:
    """Stream the official paired TSV directly from WMT's gzip distribution."""
    request = Request(
        DATASET_URL,
        headers={"User-Agent": "TwinLine-Corpus-Benchmark/1.0"},
    )
    with urlopen(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response, mode="rb") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_stream:
                for line in text_stream:
                    yield parse_parallel_line(line)


def download_pairs(
    output: Path,
    limit: int = 0,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
        "seen": 0, "accepted": 0, "written": 0, "empty": 0,
        "not_english": 0, "not_chinese": 0, "traditional": 0,
        "likely_cantonese": 0,
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
                counters["accepted"] += 1
                if not limit or counters["written"] < limit:
                    stream.write(en + "\t" + zh + "\n")
                    counters["written"] += 1

        if counters["written"] < 100:
            raise RuntimeError(
                f"有效英中简体句对只有 {counters['written']} 条；拒绝生成不足以评测的语料。"
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "dataset": DATASET_NAME,
        "source_url": DATASET_URL,
        "language_pair": "en-zh",
        "script_policy": "Simplified Chinese only; rows changed by OpenCC t2s are rejected",
        "dialect_policy": "heuristic written-Cantonese marker filter; not a complete dialect classifier",
        "limit": limit,
        **counters,
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/news-commentary-en-zh.tsv")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum accepted pairs to write; 0 writes all accepted pairs")
    args = parser.parse_args()
    report = download_pairs(Path(args.output), args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
