#!/usr/bin/env python3
"""Download paired EN–Simplified-Chinese rows from the OPUS OpenSubtitles mirror.

Uses the Hugging Face Dataset Viewer JSON API instead of the datasets/pyarrow
runtime. Pages are fetched concurrently but yielded in source order, and each
row carries its own source/target pair.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET_ID = "FradSer/OpenSubtitles-en-zh-cn-20m"
DATASET_VIEWER = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100
PREFETCH_PAGES = 8
CANTONESE_MARKERS = re.compile(r"[嘅咗喺冇唔咩噉嗰喎啱嚟乜係佢]")
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
    if traditional_to_simplified.convert(zh) != zh:
        return "", "", "traditional"
    if CANTONESE_MARKERS.search(zh):
        return "", "", "likely_cantonese"
    en = re.sub(r"\s+", " ", en).replace("\t", " ")
    zh = re.sub(r"\s+", " ", zh).replace("\t", " ")
    return en, zh, None


def request_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = DATASET_VIEWER + endpoint + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "TwinLine-Corpus-Benchmark/1.0"})
    last_error = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(f"Hugging Face dataset viewer error: {payload['error']}")
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Dataset Viewer request failed: {url}: {last_error}")


def iter_dataset_rows() -> Iterator[dict[str, Any]]:
    """Fetch 100-row API pages concurrently while preserving source order."""
    split_info = request_json("/splits", {"dataset": DATASET_ID})
    train_splits = [item for item in split_info.get("splits", []) if item.get("split") == "train"]
    if len(train_splits) != 1:
        raise RuntimeError(f"Expected one train split, found {len(train_splits)}: {train_splits}")
    config = train_splits[0]["config"]

    first = request_json("/rows", {
        "dataset": DATASET_ID, "config": config, "split": "train",
        "offset": 0, "length": PAGE_SIZE,
    })
    total_rows = int(first.get("num_rows_total", 0))
    if total_rows <= 0 or first.get("partial"):
        raise RuntimeError("Dataset Viewer did not return a complete, non-empty train split.")

    yield from _extract_page_rows(first, 0, min(PAGE_SIZE, total_rows))

    offsets = range(PAGE_SIZE, total_rows, PAGE_SIZE)
    with ThreadPoolExecutor(max_workers=PREFETCH_PAGES) as executor:
        pending_offsets: list[int] = []
        for offset in offsets:
            pending_offsets.append(offset)
            if len(pending_offsets) == PREFETCH_PAGES:
                yield from _fetch_pages_in_order(executor, pending_offsets, config, total_rows)
                pending_offsets.clear()
        if pending_offsets:
            yield from _fetch_pages_in_order(executor, pending_offsets, config, total_rows)


def _extract_page_rows(payload: dict[str, Any], offset: int, expected_count: int):
    items = payload.get("rows", [])
    if len(items) != expected_count:
        raise RuntimeError(
            f"Dataset Viewer page at offset {offset} is incomplete: "
            f"expected {expected_count} rows, received {len(items)}."
        )
    for index, item in enumerate(items):
        if int(item.get("row_idx", -1)) != offset + index:
            raise RuntimeError(
                f"Dataset Viewer row order mismatch at offset {offset + index}."
            )
        row = item.get("row")
        if not isinstance(row, dict) or not {"source", "target"}.issubset(row):
            raise RuntimeError("Dataset Viewer row is missing source/target pair fields.")
        yield row


def _fetch_pages_in_order(executor, offsets: list[int], config: str, total_rows: int):
    futures = [
        executor.submit(request_json, "/rows", {
            "dataset": DATASET_ID, "config": config, "split": "train",
            "offset": offset, "length": PAGE_SIZE,
        })
        for offset in offsets
    ]
    for offset, future in zip(offsets, futures):
        payload = future.result()
        if payload.get("partial"):
            raise RuntimeError("Dataset Viewer returned a partial page; refusing incomplete corpus.")
        expected_count = min(PAGE_SIZE, total_rows - offset)
        yield from _extract_page_rows(payload, offset, expected_count)


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
                f"有效英中简体句对只有 {counters['written']} 条；拒绝生成不足以评测的语料。"
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "dataset": DATASET_ID,
        "source": "OPUS OpenSubtitles v2024 via Hugging Face Dataset Viewer rows API",
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
