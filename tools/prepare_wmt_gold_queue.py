#!/usr/bin/env python3
"""Build automatic silver-reference data from WMT News Commentary v16 and WMT17 EN-ZH.

Sentence boundaries are detected independently on both sides. Original parallel
segments are weak silver references, not human-verified semantic gold.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from download_parallel_corpus import iter_dataset_rows, normalize_pair
from download_wmt17_test import (
    ARCHIVE_URL,
    join_segment_pairs,
    parse_sgm_segments,
    read_archive_members,
)
from prepare_opensubtitles2024_gold import split_aligned_lines


def make_record(doc_id: str, movie_id: str, split: str,
                pairs: list[tuple[str, str]], max_lines: int) -> list[dict[str, Any]]:
    records = []
    for chunk_no, start in enumerate(range(0, len(pairs), max_lines)):
        chunk = pairs[start:start + max_lines]
        source_lines = [x[0] for x in chunk]
        target_lines = [x[1] for x in chunk]
        source, target, groups, unmatched_s, unmatched_t, oversize = split_aligned_lines(
            source_lines, target_lines
        )
        records.append({
            "id": f"{doc_id}-{chunk_no:04d}",
            "movie_id": movie_id,
            "split": split,
            "source": source,
            "target": target,
            "proposed_groups": groups,
            "proposed_unmatched_source": unmatched_s,
            "proposed_unmatched_target": unmatched_t,
            "oversize_proposed_groups": oversize,
            "reference_status": "automatic_silver_reference",
            "reference_provenance": "WMT parallel segment IDs joined by original document/segment keys",
            "reference_notes": (
                "Original WMT segment pair is a weak silver reference. "
                "Sentence boundaries are detected independently; no manual verification is "
                "claimed."
            ),
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="data/wmt-silver")
    ap.add_argument("--calibration-documents", type=int, default=20)
    ap.add_argument("--max-segments-per-document", type=int, default=40)
    args = ap.parse_args()
    if args.calibration_documents < 1 or args.max_segments_per_document < 1:
        raise ValueError("Document and segment limits must be positive")

    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise SystemExit("Install opencc-python-reimplemented first") from exc
    converter = OpenCC("t2s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / "annotation_queue.jsonl"
    records: list[dict[str, Any]] = []
    calibration_counts = Counter()
    accepted_pairs: list[tuple[str, str]] = []

    # WMT News Commentary v16: use contiguous source-preserving records as
    # calibration blocks. No separate EN/ZH files are ever zipped by line.
    for row in iter_dataset_rows():
        en, zh, reason = normalize_pair(row, converter)
        if reason:
            calibration_counts[reason] += 1
            continue
        accepted_pairs.append((en, zh))
        if len(accepted_pairs) >= args.calibration_documents * args.max_segments_per_document:
            break
    if len(accepted_pairs) < args.max_segments_per_document:
        raise RuntimeError("Too few accepted News Commentary pairs for calibration queue")
    for block_no, start in enumerate(range(0, len(accepted_pairs), args.max_segments_per_document)):
        if block_no >= args.calibration_documents:
            break
        chunk = accepted_pairs[start:start + args.max_segments_per_document]
        records.extend(make_record(
            f"news-commentary-v16-{block_no:04d}",
            f"news-commentary-v16-block-{block_no:04d}",
            "calibration",
            chunk,
            args.max_segments_per_document,
        ))

    # WMT17 official test: preserve document and segment order after joining
    # source/reference by their original (docid, segment-id) keys.
    source_sgm, target_sgm = read_archive_members()
    source = parse_sgm_segments(source_sgm)
    target = parse_sgm_segments(target_sgm)
    matched = join_segment_pairs(source, target)
    by_document: dict[str, list[tuple[str, str]]] = {}
    test_counts = Counter()
    for (doc_id, segment_id), en_raw, zh_raw in matched:
        en, zh, reason = normalize_pair({"source": en_raw, "target": zh_raw}, converter)
        if reason:
            test_counts[reason] += 1
            continue
        by_document.setdefault(doc_id, []).append((en, zh))
    for doc_id in sorted(by_document):
        records.extend(make_record(
            f"wmt17-{doc_id}",
            f"wmt17-document-{doc_id}",
            "test",
            by_document[doc_id],
            args.max_segments_per_document,
        ))

    if not any(x["split"] == "calibration" for x in records):
        raise RuntimeError("No calibration review records were generated")
    if not any(x["split"] == "test" for x in records):
        raise RuntimeError("No WMT17 test review records were generated")

    group_types = Counter()
    oversize = 0
    with queue_path.open("w", encoding="utf-8") as stream:
        for record in records:
            for group in record["proposed_groups"]:
                group_types[group["suggested_type"]] += 1
            oversize += record["oversize_proposed_groups"]
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "calibration_dataset": "WMT News Commentary v16 EN-ZH",
        "calibration_source": "https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.en-zh.tsv.gz",
        "test_dataset": "WMT17 official EN-ZH test set, improved reference release",
        "test_source": ARCHIVE_URL,
        "test_pairing": "source/reference joined by original (docid, segment id)",
        "languages": {"source": "English", "target": "Simplified Chinese only"},
        "simplified_policy": "OpenCC t2s; reject target text changed by conversion",
        "dialect_policy": "heuristic written-Cantonese marker filter inherited from download_parallel_corpus",
        "reference_policy": "Original segment pair is a weak silver reference; no manual review is required or claimed.",
        "split_policy": "News Commentary calibration; separate WMT17 documents held out for test.",
        "calibration_filter_counts": dict(calibration_counts),
        "test_filter_counts": dict(test_counts),
        "records": len(records),
        "calibration_records": sum(x["split"] == "calibration" for x in records),
        "test_records": sum(x["split"] == "test" for x in records),
        "proposed_group_types": dict(group_types),
        "oversize_proposed_groups": oversize,
        "silver_reference_corpus": str(queue_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
