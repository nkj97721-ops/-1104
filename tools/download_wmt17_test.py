#!/usr/bin/env python3
"""Build an independently sourced WMT17 EN→ZH test TSV from SGML segment IDs."""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from download_parallel_corpus import normalize_pair

ARCHIVE_URL = "https://data.statmt.org/wmt17/translation-task/test-update-1.tgz"
SOURCE_FILE = "newstest2017-enzh-src.en.sgm"
TARGET_FILE = "newstest2017-enzh-ref.zh.sgm"
DOC_RE = re.compile(r"<doc\b([^>]*)>(.*?)</doc>", re.IGNORECASE | re.DOTALL)
SEG_RE = re.compile(r"<seg\b([^>]*)>(.*?)</seg>", re.IGNORECASE | re.DOTALL)
DOC_ID_RE = re.compile(r'\bdocid=["\']([^"\']+)["\']', re.IGNORECASE)
SEG_ID_RE = re.compile(r'\bid=["\']([^"\']+)["\']', re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def parse_sgm_segments(content: str) -> dict[tuple[str, str], str]:
    """Parse SGML segments keyed by (document ID, segment ID)."""
    segments: dict[tuple[str, str], str] = {}
    for doc_index, doc_match in enumerate(DOC_RE.finditer(content)):
        doc_attributes, doc_body = doc_match.groups()
        doc_id_match = DOC_ID_RE.search(doc_attributes)
        doc_id = doc_id_match.group(1) if doc_id_match else str(doc_index)
        for seg_match in SEG_RE.finditer(doc_body):
            attributes, raw_text = seg_match.groups()
            seg_id_match = SEG_ID_RE.search(attributes)
            if not seg_id_match:
                raise ValueError(f"SGM segment in document {doc_id} has no id.")
            seg_id = seg_id_match.group(1)
            text = html.unescape(TAG_RE.sub("", raw_text)).strip()
            key = (doc_id, seg_id)
            if key in segments:
                raise ValueError(f"Duplicate SGM segment ID: {key}")
            if text:
                segments[key] = re.sub(r"\s+", " ", text)
    if not segments:
        raise ValueError("No SGM segments were parsed.")
    return segments


def join_segment_pairs(
    source: dict[tuple[str, str], str],
    target: dict[tuple[str, str], str],
) -> list[tuple[tuple[str, str], str, str]]:
    if source.keys() != target.keys():
        missing_target = sorted(source.keys() - target.keys())[:10]
        missing_source = sorted(target.keys() - source.keys())[:10]
        raise RuntimeError(
            "WMT17 source/reference segment IDs do not match; refusing index-based pairing. "
            f"Missing target IDs: {missing_target}; missing source IDs: {missing_source}"
        )
    # Dict insertion order preserves the original document and segment order.
    return [(key, source[key], target[key]) for key in source]


def read_archive_members() -> tuple[str, str]:
    request = Request(ARCHIVE_URL, headers={"User-Agent": "TwinLine-WMT17-Test/1.0"})
    with urlopen(request, timeout=120) as response, tempfile.TemporaryFile() as archive_file:
        shutil.copyfileobj(response, archive_file)
        archive_file.seek(0)
        with tarfile.open(fileobj=archive_file, mode="r:gz") as archive:
            members = {Path(member.name).name: member for member in archive.getmembers()}
            if SOURCE_FILE not in members or TARGET_FILE not in members:
                raise RuntimeError(
                    "WMT17 archive is missing expected EN-ZH SGML files: "
                    f"{sorted(members)}"
                )
            source_stream = archive.extractfile(members[SOURCE_FILE])
            target_stream = archive.extractfile(members[TARGET_FILE])
            if source_stream is None or target_stream is None:
                raise RuntimeError("Unable to read WMT17 SGML members.")
            return (
                source_stream.read().decode("utf-8"),
                target_stream.read().decode("utf-8"),
            )


def download_test_pairs(output: Path) -> dict[str, Any]:
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise SystemExit(
            "缺少简繁检测依赖：请执行 pip install opencc-python-reimplemented"
        ) from exc

    source_text, target_text = read_archive_members()
    source = parse_sgm_segments(source_text)
    target = parse_sgm_segments(target_text)
    matched_pairs = join_segment_pairs(source, target)
    converter = OpenCC("t2s")
    counters = {
        "matched_segment_ids": len(source),
        "documents": len({doc_id for doc_id, _ in source}),
        "written": 0,
        "empty": 0,
        "not_english": 0,
        "not_chinese": 0,
        "traditional": 0,
        "likely_cantonese": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for key, source_text, target_text in matched_pairs:
                en, zh, reason = normalize_pair(
                    {"source": source_text, "target": target_text}, converter
                )
                if reason:
                    counters[reason] += 1
                    continue
                stream.write(en + "\t" + zh + "\n")
                counters["written"] += 1
        if counters["written"] < 100:
            raise RuntimeError(f"Only {counters['written']} valid WMT17 pairs were produced.")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "dataset": "WMT17 official EN-ZH test set, improved reference release",
        "source_url": ARCHIVE_URL,
        "pairing": "source and reference joined by (docid, segment id), never by line number",
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
    parser.add_argument("--output", default="data/wmt17-en-zh-test.tsv")
    args = parser.parse_args()
    print(json.dumps(download_test_pairs(Path(args.output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
