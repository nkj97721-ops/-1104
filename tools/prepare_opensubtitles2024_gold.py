#!/usr/bin/env python3
"""Download OpenSubtitles2024 EN/zh_CN and build an automatic silver-reference corpus.

The upstream links are subtitle-timing alignments and remain weak silver references.
No human review is required or implied by this pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from .sentence_align import split_sentences
except ImportError:
    from sentence_align import split_sentences

SOURCE_URL = (
    "https://raw.githubusercontent.com/Helsinki-NLP/OpenSubtitles-devtest/"
    "refs/heads/master/devtest-xml/multi-linksets-0.9/2024/"
    "ar-bg-cs-da-de-el-en-es-et-fa-fi-fr-he-hi-hr-hu-id-it-ko-lt-lv-ms-nl-no-pl-"
    "pt-pt_BR-ro-ru-sk-sl-sr-sv-ta-te-tr-uk-vi-zh_CN-zh_TW.zip"
)
EXPECTED_GIT_BLOB_SHA = "2ddf63137c40cd0e479e9e46c7a4367eb15b21e7"
ZIP_PASSWORD = b"OpenSubtitles2024-multiset"
LANG_SUFFIX = re.compile(r"^(?P<movie>.+?)[._-](?P<lang>zh_CN|en)(?:\.txt)?$", re.I)


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        data = destination.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "sentence-alignment-research/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read()
        destination.write_bytes(data)
    actual = git_blob_sha(data)
    if actual != EXPECTED_GIT_BLOB_SHA:
        raise RuntimeError(
            f"Upstream archive integrity mismatch: expected Git blob {EXPECTED_GIT_BLOB_SHA}, got {actual}"
        )


def read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        return zf.read(info, pwd=ZIP_PASSWORD)
    except RuntimeError:
        return zf.read(info)


def decode_text(data: bytes, member_name: str) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Cannot decode UTF-8 subtitle member: {member_name}")


def identify_text_member(name: str) -> tuple[str, str] | None:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    match = LANG_SUFFIX.match(base)
    if match:
        return match.group("movie"), match.group("lang")
    # Also accept a language-named file inside a movie directory.
    if base in ("en", "en.txt", "zh_CN", "zh_CN.txt"):
        movie = name.replace("\\", "/").split("/")[-2]
        return movie, base.split(".")[0]
    return None


def split_aligned_lines(source_lines: list[str], target_lines: list[str]) -> tuple[list[str], list[str], list[dict], list[int], list[int], int]:
    if len(source_lines) != len(target_lines):
        raise ValueError(
            f"Aligned EN/zh_CN files have different line counts: {len(source_lines)} != {len(target_lines)}"
        )
    source_sentences: list[str] = []
    target_sentences: list[str] = []
    proposed_groups: list[dict] = []
    unmatched_source: list[int] = []
    unmatched_target: list[int] = []
    oversize = 0

    for line_index, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):
        src = split_sentences(source_line.strip())
        trg = split_sentences(target_line.strip())
        src_indices = list(range(len(source_sentences), len(source_sentences) + len(src)))
        trg_indices = list(range(len(target_sentences), len(target_sentences) + len(trg)))
        source_sentences.extend(src)
        target_sentences.extend(trg)

        if src_indices and trg_indices:
            proposed_groups.append({
                "source": src_indices,
                "target": trg_indices,
                "alignment_line": line_index,
                "suggested_type": f"{len(src_indices)}:{len(trg_indices)}",
            })
            if len(src_indices) > 2 or len(trg_indices) > 2:
                oversize += 1
        elif src_indices:
            unmatched_source.extend(src_indices)
        elif trg_indices:
            unmatched_target.extend(trg_indices)

    return source_sentences, target_sentences, proposed_groups, unmatched_source, unmatched_target, oversize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="data/opensubtitles2024-silver")
    ap.add_argument("--max-lines-per-document", type=int, default=80)
    args = ap.parse_args()
    if args.max_lines_per_document < 1:
        raise ValueError("--max-lines-per-document must be positive")

    out_dir = Path(args.output_dir)
    archive_path = out_dir / "source" / "opensubtitles2024-multiset.zip"
    download(SOURCE_URL, archive_path)

    by_movie: dict[str, dict[str, list[str]]] = {}
    member_names = []
    with zipfile.ZipFile(archive_path) as zf:
        # Members are password-protected. CRC is verified by ZipFile.read
        # after read_member supplies the upstream password.
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_names.append(info.filename)
            identified = identify_text_member(info.filename)
            if not identified:
                continue
            movie_id, language = identified
            if language not in ("en", "zh_CN"):
                continue
            text = decode_text(read_member(zf, info), info.filename)
            by_movie.setdefault(movie_id, {})[language] = text.splitlines()

    complete = {
        movie: pair for movie, pair in by_movie.items()
        if "en" in pair and "zh_CN" in pair
    }
    if not complete:
        sample = "\n".join(member_names[:80])
        raise RuntimeError(
            "No paired movie text files named with .en and .zh_CN were found. "
            "Archive member sample:\n" + sample
        )

    # Split by whole movie, never by neighboring chunks from the same subtitle.
    movie_ids = sorted(complete)
    test_count = max(1, round(len(movie_ids) * 0.25))
    test_movies = set(movie_ids[-test_count:])
    queue_path = out_dir / "annotation_queue.jsonl"
    manifest = {
        "dataset": "OpenSubtitles2024 multilingual multiset, alignment threshold 0.9",
        "source_url": SOURCE_URL,
        "source_git_blob_sha1": EXPECTED_GIT_BLOB_SHA,
        "languages": ["en", "zh_CN"],
        "excluded_languages": ["zh_TW", "yue", "all other language variants"],
        "alignment_reference": "automatic subtitle-timing alignment; weak silver reference only, not human-verified semantic gold",
        "split_policy": "movie-level holdout; final sorted 25% of movie IDs are test, all other movies calibration",
        "max_lines_per_document": args.max_lines_per_document,
        "movies": {},
        "counts": {
            "documents": 0, "source_sentences": 0, "target_sentences": 0,
            "proposed_groups": 0, "oversize_groups": 0,
            "proposed_group_types": {},
        },
    }
    total_types = Counter()
    with queue_path.open("w", encoding="utf-8") as writer:
        for movie_id in movie_ids:
            pair = complete[movie_id]
            if len(pair["en"]) != len(pair["zh_CN"]):
                raise ValueError(
                    f"{movie_id}: EN/zh_CN line count mismatch "
                    f"{len(pair['en'])} != {len(pair['zh_CN'])}"
                )
            split = "test" if movie_id in test_movies else "calibration"
            movie_stats = Counter()
            lines = len(pair["en"])
            for chunk_no, start in enumerate(range(0, lines, args.max_lines_per_document)):
                end = min(lines, start + args.max_lines_per_document)
                source, target, groups, unmatched_s, unmatched_t, oversize = split_aligned_lines(
                    pair["en"][start:end], pair["zh_CN"][start:end]
                )
                record = {
                    "id": f"opensubtitles2024-{movie_id}-{chunk_no:04d}",
                    "movie_id": movie_id,
                    "split": split,
                    "source": source,
                    "target": target,
                    "proposed_groups": groups,
                    "proposed_unmatched_source": unmatched_s,
                    "proposed_unmatched_target": unmatched_t,
                    "reference_status": "automatic_silver_reference",
                    "reference_provenance": "OpenSubtitles2024 upstream subtitle-timing alignment",
                    "reference_notes": (
                        "Original aligned subtitle lines are weak silver references. Sentence groups are "
                        "derived by independently splitting both sides; no manual verification is claimed. "
                        "The upstream alignment is not verified semantic gold."
                    ),
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                movie_stats["documents"] += 1
                movie_stats["source_sentences"] += len(source)
                movie_stats["target_sentences"] += len(target)
                movie_stats["proposed_groups"] += len(groups)
                movie_stats["oversize_groups"] += oversize
                total_types.update(g["suggested_type"] for g in groups)
            manifest["movies"][movie_id] = {
                "split": split,
                "aligned_lines": lines,
                **dict(movie_stats),
            }
            for key in ("documents", "source_sentences", "target_sentences", "proposed_groups", "oversize_groups"):
                manifest["counts"][key] += movie_stats[key]
    manifest["counts"]["proposed_group_types"] = dict(total_types)
    manifest["counts"]["movies"] = len(movie_ids)
    manifest["counts"]["test_movies"] = sorted(test_movies)
    manifest["counts"]["calibration_movies"] = sorted(set(movie_ids) - test_movies)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "archive": str(archive_path),
        "silver_reference_corpus": str(queue_path),
        "manifest": str(out_dir / "manifest.json"),
        "counts": manifest["counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
