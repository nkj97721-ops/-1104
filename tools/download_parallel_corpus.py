#!/usr/bin/env python3
"""Download an English–Simplified-Chinese OPUS OpenSubtitles parallel corpus.

The target language is explicitly zh-CN (Simplified Chinese), never the
ambiguous zh code used by some corpora. Traditional-only rows are filtered
with OpenCC before the benchmark is written.
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-zh-CN.txt.zip"


def locate_pair(names: list[str]) -> tuple[str, str]:
    en_files = [n for n in names if n.lower().endswith(".en")]
    zh_files = [n for n in names if n.lower().endswith((".zh-cn", ".zh_cn"))]
    matches = []
    for en in en_files:
        stem = en[:-3]
        for zh in zh_files:
            if zh.rsplit(".", 1)[0] == stem:
                matches.append((en, zh))
    if not matches:
        raise RuntimeError(
            "压缩包内没有匹配的 .en / .zh-CN（或 .zh_CN）文件；"
            "拒绝回退到含义不明确的 .zh 数据。"
        )
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/opensubtitles-en-zh-cn.tsv")
    ap.add_argument("--archive", default="data/opensubtitles-en-zh-cn.zip")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise SystemExit(
            "缺少简繁检测依赖：请执行 pip install opencc-python-reimplemented"
        ) from exc
    traditional_to_simplified = OpenCC("t2s")

    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading Simplified Chinese corpus: {URL}")
        urllib.request.urlretrieve(URL, archive)

    with zipfile.ZipFile(archive) as zf:
        en_name, zh_name = locate_pair(zf.namelist())
        english = zf.read(en_name).decode("utf-8-sig").splitlines()
        chinese = zf.read(zh_name).decode("utf-8-sig").splitlines()

    if len(english) != len(chinese):
        raise RuntimeError(f"平行文件行数不一致：EN={len(english)}，ZH-CN={len(chinese)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    traditional_filtered = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for en, zh in zip(english, chinese):
            en, zh = en.strip(), zh.strip()
            if not en or not zh:
                continue
            # The corpus is labeled zh-CN; this additional content check
            # excludes rows containing traditional-only forms.
            if traditional_to_simplified.convert(zh) != zh:
                traditional_filtered += 1
                continue
            stream.write(en.replace("\t", " ") + "\t" + zh.replace("\t", " ") + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    if count < 100:
        raise RuntimeError(f"简体中文有效句对只有 {count} 条，无法进行可靠评测。")
    print(f"Saved {count} English–Simplified-Chinese pairs to {output}")
    print(f"Filtered rows containing traditional-only characters: {traditional_filtered}")
    print("Source: OPUS OpenSubtitles v2024, language pair en–zh-CN.")


if __name__ == "__main__":
    main()
