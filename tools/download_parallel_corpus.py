#!/usr/bin/env python3
"""Fetch the OPUS TED2020 English-Chinese parallel corpus as a local TSV.

The archive is not committed to this repository. Check the corpus terms and
attribution requirements before redistribution or commercial use.
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


URL = "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-zh.txt.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/ted2020-en-zh.tsv")
    ap.add_argument("--archive", default="data/ted2020-en-zh.zip")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, archive)

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        en_files = [n for n in names if n.endswith(".en")]
        zh_files = [n for n in names if n.endswith(".zh")]
        matches = []
        for en in en_files:
            prefix = en[:-3]
            zh = prefix + ".zh"
            if zh in zh_files:
                matches.append((en, zh))
        if not matches:
            raise RuntimeError("压缩包内找不到同名前缀的 .en / .zh 文件；请检查 OPUS 文件结构。")
        en_name, zh_name = matches[0]
        english = zf.read(en_name).decode("utf-8-sig").splitlines()
        chinese = zf.read(zh_name).decode("utf-8-sig").splitlines()

    if len(english) != len(chinese):
        raise RuntimeError(f"平行文件行数不一致：EN={len(english)}，ZH={len(chinese)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for en, zh in zip(english, chinese):
            en, zh = en.strip(), zh.strip()
            if not en or not zh:
                continue
            stream.write(en.replace("\t", " ") + "\t" + zh.replace("\t", " ") + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(f"Saved {count} parallel sentence pairs to {output}")
    print("Source: OPUS TED2020; retain corpus attribution and verify current license terms.")


if __name__ == "__main__":
    main()
