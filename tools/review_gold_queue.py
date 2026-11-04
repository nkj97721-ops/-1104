#!/usr/bin/env python3
"""Human review CLI: turn OpenSubtitles silver proposals into verified gold JSONL."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def validate_groups(source: list[str], target: list[str], groups: list[dict[str, Any]]) -> None:
    used_s, used_t = set(), set()
    last_s = last_t = -1
    for number, group in enumerate(groups, 1):
        si, ti = group.get("source"), group.get("target")
        if not isinstance(si, list) or not isinstance(ti, list) or not si or not ti:
            raise ValueError(f"group {number}: source/target must be non-empty index arrays")
        if (len(si), len(ti)) not in ((1, 1), (1, 2), (2, 1), (2, 2)):
            raise ValueError(f"group {number}: supported shapes are 1:1, 1:2, 2:1, 2:2")
        if any(type(i) is not int or i < 0 or i >= len(source) for i in si):
            raise ValueError(f"group {number}: source index out of range")
        if any(type(i) is not int or i < 0 or i >= len(target) for i in ti):
            raise ValueError(f"group {number}: target index out of range")
        if si != list(range(si[0], si[0] + len(si))) or ti != list(range(ti[0], ti[0] + len(ti))):
            raise ValueError(f"group {number}: indices must be contiguous")
        if min(si) <= last_s or min(ti) <= last_t:
            raise ValueError(f"group {number}: groups must be strictly monotonic")
        if used_s.intersection(si) or used_t.intersection(ti):
            raise ValueError(f"group {number}: sentence index reused")
        used_s.update(si)
        used_t.update(ti)
        last_s, last_t = max(si), max(ti)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/gold_alignment_human.jsonl")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 means review all remaining documents")
    args = ap.parse_args()

    source_path, output_path = Path(args.input), Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    reviewed = skipped = 0
    with source_path.open(encoding="utf-8-sig") as source_file, output_path.open("a", encoding="utf-8") as output_file:
        for line in source_file:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in done:
                continue
            print("\n" + "=" * 88)
            print(f'{item["id"]} | movie={item["movie_id"]} | split={item["split"]}')
            print("\nENGLISH")
            for i, sentence in enumerate(item["source"]):
                print(f"  S{i}: {sentence}")
            print("\n简体中文")
            for i, sentence in enumerate(item["target"]):
                print(f"  T{i}: {sentence}")
            print("\n自动字幕时间对齐建议（仅供参考，不是金标准）")
            print(json.dumps(item.get("proposed_groups", []), ensure_ascii=False, indent=2))
            print("\n输入 a=逐条检查后接受建议；e=手工输入修正后的 groups JSON；s=跳过")
            choice = input("review> ").strip().lower()
            if choice == "s":
                skipped += 1
                continue
            if choice == "a":
                groups = [
                    {"source": g["source"], "target": g["target"]}
                    for g in item.get("proposed_groups", [])
                ]
            elif choice == "e":
                raw = input('groups JSON，例如 [{"source":[0],"target":[0,1]}]> ').strip()
                try:
                    groups = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"JSON 错误，跳过此文档：{exc}")
                    skipped += 1
                    continue
            else:
                print("无效选项，跳过。")
                skipped += 1
                continue

            try:
                validate_groups(item["source"], item["target"], groups)
            except (ValueError, TypeError, KeyError) as exc:
                print(f"标注未通过结构校验，跳过：{exc}")
                skipped += 1
                continue

            record = {
                "id": item["id"],
                "movie_id": item["movie_id"],
                "split": item["split"],
                "source": item["source"],
                "target": item["target"],
                "gold": groups,
                "annotation_status": "human_verified",
                "annotator": args.annotator,
                "reviewed_at": datetime.now().astimezone().isoformat(),
                "reference_source": "OpenSubtitles2024; manually reviewed against original English and zh_CN subtitle text",
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            done.add(item["id"])
            reviewed += 1
            print(f"已保存人工核验记录：{item['id']}")
            if args.limit and reviewed >= args.limit:
                break

    print(json.dumps({
        "reviewed_this_run": reviewed,
        "skipped_this_run": skipped,
        "total_verified_records": len(done),
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
