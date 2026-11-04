#!/usr/bin/env python3
"""Evaluate document sentence alignment against a real parallel EN-ZH corpus.

Input TSV must contain one gold-aligned English/Chinese sentence pair per line.
Contiguous rows are grouped into blocks; blocks, not individual sentences, are
split into calibration and held-out partitions to prevent leakage.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tools.sentence_align import align_sentence_embeddings, encode_sentences


def load_pairs(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    pairs = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t", 1)
        if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
            continue
        pairs.append((fields[0].strip(), fields[1].strip()))
        if limit and len(pairs) >= limit:
            break
    if len(pairs) < 100:
        raise ValueError(f"有效平行句对只有 {len(pairs)} 条；至少需要 100 条。")
    return pairs


def make_blocks(pairs: list[tuple[str, str]], block_size: int) -> list[list[tuple[str, str]]]:
    return [pairs[i:i + block_size] for i in range(0, len(pairs), block_size)
            if len(pairs[i:i + block_size]) >= max(5, block_size // 2)]


def evaluate_block(block: list[tuple[str, str]], vectors: list[list[float]],
                   threshold: float) -> dict[str, Any]:
    n = len(block)
    result = align_sentence_embeddings(vectors[:n], vectors[n:], threshold=threshold)
    counts: Counter[str] = Counter()
    exact = 0
    for match in result["matches"]:
        a, b = len(match["source_indices"]), len(match["target_indices"])
        if a == 1 and b == 1:
            if match["source_indices"][0] == match["target_indices"][0]:
                exact += 1
                counts["correct_1_to_1"] += 1
            else:
                counts["wrong_1_to_1"] += 1
        elif a == 1 and b > 1:
            counts["merge_1_to_n"] += 1
        elif a > 1 and b == 1:
            counts["merge_n_to_1"] += 1
        else:
            counts["other_grouping"] += 1
    counts["missed_source"] = len(result["unmatched_source"])
    counts["missed_target"] = len(result["unmatched_target"])
    predicted = len(result["matches"])
    precision = exact / predicted if predicted else 0.0
    recall = exact / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sentence_count": n, "exact_pairs": exact,
        "precision": precision, "recall": recall, "f1": f1,
        "errors": dict(counts), "objective_score": result["objective_score"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(x["sentence_count"] for x in rows)
    exact = sum(x["exact_pairs"] for x in rows)
    predicted = sum(sum(v for k, v in x["errors"].items()
                        if k in ("correct_1_to_1", "wrong_1_to_1", "merge_1_to_n",
                                 "merge_n_to_1", "other_grouping")) for x in rows)
    p = exact / predicted if predicted else 0.0
    r = exact / total_n if total_n else 0.0
    return {
        "blocks": len(rows), "gold_pairs": total_n, "exact_pairs": exact,
        "precision": round(p, 6), "recall": round(r, 6),
        "f1": round(2*p*r/(p+r), 6) if p+r else 0.0,
        "errors": dict(sum((Counter(x["errors"]) for x in rows), Counter())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="UTF-8 TSV: English<TAB>Chinese, one gold pair per row")
    ap.add_argument("--output", default="alignment-benchmark.json")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--seed", type=int, default=1104)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.input), args.limit)
    blocks = make_blocks(pairs, args.block_size)
    if len(blocks) < 5:
        raise ValueError("至少需要 5 个完整测试块，建议准备 500 条以上平行句对。")
    order = list(range(len(blocks)))
    random.Random(args.seed).shuffle(order)
    dev_count = max(1, round(len(order) * 0.2))
    dev_ids, test_ids = set(order[:dev_count]), set(order[dev_count:])
    if not test_ids:
        raise ValueError("测试集为空；请增加语料或减小 block-size。")

    # Load the transformer once. Keep block boundaries intact during scoring.
    all_texts = [x[0] for x in pairs] + [x[1] for x in pairs]
    all_vectors = encode_sentences(all_texts, args.model, args.no_cuda)
    pair_count = len(pairs)
    encoded: dict[int, list[list[float]]] = {}
    offset = 0
    for index, block in enumerate(blocks):
        size = len(block)
        encoded[index] = (
            all_vectors[offset:offset + size]
            + all_vectors[pair_count + offset:pair_count + offset + size]
        )
        offset += size

    thresholds = sorted({float(x) for x in args.thresholds.split(",")})
    if any(not 0.0 <= x <= 1.0 for x in thresholds):
        raise ValueError("阈值必须位于 0 到 1 之间。")
    calibration = []
    for threshold in thresholds:
        rows = [evaluate_block(blocks[i], encoded[i], threshold) for i in sorted(dev_ids)]
        calibration.append({"threshold": threshold, **aggregate(rows)})
    # Maximize exact-match F1 on calibration only; tie-break toward higher
    # precision (conservative auto-pairing), then higher threshold.
    chosen = max(calibration, key=lambda x: (x["f1"], x["precision"], x["threshold"]))
    selected_threshold = chosen["threshold"]
    heldout_rows = [
        evaluate_block(blocks[i], encoded[i], selected_threshold) for i in sorted(test_ids)
    ]
    report = {
        "corpus": str(Path(args.input)),
        "corpus_sentence_pairs": len(pairs),
        "block_size": args.block_size,
        "model": args.model,
        "split": {"seed": args.seed, "dev_blocks": len(dev_ids), "test_blocks": len(test_ids)},
        "calibration": calibration,
        "selected_threshold": selected_threshold,
        "heldout_test": aggregate(heldout_rows),
        "heldout_block_details": heldout_rows,
        "methodology": (
            "Gold alignment is the original parallel-corpus row pairing. Contiguous blocks "
            "are split before threshold selection. The held-out test set is not used for "
            "threshold selection. Exact 1:1 links count as correct; wrong links, 1:n/n:1 "
            "groups, and unmatched sentences are separately reported. This benchmark "
            "measures sentence alignment, not word/phrase translation correctness."
        ),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_threshold": selected_threshold,
        "calibration": {k: chosen[k] for k in ("precision", "recall", "f1", "errors")},
        "heldout_test": report["heldout_test"],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
