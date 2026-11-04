#!/usr/bin/env python3
"""Evaluate document sentence alignment against a real parallel EN-ZH corpus.

Input TSV must contain one gold-aligned English/Chinese sentence pair per line.
Contiguous rows are grouped into blocks; blocks, not individual sentences, are
split into calibration and held-out partitions to prevent leakage.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from sentence_align import align_sentence_embeddings, encode_sentences


def load_pairs(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    pairs = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t", 1)
        if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
            continue
        pair = (fields[0].strip(), fields[1].strip())
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
        if limit and len(pairs) >= limit:
            break
    if len(pairs) < 100:
        raise ValueError(f"有效平行句对只有 {len(pairs)} 条；至少需要 100 条。")
    return pairs


def make_blocks(pairs: list[tuple[str, str]], block_size: int) -> list[list[tuple[str, str]]]:
    # Use full blocks only so concatenated calibration/test vectors have no gaps.
    return [
        pairs[i:i + block_size]
        for i in range(0, len(pairs) - block_size + 1, block_size)
    ]


def build_similarity_tables(vectors: list[list[float]], n: int):
    """Vectorize all 1:1, 1:2 and 2:1 cosine scores once per block."""
    import numpy as np

    source = np.asarray(vectors[:n], dtype=np.float32)
    target = np.asarray(vectors[n:], dtype=np.float32)
    source = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    target_pairs = (target[:-1] + target[1:]) * 0.5
    source_pairs = (source[:-1] + source[1:]) * 0.5
    target_pairs = target_pairs / np.maximum(
        np.linalg.norm(target_pairs, axis=1, keepdims=True), 1e-12
    )
    source_pairs = source_pairs / np.maximum(
        np.linalg.norm(source_pairs, axis=1, keepdims=True), 1e-12
    )
    return (
        source @ target.T,
        source @ target_pairs.T,
        source_pairs @ target.T,
    )


def evaluate_block(block: list[tuple[str, str]], vectors: list[list[float]],
                   threshold: float, gap_penalty: float = 0.12,
                   merge_penalty: float = 0.06, similarity_tables=None) -> dict[str, Any]:
    n = len(block)
    result = align_sentence_embeddings(
        vectors[:n], vectors[n:], threshold=threshold,
        gap_penalty=gap_penalty, merge_penalty=merge_penalty,
        similarity_tables=similarity_tables,
    )
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
    examples = []
    for match in result["matches"]:
        si, ti = match["source_indices"], match["target_indices"]
        is_exact = len(si) == 1 and len(ti) == 1 and si[0] == ti[0]
        if not is_exact and len(examples) < 12:
            examples.append({
                "type": ("wrong_1_to_1" if len(si) == 1 and len(ti) == 1
                         else "merge_1_to_n" if len(si) == 1
                         else "merge_n_to_1" if len(ti) == 1 else "many_to_many"),
                "english": [block[k][0] for k in si],
                "predicted_chinese": [block[k][1] for k in ti],
                "expected_chinese": [block[k][1] for k in si],
                "similarity": match["similarity"],
            })
    for idx in result["unmatched_source"][:6]:
        examples.append({
            "type": "missed_english",
            "english": [block[idx][0]],
            "predicted_chinese": [],
            "expected_chinese": [block[idx][1]],
        })
    for idx in result["unmatched_target"][:6]:
        examples.append({
            "type": "missed_chinese",
            "english": [],
            "predicted_chinese": [block[idx][1]],
            "expected_chinese": [block[idx][1]],
        })
    precision = exact / predicted if predicted else 0.0
    recall = exact / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sentence_count": n, "exact_pairs": exact,
        "exact_match_accuracy": recall,
        "precision": precision, "recall": recall, "f1": f1,
        "errors": dict(counts), "objective_score": result["objective_score"],
        "examples": examples,
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
        "exact_match_accuracy": round(r, 6),
        "precision": round(p, 6), "recall": round(r, 6),
        "f1": round(2*p*r/(p+r), 6) if p+r else 0.0,
        "errors": dict(sum((Counter(x["errors"]) for x in rows), Counter())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Calibration TSV: English<TAB>Chinese, one gold pair per row")
    ap.add_argument("--test-input", help="Optional separate held-out TSV; if omitted, split --input contiguously")
    ap.add_argument("--output", default="alignment-benchmark.json")
    ap.add_argument(
        "--models",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/LaBSE",
        help="Comma-separated sentence-transformer candidates; selection uses calibration data only",
    )
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--gap-penalties", default="0.06,0.12,0.18,0.24")
    ap.add_argument("--merge-penalties", default="0.02,0.06,0.10,0.14")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    if args.block_size < 2:
        raise ValueError("--block-size 必须至少为 2。")
    calibration_pairs = load_pairs(Path(args.input), args.limit)
    if args.test_input:
        raw_test_pairs = load_pairs(Path(args.test_input))
        calibration_en = {en for en, _ in calibration_pairs}
        calibration_zh = {zh for _, zh in calibration_pairs}
        test_pairs = [
            pair for pair in raw_test_pairs
            if pair[0] not in calibration_en and pair[1] not in calibration_zh
        ]
        overlap_removed = len(raw_test_pairs) - len(test_pairs)
        dev_blocks = make_blocks(calibration_pairs, args.block_size)
        test_blocks = make_blocks(test_pairs, args.block_size)
        if len(dev_blocks) < 5 or len(test_blocks) < 5:
            raise ValueError("独立校准集和测试集都至少需要 5 个完整块。")
        used_calibration = len(dev_blocks) * args.block_size
        used_test = len(test_blocks) * args.block_size
        pairs = calibration_pairs[:used_calibration] + test_pairs[:used_test]
        blocks = dev_blocks + test_blocks
        dev_ids = set(range(len(dev_blocks)))
        test_ids = set(range(len(dev_blocks), len(blocks)))
        split_report = {
            "strategy": "separate_calibration_corpus_and_official_heldout_test",
            "calibration_source": str(Path(args.input)),
            "test_source": str(Path(args.test_input)),
            "exact_monolingual_overlap_removed": overlap_removed,
            "calibration_blocks": len(dev_ids),
            "test_blocks": len(test_ids),
            "document_independence": (
                "Calibration and held-out data come from separate WMT releases. "
                "The WMT17 test downloader matched source/reference segments by "
                "(document ID, segment ID); exact source/target sentence overlap "
                "with calibration data was removed."
            ),
        }
    else:
        pairs = calibration_pairs
        blocks = make_blocks(pairs, args.block_size)
        if len(blocks) < 5:
            raise ValueError("至少需要 5 个完整测试块，建议准备 500 条以上平行句对。")
        dev_count = max(1, round(len(blocks) * 0.2))
        dev_ids = set(range(dev_count))
        test_ids = set(range(dev_count, len(blocks)))
        if len(test_ids) < 5:
            raise ValueError("连续留出测试集至少需要 5 个完整块。")
        used_pairs = len(blocks) * args.block_size
        pairs = pairs[:used_pairs]
        split_report = {
            "strategy": "contiguous_prefix_calibration_suffix_heldout",
            "calibration_blocks": len(dev_ids),
            "test_blocks": len(test_ids),
            "document_independence": (
                "The source dataset does not expose stable document IDs; "
                "contiguous holdout reduces adjacent-block leakage but cannot prove "
                "document-level independence."
            ),
        }

    thresholds = sorted({float(x) for x in args.thresholds.split(",")})
    if not thresholds or any(not math.isfinite(x) or not -1.0 <= x <= 1.0 for x in thresholds):
        raise ValueError("阈值必须是 [-1, 1] 内的有限余弦相似度。")

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not model_names:
        raise ValueError("至少指定一个句向量模型。")
    all_texts = [x[0] for x in pairs] + [x[1] for x in pairs]
    model_calibrations = []
    import numpy as np
    model_vectors: dict[str, Any] = {}
    for model_name in model_names:
        # One model load/encoding pass per candidate model.
        all_vectors = encode_sentences(all_texts, model_name, args.no_cuda)
        model_vectors[model_name] = all_vectors
        pair_count = len(pairs)
        encoded: dict[int, Any] = {}
        offset = 0
        for index, block in enumerate(blocks):
            size = len(block)
            encoded[index] = np.concatenate((
                all_vectors[offset:offset + size],
                all_vectors[pair_count + offset:pair_count + offset + size],
            ), axis=0)
            offset += size
        gap_penalties = sorted({float(x) for x in args.gap_penalties.split(",")})
        merge_penalties = sorted({float(x) for x in args.merge_penalties.split(",")})
        if any(not math.isfinite(x) or x < 0 for x in gap_penalties + merge_penalties):
            raise ValueError("DP 惩罚参数必须是非负有限数。")
        # Cache vectorized cosine transition scores once; the parameter grid
        # then performs only lightweight dynamic-programming table lookups.
        similarity_cache = {
            i: build_similarity_tables(encoded[i], len(blocks[i]))
            for i in sorted(dev_ids | test_ids)
        }
        calibration = []
        for threshold in thresholds:
            for gap_penalty in gap_penalties:
                for merge_penalty in merge_penalties:
                    rows = [
                        evaluate_block(blocks[i], encoded[i], threshold,
                                       gap_penalty, merge_penalty,
                                       similarity_cache[i])
                        for i in sorted(dev_ids)
                    ]
                    calibration.append({
                        "threshold": threshold, "gap_penalty": gap_penalty,
                        "merge_penalty": merge_penalty, **aggregate(rows),
                    })
        best = max(calibration, key=lambda x: (
            x["f1"], x["precision"], x["recall"],
            x["threshold"], -x["gap_penalty"], -x["merge_penalty"],
        ))
        model_calibrations.append({
            "model": model_name, "parameter_grid": calibration, "best_on_dev": best,
        })

    # Select model + threshold on calibration only. Prefer precision on F1 ties.
    selected = max(
        model_calibrations,
        key=lambda x: (x["best_on_dev"]["f1"], x["best_on_dev"]["precision"],
                       x["best_on_dev"]["threshold"]),
    )
    selected_model = selected["model"]
    selected_threshold = selected["best_on_dev"]["threshold"]
    selected_gap_penalty = selected["best_on_dev"]["gap_penalty"]
    selected_merge_penalty = selected["best_on_dev"]["merge_penalty"]
    selected_vectors = model_vectors[selected_model]
    pair_count = len(pairs)
    selected_similarity_cache = {}
    offset_for_cache = 0
    for index, block in enumerate(blocks):
        size = len(block)
        if index in test_ids:
            block_vectors = np.concatenate((
                selected_vectors[offset_for_cache:offset_for_cache + size],
                selected_vectors[pair_count + offset_for_cache:pair_count + offset_for_cache + size],
            ), axis=0)
            selected_similarity_cache[index] = build_similarity_tables(block_vectors, size)
        offset_for_cache += size
    heldout_rows = []
    offset = 0
    for index, block in enumerate(blocks):
        size = len(block)
        if index in test_ids:
            block_vectors = np.concatenate((
                selected_vectors[offset:offset + size],
                selected_vectors[pair_count + offset:pair_count + offset + size],
            ), axis=0)
            heldout_rows.append(
                evaluate_block(block, block_vectors, selected_threshold,
                               selected_gap_penalty, selected_merge_penalty,
                               selected_similarity_cache[index])
            )
        offset += size
    # Evaluate every candidate model on the independent test set using only
    # parameters selected on calibration data. This prevents a single selected
    # model from hiding the held-out performance of the other candidate.
    heldout_model_comparisons = []
    for model_calibration in model_calibrations:
        model_name = model_calibration["model"]
        best = model_calibration["best_on_dev"]
        model_rows = []
        if model_name == selected_model:
            model_rows = heldout_rows
        else:
            model_vectors_for_test = model_vectors[model_name]
            offset = 0
            for index, block in enumerate(blocks):
                size = len(block)
                if index in test_ids:
                    block_vectors = np.concatenate((
                        model_vectors_for_test[offset:offset + size],
                        model_vectors_for_test[pair_count + offset:pair_count + offset + size],
                    ), axis=0)
                    tables = build_similarity_tables(block_vectors, size)
                    model_rows.append(evaluate_block(
                        block, block_vectors, best["threshold"],
                        best["gap_penalty"], best["merge_penalty"], tables,
                    ))
                offset += size
        heldout_model_comparisons.append({
            "model": model_name,
            "parameters_selected_on_calibration": {
                "threshold": best["threshold"],
                "gap_penalty": best["gap_penalty"],
                "merge_penalty": best["merge_penalty"],
            },
            **aggregate(model_rows),
            "block_details": model_rows,
        })

    # Baseline: selected model with the pre-calibration default DP settings.
    baseline_rows = []
    offset = 0
    for index, block in enumerate(blocks):
        size = len(block)
        if index in test_ids:
            block_vectors = np.concatenate((
                selected_vectors[offset:offset + size],
                selected_vectors[pair_count + offset:pair_count + offset + size],
            ), axis=0)
            baseline_rows.append(
                evaluate_block(block, block_vectors, 0.35, 0.12, 0.06,
                               selected_similarity_cache[index])
            )
        offset += size
    baseline_summary = aggregate(baseline_rows)
    tuned_summary = aggregate(heldout_rows)
    improvement = {
        "precision_delta": round(tuned_summary["precision"] - baseline_summary["precision"], 6),
        "recall_delta": round(tuned_summary["recall"] - baseline_summary["recall"], 6),
        "f1_delta": round(tuned_summary["f1"] - baseline_summary["f1"], 6),
    }

    report = {
        "corpus": str(Path(args.input)),
        "corpus_sentence_pairs": len(pairs),
        "calibration_sentence_pairs": sum(len(blocks[i]) for i in dev_ids),
        "heldout_sentence_pairs": sum(len(blocks[i]) for i in test_ids),
        "block_size": args.block_size,
        "candidate_models": model_names,
        "test_corpus": str(Path(args.test_input)) if args.test_input else None,
        "split": split_report,
        "model_calibrations": model_calibrations,
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selected_gap_penalty": selected_gap_penalty,
        "selected_merge_penalty": selected_merge_penalty,
        "heldout_baseline_default_parameters": baseline_summary,
        "heldout_test": tuned_summary,
        "heldout_model_comparisons": heldout_model_comparisons,
        "heldout_improvement_over_baseline": improvement,
        "gold_alignment_assumption": (
            "Each TSV row is treated as one gold 1:1 pair. This assumption must be "
            "checked for segment-boundary splits, merges, or source/reference defects; "
            "reported mismatch examples are candidates for human review, not automatic "
            "proof that the model is wrong."
        ),
        "heldout_block_details": heldout_rows,
        "methodology": (
            "Gold alignment is the original parallel-corpus row pairing. Model and DP "
            "parameters are selected using calibration blocks only; held-out blocks are "
            "evaluated once. Exact 1:1 links count as correct; wrong links, 1:n/n:1 groups, "
            "and unmatched sentences are separately reported. Exact-match accuracy is "
            "exact links divided by gold pairs. When --test-input is supplied, the test "
            "corpus is a separate WMT release and exact monolingual overlaps are removed. "
            "This benchmark measures sentence alignment, not word/phrase translation correctness."
        ),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selected_gap_penalty": selected_gap_penalty,
        "selected_merge_penalty": selected_merge_penalty,
        "calibration": {k: selected["best_on_dev"][k] for k in ("exact_match_accuracy", "precision", "recall", "f1", "errors")},
        "heldout_baseline": report["heldout_baseline_default_parameters"],
        "heldout_test": report["heldout_test"],
        "heldout_model_comparisons": [
            {k: item[k] for k in ("model", "parameters_selected_on_calibration",
                                  "exact_match_accuracy", "precision", "recall", "f1", "errors")}
            for item in heldout_model_comparisons
        ],
        "heldout_improvement": report["heldout_improvement_over_baseline"],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
