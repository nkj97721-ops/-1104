#!/usr/bin/env python3
"""Evaluate monotonic EN-ZH sentence alignment against manually annotated groups.

JSONL record:
{"id":"doc-001","split":"calibration","source":["...","..."],"target":["...","...","..."],"gold":[{"source":[0],"target":[0,1]},{"source":[1],"target":[2]}]}

Supported gold groups are 1:1, 1:2 and 2:1. Unmatched sentences are
inferred from indices not present in any gold group. Indices are zero-based.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .sentence_align import align_sentence_embeddings, encode_sentences
except ImportError:  # Direct execution: python tools/benchmark_gold_groups.py
    from sentence_align import align_sentence_embeddings, encode_sentences


def load_documents(path: Path) -> list[dict[str, Any]]:
    documents = []
    seen_ids = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL line {line_no}: {exc}") from exc
        doc_id = str(item.get("id", "")).strip()
        source, target, gold = item.get("source"), item.get("target"), item.get("gold")
        if not doc_id or doc_id in seen_ids:
            raise ValueError(f"JSONL line {line_no}: id must be present and unique")
        if (not isinstance(source, list) or not source or
                not isinstance(target, list) or not target or
                not isinstance(gold, list)):
            raise ValueError(f"JSONL line {line_no}: source/target/gold must be non-empty/list values")
        if any(not isinstance(x, str) or not x.strip() for x in source + target):
            raise ValueError(f"JSONL line {line_no}: sentence text must be non-empty strings")
        split = item.get("split")
        if split not in ("calibration", "test"):
            raise ValueError(f"JSONL line {line_no}: split must be calibration or test")
        groups = []
        used_s, used_t = set(), set()
        previous_s = previous_t = -1
        for group_no, group in enumerate(gold, 1):
            si, ti = group.get("source"), group.get("target")
            if not isinstance(si, list) or not isinstance(ti, list) or not si or not ti:
                raise ValueError(f"{doc_id} gold group {group_no}: source/target indices must be non-empty lists")
            if len(si) > 2 or len(ti) > 2 or (len(si), len(ti)) not in ((1, 1), (1, 2), (2, 1)):
                raise ValueError(f"{doc_id} gold group {group_no}: only 1:1, 1:2 and 2:1 are supported")
            if any(type(x) is not int or x < 0 or x >= len(source) for x in si):
                raise ValueError(f"{doc_id} gold group {group_no}: source index out of range")
            if any(type(x) is not int or x < 0 or x >= len(target) for x in ti):
                raise ValueError(f"{doc_id} gold group {group_no}: target index out of range")
            if si != list(range(si[0], si[0] + len(si))) or ti != list(range(ti[0], ti[0] + len(ti))):
                raise ValueError(f"{doc_id} gold group {group_no}: indices within a group must be contiguous")
            if min(si) <= previous_s or min(ti) <= previous_t:
                raise ValueError(f"{doc_id} gold group {group_no}: groups must be strictly monotonic")
            if used_s.intersection(si) or used_t.intersection(ti):
                raise ValueError(f"{doc_id} gold group {group_no}: sentence index reused")
            used_s.update(si)
            used_t.update(ti)
            previous_s, previous_t = max(si), max(ti)
            groups.append((tuple(si), tuple(ti)))
        item["_gold_groups"] = groups
        item["_gold_unmatched_source"] = sorted(set(range(len(source))) - used_s)
        item["_gold_unmatched_target"] = sorted(set(range(len(target))) - used_t)
        seen_ids.add(doc_id)
        documents.append(item)
    if not documents:
        raise ValueError("No annotated documents found")
    if not any(x["split"] == "calibration" for x in documents):
        raise ValueError("At least one calibration document is required")
    if not any(x["split"] == "test" for x in documents):
        raise ValueError("At least one independent test document is required")
    return documents


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def prf(correct: int, predicted: int, gold: int) -> dict[str, float]:
    p = correct / predicted if predicted else 0.0
    r = correct / gold if gold else 0.0
    return {"precision": p, "recall": r, "f1": f1(p, r)}


def evaluate_document(doc: dict[str, Any], vectors: Any, threshold: float,
                      gap_penalty: float, merge_penalty: float) -> dict[str, Any]:
    source, target = doc["source"], doc["target"]
    result = align_sentence_embeddings(
        vectors[:len(source)], vectors[len(source):],
        threshold=threshold, gap_penalty=gap_penalty,
        merge_penalty=merge_penalty,
    )
    predicted_groups = {
        (tuple(x["source_indices"]), tuple(x["target_indices"]))
        for x in result["matches"]
    }
    gold_groups = set(doc["_gold_groups"])
    exact = len(predicted_groups & gold_groups)
    predicted_links = {(i, j) for si, ti in predicted_groups for i in si for j in ti}
    gold_links = {(i, j) for si, ti in gold_groups for i in si for j in ti}
    link_correct = len(predicted_links & gold_links)
    pred_unmatched_s = set(result["unmatched_source"])
    pred_unmatched_t = set(result["unmatched_target"])
    gold_unmatched_s = set(doc["_gold_unmatched_source"])
    gold_unmatched_t = set(doc["_gold_unmatched_target"])
    unmatched_correct = len(pred_unmatched_s & gold_unmatched_s) + len(pred_unmatched_t & gold_unmatched_t)
    unmatched_predicted = len(pred_unmatched_s) + len(pred_unmatched_t)
    unmatched_gold = len(gold_unmatched_s) + len(gold_unmatched_t)
    breakdown = Counter(f"{len(si)}:{len(ti)}" for si, ti in predicted_groups)
    return {
        "id": doc["id"],
        "source_sentences": len(source),
        "target_sentences": len(target),
        "gold_groups": len(gold_groups),
        "predicted_groups": len(predicted_groups),
        "exact_groups": exact,
        "group": prf(exact, len(predicted_groups), len(gold_groups)),
        "gold_links": len(gold_links),
        "predicted_links": len(predicted_links),
        "correct_links": link_correct,
        "link": prf(link_correct, len(predicted_links), len(gold_links)),
        "gold_unmatched": unmatched_gold,
        "predicted_unmatched": unmatched_predicted,
        "correct_unmatched": unmatched_correct,
        "unmatched": prf(unmatched_correct, unmatched_predicted, unmatched_gold),
        "predicted_group_types": dict(breakdown),
        "predicted": [
            {"source": list(si), "target": list(ti)}
            for si, ti in sorted(predicted_groups)
        ],
        "gold": [
            {"source": list(si), "target": list(ti)}
            for si, ti in doc["_gold_groups"]
        ],
        "unmatched_source": result["unmatched_source"],
        "unmatched_target": result["unmatched_target"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sums = Counter()
    for row in rows:
        for key in ("exact_groups", "predicted_groups", "gold_groups",
                    "correct_links", "predicted_links", "gold_links",
                    "correct_unmatched", "predicted_unmatched", "gold_unmatched"):
            sums[key] += row[key]
    return {
        "documents": len(rows),
        "group": prf(sums["exact_groups"], sums["predicted_groups"], sums["gold_groups"]),
        "link": prf(sums["correct_links"], sums["predicted_links"], sums["gold_links"]),
        "unmatched": prf(sums["correct_unmatched"], sums["predicted_unmatched"], sums["gold_unmatched"]),
        "counts": dict(sums),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Manually annotated JSONL with calibration/test split labels")
    ap.add_argument("--output", default="gold-group-benchmark.json")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--gap-penalties", default="0.06,0.12,0.18,0.24")
    ap.add_argument("--merge-penalties", default="0.02,0.06,0.10,0.14")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    docs = load_documents(Path(args.input))
    dev = [x for x in docs if x["split"] == "calibration"]
    test = [x for x in docs if x["split"] == "test"]
    thresholds = sorted({float(x) for x in args.thresholds.split(",")})
    gaps = sorted({float(x) for x in args.gap_penalties.split(",")})
    merges = sorted({float(x) for x in args.merge_penalties.split(",")})
    if (not thresholds or any(not math.isfinite(x) or not -1 <= x <= 1 for x in thresholds)
            or any(not math.isfinite(x) or x < 0 for x in gaps + merges)):
        raise ValueError("Invalid threshold or penalty grid")

    # Load each document embedding exactly once; parameter search reuses vectors.
    vectors_by_id = {
        x["id"]: encode_sentences(x["source"] + x["target"], args.model, args.no_cuda)
        for x in docs
    }
    candidates = []
    for threshold, gap, merge in itertools.product(thresholds, gaps, merges):
        rows = [evaluate_document(x, vectors_by_id[x["id"]], threshold, gap, merge) for x in dev]
        summary = aggregate(rows)
        candidates.append({
            "threshold": threshold, "gap_penalty": gap, "merge_penalty": merge,
            "calibration": summary,
        })
    best = max(candidates, key=lambda x: (
        x["calibration"]["group"]["f1"],
        x["calibration"]["link"]["f1"],
        x["calibration"]["group"]["precision"],
    ))
    # Test documents are evaluated once, after all parameters are frozen.
    heldout_rows = [
        evaluate_document(x, vectors_by_id[x["id"]], best["threshold"],
                          best["gap_penalty"], best["merge_penalty"])
        for x in test
    ]
    report = {
        "model": args.model,
        "annotation_format": "document-level explicit sentence groups; zero-based indices",
        "supported_gold_group_types": ["1:1", "1:2", "2:1"],
        "unmatched_policy": "Any sentence omitted from all gold groups is explicitly treated as unmatched.",
        "selection": {
            "parameters": {k: best[k] for k in ("threshold", "gap_penalty", "merge_penalty")},
            "objective": "calibration group-level exact-match F1, then link-level F1",
            "calibration": best["calibration"],
        },
        "independent_test": aggregate(heldout_rows),
        "test_document_details": heldout_rows,
        "calibration_grid": candidates,
        "warning": "This score is valid only for the supplied human-annotated documents; it does not inherit or validate the legacy TSV 1:1 score.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "parameters": report["selection"]["parameters"],
        "calibration": report["selection"]["calibration"],
        "independent_test": report["independent_test"],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
