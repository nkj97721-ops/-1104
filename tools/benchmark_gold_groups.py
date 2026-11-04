#!/usr/bin/env python3
"""Evaluate monotonic EN-ZH sentence alignment against manually annotated groups.

JSONL record:
{"id":"doc-001","split":"calibration","source":["...","..."],"target":["...","...","..."],"gold":[{"source":[0],"target":[0,1]},{"source":[1],"target":[2]}]}

Supported gold groups are 1:1, 1:2, 2:1 and 2:2. Unmatched sentences are
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
    from .sentence_align import align_sentence_embeddings, build_similarity_tables, clear_model_cache, encode_sentences
except ImportError:  # Direct execution: python tools/benchmark_gold_groups.py
    from sentence_align import align_sentence_embeddings, build_similarity_tables, clear_model_cache, encode_sentences


def load_documents(
    path: Path,
    min_calibration_documents: int = 1,
    min_test_documents: int = 1,
    min_movies_per_split: int = 1,
) -> list[dict[str, Any]]:
    documents = []
    seen_ids = set()
    movie_splits: dict[str, str] = {}
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
        if item.get("annotation_status") != "human_verified":
            raise ValueError(f"JSONL line {line_no}: only human_verified gold records can be benchmarked")
        movie_id = str(item.get("movie_id", "")).strip()
        if movie_id:
            previous_split = movie_splits.setdefault(movie_id, split)
            if previous_split != split:
                raise ValueError(f"JSONL line {line_no}: movie {movie_id} leaks across calibration and test")
        if not str(item.get("annotator", "")).strip() or not str(item.get("reviewed_at", "")).strip():
            raise ValueError(f"JSONL line {line_no}: annotator and reviewed_at are required for human gold")
        groups = []
        used_s, used_t = set(), set()
        previous_s = previous_t = -1
        for group_no, group in enumerate(gold, 1):
            si, ti = group.get("source"), group.get("target")
            if not isinstance(si, list) or not isinstance(ti, list) or not si or not ti:
                raise ValueError(f"{doc_id} gold group {group_no}: source/target indices must be non-empty lists")
            if len(si) > 2 or len(ti) > 2 or (len(si), len(ti)) not in ((1, 1), (1, 2), (2, 1), (2, 2)):
                raise ValueError(f"{doc_id} gold group {group_no}: only 1:1, 1:2, 2:1 and 2:2 are supported")
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
    calibration_docs = [x for x in documents if x["split"] == "calibration"]
    test_docs = [x for x in documents if x["split"] == "test"]
    if len(calibration_docs) < min_calibration_documents:
        raise ValueError(
            f"Need at least {min_calibration_documents} human-reviewed calibration documents; "
            f"found {len(calibration_docs)}"
        )
    if len(test_docs) < min_test_documents:
        raise ValueError(
            f"Need at least {min_test_documents} human-reviewed independent test documents; "
            f"found {len(test_docs)}"
        )
    for split_name, split_docs in (("calibration", calibration_docs), ("test", test_docs)):
        independent_units = {
            str(x.get("movie_id") or x["id"]) for x in split_docs
        }
        if len(independent_units) < min_movies_per_split:
            raise ValueError(
                f"Need at least {min_movies_per_split} independent movie/document IDs in "
                f"{split_name}; found {len(independent_units)}"
            )
    return documents


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def prf(correct: int, predicted: int, gold: int) -> dict[str, float]:
    p = correct / predicted if predicted else 0.0
    r = correct / gold if gold else 0.0
    return {"precision": p, "recall": r, "f1": f1(p, r)}


def evaluate_document(doc: dict[str, Any], vectors: Any, threshold: float,
                      gap_penalty: float, merge_penalty: float,
                      similarity_tables=None) -> dict[str, Any]:
    source, target = doc["source"], doc["target"]
    result = align_sentence_embeddings(
        vectors[:len(source)], vectors[len(source):],
        threshold=threshold, gap_penalty=gap_penalty,
        merge_penalty=merge_penalty,
        similarity_tables=similarity_tables,
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
    error_examples = []
    for si, ti in sorted(predicted_groups - gold_groups)[:10]:
        error_examples.append({
            "type": "false_positive_group",
            "source_indices": list(si),
            "target_indices": list(ti),
            "source_texts": [source[i] for i in si],
            "target_texts": [target[j] for j in ti],
        })
    for si, ti in sorted(gold_groups - predicted_groups)[:10]:
        error_examples.append({
            "type": "missed_gold_group",
            "source_indices": list(si),
            "target_indices": list(ti),
            "source_texts": [source[i] for i in si],
            "target_texts": [target[j] for j in ti],
        })
    for index in sorted(pred_unmatched_s - gold_unmatched_s)[:5]:
        error_examples.append({
            "type": "false_unmatched_source",
            "source_index": index,
            "source_text": source[index],
        })
    for index in sorted(pred_unmatched_t - gold_unmatched_t)[:5]:
        error_examples.append({
            "type": "false_unmatched_target",
            "target_index": index,
            "target_text": target[index],
        })
    for index in sorted(gold_unmatched_s - pred_unmatched_s)[:5]:
        error_examples.append({
            "type": "missed_unmatched_source",
            "source_index": index,
            "source_text": source[index],
        })
    for index in sorted(gold_unmatched_t - pred_unmatched_t)[:5]:
        error_examples.append({
            "type": "missed_unmatched_target",
            "target_index": index,
            "target_text": target[index],
        })
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
        "error_examples": error_examples,
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
    ap.add_argument(
        "--models",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/LaBSE",
        help="Comma-separated sentence-transformer candidates; each model is calibrated independently",
    )
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--gap-penalties", default="0.06,0.12,0.18,0.24")
    ap.add_argument("--merge-penalties", default="0.02,0.06,0.10,0.14")
    ap.add_argument("--min-calibration-documents", type=int, default=10)
    ap.add_argument("--min-test-documents", type=int, default=10)
    ap.add_argument("--min-movies-per-split", type=int, default=3)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    if min(args.min_calibration_documents, args.min_test_documents, args.min_movies_per_split) < 1:
        raise ValueError("Minimum sample requirements must be positive")
    docs = load_documents(
        Path(args.input),
        min_calibration_documents=args.min_calibration_documents,
        min_test_documents=args.min_test_documents,
        min_movies_per_split=args.min_movies_per_split,
    )
    dev = [x for x in docs if x["split"] == "calibration"]
    test = [x for x in docs if x["split"] == "test"]
    models = list(dict.fromkeys(x.strip() for x in args.models.split(",") if x.strip()))
    if not models:
        raise ValueError("At least one sentence embedding model is required")
    thresholds = sorted({float(x) for x in args.thresholds.split(",")})
    gaps = sorted({float(x) for x in args.gap_penalties.split(",")})
    merges = sorted({float(x) for x in args.merge_penalties.split(",")})
    if (not thresholds or any(not math.isfinite(x) or not -1 <= x <= 1 for x in thresholds)
            or any(not math.isfinite(x) or x < 0 for x in gaps + merges)):
        raise ValueError("Invalid threshold or penalty grid")

    model_reports = []
    for model in models:
        # Encode each document once per model; calibration grid reuses cached vectors.
        vectors_by_id = {
            x["id"]: encode_sentences(x["source"] + x["target"], model, args.no_cuda)
            for x in docs
        }
        similarities_by_id = {
            x["id"]: build_similarity_tables(
                vectors_by_id[x["id"]][:len(x["source"])],
                vectors_by_id[x["id"]][len(x["source"]):],
            )
            for x in docs
        }
        clear_model_cache()
        candidates = []
        for threshold, gap, merge in itertools.product(thresholds, gaps, merges):
            rows = [
                evaluate_document(
                    x, vectors_by_id[x["id"]], threshold, gap, merge,
                    similarities_by_id[x["id"]],
                )
                for x in dev
            ]
            candidates.append({
                "threshold": threshold,
                "gap_penalty": gap,
                "merge_penalty": merge,
                "calibration": aggregate(rows),
            })
        best = max(candidates, key=lambda x: (
            x["calibration"]["group"]["f1"],
            x["calibration"]["link"]["f1"],
            x["calibration"]["group"]["precision"],
        ))
        # Each candidate model is evaluated on held-out documents only after
        # its own parameters have been frozen using calibration documents.
        heldout_rows = [
            evaluate_document(
                x, vectors_by_id[x["id"]], best["threshold"],
                best["gap_penalty"], best["merge_penalty"],
                similarities_by_id[x["id"]],
            )
            for x in test
        ]
        model_reports.append({
            "model": model,
            "selected_parameters": {
                k: best[k] for k in ("threshold", "gap_penalty", "merge_penalty")
            },
            "calibration": best["calibration"],
            "independent_test": aggregate(heldout_rows),
            "test_document_details": heldout_rows,
            "calibration_grid": candidates,
        })

    # Select the deployable candidate from calibration scores only. Test scores
    # are reported for comparison and never used to choose the winner.
    selected = max(model_reports, key=lambda x: (
        x["calibration"]["group"]["f1"],
        x["calibration"]["link"]["f1"],
        x["calibration"]["group"]["precision"],
    ))
    report = {
        "annotation_format": "document-level explicit sentence groups; zero-based indices",
        "supported_gold_group_types": ["1:1", "1:2", "2:1", "2:2"],
        "unmatched_policy": "Any sentence omitted from all gold groups is explicitly treated as unmatched.",
        "selection_rule": "Choose model by calibration group-level exact-match F1, then link-level F1; never select by held-out results.",
        "selected_model_on_calibration": selected["model"],
        "model_comparisons": model_reports,
        "warning": "Scores apply only to supplied human-annotated documents; they do not inherit or validate the legacy TSV 1:1 score.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_model_on_calibration": report["selected_model_on_calibration"],
        "model_comparisons": [
            {
                "model": x["model"],
                "selected_parameters": x["selected_parameters"],
                "calibration": x["calibration"],
                "independent_test": x["independent_test"],
            }
            for x in model_reports
        ],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
