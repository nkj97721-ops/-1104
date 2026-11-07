#!/usr/bin/env python3
"""Diagnose ordered 1:1 candidates inside held-out automatic-silver 2:2 groups.

This measures embedding evidence only. Upstream 2:2 segment boundaries do not
supply sentence-level semantic gold links, so this tool never reports accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    from .sentence_align import clear_model_cache, encode_sentences
except ImportError:
    from sentence_align import clear_model_cache, encode_sentences

DEFAULT_MODELS = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,"
    "sentence-transformers/LaBSE"
)


def cosine(left: Any, right: Any) -> float:
    a = [float(x) for x in left]
    b = [float(x) for x in right]
    if len(a) != len(b):
        raise ValueError("Embedding dimensions differ")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))


def load_heldout_groups(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    counts = {"documents": 0, "test_documents": 0, "two_two_groups": 0}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        counts["documents"] += 1
        if row.get("reference_status") != "automatic_silver_reference":
            raise ValueError(f"{row.get('id')}: not an automatic silver reference")
        if row.get("split") != "test":
            continue
        counts["test_documents"] += 1
        # proposed_groups index compact sentence arrays. Boundary diagnostics use
        # original sentence_index values, so map through retained origins before
        # joining; otherwise filtering/sentence splitting silently misaligns rows.
        source_origins = row.get("source_sentence_origins") or []
        target_origins = row.get("target_sentence_origins") or []
        source_origin_indices = [
            int(item["sentence_index"]) for item in source_origins
        ] if source_origins else list(range(len(row["source"])))
        target_origin_indices = [
            int(item["sentence_index"]) for item in target_origins
        ] if target_origins else list(range(len(row["target"])))
        if len(source_origin_indices) != len(row["source"]):
            raise ValueError(f"{row['id']}: source origin count does not match sentences")
        if len(target_origin_indices) != len(row["target"]):
            raise ValueError(f"{row['id']}: target origin count does not match sentences")
        groups = []
        for group in row.get("proposed_groups", []):
            if len(group.get("source", [])) == 2 and len(group.get("target", [])) == 2:
                si, ti = group["source"], group["target"]
                if any(i < 0 or i >= len(row["source"]) for i in si):
                    raise ValueError(f"{row['id']}: source index out of range")
                if any(i < 0 or i >= len(row["target"]) for i in ti):
                    raise ValueError(f"{row['id']}: target index out of range")
                groups.append({
                    "document_id": row["id"],
                    "source": [row["source"][i] for i in si],
                    "target": [row["target"][i] for i in ti],
                    "source_indices": [source_origin_indices[i] for i in si],
                    "target_indices": [target_origin_indices[i] for i in ti],
                    "source_boundary_ids": list(group.get("source_boundary_ids", [])),
                    "target_boundary_ids": list(group.get("target_boundary_ids", [])),
                })
        counts["two_two_groups"] += len(groups)
        for group in groups:
            rows.append(group)
    return rows, counts


def summarize_model(groups: list[dict[str, Any]], model: str, threshold: float,
                    no_cuda: bool) -> dict[str, Any]:
    by_document: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        by_document.setdefault(group["document_id"], []).append(group)

    results = []
    for document_id, doc_groups in by_document.items():
        sentences = []
        for group in doc_groups:
            sentences.extend(group["source"])
            sentences.extend(group["target"])
        vectors = encode_sentences(sentences, model_name=model, no_cuda=no_cuda)
        cursor = 0
        for group in doc_groups:
            src = vectors[cursor:cursor + 2]
            trg = vectors[cursor + 2:cursor + 4]
            cursor += 4
            diagonal = [cosine(src[i], trg[i]) for i in range(2)]
            crossed = [cosine(src[i], trg[1 - i]) for i in range(2)]
            results.append({
                "document_id": document_id,
                "english": group["source"],
                "chinese": group["target"],
                "source_indices": group["source_indices"],
                "target_indices": group["target_indices"],
                "source_boundary_ids": group["source_boundary_ids"],
                "target_boundary_ids": group["target_boundary_ids"],
                "ordered_pair_similarities": [round(x, 6) for x in diagonal],
                "cross_pair_similarities": [round(x, 6) for x in crossed],
                "ordered_mean": round(mean(diagonal), 6),
                "cross_mean": round(mean(crossed), 6),
                "ordered_minus_cross_margin": round(mean(diagonal) - mean(crossed), 6),
                "pairs_over_threshold": sum(x >= threshold for x in diagonal),
                "threshold": threshold,
            })
    clear_model_cache()

    both = sum(x["pairs_over_threshold"] == 2 for x in results)
    one = sum(x["pairs_over_threshold"] == 1 for x in results)
    none = sum(x["pairs_over_threshold"] == 0 for x in results)
    margins = [x["ordered_minus_cross_margin"] for x in results]
    ordered_means = [x["ordered_mean"] for x in results]
    cross_means = [x["cross_mean"] for x in results]
    # Show difficult/ambiguous cases, not just favorable examples.
    difficult = sorted(results, key=lambda x: (
        x["ordered_minus_cross_margin"], min(x["ordered_pair_similarities"])
    ))[:30]
    return {
        "model": model,
        "threshold": threshold,
        "groups": len(results),
        "candidate_support_counts": {
            "both_ordered_pairs_meet_threshold": both,
            "exactly_one_ordered_pair_meets_threshold": one,
            "neither_ordered_pair_meets_threshold": none,
        },
        "ordered_mean_similarity": {
            "mean": round(mean(ordered_means), 6) if ordered_means else None,
            "median": round(median(ordered_means), 6) if ordered_means else None,
        },
        "cross_mean_similarity": {
            "mean": round(mean(cross_means), 6) if cross_means else None,
            "median": round(median(cross_means), 6) if cross_means else None,
        },
        "ordered_vs_cross_margin": {
            "positive": sum(x > 0 for x in margins),
            "zero": sum(x == 0 for x in margins),
            "negative": sum(x < 0 for x in margins),
            "mean": round(mean(margins), 6) if margins else None,
            "median": round(median(margins), 6) if margins else None,
        },
        "all_group_diagnostics": results,
        "hardest_or_most_ambiguous_examples": difficult,
        "interpretation": (
            "Embedding evidence for monotonic ordered 1:1 candidates only. "
            "A positive diagonal-vs-cross margin is not proof of semantic correctness; "
            "automatic 2:2 segment boundaries do not identify sentence-level gold links."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/two-two-internal-candidates.json")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not -1 <= args.threshold <= 1:
        raise ValueError("--threshold must be between -1 and 1")
    groups, counts = load_heldout_groups(Path(args.input))
    if not groups:
        raise ValueError("No held-out 2:2 groups found; diagnostic is not estimable")
    models = list(dict.fromkeys(x.strip() for x in args.models.split(",") if x.strip()))
    if not models:
        raise ValueError("At least one model is required")
    report = {
        "status": "HELDOUT_2_2_EMBEDDING_DIAGNOSTIC_NOT_SEMANTIC_ACCURACY",
        "reference_type": "OpenSubtitles2024 automatic subtitle-boundary silver reference",
        "split_policy": "held-out movie split only; no calibration/test mixing",
        "counts": counts,
        "languages": {"source": "English", "target": "Simplified Chinese (zh_CN)"},
        "model_diagnostics": [
            summarize_model(groups, model, args.threshold, args.no_cuda)
            for model in models
        ],
        "limitations": [
            "No human annotation or independently verified sentence-level 2:2 semantic links are available.",
            "Both ordered pairs passing a cosine threshold is candidate evidence, not a true-positive count.",
            "Diagonal-versus-cross comparison measures model preference, not translation correctness.",
            "The original 2:2 subtitle boundary group is preserved and never split by this diagnostic.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "counts": counts,
        "models": [
            {
                "model": item["model"],
                "groups": item["groups"],
                "candidate_support_counts": item["candidate_support_counts"],
                "ordered_vs_cross_margin": item["ordered_vs_cross_margin"],
            }
            for item in report["model_diagnostics"]
        ],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
