#!/usr/bin/env python3
"""Run a clearly-labeled SILVER benchmark on OpenSubtitles2024 subtitle-timing links.

This is useful for a first real-corpus model comparison. It is NOT human-gold
accuracy. Human evaluation must use benchmark_gold_groups.py after review.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

try:
    from .benchmark_gold_groups import aggregate, evaluate_document
    from .sentence_align import encode_sentences
except ImportError:
    from benchmark_gold_groups import aggregate, evaluate_document
    from sentence_align import encode_sentences


def load_silver(path: Path, max_documents: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    calibration, test = [], []
    counts = {"input_documents": 0, "excluded_oversize": 0, "excluded_empty": 0}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        counts["input_documents"] += 1
        if item.get("annotation_status") != "pending_human_review":
            raise ValueError(f"{item.get('id')}: expected pending_human_review queue record")
        groups = item.get("proposed_groups", [])
        if not groups:
            counts["excluded_empty"] += 1
            continue
        if any(len(g["source"]) not in (1, 2) or len(g["target"]) not in (1, 2) for g in groups):
            counts["excluded_oversize"] += 1
            continue
        item["_gold_groups"] = [
            (tuple(g["source"]), tuple(g["target"])) for g in groups
        ]
        used_s = {i for s, t in item["_gold_groups"] for i in s}
        used_t = {j for s, t in item["_gold_groups"] for j in t}
        item["_gold_unmatched_source"] = sorted(
            (set(range(len(item["source"]))) - used_s)
            | set(item.get("proposed_unmatched_source", []))
        )
        item["_gold_unmatched_target"] = sorted(
            (set(range(len(item["target"]))) - used_t)
            | set(item.get("proposed_unmatched_target", []))
        )
        (test if item["split"] == "test" else calibration).append(item)

    # Stable round-robin sampling prevents one long subtitle from consuming
    # the whole evaluation budget and keeps several movie IDs represented.
    def balanced_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_movie: dict[str, list[dict[str, Any]]] = {}
        for row in sorted(rows, key=lambda x: x["id"]):
            by_movie.setdefault(row["movie_id"], []).append(row)
        selected = []
        movie_ids = sorted(by_movie)
        depth = 0
        while len(selected) < max_documents:
            added = False
            for movie_id in movie_ids:
                if depth < len(by_movie[movie_id]) and len(selected) < max_documents:
                    selected.append(by_movie[movie_id][depth])
                    added = True
            if not added:
                break
            depth += 1
        return selected

    calibration = balanced_sample(calibration)
    test = balanced_sample(test)
    if not calibration or not test:
        raise ValueError("Need supported candidate documents in both calibration and test movie splits")
    counts["calibration_documents_used"] = len(calibration)
    counts["test_documents_used"] = len(test)
    return calibration + test, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="silver-model-comparison.json")
    ap.add_argument(
        "--models",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/LaBSE",
    )
    ap.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45")
    ap.add_argument("--gap-penalties", default="0.06,0.12,0.18")
    ap.add_argument("--merge-penalties", default="0.02,0.06,0.10")
    ap.add_argument("--max-documents-per-split", type=int, default=100)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    docs, counts = load_silver(Path(args.input), args.max_documents_per_split)
    dev = [x for x in docs if x["split"] == "calibration"]
    test = [x for x in docs if x["split"] == "test"]
    thresholds = sorted({float(x) for x in args.thresholds.split(",")})
    gaps = sorted({float(x) for x in args.gap_penalties.split(",")})
    merges = sorted({float(x) for x in args.merge_penalties.split(",")})
    models = list(dict.fromkeys(x.strip() for x in args.models.split(",") if x.strip()))
    if not models or not thresholds or not gaps or not merges:
        raise ValueError("Models and parameter grids must be non-empty")

    reports = []
    for model in models:
        vectors = {
            x["id"]: encode_sentences(
                x["source"] + x["target"], model_name=model, no_cuda=args.no_cuda
            )
            for x in docs
        }
        candidates = []
        for threshold, gap, merge in itertools.product(thresholds, gaps, merges):
            rows = [
                evaluate_document(x, vectors[x["id"]], threshold, gap, merge)
                for x in dev
            ]
            candidates.append({
                "threshold": threshold,
                "gap_penalty": gap,
                "merge_penalty": merge,
                "calibration_silver": aggregate(rows),
            })
        best = max(candidates, key=lambda x: (
            x["calibration_silver"]["group"]["f1"],
            x["calibration_silver"]["link"]["f1"],
        ))
        heldout = [
            evaluate_document(
                x, vectors[x["id"]], best["threshold"],
                best["gap_penalty"], best["merge_penalty"],
            )
            for x in test
        ]
        reports.append({
            "model": model,
            "selected_parameters_on_calibration_only": {
                k: best[k] for k in ("threshold", "gap_penalty", "merge_penalty")
            },
            "calibration_silver": best["calibration_silver"],
            "heldout_silver": aggregate(heldout),
            "heldout_document_errors": heldout,
            "parameter_grid": candidates,
        })

    result = {
        "status": "PRELIMINARY_SILVER_EVALUATION_NOT_HUMAN_GOLD",
        "reference_type": "OpenSubtitles2024 automatic subtitle-timing alignment links",
        "languages": {"source": "English", "target": "Simplified Chinese (zh_CN)"},
        "split_unit": "whole movie IDs; no movie appears in both calibration and held-out splits",
        "counts": counts,
        "model_comparisons": reports,
        "warning": (
            "These scores measure agreement with automatic subtitle-timing links, not verified semantic alignment. "
            "Do not report them as final accuracy. Human adjudication is required before using the gold benchmark."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "counts": counts,
        "models": [
            {
                "model": x["model"],
                "parameters": x["selected_parameters_on_calibration_only"],
                "calibration_silver": x["calibration_silver"],
                "heldout_silver": x["heldout_silver"],
            }
            for x in reports
        ],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
