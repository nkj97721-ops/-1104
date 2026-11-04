#!/usr/bin/env python3
"""Run a fully automatic SILVER benchmark on real EN→Simplified Chinese data.

No human annotation is required. Scores measure agreement with upstream
automatic alignment references, not verified semantic accuracy.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from collections import Counter
from statistics import mean
from typing import Any

try:
    from .benchmark_gold_groups import aggregate, evaluate_document
    from .boundary_groups import reconstruct_boundary_groups
    from .sentence_align import build_similarity_tables, clear_model_cache, encode_sentences
except ImportError:
    from benchmark_gold_groups import aggregate, evaluate_document
    from boundary_groups import reconstruct_boundary_groups
    from sentence_align import build_similarity_tables, clear_model_cache, encode_sentences


def load_silver(path: Path, max_documents: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration, test = [], []
    counts: dict[str, Any] = {
        "input_documents": 0, "excluded_oversize": 0,
        "excluded_oversize_groups": 0, "excluded_empty": 0,
    }
    supported_group_types = Counter()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        counts["input_documents"] += 1
        if item.get("reference_status") != "automatic_silver_reference":
            raise ValueError(f"{item.get('id')}: expected automatic_silver_reference record")
        groups = item.get("proposed_groups", [])
        if not groups:
            counts["excluded_empty"] += 1
            continue
        supported = [
            g for g in groups
            if len(g["source"]) in (1, 2) and len(g["target"]) in (1, 2)
        ]
        oversized = [g for g in groups if g not in supported]
        counts["excluded_oversize_groups"] += len(oversized)
        supported_group_types.update(
            f'{len(g["source"])}:{len(g["target"])}' for g in supported
        )
        if not supported:
            counts["excluded_oversize"] += 1
            continue

        # Keep only sentences covered by supported silver groups or explicit
        # upstream gaps. Oversized groups are excluded as whole regions rather
        # than being mis-scored as unmatched sentences.
        used_s = {i for g in supported for i in g["source"]}
        used_t = {j for g in supported for j in g["target"]}
        oversized_s = {i for g in oversized for i in g["source"]}
        oversized_t = {j for g in oversized for j in g["target"]}
        unmatched_s = set(item.get("proposed_unmatched_source", [])) - oversized_s
        unmatched_t = set(item.get("proposed_unmatched_target", [])) - oversized_t
        keep_s = sorted(used_s | unmatched_s)
        keep_t = sorted(used_t | unmatched_t)
        map_s = {old: new for new, old in enumerate(keep_s)}
        map_t = {old: new for new, old in enumerate(keep_t)}
        original_source_count, original_target_count = len(item["source"]), len(item["target"])
        has_source_origins = "source_sentence_origins" in item
        has_target_origins = "target_sentence_origins" in item
        if has_source_origins != has_target_origins:
            raise ValueError(f"{item['id']}: source and target boundary provenance must be supplied together")
        item["source"] = [item["source"][i] for i in keep_s]
        item["target"] = [item["target"][i] for i in keep_t]
        for metadata_key, keep_indices, original_count in (
            ("source_sentence_origins", keep_s, original_source_count),
            ("target_sentence_origins", keep_t, original_target_count),
        ):
            origins = item.get(metadata_key)
            if origins is not None:
                if not isinstance(origins, list) or len(origins) != original_count:
                    raise ValueError(
                        f"{item['id']}: {metadata_key} must contain one origin per original sentence"
                    )
                item[metadata_key] = [
                    {**origins[old_index], "sentence_index": new_index}
                    for new_index, old_index in enumerate(keep_indices)
                ]
        item["_silver_groups_original"] = supported
        item["_gold_groups"] = [
            (tuple(map_s[i] for i in g["source"]),
             tuple(map_t[j] for j in g["target"]))
            for g in supported
        ]
        item["_gold_unmatched_source"] = sorted(map_s[i] for i in unmatched_s)
        item["_gold_unmatched_target"] = sorted(map_t[j] for j in unmatched_t)
        if "source_sentence_origins" in item and "target_sentence_origins" in item:
            projection = reconstruct_boundary_groups(
                item["source_sentence_origins"], item["target_sentence_origins"]
            )
            projected_supported = {
                (tuple(group["source"]), tuple(group["target"]))
                for group in projection["groups"]
                if len(group["source"]) in (1, 2) and len(group["target"]) in (1, 2)
            }
            if projected_supported != set(item["_gold_groups"]):
                raise ValueError(
                    f"{item['id']}: reconstructed boundary groups do not match "
                    "the stored automatic reference groups"
                )
            if (projection["unmatched_source"] != item["_gold_unmatched_source"]
                    or projection["unmatched_target"] != item["_gold_unmatched_target"]):
                raise ValueError(
                    f"{item['id']}: reconstructed boundary gaps do not match stored gaps"
                )
            item["_boundary_groups"] = projection["groups"]
            item["_boundary_projection_validated"] = True
        else:
            item["_boundary_groups"] = []
            item["_boundary_projection_validated"] = False
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
    counts["boundary_metadata_documents"] = sum(
        bool(x.get("source_sentence_origins")) and bool(x.get("target_sentence_origins"))
        for x in calibration + test
    )
    counts["boundary_projection_validated_documents"] = sum(
        x.get("_boundary_projection_validated", False) for x in calibration + test
    )
    counts["source_sentence_origins_retained"] = sum(
        len(x.get("source_sentence_origins", [])) for x in calibration + test
    )
    counts["target_sentence_origins_retained"] = sum(
        len(x.get("target_sentence_origins", [])) for x in calibration + test
    )
    sampled_types = {}
    for split_name, rows in (("calibration", calibration), ("test", test)):
        split_types = Counter(
            f'{len(group["source"])}:{len(group["target"])}'
            for row in rows for group in row["_silver_groups_original"]
        )
        sampled_types[split_name] = dict(split_types)
    counts["sampled_group_types_by_split"] = sampled_types
    counts["supported_group_types_sampled"] = dict(
        sum((Counter(values) for values in sampled_types.values()), Counter())
    )
    counts["candidate_group_types_all_documents"] = dict(supported_group_types)
    return calibration + test, counts


def bootstrap_confidence_intervals(rows: list[dict[str, Any]], samples: int = 2000,
                                  seed: int = 20261104) -> dict[str, Any]:
    """Document-cluster bootstrap; keeps every document's groups together."""
    if len(rows) < 2:
        return {"status": "INSUFFICIENT_DOCUMENTS", "documents": len(rows)}
    rng = random.Random(seed)
    group_scores, link_scores = [], []
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        metrics = aggregate(sample)
        group_scores.append(metrics["group"]["f1"])
        link_scores.append(metrics["link"]["f1"])
    def interval(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        return {"lower_95": low, "upper_95": high}
    return {
        "status": "DOCUMENT_CLUSTER_BOOTSTRAP",
        "samples": samples,
        "documents": len(rows),
        "group_f1": interval(group_scores),
        "link_f1": interval(link_scores),
        "macro_document_group_f1": mean(row["group"]["f1"] for row in rows),
    }


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
        similarities = {
            x["id"]: build_similarity_tables(
                vectors[x["id"]][:len(x["source"])],
                vectors[x["id"]][len(x["source"]):],
            )
            for x in docs
        }
        clear_model_cache()
        candidates = []
        for threshold, gap, merge in itertools.product(thresholds, gaps, merges):
            rows = [
                evaluate_document(
                    x, vectors[x["id"]], threshold, gap, merge,
                    similarities[x["id"]],
                )
                for x in dev
            ]
            candidates.append({
                "threshold": threshold,
                "gap_penalty": gap,
                "merge_penalty": merge,
                "calibration_silver": aggregate(rows),
            })
        best = max(candidates, key=lambda x: (
            x["calibration_silver"]["macro_group_type_f1"] or 0.0,
            x["calibration_silver"]["group"]["f1"],
            x["calibration_silver"]["link"]["f1"],
        ))
        heldout = [
            evaluate_document(
                x, vectors[x["id"]], best["threshold"],
                best["gap_penalty"], best["merge_penalty"],
                similarities[x["id"]],
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
            "heldout_confidence_intervals_95": bootstrap_confidence_intervals(heldout),
            "heldout_document_errors": heldout,
            "parameter_grid": candidates,
        })

    result = {
        "status": "AUTOMATIC_SILVER_REFERENCE_EVALUATION_NOT_HUMAN_GOLD",
        "reference_type": "Upstream automatic segment/subtitle alignment proposals; weak reference, not human-verified semantic gold",
        "evaluation_tracks": {
            "semantic_sentence_links": (
                "Link F1 measures sentence-pair connectivity from embedding-only predictions; "
                "it does not determine original subtitle/segment grouping."
            ),
            "original_boundary_grouping": (
                "Exact group F1 measures agreement with upstream paired line/segment boundaries. "
                "It is a boundary-provenance agreement score, not semantic-group accuracy."
            ),
            "model_input_isolation": (
                "Original boundary IDs are retained for diagnostics but are not passed into "
                "sentence embeddings or dynamic-programming transition scoring."
            ),
        },
        "boundary_metadata_policy": (
            "Per-sentence source/target boundary IDs are preserved through document filtering; "
            "missing or mis-sized provenance arrays fail validation."
        ),
        "languages": {"source": "English", "target": "Simplified Chinese (zh_CN)"},
        "split_unit": "whole movie IDs; no movie appears in both calibration and held-out splits",
        "counts": counts,
        "model_comparisons": reports,
        "evaluation_policy": "Fully automatic; no human annotation or adjudication is required to run this benchmark.",
        "warning": (
            "These scores measure agreement with automatic upstream alignment proposals, not verified semantic alignment. "
            "They are silver-reference agreement scores, not final semantic accuracy. "
            "Reference groups with more than two sentences on either side are excluded region-by-region."
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
                "heldout_confidence_intervals_95": x["heldout_confidence_intervals_95"],
            }
            for x in reports
        ],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
