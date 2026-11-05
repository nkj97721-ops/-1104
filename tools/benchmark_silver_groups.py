#!/usr/bin/env python3
"""Run a fully automatic SILVER benchmark on real EN→Simplified Chinese data.

No human annotation is required. Scores measure agreement with upstream
automatic alignment references, not verified semantic accuracy.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
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
    all_reference_group_types = Counter()
    excluded_group_types = Counter()
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
        all_reference_group_types.update(
            f"{len(g['source'])}:{len(g['target'])}" for g in groups
        )
        supported = [
            g for g in groups
            if len(g["source"]) in (1, 2) and len(g["target"]) in (1, 2)
        ]
        oversized = [g for g in groups if g not in supported]
        excluded_group_types.update(
            f"{len(g['source'])}:{len(g['target'])}" for g in oversized
        )
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
    counts["all_reference_group_types_all_documents"] = dict(all_reference_group_types)
    counts["excluded_group_types_all_documents"] = dict(excluded_group_types)
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
    ap.add_argument("--two-two-penalties", default="0.00,0.04,0.08,0.12,0.16")
    ap.add_argument("--two-two-order-weights", default="0.0,0.5,1.0")
    ap.add_argument("--group-type-regression-tolerance", type=float, default=0.02)
    ap.add_argument("--max-documents-per-split", type=int, default=100)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    docs, counts = load_silver(Path(args.input), args.max_documents_per_split)
    dev = [x for x in docs if x["split"] == "calibration"]
    test = [x for x in docs if x["split"] == "test"]
    thresholds = sorted({float(x) for x in args.thresholds.split(",")})
    gaps = sorted({float(x) for x in args.gap_penalties.split(",")})
    merges = sorted({float(x) for x in args.merge_penalties.split(",")})
    two_two_penalties = sorted({float(x) for x in args.two_two_penalties.split(",")})
    two_two_order_weights = sorted({float(x) for x in args.two_two_order_weights.split(",")})
    models = list(dict.fromkeys(x.strip() for x in args.models.split(",") if x.strip()))
    if (not models or not thresholds or not gaps or not merges or not two_two_penalties
            or not two_two_order_weights):
        raise ValueError("Models and parameter grids must be non-empty")
    if (any(not math.isfinite(x) or x < 0 for x in two_two_order_weights)
            or not math.isfinite(args.group_type_regression_tolerance)
            or not 0 <= args.group_type_regression_tolerance <= 1):
        raise ValueError("Invalid 2:2 order weight or group-type regression tolerance")
    if 0.0 not in two_two_order_weights:
        raise ValueError("two-two-order-weights must include 0.0 as the baseline")

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
        for threshold, gap, merge, two_two, order_weight in itertools.product(
            thresholds, gaps, merges, two_two_penalties, two_two_order_weights
        ):
            rows = [
                evaluate_document(
                    x, vectors[x["id"]], threshold, gap, merge,
                    similarities[x["id"]], two_two_penalty=two_two,
                    two_two_order_weight=order_weight,
                )
                for x in dev
            ]
            candidates.append({
                "threshold": threshold,
                "gap_penalty": gap,
                "merge_penalty": merge,
                "two_two_penalty": two_two,
                "two_two_order_weight": order_weight,
                "calibration_silver": aggregate(rows),
            })
        baseline_candidates = [x for x in candidates if x["two_two_order_weight"] == 0.0]
        baseline = max(baseline_candidates, key=lambda x: (
            x["calibration_silver"]["macro_group_type_f1"] or 0.0,
            x["calibration_silver"]["group"]["f1"],
            x["calibration_silver"]["link"]["f1"],
        ))
        baseline_types = baseline["calibration_silver"]["group_type"]
        protected_types = {
            group_type: metrics["f1"]
            for group_type, metrics in baseline_types.items()
            if group_type != "2:2" and metrics.get("status") == "ESTIMATED"
        }
        eligible = []
        for candidate in candidates:
            candidate_types = candidate["calibration_silver"]["group_type"]
            if all(
                candidate_types.get(group_type, {}).get("status") == "ESTIMATED"
                and candidate_types[group_type]["f1"] >= baseline_f1 - args.group_type_regression_tolerance
                for group_type, baseline_f1 in protected_types.items()
            ):
                eligible.append(candidate)
        baseline_two_two = baseline_types.get("2:2", {}).get("f1")
        best = baseline
        if eligible and baseline_two_two is not None:
            best_constrained = max(eligible, key=lambda x: (
                x["calibration_silver"]["group_type"].get("2:2", {}).get("f1") or 0.0,
                x["calibration_silver"]["macro_group_type_f1"] or 0.0,
                x["calibration_silver"]["group"]["f1"],
                x["calibration_silver"]["link"]["f1"],
            ))
            candidate_two_two = best_constrained["calibration_silver"]["group_type"].get("2:2", {}).get("f1") or 0.0
            if candidate_two_two > baseline_two_two:
                best = best_constrained
        # Semantic-link tuning excludes the new 2:2 bonus: 2:2 boundaries have
        # no sentence-level semantic gold links and must not affect this track.
        best_semantic = max(baseline_candidates, key=lambda x: (
            x["calibration_silver"]["semantic_link"]["f1"] or 0.0,
            x["calibration_silver"]["semantic_link"]["precision"] or 0.0,
            x["calibration_silver"]["link"]["f1"],
        ))
        heldout = [
            evaluate_document(
                x, vectors[x["id"]], best["threshold"],
                best["gap_penalty"], best["merge_penalty"],
                similarities[x["id"]], two_two_penalty=best["two_two_penalty"],
                two_two_order_weight=best["two_two_order_weight"],
            )
            for x in test
        ]
        semantic_heldout = [
            evaluate_document(
                x, vectors[x["id"]], best_semantic["threshold"],
                best_semantic["gap_penalty"], best_semantic["merge_penalty"],
                similarities[x["id"]], two_two_penalty=best_semantic["two_two_penalty"],
                two_two_order_weight=best_semantic["two_two_order_weight"],
            )
            for x in test
        ]
        reports.append({
            "model": model,
            "selected_parameters_on_calibration_only": {
                k: best[k] for k in (
                    "threshold", "gap_penalty", "merge_penalty", "two_two_penalty",
                    "two_two_order_weight"
                )
            },
            "calibration_silver": best["calibration_silver"],
            "heldout_silver": aggregate(heldout),
            "calibration_selection_diagnostics": {
                "baseline_without_order_bonus": {
                    "parameters": {
                        k: baseline[k] for k in (
                            "threshold", "gap_penalty", "merge_penalty", "two_two_penalty",
                            "two_two_order_weight"
                        )
                    },
                    "group_type": baseline_types,
                },
                "selected_group_type": best["calibration_silver"]["group_type"],
                "protected_group_types": protected_types,
                "maximum_allowed_regression": args.group_type_regression_tolerance,
                "selection_status": (
                    "2:2_IMPROVED_WITH_PROTECTED_TYPES_WITHIN_TOLERANCE"
                    if best is not baseline else "NO_SAFE_2:2_IMPROVEMENT_FOUND_BASELINE_RETAINED"
                ),
            },
            "semantic_parameters_on_calibration_only": {
                k: best_semantic[k] for k in (
                    "threshold", "gap_penalty", "merge_penalty", "two_two_penalty",
                    "two_two_order_weight"
                )
            },
            "semantic_calibration_silver": best_semantic["calibration_silver"],
            "semantic_heldout_silver": aggregate(semantic_heldout),
            "semantic_heldout_confidence_intervals_95": bootstrap_confidence_intervals(semantic_heldout),
            "heldout_confidence_intervals_95": bootstrap_confidence_intervals(heldout),
            "heldout_document_errors": heldout,
            "semantic_heldout_document_errors": semantic_heldout,
            "parameter_grid": candidates,
        })

    result = {
        "status": "AUTOMATIC_SILVER_REFERENCE_EVALUATION_NOT_HUMAN_GOLD",
        "reference_type": "Upstream automatic segment/subtitle alignment proposals; weak reference, not human-verified semantic gold",
        "evaluation_tracks": {
            "two_two_order_evidence": (
                "A non-negative order bonus may raise the pooled 2:2 score only when the mean "
                "diagonal sentence-pair cosine is stronger; it never lowers the pooled score. "
                "The bonus is calibrated on calibration documents only and is selected only if "
                "1:1, 1:2 and 2:1 F1 each remain within the configured regression tolerance."
            ),
            "semantic_sentence_links": (
                "The new semantic_link F1 scores only sentence regions outside 2:2 reference blocks. "
                "It uses 1:1, 1:2 and 2:1 weak silver links; 2:2 blocks are excluded because "
                "their segment boundaries do not specify individual sentence-to-sentence links."
            ),
            "boundary_implied_link_connectivity": (
                "The legacy link F1 expands each upstream group as a Cartesian product, including "
                "2:2 groups. It is boundary-implied connectivity agreement, not independent semantic-link accuracy."
            ),
            "semantic_2_2_alignment": (
                "NOT_ESTIMABLE_WITHOUT_INDEPENDENT_REFERENCE: automatic 2:2 segment boundaries do not "
                "identify the sentence-level correspondence pattern inside each block."
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
        "selection_policy": (
            "The boundary track first selects a zero-order-bonus baseline on calibration data. "
            "A 2:2 order bonus is adopted only when it improves calibration 2:2 F1 while keeping "
            "each estimable non-2:2 group type within the configured regression tolerance. "
            "The semantic-link track is calibrated separately with order bonus fixed at zero."
        ),
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
                "semantic_heldout_silver": x["semantic_heldout_silver"],
                "calibration_selection_diagnostics": x["calibration_selection_diagnostics"],
                "heldout_confidence_intervals_95": x["heldout_confidence_intervals_95"],
            }
            for x in reports
        ],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
