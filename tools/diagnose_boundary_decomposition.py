#!/usr/bin/env python3
"""Separate subtitle-boundary grouping from model sentence-group predictions.

When the companion 2:2 embedding report is supplied, every 2:2 boundary is
joined to its diagonal/cross cosine evidence. This is diagnostic evidence,
never sentence-level semantic ground truth.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def classify_boundary_group(source_indices, target_indices, predicted_groups):
    source, target = tuple(source_indices), tuple(target_indices)
    predicted = [
        {"source": list(item["source"]), "target": list(item["target"])}
        for item in predicted_groups
    ]
    if any(tuple(p["source"]) == source and tuple(p["target"]) == target for p in predicted):
        return "exact_boundary_group", []
    if len(source) == 2 and len(target) == 2:
        ordered = [(source[0], target[0]), (source[1], target[1])]
        crossed = [(source[0], target[1]), (source[1], target[0])]

        def has_pair(pair):
            return any(p["source"] == [pair[0]] and p["target"] == [pair[1]] for p in predicted)

        if all(has_pair(pair) for pair in ordered):
            return "split_into_two_ordered_1_to_1", [
                {"source": [s], "target": [t]} for s, t in ordered
            ]
        if all(has_pair(pair) for pair in crossed):
            return "split_into_two_crossed_1_to_1", [
                {"source": [s], "target": [t]} for s, t in crossed
            ]
    related = [
        p for p in predicted
        if set(p["source"]).intersection(source) or set(p["target"]).intersection(target)
    ]
    covered_source = {i for p in related for i in p["source"] if i in source}
    covered_target = {i for p in related for i in p["target"] if i in target}
    if covered_source == set(source) and covered_target == set(target):
        return "covered_by_other_grouping", related
    return "partial_or_missed", related


def _candidate_index(candidate_report):
    index = {}
    for model in candidate_report.get("model_diagnostics", []):
        rows = {}
        for row in model.get("all_group_diagnostics", []):
            key = (
                row.get("document_id"),
                tuple(row.get("source_boundary_ids", [])),
                tuple(row.get("target_boundary_ids", [])),
            )
            rows[key] = row
        index[model.get("model")] = rows
    return index


def diagnose_model(model_report, candidate_rows=None):
    totals = defaultdict(Counter)
    examples = []
    two_two_evidence = defaultdict(list)
    two_two_total = 0
    two_two_joined = 0
    for document in model_report.get("heldout_document_errors", []):
        predicted = document.get("predicted", [])
        for boundary in document.get("original_boundary_groups", []):
            source, target = boundary.get("source", []), boundary.get("target", [])
            group_type = boundary.get("type") or f"{len(source)}:{len(target)}"
            classification, fragments = classify_boundary_group(source, target, predicted)
            totals[group_type][classification] += 1
            if len(source) == 2 and len(target) == 2 and candidate_rows is not None:
                two_two_total += 1
                boundary_id = boundary.get("boundary_id")
                key = (document.get("id"), (boundary_id,), (boundary_id,))
                evidence = candidate_rows.get(key)
                if evidence:
                    two_two_joined += 1
                    two_two_evidence[classification].append(evidence)
            if classification != "exact_boundary_group" and len(examples) < 200:
                examples.append({
                    "document_id": document.get("id"),
                    "boundary_id": boundary.get("boundary_id"),
                    "group_type": group_type,
                    "source_indices": source,
                    "target_indices": target,
                    "classification": classification,
                    "predicted_fragments": fragments,
                })
    by_type = {
        group_type: {
            "reference_groups": sum(counts.values()),
            "classifications": dict(sorted(counts.items())),
        }
        for group_type, counts in sorted(totals.items())
    }
    correlation = {}
    for classification, rows in sorted(two_two_evidence.items()):
        margins = [row["ordered_minus_cross_margin"] for row in rows]
        correlation[classification] = {
            "groups_with_embedding_evidence": len(rows),
            "positive_ordered_minus_cross_margin": sum(x > 0 for x in margins),
            "zero_margin": sum(x == 0 for x in margins),
            "negative_margin": sum(x < 0 for x in margins),
            "mean_margin": round(mean(margins), 6) if margins else None,
            "median_margin": round(median(margins), 6) if margins else None,
            "both_ordered_pairs_over_threshold": sum(
                row.get("pairs_over_threshold") == 2 for row in rows
            ),
        }
    return {
        "model": model_report.get("model"),
        "by_group_type": by_type,
        "two_two_embedding_join_coverage": {
            "reference_2_2_boundary_groups": two_two_total,
            "joined_groups": two_two_joined,
            "missing_groups": two_two_total - two_two_joined,
            "complete": two_two_total == two_two_joined,
        } if candidate_rows is not None else None,
        "two_two_embedding_evidence_by_dp_classification": correlation,
        "non_exact_examples_limit": 200,
        "non_exact_examples": examples,
    }


def build_report(comparison_report, candidate_report=None):
    candidate_models = _candidate_index(candidate_report) if candidate_report else {}
    models = [
        diagnose_model(item, candidate_models.get(item.get("model")))
        for item in comparison_report.get("model_comparisons", [])
    ]
    return {
        "status": "AUTOMATIC_SILVER_BOUNDARY_DECOMPOSITION_WITH_EMBEDDING_EVIDENCE_NOT_SEMANTIC_ACCURACY",
        "reference_type": comparison_report.get("reference_type"),
        "languages": comparison_report.get("languages"),
        "split_unit": comparison_report.get("split_unit"),
        "models": models,
        "limitations": [
            "Original subtitle-boundary group agreement and sentence-level semantic correspondence are different evaluation targets.",
            "A 2:2 boundary decomposed into two ordered 1:1 DP groups is a grouping disagreement, not proof of semantic error or success.",
            "Cosine similarities and diagonal-versus-cross margins are model evidence, not verified sentence-level semantic labels.",
            "The automatic silver reference does not identify independently verified sentence-level links inside 2:2 groups.",
            "No result in this report is human-verified semantic accuracy.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="OpenSubtitles automatic-silver comparison JSON")
    parser.add_argument("--candidates", help="Companion 2:2 candidate embedding diagnostic JSON")
    parser.add_argument("--output", default="reports/opensubtitles2024-boundary-decomposition.json")
    args = parser.parse_args()
    comparison = json.loads(Path(args.input).read_text(encoding="utf-8"))
    candidates = (
        json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        if args.candidates else None
    )
    report = build_report(comparison, candidates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "models": [
            {
                "model": model["model"],
                "by_group_type": model["by_group_type"],
                "two_two_embedding_evidence_by_dp_classification": model["two_two_embedding_evidence_by_dp_classification"],
            }
            for model in report["models"]
        ],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
