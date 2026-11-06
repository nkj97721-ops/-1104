#!/usr/bin/env python3
"""Separate original subtitle-boundary groups from predicted sentence groupings.

This diagnostic classifies how each automatic-silver boundary group is represented
by the model's monotonic DP output. In particular, a 2:2 boundary can be preserved
as 2:2 or decomposed into two ordered 1:1 predictions. Neither outcome proves
sentence-level semantic correctness.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def classify_boundary_group(
    source_indices: list[int],
    target_indices: list[int],
    predicted_groups: list[dict[str, list[int]]],
) -> tuple[str, list[dict[str, list[int]]]]:
    """Classify one original boundary group against predicted DP groups."""
    source = tuple(source_indices)
    target = tuple(target_indices)
    predicted = [
        {"source": list(item["source"]), "target": list(item["target"])}
        for item in predicted_groups
    ]
    if any(tuple(p["source"]) == source and tuple(p["target"]) == target for p in predicted):
        return "exact_boundary_group", []

    if len(source) == 2 and len(target) == 2:
        ordered = [
            (source[0], target[0]),
            (source[1], target[1]),
        ]
        crossed = [
            (source[0], target[1]),
            (source[1], target[0]),
        ]

        def has_pair(pair: tuple[int, int]) -> bool:
            return any(p["source"] == [pair[0]] and p["target"] == [pair[1]] for p in predicted)

        if all(has_pair(pair) for pair in ordered):
            fragments = [
                {"source": [s], "target": [t]} for s, t in ordered
            ]
            return "split_into_two_ordered_1_to_1", fragments
        if all(has_pair(pair) for pair in crossed):
            fragments = [
                {"source": [s], "target": [t]} for s, t in crossed
            ]
            return "split_into_two_crossed_1_to_1", fragments

    related = [
        p for p in predicted
        if set(p["source"]).intersection(source) or set(p["target"]).intersection(target)
    ]
    covered_source = {i for p in related for i in p["source"] if i in source}
    covered_target = {i for p in related for i in p["target"] if i in target}
    if covered_source == set(source) and covered_target == set(target):
        return "covered_by_other_grouping", related
    return "partial_or_missed", related


def diagnose_model(model_report: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    for document in model_report.get("heldout_document_errors", []):
        predicted = document.get("predicted", [])
        for boundary in document.get("original_boundary_groups", []):
            source = boundary.get("source", [])
            target = boundary.get("target", [])
            group_type = boundary.get("type") or f"{len(source)}:{len(target)}"
            classification, fragments = classify_boundary_group(source, target, predicted)
            totals[group_type][classification] += 1
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
    return {
        "model": model_report.get("model"),
        "by_group_type": by_type,
        "non_exact_examples_limit": 200,
        "non_exact_examples": examples,
    }


def build_report(comparison_report: dict[str, Any]) -> dict[str, Any]:
    models = [diagnose_model(item) for item in comparison_report.get("model_comparisons", [])]
    return {
        "status": "AUTOMATIC_SILVER_BOUNDARY_DECOMPOSITION_DIAGNOSTIC_NOT_SEMANTIC_ACCURACY",
        "reference_type": comparison_report.get("reference_type"),
        "languages": comparison_report.get("languages"),
        "split_unit": comparison_report.get("split_unit"),
        "models": models,
        "limitations": [
            "Original subtitle-boundary group agreement and sentence-level semantic correspondence are different evaluation targets.",
            "A 2:2 boundary decomposed into two ordered 1:1 DP groups is a grouping disagreement, not proof of a semantic error or a semantic success.",
            "The automatic silver reference does not identify independently verified sentence-level links inside 2:2 groups.",
            "No result in this report is human-verified semantic accuracy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="OpenSubtitles automatic-silver comparison JSON")
    parser.add_argument("--output", default="reports/opensubtitles2024-boundary-decomposition.json")
    args = parser.parse_args()
    comparison = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = build_report(comparison)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "models": [
            {
                "model": model["model"],
                "by_group_type": model["by_group_type"],
            }
            for model in report["models"]
        ],
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
