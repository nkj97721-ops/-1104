"""Reconstruct upstream line/segment groups without mixing them into semantics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def reconstruct_boundary_groups(
    source_origins: list[dict[str, Any]],
    target_origins: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild paired raw-boundary groups from per-sentence provenance.

    This is a structural projection only. It must never be passed off as a
    semantic sentence alignment or fed into the embedding/DP model.
    """
    by_id: dict[str, dict[str, Any]] = {}

    for side, origins in (("source", source_origins), ("target", target_origins)):
        seen_sentence_indices: set[int] = set()
        for origin in origins:
            if not isinstance(origin, dict):
                raise ValueError(f"{side} sentence origin must be an object")
            boundary_id = str(origin.get("boundary_id", "")).strip()
            sentence_index = origin.get("sentence_index")
            boundary_index = origin.get("boundary_index")
            if not boundary_id:
                raise ValueError(f"{side} sentence origin is missing boundary_id")
            if type(sentence_index) is not int or sentence_index < 0:
                raise ValueError(f"{side} sentence origin has invalid sentence_index")
            if type(boundary_index) is not int or boundary_index < 0:
                raise ValueError(f"{side} sentence origin has invalid boundary_index")
            if sentence_index in seen_sentence_indices:
                raise ValueError(f"{side} sentence_index {sentence_index} occurs more than once")
            seen_sentence_indices.add(sentence_index)

            entry = by_id.setdefault(boundary_id, {
                "boundary_index": boundary_index, "source": [], "target": [],
            })
            if entry["boundary_index"] != boundary_index:
                raise ValueError(f"boundary_id {boundary_id} has inconsistent boundary_index")
            entry[side].append(sentence_index)

    ordered = sorted(by_id.items(), key=lambda pair: pair[1]["boundary_index"])
    boundary_indices = [entry["boundary_index"] for _, entry in ordered]
    if len(boundary_indices) != len(set(boundary_indices)):
        raise ValueError("Distinct boundary IDs share the same boundary_index")

    groups = []
    unmatched_source = []
    unmatched_target = []
    for boundary_id, entry in ordered:
        source = sorted(entry["source"])
        target = sorted(entry["target"])
        for side, indices in (("source", source), ("target", target)):
            if indices and indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"{side} sentences for boundary {boundary_id} are not contiguous")
        if source and target:
            groups.append({
                "boundary_id": boundary_id,
                "source": source,
                "target": target,
                "type": f"{len(source)}:{len(target)}",
            })
        else:
            unmatched_source.extend(source)
            unmatched_target.extend(target)

    return {
        "groups": groups,
        "unmatched_source": sorted(unmatched_source),
        "unmatched_target": sorted(unmatched_target),
    }
