"""Sentence segmentation and monotonic document-level alignment.

The dynamic program supports 1:1, 1:2 and 2:1 sentence groups, plus
unmatched sentences on either side. Embeddings are supplied by the caller.
"""
from __future__ import annotations

import re
from typing import Sequence

SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s*")


def split_sentences(text: str) -> list[str]:
    """Split English/Chinese text while preserving terminal punctuation."""
    return [part.strip() for part in SENTENCE_END.split(text.strip()) if part.strip()]


def align_sentence_embeddings(
    source_vectors: Sequence[Sequence[float]],
    target_vectors: Sequence[Sequence[float]],
    threshold: float = 0.35,
    gap_penalty: float = 0.12,
    merge_penalty: float = 0.06,
    similarity_tables=None,
) -> dict:
    """Globally align sentence vectors with a monotonic dynamic program.

    Similarity is cosine similarity. A merge transition compares the mean
    vector of two adjacent sentences with one sentence on the opposite side.
    The function is model-agnostic, which makes the DP independently testable.
    """
    import math

    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) ** 2 for x in a))
        nb = math.sqrt(sum(float(y) ** 2 for y in b))
        if not na or not nb:
            return 0.0
        return max(-1.0, min(1.0, dot / (na * nb)))

    def mean(vectors: Sequence[Sequence[float]]) -> list[float]:
        width = len(vectors[0])
        return [sum(float(v[k]) for v in vectors) / len(vectors) for k in range(width)]

    n, m = len(source_vectors), len(target_vectors)
    neg_inf = float("-inf")
    dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    def update(ni: int, nj: int, value: float, previous: tuple, operation: tuple) -> None:
        if value > dp[ni][nj]:
            dp[ni][nj] = value
            back[ni][nj] = (previous, operation)

    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]
            if current == neg_inf:
                continue
            if i < n:
                update(i + 1, j, current - gap_penalty, (i, j), ("source_gap", 1, 0, 0.0))
            if j < m:
                update(i, j + 1, current - gap_penalty, (i, j), ("target_gap", 0, 1, 0.0))

            transitions = []
            if i < n and j < m:
                score = (float(similarity_tables[0][i][j]) if similarity_tables is not None
                         else cosine(source_vectors[i], target_vectors[j]))
                transitions.append((1, 1, score, 0.0))
            if i < n and j + 1 < m:
                score = (float(similarity_tables[1][i][j]) if similarity_tables is not None
                         else cosine(source_vectors[i], mean(target_vectors[j:j + 2])))
                transitions.append((1, 2, score, merge_penalty))
            if i + 1 < n and j < m:
                score = (float(similarity_tables[2][i][j]) if similarity_tables is not None
                         else cosine(mean(source_vectors[i:i + 2]), target_vectors[j]))
                transitions.append((2, 1, score, merge_penalty))

            for take_source, take_target, similarity, penalty in transitions:
                if similarity < threshold:
                    continue
                reward = similarity - threshold - penalty
                update(
                    i + take_source, j + take_target, current + reward,
                    (i, j), ("match", take_source, take_target, similarity),
                )

    operations = []
    i, j = n, m
    while i or j:
        entry = back[i][j]
        if entry is None:
            raise RuntimeError("句子对齐动态规划未能回溯完整路径。")
        (pi, pj), operation = entry
        kind, take_source, take_target, similarity = operation
        if kind == "match":
            operations.append({
                "source_indices": list(range(pi, pi + take_source)),
                "target_indices": list(range(pj, pj + take_target)),
                "similarity": round(similarity, 4),
                "source_count": take_source,
                "target_count": take_target,
            })
        elif kind == "source_gap":
            operations.append({"unmatched_source": pi})
        else:
            operations.append({"unmatched_target": pj})
        i, j = pi, pj

    operations.reverse()
    return {
        "matches": [item for item in operations if "source_indices" in item],
        "unmatched_source": [item["unmatched_source"] for item in operations if "unmatched_source" in item],
        "unmatched_target": [item["unmatched_target"] for item in operations if "unmatched_target" in item],
        "objective_score": round(dp[n][m], 4),
        "threshold": threshold,
    }


def encode_sentences(
    sentences: Sequence[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    no_cuda: bool = False,
    batch_size: int = 16,
) -> list[list[float]]:
    """Create local multilingual sentence embeddings trained for semantic similarity."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cpu" if no_cuda or not torch.cuda.is_available() else "cuda"
    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        list(sentences),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()
