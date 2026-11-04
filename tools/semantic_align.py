#!/usr/bin/env python3
"""Model-backed English/Chinese word and phrase alignment using AWESOME-align.

The model is downloaded by Hugging Face on first run. No text is sent to a
remote translation API; inference runs locally after the model is cached.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import jieba
except ImportError:  # Character fallback keeps basic tokenization usable.
    jieba = None

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*|[\u3400-\u9fff]+|[^\w\s]", re.UNICODE)


def tokenize_english(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def tokenize_chinese(text: str) -> list[str]:
    # Jieba gives the aligner lexical Chinese words instead of isolated Han
    # characters, while keeping punctuation as independent tokens.
    tokens: list[str] = []
    for chunk in TOKEN_RE.findall(text):
        if re.fullmatch(r"[\u3400-\u9fff]+", chunk):
            tokens.extend(token for token in (jieba.cut(chunk, HMM=True) if jieba else list(chunk)) if token.strip())
        else:
            tokens.append(chunk)
    return tokens


def parse_links(link_line: str, probability_line: str, src_len: int, tgt_len: int) -> list[dict[str, Any]]:
    raw_links = link_line.strip().split()
    # Upstream AWESOME-align writes tensor reprs containing spaces, so split()
    # would incorrectly turn one probability into several fields.
    tensor_values = re.findall(r"tensor\\(\\s*([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)", probability_line)
    raw_probs = tensor_values or (
        re.findall(r"[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?", probability_line)
        if probability_line.strip() else []
    )
    if raw_probs and len(raw_probs) != len(raw_links):
        raise ValueError("模型输出的对齐链接数与概率数不一致。")
    links = []
    for index, item in enumerate(raw_links):
        match = re.fullmatch(r"(\d+)-(\d+)", item)
        if not match:
            raise ValueError(f"无法解析模型对齐链接：{item}")
        src_index, tgt_index = map(int, match.groups())
        if src_index >= src_len or tgt_index >= tgt_len:
            raise ValueError(f"模型返回越界索引：{item}")
        if raw_probs:
            # Tensor repr may look like tensor(0.83, device='cuda:0').
            numeric = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", raw_probs[index])
            if not numeric:
                raise ValueError(f"无法解析模型对齐概率：{raw_probs[index]}")
            probability = float(numeric.group())
        else:
            probability = 0.0
        if not math.isfinite(probability):
            raise ValueError("模型返回了无效概率。")
        links.append({
            "source_index": src_index,
            "target_index": tgt_index,
            "probability": max(0.0, min(1.0, probability)),
        })
    return links


def extract_phrase_candidates(source: list[str], target: list[str],
                              links: list[dict[str, Any]],
                              max_source_tokens: int = 5,
                              max_target_tokens: int = 8) -> list[dict[str, Any]]:
    """Extract consistent phrase pairs from word links (Och-Ney style).

    A candidate is rejected if a target token inside its span is aligned to a
    source token outside the source span. Candidates are suggestions, not
    guaranteed translations; confidence is an alignment score, not a calibrated
    probability that the phrases are semantically equivalent.
    """
    by_source: dict[int, set[int]] = {}
    by_target: dict[int, set[int]] = {}
    probability_by_pair: dict[tuple[int, int], float] = {}
    for link in links:
        i, j = link["source_index"], link["target_index"]
        by_source.setdefault(i, set()).add(j)
        by_target.setdefault(j, set()).add(i)
        probability_by_pair[(i, j)] = max(probability_by_pair.get((i, j), 0.0), link["probability"])

    candidates: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for src_start in range(len(source)):
        for src_end in range(src_start, min(len(source), src_start + max_source_tokens)):
            aligned_targets = {
                j for i in range(src_start, src_end + 1)
                for j in by_source.get(i, set())
            }
            if not aligned_targets:
                continue
            min_target, max_target = min(aligned_targets), max(aligned_targets)
            for tgt_start in range(max(0, min_target - 2), min_target + 1):
                for tgt_end in range(max_target, min(len(target), max_target + max_target_tokens - (max_target - min_target + 1) + 1)):
                    if tgt_end - tgt_start + 1 > max_target_tokens:
                        continue
                    # Consistency: no target token in the candidate span may
                    # align to a source token outside this source span.
                    outside = any(
                        source_index < src_start or source_index > src_end
                        for target_index in range(tgt_start, tgt_end + 1)
                        for source_index in by_target.get(target_index, set())
                    )
                    if outside:
                        continue
                    inside_links = [
                        probability_by_pair[(i, j)]
                        for i in range(src_start, src_end + 1)
                        for j in by_source.get(i, set())
                        if tgt_start <= j <= tgt_end
                    ]
                    if not inside_links:
                        continue
                    covered_source = sum(bool(by_source.get(i, set()) & set(range(tgt_start, tgt_end + 1)))
                                         for i in range(src_start, src_end + 1))
                    covered_target = sum(bool(by_target.get(j, set()) & set(range(src_start, src_end + 1)))
                                         for j in range(tgt_start, tgt_end + 1))
                    coverage = (covered_source + covered_target) / (src_end - src_start + 1 + tgt_end - tgt_start + 1)
                    mean_alignment = sum(inside_links) / len(inside_links)
                    # Small penalty for very long spans; keep score interpretable.
                    length_penalty = 1.0 / (1.0 + 0.035 * ((src_end - src_start) + (tgt_end - tgt_start)))
                    score = max(0.0, min(1.0, coverage * mean_alignment * length_penalty))
                    key = (src_start, src_end, tgt_start, tgt_end)
                    candidates[key] = {
                        "source": " ".join(source[src_start:src_end + 1]),
                        "target": "".join(target[tgt_start:tgt_end + 1]),
                        "source_span": [src_start, src_end],
                        "target_span": [tgt_start, tgt_end],
                        "score": round(score, 4),
                        "coverage": round(coverage, 4),
                        "alignment_score": round(mean_alignment, 4),
                        "confidence_band": "high" if score >= 0.75 else ("medium" if score >= 0.45 else "low"),
                    }
    return sorted(candidates.values(), key=lambda item: (
        -item["score"], item["source_span"][0], item["target_span"][0]
    ))


def align_text(source_text: str, target_text: str, model: str,
               extraction: str = "softmax", no_cuda: bool = False) -> dict[str, Any]:
    source = tokenize_english(source_text)
    target = tokenize_chinese(target_text)
    if not source or not target:
        raise ValueError("英文和中文都必须包含可识别的文字。")
    executable = shutil.which("awesome-align")
    if not executable:
        raise RuntimeError(
            "找不到 awesome-align。请先安装：pip install git+https://github.com/neulab/awesome-align.git"
        )

    with tempfile.TemporaryDirectory(prefix="twinline-align-") as temp:
        root = Path(temp)
        data_file = root / "input.txt"
        links_file = root / "links.txt"
        probs_file = root / "probabilities.txt"
        data_file.write_text(" ".join(source) + " ||| " + " ".join(target) + "\n", encoding="utf-8")
        command = [
            executable, "--data_file", str(data_file),
            "--output_file", str(links_file),
            "--output_prob_file", str(probs_file),
            "--model_name_or_path", model,
            "--extraction", extraction,
            "--batch_size", "1",
            "--num_workers", "1",
        ]
        if no_cuda:
            command.append("--no_cuda")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError("语义对齐模型执行失败：\n" + details[-5000:])
        link_lines = links_file.read_text(encoding="utf-8").splitlines()
        prob_lines = probs_file.read_text(encoding="utf-8").splitlines()
        if not link_lines:
            raise RuntimeError("模型没有返回对齐结果。")
        links = parse_links(link_lines[0], prob_lines[0] if prob_lines else "", len(source), len(target))

    candidates = extract_phrase_candidates(source, target, links)
    high = sum(link["probability"] >= 0.5 for link in links)
    return {
        "engine": "AWESOME-align / multilingual BERT",
        "model": model,
        "source_language": "en",
        "target_language": "zh",
        "source_tokens": source,
        "target_tokens": target,
        "links": [
            {
                **link,
                "source": source[link["source_index"]],
                "target": target[link["target_index"]],
            }
            for link in links
        ],
        "phrase_candidates": candidates[:100],
        "summary": {
            "source_token_count": len(source),
            "target_token_count": len(target),
            "aligned_link_count": len(links),
            "links_at_or_above_0_5": high,
            "candidate_count": len(candidates),
        },
        "notice": (
            "分数来自模型对齐矩阵的启发式链接概率，不是语义正确率或经过校准的置信概率。"
            "短语候选按词级对齐一致性提取；请检查候选，尤其是习语、省略、倒装和长距离语序变化。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local model-backed English/Chinese semantic alignment")
    parser.add_argument("--en", required=True, help="English source sentence")
    parser.add_argument("--zh", required=True, help="Chinese translation")
    parser.add_argument("--output", default="alignment.json", help="JSON output path")
    parser.add_argument("--model", default="bert-base-multilingual-cased", help="Hugging Face model name or local path")
    parser.add_argument("--extraction", choices=["softmax", "entmax15"], default="softmax")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    result = align_text(args.en, args.zh, args.model, args.extraction, args.no_cuda)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"已写入：{args.output}")


if __name__ == "__main__":
    main()
