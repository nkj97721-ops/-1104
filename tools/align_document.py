#!/usr/bin/env python3
"""Align complete English/Chinese text blocks locally, then extract phrase pairs."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.sentence_align import align_sentence_embeddings, encode_sentences, split_sentences
from tools.semantic_align import align_text_batch


def build_two_two_pair_evidence(match: dict[str, Any], source_sentences: Sequence[str],
                                target_sentences: Sequence[str], source_vectors,
                                target_vectors, threshold: float) -> dict[str, Any] | None:
    """Return ordered 1:1 model candidates inside a 2:2 group; never split the boundary."""
    source_indices, target_indices = match["source_indices"], match["target_indices"]
    if len(source_indices) != 2 or len(target_indices) != 2:
        return None
    pairs = []
    for si, ti in zip(source_indices, target_indices):
        source = [float(v) for v in source_vectors[si]]
        target = [float(v) for v in target_vectors[ti]]
        if len(source) != len(target):
            raise ValueError("Source and target sentence vector dimensions must match")
        sn = math.sqrt(sum(v*v for v in source))
        tn = math.sqrt(sum(v*v for v in target))
        score = sum(a*b for a,b in zip(source,target))/(sn*tn) if sn and tn else 0.0
        score = max(-1.0, min(1.0, score))
        pairs.append({"source_index":si,"target_index":ti,
                      "english":source_sentences[si],"chinese":target_sentences[ti],
                      "similarity":round(score,4),"meets_threshold":score >= threshold,
                      "status":"candidate_only"})
    return {
        "boundary_group_type":"2:2",
        "boundary_source_indices":list(source_indices),
        "boundary_target_indices":list(target_indices),
        "boundary_group_similarity":match["similarity"],
        "ordered_pair_candidates":pairs,
        "candidate_status":"two_ordered_pair_candidates" if all(p["meets_threshold"] for p in pairs)
                           else "insufficient_individual_pair_evidence",
        "interpretation":"模型候选，不是已验证语义金标准；原始 2:2 字幕边界组保持不变。",
    }


def align_document(english: str, chinese: str,
    sentence_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    word_model: str = "bert-base-multilingual-cased", threshold: float = 0.35,
    no_cuda: bool = False) -> dict[str, Any]:
    en_sentences, zh_sentences = split_sentences(english), split_sentences(chinese)
    if not en_sentences or not zh_sentences:
        raise ValueError("英文和中文都必须至少包含一个完整句子。")
    vectors = encode_sentences(en_sentences + zh_sentences, sentence_model, no_cuda)
    en_vectors, zh_vectors = vectors[:len(en_sentences)], vectors[len(en_sentences):]
    sentence_result = align_sentence_embeddings(en_vectors, zh_vectors, threshold=threshold)
    matched_text_pairs = [
        (" ".join(en_sentences[i] for i in m["source_indices"]),
         "".join(zh_sentences[i] for i in m["target_indices"]))
        for m in sentence_result["matches"]]
    word_results = align_text_batch(matched_text_pairs, word_model, no_cuda=no_cuda)
    all_source_tokens, all_target_tokens, all_links, all_candidates = [], [], [], []
    sentence_pairs, semantic_pair_candidates = [], []
    for pair_index, (match, aligned) in enumerate(zip(sentence_result["matches"], word_results)):
        en_text, zh_text = matched_text_pairs[pair_index]
        so, to = len(all_source_tokens), len(all_target_tokens)
        all_source_tokens.extend(aligned["source_tokens"])
        all_target_tokens.extend(aligned["target_tokens"])
        for link in aligned["links"]:
            all_links.append({**link,"source_index":link["source_index"]+so,
                              "target_index":link["target_index"]+to,"sentence_pair":pair_index})
        for candidate in aligned["phrase_candidates"]:
            all_candidates.append({**candidate,
                "source_span":[candidate["source_span"][0]+so,candidate["source_span"][1]+so],
                "target_span":[candidate["target_span"][0]+to,candidate["target_span"][1]+to],
                "sentence_pair":pair_index,"sentence_similarity":match["similarity"]})
        record = {**match,"english":en_text,"chinese":zh_text,
                  "phrase_candidate_count":len(aligned["phrase_candidates"])}
        evidence = build_two_two_pair_evidence(match,en_sentences,zh_sentences,
                                                en_vectors,zh_vectors,threshold)
        if evidence is not None:
            record["semantic_pair_evidence"] = evidence
            semantic_pair_candidates.append(evidence)
        sentence_pairs.append(record)
    return {
        "engine":"Multilingual BERT sentence alignment + batched AWESOME-align word alignment",
        "sentence_model":sentence_model,"word_model":word_model,
        "source_language":"en","target_language":"zh",
        "source_sentences":en_sentences,"target_sentences":zh_sentences,
        "boundary_groups":sentence_pairs,"sentence_pairs":sentence_pairs,
        "semantic_pair_candidates":semantic_pair_candidates,
        "unmatched_source_sentences":[{"index":i,"text":en_sentences[i]} for i in sentence_result["unmatched_source"]],
        "unmatched_target_sentences":[{"index":i,"text":zh_sentences[i]} for i in sentence_result["unmatched_target"]],
        "source_tokens":all_source_tokens,"target_tokens":all_target_tokens,
        "links":all_links,
        "phrase_candidates":sorted(all_candidates,key=lambda x:(-x["score"],x["source_span"][0],x["target_span"][0]))[:500],
        "summary":{
            "source_sentence_count":len(en_sentences),"target_sentence_count":len(zh_sentences),
            "matched_sentence_groups":len(sentence_pairs),
            "two_two_boundary_groups":len(semantic_pair_candidates),
            "two_two_groups_with_two_ordered_pair_candidates":sum(
                x["candidate_status"]=="two_ordered_pair_candidates" for x in semantic_pair_candidates),
            "unmatched_source_sentence_count":len(sentence_result["unmatched_source"]),
            "unmatched_target_sentence_count":len(sentence_result["unmatched_target"]),
            "aligned_link_count":len(all_links),"phrase_candidate_count":len(all_candidates)},
        "notice":("句子配对基于多语言句向量余弦相似度和单调动态规划，支持 1:1、1:2、2:1、2:2 及未匹配句子；"
                  "原始字幕边界组与 2:2 内部的有序 1:1 语义候选分开输出，候选不是已验证金标准。"
                  "当前不支持任意句子重排序。短语由词级对齐链接提取。所有分数均为模型/算法评分，"
                  "不是经过校准的翻译正确概率；请检查未匹配句子和短语候选。")
    }


def main():
    parser=argparse.ArgumentParser(description="Local document-level English/Chinese semantic alignment")
    parser.add_argument("--en",required=True,help="English text block")
    parser.add_argument("--zh",required=True,help="Chinese text block")
    parser.add_argument("--output",default="alignment.json")
    parser.add_argument("--sentence-model",default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--word-model",default="bert-base-multilingual-cased")
    parser.add_argument("--threshold",type=float,default=0.35)
    parser.add_argument("--no-cuda",action="store_true")
    args=parser.parse_args()
    result=align_document(args.en,args.zh,args.sentence_model,args.word_model,args.threshold,args.no_cuda)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["summary"],ensure_ascii=False,indent=2))
    print(f"已写入：{args.output}")

if __name__ == "__main__":
    main()
