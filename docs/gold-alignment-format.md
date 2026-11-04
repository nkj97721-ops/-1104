# Human-annotated sentence-alignment gold data

The legacy TSV benchmark treats each row as a gold 1:1 pair. Do not use it to
claim accuracy on documents where sentence boundaries differ. The new
`tools/benchmark_gold_groups.py` evaluates explicit document-level groups.

## JSONL schema

One JSON object per document:

```json
{
  "id": "doc-001",
  "split": "calibration",
  "annotation_status": "human_verified",
  "annotator": "reviewer-id",
  "reviewed_at": "2026-11-04T20:00:00+08:00",
  "source": ["English sentence one.", "English sentence two."],
  "target": ["第一句中文。", "第二句中文。", "第三句中文。"],
  "gold": [
    {"source": [0], "target": [0, 1]},
    {"source": [1], "target": [2]}
  ]
}
```

- `source` and `target` are independently segmented sentence arrays.
- All indices are zero-based.
- Each gold item is one alignment group, not one individual sentence pair.
- Supported group shapes are 1:1, 1:2, 2:1 and 2:2. Groups must be contiguous and
  monotonic; a sentence cannot occur in more than one group.
- Sentences omitted from every gold group are treated as explicitly unmatched.
- `annotation_status` must be `human_verified`, with a non-empty `annotator` and `reviewed_at` field. Suggested/automatic labels must never be copied into `gold` without human review.
- `split` must be `calibration` or `test`. Keep test documents separate from
  all parameter selection. Prefer different source documents, not merely
  different sentence windows from the same document.
- Annotators should inspect the original parallel passages and mark semantic
  correspondence, not infer gold labels from the model's output.

## Run

```bash
python tools/benchmark_gold_groups.py \
  --input data/gold_alignment.jsonl \
  --output reports/gold-group-benchmark.json
```

The script tunes threshold and DP penalties on calibration documents only,
then evaluates the frozen parameters on test documents once. It reports:

1. Exact alignment-group precision/recall/F1. A predicted 1:1 + 1:1 split does
   not count as an exact 1:2 group, even if its sentence links overlap.
2. Link-level precision/recall/F1, where a group contributes the Cartesian
   source-target index links.
3. Unmatched-sentence precision/recall/F1.
4. Per-document predicted and gold groups for error inspection.

The benchmark does not manufacture annotations from silver subtitle timing
links or the legacy TSV. By default it requires at least 10 human-reviewed
calibration documents, 10 independent test documents, and 3 distinct
movie/document IDs in each split. These minimums prevent a tiny convenience
sample from being reported as a reliable corpus-level score.

## Real-corpus workflow

- `tools/prepare_opensubtitles2024_gold.py` downloads the official English /
  Simplified Chinese (`zh_CN`) OpenSubtitles2024 multilingual set and creates
  a review queue. The upstream automatic subtitle-timing links are proposals,
  never human gold.
- `tools/review_gold_queue.py` presents each document and requires an
  annotator to inspect the English and Simplified Chinese text before accepting
  or editing the proposed groups.
- `tools/benchmark_silver_groups.py` compares MiniLM and LaBSE against the
  automatic timing links as a preliminary diagnostic only.
- `tools/benchmark_gold_groups.py` is the final benchmark and refuses records
  that do not explicitly carry `annotation_status=human_verified`,
  `annotator`, and `reviewed_at`.

Both MiniLM and LaBSE are calibrated independently by default and each is evaluated on the same held-out documents. The deployable candidate is selected using calibration scores only; held-out scores are never used for model selection.
