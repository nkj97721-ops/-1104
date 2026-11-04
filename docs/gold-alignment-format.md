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
- Supported group shapes are 1:1, 1:2 and 2:1. Groups must be contiguous and
  monotonic; a sentence cannot occur in more than one group.
- Sentences omitted from every gold group are treated as explicitly unmatched.
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

The benchmark does not manufacture annotations from the legacy TSV. Until a
representative human-reviewed calibration set and independent test set exist,
no new aggregate accuracy should be described as a validated real-world score.
