# Real corpus: fully automatic EN → Simplified Chinese silver benchmark

## Objective

Run two cross-lingual sentence embedding models against automatically generated
reference alignments. No human annotation or adjudication is required.

The benchmark measures agreement with upstream alignment metadata. It does
**not** claim to measure verified semantic truth. Reports must retain the
silver-reference label and must not be called final semantic accuracy.

## Sources

- OpenSubtitles2024: exact pair `en` → `zh_CN`; only English and Simplified
  Chinese are extracted. Movie-level split keeps complete movie IDs isolated
  between calibration and held-out test.
- WMT News Commentary v16: calibration documents.
- WMT17 EN-ZH: held-out test documents, joined by original document and segment
  IDs. Sentence boundaries are independently detected on both sides.

Traditional Chinese and Cantonese are excluded. The source segment/timing
relationships are retained as weak silver references.

## Run locally

```bash
python -m pip install -r requirements-benchmark.txt

python tools/prepare_opensubtitles2024_gold.py \
  --output-dir data/opensubtitles2024-silver \
  --max-lines-per-document 80

python tools/prepare_wmt_gold_queue.py \
  --output-dir data/wmt-silver \
  --calibration-documents 20 \
  --max-segments-per-document 40

python tools/benchmark_silver_groups.py \
  --input data/opensubtitles2024-silver/silver_reference.jsonl \
  --output reports/opensubtitles2024-silver-comparison.json \
  --max-documents-per-split 20 --no-cuda

python tools/benchmark_silver_groups.py \
  --input data/wmt-silver/silver_reference.jsonl \
  --output reports/wmt17-silver-comparison.json \
  --max-documents-per-split 20 --no-cuda
```

## Report contract

Each run reports:

- exact-group Precision / Recall / F1;
- separate 1:1, 1:2, 2:1 and 2:2 group metrics;
- sentence-link Precision / Recall / F1;
- document-cluster bootstrap 95% confidence intervals;
- macro document F1;
- per-document false-positive and missed-group examples;
- sampled reference-group counts by split.

Groups larger than 2 sentences on either side are excluded region-by-region,
not converted into false unmatched labels. Unmatched metrics are marked
not-estimable when the reference split contains no unmatched examples.

Calibration parameters are selected only on calibration documents. Held-out
test documents are not used for parameter selection. OpenSubtitles splits are
movie-disjoint; WMT splits are domain/document-disjoint.

## GitHub Actions

The `Real corpus alignment benchmark` workflow downloads both sources,
constructs the automatic silver references, runs MiniLM and LaBSE, and uploads
the manifests and comparison reports as an artifact. No review queue or manual
approval is part of the workflow.

## Data handling

Downloaded subtitle text and reports are retained as workflow artifacts rather
than committed to the repository. Preserve upstream attribution and check
source dataset terms before redistribution or commercial use.

## Optional human-gold research path

The separate `review_gold_queue.py` and `benchmark_gold_groups.py` tools
remain available for research that explicitly wants manually adjudicated gold.
They are optional and are not required to build, run, or continue the
automatic benchmark.
