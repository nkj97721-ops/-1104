# Real corpus: OpenSubtitles2024 EN → Simplified Chinese

## Chosen corpus and rationale

Use the OpenSubtitles2024 multilingual set with the exact language pair
`en` → `zh_CN`. Do not load `zh_TW`, Cantonese (`yue`), or any other
Chinese variant into this benchmark.

Upstream resources:

- Dataset repository and extraction notes:
  https://github.com/Helsinki-NLP/OpenSubtitles-devtest
- Multilingual testset linksets:
  https://github.com/Helsinki-NLP/OpenSubtitles-devtest/tree/master/devtest-xml/multi-linksets-0.9/2024
- Upstream paper: Tiedemann & Luo (2026), OpenSubtitles2024, LREC 2026,
  https://doi.org/10.63317/4ivg578ub2ob

The upstream project describes the multilingual package as synchronized
subtitle data for 40 languages and 16 subtitle files. Its XML links are
automatic subtitle-timing alignments, not human semantic judgments. They are
therefore useful silver proposals and a source of difficult many-to-many
examples, but must not be called gold.

The selected archive is pinned by Git blob SHA-1 in
`tools/prepare_opensubtitles2024_gold.py`. The script verifies the archive
before reading it and extracts only English and `zh_CN`.

## Build the review queue

```bash
python tools/prepare_opensubtitles2024_gold.py \
  --output-dir data/opensubtitles2024-review \
  --max-lines-per-document 80
```

This creates:

- `source/opensubtitles2024-multiset.zip`: downloaded source archive; do not
  commit or redistribute it.
- `annotation_queue.jsonl`: document chunks with independently segmented
  English and Simplified Chinese text, plus proposed groups from upstream
  subtitle timing.
- `manifest.json`: archive identity, movie-level split, counts, and proposed
  1:1 / 1:2 / 2:1 / 2:2 / larger-group distribution.

The split is by complete movie ID, not by neighboring sentence windows. This
prevents chunks from one movie leaking across calibration and held-out sets.
The last 25% of sorted movie IDs are assigned to held-out test; the rest are
calibration. Keep this split frozen after annotation begins.

## Human adjudication

```bash
python tools/review_gold_queue.py \
  --input data/opensubtitles2024-review/annotation_queue.jsonl \
  --output data/gold_alignment_human.jsonl \
  --annotator "your-reviewer-id"
```

For every document, inspect the English and Simplified Chinese text, then:

- `a`: accept the proposed groups only after checking every link.
- `e`: enter a corrected JSON array of groups.
- `s`: skip the document for later review.

The reviewer must correct segmentation or grouping errors. Any group larger
than 2 sentences on either side requires manual resegmentation into supported
groups or explicit unmatched sentences. The reviewer CLI records
`annotation_status=human_verified`, annotator identity, and review timestamp.
The final evaluator rejects pending or silver records.

## First empirical model comparison

The GitHub Actions workflow downloads the real corpus, creates the queue, and
runs MiniLM and LaBSE against the upstream timing links on a movie-disjoint
calibration/held-out split. That output is explicitly labeled
`PRELIMINARY_SILVER_EVALUATION_NOT_HUMAN_GOLD`.

This run is useful to:
- prove that both embedding models run on the actual corpus;
- measure agreement with the upstream alignment proposals;
- expose 1:2, 2:1, 2:2 and unmatched cases;
- identify error examples for human review.

It is not the final accuracy result. Once a sufficiently large, representative
set has been manually adjudicated, run:

```bash
python tools/benchmark_gold_groups.py \
  --input data/gold_alignment_human.jsonl \
  --output reports/human-gold-model-comparison.json
```

Parameter tuning uses calibration documents only. The independent test movies
are evaluated once after the model and parameters have been selected.

## Data handling

The repository stores scripts and schemas only. Keep downloaded subtitle text,
review queues, and human annotations in local or access-controlled storage.
Retain upstream attribution and check the source dataset terms before any
redistribution or commercial use.
