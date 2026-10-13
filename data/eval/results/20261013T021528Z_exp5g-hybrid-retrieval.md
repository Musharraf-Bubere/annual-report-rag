# Evaluation Report: `exp5g-hybrid-retrieval` (20261013T021528Z)

- **Mode**: `retrieval-only`
- **Split**: `dev`
- **Repeats**: `1`
- **Status**: `completed`
- **Questions evaluated**: `30`
- **Skipped draft questions**: `0` (excluded by default because status='draft_needs_fill' (missing expected answers and/or PDF pages))

> [!WARNING]
> **PRELIMINARY RESULTS**: one or more questions have `human_verified=false`. Treat these metrics as directional engineering signals rather than statistically final benchmarks.

## 1. Retrieval Metrics (Answerable Questions)

| Metric | Value |
| :--- | :--- |
| `page_hit@1` | 0.4400 |
| `page_hit@3` | 0.7600 |
| `page_hit@6` | 0.9600 |
| `MRR` | 0.6460 |
| `text_hit_any@1` | 0.7200 |
| `text_hit_any@3` | 0.8400 |
| `text_hit_any@6` | 1.0000 |
| `text_hit_all@1` | 0.6800 |
| `text_hit_all@3` | 0.8400 |
| `text_hit_all@6` | 1.0000 |
| `text_hit_any@context` | 1.0000 |
| `text_hit_all@context` | 1.0000 |
| `distinct_pages@6` | 4.92 |
| `mean_context_chunks` | 10.00 |

## 2. Answer Metrics

_Skipped in `--retrieval-only` mode._

## 3. Performance & Token Usage

| Metric | Value |
| :--- | :--- |
| `retrieve_ms` (n=30) p50 / p95 / max | 258.07 ms (n=30) / 604.35 ms (n=30) / 791.87 ms (n=30) |
| `mean_context_chunks` | 10.00 |

> [!NOTE]
> `p95` latency is not statistically meaningful below 20 samples (`n < 20`) and is heavily influenced by the single `max` sample.

## 4. Slices by Question Type

| Type | Count | `page_hit@6` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` / `CORRECT_REFUSAL` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fact` | 22 | 0.9545 | 1.0000 | 1.0000 | N/A |
| `comparison` | 2 | 1.0000 | 1.0000 | 1.0000 | N/A |
| `table` | 1 | 1.0000 | 1.0000 | 1.0000 | N/A |
| `unanswerable` | 5 | N/A | N/A | N/A | N/A |

## 5. Slices by Expected-Page Text Quality (Answerable Questions)

| Expected-Page Quality | Count | `page_hit@6` | `MRR` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_clean` | 14 | 0.9286 | 0.6452 | 1.0000 | 1.0000 | N/A |
| `all_degraded` | 8 | 1.0000 | 0.5771 | 1.0000 | 1.0000 | N/A |
| `mixed` | 3 | 1.0000 | 0.8333 | 1.0000 | 1.0000 | N/A |

## 6. Failed Questions

- **`q34`** (`fact`): Which firm is TCS's current statutory auditor?
  - Outcome: `None` | `page_hit@6`: `False` | `text_hit_any@6`: `True` | `text_hit_all@6`: `True`
  - Top 6 retrieved: tcs-ar-26 PDF p90 (0.0164), tcs-ar-26 PDF p163 (0.0149), tcs-ar-26 PDF p92 (0.0142), tcs-ar-26 PDF p4 (0.0081), tcs-ar-26 PDF p162 (0.0081), tcs-ar-26 PDF p4 (0.0079)
