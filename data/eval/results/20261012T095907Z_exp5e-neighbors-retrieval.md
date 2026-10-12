# Evaluation Report: `exp5e-neighbors-retrieval` (20261012T095907Z)

- **Mode**: `retrieval-only`
- **Split**: `dev`
- **Repeats**: `1`
- **Status**: `completed`
- **Questions evaluated**: `29`
- **Skipped draft questions**: `1` (excluded by default because status='draft_needs_fill' (missing expected answers and/or PDF pages))

> [!WARNING]
> **PRELIMINARY RESULTS**: fewer than 30 questions (29), one or more questions have `human_verified=false`. Treat these metrics as directional engineering signals rather than statistically final benchmarks.

## 1. Retrieval Metrics (Answerable Questions)

| Metric | Value |
| :--- | :--- |
| `page_hit@1` | 0.4800 |
| `page_hit@3` | 0.8000 |
| `page_hit@6` | 0.8400 |
| `MRR` | 0.6413 |
| `text_hit_any@1` | 0.6000 |
| `text_hit_any@3` | 0.7200 |
| `text_hit_any@6` | 0.8800 |
| `text_hit_all@1` | 0.5600 |
| `text_hit_all@3` | 0.7200 |
| `text_hit_all@6` | 0.8800 |
| `text_hit_any@context` | 0.9200 |
| `text_hit_all@context` | 0.9200 |
| `distinct_pages@6` | 4.88 |
| `mean_context_chunks` | 9.41 |

## 2. Answer Metrics

_Skipped in `--retrieval-only` mode._

## 3. Performance & Token Usage

| Metric | Value |
| :--- | :--- |
| `retrieve_ms` (n=29) p50 / p95 / max | 52.33 ms (n=29) / 72.36 ms (n=29) / 100.48 ms (n=29) |
| `mean_context_chunks` | 9.41 |

> [!NOTE]
> `p95` latency is not statistically meaningful below 20 samples (`n < 20`) and is heavily influenced by the single `max` sample.

## 4. Slices by Question Type

| Type | Count | `page_hit@6` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` / `CORRECT_REFUSAL` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fact` | 22 | 0.8182 | 0.8636 | 0.8636 | N/A |
| `comparison` | 2 | 1.0000 | 1.0000 | 1.0000 | N/A |
| `table` | 1 | 1.0000 | 1.0000 | 1.0000 | N/A |
| `unanswerable` | 4 | N/A | N/A | N/A | N/A |

## 5. Slices by Expected-Page Text Quality (Answerable Questions)

| Expected-Page Quality | Count | `page_hit@6` | `MRR` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_clean` | 14 | 0.8571 | 0.6333 | 0.9286 | 0.9286 | N/A |
| `all_degraded` | 8 | 0.7500 | 0.5833 | 0.7500 | 0.7500 | N/A |
| `mixed` | 3 | 1.0000 | 0.8333 | 1.0000 | 1.0000 | N/A |

## 6. Failed Questions

- **`q03`** (`fact`): What dividend did TCS declare for fiscal 2026?
  - Outcome: `None` | `page_hit@6`: `True` | `text_hit_any@6`: `False` | `text_hit_all@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p33 (0.7891), tcs-ar-26 PDF p24 (0.7534), tcs-ar-26 PDF p36 (0.7522), tcs-ar-26 PDF p46 (0.7487), tcs-ar-26 PDF p4 (0.7343), tcs-ar-26 PDF p322 (0.7284)
- **`q05`** (`fact`): What was TCS's net profit margin for fiscal 2026 (profit attributable to shareholders, excluding exceptional items, as a percentage of revenue)?
  - Outcome: `None` | `page_hit@6`: `False` | `text_hit_any@6`: `False` | `text_hit_all@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.7711), tcs-ar-26 PDF p24 (0.7640), tcs-ar-26 PDF p24 (0.7527), tcs-ar-26 PDF p36 (0.7340), tcs-ar-26 PDF p24 (0.7282), tcs-ar-26 PDF p4 (0.7230)
- **`q14`** (`fact`): By what percentage did TCS's profit attributable to shareholders, excluding exceptional items, grow in fiscal 2026?
  - Outcome: `None` | `page_hit@6`: `False` | `text_hit_any@6`: `True` | `text_hit_all@6`: `True`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.7682), tcs-ar-26 PDF p24 (0.7533), tcs-ar-26 PDF p24 (0.7486), tcs-ar-26 PDF p24 (0.7336), tcs-ar-26 PDF p36 (0.7185), tcs-ar-26 PDF p4 (0.7171)
- **`q26`** (`fact`): What was TCS's voluntary LTM attrition rate in IT services for fiscal 2026?
  - Outcome: `None` | `page_hit@6`: `False` | `text_hit_any@6`: `False` | `text_hit_all@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.6954), tcs-ar-26 PDF p4 (0.6822), tcs-ar-26 PDF p33 (0.6802), tcs-ar-26 PDF p24 (0.6745), tcs-ar-26 PDF p320 (0.6732), tcs-ar-26 PDF p46 (0.6687)
- **`q29`** (`fact`): What percentage of Infosys's workforce are women?
  - Outcome: `None` | `page_hit@6`: `False` | `text_hit_any@6`: `True` | `text_hit_all@6`: `True`
  - Top 6 retrieved: infosys-ar-26 PDF p144 (0.7416), infosys-ar-26 PDF p84 (0.6992), infosys-ar-26 PDF p168 (0.6818), infosys-ar-26 PDF p27 (0.6711), infosys-ar-26 PDF p27 (0.6614), infosys-ar-26 PDF p10 (0.6602)
