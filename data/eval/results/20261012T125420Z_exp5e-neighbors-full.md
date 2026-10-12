# Evaluation Report: `exp5e-neighbors-full` (20261012T125420Z)

- **Mode**: `full (retrieval + generation)`
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

| Metric / Outcome | Value |
| :--- | :--- |
| `CORRECT` | 21 |
| `WRONG` | 1 |
| `FALSE_REFUSAL` | 3 |
| `CORRECT_REFUSAL` | 4 |
| `ANSWERED_ANYWAY` | 0 |
| `grounded_rate` (answerable) | 0.8400 |
| `questions_with_unsupported_numbers` | 1 (['q05']) |

## 2b. Faithfulness & Citation Precision (LLM Judge)

| Metric | Value |
| :--- | :--- |
| `judged_count` (`grounded=True`) | 21 |
| `faithfulness_rate` | 1.0000 |
| `citation_precision` | 1.0000 (38/38) |
| `judge_error` count | 0 |

**Non-supporting citation numbers by question (`grounded=True`)**:
- `q01`: `[]`
- `q02`: `[]`
- `q03`: `[]`
- `q05`: `[]`
- `q06`: `[]`
- `q07`: `[]`
- `q09`: `[]`
- `q13`: `[]`
- `q14`: `[]`
- `q15`: `[]`
- `q17`: `[]`
- `q19`: `[]`
- `q21`: `[]`
- `q23`: `[]`
- `q25`: `[]`
- `q29`: `[]`
- `q31`: `[]`
- `q33`: `[]`
- `q34`: `[]`
- `q35`: `[]`
- `q37`: `[]`

## 3. Performance & Token Usage

| Metric | Value |
| :--- | :--- |
| `retrieve_ms` (n=29) p50 / p95 / max | 56.22 ms (n=29) / 68.57 ms (n=29) / 76.19 ms (n=29) |
| `mean_context_chunks` | 9.41 |
| `generate_ms` (n=29) p50 / p95 / max | 1343.62 ms (n=29) / 3134.94 ms (n=29) / 59816.67 ms (n=29) |
| `total_ms` (n=29) p50 / p95 / max | 1407.77 ms (n=29) / 3193.11 ms (n=29) / 59866.83 ms (n=29) |
| `avg_input_tokens` | 2840.1 |
| `avg_output_tokens` | 39.5 |
| `avg_cost_per_query_usd` | $0.000951 |

> [!NOTE]
> `p95` latency is not statistically meaningful below 20 samples (`n < 20`) and is heavily influenced by the single `max` sample.

## 4. Slices by Question Type

| Type | Count | `page_hit@6` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` / `CORRECT_REFUSAL` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fact` | 22 | 0.8182 | 0.8636 | 0.8636 | 18/22 |
| `comparison` | 2 | 1.0000 | 1.0000 | 1.0000 | 2/2 |
| `table` | 1 | 1.0000 | 1.0000 | 1.0000 | 1/1 |
| `unanswerable` | 4 | N/A | N/A | N/A | 4/4 |

## 5. Slices by Expected-Page Text Quality (Answerable Questions)

| Expected-Page Quality | Count | `page_hit@6` | `MRR` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_clean` | 14 | 0.8571 | 0.6333 | 0.9286 | 0.9286 | 13/14 |
| `all_degraded` | 8 | 0.7500 | 0.5833 | 0.7500 | 0.7500 | 5/8 |
| `mixed` | 3 | 1.0000 | 0.8333 | 1.0000 | 1.0000 | 3/3 |

## 6. Failed Questions

- **`q14`** (`fact`): By what percentage did TCS's profit attributable to shareholders, excluding exceptional items, grow in fiscal 2026?
  - Outcome: `WRONG` | `page_hit@6`: `False` | `text_hit_any@6`: `True` | `text_hit_all@6`: `True`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.7682), tcs-ar-26 PDF p24 (0.7533), tcs-ar-26 PDF p24 (0.7486), tcs-ar-26 PDF p24 (0.7336), tcs-ar-26 PDF p36 (0.7185), tcs-ar-26 PDF p4 (0.7171)
- **`q22`** (`fact`): What was TCS's operating income in fiscal 2026?
  - Outcome: `FALSE_REFUSAL` | `page_hit@6`: `True` | `text_hit_any@6`: `True` | `text_hit_all@6`: `True`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.7708), tcs-ar-26 PDF p4 (0.7706), tcs-ar-26 PDF p24 (0.7654), tcs-ar-26 PDF p46 (0.7649), tcs-ar-26 PDF p24 (0.7514), tcs-ar-26 PDF p24 (0.7413)
- **`q26`** (`fact`): What was TCS's voluntary LTM attrition rate in IT services for fiscal 2026?
  - Outcome: `FALSE_REFUSAL` | `page_hit@6`: `False` | `text_hit_any@6`: `False` | `text_hit_all@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.6954), tcs-ar-26 PDF p4 (0.6822), tcs-ar-26 PDF p33 (0.6802), tcs-ar-26 PDF p24 (0.6745), tcs-ar-26 PDF p320 (0.6732), tcs-ar-26 PDF p46 (0.6687)
- **`q30`** (`fact`): What was TCS's earnings per share for fiscal 2026, excluding exceptional items?
  - Outcome: `FALSE_REFUSAL` | `page_hit@6`: `True` | `text_hit_any@6`: `True` | `text_hit_all@6`: `True`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.7911), tcs-ar-26 PDF p33 (0.7655), tcs-ar-26 PDF p4 (0.7631), tcs-ar-26 PDF p36 (0.7609), tcs-ar-26 PDF p46 (0.7604), tcs-ar-26 PDF p24 (0.7383)
