# Evaluation Report: `baseline-v1-repeats` (20261011T011755Z)

- **Mode**: `full (retrieval + generation)`
- **Split**: `dev`
- **Repeats**: `3`
- **Status**: `completed`
- **Questions evaluated**: `6`

> [!WARNING]
> **PRELIMINARY RESULTS**: fewer than 30 questions (6), one or more questions have `human_verified=false`. Treat these metrics as directional engineering signals rather than statistically final benchmarks.

## 1. Retrieval Metrics (Answerable Questions)

| Metric | Value |
| :--- | :--- |
| `page_hit@1` | 0.5000 |
| `page_hit@3` | 0.6667 |
| `page_hit@6` | 1.0000 |
| `MRR` | 0.6444 |
| `text_hit_any@1` | 0.5000 |
| `text_hit_any@3` | 0.5000 |
| `text_hit_any@6` | 0.6667 |
| `text_hit_all@1` | 0.5000 |
| `text_hit_all@3` | 0.5000 |
| `text_hit_all@6` | 0.6667 |
| `distinct_pages@6` | 4.83 |

## 2. Answer Metrics

| Metric / Outcome | Value |
| :--- | :--- |
| `CORRECT` | 3 |
| `WRONG` | 1 |
| `FALSE_REFUSAL` | 2 |
| `CORRECT_REFUSAL` | 0 |
| `ANSWERED_ANYWAY` | 0 |
| `grounded_rate` (answerable) | 0.6667 |
| `questions_with_unsupported_numbers` | 0 ([]) |

## 2c. Repeats & Stability

| Metric | Value |
| :--- | :--- |
| `repeats` | 3 |
| `accuracy` (mean, min–max) | 0.5000 (0.5000–0.5000) |
| `answerable_accuracy` (mean, min–max) | 0.5000 (0.5000–0.5000) |
| `flip_rate` | 0.0000 (flipped IDs: []) |

| Question ID | Outcomes across repeats | Flipped |
| :--- | :--- | :--- |
| `q01` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q02` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q03` | `WRONG`, `WRONG`, `WRONG` | `False` |
| `q04` | `FALSE_REFUSAL`, `FALSE_REFUSAL`, `FALSE_REFUSAL` | `False` |
| `q05` | `FALSE_REFUSAL`, `FALSE_REFUSAL`, `FALSE_REFUSAL` | `False` |
| `q06` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |

## 3. Performance & Token Usage

| Metric | Value |
| :--- | :--- |
| `retrieve_ms` (n=18) p50 / p95 / max | 32.23 ms (n=18) / 44.74 ms (n=18) / 48.61 ms (n=18) |
| `generate_ms` (n=18) p50 / p95 / max | 1203.94 ms (n=18) / 21973.48 ms (n=18) / 46245.92 ms (n=18) |
| `total_ms` (n=18) p50 / p95 / max | 1240.58 ms (n=18) / 22015.98 ms (n=18) / 46278.35 ms (n=18) |
| `avg_input_tokens` | 1872.8 |
| `avg_output_tokens` | 35.6 |

> [!NOTE]
> `p95` latency is not statistically meaningful below 20 samples (`n < 20`) and is heavily influenced by the single `max` sample.

## 4. Slices by Question Type

| Type | Count | `page_hit@6` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` / `CORRECT_REFUSAL` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fact` | 6 | 1.0000 | 0.6667 | 0.6667 | 3/6 |

## 5. Slices by Expected-Page Text Quality (Answerable Questions)

| Expected-Page Quality | Count | `page_hit@6` | `MRR` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_clean` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2/2 |
| `all_degraded` | 4 | 1.0000 | 0.4667 | 0.5000 | 0.5000 | 1/4 |
| `mixed` | 0 | N/A | N/A | N/A | N/A | N/A |

## 6. Failed Questions

- **`q03`** (`fact`): What dividend did TCS declare for fiscal 2026?
  - Outcome: `WRONG` | `page_hit@6`: `True` | `text_hit_any@6`: `False` | `text_hit_all@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p33 (0.7891), tcs-ar-26 PDF p36 (0.7784), tcs-ar-26 PDF p46 (0.7487), tcs-ar-26 PDF p24 (0.7450), tcs-ar-26 PDF p4 (0.7343), tcs-ar-26 PDF p322 (0.7284)
- **`q04`** (`fact`): What was TCS's net income in fiscal 2026?
  - Outcome: `FALSE_REFUSAL` | `page_hit@6`: `True` | `text_hit_any@6`: `True` | `text_hit_all@6`: `True`
  - Top 6 retrieved: tcs-ar-26 PDF p46 (0.7631), tcs-ar-26 PDF p24 (0.7582), tcs-ar-26 PDF p4 (0.7556), tcs-ar-26 PDF p24 (0.7523), tcs-ar-26 PDF p24 (0.7313), tcs-ar-26 PDF p36 (0.7284)
- **`q05`** (`fact`): What is TCS's net profit margin?
  - Outcome: `FALSE_REFUSAL` | `page_hit@6`: `True` | `text_hit_any@6`: `False` | `text_hit_all@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.7714), tcs-ar-26 PDF p24 (0.7371), tcs-ar-26 PDF p24 (0.7320), tcs-ar-26 PDF p24 (0.7290), tcs-ar-26 PDF p58 (0.7089), tcs-ar-26 PDF p36 (0.7063)
