# Evaluation Report: `baseline-v0-retrieval` (20261010T140316Z)

- **Mode**: `retrieval-only`
- **Status**: `completed`
- **Questions evaluated**: `12`

> [!WARNING]
> **PRELIMINARY RESULTS**: fewer than 30 questions (12), one or more questions have `human_verified=false`. Treat these metrics as directional engineering signals rather than statistically final benchmarks.

## 1. Retrieval Metrics (Answerable Questions)

| Metric | Value |
| :--- | :--- |
| `page_hit@1` | 0.4444 |
| `page_hit@3` | 0.6667 |
| `page_hit@6` | 1.0000 |
| `MRR` | 0.6519 |
| `text_hit@6` | 0.7778 |

## 2. Answer Metrics

_Skipped in `--retrieval-only` mode._

## 3. Performance & Token Usage

| Metric | Value |
| :--- | :--- |
| `retrieve_ms` p50 / p95 | 36.70 ms / 45.57 ms |

## 4. Slices by Question Type

| Type | Count | `page_hit@6` | `text_hit@6` | `CORRECT` / `CORRECT_REFUSAL` |
| :--- | :--- | :--- | :--- | :--- |
| `fact` | 8 | 1.0000 | 0.7500 | N/A |
| `comparison` | 1 | 1.0000 | 1.0000 | N/A |
| `unanswerable` | 2 | N/A | N/A | N/A |
| `adversarial` | 1 | N/A | N/A | N/A |

## 5. Slices by Expected-Page Text Quality (Answerable Questions)

| Expected-Page Quality | Count | `page_hit@6` | `MRR` | `text_hit@6` | `CORRECT` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `all_clean` | 4 | 1.0000 | 0.8750 | 1.0000 | N/A |
| `all_degraded` | 4 | 1.0000 | 0.4667 | 0.5000 | N/A |
| `mixed` | 1 | 1.0000 | 0.5000 | 1.0000 | N/A |

## 6. Failed Questions

- **`q03`** (`fact`): What dividend did TCS declare for fiscal 2026?
  - Outcome: `None` | `page_hit@6`: `True` | `text_hit@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p33 (0.7891), tcs-ar-26 PDF p36 (0.7784), tcs-ar-26 PDF p46 (0.7487), tcs-ar-26 PDF p24 (0.7450), tcs-ar-26 PDF p4 (0.7343), tcs-ar-26 PDF p322 (0.7284)
- **`q05`** (`fact`): What is TCS's net profit margin?
  - Outcome: `None` | `page_hit@6`: `True` | `text_hit@6`: `False`
  - Top 6 retrieved: tcs-ar-26 PDF p24 (0.7714), tcs-ar-26 PDF p24 (0.7371), tcs-ar-26 PDF p24 (0.7320), tcs-ar-26 PDF p24 (0.7290), tcs-ar-26 PDF p58 (0.7089), tcs-ar-26 PDF p36 (0.7063)
