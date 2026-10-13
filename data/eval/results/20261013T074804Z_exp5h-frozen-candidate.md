# Evaluation Report: `exp5h-frozen-candidate` (20261013T074804Z)

- **Mode**: `full (retrieval + generation)`
- **Split**: `dev`
- **Repeats**: `3`
- **Status**: `completed`
- **Questions evaluated**: `30`
- **Skipped draft questions**: `0` (excluded by default because status='draft_needs_fill' (missing expected answers and/or PDF pages))

## 1. Retrieval Metrics (Answerable Questions)

| Metric | Value |
| :--- | :--- |
| `page_hit@1` | 0.4615 |
| `page_hit@3` | 0.7692 |
| `page_hit@6` | 0.9615 |
| `MRR` | 0.6596 |
| `text_hit_any@1` | 0.6923 |
| `text_hit_any@3` | 0.8077 |
| `text_hit_any@6` | 0.9615 |
| `text_hit_all@1` | 0.6538 |
| `text_hit_all@3` | 0.8077 |
| `text_hit_all@6` | 0.9615 |
| `text_hit_any@context` | 1.0000 |
| `text_hit_all@context` | 1.0000 |
| `distinct_pages@6` | 5.00 |
| `mean_context_chunks` | 10.00 |

## 2. Answer Metrics

| Metric / Outcome | Value |
| :--- | :--- |
| `CORRECT` | 26 |
| `WRONG` | 0 |
| `FALSE_REFUSAL` | 0 |
| `CORRECT_REFUSAL` | 4 |
| `ANSWERED_ANYWAY` | 0 |
| `grounded_rate` (answerable) | 1.0000 |
| `questions_with_unsupported_numbers` | 0 ([]) |

## 2b. Faithfulness & Citation Precision (LLM Judge)

| Metric | Value |
| :--- | :--- |
| `judged_count` (`grounded=True`) | 26 |
| `faithfulness_rate` | 0.9615 |
| `citation_precision` | 0.9592 (47/49) |
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
- `q14`: `[1, 3]`
- `q15`: `[]`
- `q17`: `[]`
- `q18`: `[]`
- `q19`: `[]`
- `q21`: `[]`
- `q22`: `[]`
- `q23`: `[]`
- `q25`: `[]`
- `q26`: `[]`
- `q27`: `[]`
- `q29`: `[]`
- `q30`: `[]`
- `q31`: `[]`
- `q33`: `[]`
- `q34`: `[]`
- `q35`: `[]`
- `q37`: `[]`

## 2c. Repeats & Stability

| Metric | Value |
| :--- | :--- |
| `repeats` | 3 |
| `accuracy` (mean, min–max) | 0.9778 (0.9667–1.0000) |
| `answerable_accuracy` (mean, min–max) | 0.9743 (0.9615–1.0000) |
| `flip_rate` | 0.0333 (flipped IDs: ['q35']) |

| Question ID | Outcomes across repeats | Flipped |
| :--- | :--- | :--- |
| `q01` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q02` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q03` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q05` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q06` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q07` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q09` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q10` | `CORRECT_REFUSAL`, `CORRECT_REFUSAL`, `CORRECT_REFUSAL` | `False` |
| `q11` | `CORRECT_REFUSAL`, `CORRECT_REFUSAL`, `CORRECT_REFUSAL` | `False` |
| `q13` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q14` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q15` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q17` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q18` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q19` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q21` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q22` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q23` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q25` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q26` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q27` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q29` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q30` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q31` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q33` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q34` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q35` | `CORRECT`, `FALSE_REFUSAL`, `FALSE_REFUSAL` | `True` |
| `q37` | `CORRECT`, `CORRECT`, `CORRECT` | `False` |
| `q38` | `CORRECT_REFUSAL`, `CORRECT_REFUSAL`, `CORRECT_REFUSAL` | `False` |
| `q39` | `CORRECT_REFUSAL`, `CORRECT_REFUSAL`, `CORRECT_REFUSAL` | `False` |

## 3. Performance & Token Usage

| Metric | Value |
| :--- | :--- |
| `retrieve_ms` (n=90) p50 / p95 / max | 61.23 ms (n=90) / 411.93 ms (n=90) / 618.53 ms (n=90) |
| `mean_context_chunks` | 10.00 |
| `generate_ms` (n=90) p50 / p95 / max | 1109.72 ms (n=90) / 1984.43 ms (n=90) / 40722.28 ms (n=90) |
| `total_ms` (n=90) p50 / p95 / max | 1213.10 ms (n=90) / 2044.68 ms (n=90) / 40784.97 ms (n=90) |
| `avg_input_tokens` | 3135.4 |
| `avg_output_tokens` | 38.0 |
| `avg_cost_per_query_usd` | $0.001036 |

> [!NOTE]
> `p95` latency is not statistically meaningful below 20 samples (`n < 20`) and is heavily influenced by the single `max` sample.

## 4. Slices by Question Type

| Type | Count | `page_hit@6` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` / `CORRECT_REFUSAL` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fact` | 23 | 0.9565 | 0.9565 | 0.9565 | 23/23 |
| `comparison` | 2 | 1.0000 | 1.0000 | 1.0000 | 2/2 |
| `table` | 1 | 1.0000 | 1.0000 | 1.0000 | 1/1 |
| `unanswerable` | 4 | N/A | N/A | N/A | 4/4 |

## 5. Slices by Expected-Page Text Quality (Answerable Questions)

| Expected-Page Quality | Count | `page_hit@6` | `MRR` | `text_hit_any@6` | `text_hit_all@6` | `CORRECT` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_clean` | 15 | 0.9333 | 0.6689 | 0.9333 | 0.9333 | 15/15 |
| `all_degraded` | 8 | 1.0000 | 0.5771 | 1.0000 | 1.0000 | 8/8 |
| `mixed` | 3 | 1.0000 | 0.8333 | 1.0000 | 1.0000 | 3/3 |

## 6. Failed Questions

_None. All evaluated questions passed._
