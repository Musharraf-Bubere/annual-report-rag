# Evaluation Harness (Phase 4a — Deterministic Metrics)

The evaluation harness measures retrieval and answer quality reproducibly across our golden question set (`data/eval/golden_questions.json`) without leaking raw PDF or chunk text into version control.

---

## 1. How to Read Each Metric

### A. Retrieval Metrics (Computed over Answerable Questions)

- **`page_hit@k` (`@1`, `@3`, `@6`)**: Fraction of answerable questions where the top-$k$ retrieved chunks contain at least one expected PDF page **from each source document that has expected pages**.
  - For single-company `fact` questions, at least one expected `(source, page)` must appear in the top $k$.
  - For multi-company `comparison` questions (where `retrieve()` fetches `top_k=6` per company and merges by score), `page_hit@k` requires at least one expected page from **every** expected company within the top $k$ merged chunks.
- **`MRR` (Mean Reciprocal Rank)**: Average of $\frac{1}{\text{rank}}$ of the **first** expected `(source, page)` across all retrieved chunks (or $0.0$ if no expected page was retrieved). An `MRR` of `1.0` means an expected page always ranked #1; `0.5` means it ranked #2 on average.
- **`text_hit@6`**: Fraction of answerable questions where at least one chunk in the top 6 contains a normalized numeric token or substring satisfying a `must_contain` entry.
  - **Why both `page_hit@6` and `text_hit@6` matter**: A single PDF page is split into multiple overlapping chunks (`chunk_index = 0, 1, 2, ...`). It is possible for `page_hit@6` to be `True` (a chunk from the right PDF page was retrieved) while `text_hit@6` is `False` (the specific chunk on that page holding the target figure ranked outside the top 6, or the token was dropped on a degraded page). Comparing `page_hit@6` against `text_hit@6` immediately separates **page-level routing/ranking issues** from **intra-page chunk ranking or extraction issues**.

### B. Answer Outcomes & Grounding Metrics

Every evaluated question is assigned exactly one of five deterministic outcomes by `evaluation/metrics.py`:

| Question Category | Outcome | Meaning |
| :--- | :--- | :--- |
| **Answerable** (`fact`, `comparison`) | `CORRECT` | The model did not refuse, triggered no `must_not_contain` entry, and its answer (after stripping `[n]` citation markers) satisfies **all** `must_contain` entries. |
| **Answerable** (`fact`, `comparison`) | `WRONG` | The model produced an answer (did not refuse), but missed at least one `must_contain` entry or matched a `must_not_contain` entry. |
| **Answerable** (`fact`, `comparison`) | `FALSE_REFUSAL` | The question is answerable in the corpus, but the system returned `NOT_FOUND_MESSAGE` (`"I could not find this in the provided reports."`)—typically because the target chunk missed the top-$k$ retrieval window. |
| **Unanswerable / Adversarial** | `CORRECT_REFUSAL` | The system properly refused with `NOT_FOUND_MESSAGE` and did not match any `must_not_contain` token. |
| **Unanswerable / Adversarial** | `ANSWERED_ANYWAY` | The system failed to refuse an out-of-corpus question, or included a forbidden `must_not_contain` string (such as `"HACKED"` on an adversarial prompt-injection probe). |

Additional answer-level diagnostics:
- **`grounded_rate`**: Fraction of answerable questions where `AnswerResult.grounded` is `True` (at least one valid in-range `[n]` citation and answer $\neq$ `NOT_FOUND_MESSAGE`).
- **`questions_with_unsupported_numbers`**: Count (and IDs) of questions where `find_unsupported_numbers()` detected a number in the generated answer that does not appear in the cited chunks.

### C. Performance, Token Usage, and Cost

- **Warmup Query**: Before timing begins, `run_eval()` executes one throwaway retrieval query (`top_k=1`) so one-time ONNX model loading is excluded from latency measurements.
- **`retrieve_ms`, `generate_ms`, `total_ms` (`p50` and `p95`)**: Median (`p50`) and 95th-percentile (`p95`) latencies in milliseconds across evaluated questions.
- **`avg_input_tokens` and `avg_output_tokens`**: Mean prompt and completion tokens reported by `GenerationResult`.
- **`avg_cost_per_query_usd` (Optional)**: Computed **only** when both `evaluation.price_per_million_input_usd` and `evaluation.price_per_million_output_usd` are set to non-null numbers in `config/settings.yaml`. Both default to `null` so the harness never guesses API pricing.

### D. Diagnostic Slices

1. **By Question Type (`fact`, `comparison`, `unanswerable`, `adversarial`)**: Isolates single-fact extraction, multi-company synthesis, out-of-scope refusal, and prompt-injection resistance.
2. **By Expected-Page Text Quality (`all_clean`, `all_degraded`, `mixed`)**: Uses `ChromaStore.page_text_quality(source, page)` to group answerable questions by whether their ground-truth PDF pages were `clean` or `degraded` during ingestion. This directly quantifies how much PDF font corruption impacts retrieval and answer accuracy.

---

## 2. Golden Set Rules & Matching Semantics

The golden set lives in `data/eval/golden_questions.json` and is validated by Pydantic (`evaluation/golden.py`):

1. **Required Schema Fields**:
   - `id` (`str`, unique across the file, e.g. `"q01"`)
   - `question` (`str`, non-empty)
   - `type` (`"fact" | "comparison" | "unanswerable" | "adversarial"`)
   - `answerable` (`bool`: must be `true` for `fact`/`comparison` and `false` for `unanswerable`/`adversarial`)
   - `expected_pages` (`list[{source, page}]` using **1-indexed PDF page numbers**; must be non-empty when `answerable=true` and empty when `answerable=false`)
   - `must_contain` (`list[str]`; must be non-empty when `answerable=true`)
   - `must_not_contain` (`list[str]`)
   - `notes` (`str`)
   - `human_verified` (`bool`; `false` for unverified seed items)
2. **Citation Stripping**: Before matching `must_contain` or `must_not_contain`, inline bracket markers (`[1]`, `[1][3]`, `[1, 3]`) are stripped so `[1]` never collides with numeric matching.
3. **Exact Numeric Token Equality**:
   - `numeric_tokens(text)` extracts numbers (with optional commas and decimals) and strips commas, so `"5,84,519"`, `"584,519"`, and `"₹2,67,021"` normalize to `"584519"` and `"267021"`, and `"4.6%"` normalizes to `"4.6"`.
   - When a `must_contain` entry is a number, matching uses **exact token equality**, never substring matching. For example, `"11,000%"` (token `"11000"`) does **not** satisfy `"110"`.
4. **Non-Numeric Substrings & Any-Of (`"a|b"`) Syntax**:
   - Non-numeric entries match via case-insensitive substring.
   - An entry written `"a|b"` succeeds if **any** alternative matches; **all** entries in `must_contain` must be satisfied.
5. **Store Coverage Warning**: `check_expected_pages_in_store()` checks every expected `(source, page)` against `ChromaStore` and emits a non-fatal warning if a page has zero indexed chunks.

---

## 3. How to Add Questions and Run Evaluations

1. **Verify the PDF page number** (not the printed report footer number) in your PDF viewer or via `uv run python scripts/search_demo.py "<keywords>"`.
2. **Append the new entry** to `data/eval/golden_questions.json`. Once a human has manually inspected the PDF page and confirmed the exact wording and figures, set `"human_verified": true`.
3. **Validate the golden set against the vector store (zero API calls)**:
   ```powershell
   uv run python scripts/run_eval.py --validate-only
   ```
4. **Run retrieval-only evaluation (offline, zero API calls)**:
   ```powershell
   uv run python scripts/run_eval.py --retrieval-only --label my-retrieval-check
   ```
5. **Run full evaluation (uses local cache in `data/eval/cache/` unless inputs changed)**:
   ```powershell
   uv run python scripts/run_eval.py --label my-eval-run
   ```
6. **Compare two runs**:
   ```powershell
   uv run python scripts/compare_runs.py data/eval/results/<runA>.metrics.json data/eval/results/<runB>.metrics.json
   ```

> [!IMPORTANT]
> **Data Privacy & Git Policy**:
> - `data/eval/runs/<timestamp>_<label>.json` contains full retrieved chunk text and answers for local debugging and is **gitignored**.
> - `data/eval/results/<timestamp>_<label>.md` and `.metrics.json` contain **only** aggregate metrics and per-question metadata outcomes (zero PDF or chunk text) and are safe to commit.

---

## 4. Known Limitations

1. **Small Sample Size ($n = 12$ seeds)**: With 12 seed questions (9 answerable, 3 unanswerable/adversarial), a single question flips answerable metrics by $\pm 11.1\%$. Reports automatically display a `PRELIMINARY RESULTS` warning whenever $n < 30$ or any question has `human_verified=false`.
2. **Weak Lexical Labels (`must_contain`)**: Deterministic `must_contain` checks verify that required numbers or keywords appear in the answer, but cannot verify semantic framing (e.g., whether a number is attributed to the right fiscal year or whether surrounding prose is coherent). An LLM-as-a-judge faithfulness and relevance layer is deferred to Phase 4b.
3. **Non-Zero Generation Temperature (`1.0` Model Default)**: Following Google's official recommendation for Gemini 3.x models (`generation.temperature: null` $\to$ default `1.0`), phrasing and borderline retrieval interpretations can vary slightly across uncached runs (`--no-cache`). Our answer cache (`data/eval/cache/`) keeps repeated runs deterministic when settings, prompt, and store fingerprint are unchanged.
4. **Single Two-Report Corpus**: All current questions evaluate two Indian IT services annual reports (`tcs-ar-26` and `infosys-ar-26`) for fiscal 2025–26. Performance on other sectors (banking, manufacturing, filings with scanned image tables) is not yet represented.
