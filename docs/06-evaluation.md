# Evaluation Harness (Phase 4a, 4b & 4c — Deterministic Metrics, Repeats, LLM Judge, and 40-Question Lifecycle)

The evaluation harness measures retrieval, answer quality, repeat stability, and citation faithfulness reproducibly across our 40-question golden set (`data/eval/golden_questions.json`) without leaking raw PDF, chunk, or unsupported-claim text into version control.

---

## 1. Dev / Test Split Rule & Draft / Verified Lifecycle

### A. Dev / Test Split Rule (`split: "dev" | "test"`)
Every question in `data/eval/golden_questions.json` carries a `split` field (`"dev"` or `"test"`, defaulting to `"dev"`), and `scripts/run_eval.py` accepts `--split dev|test|all` (default `"dev"`). Across the 40-question set, `30` questions are in `dev` (`29` ready, `1` `draft_needs_fill`) and `10` are in `test` (`10` ready, `0` `draft_needs_fill`):

- **Tune on `dev`, run `test` only for final numbers**: All prompt engineering, chunking adjustments, score thresholds, and retrieval experiments must be iterated and evaluated on `--split dev`. The `--split test` partition is a held-out benchmark run only at the end of a milestone to verify that improvements generalize rather than overfitting to the `dev` questions.

### B. Question Lifecycle (`status` and `human_verified`)
Each question progresses through three lifecycle stages via `status` (`"draft_needs_fill" | "ready_unverified" | "verified"`) and `human_verified` (`bool`):
1. **`draft_needs_fill` (`human_verified = false`)**: Question text and target sources are defined, but `expected_pages` and/or `must_contain` answers are still empty (`1` question in the corrected set: `q27`).
   - **Excluded by default**: Scored runs (`scripts/run_eval.py`) automatically exclude `status == "draft_needs_fill"` questions unless `--include-drafts` is explicitly passed, and the report states how many draft questions were skipped and why.
   - **Candidate inspection & PDF page lookup (`scripts/show_candidates.py`, `scripts/find_pages.py`)**: Use the read-only helpers `uv run python scripts/show_candidates.py` (top-6 retrieved chunks) and `uv run python scripts/find_pages.py` (direct PyMuPDF token search over `data/raw/*.pdf`), which write to gitignored `data/eval/runs/candidates_<timestamp>.md` and `data/eval/runs/page_lookup_<timestamp>.md`. **Candidates are pointers only—expected pages and answers must be confirmed in the original PDF, never copied blindly from chunk text.**
2. **`ready_unverified` (`human_verified = false`)**: `expected_pages` and `must_contain` are populated and included in scored engineering runs, pending full manual PDF audit.
3. **`verified` (`human_verified = true`)**: Expected PDF pages, figures, units, and acceptable spellings have been manually verified against the official annual report PDF (`12` questions verified so far).

> [!IMPORTANT]
> **Headline Numbers Rule**: Unverified questions (`human_verified = false`) must **not** be used for headline benchmark numbers. Reports automatically emit a `PRELIMINARY RESULTS` warning whenever fewer than 30 questions are evaluated or any evaluated question has `human_verified=false`.

---

## 2. How to Read Each Metric

### A. Retrieval Metrics (Computed over Answerable Questions)

- **Multi-Source Comparison $k$ Scaling**: When a query triggers multi-source retrieval (`len(detected_sources) > 1`, such as a two-company comparison), `retrieve()` fetches `top_k` (6) chunks per company and passes all `top_k * num_detected_sources` (12) chunks to the generator. Accordingly, `page_hit@k`, `text_hit_any@k`, and `text_hit_all@k` scale the cutoff to `k * num_detected_sources` (`effective_k`) on multi-source questions so retrieval is graded on the exact context window the LLM receives.
- **`page_hit@k` (`@1`, `@3`, `@6`)**: Fraction of answerable questions where the top-$k$ window contains at least one expected PDF page **from each source document that has expected pages**.
- **`MRR` (Mean Reciprocal Rank)**: Average of $\frac{1}{\text{rank}}$ of the **first** expected `(source, page)` across all retrieved chunks.
- **`text_hit_any@k` (`@1`, `@3`, `@6`)**: Fraction of answerable questions where **at least one** chunk in the top-$k$ window satisfies **any** `must_contain` entry.
- **`text_hit_all@k` (`@1`, `@3`, `@6`)**: Fraction of answerable questions where **every** `must_contain` entry is covered by the **union** of the top-$k$ chunks. Comparing `text_hit_any@k` with `text_hit_all@k` distinguishes partial context retrieval (e.g. retrieving only one company's growth rate in a two-fact query) from complete context coverage.
- **`distinct_pages@6` (Crowding Indicator)**: Average number of distinct `(source, page)` pairs among the top 6 retrieved chunks. A value near `6.00` means all 6 chunks come from different pages; a low value reveals **page crowding**, where multiple chunks from a single long page crowd out other relevant pages.

### B. Answer Outcomes & Grounding Metrics

Every evaluated question is assigned one of five deterministic outcomes (`evaluation/metrics.py`):

| Question Category | Outcome | Meaning |
| :--- | :--- | :--- |
| **Answerable** (`fact`, `comparison`, `table`) | `CORRECT` | Did not refuse, triggered no `must_not_contain` entry, and its answer (after stripping `[n]` citation markers) satisfies **all** `must_contain` entries. |
| **Answerable** (`fact`, `comparison`, `table`) | `WRONG` | Produced an answer (did not refuse), but missed at least one `must_contain` entry or matched a `must_not_contain` entry. |
| **Answerable** (`fact`, `comparison`, `table`) | `FALSE_REFUSAL` | Answerable in the corpus, but the system returned `NOT_FOUND_MESSAGE` (`"I could not find this in the provided reports."`). |
| **Unanswerable / Adversarial** | `CORRECT_REFUSAL` | Properly refused with `NOT_FOUND_MESSAGE` and did not match any `must_not_contain` token. |
| **Unanswerable / Adversarial** | `ANSWERED_ANYWAY` | Failed to refuse an out-of-corpus question, or included a forbidden `must_not_contain` string (such as `"HACKED"`). |

- **`grounded_rate`**: Fraction of answerable questions where `AnswerResult.grounded` is `True` (at least one valid in-range `[n]` citation and answer $\neq$ `NOT_FOUND_MESSAGE`).
- **`questions_with_unsupported_numbers`**: Count (and IDs) of questions where `find_unsupported_numbers()` detected a number in the answer not present in its cited chunks.

### C. Faithfulness & Citation Precision (`--judge`, `evaluation/judge.py`)

When `--judge` is passed, `judge_answer()` runs **only on answers with `grounded=True`**, sending the question, answer, and **full text of every cited chunk** to `GenerationProvider`:

- **`faithfulness_rate`**: Fraction of valid judged answers (`parse_ok=True`) where `verdict.faithful` is `True` (every claim in the answer is supported by its cited passages; Indian vs. Western comma differences like `6,17,437` vs `617,437` are allowed, while different numbers, labels, or units are marked unfaithful).
  - **Faithfulness vs. Correctness**: An answer can be **100% faithful** to its cited passage while still being **`WRONG`** on `outcome` (for example, `q03` faithfully reports `11,000%` of paid-up value from chunk `[3]` on PDF page 36, so `faithful=True`, even though the question's `must_contain` requires the `₹110` per-share dividend).
- **`citation_precision`**: Ratio of supporting citations (`supports=True`) to all citations across valid judged answers ($\frac{\text{supporting citations}}{\text{all citations}}$). Measures whether the model over-cites irrelevant passages alongside supporting ones.
- **`judge_error` count**: Number of judged answers where the judge returned malformed JSON twice in a row (initial call + 1 retry).
- **Non-supporting citation numbers**: For each judged question ID, the committed report lists **only the citation numbers** (`[n]`) that did not support any claim. Raw `unsupported_claims` text and passage text are written **only** to gitignored files (`data/eval/runs/<ts>_<label>.json` and `data/eval/runs/<ts>_<label>_judge_review.md`).

### D. Repeats & Stability (`--repeats N`)

Because `generation.temperature` is left at the Gemini 3.x model default (`1.0`), answers can vary across uncached runs. Running `--repeats N` (default `1`) evaluates each question $N$ times (with `repeat_index` included in the cache key):
- **`accuracy` (mean, min–max)**: Mean pass rate across the $N$ repeats along with the `[min, max]` range across repeats.
- **`flip_rate`**: Fraction of evaluated questions whose deterministic outcome (`CORRECT`, `WRONG`, `FALSE_REFUSAL`, etc.) differed across repeats.

### E. Performance, Token Usage, and Sample Size (`n`)

- **Warmup Query**: `run_eval()` executes one throwaway retrieval query (`top_k=1`) before timing so cold ONNX model loading is excluded.
- **`p50 / p95 / max (n=...)`**: Every latency percentile displays the sample count `n` and the `max` latency.
- **Why `p95` is not meaningful below 20 samples (`n < 20`)**: With fewer than 20 samples, the 95th percentile is a linear interpolation between the second-highest and single highest (`max`) sample. If a single request encounters a 45-second per-minute rate-limit sleep, `p95` is artificially inflated by that single `max` outlier.

---

## 3. Judge Calibration (`data/eval/judge_calibration.json`)

Before trusting an LLM judge on real reports, we verify its grading logic on 6 synthetic `ExampleCo` cases (`data/eval/judge_calibration.json`, containing zero corporate report text):
1. `cal01`: Correct figure and supported passage $\to$ `faithful=True, [1]=True`.
2. `cal02`: Wrong figure vs. the passage $\to$ `faithful=False, [1]=False`.
3. `cal03`: Cited passage is irrelevant to the claim $\to$ `faithful=False, [1]=False`.
4. `cal04`: Two citations, one supports and one irrelevant $\to$ `[1]=True, [2]=False`.
5. `cal05`: Right number (`18.4%`) but wrong label (`Net margin` vs. `Operating margin`) $\to$ `faithful=False, [1]=False`.
6. `cal06`: Indian vs. Western number format only (`6,17,437` vs. `617,437`) $\to$ `faithful=True, [1]=True`.

Run calibration via:
```powershell
uv run python scripts/run_judge_calibration.py
```

---

## 4. Golden Set Rules & Workflow

1. **Schema Fields (`data/eval/golden_questions.json`)**: `id` (unique), `question`, `type` (`fact | comparison | table | unanswerable | adversarial`), `answerable` (`bool`), `expected_sources` (`list[str]`), `expected_pages` (`list[{source, page}]` using 1-indexed PDF page numbers), `must_contain`, `must_not_contain`, `notes`, `human_verified` (`bool`), `split` (`"dev" | "test"`), and `status` (`"draft_needs_fill" | "ready_unverified" | "verified"`). The loader (`load_golden_set`) also accepts neutral draft JSON files (`data/eval/drafts/golden_questions_draft_v1.json`) directly.
2. **Matching Rules**: Citation markers (`[1]`, `[1, 3]`) are stripped first; numeric `must_contain` entries match by **exact normalized token equality** with commas and trailing decimal zeros stripped (`"48.00"` $\to$ `"48"`, `"25.0"` and `"25.00"` $\to$ `"25"`, `"11,000%"` $\to$ `"11000"` $\neq$ `"110"`, while Indian `5,84,519` and Western `584,519` both normalize to `"584519"`); non-numeric entries use case-insensitive substring and `"a|b"` any-of syntax. In `find_unsupported_numbers()`, a 4-digit year (e.g. `2026`) is also accepted when a cited passage contains the fiscal-year form `"2025-26"` or `"FY26"`.
3. **CLI Commands**:
   ```powershell
   # Validate golden set (reports total, dev/test, and status/draft counts)
   uv run python scripts/run_eval.py --validate-only

   # Inspect top-6 retrieved candidate chunks for draft_needs_fill questions (read-only, gitignored output)
   uv run python scripts/show_candidates.py
   uv run python scripts/show_candidates.py --ids q13,q14

   # Run dev split with LLM judge (excludes draft_needs_fill by default)
   uv run python scripts/run_eval.py --label baseline-v1-full --judge

   # Rescore cached answers from a prior run with the current scorer (zero API or judge calls)
   uv run python scripts/run_eval.py --rescore baseline-v1-full --label baseline-v2

   # Measure repeat stability across 3 repeats on dev split
   uv run python scripts/run_eval.py --label baseline-v1-full-repeats --repeats 3

   # Compare two runs
   uv run python scripts/compare_runs.py data/eval/results/<runA>.metrics.json data/eval/results/<runB>.metrics.json
   ```

---

## 5. Known Limitations & Judge Bias

1. **Same-Family Judge Bias (Self-Preference / Shared Blind Spots)**: Both the answer generator and the faithfulness judge currently use the same `GenerationProvider` (`gemini-3.1-flash-lite-preview`). An LLM judge from the same model family as the generator is systematically biased toward agreeing with the generator's phrasing and shares the same reasoning blind spots. Always spot-check the generated `data/eval/runs/<ts>_<label>_judge_review.md` by hand.
2. **Sub-Claim Granularity in Multi-Citation Sentences**: The judge prompt grades whether each cited passage *"supports at least one claim in the answer that cites it."* When a sentence makes multiple sub-claims and attaches `[5][6]` at the end, if `[6]` supports two of the three sub-claims while `[5]` supports all three, `[6]` is marked `supports=True`.
3. **Unverified & Draft Questions (`human_verified = false`)**: In `golden_questions.json`, `20` of `40` questions still have `human_verified=false` (`20` are `human_verified=true`; `39` are `ready_unverified` and `1` is `draft_needs_fill`). Unverified questions must not be used for final headline numbers until verified against the source PDFs.
4. **Weak Lexical Labels & Single Corpus**: Deterministic `must_contain` labels only check token presence, and all current questions come from two FY2025–26 Indian IT services reports (`tcs-ar-26`, `infosys-ar-26`).

---

## 6. Phase 5a Retrieval Experiment Log (`dev` Split vs. `baseline-v2`)

All experiments were evaluated on `--split dev` ($29$ ready questions, $25$ answerable; `q27` skipped as `draft_needs_fill`) against locked `baseline-v2` (`20261011T101238Z_baseline-v2`) with zero re-ingestion, prompt, golden-set, or scorer changes. Both new config switches (`retrieval.query_prefix` and `retrieval.max_chunks_per_page`) default to `null` in `config/settings.yaml` so default runs reproduce `baseline-v2` exactly.

### A. Retrieval-Only Sweep (`--retrieval-only`, 0 API Calls)

| Experiment / Config | `page_hit@1` | `page_hit@3` | `page_hit@6` | `text_hit_any@1` | `text_hit_any@3` | `text_hit_any@6` | `text_hit_all@1` | `text_hit_all@3` | `text_hit_all@6` | `MRR` | `distinct_pages@6` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`baseline-v2` (default: `top_k=6`, `prefix=null`, `page_cap=null`)** | `0.4800` | `0.8000` | `0.8400` | `0.6000` | `0.7200` | `0.8400` | `0.5600` | `0.7200` | `0.8400` | `0.6480` | `5.00` |
| **Exp A (`query_prefix=BGE`, `top_k=6`, `page_cap=null`)** | `0.5600` (`+0.08`) | `0.8000` (`0.00`) | `0.8800` (`+0.04`) | `0.5200` (`-0.08`) | `0.7200` (`0.00`) | `0.8400` (`0.00`) | `0.4800` (`-0.08`) | `0.7200` (`0.00`) | `0.8400` (`0.00`) | `0.6880` (`+0.0400`) | `4.88` (`-0.12`) |
| **Exp B (`max_chunks_per_page=2`, `top_k=6`, `prefix=null`)** | `0.4800` (`0.00`) | `0.8000` (`0.00`) | `0.8800` (`+0.04`) | `0.6000` (`0.00`) | `0.7200` (`0.00`) | `0.8400` (`0.00`) | `0.5600` (`0.00`) | `0.7200` (`0.00`) | `0.8400` (`0.00`) | `0.6547` (`+0.0067`) | `5.36` (`+0.36`) |
| **Exp C1 (`top_k=8`, `prefix=null`, `page_cap=null`)** *(at `@6` / at `@8`)* | `0.4800` (`0.00`) | `0.8000` (`0.00`) | `0.8400` (`0.8800` at `@8`) | `0.6000` (`0.00`) | `0.7200` (`0.00`) | `0.8400` (`0.8800` at `@8`) | `0.5600` (`0.00`) | `0.7200` (`0.00`) | `0.8400` (`0.8800` at `@8`) | `0.6530` (`+0.0050`) | `5.00` (`6.76` at `@8`) |
| **Exp C2 (`top_k=10`, `prefix=null`, `page_cap=null`)** *(at `@6` / at `@10`)* | `0.4800` (`0.00`) | `0.8000` (`0.00`) | `0.8400` (`0.9200` at `@10`) | `0.6000` (`0.00`) | `0.7200` (`0.00`) | `0.8400` (`0.8800` at `@10`) | `0.5600` (`0.00`) | `0.7200` (`0.00`) | `0.8400` (`0.8800` at `@10`) | `0.6574` (`+0.0094`) | `5.00` (`8.41` at `@10`) |
| **Exp D (`query_prefix=BGE` + `max_chunks_per_page=2`, `top_k=6`)** | `0.5600` (`+0.08`) | `0.8000` (`0.00`) | `0.9200` (`+0.08`) | `0.5200` (`-0.08`) | `0.7200` (`0.00`) | `0.8400` (`0.00`) | `0.4800` (`-0.08`) | `0.7200` (`0.00`) | `0.8400` (`0.00`) | `0.6960` (`+0.0480`) | `5.24` (`+0.24`) |

### B. Full-Generation Runs (`--judge`, 29 `dev` Questions)

| Run Label | Config | `CORRECT` | `WRONG` | `FALSE_REFUSAL` | `CORRECT_REFUSAL` | `ANSWERED_ANYWAY` | `faithfulness_rate` | `citation_precision` | `avg_cost_per_query_usd` | Outcome Changes vs. `baseline-v2` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`baseline-v2`** | `top_k=6, page_cap=null` | `19` | `1` | `5` | `4` | `0` | `1.0000` (`20/20`) | `1.0000` (`34/34`) | `$0.000691` | — |
| **`exp5a-pagecap2`** | `top_k=6, page_cap=2` | `19` | `1` | `5` | `4` | `0` | `1.0000` (`20/20`) | `1.0000` (`37/37`) | `$0.000687` | None (`0` regressions on `CORRECT` questions) |
| **`exp5a-topk8`** | `top_k=8, page_cap=null` | `20` (`+1`) | `1` | `4` (`-1`) | `4` | `0` | `1.0000` (`19/19`) | `1.0000` (`32/32`) | `$0.000888` | `q22`: `FALSE_REFUSAL -> CORRECT` (`0` regressions on `CORRECT` questions) |

### C. Key Findings
1. **`fastembed` Query Prefix (`Exp A`)**: `fastembed`'s `OnnxTextEmbedding` does not prepend BAAI's query instruction prefix for `BAAI/bge-base-en-v1.5`. Adding it manually boosts `MRR` (`+0.0400`) and `page_hit@6` (`+0.0400`), but hurts rank-1 text hit (`-0.0800`) and does not recover any of the 4 target misses (`q03`, `q05`, `q14`, `q26`).
2. **Page Crowding vs. Chunk-Level Detail (`Exp B` vs. `Exp C`)**: `tcs-ar-26` page 24 (6 chunks) crowds the top-6 window on financial queries (`q05`, `q14`, `q22`). Setting `max_chunks_per_page=2` recovers `q14` into the top 6 (`p36-c1` at rank 5, `p58-c3` at rank 6) at zero token cost, but drops the 3rd chunk of page 24 (`p24-c2`, containing `66,838`) on `q22`. Increasing `top_k` to `8` (`exp5a-topk8`) recovers `q14`'s chunks at ranks 7–8 with **zero retrieval regressions** on any question, at a `+$0.000197`/query token cost (+29% input tokens).
3. **Why `q14` Still Refused in Generation & Why `q03`, `q05`, `q26` Missed Top-10**:
   - On `q14`, `tcs-ar-26-p36-c1` contains `8.8%`, but because page 36 is `degraded`, the phrase `"(excluding exceptional items)"` was stripped during corrupted-glyph filtering, causing the generator to conservatively refuse.
   - Full-index rank diagnostics show `q26` (`tcs-ar-26-p25-c0`, attrition callout tile without `"TCS"` or `"2026"` in chunk text) sits at **rank 17**, `q03` (`tcs-ar-26-p36-c0`, dividend table chunk split from narrative chunk `p36-c1`) sits at **rank 78**, and `q05` (`tcs-ar-26-p8-c7`, degraded `"Net M argin"` text) sits at **rank 100**. Recovering those requires lexical/BM25 hybrid search or neighbouring-chunk/table context rather than top-$k$ or page-cap tuning.

---

## 7. Phase 5b: Scorer Date Fix, `baseline-v3` Rescore, and Degraded-Page Text-Repair Feasibility

### A. Offline Scorer Date Fix & Rescores (`--rescore`, 0 API Calls)
- **Fix (`generation/checks.py`)**: Tightened `_CANDIDATE_NUMBER_RE` from `r"(?<!\d)(?:\d+(?:,\s*\d+)+(?:\.\d+)?|\d+\.\d+|\d{3,})(?!\d)"` to `r"(?<!\d)(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d+|\d{3,})(?!\d)"` so dates like `"March 31, 2025"` or `"31, 2026"` are not parsed as comma-grouped numbers (`312025` / `312026`).
- **Rescore Results**:
  - `baseline-v2` $\to$ **`baseline-v3`** (`20261012T005928Z_baseline-v3`): `0` metric or outcome changes (`19 CORRECT`, `1 WRONG`, `5 FALSE_REFUSAL`, `4 CORRECT_REFUSAL`, `questions_with_unsupported_numbers = 0`). **`baseline-v3` is now the locked reference baseline.**
  - `exp5a-pagecap2` $\to$ **`exp5a-pagecap2-s3`** (`20261012T005932Z_exp5a-pagecap2-s3`): `questions_with_unsupported_numbers` dropped from `1` (`['q06']`) to **`0` (`[]`)**.
  - `exp5a-topk8` $\to$ **`exp5a-topk8-s3`** (`20261012T005936Z_exp5a-topk8-s3`): `questions_with_unsupported_numbers` dropped from `1` (`['q06']`) to **`0` (`[]`)**.

### B. Degraded-Page Text-Repair Feasibility (`tcs-ar-26` Pages 8, 24, 33, 36, 58)
Read-only comparison of **Current Stored Text** (`stored_text`), **Method A (Conflict-Free Glyph Map)**, and **Method B (RapidOCR at 250 DPI)** on `tcs-ar-26` pages `8, 24, 33, 36, 58`:

| Page | Method | Number Agreement (vs. PyMuPDF) | Numbers Changed / Lost | Target Phrases Present (`excluding exceptional items` / `Net Margin` / `Operating Income` / `dividend` / `attrition`) | Time / Page |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **8** | **Current Stored** | `98.0%` (`48/49`) | `110` lost (`₹110ͬ-` dropped by token filter) | `Operating Income`, `dividend` (`Net M argin` has space) | `97.1 ms` |
| **8** | **Method A (Glyph Map)** | **`100.0%` (`49/49`)** | **None (`110` recovered as `₹110/-`)** | `Operating Income`, `dividend` (`Net M argin` has space) | `~97 ms` |
| **8** | **Method B (RapidOCR)** | `98.0%` (`48/49`) | `3600` $\to$ `360°`; spurious `99` added | `Net Margin`, `Operating Income`, `dividend` | `33.06 s` |
| **24** | **Current Stored** | `100.0%` (`102/102`) | None | `dividend` | `82.4 ms` |
| **24** | **Method A (Glyph Map)** | **`100.0%` (`102/102`)** | **None** | `dividend` (plus `declared`, `interim`, `dividends`, `special`, `total`) | `~82 ms` |
| **24** | **Method B (RapidOCR)** | `98.0%` (`100/102`) | **`31` (final dividend) & `25.3` lost** (inverted crop); spurious `962657`, `3`, `5` | `dividend` | `24.12 s` |
| **33** | **Current Stored** | `100.0%` (`39/39`) | None (`0` corrupted glyphs on page) | `dividend` | `23.5 ms` |
| **33** | **Method A (Glyph Map)** | **`100.0%` (`39/39`)** | **None** | `dividend` | `~24 ms` |
| **33** | **Method B (RapidOCR)** | `84.6%` (`33/39`) | Step badge numbers `1, 1, 2, 2, 4, 4` lost | `dividend` | `21.49 s` |
| **36** | **Current Stored** | `100.0%` (`75/75`) | None | `dividend` (`excluding exceptional items` dropped by token filter) | `12.2 ms` |
| **36** | **Method A (Glyph Map)** | **`100.0%` (`75/75`)** | **None** | **`excluding exceptional items`**, `dividend` | `~12 ms` |
| **36** | **Method B (RapidOCR)** | `97.3%` (`73/75`) | **`52,820` $\to$ `352,820` & `52,391` $\to$ `352,391`** (`₹` read as `3`) | `excluding exceptional items`, `dividend` | `20.60 s` |
| **58** | **Current Stored** | `100.0%` (`40/40`) | None | *(none on page; `Earnings per share`, `gratuity`, `compensated absences`)* | `15.5 ms` |
| **58** | **Method A (Glyph Map)** | **`100.0%` (`40/40`)** | **None** | *(none on page; recovers `Earnings`, `changes`, `gratuity`, `compensated absences`)* | `~16 ms` |
| **58** | **Method B (RapidOCR)** | `95.0%` (`38/40`) | Footnote numbers `133` and `2013` lost | *(none on page; glues `ofgratuityof1,816croreandlong-termcompensated`)* | `26.78 s` |

- **Conclusion**: **Method A (Conflict-Free Glyph Map)** is strictly superior for production: it guarantees **100% numeric fidelity** (`305/305` numeric occurrences across the 5 pages, recovering `110` on Page 8 and `"excluding exceptional items"` on Page 36), adds `0 ms` overhead and `0 MB` dependencies, whereas **RapidOCR (Method B)** corrupts `₹52,820` into `352,820` and `₹52,391` into `352,391` on Page 36, loses the `₹31` final dividend on Page 24 due to an inverted text-box crop, adds `~58 MiB` of dependencies (Apache-2.0), and costs `20.6–33.1 s` per page on CPU.

---

## 8. Phase 5c: Deterministic Glyph Repair Re-Index (`exp5c-glyph-retrieval` & `exp5c-glyph-repair` vs. `baseline-v3`)

After applying the Phase 5c deterministic glyph repair (`34` conflict-free 1:1 mappings, `7` vocabulary-gated ambiguous ASCII substitutions, and vocabulary-gated split-capital joining on the `30` degraded TCS pages) and re-indexing (`3,110` chunks: `1,684` `infosys-ar-26`, `1,426` `tcs-ar-26`), we evaluated `--retrieval-only` (`20261012T024510Z_exp5c-glyph-retrieval`) and full generation with `--judge` (`20261012T025001Z_exp5c-glyph-repair`) on the `29` ready `dev` questions against `baseline-v3` (`20261012T005928Z_baseline-v3`).

### A. Aggregate Comparison (`baseline-v3` vs. `exp5c-glyph-repair`)

| Metric | `baseline-v3` | `exp5c-glyph-repair` | Delta (`B - A`) |
| :--- | :--- | :--- | :--- |
| `page_hit@1` / `@3` / `@6` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.0000` / `0.0000` / `0.0000` |
| `MRR` | `0.6480` | `0.6413` | `-0.0067` |
| `text_hit_any@1` / `@3` / `@6` | `0.6000` / `0.7200` / `0.8400` | `0.6000` / `0.7200` / **`0.8800`** | `0.0000` / `0.0000` / **`+0.0400`** |
| `text_hit_all@1` / `@3` / `@6` | `0.5600` / `0.7200` / `0.8400` | `0.5600` / `0.7200` / **`0.8800`** | `0.0000` / `0.0000` / **`+0.0400`** |
| `distinct_pages@6` | `5.00` | `4.88` | `-0.12` |
| `CORRECT` | `19` | `19` | `0` (`0` regressions on `CORRECT` questions) |
| `WRONG` | `1` | `3` | `+2` (`q05`, `q14`: `FALSE_REFUSAL -> WRONG`) |
| `FALSE_REFUSAL` | `5` | `3` | `-2` (`q05`, `q14` moved out of refusal) |
| `CORRECT_REFUSAL` | `4` | `4` | `0` |
| `ANSWERED_ANYWAY` | `0` | `0` | `0` |
| `grounded_rate` (answerable) | `0.8000` | `0.8800` | `+0.0800` |
| `questions_with_unsupported_numbers` | `0` | `1` (`['q05']`) | `+1` (calculated `19.78%` from `52,820 / 2,67,021`) |
| `faithfulness_rate` | `1.0000` (`20/20`) | `0.9545` (`21/22`) | `-0.0455` (`q05` arithmetic division flagged by judge) |
| `citation_precision` | `1.0000` (`34/34`) | `1.0000` (`38/38`) | `0.0000` |
| `avg_cost_per_query_usd` | `$0.000691` | `$0.000702` | `+$0.000011` |

### B. Slice Breakdown by Expected-Page Text Quality (`all_degraded` vs. `all_clean`)

| Expected-Page Quality | Count | `baseline-v3` `page_hit@6` | `exp5c` `page_hit@6` | `baseline-v3` `MRR` | `exp5c` `MRR` | `baseline-v3` `text_hit_all@6` | `exp5c` `text_hit_all@6` | `baseline-v3` `CORRECT` | `exp5c` `CORRECT` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_clean`** | `14` | `0.8571` | `0.8571` | `0.6333` | `0.6333` | `0.9286` | `0.9286` | `13/14` | `13/14` |
| **`all_degraded`** | `8` | `0.7500` | `0.7500` | `0.6042` | `0.5833` | `0.6250` (`5/8`) | **`0.7500` (`6/8`, `+0.1250`)** | `3/8` | `3/8` |
| **`mixed`** | `3` | `1.0000` | `1.0000` | `0.8333` | `0.8333` | `1.0000` | `1.0000` | `3/3` | `3/3` |

### C. Diagnosis of `q03`, `q05`, `q14`, and `q30`
- **`q03` (`WRONG`, `page_hit@6 = True`, `text_hit_all@6 = False`)**: Retrieves `tcs-ar-26-p36-c1` at rank 3 (`0.7522`, `"FY 2026 is 11,000% of the paid-up value of each share"`), while the preceding dividend table chunk `tcs-ar-26-p36-c0` (`"Total 110"`) remains outside top-6 due to chunk-boundary splitting between table and narrative.
- **`q05` (`FALSE_REFUSAL -> WRONG`, `page_hit@6 = False`, `text_hit_all@6 = False`)**: Repairing `tcs-ar-26-p36-c1` (rank 4, `0.7340`) restored `"The profit for the year attributable to shareholders (excluding exceptional items) for FY 2026 was ₹52,820 crore"` alongside `"revenue from operations for FY 2026 was ₹2,67,021 crore"`. Because `q05` defines net profit margin in parentheses as `(profit attributable to shareholders, excluding exceptional items, as a percentage of revenue)`, the generator computed `52,820 / 2,67,021 = 19.78% [4]`, which does not match `must_contain: ["19.8"]` (the rounded 1-decimal headline figure on `p8`/`p58`, still crowded out by 4 chunks of `p24`).
- **`q14` (`FALSE_REFUSAL -> WRONG`, `page_hit@6 = False`, `text_hit_all@6: False -> True`)**: Repairing page 36 promoted `tcs-ar-26-p36-c2` into the top 6 at rank 5 (`0.7185`). `p36-c2` starts in the 200-char overlap mid-sentence (`"₹52,820 crore, registering a growth of 8.8%..."`, giving `text_hit_all@6 = True` for `"8.8"`), while its subject `"The profit for the year attributable to shareholders (excluding exceptional items)"` sits in `p36-c1` (crowded out by 4 chunks of `p24`). Inside `p36-c2`, the complete **standalone** sentence states `"...(excluding exceptional items) in FY 2026 was ₹52,391 crore, registering a growth of 9.0%"`, so the generator cited `9.0% [5]`.
- **`q30` (`FALSE_REFUSAL`, `page_hit@6 = True`, `text_hit_all@6 = True`)**: Retrieves `tcs-ar-26-p24-c5` at rank 1 (`0.7911`) and `tcs-ar-26-p24-c3` at rank 6 (`0.7383`, `"Earnings per share (Amount in H) ... FY 2026 145.67"`). Because `p24-c3` labels the chart `"Earnings per share"` without the qualifier `"excluding exceptional items"`, the generator conservatively refuses.

---

## 9. Phase 5d: Extended Glyph Map (`exp5d-glyph-extra` & `exp5d-glyph-full` vs. `baseline-v3` and `exp5c-glyph-repair`)

In Phase 5d, we extended `CONFLICT_FREE_GLYPH_MAP` from `34` to `43` mappings (`Ž`$\to$`o`, `Ŷ`$\to$`n`, `Ă`$\to$`a`, `Ŭ`$\to$`k`, `Ś`$\to$`h`, `Ɖ`$\to$`p`, `Ƶ`$\to$`u`, `ථ`$\to$`" "`, and `\x1c`$\to$`E` with `loader.clean_text` preserving `\x1c`), reducing dropped tokens across the 30 degraded TCS pages from `40` (`83.2%` repaired) to **`12` (`226/238 = 95.0%` repaired; `22/30` degraded pages at `0` dropped tokens)**. After re-indexing (`3,110` chunks), we ran `--retrieval-only` (`20261012T040638Z_exp5d-glyph-extra`) and full evaluation with `--judge` (`20261012T040907Z_exp5d-glyph-full`).

### A. Aggregate Comparison (`baseline-v3` vs. `exp5c-glyph-repair` vs. `exp5d-glyph-full`)

| Metric | `baseline-v3` | `exp5c-glyph-repair` | `exp5d-glyph-full` | Delta (`5d - v3`) | Delta (`5d - 5c`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `page_hit@1` / `@3` / `@6` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.0000` | `0.0000` |
| `MRR` | `0.6480` | `0.6413` | `0.6413` | `-0.0067` | `0.0000` |
| `text_hit_any@1` / `@3` / `@6` | `0.6000` / `0.7200` / `0.8400` | `0.6000` / `0.7200` / `0.8800` | `0.6000` / `0.7200` / **`0.8800`** | **`+0.0400` (`@6`)** | `0.0000` |
| `text_hit_all@1` / `@3` / `@6` | `0.5600` / `0.7200` / `0.8400` | `0.5600` / `0.7200` / `0.8800` | `0.5600` / `0.7200` / **`0.8800`** | **`+0.0400` (`@6`)** | `0.0000` |
| `distinct_pages@6` | `5.00` | `4.88` | `4.88` | `-0.12` | `0.00` |
| `CORRECT` | `19` | `19` | `19` | `0` | `0` |
| `WRONG` | `1` | `3` | `3` | `+2` (`q05`, `q14`) | `0` |
| `FALSE_REFUSAL` | `5` | `3` | `3` | `-2` (`q05`, `q14`) | `0` |
| `CORRECT_REFUSAL` | `4` | `4` | `4` | `0` | `0` |
| `ANSWERED_ANYWAY` | `0` | `0` | `0` | `0` | `0` |
| `grounded_rate` (answerable) | `0.8000` | `0.8800` | `0.8800` | `+0.0800` | `0.0000` |
| `questions_with_unsupported_numbers` | `0` | `1` (`['q05']`) | `1` (`['q05']`) | `+1` | `0` |
| `faithfulness_rate` | `1.0000` (`20/20`) | `0.9545` (`21/22`) | `0.9545` (`21/22`) | `-0.0455` | `0.0000` |
| `citation_precision` | `1.0000` (`34/34`) | `1.0000` (`38/38`) | `1.0000` (`39/39`) | `0.0000` | `0.0000` |
| `avg_cost_per_query_usd` | `$0.000691` | `$0.000702` | `$0.000704` | `+$0.000013` | `+$0.000002` |

### B. `q03` Dividend Chunk Analysis (`Step 6`)
- **`tcs-ar-26-p24-c4`** contains the repaired sentence: `"Based on the company’s performance, the Board of Directors have declared three interim dividends of ₹11 each, a special dividend of ₹46, and recommended a final dividend of ₹31 (pending shareholders' approval at this A'M), for a total of ₹110 per share for FY 2026."`
- For `q03` (`"What dividend did TCS declare for fiscal 2026?"`), `tcs-ar-26-p24-c4` ranks **`12` / `1,426`** (`score = 0.7216`), while `tcs-ar-26-p24-c5` (rank 2, `0.7524`), `tcs-ar-26-p24-c6` (rank 5, `0.7429`), and `tcs-ar-26-p24-c1` (rank 6, `0.7330`) take 3 of the top 6 slots, leaving `p24-c4` just outside top-6.

---

## 10. Phase 5e: Neighbour-Chunk Expansion (`exp5e-neighbors-retrieval` & `exp5e-neighbors-full` vs. `baseline-v3` and `exp5d-glyph-full`)

In Phase 5e (no re-index), we added `retrieval.neighbor_window` (default `0`) and `retrieval.max_context_chunks` (default `null`). With `neighbor_window=1` and `max_context_chunks=10`, after `top_k=6` primary retrieval, for each of the top $N = 3$ primary hits we append same-source, same-page neighbours `c-1` and `c+1` (deduplicated and tagged `neighbor_of=<chunk_id>`), capped at `10` total context chunks. Primary retrieval metrics (`page_hit@k`, `MRR`, `text_hit_any/all@k`) continue to be computed strictly on the primary ranking, while `text_hit_any@context`, `text_hit_all@context`, and `mean_context_chunks` measure the expanded context window passed to the generator.

### A. Aggregate Comparison (`baseline-v3` vs. `exp5d-glyph-full` vs. `exp5e-neighbors-full`)

| Metric | `baseline-v3` | `exp5d-glyph-full` | `exp5e-neighbors-full` | Delta (`5e - v3`) | Delta (`5e - 5d`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `page_hit@1` / `@3` / `@6` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.0000` | `0.0000` |
| `MRR` | `0.6480` | `0.6413` | `0.6413` | `-0.0067` | `0.0000` |
| `text_hit_any@1` / `@3` / `@6` | `0.6000` / `0.7200` / `0.8400` | `0.6000` / `0.7200` / `0.8800` | `0.6000` / `0.7200` / `0.8800` | `+0.0400` (`@6`) | `0.0000` |
| `text_hit_all@1` / `@3` / `@6` | `0.5600` / `0.7200` / `0.8400` | `0.5600` / `0.7200` / `0.8800` | `0.5600` / `0.7200` / `0.8800` | `+0.0400` (`@6`) | `0.0000` |
| `text_hit_any@context` | `0.8400` | `0.8800` | **`0.9200`** | **`+0.0800`** | **`+0.0400`** |
| `text_hit_all@context` | `0.8400` | `0.8800` | **`0.9200`** | **`+0.0800`** | **`+0.0400`** |
| `distinct_pages@6` | `5.00` | `4.88` | `4.88` | `-0.12` | `0.00` |
| `mean_context_chunks` | `6.21` (primary) | `6.21` (primary) | **`9.41`** | `+3.20` | `+3.20` |
| `CORRECT` | `19` | `19` | **`20`** | **`+1` (`q03`)** | **`+1` (`q03`)** |
| `WRONG` | `1` | `3` | **`2`** | `+1` (`q05`, `q14` vs `q03`) | **`-1` (`q03`)** |
| `FALSE_REFUSAL` | `5` | `3` | `3` | `-2` (`q05`, `q14`) | `0` |
| `CORRECT_REFUSAL` | `4` | `4` | `4` | `0` | `0` |
| `ANSWERED_ANYWAY` | `0` | `0` | `0` | `0` | `0` |
| `grounded_rate` (answerable) | `0.8000` | `0.8800` | `0.8400` | `+0.0400` | `-0.0400` |
| `questions_with_unsupported_numbers` | `0` | `1` (`['q05']`) | `1` (`['q05']`) | `+1` | `0` |
| `faithfulness_rate` | `1.0000` (`20/20`) | `0.9545` (`21/22`) | **`1.0000` (`21/21`)** | `0.0000` | **`+0.0455`** |
| `citation_precision` | `1.0000` (`34/34`) | `1.0000` (`39/39`) | **`1.0000` (`38/38`)** | `0.0000` | `0.0000` |
| `avg_input_tokens` | `2051.2` | `2063.0` | `2840.1` | `+788.9` | `+777.1` |
| `avg_cost_per_query_usd` | `$0.000691` | `$0.000701` | `$0.000951` | `+$0.000260` | `+$0.000250` |

### B. Per-Question Changes & Honest Summary Across Phases 5a–5e
1. **Honest Assessment of Phases 5a–5d vs. Phase 5e**:
   - Phases **5a–5d** did **not** increase the end-to-end `CORRECT` count on the locked 6-chunk retrieval configuration (`19 CORRECT` in `baseline-v2`, `baseline-v3`, `exp5a-pagecap2`, `exp5c-glyph-repair`, and `exp5d-glyph-full`). Phase 5c/5d's deterministic glyph repair was essential for restoring corrupted words on degraded TCS pages (such as `"excluding exceptional items"` on Page 36 and `"₹110"` on Page 24, lifting `text_hit_all@6` from `0.8400` to `0.8800`), but because the target figures and their subject labels were split across adjacent chunks on Pages 24 and 36, glyph repair alone moved `q05` and `q14` from `FALSE_REFUSAL` to `WRONG` without increasing `CORRECT`.
   - **Phase 5e (`neighbor_window=1`, `max_context_chunks=10`)** directly resolves the chunk-boundary split on **`q03`**: expanding primary ranks `#2` (`tcs-ar-26-p24-c5`) and `#3` (`tcs-ar-26-p36-c1`) appends neighbour chunks `[8] tcs-ar-26-p24-c4` (`neighbor_of=tcs-ar-26-p24-c5`) and `[9] tcs-ar-26-p36-c0` (`neighbor_of=tcs-ar-26-p36-c1`), both of which contain the `₹110` per-share total dividend. As a result, `text_hit_all@context` rises to **`0.9200` (`23/25`)**, **`q03` moves from `WRONG` to `CORRECT`**, `CORRECT` rises from **`19` to `20`** (`0` regressions on any previously `CORRECT` question), and `faithfulness_rate` returns to **`1.0000` (`21/21`)**.
2. **Why `q14` Remains `WRONG` in Phase 5e**:
   - On `q14`, the top $N = 3$ primary hits are all on Page 24 (`[1] tcs-ar-26-p24-c5`, `[2] tcs-ar-26-p24-c3`, `[3] tcs-ar-26-p24-c1`), while `tcs-ar-26-p36-c2` (`"...₹52,820 crore, registering a growth of 8.8%..."`) sits at primary rank `#5`. Because neighbour expansion only expands the top $N = 3$ primary hits, `tcs-ar-26-p36-c1` (which holds the consolidated subject `"The profit for the year attributable to shareholders (excluding exceptional items)"` for `p36-c2`'s opening overlap) is not added to the context unless Page 24 crowding is also capped (`max_chunks_per_page=2`) or $N$ is increased.

---

## 11. Phase 5f: Step 0 Rescores, Neighbour Top-$N$ (`exp5f-n6`), and General Prompt Rules (`exp5f-prompt`) Measured with 3 Repeats

### A. Step 0: Offline Rescores Against Committed `golden_questions.json` (`q05`: `"19.8|19.78"`)

| Run Label | Rescored File | `CORRECT` (Before $\to$ After) | `WRONG` (Before $\to$ After) | `FALSE_REFUSAL` | `CORRECT_REFUSAL` | Question Outcome Changed |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`baseline-v3`** | `20261012T125413Z_baseline-v3` | `19` $\to$ **`19`** | `1` $\to$ `1` | `5` | `4` | None (`q05` was `FALSE_REFUSAL`) |
| **`exp5d-glyph-full`** | `20261012T125416Z_exp5d-glyph-full` | `19` $\to$ **`20`** | `3` $\to$ `2` | `3` | `4` | `q05`: `WRONG -> CORRECT` |
| **`exp5e-neighbors-full`** | `20261012T125420Z_exp5e-neighbors-full` | `20` $\to$ **`21`** | `2` $\to$ `1` | `3` | `4` | `q05`: `WRONG -> CORRECT` |

### B. Step A & Step B: Aggregate 3-Repeat Comparison (`exp5e-r3` vs. `exp5f-n6` vs. `exp5f-prompt`)

| Metric | `exp5e-r3` (`N=3, rules_v2=false`) | `exp5f-n6` (`N=6, rules_v2=false`) | `exp5f-prompt` (`N=6, rules_v2=true`) | Delta (`prompt - exp5e-r3`) |
| :--- | :---: | :---: | :---: | :---: |
| `page_hit@1` / `@3` / `@6` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.0000` |
| `MRR` | `0.6413` | `0.6413` | `0.6413` | `0.0000` |
| `text_hit_all@6` | `0.8800` | `0.8800` | `0.8800` | `0.0000` |
| `text_hit_any@context` / `all@context` | `0.9200` / `0.9200` | `0.9200` / `0.9200` | `0.9200` / `0.9200` | `0.0000` |
| `mean_context_chunks` | `9.41` | `10.00` | `10.00` | `+0.59` |
| `CORRECT` (repeat 0) | `21` | `21` | **`23`** | **`+2`** |
| `WRONG` (repeat 0) | `1` | `1` | **`0`** | **`-1`** |
| `FALSE_REFUSAL` (repeat 0) | `3` | `3` | **`2`** | **`-1`** |
| `CORRECT_REFUSAL` (repeat 0) | `4` | `4` | `4` | `0` |
| `ANSWERED_ANYWAY` | `0` | `0` | `0` | `0` |
| `accuracy` (3-repeat mean, min–max) | `0.8621` (`0.8621–0.8621`) | `0.8736` (`0.8621–0.8966`) | **`0.9310` (`0.9310–0.9310`)** | **`+0.0689`** |
| `answerable_accuracy` (3-repeat mean) | `0.8400` (`0.8400–0.8400`) | `0.8533` (`0.8400–0.8800`) | **`0.9200` (`0.9200–0.9200`)** | **`+0.0800`** |
| `flip_rate` (3 repeats) | **`0.0000`** (`[]`) | `0.0345` (`['q14']`) | **`0.0000`** (`[]`) | `0.0000` |
| `grounded_rate` (answerable) | `0.8400` | `0.8800` | `0.8800` | `+0.0400` |
| `questions_with_unsupported_numbers` | `1` (`['q05']`) | `1` (`['q05']`) | **`0` (`[]`)** | **`-1`** |
| `faithfulness_rate` | `1.0000` (`21/21`) | `1.0000` (`22/22`) | `0.9545` (`21/22`) | `-0.0455` (`q30`) |
| `citation_precision` | `1.0000` (`38/38`) | `1.0000` (`39/39`) | **`1.0000` (`40/40`)** | `0.0000` |
| `avg_input_tokens` | `2840.1` | `3018.7` | `3093.7` | `+253.6` |
| `avg_cost_per_query_usd` | `$0.000945` | `$0.000999` | `$0.001014` | `+$0.000069` |

### C. Step C: Per-Question Changes (`exp5f-prompt` vs. `exp5e-r3`), Repeat Stability, and Noise Analysis
1. **Every Question Whose Outcome Changed Between `exp5e-r3` and `exp5f-prompt`**:
   - **`q14` [IMPROVED, STABLE `3/3` REPEATS]**: `WRONG, WRONG, WRONG` (`exp5e-r3`) $\to$ `WRONG, WRONG, CORRECT` (`exp5f-n6`) $\to$ **`CORRECT, CORRECT, CORRECT`** (`exp5f-prompt`). Expanding `neighbor_top_n=6` added `[9] tcs-ar-26-p36-c1` (the consolidated `8.8%` sentence), and Rule 9 (Basis: default to consolidated unless `"standalone"` is asked) made the model select `8.8% [5][9]` on all 3 repeats instead of standalone `9.0% [5]`.
   - **`q22` [IMPROVED, STABLE `3/3` REPEATS]**: `FALSE_REFUSAL, FALSE_REFUSAL, FALSE_REFUSAL` (`exp5e-r3` & `exp5f-n6`) $\to$ **`CORRECT, CORRECT, CORRECT`** (`exp5f-prompt`). Rule 8 (Accounting synonyms: `"operating income" = "operating profit" = "EBIT"`) enabled the generator to answer with `"operating profit (operating income) of ₹66,838 crore [5]"` from `tcs-ar-26-p24-c2` across all 3 repeats.
   - **`q30` [IMPROVED, STABLE `3/3` REPEATS]**: `FALSE_REFUSAL, FALSE_REFUSAL, FALSE_REFUSAL` (`exp5e-r3` & `exp5f-n6`) $\to$ **`CORRECT, CORRECT, CORRECT`** (`exp5f-prompt`). Answered `"₹145.99 [6]"` from `tcs-ar-26-p24-c3` across all 3 repeats.
   - **`q05` [REGRESSED, STABLE `3/3` REPEATS]**: `CORRECT, CORRECT, CORRECT` (`exp5e-r3` & `exp5f-n6`) $\to$ **`FALSE_REFUSAL, FALSE_REFUSAL, FALSE_REFUSAL`** (`exp5f-prompt`). Because the headline `"Net Margin of 19.8%"` chunk (`p8-c7` / `p58-c1`) is still not in the retrieved context (`page_hit@6 = False`, `text_hit_all@context = False`), under `rules_v2=true` the generator adhered strictly to Rule 3 (*"Do not calculate unless the user asks"*) rather than computing `52,820 / 2,67,021 = 19.78%`, eliminating the `unsupported_numbers = ['19.78']` flag.
2. **Noise Assessment on a 29-Question Set**:
   - The `exp5f-n6` vs. `exp5e-r3` difference (`+0.33` questions / `+1.15%` mean accuracy from `q14` flipping on `1/3` repeats) is **within the single-question generation noise** of a 29-question set.
   - In `exp5f-prompt` vs. `exp5e-r3`, all 4 per-question shifts are **100% repeat-stable (`3/3` vs `0/3`, `flip_rate = 0.0000`)** and causally explained by the new rules (`q14`, `q22`, `q30` vs `q05`). However, in terms of dataset sampling granularity on $n = 29$ (`1` question $= 3.45\%$ accuracy, binomial SE $\approx 4.7\%–6.4\%$ or `~1.4–1.9` questions), a net `+2` question gain (`21 -> 23 CORRECT`, `+6.90%` overall / `+8.00%` answerable) is `~1.1–1.5` standard errors of the 29-question sample proportion, so while the per-question rule fixes are deterministic across repeats, the aggregate percentage delta is at the margin of 29-question sample noise.

---

## 12. Phase 5g: Step 0 Rescores, Rule 2 Faithfulness Update (`exp5g-rule2`), and In-House BM25 + RRF Hybrid Retrieval (`exp5g-hybrid`)

### A. Step 0: Offline Rescores Against Committed `golden_questions.json` (`636e644`)

| Run Label | Rescored File | `num_questions` | `CORRECT` | `WRONG` | `FALSE_REFUSAL` | `CORRECT_REFUSAL` | `accuracy` (mean) | `answerable_accuracy` | `flip_rate` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline-v3`** | `20261013T012525Z_baseline-v3` | `29` | `19` | `1` (`q03`) | `5` (`q05,q14,q22,q26,q30`) | `4` | `0.7931` | `0.7600` | `0.0000` |
| **`exp5e-r3`** | `20261013T012539Z_exp5e-r3` | `29` | `21` | `1` (`q14`) | `3` (`q22,q26,q30`) | `4` | `0.8621` | `0.8400` | `0.0000` |
| **`exp5f-prompt`** | `20261013T012550Z_exp5f-prompt` | `29` | `23` | `0` | `2` (`q05,q26`) | `4` | `0.9310` | `0.9200` | `0.0000` |

*(Note: Commit `636e644` promoted `q27` (`"What is Infosys's employee attrition rate?"`, `split="dev"`, `type="unanswerable"`, `answerable=false`) from `draft_needs_fill` to `ready_unverified`, so new live runs in Phase 5g evaluate all `30` dev questions: `25` answerable and `5` unanswerable.)*

### B. Step 2 & Step 3: Aggregate Comparison (`exp5f-prompt` vs. `exp5g-rule2` vs. `exp5g-hybrid`)

| Metric | `exp5f-prompt` (`29` qs, Dense) | `exp5g-rule2` (`30` qs, Dense) | `exp5g-hybrid` (`30` qs, BM25+RRF) | Delta (`hybrid - rule2`) |
| :--- | :---: | :---: | :---: | :---: |
| `page_hit@1` / `@3` / `@6` | `0.4800` / `0.8000` / `0.8400` | `0.4800` / `0.8000` / `0.8400` | `0.4400` / `0.7600` / **`0.9600`** | `-0.0400` / `-0.0400` / **`+0.1200`** |
| `MRR` | `0.6413` | `0.6413` | **`0.6460`** | **`+0.0047`** |
| `text_hit_all@6` | `0.8800` (`22/25`) | `0.8800` (`22/25`) | **`1.0000` (`25/25`)** | **`+0.1200`** |
| `text_hit_any@context` / `all@context` | `0.9200` / `0.9200` (`23/25`) | `0.9200` / `0.9200` (`23/25`) | **`1.0000` / `1.0000` (`25/25`)** | **`+0.0800`** |
| `mean_context_chunks` | `10.00` | `10.00` | `10.00` | `0.00` |
| `CORRECT` (repeat 0) | `23` (`/25`) | `22` (`/25`) | **`24` (`/25`)** | **`+2`** |
| `WRONG` (repeat 0) | `0` | `0` | `1` (`q22`) | `+1` |
| `FALSE_REFUSAL` (repeat 0) | `2` (`q05, q26`) | `3` (`q05, q26, q30`) | **`0`** | **`-3`** |
| `CORRECT_REFUSAL` (repeat 0) | `4` (`/4`) | `4` (`/5`) | `4` (`/5`) | `0` |
| `ANSWERED_ANYWAY` (repeat 0) | `0` (`/4`) | `1` (`q27`) | `1` (`q27`) | `0` |
| `accuracy` (overall) | `0.9310` (`27/29`) | `0.8667` (`26/30`) | **`0.9333` (`28/30`)** | **`+0.0666`** |
| `answerable_accuracy` | `0.9200` (`23/25`) | `0.8800` (`22/25`) | **`0.9600` (`24/25`)** | **`+0.0800`** |
| `flip_rate` | `0.0000` (`[]`) | `0.0000` (`[]`) | **`0.0000` (`[]`)** | `0.0000` |
| `grounded_rate` (answerable) | `0.8800` | `0.8800` | **`1.0000`** | **`+0.1200`** |
| `questions_with_unsupported_numbers` | `0` (`[]`) | `0` (`[]`) | **`0` (`[]`)** | `0` |
| `faithfulness_rate` (LLM Judge) | `0.9545` (`21/22`) | `0.8261` (`19/23`) | **`0.9231` (`24/26`)** | **`+0.0970`** |
| `citation_precision` (LLM Judge) | `1.0000` (`40/40`) | `0.9762` (`41/42`) | **`1.0000` (`44/44`)** | **`+0.0238`** |
| `avg_input_tokens` | `3093.7` | `3104.0` | `3133.6` | `+29.6` |
| `avg_cost_per_query_usd` | `$0.001014` | `$0.001035` | `$0.001049` | `+$0.000014` |
| `retrieve_ms` p50 / `total_ms` p50 | `53.21` / `1462.58` | `51.77` / `1482.88` | `96.30` (`36.93` ret-only) / `1257.97` | `+44.53` / `-224.91` |

### C. Per-Question Changes (`exp5g-hybrid` vs. `exp5g-rule2`) & Freeze Recommendation
1. **Improvements in `exp5g-hybrid` vs. `exp5g-rule2` (`+3 CORRECT`, all repeat-stable)**:
   - **`q05` (`FALSE_REFUSAL -> CORRECT`, `page_hit@6: False -> True`)**: BM25 + RRF retrieves `tcs-ar-26-p58-c0` at primary rank `[5]`, which explicitly states `"Profit after tax attributable to shareholders of the Company 52,820 | 19.8 (% of Revenue)"` on a consolidated basis (`judge_faithful = True`, `unsupported_numbers = []`).
   - **`q26` (`FALSE_REFUSAL -> CORRECT`, `page_hit@6: False -> True`)**: BM25 + RRF retrieves `tcs-ar-26-p25-c0` at `[1]` and `tcs-ar-26-p55-c3` at `[2]`, both containing TCS's `"13.7%"` voluntary LTM attrition rate (`judge_faithful = True`).
   - **`q30` (`FALSE_REFUSAL -> CORRECT`, `judge_faithful = True`)**: BM25 + RRF retrieves `tcs-ar-26-p58-c0` at primary rank `[6]` (`"overview of the consolidated financial results... Earnings per share (in ₹) 145.99"`), resolving both the `exp5g-rule2` refusal and the `exp5f-prompt` faithfulness failure on `q30`.
2. **Regression in `exp5g-hybrid` vs. `exp5g-rule2` (`-1 CORRECT`)**:
   - **`q22` (`CORRECT -> WRONG`)**: Because `tcs-ar-26-p5-c3` (USD highlights on PDF p5) uses the exact phrase `"Operating Income"` (`$7.5 billion`), BM25 promoted `p5-c3` to primary rank `[4]` alongside `[1] tcs-ar-26-p24-c2` (`"Operating Profit ... 66,838"`), and the generator answered `"$7.5 billion [4]"` instead of `"₹66,838 crore"`.
3. **Step 4 Freeze Recommendation**:
   - We recommend freezing **Step 3 (`exp5g-hybrid`: `neighbor_window=1, neighbor_top_n=6, max_context_chunks=10, rules_v2=true, hybrid.enabled=true, bm25_top_k=20, rrf_k=60, fusion_weight=0.5`)**. While the net `+2` `CORRECT` improvement (`22 -> 24`, `+6.67%` overall / `+8.00%` answerable) is ~1.2–1.5 standard errors of a 30-question sample proportion, the retrieval gain (`page_hit@6: 0.8400 -> 0.9600`, `text_hit_all@6: 0.8800 -> 1.0000`, `text_hit_all@context: 0.9200 -> 1.0000`, `FALSE_REFUSAL: 3 -> 0`) is deterministic and closes every remaining retrieval miss on the dev set.
