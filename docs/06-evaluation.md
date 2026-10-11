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
3. **Unverified & Draft Questions (`human_verified = false`)**: In `golden_questions.json`, `28` of `40` questions still have `human_verified=false` (`12` are `human_verified=true`; `39` are `ready_unverified` and `1` is `draft_needs_fill`). Unverified questions must not be used for final headline numbers until verified against the source PDFs.
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
