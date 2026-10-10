# Architecture Decision Records (ADRs)

This document captures key design decisions, their rationale, alternatives considered, and trade-offs.

---

### Decision 1: Project & Package Management with `uv` and Python 3.12

* **Context:** The project requires fast, reproducible dependency resolution, an isolated virtual environment, and standard packaging structure for Windows and cross-platform compatibility.
* **Options Considered:**
  1. `pip` + `venv` with `requirements.txt`: Ubiquitous, but lacks strict dependency lock pinning and has slower installation speeds.
  2. Poetry: Well-established package manager, but adds slower dependency resolution and custom configuration overhead.
  3. `uv` (`pyproject.toml`, `uv.lock`, `.python-version`): Modern Rust-based tool managing Python versions, virtual environments, and package locks.
* **Choice:** `uv` with `src/` layout and Python 3.12.
* **Reason:** Extremely fast resolution, atomic lockfile management (`uv.lock`), direct execution via `uv run`, and alignment with modern Python packaging standards (PEP 517/621).
* **Trade-off:** Newer toolchain compared to legacy `pip`, requiring collaborators to install `uv`.

---

### Decision 2: Configuration & Secret Separation via YAML, `.env`, and Frozen Dataclasses

* **Context:** Operational parameters (model names, chunk sizes, top-$k$, directory paths) must be tunable without code changes. Sensitive API keys must never be committed to source control. Code must have typed access to settings.
* **Options Considered:**
  1. Environment variables for all settings: Cluttered `.env` file, non-hierarchical, and error-prone.
  2. `pydantic-settings`: Automated parsing, but introduces extra framework abstractions that can obscure basic logic for interview explanations.
  3. YAML (`config/settings.yaml`) for settings + `.env` for secrets + standard Python `dataclasses`: Clean separation between operational hyperparameters and credentials, parsed into immutable typed objects.
* **Choice:** YAML file (`config/settings.yaml`) + `.env` loaded into a frozen `Settings` dataclass.
* **Reason:** Optimizes for readability and explainability. Beginners and interviewers can follow every line of parsing without framework magic. Keeps parameters organized hierarchically and secrets strictly isolated.
* **Trade-off:** Requires explicit manual mapping from loaded YAML dictionary keys to dataclass fields.

---

### Decision 3: Abstract Provider Interface (`BaseProvider`) and Factory Pattern

* **Context:** Both document ingestion and query handling require embedding and LLM generation. Directly importing third-party SDKs across application code tightly couples the system to a single vendor and makes offline unit testing difficult.
* **Options Considered:**
  1. Direct imports of `google-genai` across modules: Rapid initially, but creates tight vendor lock-in and prevents testing without live API keys.
  2. Heavy frameworks (LangChain / LlamaIndex): Provides multi-vendor adapters, but introduces complex abstractions and dependency bloat.
  3. Custom Abstract Base Class (`BaseProvider`) with a Factory (`get_provider`): Minimal ABC exposing only `embed(texts)` and `generate(prompt)`.
* **Choice:** Custom `BaseProvider` ABC with a `get_provider(settings)` factory and a local `FakeProvider`.
* **Reason:** Enforces Dependency Inversion (SOLID). The rest of the codebase remains 100% agnostic to the AI provider. Enables fast, offline unit testing with `FakeProvider` without needing external network calls or fragile mock monkey-patching.
* **Trade-off:** Requires writing thin provider adapters if new model providers are added in the future.

---

### Decision 4: Provider SDK and Model Selection

* **Context:** The application needs reliable embedding and LLM generation capabilities for financial document question answering.
* **Options Considered:**
  1. Legacy `google-generativeai` SDK.
  2. Official unified `google-genai` SDK (v2.x).
* **Choice:** `google-genai` SDK, with `gemini-3.5-flash-lite` for generation and `gemini-embedding-001` (or `text-embedding-004`) for embeddings, parameterized via `config/settings.yaml`.
* **Reason:** `google-genai` is the active, forward-looking SDK. Modern Flash models provide rapid latency and strong instruction following for grounded Q&A, while Google embedding models produce dense vectors optimal for semantic search over technical financial terminology. Model names are isolated in configuration so they can be changed without touching code.
* **Trade-off:** External API dependency for live usage; mitigated by our offline `FakeProvider` for development and testing.

---

### Decision 5: PDF Extraction via PyMuPDF

* **Context:** Multi-hundred-page annual reports (350+ pages) must be parsed quickly, accurately, and reliably while extracting clean text and page numbers.
* **Options Considered:**
  1. `pypdf`: Pure Python, but slow on large documents (several minutes for 380 pages) and struggles with complex multi-column typography.
  2. `pdfplumber`: Excellent table extraction, but high memory footprint and extremely slow on 300+ page books.
  3. `pymupdf` (MuPDF bindings): Fast C-based rendering and parsing engine.
* **Choice:** `pymupdf`.
* **Reason:** Parses 380-page financial reports in under 2 seconds. Provides clean page-by-page access, page counts, and text bounding without heavy dependencies.
* **Trade-off:** Binary C-extensions required (handled automatically via modern wheels). Tables are extracted as flattened text rather than structured dataframes.

---

### Decision 6: Page-Bounded Chunking without Cross-Page Merging

* **Context:** Financial audits require exact page citations. If a chunk merges sentences across page 42 and page 43, provenance is ambiguous.
* **Options Considered:**
  1. Rolling cross-page character windows: Chunks cross page boundaries, smearing page numbers.
  2. Document-level recursive chunking: Ignores physical page numbers altogether.
  3. Strict page-bounded chunking: Chunks are strictly confined within their source page.
* **Choice:** Strict page-bounded chunking.
* **Reason:** Guaranteed provenance integrity. Every chunk is tied to a single, unambiguous page index (`page: N`). The user and auditor can verify the exact source page for every claim.
* **Trade-off:** Information split precisely at the page boundary cannot appear within a single chunk (must be retrieved as adjacent top-$k$ chunks).

---

### Decision 7: Chunk Size (1,000 chars), Overlap (200 chars), and Word Boundary Snapping

* **Context:** We need chunk lengths that capture complete financial disclosure statements and footnotes without exceeding LLM context budgets or diluting semantic vector specificity.
* **Options Considered:**
  1. Small chunks (200-400 chars): High semantic resolution, but splits sentences, financial disclosures, and multi-line notes.
  2. Medium chunks (1,000 chars with 200 overlap): Represents ~150-200 words (1-2 cohesive paragraphs or a financial footnote).
  3. Large chunks (2,500+ chars): Loses granular retrieval precision.
* **Choice:** 1,000 characters with 200 character overlap, snapping to whitespace word boundaries.
* **Reason:** Striking balance between contextual completeness and embedding specificity. Snapping to whitespace guarantees numbers (e.g. `$14,250M`) and words are never severed in half.
* **Trade-off:** Multi-column tables may get interleaved or spread across consecutive chunks.

---

### Decision 8: Stable Deterministic Chunk IDs (`{source}-p{page}-c{chunk_index}`)

* **Context:** Ingestion must be idempotent. Re-running the pipeline should never generate duplicate records in ChromaDB or re-run expensive embedding API calls.
* **Options Considered:**
  1. Random UUIDs (`uuid.uuid4()`): Simple, but prevents deduplication because IDs change on every run.
  2. Full text hashing (`sha256(text)`): Stable, but opaque and hides provenance coordinates.
  3. Hierarchical coordinates (`{source}-p{page}-c{chunk_index}`): e.g. `infosys-ar-26-p42-c1`.
* **Choice:** Coordinate IDs (`{source}-p{page}-c{chunk_index}`).
* **Reason:** 100% deterministic, human-readable, directly traceable in vector databases, and enables instantaneous `store.existing_ids()` set-difference checks to skip previously ingested chunks.
* **Trade-off:** If chunk size or overlap settings are modified later, chunk indices shift, necessitating a full re-index.

---

### Decision 9: Document vs. Query Embedding Task Types & L2 Normalization

* **Context:** Search queries ("What was FY23 operating margin?") are short and interrogative, whereas document chunks are long, dense, and declarative.
* **Options Considered:**
  1. Symmetric generic embeddings: Uses identical model weights without task guidance.
  2. Asymmetric task-type embeddings (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) with L2 normalization.
* **Choice:** Asymmetric task types via Google GenAI SDK and L2 unit-normalization.
* **Reason:** Asymmetric embeddings project queries and document passages into a unified semantic space optimized specifically for retrieval cosine similarity. L2 normalization ensures vectors have unit length ($\|v\|_2 = 1$), guaranteeing accurate cosine distance indexing in ChromaDB.
* **Trade-off:** The provider interface must accept a `purpose` parameter (`"document"` or `"query"`).

---

### Decision 10: Exponential Backoff Retry for Provider API Calls

* **Context:** Embedding hundreds of chunks against public cloud APIs risks encountering temporary rate limits (HTTP 429) or transient backend dropouts (HTTP 5xx).
* **Options Considered:**
  1. Fail-fast: Stop the pipeline immediately on any network error.
  2. Constant retry: Retry immediately or with fixed sleep.
  3. Exponential backoff with configurable jitter and max wait cap.
* **Choice:** Exponential backoff retry (`initial_wait * 2^attempt`, capped at `max_wait_seconds`).
* **Reason:** Transparently absorbs transient rate spikes and temporary server drops without manual developer intervention or pipeline crashes. Configurable in `config/settings.yaml`.
* **Trade-off:** Transient failures will increase execution time during backoff intervals before succeeding.

---

### Decision 11: Free-Tier Rate Limiting via Client-Side Sliding-Window Limiter

* **Context:** Google Gemini API free-tier quotas restrict embedding throughput to approximately 100 requests / items per minute. Ingestion of 3,100+ chunks causes immediate HTTP 429 `RESOURCE_EXHAUSTED` failures if dispatched at full speed.
* **Options Considered:**
  1. Rely exclusively on reactive retry loops when 429 is encountered.
  2. Fixed sleep (`time.sleep(N)`) after every single batch.
  3. Client-side sliding-window rate limiter with configurable quota (`max_items_per_minute = 80`).
* **Choice:** Client-side sliding-window limiter (`SlidingWindowRateLimiter`) with injectable clock/sleep functions.
* **Reason:** A sliding window accurately models time-elapsed quotas across uneven batch requests. It paces outbound embedding calls smoothly, ensuring we stay reliably beneath the 100 req/min limit without unpredictable bursts.
* **Trade-off:** Ingestion takes longer (e.g., ~38 minutes for 3,100 chunks at 80 items/min), but runs reliably without quota crashes.

---

### Decision 12: Proactive Pacing vs. Reactive Retry Only

* **Context:** When consuming rate-limited third-party APIs, systems can either pace requests proactively or blast requests and retry upon HTTP 429.
* **Options Considered:**
  1. Reactive-only: Send batches as fast as possible; catch 429, parse "retry in 27s", and sleep.
  2. Proactive pacing + reactive fallback: Pace throughput to 80% of quota (80 items/min) and retain delay-aware retry as a safety net.
* **Choice:** Hybrid proactive pacing with delay-aware 429 retry fallback.
* **Reason:** Pushing APIs to 100% saturation and repeatedly triggering 429 responses causes quota penalties, burns network connection retries, and wastes time in long penalty cooldowns. Proactive pacing keeps the pipeline flowing smoothly, while the delay-aware 429 handler catches occasional quota fluctuations by parsing Google's exact `retry in Ns` suggestion (+1 second).
* **Trade-off:** Requires maintaining both a rate limiter and a delay-parsing retry loop.

---

### Decision 13: Empirical Text Quality Audit Before Applying Extraction Fixes

* **Context:** Ingestion revealed garbled text (e.g. TCS page 8: broken font encodings and spaced-out letters like `"i n th i s b ack d rop"`). Before applying regex heuristics or changing PDF parsing engines, we need to know the scale and distribution of the problem across both reports.
* **Options Considered:**
  1. Immediately apply ad-hoc regex fixes to `loader.py` (e.g., join single letters).
  2. Switch PDF extraction libraries immediately (e.g., to pdfplumber or OCR).
  3. Build an automated offline audit script (`audit_text_quality.py`) with explainable metrics (`single_letter_ratio`, `odd_char_ratio`, `char_count`) to quantify and catalog defects before touching ingestion code.
* **Choice:** Empirical audit first (`ingestion/quality.py` and `scripts/audit_text_quality.py`).
* **Reason:** Ad-hoc regex replacements can easily corrupt valid acronyms, equations, and tables if applied blindly. Auditing first provides clear data: Infosys is almost completely clean (P95 single-letter ratio is 8.5%), whereas TCS has 31 severely affected pages (P95 single-letter ratio is 19.5%, reaching up to 46% on letter-spaced executive pages). This isolates the exact nature of the defect for a targeted fix.
* **Trade-off:** Delays fixing the text extraction to Phase 2c, but ensures any future fix can be measured against a definitive baseline report (`data/processed/text_quality_report.csv`).

---

### Decision 14: Dual Quality Metric Refinement (Absolute Count + Ratio & Split Numbers)

* **Context:** In Phase 2b, `tcs-ar-26` page 8 contained 41 corrupted glyphs (`"ĐonƐolŝdated ƌevenue oĨ ₹ 26ϳ,021 Đƌoƌe"`), but its `odd_char_ratio` scored 0.0044, slipping under the 0.05 threshold because the page had ~9,300 total characters. Furthermore, numbers were split across spaces (`"2 0 2 6"`, `"44, 0 0 0"`).
* **Options Considered:**
  1. Only lower `odd_char_ratio_threshold` to 0.002: Catches page 8, but short pages with 1-2 innocent symbols (e.g. bullet points) can easily cross a low ratio threshold.
  2. Absolute count threshold (`odd_char_count >= 15`) combined with lowered ratio (`odd_char_ratio > 0.002`) and a dedicated split number detector (`\b\d(?: \d)+\b`).
* **Choice:** Dual threshold (`odd_char_count >= 15` or `odd_char_ratio > 0.002`) plus `number_split_count >= 3`.
* **Reason:** Long dense pages with high character counts cannot dilute an absolute defect count: 41 corrupted glyphs triggers `odd_char_count` (41 >= 15) instantly. The split number regex precisely targets runs of single digits separated by spaces, completely avoiding false positives on clean numbers (`"2026"`, `"44,000"`, `"584,519"`).
* **Trade-off:** Multi-column tables with single digits may produce false positive split number counts when newlines are flattened to spaces.

---

### Decision 15: Extractor Evaluation Findings & Recommendation (`pymupdf.TEXT_INHIBIT_SPACES`)

* **Context:** We evaluated 5 extraction candidates on all 182 flagged pages from both PDFs: (1) PyMuPDF default, (2) PyMuPDF with `TEXT_INHIBIT_SPACES`, (3) PyMuPDF "words" re-joined with gap tolerance, (4) `pypdf`, and (5) `pdfminer.six` with custom `LAParams`.
* **Options Considered:**
  1. Keep PyMuPDF default: Leaves 182 pages flagged with fragmented words and split numbers.
  2. `pypdf`: Slow (12.2s for 182 pages), cleaned only 9 pages, and introduced ligature artifacts (e.g. `/A.calt`).
  3. `pdfminer.six`: Extremely slow (46.5s for 182 pages), cleaned only 30 pages, and severed numbers (e.g. `584, 51`).
  4. PyMuPDF words re-joined: Cleaned 57 pages, but dropped single-character tokens and punctuation.
  5. `pymupdf.TEXT_INHIBIT_SPACES`: Cleaned 79 pages in 0.69 seconds, reduced median single-letter ratio from 0.0674 to 0.0344, and healed split numbers (`584,519`, `44,000+`, `2026`).
* **Choice:** Recommend `PyMuPDF` with `TEXT_INHIBIT_SPACES` as the winning extraction strategy.
* **Reason:** It runs at native C speed (0.69s), completely prevents MuPDF's heuristic space insertion on tight font tracking, and heals split letters and numbers across styled headings without breaking table layouts or dropping text.
* **Trade-off:** Font CMap/ToUnicode corruption (such as Greek Yot `ϳ` for digit `7` on page 8) is embedded inside the PDF binary itself and cannot be solved by any text layout engine without OCR or heuristic character-mapping tables.

---

### Decision 16: Guarding Vector Store Integrity via Collection Metadata and Vector Validation

* **Context:** In a production RAG application, switching embedding models (e.g. from `gemini-embedding-001` with 768 dims to a 1536-dim or 3072-dim model) without clearing the database leads to silent vector store corruption or runtime dimension crashes. Similarly, malformed provider responses must not enter the store.
* **Options Considered:**
  1. Rely on runtime ChromaDB dimension checks: Chroma will raise a cryptic low-level HNSW dimension error, but provides no user guidance on model name mismatches.
  2. Silent overwrite or automatic collection purge: Destructive and can silently delete user data on accidental config edits.
  3. Explicit metadata contract: Store `embedding_model` and `embedding_dimension` in collection metadata. On initialization, validate existing metadata against current settings, raising a descriptive error prompting `--reset` if mismatched. In `GeminiProvider.embed`, strictly validate response length and vector dimensions.
* **Choice:** Explicit metadata tracking and fail-fast startup validation.
* **Reason:** Prevents mixing incompatible vector spaces. The user is immediately alerted if their `settings.yaml` model differs from the indexed collection and is instructed to run `scripts/ingest.py --reset`.
* **Trade-off:** Existing collections created before Phase 2c without metadata will be validated once reset.

---

### Decision 17: Per-Page Best-Of Extraction Selection Over Global Inhibit Flag

* **Context:** In Phase 2c, `pymupdf.TEXT_INHIBIT_SPACES` resolved spaced-out letters and split numbers on TCS headings and tables. However, before deploying it to production, we conducted a full-document regression audit (`check_inhibit_regression.py`) across all 743 pages. The audit revealed that while Infosys had zero regressions, TCS had **38 pages** where word counts dropped or long tokens formed, of which **12 were clean pages** where inhibit spaces concatenated words that were legitimately separate (because the PDF used visual character positioning instead of space characters).
* **Options Considered:**
  1. Global inhibit: Fast and simple, but regresses 12 previously clean pages in TCS.
  2. Heuristic word-splitting post-processor: Complex dictionary lookups, slow, and prone to breaking domain-specific corporate and financial terminology.
  3. Per-page best-of selection: Extract with default flags first. Only inspect inhibit mode if default extraction triggers defect flags (`single_letter_ratio > 0.15` or `number_split_count >= 3`). If inhibit yields a lower composite defect score without introducing glued tokens ($\ge 25$ alphabetic characters), choose inhibit; otherwise retain default.
* **Choice:** Per-page best-of selection (`choose_better_extraction`).
* **Reason:** It provides surgical remediation. 100% of Infosys pages (381 loaded pages) and 325 TCS pages remain on standard extraction with zero risk of regression. Exactly 35 defective TCS pages are healed.
* **Trade-off:** Evaluates two extractions only on the ~5% of pages that trigger defect thresholds, introducing negligible latency overhead.

---

### Decision 18: Dropping Corrupted Font Tokens with Page-Level Anomaly Gating and Allowlisting

* **Context:** In Phase 2d, initial token filtering dropped tokens containing characters in Unicode range `\u0100` to `\u052F` (Latin Extended and Greek blocks) across all pages. However, an empirical audit (`dropped_tokens_report.csv`) revealed a critical false positive: on `infosys-ar-26` page 5, the filter dropped 6 occurrences of the legitimate client brand name `"Mondelēz"` because `ē` is `U+0113`. Furthermore, filtering any page with a single character in that range marked 31 TCS pages and 1 Infosys page as degraded, violating the requirement that only pages with `odd_char_count >= odd_char_count_threshold` (17 pages in TCS) be filtered.
* **Options Considered:**
  1. Blanket Unicode range filter across all pages: Eliminates all corrupted glyphs, but destroys legitimate multi-national brand names, foreign executive names, and accented words (`Mondelēz`).
  2. Heuristic regex character replacement: Substitute `ϳ` with `7`, `Đ` with `C`, etc. Unreliable, domain-fragile, and risks corrupting financial digits.
  3. Two-stage defense: Page-level defect gating + token allowlisting:
     - **Stage 1 (Page-level gating)**: Only evaluate token filtering on pages where `odd_char_count >= odd_char_count_threshold` (default 15). Pages below this threshold are left completely untouched and marked `"clean"`.
     - **Stage 2 (Token allowlist)**: Add a configurable `keep_tokens` list in `settings.yaml` (containing verified legitimate names like `"Mondelēz"`). Even on pages that cross the anomaly threshold, allowlisted tokens are strictly preserved.
* **Choice:** Two-stage defense: Gated filtering on `odd_char_count >= 15` combined with `keep_tokens` allowlist.
* **Reason:** In financial RAG, data extraction must avoid false positives on legitimate business terminology while protecting semantic embeddings from font corruption. Infosys page 5 (`odd_char_count = 6 < 15`) remains completely clean with `"Mondelēz"` preserved. Exactly 17 severely corrupted TCS pages (`odd_char_count >= 15`) have their corrupted tokens (`ĐonƐolŝdated`, `ƌevenue`, `26ϳ,021`) filtered out and are cleanly tagged as `"degraded"`.
* **Trade-off:** 14 TCS pages with low-level font defects (`odd_char_count` between 1 and 13) are left untouched and marked clean; their minor corrupted symbols remain in chunks until OCR is introduced or a lower threshold is configured.

---

### Decision 19: Deferring Visual OCR to a Dedicated Future Pipeline Phase

* **Context:** While visual OCR (e.g. Tesseract or a vision multimodal LLM) could visually read the corrupted numbers on TCS page 8, introducing OCR for the entire corpus adds heavy binary dependencies, external installation requirements, and substantial latency or API expense.
* **Options Considered:**
  1. Add an OCR library (e.g. pytesseract) immediately and re-OCR the entire corpus.
  2. OCR only pages flagged as degraded during ingestion.
  3. Defer OCR, log degraded pages to `data/processed/degraded_pages.csv`, and preserve clean native extraction for 95.8% of corpus pages.
* **Choice:** Defer OCR and isolate degraded pages cleanly via metadata.
* **Reason:** 709 of 741 loaded pages (95.8%) extract crisply with native C-speed PyMuPDF (< 10 seconds total). Deferring OCR keeps the core ingestion pipeline lean, explainable, and fast, while the audit CSV and chunk metadata pinpoint exactly which 32 pages require visual OCR remediation in a future phase.
* **Trade-off:** Users cannot retrieve numbers from the 32 degraded pages until the OCR phase is built.

---

### Decision 20: Gemini Free-Tier Daily Quota Discovery and Fast-Fail Architecture

* **Context:** During full ingestion of 3,109 chunks with Google Gemini embeddings, the pipeline abruptly halted around chunk ~800 with HTTP 429 `RESOURCE_EXHAUSTED`. The underlying error details exposed a daily quota limit:
  `quotaId: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier`, limit `1000` requests/day, retry delay `~75,600s` (~21 hours).
  The Phase 2b retry mechanism mistook this daily exhaustion for a transient per-minute rate limit, waited 22 seconds 5 times in vain, and crashed with an unhandled traceback.
* **Options Considered:**
  1. Retain the existing retry loop: Uselessly attempts retries when the quota window is 21 hours away.
  2. Sleep for 21 hours: Blocks process execution and risks operating system termination.
  3. Immediate detection and graceful fast-fail (`DailyQuotaExceeded`): Inspect error details for `"PerDay"` or retry delays $> 300\text{s}$. Raise `DailyQuotaExceeded` immediately with zero retries. In the CLI, print chunks stored so far, chunks remaining, and estimated resume time, exiting cleanly with exit code 2.
* **Choice:** Immediate detection and graceful exit via `DailyQuotaExceeded`.
* **Reason:** Daily quota ceilings cannot be resolved by short sleep loops. Fast-failing preserves all previously indexed chunks in ChromaDB, informs the developer of exact resume timing, and enables resuming without `--reset` once quota resets.
* **Trade-off:** Requires differentiating per-minute transient throttles (~20-30s) from daily account quotas.

---

### Decision 21: Local Embeddings by Default (FastEmbed ONNX) and Gemini Cloud Generation

* **Context:** A daily limit of 1,000 embedding calls renders Gemini free-tier embeddings impractical for active development, automated CI, and multi-document ingestion (~3,109 chunks requires 4 days of free quota). However, LLM synthesis requires powerful reasoning best served by Gemini 2.5/3.5 Flash models.
* **Options Considered:**
  1. Mandate paid Gemini API billing for all development: Imposes monetary costs on students, interviewers, and open-source contributors.
  2. Full PyTorch local models (`sentence-transformers`): Requires multi-gigabyte PyTorch and CUDA installations, complicating setup on Windows.
  3. Lightweight local ONNX embeddings (`fastembed`) by default, Gemini for generation: Use `BAAI/bge-base-en-v1.5` (768 dimensions) running via ONNX Runtime without PyTorch. Ingestion requires zero API keys and has zero quota limits. Generation uses Gemini via cloud API.
* **Choice:** Local FastEmbed (`BAAI/bge-base-en-v1.5`) as default embedder, Gemini for generation, with Gemini embeddings available via configuration.
* **Reason:** Ingestion of 3,109 chunks runs completely offline in under 60 seconds on CPU, costs \$0, and never exhausts API quotas. `BAAI/bge-base-en-v1.5` produces 768-dimensional embeddings, matching Gemini's vector dimension. Switching between local and Gemini embeddings is a single string change in `settings.yaml`.
* **Trade-off:** FastEmbed downloads an ONNX model file (~210MB) once upon initial use; automated unit tests use mocks and lazy loading to keep standard test runs offline.

---

### Decision 22: Interface Segregation: Separate EmbeddingProvider and GenerationProvider

* **Context:** In Phase 1, `BaseProvider` combined both `embed(texts)` and `generate(prompt)` into a single interface. When using local ONNX embeddings alongside cloud-based Gemini text generation, forcing a single provider class to implement both models violates the Interface Segregation Principle (the "I" in SOLID).
* **Options Considered:**
  1. Keep a single unified `BaseProvider` and implement dummy/noop `generate()` in `LocalEmbeddingProvider`: Violates LSP and ISP; confuses callers when an embedder cannot generate text.
  2. Split into two discrete abstract base classes: `EmbeddingProvider` (`embed()`) and `GenerationProvider` (`generate()`), with `FakeProvider` implementing both for test convenience.
* **Choice:** Split into `EmbeddingProvider` and `GenerationProvider` with separate factory functions (`get_embedder` and `get_generator`).
* **Reason:** Decouples embedding backends from text generation backends. Application pipelines only request the exact capability they need (`indexer.py` requests `EmbeddingProvider`; generation modules request `GenerationProvider`).
* **Trade-off:** Call sites that previously instantiated `get_provider` must specify whether they need an embedder or generator.

---

### Decision 23: Confirmed Glyph Repair for Corrupted Numerals and Punctuation (Zero Guessing on Letters)

* **Context:** In `tcs-ar-26.pdf`, 17 pages exhibited font encoding defects where numbers in headings and financial disclosures extracted as corrupted Greek/Coptic symbols (e.g. `"26ϳ,021"`, `"₹Ϯ͕ ϲϳ͕ ϬϮ1"`). Dropping these tokens obliterated headline financial figures (revenue, operating income, dividends).
* **Options Considered:**
  1. Heuristic regex substitutions for letters and numbers: Highly prone to false positives and hallucinated words.
  2. Empirical, diagnostic-verified 1:1 lookup table applied *before* token filtering:
     - Ran a full-corpus diagnostic of top 25 characters in `\u0100` to `\u0FFF`.
     - Confirmed that digits 0–9 map strictly to Coptic/Greek block `U+03EC` to `U+03F5`.
     - Confirmed that `U+0355` (combining right arrowhead below) is used as comma `,`, `U+0358` (combining dot) is used as period `.`, and `U+0F24` (Tibetan digit 4) is used as rupee sign `₹`.
     - Excluded unconfirmed letters and symbols (`\u0439` for `%`, `\u010F` for `b`, `\u037E`/`\u037F` for parentheses).
* **Choice:** Empirical glyph lookup table (`ingestion/glyph_repair.py`) applied before token filtering, strictly confined to confirmed digits, commas, periods, and currency symbols.
* **Reason:** Heals 429 corrupted numerical characters across TCS without guessing letters. `"26ϳ,021"` becomes `"267,021"`, `"₹Ϯ͕ ϲϳ͕ ϬϮ1"` becomes `"₹2,67,021"`, and `"ϮϬϮϲ"` becomes `"2026"`. Financial figures are restored to grounded accuracy.
* **Trade-off:** Unconfirmed characters (such as `\u0439` where `%` sits) are not substituted without user approval, preserving conservatism.

---

### Decision 24: Lowering Anomaly Threshold to 1 with Three-State Quality Labeling

* **Context:** In Phase 2d.1, `odd_char_count_threshold` was set to 15 to avoid false positives on `"Mondelēz"` (`odd_char_count = 6`). However, this left 14 TCS pages with 1–13 corrupted characters unfiltered. Now that `glyph_repair.py` repairs digits upfront and allowlists `"Mondelēz"`, the threshold can be lowered safely.
* **Options Considered:**
  1. Keep threshold at 15: Leaves residual corrupted characters in 14 TCS pages.
  2. Lower threshold to 1: With glyph repair applied first, any remaining character in `\u0100-U+052F` is true font corruption (unless in `keep_tokens`).
* **Choice:** Lower threshold to 1 and introduce three-state quality labeling:
  - `"clean"`: No repairs performed, zero tokens dropped.
  - `"repaired"`: Corrupted digits/punctuation successfully repaired, zero tokens dropped.
  - `"degraded"`: At least one irrecoverable token dropped.
* **Reason:** Ensures zero corrupted font glyphs enter chunks in TCS, while providing full provenance tracking. Pages that merely underwent numeral repair are marked `"repaired"` rather than degraded.
* **Trade-off:** Pages with minor single-token letter corruption are now marked `"degraded"` rather than passing as clean.

---

### Decision 25: Native Python Float Vector Type Contract at Store Boundaries (and Why Fakes Hid the Bug)

* **Context:** ChromaDB validates embedding vectors by checking `isinstance(val, (int, float))` on elements. FastEmbed yields NumPy arrays with `np.float32` scalars. Converting an array with `list(vec)` creates a Python list whose elements remain `np.float32` objects, which fail Chroma's strict type validation and raise `ValueError: Expected embeddings to be a list of floats or ints...`.
* **Why Fakes Hid the Bug:** `FakeProvider` used standard Python math (`math.sin(...)`) inside a list comprehension, returning native Python `float` scalars. Unit tests relying on `FakeProvider` passed completely, masking the concrete ONNX scalar type mismatch until real end-to-end execution.
* **Options Considered:**
  1. Rely solely on the embedder to return Python floats.
  2. Rely solely on the vector store to sanitize inputs.
  3. Defense-in-depth: Specify strict `list[list[float]]` type contract on `EmbeddingProvider.embed` and enforce defensive conversion (`np.asarray(..., dtype=np.float32).tolist()`) at both the provider source and the `ChromaStore` boundary (`add_chunks` and `query`).
* **Choice:** Defense-in-depth contract enforcement.
* **Reason:** Guarantees vector store operations never fail due to scalar type discrepancies regardless of embedder backend (FastEmbed, PyTorch, NumPy, or REST APIs), while keeping failure messages concise and sanitizing raw vector dumps from logs.
* **Trade-off:** Negligible conversion overhead on batches.

---

### Decision 26: Confirmed Cyrillic Glyph Mapping for Percent Sign (`\u0439` -> `%`)

* **Context:** In `tcs-ar-26.pdf`, 3 occurrences of `\u0439` ('й', Cyrillic small letter i short) sat immediately after decimal numbers: `"4.6й"`, `"25.0й"`, `"2.8й"`.
* **Evidence:** Stated percentages match audited annual figures (e.g. fiscal 2026 revenue 267,021 vs 255,324 crore is $+4.58\% \approx 4.6\%$; operating income 66,838 / 267,021 is $25.02\% \approx 25.0\%$).
* **Diagnostic on `U+036C` ('ͬ'):** Evaluated for potential mapping to `/`. Occurred in 2 places: page 8 (`"₹110ͬ-"`) and page 173 (`"disposaůͬ redeŵption"`). Because page 173 sits between alphabetic characters rather than strictly matching `"₹<digits>ͬ-"`, `U+036C` was excluded following strict conservatism principles.
* **Choice:** Add `"\u0439": "%"` to `GLYPH_REPAIR_MAP`. Do not map `U+036C`.
* **Reason:** Heals 3 additional corrupted characters in headline financial figures across TCS, restoring `%` without guessing unconfirmed glyphs.
* **Trade-off:** None.

---

### Decision 27: Strict 1:1 Character Translation without Number Reformatting

* **Context:** In Phase 2f, a regex substitution was introduced to reformat 6-digit rupee figures into Indian numbering format (e.g. `₹107,240` -> `₹1,07,240`).
* **Options Considered:**
  1. Keep regex formatting: Modifies token structure based on assumptions.
  2. Pure 1:1 character translation: Remove number formatting regexes entirely; decode `"༤ϭϬϳ͕ ϮϰϬ"` literally to `"₹107,240"`.
* **Choice:** Pure 1:1 character translation without number reformatting.
* **Reason:** Automated data repair must restore corrupted encoding faithfully without altering original formatting. Arbitrary formatting rewrites risk corrupting financial values if inputs deviate from anticipated patterns.
* **Trade-off:** Numerical strings reflect the document's literal comma grouping rather than standardized grouping.

