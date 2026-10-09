# Build Log

## Phase 1: Skeleton, Configuration, and Provider Interface

### What problem this phase solves
Before building ingestion, chunking, or retrieval, any robust machine learning application needs a solid foundation. If we jump straight into writing RAG pipelines, we risk scattering hardcoded model names, raw API keys, and tight vendor couplings throughout the codebase. Later, when an API key rotates or a model deprecates, debugging becomes a nightmare, and writing automated tests requires messy network mocking.

Phase 1 solves this by creating a clean project skeleton, isolating configuration and secrets, and establishing a unified provider interface so the rest of the application never depends on a specific AI vendor or requires live internet access during testing.

### What we built
1. **Modern Project Skeleton**: Initialized the project with `uv` on Python 3.12 using a clean `src/` layout (`src/annual_report_rag/`) with dedicated subpackages for `providers`, `ingestion`, `retrieval`, `generation`, `api`, and `evaluation`.
2. **Strict Git & Security Hygiene**: Added `.gitignore` rules to guarantee that sensitive `.env` files, raw financial filings (`data/raw/`), vector store data, and cache directories never leak into version control. Provided `.env.example` as a safe template.
3. **Externalized & Typed Configuration**: Implemented `config/settings.yaml` and `src/annual_report_rag/config.py`. All operational knobs (model names, chunk sizes, top-$k$, directory paths) live in YAML, while secrets reside in `.env`, parsed into immutable typed dataclasses (`Settings`).
4. **Decoupled Provider Interface**: Created `BaseProvider` abstract base class defining `embed()` and `generate()`. Implemented `GeminiProvider` using the official `google-genai` SDK and `FakeProvider` for fast in-memory offline testing.
5. **Provider Factory**: Implemented `get_provider(settings)` in `providers/factory.py`, enabling zero-code provider switching via configuration.
6. **Verification Scripts & Tests**: Added `scripts/smoke_test.py` to verify API keys against real Gemini endpoints, and wrote offline unit tests in `tests/` covering configuration parsing, error handling, and provider behaviors without making network calls.

### How it works in plain English
When the application starts, `load_settings()` reads `config/settings.yaml` for system tuning parameters and reads `.env` for secrets like `GEMINI_API_KEY`. It bundles these into a typed Python object (`Settings`) where every field has an explicit type hint.

When a component needs to generate embeddings or answer a query, it calls `get_provider(settings)`. The factory inspects the configuration: if it says `"gemini"`, it spins up `GeminiProvider` with the API key and Google's recommended models (`text-embedding-004` and `gemini-2.5-flash`). If running tests, the configuration specifies `"fake"`, returning `FakeProvider`, which produces deterministic vector shapes and mock answers instantly in memory.

Because application modules only know about `BaseProvider.embed` and `BaseProvider.generate`, the rest of the RAG system never knows or cares whether it is talking to Google Gemini, an offline test double, or another LLM provider in the future.

### What could go wrong
* **Missing or Invalid API Key**: If a developer forgets to copy `.env.example` to `.env` or enters an invalid `GEMINI_API_KEY`, `GeminiProvider` raises a clear, actionable `ValueError` at initialization rather than failing midway through a long pipeline.
* **Model Name Deprecation**: LLM vendors frequently update and deprecate model IDs. If a model name is hardcoded in Python files, fixing it requires code edits and redeployments. Because model names live exclusively in `config/settings.yaml`, updating a model requires only a one-line config change.
* **Accidental Secret Commits**: Developers might accidentally commit `.env` or raw company files. We mitigate this with strict `.gitignore` rules and mandatory pre-push checks (`git status`, `git ls-files`, string scans) before any code is pushed to GitHub.

---

## Phase 2: Ingestion Pipeline (PDFs -> Chunks -> Embeddings -> Chroma)

### What problem this phase solves
Corporate annual reports are 300+ page behemoths filled with dense financial disclosures, auditing opinions, and narrative footnotes. Naively dumping entire reports into an LLM exceeds context limits, costs significant money, and triggers hallucinations.

Furthermore, financial question answering requires exact traceability: when a model quotes a profit figure, the analyst must be able to verify the exact page in the report. If chunking indiscriminately mashes text across page boundaries, citations become unreliable.

Phase 2 builds an offline, deterministic ingestion engine that extracts clean text from PDFs, breaks it into structured chunks with strict page-level isolation, computes L2-normalized semantic embeddings using Google's retrieval task types, and stores them in persistent ChromaDB storage—built to be completely idempotent so re-runs never duplicate vectors or burn unnecessary API credits.

### What we built
1. **Asymmetric Embedding Contract**: Updated `BaseProvider.embed(texts, purpose="document")` to distinguish between indexing documents (`RETRIEVAL_DOCUMENT`) and searching queries (`RETRIEVAL_QUERY`). Added L2 unit vector normalization and automatic exponential backoff retry for HTTP 429 and 5xx errors.
2. **Lightning-Fast PDF Text Extraction (`ingestion/loader.py`)**: Built a loader using PyMuPDF (`fitz`) that normalizes irregular whitespace, extracts clean page text, filters out blank or cover pages below `min_page_chars` (default 50), and warns on scanned image documents.
3. **Word-Boundary, Page-Bounded Chunker (`ingestion/chunker.py`)**: Designed a chunker that windows text into ~1,000 character slices with 200 character overlap without cutting words in half. It strictly guarantees that **no chunk ever blends text across pages**, and assigns stable deterministic IDs (`{source}-p{page}-c{chunk_index}`).
4. **Persistent Vector Store Wrapper (`retrieval/store.py`)**: Wrapped ChromaDB with a persistent client and cosine metric (`hnsw:space: cosine`). Uses our own pre-computed embeddings rather than Chroma's built-in models, and strictly isolates all ChromaDB imports away from application logic.
5. **Idempotent Indexing Orchestrator (`ingestion/indexer.py`)**: Coordinates loading, chunking, querying existing IDs from Chroma, batch-embedding only new chunks, and inserting into vector storage. Re-running the script skips previously indexed chunks without making repeated API calls.
6. **Command-Line Tools (`scripts/ingest.py` & `scripts/search_demo.py`)**:
   - `scripts/ingest.py`: Supports `--dry-run` to inspect page/chunk counts and estimate API call volume without spending money, as well as `--limit-pages N` for cheap test runs.
   - `scripts/search_demo.py`: Enables immediate manual query testing against indexed vector representations.

### How it works in plain English
When you run `ingest.py`, PyMuPDF opens each PDF in `data/raw/` and reads every page. It cleans up multiple spaces and newline breaks. Blank cover pages (fewer than 50 characters) are discarded with a notification.

Next, each page's text is divided into manageable passages of roughly 1,000 characters. If a 1,000-character boundary falls in the middle of a word like `"depreciation"`, the chunker finds the adjacent space and splits cleanly there. It moves forward with a 200-character overlap to preserve semantic flow. Each chunk receives an address, such as `infosys-ar-26-p42-c0`.

The indexer then checks ChromaDB: *"Do you already have `infosys-ar-26-p42-c0`?"* If yes, the chunk is skipped. If no, new chunks are batched into groups of 50 and sent to Google Gemini with `task_type="RETRIEVAL_DOCUMENT"`. The returned vectors are normalized so their lengths equal 1.0, and they are stored in ChromaDB along with the chunk's text and page metadata.

When searching via `search_demo.py`, your question is embedded with `task_type="RETRIEVAL_QUERY"`, and Chroma calculates cosine distances to return the top-$k$ most similar chunks in seconds.

### Known Limitations & Failure Modes
* **Multi-Column & Nested Tables**: Financial reports contain complex balance sheet tables. PyMuPDF extracts text in reading-order blocks, which can linearize multi-column tables into flattened text. Specialized tabular extractors or vision models will be needed if queries require complex columnar cross-tabulations.
* **Scanned or Image-Only PDFs**: The current pipeline relies on born-digital text streams. Image-only PDFs (scanned documents) contain zero extractable text. The loader prints a clear warning, but OCR processing is intentionally out of scope for this phase.
* **Facts Split Across Page Boundaries**: Because chunks are strictly bounded by individual pages to guarantee 100% accurate page citations, a multi-sentence explanation that begins on page 10 and concludes on page 11 will be split across two separate chunks. The query retriever must retrieve both chunks (via top-$k$) to assemble the complete context.
* **Physical PDF Page Index vs. Printed Page Number**: PDF page indexes (`doc[idx]` where `page = idx + 1`) represent the absolute physical sheet in the PDF viewer. Annual reports typically feature cover pages, tables of contents, and Roman-numeral prefaces (e.g. pages i–iv), meaning physical PDF page 42 might correspond to printed report page number 38.

---

## Phase 2b: Free-Tier Rate Limiting and Text Quality Audit

### What problem this phase solves
During our initial trial run attempting to index the real annual report PDFs, two severe operational realities emerged:
1. **API Rate-Limit Crashes**: Google Gemini free-tier accounts enforce a ceiling of approximately 100 requests / items per minute. Dispatching large 50-chunk batches at maximum network speed quickly triggered HTTP 429 `RESOURCE_EXHAUSTED` with server messages stating `"Please retry in 27s"`. Our initial 3-attempt exponential backoff gave up after ~7 seconds, terminating ingestion prematurely.
2. **Extracted Text Corruption**: Inspecting extracted passages revealed that certain styled pages extracted as garbled text. In `tcs-ar-26.pdf` (notably page 8, the Chairman's letter), words were severely spaced out (`"i n th i s b ack d rop"`) and numbers/letters were corrupted (`"ĐonƐolŝdated ƌevenue oĨ ₹ 26ϳ,021 Đƌoƌe"`). In a financial RAG system, corrupted numbers invalidate the entire premise of grounded fact retrieval.

Crucially, **we stopped before proceeding with full ingestion**. If we had blindly pushed ahead, we would have burned hours of API quota embedding corrupted tokens that would permanently pollute ChromaDB with hallucination-inducing noise.

### What we built
1. **Sliding-Window Rate Limiter (`providers/rate_limiter.py`)**: A client-side throughput regulator tracking timestamped items over a 60-second sliding window. It transparently paces embedding calls to 80 items/minute (safely below the 100/min quota) with smaller batch sizes (20 items). Built with injectable clock and sleep functions for sub-millisecond offline unit testing.
2. **Delay-Aware 429 Retry Loop (`providers/gemini.py`)**: On encountering HTTP 429, the provider parses Google's exact recommended wait interval (`"retry in Ns"`), waits that duration plus 1 second, and retries up to 5 times while printing transparent wait notifications. Standard exponential backoff is preserved for 5xx server drops.
3. **Pausable, Resumable Progress & Collection Reset**: `indexer.py` reports real-time batch progress (`Embedded 300/3118 chunks`). `scripts/ingest.py` dry-run mode now prints estimated embedding time (~39 minutes for 3,118 chunks at 80 items/min), and includes a `--reset` flag to wipe ChromaDB before re-indexing.
4. **Explainable Text Quality Metrics (`ingestion/quality.py`)**: Implemented simple, explainable mathematical metrics per page:
   - `single_letter_ratio`: proportion of alphabetic tokens of length 1 (captures spaced-out words like `"i n th i s"`).
   - `odd_char_ratio`: proportion of characters outside standard ASCII/punctuation/symbols (captures broken font glyph encodings).
   - `char_count`: total characters on the page.
5. **Automated Document Audit Tool (`scripts/audit_text_quality.py`)**: Audits every page of all raw PDFs, calculates median and 95th percentile distributions, flags pages exceeding thresholds (single-letter > 0.15, odd-char > 0.05), surfaces the 10 worst pages with excerpts, and exports a complete CSV report to `data/processed/text_quality_report.csv`.

### What the audit revealed
Running the audit across both annual reports (743 pages total) provided clear diagnostic clarity:
* **`infosys-ar-26.pdf` (383 pages)**: Clean. Median single-letter ratio is 1.9% (P95: 8.5%). Zero pages exceeded the odd-character threshold. Only 4 pages exceeded the single-letter threshold, all consisting of corporate subsidiary tables with abbreviations (e.g. `ANZ Pty Ltd. (9)`).
* **`tcs-ar-26.pdf` (360 pages)**: Heavily affected by font-tracking defects. 31 pages exceeded the single-letter ratio threshold, with styled executive letters reaching up to 46% single-character tokens (pages 8, 9, 22, 25, 28). Numbers were fractured into pieces (`"584, 51 9"` and `"44, 0 0 0 +"`).

Having quantified and cataloged the exact pages and failure modes into `data/processed/text_quality_report.csv`, we can address extraction repairs with targeted verification before doing any real embedding runs.

---

## Phase 2c: Audit Metrics Fix and Extractor Comparison Experiment

### What problem this phase solves
Our Phase 2b quality audit successfully surfaced severe letter-spacing and split numbers in `tcs-ar-26.pdf`. However, a deeper inspection exposed two critical shortcomings in our quality metrics and an unresolved extraction problem:
1. **Long-Page Defect Dilution**: TCS Page 8 (the Chairman's letter) contains 41 corrupted glyphs (`"ĐonƐolŝdated ƌevenue oĨ ₹ 26ϳ,021 Đƌoƌe"`), but its total character count was ~9,300. As a result, its `odd_char_ratio` was only $41 / 9300 \approx 0.0044$, slipping far beneath the Phase 2b threshold of 0.05.
2. **Missing Split Number Metric**: Key financial figures (e.g. `"584, 51 9"`, `"44, 0 0 0 +"`, `"2 0 2 6"`) had digits separated by spaces. The single-letter alphabetic check only looked at letters, completely missing numeric fragmentation.
3. **Extractor Blindness**: We did not know whether PyMuPDF was fundamentally broken or if other open-source extractors (`pypdf`, `pdfminer.six`) could extract these pages cleanly.
4. **Vector Store Vulnerability**: If a user switches embedding models (e.g. from `gemini-embedding-001` with 768 dims to another model with 1536 dims), ChromaDB could accept mixed vectors or crash cryptically without warning.

Phase 2c solves this with zero API calls: we fix the audit metrics, conduct an empirical comparison across 5 extraction candidates on all 182 flagged pages, and add strict metadata and dimensionality guards to `GeminiProvider` and `ChromaStore`.

### What we built
1. **Dual Anomaly Metrics (`ingestion/quality.py`)**:
   - `odd_char_count`: Absolute count of non-standard characters, flagging pages with $\ge 15$ odd characters regardless of total page length. Lowered default `odd_char_ratio` threshold to 0.002. (TCS Page 8 is now immediately caught by `odd_char_count = 41 >= 15` alone).
   - `number_split_count`: Detects sequences where single digits are separated by single spaces (`\b\d(?: \d)+\b`), catching `"2 0 2 6"` and `"44, 0 0 0"` while scoring 0 on clean numbers (`"2026"`, `"44,000"`, `"584,519"`).
2. **Offline Extractor Benchmark Suite (`scripts/compare_extractors.py`)**:
   Implemented five candidate extractors in modular functions and benchmarked them against all 182 flagged pages:
   - Candidate 1: PyMuPDF default
   - Candidate 2: PyMuPDF with `TEXT_INHIBIT_SPACES` flag
   - Candidate 3: PyMuPDF word tuples re-joined with horizontal gap tolerance (2.0 pts)
   - Candidate 4: `pypdf`
   - Candidate 5: `pdfminer.six` with custom `LAParams` (word_margin=0.1, char_margin=2.0)
3. **Side-by-Side Excerpt Inspector**: Evaluated excerpts across worst pages (TCS 8, 22, 25), testing whether the exact string `"267,021"` could be extracted by any tool.
4. **Vector Store and Provider Integrity Protections**:
   - In `store.py`: Embedded model name and dimension are saved into Chroma collection metadata. Reopening an existing collection with mismatched settings immediately raises a `ValueError` prompting the user to run `ingest.py --reset`.
   - In `gemini.py`: Response embedding vector counts and individual vector dimensions are validated before acceptance.

### How it works in plain English & Benchmark Results
When `compare_extractors.py` runs, it loads the 182 flagged pages from `text_quality_report.csv` and tests each candidate:

```
Candidate                                  | Cleaned    | Med Single | Med Odd  | Med Split | Time (s)
------------------------------------------------------------------------------------------
PyMuPDF (default)                          | 0/182      | 0.0674     | 0        | 4         | 0.73    
PyMuPDF (TEXT_INHIBIT_SPACES)              | 79/182     | 0.0344     | 0        | 1         | 0.69    
PyMuPDF (words rejoined, gap=2.0)          | 57/182     | 0.0463     | 0        | 3         | 0.76    
pypdf                                      | 9/182      | 0.0670     | 0        | 4         | 12.21   
pdfminer.six (wm=0.1, cm=2.0)              | 30/182     | 0.0670     | 0        | 5         | 46.46   
```

Key discoveries:
* **The Winner**: `PyMuPDF (TEXT_INHIBIT_SPACES)` cleaned **79 of 182** pages, cut the median single-letter ratio in half (from 0.0674 down to 0.0344), and repaired broken numbers (`"584,519"`, `"44,000+"`, `"2026"`)—all in **0.69 seconds** (67x faster than `pdfminer.six`).
* **The `"267,021"` Mystery on Page 8**: Exact `"267,021"` appeared in **NONE** of the five extractors. Why? In `tcs-ar-26.pdf`, the heading font (`HouschkaRounded-DemiBold`) contains an erroneous ToUnicode CMap table embedded by the document designer: the glyph representing `7` is mapped to Unicode character `0x03F3` (`ϳ`, Greek Small Letter Yot). All text extractors faithfully read the embedded font table and produce `26ϳ,021` (or `cid:1011` in pdfminer). No text extractor can fix this without OCR or domain-specific character replacement heuristics.

### What could go wrong
* **Residual Table False Positives**: In dense financial tables where single digits appear in adjacent cells (e.g. Infosys subsidiary notes), whitespace cleaning converts table newlines into spaces, triggering split number warnings (`num_split=3-5`). These are false positives caused by table structure rather than broken words.
* **Corrupted Font CMaps**: When a PDF creator embeds a damaged or incomplete ToUnicode CMap, characters extract as wrong Unicode code points. Standard text extractors cannot deduce the visual glyph shape without an OCR engine like Tesseract.
