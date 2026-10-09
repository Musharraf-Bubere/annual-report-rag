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
