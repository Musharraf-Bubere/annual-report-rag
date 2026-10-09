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
