# Interview Notes

Key interview questions, concepts, and answers across project phases.

---

## Phase 1: Skeleton, Configuration, and Provider Interface

### Question 1: Why did you introduce an abstract `BaseProvider` instead of importing the Google GenAI SDK directly in your ingestion and query code?
**Answer:**
Directly importing vendor SDKs across the codebase tightly couples business logic to a single third-party provider (vendor lock-in) and violates the Dependency Inversion Principle (the "D" in SOLID). By defining `BaseProvider` with minimal contracts (`embed` and `generate`), our core ingestion and retrieval pipelines depend only on an abstraction. This makes swapping providers a one-line config change and allows us to use an in-memory `FakeProvider` for instantaneous, deterministic unit tests without network calls or API credit usage.

---

### Question 2: How do you handle configuration and secrets, and why avoid hardcoding model names in code?
**Answer:**
We separate operational parameters from secrets following Twelve-Factor App principles. Non-sensitive settings (model names, chunk sizes, top-$k$, directory paths) live in `config/settings.yaml`, while credentials (`GEMINI_API_KEY`) live in `.env` (strictly git-ignored). At runtime, `config.py` loads both into an immutable, typed `Settings` dataclass. Model names are never hardcoded because AI vendors regularly deprecate model identifiers; keeping them in configuration allows zero-downtime, zero-code-change updates.

---

### Question 3: Why choose `uv` over standard `pip` and `requirements.txt`?
**Answer:**
`uv` is an extremely fast Python package manager written in Rust. It provides:
1. **Deterministic Lockfiles**: `uv.lock` records exact transitive dependency hashes, ensuring identical builds across development and production environments.
2. **Unified Toolchain**: It manages virtual environments, Python runtime versions (`.python-version`), and package execution (`uv run`) in one unified CLI.
3. **Speed**: Resolves and installs packages up to 10-100x faster than traditional `pip`, drastically accelerating development feedback loops and CI runs.

---

### Question 4: How do you ensure automated tests don't make external network requests or fail when an API key is not set?
**Answer:**
We implement a `FakeProvider` that inherits from `BaseProvider`. It generates deterministic pseudo-embeddings based on text length and returns mock responses in memory. Unit tests instantiate either `FakeProvider` directly or pass `provider: {name: "fake"}` to `get_provider(settings)`. This guarantees that our test suite runs in under two seconds, is completely offline, costs zero dollars, and reliably runs in isolated CI environments where external secrets should not be exposed.

---

### Question 5: What is the architectural role difference between the embedding model and the generation model?
**Answer:**
- **Embedding Model (`gemini-embedding-001` / `text-embedding-004`)**: Converts text strings into fixed-length dense numerical vectors (e.g., 768 dimensions) that capture semantic meaning. It is used during both ingestion (indexing report passages) and retrieval (vectorizing user queries for cosine similarity search).
- **Generation Model (`gemini-3.5-flash-lite`)**: An autoregressive large language model that takes a prompt containing the retrieved context passages and the user's question, and generates a natural language answer with citations. It does not search; it synthesizes and reasons over the retrieved evidence.

---

## Phase 2: Ingestion Pipeline (PDFs -> Chunks -> Embeddings -> Chroma)

### Question 6: Why did you enforce strict page-bounded chunking instead of allowing chunks to span across page boundaries?
**Answer:**
In annual report RAG, answer grounding and auditability depend on accurate page citations. If a chunk spans the boundary between page 42 and page 43, it is impossible to determine with certainty whether a quoted figure originated on page 42 or page 43 without re-parsing the document. Bounding chunks strictly to individual pages ensures 100% provenance integrity: every chunk maps unambiguously to a single page number. The slight trade-off—facts split across page boundaries—is handled naturally by retrieving multiple adjacent chunks via top-$k$.

---

### Question 7: Why is L2 normalization applied to embeddings before indexing into ChromaDB with cosine distance?
**Answer:**
Cosine distance measures the angle between two vectors rather than their magnitude:
$$\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
By L2-normalizing vectors upfront ($\|u\|_2 = 1$), the cosine similarity simplifies to a pure dot product ($u \cdot v$), making similarity computation and distance indexing much faster and mathematically stable across vector stores. It also prevents passages with larger word counts from skewing Euclidean vector magnitudes.

---

### Question 8: What is the purpose of asymmetric embedding task types (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`)?
**Answer:**
In information retrieval, user queries and target document passages have fundamentally different structures: queries are short, interrogative phrases (e.g. "What was FY23 revenue?"), whereas document chunks are long, factual paragraphs full of corporate jargon. Asymmetric embedding models use different projection heads for documents and queries to map these two different semantic forms into a shared vector space where semantic relevance maximizes inner-product similarity.

---

### Question 9: How does your pipeline ensure idempotency and prevent duplicate embeddings?
**Answer:**
We construct deterministic coordinate chunk IDs formatted as `{source}-p{page}-c{chunk_index}` (e.g. `infosys-ar-26-p42-c1`). Before calling embedding APIs, `indexer.py` queries ChromaDB using `store.existing_ids(all_ids)`. Any chunk ID already present in the collection is excluded from the embedding queue. This guarantees that re-running ingestion costs zero additional API calls, generates zero duplicate database records, and completes in milliseconds.

---

### Question 10: How does your chunking algorithm ensure words and numbers are never cut in half?
**Answer:**
Rather than slicing naively on fixed character indices, our chunker identifies the target boundary (`start + chunk_size`) and then scans backwards to locate the nearest whitespace character (`rfind(' ')`). It slices at that space boundary. When computing the start of the next overlapping chunk (`end - chunk_overlap`), it similarly snaps to the next available word boundary. This guarantees that critical financial numbers (e.g. `$27,950M`) and technical terms are never split into meaningless fragments across chunks.

---

## Phase 2b: Rate Limiting, Retries, and Document Quality

### Question 11: How do you handle cloud provider rate limits (such as HTTP 429) during batch document ingestion?
**Answer:**
We employ a two-layer strategy:
1. **Proactive Throughput Regulation**: A client-side sliding-window rate limiter (`SlidingWindowRateLimiter`) measures item throughput over the previous 60 seconds and pauses execution before requests violate provider ceilings (pacing to 80 items/min for a 100/min quota).
2. **Reactive Delay-Aware Retries**: If HTTP 429 `RESOURCE_EXHAUSTED` occurs due to transient backend congestion, the provider parses the server's exact backoff directive (e.g. `"Please retry in 27s"`), pauses for that duration plus a 1-second safety buffer, and retries up to 5 times.

### Question 12: Why combine proactive rate limiting with reactive retry logic rather than relying on retries alone?
**Answer:**
Relying solely on reactive retries forces the system to repeatedly hit rate limit walls. When cloud providers issue HTTP 429, they often enforce mandatory cooldown penalties (e.g. 27–60 seconds). Rapidly blasting requests until failure wastes TCP connections, triggers quota tripwires, and degrades total throughput. Proactive pacing throttles calls just beneath the ceiling, keeping processing smooth and steady, while reactive retries act as a resilient safety net for unexpected quota drops.

### Question 13: Why is an offline text quality audit critical before vector indexing in a RAG pipeline?
**Answer:**
"Garbage in, vector store out, hallucinations generated." In financial RAG, models must answer specific questions regarding operating margins, headcount, and liabilities. If PDF extraction produces spaced-out text (e.g. `"584, 51 9"` instead of `584,519`) or broken character encodings (`"ĐonƐolŝdated"`), the embedding model receives fragmented subwords that destroy semantic similarity matching. Vectorizing corrupted text wastes API credits, pollutes vector indexes, and guarantees downstream hallucination. Performing an automated audit with explainable metrics (`single_letter_ratio`, `odd_char_ratio`) catches and localizes extraction defects before expensive indexing begins.

### Question 14: How can rate-limited batch ingestion pipelines be designed to support crash recovery and resumption?
**Answer:**
By ensuring strict idempotency at every batch insertion:
1. Deterministic coordinate chunk IDs (`{source}-p{page}-c{chunk_index}`) are generated upfront.
2. Before dispatching embedding calls, the pipeline queries the vector store (`existing_ids`) and removes already indexed chunks from the processing queue.
3. Chunks are committed to the persistent vector store immediately after each batch call.
If the pipeline is terminated halfway through a 3,000-chunk ingest, re-running the script automatically skips the completed batches in milliseconds and seamlessly resumes from the exact chunk where it left off.

---

## Phase 2c: Audit Metrics Fix and Extractor Comparison Experiment

### Question 15: Why is PDF text extraction notoriously hard, and how do font encodings and character tracking lead to corrupted text or spaced-out letters?
**Answer:**
PDF is a visual presentation format, not a structured text document.
1. **Space Insertion Heuristics**: PDFs do not store ASCII space characters (`" "`) between words; they place glyphs at physical $(x, y)$ coordinates. Text extractors must deduce where spaces belong by checking if the horizontal gap between consecutive characters exceeds an advance threshold. When designers use tight letter-tracking or custom fonts, extractors (like PyMuPDF by default) misinterpret font advance gaps as spaces, splitting words into single letters (`"F ortune M ag az i ne"`) and fracturing numbers (`"584, 51 9"`). PyMuPDF's `TEXT_INHIBIT_SPACES` flag disables this heuristic.
2. **Font Encodings & CMap Tables**: PDFs map glyph IDs to characters using `/Encoding` and `/ToUnicode` mapping tables. If a document designer uses a non-standard or corrupted font subset, glyph IDs map to arbitrary Unicode characters (e.g. `0x03F3` instead of ASCII `7`). The extractor faithfully returns the mapped character without knowing it looks like a `7` visually.

---

### Question 16: How do you evaluate and benchmark text extraction quality offline on unlabelled document corpora?
**Answer:**
Without manual ground-truth transcriptions, we use statistical rule-based heuristics that measure deviations from natural language:
1. **`single_letter_ratio`**: In normal English, words are several characters long; single letters account for $< 3\%$ of tokens (`"a"`, `"I"`). When tracking issues fragment words, this ratio spikes to $20\%-50\%$.
2. **`odd_char_count` & `odd_char_ratio`**: Measures characters outside standard ASCII, whitespace, and financial currency symbols ($\$$, $€$, ₹). High counts capture broken font glyph encodings.
3. **`number_split_count`**: Detects runs of single digits separated by spaces (`\b\d(?: \d)+\b`), capturing fractured numbers (`"2 0 2 6"`, `"44, 0 0 0"`).
By running these metrics across multiple candidate extractors, we can objectively measure the percentage of flagged pages cleaned, median metric reductions, and execution latency.

---

### Question 17: Why did `"267,021"` fail to extract as ASCII `"267,021"` across all five PDF extractors on TCS Page 8?
**Answer:**
The font used in that heading (`HouschkaRounded-DemiBold`) has an internal ToUnicode CMap table where the glyph for digit `7` was assigned character code `0x03F3` (`ϳ`, Greek Small Letter Yot). Every standard PDF parser (`PyMuPDF`, `pypdf`, `pdfminer.six`) parses the font stream according to the official PDF specification and reads the character code mapped in the ToUnicode table. Because all extractors correctly read `0x03F3` rather than ASCII `0x0037`, none of them can output `"267,021"` without visual OCR (e.g. Tesseract) or custom heuristic glyph replacement tables.

---

### Question 18: Why is storing embedding model name and dimension metadata in ChromaDB essential for production RAG systems?
**Answer:**
Vector embeddings from different models inhabit completely incompatible geometric manifolds. Even if two models share the same output dimension (e.g. 768), a vector from `gemini-embedding-001` cannot be compared via cosine similarity against a vector from `text-embedding-004` or an open-source model. If a collection lacks metadata tracking, changing model settings in `settings.yaml` silently allows the database to accept mixed vectors, destroying retrieval quality and hallucinating answers. Storing `embedding_model` and `embedding_dimension` in collection metadata enables fail-fast validation on startup, forcing a clean re-index with `--reset`.

---

## Phase 2d: Safe Production Extraction Fix, Best-of Selection, and Corrupted Token Filtering

### Question 19: Why can't a global PDF extraction flag like `TEXT_INHIBIT_SPACES` be enabled unconditionally across an entire document corpus?
**Answer:**
PDFs do not store space characters; they place glyphs at $(x, y)$ coordinate positions. When `TEXT_INHIBIT_SPACES` is enabled, the PDF parser disables heuristic space insertion based on character advances. In documents where text flows across standard character widths, this prevents spurious space insertion (healing `"F ortune M ag az i ne"` into `"Fortune Magazine"`).
However, when a document uses visual positioning or loose tracking instead of explicit spaces, suppressing heuristic spaces causes legitimate, distinct words to be glued together into giant tokens (e.g. `operatingprofit` or tokens $\ge 25$ characters).
In our regression audit across 743 pages, global inhibit spaces broke **12 previously clean pages** in TCS. A production system must therefore employ a **per-page best-of selection**: test default extraction first, and only activate inhibit mode if defect thresholds are triggered and the inhibit extraction proves mathematically superior without introducing glued tokens.

---

### Question 20: How should a production RAG ingestion pipeline handle irrecoverable font CMap corruption without guessing or hallucinating data?
**Answer:**
When a PDF embedding font has a damaged ToUnicode CMap (e.g. mapping visual digit `7` to Greek Yot `ϳ`, or `C` to Latin D-with-stroke `Đ`), the true visual character is physically absent from the text stream.
In high-stakes financial RAG, there are two common mistakes:
1. **Blindly ingesting it**: Corrupted Unicode tokens enter the vector store, degrading semantic similarity queries and causing retrieval failures.
2. **Hallucinating character substitutions**: Using ad-hoc regex heuristics to guess character replacements (e.g. assuming `ϳ` is `7`) risks corrupting balance sheet figures or legal names if the assumption is wrong.

The disciplined engineering choice is:
1. **Filter corrupted tokens**: Drop whitespace-delimited tokens containing characters in the corrupted Unicode range (`\u0100` to `\u052F`). This preserves clean surrounding text (e.g. entire executive letters) while ensuring zero corrupted tokens enter vector chunks.
2. **Explicit metadata tagging**: Tag affected pages as `text_quality = "degraded"` and log exact page numbers, dropped token counts, and odd character counts to an audit CSV (`degraded_pages.csv`).
3. **Downstream transparency**: Expose the `text_quality` flag in retrieval chunks and UI citations so that the LLM and end user know the page underwent lossy filtering, and queue those specific pages for visual OCR remediation.

---

### Question 21: How do you enforce data quality invariants before spending time and quota on vector embeddings?
**Answer:**
Embedding large document corpora against external APIs involves substantial rate limits, wall-clock time, and API cost. Incurring these costs on unverified data is a major engineering anti-pattern.
A robust ingestion architecture enforces quality invariants through:
1. **Deterministic dry-run mode (`--dry-run`)**: Executes the entire loading, cleaning, filtering, and chunking pipeline offline, reporting page methods, degraded counts, chunk counts, and estimated embedding calls/duration.
2. **Pre-embedding invariant assertions**: The dry-run scans all generated chunks against quality invariants—for example, asserting that exactly **0 chunks** contain characters in the corrupted Unicode range before a single API call is allowed.
3. **Idempotent checkpointing**: By computing deterministic chunk IDs (`{source}-p{page}-c{chunk_index}`) and checking vector store presence before each batch, runs can be safely interrupted and resumed without duplicating work or embeddings.

---

### Question 22: How do you identify and mitigate false positives in automated text cleaning filters?
**Answer:**
Heuristic cleaning rules (such as dropping tokens based on Unicode blocks) frequently produce false positives on legitimate domain vocabulary: multi-national brand names, foreign executive names, or loanwords often contain non-ASCII characters (e.g. `"Mondelēz"` with `ē` `U+0113`). A naive filter destroys valid entity names, harming downstream question-answering.
A production RAG pipeline prevents false positives through a three-layer defense:
1. **Offline Diagnostic Auditing**: Run a diagnostic script (`diagnose_dropped_tokens.py`) that exports every dropped token and its surrounding page metrics to an inspection CSV (`dropped_tokens_report.csv`). This immediately surfaces legitimate words like `"Mondelēz"` before changes are deployed.
2. **Page-Level Anomaly Gating**: Do not filter tokens unconditionally. Gate token filtering on page-level defect density (e.g. `odd_char_count >= 15`). An isolated foreign name on a clean page has a low score (e.g. 6 odd characters) and is left untouched, eliminating false positives on 99% of pages.
3. **Configurable Token Allowlisting (`keep_tokens`)**: Maintain an explicit list in configuration for verified legitimate entities. Even on a heavily degraded page that exceeds the defect threshold, allowlisted tokens are strictly preserved.
4. **Informative Breakdown Reporting**: Rather than asserting an inflexible zero-character invariant across all chunks, report the exact chunks, pages, and Unicode characters retained, giving full transparency to engineering teams.
