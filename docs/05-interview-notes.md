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
