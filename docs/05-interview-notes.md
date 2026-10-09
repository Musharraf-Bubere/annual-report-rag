# Interview Notes

Key interview questions, concepts, and answers for **Phase 1: Skeleton, Configuration, and Provider Interface**.

---

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
- **Embedding Model (`text-embedding-004`)**: Converts text strings into fixed-length dense numerical vectors (e.g., 768 dimensions) that capture semantic meaning. It is used during both ingestion (indexing report passages) and retrieval (vectorizing user queries for cosine similarity search).
- **Generation Model (`gemini-2.5-flash`)**: An autoregressive large language model that takes a prompt containing the retrieved context passages and the user's question, and generates a natural language answer with citations. It does not search; it synthesizes and reasons over the retrieved evidence.
