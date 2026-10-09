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
