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
* **Choice:** `google-genai` SDK, with `gemini-2.5-flash` for generation and `text-embedding-004` for embeddings, parameterized via `config/settings.yaml`.
* **Reason:** `google-genai` is the active, forward-looking SDK. `gemini-2.5-flash` provides rapid latency and strong instruction following for grounded Q&A, while `text-embedding-004` produces 768-dimensional dense vectors optimal for semantic search over technical financial terminology. Model names are isolated in configuration so they can be changed without touching code.
* **Trade-off:** External API dependency for live usage; mitigated by our offline `FakeProvider` for development and testing.
