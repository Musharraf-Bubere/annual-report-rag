# Architecture

The system is organized into two primary execution pipelines—an offline **Ingestion Lane** and an online **Query Lane**—coordinated through decoupled **EmbeddingProvider** and **GenerationProvider** interfaces, alongside an **Evaluation Harness** and **Tracing** observability layer.

## Architecture Diagram (Overview)

```mermaid
flowchart TD
    subgraph IngestionLane["Ingestion Lane (Offline - Detailed)"]
        PDF["Annual Report PDFs<br/>(data/raw/*.pdf)"] --> LOAD["PyMuPDF Loader<br/>(Best-of extraction, glyph repair, token filter)"]
        LOAD --> FILTER["Page Filter<br/>(Skip if < min_page_chars)"]
        FILTER --> CHUNK["Page Chunker<br/>(chunk_size ~1000, overlap ~200<br/>Word boundary, NO cross-page merge)"]
        CHUNK --> IDS["Stable Chunk IDs<br/>({source}-p{page}-c{chunk_index})"]
        IDS --> DEDUP["Idempotency Filter<br/>(ChromaStore.existing_ids)"]
        DEDUP -->|New Chunks Only| BATCH["Batch Slicer<br/>(batch_size: 20)"]
        BATCH --> EMB_CALL["EmbeddingProvider.embed<br/>(purpose='document', L2-norm)"]
        EMB_CALL --> VSTORE["Persistent ChromaDB<br/>(hnsw:space: cosine)"]
    end

    subgraph QueryLane["Query Lane (Online)"]
        USER["User / CLI / Client"] --> API["FastAPI Service<br/>(POST /ask, GET /health, GET /sources)"]
        API --> ROUTER["Source Router<br/>(detect_sources via alias regex)"]
        ROUTER --> RETRIEVE["Retriever<br/>(Top-k per source + min_score filter)"]
        VSTORE -.->|Context chunks + PDF page N| RETRIEVE
        RETRIEVE --> PROMPT["Prompt Builder<br/>([1]..[n] headers + grounding rules)"]
        PROMPT --> LLM_CALL["GenerationProvider.generate<br/>(Gemini 3.5 Flash-Lite)"]
        LLM_CALL --> VERIFY["Citation & Number Verifier<br/>(resolve_citations + find_unsupported_numbers)"]
        VERIFY --> ANSWER["Grounded Answer + Citations"]
    end

    subgraph ProviderLayer["Provider Layer (Decoupled)"]
        EMB_PROV["EmbeddingProvider<br/>(Local FastEmbed ONNX / Gemini)"]
        GEN_PROV["GenerationProvider<br/>(GeminiProvider / FakeProvider)"]
    end

    EMB_CALL -->|calls embed| EMB_PROV
    RETRIEVE -->|embeds query| EMB_PROV
    LLM_CALL -->|calls generate| GEN_PROV
```

## Query Lane Sequence Diagram (Phase 3)

```mermaid
sequenceDiagram
    actor Client as User / CLI / HTTP Client
    participant API as FastAPI (POST /ask)
    participant Answerer as answer_question()
    participant Router as detect_sources()
    participant Retriever as retrieve()
    participant Embedder as EmbeddingProvider
    participant Store as ChromaStore
    participant Prompts as build_prompt()
    participant Generator as GenerationProvider
    participant Checks as resolve_citations() & find_unsupported_numbers()

    Client->>API: POST /ask {"question": "..."}
    API->>API: Validate question length & check store readiness
    API->>Answerer: answer_question(question, deps, top_k, sources)
    Answerer->>Router: detect_sources(question, settings)
    Router-->>Answerer: matched source stems (0, 1, or N)
    Answerer->>Retriever: retrieve(question, embedder, store, settings, sources)
    Retriever->>Embedder: embed([question], purpose="query")
    Embedder-->>Retriever: query_vector (L2-normalized float list)
    alt 0 sources detected
        Retriever->>Store: query(query_vector, top_k=6, sources=None)
    else 1 source detected
        Retriever->>Store: query(query_vector, top_k=6, sources=[src])
    else multiple sources detected
        loop each detected source
            Retriever->>Store: query(query_vector, top_k=6, sources=[src])
        end
        Retriever->>Retriever: Merge & sort by score descending
    end
    Store-->>Retriever: list[QueryResult]
    Retriever-->>Answerer: list[RetrievedChunk] (filtered by min_score)
    alt No chunks retrieved
        Answerer-->>API: AnswerResult(answer=NOT_FOUND_MESSAGE, grounded=False, reason="no_context")
    else Chunks available
        Answerer->>Prompts: build_prompt(question, chunks, settings)
        Prompts-->>Answerer: (SYSTEM_PROMPT, numbered user_prompt)
        Answerer->>Generator: generate(user_prompt, system=SYSTEM_PROMPT)
        Generator-->>Answerer: GenerationResult(text, input_tokens, output_tokens)
        Answerer->>Checks: extract_citation_numbers() & resolve_citations()
        Checks-->>Answerer: (citations, cited_chunks, invalid_citations)
        Answerer->>Checks: find_unsupported_numbers(answer, cited_chunks)
        Checks-->>Answerer: unsupported_numbers
        Answerer-->>API: AnswerResult(answer, grounded, citations, timings, tokens, ...)
    end
    API-->>Client: 200 OK AskResponse JSON
```

## Detailed System Explanation

### 1. Ingestion Lane (Phase 2 – 2f.2)
The ingestion pipeline processes raw annual report documents into searchable vector representations:
1. **Document Loading & Quality Remediation (`ingestion/loader.py`, `ingestion/glyph_repair.py`)**: Uses PyMuPDF (`fitz`) to extract text page-by-page. Repairs confirmed Coptic/Greek/Cyrillic numeral and symbol font defects upfront (`glyph_repair.py`), selects between default extraction and `TEXT_INHIBIT_SPACES` on a per-page basis (`choose_better_extraction`), filters residual corrupted tokens (`\u0100`–`\u052F`) unless allowlisted (`keep_tokens`), and tags each page as `"clean"`, `"repaired"`, or `"degraded"`.
2. **Page-Bounded Chunking (`ingestion/chunker.py`)**: Slices text into overlapping windows (default 1,000 characters, 200 overlap) at natural word boundaries without ever merging across page boundaries, preserving exact `PDF page N` provenance.
3. **Stable Identifiers & Idempotent Indexing (`ingestion/indexer.py`)**: Assigns deterministic IDs (`{source}-p{page}-c{chunk_index}`) and skips chunks already present in ChromaDB.
4. **Vector Storage (`retrieval/store.py`)**: Stores L2-normalized 768-dim vectors (`BAAI/bge-base-en-v1.5` via local FastEmbed ONNX by default) in persistent ChromaDB configured with cosine distance (`hnsw:space: cosine`) and collection metadata validation.

### 2. Query Lane (Phase 3)
1. **Source Routing (`retrieval/router.py`)**: Inspects the user's question against configured company aliases (`"tcs"`, `"tata consultancy services"`, `"infosys"`) using case-insensitive whole-word regexes (`\b...\b`).
2. **Multi-Mode Retrieval (`retrieval/retriever.py`)**: Embeds the question with `purpose="query"`. Queries ChromaDB for a single company (`top_k=6`), per company merged by score (`top_k=6` per company for comparative questions), or globally across all reports (`top_k=6` when no company is mentioned).
3. **Grounded Prompting (`generation/prompts.py`)**: Formats retrieved passages as numbered `[1]..[n]` blocks labeled with the report's `display_name`, `PDF page N`, and `text quality`, paired with a 7-rule system instruction enforcing strict grounding, bracket citations, degraded-text caution, prompt-injection resistance, and a rupee-sign (`H` $\to$ `₹`) stopgap rule.
4. **Citation & Numeric Verification (`generation/answerer.py`, `generation/checks.py`)**: Parses `[n]` markers (`[1]`, `[1][2]`, `[1, 2]`), drops out-of-range references, sets `grounded=True` only when at least one valid citation exists and the answer is not `NOT_FOUND_MESSAGE`, and runs `find_unsupported_numbers()` to flag any numeric token in the answer whose normalized digits do not appear in the cited chunks.
5. **FastAPI Service (`api/main.py`)**: Initializes the store, embedder, and generator once in a `lifespan` context manager, and exposes synchronous `def` endpoints (`GET /health`, `GET /sources`, `POST /ask`) so FastAPI runs blocking ONNX/SDK calls in its worker threadpool without blocking the asyncio event loop.
