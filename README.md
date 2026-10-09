# annual-report-rag

Question answering over company annual reports using Retrieval-Augmented Generation (RAG). The application parses complex, multi-page financial disclosures, indexes them into a vector database, and generates concise answers grounded strictly in the retrieved text with explicit page citations, accompanied by an empirical evaluation harness measuring hit rate and faithfulness.

## Quick start

Follow these steps to set up and verify the environment:

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment variables**:
   Copy `.env.example` to `.env` and insert your Gemini API key:
   - On Windows (PowerShell):
     ```powershell
     Copy-Item .env.example .env
     ```
   - On Linux/macOS:
     ```bash
     cp .env.example .env
     ```
   Then open `.env` and set `GEMINI_API_KEY=your_actual_key`.

3. **Run the smoke test**:
   Verify your API key, embeddings, and response generation:
   ```bash
   uv run python scripts/smoke_test.py
   ```

## Data

Place your source annual report PDF files inside the `data/raw/` directory (for example, `infosys-ar-26.pdf` and `tcs-ar-26.pdf`).

> [!NOTE]
> Files in `data/raw/` are intentionally gitignored and never committed to source control due to repository file size limits and corporate licensing restrictions.

### Running Document Ingestion

1. **Dry-run estimation (no API calls or costs)**:
   Parse and chunk documents to preview page counts, chunk counts, and estimated embedding API calls:
   ```bash
   uv run python scripts/ingest.py --dry-run
   ```

2. **Trial run with limited pages**:
   Test end-to-end embedding on just the first 5 pages of each PDF:
   ```bash
   uv run python scripts/ingest.py --limit-pages 5
   ```

3. **Full ingestion into Chroma vector store**:
   Idempotently embed all new chunks in batches and persist to ChromaDB:
   ```bash
   uv run python scripts/ingest.py
   ```
   Re-running this command is completely safe: existing chunk IDs are detected and skipped without repeating embedding API calls.

### Running the Search Demo

Once documents are indexed, run a manual similarity search against the vector store:
```bash
uv run python scripts/search_demo.py "What was the total revenue growth?"
```
This embeds your question with retrieval query optimization, queries Chroma using cosine similarity, and displays top-$k$ matches with page numbers, similarity scores, and excerpt previews.
