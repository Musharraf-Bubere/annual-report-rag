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
