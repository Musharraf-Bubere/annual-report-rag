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

1. **Audit document text quality (offline, zero API costs)**:
   Audit raw PDFs for text corruption (broken fonts, spaced-out words) and export a report:
   ```bash
   uv run python scripts/audit_text_quality.py
   ```

2. **Benchmark PDF extractors on flagged pages (offline, zero API costs)**:
   Compare 5 extraction candidates (`PyMuPDF default`, `PyMuPDF TEXT_INHIBIT_SPACES`, `PyMuPDF words rejoined`, `pypdf`, `pdfminer.six`) on flagged defective pages:
   ```bash
   uv run python scripts/compare_extractors.py
   ```

3. **Dry-run estimation (no API calls or costs)**:
   Parse and chunk documents to preview page counts, chunk counts, estimated embedding calls, and estimated runtime:
   ```bash
   uv run python scripts/ingest.py --dry-run
   ```

4. **Trial run with limited pages**:
   Test end-to-end embedding on just the first 5 pages of each PDF:
   ```bash
   uv run python scripts/ingest.py --limit-pages 5
   ```

5. **Full ingestion into Chroma vector store**:
   Idempotently embed all new chunks in rate-paced batches (default 20 chunks, max 80 items/min) and persist to ChromaDB:
   ```bash
   uv run python scripts/ingest.py
   ```
   > [!TIP]
   > On the Google Gemini free tier (capped at 100 requests/minute), full ingestion of ~3,100 chunks takes approximately **35–45 minutes**. You can interrupt the run at any point (Ctrl+C) and resume later; previously indexed chunks are detected and skipped in milliseconds without repeating API calls. To wipe and rebuild the index from scratch, pass the `--reset` flag.

### Running the Search Demo

Once documents are indexed, run a manual similarity search against the vector store:
```bash
uv run python scripts/search_demo.py "What was the total revenue growth?"
```
This embeds your question with retrieval query optimization, queries Chroma using cosine similarity, and displays top-$k$ matches with page numbers, similarity scores, extraction methods, quality status, and excerpt previews.

## Known data quality issues

Corporate annual reports frequently employ stylized typography, custom subsetted fonts, and complex multi-column layouts that introduce extraction defects. Our ingestion pipeline includes an automated quality auditor (`scripts/audit_text_quality.py`) and a regression safety harness (`scripts/check_inhibit_regression.py`). Current corpus metrics across 743 examined pages (741 loaded, 2 empty skipped in Infosys):

| Metric | Infosys (`infosys-ar-26.pdf`) | TCS (`tcs-ar-26.pdf`) | Total Corpus |
| :--- | :--- | :--- | :--- |
| **Total Pages Examined** | 383 | 360 | 743 |
| **Pages Loaded ($\ge 50$ chars)** | 381 | 360 | 741 |
| **Default Extraction Pages** | 381 (100%) | 325 (90.3%) | 706 (95.3%) |
| **Inhibit Spaces Pages** | 0 (0.0%) | 35 (9.7%) | 35 (4.7%) |
| **Degraded Quality Pages** | 0 (0.0%) | 17 (4.7%) | 17 (2.3%) |
| **Total Chunks Generated** | 1,684 | 1,425 | 3,109 |
| **Chunks with U+0100-U+052F** | 1 (0.06%) | 18 (1.26%) | 19 (0.61%) |

### Specific Quality Findings

1. **Letter-Spaced Words and Split Numbers (Healed)**:
   - On 35 TCS pages (e.g. pages 22, 25), styled section headings had words spaced into single letters (e.g. `"F ortune M ag az i ne"`) and split numbers (e.g. `"584, 51 9"`, `"44, 0 0 0 +"`).
   - Our **per-page best-of extractor** automatically routes these 35 pages to `PyMuPDF` with `TEXT_INHIBIT_SPACES`, healing them into `"Fortune Magazine"`, `"584,519"`, and `"44,000+"`.
   - The other 706 pages remain on standard default extraction, protecting clean pages from word-gluing regressions.

2. **Irrecoverable Font CMap Corruption (Filtered & Gated)**:
   - 17 TCS pages (Pages 8, 22, 23, 24, 36, 37, 48, 51, 56, 58, 59, 69, 70, 83, 130, 170, 173) contain severe font encoding defects (`odd_char_count >= 15`) in the embedded font `HouschkaRounded-DemiBold`. Visual glyphs map to incorrect Unicode points in the Latin Extended and Greek blocks (`\u0100` to `\u052F`, such as Greek Yot `ϳ` for digit `7`, and `Đ` for `C`).
   - On these 17 pages, corrupted tokens are filtered out, the chunks are marked `text_quality = "degraded"`, and defect details are logged to `data/processed/degraded_pages.csv`. Surrounding clean text (such as Chairman Chandrasekaran's letter on AI) remains intact.
   - **Allowlisting & False Positive Prevention**: On `infosys-ar-26` page 5, the client brand name `"Mondelēz"` contains the accented character `ē` (`\u0113`). Because page 5 has only 6 odd characters ($< 15$), it is left untouched and marked `"clean"`. Furthermore, `"Mondelēz"` is allowlisted in `settings.yaml` under `quality.keep_tokens` so that it is never dropped even on degraded pages.
   - **Residual Characters Below Threshold**: 14 TCS pages have minor font artifacts below the threshold (1 to 13 odd characters, appearing in 18 chunks) and remain marked `"clean"`, avoiding over-filtering while isolating major defects.

3. **Visual OCR Deferred**:
   - Visual OCR (e.g. Tesseract) to reconstruct headline figures from the 17 degraded pages is deferred to a future phase to maintain a lean, high-throughput native extraction pipeline for the 97.7% clean corpus.
