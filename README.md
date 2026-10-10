# annual-report-rag

Question answering over company annual reports using Retrieval-Augmented Generation (RAG). The application parses complex, multi-page financial disclosures, indexes them into a vector database, and generates concise answers grounded strictly in the retrieved text with explicit page citations, accompanied by an empirical evaluation harness measuring hit rate and faithfulness.

## Quick start

Follow these steps to set up and verify the environment:

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment variables (needed for answer generation)**:
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
   > [!NOTE]
   > Document ingestion runs locally by default using FastEmbed ONNX and does **not** require an API key or external network calls. The `GEMINI_API_KEY` is only needed when generating answers or if switching the embedding provider to Gemini.

3. **Run the smoke test**:
   Verify your local embedder and Gemini response generation:
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
   Parse and chunk documents to preview page counts, quality breakdown, repaired glyphs, and chunk volume:
   ```bash
   uv run python scripts/ingest.py --dry-run
   ```

4. **Full ingestion into Chroma vector store (Local FastEmbed, zero API cost)**:
   Idempotently embed all chunks locally using ONNX `BAAI/bge-base-en-v1.5` (768 dimensions) and persist to ChromaDB:
   ```bash
   uv run python scripts/ingest.py
   ```
   > [!TIP]
   > Full ingestion of ~3,109 chunks runs locally on CPU in **under 60 seconds** with zero API keys or rate limits. To wipe and rebuild the index from scratch, pass the `--reset` flag:
   > ```bash
   > uv run python scripts/ingest.py --reset
   > ```

### Switching to Gemini Embeddings & Daily Quota Limits

To switch from local FastEmbed to cloud-based Gemini embeddings:
1. Open `config/settings.yaml` and update the `embedding` block:
   ```yaml
   embedding:
     provider: "gemini"
     model: "gemini-embedding-001"
     dimension: 768
     batch_size: 20
     max_items_per_minute: 80
   ```
2. Re-index the store with `--reset`:
   ```bash
   uv run python scripts/ingest.py --reset
   ```

> [!WARNING]
> **Free-Tier Daily Quota Limit**: On the Google Gemini free tier, the embedding API enforces a hard ceiling of **1,000 requests per day** (`quotaId: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier`). Full ingestion of ~3,109 chunks exhausts the daily free quota around ~800 chunks with a ~21-hour backoff delay.
> Our ingestion pipeline detects daily quota exhaustion immediately (`DailyQuotaExceeded`), prints chunks stored vs remaining, and exits cleanly with exit code 2. You can safely resume next day by running `uv run python scripts/ingest.py` without `--reset`.

### Running the Search Demo

Once documents are indexed, run a manual similarity search against the vector store:
```bash
uv run python scripts/search_demo.py "What was the total revenue growth?"
```
This embeds your question with retrieval query optimization, queries Chroma using cosine similarity, and displays top-$k$ matches with page numbers, similarity scores, extraction methods, quality status, and excerpt previews.

### Asking Grounded Questions via CLI (`scripts/ask.py`)

Run the full RAG query pipeline (source routing, retrieval, grounded generation with `[n]` citations, and numeric grounding verification) directly from the command line without starting a server:
```powershell
uv run python scripts/ask.py "What was TCS's consolidated revenue in fiscal 2026 and its growth?"
```
Pass `--show-context` to inspect the full numbered passages sent to the generator:
```powershell
uv run python scripts/ask.py "Compare TCS and Infosys revenue growth." --show-context
```

> [!IMPORTANT]
> **Why citations say `PDF page N`**: Corporate annual reports include unnumbered cover pages, introductory foldouts, and Roman-numeral prefaces before printed page `1` begins (or omit pages from the PDF file). Because PyMuPDF indexes physical PDF sheets starting at `1`, `PDF page N` refers to the 1-indexed page number in your PDF viewer—which often differs from the printed page number inside the report header/footer (for example, TCS `PDF page 108` carries printed page number `138`, TCS `PDF page 224` carries printed page `254`, and Infosys `PDF page 80` carries printed page `111`). Always navigate to `PDF page N` in your PDF reader when verifying citations.

### Running the FastAPI Service

1. **Start the API server**:
   ```powershell
   uv run uvicorn annual_report_rag.api.main:app --reload
   ```
   Once running, open **`http://127.0.0.1:8000/docs`** in your browser to explore the interactive Swagger/OpenAPI documentation and test endpoints directly.

2. **Query the API from PowerShell (`Invoke-RestMethod`)**:
   - Check service readiness and model configuration:
     ```powershell
     Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
     ```
   - Inspect indexed chunk counts per source:
     ```powershell
     Invoke-RestMethod -Uri "http://127.0.0.1:8000/sources" -Method Get
     ```
   - Ask a grounded question (`POST /ask`):
     ```powershell
     $body = @{ question = "How many employees does TCS have?" } | ConvertTo-Json
     Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body $body
     ```

## Known data quality issues

Corporate annual reports frequently employ stylized typography, custom subsetted fonts, and complex multi-column layouts that introduce extraction defects. Our ingestion pipeline includes an automated quality auditor (`scripts/audit_text_quality.py`), glyph repair table (`ingestion/glyph_repair.py`), and a regression safety harness (`scripts/check_inhibit_regression.py`). Current corpus metrics across 743 examined pages (741 loaded, 2 empty skipped in Infosys):

| Metric | Infosys (`infosys-ar-26.pdf`) | TCS (`tcs-ar-26.pdf`) | Total Corpus |
| :--- | :--- | :--- | :--- |
| **Total Pages Examined** | 383 | 360 | 743 |
| **Pages Loaded ($\ge 50$ chars)** | 381 | 360 | 741 |
| **Default Extraction Pages** | 381 (100%) | 325 (90.3%) | 706 (95.3%) |
| **Inhibit Spaces Pages** | 0 (0.0%) | 35 (9.7%) | 35 (4.7%) |
| **Text Quality: Clean Pages** | 381 (100%) | 329 (91.4%) | 710 (95.8%) |
| **Text Quality: Repaired Pages** | 0 (0.0%) | 1 (0.3%) | 1 (0.1%) |
| **Text Quality: Degraded Pages** | 0 (0.0%) | 30 (8.3%) | 30 (4.1%) |
| **Repaired Characters** | 0 | 432 | 432 |
| **Dropped Corrupted Tokens** | 0 | 224 | 224 |
| **Total Chunks Generated** | 1,684 | 1,425 | 3,109 |
| **Chunks with U+0100-U+052F** | 3 (0.18%) | 0 (0.00%) | 3 (0.10%) |

### Specific Quality Findings

1. **Letter-Spaced Words and Split Numbers (Healed)**:
   - On 35 TCS pages (e.g. pages 22, 25), styled section headings had words spaced into single letters (e.g. `"F ortune M ag az i ne"`) and split numbers (e.g. `"584, 51 9"`, `"44, 0 0 0 +"`).
   - Our **per-page best-of extractor** automatically routes these 35 pages to `PyMuPDF` with `TEXT_INHIBIT_SPACES`, healing them into `"Fortune Magazine"`, `"584,519"`, and `"44,000+"`.
   - The other 706 pages remain on standard default extraction, protecting clean pages from word-gluing regressions.

2. **Empirical Glyph Repair for Corrupted Numerals and Symbols (432 Characters Healed)**:
   - In `tcs-ar-26.pdf`, corrupted font encodings caused digits to map to Coptic/Greek characters (`\u03EC-\u03F5`), commas to combining mark `\u0355`, periods to `\u0358`, the rupee sign to `\u0F24`, and percent signs to Cyrillic `\u0439` ('й').
   - Our `glyph_repair.py` lookup table repairs these characters upfront: `"26ϳ,021"` becomes `"267,021"`, `"₹Ϯ͕ ϲϳ͕ ϬϮ1"` becomes `"₹2,67,021"`, `"4.6й"` becomes `"4.6%"`, and `"Equity ༤ϭϬϳ͕ϮϰϬ"` decodes literally to `"Equity ₹107,240"`.
   - In accordance with safety rules, **zero letter guessing is performed** (letters like `\u010F` for `b` are excluded from repair), and number formatting is never rewritten.

3. **Gated Token Filtering & Allowlisting**:
   - For remaining corrupted glyphs, tokens in `\u0100` to `\u052F` are dropped on degraded pages (`odd_char_count >= 1`), marking 30 TCS pages as `"degraded"` and logging details to `data/processed/degraded_pages.csv`.
   - **Zero corrupted chunks remain in TCS**: All 1,425 TCS chunks are 100% free of corrupted Unicode characters.
   - **Allowlisting**: Legitimate brand name `"Mondelēz"` (`ē` is `\u0113`) on Infosys page 5 is preserved via `settings.yaml` under `quality.keep_tokens`.

4. **Visual OCR Deferred**:
   - Visual OCR (e.g. Tesseract) to reconstruct words from the 30 degraded pages is deferred to a future phase to maintain a lean, high-throughput native extraction pipeline for the 95.8% clean corpus.

## Evaluation (`baseline-v0`)

The repository includes a deterministic evaluation harness (`src/annual_report_rag/evaluation/`) and a 12-question golden seed set (`data/eval/golden_questions.json`) covering factual lookups, cross-company comparisons, out-of-scope unanswerable queries, and adversarial prompt-injection probes. See [`docs/06-evaluation.md`](docs/06-evaluation.md) for full documentation on metrics, golden set rules, and limitations.

### Running the Evaluation Harness

1. **Validate `golden_questions.json` and check expected pages in ChromaDB (offline, zero API calls)**:
   ```powershell
   uv run python scripts/run_eval.py --validate-only
   ```
2. **Run retrieval-only evaluation (offline, zero API calls)**:
   ```powershell
   uv run python scripts/run_eval.py --retrieval-only --label baseline-v0-retrieval
   ```
3. **Run full evaluation (cached in `data/eval/cache/` keyed by question, settings, prompt, and store fingerprint)**:
   ```powershell
   uv run python scripts/run_eval.py --label baseline-v0
   ```
4. **Compare two evaluation runs**:
   ```powershell
   uv run python scripts/compare_runs.py data/eval/results/<runA>.metrics.json data/eval/results/<runB>.metrics.json
   ```

### Baseline v0 Summary (`12` seed questions: `9` answerable, `3` unanswerable/adversarial)

> [!WARNING]
> **Preliminary Benchmark**: Evaluated on $n = 12$ seed questions (`human_verified = false`). Full committed reports are in [`data/eval/results/`](data/eval/results/).

| Metric Category | Metric | `baseline-v0` Value |
| :--- | :--- | :--- |
| **Retrieval ($n=9$ answerable)** | `page_hit@1` / `page_hit@3` / `page_hit@6` | `0.4444` / `0.6667` / `1.0000` |
| **Retrieval ($n=9$ answerable)** | `MRR` / `text_hit@6` | `0.6519` / `0.7778` |
| **Answer Outcomes ($n=12$)** | `CORRECT` / `WRONG` / `FALSE_REFUSAL` | `6` / `1` / `2` |
| **Refusal & Safety ($n=3$)** | `CORRECT_REFUSAL` / `ANSWERED_ANYWAY` | `3` / `0` |
| **Grounding & Numbers** | `grounded_rate` (answerable) / `unsupported_numbers` | `0.7778` / `0` |
| **Quality Slice: `all_clean` ($n=4$)** | `page_hit@6` / `MRR` / `text_hit@6` / `CORRECT` | `1.0000` / `0.8750` / `1.0000` / `4/4` |
| **Quality Slice: `mixed` ($n=1$)** | `page_hit@6` / `MRR` / `text_hit@6` / `CORRECT` | `1.0000` / `0.5000` / `1.0000` / `1/1` |
| **Quality Slice: `all_degraded` ($n=4$)** | `page_hit@6` / `MRR` / `text_hit@6` / `CORRECT` | `1.0000` / `0.4667` / `0.5000` / `1/4` |
| **Latency (`p50` / `p95`)** | `retrieve_ms` / `generate_ms` / `total_ms` | `30.86 / 92.71 ms` \| `1025.96 / 8850.71 ms` \| `1054.59 / 8886.28 ms` |
| **Token Usage** | `avg_input_tokens` / `avg_output_tokens` | `2073.9` / `32.0` |

