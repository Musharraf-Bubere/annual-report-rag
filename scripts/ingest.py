"""CLI script for ingesting PDF annual reports into Chroma vector store."""

import argparse
import math
import sys
from pathlib import Path

from annual_report_rag.config import Settings, load_settings
from annual_report_rag.ingestion.chunker import chunk_pages
from annual_report_rag.ingestion.indexer import index_pdfs
from annual_report_rag.ingestion.loader import get_pdf_page_count, load_pdf
from annual_report_rag.providers.factory import get_embedder
from annual_report_rag.providers.gemini import DailyQuotaExceeded
from annual_report_rag.retrieval.store import ChromaStore


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest annual report PDFs into Chroma vector store."
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/raw",
        help="Directory containing PDF files (default: data/raw)",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Optional limit on number of pages to process per PDF",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only, estimating API calls and runtime without embedding",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before indexing to re-index all documents",
    )
    return parser.parse_args()


def execute_dry_run(
    pdf_paths: list[Path],
    settings: Settings,
    limit_pages: int | None,
) -> None:
    """Executes parsing and chunking in dry-run mode without calling embedding APIs."""
    print("\n--- Running Ingestion Dry-Run ---")
    batch_size = settings.embedding.batch_size
    rate = settings.embedding.max_items_per_minute
    provider_name = settings.embedding.provider
    total_chunks = 0
    total_pages_read = 0
    total_pages_skipped = 0
    total_default_pages = 0
    total_inhibit_pages = 0
    total_clean_pages = 0
    total_repaired_pages = 0
    total_degraded_pages = 0
    total_repaired_chars = 0
    total_dropped_tokens = 0
    all_chunks = []

    for path in pdf_paths:
        total_in_pdf = get_pdf_page_count(path)
        examined = min(total_in_pdf, limit_pages) if limit_pages else total_in_pdf
        pages = load_pdf(
            path,
            min_page_chars=settings.ingestion.min_page_chars,
            limit_pages=limit_pages,
            single_threshold=settings.quality.single_letter_ratio_threshold,
            number_split_threshold=settings.quality.number_split_threshold,
            odd_count_threshold=settings.quality.odd_char_count_threshold,
            glyph_range_start=settings.quality.corrupted_glyph_range_start,
            glyph_range_end=settings.quality.corrupted_glyph_range_end,
            keep_tokens=settings.quality.keep_tokens,
        )
        skipped = examined - len(pages)
        chunks = chunk_pages(
            pages,
            chunk_size=settings.ingestion.chunk_size,
            chunk_overlap=settings.ingestion.chunk_overlap,
        )
        all_chunks.extend(chunks)
        api_calls = (
            math.ceil(len(chunks) / batch_size) if len(chunks) > 0 else 0
        )

        n_default = sum(1 for p in pages if p.extraction_method == "default")
        n_inhibit = sum(1 for p in pages if p.extraction_method == "inhibit_spaces")
        n_clean = sum(1 for p in pages if p.text_quality == "clean")
        n_repaired = sum(1 for p in pages if p.text_quality == "repaired")
        n_degraded = sum(1 for p in pages if p.text_quality == "degraded")
        repaired_chars = sum(p.repaired_chars for p in pages)
        dropped_tokens = sum(p.dropped_tokens for p in pages)

        # PDF chunks with U+0100-U+052F
        r_start = settings.quality.corrupted_glyph_range_start
        r_end = settings.quality.corrupted_glyph_range_end
        pdf_chunks_with_range: list[tuple[int, list[str]]] = []
        for c in chunks:
            chs = sorted({ch for ch in c.text if r_start <= ch <= r_end})
            if chs:
                pdf_chunks_with_range.append((c.page, chs))

        print(f"\nPDF: {path.name}")
        print(f"  Pages examined:        {examined}")
        print(f"  Pages loaded:          {len(pages)}")
        print(f"  Pages skipped:         {skipped}")
        print(f"  Extraction methods:    default: {n_default}, inhibit_spaces: {n_inhibit}")
        print(f"  Text quality pages:    clean: {n_clean}, repaired: {n_repaired}, degraded: {n_degraded}")
        print(f"  Repaired characters:   {repaired_chars}")
        print(f"  Dropped tokens:        {dropped_tokens}")
        print(f"  Chunks generated:      {len(chunks)}")
        print(f"  Chunks with U+0100-U+052F: {len(pdf_chunks_with_range)}")
        if pdf_chunks_with_range:
            page_map: dict[int, set[str]] = {}
            for pg, chs in pdf_chunks_with_range:
                page_map.setdefault(pg, set()).update(chs)
            for pg, chs in sorted(page_map.items()):
                formatted = ", ".join(f"{c!a} (U+{ord(c):04X})" for c in sorted(chs))
                print(f"    - Page {pg}: {formatted}")

        if provider_name == "gemini":
            print(f"  Est. embedding calls:  {api_calls} (batch size {batch_size})")

        total_pages_read += len(pages)
        total_pages_skipped += skipped
        total_default_pages += n_default
        total_inhibit_pages += n_inhibit
        total_clean_pages += n_clean
        total_repaired_pages += n_repaired
        total_degraded_pages += n_degraded
        total_repaired_chars += repaired_chars
        total_dropped_tokens += dropped_tokens
        total_chunks += len(chunks)

    # Inspect all chunks containing characters in U+0100-U+052F
    r_start = settings.quality.corrupted_glyph_range_start
    r_end = settings.quality.corrupted_glyph_range_end
    chunks_with_range_chars: list[tuple[str, int, list[str]]] = []
    for c in all_chunks:
        chars = sorted({ch for ch in c.text if r_start <= ch <= r_end})
        if chars:
            chunks_with_range_chars.append((c.source, c.page, chars))

    print("\n--- Dry-Run Summary ---")
    print(f"Total PDFs:               {len(pdf_paths)}")
    print(f"Total pages read:         {total_pages_read}")
    print(f"Total pages skipped:      {total_pages_skipped}")
    print(f"Extraction methods:       default: {total_default_pages}, inhibit_spaces: {total_inhibit_pages}")
    print(f"Text quality pages:       clean: {total_clean_pages}, repaired: {total_repaired_pages}, degraded: {total_degraded_pages}")
    print(f"Total repaired characters:{total_repaired_chars}")
    print(f"Total dropped tokens:     {total_dropped_tokens}")
    print(f"Total chunks created:     {total_chunks}")
    print(f"Chunks with U+0100-U+052F:{len(chunks_with_range_chars)}")
    if chunks_with_range_chars:
        page_chars: dict[tuple[str, int], set[str]] = {}
        for src, pg, chs in chunks_with_range_chars:
            page_chars.setdefault((src, pg), set()).update(chs)
        for (src, pg), chs in sorted(page_chars.items()):
            formatted_chars = ", ".join(
                f"{c!a} (U+{ord(c):04X})" for c in sorted(chs)
            )
            print(f"  - {src} page {pg}: {formatted_chars}")

    if provider_name == "gemini":
        total_api_calls = (
            math.ceil(total_chunks / batch_size) if total_chunks > 0 else 0
        )
        est_minutes = total_chunks / rate if rate > 0 else 0.0
        print(f"Total est. API calls:     {total_api_calls} (batch size {batch_size})")
        print(f"Est. embedding time:      ~{est_minutes:.1f} min (at {rate} items/min limit)")
    else:
        print(f"Embedding provider:       local ({settings.embedding.model}) via FastEmbed ONNX (offline)")


def main() -> None:
    """Entry point coordinating live ingestion or dry-run estimation."""
    args = parse_args()
    settings = load_settings()
    pdf_dir = Path(args.pdf_dir)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in '{pdf_dir}'.")
        return

    if args.dry_run:
        execute_dry_run(pdf_paths, settings, args.limit_pages)
        return

    store = ChromaStore(
        persist_path=settings.storage.vector_store_path,
        collection_name=settings.storage.collection_name,
        embedding_provider=settings.embedding.provider,
        embedding_model=settings.embedding.model,
        embedding_dimension=settings.embedding.dimension,
        reset_existing=args.reset,
    )

    if args.reset:
        print(f"Collection '{settings.storage.collection_name}' reset successfully.")

    embedder = get_embedder(settings)
    print(
        f"Indexing {len(pdf_paths)} document(s) with "
        f"provider '{settings.embedding.provider}' (model: {settings.embedding.model})..."
    )

    try:
        summary = index_pdfs(
            pdf_paths, embedder, store, settings, limit_pages=args.limit_pages
        )
    except DailyQuotaExceeded as exc:
        stored = exc.chunks_stored if exc.chunks_stored is not None else store.count()
        remaining = exc.chunks_remaining if exc.chunks_remaining is not None else "unknown"
        if exc.retry_delay and exc.retry_delay > 0:
            hours = exc.retry_delay / 3600.0
            resume_msg = f"in roughly {hours:.1f} hours ({exc.retry_delay:.0f}s)"
        else:
            resume_msg = "tomorrow when your daily quota resets"
        print(f"\n[Daily Quota Exceeded] {exc}")
        print(f"Chunks stored so far: {stored}")
        print(f"Chunks remaining:     {remaining}")
        print(f"Estimated resume:     {resume_msg}")
        print("To resume ingestion, re-run without the --reset flag.")
        sys.exit(2)

    print("\n--- Indexing Summary ---")
    print(f"Pages read:       {summary.pages_read}")
    print(f"Pages skipped:    {summary.pages_skipped}")
    print(f"Chunks created:   {summary.chunks_created}")
    print(f"New chunks added: {summary.chunks_new}")
    print(f"Chunks skipped:   {summary.chunks_skipped}")
    print(f"Time taken:       {summary.seconds_taken}s")
    print(f"Total in store:   {store.count()}")


if __name__ == "__main__":
    main()
