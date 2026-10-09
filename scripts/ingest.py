"""CLI script for ingesting PDF annual reports into Chroma vector store."""

import argparse
import math
from pathlib import Path

from annual_report_rag.config import Settings, load_settings
from annual_report_rag.ingestion.chunker import chunk_pages
from annual_report_rag.ingestion.indexer import index_pdfs
from annual_report_rag.ingestion.loader import get_pdf_page_count, load_pdf
from annual_report_rag.providers.factory import get_provider
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
    batch_size = settings.provider.batch_size
    rate = settings.provider.max_items_per_minute
    total_chunks = 0
    total_pages_read = 0
    total_pages_skipped = 0

    for path in pdf_paths:
        total_in_pdf = get_pdf_page_count(path)
        examined = min(total_in_pdf, limit_pages) if limit_pages else total_in_pdf
        pages = load_pdf(
            path,
            min_page_chars=settings.ingestion.min_page_chars,
            limit_pages=limit_pages,
        )
        skipped = examined - len(pages)
        chunks = chunk_pages(
            pages,
            chunk_size=settings.ingestion.chunk_size,
            chunk_overlap=settings.ingestion.chunk_overlap,
        )
        api_calls = (
            math.ceil(len(chunks) / batch_size) if len(chunks) > 0 else 0
        )

        print(f"\nPDF: {path.name}")
        print(f"  Pages examined:        {examined}")
        print(f"  Pages loaded:          {len(pages)}")
        print(f"  Pages skipped:         {skipped}")
        print(f"  Chunks generated:      {len(chunks)}")
        print(
            f"  Est. embedding calls:  {api_calls} (batch size {batch_size})"
        )

        total_pages_read += len(pages)
        total_pages_skipped += skipped
        total_chunks += len(chunks)

    total_api_calls = (
        math.ceil(total_chunks / batch_size) if total_chunks > 0 else 0
    )
    est_minutes = total_chunks / rate if rate > 0 else 0.0

    print("\n--- Dry-Run Summary ---")
    print(f"Total PDFs:               {len(pdf_paths)}")
    print(f"Total pages read:         {total_pages_read}")
    print(f"Total pages skipped:      {total_pages_skipped}")
    print(f"Total chunks created:     {total_chunks}")
    print(
        f"Total est. API calls:     {total_api_calls} (batch size {batch_size})"
    )
    print(
        f"Est. embedding time:      ~{est_minutes:.1f} min (at {rate} items/min limit)"
    )


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
        embedding_model=settings.provider.embedding_model,
        embedding_dimension=settings.provider.embedding_dimension,
        reset_existing=args.reset,
    )

    if args.reset:
        print(f"Collection '{settings.storage.collection_name}' reset successfully.")

    provider = get_provider(settings)
    print(f"Indexing {len(pdf_paths)} document(s)...")
    summary = index_pdfs(
        pdf_paths, provider, store, settings, limit_pages=args.limit_pages
    )

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
