"""Indexing pipeline orchestrator for PDF documents into Chroma vector store."""

import time
from dataclasses import dataclass
from pathlib import Path

from annual_report_rag.config import Settings
from annual_report_rag.ingestion.chunker import Chunk, chunk_pages
from annual_report_rag.ingestion.loader import get_pdf_page_count, load_pdf
from annual_report_rag.providers.base import BaseProvider
from annual_report_rag.retrieval.store import ChromaStore


@dataclass(frozen=True)
class IndexingSummary:
    """Holds pipeline execution counts and elapsed time metrics."""

    pages_read: int
    pages_skipped: int
    chunks_created: int
    chunks_new: int
    chunks_skipped: int
    seconds_taken: float


def embed_and_store_batches(
    provider: BaseProvider,
    store: ChromaStore,
    chunks: list[Chunk],
    batch_size: int,
) -> None:
    """Embeds new chunks in batches, reporting incremental progress and storing vectors."""
    total = len(chunks)
    processed = 0
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vectors = provider.embed(texts, purpose="document")
        store.add_chunks(batch, vectors)
        processed += len(batch)
        print(f"Embedded {processed}/{total} chunks")


def index_pdfs(
    pdf_paths: list[Path | str],
    provider: BaseProvider,
    store: ChromaStore,
    settings: Settings,
    limit_pages: int | None = None,
) -> IndexingSummary:
    """Executes idempotent indexing pipeline across a list of PDF documents."""
    start_time = time.perf_counter()
    all_chunks: list[Chunk] = []
    total_pages_read = 0
    total_pages_skipped = 0

    for path in pdf_paths:
        total_in_pdf = get_pdf_page_count(path)
        examined = min(total_in_pdf, limit_pages) if limit_pages else total_in_pdf
        pages = load_pdf(
            path,
            min_page_chars=settings.ingestion.min_page_chars,
            limit_pages=limit_pages,
            single_threshold=settings.quality.single_letter_ratio_threshold,
            number_split_threshold=settings.quality.number_split_threshold,
            glyph_range_start=settings.quality.corrupted_glyph_range_start,
            glyph_range_end=settings.quality.corrupted_glyph_range_end,
        )
        total_pages_read += len(pages)
        total_pages_skipped += examined - len(pages)
        all_chunks.extend(
            chunk_pages(
                pages,
                chunk_size=settings.ingestion.chunk_size,
                chunk_overlap=settings.ingestion.chunk_overlap,
            )
        )

    all_ids = [c.id for c in all_chunks]
    existing_ids = store.existing_ids(all_ids)
    new_chunks = [c for c in all_chunks if c.id not in existing_ids]

    if new_chunks:
        print(f"Found {len(new_chunks)} new chunk(s) to embed out of {len(all_chunks)} total.")
        embed_and_store_batches(
            provider, store, new_chunks, settings.provider.batch_size
        )
    else:
        print("All chunks already exist in vector store. Nothing new to embed.")

    elapsed = time.perf_counter() - start_time
    return IndexingSummary(
        pages_read=total_pages_read,
        pages_skipped=total_pages_skipped,
        chunks_created=len(all_chunks),
        chunks_new=len(new_chunks),
        chunks_skipped=len(all_chunks) - len(new_chunks),
        seconds_taken=round(elapsed, 2),
    )
