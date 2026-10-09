"""Unit tests for indexer orchestration and idempotency."""

from pathlib import Path

import pymupdf

from annual_report_rag.config import (
    IngestionConfig,
    ProviderConfig,
    RetrievalConfig,
    RetryConfig,
    Settings,
    StorageConfig,
)
from annual_report_rag.ingestion.indexer import index_pdfs
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.retrieval.store import ChromaStore


def _build_test_settings(db_dir: Path) -> Settings:
    """Creates temporary Settings pointing to isolated test directories."""
    return Settings(
        provider=ProviderConfig(
            name="fake",
            embedding_model="fake-embed",
            generation_model="fake-gen",
            embedding_dimension=32,
            batch_size=10,
            retry=RetryConfig(
                max_retries=1, initial_wait_seconds=0.1, max_wait_seconds=0.5
            ),
        ),
        ingestion=IngestionConfig(
            chunk_size=100, chunk_overlap=20, min_page_chars=10
        ),
        retrieval=RetrievalConfig(top_k=3),
        storage=StorageConfig(
            vector_store_path=str(db_dir),
            collection_name="test_idempotence",
            raw_data_dir=str(db_dir / "raw"),
            eval_data_dir=str(db_dir / "eval"),
        ),
    )


def test_indexer_idempotence(tmp_path: Path) -> None:
    """Verifies that re-running the indexer adds no duplicate chunks or redundant embeddings."""
    pdf_path = tmp_path / "annual_report.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "First page discussing audited balance sheet figures.")
    p2 = doc.new_page()
    p2.insert_text((50, 72), "Second page analyzing comprehensive cash flow statements.")
    doc.save(pdf_path)
    doc.close()

    db_path = tmp_path / "chroma_test"
    settings = _build_test_settings(db_path)
    provider = FakeProvider(embedding_dim=32)
    store = ChromaStore(
        persist_path=str(db_path), collection_name="test_idempotence"
    )

    # First run: should index all chunks
    summary1 = index_pdfs([pdf_path], provider, store, settings)
    assert summary1.pages_read == 2
    assert summary1.chunks_new > 0
    assert summary1.chunks_skipped == 0
    initial_store_count = store.count()
    assert initial_store_count == summary1.chunks_new

    # Second run: must skip all existing chunks and perform zero new inserts
    summary2 = index_pdfs([pdf_path], provider, store, settings)
    assert summary2.pages_read == 2
    assert summary2.chunks_new == 0
    assert summary2.chunks_skipped == summary1.chunks_created
    assert store.count() == initial_store_count
