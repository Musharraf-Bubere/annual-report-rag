"""Unit tests for indexer orchestration and idempotency."""

from pathlib import Path

import pymupdf

from annual_report_rag.config import (
    EmbeddingConfig,
    GenerationConfig,
    IngestionConfig,
    QualityConfig,
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
        embedding=EmbeddingConfig(
            provider="fake",
            model="fake-embed",
            dimension=32,
            batch_size=10,
            max_items_per_minute=60,
            retry=RetryConfig(
                max_retries=1,
                max_rate_limit_retries=2,
                initial_wait_seconds=0.1,
                max_wait_seconds=0.5,
            ),
        ),
        generation=GenerationConfig(
            provider="fake",
            model="fake-gen",
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
        quality=QualityConfig(
            single_letter_ratio_threshold=0.15,
            odd_char_ratio_threshold=0.05,
            odd_char_count_threshold=1,
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
        persist_path=str(db_path),
        collection_name="test_idempotence",
        embedding_provider="fake",
        embedding_model="fake-embed",
        embedding_dimension=32,
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


def test_indexer_with_stubbed_local_embedder(tmp_path: Path) -> None:
    """Verifies that index_pdfs works end-to-end with LocalEmbeddingProvider and ChromaStore using temp dirs."""
    from unittest.mock import MagicMock

    import numpy as np

    from annual_report_rag.providers.local_embedder import LocalEmbeddingProvider

    pdf_path = tmp_path / "sample_local.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Sample page content indexed with stubbed local embedder.")
    doc.save(pdf_path)
    doc.close()

    db_path = tmp_path / "chroma_local"
    settings = Settings(
        embedding=EmbeddingConfig(
            provider="local",
            model="test-bge",
            dimension=4,
            batch_size=10,
            max_items_per_minute=60,
            retry=RetryConfig(
                max_retries=1,
                max_rate_limit_retries=2,
                initial_wait_seconds=0.1,
                max_wait_seconds=0.5,
            ),
        ),
        generation=GenerationConfig(
            provider="fake",
            model="fake-gen",
        ),
        ingestion=IngestionConfig(
            chunk_size=100, chunk_overlap=20, min_page_chars=10
        ),
        retrieval=RetrievalConfig(top_k=3),
        storage=StorageConfig(
            vector_store_path=str(db_path),
            collection_name="test_local_embed",
            raw_data_dir=str(tmp_path / "raw"),
            eval_data_dir=str(tmp_path / "eval"),
        ),
        quality=QualityConfig(
            single_letter_ratio_threshold=0.15,
            odd_char_ratio_threshold=0.05,
            odd_char_count_threshold=1,
        ),
    )

    mock_model = MagicMock()

    def fake_passage_embed(texts: list[str]):
        for _ in texts:
            yield np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    mock_model.passage_embed.side_effect = fake_passage_embed

    local_embedder = LocalEmbeddingProvider(
        model_name="test-bge",
        embedding_dimension=4,
        model=mock_model,
    )
    store = ChromaStore(
        persist_path=str(db_path),
        collection_name="test_local_embed",
        embedding_provider="local",
        embedding_model="test-bge",
        embedding_dimension=4,
    )

    summary = index_pdfs([pdf_path], local_embedder, store, settings)
    assert summary.pages_read == 1
    assert summary.chunks_new > 0
    assert store.count() == summary.chunks_new

    # Verify query against store works cleanly
    mock_model.query_embed.return_value = iter(
        [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
    )
    query_vec = local_embedder.embed(["Sample query"], purpose="query")[0]
    results = store.query(query_vec, top_k=1)
    assert len(results) == 1
    assert "Sample page content" in results[0].text
