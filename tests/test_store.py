"""Unit tests for ChromaStore persistent vector store operations."""

from pathlib import Path

import pytest

from annual_report_rag.ingestion.chunker import Chunk
from annual_report_rag.retrieval.store import ChromaStore


def test_chroma_store_operations(tmp_path: Path) -> None:
    """Verifies chunk addition, existence verification, counting, and nearest neighbor queries."""
    db_path = str(tmp_path / "test_chroma")
    store = ChromaStore(persist_path=db_path, collection_name="test_col")

    assert store.count() == 0

    chunk1 = Chunk(
        id="infy-p1-c0",
        text="Cloud services generated strong quarterly revenue.",
        source="infy",
        page=1,
        chunk_index=0,
        extraction_method="default",
        text_quality="clean",
    )
    chunk2 = Chunk(
        id="tcs-p5-c0",
        text="Headcount expanded in international delivery centers.",
        source="tcs",
        page=5,
        chunk_index=0,
        extraction_method="inhibit_spaces",
        text_quality="degraded",
    )

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    store.add_chunks([chunk1, chunk2], [vec1, vec2])

    assert store.count() == 2

    existing = store.existing_ids(["infy-p1-c0", "non-existent-id"])
    assert existing == {"infy-p1-c0"}

    # Query with vector identical to vec1
    query_vec = [1.0, 0.0, 0.0]
    results = store.query(query_vec, top_k=1)

    assert len(results) == 1
    assert results[0].source == "infy"
    assert results[0].page == 1
    assert results[0].text == "Cloud services generated strong quarterly revenue."
    assert results[0].score == 1.0
    assert results[0].extraction_method == "default"
    assert results[0].text_quality == "clean"


def test_empty_store_query(tmp_path: Path) -> None:
    """Verifies that querying an empty store returns an empty list without raising errors."""
    db_path = str(tmp_path / "empty_db")
    store = ChromaStore(persist_path=db_path, collection_name="empty_col")

    results = store.query([0.5, 0.5], top_k=5)
    assert results == []
    assert store.existing_ids([]) == set()


def test_chroma_store_reset(tmp_path: Path) -> None:
    """Verifies that reset() deletes all existing records from the collection."""
    db_path = str(tmp_path / "reset_db")
    store = ChromaStore(persist_path=db_path, collection_name="reset_col")
    chunk = Chunk(id="c1", text="Sample", source="s", page=1, chunk_index=0)
    store.add_chunks([chunk], [[1.0, 0.0]])
    assert store.count() == 1

    store.reset()
    assert store.count() == 0


def test_chroma_store_reset_on_missing_collection(tmp_path: Path) -> None:
    """Verifies that resetting a collection that does not exist yet succeeds cleanly."""
    db_path = str(tmp_path / "missing_reset_db")
    # Initializing with reset_existing=True when no collection exists yet
    store = ChromaStore(
        persist_path=db_path,
        collection_name="new_never_existed_col",
        embedding_provider="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
        reset_existing=True,
    )
    assert store.count() == 0

    # Calling reset() again on empty collection
    store.reset()
    assert store.count() == 0


def test_chroma_store_metadata_matching(tmp_path: Path) -> None:
    """Verifies that opening an existing collection with matching metadata succeeds."""
    db_path = str(tmp_path / "meta_db")
    ChromaStore(
        persist_path=db_path,
        collection_name="meta_col",
        embedding_provider="local",
        embedding_model="model-v1",
        embedding_dimension=768,
    )

    # Reopening with identical configuration must succeed
    reopened = ChromaStore(
        persist_path=db_path,
        collection_name="meta_col",
        embedding_provider="local",
        embedding_model="model-v1",
        embedding_dimension=768,
    )
    assert reopened.collection.metadata.get("embedding_provider") == "local"
    assert reopened.collection.metadata.get("embedding_model") == "model-v1"


def test_chroma_store_metadata_mismatch_raises(tmp_path: Path) -> None:
    """Verifies that opening an existing collection with mismatched settings raises ValueError."""
    db_path = str(tmp_path / "mismatch_db")
    ChromaStore(
        persist_path=db_path,
        collection_name="mismatch_col",
        embedding_provider="local",
        embedding_model="model-v1",
        embedding_dimension=768,
    )

    # Mismatched provider
    with pytest.raises(ValueError, match="Please re-index using --reset"):
        ChromaStore(
            persist_path=db_path,
            collection_name="mismatch_col",
            embedding_provider="gemini",
            embedding_model="model-v1",
            embedding_dimension=768,
        )

    # Mismatched model name
    with pytest.raises(ValueError, match="Please re-index using --reset"):
        ChromaStore(
            persist_path=db_path,
            collection_name="mismatch_col",
            embedding_provider="local",
            embedding_model="model-v2",
            embedding_dimension=768,
        )

    # Mismatched dimension
    with pytest.raises(ValueError, match="Please re-index using --reset"):
        ChromaStore(
            persist_path=db_path,
            collection_name="mismatch_col",
            embedding_provider="local",
            embedding_model="model-v1",
            embedding_dimension=1536,
        )


def test_chroma_store_reset_overrides_mismatch(tmp_path: Path) -> None:
    """Verifies that reset_existing=True resets mismatched collection without raising error."""
    db_path = str(tmp_path / "override_db")
    ChromaStore(
        persist_path=db_path,
        collection_name="override_col",
        embedding_provider="gemini",
        embedding_model="model-v1",
        embedding_dimension=768,
    )

    # Resetting with new provider and model configuration
    recreated = ChromaStore(
        persist_path=db_path,
        collection_name="override_col",
        embedding_provider="local",
        embedding_model="model-v2",
        embedding_dimension=512,
        reset_existing=True,
    )
    assert recreated.collection.metadata.get("embedding_provider") == "local"
    assert recreated.collection.metadata.get("embedding_model") == "model-v2"
    assert recreated.collection.metadata.get("embedding_dimension") == 512
