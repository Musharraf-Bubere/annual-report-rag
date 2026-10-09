"""Unit tests for ChromaStore persistent vector store operations."""

from pathlib import Path

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
    )
    chunk2 = Chunk(
        id="tcs-p5-c0",
        text="Headcount expanded in international delivery centers.",
        source="tcs",
        page=5,
        chunk_index=0,
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


def test_empty_store_query(tmp_path: Path) -> None:
    """Verifies that querying an empty store returns an empty list without raising errors."""
    db_path = str(tmp_path / "empty_db")
    store = ChromaStore(persist_path=db_path, collection_name="empty_col")

    results = store.query([0.5, 0.5], top_k=5)
    assert results == []
    assert store.existing_ids([]) == set()
