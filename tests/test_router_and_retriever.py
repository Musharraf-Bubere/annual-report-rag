"""Unit and integration tests for source router and retriever with a real temporary ChromaStore."""

from pathlib import Path

from annual_report_rag.config import (
    ApiConfig,
    EmbeddingConfig,
    GenerationConfig,
    IngestionConfig,
    QualityConfig,
    RetrievalConfig,
    RetryConfig,
    Settings,
    SourceMetadata,
    StorageConfig,
)
from annual_report_rag.ingestion.chunker import Chunk
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.retrieval.retriever import retrieve
from annual_report_rag.retrieval.router import detect_sources
from annual_report_rag.retrieval.store import ChromaStore


def _build_test_settings(min_score: float = 0.0, top_k: int = 2) -> Settings:
    """Builds a test Settings object with TCS and Infosys source aliases."""
    return Settings(
        embedding=EmbeddingConfig(
            provider="fake",
            model="fake-embed",
            dimension=4,
            batch_size=10,
            max_items_per_minute=100,
            retry=RetryConfig(1, 1, 0.1, 0.5),
        ),
        generation=GenerationConfig(
            provider="fake",
            model="fake-gen",
            temperature=None,
        ),
        ingestion=IngestionConfig(1000, 200, 50),
        retrieval=RetrievalConfig(
            top_k=top_k,
            min_score=min_score,
            max_question_chars=1000,
        ),
        storage=StorageConfig("chroma_db", "test_col", "data/raw", "data/eval"),
        quality=QualityConfig(0.15, 0.002, 1),
        api=ApiConfig(cors_origins=("http://localhost:8501",)),
        sources={
            "tcs-ar-26": SourceMetadata(
                display_name="TCS Integrated Annual Report 2025-26",
                aliases=("tcs", "tata consultancy services"),
            ),
            "infosys-ar-26": SourceMetadata(
                display_name="Infosys Integrated Annual Report 2025-26",
                aliases=("infosys",),
            ),
        },
        gemini_api_key=None,
    )


def test_detect_sources_variants() -> None:
    """Verifies whole-word alias matching for single, multiple, possessive, and non-matching inputs."""
    settings = _build_test_settings()

    # TCS alone
    assert detect_sources("How many employees does TCS have?", settings) == ["tcs-ar-26"]
    assert detect_sources("Tell me about Tata Consultancy Services.", settings) == ["tcs-ar-26"]

    # Infosys alone
    assert detect_sources("What is Infosys revenue growth?", settings) == ["infosys-ar-26"]

    # Both companies
    assert detect_sources("Compare TCS and Infosys revenue growth.", settings) == [
        "tcs-ar-26",
        "infosys-ar-26",
    ]

    # Neither company
    assert detect_sources("What was Wipro's revenue?", settings) == []
    assert detect_sources("What is the operating margin?", settings) == []

    # Possessive and mixed case
    assert detect_sources("What was tcs's revenue in FY26?", settings) == ["tcs-ar-26"]
    assert detect_sources("inFoSys dividend policy", settings) == ["infosys-ar-26"]

    # Partial token "tcsx" must NOT match
    assert detect_sources("What is tcsx revenue?", settings) == []


def test_retriever_single_two_source_global_and_min_score(tmp_path: Path) -> None:
    """Verifies retrieve() against a real temporary ChromaStore across routing and min_score modes."""
    db_path = str(tmp_path / "retriever_chroma")
    store = ChromaStore(persist_path=db_path, collection_name="ret_col")
    embedder = FakeProvider(embedding_dim=4)
    settings = _build_test_settings(min_score=0.0, top_k=2)

    chunks = [
        Chunk(
            id="tcs-p10-c0",
            text="TCS revenue grew strongly.",
            source="tcs-ar-26",
            page=10,
            chunk_index=0,
            extraction_method="default",
            text_quality="clean",
        ),
        Chunk(
            id="tcs-p20-c0",
            text="TCS headcount reached 607,979.",
            source="tcs-ar-26",
            page=20,
            chunk_index=0,
            extraction_method="inhibit_spaces",
            text_quality="degraded",
        ),
        Chunk(
            id="infy-p15-c0",
            text="Infosys revenue grew by 4.2%.",
            source="infosys-ar-26",
            page=15,
            chunk_index=0,
            extraction_method="default",
            text_quality="clean",
        ),
    ]
    # Compute actual embeddings from FakeProvider so query vector aligns predictably
    q_vec = embedder.embed(["Compare TCS and Infosys revenue"], purpose="query")[0]
    orthogonal = [-q_vec[0], -q_vec[1], -q_vec[2], -q_vec[3]]
    close_vec = [q_vec[0], q_vec[1], q_vec[2], -q_vec[3]]
    store.add_chunks(chunks, [close_vec, orthogonal, q_vec])

    # 1. Single source filter (explicit or router)
    tcs_only = retrieve(
        "Compare TCS and Infosys revenue",
        embedder,
        store,
        settings,
        sources=["tcs-ar-26"],
    )
    assert all(c.source == "tcs-ar-26" for c in tcs_only)
    assert len(tcs_only) == 2
    assert tcs_only[0].chunk_id == "tcs-p10-c0"

    # 2. Two-source merge ordered by score descending
    merged = retrieve(
        "Compare TCS and Infosys revenue",
        embedder,
        store,
        settings,
        sources=["tcs-ar-26", "infosys-ar-26"],
    )
    assert len(merged) == 3
    scores = [c.score for c in merged]
    assert scores == sorted(scores, reverse=True)
    assert merged[0].chunk_id == "infy-p15-c0"

    # 3. No source detected -> global search up to top_k (2)
    global_res = retrieve(
        "What was the revenue growth?",
        embedder,
        store,
        settings,
        top_k=2,
    )
    assert len(global_res) == 2

    # 4. min_score filter drops low-scoring chunks
    strict_settings = _build_test_settings(min_score=0.5, top_k=5)
    filtered = retrieve(
        "Compare TCS and Infosys revenue",
        embedder,
        store,
        strict_settings,
        sources=["tcs-ar-26"],
    )
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "tcs-p10-c0"


def test_retriever_query_prefix_and_max_chunks_per_page(tmp_path: Path) -> None:
    """Verifies that retrieval.query_prefix prepends to the embedded query and max_chunks_per_page caps chunks per page."""
    from dataclasses import replace

    db_path = str(tmp_path / "retriever_cap_chroma")
    store = ChromaStore(persist_path=db_path, collection_name="cap_col")
    embedder = FakeProvider(embedding_dim=4)
    base_settings = _build_test_settings(min_score=0.0, top_k=3)

    # Add 3 chunks on page 24 and 1 chunk on page 8 for TCS
    chunks = [
        Chunk(
            id="tcs-p24-c0",
            text="Page 24 chunk 0",
            source="tcs-ar-26",
            page=24,
            chunk_index=0,
        ),
        Chunk(
            id="tcs-p24-c1",
            text="Page 24 chunk 1",
            source="tcs-ar-26",
            page=24,
            chunk_index=1,
        ),
        Chunk(
            id="tcs-p24-c2",
            text="Page 24 chunk 2",
            source="tcs-ar-26",
            page=24,
            chunk_index=2,
        ),
        Chunk(
            id="tcs-p8-c0",
            text="Page 8 chunk 0",
            source="tcs-ar-26",
            page=8,
            chunk_index=0,
        ),
    ]
    q_text = "What was TCS revenue growth?"
    q_vec = embedder.embed([q_text], purpose="query")[0]
    v0 = list(q_vec)
    v1 = [q_vec[0] * 0.99, q_vec[1], q_vec[2], q_vec[3]]
    v2 = [q_vec[0] * 0.95, q_vec[1], q_vec[2], q_vec[3]]
    v3 = [q_vec[0] * 0.80, q_vec[1], q_vec[2], -q_vec[3] * 0.1]
    store.add_chunks(chunks, [v0, v1, v2, v3])

    # Without page cap: top 3 are all from page 24
    uncapped = retrieve(q_text, embedder, store, base_settings)
    assert [c.page for c in uncapped] == [24, 24, 24]

    # With max_chunks_per_page=2: at most 2 from page 24, so page 8 enters top 3
    capped_settings = replace(
        base_settings,
        retrieval=replace(base_settings.retrieval, max_chunks_per_page=2),
    )
    capped = retrieve(q_text, embedder, store, capped_settings)
    assert [c.page for c in capped] == [24, 24, 8]
    assert capped[2].chunk_id == "tcs-p8-c0"

    # With query_prefix: verify prefix is prepended before calling embedder.embed
    recorded_texts: list[str] = []

    class RecordingEmbedder(FakeProvider):
        def embed(self, texts: list[str], purpose: str = "document") -> list[list[float]]:
            recorded_texts.extend(texts)
            return super().embed(texts, purpose=purpose)

    rec_embedder = RecordingEmbedder(embedding_dim=4)
    prefix = "Represent this sentence for searching relevant passages: "
    prefixed_settings = replace(
        base_settings,
        retrieval=replace(base_settings.retrieval, query_prefix=prefix),
    )
    retrieve(q_text, rec_embedder, store, prefixed_settings)
    assert recorded_texts == [f"{prefix}{q_text}"]

