"""Retrieval orchestrator coordinating source routing, query embedding, and store queries."""

from dataclasses import dataclass

from annual_report_rag.config import Settings
from annual_report_rag.providers.base import EmbeddingProvider
from annual_report_rag.retrieval.router import detect_sources
from annual_report_rag.retrieval.store import ChromaStore, QueryResult


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a retrieved passage with provenance metadata and similarity score."""

    chunk_id: str
    text: str
    source: str
    page: int
    score: float
    text_quality: str = "clean"
    extraction_method: str = "default"


def _to_retrieved_chunk(hit: QueryResult) -> RetrievedChunk:
    """Converts a store QueryResult into a RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=hit.chunk_id,
        text=hit.text,
        source=hit.source,
        page=hit.page,
        score=hit.score,
        text_quality=hit.text_quality,
        extraction_method=hit.extraction_method,
    )


def _query_by_sources(
    store: ChromaStore,
    query_vector: list[float],
    top_k: int,
    target_sources: list[str],
) -> list[QueryResult]:
    """Queries the store globally, for a single source, or per-source with score ordering."""
    if not target_sources:
        return store.query(query_vector, top_k=top_k, sources=None)

    if len(target_sources) == 1:
        return store.query(query_vector, top_k=top_k, sources=[target_sources[0]])

    combined_hits: list[QueryResult] = []
    for src in target_sources:
        combined_hits.extend(store.query(query_vector, top_k=top_k, sources=[src]))

    combined_hits.sort(key=lambda h: h.score, reverse=True)
    return combined_hits


def retrieve(
    question: str,
    embedder: EmbeddingProvider,
    store: ChromaStore,
    settings: Settings,
    top_k: int | None = None,
    sources: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Embeds a user question and retrieves top-k passages routed by company aliases.

    Args:
        question: User's natural language question.
        embedder: Embedding provider for computing query vector.
        store: Vector store containing indexed chunks.
        settings: Application configuration.
        top_k: Optional override for number of chunks to retrieve (per source).
        sources: Optional explicit list of source stems to filter retrieval.

    Returns:
        list[RetrievedChunk]: Ranked retrieved passages passing min_score threshold.
    """
    cleaned_question = question.strip()
    if not cleaned_question or store.count() == 0:
        return []

    effective_top_k = top_k if top_k is not None else settings.retrieval.top_k
    if effective_top_k <= 0:
        return []

    target_sources = (
        sources if sources is not None else detect_sources(cleaned_question, settings)
    )

    vectors = embedder.embed([cleaned_question], purpose="query")
    if not vectors:
        return []

    raw_hits = _query_by_sources(
        store=store,
        query_vector=vectors[0],
        top_k=effective_top_k,
        target_sources=target_sources,
    )

    min_score = settings.retrieval.min_score
    if min_score > 0.0:
        raw_hits = [hit for hit in raw_hits if hit.score >= min_score]

    return [_to_retrieved_chunk(hit) for hit in raw_hits]
