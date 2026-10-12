"""Answer orchestrator combining retrieval, grounded prompt generation, and citation verification."""

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from annual_report_rag.config import Settings
from annual_report_rag.generation.checks import find_unsupported_numbers
from annual_report_rag.generation.prompts import (
    NOT_FOUND_MESSAGE,
    build_prompt,
    get_display_name,
)
from annual_report_rag.providers.base import EmbeddingProvider, GenerationProvider
from annual_report_rag.retrieval.retriever import RetrievedChunk, retrieve
from annual_report_rag.retrieval.router import detect_sources
from annual_report_rag.retrieval.store import ChromaStore

_BRACKET_GROUP_RE = re.compile(r"\[(\s*\d+(?:\s*,\s*\d+)*\s*)\]")


@dataclass(frozen=True)
class QueryDeps:
    """Holds shared dependencies required to execute a query."""

    settings: Settings
    store: ChromaStore
    embedder: EmbeddingProvider
    generator: GenerationProvider


@dataclass(frozen=True)
class Citation:
    """Represents a validated citation pointing to a specific retrieved chunk (primary or neighbour)."""

    n: int
    chunk_id: str
    source: str
    display_name: str
    page: int
    text_quality: str
    excerpt: str
    neighbor_of: str | None = None


@dataclass(frozen=True)
class Timings:
    """Holds execution latency breakdown in milliseconds."""

    retrieve_ms: float
    generate_ms: float
    total_ms: float


@dataclass(frozen=True)
class AnswerResult:
    """Encapsulates the generated answer, grounding status, citations, and diagnostics."""

    answer: str
    grounded: bool
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    detected_sources: list[str]
    invalid_citations: int
    unsupported_numbers: list[str]
    timings: Timings
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reason: str | None = None


def extract_citation_numbers(text: str) -> list[int]:
    """Extracts all integer citation references from bracket patterns like [1], [1][2], or [1, 2]."""
    if not text:
        return []
    numbers: list[int] = []
    for group in _BRACKET_GROUP_RE.findall(text):
        for part in group.split(","):
            stripped = part.strip()
            if stripped.isdigit():
                numbers.append(int(stripped))
    return numbers


def resolve_citations(
    raw_numbers: Sequence[int],
    chunks: Sequence[RetrievedChunk],
    settings: Settings,
) -> tuple[list[Citation], list[RetrievedChunk], int]:
    """Filters extracted citation numbers against available chunks and counts invalid references."""
    max_idx = len(chunks)
    valid_nums: list[int] = []
    invalid_count = 0

    for num in raw_numbers:
        if 1 <= num <= max_idx:
            if num not in valid_nums:
                valid_nums.append(num)
        else:
            invalid_count += 1

    citations: list[Citation] = []
    cited_chunks: list[RetrievedChunk] = []
    for num in valid_nums:
        chunk = chunks[num - 1]
        cited_chunks.append(chunk)
        citations.append(
            Citation(
                n=num,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                display_name=get_display_name(chunk.source, settings),
                page=chunk.page,
                text_quality=chunk.text_quality,
                excerpt=chunk.text[:300],
                neighbor_of=chunk.neighbor_of,
            )
        )

    return citations, cited_chunks, invalid_count


def answer_question(
    question: str,
    deps: QueryDeps,
    top_k: int | None = None,
    sources: list[str] | None = None,
) -> AnswerResult:
    """Executes the full RAG query flow: route, retrieve, prompt, generate, and verify citations."""
    t_start = time.perf_counter()
    cleaned_question = question.strip()

    detected = (
        list(sources)
        if sources is not None
        else detect_sources(cleaned_question, deps.settings)
    )

    t_ret0 = time.perf_counter()
    chunks = retrieve(
        question=cleaned_question,
        embedder=deps.embedder,
        store=deps.store,
        settings=deps.settings,
        top_k=top_k,
        sources=detected,
    )
    retrieve_ms = round((time.perf_counter() - t_ret0) * 1000.0, 2)

    if not chunks:
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        return AnswerResult(
            answer=NOT_FOUND_MESSAGE,
            grounded=False,
            citations=[],
            retrieved=[],
            detected_sources=detected,
            invalid_citations=0,
            unsupported_numbers=[],
            timings=Timings(
                retrieve_ms=retrieve_ms,
                generate_ms=0.0,
                total_ms=total_ms,
            ),
            model=deps.settings.generation.model,
            input_tokens=None,
            output_tokens=None,
            reason="no_context",
        )

    system_prompt, user_prompt = build_prompt(cleaned_question, chunks, deps.settings)

    t_gen0 = time.perf_counter()
    gen_result = deps.generator.generate(user_prompt, system=system_prompt)
    generate_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)

    answer_text = gen_result.text.strip()
    raw_nums = extract_citation_numbers(answer_text)
    citations, cited_chunks, invalid_count = resolve_citations(
        raw_nums, chunks, deps.settings
    )

    is_not_found = answer_text == NOT_FOUND_MESSAGE
    grounded = len(citations) > 0 and not is_not_found
    if is_not_found:
        reason = "not_found"
    elif not citations:
        reason = "no_valid_citations"
    else:
        reason = "ok"

    unsupported = (
        find_unsupported_numbers(answer_text, cited_chunks) if cited_chunks else []
    )
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return AnswerResult(
        answer=answer_text,
        grounded=grounded,
        citations=citations,
        retrieved=chunks,
        detected_sources=detected,
        invalid_citations=invalid_count,
        unsupported_numbers=unsupported,
        timings=Timings(
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
        ),
        model=deps.settings.generation.model,
        input_tokens=gen_result.input_tokens,
        output_tokens=gen_result.output_tokens,
        reason=reason,
    )
