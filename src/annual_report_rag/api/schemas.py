"""Pydantic request and response schemas for the FastAPI service."""

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Request payload for asking a question against indexed annual reports."""

    question: str = Field(
        ...,
        description="Natural language question (1 to max_question_chars characters after stripping whitespace).",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Optional override for number of chunks to retrieve per company source.",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Optional explicit list of source stems (e.g. ['tcs-ar-26']) to filter retrieval.",
    )

    @field_validator("question")
    @classmethod
    def validate_question_non_empty(cls, value: str) -> str:
        """Strips whitespace and ensures question is non-empty."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be empty or whitespace only.")
        return stripped


class CitationOut(BaseModel):
    """Citation reference pointing to a retrieved context passage."""

    n: int = Field(..., description="1-based citation index matching [n] in the answer.")
    chunk_id: str = Field(..., description="Deterministic chunk identifier.")
    source: str = Field(..., description="Source PDF stem (e.g. 'tcs-ar-26').")
    display_name: str = Field(..., description="Human-readable report title.")
    page: int = Field(
        ...,
        description="PDF page N (1-indexed PDF page number, which may differ from printed report page numbers).",
    )
    text_quality: str = Field(
        ...,
        description="Page text quality state ('clean', 'repaired', or 'degraded').",
    )
    excerpt: str = Field(
        ...,
        description="First 300 characters of the cited chunk text.",
    )


class RetrievedChunkOut(BaseModel):
    """Retrieved passage with similarity score and provenance metadata."""

    chunk_id: str = Field(..., description="Deterministic chunk identifier.")
    text: str = Field(..., description="Full chunk text.")
    source: str = Field(..., description="Source PDF stem.")
    page: int = Field(
        ...,
        description="PDF page N (1-indexed PDF page number, which may differ from printed report page numbers).",
    )
    score: float = Field(..., description="Cosine similarity score.")
    text_quality: str = Field(
        ...,
        description="Page text quality state ('clean', 'repaired', or 'degraded').",
    )


class TimingsOut(BaseModel):
    """Execution latency breakdown in milliseconds."""

    retrieve_ms: float
    generate_ms: float
    total_ms: float


class AskResponse(BaseModel):
    """Response payload for POST /ask containing grounded answer, citations, and diagnostics."""

    answer: str
    grounded: bool
    citations: list[CitationOut]
    retrieved: list[RetrievedChunkOut]
    detected_sources: list[str]
    invalid_citations: int
    unsupported_numbers: list[str]
    timings: TimingsOut
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reason: str | None = None


class HealthResponse(BaseModel):
    """Response payload for GET /health reporting store readiness and model configuration."""

    ready: bool
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    generation_provider: str
    generation_model: str
    detail: str | None = None


class SourcesResponse(BaseModel):
    """Response payload for GET /sources reporting chunk counts per source stem."""

    sources: dict[str, int]
    total_chunks: int
