"""FastAPI application serving grounded annual report question answering."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from annual_report_rag.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
    RetrievedChunkOut,
    SourcesResponse,
    TimingsOut,
)
from annual_report_rag.config import Settings, load_settings
from annual_report_rag.generation.answerer import QueryDeps, answer_question
from annual_report_rag.providers.base import (
    DailyQuotaExceeded,
    EmbeddingProvider,
    GenerationProvider,
    ProviderError,
)
from annual_report_rag.providers.factory import get_embedder, get_generator
from annual_report_rag.retrieval.store import ChromaStore

logger = logging.getLogger("annual_report_rag.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ServiceContainer:
    """Holds singleton service instances initialized once at application startup."""

    settings: Settings
    store: ChromaStore | None = None
    embedder: EmbeddingProvider | None = None
    generator: GenerationProvider | None = None
    startup_error: str | None = None

    def check_readiness(self) -> tuple[bool, int, str | None]:
        """Evaluates whether the vector store and model providers are ready to serve requests."""
        if self.store is None:
            detail = (
                self.startup_error
                or "Vector store is unavailable. Please run scripts/ingest.py."
            )
            return False, 0, detail

        try:
            count = self.store.count()
        except Exception as exc:  # noqa: BLE001
            return False, 0, f"Store error ({type(exc).__name__}): {str(exc)[:300]}"

        if count == 0:
            return (
                False,
                0,
                "Vector store is empty. Please run scripts/ingest.py to index documents.",
            )

        if self.startup_error:
            return False, count, self.startup_error

        if self.embedder is None or self.generator is None:
            return False, count, "Model providers are not initialized."

        return True, count, None


def build_service_container(settings: Settings | None = None) -> ServiceContainer:
    """Initializes settings, ChromaStore, embedder, and generator for the application lifespan."""
    cfg = settings or load_settings()
    container = ServiceContainer(settings=cfg)

    try:
        container.store = ChromaStore(
            persist_path=cfg.storage.vector_store_path,
            collection_name=cfg.storage.collection_name,
            embedding_provider=cfg.embedding.provider,
            embedding_model=cfg.embedding.model,
            embedding_dimension=cfg.embedding.dimension,
        )
    except Exception as exc:  # noqa: BLE001
        container.startup_error = (
            f"Vector store initialization failed ({type(exc).__name__}): "
            f"{str(exc)[:300]}. Please run scripts/ingest.py."
        )
        return container

    try:
        container.embedder = get_embedder(cfg)
        container.generator = get_generator(cfg)
    except Exception as exc:  # noqa: BLE001
        container.startup_error = (
            f"Provider initialization failed ({type(exc).__name__}): {str(exc)[:300]}"
        )

    return container


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager creating singleton clients once at startup."""
    application.state.container = build_service_container()
    yield


def get_container(request: Request) -> ServiceContainer:
    """Dependency returning the startup ServiceContainer instance."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        container = build_service_container()
        request.app.state.container = container
    return container


ContainerDep = Annotated[ServiceContainer, Depends(get_container)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Constructs and configures the FastAPI application with CORS and routes."""
    cfg = settings or load_settings()
    application = FastAPI(
        title="Annual Report RAG API",
        description=(
            "Question answering over corporate annual reports with PDF page citations "
            "(PDF page N) and grounded verification."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.api.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", response_model=HealthResponse)
    def health(container: ContainerDep) -> HealthResponse:
        """Reports store chunk count, model names, and service readiness without exposing secrets."""
        ready, count, detail = container.check_readiness()
        s = container.settings
        return HealthResponse(
            ready=ready,
            chunk_count=count,
            embedding_provider=s.embedding.provider,
            embedding_model=s.embedding.model,
            generation_provider=s.generation.provider,
            generation_model=s.generation.model,
            detail=detail,
        )

    @application.get("/sources", response_model=SourcesResponse)
    def sources(container: ContainerDep) -> SourcesResponse:
        """Returns the number of indexed chunks per document source."""
        if container.store is None:
            return SourcesResponse(sources={}, total_chunks=0)
        counts = container.store.source_counts()
        return SourcesResponse(
            sources=counts,
            total_chunks=sum(counts.values()),
        )

    @application.post("/ask", response_model=AskResponse)
    def ask(
        payload: AskRequest,
        container: ContainerDep,
    ) -> AskResponse:
        """Answers a question grounded in indexed annual report passages with PDF page citations."""
        max_chars = container.settings.retrieval.max_question_chars
        if len(payload.question) > max_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Question length ({len(payload.question)}) exceeds max_question_chars ({max_chars}).",
            )

        ready, _count, detail = container.check_readiness()
        if not ready or container.store is None or container.embedder is None or container.generator is None:
            raise HTTPException(
                status_code=503,
                detail=detail or "Service not ready. Please run scripts/ingest.py.",
            )

        deps = QueryDeps(
            settings=container.settings,
            store=container.store,
            embedder=container.embedder,
            generator=container.generator,
        )

        try:
            result = answer_question(
                question=payload.question,
                deps=deps,
                top_k=payload.top_k,
                sources=payload.sources,
            )
        except (DailyQuotaExceeded, ProviderError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"{type(exc).__name__}: {str(exc)[:300]}",
            ) from exc

        top_score = result.retrieved[0].score if result.retrieved else 0.0
        logger.info(
            "ask question_len=%d detected_sources=%s top_score=%.4f retrieve_ms=%.2f generate_ms=%.2f grounded=%s",
            len(payload.question),
            result.detected_sources,
            top_score,
            result.timings.retrieve_ms,
            result.timings.generate_ms,
            result.grounded,
        )

        return AskResponse(
            answer=result.answer,
            grounded=result.grounded,
            citations=[CitationOut(**asdict(c)) for c in result.citations],
            retrieved=[
                RetrievedChunkOut(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    source=r.source,
                    page=r.page,
                    score=r.score,
                    text_quality=r.text_quality,
                )
                for r in result.retrieved
            ],
            detected_sources=result.detected_sources,
            invalid_citations=result.invalid_citations,
            unsupported_numbers=result.unsupported_numbers,
            timings=TimingsOut(**asdict(result.timings)),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            reason=result.reason,
        )

    return application


app = create_app()
