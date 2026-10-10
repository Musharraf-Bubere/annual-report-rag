"""FastAPI integration tests using TestClient, dependency overrides, and a real temporary ChromaStore."""

from pathlib import Path

from fastapi.testclient import TestClient

from annual_report_rag.api.main import ServiceContainer, app, get_container
from annual_report_rag.ingestion.chunker import Chunk
from annual_report_rag.providers.base import (
    DailyQuotaExceeded,
    GenerationProvider,
    GenerationResult,
    ProviderError,
)
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.retrieval.store import ChromaStore
from tests.test_router_and_retriever import _build_test_settings


def _build_populated_container(
    tmp_path: Path,
    response_text: str = "TCS revenue was ₹2,67,021 crore [1].",
) -> ServiceContainer:
    """Creates a ServiceContainer backed by a real temporary ChromaStore and FakeProviders."""
    db_path = str(tmp_path / "api_chroma")
    store = ChromaStore(persist_path=db_path, collection_name="api_col")
    embedder = FakeProvider(embedding_dim=4)
    generator = FakeProvider(embedding_dim=4, default_response=response_text)
    settings = _build_test_settings(min_score=0.0, top_k=2)

    q_vec = embedder.embed(["What was TCS revenue?"], purpose="query")[0]
    store.add_chunks(
        [
            Chunk(
                id="tcs-p8-c0",
                text="Consolidated revenue in FY26 was ₹2,67,021 crore.",
                source="tcs-ar-26",
                page=8,
                chunk_index=0,
            ),
            Chunk(
                id="infy-p10-c0",
                text="Infosys revenue grew by 4.2%.",
                source="infosys-ar-26",
                page=10,
                chunk_index=0,
            ),
        ],
        [q_vec, q_vec],
    )

    return ServiceContainer(
        settings=settings,
        store=store,
        embedder=embedder,
        generator=generator,
        startup_error=None,
    )


def test_api_health_sources_and_ask_ok(tmp_path: Path) -> None:
    """Verifies /health, /sources, and /ask happy paths with a real temporary ChromaStore."""
    container = _build_populated_container(tmp_path)
    app.dependency_overrides[get_container] = lambda: container
    try:
        client = TestClient(app)

        # 1. GET /health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["ready"] is True
        assert health_data["chunk_count"] == 2
        assert health_data["embedding_provider"] == "fake"
        assert health_data["generation_provider"] == "fake"

        # 2. GET /sources
        r_sources = client.get("/sources")
        assert r_sources.status_code == 200
        sources_payload = r_sources.json()
        assert sources_payload["total_chunks"] == 2
        assert sources_payload["sources"] == {"infosys-ar-26": 1, "tcs-ar-26": 1}

        # 3. POST /ask
        r_ask = client.post("/ask", json={"question": "What was TCS revenue?"})
        assert r_ask.status_code == 200
        ask_data = r_ask.json()
        assert ask_data["grounded"] is True
        assert ask_data["reason"] == "ok"
        assert ask_data["detected_sources"] == ["tcs-ar-26"]
        assert len(ask_data["citations"]) == 1
        assert ask_data["citations"][0]["page"] == 8
        assert ask_data["unsupported_numbers"] == []
    finally:
        app.dependency_overrides.clear()


def test_api_validation_422_empty_and_too_long(tmp_path: Path) -> None:
    """Verifies 422 Unprocessable Entity on empty, whitespace-only, or overly long questions."""
    container = _build_populated_container(tmp_path)
    app.dependency_overrides[get_container] = lambda: container
    try:
        client = TestClient(app)

        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={"question": "   \n\t  "}).status_code == 422
        assert client.post("/ask", json={"question": "a" * 1001}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_api_empty_store_and_provider_errors_503(tmp_path: Path) -> None:
    """Verifies 503 Service Unavailable on empty store, DailyQuotaExceeded, and ProviderError."""
    # 1. Empty store -> /health ready=False and /ask 503
    empty_db = str(tmp_path / "empty_api_chroma")
    empty_store = ChromaStore(persist_path=empty_db, collection_name="empty_col")
    embedder = FakeProvider(embedding_dim=4)
    generator = FakeProvider(embedding_dim=4)
    settings = _build_test_settings()
    empty_container = ServiceContainer(
        settings=settings,
        store=empty_store,
        embedder=embedder,
        generator=generator,
        startup_error=None,
    )

    app.dependency_overrides[get_container] = lambda: empty_container
    try:
        client = TestClient(app)
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["ready"] is False
        assert r_health.json()["chunk_count"] == 0

        r_ask = client.post("/ask", json={"question": "What was TCS revenue?"})
        assert r_ask.status_code == 503
        assert "empty" in r_ask.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()

    # 2. DailyQuotaExceeded and ProviderError -> 503
    class QuotaFailingGenerator(GenerationProvider):
        def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
            raise DailyQuotaExceeded(
                "Daily API quota exceeded (429 PerDay).",
                retry_delay=3600.0,
            )

    class UpstreamFailingGenerator(GenerationProvider):
        def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
            raise ProviderError("Gemini provider error (RuntimeError): 503 Service Unavailable")

    base_container = _build_populated_container(tmp_path)

    quota_container = ServiceContainer(
        settings=base_container.settings,
        store=base_container.store,
        embedder=base_container.embedder,
        generator=QuotaFailingGenerator(),
    )
    app.dependency_overrides[get_container] = lambda: quota_container
    try:
        client = TestClient(app)
        r_quota = client.post("/ask", json={"question": "What was TCS revenue?"})
        assert r_quota.status_code == 503
        assert "DailyQuotaExceeded" in r_quota.json()["detail"]
    finally:
        app.dependency_overrides.clear()

    err_container = ServiceContainer(
        settings=base_container.settings,
        store=base_container.store,
        embedder=base_container.embedder,
        generator=UpstreamFailingGenerator(),
    )
    app.dependency_overrides[get_container] = lambda: err_container
    try:
        client = TestClient(app)
        r_err = client.post("/ask", json={"question": "What was TCS revenue?"})
        assert r_err.status_code == 503
        assert "Gemini provider error" in r_err.json()["detail"]
    finally:
        app.dependency_overrides.clear()
