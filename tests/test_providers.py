"""Unit tests for model providers and provider factory."""

import math

import pytest

from annual_report_rag.config import (
    IngestionConfig,
    ProviderConfig,
    RetrievalConfig,
    RetryConfig,
    Settings,
    StorageConfig,
)
from annual_report_rag.providers.base import BaseProvider
from annual_report_rag.providers.factory import get_provider
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.providers.gemini import (
    GeminiProvider,
    l2_normalize,
    map_purpose_to_task_type,
)


def _build_dummy_settings(
    provider_name: str,
    api_key: str | None = None,
) -> Settings:
    """Helper function to construct a minimal Settings object for testing."""
    return Settings(
        provider=ProviderConfig(
            name=provider_name,
            embedding_model="gemini-embedding-001",
            generation_model="gemini-3.5-flash-lite",
            embedding_dimension=128,
            batch_size=25,
            retry=RetryConfig(
                max_retries=3, initial_wait_seconds=1.0, max_wait_seconds=5.0
            ),
        ),
        ingestion=IngestionConfig(
            chunk_size=1000, chunk_overlap=200, min_page_chars=50
        ),
        retrieval=RetrievalConfig(top_k=5),
        storage=StorageConfig(
            vector_store_path="chroma_db",
            collection_name="test_col",
            raw_data_dir="data/raw",
            eval_data_dir="data/eval",
        ),
        gemini_api_key=api_key,
    )


def test_fake_provider_embedding_and_purpose() -> None:
    """Verifies that FakeProvider handles purpose parameters and returns normalized vectors."""
    provider = FakeProvider(embedding_dim=64)
    texts = ["First test sentence.", "Second sentence."]

    vectors = provider.embed(texts, purpose="document")

    assert len(vectors) == 2
    assert len(vectors[0]) == 64
    assert len(vectors[1]) == 64
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert pytest.approx(norm, 0.001) == 1.0
    assert provider.embedded_purposes == ["document"]


def test_fake_provider_empty_embedding() -> None:
    """Verifies that FakeProvider handles empty input gracefully."""
    provider = FakeProvider()
    vectors = provider.embed([])

    assert vectors == []


def test_fake_provider_generate() -> None:
    """Verifies that FakeProvider returns configured responses and records prompts."""
    expected_answer = "Audited financial disclosure from page 10."
    provider = FakeProvider(default_response=expected_answer)

    result = provider.generate("What was FY23 revenue?")

    assert result == expected_answer
    assert provider.generated_prompts == ["What was FY23 revenue?"]


def test_factory_returns_fake_provider() -> None:
    """Verifies that factory returns a FakeProvider when configured in settings."""
    settings = _build_dummy_settings(provider_name="fake")

    provider = get_provider(settings)

    assert isinstance(provider, BaseProvider)
    assert isinstance(provider, FakeProvider)
    assert provider.embedding_dim == 128


def test_factory_unsupported_provider_raises_error() -> None:
    """Verifies that factory raises ValueError when an unknown provider name is specified."""
    settings = _build_dummy_settings(provider_name="unsupported_provider")

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider(settings)


def test_gemini_provider_missing_key_raises_error() -> None:
    """Verifies that GeminiProvider raises ValueError if initialized without an API key."""
    with pytest.raises(ValueError, match="Gemini API key is missing"):
        GeminiProvider(
            api_key=None,
            embedding_model="gemini-embedding-001",
            generation_model="gemini-3.5-flash-lite",
        )


def test_map_purpose_to_task_type() -> None:
    """Verifies correct mapping from purpose string to Google GenAI task types."""
    assert map_purpose_to_task_type("document") == "RETRIEVAL_DOCUMENT"
    assert map_purpose_to_task_type("query") == "RETRIEVAL_QUERY"
    with pytest.raises(ValueError, match="Unknown purpose"):
        map_purpose_to_task_type("invalid_purpose")


def test_l2_normalize() -> None:
    """Verifies that l2_normalize correctly scales vectors to unit length."""
    vector = [3.0, 4.0]
    normalized = l2_normalize(vector)
    assert pytest.approx(normalized[0], 0.001) == 0.6
    assert pytest.approx(normalized[1], 0.001) == 0.8
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]
