"""Unit tests for model providers and provider factory."""

import pytest

from annual_report_rag.config import (
    IngestionConfig,
    ProviderConfig,
    RetrievalConfig,
    Settings,
    StorageConfig,
)
from annual_report_rag.providers.base import BaseProvider
from annual_report_rag.providers.factory import get_provider
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.providers.gemini import GeminiProvider


def _build_dummy_settings(
    provider_name: str,
    api_key: str | None = None,
) -> Settings:
    """Helper function to construct a minimal Settings object for testing."""
    return Settings(
        provider=ProviderConfig(
            name=provider_name,
            embedding_model="text-embedding-004",
            generation_model="gemini-2.5-flash",
        ),
        ingestion=IngestionConfig(chunk_size=1000, chunk_overlap=200),
        retrieval=RetrievalConfig(top_k=5),
        storage=StorageConfig(
            vector_store_path="chroma_db",
            raw_data_dir="data/raw",
            eval_data_dir="data/eval",
        ),
        gemini_api_key=api_key,
    )


def test_fake_provider_embedding() -> None:
    """Verifies that FakeProvider generates deterministic vectors with expected dimensions."""
    provider = FakeProvider(embedding_dim=128)
    texts = ["First test sentence.", "Second sentence."]

    vectors = provider.embed(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == 128
    assert len(vectors[1]) == 128
    assert provider.embedded_texts == texts


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
            embedding_model="text-embedding-004",
            generation_model="gemini-2.5-flash",
        )
