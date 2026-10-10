"""Unit tests for model providers, local FastEmbed provider, and provider factories."""

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from annual_report_rag.config import (
    EmbeddingConfig,
    GenerationConfig,
    IngestionConfig,
    QualityConfig,
    RetrievalConfig,
    RetryConfig,
    Settings,
    StorageConfig,
)
from annual_report_rag.providers.base import (
    BaseProvider,
    EmbeddingProvider,
    GenerationProvider,
)
from annual_report_rag.providers.factory import (
    get_embedder,
    get_generator,
    get_provider,
)
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.providers.gemini import (
    GeminiProvider,
    l2_normalize,
    map_purpose_to_task_type,
)
from annual_report_rag.providers.local_embedder import LocalEmbeddingProvider


def _build_dummy_settings(
    embedding_provider: str = "fake",
    generation_provider: str = "fake",
    api_key: str | None = None,
) -> Settings:
    """Helper function to construct a minimal Settings object for testing."""
    return Settings(
        embedding=EmbeddingConfig(
            provider=embedding_provider,
            model="BAAI/bge-base-en-v1.5" if embedding_provider == "local" else "test-embed",
            dimension=128,
            batch_size=25,
            max_items_per_minute=80,
            retry=RetryConfig(
                max_retries=3,
                max_rate_limit_retries=5,
                initial_wait_seconds=1.0,
                max_wait_seconds=5.0,
            ),
        ),
        generation=GenerationConfig(
            provider=generation_provider,
            model="gemini-3.5-flash-lite" if generation_provider == "gemini" else "test-gen",
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
        quality=QualityConfig(
            single_letter_ratio_threshold=0.15,
            odd_char_ratio_threshold=0.05,
            odd_char_count_threshold=1,
        ),
        gemini_api_key=api_key,
    )


def test_fake_provider_implements_interfaces() -> None:
    """Verifies that FakeProvider implements both EmbeddingProvider and GenerationProvider."""
    provider = FakeProvider(embedding_dim=64)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, GenerationProvider)
    assert isinstance(provider, BaseProvider)


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


def test_factory_get_embedder() -> None:
    """Verifies factory get_embedder for local and fake providers."""
    fake_settings = _build_dummy_settings(embedding_provider="fake")
    fake_embedder = get_embedder(fake_settings)
    assert isinstance(fake_embedder, EmbeddingProvider)
    assert isinstance(fake_embedder, FakeProvider)

    local_settings = _build_dummy_settings(embedding_provider="local")
    local_embedder = get_embedder(local_settings)
    assert isinstance(local_embedder, EmbeddingProvider)
    assert isinstance(local_embedder, LocalEmbeddingProvider)
    assert local_embedder.embedding_dimension == 128


def test_factory_get_generator() -> None:
    """Verifies factory get_generator for fake and gemini providers."""
    fake_settings = _build_dummy_settings(generation_provider="fake")
    fake_gen = get_generator(fake_settings)
    assert isinstance(fake_gen, GenerationProvider)
    assert isinstance(fake_gen, FakeProvider)

    gemini_settings = _build_dummy_settings(
        generation_provider="gemini", api_key="test_key"
    )
    gemini_gen = get_generator(gemini_settings)
    assert isinstance(gemini_gen, GenerationProvider)
    assert isinstance(gemini_gen, GeminiProvider)


def test_factory_get_provider_backward_compatibility() -> None:
    """Verifies that get_provider returns a composite provider for backward compatibility."""
    fake_settings = _build_dummy_settings(embedding_provider="fake")
    provider = get_provider(fake_settings)
    assert isinstance(provider, BaseProvider)


def test_factory_unsupported_embedder_raises() -> None:
    """Verifies factory raises ValueError for unsupported embedding provider."""
    settings = _build_dummy_settings(embedding_provider="unknown_embedder")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedder(settings)


def test_factory_unsupported_generator_raises() -> None:
    """Verifies factory raises ValueError for unsupported generation provider."""
    settings = _build_dummy_settings(generation_provider="unknown_gen")
    with pytest.raises(ValueError, match="Unsupported generation provider"):
        get_generator(settings)


def test_local_embedder_mocked_methods_and_normalization() -> None:
    """Verifies LocalEmbeddingProvider query vs passage purpose mapping and normalization."""
    mock_model = MagicMock()
    mock_model.passage_embed.return_value = iter([[3.0, 4.0]])
    mock_model.query_embed.return_value = iter([[1.0, 0.0]])

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        embedding_dimension=2,
        model=mock_model,
    )

    # Empty inputs
    assert provider.embed([]) == []

    # Passage purpose
    doc_vectors = provider.embed(["Sample document passage"], purpose="document")
    mock_model.passage_embed.assert_called_once_with(["Sample document passage"])
    assert len(doc_vectors) == 1
    assert pytest.approx(doc_vectors[0][0], 0.001) == 0.6
    assert pytest.approx(doc_vectors[0][1], 0.001) == 0.8

    # Query purpose
    query_vectors = provider.embed(["Sample query"], purpose="query")
    mock_model.query_embed.assert_called_once_with(["Sample query"])
    assert len(query_vectors) == 1
    assert query_vectors[0] == [1.0, 0.0]

    # Invalid purpose
    with pytest.raises(ValueError, match="Unknown purpose"):
        provider.embed(["test"], purpose="invalid")


def test_local_embedder_count_and_dimension_validation() -> None:
    """Verifies that LocalEmbeddingProvider validates return count and dimensions."""
    mock_model = MagicMock()
    provider = LocalEmbeddingProvider(
        model_name="test-model",
        embedding_dimension=3,
        model=mock_model,
    )

    # Count mismatch
    mock_model.passage_embed.return_value = iter([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Expected 2 embeddings from local embedder, but received 1"):
        provider.embed(["text1", "text2"])

    # Dimension mismatch
    mock_model.passage_embed.return_value = iter([[1.0, 2.0]])
    with pytest.raises(ValueError, match="dimension 2, expected 3"):
        provider.embed(["text1"])


@pytest.mark.slow
def test_local_embedder_real_model_loading() -> None:
    """Verifies real FastEmbed model download and execution. Skipped by default via -m 'not slow'."""
    provider = LocalEmbeddingProvider(
        model_name="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
    )
    vectors = provider.embed(["Test sentence for local ONNX embedding"], purpose="query")
    assert len(vectors) == 1
    assert len(vectors[0]) == 768
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert pytest.approx(norm, 0.001) == 1.0


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


def test_gemini_embed_validation_length_mismatch() -> None:
    """Verifies that GeminiProvider raises ValueError when returned embeddings count differs from input."""
    provider = GeminiProvider(
        api_key="mock_key",
        embedding_model="gemini-embedding-001",
        generation_model="gemini-3.5-flash-lite",
        embedding_dimension=768,
    )

    class MockEmbedding:
        def __init__(self, values: list[float]) -> None:
            self.values = values

    class MockResponse:
        def __init__(self, embeddings: list[Any]) -> None:
            self.embeddings = embeddings

    provider.client.models.embed_content = lambda **kwargs: MockResponse(
        embeddings=[MockEmbedding(values=[0.1] * 768)]
    )

    with pytest.raises(ValueError, match="Expected 2 embeddings from Gemini, but received 1"):
        provider.embed(["text1", "text2"])


def test_gemini_embed_validation_dimension_mismatch() -> None:
    """Verifies that GeminiProvider raises ValueError when embedding dimension does not match settings."""
    provider = GeminiProvider(
        api_key="mock_key",
        embedding_model="gemini-embedding-001",
        generation_model="gemini-3.5-flash-lite",
        embedding_dimension=768,
    )

    class MockEmbedding:
        def __init__(self, values: list[float]) -> None:
            self.values = values

    class MockResponse:
        def __init__(self, embeddings: list[Any]) -> None:
            self.embeddings = embeddings

    provider.client.models.embed_content = lambda **kwargs: MockResponse(
        embeddings=[MockEmbedding(values=[0.1] * 128)]
    )

    with pytest.raises(ValueError, match="dimension 128, expected 768"):
        provider.embed(["text1"])
