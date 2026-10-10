"""Provider factory for instantiating embedding and generation model providers."""

from annual_report_rag.config import Settings
from annual_report_rag.providers.base import (
    BaseProvider,
    EmbeddingProvider,
    GenerationProvider,
)
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.providers.gemini import GeminiProvider
from annual_report_rag.providers.local_embedder import LocalEmbeddingProvider


def get_embedder(settings: Settings) -> EmbeddingProvider:
    """Instantiates and returns the configured embedding provider."""
    provider_name = settings.embedding.provider.strip().lower()

    if provider_name == "local":
        return LocalEmbeddingProvider(
            model_name=settings.embedding.model,
            embedding_dimension=settings.embedding.dimension,
        )

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            embedding_model=settings.embedding.model,
            generation_model=settings.generation.model,
            embedding_dimension=settings.embedding.dimension,
            temperature=settings.generation.temperature,
            max_items_per_minute=settings.embedding.max_items_per_minute,
            max_retries=settings.embedding.retry.max_retries,
            max_rate_limit_retries=settings.embedding.retry.max_rate_limit_retries,
            initial_wait_seconds=settings.embedding.retry.initial_wait_seconds,
            max_wait_seconds=settings.embedding.retry.max_wait_seconds,
        )

    if provider_name == "fake":
        return FakeProvider(embedding_dim=settings.embedding.dimension)

    raise ValueError(f"Unsupported embedding provider: '{provider_name}'")


def get_generator(settings: Settings) -> GenerationProvider:
    """Instantiates and returns the configured generation provider."""
    provider_name = settings.generation.provider.strip().lower()

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            embedding_model=settings.embedding.model,
            generation_model=settings.generation.model,
            embedding_dimension=settings.embedding.dimension,
            temperature=settings.generation.temperature,
            max_items_per_minute=settings.embedding.max_items_per_minute,
            max_retries=settings.embedding.retry.max_retries,
            max_rate_limit_retries=settings.embedding.retry.max_rate_limit_retries,
            initial_wait_seconds=settings.embedding.retry.initial_wait_seconds,
            max_wait_seconds=settings.embedding.retry.max_wait_seconds,
        )

    if provider_name == "fake":
        return FakeProvider(embedding_dim=settings.embedding.dimension)

    raise ValueError(f"Unsupported generation provider: '{provider_name}'")


def get_provider(settings: Settings) -> BaseProvider:
    """Backward compatibility helper returning a composite provider."""
    provider_name = settings.embedding.provider.strip().lower()

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            embedding_model=settings.embedding.model,
            generation_model=settings.generation.model,
            embedding_dimension=settings.embedding.dimension,
            temperature=settings.generation.temperature,
            max_items_per_minute=settings.embedding.max_items_per_minute,
            max_retries=settings.embedding.retry.max_retries,
            max_rate_limit_retries=settings.embedding.retry.max_rate_limit_retries,
            initial_wait_seconds=settings.embedding.retry.initial_wait_seconds,
            max_wait_seconds=settings.embedding.retry.max_wait_seconds,
        )

    return FakeProvider(embedding_dim=settings.embedding.dimension)
