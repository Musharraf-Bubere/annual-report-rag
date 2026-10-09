"""Provider factory for instantiating the configured model provider."""

from annual_report_rag.config import Settings
from annual_report_rag.providers.base import BaseProvider
from annual_report_rag.providers.fake import FakeProvider
from annual_report_rag.providers.gemini import GeminiProvider


def get_provider(settings: Settings) -> BaseProvider:
    """Instantiates and returns the provider specified in application settings."""
    provider_name = settings.provider.name.strip().lower()

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            embedding_model=settings.provider.embedding_model,
            generation_model=settings.provider.generation_model,
            embedding_dimension=settings.provider.embedding_dimension,
            max_items_per_minute=settings.provider.max_items_per_minute,
            max_retries=settings.provider.retry.max_retries,
            max_rate_limit_retries=settings.provider.retry.max_rate_limit_retries,
            initial_wait_seconds=settings.provider.retry.initial_wait_seconds,
            max_wait_seconds=settings.provider.retry.max_wait_seconds,
        )

    if provider_name == "fake":
        return FakeProvider(embedding_dim=settings.provider.embedding_dimension)

    raise ValueError(f"Unsupported provider: '{provider_name}'")
