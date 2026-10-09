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
        )

    if provider_name == "fake":
        return FakeProvider()

    raise ValueError(f"Unsupported provider: '{provider_name}'")
