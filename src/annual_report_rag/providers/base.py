"""Abstract base provider interface for embedding and text generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Raised when an embedding or generation provider fails with a non-quota error."""


class DailyQuotaExceeded(ProviderError):
    """Raised when the daily API quota has been exhausted."""

    def __init__(
        self,
        message: str,
        retry_delay: float | None = None,
        chunks_stored: int | None = None,
        chunks_remaining: int | None = None,
    ) -> None:
        """Initializes the quota exception with error message and optional progress state."""
        super().__init__(message)
        self.retry_delay = retry_delay
        self.chunks_stored = chunks_stored
        self.chunks_remaining = chunks_remaining


@dataclass(frozen=True)
class GenerationResult:
    """Represents a generated completion with optional token usage metadata."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        purpose: str = "document",
    ) -> list[list[float]]:
        """Converts a batch of texts into high-dimensional vector embeddings.

        Must return a list of lists containing standard Python float scalars
        (list[list[float]]), not numpy scalar types, to ensure compatibility
        across vector store backends.
        """


class GenerationProvider(ABC):
    """Abstract interface for text generation providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> GenerationResult:
        """Generates a text completion for a given user prompt and optional system instruction."""


class BaseProvider(EmbeddingProvider, GenerationProvider, ABC):
    """Unified composite interface supporting both embedding and generation."""
