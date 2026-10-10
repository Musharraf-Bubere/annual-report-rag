"""Abstract base provider interface for embedding and text generation."""

from abc import ABC, abstractmethod


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
    def generate(self, prompt: str) -> str:
        """Generates a text completion for a given user prompt."""


class BaseProvider(EmbeddingProvider, GenerationProvider, ABC):
    """Unified composite interface supporting both embedding and generation."""
