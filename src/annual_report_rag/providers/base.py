"""Abstract base provider interface for embedding and text generation."""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Defines a unified contract for embedding and generation model providers."""

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        purpose: str = "document",
    ) -> list[list[float]]:
        """Converts a batch of texts into high-dimensional vector embeddings."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a text completion for a given user prompt."""
