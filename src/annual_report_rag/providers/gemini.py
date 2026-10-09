"""Gemini provider using the official google-genai SDK."""

import math
import time
from collections.abc import Callable
from typing import TypeVar

from google import genai
from google.genai import errors, types

from annual_report_rag.providers.base import BaseProvider

T = TypeVar("T")


def l2_normalize(vector: list[float]) -> list[float]:
    """Applies L2 normalization to scale a vector to unit Euclidean norm."""
    norm = math.sqrt(sum(val * val for val in vector))
    if norm == 0.0:
        return vector
    return [val / norm for val in vector]


def map_purpose_to_task_type(purpose: str) -> str:
    """Maps retrieval purpose ('document' or 'query') to Google GenAI embedding task type."""
    normalized = purpose.strip().lower()
    if normalized == "query":
        return "RETRIEVAL_QUERY"
    if normalized == "document":
        return "RETRIEVAL_DOCUMENT"
    raise ValueError(f"Unknown purpose: '{purpose}'. Expected 'document' or 'query'.")


def is_retryable_error(exc: Exception) -> bool:
    """Checks whether an exception corresponds to a retryable 429 or 5xx API error."""
    if isinstance(exc, errors.APIError) and exc.code is not None:
        return exc.code == 429 or 500 <= exc.code < 600
    return False


class GeminiProvider(BaseProvider):
    """Integrates Google Gemini models for embeddings and text generation."""

    def __init__(
        self,
        api_key: str | None,
        embedding_model: str,
        generation_model: str,
        embedding_dimension: int = 768,
        max_retries: int = 3,
        initial_wait_seconds: float = 1.0,
        max_wait_seconds: float = 10.0,
    ) -> None:
        """Initializes the Gemini client, model parameters, and retry policies."""
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY in .env or environment."
            )
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.embedding_dimension = embedding_dimension
        self.max_retries = max_retries
        self.initial_wait_seconds = initial_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.client = genai.Client(api_key=api_key)

    def _execute_with_retry(self, func: Callable[[], T]) -> T:
        """Executes a function with exponential backoff on retryable API errors."""
        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except Exception as exc:
                if attempt == self.max_retries or not is_retryable_error(exc):
                    raise
                delay = min(
                    self.max_wait_seconds,
                    self.initial_wait_seconds * (2**attempt),
                )
                time.sleep(delay)
        raise RuntimeError("Retry loop exited unexpectedly without result.")

    def embed(
        self,
        texts: list[str],
        purpose: str = "document",
    ) -> list[list[float]]:
        """Generates L2-normalized vector embeddings using configured task types."""
        if not texts:
            return []

        task_type = map_purpose_to_task_type(purpose)
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.embedding_dimension,
        )

        response = self._execute_with_retry(
            lambda: self.client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config=config,
            )
        )

        if not response.embeddings:
            return []

        return [
            l2_normalize(emb.values)
            for emb in response.embeddings
            if emb.values is not None
        ]

    def generate(self, prompt: str) -> str:
        """Generates a text completion for the prompt using configured generation model."""
        response = self._execute_with_retry(
            lambda: self.client.models.generate_content(
                model=self.generation_model,
                contents=prompt,
            )
        )
        return response.text or ""
