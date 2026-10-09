"""Gemini provider using the official google-genai SDK."""

import math
import re
import time
from collections.abc import Callable
from typing import TypeVar

from google import genai
from google.genai import errors, types

from annual_report_rag.providers.base import BaseProvider
from annual_report_rag.providers.rate_limiter import SlidingWindowRateLimiter

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


def extract_retry_delay(exc: Exception, default_delay: float = 27.0) -> float:
    """Extracts suggested retry delay in seconds from 429 error message or details."""
    msg = str(exc)
    match = re.search(
        r"retry\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*s?",
        msg,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return default_delay


def is_rate_limit_error(exc: Exception) -> bool:
    """Checks whether an exception indicates HTTP 429 rate limit exhaustion."""
    if isinstance(exc, errors.APIError) and exc.code == 429:
        return True
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


def is_server_error(exc: Exception) -> bool:
    """Checks whether an exception indicates a transient 5xx server error."""
    if isinstance(exc, errors.APIError) and exc.code is not None:
        return 500 <= exc.code < 600
    return False


class GeminiProvider(BaseProvider):
    """Integrates Google Gemini models for embeddings and text generation."""

    def __init__(
        self,
        api_key: str | None,
        embedding_model: str,
        generation_model: str,
        embedding_dimension: int = 768,
        max_items_per_minute: int = 80,
        max_retries: int = 3,
        max_rate_limit_retries: int = 5,
        initial_wait_seconds: float = 1.0,
        max_wait_seconds: float = 60.0,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initializes the Gemini client, rate limiter, and retry policies."""
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY in .env or environment."
            )
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.embedding_dimension = embedding_dimension
        self.max_retries = max_retries
        self.max_rate_limit_retries = max_rate_limit_retries
        self.initial_wait_seconds = initial_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.sleep_fn = sleep_fn
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter(
            max_items_per_minute=max_items_per_minute,
            sleep_fn=sleep_fn,
        )
        self.client = genai.Client(api_key=api_key)

    def _execute_with_retry(self, func: Callable[[], T]) -> T:
        """Executes a function with backoff retry handling 429 quotas and 5xx errors."""
        rate_limit_attempts = 0
        server_attempts = 0

        while True:
            try:
                return func()
            except Exception as exc:
                if is_rate_limit_error(exc):
                    if rate_limit_attempts >= self.max_rate_limit_retries:
                        raise
                    rate_limit_attempts += 1
                    delay = extract_retry_delay(exc) + 1.0
                    print(
                        f"Rate limit (429) hit. Waiting {delay:.1f}s before retry "
                        f"({rate_limit_attempts}/{self.max_rate_limit_retries})..."
                    )
                    self.sleep_fn(delay)
                    continue

                if is_server_error(exc):
                    if server_attempts >= self.max_retries:
                        raise
                    server_attempts += 1
                    delay = min(
                        self.max_wait_seconds,
                        self.initial_wait_seconds * (2 ** (server_attempts - 1)),
                    )
                    self.sleep_fn(delay)
                    continue

                raise

    def embed(
        self,
        texts: list[str],
        purpose: str = "document",
    ) -> list[list[float]]:
        """Generates L2-normalized vector embeddings using rate pacing and task types."""
        if not texts:
            return []

        self.rate_limiter.acquire(len(texts))

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
