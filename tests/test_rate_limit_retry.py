"""Unit tests for 429 rate limit error parsing and retry backoff behavior."""

import pytest

from annual_report_rag.providers.gemini import (
    GeminiProvider,
    extract_retry_delay,
)
from annual_report_rag.providers.rate_limiter import SlidingWindowRateLimiter


def test_extract_retry_delay() -> None:
    """Verifies parsing of retry duration recommendations from error messages."""
    exc1 = RuntimeError("RESOURCE_EXHAUSTED: Quota exceeded. Please retry in 27s.")
    assert extract_retry_delay(exc1) == 27.0

    exc2 = RuntimeError("Rate limit hit. retry after 14.5s")
    assert extract_retry_delay(exc2) == 14.5

    exc3 = RuntimeError("Generic 429 without explicit delay hint.")
    assert extract_retry_delay(exc3, default_delay=30.0) == 30.0


def test_retry_on_429_with_suggested_delay() -> None:
    """Verifies that 429 errors sleep for suggested delay plus one second and retry successfully."""
    slept_durations: list[float] = []

    def mock_sleep(seconds: float) -> None:
        slept_durations.append(seconds)

    limiter = SlidingWindowRateLimiter(
        max_items_per_minute=80, sleep_fn=mock_sleep
    )
    provider = GeminiProvider(
        api_key="test_api_key",
        embedding_model="gemini-embedding-2",
        generation_model="gemini-3.5-flash-lite",
        max_rate_limit_retries=3,
        rate_limiter=limiter,
        sleep_fn=mock_sleep,
    )

    attempt = 0

    def mock_operation() -> str:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError(
                "RESOURCE_EXHAUSTED: Rate limit exceeded. Please retry in 20s"
            )
        return "operation_success"

    result = provider._execute_with_retry(mock_operation)

    assert result == "operation_success"
    assert attempt == 2
    # Expect 20s + 1s = 21s wait
    assert slept_durations == [21.0]


def test_retry_exceeds_max_attempts_raises() -> None:
    """Verifies that repeated 429 errors eventually raise after exhausting max retries."""
    slept_durations: list[float] = []

    provider = GeminiProvider(
        api_key="test_api_key",
        embedding_model="gemini-embedding-2",
        generation_model="gemini-3.5-flash-lite",
        max_rate_limit_retries=2,
        sleep_fn=slept_durations.append,
    )

    def always_fail() -> None:
        raise RuntimeError("RESOURCE_EXHAUSTED: 429 Please retry in 5s")

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        provider._execute_with_retry(always_fail)

    assert len(slept_durations) == 2
