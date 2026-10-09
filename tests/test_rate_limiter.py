"""Unit tests for SlidingWindowRateLimiter using simulated clocks."""

from annual_report_rag.providers.rate_limiter import SlidingWindowRateLimiter


class FakeClock:
    """Simulates elapsed time monotonically for deterministic rate limiter testing."""

    def __init__(self, start_time: float = 0.0) -> None:
        self.current_time = start_time
        self.slept_durations: list[float] = []

    def time(self) -> float:
        return self.current_time

    def sleep(self, duration: float) -> None:
        self.slept_durations.append(duration)
        self.current_time += duration


def test_rate_limiter_allows_requests_under_limit() -> None:
    """Verifies that requests within the quota proceed without any sleeping."""
    fake_clock = FakeClock()
    limiter = SlidingWindowRateLimiter(
        max_items_per_minute=80,
        window_seconds=60.0,
        clock=fake_clock.time,
        sleep_fn=fake_clock.sleep,
    )

    wait1 = limiter.acquire(20)
    wait2 = limiter.acquire(20)
    wait3 = limiter.acquire(40)

    assert wait1 == 0.0
    assert wait2 == 0.0
    assert wait3 == 0.0
    assert len(fake_clock.slept_durations) == 0


def test_rate_limiter_throttles_when_quota_exceeded() -> None:
    """Verifies that exceeding quota triggers sleep until earlier items expire from window."""
    fake_clock = FakeClock()
    limiter = SlidingWindowRateLimiter(
        max_items_per_minute=80,
        window_seconds=60.0,
        clock=fake_clock.time,
        sleep_fn=fake_clock.sleep,
    )

    limiter.acquire(40)  # at t=0
    fake_clock.current_time = 10.0
    limiter.acquire(40)  # at t=10; total 80 used

    fake_clock.current_time = 20.0
    # Requesting 20 more items at t=20: requires the first batch (t=0) to expire at t=60
    wait_time = limiter.acquire(20)

    assert wait_time == 40.0
    assert fake_clock.slept_durations == [40.0]
    assert fake_clock.current_time == 60.0
