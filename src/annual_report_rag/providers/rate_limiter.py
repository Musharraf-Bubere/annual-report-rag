"""Sliding-window rate limiter to enforce API request and item quotas."""

import time
from collections import deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    """Regulates throughput over a sliding time window to stay within API rate limits."""

    def __init__(
        self,
        max_items_per_minute: int = 80,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initializes rate limiter with item quotas, sliding window, and injectable clocks."""
        self.max_items_per_minute = max_items_per_minute
        self.window_seconds = window_seconds
        self.clock = clock
        self.sleep_fn = sleep_fn
        self.history: deque[tuple[float, int]] = deque()

    def _clean_expired(self, current_time: float) -> None:
        """Removes records that have fallen outside the active sliding time window."""
        while self.history and (current_time - self.history[0][0]) >= self.window_seconds:
            self.history.popleft()

    def acquire(self, num_items: int) -> float:
        """Blocks until capacity is available for the given item count, returning wait seconds."""
        now = self.clock()
        self._clean_expired(now)
        total_wait = 0.0

        current_usage = sum(count for _, count in self.history)
        while current_usage + num_items > self.max_items_per_minute:
            if not self.history:
                break
            oldest_time, _ = self.history[0]
            wait_time = (oldest_time + self.window_seconds) - now
            if wait_time > 0:
                self.sleep_fn(wait_time)
                total_wait += wait_time
            now = self.clock()
            self._clean_expired(now)
            current_usage = sum(count for _, count in self.history)

        self.history.append((now, num_items))
        return total_wait
