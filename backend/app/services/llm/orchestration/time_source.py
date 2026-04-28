"""Single consistent time source for the orchestration layer.

Every component reads the wall clock or monotonic clock through one of these
two methods. Tests inject a `FakeTimeSource` to make deadline/cost/rate-limit
behaviour deterministic.

Why a class — the obvious alternative (`datetime.utcnow()` + `time.monotonic()`
sprinkled across modules) makes deadline arithmetic untestable and lets bugs
hide behind real time advancing.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


class TimeSource:
    """Default time source. Returns timezone-aware UTC datetimes."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        """Monotonic seconds — for token-bucket refills, latency timing."""
        return time.monotonic()

    def unix(self) -> float:
        """Wall-clock seconds since epoch — for Redis TTLs and ZSET scores."""
        return self.now().timestamp()


DEFAULT_TIME_SOURCE = TimeSource()
