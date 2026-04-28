"""Priority-aware token-bucket rate limiter.

Three pools — HIGH, NORMAL, LOW — each with its own capacity and refill rate.
Acquisition rules:

    HIGH    : own pool only.
    NORMAL  : own pool, then borrows from LOW if NORMAL is empty.
    LOW     : own pool only.

Higher-priority traffic NEVER consumes lower-priority overflow that the lower
tier still needs — borrowing is one-way and only NORMAL can borrow (the
canonical "elastic middle" pattern). HIGH is intentionally isolated so a flood
of LOW traffic cannot starve it.

Memory bound
------------
State is exactly three `_Bucket` records. No per-request history, no
unbounded queue, no per-key dictionary that could grow with traffic. The
mutex is a single `asyncio.Lock` — contention is bounded by the number of
in-flight orchestrator calls.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from app.services.llm.orchestration.metrics import MetricsCollector, NullMetrics
from app.services.llm.orchestration.schemas import Priority
from app.services.llm.orchestration.time_source import (
    DEFAULT_TIME_SOURCE,
    TimeSource,
)

log = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    capacity: int
    refill_per_sec: float

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.refill_per_sec <= 0:
            raise ValueError("refill_per_sec must be > 0")


@dataclass
class _Bucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_refill: float


# Borrow chains. HIGH is isolated; NORMAL falls through to LOW only.
_BORROW_ORDER: dict[Priority, tuple[Priority, ...]] = {
    Priority.HIGH: (),
    Priority.NORMAL: (Priority.LOW,),
    Priority.LOW: (),
}


class RateLimiter:
    """Single-process priority token-bucket limiter."""

    def __init__(
        self,
        *,
        pools: dict[Priority, PoolConfig],
        time_source: TimeSource = DEFAULT_TIME_SOURCE,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        for required in (Priority.HIGH, Priority.NORMAL, Priority.LOW):
            if required not in pools:
                raise ValueError(f"missing pool config for priority {required}")

        now = time_source.monotonic()
        self._buckets: dict[Priority, _Bucket] = {
            p: _Bucket(
                capacity=float(cfg.capacity),
                refill_per_sec=cfg.refill_per_sec,
                tokens=float(cfg.capacity),
                last_refill=now,
            )
            for p, cfg in pools.items()
        }
        self._time = time_source
        self._metrics: MetricsCollector = metrics or NullMetrics()
        self._lock = asyncio.Lock()

    async def try_acquire(self, priority: Priority) -> bool:
        """Acquire one token. Returns True on success, False if all eligible
        pools are empty. Never blocks waiting for refill — caller decides
        whether to retry, queue, or fail."""
        async with self._lock:
            self._refill_all()
            if self._consume(priority):
                self._metrics.record_rate_limit(
                    priority.value, accepted=True, borrowed_from=None
                )
                return True

            for fallback in _BORROW_ORDER[priority]:
                if self._consume(fallback):
                    log.debug(
                        "rate_limiter_borrowed",
                        extra={"priority": priority.value, "borrowed_from": fallback.value},
                    )
                    self._metrics.record_rate_limit(
                        priority.value, accepted=True, borrowed_from=fallback.value
                    )
                    return True

            self._metrics.record_rate_limit(
                priority.value, accepted=False, borrowed_from=None
            )
            return False

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Read-only view of bucket levels (for /health, debugging, tests)."""
        self._refill_all()
        return {
            p.value: {
                "tokens": round(b.tokens, 4),
                "capacity": b.capacity,
                "refill_per_sec": b.refill_per_sec,
            }
            for p, b in self._buckets.items()
        }

    # ── private ──────────────────────────────────────────────────────────

    def _refill_all(self) -> None:
        now = self._time.monotonic()
        for bucket in self._buckets.values():
            elapsed = max(0.0, now - bucket.last_refill)
            if elapsed > 0:
                bucket.tokens = min(
                    bucket.capacity,
                    bucket.tokens + elapsed * bucket.refill_per_sec,
                )
                bucket.last_refill = now

    def _consume(self, priority: Priority) -> bool:
        bucket = self._buckets[priority]
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False
