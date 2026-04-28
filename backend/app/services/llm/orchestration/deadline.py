"""Absolute-datetime deadline propagated through the orchestration pipeline.

Each `OrchestratedRequest` carries one `Deadline`. The orchestrator and every
downstream component (cost reserve, rate limit, fallback executor, schema
retry) checks remaining time against the *same* `TimeSource` so wall-clock
drift cannot smuggle a request past its deadline.

The deadline is absolute — not a per-call timeout — so a request that has
spent 20s in queue gets a shorter remaining budget when the provider call
finally happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.llm.orchestration.time_source import TimeSource


@dataclass(frozen=True)
class Deadline:
    """An absolute UTC deadline."""

    deadline_at: datetime

    def __post_init__(self) -> None:
        if self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone-aware")

    @classmethod
    def in_seconds(cls, seconds: float, *, time_source: "TimeSource") -> "Deadline":
        if seconds < 0:
            raise ValueError("deadline seconds must be non-negative")
        return cls(time_source.now() + timedelta(seconds=seconds))

    @classmethod
    def at(cls, when: datetime) -> "Deadline":
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return cls(when)

    def remaining_seconds(self, time_source: "TimeSource") -> float:
        delta = (self.deadline_at - time_source.now()).total_seconds()
        return max(0.0, delta)

    def is_expired(self, time_source: "TimeSource") -> bool:
        return (self.deadline_at - time_source.now()).total_seconds() <= 0.0
