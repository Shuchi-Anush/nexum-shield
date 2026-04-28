"""ProviderRouter — health-aware + cost-aware ordering.

Each provider has a sliding window of recent attempts. The window is bounded
by both `max_size` (deque cap) and `max_age_seconds` (entries older than this
are dropped on read). Fewer than `min_observations` recent attempts → treat
as healthy (no penalty for quiet providers).

Sort key per call:

    (-health_score, cost_estimate, provider_name)

— so highest health wins, ties broken by lower cost, then by name for stable
ordering. The router never *removes* providers; downstream `FallbackExecutor`
will skip ones that fail at call time. This separation keeps routing pure
and observable.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.services.llm.base import BaseLLMProvider
from app.services.llm.orchestration.metrics import MetricsCollector, NullMetrics
from app.services.llm.orchestration.time_source import (
    DEFAULT_TIME_SOURCE,
    TimeSource,
)
from app.services.llm.schemas import LLMRequest

log = logging.getLogger(__name__)


CostEstimator = Callable[[str, LLMRequest], float]
"""Pure function: (provider_name, request) → estimated USD cost."""


@dataclass
class _HealthWindow:
    max_size: int
    max_age_seconds: float
    entries: deque[tuple[float, bool]] = field(default_factory=deque)

    def record(self, ts: float, success: bool) -> None:
        self.entries.append((ts, success))
        while len(self.entries) > self.max_size:
            self.entries.popleft()

    def score(self, now: float, min_observations: int) -> float:
        """Success rate over the (bounded, fresh) window. 1.0 when too few
        observations to judge — keeps untested providers from being penalised."""
        cutoff = now - self.max_age_seconds
        while self.entries and self.entries[0][0] < cutoff:
            self.entries.popleft()
        n = len(self.entries)
        if n < min_observations:
            return 1.0
        successes = sum(1 for _, ok in self.entries if ok)
        return successes / n


class ProviderRouter:
    """Orders an immutable provider set by health and cost."""

    def __init__(
        self,
        *,
        providers: list[BaseLLMProvider],
        cost_estimator: CostEstimator,
        window_size: int = 100,
        window_seconds: float = 300.0,
        min_observations: int = 5,
        time_source: TimeSource = DEFAULT_TIME_SOURCE,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider must be registered")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if min_observations < 0:
            raise ValueError("min_observations must be >= 0")

        self._providers: dict[str, BaseLLMProvider] = {
            p.provider_name: p for p in providers
        }
        if len(self._providers) != len(providers):
            raise ValueError("provider_name collisions in providers list")

        self._cost = cost_estimator
        self._min_obs = min_observations
        self._time = time_source
        self._metrics: MetricsCollector = metrics or NullMetrics()
        self._windows: dict[str, _HealthWindow] = {
            name: _HealthWindow(max_size=window_size, max_age_seconds=window_seconds)
            for name in self._providers
        }

    @property
    def providers(self) -> list[BaseLLMProvider]:
        return list(self._providers.values())

    def order(self, request: LLMRequest) -> list[BaseLLMProvider]:
        """Return providers ordered by (-health, cost, name)."""
        now = self._time.monotonic()
        ranked = []
        for name, provider in self._providers.items():
            health = self._windows[name].score(now, self._min_obs)
            cost = self._cost(name, request)
            ranked.append((-health, cost, name, provider))
        ranked.sort(key=lambda r: (r[0], r[1], r[2]))
        return [r[3] for r in ranked]

    def record(self, provider_name: str, success: bool) -> None:
        window = self._windows.get(provider_name)
        if window is None:
            return
        window.record(self._time.monotonic(), success)

    def health_snapshot(self) -> dict[str, float]:
        """Current per-provider health score. For /health, debugging, tests."""
        now = self._time.monotonic()
        return {
            name: round(window.score(now, self._min_obs), 4)
            for name, window in self._windows.items()
        }
