"""Orchestration metrics protocol + null implementation.

Components depend on the `MetricsCollector` Protocol — they never import a
concrete backend. Production wires a Prometheus / OpenTelemetry adapter that
satisfies the same shape; tests pass a recording fake; default is `NullMetrics`.
"""

from __future__ import annotations

from typing import Optional, Protocol


class MetricsCollector(Protocol):
    """Hook surface fired by orchestrator components."""

    def record_attempt(
        self,
        provider: str,
        success: bool,
        latency_ms: float,
        error_kind: Optional[str],
    ) -> None: ...

    def record_cache(self, hit: bool) -> None: ...

    def record_cost(
        self,
        operation: str,        # 'reserve' | 'confirm' | 'release' | 'reaped'
        amount: float,
        accepted: bool,
    ) -> None: ...

    def record_rate_limit(
        self,
        priority: str,
        accepted: bool,
        borrowed_from: Optional[str],
    ) -> None: ...

    def record_guardrail(
        self,
        kind: str,             # 'json_parse' | 'schema_validation' | 'correction'
        violation: bool,
    ) -> None: ...

    def record_orchestration(
        self,
        outcome: str,          # 'ok' | 'deadline' | 'budget' | 'rate_limit' | 'max_attempts' | 'guardrail' | 'error'
        attempts: int,
        latency_ms: float,
    ) -> None: ...


class NullMetrics:
    """No-op collector — the default. Every method swallows its inputs."""

    def record_attempt(
        self,
        provider: str,
        success: bool,
        latency_ms: float,
        error_kind: Optional[str],
    ) -> None:
        return None

    def record_cache(self, hit: bool) -> None:
        return None

    def record_cost(
        self,
        operation: str,
        amount: float,
        accepted: bool,
    ) -> None:
        return None

    def record_rate_limit(
        self,
        priority: str,
        accepted: bool,
        borrowed_from: Optional[str],
    ) -> None:
        return None

    def record_guardrail(self, kind: str, violation: bool) -> None:
        return None

    def record_orchestration(
        self,
        outcome: str,
        attempts: int,
        latency_ms: float,
    ) -> None:
        return None
