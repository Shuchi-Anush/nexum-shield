"""Placeholder metrics hook for the LLM service layer.

Providers call these functions at well-known points (request start, success,
failure). The default implementations are no-ops so the LLM layer runs
without any external metrics backend.

Replace the bodies — or monkeypatch the symbols — when wiring Prometheus,
OpenTelemetry, StatsD, or any other exporter. No external dependencies are
imported here so this module is cheap to import and safe in any environment.

Example future wiring (pseudo, not enabled):

    from prometheus_client import Counter, Histogram
    _REQ = Counter("llm_requests_total", "...", ["provider", "model"])
    def record_request(provider, model, request_id):
        _REQ.labels(provider=provider, model=model).inc()
"""

from __future__ import annotations

from typing import Optional


def record_request(
    provider: str,
    model: str,
    request_id: Optional[str],
) -> None:
    """Fired immediately before an LLM HTTP call."""
    return None


def record_success(
    provider: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    request_id: Optional[str],
) -> None:
    """Fired after a successful, parsed LLM response."""
    return None


def record_failure(
    provider: str,
    model: str,
    kind: str,
    status_code: Optional[int],
    request_id: Optional[str],
) -> None:
    """Fired when a typed `LLMProviderError` (or subclass) propagates out."""
    return None


def record_retry(
    provider: str,
    model: str,
    attempt: int,
    kind: str,
    request_id: Optional[str],
) -> None:
    """Fired between retry attempts inside a single `complete()` call."""
    return None
