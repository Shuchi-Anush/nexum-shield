"""FallbackExecutor — deadline-aware retries across providers.

Walk the router-ordered provider list, calling `complete()` on each. Skip
once a provider raises a non-retryable error and move to the next; raise
when:

  * the absolute deadline elapses (`DeadlineExceededError`),
  * the global per-task attempt cap is consumed (`MaxAttemptsExceededError`),
  * the provider list is empty (`NoProvidersAvailableError`),
  * an `LLMAuthError` is hit — auth bugs don't get masked by fallback.

Each provider call is wrapped in `asyncio.wait_for(remaining_seconds)` so the
deadline is enforced even if a provider misbehaves. The provider's own
internal timeout still applies; whichever fires first wins.

Reports outcomes back via `on_attempt_outcome(provider_name, success, exc)`
so the router can update its health window.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import (
    LLMAuthError,
    LLMProviderError,
)
from app.services.llm.orchestration.deadline import Deadline
from app.services.llm.orchestration.exceptions import (
    DeadlineExceededError,
    MaxAttemptsExceededError,
    NoProvidersAvailableError,
)
from app.services.llm.orchestration.metrics import MetricsCollector, NullMetrics
from app.services.llm.orchestration.time_source import (
    DEFAULT_TIME_SOURCE,
    TimeSource,
)
from app.services.llm.schemas import LLMRequest, LLMResponse

log = logging.getLogger(__name__)


AttemptCallback = Callable[[str, bool, Optional[BaseException]], Awaitable[None] | None]


class FallbackExecutor:
    """Executes one logical LLM call with cross-provider fallback."""

    def __init__(
        self,
        *,
        time_source: TimeSource = DEFAULT_TIME_SOURCE,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self._time = time_source
        self._metrics: MetricsCollector = metrics or NullMetrics()

    async def execute(
        self,
        request: LLMRequest,
        providers: list[BaseLLMProvider],
        *,
        deadline: Deadline,
        attempts_consumed: int,
        max_attempts: int,
        on_attempt_outcome: Optional[AttemptCallback] = None,
    ) -> tuple[LLMResponse, str, int]:
        """Run the request, returning (response, provider_name, attempts_made).

        `attempts_consumed` is the number of attempts already spent for this
        task (e.g. by a prior schema-correction round). Adds to the count
        used to detect global-cap exhaustion.
        """
        if not providers:
            raise NoProvidersAvailableError("no providers supplied to fallback executor")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        attempts_made = 0
        last_error: Optional[BaseException] = None

        for provider in providers:
            if attempts_consumed + attempts_made >= max_attempts:
                raise MaxAttemptsExceededError(
                    attempts_consumed + attempts_made, max_attempts
                )

            remaining = deadline.remaining_seconds(self._time)
            if remaining <= 0:
                raise DeadlineExceededError(
                    "deadline elapsed before next provider attempt"
                )

            attempts_made += 1
            name = provider.provider_name
            t_start = self._time.monotonic()
            error_kind: Optional[str] = None
            success = False

            try:
                response = await asyncio.wait_for(
                    provider.complete(request), timeout=remaining,
                )
                success = True
                latency_ms = (self._time.monotonic() - t_start) * 1000.0
                self._metrics.record_attempt(name, True, latency_ms, None)
                await _maybe_await(on_attempt_outcome, name, True, None)
                log.info(
                    "fallback_attempt_ok",
                    extra={
                        "provider": name,
                        "attempt": attempts_consumed + attempts_made,
                        "latency_ms": round(latency_ms, 2),
                    },
                )
                return response, name, attempts_made

            except LLMAuthError as exc:
                # Configuration bug — a provider with a bad key won't fix
                # itself. Surface immediately rather than burn the attempt
                # budget across every provider in the list.
                latency_ms = (self._time.monotonic() - t_start) * 1000.0
                error_kind = exc.kind.value
                self._metrics.record_attempt(name, False, latency_ms, error_kind)
                await _maybe_await(on_attempt_outcome, name, False, exc)
                log.error(
                    "fallback_auth_error_terminal",
                    extra={"provider": name, "error": str(exc)},
                )
                raise

            except asyncio.TimeoutError as exc:
                latency_ms = (self._time.monotonic() - t_start) * 1000.0
                error_kind = "deadline_timeout"
                self._metrics.record_attempt(name, False, latency_ms, error_kind)
                await _maybe_await(on_attempt_outcome, name, False, exc)
                log.warning(
                    "fallback_attempt_timed_out",
                    extra={
                        "provider": name,
                        "attempt": attempts_consumed + attempts_made,
                        "remaining_at_start": round(remaining, 3),
                    },
                )
                last_error = exc
                # Deadline may already be expired — check before continuing.
                if deadline.is_expired(self._time):
                    raise DeadlineExceededError(
                        "deadline elapsed during provider call"
                    ) from exc
                continue

            except LLMProviderError as exc:
                latency_ms = (self._time.monotonic() - t_start) * 1000.0
                error_kind = exc.kind.value
                self._metrics.record_attempt(name, False, latency_ms, error_kind)
                await _maybe_await(on_attempt_outcome, name, False, exc)
                log.warning(
                    "fallback_attempt_failed",
                    extra={
                        "provider": name,
                        "attempt": attempts_consumed + attempts_made,
                        "kind": error_kind,
                        "status_code": exc.status_code,
                    },
                )
                last_error = exc
                continue

            except Exception as exc:  # noqa: BLE001 — unexpected errors should not crash the orchestrator
                latency_ms = (self._time.monotonic() - t_start) * 1000.0
                error_kind = "unexpected"
                self._metrics.record_attempt(name, False, latency_ms, error_kind)
                await _maybe_await(on_attempt_outcome, name, False, exc)
                log.exception(
                    "fallback_attempt_unexpected_error",
                    extra={"provider": name},
                )
                last_error = exc
                continue
            finally:
                # Defensive: ensure success metric is consistent with branch taken.
                if not success and error_kind is None:
                    self._metrics.record_attempt(
                        name, False, (self._time.monotonic() - t_start) * 1000.0, "unknown"
                    )

        # Exhausted the provider list. Bubble the most informative failure.
        if last_error is not None:
            raise last_error
        raise NoProvidersAvailableError("provider list exhausted with no errors recorded")


async def _maybe_await(
    cb: Optional[AttemptCallback],
    name: str,
    success: bool,
    exc: Optional[BaseException],
) -> None:
    if cb is None:
        return
    result = cb(name, success, exc)
    if asyncio.iscoroutine(result):
        await result
