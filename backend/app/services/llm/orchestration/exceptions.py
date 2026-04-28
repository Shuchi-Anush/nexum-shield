"""Typed exceptions raised by the orchestration pipeline.

Callers branch on the concrete subclass to choose recovery behaviour
(retry the whole task, fall back to non-LLM heuristic, mark job FAILED, …).
"""

from __future__ import annotations

from typing import Optional


class OrchestrationError(Exception):
    """Base for every orchestrator-raised error."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class DeadlineExceededError(OrchestrationError):
    """Absolute deadline elapsed before a usable response was produced."""


class CostBudgetExceededError(OrchestrationError):
    """Reservation would push committed+reserved spend over the budget."""

    def __init__(self, *, requested: float, spent: float, reserved: float, limit: float) -> None:
        super().__init__(
            f"cost budget exceeded: requested={requested:.4f}, "
            f"spent={spent:.4f}, reserved={reserved:.4f}, limit={limit:.4f}"
        )
        self.requested = requested
        self.spent = spent
        self.reserved = reserved
        self.limit = limit


class RateLimitedError(OrchestrationError):
    """No tokens available in the request priority's pool (or any borrowable pool)."""

    def __init__(self, priority: str) -> None:
        super().__init__(f"rate limit exhausted for priority {priority!r}")
        self.priority = priority


class NoProvidersAvailableError(OrchestrationError):
    """Router returned an empty ordered list — nothing to call."""


class MaxAttemptsExceededError(OrchestrationError):
    """Global per-task attempt cap consumed without a valid response."""

    def __init__(self, attempts: int, max_attempts: int) -> None:
        super().__init__(
            f"exhausted {attempts}/{max_attempts} attempts without a valid response"
        )
        self.attempts = attempts
        self.max_attempts = max_attempts


class GuardrailViolationError(OrchestrationError):
    """Base for guardrail-detected problems (schema, JSON parse, …)."""


class SchemaValidationError(GuardrailViolationError):
    """Response was JSON but did not match the declared schema."""

    def __init__(self, message: str, *, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class JSONParseError(GuardrailViolationError):
    """Response was not parseable as JSON when JSON grounding was requested."""

    def __init__(self, message: str, *, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content
