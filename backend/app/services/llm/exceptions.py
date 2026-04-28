"""Typed exception hierarchy for LLM provider errors.

Callers branch on the concrete subclass to decide retry / fail-job / fall-back
behaviour. The `retryable` flag is the canonical signal consulted by the
provider's tenacity retry predicate.
"""

from __future__ import annotations

from typing import Optional

from app.services.llm.schemas import LLMErrorKind


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""

    def __init__(
        self,
        message: str,
        kind: LLMErrorKind = LLMErrorKind.UNKNOWN,
        provider: str = "unknown",
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code

    def __str__(self) -> str:
        suffix = f" (status={self.status_code})" if self.status_code is not None else ""
        return f"[{self.provider}/{self.kind.value}] {self.message}{suffix}"


class LLMTimeoutError(LLMProviderError):
    """Hard timeout exceeded. Not retried inside the provider — surfaced to caller."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            f"request exceeded timeout of {timeout_seconds:.2f}s",
            kind=LLMErrorKind.TIMEOUT,
            provider=provider,
            retryable=False,
        )
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMProviderError):
    """429 / quota exhausted — always retryable."""

    def __init__(self, provider: str, retry_after: Optional[float] = None) -> None:
        suffix = f" (retry_after={retry_after}s)" if retry_after is not None else ""
        super().__init__(
            f"rate limit exceeded{suffix}",
            kind=LLMErrorKind.RATE_LIMIT,
            provider=provider,
            retryable=True,
            status_code=429,
        )
        self.retry_after = retry_after


class LLMAuthError(LLMProviderError):
    """Invalid or missing API key — never retryable."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            "missing or invalid API credentials",
            kind=LLMErrorKind.AUTH,
            provider=provider,
            retryable=False,
        )
