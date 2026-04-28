"""Abstract LLM provider contract.

Every provider implementation translates `LLMRequest` to its native wire
format, calls the external API with a hard timeout, and returns a normalised
`LLMResponse`. Providers raise typed exceptions on failure so callers can
branch on error kind without parsing strings.

Providers are *transport*. They do not own retry orchestration semantics
visible to callers (retries internal to a single `complete()` call are fine),
caching, or business logic — those belong in the caller layer (workers,
engines).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.services.llm.schemas import LLMRequest, LLMResponse

if TYPE_CHECKING:
    from app.core.config import Settings


class BaseLLMProvider(ABC):
    """Contract that every LLM provider must fulfil."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical lowercase name: 'gemini', 'claude', 'openai'."""

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: "Settings") -> "BaseLLMProvider":
        """Construct a configured instance from application Settings.

        Each provider knows which Settings fields it needs. The factory
        delegates here so registering a new provider does not require
        editing factory code.
        """

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return a normalised response.

        Raises:
            LLMTimeoutError    — hard timeout exceeded
            LLMRateLimitError  — 429 / quota
            LLMAuthError       — invalid API key
            LLMProviderError   — everything else
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Lightweight reachability probe. Must NOT raise."""

    async def close(self) -> None:
        """Tear down HTTP clients and release resources. Optional override."""
        return None
