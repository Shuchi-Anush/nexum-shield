"""Provider-agnostic transport schemas for the LLM service layer.

These types are *internal transport* — not API contract types — so they live
under `services/llm/` rather than `models/`. If they ever surface in an HTTP
response, wrap them in a thin model under `app/models/`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    role: LLMRole
    content: str


class LLMRequest(BaseModel):
    """Provider-agnostic prompt envelope."""

    messages: list[LLMMessage]
    model: Optional[str] = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    request_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Normalised completion result."""

    content: str
    model: str
    usage: LLMTokenUsage
    provider: str
    latency_ms: float
    request_id: Optional[str] = None


class LLMErrorKind(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"
