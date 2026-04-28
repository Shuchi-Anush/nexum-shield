"""Orchestration-layer transport types.

These are *internal* — they sit between the caller (worker / engine) and the
provider layer. They wrap an `LLMRequest` with the policy knobs the
orchestrator needs: priority, deadline, cost estimate, schema, idempotency
key, cache hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.services.llm.orchestration.deadline import Deadline
from app.services.llm.schemas import LLMRequest, LLMResponse


class Priority(str, Enum):
    """Tiers consumed by `RateLimiter` and surface in metrics."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class OrchestratedRequest:
    """A caller's intent plus all the policy the orchestrator enforces."""

    task_id: str
    """Stable idempotency key. Used as the cost-reservation key, so callers
    must reuse the same `task_id` for retries of the same logical task."""

    request: LLMRequest

    deadline: Deadline
    """Absolute deadline. The orchestrator never blocks past this point."""

    priority: Priority = Priority.NORMAL

    estimated_cost_usd: float = 0.0
    """Pre-call estimate. The CostController applies its safety margin on top."""

    response_schema: Optional[dict[str, Any]] = None
    """If set, the orchestrator enforces JSON grounding + schema validation."""

    max_attempts: int = 5
    """Hard cap on total provider calls per task (across fallback + schema correction)."""

    cache_key: Optional[str] = None
    """If set, response is consulted/written. Subject to the cache safety predicate."""

    cache_ttl_seconds: int = 3600

    contains_pii: bool = False
    """Suppresses caching when True. Caller marks this; orchestrator does not infer."""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratedResponse:
    """What the orchestrator hands back."""

    response: LLMResponse
    provider_name: str
    attempts: int
    cached: bool
    cost_usd: float
    parsed_json: Optional[Any] = None
    """Populated when `response_schema` was set and validation passed."""
