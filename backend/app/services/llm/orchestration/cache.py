"""Redis-backed response cache with a strict safety predicate.

Caching an LLM response is dangerous when the request leaks into the cached
artefact (PII), when the response was non-deterministic (temperature > 0,
random seed), or when the caller has not opted in (no `cache_key`). The
predicate refuses cache writes in all of those cases. There is no
"smart inference" — anything ambiguous defaults to *no cache*.

The cache stores `LLMResponse` JSON; reads return a freshly validated
`LLMResponse` so the consumer never sees raw bytes.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from redis.asyncio import Redis

from app.services.llm.orchestration.metrics import MetricsCollector, NullMetrics
from app.services.llm.orchestration.schemas import OrchestratedRequest
from app.services.llm.schemas import LLMResponse

log = logging.getLogger(__name__)


class ResponseCache:
    """Cache wrapper around an async redis client."""

    def __init__(
        self,
        *,
        redis: Redis,
        namespace: str,
        default_ttl_seconds: int = 3600,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be > 0")
        self._redis = redis
        self._ns = namespace.rstrip(":")
        self._default_ttl = default_ttl_seconds
        self._metrics: MetricsCollector = metrics or NullMetrics()

    def is_safe_to_cache(
        self,
        req: OrchestratedRequest,
        response: LLMResponse,
    ) -> bool:
        """Strict predicate. Anything fuzzy → False.

        Conditions (all must hold):

          * Caller supplied `cache_key`.
          * Request temperature is exactly 0.0 (deterministic decoding).
          * Caller did not flag `contains_pii`.
          * `metadata['cacheable'] is False` is honoured as an explicit opt-out.
          * Response.content is non-empty (we don't cache nothings).
        """
        if not req.cache_key:
            return False
        if req.contains_pii:
            return False
        if req.metadata.get("cacheable") is False:
            return False
        if req.request.temperature != 0.0:
            return False
        if not response.content:
            return False
        return True

    async def get(self, key: str) -> Optional[LLMResponse]:
        raw = await self._redis.get(self._key(key))
        if raw is None:
            self._metrics.record_cache(hit=False)
            return None
        try:
            data = json.loads(raw)
            response = LLMResponse.model_validate(data)
        except (ValueError, TypeError) as exc:
            log.warning(
                "response_cache_decode_failed",
                extra={"key": key, "error": repr(exc)},
            )
            self._metrics.record_cache(hit=False)
            return None
        self._metrics.record_cache(hit=True)
        return response

    async def set(
        self,
        key: str,
        response: LLMResponse,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        ttl = int(ttl_seconds) if ttl_seconds is not None else self._default_ttl
        if ttl <= 0:
            return
        await self._redis.set(
            self._key(key),
            response.model_dump_json(),
            ex=ttl,
        )

    def _key(self, key: str) -> str:
        return f"{self._ns}:{key}"
