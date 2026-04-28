"""Gemini provider via raw httpx (no google-genai SDK).

Wire format reference:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  Header: x-goog-api-key: <key>          (preferred over ?key= to keep it out of URL logs)
  Body:   {contents, systemInstruction, generationConfig, safetySettings}

The class:

  * Owns one shared `httpx.AsyncClient` for connection pooling. Caller MUST
    invoke `await provider.close()` on shutdown to release sockets.
  * Caps in-flight requests via an `asyncio.Semaphore` (concurrency control).
  * Hard-times-out each HTTP call at `timeout_seconds`.
  * Retries with tenacity (exponential 1s..10s) only on retryable errors —
    rate limits and transient 5xx. Auth, timeouts, and 4xx fail fast.
  * Maps every failure mode to a typed exception from `app.services.llm.exceptions`.
  * Enforces a basic serialized-payload size guard before hitting the wire.
  * Emits structured logs (request_id correlation) and fires metric hooks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.services.llm import metrics
from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import (
    LLMAuthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.services.llm.schemas import (
    LLMErrorKind,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMTokenUsage,
)

log = logging.getLogger(__name__)


_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_PROVIDER_NAME = "gemini"


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate — only retry exceptions that opted in."""
    return isinstance(exc, LLMProviderError) and getattr(exc, "retryable", False)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini implementation over the v1beta REST API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        max_concurrency: int = 8,
        max_request_bytes: int = 1_048_576,
    ) -> None:
        if not api_key:
            raise LLMAuthError(provider=_PROVIDER_NAME)
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be >= 1")

        self._api_key = api_key
        self._model = model
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._max_request_bytes = int(max_request_bytes)
        self._semaphore = asyncio.Semaphore(int(max_concurrency))
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=httpx.Timeout(self._timeout_seconds),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
        )

    @classmethod
    def from_settings(cls, settings: Any) -> "GeminiProvider":
        return cls(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            timeout_seconds=settings.GEMINI_TIMEOUT_SECONDS,
            max_retries=settings.GEMINI_MAX_RETRIES,
            max_concurrency=settings.GEMINI_MAX_CONCURRENCY,
            max_request_bytes=settings.GEMINI_MAX_REQUEST_BYTES,
        )

    @property
    def provider_name(self) -> str:
        return _PROVIDER_NAME

    # ── Public API ───────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not request.messages:
            raise LLMProviderError(
                "request.messages must contain at least one message",
                kind=LLMErrorKind.INVALID_REQUEST,
                provider=_PROVIDER_NAME,
                retryable=False,
            )

        model = request.model or self._model
        payload = self._build_payload(request)
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._enforce_size_guard(payload_bytes, request.request_id)

        metrics.record_request(_PROVIDER_NAME, model, request.request_id)
        log.info(
            "llm_request_start",
            extra={
                "provider": _PROVIDER_NAME,
                "model": model,
                "request_id": request.request_id,
                "message_count": len(request.messages),
                "payload_bytes": len(payload_bytes),
            },
        )

        start = time.perf_counter()
        try:
            async with self._semaphore:
                raw = await self._call_api(model, payload_bytes, request.request_id)
        except LLMProviderError as exc:
            metrics.record_failure(
                _PROVIDER_NAME,
                model,
                exc.kind.value,
                exc.status_code,
                request.request_id,
            )
            log.warning(
                "llm_request_failed",
                extra={
                    "provider": _PROVIDER_NAME,
                    "model": model,
                    "request_id": request.request_id,
                    "kind": exc.kind.value,
                    "status_code": exc.status_code,
                },
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        response = self._parse_response(raw, model, latency_ms, request.request_id)

        metrics.record_success(
            _PROVIDER_NAME,
            response.model,
            latency_ms,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            request.request_id,
        )
        log.info(
            "llm_request_ok",
            extra={
                "provider": _PROVIDER_NAME,
                "model": response.model,
                "request_id": request.request_id,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )
        return response

    async def health_check(self) -> bool:
        """GET /models/{model} — 200 means we can reach the provider with valid creds."""
        try:
            response = await self._client.get(
                f"/models/{self._model}",
                timeout=min(5.0, self._timeout_seconds),
            )
            return response.status_code == 200
        except Exception as exc:  # noqa: BLE001 — health probe must never raise
            log.debug(
                "llm_health_check_failed",
                extra={"provider": _PROVIDER_NAME, "error": repr(exc)},
            )
            return False

    async def close(self) -> None:
        await self._client.aclose()

    # ── Private helpers ──────────────────────────────────────────────────

    def _enforce_size_guard(
        self, payload_bytes: bytes, request_id: Optional[str]
    ) -> None:
        if len(payload_bytes) > self._max_request_bytes:
            log.warning(
                "llm_request_too_large",
                extra={
                    "provider": _PROVIDER_NAME,
                    "request_id": request_id,
                    "size_bytes": len(payload_bytes),
                    "limit_bytes": self._max_request_bytes,
                },
            )
            raise LLMProviderError(
                f"serialized request is {len(payload_bytes)} bytes; "
                f"limit is {self._max_request_bytes}",
                kind=LLMErrorKind.INVALID_REQUEST,
                provider=_PROVIDER_NAME,
                retryable=False,
            )

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """Translate `LLMRequest` → Gemini `generateContent` body.

        SYSTEM messages collapse into the dedicated `systemInstruction` field;
        USER/ASSISTANT messages turn into `contents[]` entries with role
        mapping `assistant → model`.
        """
        contents: list[dict[str, Any]] = []
        system_chunks: list[str] = []

        for msg in request.messages:
            if msg.role == LLMRole.SYSTEM:
                system_chunks.append(msg.content)
                continue
            role = "user" if msg.role == LLMRole.USER else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload: dict[str, Any] = {"contents": contents}

        if system_chunks:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_chunks)}]
            }

        generation_config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        payload["generationConfig"] = generation_config

        if request.extra:
            safety = request.extra.get("safety_settings") or request.extra.get(
                "safetySettings"
            )
            if safety is not None:
                payload["safetySettings"] = safety

        return payload

    async def _call_api(
        self,
        model: str,
        payload_bytes: bytes,
        request_id: Optional[str],
    ) -> dict[str, Any]:
        """Single-shot send wrapped in tenacity exponential-backoff retry.

        Retry budget = `max_retries` total attempts. Only exceptions whose
        `retryable` attribute is truthy participate (rate limits + 5xx).
        """
        attempt_box: dict[str, int] = {"n": 0}

        async def _one_attempt() -> dict[str, Any]:
            attempt_box["n"] += 1
            n = attempt_box["n"]
            if n > 1:
                metrics.record_retry(
                    _PROVIDER_NAME, model, n, "retry", request_id
                )
                log.info(
                    "llm_retry_attempt",
                    extra={
                        "provider": _PROVIDER_NAME,
                        "model": model,
                        "request_id": request_id,
                        "attempt": n,
                    },
                )
            return await self._send_once(model, payload_bytes, request_id)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    return await _one_attempt()
        except RetryError as exc:  # safety net — `reraise=True` should prevent this
            inner = exc.last_attempt.exception() if exc.last_attempt else None
            if isinstance(inner, LLMProviderError):
                raise inner from exc
            raise LLMProviderError(
                f"retry budget exhausted: {exc!r}",
                kind=LLMErrorKind.PROVIDER_ERROR,
                provider=_PROVIDER_NAME,
                retryable=False,
            ) from exc

        # Unreachable: AsyncRetrying with reraise=True either returns inside
        # the `with attempt` block or raises.
        raise LLMProviderError(
            "retry loop exited without producing a response",
            kind=LLMErrorKind.UNKNOWN,
            provider=_PROVIDER_NAME,
            retryable=False,
        )

    async def _send_once(
        self,
        model: str,
        payload_bytes: bytes,
        request_id: Optional[str],
    ) -> dict[str, Any]:
        url = f"/models/{model}:generateContent"
        headers: dict[str, str] = {}
        if request_id:
            headers["x-request-id"] = request_id

        try:
            response = await self._client.post(
                url,
                content=payload_bytes,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                provider=_PROVIDER_NAME,
                timeout_seconds=self._timeout_seconds,
            ) from exc
        except httpx.HTTPError as exc:
            # Connection refused, DNS, read errors, etc. — treat as transient.
            raise LLMProviderError(
                f"transport error: {exc!r}",
                kind=LLMErrorKind.PROVIDER_ERROR,
                provider=_PROVIDER_NAME,
                retryable=True,
            ) from exc

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise LLMProviderError(
                    f"invalid JSON in 200 response: {exc!r}",
                    kind=LLMErrorKind.PROVIDER_ERROR,
                    provider=_PROVIDER_NAME,
                    retryable=False,
                    status_code=response.status_code,
                ) from exc

        self._raise_for_status(response)
        # _raise_for_status always raises; this satisfies the type checker.
        raise LLMProviderError(
            "unreachable",
            kind=LLMErrorKind.UNKNOWN,
            provider=_PROVIDER_NAME,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        code = response.status_code
        body_preview = response.text[:500] if response.text else ""

        if code in (401, 403):
            raise LLMAuthError(provider=_PROVIDER_NAME)

        if code == 429:
            raise LLMRateLimitError(
                provider=_PROVIDER_NAME,
                retry_after=_parse_retry_after(response.headers.get("retry-after")),
            )

        if 400 <= code < 500:
            raise LLMProviderError(
                f"client error {code}: {body_preview}",
                kind=LLMErrorKind.INVALID_REQUEST,
                provider=_PROVIDER_NAME,
                retryable=False,
                status_code=code,
            )

        # 5xx and anything else non-2xx — assume transient.
        raise LLMProviderError(
            f"server error {code}: {body_preview}",
            kind=LLMErrorKind.PROVIDER_ERROR,
            provider=_PROVIDER_NAME,
            retryable=True,
            status_code=code,
        )

    def _parse_response(
        self,
        raw: dict[str, Any],
        requested_model: str,
        latency_ms: float,
        request_id: Optional[str],
    ) -> LLMResponse:
        candidates = raw.get("candidates") or []
        if not candidates:
            prompt_feedback = raw.get("promptFeedback") or {}
            block_reason = prompt_feedback.get("blockReason")
            detail = (
                f"blocked: {block_reason}"
                if block_reason is not None
                else "no candidates returned"
            )
            raise LLMProviderError(
                detail,
                kind=LLMErrorKind.PROVIDER_ERROR,
                provider=_PROVIDER_NAME,
                retryable=False,
            )

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))

        usage_raw = raw.get("usageMetadata") or {}
        usage = LLMTokenUsage(
            prompt_tokens=int(usage_raw.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage_raw.get("candidatesTokenCount", 0) or 0),
            total_tokens=int(usage_raw.get("totalTokenCount", 0) or 0),
        )

        # Gemini echoes the resolved model version (e.g. "gemini-2.5-flash-001").
        # Fall back to the requested model so the field is always populated.
        model_version = raw.get("modelVersion") or requested_model

        return LLMResponse(
            content=text,
            model=model_version,
            usage=usage,
            provider=_PROVIDER_NAME,
            latency_ms=latency_ms,
            request_id=request_id,
        )


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date form is allowed by spec but Gemini returns seconds; we
        # silently drop unparseable values rather than fail the whole call.
        return None
