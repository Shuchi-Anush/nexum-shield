"""CostController — atomic budget reservations backed by Redis Lua.

Why Lua: each of `reserve` / `confirm` / `release` is a read-modify-write that
must be atomic across many orchestrator processes. A multi-step Python pipeline
would race; Lua runs server-side as a single command.

State (all keys live under one configurable namespace):

    {ns}:budget        HASH    {limit, spent}
    {ns}:reservations  HASH    {task_id: amount}
    {ns}:expiry        ZSET    task_id -> expiry_unix_ts

A reservation's lifetime is bounded by its ZSET score. The reaper background
task — and `reserve` itself opportunistically — drops anything whose score is
in the past, so a crashed worker can never permanently consume budget.

Idempotency
-----------
- `reserve(task_id)` for an already-reserved task_id returns success without
  double-charging. Workers may safely retry.
- `confirm(task_id)` after release/expire/confirm is a no-op. Same for
  `release`. The reaper runs the same "drop expired" code path inside `reserve`
  and in its own loop, and the result is identical no matter how many times
  it executes.

Safety margin
-------------
Caller passes their best-effort `estimated_cost`; the controller multiplies by
`safety_margin` (default 1.10) before checking the budget. This absorbs token
miscounts and stops a borderline estimate from sneaking past the cap.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from redis.asyncio import Redis

from app.services.llm.orchestration.exceptions import CostBudgetExceededError
from app.services.llm.orchestration.metrics import MetricsCollector, NullMetrics
from app.services.llm.orchestration.time_source import (
    DEFAULT_TIME_SOURCE,
    TimeSource,
)

log = logging.getLogger(__name__)


# Lua return convention: { ok, status, ...detail }
# ok       = 1|0 (truthy/falsy)
# status   = string label (for logging / metrics)
# detail   = numeric strings — Lua's number→string conversion is locale-safe
#            with %g, so we format inside Lua before returning.

_RESERVE_LUA = """
local budget_key = KEYS[1]
local res_key    = KEYS[2]
local exp_key    = KEYS[3]

local task_id    = ARGV[1]
local amount     = tonumber(ARGV[2])
local expiry_at  = tonumber(ARGV[3])
local now        = tonumber(ARGV[4])

if amount == nil or amount < 0 then
    return {0, 'invalid_amount'}
end

-- Idempotent: same task_id already holds a reservation. Refresh expiry so
-- a long-running task does not get reaped while it is genuinely in flight.
local existing = redis.call('HGET', res_key, task_id)
if existing then
    redis.call('ZADD', exp_key, expiry_at, task_id)
    return {1, 'already_reserved', existing}
end

-- Reap expired in the same atomic execution so concurrent reservers cannot
-- race the reaper and observe inconsistent reserved totals.
local expired = redis.call('ZRANGEBYSCORE', exp_key, '-inf', '(' .. now)
for i = 1, #expired do
    redis.call('HDEL', res_key, expired[i])
    redis.call('ZREM', exp_key, expired[i])
end

local limit_raw = redis.call('HGET', budget_key, 'limit')
if not limit_raw then
    return {0, 'no_budget'}
end
local limit = tonumber(limit_raw)
local spent = tonumber(redis.call('HGET', budget_key, 'spent') or '0')

local reserved = 0
local vals = redis.call('HVALS', res_key)
for i = 1, #vals do
    reserved = reserved + tonumber(vals[i])
end

if (spent + reserved + amount) > limit then
    return {0, 'budget_exceeded',
            string.format('%.10f', spent),
            string.format('%.10f', reserved),
            string.format('%.10f', limit),
            string.format('%.10f', amount)}
end

redis.call('HSET', res_key, task_id, string.format('%.10f', amount))
redis.call('ZADD', exp_key, expiry_at, task_id)
return {1, 'reserved', string.format('%.10f', amount)}
"""


_CONFIRM_LUA = """
local budget_key = KEYS[1]
local res_key    = KEYS[2]
local exp_key    = KEYS[3]

local task_id = ARGV[1]
local actual  = tonumber(ARGV[2])

if actual == nil or actual < 0 then
    return {0, 'invalid_amount'}
end

local reserved = redis.call('HGET', res_key, task_id)
if not reserved then
    -- Idempotent: already confirmed / released / reaped. Treat as success
    -- without double-charging; caller cannot tell apart their own retry from
    -- a rare expiry-then-confirm race, and double-charging is the worse
    -- outcome.
    return {1, 'already_settled'}
end

redis.call('HDEL', res_key, task_id)
redis.call('ZREM', exp_key, task_id)
redis.call('HINCRBYFLOAT', budget_key, 'spent', string.format('%.10f', actual))
return {1, 'confirmed', string.format('%.10f', actual)}
"""


_RELEASE_LUA = """
local res_key = KEYS[1]
local exp_key = KEYS[2]

local task_id = ARGV[1]

local existed = redis.call('HDEL', res_key, task_id)
redis.call('ZREM', exp_key, task_id)
return {1, 'released', tostring(existed)}
"""


_REAP_LUA = """
local res_key = KEYS[1]
local exp_key = KEYS[2]
local now = tonumber(ARGV[1])

local expired = redis.call('ZRANGEBYSCORE', exp_key, '-inf', '(' .. now)
for i = 1, #expired do
    redis.call('HDEL', res_key, expired[i])
    redis.call('ZREM', exp_key, expired[i])
end
return #expired
"""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class CostController:
    """Atomic budget reservations + idempotent reaper.

    Construct with:
        controller = CostController(
            redis=async_redis_client,
            namespace="llm:cost:default",
            reservation_ttl_seconds=300,
            safety_margin=1.10,
        )
    """

    def __init__(
        self,
        *,
        redis: Redis,
        namespace: str,
        reservation_ttl_seconds: int,
        safety_margin: float = 1.10,
        time_source: TimeSource = DEFAULT_TIME_SOURCE,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        if reservation_ttl_seconds <= 0:
            raise ValueError("reservation_ttl_seconds must be > 0")
        if safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1.0")

        self._redis = redis
        self._ns = namespace.rstrip(":")
        self._budget_key = f"{self._ns}:budget"
        self._res_key = f"{self._ns}:reservations"
        self._exp_key = f"{self._ns}:expiry"
        self._reservation_ttl = reservation_ttl_seconds
        self._safety_margin = float(safety_margin)
        self._time = time_source
        self._metrics: MetricsCollector = metrics or NullMetrics()

        self._reserve_script = redis.register_script(_RESERVE_LUA)
        self._confirm_script = redis.register_script(_CONFIRM_LUA)
        self._release_script = redis.register_script(_RELEASE_LUA)
        self._reap_script = redis.register_script(_REAP_LUA)

        self._reaper_task: Optional[asyncio.Task[None]] = None
        self._reaper_stop: Optional[asyncio.Event] = None

    # ── Budget admin ─────────────────────────────────────────────────────

    async def set_budget(self, limit_usd: float) -> None:
        """Initialise or replace the budget cap. Does not touch `spent`."""
        if limit_usd < 0:
            raise ValueError("limit must be >= 0")
        await self._redis.hset(
            self._budget_key,
            mapping={"limit": f"{limit_usd:.10f}"},
        )
        # Ensure 'spent' exists so downstream HGET parses cleanly.
        await self._redis.hsetnx(self._budget_key, "spent", "0")

    async def get_budget_state(self) -> dict[str, float]:
        """Snapshot of (limit, spent, reserved). Eventually consistent."""
        budget = await self._redis.hgetall(self._budget_key)
        limit = float(_decode(budget.get(b"limit", budget.get("limit", "0"))))
        spent = float(_decode(budget.get(b"spent", budget.get("spent", "0"))))
        reservation_values = await self._redis.hvals(self._res_key)
        reserved = sum(float(_decode(v)) for v in reservation_values)
        return {"limit": limit, "spent": spent, "reserved": reserved}

    # ── Atomic operations ────────────────────────────────────────────────

    async def reserve(self, task_id: str, estimated_cost: float) -> float:
        """Atomically reserve estimated_cost × safety_margin against the budget.

        Returns the actually-reserved amount (post-margin). Raises
        `CostBudgetExceededError` if reservation would breach the cap.
        Idempotent for the same task_id within the reservation TTL.
        """
        if estimated_cost < 0:
            raise ValueError("estimated_cost must be >= 0")
        amount = float(estimated_cost) * self._safety_margin
        now = self._time.unix()
        expiry = now + self._reservation_ttl

        result = await self._reserve_script(
            keys=[self._budget_key, self._res_key, self._exp_key],
            args=[task_id, f"{amount:.10f}", f"{expiry:.10f}", f"{now:.10f}"],
        )
        ok = bool(int(result[0]))
        status = _decode(result[1])

        if not ok:
            if status == "no_budget":
                self._metrics.record_cost("reserve", amount, accepted=False)
                raise CostBudgetExceededError(
                    requested=amount, spent=0.0, reserved=0.0, limit=0.0,
                )
            if status == "budget_exceeded":
                spent = float(_decode(result[2]))
                reserved = float(_decode(result[3]))
                limit = float(_decode(result[4]))
                self._metrics.record_cost("reserve", amount, accepted=False)
                raise CostBudgetExceededError(
                    requested=amount, spent=spent, reserved=reserved, limit=limit,
                )
            self._metrics.record_cost("reserve", amount, accepted=False)
            raise CostBudgetExceededError(
                requested=amount, spent=0.0, reserved=0.0, limit=0.0,
            )

        self._metrics.record_cost("reserve", amount, accepted=True)
        log.debug(
            "cost_reserved",
            extra={
                "task_id": task_id,
                "amount": amount,
                "status": status,
                "namespace": self._ns,
            },
        )
        return amount

    async def confirm(self, task_id: str, actual_cost: float) -> None:
        """Settle the reservation against actual measured cost.

        Idempotent — calling after release/expire/confirm is a no-op. Caller
        must arrange for a `release()` if the call failed without producing
        a billable response.
        """
        if actual_cost < 0:
            raise ValueError("actual_cost must be >= 0")
        result = await self._confirm_script(
            keys=[self._budget_key, self._res_key, self._exp_key],
            args=[task_id, f"{actual_cost:.10f}"],
        )
        ok = bool(int(result[0]))
        status = _decode(result[1])
        self._metrics.record_cost("confirm", actual_cost, accepted=ok)
        log.debug(
            "cost_confirmed",
            extra={
                "task_id": task_id,
                "amount": actual_cost,
                "status": status,
                "namespace": self._ns,
            },
        )

    async def release(self, task_id: str) -> None:
        """Drop the reservation without charging. Idempotent."""
        result = await self._release_script(
            keys=[self._res_key, self._exp_key],
            args=[task_id],
        )
        status = _decode(result[1])
        self._metrics.record_cost("release", 0.0, accepted=True)
        log.debug(
            "cost_released",
            extra={
                "task_id": task_id,
                "status": status,
                "namespace": self._ns,
            },
        )

    async def reap_once(self) -> int:
        """Run a single sweep over expired reservations. Returns count reaped."""
        now = self._time.unix()
        result = await self._reap_script(
            keys=[self._res_key, self._exp_key],
            args=[f"{now:.10f}"],
        )
        count = int(result)
        if count:
            self._metrics.record_cost("reaped", 0.0, accepted=True)
            log.info(
                "cost_reservations_reaped",
                extra={"count": count, "namespace": self._ns},
            )
        return count

    # ── Reaper lifecycle ─────────────────────────────────────────────────

    def start_reaper(self, interval_seconds: float) -> None:
        """Spawn the background reaper. Idempotent — second call is a no-op."""
        if interval_seconds <= 0:
            raise ValueError("interval must be > 0")
        if self._reaper_task is not None and not self._reaper_task.done():
            return
        self._reaper_stop = asyncio.Event()
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(interval_seconds), name="cost-reaper"
        )

    async def stop_reaper(self) -> None:
        """Signal the reaper to exit and wait for it. Safe to call multiple times."""
        if self._reaper_stop is not None:
            self._reaper_stop.set()
        task = self._reaper_task
        if task is not None and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._reaper_task = None
        self._reaper_stop = None

    async def _reaper_loop(self, interval_seconds: float) -> None:
        assert self._reaper_stop is not None
        try:
            while not self._reaper_stop.is_set():
                try:
                    await self.reap_once()
                except Exception as exc:  # noqa: BLE001 — reaper must never crash silently
                    log.warning(
                        "cost_reaper_iteration_failed",
                        extra={"error": repr(exc), "namespace": self._ns},
                    )
                try:
                    await asyncio.wait_for(
                        self._reaper_stop.wait(), timeout=interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            log.info("cost_reaper_cancelled", extra={"namespace": self._ns})
            raise
