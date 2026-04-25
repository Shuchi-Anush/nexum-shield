"""Event store / audit log.

Append-only audit trail of pipeline execution. Each emit writes two Redis
keys:

  * `event:{job_id}:{ts_ns}`         — JSON blob of the full event
  * `events_by_job:{job_id}`         — sorted set of event keys, scored by
                                        nanosecond timestamp for ordered reads

`stage_event(job_id, stage)` is a context manager that emits STARTED on
entry, COMPLETED on clean exit, or FAILED (with exception type + message)
if the wrapped block raises. The original exception is always re-raised so
the existing job-failure flow remains intact.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterator, List, Optional

from app.core.queue import redis_conn


class EventType(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Event:
    event_id: str
    job_id: str
    stage: str
    event_type: EventType
    timestamp: float
    payload: Optional[dict] = None
    latency_ms: Optional[float] = None


def _event_key(job_id: str, ts_ns: int) -> str:
    return f"event:{job_id}:{ts_ns}"


def _index_key(job_id: str) -> str:
    return f"events_by_job:{job_id}"


def emit(
    job_id: str,
    stage: str,
    event_type: EventType,
    payload: Optional[dict] = None,
    latency_ms: Optional[float] = None,
) -> None:
    ts_ns = time.time_ns()
    event = Event(
        event_id=str(uuid.uuid4()),
        job_id=job_id,
        stage=stage,
        event_type=event_type,
        timestamp=ts_ns / 1e9,
        payload=payload,
        latency_ms=latency_ms,
    )
    key = _event_key(job_id, ts_ns)
    redis_conn.set(key, json.dumps(asdict(event)))
    redis_conn.zadd(_index_key(job_id), {key: ts_ns})


def list_events(job_id: str) -> List[dict]:
    keys = redis_conn.zrange(_index_key(job_id), 0, -1)
    if not keys:
        return []
    blobs = redis_conn.mget(keys)
    return [json.loads(blob) for blob in blobs if blob is not None]


@contextmanager
def stage_event(job_id: str, stage: str) -> Iterator[None]:
    started = time.perf_counter()
    emit(job_id, stage, EventType.STARTED)
    try:
        yield
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        emit(
            job_id,
            stage,
            EventType.FAILED,
            payload={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            latency_ms=latency_ms,
        )
        raise
    else:
        latency_ms = (time.perf_counter() - started) * 1000.0
        emit(job_id, stage, EventType.COMPLETED, latency_ms=latency_ms)
