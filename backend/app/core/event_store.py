"""Event store / audit log + canonical pipeline event bus.

Append-only audit trail of pipeline execution. Every emit writes two Redis
keys:

  * `event:{job_id}:{ts_ns}`         — JSON blob of the full event
  * `events_by_job:{job_id}`         — sorted set of event keys, scored by
                                        nanosecond timestamp for ordered reads

Two layers ride on the same backing store:

  * Lifecycle audit (`EventType.STARTED|COMPLETED|FAILED` + `stage_event`
    context manager): operational signal — when a stage entered, completed,
    or threw. Used by ops dashboards and the per-stage latency view.

  * Canonical pipeline events (`PipelineEventType` + `publish_event` /
    `consume_events`): the strict-schema domain bus from
    `.claude/rules/eventing.md`. Every event carries `event_id`, `job_id`,
    `timestamp`, `type`, and a Pydantic-validated `payload`. Downstream
    readers (API endpoints, audit, future workers, the LLM orchestrator)
    consume from here.

Both layers persist to the same Redis log and are distinguished by the
`event_type` string in the stored record. Splitting them across separate
keys would fragment the per-job audit timeline; sharing one ordered log
keeps "what happened to job X" answerable with a single sorted-set scan.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Optional, Type, Union

from pydantic import BaseModel

from app.core.queue import redis_conn


# ---------------------------------------------------------------------------
# Lifecycle audit (operational stage signal)
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Canonical pipeline events (.claude/rules/eventing.md)
# ---------------------------------------------------------------------------


class PipelineEventType(str, Enum):
    """Domain event types for the canonical pipeline bus.

    These are the named transitions a downstream consumer can subscribe
    to. They sit alongside lifecycle audit events in the same store.
    """

    INGEST_RECEIVED = "INGEST_RECEIVED"
    FINGERPRINT_READY = "FINGERPRINT_READY"
    EMBEDDING_READY = "EMBEDDING_READY"
    MATCH_FOUND = "MATCH_FOUND"
    MATCH_NOT_FOUND = "MATCH_NOT_FOUND"
    SCORED = "SCORED"
    ENFORCED = "ENFORCED"
    JOB_COMPLETED = "JOB_COMPLETED"
    JOB_FAILED = "JOB_FAILED"


# Per-event payload schemas. Strict by construction: pydantic rejects
# unknown fields by default in v2 only when explicitly configured, but
# the typed signature of `publish_event` enforces the right class at the
# call site, which is the contract we actually want.


class IngestReceivedPayload(BaseModel):
    content_type: str
    source_url: Optional[str] = None
    has_metadata: bool = False


class FingerprintReadyPayload(BaseModel):
    content_hash: str
    model_version: Optional[str] = None
    source_mode: Optional[str] = None


class EmbeddingReadyPayload(BaseModel):
    dimension: int
    model_version: str


class MatchFoundPayload(BaseModel):
    matched_asset_id: str
    similarity: float
    owner: Optional[str] = None
    trust_level: Optional[str] = None


class MatchNotFoundPayload(BaseModel):
    similarity: float


class ScoredPayload(BaseModel):
    band: str
    similarity: float


class EnforcedPayload(BaseModel):
    action: str
    similarity: float
    band: str
    model_version: str
    matched_media_id: Optional[str] = None


class JobCompletedPayload(BaseModel):
    terminal_status: str          # "completed" | "flagged"
    action: str


class JobFailedPayload(BaseModel):
    error_type: str
    error_message: str
    stage: Optional[str] = None


_PAYLOAD_SCHEMA: Dict[PipelineEventType, Type[BaseModel]] = {
    PipelineEventType.INGEST_RECEIVED: IngestReceivedPayload,
    PipelineEventType.FINGERPRINT_READY: FingerprintReadyPayload,
    PipelineEventType.EMBEDDING_READY: EmbeddingReadyPayload,
    PipelineEventType.MATCH_FOUND: MatchFoundPayload,
    PipelineEventType.MATCH_NOT_FOUND: MatchNotFoundPayload,
    PipelineEventType.SCORED: ScoredPayload,
    PipelineEventType.ENFORCED: EnforcedPayload,
    PipelineEventType.JOB_COMPLETED: JobCompletedPayload,
    PipelineEventType.JOB_FAILED: JobFailedPayload,
}


_STAGE_FOR: Dict[PipelineEventType, str] = {
    PipelineEventType.INGEST_RECEIVED: "ingest",
    PipelineEventType.FINGERPRINT_READY: "fingerprint",
    PipelineEventType.EMBEDDING_READY: "embedding",
    PipelineEventType.MATCH_FOUND: "matching",
    PipelineEventType.MATCH_NOT_FOUND: "matching",
    PipelineEventType.SCORED: "scoring",
    PipelineEventType.ENFORCED: "enforcement",
    PipelineEventType.JOB_COMPLETED: "job",
    PipelineEventType.JOB_FAILED: "job",
}


# ---------------------------------------------------------------------------
# Storage primitive
# ---------------------------------------------------------------------------


@dataclass
class Event:
    event_id: str
    job_id: str
    stage: str
    event_type: str            # EventType.value or PipelineEventType.value
    timestamp: float
    payload: Optional[dict] = None
    latency_ms: Optional[float] = None


EventTypeLike = Union[EventType, PipelineEventType, str]


def _event_key(job_id: str, ts_ns: int) -> str:
    return f"event:{job_id}:{ts_ns}"


def _index_key(job_id: str) -> str:
    return f"events_by_job:{job_id}"


def _coerce_type(event_type: EventTypeLike) -> str:
    if isinstance(event_type, Enum):
        return str(event_type.value)
    return event_type


def emit(
    job_id: str,
    stage: str,
    event_type: EventTypeLike,
    payload: Optional[dict] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """Low-level: append a single event to the per-job log.

    Accepts either a lifecycle `EventType`, a canonical `PipelineEventType`,
    or a raw string. Payload is the already-serialisable dict — typed
    callers go through `publish_event` so their payload is validated first.
    """
    ts_ns = time.time_ns()
    event = Event(
        event_id=str(uuid.uuid4()),
        job_id=job_id,
        stage=stage,
        event_type=_coerce_type(event_type),
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


# ---------------------------------------------------------------------------
# Lifecycle audit context manager (unchanged contract)
# ---------------------------------------------------------------------------


@contextmanager
def stage_event(job_id: str, stage: str) -> Iterator[None]:
    """Wrap a stage of work to emit STARTED on entry and COMPLETED/FAILED on exit.

    Re-raises the original exception so the worker's failure path is
    untouched. Latency on the closing event is wall time inside the block.
    """
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


# ---------------------------------------------------------------------------
# Canonical pipeline publisher / consumer
# ---------------------------------------------------------------------------


def publish_event(
    job_id: str,
    event_type: PipelineEventType,
    payload: BaseModel,
    latency_ms: Optional[float] = None,
) -> None:
    """Validate and publish a canonical pipeline event.

    The payload class must match the registered schema for the event
    type — wrong pairing is a programmer error and raises immediately
    rather than corrupting the audit log.
    """
    expected = _PAYLOAD_SCHEMA.get(event_type)
    if expected is None:
        raise ValueError(f"unknown PipelineEventType: {event_type!r}")
    if not isinstance(payload, expected):
        raise TypeError(
            f"payload for {event_type.value} must be {expected.__name__}, "
            f"got {type(payload).__name__}"
        )
    emit(
        job_id=job_id,
        stage=_STAGE_FOR[event_type],
        event_type=event_type,
        payload=payload.model_dump(mode="json"),
        latency_ms=latency_ms,
    )


def consume_events(
    job_id: str,
    event_types: Optional[Iterable[PipelineEventType]] = None,
) -> List[dict]:
    """Read canonical pipeline events for a job, ordered by timestamp.

    With `event_types=None` returns the full per-job log (lifecycle +
    domain). Pass a set of `PipelineEventType` values to filter to just
    those domain transitions — used by downstream readers that should
    not see operational lifecycle audit noise.
    """
    raw = list_events(job_id)
    if event_types is None:
        return raw
    wanted = {et.value for et in event_types}
    return [e for e in raw if e.get("event_type") in wanted]
