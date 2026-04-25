"""JobStore — distributed, Redis-Hash-backed.

Each job is persisted as a Redis Hash at key `job:{job_id}` with these fields:

    job_id          str
    status          str  (JobStatus.value)
    created_at      str  (float seconds since epoch)
    updated_at      str  (float seconds since epoch)
    metadata        str  (JSON-encoded dict, "" → None)
    stages          str  (JSON-encoded dict)
    result          str  (JSON-encoded dict, "" → None)
    failure_reason  str  ("" → None)

Public surface (Job, JobStatus, JobStore.{create_job, get_job, update_status,
update_stage, set_result, set_failure}, the module-level `job_store` singleton)
is identical to the prior JSON-blob version, so the API layer, pipeline_worker,
and engines need no changes.

Concurrency: per-field HSET writes never clobber concurrent updates to other
fields. Read-modify-write paths (state transitions, stage merges, failure
marking) wrap their critical sections in WATCH/MULTI/EXEC so the API process
and any number of RQ workers mutate the same job safely without an in-process
lock.

A configurable TTL (`settings.JOB_TTL_SECONDS`) optionally expires job records
on a sliding window (refreshed on each write); when None or non-positive,
jobs persist indefinitely.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Set

from redis.exceptions import WatchError

from app.core.config import get_settings
from app.core.queue import redis_conn


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    FLAGGED = "flagged"


_VALID_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.PROCESSING, JobStatus.FAILED},
    JobStatus.PROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.FLAGGED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.FLAGGED: set(),
}


@dataclass
class Job:
    job_id: str
    status: JobStatus
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Optional[dict] = None
    stages: Dict[str, Any] = field(default_factory=dict)
    result: Optional[dict] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _decode(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else str(value)


def _encode_optional(value: Optional[Any]) -> str:
    return "" if value is None else json.dumps(value)


def _decode_optional(value: Optional[str]) -> Optional[Any]:
    if value is None or value == "":
        return None
    return json.loads(value)


def _decode_dict(value: Optional[str]) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    return json.loads(value)


def _hash_to_job(raw: Mapping[Any, Any]) -> Job:
    if not raw:
        raise ValueError("Job not found")
    decoded: Dict[str, Optional[str]] = {
        _decode(k): _decode(v) for k, v in raw.items()
    }
    return Job(
        job_id=decoded["job_id"],
        status=JobStatus(decoded["status"]),
        created_at=float(decoded["created_at"]),
        updated_at=float(decoded["updated_at"]),
        metadata=_decode_optional(decoded.get("metadata")),
        stages=_decode_dict(decoded.get("stages")),
        result=_decode_optional(decoded.get("result")),
        failure_reason=(decoded.get("failure_reason") or None),
    )


def _ttl_seconds() -> Optional[int]:
    ttl = getattr(get_settings(), "JOB_TTL_SECONDS", None)
    if ttl is None or ttl <= 0:
        return None
    return ttl


def _refresh_ttl(pipe_or_conn: Any, key: str) -> None:
    ttl = _ttl_seconds()
    if ttl:
        pipe_or_conn.expire(key, ttl)


class JobStore:
    """Redis-backed, distributed job store.

    Backing storage is a Redis Hash per job. Per-field updates avoid
    overwriting concurrent writes; read-modify-write paths use
    WATCH/MULTI/EXEC for cross-process atomicity.
    """

    def create_job(self, job_id: str, metadata: Optional[dict] = None) -> Job:
        key = _key(job_id)
        with redis_conn.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    existing = pipe.hgetall(key)
                    if existing:
                        pipe.unwatch()
                        return _hash_to_job(existing)
                    now = time.time()
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "job_id": job_id,
                            "status": JobStatus.QUEUED.value,
                            "created_at": repr(now),
                            "updated_at": repr(now),
                            "metadata": _encode_optional(metadata),
                            "stages": json.dumps({}),
                            "result": "",
                            "failure_reason": "",
                        },
                    )
                    _refresh_ttl(pipe, key)
                    pipe.execute()
                    return Job(
                        job_id=job_id,
                        status=JobStatus.QUEUED,
                        created_at=now,
                        updated_at=now,
                        metadata=metadata,
                    )
                except WatchError:
                    continue

    def get_job(self, job_id: str) -> Optional[Job]:
        raw = redis_conn.hgetall(_key(job_id))
        if not raw:
            return None
        return _hash_to_job(raw)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        key = _key(job_id)
        with redis_conn.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    current_raw = pipe.hget(key, "status")
                    if current_raw is None:
                        pipe.unwatch()
                        raise ValueError("Job not found")
                    current = JobStatus(_decode(current_raw))
                    if status not in _VALID_TRANSITIONS.get(current, set()):
                        pipe.unwatch()
                        raise ValueError(
                            f"Invalid status transition: {current.value} -> {status.value}"
                        )
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "status": status.value,
                            "updated_at": repr(time.time()),
                        },
                    )
                    _refresh_ttl(pipe, key)
                    pipe.execute()
                    return
                except WatchError:
                    continue

    def update_stage(self, job_id: str, stage: str, output: Any) -> None:
        key = _key(job_id)
        with redis_conn.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw_stages = pipe.hget(key, "stages")
                    if raw_stages is None and not pipe.exists(key):
                        pipe.unwatch()
                        raise ValueError("Job not found")
                    stages = _decode_dict(_decode(raw_stages))
                    stages[stage] = output
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "stages": json.dumps(stages),
                            "updated_at": repr(time.time()),
                        },
                    )
                    _refresh_ttl(pipe, key)
                    pipe.execute()
                    return
                except WatchError:
                    continue

    def set_result(self, job_id: str, result: dict) -> None:
        key = _key(job_id)
        if not redis_conn.exists(key):
            raise ValueError("Job not found")
        pipe = redis_conn.pipeline()
        pipe.hset(
            key,
            mapping={
                "result": json.dumps(result),
                "updated_at": repr(time.time()),
            },
        )
        _refresh_ttl(pipe, key)
        pipe.execute()

    def set_failure(self, job_id: str, reason: str) -> None:
        key = _key(job_id)
        with redis_conn.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    current_raw = pipe.hget(key, "status")
                    if current_raw is None:
                        pipe.unwatch()
                        raise ValueError("Job not found")
                    current = JobStatus(_decode(current_raw))
                    if JobStatus.FAILED not in _VALID_TRANSITIONS.get(current, set()):
                        pipe.unwatch()
                        raise ValueError(
                            f"Cannot fail from terminal status: {current.value}"
                        )
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "status": JobStatus.FAILED.value,
                            "failure_reason": reason,
                            "updated_at": repr(time.time()),
                        },
                    )
                    _refresh_ttl(pipe, key)
                    pipe.execute()
                    return
                except WatchError:
                    continue


# GLOBAL SINGLETON
job_store = JobStore()
