"""JobStore — distributed, Redis-backed.

Public surface (Job, JobStatus, JobStore methods, the module-level
`job_store` singleton) is identical to the prior in-memory version so that
the API layer, pipeline_worker, and engines need no changes. Each job is
serialised as a JSON blob under the key `job:{job_id}` so the API process
and RQ worker process share state via Redis.

Concurrency: a local threading.Lock still serialises in-process
read-modify-write sequences. Cross-process atomicity (WATCH/MULTI) is
deferred per the current task scope — see .claude/rules/job-processing.md.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Union

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


def _serialize(job: Job) -> str:
    return json.dumps(asdict(job))


def _deserialize(blob: Union[bytes, str]) -> Job:
    data = json.loads(blob)
    data["status"] = JobStatus(data["status"])
    return Job(**data)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _load(self, job_id: str) -> Job:
        blob = redis_conn.get(_key(job_id))
        if blob is None:
            raise ValueError("Job not found")
        return _deserialize(blob)

    def _save(self, job: Job) -> None:
        redis_conn.set(_key(job.job_id), _serialize(job))

    def create_job(self, job_id: str, metadata: Optional[dict] = None) -> Job:
        with self._lock:
            existing = redis_conn.get(_key(job_id))
            if existing is not None:
                return _deserialize(existing)
            job = Job(job_id=job_id, status=JobStatus.QUEUED, metadata=metadata)
            self._save(job)
            return job

    def get_job(self, job_id: str) -> Optional[Job]:
        blob = redis_conn.get(_key(job_id))
        if blob is None:
            return None
        return _deserialize(blob)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._load(job_id)
            valid_next = _VALID_TRANSITIONS.get(job.status, set())
            if status not in valid_next:
                raise ValueError(
                    f"Invalid status transition: {job.status.value} -> {status.value}"
                )
            job.status = status
            job.updated_at = time.time()
            self._save(job)

    def update_stage(self, job_id: str, stage: str, output: Any) -> None:
        with self._lock:
            job = self._load(job_id)
            job.stages[stage] = output
            job.updated_at = time.time()
            self._save(job)

    def set_result(self, job_id: str, result: dict) -> None:
        with self._lock:
            job = self._load(job_id)
            job.result = result
            job.updated_at = time.time()
            self._save(job)

    def set_failure(self, job_id: str, reason: str) -> None:
        with self._lock:
            job = self._load(job_id)
            valid_next = _VALID_TRANSITIONS.get(job.status, set())
            if JobStatus.FAILED not in valid_next:
                raise ValueError(
                    f"Cannot fail from terminal status: {job.status.value}"
                )
            job.failure_reason = reason
            job.status = JobStatus.FAILED
            job.updated_at = time.time()
            self._save(job)


# GLOBAL SINGLETON
job_store = JobStore()
