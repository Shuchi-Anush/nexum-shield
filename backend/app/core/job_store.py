from typing import Any, Dict, Optional, Set
from enum import Enum
from dataclasses import dataclass, field, asdict
import threading
import time


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    FLAGGED = "flagged"


# Valid forward transitions. Terminal states have no successors.
# Mirrors .claude/rules/job-processing.md (no skipping states).
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


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, metadata: Optional[dict] = None) -> Job:
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return existing
            job = Job(job_id=job_id, status=JobStatus.QUEUED, metadata=metadata)
            self._jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("Job not found")
            valid_next = _VALID_TRANSITIONS.get(job.status, set())
            if status not in valid_next:
                raise ValueError(
                    f"Invalid status transition: {job.status.value} -> {status.value}"
                )
            job.status = status
            job.updated_at = time.time()

    def update_stage(self, job_id: str, stage: str, output: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("Job not found")
            job.stages[stage] = output
            job.updated_at = time.time()

    def set_result(self, job_id: str, result: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("Job not found")
            job.result = result
            job.updated_at = time.time()

    def set_failure(self, job_id: str, reason: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("Job not found")
            valid_next = _VALID_TRANSITIONS.get(job.status, set())
            if JobStatus.FAILED not in valid_next:
                raise ValueError(
                    f"Cannot fail from terminal status: {job.status.value}"
                )
            job.failure_reason = reason
            job.status = JobStatus.FAILED
            job.updated_at = time.time()


# GLOBAL SINGLETON
job_store = JobStore()
