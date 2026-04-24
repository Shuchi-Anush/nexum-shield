from typing import Dict, Optional
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


@dataclass
class Job:
    job_id: str
    status: JobStatus
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Optional[dict] = None

    def to_dict(self):
        return asdict(self)


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, metadata: Optional[dict] = None) -> Job:
        with self._lock:
            job = Job(job_id=job_id, status=JobStatus.QUEUED, metadata=metadata)
            self._jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError("Job not found")
            job.status = status
            job.updated_at = time.time()


# GLOBAL SINGLETON
job_store = JobStore()