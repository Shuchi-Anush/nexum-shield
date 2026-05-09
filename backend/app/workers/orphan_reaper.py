"""Orphan PROCESSING reaper.

A worker that crashes mid-pipeline (SIGKILL, OOM, container eviction)
leaves its job at ``status == PROCESSING`` with no live writer. The
distributed lock (``lock:job:{id}``) auto-expires after 300 s; the
JobStore record stays in PROCESSING forever because the state guard
in :func:`run_pipeline` exits silently when ``status != QUEUED``. The
job is permanently stranded.

This module rescues those jobs at worker startup. A reap is safe iff:

  1. ``status == PROCESSING``
  2. ``updated_at < now - stale_after_seconds``  (default 2× lock TTL)
  3. ``lock:job:{id}`` is absent (no live worker holds the lock).

Condition (3) is the critical guard: if a worker is genuinely running,
its lock is present. We never reap a job whose lock exists, even if
``updated_at`` looks stale, because a long-running stage (a real
fingerprint over a 4-hour video) will not refresh ``updated_at``
between stage transitions.

Reaped jobs transition to ``FAILED`` via ``job_store.set_failure`` so
the published ``JOB_FAILED`` event preserves the audit chain.
"""

from __future__ import annotations

import logging
import time
from typing import List

from app.core.event_store import (
    JobFailedPayload,
    PipelineEventType,
    publish_event,
)
from app.core.job_store import JobStatus, _decode, job_store
from app.core.queue import redis_conn

_logger = logging.getLogger(__name__)

# Default: 2× the worker lock TTL (lock_ttl=300 in pipeline_worker).
# A reaped job has been "abandoned" for longer than any worker could
# legitimately hold a lock, so the lock-expiry race is closed.
_DEFAULT_STALE_AFTER_SECONDS = 600.0


def _lock_key(job_id: str) -> str:
    return f"lock:job:{job_id}"


def reap_orphans(
    stale_after_seconds: float = _DEFAULT_STALE_AFTER_SECONDS,
    scan_count: int = 200,
) -> List[str]:
    """Scan all jobs and FAIL any orphaned PROCESSING records.

    Uses ``SCAN`` (not ``KEYS``) so it never blocks the Redis main
    thread on a large keyspace. Returns the list of reaped job_ids.
    """
    now = time.time()
    reaped: List[str] = []

    for raw_key in redis_conn.scan_iter(match="job:*", count=scan_count):
        key = _decode(raw_key)
        if not key or not key.startswith("job:"):
            continue
        job_id = key[len("job:"):]

        # Read status + updated_at in one round trip.
        status_raw, updated_raw = redis_conn.hmget(
            key, "status", "updated_at"
        )
        status = _decode(status_raw)
        if status != JobStatus.PROCESSING.value:
            continue

        # If the lock is still alive, a worker is genuinely running.
        # Never reap.
        if redis_conn.exists(_lock_key(job_id)):
            continue

        try:
            updated_at = float(_decode(updated_raw) or "0")
        except ValueError:
            updated_at = 0.0
        if (now - updated_at) < stale_after_seconds:
            continue

        reason = (
            f"orphaned: PROCESSING with no active lock; "
            f"last updated {now - updated_at:.0f}s ago"
        )
        try:
            job_store.set_failure(job_id, reason)
        except ValueError:
            # Race: another reaper or a recovering worker has already
            # transitioned this job out of PROCESSING. Skip silently.
            continue
        except Exception as exc:
            _logger.warning(
                "orphan reaper failed to mark job FAILED",
                extra={"job_id": job_id, "error": str(exc)},
            )
            continue

        try:
            publish_event(
                job_id,
                PipelineEventType.JOB_FAILED,
                JobFailedPayload(
                    error_type="OrphanedJobReaped",
                    error_message=reason,
                ),
            )
        except Exception as exc:
            # Audit publish failed, but the state transition succeeded.
            # Log and continue — losing one audit row beats a wedged job.
            _logger.warning(
                "orphan reaper failed to publish JOB_FAILED",
                extra={"job_id": job_id, "error": str(exc)},
            )

        reaped.append(job_id)

    if reaped:
        _logger.info(
            "orphan reaper reaped %d job(s) at boot",
            len(reaped),
            extra={"reaped_count": len(reaped)},
        )
    return reaped
