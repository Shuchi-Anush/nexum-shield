"""Sequential pipeline worker — idempotent under retries and redelivery.

Consumes from the RQ "pipeline" queue. Each invocation of run_pipeline
acquires a Redis lock at `lock:job:{job_id}` (SET NX EX 300) so concurrent
workers cannot race on the same job, then re-validates that the job is
still QUEUED before transitioning to PROCESSING. Together these make
duplicate deliveries, RQ retries, and crash-resume safe: a second invocation
either fails to take the lock or sees a non-QUEUED status and exits.

The lock is released via a token-checked Lua script so a worker can never
delete the lock of a successor that acquired the key after the original
TTL expired. Any exception during pipeline execution flips the job to
FAILED before the lock is released, so jobs are never stranded in
PROCESSING by a clean exception path.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from app.core.event_store import stage_event
from app.core.job_store import JobStatus, job_store
from app.core.queue import redis_conn
from app.engines import (
    embedding_engine,
    enforcement_engine,
    fingerprint_engine,
    matching_engine,
    scoring_engine,
)


_LOCK_TTL_SECONDS = 300

# Compare-and-delete: only release the lock if the token still matches the
# one we wrote. Prevents releasing a successor's lock if our TTL expired
# mid-execution and another worker has since re-acquired the key.
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def _lock_key(job_id: str) -> str:
    return f"lock:job:{job_id}"


def _acquire_lock(job_id: str) -> Optional[str]:
    token = uuid.uuid4().hex
    acquired = redis_conn.set(
        _lock_key(job_id),
        token,
        nx=True,
        ex=_LOCK_TTL_SECONDS,
    )
    return token if acquired else None


def _release_lock(job_id: str, token: str) -> None:
    try:
        redis_conn.eval(_RELEASE_SCRIPT, 1, _lock_key(job_id), token)
    except Exception:
        # Lock will auto-expire via TTL; never let release errors mask
        # a real pipeline exception or block worker shutdown.
        pass


def run_pipeline(job_id: str) -> None:
    token = _acquire_lock(job_id)
    if token is None:
        # Another worker is already processing this job. Redelivery is a no-op.
        return

    try:
        job = job_store.get_job(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            # State guard: only QUEUED jobs may advance. Anything else
            # (PROCESSING / COMPLETED / FAILED / FLAGGED / missing) means
            # this is a duplicate delivery — exit without side effects.
            return

        try:
            job_store.update_status(job_id, JobStatus.PROCESSING)

            payload: Any = job.metadata or {}

            with stage_event(job_id, "fingerprint"):
                fingerprint = fingerprint_engine.compute_fingerprint(payload)
                job_store.update_stage(job_id, "fingerprint", {"hash": fingerprint})

            with stage_event(job_id, "embedding"):
                vector = embedding_engine.embed(fingerprint)
                job_store.update_stage(
                    job_id,
                    "embedding",
                    {
                        "vector": vector,
                        "model_version": embedding_engine.MODEL_VERSION,
                    },
                )

            with stage_event(job_id, "matching"):
                match = matching_engine.find_best_match(vector)
                matched_asset_dict: Optional[dict] = (
                    {
                        "asset_id": match.matched_asset.asset_id,
                        "owner": match.matched_asset.owner,
                        "trust_level": match.matched_asset.trust_level,
                    }
                    if match.matched_asset is not None
                    else None
                )
                job_store.update_stage(
                    job_id,
                    "matching",
                    {
                        "matched_asset": matched_asset_dict,
                        "similarity": match.similarity,
                    },
                )

            with stage_event(job_id, "scoring"):
                band = scoring_engine.score(match.similarity)
                job_store.update_stage(job_id, "scoring", {"band": band.value})

            with stage_event(job_id, "enforcement"):
                decision = enforcement_engine.decide(
                    input_media_id=fingerprint,
                    matched_asset=matched_asset_dict,
                    similarity=match.similarity,
                    band=band,
                    model_version=embedding_engine.MODEL_VERSION,
                )
                job_store.update_stage(job_id, "enforcement", decision)

            result = {
                "match": match.matched_asset is not None,
                "owner": match.matched_asset.owner if match.matched_asset else None,
                "confidence": match.similarity,
                "action": decision["action"],
                "reason": decision["reason"],
            }
            job_store.set_result(job_id, result)

            terminal = (
                JobStatus.FLAGGED
                if decision["action"] in ("FLAG", "BLOCK")
                else JobStatus.COMPLETED
            )
            job_store.update_status(job_id, terminal)

        except Exception as exc:
            # Pipeline body failed — never strand the job in PROCESSING.
            try:
                job_store.set_failure(job_id, f"{type(exc).__name__}: {exc}")
            except ValueError:
                # Already in a terminal state; nothing to record.
                pass
    finally:
        _release_lock(job_id, token)
