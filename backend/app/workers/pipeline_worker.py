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

At each stage transition the worker publishes a canonical pipeline event
(`.claude/rules/eventing.md`) via `publish_event`, alongside the lifecycle
audit emitted by the `stage_event` context manager. Domain events carry
strict typed payloads; lifecycle events carry latency. Both share the same
per-job audit log, so downstream readers reconstruct the timeline with a
single sorted-set scan.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from app.core.event_store import (
    EmbeddingReadyPayload,
    EnforcedPayload,
    FingerprintReadyPayload,
    JobCompletedPayload,
    JobFailedPayload,
    MatchFoundPayload,
    MatchNotFoundPayload,
    PipelineEventType,
    ScoredPayload,
    publish_event,
    stage_event,
)
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

            # ------------------------------------------------------------
            # Fingerprint
            # ------------------------------------------------------------
            with stage_event(job_id, "fingerprint"):
                fingerprint = fingerprint_engine.compute_fingerprint(payload)
                content_hash = fingerprint.content_hash
                job_store.update_stage(
                    job_id, "fingerprint", {"hash": content_hash}
                )

            publish_event(
                job_id,
                PipelineEventType.FINGERPRINT_READY,
                FingerprintReadyPayload(
                    content_hash=content_hash,
                    model_version=fingerprint.model_version,
                    source_mode=fingerprint.source_mode,
                ),
            )

            # ------------------------------------------------------------
            # Embedding
            # ------------------------------------------------------------
            with stage_event(job_id, "embedding"):
                vector = embedding_engine.embed(content_hash)
                job_store.update_stage(
                    job_id,
                    "embedding",
                    {
                        "vector": vector,
                        "model_version": embedding_engine.MODEL_VERSION,
                    },
                )

            publish_event(
                job_id,
                PipelineEventType.EMBEDDING_READY,
                EmbeddingReadyPayload(
                    dimension=len(vector),
                    model_version=embedding_engine.MODEL_VERSION,
                ),
            )

            # ------------------------------------------------------------
            # Matching
            # ------------------------------------------------------------
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

            if match.matched_asset is not None:
                publish_event(
                    job_id,
                    PipelineEventType.MATCH_FOUND,
                    MatchFoundPayload(
                        matched_asset_id=match.matched_asset.asset_id,
                        similarity=match.similarity,
                        owner=match.matched_asset.owner,
                        trust_level=match.matched_asset.trust_level,
                    ),
                )
            else:
                publish_event(
                    job_id,
                    PipelineEventType.MATCH_NOT_FOUND,
                    MatchNotFoundPayload(similarity=match.similarity),
                )

            # ------------------------------------------------------------
            # Scoring
            # ------------------------------------------------------------
            with stage_event(job_id, "scoring"):
                band = scoring_engine.score(match.similarity)
                job_store.update_stage(job_id, "scoring", {"band": band.value})

            publish_event(
                job_id,
                PipelineEventType.SCORED,
                ScoredPayload(band=band.value, similarity=match.similarity),
            )

            # ------------------------------------------------------------
            # Enforcement
            # ------------------------------------------------------------
            with stage_event(job_id, "enforcement"):
                decision = enforcement_engine.decide(
                    input_media_id=content_hash,
                    matched_asset=matched_asset_dict,
                    similarity=match.similarity,
                    band=band,
                    model_version=embedding_engine.MODEL_VERSION,
                )
                job_store.update_stage(job_id, "enforcement", decision)

            publish_event(
                job_id,
                PipelineEventType.ENFORCED,
                EnforcedPayload(
                    action=decision["action"],
                    similarity=match.similarity,
                    band=band.value,
                    model_version=embedding_engine.MODEL_VERSION,
                    matched_media_id=(
                        matched_asset_dict["asset_id"]
                        if matched_asset_dict
                        else None
                    ),
                ),
            )

            # ------------------------------------------------------------
            # Terminal transition
            # ------------------------------------------------------------
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

            publish_event(
                job_id,
                PipelineEventType.JOB_COMPLETED,
                JobCompletedPayload(
                    terminal_status=terminal.value,
                    action=decision["action"],
                ),
            )

        except Exception as exc:
            # Pipeline body failed — never strand the job in PROCESSING.
            # Order: try to flip state first, then publish JOB_FAILED only
            # if we owned the transition, so we never double-publish a
            # terminal event for a job another writer already failed.
            failed = False
            try:
                job_store.set_failure(job_id, f"{type(exc).__name__}: {exc}")
                failed = True
            except ValueError:
                # Already in a terminal state; nothing to record.
                pass

            if failed:
                publish_event(
                    job_id,
                    PipelineEventType.JOB_FAILED,
                    JobFailedPayload(
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ),
                )
    finally:
        _release_lock(job_id, token)
