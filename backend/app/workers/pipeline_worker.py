"""Sequential pipeline worker.

Runs fingerprint -> embedding -> matching -> scoring -> enforcement against
the in-memory JobStore. The function is synchronous and stateless: FastAPI
BackgroundTasks dispatches it to its threadpool so the ingest response path
is never blocked. When BackgroundTasks is later replaced by a real queue
(Redis / Kafka / Cloud Tasks) this same body becomes the consumer, unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.job_store import JobStatus, job_store
from app.engines import (
    embedding_engine,
    enforcement_engine,
    fingerprint_engine,
    matching_engine,
    scoring_engine,
)


def run_pipeline(job_id: str) -> None:
    job = job_store.get_job(job_id)
    if job is None:
        return

    try:
        job_store.update_status(job_id, JobStatus.PROCESSING)

        payload: Any = job.metadata or {}

        fingerprint = fingerprint_engine.compute_fingerprint(payload)
        job_store.update_stage(job_id, "fingerprint", {"hash": fingerprint})

        vector = embedding_engine.embed(fingerprint)
        job_store.update_stage(
            job_id,
            "embedding",
            {
                "vector": vector,
                "model_version": embedding_engine.MODEL_VERSION,
            },
        )

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

        band = scoring_engine.score(match.similarity)
        job_store.update_stage(job_id, "scoring", {"band": band.value})

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
        try:
            job_store.set_failure(job_id, f"{type(exc).__name__}: {exc}")
        except ValueError:
            pass
