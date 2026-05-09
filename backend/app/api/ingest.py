from fastapi import APIRouter
from rq import Retry
from app.models.ingest import IngestRequest, IngestResponse
from app.core.event_store import (
    IngestReceivedPayload,
    PipelineEventType,
    publish_event,
)
from app.core.job_store import job_store
from app.core.queue import pipeline_queue
import uuid

router = APIRouter(prefix="/v1", tags=["ingest"])

# H-7: bounded retry with backoff. RQ defaults to zero retries; a single
# transient Redis blip terminally fails the job. Three attempts at
# 10/60/300s covers fail-overs and brief network partitions; further
# failures fall through to JOB_FAILED + DLQ (future work).
_INGEST_RETRY = Retry(max=3, interval=[10, 60, 300])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    job_id = str(uuid.uuid4())

    job_store.create_job(job_id, metadata=req.model_dump(mode="json"))
    pipeline_queue.enqueue(
        "app.workers.pipeline_worker.run_pipeline",
        job_id,
        job_timeout=300,
        retry=_INGEST_RETRY,
    )

    # Audit trail starts at the API boundary, after the job is durably
    # created and queued. Route stays thin: validate → create → enqueue →
    # publish → return. No business logic, no blocking work.
    publish_event(
        job_id,
        PipelineEventType.INGEST_RECEIVED,
        IngestReceivedPayload(
            content_type=req.content_type,
            source_url=str(req.source_url) if req.source_url else None,
            has_metadata=req.metadata is not None,
        ),
    )

    return IngestResponse(
        job_id=job_id,
        status="queued",
    )
