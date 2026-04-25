from fastapi import APIRouter, BackgroundTasks
from app.models.ingest import IngestRequest, IngestResponse
from app.core.job_store import job_store
from app.workers.pipeline_worker import run_pipeline
import uuid

router = APIRouter(prefix="/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    job_id = str(uuid.uuid4())

    job_store.create_job(job_id, metadata=req.model_dump(mode="json"))
    background_tasks.add_task(run_pipeline, job_id)

    return IngestResponse(
        job_id=job_id,
        status="queued",
    )
