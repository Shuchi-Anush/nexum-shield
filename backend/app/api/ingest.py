from fastapi import APIRouter
from app.models.ingest import IngestRequest, IngestResponse
from app.core.job_store import job_store
import uuid

router = APIRouter(prefix="/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    job_id = str(uuid.uuid4())

    job_store.create_job(job_id, metadata=req.model_dump())

    return IngestResponse(
        job_id=job_id,
        status="queued"
    )