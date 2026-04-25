from fastapi import APIRouter, HTTPException
from app.core.event_store import list_events
from app.core.job_store import job_store

router = APIRouter(prefix="/v1", tags=["jobs"])


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job.to_dict()


@router.get("/jobs/{job_id}/events")
def get_job_events(job_id: str):
    return list_events(job_id)