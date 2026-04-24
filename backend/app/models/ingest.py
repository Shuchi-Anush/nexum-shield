from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any


class IngestRequest(BaseModel):
    source_url: Optional[HttpUrl] = None
    content_type: str  # video | image | stream
    metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    job_id: str
    status: str