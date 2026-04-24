# app/models/job.py

from pydantic import BaseModel
from typing import Optional

class Job(BaseModel):
    id: int
    status: str
    result: Optional[str] = None