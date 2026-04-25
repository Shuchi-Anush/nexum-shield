"""Redis Queue (RQ) wiring.

Owns the singleton Redis connection and the "pipeline" Queue used by the
ingest API to dispatch run_pipeline asynchronously. Replaces FastAPI's
in-process BackgroundTasks so the pipeline can run in a separate, scalable
worker process.
"""

from __future__ import annotations

from redis import Redis
from rq import Queue

from app.core.config import get_settings


_settings = get_settings()

redis_conn: Redis = Redis.from_url(_settings.REDIS_URL)
pipeline_queue: Queue = Queue("pipeline", connection=redis_conn)
