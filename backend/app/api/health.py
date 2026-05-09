"""Liveness + readiness for the API.

`/v1/health` returns 200 only when the Redis control plane is reachable.
Returns 503 when Redis is down so K8s / load balancers can fail traffic
out of an unhealthy pod (M-4). The check is cheap (`PING`) and uses
Redis' configured socket timeout so a hung Redis cannot pin the request.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.queue import redis_conn

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health")
def health_check():
    try:
        if redis_conn.ping():
            return {"status": "healthy", "service": "nexum-shield"}
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "redis_ping_falsy"},
        )
    except Exception as exc:  # redis.exceptions.* — keep broad for any IO err
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "redis_unreachable",
                "error_type": type(exc).__name__,
            },
        )
