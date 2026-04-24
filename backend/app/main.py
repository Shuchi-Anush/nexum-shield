from fastapi import FastAPI
from app.api import health, ingest, jobs


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nexum Shield API",
        version="1.0.0"
    )

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(jobs.router)

    return app


app = create_app()