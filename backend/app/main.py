from fastapi import FastAPI
from app.api import health, ingest, jobs
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Nexum Shield API",
        version="1.0.0"
    )

    app.state.settings = settings  # IMPORTANT

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(jobs.router)

    return app


app = create_app()