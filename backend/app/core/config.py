from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    # --- Core ---
    ENV: str = "development"
    PORT: int = 8000

    # --- Storage ---
    LOCAL_STORAGE_PATH: str = str(BASE_DIR / "uploads")

    # --- Database ---
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'nexum.db'}"

    # --- Queue ---
    REDIS_URL: str = "redis://localhost:6379"

    # --- ML ---
    EMBEDDING_MODEL: str = "clip-base"

    class Config:
        env_file = BASE_DIR / ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()