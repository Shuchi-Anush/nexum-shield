from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    # ======================
    # CORE
    # ======================
    ENV: str = "dev"

    # ======================
    # REDIS (CRITICAL)
    # ======================
    REDIS_URL: str = "redis://127.0.0.1:6379/0"

    # ======================
    # JOB SYSTEM
    # ======================
    # Sliding TTL for job records (None = no expiry)
    JOB_TTL_SECONDS: Optional[int] = None

    # ======================
    # FINGERPRINT
    # ======================
    KEYFRAME_INTERVAL_SEC: float = 1.0
    PHASH_SIZE: int = 8                          # 8x8 → 64-bit hash
    MAX_KEYFRAMES: int = 240                     # cap for long videos
    DOWNSCALE_LONG_EDGE: int = 320               # speed; pHash is scale-invariant

    # ======================
    # MATCHING / IDENTITY (Hamming on 64-bit pHash)
    # ======================
    EXACT_THRESHOLD: float = 4.0                 # ≤ → EXACT (resolve to existing)
    VARIANT_THRESHOLD: float = 12.0              # ≤ → VARIANT (new id + edge)
    LSH_PREFIX_BITS: int = 16                    # bucket key width
    LSH_PROBE_RADIUS: int = 1                    # neighbour buckets (Hamming-N prefix)
    MATCHING_TOPK: int = 5                       # candidates surfaced

    # Fusion weights (pHash similarity vs embedding cosine)
    FUSION_WEIGHT_PHASH: float = 0.7
    FUSION_WEIGHT_EMBEDDING: float = 0.3

    # ======================
    # OBSERVATIONS
    # ======================
    OBSERVATION_TTL_SECONDS: int = 7 * 24 * 3600
    OBSERVATION_DEDUP_TTL_SECONDS: int = 3600
    PLATFORM_RECENT_CAP: int = 10_000

    # ======================
    # GRAPH
    # ======================
    GRAPH_MAX_NODES: int = 500
    GRAPH_DEFAULT_DEPTH: int = 3

    # ======================
    # LLM PROVIDER
    # ======================
    LLM_PROVIDER: str = "gemini"                 # "gemini" | future: "claude" | "openai"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT_SECONDS: float = 30.0
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_MAX_CONCURRENCY: int = 8              # asyncio.Semaphore bound
    GEMINI_MAX_REQUEST_BYTES: int = 1_048_576    # 1 MiB serialized payload guard

    # ======================
    # LLM ORCHESTRATION
    # ======================
    LLM_ORCH_BUDGET_KEY: str = "default"
    LLM_ORCH_BUDGET_LIMIT_USD: float = 100.0
    LLM_ORCH_RESERVATION_TTL_SECONDS: int = 300
    LLM_ORCH_REAPER_INTERVAL_SECONDS: float = 30.0
    LLM_ORCH_COST_SAFETY_MARGIN: float = 1.10    # multiplier on estimates

    LLM_ORCH_RATE_HIGH_CAPACITY: int = 60
    LLM_ORCH_RATE_HIGH_REFILL_PER_SEC: float = 1.0
    LLM_ORCH_RATE_NORMAL_CAPACITY: int = 30
    LLM_ORCH_RATE_NORMAL_REFILL_PER_SEC: float = 0.5
    LLM_ORCH_RATE_LOW_CAPACITY: int = 10
    LLM_ORCH_RATE_LOW_REFILL_PER_SEC: float = 0.2

    LLM_ORCH_HEALTH_WINDOW_SIZE: int = 100
    LLM_ORCH_HEALTH_WINDOW_SECONDS: float = 300.0
    LLM_ORCH_HEALTH_MIN_OBSERVATIONS: int = 5

    LLM_ORCH_CACHE_NAMESPACE: str = "llm:cache"
    LLM_ORCH_CACHE_DEFAULT_TTL_SECONDS: int = 3600

    LLM_ORCH_DEFAULT_DEADLINE_SECONDS: float = 30.0
    LLM_ORCH_DEFAULT_MAX_ATTEMPTS: int = 5
    LLM_ORCH_SCHEMA_CORRECTION_ENABLED: bool = True

    # ======================
    # FUTURE (DO NOT USE YET)
    # ======================
    DATABASE_URL: Optional[str] = None
    LOCAL_STORAGE_PATH: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None

    class Config:
        env_file = BASE_DIR / ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
