"""In-memory registry of protected assets.

Temporary scaffolding: production replaces this with a vector DB plus a
rights-holder catalogue. Seeds are constructed via IngestRequest so that
fingerprints computed here exactly match what the worker computes for an
identical incoming request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.engines import fingerprint_engine
from app.models.ingest import IngestRequest


@dataclass(frozen=True)
class ProtectedAsset:
    asset_id: str
    fingerprint: str
    owner: str
    trust_level: str  # verified | premium | basic


_SEED_SOURCES = [
    {
        "asset_id": "asset_001",
        "owner": "ESPN",
        "trust_level": "verified",
        "request": IngestRequest(
            source_url="https://espn.com/highlights/match-001.mp4",
            content_type="video",
        ),
    },
    {
        "asset_id": "asset_002",
        "owner": "FIFA",
        "trust_level": "verified",
        "request": IngestRequest(
            source_url="https://fifa.com/world-cup/final-2026.mp4",
            content_type="video",
        ),
    },
    {
        "asset_id": "asset_003",
        "owner": "NBA",
        "trust_level": "premium",
        "request": IngestRequest(
            source_url="https://nba.com/clips/buzzer-beater.mp4",
            content_type="video",
        ),
    },
]


_REGISTRY: List[ProtectedAsset] = [
    ProtectedAsset(
        asset_id=entry["asset_id"],
        fingerprint=fingerprint_engine.compute_fingerprint(
            entry["request"].model_dump(mode="json")
        ),
        owner=entry["owner"],
        trust_level=entry["trust_level"],
    )
    for entry in _SEED_SOURCES
]


def get_all_assets() -> List[ProtectedAsset]:
    return list(_REGISTRY)
