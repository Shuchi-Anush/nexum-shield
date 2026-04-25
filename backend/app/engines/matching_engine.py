"""Matching stage.

Stub implementation: cosine-similarity nearest-neighbour scan over the
in-memory reference registry. Production replaces this with a vector DB
(FAISS / Milvus / Pinecone) plus a coarse fingerprint pre-filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.engines import embedding_engine
from app.services.reference_store import ProtectedAsset, get_all_assets


MATCH_THRESHOLD = 0.3


@dataclass
class MatchResult:
    matched_asset: Optional[ProtectedAsset]
    similarity: float


def find_best_match(candidate_embedding: list[float]) -> MatchResult:
    best_asset: Optional[ProtectedAsset] = None
    best_sim = 0.0

    for asset in get_all_assets():
        ref_embedding = embedding_engine.embed(asset.fingerprint)
        sim = _cosine(candidate_embedding, ref_embedding)
        if sim > best_sim:
            best_sim = sim
            best_asset = asset

    if best_sim < MATCH_THRESHOLD:
        return MatchResult(matched_asset=None, similarity=best_sim)

    return MatchResult(matched_asset=best_asset, similarity=best_sim)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
