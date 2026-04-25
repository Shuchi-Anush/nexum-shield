"""Enforcement stage.

Combines confidence band + matched asset into a structured action plus an
auditable explanation. Every decision carries the full evidence record
required by .claude/rules/enforcement.md (input id, matched id, similarity,
model version, timestamp).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.engines.scoring_engine import ConfidenceBand


_BAND_TO_ACTION = {
    ConfidenceBand.LOW: "ALLOW",
    ConfidenceBand.MEDIUM: "FLAG",
    ConfidenceBand.HIGH: "BLOCK",
}


def decide(
    *,
    input_media_id: str,
    matched_asset: Optional[dict],
    similarity: float,
    band: ConfidenceBand,
    model_version: str,
) -> dict[str, Any]:
    action = _BAND_TO_ACTION[band] if matched_asset is not None else "ALLOW"

    reason = {
        "input_media_id": input_media_id,
        "matched_media_id": (
            matched_asset.get("asset_id") if matched_asset else None
        ),
        "owner": matched_asset.get("owner") if matched_asset else None,
        "trust_level": (
            matched_asset.get("trust_level") if matched_asset else None
        ),
        "similarity_score": similarity,
        "band": band.value,
        "model_version": model_version,
        "timestamp": time.time(),
        "explanation": _explain(action, band, matched_asset),
    }
    return {"action": action, "reason": reason}


def _explain(
    action: str, band: ConfidenceBand, matched_asset: Optional[dict]
) -> str:
    if matched_asset is None:
        return "No matching protected asset; content allowed."
    return (
        f"Match against {matched_asset.get('asset_id')} "
        f"(owner={matched_asset.get('owner')}, "
        f"trust={matched_asset.get('trust_level')}); "
        f"confidence={band.value} -> {action}."
    )
