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
    if matched_asset is None:
        action = "ALLOW"
    else:
        trust = matched_asset.get("trust_level")
        if trust == "verified":
            if similarity >= 0.8:
                action = "BLOCK"
            elif similarity >= 0.4:
                action = "FLAG"
            else:
                # Verified rights-holder match: stay suspicious below 0.4.
                action = "FLAG"
        else:
            if similarity >= 0.85:
                action = "BLOCK"
            elif similarity >= 0.5:
                action = "FLAG"
            else:
                action = "ALLOW"

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
        "explanation": _explain(action, matched_asset, similarity),
    }
    return {"action": action, "reason": reason}


def _explain(
    action: str, matched_asset: Optional[dict], similarity: float
) -> str:
    if matched_asset is None:
        return (
            f"No matching protected asset; similarity={similarity:.3f}; "
            f"policy=trust-aware-threshold → {action}."
        )
    return (
        f"Match against {matched_asset.get('asset_id')} "
        f"(owner={matched_asset.get('owner')}, "
        f"trust={matched_asset.get('trust_level')}); "
        f"similarity={similarity:.3f}; "
        f"policy=trust-aware-threshold → {action}."
    )
