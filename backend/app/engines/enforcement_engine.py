"""Enforcement stage — DEPRECATED transitional bridge.

Combines confidence band + matched asset into a structured 3-action
decision (ALLOW/FLAG/BLOCK) with an auditable explanation. The canonical
runtime now performs the DECISION phase via
``app.engines.policy_engine.evaluate_policy`` over a 5-action
``PolicyAction`` ladder (ALLOW/FLAG/REVIEW/RESTRICT/TAKEDOWN) per
docs/specs/policy_engine.md §2.1.

Status:
    DEPRECATED — pipeline_worker no longer invokes this module. Retained
    for the transition window per docs/specs/job_processing.md §5.5
    (Phase C deprecation). The 3-action vocabulary is preserved on the
    ``ENFORCED`` event's ``action`` field for backward-compatible
    consumers; the canonical ``policy_action`` field carries the
    5-level value.

Successor:
    ``app.engines.policy_engine.evaluate_policy`` performs the full PBRA
    decision (PROPOSE / BOUND / REFINE / ASSERT) producing a
    ``PolicyResult`` with deterministic audit lineage
    (``evaluation_hash``, ``triggered_rules``, version triad).
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
