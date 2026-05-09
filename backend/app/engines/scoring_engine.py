"""Scoring stage — DEPRECATED transitional bridge.

Maps a raw similarity score into a confidence band. Originally drove
enforcement directly; the canonical runtime now performs risk + confidence
+ policy decomposition via the engine triple
(``app.engines.decision_engine`` + ``app.engines.confidence_engine`` +
``app.engines.policy_engine``).

Status:
    DEPRECATED — pipeline_worker no longer invokes this module. Retained
    for the transition window per docs/specs/job_processing.md §5.5
    (Phase C deprecation). Removal scheduled when external consumers (if
    any) migrate; tracked under STATE.md as DEPRECATED once the registry
    catches up to the runtime.

Successor:
    ``app.engines.decision_engine.compute_risk`` produces the canonical
    ``RiskBand`` over a five-term composite risk score; the current
    ``ConfidenceBand`` enum is its legacy 3-level analog.
"""

from __future__ import annotations

from enum import Enum


class ConfidenceBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


LOW_UPPER = 0.50
MEDIUM_UPPER = 0.85


def score(similarity: float) -> ConfidenceBand:
    if similarity < LOW_UPPER:
        return ConfidenceBand.LOW
    if similarity < MEDIUM_UPPER:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.HIGH
