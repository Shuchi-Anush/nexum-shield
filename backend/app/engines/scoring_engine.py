"""Scoring stage.

Maps a raw similarity score into a confidence band. Bands drive enforcement:
LOW -> ignore, MEDIUM -> human review, HIGH -> auto-flag. Thresholds live
here as module constants so they can be replaced or driven from settings
later (see .claude/rules/ml-evaluation.md on calibration).
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
