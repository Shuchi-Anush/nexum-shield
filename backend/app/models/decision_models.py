"""Decision-pipeline data contracts.

Frozen dataclasses for the inputs and outputs of compute_risk, the
RiskBand enum that classifies the composite score, and the RiskBreakdown
that exposes both raw and weighted per-term contributions for
auditability. Lives in the models package so the engine module stays
purely behavioural.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class RiskBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class MatchSignal:
    similarity: float


@dataclass(frozen=True)
class TrustSignal:
    trust_score: float


@dataclass(frozen=True)
class ScoreSignal:
    signal_source: str


@dataclass(frozen=True)
class DecisionInput:
    match: MatchSignal
    trust_owner: TrustSignal
    trust_uploader: TrustSignal
    score: ScoreSignal
    observation_count: int
    config_version: str
    observation_timestamps: Sequence[float] = field(default_factory=tuple)


@dataclass(frozen=True)
class ThresholdConfig:
    w_similarity: float = 0.45
    w_trust_owner: float = 0.15
    w_trust_uploader: float = 0.10
    w_velocity: float = 0.15
    w_match_quality: float = 0.15
    low_upper: float = 0.50
    medium_upper: float = 0.85

    def __post_init__(self) -> None:
        total = (
            self.w_similarity
            + self.w_trust_owner
            + self.w_trust_uploader
            + self.w_velocity
            + self.w_match_quality
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"ThresholdConfig weights must sum to 1.0; got {total!r}"
            )


@dataclass(frozen=True)
class TermContribution:
    raw: float
    weighted: float


@dataclass(frozen=True)
class RiskBreakdown:
    similarity: TermContribution
    trust_owner: TermContribution
    trust_uploader: TermContribution
    velocity: TermContribution
    match_quality: TermContribution


@dataclass(frozen=True)
class RiskScore:
    composite: float
    band: RiskBand
    breakdown: RiskBreakdown
