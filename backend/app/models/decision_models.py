"""Decision-pipeline data contracts.

Frozen dataclasses for the inputs and outputs of compute_risk, the
RiskBand enum that classifies the composite score, and the RiskBreakdown
that exposes both raw and weighted per-term contributions for
auditability. Lives in the models package so the engine module stays
purely behavioural.

Also defines the DecisionOutput envelope consumed by the PolicyEngine
(per docs/specs/decision_engine.md §5.3 + docs/specs/policy_engine.md
§4.1). The envelope wraps RiskScore plus an InputSnapshot whose
config_version field carries the *confidence* config version through
to PolicyEngine — see docs/specs/confidence_engine.md §10.1 / C-CE-9.
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
    config_version: str = ""        # threaded from DecisionInput.config_version


# -----------------------------------------------------------------------
# DecisionOutput envelope — consumed by PolicyEngine
# -----------------------------------------------------------------------
#
# Per docs/specs/decision_engine.md §5.3 and docs/specs/policy_engine.md
# §4.1, the PolicyEngine consumes a structural envelope wrapping RiskScore
# plus an InputSnapshot. Materialising this as a concrete dataclass closes
# decision_engine.md D-DE-1 part 1 and is the runtime convergence point
# for the engine triple.


@dataclass(frozen=True)
class MatchInputSnapshot:
    """Match facts captured at the moment risk was computed."""

    matched: bool
    similarity: float


@dataclass(frozen=True)
class InputSnapshot:
    """Snapshot of inputs handed to the DECISION phase.

    The ``config_version`` field carries the **confidence** config version
    through to PolicyEngine (see docs/specs/confidence_engine.md §10.1 /
    C-CE-9). The pipeline worker is responsible for setting it correctly
    when assembling the envelope.
    """

    match: MatchInputSnapshot
    config_version: str


@dataclass(frozen=True)
class DecisionOutput:
    """Envelope consumed by PolicyEngine.evaluate_policy.

    Pairs the EVALUATION-phase ``RiskScore`` with the ``InputSnapshot``
    that produced it. Owned by the decision domain; assembled by the
    pipeline worker (per docs/specs/job_processing.md §5.4).
    """

    risk: RiskScore
    input_snapshot: InputSnapshot
