"""Confidence-engine data contracts.

Frozen dataclasses for the inputs / outputs of compute_confidence plus
the ConfidenceTier and ConfidenceReasonCode enums. Source of truth:
.claude/memory/confidence_engine_spec.md (v3 final merged).

Trust signals are passed as ``TrustState(trust_score, is_default)``;
``is_default`` is the explicit registry signal — never inferred
numerically. Every policy-level condition (S1..S5, U1..U5, gray zone)
is derived inside the engine from these primitives plus the
ConfidenceConfig values; no caller-supplied policy flags exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class ConfidenceTier(str, Enum):
    LOW = "LOW"        # [0.00, 0.40)  — insufficient for autonomous action
    MEDIUM = "MEDIUM"  # [0.40, 0.70)  — acceptable with caveats
    HIGH = "HIGH"      # [0.70, 1.00]  — suitable for autonomous enforcement


class ConfidenceReasonCode(str, Enum):
    # Spec §9 reason codes.
    NO_MATCH = "NO_MATCH"
    GRAY_ZONE = "GRAY_ZONE"
    OWNER_TRUST_DEFAULT = "OWNER_TRUST_DEFAULT"
    UPLOADER_TRUST_DEFAULT = "UPLOADER_TRUST_DEFAULT"
    LOW_OBSERVATIONS = "LOW_OBSERVATIONS"
    SINGLE_ENGINE_MATCH = "SINGLE_ENGINE_MATCH"
    TRUST_CAP_APPLIED = "TRUST_CAP_APPLIED"
    UNCERTAINTY_GLOBAL_CAP = "UNCERTAINTY_GLOBAL_CAP"
    # Engine-internal extras (kept for backward compatibility).
    BOTH_TRUSTS_DEFAULT = "BOTH_TRUSTS_DEFAULT"
    TRUST_CAP_GRAY_ZONE = "TRUST_CAP_GRAY_ZONE"
    INPUT_VALIDATION_FAILED = "INPUT_VALIDATION_FAILED"
    INPUT_QUALITY_LOW = "INPUT_QUALITY_LOW"


@dataclass(frozen=True)
class TrustState:
    """Registry trust signal for an entity (owner or uploader).

    ``is_default`` is True iff the trust registry has no record for this
    entity. ``trust_score`` is consulted only when ``is_default`` is False;
    when ``is_default`` is True the score MUST be ignored by all callers.
    Default is never inferred from the numeric value of trust_score.
    """

    trust_score: float
    is_default: bool


@dataclass(frozen=True)
class ConfidenceInput:
    """Inputs to compute_confidence — primitives only, no policy flags.

    The engine derives S1..S5, U1..U5, and the gray-zone flag from these
    fields plus :class:`ConfidenceConfig`. Callers MUST NOT precompute
    policy conditions and pass them in.
    """

    match_found: bool
    similarity: float
    trust_owner: TrustState
    trust_uploader: TrustState
    observation_count: int
    signal_source: str
    config_version: str = "v3"


@dataclass(frozen=True)
class ConfidenceConfig:
    # Composite weights — must sum to 1.0 (spec §6: 0.4 / 0.3 / 0.3).
    w_agreement: float = 0.40
    w_completeness: float = 0.30
    w_uncertainty: float = 0.30

    # Laplace smoothing — `(sum + 0.5) / (n + 1)` (spec §3).
    laplace_numerator: float = 0.5
    laplace_denominator_offset: float = 1.0

    # Trust uncertainty cap (spec §5): 0.20 baseline, widened to 0.25 in
    # gray zone so the correlated trust+gray penalty is not discounted.
    trust_cap_default: float = 0.20
    trust_cap_gray_zone: float = 0.25

    # Per-uncertainty magnitudes (spec §5).
    u1_value: float = 0.25  # gray zone
    u2_value: float = 0.15  # owner default
    u3_value: float = 0.10  # uploader default
    u4_value: float = 0.05  # observation_count < threshold
    u5_value: float = 0.05  # signal_source != fusion

    # Global uncertainty ceiling (spec §5).
    uncertainty_global_cap: float = 0.50

    # Tier thresholds (spec §7).
    low_upper: float = 0.40
    medium_upper: float = 0.70

    # S4 / U4 threshold: present iff observation_count >= this value.
    # Spec §4 / §5 both pin this at 3.
    s4_observation_threshold: int = 3

    # S5 / U5 fusion sentinel: signal_source matched against this string.
    # Spec §2 enumerates FINGERPRINT / EMBEDDING / FUSION.
    fusion_signal_source: str = "FUSION"

    # U1 gray-zone bounds (spec §5: 0.75 ≤ similarity < 0.85).
    gray_zone_lower: float = 0.75
    gray_zone_upper: float = 0.85

    def __post_init__(self) -> None:
        wsum = self.w_agreement + self.w_completeness + self.w_uncertainty
        if abs(wsum - 1.0) > 1e-9:
            raise ValueError(
                f"ConfidenceConfig weights must sum to 1.0; got {wsum!r}"
            )


@dataclass(frozen=True)
class ConfidenceBreakdown:
    agreement: float
    completeness: float
    uncertainty: float
    composite: float
    tier: ConfidenceTier
    triggered_conditions: Sequence[ConfidenceReasonCode] = field(
        default_factory=tuple
    )
