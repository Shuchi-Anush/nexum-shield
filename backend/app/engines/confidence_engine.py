"""Confidence engine.

Pure function that produces a ConfidenceBreakdown — agreement,
completeness, uncertainty, composite, tier, and a list of reason
codes — from a ConfidenceInput. No IO, no logging, deterministic.

Source of truth: .claude/memory/confidence_engine_spec.md (v3).

All policy conditions (S1..S5, U1..U5, gray zone) are derived inside
the engine from the input primitives and ConfidenceConfig values.
Trust defaults are taken from ``trust_*.is_default``; trust scores are
consulted only when is_default is False.
"""

from __future__ import annotations

import math
from typing import Sequence

from app.models.confidence_models import (
    ConfidenceBreakdown,
    ConfidenceConfig,
    ConfidenceInput,
    ConfidenceReasonCode,
    ConfidenceTier,
)


def compute_confidence(
    input: ConfidenceInput, config: ConfidenceConfig
) -> ConfidenceBreakdown:
    if not bool(input.match_found):
        return _no_match_result(config)

    reasons: list[ConfidenceReasonCode] = []

    agreement, agreement_reasons = _agreement(input, config)
    reasons.extend(agreement_reasons)

    completeness = _completeness(input, config)

    uncertainty, uncertainty_reasons = _uncertainty(input, config)
    reasons.extend(uncertainty_reasons)

    composite = _composite(agreement, completeness, uncertainty, config)

    return ConfidenceBreakdown(
        agreement=agreement,
        completeness=completeness,
        uncertainty=uncertainty,
        composite=composite,
        tier=_tier(composite, config),
        triggered_conditions=tuple(reasons),
    )


# --- Stage helpers ----------------------------------------------------


def _no_match_result(config: ConfidenceConfig) -> ConfidenceBreakdown:
    """Spec §8 special case — match_found == False forces the fixed
    triple (agreement=0.50, completeness=0, uncertainty=0). Every Ui in
    the spec is gated by ``AND match_found`` so no uncertainty fires
    here, and no S* signal is countable.
    """
    agreement = 0.5
    completeness = 0.0
    uncertainty = 0.0
    composite = _composite(agreement, completeness, uncertainty, config)
    return ConfidenceBreakdown(
        agreement=agreement,
        completeness=completeness,
        uncertainty=uncertainty,
        composite=composite,
        tier=_tier(composite, config),
        triggered_conditions=(ConfidenceReasonCode.NO_MATCH,),
    )


def _agreement(
    input: ConfidenceInput, config: ConfidenceConfig
) -> tuple[float, list[ConfidenceReasonCode]]:
    """Magnitude-aware agreement with R2..R5 pair exclusion (spec §3)."""
    reasons: list[ConfidenceReasonCode] = []

    norm_sim = _clamp01(_safe(input.similarity))
    owner_default = bool(input.trust_owner.is_default)
    uploader_default = bool(input.trust_uploader.is_default)

    if owner_default and uploader_default:  # R2
        reasons.append(ConfidenceReasonCode.BOTH_TRUSTS_DEFAULT)
        return _smoothed_agreement([], config), reasons

    if owner_default:  # R3 — sim ↔ uploader only
        reasons.append(ConfidenceReasonCode.OWNER_TRUST_DEFAULT)
        norm_uploader = _clamp01(
            1.0 - _safe(input.trust_uploader.trust_score)
        )
        pairs = [_pair(norm_sim, norm_uploader)]
    elif uploader_default:  # R4 — sim ↔ owner only
        reasons.append(ConfidenceReasonCode.UPLOADER_TRUST_DEFAULT)
        norm_owner = _clamp01(_safe(input.trust_owner.trust_score))
        pairs = [_pair(norm_sim, norm_owner)]
    else:  # R5 — all three pairs
        norm_owner = _clamp01(_safe(input.trust_owner.trust_score))
        norm_uploader = _clamp01(
            1.0 - _safe(input.trust_uploader.trust_score)
        )
        pairs = [
            _pair(norm_sim, norm_owner),
            _pair(norm_sim, norm_uploader),
            _pair(norm_owner, norm_uploader),
        ]

    return _smoothed_agreement(pairs, config), reasons


def _pair(norm_a: float, norm_b: float) -> float:
    """Continuous pair score: ``1 - |norm_a - norm_b|``, clamped to [0,1]."""
    return _clamp01(1.0 - abs(norm_a - norm_b))


def _smoothed_agreement(
    pair_scores: Sequence[float], config: ConfidenceConfig
) -> float:
    n = len(pair_scores)
    return _clamp01(
        (sum(pair_scores) + config.laplace_numerator)
        / (n + config.laplace_denominator_offset)
    )


def _completeness(
    input: ConfidenceInput, config: ConfidenceConfig
) -> float:
    """Engine-derived S1..S5 (spec §4).

    S1: match_found
    S2: trust_owner.is_default == False
    S3: trust_uploader.is_default == False
    S4: observation_count >= s4_observation_threshold
    S5: signal_source == fusion_signal_source
    """
    s1 = bool(input.match_found)
    s2 = not bool(input.trust_owner.is_default)
    s3 = not bool(input.trust_uploader.is_default)
    s4 = int(input.observation_count) >= int(config.s4_observation_threshold)
    s5 = _norm_source(input.signal_source) == _norm_source(
        config.fusion_signal_source
    )
    signals_present = sum(1 for s in (s1, s2, s3, s4, s5) if s)
    return _clamp01(signals_present / 5.0)


def _uncertainty(
    input: ConfidenceInput, config: ConfidenceConfig
) -> tuple[float, list[ConfidenceReasonCode]]:
    """U1 + min(U2+U3, trust_cap) + U4 + U5, clamped by global cap (spec §5).

    Only invoked from the matched path; the no-match branch sets
    uncertainty = 0.0 directly per spec §8 (every Ui is gated by
    ``AND match_found``).
    """
    reasons: list[ConfidenceReasonCode] = []

    norm_sim = _clamp01(_safe(input.similarity))
    gray_zone = (
        config.gray_zone_lower <= norm_sim < config.gray_zone_upper
    )

    u1 = config.u1_value if gray_zone else 0.0
    if u1 > 0.0:
        reasons.append(ConfidenceReasonCode.GRAY_ZONE)

    owner_default = bool(input.trust_owner.is_default)
    uploader_default = bool(input.trust_uploader.is_default)
    u2 = config.u2_value if owner_default else 0.0
    u3 = config.u3_value if uploader_default else 0.0

    if gray_zone:
        trust_cap = config.trust_cap_gray_zone
        if (u2 + u3) > 0.0:
            reasons.append(ConfidenceReasonCode.TRUST_CAP_GRAY_ZONE)
    else:
        trust_cap = config.trust_cap_default

    trust_unbounded = u2 + u3
    trust_combined = min(trust_unbounded, trust_cap)
    if trust_unbounded > trust_combined:
        reasons.append(ConfidenceReasonCode.TRUST_CAP_APPLIED)

    low_observations = int(input.observation_count) < int(
        config.s4_observation_threshold
    )
    u4 = config.u4_value if low_observations else 0.0
    if u4 > 0.0:
        reasons.append(ConfidenceReasonCode.LOW_OBSERVATIONS)

    not_fusion = _norm_source(input.signal_source) != _norm_source(
        config.fusion_signal_source
    )
    u5 = config.u5_value if not_fusion else 0.0
    if u5 > 0.0:
        reasons.append(ConfidenceReasonCode.SINGLE_ENGINE_MATCH)

    raw = u1 + trust_combined + u4 + u5
    capped = min(raw, config.uncertainty_global_cap)
    if raw > capped:
        reasons.append(ConfidenceReasonCode.UNCERTAINTY_GLOBAL_CAP)

    return _clamp01(capped), reasons


def _composite(
    agreement: float,
    completeness: float,
    uncertainty: float,
    config: ConfidenceConfig,
) -> float:
    return _clamp01(
        config.w_agreement * agreement
        + config.w_completeness * completeness
        + config.w_uncertainty * (1.0 - uncertainty)
    )


def _tier(composite: float, config: ConfidenceConfig) -> ConfidenceTier:
    if composite < config.low_upper:
        return ConfidenceTier.LOW
    if composite < config.medium_upper:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.HIGH


# --- Primitives -------------------------------------------------------


def _safe(x: object) -> float:
    if x is None:
        return 0.0
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return f


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _norm_source(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()
