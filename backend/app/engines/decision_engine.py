"""Decision engine.

Pure scoring function fusing five signals — similarity, owner trust,
uploader trust, observation velocity, and match-source quality — into a
composite risk score, a RiskBand classification, and a per-term
breakdown of raw and weighted contributions. Side-effect free: no IO,
no logging, deterministic given the same input. None/NaN inputs are
coerced to 0.0 before clamping; trust signals receive a 0.1 floor only
when their weighted contribution is computed, so breakdown.raw always
reflects the original normalized signal untouched.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from app.models.decision_models import (
    DecisionInput,
    RiskBand,
    RiskBreakdown,
    RiskScore,
    ScoreSignal,
    TermContribution,
    ThresholdConfig,
)


_VELOCITY_K = 100.0
_MAX_OBSERVATIONS = 1000.0
_MIN_TRUST = 0.1

_SIGNAL_SOURCE_QUALITY: Mapping[str, float] = {
    "fingerprint+embedding": 1.00,
    "embedding": 0.90,
    "fingerprint": 0.70,
    "metadata": 0.40,
}
_DEFAULT_SIGNAL_QUALITY = 0.50


def compute_risk(input: DecisionInput, config: ThresholdConfig) -> RiskScore:
    similarity_raw = _clamp01(_safe(input.match.similarity))
    similarity_weighted = config.w_similarity * similarity_raw

    trust_owner_raw = _clamp01(_safe(input.trust_owner.trust_score))
    trust_owner_effective = max(_MIN_TRUST, trust_owner_raw)
    trust_owner_weighted = config.w_trust_owner * trust_owner_effective

    trust_uploader_raw = _clamp01(_safe(input.trust_uploader.trust_score))
    trust_uploader_effective = max(_MIN_TRUST, trust_uploader_raw)
    trust_uploader_weighted = config.w_trust_uploader * (
        1.0 - trust_uploader_effective
    )

    velocity_raw = _velocity_norm(input)
    velocity_weighted = config.w_velocity * velocity_raw

    match_quality_raw = _match_quality(input.score)
    match_quality_weighted = config.w_match_quality * match_quality_raw

    raw_score = (
        similarity_weighted
        + trust_owner_weighted
        + trust_uploader_weighted
        + velocity_weighted
        + match_quality_weighted
    )
    composite = _clamp01(raw_score)

    breakdown = RiskBreakdown(
        similarity=TermContribution(
            raw=similarity_raw, weighted=similarity_weighted
        ),
        trust_owner=TermContribution(
            raw=trust_owner_raw, weighted=trust_owner_weighted
        ),
        trust_uploader=TermContribution(
            raw=trust_uploader_raw, weighted=trust_uploader_weighted
        ),
        velocity=TermContribution(
            raw=velocity_raw, weighted=velocity_weighted
        ),
        match_quality=TermContribution(
            raw=match_quality_raw, weighted=match_quality_weighted
        ),
    )
    return RiskScore(
        composite=composite,
        band=_band(composite, config),
        breakdown=breakdown,
    )


def _velocity_norm(data: DecisionInput) -> float:
    timestamps = _clean_timestamps(data.observation_timestamps)
    if len(timestamps) >= 2:
        span = max(timestamps) - min(timestamps)
        if span > 0.0:
            rate = len(timestamps) / span
            return _clamp01(rate / (rate + _VELOCITY_K))
    # Fallback when timestamps < 2 OR span <= 0.
    if _MAX_OBSERVATIONS <= 0.0:
        return 0.0
    count = max(0.0, _safe(data.observation_count))
    return _clamp01(math.log1p(count) / math.log1p(_MAX_OBSERVATIONS))


def _match_quality(score: ScoreSignal) -> float:
    quality = _SIGNAL_SOURCE_QUALITY.get(
        score.signal_source, _DEFAULT_SIGNAL_QUALITY
    )
    return _clamp01(_safe(quality))


def _band(composite: float, config: ThresholdConfig) -> RiskBand:
    if composite < config.low_upper:
        return RiskBand.LOW
    if composite < config.medium_upper:
        return RiskBand.MEDIUM
    return RiskBand.HIGH


def _clean_timestamps(seq: Sequence[float]) -> tuple[float, ...]:
    if not seq:
        return ()
    out: list[float] = []
    for t in seq:
        if t is None:
            continue
        try:
            f = float(t)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        out.append(f)
    return tuple(out)


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
