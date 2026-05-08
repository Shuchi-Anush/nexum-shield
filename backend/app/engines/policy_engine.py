"""Policy engine.

Pure, deterministic policy evaluation. Maps a (DecisionOutput,
ConfidenceBreakdown, PolicyContext) triple onto a final ``PolicyAction``
through a strict 6-phase pipeline. No I/O outside an ERROR-level log
emitted only when the terminal invariant guard fires (a programmer-error
indicator that should NEVER trigger if phase logic is correct).

Source of truth — merged spec versions:

  v1 — base matrix, S1..S5 / R1..R5 catalogue, ActionTrace shape.
  v2 — typed enums, EvidenceStrength derived internally, S4 caps at
        FLAG, named 6-phase pipeline, simplified audit, version triad,
        recent_violations_count signal.
  v3 — phase5_ceiling formalized, R1 gated on confidence >= 0.30,
        evaluation_hash + rules_checked_count, multi-match evidence,
        normalized prior_dispute_outcomes, decision/confidence config
        version echo.
  v4 — phase5_ceiling split (GLOBAL_MAX_UPGRADE vs phase4_action),
        evaluation_hash extended with risk_band / confidence_tier /
        confidence_composite, multi-match bumps one level (STRONG only
        with distinct_owner_count >= 2 + similarity >= 0.70), disputes
        preserve order, R2 skipped when evidence == STRONG.
  v5 — terminal invariant guard, evaluation_hash extended to 12 fields
        (signal_source, similarity, is_first_offense, has_prior_disputes,
        distinct_matched_owner_count), multi-match stabilization (no
        MODERATE → STRONG via single-owner multi-match), signal_source
        normalization at entry.

PBRA execution model (canonical mental model)
---------------------------------------------

  PROPOSE  — base-matrix lookup yields the proposed action from
             (risk_band × confidence_tier).
  BOUND    — safety rules narrow ``FeasibleBounds.upper`` to the
             lowest cap target across S2..S5 (S1 is a hard override
             — forces ALLOW and short-circuits BOUND/REFINE/ASSERT).
             The current action is also clamped to that bound when
             a safety rule fires.
  REFINE   — risk-control rules: R1 / R4 upgrades clamp to
             ``FeasibleBounds.upper``; R2 / R3 / R5 downgrades have
             final authority (Phase 6 wins over Phase 5).
  ASSERT   — terminal invariant guard catches any TAKEDOWN that
             leaked through despite ``confidence_tier != HIGH``.

Phase numbering (retained from v2 §5 for spec traceability):

  Phase 1 = PROPOSE override (S1).
  Phase 2 = BOUND upper-cap   (S2, S3 — matrix-evolution guards).
  Phase 3 = BOUND lower-cap   (S4 — confidence floor).
  Phase 4 = BOUND type-cap    (S5).
  Phase 5 = REFINE upgrade    (R1, R4).
  Phase 6 = REFINE downgrade  (R2, R3, R5).
  ASSERT  = terminal invariant guard (post-Phase-6 backstop for S2).

v2 §5 / v3 §1 / v4 §1 reconciliation
------------------------------------
The literal v3/v4 reading ``phase5_ceiling = phase4_action`` is
algebraically inconsistent with the v2 §5 worked example
(``After Phase 4: REVIEW. Phase 5: R4 fires → upgrade to RESTRICT``):
under the literal reading R1 and R4 are unreachable in 100% of inputs
(R4 needs ``current == REVIEW``, which forces ``phase4_action ==
REVIEW``, which collapses the ceiling to REVIEW and clamps R4's
effective target back to REVIEW; R1 collapses similarly). v3 §3 also
mandates ``EXPECTED_RULE_COUNT = 10`` — every rule must be operationally
meaningful.

The semantic reading — "Phase 5 cannot exceed what safety **allowed**"
where *allowed* = the lowest cap target of safety rules that actually
fired — is the unique interpretation that satisfies (a) the v2 example,
(b) every v3 §1 / v4 §1 trace, and (c) the v3 completeness contract.
That is what ``FeasibleBounds.upper`` tracks here.

Matrix-evolution guards (S2, S3)
--------------------------------
S2 and S3 are intentionally unreachable from the current
``_BASE_MATRIX``: TAKEDOWN appears only at HIGH×HIGH (so S2's
``confidence_tier != HIGH`` precondition cannot fire from PROPOSE),
and RESTRICT appears only at HIGH×MEDIUM (so S3's ``confidence_tier
== LOW`` precondition cannot fire). They remain in the rule count
as **matrix-evolution guards**: should a future change alter the
matrix to propose TAKEDOWN at non-HIGH confidence or RESTRICT at
LOW confidence, S2/S3 immediately become active downgrades.
:func:`_validate_matrix_constraint_consistency` runs at import time
to detect such matrix changes during code review rather than silently
in production.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable, Protocol

from app.models.confidence_models import (
    ConfidenceBreakdown,
    ConfidenceReasonCode,
    ConfidenceTier,
)
from app.models.decision_models import RiskBand
from app.models.policy_models import (
    ActionTrace,
    EvidenceStrength,
    PolicyAction,
    PolicyContext,
    PolicyResult,
)


# ---------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------

POLICY_VERSION = "v1.0"
EXPECTED_RULE_COUNT = 10  # 5 safety (S1..S5) + 5 risk control (R1..R5)

GLOBAL_MAX_UPGRADE: PolicyAction = PolicyAction.RESTRICT  # Phase 5 ceiling

CONFIDENCE_FLOOR = 0.30  # S4 trigger threshold
GRAY_ZONE_CONFIDENCE_UPPER = 0.75  # R1 upper bound
GRAY_ZONE_CONFIDENCE_LOWER = 0.30  # R1 lower bound (v3 fix)
VERIFIED_OWNER_SIMILARITY = 0.85  # R4 similarity threshold

# Multi-match evidence thresholds (v5 stabilization)
MULTI_MATCH_DISTINCT_SIMILARITY = 0.70

# Evidence base thresholds
FUSION_STRONG_SIMILARITY = 0.70
FUSION_MODERATE_SIMILARITY = 0.40
SINGLE_MODERATE_SIMILARITY = 0.70

SUPPORTED_ENFORCEMENT_TYPES: frozenset[str] = frozenset({"video", "image"})

VALID_DISPUTE_OUTCOMES: frozenset[str] = frozenset(
    {"upheld", "overturned", "withdrawn", "pending"}
)
MAX_DISPUTE_HISTORY = 5

VALID_SIGNAL_SOURCES: frozenset[str] = frozenset(
    {"fingerprint", "embedding", "fusion"}
)
DEFAULT_SIGNAL_SOURCE = "fingerprint"  # safe fallback (cannot reach STRONG alone)


# Rule IDs (canonical form used in triggered_rules / audit)
S1_NO_MATCH = "S1_NO_MATCH"
S2_TAKEDOWN_CONFIDENCE_GATE = "S2_TAKEDOWN_CONFIDENCE_GATE"
S3_RESTRICT_CONFIDENCE_GATE = "S3_RESTRICT_CONFIDENCE_GATE"
S4_CONFIDENCE_CEILING = "S4_CONFIDENCE_CEILING"
S5_CONTENT_TYPE_GATE = "S5_CONTENT_TYPE_GATE"
R1_GRAY_ZONE = "R1_GRAY_ZONE"
R2_FIRST_OFFENSE = "R2_FIRST_OFFENSE"
R3_EVIDENCE_GATE = "R3_EVIDENCE_GATE"
R4_VERIFIED_OWNER = "R4_VERIFIED_OWNER"
R5_DISPUTE_CAUTION = "R5_DISPUTE_CAUTION"
INVARIANT_TAKEDOWN_GUARD = "INVARIANT_TAKEDOWN_GUARD"


# Severity ordering for downgrade / upgrade arithmetic.
_SEVERITY: dict[PolicyAction, int] = {
    PolicyAction.ALLOW: 0,
    PolicyAction.FLAG: 1,
    PolicyAction.REVIEW: 2,
    PolicyAction.RESTRICT: 3,
    PolicyAction.TAKEDOWN: 4,
}

_TIER_ORDER: dict[ConfidenceTier, int] = {
    ConfidenceTier.LOW: 0,
    ConfidenceTier.MEDIUM: 1,
    ConfidenceTier.HIGH: 2,
}


# Decision × Confidence → base PolicyAction (v1 §3).
_BASE_MATRIX: dict[tuple[RiskBand, ConfidenceTier], PolicyAction] = {
    (RiskBand.LOW,    ConfidenceTier.LOW):    PolicyAction.ALLOW,
    (RiskBand.LOW,    ConfidenceTier.MEDIUM): PolicyAction.ALLOW,
    (RiskBand.LOW,    ConfidenceTier.HIGH):   PolicyAction.ALLOW,
    (RiskBand.MEDIUM, ConfidenceTier.LOW):    PolicyAction.FLAG,
    (RiskBand.MEDIUM, ConfidenceTier.MEDIUM): PolicyAction.REVIEW,
    (RiskBand.MEDIUM, ConfidenceTier.HIGH):   PolicyAction.REVIEW,
    (RiskBand.HIGH,   ConfidenceTier.LOW):    PolicyAction.REVIEW,
    (RiskBand.HIGH,   ConfidenceTier.MEDIUM): PolicyAction.RESTRICT,
    (RiskBand.HIGH,   ConfidenceTier.HIGH):   PolicyAction.TAKEDOWN,
}


# Priority used to select primary_reason. Higher wins. Phase-6 downgrades
# rank above Phase-5 upgrades because Phase 6 is final authority.
_RULE_PRIORITY: dict[str, int] = {
    S1_NO_MATCH: 100,
    INVARIANT_TAKEDOWN_GUARD: 95,
    S2_TAKEDOWN_CONFIDENCE_GATE: 90,
    S3_RESTRICT_CONFIDENCE_GATE: 80,
    S4_CONFIDENCE_CEILING: 70,
    S5_CONTENT_TYPE_GATE: 60,
    R2_FIRST_OFFENSE: 50,
    R3_EVIDENCE_GATE: 45,
    R5_DISPUTE_CAUTION: 40,
    R4_VERIFIED_OWNER: 30,
    R1_GRAY_ZONE: 20,
}


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Structural typing for the upstream DecisionEngine output
# ---------------------------------------------------------------------
#
# v2 spec §1: PolicyContext duplicated DecisionOutput / ConfidenceBreakdown
# fields. Fix is to read them from the upstream objects directly. Since
# the full DecisionOutput type is defined in a sibling module not yet
# committed, we use a Protocol to express the structural contract here.


class _MatchAccess(Protocol):
    matched: bool
    similarity: float


class _InputSnapshotAccess(Protocol):
    match: _MatchAccess
    config_version: str


class _RiskAccess(Protocol):
    band: RiskBand
    composite: float
    config_version: str


class DecisionOutput(Protocol):
    """Structural contract for the upstream DecisionEngine output.

    Required attributes:

      ``risk.band``                     — :class:`RiskBand`
      ``risk.composite``                — float (informational)
      ``risk.config_version``           — str (echoed into PolicyResult)
      ``input_snapshot.match.matched``  — bool
      ``input_snapshot.match.similarity`` — float
      ``input_snapshot.config_version`` — str (echoed into PolicyResult)
    """

    risk: _RiskAccess
    input_snapshot: _InputSnapshotAccess


# ---------------------------------------------------------------------
# Pure helpers (deterministic, no I/O)
# ---------------------------------------------------------------------


def _normalize_signal_source(raw: str) -> str:
    """Strip whitespace, lowercase. Unknown values handled by caller."""
    if raw is None:
        return ""
    return str(raw).strip().lower()


def _normalize_dispute_outcomes(raw: list[str]) -> list[str]:
    """Lowercase, filter to known outcomes, take last MAX_DISPUTE_HISTORY.
    Insertion order preserved (chronological — most recent last)."""
    if not raw:
        return []
    cleaned: list[str] = []
    for entry in raw:
        if entry is None:
            continue
        normalized = str(entry).strip().lower()
        if normalized in VALID_DISPUTE_OUTCOMES:
            cleaned.append(normalized)
    if len(cleaned) > MAX_DISPUTE_HISTORY:
        cleaned = cleaned[-MAX_DISPUTE_HISTORY:]
    return cleaned


def _derive_evidence_strength(
    *,
    match_found: bool,
    signal_source: str,
    similarity: float,
    has_multiple_matches: bool,
    distinct_owner_count: int,
) -> EvidenceStrength:
    """Two-step derivation per v5 §3.

    Step 1 — base strength (no multi-match):
      NONE     match_found is False
      STRONG   signal_source == 'fusion' AND similarity >= 0.70
      MODERATE (fusion AND sim >= 0.40) OR (non-fusion AND sim >= 0.70)
      WEAK     match_found is True and none of the above

    Step 2 — multi-match adjustment (only if match_found):
      distinct_owner_count >= 2 AND similarity >= 0.70  → STRONG
      base == WEAK                                       → MODERATE
      otherwise                                          → no change
    """
    if not match_found:
        return EvidenceStrength.NONE

    sim = similarity if similarity is not None else 0.0
    is_fusion = signal_source == "fusion"

    if is_fusion and sim >= FUSION_STRONG_SIMILARITY:
        base = EvidenceStrength.STRONG
    elif (is_fusion and sim >= FUSION_MODERATE_SIMILARITY) or (
        not is_fusion and sim >= SINGLE_MODERATE_SIMILARITY
    ):
        base = EvidenceStrength.MODERATE
    else:
        base = EvidenceStrength.WEAK

    if not has_multiple_matches:
        return base

    if distinct_owner_count >= 2 and sim >= MULTI_MATCH_DISTINCT_SIMILARITY:
        return EvidenceStrength.STRONG

    if base == EvidenceStrength.WEAK:
        return EvidenceStrength.MODERATE

    # MODERATE stays MODERATE; STRONG stays STRONG (v5 stabilization).
    return base


def _severity(action: PolicyAction) -> int:
    return _SEVERITY[action]


def _min_action(a: PolicyAction, b: PolicyAction) -> PolicyAction:
    """Lower severity wins (used as a clamp)."""
    return a if _SEVERITY[a] <= _SEVERITY[b] else b


@dataclass(frozen=True)
class FeasibleBounds:
    """Action-space upper bound produced by the BOUND phase.

    Carries the most-severe action permitted by the safety rules that
    fired so far. ``upper`` is initialised to :data:`GLOBAL_MAX_UPGRADE`
    and only ever narrows — :meth:`tighten_upper` returns a new
    instance with ``upper = min(self.upper, target)`` so the bounds
    object remains immutable per evaluation. REFINE-phase upgrade
    rules (R1, R4) clamp their desired target through :meth:`clamp`.

    A static lower bound is unnecessary because REFINE-phase upgrades
    never drop below the proposed action and downgrades have final
    authority irrespective of the bounds.
    """

    upper: PolicyAction

    def tighten_upper(self, target: PolicyAction) -> "FeasibleBounds":
        """Return new bounds with ``upper`` narrowed to ``min(upper, target)``.

        Idempotent: a tightening that does not lower ``upper`` returns
        bounds with the same ``upper`` value (a fresh frozen instance).
        """
        return FeasibleBounds(upper=_min_action(self.upper, target))

    def clamp(self, action: PolicyAction) -> PolicyAction:
        """Return ``action`` clamped to ``upper`` (severity-wise minimum)."""
        return _min_action(action, self.upper)


def _tier_at_least(tier: ConfidenceTier, floor: ConfidenceTier) -> bool:
    return _TIER_ORDER[tier] >= _TIER_ORDER[floor]


def _compute_evaluation_hash(
    *,
    triggered_rules: list[str],
    final_action: PolicyAction,
    policy_version: str,
    rules_checked_count: int,
    risk_band: RiskBand,
    confidence_tier: ConfidenceTier,
    confidence_composite: float,
    signal_source: str,
    similarity: float,
    is_first_offense: bool,
    has_prior_disputes: bool,
    distinct_matched_owner_count: int,
) -> str:
    """SHA-256 over 12 deterministically-serialized inputs (v5 §2).

    Floats rounded to 4 decimal places to remove platform float drift.
    Triggered rules sorted alphabetically. Output is the first 16 hex
    characters of the digest.
    """
    parts = [
        str(sorted(triggered_rules)),
        final_action.value,
        policy_version,
        str(rules_checked_count),
        risk_band.value,
        confidence_tier.value,
        f"{confidence_composite:.4f}",
        signal_source,
        f"{similarity:.4f}",
        str(is_first_offense),
        str(has_prior_disputes),
        str(distinct_matched_owner_count),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _select_primary_reason(
    candidates: dict[str, str],
    *,
    risk_band: RiskBand,
    confidence_tier: ConfidenceTier,
    final_action: PolicyAction,
) -> str:
    """Highest-priority firing rule wins. Falls back to base-matrix
    description when no rule fired."""
    if not candidates:
        return (
            f"Base matrix: {risk_band.value} risk × "
            f"{confidence_tier.value} confidence → {final_action.value}"
        )
    winner = max(
        candidates.items(), key=lambda item: _RULE_PRIORITY.get(item[0], 0)
    )
    return winner[1]


def _has_gray_zone_reason(
    reasons: Iterable[ConfidenceReasonCode],
) -> bool:
    return ConfidenceReasonCode.GRAY_ZONE in tuple(reasons)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def evaluate_policy(
    decision: DecisionOutput,
    confidence: ConfidenceBreakdown,
    context: PolicyContext,
) -> PolicyResult:
    """Pure, deterministic policy evaluation — strict 6-phase pipeline.

    Inputs:
      decision    — DecisionEngine output exposing risk + input_snapshot.
      confidence  — ConfidenceEngine breakdown (tier, composite, reasons).
      context     — operational + evidence signals (PolicyContext).

    Guarantees:
      * Same inputs → same output (no randomness, no I/O).
      * ``rules_checked_count == EXPECTED_RULE_COUNT`` always (asserted).
      * ``triggered_rules`` lists only rules whose effect was applied.
      * ``evaluation_hash`` digests 12 input fields for replay verification.
      * Terminal invariant: ``confidence_tier != HIGH`` ⇒ never TAKEDOWN.
    """
    # ---- Read upstream values (v2 §1: do not duplicate in PolicyContext)
    risk_band: RiskBand = decision.risk.band
    decision_config_version: str = decision.risk.config_version
    confidence_config_version: str = decision.input_snapshot.config_version
    match_found: bool = bool(decision.input_snapshot.match.matched)
    similarity_raw = decision.input_snapshot.match.similarity
    similarity: float = float(similarity_raw) if similarity_raw is not None else 0.0

    confidence_tier: ConfidenceTier = confidence.tier
    confidence_composite: float = float(confidence.composite)

    # ---- Normalize entry-point inputs (v5 §4, v3 §5)
    signal_source = _normalize_signal_source(context.signal_source)
    if signal_source not in VALID_SIGNAL_SOURCES:
        signal_source = DEFAULT_SIGNAL_SOURCE
    normalized_disputes = _normalize_dispute_outcomes(
        list(context.prior_dispute_outcomes)
    )

    # ---- Derive evidence (used by R2 and R3)
    evidence_strength = _derive_evidence_strength(
        match_found=match_found,
        signal_source=signal_source,
        similarity=similarity,
        has_multiple_matches=context.has_multiple_matches,
        distinct_owner_count=context.distinct_matched_owner_count,
    )

    # ---- Phase 0: base matrix lookup
    base_action: PolicyAction = _BASE_MATRIX[(risk_band, confidence_tier)]
    current_action: PolicyAction = base_action

    triggered_rules: list[str] = []
    upgrades_applied: list[str] = []
    downgrades_applied: list[str] = []
    primary_reason_candidates: dict[str, str] = {}

    # BOUND-phase output (PBRA model — see module docstring).
    # Initialised to the static global ceiling and narrowed each time
    # a safety rule fires. Consumed by REFINE-phase upgrade rules.
    bounds = FeasibleBounds(upper=GLOBAL_MAX_UPGRADE)

    rules_checked_count = 0

    # =================================================================
    # Phase 1 — HARD OVERRIDE (S1)
    # =================================================================
    rules_checked_count += 1
    if not match_found:
        triggered_rules.append(S1_NO_MATCH)
        downgrades_applied.append(S1_NO_MATCH)
        primary_reason_candidates[S1_NO_MATCH] = (
            "No matching protected asset. "
            "Enforcement without match is forbidden."
        )
        current_action = PolicyAction.ALLOW
        after_safety = current_action

        # Short-circuit phases 2..6: count remaining rules as checked.
        rules_checked_count = EXPECTED_RULE_COUNT
        final_action = current_action

        return _build_result(
            base_action=base_action,
            after_safety=after_safety,
            final_action=final_action,
            triggered_rules=triggered_rules,
            upgrades_applied=upgrades_applied,
            downgrades_applied=downgrades_applied,
            primary_reason_candidates=primary_reason_candidates,
            risk_band=risk_band,
            confidence_tier=confidence_tier,
            confidence_composite=confidence_composite,
            decision_config_version=decision_config_version,
            confidence_config_version=confidence_config_version,
            rules_checked_count=rules_checked_count,
            signal_source=signal_source,
            similarity=similarity,
            is_first_offense=context.is_first_offense,
            has_prior_disputes=context.has_prior_disputes,
            distinct_matched_owner_count=context.distinct_matched_owner_count,
        )

    # =================================================================
    # Phase 2 — BOUND upper-cap (S2, S3 — matrix-evolution guards)
    # Both rules are unreachable from the current base matrix and
    # exist as guards against future matrix changes that would
    # propose constraint-violating actions. See module docstring
    # ("Matrix-evolution guards") and
    # :func:`_validate_matrix_constraint_consistency` (startup check).
    # =================================================================
    # S2: TAKEDOWN requires HIGH confidence (matrix-evolution guard).
    rules_checked_count += 1
    if (
        current_action == PolicyAction.TAKEDOWN
        and confidence_tier != ConfidenceTier.HIGH
    ):
        triggered_rules.append(S2_TAKEDOWN_CONFIDENCE_GATE)
        downgrades_applied.append(S2_TAKEDOWN_CONFIDENCE_GATE)
        primary_reason_candidates[S2_TAKEDOWN_CONFIDENCE_GATE] = (
            "TAKEDOWN requires HIGH confidence. Downgraded to RESTRICT."
        )
        current_action = PolicyAction.RESTRICT
        bounds = bounds.tighten_upper(PolicyAction.RESTRICT)

    # S3: RESTRICT requires at least MEDIUM confidence (matrix-evolution guard).
    rules_checked_count += 1
    if (
        current_action == PolicyAction.RESTRICT
        and confidence_tier == ConfidenceTier.LOW
    ):
        triggered_rules.append(S3_RESTRICT_CONFIDENCE_GATE)
        downgrades_applied.append(S3_RESTRICT_CONFIDENCE_GATE)
        primary_reason_candidates[S3_RESTRICT_CONFIDENCE_GATE] = (
            "RESTRICT requires at least MEDIUM confidence. "
            "Downgraded to REVIEW."
        )
        current_action = PolicyAction.REVIEW
        bounds = bounds.tighten_upper(PolicyAction.REVIEW)

    # =================================================================
    # Phase 3 — LOWER CAP (S4) — cap at FLAG when confidence < 0.30
    # =================================================================
    rules_checked_count += 1
    if (
        confidence_composite < CONFIDENCE_FLOOR
        and _severity(current_action) > _severity(PolicyAction.FLAG)
    ):
        triggered_rules.append(S4_CONFIDENCE_CEILING)
        downgrades_applied.append(S4_CONFIDENCE_CEILING)
        primary_reason_candidates[S4_CONFIDENCE_CEILING] = (
            f"Confidence {confidence_composite:.2f} below 0.30. "
            f"Maximum action capped at FLAG."
        )
        current_action = PolicyAction.FLAG
        bounds = bounds.tighten_upper(PolicyAction.FLAG)

    # =================================================================
    # Phase 4 — TYPE CONSTRAINTS (S5)
    # =================================================================
    rules_checked_count += 1
    if (
        context.content_type not in SUPPORTED_ENFORCEMENT_TYPES
        and _severity(current_action) > _severity(PolicyAction.REVIEW)
    ):
        triggered_rules.append(S5_CONTENT_TYPE_GATE)
        downgrades_applied.append(S5_CONTENT_TYPE_GATE)
        primary_reason_candidates[S5_CONTENT_TYPE_GATE] = (
            f"Content type '{context.content_type}' not approved "
            f"for automated enforcement."
        )
        current_action = PolicyAction.REVIEW
        bounds = bounds.tighten_upper(PolicyAction.REVIEW)

    after_safety: PolicyAction = current_action

    # =================================================================
    # Phase 5 — REFINE upgrades (R1, R4)
    # Ceiling = bounds.upper clamped against GLOBAL_MAX_UPGRADE. The
    # clamp is structurally redundant (bounds.upper is initialised to
    # GLOBAL_MAX_UPGRADE and only narrows) but kept explicit per the
    # v4 §1 split between dynamic (safety) and static (global) limits.
    # =================================================================
    effective_ceiling = bounds.clamp(GLOBAL_MAX_UPGRADE)

    # R1 — Gray zone conditional escalation.
    rules_checked_count += 1
    gray_zone_reason = _has_gray_zone_reason(confidence.triggered_conditions)
    if (
        gray_zone_reason
        and confidence_composite < GRAY_ZONE_CONFIDENCE_UPPER
        and confidence_composite >= GRAY_ZONE_CONFIDENCE_LOWER  # v3 §2
        and match_found
        and _severity(current_action) < _severity(PolicyAction.REVIEW)
    ):
        desired = PolicyAction.REVIEW
        effective = _min_action(desired, effective_ceiling)
        if _severity(current_action) < _severity(effective):
            triggered_rules.append(R1_GRAY_ZONE)
            upgrades_applied.append(R1_GRAY_ZONE)
            primary_reason_candidates[R1_GRAY_ZONE] = (
                f"Similarity in adversarial gray zone with confidence "
                f"{confidence_composite:.2f} < 0.75."
            )
            current_action = effective

    # R4 — Verified owner upgrade.
    rules_checked_count += 1
    if (
        context.trust_owner_tier == "verified"
        and similarity >= VERIFIED_OWNER_SIMILARITY
        and _tier_at_least(confidence_tier, ConfidenceTier.MEDIUM)
        and current_action == PolicyAction.REVIEW
    ):
        desired = PolicyAction.RESTRICT
        effective = _min_action(desired, effective_ceiling)
        if _severity(current_action) < _severity(effective):
            triggered_rules.append(R4_VERIFIED_OWNER)
            upgrades_applied.append(R4_VERIFIED_OWNER)
            primary_reason_candidates[R4_VERIFIED_OWNER] = (
                "Verified rights-holder with high similarity. "
                "Upgrading to RESTRICT."
            )
            current_action = effective

    phase5_action: PolicyAction = current_action

    # =================================================================
    # Phase 6 — RISK DOWNGRADES (R2, R3, R5) — final authority
    # =================================================================
    # R2 — First-offense downgrade. Skipped when evidence is STRONG (v4 §5).
    rules_checked_count += 1
    if (
        context.is_first_offense
        and _severity(current_action) >= _severity(PolicyAction.RESTRICT)
        and evidence_strength != EvidenceStrength.STRONG
    ):
        triggered_rules.append(R2_FIRST_OFFENSE)
        downgrades_applied.append(R2_FIRST_OFFENSE)
        primary_reason_candidates[R2_FIRST_OFFENSE] = (
            "First offense for this uploader. "
            "Downgrading to human review."
        )
        current_action = PolicyAction.REVIEW

    # R3 — Evidence gate for TAKEDOWN.
    rules_checked_count += 1
    if (
        current_action == PolicyAction.TAKEDOWN
        and evidence_strength != EvidenceStrength.STRONG
    ):
        triggered_rules.append(R3_EVIDENCE_GATE)
        downgrades_applied.append(R3_EVIDENCE_GATE)
        primary_reason_candidates[R3_EVIDENCE_GATE] = (
            f"TAKEDOWN requires STRONG evidence. "
            f"Current: {evidence_strength.value}."
        )
        current_action = PolicyAction.RESTRICT

    # R5 — Dispute history caution.
    rules_checked_count += 1
    if (
        context.has_prior_disputes
        and "overturned" in normalized_disputes
        and _severity(current_action) >= _severity(PolicyAction.RESTRICT)
    ):
        triggered_rules.append(R5_DISPUTE_CAUTION)
        downgrades_applied.append(R5_DISPUTE_CAUTION)
        primary_reason_candidates[R5_DISPUTE_CAUTION] = (
            "Uploader has overturned disputes. Applying caution. "
            "Human review required."
        )
        current_action = PolicyAction.REVIEW

    phase6_action: PolicyAction = current_action

    # Phase 6 = final authority over Phase 5 (downgrade-only ⇒ min wins).
    final_action: PolicyAction = _min_action(phase5_action, phase6_action)

    # =================================================================
    # Terminal invariant guard (v5 §1)
    # NEVER fires when phase logic is correct. ERROR-log on activation.
    # =================================================================
    if (
        confidence_tier != ConfidenceTier.HIGH
        and final_action == PolicyAction.TAKEDOWN
    ):
        _logger.error(
            "PolicyEngine invariant violated: TAKEDOWN at confidence_tier=%s. "
            "Forcing RESTRICT. risk_band=%s, triggered_rules=%s",
            confidence_tier.value,
            risk_band.value,
            triggered_rules,
        )
        final_action = PolicyAction.RESTRICT
        triggered_rules.append(INVARIANT_TAKEDOWN_GUARD)
        downgrades_applied.append(INVARIANT_TAKEDOWN_GUARD)
        primary_reason_candidates[INVARIANT_TAKEDOWN_GUARD] = (
            "Invariant guard: TAKEDOWN forbidden below HIGH confidence. "
            "Forced to RESTRICT."
        )

    # ---- Completeness assertion (v3 §3)
    if rules_checked_count != EXPECTED_RULE_COUNT:  # pragma: no cover
        raise AssertionError(
            f"PolicyEngine programmer error: rules_checked_count="
            f"{rules_checked_count}, expected {EXPECTED_RULE_COUNT}."
        )

    return _build_result(
        base_action=base_action,
        after_safety=after_safety,
        final_action=final_action,
        triggered_rules=triggered_rules,
        upgrades_applied=upgrades_applied,
        downgrades_applied=downgrades_applied,
        primary_reason_candidates=primary_reason_candidates,
        risk_band=risk_band,
        confidence_tier=confidence_tier,
        confidence_composite=confidence_composite,
        decision_config_version=decision_config_version,
        confidence_config_version=confidence_config_version,
        rules_checked_count=rules_checked_count,
        signal_source=signal_source,
        similarity=similarity,
        is_first_offense=context.is_first_offense,
        has_prior_disputes=context.has_prior_disputes,
        distinct_matched_owner_count=context.distinct_matched_owner_count,
    )


# ---------------------------------------------------------------------
# Result builder (kept private; isolates PolicyResult assembly)
# ---------------------------------------------------------------------


def _build_result(
    *,
    base_action: PolicyAction,
    after_safety: PolicyAction,
    final_action: PolicyAction,
    triggered_rules: list[str],
    upgrades_applied: list[str],
    downgrades_applied: list[str],
    primary_reason_candidates: dict[str, str],
    risk_band: RiskBand,
    confidence_tier: ConfidenceTier,
    confidence_composite: float,
    decision_config_version: str,
    confidence_config_version: str,
    rules_checked_count: int,
    signal_source: str,
    similarity: float,
    is_first_offense: bool,
    has_prior_disputes: bool,
    distinct_matched_owner_count: int,
) -> PolicyResult:
    primary_reason = _select_primary_reason(
        primary_reason_candidates,
        risk_band=risk_band,
        confidence_tier=confidence_tier,
        final_action=final_action,
    )
    evaluation_hash = _compute_evaluation_hash(
        triggered_rules=triggered_rules,
        final_action=final_action,
        policy_version=POLICY_VERSION,
        rules_checked_count=rules_checked_count,
        risk_band=risk_band,
        confidence_tier=confidence_tier,
        confidence_composite=confidence_composite,
        signal_source=signal_source,
        similarity=similarity,
        is_first_offense=is_first_offense,
        has_prior_disputes=has_prior_disputes,
        distinct_matched_owner_count=distinct_matched_owner_count,
    )
    return PolicyResult(
        final_action=final_action,
        action_trace=ActionTrace(
            base_action=base_action,
            after_safety=after_safety,
            after_risk_control=final_action,
            upgrades_applied=list(upgrades_applied),
            downgrades_applied=list(downgrades_applied),
        ),
        triggered_rules=list(triggered_rules),
        primary_reason=primary_reason,
        risk_band=risk_band,
        confidence_tier=confidence_tier,
        confidence_composite=confidence_composite,
        policy_version=POLICY_VERSION,
        decision_config_version=decision_config_version,
        confidence_config_version=confidence_config_version,
        rules_checked_count=rules_checked_count,
        evaluation_hash=evaluation_hash,
    )


# ---------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------


def _validate_matrix_constraint_consistency() -> None:
    """Verify that ``_BASE_MATRIX`` is constraint-safe at import time.

    Three checks, all of which abort module import on failure:

      1. **Completeness** — every (RiskBand, ConfidenceTier) pair is
         defined and no unexpected keys are present. Missing cells
         would surface only at request time as :class:`KeyError`.

      2. **S2 invariant** — no cell proposes ``TAKEDOWN`` at a
         ``confidence_tier`` other than ``HIGH``. If violated, S2
         silently becomes an active downgrade rather than the
         documented matrix-evolution guard, changing the engine's
         behavioural profile.

      3. **S3 invariant** — no cell proposes ``RESTRICT`` at
         ``confidence_tier == LOW``. Same reasoning as (2) for S3.

    A fourth belt-and-braces check ensures every proposed action is
    a recognised :class:`PolicyAction` (covered by enum typing but
    kept for future-proofing if the matrix is loaded from a config).

    Failures are :class:`AssertionError` so they halt application
    startup with a stack trace rather than failing in production.
    """
    expected_cells = {(b, t) for b in RiskBand for t in ConfidenceTier}
    actual_cells = set(_BASE_MATRIX.keys())

    missing = expected_cells - actual_cells
    if missing:
        raise AssertionError(
            f"_BASE_MATRIX is missing cells: "
            f"{sorted((b.value, t.value) for b, t in missing)}"
        )

    unexpected = actual_cells - expected_cells
    if unexpected:
        raise AssertionError(
            f"_BASE_MATRIX has unexpected cells: {sorted(unexpected)}"
        )

    for (band, tier), action in _BASE_MATRIX.items():
        if action not in _SEVERITY:
            raise AssertionError(
                f"_BASE_MATRIX cell ({band.value}, {tier.value}) "
                f"produces unknown PolicyAction {action!r}."
            )
        if (
            action == PolicyAction.TAKEDOWN
            and tier != ConfidenceTier.HIGH
        ):
            raise AssertionError(
                f"_BASE_MATRIX cell ({band.value}, {tier.value}) "
                f"proposes TAKEDOWN below HIGH confidence — violates "
                f"S2 invariant. S2 is documented as a matrix-evolution "
                f"guard; reclassify or fix the cell before enabling."
            )
        if (
            action == PolicyAction.RESTRICT
            and tier == ConfidenceTier.LOW
        ):
            raise AssertionError(
                f"_BASE_MATRIX cell ({band.value}, {tier.value}) "
                f"proposes RESTRICT at LOW confidence — violates S3 "
                f"invariant. S3 is documented as a matrix-evolution "
                f"guard; reclassify or fix the cell before enabling."
            )


# Run at import. Fails fast so production never sees an inconsistent
# matrix, and code reviews catch drift between matrix and safety rules.
_validate_matrix_constraint_consistency()
