"""Policy-engine data contracts.

Pydantic models and enums consumed by ``app.engines.policy_engine``.
This module is pure schema — no logic, no helpers, no engine-layer
imports. Source of truth: ``.claude/memory/policy_engine_spec_v1.md``
through ``policy_engine_spec_v5.md`` (final merged).

Five public types:

* ``PolicyAction``       — five-level enforcement severity ladder.
* ``EvidenceStrength``   — derived corroboration tier for matches.
* ``PolicyContext``      — operational + evidence signals not present
                           in DecisionOutput / ConfidenceBreakdown.
* ``ActionTrace``        — base / after-safety / after-risk-control
                           plus per-phase rule_id lists, for audit.
* ``PolicyResult``       — full engine output: final_action, audit
                           trail, version triad, integrity digest.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.confidence_models import ConfidenceTier
from app.models.decision_models import RiskBand


class PolicyAction(str, Enum):
    """Five ordered severity levels (ALLOW < FLAG < REVIEW < RESTRICT < TAKEDOWN)."""

    ALLOW = "ALLOW"
    FLAG = "FLAG"
    REVIEW = "REVIEW"
    RESTRICT = "RESTRICT"
    TAKEDOWN = "TAKEDOWN"


class EvidenceStrength(str, Enum):
    """Corroboration tier derived from match presence, signal source,
    similarity, multi-match, and distinct-owner count."""

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class PolicyContext(BaseModel):
    """Operational and evidence signals NOT available in DecisionOutput
    or ConfidenceBreakdown. Assembled by the pipeline worker.
    """

    # ── Operational context ──
    is_first_offense: bool
    has_prior_disputes: bool
    prior_dispute_outcomes: list[str] = Field(default_factory=list)
    content_type: str
    trust_owner_tier: str
    trust_uploader_tier: str
    uploader_is_default: bool
    owner_is_default: bool
    recent_violations_count: int = 0

    # ── Evidence context ──
    signal_source: str
    has_multiple_matches: bool = False
    matched_asset_owner: Optional[str] = None
    distinct_matched_owner_count: int = 1

    # ── Lineage context (H-1 / H-2) ──
    # Carries the upstream embedding model version into the policy
    # evaluation_hash so that a model upgrade is reflected in the hash
    # even when the (risk_band, confidence_tier, signals...) tuple is
    # otherwise unchanged. Optional + default None preserves smoke-test
    # compatibility for callers that don't supply it.
    embedding_model_version: Optional[str] = None


class ActionTrace(BaseModel):
    """Per-phase mutation chain for the action."""

    base_action: PolicyAction
    after_safety: PolicyAction
    after_risk_control: PolicyAction
    upgrades_applied: list[str] = Field(default_factory=list)
    downgrades_applied: list[str] = Field(default_factory=list)


class PolicyResult(BaseModel):
    """Complete output of the PolicyEngine."""

    # Final decision
    final_action: PolicyAction

    # Audit trail
    action_trace: ActionTrace
    triggered_rules: list[str] = Field(default_factory=list)

    # Human-readable summary
    primary_reason: str

    # Context echo (for audit reproducibility)
    risk_band: RiskBand
    confidence_tier: ConfidenceTier
    confidence_composite: float

    # Version triad — full replay parameters
    policy_version: str
    decision_config_version: str
    confidence_config_version: str

    # Lineage echo (H-1 / H-2) — upstream model version contributing
    # to evaluation_hash. Optional for compatibility with callers that
    # do not provide it.
    embedding_model_version: Optional[str] = None

    # Integrity / completeness
    rules_checked_count: int
    evaluation_hash: str
