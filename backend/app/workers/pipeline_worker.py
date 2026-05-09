"""Sequential pipeline worker — idempotent under retries and redelivery.

Consumes from the RQ "pipeline" queue. Each invocation of run_pipeline
acquires a Redis lock at `lock:job:{job_id}` (SET NX EX 300) so concurrent
workers cannot race on the same job, then re-validates that the job is
still QUEUED before transitioning to PROCESSING. Together these make
duplicate deliveries, RQ retries, and crash-resume safe: a second invocation
either fails to take the lock or sees a non-QUEUED status and exits.

The lock is released via a token-checked Lua script so a worker can never
delete the lock of a successor that acquired the key after the original
TTL expired. Any exception during pipeline execution flips the job to
FAILED before the lock is released, so jobs are never stranded in
PROCESSING by a clean exception path.

Engine-triple wiring (post-runtime-convergence)
-----------------------------------------------

EVALUATION phase invokes ``decision_engine.compute_risk`` and
``confidence_engine.compute_confidence`` in sequence within a single
``evaluation`` stage. The DECISION phase invokes
``policy_engine.evaluate_policy`` over the assembled DecisionOutput +
ConfidenceBreakdown + PolicyContext (per docs/specs/job_processing.md
§5.2). The five-level ``PolicyAction`` is mapped to the legacy
3-action ``ALLOW/FLAG/BLOCK`` vocabulary for backward-compatible
``ENFORCED`` payloads and to the terminal ``JobStatus`` per
docs/specs/job_processing.md §3.3 / §5.5.

Backward compatibility during transition
----------------------------------------

The worker still emits ``SCORED`` (with band derived from the new
``RiskScore.band``) and ``ENFORCED`` (with the legacy ``action`` field
plus additive ``policy_action`` / ``evaluation_hash`` / ``policy_version``
fields) so existing consumers continue to work. The canonical
``RISK_SCORED`` / ``CONFIDENCE_COMPUTED`` / ``POLICY_DECIDED`` events
are emitted alongside.

Legacy ``scoring_engine`` and ``enforcement_engine`` are no longer
invoked by the worker; the modules remain in the repository as
deprecated transition-window code (per docs/specs/job_processing.md
§5.5 phase C).

At each stage transition the worker publishes a canonical pipeline event
(`.claude/rules/eventing.md`) via `publish_event`, alongside the lifecycle
audit emitted by the `stage_event` context manager. Domain events carry
strict typed payloads; lifecycle events carry latency. Both share the same
per-job audit log, so downstream readers reconstruct the timeline with a
single sorted-set scan.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

from app.core.event_store import (
    ConfidenceComputedPayload,
    EmbeddingReadyPayload,
    EnforcedPayload,
    FingerprintReadyPayload,
    JobCompletedPayload,
    JobFailedPayload,
    MatchFoundPayload,
    MatchNotFoundPayload,
    PipelineEventType,
    PolicyDecidedPayload,
    RiskScoredPayload,
    ScoredPayload,
    publish_event,
    stage_event,
)
from app.core.evidence_store import record_evidence
from app.core.job_store import JobStatus, job_store
from app.core.queue import redis_conn
from app.engines import (
    confidence_engine,
    decision_engine,
    embedding_engine,
    fingerprint_engine,
    matching_engine,
    policy_engine,
)
from app.models.confidence_models import (
    ConfidenceConfig,
    ConfidenceInput,
    TrustState,
)
from app.models.decision_models import (
    DecisionInput,
    DecisionOutput,
    InputSnapshot,
    MatchInputSnapshot,
    MatchSignal,
    RiskScore,
    ScoreSignal,
    ThresholdConfig,
    TrustSignal,
)
from app.models.policy_models import PolicyAction, PolicyContext, PolicyResult


# ---------------------------------------------------------------------------
# Configuration (engine triple)
# ---------------------------------------------------------------------------

# Module-level config singletons. ThresholdConfig and ConfidenceConfig both
# validate their weights at construction time (per
# docs/specs/decision_engine.md §4.2 and docs/specs/confidence_engine.md §4.2)
# so any drift surfaces at import, not at request time.
_THRESHOLD_CONFIG = ThresholdConfig()
_CONFIDENCE_CONFIG = ConfidenceConfig()

# Version strings carried into engine_lineage. Bumped in lockstep with the
# canonical specs (per docs/specs/decision_engine.md §16 and
# docs/specs/confidence_engine.md §16).
_DECISION_CONFIG_VERSION = "v1.0"
_CONFIDENCE_CONFIG_VERSION = "v3"


# ---------------------------------------------------------------------------
# Lock semantics (per docs/specs/job_processing.md §8)
# ---------------------------------------------------------------------------

_LOCK_TTL_SECONDS = 300

# Compare-and-delete: only release the lock if the token still matches the
# one we wrote. Prevents releasing a successor's lock if our TTL expired
# mid-execution and another worker has since re-acquired the key.
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def _lock_key(job_id: str) -> str:
    return f"lock:job:{job_id}"


def _acquire_lock(job_id: str) -> Optional[str]:
    token = uuid.uuid4().hex
    acquired = redis_conn.set(
        _lock_key(job_id),
        token,
        nx=True,
        ex=_LOCK_TTL_SECONDS,
    )
    return token if acquired else None


def _release_lock(job_id: str, token: str) -> None:
    try:
        redis_conn.eval(_RELEASE_SCRIPT, 1, _lock_key(job_id), token)
    except Exception:
        # Lock will auto-expire via TTL; never let release errors mask
        # a real pipeline exception or block worker shutdown.
        pass


# ---------------------------------------------------------------------------
# Action mapping (5-action PolicyAction → legacy 3-action / terminal status)
# ---------------------------------------------------------------------------

# Maps the 5-action canonical PolicyAction (per
# docs/specs/policy_engine.md §2.1) to the legacy 3-action vocabulary
# carried in the ENFORCED event's ``action`` field for backward-compatible
# consumers. REVIEW collapses to FLAG (human review = soft flag). RESTRICT
# and TAKEDOWN both collapse to BLOCK (automated enforcement applied).
_POLICY_ACTION_TO_LEGACY: Dict[PolicyAction, str] = {
    PolicyAction.ALLOW: "ALLOW",
    PolicyAction.FLAG: "FLAG",
    PolicyAction.REVIEW: "FLAG",
    PolicyAction.RESTRICT: "BLOCK",
    PolicyAction.TAKEDOWN: "BLOCK",
}

# Maps PolicyAction to the terminal JobStatus per
# docs/specs/job_processing.md §3.3: FLAGGED reserved for actions
# requiring human follow-up (FLAG, REVIEW); COMPLETED for fully-resolved
# actions (ALLOW + automated enforcement).
_POLICY_ACTION_TO_TERMINAL: Dict[PolicyAction, JobStatus] = {
    PolicyAction.ALLOW: JobStatus.COMPLETED,
    PolicyAction.FLAG: JobStatus.FLAGGED,
    PolicyAction.REVIEW: JobStatus.FLAGGED,
    PolicyAction.RESTRICT: JobStatus.COMPLETED,
    PolicyAction.TAKEDOWN: JobStatus.COMPLETED,
}


# ---------------------------------------------------------------------------
# Trust + signal source mapping
# ---------------------------------------------------------------------------

# Maps the matching engine's ``trust_level`` strings to numeric trust
# scores consumed by the DecisionEngine. Conservative mapping: unknown
# defaults near zero so missing-registry signals don't inflate risk.
_TRUST_LEVEL_TO_SCORE: Dict[str, float] = {
    "verified": 0.95,
    "premium": 0.75,
    "basic": 0.50,
    "unknown": 0.20,
}

# Vocabulary divergence between engines is documented in
# docs/specs/decision_engine.md §3.5 / D-DE-3 — DecisionEngine's quality
# table uses ``"fingerprint+embedding"`` while ConfidenceEngine and
# PolicyEngine use ``"fusion"``. The current matching pipeline produces
# both fingerprint (via pHash) and embedding (cosine) signals, so we emit
# the fusion-equivalent value to each consumer in the form it expects.
_DECISION_SIGNAL_SOURCE = "fingerprint+embedding"
_CONFIDENCE_SIGNAL_SOURCE = "fusion"
_POLICY_SIGNAL_SOURCE = "fusion"


def _trust_score(trust_level: Optional[str]) -> float:
    if trust_level is None:
        return _TRUST_LEVEL_TO_SCORE["unknown"]
    return _TRUST_LEVEL_TO_SCORE.get(trust_level, _TRUST_LEVEL_TO_SCORE["unknown"])


def _trust_signal(trust_level: Optional[str]) -> TrustSignal:
    return TrustSignal(trust_score=_trust_score(trust_level))


def _trust_state(trust_level: Optional[str]) -> TrustState:
    """Map a trust-registry tier to the ConfidenceEngine's TrustState.

    A missing or 'unknown' tier sets ``is_default=True`` so the
    ConfidenceEngine treats the trust signal as registry-default (per
    docs/specs/confidence_engine.md §4.3); the trust_score field is
    consulted only when ``is_default == False``.
    """
    if trust_level is None or trust_level == "unknown":
        return TrustState(trust_score=0.0, is_default=True)
    return TrustState(
        trust_score=_TRUST_LEVEL_TO_SCORE.get(trust_level, 0.0),
        is_default=False,
    )


# ---------------------------------------------------------------------------
# Engine input assembly
# ---------------------------------------------------------------------------


def _build_decision_input(
    *,
    similarity: float,
    matched_asset: Optional[Dict[str, Any]],
    uploader_trust_level: Optional[str],
) -> DecisionInput:
    """Assemble the DecisionEngine input from upstream stage outputs.

    Trust signals are derived from ``matched_asset.trust_level`` (owner)
    and the request's uploader trust hint (today defaulting to 'basic';
    the trust-reader spec is planned per
    docs/specs/decision_engine.md D-DE-4). Observation count + timestamps
    default to zero today — the observation store integration is a
    future expansion.
    """
    owner_trust_level = (
        matched_asset.get("trust_level") if matched_asset else None
    )
    return DecisionInput(
        match=MatchSignal(similarity=similarity),
        trust_owner=_trust_signal(owner_trust_level),
        trust_uploader=_trust_signal(uploader_trust_level),
        score=ScoreSignal(signal_source=_DECISION_SIGNAL_SOURCE),
        observation_count=0,
        config_version=_DECISION_CONFIG_VERSION,
        observation_timestamps=(),
    )


def _build_confidence_input(
    *,
    match_found: bool,
    similarity: float,
    matched_asset: Optional[Dict[str, Any]],
    uploader_trust_level: Optional[str],
) -> ConfidenceInput:
    """Assemble the ConfidenceEngine input.

    Trust states use ``TrustState(is_default=...)`` semantics so the
    confidence engine derives S2/S3 (signal presence) and U2/U3
    (uncertainty) correctly. observation_count defaults to zero today.
    """
    owner_trust_level = (
        matched_asset.get("trust_level") if matched_asset else None
    )
    return ConfidenceInput(
        match_found=match_found,
        similarity=similarity,
        trust_owner=_trust_state(owner_trust_level),
        trust_uploader=_trust_state(uploader_trust_level),
        observation_count=0,
        signal_source=_CONFIDENCE_SIGNAL_SOURCE,
        config_version=_CONFIDENCE_CONFIG_VERSION,
    )


def _build_decision_output(
    *,
    risk: RiskScore,
    match_found: bool,
    similarity: float,
) -> DecisionOutput:
    """Wrap RiskScore into the DecisionOutput envelope consumed by
    PolicyEngine.

    Per docs/specs/confidence_engine.md §10.1 / C-CE-9, the
    ``input_snapshot.config_version`` carries the **confidence** config
    version (NOT the decision config version) — this is the canonical
    quirk PolicyEngine relies on at
    docs/specs/policy_engine.md §4.1.
    """
    return DecisionOutput(
        risk=risk,
        input_snapshot=InputSnapshot(
            match=MatchInputSnapshot(matched=match_found, similarity=similarity),
            config_version=_CONFIDENCE_CONFIG_VERSION,
        ),
    )


def _build_policy_context(
    *,
    content_type: str,
    matched_asset: Optional[Dict[str, Any]],
    uploader_trust_level: Optional[str],
) -> PolicyContext:
    """Assemble the PolicyContext.

    Operational fields default conservatively (first_offense=True;
    has_prior_disputes=False; no recent violations) — these will be
    sourced from a future operator/observation store integration. The
    defaults are biased toward leniency so PolicyEngine's R2 first-
    offense downgrade fires; production rollout SHOULD wire these to
    durable state before relying on the ladder for automated
    enforcement.
    """
    owner_trust_level = (
        matched_asset.get("trust_level") if matched_asset else None
    )
    matched_asset_owner = matched_asset.get("owner") if matched_asset else None
    return PolicyContext(
        is_first_offense=True,
        has_prior_disputes=False,
        prior_dispute_outcomes=[],
        content_type=content_type,
        trust_owner_tier=owner_trust_level or "unknown",
        trust_uploader_tier=uploader_trust_level or "basic",
        uploader_is_default=(uploader_trust_level is None),
        owner_is_default=(matched_asset is None),
        recent_violations_count=0,
        signal_source=_POLICY_SIGNAL_SOURCE,
        has_multiple_matches=False,
        matched_asset_owner=matched_asset_owner,
        distinct_matched_owner_count=1,
        # H-2 lineage: bind the upstream embedding model version into
        # the evaluation_hash so a model upgrade is reflected in the
        # audit hash even when (band, tier, signals) are unchanged.
        embedding_model_version=embedding_engine.MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Output serialisation (for stages dict + payloads)
# ---------------------------------------------------------------------------


def _serialise_breakdown(risk: RiskScore) -> Dict[str, Dict[str, float]]:
    bd = risk.breakdown
    return {
        "similarity": {"raw": bd.similarity.raw, "weighted": bd.similarity.weighted},
        "trust_owner": {"raw": bd.trust_owner.raw, "weighted": bd.trust_owner.weighted},
        "trust_uploader": {
            "raw": bd.trust_uploader.raw,
            "weighted": bd.trust_uploader.weighted,
        },
        "velocity": {"raw": bd.velocity.raw, "weighted": bd.velocity.weighted},
        "match_quality": {
            "raw": bd.match_quality.raw,
            "weighted": bd.match_quality.weighted,
        },
    }


def _serialise_risk(risk: RiskScore) -> Dict[str, Any]:
    return {
        "composite": risk.composite,
        "band": risk.band.value,
        "breakdown": _serialise_breakdown(risk),
        "decision_config_version": risk.config_version,
    }


def _serialise_confidence(confidence) -> Dict[str, Any]:
    return {
        "composite": confidence.composite,
        "tier": confidence.tier.value,
        "agreement": confidence.agreement,
        "completeness": confidence.completeness,
        "uncertainty": confidence.uncertainty,
        "triggered_conditions": [r.value for r in confidence.triggered_conditions],
        "confidence_config_version": _CONFIDENCE_CONFIG_VERSION,
    }


def _serialise_policy_result(result: PolicyResult) -> Dict[str, Any]:
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Pipeline body
# ---------------------------------------------------------------------------


def run_pipeline(job_id: str) -> None:
    token = _acquire_lock(job_id)
    if token is None:
        # Another worker is already processing this job. Redelivery is a no-op.
        return

    try:
        job = job_store.get_job(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            # State guard: only QUEUED jobs may advance. Anything else
            # (PROCESSING / COMPLETED / FAILED / FLAGGED / missing) means
            # this is a duplicate delivery — exit without side effects.
            return

        try:
            job_store.update_status(job_id, JobStatus.PROCESSING)

            payload: Any = job.metadata or {}
            metadata_dict: Dict[str, Any] = (
                payload if isinstance(payload, dict) else {}
            )
            content_type: str = metadata_dict.get("content_type") or "video"
            # H-3: uploader trust MUST come from an authenticated identity
            # source, not from caller-controlled metadata. Until auth lands,
            # we hardcode to None so the ConfidenceEngine treats trust as
            # registry-default and the DecisionEngine uses the trust floor.
            # Any caller-supplied value is ignored and logged as a probable
            # privilege-escalation attempt.
            if metadata_dict.get("uploader_trust_level") is not None or (
                isinstance(metadata_dict.get("metadata"), dict)
                and metadata_dict["metadata"].get("uploader_trust_level")
                is not None
            ):
                _logger.warning(
                    "ignored caller-supplied uploader_trust_level",
                    extra={"job_id": job_id},
                )
            uploader_trust_level: Optional[str] = None

            # ------------------------------------------------------------
            # Fingerprint
            # ------------------------------------------------------------
            with stage_event(job_id, "fingerprint"):
                fingerprint = fingerprint_engine.compute_fingerprint(payload)
                content_hash = fingerprint.content_hash
                job_store.update_stage(
                    job_id, "fingerprint", {"hash": content_hash}
                )

            publish_event(
                job_id,
                PipelineEventType.FINGERPRINT_READY,
                FingerprintReadyPayload(
                    content_hash=content_hash,
                    model_version=fingerprint.model_version,
                    source_mode=fingerprint.source_mode,
                ),
            )

            # ------------------------------------------------------------
            # Embedding
            # ------------------------------------------------------------
            with stage_event(job_id, "embedding"):
                vector = embedding_engine.embed(content_hash)
                job_store.update_stage(
                    job_id,
                    "embedding",
                    {
                        "vector": vector,
                        "model_version": embedding_engine.MODEL_VERSION,
                    },
                )

            publish_event(
                job_id,
                PipelineEventType.EMBEDDING_READY,
                EmbeddingReadyPayload(
                    dimension=len(vector),
                    model_version=embedding_engine.MODEL_VERSION,
                ),
            )

            # ------------------------------------------------------------
            # Matching
            # ------------------------------------------------------------
            with stage_event(job_id, "matching"):
                match = matching_engine.find_best_match(vector)
                matched_asset_dict: Optional[dict] = (
                    {
                        "asset_id": match.matched_asset.asset_id,
                        "owner": match.matched_asset.owner,
                        "trust_level": match.matched_asset.trust_level,
                    }
                    if match.matched_asset is not None
                    else None
                )
                job_store.update_stage(
                    job_id,
                    "matching",
                    {
                        "matched_asset": matched_asset_dict,
                        "similarity": match.similarity,
                    },
                )

            match_found = match.matched_asset is not None

            if match_found:
                publish_event(
                    job_id,
                    PipelineEventType.MATCH_FOUND,
                    MatchFoundPayload(
                        matched_asset_id=match.matched_asset.asset_id,
                        similarity=match.similarity,
                        owner=match.matched_asset.owner,
                        trust_level=match.matched_asset.trust_level,
                    ),
                )
            else:
                publish_event(
                    job_id,
                    PipelineEventType.MATCH_NOT_FOUND,
                    MatchNotFoundPayload(similarity=match.similarity),
                )

            # ------------------------------------------------------------
            # Evaluation — DecisionEngine + ConfidenceEngine
            # ------------------------------------------------------------
            # Per docs/specs/job_processing.md §5.2, the EVALUATION phase
            # invokes both engines; their outputs are independent and
            # MAY be parallelised in the future. The DecisionOutput
            # envelope assembled here is the canonical input contract
            # for the DECISION phase below.
            with stage_event(job_id, "evaluation"):
                decision_input = _build_decision_input(
                    similarity=match.similarity,
                    matched_asset=matched_asset_dict,
                    uploader_trust_level=uploader_trust_level,
                )
                confidence_input = _build_confidence_input(
                    match_found=match_found,
                    similarity=match.similarity,
                    matched_asset=matched_asset_dict,
                    uploader_trust_level=uploader_trust_level,
                )

                risk = decision_engine.compute_risk(
                    decision_input, _THRESHOLD_CONFIG
                )
                confidence = confidence_engine.compute_confidence(
                    confidence_input, _CONFIDENCE_CONFIG
                )

                decision_output = _build_decision_output(
                    risk=risk,
                    match_found=match_found,
                    similarity=match.similarity,
                )

                job_store.update_stage(
                    job_id,
                    "evaluation",
                    {
                        "risk": _serialise_risk(risk),
                        "confidence": _serialise_confidence(confidence),
                    },
                )

            # Canonical engine-triple events
            publish_event(
                job_id,
                PipelineEventType.RISK_SCORED,
                RiskScoredPayload(
                    composite=risk.composite,
                    band=risk.band.value,
                    decision_config_version=risk.config_version,
                    breakdown=_serialise_breakdown(risk),
                ),
            )
            publish_event(
                job_id,
                PipelineEventType.CONFIDENCE_COMPUTED,
                ConfidenceComputedPayload(
                    composite=confidence.composite,
                    tier=confidence.tier.value,
                    agreement=confidence.agreement,
                    completeness=confidence.completeness,
                    uncertainty=confidence.uncertainty,
                    triggered_conditions=[
                        r.value for r in confidence.triggered_conditions
                    ],
                    confidence_config_version=_CONFIDENCE_CONFIG_VERSION,
                ),
            )

            # Legacy bridge: SCORED event with band derived from RiskScore.
            # Retained for backward-compatible consumers; will be retired
            # post-rollout.
            publish_event(
                job_id,
                PipelineEventType.SCORED,
                ScoredPayload(band=risk.band.value, similarity=match.similarity),
            )

            # ------------------------------------------------------------
            # Decision — PolicyEngine
            # ------------------------------------------------------------
            with stage_event(job_id, "decision"):
                policy_context = _build_policy_context(
                    content_type=content_type,
                    matched_asset=matched_asset_dict,
                    uploader_trust_level=uploader_trust_level,
                )
                policy_result = policy_engine.evaluate_policy(
                    decision_output, confidence, policy_context
                )
                job_store.update_stage(
                    job_id,
                    "decision",
                    _serialise_policy_result(policy_result),
                )

            publish_event(
                job_id,
                PipelineEventType.POLICY_DECIDED,
                PolicyDecidedPayload(
                    action=policy_result.final_action.value,
                    triggered_rules=list(policy_result.triggered_rules),
                    primary_reason=policy_result.primary_reason,
                    evaluation_hash=policy_result.evaluation_hash,
                    policy_version=policy_result.policy_version,
                    decision_config_version=policy_result.decision_config_version,
                    confidence_config_version=policy_result.confidence_config_version,
                    risk_band=policy_result.risk_band.value,
                    confidence_tier=policy_result.confidence_tier.value,
                    rules_checked_count=policy_result.rules_checked_count,
                ),
            )

            # ------------------------------------------------------------
            # Enforcement — apply the final action
            # ------------------------------------------------------------
            # The PolicyEngine has already selected the canonical action
            # under PBRA. The enforcement stage today is the boundary at
            # which the action becomes visible; the actual takedown /
            # restriction effector lives in the (planned) enforcement-
            # audit layer (docs/security/enforcement_audit.md). This
            # stage records the evidence and emits the ENFORCED event.
            final_action = policy_result.final_action
            legacy_action = _POLICY_ACTION_TO_LEGACY[final_action]

            with stage_event(job_id, "enforcement"):
                evidence: Dict[str, Any] = {
                    "input_media_id": content_hash,
                    "matched_media_id": (
                        matched_asset_dict["asset_id"]
                        if matched_asset_dict
                        else None
                    ),
                    "owner": (
                        matched_asset_dict["owner"]
                        if matched_asset_dict
                        else None
                    ),
                    "trust_level": (
                        matched_asset_dict["trust_level"]
                        if matched_asset_dict
                        else None
                    ),
                    "similarity_score": match.similarity,
                    "band": risk.band.value,
                    "model_version": embedding_engine.MODEL_VERSION,
                    "timestamp": time.time(),
                    # Policy lineage (per A4 audit-completeness-with-provenance)
                    "policy_action": final_action.value,
                    "policy_legacy_action": legacy_action,
                    "primary_reason": policy_result.primary_reason,
                    "triggered_rules": sorted(policy_result.triggered_rules),
                    "evaluation_hash": policy_result.evaluation_hash,
                    "policy_version": policy_result.policy_version,
                    "decision_config_version": policy_result.decision_config_version,
                    "confidence_config_version": policy_result.confidence_config_version,
                    "embedding_model_version": embedding_engine.MODEL_VERSION,
                }

                # H-4: durable, append-only evidence record. The JobStore
                # `enforcement` stage carries the same dict for the API
                # response, but its TTL means it cannot be the long-term
                # source of truth (storage rules §6).
                evidence_record = record_evidence(job_id, evidence)
                evidence_key = evidence_record.key

                job_store.update_stage(
                    job_id,
                    "enforcement",
                    {
                        "action": legacy_action,
                        "policy_action": final_action.value,
                        "evidence_key": evidence_key,
                        "reason": evidence,
                    },
                )

            publish_event(
                job_id,
                PipelineEventType.ENFORCED,
                EnforcedPayload(
                    action=legacy_action,
                    similarity=match.similarity,
                    band=risk.band.value,
                    model_version=embedding_engine.MODEL_VERSION,
                    matched_media_id=(
                        matched_asset_dict["asset_id"]
                        if matched_asset_dict
                        else None
                    ),
                    policy_action=final_action.value,
                    evaluation_hash=policy_result.evaluation_hash,
                    policy_version=policy_result.policy_version,
                ),
            )

            # ------------------------------------------------------------
            # Terminal transition
            # ------------------------------------------------------------
            result = {
                "match": match_found,
                "owner": (
                    match.matched_asset.owner if match.matched_asset else None
                ),
                "confidence": match.similarity,           # legacy field — echoed for compat
                "action": legacy_action,                  # legacy 3-action
                "policy_action": final_action.value,      # canonical 5-action
                "primary_reason": policy_result.primary_reason,
                "evaluation_hash": policy_result.evaluation_hash,
                "evidence_key": evidence_key,             # H-4 durable pointer
                "reason": evidence,
            }
            job_store.set_result(job_id, result)

            terminal = _POLICY_ACTION_TO_TERMINAL[final_action]
            job_store.update_status(job_id, terminal)

            publish_event(
                job_id,
                PipelineEventType.JOB_COMPLETED,
                JobCompletedPayload(
                    terminal_status=terminal.value,
                    action=legacy_action,
                ),
            )

        except Exception as exc:
            # Pipeline body failed — never strand the job in PROCESSING.
            # Order: try to flip state first, then publish JOB_FAILED only
            # if we owned the transition, so we never double-publish a
            # terminal event for a job another writer already failed.
            failed = False
            try:
                job_store.set_failure(job_id, f"{type(exc).__name__}: {exc}")
                failed = True
            except ValueError:
                # Already in a terminal state; nothing to record.
                pass

            if failed:
                publish_event(
                    job_id,
                    PipelineEventType.JOB_FAILED,
                    JobFailedPayload(
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ),
                )
    finally:
        _release_lock(job_id, token)
