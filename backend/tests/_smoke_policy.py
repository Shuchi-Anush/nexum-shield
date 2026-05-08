from dataclasses import dataclass
from app.models.policy_models import PolicyAction, PolicyContext, EvidenceStrength
from app.models.decision_models import RiskBand
from app.models.confidence_models import ConfidenceBreakdown, ConfidenceTier, ConfidenceReasonCode
from app.engines.policy_engine import evaluate_policy, _derive_evidence_strength


@dataclass
class _M:
    matched: bool
    similarity: float


@dataclass
class _IS:
    match: _M
    config_version: str


@dataclass
class _R:
    band: RiskBand
    composite: float
    config_version: str


@dataclass
class _D:
    risk: _R
    input_snapshot: _IS


def mk(band, comp, matched, sim, dec_v="dv1", conf_v="cv1"):
    return _D(
        risk=_R(band=band, composite=comp, config_version=dec_v),
        input_snapshot=_IS(match=_M(matched=matched, similarity=sim), config_version=conf_v),
    )


def conf(tier, comp, reasons=()):
    return ConfidenceBreakdown(
        agreement=0.5, completeness=0.5, uncertainty=0.1,
        composite=comp, tier=tier, triggered_conditions=tuple(reasons),
    )


def ctx(**overrides):
    base = dict(
        is_first_offense=False, has_prior_disputes=False, prior_dispute_outcomes=[],
        content_type="video", trust_owner_tier="basic", trust_uploader_tier="basic",
        uploader_is_default=False, owner_is_default=False, signal_source="fingerprint",
        has_multiple_matches=False, distinct_matched_owner_count=1,
    )
    base.update(overrides)
    return PolicyContext(**base)


def main():
    # 1) NO MATCH -> ALLOW (S1 short-circuit)
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.9, False, 0.0),
        conf(ConfidenceTier.LOW, 0.35, [ConfidenceReasonCode.NO_MATCH]),
        ctx(),
    )
    assert r.final_action == PolicyAction.ALLOW, r.final_action
    assert r.triggered_rules == ["S1_NO_MATCH"]
    assert r.rules_checked_count == 10
    print("1 OK no-match -> ALLOW")

    # 2) HIGH x HIGH + STRONG evidence -> TAKEDOWN
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.9, True, 0.92),
        conf(ConfidenceTier.HIGH, 0.85),
        ctx(signal_source="fusion"),
    )
    assert r.final_action == PolicyAction.TAKEDOWN, r.final_action
    print("2 OK high+high+fusion -> TAKEDOWN")

    # 3) Invariant: non-HIGH never reaches TAKEDOWN
    for band in [RiskBand.LOW, RiskBand.MEDIUM, RiskBand.HIGH]:
        for tier in [ConfidenceTier.LOW, ConfidenceTier.MEDIUM]:
            r = evaluate_policy(mk(band, 0.5, True, 0.5), conf(tier, 0.5), ctx())
            assert r.final_action != PolicyAction.TAKEDOWN
    print("3 OK non-HIGH never reaches TAKEDOWN")

    # 4) S4 caps at FLAG when confidence < 0.30
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.75, True, 0.5),
        conf(ConfidenceTier.LOW, 0.25),
        ctx(),
    )
    assert r.final_action == PolicyAction.FLAG, r.final_action
    assert "S4_CONFIDENCE_CEILING" in r.triggered_rules
    print("4 OK S4 caps at FLAG")

    # 5) R2 first-offense (evidence MODERATE)
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.85, True, 0.75),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(is_first_offense=True, signal_source="fingerprint"),
    )
    assert r.final_action == PolicyAction.REVIEW, r.final_action
    assert "R2_FIRST_OFFENSE" in r.triggered_rules
    print("5 OK R2 first-offense -> REVIEW")

    # 6) R2 SKIPPED when evidence STRONG (v4 fix)
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.85, True, 0.85),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(is_first_offense=True, signal_source="fusion"),
    )
    assert r.final_action == PolicyAction.RESTRICT, r.final_action
    assert "R2_FIRST_OFFENSE" not in r.triggered_rules
    print("6 OK R2 skipped when STRONG")

    # 7) R5 dispute caution (with case + space normalization)
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.85, True, 0.75),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(
            has_prior_disputes=True,
            prior_dispute_outcomes=["Upheld", "OVERTURNED"],
            signal_source="fingerprint",
        ),
    )
    assert r.final_action == PolicyAction.REVIEW, r.final_action
    assert "R5_DISPUTE_CAUTION" in r.triggered_rules
    print("7 OK R5 overturned -> REVIEW")

    # 8) R1 gray zone upgrade
    r = evaluate_policy(
        mk(RiskBand.MEDIUM, 0.45, True, 0.78),
        conf(ConfidenceTier.LOW, 0.38, [ConfidenceReasonCode.GRAY_ZONE]),
        ctx(),
    )
    assert r.final_action == PolicyAction.REVIEW, r.final_action
    assert "R1_GRAY_ZONE" in r.triggered_rules
    print("8 OK R1 gray-zone -> REVIEW")

    # 9) R1 NOT firing when confidence < 0.30
    r = evaluate_policy(
        mk(RiskBand.MEDIUM, 0.45, True, 0.78),
        conf(ConfidenceTier.LOW, 0.20, [ConfidenceReasonCode.GRAY_ZONE]),
        ctx(),
    )
    assert r.final_action == PolicyAction.FLAG, r.final_action
    assert "R1_GRAY_ZONE" not in r.triggered_rules
    print("9 OK R1 gated below 0.30")

    # 10) R4 verified-owner upgrade
    r = evaluate_policy(
        mk(RiskBand.MEDIUM, 0.55, True, 0.90),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(trust_owner_tier="verified", signal_source="fingerprint"),
    )
    assert r.final_action == PolicyAction.RESTRICT, r.final_action
    assert "R4_VERIFIED_OWNER" in r.triggered_rules
    print("10 OK R4 verified-owner -> RESTRICT")

    # 11) S5 unsupported content_type -> REVIEW
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.85, True, 0.75),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(content_type="audio", signal_source="fingerprint"),
    )
    assert r.final_action == PolicyAction.REVIEW, r.final_action
    assert "S5_CONTENT_TYPE_GATE" in r.triggered_rules
    print("11 OK S5 unsupported type")

    # 12) Evidence derivation
    assert _derive_evidence_strength(
        match_found=False, signal_source="fusion", similarity=0.9,
        has_multiple_matches=True, distinct_owner_count=2,
    ) == EvidenceStrength.NONE
    assert _derive_evidence_strength(
        match_found=True, signal_source="fusion", similarity=0.75,
        has_multiple_matches=False, distinct_owner_count=1,
    ) == EvidenceStrength.STRONG
    # WEAK + multi-match (any) -> MODERATE
    assert _derive_evidence_strength(
        match_found=True, signal_source="fingerprint", similarity=0.50,
        has_multiple_matches=True, distinct_owner_count=1,
    ) == EvidenceStrength.MODERATE
    # distinct owners + similarity>=0.70 -> STRONG
    assert _derive_evidence_strength(
        match_found=True, signal_source="fingerprint", similarity=0.75,
        has_multiple_matches=True, distinct_owner_count=2,
    ) == EvidenceStrength.STRONG
    # MODERATE + single-owner multi -> stays MODERATE (v5 stabilization)
    assert _derive_evidence_strength(
        match_found=True, signal_source="fingerprint", similarity=0.75,
        has_multiple_matches=True, distinct_owner_count=1,
    ) == EvidenceStrength.MODERATE
    print("12 OK evidence derivation")

    # 13) Determinism
    r1 = evaluate_policy(
        mk(RiskBand.HIGH, 0.85, True, 0.75),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(is_first_offense=True, signal_source="fingerprint"),
    )
    r2 = evaluate_policy(
        mk(RiskBand.HIGH, 0.85, True, 0.75),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(is_first_offense=True, signal_source="fingerprint"),
    )
    assert r1.evaluation_hash == r2.evaluation_hash
    print("13 OK deterministic hash =", r1.evaluation_hash)

    # 14) signal_source normalization
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.9, True, 0.92),
        conf(ConfidenceTier.HIGH, 0.85),
        ctx(signal_source="  Fusion  "),
    )
    assert r.final_action == PolicyAction.TAKEDOWN, r.final_action
    print("14 OK signal_source normalization")

    # 15) rules_checked_count == 10
    assert r1.rules_checked_count == 10
    print("15 OK rules_checked_count == 10")

    # 16) primary_reason priority — S2 wins over R3 if both fire
    #     (Construct: HIGH HIGH base TAKEDOWN with non-fusion + sim<0.70 -> evidence MODERATE.
    #      But S2 needs conf != HIGH; matrix only emits TAKEDOWN at HIGH conf.
    #      Skip — S2 vs R3 is structurally impossible with current matrix.)

    # 17) policy_version + version triad echo
    assert r1.policy_version == "v1.0"
    assert r1.decision_config_version == "dv1"
    assert r1.confidence_config_version == "cv1"
    print("17 OK version triad echoed")

    # 18) S5 vetoes R4 — verified owner + unsupported content type
    #     base=RESTRICT (HIGH x MED). S5 caps at REVIEW. R4 conditions
    #     met (verified, sim>=0.85, MED+, current==REVIEW) but
    #     safety_cap_target=REVIEW so R4's effective target = REVIEW =
    #     current. R4 does NOT upgrade. Final = REVIEW.
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.85, True, 0.90),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(content_type="audio", trust_owner_tier="verified",
            signal_source="fingerprint"),
    )
    assert r.final_action == PolicyAction.REVIEW, r.final_action
    assert "S5_CONTENT_TYPE_GATE" in r.triggered_rules
    assert "R4_VERIFIED_OWNER" not in r.triggered_rules
    print("18 OK S5 vetoes R4 (safety_cap_target=REVIEW)")

    # 19) S4 vetoes R1 — confidence below floor + gray zone reason
    #     base=FLAG (MED x LOW). S4 fires at conf<0.30, capping at FLAG
    #     and lowering safety_cap_target to FLAG. R1's gray-zone gate
    #     would normally upgrade FLAG -> REVIEW, but the ceiling clamp
    #     blocks it. (R1 also self-rejects via the 0.30 floor.)
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.65, True, 0.78),
        conf(ConfidenceTier.LOW, 0.20, [ConfidenceReasonCode.GRAY_ZONE]),
        ctx(),
    )
    assert r.final_action == PolicyAction.FLAG, r.final_action
    assert "S4_CONFIDENCE_CEILING" in r.triggered_rules
    assert "R1_GRAY_ZONE" not in r.triggered_rules
    print("19 OK S4 vetoes R1 (safety_cap_target=FLAG)")

    # 20) v2 §5 worked example exactly: Phase 6 downgrade beats Phase 5
    #     upgrade. base=REVIEW (MED x MED), R4 fires (REVIEW->RESTRICT),
    #     R2 fires (RESTRICT->REVIEW). Final = REVIEW.
    r = evaluate_policy(
        mk(RiskBand.MEDIUM, 0.55, True, 0.90),
        conf(ConfidenceTier.MEDIUM, 0.55),
        ctx(is_first_offense=True, trust_owner_tier="verified",
            signal_source="fingerprint"),
    )
    assert r.final_action == PolicyAction.REVIEW, r.final_action
    assert "R4_VERIFIED_OWNER" in r.triggered_rules
    assert "R2_FIRST_OFFENSE" in r.triggered_rules
    print("20 OK v2 §5 example: Phase 6 wins over Phase 5")

    # 21) v4 §1 Case C — phase4=TAKEDOWN, GLOBAL_MAX_UPGRADE blocks
    #     Phase 5 from elevating further. Here R3 fires because evidence
    #     is MODERATE (non-fusion + sim<0.85 with sim>=0.70) so TAKEDOWN
    #     downgrades to RESTRICT.
    r = evaluate_policy(
        mk(RiskBand.HIGH, 0.95, True, 0.75),
        conf(ConfidenceTier.HIGH, 0.85),
        ctx(signal_source="fingerprint"),  # non-fusion → MODERATE
    )
    assert r.final_action == PolicyAction.RESTRICT, r.final_action
    assert "R3_EVIDENCE_GATE" in r.triggered_rules
    print("21 OK R3 evidence gate downgrades TAKEDOWN")

    print("\nALL SCENARIOS PASS")


if __name__ == "__main__":
    main()
