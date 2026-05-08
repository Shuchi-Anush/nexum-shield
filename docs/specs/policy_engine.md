---
authority: SPEC
domain: policy
status: ACTIVE
version: 1.0
stability: EVOLVING
owner: policy (interim: architect)
supersedes:
  - .claude/memory/policy_engine_spec_v1.md (working draft, untracked) — original implementation-ready spec
  - .claude/memory/policy_engine_spec_v2.md (refinement delta)
  - .claude/memory/policy_engine_spec_v3.md (strict fixes delta)
  - .claude/memory/policy_engine_spec_v4.md (final corrections delta)
  - .claude/memory/policy_engine_spec_v5.md (production safety delta)
  - docs/architecture/POLICY_ENGINE_CANONICAL_SEMANTICS.md (zero-byte placeholder)
  - docs/architecture/POLICY_INVARIANTS.md (zero-byte placeholder)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; canonical-spec ratification; to be backfilled)
---

# Policy Engine — Canonical Specification

The PolicyEngine is the **DECISION** phase of the Nexum Shield
pipeline (per A1). It maps a (DecisionOutput, ConfidenceBreakdown,
PolicyContext) triple to a `PolicyAction`, accompanied by an audit
record sufficient for replay (A5), audit reconstruction (A4),
human-review reversal (A6), and evidence preservation (A7).

This document is the canonical specification — Tier 2 (SPEC) — and
supersedes the v1–v5 spec series in `.claude/memory/` plus the two
zero-byte placeholders in `docs/architecture/`. The implementation
is `backend/app/engines/policy_engine.py` and
`backend/app/models/policy_models.py`.

---

## §1 Purpose and Authority

### §1.1 Purpose

The PolicyEngine performs the **DECISION** phase: given an upstream
EVALUATION (risk + confidence) and operational/evidence context, it
selects the enforcement action and emits an audit record.

It is a **pure deterministic function**. No randomness. No I/O
outside an ERROR-level diagnostic log when the terminal invariant
guard fires (§9). No external state.

### §1.2 Position in the pipeline

Per A1 (PIPELINE PHASE INTEGRITY):

```
INGESTION → ANALYSIS → EVALUATION → DECISION → ENFORCEMENT
                                       ↑
                                  PolicyEngine
```

AUDITABILITY runs continuously alongside; PolicyEngine emits its
contribution to the A4 audit record on every call.

### §1.3 Authority

This document is **TIER 2 (SPEC)** per
`docs/constitution/GOVERNANCE.md` §1. Owned by the **policy** domain
(`docs/governance/DOMAINS.md`). Modification:

| Change | ADR tier |
|---|---|
| Threshold tweak (e.g., similarity floor 0.70 → 0.72) | Lightweight |
| New uncertainty term, condition refinement | Lightweight |
| New rule (e.g., R6) | Standard |
| Action-ladder change (cross-domain consumer surface) | Constitutional |
| Base matrix cell change | Standard |
| New EvidenceStrength tier | Standard |
| Phase reordering (PBRA contract) | Constitutional |

### §1.4 Stability

**EVOLVING** per `docs/constitution/GOVERNANCE.md` §8. Compatibility
expectations: low — consumers should expect change at each minor
version bump. Graduation to STABLE requires a Standard ADR (§16.1).

The `PolicyAction` enum is a separate concern: it is consumed
cross-domain (api, pipeline, security per `DOMAINS.md`). Promoting
the enum specifically to LOCKED status (independently of this spec)
is a design goal once enforcement integrations stabilise.

---

## §2 Action Model

### §2.1 PolicyAction

Five ordered severity levels:

| Severity | Value | Effect |
|---|---|---|
| 0 | `ALLOW` | no action; content passes |
| 1 | `FLAG` | internal tracking only; soft flag for analytics |
| 2 | `REVIEW` | queued for human moderation |
| 3 | `RESTRICT` | automated restriction (geo-block, demonetize, etc.) |
| 4 | `TAKEDOWN` | automated removal |

Severity ordering: `ALLOW < FLAG < REVIEW < RESTRICT < TAKEDOWN`.
Used for:
- Conflict resolution among same-direction rules (higher severity
  wins).
- Downgrade clamping (`min_action(a, b)` returns the lower-severity
  action).
- Audit comparison (more or less severe than the proposed?).

### §2.2 Reversibility

Per A6 (HUMAN REVIEW AUTHORITY), every action MUST remain
reversible:

| Action | Reversible? | Human required to reverse? |
|---|---|---|
| ALLOW | N/A | N/A |
| FLAG | yes | no (auto-unflag possible) |
| REVIEW | yes | no (review can clear) |
| RESTRICT | yes | yes (manual unrestrict) |
| TAKEDOWN | yes | yes (manual restore + appeal) |

Per A4's append-only rule, reversal is recorded by **appending** a
reversal entry. The original automated decision is preserved.
Reversal mechanics live at the platform / audit layer (see
`docs/security/enforcement_audit.md` *(planned)*).

### §2.3 Mapping from legacy 3-level model

The legacy 3-level model in `.claude/rules/enforcement.md`
(`SOFT FLAG`, `REVIEW REQUIRED`, `AUTO ENFORCEMENT`) is REMOVED
(per `docs/state/STATE.md`).

| Legacy | Current | Notes |
|---|---|---|
| SOFT FLAG | FLAG | internal tracking unchanged |
| REVIEW REQUIRED | REVIEW | human queue unchanged |
| AUTO ENFORCEMENT | RESTRICT or TAKEDOWN | split per A3: RESTRICT = partial; TAKEDOWN = full removal at HIGH confidence only |

The split was driven by A3 (HIGH-confidence requirement for
TAKEDOWN); the legacy model conflated all automated enforcement
into one bin.

---

## §3 PBRA Execution Model

The engine follows **PBRA**: PROPOSE → BOUND → REFINE → ASSERT.
Each phase has a contract; ordering is fixed; downstream phases
respect upstream constraints.

### §3.1 Phase ladder

```
┌─────────────────────────────────────────────────────────────────┐
│  PROPOSE                                                       │
│    Phase 1 — base matrix lookup (§5)                          │
│            S1 (NO_MATCH) is the hard override here              │
├─────────────────────────────────────────────────────────────────┤
│  BOUND                                                         │
│    Phase 2 — S2, S3 (matrix-evolution guards): upper caps     │
│    Phase 3 — S4 (confidence floor): cap at FLAG                │
│    Phase 4 — S5 (content type gate): cap at REVIEW             │
│    Each fired safety rule narrows FeasibleBounds.upper.        │
├─────────────────────────────────────────────────────────────────┤
│  REFINE                                                        │
│    Phase 5 — R1, R4 upgrades, clamped by FeasibleBounds.upper │
│    Phase 6 — R2, R3, R5 downgrades; final authority            │
├─────────────────────────────────────────────────────────────────┤
│  ASSERT                                                        │
│    Terminal invariant guard:                                   │
│      (confidence_tier ≠ HIGH ∧ action == TAKEDOWN) → RESTRICT │
│    Defense-in-depth backup for S2.                             │
└─────────────────────────────────────────────────────────────────┘
```

### §3.2 Phase contracts

| Phase | Direction | Can override prior? | Mechanism |
|---|---|---|---|
| 1 — Hard override (S1) | force | yes (terminates) | short-circuit: action ← ALLOW |
| 2 — Upper caps (S2, S3) | down | constrains base | tighten `bounds.upper` |
| 3 — Lower cap (S4) | down | constrains 1–2 | tighten `bounds.upper` |
| 4 — Type cap (S5) | down | constrains 1–3 | tighten `bounds.upper` |
| 5 — Risk upgrades (R1, R4) | up only | clamped by `bounds.upper` | `current ← min(target, bounds.upper)` if `severity(current) < severity(effective)` |
| 6 — Risk downgrades (R2, R3, R5) | down | final over Phase 5 | `current ← target` (downgrade only) |
| ASSERT | force | post-phase audit | re-clamps if A3 invariant violated |

### §3.3 Numbering correspondence

The phase numbers (1–6 + ASSERT) preserve the v2 §5 spec numbering
for traceability with archived working drafts. PBRA's four
super-stages (PROPOSE / BOUND / REFINE / ASSERT) are the canonical
mental model.

---

## §4 Inputs

```
evaluate_policy(decision, confidence, context) -> PolicyResult
```

### §4.1 DecisionOutput (structural protocol)

The engine accesses upstream output via duck-typed `Protocol`. The
DecisionEngine spec (`docs/specs/decision_engine.md` *(planned)*)
defines the concrete shape; PolicyEngine requires only attribute
access:

| Path | Type | Used for |
|---|---|---|
| `decision.risk.band` | `RiskBand` | base matrix lookup (§5) |
| `decision.risk.composite` | `float` | informational; echoed to audit |
| `decision.risk.config_version` | `str` | audit lineage (A4) |
| `decision.input_snapshot.match.matched` | `bool` | S1 + R1 + evidence derivation |
| `decision.input_snapshot.match.similarity` | `float` | evidence + R4 + audit |
| `decision.input_snapshot.config_version` | `str` | audit lineage (A4) |

### §4.2 ConfidenceBreakdown

Read-only access to:
- `confidence.tier` — `ConfidenceTier` (LOW / MEDIUM / HIGH); base
  matrix lookup, S2/S3, R4 condition.
- `confidence.composite` — `float ∈ [0, 1]`; S4 condition,
  R1 condition, audit + hash.
- `confidence.triggered_conditions` — sequence of
  `ConfidenceReasonCode`; checked for `GRAY_ZONE` (R1).

Defined in `docs/specs/confidence_engine.md` *(planned)*.

### §4.3 PolicyContext

Operational and evidence signals NOT available in DecisionOutput or
ConfidenceBreakdown. Per the v2 §1 reconciliation, duplicating
upstream signals is forbidden — read those via the upstream objects
directly.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `is_first_offense` | `bool` | — | R2 condition |
| `has_prior_disputes` | `bool` | — | R5 condition |
| `prior_dispute_outcomes` | `list[str]` | `[]` | R5 condition (normalized internally per §11.2) |
| `content_type` | `str` | — | S5 condition (`"video"` and `"image"` are supported) |
| `trust_owner_tier` | `str` | — | R4 condition (`"verified"` triggers) |
| `trust_uploader_tier` | `str` | — | informational |
| `uploader_is_default` | `bool` | — | informational |
| `owner_is_default` | `bool` | — | informational |
| `recent_violations_count` | `int` | `0` | reserved for R6_REPEAT_OFFENDER (future, §15) |
| `signal_source` | `str` | — | evidence derivation; normalized at entry per §11.1 |
| `has_multiple_matches` | `bool` | `False` | evidence derivation |
| `matched_asset_owner` | `Optional[str]` | `None` | audit only |
| `distinct_matched_owner_count` | `int` | `1` | evidence derivation |

PolicyContext is a Pydantic model (`backend/app/models/policy_models.py::PolicyContext`).

---

## §5 PROPOSE: Base Matrix

Phase 1 lookup. The base action depends on risk band and confidence
tier only.

|  | Conf LOW `[0, 0.40)` | Conf MEDIUM `[0.40, 0.70)` | Conf HIGH `[0.70, 1.0]` |
|---|---|---|---|
| Risk LOW `[0, 0.40)` | ALLOW | ALLOW | ALLOW |
| Risk MEDIUM `[0.40, 0.70)` | FLAG | REVIEW | REVIEW |
| Risk HIGH `[0.70, 1.0]` | REVIEW | RESTRICT | TAKEDOWN |

### §5.1 Cell rationale

| Cell | Action | Reason |
|---|---|---|
| LOW × * | ALLOW | low risk = nothing actionable, regardless of confidence |
| MED × LOW | FLAG | some risk but too uncertain to escalate; internal tracking |
| MED × MED/HIGH | REVIEW | moderate risk warrants human eyes |
| HIGH × LOW | REVIEW | high risk + low confidence MUST review (cannot auto-enforce) |
| HIGH × MED | RESTRICT | high risk + moderate confidence = restrict but don't remove |
| HIGH × HIGH | TAKEDOWN | only cell where automated removal is permitted (per A3) |

### §5.2 Matrix constraint validation

On module import, `_validate_matrix_constraint_consistency()`
asserts:

1. **Completeness** — all 9 cells (RiskBand × ConfidenceTier) are
   populated; no extras.
2. **S2 invariant** — no cell proposes TAKEDOWN at non-HIGH
   confidence. Otherwise S2 silently becomes an active downgrade
   rather than the documented matrix-evolution guard (§6.3).
3. **S3 invariant** — no cell proposes RESTRICT at LOW confidence.
   Otherwise S3 silently becomes an active downgrade (§6.4).
4. **Action validity** — every proposed action is a known
   `PolicyAction` value.

Violations are `AssertionError` at import time, blocking startup.
This is a **P0** check per `docs/constitution/GOVERNANCE.md` §5.

---

## §6 BOUND: Safety Rules (S1–S5)

Five safety rules across three phases. When a rule fires, it both
mutates `current_action` and tightens `FeasibleBounds.upper` to its
cap target. The bounds object is consumed by Phase 5 (REFINE
upgrades).

### §6.1 FeasibleBounds

```python
@dataclass(frozen=True)
class FeasibleBounds:
    upper: PolicyAction      # severity-wise upper limit on Phase-5 upgrades

    def tighten_upper(self, target: PolicyAction) -> "FeasibleBounds":
        return FeasibleBounds(upper=min_action(self.upper, target))

    def clamp(self, action: PolicyAction) -> PolicyAction:
        return min_action(action, self.upper)
```

Invariants:
- `bounds.upper` is initialised to `GLOBAL_MAX_UPGRADE = RESTRICT`
  (Phase 5 may NEVER produce TAKEDOWN via upgrade).
- `bounds.upper` only ever narrows (`tighten_upper` returns a new
  instance whose `upper` is ≤ the prior `upper`).
- `bounds` is immutable per evaluation (frozen dataclass).

### §6.2 S1 — NO_MATCH (hard override)

```
ID:        S1_NO_MATCH
Condition: match_found == False
Effect:    force action ← ALLOW; short-circuit Phases 2–6 + ASSERT
Reason:    "No matching protected asset. Enforcement without match is forbidden."
Authority: A2 (MATCH PREREQUISITE).
```

When S1 fires the engine still records `rules_checked_count = 10`
(EXPECTED_RULE_COUNT) per the v3 §3 audit-completeness contract.
Remaining rules are conceptually visited but their preconditions
are unsatisfied due to the forced ALLOW.

### §6.3 S2 — TAKEDOWN_CONFIDENCE_GATE (matrix-evolution guard)

```
ID:        S2_TAKEDOWN_CONFIDENCE_GATE
Condition: current_action == TAKEDOWN AND confidence_tier != HIGH
Effect:    current_action ← RESTRICT
           bounds ← bounds.tighten_upper(RESTRICT)
Reason:    "TAKEDOWN requires HIGH confidence. Downgraded to RESTRICT."
Authority: A3 (CONFIDENCE-GATED ENFORCEMENT).
```

**Status: matrix-evolution guard.** The base matrix proposes
TAKEDOWN only at HIGH×HIGH; S2's precondition is unreachable from
PROPOSE today. S2 remains in the rule set as defense-in-depth: any
future matrix change introducing TAKEDOWN at non-HIGH confidence
would activate S2 immediately. The startup validator (§5.2)
prevents accidental drift.

### §6.4 S3 — RESTRICT_CONFIDENCE_GATE (matrix-evolution guard)

```
ID:        S3_RESTRICT_CONFIDENCE_GATE
Condition: current_action == RESTRICT AND confidence_tier == LOW
Effect:    current_action ← REVIEW
           bounds ← bounds.tighten_upper(REVIEW)
Reason:    "RESTRICT requires at least MEDIUM confidence. Downgraded to REVIEW."
Authority: A3 derivative; codified independently for defense-in-depth.
```

**Status: matrix-evolution guard.** Same justification as S2 — base
matrix produces RESTRICT only at HIGH×MEDIUM, never at LOW. The
startup validator catches drift.

### §6.5 S4 — CONFIDENCE_CEILING

```
ID:        S4_CONFIDENCE_CEILING
Condition: confidence.composite < 0.30
           AND severity(current_action) > severity(FLAG)
           (implicitly requires match_found, since S1 short-circuits
            otherwise)
Effect:    current_action ← FLAG
           bounds ← bounds.tighten_upper(FLAG)
Reason:    "Confidence {composite:.2f} below 0.30. Maximum action capped at FLAG."
Authority: A4 + A6 (uncertain decisions are not auto-enforced).
```

**Renamed from v1 `S4_CONFIDENCE_FLOOR`.** Per v2 §4 reconciliation:
v1's "upgrade to REVIEW" forced uncertain cases into the human
review queue (wasting reviewer time on noise). The v2 correction
caps at FLAG — uncertain cases get internal tracking only.

The threshold is `CONFIDENCE_FLOOR = 0.30`.

### §6.6 S5 — CONTENT_TYPE_GATE

```
ID:        S5_CONTENT_TYPE_GATE
Condition: context.content_type ∉ SUPPORTED_ENFORCEMENT_TYPES
           AND severity(current_action) > severity(REVIEW)
Effect:    current_action ← REVIEW
           bounds ← bounds.tighten_upper(REVIEW)
Reason:    "Content type '{type}' not approved for automated enforcement."
Authority: A6 (unsupported types cannot be auto-enforced).
```

```
SUPPORTED_ENFORCEMENT_TYPES = {"video", "image"}
```

Adding a content type to the supported set is a **Lightweight ADR**
within the policy domain.

### §6.7 Rule firing semantics

A rule "fires" iff its condition is True at evaluation time AND its
effect mutates `current_action` or `bounds.upper`. Rules that match
condition but produce no effect (e.g., R5 condition met but
current_action is already < RESTRICT) are NOT recorded in
`triggered_rules`.

Rationale (v3 §2): a rule appearing in `triggered_rules` but
without effect is misleading audit. Conditions are tightened to
prevent silent no-op firing.

---

## §7 Evidence Derivation

`EvidenceStrength` is derived **internally** by the engine from
`match_found`, `signal_source`, `similarity`, `has_multiple_matches`,
and `distinct_matched_owner_count`. Per v2 §3, the caller does NOT
precompute it.

### §7.1 EvidenceStrength enum

| Value | Semantic |
|---|---|
| `NONE` | no match found |
| `WEAK` | match found; below moderate thresholds |
| `MODERATE` | base evidence sufficient for RESTRICT |
| `STRONG` | base evidence sufficient for TAKEDOWN |

### §7.2 Derivation algorithm (v5 §3)

Two-step derivation:

```
Step 1 — base strength (no multi-match adjustment):

  IF NOT match_found:
      → NONE
  ELIF signal_source == "fusion" AND similarity >= 0.70:
      → STRONG
  ELIF (signal_source == "fusion" AND similarity >= 0.40)
       OR (signal_source != "fusion" AND similarity >= 0.70):
      → MODERATE
  ELSE:
      → WEAK


Step 2 — multi-match adjustment (only if has_multiple_matches AND base != NONE):

  IF distinct_matched_owner_count >= 2 AND similarity >= 0.70:
      → STRONG
        (independent rights-holders + high similarity = direct STRONG)

  ELIF base == WEAK:
      → MODERATE
        (multi-match upgrades weak evidence by one level)

  ELSE:
      → no change
        (MODERATE stays MODERATE under single-owner multi-match;
         STRONG stays STRONG.)
```

Threshold constants:
- `FUSION_STRONG_SIMILARITY = 0.70`
- `FUSION_MODERATE_SIMILARITY = 0.40`
- `SINGLE_MODERATE_SIMILARITY = 0.70`
- `MULTI_MATCH_DISTINCT_SIMILARITY = 0.70`

### §7.3 STRONG reachability

STRONG is the maximum and is reachable via two paths only:

1. **Fusion + high similarity** — `signal_source == "fusion" AND similarity >= 0.70`
2. **Multi-owner + high similarity** — `has_multiple_matches AND distinct_matched_owner_count >= 2 AND similarity >= 0.70`

Both paths require `similarity >= 0.70`. STRONG always implies
high-similarity evidence. This guarantees R3's evidence-gate
semantics are intact (§8.2.2).

### §7.4 Internality

The derived value is consumed by R2 (§8.2.1) and R3 (§8.2.2). It
does NOT appear in PolicyResult — the audit signal is the rule_id
in `triggered_rules` rather than the strength value itself
(per v2 §3 reconciliation).

---

## §8 REFINE: Risk Control Rules (R1–R5)

Five rules across two phases. Phase 5 = upgrades; Phase 6 =
downgrades. Phase 6 has final authority.

### §8.1 Phase 5 — Upgrades (R1, R4)

Both rules clamp via `FeasibleBounds`:

```
desired   = <rule's target action>
ceiling   = bounds.clamp(GLOBAL_MAX_UPGRADE)   # = bounds.upper
effective = min_action(desired, ceiling)
IF severity(current_action) < severity(effective):
    current_action ← effective
    record rule_id in triggered_rules + upgrades_applied
```

#### §8.1.1 R1 — GRAY_ZONE

```
ID:        R1_GRAY_ZONE
Condition: ConfidenceReasonCode.GRAY_ZONE in confidence.triggered_conditions
           AND 0.30 <= confidence.composite < 0.75
           AND match_found
           AND severity(current_action) < severity(REVIEW)
Desired:   REVIEW
Reason:    "Similarity in adversarial gray zone with confidence {composite:.2f} < 0.75."
```

The lower bound `composite >= 0.30` (v3 §2) prevents R1 from
firing-then-being-clamped silently when conf < 0.30 (S4 already
caps at FLAG in that case).

#### §8.1.2 R4 — VERIFIED_OWNER

```
ID:        R4_VERIFIED_OWNER
Condition: trust_owner_tier == "verified"
           AND similarity >= 0.85
           AND tier_at_least(confidence_tier, MEDIUM)
           AND current_action == REVIEW
Desired:   RESTRICT
Reason:    "Verified rights-holder with high similarity. Upgrading to RESTRICT."
```

Intent: a verified rights-holder match at ≥ 0.85 similarity with
MEDIUM+ confidence should not sit in a review queue; upgrade to
RESTRICT. R4 cannot upgrade past RESTRICT (per `GLOBAL_MAX_UPGRADE`
clamp).

R4 fires correctly under the semantic FeasibleBounds model when no
prior safety rule (S5 / S4 / S3) has tightened the ceiling below
RESTRICT — see §14.3 for the reconciliation history.

### §8.2 Phase 6 — Downgrades (R2, R3, R5)

Apply in order. Each rule may downgrade `current_action`. If any
fires, the result is final regardless of Phase 5 upgrades.

#### §8.2.1 R2 — FIRST_OFFENSE

```
ID:        R2_FIRST_OFFENSE
Condition: is_first_offense
           AND severity(current_action) >= severity(RESTRICT)
           AND evidence_strength != STRONG          # v4 §5
Effect:    current_action ← REVIEW
Reason:    "First offense for this uploader. Downgrading to human review."
```

The STRONG-evidence exception (v4 §5): if evidence is STRONG
(fusion + high similarity, OR multi-owner + high similarity), R2
does NOT fire. Strong evidence overrides first-offense leniency.

#### §8.2.2 R3 — EVIDENCE_GATE

```
ID:        R3_EVIDENCE_GATE
Condition: current_action == TAKEDOWN
           AND evidence_strength != STRONG
Effect:    current_action ← RESTRICT
Reason:    "TAKEDOWN requires STRONG evidence. Current: {evidence_strength}."
Authority: A3 derivative + spec evidence dimension.
```

R3 ensures TAKEDOWN requires both HIGH confidence (S2 + ASSERT)
AND STRONG evidence (R3).

#### §8.2.3 R5 — DISPUTE_CAUTION

```
ID:        R5_DISPUTE_CAUTION
Condition: has_prior_disputes
           AND "overturned" in normalized_disputes      # see §11.2
           AND severity(current_action) >= severity(RESTRICT)
Effect:    current_action ← REVIEW
Reason:    "Uploader has overturned disputes. Applying caution. Human review required."
```

Rationale: previous false positives against this uploader make
automated enforcement risky. Per v4 §4, `normalized_disputes`
preserves chronological order — supports future temporal rules.

### §8.3 Phase 5 + Phase 6 conflict resolution

Per v2 §5: if any Phase 6 downgrade fires, the result is FINAL,
regardless of Phase 5 upgrades.

```
phase5_action = current_action after Phase 5
phase6_action = current_action after Phase 6
final_action  = min_action(phase5_action, phase6_action)
              = phase6_action  (Phase 6 downgrades never raise)
```

The `min_action` form is a safety equivalence: it works whether or
not any Phase 6 downgrade fires.

### §8.4 Rule priority for primary_reason selection

`PolicyResult.primary_reason` is the reason string of the
highest-priority firing rule:

| Rule | Priority |
|---|---|
| S1 | 100 |
| INVARIANT_TAKEDOWN_GUARD | 95 |
| S2 | 90 |
| S3 | 80 |
| S4 | 70 |
| S5 | 60 |
| R2 | 50 |
| R3 | 45 |
| R5 | 40 |
| R4 | 30 |
| R1 | 20 |

Phase-6 downgrades outrank Phase-5 upgrades because Phase 6 is
final authority. Safety rules outrank both because they reflect
axiom-level constraints. If no rule fires, `primary_reason` is a
formatted base-matrix description.

---

## §9 ASSERT: Terminal Invariant Guard

After all phases complete, the engine asserts the A3 invariant
(per v5 §1):

```
IF confidence_tier != HIGH AND final_action == TAKEDOWN:
    final_action ← RESTRICT
    triggered_rules.append("INVARIANT_TAKEDOWN_GUARD")
    downgrades_applied.append("INVARIANT_TAKEDOWN_GUARD")
    logger.error(
        "PolicyEngine invariant violated: TAKEDOWN at confidence_tier=%s",
        confidence_tier.value, ...
    )
```

This is **defense-in-depth** for S2. If S2 worked correctly, the
guard NEVER fires. If the guard fires, it indicates a bug in phase
logic — the ERROR-level log surfaces this immediately for
operability.

The guard's appearance in `triggered_rules` is itself an audit
signal: any production occurrence of `INVARIANT_TAKEDOWN_GUARD` is
a **P0** per `docs/constitution/GOVERNANCE.md` §5.

The guard is the engine's only permitted side effect (§12.1). The
log does not affect the function's return value, so it does not
violate A5.

---

## §10 Output: PolicyResult

```python
class PolicyResult(BaseModel):
    final_action: PolicyAction
    action_trace: ActionTrace
    triggered_rules: list[str]
    primary_reason: str
    risk_band: RiskBand
    confidence_tier: ConfidenceTier
    confidence_composite: float
    policy_version: str
    decision_config_version: str
    confidence_config_version: str
    rules_checked_count: int       # MUST equal EXPECTED_RULE_COUNT (10)
    evaluation_hash: str           # SHA-256 over 12 fields, first 16 hex chars

class ActionTrace(BaseModel):
    base_action: PolicyAction              # from §5 PROPOSE
    after_safety: PolicyAction             # after Phase 4 (BOUND complete)
    after_risk_control: PolicyAction       # after Phase 6 (= final_action absent ASSERT)
    upgrades_applied: list[str]            # Phase-5 rule_ids that fired
    downgrades_applied: list[str]          # Phase-6 + S* + INVARIANT guard if applicable
```

### §10.1 evaluation_hash composition (v5 §2)

12 fields, deterministic concatenation:

```
sha256(
    "|".join([
        str(sorted(triggered_rules)),         #  1
        final_action.value,                    #  2
        policy_version,                        #  3
        str(rules_checked_count),              #  4
        risk_band.value,                       #  5
        confidence_tier.value,                 #  6
        f"{confidence_composite:.4f}",         #  7  — 4dp for float stability
        signal_source,                         #  8  — normalized
        f"{similarity:.4f}",                   #  9  — 4dp
        str(is_first_offense),                 # 10  — "True"/"False"
        str(has_prior_disputes),               # 11
        str(distinct_matched_owner_count),     # 12
    ])
).hexdigest()[:16]
```

Determinism notes:
- `sorted(triggered_rules)` provides canonical ordering regardless
  of rule firing order.
- Floats rounded to 4 decimal places eliminate platform-specific
  representation drift.
- Booleans serialise as `"True"` / `"False"` via `str()`.

The hash is **tamper detection**: a stored PolicyResult can be
re-validated by recomputing the hash from its inputs and comparing.
Hash mismatch is a **P0** governance violation
(`docs/constitution/GOVERNANCE.md` §5).

### §10.2 rules_checked_count assertion

`rules_checked_count` MUST equal `EXPECTED_RULE_COUNT = 10` at the
end of every `evaluate_policy` call. The engine raises
`AssertionError` on mismatch — this is a programmer-error guard:
adding or removing a rule requires updating `EXPECTED_RULE_COUNT`
in lockstep.

### §10.3 Cross-references to A4

PolicyResult is the policy-domain contribution to the A4 audit
record. The A4 minimum schema additionally requires `input_id`,
`matched_id`, `similarity`, `risk_score`, `timestamp`, and
`upstream_event_ref` — these come from the platform / storage layer
that wraps the PolicyResult. PolicyResult itself provides:

- `policy_lineage_ref` — `triggered_rules` + `evaluation_hash`
- `engine_lineage` — `policy_version` + `decision_config_version` +
  `confidence_config_version`

both required by A4.

---

## §11 Normalization

Two PolicyContext fields are normalized at engine entry.

### §11.1 signal_source (v5 §4)

```
def _normalize_signal_source(raw: str) -> str:
    return raw.strip().lower()

VALID_SIGNAL_SOURCES = {"fingerprint", "embedding", "fusion"}
DEFAULT_SIGNAL_SOURCE = "fingerprint"
```

After normalization, if the value is NOT in `VALID_SIGNAL_SOURCES`,
the engine substitutes `DEFAULT_SIGNAL_SOURCE = "fingerprint"`.
Rationale: `fingerprint` is the weakest single-source signal —
defaulting to it on invalid input is conservative and cannot
accidentally enable STRONG evidence.

### §11.2 prior_dispute_outcomes (v3 §5 + v4 §4)

```
VALID_DISPUTE_OUTCOMES = {"upheld", "overturned", "withdrawn", "pending"}
MAX_DISPUTE_HISTORY = 5

def _normalize_dispute_outcomes(raw: list[str]) -> list[str]:
    cleaned = [s.strip().lower() for s in raw if s]
    cleaned = [s for s in cleaned if s in VALID_DISPUTE_OUTCOMES]
    if len(cleaned) > MAX_DISPUTE_HISTORY:
        cleaned = cleaned[-MAX_DISPUTE_HISTORY:]
    return cleaned          # chronological order preserved (v4 §4)
```

R5 reads `"overturned" in normalized_disputes`. The chronological
order is preserved (the v3 alphabetical sort was reverted in v4)
to support future temporal rules like `disputes[-1] == "overturned"`.

---

## §12 Determinism Guarantees

The engine is a pure deterministic function per **A5
(DETERMINISTIC REPLAY)**.

- No randomness.
- No time-of-day reads.
- No environment-variable reads inside engine code.
- No external state.
- All inputs flow through `evaluate_policy(decision, confidence, context)`.

### §12.1 I/O envelope

The single permitted side effect is the **ERROR-level log** when
the terminal invariant guard fires (§9). This log:

- does NOT affect the function's return value;
- exists for operability — silent invariant violations would be
  unrecoverable;
- is captured by standard observability infrastructure
  (`docs/specs/observability.md` *(planned)*).

No other I/O is permitted. Adding I/O is a **P0** violation.

### §12.2 Replay attribution

Given an A4 audit record, replay reconstructs:

1. Load `decision_config_version` of the threshold config used.
2. Load `confidence_config_version` of the confidence config used.
3. Load `policy_version` of the engine binary.
4. Re-run `evaluate_policy(...)` with reconstructed inputs.
5. Recompute `evaluation_hash` and compare to the stored value.

Hash mismatch indicates either a non-deterministic code path (P0
bug) or an audit record that has been tampered with (P0 evidence
violation per A7).

---

## §13 Invariants

Properties that MUST hold for any input combination.

| # | Invariant | Enforced by | Axiom |
|---|---|---|---|
| I1 | `match_found == False → final_action == ALLOW` | S1 (Phase 1) | A2 |
| I2 | `confidence_tier != HIGH → final_action != TAKEDOWN` | S2 (Phase 2) + ASSERT (terminal guard) | A3 |
| I3 | `confidence_tier == LOW AND current == RESTRICT → action ≤ REVIEW` | S3 (matrix-evolution guard) | A3 derivative |
| I4 | `confidence.composite < 0.30 → action ≤ FLAG` | S4 (Phase 3) | A4 / A6 |
| I5 | `evidence_strength != STRONG → action != TAKEDOWN` | R3 (Phase 6) | A3 derivative |
| I6 | `is_first_offense AND evidence != STRONG → action ≤ REVIEW` | R2 (Phase 6) | spec |
| I7 | Phase 5 cannot exceed `bounds.upper` | `FeasibleBounds.clamp` | spec |
| I8 | Phase 6 downgrade wins over Phase 5 upgrade | `min_action(phase5, phase6)` | spec |
| I9 | `triggered_rules` contains only rules with effect | tightened conditions per v3 §2 | spec |
| I10 | `rules_checked_count == 10` always | end-of-function assertion | A4 derivative |
| I11 | `evaluation_hash` reproducible from same inputs | A5 + 4dp float rounding | A5 |
| I12 | `final_action == action_trace.after_risk_control` (absent ASSERT) | structural | spec |

### §13.1 Cross-axiom mapping

- **A2 (MATCH PREREQUISITE)** → I1 (S1 forces ALLOW).
- **A3 (CONFIDENCE-GATED ENFORCEMENT)** → I2 (S2 + ASSERT), I3
  (S3), I5 (R3).
- **A4 (AUDIT COMPLETENESS WITH PROVENANCE)** → I10 (rules count),
  I11 (evaluation_hash), §10.3 (provenance fields).
- **A5 (DETERMINISTIC REPLAY)** → §12.
- **A6 (HUMAN REVIEW AUTHORITY)** → reversal lives at platform
  layer; the engine emits the decision, A6 governs reversal of
  that decision (`docs/security/enforcement_audit.md` *(planned)*).
- **A7 (EVIDENCE PRESERVATION)** → §10.3 (audit record content);
  storage layer responsibility.

### §13.2 Test coverage

The current implementation passes **21 smoke scenarios** in
`backend/_smoke_policy.py` covering each invariant. The canonical
invariant test catalogue is `docs/testing/INVARIANT_TESTS.md`
*(planned, Phase 2D)*. Promotion to a proper test suite happens in
Phase 5 of the migration.

---

## §14 Reconciliation history

Five spec versions (v1 → v5) plus two zero-byte placeholders are
superseded by this document. Major reconciliations:

### §14.1 PolicyContext deduplication (v1 → v2)

v1 had `risk_band`, `confidence_tier`, `match_found`, `similarity`,
`confidence_reasons` in PolicyContext. v2 §1 stripped these because
they exist in upstream objects (DecisionOutput / ConfidenceBreakdown).
Adopted: PolicyContext is operational + evidence only.

### §14.2 S4 semantics correction (v1 → v2)

v1 said "confidence < 0.30 → upgrade to REVIEW". v2 §4 corrected
to "cap at FLAG" — uncertain cases should not flood the human
review queue. Adopted.

### §14.3 Phase 5 ceiling — the major reconciliation

v3/v4 wrote `phase5_ceiling = phase4_action` literally. The
v2 §5 worked example explicitly shows R4 firing
(`After Phase 4: REVIEW. Phase 5: R4 fires → upgrade to RESTRICT`).
Under literal reading, R1 and R4 are mathematically unreachable —
R4 needs `current == REVIEW`, which forces `phase4 == REVIEW`,
which collapses the ceiling to REVIEW, which clamps R4's effective
target back to REVIEW = current, blocking the upgrade. R1
collapses similarly. v3 §3's `EXPECTED_RULE_COUNT = 10` mandates
all rules be operationally meaningful — incompatible with R1/R4
deadness.

This document adopts the **semantic reading**: `bounds.upper` is
the lowest cap target across actually-fired safety rules,
defaulting to `GLOBAL_MAX_UPGRADE = RESTRICT` when no safety rule
fires. The semantic reading is the unique interpretation that
satisfies (a) the v2 worked example, (b) every v3 §1 / v4 §1
trace, and (c) the v3 completeness contract.

The literal interpretation is recorded as a rejected design
**REJ-003** (or equivalent — backfill in Phase 3,
`.claude/archive/rejected/`).

### §14.4 Evidence stabilization (v3 → v5)

- v3 §4 introduced multi-match evidence with direct STRONG at
  `similarity >= 0.60`.
- v4 §3 narrowed: direct STRONG only with `distinct_owner_count >= 2`.
- v5 §3 stabilized: MODERATE does NOT promote to STRONG via
  single-owner multi-match (preventing R3-bypass).

Adopted v5 final form.

### §14.5 R2 STRONG-evidence exception (v4 §5)

v1's R2 unconditionally downgraded RESTRICT/TAKEDOWN to REVIEW for
first-time offenders. v4 §5 added the STRONG-evidence exception:
strong evidence overrides first-offense leniency. Adopted.

### §14.6 evaluation_hash growth (v3 → v5)

- v3 §3 — 4 fields (triggered_rules, final_action, policy_version,
  rules_checked_count).
- v4 §2 — added `risk_band`, `confidence_tier`, `confidence_composite`
  (7 fields).
- v5 §2 — added `signal_source`, `similarity`, `is_first_offense`,
  `has_prior_disputes`, `distinct_matched_owner_count` (12 fields).

Adopted v5.

### §14.7 Terminal invariant guard (v5 §1)

v5 added a post-phase guard for `confidence_tier != HIGH AND
TAKEDOWN`. Defense-in-depth backup for S2. Adopted (§9).

### §14.8 S2 / S3 reclassification

S2 and S3 are unreachable from the current base matrix (see §6.3,
§6.4). They remain in the rule set as **matrix-evolution guards**:
they activate immediately if a future matrix change introduces
TAKEDOWN at non-HIGH or RESTRICT at LOW. The startup matrix
validator (§5.2) prevents accidental drift. This canonical doc
records the classification.

### §14.9 3-level → 5-action transition

The legacy 3-level enforcement model from
`.claude/rules/enforcement.md` is REMOVED (per
`docs/state/STATE.md`). Mapping in §2.3.

### §14.10 Documentation lineage

| Source | Status | Location |
|---|---|---|
| `policy_engine_spec_v1.md` | superseded; working draft retained | `.claude/memory/` (untracked) |
| `policy_engine_spec_v2.md` | superseded | `.claude/memory/` (untracked) |
| `policy_engine_spec_v3.md` | superseded | `.claude/memory/` (untracked) |
| `policy_engine_spec_v4.md` | superseded | `.claude/memory/` (untracked) |
| `policy_engine_spec_v5.md` | superseded | `.claude/memory/` (untracked) |
| `docs/architecture/POLICY_ENGINE_CANONICAL_SEMANTICS.md` | redirected; zero-byte | to be retired in Phase-2 closeout |
| `docs/architecture/POLICY_INVARIANTS.md` | redirected; zero-byte | to be retired in Phase-2 closeout |

Per the append-only migration constraint, none of the above sources
is deleted in this batch. The two zero-byte placeholders are
candidates for removal in Phase-2 closeout once all canonical specs
are landed.

---

## §15 Open questions / Future work

Documented for visibility; not commitments.

- **R6_REPEAT_OFFENDER** (v2 §7 reserved signal): use
  `recent_violations_count` to skip first-offense leniency for
  repeat abusers. Lightweight ADR + threshold tuning.
- **PolicyAction LOCKED status**: once enforcement integrations
  stabilize (api, pipeline, security), promote the enum to LOCKED
  via Constitutional ADR — independent of this spec's stability.
- **Audit hash chain**: extend `evaluation_hash` to include the
  prior decision's hash for the same uploader/asset, enabling
  tamper-evident audit trails (per A7 strengthening).
- **Region/jurisdiction overrides**: per-jurisdiction action
  modifications (e.g., DSA-driven REVIEW thresholds in EU). Likely
  a future Phase-5 rule R7.
- **Multi-tenant policy variation**: same engine, per-tenant
  thresholds. Today the engine is single-tenant by config_version.

---

## §16 Versioning and Change Process

This spec is **EVOLVING** per
`docs/constitution/GOVERNANCE.md` §8. Compatibility expectations
are low — consumers should expect change at each minor bump.

The `policy_version` constant in
`backend/app/engines/policy_engine.py` MUST be bumped in lockstep
with this spec's `version:` field. Mismatch is a **P1**
governance violation per
`docs/constitution/GOVERNANCE.md` §5.

### §16.1 Graduation to STABLE

This spec graduates from EVOLVING to STABLE via a **Standard ADR**
when:

1. The rule set has been unchanged for at least one minor revision
   cycle.
2. No production incidents have implicated policy logic in 90 days.
3. Consumer integrations (api, pipeline, security) report
   stability.
4. The invariant test suite (`docs/testing/INVARIANT_TESTS.md`)
   covers all 12 invariants in §13.

Architect approves graduation.

### §16.2 Demoting from STABLE

If a STABLE spec needs material change that breaks the STABLE
contract, a Standard ADR may demote it back to EVOLVING per
GOVERNANCE.md §8. The ADR MUST justify why EVOLVING is preferred
over a deprecation cycle.

---

## §17 Cross-references

- **Axioms** (`../constitution/AXIOMS.md`): A2, A3, A4, A5, A6, A7.
- **Constitutional governance**
  (`../constitution/GOVERNANCE.md`): §1 (tier hierarchy), §3 (ADR
  tiers), §5 (severity model), §7 (EGM), §8 (stability levels).
- **Domain ownership** (`../governance/DOMAINS.md`): policy
  domain.
- **Architecture state** (`../state/STATE.md`): PolicyEngine ACTIVE
  v1.0; 3-level model REMOVED.
- **Implementation**:
  - `backend/app/engines/policy_engine.py`
  - `backend/app/models/policy_models.py`
- **Smoke tests**: `backend/_smoke_policy.py` (21 scenarios).
  Canonical test surface: `../testing/INVARIANT_TESTS.md` *(planned)*.
- **Upstream contracts**:
  - `./decision_engine.md` *(planned)* — DecisionOutput producer
  - `./confidence_engine.md` *(planned)* — ConfidenceBreakdown producer
- **Downstream contracts**:
  - `./api_contracts.md` *(planned)* — PolicyAction consumer surface
  - `./eventing.md` *(planned)* — pipeline integration
  - `../security/enforcement_audit.md` *(planned)* — audit + dispute layer
- **Working drafts (untracked)**:
  - `.claude/memory/policy_engine_spec_v1.md` through `_spec_v5.md`
- **Rejected designs** *(planned, Phase 3)*:
  - `.claude/archive/rejected/REJ-001..REJ-004` — including the
    literal `phase5_ceiling = phase4_action` interpretation
    (anticipated REJ-003).
