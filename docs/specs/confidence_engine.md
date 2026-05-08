---
authority: SPEC
domain: confidence
status: ACTIVE
version: 1.0
stability: EVOLVING
owner: confidence (interim: architect)
supersedes:
  - .claude/memory/confidence_engine_spec.md (v3 final merged; working draft, untracked)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; canonical-spec ratification; to be backfilled)
---

# Confidence Engine — Canonical Specification

The ConfidenceEngine computes the **certainty of a risk assessment**,
not the risk itself. It maps a `ConfidenceInput` to a
`ConfidenceBreakdown` containing three component scores
(agreement, completeness, uncertainty), a composite confidence value,
a tier classification, and a list of triggered reason codes for audit
provenance.

This document is the canonical specification — Tier 2 (SPEC) — and
supersedes `.claude/memory/confidence_engine_spec.md` (v3 final
merged). The implementation is `backend/app/engines/confidence_engine.py`
and `backend/app/models/confidence_models.py`.

The engine is consumed by the PolicyEngine (DECISION phase per A1)
and provides the confidence dimension that gates enforcement under
A3 (CONFIDENCE-GATED ENFORCEMENT). See §17 cross-references.

---

## §1 Purpose and Authority

### §1.1 Purpose

The ConfidenceEngine answers a single question: **how certain are we
that the risk assessment is correct?**

It is structurally distinct from the DecisionEngine (which computes
the risk score). DecisionEngine says *how dangerous is this content?*;
ConfidenceEngine says *how confident are we in that assessment?* The
two outputs together inform the PolicyEngine's action selection
(see `./policy_engine.md` §5).

It is a **pure deterministic function** per A5. No randomness. No
I/O. No external state. All policy-level conditions (S1–S5, U1–U5,
gray-zone flag) are derived **inside** the engine from input
primitives plus configuration; callers MUST NOT precompute and pass
in policy flags.

### §1.2 Position in the pipeline

Per A1 (PIPELINE PHASE INTEGRITY), confidence is computed during the
**EVALUATION** phase, alongside risk:

```
INGESTION → ANALYSIS → EVALUATION → DECISION → ENFORCEMENT
                          ↑                ↑
                          confidence       PolicyEngine consumes
                          is computed      tier + composite
                          here
```

### §1.3 Authority

This document is **TIER 2 (SPEC)** per
`docs/constitution/GOVERNANCE.md` §1. Owned by the **confidence**
domain (`docs/governance/DOMAINS.md`). Modification:

| Change | ADR tier |
|---|---|
| Threshold tweak (e.g., trust cap 0.20 → 0.22) | Lightweight |
| New uncertainty term (U6) | Standard |
| Tier boundary shift (e.g., MEDIUM/HIGH boundary 0.70 → 0.65) | Standard |
| Composite weight redistribution (e.g., 0.4/0.3/0.3 → 0.5/0.25/0.25) | Standard |
| New scoring dimension (additional component beyond agreement/completeness/uncertainty) | Constitutional (changes the model's structural shape) |
| Reason-code rename or removal | Standard (cross-domain consumer surface — policy / qa) |
| Reason-code addition | Lightweight |

### §1.4 Stability

**EVOLVING** per `docs/constitution/GOVERNANCE.md` §8. The
three-component model is structurally stable; thresholds and reason
codes may continue to refine via Lightweight or Standard ADRs.
Graduation to STABLE follows the same gates as `policy_engine.md`
§16.1.

The `ConfidenceTier` enum is a cross-domain consumer surface
(PolicyEngine reads it; qa tests assert against it). Promoting the
enum to LOCKED status is a future design goal independent of this
spec's stability.

---

## §2 The Composite Model

### §2.1 Three components

Confidence is a weighted sum of three independent components.

| Component | Range | Semantic |
|---|---|---|
| **agreement** | [0, 1] | Do the available signals agree on the risk assessment? Higher = signals concur. |
| **completeness** | [0, 1] | How many independent signals do we have? Higher = more corroboration. |
| **uncertainty** | [0, 0.50] | What known conditions reduce our trust in the assessment? Higher = more reasons for doubt. |

Each component answers a different question. They are **independent**
in the sense that one being high or low does not constrain the
others. The composite combines them into a single confidence value.

### §2.2 Composite formula

```
confidence = w_agreement   * agreement
           + w_completeness * completeness
           + w_uncertainty  * (1 - uncertainty)

confidence = clamp(confidence, 0.05, 1.0)
```

With default weights `0.4 / 0.3 / 0.3`. Note: the **complement**
form `(1 - uncertainty)` — uncertainty *reduces* confidence, so the
weighted contribution is its complement. High uncertainty → low
contribution to confidence.

The weights MUST sum to exactly 1.0. `ConfidenceConfig.__post_init__`
validates this.

### §2.3 Composite floor

Per spec §6, the composite is clamped to **[0.05, 1.0]** — never
below 0.05. This is defense-in-depth: smoothing math elsewhere in
this spec keeps composite ≥ 0.20 in normal operation, but the
explicit 0.05 floor protects against future config changes that
might push the value below.

> **Implementation note (C-CE-1):** the current implementation
> (`backend/app/engines/confidence_engine.py::_composite`) clamps to
> `[0, 1]` via `_clamp01`, not `[0.05, 1.0]`. The 0.05 lower bound
> is mathematically irrelevant given current smoothing constants
> (agreement floor 0.125 across 3 pairs; minimum composite ≈ 0.20),
> but the spec's explicit floor should be added for defense-in-depth.
> A Lightweight ADR will reconcile this.

### §2.4 Confidence tier classification

The composite is bucketed into three tiers consumed by the
PolicyEngine.

| Range | Tier |
|---|---|
| `[0.00, 0.40)` | `LOW` — insufficient for autonomous enforcement |
| `[0.40, 0.70)` | `MEDIUM` — acceptable with caveats; review-grade |
| `[0.70, 1.00]` | `HIGH` — suitable for autonomous enforcement (per A3) |

Tier boundaries are configured by `ConfidenceConfig.low_upper`
(default 0.40) and `ConfidenceConfig.medium_upper` (default 0.70).

Per `./policy_engine.md` §5, only `HIGH × HIGH` (risk × confidence)
proposes TAKEDOWN — A3's confidence gate is realised through this
tiering.

---

## §3 Execution Model

### §3.1 Stages

The engine executes in fixed sequence:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — Match check                                         │
│            If match_found == False → SHORT-CIRCUIT (§3.2)      │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2 — Agreement (§5)                                      │
│            Pair-selection rule + magnitude-aware scoring +     │
│            Laplace smoothing.                                  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3 — Completeness (§6)                                   │
│            Count signals present (S1–S5) ÷ 5.                  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4 — Uncertainty (§7)                                    │
│            U1 + min(U2+U3, trust_cap) + U4 + U5,               │
│            then clamped to global cap (0.50).                  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 5 — Composite assembly + Tier (§8)                      │
│            Weighted sum (§2.2) → clamp → tier lookup.          │
└─────────────────────────────────────────────────────────────────┘
```

Each stage is a pure function of its inputs and the config.

### §3.2 No-match short-circuit (special case)

When `match_found == False`, the engine produces a **fixed result**
and skips the per-stage computations:

```
agreement   = 0.50
completeness = 0.0
uncertainty  = 0.0    (no Ui fires; every Ui is gated by AND match_found)
composite    = 0.4 * 0.50 + 0.3 * 0 + 0.3 * (1 - 0)
             = 0.20 + 0   + 0.30
             = 0.50
tier         = MEDIUM
triggered_conditions = (NO_MATCH,)
```

The triple `(0.50, 0, 0)` is canonical: the engine cannot be
asked to evaluate an absent match. The 0.50 agreement reflects
maximal uncertainty in either direction (neither concur nor disagree).
Confidence 0.50 lands in the MEDIUM tier, but the PolicyEngine
short-circuits earlier via S1 (`match_found == False → ALLOW`), so
the no-match confidence value is never load-bearing for enforcement.

---

## §4 Inputs

### §4.1 ConfidenceInput

Primitives only — no precomputed policy flags. The engine derives
all policy-level conditions internally.

```python
@dataclass(frozen=True)
class ConfidenceInput:
    match_found: bool
    similarity: float                   # [0, 1]
    trust_owner: TrustState
    trust_uploader: TrustState
    observation_count: int
    signal_source: str                  # case-insensitive (§11)
    config_version: str = "v3"          # see §4.4
```

### §4.2 ConfidenceConfig

The engine reads thresholds and weights from a config object. Default
values match `confidence_engine_spec.md` v3.

| Field | Default | Used by |
|---|---|---|
| `w_agreement` | 0.40 | §2.2 composite |
| `w_completeness` | 0.30 | §2.2 composite |
| `w_uncertainty` | 0.30 | §2.2 composite |
| `laplace_numerator` | 0.5 | §5.4 smoothed aggregation |
| `laplace_denominator_offset` | 1.0 | §5.4 smoothed aggregation |
| `trust_cap_default` | 0.20 | §7.2 trust-cap composition |
| `trust_cap_gray_zone` | 0.25 | §7.2 trust-cap composition |
| `u1_value` | 0.25 | §7.1 U1 magnitude |
| `u2_value` | 0.15 | §7.1 U2 magnitude |
| `u3_value` | 0.10 | §7.1 U3 magnitude |
| `u4_value` | 0.05 | §7.1 U4 magnitude |
| `u5_value` | 0.05 | §7.1 U5 magnitude |
| `uncertainty_global_cap` | 0.50 | §7.3 global cap |
| `low_upper` | 0.40 | §2.4 tier boundary |
| `medium_upper` | 0.70 | §2.4 tier boundary |
| `s4_observation_threshold` | 3 | §6 / §7 (S4 + U4) |
| `fusion_signal_source` | `"FUSION"` | §6 / §7 (S5 + U5); compared case-insensitively |
| `gray_zone_lower` | 0.75 | §7.1 U1 lower bound |
| `gray_zone_upper` | 0.85 | §7.1 U1 upper bound |

`__post_init__` validates that `w_agreement + w_completeness +
w_uncertainty == 1.0` (within float tolerance).

### §4.3 TrustState semantics

```python
@dataclass(frozen=True)
class TrustState:
    trust_score: float       # [0, 1]
    is_default: bool         # registry hit / miss
```

**Authoritative rule (C-CE-8):** when `is_default == True`, callers
and the engine MUST IGNORE the `trust_score` value. The default
state is set by the trust registry's explicit signal (registry
miss); it is never inferred from a numeric value of `trust_score`.

When `is_default == True`:
- The corresponding S2/S3 signal is NOT counted as present (§6).
- The corresponding U2/U3 uncertainty is added (§7.1).
- The corresponding pair-selection rule applies (§5.2).

### §4.4 config_version semantics

`ConfidenceInput.config_version` (default `"v3"`) is a string
identifier for the config schema in use. It is NOT validated by the
engine; the caller is responsible for setting it consistent with
the actual `ConfidenceConfig` instance passed.

The output `ConfidenceBreakdown` does NOT carry the version (see
C-CE-9). The pipeline worker is responsible for threading the
version through to the audit record (per A4 `engine_lineage`). The
PolicyEngine reads `confidence_config_version` from
`DecisionOutput.input_snapshot.config_version` — i.e., the upstream
worker is expected to expose the confidence config version on the
decision-input snapshot. A future Lightweight ADR may move this
into the ConfidenceBreakdown directly.

---

## §5 Agreement (magnitude-aware)

### §5.1 Normalization

Three signals are normalised into a comparable [0, 1] space:

```
norm_similarity = clamp01(similarity)
norm_owner      = clamp01(trust_owner.trust_score)            # only when not default
norm_uploader   = clamp01(1 - trust_uploader.trust_score)     # only when not default
```

Note the **uploader inversion**: high uploader trust = low risk =
*disagrees with* high similarity. Inverting brings uploader into the
same direction as the other two signals (high values = high risk
contribution). This is the magnitude-aware convention from spec §3.

### §5.2 Pair-selection rules

Different pair sets are scored depending on which trust signals are
available. Per **C-CE-5**, these are renamed from the spec's R1–R5
to descriptive identifiers to avoid collision with PolicyEngine's
risk-control rules.

| Identifier | Condition | Pairs scored |
|---|---|---|
| `NO_MATCH` | `match_found == False` | none — short-circuit (§3.2) |
| `BOTH_DEFAULT` | both trusts default | none — empty pair list, smoothed alone |
| `OWNER_DEFAULT` | only owner is default | `(norm_similarity, norm_uploader)` |
| `UPLOADER_DEFAULT` | only uploader is default | `(norm_similarity, norm_owner)` |
| `ALL_PRESENT` | both trusts present | three pairs: `(sim, owner)`, `(sim, uploader)`, `(owner, uploader)` |

Mutual exclusion: exactly one rule applies per evaluation.

### §5.3 Pair scoring

For each pair (a, b) the **continuous, magnitude-aware** score is:

```
pair_score(a, b) = clamp01(1 - |a - b|)
```

This is NOT a binary "do they concur" — it's the absolute distance
inverted. Two signals at 0.95 and 0.85 score 0.90 (high agreement);
two at 0.95 and 0.10 score 0.15 (high disagreement).

The continuous form rejects two failed alternatives:
- **Binary concordance** (e.g., "agree if both > 0.5") is
  insensitive to the *magnitude* of agreement.
- **Multiplicative** (e.g., `a * b`) collapses to ~0 when any
  factor is low even if all signals genuinely agree on a low value.

### §5.4 Smoothed aggregation

The pair scores are aggregated with **Laplace smoothing**:

```
agreement = (sum(pair_scores) + 0.5) / (n_pairs + 1.0)
agreement = clamp01(agreement)
```

With defaults `laplace_numerator = 0.5`,
`laplace_denominator_offset = 1.0`. Smoothing serves two purposes:

1. **Empty case** (`BOTH_DEFAULT`): with no pairs, agreement =
   `0.5 / 1.0 = 0.50` — neutral, neither high nor low confidence in
   absence of comparable signals.
2. **Sparse case** (`OWNER_DEFAULT` or `UPLOADER_DEFAULT`): with one
   pair scoring 0, agreement = `(0 + 0.5) / (1 + 1) = 0.25`, not
   absolute zero. Smoothing prevents overconfidence on a single weak
   pair.

### §5.5 Reason-code emission

| Pair-selection case | Reason emitted |
|---|---|
| `NO_MATCH` | `NO_MATCH` (in the §3.2 short-circuit branch) |
| `BOTH_DEFAULT` | `BOTH_TRUSTS_DEFAULT` (canonical per **C-CE-3**) |
| `OWNER_DEFAULT` | `OWNER_TRUST_DEFAULT` |
| `UPLOADER_DEFAULT` | `UPLOADER_TRUST_DEFAULT` |
| `ALL_PRESENT` | (no reason emitted — normal case) |

These appear in `ConfidenceBreakdown.triggered_conditions`.

---

## §6 Completeness

The engine derives five signal-presence indicators (S1–S5) and
computes completeness as the fraction present:

```
completeness = signals_present / 5
```

| Signal | Condition | Source |
|---|---|---|
| **S1** | `match_found` | input |
| **S2** | `not trust_owner.is_default` | input (per §4.3 rule) |
| **S3** | `not trust_uploader.is_default` | input |
| **S4** | `observation_count >= s4_observation_threshold` (default 3) | input |
| **S5** | `signal_source == fusion_signal_source` (case-insensitive) | input |

S1–S5 are derived **inside the engine**. Callers MUST NOT precompute
and pass these flags — per spec §1 and the engine's docstring,
ConfidenceInput is primitives-only.

> **Naming note:** these S1–S5 are confidence-domain *signal-presence
> indicators*. They are unrelated to the PolicyEngine's S1–S5 *safety
> rules*. Both engines reuse the "S" prefix in their respective
> domains; cross-domain references should always qualify (e.g.,
> "policy.S2", "confidence.S2"). See §14.10 for vocabulary surface.

When `match_found == False`, the engine short-circuits to
`completeness = 0.0` (per §3.2); the per-signal computation is
skipped entirely.

---

## §7 Uncertainty

Uncertainty is the sum of penalties for known doubt-inducing
conditions, with internal capping to prevent double-counting and a
global ceiling to prevent total uncertainty from dominating the
composite.

### §7.1 Per-term magnitudes (U1–U5)

Every Ui is gated by `AND match_found` — uncertainty is zero when
no match exists.

| Term | Condition | Magnitude (default) | Reason emitted |
|---|---|---|---|
| **U1** | `gray_zone_lower ≤ similarity < gray_zone_upper` (default `0.75 ≤ sim < 0.85`) AND `match_found` | 0.25 | `GRAY_ZONE` |
| **U2** | `trust_owner.is_default` AND `match_found` | 0.15 | `OWNER_TRUST_DEFAULT` (already from §5.5; U2 does not duplicate) |
| **U3** | `trust_uploader.is_default` AND `match_found` | 0.10 | `UPLOADER_TRUST_DEFAULT` (same — no duplicate) |
| **U4** | `observation_count < s4_observation_threshold` (default `< 3`) AND `match_found` | 0.05 | `LOW_OBSERVATIONS` |
| **U5** | `signal_source != fusion_signal_source` AND `match_found` | 0.05 | `SINGLE_ENGINE_MATCH` |

### §7.2 Trust-cap composition

U2 and U3 are correlated (both relate to trust-registry coverage).
Their sum is capped to prevent double-counting:

```
trust_cap     = trust_cap_gray_zone if gray_zone else trust_cap_default
              # default 0.25 vs 0.20

trust_combined = min(U2 + U3, trust_cap)
```

When `gray_zone == True`, the cap widens from 0.20 to 0.25 to ensure
the **correlated trust+gray penalty is not discounted** by the cap.
This is intentional: gray-zone similarity *plus* missing trust
together represent a real epistemic state that should compound to
the larger cap.

### §7.3 Aggregate and global cap

```
raw_uncertainty = U1 + trust_combined + U4 + U5
uncertainty     = min(raw_uncertainty, uncertainty_global_cap)
                # default cap 0.50
uncertainty     = clamp01(uncertainty)
```

The global cap prevents uncertainty from exceeding 0.50, which
guarantees `(1 - uncertainty) >= 0.50` — uncertainty alone cannot
collapse confidence to zero.

When the raw value exceeds the cap, the reason
`UNCERTAINTY_GLOBAL_CAP` is emitted (canonical name per **C-CE-2**;
the spec's earlier `GLOBAL_CAP_APPLIED` is superseded terminology).

### §7.4 Trust-cap reason emission

When trust_combined was capped (i.e., `U2 + U3 > trust_cap`), the
reason `TRUST_CAP_APPLIED` is emitted. Additionally, when the
gray-zone widened cap was in effect (regardless of whether it bound),
`TRUST_CAP_GRAY_ZONE` is emitted (canonical per **C-CE-3**).

---

## §8 Composite assembly + Tier

```
composite = w_agreement   * agreement
          + w_completeness * completeness
          + w_uncertainty  * (1 - uncertainty)

composite = clamp(composite, 0.05, 1.0)         # canonical floor (§2.3)

tier = LOW    if composite < low_upper           # default 0.40
       MEDIUM if composite < medium_upper        # default 0.70
       HIGH   otherwise                          # [0.70, 1.00]
```

The result is wrapped in `ConfidenceBreakdown` (§10).

---

## §9 Reason codes

Reason codes provide **provenance** for the composite confidence —
they identify which conditions fired during evaluation. Consumed by
the PolicyEngine (e.g., `R1_GRAY_ZONE` checks for
`GRAY_ZONE` in `triggered_conditions`) and by audit replay.

### §9.1 Canonical reason codes

Listed alphabetically. Each is emitted at most once per evaluation.

| Code | Source | Emission rule |
|---|---|---|
| `BOTH_TRUSTS_DEFAULT` | §5.2 | both trusts default → empty pair set |
| `GRAY_ZONE` | §7.1 U1 | `gray_zone_lower ≤ similarity < gray_zone_upper` AND `match_found` |
| `LOW_OBSERVATIONS` | §7.1 U4 | `observation_count < s4_observation_threshold` AND `match_found` |
| `NO_MATCH` | §3.2 | `match_found == False` (short-circuit branch only) |
| `OWNER_TRUST_DEFAULT` | §5.2 / §7.1 U2 | `trust_owner.is_default` AND `match_found` |
| `SINGLE_ENGINE_MATCH` | §7.1 U5 | `signal_source != fusion_signal_source` AND `match_found` |
| `TRUST_CAP_APPLIED` | §7.4 | `U2 + U3` exceeded `trust_cap` |
| `TRUST_CAP_GRAY_ZONE` | §7.4 | gray-zone widened trust cap was in effect |
| `UNCERTAINTY_GLOBAL_CAP` | §7.3 | raw uncertainty exceeded `uncertainty_global_cap` |
| `UPLOADER_TRUST_DEFAULT` | §5.2 / §7.1 U3 | `trust_uploader.is_default` AND `match_found` |

Per **C-CE-2**, `UNCERTAINTY_GLOBAL_CAP` is canonical; the spec's
earlier `GLOBAL_CAP_APPLIED` name is superseded.

Per **C-CE-3**, `BOTH_TRUSTS_DEFAULT` and `TRUST_CAP_GRAY_ZONE` are
canonical extensions over the v3 spec list.

### §9.2 Deprecated reason codes

Per **C-CE-4**, two enum values exist for backward compatibility but
are not emitted by current engine logic:

| Code | Status | Notes |
|---|---|---|
| `INPUT_VALIDATION_FAILED` | DEPRECATED | reserved for future input-validation pipeline; never emitted |
| `INPUT_QUALITY_LOW` | DEPRECATED | reserved for future quality-gate signal; never emitted |

A future **Standard ADR** may remove these from the enum.

### §9.3 Ordering

`triggered_conditions` is a `Sequence[ConfidenceReasonCode]`. The
**emission order** within the engine is deterministic but is NOT
required to be sorted — the qa surface and replay logic SHOULD
sort by `.value` before comparing reason-code sets across runs.

---

## §10 Output: ConfidenceBreakdown

```python
@dataclass(frozen=True)
class ConfidenceBreakdown:
    agreement: float                                  # [0, 1]
    completeness: float                               # [0, 1]
    uncertainty: float                                # [0, 0.50]
    composite: float                                  # [0.05, 1.00]
    tier: ConfidenceTier                              # LOW / MEDIUM / HIGH
    triggered_conditions: Sequence[ConfidenceReasonCode]
                                                       # provenance per §9
```

### §10.1 Cross-references to A4

ConfidenceBreakdown is the confidence-domain contribution to the A4
audit record. Per A4 the audit must carry `engine_lineage` including
the confidence config version. **C-CE-9** notes that the breakdown
itself does NOT carry `config_version`; the pipeline worker threads
it through. The PolicyEngine reads
`confidence_config_version` from
`DecisionOutput.input_snapshot.config_version` (a Tier-2 boundary
contract).

### §10.2 ConfidenceTier authority

Per **C-CE-7**, this spec is the canonical authority for the
`ConfidenceTier` enum. The values `LOW / MEDIUM / HIGH` and the
boundaries `0.40` and `0.70` are owned by the confidence domain;
PolicyEngine consumes them per `./policy_engine.md` §5 and
`docs/governance/DOMAINS.md` cross-domain components.

A boundary change (e.g., `MEDIUM_UPPER 0.70 → 0.65`) is a Standard
ADR because it ripples into PolicyEngine's base matrix.

---

## §11 Normalization

Two inputs are normalised at engine entry. Normalisation is
deterministic and is part of the engine's purity contract.

### §11.1 signal_source

Per **C-CE-6**, the engine accepts `signal_source` in any casing
and with leading/trailing whitespace. Comparison against
`fusion_signal_source` (default `"FUSION"`) is performed via a
case-insensitive helper:

```python
def _norm_source(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()
```

The spec §2's enumeration `(FINGERPRINT, EMBEDDING, FUSION)` is
nominally uppercase, but the engine accepts lowercase, mixed-case,
or whitespace-padded forms transparently.

### §11.2 Numeric clamps

All numeric inputs that should be in `[0, 1]` (similarity,
trust_score) are clamped via `_clamp01` at every read site. NaN and
None are coerced to 0.0 (via `_safe`). Negative or > 1 inputs are
clamped to the bound rather than rejected — the engine never
raises on numeric input shape.

This is intentional: an upstream bug should not crash the engine in
the EVALUATION phase. Numeric anomalies become explicit via the
audit record (e.g., similarity = 0.0 with match_found = True is
visible in PolicyResult and can be flagged by qa).

---

## §12 Determinism Guarantees

The engine is a pure deterministic function per **A5
(DETERMINISTIC REPLAY)**.

- No randomness.
- No time-of-day reads.
- No environment-variable reads inside engine code.
- No external state.
- All inputs flow through `compute_confidence(input, config)`.

### §12.1 I/O envelope

**ZERO** I/O. No logging, no metrics, no observability calls inside
engine code. Operability hooks attach at the call site (the
pipeline worker), not in the engine.

This is stricter than PolicyEngine, which permits an ERROR-level log
on terminal-invariant violation (`./policy_engine.md` §12.1). The
ConfidenceEngine has no such escape hatch because it has no
analogous invariant violation: confidence computation cannot
"violate" itself in a way that a runtime log would diagnose.

### §12.2 Replay attribution

Given an A4 audit record's confidence portion (composite, tier,
triggered_conditions, config_version), replay reconstructs:

1. Load `ConfidenceConfig` at `confidence_config_version`.
2. Reconstruct `ConfidenceInput` from upstream signals
   (similarity, trust states, observation_count, signal_source).
3. Re-run `compute_confidence(input, config)`.
4. Compare `composite` (4dp tolerance), `tier`, sorted
   `triggered_conditions`.

Mismatch = either non-deterministic code path (P0 bug) or audit
tampering (P0 evidence violation under A7).

---

## §13 Invariants

Properties that MUST hold for any input combination.

| # | Invariant | Enforced by | Axiom / spec |
|---|---|---|---|
| CI1 | `match_found == False → composite == 0.50 AND tier == MEDIUM` | §3.2 short-circuit | spec §8 |
| CI2 | `agreement ∈ [0, 1]` | §5.4 + clamp | spec |
| CI3 | `completeness ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}` (5 signals → 6 levels) | §6 | spec |
| CI4 | `uncertainty ∈ [0, 0.50]` (global cap) | §7.3 | spec §5 |
| CI5 | `composite ∈ [0.05, 1.00]` | §2.3 floor + clamp | spec §6 |
| CI6 | `w_agreement + w_completeness + w_uncertainty == 1.0` | `__post_init__` | spec |
| CI7 | uncertainty terms gated by `match_found` (no Ui without match) | §7.1 conditions | spec §5 |
| CI8 | exactly one pair-selection rule fires per evaluation | §5.2 (mutual exclusion) | spec |
| CI9 | trust score ignored when `is_default == True` | §4.3 rule | spec / impl docstring |
| CI10 | composite reproducible from same inputs (per A5) | §12 | A5 |
| CI11 | `tier` boundaries align with `low_upper` / `medium_upper` config | §2.4 | spec §7 |
| CI12 | `triggered_conditions` membership encodes which Ui fired (audit provenance) | §9 | A4 derivative |

### §13.1 Cross-axiom mapping

- **A3 (CONFIDENCE-GATED ENFORCEMENT)** — CI11 (tier boundary
  authority). The PolicyEngine's HIGH-confidence gate for TAKEDOWN
  depends on this spec's `medium_upper` boundary.
- **A4 (AUDIT COMPLETENESS WITH PROVENANCE)** — CI12
  (triggered_conditions provide policy_lineage_ref components).
  The confidence portion of `engine_lineage` requires the config
  version to be threaded through (§10.1).
- **A5 (DETERMINISTIC REPLAY)** — CI10 + §12.

### §13.2 Test coverage

The current implementation has no dedicated invariant test suite.
Tests exist as part of the smoke surface
(`backend/_smoke_policy.py` exercises the engine indirectly via
PolicyEngine). The canonical invariant test catalogue is
`docs/testing/INVARIANT_TESTS.md` *(planned)*; promotion in
Phase 5 of the migration.

---

## §14 Reconciliation history

The v3-merged spec, the implementation
(`backend/app/engines/confidence_engine.py`), and the data-model
layer (`backend/app/models/confidence_models.py`) had eleven points
of semantic drift, each resolved in this canonical document.

### §14.1 Composite floor (C-CE-1)

Spec §6: `clamp(0.05, 1.0)`. Impl: `_clamp01` (i.e., `[0, 1]`).
Adopt spec's 0.05 floor as canonical (§2.3). Note: smoothing math
keeps composite ≥ 0.20 in practice, so the gap is dormant. A
**Lightweight ADR** adds the explicit floor to the implementation.

### §14.2 Reason-code rename (C-CE-2)

Spec §9: `GLOBAL_CAP_APPLIED`. Impl: `UNCERTAINTY_GLOBAL_CAP`.
Adopt impl's name (more specific, in-production). Spec name
recorded as superseded terminology (§14.10).

### §14.3 Engine-extra reason codes (C-CE-3)

`BOTH_TRUSTS_DEFAULT` and `TRUST_CAP_GRAY_ZONE` are emitted by the
implementation but were not listed in spec §9. Promoted to
canonical (§9.1) — they provide useful provenance.

### §14.4 Vestigial reason codes (C-CE-4)

`INPUT_VALIDATION_FAILED` and `INPUT_QUALITY_LOW` are defined in the
enum but never emitted. Marked DEPRECATED in §9.2; retained for
backward compatibility per the impl docstring. A future Standard
ADR may remove.

### §14.5 Pair-rule terminology collision (C-CE-5)

Spec §3 named pair-selection rules R1–R5. PolicyEngine §8 uses
R1–R5 for risk-control rules. To avoid cross-domain confusion,
the agreement pair-selection rules are renamed to descriptive
identifiers (`NO_MATCH`, `BOTH_DEFAULT`, `OWNER_DEFAULT`,
`UPLOADER_DEFAULT`, `ALL_PRESENT` — §5.2). The "R" prefix going
forward is reserved for risk-control rules in PolicyEngine.

### §14.6 Signal-source casing (C-CE-6)

Spec §2 enumerated signal sources in uppercase. The implementation
normalises to lowercase for comparison via `_norm_source`. Behaviour
is case-insensitive in practice. Documented explicitly in §11.1.

### §14.7 ConfidenceTier authority (C-CE-7)

The enum's boundaries (0.40, 0.70) and values (LOW/MEDIUM/HIGH)
are now formally owned by this document, per
`docs/governance/DOMAINS.md` cross-domain components. PolicyEngine
consumes them.

### §14.8 TrustState semantics (C-CE-8)

The implementation docstring rule "trust_score is consulted only
when is_default == False" is promoted to canonical in §4.3 and
codified as invariant CI9.

### §14.9 ConfidenceBreakdown lineage gap (C-CE-9)

The output type does not carry `config_version`. Documented as a
known gap in §10.1; pipeline worker is responsible for threading.
A future Lightweight ADR may add `config_version` to
`ConfidenceBreakdown` for self-contained provenance.

### §14.10 Composite formula meaning (C-CE-10)

The `(1 - uncertainty)` form in §2.2 is documented as the
**complement**: uncertainty *reduces* confidence. The weighted
contribution from the uncertainty component goes UP when uncertainty
goes DOWN. The form is canonical and intentional.

### §14.11 Vocabulary drift surface (C-CE-11)

Several terms drifted between spec and impl. Surfaced here for
migration into a future `docs/governance/VOCABULARY.md` (Phase 4
Registry sub-batch):

| Concept | Spec term | Canonical term | Notes |
|---|---|---|---|
| Uncertainty global cap reason | `GLOBAL_CAP_APPLIED` | `UNCERTAINTY_GLOBAL_CAP` | rename per C-CE-2 |
| Pair-selection rules | `R1`..`R5` (spec §3) | `NO_MATCH` / `BOTH_DEFAULT` / `OWNER_DEFAULT` / `UPLOADER_DEFAULT` / `ALL_PRESENT` | rename per C-CE-5 to avoid PolicyEngine R1-R5 collision |
| Signal source enumeration | `FINGERPRINT` / `EMBEDDING` / `FUSION` (uppercase enumeration) | case-insensitive (lowercased internally) | per C-CE-6 |
| Confidence-domain S1-S5 | "S1..S5" (spec §4) | retained, but always qualified `confidence.S1`..`confidence.S5` cross-domain | disambiguation from `policy.S1`..`policy.S5` |

`docs/governance/VOCABULARY.md` is **not yet planned in any
specific batch**; it will canonicalise these and equivalents from
other engines.

---

## §15 Open questions / Future work

Documented for visibility; not commitments.

- **Composite floor in implementation** — add explicit `clamp(0.05, 1.0)`
  in `_composite` (Lightweight ADR). Recovers spec/impl alignment for
  C-CE-1.
- **`config_version` in `ConfidenceBreakdown`** — promote
  config_version to the output type so confidence provenance is
  self-contained, removing the pipeline-worker dependency for A4
  threading (Lightweight ADR).
- **Removal of vestigial reason codes** — Standard ADR to remove
  `INPUT_VALIDATION_FAILED` and `INPUT_QUALITY_LOW` from the enum.
- **Per-tenant confidence configs** — multi-tenant operation with
  per-tenant threshold overrides (Standard ADR; ties to the same
  open question in `./policy_engine.md` §15).
- **Calibration loop** — feed false-positive corrections (per A6)
  back into confidence configuration tuning. Currently there is no
  feedback loop; out of scope for this canonical batch.
- **Replay test surface** — once `INVARIANT_TESTS.md` exists,
  CI1–CI12 should each be covered by ≥ 1 test (Phase 5).

> **Important constraint reminder:** new scoring dimensions (i.e., a
> 4th component beyond agreement / completeness / uncertainty)
> require a **Constitutional ADR** per §1.3. Adding a U6 or pair-
> selection variant within an existing component is a Standard or
> Lightweight ADR.

---

## §16 Versioning and Change Process

This spec is **EVOLVING** per
`docs/constitution/GOVERNANCE.md` §8. Compatibility expectations
are low — consumers should expect change at each minor bump.

| Change type | ADR tier |
|---|---|
| Threshold tweak (e.g., `gray_zone_upper 0.85 → 0.83`) | Lightweight |
| Per-term magnitude (`u4_value 0.05 → 0.07`) | Lightweight |
| New reason code | Lightweight |
| New uncertainty term (U6) | Standard |
| Rename existing reason code | Standard (cross-domain consumer surface) |
| Tier boundary (`medium_upper 0.70 → 0.65`) | Standard (ripples into PolicyEngine matrix) |
| Composite weight redistribution (e.g., `0.4/0.3/0.3 → 0.5/0.25/0.25`) | Standard |
| New scoring dimension (4th component) | Constitutional |
| Removing a reason code (including vestigial) | Standard |

The `ConfidenceConfig.config_version` constant in
`backend/app/models/confidence_models.py` MUST be bumped in lockstep
with this spec's `version:` field when threshold defaults or weight
defaults change. Mismatch is a **P1** governance violation per
`docs/constitution/GOVERNANCE.md` §5.

### §16.1 Graduation to STABLE

Same gates as PolicyEngine (`./policy_engine.md` §16.1):

1. Rule and threshold set unchanged for at least one minor revision
   cycle.
2. No production incidents implicating confidence logic in 90 days.
3. Consumer integrations (policy, qa) report stability.
4. The invariant test suite covers all 12 invariants in §13.

Architect approves graduation.

---

## §17 Cross-references

- **Axioms** (`../constitution/AXIOMS.md`): A3 (HIGH tier gates
  TAKEDOWN), A4 (audit lineage), A5 (deterministic replay), A7
  (evidence preservation — confidence values are part of the
  evidence record).
- **Constitutional governance**
  (`../constitution/GOVERNANCE.md`): §1 (tier hierarchy), §3 (ADR
  tiers), §5 (severity model), §8 (stability levels).
- **Domain ownership** (`../governance/DOMAINS.md`): confidence
  domain owns this spec; ConfidenceTier listed as cross-domain
  component.
- **Architecture state** (`../state/STATE.md`): ConfidenceEngine
  ACTIVE v1.0.
- **Implementation**:
  - `backend/app/engines/confidence_engine.py`
  - `backend/app/models/confidence_models.py`
- **Consumer specs**:
  - `./policy_engine.md` — §3 (PBRA), §5 (base matrix
    consumes ConfidenceTier), §8 (R1 reads
    `GRAY_ZONE`), §10 (PolicyResult echoes
    `confidence_tier` and `confidence_composite`).
- **Producer specs**:
  - `./decision_engine.md` *(planned)* — RiskScore producer; runs
    in parallel with this engine in EVALUATION.
- **Working drafts (untracked)**:
  - `.claude/memory/confidence_engine_spec.md` — v3 final merged.
- **Future canonical references**:
  - `../governance/VOCABULARY.md` *(unscheduled)* — for
    cross-engine terminology drift (§14.11).
  - `../testing/INVARIANT_TESTS.md` *(planned)* — CI1–CI12
    coverage.
