---
authority: SPEC
domain: decision
status: ACTIVE
version: 1.0
stability: EVOLVING
owner: decision (interim: architect)
supersedes:
  - (no predecessor — fresh canonical per docs/specs/README.md)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; canonical-spec ratification; to be backfilled)
---

# Decision Engine — Canonical Specification

The DecisionEngine performs **risk scoring** during the EVALUATION
phase (per A1). It maps a `DecisionInput` to a `RiskScore` carrying a
composite risk value, a `RiskBand` classification, and a per-term
`RiskBreakdown` exposing raw and weighted contributions for audit.

This document is the canonical specification — Tier 2 (SPEC) — and is
the sole authority on DecisionEngine semantics. The implementation is
`backend/app/engines/decision_engine.py` and
`backend/app/models/decision_models.py`.

The engine is consumed by the **PolicyEngine** (DECISION phase per
A1) and runs in parallel with the **ConfidenceEngine** during
EVALUATION. See §17 cross-references.

---

## §1 Purpose and Authority

### §1.1 Purpose

The DecisionEngine answers a single question: **how dangerous is this
candidate match?** It produces a deterministic, auditable risk score
that downstream phases compose with confidence to select an
enforcement action.

It is structurally distinct from the **PolicyEngine** — which is the
DECISION phase per A1. DecisionEngine produces the *risk dimension*
of the EVALUATION output; ConfidenceEngine produces the *certainty
dimension*; PolicyEngine fuses both into a `PolicyAction`. The naming
follows pipeline lineage (DecisionEngine = engine producing inputs to
the DECISION phase), not the A1 phase to which it belongs. See §1.5
for the canonical disambiguation.

It is a **pure deterministic function** per A5. No randomness. No
I/O. No external state. Numeric anomalies (None, NaN, out-of-range)
are coerced to safe defaults rather than raised; the only construction
failure is `ThresholdConfig` weight validation (§4.2).

### §1.2 Position in the pipeline

Per A1 (PIPELINE PHASE INTEGRITY), risk is computed during the
**EVALUATION** phase, alongside confidence:

```
INGESTION → ANALYSIS → EVALUATION → DECISION → ENFORCEMENT
                          ↑              ↑
                          DecisionEngine  PolicyEngine consumes
                          (this spec)     RiskScore.band +
                          ConfidenceEngine RiskScore.composite
                          (parallel)      (via DecisionOutput)
```

The DecisionEngine and ConfidenceEngine are **siblings** in
EVALUATION. Their outputs are independent: neither reads the other.
Both are bundled by the pipeline worker into the inputs the
PolicyEngine consumes (§5.3, §6.1).

### §1.3 Authority

This document is **TIER 2 (SPEC)** per
`docs/constitution/GOVERNANCE.md` §1. Owned by the **decision**
domain (`docs/governance/DOMAINS.md`). Modification:

| Change | ADR tier |
|---|---|
| Threshold tweak (e.g., `low_upper 0.50 → 0.55`) | Lightweight |
| Per-term weight tweak (e.g., `w_similarity 0.45 → 0.50`) within current 5-term shape | Lightweight |
| Velocity-normalisation constant (`_VELOCITY_K`, `_MAX_OBSERVATIONS`) | Lightweight |
| Trust-floor constant (`_MIN_TRUST = 0.1`) | Lightweight |
| Signal-source quality table entry add/edit | Lightweight |
| Tier boundary shift (e.g., `medium_upper 0.85 → 0.80`) | Standard (ripples into PolicyEngine matrix and into ConfidenceEngine boundary parity expectations) |
| Composite weight redistribution (renormalisation across all 5 terms) | Standard |
| New scoring term (6th term) | Standard (extends formula shape) |
| Removing or renaming an existing term | Standard |
| Reordering of EVALUATION phase relative to other A1 phases | Constitutional (axiom-level pipeline change) |
| New scoring **dimension** beyond risk and confidence | Constitutional (changes the pipeline's structural shape) |

### §1.4 Stability

**EVOLVING** per `docs/constitution/GOVERNANCE.md` §8. The five-term
linear additive model is structurally stable; weights, thresholds,
and per-term internals may continue to refine via Lightweight or
Standard ADRs. Compatibility expectations are low — consumers should
expect change at each minor version bump. Graduation to STABLE
follows the same gates as `policy_engine.md` §16.1 and
`confidence_engine.md` §16.1.

The `RiskBand` enum is a cross-domain consumer surface
(PolicyEngine reads it; qa tests assert against it; confidence
boundary parity is expected). Promoting the enum to LOCKED status is
a future design goal independent of this spec's stability — and is
deliberately deferred until the runtime pipeline materialises the
DecisionOutput envelope (§14.1).

### §1.5 Naming disambiguation

Three names in this repo collide on the word *decision*. The
canonical mapping is:

| Term | Refers to | Owner |
|---|---|---|
| `DecisionEngine` | the engine in this spec — risk scoring | decision domain |
| `DecisionOutput` | the envelope that wraps `RiskScore` + `InputSnapshot` and is consumed by PolicyEngine | decision domain (§5.3 — currently a structural Protocol; not yet a concrete type) |
| **DECISION phase** | the A1 phase performed by the **PolicyEngine** | policy domain |

Future writers MUST NOT use "DecisionEngine" as a synonym for the
DECISION phase. The DECISION phase is implemented by `policy_engine`
(see `./policy_engine.md` §1.2). The DecisionEngine is an EVALUATION-
phase risk producer.

This naming is historical: the DecisionEngine was named for what its
output *feeds* (the Decision phase), not for what the engine itself
*does* (compute risk). A future Lightweight ADR may rename it to
`RiskEngine` for clarity; renaming is deferred until the
DecisionOutput envelope is materialised (§14.1) so the rename and the
envelope land together.

---

## §2 The Risk Model

### §2.1 Five terms

Risk is a **weighted linear sum** of five independent signals.

| Term | Range (raw) | Direction | Semantic |
|---|---|---|---|
| **similarity** | [0, 1] | + | Strength of the perceptual match between input and matched protected asset. Higher = stronger match → higher risk. |
| **trust_owner** | [0, 1] | + | Trust score of the matched asset's rights-holder. Higher owner trust = the match is on a credibly-protected asset → higher risk of unauthorized use. |
| **trust_uploader** | [0, 1] | − (inverted) | Trust score of the uploader. **Inverted**: high uploader trust = low risk; the engine consumes `(1 - trust_uploader)` so the term contributes in the same direction as the others. |
| **velocity** | [0, 1] | + | Rate of independent observations of this match. Higher velocity = viral spread → higher enforcement urgency. |
| **match_quality** | [0, 1] | + | Quality of the signal source that produced the match (fingerprint, embedding, fusion, …). Higher source quality = the match is more reliable → higher risk. |

These five terms answer different questions. They are **independent**
in the modelling sense: one being high or low does not by itself
constrain the others. The composite combines them via weighted sum.

### §2.2 Composite formula

```
raw_score   = w_similarity     * similarity_raw
            + w_trust_owner    * max(trust_owner_raw, MIN_TRUST)
            + w_trust_uploader * (1 - max(trust_uploader_raw, MIN_TRUST))
            + w_velocity       * velocity_raw
            + w_match_quality  * match_quality_raw

composite   = clamp(raw_score, 0, 1)
```

with default weights `0.45 / 0.15 / 0.10 / 0.15 / 0.15` (sum = 1.0).
Weight defaults are validated at `ThresholdConfig` construction time
(`__post_init__` raises `ValueError` if the sum drifts beyond `1e-9`).

The `MIN_TRUST = 0.1` floor is applied **only to the weighted
contribution** of each trust term. The `breakdown.raw` field
preserves the original normalised value so audit consumers see the
true input. See §3.4 for the rationale.

### §2.3 RiskBand classification

The composite is bucketed into three `RiskBand` values consumed by
the PolicyEngine (`./policy_engine.md` §5).

| Range | Band | Semantic |
|---|---|---|
| `[0.00, 0.50)` | `LOW` | low risk; PolicyEngine proposes ALLOW regardless of confidence |
| `[0.50, 0.85)` | `MEDIUM` | moderate risk; PolicyEngine proposes FLAG / REVIEW |
| `[0.85, 1.00]` | `HIGH` | high risk; PolicyEngine may propose REVIEW / RESTRICT / TAKEDOWN depending on confidence |

Tier boundaries are configured by `ThresholdConfig.low_upper`
(default 0.50) and `ThresholdConfig.medium_upper` (default 0.85).

> **Note — RiskBand vs ConfidenceTier.** The RiskBand boundaries
> (0.50 / 0.85) are **NOT** the same as the ConfidenceTier boundaries
> (0.40 / 0.70). They are independently calibrated. PolicyEngine's
> base matrix (`./policy_engine.md` §5) cross-tabulates the two
> classifications; changing either set of boundaries is therefore a
> Standard ADR because it ripples into the matrix.

### §2.4 Term independence and additivity

The five-term **additive** form is intentional. Two failed
alternatives are explicitly rejected:

- **Multiplicative composition** (e.g., `similarity * trust_owner * …`)
  collapses to ~0 whenever any single factor is low. Real-world cases
  routinely have one weak signal (e.g., low velocity for a fresh
  upload of high-similarity content). Additivity preserves the
  contribution of every other strong signal.
- **Min/max composition** loses magnitude information from the
  non-extremal terms. Audit consumers need per-term visibility, which
  the additive + breakdown form provides.

A 6th term (or any restructuring of the additive shape into a
non-linear form) is a Standard ADR per §1.3.

---

## §3 Per-term Computation

### §3.1 Similarity

```
similarity_raw = clamp01(safe(input.match.similarity))
similarity_weighted = w_similarity * similarity_raw
```

`similarity` is the most direct evidence of a match. Raw value comes
from the matching stage (perceptual hash distance, embedding cosine,
or fusion result; see `content_similarity.md` *(planned)*). The
DecisionEngine treats it as a black-box [0, 1] scalar.

### §3.2 Trust signals (owner and uploader)

Trust signals are read from the `TrustState`-shaped inputs
(`MatchSignal.trust_owner`, `MatchSignal.trust_uploader` per the data
contracts in §4). Both have the same numeric shape (`trust_score:
float`). Their *direction* differs:

```
trust_owner_raw       = clamp01(safe(input.trust_owner.trust_score))
trust_owner_effective = max(MIN_TRUST, trust_owner_raw)
trust_owner_weighted  = w_trust_owner * trust_owner_effective

trust_uploader_raw       = clamp01(safe(input.trust_uploader.trust_score))
trust_uploader_effective = max(MIN_TRUST, trust_uploader_raw)
trust_uploader_weighted  = w_trust_uploader * (1.0 - trust_uploader_effective)
```

Owner trust contributes in the **positive** direction: a credible
rights-holder's match is more likely to represent unauthorised use.
Uploader trust contributes in the **inverted** direction: a credible
uploader's content is less likely to be a violation.

### §3.3 Velocity

Velocity captures the **rate** of independent observations of the
matched content, with two computation paths:

```
def velocity_raw(input) -> float:
    timestamps = clean_timestamps(input.observation_timestamps)
    if len(timestamps) >= 2:
        span = max(timestamps) - min(timestamps)
        if span > 0:
            rate = len(timestamps) / span
            return clamp01(rate / (rate + VELOCITY_K))   # K = 100.0
    # Fallback path — fewer than 2 timestamps OR span <= 0.
    count = max(0.0, safe(input.observation_count))
    return clamp01(log1p(count) / log1p(MAX_OBSERVATIONS))   # MAX = 1000.0
```

- **Pair-rate path** (preferred when timestamps are present): produces
  a saturation curve `rate / (rate + K)`. With `K = 100`, a steady
  rate of 100 observations per timestamp-unit yields velocity = 0.5.
- **Count-fallback path** (when `< 2` timestamps OR `span <= 0`):
  log-normalised count divided by `log1p(MAX_OBSERVATIONS)`. With
  `MAX_OBSERVATIONS = 1000`, an observation_count of 1000 yields
  velocity = 1.0.

Timestamps are cleaned: None, non-numeric, and NaN entries are
filtered out before computing the span. Out-of-order timestamps are
preserved as-is (the engine does not assume monotonicity); only the
min/max span matters.

The constants `VELOCITY_K`, `MAX_OBSERVATIONS` are private
implementation details and do not currently appear in `ThresholdConfig`.
Promoting them to config is a future Lightweight ADR (§15).

### §3.4 Trust floor and breakdown semantics (audit fidelity)

The `MIN_TRUST = 0.1` floor is applied **only when computing the
weighted contribution** of trust terms. The breakdown structure
preserves both views:

```
breakdown.trust_owner.raw        = trust_owner_raw         # original [0, 1]
breakdown.trust_owner.weighted   = w_trust_owner * trust_owner_effective
                                 = w_trust_owner * max(MIN_TRUST, trust_owner_raw)
```

So a `trust_owner_raw = 0.0` value yields:
- `breakdown.trust_owner.raw = 0.0` (auditable: owner trust was zero)
- `breakdown.trust_owner.weighted = 0.15 * 0.1 = 0.015` (composite math
  uses the floored value)

This split is intentional. The floor prevents owner/uploader trust
of exactly zero from collapsing risk to ALLOW even when other signals
are strong. Auditors and downstream consumers can still see the
original input via `breakdown.*.raw`.

### §3.5 Match-quality (signal source)

```
match_quality_raw = clamp01(safe(SIGNAL_SOURCE_QUALITY.get(
    score.signal_source, DEFAULT_SIGNAL_QUALITY)))
match_quality_weighted = w_match_quality * match_quality_raw
```

Lookup table:

| `signal_source` | Quality |
|---|---|
| `"fingerprint+embedding"` | 1.00 |
| `"embedding"` | 0.90 |
| `"fingerprint"` | 0.70 |
| `"metadata"` | 0.40 |
| (any other / unknown) | 0.50 (default) |

The default is mid-range so an unknown source is neither dismissed
nor over-weighted. Adding a row to the table is a **Lightweight ADR**.

> **Cross-engine note (vocabulary drift).** The DecisionEngine table
> uses `"fingerprint+embedding"` as the highest-quality value. The
> ConfidenceEngine uses `"FUSION"` (case-insensitive) as the analogous
> high-quality marker (`./confidence_engine.md` §4.2). The PolicyEngine
> normalises `signal_source` to lowercase and accepts the value
> `"fusion"` (`./policy_engine.md` §11.1). These three vocabularies
> describe the same upstream signal but disagree on spelling; full
> reconciliation belongs to `docs/governance/VOCABULARY.md`
> *(unscheduled — flagged in `./confidence_engine.md` §14.11 / D-DE-3
> here)*. Until then, the pipeline worker is responsible for emitting
> the value in the form each downstream consumer expects.

### §3.6 Numeric primitives

All numeric inputs are passed through:

- `safe(x)` — coerces None and non-numeric to `0.0`; coerces NaN to
  `0.0`. Never raises on shape errors.
- `clamp01(x)` — clamps to `[0, 1]`. Never raises.

Out-of-range or invalid input becomes a **conservative zero**
contribution rather than an exception. This is intentional: the
EVALUATION phase MUST NOT crash on upstream numeric drift; the audit
record makes the drift visible (e.g., `breakdown.similarity.raw =
0.0` while `match_found = True` in the upstream snapshot is a
signal qa can detect).

---

## §4 Inputs

### §4.1 DecisionInput

Frozen dataclass — primitives only. The engine derives everything
internally; callers MUST NOT precompute scoring decisions.

```python
@dataclass(frozen=True)
class DecisionInput:
    match: MatchSignal                  # match.similarity
    trust_owner: TrustSignal            # trust_owner.trust_score
    trust_uploader: TrustSignal         # trust_uploader.trust_score
    score: ScoreSignal                  # score.signal_source
    observation_count: int
    config_version: str
    observation_timestamps: Sequence[float] = ()
```

Required attribute access:

| Path | Type | Used by |
|---|---|---|
| `input.match.similarity` | `float` | §3.1 |
| `input.trust_owner.trust_score` | `float` | §3.2 |
| `input.trust_uploader.trust_score` | `float` | §3.2 |
| `input.score.signal_source` | `str` | §3.5 |
| `input.observation_count` | `int` | §3.3 fallback |
| `input.observation_timestamps` | `Sequence[float]` | §3.3 primary |
| `input.config_version` | `str` | echoed into the audit record (A4 lineage) |

Defined in `backend/app/models/decision_models.py`. The data shapes
`MatchSignal`, `TrustSignal`, `ScoreSignal` are intentionally minimal
(single-field dataclasses) so cross-engine refactors of trust
representation, match metadata, etc., remain decoupled from this
engine.

> **Decision/Confidence trust shape divergence.** The DecisionEngine
> currently consumes `TrustSignal(trust_score: float)` while the
> ConfidenceEngine consumes `TrustState(trust_score, is_default)`
> (`./confidence_engine.md` §4.3). The DecisionEngine therefore has
> no view of the registry-default signal at this layer; the floor
> (§3.4) is applied uniformly. Aligning the trust shape across both
> engines is a Standard ADR (§15) and is tied to the trust-reader
> spec *(planned: `docs/specs/trust_reader.md`)*.

### §4.2 ThresholdConfig

```python
@dataclass(frozen=True)
class ThresholdConfig:
    w_similarity:     float = 0.45
    w_trust_owner:    float = 0.15
    w_trust_uploader: float = 0.10
    w_velocity:       float = 0.15
    w_match_quality:  float = 0.15
    low_upper:        float = 0.50
    medium_upper:     float = 0.85
```

`__post_init__` validates that `sum(weights) == 1.0` (within `1e-9`
tolerance). Construction with non-summing weights raises
`ValueError` — this is the **only** exception path the engine and
its data contracts emit during normal operation.

Default ordering (similarity dominates at 45%) reflects the
modelling assumption that *evidence of match* is the strongest
single signal. The trust + velocity + match-quality combination
contributes the remaining 55%.

### §4.3 config_version semantics

`DecisionInput.config_version` is the version identifier of the
`ThresholdConfig` instance the caller passes. It is **NOT** validated
by the engine; the caller is responsible for setting it consistent
with the actual config used.

The version threads downstream through:

1. The `RiskScore` output does **not** carry `config_version`
   directly (see §5.4).
2. The pipeline worker is responsible for assembling the
   `DecisionOutput.risk.config_version` field from `DecisionInput.config_version`
   when it builds the envelope consumed by the PolicyEngine.
3. The PolicyEngine reads it as `decision.risk.config_version` and
   emits it in `PolicyResult.decision_config_version`
   (`./policy_engine.md` §10).

This mirrors the version-threading pattern in the ConfidenceEngine
(`./confidence_engine.md` §4.4 / §10.1). A future Lightweight ADR
may add `config_version` to `RiskScore` directly for self-contained
provenance (D-DE-1 in §14.1).

---

## §5 Outputs

### §5.1 RiskScore

```python
@dataclass(frozen=True)
class RiskScore:
    composite: float          # [0, 1]
    band: RiskBand            # LOW / MEDIUM / HIGH
    breakdown: RiskBreakdown
```

### §5.2 RiskBreakdown

```python
@dataclass(frozen=True)
class RiskBreakdown:
    similarity:       TermContribution
    trust_owner:      TermContribution
    trust_uploader:   TermContribution
    velocity:         TermContribution
    match_quality:    TermContribution

@dataclass(frozen=True)
class TermContribution:
    raw:      float    # original normalised value (post-clamp, pre-floor)
    weighted: float    # contribution to composite (weight × effective)
```

Per-term auditability is mandatory. `breakdown.*.raw` MUST always
reflect the input value after numeric clamping but before any
domain-specific transformation (the trust floor, the uploader
inversion). This is what makes a stored RiskScore explainable
without re-running the engine.

The five `TermContribution` records are an exhaustive decomposition:
`composite ≈ sum(b.weighted for b in breakdown)` (subject to clamping
to [0, 1]). Auditors can sum them and compare to `composite` as a
sanity check.

### §5.3 DecisionOutput envelope (planned concrete type)

The PolicyEngine consumes a `DecisionOutput` envelope, NOT a
`RiskScore` directly. The envelope's shape is defined as a structural
`Protocol` in `backend/app/engines/policy_engine.py` (see
`./policy_engine.md` §4.1). The required attribute paths are:

| Path | Type | Source |
|---|---|---|
| `decision.risk.band` | `RiskBand` | `RiskScore.band` |
| `decision.risk.composite` | `float` | `RiskScore.composite` |
| `decision.risk.config_version` | `str` | `DecisionInput.config_version` |
| `decision.input_snapshot.match.matched` | `bool` | upstream matching stage |
| `decision.input_snapshot.match.similarity` | `float` | matching stage |
| `decision.input_snapshot.config_version` | `str` | **confidence** config version (per `./confidence_engine.md` §10.1 / C-CE-9) |

**Implementation status:** the concrete `DecisionOutput` type does
**not** exist as of v1.0 of this spec. The `policy_engine.py`
Protocol expresses the structural contract; smoke tests
(`backend/tests/_smoke_policy.py`) build ad-hoc namespaces matching
the Protocol. The pipeline worker (when it adopts the engine triple
— see §9.4) will need to construct a concrete envelope type. This is
**D-DE-1**, the highest-priority gap noted in §14.1 and the open
work in §15.

**The envelope is owned by the decision domain** even though it
includes confidence-config metadata, because it is the *input
contract for the DECISION phase* and the decision domain owns the
producer side of that contract. ConfidenceEngine continues to own
the confidence config schema; the DecisionOutput envelope merely
carries the version string through.

### §5.4 RiskScore lineage gap

The `RiskScore` output does NOT carry `config_version`. This is a
known gap (D-DE-1, §14.1) symmetric with `ConfidenceBreakdown`'s
analogous gap (`./confidence_engine.md` §10.1 / C-CE-9). Until
resolved by ADR, the pipeline worker is responsible for threading
`DecisionInput.config_version` into `DecisionOutput.risk.config_version`.

### §5.5 A4 audit contribution

Per A4, the audit record for an enforcement decision MUST include:

- `risk_score` — the `composite` field of `RiskScore`.
- `engine_lineage.decision_config_version` — sourced from
  `DecisionInput.config_version` and threaded through `DecisionOutput`.

Per-term `breakdown.*.raw` and `breakdown.*.weighted` SHOULD be
included in the audit record (the storage spec
`docs/specs/storage.md` *(planned)* will pin "MUST" vs "SHOULD" for
the breakdown). Storing the breakdown enables rule-by-rule
attribution of any individual decision under A6 (Human Review
Authority) and is necessary for fully reproducible replay under A5
when the implementation evolves.

---

## §6 Execution Semantics

### §6.1 Runtime placement

The engine is invoked from the EVALUATION stage of the pipeline. The
canonical invocation pattern (target-state runtime — see §9.4 for the
current MVP gap):

```
# After ANALYSIS (matching stage produces match + similarity):

risk = decision_engine.compute_risk(decision_input, threshold_config)

# In parallel (or sequentially — outputs are independent):
confidence = confidence_engine.compute_confidence(confidence_input, confidence_config)

# Pipeline worker assembles the DecisionOutput envelope:
decision_output = build_decision_output(
    risk=risk,
    input_snapshot=...,                # confidence-config version threaded here
    decision_config_version=decision_input.config_version,
)

# DECISION phase (PolicyEngine):
result = policy_engine.evaluate_policy(decision_output, confidence, policy_context)
```

The DecisionEngine and ConfidenceEngine outputs are independent —
neither depends on the other. The pipeline worker MAY parallelise
their computation (per A1 sub-phase flexibility); execution order
within EVALUATION is NOT a phase-ordering guarantee. The only
ordering guarantee is that BOTH MUST complete before DECISION (per
A1 phase integrity).

### §6.2 No PBRA

DecisionEngine does **not** use the PBRA model (PROPOSE / BOUND /
REFINE / ASSERT). PBRA is the PolicyEngine's execution model
(`./policy_engine.md` §3) for selecting an action under safety and
risk-control rules. DecisionEngine's job is upstream: produce a
deterministic numeric score. No multi-phase clamping, no
override hierarchy — just `score = sum(weights × terms)`.

The PolicyEngine performs PBRA over (RiskScore, ConfidenceBreakdown,
PolicyContext); DecisionEngine performs **single-pass risk
quantification**. Conflating the two execution models is forbidden
(see §8.4).

### §6.3 Five-stage internal flow

```
┌───────────────────────────────────────────────────────────────────┐
│  Stage 1 — Input normalization                                    │
│            safe() + clamp01() per term; clean_timestamps().       │
├───────────────────────────────────────────────────────────────────┤
│  Stage 2 — Per-term raw computation                               │
│            similarity, trust_owner, trust_uploader (inverted),    │
│            velocity (pair-rate or log-fallback), match_quality.   │
├───────────────────────────────────────────────────────────────────┤
│  Stage 3 — Per-term weighting                                     │
│            apply trust floor (MIN_TRUST = 0.1) to weighted        │
│            contribution only; preserve raw in breakdown.          │
├───────────────────────────────────────────────────────────────────┤
│  Stage 4 — Composite assembly                                     │
│            raw_score = sum(weighted); composite = clamp01(raw).   │
├───────────────────────────────────────────────────────────────────┤
│  Stage 5 — Band classification                                    │
│            composite < low_upper → LOW                            │
│            composite < medium_upper → MEDIUM                      │
│            otherwise → HIGH                                       │
└───────────────────────────────────────────────────────────────────┘
```

Each stage is a pure function of its inputs and the
`ThresholdConfig`. There are no conditional jumps that depend on
external state. Output is total: every well-typed `DecisionInput` +
`ThresholdConfig` pair produces a `RiskScore` (no exceptions during
computation; only construction-time `ValueError` on weight
validation).

### §6.4 No conflict resolution

There is no rule firing, no override hierarchy, no escalation /
demotion. Every term contributes its weighted value; there is no
arithmetic short-circuit. The only "conflict" the engine resolves
is **input numeric anomaly** (None / NaN / out-of-range), which is
silently coerced to `0.0` per §3.6.

Decisions about action escalation and demotion live entirely in the
PolicyEngine (`./policy_engine.md` §3, §6, §8). DecisionEngine
provides one of the two numeric inputs that drive those decisions.

---

## §7 Invariants

Properties that MUST hold for any input combination.

| # | Invariant | Enforced by | Axiom / spec |
|---|---|---|---|
| DI1 | `composite ∈ [0, 1]` | §6.3 stage 4 (clamp01) | spec §2.2 |
| DI2 | `band` is one of `{LOW, MEDIUM, HIGH}` | §6.3 stage 5 + enum typing | spec §2.3 |
| DI3 | `band` boundaries align with `low_upper` / `medium_upper` config | §6.3 stage 5 | spec §2.3 |
| DI4 | `breakdown.*.raw ∈ [0, 1]` for every term | §3.6 clamp01 | spec |
| DI5 | `breakdown.trust_owner.raw == clamp01(input.trust_owner.trust_score)` (no floor in raw) | §3.4 | spec — auditability |
| DI6 | `breakdown.trust_uploader.raw == clamp01(input.trust_uploader.trust_score)` (raw is uninverted, unfloored) | §3.4 | spec — auditability |
| DI7 | weight sum invariant: `Σ w_i == 1.0` (within 1e-9) | `ThresholdConfig.__post_init__` | spec |
| DI8 | `composite ≈ sum(breakdown.*.weighted)` clamped to [0, 1] | §6.3 stage 4 | spec — auditability |
| DI9 | `composite` reproducible from same `(DecisionInput, ThresholdConfig)` (per A5) | §8.1 | A5 |
| DI10 | engine emits no exceptions on numeric input shape (None / NaN / out-of-range coerced) | §3.6 | spec — robustness |
| DI11 | velocity fallback path activates iff `len(clean_timestamps) < 2 OR span <= 0` | §3.3 | spec |
| DI12 | trust floor (`MIN_TRUST = 0.1`) is applied to `*.weighted` only, never to `*.raw` | §3.4 | spec — auditability |

### §7.1 Cross-axiom mapping

- **A1 (PIPELINE PHASE INTEGRITY)** — DecisionEngine is an
  EVALUATION-phase engine; its placement is verified by the spec
  being indexed under the **decision** domain (DOMAINS.md) and being
  consumed by `policy_engine` (DECISION phase).
- **A4 (AUDIT COMPLETENESS WITH PROVENANCE)** — DI5, DI6, DI8 give
  consumers per-term auditability for `policy_lineage_ref` /
  `engine_lineage`. §5.5 enumerates the audit-record contributions.
- **A5 (DETERMINISTIC REPLAY)** — DI9 + §8.

A2 (MATCH PREREQUISITE) and A3 (CONFIDENCE-GATED ENFORCEMENT) are
satisfied **downstream** by the PolicyEngine; the DecisionEngine
does not enforce them at this layer. The DecisionEngine produces a
risk score whose absolute value alone never authorises enforcement
(that takes a `PolicyAction` from the PolicyEngine, gated by both
A2 and A3).

### §7.2 Test coverage

The current implementation has **no dedicated invariant test suite**.
Smoke coverage is indirect: `backend/tests/_smoke_policy.py` exercises
DecisionOutput-shaped inputs but does not test `compute_risk`
directly. The canonical invariant test catalogue is
`docs/testing/INVARIANT_TESTS.md` *(planned, Phase 2D)*; promotion
to a proper test suite happens in Phase 5 of the migration. DI1–DI12
should each have ≥ 1 dedicated test in that catalogue.

---

## §8 Determinism Guarantees

The engine is a pure deterministic function per **A5
(DETERMINISTIC REPLAY)**.

- No randomness.
- No time-of-day reads.
- No environment-variable reads inside engine code.
- No external state.
- All inputs flow through `compute_risk(input, config)`.

### §8.1 I/O envelope

**ZERO** I/O. No logging, no metrics, no observability calls inside
engine code. Operability hooks attach at the call site (the pipeline
worker), not in the engine. This matches the ConfidenceEngine
contract (`./confidence_engine.md` §12.1) and is **stricter** than
the PolicyEngine, which permits an ERROR-level log on terminal-
invariant violation.

There is no analogous invariant violation in DecisionEngine: a
deterministic linear sum of clamped inputs cannot "violate" itself
in a way runtime logging would diagnose. Adding I/O is a **P0**
governance violation per `docs/constitution/GOVERNANCE.md` §5.

### §8.2 Replay attribution

Given an A4 audit record's decision portion (`risk_score`,
`risk_band`, `engine_lineage.decision_config_version`, and ideally
the per-term breakdown), replay reconstructs:

1. Load `ThresholdConfig` at `decision_config_version`.
2. Reconstruct `DecisionInput` from upstream signals (matching
   stage's `similarity`; trust scores from the trust reader at the
   relevant snapshot; observation counts/timestamps from the
   observation store).
3. Re-run `compute_risk(input, config)`.
4. Compare `composite` (suggested 4dp tolerance, same as PolicyEngine
   `./policy_engine.md` §10.1 evaluation hash), `band`, and per-term
   `breakdown` if stored.

Mismatch = either a non-deterministic code path (P0 bug per
GOVERNANCE.md §5) or audit tampering (P0 evidence violation under
A7).

### §8.3 Determinism boundaries

The following are **outside** the engine's determinism contract:

- The matching stage producing `similarity` MAY be approximate
  (vector ANN search, etc.); replaying it requires the matching
  spec's own determinism guarantees (`docs/specs/content_similarity.md`
  *(planned)*).
- The trust reader producing `trust_owner.trust_score` and
  `trust_uploader.trust_score` MAY query an external registry; its
  determinism contract belongs to `docs/specs/trust_reader.md`
  *(planned)*.
- The observation store producing `observation_count` /
  `observation_timestamps` MAY have eventual-consistency semantics
  per `docs/specs/storage.md` *(planned)*.

Replay determinism end-to-end requires that *every* upstream
producer be replay-attributable. The DecisionEngine guarantees
**function-level** determinism: same `(DecisionInput, ThresholdConfig)`
→ same `RiskScore`. End-to-end replay determinism is a system-level
property assembled from each engine's contract.

### §8.4 Anti-conflation with PolicyEngine

DecisionEngine MUST NOT import or reference `PolicyEngine`,
`PolicyAction`, `PolicyContext`, or `EvidenceStrength`. The
dependency direction is:

```
decision_models ←—————————┐
                          │
confidence_models ←————┐  │
                       │  │
                  policy_models  ── imports both
                       ↑  ↑
                       │  │
                  policy_engine  ── consumes RiskBand + ConfidenceTier
```

Reverse imports from decision into policy are forbidden. Any
DecisionEngine refinement that requires reading PolicyAction or
PBRA semantics indicates an architectural error and SHOULD be
escalated to a Standard ADR rather than fixed inline.

---

## §9 Failure Model

### §9.1 Numeric anomalies

| Anomaly | Behavior | Rationale |
|---|---|---|
| `None` for any numeric field | coerced to `0.0` | upstream signal absent → conservative no-contribution |
| `NaN` | coerced to `0.0` | math safety; clamps don't apply to NaN |
| value `< 0` | clamped to `0.0` | normalisation invariant |
| value `> 1` | clamped to `1.0` | normalisation invariant |
| non-numeric type | coerced to `0.0` | shape robustness |
| empty `observation_timestamps` | velocity fallback path | §3.3 |
| `observation_timestamps` with `< 2` valid entries (after cleaning) | velocity fallback path | §3.3 |
| timestamp `span <= 0` | velocity fallback path | §3.3 — guards division by zero |
| unknown `signal_source` | match-quality default = 0.50 | §3.5 — neither dismiss nor over-weight |

### §9.2 Construction failures

The only **raised** exception path is at `ThresholdConfig`
construction:

```python
ThresholdConfig(w_similarity=0.5, w_trust_owner=0.5, w_trust_uploader=0.5, ...)
# raises ValueError: weights must sum to 1.0; got 1.5
```

This is a programmer-error guard, equivalent to the
`ConfidenceConfig.__post_init__` weight-sum validation
(`./confidence_engine.md` §2.2). Production runtime should never see
this raise: the config object is constructed once at startup or
at config-reload and validated then.

### §9.3 Degraded execution mode

There is no "degraded mode" — the engine is total. Every well-typed
input produces a well-typed output. Numeric anomalies degrade to
conservative zeros (§9.1) but the engine still returns a `RiskScore`.

The **system** may treat the output as low-confidence when input
quality is poor (e.g., `breakdown.similarity.raw == 0.0` while
`match_found == True` is a signal that ConfidenceEngine and qa
inspection should flag) — but this is a downstream concern, not the
DecisionEngine's responsibility.

### §9.4 Current runtime gap (D-DE-2)

The runtime pipeline worker (`backend/app/workers/pipeline_worker.py`)
**does not currently invoke** `decision_engine.compute_risk`. The
worker uses the legacy `scoring_engine.score(similarity)` (a simple
band lookup over similarity alone) and the legacy
`enforcement_engine.decide(...)` (a trust-aware threshold function
that produces the legacy 3-level `ALLOW/FLAG/BLOCK` action set).

The new triple — DecisionEngine + ConfidenceEngine + PolicyEngine —
exists as deterministic library code. The pipeline-worker rewiring
is tracked under `docs/specs/eventing.md` *(planned)* +
`docs/specs/job_processing.md` *(planned)*. Until that lands:

- This spec is **AUTHORITATIVE for the engine contract**.
- The worker still emits the legacy `EnforcedPayload` shape
  (`action: str` from `{ALLOW, FLAG, BLOCK}`), which is **not** the
  five-action `PolicyAction` ladder.
- `docs/state/STATE.md` records DecisionEngine as ACTIVE v1.0
  (the **engine** is active; the **runtime wiring** is what's
  pending).
- Per `docs/state/STATE.md`, the in-memory JobStore + lack of event
  bus + lack of worker fleet are all EXPERIMENTAL or MVP-only.

This is the highest-leverage runtime gap blocking PolicyEngine
integration. See §15 / D-DE-2.

### §9.5 Timeout behavior

The engine has no notion of timeout. `compute_risk` is a CPU-bound
pure function with predictable cost (5 weighted multiplications +
constant-time helpers + linear scan over `observation_timestamps`).
For typical inputs (< 1000 timestamps), wall-clock cost is
microseconds.

If a timeout is required at the call site (e.g., the pipeline worker
caps EVALUATION-stage time), it MUST be implemented at the worker
layer with a cancellation primitive — never inside the engine.
Adding cancellation primitives to a pure function is forbidden per
§8.1 (zero I/O / zero side effects).

### §9.6 Observability obligations

Observability is the **call site's** responsibility. The pipeline
worker is expected to emit (per `docs/specs/observability.md`
*(planned)*):

- a `risk_computed` event (or equivalent) carrying `composite`,
  `band`, `decision_config_version`, and `breakdown` summary;
- a structured timing measurement of the call;
- correlation identifiers (`request_id`, `job_id`) propagated from
  upstream.

The engine itself contributes ZERO observability calls (§8.1) but
its output is **richly observable** through the breakdown structure.

---

## §10 Observability

### §10.1 Engine-internal observability

ZERO. The engine emits no logs, metrics, or traces. See §8.1.

### §10.2 Engine-output observability

The output structure is the observability surface. Consumers
(pipeline worker, audit storage, qa replay) extract:

- `composite` — primary risk signal.
- `band` — discrete classifier; useful as a metric label.
- `breakdown.<term>.raw` — input visibility (per-term).
- `breakdown.<term>.weighted` — contribution visibility (per-term).

Recommended call-site emission (per `docs/specs/observability.md`
*(planned)*):

| Signal | Type | Cardinality | Notes |
|---|---|---|---|
| `decision_engine.invocation_count` | counter | low | rate of computation |
| `decision_engine.composite` | histogram | n/a | distribution of risk values |
| `decision_engine.band` | counter (label) | 3 (LOW/MEDIUM/HIGH) | band balance |
| `decision_engine.compute_duration_ms` | histogram | n/a | wall-clock cost |
| `decision_engine.term.<name>.raw` | histogram | per term | input distribution |
| `decision_engine.term.<name>.weighted` | histogram | per term | contribution distribution |
| `decision_engine.config_version` | gauge (label) | low | active config |

Term-level histograms (last two rows) are **recommended but not
required**; cost vs benefit depends on cardinality budgets and is
governed by the platform domain.

### §10.3 Lineage propagation

Per A4, every audit record consuming a DecisionEngine output MUST
carry:

- `engine_lineage.decision_config_version` — the
  `DecisionInput.config_version` value used.
- (recommended) the per-term `breakdown` object, for full
  reproducibility under A5 + auditability under A4.

Lineage propagation is the **pipeline worker's** responsibility. The
engine produces the values; the worker carries them through the
audit record.

### §10.4 Correlation identifiers

The engine itself does not consume `request_id` / `job_id` / `event_id`.
These propagate through the call site and into the audit record.
A future Lightweight ADR may add an opaque `correlation_id` field to
`DecisionInput` if observability evolution requires it; today the
correlation is carried by the worker.

---

## §11 Extensibility Rules

### §11.1 What may evolve safely (Lightweight ADR)

- Per-term weight defaults (within current 5-term shape).
- `low_upper` / `medium_upper` boundary defaults — but **note** that
  shifting these ripples into the PolicyEngine base matrix
  (`./policy_engine.md` §5) and SHOULD be coordinated with the policy
  domain; therefore see §1.3 — boundary shifts are **Standard** ADR
  in practice.
- `_VELOCITY_K`, `_MAX_OBSERVATIONS`, `_MIN_TRUST` constants.
- Adding rows to the `_SIGNAL_SOURCE_QUALITY` table.
- Adjusting the velocity-fallback log-normalisation function (§3.3),
  provided the input/output contracts remain unchanged.
- Documentation refinements with no semantic change (clarifications,
  cross-reference updates, vocabulary alignment).

### §11.2 What requires Standard ADR

- Composite weight redistribution that crosses term-importance
  boundaries (e.g., reducing `w_similarity` below 0.30).
- Tier boundary shifts that touch the PolicyEngine matrix.
- New scoring term (6th term).
- Removing or renaming an existing term.
- Changing the trust-floor application semantics (§3.4).
- Introducing config-driven internals (e.g., promoting `_VELOCITY_K`
  into `ThresholdConfig`).
- Renaming `DecisionEngine` to `RiskEngine` (§1.5).
- Materialising a concrete `DecisionOutput` envelope type (§5.3).
- Aligning the trust-input shape with `confidence_engine`'s
  `TrustState` (§4.1 note).

### §11.3 What requires Constitutional ADR

- Reordering the EVALUATION phase relative to other A1 phases.
- Introducing a new scoring **dimension** (e.g., a "regulatory
  band" alongside risk and confidence) — changes the pipeline's
  structural shape.
- Removing the deterministic-replay guarantee (A5 violation).
- Adding I/O to engine code (A5 / determinism violation).

### §11.4 Forbidden coupling (anti-patterns)

- **DecisionEngine reading PolicyEngine internals.** Reverse
  imports from `decision` to `policy` are forbidden (§8.4). Risk
  computation MUST NOT depend on the action ladder, PBRA phases, or
  `EvidenceStrength`.
- **DecisionEngine reading ConfidenceEngine output.** The two
  engines are siblings in EVALUATION; risk does not consume
  confidence and vice versa. The PolicyEngine fuses them downstream.
- **State-bearing behavior.** The engine is a pure function; adding
  caches, memoisation across calls, or background workers is
  forbidden (A5 violation).
- **Direct DB / network access.** Trust scores, observation counts,
  match data, and signal source MUST be supplied by the caller via
  `DecisionInput`. The engine never reads from a registry, a DB, or
  a remote service (`./eventing.md` *(planned)*'s "no synchronous
  ML / external calls inside engines" rule applies here equivalently).
- **Hard-coded thresholds outside ThresholdConfig.** Per
  `.claude/rules/ml-evaluation.md` ("Thresholding"), thresholds MUST
  be configurable. The current `_VELOCITY_K`, `_MAX_OBSERVATIONS`,
  `_MIN_TRUST`, and `_SIGNAL_SOURCE_QUALITY` constants are
  module-level and are documented gaps (§15) — promotion to
  `ThresholdConfig` is a Lightweight ADR.

### §11.5 Cross-domain constraints

- `RiskBand` enum is a **cross-domain consumer surface**
  (`docs/governance/DOMAINS.md` cross-domain components). Renaming
  values, changing the enum order, or removing a band requires a
  Standard ADR with the policy domain reviewing.
- `ThresholdConfig` is **decision-owned**, but `low_upper` /
  `medium_upper` ripple into the `PolicyEngine` base matrix; their
  effective change is cross-domain.
- `DecisionOutput` envelope (when materialised — §5.3) is
  **decision-owned but jointly consumed**: the policy domain
  consumes it, and a future `eventing` spec may emit serialised
  forms of it. Schema changes are Standard ADR.

---

## §12 State + Runtime Model

### §12.1 Stateless function

The engine is **stateless**. `compute_risk` has no memory of prior
calls, no caches, no memoisation. Each invocation is independent.

```
compute_risk(input₁, config) ≡ compute_risk(input₁, config)
                              ⊥ compute_risk(input₂, config)   for input₁ ≠ input₂
```

No global mutable state. No singletons. No locks. The function is
**thread-safe by construction** — concurrent calls cannot interfere.

### §12.2 Idempotency

By A5 + statelessness:

- Same `(DecisionInput, ThresholdConfig)` → same `RiskScore`,
  bit-for-bit (modulo platform float representation, mitigated by
  the recommended 4dp tolerance for replay comparison — §8.2).
- Repeated invocation has no cumulative effect.

This is the strongest form of idempotency. Pipeline retry semantics
(at-least-once delivery, RQ retries, crash-resume) interact with the
engine **only** through repeated `compute_risk` calls — which are
guaranteed identical and therefore safe.

### §12.3 Retry semantics

The engine itself never retries. Retry is the **call site's**
responsibility:

- The pipeline worker MAY retry an EVALUATION-phase invocation if
  the surrounding I/O fails (e.g., the trust-reader call that
  produces `trust_owner.trust_score` raises). Retrying does not
  affect engine semantics.
- The engine raises only `ValueError` from `ThresholdConfig`
  construction (§9.2). This is a programmer error and MUST NOT be
  retried — re-running with the same broken config produces the
  same `ValueError`.

### §12.4 Pipeline integration boundaries

```
┌────────────────────────────────────────────────────────────────────┐
│  ANALYSIS phase (matching, fingerprint, embedding)                 │
│       │                                                            │
│       ▼                                                            │
│  EVALUATION phase                                                  │
│   ├─ DecisionEngine.compute_risk(...) ──▶ RiskScore               │
│   ├─ ConfidenceEngine.compute_confidence(...) ──▶ ConfidenceBreakdown │
│   └─ pipeline_worker.build_decision_output(...) ──▶ DecisionOutput │
│       │                                                            │
│       ▼                                                            │
│  DECISION phase (PolicyEngine — see ./policy_engine.md)            │
└────────────────────────────────────────────────────────────────────┘
```

The DecisionEngine's runtime contract is exhausted at the call
boundary. Everything downstream — including the construction of the
DecisionOutput envelope — is the pipeline worker's responsibility.

### §12.5 Snapshot policy

Per A7 (EVIDENCE PRESERVATION), the inputs to a
DecisionEngine call MUST be reconstructable for replay. The
recommended snapshot at the storage layer:

| Field | Reason |
|---|---|
| `input.match.similarity` | reproduces §3.1 |
| `input.trust_owner.trust_score`, `input.trust_uploader.trust_score` | reproduces §3.2 |
| `input.score.signal_source` | reproduces §3.5 |
| `input.observation_count`, `input.observation_timestamps` | reproduces §3.3 |
| `input.config_version` | reproduces ThresholdConfig (provenance) |
| (recommended) the `breakdown` object | direct verification of §6.3 stage 4 |

The exact snapshot schema lives in `docs/specs/storage.md`
*(planned)* under the EvidenceStore.

---

## §13 Governance Alignment

### §13.1 Constitutional dependencies

This spec is subordinate to:

- **A1 (PIPELINE PHASE INTEGRITY)** — DecisionEngine is an
  EVALUATION-phase engine. Phase reordering or relabelling requires
  a Constitutional ADR (§1.3).
- **A4 (AUDIT COMPLETENESS WITH PROVENANCE)** — DI5, DI6, DI8
  guarantee per-term auditability for the audit record's
  `policy_lineage_ref` / `engine_lineage` fields (A4 minimum
  schema).
- **A5 (DETERMINISTIC REPLAY)** — engine determinism guarantee
  (§8). Adding I/O is a Constitutional violation.
- **A7 (EVIDENCE PRESERVATION)** — input snapshot policy (§12.5)
  ensures every decision is replay-reproducible.

A2 (MATCH PREREQUISITE) and A3 (CONFIDENCE-GATED ENFORCEMENT) are
satisfied by the PolicyEngine downstream; the DecisionEngine does
not implement them at this layer.

### §13.2 Canonical ownership

- **decision domain** owns this spec, the `RiskScore` / `RiskBand` /
  `RiskBreakdown` / `TermContribution` types, the
  `DecisionInput` / `MatchSignal` / `TrustSignal` / `ScoreSignal` /
  `ThresholdConfig` types, the `DecisionOutput` envelope (when
  materialised — §5.3), and the planned `risk_assessment.md` and
  `content_similarity.md` specs (per `docs/governance/DOMAINS.md`).
- The `RiskBand` enum is a **cross-domain consumer surface**;
  changes require Standard ADR with the policy domain reviewing.
- The `ThresholdConfig` boundary fields (`low_upper`,
  `medium_upper`) ripple into PolicyEngine and are de facto
  cross-domain.

### §13.3 ADR requirements for future modifications

Modifications to this spec MUST:

1. Bump `version:` per the matrix in §1.3.
2. Reference an ADR in `adr_references:` if the change is anything
   other than a documentation clarification.
3. Update `backend/app/engines/decision_engine.py` and
   `backend/app/models/decision_models.py` in lockstep with any
   semantic change.
4. Bump the implementation's version constant (when introduced —
   D-DE-1) in lockstep with this spec's `version:` field. Mismatch
   is a **P1** governance violation per
   `docs/constitution/GOVERNANCE.md` §5.
5. Notify the policy domain when modifying any field that is
   cross-domain (RiskBand, boundary thresholds, or the
   DecisionOutput envelope shape).

### §13.4 Same-tier conflict resolution

If this spec and another Tier-2 spec disagree (e.g., a future
`risk_assessment.md` adopts a different per-term decomposition), the
conflict resolves per `docs/constitution/GOVERNANCE.md` §2:

1. ADR required (Standard, since cross-spec).
2. Owning-domain authority decides; both specs are decision-owned,
   so the decision lead resolves.
3. If the conflict crosses domains (e.g., `risk_assessment.md`
   conflicts with `policy_engine.md` on RiskBand semantics), the
   architect breaks the tie via Standard ADR.

### §13.5 Stability graduation

Same gates as `policy_engine.md` §16.1 / `confidence_engine.md`
§16.1:

1. Per-term decomposition unchanged for at least one minor revision
   cycle.
2. No production incidents implicating decision logic in 90 days.
3. Consumer integrations (policy, qa) report stability.
4. The invariant test suite covers DI1–DI12 (`docs/testing/INVARIANT_TESTS.md`
   *(planned)*).
5. The DecisionOutput envelope (§5.3) is materialised and the
   pipeline worker is wired to use it (§9.4 closure).

Architect approves graduation.

---

## §14 Reconciliation history

This spec has **no superseded predecessor** in `.claude/memory/`
(per `docs/specs/README.md`'s "no predecessor — fresh canonical").
The reconciliation surface is between this canonical spec and the
existing implementation + sibling specs.

### §14.1 D-DE-1 — DecisionOutput envelope materialisation

**Drift:** `policy_engine.py` defines `DecisionOutput` only as a
structural Protocol; no concrete dataclass / model exists. The
pipeline worker has no production code path that constructs it
(§9.4). Smoke tests (`backend/tests/_smoke_policy.py`) build ad-hoc
namespaces matching the Protocol shape.

**Resolution adopted:** §5.3 records the structural contract in this
canonical spec; materialisation as a concrete type (`DecisionOutput`
dataclass or Pydantic model in `backend/app/models/decision_models.py`)
is the highest-priority follow-up — a **Lightweight ADR** that:

- adds the concrete type;
- defines the boundary contract for `input_snapshot.config_version`
  carrying the **confidence** config version (per `./confidence_engine.md`
  §10.1 / C-CE-9);
- updates `policy_engine.py` to consume the concrete type
  (replacing the Protocol);
- threads `config_version` into `RiskScore` for self-contained
  provenance (the `RiskScore.config_version` gap noted in §5.4) —
  symmetric with the analogous `ConfidenceBreakdown.config_version`
  ADR in `./confidence_engine.md` §15.

These are bundled in one Lightweight ADR because the pieces co-evolve.

### §14.2 D-DE-2 — Runtime wiring gap

**Drift:** the pipeline worker still uses
`scoring_engine.score(similarity)` and
`enforcement_engine.decide(...)`. The new triple
(DecisionEngine + ConfidenceEngine + PolicyEngine) is implemented
but unwired (§9.4).

**Resolution adopted:** this spec is authoritative for the
DecisionEngine **library contract**. Runtime wiring belongs to
`docs/specs/eventing.md` *(planned)* + `docs/specs/job_processing.md`
*(planned)*, both of which are the next-priority canonical specs
after `decision_engine.md` lands. The legacy `scoring_engine.py`
and `enforcement_engine.py` are tracked for **deprecation** in
`docs/state/STATE.md` once the new triple is wired:

| Component | Current state | Successor | Removal trigger |
|---|---|---|---|
| `backend/app/engines/scoring_engine.py` | ACTIVE (legacy) | DecisionEngine + PolicyEngine | when `eventing.md` + `job_processing.md` land and worker is rewired |
| `backend/app/engines/enforcement_engine.py` | ACTIVE (legacy) | PolicyEngine + (planned) `enforcement_audit.md` actor | same trigger |

Per the append-only migration constraint, neither legacy file is
deleted in this batch. They remain ACTIVE in MVP and transition to
DEPRECATED in `STATE.md` when their successor is PROPOSED.

### §14.3 D-DE-3 — Signal-source vocabulary drift

**Drift:** three engines spell the high-quality signal source
differently:

- DecisionEngine: `"fingerprint+embedding"` (string literal in the
  match-quality table — §3.5).
- ConfidenceEngine: `"FUSION"` (case-insensitive comparison — §3.5
  / `./confidence_engine.md` §4.2).
- PolicyEngine: `"fusion"` (post-normalisation — `./policy_engine.md`
  §11.1).

**Resolution adopted:** this spec records the DecisionEngine
vocabulary (§3.5) as authoritative for the match-quality lookup.
The pipeline worker is responsible for emitting the value in the
form each downstream consumer expects until full reconciliation in
`docs/governance/VOCABULARY.md` *(unscheduled)*. This drift is
**deferred**, not resolved, and is jointly tracked with C-CE-11 in
`./confidence_engine.md` §14.11.

### §14.4 D-DE-4 — Trust-signal shape divergence

**Drift:** DecisionEngine consumes `TrustSignal(trust_score: float)`
while ConfidenceEngine consumes
`TrustState(trust_score: float, is_default: bool)`
(`./confidence_engine.md` §4.3). The DecisionEngine has no
visibility into the registry-default state; its trust floor (§3.4)
is applied uniformly.

**Resolution adopted:** §4.1 documents the divergence. Aligning
the two trust shapes — likely by promoting `TrustSignal` to carry
`is_default` and applying conditional logic in §3.2 — is a
**Standard ADR** and is bundled with the planned
`docs/specs/trust_reader.md`. Until then, the DecisionEngine treats
all trust scores uniformly; the registry-default semantic is fully
realised only at the confidence layer.

### §14.5 D-DE-5 — Hard-coded constants outside ThresholdConfig

**Drift:** `_VELOCITY_K = 100.0`, `_MAX_OBSERVATIONS = 1000.0`,
`_MIN_TRUST = 0.1`, and the `_SIGNAL_SOURCE_QUALITY` table are
module-level constants in `backend/app/engines/decision_engine.py`,
not config-driven. `.claude/rules/ml-evaluation.md` requires
thresholds to be configurable.

**Resolution adopted:** §3.3, §3.4, §3.5 describe the constants
canonically. Promoting them into `ThresholdConfig` is a
**Lightweight ADR** (each constant separately, or bundled). The
spec records this as a known gap rather than treating it as a
blocking violation; the values are sufficiently conservative for
MVP use.

### §14.6 D-DE-6 — Naming: DecisionEngine ≠ DECISION phase

**Drift:** the engine name implies it performs the DECISION phase
(per A1), but in fact PolicyEngine performs DECISION; this engine
performs **risk scoring** in EVALUATION. The naming is historical
(this engine produces an *input* to DECISION).

**Resolution adopted:** §1.5 is the canonical disambiguation. A
future Lightweight ADR may rename the engine to `RiskEngine` for
clarity; renaming is bundled with the DecisionOutput envelope
materialisation (D-DE-1) so the rename and the envelope land
together. Until that ADR, `DecisionEngine` and `RiskEngine` are
interchangeable in informal communication; in formal documentation
the existing name (`DecisionEngine`) is canonical.

### §14.7 Documentation lineage

| Source | Status | Location |
|---|---|---|
| (none — this spec is fresh canonical) | — | — |
| `backend/app/engines/decision_engine.py` docstring | implementation; remains authoritative for code-level details | `backend/app/engines/` |
| `backend/app/models/decision_models.py` docstring | implementation schema; remains authoritative for code-level details | `backend/app/models/` |
| `.claude/rules/ml-evaluation.md` | TRANSITIONAL; partially superseded by this spec for thresholding rules | `.claude/rules/` |
| `.claude/rules/ml_pipeline.md` | TRANSITIONAL; partially superseded by this spec for embeddings/similarity threading | `.claude/rules/` |

Per Tier-5 demotion (`docs/constitution/GOVERNANCE.md` §1), the
`.claude/rules/ml-evaluation.md` and `.claude/rules/ml_pipeline.md`
files SHOULD be annotated with a `superseded by:` deprecation note
pointing to this spec, in their next edit. The files themselves are
**not** deleted in this batch (append-only migration constraint).

---

## §15 Open questions / Future work

Documented for visibility; not commitments.

- **Materialise `DecisionOutput` envelope** (D-DE-1, §14.1) —
  Lightweight ADR. Highest-priority follow-up; bundles with
  `RiskScore.config_version` self-contained provenance.
- **Wire pipeline worker to the new engine triple** (D-DE-2,
  §9.4 / §14.2) — anchored in `docs/specs/eventing.md` *(planned)*
  + `docs/specs/job_processing.md` *(planned)*. Closes the
  legacy-scoring deprecation path.
- **Promote module-level constants into `ThresholdConfig`** (D-DE-5,
  §14.5) — `_VELOCITY_K`, `_MAX_OBSERVATIONS`, `_MIN_TRUST`,
  `_SIGNAL_SOURCE_QUALITY`. Lightweight ADR per constant or bundled.
- **Align trust-signal shape with ConfidenceEngine `TrustState`**
  (D-DE-4, §14.4) — Standard ADR, bundled with
  `docs/specs/trust_reader.md` *(planned)*.
- **Vocabulary reconciliation for signal_source** (D-DE-3, §14.3) —
  unscheduled; ties to `docs/governance/VOCABULARY.md` *(unscheduled)*
  and `./confidence_engine.md` §14.11.
- **Per-tenant ThresholdConfig** — multi-tenant operation with
  per-tenant boundary overrides. Standard ADR; ties to the same
  open question in `./policy_engine.md` §15 and
  `./confidence_engine.md` §15.
- **Calibration loop** — feed false-positive corrections (per A6) and
  enforcement outcomes back into ThresholdConfig tuning. No feedback
  loop currently; out of scope for this canonical batch.
- **Replay test surface** — once `INVARIANT_TESTS.md` exists, DI1–DI12
  should each be covered by ≥ 1 test (Phase 5).
- **Promote `RiskBand` to LOCKED** — once enforcement integrations
  stabilise across api / pipeline / security and the DecisionOutput
  envelope is materialised, `RiskBand` is a candidate for LOCKED
  status (Constitutional ADR). Today it remains EVOLVING with this
  spec.
- **Velocity model evolution** — the current pair-rate + log-fallback
  is a pragmatic MVP. Future work may introduce time-decay weighting,
  per-jurisdiction velocity floors, or signal-specific normalisation.
  Standard ADR if the pipeline shape changes; Lightweight if only
  constants change.

> **Important constraint reminder:** introducing a new scoring
> **dimension** (a 6th dimension beyond risk and confidence — e.g.,
> a "regulatory band") requires a **Constitutional ADR** per §1.3.
> Adding a new term to the existing 5-term risk decomposition is a
> Standard ADR.

---

## §16 Versioning and Change Process

This spec is **EVOLVING** per `docs/constitution/GOVERNANCE.md` §8.
Compatibility expectations are low — consumers should expect change
at each minor bump.

| Change type | ADR tier |
|---|---|
| Threshold tweak (boundary or constant) | Lightweight |
| Per-term weight tweak (within current shape) | Lightweight |
| New `_SIGNAL_SOURCE_QUALITY` row | Lightweight |
| New reason or audit field on `RiskBreakdown` | Lightweight |
| Promoting a module-level constant into `ThresholdConfig` | Lightweight |
| Composite weight redistribution | Standard |
| Tier boundary shift (`low_upper` / `medium_upper`) | Standard (ripples into PolicyEngine matrix) |
| New scoring term (6th term in current dimension) | Standard |
| Removing or renaming an existing term | Standard |
| Materialising `DecisionOutput` envelope | Standard (cross-domain — policy consumes) |
| Renaming `DecisionEngine` → `RiskEngine` | Standard (cross-domain consumer surface) |
| Aligning trust-signal shape with ConfidenceEngine | Standard |
| New scoring **dimension** (4th composite alongside risk/confidence/...) | Constitutional |
| Removing the deterministic-replay guarantee | Constitutional |

A `decision_version` constant is **not yet present** in
`backend/app/engines/decision_engine.py` (D-DE-1). When introduced,
it MUST be bumped in lockstep with this spec's `version:` field.
Mismatch is a **P1** governance violation per
`docs/constitution/GOVERNANCE.md` §5.

### §16.1 Graduation to STABLE

Same gates as `policy_engine.md` §16.1 / `confidence_engine.md`
§16.1, plus:

- The `DecisionOutput` envelope (§5.3) is materialised as a concrete
  type and the pipeline worker is wired to use it (§9.4).
- All P1 gaps surfaced in §14 are resolved via ADR.

Architect approves graduation.

### §16.2 Demoting from STABLE

If a STABLE spec needs material change that breaks the STABLE
contract, a Standard ADR may demote it back to EVOLVING per
`docs/constitution/GOVERNANCE.md` §8. The ADR MUST justify why
EVOLVING is preferred over a deprecation cycle.

---

## §17 Cross-references

- **Axioms** (`../constitution/AXIOMS.md`): A1 (EVALUATION phase),
  A4 (audit lineage), A5 (deterministic replay), A7 (evidence
  preservation — risk-score values are part of the evidence record).
- **Constitutional governance**
  (`../constitution/GOVERNANCE.md`): §1 (tier hierarchy), §3 (ADR
  tiers), §5 (severity model), §8 (stability levels).
- **Domain ownership** (`../governance/DOMAINS.md`): decision
  domain owns this spec; `RiskBand` listed as cross-domain
  component (consumed by policy, qa).
- **Architecture state** (`../state/STATE.md`): DecisionEngine
  ACTIVE v1.0; legacy `scoring_engine.py` and `enforcement_engine.py`
  remain ACTIVE in MVP, slated for DEPRECATED on pipeline rewire
  (D-DE-2).
- **Implementation**:
  - `backend/app/engines/decision_engine.py`
  - `backend/app/models/decision_models.py`
- **Consumer specs**:
  - `./policy_engine.md` — §3 (PBRA), §4 (DecisionOutput protocol),
    §5 (base matrix consumes RiskBand).
- **Sibling specs (parallel in EVALUATION)**:
  - `./confidence_engine.md` — independent computation; consumed
    jointly by PolicyEngine.
- **Producer specs (planned)**:
  - `../specs/content_similarity.md` *(planned)* — produces
    `match.similarity`.
  - `../specs/risk_assessment.md` *(planned)* — superset / sibling
    spec for the broader risk-assessment surface; will resolve any
    overlap with this spec via §13.4.
  - `../specs/trust_reader.md` *(planned)* — produces
    `trust_owner.trust_score`, `trust_uploader.trust_score`.
- **Downstream contracts (planned)**:
  - `../specs/api_contracts.md` *(planned)* — `RiskBand` may be
    surfaced in API responses; cross-domain.
  - `../specs/eventing.md` *(planned)* — pipeline integration that
    closes D-DE-2.
  - `../specs/job_processing.md` *(planned)* — worker rewiring that
    closes D-DE-2.
  - `../specs/observability.md` *(planned)* — formalises the
    metrics surface in §10.2.
  - `../specs/storage.md` *(planned)* — defines the EvidenceStore
    snapshot policy referenced in §12.5 and §5.5.
  - `../security/enforcement_audit.md` *(planned)* — audit + dispute
    layer that consumes `risk_score` and `breakdown`.
- **TRANSITIONAL sources** (Tier 5 — partially superseded):
  - `.claude/rules/ml-evaluation.md` — thresholding rules.
  - `.claude/rules/ml_pipeline.md` — pipeline component constraints.
- **Future canonical references**:
  - `../governance/VOCABULARY.md` *(unscheduled)* — vocabulary
    reconciliation for signal_source (D-DE-3).
  - `../testing/INVARIANT_TESTS.md` *(planned)* — DI1–DI12 coverage.
