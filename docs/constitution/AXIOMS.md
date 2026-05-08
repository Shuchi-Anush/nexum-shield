---
authority: AXIOM
domain: governance
status: ACTIVE
version: 2.0
owner: architect
supersedes:
  - .claude/memory/implementation_plan.md §1 (working draft, untracked)
  - .claude/CLAUDE.md NON-NEGOTIABLE CONSTRAINTS section (partial — pipeline + audit + async + evidence)
  - CLAUDE.md NON-NEGOTIABLE CONSTRAINTS section (partial — same content)
  - .claude/rules/architecture.md (partial — Pipeline Integrity → A1 semantic phase model)
  - .claude/rules/enforcement.md (partial — Audit/Evidence → A4/A7; Human-in-the-loop + Dispute Handling → A6)
  - .claude/rules/storage.md §6 EVIDENCE STORE (partial — evidence preservation → A7)
  - .claude/rules/ml-evaluation.md (partial — Feedback Loop / FP correction → A6)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; constitutional v1→v2 amendment; to be backfilled)
---

# AXIOMS — Nexum Shield

The seven statements below are constitutional. Violating any of them
makes the system fundamentally unsafe or unauditable. Every spec,
every rule, every agent instruction, and every ADR is subordinate to
this file.

## Authority

This file overrides every other document in the repository. The only
mechanism for changing an axiom is a Constitutional ADR (see
`./GOVERNANCE.md`). Constitutional ADRs require unanimous approval
from all domain leads, a 72-hour comment window, and the architect's
sign-off. They are open to no abstentions.

## Axiom budget

This file MUST contain at most 10 axioms. The current count is 7.
Adding an axiom requires deprecating an existing one if the budget
is full. The budget exists to keep the constitution short enough
that every engineer can hold it in working memory; if everything is
constitutional, nothing is.

---

## A1 — PIPELINE PHASE INTEGRITY

The system MUST execute the following semantic phases. Specific
orchestration (event-driven, monolithic, batch, streaming, etc.) is
NOT axiomatic — only the semantic phase set and ordering are.

Required phases:

  1. **INGESTION**    — accept and validate an input asset.
  2. **ANALYSIS**     — extract deterministic fingerprints / embeddings
                        and identify candidate matches.
  3. **EVALUATION**   — compute risk and confidence scores.
  4. **DECISION**     — select a PolicyAction from score and confidence.
  5. **ENFORCEMENT**  — apply or queue the chosen action.
  6. **AUDITABILITY** — produce and seal an immutable audit record per
                        A4 and preserve evidence per A7.

The semantic ordering ANALYSIS → EVALUATION → DECISION → ENFORCEMENT
MUST NOT be transposed: enforcement cannot precede decision; decision
cannot precede evaluation; evaluation cannot precede analysis.
INGESTION is always first. AUDITABILITY runs continuously alongside
phases 1–5 (every phase emits to the audit record) and reaches
terminal sealing at or before ENFORCEMENT commits.

Sub-phases MAY be introduced (e.g., ANALYSIS may be split into
fingerprint, embedding, and matching). Phases MAY be reorganised
across services, jobs, or events provided the semantic ordering
above is preserved. Sub-phase breakdowns and orchestration choices
live in spec-level documents (`docs/specs/eventing.md`,
`docs/specs/job_processing.md`, etc.).

Earlier exact-topology forms found in legacy documents
(`ingest → job → process → match → score → enforce` or
`ingest → job → process → match → score → decide → enforce`) are
particular implementations of this axiom, not the axiom itself.

## A2 — MATCH PREREQUISITE

No enforcement action MAY occur without a match.

Formally: if `match_found == False`, then `final_action ∈ {ALLOW, FLAG}`.

The action ladder is `ALLOW < FLAG < REVIEW < RESTRICT < TAKEDOWN`
(severity order). RESTRICT and TAKEDOWN — the actions that affect a
publisher's reach or remove content — are forbidden in the absence
of a match. FLAG (internal-tracking-only) is permitted because it
preserves an analytics signal without taking enforcement action.

## A3 — CONFIDENCE-GATED ENFORCEMENT

Automated content removal (TAKEDOWN) requires the highest confidence
tier (`HIGH`). No exception. No override. No future rule may bypass
this.

Spec-level rules (PolicyEngine S2 + the terminal invariant guard)
implement this axiom in defense-in-depth. The axiom is the parent
constraint; the spec is the enforcement.

## A4 — AUDIT COMPLETENESS WITH PROVENANCE

Every enforcement decision MUST produce an immutable audit record
sufficient for replay (A5), audit reconstruction, and dispute
attribution (A6). The record MUST contain at minimum:

**Content & decision fields**

- `input_id` — content hash (SHA-256 or equivalent) of the input asset.
- `matched_id` — content hash of the matched protected asset, or
  null when `match_found == False`.
- `similarity` — similarity score that produced the match, or null.
- `risk_score` — composite risk score from the EVALUATION phase.
- `confidence` — composite confidence score from the DECISION phase.
- `action` — final PolicyAction emitted by DECISION.
- `timestamp` — wall-clock time the decision was produced.

**Provenance lineage** (causal traceability for replay attribution)

- `upstream_event_ref` — identifier (event_id, job_id, or equivalent
  primary key) for the event/job that triggered the decision.
  Permits reconstructing the upstream pipeline path that led to this
  enforcement.
- `policy_lineage_ref` — identifiers of the policy elements that
  shaped the decision (e.g., `triggered_rules` + `evaluation_hash`
  for PolicyEngine output). Permits attribution of the action to
  specific rules and constraints.
- `engine_lineage` — version of every engine whose output influenced
  the decision. At minimum: `policy_version`,
  `decision_config_version`, and `confidence_config_version`.
  Sufficient for deterministic replay (per A5).

Missing any field is a system failure. Spec-level documents (e.g.,
the PolicyEngine spec, the storage spec) MAY require additional
fields above this floor. Stricter is allowed; weaker is not.

The audit record MUST be append-only. Mutating an existing record
constitutes evidence destruction. A reversal under A6 is recorded as
an *appended* entry, not as a mutation of the original.

## A5 — DETERMINISTIC REPLAY

Given identical inputs and identical config versions, the system
MUST produce identical outputs.

No randomness. No time-of-day effects. No external state read from
inside engine code. No environment-dependent behaviour.

This applies to every pure-function engine in the pipeline
(decision, confidence, policy). I/O necessary for diagnostics
(logging at ERROR for invariant violations) does not affect the
function's return value and therefore does not violate this axiom.
Spec-level documents define the precise I/O envelope per engine.

## A6 — HUMAN REVIEW AUTHORITY

Human review authority supersedes automated enforcement.

Every enforcement action — RESTRICT or TAKEDOWN — MUST remain
reversible by human authority. The system MUST support, at minimum:

- **Appeals** — a path by which a publisher or owner can request
  review of an enforcement action affecting their content.
- **Disputes** — a path by which third parties (rights-holders,
  regulators) can flag a decision for re-evaluation.
- **Legal intervention** — a path by which legal/compliance
  authority can mandate reversal regardless of the automated
  decision pipeline.
- **False-positive correction** — a workflow that overturns an
  incorrect enforcement and feeds the correction back into model
  evaluation.

Reversal MUST update the audit record by *appending* a reversal
entry; the original automated decision is preserved (per A4's
append-only requirement). The reversal entry inherits all A4 fields
plus a `reversal_reason` identifier and the operator/authority that
issued it.

A spec-level document (`docs/specs/policy_engine.md`,
`docs/security/enforcement_audit.md`) defines the reversal record's
exact shape and the surfacing mechanics. The axiom is the floor: no
enforcement action may be permanent and unreversable.

## A7 — EVIDENCE PRESERVATION

Enforcement decisions MUST preserve sufficient evidence for replay
(A5), audit reconstruction (A4), and dispute resolution (A6).

Required evidence elements (in addition to the A4 audit record):

- The input asset's stored representation — raw bytes or
  content-addressed reference per the storage spec. Never deleted on
  enforcement.
- The matched asset's reference at the time of the decision.
- Snapshots of policy and config versions used (the `engine_lineage`
  referenced in A4).
- Sufficient context to recompute the same decision under A5
  determinism (i.e., replay-attributable inputs).

Evidence MUST be:

- **IMMUTABLE** once written.
- **QUERYABLE** — accessible by audit and dispute workflows.
- **LONG-LIVED** — retention policy defined at spec level
  (`docs/specs/storage.md`); deletion before the retention window
  expires constitutes evidence destruction.

Spec-level documents define operational details (storage layer,
retention windows, query APIs). The axiom is the floor: no
enforcement may proceed without the evidence that supports it being
preserved durably and immutably.

---

## Amendment process

See `./GOVERNANCE.md` §4 ("Axiom amendment process"). Summary:

1. Author drafts a Constitutional ADR proposing the amendment.
2. All domain leads review during a 72-hour comment window.
3. Architect sign-off required.
4. Approval is unanimous (no abstentions).
5. On approval, this file is updated, the ADR is linked from the
   amendment site, and a git tag (`axioms-vX.Y`) is created.

## Versioning

This file is versioned in its frontmatter (`version:`).

- A minor bump (`x.y → x.(y+1)`) accompanies a clarification with no
  semantic change.
- A major bump (`x.0 → (x+1).0`) accompanies any axiom add / remove /
  rephrase that changes meaning.

Both kinds of bump require a Constitutional ADR.

The current version is **2.0**. Changes from 1.0:
- A1 rephrased from exact topology to semantic phases (preserves
  ordering guarantee; removes orchestration prescription).
- A4 extended with `upstream_event_ref`, `policy_lineage_ref`,
  `engine_lineage` (the v1 `config_version` is absorbed under
  `engine_lineage`).
- A6 (Human Review Authority) added.
- A7 (Evidence Preservation) added.
- Axiom count: 5 → 7. Budget unchanged at 10.
