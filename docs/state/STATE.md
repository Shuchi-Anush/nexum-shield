---
authority: OPERATIONAL
domain: governance
status: ACTIVE
version: 1.0
owner: architect
supersedes:
  - .claude/memory/implementation_plan.md §4 (working draft, untracked) — adopts the seven-state lifecycle and registry shape; example versions replaced with actual implementation versions
  - .claude/CLAUDE.md CURRENT STATE section (partial — superseded by structured registry below)
  - CLAUDE.md CURRENT STATE section (partial — same)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; to be backfilled)
---

# STATE — Nexum Shield architecture state registry

This file is the single source of truth for the lifecycle state of
every engine, store, queue, contract, and operational component in
the system.

## Authority

This document is **TIER 3 (OPERATIONAL)** per
`docs/constitution/GOVERNANCE.md` §1. State changes are recorded
here via PR review. The PR MUST link to the relevant ADR if the
transition is governed by one (e.g. ACCEPTED → ROLLING_OUT triggered
by a Standard ADR). No ADR is required for component creation in
EXPERIMENTAL or for version-only patch bumps within ACTIVE.

## Lifecycle

```
EXPERIMENTAL → PROPOSED → ACCEPTED → ROLLING_OUT → ACTIVE → DEPRECATED → REMOVED
```

State definitions (per `docs/constitution/GOVERNANCE.md` §1 +
`.claude/memory/implementation_plan.md` §4):

- **EXPERIMENTAL** — lives in `.claude/labs/EXP-NNN/` or is otherwise
  isolated from production. Spec stability EXPERIMENTAL (per
  GOVERNANCE.md §8). Not depended on by production code.
- **PROPOSED** — spec drafted in `docs/specs/`. ADR drafted. Design
  agreed in principle.
- **ACCEPTED** — ADR approved. Implementation may begin; not yet
  running in any environment.
- **ROLLING_OUT** — implementation exists; the prior implementation
  still runs. Both coexist, gated by a config flag.
- **ACTIVE** — sole production path. The prior implementation is
  REMOVED.
- **DEPRECATED** — successor is PROPOSED or ACCEPTED. Existing
  system still works for the compatibility window.
- **REMOVED** — code deleted. Spec archived in git history; this
  registry's row is preserved for replay attribution.

## Current registry

### Engines

| Component | State | Version | Successor | Compat Window | ADR |
|---|---|---|---|---|---|
| PolicyEngine (PBRA) | ACTIVE | v1.0 | — | — | ADR-0001 (bootstrap) |
| ConfidenceEngine | ACTIVE | v1.0 | — | — | — |
| DecisionEngine | ACTIVE | v1.0 | — | — | — |

ConfidenceEngine consumes a threshold config whose schema is
versioned independently (`ConfidenceConfig.config_version = "v3"`
in code). Engine version v1.0 refers to the runnable component;
the config schema version is recorded in the spec
(`docs/specs/confidence_engine.md`, planned).

### Pipeline orchestration

| Component | State | Version | Successor | Compat Window | ADR |
|---|---|---|---|---|---|
| Ingest API (`POST /v1/ingest`) | ACTIVE | v1.0 | — | — | — |
| Jobs API (`GET /v1/jobs/{id}`) | ACTIVE | v1.0 | — | — | — |
| Health API (`GET /v1/health`) | ACTIVE | v1.0 | — | — | — |
| In-memory JobStore | ACTIVE (MVP-only) | v0.1 | persistent JobStore *(TBD per `docs/specs/job_processing.md`)* | until production cutover | — |
| Event bus | EXPERIMENTAL | not built | per A1 mandate + `docs/specs/eventing.md` *(planned)* | — | — |
| Worker fleet | EXPERIMENTAL | not built | per `docs/specs/job_processing.md` *(planned)* | — | — |

The In-memory JobStore is acknowledged in `CLAUDE.md` and
`.claude/CLAUDE.md` as temporary. Per `.claude/rules/storage.md`
§5, in-memory queues are forbidden in production. The component
is therefore ACTIVE in MVP and slated for replacement before any
production rollout. It transitions to DEPRECATED in this registry
when its successor reaches PROPOSED.

### Storage layers (per `docs/specs/storage.md` *(planned)*)

| Component | State | Version | Successor | Compat Window | ADR |
|---|---|---|---|---|---|
| Raw media store | EXPERIMENTAL | DEV: filesystem | PROD: S3 / GCS *(TBD)* | — | — |
| Metadata store | EXPERIMENTAL | DEV: SQLite *(planned)* | PROD: Postgres *(TBD)* | — | — |
| Embeddings store | EXPERIMENTAL | not built | Vector DB (FAISS / Milvus / Pinecone) | — | — |
| Fingerprint store | EXPERIMENTAL | not built | TBD | — | — |
| Job/queue store | EXPERIMENTAL | not built | Redis (queue) + Postgres (durable) | — | — |
| **Evidence store** | **PROPOSED** (A7 mandates) | not built | TBD | — | — |

The Evidence store is **PROPOSED** rather than EXPERIMENTAL because
its existence is constitutional (axiom A7); the design is yet to be
made concrete but the store is non-optional. EXPERIMENTAL would
imply the design might be abandoned; PROPOSED reflects the
axiomatic mandate.

### Removed / superseded

| Component | State | Predecessor | Successor | Removed at | ADR |
|---|---|---|---|---|---|
| 3-level enforcement model (SOFT FLAG / REVIEW / AUTO ENFORCEMENT) | REMOVED | `.claude/rules/enforcement.md` (TRANSITIONAL until rule-file deprecation) | 5-action ladder (`PolicyAction` in `policy_engine.py`) | superseded by PolicyEngine v1.0 (Phase-2 bootstrap) | ADR-0001 |

The 5-action ladder (`ALLOW < FLAG < REVIEW < RESTRICT < TAKEDOWN`)
replaces the 3-level model. The legacy spec file
(`.claude/rules/enforcement.md`) remains in place per the
append-only migration constraint; only the 3-level *model* is
REMOVED — the file itself stays TRANSITIONAL until its
non-obsolete content (audit / dispute / human-in-the-loop) migrates
to `docs/security/enforcement_audit.md` in a later batch, at which
point the rule file gets a deprecation note.

## Cross-component invariants tracked here

- Every ACTIVE component has a corresponding canonical spec, an
  implementation file, and a defined version.
- Every DEPRECATED component has a successor entry (PROPOSED or
  later) and a compat window.
- Every REMOVED entry retains its predecessor / successor links;
  rows are NEVER deleted from this file.
- A PR that moves a component to ACTIVE without first passing
  through ACCEPTED + ROLLING_OUT is a **P1** governance violation
  (per `docs/constitution/GOVERNANCE.md` §5).

## Versioning of this file

- **Patch bump** (e.g. `1.0 → 1.0.1`) — single state change,
  version update of an existing component, successor reassignment.
- **Minor bump** (e.g. `1.0 → 1.1`) — adding a new component row,
  new section.
- **Major bump** (e.g. `1.0 → 2.0`) — schema change to the registry
  table itself (new columns, removed columns). Major bump requires
  Standard ADR.

## Cross-references

- Lifecycle definition: `../constitution/GOVERNANCE.md` §1.
- Spec stability framework (different concept): `../constitution/GOVERNANCE.md` §8.
- Severity model (state-violation classification): `../constitution/GOVERNANCE.md` §5.
- Domain ownership for each component: `../governance/DOMAINS.md`.
- Emergency state changes: `../constitution/GOVERNANCE.md` §7 (EGM).
