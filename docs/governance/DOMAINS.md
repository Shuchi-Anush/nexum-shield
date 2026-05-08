---
authority: OPERATIONAL
domain: governance
status: ACTIVE
version: 1.0
owner: architect
supersedes:
  - .claude/memory/implementation_plan.md §6 domain registrar (working draft, untracked) — adds 3 domains (security, qa, governance) and refines scopes
adr_references:
  - ADR-0001 (Phase-2 bootstrap; to be backfilled)
---

# DOMAINS — Nexum Shield

This file is the domain registrar. Every Tier-2 spec, every
operational record, and every ADR declares an owning domain via the
`owner:` frontmatter field. This file maps domains to leads,
backups, and primary scope.

## Authority

This document is **TIER 3 (OPERATIONAL)** per
`docs/constitution/GOVERNANCE.md` §1. Lead/backup assignments change
via PR review; no ADR required. Adding or removing a domain requires
a Constitutional ADR (per `docs/constitution/GOVERNANCE.md` §6 —
architect required, all leads notified).

## Domain table

| Domain | Lead | Backup | Primary scope |
|---|---|---|---|
| **governance** | architect | (designate) | constitution, ADR process, conflict resolution, EGM, stability framework, registries |
| **policy** | (unassigned) | (unassigned) | PolicyEngine, PBRA execution model, action ladder, safety + risk-control rules |
| **confidence** | (unassigned) | (unassigned) | ConfidenceEngine, agreement / completeness / uncertainty composition, tier thresholds |
| **decision** | (unassigned) | (unassigned) | DecisionEngine, risk formula, trust reader, threshold config, risk_assessment, content_similarity |
| **pipeline** | (unassigned) | (unassigned) | event-driven backbone, job lifecycle and state, queue topology, retry / DLQ |
| **api** | (unassigned) | (unassigned) | request contracts, response schemas, versioning policy |
| **platform** | (unassigned) | (unassigned) | infrastructure, deployment, observability, storage architecture |
| **security** | (unassigned) | (unassigned) | threat model, secrets handling, enforcement audit, dispute mechanics |
| **qa** | (unassigned) | (unassigned) | test strategy, invariant catalogue, replay / determinism testing |

## Primary spec ownership

Maps each domain to the canonical Tier-2 / Tier-3 docs that report
up to it. A spec that is not yet materialised is shown as
*(planned)* — it exists in the migration plan but its file is not
yet written.

| Domain | Owns specs |
|---|---|
| governance | `docs/constitution/AXIOMS.md`, `docs/constitution/GOVERNANCE.md`, `docs/governance/DOMAINS.md`, `docs/state/STATE.md`, *(planned)* `docs/governance/ai_agent_policy.md`, `docs/governance/coding_standards.md`, `docs/governance/severity_runbook.md` |
| policy | `docs/specs/policy_engine.md` |
| confidence | `docs/specs/confidence_engine.md` |
| decision | *(planned)* `docs/specs/decision_engine.md`, *(planned)* `docs/specs/risk_assessment.md`, *(planned)* `docs/specs/content_similarity.md` |
| pipeline | *(planned)* `docs/specs/eventing.md`, *(planned)* `docs/specs/job_processing.md` |
| api | *(planned)* `docs/specs/api_contracts.md` |
| platform | *(planned)* `docs/specs/storage.md`, *(planned)* `docs/specs/observability.md` |
| security | *(planned)* `docs/security/secrets_policy.md`, *(planned)* `docs/security/enforcement_audit.md` |
| qa | *(planned)* `docs/testing/INVARIANT_TESTS.md` |

`risk_assessment.md` and `content_similarity.md` (renamed from
`scoring_engine` and `matching_engine` respectively) are
decision-domain specs because their outputs (risk score, similarity
decision) are direct inputs to DecisionEngine and PolicyEngine.
Implementation files for matching, scoring, fingerprinting, and
embedding may live under `backend/app/engines/` or be distributed
across modules; ownership is determined by the spec, not the file
path.

## Cross-domain components

Some artefacts are owned by one domain but consumed by many.
Cross-domain change requires a Standard ADR (per
`docs/constitution/GOVERNANCE.md` §3 and §6).

| Component | Owning domain | Consumers |
|---|---|---|
| `PolicyAction` enum | policy | api, pipeline, security |
| `ConfidenceTier` enum | confidence | policy, qa |
| `RiskBand` enum | decision | policy, qa |
| Audit-record schema (per A4) | governance + policy + security | platform (storage), qa (replay tests) |
| Job state machine (`QUEUED → PROCESSING → ...`) | pipeline | api, qa |

## Lead and backup assignment

### Assignment

A new lead or backup is assigned via PR review. The PR:

1. Updates the Domain table above.
2. Bumps the `version:` field by patch (e.g. `1.0 → 1.0.1`).
3. References the assignee by their identity in the project's
   contributor system (e.g. GitHub username).
4. Requires architect approval.

No ADR is required for individual assignment changes — the table
itself is the record.

### Vacancy and interim authority

A `(unassigned)` cell means the architect operates as **interim
lead** for that domain. Per `docs/constitution/GOVERNANCE.md` §6,
the architect's authority is plenary until delegated. Interim
authority is real but not preferred; assigning a permanent lead
reduces architect load and improves domain expertise.

The `governance` domain's lead is **always the architect** — this
is constitutional, not assignable. The backup slot is the only
configurable cell for `governance`.

### Backup obligation

Per `docs/constitution/GOVERNANCE.md` §6, a domain lead taking
leave exceeding 5 working days MUST have an assigned backup.
Vacation coverage without a backup defaults the lead's authority
to the architect for the duration of the leave.

## AI agent envelope

Agents in `.claude/agents/*.md` (TIER 6 REFERENCE) operate WITHIN
domains under the domain lead's authority. They are never
themselves domain leads — this is constitutional
(`docs/constitution/GOVERNANCE.md` §6).

The detailed operating policy — which agent operates in which
domain, prompt-shape requirements, session-log discipline, plan-mode
requirements — lives in `docs/governance/ai_agent_policy.md` (later
batch). This file records only the constitutional bound: agents
inherit the domain lead's authority limits and cannot exceed them.

## Adding a domain

Domains exist to scope authority. Adding one is heavyweight:

1. Constitutional ADR proposing the domain, its scope, the specs
   it would own, and any specs it would absorb from another domain.
2. Architect approval required (per
   `docs/constitution/GOVERNANCE.md` §6).
3. All existing leads notified during the 72-hour comment window.
4. On approval: the new row is added to the Domain table, the new
   domain's first lead is assigned in the same PR, and the
   `version:` field bumps minor (e.g. `1.x → 1.(x+1)`).

## Removing a domain

Same process. The ADR MUST document where the removed domain's
specs go (which other domain absorbs them) and what happens to its
lead (typically reassigned to the absorbing domain).

## Cross-references

- Tier hierarchy: `../constitution/GOVERNANCE.md` §1.
- Lead authority and limits: `../constitution/GOVERNANCE.md` §6.
- Architecture state by component: `../state/STATE.md`.
- Operational AI-agent policy: `./ai_agent_policy.md` *(planned)*.
- Severity escalation runbooks: `./severity_runbook.md` *(planned)*.
