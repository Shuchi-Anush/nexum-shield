---
authority: AXIOM
domain: governance
status: ACTIVE
version: 2.0
owner: architect
supersedes:
  - .claude/memory/implementation_plan.md §2 (ADR tiers), §3 (labs concept), §4 (state lifecycle), §5 (severity model), §6 (domain registrar) — working draft, untracked
  - .claude/rules/context-governance.md (constitutional parts; operational parts move to docs/governance/ai_agent_policy.md)
  - .claude/CLAUDE.md WHEN BUILDING FEATURES section (partial — plan-mode requirement)
  - CLAUDE.md WHEN BUILDING FEATURES section (partial — same)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; constitutional v1→v2 amendment; to be backfilled)
---

# GOVERNANCE — Nexum Shield

This file carries the constitutional rules that govern governance
itself: authority precedence, conflict resolution, ADR tiers, axiom
amendment, severity classification, and the domain registrar pattern.

It is subordinate only to `./AXIOMS.md`. Like AXIOMS.md, it is
modifiable only via Constitutional ADR.

Documents and registries that change via PR review (DOMAINS.md,
coding standards, AI-agent policy, severity escalation runbooks) live
in `../governance/` — not here.

---

## §1 Authority precedence

Six tiers, highest authority first. Higher tiers override lower; the
constitution overrides every spec, every rule, every agent
instruction, and every ADR.

| Tier | Name | Members | Modification |
|------|------|---------|--------------|
| 1 | AXIOMATIC | `docs/constitution/AXIOMS.md`, `docs/constitution/GOVERNANCE.md` | Constitutional ADR only |
| 2 | SPEC | `docs/specs/*.md` | Standard or Lightweight ADR (per §3 below); domain lead owns |
| 3 | OPERATIONAL | `docs/state/STATE.md`, `docs/governance/*.md`, `docs/security/*.md`, `docs/testing/*.md` | PR review by owning role (architect, security lead, qa lead) |
| 4 | RECORD | `docs/adr/ADR-NNNN-*.md` | immutable once accepted |
| 5 | TRANSITIONAL | `.claude/CLAUDE.md`, `.claude/rules/*.md`, `CLAUDE.md` (root) | currently authoritative pre-Phase-2; superseded as Tier-2 specs land |
| 6 | REFERENCE | `.claude/agents/*.md`, `.claude/skills/*.md`, `.claude/archive/**` | informational; no authority |

**Untracked, never authoritative**: `.claude/memory/**`,
`.claude/claude_sessions/**`, `.claude/labs/**`,
`.claude/settings*.json`. These are working cognition; canonical
knowledge migrates from these locations into Tiers 1–4 before
becoming authoritative.

### Tier-5 demotion schedule

Documents in TRANSITIONAL (`.claude/rules/*.md`,
`.claude/CLAUDE.md`, root `CLAUDE.md`) are currently authoritative
because the Phase-2 specs that supersede them are not yet written.
As each Tier-2 spec lands and is referenced from this file's
`supersedes:` chain, the corresponding Tier-5 source MUST be
annotated with a `superseded by:` deprecation note pointing to the
new spec. The Tier-5 source is NOT deleted; it stays in place for
historical reference.

---

## §2 Conflict resolution

### Cross-tier conflicts

The higher tier wins automatically. A spec cannot override an axiom;
an ADR cannot override a spec. Authors of lower-tier documents are
responsible for harmonising with higher-tier statements; if they
cannot, the conflict is resolved by amending the higher-tier
document (which requires the higher-tier modification process —
e.g., a Constitutional ADR if the higher tier is AXIOMATIC).

### Same-tier conflicts

When two documents in the same tier disagree, the resolution rule is:

1. **ADR is required.** Same-tier conflicts MUST be resolved by an
   ADR appropriate to the tier (Standard for Tier 2, PR-review +
   record for Tier 3).
2. **Owning domain authority decides.** Each spec, registry, and
   policy document declares its owning domain in frontmatter (`owner:`
   field). The owning domain's lead has decision authority within
   the domain. If the conflict crosses domains, the relevant leads
   convene; if they cannot agree, the architect breaks the tie via a
   Standard ADR.
3. **Timestamp is not authority.** "More recently modified" or
   "more recently versioned" MUST NOT establish semantic precedence.
   If two contemporaneous documents disagree, an ADR is still
   required. Dates are governance metadata, not authority.

### Documenting the resolution

The ADR that resolves a same-tier conflict MUST list both
conflicting documents in its `Context` section, name the chosen
authority in `Decision`, and update both source documents'
`supersedes:` and `version:` fields atomically with the ADR's
acceptance.

---

## §3 ADR tier model

Three tiers. All three share one numbering sequence
(`ADR-NNNN-short-kebab-title.md`).

### Lightweight ADR

```
When:        Single-spec change within one domain. No cross-domain
             impact. Examples: threshold tweak, new uncertainty term,
             enum extension within the domain that owns the enum.
Approval:    Domain lead self-approves (or peer if solo).
Template:    one page (Title, Status, Date, Author, Type=LIGHTWEIGHT,
             Spec, Version bump, Change, Impact).
```

### Standard ADR

```
When:        Cross-spec change, new engine, new data contract,
             execution-model refinement, same-tier conflict
             resolution.
Approval:    Domain lead + one reviewer (architect or peer lead).
Template:    Full (Context, Decision, Consequences, Spec
             References, Rollback Plan).
```

### Constitutional ADR

```
When:        Axiom add/remove/rephrase, governance rule change, tier
             model change, base-matrix change in PolicyEngine,
             severity-classification change.
Approval:    All domain leads + architect. Unanimous. No abstentions.
             72-hour comment window before approval.
Template:    Full + Impact Analysis + Migration Plan + Rollback Plan.
             Status line tagged CONSTITUTIONAL.
```

### Decision tree

```
Does it modify an axiom or a governance rule?
   YES → Constitutional
Does it affect more than one spec file or cross domain boundaries?
   YES → Standard
Otherwise (single-domain, single-spec)
        → Lightweight
```

### ADR lifecycle states

```
DRAFT → IN_REVIEW → ACCEPTED → (rarely) SUPERSEDED
```

`ACCEPTED` ADRs are immutable. A later decision that overturns an
earlier ADR creates a new ADR with `Supersedes: ADR-NNNN`; the
earlier ADR's status flips to `SUPERSEDED` but its content is not
edited.

---

## §4 Axiom amendment process

To **modify** an axiom in `./AXIOMS.md`:

1. Author drafts a Constitutional ADR proposing the amendment.
2. The ADR enters a 72-hour comment window. All domain leads MUST
   review.
3. Architect sign-off required.
4. Approval is unanimous; no abstentions are allowed.
5. On approval, `AXIOMS.md` is updated atomically with the ADR's
   acceptance. Version bumps minor (`1.0 → 1.1`) for clarification
   and major (`1.0 → 2.0`) for any meaning change. A git tag
   (`axioms-vX.Y`) is created on the merge commit.

To **add** an axiom: same process. The total count MUST NOT exceed
10. If the budget is full, an existing axiom MUST be deprecated in
the same Constitutional ADR.

To **remove** an axiom: same process. The ADR MUST document why the
axiom is no longer load-bearing and what (if anything) covers the
gap.

The same process applies to this file (`GOVERNANCE.md`).

---

## §5 Severity model

Four levels. All checks (CI gates, runtime invariants, lint,
process audits) classify into exactly one level.

| Level | Name | CI behaviour | Response time | Examples |
|---|---|---|---|---|
| **P0** | Critical block | Build fails. Cannot merge. | Immediate. | Axiom violation, invariant test failure, terminal-invariant guard fired at runtime, `rules_checked_count` mismatch, constitution content deleted |
| **P1** | Standard block | Build fails. Cannot merge. | Before merge. | Spec hash mismatch (impl ↔ spec), enum mismatch between spec and code, missing spec hash, `.claude/labs/` imported by `backend/`, `STATE.md` reference unresolved |
| **P2** | Warning | Build passes with warning. | Within sprint. | Spec changed without ADR reference in commit, experiment past TTL, deprecation note missing on superseded source |
| **P3** | Informational | Build passes. Logged only. | Best effort. | Session log size growing, unused rejection record, stale README phase reference |

### Architectural vs operational violations

```
ARCHITECTURAL VIOLATION:
  Contradicts a spec or axiom.
  Always P0 or P1.
  Examples: wrong enum value, missing audit field,
            non-deterministic engine output.
  Response: fix code or amend spec (with appropriate ADR).

OPERATIONAL VIOLATION:
  Contradicts a governance process.
  Usually P2 or P3.
  Examples: missing ADR for a spec change, expired
            experiment, stale session log.
  Response: create ADR, archive experiment, clean up.
  Operational violations do NOT permanently block the build.
  They escalate from P2 → P1 after a grace period (default 14
  days from first detection).
```

---

## §6 Domain registrar pattern

Each spec, registry, and policy document declares an `owner:` field
naming a domain (e.g. `policy`, `confidence`, `decision`,
`pipeline`, `api`, `platform`, `security`, `qa`, `governance`).
Every domain has one **lead** and one **backup**, listed in
`docs/governance/DOMAINS.md` (next batch).

### Authority delegation

A **domain lead** MAY:
- Self-approve Lightweight ADRs within their domain.
- Bump spec minor versions within their domain.
- Accept or reject experiments scoped to their domain.
- Approve implementation PRs in their domain.

A **domain lead** MAY NOT:
- Modify specs outside their domain.
- Approve Standard ADRs that cross domain boundaries
  (those need affected domain leads + architect).
- Modify axioms or governance rules.
- Change enum definitions used outside their domain. For example,
  `PolicyAction` is owned by the `policy` domain but consumed by
  `pipeline`, so changes require a Standard ADR with `pipeline`
  reviewing.

Cross-domain changes require:
- A Standard ADR (or Constitutional, if it touches axioms / governance).
- Review by all affected domain leads.
- Architect tiebreaker if leads disagree.

### Architect role

The architect is NOT a gatekeeper for:
- Within-domain spec changes (domain lead owns).
- Lightweight ADRs (self-approving).
- Implementation PRs (domain lead + peer review).
- Experiment creation (anyone in the domain).

The architect IS required for:
- Constitutional ADRs (all-hands).
- Cross-domain Standard ADRs (tiebreaker).
- New domain creation (registrar update + Constitutional ADR if it
  changes the domain set materially).
- Axiom budget management.
- Same-tier conflict resolution when leads cannot agree.

### AI agents

AI agents operate WITHIN a domain under that domain's lead. They are
never themselves domain leads. They follow the same rules as human
engineers:

- Read `./AXIOMS.md` and the relevant spec(s) before working.
- Implement within spec boundaries.
- Flag contradictions; do not silently resolve them.
- Create experiments in `.claude/labs/`, never in production paths.

The detailed AI-agent operating policy (when to ask, when to assume,
plan-mode requirement, session-log handling) lives in
`docs/governance/ai_agent_policy.md` (later batch). This file
carries only the constitutional bound: the same authority limits
that apply to human contributors apply to agents.

---

## §7 Emergency Governance Mode

When a production incident requires immediate change to specs,
axioms, or enforcement code that would otherwise require a Standard
or Constitutional ADR, **Emergency Governance Mode (EGM)** provides
a controlled escape hatch. EGM is a scheduling accelerant, not an
authority bypass.

### Activation

EGM is activated by:
- The architect (default trigger), or
- A designated emergency lead listed in
  `docs/governance/DOMAINS.md` (e.g., on-call security lead during
  after-hours incidents).

Activation requires:
- An `incident_id` linking to the active incident (in the incident
  tracker; if none exists, an entry in
  `docs/governance/incidents/INCIDENT-NNNN.md`).
- A scoped declaration listing which specs / axioms / files are
  touched.
- A wall-clock expiry timestamp (default 72 hours; maximum 7 days).

### Authority during EGM

Within the declared scope:
- Domain leads MAY make changes that would normally require Standard
  or Constitutional ADRs.
- Changes MUST be committed with a commit-message tag
  `emergency: <incident_id>` so they are discoverable in history.
- Terminal invariant guards (axiom-enforcement code such as the
  PolicyEngine post-phase guard) remain active. EGM does NOT bypass
  A2, A3, A6, or A7 — it accelerates the approval process for
  changes that remain within the bounds of those axioms.

### Hard limits

EGM CANNOT:
- Modify A1–A7 directly. Axiom modification ALWAYS requires a
  Constitutional ADR. Emergencies grant scheduling flexibility, not
  bypass of the amendment process.
- Delete an axiom or remove a tier from the authority hierarchy.
- Permanently disable invariant CI checks. Temporary disable for
  the duration of EGM is permitted with explicit logged
  justification.
- Be invoked retroactively to legitimise prior changes. EGM is
  declared *before* changes are made; the declaration creates the
  authority window.

### Expiry and rollback

When the expiry timestamp passes (or earlier, when the incident is
resolved):

- A **retroactive ADR** MUST be filed for every change made under
  EGM, using the template appropriate to the change's normal
  classification (Lightweight / Standard / Constitutional).
- The retroactive ADR's `Context` section MUST cite the
  `incident_id` and link to the incident report.
- The retroactive ADR's `Rollback Plan` section MUST be concrete
  (steps, not intent) since the change is already in production.
- If the retroactive ADR is **rejected**, the change MUST be rolled
  back per its Rollback Plan. There is no "we'll fix it later"
  exemption: rejection is binding.

### Incident linkage and audit

Every EGM activation produces an entry in
`docs/governance/incidents/INCIDENT-NNNN.md` recording:

- Incident summary and trigger.
- Activation time, declared expiry, actual close time.
- All commits / files / specs touched under the EGM tag.
- Retroactive ADR identifiers and their statuses.

EGM activations MUST be visible to all domain leads within 24 hours
of declaration via the incident tracker or designated channel.
**Silent EGMs are forbidden.**

---

## §8 Spec stability classifications

Every Tier-2 spec (and any Tier-3 document treated as a contract by
downstream consumers) declares a `stability:` field in its
frontmatter. The classification governs compatibility expectations
and the bar for change.

### Levels

- **EXPERIMENTAL** — captures a hypothesis. No compatibility
  guarantees. Implementation may diverge from spec without ADR.
  Consumers MUST NOT depend on the spec's contents in production.
  Default for any document under `.claude/labs/EXP-NNN/`.

- **EVOLVING** — actively iterating. Minor changes (additive fields,
  threshold adjustments) are expected. Consumers should expect
  change at every minor version bump (`x.y → x.(y+1)`). No backward-
  compatibility guarantee across minor bumps. Default for new specs
  in `docs/specs/` until they reach STABLE.

- **STABLE** — in steady production use. Backward-compatible changes
  only. Breaking changes require a deprecation cycle with at least
  one minor-version compatibility window before removal. Consumers
  can rely on the spec's invariants across minor versions. Reaching
  STABLE requires a Standard ADR confirming production readiness.

- **LOCKED** — frozen. Any change requires a Constitutional ADR.
  Used for specs that other specs depend on heavily and where drift
  would cascade (e.g., the `PolicyAction` enum once the enforcement
  system is live across all integrations).

### Transitions

```
EXPERIMENTAL → EVOLVING:    Lightweight ADR. Spec exits .claude/labs/.
EVOLVING     → STABLE:      Standard ADR. Domain lead + one reviewer.
STABLE       → LOCKED:      Constitutional ADR. Architect + all leads.
STABLE       → EVOLVING:    Standard ADR — used when material change
                            breaks the STABLE contract; domain lead
                            must justify why EVOLVING is preferred
                            over a deprecation cycle.
LOCKED       → STABLE:      Constitutional ADR (rare).
EXPERIMENTAL → REJECTED:    documented in `.claude/archive/rejected/`.
                            No ADR required for rejection of
                            experimental work.
```

### Compatibility expectations

| Stability | Breaking changes | ADR for change | Consumer guarantee |
|---|---|---|---|
| EXPERIMENTAL | unrestricted | none required | none |
| EVOLVING | allowed at minor bumps | Lightweight | low — expect churn |
| STABLE | deprecation cycle required | Standard | minor versions are compatible |
| LOCKED | none without Constitutional ADR | Constitutional | absolute (until LOCK is lifted) |

### Default at creation

A new spec lands as **EVOLVING** unless:
- It is created inside `.claude/labs/` → **EXPERIMENTAL**.
- It is explicitly justified at LOCKED creation (extremely rare;
  e.g., a frozen API contract for an external partner) → **LOCKED**.

### Independence from version

The stability level is independent of `version:`. A spec at version
2.3 may still be EVOLVING; another at 1.0 may already be LOCKED if
it is a frozen contract. Stability speaks to *what consumers can
rely on*; version speaks to *what changed*.

---

## Cross-references

- Authority hierarchy is materialised in `docs/README.md` (the index)
  and in this §1.
- Domain table: `docs/governance/DOMAINS.md` (next batch).
- Architecture state registry: `docs/state/STATE.md` (next batch).
- ADR templates: `docs/adr/_template_*.md` (Phase 3).
- Severity escalation runbooks (P0–P3 response procedures):
  `docs/governance/severity_runbook.md` (later batch).
- Operational AI-agent policy: `docs/governance/ai_agent_policy.md`
  (later batch).
