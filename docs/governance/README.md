---
authority: SCAFFOLD
domain: governance
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/governance/

Operational governance: registries, ownership maps, severity
escalation runbooks, and process documents that don't fit in
`docs/constitution/GOVERNANCE.md` (which carries only the immutable
governance *rules*).

## Distinction from docs/constitution/GOVERNANCE.md

- `docs/constitution/GOVERNANCE.md` — the rules that govern governance
  (ADR tiers, axiom amendment process, severity model). Constitutional;
  changeable only via Constitutional ADR.
- `docs/governance/` — the *operational* layer: who owns what, where
  to escalate, how registries are kept fresh. Changeable via PR review.

## Planned content

- `DOMAINS.md` — domain registrar table per implementation_plan §6
  (policy / confidence / decision / pipeline / api / platform / etc.,
  each with lead + backup).
- Severity escalation runbooks (P0 / P1 / P2 / P3 response procedures).
- Spec / capability / experiment / invariant registries land here in
  Phase 4.

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: `DOMAINS.md` first cut.
- **Phase 4**: registries (capability, domain, spec, invariant,
  experiment).
