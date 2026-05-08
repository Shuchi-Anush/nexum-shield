---
authority: SCAFFOLD
domain: governance
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/constitution/

**Phase-2 destination**: `AXIOMS.md` and `GOVERNANCE.md` (two files only).

## Purpose

Houses statements that, if violated, make the system fundamentally
unsafe or unauditable. Content blueprint per
`.claude/memory/implementation_plan.md` §1, §2, §5, §6.

- `AXIOMS.md` — at most 10 axioms (initial target: 5: pipeline
  immutability, match prerequisite, confidence-gated enforcement,
  audit completeness, deterministic replay).
- `GOVERNANCE.md` — ADR tier model, axiom amendment process,
  P0–P3 severity model, domain-registrar pattern.

If a statement is implementation-specific or could be re-expressed
under a different execution model, it does NOT belong here — it
belongs in `docs/specs/`.

## Authority

`AXIOMS.md` overrides every spec, every `.claude/rules/*` file,
every agent instruction, and every ADR. Modifying an axiom requires
a Constitutional ADR (unanimous, all domain leads, 72-hour window).

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: materialise `AXIOMS.md` and `GOVERNANCE.md` from
  `.claude/memory/implementation_plan.md` §1 / §2 / §5 / §6.
- `docs/canonical/` (currently empty) will be reconciled here in
  Phase 2 — either redirect or removal.
