---
authority: SCAFFOLD
domain: governance
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/specs/

**Phase-2 destination**: per-engine canonical specifications.

## Purpose

Implementation-specific specifications. Each spec is owned by one
domain lead and is subordinate to `docs/constitution/AXIOMS.md`.

Every spec carries the five-field frontmatter (see `docs/README.md`)
and references its predecessor via `supersedes: <path>@<git-sha>`.

## Planned files (Phase 2)

| File | Supersedes |
|---|---|
| `policy_engine.md` | `.claude/memory/policy_engine_spec_v[1-5].md` + the two zero-byte placeholders in `docs/architecture/` |
| `confidence_engine.md` | `.claude/memory/confidence_engine_spec.md` |
| `decision_engine.md` | (no predecessor — fresh canonical) |
| `api_contracts.md` | `.claude/rules/api_contracts.md` + `docs/api/` |

## What does NOT live here

- Axioms — `docs/constitution/AXIOMS.md`.
- Architecture state registry — `docs/state/STATE.md`.
- Decision records — `docs/adr/`.
- Domain ownership table — `docs/governance/DOMAINS.md`.

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: per-engine canonical specs.
- **Phase 5**: spec-hash verification stub gates implementation/spec
  drift in CI.
