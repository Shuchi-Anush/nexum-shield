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

## Canonical specs (Phase 2 — landed)

| File | Domain | Status | Supersedes |
|---|---|---|---|
| `policy_engine.md` | policy | ACTIVE v1.0 (EVOLVING) | `.claude/memory/policy_engine_spec_v[1-5].md` + the two zero-byte placeholders in `docs/architecture/` |
| `confidence_engine.md` | confidence | ACTIVE v1.0 (EVOLVING) | `.claude/memory/confidence_engine_spec.md` |
| `decision_engine.md` | decision | ACTIVE v1.0 (EVOLVING) | (no predecessor — fresh canonical) |
| `eventing.md` | pipeline | ACTIVE v1.0 (EVOLVING) | `.claude/rules/eventing.md` (partial) |
| `job_processing.md` | pipeline | ACTIVE v1.0 (EVOLVING) | `.claude/rules/job-processing.md` + `.claude/rules/job_system.md` (full) |

## Planned files (Phase 2 — pending)

| File | Domain | Supersedes |
|---|---|---|
| `api_contracts.md` | api | `.claude/rules/api_contracts.md` + `docs/api/` |
| `storage.md` | platform | `.claude/rules/storage.md` |
| `observability.md` | platform | `.claude/rules/observability.md` |
| `risk_assessment.md` | decision | (overlap with `decision_engine.md` resolved per `decision_engine.md` §13.4) |
| `content_similarity.md` | decision | (produces `match.similarity` consumed by `decision_engine.md`) |
| `trust_reader.md` | decision | (produces trust signals consumed by `decision_engine.md` and `confidence_engine.md`) |

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
