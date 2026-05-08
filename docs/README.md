---
authority: SCAFFOLD
domain: governance
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/

Repository documentation root. Authority precedence (highest to lowest):

1. `docs/constitution/AXIOMS.md` and `docs/constitution/GOVERNANCE.md`
2. `docs/specs/*.md` (per-engine specifications)
3. `docs/state/STATE.md` (architecture state registry)
4. `docs/security/*`, `docs/testing/*`, `docs/governance/*`
5. `docs/adr/ADR-*.md` (records of decisions, not authority)

`.claude/rules/*.md` and `.claude/memory/*.md` remain temporary
authorities until Phase 2 migration completes; they will be
re-anchored to specs in this tree.

## Directories

| Path | Purpose | Phase |
|---|---|---|
| `constitution/` | axioms + governance rules | 1 (scaffold), 2 (content) |
| `specs/` | per-engine specifications | 1 (scaffold), 2 (content) |
| `state/` | architecture state registry (STATE.md) | 1 (scaffold), 2 (content) |
| `governance/` | registries, ownership, runbooks | 1 (scaffold), 2 + 4 (content) |
| `security/` | security policy | 1 (scaffold), 2 (content) |
| `testing/` | test strategy / invariant catalogue | 1 (scaffold), 2 + 5 (content) |
| `adr/` | ADRs (Lightweight / Standard / Constitutional) | 1 (scaffold), 2 / 3 (content) |
| `architecture/` | **legacy** — to be reconciled into `specs/` in Phase 2 | — |
| `canonical/` | **legacy** — to be reconciled into `constitution/` in Phase 2 | — |
| `decisions/` | **legacy** — to be reconciled into `adr/` in Phase 2 | — |
| `api/` | **legacy** — to become `specs/api_contracts.md` in Phase 2 | — |
| `invariants/` | **legacy** — invariants live with specs per plan §1 | — |
| `operations/` | **legacy** — not in plan; Phase 2 will redirect or remove | — |

`legacy` = pre-Phase-1, kept untouched per the append-only migration
constraint. Contents are not authoritative until Phase 2 reconciles
them.

## Metadata convention

Every canonical document in this tree carries a five-field frontmatter:

```
---
authority: AXIOM | SPEC | OPERATIONAL | RECORD | SCAFFOLD | LEGACY
domain: policy | confidence | decision | pipeline | api | platform | governance | security | testing
status: DRAFT | PROPOSED | ACCEPTED | ROLLING_OUT | ACTIVE | DEPRECATED | REMOVED | SCAFFOLD
version: x.y
supersedes: <path>@<git-sha> | (none)
---
```

Phase 1 establishes the convention via these scaffold READMEs. Phase 2
materialises the first canonical documents.
