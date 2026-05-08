---
authority: SCAFFOLD
domain: governance
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/adr/

Architecture Decision Records. Three tiers per
`.claude/memory/implementation_plan.md` §2, sharing one numbering
sequence (`ADR-NNNN-short-kebab-title.md`).

## Tier matrix

| Tier | When | Approval | Template |
|---|---|---|---|
| **Lightweight** | single-spec change, single domain | author self-approves (peer if solo) | one page |
| **Standard** | cross-spec, new engine, new contract | domain lead + one reviewer | full (Context / Decision / Consequences / Spec refs) |
| **Constitutional** | axiom or governance change | unanimous, all domain leads, 72-hour window | full + Impact Analysis + Rollback Plan |

## Tier decision

```
Does it change an axiom or governance rule? → Constitutional
Does it affect more than one spec file?     → Standard
Otherwise (single domain, single spec)      → Lightweight
```

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: backfill ADRs for already-made decisions
  (PBRA execution model, 5-action ladder, FeasibleBounds, R2 STRONG
  exception).
- **Phase 3**: ADR templates land here as
  `_template_lightweight.md`, `_template_standard.md`,
  `_template_constitutional.md`.
- `docs/decisions/` (currently empty) will be reconciled here in
  Phase 2 — either redirect or removal.
