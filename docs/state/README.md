---
authority: SCAFFOLD
domain: governance
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/state/

**Phase-2 destination**: `STATE.md` — architecture state registry.

## Purpose

Single source of truth for which components are live, deprecated, or
removed. Every component travels the seven-state lifecycle defined in
`.claude/memory/implementation_plan.md` §4:

```
EXPERIMENTAL → PROPOSED → ACCEPTED → ROLLING_OUT → ACTIVE → DEPRECATED → REMOVED
```

`STATE.md` records the current state, version, successor (if any),
compatibility window, and ADR for each component.

## Why separate from docs/specs/

A spec describes one component; `STATE.md` is a cross-component
registry. Keeping them separate avoids circular ownership: spec
authors set version, but `STATE.md` authority belongs to the
architect / release process.

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: `STATE.md` populated for the live components
  (DecisionEngine v1.0, ConfidenceEngine v1.2, PolicyEngine PBRA v2.x,
  legacy 3-action model marked REMOVED).
- **Phase 5**: state-validation stub asserts `STATE.md` references
  resolve to actual files / git tags.
