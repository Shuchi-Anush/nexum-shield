---
authority: SCAFFOLD
domain: testing
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/testing/

Test *strategy* and *catalogue*, not test code. Test code lives in
`backend/tests/` (currently empty per `CLAUDE.md`).

## Planned content

- `INVARIANT_TESTS.md` — catalogue of tests that enforce each
  axiom in `docs/constitution/AXIOMS.md`. One axiom may map to
  many tests; every axiom maps to at least one.
- `determinism_policy.md` — replay test methodology (per A5).
- `taxonomy.md` — smoke vs. regression vs. property-test
  classification, with policy on which engines require which.
- `coverage_policy.md` — minimum coverage by domain, exemption
  process.

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: `INVARIANT_TESTS.md` first cut.
- **Phase 5**: spec-hash + invariant validation stubs produce
  CI gates that reference this catalogue.
