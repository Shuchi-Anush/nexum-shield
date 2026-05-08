---
authority: SCAFFOLD
domain: governance
status: PHASE_1
version: 0.1
supersedes: (none)
---

# .claude/archive/

**Historical / rejected zone.** Authority is zero — never re-imported
by production code or by `.claude/labs/`.

## What goes here

- Graduated experiments (kept as reference for "this was tried, this
  is what we learned").
- Expired experiments past TTL (status `EXPIRED`).
- Rejected designs in `rejected/REJ-NNN-short-name.md` per
  implementation_plan §3.
- Superseded specs once they are referenced by SHA from a canonical
  spec in `docs/specs/`. Until then they remain in their original
  location (`.claude/memory/`) — the move happens after Phase 2's
  canonical specs are landed.

## Why kept separate from .claude/labs/

`.claude/labs/` = active. `.claude/archive/` = inactive. Keeping them
separate makes "what's currently being explored" obvious at a glance.

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: once `docs/specs/policy_engine.md` is canonical, the
  `policy_engine_spec_v[1-5].md` series may be moved here from
  `.claude/memory/`, with each pinned by git SHA in the canonical
  spec's `supersedes:` frontmatter.
- **Phase 3**: rejected-design template lands as
  `rejected/_template.md`. Backfill REJ-001..REJ-004 from
  implementation_plan §3.
