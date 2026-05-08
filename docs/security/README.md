---
authority: SCAFFOLD
domain: security
status: PHASE_1
version: 0.1
supersedes: (none)
---

# docs/security/

Security policy documents that are not axiomatic but are mandatory.

## Distinction from docs/constitution/

- Security axioms (e.g., "every enforcement decision must be
  auditable") belong in `AXIOMS.md`.
- Specifics (threat model, secret-handling, KMS keys, dispute / audit
  storage) live here.

## Planned content (Phase 2)

| File | Supersedes |
|---|---|
| `threat_model.md` | (new) |
| `secrets_policy.md` | `.claude/rules/security.md` |
| `enforcement_audit.md` | enforcement-relevant parts of `.claude/rules/enforcement.md` |
| `incident_response.md` | (new) |

## Migration status

- **Phase 1 (now)**: directory + this README.
- **Phase 2**: migrate `.claude/rules/security.md` and
  enforcement-audit-relevant parts of `.claude/rules/enforcement.md`.
  The original `.claude/rules/*` files remain in place during the
  reconciliation window.
