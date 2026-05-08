# ENFORCEMENT & LEGAL PIPELINE

## Core Principle
False positives = legal risk

## Enforcement Levels

1. SOFT FLAG
   - internal tracking only

2. REVIEW REQUIRED
   - sent to human moderation

3. AUTO ENFORCEMENT
   - takedown / blocking

## Requirements

### Evidence Storage
Each decision MUST store:
- original asset reference
- matched asset reference
- similarity score
- timestamp
- model version

### Auditability
All actions MUST be:
- reproducible
- explainable

### Human-in-the-loop

Required when:
- confidence is medium
- high-impact content
- uncertain matches

### Dispute Handling

System MUST support:
- appeal requests
- reversal of decisions
- audit trail

## Anti-Patterns (FORBIDDEN)

- irreversible automated enforcement
- no evidence storage
- no audit logs