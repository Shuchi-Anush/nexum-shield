# CONTEXT GOVERNANCE (ANTI-DRIFT CONTROL)

## Core Principle
Claude MUST follow architecture — not invent it.

## Rules

- NEVER introduce new architecture without justification
- ALWAYS follow:
  - event-driven design
  - job-based processing
  - async pipelines

## When unsure:
- ask for clarification
- DO NOT assume

## Priority Order

1. System architecture rules
2. Security constraints
3. Performance constraints
4. Developer convenience

## Forbidden Behavior

- hallucinated services
- breaking pipeline flow
- bypassing queues
- mixing sync + async incorrectly