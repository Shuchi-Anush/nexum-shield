# AGENT ORCHESTRATION STRATEGY

## Core Principle
Use the right agent for the right task.

## Workflow

1. Backend Agent
   - defines APIs
   - defines job creation

2. ML Agent
   - defines embedding + similarity logic

3. Orchestrator
   - ensures pipeline correctness
   - validates integration

## Rules

- Agents MUST NOT override each other
- Conflicts → resolve via architecture rules

## Parallelization

Allowed for:
- research
- independent modules

Not allowed for:
- shared state mutation