# EVENT-DRIVEN ARCHITECTURE (MANDATORY)

## Core Principle
All heavy operations MUST be asynchronous and event-driven.

## Pipeline (Canonical Flow)
ingest → enqueue → worker → embedding → matching → scoring → enforcement

## Rules

- API layer MUST NOT perform:
  - embedding generation
  - similarity search
  - ML inference

- API layer ONLY:
  - validates request
  - creates job
  - publishes event

## Event Bus Contract

Each event MUST include:
- event_id (UUID)
- job_id
- timestamp
- type (INGEST_RECEIVED, EMBEDDING_READY, MATCH_FOUND, etc.)
- payload (strict schema)

## Idempotency

All consumers MUST be idempotent:
- Same event processed multiple times → same result
- No duplicate side effects

## Backpressure

System MUST:
- queue events when overloaded
- never block ingestion synchronously

## Failure Handling

- Failed events → retry queue
- Max retries → dead letter queue (DLQ)

## Anti-Patterns (FORBIDDEN)

- Synchronous ML calls inside API
- Direct DB writes across services
- Tight coupling between pipeline stages