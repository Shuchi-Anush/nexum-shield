# JOB PROCESSING & STATE MANAGEMENT

## Core Principle
Jobs are the source of truth for pipeline execution.

## Job Lifecycle

QUEUED → PROCESSING → COMPLETED | FAILED | FLAGGED

## Requirements

### 1. Idempotency
- create_job MUST NOT create duplicates
- job_id MUST be deterministic or UUID

### 2. State Transitions
- Only valid transitions allowed
- No skipping states

### 3. Concurrency Safety
- All updates MUST be atomic
- Use locking or DB transactions

### 4. Retry Strategy

Each job MUST include:
- retry_count
- max_retries

Retry conditions:
- transient failures → retry
- permanent failures → FAILED

### 5. Persistence

In-memory store is NOT allowed in production.

Replace with:
- PostgreSQL OR
- Redis (for ephemeral state)

## Observability

Each job MUST log:
- created_at
- updated_at
- processing duration
- failure reason

## Anti-Patterns (FORBIDDEN)

- Global singleton job stores (in production)
- Silent failures
- Missing retry logic