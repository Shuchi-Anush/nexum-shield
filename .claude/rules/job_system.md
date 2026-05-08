# Job System Rules

## Job Lifecycle
QUEUED → PROCESSING → COMPLETED | FAILED | FLAGGED

## Requirements
- Jobs must be idempotent
- Status updates must be atomic
- Every job must have metadata

## Future Direction
- Replace in-memory store with Redis/Postgres
- Add distributed queue (Kafka / Redis Streams)

## DO NOT
- Store business logic inside JobStore
- Block execution during job processing