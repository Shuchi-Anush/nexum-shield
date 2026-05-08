# Architecture Rules

## System Type
Event-driven distributed pipeline

## Rules
- All stages communicate via events or async jobs
- No direct tight coupling between modules
- Each stage must be independently scalable

## Pipeline Integrity
- ingest → job → process → match → score → enforce
- This order MUST NOT change

## Anti-Patterns
- Direct DB calls across modules
- Sync chaining across services
- Business logic inside API routes