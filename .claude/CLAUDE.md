# Nexum Shield — Media Integrity Platform

## SYSTEM PURPOSE

Detect, track, and enforce unauthorized sports media distribution at internet scale.

## TARGET-STATE PIPELINE

1. Ingest Service (API)
2. Job Orchestration Layer (state machine)
3. Processing Engine (fingerprinting + embeddings)
4. Matching Engine (vector similarity search)
5. Risk Scoring Engine
6. Enforcement Engine (takedown / flagging)

## NON-NEGOTIABLE CONSTRAINTS

- All processing must be asynchronous beyond ingestion
- Jobs must be idempotent and retry-safe
- No stage may block API response path
- Every stage must emit structured events
- All decisions must be explainable and auditable

## FAILURE MODEL

- Partial failures allowed
- Jobs must support retry + recovery
- No silent failures

## DATA MODEL PRINCIPLES

- Immutable input data
- Versioned outputs
- Evidence must be stored for enforcement actions

## PERFORMANCE TARGETS

- High-throughput ingestion (10M–100M assets)
- Low-latency API (<200ms for ingestion)
- Async processing pipelines

## DO NOT

- Introduce synchronous heavy processing in API
- Modify pipeline order without explicit instruction
- Break API contracts
- Remove logging or auditability

## WHEN BUILDING FEATURES

- Always use Plan Mode first
- Identify impacted pipeline stage
- Maintain backward compatibility

## CURRENT STATE

- Basic FastAPI backend
- In-memory JobStore (temporary)
- No queue system yet

## CURRENT MVP STATE

- FastAPI ingestion service
- In-memory JobStore
- Deterministic pHash-based similarity
- Hamming-distance evaluation
- PBRA enforcement logic
- No distributed queue yet
- No vector retrieval yet
- No embedding pipeline yet
