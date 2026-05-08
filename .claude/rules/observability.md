# OBSERVABILITY (NON-NEGOTIABLE)

## Core Principle
If you cannot observe it, you cannot operate it.

## Required Signals

### Logs
- structured (JSON)
- include job_id, event_id

### Metrics
- ingestion rate
- processing latency
- error rate
- queue depth

### Tracing
- request → job → worker → result

## Alerts

System MUST alert on:
- high failure rate
- queue backlog growth
- latency spikes

## Correlation

Every request MUST propagate:
- request_id
- job_id

## Anti-Patterns (FORBIDDEN)

- print debugging
- missing correlation IDs
- silent failures