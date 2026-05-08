# API Contracts

## Principles
- APIs are stable contracts
- Never break existing response schema

## Ingest API
POST /v1/ingest
- Must return immediately (job_id)
- Must not process data inline

## Jobs API
GET /v1/jobs/{id}
- Must reflect current state
- Must be eventually consistent

## DO NOT
- Add heavy logic in endpoints
- Change response formats without versioning