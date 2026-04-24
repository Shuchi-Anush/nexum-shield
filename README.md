# Nexum Shield — Media Integrity Platform

Nexum Shield is a real-time AI-powered platform designed to detect, track, and enforce against unauthorized media distribution at internet scale.

## Problem

Unauthorized redistribution of sports and premium media content results in massive revenue loss and weak enforcement due to:

- Fragmented detection systems
- Lack of real-time monitoring
- Poor cross-platform traceability

## Vision

Build a scalable, distributed system capable of:

- Detecting media reuse via perceptual hashing and embeddings
- Tracking propagation across platforms
- Enforcing takedowns and compliance workflows

## Current Status

⚠️ Early-stage system (MVP)

Implemented:

- FastAPI backend
- Job ingestion API
- Basic job tracking (in-memory)

Planned:

- Distributed ingestion pipeline
- Queue-based processing (Redis/Kafka)
- Embedding + similarity search pipeline
- Enforcement engine

## Architecture (High-Level)

Client → API → Job Queue → Workers → Processing Pipeline → Storage

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: Next.js
- Future: Redis, PostgreSQL, Vector DB

## Running Locally

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8001
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

Notes
Current JobStore is in-memory (will be replaced with persistent storage)
System is being built for large-scale distributed processing
