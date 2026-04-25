# Nexum Shield — Media Integrity Platform

Nexum Shield is a real-time, distributed system for detecting, analyzing, and enforcing against unauthorized media distribution at internet scale.

---

## 🚨 Problem

Unauthorized redistribution of sports and premium media content leads to:

- Massive revenue leakage  
- Weak, delayed enforcement  
- Lack of cross-platform visibility  
- Easy evasion via minor transformations  

---

## 🎯 Objective

Build a system capable of:

- Identifying media reuse using fingerprints + embeddings  
- Matching against protected content at scale  
- Scoring risk and confidence  
- Enforcing actions (ALLOW / FLAG / BLOCK)  
- Providing auditable, explainable decisions  

---

## ⚙️ System Overview

Nexum Shield is built as an **event-driven distributed pipeline**:

```text
Client
  ↓
Ingest API (FastAPI)
  ↓
Redis Queue (RQ)
  ↓
Worker Processes
  ↓
Processing Pipeline
  ↓
Decision + Evidence Storage
```

---

## 🔄 Processing Pipeline

Each media ingestion request flows through:

### 1. Fingerprinting

- Deterministic SHA-256 (current stub)
- Future: perceptual hashing (pHash, video keyframes)

### 2. Embedding

- 32-dimensional deterministic vector (stub)
- Future: CLIP / multimodal embeddings

### 3. Matching

- Cosine similarity against protected assets
- Future: vector DB (FAISS / Milvus / RedisSearch)

### 4. Scoring

- Maps similarity → confidence band:
  - LOW
  - MEDIUM
  - HIGH

### 5. Enforcement

- Decision mapping:
  - LOW → ALLOW
  - MEDIUM → FLAG
  - HIGH → BLOCK
- Generates full explainable evidence record

---

## 🧠 Key Design Principles

- Fully asynchronous processing (non-blocking API)  
- Event-driven pipeline architecture  
- Deterministic, reproducible processing  
- Explainable and auditable decisions  
- Modular stage-based execution  
- Designed for horizontal scalability  

---

## 🏗️ Current Architecture Status

### ✅ Implemented

- FastAPI ingestion API  
- Redis-backed queue (RQ)  
- Worker execution system (Windows-safe SimpleWorker)  
- End-to-end processing pipeline  
- Matching + scoring + enforcement engines  
- Structured job tracking  
- Basic event flow  

---

### ⚠️ In Progress

- Redis-backed JobStore (shared state across API + workers)  
- Idempotency and retry-safe execution  
- Observability (metrics, latency, failure tracking)  

---

### 🚧 Planned

- Perceptual media fingerprinting (video/audio)  
- Vector database integration (ANN search)  
- Cross-platform content tracking  
- Rights-holder registry system  
- Human review workflows  
- Legal enforcement (DMCA-style pipelines)  

---

## 🧪 Example Flow

### Request

```json
POST /v1/ingest
{
  "source_url": "https://espn.com/highlights/match-001.mp4",
  "content_type": "video"
}
```

### Execution

- Job created and stored  
- Job pushed to Redis queue  
- Worker picks up job  
- Pipeline executes all stages  
- Matching identifies protected content  
- Scoring assigns HIGH confidence  
- Enforcement triggers BLOCK action  

### Result

```json
{
  "status": "flagged",
  "match": true,
  "owner": "ESPN",
  "confidence": 1.0,
  "action": "BLOCK"
}
```

---

## ⚠️ Known Limitations (Current Stage)

- Fingerprinting is metadata-based (not content-based)  
- JobStore is in-memory (not distributed yet)  
- No idempotency safeguards  
- No retry-safe execution  
- Limited observability  

---

## 🧰 Tech Stack

- **Backend**: FastAPI (Python)  
- **Queue**: Redis + RQ  
- **Workers**: Python (SimpleWorker / Worker)  
- **Frontend**: Next.js  

---

## 🚀 Running Locally

### 1. Start Redis

```bash
docker run -p 6379:6379 redis
```

### 2. Start Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

### 3. Start Worker

```bash
cd backend
python -m app.workers.worker
```

### 4. Access API Docs

```text
http://127.0.0.1:8000/docs
```

---

## 🧭 Future Direction

Nexum Shield is evolving toward:

- Internet-scale ingestion (10M–100M+ assets)  
- Real-time detection and enforcement  
- AI-powered content identity systems  
- Trust & Safety infrastructure  

---

## 📌 Project Status

Actively under development — transitioning from a prototype to a production-grade distributed system.
