# NEXUM SHIELD — Media Integrity Platform

![Architecture: Event-Driven](https://img.shields.io/badge/Architecture-Event--Driven-blue)
![Runtime: Deterministic](https://img.shields.io/badge/Runtime-Deterministic-success)
![Governance: Canonical](https://img.shields.io/badge/Governance-Canonical-important)
![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange)

**Distributed AI-powered media integrity infrastructure for detecting, tracking, analyzing, and enforcing against unauthorized sports media distribution at internet scale.**

Event-Driven • Deterministic • Explainable • Adversarial-Resilient • Governance-Driven

---

## Overview

NEXUM SHIELD is a production-oriented distributed systems program focused on large-scale media integrity enforcement.

The platform is designed to detect unauthorized redistribution of sports and premium media assets using:

- perceptual fingerprinting
- multimodal embedding analysis
- vector similarity retrieval
- explainable policy orchestration
- trust-aware enforcement systems
- event-driven distributed pipelines

The long-term target state is a horizontally scalable integrity platform capable of processing:

```text
10M–100M+ media assets
```

with deterministic policy evaluation, auditable enforcement decisions, replay-safe execution, and adversarial-resilient analysis pipelines.

---

## Table of Contents

- [Overview](#overview)
- [Problem Space](#problem-space)
- [Core Objectives](#core-objectives)
- [Runtime Pipeline](#high-level-runtime-pipeline)
- [Core Runtime Engines](#core-runtime-engines)
- [Governance Architecture](#governance-architecture)
- [Current Architecture State](#current-architecture-state)
- [Local Development](#local-development)
- [Engineering Model](#engineering-model)
- [Long-Term Vision](#long-term-vision)

---

## Problem Space

Unauthorized redistribution of sports and premium media content creates:

- massive revenue leakage
- delayed enforcement response
- weak attribution capability
- poor cross-platform visibility
- easy evasion via transformations and recompression
- fragmented evidence collection
- non-explainable moderation decisions

Modern media integrity systems require:

- scalable ingestion
- resilient similarity detection
- deterministic policy evaluation
- explainable escalation behavior
- distributed orchestration
- audit-preserving evidence chains

---

## Core Objectives

- Detect unauthorized media redistribution in near real-time
- Support large-scale asynchronous ingestion pipelines
- Maintain deterministic and explainable policy decisions
- Build auditable enforcement workflows
- Enable adversarial-resistant media matching
- Preserve governance and architectural consistency at scale
- Support replay-safe distributed execution
- Enable future legal and human-review workflows

---

## System Characteristics

| Dimension | Target |  
| --------- | ------ |
| Architecture | Event-driven distributed system |
| Processing Model | Fully asynchronous beyond ingestion |
| Decision System | Deterministic + explainable |
| Scale Target | 10M–100M+ assets |
| Matching | Fingerprint + embedding fusion |
| Enforcement | Policy-governed escalation |
| Reliability | Retry-safe + idempotent |
| Governance | Constitution + canonical specs |
| Observability | Structured event emission |
| Security | Trust-aware enforcement model |

---

## High-Level Runtime Pipeline

```text
┌────────────────────┐
│   Client Upload    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│   Ingestion API    │
│      (FastAPI)     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Job Orchestration  │
│       Layer        │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Fingerprint Engine │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Embedding Engine   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Matching Engine   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Decision Engine   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Confidence Engine  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│   Policy Engine    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Enforcement Engine │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Audit + Evidence   │
│      Storage       │
└────────────────────┘
```

---

---

## System Visual Architecture

### Unified System Architecture

The platform operates as a distributed, event-driven intelligence system designed for scalable media integrity enforcement.

![NEXUM SHIELD Architecture](docs/assets/architecture-overview.png)

---

### End-to-End Data Flow

The runtime pipeline processes ingestion, feature extraction, matching, intelligence analysis, decisioning, and enforcement through asynchronous event-driven orchestration.

![NEXUM SHIELD Data Flow](docs/assets/runtime-flow.png)

---

### AI & Detection Pipeline

The AI pipeline combines perceptual fingerprinting, deep embeddings, vector similarity search, anomaly analysis, and explainable risk scoring.

![NEXUM SHIELD AI Pipeline](docs/assets/ai-pipeline.png)

---

### Propagation & Anomaly Intelligence

The intelligence layer tracks cross-platform propagation patterns, abnormal spread velocity, behavioral anomalies, and risk amplification signals.

![NEXUM SHIELD Propagation & Anomaly](docs/assets/propagation-anomaly.png)

---

### Deployment & Infrastructure Architecture

The deployment model follows cloud-native distributed infrastructure principles with asynchronous processing, scalable workers, observability, and fault isolation.

![NEXUM SHIELD Deployment Architecture](docs/assets/deployment-topology.png)

---

### Policy Runtime & Governance Engine

The deterministic policy runtime governs legal constraints, explainability, human review requirements, escalation workflows, and audit-safe enforcement execution.

![NEXUM SHIELD Policy Runtime](docs/assets/policy-runtime.png)

---

## Runtime Execution Model

The platform is designed around asynchronous execution boundaries.

No heavy computation is allowed in synchronous API paths.

```text
API → Queue → Workers → Processing → Decision → Enforcement
```

Target execution characteristics:

- non-blocking ingestion
- retry-safe jobs
- deterministic evaluation
- replay-safe execution
- structured event emission
- failure isolation
- horizontally scalable workers

---

## Core Runtime Engines

### DecisionEngine

Transforms detection evidence into deterministic risk scoring outputs.

Responsibilities:

- evidence fusion
- weighted risk aggregation
- risk band assignment
- deterministic scoring behavior

---

### ConfidenceEngine

Evaluates confidence quality and agreement strength across heterogeneous signals.

Responsibilities:

- confidence aggregation
- signal concordance analysis
- weighted confidence evaluation
- confidence-tier assignment

---

### PolicyEngine

Deterministic policy execution engine implementing PBRA semantics:

```text
Propose → Bound → Refine → Assert
```

Responsibilities:

- enforcement resolution
- escalation bounding
- safety invariant enforcement
- deterministic policy evaluation
- invariant verification

The runtime guarantees:

- deterministic escalation
- bounded enforcement semantics
- explainable resolution behavior
- replay-safe evaluation

---

### TrustReader

Read-only trust intelligence layer.

Responsibilities:

- owner trust retrieval
- uploader trust retrieval
- pessimistic-default enforcement
- trust signal interpretation

---

### MatchingEngine

Responsible for similarity retrieval and candidate resolution.

Responsibilities:

- ANN retrieval
- candidate ranking
- similarity thresholding
- cross-signal fusion

---

### FingerprintEngine

Generates perceptual fingerprints for media analysis.

Responsibilities:

- frame fingerprinting
- perceptual hashing
- media signature extraction
- transformation-tolerant matching

---

## Architectural Principles

### Deterministic Runtime

Identical inputs must produce:

- identical outputs
- identical evaluation hashes
- identical escalation paths

All enforcement behavior must remain reproducible.

---

### Asynchronous Processing

No stage may block the ingestion API path.

All expensive workloads are delegated to:

- orchestration workers
- queue systems
- asynchronous processing pipelines

---

### Explainable Enforcement

Every enforcement action must preserve:

- evidence lineage
- confidence reasoning
- escalation rationale
- policy traceability

---

### Idempotent Execution

All jobs and event handlers must remain retry-safe.

The platform is designed around:

- replay tolerance
- deduplication safety
- distributed retry semantics

---

### Governance-Driven Engineering

Architecture evolution is governed through:

- constitutional invariants
- canonical specifications
- domain ownership boundaries
- ADR-oriented evolution

---

## Governance Architecture

NEXUM SHIELD uses layered governance to prevent architectural drift and semantic inconsistency.

### Constitution Layer

Immutable system-level axioms and invariants.

Examples:

- deterministic runtime guarantees
- explainability requirements
- enforcement safety boundaries

---

### Specification Layer

Canonical implementation specifications for runtime systems.

Examples:

- PolicyEngine semantics
- ConfidenceEngine behavior
- event contracts
- storage guarantees

---

### Working Memory Layer

Mutable operational context for implementation planning and AI-assisted engineering workflows.

---

## Repository Structure

```text
backend/        Runtime services + processing engines
frontend/       Investigation and analyst interface
contracts/      Schemas and API contracts
infra/          Infrastructure + deployment configuration
docs/           Governance + canonical specifications
.claude/        AI-assisted engineering governance
```

---

## Current Architecture State

### Implemented

- FastAPI backend foundation
- deterministic PolicyEngine runtime
- DecisionEngine
- ConfidenceEngine
- governance foundation
- canonical semantic specifications
- Dockerized development setup
- repository governance model
- invariant enforcement structure
- structured policy runtime

---

### In Progress

- persistent storage integration
- distributed queue orchestration
- vector similarity infrastructure
- replay pipelines
- observability stack
- enforcement integrations
- event bus evolution

---

### Planned

- perceptual video fingerprinting
- ANN vector retrieval infrastructure
- propagation graph intelligence
- human review workflows
- trust analytics
- evidence chain integrity
- automated legal escalation workflows
- distributed event-stream processing

---

## Example Runtime Flow

### Request

```http
POST /v1/ingest
Content-Type: application/json
```

```json
{
  "source_url": "https://example.com/highlights/final-match.mp4",
  "content_type": "video"
}
```

---

### Execution

```text
1. Ingestion API accepts request
2. Job is persisted and queued
3. Worker consumes processing task
4. Fingerprint extraction executes
5. Embedding generation executes
6. Similarity matching retrieves candidates
7. DecisionEngine computes risk score
8. ConfidenceEngine computes confidence tier
9. PolicyEngine resolves enforcement action
10. Evidence + audit trail stored
```

---

### Result

```json
{
  "status": "flagged",
  "match": true,
  "owner": "SportsNetwork",
  "confidence_tier": "HIGH",
  "policy_action": "RESTRICT",
  "evaluation_hash": "dd33a2e843cfa467"
}
```

---

## Observability Goals

Every major runtime stage emits structured events.

Target observability domains:

- ingestion lifecycle
- queue latency
- worker throughput
- similarity retrieval quality
- policy transitions
- enforcement escalation
- invariant violations
- runtime failures

---

## Security Model

The platform assumes adversarial environments.

Key security principles:

- trust-aware enforcement
- immutable evidence lineage
- replay-safe execution
- bounded escalation logic
- audit-preserving decisions
- invariant enforcement
- deterministic evaluation

---

## Current Known Limitations

The repository is still transitioning from prototype infrastructure toward production-grade distributed runtime architecture.

Current limitations include:

- partial in-memory runtime state
- no production queue orchestration yet
- vector infrastructure still evolving
- limited replay tooling
- partial observability coverage
- stub fingerprinting implementations
- simplified embedding generation

These limitations are explicitly acknowledged and tracked to preserve architectural honesty and implementation clarity.

---

## Tech Stack

| Layer | Technology |
| ----- | ---------- |
| Backend | FastAPI + Python |
| Frontend | Next.js |
| Queue | Redis / RQ (transitional) |
| Runtime | Docker |
| Dependency Management | uv |
| Future ANN | FAISS / Milvus / RedisSearch |
| Future ML | CLIP-style multimodal embeddings |

---

## Local Development

## Prerequisites

- Python 3.12+
- Node.js 20+
- Docker Desktop
- uv
- Git

---

## Environment Setup

### Root

```env
PROJECT_NAME=nexum-shield
ENV=dev
```

---

### Backend

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nexum

REDIS_URL=redis://127.0.0.1:6379/0

GEMINI_MODEL=gemini-2.5-flash
```

---

### Frontend

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## Backend Setup

```bash
cd backend

uv sync

uv run uvicorn app.main:create_app \
  --factory \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

---

## Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

---

## Docker Setup

```bash
docker compose up --build
```

---

## API Documentation

After backend startup:

```text
http://127.0.0.1:8000/docs
```

---

## Engineering Model

NEXUM SHIELD uses AI-assisted engineering workflows with explicit governance constraints.

The repository includes:

- architecture governance
- operational rules
- engineering agents
- implementation planning memory
- invariant documentation
- canonical specifications

to preserve long-term architectural consistency across both humans and AI agents.

---

## Long-Term Vision

NEXUM SHIELD aims to evolve into a large-scale media integrity infrastructure platform combining:

- distributed systems engineering
- AI-powered similarity analysis
- adversarial resilience
- trust-aware policy orchestration
- explainable enforcement pipelines
- governance-driven architecture

for internet-scale media protection workflows.

---

## License

MIT License

---

## Status

```text
Architecture Program Phase:
Governance Foundation + Deterministic Runtime Establishment
```
