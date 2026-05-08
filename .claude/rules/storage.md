# STORAGE ARCHITECTURE — NEXUM SHIELD (PRODUCTION RULES)

## CORE PRINCIPLES

- Raw data MUST be immutable
- Derived data MUST be versioned
- Storage responsibilities MUST be strictly separated
- Every enforcement decision MUST be traceable to stored evidence

---

## STORAGE LAYERS

### 1. RAW MEDIA (SOURCE OF TRUTH)
- Stores original uploaded content
- MUST be immutable
- MUST be content-addressable (hash-based ID)

Storage:
- DEV → local filesystem
- PROD → object storage (S3 / GCS)

Rules:
- NEVER overwrite
- NEVER mutate
- ALWAYS validate uploads (type, size, format)

---

### 2. METADATA STORE (CONTROL PLANE)
- Job state
- ingestion records
- processing status
- references to media

Storage:
- DEV → SQLite
- PROD → PostgreSQL

Requirements:
- MUST support strong consistency for:
  - job state transitions
  - enforcement decisions

---

### 3. EMBEDDINGS STORE
- High-dimensional vectors
- Used for similarity search

Storage:
- Vector DB (FAISS / Milvus / Pinecone)

Each embedding MUST include:
- model_version
- timestamp
- media_id reference

Rules:
- NEVER store embeddings in relational DB
- MUST support fast nearest-neighbor queries

---

### 4. FINGERPRINT STORE (LIGHTWEIGHT MATCHING)
- Perceptual hashes
- Fast pre-filter before embeddings

Rules:
- Stored separately from embeddings
- Used for coarse matching stage

---

### 5. JOB / QUEUE STORE
- Job state tracking
- Queue processing

Storage:
- Redis (queue + ephemeral state)
- Postgres (persistent job metadata)

Rules:
- MUST support retries
- MUST support idempotency

---

### 6. EVIDENCE STORE (CRITICAL)

This is NON-NEGOTIABLE.

Each detection MUST store:

- input_media_id
- matched_media_id
- similarity_score
- model_version
- timestamp

Purpose:
- enforcement decisions
- audit trails
- dispute handling

Rules:
- MUST be immutable
- MUST be queryable
- MUST persist long-term

---

## NAMING & IDENTIFIERS

- Media → content hash (SHA-256 or similar)
- Jobs → UUID
- NEVER rely on filenames

---

## ACCESS PATTERNS

- Write-heavy → ingestion
- Read-heavy → embeddings + matching
- MUST support horizontal scaling

---

## CONSISTENCY MODEL

- Eventual consistency allowed for:
  - embeddings
  - indexing

- Strong consistency REQUIRED for:
  - job state transitions
  - enforcement decisions
  - evidence storage

---

## API CONSTRAINTS

- API layer MUST NOT:
  - store large files synchronously
  - perform heavy storage operations inline

- MUST:
  - upload asynchronously
  - return job_id immediately

---

## SECURITY

- Validate all uploads
- Prevent path traversal
- Restrict direct storage access
- Use signed URLs for media access

---

## FAILURE MODES & RECOVERY

System MUST handle:

- Partial uploads → retry or rollback
- Missing embeddings → re-trigger pipeline
- Storage outage → queue retry (no data loss)

---

## ANTI-PATTERNS (FORBIDDEN)

- Storing blobs in relational DB
- Mixing embeddings with metadata
- Mutating raw media
- Losing linkage between:
  - media
  - embeddings
  - evidence

---

## FUTURE CONSIDERATIONS

- Deduplication using content hash
- Tiered storage (hot vs cold)
- CDN for media delivery
- Multi-region replication