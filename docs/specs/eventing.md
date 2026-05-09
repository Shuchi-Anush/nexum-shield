---
authority: SPEC
domain: pipeline
status: ACTIVE
version: 1.0
stability: EVOLVING
owner: pipeline (interim: architect)
supersedes:
  - .claude/rules/eventing.md (TRANSITIONAL — partial; the constitutional parts (idempotency, async pipeline, no-sync-ML rule) are also reflected in AXIOMS A1 / A5)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; canonical-spec ratification; to be backfilled)
---

# Eventing — Canonical Specification

The Eventing layer is the **runtime substrate** of the Nexum Shield
pipeline. It governs how the semantic phases of A1
(`INGESTION → ANALYSIS → EVALUATION → DECISION → ENFORCEMENT +
AUDITABILITY`) are materialised across processes, how events are
identified, ordered, retained, and replayed, and how delivery,
failure, and orchestration boundaries behave.

This document is the canonical specification — Tier 2 (SPEC) — and
supersedes `.claude/rules/eventing.md` for the partial scope described
in the frontmatter. The implementation surface is:

- `backend/app/core/queue.py` — Redis + RQ singleton wiring.
- `backend/app/core/event_store.py` — append-only event log (lifecycle
  + canonical pipeline events).
- `backend/app/core/job_store.py` — Redis-backed job state machine
  (consumer-side state).
- `backend/app/workers/pipeline_worker.py` — orchestrator (current
  monolithic worker that walks all stages sequentially).
- `backend/app/workers/worker.py` — RQ entrypoint.
- `backend/app/api/ingest.py` — ingest API (event emission boundary).

The eventing layer is a **shared cross-domain substrate**. Owned by
the **pipeline** domain (`docs/governance/DOMAINS.md`), it is consumed
by api, policy, decision, confidence, security, qa, and platform.
Schema and contract changes are cross-domain by nature.

---

## §1 System Role

### §1.1 Purpose

The Eventing layer answers four questions:

1. **How does work move through the pipeline?** — what produces an
   event, what consumes it, and how the per-job execution graph is
   materialised at runtime.
2. **What guarantees does the runtime offer?** — delivery (at-least-
   once), ordering (per-job by emission timestamp), idempotency
   (consumer-side), retention (append-only).
3. **What is the audit trail?** — every event is part of the A4 audit
   record; the per-job event log IS the audit timeline for that job.
4. **Where can the runtime evolve safely?** — what counts as a schema
   change, who owns event types, what coupling is forbidden.

The eventing layer is **NOT**:

- A general message bus for unrelated systems.
- A cross-job orchestration engine (job-to-job dependencies are
  outside scope).
- A workflow engine with retries, schedules, or DAG semantics.
- An exactly-once delivery system.

### §1.2 Position vs A1 phases

Per A1 (PIPELINE PHASE INTEGRITY), the system MUST execute the
semantic phases `INGESTION → ANALYSIS → EVALUATION → DECISION →
ENFORCEMENT + AUDITABILITY`. The axiom is **explicitly orchestration-
agnostic**: any orchestration choice (event-driven, monolithic,
batch, streaming) that preserves the semantic ordering is permitted.

This spec records the **current chosen orchestration** — Redis +
RQ + a single sequential pipeline worker — as the canonical MVP
implementation, and defines extension points for evolution (§13,
§11). The orchestration choice is **not** axiomatic and may be
revised by a Standard ADR per §16.

```
HTTP request                    Worker process
────────────                    ──────────────
POST /v1/ingest    enqueue       run_pipeline(job_id)
   │           ─────────────▶       │
   │  job_id                         │  fingerprint
   │  status=QUEUED                  │  embedding
   ▼                                 │  matching
return 202                           │  scoring     ← legacy (E-EV-6)
                                     │  enforcement ← legacy (E-EV-6)
                                     ▼
                                  job_store.update_status(...)
                                  publish_event(JOB_COMPLETED|FAILED)

Throughout: every stage emits {STARTED, COMPLETED|FAILED} lifecycle
            events + a typed PipelineEventType domain event into the
            same per-job append-only log.
```

### §1.3 Authority

This document is **TIER 2 (SPEC)** per
`docs/constitution/GOVERNANCE.md` §1. Owned by the **pipeline**
domain (`docs/governance/DOMAINS.md`). Modification:

| Change | ADR tier |
|---|---|
| Adjusting RQ `job_timeout` / lock TTL constant | Lightweight |
| Adding a new payload field to an existing `PipelineEventType` (additive only) | Lightweight |
| Adding a new `PipelineEventType` value | Standard (consumed by api, security, qa) |
| Renaming or removing an existing `PipelineEventType` value | Standard |
| Schema-version bump to event envelope (`Event` dataclass) | Standard |
| Changing per-job ordering guarantees | Standard |
| Switching broker (Redis → Kafka / NATS / SQS) | Standard (see §11.4) |
| Splitting `pipeline` queue into multiple queues | Standard |
| Removing append-only retention guarantee | Constitutional (violates A4 / A7) |
| Removing the at-least-once delivery floor | Constitutional |
| Reordering A1 semantic phases | Constitutional (axiom-level) |

### §1.4 Stability

**EVOLVING**. The current Redis + RQ runtime is sufficient for MVP
but several known gaps (retention, DLQ, schema versioning,
multi-broker abstraction) are tracked in §14. Compatibility
expectations are low — consumers should expect change at each minor
version bump until graduation gates (§16.1) are met.

The `PipelineEventType` enum and the `Event` envelope schema are
**cross-domain consumer surfaces**. They are EVOLVING today;
promoting them to STABLE/LOCKED is a multi-step path tied to
production rollout (see §16.1).

---

## §2 Event Model

### §2.1 Canonical Event envelope

Every event in the system — lifecycle audit or canonical pipeline
event — is persisted as a single record with the following shape:

```python
@dataclass
class Event:
    event_id:    str        # UUID4, engine-assigned at emit time
    job_id:      str        # correlation key — partitions per-job log
    stage:       str        # mapped from event_type (§5.3)
    event_type:  str        # EventType.value or PipelineEventType.value
    timestamp:   float      # seconds since epoch (derived from time.time_ns())
    payload:     dict?      # already-serialisable; pre-validated for typed events
    latency_ms:  float?     # only set for lifecycle COMPLETED/FAILED events
```

The envelope has been live for the full Phase-2 governance series
and is **frozen at this shape** until a Standard ADR adds new fields
(§1.3, §10.4).

### §2.2 Event identity

| Field | Source | Uniqueness scope | Stability |
|---|---|---|---|
| `event_id` | `uuid.uuid4()` inside `emit()` | global, per-process collision-free | replay-safe — never derived from inputs |
| `job_id` | API boundary (`uuid.uuid4()` in `POST /v1/ingest`) | global | the job correlation key for the entire pipeline |
| `stage` | `_STAGE_FOR[event_type]` lookup | per event-type | derived; not authoritative |
| `event_type` | enum value | global | the canonical taxonomy; see §3 |
| `timestamp` | `time.time_ns() / 1e9` | per-emitter | NOT a unique key — collisions possible across pods on millisecond-aligned emits, but per-job ordering on a single Redis is monotonic by ns |

### §2.3 Correlation and lineage

Today the **only** correlation primitive is `job_id`. All events
emitted during the lifetime of a single job carry the same `job_id`
and form the per-job audit timeline.

There is **no** `causation_id` / `parent_event_id` field today. The
causal relationship between two events (e.g., FINGERPRINT_READY
caused EMBEDDING_READY) is implicit in:

1. Their shared `job_id` (correlation).
2. Their `timestamp` ordering on the per-job sorted index.
3. The pipeline worker's hard-coded sequential order (§4.3).

**Gap E-EV-4** — the absence of an explicit causation chain is a
known limitation. Adding `parent_event_id` requires a Standard ADR
(envelope schema change + consumer migration); §15 records it as
open work.

The system also lacks an explicit `request_id` propagation primitive
from the HTTP boundary into the worker (the API generates a
`job_id` directly, not a separate `request_id`). For HTTP-level
correlation today, `job_id` doubles as the request identifier.
A future Lightweight ADR may introduce `request_id` (E-EV-4 family).

### §2.4 Timestamp semantics

Timestamps are derived from `time.time_ns()` at emit time, divided
by `1e9` to produce float seconds. They satisfy:

- **Per-Redis-instance monotonicity** — two emits from the same
  process to the same Redis instance are strictly ordered.
- **Per-job ordering preservation** — the sorted-set index
  `events_by_job:{job_id}` is scored by the integer `ts_ns` value,
  preserving emission order.
- **Cross-job ordering — undefined.** Events for two different
  `job_id`s have no guaranteed relative order.
- **Cross-pod ordering — undefined.** Once the worker scales to >1
  pod, two emits at near-identical wall-clock times from different
  pods may interleave. This is acceptable because all consumer
  semantics (ordering, audit replay) operate per-job.

### §2.5 Versioning

The envelope itself is **unversioned**. Payload schemas are
version-implicit — defined as Pydantic models in
`backend/app/core/event_store.py`. Adding a new optional field to
an existing payload class is **additive (Lightweight)**; renaming or
removing fields is breaking (Standard ADR).

**Gap E-EV-3** — there is no explicit `schema_version` field on
either envelope or payload. Replay against a future codebase whose
Pydantic schemas have evolved could deserialize against the new
shape silently, masking semantic drift. Promoting `schema_version`
into the envelope is a Standard ADR (cross-domain consumer surface);
§15 records it as open work.

### §2.6 Immutability rule

The event store is **append-only**. The `emit()` API has no update
or delete primitive. The Redis storage uses `SET` for the JSON blob
and `ZADD` for the index — neither call site overwrites. Mutating
an existing event is forbidden and is a **P0** violation per
`docs/constitution/GOVERNANCE.md` §5 (it would corrupt the A4 audit
record).

Per A4, "the audit record MUST be append-only. Mutating an existing
record constitutes evidence destruction. A reversal under A6 is
recorded as an *appended* entry, not as a mutation of the original."
This spec materialises that axiom for the event log.

---

## §3 Event Taxonomy

### §3.1 Layered event types

The event log carries **two layers** in the same per-job index:

```
┌────────────────────────────────────────────────────────────────┐
│  Lifecycle audit layer (operational)                           │
│    EventType.STARTED                                           │
│    EventType.COMPLETED          (latency_ms set)               │
│    EventType.FAILED             (latency_ms + error payload)   │
│  Emitted by: stage_event() context manager around each stage   │
│  Consumed by: ops dashboards, per-stage latency views          │
├────────────────────────────────────────────────────────────────┤
│  Canonical pipeline events (domain)                            │
│    PipelineEventType.{INGEST_RECEIVED, FINGERPRINT_READY,      │
│                       EMBEDDING_READY, MATCH_FOUND,            │
│                       MATCH_NOT_FOUND, SCORED, ENFORCED,       │
│                       JOB_COMPLETED, JOB_FAILED}               │
│  Emitted by: api/ingest.py + workers/pipeline_worker.py        │
│  Consumed by: API readers, audit, future workers, qa replay    │
└────────────────────────────────────────────────────────────────┘
```

Co-mingling is **deliberate** (`event_store.py` docstring): one
ordered log keeps "what happened to job X" answerable with a single
sorted-set scan. Splitting them across separate keys would fragment
the per-job audit timeline.

### §3.2 Domain event catalogue

| Event | Phase (A1) | Payload class | Source |
|---|---|---|---|
| `INGEST_RECEIVED` | INGESTION | `IngestReceivedPayload` | API: `POST /v1/ingest` |
| `FINGERPRINT_READY` | ANALYSIS | `FingerprintReadyPayload` | worker: post-fingerprint stage |
| `EMBEDDING_READY` | ANALYSIS | `EmbeddingReadyPayload` | worker: post-embedding stage |
| `MATCH_FOUND` | ANALYSIS | `MatchFoundPayload` | worker: matching stage (positive) |
| `MATCH_NOT_FOUND` | ANALYSIS | `MatchNotFoundPayload` | worker: matching stage (negative) |
| `SCORED` | EVALUATION + DECISION (currently conflated — see E-EV-6) | `ScoredPayload` | worker: scoring stage (legacy `scoring_engine`) |
| `ENFORCED` | ENFORCEMENT | `EnforcedPayload` | worker: enforcement stage (legacy `enforcement_engine`) |
| `JOB_COMPLETED` | terminal (AUDITABILITY closes) | `JobCompletedPayload` | worker: post-enforcement, success |
| `JOB_FAILED` | terminal (AUDITABILITY closes) | `JobFailedPayload` | worker: exception handler |

Note on `SCORED`/`ENFORCED`: today the worker uses **legacy**
engines (band-only scoring + 3-action enforcement). The new triple
(DecisionEngine + ConfidenceEngine + PolicyEngine) is implemented
but not wired (see `./decision_engine.md` D-DE-2 / `./policy_engine.md`
§17). Target-state event types replacing `SCORED` + `ENFORCED` are
defined in §13.4.

### §3.3 Lifecycle event catalogue

| Event | Emitted at | Payload | latency_ms |
|---|---|---|---|
| `STARTED` | enter `stage_event(job_id, stage)` | none | not set |
| `COMPLETED` | clean exit from `stage_event` | none | wall-clock duration of the block |
| `FAILED` | exception in `stage_event` | `{error_type, error_message}` | wall-clock duration to exception |

Lifecycle events are **stage-scoped**. The worker wraps every
domain stage with `stage_event(...)`; the API does **not** emit
lifecycle events around `POST /v1/ingest` itself (no API-side
`stage_event` block exists today). Adding API-side lifecycle
emission is a Lightweight ADR.

### §3.4 Command events

The system has **no command events on the bus**. The closest
analog is `pipeline_queue.enqueue(...)` (an RQ command, not a bus
event). Command-pattern semantics — submit a request, get a
correlation id back, await completion via events — are implemented
out-of-band by the API endpoint:

- `POST /v1/ingest` is the implicit "begin pipeline" command.
- The response carries `{job_id, status: "queued"}`.
- Completion is observed via `GET /v1/jobs/{id}` (state) and
  `GET /v1/jobs/{id}/events` (timeline).

A future Standard ADR may introduce explicit command events
(`PIPELINE_REQUESTED`, `RETRY_REQUESTED`, …) if the system gains
features like manual replay or operator-triggered re-evaluation.

### §3.5 System events

The system has **no global system events** (broker connection, lag
alerts, configuration reloads, etc.). System-level signals are
emitted via Python `logging` rather than the bus. Adding system
events to the bus is out of scope for this spec; treat the bus as a
**job-scoped audit trail**, not a system telemetry channel.

### §3.6 Failure events

`JOB_FAILED` is the canonical failure terminal event. Per
`pipeline_worker.py`, the worker:

1. Catches any exception during pipeline execution.
2. Calls `job_store.set_failure(job_id, "<type>: <msg>")` —
   transitions QUEUED/PROCESSING → FAILED, sets `failure_reason`.
3. Publishes `JOB_FAILED` with `{error_type, error_message}` (the
   `stage` field is currently always `"job"` per `_STAGE_FOR` —
   stage-context is in the preceding lifecycle FAILED event,
   E-EV-2).
4. Releases the per-job lock.

In addition, the lifecycle layer's `stage_event` context manager
emits a `FAILED` event with the stage name and `{error_type,
error_message}` payload. Both events appear in the timeline; the
lifecycle event names the failing stage, and the `JOB_FAILED`
event closes out the job.

### §3.7 Audit events

There is **no separate "audit event" type**. The entire append-only
event log IS the audit trail. Per A4, every enforcement decision
MUST produce an immutable audit record sufficient for replay /
reconstruction / dispute attribution. This spec maps that mandate
onto the existing two-layer log:

- **Lifecycle events** = operational audit (what stages ran, when,
  for how long, success/failure).
- **Domain events** = decision audit (what the pipeline observed
  and decided at each stage).
- **Job hash** (`job_store`) = consolidated current state and
  enforcement outcome.

The audit-record-shaped projection of the event log is the
responsibility of `docs/security/enforcement_audit.md` *(planned)*
and `docs/specs/storage.md` *(planned)*. This spec defines the
**raw substrate**; the projection layer assembles A4-compliant
records from it.

### §3.8 Replay events

There is **no replay event type today**. Replay is a read-side
operation:

1. `consume_events(job_id)` returns the full log.
2. A replay tool (planned, see `docs/testing/INVARIANT_TESTS.md`
   *(planned)*) reconstructs `DecisionInput` / `ConfidenceInput` /
   `PolicyContext` from the events and re-runs the engines.
3. Mismatch is a P0 violation per A5 (`./policy_engine.md` §12,
   `./confidence_engine.md` §12, `./decision_engine.md` §8).

A future Standard ADR may introduce `REPLAY_REQUESTED` /
`REPLAY_COMPLETED` events to make replay observable on the bus
itself (currently it would happen out-of-band).

---

## §4 Pipeline Execution Model

### §4.1 Phase materialisation

A1's six semantic phases map onto the current runtime as follows:

| A1 Phase | Runtime materialisation | Event(s) |
|---|---|---|
| INGESTION | API endpoint `POST /v1/ingest` (validation + enqueue + emit) | `INGEST_RECEIVED` |
| ANALYSIS | Worker stages `fingerprint`, `embedding`, `matching` (sequential) | `FINGERPRINT_READY`, `EMBEDDING_READY`, `MATCH_FOUND \| MATCH_NOT_FOUND` |
| EVALUATION | Worker stage `scoring` (legacy — band lookup) | `SCORED` (today conflates risk + decision) |
| DECISION | Worker stage `enforcement` (legacy — 3-action selection) | merged into `ENFORCED` today |
| ENFORCEMENT | Worker stage `enforcement` (action emission + state transition) | `ENFORCED`, terminal `JOB_COMPLETED \| JOB_FAILED` |
| AUDITABILITY | Append-only event log accumulating throughout | every event above + lifecycle layer |

A1 forbids reordering of `ANALYSIS → EVALUATION → DECISION →
ENFORCEMENT`. The current worker enforces this by **hard-coded
sequential execution** (§4.3), not by event-routing logic. This is
sufficient for MVP; an event-routed evolution is in §13.4.

### §4.2 Stage isolation

Each stage in `pipeline_worker.run_pipeline` is wrapped in a
`stage_event(job_id, stage_name)` block. Within the block:

- A pure-function engine call (e.g.,
  `fingerprint_engine.compute_fingerprint(payload)`).
- A `job_store.update_stage(job_id, stage_name, output)` write.
- (after the block exits) one or more `publish_event(...)` calls.

The engine call is the only step that performs business logic; the
storage write and event emission are mechanical. Adding I/O to an
engine is forbidden by the engine specs (`./decision_engine.md` §8,
`./confidence_engine.md` §12, `./policy_engine.md` §12). Engines
**MUST NOT** call `publish_event` directly — that's the worker's
responsibility.

### §4.3 Sequential handoff

The current worker is **monolithic-sequential**: every stage runs
in the same Python process, serially, in fixed order:

```
fingerprint → embedding → matching → scoring → enforcement
```

Handoff is by **in-memory variable**, not by event consumption:
the next stage reads from the prior stage's local return value
(e.g., `vector = embedding_engine.embed(content_hash)` then
`match = matching_engine.find_best_match(vector)`).

This is **intentional for MVP**: a single Python process eliminates
cross-process state machinery and lets the pipeline compose against
a single Redis. The drawback is that `pipeline` queue scaling = job
parallelism, NOT stage parallelism. Stage-level fan-out (e.g.,
embedding and fingerprint in parallel) requires breaking the
monolithic worker into per-stage workers.

**Gap E-EV-5** — the fixed sequential order is a hard-coded
property of the worker, not a property derivable from the event
graph. Reordering or parallelising stages today requires editing
`pipeline_worker.py`. Target-state event-routed orchestration is
sketched in §13.4; the migration is a Standard ADR.

### §4.4 Worker boundaries

The worker process boundary is:

- **In-process**: stage execution, engine calls, lock acquisition,
  state transitions, event emission.
- **Cross-process**: API → queue (RQ enqueue), worker → Redis
  (job_store, event_store).

The API process MUST NOT perform stage execution. The worker
process MUST NOT serve HTTP. This is enforced by the deployment
topology (separate processes / containers — see `docker-compose.yml`)
and by the absence of cross-imports (`api/` does not import
`workers/pipeline_worker`; `workers/` does not import FastAPI
routers).

### §4.5 Async guarantees

The API endpoint MUST return without performing pipeline work
(per `.claude/rules/api_contracts.md` and A1's "no stage may block
API response path" derivative). The current implementation:

1. Validates the request (Pydantic).
2. Generates `job_id`.
3. Calls `job_store.create_job(...)` — Redis HSET + WATCH/MULTI/EXEC.
4. Calls `pipeline_queue.enqueue(...)` — RQ Redis push.
5. Calls `publish_event(...)` — Redis SET + ZADD.
6. Returns `{job_id, "queued"}`.

All steps are I/O-bound on Redis but each is single-digit
milliseconds in normal operation. The endpoint synchronously
returns; the heavy work happens in the worker process.

Two ordering observations:

- **Enqueue precedes emit.** The worker can technically dequeue
  and start before `INGEST_RECEIVED` is emitted. Because emit
  ordering on a single Redis is ns-precision and the worker's first
  emission (`STARTED` for fingerprint stage) happens after the
  worker has dequeued + entered `run_pipeline`, in practice the
  per-job timeline is well-ordered. Subscribers MUST NOT assume
  `INGEST_RECEIVED` is the **first** event in the per-job log; they
  MUST assume only that it is **present**.
- **Job creation precedes enqueue.** If enqueue fails, the job
  hash exists but no worker will pick it up. The job remains in
  QUEUED status indefinitely. Cleanup is the operator's
  responsibility today; sliding TTL (`JOB_TTL_SECONDS`) eventually
  reclaims the hash. **Gap E-EV-1**.

### §4.6 No stage-level fan-out

There is no fan-out in the current pipeline. Each domain event has
exactly one consumer (the worker's next stage, by in-memory
variable). The bus is **observe-only** for everyone except the
worker. Future fan-out (e.g., `MATCH_FOUND` triggers both
`scoring` and a `propagation_graph_update`) requires either:

- An RQ-level subscriber per consumer, OR
- A broker-level fan-out primitive (Redis pub/sub, Kafka consumer
  groups, etc.).

Both are out of scope for this spec. The bus today supports
fan-out **on read** (multiple HTTP clients can call
`GET /v1/jobs/{id}/events` concurrently) but not on dispatch.

---

## §5 Delivery Semantics

### §5.1 At-least-once delivery

RQ provides at-least-once delivery: if a worker crashes mid-job
without acknowledging completion, the job MAY be redelivered.
This spec does NOT require, claim, or implement exactly-once
delivery. Anything that depends on exactly-once is a forbidden
coupling (§12.4).

The system's **idempotency contract** (§5.4) is the consumer-side
guarantee that compensates for at-least-once delivery.

### §5.2 Per-job ordering

Events for a single `job_id` are ordered by emission timestamp on
the per-job sorted index. This holds **as long as a single Redis
instance is the backing store** (the current MVP topology). Once
a multi-shard or multi-region broker is introduced (§11.4), per-job
ordering becomes a property of the partition assignment.

The ordering guarantee:

```
events_by_job:{job_id} ordered by ts_ns ASC
⇔ for any (e₁, e₂) emitted in order from any pod,
  e₁.ts_ns < e₂.ts_ns
  AND e₁ appears before e₂ in any consume_events(job_id) result
```

Cross-job ordering is **not guaranteed**.

### §5.3 Stage routing

The lookup `_STAGE_FOR[event_type]` maps each `PipelineEventType`
to a single stage label:

| Event | Stage |
|---|---|
| `INGEST_RECEIVED` | `ingest` |
| `FINGERPRINT_READY` | `fingerprint` |
| `EMBEDDING_READY` | `embedding` |
| `MATCH_FOUND`, `MATCH_NOT_FOUND` | `matching` |
| `SCORED` | `scoring` |
| `ENFORCED` | `enforcement` |
| `JOB_COMPLETED`, `JOB_FAILED` | `job` |

This is a **derived property**, not authoritative. The same stage
may produce multiple event types (e.g., `MATCH_FOUND` /
`MATCH_NOT_FOUND` both come from `matching`). The lookup is used
for log filtering and dashboarding; consumers MUST NOT depend on
the stage value being a primary key.

### §5.4 Idempotency requirements

Every consumer of the pipeline (today: only the worker itself; in
the future: planned subscribers) MUST be idempotent under
redelivery. The current worker's idempotency mechanism is layered:

#### §5.4.1 Distributed lock + state guard (job-level)

```python
def run_pipeline(job_id):
    token = _acquire_lock(job_id)            # SET lock:job:{job_id} NX EX 300
    if token is None:
        return                                # someone else is processing — no-op

    try:
        job = job_store.get_job(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            return                            # state guard — already advanced

        try:
            job_store.update_status(job_id, PROCESSING)
            ... pipeline body ...
        except Exception:
            job_store.set_failure(job_id, ...)
            publish_event(JOB_FAILED, ...)
    finally:
        _release_lock(job_id, token)         # CAD: only release our token
```

**Three layers protect against duplicate delivery:**

1. **NX EX lock** at `lock:job:{job_id}`, 300s TTL, random
   per-attempt token. A second redelivery either fails to take
   the lock (early exit) or succeeds with a fresh token.
2. **State guard** — `if job.status != QUEUED: return`. Even if
   the lock is acquired (e.g., the prior worker's lock TTL expired
   mid-execution), a non-QUEUED status means the job has already
   advanced and the redelivery is a no-op.
3. **Compare-and-delete release** — the Lua script only deletes
   the lock if the current token matches. A worker whose TTL
   expired mid-execution will fail to release a successor's
   lock; the lock auto-expires under TTL.

**Lock TTL hazard**: if a stage takes >300s, the prior worker's
lock TTL expires and a redelivery may take a fresh lock. The state
guard then catches it (PROCESSING ≠ QUEUED → early return). So:

- Duplicate work is **prevented**.
- The original worker continues to completion (it does not check
  the lock mid-execution; it only releases at the end).
- The original worker's `set_failure` and `publish_event` calls
  during cleanup may race against the redelivered worker's
  early-exit, but neither writes to a state the other could
  corrupt: the redelivery exits before any mutation; the original
  completes its single state transition.

#### §5.4.2 Engine-level idempotency

All three Tier-2 engines (DecisionEngine, ConfidenceEngine,
PolicyEngine) are pure deterministic functions. Repeated invocation
with the same inputs produces the same output. This is the
strongest form of idempotency and is guaranteed by A5.

#### §5.4.3 Storage-level idempotency

- `job_store.create_job(job_id, ...)` is idempotent: re-creating
  with the same `job_id` returns the existing job (per the WATCH
  branch in `job_store.py`).
- `job_store.update_status(...)` enforces `_VALID_TRANSITIONS` and
  refuses non-progressive transitions (e.g., COMPLETED → PROCESSING
  raises). Re-submitting the same transition raises if the source
  state has already moved.
- `update_stage(...)` overwrites the stage's entry in the `stages`
  dict. Repeated calls leave the same final state.

#### §5.4.4 Event-level idempotency

The event store has **no deduplication primitive**. Two `emit()`
calls for the same logical event produce two distinct rows with
distinct `event_id` values. This is **acceptable** because
consumers query by `job_id` and process the timeline declaratively;
duplicate events are visible but do not corrupt downstream
semantics.

**Gap E-EV-2** — `emit()` does not check for prior identical
events. Strict deduplication (e.g., a `(job_id, event_type)`
unique constraint with idempotency-key semantics) is a future
Lightweight ADR; current MVP is acceptable because the only
producer is the worker itself, which already has lock + state
guards above the emit call.

### §5.5 Poison events

A "poison event" is an event whose payload causes the consumer to
crash repeatedly under redelivery. Today the only consumer (the
worker) does not consume the bus; it produces. The risk is
inverted:

- A **poison job** is a `job_id` whose ingest payload causes the
  worker to crash. Under at-least-once delivery, RQ does not retry
  failed jobs by default; the worker catches the exception, sets
  status to FAILED, and the job stops there. Subsequent redelivery
  hits the state guard and exits.
- A **poison metadata** is a `job.metadata` payload that crashes
  the engine call. Same outcome: caught, FAILED, terminal.

There is **no poison-event quarantine** (DLQ-like mechanism) today.
Adding one is **Gap E-EV-2**.

### §5.6 Dead-letter semantics

There is **no DLQ** today. Failed jobs:

1. Reach FAILED terminal state.
2. Persist in the job store (subject to `JOB_TTL_SECONDS` sliding
   TTL).
3. Are visible via `GET /v1/jobs/{id}` and
   `GET /v1/jobs/{id}/events`.
4. Are **not** auto-replayed. Re-running a failed job requires
   either:
   - A manual re-ingest with the same payload (produces a new
     `job_id`), OR
   - A planned operator tool that constructs a new pipeline run
     from a stored evidence record.

A canonical DLQ — a separate RQ queue (`pipeline_dlq`) seeded with
poison-classified failures, plus a re-driver — is **planned** under
`docs/specs/job_processing.md` *(planned, next in queue)*. This
spec records the runtime contract; the operational re-drive
mechanics live in the job-processing spec.

### §5.7 Replay safety

Per A5 (DETERMINISTIC REPLAY), re-running the engines with the same
inputs produces the same outputs. Replay safety on the bus side
means:

- Re-reading `consume_events(job_id)` returns the same timeline
  every time (the log is append-only).
- Re-running the engines from the snapshotted inputs reproduces
  the engine outputs (per each engine's determinism spec).
- Re-emitting events as part of replay is **forbidden** — it
  would corrupt the original timeline. Replay is a read-side
  operation; if a replay tool wants to record its findings, it
  emits to a separate replay log (planned).

---

## §6 Failure Model

### §6.1 Failure taxonomy

| Class | Examples | Today's behavior | Future direction |
|---|---|---|---|
| **Transient** | Redis network blip, RQ broker unavailability | RQ surfaces exception → worker `set_failure` → terminal FAILED | Standard ADR introduces retry policy with backoff (Gap E-EV-2) |
| **Permanent** | malformed payload, deterministic engine bug | worker `set_failure` → terminal FAILED | DLQ + operator triage (Gap E-EV-2) |
| **Stage-level** | `embedding_engine.embed` raises mid-pipeline | lifecycle FAILED + `JOB_FAILED` → terminal FAILED, prior stages remain in `stages` dict | future: per-stage retry possible if event-routed (§13.4) |
| **Lock-loss** | stage exceeds 300s, lock TTL expires | original worker continues to completion; redelivery exits via state guard | future: heartbeat-based lock extension |
| **State-corruption** | external write to `job:{id}` hash | not detected; pipeline trusts the read | future: hash chain on state transitions (planned, §11.5) |
| **Broker-loss** | Redis dies mid-pipeline | RQ raises connection error; in-flight events MAY be lost between emit and ZADD | requires multi-broker abstraction (§11.4 + Gap E-EV-9) |

### §6.2 Retry strategy

**There is no retry strategy today.** RQ's `Retry(max=N)` is not
configured. Failed jobs go directly to FAILED terminal state.

The intended retry strategy (Gap E-EV-2; deferred to
`docs/specs/job_processing.md` *(planned)*):

```
Transient → exponential backoff with jitter, max 3 retries
Permanent → no retry; route to DLQ
Stage-level → retry the failing stage only (requires event-routed pipeline)
```

Until the retry strategy is canonicalised, jobs that fail for
transient reasons (e.g., a Redis hiccup) must be re-ingested as
new jobs.

### §6.3 Timeout behavior

Two timeouts are configured:

- **`job_timeout=300`** on `pipeline_queue.enqueue(...)` — RQ
  cancels the worker if it exceeds 5 minutes per job. Cancelation
  raises a `JobTimeoutException` inside the worker.
- **`socket_timeout=5` / `socket_connect_timeout=5`** on the Redis
  connection — protects against a hung Redis call.

A worker hitting `JobTimeoutException` flows through the same
exception handler as any other failure: lifecycle FAILED →
`set_failure` → `JOB_FAILED` → lock release. The 300s value is
**deliberately matched** to the lock TTL (300s) so the lock and
the job timeout expire together.

A timeout is treated as a **permanent failure** today (no retry,
direct to FAILED). The retry strategy in §6.2 will subdivide this:
"timeout under load" → transient retry; "timeout from logic bug"
→ permanent.

### §6.4 Degraded execution

The pipeline does not have a "degraded mode". Every stage either
succeeds or fails the entire job. Partial-success scenarios
(e.g., embedding fails but matching could proceed on fingerprint
alone) are **not supported** today.

This is a deliberate MVP simplification: the engines have a
deterministic input contract, and the worker's job is to construct
those inputs strictly. Allowing degraded inputs is a Standard
ADR — it changes the engine contracts and the audit-record
completeness story.

### §6.5 Partial pipeline continuation

There is **no partial continuation** today. If `embedding` fails,
the job is FAILED — `matching`, `scoring`, `enforcement` do not
run. Per A1, the semantic ordering forbids skipping phases; partial
continuation that respects A1 would require routing logic
(e.g., `EMBEDDING_FAILED` → emit a synthetic embedding zero vector
→ continue) and is out of scope.

The `stages` dict in the job hash retains the outputs of any
stages that **completed before the failure**. Operators can
inspect partial state to diagnose; replay tools can use the
partial state plus the ingest payload to re-run from a known
prefix.

### §6.6 Compensating semantics

The system has **no compensating actions** (no "undo enforcement",
no "retract event"). Per A4 + A6 + this spec's append-only rule:

- A reversal of a prior enforcement decision (e.g., human-review
  overrules an automated TAKEDOWN — `./policy_engine.md` §2.2) is
  recorded as a **new appended entry** in the audit record, not
  by modifying the original `ENFORCED` event.
- The mechanics of the reversal entry live at the platform / audit
  layer (`docs/security/enforcement_audit.md` *(planned)*). This
  spec records only that the eventing layer's append-only
  guarantee is the substrate that makes reversal-by-append
  possible.

A reversal would, in target state, be modelled as a new event
type (e.g., `ENFORCEMENT_REVERSED`) emitted by an authorised
actor. This is a Standard ADR when the security spec lands.

---

## §7 State + Orchestration

### §7.1 Job state machine

Per `backend/app/core/job_store.py::_VALID_TRANSITIONS`:

```
QUEUED ──▶ PROCESSING ──┬──▶ COMPLETED   (terminal, absorbing)
   │                    ├──▶ FLAGGED     (terminal, absorbing)
   │                    └──▶ FAILED      (terminal, absorbing)
   └──▶ FAILED                            (set_failure from QUEUED — defensive)
```

Properties:

- **Terminal states are absorbing.** No transition out of
  COMPLETED / FLAGGED / FAILED. Reversal under A6 produces an
  *append* to the audit log, not a state-machine transition.
- **Skipping states is forbidden.** No QUEUED → COMPLETED direct
  transition. The worker MUST flip QUEUED → PROCESSING before
  reaching a terminal state.
- **Atomic transitions.** Every `update_status` / `set_failure`
  uses Redis WATCH/MULTI/EXEC over the job hash to guard against
  concurrent writers.
- **State + audit are eventually consistent.** The status update
  and the corresponding `JOB_COMPLETED` / `JOB_FAILED` event are
  two separate Redis calls. Between them, the job hash and the
  event log can disagree for tens of microseconds. Consumers MUST
  query both (state + events) and reconcile if they need a strong
  consistency view.

### §7.2 Orchestration boundaries

The orchestration boundary in the current MVP is the
`run_pipeline` function. Its responsibilities:

1. Acquire per-job lock (idempotency).
2. State guard against redelivery.
3. QUEUED → PROCESSING transition.
4. Sequential stage execution (engines + storage writes).
5. Per-stage event emission.
6. Terminal transition (COMPLETED / FLAGGED / FAILED).
7. Lock release.

This is **a single point of orchestration**. There is no
external state machine, no workflow engine, no Step Functions /
Temporal / Cadence. Adding one is a Standard ADR (§13.4).

### §7.3 Retry lifecycle

Today: no retry lifecycle. A failed job is terminal.

Target state (deferred to `docs/specs/job_processing.md`
*(planned)*): the job state machine gains a `RETRYING` transient
state and a `retry_count` field. RQ's `Retry(max=N)` is configured
per-stage. Exhausted retries route to DLQ.

### §7.4 Cancellation semantics

There is **no cancellation primitive today**. A job that the API
client wants to abort cannot be cancelled — the worker will run
to completion or timeout. Cancellation is **out of scope** for
this spec. Adding it requires:

- A new state (`CANCELLED`) added to the state machine.
- A new transition (`PROCESSING → CANCELLED`).
- A worker-side check inside `run_pipeline` that polls a
  cancellation flag in the job hash.
- An API endpoint to set the cancellation flag.

This is a Standard ADR if added.

### §7.5 Replay lifecycle

Replay today is a **read-side operation** (§3.8). It does not
touch the state machine; the original job's terminal state is
preserved. A replay tool can:

1. Read `job_store.get_job(id)` and `consume_events(id)`.
2. Reconstruct engine inputs from the event payloads + stage
   outputs (in `job.stages`).
3. Re-run the engines.
4. Compare outputs to the original audit record.

Replay does NOT mutate the original job. A future replay-as-
first-class-op feature (§3.8) would emit `REPLAY_REQUESTED` /
`REPLAY_COMPLETED` events into a separate log, never into the
original job's timeline.

### §7.6 Cross-job orchestration

There is **no cross-job orchestration** today. Each `job_id`
runs independently. There is no "this job depends on that job"
primitive; no fan-in (waiting for multiple jobs to complete);
no fan-out (one ingest spawning multiple jobs).

The system has architectural facade for cross-content lineage
(`backend/app/core/propagation_graph.py` builds a parent→child
content DAG), but that operates on `content_id` (a derived
identifier), not on `job_id`. Cross-job orchestration on jobs
themselves is out of scope for this spec.

---

## §8 Event Storage

### §8.1 Backing store

Redis is the **sole** backing store for events today. Three
key namespaces:

| Pattern | Type | Purpose |
|---|---|---|
| `event:{job_id}:{ts_ns}` | STRING (JSON blob) | the canonical event record |
| `events_by_job:{job_id}` | ZSET (sorted set) | per-job timeline index, scored by `ts_ns` |
| `lock:job:{job_id}` | STRING (token + EX TTL) | per-job processing lock (§5.4.1) |

Storage operations:

```
emit():
    SET event:{job_id}:{ts_ns} <JSON blob>
    ZADD events_by_job:{job_id} {ts_ns} event:{job_id}:{ts_ns}

list_events(job_id):
    keys = ZRANGE events_by_job:{job_id} 0 -1
    blobs = MGET keys
    return [json.loads(b) for b in blobs]
```

The two storage calls in `emit()` are **not transactional**. A
crash between SET and ZADD leaves the blob orphaned (not in the
index). This is a known durability gap — Gap E-EV-2 family — and
is acceptable for MVP because:

- The blob without an index entry is invisible to `list_events`,
  so consumers don't see a partial state.
- The orphan persists until manual cleanup or `JOB_TTL_SECONDS`
  expiry.

A transactional outbox pattern is the target-state fix
(§13.4).

### §8.2 Append-only enforcement

The event store has **no public update or delete API**. The only
write path is `emit()`. The only read paths are `list_events()`
and `consume_events()`. Append-only is enforced by **API surface
absence**, not by Redis-level constraints (Redis allows arbitrary
overwrites).

This is a **policy guarantee**, not a hardware guarantee.
Consumers MUST trust the API surface; direct Redis access (e.g.,
debugging) MUST NOT mutate event keys. The spec records this as
a P0 violation per `docs/constitution/GOVERNANCE.md` §5: any
discovered mutation of a stored event is a P0 evidence-destruction
violation under A4 + A7.

### §8.3 Retention

**Default: unbounded retention** (events persist forever in
Redis). The `JOB_TTL_SECONDS` setting applies a sliding TTL to the
**job hash** (`job:{job_id}`), but the **event log** keys
(`event:...`, `events_by_job:...`) are NOT subject to the same
TTL today.

This is **Gap E-EV-1**: in production, the event log will grow
without bound. Mitigation paths:

1. Apply a parallel TTL to event keys on every emit (Lightweight
   ADR — additive change).
2. Compaction: write events to durable cold storage (S3-shaped),
   prune Redis after N days (Standard ADR — introduces a new
   storage layer; ties to `docs/specs/storage.md` *(planned)*).
3. Retention-by-business-rule: events tied to enforcement actions
   under appeal MUST persist longer than events for ALLOWed jobs
   (planned in `docs/security/enforcement_audit.md` *(planned)*).

For MVP, operators are expected to size Redis to absorb the load
and to monitor key growth.

### §8.4 Auditability guarantees

Per A4, every enforcement decision MUST be audit-reconstructable.
The eventing layer satisfies A4 via:

- **Per-job event log** (the entire timeline).
- **Stage outputs in `job.stages`** (the engine outputs at each
  step).
- **Job result in `job.result`** (the consolidated terminal
  output).

Together these form the **A4 audit-record substrate**. The
projection of this substrate into the canonical A4 schema (with
fields like `input_id`, `matched_id`, `policy_lineage_ref`,
`engine_lineage`) is the responsibility of
`docs/security/enforcement_audit.md` *(planned)* +
`docs/specs/storage.md` *(planned)*.

### §8.5 Evidence linkage

Per A7, evidence MUST be preserved with sufficient lineage to
reconstruct the decision. The eventing layer contributes:

- **`job_id`** as the primary key linking events, state, and
  evidence.
- **Per-event `event_id`** for fine-grained linkage from external
  records (e.g., a dispute ticket cites
  `event:{job_id}:{ts_ns}`).
- **`engine_lineage` payloads** — `model_version` /
  `config_version` strings carried in the event payloads (e.g.,
  `FingerprintReadyPayload.model_version`,
  `EnforcedPayload.model_version`).

The current `engine_lineage` coverage is partial — model versions
are captured but the policy / decision / confidence config
versions are NOT yet on the bus (the new triple isn't wired —
E-EV-6). Target-state events (§13.4) close this gap.

### §8.6 Replayability requirements

For the bus to be **replayable**, the per-job log MUST contain
sufficient information to reconstruct every input the engines
saw. Today:

- Ingest payload — captured in `job.metadata`, NOT in
  `INGEST_RECEIVED` payload (which only carries
  `content_type`, `source_url`, `has_metadata`).
- Fingerprint output — captured in `job.stages.fingerprint` AND
  `FINGERPRINT_READY` payload.
- Embedding vector — captured in `job.stages.embedding.vector` AND
  `EMBEDDING_READY` payload (only `dimension` + `model_version`,
  the vector itself is in `job.stages` only).
- Match result — captured in `job.stages.matching` AND
  `MATCH_FOUND` / `MATCH_NOT_FOUND` payloads.
- Score / band — captured in `job.stages.scoring` AND `SCORED`
  payload.
- Enforcement decision — captured in `job.stages.enforcement` AND
  `ENFORCED` payload.

Replay therefore requires reading **both** `job.stages` and
`consume_events(job_id)`. This is a known split — Gap E-EV-7
records that the bus alone is not currently sufficient for full
replay; the job hash carries some payloads (e.g., the embedding
vector) that are too big to put on the bus by default. Target-
state replay (`docs/testing/INVARIANT_TESTS.md` *(planned)*)
formalises which subset is required for which invariant.

---

## §9 Observability

### §9.1 Tracing

There is **no distributed tracing** (no OpenTelemetry, no Jaeger,
no Zipkin) today. Tracing is **out of scope for v1.0** of this
spec. The closest substitute is the per-job event log, which
provides per-stage timing via lifecycle `COMPLETED`/`FAILED`
events with `latency_ms`.

A future Standard ADR may introduce an OpenTelemetry layer that
projects events into spans. The **`event_id`** field is the
natural span identifier; **`job_id`** is the natural trace
identifier.

### §9.2 Metrics

The system emits **no metrics today** (no Prometheus, no StatsD,
no internal counters). Metrics are out of scope for v1.0 of this
spec. The closest substitute is reading the per-job event log and
deriving:

- Per-stage latency from `latency_ms` on lifecycle COMPLETED.
- Failure rate from `JOB_FAILED` count vs `JOB_COMPLETED` count.
- Throughput from event timestamps over a window.

A future Standard ADR (anchored in
`docs/specs/observability.md` *(planned)*) introduces metric
emission. Recommended metrics:

| Metric | Type | Cardinality | Notes |
|---|---|---|---|
| `pipeline.events.emitted` | counter | per event_type | rate of emits |
| `pipeline.stage.duration_ms` | histogram | per stage | from lifecycle events |
| `pipeline.jobs.terminal` | counter | per terminal status | COMPLETED / FAILED / FLAGGED rates |
| `pipeline.queue.depth` | gauge | global | RQ queue length |
| `pipeline.worker.active` | gauge | per worker | currently-processing job count |
| `pipeline.lock.acquired` / `lock.contended` | counter | global | redelivery signal |

### §9.3 Lineage visibility

Lineage is visible via:

- `GET /v1/jobs/{id}` — full job state including `stages` map.
- `GET /v1/jobs/{id}/events` — full per-job timeline.
- (target state) audit storage layer projecting events into A4
  records.

The current API surfaces are sufficient for **debugging** but
not for **operator-grade audit search** (no cross-job query, no
search-by-action, no search-by-error-class). Adding a query layer
is the responsibility of `docs/specs/storage.md` *(planned)*.

### §9.4 Correlation propagation

`job_id` is the only correlation primitive. It propagates from
the API endpoint into the job hash, the queue payload, the
worker, and every event emitted during the job's lifetime.
Recommended propagation patterns for future client integrations:

- API responses include `job_id`.
- Subsequent client calls echo `job_id` in headers (e.g.,
  `X-Job-Id`) for log correlation.
- External systems (e.g., the LLM orchestrator in
  `app/services/llm/`) attach `job_id` to their internal traces
  when they participate in a job's pipeline.

`request_id` (an HTTP-level identifier distinct from `job_id`)
is **not propagated** today. Adding it is a Lightweight ADR if
the API gains middleware that sets `X-Request-Id`. Until then,
`job_id` doubles as `request_id`.

### §9.5 Event-debugging requirements

The minimum bar for debugging a failed job:

1. Look up the job: `GET /v1/jobs/{id}` → see `failure_reason`,
   `stages` map (which stages completed).
2. Read the timeline: `GET /v1/jobs/{id}/events` → see ordered
   events including the lifecycle `FAILED` event with
   `error_type` / `error_message`.
3. Cross-reference with `app/workers/pipeline_worker` source
   to find the failing stage.

This is **operator-grade**, not auditor-grade. An auditor-grade
debugging surface (replay, what-if analysis, cross-job search)
is planned downstream.

### §9.6 Audit trace guarantees

The per-job timeline is a **complete record of stage transitions
for that job**. It satisfies the A4 audit-completeness floor for
that single job. Cross-job audit (e.g., "show me all enforcement
TAKEDOWNs in the last 24 hours") requires either:

- Iteration over jobs (not scalable).
- A projection layer that indexes events by event_type and time
  (planned, `docs/specs/storage.md` *(planned)*).

The eventing layer guarantees the **substrate**; the projection
layer enables operator-grade query.

---

## §10 Security + Governance

### §10.1 Immutability

The event log is append-only by API surface (§8.2). Mutating an
existing event is a **P0 violation** per
`docs/constitution/GOVERNANCE.md` §5. This is the
runtime materialisation of A4 ("the audit record MUST be
append-only").

### §10.2 Access control

The current implementation has **no authentication or
authorization** on event reads or writes. `GET /v1/jobs/{id}/events`
is publicly accessible; any client with a valid `job_id` can read
the full timeline. Direct Redis access (debugging, ops tooling)
has whatever ACLs are configured at the Redis layer.

This is **acceptable for MVP** but is a known gap for production:

- Per-job ACLs (caller can only read their own jobs) belong to
  `docs/security/secrets_policy.md` *(planned)* / API-auth spec.
- Redis-level ACLs (production hardening) belong to platform
  spec.

This spec **records** the gap; remediation is delegated.

### §10.3 Forbidden mutations

In addition to event mutation (§10.1), the following are
forbidden at the runtime layer:

- Deleting events from the log (`DEL event:...`).
- Reordering events in the index (`ZADD` with adjusted scores).
- Writing events with `event_id`s reused from a prior emit
  (would corrupt fine-grained linkage).
- Writing events with backdated timestamps (would corrupt the
  per-job ordering invariant).
- Bypassing `publish_event` to inject untyped payloads onto the
  bus.

Each is a **P0** violation per GOVERNANCE.md §5.

### §10.4 Schema evolution

Payload schemas are defined as Pydantic models in
`backend/app/core/event_store.py`. Evolution rules:

| Change | ADR tier | Migration |
|---|---|---|
| Add a new optional field to a payload | Lightweight | none — old consumers ignore unknown fields; new consumers default |
| Add a new required field to a payload | Standard | producer + consumer migrate together; old events without the field cannot replay against the new schema |
| Rename a payload field | Standard | breaking; deprecation cycle if STABLE |
| Remove a payload field | Standard | breaking; deprecation cycle if STABLE |
| Change a payload field's type | Standard | breaking; deprecation cycle if STABLE |
| Add a new `PipelineEventType` value | Standard | additive on the producer side; consumers must handle unknown values gracefully |
| Rename / remove an existing `PipelineEventType` value | Standard | breaking |
| Change the `Event` envelope shape | Standard | breaking; affects every consumer |

The absence of a `schema_version` field (Gap E-EV-3) makes
evolution risky: replays of historical events against a future
schema can deserialize silently against the new shape. The
recommended near-term mitigation:

1. Always make additive changes Lightweight; preserve backward
   compatibility.
2. Treat any non-additive change as a **major** version bump on
   this spec, gated by a Standard ADR.
3. When `schema_version` is added (planned, §15), historical
   events get version `0` retroactively and the deserialization
   path can dispatch by version.

### §10.5 Governance ownership

- **pipeline domain** owns this spec, the `Event` envelope, the
  `PipelineEventType` enum, the `EventType` lifecycle enum, the
  `_PAYLOAD_SCHEMA` registry, the storage key namespaces (`event:`,
  `events_by_job:`, `lock:`), the worker boundary contract, and
  the at-least-once / per-job-ordering guarantees.
- **api domain** owns `IngestRequest` / `IngestResponse` shapes
  (consumed at `POST /v1/ingest`).
- **policy / decision / confidence domains** consume the bus
  output but do NOT own event types directly. New event types
  arising from those domains' rollouts (§13.4) are jointly
  authored: the producing domain proposes the event, pipeline
  domain ratifies the schema and adds it to the registry.
- **security domain** owns `docs/security/enforcement_audit.md`
  *(planned)* which projects the bus into A4 records and adds
  reversal events.

### §10.6 ADR requirements

Modifications to this spec MUST:

1. Bump `version:` per §1.3 / §16.
2. Reference an ADR in `adr_references:` if the change is
   anything other than a documentation clarification.
3. Update the implementation in lockstep (Pydantic models, the
   `PipelineEventType` enum, and the worker emission sites).
4. Notify api / policy / decision / confidence / security domains
   when modifying any event type, payload, or envelope field —
   these are cross-domain consumer surfaces.
5. Provide a migration plan for any non-additive change (per
   §10.4 table).

---

## §11 Scaling Model

### §11.1 Current capacity envelope

The MVP runtime supports:

- **One Redis instance** as the broker, lock store, event store,
  and job store.
- **One `pipeline` queue** (no per-content-type partitioning).
- **N worker processes** consuming the same queue (RQ
  natively load-balances).
- **Sequential per-job execution** (one worker handles one job
  end-to-end).

Realistic per-job latency (back-of-envelope, not benchmarked):

- API response: < 50ms (validate + Redis writes + return).
- Job pipeline: dominated by stage cost; current stubs are sub-
  second; embedding/matching against real models will dominate.

### §11.2 Partitioning

There is **no partitioning** today. All jobs flow through the
single `pipeline` queue. A job's per-job state (hash, lock,
events) lives on whichever Redis instance the connection points
to.

Target-state partitioning options (Standard ADR):

- **By tenant**: `pipeline:{tenant_id}` queue per tenant; per-
  tenant Redis instances; per-tenant rate limits.
- **By content type**: `pipeline:video`, `pipeline:image`
  queues, allowing dedicated worker fleets for heavy media.
- **By region / jurisdiction**: GDPR / DSA-driven sharding.

Each option is a Standard ADR; this spec records that partitioning
is feasible without violating A1 (since A1 is orchestration-
agnostic) but requires the **per-job-ordering** guarantee to
become **per-partition-job-ordering** (the partition becomes the
new ordering scope).

### §11.3 Consumer-group semantics

RQ does not provide consumer groups. All workers consume from a
single queue with at-most-one-claim-per-job (the BLPOP semantic).
Adding consumer-group semantics — e.g., "events of type
`MATCH_FOUND` go to consumers in group A; events of type
`ENFORCED` go to consumers in group B" — requires either:

- A second RQ queue per consumer group (workable but verbose).
- A different broker (Kafka, NATS, Redis Streams) that has
  consumer groups natively (§11.4).

Today the only consumer of pipeline events **on the bus** is the
worker itself, by in-memory variable handoff (§4.3). Other
"consumers" (the API endpoints that read the log) operate on
read, not on dispatch.

### §11.4 Future broker abstraction

A multi-broker abstraction (so the runtime can swap Redis for
Kafka / NATS / SQS / Pub-Sub without rewriting engines) is **a
target-state design** but is NOT implemented today. Adding it
requires:

1. Defining a `PipelineBus` Protocol with `publish` / `consume` /
   `subscribe` methods.
2. Implementing a `RedisRQBus` adapter (existing wiring).
3. Implementing a per-broker adapter (e.g., `KafkaBus`).
4. Swapping the import in `event_store.py` to depend on the
   Protocol, not the concrete `redis_conn`.

This is a Standard ADR and a sizable refactor. Until it lands,
the spec records the broker abstraction as **future work** and
forbids depending on broker-specific features in engine code
(§12.4).

> **Constitutional reminder.** The user-facing system contract
> guaranteed by this spec is "at-least-once delivery, per-job
> ordering, append-only audit". It does NOT guarantee Kafka,
> Redis, or any specific broker. Adopters MUST NOT design
> consumers around broker-specific affordances.

### §11.5 Backpressure

There is **no backpressure mechanism** today. If ingest exceeds
worker throughput, the `pipeline` queue grows without bound.
Mitigations:

- Operator monitors RQ queue depth and provisions more workers.
- API gateway rate limits (out of scope of this spec).
- Optional: future API-side check that rejects ingest when queue
  depth exceeds a threshold (Lightweight ADR).

This is **acceptable for MVP** because the API has no SLA-bound
latency for the pipeline (API returns in <50ms; pipeline can take
arbitrarily long). Production deployment SHOULD configure either
gateway rate limits or queue-depth-based backpressure.

### §11.6 Fan-out

Fan-out is **not supported on dispatch** (§4.6). Multiple
**read** subscribers can scan the per-job event log
concurrently — this is unlimited because Redis ZRANGE is read-
only. Multiple **write** subscribers (e.g., one event triggers
N independent downstream pipelines) require either:

- Multiple RQ queues with explicit fan-out at the producer.
- Broker-level fan-out (Kafka topics, NATS subjects, Redis
  Streams).

Either is a Standard ADR. The current bus MUST NOT be
overloaded with fan-out semantics it doesn't natively support.

---

## §12 Extensibility Rules

### §12.1 What may evolve safely (Lightweight ADR)

- Adding a new optional field to an existing payload class.
- Adding a new lifecycle event type (subordinate to the
  `EventType` enum).
- Adjusting the `_STAGE_FOR` mapping when a new event type is
  added.
- Adjusting RQ `job_timeout` / lock TTL / Redis socket timeouts.
- Documentation refinements with no semantic change.
- Sliding `JOB_TTL_SECONDS` defaults.
- Applying a parallel TTL to event keys (Gap E-EV-1
  remediation step 1).

### §12.2 What requires Standard ADR

- Adding a new `PipelineEventType`.
- Renaming or removing an existing `PipelineEventType`.
- Renaming or removing a payload field.
- Changing a payload field's type.
- Adding `schema_version` to the envelope (Gap E-EV-3
  remediation).
- Adding `causation_id` / `parent_event_id` to the envelope
  (Gap E-EV-4 remediation).
- Introducing a DLQ + retry policy (Gap E-EV-2 remediation;
  jointly with `docs/specs/job_processing.md` *(planned)*).
- Splitting `pipeline` queue into multiple queues.
- Switching broker (Redis → Kafka / NATS / etc.).
- Introducing partitioning (per-tenant, per-content-type, per-
  region).
- Introducing fan-out / consumer-group semantics.
- Adding cancellation primitives (§7.4).
- Adding event-routed orchestration (§13.4).
- Wiring the new engine triple (Decision/Confidence/Policy) and
  retiring legacy `scoring_engine` / `enforcement_engine`
  (Gap E-EV-6).

### §12.3 What requires Constitutional ADR

- Removing the at-least-once delivery floor.
- Removing the per-job ordering guarantee.
- Removing append-only retention.
- Removing the worker-boundary rule (allowing engines to call
  `publish_event` directly).
- Reordering A1 semantic phases.
- Allowing exactly-once delivery claim that the broker cannot
  back.

### §12.4 Forbidden coupling (anti-patterns)

- **Engines emitting events.** The DecisionEngine,
  ConfidenceEngine, and PolicyEngine MUST NOT call
  `publish_event` or `emit`. They are pure deterministic
  functions per their respective specs (`./decision_engine.md`
  §8, `./confidence_engine.md` §12, `./policy_engine.md` §12).
  The pipeline worker is the sole emitter for engine outputs.
- **Consumers depending on broker-specific features.** No
  consumer code may rely on Redis-specific behavior (sorted-set
  scoring, key naming patterns) for its semantics. Consumers
  use the public APIs (`consume_events`, `list_events`,
  `publish_event`).
- **Cross-domain emit calls.** The api domain emits
  `INGEST_RECEIVED`; the pipeline domain emits everything else.
  Other domains (policy, decision, confidence, security) do NOT
  emit on the bus directly. Instead, they hand outputs to the
  worker, which emits.
- **Bypassing the `publish_event` validator.** The Pydantic
  payload validation in `publish_event` is the schema gate. No
  caller may use raw `emit()` with a domain event type.
- **Mutating a stored event.** P0 violation per §10.3.
- **Deleting a stored event.** P0 violation per §10.3.
- **Reading the bus to make pipeline decisions inside an engine.**
  Engines are pure — they consume their typed input and produce
  their typed output. The bus is for the worker / observers.
- **Treating event ordering as a global stream.** Per-job
  ordering is the only guarantee; cross-job timeline reads must
  reconcile with explicit `job_id` partitioning.
- **Inferring causation from timestamps alone.** Until E-EV-4
  lands, two events with adjacent timestamps in the same `job_id`
  log are *correlated*, not necessarily *causally linked*. The
  worker's hard-coded order is the only causation source today.

### §12.5 Cross-domain invariants

- **One enforcement decision = one A4 audit record.** The
  per-job event log + the job hash + (target state) the audit
  storage projection together satisfy A4.
- **Replay-attribution holds end-to-end.** Each engine guarantees
  function-level determinism (per its spec); the eventing layer
  guarantees the inputs to each engine call are recoverable from
  the per-job log + stages dict (§8.6). End-to-end replay is the
  composition.
- **Idempotency is the consumer's responsibility.** The bus
  guarantees at-least-once delivery; consumers MUST be
  idempotent. The current single consumer (the worker) is
  idempotent via §5.4 mechanisms.
- **The bus is a substrate, not a workflow engine.** Job-to-job
  dependencies, conditional routing, scheduled retries — all
  out of scope. The bus carries events; orchestration logic
  lives in the worker.

---

## §13 Current vs Target State

### §13.1 Implemented runtime (as of v1.0)

| Component | Status | Notes |
|---|---|---|
| Redis-backed JobStore | **IMPLEMENTED** | `job_store.py` with WATCH/MULTI/EXEC. STATE.md is stale (says in-memory) — Gap E-EV-8. |
| RQ pipeline queue | **IMPLEMENTED** | `queue.py`. STATE.md is stale (says EXPERIMENTAL — not built). |
| RQ worker | **IMPLEMENTED** | `worker.py`. STATE.md is stale (says EXPERIMENTAL — not built). |
| Two-layer event store | **IMPLEMENTED** | `event_store.py` with lifecycle + canonical pipeline events. |
| Per-job event timeline | **IMPLEMENTED** | sorted-set index, ns-precision ordering. |
| Distributed lock + state guard | **IMPLEMENTED** | `pipeline_worker.py` SET NX EX 300 + Lua CAD release + state double-check. |
| Sequential pipeline (fingerprint → embedding → matching → scoring → enforcement) | **IMPLEMENTED** | hard-coded order in `pipeline_worker.run_pipeline`. |
| `INGEST_RECEIVED` emission | **IMPLEMENTED** | API boundary. |
| Lifecycle events (STARTED / COMPLETED / FAILED) | **IMPLEMENTED** | wraps every stage. |
| 9 canonical pipeline event types | **IMPLEMENTED** | with Pydantic-validated payloads. |
| Append-only event log | **IMPLEMENTED** | by API-surface absence (§10.1). |
| State machine with absorbing terminals | **IMPLEMENTED** | `_VALID_TRANSITIONS` enforced. |
| Sliding job-hash TTL | **IMPLEMENTED** | `JOB_TTL_SECONDS`. |

### §13.2 Partially implemented runtime

| Component | Status | Gap |
|---|---|---|
| Engine-triple integration | **PARTIAL** | DecisionEngine + ConfidenceEngine + PolicyEngine implemented as library code; pipeline worker uses legacy `scoring_engine` + `enforcement_engine`. **E-EV-6**. |
| Engine lineage in events | **PARTIAL** | model_version captured (fingerprint, embedding); config_version for decision / confidence / policy NOT yet on bus (depends on E-EV-6). |
| Audit projection layer | **PARTIAL** | per-job event log is the substrate; A4-record projection is unimplemented (planned: `docs/security/enforcement_audit.md`). |
| Operator-grade replay | **PARTIAL** | inputs are reconstructable from `job.stages` + events; no replay tool exists. |
| Postgres durable mirror | **PROPOSED** | `propagation_graph.py` and `content_registry.py` reference Postgres mirrors but they are NOT implemented; Redis is the sole store today. |

### §13.3 Target-state runtime (deferred to ADR)

| Capability | Target spec / ADR |
|---|---|
| DLQ + retry policy | `docs/specs/job_processing.md` *(planned)* — Standard ADR |
| Bounded event-log retention | this spec §15 — Lightweight ADR (parallel TTL) → Standard ADR (cold storage) |
| `schema_version` on envelope | this spec §15 — Standard ADR |
| `causation_id` on envelope | this spec §15 — Standard ADR |
| Event-routed orchestration (replace monolithic worker) | this spec §15 — Standard ADR |
| Engine-triple wiring | `docs/specs/job_processing.md` *(planned)* + this spec §13.4 — Standard ADR |
| Cross-broker abstraction | this spec §11.4 — Standard ADR |
| Partitioning (tenant / content-type / region) | this spec §11.2 — Standard ADR |
| Fan-out / consumer groups | this spec §11.6 — Standard ADR |
| Authentication on event reads | `docs/security/secrets_policy.md` *(planned)* — Standard ADR |
| OpenTelemetry tracing | `docs/specs/observability.md` *(planned)* — Standard ADR |
| Metrics emission | `docs/specs/observability.md` *(planned)* — Standard ADR |
| Cancellation primitive | this spec §7.4 — Standard ADR |
| Reversal event type | `docs/security/enforcement_audit.md` *(planned)* — Standard ADR |

### §13.4 Engine-triple wiring (the highest-leverage runtime closure)

The currently-deferred runtime change is replacing the legacy
scoring + enforcement stages with a target-state EVALUATION +
DECISION pair invoking the new engine triple. Sketch:

```
worker.run_pipeline (target state):

  with stage_event(job_id, "fingerprint"):  ... (unchanged)
  publish FINGERPRINT_READY                  ... (unchanged)

  with stage_event(job_id, "embedding"):    ... (unchanged)
  publish EMBEDDING_READY                    ... (unchanged)

  with stage_event(job_id, "matching"):     ... (unchanged)
  publish MATCH_{FOUND|NOT_FOUND}            ... (unchanged)

  # NEW — EVALUATION phase split
  with stage_event(job_id, "evaluation"):
      risk = decision_engine.compute_risk(decision_input, threshold_config)
      confidence = confidence_engine.compute_confidence(conf_input, conf_config)
      decision_output = build_decision_output(risk, input_snapshot, ...)
  publish RISK_SCORED(composite, band, breakdown, decision_config_version)
  publish CONFIDENCE_COMPUTED(composite, tier, triggered_conditions, conf_config_version)

  # NEW — DECISION phase
  with stage_event(job_id, "decision"):
      result = policy_engine.evaluate_policy(decision_output, confidence, policy_context)
  publish POLICY_DECIDED(action, triggered_rules, evaluation_hash,
                         policy_version, decision_config_version, conf_config_version)

  # ENFORCEMENT (already exists; semantics evolve to 5-action ladder)
  with stage_event(job_id, "enforcement"):
      enforcement_engine.apply(action, ...)
  publish ENFORCED(...)                      # payload extended for 5-action

  publish JOB_COMPLETED(...) | JOB_FAILED(...)
```

New event types (all Standard ADR — bundled):

- `RISK_SCORED` — payload: `{composite, band, breakdown_summary,
  decision_config_version}`.
- `CONFIDENCE_COMPUTED` — payload: `{composite, tier,
  triggered_conditions, conf_config_version}`.
- `POLICY_DECIDED` — payload: `{action, triggered_rules,
  evaluation_hash, policy_version, decision_config_version,
  conf_config_version}`.

The current `SCORED` event type is retired (or kept for backward
compatibility with the legacy stage during a transition window).
The current `ENFORCED` payload schema is extended to carry the
5-action `PolicyAction` value.

This is **the single highest-leverage closure** in the runtime.
It is not part of v1.0 of this spec — it is the next-step ADR
that this spec enables. See §15 / Gap E-EV-6.

---

## §14 Reconciliation history

This spec consolidates `.claude/rules/eventing.md` (TRANSITIONAL)
with the actual implementation. Major reconciliations:

### §14.1 E-EV-1 — Unbounded event-log retention

**Drift:** `JOB_TTL_SECONDS` applies a sliding TTL to the job
hash but NOT to event keys (`event:...`,
`events_by_job:...`). Per A4 + A7 + this spec's append-only
guarantee, evidence MUST persist long-term — but unbounded
in-memory persistence in Redis is operationally fragile.

**Resolution adopted:** §8.3 records the gap. Two-step
remediation:

1. **Step 1 (Lightweight ADR):** apply a parallel TTL to event
   keys at every emit, mirroring `JOB_TTL_SECONDS`. Same
   sliding semantics as the job hash.
2. **Step 2 (Standard ADR):** introduce a cold-storage layer
   (S3-shaped) and compaction policy. Events older than N days
   move to cold storage; Redis prunes after migration.

This spec records the gap; remediation is staged via ADR.

### §14.2 E-EV-2 — No DLQ / no retry policy

**Drift:** RQ's retry primitives are unconfigured; failed jobs
land in FAILED terminal with no re-driver. Per
`.claude/rules/eventing.md` ("Failed events → retry queue. Max
retries → dead letter queue (DLQ).") this is an explicit gap.

**Resolution adopted:** §6.2 records the gap. Resolution is
deferred to `docs/specs/job_processing.md` *(planned, next in
queue)*. That spec defines:

- Retry policy per failure class (transient / permanent / stage-
  level).
- Backoff strategy (exponential + jitter).
- DLQ topology (separate `pipeline_dlq` queue).
- Re-driver tool (operator-triggered re-run).

Until that spec lands, the current "FAILED is terminal" behavior
is canonical for this spec but is **explicitly insufficient** for
production deployment.

### §14.3 E-EV-3 — No event schema versioning

**Drift:** the `Event` envelope has no `schema_version` field.
Replay against a future codebase whose payload schemas have
evolved can deserialize silently against the new shape.

**Resolution adopted:** §2.5 + §10.4 document the gap and the
near-term mitigations (additive-only changes; major spec bump
on non-additive changes). Adding `schema_version` is a Standard
ADR (envelope schema change + consumer migration); §15 records
it as open work.

### §14.4 E-EV-4 — No causation chain

**Drift:** events are linked only by `job_id` (correlation).
Causation (event B was caused by event A) is implicit in the
worker's hard-coded order, not explicit on the envelope.

**Resolution adopted:** §2.3 documents the gap. Adding
`parent_event_id` is a Standard ADR. The same ADR may add
`request_id` for HTTP-level correlation distinct from `job_id`.

### §14.5 E-EV-5 — Pipeline order hard-coded in worker

**Drift:** stage order (fingerprint → embedding → matching →
scoring → enforcement) is sequenced in `pipeline_worker.run_pipeline`,
not derived from event types or a workflow definition.

**Resolution adopted:** §4.3 + §13.4 document the gap. A
target-state event-routed orchestration is sketched in §13.4.
Migration is a Standard ADR — likely bundled with E-EV-6.

### §14.6 E-EV-6 — Engine triple not wired

**Drift:** `decision_engine.compute_risk`,
`confidence_engine.compute_confidence`,
`policy_engine.evaluate_policy` exist as deterministic library
code but are NOT invoked by `pipeline_worker.run_pipeline`. The
worker uses legacy `scoring_engine` (band lookup) +
`enforcement_engine` (3-action selection).

**Resolution adopted:** §3.2, §13.2, §13.4 document the gap.
This is **the highest-priority runtime closure**, anchored in
`docs/specs/job_processing.md` *(planned, next)* + a Standard ADR
that:

- Adds new event types (`RISK_SCORED`, `CONFIDENCE_COMPUTED`,
  `POLICY_DECIDED`) to the canonical taxonomy.
- Materialises the `DecisionOutput` envelope (per
  `./decision_engine.md` D-DE-1).
- Rewires `pipeline_worker` to call the engine triple.
- Extends `ENFORCED` payload to carry the 5-action
  `PolicyAction`.
- Marks `scoring_engine.py` and `enforcement_engine.py` as
  DEPRECATED in `docs/state/STATE.md`.

This single ADR closes E-EV-6 and substantially closes
`./decision_engine.md` D-DE-2.

### §14.7 E-EV-7 — Auxiliary stores don't emit events

**Drift:** `content_registry.py`, `observation_store.py`, and
`propagation_graph.py` are storage facades but do NOT emit
canonical pipeline events. Operations against them (registering
a new content_id, observing a sighting, adding a propagation
edge) are invisible on the bus.

**Resolution adopted:** §3.2 implicitly records the gap (no
event types for these operations). Adding them is a Standard
ADR per store. Examples of candidate event types:

- `CONTENT_REGISTERED(content_id, owner, trust_level, ...)`.
- `OBSERVATION_RECORDED(observation_id, content_id, source_url, platform, ...)`.
- `PROPAGATION_EDGE_ADDED(parent_id, child_id, transformation, ...)`.

These are out of scope for v1.0 of this spec because the auxiliary
stores are not currently part of the synchronous pipeline path.
Adding them when they integrate is a future Standard ADR.

### §14.8 E-EV-8 — STATE.md stale

**Drift:** `docs/state/STATE.md` lists:

- "In-memory JobStore | ACTIVE (MVP-only) | v0.1" — actually
  Redis-backed.
- "Event bus | EXPERIMENTAL | not built" — actually built (RQ +
  Redis event log).
- "Worker fleet | EXPERIMENTAL | not built" — actually built (RQ
  worker).

**Resolution adopted:** §13.1 documents the actual state. STATE.md
should be updated in a follow-up PR (Tier-3 OPERATIONAL — no ADR
required, just PR review per
`docs/constitution/GOVERNANCE.md` §1). Updating STATE.md is
explicitly **out of scope for this spec landing** but is the
highest-priority follow-up alongside E-EV-6.

### §14.9 E-EV-9 — Single broker; no multi-broker abstraction

**Drift:** the system depends on Redis directly (both as broker
and as state store). No `PipelineBus` Protocol exists; engines
and workers import `redis_conn` and use Redis-specific calls.

**Resolution adopted:** §11.4 records the gap. Cross-broker
abstraction is a Standard ADR + sizable refactor; NOT part of
v1.0. The spec records that consumers MUST NOT design around
Redis specifics (§12.4).

### §14.10 E-EV-10 — Lifecycle + domain events co-mingled

**Drift:** `event_store.py` stores both `EventType` (lifecycle)
and `PipelineEventType` (domain) events in the same per-job
sorted set. `consume_events(job_id, event_types=...)` permits
filtering, but new consumers may be confused by the two-layer
shape.

**Resolution adopted:** §3.1 documents the layering as
**deliberate**, not accidental. The unified log is the per-job
audit timeline. Splitting them across separate keys would
fragment the audit timeline and is explicitly forbidden. The
spec canonicalises the dual-layer shape; it is NOT a gap.

### §14.11 Documentation lineage

| Source | Status | Location |
|---|---|---|
| `.claude/rules/eventing.md` | TRANSITIONAL — superseded by this spec for the partial scope in frontmatter; constitutional parts (idempotency, async, no-sync-ML) also reflected in A1 / A5 | `.claude/rules/` |
| `.claude/rules/job_system.md` | TRANSITIONAL — partially supersedes by this spec for queue / worker semantics; remaining job-state / retry semantics will move to `docs/specs/job_processing.md` *(planned)* | `.claude/rules/` |
| `.claude/rules/job-processing.md` | TRANSITIONAL — same as above; canonical successor is `docs/specs/job_processing.md` *(planned)* | `.claude/rules/` |

Per the append-only migration constraint, none of the above
sources is deleted. They are annotated with `superseded by:`
deprecation notes when next edited.

---

## §15 Open questions / Future work

Documented for visibility; not commitments.

- **Bounded event-log retention** (E-EV-1) — Lightweight ADR
  (Step 1: parallel TTL on event keys); Standard ADR (Step 2:
  cold storage + compaction). Step 1 is the easiest and
  highest-leverage near-term win.
- **DLQ + retry policy** (E-EV-2) — anchored in
  `docs/specs/job_processing.md` *(planned, next in queue)*.
- **`schema_version` on envelope** (E-EV-3) — Standard ADR;
  bundled with envelope evolution if `causation_id` is also
  added.
- **`causation_id` / `parent_event_id` + `request_id`**
  (E-EV-4) — Standard ADR. Adds explicit causation chain to the
  envelope; resolves implicit-causation hazard (§2.3).
- **Event-routed orchestration** (E-EV-5) — Standard ADR.
  Replaces monolithic `run_pipeline` with per-stage workers
  consuming specific event types. Bundles with E-EV-6.
- **Engine-triple wiring** (E-EV-6) — **the highest-leverage
  runtime closure**. Anchored in `docs/specs/job_processing.md`
  *(planned, next)*. New event types (`RISK_SCORED`,
  `CONFIDENCE_COMPUTED`, `POLICY_DECIDED`); legacy
  `scoring_engine` + `enforcement_engine` deprecation.
- **Auxiliary store events** (E-EV-7) — Standard ADR per store.
  Defers until those stores integrate into the synchronous
  pipeline path.
- **STATE.md sync** (E-EV-8) — PR-review-only (Tier-3
  OPERATIONAL); no ADR required. Highest-priority follow-up.
- **Multi-broker abstraction** (E-EV-9) — Standard ADR +
  refactor. Defer until production scale forces the issue.
- **Reversal event type** — Standard ADR; jointly with
  `docs/security/enforcement_audit.md` *(planned)*.
- **Replay event types** (`REPLAY_REQUESTED`, `REPLAY_COMPLETED`) —
  Standard ADR; jointly with `docs/testing/INVARIANT_TESTS.md`
  *(planned)* + replay-tool design.
- **Backpressure mechanism** (§11.5) — Lightweight ADR (queue-
  depth gate at API).
- **Per-tenant / per-content-type / per-region partitioning**
  (§11.2) — Standard ADR per dimension.
- **Cancellation primitive** (§7.4) — Standard ADR.
- **Authentication on event reads** — anchored in API-auth
  spec *(planned)*.

> **Important constraint reminder.** Switching broker (Redis →
> Kafka / NATS / etc.) is a **Standard ADR**. Removing
> at-least-once delivery, per-job ordering, or append-only
> retention is **Constitutional**. Adding exactly-once-claim
> semantics that the broker cannot back is **forbidden**.

---

## §16 Versioning and Change Process

This spec is **EVOLVING** per
`docs/constitution/GOVERNANCE.md` §8. Compatibility expectations
are low — consumers should expect change at each minor bump.

| Change type | ADR tier |
|---|---|
| Doc clarification (no semantic change) | none |
| Adding optional field to existing payload | Lightweight |
| Adding lifecycle event subtype | Lightweight |
| Adjusting RQ timeouts / lock TTL / Redis socket timeouts | Lightweight |
| Sliding `JOB_TTL_SECONDS` defaults | Lightweight |
| Parallel TTL on event keys (E-EV-1 step 1) | Lightweight |
| Backpressure gate at API | Lightweight |
| Adding a new `PipelineEventType` | Standard |
| Renaming / removing `PipelineEventType` | Standard |
| Adding a required field to a payload | Standard |
| Renaming / removing / type-changing a payload field | Standard |
| Envelope schema change (`schema_version`, `causation_id`, …) | Standard |
| DLQ + retry policy | Standard |
| Cold-storage retention | Standard |
| Multi-broker abstraction | Standard |
| Partitioning (any dimension) | Standard |
| Fan-out / consumer groups | Standard |
| Cancellation primitive | Standard |
| Engine-triple wiring | Standard |
| Event-routed orchestration | Standard |
| Reversal event type | Standard |
| Removing at-least-once / per-job-order / append-only | Constitutional |
| Reordering A1 phases | Constitutional |
| Removing the worker-boundary rule | Constitutional |

There is no `event_store_version` constant in
`backend/app/core/event_store.py` today (Gap E-EV-3 family).
When introduced, it MUST be bumped in lockstep with this spec's
`version:` field. Mismatch is a **P1** governance violation.

### §16.1 Graduation to STABLE

Same gates as the engine specs (`./policy_engine.md` §16.1 /
`./confidence_engine.md` §16.1 / `./decision_engine.md` §16.1),
plus eventing-specific gates:

1. The event taxonomy and envelope schema are unchanged for at
   least one minor revision cycle.
2. No production incidents implicating eventing logic in 90 days.
3. Consumer integrations (api, security, qa) report stability.
4. The invariant test suite (`docs/testing/INVARIANT_TESTS.md`
   *(planned)*) covers per-job ordering, at-least-once delivery,
   idempotency, and append-only invariants.
5. E-EV-1 (retention) and E-EV-3 (`schema_version`) are resolved.
6. STATE.md is in sync (E-EV-8 closed).

Architect approves graduation.

### §16.2 Demoting from STABLE

If a STABLE spec needs material change that breaks the STABLE
contract, a Standard ADR may demote it back to EVOLVING per
`docs/constitution/GOVERNANCE.md` §8.

---

## §17 Cross-references

- **Axioms** (`../constitution/AXIOMS.md`): A1 (semantic phase
  integrity — orchestration-agnostic; this spec records the
  current orchestration choice), A4 (audit completeness — the
  per-job event log is the substrate), A5 (deterministic replay —
  the bus must support replay; engines guarantee determinism), A7
  (evidence preservation — the append-only log is part of the
  evidence record).
- **Constitutional governance**
  (`../constitution/GOVERNANCE.md`): §1 (tier hierarchy), §3 (ADR
  tiers), §5 (severity model — P0 for mutation / P1 for spec-
  impl drift), §7 (EGM applies during incidents), §8 (stability
  levels).
- **Domain ownership** (`../governance/DOMAINS.md`): pipeline
  domain owns this spec; api / policy / decision / confidence /
  security / qa / platform domains consume its outputs.
- **Architecture state** (`../state/STATE.md`): the registry is
  partially stale; this spec records the canonical truth (§13.1).
  STATE.md sync is E-EV-8.
- **Implementation**:
  - `backend/app/core/queue.py`
  - `backend/app/core/event_store.py`
  - `backend/app/core/job_store.py`
  - `backend/app/workers/pipeline_worker.py`
  - `backend/app/workers/worker.py`
  - `backend/app/api/ingest.py`
  - `backend/app/api/jobs.py`
- **Consumer specs**:
  - `./decision_engine.md` — produces `RiskScore` consumed by
    target-state `RISK_SCORED` event.
  - `./confidence_engine.md` — produces `ConfidenceBreakdown`
    consumed by target-state `CONFIDENCE_COMPUTED` event.
  - `./policy_engine.md` — produces `PolicyResult` consumed by
    target-state `POLICY_DECIDED` event.
- **Producer specs (downstream)**:
  - `./api_contracts.md` *(planned)* — defines `IngestRequest` /
    `IngestResponse` / `Job` shapes consumed by API readers.
- **Sibling specs**:
  - `./job_processing.md` *(planned, next in queue)* — DLQ,
    retry policy, state-machine details, engine-triple wiring.
- **Future canonical references**:
  - `./storage.md` *(planned)* — durable persistence layer for
    events and jobs (Postgres mirror, cold storage).
  - `./observability.md` *(planned)* — metrics, traces.
  - `../security/enforcement_audit.md` *(planned)* — A4 record
    projection from the event log; reversal event semantics.
  - `../testing/INVARIANT_TESTS.md` *(planned)* — eventing
    invariant test catalogue (per-job ordering, idempotency,
    append-only).
- **TRANSITIONAL sources** (Tier 5 — partially superseded):
  - `.claude/rules/eventing.md` — partial supersedure per
    frontmatter.
  - `.claude/rules/job_system.md` — partial; remainder canonical
    in `./job_processing.md` *(planned)*.
  - `.claude/rules/job-processing.md` — partial; remainder
    canonical in `./job_processing.md` *(planned)*.
