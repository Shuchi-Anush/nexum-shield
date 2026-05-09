---
authority: SPEC
domain: pipeline
status: ACTIVE
version: 1.0
stability: EVOLVING
owner: pipeline (interim: architect)
supersedes:
  - .claude/rules/job-processing.md (TRANSITIONAL — full canonical successor)
  - .claude/rules/job_system.md (TRANSITIONAL — full canonical successor)
adr_references:
  - ADR-0001 (Phase-2 bootstrap; canonical-spec ratification; to be backfilled)
---

# Job Processing — Canonical Specification

The Job Processing layer is the **execution orchestrator** of the
Nexum Shield pipeline. It owns the per-job lifecycle, state-machine
transitions, retry policy, dead-letter quarantine, distributed
locking, replay semantics, and stage-by-stage engine invocation.

This document is the canonical specification — Tier 2 (SPEC) — and
is the **full successor** to `.claude/rules/job-processing.md` and
`.claude/rules/job_system.md`. It composes with `./eventing.md`
(the runtime substrate) without duplicating its contents:

```
┌─────────────────────────────────────────────────────────────────┐
│   Job Processing  (this spec)                                   │
│     • Job model, state machine, retry, DLQ, locking, replay    │
│     • Engine-triple invocation order                            │
│     • Worker boundary and execution contract                    │
├─────────────────────────────────────────────────────────────────┤
│   Eventing        (./eventing.md — sibling spec)                │
│     • Event envelope, taxonomy, delivery, ordering, retention   │
│     • Append-only audit substrate                               │
└─────────────────────────────────────────────────────────────────┘
```

The implementation surface is:

- `backend/app/core/job_store.py` — Redis-backed JobStore + state
  machine.
- `backend/app/core/queue.py` — RQ pipeline queue.
- `backend/app/workers/pipeline_worker.py` — sequential orchestrator.
- `backend/app/workers/worker.py` — RQ worker entrypoint.
- `backend/app/api/ingest.py` — job-creation entry point.
- `backend/app/api/jobs.py` — job-state read endpoints.

Owned by the **pipeline** domain
(`docs/governance/DOMAINS.md`). Consumed by api, decision,
confidence, policy, security, qa, and platform.

---

## §1 System Role

### §1.1 Purpose

The Job Processing layer answers five questions:

1. **What is the unit of pipeline execution?** — the *job*,
   identified by `job_id`, the canonical correlation key.
2. **How does a job progress through the pipeline?** — state
   transitions, stage handoff, terminal semantics.
3. **What happens when a stage fails?** — retry policy, dead-
   letter quarantine, operator re-drive paths.
4. **How are duplicates and crashes handled?** — distributed
   locking, idempotency, redelivery interaction.
5. **How is a past job re-executed?** — replay against immutable
   evidence with deterministic engines.

The Job Processing layer is **NOT**:

- The event bus (delivery / ordering / retention live in
  `./eventing.md`).
- The audit-record projection layer (lives in
  `docs/security/enforcement_audit.md` *(planned)*).
- A general workflow engine (no Temporal/Cadence semantics — see
  §13.5).
- An engine — it *invokes* engines but never *implements* business
  logic (per `./decision_engine.md` §8.4 +
  `./policy_engine.md` §12).

### §1.2 Position vs A1 phases

Per A1 (PIPELINE PHASE INTEGRITY), the system MUST execute six
semantic phases. Job Processing materialises **the orchestration of
all six phases as a single coordinated execution unit identified
by `job_id`**:

```
INGESTION → ANALYSIS → EVALUATION → DECISION → ENFORCEMENT
   │           │           │           │           │
   └─────── single job_id propagates throughout ───────┘
                              │
                              ▼
                       AUDITABILITY
                       (event log accumulates)
```

The job is the coarsest unit of orchestration. Sub-stages within a
job are NOT separately addressable from the API; only the job
itself is. This is intentional: the audit / dispute / replay
machinery operates at the job granularity (per A4 + A6 + A7).

### §1.3 Authority boundaries

| Boundary | Owner | Responsibility |
|---|---|---|
| Request shape | api domain | `IngestRequest`, `IngestResponse`, `Job`-as-API |
| Event delivery + ordering + retention | pipeline (eventing) | `./eventing.md` |
| **Job lifecycle** | **pipeline (this spec)** | **state machine, retry, DLQ, locking** |
| Engine determinism | decision / confidence / policy | per-engine spec |
| Audit-record projection | security | `enforcement_audit.md` *(planned)* |
| Storage durability | platform | `storage.md` *(planned)* |

A worker's responsibility is **strictly orchestration**: validate
preconditions, acquire lock, transition state, invoke engines via
the worker's input-construction logic, emit events, transition to
terminal. **Business logic stays in engines**.

### §1.4 Authority

This document is **TIER 2 (SPEC)** per
`docs/constitution/GOVERNANCE.md` §1. Owned by the **pipeline**
domain. Modification:

| Change | ADR tier |
|---|---|
| Adjusting RQ `job_timeout`, lock TTL, or backoff defaults | Lightweight |
| Adding operational metadata field to `Job` (additive) | Lightweight |
| Adding a non-state field to the state machine record (e.g., `retry_count`) | Lightweight |
| Adding a new state to the state machine | Standard |
| Removing or renaming an existing state | Standard |
| Changing legal transitions in `_VALID_TRANSITIONS` | Standard |
| Engine-triple wiring (replacing legacy scoring/enforcement engines) | Standard (cross-domain — this spec + decision + confidence + policy + eventing co-evolve) |
| Introducing DLQ topology | Standard |
| Introducing retry policy with backoff | Standard |
| Switching queue technology (RQ → Celery / SQS / Kafka) | Standard |
| Introducing cancellation primitive | Standard |
| Removing the at-least-once delivery floor (cross-spec — would also touch `./eventing.md`) | Constitutional |
| Removing append-only audit (cross-spec — would also touch `./eventing.md`) | Constitutional |
| Reordering A1 phases | Constitutional |
| Removing terminal-state absorption | Constitutional |

### §1.5 Stability

**EVOLVING**. The current state machine (5 states) is in
production; the target state machine (8 states — adds CANCELLED,
RETRY_PENDING, DEAD_LETTERED) is canonicalised here but not yet
implemented. Compatibility expectations are low — consumers should
expect change at each minor version bump until graduation gates
(§18.1) are met.

The `JobStatus` enum is a **cross-domain consumer surface**:
`api/jobs.py` returns `status` strings to HTTP clients; qa replay
reads them. Evolution is governed by §13.4 + §18.

---

## §2 Job Model

### §2.1 Identity

A job is identified by `job_id`, a UUID4 string generated at the
API boundary (`backend/app/api/ingest.py`):

```python
job_id = str(uuid.uuid4())
```

| Property | Value |
|---|---|
| Generation point | API endpoint, **before** `create_job` |
| Format | UUID4 (RFC 4122 §4.4) lowercase hex with dashes |
| Uniqueness | global, per-process |
| Determinism | non-deterministic (random); identical request bodies yield distinct `job_id`s |
| Immutability | once generated, NEVER reassigned |

The choice of UUID4 satisfies the
`.claude/rules/job-processing.md` requirement that `job_id` be
"deterministic or UUID" — UUID4 is the chosen branch. This spec
does NOT canonicalise content-deterministic IDs (e.g., SHA-256 of
metadata); a future Standard ADR may introduce them as a parallel
identity scheme bundled with replay-as-first-class-op.

### §2.2 Job record

The canonical `Job` record (`backend/app/core/job_store.py::Job`):

```python
@dataclass
class Job:
    job_id:         str
    status:         JobStatus
    created_at:     float                       # seconds since epoch
    updated_at:     float
    metadata:       Optional[dict] = None       # original ingest payload
    stages:         Dict[str, Any] = {}         # per-stage outputs
    result:         Optional[dict] = None       # consolidated terminal output
    failure_reason: Optional[str] = None
```

> **Naming reconciliation.** A second `Job` model exists at
> `backend/app/models/job.py` (`Job(id: int, status: str, result:
> Optional[str])`). This is an **unused stub** carried over from
> early scaffolding. The canonical `Job` is the dataclass in
> `backend/app/core/job_store.py`. The stub is tracked for
> deletion (Gap **J-JP-7**).

### §2.3 Immutable vs mutable fields

Every `Job` field has explicit mutability rules.

| Field | Mutability | Authority |
|---|---|---|
| `job_id` | **immutable** after `create_job` | none (never written again) |
| `created_at` | **immutable** after `create_job` | none (never written again) |
| `metadata` | **immutable** after `create_job` (the original ingest payload is preserved verbatim) | none (never written again) |
| `status` | **mutable** through `_VALID_TRANSITIONS` (§3) | worker (`update_status`, `set_failure`) |
| `updated_at` | mutable, written on every state-touching call | worker / API (cascading from other writes) |
| `stages[<name>]` | append-only by stage; same stage name overwrites if a future retry replays the stage | worker (`update_stage`) |
| `result` | mutable once (set at terminal-success transition) | worker (`set_result`) |
| `failure_reason` | mutable once (set at terminal-failure transition) | worker (`set_failure`) |

Mutating `job_id`, `created_at`, or `metadata` after creation is a
**P0 violation** per `docs/constitution/GOVERNANCE.md` §5: it
corrupts replay attribution under A5 and evidence under A7.

### §2.4 Stage metadata

Each completed stage writes its output to `job.stages[<stage>]`.
Today's stage names (per `pipeline_worker.run_pipeline`):

```
fingerprint    → {"hash": <content_hash>}
embedding      → {"vector": [...], "model_version": ...}
matching       → {"matched_asset": {...} | null, "similarity": float}
scoring        → {"band": "LOW"|"MEDIUM"|"HIGH"}        ← legacy (J-JP-4)
enforcement    → {"action": ..., "reason": ..., ...}    ← legacy (J-JP-4)
```

Target-state stage names after engine-triple wiring (§5.4):

```
fingerprint, embedding, matching                        (unchanged)
evaluation     → {"risk": {composite, band, breakdown,
                            decision_config_version},
                  "confidence": {composite, tier,
                                 triggered_conditions,
                                 confidence_config_version}}
decision       → PolicyResult.dict()                    (5-action, full audit)
enforcement    → {"action": <PolicyAction>, "evidence_ref": ...,
                  "applied_at": float, ...}
```

The `stages` dict is the **per-job working memory**. It is NOT the
audit record (the event log is). Together with the event log,
`stages` is sufficient to reconstruct engine inputs for replay
(§9).

### §2.5 Replay metadata

Per A5 (DETERMINISTIC REPLAY), every job MUST carry sufficient
metadata to recompute its decision deterministically. The
replay-attributable surface:

| Source | Field | A5 lineage |
|---|---|---|
| `Job.metadata` | original ingest payload | the fixed input |
| `Job.stages.<engine>` | per-stage outputs | the worker's intermediate computations |
| event log | per-event payloads | redundant projection of stages — useful for streaming consumers |
| (target) `Job.metadata.decision_config_version` | threshold config snapshot | per `./decision_engine.md` §4.3 |
| (target) `Job.metadata.confidence_config_version` | confidence config snapshot | per `./confidence_engine.md` §4.4 |
| (target) `Job.metadata.policy_version` | policy engine version | per `./policy_engine.md` §16 |

Today the version triad lives **inside engine output payloads**
that the worker captures into `stages`. Promoting it to top-level
`Job.metadata` is a Lightweight ADR (Gap **J-JP-8**).

### §2.6 Execution attempts

Today the `Job` record has **no `retry_count` field**. Every
delivery is treated as a fresh attempt; the lock + state guard
(§8) ensures only one delivery actually executes the pipeline body.

Target state introduces:

| Field | Type | Semantic |
|---|---|---|
| `retry_count` | `int`, default `0` | number of times this job has transitioned QUEUED → PROCESSING after a prior FAILED |
| `max_retries` | `int`, default per-config | upper bound; on exhaustion, transition is RETRY_PENDING → DEAD_LETTERED instead of RETRY_PENDING → QUEUED |
| `last_attempt_id` | `str`, UUID4 | per-attempt identifier — the random lock token from the most recent acquisition (§8.2) |
| `next_attempt_at` | `float`, optional | absolute wall-clock timestamp at which the job becomes eligible for redelivery (backoff scheduler) |
| `failure_class` | `str`, optional | `"transient" | "permanent" | "stage_timeout" | "lock_loss" | "broker_loss"` (§6.1) |

These are **target-state additive fields** (Lightweight ADR,
bundled with the retry-policy ADR in §6).

### §2.7 Terminal semantics

A terminal state is **absorbing** — no transition out. Reaching a
terminal state means:

- The pipeline body has completed (success, failure, or
  quarantine).
- The job's audit timeline is closed (the corresponding `JOB_COMPLETED`
  / `JOB_FAILED` / future `JOB_CANCELLED` / `JOB_DEAD_LETTERED`
  event has been emitted).
- The `Job.result` and / or `Job.failure_reason` fields carry the
  terminal payload.
- Subsequent redeliveries hit the state guard and exit
  (`./eventing.md` §5.4.1).

Reversal of a terminal action under A6 (HUMAN REVIEW AUTHORITY)
is recorded as an **append to the audit log**, not as a
state-machine transition. The job remains in its terminal state
forever. This is the runtime materialisation of A4's append-only
guarantee (`./eventing.md` §10.1).

---

## §3 State Machine

### §3.1 Current implemented state machine

`backend/app/core/job_store.py::_VALID_TRANSITIONS`:

```
                ┌──────────────────────┐
                │       QUEUED         │
                └─────────┬────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   ┌──────────────────┐    ┌──────────────────┐
   │   PROCESSING     │    │     FAILED       │  (terminal, absorbing)
   └─────────┬────────┘    └──────────────────┘
             │
   ┌─────────┼──────────┐
   ▼         ▼          ▼
COMPLETED  FAILED   FLAGGED   (all terminal, absorbing)
```

Implemented state set: `{QUEUED, PROCESSING, COMPLETED, FAILED,
FLAGGED}` (5 states, 3 absorbing).

| Source | Target | Authority | Effect |
|---|---|---|---|
| QUEUED | PROCESSING | worker (lock acquired + state guard passed) | begins pipeline body |
| QUEUED | FAILED | worker (`set_failure` from QUEUED — defensive) | rare; only if worker fails before flipping to PROCESSING |
| PROCESSING | COMPLETED | worker (action ∈ {ALLOW} per legacy) | clean terminal |
| PROCESSING | FLAGGED | worker (action ∈ {FLAG, BLOCK} per legacy) | terminal-with-attention |
| PROCESSING | FAILED | worker (any exception in pipeline body) | failure terminal |

### §3.2 Target state machine

The canonical target-state machine adds CANCELLED, RETRY_PENDING,
and DEAD_LETTERED to support cancellation, retry-with-backoff, and
poison-job quarantine. **FLAGGED is preserved** (despite its
omission from the user-facing state list in this spec's commission
prompt) — it carries semantic value as "completed-with-human-
follow-up-required" that COMPLETED-with-action-metadata does not
carry equivalently for operator dashboards. Folding FLAGGED into
COMPLETED is a future Standard ADR; this spec does NOT adopt that
fold.

```
              ┌─────────────────────────────┐
              │           QUEUED            │◀──┐ (from RETRY_PENDING)
              └────────────┬────────────────┘   │
                           │                    │
              ┌────────────┼────────────────┐   │
              ▼            ▼                ▼   │
        ┌──────────┐  ┌─────────┐  ┌────────────────┐
        │CANCELLED │  │ FAILED  │  │   PROCESSING   │
        └──────────┘  └─────────┘  └────────┬───────┘
        (terminal)    (terminal)            │
                                            │
                  ┌─────────────────┬───────┴─────┐
                  ▼                 ▼             ▼
            ┌──────────┐    ┌──────────┐  ┌──────────────┐
            │COMPLETED │    │ FLAGGED  │  │ RETRY_PENDING│
            └──────────┘    └──────────┘  └──────┬───────┘
            (terminal)      (terminal)           │
                                  ┌──────────────┴────┐
                                  ▼                   ▼
                              QUEUED         ┌──────────────┐
                              (re-attempt)   │DEAD_LETTERED │
                                             └──────────────┘
                                              (terminal)

(plus PROCESSING → CANCELLED if cancellation is requested mid-job;
 plus PROCESSING → FAILED on unrecoverable exception)
```

### §3.3 Canonical state semantics

| State | Class | Semantic |
|---|---|---|
| `QUEUED` | non-terminal | enqueued for execution; awaiting worker pickup |
| `PROCESSING` | non-terminal | a worker holds the lock and is executing the pipeline body |
| `COMPLETED` | terminal | pipeline finished; final action is one that does NOT require human follow-up (today: `ALLOW`; target: `ALLOW`, `RESTRICT`, `TAKEDOWN` — automated enforcement applied) |
| `FLAGGED` | terminal | pipeline finished; final action requires human follow-up (today: `FLAG`, `BLOCK`; target: `FLAG`, `REVIEW` per `./policy_engine.md` §2.1 reversibility table) |
| `FAILED` | terminal | pipeline raised an unrecoverable exception OR the failure class was `permanent` (no retry eligible) |
| `CANCELLED` | terminal (target-state) | operator-initiated cancellation accepted; pipeline body did not complete |
| `RETRY_PENDING` | non-terminal (target-state) | transient failure observed; awaiting backoff window before re-enqueue |
| `DEAD_LETTERED` | terminal (target-state) | retry budget exhausted; quarantined for operator review |

### §3.4 Legal transitions

The **target-state** `_VALID_TRANSITIONS` table:

| Source | Allowed targets |
|---|---|
| `QUEUED` | `PROCESSING`, `FAILED`, `CANCELLED` |
| `PROCESSING` | `COMPLETED`, `FLAGGED`, `FAILED`, `CANCELLED`, `RETRY_PENDING` |
| `RETRY_PENDING` | `QUEUED`, `DEAD_LETTERED`, `CANCELLED` |
| `COMPLETED` | ∅ (absorbing) |
| `FLAGGED` | ∅ (absorbing) |
| `FAILED` | ∅ (absorbing) |
| `CANCELLED` | ∅ (absorbing) |
| `DEAD_LETTERED` | ∅ (absorbing) |

The **current** code's `_VALID_TRANSITIONS` is a strict subset of
the target state's table. Migration is additive:

1. Add new states to `JobStatus`.
2. Extend `_VALID_TRANSITIONS` with new transitions.
3. Existing transitions remain valid.
4. Existing code (worker, API) compiles unchanged.

This is **Lightweight + Standard ADR bundled**: adding states is
Standard; extending the transition table is Lightweight per §1.4.

### §3.5 Forbidden transitions

Forbidden transitions (and the corresponding code-level guards):

| Forbidden | Why | Enforcement |
|---|---|---|
| Skipping states (e.g., `QUEUED → COMPLETED` without passing through `PROCESSING`) | violates `.claude/rules/job-processing.md` §2 ("no skipping states") and breaks audit timeline | `_VALID_TRANSITIONS` excludes the pair |
| Out of any terminal state | absorbing terminals are constitutional | `_VALID_TRANSITIONS[terminal] == ∅` raises on attempt |
| Target-state-only transitions (e.g., to/from `CANCELLED`, `RETRY_PENDING`, `DEAD_LETTERED`) before those states are implemented | drift between spec and impl | code guards via `JobStatus(...)` enum membership; missing values raise |
| Mutating `job_id`, `created_at`, or `metadata` post-creation | corrupts replay attribution + evidence | not enforced by `JobStore` API today (J-JP-7 family); P0 violation if attempted |

A transition not in `_VALID_TRANSITIONS` raises
`ValueError("Invalid status transition: <from> -> <to>")`. This is
checked under WATCH/MULTI/EXEC for cross-process atomicity.

### §3.6 Transition authority

| Transition class | Authority | Mechanism |
|---|---|---|
| `QUEUED → PROCESSING` | worker (sole) | `update_status(job_id, PROCESSING)` after lock + state guard |
| `PROCESSING → COMPLETED \| FLAGGED` | worker (sole) | post-pipeline-body, terminal selection per action |
| `PROCESSING → FAILED` | worker (exception path) | `set_failure(job_id, reason)` |
| `QUEUED → FAILED` | worker (defensive) | `set_failure(job_id, reason)` from QUEUED — rare |
| `PROCESSING → RETRY_PENDING` (target) | worker (transient classification) | new `update_status(job_id, RETRY_PENDING)` call |
| `RETRY_PENDING → QUEUED` (target) | retry scheduler (separate process / RQ deferred job) | re-enqueue via `pipeline_queue.enqueue_in(...)` |
| `RETRY_PENDING → DEAD_LETTERED` (target) | retry scheduler (budget exhausted) | terminal transition + DLQ event |
| `* → CANCELLED` (target) | operator API endpoint | `cancel_job(job_id)` — see §7.5 / §13.4 |
| Reversal of terminal action under A6 | security layer | append entry, NOT a transition (per §2.7) |

The worker is the **sole** authority for transitions out of
QUEUED and PROCESSING in target state. Operator-initiated
transitions (cancellation) require a separate authorisation path
(API endpoint with auth — out of scope for this spec; defers to
the API + security specs).

### §3.7 Atomicity guarantees

Every state transition is **atomic** at the Redis level:

```
WATCH job:{job_id}
HGET   job:{job_id} status              ← read current status
(check transition validity in-memory)
MULTI
HSET   job:{job_id} status <new>
HSET   job:{job_id} updated_at <now>
EXPIRE job:{job_id} <ttl>               ← if JOB_TTL_SECONDS is set
EXEC                                    ← atomic, fails on WATCH conflict
```

If two writers attempt the same `update_status` concurrently, the
WATCH conflict on the second writer causes its `EXEC` to fail; the
write is retried (the `while True` loop in `job_store.py`). The
retry re-reads the current status, which is now post-first-writer,
and either accepts the new transition or raises `ValueError` if
the new transition is not legal from the new source state.

This is **strong consistency** for state transitions — required by
`docs/constitution/GOVERNANCE.md` and `.claude/rules/storage.md` §
"Strong consistency REQUIRED for: job state transitions".

### §3.8 Replay does not transition state

Replay (§9) is a **read-only** operation. It does NOT mutate the
original job's state. A replay attempt against a COMPLETED job
re-runs the engines against the immutable evidence and compares
the recomputed output to the stored output; the COMPLETED status
is preserved. The replay tool MAY emit replay-specific events into
a separate log (target-state — see `./eventing.md` §3.8).

### §3.9 Compensating behavior (no state-level rollback)

There is **no rollback transition** in the state machine. The
state machine is forward-only:

- Cannot transition out of a terminal state.
- Cannot un-fail a FAILED job (the operator re-drives via a new
  job, not by mutating the failed one).
- Cannot un-flag a FLAGGED job (the human-review verdict is
  recorded as an *appended audit entry* per A6 + §2.7, not as a
  state-machine transition).
- Cannot un-cancel a CANCELLED job (same — re-drive as a new
  job).

This is a deliberate constraint inherited from A4's append-only
audit guarantee. State rollback would imply audit mutation, which
is forbidden.

---

## §4 Pipeline Execution Model

### §4.1 Sequential execution

Today's `pipeline_worker.run_pipeline` executes stages **strictly
sequentially in the same Python process**:

```
fingerprint → embedding → matching → scoring → enforcement
```

Each stage is wrapped in a `stage_event(job_id, stage_name)`
context manager (`./eventing.md` §3.3) that emits lifecycle
STARTED + COMPLETED|FAILED events with wall-clock latency. Within
each stage, the worker:

1. Calls a pure-function engine (`fingerprint_engine.compute_fingerprint`,
   `embedding_engine.embed`, etc.).
2. Writes the output to `job.stages[<stage>]` via
   `update_stage(job_id, <stage>, output)`.
3. Emits a typed `PipelineEventType` domain event via
   `publish_event(...)`.

The next stage reads its inputs from the prior stage's local
return value (in-memory variable handoff — `./eventing.md` §4.3).
There is no event-driven dispatch; stage progression is determined
by Python control flow.

### §4.2 Deterministic execution ordering

The stage order is **fixed** in code; it cannot be reordered
per-request. This satisfies A1 (PIPELINE PHASE INTEGRITY) by
construction: the worker enforces the canonical phase ordering as
a hard property of its source.

The current code-fixed order maps to A1 phases as:

| Stage | A1 Phase |
|---|---|
| `fingerprint`, `embedding`, `matching` | ANALYSIS (sub-stages permitted per A1) |
| `scoring` | EVALUATION (legacy; collapses risk + confidence — J-JP-4) |
| `enforcement` | DECISION + ENFORCEMENT (legacy; collapses both — J-JP-4) |

Target-state ordering after engine-triple wiring (§5.4):

| Stage | A1 Phase |
|---|---|
| `fingerprint`, `embedding`, `matching` | ANALYSIS (unchanged) |
| `evaluation` | EVALUATION (parallel risk + confidence) |
| `decision` | DECISION (PolicyEngine over DecisionOutput + ConfidenceBreakdown + PolicyContext) |
| `enforcement` | ENFORCEMENT (apply 5-action `PolicyAction`) |

### §4.3 Engine invocation boundaries

The worker is the **sole invoker** of engines. This is enforced by:

- Engines are imported only by the worker (and by tests).
- API endpoints (`api/ingest.py`, `api/jobs.py`) MUST NOT import
  engines.
- One engine MUST NOT import another engine. Cross-engine
  references go through models (e.g., `policy_engine` imports
  `RiskBand` from `decision_models`, not from `decision_engine`).

Engines have **zero I/O** per their specs (`./decision_engine.md`
§8.1, `./confidence_engine.md` §12.1, `./policy_engine.md` §12.1).
The worker performs all I/O — Redis writes, event emissions, lock
operations — *around* the engine call.

```
worker:
    inputs = build_inputs_from(stage_outputs_so_far, job.metadata)
    output = engine.compute(inputs, config)         ← pure function call
    job_store.update_stage(job_id, stage_name, output)
    publish_event(...)                              ← typed bus emission
```

### §4.4 Stage contracts

Every stage MUST satisfy:

| Contract | Statement |
|---|---|
| **Pure invocation** | the engine call is a pure function of its typed input |
| **Idempotency** | repeating the stage with the same input produces the same output (per A5 + engine specs) |
| **Output capture** | the engine's full return value is written to `job.stages[<stage>]` |
| **Event emission** | one typed `PipelineEventType` domain event is published per stage (modulo split events like `MATCH_FOUND` / `MATCH_NOT_FOUND`) |
| **Lifecycle wrapping** | the stage body executes inside a `stage_event(job_id, name)` block |
| **No cross-stage state** | a stage MUST NOT depend on global state, prior-job state, or external services beyond its declared inputs |

Stage contracts are enforced by **code review** today; static
enforcement (e.g., a stage decorator that builds the wrapping +
event emission) is a future Lightweight ADR (Gap **J-JP-9**).

### §4.5 Handoff guarantees

The handoff from stage N to stage N+1 today carries the engine's
return value as a **Python local variable**. This is the strongest
possible handoff (no serialisation, no broker, no network) — but
also limits scaling to "one process per job".

When the engine triple is wired (§5.4), the handoff between
EVALUATION and DECISION carries the `DecisionOutput` envelope (per
`./decision_engine.md` §5.3) as a local variable. The DECISION
phase consumes both the envelope and the `ConfidenceBreakdown`.

**Future stage-level fan-out** (§12.4) requires breaking the
in-memory handoff into a serialised handoff via the bus. This is
deferred to a future Standard ADR.

### §4.6 Partial-stage semantics

There is **no partial-stage execution** today. A stage either:

- Completes fully (engine returns; output captured; event emitted),
  or
- Fails (engine raises; lifecycle FAILED emitted; pipeline body's
  exception handler in the worker takes over).

There is no "60% of the embedding completed; checkpoint and
resume" mode. This is acceptable for MVP because the engines are
sub-second pure functions; a partial-completion model is only
needed when stages take long enough to checkpoint.

A stage that completes successfully but partially populates its
output (e.g., embedding returns a degraded vector) is a **stage
contract violation**, not a state-machine concept. The engine
MUST return its complete typed output or raise; partial returns
are forbidden.

### §4.7 Stage isolation

Within a job, stages share `job.stages` as their working memory.
Across jobs, **stages are fully isolated** — no shared mutable
state, no cross-job locks, no cross-job signal passing.

This isolation is enforced by:

- The Redis key namespace (`job:{job_id}` is per-job).
- The lock namespace (`lock:job:{job_id}` is per-job).
- The event log namespace (`events_by_job:{job_id}` is per-job).

Cross-job signal-passing on the bus is **forbidden** (`./eventing.md`
§7.6). Operations that span jobs (e.g., the propagation graph
updating after multiple jobs detect related content) operate on
**`content_id`** (a derived identifier), not on `job_id` (per
`./eventing.md` §7.6).

---

## §5 Engine-Triple Integration

This is the **highest-leverage runtime closure** in the repository
(Gap **J-JP-4** = `./eventing.md` E-EV-6 = `./decision_engine.md`
D-DE-2).

### §5.1 Current implementation

The current `pipeline_worker.run_pipeline` uses **legacy engines**:

```python
band = scoring_engine.score(match.similarity)             # legacy
decision = enforcement_engine.decide(                     # legacy
    input_media_id=content_hash,
    matched_asset=matched_asset_dict,
    similarity=match.similarity,
    band=band,
    model_version=embedding_engine.MODEL_VERSION,
)
```

These produce:

- `band` ∈ `{LOW, MEDIUM, HIGH}` from a similarity-only band lookup
  (no risk model, no confidence dimension).
- `decision.action` ∈ `{ALLOW, FLAG, BLOCK}` from a trust-aware
  threshold function (no PBRA, no policy engine, no audit hash, no
  evidence strength).

The new engine triple — `decision_engine.compute_risk`,
`confidence_engine.compute_confidence`,
`policy_engine.evaluate_policy` — exists as canonical Tier-2
engine-spec library code and is **not invoked**.

### §5.2 Target invocation order

After engine-triple wiring, `run_pipeline` invokes the engines in
this order:

```
                ┌──────────────────────────────────────┐
                │   stage: matching (unchanged)        │
                │   produces: MatchResult              │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │   stage: evaluation                  │
                │   ─────────────────────────────────  │
                │   risk = decision_engine.compute_risk│
                │              (decision_input,        │
                │               threshold_config)      │
                │                                      │
                │   confidence =                       │
                │       confidence_engine.compute_     │
                │           confidence(conf_input,     │
                │                     conf_config)     │
                │                                      │
                │   decision_output =                  │
                │       build_decision_output(         │
                │           risk, input_snapshot,      │
                │           decision_config_version)   │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │   emits:                             │
                │     RISK_SCORED                      │
                │     CONFIDENCE_COMPUTED              │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │   stage: decision                    │
                │   ─────────────────────────────────  │
                │   policy_context =                   │
                │       build_policy_context(...)      │
                │                                      │
                │   result = policy_engine.            │
                │       evaluate_policy(               │
                │           decision_output,           │
                │           confidence,                │
                │           policy_context)            │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │   emits: POLICY_DECIDED              │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │   stage: enforcement                 │
                │   apply 5-action PolicyAction        │
                │   emits: ENFORCED (extended payload) │
                └──────────────────────────────────────┘
```

DecisionEngine and ConfidenceEngine outputs are **independent** —
neither reads the other (per `./decision_engine.md` §6.1 +
`./confidence_engine.md` §1.1). The pipeline worker MAY parallelise
their computation. Today the implementation will be sequential
within the single `evaluation` stage; parallelisation is a future
optimisation.

### §5.3 New event emissions

Three new `PipelineEventType` values are introduced (jointly
authored with `./eventing.md` §13.4):

| Event | Payload | Notes |
|---|---|---|
| `RISK_SCORED` | `{composite, band, breakdown_summary, decision_config_version}` | from DecisionEngine output; band is `RiskBand.value` |
| `CONFIDENCE_COMPUTED` | `{composite, tier, triggered_conditions, confidence_config_version}` | from ConfidenceEngine output; tier is `ConfidenceTier.value` |
| `POLICY_DECIDED` | `{action, triggered_rules, evaluation_hash, policy_version, decision_config_version, confidence_config_version, primary_reason}` | from PolicyEngine `PolicyResult`; action is `PolicyAction.value` (5-level) |

The legacy `SCORED` event is **retired** in favour of
`RISK_SCORED` + `CONFIDENCE_COMPUTED`. The legacy `ENFORCED`
payload schema is **extended** to carry the 5-action `PolicyAction`
(today's payload carries `action: str` from `{ALLOW, FLAG, BLOCK}`).

A backward-compatibility transition window — emitting both legacy
and new events during a configurable rollout — is the
recommended migration pattern (Standard ADR §5.5).

### §5.4 State propagation

The data flow through the engine triple:

```
DecisionInput  → compute_risk    → RiskScore
                                      │
                                      ▼
                                 DecisionOutput envelope
                                      │
                                      │  (paired with)
                                      ▼
ConfidenceInput → compute_confidence → ConfidenceBreakdown
                                      │
                                      ▼
                                 PolicyEngine.evaluate_policy
                                      │
                                      ▼
                                 PolicyResult
                                      │
                                      ▼
                                 enforcement_engine.apply(...)
                                      │
                                      ▼
                                 ENFORCED + JOB_COMPLETED|FLAGGED
```

The `DecisionOutput` envelope is owned by the **decision domain**
(per `./decision_engine.md` §5.3) and materialised by the pipeline
worker. This spec does NOT redefine it; it consumes the contract
defined in `decision_engine.md`.

The pipeline worker constructs:

- `DecisionInput` from `match`, `trust_owner`, `trust_uploader`,
  `signal_source`, `observation_count`, `observation_timestamps`,
  `decision_config_version`.
- `ConfidenceInput` from the same upstream signals plus the
  confidence-domain primitives.
- `PolicyContext` from operational + evidence signals (per
  `./policy_engine.md` §4.3).
- `DecisionOutput` envelope from `RiskScore` + an
  `input_snapshot` carrying the **confidence** config version
  (per `./policy_engine.md` §4.1 / `./confidence_engine.md` §10.1
  / C-CE-9).

### §5.5 Migration plan

Migration from legacy to engine triple is a **Standard ADR**
(cross-domain — pipeline + decision + confidence + policy +
eventing all co-evolve). The recommended phases:

**Phase A — Materialisation (one PR, Lightweight ADR)**:

1. Add concrete `DecisionOutput` dataclass to
   `backend/app/models/decision_models.py` (closes
   `./decision_engine.md` D-DE-1).
2. Add `RiskScore.config_version` field for self-contained
   provenance.
3. Replace the `Protocol` import in `policy_engine.py` with the
   concrete class.

**Phase B — Worker rewire (one PR, Standard ADR)**:

4. Add `RISK_SCORED`, `CONFIDENCE_COMPUTED`, `POLICY_DECIDED`
   to `PipelineEventType`.
5. Add corresponding payload classes to `event_store.py`.
6. Rewrite `run_pipeline` to invoke the engine triple in the
   stage order of §5.2.
7. Extend `ENFORCED` payload to carry the 5-action `PolicyAction`.
8. Maintain backward-compatible emission of legacy `SCORED` event
   for a configurable transition window (gated by
   `EMIT_LEGACY_SCORED_EVENTS = True/False`).
9. Update `_VALID_TRANSITIONS` to fold `REVIEW` into FLAGGED and
   `RESTRICT`/`TAKEDOWN` into COMPLETED (§3.3).

**Phase C — Legacy deprecation (one PR + STATE.md update)**:

10. Annotate `backend/app/engines/scoring_engine.py` and
    `backend/app/engines/enforcement_engine.py` with
    `@deprecated` notices.
11. Mark them DEPRECATED in `docs/state/STATE.md`.
12. Plan removal in a future minor version (after the transition
    window closes).

This is exactly the migration path summarised in
`./decision_engine.md` §14.2 / `./eventing.md` §13.4.

---

## §6 Retry Model

### §6.1 Failure taxonomy

Every pipeline-body exception is classified into one of five
classes. Today's worker classifies **none** explicitly — every
exception is treated as terminal-FAILED. The taxonomy is the
**target-state** classification:

| Class | Examples | Retry eligible? |
|---|---|---|
| **transient** | Redis connection blip, DNS resolution, RQ broker hiccup, transient `requests.exceptions.ConnectionError` from external services | yes — backoff + retry |
| **stage_timeout** | a stage exceeded its per-stage budget (target-state per-stage timeout) | yes if root cause is contention; no if root cause is logic loop |
| **lock_loss** | this worker's lock TTL expired mid-execution AND a successor lost the race (rare; today the state guard handles it without classification) | no — already a no-op via state guard |
| **broker_loss** | Redis fully unavailable mid-execution; in-flight emits may be lost | yes — re-queue when broker returns |
| **permanent** | malformed metadata, deterministic engine bug, payload-validation `ValueError`, schema mismatch | no — direct to FAILED or DEAD_LETTERED |

The classification is performed by an **explicit classifier
function** (target-state — Gap **J-JP-2**), invoked from the
exception path:

```python
except Exception as exc:
    failure_class = classify(exc)            # target-state
    if failure_class in RETRYABLE_CLASSES and retry_count < max_retries:
        job_store.transition_to_retry(job_id, failure_class, ...)
        publish_event(JOB_RETRY_PENDING, ...)  # target-state event
    else:
        job_store.set_failure(job_id, ...)
        publish_event(JOB_FAILED, ...)
```

### §6.2 Retry eligibility

A job is retry-eligible iff:

1. The classifier returns a class in `RETRYABLE_CLASSES = {transient,
   stage_timeout, broker_loss}`.
2. `retry_count < max_retries` (default `max_retries = 3`; configurable).
3. The job is currently in `PROCESSING` state (transitions
   QUEUED → PROCESSING → RETRY_PENDING; not from any other state).
4. The job has not been operator-cancelled (no pending CANCELLED
   transition).

If any condition fails, the job transitions to FAILED (or
DEAD_LETTERED if the cause is retry-budget exhaustion, §7).

### §6.3 Retry counters

Target-state additive `Job` fields (per §2.6):

```
retry_count       int    default 0     incremented on each PROCESSING→RETRY_PENDING
max_retries       int    default 3     per-config; capped via Lightweight ADR
last_attempt_id   str    UUID4         lock-token of most-recent attempt
next_attempt_at   float  optional      epoch-seconds until eligible for re-enqueue
failure_class     str    optional      most-recent classification
```

Bumping `max_retries` default is a Lightweight ADR. Per-content-type
or per-tenant override schemes (e.g., higher retry budget for video
pipelines) are Standard ADR.

### §6.4 Retry windows and backoff

Backoff is **exponential with full jitter**, per
[AWS Architecture Blog's recommended pattern]:

```python
base_delay_seconds  = 2.0
max_delay_seconds   = 300.0       # capped at job_timeout
jitter_factor       = 1.0         # full jitter

delay = min(
    max_delay_seconds,
    random.uniform(0, base_delay_seconds * (2 ** retry_count) * jitter_factor),
)
next_attempt_at = time.time() + delay
```

For default `max_retries = 3` and `base = 2.0`:

| `retry_count` after failure | Window range (seconds) |
|---|---|
| 1 | `[0, 4]` |
| 2 | `[0, 8]` |
| 3 | `[0, 16]` |
| 4 (would-be) | `[0, 32]` — but `max_retries` exhausted → DEAD_LETTERED |

The constants (`base_delay_seconds`, `max_delay_seconds`,
`jitter_factor`) are config-driven (per `Settings` class — added in
the retry-policy ADR).

### §6.5 Retry orchestration ownership

Retry orchestration is **NOT** the worker's responsibility. The
worker's role is to **classify and transition**:

```
worker:
    classify exception
    if retryable:
        transition PROCESSING → RETRY_PENDING
        emit JOB_RETRY_PENDING(retry_count, next_attempt_at)
    else:
        transition PROCESSING → FAILED  (or DEAD_LETTERED)
        emit JOB_FAILED  (or JOB_DEAD_LETTERED)
```

The actual re-enqueue (RETRY_PENDING → QUEUED) is the
**retry scheduler's** responsibility. Two implementation options
(both target-state):

| Option | Mechanism | Trade-off |
|---|---|---|
| **A — RQ deferred jobs** | use `pipeline_queue.enqueue_in(timedelta(seconds=delay), ...)` to schedule the re-enqueue | requires RQ scheduler running (`rq worker --with-scheduler`); native to the chosen broker |
| **B — Separate scheduler process** | a dedicated process scans `RETRY_PENDING` jobs whose `next_attempt_at <= now()` and re-enqueues | independent of RQ scheduler features; simpler topology |

This spec does NOT prescribe one over the other; the retry-policy
ADR will choose. **Option A is recommended** because it keeps the
orchestration inside the existing broker (no new process; no new
failure mode).

### §6.6 Replay interaction with retry

Retry and replay are **distinct concepts**:

| | Retry | Replay |
|---|---|---|
| **Trigger** | transient failure | operator / qa request |
| **Authority** | retry scheduler (automatic) | operator (manual) |
| **State change** | RETRY_PENDING → QUEUED → PROCESSING again | NONE — replay is read-only |
| **Engine inputs** | reconstructed from `Job.metadata` (same as original) | reconstructed from `Job.metadata` + `Job.stages` (same as original) |
| **Emits new events?** | yes — second pipeline-body emission | no — replay events go to a separate log (target-state) |
| **Affects audit?** | yes — appended to original timeline | no |

Retry uses A5 determinism: re-running the engines against the
same inputs yields the same outputs. So retry is "safe" in the
audit sense — the second attempt produces the same decision
unless a transient input (e.g., trust-registry hit/miss) changed.

### §6.7 Anti-patterns

Forbidden retry behaviours (each is a P0 or P1 violation):

- **Retrying permanent failures.** A `ValueError` from
  `ThresholdConfig` weight validation (per
  `./decision_engine.md` §9.2) is a programmer error; retrying
  reproduces the same exception. The classifier MUST flag these
  as `permanent`.
- **Retrying without backoff.** Tight-loop retry hammers the
  broker and never recovers from genuinely transient outages.
- **Retrying past `max_retries`.** Drives up cost; produces no
  semantic value.
- **Retrying in a different worker process from the lock-holder.**
  Concurrent retries are exactly what the lock + state guard
  exists to prevent. Retry MUST go through the queue (re-enqueue),
  never as an in-process loop.
- **Mutating `retry_count` from outside the worker.** Operators
  who want to "reset" retry counts re-drive the job (which creates
  a new `job_id`), not edit the existing record.

---

## §7 DLQ Model

### §7.1 Poison-job semantics

A **poison job** is a `job_id` whose pipeline body fails
repeatedly under retry — typically because the failure class is
permanent or a transient cause has not resolved within the retry
budget. Today the system does NOT classify poison jobs; every
failure becomes terminal FAILED on first occurrence (Gap **J-JP-3**).

Target-state poison-job handling:

```
PROCESSING raises → classify → transient + retry_count < max_retries
                              → RETRY_PENDING
                              → (after backoff) re-enqueue → QUEUED → PROCESSING ...
                              → after max_retries: DEAD_LETTERED
```

A job entering DEAD_LETTERED is **terminal** and **quarantined**
in a separate RQ queue (`pipeline_dlq`). Operator action is
required to release it.

### §7.2 Dead-letter triggers

A job transitions to DEAD_LETTERED iff:

1. `retry_count >= max_retries` (retry budget exhausted), OR
2. The classifier returned a class in `NON_RETRYABLE_BUT_QUARANTINE
   = {permanent}` (depending on policy; some teams prefer FAILED
   for permanent and DEAD_LETTERED for retry-exhausted), OR
3. An operator forces dead-letter via `POST /v1/jobs/{id}/dead-letter`
   (target-state operator endpoint — see §13.4).

Today only condition (3) is meaningful, and it is not implemented.

### §7.3 Quarantine guarantees

A DEAD_LETTERED job:

- **Persists** in Redis (subject to `JOB_TTL_SECONDS` sliding TTL,
  but DLQ entries SHOULD have a longer TTL — Lightweight ADR per
  §1.4).
- Is **enqueued** to a separate RQ queue (`pipeline_dlq`) with a
  marker so operator tooling can list DLQ contents.
- Is **NOT** automatically retried. The DLQ is a quarantine, not a
  retry queue.
- Its **per-job event log** is preserved (per A4 + A7 + this
  spec's append-only guarantee).
- Its `Job.failure_reason` carries the most-recent terminal
  reason; `Job.failure_class` carries the classifier's verdict.

### §7.4 Replay-from-DLQ

Re-driving a DEAD_LETTERED job is **NOT** a state-machine
transition out of DEAD_LETTERED (the state is absorbing). Re-drive
creates a **new job** with a **new `job_id`** and a `parent_job_id`
linkage:

```
DEAD_LETTERED job:  job_id = abc, status = DEAD_LETTERED, ...
                              ↓ operator re-drive
New job:            job_id = xyz, status = QUEUED,
                    metadata = {...original payload...,
                                replay_of: "abc",
                                replay_reason: "<operator note>"}
```

The new job runs end-to-end. The original DEAD_LETTERED job
remains in DEAD_LETTERED forever. The relationship is recorded
via:

- `Job.metadata.replay_of` — back-reference from new job to
  original.
- (target-state) `JOB_REPLAY_REQUESTED` event emitted at the
  re-drive time, carrying both `job_id`s.

### §7.5 Audit requirements

Per A4 + A7, DEAD_LETTERED is an audit terminal. Required event
log entries:

- `JOB_FAILED` per failed attempt (existing event, unchanged).
- (target-state) `JOB_RETRY_PENDING` per retry transition,
  carrying `retry_count` + `next_attempt_at` + `failure_class`.
- (target-state) `JOB_DEAD_LETTERED` per DLQ transition, carrying
  `final_retry_count` + `final_failure_class` + `final_failure_reason`.

The DLQ entry is itself an audit signal: each `pipeline_dlq` queue
entry corresponds to one DEAD_LETTERED job. Operator dashboards
SHOULD list DLQ size as a P1 metric (Gap **J-JP-3** family — ties
to `docs/specs/observability.md` *(planned)*).

### §7.6 DLQ topology

A separate RQ queue named `pipeline_dlq` holds DLQ entries. The
worker process does NOT consume this queue — only operator tooling
does. Recommended operator actions:

| Action | Effect |
|---|---|
| **List** | `GET /v1/jobs/dead-letter` returns paginated DEAD_LETTERED jobs |
| **Inspect** | `GET /v1/jobs/{id}` returns full state + event log of a DEAD_LETTERED job |
| **Re-drive** | `POST /v1/jobs/{id}/replay` creates a new job with `replay_of` linkage |
| **Purge** (rare) | `POST /v1/jobs/{id}/purge` removes the DLQ entry; the job hash + event log are preserved (NEVER deleted) |

Exact endpoint shapes belong to `docs/specs/api_contracts.md`
*(planned)* + `docs/security/secrets_policy.md` *(planned)*. This
spec specifies the **runtime contract**: operator authorisation is
required for any DLQ operation.

---

## §8 Locking + Concurrency

### §8.1 Distributed lock semantics

Every PROCESSING entry MUST hold the per-job lock. The lock
contract:

| Property | Value |
|---|---|
| **Key** | `lock:job:{job_id}` |
| **Acquisition** | `SET lock:job:{job_id} <token> NX EX 300` |
| **TTL** | 300 seconds (matched to RQ `job_timeout`) |
| **Token** | `uuid.uuid4().hex` per acquisition attempt |
| **Release** | Lua compare-and-delete script |
| **Expiry** | automatic via Redis TTL if release is missed |

The lock is the **only** mechanism preventing concurrent
PROCESSING for the same `job_id`.

### §8.2 Lock ownership

| Concept | Owner |
|---|---|
| Lock acquisition / release | the worker holding the lock |
| Lock semantics (TTL, key naming, token comparison) | this spec |
| Lock storage | Redis (the broker) |

A worker that fails to acquire the lock (because another worker
holds it) is a **silent no-op** — it returns from `run_pipeline`
without raising and without logging an error. This is the desired
behavior under at-least-once redelivery.

### §8.3 Lease behavior

The lock is a **lease with hard TTL**, not a heartbeat-extended
lease. Once acquired, the worker has 300 seconds before TTL
expiry; there is **no in-band lease-renewal mechanism today**.

This creates a **lock-loss hazard** for long-running stages
(Gap **J-JP-5**). Two concrete scenarios:

| Scenario | Today's outcome |
|---|---|
| Stage exceeds 300s, lock TTL expires while worker is mid-execution | a redelivery may take a fresh lock, but the state guard (§8.5) catches it as `status != QUEUED` and exits silently |
| Worker process crashes mid-execution | lock auto-expires after 300s; the next redelivery acquires a fresh lock; state guard sees PROCESSING (not QUEUED) and exits — **the job is stranded in PROCESSING forever** unless an operator force-fails it |

The second scenario — stranded PROCESSING jobs — is a real
production hazard. Today's mitigation: rely on `JOB_TTL_SECONDS`
sliding TTL to eventually reclaim the job hash (and lose the
audit trail). Better mitigation: a "stranded job reaper" process
that flips PROCESSING → FAILED for jobs with `updated_at` older
than a threshold (target-state — Gap **J-JP-5** family).

### §8.4 Heartbeat (target-state)

A target-state heartbeat extends the lock TTL during long-running
stages:

```python
def stage_with_heartbeat(job_id, stage_name, work, *, heartbeat_interval=60):
    """Run `work` in this thread; bump lock TTL every `heartbeat_interval`s
    in a daemon thread. The daemon thread checks its own token against the
    Redis-stored token before each EXPIRE, refusing to extend a lock that
    has been re-acquired by a successor (per §8.6 CAD semantics applied to
    EXPIRE)."""
    ...
```

Implementation is a Lightweight ADR (additive — no API change;
only an internal worker enhancement). The heartbeat thread MUST:

- Use the same token-check pattern as the release script (§8.6).
- Refuse to extend a lock whose token has been overwritten.
- Stop heartbeating on stage exit (success or failure).

### §8.5 Duplicate execution prevention

Three layers protect against duplicate PROCESSING for the same
`job_id`:

1. **NX EX lock acquisition.** Only one worker can hold the lock
   at any moment.
2. **State guard.** Even if the lock is acquired (stale TTL), a
   non-QUEUED status means the job has already advanced; the
   redelivered worker exits.
3. **Compare-and-delete release.** A worker whose TTL expired
   mid-execution will fail to release a successor's lock.

These layers are detailed in `./eventing.md` §5.4.1. This spec
adopts the contract verbatim.

### §8.6 Compare-and-delete release

The release script (Lua atomic):

```lua
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

This script ensures the released lock is the one this worker
acquired — never a successor's. The pipeline_worker uses this
script via `redis_conn.eval(...)`. Errors during release are
**suppressed** (the lock will auto-expire via TTL); the worker
NEVER lets release errors mask the original pipeline exception.

### §8.7 Worker race conditions

| Race | Today's resolution | Hazard? |
|---|---|---|
| Two workers receive the same redelivery | first acquires lock; second sees lock held → exits | resolved |
| Worker A's TTL expires; Worker B acquires fresh lock | Worker B sees `status != QUEUED` → exits | resolved (silent no-op) |
| Worker A's TTL expires; Worker A continues to completion | Worker A's release no-ops (CAD); Worker A's state transitions still go through (and pass `_VALID_TRANSITIONS`) | **hazard if Worker B also transitioned** — see below |
| Worker A's TTL expires; Worker B acquires lock; Worker A and B both reach the terminal transition concurrently | only one transition succeeds (WATCH/MULTI/EXEC). The losing transition raises `ValueError` and is caught | partial — see §8.8 |
| The hash is mutated externally (debugger, ops tool) | not detected | **P0 violation** if used in prod |

### §8.8 Concurrent terminal transition hazard

When Worker A's TTL expired but Worker A is still running, and
Worker B (a redelivery) also reaches `run_pipeline`, both workers
race against each other:

- Worker B enters `run_pipeline`, takes a fresh lock, sees
  `status == PROCESSING` (because Worker A flipped it earlier) →
  exits via state guard. Safe.
- BUT Worker A's lock has expired; if Worker A's `update_status(...,
  COMPLETED)` happens after Worker B's release-attempt, Worker A's
  release is a no-op (Worker B's token doesn't match Worker A's).
  Worker A's terminal transition still succeeds. Safe.

The only **genuinely dangerous** scenario is if Worker A
re-emits typed events (e.g., re-publishes `JOB_COMPLETED`) when
the consumer expected exactly-once. But the eventing layer
guarantees only at-least-once (`./eventing.md` §5.1); duplicate
terminal events are visible but consumers MUST be idempotent. This
is the system's at-least-once contract working correctly.

This concurrent-terminal-transition hazard is acceptable today.
A future heartbeat (§8.4) tightens the window to near-zero by
preventing TTL expiry in the first place.

### §8.9 Lock expiry hazards summary

The two operationally-meaningful hazards:

| Hazard | Mitigation today | Target-state mitigation |
|---|---|---|
| Stranded PROCESSING after worker crash | `JOB_TTL_SECONDS` eventually reclaims hash (loses audit) | reaper that force-fails stranded PROCESSING |
| Lock-loss for long stages (TTL expires mid-execution) | state guard prevents duplicate body execution | heartbeat-based lock extension (§8.4) |

Both are tracked under Gap **J-JP-5**.

---

## §9 Replay + Determinism

### §9.1 Replay definition

Replay = re-running a job's pipeline body against its preserved
inputs to verify or re-derive the original output. The
authoritative inputs are:

- `Job.metadata` — the original ingest payload (immutable per §2.3).
- `Job.stages` — per-stage outputs from the original run (the
  intermediate engine inputs).
- (target-state) snapshotted config versions (`Job.metadata.<engine>_config_version`).
- The event log per `./eventing.md` §8.6 (for events that carry
  payload not already in `stages`).

### §9.2 Replay authority

Replay is performed by:

| Authority | Purpose |
|---|---|
| **qa domain** | invariant test suite — verifies engines reproduce stored decisions (per `./policy_engine.md` §13.2, `./confidence_engine.md` §13.2, `./decision_engine.md` §7.2) |
| **operator** | dispute resolution — re-runs a flagged decision under different config to evaluate proposed threshold changes (per A6 dispute path) |
| **security domain** | enforcement audit — verifies stored A4 records match recomputed values (`docs/security/enforcement_audit.md` *(planned)*) |

Replay is **never** performed by:

- The worker (the worker only runs forward-pipeline, never replay).
- The API endpoints (HTTP-side replay would imply a state-machine
  effect; replay is read-only).
- An engine (engines have zero I/O and zero memory of past runs).

### §9.3 Replay boundaries

Replay is **read-only**:

- Does NOT mutate `Job.status` (a COMPLETED job stays COMPLETED).
- Does NOT mutate `Job.stages` or any other `Job` field.
- Does NOT mutate the per-job event log (replay events go to a
  separate target-state log per `./eventing.md` §3.8).
- Does NOT trigger downstream enforcement (e.g., re-running a
  TAKEDOWN does NOT re-issue the takedown notice).

### §9.4 Determinism requirements

For replay to be meaningful, the inputs MUST reproduce the same
outputs. The system's determinism guarantees, by layer:

| Layer | Determinism contract | Reference |
|---|---|---|
| `decision_engine.compute_risk` | pure function; same `(DecisionInput, ThresholdConfig)` → same `RiskScore` | `./decision_engine.md` §8 |
| `confidence_engine.compute_confidence` | pure function; same `(ConfidenceInput, ConfidenceConfig)` → same `ConfidenceBreakdown` | `./confidence_engine.md` §12 |
| `policy_engine.evaluate_policy` | pure function; same `(DecisionOutput, ConfidenceBreakdown, PolicyContext)` → same `PolicyResult` (modulo A3 invariant guard) | `./policy_engine.md` §12 |
| Worker input construction | NOT a pure function — reads from `job.metadata`, `job.stages`, trust registry, observation store; replay MUST snapshot these | this spec — Gap **J-JP-6** |
| Storage of inputs | snapshotted in `job.metadata`, `job.stages`, event log, trust registry, observation store | this spec + `storage.md` *(planned)* |

**The replay-determinism gap (J-JP-6)**: the worker's input
construction is not currently snapshotted as a unit. The trust
scores read at decision time depend on the trust registry's
*current* contents; the observation count depends on the
observation store's *current* contents. If those stores have been
mutated since the original run, replay produces different inputs
→ different outputs → false-positive replay mismatches.

The fix: snapshot the constructed inputs (the `DecisionInput` /
`ConfidenceInput` / `PolicyContext` structs that the worker
assembled) into `job.stages.evaluation.inputs` and
`job.stages.decision.inputs`. Then replay reconstructs from the
snapshot, not from the live registries.

### §9.5 Immutable evidence requirements

Per A7 (EVIDENCE PRESERVATION), the replay-attributable inputs
MUST be:

- **Immutable** once written.
- **Queryable** by replay tools.
- **Long-lived** per the retention policy.

Today:

- `Job.metadata` is immutable (§2.3) — ✅
- `Job.stages.<stage>` is overwritten only by retry replays of the
  same stage — ⚠️ partial: retry will overwrite, losing the
  original's stage output.
- Event log is append-only — ✅
- Trust registry, observation store mutate freely — ❌ (not part
  of the per-job snapshot today).

Hardening (target-state):

1. On retry, record the prior stage outputs into a
   `job.stages_history[]` list before overwriting (Lightweight
   ADR).
2. Snapshot trust + observation reads into `job.stages.evaluation.inputs`
   (closes J-JP-6).
3. Pin engine config versions in `Job.metadata` at create-job time
   (closes the open Lightweight ADR in §2.5).

### §9.6 Engine replay expectations

The replay tool's per-engine contract:

```
def replay_decision(job_id) -> bool:
    job = job_store.get_job(job_id)
    decision_input = reconstruct_decision_input(job)
    threshold_config = load_threshold_config(job.metadata.decision_config_version)
    recomputed = decision_engine.compute_risk(decision_input, threshold_config)
    stored = job.stages.evaluation.risk
    return float_close(recomputed.composite, stored.composite, tolerance=1e-4)
```

Tolerance suggested at `1e-4` (4-decimal-place) per
`./policy_engine.md` §10.1's `evaluation_hash` 4dp rounding
convention — same tolerance for cross-engine replay comparisons.

Mismatch indicates either:

- A non-deterministic code path (P0 bug per
  `docs/constitution/GOVERNANCE.md` §5).
- An evidence-corruption (P0 violation under A7).
- A config-version mismatch (P1 — not a determinism violation but
  worth investigating).

---

## §10 Failure Handling

### §10.1 Stage failure semantics

A stage failure = an engine call inside a `stage_event` block
raises an exception. The flow:

```
1. The stage_event context manager catches the exception.
2. It emits a lifecycle FAILED event with {error_type, error_message}
   and the stage's wall-clock latency.
3. It re-raises the original exception.
4. The worker's outer try/except catches the re-raised exception.
5. Worker calls job_store.set_failure(job_id, "<type>: <msg>") —
   transitions PROCESSING → FAILED.
6. Worker publishes JOB_FAILED with {error_type, error_message,
   stage} (target-state stage field per `./eventing.md` §3.6).
7. Worker releases the lock (CAD).
```

Today **all** stage failures end in FAILED terminal. Target-state
introduces classification before step 5: transient failures route
to RETRY_PENDING instead.

### §10.2 Partial completion

Stages that succeeded before the failure retain their outputs in
`job.stages`:

| Pre-failure stage | Post-failure visibility |
|---|---|
| fingerprint | `job.stages.fingerprint = {hash, model_version, source_mode}` preserved |
| embedding | `job.stages.embedding = {vector, model_version}` preserved |
| ... | each stage's output is a Redis HSET that has already committed |

This partial state is **operator-grade debugging signal**, not
a "resume from checkpoint" facility. The system does NOT
checkpoint-resume; a retry restarts from the beginning of
`run_pipeline` (§10.3).

### §10.3 Retry restarts from beginning

A retry re-runs the **entire pipeline body**. The worker does not
inspect `job.stages` to decide which stages to skip. This is
intentional:

- Engine determinism guarantees re-running an already-completed
  stage produces the same output (zero side effect on the result).
- Skip-existing-stages logic adds complexity for marginal benefit
  (the pipeline body is fast).
- Restart-from-beginning sidesteps the cross-stage state hazard
  ("if the registry mutated between attempts, do the inputs to
  later stages still make sense?").

A future Standard ADR may introduce stage-level skip-on-existing
optimisation if the pipeline body becomes long enough to warrant
it; today's body is fast enough that the optimisation is not
worth the complexity.

### §10.4 Orphan handling

An "orphan" is an event blob written without its index entry — the
result of a worker crash between the SET and ZADD calls in
`emit()` (`./eventing.md` §8.1). Orphans are:

- **Invisible** to `list_events` (no index entry → not returned).
- **Persistent** in Redis until manual cleanup or
  `JOB_TTL_SECONDS` expiry.

Today there is no orphan reaper. Acceptable for MVP because
orphans don't corrupt downstream state. A target-state transactional
outbox closes the gap (`./eventing.md` §13.4).

A second class of orphans is **stranded PROCESSING jobs** (§8.3).
A reaper process that scans for `status == PROCESSING AND
updated_at < (now - threshold)` and force-fails them is a
target-state Lightweight ADR.

### §10.5 Worker crash behavior

When a worker process crashes mid-execution:

| Resource | State |
|---|---|
| `lock:job:{job_id}` | held by the crashed worker; auto-expires after 300s TTL |
| `job:{job_id}.status` | stuck at PROCESSING (no transition occurred) |
| `job:{job_id}.stages` | partial — last successful stage's output is committed |
| Per-job event log | partial — events up to the crash are committed |
| Subsequent redelivery | acquires fresh lock after TTL expiry; state guard sees PROCESSING; **exits silently** |

The job is **stranded** (Gap **J-JP-5**). Operators must
force-fail or rely on `JOB_TTL_SECONDS` to reclaim. Target-state
reaper closes this gap.

### §10.6 Broker redelivery interaction

Broker redelivery interacts with the lock + state guard in three
patterns:

| Pattern | Today's outcome |
|---|---|
| RQ redelivers a job after worker ACK timeout | new worker acquires lock; sees PROCESSING (or terminal); exits silently |
| RQ redelivers a job after worker process death (no ACK) | new worker acquires lock; sees PROCESSING (stranded — §10.5); exits silently → job stranded forever |
| RQ redelivers a job whose transient failure already scheduled retry (target-state) | new worker acquires lock; sees RETRY_PENDING; exits silently. (Note: RQ does not natively track our state machine; this requires the retry scheduler to manage redelivery lifecycle.) |

The state guard is the **load-bearing primitive** that makes
at-least-once delivery safe. Removing it would expose the
pipeline to genuine duplicate execution.

### §10.7 Compensating semantics

Per `./eventing.md` §6.6, the system has **no compensating
actions** at the runtime layer. A reversal of a terminal action
under A6 (HUMAN REVIEW AUTHORITY) is recorded as an **appended
audit entry** at the security layer — never as a state-machine
transition.

This is consistent with terminal-state absorption (§3.9) and the
append-only audit guarantee (§2.7).

---

## §11 Observability

### §11.1 Execution tracing

Today's tracing is the **per-job event log**. A job's execution
trace is:

1. `INGEST_RECEIVED` (API boundary).
2. Lifecycle STARTED + COMPLETED|FAILED for each stage.
3. Domain events per stage (FINGERPRINT_READY, EMBEDDING_READY,
   ...).
4. Terminal `JOB_COMPLETED` | `JOB_FAILED` (+ target-state
   `JOB_RETRY_PENDING`, `JOB_DEAD_LETTERED`, `JOB_CANCELLED`).

The lifecycle layer carries `latency_ms` per stage, enabling
per-stage timing analysis. There is no distributed tracing
(OpenTelemetry / Jaeger) today; a future ADR adds it
(`docs/specs/observability.md` *(planned)*).

### §11.2 Stage metrics

Recommended metrics (target-state, per
`docs/specs/observability.md` *(planned)*):

| Metric | Type | Cardinality |
|---|---|---|
| `pipeline.job.created` | counter | global |
| `pipeline.job.terminal.{COMPLETED,FAILED,FLAGGED,CANCELLED,DEAD_LETTERED}` | counter | per terminal state |
| `pipeline.stage.duration_ms.{<stage>}` | histogram | per stage |
| `pipeline.stage.failure.{<stage>}` | counter | per stage |
| `pipeline.retry.{transient,broker_loss,...}` | counter | per failure class |
| `pipeline.dlq.depth` | gauge | global |
| `pipeline.lock.contended` | counter | global |
| `pipeline.lock.expired_during_processing` | counter | global (Gap J-JP-5 signal) |
| `pipeline.stranded_processing.detected` | counter | global (target-state reaper signal) |

These are **emission** points; the metric store is the platform
domain's responsibility.

### §11.3 Latency accounting

Per-stage latency is captured by `stage_event`'s `latency_ms`
field on lifecycle COMPLETED|FAILED events. End-to-end job
latency is `(JOB_COMPLETED|FAILED.timestamp -
INGEST_RECEIVED.timestamp)`.

Latency dimensions to slice:

- Per-stage (from lifecycle events).
- Per-engine (within a target-state stage like `evaluation`,
  per-engine timing requires the worker to emit sub-stage
  lifecycle events — Lightweight ADR).
- Per-terminal-state (success vs failure latency distributions).

### §11.4 Failure attribution

Every failure carries:

- `error_type` (Python exception class name).
- `error_message` (str(exc) — single line).
- `stage` (target-state — which stage produced the exception).
- `failure_class` (target-state — classifier verdict).
- `retry_count` at time of failure (target-state).

Sufficient to attribute a failure to:

1. A specific stage (which engine raised).
2. A specific failure class (transient vs permanent).
3. A specific attempt (retry_count + last_attempt_id).
4. A specific cause (error_message + error_type).

### §11.5 Audit lineage

Every job's audit lineage is:

```
job_id ── correlates ──▶ all events
       ── correlates ──▶ Job.metadata + stages + result
       ── target-state ──▶ replay_of (link to predecessor on re-drive)
       ── target-state ──▶ A4 record (projected by security layer)
```

`job_id` is the **primary key for everything**. External systems
(audit tooling, dispute-handling UI, replay tools) all key off
`job_id`.

### §11.6 Replay visibility

Today replay is invisible on the bus (replay is hypothetical —
no tool exists). Target-state:

- A `JOB_REPLAY_REQUESTED` event is emitted at re-drive (carrying
  original `job_id` + new `job_id`).
- A `JOB_REPLAY_COMPLETED` event marks comparison success/failure.
- These events go to a **separate** replay log (per `./eventing.md`
  §3.8), not to the original job's timeline (which would corrupt
  the original audit).

---

## §12 Scaling Model

### §12.1 Horizontal worker scaling

RQ supports **N workers on the same queue**:

```
$ uv run python -m app.workers.worker  &   # worker 1
$ uv run python -m app.workers.worker  &   # worker 2
$ uv run python -m app.workers.worker  &   # worker 3
```

Workers compete for queue items via Redis BLPOP semantics:
exactly one worker dequeues each enqueued item. Combined with the
per-job lock (§8), this gives:

- **One worker per job** (the lock guarantees this).
- **Multiple jobs in flight** (one per worker).

Throughput scales linearly with worker count up to Redis
saturation. Beyond Redis saturation, the bottleneck is the broker
itself; sharding (§12.3) is the answer.

### §12.2 Concurrency boundaries

| Concept | Concurrency unit |
|---|---|
| Per-job execution | **strictly serial** (the lock ensures this) |
| Per-stage execution within a job | strictly serial (the worker is sequential) |
| Multiple jobs across workers | parallel (RQ load-balances) |
| Multiple A1 phases within one job | sequential (per A1 ordering) |
| (target-state) DecisionEngine + ConfidenceEngine within EVALUATION | optional parallel (engines are independent) |

The fundamental serialisation point is the **per-job lock**. The
fundamental parallelism point is the **per-job pickup by RQ**.

### §12.3 Queue partitioning

Today's single `pipeline` queue is sufficient for MVP. Target-
state partitioning options (per `./eventing.md` §11.2):

| Dimension | Queue scheme | Trade-off |
|---|---|---|
| **By tenant** | `pipeline:{tenant_id}` | per-tenant rate-limit + isolation; harder ops |
| **By content type** | `pipeline:video`, `pipeline:image` | dedicated worker pools for heavy media |
| **By region** | `pipeline:eu`, `pipeline:us` | GDPR / DSA compliance; latency localisation |
| **By priority** | `pipeline:high`, `pipeline:normal`, `pipeline:low` | priority lanes for SLA-bound work |

Each is a **Standard ADR** with cross-domain impact (pipeline +
api + platform + security all participate). This spec records
that partitioning is feasible without violating A1 (since A1 is
orchestration-agnostic), but the per-job ordering guarantee
becomes per-partition-job-ordering after partitioning lands
(per `./eventing.md` §11.2).

### §12.4 Backpressure semantics

Today there is no backpressure (per `./eventing.md` §11.5). If
ingest exceeds worker throughput, the queue grows.

Target-state backpressure options:

| Option | Mechanism | Trade-off |
|---|---|---|
| **API gate** | reject `POST /v1/ingest` with 503 when queue depth exceeds threshold | simplest; rejects the request rather than absorbing it |
| **API throttle** | rate-limit `POST /v1/ingest` per source / tenant | smoother UX; doesn't address worker scarcity directly |
| **Rolling-pause** | workers stop dequeuing temporarily under broker stress | worker-side; harder to operate |

Recommended near-term: API gate (Lightweight ADR per
`./eventing.md` §11.5). Target-state: combination of API throttle
+ per-tenant priority queues.

### §12.5 Fairness

RQ provides **FIFO** within a queue. Two consequences:

- The first job enqueued is the first dequeued.
- Long-running jobs do NOT block subsequent jobs from being picked
  up by other workers.
- A flood of jobs from a single source can starve other sources
  unless partitioning is in place.

Per-source fairness is a target-state concern (per-tenant queues
or weighted-fair-queueing) and is out of scope for this v1.0
spec.

### §12.6 RQ scheduler

The current `worker.py::main()` invokes `worker.work(with_scheduler=False)`.
This means RQ's `enqueue_in(...)` (deferred / scheduled jobs) is
**not active**. Enabling the scheduler is required for the
target-state retry mechanism (§6.5 Option A).

Enabling is a **Lightweight ADR** (single-line change:
`with_scheduler=True` and ensure exactly one worker process runs
the scheduler — RQ supports `--with-scheduler` per worker).

### §12.7 Worker class selection (Windows vs Linux)

The current `worker.py::_select_worker_class()`:

- **Windows** → `SimpleWorker` (no fork; single-process).
- **Linux / macOS** → `Worker` (forks a child per job).

This is a **deployment-portability** decision: SimpleWorker
guarantees the same behaviour on Windows that Worker provides on
Unix, at the cost of fork-based parallelism (one job at a time
per process). On Windows, scaling = launching more SimpleWorker
processes.

This is canonical and remains so under all foreseeable evolutions.
Switching to a different parallelism primitive (asyncio,
multiprocessing) is a Standard ADR per §1.4.

---

## §13 Security + Governance

### §13.1 Execution authority

| Action | Authority |
|---|---|
| Create job | API endpoint (`POST /v1/ingest`) |
| Transition QUEUED → PROCESSING | worker (lock + state guard) |
| Transition PROCESSING → COMPLETED | worker (terminal, success path) |
| Transition PROCESSING → FAILED | worker (terminal, exception path) |
| Transition PROCESSING → FLAGGED | worker (terminal, action-requires-attention) |
| Transition PROCESSING → RETRY_PENDING (target) | worker (transient classification) |
| Transition RETRY_PENDING → QUEUED (target) | retry scheduler (RQ deferred / separate process) |
| Transition RETRY_PENDING → DEAD_LETTERED (target) | retry scheduler (budget exhaustion) |
| Transition * → CANCELLED (target) | API endpoint with operator authorisation |
| Re-drive DEAD_LETTERED (target) | API endpoint with operator authorisation |
| Mutate `job_id` / `created_at` / `metadata` | **forbidden — P0** |

### §13.2 Forbidden mutations

| Forbidden | Severity | Why |
|---|---|---|
| Mutating `job_id` | **P0** | corrupts replay attribution + audit lineage |
| Mutating `created_at` | **P0** | corrupts retention windows + temporal audit |
| Mutating `metadata` | **P0** | corrupts replay attribution under A5 |
| Skipping states (transition not in `_VALID_TRANSITIONS`) | **P0** | violates state-machine integrity |
| Transitioning out of a terminal state | **P0** | violates absorbing-state guarantee |
| Direct Redis writes to `job:*` keys (bypassing JobStore) | **P1** | bypasses validation + atomicity guards |
| Direct Redis writes to event log (bypassing `emit()`) | **P0** | corrupts append-only audit |
| Acquiring per-job lock via raw `SET` (bypassing `_acquire_lock`) | **P1** | bypasses CAD release contract |
| Releasing a lock without token check (raw `DEL`) | **P0** | can release a successor's lock; corrupts concurrency safety |
| Bypassing `_VALID_TRANSITIONS` to force a transition | **P0** | corrupts state-machine integrity |
| Removing the at-least-once delivery floor | Constitutional | `./eventing.md` §12.3 |

### §13.3 Operator controls

Target-state operator controls:

| Control | Endpoint (target-state) | Purpose |
|---|---|---|
| Cancel a job | `POST /v1/jobs/{id}/cancel` | abort QUEUED or PROCESSING |
| Force-fail a stranded PROCESSING job | `POST /v1/jobs/{id}/force-fail` | unstrand stuck jobs (J-JP-5) |
| Re-drive a DEAD_LETTERED job | `POST /v1/jobs/{id}/replay` | quarantine release |
| List DLQ contents | `GET /v1/jobs/dead-letter` | DLQ inspection |
| Pause queue | `POST /v1/admin/pipeline/pause` | drain workers in incidents |
| Resume queue | `POST /v1/admin/pipeline/resume` | post-incident recovery |

Each requires authorisation. The auth contract belongs to the
api spec + security spec; this spec records the **runtime
contract**: the worker honours these primitives, but does not
itself authorise.

### §13.4 ADR boundaries

Per the §1.4 modification matrix, new state additions are
Standard, transition-table edits are Standard, and DLQ /
retry / cancellation introductions are Standard. The bundled
"engine-triple wiring" ADR is the largest cross-domain change
this spec anticipates (§5.5).

Some changes are **constitutional**: removing terminal-state
absorption, reordering A1 phases, removing append-only audit —
all of these would touch axioms and require Constitutional ADR.

### §13.5 Workflow-engine antipattern

This spec MUST NOT evolve into a workflow engine. Forbidden
conflations:

- **No DAG semantics.** The pipeline is a fixed linear sequence,
  not a configurable DAG. Adding conditional routing /
  branching is forbidden unless a Constitutional ADR re-aligns
  with A1.
- **No cron / schedule semantics.** Jobs are triggered by API
  calls (or future event subscribers), not by schedules.
- **No subscriber-driven workflow.** Subscribers consume events;
  they do NOT drive job state transitions.
- **No Temporal/Cadence/Step Functions abstraction.** The job is
  a Python control-flow object inside a single worker; the
  durability comes from Redis state + event log. We do not
  implement workflow primitives (signals, queries, child
  workflows).

The spec is a **distributed batch executor**, not a workflow
engine. This boundary is **constitutional** in spirit (the
spec's character) even if not codified in AXIOMS.md as such.

### §13.6 Governance ownership

- **pipeline domain** owns this spec, the `Job` record, the
  `JobStatus` enum, the `_VALID_TRANSITIONS` table, the worker
  source, the queue topology, the lock contract, and (with
  eventing) the at-least-once + per-job-ordering invariants.
- **api domain** owns the HTTP shapes that surface job state.
- **decision / confidence / policy domains** own the engines
  that the worker invokes; they do NOT own the worker.
- **security domain** owns the authorisation contract for
  operator controls + the audit-record projection.
- **platform domain** owns the broker (Redis), the deployment
  topology, and the metric infrastructure.

Cross-domain changes require Standard ADR per
`docs/constitution/GOVERNANCE.md` §3 + §6.

---

## §14 Current vs Target State

### §14.1 Implemented runtime

| Component | Status |
|---|---|
| `Job` dataclass + `JobStore` Redis-backed | **IMPLEMENTED** |
| 5-state state machine (QUEUED, PROCESSING, COMPLETED, FAILED, FLAGGED) | **IMPLEMENTED** |
| `_VALID_TRANSITIONS` enforced via WATCH/MULTI/EXEC | **IMPLEMENTED** |
| Distributed lock (SET NX EX 300 + Lua CAD release) | **IMPLEMENTED** |
| State guard (status != QUEUED → exit) | **IMPLEMENTED** |
| Sequential pipeline worker | **IMPLEMENTED** |
| `INGEST_RECEIVED` emission at API boundary | **IMPLEMENTED** |
| Per-stage `stage_event` lifecycle wrapping | **IMPLEMENTED** |
| Per-stage `update_stage(...)` storage | **IMPLEMENTED** |
| Terminal `JOB_COMPLETED` / `JOB_FAILED` | **IMPLEMENTED** |
| `set_failure(job_id, reason)` exception handler | **IMPLEMENTED** |
| RQ worker (Windows SimpleWorker / Linux Worker) | **IMPLEMENTED** |
| Sliding `JOB_TTL_SECONDS` on job hash | **IMPLEMENTED** |
| `GET /v1/jobs/{id}` + `GET /v1/jobs/{id}/events` | **IMPLEMENTED** |

### §14.2 Partially implemented runtime

| Component | Gap |
|---|---|
| Engine-triple integration | **legacy `scoring_engine` + `enforcement_engine` in use** (J-JP-4) |
| Replay-attributable inputs | `Job.metadata` immutable but trust + observation reads not snapshotted (J-JP-6) |
| Engine version triad in `Job.metadata` | versions live in stage outputs only (J-JP-8) |
| Action mapping in terminal selection | 3-action `ALLOW/FLAG/BLOCK` only (J-JP-1) |
| Unused stub Job model | `backend/app/models/job.py` carried forward (J-JP-7) |
| docker-compose runtime stack | API + frontend only; **no Redis service, no worker service in compose** — production deployment incomplete (J-JP-10) |

### §14.3 Target-state runtime

| Capability | Resolution path |
|---|---|
| 8-state state machine (adds CANCELLED, RETRY_PENDING, DEAD_LETTERED) | Standard ADR (§3.4) |
| Failure classifier | Standard ADR bundled with retry policy (§6.1) |
| Retry counters + backoff (`retry_count`, `max_retries`, `next_attempt_at`, `failure_class`) | Standard ADR (§6.3) |
| RQ scheduler enabled (`with_scheduler=True`) | Lightweight ADR (§12.6) |
| Retry orchestration (RQ deferred jobs) | Standard ADR (§6.5) |
| DLQ topology (`pipeline_dlq` queue) | Standard ADR (§7.6) |
| Operator endpoints (cancel / force-fail / re-drive / list DLQ) | Standard ADR (cross-domain — api + security + this spec) (§13.3) |
| Cancellation primitive | Standard ADR (§7.4) |
| Heartbeat-based lock extension | Lightweight ADR (§8.4) |
| Stranded-PROCESSING reaper | Lightweight ADR (§10.5) |
| Engine-triple wiring | Standard ADR (§5.5) |
| Engine version triad in `Job.metadata` | Lightweight ADR (§2.5) |
| Snapshot of replay-attributable inputs | Lightweight ADR (§9.4–§9.5) |
| Trust shape unification with ConfidenceEngine | Standard ADR (per `./decision_engine.md` D-DE-4) |
| Per-tenant / per-content-type / per-region partitioning | Standard ADR per dimension (§12.3) |
| Backpressure (API gate) | Lightweight ADR (§12.4) |
| Pause/resume admin endpoints | Standard ADR (§13.3) |
| Replay tooling | jointly with `docs/testing/INVARIANT_TESTS.md` *(planned)* |
| Postgres durable mirror | Standard ADR; tied to `docs/specs/storage.md` *(planned)* |

### §14.4 Deployment topology gap

`docker-compose.yml` today contains only:

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
```

Missing:

- **`redis`** service — the broker, lock store, event store, and
  job store. Without it, the API can start but the worker has no
  Redis to connect to, and the API's enqueue / publish_event
  calls fail.
- **`worker`** service — runs `python -m app.workers.worker`.
  Without it, jobs are queued but never processed.
- **(future) `pipeline_dlq` worker** — separate worker for the
  DLQ queue.
- **(future) `scheduler`** — RQ scheduler process for retry
  delivery.

This is **Gap J-JP-10** — a deployment-level closure, separate
from the runtime closure. A target-state docker-compose has all
five services with appropriate dependencies.

---

## §15 Unresolved Gaps

Formal gap IDs for tracking. These compose with `./eventing.md`
E-EV-* and `./decision_engine.md` D-DE-* via cross-references.

| ID | Gap | Severity | Resolution path |
|---|---|---|---|
| **J-JP-1** | Action mapping is 3-action (`ALLOW/FLAG/BLOCK`) at the terminal-selection step; target is 5-action `PolicyAction` | High | bundled with J-JP-4 (engine-triple wiring) |
| **J-JP-2** | No failure classifier — every exception is treated as terminal FAILED; transient and permanent are not distinguished | High | Standard ADR (retry policy) |
| **J-JP-3** | No DLQ — failed jobs accumulate in FAILED terminal with no quarantine | High | Standard ADR (DLQ topology + operator endpoints) |
| **J-JP-4** | Engine triple unwired — alias for `./eventing.md` E-EV-6 + `./decision_engine.md` D-DE-2 | **Highest leverage** | Standard ADR (§5.5) |
| **J-JP-5** | No heartbeat-based lock extension; long stages can race against TTL; stranded PROCESSING jobs accumulate after worker crashes | Medium | Lightweight ADR (heartbeat) + Lightweight ADR (reaper) |
| **J-JP-6** | Replay-attributable inputs not snapshotted; trust + observation reads can mutate between original and replay | Medium | Lightweight ADR (snapshot inputs in `job.stages.evaluation.inputs`) |
| **J-JP-7** | Unused stub `Job` model in `backend/app/models/job.py` shadows the canonical `Job` in `core/job_store.py`; risk of accidental import | Low | PR-only (delete stub); jointly with the engine-triple wiring PR |
| **J-JP-8** | Engine version triad (`policy_version`, `decision_config_version`, `confidence_config_version`) lives in stage outputs only; not in `Job.metadata` | Low | Lightweight ADR (additive `Job.metadata` field) |
| **J-JP-9** | No static enforcement of stage contracts (lifecycle wrapping, `update_stage`, `publish_event`); enforcement via code review | Low | Lightweight ADR (stage decorator) |
| **J-JP-10** | docker-compose has no `redis`, `worker`, scheduler, or DLQ services; deployment topology incomplete | High | PR (docker-compose + helm chart skeleton); platform-domain ownership |
| **J-JP-11** | API ingest order: `enqueue` precedes `publish_event(INGEST_RECEIVED)`; consumers MUST NOT assume INGEST_RECEIVED is the first event in the per-job log (also documented in `./eventing.md` §4.5) | Low | Lightweight ADR (swap order) OR canonical doc note (already done in eventing.md) |
| **J-JP-12** | `Settings.JOB_TTL_SECONDS` defaults to `None` (no expiry) — production risk if not set | Medium | Lightweight ADR (set production-safe default + docs) |
| **J-JP-13** | RQ scheduler not enabled (`with_scheduler=False`); blocks target-state retry mechanism Option A | Low | Lightweight ADR (§12.6) |

### §15.1 Cross-spec gap aliases

| This spec | Aliased gap |
|---|---|
| J-JP-4 | `./eventing.md` E-EV-6 |
| J-JP-4 | `./decision_engine.md` D-DE-2 |
| J-JP-5 (stranded PROCESSING) | `./eventing.md` operational hazard noted in §6.1 / §10.5 |
| J-JP-10 (deployment) | new — not in eventing or decision specs |
| J-JP-11 (ingest emit order) | `./eventing.md` §4.5 |

---

## §16 Reconciliation history

This spec is the **full canonical successor** to two
TRANSITIONAL rule files. Each is annotated below with the
specific drift and the resolution adopted.

### §16.1 J-JP-A — `.claude/rules/job-processing.md` superseded

**Drift:**

- The file pinned the lifecycle as `QUEUED → PROCESSING →
  COMPLETED | FAILED | FLAGGED` but did not address CANCELLED,
  RETRY_PENDING, or DEAD_LETTERED.
- Required `retry_count` + `max_retries` per job, but no such
  fields exist in the current `Job` dataclass.
- Required atomic transitions and locking — implemented via
  WATCH/MULTI/EXEC + Lua CAD lock.
- Required structured per-job logging (created_at, updated_at,
  duration, failure_reason) — implemented in `Job` record.
- Forbade in-memory stores in production — `JobStore` is now
  Redis-backed (the rule's "future direction" is the present).

**Resolution adopted:** §3 (state machine), §6 (retry), §7 (DLQ),
§8 (locking), §11 (observability) materialise the rule's
requirements at canonical-spec authority. The 5-state machine is
acknowledged as MVP; the 8-state machine is the canonical target.

The rule file is **superseded in full**. Per the append-only
migration constraint, the file is NOT deleted; it is annotated
with a `superseded by:` deprecation note (PR-only) when next
edited.

### §16.2 J-JP-B — `.claude/rules/job_system.md` superseded

**Drift:**

- Same lifecycle pinning as J-JP-A.
- Required idempotent jobs, atomic status updates, every-job
  metadata — all implemented or canonicalised in this spec.
- "Future direction": replace in-memory with Redis/Postgres + add
  distributed queue (Kafka / Redis Streams) — Redis + RQ is the
  current realisation; Postgres mirror remains target-state.
- Forbade business logic in JobStore — preserved (this spec §1.3
  + §4.3).
- Forbade blocking execution during job processing — preserved
  (the API endpoint is non-blocking; worker is the blocking
  point but isolated in its own process).

**Resolution adopted:** §1.3 (separation of concerns), §2 (job
model), §3 (state machine), §4 (worker contracts), §13 (governance)
canonicalise the rule's requirements.

The rule file is **superseded in full**. Same append-only
constraint as J-JP-A.

### §16.3 Documentation lineage

| Source | Status | Location |
|---|---|---|
| `.claude/rules/job-processing.md` | TRANSITIONAL — superseded in full by this spec | `.claude/rules/` |
| `.claude/rules/job_system.md` | TRANSITIONAL — superseded in full by this spec | `.claude/rules/` |
| `backend/app/core/job_store.py` docstring | implementation; remains authoritative for code-level details | `backend/app/core/` |
| `backend/app/workers/pipeline_worker.py` docstring | implementation; remains authoritative for code-level details | `backend/app/workers/` |

Both rule files are candidates for removal in Phase-2 closeout
once all canonical specs are landed.

### §16.4 Sibling-spec reconciliation

This spec coexists with three siblings that share the runtime
contract surface:

- **`./eventing.md`** — substrate (delivery, ordering, retention).
  This spec consumes its primitives without duplicating them.
- **`./decision_engine.md`** / **`./confidence_engine.md`** /
  **`./policy_engine.md`** — the engines this spec invokes. Their
  determinism contracts back this spec's replay semantics (§9).

The four together — eventing + job_processing + the three engine
specs — fully canonicalise the system's runtime + decision
substrate. Remaining specs (`api_contracts.md`, `storage.md`,
`observability.md`, `enforcement_audit.md`) are **consumers** of
the substrate, not co-definers.

---

## §17 Open questions / Future work

Documented for visibility; not commitments.

- **Engine-triple wiring** (J-JP-4) — **the highest-leverage runtime
  closure**. Bundled Standard ADR per §5.5; closes
  `./eventing.md` E-EV-6 and `./decision_engine.md` D-DE-2 in the
  same PR.
- **Failure classifier + retry policy** (J-JP-2, J-JP-13) —
  Standard ADR; introduces `RETRY_PENDING` state, `retry_count`
  + `max_retries`, exponential backoff with full jitter, RQ
  scheduler enabled.
- **DLQ** (J-JP-3) — Standard ADR; introduces `DEAD_LETTERED`
  state, `pipeline_dlq` queue, operator endpoints.
- **Cancellation** (§7.4 / J-JP table) — Standard ADR; introduces
  `CANCELLED` state, `POST /v1/jobs/{id}/cancel` endpoint.
- **Heartbeat-based lock extension** (J-JP-5 part 1) —
  Lightweight ADR; daemon thread per stage extends lock TTL.
- **Stranded-PROCESSING reaper** (J-JP-5 part 2) — Lightweight
  ADR; periodic process flips stale PROCESSING → FAILED.
- **Snapshot replay-attributable inputs** (J-JP-6) — Lightweight
  ADR; capture trust + observation reads in
  `job.stages.evaluation.inputs`.
- **Engine version triad in `Job.metadata`** (J-JP-8) —
  Lightweight ADR; additive field set.
- **Stage decorator for contract enforcement** (J-JP-9) —
  Lightweight ADR; reduces boilerplate + enforces lifecycle
  wrapping.
- **Production deployment topology** (J-JP-10) — PR (docker-
  compose + helm chart skeleton); platform-domain ownership.
- **`Settings.JOB_TTL_SECONDS` production default** (J-JP-12) —
  Lightweight ADR; set conservative default + document
  retention semantics.
- **Stub `Job` model removal** (J-JP-7) — PR-only.
- **Per-tenant / per-content-type / per-region partitioning**
  (§12.3) — Standard ADR per dimension.
- **Backpressure** (§12.4) — Lightweight ADR (API gate).
- **Pause/resume admin endpoints** (§13.3) — Standard ADR.
- **Replay tooling** (§9 + `./eventing.md` §3.8) — jointly with
  `docs/testing/INVARIANT_TESTS.md` *(planned)*.
- **Postgres durable mirror** — Standard ADR; tied to
  `docs/specs/storage.md` *(planned)*.

> **Important constraint reminder.** Adopting workflow-engine
> abstractions (Temporal / Cadence / Step Functions) is
> **forbidden** at the spec level (§13.5). The pipeline is a
> distributed batch executor, not a workflow engine. Switching
> queue technology (RQ → Celery / SQS) is a **Standard ADR**.
> Removing terminal-state absorption is **Constitutional**.

---

## §18 Versioning and change process

This spec is **EVOLVING** per
`docs/constitution/GOVERNANCE.md` §8. Compatibility expectations
are low — consumers should expect change at each minor bump.

| Change type | ADR tier |
|---|---|
| Doc clarification (no semantic change) | none |
| Adding additive `Job` field (e.g., `retry_count`) | Lightweight |
| Adjusting RQ `job_timeout`, lock TTL, backoff defaults | Lightweight |
| Adjusting `JOB_TTL_SECONDS` defaults | Lightweight |
| Enabling RQ scheduler (`with_scheduler=True`) | Lightweight |
| Stage decorator for contract enforcement | Lightweight |
| Heartbeat-based lock extension | Lightweight |
| Stranded-PROCESSING reaper | Lightweight |
| Snapshot replay inputs into `job.stages.evaluation.inputs` | Lightweight |
| API gate backpressure | Lightweight |
| Adding a new state to `JobStatus` | Standard |
| Removing or renaming an existing state | Standard |
| Editing `_VALID_TRANSITIONS` | Standard |
| Failure classifier + retry policy | Standard |
| DLQ topology | Standard |
| Cancellation primitive | Standard |
| Engine-triple wiring (cross-domain) | Standard |
| Operator endpoints (cross-domain — api + security) | Standard |
| Switching queue technology (RQ → Celery / SQS / Kafka) | Standard |
| Partitioning (per-tenant / per-content-type / per-region) | Standard |
| Pause/resume admin endpoints | Standard |
| Removing at-least-once / per-job-ordering / append-only | Constitutional (cross-spec — also touches `./eventing.md`) |
| Reordering A1 phases | Constitutional |
| Removing terminal-state absorption | Constitutional |
| Adopting workflow-engine abstraction (Temporal / Cadence) | Constitutional (changes the system's character per §13.5) |

A `job_processing_version` constant is **not yet present** in
`backend/app/core/job_store.py` (related to J-JP-8). When
introduced, it MUST be bumped in lockstep with this spec's
`version:` field. Mismatch is a **P1** governance violation.

### §18.1 Graduation to STABLE

Same gates as `./eventing.md` §16.1 / `./decision_engine.md` §13.5
/ `./confidence_engine.md` §16.1, plus job-processing-specific:

1. The state machine, retry policy, and DLQ topology are unchanged
   for at least one minor revision cycle.
2. No production incidents implicating job lifecycle in 90 days.
3. Consumer integrations (api, security, qa) report stability.
4. The invariant test suite covers state-transition legality,
   retry semantics, replay determinism, and lock semantics.
5. Gaps J-JP-2, J-JP-3, J-JP-4, J-JP-5, J-JP-10 are closed (or
   deliberate non-closures justified in ADR).

Architect approves graduation.

### §18.2 Demoting from STABLE

If a STABLE spec needs material change that breaks the STABLE
contract, a Standard ADR may demote it back to EVOLVING per
`docs/constitution/GOVERNANCE.md` §8.

---

## §19 Cross-references

- **Axioms** (`../constitution/AXIOMS.md`): A1 (semantic phase
  integrity — this spec materialises the orchestration that
  preserves it), A4 (audit completeness — the per-job event log +
  job hash + stages dict are the substrate), A5 (deterministic
  replay — replay tooling backs this), A6 (human review authority
  — reversal as appended audit, never state transition), A7
  (evidence preservation — replay-attributable inputs).
- **Constitutional governance**
  (`../constitution/GOVERNANCE.md`): §1 (tier hierarchy), §3 (ADR
  tiers), §5 (severity model — P0 for forbidden mutations, P1
  for spec/impl drift), §7 (EGM applies during incidents), §8
  (stability levels).
- **Domain ownership** (`../governance/DOMAINS.md`): pipeline
  domain owns this spec; api / decision / confidence / policy /
  security / qa / platform consume.
- **Architecture state** (`../state/STATE.md`): partially stale
  on JobStore + queue + worker rows — same as `./eventing.md`
  E-EV-8. STATE.md sync is a PR-only follow-up.
- **Implementation**:
  - `backend/app/core/job_store.py`
  - `backend/app/core/queue.py`
  - `backend/app/workers/pipeline_worker.py`
  - `backend/app/workers/worker.py`
  - `backend/app/api/ingest.py`
  - `backend/app/api/jobs.py`
- **Sibling specs**:
  - `./eventing.md` — runtime substrate; this spec consumes its
    primitives.
  - `./decision_engine.md` — risk-scoring engine; invoked by
    target-state EVALUATION stage.
  - `./confidence_engine.md` — confidence-scoring engine; invoked
    in parallel with DecisionEngine.
  - `./policy_engine.md` — DECISION-phase engine; invoked after
    EVALUATION.
- **Producer specs (downstream)**:
  - `./api_contracts.md` *(planned)* — defines `IngestRequest` /
    `IngestResponse` / `Job`-as-API surface; cross-references
    operator endpoints (§13.3).
- **Future canonical references**:
  - `./storage.md` *(planned)* — durable mirror (Postgres),
    retention / cold storage policy.
  - `./observability.md` *(planned)* — metrics, tracing.
  - `../security/enforcement_audit.md` *(planned)* — A4 record
    projection from `Job.result` + event log; reversal event
    semantics.
  - `../testing/INVARIANT_TESTS.md` *(planned)* — state-machine
    legality, replay determinism, retry semantics, lock
    semantics.
- **TRANSITIONAL sources** (Tier 5 — fully superseded by this
  spec):
  - `.claude/rules/job-processing.md`
  - `.claude/rules/job_system.md`
