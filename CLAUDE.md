# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo Layout

Monorepo with two deployables:

- `backend/` — FastAPI service (Python 3.11, `uv`-managed)
- `frontend/` — Next.js 16 + React 19 + Tailwind v4 (App Router, TypeScript)
- `contracts/` — OpenAPI + JSON schemas shared between services (currently empty scaffolds)
- `infra/` — Cloud Run, Docker, Terraform, Vercel configs (scaffolds)
- `docs/` — architecture + ADRs
- `.claude/CLAUDE.md` + `.claude/rules/*.md` — **authoritative architecture and policy rules**. Read these before any non-trivial change. They override defaults.

## Commands

### Backend (from `backend/`)
```bash
uv sync                                            # install deps
uv run uvicorn app.main:app --reload --port 8001   # dev server (README/compose use 8001)
uv run python start.py                             # alternate entrypoint; reads PORT from .env (defaults to 8000)
```
No lint/test runner is wired up yet; `tests/` is empty.

Note: the `.env` ships `PORT=8000` while README + `docker-compose.yml` use `8001`. `uvicorn --port 8001` is the canonical dev invocation.

### Frontend (from `frontend/`)
```bash
npm ci
npm run dev      # next dev on :3000
npm run build
npm run lint     # eslint
```

### Full stack
```bash
docker compose up     # backend:8001, frontend:3000
```

## Architecture

The system is an **event-driven media-integrity pipeline**:

```
ingest → job → process → match → score → enforce
```

This order is fixed by `.claude/rules/architecture.md` and MUST NOT be reordered. Current code implements only the first two stages as stubs; everything beyond `job` is planned.

### Backend module layout (enforced by `.claude/rules/coding_standards.md`)
```
app/api/       # FastAPI routers — thin, no business logic
app/core/      # config + infra primitives (settings, job_store)
app/models/    # Pydantic schemas
app/services/  # business logic (empty)
app/engines/   # ML / matching engines (empty)
app/workers/   # async job consumers (empty)
app/utils/     # (empty)
```

Routers live under `/v1`: `health.py`, `ingest.py`, `jobs.py`. `main.py::create_app` wires them and attaches `settings` to `app.state`.

### Current implementation reality vs. rules

The repo is mid-MVP and intentionally violates several of its own production rules. Do not "fix" these without explicit instruction — they are known and tracked:

- `app/core/job_store.py` is an **in-memory global singleton** with a `threading.Lock`. The rules (`job-processing.md`, `storage.md`, `job_system.md`) forbid this in production and require Redis/Postgres. Treat the current store as temporary scaffolding.
- No event bus, queue, or worker exists yet — `POST /v1/ingest` creates a job in memory and returns. The planned path is `ingest → enqueue → worker → embedding → matching → scoring → enforcement` (`.claude/rules/eventing.md`).
- No persistence layer, no embeddings/fingerprint stores, no evidence store. `.claude/rules/storage.md` specifies the six storage layers the system will eventually need.
- Two `Job` definitions currently exist: the real one in `app/core/job_store.py` (dataclass with `JobStatus` enum, the source of truth) and a stub in `app/models/job.py` (`id: int, status: str`) that is unused. Prefer the `core/job_store.py` version.

### Hard constraints when extending the backend

From `.claude/CLAUDE.md` and `.claude/rules/`:

- **API routes stay thin.** No ML, no DB calls, no heavy work in `app/api/*`. Validate → create job → return `job_id`. `POST /v1/ingest` must return immediately and never process inline.
- **All heavy work is async and event-driven.** No sync ML calls in the request path. No direct DB writes across modules.
- **Jobs are idempotent and retry-safe.** Valid transitions only: `QUEUED → PROCESSING → COMPLETED | FAILED | FLAGGED`. No skipping states. `create_job` must not create duplicates.
- **Every decision must be auditable.** Each detection stores `input_media_id`, `matched_media_id`, `similarity_score`, `model_version`, `timestamp` — immutably.
- **Media IDs are content hashes (SHA-256).** Never rely on filenames. Raw media is immutable and content-addressable.
- **Thresholds are bands, not booleans.** `LOW → ignore`, `MEDIUM → human review`, `HIGH → auto-flag`. Never hardcode.
- **Observability is mandatory.** Structured JSON logs with `job_id` + `event_id`; propagate `request_id` through the pipeline. No `print` debugging.
- **Use Plan Mode first** for any feature work (`.claude/CLAUDE.md`).

### API contract (stable — do not break)

- `POST /v1/ingest` → `{ job_id, status }`, returns immediately
- `GET /v1/jobs/{job_id}` → current job state, eventually consistent
- `GET /v1/health` → liveness

Response schemas must not change without versioning (`.claude/rules/api_contracts.md`).

## Frontend notes

`frontend/AGENTS.md` carries a load-bearing warning: **this is Next.js 16 with breaking changes vs. older training data** — APIs, conventions, and file structure may differ. Check `node_modules/next/dist/docs/` before writing Next-specific code, and honor deprecation notices. `frontend/CLAUDE.md` is just `@AGENTS.md`, so the AGENTS file is the real source.

The frontend currently has only the default Next scaffold (`src/app/{layout,page}.tsx`) — no API client to the backend yet.
