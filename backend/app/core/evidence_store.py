"""Evidence store — immutable, append-only, TTL-free.

The system's enforcement actions MUST be defensible long after the
operational job state is gone. The JobStore (`job:{id}`) carries a
sliding TTL via `JOB_TTL_SECONDS`; once it expires, every field —
including the per-stage `enforcement` summary previously used as the
"evidence" record — is unrecoverable. That violates A4 / A7
(audit-completeness-with-provenance / evidence preservation) and
the storage rule "Each detection MUST store input_media_id,
matched_media_id, similarity_score, model_version, timestamp"
(`.claude/rules/storage.md` §6).

This module is the durable layer the rules mandate. Each call to
:func:`record_evidence` writes three keys atomically:

  * ``evidence:{job_id}:{ts_ns}``        — JSON blob of the evidence dict.
  * ``evidence_by_job:{job_id}``         — sorted set, scored by ts_ns.
  * ``evidence_by_input:{content_hash}`` — sorted set, scored by ts_ns;
                                            enables "show me every
                                            decision touching this asset."

Writes use ``SET NX`` so re-runs of the same `(job_id, ts_ns)` cannot
silently overwrite an earlier record. The keys carry NO TTL — evidence
persists for the life of the Redis instance. A relational mirror
(Postgres) is the long-term durable target per the storage rules; the
present in-Redis layer is the bridge that keeps governance honest while
the relational store catches up.

Concurrency: each ``record_evidence`` writes a unique-per-nanosecond
key, so there is no read-modify-write hazard. The two index updates
ride a single Redis pipeline so a partial write cannot leave a blob
unindexed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.queue import redis_conn


# ---------------------------------------------------------------------------
# Required fields (per .claude/rules/storage.md §6 + .claude/rules/enforcement.md)
# ---------------------------------------------------------------------------

REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = (
    "input_media_id",
    "matched_media_id",
    "similarity_score",
    "model_version",
    "timestamp",
)


@dataclass(frozen=True)
class EvidenceRecord:
    job_id: str
    ts_ns: int
    key: str
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# Key naming
# ---------------------------------------------------------------------------


def _evidence_key(job_id: str, ts_ns: int) -> str:
    return f"evidence:{job_id}:{ts_ns}"


def _job_index_key(job_id: str) -> str:
    return f"evidence_by_job:{job_id}"


def _input_index_key(content_hash: str) -> str:
    return f"evidence_by_input:{content_hash}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(evidence: Dict[str, Any]) -> None:
    """Ensure the storage-rule mandatory fields are present.

    `matched_media_id` is allowed to be None (no-match decisions still
    have to be auditable). Missing keys raise ValueError so callers
    cannot silently record incomplete evidence.
    """
    missing = [f for f in REQUIRED_EVIDENCE_FIELDS if f not in evidence]
    if missing:
        raise ValueError(
            f"evidence is missing required fields: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def record_evidence(job_id: str, evidence: Dict[str, Any]) -> EvidenceRecord:
    """Write an immutable evidence record. SET NX guards against
    overwrite; the two indices are updated in the same pipeline so a
    partial write cannot leave a blob unindexed.

    Returns the persisted :class:`EvidenceRecord` so callers can echo
    the durable key into events / response bodies.
    """
    _validate(evidence)

    ts_ns = time.time_ns()
    key = _evidence_key(job_id, ts_ns)
    blob = json.dumps(evidence, sort_keys=True, default=_json_default)
    content_hash = evidence.get("input_media_id")

    pipe = redis_conn.pipeline(transaction=False)
    pipe.set(key, blob, nx=True)
    pipe.zadd(_job_index_key(job_id), {key: ts_ns})
    if content_hash:
        pipe.zadd(_input_index_key(content_hash), {key: ts_ns})
    pipe.execute()

    return EvidenceRecord(
        job_id=job_id,
        ts_ns=ts_ns,
        key=key,
        payload=dict(evidence),
    )


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def list_evidence_for_job(job_id: str) -> List[Dict[str, Any]]:
    """Return all evidence records for a given job, time-ordered."""
    keys = redis_conn.zrange(_job_index_key(job_id), 0, -1)
    return _load(keys)


def list_evidence_for_input(content_hash: str) -> List[Dict[str, Any]]:
    """Return all evidence records touching a given input media id,
    time-ordered. Enables dispute reconstruction for an asset across
    every job that ever processed it."""
    keys = redis_conn.zrange(_input_index_key(content_hash), 0, -1)
    return _load(keys)


def _load(keys) -> List[Dict[str, Any]]:
    if not keys:
        return []
    blobs = redis_conn.mget(keys)
    return [json.loads(b) for b in blobs if b is not None]


def _json_default(o: Any) -> Any:
    """Coerce non-JSON-native values (sets, frozensets, bytes) into
    serialisable forms. Keeps record_evidence permissive about the
    shapes the worker assembles."""
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, bytes):
        return o.decode("utf-8", errors="replace")
    raise TypeError(f"unserialisable type: {type(o).__name__}")
