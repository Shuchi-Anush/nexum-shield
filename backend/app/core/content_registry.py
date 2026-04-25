"""ContentRegistry — canonical identity layer.

Each unique piece of content gets exactly one row keyed by `content_id` —
deterministic = sha256 of the canonicalised pHash sequence. Two ingests of
byte-identical media always produce the same content_id; two ingests of
perceptually-equivalent-but-not-bit-identical media produce *distinct*
content_ids resolved to the same canonical via mean-Hamming threshold.

Redis schema:

    content:{cid}                  HASH   canonical row
    content:{cid}:keyframes        LIST   ordered hex pHashes (one per keyframe)
    content:{cid}:stats            HASH   denormalised counters
    content:{cid}:platforms        SET    platform names where seen
    content:{cid}:registered       STRING SETNX election lock

    content_index:phash_b{N}:{prefix}     SET    LSH bucket → content_ids
    content_index:owner:{owner_id}        SET    owner → content_ids

Postgres mirror (table content_registry) is the durable source of truth;
Redis is the hot cache. Write-through is best-effort here — if the durable
write fails we still return the registered id (Postgres reaper reconciles).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.queue import redis_conn
from app.engines.fingerprint_engine import Fingerprint, MODEL_VERSION

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ContentRecord:
    content_id: str
    canonical_phash: int
    keyframe_phashes: List[int]
    keyframe_count: int
    owner: Optional[str]
    trust_level: str                   # verified | premium | basic | unknown
    duration_seconds: float
    sample_fps: float
    model_version: str
    metadata: Dict[str, Any]
    created_at: float
    first_seen_job_id: Optional[str]

    def to_public_dict(self) -> dict:
        d = asdict(self)
        d["canonical_phash_hex"] = f"0x{self.canonical_phash:016x}"
        # Drop the big phash list from public payload by default; expose count.
        d.pop("keyframe_phashes", None)
        return d


@dataclass
class MatchCandidate:
    content_id: str
    hamming_distance: float            # mean Hamming over best-aligned window
    similarity: float                  # 1 - hamming/64
    owner: Optional[str] = None
    trust_level: str = "unknown"


@dataclass
class ResolvedIdentity:
    content_id: str
    role: str                          # EXACT | VARIANT | NOVEL
    parent_content_id: Optional[str]
    similarity: float
    hamming_distance: float
    is_new_content: bool
    owner: Optional[str]
    trust_level: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(v: Any) -> Optional[str]:
    if v is None:
        return None
    return v.decode("utf-8") if isinstance(v, bytes) else (v if isinstance(v, str) else str(v))


def _key_content(cid: str) -> str:               return f"content:{cid}"
def _key_keyframes(cid: str) -> str:             return f"content:{cid}:keyframes"
def _key_stats(cid: str) -> str:                 return f"content:{cid}:stats"
def _key_platforms(cid: str) -> str:             return f"content:{cid}:platforms"
def _key_registered(cid: str) -> str:            return f"content:{cid}:registered"
def _key_lsh(prefix: int, bits: int) -> str:     return f"content_index:phash_b{bits}:{prefix:0{(bits+3)//4}x}"
def _key_owner(owner: str) -> str:               return f"content_index:owner:{owner}"


def _phash_prefix(phash: int, bits: int) -> int:
    return phash >> (64 - bits)


def _probe_prefixes(prefix: int, bits: int, radius: int) -> List[int]:
    """Multi-probe LSH: prefix + all prefixes within Hamming-`radius` flips.

    Radius 1 → 1 + bits neighbours (single-bit flips).
    Higher radii are O(C(bits, r)) — keep ≤ 1 in production.
    """
    if radius <= 0:
        return [prefix]
    out = {prefix}
    if radius >= 1:
        for i in range(bits):
            out.add(prefix ^ (1 << i))
    if radius >= 2:
        for i in range(bits):
            for j in range(i + 1, bits):
                out.add(prefix ^ (1 << i) ^ (1 << j))
    return list(out)


def derive_content_id(keyframe_phashes: List[int]) -> str:
    """Deterministic content_id from a pHash sequence.

    Stable across re-ingests of the same byte-identical media. Re-encoded
    media will differ here (different pHash sequence), but identity
    resolution will collapse them to the existing canonical via threshold.
    """
    import hashlib
    canonical_bytes = b"".join(p.to_bytes(8, "big") for p in keyframe_phashes)
    return hashlib.sha256(canonical_bytes).hexdigest()


# ---------------------------------------------------------------------------
# ContentRegistry
# ---------------------------------------------------------------------------

class ContentRegistry:
    # ---- READ ----------------------------------------------------------------

    def get(self, content_id: str) -> Optional[ContentRecord]:
        raw = redis_conn.hgetall(_key_content(content_id))
        if not raw:
            return None
        kfs = self.keyframes_of(content_id)
        return _hash_to_record(raw, kfs)

    def keyframes_of(self, content_id: str) -> List[int]:
        items = redis_conn.lrange(_key_keyframes(content_id), 0, -1)
        return [int(_decode(x), 16) for x in items] if items else []

    def stats_of(self, content_id: str) -> Dict[str, Any]:
        raw = redis_conn.hgetall(_key_stats(content_id))
        if not raw:
            return {}
        return {_decode(k): _coerce_stat(_decode(v)) for k, v in raw.items()}

    def platforms_of(self, content_id: str) -> List[str]:
        items = redis_conn.smembers(_key_platforms(content_id))
        return sorted([_decode(x) for x in items])

    def lsh_candidates(self, canonical_phash: int) -> List[str]:
        s = get_settings()
        prefix = _phash_prefix(canonical_phash, s.LSH_PREFIX_BITS)
        prefixes = _probe_prefixes(prefix, s.LSH_PREFIX_BITS, s.LSH_PROBE_RADIUS)
        seen = set()
        out: List[str] = []
        for p in prefixes:
            members = redis_conn.smembers(_key_lsh(p, s.LSH_PREFIX_BITS))
            for m in members:
                cid = _decode(m)
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
        return out

    # ---- WRITE ---------------------------------------------------------------

    def register(
        self,
        fingerprint: Fingerprint,
        *,
        owner: Optional[str] = None,
        trust_level: str = "unknown",
        job_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ContentRecord:
        """Idempotent registration.

        Election: SETNX on `content:{cid}:registered`. The winner writes the
        canonical row + keyframes + LSH bucket entry; losers and re-registers
        skip writes and return the existing record. Two workers minting the
        SAME content_id (deterministic from phash sequence) are race-safe.
        """
        s = get_settings()
        content_id = fingerprint.content_hash

        elected = redis_conn.set(_key_registered(content_id), "1", nx=True)
        if not elected:
            existing = self.get(content_id)
            if existing is not None:
                return existing
            # Lost election but row missing — race window between SETNX and HSET.
            # Brief retry; otherwise fall through and write (idempotent fields).
            log.info("content_register_lost_election_no_row", extra={"content_id": content_id})

        now = time.time()
        record = ContentRecord(
            content_id=content_id,
            canonical_phash=fingerprint.canonical_phash,
            keyframe_phashes=list(fingerprint.keyframe_phashes),
            keyframe_count=len(fingerprint.keyframe_phashes),
            owner=owner,
            trust_level=trust_level or "unknown",
            duration_seconds=fingerprint.duration_seconds,
            sample_fps=fingerprint.sample_fps,
            model_version=fingerprint.model_version or MODEL_VERSION,
            metadata=metadata or {},
            created_at=now,
            first_seen_job_id=job_id,
        )

        pipe = redis_conn.pipeline()
        pipe.hset(_key_content(content_id), mapping=_record_to_hash(record))
        if record.keyframe_phashes:
            pipe.delete(_key_keyframes(content_id))
            pipe.rpush(_key_keyframes(content_id), *[f"{p:016x}" for p in record.keyframe_phashes])
        prefix = _phash_prefix(record.canonical_phash, s.LSH_PREFIX_BITS)
        pipe.sadd(_key_lsh(prefix, s.LSH_PREFIX_BITS), content_id)
        if owner:
            pipe.sadd(_key_owner(owner), content_id)
        pipe.execute()
        return record

    # ---- IDENTITY RESOLUTION (the core decision) -----------------------------

    def resolve_identity(
        self,
        fingerprint: Fingerprint,
        candidates: List[MatchCandidate],
        *,
        owner: Optional[str] = None,
        trust_level: str = "unknown",
        job_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ResolvedIdentity:
        """Decide EXACT / VARIANT / NOVEL.

        EXACT   — best candidate within EXACT_THRESHOLD. Reuse its content_id.
                  Owner-promotion: if ingest carries verified owner and existing
                  record is unowned, attach the owner (single-direction; never
                  silently overwrites a known owner).
        VARIANT — best within VARIANT_THRESHOLD. Mint NEW content_id, return
                  parent for graph-edge attachment.
        NOVEL   — no candidate or distance > VARIANT_THRESHOLD. Mint NEW.
        """
        s = get_settings()
        best = candidates[0] if candidates else None

        if best and best.hamming_distance <= s.EXACT_THRESHOLD:
            self._maybe_promote_owner(best.content_id, owner, trust_level)
            existing = self.get(best.content_id)
            return ResolvedIdentity(
                content_id=best.content_id,
                role="EXACT",
                parent_content_id=None,
                similarity=best.similarity,
                hamming_distance=best.hamming_distance,
                is_new_content=False,
                owner=existing.owner if existing else best.owner,
                trust_level=existing.trust_level if existing else best.trust_level,
            )

        # Mint a new content row.
        record = self.register(
            fingerprint,
            owner=owner,
            trust_level=trust_level,
            job_id=job_id,
            metadata=metadata,
        )

        if best and best.hamming_distance <= s.VARIANT_THRESHOLD:
            return ResolvedIdentity(
                content_id=record.content_id,
                role="VARIANT",
                parent_content_id=best.content_id,
                similarity=best.similarity,
                hamming_distance=best.hamming_distance,
                is_new_content=record.first_seen_job_id == job_id,
                owner=record.owner,
                trust_level=record.trust_level,
            )

        return ResolvedIdentity(
            content_id=record.content_id,
            role="NOVEL",
            parent_content_id=None,
            similarity=best.similarity if best else 0.0,
            hamming_distance=best.hamming_distance if best else 64.0,
            is_new_content=record.first_seen_job_id == job_id,
            owner=record.owner,
            trust_level=record.trust_level,
        )

    # ---- STATS ---------------------------------------------------------------

    def increment_observation_stats(
        self,
        content_id: str,
        platform: str,
        band: Optional[str] = None,
        observed_at: Optional[float] = None,
    ) -> None:
        ts = observed_at or time.time()
        pipe = redis_conn.pipeline()
        pipe.hincrby(_key_stats(content_id), "observations_total", 1)
        pipe.hsetnx(_key_stats(content_id), "first_seen_at", repr(ts))
        pipe.hset(_key_stats(content_id), "last_seen_at", repr(ts))
        if band:
            pipe.hincrby(_key_stats(content_id), f"band_{band.lower()}", 1)
        if platform:
            pipe.sadd(_key_platforms(content_id), platform)
        pipe.execute()

    # ---- Internals -----------------------------------------------------------

    def _maybe_promote_owner(
        self,
        content_id: str,
        owner: Optional[str],
        trust_level: str,
    ) -> None:
        if not owner:
            return
        existing_owner = redis_conn.hget(_key_content(content_id), "owner")
        existing_owner = _decode(existing_owner) or ""
        if existing_owner:
            return
        redis_conn.hset(
            _key_content(content_id),
            mapping={"owner": owner, "trust_level": trust_level or "unknown"},
        )
        redis_conn.sadd(_key_owner(owner), content_id)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _record_to_hash(r: ContentRecord) -> Dict[str, str]:
    return {
        "content_id": r.content_id,
        "canonical_phash": f"{r.canonical_phash:016x}",
        "keyframe_count": str(r.keyframe_count),
        "owner": r.owner or "",
        "trust_level": r.trust_level,
        "duration_seconds": repr(r.duration_seconds),
        "sample_fps": repr(r.sample_fps),
        "model_version": r.model_version,
        "metadata": json.dumps(r.metadata or {}),
        "created_at": repr(r.created_at),
        "first_seen_job_id": r.first_seen_job_id or "",
    }


def _hash_to_record(raw: Dict[Any, Any], keyframe_phashes: List[int]) -> ContentRecord:
    d = {_decode(k): _decode(v) for k, v in raw.items()}
    return ContentRecord(
        content_id=d["content_id"],
        canonical_phash=int(d["canonical_phash"], 16),
        keyframe_phashes=keyframe_phashes,
        keyframe_count=int(d.get("keyframe_count", "0") or 0),
        owner=(d.get("owner") or None),
        trust_level=d.get("trust_level", "unknown") or "unknown",
        duration_seconds=float(d.get("duration_seconds", "0") or 0),
        sample_fps=float(d.get("sample_fps", "0") or 0),
        model_version=d.get("model_version", MODEL_VERSION),
        metadata=json.loads(d.get("metadata") or "{}"),
        created_at=float(d.get("created_at", "0") or 0),
        first_seen_job_id=(d.get("first_seen_job_id") or None),
    )


def _coerce_stat(v: Optional[str]) -> Any:
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


# Module-level singleton
content_registry = ContentRegistry()
