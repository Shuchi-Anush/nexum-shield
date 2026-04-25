"""ObservationStore — sightings of content in the wild.

Each observation is one (URL, content_id, moment) triple. The store is
idempotent on (source_url_hash, content_id): re-detecting the same content
at the same URL within OBSERVATION_DEDUP_TTL_SECONDS increments
`detection_count` and refreshes `last_seen_at` rather than creating a new
row.

Redis schema:

    observation:{obs_id}                       HASH    full row
    observation_dedup:{url_hash}:{cid}         STRING  obs_id  (TTL)
    observation_index:url:{url_hash}           SET<obs_id>
    platform:{p}:recent                        ZSET<obs_id>  capped, score=ts
    content:{cid}:observations                 ZSET<obs_id>  score=ts (timeline)
    observations:recent                        ZSET<obs_id>  global feed, capped

Postgres mirror (observations table) is the durable source of truth with
UNIQUE (source_url_hash, content_id). Write-through is best-effort here;
the Redis path is what the dashboard reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.config import get_settings
from app.core.queue import redis_conn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    observation_id: str
    content_id: str
    source_url: str
    source_url_hash: str
    platform: str
    observed_at: float
    last_seen_at: float
    detection_count: int
    detected_via: str                  # ingest_api | crawler | user_report
    similarity_score: float
    match_distance: float
    job_id: str
    status: str                        # ACTIVE | REMOVED | DISPUTED
    enforcement_action: Optional[str]  # ALLOW | FLAG | BLOCK | TAKEDOWN_SENT | None
    evidence_id: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "yclid", "mc_cid", "mc_eid", "_ga", "ref", "ref_src",
}


def normalize_url(url: str) -> str:
    """Best-effort canonical form for dedup. Platform-specific rules can be
    added when stable identifiers exist (e.g., youtube ?v=). For now: lower
    scheme/host, strip default ports + fragment, drop common tracking params,
    sort query, no trailing slash."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "http").lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/") or "/"
    qs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_PARAMS
    ]
    qs.sort()
    return urlunsplit((scheme, host, path, urlencode(qs), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


def _decode(v: Any) -> Optional[str]:
    if v is None:
        return None
    return v.decode("utf-8") if isinstance(v, bytes) else (v if isinstance(v, str) else str(v))


def _key_obs(oid: str) -> str:                           return f"observation:{oid}"
def _key_dedup(uhash: str, cid: str) -> str:             return f"observation_dedup:{uhash}:{cid}"
def _key_url_idx(uhash: str) -> str:                     return f"observation_index:url:{uhash}"
def _key_platform_recent(platform: str) -> str:          return f"platform:{platform}:recent"
def _key_content_timeline(cid: str) -> str:              return f"content:{cid}:observations"
_KEY_GLOBAL_RECENT = "observations:recent"


# ---------------------------------------------------------------------------
# ObservationStore
# ---------------------------------------------------------------------------

class ObservationStore:

    # ---- WRITE ---------------------------------------------------------------

    def record(
        self,
        *,
        content_id: str,
        source_url: Optional[str],
        platform: str,
        job_id: str,
        similarity_score: float,
        match_distance: float,
        detected_via: str = "ingest_api",
        metadata: Optional[dict] = None,
    ) -> Observation:
        """Idempotent write.

        If an observation already exists for (url, content_id) within the
        dedup TTL → increment detection_count, refresh last_seen_at, return
        the existing observation. Otherwise create a new row and index it.
        """
        s = get_settings()
        normalized_url = normalize_url(source_url or "")
        uhash = url_hash(source_url or "")
        now = time.time()

        # Dedup election (atomic): SETNX on the dedup key.
        new_obs_id = uuid.uuid4().hex
        elected = redis_conn.set(
            _key_dedup(uhash, content_id),
            new_obs_id,
            nx=True,
            ex=s.OBSERVATION_DEDUP_TTL_SECONDS,
        )

        if not elected:
            existing_id = _decode(redis_conn.get(_key_dedup(uhash, content_id)))
            if existing_id:
                self._increment_existing(existing_id, now, similarity_score, match_distance)
                obs = self.get(existing_id)
                if obs:
                    return obs
            # Fall through and create — dedup key vanished between SETNX and GET.

        observation = Observation(
            observation_id=new_obs_id,
            content_id=content_id,
            source_url=normalized_url,
            source_url_hash=uhash,
            platform=platform or "unknown",
            observed_at=now,
            last_seen_at=now,
            detection_count=1,
            detected_via=detected_via,
            similarity_score=float(similarity_score),
            match_distance=float(match_distance),
            job_id=job_id,
            status="ACTIVE",
            enforcement_action=None,
            evidence_id=None,
            metadata=metadata or {},
        )

        pipe = redis_conn.pipeline()
        pipe.hset(_key_obs(new_obs_id), mapping=_obs_to_hash(observation))
        if s.OBSERVATION_TTL_SECONDS > 0:
            pipe.expire(_key_obs(new_obs_id), s.OBSERVATION_TTL_SECONDS)
        pipe.zadd(_key_content_timeline(content_id), {new_obs_id: now})
        pipe.zadd(_key_platform_recent(observation.platform), {new_obs_id: now})
        pipe.zremrangebyrank(
            _key_platform_recent(observation.platform),
            0, -(s.PLATFORM_RECENT_CAP + 1),
        )
        pipe.zadd(_KEY_GLOBAL_RECENT, {new_obs_id: now})
        pipe.zremrangebyrank(_KEY_GLOBAL_RECENT, 0, -(s.PLATFORM_RECENT_CAP + 1))
        pipe.sadd(_key_url_idx(uhash), new_obs_id)
        pipe.execute()
        return observation

    def set_enforcement_action(
        self,
        observation_id: str,
        action: str,
        evidence_id: Optional[str] = None,
    ) -> None:
        if not redis_conn.exists(_key_obs(observation_id)):
            raise ValueError("Observation not found")
        mapping = {"enforcement_action": action, "last_seen_at": repr(time.time())}
        if evidence_id:
            mapping["evidence_id"] = evidence_id
        redis_conn.hset(_key_obs(observation_id), mapping=mapping)

    def set_status(self, observation_id: str, status: str) -> None:
        if status not in {"ACTIVE", "REMOVED", "DISPUTED"}:
            raise ValueError(f"invalid status: {status}")
        if not redis_conn.exists(_key_obs(observation_id)):
            raise ValueError("Observation not found")
        redis_conn.hset(_key_obs(observation_id), mapping={"status": status})

    # ---- READ ----------------------------------------------------------------

    def get(self, observation_id: str) -> Optional[Observation]:
        raw = redis_conn.hgetall(_key_obs(observation_id))
        if not raw:
            return None
        return _hash_to_obs(raw)

    def list_for_content(
        self,
        content_id: str,
        *,
        limit: int = 50,
        before_ts: Optional[float] = None,
    ) -> List[Observation]:
        upper = f"({before_ts}" if before_ts else "+inf"
        ids = redis_conn.zrevrangebyscore(
            _key_content_timeline(content_id),
            upper, "-inf", start=0, num=limit,
        )
        return self._mget(ids)

    def list_recent(
        self,
        *,
        platform: Optional[str] = None,
        limit: int = 50,
    ) -> List[Observation]:
        key = _key_platform_recent(platform) if platform else _KEY_GLOBAL_RECENT
        ids = redis_conn.zrevrange(key, 0, limit - 1)
        return self._mget(ids)

    def count_for_content(self, content_id: str) -> int:
        return int(redis_conn.zcard(_key_content_timeline(content_id)) or 0)

    # ---- Internals -----------------------------------------------------------

    def _increment_existing(
        self,
        observation_id: str,
        now: float,
        similarity_score: float,
        match_distance: float,
    ) -> None:
        pipe = redis_conn.pipeline()
        pipe.hincrby(_key_obs(observation_id), "detection_count", 1)
        pipe.hset(
            _key_obs(observation_id),
            mapping={
                "last_seen_at": repr(now),
                "similarity_score": repr(similarity_score),
                "match_distance": repr(match_distance),
            },
        )
        pipe.execute()

    def _mget(self, ids: List[Any]) -> List[Observation]:
        if not ids:
            return []
        pipe = redis_conn.pipeline()
        for oid in ids:
            pipe.hgetall(_key_obs(_decode(oid)))
        results = pipe.execute()
        out: List[Observation] = []
        for raw in results:
            if raw:
                out.append(_hash_to_obs(raw))
        return out


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _obs_to_hash(o: Observation) -> Dict[str, str]:
    return {
        "observation_id": o.observation_id,
        "content_id": o.content_id,
        "source_url": o.source_url,
        "source_url_hash": o.source_url_hash,
        "platform": o.platform,
        "observed_at": repr(o.observed_at),
        "last_seen_at": repr(o.last_seen_at),
        "detection_count": str(o.detection_count),
        "detected_via": o.detected_via,
        "similarity_score": repr(o.similarity_score),
        "match_distance": repr(o.match_distance),
        "job_id": o.job_id,
        "status": o.status,
        "enforcement_action": o.enforcement_action or "",
        "evidence_id": o.evidence_id or "",
        "metadata": json.dumps(o.metadata or {}),
    }


def _hash_to_obs(raw: Dict[Any, Any]) -> Observation:
    d = {_decode(k): _decode(v) for k, v in raw.items()}
    return Observation(
        observation_id=d["observation_id"],
        content_id=d["content_id"],
        source_url=d.get("source_url", "") or "",
        source_url_hash=d.get("source_url_hash", "") or "",
        platform=d.get("platform", "unknown") or "unknown",
        observed_at=float(d.get("observed_at", "0") or 0),
        last_seen_at=float(d.get("last_seen_at", "0") or 0),
        detection_count=int(d.get("detection_count", "1") or 1),
        detected_via=d.get("detected_via", "ingest_api") or "ingest_api",
        similarity_score=float(d.get("similarity_score", "0") or 0),
        match_distance=float(d.get("match_distance", "64") or 64),
        job_id=d.get("job_id", "") or "",
        status=d.get("status", "ACTIVE") or "ACTIVE",
        enforcement_action=(d.get("enforcement_action") or None),
        evidence_id=(d.get("evidence_id") or None),
        metadata=json.loads(d.get("metadata") or "{}"),
    )


# Module-level singleton
observation_store = ObservationStore()
