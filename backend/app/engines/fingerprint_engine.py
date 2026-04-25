"""Perceptual fingerprinting — keyframes + pHash sequence.

Replaces the SHA-256 stub. Outputs:

  * content_hash       sha256 over the canonicalised pHash sequence (deterministic
                       identity — byte-identical media always produces the same id)
  * canonical_phash    representative 64-bit pHash (median) for LSH bucketing
  * keyframe_phashes   ordered list of per-keyframe 64-bit pHashes for sliding-window
                       sequence matching against the registry
  * timestamps         seconds per keyframe (for evidence/UX)
  * duration_seconds   asset duration

Two execution modes:

  REAL   — payload supplies `local_path` to a video readable by OpenCV; we
           sample at KEYFRAME_INTERVAL_SEC, downscale, and pHash.
  SYNTH  — payload has no usable file (demo / no-binary ingest); we
           deterministically derive a pHash sequence from the payload so two
           identical metadata payloads collapse to the same content_id.

SYNTH mode is required because the existing demo flow ingests via JSON without
a local video. Production replaces this with a streaming download stage in the
worker before fingerprinting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

import numpy as np

from app.core.config import get_settings

log = logging.getLogger(__name__)

MODEL_VERSION = "phash-8x8-v1"


@dataclass
class Fingerprint:
    content_hash: str                            # sha256 of canonical phash sequence
    canonical_phash: int                         # representative pHash (median)
    keyframe_phashes: List[int] = field(default_factory=list)
    keyframe_timestamps: List[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    sample_fps: float = 0.0
    source_mode: str = "synth"                   # "real" | "synth"
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class FingerprintError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def compute_fingerprint(payload: Any) -> Fingerprint:
    """Pipeline entrypoint.

    Accepts whatever the worker passes through (typically the IngestRequest
    payload as a dict). Decides REAL vs SYNTH internally; never throws on
    missing files — falls back to SYNTH so the pipeline always produces a
    fingerprint.
    """
    payload_dict = _coerce_dict(payload)
    local_path = payload_dict.get("local_path")

    if local_path and os.path.exists(local_path):
        try:
            return _fingerprint_real(local_path)
        except FingerprintError as e:
            log.warning("fingerprint_real_failed_falling_back_synth", extra={"reason": str(e)})

    return _fingerprint_synth(payload_dict)


# ---------------------------------------------------------------------------
# REAL — OpenCV + imagehash
# ---------------------------------------------------------------------------

def _fingerprint_real(path: str) -> Fingerprint:
    # Imports kept local so SYNTH path works even if cv2 is unavailable.
    import cv2
    import imagehash
    from PIL import Image

    s = get_settings()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FingerprintError(f"cannot open: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0 or fps <= 0:
            raise FingerprintError("invalid fps/frame_count")

        duration = total / fps
        target_count = max(1, min(int(duration / s.KEYFRAME_INTERVAL_SEC), s.MAX_KEYFRAMES))
        stride = max(1, total // target_count)

        phashes: List[int] = []
        timestamps: List[float] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                phashes.append(_phash_frame(frame, s.PHASH_SIZE, s.DOWNSCALE_LONG_EDGE))
                timestamps.append(idx / fps)
                if len(phashes) >= s.MAX_KEYFRAMES:
                    break
            idx += 1

        if not phashes:
            raise FingerprintError("no frames decoded")

        canonical_phash = int(np.median(phashes))
        canonical_bytes = b"".join(p.to_bytes(8, "big") for p in phashes)
        content_hash = hashlib.sha256(canonical_bytes).hexdigest()

        return Fingerprint(
            content_hash=content_hash,
            canonical_phash=canonical_phash,
            keyframe_phashes=phashes,
            keyframe_timestamps=timestamps,
            duration_seconds=duration,
            sample_fps=fps,
            source_mode="real",
        )
    finally:
        cap.release()


def _phash_frame(frame_bgr, hash_size: int, long_edge: int) -> int:
    import cv2
    import imagehash
    from PIL import Image

    h, w = frame_bgr.shape[:2]
    if max(h, w) > long_edge:
        scale = long_edge / max(h, w)
        frame_bgr = cv2.resize(
            frame_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return int(str(imagehash.phash(pil, hash_size=hash_size)), 16)


# ---------------------------------------------------------------------------
# SYNTH — deterministic from payload (demo / no-binary path)
# ---------------------------------------------------------------------------

def _fingerprint_synth(payload: dict) -> Fingerprint:
    """Deterministic pHash sequence derived from a canonical JSON of the payload.

    Property: identical payloads → identical fingerprints (so identity
    resolution sees them as EXACT). Slight payload mutations (e.g.
    different metadata noise) produce different sequences — they will look
    NOVEL, which is honest behaviour given we have no perceptual signal.
    """
    canonical = json.dumps(_normalise_for_synth(payload), sort_keys=True, default=str)
    seed_bytes = hashlib.sha256(canonical.encode("utf-8")).digest()

    # Deterministic 30-frame sequence (≈30s clip @ 1fps), 64-bit pHashes.
    # Use BLAKE2 chaining for cheap, deterministic expansion.
    rng = hashlib.blake2b(seed_bytes, digest_size=8 * 30)
    raw = rng.digest()
    phashes: List[int] = [
        struct.unpack(">Q", raw[i : i + 8])[0] for i in range(0, len(raw), 8)
    ]
    timestamps = [float(i) for i in range(len(phashes))]

    canonical_phash = int(np.median(phashes))
    content_hash = hashlib.sha256(
        b"".join(p.to_bytes(8, "big") for p in phashes)
    ).hexdigest()

    return Fingerprint(
        content_hash=content_hash,
        canonical_phash=canonical_phash,
        keyframe_phashes=phashes,
        keyframe_timestamps=timestamps,
        duration_seconds=float(len(phashes)),
        sample_fps=1.0,
        source_mode="synth",
    )


def _normalise_for_synth(payload: dict) -> dict:
    # Strip volatile fields so re-ingests of the "same" demo asset collapse.
    drop = {"local_path", "metadata"}
    return {k: v for k, v in payload.items() if k not in drop}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_dict(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray, str)):
        return {"raw": payload if isinstance(payload, str) else payload.decode("latin-1")}
    return {"value": str(payload)}


# ---------------------------------------------------------------------------
# Sequence-distance primitives (used by matching engine)
# ---------------------------------------------------------------------------

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sequence_distance(query: List[int], reference: List[int]) -> float:
    """Best-window mean Hamming distance.

    Sliding-window alignment makes matching robust to clipping, prepended/
    appended frames, and small re-encode boundary effects. Cost:
    O(min(n,m) * (max(n,m) - min(n,m) + 1)). LSH pre-filter at the
    matching engine bounds the reference-pool size, so this stays cheap.
    """
    if not query or not reference:
        return 64.0
    short, long_ = (query, reference) if len(query) <= len(reference) else (reference, query)
    best = 64.0
    span = len(long_) - len(short)
    for offset in range(span + 1):
        s = 0
        for i in range(len(short)):
            s += (short[i] ^ long_[offset + i]).bit_count()
        avg = s / len(short)
        if avg < best:
            best = avg
            if best == 0.0:
                break
    return best
