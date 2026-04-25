"""Fingerprint stage.

Stub implementation: a deterministic SHA-256 over a canonical representation
of the input. Production replaces this with perceptual hashing (pHash, dHash,
video keyframe hashes) computed from raw media bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_fingerprint(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
