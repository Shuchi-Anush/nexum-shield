"""Embedding stage.

Stub implementation: deterministic 32-dimensional unit vector derived from
the fingerprint bytes. Identical fingerprints always yield identical vectors.
Production replaces this with a real encoder (CLIP, SigLIP, Whisper, etc.).
"""

from __future__ import annotations

import math


MODEL_VERSION = "embed-stub-v0.1"
EMBEDDING_DIM = 32


def embed(fingerprint: str) -> list[float]:
    raw = bytes.fromhex(fingerprint)[:EMBEDDING_DIM]
    if len(raw) < EMBEDDING_DIM:
        raw = raw + b"\x00" * (EMBEDDING_DIM - len(raw))
    centred = [(b - 127.5) / 127.5 for b in raw]
    norm = math.sqrt(sum(x * x for x in centred))
    if norm == 0.0:
        return [0.0] * EMBEDDING_DIM
    return [x / norm for x in centred]
