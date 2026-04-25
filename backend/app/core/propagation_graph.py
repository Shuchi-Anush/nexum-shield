"""PropagationGraph — directed parent→child variant edges.

A child node is a content_id that's a derivative of its parent (re-encode,
crop, watermark add/remove, etc.). The graph forms a DAG by construction —
edge insertion checks the prospective parent's ancestor set and rejects
cycles. (Best-effort: under heavy concurrent insertion the check could race;
production would back this with a Postgres edge table + transactional
constraint.)

Redis schema:

    graph:children:{cid}                SET<cid>
    graph:parents:{cid}                 SET<cid>
    graph:edge:{parent}:{child}         HASH   (similarity, transformation, …)
    graph:origin_cache:{cid}            STRING memoised root content_id

BFS is bounded (depth + node cap from settings) so a viral origin can't
fan out into a multi-thousand-node response.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import get_settings
from app.core.queue import redis_conn

log = logging.getLogger(__name__)


VALID_TRANSFORMATIONS = {
    "CROP", "RESIZE", "RE_ENCODE", "WATERMARK_ADD", "WATERMARK_REMOVE",
    "COLOR_SHIFT", "SPEED_CHANGE", "FRAME_REORDER", "UNKNOWN",
}


@dataclass
class Edge:
    parent_id: str
    child_id: str
    similarity: float
    hamming_distance: float
    transformation: str
    detected_at: float
    evidence_id: Optional[str]
    observation_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _decode(v: Any) -> Optional[str]:
    if v is None:
        return None
    return v.decode("utf-8") if isinstance(v, bytes) else (v if isinstance(v, str) else str(v))


def _key_children(cid: str) -> str:                  return f"graph:children:{cid}"
def _key_parents(cid: str) -> str:                   return f"graph:parents:{cid}"
def _key_edge(p: str, c: str) -> str:                return f"graph:edge:{p}:{c}"
def _key_origin_cache(cid: str) -> str:              return f"graph:origin_cache:{cid}"


# ---------------------------------------------------------------------------
# PropagationGraph
# ---------------------------------------------------------------------------

class PropagationGraph:

    # ---- WRITE ---------------------------------------------------------------

    def attach_edge(
        self,
        *,
        parent_id: str,
        child_id: str,
        similarity: float,
        hamming_distance: float,
        transformation: str = "UNKNOWN",
        evidence_id: Optional[str] = None,
    ) -> Optional[Edge]:
        """Attach (or upsert) a parent→child edge.

        Idempotent: re-attaching the same edge increments observation_count
        and refreshes timestamps; never duplicates. Returns None if the
        proposed edge would create a cycle.
        """
        if parent_id == child_id:
            return None
        if transformation not in VALID_TRANSFORMATIONS:
            transformation = "UNKNOWN"

        if self._would_cycle(parent_id, child_id):
            log.warning(
                "graph_cycle_rejected",
                extra={"parent_id": parent_id, "child_id": child_id},
            )
            return None

        now = time.time()
        existing_count = redis_conn.hget(_key_edge(parent_id, child_id), "observation_count")
        already = existing_count is not None

        pipe = redis_conn.pipeline()
        pipe.sadd(_key_children(parent_id), child_id)
        pipe.sadd(_key_parents(child_id), parent_id)
        if already:
            pipe.hincrby(_key_edge(parent_id, child_id), "observation_count", 1)
            pipe.hset(
                _key_edge(parent_id, child_id),
                mapping={
                    "last_detected_at": repr(now),
                    "similarity": repr(similarity),
                    "hamming_distance": repr(hamming_distance),
                    "transformation": transformation,
                    **({"evidence_id": evidence_id} if evidence_id else {}),
                },
            )
        else:
            pipe.hset(
                _key_edge(parent_id, child_id),
                mapping={
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "similarity": repr(similarity),
                    "hamming_distance": repr(hamming_distance),
                    "transformation": transformation,
                    "detected_at": repr(now),
                    "last_detected_at": repr(now),
                    "evidence_id": evidence_id or "",
                    "observation_count": "1",
                },
            )
        # Adding an edge invalidates the cached origin of the child subtree.
        pipe.delete(_key_origin_cache(child_id))
        pipe.execute()

        return self.get_edge(parent_id, child_id)

    # ---- READ ----------------------------------------------------------------

    def get_edge(self, parent_id: str, child_id: str) -> Optional[Edge]:
        raw = redis_conn.hgetall(_key_edge(parent_id, child_id))
        if not raw:
            return None
        d = {_decode(k): _decode(v) for k, v in raw.items()}
        return Edge(
            parent_id=d.get("parent_id", parent_id),
            child_id=d.get("child_id", child_id),
            similarity=float(d.get("similarity", "0") or 0),
            hamming_distance=float(d.get("hamming_distance", "0") or 0),
            transformation=d.get("transformation", "UNKNOWN"),
            detected_at=float(d.get("detected_at", "0") or 0),
            evidence_id=(d.get("evidence_id") or None),
            observation_count=int(d.get("observation_count", "1") or 1),
        )

    def children_of(self, content_id: str) -> List[str]:
        return sorted([_decode(x) for x in redis_conn.smembers(_key_children(content_id))])

    def parents_of(self, content_id: str) -> List[str]:
        return sorted([_decode(x) for x in redis_conn.smembers(_key_parents(content_id))])

    def origin_of(self, content_id: str) -> str:
        """Return the ultimate root (no parents) reachable upward.

        Memoised in Redis; cache invalidated on any edge insertion below the
        node.  If no parent → self is origin.
        """
        cached = _decode(redis_conn.get(_key_origin_cache(content_id)))
        if cached:
            return cached
        cur = content_id
        seen: Set[str] = set()
        while True:
            parents = self.parents_of(cur)
            if not parents:
                break
            if cur in seen:                # cycle defence
                log.warning("graph_origin_cycle_break", extra={"start": content_id, "at": cur})
                break
            seen.add(cur)
            cur = parents[0]               # prefer first; ties broken alphabetically
        redis_conn.set(_key_origin_cache(content_id), cur, ex=3600)
        return cur

    def bfs(
        self,
        root_id: str,
        *,
        depth: Optional[int] = None,
        direction: str = "both",            # "down" | "up" | "both"
        max_nodes: Optional[int] = None,
    ) -> Tuple[List[str], List[Edge], bool]:
        """Bounded BFS. Returns (nodes, edges, truncated)."""
        s = get_settings()
        depth = s.GRAPH_DEFAULT_DEPTH if depth is None else max(0, depth)
        max_nodes = s.GRAPH_MAX_NODES if max_nodes is None else max_nodes
        truncated = False

        nodes: List[str] = [root_id]
        node_set: Set[str] = {root_id}
        edges: List[Edge] = []
        edge_seen: Set[Tuple[str, str]] = set()

        frontier = [root_id]
        for _ in range(depth):
            next_frontier: List[str] = []
            for cid in frontier:
                neighbours: List[Tuple[str, str]] = []  # (parent, child)
                if direction in ("down", "both"):
                    for child in self.children_of(cid):
                        neighbours.append((cid, child))
                if direction in ("up", "both"):
                    for parent in self.parents_of(cid):
                        neighbours.append((parent, cid))
                for (p, c) in neighbours:
                    if (p, c) in edge_seen:
                        continue
                    edge_seen.add((p, c))
                    edge = self.get_edge(p, c)
                    if edge:
                        edges.append(edge)
                    for nb in (p, c):
                        if nb not in node_set:
                            if len(nodes) >= max_nodes:
                                truncated = True
                                continue
                            node_set.add(nb)
                            nodes.append(nb)
                            next_frontier.append(nb)
            if not next_frontier or truncated:
                break
            frontier = next_frontier
        return nodes, edges, truncated

    # ---- Internals -----------------------------------------------------------

    def _would_cycle(self, parent_id: str, child_id: str) -> bool:
        """Rejects edges where `child_id` is already an ancestor of `parent_id`.

        Best-effort check — concurrent inserts could race past this.
        Production-grade: enforce with a Postgres CHECK or recursive CTE.
        """
        # Walk UP from parent; if we encounter child → cycle.
        seen: Set[str] = set()
        frontier = [parent_id]
        while frontier:
            cur = frontier.pop()
            if cur == child_id:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            frontier.extend(self.parents_of(cur))
            if len(seen) > 10_000:
                # Safety bound; treat as "would cycle" to refuse runaway graphs.
                return True
        return False


# Module-level singleton
propagation_graph = PropagationGraph()
