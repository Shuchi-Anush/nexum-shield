"""RQ worker entrypoint.

Start with:
    uv run python -m app.workers.worker

Consumes jobs from the "pipeline" queue and executes
`app.workers.pipeline_worker.run_pipeline`.

Windows → SimpleWorker (no fork)
Linux   → Worker (fork, parallelism)
"""

from __future__ import annotations

import os
import platform

from app.core.queue import pipeline_queue, redis_conn


def _select_worker_class():
    # Windows has no fork → must use SimpleWorker
    if platform.system() == "Windows":
        from rq import SimpleWorker

        return SimpleWorker

    # Linux / macOS → full worker
    from rq import Worker

    return Worker


def main() -> None:
    WorkerClass = _select_worker_class()

    worker = WorkerClass([pipeline_queue], connection=redis_conn)

    print(f"[worker] Starting {WorkerClass.__name__} on {platform.system()}")

    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()