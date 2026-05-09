"""RQ worker entrypoint.

Start with:
    uv run python -m app.workers.worker

Consumes jobs from the "pipeline" queue and executes
`app.workers.pipeline_worker.run_pipeline`.

Windows → SimpleWorker (no fork)
Linux   → Worker (fork, parallelism)

C-2: at boot, sweep the JobStore for jobs stranded in PROCESSING by
a previous worker crash and FAIL them so they don't sit invisibly
forever. The reaper is conservative — it never touches a job whose
distributed lock is still alive.
"""

from __future__ import annotations

import logging
import os
import platform

from app.core.queue import pipeline_queue, redis_conn
from app.workers.orphan_reaper import reap_orphans

_logger = logging.getLogger(__name__)


def _select_worker_class():
    # Windows has no fork → must use SimpleWorker
    if platform.system() == "Windows":
        from rq import SimpleWorker

        return SimpleWorker

    # Linux / macOS → full worker
    from rq import Worker

    return Worker


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format='{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}',
    )

    # C-2: reap orphans before starting to consume. Failures here are
    # non-fatal — startup proceeds even if reaping cannot reach Redis,
    # because the worker's first ping will surface the same error.
    try:
        reaped = reap_orphans()
        _logger.info("worker boot: reaped %d orphan job(s)", len(reaped))
    except Exception as exc:
        _logger.warning(
            "worker boot: orphan reaper failed (continuing): %s", exc
        )

    WorkerClass = _select_worker_class()
    worker = WorkerClass([pipeline_queue], connection=redis_conn)
    _logger.info(
        "worker starting", extra={
            "worker_class": WorkerClass.__name__,
            "platform": platform.system(),
        },
    )
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()