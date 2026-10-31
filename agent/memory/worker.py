"""Single-writer worker thread with a FIFO job queue (store/promote/bootstrap)."""

from __future__ import annotations

import datetime
import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from agent.memory.memory_config import MemoryConfig
from agent.memory.models import JobReceipt

logger = logging.getLogger(__name__)

STOP_KIND = "_stop"


@dataclass
class _Job:
    job_id: int
    kind: str
    payload: dict
    receipt: JobReceipt | None


class MemoryWorker:
    """Owns a FIFO queue and one daemon thread; the thread is the only writer.

    Readers (``invoke``/``semantic_search``) take :attr:`lock` for synchronous
    access. A job exception is logged and dropped; the worker never dies and
    never raises into the caller.
    """

    def __init__(self, config: MemoryConfig, handlers: dict[str, Callable[[dict], object]]):
        self._config = config
        self._handlers = dict(handlers)
        self._queue: queue.Queue[_Job] = queue.Queue()
        self.lock = threading.RLock()
        self._ids = itertools.count(1)
        self._warn_depth = config.queue_warn_depth
        self._coalesce = config.promote_coalesce
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="memory-worker", daemon=True)
        self._thread.start()

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, kind: str, payload: dict | None = None) -> JobReceipt:
        if self._stopped:
            raise RuntimeError("memory worker is shut down")
        if kind not in self._handlers:
            raise ValueError(f"unknown memory job kind: {kind!r}")
        payload = dict(payload or {})
        if kind == "promote" and self._coalesce and not payload.get("force"):
            with self._queue.mutex:
                for pending in self._queue.queue:
                    if pending.kind == "promote":
                        assert pending.receipt is not None
                        return pending.receipt
        receipt = JobReceipt(
            job_id=next(self._ids), kind=kind, enqueued_at=datetime.datetime.now()
        )
        job = _Job(job_id=receipt.job_id, kind=kind, payload=payload, receipt=receipt)
        if kind == "bootstrap":
            # Priority insert ahead of queued stores (same locking as Queue.put).
            with self._queue.not_empty:
                self._queue.queue.appendleft(job)
                self._queue.unfinished_tasks += 1
                self._queue.not_empty.notify()
        else:
            self._queue.put(job)
        depth = self._queue.qsize()
        if depth > self._warn_depth:
            logger.warning(
                "memory worker queue depth %d exceeds warn depth %d",
                depth,
                self._warn_depth,
            )
        return receipt

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job.kind == STOP_KIND:
                    return
                handler = self._handlers.get(job.kind)
                if handler is None or job.receipt is None:
                    continue
                with self.lock:
                    try:
                        result = handler(job.payload)
                    except Exception as exc:  # noqa: BLE001 - failure isolation
                        logger.exception(
                            "memory worker job %d (%s) failed; dropping it",
                            job.job_id,
                            job.kind,
                        )
                        job.receipt.mark_done(error=exc)
                    else:
                        job.receipt.mark_done(result=result)
            finally:
                self._queue.task_done()

    def drain(self, timeout: float | None = None) -> None:
        """Block until the queue is empty; ``TimeoutError`` on timeout."""
        if timeout is None:
            self._queue.join()
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("memory worker drain timed out")
            time.sleep(0.005)

    def shutdown(self, timeout: float | None = None) -> None:
        """Drain, then stop the thread. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self.drain(timeout)
        self._queue.put(_Job(job_id=0, kind=STOP_KIND, payload={}, receipt=None))
        self._thread.join(timeout)
