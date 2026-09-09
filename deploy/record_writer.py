"""Bounded serialized recording detached from the ASGI request lifetime."""
from collections import deque
import logging
import threading
import time


logger = logging.getLogger(__name__)


class RecordWriter:
    def __init__(self, capacity=16):
        if capacity < 1:
            raise ValueError("record queue capacity must be positive")
        self.capacity = capacity
        self._queue = deque()
        self._condition = threading.Condition()
        self._thread = None
        self._error = None
        self._closing = False

    def status(self):
        with self._condition:
            return {"recording_error": self._error,
                    "recording_queue_depth": len(self._queue),
                    "recording_queue_capacity": self.capacity}

    def submit(self, operation, **kwargs):
        with self._condition:
            if self._closing or self._error is not None:
                raise RuntimeError(self._error or "record writer is shutting down")
            if len(self._queue) >= self.capacity:
                self._error = f"recording backlog exceeded capacity {self.capacity}; inference stopped"
                raise RuntimeError(self._error)
            self._queue.append((operation, kwargs))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="arx-record-writer", daemon=False)
                self._thread.start()

    def _run(self):
        while True:
            with self._condition:
                if not self._queue:
                    # No idle worker remains, including in ASGI tests that omit lifespan.
                    self._thread = None
                    self._condition.notify_all()
                    return
                operation, kwargs = self._queue.popleft()
            try:
                operation(**kwargs)
            except Exception as error:
                with self._condition:
                    if self._error is None:
                        self._error = f"{type(error).__name__}: {error}"
                logger.exception("ARX recording failed for %s", kwargs.get("target", "request"))
                # Already accepted records are still attempted, never discarded silently.

    def wait(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._thread is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout=30.0):
        with self._condition:
            self._closing = True
            worker = self._thread
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                with self._condition:
                    self._error = f"record writer did not drain within {timeout}s"
                raise TimeoutError(self._error)
