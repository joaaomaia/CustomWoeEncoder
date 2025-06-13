from __future__ import annotations

import time
import logging
from contextlib import contextmanager

try:
    import psutil
except Exception:  # pragma: no cover - fallback if psutil not available
    psutil = None

class MemoryManager:
    """Monitor available RAM and CPU usage."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def free_ram(self) -> int:
        if psutil:
            return int(psutil.virtual_memory().available)
        try:
            import resource
            return int(resource.getrlimit(resource.RLIMIT_AS)[1])
        except Exception:
            return 0

    def memory_ok(self, predicted_size: int) -> bool:
        return predicted_size < 0.5 * self.free_ram()

    @contextmanager
    def profile(self, op_name: str):
        start = time.time()
        mem_before = self.free_ram()
        try:
            yield
        finally:
            dur = time.time() - start
            mem_after = self.free_ram()
            self.logger.info(
                "%s finished in %.3fs (Δmem=%d bytes)",
                op_name,
                dur,
                mem_before - mem_after,
            )
