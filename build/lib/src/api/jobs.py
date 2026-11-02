"""Выполнение фоновых задач в потоках с ограничением параллелизма и длины очереди."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from src.api.errors import ApiError


class QueueFull(ApiError):
    status_code = 429


class JobRunner:
    """inline=True выполняет задачу сразу в вызывающем потоке (для тестов: результат детерминирован)."""

    def __init__(self, workers: int = 2, max_pending: int = 20, inline: bool = False):
        self.inline = inline
        self.max_pending = max_pending
        self._pending = 0
        self._lock = threading.Lock()
        self._pool = None if inline else ThreadPoolExecutor(max_workers=workers, thread_name_prefix="validation")

    def submit(self, fn: Callable[[], None]) -> None:
        with self._lock:
            if self._pending >= self.max_pending:
                raise QueueFull("Слишком много запусков в очереди. Повторите позже.")
            self._pending += 1

        def wrapped():
            try:
                fn()
            finally:
                with self._lock:
                    self._pending -= 1

        if self.inline:
            wrapped()
        else:
            self._pool.submit(wrapped)

    @property
    def pending(self) -> int:
        return self._pending

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
