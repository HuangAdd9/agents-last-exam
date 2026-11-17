"""Resizable concurrency limiter used by a running experiment."""

from __future__ import annotations

import asyncio


class ResizableLimiter:
    """A semaphore-like limiter whose capacity can change at runtime.

    Lowering the limit never interrupts holders; it only prevents new
    acquisitions until the number of active holders is below the new limit.
    Raising it wakes queued acquisitions immediately.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("concurrency must be at least 1")
        self._limit = limit
        self._active = 0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active < self._limit)
            self._active += 1

    def release(self) -> None:
        # release is called by the event-loop task that acquired the slot.
        if self._active <= 0:
            raise RuntimeError("release called without a matching acquire")
        self._active -= 1
        # Condition.notify_all() requires the lock. Schedule a tiny coroutine
        # instead of making every lifecycle finally block async-only.
        async def notify() -> None:
            async with self._condition:
                self._condition.notify_all()
        asyncio.create_task(notify())

    async def set_limit(self, limit: int) -> bool:
        if limit < 1:
            raise ValueError("concurrency must be at least 1")
        async with self._condition:
            changed = limit != self._limit
            self._limit = limit
            self._condition.notify_all()
            return changed
