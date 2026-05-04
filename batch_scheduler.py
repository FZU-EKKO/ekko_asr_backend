from __future__ import annotations

import asyncio

from config import ASR_STREAM_BATCH_SIZE, ASR_STREAM_BATCH_WAIT_MS
from protocol import AsrTask


class BatchScheduler:
    def __init__(self):
        self._queue: asyncio.Queue[AsrTask | None] = asyncio.Queue()
        self._closed = False

    async def submit(self, task: AsrTask) -> None:
        if self._closed:
            raise RuntimeError("BatchScheduler has been closed")
        await self._queue.put(task)

    async def next_batch(self) -> list[AsrTask]:
        first = await self._queue.get()
        if first is None:
            return []

        batch = [first]
        deadline = asyncio.get_running_loop().time() + (ASR_STREAM_BATCH_WAIT_MS / 1000)

        while len(batch) < ASR_STREAM_BATCH_SIZE:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if item is None:
                self._closed = True
                break
            batch.append(item)

        return batch

    async def close(self) -> None:
        self._closed = True
        await self._queue.put(None)
