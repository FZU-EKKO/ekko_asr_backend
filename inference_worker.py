from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from batch_scheduler import BatchScheduler
from protocol import AsrResultEvent, AsrTask
from service import AsrService

logger = logging.getLogger("ekko_asr_service.inference")

ResultCallback = Callable[[AsrResultEvent], Awaitable[None]]


class InferenceWorker:
    def __init__(self, scheduler: BatchScheduler, result_callback: ResultCallback):
        self._scheduler = scheduler
        self._result_callback = result_callback
        self._service = AsrService()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        await self._scheduler.close()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            batch = await self._scheduler.next_batch()
            if not batch:
                return

            results = await asyncio.gather(*(self._transcribe_task(task) for task in batch), return_exceptions=True)
            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    await self._result_callback(
                        AsrResultEvent(
                            type="error",
                            session_id=task.session_id,
                            user_id=task.user_id,
                            detail=str(result),
                        )
                    )
                    continue
                if result is not None:
                    await self._result_callback(result)

    async def _transcribe_task(self, task: AsrTask) -> AsrResultEvent | None:
        logger.info(
            "transcribe start session_id=%s user_id=%s final=%s revision=%s start_ms=%s end_ms=%s bytes=%s",
            task.session_id,
            task.user_id,
            task.is_final,
            task.revision,
            task.start_ms,
            task.end_ms,
            len(task.pcm_bytes),
        )
        result = await asyncio.to_thread(
            self._service.transcribe_pcm_bytes,
            pcm_bytes=task.pcm_bytes,
            sample_rate=task.sample_rate,
            channels=task.channels,
            sample_width=task.sample_width,
            language=task.language,
            prompt_text=task.prompt_text,
        )
        text = str(result.get("text", "")).strip()
        logger.info(
            "transcribe done session_id=%s user_id=%s final=%s revision=%s text=%r",
            task.session_id,
            task.user_id,
            task.is_final,
            task.revision,
            text,
        )
        if not text or text == "[unrecognized speech]":
            return None
        return AsrResultEvent(
            type="final_result" if task.is_final else "partial_result",
            session_id=task.session_id,
            user_id=task.user_id,
            text=text,
            words=result.get("words", []),
            start_ms=task.start_ms,
            end_ms=task.end_ms,
            duration=float(result.get("duration", 0.0)),
            is_final=task.is_final,
            revision=task.revision,
        )
