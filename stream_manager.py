from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from batch_scheduler import BatchScheduler
from inference_worker import InferenceWorker
from protocol import AsrResultEvent
from vad_state import UserVadState

logger = logging.getLogger("ekko_asr_service.stream")


class StreamManager:
    def __init__(self):
        self._states: dict[tuple[int, str], UserVadState] = {}
        self._events: asyncio.Queue[AsrResultEvent] = asyncio.Queue()
        self._scheduler = BatchScheduler()
        self._worker = InferenceWorker(self._scheduler, self._handle_inference_result)
        self._started = False
        self._chunk_counts: dict[tuple[int, str], int] = {}

    async def start(self) -> None:
        if not self._started:
            await self._worker.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._worker.stop()
            self._started = False

    async def open_stream(self, payload: dict[str, Any]) -> None:
        state = UserVadState.from_payload(payload)
        key = (state.config.session_id, state.config.user_id)
        self._states[key] = state
        self._chunk_counts[key] = 0
        logger.info(
            "session_started session_id=%s user_id=%s sample_rate=%s channels=%s sample_width=%s language=%s",
            state.config.session_id,
            state.config.user_id,
            state.config.sample_rate,
            state.config.channels,
            state.config.sample_width,
            state.config.language,
        )
        await self._events.put(
            AsrResultEvent(
                type="session_started",
                session_id=state.config.session_id,
                user_id=state.config.user_id,
            )
        )

    async def push_audio_chunk(self, payload: dict[str, Any]) -> None:
        state = self._require_state(payload)
        key = (state.config.session_id, state.config.user_id)
        self._chunk_counts[key] = self._chunk_counts.get(key, 0) + 1
        chunk_count = self._chunk_counts[key]
        pcm_bytes = base64.b64decode(str(payload.get("audio_base64", "")))
        if chunk_count <= 3 or chunk_count % 50 == 0:
            logger.info(
                "audio_chunk session_id=%s user_id=%s chunk_count=%s bytes=%s",
                state.config.session_id,
                state.config.user_id,
                chunk_count,
                len(pcm_bytes),
            )
        immediate_events, tasks = state.process_audio_chunk(payload)
        for event in immediate_events:
            logger.info(
                "vad_event type=%s session_id=%s user_id=%s start_ms=%s end_ms=%s",
                event.type,
                event.session_id,
                event.user_id,
                event.start_ms,
                event.end_ms,
            )
            await self._events.put(event)
        for task in tasks:
            logger.info(
                "asr_task queued session_id=%s user_id=%s final=%s revision=%s start_ms=%s end_ms=%s bytes=%s",
                task.session_id,
                task.user_id,
                task.is_final,
                task.revision,
                task.start_ms,
                task.end_ms,
                len(task.pcm_bytes),
            )
            await self._scheduler.submit(task)

    async def close_stream(self, payload: dict[str, Any]) -> None:
        state = self._require_state(payload)
        key = (state.config.session_id, state.config.user_id)
        logger.info(
            "end_stream session_id=%s user_id=%s total_chunks=%s",
            state.config.session_id,
            state.config.user_id,
            self._chunk_counts.get(key, 0),
        )
        for task in state.end_stream():
            logger.info(
                "asr_task queued session_id=%s user_id=%s final=%s revision=%s start_ms=%s end_ms=%s bytes=%s",
                task.session_id,
                task.user_id,
                task.is_final,
                task.revision,
                task.start_ms,
                task.end_ms,
                len(task.pcm_bytes),
            )
            await self._scheduler.submit(task)

    async def next_event(self) -> AsrResultEvent:
        return await self._events.get()

    async def emit_stream_closed(self, session_id: int, user_id: str) -> None:
        self._chunk_counts.pop((session_id, user_id), None)
        await self._events.put(
            AsrResultEvent(
                type="stream_closed",
                session_id=session_id,
                user_id=user_id,
            )
        )

    async def _handle_inference_result(self, event: AsrResultEvent) -> None:
        key = (event.session_id, event.user_id)
        state = self._states.get(key)
        if state is None:
            return

        if event.type == "partial_result":
            if not state.should_emit_partial_text(event.text):
                logger.info(
                    "partial_result skipped session_id=%s user_id=%s revision=%s text=%r",
                    event.session_id,
                    event.user_id,
                    event.revision,
                    event.text,
                )
                return
        elif event.type == "final_result":
            event.seq_no = state.apply_final_text(event.text)

        logger.info(
            "asr_result type=%s session_id=%s user_id=%s revision=%s seq_no=%s text=%r",
            event.type,
            event.session_id,
            event.user_id,
            event.revision,
            event.seq_no,
            event.text,
        )
        await self._events.put(event)

    def _require_state(self, payload: dict[str, Any]) -> UserVadState:
        session_id = int(payload["session_id"])
        user_id = str(payload["user_id"])
        key = (session_id, user_id)
        state = self._states.get(key)
        if state is None:
            raise ValueError("Streaming session has not been initialized")
        return state
