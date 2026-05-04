from __future__ import annotations

import audioop
import base64
from dataclasses import dataclass, field
from typing import Any

from config import (
    ASR_DEFAULT_LANGUAGE,
    ASR_STREAM_ENERGY_THRESHOLD,
    ASR_STREAM_MAX_UTTERANCE_MS,
    ASR_STREAM_MIN_UTTERANCE_MS,
    ASR_STREAM_PARTIAL_INTERVAL_MS,
    ASR_STREAM_SILENCE_MS,
)
from protocol import AsrResultEvent, AsrTask, StreamConfig


@dataclass(slots=True)
class UserVadState:
    config: StreamConfig
    stream_offset_ms: int = 0
    utterance_start_ms: int | None = None
    speech_buffer: bytearray = field(default_factory=bytearray)
    speech_duration_ms: int = 0
    trailing_silence_ms: int = 0
    partial_revision: int = 0
    last_partial_emit_ms: int = 0
    last_partial_text: str = ""
    prompt_text: str = ""
    seq_no: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "UserVadState":
        config = StreamConfig(
            session_id=int(payload["session_id"]),
            user_id=str(payload["user_id"]),
            sample_rate=int(payload.get("sample_rate", 16000)),
            channels=int(payload.get("channels", 1)),
            sample_width=int(payload.get("sample_width", 2)),
            language=str(payload.get("language") or ASR_DEFAULT_LANGUAGE),
            prompt_text=str(payload.get("prompt_text") or ""),
        )
        state = cls(config=config)
        state.prompt_text = config.prompt_text
        return state

    def process_audio_chunk(self, payload: dict[str, Any]) -> tuple[list[AsrResultEvent], list[AsrTask]]:
        pcm_bytes = base64.b64decode(str(payload.get("audio_base64", "")))
        if not pcm_bytes:
            return [], []

        packet_duration_ms = self.compute_duration_ms(
            pcm_bytes=pcm_bytes,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            sample_width=self.config.sample_width,
        )
        if packet_duration_ms <= 0:
            return [], []

        events: list[AsrResultEvent] = []
        tasks: list[AsrTask] = []
        is_speech = self.is_speech(pcm_bytes=pcm_bytes, sample_width=self.config.sample_width)

        if is_speech and self.utterance_start_ms is None:
            self.utterance_start_ms = self.stream_offset_ms
            self.speech_buffer.clear()
            self.speech_duration_ms = 0
            self.trailing_silence_ms = 0
            self.partial_revision = 0
            self.last_partial_emit_ms = 0
            self.last_partial_text = ""
            events.append(
                AsrResultEvent(
                    type="speech_start",
                    session_id=self.config.session_id,
                    user_id=self.config.user_id,
                    start_ms=self.utterance_start_ms,
                )
            )

        if self.utterance_start_ms is not None:
            self.speech_buffer.extend(pcm_bytes)
            self.speech_duration_ms += packet_duration_ms
            if is_speech:
                self.trailing_silence_ms = 0
            else:
                self.trailing_silence_ms += packet_duration_ms

            should_emit_partial = (
                self.speech_duration_ms >= ASR_STREAM_MIN_UTTERANCE_MS
                and self.speech_duration_ms - self.last_partial_emit_ms >= ASR_STREAM_PARTIAL_INTERVAL_MS
            )
            if should_emit_partial:
                self.partial_revision += 1
                self.last_partial_emit_ms = self.speech_duration_ms
                tasks.append(self._build_task(is_final=False, revision=self.partial_revision))

            should_finalize = (
                self.trailing_silence_ms >= ASR_STREAM_SILENCE_MS
                or self.speech_duration_ms >= ASR_STREAM_MAX_UTTERANCE_MS
            )
            if should_finalize:
                final_task = self._flush_current_utterance(force=False)
                if final_task:
                    tasks.append(final_task)

        self.stream_offset_ms += packet_duration_ms
        return events, tasks

    def end_stream(self) -> list[AsrTask]:
        final_task = self._flush_current_utterance(force=True)
        return [final_task] if final_task else []

    def apply_final_text(self, text: str) -> int:
        self.seq_no += 1
        self.prompt_text = f"{self.prompt_text} {text}".strip()
        self.last_partial_text = ""
        return self.seq_no

    def should_emit_partial_text(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized or normalized == self.last_partial_text:
            return False
        self.last_partial_text = normalized
        return True

    def _flush_current_utterance(self, *, force: bool) -> AsrTask | None:
        if self.utterance_start_ms is None:
            return None
        if not force and self.speech_duration_ms < ASR_STREAM_MIN_UTTERANCE_MS:
            if self.trailing_silence_ms >= ASR_STREAM_SILENCE_MS:
                self._reset_utterance()
            return None
        task = self._build_task(is_final=True, revision=0)
        self._reset_utterance()
        return task

    def _build_task(self, *, is_final: bool, revision: int) -> AsrTask:
        assert self.utterance_start_ms is not None
        return AsrTask(
            session_id=self.config.session_id,
            user_id=self.config.user_id,
            start_ms=self.utterance_start_ms,
            end_ms=self.utterance_start_ms + self.speech_duration_ms,
            pcm_bytes=bytes(self.speech_buffer),
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            sample_width=self.config.sample_width,
            language=self.config.language,
            prompt_text=self.prompt_text,
            is_final=is_final,
            revision=revision,
        )

    def _reset_utterance(self) -> None:
        self.utterance_start_ms = None
        self.speech_buffer = bytearray()
        self.speech_duration_ms = 0
        self.trailing_silence_ms = 0
        self.last_partial_emit_ms = 0
        self.last_partial_text = ""

    @staticmethod
    def is_speech(*, pcm_bytes: bytes, sample_width: int) -> bool:
        return audioop.rms(pcm_bytes, sample_width) >= ASR_STREAM_ENERGY_THRESHOLD

    @staticmethod
    def compute_duration_ms(*, pcm_bytes: bytes, sample_rate: int, channels: int, sample_width: int) -> int:
        frame_width = channels * sample_width
        if frame_width <= 0 or sample_rate <= 0:
            return 0
        frames = len(pcm_bytes) / frame_width
        return int(frames / sample_rate * 1000)
