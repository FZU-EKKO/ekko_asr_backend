from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StreamConfig:
    session_id: int
    user_id: str
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    language: str = "zh"
    prompt_text: str = ""


@dataclass(slots=True)
class AsrTask:
    session_id: int
    user_id: str
    start_ms: int
    end_ms: int
    pcm_bytes: bytes
    sample_rate: int
    channels: int
    sample_width: int
    language: str
    prompt_text: str
    is_final: bool
    revision: int = 0


@dataclass(slots=True)
class AsrResultEvent:
    type: str
    session_id: int
    user_id: str
    text: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    start_ms: int = 0
    end_ms: int = 0
    duration: float = 0.0
    is_final: bool = False
    revision: int = 0
    seq_no: int = 0
    detail: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "session_id": self.session_id,
            "user_id": self.user_id,
        }
        if self.text:
            payload["text"] = self.text
        if self.words:
            payload["words"] = self.words
        if self.start_ms or self.end_ms:
            payload["start_ms"] = self.start_ms
            payload["end_ms"] = self.end_ms
        if self.duration:
            payload["duration"] = self.duration
        if self.is_final:
            payload["is_final"] = True
        if self.revision:
            payload["revision"] = self.revision
        if self.seq_no:
            payload["seq_no"] = self.seq_no
        if self.detail:
            payload["detail"] = self.detail
        return payload
