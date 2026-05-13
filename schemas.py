from __future__ import annotations

from pydantic import BaseModel, Field


class QueueTranscribeRequest(BaseModel):
    voice_message_id: int = Field(..., gt=0)
    audio_base64: str = Field(..., min_length=1)
    audio_format: str = Field(default="wav")
    language: str = Field(default="zh")
    callback_url: str = Field(..., min_length=1)
    callback_token: str = Field(..., min_length=1)


class QueueTranscribeResponse(BaseModel):
    queued: bool
    voice_message_id: int
