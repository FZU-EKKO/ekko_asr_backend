from __future__ import annotations

from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    audio_format: str = Field(default="wav")
    language: str = Field(default="zh")
    prompt_text: str = Field(default="")
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    sample_width: int = Field(default=2, gt=0)


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[dict]
    words: list[dict] = Field(default_factory=list)
