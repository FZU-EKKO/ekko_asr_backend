from __future__ import annotations

from pydantic import BaseModel, Field,ConfigDict


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


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    ready: bool
    model_path: str
    device: str
    compute_type: str
    default_language: str
    beam_size: int
    vad_filter: bool
    replace_map_entries: int = 0
    model_loaded: bool = False
    model_load_error: str | None = None
    last_transcribe_error: str | None = None
    queue_size: int = 0
    queue_processing: bool = False
