from __future__ import annotations

import base64
import io
import os
import wave
from functools import lru_cache
from tempfile import mkstemp
from typing import Any

from faster_whisper import WhisperModel

from config import (
    ASR_BEAM_SIZE,
    ASR_COMPUTE_TYPE,
    ASR_DEFAULT_LANGUAGE,
    ASR_DEVICE,
    ASR_MODEL_SIZE,
    ASR_VAD_FILTER,
)


class AsrService:
    def transcribe(
        self,
        *,
        audio_base64: str,
        audio_format: str,
        language: str,
        prompt_text: str,
    ) -> dict[str, Any]:
        audio_bytes = self._decode_audio(audio_base64)
        normalized_format = (audio_format or "wav").strip().lower()
        if normalized_format != "wav":
            raise ValueError(f"Unsupported audio_format: {audio_format}")
        return self._transcribe_wav_bytes(audio_bytes=audio_bytes, language=language, prompt_text=prompt_text)

    def transcribe_pcm_bytes(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        sample_width: int,
        language: str,
        prompt_text: str,
    ) -> dict[str, Any]:
        wav_bytes = self._build_wav_bytes(
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )
        return self._transcribe_wav_bytes(audio_bytes=wav_bytes, language=language, prompt_text=prompt_text)

    @property
    def _model(self) -> WhisperModel:
        return get_whisper_model()

    @staticmethod
    def _decode_audio(audio_base64: str) -> bytes:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as exc:
            raise ValueError("audio_base64 is not valid base64") from exc

        if not audio_bytes:
            raise ValueError("audio_base64 is empty")

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                if wav_file.getnframes() <= 0:
                    raise ValueError("wav contains no frames")
        except wave.Error as exc:
            raise ValueError("audio payload is not a valid wav file") from exc

        return audio_bytes

    def _transcribe_wav_bytes(self, *, audio_bytes: bytes, language: str, prompt_text: str) -> dict[str, Any]:
        fd, temp_path = mkstemp(suffix=".wav")
        os.close(fd)

        try:
            with open(temp_path, "wb") as temp_file:
                temp_file.write(audio_bytes)

            segments, info = self._model.transcribe(
                temp_path,
                language=(language or ASR_DEFAULT_LANGUAGE).strip() or None,
                initial_prompt=prompt_text.strip() or None,
                beam_size=ASR_BEAM_SIZE,
                vad_filter=ASR_VAD_FILTER,
                word_timestamps=True,
            )
            segment_list = [
                {
                    "id": item.id,
                    "start": round(item.start, 3),
                    "end": round(item.end, 3),
                    "text": item.text.strip(),
                    "words": [
                        {
                            "start": round(float(word.start), 3),
                            "end": round(float(word.end), 3),
                            "word": (word.word or "").strip(),
                            "probability": round(float(word.probability), 4),
                        }
                        for word in (item.words or [])
                        if (word.word or "").strip()
                    ],
                }
                for item in segments
                if item.text.strip()
            ]
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
        word_list = self._extract_words(segment_list)
        text = " ".join(item["text"] for item in segment_list).strip()
        if not text:
            text = "[unrecognized speech]"

        return {
            "text": text,
            "language": info.language or language or ASR_DEFAULT_LANGUAGE,
            "duration": round(info.duration, 3),
            "segments": segment_list,
            "words": word_list,
        }

    @staticmethod
    def _build_wav_bytes(*, pcm_bytes: bytes, sample_rate: int, channels: int, sample_width: int) -> bytes:
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return stream.getvalue()

    @staticmethod
    def _extract_words(segment_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        words: list[dict[str, Any]] = []
        for segment in segment_list:
            raw_words = segment.get("words") or []
            for item in raw_words:
                token = str(item.get("word", "") or "").strip()
                if not token:
                    continue
                words.append(
                    {
                        "start": round(float(item.get("start", 0.0)), 3),
                        "end": round(float(item.get("end", 0.0)), 3),
                        "word": token,
                        "probability": round(float(item.get("probability", 0.0)), 4),
                    }
                )
        return words

@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel(
        ASR_MODEL_SIZE,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
    )
