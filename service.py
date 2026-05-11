from __future__ import annotations

import base64
import io
import logging
import os
import wave
from functools import lru_cache
from tempfile import mkstemp
from typing import Any

from config import (
    ASR_BEAM_SIZE,
    ASR_COMPUTE_TYPE,
    ASR_DEFAULT_LANGUAGE,
    ASR_DEVICE,
    ASR_MODEL_PATH,
    ASR_VAD_FILTER,
)
from faster_whisper import WhisperModel


logger = logging.getLogger("ekko_asr_service")

# Replace rare but phonetically similar misrecognitions here.
ASR_REPLACE_MAP: dict[str, str] = {
    "落毙": "RUSH B",
    "络币": "RUSH B",
    "络A": "RUSH A",
    "耶": "烟",
}

# Keep hotwords intentionally small and clean. Too many bias terms, or malformed
# terms, will pull decoding toward words that were never spoken.
ASR_HOTWORDS: tuple[str, ...] = (
    "AK",
    "M4",
    "AWP",
    "A大",
    "A小",
    "A点",
    "B点",
    "中门",
    "烟",
    "闪",
    "雷",
    "C4",
    "rush B",
)

# Use only a short, generic prompt. Long scripted prompts make the decoder guess.
ASR_INITIAL_PROMPT = "这是一段简体中文的对话(包含English)，内容与fps游戏相关。"


class AsrService:
    def transcribe(
        self,
        *,
        audio_base64: str,
        audio_format: str,
        language: str,
        prompt_text: str,
    ) -> dict[str, Any]:
        normalized_format = (audio_format or "wav").strip().lower()
        if normalized_format != "wav":
            raise ValueError(f"Unsupported audio_format: {audio_format}")

        audio_bytes = self._decode_wav_base64(audio_base64)
        logger.info(
            "transcribe request bytes=%s language=%s request_prompt_chars=%s configured_prompt_chars=%s hotwords=%s",
            len(audio_bytes),
            language or ASR_DEFAULT_LANGUAGE,
            len(prompt_text or ""),
            len(ASR_INITIAL_PROMPT),
            len(ASR_HOTWORDS),
        )
        return self._transcribe_wav_bytes(
            audio_bytes=audio_bytes,
            language=language or ASR_DEFAULT_LANGUAGE,
            prompt_text=prompt_text,
        )

    @staticmethod
    def _decode_wav_base64(audio_base64: str) -> bytes:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as exc:
            raise ValueError("audio_base64 is not valid base64") from exc

        if not audio_bytes:
            raise ValueError("audio_base64 is empty")

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
        except wave.Error as exc:
            raise ValueError("audio payload is not a valid wav file") from exc

        if frames <= 0:
            raise ValueError("wav contains no frames")

        logger.info(
            "wav meta channels=%s sample_width=%s sample_rate=%s frames=%s",
            channels,
            sample_width,
            sample_rate,
            frames,
        )
        return audio_bytes

    def _transcribe_wav_bytes(self, *, audio_bytes: bytes, language: str, prompt_text: str) -> dict[str, Any]:
        fd, temp_path = mkstemp(suffix=".wav")
        os.close(fd)

        try:
            with open(temp_path, "wb") as temp_file:
                temp_file.write(audio_bytes)

            logger.info("model transcribe start path=%s", temp_path)
            try:
                segments_iter, info = self._model.transcribe(
                    temp_path,
                    language=self._normalize_language(language),
                    beam_size=ASR_BEAM_SIZE,
                    vad_filter=ASR_VAD_FILTER,
                    word_timestamps=True,
                    initial_prompt=self._build_initial_prompt(),
                    # hotwords=self._build_hotwords(),
                )
                raw_result = self._normalize_result(list(segments_iter), info, language)
            except Exception as exc:
                if "maximum decoding length must be > 0" in str(exc):
                    logger.warning("model transcribe skipped empty_decoding_window path=%s detail=%s", temp_path, exc)
                    return self._empty_result(language)
                raise
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

        return raw_result

    @staticmethod
    def _empty_result(requested_language: str) -> dict[str, Any]:
        detected_language = requested_language or ASR_DEFAULT_LANGUAGE
        return {
            "text": "",
            "language": detected_language,
            "duration": 0.0,
            "segments": [],
            "words": [],
        }

    @property
    def _model(self):
        return get_whisper_model()

    @staticmethod
    def _normalize_language(language: str) -> str | None:
        normalized = (language or ASR_DEFAULT_LANGUAGE).strip().lower()
        if not normalized:
            return None
        mapping = {
            "zh": "zh",
            "zh-cn": "zh",
            "cn": "zh",
            "chinese": "zh",
            "en": "en",
            "english": "en",
            "ja": "ja",
            "jp": "ja",
            "japanese": "ja",
            "auto": "",
            "": "",
        }
        resolved = mapping.get(normalized, normalized)
        return resolved or None

    @staticmethod
    def _build_hotwords() -> str | None:
        hotwords = [word.strip() for word in ASR_HOTWORDS if word and word.strip()]
        if not hotwords:
            return None
        return ", ".join(dict.fromkeys(hotwords))

    @staticmethod
    def _build_initial_prompt() -> str | None:
        prompt = ASR_INITIAL_PROMPT.strip()
        return prompt or None

    @staticmethod
    def _normalize_result(raw_segments: list[Any], info: Any, requested_language: str) -> dict[str, Any]:
        segments = AsrService._build_segments(raw_segments)
        text = "".join(str(getattr(raw_segment, "text", "") or "") for raw_segment in raw_segments).strip()
        text = AsrService._apply_replace_map(text) or "[unrecognized speech]"
        words = [word for segment in segments for word in segment.get("words", [])]
        duration = round(max((float(segment.get("end", 0.0) or 0.0) for segment in segments), default=0.0), 3)
        detected_language = str(
            getattr(info, "language", "") or requested_language or ASR_DEFAULT_LANGUAGE
        )

        logger.info(
            "model transcribe done segments=%s text_chars=%s duration=%s language=%s",
            len(segments),
            len(text),
            duration,
            detected_language,
        )
        return {
            "text": text,
            "language": detected_language,
            "duration": duration,
            "segments": segments,
            "words": words,
        }

    @staticmethod
    def _build_segments(raw_segments: list[Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for index, raw_segment in enumerate(raw_segments):
            segment_words: list[dict[str, Any]] = []
            for raw_word in getattr(raw_segment, "words", None) or []:
                word_text = AsrService._apply_replace_map(str(getattr(raw_word, "word", "") or "").strip())
                if not word_text:
                    continue
                probability = getattr(raw_word, "probability", None)
                segment_words.append(
                    {
                        "start": round(float(getattr(raw_word, "start", 0.0) or 0.0), 3),
                        "end": round(float(getattr(raw_word, "end", 0.0) or 0.0), 3),
                        "word": word_text,
                        "probability": round(float(probability), 4) if isinstance(probability, (int, float)) else 0.0,
                    }
                )

            segments.append(
                {
                    "id": index,
                    "start": round(float(getattr(raw_segment, "start", 0.0) or 0.0), 3),
                    "end": round(float(getattr(raw_segment, "end", 0.0) or 0.0), 3),
                    "text": AsrService._apply_replace_map(str(getattr(raw_segment, "text", "") or "").strip()),
                    "words": segment_words,
                }
            )
        if segments:
            return segments
        return [
            {
                "id": 0,
                "start": 0.0,
                "end": 0.0,
                "text": "[unrecognized speech]",
                "words": [],
            }
        ]

    @staticmethod
    def _apply_replace_map(text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return normalized

        for source, target in ASR_REPLACE_MAP.items():
            source_text = str(source or "")
            if not source_text:
                continue
            normalized = normalized.replace(source_text, str(target or ""))
        return normalized


def get_runtime_status() -> dict[str, Any]:
    return {
        "model_path": ASR_MODEL_PATH,
        "device": ASR_DEVICE,
        "compute_type": ASR_COMPUTE_TYPE,
        "default_language": ASR_DEFAULT_LANGUAGE,
        "beam_size": ASR_BEAM_SIZE,
        "vad_filter": ASR_VAD_FILTER,
        "replace_map_entries": len(ASR_REPLACE_MAP),
    }


def warmup_model() -> tuple[bool, str | None]:
    try:
        _ = get_whisper_model()
        return True, None
    except Exception as exc:
        logger.exception("warmup model failed")
        return False, repr(exc)


@lru_cache(maxsize=1)
def get_whisper_model():
    logger.info(
        "load faster_whisper model model=%s device=%s compute_type=%s",
        ASR_MODEL_PATH,
        ASR_DEVICE,
        ASR_COMPUTE_TYPE,
    )
    return WhisperModel(
        ASR_MODEL_PATH,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
    )  # type: ignore[misc]
