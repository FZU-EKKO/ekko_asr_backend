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

# Hard-coded replacements applied after ASR decoding.
# Edit this map directly when you need to normalize domain-specific terms.
ASR_REPLACE_MAP: dict[str, str] = {
    # === 字母 / 点位 ===
    "阿大": "A大",
    "阿小": "A小",
    "a 大": "A大",
    "a 小": "A小",
    "b 包": "B包",
    "b 洞": "B洞",
    "b 小": "B小",
    "中门对局": "中门对狙",
    "匪家": "匪家",             # 可能不需要，但保留位置
    "警家": "警家",
    # === 武器 ===
    "阿卡": "AK",
    "诶剋": "AK",
    "m 四": "M4",
    "诶 w p": "AWP",
    "大橘": "大狙",
    "鸟居": "鸟狙",
    "沙鹰": "沙鹰",
    "喷子": "喷子",
    "手枪局": "手枪局",
    "长枪局": "长枪局",
    # === 道具 ===
    "烟雾但": "烟雾弹",
    "烟幕弹": "烟雾弹",
    "闪关弹": "闪光弹",
    "闪光但": "闪光弹",
    "燃烧但": "燃烧弹",
    "手雷": "手雷",             # 通常不易错，可保留
    "钳子": "钳子",
    "c 四": "C4",
    "西四": "C4",
    # === 状态 ===
    "残血": "残血",
    "一枪使": "一枪死",
    "一枪屎": "一枪死",
    "打腿": "打腿",
    "包头": "爆头",
    "爆头": "爆头",             # 可能识别成“抱头”
    "背闪": "被闪",
    # === 战术 / 行动 ===
    "拉屎逼": "rush B",
    "如阿逼": "rush B",
    "如阿 b": "rush B",
    "拉枪": "拉枪线",          # 可能漏字
    "下巴": "下包",
    "拆爆": "拆包",
    "保枪": "保枪",
    "回访": "回防",
    "转点": "转点",
    "断后": "断后",
    "换蛋": "换弹",
    # === 沟通 ===
    "发把枪": "发把枪",
    "我没钱": "我没钱",
    "一扣局": "ECO局",
    "一 c o 局": "ECO局",
    "两个": "两个",             # 有时识别成“俩”
    "三个": "三个",
    "脚步": "脚步",
    "最后一把": "最后一把",
    "救救救": "救救救",
}

# Hard-coded bias terms for faster-whisper decoding.
# Edit this list directly when you need to tune recognition for domain terms.
# 位置 / 地图点位
hotwords_location = [
    "A大", "A小", "A门", "A包点",
    "B洞", "B一层", "B二楼", "B小",
    "中门", "中远", "匪家", "警家",
    "阁楼", "长廊", "VIP", "车道",
    "跳台", "黄房", "绿通", "蓝箱"
]

# 武器
hotwords_weapons = [
    "AK", "火麒麟", "M4", "雷神",
    "AWP", "大狙", "鸟狙", "沙鹰",
    "冲锋枪", "喷子", "手枪", "刀",
    "长枪", "手枪局", "长枪局"
]

# 道具
hotwords_equipment = [
    "烟雾弹", "闪光弹", "高闪", "燃烧弹",
    "手雷", "诱饵弹", "钳子", "C4", "包"
]

# 状态 / 血量
hotwords_status = [
    "满血", "残血", "一枪死", "大残",
    "打腿", "爆头", "被闪", "耳鸣"
]

# 行动 / 战术
hotwords_tactics = [
    "rush B", "慢摸", "架枪", "拉枪线",
    "反清", "前压", "回防", "转点",
    "保枪", "下包", "安包", "拆包",
    "假拆", "断后", "掩护", "补枪",
    "换弹", "压脚步", "干拉"
]

# 报点 / 沟通
hotwords_comms = [
    "有人", "两个", "三个", "一个",
    "人多", "人少", "脚步", "拆包声音",
    "最后一把", "我没钱", "发把枪",
    "ECO局", "强起", "对面ECO",
    "打他", "中门对狙", "救救救"
]

ASR_HOTWORDS: tuple[str, ...] = (
    hotwords_location +
    hotwords_weapons +
    hotwords_equipment +
    hotwords_status +
    hotwords_tactics +
    hotwords_comms
)

# Hard-coded initial prompt for faster-whisper decoding.
# Edit this text directly when you need fixed contextual guidance.
ASR_INITIAL_PROMPT = """
以下是玩家在游戏中的语音聊天内容：
敌人来了，A大两个，小心狙。中路一个，B洞有脚步。快发把狙，给我 AK。下包下包，拆包掩护我。Rush B，不要怂。残血一枪死，A小烟封上。有老六，断后断后。闪光弹躲一下，爆头秒了。回防回防，保枪吧。没子弹换弹，队友补枪。
"""

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
            segments_iter, info = self._model.transcribe(
                temp_path,
                language=self._normalize_language(language),
                beam_size=ASR_BEAM_SIZE,
                vad_filter=ASR_VAD_FILTER,
                word_timestamps=True,
                initial_prompt=self._build_initial_prompt(),
                hotwords=self._build_hotwords(),
            )
            raw_result = self._normalize_result(list(segments_iter), info, language)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

        return raw_result

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
