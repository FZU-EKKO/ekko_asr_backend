from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_bool_env(name: str, default: bool) -> bool:
    raw = _get_env(name, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    raw = _get_env(name, "")
    if not raw:
        return default
    return int(raw)


ASR_SERVICE_TOKEN = _get_env("EKKO_ASR_SERVICE_TOKEN", "")
ASR_MODEL_SIZE = _get_env("EKKO_ASR_MODEL_SIZE", "small")
ASR_DEVICE = _get_env("EKKO_ASR_DEVICE", "auto")
ASR_COMPUTE_TYPE = _get_env("EKKO_ASR_COMPUTE_TYPE", "auto")
ASR_DEFAULT_LANGUAGE = _get_env("EKKO_ASR_DEFAULT_LANGUAGE", "zh")
ASR_BEAM_SIZE = _get_int_env("EKKO_ASR_BEAM_SIZE", 5)
ASR_VAD_FILTER = _get_bool_env("EKKO_ASR_VAD_FILTER", True)
ASR_STREAM_ENERGY_THRESHOLD = _get_int_env("EKKO_ASR_STREAM_ENERGY_THRESHOLD", 450)
ASR_STREAM_SILENCE_MS = _get_int_env("EKKO_ASR_STREAM_SILENCE_MS", 700)
ASR_STREAM_MIN_UTTERANCE_MS = _get_int_env("EKKO_ASR_STREAM_MIN_UTTERANCE_MS", 500)
ASR_STREAM_MAX_UTTERANCE_MS = _get_int_env("EKKO_ASR_STREAM_MAX_UTTERANCE_MS", 6000)
ASR_STREAM_PARTIAL_INTERVAL_MS = _get_int_env("EKKO_ASR_STREAM_PARTIAL_INTERVAL_MS", 800)
ASR_STREAM_BATCH_SIZE = _get_int_env("EKKO_ASR_STREAM_BATCH_SIZE", 4)
ASR_STREAM_BATCH_WAIT_MS = _get_int_env("EKKO_ASR_STREAM_BATCH_WAIT_MS", 120)
