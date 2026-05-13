from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def get_int_env(name: str, default: int) -> int:
    raw = get_env(name, "")
    if not raw:
        return default
    return int(raw)


def get_bool_env(name: str, default: bool) -> bool:
    raw = get_env(name, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


ASR_SERVICE_TOKEN = get_env("EKKO_ASR_SERVICE_TOKEN", "")
ASR_MODEL_PATH = get_env("EKKO_ASR_MODEL_PATH", "large-v3")
ASR_DEVICE = get_env("EKKO_ASR_DEVICE", "auto")
ASR_COMPUTE_TYPE = get_env("EKKO_ASR_COMPUTE_TYPE", "auto")
ASR_DEFAULT_LANGUAGE = get_env("EKKO_ASR_DEFAULT_LANGUAGE", "zh")
ASR_BEAM_SIZE = get_int_env("EKKO_ASR_BEAM_SIZE", 5)
ASR_VAD_FILTER = get_bool_env("EKKO_ASR_VAD_FILTER", True)
ASR_QUEUE_MAX_SIZE = get_int_env("EKKO_ASR_QUEUE_MAX_SIZE", 0)
ASR_CALLBACK_TIMEOUT_SECONDS = get_int_env("EKKO_ASR_CALLBACK_TIMEOUT_SECONDS", 30)
ASR_TEMP_AUDIO_DIR = get_env("EKKO_ASR_TEMP_AUDIO_DIR", "./tmp_audio")
