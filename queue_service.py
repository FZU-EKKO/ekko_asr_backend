from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib import parse
from urllib import request
from urllib.error import HTTPError, URLError

from config import ASR_CALLBACK_TIMEOUT_SECONDS, ASR_QUEUE_MAX_SIZE, ASR_TEMP_AUDIO_DIR
from service import AsrService


logger = logging.getLogger("ekko_asr_service.queue")

TRANSCRIPTION_PROCESSING = "processing"
TRANSCRIPTION_DONE = "done"
TRANSCRIPTION_FAILED = "failed"
TRANSCRIPTION_DROPPED = "dropped"
UNRECOGNIZED_SPEECH_TEXT = "[unrecognized speech]"


@dataclass(slots=True)
class QueueJob:
    voice_message_id: int
    audio_path: str
    audio_format: str
    language: str
    callback_url: str
    callback_token: str


class AsrQueueService:
    def __init__(self) -> None:
        maxsize = max(0, ASR_QUEUE_MAX_SIZE)
        self._queue: asyncio.Queue[QueueJob | None] = asyncio.Queue(maxsize=maxsize)
        self._worker_task: asyncio.Task[None] | None = None
        self._enqueued_ids: set[int] = set()
        self._temp_root = Path(ASR_TEMP_AUDIO_DIR).resolve()

    async def start(self) -> None:
        self._temp_root.mkdir(parents=True, exist_ok=True)
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def shutdown(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None
        self._enqueued_ids.clear()

    async def enqueue(self, job: QueueJob) -> bool:
        if job.voice_message_id in self._enqueued_ids:
            return False
        self._enqueued_ids.add(job.voice_message_id)
        await self._queue.put(job)
        return True

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                break

            try:
                await self._process_job(job)
            except Exception:
                logger.exception("queue_job_failed id=%s", job.voice_message_id)
            finally:
                self._enqueued_ids.discard(job.voice_message_id)
                self._queue.task_done()

    async def _process_job(self, job: QueueJob) -> None:
        await asyncio.to_thread(
            _post_callback,
            callback_url=job.callback_url,
            callback_token=job.callback_token,
            payload={
                "voice_message_id": job.voice_message_id,
                "transcription_status": TRANSCRIPTION_PROCESSING,
                "transcript_text": None,
            },
        )

        try:
            result = await asyncio.to_thread(
                AsrService().transcribe_file,
                audio_path=job.audio_path,
                audio_format=job.audio_format,
                language=job.language,
            )
        except Exception as exc:
            logger.warning("queue_transcribe_failed id=%s detail=%s", job.voice_message_id, exc)
            await asyncio.to_thread(
                _post_callback,
                callback_url=job.callback_url,
                callback_token=job.callback_token,
                payload={
                    "voice_message_id": job.voice_message_id,
                    "transcription_status": TRANSCRIPTION_FAILED,
                    "transcript_text": None,
                },
            )
            return

        transcript_text = str(result.get("text", "") or "").strip()
        status = TRANSCRIPTION_DROPPED if transcript_text.casefold() == UNRECOGNIZED_SPEECH_TEXT.casefold() else TRANSCRIPTION_DONE
        await asyncio.to_thread(
            _post_callback,
            callback_url=job.callback_url,
            callback_token=job.callback_token,
            payload={
                "voice_message_id": job.voice_message_id,
                "transcription_status": status,
                "transcript_text": transcript_text or None,
            },
        )
        _delete_temp_audio_file(job.audio_path)
        logger.info(
            "queue_transcribe_done id=%s status=%s text_chars=%s path=%s",
            job.voice_message_id,
            status,
            len(transcript_text),
            job.audio_path,
        )

    async def save_audio_and_build_job(
        self,
        *,
        voice_message_id: int,
        audio_base64: str,
        audio_format: str,
        language: str,
        callback_url: str,
        callback_token: str,
    ) -> QueueJob:
        audio_path = await asyncio.to_thread(
            _save_temp_audio_file,
            temp_root=self._temp_root,
            voice_message_id=voice_message_id,
            audio_base64=audio_base64,
            audio_format=audio_format,
        )
        return QueueJob(
            voice_message_id=voice_message_id,
            audio_path=audio_path,
            audio_format=audio_format,
            language=language,
            callback_url=callback_url,
            callback_token=callback_token,
        )

    async def cleanup_job_file(self, audio_path: str) -> None:
        await asyncio.to_thread(_delete_temp_audio_file, audio_path)


def _post_callback(*, callback_url: str, callback_token: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {callback_token}",
    }
    req = request.Request(callback_url, data=body, headers=headers, method="POST")
    opener = request.build_opener(request.ProxyHandler({})) if _should_bypass_proxy(callback_url) else request.build_opener()

    try:
        with opener.open(req, timeout=ASR_CALLBACK_TIMEOUT_SECONDS) as resp:
            resp.read()
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        logger.error(
            "callback_http_error id=%s status=%s url=%s body=%s",
            payload.get("voice_message_id"),
            exc.code,
            callback_url,
            error_body,
        )
        raise RuntimeError(f"Callback HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        logger.error(
            "callback_connection_failed id=%s url=%s reason=%s",
            payload.get("voice_message_id"),
            callback_url,
            exc.reason,
        )
        raise RuntimeError(f"Callback connection failed: {exc.reason}") from exc


def _should_bypass_proxy(url: str) -> bool:
    hostname = (parse.urlparse(url).hostname or "").strip().lower()
    if not hostname:
        return False
    if hostname == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
        return address.is_loopback or address.is_private or address.is_link_local
    except ValueError:
        return False


def _save_temp_audio_file(
    *,
    temp_root: Path,
    voice_message_id: int,
    audio_base64: str,
    audio_format: str,
) -> str:
    normalized_format = (audio_format or "wav").strip().lower()
    if normalized_format != "wav":
        raise ValueError(f"Unsupported audio_format: {audio_format}")

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        raise ValueError("audio_base64 is not valid base64") from exc
    if not audio_bytes:
        raise ValueError("audio_base64 is empty")

    temp_root.mkdir(parents=True, exist_ok=True)
    filename = f"{voice_message_id}_{uuid.uuid4().hex}.{normalized_format}"
    audio_path = temp_root / filename
    audio_path.write_bytes(audio_bytes)
    logger.info(
        "saved_temp_audio_file id=%s path=%s bytes=%s",
        voice_message_id,
        audio_path,
        len(audio_bytes),
    )
    return str(audio_path)


def _delete_temp_audio_file(audio_path: str) -> None:
    try:
        os.unlink(audio_path)
        logger.info("deleted_temp_audio_file path=%s", audio_path)
    except FileNotFoundError:
        return
