from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from config import ASR_SERVICE_TOKEN
from queue_service import AsrQueueService
from schemas import QueueTranscribeRequest, QueueTranscribeResponse
from service import warmup_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ekko_asr_service")

app = FastAPI(title="ekko_asr_service")
queue_service = AsrQueueService()


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not ASR_SERVICE_TOKEN:
        return
    expected = f"Bearer {ASR_SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.on_event("startup")
async def startup_warmup() -> None:
    ready, error = warmup_model()
    if ready:
        logger.info("startup warmup success")
    else:
        logger.error("startup warmup failed detail=%s", error)
    await queue_service.start()


@app.on_event("shutdown")
async def shutdown_queue() -> None:
    await queue_service.shutdown()


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception("unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": repr(exc)},
    )

@app.post("/asr/transcribe/queue", response_model=QueueTranscribeResponse, dependencies=[Depends(verify_token)])
async def enqueue_transcribe(req: QueueTranscribeRequest) -> QueueTranscribeResponse:
    logger.info(
        "http queue request id=%s format=%s language=%s audio_base64_chars=%s",
        req.voice_message_id,
        req.audio_format,
        req.language,
        len(req.audio_base64 or ""),
    )
    job = await queue_service.save_audio_and_build_job(
        voice_message_id=req.voice_message_id,
        audio_base64=req.audio_base64,
        audio_format=req.audio_format,
        language=req.language,
        callback_url=req.callback_url,
        callback_token=req.callback_token,
    )
    queued = await queue_service.enqueue(job)
    if not queued:
        await queue_service.cleanup_job_file(job.audio_path)
    return QueueTranscribeResponse(queued=queued, voice_message_id=req.voice_message_id)
