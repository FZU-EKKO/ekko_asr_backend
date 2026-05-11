from __future__ import annotations

import asyncio
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from config import ASR_SERVICE_TOKEN
from schemas import TranscribeRequest, TranscribeResponse
from service import AsrService, warmup_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ekko_asr_service")

app = FastAPI(title="ekko_asr_service")


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


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception("unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": repr(exc)},
    )


@app.post("/asr/transcribe", response_model=TranscribeResponse, dependencies=[Depends(verify_token)])
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    logger.info(
        "http transcribe request format=%s language=%s audio_base64_chars=%s",
        req.audio_format,
        req.language,
        len(req.audio_base64 or ""),
    )
    try:
        result = await asyncio.to_thread(
            AsrService().transcribe,
            audio_base64=req.audio_base64,
            audio_format=req.audio_format,
            language=req.language,
            prompt_text=req.prompt_text,
        )
    except ValueError as exc:
        logger.warning("http transcribe bad_request detail=%s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("http transcribe failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=repr(exc)) from exc

    logger.info(
        "http transcribe success text_chars=%s duration=%s language=%s",
        len(result.get("text", "") or ""),
        result.get("duration"),
        result.get("language"),
    )
    return TranscribeResponse(**result)
