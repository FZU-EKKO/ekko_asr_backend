from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from config import ASR_SERVICE_TOKEN
from schemas import HealthResponse, TranscribeRequest, TranscribeResponse
from service import AsrService, get_runtime_status, warmup_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ekko_asr_service")

app = FastAPI(title="ekko_asr_service")
app.state.model_loaded = False
app.state.model_load_error = None
app.state.last_transcribe_error = None


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not ASR_SERVICE_TOKEN:
        return
    expected = f"Bearer {ASR_SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.on_event("startup")
def startup_warmup() -> None:
    ready, error = warmup_model()
    app.state.model_loaded = ready
    app.state.model_load_error = error
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


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    runtime = get_runtime_status()
    ready = bool(app.state.model_loaded and not app.state.model_load_error)
    return HealthResponse(
        status="ok" if ready else "degraded",
        ready=ready,
        model_path=runtime["model_path"],
        device=runtime["device"],
        compute_type=runtime["compute_type"],
        default_language=runtime["default_language"],
        beam_size=runtime["beam_size"],
        vad_filter=runtime["vad_filter"],
        model_loaded=bool(app.state.model_loaded),
        model_load_error=app.state.model_load_error,
        last_transcribe_error=app.state.last_transcribe_error,
    )


@app.post("/asr/transcribe", response_model=TranscribeResponse, dependencies=[Depends(verify_token)])
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    logger.info(
        "http transcribe request format=%s language=%s audio_base64_chars=%s",
        req.audio_format,
        req.language,
        len(req.audio_base64 or ""),
    )
    app.state.last_transcribe_error = None

    try:
        result = AsrService().transcribe(
            audio_base64=req.audio_base64,
            audio_format=req.audio_format,
            language=req.language,
            prompt_text=req.prompt_text,
        )
    except ValueError as exc:
        app.state.last_transcribe_error = str(exc)
        logger.warning("http transcribe bad_request detail=%s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        app.state.last_transcribe_error = repr(exc)
        logger.exception("http transcribe failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=repr(exc)) from exc

    logger.info(
        "http transcribe success text_chars=%s duration=%s language=%s",
        len(result.get("text", "") or ""),
        result.get("duration"),
        result.get("language"),
    )
    return TranscribeResponse(**result)
