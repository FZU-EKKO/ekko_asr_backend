from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

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
app.state.transcribe_queue = None
app.state.transcribe_worker = None
app.state.queue_processing = False


@dataclass
class QueueItem:
    req: TranscribeRequest
    future: asyncio.Future[dict]


async def transcribe_worker_loop() -> None:
    queue: asyncio.Queue[QueueItem | None] = app.state.transcribe_queue
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        app.state.queue_processing = True
        try:
            result = await asyncio.to_thread(
                AsrService().transcribe,
                audio_base64=item.req.audio_base64,
                audio_format=item.req.audio_format,
                language=item.req.language,
                prompt_text=item.req.prompt_text,
            )
        except Exception as exc:
            if not item.future.done():
                item.future.set_exception(exc)
        else:
            if not item.future.done():
                item.future.set_result(result)
        finally:
            app.state.queue_processing = False
            queue.task_done()


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not ASR_SERVICE_TOKEN:
        return
    expected = f"Bearer {ASR_SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.on_event("startup")
async def startup_warmup() -> None:
    ready, error = warmup_model()
    app.state.model_loaded = ready
    app.state.model_load_error = error
    app.state.transcribe_queue = asyncio.Queue()
    app.state.transcribe_worker = asyncio.create_task(transcribe_worker_loop())
    if ready:
        logger.info("startup warmup success")
    else:
        logger.error("startup warmup failed detail=%s", error)


@app.on_event("shutdown")
async def shutdown_worker() -> None:
    queue: asyncio.Queue[QueueItem | None] | None = app.state.transcribe_queue
    worker: asyncio.Task | None = app.state.transcribe_worker
    if queue is not None:
        await queue.put(None)
    if worker is not None:
        await worker


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
        queue_size=0 if app.state.transcribe_queue is None else app.state.transcribe_queue.qsize(),
        queue_processing=bool(app.state.queue_processing),
    )


@app.post("/asr/transcribe", response_model=TranscribeResponse, dependencies=[Depends(verify_token)])
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    logger.info(
        "http transcribe request format=%s language=%s audio_base64_chars=%s",
        req.audio_format,
        req.language,
        len(req.audio_base64 or ""),
    )
    app.state.last_transcribe_error = None
    queue: asyncio.Queue[QueueItem | None] | None = app.state.transcribe_queue
    if queue is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ASR queue is not ready")

    try:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        await queue.put(QueueItem(req=req, future=future))
        result = await future
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
