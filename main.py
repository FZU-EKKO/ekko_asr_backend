from __future__ import annotations

import asyncio
import json
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi import WebSocket, WebSocketDisconnect

from config import ASR_SERVICE_TOKEN
from schemas import TranscribeRequest, TranscribeResponse
from service import AsrService
from stream_manager import StreamManager

logger = logging.getLogger("ekko_asr_service.ws")

app = FastAPI(title="ekko_asr_service")


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not ASR_SERVICE_TOKEN:
        return
    expected = f"Bearer {ASR_SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def verify_ws_token(websocket: WebSocket) -> None:
    if not ASR_SERVICE_TOKEN:
        return
    authorization = websocket.headers.get("authorization")
    expected = f"Bearer {ASR_SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/asr/transcribe", response_model=TranscribeResponse, dependencies=[Depends(verify_token)])
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    try:
        result = AsrService().transcribe(
            audio_base64=req.audio_base64,
            audio_format=req.audio_format,
            language=req.language,
            prompt_text=req.prompt_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return TranscribeResponse(**result)


@app.websocket("/asr/stream")
async def stream_transcribe(websocket: WebSocket) -> None:
    try:
        verify_ws_token(websocket)
    except HTTPException:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    manager = StreamManager()
    await manager.start()
    active_session_id: int | None = None
    active_user_id: str | None = None
    logger.info("websocket accepted client=%s", websocket.client)

    async def send_events() -> None:
        while True:
            event = await manager.next_event()
            logger.info(
                "send_event type=%s session_id=%s user_id=%s revision=%s seq_no=%s text=%r detail=%r",
                event.type,
                event.session_id,
                event.user_id,
                event.revision,
                event.seq_no,
                event.text,
                event.detail,
            )
            await websocket.send_json(event.to_payload())
            if event.type == "stream_closed":
                return

    sender_task: asyncio.Task | None = None
    try:
        sender_task = asyncio.create_task(send_events())
        while True:
            raw_message = await websocket.receive_text()
            payload = json.loads(raw_message)
            message_type = str(payload.get("type", "")).strip()

            if message_type == "start_session":
                active_session_id = int(payload["session_id"])
                active_user_id = str(payload["user_id"])
                logger.info(
                    "recv start_session session_id=%s user_id=%s sample_rate=%s channels=%s sample_width=%s language=%s",
                    active_session_id,
                    active_user_id,
                    payload.get("sample_rate"),
                    payload.get("channels"),
                    payload.get("sample_width"),
                    payload.get("language"),
                )
                await manager.open_stream(payload)
                continue

            if message_type == "audio_chunk":
                if active_session_id is None or active_user_id is None:
                    raise ValueError("start_session must be sent before audio_chunk")
                payload["session_id"] = active_session_id
                payload["user_id"] = active_user_id
                await manager.push_audio_chunk(payload)
                continue

            if message_type == "end_stream":
                if active_session_id is None or active_user_id is None:
                    raise ValueError("start_session must be sent before end_stream")
                logger.info("recv end_stream session_id=%s user_id=%s", active_session_id, active_user_id)
                payload["session_id"] = active_session_id
                payload["user_id"] = active_user_id
                await manager.close_stream(payload)
                await manager.stop()
                await manager.emit_stream_closed(active_session_id, active_user_id)
                if sender_task is not None:
                    await sender_task
                await websocket.close()
                return

            await websocket.send_json({"type": "error", "detail": f"Unsupported message type: {message_type}"})
    except WebSocketDisconnect:
        logger.info("websocket disconnected session_id=%s user_id=%s", active_session_id, active_user_id)
        await manager.stop()
        return
    except Exception as exc:
        logger.exception("websocket stream error session_id=%s user_id=%s", active_session_id, active_user_id)
        await manager.stop()
        if websocket.client_state.name.lower() != "disconnected":
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1011)
    finally:
        if sender_task is not None and not sender_task.done():
            sender_task.cancel()
